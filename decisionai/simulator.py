from collections import defaultdict, Counter
import numpy as np
from typing import Iterable, Dict, List, Union, Set, Tuple, Optional, Any

from .datasets import (
    Dataset,
    DatasetAdditionVar,
    DatasetVariableValueHolder,
)
from .external_models import parse_model_defn, ExternalModelDefinition, Model
from .policies import PolicyAttribute, PolicyDefinition, policies_to_attributes, get_num_policies
from .variables import BaseVariable, Variable
from .topo_sort import topo_sort_removing_errors
from .errors import Error, VisibleError, SilentError
from .formulas import DependencyCollector, TreeTransformer, TreeEvaluator
from .formulas.dep_collector import LagCollector
from . import constants


class Simulation:
    num_steps: int
    num_sims: int
    # NB: this is set to 1 if no explicit policy labels or attributes are defined
    num_policies: int

    # TODO: replacing these lists with dicts (mapping from var name) might simplify 
    # some things
    attributes: List[PolicyAttribute]
    variables: List[Variable]
    datasets: Dict[str, Dataset]
    # Values are None if the model isn't usable
    external_models: Dict[str, Optional[Model]]

    # Array shapes for vars/attribs is (num_steps+1, num_policies, num_sims).
    var_values: Dict[str, np.ndarray]
    attrib_values: Dict[str, np.ndarray]
    # Inner values here can be treated like ndarrays with same shape as above plus
    # an additional final n_rows dimension. (In practice, there's some black magic
    # happening under the hood to keep only a subset of the full tensor as a memory
    # optimization.)
    dataset_var_values: Dict[str, Dict[str, DatasetVariableValueHolder]]
    # Same structure as above, except now the ndarrays have a shape like
    # (5, num_policies, num_rows). We store here just the data for t in
    # range 1-5, and for a single simulation.
    excerpts: Dict[str, Dict[str, np.ndarray]]

    var_lookup: Dict[str, BaseVariable]
    # Variable dependencies. This only tracks direct dependencies (not transitive).
    deps: Dict[str, Set[str]]
    
    # These are passed in the 'main errors' category of the API response.
    # TODO: This is sort of a hack. Should have dedicated category for external error models in API response.
    extra_errors: List[Error]

    nodes_in_order: List[str]

    @property
    def dataset_variables(self) -> List[DatasetAdditionVar]:
        return [
            var for dataset in self.datasets.values() 
            for var in dataset.added_vars
        ]

    @property
    def all_basevariables(self) -> List[BaseVariable]:
        return self.variables + self.attributes + self.dataset_variables

    def __init__(self,
        variables: Iterable[Variable] = [],
        policies: Iterable[PolicyDefinition] = [],
        datasets: Iterable[Dataset] = [],
        external_models: Iterable[ExternalModelDefinition] = [],
        num_steps: int = 50,
        num_sims: int = 50,
    ):
        self.extra_errors = []
        self.num_steps = num_steps
        self.num_sims = num_sims
        self.attributes = policies_to_attributes(policies)
        self.variables = variables
        self.datasets = {ds.name: ds for ds in datasets}
        
        self._validate_unique_names()
        self.var_lookup = {var.dotted_name: var for var in self.all_basevariables}
        self.external_models = {}
        for defn in external_models:
            try:
                model = parse_model_defn(defn)
                self.external_models[model.name] = model
            except VisibleError as e:
                name = defn['label']
                self.extra_errors.append([name, '', str(e)])
                self.external_models[name] = None
        self.num_policies = get_num_policies(self.attributes)
        self._transform_trees()
        self.deps = self._get_var_deps()
        top_order, circular_issues = topo_sort_removing_errors(self.deps)
        self.nodes_in_order = top_order
        self._add_preprocessing_errors(circular_issues)

        # Get data structures to hold variable values. Also populates with
        # any static initial values.
        self.var_values = _get_standard_value_holder(self.variables, self.num_policies, num_steps, num_sims)
        self.var_values["t"] = np.tile(
                np.reshape(range(self.num_steps+1), (self.num_steps+1, 1, 1)),
                (1, self.num_policies, self.num_sims)
        )
        self.attrib_values = _get_standard_value_holder(
            self.attributes, self.num_policies, num_steps, num_sims
        )
        self.dataset_var_values = self._get_dataset_value_holders()
        self.excerpts = {
                ds.name: _get_excerpt_placeholder(ds, self)
                for ds in self.datasets.values()
        }

    def _validate_unique_names(self):
        """Check that all variable names are unique. Record appropriate error
        message if any dupes are found."""
        namecount = Counter()
        for var in self.all_basevariables:
            name = var.dotted_name
            namecount[name] += 1
            if namecount[name] == 2:
                var.errors.append([
                    name, 'short_name',
                    'Duplicated variable name. This will result in undefined behaviour',
                ])

    def _get_var_deps(self) -> Dict[str, Set[str]]:
        """Return a dict with keys corresponding to (dotted) variable names (of all three
        BaseVariable types) and values are sets of variable names on which they
        depend.
        """
        var_deps = {}
        for qv in self.all_basevariables:
            name = qv.dotted_name
            trees = qv.trees if isinstance(qv, PolicyAttribute) else [qv.tree]
            collector = DependencyCollector()
            for tree in filter(None, trees):
                collector.visit(tree)
            var_deps[name] = collector.deps
        return var_deps

    def _get_dataset_value_holders(self) -> Dict[str, Dict[str, DatasetVariableValueHolder]]:
        # Calculate the deepest past references for each variable. This will determine
        # the size of the rolling window we need to keep for that variable. e.g. if
        # we have a formula referencing x[t-3], we need to keep a window of the 4 most
        # recent timesteps for x.
        collector = LagCollector()
        for qv in self.all_basevariables:
            trees = qv.trees if isinstance(qv, PolicyAttribute) else [qv.tree]
            for tree in filter(None, trees):
                collector.visit(tree)

        res = {}
        for dataset in self.datasets.values():
            ds_vals = {}
            for ds_var in dataset.added_vars:
                lag = collector.lags[ds_var]
                # lag=0 means no references to earlier timesteps. 1 means a reference
                # to t-1, etc. Width (the number of rolling timesteps we need to 
                # track for this var) will be 1 greater than this.
                width = min(lag+1, self.num_steps + 1)
                ds_vals[ds_var.name] = DatasetVariableValueHolder(
                        ds_var,
                        (width, self.num_policies, self.num_sims, dataset.n_rows),
                )
            res[dataset.name] = ds_vals
        return res

    def _run_one_update(
        self,
        node: BaseVariable,
        time: int,
        policy_index=None, # if None, we calculate values for all policies. Only set to non-None values when node is a PolicyAttribute.
    ) -> None:
        """Evaluates the given node at the given time and policy index, and 
        updates (var|dataset_var|attrib)_values with the calculated values.
        """
        if isinstance(node, PolicyAttribute):
            assert policy_index is not None
            tree = node.trees[policy_index]
        else:
            tree = node.tree
        if node.is_poisoned(policy_index):
            return
        if time == 0 and node.initial is not None:
            # don't evaluate. just use explicitly set initial value.
            return
        walker = TreeEvaluator(
            time, policy_index, self, node,
        )
        try:
            result = walker.enter(tree)
        except VisibleError as e:
            node.record_exception(e)
            node.set_poisoned(policy_index)
            return
        except SilentError:
            # as above, but don't record anything
            node.set_poisoned(policy_index)
            return
        if isinstance(node, DatasetAdditionVar):
            self.dataset_var_values[node.parent_table_label][node.name][time] = result
        elif isinstance(node, PolicyAttribute):
            self.attrib_values[node.name][time, policy_index, :] = result
        elif isinstance(node, Variable):
            self.var_values[node.name][time] = result

        else:
            raise ValueError(f"Not a BaseVariable: {node}")

    def _lookup_var(
        self,
        varname, # dotted
        time,
        policy_index,
    ):
        var = self.var_lookup[varname]
        if isinstance(var, PolicyAttribute):
            data = self.attrib_values[varname]
        elif isinstance(var, DatasetAdditionVar):
            ds_name = var.parent_table_label
            data = self.dataset_var_values[ds_name][var.name]
        # Types above are subsets of Variable, so this needs to come last
        # to avoid catching other lookup types
        elif isinstance(var, Variable):
            data = self.var_values[varname]
        else:
            raise ValueError(f"Unrecognized var type: {var}")
        timed_data = data[time]
        if policy_index is None:
            return timed_data
        else:
            return timed_data[policy_index]

    def _transform_trees(self):
        """Applies a transformation to all variables' equation syntax trees to
        replace certain node types (Attribute, Name, Subscript) with more 
        semantically transparent custom node types.
        """
        for qv in self.all_basevariables:
            transformer = TreeTransformer(qv, self)
            def transform(tree):
                if tree is None:
                    return tree
                try:
                    return transformer.visit(tree)
                except SilentError as e:
                    return None
                except VisibleError as e:
                    qv.record_exception(e)
                    return None
            if isinstance(qv, PolicyAttribute):
                qv.trees = [transform(t) for t in qv.trees]
            else:
                qv.tree = transform(qv.tree)

    def _compute_node(self, relevant_node: BaseVariable, t):
        if type(relevant_node) == PolicyAttribute:
            # For policy attributes, we do separate updates for each policy
            for policy_index in range(self.num_policies):
                _ = self._run_one_update(
                    relevant_node,
                    t,
                    policy_index=policy_index,
                )
        else:
            # for variables, we simulate results of all policies in one call
            _ = self._run_one_update(relevant_node, t)

    def run(self):
        for t in range(self.num_steps + 1):
            for name in self.nodes_in_order:
                relevant_node = self.var_lookup[name]
                self._compute_node(relevant_node, t)
            self._post_timestep_hook(t)

    def _post_timestep_hook(self, t):
        """Called after computing all variable values for given value of t.
        Used to invoke various bookkeeping tasks.
        """
        # update excerpt data (if t in [1,5])
        if constants.EXCERPT_START <= t <= constants.EXCERPT_END:
            for ds in self.datasets.values():
                for dsvar in ds.added_vars:
                    vals = self.dataset_var_values[ds.name][dsvar.name]
                    # We start our excerpts at t=1, not 0
                    self.excerpts[ds.name][dsvar.name][t-1] = vals[t, :, 0]
        # advance rolling data window for dataset variable value holders
        for varmap in self.dataset_var_values.values():
            for holder in varmap.values():
                holder.advance_time(t+1)

    def _organize_errors(self):
        collect = lambda vars: _organize_errors(
                [err for var in vars for err in var.errors]
        )
        return (
                _organize_errors([err for var in self.variables for err in var.errors]
                    + self.extra_errors),
                collect(self.attributes),
                collect(self.dataset_variables),
        )

    def _add_preprocessing_errors(
        self,
        circular_issues: Set[str],
    ) -> None:
        """
        Mutates main_errors, policy_errors and dataset_errors by adding some common error types identified before evaluation
        """
        # Dependencies on attrs with incomplete specification
        missing_attributes = {attr.name for attr in self.attributes if not attr.complete}
        for key, var_refs in self.deps.items():
            for var_ref in var_refs:
                if var_ref in missing_attributes:
                    self.var_lookup[key].errors.append(
                        [
                            key,
                            "equation",
                            f"depends on {var_ref} which is policy attribute that is not fully populated",
                        ]
                    )

        # Self refs
        self_var_ref_names = _get_self_refs(self.deps)
        for name in self_var_ref_names:
            var = self.var_lookup[name]
            var.errors.append(
                [
                    name,
                    "equation",
                    "a variable must not depend on itself in the same time period",
                ]
            )
        # Circular dependencies
        for name in circular_issues:
            var = self.var_lookup[name]
            var.errors.append(
                [name, "equation", "cannot determine values due to circular dependency"]
            )


def _get_standard_value_holder(
    nodes: Iterable[Union[Variable, PolicyAttribute]],
    num_policies,
    num_time_steps,
    num_sims,
    extra_dim_size=None,
) -> Dict[str, np.ndarray]:
    """Outputs the data structures that hold numeric values for variables and attribute values
    
    Data structure for each is a dict mapping variable name to ndarray having
    shape (num_time_steps+1, num_policies, num_sims) plus, optionally, extra_dim_size.
    """
    dims = (num_time_steps + 1, num_policies, num_sims)
    if extra_dim_size:
        dims += (extra_dim_size,)
    res = {
            node.name: np.full(dims, np.nan)
            for node in nodes
    }
    for node in nodes:
        res[node.name][0] = node.initial
    return res


def _get_excerpt_placeholder(dataset, sim) -> Dict[str, np.ndarray]:
    """Given a dataset and its parent simulation, return a dict mapping variable
    names (from that dataset) to ndarrays allocated with space to hold an excerpt
    of dataset variable values (returned as part of the main API response). The
    excerpt is characterized by:
    - using only the first few timesteps, starting from t=1
    - having data from only a single simulation
    """
    res = {}
    n_timesteps = min(constants.N_EXCERPT_TIMESTEPS, sim.num_steps)
    shape = (n_timesteps, sim.num_policies, dataset.n_rows)
    for dsvar in dataset.added_vars:
        res[dsvar.name] = np.full(shape, np.nan)
    return res

def _organize_errors(errors_list):
    """Given a collection of errors (encoded as 3-tuples), return a dict-of-dicts
    mapping name to error_type to lists of error details.
    """
    errors_by_name_by_type = defaultdict(lambda: defaultdict(list))
    for name, error_type, error_detail in errors_list:
        errors_by_name_by_type[name][error_type].append(error_detail)
    return errors_by_name_by_type


def _get_self_refs(refs: Dict[str, Set[str]]) -> List[str]:
    return [k for k in refs if k in refs[k]]
