from collections import Counter
from typing import Iterable, List
import numpy as np
import pytest

from decisionai import api_helpers, constants
from decisionai.simulation import Simulation
from decisionai import api_helpers
from decisionai.variable import Variable
from decisionai.dataset import Dataset
from decisionai.policy import PolicyDefinition

def pols(**kwargs) -> List[PolicyDefinition]:
    """
    kwargs should be structured such that keys are the names of attributes
    and values are lists of strings, each representing a formula.
    Returns a list of PolicyDefinition dicts. Policies will be named "p1, "p2",
    etc.
    """
    pols = []
    npols = None
    for attr, formulas in kwargs.items():
        if npols is None:
            npols = len(formulas)
            for i in range(1, 1+npols):
                pols.append( {'policy_name': f'p{i}'} )
        assert len(formulas) == npols
        for i, formula in enumerate(formulas):
            pols[i][attr] = formula
    return pols

class TestSim(Simulation):
    """A wrapper around Simulation with some additional methods useful for testing.
    """
    @property
    def var_df(self):
        return api_helpers.var_values_df(self)

    def dataset_df(self, label=None):
        # NB: because we're using the excerpt code path, there will be no
        # simId column, and we'll only have data for one sim.
        if label is None:
            assert len(self.datasets) == 1, "Explicit label must be provided if n datasets != 1"
            label = list(self.datasets.keys())[0]
        df_dict = api_helpers.dataset_var_values_excerpt_dfs(self, [label])
        return df_dict[label]

    def assert_values_match_across_time(self, varname, expected, **kwargs):
        assert 't' not in kwargs
        if '.' in varname:
            # For dataset variables, we'll be using the excerpt df, which has only
            # a limited set of timesteps
            steps = range(
                    constants.EXCERPT_START,
                    min(constants.EXCERPT_END, self.num_steps)+1
            )
        else:
            steps = range(self.num_steps)
        for t in steps:
            kwargs['t'] = t
            self.assert_values_match(varname, expected, **kwargs)

    def get_values(self, varname,
            policy=None, sim_id=None,
            t=None,
            row=None,
    ):
        if '.' in varname:
            assert sim_id is None, ("sim_id filtering not supported for dataset"
                    " variables (we always use sim 0 due to using excerpt data)")
            label, varname = varname.split('.')
            df = self.dataset_df(label)
        else:
            df = self.var_df
        if policy is not None:
            df = df[df.policyId == policy]
        if sim_id is not None:
            df = df[df.simId == sim_id]
        if t is not None:
            df = df[df.t == t]
        if row is not None:
            df = df[df.rowId == row]
        if t is None:
            sortkey = 't'
        elif row is None:
            sortkey='rowId'
        else:
            sortkey = ['t', 'rowId']
        return df.sort_values(by=sortkey).loc[:,varname].values


    def assert_values_match(self, varname, expected, **kwargs):
        """Asserts that the flattened values for varname (when sorted by t then
        rowId) match the given expected value(s). Notes:
        - expected may be a list, numpy array, or a scalar. We do a broadcasting comparison.
        - if varname is the name of a dataset variable, then we will compare against
          our *excerpt* data, which ranges from timesteps 1-5 only, and only 
          corresponds to a single simulation. (This is not ideal, but is arguably the
          least bad option. Alternative would be passing in some flag to the simulation
          for testing telling it to store data for all timesteps in its excerpt, or to
          use some drop-in replacement for DatasetVariableValueHolder that remembers
          everything. But then we're testing something substantially different from
          the 'real' code. But at least, should maybe consider having a differently
          named method like assert_dataset_excerpt_values_match, for clarity.)
        """
        # NB: expected may be a scalar, in which case we broadcast
        vals = self.get_values(varname, **kwargs)
        np.testing.assert_allclose(vals, expected)

    def assert_null_values(self, varname):
        assert self.var_df[varname].isnull().all()

    def errors(self):
        return (
            [err for var in self.all_basevariables for err in var.errors]
            + self.extra_errors
        )

    def assert_no_errors(self):
        __tracebackhide__ = True
        errs = self.errors()
        if len(errs) > 0:
            pytest.fail(f"Expected no errors. Got {len(errs)}: {errs}")

    def assert_errors_match(self, patterns: List[List[str]]):
        """patterns should be a list of length 2 or 3 lists. The first two elements
        are prefixes of an expected Error (i.e. variable name, and error type). The third
        element, if present, is an expected substring of the third element of the Error
        list, i.e. the message.
        """
        # TODO: should give nicer messages on assertion failures
        __tracebackhide__ = True
        errs = self.errors()
        if len(patterns) != len(errs):
            pytest.fail(f"Expected {len(patterns)} errors, but got {len(errs)}."
                    f" Errors were: {errs}")
        # NB: Assuming (for now) that [name, type] prefix is unique
        pre_map = {tuple(err[:2]) : err for err in errs}
        pat_map = {tuple(pat[:2]) : pat for pat in patterns}
        if pre_map.keys() != pat_map.keys():
            pytest.fail(f"Expected error signatures to match {pat_map.keys()!r}. Were: {pre_map.keys()!r}")
        for pat in patterns:
            if len(pat) < 3:
                continue
            name, errtype, substr = pat
            matched_msg = pre_map[(name, errtype)][2]
            if not substr in matched_msg:
                pytest.fail(f"Error message was {matched_msg!r}. Failed to match "
                        f"substring {substr!r}")

    def assert_errors_include(self, patterns):
        errs = self.errors()
        assert all(len(pat) == 2 for pat in patterns), "Other pattern lengths not implemented"
        patterns = map(tuple, patterns)
        prefixes = [tuple(err[:2]) for err in errs]
        for pat in patterns:
            assert pat in prefixes



def run_sim(
    variables: Iterable[Variable] = None,
    policies: Iterable[PolicyDefinition] = None,
    datasets: Iterable[Dataset] = None,
    models = None,
    num_steps: int = 3,
    num_sims: int = 1,
    allow_errs=False,
    # convenience params for the common case where there's only one dataset/model
    dataset : Dataset = None,
    model = None,
):
    variables = variables or []
    policies = policies or []
    datasets = datasets or []
    if dataset:
        assert not datasets, "Params datasets and dataset are mutually exclusive"
        datasets = [dataset]
    models = models or []
    if model:
        assert not models
        models = [model]
    sim = TestSim(variables, policies, datasets, models, num_steps, num_sims)
    sim.run()
    if not allow_errs:
        sim.assert_no_errors()
    return sim