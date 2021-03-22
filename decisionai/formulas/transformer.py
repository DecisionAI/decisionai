"""
TreeTransformer is used for an initial transformation of the parse trees for equations.
It has the effect of abstracting away some low-level parsing details so we don't need
to re-do them in multiple places where we interact with the tree.

For example, in the context of a dataset variable formula, the expressions "ds_var_1",
"ds_var_1[t]", and "MY_DATASET.ds_var_1" all have the same meaning (assuming that
the formula is for a dataset with label "MY_DATASET" which has a variable called "ds_var_1"),
but they have very different surface ast structure. TreeTransformer will transform each
of these three expressions into identical "SimpleVarNode" objects.

See README for more context.
"""
import ast
import numpy as np

from .synthetic_nodes import *
from .errors import *
from .tree_utils import varname_for_simple_indexee
from .provided_fns import FUNCTIONS
from .randomness import sample, RANDOM_SAMPLING_FNS
from decisionai.datasets import DatasetAdditionVar
from decisionai.errors import VisibleError, SilentError
from decisionai.variables import BaseVariable

class TreeTransformer(ast.NodeTransformer):
    def __init__(self, qv:BaseVariable, sim):
        super().__init__()
        self.qv = qv
        self.sim = sim

    def visit_Call(self, node):
        # node.func is the thing being called. Usually a Name (regular fn call),
        # or an Attribute (method call). Theoretically could be something more exotic,
        # e.g. where(t==5, uniform, normal), though not clear if this kind of thing is
        # currently supported, or intended to be supported. Niche, for sure.
        callee = node.func
        # The specific case of a sum wrapping a join is delicate enough (and different
        # enough from the normal semantics of sum()) that we'll handle it as a separate
        # node type.
        if (isinstance(callee, ast.Name) 
                and callee.id == "sum"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)
                and node.args[0].func.id == 'join'
        ):
            return self._parse_Sumjoin(node)
        args = [self.visit(arg) for arg in node.args]
        # Option 1: call to a provided function (e.g. min, sum_from_dataset)
        if isinstance(callee, ast.Name):
            func_name = callee.id
            return FunctionCallNode(func_name, args)
        # Option 2: predict method of an external model
        elif isinstance(callee, ast.Attribute):
            if not isinstance(callee.value, ast.Name):
                raise PreprocessingError("Indirect attribute references not supported.")
            model_name = callee.value.id
            method_name = callee.attr
            try:
                model = self.sim.external_models[model_name]
            except KeyError:
                raise PreprocessingError(f"No external model named {model_name!r}")
            if model is None:
                # This indicates we tried to load a model with this name but it was
                # invalid. There will already be an error about this fact, so fail silently.
                raise SilentError()
            return ModelMethodNode(model, method_name, args)
        else:
            raise PreprocessingError("Indirect function calling not supported.")

    def _parse_Sumjoin(self, node):
        args = node.args
        joinargs = args[0].args
        if len(joinargs) != 2:
            raise PreprocessingError(f"join takes exactly 2 args, got {len(joinargs)}")
        mask, col = [self.visit(arg) for arg in joinargs]
        return SumJoinNode(mask, col)

    def visit_Attribute(self, node):
        """Possibilities:
        - dotted reference to dataset column
        - dotted reference to dataset addition var
        (Note that model.predict is *not* a possibility because visit_Call
        doesn't recurse.)
        """
        if not isinstance(node.value, ast.Name):
            raise PreprocessingError("Indirect attribute references not supported.")
        label = node.value.id
        attr = node.attr
        if label not in self.sim.datasets:
            raise MissingDatasetError(label)
        ds = self.sim.datasets[label]
        if attr in ds.shortnames:
            dotted_name = label + '.' + attr
            var = self.sim.var_lookup[dotted_name]
            return SimpleVarNode(var)
        elif attr in ds.columns:
            return ColumnNode(label, attr)
        else:
            raise PreprocessingError(f"Dataset {label} has no column or variable"
                    f" named {attr}")

    def _resolve_var_name(self, name) -> BaseVariable:
        if isinstance(self.qv, DatasetAdditionVar) and '.' not in name:
            # In case of ambiguity, a dataset var/column should take precedence
            # over a regular variable/attribute
            label = self.qv.parent_table_label
            ds = self.sim.datasets[label]
            if name in ds.shortnames:
                dotted_name = label + '.' + name
                var = self.sim.var_lookup[dotted_name]
                return var
        try:
            return self.sim.var_lookup[name]
        except KeyError:
            raise MissingVariableError(name)
            
    def visit_Name(self, node):
        name = node.id
        if name == 't':
            return TNode()
        if isinstance(self.qv, DatasetAdditionVar):
            # In case of ambiguity, a dataset var/column should take precedence
            # over a regular variable/attribute
            label = self.qv.parent_table_label
            ds = self.sim.datasets[label]
            if name in ds.columns:
                return ColumnNode(label, name)
        var = self._resolve_var_name(name)
        return SimpleVarNode(var)

    def visit_Subscript(self, node):
        if not isinstance(node.slice, ast.Index):
            raise PreprocessingError("Slicing not supported")
        indexer = node.slice.value
        indexed_varname = varname_for_simple_indexee(node.value)
        # Option 1: reference to dataset column. e.g. MY_DATA["col_1"]
        # nb: ast.Str will be deprecated in python 3.8 in favour of ast.Constant
        if isinstance(indexer, ast.Str):
            if indexed_varname not in self.sim.datasets:
                raise MissingDatasetError(indexed_varname)
            ds = self.sim.datasets[indexed_varname]
            colname = indexer.s
            if colname not in ds.columns:
                raise PreprocessingError(f"Dataset {indexed_varname} has no"
                        " column {colname!r}")
            return ColumnNode(indexed_varname, colname)
        # Option 2: a variable with a timestep index. e.g. profit[t-1]
        var = self._resolve_var_name(indexed_varname)
        try:
            offset = self._parse_time_offset(indexer)
        except:
            # offset cannot be statically determined. Assuming this
            # induces no dependency. But we could be wrong.
            # we'll walk the index expression, because it could be arbitrarily
            # complicated, and include its own variable references
            self.visit(indexer)
            return RichlyIndexedVarNode(var, indexer)
        return SimpleVarNode(var, offset)

    def _parse_time_offset(self, indexer):
        """Only handles index expressions of the form [t] or [t-constant].
        Otherwise, raises an exception.
        """
        if isinstance(indexer, ast.Name):
            assert indexer.id == "t"
            return 0
        elif isinstance(indexer, ast.BinOp):
            left = indexer.left
            op = indexer.op
            right = indexer.right
            assert left.id == 't', indexer
            assert isinstance(op, ast.Sub), op
            assert isinstance(right, (ast.Num, ast.Constant)), right
            offset = getattr(right, 'n', getattr(right, 'value', None))
            return -offset
        else:
            raise ValueError("Failed to statically analyze index expression")
