"""
Custom ast.AST subclasses produced by our transformer.
"""
import ast
from typing import Sequence, Callable
import numpy as np

from .errors import PreprocessingError
from .provided_fns import FUNCTIONS
from decisionai.variable import BaseVariable
from decisionai.external_models import Model

class BaseVarNode(ast.AST):
    var: BaseVariable

    def __repr__(self):
        return f"{self.__class__.__name__} referencing variable {self.var!r}"

class SimpleVarNode(BaseVarNode):
    # Should generally be non-positive. e.g. x[t - 3] will result in offset=-3
    offset: int

    def __init__(self, var, offset=0):
        super().__init__()
        self.var = var
        self.offset = offset

    def __repr__(self):
        return (f"{self.__class__.__name__} referencing variable {self.var!r}"
                f" with offset {self.offset}")

class RichlyIndexedVarNode(BaseVarNode):
    index_tree: ast.AST

    def __init__(self, var, tree):
        super().__init__()
        self.var = var
        self.index_tree = tree
        # This needs to be set so that this is considered a child node, for the
        # purposes of ast.iter_child_nodes (which is used by generic_visit)
        self._fields = ['index_tree']

class TNode(ast.AST):
    """A reference to 't'. We treat this differently from other variable references
    because t has some unique properties. For one thing, it has no corresponding
    BaseVariable object. Also, we're not interested in tracking dependencies on t,
    since we know everything already implicitly depends on it.
    """
    pass

class ColumnNode(ast.AST):
    dataset_name: str
    column: str

    def __init__(self, dataset_name, column):
        self.dataset_name = dataset_name
        self.column = column

class ModelMethodNode(ast.AST):
    """A call of the form MY_MODEL.predict(...) or MY_MODEL.predict_proba(...)
    """
    model: Model
    method: Callable
    method_name: str
    args: Sequence[ast.AST]

    def __init__(self, model, method_name, args):
        self.model = model
        self.method_name = method_name
        allowed_methods = {'predict', 'predict_proba'}
        if method_name not in allowed_methods:
            raise PreprocessingError("Only permitted methods for external "
                f"models are predict and predict_proba. Got {method_name!r}"
            )
        try:
            self.method = getattr(model, method_name)
        except AttributeError:
            raise PreprocessingError(f"Model {model.name} does not support"
                    " method {method_name!r}"
            )
        self.args = args
        self._fields = ['args']


class FunctionCallNode(ast.AST):
    """An invocation of a provided function such as where(...), join(...), etc.
    """
    fn: Callable
    fn_name: str
    args: Sequence[ast.AST]
    _fields = ['args']

    def __init__(self, fn_name, args):
        self.fn_name = fn_name
        try:
            self.fn = FUNCTIONS[fn_name]
        except KeyError:
            raise PreprocessingError(f"Unrecognized function name {fn_name!r}")
        self.args = args
        self._validate()

    def _validate(self):
        if isinstance(self.fn, np.ufunc):
            # numpy ufuncs have a .nin attribute, which is the number of 
            # arrays they operate on. Verify that we're actually called with
            # that many args. Note that these functions generally also have additional
            # optional arguments (which we don't expose to the user). So if they
            # are called with too many args, we can't necessarily rely on it
            # raising a TypeError. It may instead fail subtly.
            if len(self.args) != self.fn.nin:
                raise PreprocessingError(f"Function {self.fn_name!r} takes"
                        f" {self.fn.nin} arguments, but was called with"
                        f" {len(self.args)}."
                )


    def __repr__(self):
        return (f"{self.__class__.__name__} node for fn {self.fn_name}"
                f" with args: {', '.join(repr(arg) for arg in self.args)}"
        )

class SumJoinNode(ast.AST):
    """Encapsulates a compound expression of the form sum(join(BOOL_EXPR, COL))
    Under the hood, we treat these together as if they form one function because
    join() has context-sensitive behaviour depending on whether it is
    wrapped by an aggregator.
    """
    # The first arg to join
    mask_node: ast.AST
    # The second arg
    col_node: ast.AST
    _fields = ['mask_node', 'col_node']

    def __init__(self, mask, col):
        self.mask_node = mask
        self.col_node = col
