import ast

from decisionai.dataset import DatasetAdditionVar
from .errors import EvaluationError
from .synthetic_nodes import SimpleVarNode, ColumnNode

def varname_for_simple_indexee(indexee):
    """indexee is a thing being indexed. It should be a variable reference
    (possibly a dotted reference to a dataset addition var). This function returns
    a corresponding dotted variable name.
    Will fail if the indexee is not of the required form."""
    if isinstance(indexee, ast.Name):
        return indexee.id
    elif isinstance(indexee, ast.Attribute):
        # Dotted reference to a dataset addition variable
        assert isinstance(indexee.value, ast.Name)
        return indexee.value.id + '.' + indexee.attr
    else:
        raise EvaluationError("Can't parse complex left-hand-side of index expression.")

def dataset_for_node(node):
    """If the given (post-transform) node corresponds to a dataset variable
    or a dataset column reference, return the corresponding dataset name.
    Otherwise, return none.
    """
    if isinstance(node, SimpleVarNode) and isinstance(node.var, DatasetAdditionVar):
        return node.var.parent_table_label
    elif isinstance(node, ColumnNode):
        return node.dataset_name
