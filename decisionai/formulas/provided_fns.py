import numpy as np
import ast
import operator as op

from decisionai.errors import VisibleError
from .randomness import RANDOM_SAMPLING_FNS

class BadJoinError(VisibleError):
    error_type = 'equation'

OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.Lt: op.lt,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.LtE: op.le,
    # Consider allowing = instead of just == for equivalence testing
    ast.NotEq: op.ne,
    ast.Eq: op.eq,
    ast.And: np.logical_and,
    ast.Or: np.logical_or,
    ast.BitAnd: np.logical_and,
    ast.BitOr: np.logical_or
    # there must be some way to avoid hand-coding all of these... or maybe not
}

def summed_join(mask, coldat):
    """Same behaviour as join(), except:
    - innermost arrays in mask may sum to a number other than 1
    - we return the result of summing all matched values per row
    """
    assert mask.shape == coldat.shape or (
            mask.ndim == 4 and mask.shape[-1] == coldat.shape[-1]
    )
    return (mask * coldat).sum(axis=-1)


def join(mask, coldat):
    """
    mask: boolean array. Shape is either (npols, nsims, nrows), or
        (npols, nsims, other_rows, nrows), where the last two values
        are the (possibly equal) rowcounts of two distinct datasets.
        Furthermore, each row mask (i.e. each innermost array) must
        have exactly 1 True. For joins that may match 0 or many rows,
        there must be an associated aggregator, and thus the above function
        will be hit instead.
    coldat: array of dataset values. Shape is (npols, nsims, nrows) or
        (npols, nsims, 1, nrows)
    coldat and mask must have the same number of dimensions.
    
    Return value will be an array of shape mask.shape[:-1]
    """
    assert mask.shape == coldat.shape or (
            mask.ndim == 4 
            and mask.shape[-1] == coldat.shape[-1]
            and mask.ndim == coldat.ndim
    )
    mask_sizes = mask.sum(axis=-1)
    if not (mask_sizes == 1).all():
        raise BadJoinError(f"Join must be 1:1, but matched {mask_sizes.min()}-{mask_sizes.max()}"
                " rows.")
    if mask.shape != coldat.shape:
        # This means that coldat has a size-1 penultimate dimension whereas mask
        # has a larger one (correspond to rowcount of other dataset)
        puffed = np.broadcast_to(coldat, mask.shape)
    else:
        puffed = coldat

    out = puffed[mask].reshape(mask.shape[:-1])
    return out

def sum(input):
    return (
        np.sum(input, axis=2)
    )  # axis 2 is the rows of dataset in current indexing setup

def _reduced(ufunc):
    """Given a numpy ufunc that takes two arrays and returns one, return a function
    that processes an arbitrary number of arrays by reducing that ufunc.
    """
    return lambda *args: ufunc.reduce(args)

FUNCTIONS = {
    **RANDOM_SAMPLING_FNS,
    "ifelse": np.where,
    "where": np.where,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "min": _reduced(np.minimum),
    "max": _reduced(np.maximum),
    "sum": sum,
    "sum_from_dataset": sum,
    "join": join,
}

# sum_from_dataset is a deprecated alias for sum
EXPOSED_FUNCTION_NAMES = FUNCTIONS.keys() - {'sum_from_dataset'}
