from io import StringIO
from numbers import Number
import pandas as pd
import numpy as np
import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
from typing import (
        Iterable, List, NamedTuple, Dict, Any, Set, Sequence,
        Union, Tuple,
)

from .variable import Variable


class DatasetAdditionVar(Variable):
    # initial values may be a sequence if a column is provided
    initial: Union[Number,None,Sequence]
    parent_table_label: str

    def __init__(self, var, parent_table_label):
        # TODO: Use less hacky way to take Variable as input and hold a wrapped version as a DatasetAdditionVar
        self.name = var.name
        self.formula = var.formula
        self.initial = var.initial
        self.errors = var.errors
        self._poisoned = var._poisoned
        self.tree = var.tree
        self.parent_table_label = parent_table_label

    @property
    def dotted_name(self):
        return self.parent_table_label + '.' + self.name


class Dataset():
    # TODO: improve consistency between "name" and "label" for naming datasets
    name: str
    df: pd.DataFrame
    added_vars: Iterable[Variable]

    def __init__(self, name, df, vars):
        self.name = name
        self.df = df
        self.added_vars = [DatasetAdditionVar(var, self.name) for var in vars]

    # TODO: Did this to simplify modifying a test. In long term, we shouldn't be mutating a dataset by adding a var
    def _add_var(self, var):
        self.added_vars.append(DatasetAdditionVar(var, self.name))

    @property
    def shortnames(self) -> Set[str]:
        return {v.name for v in self.added_vars}

    @property
    def columns(self):
        return self.df.columns

    @property
    def n_rows(self):
        return len(self.df)

class DatasetVariableValueHolder:
    """Used by Simulator to track dataset variable values. From an outside
    perspective, this can be treated as if it's an ndarray with the full
    tensor of variable data of shape 
        (n_steps, n_policies, n_simulations, n_rows).

    Under the hood, we're doing some optimization to keep only values corresponding
    to a limited rolling window of timesteps.

    The advance_time() method must be called after computing the values for 
    each timestep so that we can keep our window in sync.
    """
    var: DatasetAdditionVar
    # The timestep after the most recent one that's been completely computed.
    current_timestep: int
    # We do a list of arrays instead of one array with a timestep dimension
    # to avoid excessive copying when advancing time
    # List starts at length 1 and grows up to max width specified at initialization.
    # Lists are ordered by increasing time, with the last index always corresponding
    # to current_timestep.
    arrs: List[np.ndarray]
    # Each array in arrs will have this shape. Namely: (n_policies, n_sims, n_rows)
    inner_shape: Tuple[int]
    def __init__(self, var, shape):
        self.var = var
        self.current_timestep = 0
        width, n_pols, n_sims, n_rows = shape
        assert width > 0
        self.width = width
        self.inner_shape = shape[1:]
        self.arrs = [np.full(self.inner_shape, (np.nan if var.initial is None else var.initial))]

    def __repr__(self):
        return (f"{self.__class__.__name__} for variable {self.var.dotted_name} with "
        f"width={self.width}, inner_shape={self.inner_shape}"
        #f", arrs={self.arrs}"
        )

    @property
    def full(self):
        return len(self.arrs) == self.width

    def _translate_time_index(self, t):
        """Given a reference to an absolute timestep, return the corresponding
        relative index for self.arrs.
        """
        assert isinstance(t, int), f"Indexing by type {type(t)} not supported"
        assert t >= 0, "Negative indices not supported"
        assert t <= self.current_timestep, (f"Illegal future lookup. Index"
                f" was {t}, current timestep was {self.current_timestep}.")
        # If t == current_timestep, we want to return -1 (data for current timestep
        # is always at the end of the list). If t is one less than current timestep,
        # we want to return -2, etc.
        # NB: we don't need to check whether this is out of bounds. Should naturally
        # lead to an IndexError when we try to use it.
        return t - self.current_timestep - 1

    def __getitem__(self, key):
        # Key could be an int, a slice, or a tuple of the same
        if isinstance(key, tuple):
            time, rest = key[0], key[1:]
        else:
            time = key
            rest = slice(None) # no-op
        rel = self._translate_time_index(time)
        return self.arrs[rel][rest]

    def __setitem__(self, key, val):
        assert key == self.current_timestep
        assert val.shape == self.inner_shape
        self.arrs[-1] = val

    def advance_time(self, to):
        assert to==self.current_timestep + 1
        placeholder = np.full(self.inner_shape, np.nan)
        if not self.full:
            self.arrs.append(placeholder)
        else:
            # evict the oldest timestep
            self.arrs = self.arrs[1:] + [placeholder]
        self.current_timestep += 1

