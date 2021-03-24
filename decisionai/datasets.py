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

from .variables import SingleTreeVariable


class DatasetVarDefinition(TypedDict, total=False):
    short_name: str
    equation: str
    # Optional: initial value. May be number or name of a dataset column.
    initial: str

class DatasetAdditionVar(SingleTreeVariable):
    # initial values may be a sequence if a column is provided
    initial: Union[Number,None,Sequence]
    parent_table_label: str

    def __init__(self, name, tree, initial, parent_table_label, errors=None):
        self.parent_table_label = parent_table_label
        self.tree = tree
        super().__init__(name, initial, errors)

    @property
    def dotted_name(self):
        return self.parent_table_label + '.' + self.name

    @classmethod
    def from_json(cls, raw: DatasetVarDefinition, dataset: 'Dataset'):
        name = raw['short_name']
        formula = raw["equation"]
        errors = []
        parent_table_label = dataset.name
        try:
            tree = cls._parse_eqn(formula)
        except SyntaxError as err:
            errors.append([name, "equation", err.msg])
            tree = None

        init = raw.get("initial")
        if init is None:
            pass
        elif init in dataset.df.columns:
            init = dataset.df.loc[:, init].values
        else:
            try:
                init = float(init)
            except ValueError:
                errors.append([parent_table_label + '.' + name, "initial",
                    f"Initial value must be a number or column name. Was: {init}"])
                init = None
        return cls(name, tree, init, parent_table_label, errors)

class DatasetDefinition(TypedDict, total=False):
    label: str
    variables: List[DatasetVarDefinition]
    objectPath: List[str]
    # Used only in unit tests (in place of above two keys)
    csvLiteral: str

class Dataset(NamedTuple):
    # TODO: should probably call this 'label' for consistency with naming elsewhere
    name: str
    df: pd.DataFrame
    added_vars: Iterable[DatasetAdditionVar]

    @property
    def shortnames(self) -> Set[str]:
        return {v.name for v in self.added_vars}

    @property
    def columns(self):
        return self.df.columns

    def add_variable(self, vardef: DatasetVarDefinition):
        var = DatasetAdditionVar.from_json(vardef, self)
        self.added_vars.append(var)

    @classmethod
    def from_json(cls, raw: DatasetDefinition) -> 'Dataset':
        file_label = raw["label"]
        literal = raw.get('csvLiteral')
        # TODO: Have this access real data
        df = pd.DataFrame()
        ds = cls(name=file_label, df=df, added_vars=[])
        for vardef in raw.get("variables", []):
            ds.add_variable(vardef)
        return ds

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

