from typing import List, NamedTuple, Optional
import numpy as np

class DrilldownDefinition(NamedTuple):
    """Specification for a 'drilldown' analysis, as made available in the
    explore view in the UI. For each (row, variable) tuple in the product
    of the given row labels and variables, we are to compute either:
    - the value of that variable for that row at a given point in time
    - the average value over all time steps
    - the total value over all timesteps
    Depending on agg_method.
    """
    # One of "total", "mean", "pointInTime"
    agg_method: str
    # a value for t if agg_method='pointInTime'. None, otherwise.
    point: Optional[int]
    dataset_label: str
    # Row indices
    row_labels: List[int]
    # These should be the names of dataset addition variables
    variables: List[str]

    @classmethod
    def from_json(cls, dat):
        agg = dat['agg']
        if isinstance(agg, str):
            agg_method = agg
            point = None
        else:
            agg_method = 'pointInTime'
            point = agg['pointInTime']
        return cls(
                agg_method,
                point,
                dat['datasetLabel'],
                dat['rowIndices'],
                dat['variables'],
        )

class DrilldownComputer:
    """Incrementally accumulates values in order to calculate the results for
    a drilldown specification. Managed by a Simulation object. After calculating
    all values for a given timestep t, the sim will call each computer's ingest
    method so that it can process any data it needs from time t.
    """
    spec: DrilldownDefinition
    # Has shape (n_vars, n_row_indices, n_policies)
    # Used for different purposes by different computers
    dat: np.ndarray
    # Value to fill self.dat with when initializing it
    initial = np.nan

    def __init__(self, spec, n_policies):
        self.spec = spec
        n_vars = len(spec.variables)
        n_rows = len(spec.row_labels)
        shape = (n_vars, n_rows, n_policies)
        self.dat = np.full(shape, self.initial)

    def _timestep_is_interesting(self, t):
        """ingest() will exit immediately if this returns False for the timestep
        under consideration.
        """
        # None of our aggregation methods use the values from t=0. 
        # e.g. the 'total' aggregator sums up the total for t=1 + t=2 + ... t=num_steps
        return t > 0

    def ingest(self, ds_var_values, t):
        if not self._timestep_is_interesting(t):
            return
        varmap = ds_var_values[self.spec.dataset_label]
        # TODO: Could probably vectorize this more if it turns out to be a bottleneck
        for iv, varname in enumerate(self.spec.variables):
            # Shape (n_policies, n_sims, n_rows)
            vals = varmap[varname][t]
            for ir, row_ix in enumerate(self.spec.row_labels):
                # New shape = (n_policies,)
                v = vals[:, :, row_ix].mean(axis=1)
                self.ingest_row(iv, ir, t, v)

    def ingest_row(self, var_ix, row_label_ix, t, vals):
        """
        var_ix and row_label_ix are indices into spec.row_labels and spec.variables,
        respectively. vals is an array of shape (n_policies). (The simulations
        dimension having already been averaged out.)
        """
        raise NotImplementedError

    def results_dict(self):
        return {
                row_label: {
                    varname: self.values_for(row_ix, var_ix)
                    for (var_ix, varname) in enumerate(self.spec.variables)
                }
                for (row_ix, row_label) in enumerate(self.spec.row_labels)
        }

    def values_for(self, row_ix, var_ix):
        return self.dat[var_ix, row_ix].tolist()

    @staticmethod
    def for_spec(spec, sim):
        agg_to_cls = dict(
                total=SummingComputer,
                mean=MeanComputer,
                pointInTime=PointInTimeComputer,
        )
        cls = agg_to_cls[spec.agg_method]
        return cls(spec, sim.num_policies)


class PointInTimeComputer(DrilldownComputer):
    # dat values correspond to value at given point in time (avged over sims)

    def _timestep_is_interesting(self, t):
        return t == self.spec.point

    def ingest_row(self, var_ix, row_label_ix, t, vals):
        if t == self.spec.point:
            self.dat[var_ix, row_label_ix] = vals

class SummingComputer(DrilldownComputer):
    # dat values correspond to sum over timesteps (avged over sims)
    # Since dat is a running sum, initialize it to all zeros
    initial = 0.0

    def ingest_row(self, var_ix, row_label_ix, t, vals):
        self.dat[var_ix, row_label_ix] += vals

class MeanComputer(SummingComputer):
    n_timesteps: int # denom of mean

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_timesteps = 0

    def ingest(self, vals, t):
        # inelegant, but whatever
        if self._timestep_is_interesting(t):
            self.n_timesteps += 1
            super().ingest(vals, t)

    def values_for(self, row_ix, var_ix):
        return (self.dat[var_ix, row_ix] / self.n_timesteps).tolist()
