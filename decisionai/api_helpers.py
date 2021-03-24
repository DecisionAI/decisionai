"""
Helper functions related to reshaping and munging data that we intend to send
in API responses.
"""
import itertools
from typing import Iterable, List, Dict, Tuple, Any
from numbers import Number
import numpy as np
import pandas as pd

from decisionai import constants

def excerpt_dataset_results(ds_var_values):
    """Input is a dict of dataset var values, in the format returned by
    format_var_values
    """
    res = {}
    for ds_label, rows in ds_var_values.items():
        newrows = [row for row in rows
            if 
                (row['simId'] == 0)
                and
                (constants.EXCERPT_START <= row['t'] <= constants.EXCERPT_END)
        ]
        # don't include simId key
        for row in newrows:
            del row['simId']
        res[ds_label] = newrows
    return res

def var_values_df(sim):
    return _var_values_df(sim.var_values)

def dataset_var_values_excerpt_dfs(sim, labels=None):
    """
    Return a dict mapping dataset labels to dataframes of dataset variable
    values. df will have a column per variable plus: [rowId, t, policyId]
    Data will be excerpted to contain:
    - only timesteps 1-5
    - only simId = 0 (hence, unlike var values df, there is no SimId column)
    """
    timeslice = slice(constants.EXCERPT_START, 1+constants.EXCERPT_END) # only t from 1-5
    # List of ~5 arrays, each having shape (n_policies,)
    time_arrays = [
            arr[:,0] # only simId = 0
            for arr in
            sim.var_values["t"][timeslice]
    ]
    n_timesteps = len(time_arrays)
    ts = np.array(time_arrays)
    assert len(time_arrays[0]) == sim.num_policies
    base_polIds = range(sim.num_policies)
    polIds = np.tile(base_polIds, n_timesteps)
    res = {}
    for ds_label, var_values in sim.excerpts.items():
        if labels and ds_label not in labels:
            continue
        n_rows = sim.datasets[ds_label].n_rows
        base_rowIds = range(n_rows)
        cols = dict(
                t=np.repeat(ts, n_rows),
                policyId=np.tile(
                    np.repeat(base_polIds, n_rows),
                    n_timesteps
                    ),
                rowId=np.tile(base_rowIds, sim.num_policies*n_timesteps),
        )
        for var, excerpt in var_values.items():
            cols[var] = excerpt.flatten()
        df = pd.DataFrame(cols)
        res[ds_label] = df
    return res


def _var_values_df(var_values):
    """Return a dataframe with columns [t, simId, policyId] + a column for each
    variable name in var_values.
    """
    time_arrays = var_values["t"]
    # List of tuples indexing values in any given timestep. (0, 0), (0, 1), ... (1, 0)...
    rowcols = list(itertools.product(*map(range, time_arrays[0].shape)))

    # Adding a timestep index. ((0, 0), 0), ((0, 1), 0), ... ((1, 0), 0), ... ((0, 0), 1), ...
    rowcol_ts = list(
        itertools.chain(
            *([[(rowcol, t) for rowcol in rowcols] for t in range(len(time_arrays))])
        )
    )

    simIds = [x[0][1] for x in rowcol_ts]
    policyIds = [x[0][0] for x in rowcol_ts]
    ts = [x[1] for x in rowcol_ts]

    sim_pol_ts = list(zip(simIds, policyIds, ts))

    def get_item(arrays, sim_pol_t):
        sim, pol, t = sim_pol_t
        a = arrays[t]
        return a[pol, sim]

    def add_sim_policy_t_columns(in_results: Dict[str, List]):
        # Map these three standard keys to flat lists of their values
        in_results["simId"] = map(lambda x: x, simIds)
        in_results["policyId"] = policyIds
        in_results["t"] = ts

    # Map variable name to flat list of values
    var_results = {
        varname: list(map(lambda sim_pol_t: get_item(var_val_array, sim_pol_t), sim_pol_ts))
        for varname, var_val_array in var_values.items()
    }

    # Add standard keys (simId, policyId, t)
    add_sim_policy_t_columns(var_results)
    return pd.DataFrame(var_results)
