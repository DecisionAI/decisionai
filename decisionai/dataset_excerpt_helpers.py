"""
Helper functions related to reshaping and munging data that we intend to send
in API responses for web app. 

Once we solve https://github.com/DecisionAI/decisionai/issues/2 we can remove this
"""
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
            sim._raw_var_values["t"][timeslice]
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



