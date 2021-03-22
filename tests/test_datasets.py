from io import StringIO
import pandas as pd
import numpy as np
import pytest

from decisionai.datasets import Dataset, DatasetAdditionVar

from sample_datasets import *
from helpers import run_sim, v, pols

def test_simple_df_retrieval():
    expected_as_string = """
name,workers,max_production
low, 1, 100
medium, 30, 200
"""
    expected = {"EXAMPLE_LABEL": pd.read_csv(StringIO(expected_as_string))}

    sample_db_entry = {
        "label": "EXAMPLE_LABEL",
        "objectPath": ["userFiles/TEST_uDt3x6DAX7Wkto3aFOxk_1589651148084.csv"],
        "uri": [
            "gs://dai-sim-app.appspot.com/userFiles/TEST_uDt3x6DAX7Wkto3aFOxk_1589651148084.csv"
        ],
        "variables": [],
    }
    dataset = Dataset.from_json(sample_db_entry)
    assert dataset.df.equals(expected["EXAMPLE_LABEL"])


def test_ds_var_parsing():
    sample_db_entry = {
        "label": "EXAMPLE_LABEL",
        "objectPath": ["userFiles/TEST_uDt3x6DAX7Wkto3aFOxk_1589651148084.csv"],
        "uri": [
            "gs://dai-sim-app.appspot.com/userFiles/TEST_uDt3x6DAX7Wkto3aFOxk_1589651148084.csv"
        ],
        "variables": [
            {
                "equation": "1",
                "short_name": "always_one",
            },
            {
                "equation": "t",
                "short_name": "t_again",
            },
        ],
    }

    dataset = Dataset.from_json(sample_db_entry)
    assert len(dataset.added_vars) == 2
    assert all(type(var) == DatasetAdditionVar for var in dataset.added_vars)


def test_forbid_var_with_dataset_dimension():
    db = WORKERS_DATASET_DEF.copy()
    user_variables = [
        {"short_name": "x", "equation": "EXAMPLE_LABEL.workers"}
    ]
    sim = run_sim(user_variables, datasets=[db], allow_errs=True)
    sim.assert_errors_match([ ['x', 'equation', 'extra dimension'] ])
    sim.assert_null_values('x')

def test_accessing_raw_dataset_with_full_name():
    db = WORKERS_DATASET_DEF.copy()
    db['variables'] = [
            v('capacity_per_worker', "EXAMPLE_LABEL.max_production / EXAMPLE_LABEL.workers"),
    ]
    sim = run_sim(datasets=[db])
    df = sim.dataset_df()
    assert (df.loc[df.rowId==0, 'capacity_per_worker'] == 100).all()
    assert (df.loc[df.rowId==1, 'capacity_per_worker'] == 200/30).all()

def test_accessing_raw_dataset_with_implicit_name():
    db = WORKERS_DATASET_DEF.copy()
    db['variables'] = [
            v('capacity_per_worker', "max_production / EXAMPLE_LABEL.workers"),
    ]
    sim = run_sim(datasets=[db])
    df = sim.dataset_df()
    assert (df.loc[df.rowId==0, 'capacity_per_worker'] == 100).all()
    assert (df.loc[df.rowId==1, 'capacity_per_worker'] == 200/30).all()

def test_dataset_addition_var_has_expanded_dimension():
    db = WORKERS_DATASET_DEF.copy()
    db['variables'] = [v('just_1', '1')]
    sim = run_sim(datasets=[db])
    df = sim.dataset_df()
    assert (df.just_1 == 1).all()
    assert len(df) == 2 * 3 # 2 rows, 3 timesteps, 1 simulation, 1 policy

def test_column_shadows_variable_name_in_dataset_equation():
    db = WORKERS_DATASET_DEF.copy()
    db['variables'] = [
            # the 'workers' here should be read as a ref to the dataset column,
            # and not the 'workers' variable defined separately below.
            v('capacity_per_worker', "max_production / workers"),
    ]
    sim = run_sim(
            [v('workers', '1000')],
            dataset=db,
    )
    df = sim.dataset_df()
    assert (df.loc[df.rowId==0, 'capacity_per_worker'] == 100).all()
    assert (df.loc[df.rowId==1, 'capacity_per_worker'] == 200/30).all()

def test_column_ref_via_square_brackets():
    sim = run_sim(
            [v('x', 'sum(EXAMPLE_LABEL["workers"])')],
            datasets=[WORKERS_DATASET_DEF],
    )
    # expected sum = 1 + 30
    sim.assert_values_match('x', [31, 31, 31, 31])

def test_initial_value_from_dataset_column():
    db = WORKERS_DATASET_DEF.copy()
    # A variable whose initial value is set to the 'workers' column of the dataset
    db['variables'] = [
            v('x', 'x[t-1] + 1', 'workers'),
    ]
    sim = run_sim(datasets=[db])
    expected = [
            [2, 3, 4],
            [31, 32, 33],
    ] # expected vals across timesteps per row - starting from t=1
    for row_ix in range(2):
        sim.assert_values_match(
                'EXAMPLE_LABEL.x',
                expected[row_ix],
                row=row_ix,
        )

def test_gc():
    """Test our garbage collection optimization for dataset variables (cf.
    DatasetVariableValueHolder)
    """
    db = INSURANCE_DATASET_DEF.copy()
    db['variables'] = [
            v('cy', 'max(cy[t-1], ctp)', '0'), # Should have width=2
            v('ctp', '0'), # Should have width=3
            v('rev', 'cy'),
            v('z', 'ctp[t-2] + 1'),
    ]
    sim = run_sim(
            dataset=db,
            num_steps=10,
            num_sims=50,
    )
    holders = sim.dataset_var_values['INS']
    var_to_width = {var: holder.width for (var, holder) in holders.items()}
    assert var_to_width == {
            'cy': 2,
            'ctp': 3,
            'rev': 1,
            'z': 1,
    }
