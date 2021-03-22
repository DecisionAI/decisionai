import numpy as np

from helpers import run_sim, v
from sample_datasets import *
from decisionai.drilldown import DrilldownDefinition

def dd(agg_method, point=None, row_labels=(1, 3, 5), variables=('x',)):
    return DrilldownDefinition(
            agg_method, point, 
            'EXAMPLE_LABEL',
            row_labels,
            variables,
    )

def assert_drilldown_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for row_id, var_vals in actual.items():
        exp_vals = expected[row_id]
        assert var_vals.keys() == exp_vals.keys()
        for var, vals in var_vals.items():
            try:
                assert vals == exp_vals[var]
            except ValueError:
                assert (vals == exp_vals[var]).all()

def test_pointInTime_drilldown():
    db = ASSETS_DATASET_DEF.copy()
    db['variables'] = [
            {'short_name': 'x', 'equation': 'factor_4 * t'},
    ]
    endtime_drill = dd('pointInTime', 10)
    sim = run_sim(
            datasets=[db],
            num_steps=10,
            num_sims=2,
            drilldown=endtime_drill,
    )
    res = sim.drilldown()

    expected = {
            1: {'x': [10], },
            3: {'x': [30], },
            5: {'x': [50], },
    }
    assert res == expected

def test_mean_drilldown():
    db = ASSETS_DATASET_DEF.copy()
    db['variables'] = [
            {'short_name': 'x', 'equation': 'factor_4 * t'},
    ]
    mean_drill = dd('mean')
    sim = run_sim(
            datasets=[db],
            num_steps=10,
            num_sims=2,
            drilldown=mean_drill,
    )
    res = sim.drilldown()
    dc = sim.drilldown_computers[0]
    # row 1's factor_4 is equal to 1
    row1_x_vals = list(range(1, 11))
    row1_mean = np.mean(row1_x_vals)
    expected = {
            1: {'x': [row1_mean]},
            3: {'x': [row1_mean*3]},
            5: {'x': [row1_mean*5]},
    }
    assert res == expected

def test_total_drilldown():
    db = ASSETS_DATASET_DEF.copy()
    db['variables'] = [
            {'short_name': 'x', 'equation': 'factor_4 * t'},
    ]
    total_drill = dd('total')
    sim = run_sim(
            datasets=[db],
            num_steps=10,
            num_sims=2,
            drilldown=total_drill,
    )
    res = sim.drilldown()
    row1_x_vals = list(range(1, 11))
    row1_sum = sum(row1_x_vals)
    expected = {
            1: {'x': [row1_sum]},
            3: {'x': [row1_sum*3]},
            5: {'x': [row1_sum*5]},
    }
    assert res == expected

def test_multipolicy_pointInTime_drilldown():
    db = ASSETS_DATASET_DEF.copy()
    db['variables'] = [
            {'short_name': 'x', 'equation': '(factor_4 + attr) * t'},
    ]
    endtime_drill = dd('pointInTime', 10)
    sim = run_sim(
            policies=[
                {'policy_name': 'p1', 'attr': '0'},
                {'policy_name': 'p2', 'attr': '1'},
            ],
            datasets=[db],
            num_steps=10,
            num_sims=2,
            drilldown=endtime_drill,
    )
    res = sim.drilldown()

    expected = {
            1: {'x': [10, 20], },
            3: {'x': [30, 40], },
            5: {'x': [50, 60], },
    }
    assert_drilldown_equal(res, expected)

def test_multipolicy_pointInTime_drilldown():
    db = ASSETS_DATASET_DEF.copy()
    db['variables'] = [
            {'short_name': 'x', 'equation': '(factor_4 + attr) * t'},
    ]
    mean_drill = dd('mean')
    sim = run_sim(
            policies=[
                {'policy_name': 'p1', 'attr': '0'},
                {'policy_name': 'p2', 'attr': '1'},
            ],
            datasets=[db],
            num_steps=10,
            num_sims=2,
            drilldown=mean_drill,
    )
    res = sim.drilldown()
    # row 1's factor_4 is equal to 1
    row1_x_vals = list(range(1, 11))
    row1_mean = np.mean(row1_x_vals)
    expected = {
            1: {'x': [row1_mean, row1_mean*2]},
            3: {'x': [row1_mean*3, row1_mean*4]},
            5: {'x': [row1_mean*5, row1_mean*6]},
    }
    assert_drilldown_equal(res, expected)

def test_multivar_drilldown():
    db = ASSETS_DATASET_DEF.copy()
    db['variables'] = [
            {'short_name': 'x', 'equation': 'factor_4 * t'},
            {'short_name': 'y', 'equation': 'factor_4'},
    ]
    endtime_drill = dd('pointInTime', 10, variables=['x', 'y'])
    sim = run_sim(
            datasets=[db],
            num_steps=10,
            num_sims=2,
            drilldown=endtime_drill,
    )
    res = sim.drilldown()

    expected = {
            1: {'x': [10], 'y': [1], },
            3: {'x': [30], 'y': [3], },
            5: {'x': [50], 'y': [5], },
    }
    assert res == expected
