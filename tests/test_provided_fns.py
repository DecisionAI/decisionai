"""Tests for custom functions/operators we provide for variable formulas.
join() is handled in a separate test module (test_joins.py)
"""
import pytest

from sample_datasets import *
from helpers import run_sim, pols
from copy import deepcopy
from decisionai.variable import Variable as v

def test_sum_fn():
    db = deepcopy(WORKERS_DATASET_DEF)
    sim = run_sim(
            [v('x', 'sum(EXAMPLE_LABEL.workers)')],
            dataset=db,
    )
    assert sim.deps == {'x': set()}
    # expected sum = 1 + 30
    sim.assert_values_match('x', [31, 31, 31, 31])

def test_sum_from_dataset_fn_deprecated_alias():
    # what we now call sum we used to call sum_from_dataset. We still want to
    # support that old name as an alias (for now)
    db = deepcopy(WORKERS_DATASET_DEF)
    sim = run_sim(
            [v('x', 'sum_from_dataset(EXAMPLE_LABEL.workers)')],
            dataset=db,
    )
    assert sim.deps == {'x': set()}
    # expected sum = 1 + 30
    sim.assert_values_match('x', [31, 31, 31, 31])

def test_sum_user_dataset_var():
    """Calling sum on a "DatasetAdditionVar" rather than a column
    in the original dataset."""
    db = deepcopy(WORKERS_DATASET_DEF)
    db._add_var(v('workertime', 'EXAMPLE_LABEL.workers * t'))
    sim = run_sim([
            v('x', 'sum(EXAMPLE_LABEL.workertime)'),
        ],
        dataset=db,
    )
    assert sim.deps == {
        'x': {'EXAMPLE_LABEL.workertime'},
        'EXAMPLE_LABEL.workertime': set(),
    }
    sim.assert_values_match('x', [0, 31, 62, 93])

def test_sum_constant_user_dataset_var():
    """Calling sum on a DatasetAdditionVar with a constant value
    (and also using multiple sims and policies)"""
    db = deepcopy(WORKERS_DATASET_DEF)
    db._add_var(v('seven', '7'))
    sim = run_sim([
            v('x', 'sum(EXAMPLE_LABEL.seven)'),
        ],
        policies=pols(attribute1=['1', '2']),
        dataset=db,
        num_sims=5,
    )
    x = sim.var_df.x
    assert (x == 14).all()
    assert len(x) == 40 # 5 sims * 2 policies * 4 timesteps

def test_sum_non_dataset_dimension_input():
    sim = run_sim(
        [
            v('x', 'sum(1)'),
            v('one', '1'),
            v('y', 'sum(one)'),
        ], dataset=WORKERS_DATASET_DEF,
        allow_errs=True,
    )
    sim.assert_errors_match([ ['x', 'equation',], ['y', 'equation',] ])

def test_sum_too_many_args():
    sim = run_sim(
        [
            v('x', 'sum(EXAMPLE_LABEL.workers, EXAMPLE_LABEL.max_production)'),
        ], dataset=WORKERS_DATASET_DEF,
        allow_errs=True,
    )
    sim.assert_errors_match([ ['x', 'equation', 'sum() takes 1 positional argument'] ])

def test_np_function_wrong_number_of_args():
    sim = run_sim(
            [v('x', '1'), v('y', '2'), v('z', 'ceil(x, y)'),],
            allow_errs=True,
    )
    sim.assert_errors_match([ ['z', 'equation', 'arguments'] ])

def test_multi_arg_max():
    sim = run_sim(
            [v('x', '1'), v('y', '2'), v('z', 't'),
                v('m', 'max(x, y, z)'),
            ],
            num_steps=4,
    )
    sim.assert_values_match(
            'm',
            [2, 2, 2, 3, 4],
    )

def test_binomial():
    sim = run_sim([
        v('x', 'binomial(100, .5)'),
        ],
    )
    values = sim.get_values('x')
    assert (0 <= values).all()
    assert (values <= 100).all()

def test_randomness_spread_over_dataset_rows():
    workers = deepcopy(MOREWORKERS_DATASET_DEF)
    workers._add_var(v('x', 'uniform(0, 100)'))
    sim = run_sim(dataset=workers)
    values = sim.get_values('MOREWORKERS.x', t=1)
    # We don't want to reuse randomness across rows, so ensure these values
    # aren't all equal (yes, there's an astronomically small chance of a false positive)
    assert values.min() != values.max()

