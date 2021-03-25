import numpy as np
import pandas as pd
import pytest

from copy import deepcopy
from sample_datasets import *
from helpers import run_sim, pols
from decisionai.variable import Variable as v


def test_fibonacci():
    sim = run_sim([v('a', 'a[t-1] + a[t-2]', initial=1)], num_steps=5)
    sim.assert_values_match('a', [1, 2, 3, 5, 8, 13])

def test_two_vars():
    sim = run_sim([
        v('a', 'a[t-1] + a[t-2]', '1'),
        v('b', 'a + 1'),
    ])
    sim.assert_values_match('a', [1, 2, 3, 5])
    sim.assert_values_match('b', [2, 3, 4, 6])

def test_constant_index_expr():
    sim = run_sim([
        v('x', 't', '0'),
        v('y', 'x[0]')
    ])
    sim.assert_values_match('y', [0, 0, 0, 0])

def test_arithmetic_index_expr():
    sim = run_sim([
        v('x', 't', '0'),
        v('y', 'x[5-5]'),
    ])
    sim.assert_values_match('y', [0, 0, 0, 0])

def test_sim_with_no_initial_vals():
    sim = run_sim([
        v('a', 't'),
        v('b', '2*a'),
    ], num_steps=2)
    sim.assert_values_match('a', [0, 1, 2])
    sim.assert_values_match('b', [0, 2, 4])
    assert sim.deps == {'a': set(), 'b': {'a'}}

def test_with_numeric_policies():
    sim = run_sim(
            [v('a', 't*x + 1')],
            policies=pols(x=['100', '1000']),
            num_steps=2,
    )
    sim.assert_values_match('a', [1, 101, 201], policy=0)
    sim.assert_values_match('a', [1, 1001, 2001], policy=1)

def test_with_numeric_policies_and_two_sims():
    # As above, but with 2 sims
    sim = run_sim(
            [v('a', 't * x + 1')],
            policies=pols(x=['100', '1000']),
            num_steps=2,
            num_sims=2,
    )
    for sim_id in range(2):
        # Values should be identical across simulations.
        sim.assert_values_match('a', [1, 101, 201], policy=0, sim_id=sim_id)
        sim.assert_values_match('a', [1, 1001, 2001], policy=1, sim_id=sim_id)


def test_with_policy_formulas():
    sim = run_sim([
        v('a', 'where(t>1, x[t-1], 1)', '1'),
        v('b', 'x[t-1]', '0'),
    ], policies=pols(x=['a*2', 'a*3']),
    )
    sim.assert_values_match('a', [1, 1, 2, 4], policy=0)
    sim.assert_values_match('a', [1, 1, 3, 9], policy=1)
    sim.assert_values_match('b', [0, 2, 2, 4], policy=0)
    sim.assert_values_match('b', [0, 3, 3, 9], policy=1)

def test_basic_dataset_addition_variable():
    db = deepcopy(WORKERS_DATASET_DEF)
    db._add_var(v('rev_per_worker', "EXAMPLE_LABEL.max_production / EXAMPLE_LABEL.workers"))
    sim = run_sim(datasets=[db])
    # At every timestep, these should be our 2 values, in order of row index.
    single_timestep_val = [100/1, 200/30]
    sim.assert_values_match_across_time(
            'EXAMPLE_LABEL.rev_per_worker',
            single_timestep_val,
    )

def test_with_dataset_formulas_accessing_raw_data():
    db = deepcopy(WORKERS_DATASET_DEF)
    db._add_var(v('t_again', 't'))
            
    sim = run_sim(
            policies=pols(price_per_unit=['10', '1']),
            dataset=db,
    )
    sim.assert_values_match(
            'EXAMPLE_LABEL.t_again',
            [1, 1, 2, 2, 3, 3,],
            policy=0,
    )

def test_with_dataset_formulas_accessing_attributes():
    db = deepcopy(WORKERS_DATASET_DEF)
    db._add_var(v('rev_per_worker', "EXAMPLE_LABEL.max_production / EXAMPLE_LABEL.workers * price_per_unit"))
            
    sim = run_sim([], 
            policies=pols(price_per_unit=['10', '5']),
            dataset=db,
            num_sims=4,
    )
    sim.assert_values_match_across_time(
            'EXAMPLE_LABEL.rev_per_worker',
            [1000, 200/3],
            policy=0,
    )
    sim.assert_values_match_across_time(
            'EXAMPLE_LABEL.rev_per_worker',
            [500, 100/3],
            policy=1,
    )

def test_with_dataset_formulas_accessing_raw_data_and_attributes_and_vars():
    db = deepcopy(WORKERS_DATASET_DEF)
    db._add_var(v('rev_per_worker', 
                "EXAMPLE_LABEL.max_production / EXAMPLE_LABEL.workers * price_per_unit * a_var",
            ))
    sim = run_sim(
            [v('a_var', 't/10')],
            policies=pols(price_per_unit=['10', '5']),
            dataset=db,
            num_sims=4,
    )
    sim.assert_values_match(
            'EXAMPLE_LABEL.rev_per_worker',
            [50, 10/3],
            t=1, policy=1,
    )

def test_time_indexing_dataset_vars():
    db = deepcopy(ASSETS_DATASET_DEF)
    db._add_var(v('ds_var1', 'EXAMPLE_LABEL.factor_1'))
    db._add_var(v('ds_var2', 'ds_var1[t-1]', '0'))

    sim = run_sim(
            [v('lagged_var', 'sum(EXAMPLE_LABEL.ds_var1[t-1] + EXAMPLE_LABEL.ds_var2)', '0')],
            policies=pols(attribute1=['1']),
            dataset=db,
    )
    sim.assert_values_match(
            'lagged_var',
            [0, 48, 48, 48],
    )

def test_broadcasting_dataset_var_in_policy_expression():
    sim = run_sim(
            [v('x', 'attr')],
            policies=pols(attr=['EXAMPLE_LABEL.workers == 1', '5']),
            dataset=WORKERS_DATASET_DEF,
            allow_errs=True,
    )
    sim.assert_errors_match([ ['attr', 'equation', 'extra dimension'] ])

def test_multiline_formula():
    sim = run_sim(
            [v('x', '1 +\n1')],
    )
    sim.assert_values_match('x', 2) # broadcasting comparison - should be 2 everywhere
