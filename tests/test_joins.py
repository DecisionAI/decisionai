import pytest

from sample_datasets import *
from helpers import run_sim, pols
from decisionai.variable import Variable as v

from copy import deepcopy

# TODO: All through this file we're reusing some mocks, copying and mutating
# that's unlike actual usage patterns. Fix that.

# Used when we want to test a sim with multiple policies, but we don't really
# care about the contents of their attributes.
DUMMY_POLS = pols(unused_attr=['0', '5'])

def test_join_fn_simple():
    sim = run_sim([
        v('a_var', '3'),
        v('b_var', "join(EXAMPLE_LABEL.factor_4 == a_var, EXAMPLE_LABEL.factor_2)"),
        ],
        dataset=ASSETS_DATASET_DEF,
        num_sims=2,
    )
    sim.assert_values_match(
            'b_var',
            [20, 20, 20, 20],
            sim_id=1,
    )

def test_join_fn_many_matching_rows():
    """Verify that we raise an error if more than one row matches the join
    condition and the join is not wrapped by an aggregator.
    """
    # This join is 1:1 for timesteps 0-2, but there are two rows with factor_1 == 3.
    # The first has factor_2=20, the other has factor_2=25.
    sim = run_sim([
        v('a_var', 't'),
        v('b_var', "join(EXAMPLE_LABEL.factor_1 == a_var, EXAMPLE_LABEL.factor_2)"),
        ],
        dataset=ASSETS_DATASET_DEF,
        allow_errs=True,
    )
    sim.assert_errors_match([ ['b_var', 'equation', 'Join'] ])

def test_sum_join_with_multimatch():
    sim = run_sim([
        v('b_var', "sum(join(EXAMPLE_LABEL.factor_1 == t, EXAMPLE_LABEL.factor_2))"),
        ],
        dataset=ASSETS_DATASET_DEF,
        num_steps=3,
    )
    sim.assert_values_match(
            'b_var',
            [0, 10, 15, 20+25],
    )

def test_join_fn_no_matching_rows():
    """Verify that we raise an error if no rows match the join condition and the
    join is not wrapped in an aggregator.
    """
    # We have a matching value (of factor_2=30) for a_var=5 (when t=1), but not for any other
    # timesteps
    sim = run_sim([
        v('a_var', 't+4'),
        v('b_var', "join(EXAMPLE_LABEL.factor_1 == a_var, EXAMPLE_LABEL.factor_2)"),
        ],
        dataset=ASSETS_DATASET_DEF,
        allow_errs=True,
        num_steps=3,
    )
    sim.assert_errors_match([ ['b_var', 'equation', 'Join'] ])

def test_join_fn_multiple_policies():
    sim = run_sim([
        v('a_var', '3'),
        v('b_var', "join(EXAMPLE_LABEL.factor_4 == a_var, EXAMPLE_LABEL.factor_2)"),
        ],
        policies=pols(attribute1=['1', '3']),
        dataset=ASSETS_DATASET_DEF,
        num_sims=2,
    )
    assert (sim.results.b_var == 20).all()

def test_join_fn_reduced_dim_boolean_expr():
    """Verify that joins work even when the mask (the first arg) doesn't naturally
    have the shape of a dataset variable.
    """
    # This is another overmatching join, but it has the unusual feature that
    # the first arg does not have dataset dimensionality. Want to make sure
    # we just raise a user-visible join exception, rather than a hard crash.
    sim = run_sim([
        v('a_var', 't+4'),
        v('b_var', "join(1==1, EXAMPLE_LABEL.factor_2)"),
        ],
        dataset=ASSETS_DATASET_DEF,
        num_sims=2,
        num_steps=3,
        allow_errs=True,
    )
    sim.assert_errors_match([ ['b_var', 'equation', 'Join'] ])

def test_empty_sumjoin():
    sim = run_sim([
        v('b_var', "sum(join(1==0, EXAMPLE_LABEL.factor_2))"),
        ],
        dataset=ASSETS_DATASET_DEF,
        num_sims=2,
        num_steps=3,
    )
    sim.assert_values_match(
            'b_var',
            [0, 0, 0, 0],
            sim_id=1,
    )

def test_join_fn_policy_dependent_mismatch():
    # For policy 1, this will give a valid 1:1 join. But for policy 2, it will
    # overmatch (due to there being 2 rows having factor_1=3)
    sim = run_sim([
        v('b_var', "join(EXAMPLE_LABEL.factor_1 == att1, EXAMPLE_LABEL.factor_2)"),
        ],
        policies=pols(
            att1=["1", "3"],
        ),
        dataset=ASSETS_DATASET_DEF,
        allow_errs=True,
    )
    sim.assert_errors_match([ ['b_var', 'equation', 'Join'] ])

def test_join_fn_indexing_matters():
    sim = run_sim(
            [v('b', 'join(t==TEST_FILE.keyCol, TEST_FILE.someValue)')],
            dataset=CSV3_DATASET_DEF,
            num_steps=5,
            num_sims=2,
    )
    df = sim.results
    assert (df.loc[df.t == 1, 'b'] == 3).all()
    assert (df.loc[df.t == 5, 'b'] == 15).all()

def test_join_multiple_datasets_bijection():
    """A join involving two datasets which happen to have a perfect 1-to-1 correspondence.
    """
    workers = deepcopy(WORKERS_DATASET_DEF)
    workers.name = 'WORKERS'
    para = PARAWORKERS_DATASET_DEF
    workers._add_var(v('joined', 'join(name == PARAWORKERS.name, PARAWORKERS.widgets)'))

    sim = run_sim(
            [],
            datasets=[para, workers],
    )
    sim.assert_values_match(
            'WORKERS.joined',
            [10, 20],
            t=1,
    )

def test_join_multiple_datasets_injection():
    """Similar to above scenario, except the dataset we're joining to has an
    extra row which doesn't participate in the join results
    """
    workers = deepcopy(WORKERS_DATASET_DEF)
    workers.name = 'WORKERS'
    more = deepcopy(MOREWORKERS_DATASET_DEF)
    workers._add_var(v('joined', 'join(name == MOREWORKERS.name, MOREWORKERS.widgets)'))
    
    sim = run_sim(
            [],
            datasets=[more, workers],
    )
    sim.assert_values_match(
            'WORKERS.joined',
            [10, 20],
            t=1,
    )

def test_sum_join_multiple_datasets_bijection():
    workers = deepcopy(WORKERS_DATASET_DEF)
    workers.name = 'WORKERS'
    para = deepcopy(PARAWORKERS_DATASET_DEF)
    para.name = 'PARAWORKERS'
    workers._add_var(v('joined', 'sum(join(name == PARAWORKERS.name, PARAWORKERS.widgets))'))
            
    sim = run_sim(
            [],
            datasets=[para, workers],
    )
    # Because this join is 1-1 the sum() should be a no-op 
    sim.assert_values_match(
            'WORKERS.joined',
            [10, 20],
            t=1,
    )

def test_sum_join_multiple_datasets_undermatching():
    workers = deepcopy(WORKERS_DATASET_DEF)
    workers.name = 'WORKERS'
    more = deepcopy(MOREWORKERS_DATASET_DEF)
    more._add_var(v('joined', 'sum(join(name == WORKERS.name, WORKERS.max_production))'))
            
    sim = run_sim(
            [],
            datasets=[more, workers],
    )
    # Second row of MOREWORKERS doesn't match any rows in WORKERS, hence sum is 0
    sim.assert_values_match(
            'MOREWORKERS.joined',
            [200, 0, 100],
            t=1,
    )

def test_sum_join_multiple_datasets_overmatching():
    workers = deepcopy(WORKERS_DATASET_DEF)
    workers.name = 'WORKERS'
    more = deepcopy(MOREWORKERS_DATASET_DEF)
    workers._add_var(
            v('joined', 'sum(join(workers == MOREWORKERS.borkers, MOREWORKERS.widgets))')
    )
    sim = run_sim(
            [],
            datasets=[more, workers],
            num_sims=2,
    )
    # Second row of WORKERS matches two rows of MOREWORKERS having values of 20 and 2
    sim.assert_values_match(
            'WORKERS.joined',
            [10, 20+2],
            t=1,
    )

def test_sum_join_multiple_datasets_undermatching_multiple_policies():
    """NB: because dimensions of size 1 get a uniquely lax treatment under
    numpy broadcasting rules, it's important to include some tests with >1
    policy/sim, even if the quantity being tested has no dependence on policy
    attributes or inter-simulation randomness.
    """
    workers = deepcopy(WORKERS_DATASET_DEF)
    workers.name = 'WORKERS'
    more = deepcopy(MOREWORKERS_DATASET_DEF)
    more._add_var(
            v('joined', 'sum(join(name == WORKERS.name, WORKERS.max_production))')
    )
    sim = run_sim(
            [],
            policies=DUMMY_POLS,
            datasets=[more, workers],
            num_sims=5,
    )
    # Second row of MOREWORKERS doesn't match any rows in WORKERS, hence sum is 0
    sim.assert_values_match(
            'MOREWORKERS.joined',
            [200, 0, 100],
            t=1,
            policy=0,
    )

def test_sum_join_composite_mask_with_different_shapes():
    workers = deepcopy(WORKERS_DATASET_DEF)
    workers.name = 'WORKERS'
    more = deepcopy(MOREWORKERS_DATASET_DEF)
    workers._add_var(v(
                'joined',
                'sum(join(workers == MOREWORKERS.borkers and MOREWORKERS.widgets==t, MOREWORKERS.borkers))'
            ))
            
    sim = run_sim(
            [],
            datasets=[more, workers],
            num_sims=2,
    )
    # We only get any matches when time is 2
    sim.assert_values_match(
            'WORKERS.joined',
            [0, 0],
            t=1,
    )
    sim.assert_values_match(
            'WORKERS.joined',
            [0, 30],
            t=2,
    )
