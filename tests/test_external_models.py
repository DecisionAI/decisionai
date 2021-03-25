import pandas as pd
import numpy as np
import os
import pytest

from helpers import run_sim, v, pols
from sample_datasets import *
from copy import deepcopy

PICKLE_DIR = 'sample_models'
def clf_factory(fname):
    dirname = os.path.dirname(__file__)
    pickle_path = os.path.join(dirname, PICKLE_DIR, fname)
    return [
        {"label": "CLF", "localPath": pickle_path},
    ]


def test_external_model_basic():
    # A regression model taking three inputs.
    model = {"label": "DEMAND_MODEL", "localPath": "tests/sample_models/three_input_model.pkl",}
    sim = run_sim(
            [
                v('competitor_price', 'uniform(100, 200)'),
                v('temperature', 'uniform(20, 30)'),
                v('demand', 'DEMAND_MODEL.predict(competitor_price, temperature, t)'),
            ],
            model=model,
    )
    df = sim.var_df
    assert df.demand.isnull().sum() == 0
    assert df.demand.std() > 0

def test_dummy_classifier():
    # Load a dummy binary classifier trained with strat=uniform.
    # Should always assign each class a probability of .5
    sim = run_sim([
        v('pred', 'CLF.predict(t)'),
        v('prob', 'CLF.predict_proba(t)'),
        ], models=clf_factory('dummy.pickle'),
    )
    df = sim.var_df
    # All predicted labels should be 0 or 1
    assert df.pred.isin([0, 1]).all()
    # All probabilities should be .5
    assert (df.prob == .5).all()

def test_dummy_classifier_multiple_policies():
    sim = run_sim([
        v('pred', 'CLF.predict(t)'),
        v('prob', 'CLF.predict_proba(t)'),
        ],
        policies=pols(x=['1', '2']),
        models=clf_factory('dummy.pickle'),
    )
    df = sim.var_df
    # All predicted labels should be 0 or 1
    assert df.pred.isin([0, 1]).all()
    # All probabilities should be .5
    assert (df.prob == .5).all()
    assert len(df) == 4 * 2 # 4 timesteps, 2 policies, 1 sim

def test_multiclass_classifier():
    vars = [
        {'short_name': 'prob', 'equation': 'CLF.predict_proba(t)'},
    ]
    # A dummy classifier with 3 classes. Not supported.
    sim = run_sim([
            v('prob', 'CLF.predict_proba(t)'),
        ],
        models=clf_factory('dummy_multiclass.pickle'),
        allow_errs=True,
    )
    sim.assert_errors_match([ ['CLF', '', 'binary'] ])
    sim.assert_null_values('prob')

def test_probabilistic_sampling():
    """Ensure that .predict for a classifier samples rather than returning the argmax.
    """
    vars = [
            v('x', 'uniform(0, 1)'),
            v('prob', 'CLF.predict_proba(x[t])'),
            v('pred', 'CLF.predict(x[t])'),
    ]
    sim = run_sim(vars, models=clf_factory('usually_positive_classifier.pickle'),
            num_steps=1000,
    )
    # We trained a logistic classifier on noise in (0, 1), and with 90% positive
    # labels. So all our predicted probabilities should be around .9, but we
    # should still almost always get at least one negative prediction.
    # (NB: There's a ~1e-46 of this test spuriously failing.)
    df = sim.var_df
    assert (df.prob > .5).all()
    assert 0 in df.pred
    assert 1 in df.pred

def test_external_model_dataset_variable():
    db = deepcopy(WORKERS_DATASET_DEF)
    db._add_var(v('pred', 'CLF.predict(EXAMPLE_LABEL.max_production, 25, t)'))
    models = clf_factory('dummy.pickle')
    sim = run_sim(models=models, dataset=db)
    df = sim.dataset_df()
    assert len(df) == 3*2 # 3 timesteps * 2 rows
    assert df.pred.isin([0, 1]).all()

def test_bad_predict_input_size():
    sim = run_sim(
            [v('pred', 'CLF.predict(t, t+1)')],
            models=clf_factory('three_input_model.pkl'),
            allow_errs=True,
    )
    sim.assert_errors_match([ ['pred', 'equation', 'features',] ])

def test_tf_regression_model():
    # a tf.keras.Sequential model trained on input noise of shape (*, 3) and
    # random labels in the range (0, 1). 
    sim = run_sim(
            [v('pred', 'CLF.predict(t, t+1, 0.5)'),],
            models=clf_factory('tf_three_input_model.h5'),
    )
    preds = sim.var_df.pred
    assert not preds.isnull().any()
    assert preds.dtype.kind == 'f' # float

def test_tf_regression_model_wrong_input_size():
    sim = run_sim(
            [v('pred', 'CLF.predict(t)'),],
            models=clf_factory('tf_three_input_model.h5'),
            allow_errs=True,
    )
    sim.assert_errors_match([ ['pred', 'equation', 'features',] ])

def test_malformed_pickle():
    sim = run_sim([],
            models=clf_factory('not_really_a_pickle.pickle'),
            allow_errs=True,
    )
    sim.assert_errors_match([ ['CLF', '', 'Unpickling CLF model file triggered the following exception: "invalid load key'] ])

def test_sklearn_pipeline():
    """insurance_churn_model.pkl is an sklearn pipeline, with a classifier on
    the tail end. It takes as input [number, number, number, number, string].
    """
    ds = deepcopy(INSURANCE_DATASET_DEF)
    ds._add_var(v('churn', 'CLF.predict(t, t, initial_premium, initial_total_claims, location)'))

    sim = run_sim([],
            models=clf_factory('insurance_churn_model.pkl'),
            dataset=ds,
    )
    df = sim.dataset_df()
    assert df.churn.isin([0, 1]).all()

def test_model_predict_dataset_deps():
    ds = deepcopy(INSURANCE_DATASET_DEF)
    ds._add_var(v('foo', '10'))
    ds._add_var(v('churn', 'CLF.predict(t, t, foo, initial_total_claims, location)'))
    sim = run_sim([],
            models=clf_factory('insurance_churn_model.pkl'),
            dataset=ds,
    )
    df = sim.dataset_df()
    assert df.churn.isin([0, 1]).all()
    assert sim.deps == {
            'INS.churn': {'INS.foo'},
            'INS.foo': set(),
    }
