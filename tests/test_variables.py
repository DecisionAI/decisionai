from sample_datasets import *
from decisionai.variable import Variable
from decisionai.dataset import Dataset

def test_var_name():


    var = Variable(name="foo", formula="t + 1", initial=0)
    assert var.name == "foo"
    assert var.initial == 0

def test_dbvar_parsing():
    ds = WORKERS_DATASET_DEF
    ds._add_var(Variable(name="x", formula="t + 1", initial=0))
    
    assert ds.name == 'EXAMPLE_LABEL'
    dsvars = ds.added_vars
    assert len(dsvars) == 1
    xvar = dsvars[0]
    assert xvar.name == 'x'
    assert xvar.dotted_name == 'EXAMPLE_LABEL.x'
    assert xvar.initial == 0
