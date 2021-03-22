from sample_datasets import *
from decisionai.variables import Variable
from decisionai.datasets import Dataset

def test_var_parsing():
    vardef = dict(
            short_name="foo",
            equation="t + 1",
            initial="0",
    )
    var = Variable.from_json(vardef)
    assert var.name == "foo"
    assert var.initial == 0

def test_dbvar_parsing():
    dsdef = WORKERS_DATASET_DEF
    dsdef['variables'] = [
        dict(
            short_name="x",
            equation="t + 1",
            initial="0",
        ),
    ]

    ds = Dataset.from_json(dsdef)
    assert ds.name == 'EXAMPLE_LABEL'
    dsvars = ds.added_vars
    assert len(dsvars) == 1
    xvar = dsvars[0]
    assert xvar.name == 'x'
    assert xvar.dotted_name == 'EXAMPLE_LABEL.x'
    assert xvar.initial == 0
