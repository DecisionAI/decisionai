# hack to import main.py
import sys
sys.path.append('../')
import main

import json
import numpy as np
import pandas as pd
from unittest.mock import Mock

# https://app.decision.ai/model/v3InYurvEsu69r5BEXGJ
BASIC_PAYLOAD = {
    "variables":[
        {"equation":"t + att1","short_name":"x"},
    ],
    "policies":[
        {"att1":"1","policy_name":"pol1"},
        {"policy_name":"pol2","att1":"2"},
    ],
    "datasets":[
        {"label":"ASSETS",
        "objectPath":["userFiles/ASSETS_LqjqBAUwctSfoaC30GDs_1600718088595.csv"],
        "uri":["gs://dai-sim-app.appspot.com/userFiles/ASSETS_LqjqBAUwctSfoaC30GDs_1600718088595.csv"],
        "variables":[
            {"equation":"factor_4","short_name":"foo"},
            {"equation":"factor_4 + x","short_name":"bar"},
        ],
        }
    ],
    "externalModels":[],
    "numPeriods":10,
    "numSims":50,
    "numPreviousSims":0,
}

def test_main():
    payload = BASIC_PAYLOAD
    req = Mock(get_json=Mock(return_value=payload), args=payload, method="POST")
    response, code, headers = main.main(req)
    assert response['mainErrors'] == {}
    assert response['policyErrors'] == {}
    assert response['datasetErrors'] == {}

    assert response['deps'] == {
            'x': ['att1'],
            'ASSETS.bar': ['x'],
            'ASSETS.foo': [],
            'att1': [],
    }
    var_results = json.loads(response['mainSimResults'])
    cols = ['x', 't', 'simId', 'policyId']
    assert var_results['columns'] == cols
    dat = var_results['data']
    arr = np.array(dat)
    # 50 simulations, 10+1 timesteps, 2 policies
    nrows = 50 * 11 * 2
    assert arr.shape == (nrows, len(cols))
    assert (arr[0] == [1, 0, 0, 0]).all()

    # Dataset variable data limited to:
    # 1. a single simulation
    # 2. only the first 5 timesteps
    excerpt = json.loads(response['datasetVarValuesExcerpt'])
    assert excerpt.keys() == {'ASSETS'}
    df = pd.read_json(excerpt['ASSETS'], orient='split')
    ex_cols = ['foo', 'bar', 'policyId', 't', 'rowId']
    assert set(df.columns) == set(ex_cols)
    nrows = 7 # n rows in the dataset
    # 1 simulation, 5 timesteps, 2 policies, 7 rows
    assert len(df) == 5 * 2 * 7
    sample_foo = df[
            (df.policyId==0)
            & (df.t == 1)
    ].sort_values(by='rowId', ascending=True)['foo']
    factor_4 = [0, 1, 2, 3, 4, 5, 6]
    assert (sample_foo.values == factor_4).all()
    assert not df.policyId.isnull().any(), "Got {} null policy Ids".format(
            df.policyId.isnull().sum()
    )
