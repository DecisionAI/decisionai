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
    "drilldown": {
        "agg": "total", # TODO: add tests for other agg methods
        "datasetLabel": "ASSETS",
        "rowIndices": [0, 1],
        "variables": ["bar"],
    },
}

# TODO: at one point we had an issue with this endpoint where it was returning
# ndarrays as part of its response, which we were not able to serialize to json,
# causing 500 errors. Is there some way we can generally catch this problem of 
# returning non-serializable data? Possibly just have a unit test where we call
# the endpoint and try to call json.dumps on the response object?
def test_drilldown():
    payload = BASIC_PAYLOAD
    req = Mock(get_json=Mock(return_value=payload), args=payload, method="POST")
    response, code, headers = main.drilldown(req)
    def barsum(fact4, att):
        """Given values of the factor4 column and the att1 attribute, 
        return the sum of the "bar" variable over 10 timesteps.
        """
        s = 0
        for t in range(1, 11):
            s += t + att + fact4
        return s
    expected = {
            0: {
                "bar": [barsum(0, 1), barsum(0, 2)],
                },
            1: {
                "bar": [barsum(1, 1), barsum(1, 2)],
                },
            }
    assert code == 200
    assert isinstance(response, dict)
    assert len(response) == len(expected)
    assert response.keys() == expected.keys()
    for k, v in response.items():
        assert isinstance(v, dict)
        assert len(v) == 1
        inner_val = list(v.values())[0]
        assert isinstance(inner_val, list)
    assert response == expected
