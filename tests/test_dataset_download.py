from helpers import run_sim, v, pols

# Identical to test with same name in test_sim.py, except this one exercises the
# download_file code path, rather than mocking up the csv with a string literal
def test_with_dataset_formulas_accessing_raw_data():
    sample_db_entry = [
        {
            "label": "EXAMPLE_LABEL",
            "objectPath": ["userFiles/TEST_uDt3x6DAX7Wkto3aFOxk_1589651148084.csv"],
            "uri": [
                "gs://dai-sim-app.appspot.com/userFiles/TEST_uDt3x6DAX7Wkto3aFOxk_1589651148084.csv"
            ],
            "variables": [v('t_again', 't')],
        }
    ]
    sim = run_sim(
            policies=pols(price_per_unit=['10', '1']),
            datasets=sample_db_entry,
    )
    sim.assert_values_match(
            'EXAMPLE_LABEL.t_again',
            [1, 1, 2, 2, 3, 3,],
            policy=0,
    )
