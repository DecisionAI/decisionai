WORKERS_CSV = """name,workers,max_production
low,1,100
medium,30,200"""

WORKERS_DATASET_DEF = dict(
    label = "EXAMPLE_LABEL",
    csvLiteral = WORKERS_CSV,
    variables = [],
)

ASSETS_CSV = """asset_class,factor_1,factor_2,factor_3,factor_4
zero,0,0,0,0
low,1,10, 0,1
lowish,2,15,0,2
medium,3,20,0,3
highish,3,25,0,4
high,5,30,0,5
extreme,10,40,10,6"""
ASSETS_DATASET_DEF = dict(
    label = "EXAMPLE_LABEL",
    csvLiteral = ASSETS_CSV,
    variables = [],
)

CSV3 = "keyCol,someValue\n" + "\n".join(f'{n},{n*3}' for n in range(50))
CSV3_DATASET_DEF = dict(
    label = "TEST_FILE",
    csvLiteral = CSV3,
    variables = [],
)

INSURANCE_CSV = """id,initial_premium,initial_total_claims,location
0,223,0,Lima
1,196,0,Cusco
2,208,0,Lima
3,233,0,Cusco
4,225,0,Cusco"""
INSURANCE_DATASET_DEF = dict(
        label="INS",
        csvLiteral=INSURANCE_CSV,
        variables = [],
)

# A dataset with structure parallel to the workers sample dataset. Used for
# testing simple cross-dataset joins.
PARAWORKERS_CSV = """name,borkers,widgets
medium,30,20
low,1,10"""

PARAWORKERS_DATASET_DEF = dict(
    label = "PARAWORKERS",
    csvLiteral = PARAWORKERS_CSV,
    variables = [],
)
# Similar to workers, but having an extra (non-matching) row
MOREWORKERS_CSV = """name,borkers,widgets
medium,30,20
high,30,2
low,1,10"""

MOREWORKERS_DATASET_DEF = dict(
    label = "MOREWORKERS",
    csvLiteral = MOREWORKERS_CSV,
    variables = [],
)
