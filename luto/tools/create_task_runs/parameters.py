TASK_ROOT_DIR = '../Custom_runs'


# The path to the script that import settings as variables
SETTING_IN_USE = [
    'luto/data.py'
    'luto/dataprep.py'
    'luto/simulation.py'
    'luto/economics/agricultural/water.pyluto/economics/non_agricultural/biodiversity.py'
    'luto/economics/non_agricultural/cost.py'
    'luto/economics/non_agricultural/revenue.py'
    'luto/economics/off_land_commodity/__init__.py'
    'luto/solvers/solver.py'
    'luto/tools/__init__.py'
    'luto/tools/compmap.py'
    'luto/tools/plotmap.py'
    'luto/tools/spatializers.py'
    'luto/tools/report/create_report_data.py'
    'luto/tools/report/create_static_maps.py'
    'luto/tools/report/write_input_data/array2tif.py'
    ]



PARAMS_TO_EVAL = [
    'AMORTISE_UPFRONT_COSTS',
    'WRITE_OUTPUT_GEOTIFFS',
    'PARALLEL_WRITE',
    'OFF_LAND_COMMODITIES',
    'GHG_LIMITS',
    'BIODIV_GBF_TARGET_2_DICT',
    'AG_MANAGEMENTS',
    'AG_MANAGEMENTS_REVERSIBLE',
    'NON_AG_LAND_USES',
    'NON_AG_LAND_USES_REVERSIBLE'

]

EXCLUDE_DIRS = ['input', 'output', '.git', '.vscode', '__pycache__', 'jinzhu_inspect_code']