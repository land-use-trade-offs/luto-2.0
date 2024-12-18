import os
import numpy as np
from luto.tools.create_task_runs.helpers import create_grid_search_template, create_task_runs, create_settings_template


grid_search = {
    ###############################################################
    # Computational settings for submitting the job to the cluster
    ###############################################################
    'MEM': ['40GB'],
    'NCPUS':[10],
    'TIME': ['5:30:00'],
    
    ###############################################################
    # Working settings for the model run
    ###############################################################
    'MODE': ['timeseries'],                # 'snapshot' or 'timeseries'
    'RESFACTOR': [10],
    'WRITE_THREADS': [10],
    'WRITE_OUTPUT_GEOTIFFS': [False],
    
    ###############################################################
    # Model run settings
    ###############################################################
    'GHG_CONSTRAINT_TYPE': ['hard'],       # 'hard' or 'soft'

    ###############################################################
    # Scenario settings for the model run
    ###############################################################
    'SOLVE_ECONOMY_WEIGHT': 
        # list(np.linspace(0.1, 0.12, 10)) + \
        # [0.20, 0.25, 0.30],
        [
            round(10**(-i) * (1 - j/10))
            for i in range(3)    
            for j in range(9)     
        ],
    'GHG_LIMITS_FIELD': [
        '1.5C (67%) excl. avoided emis', 
        # '1.5C (50%) excl. avoided emis', 
        # '1.8C (67%) excl. avoided emis'
    ],
    'BIODIV_GBF_TARGET_2_DICT': [
        {2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3 }, 
        # {2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5 }
    ],
    # 'DIET_GLOB': ['BAU', 'FLX', 'VEG', 'VGN'],
}


# Read the template for the custom settings
template_df = create_settings_template()
grid_search_df = create_grid_search_template(template_df, grid_search)

# Create the task runs
if os.name == 'posix':
    create_task_runs(grid_search_df)
elif os.name == 'nt':
    create_task_runs(grid_search_df, python_path='F:/jinzhu/conda_env/luto/python.exe', n_workers=10)


