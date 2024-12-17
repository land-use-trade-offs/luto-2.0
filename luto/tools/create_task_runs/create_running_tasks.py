import os
import numpy as np
from luto.tools.create_task_runs.helpers import create_grid_search_template, create_task_runs, create_settings_template


grid_search = {
    ###############################################################
    # Computational settings for submitting the job to the cluster
    ###############################################################
    'MEM': ['40GB'],
    'NCPUS':[10],
    'TIME': ['0:30:00'],
    
    ###############################################################
    # Working settings for the model run
    ###############################################################
    'MODE': ['timeseries'],                 # 'snapshot' or 'timeseries'
    'RESFACTOR': [10],
    'WRITE_THREADS': [10],
    'WRITE_OUTPUT_GEOTIFFS': [False],
    
    ###############################################################
    # Background settings for the model run
    ###############################################################
    'CO2_FERT': ['off'],
    'INCLUDE_WATER_LICENSE_COSTS': [0],     # 0 [off] or 1 [on]
    'WATER_STRESS': [0.4],
    'AG_SHARE_OF_WATER_USE': [0.7],

    ###############################################################
    # Scenario settings for the model run
    ###############################################################
    'SOLVE_ECONOMY_WEIGHT': 
        list(np.linspace(0.1, 0.12, 10)) + \
        [0.001, 0.01, 0.05, 0.08, 0.2, 0.3, 0.4, 0.5,  0.7, 0.9],
        # [
        #     10**(-i) * (1 - j/10) 
        #     for i in range(3)     # The range of the exponent: 1, 0.1, 0.01, ...
        #     for j in range(9)     # The range of the decimal: 1, 0.9, 0.8, ...
        # ],
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


