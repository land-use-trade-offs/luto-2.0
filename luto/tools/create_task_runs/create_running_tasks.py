import os
import numpy as np
from luto.tools.create_task_runs.helpers import create_grid_search_template, create_task_runs, create_settings_template


grid_search = {
    ###############################################################
    # Task run settings for submitting the job to the cluster
    ###############################################################
    'MEM': ['40GB'],
    'NCPUS':[10],
    'TIME': ['5:00:00'],
    'QUEUE': ['normalsr'],
    
    ###############################################################
    # Working settings for the model run
    ###############################################################
    'MODE': ['timeseries'],                # 'snapshot' or 'timeseries'
    'RESFACTOR': [10],
    'WRITE_THREADS': [10],
    'WRITE_OUTPUT_GEOTIFFS': [True],
    
    ###############################################################
    # Model run settings
    ###############################################################
    'DEMAND_CONSTRAINT_TYPE': ['soft'],     # 'hard' or 'soft'
    'GHG_CONSTRAINT_TYPE': ['soft'],        # 'hard' or 'soft'
    'BIODIVERSITY_LIMITS': ['on'],           # 'on' or 'off'
    
    ###############################################################
    # Scenario settings for the model run
    ###############################################################
    'SOLVE_ECONOMY_WEIGHT': 
        # list(np.arange(5, 51, 2)/100),
        [0.25],
        # sorted([
        #     round(10**(-i) * (j/20), 10)
        #     for i in range(4)
        #     for j in range(1,19)
        # ]),
    'GHG_LIMITS_FIELD': [
        '1.5C (67%) excl. avoided emis', 
        # '1.5C (50%) excl. avoided emis', 
        # '1.8C (67%) excl. avoided emis'
    ],
    'BIODIV_GBF_TARGET_2_DICT': [
        {2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3 }, 
        # {2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5 }
    ],
    # 'DIET_DOM': ['BAU',],            # 'BAU' or 'FLX'
    # 'DIET_GLOB': ['BAU',],           # 'BAU' or 'FLX'
    # 'WASTE': [1],                    # 1 or 0.5
    # 'FEED_EFFICIENCY': ['BAU'],      # 'BAU' or 'High'
}


# Read the template for the custom settings
template_df = create_settings_template()
grid_search_df = create_grid_search_template(template_df, grid_search)


# Create the task runs
if os.name == 'posix':
    create_task_runs(grid_search_df)
elif os.name == 'nt':
    create_task_runs(grid_search_df, python_path='F:/jinzhu/conda_env/luto/python.exe', n_workers=10)


