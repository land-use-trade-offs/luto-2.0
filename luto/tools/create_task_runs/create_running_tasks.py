import numpy as np
from luto.tools.create_task_runs.helpers import create_grid_search_template, create_task_runs, create_settings_template



# Define the grid search
''' Each value in the grid search has to be list iterable.'''

grid_search = {
    # Computational settings, which are not relevant to LUTO itself
    'MEM': ['40GB'],
    'NCPUS':[10],
    'TIME': ['0:30:00'],

    # LUTO settings to be grid searched
    'MODE': [
        'snapshot', 
        # 'timeseries'
    ],
    'SOLVE_ECONOMY_WEIGHT': 
        np.linspace(1e-6, 0.1, 100),
    'GHG_LIMITS_FIELD': [
        '1.5C (67%) excl. avoided emis', 
        # '1.5C (50%) excl. avoided emis', 
        # '1.8C (67%) excl. avoided emis'
    ],
    'BIODIV_GBF_TARGET_2_DICT': [
        {2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3 }, 
        # {2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5 }
    ]
}


# Read the template for the custom settings
template_df = create_settings_template()
grid_search_df = create_grid_search_template(template_df, grid_search)

# Create the task runs
create_task_runs(grid_search_df)