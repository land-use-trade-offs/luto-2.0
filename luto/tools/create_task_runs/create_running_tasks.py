# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.

import os
import numpy as np
from luto.tools.create_task_runs.helpers import (
    create_grid_search_parameters, 
    create_grid_search_template, 
    create_task_runs
)

# Set the grid search parameters
grid_search = {
    ###############################################################
    # Task run settings for submitting the job to the cluster
    ###############################################################
    'MEM': ['24GB'],
    'NCPUS':[6],
    'TIME': ['2:00:00'],
    'QUEUE': ['normalsr'],
    
 
    ###############################################################
    # Working settings for the model run
    ###############################################################
    'OBJECTIVE': ['maxprofit'],             # 'maxprofit' or 'maxutility'
    'MODE': ['timeseries'],                 # 'snapshot' or 'timeseries'
    'RESFACTOR': [15],
    'SIM_YERAS': [list(range(2020, 2051))],   # Years to run the model 
    'WRITE_THREADS': [5],
    'WRITE_OUTPUT_GEOTIFFS': [False],
    'KEEP_OUTPUTS': [False],                # If false, only keep report HTML
    
 
    ###############################################################
    # Model run settings
    ###############################################################
    
    # --------------- Demand settings ---------------
    'DEMAND_CONSTRAINT_TYPE': ['hard'],     # 'hard' or 'soft'    
    
    # --------------- GHG settings ---------------
    'GHG_CONSTRAINT_TYPE': ['hard'],        # 'hard' or 'soft'
    'GHG_LIMITS_FIELD': [
        '1.5C (67%) excl. avoided emis', 
        '1.5C (50%) excl. avoided emis', 
        '1.8C (67%) excl. avoided emis'
    ],
    
    # --------------- Water constraints ---------------
    'WATER_CONSTRAINT_TYPE': ['hard'],        # 'hard' or 'soft'
    'WATER_PENALTY': [1],
    'INCLUDE_WATER_LICENSE_COSTS': [0],
    
    # --------------- Biodiversity priority zone ---------------
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50],
    
    # --------------- Biodiversity settings - GBF 2 ---------------
    'BIODIVERSTIY_TARGET_GBF_2': ['on'],    # 'on' or 'off'
    'BIODIV_GBF_TARGET_2_DICT': [
        {2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5}, 
        {2010: 0, 2030: 0.5, 2050: 0.5, 2100: 0.5},
    ],

    # --------------- Biodiversity settings - GBF 3 ---------------
    'BIODIVERSTIY_TARGET_GBF_3': ['off'],   # 'on' or 'off'
    
    # --------------- Biodiversity settings - GBF 4 ---------------
    'BIODIVERSTIY_TARGET_GBF_4_SNES' : ['off'],         # 'on' or 'off'.
    'BIODIVERSTIY_TARGET_GBF_4_ECNES' : ['off'],           # 'on' or 'off'.
    
    # --------------- Biodiversity settings - GBF 8 ---------------
    'BIODIVERSTIY_TARGET_GBF_8': ['off'],   # 'on' or 'off'

 
    ###############################################################
    # Scenario settings for the model run
    ###############################################################
    'SOLVE_BIODIV_PRIORITY_WEIGHT': (
        (np.arange(0,90,20)/100).tolist() + 
        (np.arange(900,1000,20)/1000).tolist()
    ), 
    'SOLVE_WEIGHT_ALPHA': (
        (np.arange(0,10,1)/100).tolist() + 
        (np.arange(10,100,20)/100).tolist() 
    ),
    
    #-------------------- Diet BAU --------------------
    'DIET_DOM': ['BAU',],            # 'BAU' or 'FLX'
    'DIET_GLOB': ['BAU',],           # 'BAU' or 'FLX'
    'WASTE': [1],                    # 1 or 0.5
    'FEED_EFFICIENCY': ['BAU'],      # 'BAU' or 'High'
    #---------------------Diet FLX --------------------
    # 'DIET_DOM': ['FLX',],            # 'BAU' or 'FLX'
    # 'DIET_GLOB': ['FLX',],           # 'BAU' or 'FLX'
    # 'WASTE': [0.5],                    # 1 or 0.5
    # 'FEED_EFFICIENCY': ['High'],      # 'BAU' or 'High'
}

# Create the settings parameters
''' This will create a parameter CSV based on `grid_search`. '''
create_grid_search_parameters(grid_search)

# Read the template for the custom settings
grid_search_df = create_grid_search_template()


# Create the task runs

# # 1) Submit task to a single linux machine, and run simulations parallely
# create_task_runs(grid_search_df, mode='single', python_path='/home/582/jw6041/miniforge3/envs/luto/bin/python', n_workers=40, waite_mins=1.5)

# 2) Submit task to multiple linux computation nodes
create_task_runs(grid_search_df, mode='cluster')

# # 3) Submit task to a single windows machine, and run simulations parallely
# create_task_runs(grid_search_df, mode='single', python_path='F:/jinzhu/conda_env/luto/python.exe', n_workers=40, waite_mins=1.5)


