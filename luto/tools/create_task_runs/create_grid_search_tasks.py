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


import numpy as np
from luto.tools.create_task_runs.helpers import (
    get_settings_df,
    get_grid_search_param_df,
    get_grid_search_settings_df, 
    create_task_runs,
)

# Define the root dir for the task runs
TASK_ROOT_DIR = '../Custom_runs/20250924_RES5_HIGH_HIGH' # Do not include the trailing slash (/) in the end of the path


# Set the grid search parameters
grid_search = {
    ###############################################################
    # Task run settings for submitting the job to the cluster
    ###############################################################
    'MEM': ['60GB'],
    'NCPUS':[15],
    'TIME': ['5:00:00'],
    'QUEUE': ['normalsr'],                                                  # normalsr for CPU, hugemembw for memory intensive jobs
    
 
    ###############################################################
    # Working settings for the model run
    ###############################################################
    'OBJECTIVE': ['maxprofit'],                                             # 'maxprofit' or 'mincost'
    'RESFACTOR': [5],
    'SIM_YEARS': [list(range(2020,2051,5))],                                    # Years to run the model 
    'WRITE_THREADS': [2],
    
 
    ###############################################################
    # Model run settings
    ###############################################################
    
    
    # --------------- Scenarios ---------------
    'SSP': ['245'],                                                         #'126', '245', '370', '585'
    
    # --------------- Economics ---------------
    'BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR': [10, 20, 30, 50, 100, 200],  # AUD/ha/year       
    'SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR':[10, 20, 30, 50, 100, 200],  # AUD/ha/year  

    # --------------- Target deviation weight ---------------
    'SOLVER_WEIGHT_DEMAND': [1], 
    'SOLVER_WEIGHT_GHG': [1],
    'SOLVER_WEIGHT_WATER': [1],
    'SOLVER_WEIGHT_GBF2': [1],


    # --------------- Social license ---------------
    'EXCLUDE_NO_GO_LU': [False],                                            # True or False
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],                               # 'off', 'on', 'NON_AG_UNIFORM'    
    'REGIONAL_ADOPTION_NON_AG_UNIFORM': [15],                               # Only work under 'NON_AG_UNIFORM'; None or numbers between 0-100 (both inclusive);  E.g., 5 means each non-ag land can not exceed 5% adoption in every region
    'REGIONAL_ADOPTION_ZONE': ['NRM_CODE'],                                 # One of 'ABARES_AAGIS', 'LGA_CODE', 'NRM_CODE', 'IBRA_ID', 'SLA_5DIGIT'


    # --------------- GHG settings ---------------
    'GHG_EMISSIONS_LIMITS': ['high'],                                       # 'off', 'low', 'medium', 'high'
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'GHG_CONSTRAINT_TYPE': ['hard'],                                        # 'hard' or 'soft'
    'USE_GHG_SCOPE_1': [True],                                              # True or False

    
    # --------------- Water constraints ---------------
    'WATER_REGION_DEF':['Drainage Division'],                               # 'River Region' or 'Drainage Division' Bureau of Meteorology GeoFabric definition
    'WATER_LIMITS': ['on'],                                                 # 'on' or 'off'
    'WATER_CONSTRAINT_TYPE': ['hard'],                                      # 'hard' or 'soft'
    'WATER_PENALTY': [1e-5],
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    
    # --------------- Biodiversity overall ---------------
    'HABITAT_CONDITION': ['USER_DEFINED'],                                  # One of [10, 25, 50, 75, 90], or 'USER_DEFINED'              
    'CONNECTIVITY_SOURCE': ['NCI'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [40],                    # Percentage of degraded areas to cut in GBF2 priority areas
    
    # --------------- Biodiversity settings - GBF 2 ---------------
    'BIODIVERSITY_TARGET_GBF_2': ['high'],                                  # 'off', 'low', 'medium', 'high'
    'GBF2_CONSTRAINT_TYPE': ['hard'],                                       # 'hard' or 'soft'

    # --------------- Biodiversity settings - GBF 3 ---------------
    'BIODIVERSITY_TARGET_GBF_3': ['off'],                                   # 'off', 'medium', 'high', 'USER_DEFINED'
    
    # --------------- Biodiversity settings - GBF 4 ---------------
    'BIODIVERSITY_TARGET_GBF_4_SNES': ['off'],                              # 'on' or 'off'.
    'BIODIVERSITY_TARGET_GBF_4_ECNES': ['off'],                             # 'on' or 'off'.

    # --------------- Biodiversity settings - GBF 8 ---------------
    'BIODIVERSITY_TARGET_GBF_8': ['off'],                                   # 'on' or 'off'

    ###############################################################
    # Scenario settings for the model run
    ###############################################################
    'SOLVE_WEIGHT_ALPHA': [1],                                              # between 0 and 1, if 1 will turn off biodiversity objective, if 0 will turn off profit objective
    'SOLVE_WEIGHT_BETA':  [0.5],         
    
    
    #-------------------- Diet BAU --------------------
    'DIET_DOM': ['BAU',],                                                   # 'BAU' or 'FLX'
    'DIET_GLOB': ['BAU',],                                                  # 'BAU' or 'FLX'
    'WASTE': [1],                                                           # 1 or 0.5
    'FEED_EFFICIENCY': ['BAU'],                                             # 'BAU' or 'High'
    #---------------------Diet FLX --------------------
    # 'DIET_DOM': ['FLX',],                                                 # 'BAU' or 'FLX'
    # 'DIET_GLOB': ['FLX',],                                                # 'BAU' or 'FLX'
    # 'WASTE': [0.5],                                                       # 1 or 0.5
    # 'FEED_EFFICIENCY': ['High'],                                          # 'BAU' or 'High'
}


duplicate_runs = {
    'REGIONAL_ADOPTION_CONSTRAINTS': ('off', 'REGIONAL_ADOPTION_NON_AG_UNIFORM'),
    'BIODIVERSITY_TARGET_GBF_2': ('off', 'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT'),
}




if __name__ == '__main__':
    
    # Create the grid settings parameters
    default_settings_df = get_settings_df(TASK_ROOT_DIR)
    grid_search_param_df = get_grid_search_param_df(grid_search)
    grid_search_settings_df = get_grid_search_settings_df(TASK_ROOT_DIR, default_settings_df, grid_search_param_df)
    
    # Remove unnecessary runs
    rm_idx = []
    for idx, row in grid_search_param_df.iterrows():
        for k, v in duplicate_runs.items():
            if (row[k] == v[0]) and (str(row[v[1]]) != str(grid_search[v[1]][0])):
                rm_idx.append(row['run_idx'])
                
    grid_search_param_df = grid_search_param_df[~grid_search_param_df['run_idx'].isin(rm_idx)]
    grid_search_param_df.to_csv(f'{TASK_ROOT_DIR}/grid_search_parameters.csv', index=False)
    print(f'Removed {len(set(rm_idx))} unnecessary runs!')
    
    # Get full settings df
    grid_search_settings_df = get_grid_search_settings_df(TASK_ROOT_DIR, default_settings_df, grid_search_param_df)

    # 1) Submit task to a single linux machine, and run simulations parallely
    # create_task_runs(TASK_ROOT_DIR, grid_search_settings_df, mode='single', n_workers=min(len(grid_search_param_df), 100))

    # 2) Submit task to multiple linux computation nodes
    create_task_runs(TASK_ROOT_DIR, grid_search_settings_df, mode='cluster', max_concurrent_tasks = 200)

