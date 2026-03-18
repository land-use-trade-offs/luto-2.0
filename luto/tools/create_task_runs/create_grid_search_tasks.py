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
import pandas as pd
from luto.tools.create_task_runs.helpers import (
    get_settings_df,
    get_grid_search_param_df,
    get_grid_search_settings_df, 
    create_task_runs,
)

# Define the root dir for the task runs
TASK_ROOT_DIR = "/g/data/jk53/jinzhu/LUTO/Custom_runs/LUF_20260318_RE5_CORE"


# Set the grid search parameters
grid_search = {
    ###############################################################
    # Task run settings for submitting the job to the cluster
    ###############################################################
    'MEM': ['64GB'],
    'WRITE_REPORT_MAX_MEM_GB': [64],                                       # Max memory for writing report (in GB)
    'NCPUS':[16],
    'TIME': ['12:00:00'],
    'QUEUE': ['normalsr'],                                                  # normalsr for CPU, hugemembw for memory intensive jobs
    
 
    ###############################################################
    # Working settings for the model run
    ###############################################################
    'OBJECTIVE': ['maxprofit'],                                             # 'maxprofit' or 'mincost'
    'RESFACTOR': [5],
    'SIM_YEARS': [range(2010,2051,5)],                                      # 2010-2050 (2010 is base year; 2060 not yet supported)
    'WRITE_THREADS': [4],
    
 
    ###############################################################
    # Model run settings
    ###############################################################
    
    
    # --------------- Scenarios ---------------
    'SSP': ['245'],                                                         # Core: SSP2-RCP4.5. Add '585' separately for higher climate impacts sensitivity (lower priority, LUF Report 2026)
    'CARBON_EFFECTS_WINDOW': [60],
    'RISK_OF_REVERSAL': [0.05],                                             # Risk of reversal buffer under ERF (aligns with ACCU methods)
    'FIRE_RISK': ['med'],                                                   # 'low', 'med', 'high' → 5th/50th/95th percentile of modelled fire impacts; 'med' = 94% median
    'CONVERGENCE': [2050],                                                  # Year at which dietary transformation is completed
    'CO2_FERT': ['off'],                                                    # 'on' or 'off'
    'APPLY_DEMAND_MULTIPLIERS': [True],                                     # True or False. Whether to apply demand multipliers from AusTIME model.
    'NON_AG_LAND_USES' : [{
        'Environmental Plantings': True,
        'Riparian Plantings': True,
        'Sheep Agroforestry': True,
        'Beef Agroforestry': True,
        'Carbon Plantings (Block)': True,       
        'Sheep Carbon Plantings (Belt)': True,  
        'Beef Carbon Plantings (Belt)': True,   
        'BECCS': False,
        'Destocked - natural land': True,
    }],

    # --------------- Economics ---------------
    'DYNAMIC_PRICE' : [True],                                               # True or False (ON per LUF Report 2026)
    'AMORTISE_UPFRONT_COSTS': [False],                                      # OFF per LUF Report 2026
    'DISCOUNT_RATE': [0.07],                                                # 7% per LUF Report 2026
    'CARBON_PRICE_COSTANT': [0],                                            # $0/tonne; LUTO does not consider cost/revenue for pure carbon emission/sequestration
    'BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR': [100, 50],                 # AUD/ha/year; 100=core, 50=Cheaper HIR sensitivity (LUF Report 2026)
    'SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR':[100, 50],                 # AUD/ha/year; 100=core, 50=Cheaper HIR sensitivity (LUF Report 2026)
    'HIR_CEILING_PERCENTAGE': [0.9],                                        # HIR achieves 90% of bio/GHG benefits of Destocked - natural land

    # --------------- Target deviation weight ---------------
    'SOLVER_WEIGHT_DEMAND': [1], 
    'SOLVER_WEIGHT_GHG': [1],
    'SOLVER_WEIGHT_WATER': [1],


    # --------------- Social license ---------------
    'EXCLUDE_NO_GO_LU': [False],                                            # True or False
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],                               # 'off' = core (LUF Report 2026); 'NON_AG_UNIFORM' = lower-priority sensitivity
    'REGIONAL_ADOPTION_NON_AG_UNIFORM': [5],                                # Default value (not active when REGIONAL_ADOPTION_CONSTRAINTS='off')
    'REGIONAL_ADOPTION_ZONE': ['NRM_CODE'],                                 # One of 'ABARES_AAGIS', 'LGA_CODE', 'NRM_CODE', 'IBRA_ID', 'SLA_5DIGIT'


    # --------------- GHG settings ---------------
    'GHG_EMISSIONS_LIMITS': ['low'],                                        # 'low'=core 1.8C 67% (LUF Report 2026); 'high'=higher ambition 1.5C 50% (lower priority sensitivity)
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'GHG_CONSTRAINT_TYPE': ['hard'],                                        # 'hard' or 'soft'
    'USE_GHG_SCOPE_1': [True],                                              # True or False

    
    # --------------- Water constraints ---------------
    'WATER_REGION_DEF':['Drainage Division'],                               # 'River Region' or 'Drainage Division' Bureau of Meteorology GeoFabric definition
    'WATER_LIMITS': ['on'],                                                 # 'on' or 'off'
    'WATER_CONSTRAINT_TYPE': ['hard'],                                      # 'hard' or 'soft'
    'WATER_STRESS': [0.6],                                                  # Water yield must be >= 60% of historical; aligns with 2023 Planetary Boundaries update
    'WATER_CLIMATE_CHANGE_IMPACT': ['on'],                                  # 'on' or 'off'; climate change impacts on water yields
    'LIVESTOCK_DRINKING_WATER': [1],                                        # 1=ON; include livestock drinking water in water balance
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    
    # --------------- Biodiversity overall ---------------
    'BIO_QUALITY_LAYER': ['Suitability'],
    'CONTRIBUTION_PERCENTILE': ['USER_DEFINED'],                            # 50th percentile of HCAS per LUF Report 2026 (was 'USER_DEFINED')
    'CONNECTIVITY_SOURCE': ['NCI'],
    'CONNECTIVITY_LB': [0.7],                                               # Connectivity score importance: 0.7 per LUF Report 2026
    
    # --------------- Biodiversity settings - GBF 2 ---------------
    'BIODIVERSITY_TARGET_GBF_2': ['high'],                                  # 'off', 'low', 'medium', 'high'
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [15, 20, 25, 30],        # Core: 20% central; test [15,25,30] alongside (LUF Report 2026)
    'GBF2_CONSTRAINT_TYPE': ['hard'],                                       # 'hard' or 'soft'

    # --------------- Biodiversity settings - GBF 3 ---------------
    'BIODIVERSITY_TARGET_GBF_3_NVIS': ['off'],                              # 'off', 'medium', 'high', 'USER_DEFINED'
    'BIODIVERSITY_TARGET_GBF_3_IBRA': ['off'],                              # 'off', 'medium', 'high', 'USER_DEFINED'

    # --------------- Biodiversity settings - GBF 4 ---------------
    'BIODIVERSITY_TARGET_GBF_4_SNES': ['off'],                              # 'on' or 'off'.
    'BIODIVERSITY_TARGET_GBF_4_ECNES': ['off'],                             # 'off'=core (LUF Report 2026); 'on'=MNES biodiversity sensitivity (lower priority)

    # --------------- Biodiversity settings - GBF 8 ---------------
    'BIODIVERSITY_TARGET_GBF_8': ['off'],                                   # 'on' or 'off'

    # --------------- Renewable energy ---------------
    # Core: RE OFF. REN1-REN4 are separate renewable energy scenario runs (not part of this grid search).
    # REN1: No GBF2 + RE ON; REN2: Core + RE ON; REN3: REN2 + GBF2 exclusion mask; REN4: REN3 + QLD EPBC MNES layer
    'RENEWABLE_ENERGY_CONSTRAINTS': ['off'],                               # 'off'=core (LUF Report 2026); 'on' for REN1-REN4 scenarios
    'RENEWABLE_TARGET_SCENARIO_TARGETS': ['Gladstone - Current Targets'],  # 'CNS - Accelerated Transition', 'CNS - Current Targets', 'Gladstone - Current Targets'
    
    ###############################################################
    # Scenario settings for the model run
    ###############################################################
    'SOLVE_WEIGHT_ALPHA': [1],                                              # between 0 and 1, if 1 will turn off biodiversity objective, if 0 will turn off profit objective
    'SOLVE_WEIGHT_BETA':  [0.5],         
    
    
    # --------------- Ag management ---------------
    'AG_MANAGEMENTS': [{
        'Asparagopsis taxiformis': True,
        'Precision Agriculture': True,
        'Ecological Grazing': False,                                        # OFF per LUF Report 2026 (controversial)
        'Savanna Burning': True,
        'AgTech EI': True,
        'Biochar': True,
        'HIR - Beef': True,
        'HIR - Sheep': True,
        'Utility Solar PV': False,                                          # OFF for core; ON for REN1-REN4
        'Onshore Wind': False,                                              # OFF for core; ON for REN1-REN4
    }],

    #-------------------- Dietary --------------------
    'DIET_DOM': ['BAU'],                                                    # 'BAU', 'FLX', 'VEG', 'VGN'
    'DIET_GLOB': ['BAU'],                                                   # 'BAU' or 'FLX'
    'WASTE': [1],                                                           # 1 or 0.5
    'FEED_EFFICIENCY': ['BAU'],                                             # 'BAU' or 'High'
    'IMPORT_TREND':['Trend'],                                               # 'Trend' per LUF Report 2026 (was 'Static')
}


duplicate_runs = {
    'REGIONAL_ADOPTION_CONSTRAINTS': ('off', 'REGIONAL_ADOPTION_NON_AG_UNIFORM'),
    'BIODIVERSITY_TARGET_GBF_2': ('off', 'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT'),
}




if __name__ == '__main__':
    
    # Create the grid settings parameters
    default_settings_df = get_settings_df(TASK_ROOT_DIR)
    grid_search_param_df = get_grid_search_param_df(grid_search)


    # Remove unnecessary runs
    rm_idx = []
    for idx, row in grid_search_param_df.iterrows():
        for k, v in duplicate_runs.items():
            if (row[k] == v[0]) and (str(row[v[1]]) != str(grid_search[v[1]][0])):
                rm_idx.append(row['run_idx'])
                
    # HIR costs must be paired; remove beef_cost != sheep_cost runs
    hir_mismatch = grid_search_param_df[
        grid_search_param_df['BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR'] !=
        grid_search_param_df['SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR']
    ]['run_idx'].tolist()
    rm_idx.extend(hir_mismatch)

    grid_search_param_df = grid_search_param_df[~grid_search_param_df['run_idx'].isin(rm_idx)]
    grid_search_param_df.to_csv(f'{TASK_ROOT_DIR}/grid_search_parameters.csv', index=False)
    print(f'Removed {len(set(rm_idx))} unnecessary runs!')
    

    # Get full settings df
    grid_search_settings_df = get_grid_search_settings_df(TASK_ROOT_DIR, default_settings_df, grid_search_param_df)
    

    # 1) Submit task to a single linux machine, and run simulations parallely
    # create_task_runs(TASK_ROOT_DIR, grid_search_settings_df, mode='single', n_workers=min(len(grid_search_param_df), 100))

    # 2) Submit task to multiple linux computation nodes
    create_task_runs(TASK_ROOT_DIR, grid_search_settings_df, mode='cluster', max_concurrent_tasks = 200)

