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
TASK_ROOT_DIR = "/g/data/jk53/jinzhu/LUTO/Custom_runs/RES1_TEST"


# Set the grid search parameters
grid_search = {
    ###############################################################
    # Task run settings for submitting the job to the cluster
    ###############################################################
    'MEM': ['300GB'],
    'WRITE_REPORT_MAX_MEM_GB': [300],                                        # Max memory for writing report (in GB)
    'NCPUS':[32],
    'TIME': ['24:00:00'],
    'QUEUE': ['normalsr'],                                                  # normalsr for CPU, hugemembw for memory intensive jobs


    ###############################################################
    # Working settings for the model run
    ###############################################################
    'OBJECTIVE':        ['maxprofit'],                                             # 'maxprofit' or 'mincost'
    'RESFACTOR':        [1],
    'SIM_YEARS':        [[2020, 2025, 2030, 2035, 2050]],                          # base year 2010 implicit; explicit step years
    'WRITE_PARALLEL':   [True],
    'WRITE_THREADS':    [4],
    'DO_IIS':           [False],  
    'WRITE_OUTPUTS':    [True],
 
    ###############################################################
    # Model run settings
    ###############################################################
    
    
    # --------------- Scenarios ---------------
    'SSP': ['245'],                                                         # Core: SSP2-RCP4.5. Add '585' separately for higher climate impacts sensitivity (lower priority, LUF Report 2026)
    'CARBON_EFFECTS_WINDOW': [60],
    'RISK_OF_REVERSAL': [0],                                                # OFF per Third iteration (aligns with ACCU methods)
    'FIRE_RISK': ['med'],                                                   # Not effect as of 20260318 following decision to drop fire risk and just use the 5% ERF risk of reversal
    'CONVERGENCE': [2050],                                                  # Year at which dietary transformation is completed
    'CO2_FERT': ['off'],                                                    # 'on' or 'off'
    'APPLY_DEMAND_MULTIPLIERS': [True],                                     # True or False. Whether to apply demand multipliers from AusTIME model.
    'PRODUCTIVITY_TREND': ['BAU'],                                          # 'BAU', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'


    # --------------- Economics ---------------
    'DYNAMIC_PRICE' : [True],                                               # True or False (ON per LUF Report 2026)
    'AMORTISE_UPFRONT_COSTS': [False],                                      # OFF per LUF Report 2026
    'DISCOUNT_RATE': [0.07],                                                # 7% per LUF Report 2026
    'AMORTISATION_PERIOD': [30],                                            # 30 years per LUF Report 2026
    'CARBON_PRICE_COSTANT': [0],                                            # $0/tonne; LUTO does not consider cost/revenue for pure carbon emission/sequestration
    'BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR': [100],                     # AUD/ha/year; $100/ha/year full maintenance cost
    'SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR':[100],                     # AUD/ha/year; $100/ha/year full maintenance cost
    'HIR_CEILING_PERCENTAGE': [0.8],                                        # HIR achieves 80% of bio/GHG benefits of Destocked - natural land

    # --------------- Target deviation weight ---------------
    'SOLVER_WEIGHT_DEMAND': [1], 
    'SOLVER_WEIGHT_GHG': [1],
    'SOLVER_WEIGHT_WATER': [1],


    # --------------- Social license ---------------
    'EXCLUDE_NO_GO_LU': [False],                                            # True or False
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],                               # 'off' = core (LUF Report 2026); 'NON_AG_CAP' = lower-priority sensitivity
    'REGIONAL_ADOPTION_NON_AG_CAP': [15],                                   # Default value (not active when REGIONAL_ADOPTION_CONSTRAINTS='off')
    'REGIONAL_ADOPTION_NON_AG_REGION': ['NRM'],                             # 'NRM' or 'IBRA' — region scope for the cap
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
    'CONTRIBUTION_PERCENTILE': ['USER_DEFINED'],                            # 50th percentile of HCAS per LUF Report 2026 (need to be 'USER_DEFINED', which is 50th percentile but with nudges for sheep/beef/dairy nat land)
    'CONNECTIVITY_SOURCE': ['NCI'],
    'CONNECTIVITY_LB': [0.7],                                               # Connectivity score importance: 0.7 per LUF Report 2026

    # --------------- Biodiversity contribution parameters ---------------
    'BIO_CONTRIBUTION_LDS': [0.75],                                         # Late dry season savanna fire regime
    'BIO_CONTRIBUTION_ENV_PLANTING': [0.7],                                 # Environmental plantings (doc=0.7, default=0.8)
    'BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK': [0.12],                       # Carbon plantings block (doc=0.12, default=0.1)
    'BIO_CONTRIBUTION_CARBON_PLANTING_BELT': [0.12],                        # Carbon plantings belt (doc=0.12, default=0.1)
    'BIO_CONTRIBUTION_RIPARIAN_PLANTING': [1.0],                            # Riparian plantings (doc=1.0, default=1.2)
    'BIO_CONTRIBUTION_AGROFORESTRY': [0.7],                                 # Agroforestry (doc=0.7, default=0.75)
    'BIO_CONTRIBUTION_BECCS': [0],                                          # BECCS (doc=0, default=0)
    'BIO_CONTRIBUTION_DESTOCKING': [0.75],                                  # Destocking (doc=0.75, default=None=uses HCAS lookup difference)
    
    # --------------- Biodiversity settings - GBF 2 ---------------
    'BIODIVERSITY_TARGET_GBF_2': ['high'],                                  # 'off', 'low', 'medium', 'high'
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [15],                    # Core: 20% central; test [15,30] alongside (Third iteration)
    'GBF2_CONSTRAINT_TYPE': ['hard'],                                       # 'hard' or 'soft'

    # --------------- Biodiversity settings - GBF 3 ---------------
    'BIODIVERSITY_TARGET_GBF_3_NVIS': ['high'],                             # 'off', 'medium', 'high', 'USER_DEFINED'
    'GBF3_NVIS_TARGET_CLASS': ['MVS'],                                      # 'MVG' or 'MVS' NVIS class
    'GBF3_NVIS_REGION_MODE': ['Australia'],                                 # 'Australia' or 'NRM'
    'GBF3_NVIS_SELECTED_REGIONS': [['North East', 'Goulburn Broken']],      # Only used when mode = 'NRM'
    'BIODIVERSITY_TARGET_GBF_3_IBRA': ['off'],                              # 'off', 'medium', 'high', 'USER_DEFINED'

    # --------------- Biodiversity settings - GBF 4 ---------------
    'BIODIVERSITY_TARGET_GBF_4_SNES': ['on'],                               # 'on' or 'off'
    'GBF4_SNES_REGION_MODE': ['Australia'],                                 # 'Australia' or 'NRM'
    'GBF4_SNES_SELECTED_REGIONS': [['North East', 'Goulburn Broken']],      # Only used when mode = 'NRM'
    'BIODIVERSITY_TARGET_GBF_4_ECNES': ['on'],                              # 'on' or 'off'
    'GBF4_ECNES_REGION_MODE': ['Australia'],                                # 'Australia' or 'NRM'
    'GBF4_ECNES_SELECTED_REGIONS': [['North East', 'Goulburn Broken']],     # Only used when mode = 'NRM'

    # --------------- Biodiversity settings - GBF 8 ---------------
    'BIODIVERSITY_TARGET_GBF_8': ['off'],                                   # 'on' or 'off'

    # --------------- Renewable energy ---------------
    # Core: RE OFF. REN1-REN4 are separate renewable energy scenario runs (not part of this grid search).
    # REN1: No GBF2 + RE ON; REN2: Core + RE ON; REN3: REN2 + GBF2 exclusion mask; REN4: REN3 + EPBC MNES mask
    'RENEWABLES_OPTIONS':[{
        'Utility Solar PV': False,                                          # OFF for core; ON for REN1-REN4
        'Onshore Wind': False,                                              # OFF for core; ON for REN1-REN4
    }],
    'RENEWABLE_TARGET_SCENARIO_TARGETS': ['Gladstone - Core'],              # 'Gladstone - Core', 'Gladstone - BESS Sensitivity', 'AEMO 2026 ISP - Step Change', etc.
    'RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS': ['step_change'],              # 'step_change', 'accelerated_transition', 'ANU_transmission_T3/T5/T10'
    'EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS': [True],                      # True: exclude renewables from GBF2 high-biodiversity cells
    'RENEWABLE_GBF2_CUT_SOLAR': [20],                                       # % coverage threshold for GBF2 exclusion mask (solar)
    'RENEWABLE_GBF2_CUT_WIND': [20],                                        # % coverage threshold for GBF2 exclusion mask (wind)
    'EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK': [True],                         # True: exclude renewables from EPBC MNES high-priority cells
    'RENEWABLE_EPBC_MNES_CUT_SOLAR': [10],                                  # % MNES coverage threshold for exclusion mask (solar)
    'RENEWABLE_EPBC_MNES_CUT_WIND': [10],                                   # % MNES coverage threshold for exclusion mask (wind)


    # --------------- Objective function weights ---------------
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
    'AG_MANAGEMENTS_REVERSIBLE': [{
        'Asparagopsis taxiformis': True,
        'Precision Agriculture': True,
        'Ecological Grazing': True,
        'Savanna Burning': True,
        'AgTech EI': True,
        'Biochar': True,
        'HIR - Beef': False,                                                # Irreversible: permanent land use change once HIR is adopted
        'HIR - Sheep': False,                                               # Irreversible: permanent land use change once HIR is adopted
        'Utility Solar PV': False,
        'Onshore Wind': False,
    }],
    
    # --------------- Non-agricultural land uses ---------------
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
    'NON_AG_LAND_USES_REVERSIBLE': [{                                       
        'Environmental Plantings': False,
        'Riparian Plantings': False,
        'Sheep Agroforestry': False,
        'Beef Agroforestry': False,
        'Carbon Plantings (Block)': False,
        'Sheep Carbon Plantings (Belt)': False,
        'Beef Carbon Plantings (Belt)': False,
        'BECCS': False,
        'Destocked - natural land': True,                                   # Destocking is reversible in the model (can switch back to grazing), but we consider it a permanent land use change for the purposes of the reversal risk buffer
    }],

    #-------------------- Dietary --------------------
    'DIET_DOM': ['BAU'],                                                    # 'BAU', 'FLX', 'VEG', 'VGN'
    'DIET_GLOB': ['BAU'],                                                   # 'BAU' or 'FLX'
    'WASTE': [1],                                                           # 1 or 0.5
    'FEED_EFFICIENCY': ['BAU'],                                             # 'BAU' or 'High'
    'IMPORT_TREND':['Trend'],                                               # 'Trend' per LUF Report 2026 (was 'Static')
}


duplicate_runs = {
    'REGIONAL_ADOPTION_CONSTRAINTS': ('off', 'REGIONAL_ADOPTION_NON_AG_CAP'),
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

