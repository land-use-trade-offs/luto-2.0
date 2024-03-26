# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
#
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>.

"""
To maintain state and handle iteration and data-view changes. This module
functions as a singleton class. It is intended to be the _only_ part of the
model that has 'global' varying state.
"""


import os
import math
import time
from datetime import datetime
import numpy as np

import luto.settings as settings

import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.transitions as ag_transition
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.biodiversity as ag_biodiversity

import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.quantity as non_ag_quantity
import luto.economics.non_agricultural.revenue as non_ag_revenue
import luto.economics.non_agricultural.transitions as non_ag_transition
import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.non_agricultural.biodiversity as non_ag_biodiversity

from luto import tools
from luto.economics import land_use_culling
from luto.solvers.solver import InputData, LutoSolver

import luto.data as bdata


# Get the total demand quantities by commodity for 2010 to 2100 by combining the demand deltas with 2010 production
prod_2010_c = tools.get_production( bdata
                                  , bdata.YR_CAL_BASE
                                  , tools.lumap2ag_l_mrj(bdata.LUMAP, bdata.LMMAP)
                                  , tools.lumap2non_ag_l_mk(bdata.LUMAP, len(bdata.NON_AGRICULTURAL_LANDUSES))
                                  , tools.get_base_am_vars(bdata.NCELLS, bdata.NLMS, bdata.N_AG_LUS)
                                  )

# Demand deltas can be a time series (shape year x commodity) or a single array (shape = n commodites).
# d_cy = bdata.DEMAND_DELTAS_C * prod_2010_c
d_cy = bdata.DEMAND_C # new demand is in tonnes rather than deltas

# The GHG from off-land commodities 
ghg_offland_cy = bdata.OFF_LAND_GHG_EMISSION_C


class Data():
    """Provide simple object to mimic 'data' namespace from `luto.data`."""

    def __init__( self
                , bdata  # Data object like `luto.data`.
                , yr_idx # Year index (number of years since 2010). To slice HDF5 bricks.
                ):
        """Initialise Data object based on land-use map `lumap`."""

        # Import all UPPERCASE objects from bdata into data object.
        for key in bdata.__dict__:
            if key.isupper():
             self.__dict__[key] = bdata.__dict__[key]

        # Spatial data is sub-setted based on the above masks.
        self.NCELLS = self.MASK.sum()
        self.EXCLUDE = bdata.EXCLUDE[:, self.MASK, :]
        self.AGEC_CROPS = bdata.AGEC_CROPS.iloc[self.MASK]                      # MultiIndex Dataframe [4218733 rows x 342 columns]
        self.AGEC_LVSTK = bdata.AGEC_LVSTK.iloc[self.MASK]                      # MultiIndex Dataframe [4218733 rows x 39 columns]
        self.AGGHG_CROPS = bdata.AGGHG_CROPS.iloc[self.MASK]                    # MultiIndex Dataframe [4218733 rows x ? columns]
        self.AGGHG_LVSTK = bdata.AGGHG_LVSTK.iloc[self.MASK]                    # MultiIndex Dataframe [4218733 rows x ? columns]
        self.REAL_AREA = bdata.REAL_AREA[self.MASK] * bdata.RESMULT             # Actual Float32
        self.LUMAP = bdata.LUMAP[self.MASK]                                     # Int8
        self.LMMAP = bdata.LMMAP[self.MASK]                                     # Int8
        self.AMMAP_DICT = {
            am: array[self.MASK] for am, array in bdata.AMMAP_DICT.items()
        }                                                                       # Dictionary containing Int8 arrays
        self.AG_L_MRJ = tools.lumap2ag_l_mrj(self.LUMAP, self.LMMAP)            # Boolean [2, 4218733, 28]
        self.NON_AG_L_RK = tools.lumap2non_ag_l_mk(
            self.LUMAP, len(self.NON_AGRICULTURAL_LANDUSES)
        )                                                                       # Int8
        self.AG_MAN_L_MRJ_DICT = tools.get_base_am_vars(
            self.NCELLS, self.NLMS, self.N_AG_LUS
        )                                                                       # Dictionary containing Int8 arrays
        self.PROD_2010_C = prod_2010_c                                          # Float, total agricultural production in 2010, shape n commodities
        self.D_CY = d_cy                                                        # Float, total demand for agricultural production, shape n commodities by 91 years
        self.WREQ_IRR_RJ = bdata.WREQ_IRR_RJ[self.MASK]                         # Water requirements for irrigated landuses
        self.WREQ_DRY_RJ = bdata.WREQ_DRY_RJ[self.MASK]                         # Water requirements for dryland landuses
        self.WATER_LICENCE_PRICE = bdata.WATER_LICENCE_PRICE[self.MASK]         # Int16
        self.WATER_DELIVERY_PRICE = bdata.WATER_DELIVERY_PRICE[self.MASK]       # Float32
        # self.WATER_YIELD_BASE_DR = bdata.WATER_YIELD_BASE_DR[self.MASK]         # Float32
        self.WATER_YIELD_BASE_SR = bdata.WATER_YIELD_BASE_SR[self.MASK]         # Float32
        self.WATER_YIELD_BASE = bdata.WATER_YIELD_BASE[self.MASK]               # Float32
        self.FEED_REQ = bdata.FEED_REQ[self.MASK]                               # Float32
        self.PASTURE_KG_DM_HA = bdata.PASTURE_KG_DM_HA[self.MASK]               # Int16  
        self.SAFE_PUR_MODL = bdata.SAFE_PUR_MODL[self.MASK]                     # Float32
        self.SAFE_PUR_NATL = bdata.SAFE_PUR_NATL[self.MASK]                     # Float32
        self.RIVREG_ID = bdata.RIVREG_ID[self.MASK]                             # Int16
        self.DRAINDIV_ID = bdata.DRAINDIV_ID[self.MASK]                         # Int8
        self.CLIMATE_CHANGE_IMPACT = bdata.CLIMATE_CHANGE_IMPACT[self.MASK]
        self.EP_EST_COST_HA = bdata.EP_EST_COST_HA[self.MASK]                   # Float32
        self.AG2EP_TRANSITION_COSTS_HA = bdata.AG2EP_TRANSITION_COSTS_HA        # Float32
        self.EP2AG_TRANSITION_COSTS_HA = bdata.EP2AG_TRANSITION_COSTS_HA        # Float32
        self.EP_BLOCK_AVG_T_CO2_HA = bdata.EP_BLOCK_AVG_T_CO2_HA[self.MASK]     # Float32
        self.NATURAL_LAND_T_CO2_HA = bdata.NATURAL_LAND_T_CO2_HA[self.MASK]     # Float32
        self.SOIL_CARBON_AVG_T_CO2_HA = bdata.SOIL_CARBON_AVG_T_CO2_HA[self.MASK]
        self.AGGHG_IRRPAST = bdata.AGGHG_IRRPAST[self.MASK]                     # Float32
        self.BIODIV_SCORE_RAW = bdata.BIODIV_SCORE_RAW[self.MASK]               # Float32
        self.BIODIV_SCORE_WEIGHTED = bdata.BIODIV_SCORE_WEIGHTED[self.MASK]     # Float32
        self.RP_PROPORTION = bdata.RP_PROPORTION[self.MASK]                     # Float32
        self.RP_FENCING_LENGTH = bdata.RP_FENCING_LENGTH[self.MASK]             # Float32
        self.EP_RIP_AVG_T_CO2_HA = bdata.EP_RIP_AVG_T_CO2_HA[self.MASK]         # Float32
        self.EP_BELT_AVG_T_CO2_HA = bdata.EP_BELT_AVG_T_CO2_HA[self.MASK]       # Float32
        self.CP_BLOCK_AVG_T_CO2_HA = bdata.CP_BLOCK_AVG_T_CO2_HA[self.MASK]     # Float32
        self.CP_BELT_AVG_T_CO2_HA = bdata.CP_BELT_AVG_T_CO2_HA[self.MASK]       # Float32
        self.CP_EST_COST_HA = bdata.CP_EST_COST_HA[self.MASK]

        # Slice this year off HDF5 bricks. TODO: This field is not in luto.data.
        # with h5py.File(bdata.fname_dr, 'r') as wy_dr_file:
        #     self.WATER_YIELD_DR = wy_dr_file[list(wy_dr_file.keys())[0]][yr_idx][self.MASK]
        # with h5py.File(bdata.fname_sr, 'r') as wy_sr_file:
        #     self.WATER_YIELD_SR = wy_sr_file[list(wy_sr_file.keys())[0]][yr_idx][self.MASK]


# Get date and time
timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

def sync_years(base, target):
    global data, base_year, target_index
    base_year = base
    target_index = target - bdata.YR_CAL_BASE
    data = Data(bdata, target_index)
    
    
def get_path(bdata, start, end):
    """Create a folder for storing outputs and return folder name."""
    
    # Get the years to write
    if settings.MODE == 'snapshot':
        yr_all = [start,end]
    elif settings.MODE == 'timeseries':
        yr_all = list(range(start,end+1))

    # Add some shorthand details about the model run
    post = '_'    + settings.DEMAND_CONSTRAINT_TYPE + \
           '_'    + settings.OBJECTIVE + \
           '_RF'  + str(settings.RESFACTOR) + \
           '_'    + str(yr_all[0]) + '-' + str(yr_all[-1]) + \
           '_'    + settings.MODE + \
           '_'    + str( int( bdata.GHG_TARGETS[yr_all[-1]] / 1e6)) + 'Mt'


    # Create path name
    path = 'output/' + timestamp + post

    # Get all paths 
    paths = [path]\
            + [f"{path}/out_{yr}" for yr in yr_all]\
            + [f"{path}/out_{yr}/lucc_separate" for yr in yr_all[1:]] # Skip creating lucc_separate for base year
    
    # Add the path for the comparison between base-year and target-year if in the timeseries mode
    if settings.MODE == 'timeseries':
        path_begin_end_compare = f"{path}/begin_end_compare_{yr_all[0]}_{yr_all[-1]}"
        paths = paths\
                + [path_begin_end_compare]\
                + [f"{path_begin_end_compare}/out_{yr_all[0]}",
                   f"{path_begin_end_compare}/out_{yr_all[-1]}",
                   f"{path_begin_end_compare}/out_{yr_all[-1]}/lucc_separate"]
    
    # Create all paths
    for p in paths:
        if not os.path.exists(p):
            os.mkdir(p)

    return path



# Local matrix-getters.

def get_ag_c_mrj():
    print('Getting agricultural production cost matrices...')
    output = ag_cost.get_cost_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_c_rk():
    print('Getting non-agricultural production cost matrices...')
    output = non_ag_cost.get_cost_matrix(data)
    return output.astype(np.float32)


def get_ag_r_mrj():
    print('Getting agricultural production revenue matrices...')
    output = ag_revenue.get_rev_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_r_rk():
    print('Getting non-agricultural production revenue matrices...')
    output = non_ag_revenue.get_rev_matrix(data)
    return output.astype(np.float32)


def get_ag_g_mrj():
    print('Getting agricultural GHG emissions matrices...')
    output = ag_ghg.get_ghg_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_g_rk():
    print('Getting non-agricultural GHG emissions matrices...')
    output = non_ag_ghg.get_ghg_matrix(data)
    return output.astype(np.float32)


def get_ag_w_mrj():
    print('Getting agricultural water requirement matrices...')
    output = ag_water.get_wreq_matrices(data, target_index)
    return output.astype(np.float32)


def get_ag_b_mrj():
    print('Getting agricultural biodiversity requirement matrices...')
    output = ag_biodiversity.get_breq_matrices(data)
    return output.astype(np.float32)


def get_non_ag_w_rk():
    print('Getting non-agricultural water requirement matrices...')
    output = non_ag_water.get_wreq_matrix(data)
    return output.astype(np.float32)


def get_non_ag_b_rk():
    print('Getting non-agricultural biodiversity requirement matrices...', end = ' ', flush = True)
    output = non_ag_biodiversity.get_breq_matrix(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_ag_q_mrp():
    print('Getting agricultural production quantity matrices...')
    output = ag_quantity.get_quantity_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_q_crk():
    print('Getting non-agricultural production quantity matrices...\n')
    output = non_ag_quantity.get_quantity_matrix(data)
    return output.astype(np.float32)


def get_ag_t_mrj():
    print('Getting agricultural transition cost matrices...')
    output = ag_transition.get_transition_matrices( data
                                                  , target_index
                                                  , base_year
                                                  , lumaps
                                                  , lmmaps)
    return output.astype(np.float32)


def get_ag_ghg_t_mrj():
    print('Getting agricultural transitions GHG emissions...')
    output = ag_ghg.get_ghg_transition_penalties(data, lumaps[base_year])
    return output.astype(np.float32)


def get_ag_to_non_ag_t_rk():
    print('Getting agricultural to non-agricultural transition cost matrices...\n')
    output = non_ag_transition.get_from_ag_transition_matrix(data
                                                           , base_year
                                                           , lumaps[base_year]
                                                           , lmmaps[base_year])
    return output.astype(np.float32)


def get_non_ag_to_ag_t_mrj():
    print('Getting non-agricultural to agricultural transition cost matrices...')
    output = non_ag_transition.get_to_ag_transition_matrix(data, base_year, lumaps[base_year], lmmaps[base_year])
    return output.astype(np.float32)


def get_non_ag_t_rk():
    print('Getting non-agricultural transition cost matrices...')
    output = non_ag_transition.get_non_ag_transition_matrix(data, base_year, lumaps[base_year], lmmaps[base_year])
    return output.astype(np.float32)


def get_ag_x_mrj():
    print('Getting agricultural exclusion matrices...')
    output = ag_transition.get_exclude_matrices(data, base_year, lumaps)
    return output


def get_non_ag_x_rk():
    print('Getting non-agricultural exclude matrices...')
    output = non_ag_transition.get_exclude_matrices(data, lumaps[base_year])
    return output


def get_ag_man_costs(ag_c_mrj):
    print('Getting agricultural management options\' cost effects...')
    output = ag_cost.get_agricultural_management_cost_matrices(data, ag_c_mrj, target_index)
    return output


def get_ag_man_ghg(ag_g_mrj):
    print('Getting agricultural management options\' GHG emission effects...')
    output = ag_ghg.get_agricultural_management_ghg_matrices(data, ag_g_mrj, target_index)
    return output


def get_ag_man_quantity(ag_q_mrp):
    print('Getting agricultural management options\' quantity effects...')
    output = ag_quantity.get_agricultural_management_quantity_matrices(data, ag_q_mrp, target_index)
    return output


def get_ag_man_revenue(ag_r_mrj):
    print('Getting agricultural management options\' revenue effects...')
    output = ag_revenue.get_agricultural_management_revenue_matrices(data, ag_r_mrj, target_index)
    return output


def get_ag_man_transitions(ag_t_mrj):
    print('Getting agricultural management options\' transition cost effects...')
    output = ag_transition.get_agricultural_management_transition_matrices(data, ag_t_mrj, target_index)
    return output


def get_ag_man_water(ag_w_mrj):
    print('Getting agricultural management options\' water requirement effects...')
    output = ag_water.get_agricultural_management_water_matrices(data, ag_w_mrj, target_index)
    return output


def get_ag_man_biodiversity(ag_b_mrj):
    print('Getting agricultural management options\' biodiversity effects...')
    output = ag_biodiversity.get_agricultural_management_biodiversity_matrices(data)
    return output


def get_ag_man_limits():
    print('Getting agricultural management options\' adoption limits...\n')
    output = ag_transition.get_agricultural_management_adoption_limits(data, target_index)
    return output


def get_limits(target: int):
    print('Getting environmental limits...\n')
    # Limits is a dictionary with heterogeneous value sets.
    limits = {}
    
    if settings.WATER_USE_LIMITS == 'on': limits['water'] = ag_water.get_wuse_limits(data)
    if settings.GHG_EMISSIONS_LIMITS == 'on':  limits['ghg'] = ag_ghg.get_ghg_limits(data, target)
    if settings.BIODIVERSITY_LIMITS == 'on':  limits['biodiversity'] = ag_biodiversity.get_biodiversity_limits(data, target)
    
    return limits


def get_input_data(target: int):
    ag_c_mrj = get_ag_c_mrj()
    ag_g_mrj = get_ag_g_mrj()
    ag_q_mrp = get_ag_q_mrp()
    ag_r_mrj = get_ag_r_mrj()
    ag_t_mrj = get_ag_t_mrj()
    ag_w_mrj = get_ag_w_mrj()
    ag_b_mrj = get_ag_b_mrj()
    ag_x_mrj = get_ag_x_mrj()

    land_use_culling.apply_agricultural_land_use_culling(
        ag_x_mrj, ag_c_mrj, ag_t_mrj, ag_r_mrj
    )

    return InputData(
        ag_t_mrj=ag_t_mrj,
        ag_c_mrj=ag_c_mrj,
        ag_r_mrj=ag_r_mrj,
        ag_g_mrj=ag_g_mrj,
        ag_w_mrj=ag_w_mrj,
        ag_b_mrj=ag_b_mrj,
        ag_x_mrj=ag_x_mrj,
        ag_q_mrp=ag_q_mrp,
        ag_ghg_t_mrj=get_ag_ghg_t_mrj(),
        ag_to_non_ag_t_rk=get_ag_to_non_ag_t_rk(),
        non_ag_to_ag_t_mrj=get_non_ag_to_ag_t_mrj(),
        non_ag_t_rk=get_non_ag_t_rk(),
        non_ag_c_rk=get_non_ag_c_rk(),
        non_ag_r_rk=get_non_ag_r_rk(),
        non_ag_g_rk=get_non_ag_g_rk(),
        non_ag_w_rk=get_non_ag_w_rk(),
        non_ag_b_rk=get_non_ag_b_rk(),
        non_ag_x_rk=get_non_ag_x_rk(),
        non_ag_q_crk=get_non_ag_q_crk(),
        ag_man_c_mrj=get_ag_man_costs(ag_c_mrj),
        ag_man_g_mrj=get_ag_man_ghg(ag_g_mrj),
        ag_man_q_mrp=get_ag_man_quantity(ag_q_mrp),
        ag_man_r_mrj=get_ag_man_revenue(ag_r_mrj),
        ag_man_t_mrj=get_ag_man_transitions(ag_t_mrj),
        ag_man_w_mrj=get_ag_man_water(ag_w_mrj),
        ag_man_b_mrj=get_ag_man_biodiversity(ag_b_mrj),
        ag_man_limits=get_ag_man_limits(),
        offland_ghg=ghg_offland_cy[target - bdata.YR_CAL_BASE],
        lu2pr_pj=data.LU2PR,
        pr2cm_cp=data.PR2CM,
        limits=get_limits(target),
        desc2aglu=data.DESC2AGLU,
    )


def prepare_input_data(base: int, target: int) -> InputData:
    # Synchronise base and target years across module so matrix-getters know.
    sync_years(base, target)

    # Add initial masked/resfactored data to data containers
    if base == data.YR_CAL_BASE: 
        lumaps[base] = data.LUMAP
        lmmaps[base] = data.LMMAP
        ammaps[base] = data.AMMAP_DICT
        ag_dvars[base]  = data.AG_L_MRJ
        non_ag_dvars[base] = data.NON_AG_L_RK
        ag_man_dvars[base] = data.AG_MAN_L_MRJ_DICT

    return get_input_data(target)


def solve_timeseries(steps: int, base: int, target: int):
    print('\n')
    print( f"Running LUTO {settings.VERSION} timeseries from {base} to {target} at resfactor {settings.RESFACTOR}\n" )

    for s in range(steps):
        print( "-------------------------------------------------")
        print( f"Running for year {base + s + 1}"   )
        print( "-------------------------------------------------\n" )
        start_time = time.time()

        input_data = prepare_input_data(base + s, base + s + 1)
        d_c = d_cy[s]
        
        if s == 0:
            luto_solver = LutoSolver(input_data, d_c)
            luto_solver.formulate()

        if s > 0:
            old_ag_x_mrj = luto_solver._input_data.ag_x_mrj.copy()
            old_non_ag_x_rk = luto_solver._input_data.non_ag_x_rk.copy()

            luto_solver.update_formulation(
                input_data=input_data,
                d_c=d_c,
                old_ag_x_mrj=old_ag_x_mrj,
                old_non_ag_x_rk=old_non_ag_x_rk,
                old_lumap=lumaps[base + s - 1],
                current_lumap=lumaps[base + s],
                old_lmmap=lmmaps[base + s - 1],
                current_lmmap=lmmaps[base + s],
            )

        (
            lumaps[base + s + 1],
            lmmaps[base + s + 1],
            ammaps[base + s + 1],
            ag_dvars[base + s + 1],
            non_ag_dvars[base + s + 1],
            ag_man_dvars[base + s + 1],
            prod_data[base + s + 1],
        ) = luto_solver.solve()

        print(f'Processing for {base + s + 1} completed in {round(time.time() - start_time)} seconds\n\n' )


def solve_snapshot(base: int, target: int):
    if len(d_cy.shape) == 2:
        d_c = d_cy[ target - bdata.YR_CAL_BASE ]       # Demands needs to be a timeseries from 2010 to target year
    else:
        d_c = d_cy

    print('\n')
    print( f"Running LUTO {settings.VERSION} snapshot for {target} at resfactor {settings.RESFACTOR}" )
    print( "-------------------------------------------------" )
    print( f"Running for year {target}" )
    print( "-------------------------------------------------" )

    start_time = time.time()
    input_data = prepare_input_data(base, target)
    luto_solver = LutoSolver(input_data, d_c)
    luto_solver.formulate()

    (
        lumaps[target],
        lmmaps[target],
        ammaps[target],
        ag_dvars[target],
        non_ag_dvars[target],
        ag_man_dvars[target],
        prod_data[target],
    ) = luto_solver.solve()
    
    print(f'Processing for {target} completed in {round(time.time() - start_time)} seconds\n\n')

@tools.LogToFile(f"{settings.OUTPUT_DIR}/run_{timestamp}")
def run( base
       , target
       ):
    """Run the simulation."""
    
    # Create output directories
    global path
    path = get_path(bdata, base, target)

    # Run the simulation up to `year` sequentially.         *** Not sure that timeseries mode is working ***
    if settings.MODE == 'timeseries':
        if len(d_cy.shape) != 2:
            raise ValueError( "Demands need to be a time series array of shape (years, commodities) and years > 0." )
        if target - base > d_cy.shape[0]:
            raise ValueError( "Not enough years in demands time series.")

        steps = target - base
        solve_timeseries(steps, base, target)

    # Run the simulation from YR_CAL_BASE to `target` year directly.
    elif settings.MODE == 'snapshot':
        # If demands is a time series, choose the appropriate entry.
        solve_snapshot(base, target)

    else:
        raise ValueError("Unkown MODE: %s." % settings.MODE)



##################################################################################
# Main code                                                                      #
##################################################################################

# Containers for simulation output. 
lumaps = {}
lmmaps = {}
ammaps = {}
ag_dvars = {}
non_ag_dvars = {}
ag_man_dvars = {}
prod_data = {}


