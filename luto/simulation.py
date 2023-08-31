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


import numpy as np
import h5py, time

import luto.data as bdata
import luto.settings as settings

import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.transitions as ag_transition
import luto.economics.agricultural.water as ag_water

import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.quantity as non_ag_quantity
import luto.economics.non_agricultural.transitions as non_ag_transition
import luto.economics.non_agricultural.revenue as non_ag_revenue

from luto.solvers.solver import solve
from luto import tools


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
        self.AG_L_MRJ = tools.lumap2ag_l_mrj(self.LUMAP, self.LMMAP)                  # Boolean [2, 4218733, 28]
        self.NON_AG_L_RK = tools.lumap2non_ag_l_mk(
            self.LUMAP, len(self.NON_AGRICULTURAL_LANDUSES)
        )                                                                       # Int8
        self.PROD_2010_C = prod_2010_c                                          # Float, total agricultural production in 2010, shape n commodities
        self.D_CY = d_cy                                                        # Float, total demand for agricultural production, shape n commodities by 91 years
        self.WREQ_IRR_RJ = bdata.WREQ_IRR_RJ[self.MASK]                         # Water requirements for irrigated landuses
        self.WREQ_DRY_RJ = bdata.WREQ_DRY_RJ[self.MASK]                         # Water requirements for dryland landuses
        self.WATER_LICENCE_PRICE = bdata.WATER_LICENCE_PRICE[self.MASK]         # Int16
        self.WATER_DELIVERY_PRICE = bdata.WATER_DELIVERY_PRICE[self.MASK]       # Float32
        self.WATER_YIELD_BASE_DR = bdata.WATER_YIELD_BASE_DR                    # Float32, no mask
        self.WATER_YIELD_BASE_SR = bdata.WATER_YIELD_BASE_SR[self.MASK]         # Float32
        self.WATER_YIELD_BASE_DIFF = bdata.WATER_YIELD_BASE_DIFF[self.MASK]     # Float32
        self.FEED_REQ = bdata.FEED_REQ[self.MASK]                               # Float32
        self.PASTURE_KG_DM_HA = bdata.PASTURE_KG_DM_HA[self.MASK]               # Int16  
        self.SAFE_PUR_MODL = bdata.SAFE_PUR_MODL[self.MASK]                     # Float32
        self.SAFE_PUR_NATL = bdata.SAFE_PUR_NATL[self.MASK]                     # Float32
        self.RIVREG_ID = bdata.RIVREG_ID[self.MASK]                             # Int16
        self.DRAINDIV_ID = bdata.DRAINDIV_ID[self.MASK]                         # Int8
        self.CLIMATE_CHANGE_IMPACT = bdata.CLIMATE_CHANGE_IMPACT[self.MASK]
        self.EP_EST_COST_HA = bdata.EP_EST_COST_HA[self.MASK]                   # Float32
        self.AG2EP_TRANSITION_COSTS = bdata.AG2EP_TRANSITION_COSTS_HA           # Float32
        self.EP2AG_TRANSITION_COSTS = bdata.AG2EP_TRANSITION_COSTS_HA           # Float32
        self.EP_BLOCK_AVG_T_C02_HA = bdata.EP_BLOCK_AVG_T_C02_HA[self.MASK]     # Float32
        self.NATURAL_LAND_T_CO2_HA = bdata.NATURAL_LAND_T_CO2_HA[self.MASK]     # Float32

        # Slice this year off HDF5 bricks. TODO: This field is not in luto.data.
        # with h5py.File(bdata.fname_dr, 'r') as wy_dr_file:
        #     self.WATER_YIELD_DR = wy_dr_file[list(wy_dr_file.keys())[0]][yr_idx][self.MASK]
        # with h5py.File(bdata.fname_sr, 'r') as wy_sr_file:
        #     self.WATER_YIELD_SR = wy_sr_file[list(wy_sr_file.keys())[0]][yr_idx][self.MASK]


def sync_years(base, target):
    global data, base_year, target_index
    base_year = base
    target_index = target - bdata.YR_CAL_BASE
    data = Data(bdata, target_index)


# Local matrix-getters.

def get_ag_c_mrj():
    print('Getting agricultural production cost matrices...', end = ' ')
    output = ag_cost.get_cost_matrices(data, target_index, lumaps[base_year])
    print('Done.')
    return output.astype(np.float32)


def get_non_ag_c_rk():
    print('Getting non-agricultural production cost matrices...', end=' ')
    output = non_ag_cost.get_cost_matrix(data)
    print('Done.')
    return output.astype(np.float32)


def get_ag_r_mrj():
    print('Getting agricultural production revenue matrices...', end = ' ')
    output = ag_revenue.get_rev_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_non_ag_r_rk():
    print('Getting non-agricultural production revenue matrices...', end = ' ')
    output = non_ag_revenue.get_rev_matrix(data)
    print('Done.')
    return output.astype(np.float32)


def get_ag_g_mrj():
    print('Getting agricultural GHG emissions matrices...', end = ' ')
    output = ag_ghg.get_ghg_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_non_ag_g_rk():
    print('Getting non-agricultural GHG emissions matrices...', end = ' ')
    output = non_ag_ghg.get_ghg_matrix(data)
    print('Done.')
    return output.astype(np.float32)


def get_ag_w_mrj():
    print('Getting agricultural water requirement matrices...', end = ' ')
    output = ag_water.get_wreq_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_non_ag_w_rk():
    print('Getting non-agricultural water requirement matrices...', end = ' ')
    output = non_ag_water.get_wreq_matrix(data)
    print('Done.')
    return output.astype(np.float32)


def get_ag_q_mrp():
    print('Getting agricultural production quantity matrices...', end = ' ')
    output = ag_quantity.get_quantity_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_non_ag_q_crk():
    print('Getting non-agricultural production quantity matrices...', end = ' ')
    output = non_ag_quantity.get_quantity_matrix(data)
    print('Done.')
    return output.astype(np.float32)


def get_ag_t_mrj():
    print('Getting agricultural transition cost matrices...', end = ' ')
    output = ag_transition.get_transition_matrices( data
                                                  , target_index
                                                  , base_year
                                                  , lumaps
                                                  , lmmaps)
    print('Done.')
    return output.astype(np.float32)


def get_ag_ghg_t_mrj():
    print('Getting agricultural transitions GHG emissions...', end = ' ')
    output = ag_ghg.get_ghg_transition_penalties(data, lumaps[base_year])
    print('Done.')
    return output.astype(np.float32)


def get_ag_to_non_ag_t_rk():
    print('Getting agricultural to non-agricultural transition cost matrices...', end = ' ')
    output = non_ag_transition.get_from_ag_transition_matrix(data
                                                           , base_year
                                                           , lumaps[base_year]
                                                           , lmmaps[base_year])
    print('Done.')
    return output.astype(np.float32)


def get_non_ag_to_ag_t_mrj():
    print('Getting non-agricultural to agricultural transition cost matrices...', end=' ')
    output = non_ag_transition.get_to_ag_transition_matrix(data, base_year, lumaps[base_year], lmmaps[base_year])
    print('Done.')
    return output.astype(np.float32)


def get_ag_x_mrj():
    print('Getting agricultural exclude matrices...', end = ' ')
    output = ag_transition.get_exclude_matrices(data, base_year, lumaps)
    print('Done.')
    return output


def get_non_ag_x_rk():
    print('Getting non-agricultural exclude matrices...', end=' ')
    output = non_ag_transition.get_exclude_matrices(data, lumaps[base_year])
    print('Done.')
    return output


def get_limits():
    print('Getting environmental limits...', end = ' ')
    # Limits is a dictionary with heterogeneous value sets.
    limits = {}
    
    if settings.WATER_USE_LIMITS == 'on': limits['water'] = ag_water.get_wuse_limits(data)
    if settings.GHG_EMISSIONS_LIMITS == 'on':  limits['ghg'] = ag_ghg.get_ghg_limits(data)

    # # Water limits.
    # wuse_limits = [] # A list of water use limits by drainage division.
    # for region in np.unique(data.DRAINDIV_ID): # 7 == MDB
    #     mask = np.where(data.DRAINDIV_ID == region, True, False)[data.mindices]
    #     basefrac = get_water_stress_basefrac(data, mask)
    #     stress = get_water_stress(data, target_index, mask)
    #     stresses.append((basefrac, stress))
    #     limits['water'] = stresses
    
    print('Done.')
    return limits

    
def step( base    # Base year from which the data is taken.
        , target  # Year to be solved for.
        , d_c     # Demand in the form of total quantities of agricultural commodities by year (d_c) array.
        ):
    """Solve the linear programme using the `base` lumap for `target` year."""

    # Synchronise base and target years across module so matrix-getters know.
    sync_years(base, target)

    # Add initial masked/resfactored data to data containers
    if base == data.YR_CAL_BASE: 
        lumaps[base] = data.LUMAP
        lmmaps[base] = data.LMMAP
        ag_dvars[base]  = data.AG_L_MRJ
        non_ag_dvars[base] = data.NON_AG_L_RK

        
    # Magic.
    (
        lumaps[target],
        lmmaps[target],
        ag_dvars[target],
        non_ag_dvars[target],
    ) = solve( get_ag_t_mrj()
             , get_ag_c_mrj()
             , get_ag_r_mrj()
             , get_ag_g_mrj()
             , get_ag_w_mrj()
             , get_ag_x_mrj()
             , get_ag_q_mrp()
             , get_ag_ghg_t_mrj()
             , get_ag_to_non_ag_t_rk()
             , get_non_ag_to_ag_t_mrj()
             , get_non_ag_c_rk()
             , get_non_ag_r_rk()
             , get_non_ag_g_rk()
             , get_non_ag_w_rk()
             , get_non_ag_x_rk()
             , get_non_ag_q_crk()
             , d_c
             , data.LU2PR
             , data.PR2CM
             , get_limits()
             )

def run( base
       , target
       ):
    """Run the simulation."""
    
    # The number of times the solver is to be called.
    steps = target - base
    
    # Run the simulation up to `year` sequentially.         *** Not sure that timeseries mode is working ***
    if settings.MODE == 'timeseries':
        if len(d_cy.shape) != 2:
            raise ValueError( "Demands need to be a time series array of shape (years, commodities) and years > 0." )
        elif target - base > d_cy.shape[0]:
            raise ValueError( "Not enough years in demands time series.")
        else:
            print( "\nRunning LUTO %s timeseries from %s to %s at resfactor %s, starting at %s." % (settings.VERSION, base, target, settings.RESFACTOR, time.ctime()) )
            for s in range(steps):
                print( "\n-------------------------------------------------" )
                print( "Running for year %s..." % (base + s + 1) )
                print( "-------------------------------------------------\n" )
                d_c = d_cy[s]
                step(base + s, base + s + 1, d_c)
                
                # Need to fix how the 'base' land-use map is updated in timeseries runs
                
    # Run the simulation from YR_CAL_BASE to `target` year directly.
    elif settings.MODE == 'snapshot':
        # If demands is a time series, choose the appropriate entry.
        if len(d_cy.shape) == 2:
            d_c = d_cy[ target - bdata.YR_CAL_BASE ]       # Demands needs to be a timeseries from 2010 to target year
        else:
            d_c = d_cy
        print( "\nRunning LUTO %s snapshot for %s at resfactor %s, starting at %s" % (settings.VERSION, target, settings.RESFACTOR, time.ctime()) )
        print( "\n-------------------------------------------------" )
        print( "Running for year %s..." % target )
        print( "-------------------------------------------------\n" )
        step(base, target, d_c)

    else:
        raise ValueError("Unkown MODE: %s." % settings.MODE)



##################################################################################
# Main code                                                                      #
##################################################################################

# Containers for simulation output. 
lumaps = {}
lmmaps = {}
ag_dvars = {}
non_ag_dvars = {}

# Get the total demand quantities by commodity for 2010 to 2100 by combining the demand deltas with 2010 production
prod_2010_c = tools.get_production( bdata
                            , bdata.YR_CAL_BASE
                            , tools.lumap2ag_l_mrj(bdata.LUMAP, bdata.LMMAP)
                            , tools.lumap2non_ag_l_mk(bdata.LUMAP, len(bdata.NON_AGRICULTURAL_LANDUSES))
                            )

# Demand deltas can be a time series (shape year x commodity) or a single array (shape = n commodites).
d_cy = bdata.DEMAND_DELTAS_C * prod_2010_c
