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

from luto.economics.cost import get_cost_matrices
from luto.economics.revenue import get_rev_matrices
from luto.economics.water import get_wreq_matrices, get_wuse_limits
from luto.economics.ghg import get_ghg_matrices, get_ghg_limits
from luto.economics.quantity import get_quantity_matrices
from luto.economics.transitions import (get_transition_matrices, get_exclude_matrices)
from luto.solvers.solver import solve
from luto.tools import lumap2l_mrj


class Data():
    """Provide simple object to mimic 'data' namespace from `luto.data`."""

    def __init__( self
                , bdata # Data object like `luto.data`.
                , year # Year index (number of years since 2010). To slice HDF5 bricks.
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
        self.L_MRJ = lumap2l_mrj(self.LUMAP, self.LMMAP)                        # Boolean [2, 4218733, 28]
        self.WREQ_IRR_RJ = bdata.WREQ_IRR_RJ[self.MASK]                         # Water requirements for irrigated landuses
        self.WREQ_DRY_RJ = bdata.WREQ_DRY_RJ[self.MASK]                         # Water requirements for dryland landuses
        self.WATER_LICENCE_PRICE = bdata.WATER_LICENCE_PRICE[self.MASK]         # Int16
        self.WATER_DELIVERY_PRICE = bdata.WATER_DELIVERY_PRICE[self.MASK]       # Float32
        self.WATER_YIELD_BASE_DR = bdata.WATER_YIELD_BASE_DR                    # Float32, no mask
        self.WATER_YIELD_BASE_SR = bdata.WATER_YIELD_BASE_SR[self.MASK]         # Float32
        self.FEED_REQ = bdata.FEED_REQ[self.MASK]                               # Float32 
        self.PASTURE_KG_DM_HA = bdata.PASTURE_KG_DM_HA[self.MASK]               # Int16  
        self.SAFE_PUR_MODL = bdata.SAFE_PUR_MODL[self.MASK]                     # Float32
        self.SAFE_PUR_NATL = bdata.SAFE_PUR_NATL[self.MASK]                     # Float32
        self.RIVREG_ID = bdata.RIVREG_ID[self.MASK]                             # Int16
        self.DRAINDIV_ID = bdata.DRAINDIV_ID[self.MASK]                         # Int8
        self.CLIMATE_CHANGE_IMPACT = bdata.CLIMATE_CHANGE_IMPACT[self.MASK]

        # Slice this year off HDF5 bricks. TODO: This field is not in luto.data.
        # self.WATER_YIELD_NUNC_DR = bdata.WATER_YIELDS_DR[year][self.mindices]
        # self.WATER_YIELD_NUNC_SR = bdata.WATER_YIELDS_SR[year][self.mindices]
        with h5py.File(bdata.fname_dr, 'r') as wy_dr_file:
            self.WATER_YIELD_NUNC_DR = wy_dr_file[list(wy_dr_file.keys())[0]][year][self.MASK]
        with h5py.File(bdata.fname_sr, 'r') as wy_sr_file:
            self.WATER_YIELD_NUNC_SR = wy_sr_file[list(wy_sr_file.keys())[0]][year][self.MASK]


def sync_years(base, target):
    global data, base_year, target_index
    base_year = base
    target_index = target - bdata.YR_CAL_BASE
    data = Data(bdata, target_index)


# Local matrix-getters.

def get_c_mrj():
    print('Getting production cost matrices...', end = ' ')
    output = get_cost_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_r_mrj():
    print('Getting production revenue matrices...', end = ' ')
    output = get_rev_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_g_mrj():
    print('Getting GHG emissions matrices...', end = ' ')
    output = get_ghg_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_w_mrj():
    print('Getting water requirement matrices...', end = ' ')
    output = get_wreq_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_q_mrp():
    print('Getting production quantity matrices...', end = ' ')
    output = get_quantity_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_t_mrj():
    print('Getting transition cost matrices...', end = ' ')
    output = get_transition_matrices( data
                                    , target_index
                                    , lumaps[base_year]
                                    , lmmaps[base_year] )
    print('Done.')
    return output.astype(np.float32)


def get_x_mrj():
    print('Getting exclude matrices...', end = ' ')
    output = get_exclude_matrices(data, lumaps[base_year])
    print('Done.')
    return output


def get_limits():
    print('Getting environmental limits...', end = ' ')
    # Limits is a dictionary with heterogeneous value sets.
    limits = {}
    
    limits['water'] = get_wuse_limits(data)
    # limits['ghg'] = get_ghg_limits(data)

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
        , demands # Demands in the form of a demand by commodity by year (d_c) array.
        ):
    """Solve the linear programme using the `base` lumap for `target` year."""

    # Synchronise base and target years across module so matrix-getters know.
    sync_years(base, target)

    # Add initial masked/resfactored data to data containers
    if base == data.YR_CAL_BASE: 
        lumaps[base] = data.LUMAP
        lmmaps[base] = data.LMMAP
        dvars[base]  = data.L_MRJ
        
    # Magic.
    lumaps[target], lmmaps[target], dvars[target] = solve( get_t_mrj()
                                                         , get_c_mrj()
                                                         , get_r_mrj()
                                                         , get_g_mrj()
                                                         , get_w_mrj()
                                                         , get_x_mrj()
                                                         , get_q_mrp()
                                                         , demands
                                                         , data.LU2PR
                                                         , data.PR2CM
                                                         , get_limits()
                                                         )

def run( base
       , target
       , demands
       ):
    """Run the simulation."""
    
    # The number of times the solver is to be called.
    steps = target - base
    
    # Run the simulation up to `year` sequentially.         *** Not sure that this is working ***
    if settings.STYLE == 'timeseries':
        if len(demands.shape) != 2:
            raise ValueError( "Demands need to be a time series array of "
                              "shape (years, commodities) and years > 0." )
        elif target - base > demands.shape[0]:
            raise ValueError( "Not enough years in demands time series.")
        else:
            print( "\nRunning LUTO %s timeseries from %s to %s at resfactor %s, starting at %s." % (settings.VERSION, base, target, settings.RESFACTOR, time.ctime()) )
            for s in range(steps):
                print( "\n-------------------------------------------------" )
                print( "Running for year %s..." % (base + s + 1) )
                print( "-------------------------------------------------\n" )
                step(base + s, base + s + 1 , demands[s], x_mrj)
                
                # Need to fix how the 'base' land-use map is updated in timeseries runs
                
    # Run the simulation from YR_CAL_BASE to `target` year directly.
    elif settings.STYLE == 'snapshot':
        # If demands is a time series, choose the appropriate entry.
        if len(demands.shape) == 2:
            demands = demands[target - bdata.YR_CAL_BASE ]       # Demands needs to be a timeseries from 2010 to target year                   # ******************* check the -1 is correct indexing
        print( "\nRunning LUTO %s snapshot for %s at resfactor %s, starting at %s" % (settings.VERSION, target, settings.RESFACTOR, time.ctime()) )
        print( "\n-------------------------------------------------" )
        print( "Running for year %s..." % target )
        print( "-------------------------------------------------\n" )
        step(base, target, demands)

    else:
        raise ValueError("Unkown style: %s." % settings.STYLE)



##################################################################################
# Main code                                                                      #
##################################################################################

# Containers for simulation output. 
lumaps = {}
lmmaps = {}
dvars = {}

