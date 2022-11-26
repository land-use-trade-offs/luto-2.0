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
from scipy.interpolate import interp1d, NearestNDInterpolator
import h5py, time

import luto.data as bdata
import luto.settings as settings
# from luto.data.economic import exclude
from luto.economics.cost import get_cost_matrices
from luto.economics.quantity import get_quantity_matrices
from luto.economics.transitions import (get_transition_matrices, get_exclude_matrices)
from luto.economics.water import get_aqreq_limits
from luto.solvers.solver import solve
from luto.tools.plotmap import plotmap
from luto.tools import lumap2x_mrj


class Data():
    """Provide simple object to mimic 'data' namespace from `luto.data`."""

    def __init__( self
                , bdata # Data object like `luto.data`.
                , year # Year (zero-based). To slice HDF5 bricks.
                , lumap # Land-use map from which spatial domain is inferred.
                , resmask = None # Spatial coarse-graining mask.
                ):
        """Initialise Data object based on land-use map `lumap`."""

        # Most data is taken over as is.
        for key in bdata.__dict__:
            if key.isupper():
             self.__dict__[key] = bdata.__dict__[key]
        self.LU2PR = bdata.LU2PR
        self.PR2CM = bdata.PR2CM

        # Mask from lumap.
        self.mask_lu_code = -1 # The lu code to exclude.
        self.lumask = lumap != self.mask_lu_code   # True means _included_. Boolean dtype.
        # Mask from resfactor is superimposed on lumap mask.
        if resmask is not None:
            # self.resmask = resmask
            self.mask = self.lumask * resmask      # Boolean dtype
        else:
            self.mask = self.lumask                # Boolean dtype
        self.mindices = np.where(self.mask)[0].astype(np.int32)

        # Spatial data is sub-setted based on the above masks.
        self.NCELLS = self.mask.sum()
        self.EXCLUDE = bdata.EXCLUDE[:, self.mask, :]
        self.AGEC_CROPS = bdata.AGEC_CROPS.iloc[self.mask]                      # MultiIndex Dataframe [4218733 rows x 342 columns]
        self.AGEC_LVSTK = bdata.AGEC_LVSTK.iloc[self.mask]                      # MultiIndex Dataframe [4218733 rows x 39 columns]
        self.REAL_AREA = bdata.REAL_AREA[self.mask]                             # Actual Float32
        self.LUMAP = bdata.LUMAP[self.mask]                                     # Int8
        self.LMMAP = bdata.LMMAP[self.mask]                                     # Int8
        self.AQ_REQ_IRR_RJ = bdata.AQ_REQ_IRR_RJ[self.mask]                     # Water requirements for irrigated landuses
        self.AQ_REQ_DRY_RJ = bdata.AQ_REQ_DRY_RJ[self.mask]                     # Water requirements for dryland landuses
        self.WATER_LICENCE_PRICE = bdata.WATER_LICENCE_PRICE[self.mask]         # Int16
        self.WATER_DELIVERY_PRICE = bdata.WATER_DELIVERY_PRICE[self.mask]       # Float32
        self.WATER_YIELD_BASE_DR = bdata.WATER_YIELD_BASE_DR                    # Float32, no mask
        self.WATER_YIELD_BASE_SR = bdata.WATER_YIELD_BASE_SR[self.mask]         # Float32
        self.FEED_REQ = bdata.FEED_REQ[self.mask]                               # Float32 
        self.PASTURE_KG_DM_HA = bdata.PASTURE_KG_DM_HA[self.mask]               # Int16  
        self.SAFE_PUR_MODL = bdata.SAFE_PUR_MODL[self.mask]                     # Float32
        self.SAFE_PUR_NATL = bdata.SAFE_PUR_NATL[self.mask]                     # Float32
        self.RIVREG_ID = bdata.RIVREG_ID[self.mask]                             # Int16
        self.DRAINDIV_ID = bdata.DRAINDIV_ID[self.mask]                         # Int8
        self.CLIMATE_CHANGE_IMPACT = bdata.CLIMATE_CHANGE_IMPACT[self.mask]
        self.RIVREG_DICT = bdata.RIVREG_DICT
        self.DRAINDIV_DICT = bdata.DRAINDIV_DICT

        # Slice this year off HDF5 bricks. TODO: This field is not in luto.data.
        # self.WATER_YIELD_NUNC_DR = bdata.WATER_YIELDS_DR[year][self.mindices]
        # self.WATER_YIELD_NUNC_SR = bdata.WATER_YIELDS_SR[year][self.mindices]
        with h5py.File(bdata.fname_dr, 'r') as wy_dr_file:
            self.WATER_YIELD_NUNC_DR = wy_dr_file[list(wy_dr_file.keys())[0]][year][self.mindices]
        with h5py.File(bdata.fname_sr, 'r') as wy_sr_file:
            self.WATER_YIELD_NUNC_SR = wy_sr_file[list(wy_sr_file.keys())[0]][year][self.mindices]


def sync_years(base, target):
    global data, base_year, target_year, target_index
    base_year = base
    target_year = target
    target_index = target - bdata.ANNUM # - 1                     ************ Changed by BB. I think -1 is wrong. All timeseries data must incude 2010 then dont need -1
    data = Data(bdata, target_index, lumaps[base], resmask)


def is_resfactor():
    """Return whether resfactor (spatial coarse-graining) is on."""
    return resfactor


def get_resfactor():
    """Return current spatial coarse-graining mask and correction factor."""
    if is_resfactor():
        return resmask, resmult
    else:
        raise ValueError("Resfactor (spatial coarse-graining) is off.")


def set_resfactor(factor, sampling):
    global resfactor, ressamp, resmask, resmult
    if factor and factor != 1:
        resfactor = True
        ressamp = sampling
        resmask, resmult = rfparams(bdata.LUMAP, factor, sampling)
    else:
        resfactor = False
        resmask, resmult = None, 1


def get_shape(year=None):
    """Return the shape (NLMS, NCELLS, NLUS) of the problem at `year`."""
    if year is None:
        return data.NLMS, data.NCELLS, data.NLUS
    else:
        return shapes[year]


# Local matrix-getters with resfactor multiplier.

def get_c_mrj():
    print('Getting cost matrices...', end = ' ')
    output = resmult * get_cost_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_q_mrp():
    print('Getting quantity matrices...', end = ' ')
    output = resmult * get_quantity_matrices(data, target_index)
    print('Done.')
    return output.astype(np.float32)


def get_t_mrj():
    print('Getting transition matrices...', end = ' ')
    output = resmult * get_transition_matrices( data
                                            , target_index
                                            , lumaps[base_year][data.mask]
                                            , lmmaps[base_year][data.mask] )
    print('Done.')
    return output.astype(np.float32)


def get_x_mrj():
    print('Getting exclude matrices...', end = ' ')
    output = get_exclude_matrices(data, lumaps[base_year][data.mask])
    print('Done.')
    return output


def get_limits():
    print('Getting environmental limits...', end = ' ')
    # Limits is a dictionary with heterogeneous value sets.
    limits = {}
    
    aqreq_mrj, aqreq_limits = get_aqreq_limits(data)
    
    limits['water'] = [aqreq_mrj, aqreq_limits]

    # # Water limits.
    # aquse_limits = [] # A list of water use limits by drainage division.
    # for region in np.unique(data.DRAINDIV_ID): # 7 == MDB
    #     mask = np.where(data.DRAINDIV_ID == region, True, False)[data.mindices]
    #     basefrac = get_water_stress_basefrac(data, mask)
    #     stress = get_water_stress(data, target_index, mask)
    #     stresses.append((basefrac, stress))
    #     limits['water'] = stresses
    
    print('Done.')
    return limits


def reconstitute(lxmap, filler = -1):
    """Return lxmap reconstituted to original size spatial domain."""
    # First case is when resfactor is False/1.
    if data.lumask.sum() == lxmap.shape[0]:
        indices = np.cumsum(data.lumask) - 1
        return np.where(data.lumask, lxmap[indices], filler)
    # With resfactor on, then the map is filled out differently.
    elif lxmap.shape != data.lumask.shape: # Uncoursify returns full size maps.
        raise ValueError("Map not of right shape.")
    else:
        return np.where(data.lumask, lxmap, filler)


def uncoarse1D(lxmap, mask):
    # Array with all x-coordinates on the larger map.
    allindices = np.arange(mask.shape[0])
    # Array of x-coordinates on larger map of entries in lxmap.
    knownindices = allindices[mask]
    # Instantiate an interpolation function.
    f = interp1d(knownindices, lxmap, kind='nearest', fill_value='extrapolate')
    # The uncoursified map is obtained by interpolating the missing values.
    return f(allindices).astype(np.int8)


def uncoarse2D(lxmap, mask):
    # Arrays with all x, y -coordinates on the larger map.
    allindices = np.nonzero(bdata.NLUM_MASK)
    # Arrays with x, y -coordinates on the larger map of entries in lxmap.
    knownindices = tuple(np.stack(allindices)[:, mask])
    # Instantiate an interpolation function.
    f = NearestNDInterpolator(knownindices, lxmap)
    # The uncoursified map is obtained by interpolating the missing values.
    return f(allindices).astype(np.int8)


def uncoursify(lxmap, mask, sampling):
    """Restore the lxmap to pre-resfactored extent."""
    if mask.sum() != lxmap.shape[0]:
        raise ValueError("Map and mask shapes do not match.")
    elif sampling == 'linear':
        return uncoarse1D(lxmap, mask)
    elif sampling == 'quadratic':
        return uncoarse2D(lxmap, mask)
    else:
        raise ValueError("Unidentified problem.")


def rfparams(lxmap, factor, sampling):
    """Return coarse-graining mask and correction given map and strategy."""
    if sampling == 'linear':
        xs = np.zeros_like(lxmap)
        xs[::factor] = 1
        xs = xs.astype(bool)
        mult = xs.shape[0] / xs.sum()
        return xs, mult
    elif sampling == 'quadratic':
        rf_mask = bdata.NLUM_MASK.copy()
        nonzeroes = np.nonzero(rf_mask)
        rf_mask[::factor, ::factor] = 0
        xs = np.where(rf_mask[nonzeroes] == 0, True, False)
        mult = xs.shape[0] / xs.sum()
        return xs, mult
    else:
        raise ValueError("Unknown sampling: %s" % sampling)

    
    
def step( base    # Base year from which the data is taken.
        , target  # Year to be solved for.
        , demands # Demands in the form of a d_c array.
        # , penalty # Penalty level.
        # , limits  # (Environmental) limits as additional soft/hard constraints.
        ):
    """Solve the linear programme using the `base` lumap for `target` year."""

    # Synchronise base and target years across module so matrix-getters know.
    sync_years(base, target)

    # Magic.
    lumap, lmmap, X_mrj = solve( get_t_mrj()
                               , get_c_mrj()
                               , get_q_mrp()
                               , demands
                               , settings.PENALTY
                               , get_x_mrj()
                               , data.LU2PR
                               , data.PR2CM
                               , limits = get_limits()
                               )

    # First undo the doings of resfactor if it is set.
    if is_resfactor():
        lumap = uncoursify(lumap, data.mask, sampling = ressamp)
        lmmap = uncoursify(lmmap, data.mask, sampling = ressamp)

    # Then put the excluded land-use and land-man back where they were.
    lumaps[target] = reconstitute(lumap, filler = data.mask_lu_code).astype(np.int8)
    lmmaps[target] = reconstitute(lmmap, filler = 0).astype(np.int8)

    # And update the shapes dictionary.
    shapes[target] = get_shape()

    # Save the raw decision variables.
    dvars[target] = X_mrj


def run( base
       , target
       , demands
       # , penalty
       # , style = 'snapshot'
       # , resfactor = False
       ):
    """Run the simulation."""
    
    # The number of times the solver is to be called.
    steps = target - base

    # Set the options if applicable.
    # set_resfactor(resfactor, sampling= 'quadratic')
    set_resfactor(settings.RESFACTOR, settings.SAMPLING)

    # Run the simulation up to `year` sequentially.
    if settings.STYLE == 'timeseries':
        if len(demands.shape) != 2:
            raise ValueError( "Demands need to be a time series array of "
                              "shape (years, commodities) and years > 0." )
        elif target - base > demands.shape[0]:
            raise ValueError( "Not enough years in demands time series.")
        else:
            print( "\nRunning LUTO 2.0 timeseries from %s to %s at resfactor %s." % (base, target, settings.RESFACTOR) )
            for s in range(steps):
                print( "\n-------------------------------------------------" )
                print( "Running for year %s..." % (base + s + 1) )
                print( "-------------------------------------------------\n" )
                step(base + s, base + s + 1 , demands[s])

    # Run the simulation from ANNUM to `target` year directly.
    elif settings.STYLE == 'snapshot':
        # If demands is a time series, choose the appropriate entry.
        if len(demands.shape) == 2:
            demands = demands[target - bdata.ANNUM ] # - 1]                        ### Demands needs to be a timeseries from 2010 to target year                   # ******************* check the -1 is correct indexing
        print( "\nRunning LUTO 2.0 snapshot for %s at resfactor %s." % (target, settings.RESFACTOR) )
        print( "\n-------------------------------------------------" )
        print( "Running for year %s..." % target )
        print( "-------------------------------------------------\n" )
        step(base, target, demands)

    else:
        raise ValueError("Unkown style: %s." % settings.STYLE)


def info():
    """Return information about state of the simulation."""

    print("Years solved: %i." % (len(lumaps)-1))

    if is_resfactor():
        _, rmult = get_resfactor()
        rf = int(np.sqrt(rmult).round())
        print( "Resfactor set at %i." % rf, "Sampling one in every %i x %i cells of spatial domain." % (rf, rf) )
    else:
        print("Resfactor not set. Spatial coarse graining bypassed.")
    print()

    print("{:<6} {:>10} {:>10}".format("Year", "Cells [#]", "Cells [%]"))
    for year, shape in shapes.items():
        _, ncells, _ = shape
        fraction = ncells / bdata.NCELLS
        print("{:<6} {:>10,} {:>10.2%}".format(year, ncells, fraction))

    print()


def show_map(year):
    """Show a plot of the lumap of `year`."""
    plotmap(lumaps[year], labels = bdata.LU2DESC)


def get_results(year = None):
    """Return the simulation results for `year` or all if `year is None`."""
    if year is None:
        return lumaps, lmmaps
    else:
        return lumaps[year], lmmaps[year]


def get_production(year):
    """Return total production of commodities for a specific year...
       
    Can return base year production (e.g., year = 2010) or can return production for 
    a simulated year if one exists (i.e., year = 2030) check sim.info()).
    
    Includes the impacts of land-use change, productivity increases, and 
    climate change on yield."""

    # Acquire local names for matrices and shapes.
    lu2pr_pj = bdata.LU2PR
    pr2cm_cp = bdata.PR2CM
    nlus = bdata.NLUS
    nprs = bdata.NPRS
    ncms = bdata.NCMS

    # Get the maps in decision-variable format. 0/1 array of shape r, j
    X_dry, X_irr = lumap2x_mrj(lumaps[year], lmmaps[year])
    X_dry = X_dry.T
    X_irr = X_irr.T

    # Get the quantity matrices. Quantity array of shape m, r, p
    # q_mrp = get_quantity_matrices(bdata, 0)                                 ***************** Should year be hard coded as 0? Changed by BB as it seems wrong.
    q_mrp = get_quantity_matrices(bdata, year - bdata.ANNUM)

    X_dry_pr = [ X_dry[j] for p in range(nprs) for j in range(nlus)
                    if lu2pr_pj[p, j] ]
    X_irr_pr = [ X_irr[j] for p in range(nprs) for j in range(nlus)
                    if lu2pr_pj[p, j] ]

    # Quantities in product (PR/p) representation by land management (dry/irr).
    q_dry_p = [ q_mrp[0].T[p] @ X_dry_pr[p] for p in range(nprs) ]
    q_irr_p = [ q_mrp[1].T[p] @ X_irr_pr[p] for p in range(nprs) ]

    # Transform quantities to commodity (CM/c) representation by land management (dry/irr).
    q_dry_c = [ sum(q_dry_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                for c in range(ncms) ]
    q_irr_c = [ sum(q_irr_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                for c in range(ncms) ]

    # Total quantities in commodity (CM/c) representation.
    q_c = [q_dry_c[c] + q_irr_c[c] for c in range(ncms)]

    # Return total commodity production.
    return np.array(q_c)



############################################################################################################################
# Main code                                                                                                                #
############################################################################################################################

# Containers for simulation output. First entry from base data.
lumaps = {bdata.ANNUM: bdata.LUMAP}
lmmaps = {bdata.ANNUM: bdata.LMMAP}
shapes = {bdata.ANNUM: (bdata.NLMS, bdata.NCELLS, bdata.NLUS)}
dvars = {}

# Provide agricultural commodity demands as timeseries from 2010 to target year.
# rnd = (np.random.rand(100, len(bdata.COMMODITIES)) + 1) # random numbers [-0.5 - +0.5], shape = 100 years x 26 commodities. Placeholder demand multiplier.
# prod_2010 = get_production(bdata.ANNUM)
# d_c = prod_2010 + (prod_2010 * rnd / 10) # Demands calculated as random +/- difference compared to 2010 production
# d_c = np.round(d_c).astype(np.int32)
# np.save('input/d_c.npy', d_c)
d_c = np.load('input/d_c.npy') # Saved so as to be reproducible
### 
