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
This module functions as a singleton class. It is intended to be the _only_
part of the model that has 'global' varying state.
"""


import numpy as np
from scipy.interpolate import interp1d, NearestNDInterpolator

import luto.data as bdata
from luto.data.economic import exclude
from luto.economics.cost import get_cost_matrices
from luto.economics.quantity import get_quantity_matrices
from luto.economics.transitions import ( get_transition_matrices
                                       , get_exclude_matrices )
from luto.economics.water import get_water_stress, get_water_stress_basefrac
from luto.solvers.solver import solve
from luto.tools.plotmap import plotmap
from luto.tools import lumap2x_mrj


class Data():
    """Provide simple object to mimic 'data' namespace from `luto.data`."""

    def __init__( self
                , bdata # Data object like `luto.data`.
                , year # Year (zero-based). To slice HDF5 bricks.
                , lumap # Land-use map from which spatial domain is inferred.
                , resmask=None # Spatial coarse-graining mask.
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
        self.lumask = lumap != self.mask_lu_code # True means _included_.
        # Mask from resfactor is superimposed on lumap mask.
        if resmask is not None:
            self.resmask = resmask
            self.mask = self.lumask * self.resmask
        else:
            self.mask = self.lumask
        self.mindices = np.where(self.mask)[0]

        # Spatial data is sub-setted based on the above masks.
        self.NCELLS = self.mask.sum()
        self.EXCLUDE = bdata.EXCLUDE[:, self.mask, :]
        self.AGEC_CROPS = bdata.AGEC_CROPS.iloc[self.mask]
        self.AGEC_LVSTK = bdata.AGEC_LVSTK.iloc[self.mask]
        self.REAL_AREA = bdata.REAL_AREA[self.mask]
        self.LUMAP = bdata.LUMAP[self.mask]
        self.LMMAP = bdata.LMMAP[self.mask]
        self.AQ_REQ_CROPS_DRY_RJ = bdata.AQ_REQ_CROPS_DRY_RJ[self.mask]
        self.AQ_REQ_CROPS_IRR_RJ = bdata.AQ_REQ_CROPS_IRR_RJ[self.mask]
        self.AQ_REQ_LVSTK_DRY_RJ = bdata.AQ_REQ_LVSTK_DRY_RJ[self.mask]
        self.AQ_REQ_LVSTK_IRR_RJ = bdata.AQ_REQ_LVSTK_IRR_RJ[self.mask]
        self.WATER_LICENCE_PRICE = bdata.WATER_LICENCE_PRICE[self.mask]
        self.WATER_DELIVERY_PRICE = bdata.WATER_DELIVERY_PRICE[self.mask]
        self.WATER_YIELD_BASE_DR = bdata.WATER_YIELD_BASE_DR[self.mask]
        self.WATER_YIELD_BASE_SR = bdata.WATER_YIELD_BASE_SR[self.mask]
        self.FEED_REQ = bdata.FEED_REQ[self.mask]
        self.PASTURE_KG_DM_HA = bdata.PASTURE_KG_DM_HA[self.mask]
        self.SAFE_PUR_MODL = bdata.SAFE_PUR_MODL[self.mask]
        self.SAFE_PUR_NATL = bdata.SAFE_PUR_NATL[self.mask]
        self.RIVREGS = bdata.RIVREGS[self.mask]
        self.CLIMATE_CHANGE_IMPACT = bdata.CLIMATE_CHANGE_IMPACT[self.mask]

        # Slice this year off HDF5 bricks. TODO: This field is not in luto.data.
        self.WATER_YIELD_NUNC_DR = bdata.WATER_YIELDS_DR[year][self.mindices]
        self.WATER_YIELD_NUNC_SR = bdata.WATER_YIELDS_SR[year][self.mindices]

# Print Gurobi output to console if True.
verbose = False

# Placeholder for module-global data object. To be set by `prep()`.
data = None

# Is the model ready for the next simulation step? Set by `prep()`.
ready = False

# Optionally coarse grain spatial domain. (Set to 1 to bypass.)
resfactor = False
ressamp = None
resmask = None
resmult = 1

# Containers for simulation output. First entry from base data.
lumaps = {bdata.ANNUM: bdata.LUMAP}
lmmaps = {bdata.ANNUM: bdata.LMMAP}
shapes = {bdata.ANNUM: (bdata.NLMS, bdata.NCELLS, bdata.NLUS)}
dvars = {}

def sync_years(base, target):
    global data, base_year, target_year, target_index
    base_year = base
    target_year = target
    target_index = target - bdata.ANNUM - 1
    data = Data(bdata, target_index, lumaps[base], resmask)

def set_verbose(flag):
    """Print Gurobi output to console if set to True."""
    global verbose
    verbose = flag

def is_verbose():
    """Return whether Gurobi output is printed to console."""
    return verbose

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
        resmask, resmult = rfparams(bdata.LUMAP, factor, sampling=sampling)
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
    return resmult * get_cost_matrices(data, target_index)
def get_q_mrp():
    return resmult * get_quantity_matrices(data, target_index)
def get_t_mrj():
    return resmult * get_transition_matrices( data
                                            , target_index
                                            , lumaps[base_year][data.mask]
                                            , lmmaps[base_year][data.mask]
                                            )
def get_x_mrj():
    return get_exclude_matrices(data, lumaps[base_year][data.mask])

def get_limits():
    # Limits are a dictionary with heterogeneous value sets.
    limits = {}

    # Water limits.
    stresses = [] # A list of the water stress by river region.
    for region in [7]: # 7 == MDB. # np.unique(data.DRAINDIVS):
        mask = np.where(data.DRAINDIVS == region, True, False)[data.mindices]
        basefrac = get_water_stress_basefrac(data, mask)
        stress = get_water_stress(data, target_index, mask)
        stresses.append((basefrac, stress))
        limits['water'] = stresses
    return limits

def reconstitute(lxmap, filler=-1):
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

def rfparams(lxmap, resfactor, sampling):
    """Return coarse-graining mask and correction given map and strategy."""
    if sampling == 'linear':
        xs = np.zeros_like(lxmap)
        xs[::resfactor] = 1
        xs = xs.astype(bool)
        factor = xs.shape[0] / xs.sum()
        return xs, factor
    elif sampling == 'quadratic':
        rf_mask = bdata.NLUM_MASK.copy()
        nonzeroes = np.nonzero(rf_mask)
        rf_mask[::resfactor, ::resfactor] = 0
        xs = np.where(rf_mask[nonzeroes] == 0, True, False)
        factor = xs.shape[0] / xs.sum()
        return xs, factor
    else:
        raise ValueError("Unknown sampling style: %s" % sampling)

def step( base    # Base year from which the data is taken.
        , target  # Year to be solved for.
        , demands # Demands in the form of a d_c array.
        , penalty # Penalty level.
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
                               , penalty
                               , get_x_mrj()
                               , data.LU2PR
                               , data.PR2CM
                               , get_limits()
                               , verbose=is_verbose() )

    # First undo the doings of resfactor if it is set.
    if is_resfactor():
        lumap = uncoursify(lumap, data.mask, sampling=ressamp)
        lmmap = uncoursify(lmmap, data.mask, sampling=ressamp)

    # Then put the excluded land-use and land-man back where they were.
    lumaps[target] = reconstitute(lumap, filler=data.mask_lu_code)
    lmmaps[target] = reconstitute(lmmap, filler=0)

    # And update the shapes dictionary.
    shapes[target] = get_shape()

    # Save the raw decision variables.
    dvars[target] = X_mrj

def run( base
       , target
       , demands
       , penalty
       , style='sequential'
       , resfactor=False
       , verbose=False
       ):
    """Run the simulation."""

    # The number of times the solver is to be called.
    steps = target - base

    # Set the options if applicable.
    set_resfactor(resfactor, sampling='quadratic')
    if verbose: set_verbose(verbose)

    # Run the simulation up to `year` sequentially.
    if style == 'sequential':
        if len(demands.shape) != 2:
            raise ValueError( "Demands need to be a time series array of "
                              "shape (years, commodities) and years > 0." )
        elif target - base > demands.shape[0]:
            raise ValueError( "Not enough years in demands time series.")
        else:
            print( "Running neoLUTO from base year %s through to %s."
                 % (base, target) )
            for s in range(steps):
                print( "* Solving linear program for year %s..."
                     % (base + s + 1), end = ' ' )
                step(base + s, base + s + 1 , demands[s], penalty)
                print("Done.")

    # Run the simulation from ANNUM to `target` year directly.
    elif style == 'direct':
        # If demands is a time series, choose the appropriate entry.
        if len(demands.shape) == 2:
            demands = demands[target - bdata.ANNUM - 1]
        print( "Running neoLUTO for year %s directly from base year %s."
             % (target, base) )
        print( "* Solving linear program for year %s..."
             % target, end = ' ' )
        step(base, target, demands, penalty)
        print("Done.")

    else:
        raise ValueError("Unkown style: %s." % style)

def info():
    """Return information about state of the simulation."""

    print("Years solved: %i." % (len(lumaps)-1))

    if is_resfactor():
        _, rmult = get_resfactor()
        rf = int(np.sqrt(rmult).round())
        print( "Resfactor set at %i." % rf
             , "Sampling one in every %i x %i cells of spatial domain."
             % (rf, rf) )
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
    plotmap(lumaps[year], labels=bdata.LU2DESC)

def get_results(year=None):
    """Return the simulation results for `year` or all if `year is None`."""
    if year is None:
        return lumaps, lmmaps
    else:
        return lumaps[year], lmmaps[year]

demands = np.zeros((99, len(bdata.COMMODITIES)))

def get_production(lumap=None, lmmap=None):
    """Return production levels of commodities from input- or base-data maps."""

    # Acquire local names for maps, matrices and shapes.
    if lumap is None or lmmap is None:
        lumap = bdata.LUMAP
        lmmap = bdata.LMMAP

    lu2pr_pj = bdata.LU2PR
    pr2cm_cp = bdata.PR2CM

    nlus = bdata.NLUS
    ncells = bdata.NCELLS
    nprs = bdata.NPRS
    ncms = bdata.NCMS

    # Get the maps in decision-variable format.
    X_dry, X_irr = lumap2x_mrj(lumap, lmmap)
    X_dry = X_dry.T
    X_irr = X_irr.T

    # Get the quantity matrices.
    q_mrp = get_quantity_matrices(bdata, 0)

    X_dry_pr = [ X_dry[j] for p in range(nprs) for j in range(nlus)
                    if lu2pr_pj[p, j] ]
    X_irr_pr = [ X_irr[j] for p in range(nprs) for j in range(nlus)
                    if lu2pr_pj[p, j] ]

    # Quantities in PR/p representation by land-management (dry/irr).
    q_dry_p = [ q_mrp[0].T[p] @ X_dry_pr[p] for p in range(nprs) ]
    q_irr_p = [ q_mrp[1].T[p] @ X_irr_pr[p] for p in range(nprs) ]

    # Transform quantities to CM/c representation by land-man (dry/irr).
    q_dry_c = [ sum(q_dry_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                for c in range(ncms) ]
    q_irr_c = [ sum(q_irr_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                for c in range(ncms) ]

    # Total quantities in CM/c representation.
    q_c = [q_dry_c[c] + q_irr_c[c] for c in range(ncms)]

    # Return current production as demands.
    return np.array(q_c)
