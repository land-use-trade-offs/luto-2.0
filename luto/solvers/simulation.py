#!/bin/env python3
#
# simulation.py - maintain state and handle iteration and data-view changes.
#
# This module functions as a singleton class. It is intended to be the _only_
# part of the model that has 'global' varying state.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-08-06
# Last modified: 2021-09-04
#

import numpy as np
from scipy.interpolate import interp1d

import luto.data as bdata
from luto.data.economic import exclude
from luto.economics.cost import get_cost_matrices
from luto.economics.quantity import get_quantity_matrices
from luto.economics.transitions import ( get_transition_matrices
                                       , get_exclude_matrices )
from luto.solvers.solver import solve
from luto.tools.plotmap import plotmap


class Data():
    """Provide simple object to mimic 'data' namespace from `luto.data`."""

    def __init__( self
                , lumap # Land-use map from which spatial domain is inferred.
                , resfactor # Spatial course-graining factor.
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
        if resfactor != 1:
            self.rfmask = resmask(bdata.LUMAP, resfactor)
            self.mask = self.lumask * self.rfmask
        else:
            self.mask = self.lumask

        # Spatial data is sub-setted based on the above masks.
        self.NCELLS = self.mask.sum()
        self.EXCLUDE = bdata.EXCLUDE[:, self.mask, :]
        self.AGEC_CROPS = bdata.AGEC_CROPS.iloc[self.mask]
        self.AGEC_LVSTK = bdata.AGEC_LVSTK.iloc[self.mask]
        self.REAL_AREA = bdata.REAL_AREA[self.mask]
        self.LUMAP = bdata.LUMAP[self.mask]
        self.LMMAP = bdata.LMMAP[self.mask]
        self.WATER_REQUIRED = bdata.WATER_REQUIRED[self.mask]
        self.WATER_LICENCE_PRICE = bdata.WATER_LICENCE_PRICE[self.mask]
        self.WATER_DELIVERY_PRICE = bdata.WATER_DELIVERY_PRICE[self.mask]
        self.FEED_REQ = bdata.FEED_REQ[self.mask]
        self.PASTURE_KG_DM_HA = bdata.PASTURE_KG_DM_HA[self.mask]
        self.SAFE_PUR_MODL = bdata.SAFE_PUR_MODL[self.mask]
        self.SAFE_PUR_NATL = bdata.SAFE_PUR_NATL[self.mask]

# Print Gurobi output to console if True.
verbose = False

# Placeholder for module-global data object. To be set by `prep()`.
data = None

# Is the model ready for the next simulation step? Set by `prep()`.
ready = False

# Optionally course grain spatial domain. (Set to 1 to bypass.)
resfactor = 1

# Containers for simulation output. First entry from base data.
lumaps = {bdata.ANNUM: bdata.LUMAP}
lmmaps = {bdata.ANNUM: bdata.LMMAP}
shapes = {bdata.ANNUM: (bdata.NLMS, bdata.NCELLS, bdata.NLUS)}
# lumaps = [bdata.LUMAP]
# lmmaps = [bdata.LMMAP]
# shapes = [(bdata.NLMS, bdata.NCELLS, bdata.NLUS)]

def sync_years(base, target):
    global base_year, target_year, target_index
    base_year = base
    target_year = target
    target_index = target - bdata.ANNUM - 1

def set_verbose(flag):
    """Print Gurobi output to console if set to True."""
    global verbose
    verbose = flag

def is_verbose():
    """Return whether Gurobi output is printed to console."""
    return verbose

def get_resfactor():
    """Return the resfactor (spatial course-graining factor)."""
    return resfactor

def set_resfactor(factor):
    global resfactor
    resfactor = factor

def get_shape(year=None):
    """Return the shape (NLMS, NCELLS, NLUS) of the problem at `year`."""
    if year is None:
        return data.NLMS, data.NCELLS, data.NLUS
    else:
        return shapes[year]

# Local matrix-getters. Zero-based (sans ANNUM off set), so get_step().
def get_c_mrj():
    return get_cost_matrices(data, target_index)
def get_q_mrp():
    return get_quantity_matrices(data, target_index)
def get_t_mrj():
    return get_transition_matrices( data
                                  , target_index
                                  , lumaps[base_year][data.mask]
                                  , lmmaps[base_year][data.mask]
                                  )
def get_x_mrj():
    return get_exclude_matrices(data, lumaps[base_year][data.mask])

def reconstitute(l_map, filler=-1):
    """Return l?map reconstituted to original size spatial domain."""
    indices = np.cumsum(data.lumask) - 1
    return np.where(data.lumask, l_map[indices], filler)

def uncoursify(lxmap, mask):
    """Restore the l?map to pre-resfactored extent."""

    if mask.sum() != lxmap.shape[0]:
        raise ValueError("Map and mask shapes do not match.")
    else:
        # Array with all x-coordinates on the larger map.
        domain = np.arange(mask.shape[0])
        # Array of x-coordinates on larger map of entries in lxmap.
        xs = domain[mask]
        # Instantiate an interpolation function.
        f = interp1d(xs, lxmap, kind='nearest', fill_value='extrapolate')
        # The uncoursified map is obtained by interpolating the missing values.
        return f(domain).astype(np.int8)

def resmask(lxmap, resfactor, sampling='linear'):
    if sampling == 'linear':
        xs = np.zeros_like(lxmap)
        xs[::resfactor] = 1
        return xs.astype(bool)
    elif sampling == 'quadratic':
        ...
        # nlum_mask = data.NLUM_MASK.copy()
        # rf_nlum = nlum_mask[::resfactor, ::resfactor]
    else:
        raise ValueError("Unknown sampling style: %s" % sampling)

def step( base    # Base year from which the data is taken.
        , target  # Year to be solved for.
        , demands # Demands in the form of a d_c array.
        , penalty # Penalty level.
        ):
    """Solve the linear programme using the `base` lumap for `target` year."""

    global data
    data = Data(lumaps[base], resfactor)

    # Synchronise base and target years across module so matrix-getters know.
    sync_years(base, target)

    # Magic.
    lumap, lmmap = solve( get_t_mrj()
                        , get_c_mrj()
                        , get_q_mrp()
                        , demands
                        , penalty
                        , get_x_mrj()
                        , data.LU2PR
                        , data.PR2CM
                        , verbose=is_verbose() )

    # First undo the doings of resfactor.
    lumap = uncoursify(lumap, data.mask)
    lmmap = uncoursify(lmmap, data.mask)

    # Then put the excluded land-use and land-man back where they were.
    lumaps[target] = reconstitute(lumap, filler=data.mask_lu_code)
    lmmaps[target] = reconstitute(lmmap, filler=0)

    # And update the shapes dictionary.
    shapes[target] = get_shape()

def run( base
       , target
       , demands
       , penalty
       , style='sequential'
       , resfactor=1
       , verbose=False
       ):
    """Run the simulation."""

    # The number of times the solver is to be called.
    steps = target - base

    # Set the options if applicable.
    if resfactor != 1: set_resfactor(resfactor)
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

    # Run the simulation from ANNUM to `year` + 1 directly.
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

    if resfactor == 1:
        print("Resfactor set at 1. Spatial course graining bypassed.")
    else:
        print( "Resfactor set at %i." % resfactor
             , "Sampling one of every %i cells in spatial domain." % resfactor )

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

def _demands():
    """Return demands based on current production."""
    if not is_ready(): prep()
    lumap = data.LUMAP
    lmmap = data.LMMAP
    _, _, nlus = get_shape()
    q_mrj = get_q_mrj()

    d_j = np.zeros(nlus) # Prepare the demands array - one cell per LU.
    for j in range(nlus):
        # The demand for LU j is the dot product of the yield vector with a
        # summation vector indicating where LU actually occurs as per SID array.
        d_j[j] = ( q_mrj[0].T[j] @ np.where((lumap==j) & (lmmap==0), 1, 0)
                + q_mrj[1].T[j] @ np.where((lumap==j) & (lmmap==1), 1, 0) )

    return d_j
