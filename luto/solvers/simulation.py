#!/bin/env python3
#
# simulation.py - maintain state and handle iteration and data-view changes.
#
# This module functions as a singleton class. It is intended to be the _only_
# part of the model that has 'global' varying state.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-08-06
# Last modified: 2021-08-06
#

import numpy as np

import luto.data as bdata
from luto.data.economic import exclude
from luto.economics.cost import get_cost_matrices
from luto.economics.quantity import get_quantity_matrices
from luto.economics.transitions import get_transition_matrices
from luto.solvers.solver import solve


class Data():
    """Provide simple object to mimic 'data' namespace from `luto.data`"""

    def __init__( self
                , lumap # Land-use map from which spatial domain is inferred.
                ):
        """Initialise Data object based on land-use map `lumap`."""
        # Most data is taken over as is.
        for key in bdata.__dict__:
            if key.isupper():
             self.__dict__[key] = bdata.__dict__[key]
        # Spatial data is sub-setted based on the lumap.
        mask = lumap == 0 # TODO: This needs to be lumap != -1.
        self.NCELLS = mask.sum()
        self.AGEC = bdata.AGEC.iloc[mask]
        self.REAL_AREA = bdata.REAL_AREA[mask]
        self.AG_DRYLAND_DAMAGE = bdata.AG_DRYLAND_DAMAGE[:, mask]
        self.AG_PASTURE_DAMAGE = bdata.AG_PASTURE_DAMAGE[:, mask]
        self.LUMAP = bdata.LUMAP[mask]
        self.LMMAP = bdata.LMMAP[mask]

# Placeholder for module-global data object. To be set by `prep()`.
data = None

# Is the model ready for the next simulation step? Set by `prep()`.
ready = False

# Containers for simulation output. First entry from base data.
lumaps = [bdata.LUMAP]
lmmaps = [bdata.LMMAP]

def get_year():
    """Return the current time step, i.e. number of years since ANNUM."""
    return len(lumaps) - 1

def is_ready():
    """Return True if model ready for next simulation step, False otherwise."""
    return ready

def get_shape():
    """Return the current shape (NLMS, NCELLS, NLUS) of the problem."""
    NLMS = bdata.NLMS
    NCELLS, = data.AGEC.index.shape
    NLUS = bdata.NLUS
    return NLMS, NCELLS, NLUS

# Obtain local versions of get_*_matrices.
def get_c_mrj():
    return get_cost_matrices(data, get_year())
def get_q_mrj():
    return get_quantity_matrices(data, get_year())
def get_t_mrj():
    return get_transition_matrices( data
                                  , get_year()
                                  , data.LUMAP # TODO: really: lumaps[-1][mask]
                                  , data.LMMAP # TODO: really: lmmaps[-1][mask]
                                  )
def get_x_mrj():
    return exclude(data.AGEC)

def prep():
    """Prepare for the next simulation step."""
    global data, ready
    data = Data(lumaps[-1])
    ready = True

def step( d_j # Demands.
        , p   # Penalty level.
        ):
    """Solve the upcoming time step (year)."""
    global ready
    if not is_ready(): prep()
    lumap, lmmap = solve( get_t_mrj()
                        , get_c_mrj()
                        , get_q_mrj()
                        , d_j
                        , p
                        , get_x_mrj() )
    # TODO: Below appends should not append the outputs of solve directly. They
    # should first be reconstituted to the original spatial domain. Otherwise
    # (amongst other things), the next iteration will fail to apply the mask.
    lumaps.append(lumap)
    lmmaps.append(lmmap)
    ready = False

def info():
    """Return information about state of the simulation."""
    print( "Current time step (year):"
         , str(get_year())
         , "(" + str(bdata.ANNUM + get_year()) + ")")
    if is_ready():
        print("Simulation is ready to solve next time step. Run `step()`")
    else:
        print("Simulation is not ready to solve next time step. Run `prep()`.")

def get_results(year=None):
    """Return the simulation results for `year` or all if `year is None`."""
    if year is None:
        return np.stack(lumaps), np.stack(lmmaps)
    else:
        return lumaps[year], lmmaps[year]

def demands():
    """Return demands based on current production."""
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
