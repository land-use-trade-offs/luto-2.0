#!/bin/env python3
#
# solver.py - provides minimalist Solver class and pure helper functions.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-02-22
# Last modified: 2021-07-06
#

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Default constraint settings.
constraints = { 'water': True
              , 'nutrients': True
              , 'carbon': True
              , 'biodiversity': True
              }

def coursify(array, resfactor):
    """Return course-grained version of array, coursened by `resfactor`."""
    # Every index * `resfactor` is sampled. New array length varies.
    return array[::resfactor]

def uncoursify(array, resfactor, presize=None):
    """Return array inflated by `resfactor`. Inverse of `coursify()`."""

    r, j = array.shape
    dtype = array.dtype

    if presize is None:
        bshape = (r * resfactor, j)
        brray = np.ones(bshape, dtype=dtype)
    else:
        brray = np.ones((presize, j), dtype=dtype)

    for i in range(0, len(brray), resfactor):
        for k in range(resfactor):
            brray[i+k] = array[i // resfactor]

    return brray






def solve( t_rj  # Transition cost to commodity j at cell r --- with lumap info.
         , c_rj  # Cost of producing commodity j at cell r.
         , q_rj  # Yield of commodity j at cell r.
         , d_j   # Demand for commodity j.
         , p     # Penalty level. Multiplies cost per unit surplus or deficits.
         , x_rj  # Possible/allowed land uses j at cell r.
         , constraints = constraints # Which constraints to use (default all).
         ):
    """Return new lu map. Least-cost, surplus/deficit penalties as costs.

    Note that the t_rj transition-cost matrix _includes_ the information of the
    current land-use map, which therefore is not passed seperately.

    All inputs are Numpy arrays of the appropriate shapes, except for p which
    is a scalar multiplier.

    To run with only a subset of the constraints, pass a custom `constraints`
    dictionary. Format {key: value} where 'key' is a string, one of 'water',
    'nutrients', 'carbon' or 'biodiversity' and 'value' is either True or False.
    """

    # Extract the shape of the problem.
    ncells = t_rj.shape[0]
    nlus = t_rj.shape[1]

    # Calculate the penalty units for each land use j as its maximum cost.
    p_j = np.zeros(nlus // 2)
    for j in range(nlus // 2):
        k = 2 * j
        p_j[j] = max( c_rj.T[k].max()
                    , c_rj.T[k+1].max() )

    # Apply the multiplier.
    p_j *= p

    try:
        # Make Gurobi model instance.
        model = gp.Model('neoLUTO v0.03')
        # model.Params.Method = 1    # Use dual simplex only. Save memory.

        # Land-use indexed list of ncells-sized decision variable vectors.
        X = [model.addMVar(ncells, ub=x_rj.T[j], name='X') for j in range(nlus)]

        # Decision variables to minimise the deviations.
        V = model.addMVar(nlus // 2, name='V')

        # Set the objective function and the model sense.
        objective = ( sum( c_rj.T[j] @ X[j]              # Cost of producing.
                         + t_rj.T[j] @ X[j]              # Cost of switching.
                           for j in range(nlus) )        # For each land use.
                    + sum( V[k]
                           for k in range(nlus // 2) ) # Cost of surpl/defct.
                    )
        model.setObjective(objective, GRB.MINIMIZE)

        # Constraint that all of every cell is used for some land use.
        model.addConstr(sum(X) == np.ones(ncells))

        # Constraints that penalise deficits and surpluses, respectively.
        # Demands per actual land-use, irrespective of irrigation status.
        model.addConstrs( p_j[j]
                        * ( d_j[j] - ( q_rj.T[2*j]   @ X[2*j]
                                     + q_rj.T[2*j+1] @ X[2*j+1] ) ) <= V[j]
                          for j in range(nlus // 2) )
        model.addConstrs( p_j[j]
                        * ( ( q_rj.T[2*j]   @ X[2*j]
                            + q_rj.T[2*j+1] @ X[2*j+1] ) - d_j[j] ) <= V[j]
                          for j in range(nlus // 2) )

        # Only add the following constraints if requested.

        # Water use capped, per catchment, at volume consumed in base year.
        if constraints['water']:
            ...
            # # For each catchment
            # for c, cap in watercaps:
                # model.addConstrs( w_rj.T[2*j+1] @ X[2*j+1] <= cap
                                  # for j in range(nlus // 2) )

        if constraints['nutrients']:
            ...

        if constraints['carbon']:
            ...

        if constraints['biodiversity']:
            ...

        # Magic.
        model.optimize()

        # Collect optimised decision variables in tuple of 1D Numpy arrays.
        prestack = tuple(X[j].X for j in range(nlus))

        # Flatten into land-use map of the usual highpos integer array format.
        highpos = np.stack(prestack).argmax(axis=0)

        # Return the land-use map, i.e. the highpos array.
        return highpos #, np.stack(prestack)

    except gp.GurobiError as e:
        print('Gurobi error code', str(e.errno), ':', str(e))

    except AttributeError:
        print('Encountered an attribute error')

