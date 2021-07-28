#!/bin/env python3
#
# solver.py - provides minimalist Solver class and pure helper functions.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-02-22
# Last modified: 2021-07-28
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

    # Determine the shape of the output array as well as the dtype.
    if presize is None: presize = resfactor * array.shape[0]
    bshape = list(array.shape)
    bshape[0] = presize
    bshape = tuple(bshape)
    brray = np.ones(bshape, dtype=array.dtype)

    # Output will be length presize. Second for-loop to fill out tail end.
    for i in range(0, len(brray) - resfactor, resfactor):
        for k in range(resfactor):
            brray[i+k] = array[i // resfactor]
    i += resfactor
    for k in range(presize - (array.shape[0]-1)*resfactor):
        brray[i+k] = array[i // resfactor]

    return brray

def solve( t_mrj  # Transition cost matrices.
         , c_mrj  # Production cost matrices.
         , q_mrj  # Yield matrices.
         , d_j    # Demands.
         , p      # Penalty level.
         , x_mrj  # Exclude matrices.
         , constraints = constraints # Constraints to use (default all).
         ):
    """Return land-use, land-man maps under constraints and minimised costs."""

    # Extract the shape of the problem.
    nlms = t_mrj.shape[0]   # Number of land-management types (m index).
    ncells = t_mrj.shape[1] # Number of cells in spatial domain (r index).
    nlus = t_mrj.shape[2]   # Number of land-uses (j index).

    # Penalty units for each j as maximum cost.
    p_j = np.zeros(nlus)
    for j in range(nlus):
        p_j[j] = c_mrj.T.max()
    # Apply the penalty-level multiplier.
    p_j *= p


    try:
        # Make Gurobi model instance.
        model = gp.Model('neoLUTO v0.1.0')

        # Land-use indexed lists of ncells-sized decision variable vectors.
        X_dry = [ model.addMVar(ncells, ub=x_mrj[0, :, j], name='X_dry')
                  for j in range(nlus) ]
        X_irr = [ model.addMVar(ncells, ub=x_mrj[1, :, j], name='X_irr')
                  for j in range(nlus) ]

        # Decision variables to minimise the deviations.
        V = model.addMVar(nlus, name='V')

        # Set the objective function and the model sense.
        objective = ( sum( # Production costs.
                           c_mrj[0].T[j] @ X_dry[j]
                         + c_mrj[1].T[j] @ X_irr[j]
                           # Transition costs.
                         + t_mrj[0].T[j] @ X_dry[j]
                         + t_mrj[1].T[j] @ X_irr[j]
                           # Penalties.
                         + V[j]
                           # For all land uses.
                           for j in range(nlus) )
                    )
        model.setObjective(objective, GRB.MINIMIZE)

        # Constraint that all of every cell is used for some land use.
        model.addConstr( sum( X_dry
                            + X_irr ) == np.ones(ncells) )

        # Constraints that penalise deficits and surpluses, respectively.
        model.addConstrs( p_j[j]
                        * ( d_j[j] - ( q_mrj[0].T[j] @ X_dry[j]
                                     + q_mrj[1].T[j] @ X_irr[j] ) ) <= V[j]
                          for j in range(nlus) )
        model.addConstrs( p_j[j]
                        * ( ( q_mrj[0].T[j] @ X_dry[j]
                            + q_mrj[1].T[j] @ X_irr[j] ) - d_j[j] ) <= V[j]
                          for j in range(nlus) )

        # Only add the following constraints if requested.

        # Magic.
        model.optimize()

    except gp.GurobiError as e:
        print('Gurobi error code', str(e.errno), ':', str(e))

    except AttributeError:
        print('Encountered an attribute error')



def solve_old( t_rj  # Transition cost to commodity j at cell r - w lumap info.
             , c_rj  # Cost of producing commodity j at cell r.
             , q_rj  # Yield of commodity j at cell r.
             , d_j   # Demand for commodity j.
             , p     # Penalty level. Multiplies cost/unit surplus or deficits.
             , x_rj  # Possible/allowed land uses j at cell r.
             , constraints = constraints # Constraints to use (default all).
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

