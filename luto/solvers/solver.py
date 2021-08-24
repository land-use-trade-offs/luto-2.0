#!/bin/env python3
#
# solver.py - provides minimalist Solver class and pure helper functions.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-02-22
# Last modified: 2021-08-24
#

import numpy as np
import pandas as pd
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

# Default constraint settings.
constraints = { 'water': True
              , 'nutrients': True
              , 'carbon': True
              , 'biodiversity': True
              }

def solve( t_mrj  # Transition cost matrices.
         , c_mrj  # Production cost matrices.
         , q_mrp  # Yield matrices -- note the `p` index instead of `j`.
         , d_c    # Demands -- note the `c` index instead of `j`.
         , p      # Penalty level.
         , x_mrj  # Exclude matrices.
         , lu2pr_pj # Conversion matrix: land-use to product(s).
         , pr2cm_cp # Conversion matrix: product(s) to commodity.
         , constraints = constraints # Constraints to use (default all).
         ):
    """Return land-use, land-man maps under constraints and minimised costs.

    All inputs are Numpy arrays of the appropriate shapes, except for `p` which
    is a scalar and `constraints` which is a dictionary.

    To run with only a subset of the constraints, pass a custom `constraints`
    dictionary. Format {key: value} where 'key' is a string, one of 'water',
    'nutrients', 'carbon' or 'biodiversity' and 'value' is either True or False.
    """

    # Extract the shape of the problem.
    nlms, ncells, nlus = t_mrj.shape # Number of landmans, cells, landuses.
    _, _, nprs = q_mrp.shape # Number of products.
    ncms, = d_c.shape # Number of commodities.

    # Penalty units for each j as maximum cost.
    p_c = np.zeros(ncms)
    for c in range(ncms):
        p_c[c] = c_mrj.T.max()
    # Apply the penalty-level multiplier.
    p_c *= p

    try:
        # Make Gurobi model instance.
        model = gp.Model('neoLUTO v0.1.0')

        # ------------------- #
        # Decision variables. #
        # ------------------- #

        # Land-use indexed lists of ncells-sized decision variable vectors.
        X_dry = [ model.addMVar(ncells, ub=x_mrj[0, :, j], name='X_dry')
                  for j in range(nlus) ]
        X_irr = [ model.addMVar(ncells, ub=x_mrj[1, :, j], name='X_irr')
                  for j in range(nlus) ]

        # Decision variables to minimise the deviations.
        V = model.addMVar(ncms, name='V')

        # ------------------- #
        # Objective function. #
        # ------------------- #

        # Set the objective function and the model sense.
        objective = ( sum( # Production costs.
                           c_mrj[0].T[j] @ X_dry[j]
                         + c_mrj[1].T[j] @ X_irr[j]
                           # Transition costs.
                         + t_mrj[0].T[j] @ X_dry[j]
                         + t_mrj[1].T[j] @ X_irr[j]
                           # For all land uses.
                           for j in range(nlus) )
                    + sum( # Penalties.
                           V[c] for c in range(ncms) )
                    )
        model.setObjective(objective, GRB.MINIMIZE)

        # ------------ #
        # Constraints. #
        # ------------ #

        # Constraint that all of every cell is used for some land use.
        model.addConstr( sum( X_dry
                            + X_irr ) == np.ones(ncells) )

        # Constraints that penalise deficits and surpluses, respectively.
        #

        # Transform decision vars from LU/j to PR/p representation.
        X_dry_pr = [ X_dry[j] for p in range(nprs) for j in range(nlus)
                     if lu2pr_pj[p, j] ]
        X_irr_pr = [ X_irr[j] for p in range(nprs) for j in range(nlus)
                     if lu2pr_pj[p, j] ]

        # Quantities in PR/p representation by land-management (dry/irr).
        q_dry_p = [ q_mrp[0].T[p] @ X_dry_pr[p] for p in range(nprs) ]
        q_irr_p = [ q_mrp[0].T[p] @ X_irr_pr[p] for p in range(nprs) ]

        # Transform quantities to CM/c representation by land-man (dry/irr).
        q_dry_c = [ sum(q_dry_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                    for c in range(ncms) ]
        q_irr_c = [ sum(q_irr_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                    for c in range(ncms) ]

        # Total quantities in CM/c representation.
        q_c = [q_dry_c[c] + q_irr_c[c] for c in range(ncms)]

        # Finally, add the constraint in the CM/c representation.
        model.addConstrs( p_c[c] * (d_c[c] - q_c[c]) <= V[c]
                          for c in range(ncms) )
        model.addConstrs( p_c[c] * (q_c[c] - d_c[c]) <= V[c]
                          for c in range(ncms) )

        # Only add the following constraints if requested.

        # Water use capped, per catchment, at volume consumed in base year.
        if constraints['water']:
            ...

        if constraints['nutrients']:
            ...

        if constraints['carbon']:
            ...

        if constraints['biodiversity']:
            ...

        # -------------------------- #
        # Solve and extract results. #
        # -------------------------- #

        # Magic.
        model.optimize()

        # Collect optimised decision variables in tuple of 1D Numpy arrays.
        prestack_dry = tuple(X_dry[j].X for j in range(nlus))
        stack_dry = np.stack(prestack_dry)
        highpos_dry = stack_dry.argmax(axis=0)

        prestack_irr = tuple(X_irr[j].X for j in range(nlus))
        stack_irr = np.stack(prestack_irr)
        highpos_irr = stack_irr.argmax(axis=0)

        lumap = np.where( stack_dry.max(axis=0) >= stack_irr.max(axis=0)
                        , highpos_dry, highpos_irr )

        lmmap = np.where( stack_dry.max(axis=0) >= stack_irr.max(axis=0)
                        , 0, 1 )

        return lumap, lmmap

    except gp.GurobiError as e:
        print('Gurobi error code', str(e.errno), ':', str(e))

    except AttributeError:
        print('Encountered an attribute error')

        # Non landuse-product-commodity version. For reference:
        # model.addConstrs( p_j[j]
                        # * ( d_j[j] - ( q_mrj[0].T[j] @ X_dry[j]
                                     # + q_mrj[1].T[j] @ X_irr[j] ) ) <= V[j]
                          # for j in range(nlus) )
        # model.addConstrs( p_j[j]
                        # * ( ( q_mrj[0].T[j] @ X_dry[j]
                            # + q_mrj[1].T[j] @ X_irr[j] ) - d_j[j] ) <= V[j]
                          # for j in range(nlus) )

if __name__ == '__main__':

    nlms = 2
    ncells = 10
    nlus = 5
    nprs = 7
    ncms = 6

    t_mrj = np.zeros((nlms, ncells, nlus))
    t_mrj = 5 * np.ones((nlms, ncells, nlus))
    t_mrj[:, :, -1] = 0 # LU == 4 everywhere.

    q_mrp = 1 * np.ones((nlms, ncells, nprs))
    c_mrj = 1 * np.ones((nlms, ncells, nlus))
    x_mrj = 1 * np.ones((nlms, ncells, nlus))

    p = 1000

    d_c = np.zeros(ncms)
    d_c[3] = 1

    lu2pr_pj = np.array([ [1, 0, 0, 0, 0]
                        , [1, 0, 0, 0, 0]
                        , [0, 1, 0, 0, 0]
                        , [0, 1, 0, 0, 0]
                        , [0, 0, 1, 0, 0]
                        , [0, 0, 0, 1, 0]
                        , [0, 0, 0, 0, 1]
                        ])

    pr2cm_cp = np.array([ [1, 0, 0, 0, 0, 0, 0]
                        , [0, 1, 1, 0, 0, 0, 0]
                        , [0, 0, 0, 1, 0, 0, 0]
                        , [0, 0, 0, 0, 1, 0, 0]
                        , [0, 0, 0, 0, 0, 1, 0]
                        , [0, 0, 0, 0, 0, 0, 1]
                        ])

    lumap, lmmap = solve( t_mrj
                        , c_mrj
                        , q_mrp
                        , d_c
                        , p
                        , x_mrj
                        , lu2pr_pj
                        , pr2cm_cp
                        , constraints = constraints
                        )
    print(lumap)
