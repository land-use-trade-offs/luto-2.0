#!/bin/env python3
#
# solver.py - provides minimalist Solver class and pure helper functions.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-02-22
# Last modified: 2021-07-30
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

def solve( t_mrj  # Transition cost matrices.
         , c_mrj  # Production cost matrices.
         , q_mrj  # Yield matrices.
         , d_j    # Demands.
         , p      # Penalty level.
         , x_mrj  # Exclude matrices.
         , constraints = constraints # Constraints to use (default all).
         ):
    """Return land-use, land-man maps under constraints and minimised costs.

    All inputs are Numpy arrays of the appropriate shapes, except for `p` which
    is a scalar and `constraints` which is a dictionary.

    To run with only a subset of the constraints, pass a custom `constraints`
    dictionary. Format {key: value} where 'key' is a string, one of 'water',
    'nutrients', 'carbon' or 'biodiversity' and 'value' is either True or False.
    """

    # Extract the shape of the problem (# land-man types, cells and land-uses).
    nlms, ncells, nlus = t_mrj.shape

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
                           for j in range(nlus) ) )
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

        # Water use capped, per catchment, at volume consumed in base year.
        if constraints['water']:
            ...

        if constraints['nutrients']:
            ...

        if constraints['carbon']:
            ...

        if constraints['biodiversity']:
            ...

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
