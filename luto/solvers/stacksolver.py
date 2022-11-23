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
Provides minimalist Solver class and pure helper functions.
"""


import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from luto.tools import timethis

def solve( t_rj  # Transition cost to commodity j at cell r --- with lumap info.
         , tm_rj # Additional cost for management of commodity j at cell r.
         , c_rj  # Cost of producing commodity j at cell r.
         , cm_rj # Additional production cost of management of com. j at cell r.
         , q_rj  # Yield of commodity j at cell r.
         , qm_rj # Additional yield of commodity j at cell r under management.
         , d_j   # Demand for commodity j.
         , p     # Penalty level. Multiplies cost per unit surplus or deficits.
         , x_rj  # Possible/allowed land uses j at cell r.
         ):
    """Solver allowing land uses to have stacked management practices."""

    # Extract the shape of the problem.
    ncells = t_rj.shape[0]
    nlus = t_rj.shape[1]

    # Calculate the penalty units for each land use j as its maximum cost.
    p_j = np.zeros(nlus)
    for j in range(nlus):
        p_j[j] = c_rj.T[j].max()

    # Apply the multiplier.
    p_j *= p

    try:
        # Make Gurobi model instance.
        model = gp.Model('neoLUTO StackSolver v0.01')

        # Land-use indexed list of ncells-sized, cont. decision var. vectors.
        X = [ model.addMVar( ncells
                           , vtype=GRB.BINARY
                           , ub=x_rj.T[j]
                           , name='X' )
              for j in range(nlus) ]

        # Land-use management type boolean decision variable.
        M = [ model.addMVar( ncells
                           , vtype=GRB.BINARY
                           , name='M' )
              for j in range(nlus) ]

        # Decision variables to minimise the deviations.
        V = model.addMVar(nlus, name='V')

        # Set the objective function and the model sense.
        objective = sum( c_rj.T[j] @ X[j]         # Base cost of producing.
                       + t_rj.T[j] @ X[j]         # Base cost of switching.
                       + cm_rj.T[j] @ M[j]        # Management production cost.
                       + tm_rj.T[j] @ M[j]        # Management switching cost.
                       + V[j]                     # Cost of surplus/deficit.
                         for j in range(nlus) )   # For each land use.

        model.setObjective(objective, GRB.MINIMIZE)

        # Constraint that each cell is used for some land use.
        model.addConstr(sum(X) == np.ones(ncells))

        # Constraint to ensure management applies only to the selected land use.
        model.addConstrs( M[j] - X[j] <= np.zeros(ncells) for j in range(nlus))

        # Constraints that penalise deficits and surpluses, respectively.
        model.addConstrs( p_j[j] * ( d_j[j] - ( q_rj.T[j] @ X[j]
                                              + qm_rj.T[j] @ M[j] ) ) <= V[j]
                          for j in range(nlus) )
        model.addConstrs( p_j[j] * ( ( q_rj.T[j] @ X[j]
                                     + qm_rj.T[j] @ M[j] ) - d_j[j] ) <= V[j]
                          for j in range(nlus) )

        # Magic.
        model.optimize()

        # Collect optimised decision variables in tuple of 1D Numpy arrays.
        prestackhp = tuple(X[j].X for j in range(nlus))
        prestackmn = tuple(M[j].X for j in range(nlus))

        # Flatten into land-use map of the usual highpos integer array format.
        highpos = np.stack(prestackhp).argmax(axis=0)
        mngmnt = np.stack(prestackmn).sum(axis=0)

        # Return the land-use map, i.e. the highpos array.
        return highpos, mngmnt

    except gp.GurobiError as e:
        print('Gurobi error code', str(e.errno), ':', str(e))

    except AttributeError:
        print('Encountered an attribute error')

def runstack_random(ncells, nlus, p=1):
    """Run stacked model w/ rand data, penalty-level `p`, shape ncells, nlus."""

    # Bogus lumap.
    lumap = np.random.randint(nlus, size=ncells)

    # Transition cost matrix.
    t_ij = 10 * np.random.random((nlus, nlus))
    for i in range(nlus): t_ij[i, i] = 0

    t_rj = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))
    tm_rj = 0.1 * t_rj
    t_rj[:] = 0
    tm_rj[:] = 0

    # Production cost matrix.
    c_rj = 10 * np.random.random((ncells, nlus))
    cm_rj = 0 * c_rj

    # Yield matrix.
    q_rj = 10 * np.random.random((ncells, nlus))
    qm_rj = .5 * q_rj

    # Demands.
    d_j = 100 * np.random.random(nlus)

    # Exclude matrix.
    x_rj = np.ones((ncells, nlus))
    # np.random.randint(2, size=(ncells, nlus), dtype=np.int8)

    highpos, mngmnt = timethis( solve
                              , t_rj
                              , tm_rj
                              , c_rj
                              , cm_rj
                              , q_rj
                              , qm_rj
                              , d_j
                              , p
                              , x_rj )

    return highpos, mngmnt
