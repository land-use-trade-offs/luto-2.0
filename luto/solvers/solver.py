#!/bin/env python3
#
# solver.py - provides minimalist Solver class and pure helper functions.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-02-22
# Last modified: 2021-05-14
#

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from luto.tools import timethis
import luto.data as data

def solve( lumap # Present land-use map in highpos format.
         , t_ij  # Transition cost from commodity i to j at any cell.
         , c_rj  # Cost of producing commodity j at cell r.
         , q_rj  # Yield of commodity j at cell r.
         , d_j   # Demand for commodity j.
         , p_j   # Penalty level or cost/unit surplus or deficit of commodity j.
         , x_rj  # Possible/allowed land uses j at cell r.
         , pen_norm = False # If `True`, normalise penalties against demands.
         ):
    """Return new lu map. Least-cost, surplus/deficit penalties as costs.

    All inputs are Numpy arrays of the appropriate shapes, except for p_j which
    may either be a scalar --- in which case that value is assumed for all j ---
    or a numpy array. If `pen_norm` is set to `True` the penalties will be
    normalised by dividing out the demands. This would typically be
    accompanied by a high, scalar, value for p_j.
    """

    # Extract the shape of the problem.
    nlus = t_ij.shape[0]
    ncells = lumap.size

    # Obtain transition costs to commodity j at cell r using present lumap.
    t_rj = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))

    # Assume p_j is only passed as a scalar or a Numpy array.
    if type(p_j) != np.ndarray:
        p_j = np.repeat(p_j, nlus // 2)

    # If `pen_norm` is `True` perform the normalisation.
    if pen_norm:
        d_j_safe = np.where(d_j == 0, 1e-10, d_j) # Avoid division by zero.
        p_j = p_j / d_j_safe

    try:
        # Make Gurobi model instance.
        model = gp.Model('PseudoLUTO demdriv/comswitch v0.02')
        # model.Params.Method = 1    # Use dual simplex only. Save memory.

        # Land-use indexed list of ncells-sized decision variable vectors.
        X = [model.addMVar(ncells, ub=x_rj.T[j], name='X') for j in range(nlus)]

        # Decision variables to minimise the deviations.
        V = model.addMVar(nlus // 2, name='V')

        # Set the objective function and the model sense.
        objective = ( sum( c_rj.T[j] @ X[j]              # Cost of producing.
                         + t_rj.T[j] @ X[j]              # Cost of switching.
                         # + V[j]
                           for j in range(nlus) )        # For each land use.
                    + sum( V[k]
                           for k in range(nlus // 2) ) # Cost of surpl/defct.
                    )
        model.setObjective(objective, GRB.MINIMIZE)

        # Constraint that all of every cell is used for some land use.
        model.addConstr(sum(X) == np.ones(ncells))

        # Constraints that penalise deficits and surpluses, respectively.
        # model.addConstrs( p_j[j] * (d_j[j] - q_rj.T[j] @ X[j]) <= V[j]
                          # for j in range(nlus) )
        # model.addConstrs( p_j[j] * (q_rj.T[j] @ X[j] - d_j[j]) <= V[j]
                          # for j in range(nlus) )

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

def inspect(lumap, highpos, d_j, q_rj, c_rj):

    # Prepare the pandas.DataFrame.
    columns = [ "Cells [#]"
              , "Excess [units]"
              , "Excess [%]"
              , "Cells [#]"
              , "Excess [units]"
              , "Excess [%]"
              , "Cells [#]"
              , "Cells [%]"
              , "Excess [units]"
              , "Excess [%]" ]
    categories = [ "Input LU Map"
                 , "Input LU Map"
                 , "Input LU Map"
                 , "Output LU Map"
                 , "Output LU Map"
                 , "Output LU Map"
                 , "Delta"
                 , "Delta"
                 , "Delta"
                 , "Delta" ]
    mcols = pd.MultiIndex.from_arrays([categories, columns])
    index = data.LANDUSES
    df = pd.DataFrame(index=index, columns=mcols)

    precost = 0
    postcost = 0

    # Reduced land-use list with half of the land-uses.
    lus = [lu for lu in data.LANDUSES if '_dry' in lu]

    for j, lu in enumerate(lus):
        k = 2*j
        prewhere_dry = np.where(lumap == k, 1, 0)
        prewhere_irr = np.where(lumap == k+1, 1, 0)
        precount_dry = prewhere_dry.sum()
        precount_irr = prewhere_irr.sum()
        prequantity_dry = prewhere_dry @ q_rj.T[k]
        prequantity_irr = prewhere_irr @ q_rj.T[k+1]

        postwhere_dry = np.where(highpos == k, 1, 0)
        postwhere_irr = np.where(highpos == k+1, 1, 0)
        postcount_dry = postwhere_dry.sum()
        postcount_irr = postwhere_irr.sum()
        postquantity_dry = postwhere_dry @ q_rj.T[k]
        postquantity_irr = postwhere_irr @ q_rj.T[k+1]


        predeviation = prequantity_dry + prequantity_irr - d_j[j]
        predevfrac = predeviation / d_j[j]

        postdeviation = postquantity_dry + postquantity_irr - d_j[j]
        postdevfrac = postdeviation / d_j[j]

        df.loc[lu] = [ precount_dry
                     , predeviation
                     , predevfrac
                     , postcount_dry
                     , postdeviation
                     , postdevfrac
                     , postcount_dry - precount_dry
                     , 100 * (postcount_dry - precount_dry) / precount_dry
                     , postdeviation - predeviation
                     , 100 * (postdeviation - predeviation) / predeviation ]

        lu_irr = lu.replace('_dry', '_irr')
        df.loc[lu_irr] = [ precount_irr
                         , predeviation
                         , predevfrac
                         , postcount_irr
                         , postdeviation
                         , postdevfrac
                         , postcount_irr - precount_irr
                         , 100 * (postcount_irr - precount_irr) / precount_irr
                         , postdeviation - predeviation
                         , 100 * (postdeviation - predeviation) / predeviation ]

        precost += prewhere_dry @ c_rj.T[k] + prewhere_irr @ c_rj.T[k+1]
        postcost += postwhere_dry @ c_rj.T[k] + postwhere_irr @ c_rj.T[k+1]

    return df, precost, postcost

if __name__ == '__main__':

    # Three commodities, ten cells.
    #

    nlus = 24 + 8    # agricultural commodities + alternative land uses.
    ncells = 2000

    # A simple land-use map.
    lumap = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    # lumap = np.random.randint(nlus, size=ncells)

    # Transition cost matrix. Note it does not depend on r.
    t_ij = np.array( [ [ 0, 2, 3 ]
                     , [ 1, 0, 2 ]
                     , [ 1, 1, 0 ] ] ).T

    # Production cost matrix. Note it does not depend on r.
    c_rj = np.array( [ [3, 2, 3]
                     , [3, 2, 3]
                     , [3, 2, 3]
                     , [3, 2, 3]
                     , [3, 2, 3]
                     , [3, 2, 3]
                     , [3, 2, 3]
                     , [3, 2, 3]
                     , [3, 2, 3]
                     , [3, 2, 3] ] )

    # Yield matrix. Note it does not depend on r.
    q_rj = np.array( [ [1, 1, 1]
                     , [1, 1, 1]
                     , [1, 1, 1]
                     , [1, 1, 1]
                     , [1, 1, 1]
                     , [1, 1, 1]
                     , [1, 1, 1]
                     , [1, 1, 1]
                     , [1, 1, 1]
                     , [1, 1, 1] ] )

    # Demands.
    d_j = np.array([1, 1, 1])

    # Penalties for surpluses and deficits, respectively.
    p_j = 10 * np.random.random(nlus)

    # Shape of the problem.
    nlus = d_j.size
    ncells = lumap.size

    def run(ncells, nlus):
        # A simple land-use map.
        lumap = np.random.randint(nlus, size=ncells)

        # Transition cost matrix. Note it does not depend on r.
        t_ij = 10 * np.random.random((nlus, nlus))
        for i in range(nlus): t_ij[i, i] = 0
        # t_ij = np.zeros((nlus, nlus))

        # Production cost matrix. Note it does not depend on r.
        c_rj = 10 * np.random.random((ncells, nlus))
        # c_rj[:, 0] = 0

        # Yield matrix. Note it does not depend on r.
        q_rj = 10 * np.random.random((ncells, nlus))

        # Demands.
        d_j = np.zeros(nlus)
        # d_j = 10 * np.random.random(nlus)

        # Penalties for surpluses and deficits, respectively.
        p_j = np.zeros(nlus)
        # p_j = 10 * np.random.random(nlus)

        for j in range(nlus):
            d_j[j] = q_rj.T[j] @ np.where(lumap == j, 1, 0)
            p_j[j] = c_rj.T[j].mean()

        x_rj = np.ones((ncells, nlus))
        # x_rj = np.random.randint(2, size=(ncells, nlus), dtype=np.int8)
        highpos = solve( lumap
                       , t_ij
                       , c_rj
                       , q_rj
                       , d_j
                       , p_j
                       , x_rj )

        return inspect(lumap, highpos, d_j, q_rj, c_rj)

        # return timethis( solve
                       # , lumap
                       # , t_ij
                       # , c_rj
                       # , q_rj
                       # , d_j
                       # , p_j
                       # , x_rj )

