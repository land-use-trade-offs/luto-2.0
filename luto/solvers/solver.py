#!/bin/env python3
#
# solver.py - provides minimalist Solver class and pure helper functions.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-02-22
# Last modified: 2021-06-25
#

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from luto.tools import timethis
import luto.data as data

# Default constraint settings.
constraints = { 'water': True
              , 'nutrients': True
              , 'carbon': True
              , 'biodiversity': True
              }

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

def inspect(lumap, highpos, d_j, q_rj, c_rj):

    # Prepare the pandas.DataFrame.
    columns = [ "Pre Cells [#]"
              , "Pre Costs [AUD]"
              , "Pre Deviation [units]"
              , "Pre Deviation [%]"
              , "Post Cells [#]"
              , "Post Costs [AUD]"
              , "Post Deviation [units]"
              , "Post Deviation [%]"
              , "Delta Total Cells [#]"
              , "Delta Total Cells [%]"
              , "Delta Costs [AUD]"
              , "Delta Costs [%]"
              , "Delta Moved Cells [#]"
              , "Delta Deviation [units]"
              , "Delta Deviation [%]" ]
    index = data.LANDUSES
    df = pd.DataFrame(index=index, columns=columns)

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

        precost_dry = prewhere_dry @ c_rj.T[k]
        precost_irr = prewhere_irr @ c_rj.T[k+1]

        postwhere_dry = np.where(highpos == k, 1, 0)
        postwhere_irr = np.where(highpos == k+1, 1, 0)
        postcount_dry = postwhere_dry.sum()
        postcount_irr = postwhere_irr.sum()
        postquantity_dry = postwhere_dry @ q_rj.T[k]
        postquantity_irr = postwhere_irr @ q_rj.T[k+1]

        postcost_dry = postwhere_dry @ c_rj.T[k]
        postcost_irr = postwhere_irr @ c_rj.T[k+1]

        predeviation = prequantity_dry + prequantity_irr - d_j[j]
        predevfrac = predeviation / d_j[j]

        postdeviation = postquantity_dry + postquantity_irr - d_j[j]
        postdevfrac = postdeviation / d_j[j]

        df.loc[lu] = [ precount_dry
                     , precost_dry
                     , predeviation
                     , 100 * predevfrac
                     , postcount_dry
                     , postcost_dry
                     , postdeviation
                     , 100 * postdevfrac
                     , postcount_dry - precount_dry
                     , 100 * (postcount_dry - precount_dry) / precount_dry
                     , postcost_dry - precost_dry
                     , 100 * (postcost_dry - precost_dry) / precost_dry
                     , np.sum(postwhere_dry - (prewhere_dry*postwhere_dry))
                     , np.abs(postdeviation) - np.abs(predeviation)
                     , 100 * ( np.abs(postdeviation) - np.abs(predeviation)
                             / predeviation ) ]

        lu_irr = lu.replace('_dry', '_irr')
        df.loc[lu_irr] = [ precount_irr
                         , precost_irr
                         , predeviation
                         , 100 * predevfrac
                         , postcount_irr
                         , postcost_irr
                         , postdeviation
                         , 100 * postdevfrac
                         , postcount_irr - precount_irr
                         , 100 * (postcount_irr - precount_irr) / precount_irr
                         , postcost_irr - precost_irr
                         , 100 * (postcost_irr - precost_irr) / precost_irr
                         , np.sum(postwhere_irr - (prewhere_irr*postwhere_irr))
                         , np.abs(postdeviation) - np.abs(predeviation)
                         , 100 * ( np.abs(postdeviation) - np.abs(predeviation)
                                 / predeviation ) ]

        precost += prewhere_dry @ c_rj.T[k] + prewhere_irr @ c_rj.T[k+1]
        postcost += postwhere_dry @ c_rj.T[k] + postwhere_irr @ c_rj.T[k+1]

    df.loc['total'] = [df[col].sum() for col in df.columns]

    return df

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

