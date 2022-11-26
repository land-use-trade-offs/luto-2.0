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


import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

import luto.settings as settings

# Set Gurobi environment.
gurenv = gp.Env(empty = True)
gurenv.setParam('Method', settings.SOLVE_METHOD)
gurenv.setParam('OutputFlag', settings.VERBOSE)
gurenv.start()


def solve( t_mrj          # Transition cost matrices.
         , c_mrj          # Production cost matrices.
         , q_mrp          # Yield matrices -- note the `p` (product) index instead of `j` (land-use).
         , d_c            # Demands -- note the `c` ('commodity') index instead of `j` (land-use).
         , penalty        # Penalty level.
         , x_mrj          # Exclude matrices.
         , lu2pr_pj       # Conversion matrix: land-use to product(s).
         , pr2cm_cp       # Conversion matrix: product(s) to commodity.
         , limits = None  # Targets to use.
         ):
    """Return land-use and land management maps under constraints and minimised costs.

    All inputs are Numpy arrays of the appropriate shapes, except for `p` which
    is a scalar and `limits` which is a dictionary.

    To run with only a subset of the constraints, pass a custom `constraints`
    dictionary. Format {key: value} where 'key' is a string, one of 'water',
    'nutrients', 'carbon' or 'biodiversity' and 'value' is either True or False.
    """

    # Set up a timer
    start_time = time.time()

    # Ensure there is a dictionary to test against.
    if limits is None: limits = {}

    # Extract the shape of the problem.
    nlms, ncells, nlus = t_mrj.shape # Number of landmans, cells, landuses.
    _, _, nprs = q_mrp.shape # Number of products.
    ncms, = d_c.shape # Number of commodities.

    # Cost per tonne = cost per hectare / tonnes per hectare.

    # Cost per hectare in PR/p representation:
    # c_rp_dry = (lu2pr_pj @ c_mrj[0].T).T
    # c_rp_irr = (lu2pr_pj @ c_mrj[1].T).T
    # c_mrp = np.stack((c_rp_dry, c_rp_irr))

    # # Divide by quantity per hectare to obtain cost per tonne.
    # qprime_mrp = np.where(q_mrp != 0, q_mrp, np.inf) # Avoid division by zero.
    # cpert_mrp = c_mrp / qprime_mrp
    
    # # Cost per tonne in PR/p representation.
    # c_p = [cpert_mrp.T[p].max() for p in range(nprs)]
    
    # # Commodities from multiple sources get average costs.
    # p2c_cp = pr2cm_cp / pr2cm_cp.sum(axis=1)[:, np.newaxis]
    
    # # Finally, cost per tonne in CM/c representation.
    # c_c = p2c_cp @ c_p
    
    # # Apply the penalty-level multiplier.
    # p_c = penalty * c_c

    try:
        print('\nSetting up the model...', time.ctime() + '\n')

        # Make Gurobi model instance.
        model = gp.Model('LUTO 2.0', env = gurenv)

        # ------------------- #
        # Decision variables. #
        # ------------------- #

        # Land-use indexed lists of ncells-sized decision variable vectors.
        X_dry = [ model.addMVar(ncells, ub = x_mrj[0, :, j], name = 'X_dry')
                  for j in range(nlus) ]
        
        X_irr = [ model.addMVar(ncells, ub = x_mrj[1, :, j], name = 'X_irr')
                  for j in range(nlus) ]

        # Decision variables to minimise the deviations from demand.
        V = model.addMVar(ncms, name = 'V')

        # ------------------- #
        # Objective function. #
        # ------------------- #

        # Set the objective function.
        # objective = ( sum( # Production costs.
        #                    c_mrj[0].T[j] @ X_dry[j]
        #                  + c_mrj[1].T[j] @ X_irr[j]
                         
        #                    # Transition costs.
        #                  + t_mrj[0].T[j] @ X_dry[j]
        #                  + t_mrj[1].T[j] @ X_irr[j]
                         
        #                    # For all land uses.
        #                    for j in range(nlus) )
                     
        #             + sum( # Penalties.
        #                    V[c] for c in range(ncms) )
        #             )

        # BB mod 1        
        # objective = ( sum( # Production costs + transition costs.
        #                    ( c_mrj[0].T[j] + t_mrj[0].T[j] ) @ X_dry[j]
        #                  + ( c_mrj[1].T[j] + t_mrj[1].T[j] ) @ X_irr[j]
                         
        #                    # For all land uses.
        #                    for j in range(nlus) )
                     
        #             + sum( # Penalties.
        #                    V[c] for c in range(ncms) )
        #             )
        
        #BB mod 2
        # Pre-calculate sum of production and transition costs and apply penalty
        ct_mrj = (c_mrj + t_mrj) / penalty

        objective = ( sum( # Production costs + transition costs.
                         #   ct_mrj[0].T[j] @ X_dry[j]
                         # + ct_mrj[1].T[j] @ X_irr[j]
                           ct_mrj[0, :, j] @ X_dry[j]  # using matrix indexing and slicing
                         + ct_mrj[1, :, j] @ X_irr[j]
                         
                           # For all land uses.
                           for j in range(nlus) )
                     
                    + sum( # Add variables for ensuring demand of each commodity is met (approximately).
                           V[c] for c in range(ncms) )
                    )

        
        model.setObjective(objective, GRB.MINIMIZE)
        

        # ------------ #
        # Constraints. #
        # ------------ #

        # Constraint that all of every cell is used for some land use.
        model.addConstr( sum( X_dry + X_irr ) == np.ones(ncells) )
        
        # Constraints to penalise under and over production compared to demand.

        # Transform decision vars from LU/j to PR/p representation.
        X_dry_pr = [ X_dry[j] for p in range(nprs) for j in range(nlus)
                     if lu2pr_pj[p, j] ]
        X_irr_pr = [ X_irr[j] for p in range(nprs) for j in range(nlus)
                     if lu2pr_pj[p, j] ]

        # Quantities in PR/p representation by land-management (dry/irr).
        q_dry_p = [ q_mrp[0].T[p] @ X_dry_pr[p] for p in range(nprs) ]
        q_irr_p = [ q_mrp[1].T[p] @ X_irr_pr[p] for p in range(nprs) ]

        # Transform quantities to CM/c representation by land management (dry/irr).
        q_dry_c = [ sum(q_dry_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                    for c in range(ncms) ]
        q_irr_c = [ sum(q_irr_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                    for c in range(ncms) ]

        # Total quantities in CM/c representation.
        q_c = [q_dry_c[c] + q_irr_c[c] for c in range(ncms)]

        # Finally, add the constraint in the CM/c representation.
        model.addConstrs( (d_c[c] - q_c[c]) <= V[c] 
                          for c in range(ncms) )
        model.addConstrs( (q_c[c] - d_c[c]) <= V[c] 
                          for c in range(ncms) )

        # Only add the following constraints if target provided.

        if 'water' in limits:
            
            # Returns water requirements for agriculture in mrj format and region-specific water use limits
            aqreq_mrj, aqreq_limits = limits['water']
            
            # Ensure water use remains below limit for each region
            for region, aqreq_reg_limit, ind in aqreq_limits:
                
                aqreq_region = sum( aqreq_mrj[0, ind, j] @ X_dry[j][ind]
                                  + aqreq_mrj[1, ind, j] @ X_irr[j][ind]
                                    for j in range(nlus) )
                
                model.addConstr(aqreq_region <= aqreq_reg_limit)
                print('...water limit region %s <= %s ML' % (region, aqreq_reg_limit))


        if 'nutrients' in limits:
            ...

        if 'carbon' in limits:
            ...

        if 'biodiversity' in limits:
            ...

        # -------------------------- #
        # Solve and extract results. #
        # -------------------------- #

        st = time.time()
        print('Starting solve... ', time.ctime())
        
        # Magic.
        model.optimize()
        
        ft = time.time()
        print('Completed solve...', time.ctime())
        print('Found optimal objective value', round(model.objVal, 2), 'in', round(ft - st), 'seconds\n')
        
        print('Collecting results...', end = '')
              
        # Collect optimised decision variables in one X_mrj Numpy array.              
        X_dry_rj = np.stack([X_dry[j].X for j in range(nlus)])
        X_irr_rj = np.stack([X_irr[j].X for j in range(nlus)])
        X_mrj = np.stack((X_dry_rj, X_irr_rj))

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
        
        print('Done\n')
        print('Total processing time...', round(time.time() - start_time), 'seconds')
        
        return lumap, lmmap, X_mrj

    except gp.GurobiError as e:
        print('Gurobi error code', str(e.errno), ':', str(e))

    except AttributeError:
        print('Encountered an attribute error')
        



# Toy model

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
    q_mrp[:, :, -2] = 0 # An 'unallocated' landuse.
    c_mrj = 1 * np.ones((nlms, ncells, nlus))
    c_mrj[:, :, -2] = 0 # An 'unallocated' landuse.
    x_mrj = 1 * np.ones((nlms, ncells, nlus))

    p = 5

    d_c = np.zeros(ncms)
    #d_c[3] = 1

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

    lumap, lmmap, xmrj = solve(   t_mrj
                                , c_mrj
                                , q_mrp
                                , d_c
                                , p
                                , x_mrj
                                , lu2pr_pj
                                , pr2cm_cp
                                )
    print(lumap)
