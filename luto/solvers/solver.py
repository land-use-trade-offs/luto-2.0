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
gurenv.setParam('OptimalityTol', settings.OPTIMALITY_TOLERANCE)
# gurenv.setParam('Threads', settings.THREADS)


def solve( t_mrj          # Transition cost matrices.
         , c_mrj          # Production cost matrices.
         , r_mrj          # Production revenue matrices.
         , g_mrj          # Greenhouse gas emissions matrices.
         , w_mrj          # Water requirements matrices.
         , x_mrj          # Exclude matrices.
         , q_mrp          # Yield matrices -- note the `p` (product) index instead of `j` (land-use).
         , d_c            # Demands -- note the `c` ('commodity') index instead of `j` (land-use).
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


    try:
        print('\nSetting up the model...', time.ctime() + '\n')

        # Make Gurobi model instance.
        model = gp.Model('LUTO ' + settings.VERSION, env = gurenv)
        
        # ------------------- #
        # Decision variables. #
        # ------------------- #
        print('Adding decision variables...', time.ctime() + '\n')
        
        # Land-use indexed lists of ncells-sized decision variable vectors.
        X_dry = [ model.addMVar(ncells, ub = x_mrj[0, :, j], name = 'X_dry')
                  for j in range(nlus) ]
        
        X_irr = [ model.addMVar(ncells, ub = x_mrj[1, :, j], name = 'X_irr')
                  for j in range(nlus) ]

        # Decision variables, one for each commodity, to minimise the deviations from demand.
        V = model.addMVar(ncms, name = 'V')
        
        
        
        # ------------------- #
        # Objective function. #
        # ------------------- #
        print('Setting up objective function to %s...' % settings.OBJECTIVE, time.ctime() + '\n')
        
      
        if settings.OBJECTIVE == 'maximise revenue':
                    
            # Pre-calculate revenue minus (production and transition) costs
            obj_mrj = -( r_mrj - (c_mrj + t_mrj) ) / settings.PENALTY
         
        elif settings.OBJECTIVE == 'minimise cost':
            
            # Pre-calculate sum of production and transition costs
            obj_mrj = (c_mrj + t_mrj) / settings.PENALTY
            
        else:
            print('Unknown objective')

            
        # Specify objective function
        objective = ( 
                     # Production costs + transition costs for all land uses.
                     sum( obj_mrj[0, :, j] @ X_dry[j]
                        + obj_mrj[1, :, j] @ X_irr[j]
                          for j in range(nlus) )
                    
                     # Add deviation-from-demand variables for ensuring demand of each commodity is met (approximately). 
                     + sum( V[c] for c in range(ncms) )
                    ) 

        model.setObjective(objective, GRB.MINIMIZE)
        

        # ------------ #
        # Constraints. #
        # ------------ #
        print('Setting up constraints...', time.ctime() + '\n')
        
        # Constraint that all of every cell is used for some land use.
        model.addConstr( sum( X_dry + X_irr ) == np.ones(ncells) )
        
        # Constraints to penalise under and over production compared to demand.

        # Transform decision vars from LU/j to PR/p representation.
        X_dry_pr = [ X_dry[j] for p in range(nprs) for j in range(nlus)
                     if lu2pr_pj[p, j] ]
        X_irr_pr = [ X_irr[j] for p in range(nprs) for j in range(nlus)
                     if lu2pr_pj[p, j] ]

        # Quantities in PR/p representation by land-management (dry/irr).
        q_dry_p = [ q_mrp[0, :, p] @ X_dry_pr[p] for p in range(nprs) ]
        q_irr_p = [ q_mrp[1, :, p] @ X_irr_pr[p] for p in range(nprs) ]

        # Transform quantities to CM/c representation by land management (dry/irr).
        q_dry_c = [ sum(q_dry_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                    for c in range(ncms) ]
        q_irr_c = [ sum(q_irr_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                    for c in range(ncms) ]

        # Total quantities in CM/c representation.
        q_c = [ q_dry_c[c] + q_irr_c[c] for c in range(ncms) ]

        # Finally, add the constraint in the CM/c representation.
        print('Adding constraints...', time.ctime() + '\n')
        
        model.addConstrs((d_c[c] - q_c[c]) <= V[c] 
                          for c in range(ncms) )
        model.addConstrs((q_c[c] - d_c[c]) <= V[c] 
                          for c in range(ncms) )
        
        

        # Only add the following constraints if 'limits' are provided.

        if 'water' in limits:
            print('Adding water constraints by', settings.WATER_REGION_DEF + '...', time.ctime())
            
            # Returns water requirements for agriculture in mrj format and region-specific water use limits
            w_limits = limits['water']
            
            # Ensure water use remains below limit for each region
            for region, wreq_reg_limit, ind in w_limits:
                
                wreq_region = sum( w_mrj[0, ind, j] @ X_dry[j][ind]
                                 + w_mrj[1, ind, j] @ X_irr[j][ind]
                                   for j in range(nlus) )
                
                model.addConstr(wreq_region <= wreq_reg_limit)
                if settings.VERBOSE == 1:
                    print('    ...setting water limit for %s <= %.2f ML' % (region, wreq_reg_limit))


        if 'ghg' in limits:
            print('\nAdding GHG emissions constraints...', time.ctime() + '\n')
            
            # Returns GHG emissions limits 
            ghg_limits = limits['ghg']
                
            ghg_emissions = sum( g_mrj[0, :, j] @ X_dry[j]
                               + g_mrj[1, :, j] @ X_irr[j]
                                 for j in range(nlus) )
            
            print('    ...setting {:,.0f}% GHG emissions reduction target: {:,.0f} tCO2e\n'.format(
                          settings.GHG_REDUCTION_PERCENTAGE, ghg_limits )
                 )
            model.addConstr(ghg_emissions <= ghg_limits)

            

        if 'biodiversity' in limits:
            ...
            
        if 'nutrients' in limits:
            ...


        # -------------------------- #
        # Solve and extract results. #
        # -------------------------- #

        st = time.time()
        print('Starting solve... ', time.ctime(), '\n')
        
        # Magic.
        model.optimize()
        
        ft = time.time()
        print('Completed solve... ', time.ctime())
        print('Found optimal objective value', round(model.objVal, 2), 'in', round(ft - st), 'seconds\n')
        
        print('Collecting results...', end = ' ')
        
        # Collect optimised decision variables in one X_mrj Numpy array.              
        X_dry_rj = np.stack([X_dry[j].X for j in range(nlus)]).T.astype(np.float32)
        X_irr_rj = np.stack([X_irr[j].X for j in range(nlus)]).T.astype(np.float32)

        
        """Note that output decision variables are mostly 0 or 1 but in some cases they are somewhere in between which creates issues 
           when converting to maps etc. as individual cells can have non-zero values for multiple land-uses and land management type.
           This code creates a boolean X_mrj output matrix and ensure that each cell has one and only one land-use and land management"""
        
        # Stack dryland and irrigated decision variables and get the shape
        X_mrj = np.stack((X_dry_rj, X_irr_rj)) # Float32
        X_mrj_shape = X_mrj.shape
        
        # Reshape so that cells are along the first axis and land management and use are flattened along second axis i.e. (XXXXXXX, 56)
        X_mrj = np.moveaxis(X_mrj, 1, 0)
        X_mrj = X_mrj.reshape(X_mrj.shape[0], -1)
        
        # Boolean matrix where the maximum value for each cell across all land management types and land uses is True
        X_mrj = X_mrj.argmax(axis = 1)[:, np.newaxis] == range(X_mrj.shape[1])
        
        # Reshape to mrj structure
        X_mrj = X_mrj.reshape((X_mrj_shape[1], X_mrj_shape[0], X_mrj_shape[2]))
        X_mrj = np.moveaxis(X_mrj, 0, 1)

        
        # Calculate 1D array (maps) of land-use and land management
        lumap = X_mrj.sum(axis = 0).argmax(axis = 1).astype('int8')
        lmmap = X_mrj.sum(axis = 2).argmax(axis = 0).astype('int8')
        
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
