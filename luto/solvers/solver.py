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
gurenv = gp.Env(logfilename = 'gurobi.log', empty = True) # (empty = True)
gurenv.setParam('Method', settings.SOLVE_METHOD)
gurenv.setParam('OutputFlag', settings.VERBOSE)
gurenv.setParam('OptimalityTol', settings.OPTIMALITY_TOLERANCE)
gurenv.setParam('Threads', settings.THREADS)
gurenv.start()


def solve( ag_t_mrj                  # Agricultural transition cost matrices.
         , ag_c_mrj                # Agricultural production cost matrices.
         , ag_r_mrj                # Agricultural production revenue matrices.
         , ag_g_mrj                # Agricultural greenhouse gas emissions matrices.
         , ag_w_mrj                # Agricultural water requirements matrices.
         , ag_x_mrj                # Agricultural exclude matrices.
         , ag_q_mrp                # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
         , ag_ghg_t_mrj            # GHG emissions released during transitions between agricultural land uses.
         , ag_to_non_ag_t_rk       # Agricultural to non-agricultural transition cost matrix.
         , non_ag_to_ag_t_mrj      # Non-agricultural to agricultural transition cost matrices.
         , non_ag_c_rk             # Non-agricultural production cost matrix.
         , non_ag_r_rk             # Non-agricultural revenue matrix.
         , non_ag_g_rk             # Non-agricultural greenhouse gas emissions matrix.
         , non_ag_w_rk             # Non-agricultural water requirements matrix.
         , non_ag_x_rk             # Non-agricultural exclude matrices.
         , non_ag_q_crk            # Non-agricultural yield matrix.
         , d_c                     # Demands -- note the `c` ('commodity') index instead of `j` (land-use).
         , lu2pr_pj                # Conversion matrix: land-use to product(s).
         , pr2cm_cp                # Conversion matrix: product(s) to commodity.
         , limits = None           # Targets to use.
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
    n_ag_lms, ncells, n_ag_lus = ag_t_mrj.shape # Number of agricultural landmans, cells, agricultural landuses.
    _, n_non_ag_lus = non_ag_c_rk.shape         # Number of non-agricultural landuses.

    _, _, nprs = ag_q_mrp.shape # Number of products.
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
        X_ag_dry = [ model.addMVar(ncells, ub =ag_x_mrj[0, :, j], name ='X_ag_dry')
                  for j in range(n_ag_lus) ]
        
        X_ag_irr = [ model.addMVar(ncells, ub =ag_x_mrj[1, :, j], name ='X_ag_irr')
                  for j in range(n_ag_lus) ]

        # Non-agricultural decision variable vectors
        X_non_ag = [ model.addMVar(ncells, ub =non_ag_x_rk[:, k], name ='X_ep')
                  for k in range(n_non_ag_lus) ]

        # Decision variables, one for each commodity, to minimise the deviations from demand.
        V = model.addMVar(ncms, name = 'V')
        
        
        
        # ------------------- #
        # Objective function. #
        # ------------------- #
        print('Setting up objective function to %s...' % settings.OBJECTIVE, time.ctime() + '\n')
        
      
        if settings.OBJECTIVE == 'maxrev':
                    
            # Pre-calculate revenue minus (production and transition) costs
            ag_obj_mrj = -(ag_r_mrj - (ag_c_mrj + ag_t_mrj + non_ag_to_ag_t_mrj)) / settings.PENALTY
            non_ag_obj_rk = -(non_ag_r_rk - (non_ag_c_rk + ag_to_non_ag_t_rk)) / settings.PENALTY
         
        elif settings.OBJECTIVE == 'mincost':
            
            # Pre-calculate sum of production and transition costs
            ag_obj_mrj = (ag_c_mrj + ag_t_mrj + non_ag_to_ag_t_mrj) / settings.PENALTY
            non_ag_obj_rk = (non_ag_c_rk + ag_to_non_ag_t_rk) / settings.PENALTY
            
        else:
            print('Unknown objective')

        # Specify objective function
        objective = ( 
                     # Production costs + transition costs for all agricultural land uses.
                     sum( ag_obj_mrj[0, :, j] @ X_ag_dry[j]
                        + ag_obj_mrj[1, :, j] @ X_ag_irr[j]
                          for j in range(n_ag_lus) )

                     # Production costs + transition costs for all non-agricultural land uses.
                     # Environmental plantings:
                   + sum( non_ag_obj_rk[:, k] @ X_non_ag[k]
                          for k in range(n_non_ag_lus) )
                    
                     # Add deviation-from-demand variables for ensuring demand of each commodity is met (approximately). 
                   + sum( V[c] for c in range(ncms) )
                    ) 

        model.setObjective(objective, GRB.MINIMIZE)
        

        # ------------ #
        # Constraints. #
        # ------------ #
        print('Setting up constraints...', time.ctime() + '\n')
        
        # Constraint that all of every cell is used for some land use.
        model.addConstr( sum(X_ag_dry + X_ag_irr + X_non_ag ) == np.ones(ncells) )
        
        # Constraints to penalise under and over production compared to demand.

        # Transform agricultural decision vars from LU/j to PR/p representation.
        X_dry_pr = [ X_ag_dry[j] for p in range(nprs) for j in range(n_ag_lus)
                     if lu2pr_pj[p, j] ]
        X_irr_pr = [ X_ag_irr[j] for p in range(nprs) for j in range(n_ag_lus)
                     if lu2pr_pj[p, j] ]

        # Quantities in PR/p representation by land-management (dry/irr).
        ag_q_dry_p = [ag_q_mrp[0, :, p] @ X_dry_pr[p] for p in range(nprs)]
        ag_q_irr_p = [ag_q_mrp[1, :, p] @ X_irr_pr[p] for p in range(nprs)]

        # Transform quantities to CM/c representation by land management (dry/irr).
        ag_q_dry_c = [ sum(ag_q_dry_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                    for c in range(ncms) ]
        ag_q_irr_c = [ sum(ag_q_irr_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                    for c in range(ncms) ]

        # Add non-agricultural commodity contributions
        # Environmental plantings:
        non_ag_q_c = [ sum(non_ag_q_crk[c, :, [k]] @ X_non_ag[k] for k in range(n_non_ag_lus))
                    for c in range(ncms) ]

        # Total quantities in CM/c representation.
        total_q_c = [ ag_q_dry_c[c] + ag_q_irr_c[c] + non_ag_q_c[c] for c in range(ncms) ]

        # Finally, add the constraint in the CM/c representation.
        print('Adding constraints...', time.ctime() + '\n')
        
        model.addConstrs((d_c[c] - total_q_c[c]) <= V[c]
                          for c in range(ncms) )
        model.addConstrs((total_q_c[c] - d_c[c]) <= V[c]
                          for c in range(ncms) )
        
        

        # Only add the following constraints if 'limits' are provided.

        if settings.WATER_USE_LIMITS == 'on':
            print('Adding water constraints by', settings.WATER_REGION_DEF + '...', time.ctime())

            # Returns water requirements for agriculture in mrj format and region-specific water use limits
            w_limits = limits['water']

            # Ensure water use remains below limit for each region
            for region, wreq_reg_limit, ind in w_limits:

                wreq_region = (
                    sum(ag_w_mrj[0, ind, j] @ X_ag_dry[j][ind]   # Dryland agriculture contribution
                      + ag_w_mrj[1, ind, j] @ X_ag_irr[j][ind]   # Irrigated agriculture contribution
                        for j in range(n_ag_lus))

                  + sum( non_ag_w_rk[ind, k] @ X_non_ag[k][ind]  # Non-agricultural contribution
                        for k in range(n_non_ag_lus))
                )

                model.addConstr(wreq_region <= wreq_reg_limit)
                if settings.VERBOSE == 1:
                    print('    ...setting water limit for %s <= %.2f ML' % (region, wreq_reg_limit))


        if settings.GHG_EMISSIONS_LIMITS == 'on':
            print('\nAdding GHG emissions constraints...', time.ctime() + '\n')
            
            # Returns GHG emissions limits 
            ghg_limits = limits['ghg']

            g_dry_contr = ag_g_mrj[0, :, :] + ag_ghg_t_mrj[0, :, :]
            g_irr_contr = ag_g_mrj[1, :, :] + ag_ghg_t_mrj[1, :, :]

            ghg_emissions = (
                sum(g_dry_contr[:, j] @ X_ag_dry[j]     # Dryland agriculture contribution
                  + g_irr_contr[:, j] @ X_ag_irr[j]     # Irrigated agriculture contribution
                    for j in range(n_ag_lus) )

              + sum(non_ag_g_rk[:, k] @ X_non_ag[k]      # Non-agricultural contribution
                    for k in range(n_non_ag_lus) )
            )
            
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
        X_dry_rj = np.stack([X_ag_dry[j].X for j in range(n_ag_lus)]).T.astype(np.float32)
        X_irr_rj = np.stack([X_ag_irr[j].X for j in range(n_ag_lus)]).T.astype(np.float32)
        non_ag_X_rk = np.stack([X_non_ag[k].X for k in range(n_non_ag_lus)]).T.astype(np.float32)

        
        """Note that output decision variables are mostly 0 or 1 but in some cases they are somewhere in between which creates issues 
           when converting to maps etc. as individual cells can have non-zero values for multiple land-uses and land management type.
           This code creates a boolean X_mrj output matrix and ensure that each cell has one and only one land-use and land management"""

        # Process agricultural land usage information
        # Stack dryland and irrigated decision variables
        ag_X_mrj = np.stack((X_dry_rj, X_irr_rj))  # Float32
        ag_X_mrj_shape = ag_X_mrj.shape
        
        # Reshape so that cells are along the first axis and land management and use are flattened along second axis i.e. (XXXXXXX, 56)
        ag_X_mrj_processed = np.moveaxis(ag_X_mrj, 1, 0)
        ag_X_mrj_processed = ag_X_mrj_processed.reshape(ag_X_mrj_processed.shape[0], -1)
        
        # Boolean matrix where the maximum value for each cell across all land management types and land uses is True
        ag_X_mrj_processed = ag_X_mrj_processed.argmax(axis = 1)[:, np.newaxis] == range(ag_X_mrj_processed.shape[1])
        
        # Reshape to mrj structure
        ag_X_mrj_processed = ag_X_mrj_processed.reshape(
            (ag_X_mrj_shape[1], ag_X_mrj_shape[0], ag_X_mrj_shape[2])
        )
        ag_X_mrj_processed = np.moveaxis(ag_X_mrj_processed, 0, 1)

        # Process non-agricultural land usage information
        # Boolean matrix where the maximum value for each cell across all non-ag LUs is True
        non_ag_X_rk_processed = non_ag_X_rk.argmax(axis=1)[:, np.newaxis] == range(n_non_ag_lus)

        # Make land use and land management maps
        # Vector indexed by cell that denotes whether the cell is non-agricultural land (True) or agricultural land (False)
        non_ag_bools_r = non_ag_X_rk.max(axis=1) > ag_X_mrj.max(axis=(0, 2))
        
        # Calculate 1D array (maps) of land-use and land management, considering only agricultural LUs
        lumap = ag_X_mrj_processed.sum(axis = 0).argmax(axis = 1).astype('int8')
        lmmap = ag_X_mrj_processed.sum(axis = 2).argmax(axis = 0).astype('int8')

        # Update lxmaps and processed variable matrices to consider non-agricultural LUs
        lumap[non_ag_bools_r] = non_ag_X_rk[non_ag_bools_r, :].argmax(axis=1) + settings.NON_AGRICULTURAL_LU_BASE_CODE
        lmmap[non_ag_bools_r] = 0  # Assume that all non-agricultural land uses are dryland

        ag_X_mrj_processed[:, non_ag_bools_r, :] = False
        non_ag_X_rk_processed[~non_ag_bools_r, :] = False

        print('Done\n')
        print('Total processing time...', round(time.time() - start_time), 'seconds')

        return lumap, lmmap, ag_X_mrj_processed.astype(int), non_ag_X_rk_processed.astype(int)

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
