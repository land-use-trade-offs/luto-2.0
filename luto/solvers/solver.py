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

from collections import defaultdict
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass

import luto.settings as settings
from luto import tools
from luto.ag_managements import SORTED_AG_MANAGEMENTS, AG_MANAGEMENTS_TO_LAND_USES

# Set Gurobi environment.
gurenv = gp.Env(logfilename = 'gurobi.log', empty = True) # (empty = True)
gurenv.setParam('Method', settings.SOLVE_METHOD)
gurenv.setParam('OutputFlag', settings.VERBOSE)
gurenv.setParam('OptimalityTol', settings.OPTIMALITY_TOLERANCE)
gurenv.setParam('Threads', settings.THREADS)
gurenv.setParam('BarHomogeneous', settings.BARHOMOGENOUS)
gurenv.start()


@dataclass
class InputData:
    """
    An object that collects and stores all relevant data for solver.py.
    """    
    ag_t_mrj: np.ndarray            # Agricultural transition cost matrices.
    ag_c_mrj: np.ndarray            # Agricultural production cost matrices.
    ag_r_mrj: np.ndarray            # Agricultural production revenue matrices.
    ag_g_mrj: np.ndarray            # Agricultural greenhouse gas emissions matrices.
    ag_w_mrj: np.ndarray            # Agricultural water requirements matrices.
    ag_x_mrj: np.ndarray            # Agricultural exclude matrices.
    ag_q_mrp: np.ndarray            # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
    ag_ghg_t_mrj: np.ndarray        # GHG emissions released during transitions between agricultural land uses.

    ag_to_non_ag_t_rk: np.ndarray   # Agricultural to non-agricultural transition cost matrix.
    non_ag_to_ag_t_mrj: np.ndarray  # Non-agricultural to agricultural transition cost matrices.
    non_ag_c_rk: np.ndarray         # Non-agricultural production cost matrix.
    non_ag_r_rk: np.ndarray         # Non-agricultural revenue matrix.
    non_ag_g_rk: np.ndarray         # Non-agricultural greenhouse gas emissions matrix.
    non_ag_w_rk: np.ndarray         # Non-agricultural water requirements matrix.
    non_ag_x_rk: np.ndarray         # Non-agricultural exclude matrices.
    non_ag_q_crk: np.ndarray        # Non-agricultural yield matrix.

    ag_man_c_mrj: np.ndarray        # Agricultural management options' cost effects.
    ag_man_g_mrj: np.ndarray        # Agricultural management options' GHG emission effects.
    ag_man_q_mrp: np.ndarray        # Agricultural management options' quantity effects.
    ag_man_r_mrj: np.ndarray        # Agricultural management options' revenue effects.
    ag_man_t_mrj: np.ndarray        # Agricultural management options' transition cost effects.
    ag_man_w_mrj: np.ndarray        # Agricultural management options' water requirement effects.
    ag_man_limits: np.ndarray       # Agricultural management options' adoption limits.

    lu2pr_pj: np.ndarray            # Conversion matrix: land-use to product(s).
    pr2cm_cp: np.ndarray            # Conversion matrix: product(s) to commodity.
    limits: dict                    # Targets to use.
    desc2aglu: dict                 # Map of agricultural land use descriptions to codes.


def solve( d_c                    # Demands -- note the `c` ('commodity') index instead of `j` (land-use).
         , input_data: InputData  # InputData object, containing all other input data matrices and limits
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
    limits = input_data.limits
    if limits is None: limits = {}

    # Extract the shape of the problem.
    n_ag_lms, ncells, n_ag_lus = input_data.ag_t_mrj.shape # Number of agricultural landmans, cells, agricultural landuses.
    _, n_non_ag_lus = input_data.non_ag_c_rk.shape         # Number of non-agricultural landuses.

    _, _, nprs = input_data.ag_q_mrp.shape # Number of products.
    ncms, = d_c.shape # Number of commodities.

    # Make an index of each cell permitted to transform to each land use / land management combination
    ag_lu2cells = {
        (m, j): np.where(input_data.ag_x_mrj[m, :, j])[0]
                for j in range(n_ag_lus)
                for m in range(n_ag_lms)
    }
    non_ag_lu2cells = {
        k: np.where(input_data.non_ag_x_rk[:, k])[0]
           for k in range(n_non_ag_lus)
    }

    # Create a version of the agricultural management to land uses map that contains LU numbers (instead of names)
    am2j = {
        am: [input_data.desc2aglu[lu] for lu in am_lus]
            for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items()
    }

    # Create the reverse map
    j2am = defaultdict(list)
    for am, am_j_list in am2j.items():
        for j in am_j_list:
            j2am[j].append(am)

    try:
        print('\nSetting up the model...', time.ctime() + '\n')

        # Make Gurobi model instance.
        model = gp.Model('LUTO ' + settings.VERSION, env = gurenv)
        
        # ------------------- #
        # Decision variables. #
        # ------------------- #
        print('Adding decision variables...', time.ctime() + '\n')

        # Initialise a sparse, matrix version of the variable for adding constraints later
        X_ag_dry_vars_jr = np.zeros((n_ag_lus, ncells), dtype=object)
        X_ag_irr_vars_jr = np.zeros((n_ag_lus, ncells), dtype=object)
        X_non_ag_vars_kr = np.zeros((n_non_ag_lus, ncells), dtype=object)
        
        for j in range(n_ag_lus):
            dry_lu_cells = ag_lu2cells[0, j]
            for r in dry_lu_cells:
                X_ag_dry_vars_jr[j, r] = model.addVar(ub=1, name=f"X_ag_dry_{j}_{r}")

            irr_lu_cells = ag_lu2cells[1, j]
            for r in irr_lu_cells:
                X_ag_irr_vars_jr[j, r] = model.addVar(ub=1, name=f"X_ag_irr_{j}_{r}")

        for k in range(n_non_ag_lus):
            lu_cells = non_ag_lu2cells[k]
            for r in lu_cells:
                X_non_ag_vars_kr[k, r] = model.addVar(ub=1, name=f"X_non_ag_{k}_{r}")

        # Create extra variables for alternative agricultural management options (e.g. Asparagopsis taxiformis)
        # Prepare the variable matrices as zeros
        X_ag_man_dry_vars_jr = {
            am: np.zeros((len(am_j_list), ncells), dtype=object)
            for am, am_j_list in am2j.items()
        }
        X_ag_man_irr_vars_jr = {
            am: np.zeros((len(am_j_list), ncells), dtype=object)
            for am, am_j_list in am2j.items()
        }

        for am, am_j_list in am2j.items():
            am_name = tools.am_name_snake_case(am)

            for j_idx, j in enumerate(am_j_list):                
                # Create variable for all eligible cells
                dry_lu_cells = ag_lu2cells[0, j]
                for r in dry_lu_cells:
                    dry_var_name = f"X_ag_man_dry_{am_name}_{j}_{r}"
                    X_ag_man_dry_vars_jr[am][j_idx, r] = model.addVar(ub=1, name=dry_var_name)

                irr_lu_cells = ag_lu2cells[1, j]
                for r in irr_lu_cells:
                    irr_var_name = f"X_ag_man_irr_{am_name}_{j}_{r}"
                    X_ag_man_irr_vars_jr[am][j_idx, r] = model.addVar(ub=1, name=irr_var_name)

        # Decision variables, one for each commodity, to minimise the deviations from demand.
        V = model.addMVar(ncms, name = 'V')
        
        
        # ------------------- #
        # Objective function. #
        # ------------------- #
        print('Setting up objective function to %s...' % settings.OBJECTIVE, time.ctime() + '\n')
        
      
        if settings.OBJECTIVE == 'maxrev':
                    
            # Pre-calculate revenue minus (production and transition) costs
            ag_obj_mrj = -(
                  input_data.ag_r_mrj 
                - (input_data.ag_c_mrj + input_data.ag_t_mrj + input_data.non_ag_to_ag_t_mrj)
            ) / settings.PENALTY

            non_ag_obj_rk = -(
                  input_data.non_ag_r_rk 
                - (input_data.non_ag_c_rk + input_data.ag_to_non_ag_t_rk)
            ) / settings.PENALTY

            # Get effects of alternative agr. management options (stored in a dict)
            ag_man_objs = {
                am: -(
                      input_data.ag_man_r_mrj[am] 
                    - (input_data.ag_man_c_mrj[am] + input_data.ag_man_t_mrj[am])  
                ) / settings.PENALTY
                for am in am2j
            }
         
        elif settings.OBJECTIVE == 'mincost':
            
            # Pre-calculate sum of production and transition costs
            ag_obj_mrj = (
                  input_data.ag_c_mrj 
                + input_data.ag_t_mrj 
                + input_data.non_ag_to_ag_t_mrj
            ) / settings.PENALTY
            non_ag_obj_rk = (
                  input_data.non_ag_c_rk 
                + input_data.ag_to_non_ag_t_rk
            ) / settings.PENALTY

            # Store calculations for each agricultural management option in a dict
            ag_man_objs = {
                am: ( input_data.ag_man_c_mrj[am]
                    + input_data.ag_man_t_mrj[am] 
                    ) / settings.PENALTY
                for am in am2j
            }
            
        else:
            print('Unknown objective')

        ag_obj_mrj = np.nan_to_num(ag_obj_mrj)
        non_ag_obj_rk = np.nan_to_num(non_ag_obj_rk)

        # Specify objective function
        # Production costs + transition costs for all agricultural land uses.
        ag_obj_contr = gp.quicksum( 
            ag_obj_mrj[0, ag_lu2cells[0, j], j] @ X_ag_dry_vars_jr[j, ag_lu2cells[0, j]]
          + ag_obj_mrj[1, ag_lu2cells[1, j], j] @ X_ag_irr_vars_jr[j, ag_lu2cells[1, j]]
            for j in range(n_ag_lus) 
        )
        
        # Effects on production costs + transition costs for alternative agr. management options
        ag_man_obj_contr = gp.quicksum( 
            ag_man_objs[am][0, ag_lu2cells[0, j], j_idx] @ X_ag_man_dry_vars_jr[am][j_idx, ag_lu2cells[0, j]]
          + ag_man_objs[am][1, ag_lu2cells[1, j], j_idx] @ X_ag_man_irr_vars_jr[am][j_idx, ag_lu2cells[1, j]]
            for am, am_j_list in am2j.items()
            for j_idx, j in enumerate(am_j_list) 
        )

        # Production costs + transition costs for all non-agricultural land uses.
        non_ag_obj_contr = gp.quicksum( 
            non_ag_obj_rk[:, k][non_ag_lu2cells[k]] @ X_non_ag_vars_kr[k, non_ag_lu2cells[k]]
            for k in range(n_non_ag_lus) 
        )

        objective = ag_obj_contr + ag_man_obj_contr + non_ag_obj_contr + gp.quicksum( V[c] for c in range(ncms) )
        model.setObjective(objective, GRB.MINIMIZE)
        

        # ------------ #
        # Constraints. #
        # ------------ #
        print('Setting up constraints...', time.ctime() + '\n')

        # Constraint that all of every cell is used for some land use.
        # Create an array indexed by cell that contains the sums of each cell's variables.
        # Then, loop through the array and add the constraint that each expression must equal 1.
        X_sum_r = X_ag_dry_vars_jr.sum(axis=0) + X_ag_irr_vars_jr.sum(axis=0) + X_non_ag_vars_kr.sum(axis=0)
        for expr in X_sum_r:
            model.addConstr(expr == 1)

        # Constraint handling alternative agricultural management options:
        # Ag. man. variables cannot exceed the value of the agricultural variable.  TODO - uncomment
        for am, am_j_list in am2j.items():
            for j_idx, j in enumerate(am_j_list):
                for r in ag_lu2cells[0, j]:
                    model.addConstr(X_ag_man_dry_vars_jr[am][j_idx, r] <= X_ag_dry_vars_jr[j, r])
                for r in ag_lu2cells[1, j]:
                    model.addConstr(X_ag_man_irr_vars_jr[am][j_idx, r] <= X_ag_irr_vars_jr[j, r])

        # Add adoption limits constraints for agricultural management options.
        for am, am_j_list in am2j.items():
            for j_idx, j in enumerate(am_j_list):
                adoption_limit = input_data.ag_man_limits[am][j]

                # Sum of all usage of the AM option must be less than the limit
                ag_man_vars_sum = (
                    gp.quicksum(X_ag_man_dry_vars_jr[am][j_idx, :])
                  + gp.quicksum(X_ag_man_irr_vars_jr[am][j_idx, :])
                )
                all_vars_sum = (
                    gp.quicksum(X_ag_dry_vars_jr[j, :])
                  + gp.quicksum(X_ag_irr_vars_jr[j, :])
                )
                model.addConstr(ag_man_vars_sum <= adoption_limit * all_vars_sum)

        # Constraints to penalise under and over production compared to demand.
        # Transform agricultural decision vars from LU/j to PR/p representation.
        X_dry_pr = [
            X_ag_dry_vars_jr[j, :]
            for p in range(nprs)
            for j in range(n_ag_lus)
            if input_data.lu2pr_pj[p, j]
        ]
        X_irr_pr = [
            X_ag_irr_vars_jr[j, :]
            for p in range(nprs)
            for j in range(n_ag_lus)
            if input_data.lu2pr_pj[p, j]
        ]

        # Quantities in PR/p representation by land-management (dry/irr).
        ag_q_dry_p = [gp.quicksum(input_data.ag_q_mrp[0, :, p] * X_dry_pr[p]) for p in range(nprs)]
        ag_q_irr_p = [gp.quicksum(input_data.ag_q_mrp[1, :, p] * X_irr_pr[p]) for p in range(nprs)]

        # Transform quantities to CM/c representation by land management (dry/irr).
        ag_q_dry_c = [
            gp.quicksum(ag_q_dry_p[p] for p in range(nprs) if input_data.pr2cm_cp[c, p])
            for c in range(ncms)
        ]
        ag_q_irr_c = [
            gp.quicksum(ag_q_irr_p[p] for p in range(nprs) if input_data.pr2cm_cp[c, p])
            for c in range(ncms)
        ]

        # Repeat to get contributions of alternative agr. management options
        # Convert variables to PR/p representation
        for am, am_j_list in am2j.items():
            X_ag_man_dry_pr = []
            X_ag_man_irr_pr = []
            for p in range(nprs):
                for j_idx, j in enumerate(am_j_list):
                    if input_data.lu2pr_pj[p, j]:
                        X_ag_man_dry_pr.append(X_ag_man_dry_vars_jr[am][j_idx, :])
                        X_ag_man_irr_pr.append(X_ag_man_irr_vars_jr[am][j_idx, :])
                        break
                else:
                    X_ag_man_irr_pr.append(np.zeros(ncells))
                    X_ag_man_dry_pr.append(np.zeros(ncells))

            ag_man_q_dry_p = [
                gp.quicksum(input_data.ag_man_q_mrp[am][0, :, p] * X_ag_man_dry_pr[p])
                for p in range(nprs)
            ]
            ag_man_q_irr_p = [
                gp.quicksum(input_data.ag_man_q_mrp[am][1, :, p] * X_ag_man_irr_pr[p])
                for p in range(nprs)
            ]

            ag_man_q_dry_c = [
                gp.quicksum(ag_man_q_dry_p[p] for p in range(nprs) if input_data.pr2cm_cp[c, p])
                for c in range(ncms)
            ]
            ag_man_q_irr_c = [
                gp.quicksum(ag_man_q_irr_p[p] for p in range(nprs) if input_data.pr2cm_cp[c, p])
                for c in range(ncms)
            ]

            # Add to original agricultural variable commodity sums
            for c in range(ncms):
                ag_q_dry_c[c] += ag_man_q_dry_c[c]
                ag_q_irr_c[c] += ag_man_q_irr_c[c]

        # Calculate non-agricultural commodity contributions
        non_ag_q_c = [
            gp.quicksum(
                gp.quicksum(input_data.non_ag_q_crk[c, :, k] * X_non_ag_vars_kr[k, :])
                for k in range(n_non_ag_lus)
            )
            for c in range(ncms)
        ]

        # Total quantities in CM/c representation.
        total_q_c = [ag_q_dry_c[c] + ag_q_irr_c[c] + non_ag_q_c[c] for c in range(ncms)]

        # Finally, add the constraint in the CM/c representation.
        print("Adding constraints...", time.ctime() + "\n")

        model.addConstrs((d_c[c] - total_q_c[c]) <= V[c]
                          for c in range(ncms) )
        model.addConstrs((total_q_c[c] - d_c[c]) <= V[c]
                          for c in range(ncms) )
        
        

        # Only add the following constraints if 'limits' are provided.

        if settings.WATER_USE_LIMITS == 'on':
            print('Adding water constraints by', settings.WATER_REGION_DEF + '...', time.ctime())

            # Returns region-specific water use limits
            w_limits = limits['water']

            # Ensure water use remains below limit for each region
            for region, wreq_reg_limit, ind in w_limits:
                ag_contr = gp.quicksum(
                    input_data.ag_w_mrj[0, ind, j] @ X_ag_dry_vars_jr[j, ind]  # Dryland agriculture contribution
                    + input_data.ag_w_mrj[1, ind, j] @ X_ag_irr_vars_jr[j, ind]  # Irrigated agriculture contribution
                    for j in range(n_ag_lus)
                )

                ag_man_contr =gp.quicksum(
                    input_data.ag_man_w_mrj[am][0, ind, j_idx] @ X_ag_man_dry_vars_jr[am][j_idx, ind]  # Dryland alt. ag. management contributions 
                  + input_data.ag_man_w_mrj[am][1, ind, j_idx] @ X_ag_man_irr_vars_jr[am][j_idx, ind]  # Irrigated alt. ag. management contributions 
                    for am, am_j_list in am2j.items()
                    for j_idx in range(len(am_j_list))
                )

                non_ag_contr = gp.quicksum( 
                    input_data.non_ag_w_rk[ind, k] @ X_non_ag_vars_kr[k, ind]  # Non-agricultural contribution
                    for k in range(n_non_ag_lus)
                )
                    
                wreq_region = ag_contr + ag_man_contr + non_ag_contr

                if wreq_region is not 0:
                    model.addConstr(wreq_region <= wreq_reg_limit)

                if settings.VERBOSE == 1:
                    print('    ...setting water limit for %s <= %.2f ML' % (region, wreq_reg_limit))


        if settings.GHG_EMISSIONS_LIMITS == 'on':
            print('\nAdding GHG emissions constraints...', time.ctime() + '\n')
            
            # Returns GHG emissions limits
            ghg_limits = limits['ghg']

            # Pre-calculate the coefficients for each variable, 
            # both for regular culture and alternative agr. management options
            g_dry_coeff = input_data.ag_g_mrj[0, :, :] + input_data.ag_ghg_t_mrj[0, :, :]
            g_irr_coeff = input_data.ag_g_mrj[1, :, :] + input_data.ag_ghg_t_mrj[1, :, :]

            ag_contr = gp.quicksum( 
                gp.quicksum(g_dry_coeff[:, j] * X_ag_dry_vars_jr[j, :])  # Dryland agriculture contribution
                + gp.quicksum(g_irr_coeff[:, j] * X_ag_irr_vars_jr[j, :])  # Irrigated agriculture contribution
                for j in range(n_ag_lus)
            )

            ag_man_contr = gp.quicksum(
                gp.quicksum(input_data.ag_man_g_mrj[am][0, :, j_idx] * X_ag_man_dry_vars_jr[am][j_idx, :])  # Dryland alt. ag. management contributions 
                + gp.quicksum(input_data.ag_man_g_mrj[am][1, :, j_idx] * X_ag_man_irr_vars_jr[am][j_idx, :])  # Irrigated alt. ag. management contributions 
                for am, am_j_list in am2j.items()
                for j_idx in range(len(am_j_list))
            )

            non_ag_contr = gp.quicksum(
                gp.quicksum(input_data.non_ag_g_rk[:, k] * X_non_ag_vars_kr[k, :])  # Non-agricultural contribution
                for k in range(n_non_ag_lus)
            )

            ghg_emissions = ag_contr + ag_man_contr + non_ag_contr
            
            print('    ...setting GHG emissions reduction target: {:,.0f} tCO2e\n'.format( ghg_limits ))
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
        X_dry_sol_rj = np.zeros((ncells, n_ag_lus)).astype(np.float32)
        X_irr_sol_rj = np.zeros((ncells, n_ag_lus)).astype(np.float32)
        non_ag_X_sol_rk = np.zeros((ncells, n_non_ag_lus)).astype(np.float32)
        am_X_dry_sol_rj = {am: np.zeros((ncells, n_ag_lus)).astype(np.float32) for am in am2j}
        am_X_irr_sol_rj = {am: np.zeros((ncells, n_ag_lus)).astype(np.float32) for am in am2j}

        # Get agricultural results
        for j in range(n_ag_lus):
            for r in ag_lu2cells[0, j]:
                X_dry_sol_rj[r, j] = X_ag_dry_vars_jr[j, r].X
            for r in ag_lu2cells[1, j]:
                X_irr_sol_rj[r, j] = X_ag_irr_vars_jr[j, r].X

        # Get non-agricultural results
        for k in range(n_non_ag_lus):
            for r in non_ag_lu2cells[k]:
                non_ag_X_sol_rk[r, k] = X_non_ag_vars_kr[k, r].X

        # Get agricultural management results
        for am, am_j_list in am2j.items():
            for j_idx, j in enumerate(am_j_list):
                for r in ag_lu2cells[0, j]:
                    am_X_dry_sol_rj[am][r, j] = X_ag_man_dry_vars_jr[am][j_idx, r].X
                for r in ag_lu2cells[1, j]:
                    am_X_irr_sol_rj[am][r, j] = X_ag_man_irr_vars_jr[am][j_idx, r].X
        
        """Note that output decision variables are mostly 0 or 1 but in some cases they are somewhere in between which creates issues 
           when converting to maps etc. as individual cells can have non-zero values for multiple land-uses and land management type.
           This code creates a boolean X_mrj output matrix and ensure that each cell has one and only one land-use and land management"""

        # Process agricultural land usage information
        # Stack dryland and irrigated decision variables
        ag_X_mrj = np.stack((X_dry_sol_rj, X_irr_sol_rj))  # Float32
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
        non_ag_X_rk_processed = non_ag_X_sol_rk.argmax(axis=1)[:, np.newaxis] == range(n_non_ag_lus)

        # Make land use and land management maps
        # Vector indexed by cell that denotes whether the cell is non-agricultural land (True) or agricultural land (False)
        non_ag_bools_r = non_ag_X_sol_rk.max(axis=1) > ag_X_mrj.max(axis=(0, 2))

        # Process agricultural management variables
        # Repeat the steps for the regular agricultural management variables
        ag_man_X_mrj_processed = {}
        for am in am2j:
            ag_man_processed = np.stack((am_X_dry_sol_rj[am], am_X_irr_sol_rj[am]))
            ag_man_X_shape = ag_man_processed.shape

            ag_man_processed = np.moveaxis(ag_man_processed, 1, 0)
            ag_man_processed = ag_man_processed.reshape(ag_man_processed.shape[0], -1)

            ag_man_processed = (
                   ag_man_processed.argmax(axis = 1)[:, np.newaxis] 
                == range(ag_man_processed.shape[1])
            )
            ag_man_processed = ag_man_processed.reshape(
                (ag_man_X_shape[1], ag_man_X_shape[0], ag_man_X_shape[2])
            )
            ag_man_processed = np.moveaxis(ag_man_processed, 0, 1)
            ag_man_X_mrj_processed[am] = ag_man_processed
        
        # Calculate 1D array (maps) of land-use and land management, considering only agricultural LUs
        lumap = ag_X_mrj_processed.sum(axis = 0).argmax(axis = 1).astype('int8')
        lmmap = ag_X_mrj_processed.sum(axis = 2).argmax(axis = 0).astype('int8')

        # Update lxmaps and processed variable matrices to consider non-agricultural LUs
        lumap[non_ag_bools_r] = non_ag_X_sol_rk[non_ag_bools_r, :].argmax(axis=1) + settings.NON_AGRICULTURAL_LU_BASE_CODE
        lmmap[non_ag_bools_r] = 0  # Assume that all non-agricultural land uses are dryland

        # Process agricultural management usage info
        # Make ammap (agricultural management map) using the lumap and lmmap
        ammap = np.zeros(ncells, dtype=np.int8)
        for r in range(ncells):
            cell_j = lumap[r]
            cell_m = lmmap[r]

            if cell_j >= settings.NON_AGRICULTURAL_LU_BASE_CODE:
                # Non agricultural land use - no agricultural management option
                cell_am = 0

            else:
                if cell_m == 0:
                    am_values = [am_X_dry_sol_rj[am][r, cell_j] for am in SORTED_AG_MANAGEMENTS]
                else:
                    am_values = [am_X_irr_sol_rj[am][r, cell_j] for am in SORTED_AG_MANAGEMENTS]

                # Get argmax and max of the am_values list
                argmax, max_am_var_val = max(enumerate(am_values), key=lambda x: x[1])

                if max_am_var_val < settings.AGRICULTURAL_MANAGEMENT_USE_THRESHOLD:
                    # The cell doesn't use any alternative agricultural management options
                    cell_am = 0
                else:
                    # Add one to the argmax to account for the default option of no ag management being 0
                    cell_am = argmax + 1

            ammap[r] = cell_am

        ag_X_mrj_processed[:, non_ag_bools_r, :] = False
        non_ag_X_rk_processed[~non_ag_bools_r, :] = False

        print('Done\n')
        print('Total processing time...', round(time.time() - start_time), 'seconds')

        return lumap, lmmap, ammap, ag_X_mrj_processed, non_ag_X_rk_processed, ag_man_X_mrj_processed

    except gp.GurobiError as e:
        print('Gurobi error code', str(e.errno), ':', str(e))

    except AttributeError:
        print('Encountered an attribute error')
