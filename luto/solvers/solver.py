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
import gurobipy as gp
import luto.settings as settings

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from gurobipy import GRB

from luto import tools
from luto.solvers.input_data import SolverInputData
from luto.settings import AG_MANAGEMENTS, AG_MANAGEMENTS_REVERSIBLE
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.settings import NON_AG_LAND_USES, NON_AG_LAND_USES_REVERSIBLE



# Set Gurobi environment.
gurenv = gp.Env(logfilename="gurobi.log", empty=True)  # (empty = True)
gurenv.setParam("Method", settings.SOLVE_METHOD)
gurenv.setParam("OutputFlag", settings.VERBOSE)
gurenv.setParam("Presolve", settings.PRESOLVE)
gurenv.setParam("Aggregate", settings.AGGREGATE)
gurenv.setParam("OptimalityTol", settings.OPTIMALITY_TOLERANCE)
gurenv.setParam("FeasibilityTol", settings.FEASIBILITY_TOLERANCE)
gurenv.setParam("BarConvTol", settings.BARRIER_CONVERGENCE_TOLERANCE)
gurenv.setParam("ScaleFlag", settings.SCALE_FLAG)
gurenv.setParam("NumericFocus", settings.NUMERIC_FOCUS)
gurenv.setParam("Threads", settings.THREADS)
gurenv.setParam("BarHomogeneous", settings.BARHOMOGENOUS)
gurenv.setParam("Crossover", settings.CROSSOVER)
gurenv.start()


@dataclass
class SolverSolution:
    lumap: np.ndarray
    lmmap: np.ndarray
    ammaps: dict[str, np.ndarray]
    ag_X_mrj: np.ndarray
    non_ag_X_rk: np.ndarray
    ag_man_X_mrj: dict[str, np.ndarray]
    prod_data: dict[str, float]
    obj_val: dict[str, float]


class LutoSolver:
    """
    Class responsible for grouping the Gurobi model, relevant input data, and its variables.
    """

    def __init__(
        self,
        input_data: SolverInputData,
        d_c: np.array,
        final_target_year: int,
        ):


        self.final_target_year = final_target_year
        self._input_data = input_data
        self.d_c = d_c
        self.gurobi_model = gp.Model(f"LUTO {settings.VERSION}", env=gurenv)

        # Initialise variable stores
        self.X_ag_dry_vars_jr = None
        self.X_ag_irr_vars_jr = None
        self.X_non_ag_vars_kr = None
        self.X_ag_man_dry_vars_jr = None
        self.X_ag_man_irr_vars_jr = None
        self.V = None
        self.E = None

        # Initialise constraint lookups
        self.cell_usage_constraint_r = {}
        self.ag_management_constraints_r = defaultdict(list)
        self.adoption_limit_constraints = []
        self.demand_penalty_constraints = []
        self.water_limit_constraints = []
        self.ghg_emissions_expr = None
        self.ghg_emissions_limit_constraint_ub = None
        self.ghg_emissions_limit_constraint_lb = None
        self.ghg_emissions_reduction_soft_constraints = []
        self.biodiversity_expr = None
        self.biodiversity_limit_constraint = None


    def formulate(self):
        """
        Performs the initial formulation of the model - setting up decision variables,
        constraints, and the objective.
        """
        print("Setting up the model...")

        print("Adding the decision variables...")
        self._setup_vars()

        print("Adding the constraints...")
        self._setup_constraints()

        print(f"Adding the objective function - {settings.OBJECTIVE}...", flush=True)
        self._setup_objective()



    def _setup_vars(self):
        self._setup_x_vars()
        self._setup_ag_management_variables()
        self._setup_deviation_penalties()

    def _setup_constraints(self):
        self._add_cell_usage_constraints()                              
        self._add_agricultural_management_constraints()                 
        self._add_agricultural_management_adoption_limit_constraints()  
        self._add_demand_penalty_constraints()                          
        self._add_water_usage_limit_constraints() if settings.WATER_LIMITS == 'on' else print('  ...TURNING OFF water usage constraints ...')
        self._add_ghg_emissions_limit_constraints()                     
        self._add_biodiversity_limit_constraints()


    def _setup_x_vars(self):
        """
        Sets up the 'x' variables, responsible for managing how cells are used.
        """
        self.X_ag_dry_vars_jr = np.zeros(
            (self._input_data.n_ag_lus, self._input_data.ncells), dtype=object
        )
        self.X_ag_irr_vars_jr = np.zeros(
            (self._input_data.n_ag_lus, self._input_data.ncells), dtype=object
        )
        self.X_non_ag_vars_kr = np.zeros(
            (self._input_data.n_non_ag_lus, self._input_data.ncells), dtype=object
        )

        for j in range(self._input_data.n_ag_lus):
            dry_lu_cells = self._input_data.ag_lu2cells[0, j]
            for r in dry_lu_cells:
                self.X_ag_dry_vars_jr[j, r] = self.gurobi_model.addVar(
                    ub=1, name=f"X_ag_dry_{j}_{r}"
                )

            irr_lu_cells = self._input_data.ag_lu2cells[1, j]
            for r in irr_lu_cells:
                self.X_ag_irr_vars_jr[j, r] = self.gurobi_model.addVar(
                    ub=1, name=f"X_ag_irr_{j}_{r}"
                )

        for k, non_ag_lu_desc in enumerate(NON_AG_LAND_USES):
            if not NON_AG_LAND_USES[non_ag_lu_desc]:
                continue

            lu_cells = self._input_data.non_ag_lu2cells[k]
            for r in lu_cells:
                x_lb = (
                    0 if NON_AG_LAND_USES_REVERSIBLE[non_ag_lu_desc]
                    else self._input_data.non_ag_lb_rk[r, k]
                )
                self.X_non_ag_vars_kr[k, r] = self.gurobi_model.addVar(
                    lb=x_lb, ub=self._input_data.non_ag_x_rk[r, k], name=f"X_non_ag_{k}_{r}",
                )

    def _setup_ag_management_variables(self):
        """
        Create extra variables for alternative agricultural management options
        (e.g. Asparagopsis taxiformis)
        """
        # Prepare the variable matrices as zeros
        self.X_ag_man_dry_vars_jr = {
            am: np.zeros((len(am_j_list), self._input_data.ncells), dtype=object)
            for am, am_j_list in self._input_data.am2j.items()
        }
        self.X_ag_man_irr_vars_jr = {
            am: np.zeros((len(am_j_list), self._input_data.ncells), dtype=object)
            for am, am_j_list in self._input_data.am2j.items()
        }

        for am, am_j_list in self._input_data.am2j.items():
            if not AG_MANAGEMENTS[am]:
                continue

            # Get snake_case version of the AM name for the variable name
            am_name = tools.am_name_snake_case(am)

            for j_idx, j in enumerate(am_j_list):
                # Create variable for all eligible cells - all lower bounds are zero
                dry_lu_cells = self._input_data.ag_lu2cells[0, j]
                for r in dry_lu_cells:
                    dry_x_lb = 0 if AG_MANAGEMENTS_REVERSIBLE[am] else self._input_data.ag_man_lb_mrj[am][0, r, j]
                    dry_var_name = f"X_ag_man_dry_{am_name}_{j}_{r}"

                    self.X_ag_man_dry_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                        lb=dry_x_lb, ub=1, name=dry_var_name,
                    )               

                irr_lu_cells = self._input_data.ag_lu2cells[1, j]
                for r in irr_lu_cells:
                    irr_x_lb = 0 if AG_MANAGEMENTS_REVERSIBLE[am] else self._input_data.ag_man_lb_mrj[am][1, r, j]
                    irr_var_name = f"X_ag_man_irr_{am_name}_{j}_{r}"

                    self.X_ag_man_irr_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                        lb=irr_x_lb, ub=1, name=irr_var_name,
                    )
                 

    def _setup_deviation_penalties(self):
        """
        Decision variables, V and E, for soft constraints.
        1) [V] Penalty vector for demand, each one corespondes a commodity, that minimises the deviations from demand.
        2) [E] A single penalty scalar for GHG emissions, minimises its deviation from the target.
        """
        if settings.DEMAND_CONSTRAINT_TYPE == "soft":
            self.V = self.gurobi_model.addMVar(self.ncms, name="V")
            
        if settings.GHG_CONSTRAINT_TYPE == "soft":
            self.E = self.gurobi_model.addVar(name="E")

        

    def _setup_objective(self):
        """
        Formulate the objective based on settings.OBJECTIVE
        """
        print(f"Setting objective function to {settings.OBJECTIVE}...", flush=True)

        # Get the objective values matrices for each sector
        ag_obj_mrj, non_ag_obj_rk, ag_man_objs = self._input_data.economic_contr_mrj

        # Production costs + transition costs for all agricultural land uses.
        ag_obj_contr = gp.quicksum(
            ag_obj_mrj[0, self._input_data.ag_lu2cells[0, j], j]
            @ self.X_ag_dry_vars_jr[j, self._input_data.ag_lu2cells[0, j]]
            + ag_obj_mrj[1, self._input_data.ag_lu2cells[1, j], j]
            @ self.X_ag_irr_vars_jr[j, self._input_data.ag_lu2cells[1, j]]
            for j in range(self._input_data.n_ag_lus)
        )

        # Effects on production costs + transition costs for alternative agr. management options
        ag_man_obj_contr = gp.quicksum(
            ag_man_objs[am][0, self._input_data.ag_lu2cells[0, j], j_idx]
            @ self.X_ag_man_dry_vars_jr[am][j_idx, self._input_data.ag_lu2cells[0, j]]
            + ag_man_objs[am][1, self._input_data.ag_lu2cells[1, j], j_idx]
            @ self.X_ag_man_irr_vars_jr[am][j_idx, self._input_data.ag_lu2cells[1, j]]
            for am, am_j_list in self._input_data.am2j.items()
            for j_idx, j in enumerate(am_j_list)
        )

        # Production costs + transition costs for all non-agricultural land uses.
        non_ag_obj_contr = gp.quicksum(
            non_ag_obj_rk[:, k][self._input_data.non_ag_lu2cells[k]]
            @ self.X_non_ag_vars_kr[k, self._input_data.non_ag_lu2cells[k]]
            for k in range(self._input_data.n_non_ag_lus)
        )
        
        # Get the objective values for each sector
        self.obj_economy = ag_obj_contr + ag_man_obj_contr + non_ag_obj_contr

        if settings.DEMAND_CONSTRAINT_TYPE == "soft":
            self.obj_demand = gp.quicksum(v * price for v, price in zip(self.V, self._input_data.economic_BASE_YR_prices))
        else:
            self.obj_demand = 0
            
        if settings.GHG_CONSTRAINT_TYPE == "soft":
            self.obj_ghg = self.E * self._input_data.economic_target_yr_carbon_price
        else:
            self.obj_ghg = 0

        # Set the objective function
        if settings.OBJECTIVE == "maxprofit":
            sense = GRB.MINIMIZE
            objective = self.obj_economy * settings.SOLVE_ECONOMY_WEIGHT - (self.obj_demand +  self.obj_ghg) * (1 - settings.SOLVE_ECONOMY_WEIGHT)  
        elif settings.OBJECTIVE == "mincost":
            sense = GRB.MINIMIZE
            objective = self.obj_economy * settings.SOLVE_ECONOMY_WEIGHT + (self.obj_demand +  self.obj_ghg) * (1 - settings.SOLVE_ECONOMY_WEIGHT)
        else:
            raise ValueError("Unknown choice for `OBJECTIVE` setting: must be either 'maxprofit' or 'mincost'")
     
        self.gurobi_model.setObjective(objective, sense)  
        
        
    def _add_cell_usage_constraints(self, cells: Optional[np.array] = None):
        """
        Constraint that all of every cell is used for some land use.
        If `cells` is provided, only adds constraints for the given cells
        """
        print('  ...cell usage constraints...')

        if cells is None:
            cells = np.array(range(self._input_data.ncells))

        x_ag_dry_vars = self.X_ag_dry_vars_jr[:, cells]
        x_ag_irr_vars = self.X_ag_irr_vars_jr[:, cells]
        x_non_ag_vars = self.X_non_ag_vars_kr[:, cells]

        # Create an array indexed by cell that contains the sums of each cell's variables.
        # Then, loop through the array and add the constraint that each expression must equal 1.
        X_sum_r = (
            x_ag_dry_vars.sum(axis=0)
            + x_ag_irr_vars.sum(axis=0)
            + x_non_ag_vars.sum(axis=0)
        )
        for r, expr in zip(cells, X_sum_r):
            self.cell_usage_constraint_r[r] = self.gurobi_model.addConstr(expr == 1)

    def _add_agricultural_management_constraints(
        self, cells: Optional[np.array] = None
    ):
        """
        Constraint handling alternative agricultural management options:
        Ag. man. variables cannot exceed the value of the agricultural variable.
        """
        print('  ...agricultural management constraints...')

        for am, am_j_list in self._input_data.am2j.items():
            for j_idx, j in enumerate(am_j_list):
                if cells is not None:
                    lm0_r_vals = [r for r in cells if self._input_data.ag_x_mrj[0, r, j]]
                    lm1_r_vals = [r for r in cells if self._input_data.ag_x_mrj[1, r, j]]
                else:
                    lm0_r_vals = self._input_data.ag_lu2cells[0, j]
                    lm1_r_vals = self._input_data.ag_lu2cells[1, j]

                for r in lm0_r_vals:
                    constr = self.gurobi_model.addConstr(
                        self.X_ag_man_dry_vars_jr[am][j_idx, r] <= self.X_ag_dry_vars_jr[j, r]
                    )
                    self.ag_management_constraints_r[r].append(constr)
                for r in lm1_r_vals:
                    constr = self.gurobi_model.addConstr(
                        self.X_ag_man_irr_vars_jr[am][j_idx, r] <= self.X_ag_irr_vars_jr[j, r]
                    )
                    self.ag_management_constraints_r[r].append(constr)

    def _add_agricultural_management_adoption_limit_constraints(self):
        """
        Add adoption limits constraints for agricultural management options.
        """
        print('  ...agricultural management adoption constraints...')

        for am, am_j_list in self._input_data.am2j.items():
            for j_idx, j in enumerate(am_j_list):
                adoption_limit = self._input_data.ag_man_limits[am][j]

                # Sum of all usage of the AM option must be less than the limit
                ag_man_vars_sum = gp.quicksum(
                    self.X_ag_man_dry_vars_jr[am][j_idx, :]
                ) + gp.quicksum(self.X_ag_man_irr_vars_jr[am][j_idx, :])

                all_vars_sum = gp.quicksum(self.X_ag_dry_vars_jr[j, :]) + gp.quicksum(
                    self.X_ag_irr_vars_jr[j, :]
                )
                constr = self.gurobi_model.addConstr(
                    ag_man_vars_sum <= adoption_limit * all_vars_sum
                )

                self.adoption_limit_constraints.append(constr)

    def _add_demand_penalty_constraints(self):
        """
        Constraints to penalise under and over production compared to demand.
        """
        print('  ...demand constraints...')

        # Quantities in PR/p representation by land-management (dry/irr).
        ag_q_dry_p = [
            gp.quicksum(self._input_data.ag_q_mrp[0, :, p] * self.X_dry_pr[p])
            for p in range(self._input_data.nprs)
        ]
        ag_q_irr_p = [
            gp.quicksum(self._input_data.ag_q_mrp[1, :, p] * self.X_irr_pr[p])
            for p in range(self._input_data.nprs)
        ]

        # Transform quantities to CM/c representation by land management (dry/irr).
        ag_q_dry_c = [
            gp.quicksum(
                ag_q_dry_p[p]
                for p in range(self._input_data.nprs)
                if self._input_data.pr2cm_cp[c, p]
            )
            for c in range(self.ncms)
        ]
        ag_q_irr_c = [
            gp.quicksum(
                ag_q_irr_p[p]
                for p in range(self._input_data.nprs)
                if self._input_data.pr2cm_cp[c, p]
            )
            for c in range(self.ncms)
        ]

        # Repeat to get contributions of alternative agr. management options
        # Convert variables to PR/p representation
        for am, am_j_list in self._input_data.am2j.items():
            X_ag_man_dry_pr = np.zeros(
                (self._input_data.nprs, self._input_data.ncells), dtype=object
            )
            X_ag_man_irr_pr = np.zeros(
                (self._input_data.nprs, self._input_data.ncells), dtype=object
            )

            for j_idx, j in enumerate(am_j_list):
                for p in self._input_data.j2p[j]:
                    if self._input_data.lu2pr_pj[p, j]:
                        X_ag_man_dry_pr[p, :] = self.X_ag_man_dry_vars_jr[am][j_idx, :]
                        X_ag_man_irr_pr[p, :] = self.X_ag_man_irr_vars_jr[am][j_idx, :]

            ag_man_q_dry_p = [
                gp.quicksum(
                    self._input_data.ag_man_q_mrp[am][0, :, p] * X_ag_man_dry_pr[p, :]
                )
                for p in range(self._input_data.nprs)
            ]
            ag_man_q_irr_p = [
                gp.quicksum(
                    self._input_data.ag_man_q_mrp[am][1, :, p] * X_ag_man_irr_pr[p, :]
                )
                for p in range(self._input_data.nprs)
            ]

            ag_man_q_dry_c = [
                gp.quicksum(
                    ag_man_q_dry_p[p]
                    for p in range(self._input_data.nprs)
                    if self._input_data.pr2cm_cp[c, p]
                )
                for c in range(self.ncms)
            ]
            ag_man_q_irr_c = [
                gp.quicksum(
                    ag_man_q_irr_p[p]
                    for p in range(self._input_data.nprs)
                    if self._input_data.pr2cm_cp[c, p]
                )
                for c in range(self.ncms)
            ]

            # Add to original agricultural variable commodity sums
            for c in range(self.ncms):
                ag_q_dry_c[c] += ag_man_q_dry_c[c]
                ag_q_irr_c[c] += ag_man_q_irr_c[c]

        # Calculate non-agricultural commodity contributions
        non_ag_q_c = [
            gp.quicksum(
                gp.quicksum(
                    self._input_data.non_ag_q_crk[c, :, k] * self.X_non_ag_vars_kr[k, :]
                )
                for k in range(self._input_data.n_non_ag_lus)
            )
            for c in range(self.ncms)
        ]

        # Total quantities in CM/c representation.
        self.total_q_exprs_c = [
            ag_q_dry_c[c] + ag_q_irr_c[c] + non_ag_q_c[c] for c in range(self.ncms)
        ]

        if settings.DEMAND_CONSTRAINT_TYPE == "soft":
            upper_bound_constraints = self.gurobi_model.addConstrs(
                (self.d_c[c] - self.total_q_exprs_c[c]) <= self.V[c]
                for c in range(self.ncms)
            )
            lower_bound_constraints = self.gurobi_model.addConstrs(
                (self.total_q_exprs_c[c] - self.d_c[c]) <= self.V[c]
                for c in range(self.ncms)
            )

            self.demand_penalty_constraints.extend(upper_bound_constraints.values())
            self.demand_penalty_constraints.extend(lower_bound_constraints.values())

        elif settings.DEMAND_CONSTRAINT_TYPE == "hard":
            quantity_meets_demand_constraints = self.gurobi_model.addConstrs(
                (self.total_q_exprs_c[c] >= self.d_c[c]) for c in range(self.ncms)
            )
            self.demand_penalty_constraints.extend(
                quantity_meets_demand_constraints.values()
            )

        else:
            raise ValueError(
                'DEMAND_CONSTRAINT_TYPE not specified in settings, needs to be "hard" or "soft"'
            )

        

    def _add_water_usage_limit_constraints(self):
        """
        Adds constraints to handle water usage limits.
        If `cells` is provided, only adds constraints for regions containing at least one of the
        provided cells.
        """

        print(f'  ...water net yield constraints by {settings.WATER_REGION_DEF}...')

        # Ensure water use remains below limit for each region
        for region, (reg_name, limit_hist_level, ind) in self._input_data.limits["water"].items():

            ag_contr = gp.quicksum(
                gp.quicksum(
                    self._input_data.ag_w_mrj[0, ind, j] * self.X_ag_dry_vars_jr[j, ind]
                )  # Dryland agriculture contribution
                + gp.quicksum(
                    self._input_data.ag_w_mrj[1, ind, j] * self.X_ag_irr_vars_jr[j, ind]
                )  # Irrigated agriculture contribution
                for j in range(self._input_data.n_ag_lus)
            )

            ag_man_contr = gp.quicksum(
                gp.quicksum(
                    self._input_data.ag_man_w_mrj[am][0, ind, j_idx]
                    * self.X_ag_man_dry_vars_jr[am][j_idx, ind]
                )  # Dryland alt. ag. management contributions
                + gp.quicksum(
                    self._input_data.ag_man_w_mrj[am][1, ind, j_idx]
                    * self.X_ag_man_irr_vars_jr[am][j_idx, ind]
                )  # Irrigated alt. ag. management contributions
                for am, am_j_list in self._input_data.am2j.items()
                for j_idx in range(len(am_j_list))
            )

            non_ag_contr = gp.quicksum(
                gp.quicksum(
                    self._input_data.non_ag_w_rk[ind, k] * self.X_non_ag_vars_kr[k, ind]
                )  # Non-agricultural contribution
                for k in range(self._input_data.n_non_ag_lus)
            )

            # Get the water yield outside the study area in the Base Year (2010) of the whole simulation
            outside_luto_study_contr = self._input_data.water_yield_outside_study_area[region]

            # Sum of all water yield contributions
            w_net_yield_region = ag_contr + ag_man_contr + non_ag_contr + outside_luto_study_contr
            
            # Under River Regions, we need to update the water constraint when the wny_hist_level < wny_BASE_YR_level
            if settings.WATER_REGION_DEF == 'Drainage Division':
                water_yield_constraint = limit_hist_level
            elif settings.WATER_REGION_DEF == 'River Region':
                wny_BASE_YR_level = self._input_data.water_yield_RR_BASE_YR[region]
                water_yield_constraint = min(limit_hist_level, wny_BASE_YR_level)
            else:
                raise ValueError(f"Unknown choice for `WATER_REGION_DEF` setting: must be either 'River Region' or 'Drainage Division'")
        
            # Add the constraint that the water yield in the region must be greater than the limit
            constr = self.gurobi_model.addConstr(w_net_yield_region >= water_yield_constraint)
            self.water_limit_constraints.append(constr)
            
            # Report on the water yield in the region
            if settings.VERBOSE == 1:
                print(f"    ...net water yield in {reg_name} >= {limit_hist_level:.2f} ML")
            if water_yield_constraint != limit_hist_level:
                print(f"        ... updating water constraint to >= {water_yield_constraint:.2f} ML")


    def _get_total_ghg_emissions_expr(self) -> gp.LinExpr:
        # Pre-calculate the coefficients for each variable,
        # both for regular culture and alternative agr. management options
        g_dry_coeff = (
            self._input_data.ag_g_mrj[0, :, :] + self._input_data.ag_ghg_t_mrj[0, :, :]
        )
        g_irr_coeff = (
            self._input_data.ag_g_mrj[1, :, :] + self._input_data.ag_ghg_t_mrj[1, :, :]
        )
        ag_contr = gp.quicksum(
            gp.quicksum(
                g_dry_coeff[:, j] * self.X_ag_dry_vars_jr[j, :]
            )  # Dryland agriculture contribution
            + gp.quicksum(
                g_irr_coeff[:, j] * self.X_ag_irr_vars_jr[j, :]
            )  # Irrigated agriculture contribution
            for j in range(self._input_data.n_ag_lus)
        )
        ag_man_contr = gp.quicksum(
            gp.quicksum(
                self._input_data.ag_man_g_mrj[am][0, :, j_idx]
                * self.X_ag_man_dry_vars_jr[am][j_idx, :]
            )  # Dryland alt. ag. management contributions
            + gp.quicksum(
                self._input_data.ag_man_g_mrj[am][1, :, j_idx]
                * self.X_ag_man_irr_vars_jr[am][j_idx, :]
            )  # Irrigated alt. ag. management contributions
            for am, am_j_list in self._input_data.am2j.items()
            for j_idx in range(len(am_j_list))
        )
        non_ag_contr = gp.quicksum(
            gp.quicksum(
                self._input_data.non_ag_g_rk[:, k] * self.X_non_ag_vars_kr[k, :]
            )  # Non-agricultural contribution
            for k in range(self._input_data.n_non_ag_lus)
        )
        return ag_contr + ag_man_contr + non_ag_contr + self._input_data.offland_ghg


    def _add_ghg_emissions_limit_constraints(self):
        """
        Add either hard or soft GHG constraints depending on settings.GHG_CONSTRAINT_TYPE
        """
        if settings.GHG_EMISSIONS_LIMITS != "on":
            print('...GHG emissions constraints TURNED OFF ...')
            return
        
        ghg_limit_ub = self._input_data.limits["ghg_ub"]
        ghg_limit_lb = self._input_data.limits["ghg_lb"]
        self.ghg_emissions_expr = self._get_total_ghg_emissions_expr()
        
        if settings.GHG_CONSTRAINT_TYPE == 'hard':
            print(f"...GHG emissions reduction target")
            print(f'    ...GHG emissions reduction target UB: {ghg_limit_ub:,.0f} tCO2e')
            self.ghg_emissions_limit_constraint_ub = self.gurobi_model.addConstr(
                self.ghg_emissions_expr <= ghg_limit_ub
            )
            print(f'    ...GHG emissions reduction target LB: {ghg_limit_lb:,.0f} tCO2e')
            self.ghg_emissions_limit_constraint_lb = self.gurobi_model.addConstr(
                self.ghg_emissions_expr >= ghg_limit_lb
            )
        elif settings.GHG_CONSTRAINT_TYPE == 'soft':
            print(f"  ...GHG emissions reduction target: {ghg_limit_ub:,.0f} tCO2e")
            self.ghg_emissions_reduction_soft_constraints.append(
                self.gurobi_model.addConstr(self.ghg_emissions_expr - ghg_limit_ub <= self.E)
            )
            self.ghg_emissions_reduction_soft_constraints.append(
                self.gurobi_model.addConstr(ghg_limit_ub - self.ghg_emissions_expr <= self.E)
            )
        else:
            raise ValueError("Unknown choice for `GHG_CONSTRAINT_TYPE` setting: must be either 'hard' or 'soft'")


    def _add_biodiversity_limit_constraints(self):
        if settings.BIODIVERSTIY_TARGET_GBF_2 != "on":
            print('  ...biodiversity constraints target-2 TURNED OFF ...')
            return

        print('  ...biodiversity constraints...')

        # Returns biodiversity limits. Note that the biodiversity limits is 0 if BIODIVERSTIY_TARGET_GBF_2 != "on".
        biodiversity_limits = self._input_data.limits["biodiversity"]

        ag_contr = gp.quicksum(
            gp.quicksum(
                self._input_data.ag_b_mrj[0, :, :][:, j] * self.X_ag_dry_vars_jr[j, :]
            )  # Dryland agriculture contribution
            + gp.quicksum(
                self._input_data.ag_b_mrj[1, :, :][:, j] * self.X_ag_irr_vars_jr[j, :]
            )  # Irrigated agriculture contribution
            for j in range(self._input_data.n_ag_lus)
        )

        ag_man_contr = gp.quicksum(
            gp.quicksum(
                self._input_data.ag_man_b_mrj[am][0, :, j_idx]
                * self.X_ag_man_dry_vars_jr[am][j_idx, :]
            )  # Dryland alt. ag. management contributions
            + gp.quicksum(
                self._input_data.ag_man_b_mrj[am][1, :, j_idx]
                * self.X_ag_man_irr_vars_jr[am][j_idx, :]
            )  # Irrigated alt. ag. management contributions
            for am, am_j_list in self._input_data.am2j.items()
            for j_idx in range(len(am_j_list))
        )

        non_ag_contr = gp.quicksum(
            gp.quicksum(
                self._input_data.non_ag_b_rk[:, k] * self.X_non_ag_vars_kr[k, :]
            )  # Non-agricultural contribution
            for k in range(self._input_data.n_non_ag_lus)
        )

        self.biodiversity_expr = ag_contr + ag_man_contr + non_ag_contr

        print(f"    ...biodiversity target score: {biodiversity_limits:,.0f}")
        self.biodiversity_limit_constraint = self.gurobi_model.addConstr(
            self.biodiversity_expr >= biodiversity_limits
        )


    def update_formulation(
        self,
        input_data: SolverInputData,
        d_c: np.array,
        old_ag_x_mrj: np.ndarray,
        old_ag_man_lb_mrj: dict,
        old_non_ag_x_rk: np.ndarray,
        old_non_ag_lb_rk: np.ndarray,
        old_lumap: np.array,
        current_lumap: np.array,
        old_lmmap: np.array,
        current_lmmap: np.array,
    ):
        """
        Dynamically updates the existing formulation based on new input data and demands.
        """
        self._input_data = input_data
        self.d_c = d_c

        print('Updating variables...', flush=True)
        updated_cells = self._update_variables(
            old_ag_x_mrj,
            old_ag_man_lb_mrj,
            old_non_ag_x_rk,
            old_non_ag_lb_rk,
            old_lumap,
            current_lumap,
            old_lmmap,
            current_lmmap,
        )
        print('Updating constraints...', flush=True)
        self._update_constraints(updated_cells)

        print('Updating objective function...', flush=True)
        self._setup_objective()

    def _update_variables(
        self,
        old_ag_x_mrj: np.ndarray,
        old_ag_man_lb_mrj: dict,
        old_non_ag_x_rk: np.ndarray,
        old_non_ag_lb_rk: np.ndarray,
        old_lumap: np.array,
        current_lumap: np.array,
        old_lmmap: np.array,
        current_lmmap: np.array,
    ):
        """
        Updates the variables only for cells that have changed land use or land management.
        Returns an array of cells that have been updated.
        """
        # metrics
        num_cells_skipped = 0
        updated_cells = []

        for r in range(self._input_data.ncells):
            old_j = old_lumap[r]
            new_j = current_lumap[r]
            old_m = old_lmmap[r]
            new_m = current_lmmap[r]

            if (
                old_j == new_j
                and old_m == new_m
                and (old_ag_x_mrj[:, r, :] == self._input_data.ag_x_mrj[:, r, :]).all()
                and (old_non_ag_x_rk[r, :] == self._input_data.non_ag_x_rk[r, :]).all()
                and all(
                    old_non_ag_lb_rk[r, k] == self._input_data.non_ag_lb_rk[r, k]
                    for k, non_ag_desc in enumerate(NON_AG_LAND_USES) if not NON_AG_LAND_USES_REVERSIBLE[non_ag_desc]
                )
                and all(
                    (old_ag_man_lb_mrj.get(am)[:, r, :] == self._input_data.ag_man_lb_mrj.get(am)[:, r, :]).all()
                    for am in AG_MANAGEMENTS_TO_LAND_USES if not AG_MANAGEMENTS_REVERSIBLE[am]
                )
            ):
                # cell has not changed between years. No need to update variables
                num_cells_skipped += 1
                continue

            # agricultural land usage
            self.gurobi_model.remove(
                list(self.X_ag_dry_vars_jr[:, r][np.where(self.X_ag_dry_vars_jr[:, r])])
            )
            self.gurobi_model.remove(
                list(self.X_ag_irr_vars_jr[:, r][np.where(self.X_ag_irr_vars_jr[:, r])])
            )
            self.X_ag_dry_vars_jr[:, r] = np.zeros(self._input_data.n_ag_lus)
            self.X_ag_irr_vars_jr[:, r] = np.zeros(self._input_data.n_ag_lus)
            for j in range(self._input_data.n_ag_lus):
                if self._input_data.ag_x_mrj[0, r, j]:
                    self.X_ag_dry_vars_jr[j, r] = self.gurobi_model.addVar(
                        ub=1, name=f"X_ag_dry_{j}_{r}"
                    )

                if self._input_data.ag_x_mrj[1, r, j]:
                    self.X_ag_irr_vars_jr[j, r] = self.gurobi_model.addVar(
                        ub=1, name=f"X_ag_irr_{j}_{r}"
                    )

            # non-agricultural land usage
            self.gurobi_model.remove(
                list(self.X_non_ag_vars_kr[:, r][np.where(self.X_non_ag_vars_kr[:, r])])
            )
            self.X_non_ag_vars_kr[:, r] = np.zeros(self._input_data.n_non_ag_lus)
            for k, non_ag_lu_desc in zip(range(self._input_data.n_non_ag_lus), NON_AG_LAND_USES):
                if not NON_AG_LAND_USES[non_ag_lu_desc]:
                    continue

                if self._input_data.non_ag_x_rk[r, k]:
                    x_lb = (
                        0 if NON_AG_LAND_USES_REVERSIBLE[non_ag_lu_desc]
                        else self._input_data.non_ag_lb_rk[r, k]
                    )
                    self.X_non_ag_vars_kr[k, r] = self.gurobi_model.addVar(
                        lb=x_lb, ub=self._input_data.non_ag_x_rk[r, k], name=f"X_non_ag_{k}_{r}",
                    )

            # agricultural management
            for am, am_j_list in self._input_data.am2j.items():
                # remove old am variables
                self.gurobi_model.remove(
                    list(
                        self.X_ag_man_dry_vars_jr[am][:, r][
                            np.where(self.X_ag_man_dry_vars_jr[am][:, r])
                        ]
                    )
                )
                self.gurobi_model.remove(
                    list(
                        self.X_ag_man_irr_vars_jr[am][:, r][
                            np.where(self.X_ag_man_irr_vars_jr[am][:, r])
                        ]
                    )
                )
                self.X_ag_man_dry_vars_jr[am][:, r] = np.zeros(len(am_j_list))
                self.X_ag_man_irr_vars_jr[am][:, r] = np.zeros(len(am_j_list))

            for m, j in self._input_data.cells2ag_lu[r]:
                # replace am variables
                for am in self._input_data.j2am[j]:
                    if not AG_MANAGEMENTS[am]:
                        continue

                    # Get snake_case version of the AM name for the variable name
                    am_name = am.lower().replace(" ", "_")

                    x_lb = 0 if AG_MANAGEMENTS_REVERSIBLE[am] else self._input_data.ag_man_lb_mrj[am][m, r, j]
                    m_str = 'dry' if m == 0 else 'irr'
                    var_name = f"X_ag_man_{m_str}_{am_name}_{j}_{r}"

                    j_idx = self._input_data.am2j[am].index(j)
                    if m == 0:
                        self.X_ag_man_dry_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                            lb=x_lb, ub=1, name=var_name,
                        )
                    else:
                        self.X_ag_man_irr_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                            lb=x_lb, ub=1, name=var_name,
                        )

            updated_cells.append(r)

        updated_cells = np.array(updated_cells)
        print(f"    ...skipped {num_cells_skipped} cells, updated {len(updated_cells)} cells.\n")
        return updated_cells

    def _update_constraints(self, updated_cells: np.array):
        if len(updated_cells) == 0:
            print("No constraints need updating.")
            return

        print('  ...removing existing constraints...\n')
        for r in updated_cells:
            self.gurobi_model.remove(self.cell_usage_constraint_r.pop(r, []))
            self.gurobi_model.remove(self.ag_management_constraints_r.pop(r, []))

        self.gurobi_model.remove(self.adoption_limit_constraints)
        self.gurobi_model.remove(self.demand_penalty_constraints)
        if self.biodiversity_limit_constraint is not None:
            self.gurobi_model.remove(self.biodiversity_limit_constraint)
        if self.water_limit_constraints:
            self.gurobi_model.remove(self.water_limit_constraints)

        self.adoption_limit_constraints = []
        self.demand_penalty_constraints = []
        self.water_limit_constraints = []

        if self.ghg_emissions_limit_constraint_ub is not None:
            self.gurobi_model.remove(self.ghg_emissions_limit_constraint_ub)
            self.ghg_emissions_limit_constraint_ub = None
            
        if self.ghg_emissions_limit_constraint_lb is not None:
            self.gurobi_model.remove(self.ghg_emissions_limit_constraint_lb)
            self.ghg_emissions_limit_constraint_lb = None

        if len(self.ghg_emissions_reduction_soft_constraints) > 0:
            for constr in self.ghg_emissions_reduction_soft_constraints:
                self.gurobi_model.remove(constr)
            self.ghg_emissions_reduction_soft_constraints = []

        self._add_cell_usage_constraints(updated_cells)                 
        self._add_agricultural_management_constraints(updated_cells)    
        self._add_agricultural_management_adoption_limit_constraints()  
        self._add_demand_penalty_constraints()                          
        self._add_water_usage_limit_constraints() if settings.WATER_LIMITS == 'on' else print('  ...TURNING OFF water constraints...')
        self._add_ghg_emissions_limit_constraints()                    
        self._add_biodiversity_limit_constraints()                      

    def solve(self) -> SolverSolution:
        print("Starting solve...\n")

        # Magic.
        self.gurobi_model.optimize()

        print("Completed solve, collecting results...\n", flush=True)

        prod_data = {}  # Dictionary that stores information about production and GHG emissions for the write module

        # Collect optimised decision variables in one X_mrj Numpy array.
        X_dry_sol_rj = np.zeros(
            (self._input_data.ncells, self._input_data.n_ag_lus)
        ).astype(np.float32)
        X_irr_sol_rj = np.zeros(
            (self._input_data.ncells, self._input_data.n_ag_lus)
        ).astype(np.float32)
        non_ag_X_sol_rk = np.zeros(
            (self._input_data.ncells, self._input_data.n_non_ag_lus)
        ).astype(np.float32)
        am_X_dry_sol_rj = {
            am: np.zeros((self._input_data.ncells, self._input_data.n_ag_lus)).astype(np.float32)
            for am in self._input_data.am2j
        }
        am_X_irr_sol_rj = {
            am: np.zeros((self._input_data.ncells, self._input_data.n_ag_lus)).astype(np.float32)
            for am in self._input_data.am2j
        }

        # Get agricultural results
        for j in range(self._input_data.n_ag_lus):
            for r in self._input_data.ag_lu2cells[0, j]:
                X_dry_sol_rj[r, j] = self.X_ag_dry_vars_jr[j, r].X
            for r in self._input_data.ag_lu2cells[1, j]:
                X_irr_sol_rj[r, j] = self.X_ag_irr_vars_jr[j, r].X

        # Get non-agricultural results
        for k, lu in zip(range(self._input_data.n_non_ag_lus), settings.NON_AG_LAND_USES):
            if not settings.NON_AG_LAND_USES[lu]:
                non_ag_X_sol_rk[:, k] = np.zeros(self._input_data.ncells)
                continue

            for r in self._input_data.non_ag_lu2cells[k]:
                non_ag_X_sol_rk[r, k] = self.X_non_ag_vars_kr[k, r].X

        # Get agricultural management results
        for am, am_j_list in self._input_data.am2j.items():
            for j_idx, j in enumerate(am_j_list):
                for r in self._input_data.ag_lu2cells[0, j]:
                    am_X_dry_sol_rj[am][r, j] = self.X_ag_man_dry_vars_jr[am][j_idx, r].X
                for r in self._input_data.ag_lu2cells[1, j]:
                    am_X_irr_sol_rj[am][r, j] = self.X_ag_man_irr_vars_jr[am][j_idx, r].X

        """Note that output decision variables are mostly 0 or 1 but in some cases they are somewhere in between which creates issues
            when converting to maps etc. as individual cells can have non-zero values for multiple land-uses and land management type.
            This code creates a boolean X_mrj output matrix and ensure that each cell has one and only one land-use and land management"""

        # Process agricultural land usage information
        # Stack dryland and irrigated decision variables
        ag_X_mrj = np.stack((X_dry_sol_rj, X_irr_sol_rj))  # Float32
        ag_X_mrj_processed = ag_X_mrj

        ## Note - uncomment the following block of code to revert the processed agricultural variables to be binary.

        # ag_X_mrj_shape = ag_X_mrj.shape

        # # Reshape so that cells are along the first axis and land management and use are flattened along second axis i.e. (XXXXXXX,  56)
        # ag_X_mrj_processed = np.moveaxis(ag_X_mrj, 1, 0)
        # ag_X_mrj_processed = ag_X_mrj_processed.reshape(ag_X_mrj_processed.shape[0], -1)

        # # Boolean matrix where the maximum value for each cell across all land management types and land uses is True
        # ag_X_mrj_processed = ag_X_mrj_processed.argmax(axis=1)[:, np.newaxis] == range(
        #     ag_X_mrj_processed.shape[1]
        # )

        # # Reshape to mrj structure
        # ag_X_mrj_processed = ag_X_mrj_processed.reshape(
        #     (ag_X_mrj_shape[1], ag_X_mrj_shape[0], ag_X_mrj_shape[2])
        # )
        # ag_X_mrj_processed = np.moveaxis(ag_X_mrj_processed, 0, 1)

        # Process non-agricultural land usage information
        # Boolean matrix where the maximum value for each cell across all non-ag LUs is True
        non_ag_X_rk_processed = non_ag_X_sol_rk.argmax(axis=1)[:, np.newaxis] == range(
            self._input_data.n_non_ag_lus
        )

        # Make land use and land management maps
        # Vector indexed by cell that denotes whether the cell is non-agricultural land (True) or agricultural land (False)
        non_ag_bools_r = non_ag_X_sol_rk.max(axis=1) > ag_X_mrj.max(axis=(0, 2))

        # Update processed variables accordingly
        ag_X_mrj_processed[:, non_ag_bools_r, :] = False
        non_ag_X_rk_processed[~non_ag_bools_r, :] = False

        # Process agricultural management variables
        # Repeat the steps for the regular agricultural management variables
        ag_man_X_mrj_processed = {}
        for am in self._input_data.am2j:
            ag_man_processed = np.stack((am_X_dry_sol_rj[am], am_X_irr_sol_rj[am]))

            ## Note - uncomment the following block of code to revert the processed AM variables to be binary.

            # ag_man_X_shape = ag_man_processed.shape

            # ag_man_processed = np.moveaxis(ag_man_processed, 1, 0)
            # ag_man_processed = ag_man_processed.reshape(ag_man_processed.shape[0], -1)

            # ag_man_processed = (
            #        ag_man_processed.argmax(axis = 1)[:, np.newaxis]
            #     == range(ag_man_processed.shape[1])
            # )
            # ag_man_processed = ag_man_processed.reshape(
            #     (ag_man_X_shape[1], ag_man_X_shape[0], ag_man_X_shape[2])
            # )
            # ag_man_processed = np.moveaxis(ag_man_processed, 0, 1)
            ag_man_X_mrj_processed[am] = ag_man_processed

        # Calculate 1D array (maps) of land-use and land management, considering only agricultural LUs
        lumap = ag_X_mrj_processed.sum(axis=0).argmax(axis=1).astype("int8")
        lmmap = ag_X_mrj_processed.sum(axis=2).argmax(axis=0).astype("int8")

        # Update lxmaps and processed variable matrices to consider non-agricultural LUs
        lumap[non_ag_bools_r] = (
            non_ag_X_sol_rk[non_ag_bools_r, :].argmax(axis=1)
            + settings.NON_AGRICULTURAL_LU_BASE_CODE
        )
        lmmap[non_ag_bools_r] = 0  # Assume that all non-agricultural land uses are dryland

        # Process agricultural management usage info

        # Make ammaps (agricultural management maps) using the lumap and lmmap. There is a
        # separate ammap for each agricultural management option, because they can be stacked.
        ammaps = {am: np.zeros(self._input_data.ncells, dtype=np.int8) for am in AG_MANAGEMENTS_TO_LAND_USES}
        for r in range(self._input_data.ncells):
            cell_j = lumap[r]
            cell_m = lmmap[r]

            if cell_j >= settings.NON_AGRICULTURAL_LU_BASE_CODE:
                # Non agricultural land use - no agricultural management option
                continue

            for am in self._input_data.j2am[cell_j]:
                if cell_m == 0:
                    am_var_val = am_X_dry_sol_rj[am][r, cell_j]
                else:
                    am_var_val = am_X_irr_sol_rj[am][r, cell_j]

                if am_var_val >= settings.AGRICULTURAL_MANAGEMENT_USE_THRESHOLD:
                    ammaps[am][r] = 1

        # # Process production amount for each commodity
        prod_data["Production"] = [self.total_q_exprs_c[c].getValue() for c in range(self.ncms)]
        if self.ghg_emissions_expr:
            prod_data["GHG Emissions"] = self.ghg_emissions_expr.getValue()
        if self.biodiversity_expr:
            prod_data["Biodiversity"] = self.biodiversity_expr.getValue()

        return SolverSolution(
            lumap=lumap,
            lmmap=lmmap,
            ammaps=ammaps,
            ag_X_mrj=ag_X_mrj_processed,
            non_ag_X_rk=non_ag_X_sol_rk,
            ag_man_X_mrj=ag_man_X_mrj_processed,
            prod_data=prod_data,
            obj_val = {
                'SUM': self.gurobi_model.ObjVal,
                'Economy': self.obj_economy.getValue(),
                'Demand': self.obj_demand.getValue().sum()      if settings.DEMAND_CONSTRAINT_TYPE == 'soft' else 0,
                'GHG': self.obj_ghg.getValue()                  if settings.GHG_CONSTRAINT_TYPE == 'soft' else 0,            
            }
        )

    @property
    def ncms(self):
        return self.d_c.shape[0]  # Number of commodities.

    @property
    def X_dry_pr(self):
        """
        Transform agricultural decision vars from LU/j to PR/p representation.
        """
        return [
            self.X_ag_dry_vars_jr[j, :]
            for p in range(self._input_data.nprs)
            for j in range(self._input_data.n_ag_lus)
            if self._input_data.lu2pr_pj[p, j]
        ]

    @property
    def X_irr_pr(self):
        """
        Transform agricultural decision vars from LU/j to PR/p representation.
        """
        return [
            self.X_ag_irr_vars_jr[j, :]
            for p in range(self._input_data.nprs)
            for j in range(self._input_data.n_ag_lus)
            if self._input_data.lu2pr_pj[p, j]
        ]
