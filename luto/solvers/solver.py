# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J.,
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S.,
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.


"""
Provides minimalist Solver class and pure helper functions.
"""

import numpy as np
import gurobipy as gp
import luto.settings as settings

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Any
from gurobipy import GRB

from luto import tools
from luto.solvers.input_data import SolverInputData
from luto.settings import (
    AG_MANAGEMENTS, 
    AG_MANAGEMENTS_REVERSIBLE, 
    NON_AG_LAND_USES, 
    NON_AG_LAND_USES_REVERSIBLE
)


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
    prod_data: dict[str, Any]
    obj_val: dict[str, float]


class LutoSolver:
    """
    Class responsible for grouping the Gurobi model, relevant input data, and its variables.
    """

    def __init__(
        self,
        input_data: SolverInputData,
    ):

        self._input_data = input_data
        self.gurobi_model = gp.Model(f"LUTO {settings.VERSION}", env=gurenv)

        # Initialise variable stores
        self.X_ag_dry_vars_jr = None
        self.X_ag_irr_vars_jr = None
        self.X_non_ag_vars_kr = None
        self.X_ag_man_dry_vars_jr = None
        self.X_ag_man_irr_vars_jr = None
        self.V = None
        self.E = None
        self.B = None
        self.W = None

        # Initialise constraint lookups
        self.cell_usage_constraint_r = {}
        self.ag_management_constraints_r = defaultdict(list)
        self.adoption_limit_constraints = []
        self.demand_penalty_constraints = []
        self.water_limit_constraints = []
        self.water_nyiled_exprs = {}
        self.ghg_emissions_expr = None
        self.ghg_emissions_limit_constraint_ub = None
        self.ghg_emissions_limit_constraint_lb = None
        self.ghg_emissions_reduction_soft_constraints = []
        self.bio_GBF2_priority_degraded_area_expr = None
        self.bio_GBF2_priority_degraded_area_limit_constraint_hard = None
        self.bio_GBF2_priority_degraded_area_limit_constraint_soft = []
        self.bio_GBF3_major_vegetation_exprs = {}
        self.bio_GBF3_major_vegetation_limit_constraints = {}
        self.bio_GBF4_SNES_exprs = {}
        self.bio_GBF4_SNES_constrs = {}
        self.bio_GBF4_ECNES_exprs = {}
        self.bio_GBF4_ECNES_constrs = {}
        self.bio_GBF8_species_conservation_exprs = {}
        self.bio_GBF8_species_conservation_constrs = {}
        self.regional_adoption_constrs = []


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
        self._add_ghg_emissions_limit_constraints()
        self._add_biodiversity_constraints()
        self._add_regional_adoption_constraints()
        
        (
            self._add_water_usage_limit_constraints() 
            if settings.WATER_LIMITS == "on" 
            else print("  ...TURNING OFF water usage constraints ...")
        )


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

        for k, k_name in enumerate(NON_AG_LAND_USES):
            if not NON_AG_LAND_USES[k_name]:
                continue

            lu_cells = self._input_data.non_ag_lu2cells[k]
            for r in lu_cells:
                x_lb = (
                    0
                    if NON_AG_LAND_USES_REVERSIBLE[k_name]
                    else self._input_data.non_ag_lb_rk[r, k]
                )
                self.X_non_ag_vars_kr[k, r] = self.gurobi_model.addVar(
                    lb=x_lb,
                    ub=self._input_data.non_ag_x_rk[r, k],
                    name=f"X_non_ag_{k}_{r}",
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
                irr_lu_cells = self._input_data.ag_lu2cells[1, j]

                # for savanna burning, remove extra ineligible cells
                if am_name == "savanna_burning":
                    dry_lu_cells = np.intersect1d(
                        dry_lu_cells, self._input_data.savanna_eligible_r
                    )

                elif am_name == "hir_-_beef" or am_name == "hir_-_sheep":
                    dry_lu_cells = np.intersect1d(
                        dry_lu_cells, self._input_data.hir_eligible_r
                    )
                    irr_lu_cells = np.intersect1d(
                        irr_lu_cells, self._input_data.hir_eligible_r
                    )

                for r in dry_lu_cells:
                    dry_x_lb = (
                        0
                        if AG_MANAGEMENTS_REVERSIBLE[am]
                        else self._input_data.ag_man_lb_mrj[am][0, r, j]
                    )
                    self.X_ag_man_dry_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                        lb=dry_x_lb,
                        ub=1,
                        name=f"X_ag_man_dry_{am_name}_{j}_{r}",
                    )
                
                for r in irr_lu_cells:
                    irr_x_lb = (
                        0
                        if AG_MANAGEMENTS_REVERSIBLE[am]
                        else self._input_data.ag_man_lb_mrj[am][1, r, j]
                    )
                    self.X_ag_man_irr_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                        lb=irr_x_lb,
                        ub=1,
                        name=f"X_ag_man_irr_{am_name}_{j}_{r}",
                    )

    def _setup_deviation_penalties(self):
        """
        Decision variables, V, E and W, and B for soft constraints.
        1) [V] Penalty vector for demand, each one corespondes a commodity, that minimises the deviations from demand.
        2) [E] A single penalty scalar for GHG emissions, minimises its deviation from the target.
        3) [W] Penalty vector for water usage, each one corespondes a region, that minimises the deviations from the target.
        4) [B] A single penalty scalar for biodiversity, minimises its deviation from the target.
        """
        if settings.DEMAND_CONSTRAINT_TYPE == "soft":
            self.V = self.gurobi_model.addMVar(self._input_data.ncms, name="V")

        if settings.GHG_CONSTRAINT_TYPE == "soft":
            self.E = self.gurobi_model.addVar(name="E")

        if settings.WATER_CONSTRAINT_TYPE == "soft":
            num_regions = len(self._input_data.limits["water"].keys())
            self.W = self.gurobi_model.addMVar(num_regions, name="W")

        if settings.GBF2_CONSTRAINT_TYPE == "soft":
            self.B = self.gurobi_model.addVar(name="B", lb=0)   # lb=0 will force B to be positive, so that when using soft constraints, the objective function will be minimized


    def _setup_objective(self):
        """
        Formulate the objective based on settings.OBJECTIVE
        """
        print(f"Setting objective function to {settings.OBJECTIVE}...", flush=True)

        
        # Get objectives 
        self.obj_economy = self._setup_economy_objective() / self._input_data.base_yr_prod["BASE_YR Economy(AUD)"]              # Normalise to the base year economy value
        self.obj_biodiv = self._setup_biodiversity_objective() / self._input_data.base_yr_prod["BASE_YR Biodiversity (score)"]  # Normalise to the base year biodiversity value
        self.obj_penalties = self._setup_penalty_objectives()                                                                   
 
        # Set the objective function
        if settings.OBJECTIVE == "mincost":
            sense = GRB.MINIMIZE
            obj_wrap = (
                self.obj_economy  * settings.SOLVE_WEIGHT_ALPHA 
                - self.obj_biodiv * ( 1 - settings.SOLVE_WEIGHT_ALPHA)
            )
            objective = (
                obj_wrap * (1 - settings.SOLVE_WEIGHT_BETA) + 
                self.obj_penalties * settings.SOLVE_WEIGHT_BETA
            )
        elif settings.OBJECTIVE == "maxprofit":
            sense = GRB.MAXIMIZE
            obj_wrap = (
                self.obj_economy  * settings.SOLVE_WEIGHT_ALPHA 
                + self.obj_biodiv * (1 - settings.SOLVE_WEIGHT_ALPHA)
            )
            objective = (
                obj_wrap * (1 - settings.SOLVE_WEIGHT_BETA) 
                - self.obj_penalties * settings.SOLVE_WEIGHT_BETA
            )
        else:
            raise ValueError(f"Unknown objective function: {settings.OBJECTIVE}")

        self.gurobi_model.setObjective(objective, sense)
        
    def _setup_economy_objective(self):
        print("  ...setting up economy objective...")
        
        # Get economic contributions
        ag_obj_mrj, non_ag_obj_rk, ag_man_objs = self._input_data.economic_contr_mrj

        ag_exprs = []
        for j in range(self._input_data.n_ag_lus):
            ag_exprs.append(
                ag_obj_mrj[0, self._input_data.ag_lu2cells[0, j], j]
                @ self.X_ag_dry_vars_jr[j, self._input_data.ag_lu2cells[0, j]]
                + ag_obj_mrj[1, self._input_data.ag_lu2cells[1, j], j]
                @ self.X_ag_irr_vars_jr[j, self._input_data.ag_lu2cells[1, j]]
            )

        ag_mam_exprs = []
        for am, am_j_list in self._input_data.am2j.items():
            if not AG_MANAGEMENTS[am]:
                continue
            for j_idx, j in enumerate(am_j_list):
                ag_mam_exprs.append(
                    ag_man_objs[am][0, self._input_data.ag_lu2cells[0, j], j_idx]
                    @ self.X_ag_man_dry_vars_jr[am][j_idx, self._input_data.ag_lu2cells[0, j]]
                    + ag_man_objs[am][1, self._input_data.ag_lu2cells[1, j], j_idx]
                    @ self.X_ag_man_irr_vars_jr[am][j_idx, self._input_data.ag_lu2cells[1, j]]
                )

        non_ag_exprs = []
        for k, k_name in enumerate(NON_AG_LAND_USES):
            if not NON_AG_LAND_USES[k_name]:
                continue
            non_ag_exprs.append(
                non_ag_obj_rk[:, k][self._input_data.non_ag_lu2cells[k]]
                @ self.X_non_ag_vars_kr[k, self._input_data.non_ag_lu2cells[k]]
            )
        
        self.economy_ag_contr = gp.quicksum(ag_exprs)
        self.economy_ag_man_contr = gp.quicksum(ag_mam_exprs)
        self.economy_non_ag_contr = gp.quicksum(non_ag_exprs)
        return self.economy_ag_contr + self.economy_ag_man_contr + self.economy_non_ag_contr
    
    def _setup_biodiversity_objective(self):
        print("  ...setting up biodiversity objective...")
        
        ag_exprs = []
        for j in range(self._input_data.n_ag_lus):
            ag_exprs.append(
                gp.quicksum(
                    self._input_data.ag_b_mrj[0, :, j] * self.X_ag_dry_vars_jr[j, :]
                )  
                + gp.quicksum(
                    self._input_data.ag_b_mrj[1, :, j] * self.X_ag_irr_vars_jr[j, :]
                )
                
            )
 
        ag_mam_exprs = []
        for am, am_j_list in self._input_data.am2j.items():
            if not AG_MANAGEMENTS[am]:
                continue
                
            for j_idx in range(len(am_j_list)):
                ag_mam_exprs.append(
                    gp.quicksum(
                        self._input_data.ag_man_b_mrj[am][0, :, j_idx]
                        * self.X_ag_man_dry_vars_jr[am][j_idx, :]
                    )  # Dryland alt. ag. management contributions
                    + gp.quicksum(
                        self._input_data.ag_man_b_mrj[am][1, :, j_idx]
                        * self.X_ag_man_irr_vars_jr[am][j_idx, :]
                    )  # Irrigated alt. ag. management contributions   
                )
    
        non_ag_exprs = []
        for k,k_name in enumerate(NON_AG_LAND_USES):
            if not NON_AG_LAND_USES[k_name]:
                continue
            non_ag_exprs.append(
                gp.quicksum(
                    self._input_data.non_ag_b_rk[:, k] * self.X_non_ag_vars_kr[k, :]
                )
            )
        
        self.bio_ag_contr = gp.quicksum(ag_exprs)
        self.bio_ag_man_contr = gp.quicksum(ag_mam_exprs)
        self.bio_non_ag_contr = gp.quicksum(non_ag_exprs)
        
        return self.bio_ag_contr + self.bio_non_ag_contr + self.bio_ag_man_contr
        
    def _setup_penalty_objectives(self):
        print("  ...setting up penalty objectives...")
        
        # Get the penalty values for each sector
        self.penalty_biodiv = (
            self.B * settings.GBF2_PENALTY
            if settings.GBF2_CONSTRAINT_TYPE == "soft"
            else 0
        )
        self.penalty_ghg = (
            self.E * self._input_data.economic_target_yr_carbon_price
            if settings.GHG_CONSTRAINT_TYPE == "soft"
            else 0
        )
        self.penalty_water = (
            self.W.sum() * settings.WATER_PENALTY
            if settings.WATER_CONSTRAINT_TYPE == "soft"
            else 0
        )
        self.penalty_demand = (
            # gp.quicksum(
            #     v * price
            #     for v, price in zip(self.V, self._input_data.economic_BASE_YR_prices)
            # )
            (
                gp.quicksum(
                    (self.V[c] + self._input_data.base_yr_prod["BASE_YR Production (t)"][c]) 
                    / self._input_data.base_yr_prod["BASE_YR Production (t)"][c] 
                    for c in range(self._input_data.ncms)
                ) / self._input_data.ncms
            )
            if settings.DEMAND_CONSTRAINT_TYPE == "soft"
            else 0
        )

        return self.penalty_demand + self.penalty_ghg + self.penalty_water + self.penalty_biodiv + gp.LinExpr(0)
        

    def _add_cell_usage_constraints(self, cells: Optional[np.array] = None):
        """
        Constraint that all of every cell is used for some land use.
        If `cells` is provided, only adds constraints for the given cells
        """
        print("  ...cell usage constraints...")

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
            self.cell_usage_constraint_r[r] = self.gurobi_model.addConstr(
                expr == 1, 
                name=f"const_cell_usage_{r}"
            )

    def _add_agricultural_management_constraints(
        self, cells: Optional[np.array] = None
    ):
        """
        Constraint handling alternative agricultural management options:
        Ag. man. variables cannot exceed the value of the agricultural variable.
        """
        print("  ...agricultural management constraints...")

        for am, am_j_list in self._input_data.am2j.items():
            for j_idx, j in enumerate(am_j_list):
                if cells is not None:
                    lm_dry_r_vals = [
                        r for r in cells if self._input_data.ag_x_mrj[0, r, j]
                    ]
                    lm_irr_r_vals = [
                        r for r in cells if self._input_data.ag_x_mrj[1, r, j]
                    ]
                else:
                    lm_dry_r_vals = self._input_data.ag_lu2cells[0, j]
                    lm_irr_r_vals = self._input_data.ag_lu2cells[1, j]

                for r in lm_dry_r_vals:
                    constr = self.gurobi_model.addConstr(
                        self.X_ag_man_dry_vars_jr[am][j_idx, r]
                        <= self.X_ag_dry_vars_jr[j, r],
                        name=f"const_ag_mam_dry_usage_{am}_{j}_{r}",
                    )
                    self.ag_management_constraints_r[r].append(constr)
                for r in lm_irr_r_vals:
                    constr = self.gurobi_model.addConstr(
                        self.X_ag_man_irr_vars_jr[am][j_idx, r]
                        <= self.X_ag_irr_vars_jr[j, r],
                        name=f"const_ag_mam_irr_usage_{am}_{j}_{r}",
                    )
                    self.ag_management_constraints_r[r].append(constr)

    def _add_agricultural_management_adoption_limit_constraints(self):
        """
        Add adoption limits constraints for agricultural management options.
        """
        print("  ...agricultural management adoption constraints...")

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
                    ag_man_vars_sum <= adoption_limit * all_vars_sum,
                    name=f"const_ag_mam_adoption_limit_{am}_{j}",
                )

                self.adoption_limit_constraints.append(constr)

    def _add_demand_penalty_constraints(self):
        """
        Constraints to penalise under and over production compared to demand.
        """
        print("  ...demand constraints...")
        
        self.ag_q_c = [gp.LinExpr(0) for _ in range(self._input_data.ncms)]
        for j in range(self._input_data.n_ag_lus):
            X_ag_dry_r = self.X_ag_dry_vars_jr[j, :]
            X_ag_irr_r = self.X_ag_irr_vars_jr[j, :]
            
            for p in range(self._input_data.nprs):
                if not self._input_data.lu2pr_pj[p, j]:
                    continue
                ag_q_p = (
                    gp.quicksum(
                        self._input_data.ag_q_mrp[0, :, p] * X_ag_dry_r
                    ) 
                    + gp.quicksum(
                        self._input_data.ag_q_mrp[1, :, p] * X_ag_irr_r
                    ) 
                )  
                
                for c in range(self._input_data.ncms):
                    if not self._input_data.pr2cm_cp[c, p]:
                        continue
                    self.ag_q_c[c] += ag_q_p
                        
                        
        self.ag_man_q_c = [gp.LinExpr(0) for _ in range(self._input_data.ncms)]
        for am, am_j_list in self._input_data.am2j.items():
            if not AG_MANAGEMENTS[am]:
                continue
            
            for j_idx,j in enumerate(am_j_list):
                X_ag_mam_dry_r = self.X_ag_man_dry_vars_jr[am][j_idx, :]
                X_ag_mam_irr_r = self.X_ag_man_irr_vars_jr[am][j_idx, :]
                
                for p in range(self._input_data.nprs):
                    if not self._input_data.lu2pr_pj[p, j]:
                        continue
                    ag_mam_q_p = (
                        gp.quicksum(
                            self._input_data.ag_man_q_mrp[am][0, :, p] * X_ag_mam_dry_r
                        ) 
                        + gp.quicksum(
                            self._input_data.ag_man_q_mrp[am][1, :, p] * X_ag_mam_irr_r
                        ) 
                    )  
                    
                    for c in range(self._input_data.ncms):
                        if not self._input_data.pr2cm_cp[c, p]:
                            continue
                        self.ag_man_q_c[c] += ag_mam_q_p


        self.non_ag_q_c = [gp.LinExpr(0) for _ in range(self._input_data.ncms)]
        for k,k_name in enumerate(NON_AG_LAND_USES):
            if not NON_AG_LAND_USES[k_name]:
                continue
            
            for c in range(self._input_data.ncms):
                self.non_ag_q_c[c] += gp.quicksum(
                    self._input_data.non_ag_q_crk[c, :, k] * self.X_non_ag_vars_kr[k, :]
                )
            

        # Total quantities in CM/c representation.
        self.total_q_exprs_c = [
            self.ag_q_c[c]
            + self.ag_man_q_c[c] 
            + self.non_ag_q_c[c] 
            for c in range(self._input_data.ncms)
        ]

        if settings.DEMAND_CONSTRAINT_TYPE == "soft":
            upper_bound_constraints = self.gurobi_model.addConstrs(
                (
                    (self._input_data.limits['demand'][c] - self.total_q_exprs_c[c]) <= self.V[c] 
                    for c in range(self._input_data.ncms)
                ),  name="demand_soft_bound_upper"
                )
            lower_bound_constraints = self.gurobi_model.addConstrs(
                (
                    (self.total_q_exprs_c[c] - self._input_data.limits['demand'][c]) <= self.V[c] 
                    for c in range(self._input_data.ncms)
                ),  name="demand_soft_bound_lower"
            )

            self.demand_penalty_constraints.extend(upper_bound_constraints.values())
            self.demand_penalty_constraints.extend(lower_bound_constraints.values())

        elif settings.DEMAND_CONSTRAINT_TYPE == "hard":
            quantity_meets_demand_constraints = self.gurobi_model.addConstrs(
                (
                    (self.total_q_exprs_c[c] >= self._input_data.limits['demand'][c]) 
                    for c in range(self._input_data.ncms)
                ),
                    name="demand_meets_demand"
            )
            self.demand_penalty_constraints.extend(
                quantity_meets_demand_constraints.values()
            )

        else:
            raise ValueError(
                'DEMAND_CONSTRAINT_TYPE not specified in settings, needs to be "hard" or "soft"'
            )

    def _get_water_net_yield_expr_for_region(
        self,
        ind: np.ndarray,
    ) -> gp.LinExpr:
        """
        Get the Gurobi linear expression for the net water yield of a given region.
        """
        
        ag_exprs = []
        for j in range(self._input_data.n_ag_lus):
            ag_exprs.append(
                gp.quicksum(
                    self._input_data.ag_w_mrj[0, ind, j] * self.X_ag_dry_vars_jr[j, ind]
                )  # Dryland agriculture contribution
                + gp.quicksum(
                    self._input_data.ag_w_mrj[1, ind, j] * self.X_ag_irr_vars_jr[j, ind]
                )  # Irrigated agriculture contribution
            )
 
        ag_mam_exprs = []
        for am, am_j_list in self._input_data.am2j.items():
            if not AG_MANAGEMENTS[am]:
                continue
            
            for j_idx in range(len(am_j_list)):
                ag_mam_exprs.append(
                    gp.quicksum(
                        self._input_data.ag_man_w_mrj[am][0, ind, j_idx]
                        * self.X_ag_man_dry_vars_jr[am][j_idx, ind]
                    )  # Dryland alt. ag. management contributions
                    + gp.quicksum(
                        self._input_data.ag_man_w_mrj[am][1, ind, j_idx]
                        * self.X_ag_man_irr_vars_jr[am][j_idx, ind]
                    )  # Irrigated alt. ag. management contributions
                )

        non_ag_exprs = []
        for k in range(self._input_data.n_non_ag_lus):
            non_ag_exprs.append(
                gp.quicksum(
                    self._input_data.non_ag_w_rk[ind, k] * self.X_non_ag_vars_kr[k, ind]
                )  # Non-agricultural contribution
            )
        
        ag_contr = gp.quicksum(ag_exprs) 
        ag_man_contr = gp.quicksum(ag_mam_exprs)
        non_ag_contr = gp.quicksum(non_ag_exprs)
        return ag_contr + ag_man_contr + non_ag_contr


    def _add_water_usage_limit_constraints(self) -> None:
        
        print(" ... water usage constraints...")
        
        # Ensure water use remains below limit for each region
        for reg_idx, water_limit in self._input_data.limits["water"].items():
            
            ind = self._input_data.water_region_indices[reg_idx]
            reg_name = self._input_data.water_region_names[reg_idx]
            
            if water_limit == 0:
                print(f"     |-- target is {water_limit:15,.0f} ML for {reg_name} (skipped modelling)")
                continue

            print(f"     |-- target is {water_limit:15,.0f} ML for {reg_name}")

            self.water_nyiled_exprs[reg_idx] = self._get_water_net_yield_expr_for_region(ind)           # Water net yield inside LUTO study area

            if settings.WATER_CONSTRAINT_TYPE == "hard":
                constr = self.gurobi_model.addConstr(
                    self.water_nyiled_exprs[reg_idx] >= water_limit, 
                    name=f"water_yield_limit_{reg_idx}"
                )
            elif settings.WATER_CONSTRAINT_TYPE == "soft":
                constr = self.gurobi_model.addConstr(
                    water_limit - self.water_nyiled_exprs[reg_idx] <= self.W[reg_idx - 1],     # region index starts from 1
                    name=f"water_yield_limit_{reg_idx}"     
                )
            else:
                raise ValueError(
                    "Unknown choice for `WATER_CONSTRAINT_TYPE` setting: must be either 'hard' or 'soft'"
                ) 
                
            self.water_limit_constraints.append(constr)


    def _get_total_ghg_emissions_expr(self) -> gp.LinExpr:
        # Pre-calculate the coefficients for each variable,
        # both for regular culture and alternative agr. management options
        g_dry_coeff = (
            self._input_data.ag_g_mrj[0, :, :] + self._input_data.ag_ghg_t_mrj[0, :, :]
        )
        g_irr_coeff = (
            self._input_data.ag_g_mrj[1, :, :] + self._input_data.ag_ghg_t_mrj[1, :, :]
        )

        ghg_ag_exprs =[]
        for j in range(self._input_data.n_ag_lus):
            ghg_ag_exprs.append(
                gp.quicksum(
                    g_dry_coeff[:, j] * self.X_ag_dry_vars_jr[j, :]
                )
                + gp.quicksum(
                    g_irr_coeff[:, j] * self.X_ag_irr_vars_jr[j, :]
                )
            )
            
        ghg_ag_man_exprs = []
        for am, am_j_list in self._input_data.am2j.items():
            if not AG_MANAGEMENTS[am]:
                continue
            
            for j_idx in range(len(am_j_list)):
                ghg_ag_man_exprs.append(
                    gp.quicksum(
                        self._input_data.ag_man_g_mrj[am][0, :, j_idx]
                        * self.X_ag_man_dry_vars_jr[am][j_idx, :]
                    )  
                    + gp.quicksum(
                        self._input_data.ag_man_g_mrj[am][1, :, j_idx]
                        * self.X_ag_man_irr_vars_jr[am][j_idx, :]
                    )  
                )
                
        ghg_non_ag_exprs = []
        for k,k_name in enumerate(NON_AG_LAND_USES):
            if not NON_AG_LAND_USES[k_name]:
                continue
            ghg_non_ag_exprs.append(
                gp.quicksum(
                    self._input_data.non_ag_g_rk[:, k] * self.X_non_ag_vars_kr[k, :]
                )
            )
            
        self.ghg_ag_contr = gp.quicksum(ghg_ag_exprs)
        self.ghg_ag_man_contr = gp.quicksum(ghg_ag_man_exprs)
        self.ghg_non_ag_contr = gp.quicksum(ghg_non_ag_exprs)    
        
        return self.ghg_ag_contr + self.ghg_ag_man_contr + self.ghg_non_ag_contr + self._input_data.offland_ghg

    def _add_ghg_emissions_limit_constraints(self):
        """
        Add either hard or soft GHG constraints depending on settings.GHG_CONSTRAINT_TYPE
        """
        if settings.GHG_EMISSIONS_LIMITS == "off":
            print("...GHG emissions constraints TURNED OFF ...")
            return

        ghg_limit_ub = self._input_data.limits["ghg"]
        self.ghg_emissions_expr = self._get_total_ghg_emissions_expr()

        if settings.GHG_CONSTRAINT_TYPE == "hard":
            print(f"...GHG emissions reduction target")
            print(
                f"    ...GHG emissions reduction target UB: {ghg_limit_ub:,.0f} tCO2e"
            )
            self.ghg_emissions_limit_constraint_ub = self.gurobi_model.addConstr(
                self.ghg_emissions_expr <= ghg_limit_ub,
                name="ghg_emissions_limit_ub",
            )
        elif settings.GHG_CONSTRAINT_TYPE == "soft":
            print(f"  ...GHG emissions reduction target: {ghg_limit_ub:,.0f} tCO2e")
            self.ghg_emissions_reduction_soft_constraints.append(
                self.gurobi_model.addConstr(
                    self.ghg_emissions_expr - ghg_limit_ub <= self.E,
                    name="ghg_emissions_limit_soft_ub",
                )
            )
            self.ghg_emissions_reduction_soft_constraints.append(
                self.gurobi_model.addConstr(
                    ghg_limit_ub - self.ghg_emissions_expr <= self.E,
                    name="ghg_emissions_limit_soft_lb",
                )
            )
        else:
            raise ValueError(
                "Unknown choice for `GHG_CONSTRAINT_TYPE` setting: must be either 'hard' or 'soft'"
            )
            
            
    def _add_biodiversity_constraints(self) -> None:
        print("  ...biodiversity constraints...")
        self._add_GBF2_priority_degrade_areas_constraints()
        self._add_GBF3_major_vegetation_group_limit_constraints()
        self._add_GBF4_snes_constraints()
        self._add_GBF4_ecnes_constraints()
        self._add_GBF8_species_conservation_constraints()


    def _add_GBF2_priority_degrade_areas_constraints(self) -> None:
        
        if settings.BIODIVERSTIY_TARGET_GBF_2 == "off":
            print("    ...Biodiversity GBF 2 (conservation priority) constraints TURNED OFF ...")
            return

        print("    ...Biodiversity GBF 2 (conservation priority) constraints...")
        
        bio_ag_exprs = []
        bio_ag_man_exprs = []
        bio_non_ag_exprs = []
        
        for j in range(self._input_data.n_ag_lus):
            ind_dry = np.intersect1d(self._input_data.ag_lu2cells[0, j], self._input_data.priority_degraded_mask_idx)
            ind_irr = np.intersect1d(self._input_data.ag_lu2cells[1, j], self._input_data.priority_degraded_mask_idx)
            bio_ag_exprs.append(
                gp.quicksum(
                    self._input_data.GBF2_raw_priority_degraded_area_r[ind_dry]
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_dry_vars_jr[j, ind_dry]
                )
                + gp.quicksum(
                    self._input_data.GBF2_raw_priority_degraded_area_r[ind_irr]
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_irr_vars_jr[j, ind_irr]
                ) 
            )
        for am, am_j_list in self._input_data.am2j.items():
            if not AG_MANAGEMENTS[am]:
                continue
            for j_idx in range(len(am_j_list)):
                
                ind_dry = np.intersect1d(self._input_data.ag_lu2cells[0, j_idx], self._input_data.priority_degraded_mask_idx)
                ind_irr = np.intersect1d(self._input_data.ag_lu2cells[1, j_idx], self._input_data.priority_degraded_mask_idx)
                bio_ag_man_exprs.append(
                    gp.quicksum(self._input_data.GBF2_raw_priority_degraded_area_r[ind_dry]
                        * self._input_data.biodiv_contr_ag_man[am][j_idx][ind_dry]
                        * self.X_ag_man_dry_vars_jr[am][j_idx, ind_dry])
                    + gp.quicksum(
                        self._input_data.GBF2_raw_priority_degraded_area_r[ind_irr]
                        * self._input_data.biodiv_contr_ag_man[am][j_idx][ind_irr]
                        * self.X_ag_man_irr_vars_jr[am][j_idx, ind_irr]
                    )
                )  
        for k in range(self._input_data.n_non_ag_lus):
            ind = np.intersect1d(self._input_data.non_ag_lu2cells[k], self._input_data.priority_degraded_mask_idx)
            bio_non_ag_exprs.append(
                gp.quicksum(
                    self._input_data.GBF2_raw_priority_degraded_area_r[ind]
                    * self._input_data.biodiv_contr_non_ag_k[k]
                    * self.X_non_ag_vars_kr[k, ind]
                )
            )

        self.bio_GBF2_priority_degraded_area_expr = gp.quicksum(bio_ag_exprs) + gp.quicksum(bio_ag_man_exprs) + gp.quicksum(bio_non_ag_exprs)
        
        print(f"       |-- target is {self._input_data.limits["GBF2_priority_degrade_areas"]:15,.0f}")
        
        if settings.GBF2_CONSTRAINT_TYPE == "hard":
            self.bio_GBF2_priority_degraded_area_limit_constraint_hard = self.gurobi_model.addConstr(
                self.bio_GBF2_priority_degraded_area_expr >= self._input_data.limits["GBF2_priority_degrade_areas"], 
                name="bio_GBF2_priority_degraded_area_limit_hard"
            )
        elif settings.GBF2_CONSTRAINT_TYPE == "soft":
            constr = self.gurobi_model.addConstr(
                self._input_data.limits["GBF2_priority_degrade_areas"] - self.bio_GBF2_priority_degraded_area_expr <= self.B, 
                name="bio_GBF2_priority_degraded_area_limit_soft"
            )
            self.bio_GBF2_priority_degraded_area_limit_constraint_soft.append(constr)
        else:
            raise ValueError(
                f"Unknown value of GBF2_CONSTRAINT_TYPE. "
                f"Must be either 'hard' or 'soft'."
            )


    def _add_GBF3_major_vegetation_group_limit_constraints(self) -> None:
        if settings.BIODIVERSTIY_TARGET_GBF_3 == "off":
            print("    ...biodiversity GBF 3 (major vegetation group) constraints TURNED OFF ...")
            return

        v_limits = self._input_data.limits["GBF3_major_vegetation_groups"]
        v_names = self._input_data.GBF3_major_vegetation_groups_names
        v_ind = self._input_data.GBF3_major_vegetation_groups_ind

        print(f"    ...Biodiversity GBF 3 (major vegetation groups) constraints...")

        for v, v_area_lb in enumerate(v_limits):
            
            if v_limits[v] == 0:
                print(f"       |-- target is {v_area_lb:15,.0f} for {v_names[v]} (skipped modelling)  ")
                continue
            
            ind = v_ind[v]
            MVG_raw_area_r = self._input_data.GBF3_raw_MVG_area_vr[v, ind]

            ag_contr = gp.quicksum(
                gp.quicksum(
                    MVG_raw_area_r
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_dry_vars_jr[j, ind]
                )  # Dryland agriculture contribution
                + gp.quicksum(
                    MVG_raw_area_r
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_irr_vars_jr[j, ind]
                )  # Irrigated agriculture contribution
                for j in range(self._input_data.n_ag_lus)
            )

            ag_man_contr = gp.quicksum(
                gp.quicksum(
                    MVG_raw_area_r
                    * self._input_data.biodiv_contr_ag_man[am][j_idx][ind]
                    * self.X_ag_man_dry_vars_jr[am][j_idx, ind]
                )  # Dryland alt. ag. management contributions
                + gp.quicksum(
                    MVG_raw_area_r
                    * self._input_data.biodiv_contr_ag_man[am][j_idx][ind]
                    * self.X_ag_man_irr_vars_jr[am][j_idx, ind]
                )  # Irrigated alt. ag. management contributions
                for am, am_j_list in self._input_data.am2j.items()
                for j_idx in range(len(am_j_list))
            )

            non_ag_contr = gp.quicksum(
                gp.quicksum(
                    MVG_raw_area_r
                    * self._input_data.biodiv_contr_non_ag_k[k]
                    * self.X_non_ag_vars_kr[k, ind]
                )  # Non-agricultural contribution
                for k in range(self._input_data.n_non_ag_lus)
            )


            self.bio_GBF3_major_vegetation_exprs[v] = ag_contr + ag_man_contr + non_ag_contr 

            print(f"       |-- target is {v_area_lb:15,.0f} for {v_names[v]} ")
            self.bio_GBF3_major_vegetation_limit_constraints[v] = self.gurobi_model.addConstr(
                self.bio_GBF3_major_vegetation_exprs[v] >= v_area_lb,
                name=f"bio_GBF3_major_vegetation_limit_{v}",
            )


    def _add_GBF4_snes_constraints(self) -> None:
        if settings.BIODIVERSTIY_TARGET_GBF_4_SNES != "on":
            print('    ...Biodiversity GBF 4 (Species of National Environmental Significance) constraints TURNED OFF ...')
            return
        
        x_limits = self._input_data.limits["GBF4_SNES"]
        x_names = self._input_data.GBF4_SNES_names

        print(f"    ...Biodiversity GBF 4 (Species of National Environmental Significance) constraints...")
        
        for x, x_area_lb in enumerate(x_limits):
            ind = np.where(self._input_data.GBF4_SNES_xr[x] > 0)[0]

            if ind.size == 0:
                print(
                    f"        |-- WARNING: SNES species NOT added because of empty layer for {x_names[x]}")
                continue
            
            ag_contr = gp.quicksum(
                gp.quicksum(
                    self._input_data.GBF4_SNES_xr[x, ind]
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_dry_vars_jr[j, ind]
                )  # Dryland agriculture contribution
                + gp.quicksum(
                    self._input_data.GBF4_SNES_xr[x, ind]
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_irr_vars_jr[j, ind]
                )  # Irrigated agriculture contribution
                for j in range(self._input_data.n_ag_lus)
            )

            ag_man_contr = gp.quicksum(
                gp.quicksum(
                    self._input_data.GBF4_SNES_xr[x, ind]
                    * self._input_data.biodiv_contr_ag_man[am][j_idx][ind]
                    * self.X_ag_man_dry_vars_jr[am][j_idx, ind]
                )  # Dryland alt. ag. management contributions
                + gp.quicksum(
                    self._input_data.GBF4_SNES_xr[x, ind]
                    * self._input_data.biodiv_contr_ag_man[am][j_idx][ind]
                    * self.X_ag_man_irr_vars_jr[am][j_idx, ind]
                )  # Irrigated alt. ag. management contributions
                for am, am_j_list in self._input_data.am2j.items()
                for j_idx in range(len(am_j_list))
            )

            non_ag_contr = gp.quicksum(
                gp.quicksum(
                    self._input_data.GBF4_SNES_xr[x, ind]
                    * self._input_data.biodiv_contr_non_ag_k[k]
                    * self.X_non_ag_vars_kr[k, ind]
                )  # Non-agricultural contribution
                for k in range(self._input_data.n_non_ag_lus)
            )

            self.bio_GBF4_SNES_exprs[x] = (ag_contr + ag_man_contr + non_ag_contr) / x_area_lb  

            print(f"       |-- target is {x_area_lb:15,.0f} for {x_names[x]}")
            self.bio_GBF4_SNES_constrs[x] = self.gurobi_model.addConstr(
                self.bio_GBF4_SNES_exprs[x] >= 1,
                name=f"bio_GBF4_SNES_limit_{x}",
            )

    def _add_GBF4_ecnes_constraints(self) -> None:
        if settings.BIODIVERSTIY_TARGET_GBF_4_ECNES != "on":
            print('    ...Biodiversity GBF 4 (Ecological Communities of National Environmental Significance) constraints TURNED OFF ...')
            return
        
        x_limits = self._input_data.limits["GBF4_ECNES"]
        x_names = self._input_data.GBF4_ECNES_names

        print(f"    ...Biodiversity GBF 4 (Ecological Communities of National Environmental Significance) constraints...")
        
        for x, x_area_lb in enumerate(x_limits):
            ind = np.where(self._input_data.GBF4_ECNES_xr[x] > 0)[0]

            if ind.size == 0:
                print(
                    f"       |-- WARNING: ECNES species was NOT added because of empty layer for {x_names[x]}")
                continue
            
            ag_contr = gp.quicksum(
                gp.quicksum(
                    self._input_data.GBF4_ECNES_xr[x, ind]
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_dry_vars_jr[j, ind]
                )  # Dryland agriculture contribution
                + gp.quicksum(
                    self._input_data.GBF4_ECNES_xr[x, ind]
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_irr_vars_jr[j, ind]
                )  # Irrigated agriculture contribution
                for j in range(self._input_data.n_ag_lus)
            )

            ag_man_contr = gp.quicksum(
                gp.quicksum(
                    self._input_data.GBF4_ECNES_xr[x, ind]
                    * self._input_data.biodiv_contr_ag_man[am][j_idx][ind]
                    * self.X_ag_man_dry_vars_jr[am][j_idx, ind]
                )  # Dryland alt. ag. management contributions
                + gp.quicksum(
                    self._input_data.GBF4_ECNES_xr[x, ind]
                    * self._input_data.biodiv_contr_ag_man[am][j_idx][ind]
                    * self.X_ag_man_irr_vars_jr[am][j_idx, ind]
                )  # Irrigated alt. ag. management contributions
                for am, am_j_list in self._input_data.am2j.items()
                for j_idx in range(len(am_j_list))
            )

            non_ag_contr = gp.quicksum(
                gp.quicksum(
                    self._input_data.GBF4_ECNES_xr[x, ind]
                    * self._input_data.biodiv_contr_non_ag_k[k]
                    * self.X_non_ag_vars_kr[k, ind]
                )  # Non-agricultural contribution
                for k in range(self._input_data.n_non_ag_lus)
            )

            self.bio_GBF4_ECNES_exprs[x] = (ag_contr + ag_man_contr + non_ag_contr) / x_area_lb


            print(f"       |-- target is {x_area_lb:15,.0f} for {x_names[x]} ")
            self.bio_GBF4_ECNES_constrs[x] = self.gurobi_model.addConstr(
                self.bio_GBF4_ECNES_exprs[x] >= 1,
                name=f"bio_GBF4_ECNES_limit_{x}",
            )


    def _add_GBF8_species_conservation_constraints(self) -> None:
                
        if settings.BIODIVERSTIY_TARGET_GBF_8 != "on":
            print('    ...Biodiversity GBF 8 (climate change impact on species conservation) constraints TURNED OFF ...')
            return
        
        s_limits = self._input_data.limits["GBF8_species_conservation"]
        s_names = self._input_data.GBF8_species_names
        s_ind = self._input_data.GBF8_species_indices

        print(f"    ...Biodiversity GBF 8 (climate change impact on species conservation) constraints...")
        
        for s, s_area_lb in enumerate(s_limits):
            
            ind = s_ind[s]
            GBF8_raw_area_r = self._input_data.GBF8_raw_species_area_sr[s, ind]
            
            ag_contr = gp.quicksum(
                gp.quicksum(
                    GBF8_raw_area_r
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_dry_vars_jr[j, ind]
                )  # Dryland agriculture contribution
                + gp.quicksum(
                    GBF8_raw_area_r
                    * self._input_data.biodiv_contr_ag_j[j]
                    * self.X_ag_irr_vars_jr[j, ind]
                )  # Irrigated agriculture contribution
                for j in range(self._input_data.n_ag_lus)
            )

            ag_man_contr = gp.quicksum(
                gp.quicksum(
                    GBF8_raw_area_r
                    * self._input_data.biodiv_contr_ag_man[am][j_idx][ind]
                    * self.X_ag_man_dry_vars_jr[am][j_idx, ind]
                )  # Dryland alt. ag. management contributions
                + gp.quicksum(
                    GBF8_raw_area_r
                    * self._input_data.biodiv_contr_ag_man[am][j_idx][ind]
                    * self.X_ag_man_irr_vars_jr[am][j_idx, ind]
                )  # Irrigated alt. ag. management contributions
                for am, am_j_list in self._input_data.am2j.items()
                for j_idx in range(len(am_j_list))
            )

            non_ag_contr = gp.quicksum(
                gp.quicksum(
                    GBF8_raw_area_r
                    * self._input_data.biodiv_contr_non_ag_k[k]
                    * self.X_non_ag_vars_kr[k, ind]
                )  # Non-agricultural contribution
                for k in range(self._input_data.n_non_ag_lus)
            )

            # Divide by constant to reduce strain on the constraint matrix range
            self.bio_GBF8_species_conservation_exprs[s] = (ag_contr + ag_man_contr + non_ag_contr) / s_area_lb
    
            print(f"       |-- target is {s_area_lb:15,.0f} for {s_names[s]}")
            self.bio_GBF8_species_conservation_constrs[s] = self.gurobi_model.addConstr(
                self.bio_GBF8_species_conservation_exprs[s] >= 1,
                name=f"bio_GBF8_species_conservation_limit_{s}",
            )

        


    def _add_regional_adoption_constraints(self) -> None:

        if settings.REGIONAL_ADOPTION_CONSTRAINTS != "on":
            print("  ...regional adoption constraints TURNED OFF...")
            return
        
        print("  ...regional adoption constraints...")
        
        # Add adoption constraints for agricultural land uses
        reg_adopt_limits = self._input_data.limits["ag_regional_adoption"]
        for reg_id, j, lu_name, reg_ind, reg_area_limit in reg_adopt_limits:
            print(f"     |-- adding adoption limit for {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} >= {reg_area_limit:,.0f} HA...")
            reg_expr = (
                  gp.quicksum(self._input_data.real_area[reg_ind] * self.X_ag_dry_vars_jr[j, reg_ind])
                + gp.quicksum(self._input_data.real_area[reg_ind] * self.X_ag_irr_vars_jr[j, reg_ind])
            )
            self.regional_adoption_constrs.append(self.gurobi_model.addConstr(reg_expr <= reg_area_limit, name=f"reg_adopt_limit_ag_{lu_name}_{reg_id}"))
        
        # Add adoption constraints for non-agricultural land uses
        reg_adopt_limits = self._input_data.limits["non_ag_regional_adoption"]
        for reg_id, k, lu_name, reg_ind, reg_area_limit in reg_adopt_limits:
            print(f"     |-- adding adoption limit for {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} >= {reg_area_limit:,.0f} HA...")
            reg_expr = gp.quicksum(self._input_data.real_area[reg_ind] * self.X_non_ag_vars_kr[k, reg_ind])
            self.regional_adoption_constrs.append(self.gurobi_model.addConstr(reg_expr <= reg_area_limit, name=f"reg_adopt_limit_non_ag_{lu_name}_{reg_id}"))




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
        self._input_data.limits['demand'] = d_c

        print("Updating variables...", flush=True)
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
        print("Updating constraints...", flush=True)
        self._update_constraints(updated_cells)

        print("Updating objective function...", flush=True)
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
                    for k, k_name in enumerate(NON_AG_LAND_USES)
                    if not NON_AG_LAND_USES_REVERSIBLE[k_name]
                )
                and all(
                    (old_ag_man_lb_mrj.get(am)[:, r, :] == self._input_data.ag_man_lb_mrj.get(am)[:, r, :]).all()
                    for am in (i for i in AG_MANAGEMENTS if AG_MANAGEMENTS[i])
                    if not AG_MANAGEMENTS_REVERSIBLE[am]
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
            for k, k_name in enumerate(NON_AG_LAND_USES):
                if not NON_AG_LAND_USES[k_name]:
                    continue

                if self._input_data.non_ag_x_rk[r, k]:
                    x_lb = (
                        0
                        if NON_AG_LAND_USES_REVERSIBLE[k_name]
                        else self._input_data.non_ag_lb_rk[r, k]
                    )
                    self.X_non_ag_vars_kr[k, r] = self.gurobi_model.addVar(
                        lb=x_lb,
                        ub=self._input_data.non_ag_x_rk[r, k],
                        name=f"X_non_ag_{k}_{r}",
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

                    x_lb = (
                        0
                        if AG_MANAGEMENTS_REVERSIBLE[am]
                        else self._input_data.ag_man_lb_mrj[am][m, r, j]
                    )
                    m_str = "dry" if m == 0 else "irr"
                    var_name = f"X_ag_man_{m_str}_{am_name}_{j}_{r}"

                    j_idx = self._input_data.am2j[am].index(j)
                    if m == 0:
                        self.X_ag_man_dry_vars_jr[am][j_idx, r] = (
                            self.gurobi_model.addVar(
                                lb=x_lb,
                                ub=1,
                                name=var_name,
                            )
                        )
                    else:
                        self.X_ag_man_irr_vars_jr[am][j_idx, r] = (
                            self.gurobi_model.addVar(
                                lb=x_lb,
                                ub=1,
                                name=var_name,
                            )
                        )

            updated_cells.append(r)

        updated_cells = np.array(updated_cells)
        print(f"    ...skipped {num_cells_skipped} cells, updated {len(updated_cells)} cells.\n")
        return updated_cells

    def _update_constraints(self, updated_cells: np.array):
        if len(updated_cells) == 0:
            print("No constraints need updating.")
            return

        print("  ...removing existing constraints...\n")
        for r in updated_cells:
            self.gurobi_model.remove(self.cell_usage_constraint_r.pop(r, []))
            self.gurobi_model.remove(self.ag_management_constraints_r.pop(r, []))

        self.gurobi_model.remove(self.adoption_limit_constraints)
        self.gurobi_model.remove(self.demand_penalty_constraints)
        if self.bio_GBF2_priority_degraded_area_limit_constraint_hard is not None:
            self.gurobi_model.remove(self.bio_GBF2_priority_degraded_area_limit_constraint_hard)
        if self.water_limit_constraints:
            self.gurobi_model.remove(self.water_limit_constraints)
        if self.bio_GBF3_major_vegetation_limit_constraints:
            for constr in self.bio_GBF3_major_vegetation_limit_constraints.values():
                self.gurobi_model.remove(constr)
        if self.bio_GBF4_SNES_constrs:
            for constr in self.bio_GBF4_SNES_constrs.values():
                self.gurobi_model.remove(constr)
        if self.bio_GBF4_ECNES_constrs:
            for constr in self.bio_GBF4_ECNES_constrs.values():
                self.gurobi_model.remove(constr)
        if self.bio_GBF8_species_conservation_constrs:
            for constr in self.bio_GBF8_species_conservation_constrs.values():
                self.gurobi_model.remove(constr)
        

        self.adoption_limit_constraints = []
        self.demand_penalty_constraints = []
        self.water_limit_constraints = []
        self.bio_GBF3_major_vegetation_exprs = {}
        self.bio_GBF3_major_vegetation_limit_constraints = {}
        self.bio_GBF8_species_conservation_exprs = {}
        self.bio_GBF8_species_conservation_constrs = {}
        self.bio_GBF4_SNES_exprs = {}
        self.bio_GBF4_SNES_constrs = {}
        self.bio_GBF4_ECNES_exprs = {}
        self.bio_GBF4_ECNES_constrs = {}

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

        if len(self.bio_GBF2_priority_degraded_area_limit_constraint_soft) > 0:
            for constr in self.bio_GBF2_priority_degraded_area_limit_constraint_soft:
                self.gurobi_model.remove(constr)
            self.bio_GBF2_priority_degraded_area_limit_constraint_soft = []

        if self.regional_adoption_constrs:
            self.gurobi_model.remove(self.regional_adoption_constrs)

        self.regional_adoption_constrs = []

        self._add_cell_usage_constraints(updated_cells)
        self._add_agricultural_management_constraints(updated_cells)
        self._add_agricultural_management_adoption_limit_constraints()
        self._add_demand_penalty_constraints()
        (
            self._add_water_usage_limit_constraints()
            if settings.WATER_LIMITS == "on"
            else print("  ...TURNING OFF water constraints...")
        )
        self._add_ghg_emissions_limit_constraints()
        self._add_biodiversity_constraints()
        self._add_regional_adoption_constraints()

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
            am: np.zeros((self._input_data.ncells, self._input_data.n_ag_lus)).astype(
                np.float32
            )
            for am in self._input_data.am2j
        }
        am_X_irr_sol_rj = {
            am: np.zeros((self._input_data.ncells, self._input_data.n_ag_lus)).astype(
                np.float32
            )
            for am in self._input_data.am2j
        }

        # Get agricultural results
        for j in range(self._input_data.n_ag_lus):
            for r in self._input_data.ag_lu2cells[0, j]:
                X_dry_sol_rj[r, j] = self.X_ag_dry_vars_jr[j, r].X
            for r in self._input_data.ag_lu2cells[1, j]:
                X_irr_sol_rj[r, j] = self.X_ag_irr_vars_jr[j, r].X

        # Get non-agricultural results
        for k, lu in enumerate(settings.NON_AG_LAND_USES):
            if not settings.NON_AG_LAND_USES[lu]:
                non_ag_X_sol_rk[:, k] = np.zeros(self._input_data.ncells)
                continue

            for r in self._input_data.non_ag_lu2cells[k]:
                non_ag_X_sol_rk[r, k] = self.X_non_ag_vars_kr[k, r].X

        # Get agricultural management results
        for am, am_j_list in self._input_data.am2j.items():
            for j_idx, j in enumerate(am_j_list):
                eligible_dry_cells = self._input_data.ag_lu2cells[0, j]
                eligible_irr_cells = self._input_data.ag_lu2cells[1, j]

                if am == "Savanna Burning":
                    eligible_dry_cells = np.intersect1d(
                        eligible_dry_cells, self._input_data.savanna_eligible_r
                    )
                    eligible_irr_cells = np.intersect1d(
                        eligible_irr_cells, self._input_data.savanna_eligible_r
                    )

                elif am == "HIR - Beef" or am == "HIR - Sheep":
                    eligible_dry_cells = np.intersect1d(
                        eligible_dry_cells, self._input_data.hir_eligible_r
                    )
                    eligible_irr_cells = np.intersect1d(
                        eligible_irr_cells, self._input_data.hir_eligible_r
                    )

                for r in eligible_dry_cells:
                    am_X_dry_sol_rj[am][r, j] = self.X_ag_man_dry_vars_jr[am][
                        j_idx, r
                    ].X
                for r in eligible_irr_cells:
                    am_X_irr_sol_rj[am][r, j] = self.X_ag_man_irr_vars_jr[am][
                        j_idx, r
                    ].X

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
        ammaps = {
            am: np.zeros(self._input_data.ncells, dtype=np.int8)
            for am in AG_MANAGEMENTS
        }
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

        # Process production amount for each commodity
        prod_data["Production"] = [self.total_q_exprs_c[c].getValue() for c in range(self._input_data.ncms)]
        if self.ghg_emissions_expr:
            prod_data["GHG Emissions"] = self.ghg_emissions_expr.getValue()
        if self.bio_GBF2_priority_degraded_area_expr:
            prod_data["Biodiversity"] = self.bio_GBF2_priority_degraded_area_expr.getValue()
        if self.bio_GBF3_major_vegetation_exprs:
            prod_data["Major Vegetation Groups"] = {
                v: expr.getValue() for v, expr in self.bio_GBF3_major_vegetation_exprs.items()
            }
        if self.bio_GBF8_species_conservation_exprs:
            prod_data["Species Conservation"] = {
                s: expr.getValue() for s, expr in self.bio_GBF8_species_conservation_exprs.items()
            }
        if self.bio_GBF4_SNES_exprs:
            prod_data["GBF4_SNES"] = {
                z: expr.getValue() for z, expr in self.bio_GBF4_SNES_exprs.items()
            }
        if self.bio_GBF4_ECNES_exprs:
            prod_data["GBF4_ECNES"] = {
                z: expr.getValue() for z, expr in self.bio_GBF4_ECNES_exprs.items()
            }

        return SolverSolution(
            lumap=lumap,
            lmmap=lmmap,
            ammaps=ammaps,
            ag_X_mrj=ag_X_mrj_processed,
            non_ag_X_rk=non_ag_X_sol_rk,
            ag_man_X_mrj=ag_man_X_mrj_processed,
            prod_data=prod_data,
            obj_val={
                "ObjVal": self.gurobi_model.ObjVal,
                
                "Economy Total Value (AUD)": self.obj_economy.getValue(),
                'Economy Ag Value (AUD)': self.economy_ag_contr.getValue(),
                'Economy Non-Ag Value (AUD)': self.economy_non_ag_contr.getValue(),
                'Economy Ag-Man Value (AUD)': self.economy_ag_man_contr.getValue(),
                "Economy Total Objective": self.obj_economy.getValue() * settings.SOLVE_WEIGHT_ALPHA,
                
                "Biodiversity Total Priority Score (score)": self.obj_biodiv.getValue(),
                "Biodiversity Ag Priority Score (score)": self.bio_ag_contr.getValue(),
                "Biodiversity Non-Ag Priority Score (score)": self.bio_non_ag_contr.getValue(),
                "Biodiversity Ag-Man Priority Score (score)": self.bio_ag_man_contr.getValue(),
                "Biodiversity Total Objective": self.obj_biodiv.getValue() * settings.SOLVE_WEIGHT_BETA,
                
                "Penalties Value (AUD)": self.obj_penalties.getValue(),
                "Penalties Objective": self.obj_penalties.getValue() * (1 - settings.SOLVE_WEIGHT_ALPHA),
                
                "Production Ag Value (t)":          {c:self.ag_q_c[c].getValue() for c in range(self._input_data.ncms)},
                "Production Non-Ag Value (t)":      {c:self.non_ag_q_c[c].getValue() for c in range(self._input_data.ncms)},
                "Production Ag-Mam Value (t)":      {c:self.ag_man_q_c[c] .getValue() for c in range(self._input_data.ncms)},
                "Production Deviation (t)":         (self.V.X if settings.DEMAND_CONSTRAINT_TYPE == "soft" else 0),
                "Production Penalty":               (self.penalty_demand.getValue() * (1 - settings.SOLVE_WEIGHT_ALPHA)              if settings.DEMAND_CONSTRAINT_TYPE == "soft" else 0),
                            
                "Water value (ML)":                 ({k: v.getValue() for k,v in self.water_nyiled_exprs.items()}                    if settings.WATER_LIMITS == "on" else 0),
                "Water Deviation (ML)":             (self.W.X                                                                        if settings.WATER_CONSTRAINT_TYPE == "soft" else 0),
                "Water Penalty":                    (self.penalty_water.getValue() * (1 - settings.SOLVE_WEIGHT_ALPHA)               if settings.WATER_CONSTRAINT_TYPE == "soft" else 0),
                            
                "GHG Ag Value (tCO2e)":             (self.ghg_ag_contr.getValue()                                                    if settings.GHG_EMISSIONS_LIMITS != "off" else 0),
                "GHG Non-Ag Value (tCO2e)":         (self.ghg_non_ag_contr.getValue()                                                if settings.GHG_EMISSIONS_LIMITS != "off" else 0),
                "GHG Ag-Mam Value t(CO2e)":         (self.ghg_ag_man_contr.getValue()                                                if settings.GHG_EMISSIONS_LIMITS != "off" else 0),    
                "GHG Deviation (tCO2e)":            (self.E.X                                                                        if settings.GHG_CONSTRAINT_TYPE == "soft" else 0),
                "GHG Penalty":                      (self.penalty_ghg.getValue() * (1 - settings.SOLVE_WEIGHT_ALPHA)                 if settings.GHG_CONSTRAINT_TYPE == "soft" else 0),
            
                "BIO (GBF2) value (ha)":            (0                                                                               if settings.BIODIVERSTIY_TARGET_GBF_2 == "off" else self.bio_GBF2_priority_degraded_area_expr.getValue()),
                "BIO (GBF2) Deviation (ha)":        (self.B.X                                                                        if settings.GBF2_CONSTRAINT_TYPE == "soft" else 0),
                "BIO (GBF2) Penalty":               (self.penalty_biodiv.getValue() * (1 - settings.SOLVE_WEIGHT_ALPHA)              if settings.GBF2_CONSTRAINT_TYPE == "soft" else 0),
                "BIO (GBF3) value (ha)":            (0                                                                               if settings.BIODIVERSTIY_TARGET_GBF_3 == "off" else {k: v.getValue() for k,v in self.bio_GBF3_major_vegetation_exprs.items()}),
                "BIO (GBF4) SNES value (ha)":       ({k: v.getValue() for k,v in self.bio_GBF4_SNES_exprs.items()}                   if settings.BIODIVERSTIY_TARGET_GBF_4_SNES == "on" else 0),
                "BIO (GBF4) ECNES value (ha)":      ({k: v.getValue() for k,v in self.bio_GBF4_ECNES_exprs.items()}                  if settings.BIODIVERSTIY_TARGET_GBF_4_ECNES == "on" else 0),
                "BIO (GBF8) value (ha)":            ({k: v.getValue() for k,v in self.bio_GBF8_species_conservation_exprs.items()}   if settings.BIODIVERSTIY_TARGET_GBF_8 == "on" else 0),
            },
        )

