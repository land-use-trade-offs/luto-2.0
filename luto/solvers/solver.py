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
from typing import Any
import numpy as np
import pandas as pd
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from functools import cached_property

import luto.settings as settings
from luto.ag_managements import SORTED_AG_MANAGEMENTS, AG_MANAGEMENTS_TO_LAND_USES

# Set Gurobi environment.
gurenv = gp.Env(logfilename="gurobi.log", empty=True)  # (empty = True)
gurenv.setParam("Method", settings.SOLVE_METHOD)
gurenv.setParam("OutputFlag", settings.VERBOSE)
gurenv.setParam("OptimalityTol", settings.OPTIMALITY_TOLERANCE)
gurenv.setParam("Threads", settings.THREADS)
gurenv.start()


@dataclass
class InputData:
    """
    An object that collects and stores all relevant data for solver.py.

    ag_t_mrj: np.ndarray            # Agricultural transition cost matrices.
    ag_c_mrj: np.ndarray            # Agricultural production cost matrices.
    ag_r_mrj: np.ndarray            # Agricultural production revenue matrices.
    ag_g_mrj: np.ndarray            # Agricultural greenhouse gas emissions matrices.
    ag_w_mrj: np.ndarray            # Agricultural water requirements matrices.
    ag_x_mrj: np.ndarray            # Agricultural exclude matrices.
    ag_q_mrp: np.ndarray            # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
    ag_ghg_t_mrj: np.ndarray        # GHG emissions released during transitions between agricultural land uses.
    """

    ag_t_mrj: np.ndarray  # Agricultural transition cost matrices.
    ag_c_mrj: np.ndarray  # Agricultural production cost matrices.
    ag_r_mrj: np.ndarray  # Agricultural production revenue matrices.
    ag_g_mrj: np.ndarray  # Agricultural greenhouse gas emissions matrices.
    ag_w_mrj: np.ndarray  # Agricultural water requirements matrices.
    ag_x_mrj: np.ndarray  # Agricultural exclude matrices.
    ag_q_mrp: np.ndarray  # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
    ag_ghg_t_mrj: np.ndarray  # GHG emissions released during transitions between agricultural land uses.

    ag_t_mrj: np.ndarray  # Agricultural transition cost matrices.
    ag_c_mrj: np.ndarray  # Agricultural production cost matrices.
    ag_r_mrj: np.ndarray  # Agricultural production revenue matrices.
    ag_g_mrj: np.ndarray  # Agricultural greenhouse gas emissions matrices.
    ag_w_mrj: np.ndarray  # Agricultural water requirements matrices.
    ag_x_mrj: np.ndarray  # Agricultural exclude matrices.
    ag_q_mrp: np.ndarray  # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
    ag_ghg_t_mrj: np.ndarray  # GHG emissions released during transitions between agricultural land uses.

    ag_to_non_ag_t_rk: np.ndarray  # Agricultural to non-agricultural transition cost matrix.
    non_ag_to_ag_t_mrj: np.ndarray  # Non-agricultural to agricultural transition cost matrices.
    non_ag_c_rk: np.ndarray  # Non-agricultural production cost matrix.
    non_ag_r_rk: np.ndarray  # Non-agricultural revenue matrix.
    non_ag_g_rk: np.ndarray  # Non-agricultural greenhouse gas emissions matrix.
    non_ag_w_rk: np.ndarray  # Non-agricultural water requirements matrix.
    non_ag_x_rk: np.ndarray  # Non-agricultural exclude matrices.
    non_ag_q_crk: np.ndarray  # Non-agricultural yield matrix.

    ag_man_c_mrj: np.ndarray  # Agricultural management options' cost effects.
    ag_man_g_mrj: np.ndarray  # Agricultural management options' GHG emission effects.
    ag_man_q_mrp: np.ndarray  # Agricultural management options' quantity effects.
    ag_man_r_mrj: np.ndarray  # Agricultural management options' revenue effects.
    ag_man_t_mrj: np.ndarray  # Agricultural management options' transition cost effects.
    ag_man_w_mrj: np.ndarray  # Agricultural management options' water requirement effects.
    ag_man_limits: np.ndarray  # Agricultural management options' adoption limits.

    lu2pr_pj: np.ndarray  # Conversion matrix: land-use to product(s).
    pr2cm_cp: np.ndarray  # Conversion matrix: product(s) to commodity.
    limits: dict  # Targets to use.
    desc2aglu: dict  # Map of agricultural land use descriptions to codes.

    @property
    def n_ag_lms(self):
        # Number of agricultural landmans
        return self.ag_t_mrj.shape[0]

    @property
    def ncells(self):
        # Number of cells
        return self.ag_t_mrj.shape[1]

    @property
    def n_ag_lus(self):
        # Number of agricultural landuses
        return self.ag_t_mrj.shape[2]

    @property
    def n_non_ag_lus(self):
        # Number of non-agricultural landuses
        return self.non_ag_c_rk.shape[1]

    @property
    def nprs(self):
        # Number of products
        return self.ag_q_mrp.shape[2]

    @cached_property
    def am2j(self):
        # Map of agricultural management options to land use codes
        return {
            am: [self.desc2aglu[lu] for lu in am_lus]
            for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items()
        }

    @cached_property
    def j2am(self):
        _j2am = defaultdict(list)
        for am, am_j_list in self.am2j.items():
            for j in am_j_list:
                _j2am[j].append(am)
        return _j2am

    @cached_property
    def ag_lu2cells(self):
        # Make an index of each cell permitted to transform to each land use / land management combination
        return {
            (m, j): np.where(self.ag_x_mrj[m, :, j])[0]
            for j in range(self.n_ag_lus)
            for m in range(self.n_ag_lms)
        }

    @cached_property
    def non_ag_lu2cells(self):
        return {
            k: np.where(self.non_ag_x_rk[:, k])[0] for k in range(self.n_non_ag_lus)
        }


class LutoSolver:
    """
    Class responsible for grouping the Gurobi model, relevant input data, and its variables.
    """

    def __init__(self, input_data: InputData, d_c: np.array):
        self._input_data = input_data
        self.d_c = d_c
        self.gurobi_model = gp.Model("LUTO " + settings.VERSION, env=gurenv)

    def formulate(self):
        """
        Performs the initial formulation of the model - setting up variables, decision variables,
        etc.
        """
        print("\nSetting up the model...", time.ctime() + "\n")

        print("Adding decision variables...", time.ctime() + "\n")
        self._setup_x_vars()
        self._setup_ag_management_variables()
        self._setup_decision_variables()

        self._setup_objective()

        print("Setting up constraints...", time.ctime() + "\n")
        self._setup_constraints()

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

        for k in range(self._input_data.n_non_ag_lus):
            lu_cells = self._input_data.non_ag_lu2cells[k]
            for r in lu_cells:
                self.X_non_ag_vars_kr[k, r] = self.gurobi_model.addVar(
                    ub=1, name=f"X_non_ag_{k}_{r}"
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
            # Get snake_case version of the AM name for the variable name
            am_name = am.lower().replace(" ", "_")

            for j_idx, j in enumerate(am_j_list):
                # Create variable for all eligible cells
                dry_lu_cells = self._input_data.ag_lu2cells[0, j]
                for r in dry_lu_cells:
                    dry_var_name = f"X_ag_man_dry_{am_name}_{j}_{r}"
                    self.X_ag_man_dry_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                        ub=1, name=dry_var_name
                    )

                irr_lu_cells = self._input_data.ag_lu2cells[1, j]
                for r in irr_lu_cells:
                    irr_var_name = f"X_ag_man_irr_{am_name}_{j}_{r}"
                    self.X_ag_man_irr_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                        ub=1, name=irr_var_name
                    )

    def _setup_decision_variables(self):
        """
        Decision variables, one for each commodity, to minimise the deviations from demand.
        """
        self.V = self.gurobi_model.addMVar(self.ncms, name="V")

    def _setup_objective(self):
        """
        Formulate the objective based on settings.OBJECTIVE
        """
        print(
            "Setting up objective function to %s..." % settings.OBJECTIVE,
            time.ctime() + "\n",
        )

        if settings.OBJECTIVE == "maxrev":
            # Pre-calculate revenue minus (production and transition) costs
            ag_obj_mrj = (
                -(
                    self._input_data.ag_r_mrj
                    - (
                        self._input_data.ag_c_mrj
                        + self._input_data.ag_t_mrj
                        + self._input_data.non_ag_to_ag_t_mrj
                    )
                )
                / settings.PENALTY
            )

            non_ag_obj_rk = (
                -(
                    self._input_data.non_ag_r_rk
                    - (
                        self._input_data.non_ag_c_rk
                        + self._input_data.ag_to_non_ag_t_rk
                    )
                )
                / settings.PENALTY
            )

            # Get effects of alternative agr. management options (stored in a dict)
            ag_man_objs = {
                am: -(
                    self._input_data.ag_man_r_mrj[am]
                    - (
                        self._input_data.ag_man_c_mrj[am]
                        + self._input_data.ag_man_t_mrj[am]
                    )
                )
                / settings.PENALTY
                for am in self._input_data.am2j
            }

        elif settings.OBJECTIVE == "mincost":
            # Pre-calculate sum of production and transition costs
            ag_obj_mrj = (
                self._input_data.ag_c_mrj
                + self._input_data.ag_t_mrj
                + self._input_data.non_ag_to_ag_t_mrj
            ) / settings.PENALTY
            non_ag_obj_rk = (
                self._input_data.non_ag_c_rk + self._input_data.ag_to_non_ag_t_rk
            ) / settings.PENALTY

            # Store calculations for each agricultural management option in a dict
            ag_man_objs = {
                am: (
                    self._input_data.ag_man_c_mrj[am]
                    + self._input_data.ag_man_t_mrj[am]
                    # TODO - add t costs for non ag to ag mans
                )
                / settings.PENALTY
                for am in self._input_data.am2j
            }

        else:
            print("Unknown objective")

        ag_obj_mrj = np.nan_to_num(ag_obj_mrj)
        non_ag_obj_rk = np.nan_to_num(non_ag_obj_rk)

        # Specify objective function
        objective = (
            # Production costs + transition costs for all agricultural land uses.
            gp.quicksum(
                ag_obj_mrj[0, self._input_data.ag_lu2cells[0, j], j]
                @ self.X_ag_dry_vars_jr[j, self._input_data.ag_lu2cells[0, j]]
                + ag_obj_mrj[1, self._input_data.ag_lu2cells[1, j], j]
                @ self.X_ag_irr_vars_jr[j, self._input_data.ag_lu2cells[1, j]]
                for j in range(self._input_data.n_ag_lus)
            )
            # Effects on production costs + transition costs for alternative agr. management options
            + gp.quicksum(
                ag_man_objs[am][0, self._input_data.ag_lu2cells[0, j], j_idx]
                @ self.X_ag_man_dry_vars_jr[am][
                    j_idx, self._input_data.ag_lu2cells[0, j]
                ]
                + ag_man_objs[am][1, self._input_data.ag_lu2cells[1, j], j_idx]
                @ self.X_ag_man_irr_vars_jr[am][
                    j_idx, self._input_data.ag_lu2cells[1, j]
                ]
                for am, am_j_list in self._input_data.am2j.items()
                for j_idx, j in enumerate(am_j_list)
            )
            # Production costs + transition costs for all non-agricultural land uses.
            + gp.quicksum(
                non_ag_obj_rk[:, k][self._input_data.non_ag_lu2cells[k]]
                @ self.X_non_ag_vars_kr[k, self._input_data.non_ag_lu2cells[k]]
                for k in range(self._input_data.n_non_ag_lus)
            )
            # Add deviation-from-demand variables for ensuring demand of each commodity is met (approximately).
            + gp.quicksum(self.V[c] for c in range(self.ncms))
        )
        self.gurobi_model.setObjective(objective, GRB.MINIMIZE)

    def _add_cell_usage_constraints(self):
        """
        Constraint that all of every cell is used for some land use.
        """
        # Create an array indexed by cell that contains the sums of each cell's variables.
        # Then, loop through the array and add the constraint that each expression must equal 1.
        X_sum_r = (
            self.X_ag_dry_vars_jr.sum(axis=0)
            + self.X_ag_irr_vars_jr.sum(axis=0)
            + self.X_non_ag_vars_kr.sum(axis=0)
        )
        for expr in X_sum_r:
            self.gurobi_model.addConstr(expr == 1)

    def _add_agricultural_management_constraints(self):
        """
        Constraint handling alternative agricultural management options:
        Ag. man. variables cannot exceed the value of the agricultural variable.
        """
        for am, am_j_list in self._input_data.am2j.items():
            for j_idx, j in enumerate(am_j_list):
                for r in self._input_data.ag_lu2cells[0, j]:
                    self.gurobi_model.addConstr(
                        self.X_ag_man_dry_vars_jr[am][j_idx, r]
                        <= self.X_ag_dry_vars_jr[j, r]
                    )
                for r in self._input_data.ag_lu2cells[1, j]:
                    self.gurobi_model.addConstr(
                        self.X_ag_man_irr_vars_jr[am][j_idx, r]
                        <= self.X_ag_irr_vars_jr[j, r]
                    )

    def _add_agricultural_management_adoption_limit_constraints(self):
        """
        Add adoption limits constraints for agricultural management options.
        """
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
                self.gurobi_model.addConstr(
                    ag_man_vars_sum <= adoption_limit * all_vars_sum
                )

    def _add_demand_penalty_constraints(self):
        """
        Constraints to penalise under and over production compared to demand.
        """

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
            X_ag_man_dry_pr = []
            X_ag_man_irr_pr = []
            for p in range(self._input_data.nprs):
                for j_idx, j in enumerate(am_j_list):
                    if self._input_data.lu2pr_pj[p, j]:
                        X_ag_man_dry_pr.append(self.X_ag_man_dry_vars_jr[am][j_idx, :])
                        X_ag_man_irr_pr.append(self.X_ag_man_irr_vars_jr[am][j_idx, :])
                        break
                else:
                    X_ag_man_irr_pr.append(np.zeros(self._input_data.ncells))
                    X_ag_man_dry_pr.append(np.zeros(self._input_data.ncells))

            ag_man_q_dry_p = [
                gp.quicksum(
                    self._input_data.ag_man_q_mrp[am][0, :, p] * X_ag_man_dry_pr[p]
                )
                for p in range(self._input_data.nprs)
            ]
            ag_man_q_irr_p = [
                gp.quicksum(
                    self._input_data.ag_man_q_mrp[am][1, :, p] * X_ag_man_irr_pr[p]
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
        total_q_c = [
            ag_q_dry_c[c] + ag_q_irr_c[c] + non_ag_q_c[c] for c in range(self.ncms)
        ]

        self.gurobi_model.addConstrs(
            (self.d_c[c] - total_q_c[c]) <= self.V[c] for c in range(self.ncms)
        )
        self.gurobi_model.addConstrs(
            (total_q_c[c] - self.d_c[c]) <= self.V[c] for c in range(self.ncms)
        )
    
    def _add_water_usage_limit_constraints(self):
        if settings.WATER_USE_LIMITS != "on":
            return

        print(
            "Adding water constraints by",
            settings.WATER_REGION_DEF + "...",
            time.ctime(),
        )

        # Returns region-specific water use limits
        w_limits = self._input_data.limits["water"]

        # Ensure water use remains below limit for each region
        for region, wreq_reg_limit, ind in w_limits:
            wreq_region = (
                gp.quicksum(
                    self._input_data.ag_w_mrj[0, ind, j]
                    @ self.X_ag_dry_vars_jr[
                        j, ind
                    ]  # Dryland agriculture contribution
                    + self._input_data.ag_w_mrj[1, ind, j]
                    @ self.X_ag_irr_vars_jr[
                        j, ind
                    ]  # Irrigated agriculture contribution
                    for j in range(self._input_data.n_ag_lus)
                )
                + gp.quicksum(
                    self._input_data.ag_man_w_mrj[am][0, ind, j_idx]
                    @ self.X_ag_man_dry_vars_jr[am][
                        j_idx, ind
                    ]  # Dryland alt. ag. management contributions
                    + self._input_data.ag_man_w_mrj[am][1, ind, j_idx]
                    @ self.X_ag_man_irr_vars_jr[am][
                        j_idx, ind
                    ]  # Irrigated alt. ag. management contributions
                    for am, am_j_list in self._input_data.am2j.items()
                    for j_idx in range(len(am_j_list))
                )
                + gp.quicksum(
                    self._input_data.non_ag_w_rk[ind, k]
                    @ self.X_non_ag_vars_kr[k, ind]  # Non-agricultural contribution
                    for k in range(self._input_data.n_non_ag_lus)
                )
            )

            if wreq_region is not 0:
                self.gurobi_model.addConstr(wreq_region <= wreq_reg_limit)

            if settings.VERBOSE == 1:
                print(
                    "    ...setting water limit for %s <= %.2f ML"
                    % (region, wreq_reg_limit)
                )
    
    def _add_ghg_emissions_limit_constraints(self):
        if settings.GHG_EMISSIONS_LIMITS != "on":
            return

        print("\nAdding GHG emissions constraints...", time.ctime() + "\n")

        # Returns GHG emissions limits
        ghg_limits = self._input_data.limits["ghg"]

        # Pre-calculate the coefficients for each variable,
        # both for regular culture and alternative agr. management options
        g_dry_coeff = (
            self._input_data.ag_g_mrj[0, :, :]
            + self._input_data.ag_ghg_t_mrj[0, :, :]
        )
        g_irr_coeff = (
            self._input_data.ag_g_mrj[1, :, :]
            + self._input_data.ag_ghg_t_mrj[1, :, :]
        )

        ghg_emissions = (
            gp.quicksum(
                g_dry_coeff[:, j]
                @ self.X_ag_dry_vars_jr[j, :]  # Dryland agriculture contribution
                + g_irr_coeff[:, j]
                @ self.X_ag_irr_vars_jr[j, :]  # Irrigated agriculture contribution
                for j in range(self._input_data.n_ag_lus)
            )
            + gp.quicksum(
                self._input_data.ag_man_g_mrj[am][0, :, j_idx]
                @ self.X_ag_man_dry_vars_jr[am][
                    j_idx, :
                ]  # Dryland alt. ag. management contributions
                + self._input_data.ag_man_g_mrj[am][1, :, j_idx]
                @ self.X_ag_man_irr_vars_jr[am][
                    j_idx, :
                ]  # Irrigated alt. ag. management contributions
                for am, am_j_list in self._input_data.am2j.items()
                for j_idx in range(len(am_j_list))
            )
            + gp.quicksum(
                self._input_data.non_ag_g_rk[:, k]
                @ self.X_non_ag_vars_kr[k, :]  # Non-agricultural contribution
                for k in range(self._input_data.n_non_ag_lus)
            )
        )

        print(
            "    ...setting GHG emissions reduction target: {:,.0f} tCO2e\n".format(
                ghg_limits
            )
        )
        self.gurobi_model.addConstr(ghg_emissions <= ghg_limits)

    def _setup_constraints(self):
        self._add_cell_usage_constraints()
        self._add_agricultural_management_constraints()
        self._add_agricultural_management_adoption_limit_constraints()
        self._add_demand_penalty_constraints()
        self._add_water_usage_limit_constraints()
        self._add_ghg_emissions_limit_constraints()

    def update_formulation(
        self,
        input_data: InputData,
        d_c: np.array,
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

        updated_cells = self._update_variables(
            old_lumap, current_lumap, old_lmmap, current_lmmap
        )
        self._update_objective()
        self._update_constraints(updated_cells)

    def _update_variables(
        self,
        old_lumap: np.array,
        current_lumap: np.array,
        old_lmmap: np.array,
        current_lmmap: np.array,
    ) -> np.array:
        """
        Updates the variables only for cells that have changed land use or land management.
        Returns an array of cells that have been updated.
        """
        # update x vars
        print("Updating variables...", end=" ", flush=True)

        # metrics
        num_cells_skipped = 0
        updated_cells = []

        ag_lus_zeros = np.zeros(self._input_data.n_ag_lus)
        non_ag_lus_zeros = np.zeros(self._input_data.n_non_ag_lus)

        for r in range(self._input_data.ncells):
            old_j = old_lumap[r]
            new_j = current_lumap[r]
            old_m = old_lmmap[r]
            new_m = current_lmmap[r]
            if old_j == new_j and old_m == new_m:
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
            self.X_ag_dry_vars_jr[:, r] = ag_lus_zeros
            self.X_ag_irr_vars_jr[:, r] = ag_lus_zeros
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
            self.X_non_ag_vars_kr[:, r] = non_ag_lus_zeros
            for k in range(self._input_data.n_non_ag_lus):
                if self._input_data.non_ag_x_rk[r, k]:
                    self.X_non_ag_vars_kr[k, r] = self.gurobi_model.addVar(
                        ub=1, name=f"X_non_ag_{k}_{r}"
                    )

            # agricultural management
            for am, am_j_list in self._input_data.am2j.items():
                # Get snake_case version of the AM name for the variable name
                am_name = am.lower().replace(" ", "_")
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

                for j_idx, j in enumerate(am_j_list):
                    if self._input_data.ag_x_mrj[0, r, j]:
                        self.X_ag_man_dry_vars_jr[am][
                            j_idx, r
                        ] = self.gurobi_model.addVar(
                            ub=1, name=f"X_ag_man_dry_{am_name}_{j}_{r}"
                        )

                    if self._input_data.ag_x_mrj[1, r, j]:
                        self.X_ag_man_irr_vars_jr[am][
                            j_idx, r
                        ] = self.gurobi_model.addVar(
                            ub=1, name=f"X_ag_man_irr_{am_name}_{j}_{r}"
                        )

            updated_cells.append(r)

        updated_cells = np.array(updated_cells)
        print(
            f"Done. Skipped {num_cells_skipped} cells, updated {len(updated_cells)} cells."
        )
        return updated_cells

    def _update_objective(self):
        # For now, we can just setup the objective from scratch.
        # TODO: update objective dynamically
        self._setup_objective()

    def _update_constraints(self, updated_cells: np.array):
        # For now, we setup all constraints from scratch.
        # TODO: update constraints dynamically
        self.gurobi_model.remove(self.gurobi_model.getConstrs())
        self._setup_constraints()

    def solve(self):
        st = time.time()
        print("Starting solve... ", time.ctime(), "\n")

        # Magic.
        self.gurobi_model.optimize()

        ft = time.time()
        print("Completed solve... ", time.ctime())
        print(
            "Found optimal objective value",
            round(self.gurobi_model.objVal, 2),
            "in",
            round(ft - st),
            "seconds\n",
        )

        print("Collecting results...", end=" ", flush=True)

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
        for k in range(self._input_data.n_non_ag_lus):
            for r in self._input_data.non_ag_lu2cells[k]:
                non_ag_X_sol_rk[r, k] = self.X_non_ag_vars_kr[k, r].X

        # Get agricultural management results
        for am, am_j_list in self._input_data.am2j.items():
            for j_idx, j in enumerate(am_j_list):
                for r in self._input_data.ag_lu2cells[0, j]:
                    am_X_dry_sol_rj[am][r, j] = self.X_ag_man_dry_vars_jr[am][
                        j_idx, r
                    ].X
                for r in self._input_data.ag_lu2cells[1, j]:
                    am_X_irr_sol_rj[am][r, j] = self.X_ag_man_irr_vars_jr[am][
                        j_idx, r
                    ].X

        """Note that output decision variables are mostly 0 or 1 but in some cases they are somewhere in between which creates issues 
            when converting to maps etc. as individual cells can have non-zero values for multiple land-uses and land management type.
            This code creates a boolean X_mrj output matrix and ensure that each cell has one and only one land-use and land management"""

        # Process agricultural land usage information
        # Stack dryland and irrigated decision variables
        ag_X_mrj = np.stack((X_dry_sol_rj, X_irr_sol_rj))  # Float32
        ag_X_mrj_shape = ag_X_mrj.shape

        # Reshape so that cells are along the first axis and land management and use are flattened along second axis i.e. (XXXXXXX,  56)
        ag_X_mrj_processed = np.moveaxis(ag_X_mrj, 1, 0)
        ag_X_mrj_processed = ag_X_mrj_processed.reshape(ag_X_mrj_processed.shape[0], -1)

        # Boolean matrix where the maximum value for each cell across all land management types and land uses is True
        ag_X_mrj_processed = ag_X_mrj_processed.argmax(axis=1)[:, np.newaxis] == range(
            ag_X_mrj_processed.shape[1]
        )

        # Reshape to mrj structure
        ag_X_mrj_processed = ag_X_mrj_processed.reshape(
            (ag_X_mrj_shape[1], ag_X_mrj_shape[0], ag_X_mrj_shape[2])
        )
        ag_X_mrj_processed = np.moveaxis(ag_X_mrj_processed, 0, 1)

        # Process non-agricultural land usage information
        # Boolean matrix where the maximum value for each cell across all non-ag LUs is True
        non_ag_X_rk_processed = non_ag_X_sol_rk.argmax(axis=1)[:, np.newaxis] == range(
            self._input_data.n_non_ag_lus
        )

        # Make land use and land management maps
        # Vector indexed by cell that denotes whether the cell is non-agricultural land (True) or agricultural land (False)
        non_ag_bools_r = non_ag_X_sol_rk.max(axis=1) > ag_X_mrj.max(axis=(0, 2))

        # Process agricultural management variables
        # Repeat the steps for the regular agricultural management variables
        ag_man_X_mrj_processed = {}
        for am in self._input_data.am2j:
            ag_man_processed = np.stack((am_X_dry_sol_rj[am], am_X_irr_sol_rj[am]))
            ag_man_X_shape = ag_man_processed.shape

            ag_man_processed = np.moveaxis(ag_man_processed, 1, 0)
            ag_man_processed = ag_man_processed.reshape(ag_man_processed.shape[0], -1)

            ag_man_processed = ag_man_processed.argmax(axis=1)[:, np.newaxis] == range(
                ag_man_processed.shape[1]
            )
            ag_man_processed = ag_man_processed.reshape(
                (ag_man_X_shape[1], ag_man_X_shape[0], ag_man_X_shape[2])
            )
            ag_man_processed = np.moveaxis(ag_man_processed, 0, 1)
            ag_man_X_mrj_processed[am] = ag_man_processed

        # Calculate 1D array (maps) of land-use and land management, considering only agricultural LUs
        lumap = ag_X_mrj_processed.sum(axis=0).argmax(axis=1).astype("int8")
        lmmap = ag_X_mrj_processed.sum(axis=2).argmax(axis=0).astype("int8")

        # Update lxmaps and processed variable matrices to consider non-agricultural LUs
        lumap[non_ag_bools_r] = (
            non_ag_X_sol_rk[non_ag_bools_r, :].argmax(axis=1)
            + settings.NON_AGRICULTURAL_LU_BASE_CODE
        )
        lmmap[
            non_ag_bools_r
        ] = 0  # Assume that all non-agricultural land uses are dryland

        # Process agricultural management usage info
        # Get number of agr. man. options (add one for the option of no usage)
        n_am_options = range(len(self._input_data.am2j.keys()) + 1)

        # Make ammap (agricultural management map) using the lumap and lmmap
        ammap = np.zeros(self._input_data.ncells, dtype=np.int8)
        for r in range(self._input_data.ncells):
            cell_j = lumap[r]
            cell_m = lmmap[r]

            if cell_j >= settings.NON_AGRICULTURAL_LU_BASE_CODE:
                # Non agricultural land use - no agricultural management option
                cell_am = 0

            else:
                if cell_m == 0:
                    am_values = [
                        am_X_dry_sol_rj[am][r, cell_j] for am in SORTED_AG_MANAGEMENTS
                    ]
                else:
                    am_values = [
                        am_X_irr_sol_rj[am][r, cell_j] for am in SORTED_AG_MANAGEMENTS
                    ]

                # Get argmax and max of the am_values list
                argmax, max_am_var_val = max(enumerate(am_values), key=lambda x: x[1])

                if max_am_var_val < 0.5:
                    # The cell doesn't use any alternative agricultural management options
                    cell_am = 0
                else:
                    # Add one to the argmax to account for the default option of no ag management being 0
                    cell_am = argmax + 1

            ammap[r] = cell_am

        ag_X_mrj_processed[:, non_ag_bools_r, :] = False
        non_ag_X_rk_processed[~non_ag_bools_r, :] = False

        print("Done\n")

        return (
            lumap,
            lmmap,
            ammap,
            ag_X_mrj_processed,
            non_ag_X_rk_processed,
            ag_man_X_mrj_processed,
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
