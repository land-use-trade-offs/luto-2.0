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
from scipy import sparse

from luto import tools
from luto.solvers.bio_family import extract_groups, compose_row, extract_structure
from luto.solvers.input_data import SolverInputData
from luto.solvers.tools import check_constraint_names, reduce_forced_zero_rows
from luto.settings import (
    AG_MANAGEMENTS, 
    AG_MANAGEMENTS_REVERSIBLE, 
    NON_AG_LAND_USES, 
    NON_AG_LAND_USES_REVERSIBLE
)


# Set Gurobi environment.
gurenv = gp.Env(logfilename="gurobi.log", empty=True)  # (empty = True)
gurenv.setParam("OutputFlag", settings.VERBOSE)
gurenv.setParam("OptimalityTol", settings.OPTIMALITY_TOLERANCE)
gurenv.setParam("FeasibilityTol", settings.FEASIBILITY_TOLERANCE)
gurenv.setParam("BarConvTol", settings.BARRIER_CONVERGENCE_TOLERANCE)
gurenv.setParam("ScaleFlag", settings.SCALE_FLAG)
gurenv.setParam("Threads", settings.THREADS)
gurenv.start()


@dataclass
class SolverSolution:
    lumap: np.ndarray
    lmmap: np.ndarray
    ammaps: dict[str, np.ndarray]
    ag_X_mrj: np.ndarray
    non_ag_X_rk: np.ndarray
    ag_man_X_mrj: dict[str, np.ndarray]
    dvar_D_ag2ag_mrj: dict                                                # Solved ag->ag deltas, SOURCE-KEYED: {(from_m, from_j): ndarray(NLMS, ncells_src, N_AG_LUS) [to_m, local_r, to_j]} over the source's cells (get_base_dvar_mj_cell_map)
    dvar_D_ag2nonag_rk: dict                                              # Solved ag->nonag deltas, SOURCE-KEYED: {(from_m, from_j): ndarray(ncells_src, N_NON_AG_LUS) [local_r, k]}
    dvar_D_nonag2ag_mrj: dict                                             # Solved nonag->ag deltas, SOURCE-KEYED: {from_k: ndarray(NLMS, ncells_k, N_AG_LUS) [to_m, local_r, to_j]} (e.g. reversible Destocked back to ag; cells via get_base_nonag_dvar_k_cell_map)
    prod_data: dict[str, Any]
    obj_val: dict[str, float]


def _qsum(coeffs: np.ndarray, gurobi_vars: np.ndarray) -> "gp.LinExpr":
    """
    Return ``gp.quicksum(coeffs * gurobi_vars)`` filtered to ``|coeff| >= SOLVER_COEFF_MIN``.

    ``coeffs`` and ``gurobi_vars`` must be aligned (same length, same ordering).
    The caller must pre-slice both arrays with the same index before calling, so
    this function only needs a plain boolean mask — never a sub-index of a
    potentially-boolean index array (which would produce a dimension mismatch).
    """
    mask = np.abs(coeffs) >= settings.SOLVER_COEFF_MIN
    if not mask.any():
        return gp.LinExpr(0)
    return gp.quicksum(coeffs[mask] * gurobi_vars[mask])


def _floor_assembled_matrix(model) -> None:
    """Drop ``|coeff| < coeff_min`` from an ASSEMBLED model's constraint matrix AND objective vector.

    ``_qsum`` floors coefficients as each term is built, but some are created DOWNSTREAM: the water/
    GHG/etc. accounting term ``coeff × X_acct`` where a folded-sliver ``X_acct`` is a LinExpr with
    ``~1/RESFACTOR²`` weights distributes a floored-and-kept ``coeff`` into a sub-floor product on the
    dominant var (see ``docs/FINDINGS.md`` 20260721). This single post-build sweep catches those (and
    the same leak in the objective, which ``getA()`` — LHS-only — can't reach). Physically safe: a
    ``~1e-5`` ML/cell water term against a ``~1e7`` ML regional limit is negligible — the same
    negligibility contract as ``SOLVER_COEFF_MIN``, enforced on the assembled model.
    """
    model.update()
    A = model.getA().tocoo()
    cons = model.getConstrs()
    varz = model.getVars()
    mask = (np.abs(A.data) < settings.SOLVER_COEFF_MIN) & (A.data != 0.0)
    n = int(mask.sum())
    for r, c in zip(A.row[mask], A.col[mask]):
        model.chgCoeff(cons[int(r)], varz[int(c)], 0.0)
    # objective vector: same re-expression leak reaches obj coeffs via _qsum(ag_obj × X_acct)
    n_obj = 0
    for v in varz:
        oc = v.Obj
        if 0.0 < abs(oc) < settings.SOLVER_COEFF_MIN:
            v.Obj = 0.0
            n_obj += 1
    model.update()
    print(f"└── Flooring coefficients below {settings.SOLVER_COEFF_MIN:g} (post-build): "
          f"dropped {n:,} from the matrix, {n_obj:,} from the objective", flush=True)


class LutoSolver:
    """The Gurobi model, its input data, and all builder-side bookkeeping.
    """

    def __init__(
        self,
        input_data: SolverInputData,
    ):

        self._input_data = input_data
        self.gurobi_model = gp.Model(f"LUTO {settings.VERSION}", env=gurenv)

        # --- decision-variable registries (dense cell-indexed; entries are Gurobi Vars) ---
        self.X_ag_dry_vars_jr = None
        self.X_ag_irr_vars_jr = None
        self.X_non_ag_vars_kr = None
        self.X_ag_man_dry_vars_jr = None
        self.X_ag_man_irr_vars_jr = None
        self.F_ag2ag = {}       # (from_m, from_j) -> tupledict{(to_m, local_r, to_j): Var}
        self.F_ag2nonag = {}    # (from_m, from_j) -> tupledict{(k, local_r): Var}
        self.F_nonag2ag = {}    # from_k           -> tupledict{(to_m, local_r, to_j): Var}

        # --- constraint handles ---
        self.cell_usage_constraint_r = {}
        self.ag_management_constraints_r = defaultdict(list)
        self.adoption_limit_constraints = []
        self.regional_adoption_constrs = []
        self.demand_penalty_constraints = []    # hard demand rows (name kept for the tools contract)
        self.water_limit_constraints = []
        self.renewable_constraints = {}
        self.ghg_consts_ub = None               # single Constr (hard GHG)
        self.bio_GBF2_constrs = {}              # single Constr once built ({} = not built)
        self.bio_GBF3_NVIS_constrs = {}         # {(region, group): Constr}
        self.bio_GBF4_SNES_constrs = {}         # {(region, species, presence): Constr}
        self.bio_GBF4_ECNES_constrs = {}        # {(region, community, presence): Constr}
        self.bio_GBF8_constrs = {}              # {(region, species): Constr}
        self._bio_index = None                  # bio_constraint_index() cache

        # --- array-path shared caches (each built ONCE per formulate, reused by all families) ---
        self._bio_groups = None                 # bio term groups (extract_groups)
        self._policy_structure = None           # family-independent structure walk (extract_structure)
        self._bio_var_list = None               # model.getVars() cache (one materialization)

        # --- array-path constraint blocks (CSR over Var.index columns, rescaled units) ---
        # + per-row key lists; post-solve reporting = one mat-vec over each block
        self.bio_GBF2_block = None              # 1 row, no key list (single constraint)
        self.bio_GBF3_NVIS_block = None
        self.bio_GBF3_NVIS_block_pairs = []     # (region, group) per row
        self.bio_GBF4_SNES_block = None
        self.bio_GBF4_SNES_block_pairs = []     # (region, species, presence) per row
        self.bio_GBF4_ECNES_block = None
        self.bio_GBF4_ECNES_block_pairs = []    # (region, community, presence) per row
        self.bio_GBF8_block = None
        self.bio_GBF8_block_pairs = []          # (region, species) per row
        self.water_block = None
        self.water_block_regids = []            # region id per water_block row, same order
        self.ghg_block = None                   # 1 row; the offland_ghg constant lives in the RHS
        self.demand_q_block = None              # one row per commodity (the LHS, not the constraints)



    def formulate(self):
        """
        Performs the initial formulation of the model - setting up decision variables,
        constraints, and the objective.
        """
        print("Setting up the model...")
        self._setup_vars()
        self._setup_constraints()
        self._setup_objective()
        _floor_assembled_matrix(self.gurobi_model)
        if settings.REDUCE_FORCED_ZERO_ROWS:
            # Exact: rows forcing their own variables to zero carry no information, but they are
            # 30% of the matrix and the most degenerate part of it. See the setting's docstring.
            n = reduce_forced_zero_rows(self.gurobi_model)
            print(f"└── Reduced {n:,} forced-zero row(s) (exact: their variables were already "
                  f"pinned at 0)")
        self.bio_constraint_index()   # warm the cache while every constraint is still live
        # ONE name containing whitespace makes Gurobi discard EVERY name when the model is written
        # to MPS — which silently blinds all post-mortem attribution. Assert it here, not there.
        check_constraint_names(self.gurobi_model)

    def _setup_vars(self):
        print("├── Setting up decision variables...")
        self._setup_ag_folded_vars()         
        self._setup_ag_accounting_vars()     
        self._setup_non_ag_vars()
        self._setup_ag_management_variables()
        self._setup_flow_vars()  

    def _setup_constraints(self):
        print("├── Adding the constraints...")
        self._add_cell_usage_constraints()
        self._add_agricultural_management_constraints()
        self._add_agricultural_management_adoption_limit_constraints()
        self._add_demand_constraints()
        self._add_ghg_emissions_limit_constraints()
        self._add_biodiversity_constraints()
        self._add_regional_adoption_constraints()
        self._add_water_usage_limit_constraints()
        self._add_renewable_energy_constraints()
        self._add_flow_out_constraints()                    # source cap (Σ out ≤ x_old; bounds deltas)
        self._add_flow_in_constraints()                     # node balance (X = base + Σin − Σout)

    def _setup_objective(self):
        """
        Formulate the objective based on settings.OBJECTIVE
        """
        print(f"├── Setting up the objective function to {settings.OBJECTIVE}...")

        # Get objectives
        self.obj_economy = self._setup_economy_objective()

        # Set the objective function
        if settings.OBJECTIVE == "mincost":
            self.gurobi_model.setObjective(self.obj_economy, GRB.MINIMIZE)
        elif settings.OBJECTIVE == "maxprofit":
            self.gurobi_model.setObjective(self.obj_economy, GRB.MAXIMIZE)
        else:
            raise ValueError(f"    Unknown objective function: {settings.OBJECTIVE}")
           

    def _setup_ag_folded_vars(self):
        print("│   ├── setting up decision variables for agricultural land uses...")
        self.X_ag_dry_vars_jr = np.zeros(
            (self._input_data.n_ag_lus, self._input_data.ncells), dtype=object
        )
        self.X_ag_irr_vars_jr = np.zeros(
            (self._input_data.n_ag_lus, self._input_data.ncells), dtype=object
        )

        # Target-var bounds from the TO view. dvar_lb_ag/dvar_ub_ag are already cleaned in input_data
        # (0 ≤ lb ≤ base ≤ ub, with reporting), so use them directly; the node-balance/cap constant is
        # just the (cleaned, in-box) base dvar — the all-delta=0 stay point is feasible by construction.
        dvar_lb_ag = self._input_data.dvar_lb_ag
        dvar_ub_ag = self._input_data.dvar_ub_ag
        for j in range(self._input_data.n_ag_lus):
            for r in self._input_data.feasible_ag_cells_mrj[0, j]:
                self.X_ag_dry_vars_jr[j, r] = self.gurobi_model.addVar(
                    lb=dvar_lb_ag[0, r, j], ub=dvar_ub_ag[0, r, j],
                    name=f"X_ag_dry_{j}_{r}".replace(" ", "_")
                )
            for r in self._input_data.feasible_ag_cells_mrj[1, j]:
                self.X_ag_irr_vars_jr[j, r] = self.gurobi_model.addVar(
                    lb=dvar_lb_ag[1, r, j], ub=dvar_ub_ag[1, r, j],
                    name=f"X_ag_irr_{j}_{r}".replace(" ", "_")
                )
        self.const_ag = self._input_data.dvar_base_ag_mrj


    def _setup_ag_accounting_vars(self):
        """Build the ACCOUNTING stream dvar_account — a linear re-expression of the folded decision vars dvar_flow.

        MENTAL MODEL — a cell is a FIXED-COMPOSITION BUNDLE scaled by ONE scalar (the dominant's var).
        Example: a cell is 0.7 Beef + 0.3 Apple. Folding merges the minor Apple fraction into the dominant
        Beef, so a SINGLE variable X_Beef (dominant_frac = 0.7 + 0.3 = 1.0) represents "how much of this cell
        remains in its original composition". Each land use is then a CONSTANT RATIO of that one variable:
            dvar_account[Beef]  = (0.7 / 1.0) · X_Beef
            dvar_account[Apple] = (0.3 / 1.0) · X_Beef
        Reduce X_Beef by 1/3 (transition 1/3 of the cell away) and BOTH shrink by 1/3 — Beef 0.7 → 0.467,
        Apple 0.3 → 0.2 — the composition ratio 7:3 is preserved, only the scale changes.

        dvar_flow carries the FOLDED composition: every sub-θ sliver's land was merged into its cell's dominant.
        Accounting (profit/water/GHG/GBF/production) must instead score each TRUE land-use's fraction. For a
        folded group (dominant receiver d with post-fold mass dominant_frac, true base
        base_d = dominant_frac − Σ slivers, and each sliver s carrying its folded fraction `slivers[s]`):

            dvar_account[d] = (base_d / dominant_frac) · dvar_flow[d]                  dominant → its TRUE share
            dvar_account[s] = dvar_flow[s] + (slivers / dominant_frac) · dvar_flow[d]  sliver inflow-land + folded share
            dvar_account[·] = dvar_flow[·]                                             any LU not in a fold: unchanged

        This adds NO Gurobi variables and NO constraints — the same terms the retired blended coefficient
        produced, written per true LU. Σ_LU coeff·dvar_account == coeff_eff[d]·dvar_flow[d] + Σ_s coeff_s·dvar_flow[s]
        exactly (stay-exact, scales with the live dominant, → 0 on a full flip). Entries stay Gurobi Var where
        untouched and become LinExpr where adjusted; `_qsum` handles both. The dominant's ORIGINAL dvar_flow is
        read when spreading sliver shares, so scale the dominant last / read from dvar_flow (never dvar_account).
        """
        self.X_acct_dry_jr = self.X_ag_dry_vars_jr.copy()
        self.X_acct_irr_jr = self.X_ag_irr_vars_jr.copy()

        fold_map = self._input_data.ag_fold_map
        if not fold_map['cells'].size:
            return

        dvar_flow    = (self.X_ag_dry_vars_jr, self.X_ag_irr_vars_jr)   # folded decision vars (read-only source)
        dvar_account = (self.X_acct_dry_jr,    self.X_acct_irr_jr)      # accounting stream (written)

        cells          = fold_map['cells']
        from_m, from_j = fold_map['from_m'], fold_map['from_j']
        to_m, to_j     = fold_map['to_m'], fold_map['to_j']
        slivers        = fold_map['vals'].astype(np.float64)
        dominant_frac  = fold_map['folded_dom'].astype(np.float64)   # > 0 by construction (holds ≥ Σ slivers)

        # A folded dominant's var `dom` collapses the cell's original composition [raw dominant, *slivers]
        # into ONE variable (dominant_frac = raw + Σ slivers). Un-fold it by TRANSFERRING each sliver's share
        # of the LIVE var from the dominant to the sliver — the dominant keeps whatever the slivers don't take,
        # so the group total is conserved and no separate dominant-scaling pass is needed. dvar_account[d]
        # starts as `dom` (a copy of the flow var), so each subtraction whittles it down to (raw/dominant_frac)·dom.
        # NOTE the transferred share is the LIVE (slivers/dominant_frac)·dom — subtracting the constant slivers
        # would freeze the dominant at its base and go negative once it sheds (the rejected v1 error).
        for k, r in enumerate(cells):
            dom = dvar_flow[to_m[k]][to_j[k], r]                # the receiver dominant's live folded var
            if not isinstance(dom, (gp.Var, gp.LinExpr)):
                # No folded var for the dominant ⇒ the dominant LU is banned (EXCLUDE_NO_GO_LU) in this
                # region, so the folded stream force-converts that land — its whole folded group has no
                # STANDING ag to account. Skip; do NOT mint a fresh var (X_acct must stay a re-expression
                # of the folded decision vars, never a new free variable). The sliver's own var, if any,
                # is still scored via acct_cells; only the folded-in mass (force-converted) gets nothing.
                continue
            share = (slivers[k] / dominant_frac[k]) * dom       # this sliver's fraction of the folded dominant (live)
            dvar_account[from_m[k]][from_j[k], r] = dvar_account[from_m[k]][from_j[k], r] + share   # sliver gains it
            dvar_account[to_m[k]][to_j[k], r]     = dvar_account[to_m[k]][to_j[k], r]     - share   # dominant loses it


    def _setup_non_ag_vars(self):
        print("│   ├── setting up decision variables for non-agricultural land uses...")
        self.X_non_ag_vars_kr = np.zeros(
            (self._input_data.n_non_ag_lus, self._input_data.ncells), dtype=object
        )
        
        lb_n = self._input_data.dvar_lb_nonag
        ub_n = self._input_data.dvar_ub_nonag
        self.const_nonag = self._input_data.dvar_base_non_ag_rk
        
        # If the lower and upper bounds are very close (within 1% of the lower bound), collapse to a single value
        collapse = (lb_n > 0) & (np.abs(ub_n - lb_n) / np.where(lb_n > 0, lb_n, 1.0) < 0.01)
        lb_eff = np.where(collapse, self.const_nonag, lb_n)
        ub_eff = np.where(collapse, self.const_nonag, ub_n)

        for k, k_name in enumerate(NON_AG_LAND_USES):
            if not NON_AG_LAND_USES[k_name]:
                continue
            for r in self._input_data.feasible_non_ag_cells[k]:
                self.X_non_ag_vars_kr[k, r] = self.gurobi_model.addVar(
                    lb=lb_eff[r, k],
                    ub=ub_eff[r, k],
                    name=f"X_non_ag_{k}_{r}".replace(" ", "_")
                )

    def _setup_ag_management_variables(self):
        print("│   ├── setting up decision variables for agricultural management options...")
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

            # Renewable energy AMs: exist_r and GBF2 exclusion are AM-level (not j-level).
            # Cell-level ceiling constraint added after all LU variables are built.
            if am in settings.RENEWABLES_OPTIONS:
                exist_r = (
                    self._input_data.exist_renewable_solar_r
                    if am == "Utility Solar PV"
                    else self._input_data.exist_renewable_wind_r
                )
                gbf2_excl_idx = (
                    self._input_data.renewable_GBF2_mask_solar_idx
                    if am == "Utility Solar PV"
                    else self._input_data.renewable_GBF2_mask_wind_idx
                )
                renewable_cells = set()
                for j_idx, j in enumerate(am_j_list):
                    dry_lu_cells = self._input_data.feasible_ag_cells_mrj[0, j]
                    irr_lu_cells = self._input_data.feasible_ag_cells_mrj[1, j]
                    # Hard-exclude GBF2 priority cells (no variable created → effective ub = 0)
                    if gbf2_excl_idx.size:
                        dry_lu_cells = np.setdiff1d(dry_lu_cells, gbf2_excl_idx)
                        irr_lu_cells = np.setdiff1d(irr_lu_cells, gbf2_excl_idx)
                    for r in dry_lu_cells:
                        model_lb = 0 if AG_MANAGEMENTS_REVERSIBLE[am] else self._input_data.ag_man_lb_mrj[am][0, r, j]
                        self.X_ag_man_dry_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                            lb=model_lb, ub=1,
                            name=f"X_ag_man_dry_{am_name}_{j}_{r}".replace(" ", "_"),
                        )
                    for r in irr_lu_cells:
                        model_lb = 0 if AG_MANAGEMENTS_REVERSIBLE[am] else self._input_data.ag_man_lb_mrj[am][1, r, j]
                        self.X_ag_man_irr_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                            lb=model_lb, ub=1,
                            name=f"X_ag_man_irr_{am_name}_{j}_{r}".replace(" ", "_"),
                        )
                    renewable_cells.update(dry_lu_cells)
                    renewable_cells.update(irr_lu_cells)

                # Simulated and existing capacity compete for cell space [0, ag_mask].
                # exist_r is the total across ALL data years (fixed), so the ceiling never
                # decreases between periods — lb(t) <= ceiling(t-1) = ceiling(t) always holds.
                ag_mask = self._input_data.ag_mask_proportion_r
                for r in sorted(renewable_cells):
                    cap = exist_r[r]
                    if not cap:
                        continue
                    terms = [
                        v for j_idx in range(len(am_j_list))
                        for v in (
                            self.X_ag_man_dry_vars_jr[am][j_idx, r],
                            self.X_ag_man_irr_vars_jr[am][j_idx, r],
                        )
                        if isinstance(v, gp.Var) # only set ub if the cell is a valide Renewable location
                    ]
                    if terms:
                        ceiling = max(ag_mask[r] - cap, 0.0)
                        self.gurobi_model.addConstr(
                            gp.quicksum(terms) <= ceiling,
                            name=f"const_{am_name}_solvable_ub_{r}".replace(" ", "_")
                        )
                continue  # skip generic j loop below

            # Generic loop: all other AM options use transition-based lower bounds.
            for j_idx, j in enumerate(am_j_list):
                dry_lu_cells = self._input_data.feasible_ag_cells_mrj[0, j]
                irr_lu_cells = self._input_data.feasible_ag_cells_mrj[1, j]

                # For savanna burning, remove extra ineligible cells
                if am_name == "savanna_burning":
                    dry_lu_cells = np.intersect1d(
                        dry_lu_cells, self._input_data.savanna_eligible_r
                    )

                for r in dry_lu_cells:
                    dry_x_lb = (
                        0
                        if AG_MANAGEMENTS_REVERSIBLE[am]
                        else self._input_data.ag_man_lb_mrj[am][0, r, j]
                    )
                    self.X_ag_man_dry_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                        lb=dry_x_lb, ub=1,
                        name=f"X_ag_man_dry_{am_name}_{j}_{r}".replace(" ", "_"),
                    )

                for r in irr_lu_cells:
                    irr_x_lb = (
                        0
                        if AG_MANAGEMENTS_REVERSIBLE[am]
                        else self._input_data.ag_man_lb_mrj[am][1, r, j]
                    )
                    self.X_ag_man_irr_vars_jr[am][j_idx, r] = self.gurobi_model.addVar(
                        lb=irr_x_lb, ub=1,
                        name=f"X_ag_man_irr_{am_name}_{j}_{r}".replace(" ", "_"),
                    )

    def _setup_flow_vars(self):

        print("│   └── setting up transition flow delta variables (D)...")
        model = self.gurobi_model

        # Feasibility is fully resolved in input_data (feasible_ag2ag_mrj / feasible_nonag2ag_mrj /
        # feasible_ag2nonag_rk — source-keyed dicts, keyed/shaped like the flow_cost dicts): each leaf
        # already combines target eligibility ∧ the source's T_MAT row ∧ the diagonal drop. Here we
        # just materialise one delta var per True entry.

        # ── ag → ag :  D[(fm,fj)][to_m, local_r, to_j], OFF-DIAGONAL only (positive-increment delta) ──
        # No stay/diagonal var: "staying" as (fm,fj) is free — the node-balance constant carries the base.
        for (fm, fj), valid in self._input_data.feasible_ag2ag_mrj.items():
            idx = list(map(tuple, np.argwhere(valid).tolist()))
            self.F_ag2ag[(fm, fj)] = model.addVars(idx, lb=0.0, name=f"F_a2a_{fm}_{fj}")

        # ── ag → non-ag :  F[(fm,fj)][k, local_r] ──
        for (fm, fj), valid in self._input_data.feasible_ag2nonag_rk.items():
            idx = [(int(k), int(lr)) for lr, k in np.argwhere(valid)]
            self.F_ag2nonag[(fm, fj)] = model.addVars(idx, lb=0.0, name=f"F_a2n_{fm}_{fj}")

        # ── non-ag → ag :  F[k][to_m, local_r, to_j] ──
        for fk, valid in self._input_data.feasible_nonag2ag_mrj.items():
            idx = list(map(tuple, np.argwhere(valid).tolist()))
            self.F_nonag2ag[fk] = model.addVars(idx, lb=0.0, name=f"F_n2a_{fk}")

        n_a2a = sum(len(v) for v in self.F_ag2ag.values())
        n_a2n = sum(len(v) for v in self.F_ag2nonag.values())
        n_n2a = sum(len(v) for v in self.F_nonag2ag.values())
        print(f"│       ├── ag2ag    : {n_a2a:,} delta vars")
        print(f"│       ├── ag2nonag : {n_a2n:,} delta vars")
        print(f"│       ├── nonag2ag : {n_n2a:,} delta vars")
        print(f"│       └── total    : {n_a2a + n_a2n + n_n2a:,} delta vars")


    def _add_flow_out_constraints(self):
        """Source cap: a source cannot export more land than it holds.

            Σ_out D[src]  ≤  x_old[src]

        This BOUNDS the delta vars (with negative `flow_cost` entries — water/GHG deltas can be < 0 —
        the objective would otherwise push a `D` to +∞ around a negative-cost cycle → unbounded). It also
        rules out "pass-through" (a source re-exporting land it imported). Combined with the node-balance
        equality (which ties each `D` to real per-LU movement) this gives an EXACT, bounded
        min-cost transition flow. RHS = `const` (base clipped into the effective [lb,ub] box) — the same
        quantity node-balance uses, so a source may export at most the land it actually holds.

        ag source (fm,fj) at cell r:  Σ_to D_ag2ag[(fm,fj)][·,r,·] + Σ_k D_ag2nonag[(fm,fj)][k,r] ≤ const_ag[fm,r,fj]
        non-ag source k at cell r:    Σ_to D_nonag2ag[k][·,r,·]                                    ≤ const_nonag[r,k]
        """
        print("│   ├── Adding source-cap (Σ out ≤ base) constraints...")
        model     = self.gurobi_model
        const_ag  = self.const_ag
        const_non = self.const_nonag

        n = 0
        for (fm, fj), cells in self._input_data.ag_source_cells.items():
            F_a2a = self.F_ag2ag[(fm, fj)]
            F_a2n = self.F_ag2nonag[(fm, fj)]
            for local_r, r in enumerate(cells):
                out = F_a2a.sum('*', local_r, '*') + F_a2n.sum('*', local_r)
                if out.size() == 0:
                    continue
                model.addConstr(out <= const_ag[fm, r, fj], name=f"srccap_a_{fm}_{fj}_{local_r}")
                n += 1

        for fk, cells in self._input_data.nonag_source_cells.items():
            F_n2a = self.F_nonag2ag[fk]
            for local_r, r in enumerate(cells):
                out = F_n2a.sum('*', local_r, '*')
                if out.size() == 0:
                    continue
                model.addConstr(out <= const_non[r, fk], name=f"srccap_n_{fk}_{local_r}")
                n += 1
        print(f"│   │   └── added {n:,} source-cap constraints")


    def _add_flow_in_constraints(self):
        """Node-balance equality: each LU's final area = base + inflows − outflows.

            X_ag[m,r,j]  = const_ag[m,r,j]  + Σ_in D[·→(m,j)] − Σ_out D[(m,j)→·]
            X_nonag[r,k] = const_nonag[r,k] + Σ_in D_ag2nonag[·→k] − Σ_out D_nonag2ag[k→·]

        This ties every delta to REAL per-LU land movement (so a single negative-cost arc can't be
        harvested without moving land — the flaw that made an import/export-only relaxation unbounded),
        and together with the source cap gives an exact, bounded min-cost transition flow. "Staying" is
        the all-D=0 solution (X = const). `const` = the base clipped into the var's effective [lb,ub]
        box (`_setup_*_vars`) ⇒ the stay point is feasible by construction — this replaces the earlier
        raw/floor(x_old) which fell OUTSIDE the box on float-noise cells (base>ub, base<0, floor<lb) and
        made presolve infeasible. No non-ag→non-ag term exists.
        """
        print("│   └── Adding node-balance (X = base + Σin − Σout) constraints...")
        model     = self.gurobi_model
        const_ag  = self.const_ag
        const_non = self.const_nonag

        # Reverse indices (global cell keys): inflows arrive at a target, outflows leave a source LU.
        in_ag     = defaultdict(list)   # (m, r, j) -> [vars] arriving at ag LU (m,j)   (ag2ag + nonag2ag)
        out_ag    = defaultdict(list)   # (m, r, j) -> [vars] leaving  ag LU (m,j)      (ag2ag + ag2nonag)
        in_nonag  = defaultdict(list)   # (r, k)    -> [vars] arriving at non-ag k       (ag2nonag)
        out_nonag = defaultdict(list)   # (r, k)    -> [vars] leaving  non-ag k          (nonag2ag)

        for (fm, fj), cells in self._input_data.ag_source_cells.items():
            for (to_m, local_r, to_j), var in self.F_ag2ag[(fm, fj)].items():
                g = cells[local_r]
                in_ag[(to_m, g, to_j)].append(var)   # arrives at (to_m,to_j)
                out_ag[(fm, g, fj)].append(var)      # leaves source (fm,fj)
            for (k, local_r), var in self.F_ag2nonag[(fm, fj)].items():
                g = cells[local_r]
                in_nonag[(g, k)].append(var)         # arrives at non-ag k
                out_ag[(fm, g, fj)].append(var)      # leaves ag source (fm,fj)
        for fk, cells in self._input_data.nonag_source_cells.items():
            for (to_m, local_r, to_j), var in self.F_nonag2ag[fk].items():
                g = cells[local_r]
                in_ag[(to_m, g, to_j)].append(var)   # arrives at ag (to_m,to_j)
                out_nonag[(g, fk)].append(var)       # leaves non-ag source k

        n = 0
        for j in range(self._input_data.n_ag_lus):
            for m, X_row in ((0, self.X_ag_dry_vars_jr), (1, self.X_ag_irr_vars_jr)):
                for r in self._input_data.feasible_ag_cells_mrj[m, j]:
                    model.addConstr(
                        X_row[j, r] == const_ag[m, r, j]
                        + gp.quicksum(in_ag.get((m, r, j), [])) - gp.quicksum(out_ag.get((m, r, j), [])),
                        name=f"bal_a_{m}_{j}_{r}")
                    n += 1
        for k in range(self._input_data.n_non_ag_lus):
            for r in self._input_data.feasible_non_ag_cells[k]:
                model.addConstr(
                    self.X_non_ag_vars_kr[k, r] == const_non[r, k]
                    + gp.quicksum(in_nonag.get((r, k), [])) - gp.quicksum(out_nonag.get((r, k), [])),
                    name=f"bal_n_{k}_{r}")
                n += 1
        print(f"│       └── added {n:,} node-balance constraints")


    def _setup_economy_objective(self):
        print("│   ├── setting up objective for economy...")
        
        # Get economic contributions
        ag_obj_mrj, non_ag_obj_rk, ag_man_objs = self._input_data.economic_contr_mrj

        # ACCOUNTING stream: raw coeff (ag_obj_mrj) × X_acct over the accounting support (feasible ∪ slivers).
        ag_exprs = []
        for j in range(self._input_data.n_ag_lus):
            dry_cells = self._input_data.acct_cells_mrj[0, j]
            irr_cells = self._input_data.acct_cells_mrj[1, j]
            ag_exprs.append(
                _qsum(ag_obj_mrj[0, dry_cells, j], self.X_acct_dry_jr[j, dry_cells])
                + _qsum(ag_obj_mrj[1, irr_cells, j], self.X_acct_irr_jr[j, irr_cells])
            )

        ag_mam_exprs = []
        for am, am_j_list in self._input_data.am2j.items():
            if not AG_MANAGEMENTS[am]:
                continue
            for j_idx, j in enumerate(am_j_list):
                dry_cells = self._input_data.feasible_ag_cells_mrj[0, j]
                irr_cells = self._input_data.feasible_ag_cells_mrj[1, j]
                ag_mam_exprs.append(
                    _qsum(ag_man_objs[am][0, dry_cells, j_idx], self.X_ag_man_dry_vars_jr[am][j_idx, dry_cells])
                    + _qsum(ag_man_objs[am][1, irr_cells, j_idx], self.X_ag_man_irr_vars_jr[am][j_idx, irr_cells])
                )

        non_ag_exprs = []
        for k, k_name in enumerate(NON_AG_LAND_USES):
            if not NON_AG_LAND_USES[k_name]:
                continue
            non_ag_cells = self._input_data.feasible_non_ag_cells[k]
            non_ag_exprs.append(
                _qsum(non_ag_obj_rk[non_ag_cells, k], self.X_non_ag_vars_kr[k, non_ag_cells])
            )
        
        self.economy_ag_contr = gp.quicksum(ag_exprs)
        self.economy_ag_man_contr = gp.quicksum(ag_mam_exprs)
        self.economy_non_ag_contr = gp.quicksum(non_ag_exprs)

        # Land-use transition cost = Σ flow_cost · D over the positive-increment delta vars,
        # SUBTRACTED from profit (maxprofit). Source-keyed flow_cost gives the exact per-source transition
        # cost; _qsum drops |coeff| < SOLVER_COEFF_MIN, same filter as every other term.
        def _flow_cost_expr(Fdict, coeff_of):
            if not Fdict:
                return gp.LinExpr(0)
            keys   = list(Fdict.keys())
            coeffs = np.fromiter((coeff_of(k) for k in keys), dtype=np.float64, count=len(keys))
            varr   = np.fromiter((Fdict[k] for k in keys), dtype=object, count=len(keys))
            return _qsum(coeffs, varr)

        trans_a2a = gp.quicksum(
            _flow_cost_expr(self.F_ag2ag[s], (lambda k, c=self._input_data.flow_cost_ag2ag[s]: c[k[0], k[1], k[2]]))
            for s in self.F_ag2ag
        )
        trans_n2a = gp.quicksum(
            _flow_cost_expr(self.F_nonag2ag[fk], (lambda k, c=self._input_data.flow_cost_nonag2ag[fk]: c[k[0], k[1], k[2]]))
            for fk in self.F_nonag2ag
        )
        trans_a2n = gp.quicksum(
            _flow_cost_expr(self.F_ag2nonag[s], (lambda k, c=self._input_data.flow_cost_ag2nonag[s]: c[k[0]][k[1]]))
            for s in self.F_ag2nonag
        )
        self.economy_trans_ag2ag_contr    = -(trans_a2a + trans_n2a)   # all inflows INTO ag targets
        self.economy_trans_ag2nonag_contr = -trans_a2n                 # inflows INTO non-ag targets

        return (
            (
                self.economy_ag_contr 
                + self.economy_ag_man_contr 
                + self.economy_non_ag_contr
                + self.economy_trans_ag2ag_contr 
                + self.economy_trans_ag2nonag_contr
            )   
            * self._input_data.scale_factors['Economy']
            / 1e6  # Convert to million AUD
        )
    
    

    def _add_cell_usage_constraints(self, cells: Optional[np.array] = None):
        """
        Constraint that all of every cell is used for some land use.
        If `cells` is provided, only adds constraints for the given cells
        """
        print("│   ├── Adding constraints for cell usage...")

        if cells is None:
            cells = np.array(range(self._input_data.ncells))

        x_ag_dry_vars = self.X_ag_dry_vars_jr[:, cells]
        x_ag_irr_vars = self.X_ag_irr_vars_jr[:, cells]
        x_non_ag_vars = self.X_non_ag_vars_kr[:, cells]

        # Constrain total (ag + non-ag) land per cell to equal the initial (2010) agricultural proportion.
        #   E.g., under resfactoring, a cell may only be 25% agricultural in the base year,
        #   so total allocation must equal that fraction.
        ag_mask = self._input_data.ag_mask_proportion_r

        # Precompute max feasible allocation per cell.
        # A cell with any ag var can always cover ag_mask (its sources can at least "stay" — conservation,
        # dvar_ub_ag ≥ x_old). Cells with no ag var are limited by the sum of their non-ag UBs.
        has_any_ag_r = (
            (self._input_data.dvar_ub_ag[0] > 0).any(axis=1) |
            (self._input_data.dvar_ub_ag[1] > 0).any(axis=1)
        )
        max_nonag_r = self._input_data.dvar_ub_nonag.sum(axis=1)
        max_alloc_r  = np.where(has_any_ag_r, 1.0, max_nonag_r)
        # Cells where max_alloc < ag_mask cannot satisfy the equality and must be skipped.
        # This covers: (a) cells with no variables at all (max=0), and (b) cells whose only
        # non-ag option has a capped UB below ag_mask (e.g. destock cap < cell ag fraction).
        skip_r = max_alloc_r < ag_mask - 1e-6

        X_sum_r = (
            x_ag_dry_vars.sum(axis=0)
            + x_ag_irr_vars.sum(axis=0)
            + x_non_ag_vars.sum(axis=0)
        )
        # Ranged, not ==: presolve folds bal_a/bal_n into this row and demands
        # sum(base) == ag_mask between two constants summed along different float32 paths,
        # which disagree by up to ~1.75x FeasibilityTol (presolve reads constants exactly,
        # with no tolerance). The +/-10x Ftol band absorbs that residual; conservation
        # (bal_a/bal_n) still pins the cell total to sum(base), so the band is a pure
        # feasibility gate and its width is not exploitable by the objective.
        band = 10 * settings.FEASIBILITY_TOLERANCE
        n_skipped = 0
        for r, expr, ub in zip(cells, X_sum_r, ag_mask[cells]):
            if skip_r[r]:
                n_skipped += 1
                continue
            self.cell_usage_constraint_r[r] = self.gurobi_model.addRange(
                expr, ub - band, ub + band,
                name=f"const_cell_usage_{r}"
            )
        if n_skipped:
            print(f"│   │   WARNING: skipped cell-usage constraint for {n_skipped} cells "
                  f"(max feasible allocation < ag_mask).")


    def _add_agricultural_management_constraints(
        self, cells: Optional[np.array] = None
    ):
        """
        Constraint handling alternative agricultural management options:
        Ag. man. variables cannot exceed the value of the agricultural variable.
        """
        print("│   ├── Adding constraints for agricultural management options...")

        for am, am_j_list in self._input_data.am2j.items():
            for j_idx, j in enumerate(am_j_list):
                if cells is not None:
                    lm_dry_r_vals = [
                        r for r in cells if self._input_data.dvar_ub_ag[0, r, j] > 0
                    ]
                    lm_irr_r_vals = [
                        r for r in cells if self._input_data.dvar_ub_ag[1, r, j] > 0
                    ]
                else:
                    lm_dry_r_vals = self._input_data.feasible_ag_cells_mrj[0, j]
                    lm_irr_r_vals = self._input_data.feasible_ag_cells_mrj[1, j]

                for r in lm_dry_r_vals:
                    constr = self.gurobi_model.addConstr(
                        self.X_ag_man_dry_vars_jr[am][j_idx, r] <= self.X_ag_dry_vars_jr[j, r],
                        name=f"const_ag_mam_dry_usage_{am}_{j}_{r}".replace(" ", "_"),
                    )
                    self.ag_management_constraints_r[r].append(constr)
                for r in lm_irr_r_vals:
                    constr = self.gurobi_model.addConstr(
                        self.X_ag_man_irr_vars_jr[am][j_idx, r] <= self.X_ag_irr_vars_jr[j, r],
                        name=f"const_ag_mam_irr_usage_{am}_{j}_{r}".replace(" ", "_"),
                    )
                    self.ag_management_constraints_r[r].append(constr)

    def _add_agricultural_management_adoption_limit_constraints(self):
        """
        Add adoption limits constraints for agricultural management options.
        """
        print("│   ├── Adding constraints for agricultural management adoption limits...")


        for am, am_j_list in self._input_data.am2j.items():

            for j_idx, j in enumerate(am_j_list):
                adoption_limit = self._input_data.ag_man_limits[am][j]

                dry_cells = self._input_data.feasible_ag_cells_mrj[0, j]
                irr_cells = self._input_data.feasible_ag_cells_mrj[1, j]

                # Sum of all usage of the AM option must be less than the limit
                ag_man_vars_sum = (
                    gp.quicksum(self.X_ag_man_dry_vars_jr[am][j_idx, dry_cells])
                    + gp.quicksum(self.X_ag_man_irr_vars_jr[am][j_idx, irr_cells])
                )

                all_vars_sum = (
                    gp.quicksum(self.X_ag_dry_vars_jr[j, dry_cells])
                    + gp.quicksum(self.X_ag_irr_vars_jr[j, irr_cells])
                )
                
                constr = self.gurobi_model.addConstr(
                    ag_man_vars_sum <= adoption_limit * all_vars_sum,
                    name=f"const_ag_mam_adoption_limit_{am}_{j}".replace(" ", "_"),
                )

                self.adoption_limit_constraints.append(constr)

    def _add_demand_constraints(self):
        """Hard demand constraints via the array path: per-commodity
        quantity rows composed from the shared policy structure; equality where the
        DEMAND_BOUNDS lb==ub, else the SAME LHS row twice under '>' lb and '<' ub.
        """
        print("│   ├── Adding constraints for demand ...")

        print("│   ├── Adding <hard> demand constraints (equality where lb==ub, else lower + upper)...")
        demand_scale = self._input_data.scale_factors['Demand']
        if self._policy_structure is None:                          # shared by water/GHG/demand —
            self.gurobi_model.update()                              # walked once per formulate
            self._policy_structure = extract_structure(self, self._input_data)
            if self._bio_var_list is None:
                self._bio_var_list = self.gurobi_model.getVars()
        nvars = self.gurobi_model.NumVars
        ones_r = np.ones(self._input_data.ncells, dtype=np.float32)

        # Attach per-commodity quantity coefficients to the shared structure. The jc
        # products (pr2cm_cp @ ag_q_mrp slices) run over the structure's own cells —
        # the same slices as the legacy builder (structure ag cells == acct_cells,
        # both ascending), so the per-cell dot products are bitwise the legacy values.
        ncms = self._input_data.ncms
        per_c_groups = [[] for _ in range(ncms)]
        for s in self._policy_structure:
            if s['kind'] in ('ag', 'am'):
                active_p = np.where(self._input_data.lu2pr_pj[:, s['j']])[0]
                if not active_p.size:
                    continue
                q = (self._input_data.ag_q_mrp if s['kind'] == 'ag'
                     else self._input_data.ag_man_q_mrp[s['am']])
                jc = (self._input_data.pr2cm_cp[:, active_p]
                      @ q[s['m'], s['cells'], :][:, active_p].T)
                for c_idx in range(ncms):
                    per_c_groups[c_idx].append(
                        dict(cells=s['cells'], var=s['var'], w=s['w'], c=jc[c_idx]))
            else:
                for c_idx in range(ncms):
                    per_c_groups[c_idx].append(
                        dict(cells=s['cells'], var=s['var'], w=s['w'],
                             c=self._input_data.non_ag_q_crk[c_idx, s['cells'], s['k']]))

        q_rows = [compose_row(per_c_groups[c], ones_r, nvars, settings.SOLVER_COEFF_MIN)
                  for c in range(ncms)]

        rows, senses, rhs, names = [], [], [], []
        for c_idx, c_name in enumerate(self._input_data.commodity_names):
            lb, ub = settings.DEMAND_BOUNDS[c_name]
            lim = self._input_data.limits['demand'][c_idx] / demand_scale
            if lb == ub:
                rows.append(q_rows[c_idx]); senses.append('=')
                rhs.append(lim * lb); names.append(f"demand_hard_bound_eq[{c_idx}]")
            else:
                rows.append(q_rows[c_idx]); senses.append('>')
                rhs.append(lim * lb); names.append(f"demand_hard_bound_lower[{c_idx}]")
                rows.append(q_rows[c_idx]); senses.append('<')
                rhs.append(lim * ub); names.append(f"demand_hard_bound_upper[{c_idx}]")

        block = sparse.vstack(rows, format='csr')
        constrs = self.gurobi_model.addMConstr(
            block, self._bio_var_list, np.array(senses), np.asarray(rhs)).tolist()
        self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
        self.demand_penalty_constraints.extend(constrs)
        self.demand_q_block = sparse.vstack(q_rows, format='csr')   # per-commodity LHS (reporting)



    def _add_water_usage_limit_constraints(self) -> None:

        if settings.WATER_LIMITS != "on":
            print("│   ├── TURNING OFF water usage constraints ...")
            return

        print("│   ├── Adding constraints for water usage limits...")

        water_scale = self._input_data.scale_factors['Water']

        # ARRAY path: per-term water coefficients attached to the shared
        # policy structure; each region's row = compose with that region's 0/1 float32
        # indicator as the val_row (off-region terms give q = 0 -> stage-1 drop, and
        # 1.0f x c == c bitwise, so the drop test sees the legacy coefficient exactly).
        if self._policy_structure is None:                              # shared by water/GHG/demand —
            self.gurobi_model.update()                                  # walked once per formulate
            self._policy_structure = extract_structure(self, self._input_data)
            if self._bio_var_list is None:
                self._bio_var_list = self.gurobi_model.getVars()

        # Attach the water net-yield coefficients to the shared structure: per-term
        # slices of ag_w_mrj / ag_man_w_mrj / non_ag_w_rk — same values, same dtype
        # as the legacy _qsum arguments (coefficients can be NEGATIVE; the stage-1
        # drop tests |q|).
        groups = []
        for s in self._policy_structure:
            if s['kind'] == 'ag':
                c = self._input_data.ag_w_mrj[s['m'], s['cells'], s['j']]
            elif s['kind'] == 'am':
                c = self._input_data.ag_man_w_mrj[s['am']][s['m'], s['cells'], s['j_idx']]
            else:
                c = self._input_data.non_ag_w_rk[s['cells'], s['k']]
            groups.append(dict(cells=s['cells'], var=s['var'], w=s['w'], c=c))
        nvars = self.gurobi_model.NumVars

        rows, names, rhs, regids = [], [], [], []
        for reg_idx, w_limit_raw in self._input_data.limits["water"].items():
            ind = self._input_data.water_region_indices[reg_idx]
            reg_name = self._input_data.water_region_names[reg_idx]
            print(f"│   │   ├── target (inside LUTO study area) is {w_limit_raw:15,.0f} ML for {reg_name}")
            indicator = np.zeros(self._input_data.ncells, dtype=np.float32)
            indicator[ind] = 1.0
            rows.append(compose_row(groups, indicator, nvars, settings.SOLVER_COEFF_MIN))
            names.append(f"water_yield_limit_{reg_name}".replace(" ", "_"))
            rhs.append(w_limit_raw / water_scale)
            regids.append(reg_idx)

        if rows:
            block = sparse.vstack(rows, format='csr')
            constrs = self.gurobi_model.addMConstr(
                block, self._bio_var_list, '>', np.asarray(rhs)).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self.water_limit_constraints.extend(constrs)
            self.water_block, self.water_block_regids = block, regids
      


    def _add_renewable_energy_constraints(self) -> None:

        if not any(settings.RENEWABLES_OPTIONS.values()):
            print("│   ├── TURNING OFF renewable energy constraints ...")
            return

        print("│   ├── Adding constraints for renewable energy production targets ...")

        re_types = {
            'Utility Solar PV': {
                'energy_r':      self._input_data.renewable_solar_r,
                'gbf2_mask_idx': self._input_data.renewable_GBF2_mask_solar_idx,
                'mnes_mask_idx': self._input_data.renewable_MNES_mask_solar_idx,
            },
            'Onshore Wind': {
                'energy_r':      self._input_data.renewable_wind_r,
                'gbf2_mask_idx': self._input_data.renewable_GBF2_mask_wind_idx,
                'mnes_mask_idx': self._input_data.renewable_MNES_mask_wind_idx,
            },
        }

        # Work on a local copy — pop() would mutate data.REGION_STATE_NAME2CODE in-place
        # (the dict is returned by reference), causing a KeyError on subsequent simulation years.
        region_state_name2idx = dict(self._input_data.region_state_name2idx)
        act_code = region_state_name2idx.pop('Australian Capital Territory')

        # ARRAY path: each (state, type) row = the type's am-var structure entries with
        # energy_r coefficients, composed against an allowed-cells indicator (state region,
        # ACT merged into NSW, minus the per-type GBF2/MNES exclusion masks). The existing-
        # capacity constant the legacy expr carried is folded into the RHS (Gurobi's own
        # constant move). Row-inclusion keeps the legacy CELL-SET rule: a row exists iff at
        # least one compatible land use has eligible cells — even if every coefficient there
        # is sub-floor.
        if self._policy_structure is None:                              # shared policy walk —
            self.gurobi_model.update()                                  # once per formulate
            self._policy_structure = extract_structure(self, self._input_data)
            if self._bio_var_list is None:
                self._bio_var_list = self.gurobi_model.getVars()
        nvars = self.gurobi_model.NumVars

        rows, names, rhs, keys = [], [], [], []
        for reg_name, reg_id in region_state_name2idx.items():
            reg_idx = np.where(self._input_data.region_state_r == reg_id)[0]
            # Merge ACT cells into NSW so they count toward the combined NSW+ACT target
            if reg_name == 'New South Wales':
                act_idx = np.where(self._input_data.region_state_r == act_code)[0]
                reg_idx = np.union1d(reg_idx, act_idx)
            print(f"│   │   ├── Adding renewable energy constraints for {reg_name} ...")

            for am, re_data in re_types.items():
                if not settings.AG_MANAGEMENTS[am]:
                    continue

                energy_r      = re_data['energy_r']
                gbf2_mask_idx = re_data['gbf2_mask_idx']
                mnes_mask_idx = re_data['mnes_mask_idx']

                target_raw    = self._input_data.limits[f"renewable_{am}"][reg_name]
                target_rescal = target_raw / self._input_data.scale_factors[am]

                exist_power_mwh     = self._input_data.limits[f"renewable_{am}_exist"][reg_name]
                exist_power_rescale = exist_power_mwh / self._input_data.scale_factors[am]

                print(f"│   │   │   ├── target for {am} is {target_raw:5,.0f} MWh  (existing: {exist_power_mwh:5,.0f} MWh)")

                # Legacy cell-set row-inclusion rule (NOT a coefficient test)
                has_cells = False
                for j in self._input_data.am2j[am]:
                    j_cells = np.union1d(self._input_data.feasible_ag_cells_mrj[0, j],
                                         self._input_data.feasible_ag_cells_mrj[1, j])
                    rc = np.intersect1d(j_cells, reg_idx)
                    if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS == True:
                        rc = np.setdiff1d(rc, gbf2_mask_idx)            # no renewables in GBF2-masked cells
                    if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK == True:
                        rc = np.setdiff1d(rc, mnes_mask_idx)            # no renewables in EPBC MNES cells
                    if rc.size:
                        has_cells = True
                        break
                if not has_cells:
                    continue

                allowed = np.zeros(self._input_data.ncells, dtype=np.float32)
                allowed[reg_idx] = 1.0
                if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS == True:
                    allowed[gbf2_mask_idx] = 0.0
                if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK == True:
                    allowed[mnes_mask_idx] = 0.0

                groups = [dict(cells=s['cells'], var=s['var'], w=s['w'], c=energy_r[s['cells']])
                          for s in self._policy_structure
                          if s['kind'] == 'am' and s['am'] == am]
                rows.append(compose_row(groups, allowed, nvars, settings.SOLVER_COEFF_MIN))
                names.append(f"renewable_{am}_target_{reg_name}".replace(" ", "_"))
                rhs.append(target_rescal - exist_power_rescale)
                keys.append(f'{am}_{reg_name}')

        if rows:
            block = sparse.vstack(rows, format='csr')
            constrs = self.gurobi_model.addMConstr(
                block, self._bio_var_list, '>', np.asarray(rhs)).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self.renewable_constraints = dict(zip(keys, constrs))
                



    def _add_ghg_emissions_limit_constraints(self):
        """Hard GHG emissions cap via the array path. GHG is
        HARD-ONLY — the soft variant and GHG_CONSTRAINT_TYPE were removed
        2026-09-02 (with the E penalty var and _get_total_ghg_expr)."""
        if settings.GHG_EMISSIONS_LIMITS == "off":
            print("│   ├── TURNING OFF GHG emissions constraints ...")
            return

        ghg_limit_raw = self._input_data.limits["ghg"]
        ghg_limit_rescale = ghg_limit_raw / self._input_data.scale_factors['GHG']

        print(f"│   ├── Adding <hard> constraints for GHG emissions: {ghg_limit_raw:,.0f} tCO2e")
        # ARRAY path: GHG coefficients + the transition-delta stream
        # on the shared policy structure, composed with an all-ones val_row (global
        # row). The offland_ghg CONSTANT that the legacy expr carried is folded into
        # the RHS here — exactly the constant move Gurobi performed at addConstr.
        if self._policy_structure is None:                          # shared by water/GHG/demand —
            self.gurobi_model.update()                              # walked once per formulate
            self._policy_structure = extract_structure(self, self._input_data)
            if self._bio_var_list is None:
                self._bio_var_list = self.gurobi_model.getVars()

        # Attach the GHG coefficients to the shared structure ...
        groups = []
        for s in self._policy_structure:
            if s['kind'] == 'ag':
                c = self._input_data.ag_g_mrj[s['m'], s['cells'], s['j']]
            elif s['kind'] == 'am':
                c = self._input_data.ag_man_g_mrj[s['am']][s['m'], s['cells'], s['j_idx']]
            else:
                c = self._input_data.non_ag_g_rk[s['cells'], s['k']]
            groups.append(dict(cells=s['cells'], var=s['var'], w=s['w'], c=c))
        # ... plus the transition-delta stream: one group per ag source, coefficients
        # gathered float64 exactly as the legacy np.fromiter(..., dtype=np.float64).
        # Delta groups carry cells=0 placeholders — valid only under the all-ones
        # val_row below (the GHG row is global).
        for src, Fdict in self.F_ag2ag.items():
            if not Fdict:
                continue
            carr = self._input_data.flow_ghg_ag2ag[src]             # [to_m, local_r, to_j]
            keys = list(Fdict.keys())
            coeffs = np.fromiter((carr[k[0], k[1], k[2]] for k in keys),
                                 dtype=np.float64, count=len(keys))
            var_idx = np.fromiter((Fdict[k].index for k in keys),
                                  dtype=np.int32, count=len(keys))
            groups.append(dict(cells=np.zeros(len(keys), dtype=np.int32),
                               var=var_idx, w=np.ones(len(keys)), c=coeffs))
        ones_r = np.ones(self._input_data.ncells, dtype=np.float32)
        row = compose_row(groups, ones_r, self.gurobi_model.NumVars,
                          settings.SOLVER_COEFF_MIN)
        # offland_ghg arrives as a 1-element 1-D array (OFF_LAND_GHG_EMISSION_C row);
        # ravel to a length-1 RHS vector for the single row
        rhs = np.asarray(ghg_limit_rescale - self._input_data.offland_ghg,
                         dtype=np.float64).ravel()
        constrs = self.gurobi_model.addMConstr(
            row, self._bio_var_list, '<', rhs).tolist()
        self.gurobi_model.setAttr('ConstrName', constrs, ["ghg_emissions_limit_ub"])
        self.ghg_consts_ub = constrs[0]
        self.ghg_block = row
            
            
    def _add_biodiversity_constraints(self) -> None:
        print("│   ├── Adding constraints for biodiversity...")
        self._add_GBF2_constraints()
        self._add_GBF3_NVIS_constraints()
        self._add_GBF4_SNES_constraints()
        self._add_GBF4_ECNES_constraints()
        self._add_GBF8_constraints()


    def _add_GBF2_constraints(self) -> None:
        
        if settings.GBF2_TARGET == "off":
            print("│   │   ├── TURNING OFF constraints for biodiversity GBF 2...")
            return
        
        print(f'│   │   ├── Adding constraints for biodiversity GBF 2: {self._input_data.limits["GBF2"]:15,.0f}')

        # ARRAY path: the LHS is the shared bio
        # contribution operator applied to GBF2_mask_area_r — which is ZERO off-mask,
        # so the legacy index intersections (acct/feasible ∩ GBF2_mask_idx) reproduce
        # themselves through the stage-1 drop. Hard-only (GBF2_CONSTRAINT_TYPE is read
        # nowhere in the solver); the single-Constr contract of bio_GBF2_constrs kept.
        if self._bio_groups is None:                                    # shared across GBF2/3/4/8 —
            self.gurobi_model.update()                                  # extracted once per formulate
            self._bio_groups = extract_groups(self, self._input_data)
            self._bio_var_list = self.gurobi_model.getVars()            # materialized once, reused by every family
        row = compose_row(self._bio_groups, self._input_data.GBF2_mask_area_r,
                          self.gurobi_model.NumVars, settings.SOLVER_COEFF_MIN)
        rhs = self._input_data.limits["GBF2"] / self._input_data.scale_factors['GBF2']
        constrs = self.gurobi_model.addMConstr(
            row, self._bio_var_list, '>', np.asarray([rhs])).tolist()
        self.gurobi_model.setAttr(
            'ConstrName', constrs, ["bio_GBF2_priority_degraded_area_limit"])
        self.bio_GBF2_constrs = constrs[0]                              # SINGLE Constr (legacy contract)
        self.bio_GBF2_block = row


    def bio_constraint_index(self) -> dict:
        """{constraint_name: {family, region, item, presence}} for every biodiversity row built.

        CACHED, and built eagerly at the end of `formulate()`. It must be: reading `ConstrName`
        from a Constr that has since been removed from the model raises "Constr was removed from
        the model", and the whole point of this index is to describe rows that were dropped. Build
        it once while every constraint is still live, then look up freely afterwards.

        The constraint NAME is all a Gurobi model carries, and it cannot be parsed back into its
        parts: `.replace(" ", "_")` makes the separator ambiguous ("Goulburn_Broken" vs the
        underscore before the community), and the arity differs by family — SNES/ECNES have a
        presence class, GBF3/GBF8 do not, GBF2 has no key at all. So record the mapping here, where
        the tuple is still known, instead of reconstructing it later from a mangled string.
        """
        if self._bio_index is not None:
            return self._bio_index

        index = {}
        specs = [
            ('GBF3_NVIS',  self.bio_GBF3_NVIS_constrs,  ('region', 'item')),
            ('GBF4_SNES',  self.bio_GBF4_SNES_constrs,  ('region', 'item', 'presence')),
            ('GBF4_ECNES', self.bio_GBF4_ECNES_constrs, ('region', 'item', 'presence')),
            ('GBF8',       self.bio_GBF8_constrs,       ('region', 'item')),
        ]
        for family, constrs, fields in specs:
            for key, constr in (constrs or {}).items():
                row = {'family': family, 'region': None, 'item': None, 'presence': None}
                row.update(dict(zip(fields, key)))
                index[constr.ConstrName] = row

        # GBF2 is a single national row with no key, so it has no parts to record.
        gbf2 = self.bio_GBF2_constrs
        if gbf2 is not None and not isinstance(gbf2, dict):
            index[gbf2.ConstrName] = {'family': 'GBF2', 'region': None, 'item': None,
                                      'presence': None}
        self._bio_index = index
        return index


    def remove_constraints_by_name(self, names) -> None:
        """Remove rows from the Gurobi model AND from the bookkeeping the dual readers walk.

        The infeasibility flow in `simulation.py` drops rows by name. Removing them only from the
        model leaves stale `Constr` objects in these collections, and the first attribute read on
        one — `record_shadow_prices` reading `.Pi` after the next ACCEPTED solve — raises
        "Constr was removed from the model". So the two removals must happen together, here.

        The purged collections are exactly the ones `record_shadow_prices` iterates. The structural
        per-cell collections (cell usage, ag-management links) are not scanned: they are millions
        of rows, no sane `DROP_UNREACHABLE_CONSTRAINTS` policy includes them, and no dual reader
        walks them.

        `bio_constraint_index()` is deliberately NOT invalidated — it is warmed while every row is
        still live precisely so that dropped rows can be described afterwards.
        """
        if not names:
            return
        doomed = set(names)

        # Purge bookkeeping FIRST, while every held Constr can still be matched by name.
        for coll in (self.bio_GBF3_NVIS_constrs, self.bio_GBF4_SNES_constrs,
                     self.bio_GBF4_ECNES_constrs, self.bio_GBF8_constrs,
                     self.renewable_constraints):
            for key in [k for k, c in coll.items() if c.ConstrName in doomed]:
                del coll[key]

        self.water_limit_constraints    = [c for c in self.water_limit_constraints if c.ConstrName not in doomed]
        self.regional_adoption_constrs  = [c for c in self.regional_adoption_constrs if c.ConstrName not in doomed]
        self.demand_penalty_constraints = [c for c in self.demand_penalty_constraints if c.ConstrName not in doomed]

        if isinstance(self.bio_GBF2_constrs, gp.Constr) and self.bio_GBF2_constrs.ConstrName in doomed:
            self.bio_GBF2_constrs = {}      # back to the "not built" sentinel
        if self.ghg_consts_ub is not None and self.ghg_consts_ub.ConstrName in doomed:
            self.ghg_consts_ub = None

        self.gurobi_model.remove(
            [c for c in self.gurobi_model.getConstrs() if c.ConstrName in doomed])
        self.gurobi_model.update()


    def _add_GBF3_NVIS_constraints(self) -> None:
        if settings.GBF3_NVIS_TARGET == "off":
            print("│   │   ├── TURNING OFF constraints for biodiversity GBF 3 NVIS")
            return

        print("│   │   ├── Adding constraints for biodiversity GBF 3 NVIS...")
        pairs      = self._input_data.GBF3_NVIS_region_group            # list[(region, group)]
        v_limits   = self._input_data.limits["GBF3_NVIS"]               # xr [layer=(region, group)]
        scales     = self._input_data.scale_factors['GBF3_NVIS']        # xr [layer=(region, group)]
        val_matrix = self._input_data.GBF3_NVIS_pre_1750_area_vr        # xr [group, cell]
        reg_matrix = self._input_data.region_NRM_names_r                # np [cell]

        if self._bio_groups is None:                                    # shared across GBF3/GBF4/GBF8 —
            self.gurobi_model.update()                                  # extracted once per formulate
            self._bio_groups = extract_groups(self, self._input_data)
            self._bio_var_list = self.gurobi_model.getVars()      # materialized once, reused by every family
        nvars = self.gurobi_model.NumVars

        # Compose one CSR row per active pair — same skip semantics as the legacy loop.
        # NOTE the GBF3 skip is `lb_raw < 0` (NOT <= 0): a ZERO target still adds a row.
        rows, names, rhs, kept = [], [], [], []
        for region, group in pairs:
            lb_raw = v_limits.sel(dict(layer=(region, group))).item()
            if lb_raw < 0:
                continue
            val_row = val_matrix.sel(group=group, drop=True).data
            if region != "AUSTRALIA":                                   # NRM scope: mask non-region cells
                val_row = np.where(reg_matrix == region, val_row, 0)
            if not (val_row > 0).any():
                continue
            rows.append(compose_row(self._bio_groups, val_row, nvars, settings.SOLVER_COEFF_MIN))
            names.append(f"bio_GBF3_NVIS_limit_{region}_{group}".replace(" ", "_"))
            rhs.append(lb_raw / scales.sel(dict(layer=(region, group))).item())
            kept.append((region, group))

        if rows:
            block = sparse.vstack(rows, format='csr')
            constrs = self.gurobi_model.addMConstr(
                block, self._bio_var_list, '>', np.asarray(rhs)).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self.bio_GBF3_NVIS_constrs = dict(zip(kept, constrs))
            self.bio_GBF3_NVIS_block, self.bio_GBF3_NVIS_block_pairs = block, kept
        print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")





    def _add_GBF4_SNES_constraints(self) -> None:
        if settings.GBF4_TARGET_SNES == 'off':
            print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 SNES...')
            return

        print("│   │   ├── Adding constraints for biodiversity GBF 4 SNES ...")
        pairs      = self._input_data.GBF4_SNES_region_species          # list[(region, species, presence)]
        v_limits   = self._input_data.limits["GBF4_SNES"]               # xr [layer=(region, species, presence)]
        scales     = self._input_data.scale_factors['GBF4_SNES']        # xr [layer=(region, species, presence)]
        val_matrix = self._input_data.GBF4_SNES_pre_1750_area_sr        # xr [layer=(species, presence), cell]
        reg_matrix = self._input_data.region_NRM_names_r                # np [cell]

        if self._bio_groups is None:                                    # shared across GBF4/GBF8 —
            self.gurobi_model.update()                                  # extracted once per formulate
            self._bio_groups = extract_groups(self, self._input_data)
            self._bio_var_list = self.gurobi_model.getVars()      # materialized once, reused by every family
        nvars = self.gurobi_model.NumVars

        # Compose one CSR row per active key — same skip semantics as the legacy loop:
        # raw target <= 0, or no positive cell in the (region-masked) layer.
        rows, names, rhs, kept = [], [], [], []
        for region, species, presence in pairs:
            lb_raw = v_limits.sel(dict(layer=(region, species, presence))).item()
            if lb_raw <= 0:
                continue
            val_row = val_matrix.sel(dict(layer=(species, presence)), drop=True).values
            if region != "AUSTRALIA":                                   # NRM scope: mask non-region cells
                val_row = np.where(reg_matrix == region, val_row, 0)
            if not (val_row > 0).any():
                continue
            rows.append(compose_row(self._bio_groups, val_row, nvars, settings.SOLVER_COEFF_MIN))
            names.append(f"bio_GBF4_SNES_limit_{region}_{species}_{presence}".replace(" ", "_"))
            rhs.append(lb_raw / scales.sel(dict(layer=(region, species, presence))).item())
            kept.append((region, species, presence))

        if rows:
            block = sparse.vstack(rows, format='csr')
            constrs = self.gurobi_model.addMConstr(
                block, self._bio_var_list, '>', np.asarray(rhs)).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self.bio_GBF4_SNES_constrs = dict(zip(kept, constrs))
            self.bio_GBF4_SNES_block, self.bio_GBF4_SNES_block_pairs = block, kept
        print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")

    def _add_GBF4_ECNES_constraints(self) -> None:
        if settings.GBF4_TARGET_ECNES == 'off':
            print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 ECNES...')
            return

        print("│   │   ├── Adding constraints for biodiversity GBF 4 ECNES ...")
        pairs      = self._input_data.GBF4_ECNES_region_species         # list[(region, community, presence)]
        v_limits   = self._input_data.limits["GBF4_ECNES"]              # xr [layer=(region, community, presence)]
        scales     = self._input_data.scale_factors['GBF4_ECNES']       # xr [layer=(region, community, presence)]
        val_matrix = self._input_data.GBF4_ECNES_pre_1750_area_sr       # xr [layer=(community, presence), cell]
        reg_matrix = self._input_data.region_NRM_names_r                # np [cell]

        if self._bio_groups is None:                                    # shared across GBF4/GBF8 —
            self.gurobi_model.update()                                  # extracted once per formulate
            self._bio_groups = extract_groups(self, self._input_data)
            self._bio_var_list = self.gurobi_model.getVars()      # materialized once, reused by every family
        nvars = self.gurobi_model.NumVars

        # Compose one CSR row per active key — same skip semantics as the legacy loop:
        # raw target <= 0, or no positive cell in the (region-masked) layer.
        rows, names, rhs, kept = [], [], [], []
        for region, community, presence in pairs:
            lb_raw = v_limits.sel(dict(layer=(region, community, presence))).item()
            if lb_raw <= 0:
                continue
            val_row = val_matrix.sel(dict(layer=(community, presence)), drop=True).values
            if region != "AUSTRALIA":                                   # NRM scope: mask non-region cells
                val_row = np.where(reg_matrix == region, val_row, 0)
            if not (val_row > 0).any():
                continue
            rows.append(compose_row(self._bio_groups, val_row, nvars, settings.SOLVER_COEFF_MIN))
            names.append(f"bio_GBF4_ECNES_limit_{region}_{community}_{presence}".replace(" ", "_"))
            rhs.append(lb_raw / scales.sel(dict(layer=(region, community, presence))).item())
            kept.append((region, community, presence))

        if rows:
            block = sparse.vstack(rows, format='csr')
            constrs = self.gurobi_model.addMConstr(
                block, self._bio_var_list, '>', np.asarray(rhs)).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self.bio_GBF4_ECNES_constrs = dict(zip(kept, constrs))
            self.bio_GBF4_ECNES_block, self.bio_GBF4_ECNES_block_pairs = block, kept
        print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")


    def _add_GBF8_constraints(self) -> None:

        if settings.GBF8_TARGET == "off":
            print('│   │   ├── TURNING OFF constraints for biodiversity GBF 8 ...')
            return

        print("│   │   ├── Adding constraints for biodiversity GBF 8 ...")
        pairs      = self._input_data.GBF8_region_species               # list[(region, species)]
        v_limits   = self._input_data.limits["GBF8"]                    # xr [layer=(region, species)]
        scales     = self._input_data.scale_factors['GBF8']             # xr [layer=(region, species)]
        val_matrix = self._input_data.GBF8_pre_1750_area_sr             # xr [species, cell]
        reg_matrix = self._input_data.region_NRM_names_r                # np [cell]

        if self._bio_groups is None:                                    # shared across GBF4/GBF8 —
            self.gurobi_model.update()                                  # extracted once per formulate
            self._bio_groups = extract_groups(self, self._input_data)
            self._bio_var_list = self.gurobi_model.getVars()      # materialized once, reused by every family
        nvars = self.gurobi_model.NumVars

        # Compose one CSR row per active pair — same skip semantics as the legacy loop:
        # raw target <= 0, or no positive cell in the (region-masked) species layer.
        rows, names, rhs, kept = [], [], [], []
        for region, species in pairs:
            lb_raw = v_limits.sel(dict(layer=(region, species))).item()
            if lb_raw <= 0:
                continue
            val_row = val_matrix.sel(species=species, drop=True).data
            if region != "AUSTRALIA":                                   # NRM scope: mask non-region cells
                val_row = np.where(reg_matrix == region, val_row, 0)
            if not (val_row > 0).any():
                continue
            rows.append(compose_row(self._bio_groups, val_row, nvars, settings.SOLVER_COEFF_MIN))
            names.append(f"bio_GBF8_limit_{region}_{species}".replace(" ", "_"))
            rhs.append(lb_raw / scales.sel(dict(layer=(region, species))).item())
            kept.append((region, species))

        if rows:
            block = sparse.vstack(rows, format='csr')
            constrs = self.gurobi_model.addMConstr(
                block, self._bio_var_list, '>', np.asarray(rhs)).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self.bio_GBF8_constrs = dict(zip(kept, constrs))
            self.bio_GBF8_block, self.bio_GBF8_block_pairs = block, kept
        print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")


    def _add_regional_adoption_constraints(self) -> None:

        if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
            print("│   │   └── TURNING OFF constraints for regional adoption ...")
            return

        # Add adoption constraints for agricultural land uses
        reg_adopt_limits = self._input_data.limits["ag_regional_adoption"]
        for reg_id, j, lu_name, reg_ind, reg_area_limit in reg_adopt_limits:
            if len(reg_ind) == 0:
                print(f"│   │   │   ├── SKIPPING {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} (no cells at this resolution)")
                continue
            print(f"│   │   │   ├── Adding constraints for {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} <= {reg_area_limit:,.0f} HA...")
            reg_expr = (
                  _qsum(self._input_data.real_area[reg_ind], self.X_ag_dry_vars_jr[j, reg_ind])
                + _qsum(self._input_data.real_area[reg_ind], self.X_ag_irr_vars_jr[j, reg_ind])
            )
            self.regional_adoption_constrs.append(
                self.gurobi_model.addConstr(
                    reg_expr <= reg_area_limit, 
                    name=f"reg_adopt_limit_ag_{lu_name}_{reg_id}".replace(" ", "_")
                )
            )

        # Non-reversible plantings saturate the non-ag caps below, and last year's solved
        # areas become this year's exact lower bounds; float32 noise then puts the locked-in
        # floor a hair over the cap, which presolve rejects with NO tolerance (bound
        # propagation is exact). Grow the cap by 1e-6/yr RELATIVE so the RHS always recedes
        # ahead of the ratcheting floor (per-step increment ~5e-6 x cap vs float noise
        # ~2e-10 x cap). Cap erosion by 2050: ~3e-5 relative. Ag caps need no slack: ag is
        # reversible, so its floors never ratchet onto the cap.
        nonag_cap_relax = 1 + (self._input_data.target_year - settings.SIM_YEARS[0]) * 1e-6

        # Add per-(region, non-ag-landuse) caps from the xlsx ('on' mode)
        reg_adopt_non_ag_limits = self._input_data.limits.get("non_ag_regional_adoption") or []
        for reg_id, k, lu_name, reg_ind, reg_area_limit in reg_adopt_non_ag_limits:
            if len(reg_ind) == 0:
                print(f"│   │   │   ├── SKIPPING {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} (no cells at this resolution)")
                continue
            print(f"│   │   │   ├── Adding constraints for {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} <= {reg_area_limit:,.0f} HA...")
            reg_expr = _qsum(self._input_data.real_area[reg_ind], self.X_non_ag_vars_kr[k, reg_ind])
            self.regional_adoption_constrs.append(
                self.gurobi_model.addConstr(
                    reg_expr <= reg_area_limit * nonag_cap_relax, 
                    name=f"reg_adopt_limit_non_ag_{lu_name}_{reg_id}".replace(" ", "_")
                )
            )

        # Add SUM-of-non-ag adoption constraints ('NON_AG_CAP' mode):
        # the combined area of all non-ag land uses in each region cannot exceed the uniform percentage cap.
        reg_adopt_sum_limits = self._input_data.limits.get("non_ag_regional_adoption_sum") or []
        for reg_id, reg_ind, reg_area_limit in reg_adopt_sum_limits:
            if len(reg_ind) == 0:
                print(f"│   │   │   ├── SKIPPING SUM-of-non-ag constraint for {settings.REGIONAL_ADOPTION_NON_AG_REGION} region {reg_id} (no cells at this resolution)")
                continue
            print(f"│   │   │   ├── Adding SUM-of-non-ag constraint for {settings.REGIONAL_ADOPTION_NON_AG_REGION} region {reg_id} <= {reg_area_limit:,.0f} HA...")
            reg_expr = gp.LinExpr(0)
            for k in range(self.X_non_ag_vars_kr.shape[0]):
                reg_expr += _qsum(self._input_data.real_area[reg_ind], self.X_non_ag_vars_kr[k, reg_ind])
            self.regional_adoption_constrs.append(
                self.gurobi_model.addConstr(
                    reg_expr <= reg_area_limit * nonag_cap_relax, 
                    name=f"reg_adopt_limit_non_ag_sum_{reg_id}".replace(" ", "_")
                )
            )



    def solve(self) -> SolverSolution | None:
        print("Starting solve...\n")

        # Magic.
        self.gurobi_model.optimize()

        # Bail out if no solution is available (e.g., infeasible model).
        if self.gurobi_model.SolCount == 0:
            print(
                f"No solution available (Status={self.gurobi_model.Status}, SolCount=0); "
                f"skipping result collection.\n",
                flush=True,
            )
            return None

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
            for r in self._input_data.feasible_ag_cells_mrj[0, j]:
                X_dry_sol_rj[r, j] = self.X_ag_dry_vars_jr[j, r].X
            for r in self._input_data.feasible_ag_cells_mrj[1, j]:
                X_irr_sol_rj[r, j] = self.X_ag_irr_vars_jr[j, r].X

        # Get non-agricultural results
        for k, lu in enumerate(settings.NON_AG_LAND_USES):
            if not settings.NON_AG_LAND_USES[lu]:
                non_ag_X_sol_rk[:, k] = np.zeros(self._input_data.ncells)
                continue

            for r in self._input_data.feasible_non_ag_cells[k]:
                non_ag_X_sol_rk[r, k] = self.X_non_ag_vars_kr[k, r].X

        # Get agricultural management results
        for am, am_j_list in self._input_data.am2j.items():
            for j_idx, j in enumerate(am_j_list):
                eligible_dry_cells = self._input_data.feasible_ag_cells_mrj[0, j]
                eligible_irr_cells = self._input_data.feasible_ag_cells_mrj[1, j]

                if am == "Savanna Burning":
                    eligible_dry_cells = np.intersect1d(
                        eligible_dry_cells, self._input_data.savanna_eligible_r
                    )
                    eligible_irr_cells = np.intersect1d(
                        eligible_irr_cells, self._input_data.savanna_eligible_r
                    )

                if am in settings.RENEWABLES_OPTIONS:
                    gbf2_excl_idx = (
                        self._input_data.renewable_GBF2_mask_solar_idx
                        if am == "Utility Solar PV"
                        else self._input_data.renewable_GBF2_mask_wind_idx
                    )
                    if gbf2_excl_idx.size:
                        eligible_dry_cells = np.setdiff1d(eligible_dry_cells, gbf2_excl_idx)
                        eligible_irr_cells = np.setdiff1d(eligible_irr_cells, gbf2_excl_idx)

                for r in eligible_dry_cells:
                    am_X_dry_sol_rj[am][r, j] = self.X_ag_man_dry_vars_jr[am][
                        j_idx, r
                    ].X
                for r in eligible_irr_cells:
                    am_X_irr_sol_rj[am][r, j] = self.X_ag_man_irr_vars_jr[am][
                        j_idx, r
                    ].X

        # Stack dryland and irrigated decision variables — fractional values preserved as-is
        ag_X_mrj = np.stack((X_dry_sol_rj, X_irr_sol_rj))  # Float32

        # Transition deltas from the SOLVED per-source delta vars — the gross flows the objective
        # actually charged, kept SOURCE-KEYED so reporting can attribute the TRUE from→to land-use
        # flows (X-derived max(0, X_new − x_old) can neither split ag2ag from nonag2ag inflows nor
        # attribute a flow to its source LU). Leaf axes mirror the flow_cost dicts — [to_m, local_r,
        # to_j] for ag targets, [local_r, k] for non-ag targets — where local_r indexes the source's
        # cell list (recover global cells via get_base_dvar_mj_cell_map / get_base_nonag_dvar_k_cell_map
        # at the base year, the same maps that built ag_source_cells / nonag_source_cells).
        dvar_D_ag2ag_mrj    = {}   # (from_m, from_j) -> (NLMS, ncells_src, N_AG_LUS)
        dvar_D_ag2nonag_rk  = {}   # (from_m, from_j) -> (ncells_src, N_NON_AG_LUS)
        dvar_D_nonag2ag_mrj = {}   # from_k           -> (NLMS, ncells_k, N_AG_LUS)
        for (fm, fj), cells in self._input_data.ag_source_cells.items():
            arr = np.zeros((self._input_data.nlms, len(cells), self._input_data.n_ag_lus), dtype=np.float32)
            Fd = self.F_ag2ag[(fm, fj)]
            if len(Fd):
                keys = np.array(list(Fd.keys()), dtype=np.int64)                            # (n, 3): to_m, local_r, to_j
                vals = np.array(self.gurobi_model.getAttr('X', list(Fd.values())), dtype=np.float32)
                arr[keys[:, 0], keys[:, 1], keys[:, 2]] = vals
            dvar_D_ag2ag_mrj[(fm, fj)] = arr

            arr = np.zeros((len(cells), self._input_data.n_non_ag_lus), dtype=np.float32)
            Fd = self.F_ag2nonag[(fm, fj)]
            if len(Fd):
                keys = np.array(list(Fd.keys()), dtype=np.int64)                            # (n, 2): k, local_r
                vals = np.array(self.gurobi_model.getAttr('X', list(Fd.values())), dtype=np.float32)
                arr[keys[:, 1], keys[:, 0]] = vals
            dvar_D_ag2nonag_rk[(fm, fj)] = arr
        for fk, cells in self._input_data.nonag_source_cells.items():
            arr = np.zeros((self._input_data.nlms, len(cells), self._input_data.n_ag_lus), dtype=np.float32)
            Fd = self.F_nonag2ag[fk]
            if len(Fd):
                keys = np.array(list(Fd.keys()), dtype=np.int64)                            # (n, 3): to_m, local_r, to_j
                vals = np.array(self.gurobi_model.getAttr('X', list(Fd.values())), dtype=np.float32)
                arr[keys[:, 0], keys[:, 1], keys[:, 2]] = vals
            dvar_D_nonag2ag_mrj[fk] = arr

        ag_man_X_mrj = {
            am: np.stack((am_X_dry_sol_rj[am], am_X_irr_sol_rj[am]))
            for am in self._input_data.am2j
        }

        # Vector indexed by cell: True where non-ag dvar dominates (used for lumap/lmmap only)
        non_ag_bools_r = non_ag_X_sol_rk.max(axis=1) > ag_X_mrj.max(axis=(0, 2))

        # Calculate 1D array (maps) of land-use and land management, considering only agricultural LUs
        lumap = ag_X_mrj.sum(axis=0).argmax(axis=1).astype("int8")
        lmmap = ag_X_mrj.sum(axis=2).argmax(axis=0).astype("int8")

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

        x_vals = np.asarray(self.gurobi_model.getAttr('X', self.gurobi_model.getVars()),
                            dtype=np.float64)

        # Process production amount for each commodity — one mat-vec over the stored
        # per-commodity block (soft demand removed 2026-09-02; hard is the only mode)
        prod_data["Production"] = (
            [float(v) * self._input_data.scale_factors['Demand']
             for v in (self.demand_q_block @ x_vals[:self.demand_q_block.shape[1]])]
            if self.demand_q_block is not None
            else 0
        )
        prod_data["GHG"] = (
            # hard path: block excludes the offland constant — add it back (as the
            # legacy expr.getValue() included it), then unscale
            (float((self.ghg_block @ x_vals[:self.ghg_block.shape[1]])[0])
             + float(np.asarray(self._input_data.offland_ghg).ravel()[0]))
            * self._input_data.scale_factors['GHG']
            if self.ghg_block is not None
            else 0
        )
        prod_data["Water"] = (
            {reg_idx: float(v) * self._input_data.scale_factors['Water']
             for reg_idx, v in zip(self.water_block_regids,
                                   self.water_block @ x_vals[:self.water_block.shape[1]])}
            if self.water_block is not None
            else 0
        )
        # array path (GBF2/GBF3/GBF4/GBF8): one mat-vec per stored block replaces the
        # per-item LinExpr.getValue() walks (equal up to float summation order — report only)
        def _bio_block_values(block, pairs, scale_da):
            if block is None:
                return {}                                   # family on, zero rows added
            lhs = block @ x_vals[:block.shape[1]]
            return {key: float(v) * scale_da.sel(dict(layer=key)).item()
                    for key, v in zip(pairs, lhs)}

        prod_data["BIO (GBF2) value (ha)"] = (
            float((self.bio_GBF2_block @ x_vals[:self.bio_GBF2_block.shape[1]])[0])
            * self._input_data.scale_factors['GBF2']
            if settings.GBF2_TARGET != "off" and self.bio_GBF2_block is not None
            else 0
        )
        prod_data["BIO (GBF3) NVIS value (ha)"] = (
            _bio_block_values(self.bio_GBF3_NVIS_block, self.bio_GBF3_NVIS_block_pairs,
                              self._input_data.scale_factors['GBF3_NVIS'])
            if settings.GBF3_NVIS_TARGET != 'off'
            else 0
        )
        prod_data["BIO (GBF4) SNES value (ha)"] = (
            _bio_block_values(self.bio_GBF4_SNES_block, self.bio_GBF4_SNES_block_pairs,
                              self._input_data.scale_factors['GBF4_SNES'])
            if settings.GBF4_TARGET_SNES != 'off'
            else 0
        )
        prod_data["BIO (GBF4) ECNES value (ha)"] = (
            _bio_block_values(self.bio_GBF4_ECNES_block, self.bio_GBF4_ECNES_block_pairs,
                              self._input_data.scale_factors['GBF4_ECNES'])
            if settings.GBF4_TARGET_ECNES != 'off'
            else 0
        )
        prod_data["BIO (GBF8) value (ha)"] = (
            _bio_block_values(self.bio_GBF8_block, self.bio_GBF8_block_pairs,
                              self._input_data.scale_factors['GBF8'])
            if settings.GBF8_TARGET != "off"
            else 0
        )
                

        return SolverSolution(
            lumap=lumap,
            lmmap=lmmap,
            ammaps=ammaps,
            ag_X_mrj=ag_X_mrj,
            non_ag_X_rk=non_ag_X_sol_rk,
            ag_man_X_mrj=ag_man_X_mrj,
            dvar_D_ag2ag_mrj=dvar_D_ag2ag_mrj,
            dvar_D_ag2nonag_rk=dvar_D_ag2nonag_rk,
            dvar_D_nonag2ag_mrj=dvar_D_nonag2ag_mrj,
            prod_data=prod_data,
            obj_val={
                "ObjVal":(
                    None 
                    if self.gurobi_model.Status != GRB.OPTIMAL 
                    else self.gurobi_model.ObjVal
                ),
                
                "Obj Economy":                      self.obj_economy.getValue(),
                "Obj Penalties":                    0,   # soft constraints removed 2026-09-02 — no penalty objective

                'Economy (AUD) Ag':                 self.economy_ag_contr.getValue() * self._input_data.scale_factors['Economy'],
                'Economy (AUD) Non-Ag Value':       self.economy_non_ag_contr.getValue() * self._input_data.scale_factors['Economy'],
                'Economy (AUD) Ag-Man Value':       self.economy_ag_man_contr.getValue() * self._input_data.scale_factors['Economy'],

                "Deviation Production (t)":[
                    prod_data["Production"][c] - self._input_data.limits['demand'][c]
                    for c in range(self._input_data.ncms)
                ],   
                "Deviation Water (ML)":(
                    [
                        prod_data["Water"][i] - water_limit
                        for i,water_limit in self._input_data.limits['water'].items()
                    ]                                                                        
                    if settings.WATER_LIMITS == "on"       
                    else 0
                ),         
                "Deviation GHG (tCO2e)": 0,   # was soft-mode-only; GHG soft removed 2026-09-02
                "Deviation BIO (GBF2) value (ha)":(
                    0                                                                             
                    if settings.GBF2_TARGET == "off"         
                    else [
                        prod_data["BIO (GBF2) value (ha)"] - self._input_data.limits['GBF2']
                    ]         
                ),
                "Deviation BIO (GBF3) NVIS value (ha)":(
                    0                                                                               
                    if settings.GBF3_NVIS_TARGET == "off"         
                    else [
                        v - self._input_data.limits['GBF3_NVIS'].sel(dict(layer=k)).item()
                        for k,v in prod_data["BIO (GBF3) NVIS value (ha)"].items()
                    ]
                ),
                "Deviation BIO (GBF4) SNES value (ha)":(
                    [
                        v - self._input_data.limits['GBF4_SNES'].sel(dict(layer=k)).item()
                        for k,v in prod_data["BIO (GBF4) SNES value (ha)"].items() 
                    ]                  
                    if settings.GBF4_TARGET_SNES != 'off'     
                    else 0
                ),
                "Deviation BIO (GBF4) ECNES value (ha)":(
                    [
                        v - self._input_data.limits['GBF4_ECNES'].sel(dict(layer=k)).item()
                        for k,v in prod_data["BIO (GBF4) ECNES value (ha)"].items()
                    ]
                    if settings.GBF4_TARGET_ECNES != 'off'    
                    else 0
                ),
                "Deviation BIO (GBF8) value (ha)":(
                    [
                        v - self._input_data.limits['GBF8'].sel(dict(layer=k)).item()
                        for k,v in prod_data["BIO (GBF8) value (ha)"].items()   
                    ]
                    if settings.GBF8_TARGET != "off"
                    else 0
                ),
            }
        )

