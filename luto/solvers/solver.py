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
from luto.solvers.row_builder import (extract_groups, extract_structure, attach_coeffs, compose_row,
                                      keep_terms, scale_rows)
from luto.solvers.input_data import SolverInputData, OBJ_BLOCKS
from luto.settings import AG_MANAGEMENTS


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


class LutoSolver:
    """The Gurobi model, its input data, and all builder-side bookkeeping.
    """

    def __init__(
        self,
        input_data: SolverInputData,
    ):

        self._input_data = input_data
        self.gurobi_model = gp.Model(f"LUTO {settings.VERSION}", env=gurenv)

        # --- decision-variable BLOCKS (one MVar per block; input_data holds the LONG table behind each) ---
        # For every block: the MVar, its global column offset (Var.index of element 0 — blocks are
        # created back to back, so offsets chain arithmetically, no model.update() needed), and the
        # column MAP from the model's natural indices to the block position (-1 = no variable).
        # ag decision vars (folded stream; the accounting view is ag_acct_table's term list)
        self.ag_mvar = None                 # over input_data.ag_var_table
        self.ag_offset = None
        self.ag_col = None                  # (NLMS, N_AG_LUS, ncells)
        # non-ag
        self.nonag_mvar = None              # over input_data.nonag_var_table
        self.nonag_offset = None
        self.nonag_col = None               # (N_NON_AG_LUS, ncells)
        # ag-management
        self.am_mvar = None                 # over input_data.am_var_table
        self.am_offset = None
        self.am_col = None                  # {am: (NLMS, len(am_j_list), ncells)}
        # transition-flow deltas — edge lists, addressed by table row (no map needed)
        self.a2a_mvar = None                # over input_data.flow_tables['a2a'] (ag → ag)
        self.a2a_offset = None
        self.a2n_mvar = None                # over input_data.flow_tables['a2n'] (ag → non-ag)
        self.a2n_offset = None
        self.n2a_mvar = None                # over input_data.flow_tables['n2a'] (non-ag → ag)
        self.n2a_offset = None

        # --- constraint handles ---
        self.cell_usage_constraint_r = {}       # {cell: Constr}
        self.ag_mgt_link_constraints_r = defaultdict(list)   # {cell: [Constr]}  X_am <= X_ag rows
        self.ag_mgt_adoption_constraints = []   # one per (am, j)
        self.regional_adoption_constraints = []
        self.demand_constraints = []            # one per commodity bound (eq, or lower + upper)
        self.water_limit_constraints = []       # one per water region
        self.renewable_constraints = {}         # {f'{am}_{state}': Constr}
        self.ghg_constr = None                  # the single GHG row
        self.bio_GBF2_constr = None             # the single GBF2 row
        self.bio_GBF3_NVIS_constrs = {}         # {(region, group): Constr}
        self.bio_GBF4_SNES_constrs = {}         # {(region, species, presence): Constr}
        self.bio_GBF4_ECNES_constrs = {}        # {(region, community, presence): Constr}
        self.bio_GBF8_constrs = {}              # {(region, species): Constr}
        # Row scales from row_builder.scale_rows (raw row = stored row x scale), per family:
        self.demand_scales = []                 # aligned with demand_constraints
        self.water_scales = []                  # aligned with water_limit_constraints / water_block rows
        self.ghg_scale = 1.0                    # the single GHG row
        self.renewable_scales = {}              # {f'{am}_{state}': scale}
        self.bio_GBF2_scale = 1.0               # the single GBF2 row
        self.bio_GBF3_NVIS_scales = {}          # {(region, group): scale}
        self.bio_GBF4_SNES_scales = {}          # {(region, species, presence): scale}
        self.bio_GBF4_ECNES_scales = {}         # {(region, community, presence): scale}
        self.bio_GBF8_scales = {}               # {(region, species): scale}

        # --- caches (each built ONCE per formulate, lazily, reused by every reader) ---
        self._bio_groups = None                 # _get_bio_groups(): bio term groups (extract_groups)
        self._policy_structure = None           # _get_policy_structure(): policy term structure (extract_structure)
        self._vars = None                       # _all_vars(): model.getVars() in Var.index order
        self._bio_index = {}                    # bio_constraint_index(): name -> {family, region, item, presence}, recorded as each family names its rows

        # --- constraint blocks (CSR over Var.index columns; row-scaled where a scale is kept:
        # raw row = stored row x scale) + per-row key lists; post-solve reporting = one
        # mat-vec over each block ---
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
        self.source_cap_block = None            # source-cap rows (CSR) + names
        self.source_cap_keys = []
        self.node_balance_block = None          # node-balance rows (CSR) + names
        self.node_balance_keys = []
        self.cell_usage_block = None            # cross-block row blocks (CSR)
        self.ag_mgt_link_block = None
        self.ag_mgt_adopt_block = None

    def formulate(self):
        """
        Performs the initial formulation of the model - setting up decision variables,
        constraints, and the objective.
        """
        print("Setting up the model...")
        self._setup_vars()
        self._setup_constraints()
        self._setup_objective()

    def _setup_vars(self):
        print("├── Setting up decision variables...")
        self._setup_ag_vars()
        self._setup_non_ag_vars()
        self._setup_ag_management_variables()
        self._setup_flow_vars()  

    def _setup_constraints(self):
        print("├── Adding the constraints...")
        self._add_renewable_capacity_ceiling_constraints()  # first: keeps the row order of the variable-setup era (Gurobi's path depends on it)
        self._add_cell_usage_constraints()
        self._add_agricultural_management_constraints()
        self._add_agricultural_management_adoption_limit_constraints()
        self._add_demand_constraints()
        self._add_ghg_emissions_limit_constraints()
        self._add_biodiversity_constraints()
        self._add_regional_adoption_constraints()
        self._add_water_usage_limit_constraints()
        self._add_renewable_energy_constraints()
        self._add_source_cap_constraints()                  # Σ out ≤ base
        self._add_node_balance_constraints()                # X = base + Σin − Σout

    def _setup_ag_vars(self):
        """The ag block: ONE addMVar over input_data.ag_var_table (one row per variable).

        Bounds come from the table (dvar_lb_ag/dvar_ub_ag, already cleaned in input_data:
        0 ≤ lb ≤ base ≤ ub); the node-balance/cap constant is the (cleaned, in-box) base
        dvar, so the all-delta=0 stay point is feasible by construction. Var.index = block
        offset + table row. `col_ag[m, j, r]` maps a variable to its position in the block
        (-1 = no variable).
        """
        print("│   ├── setting up decision variables for agricultural land uses...")
        tab = self._input_data.ag_var_table
        n   = tab['r'].size
        # global column offset of the block: the model is fresh here, so 0 — recorded rather than
        # assumed so later blocks can chain offsets arithmetically (no model.update() needed)
        self.ag_offset = self._input_data.var_layout['ag']
        assert self.ag_offset == 0, 'the ag block must be the first block created'
        self.ag_mvar = self.gurobi_model.addMVar(n, lb=tab['lb'], ub=tab['ub'], name="X_ag")
        ag_vars = self.ag_mvar.tolist()
        self.gurobi_model.setAttr(
            'VarName', ag_vars,
            [f"X_ag_{'dry' if m == 0 else 'irr'}_{j}_{r}" for m, j, r in zip(tab['m'], tab['j'], tab['r'])])
        self.ag_col = tab['col']

        self.const_ag = self._input_data.dvar_base_ag_mrj

    def _setup_non_ag_vars(self):
        """The non-ag block: ONE addMVar over input_data.nonag_var_table (one row per variable).

        Bounds (collapse rule applied) come from the table; Var.index = block offset + table
        row. `col_nonag[k, r]` maps a variable to its position in the block (-1 = no variable).
        The block's global column offset is the ag block's offset + size (blocks are created
        back to back; pending vars are not yet in NumVars, so offsets chain arithmetically).
        """
        print("│   ├── setting up decision variables for non-agricultural land uses...")
        tab = self._input_data.nonag_var_table
        n   = tab['r'].size
        self.nonag_offset = self._input_data.var_layout['nonag']
        self.nonag_mvar = self.gurobi_model.addMVar(n, lb=tab['lb'], ub=tab['ub'], name="X_non_ag")
        nonag_vars = self.nonag_mvar.tolist()
        self.gurobi_model.setAttr(
            'VarName', nonag_vars, [f"X_non_ag_{k}_{r}" for k, r in zip(tab['k'], tab['r'])])
        self.nonag_col = tab['col']
        self.const_nonag = self._input_data.dvar_base_non_ag_rk

    def _setup_ag_management_variables(self):
        """The ag-management block: ONE addMVar over input_data.am_var_table (one row per variable).

        Bounds and cell selection (GBF2 exclusion for renewables, savanna eligibility) come from
        the table; Var.index = block offset + table row. The block's global column offset
        chains from the non-ag block. `col_am[am][m, j_idx, r]` maps a variable to its position
        in the block (-1 = no variable).
        """
        print("│   ├── setting up decision variables for agricultural management options...")
        tab = self._input_data.am_var_table
        n   = tab['r'].size
        self.am_offset = self._input_data.var_layout['am']
        self.am_mvar = self.gurobi_model.addMVar(n, lb=tab['lb'], ub=tab['ub'], name="X_ag_man")
        am_vars = self.am_mvar.tolist()
        snake = [tools.am_name_snake_case(am) for am in tab['am_list']]
        self.gurobi_model.setAttr(
            'VarName', am_vars,
            [f"X_ag_man_{'dry' if m == 0 else 'irr'}_{snake[a]}_{j}_{r}".replace(" ", "_")
             for a, m, j, r in zip(tab['am'], tab['m'], tab['j'], tab['r'])])
        self.am_col = tab['col']

    def _setup_flow_vars(self):
        """The flow blocks: ONE addMVar per sub-block (a2a, a2n, n2a) over input_data.flow_tables.

        Rows are the edge-table order (source in dict order, argwhere C-order within a
        source); Var.index = block offset + row; names are generated from the columns.
        Offsets chain from the am block.
        """
        print("│   └── setting up transition flow delta variables (D)...")
        model = self.gurobi_model
        ft = self._input_data.flow_tables
        a2a, a2n, n2a = ft['a2a'], ft['a2n'], ft['n2a']

        lay = self._input_data.var_layout
        self.a2a_offset, self.a2n_offset, self.n2a_offset = lay['a2a'], lay['a2n'], lay['n2a']

        def _block(n, name, names):
            mvar = model.addMVar(n, lb=0.0, name=name)
            model.setAttr('VarName', mvar.tolist(), names)
            return mvar

        # ── ag → ag :  F[(fm,fj)][to_m, local_r, to_j], OFF-DIAGONAL only (positive-increment delta) ──
        # No stay/diagonal var: "staying" as (fm,fj) is free — the node-balance constant carries the base.
        self.a2a_mvar = _block(
            a2a['n'], "F_a2a",
            [f"F_a2a_{fm}_{fj}[{tm},{lr},{tj}]" for fm, fj, tm, lr, tj in
             zip(a2a['fm'], a2a['fj'], a2a['to_m'], a2a['local_r'], a2a['to_j'])])
        # ── ag → non-ag :  F[(fm,fj)][k, local_r] ──
        self.a2n_mvar = _block(
            a2n['n'], "F_a2n",
            [f"F_a2n_{fm}_{fj}[{k},{lr}]" for fm, fj, k, lr in
             zip(a2n['fm'], a2n['fj'], a2n['k'], a2n['local_r'])])
        # ── non-ag → ag :  F[k][to_m, local_r, to_j] ──
        self.n2a_mvar = _block(
            n2a['n'], "F_n2a",
            [f"F_n2a_{fk}[{tm},{lr},{tj}]" for fk, tm, lr, tj in
             zip(n2a['fk'], n2a['to_m'], n2a['local_r'], n2a['to_j'])])

        print(f"│       ├── ag2ag    : {a2a['n']:,} delta vars")
        print(f"│       ├── ag2nonag : {a2n['n']:,} delta vars")
        print(f"│       ├── nonag2ag : {n2a['n']:,} delta vars")
        print(f"│       └── total    : {a2a['n'] + a2n['n'] + n2a['n']:,} delta vars")

    def _all_vars(self):
        """The model's Var list in Var.index order (materialised once), for addMConstr."""
        if self._vars is None:
            self.gurobi_model.update()
            self._vars = self.gurobi_model.getVars()
        return self._vars

    def _get_bio_groups(self):
        """The bio term groups, extracted once per formulate and shared by GBF2/3/4/8."""
        if self._bio_groups is None:
            self._bio_groups = extract_groups(self._input_data)
        return self._bio_groups

    def _get_policy_structure(self):
        """The policy term structure, extracted once per formulate and shared by demand/water/GHG/renewables."""
        if self._policy_structure is None:
            self._policy_structure = extract_structure(self._input_data)
        return self._policy_structure

    def _add_cell_usage_constraints(self, cells: Optional[np.array] = None):
        """
        Constraint that all of every cell is used for some land use.
        If `cells` is provided, only adds constraints for the given cells.

        Build: one group-by over cell of the ag ∪ non-ag variable tables, coefficient 1. Each
        row is a RANGE (lo ≤ Σ X ≤ hi), written as Gurobi stores addRange rows: an equality
        Σ X + Rg = hi with a slack variable `Rg<name>` (lb 0, ub = hi − lo) per row, the slacks
        created as one MVar before the rows.
        """
        print("│   ├── Adding constraints for cell usage...")

        if cells is None:
            cells = np.array(range(self._input_data.ncells))
        cells = np.asarray(cells)

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

        # Ranged, not ==: presolve folds bal_a/bal_n into this row and demands
        # sum(base) == ag_mask between two constants summed along different float32 paths,
        # which disagree by up to ~1.75x FeasibilityTol (presolve reads constants exactly,
        # with no tolerance). The +/-10x Ftol band absorbs that residual; conservation
        # (bal_a/bal_n) still pins the cell total to sum(base), so the band is a pure
        # feasibility gate and its width is not exploitable by the objective.
        band = 10 * settings.FEASIBILITY_TOLERANCE
        row_cells = cells[~skip_r[cells]]                                 # cells that can meet the equality
        n_skipped = int(cells.size - row_cells.size)
        n_rows = row_cells.size
        ub = ag_mask[row_cells].astype(np.float64)                        # widen before the band is applied
        lo, hi = ub - band, ub + band

        # range slacks, exactly as addRange creates them (lb 0, ub = hi − lo, name Rg<row>)
        model = self.gurobi_model
        rg = model.addMVar(n_rows, lb=0.0, ub=hi - lo, name="Rg")
        rg_vars = rg.tolist()
        model.setAttr('VarName', rg_vars, [f"Rgconst_cell_usage_{r}" for r in row_cells])
        rg_offset = self._input_data.var_layout['n_dec']                  # first var after the decision blocks
        model.update()                       # the ONE update before the constraint families: Rg is the last
        x_all = model.getVars()              # addMVar, so this Var list is complete — every later addMConstr
        self._vars = x_all                   # reads it through _all_vars()
        assert len(x_all) == rg_offset + n_rows, 'range slacks must be the last variables created so far'

        # group-by cell: ag table rows + non-ag table rows landing on their cell's row
        row_of_cell = np.full(self._input_data.ncells, -1, dtype=np.int64)
        row_of_cell[row_cells] = np.arange(n_rows)
        ag_tab, nonag_tab = self._input_data.ag_var_table, self._input_data.nonag_var_table
        rows = np.concatenate([row_of_cell[ag_tab['r']], row_of_cell[nonag_tab['r']], np.arange(n_rows)])
        cols = np.concatenate([self.ag_offset + np.arange(ag_tab['r'].size),
                               self.nonag_offset + np.arange(nonag_tab['r'].size),
                               rg_offset + np.arange(n_rows)])
        keep = rows >= 0
        A = sparse.csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])),
                              shape=(n_rows, len(x_all)))
        constrs = model.addMConstr(A, x_all, '=', hi).tolist()
        model.setAttr('ConstrName', constrs, [f"const_cell_usage_{r}" for r in row_cells])
        self.cell_usage_constraint_r = dict(zip(row_cells.tolist(), constrs))
        self.cell_usage_block = A
        if n_skipped:
            print(f"│   │   WARNING: skipped cell-usage constraint for {n_skipped} cells "
                  f"(max feasible allocation < ag_mask).")

    def _add_agricultural_management_constraints(
        self, cells: Optional[np.array] = None
    ):
        """
        Constraint handling alternative agricultural management options:
        Ag. man. variables cannot exceed the value of the agricultural variable.

        Build: one row per (am, j, m, feasible cell) — the ag table rows of (m, j) joined with
        the am table through `col_am`. Where the am var exists the row is X_am − X_ag ≤ 0; where
        it does not (GBF2-excluded / savanna-ineligible cell) the row is X_ag ≥ 0 (sense '>'),
        via a per-row sense array.
        """
        print("│   ├── Adding constraints for agricultural management options...")
        model = self.gurobi_model
        x_all = self._all_vars()
        ag_tab = self._input_data.ag_var_table
        ag_pos = {}                                                   # (m, j) -> ag table rows
        key = ag_tab['m'].astype(np.int64) * ag_tab['col'].shape[1] + ag_tab['j']
        for kk in np.unique(key):
            ag_pos[divmod(int(kk), ag_tab['col'].shape[1])] = np.flatnonzero(key == kk)

        rows, cols, vals, senses, names, row_cells = [], [], [], [], [], []
        n = 0
        for am, am_j_list in self._input_data.am2j.items():
            col_am = self.am_col[am]
            for j_idx, j in enumerate(am_j_list):
                for m, lm in ((0, 'dry'), (1, 'irr')):
                    pos = ag_pos.get((m, j), np.array([], dtype=np.int64))
                    if cells is not None:
                        pos = pos[np.isin(ag_tab['r'][pos], cells)]
                    r = ag_tab['r'][pos]
                    am_col = col_am[m, j_idx, r]
                    has = am_col >= 0
                    ridx = n + np.arange(pos.size)
                    # X_ag term: −1 where the am var exists ('<' row), +1 where it does not ('>' row)
                    rows.append(ridx); cols.append(self.ag_offset + pos)
                    vals.append(np.where(has, -1.0, 1.0))
                    rows.append(ridx[has]); cols.append(self.am_offset + am_col[has])
                    vals.append(np.ones(int(has.sum())))
                    senses.append(np.where(has, '<', '>'))
                    names += [f"const_ag_mam_{lm}_usage_{am}_{j}_{rr}".replace(" ", "_") for rr in r]
                    row_cells.append(r)
                    n += pos.size
        A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n, len(x_all)))
        constrs = model.addMConstr(A, x_all, np.concatenate(senses), np.zeros(n)).tolist()
        model.setAttr('ConstrName', constrs, names)
        self.ag_mgt_link_constraints_r = defaultdict(list)
        for c, r in zip(constrs, np.concatenate(row_cells).tolist()):
            self.ag_mgt_link_constraints_r[r].append(c)
        self.ag_mgt_link_block = A

    def _add_agricultural_management_adoption_limit_constraints(self):
        """
        Add adoption limits constraints for agricultural management options.

        Build: one row per (am, j): Σ am vars − limit · Σ ag vars ≤ 0 (Σam ≤ limit · Σag with
        the RHS moved to the LHS); zero coefficients (limit = 0) are dropped.
        """
        print("│   ├── Adding constraints for agricultural management adoption limits...")
        model = self.gurobi_model
        x_all = self._all_vars()
        ag_tab, am_tab = self._input_data.ag_var_table, self._input_data.am_var_table
        rows, cols, vals, names = [], [], [], []
        n = 0
        for am_idx, (am, am_j_list) in enumerate(self._input_data.am2j.items()):
            for j_idx, j in enumerate(am_j_list):
                adoption_limit = float(np.float64(self._input_data.ag_man_limits[am][j]))
                am_pos = np.flatnonzero((am_tab['am'] == am_idx) & (am_tab['j_idx'] == j_idx))
                ag_pos = np.flatnonzero(ag_tab['j'] == j)                 # dry + irr feasible cells
                rows.append(np.full(am_pos.size, n)); cols.append(self.am_offset + am_pos)
                vals.append(np.ones(am_pos.size))
                rows.append(np.full(ag_pos.size, n)); cols.append(self.ag_offset + ag_pos)
                vals.append(np.full(ag_pos.size, -adoption_limit))
                names.append(f"const_ag_mam_adoption_limit_{am}_{j}".replace(" ", "_"))
                n += 1
        A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n, len(x_all)))
        A.eliminate_zeros()
        constrs = model.addMConstr(A, x_all, '<', np.zeros(n)).tolist()
        model.setAttr('ConstrName', constrs, names)
        self.ag_mgt_adoption_constraints = constrs
        self.ag_mgt_adopt_block = A

    def _add_renewable_capacity_ceiling_constraints(self):
        """Renewable ag-management options: simulated and existing capacity compete for the cell's
        space [0, ag_mask]. One row per (am, cell) where existing capacity is nonzero:

            Σ_{m, j} X_am[am, m, j, r]  ≤  max(ag_mask[r] − exist_r[r], 0)

        exist_r is the total across ALL data years (fixed), so the ceiling never decreases
        between periods — lb(t) <= ceiling(t-1) = ceiling(t) always holds. Built against the am
        block's own columns (a group-by (am, r) over the am table).
        """
        print("│   ├── Adding renewable capacity ceilings (existing + simulated ≤ cell space)...")
        tab = self._input_data.am_var_table
        n = tab['r'].size
        am_vars = self.am_mvar.tolist()
        ag_mask = self._input_data.ag_mask_proportion_r
        for am_idx, (am, am_j_list) in enumerate(self._input_data.am2j.items()):
            if am not in settings.RENEWABLES_OPTIONS:
                continue
            am_name = tools.am_name_snake_case(am)
            exist_r = (
                self._input_data.exist_renewable_solar_r
                if am == "Utility Solar PV"
                else self._input_data.exist_renewable_wind_r
            )
            pos = np.flatnonzero(tab['am'] == am_idx)
            r_u, inv = np.unique(tab['r'][pos], return_inverse=True)   # the am's cells, ascending
            cap = exist_r[r_u]
            keep_row = cap != 0                                        # no existing capacity -> no ceiling row
            row_of = np.full(r_u.size, -1, dtype=np.int64)
            row_of[keep_row] = np.arange(int(keep_row.sum()))
            rows = row_of[inv]; ok = rows >= 0
            A = sparse.csr_matrix((np.ones(int(ok.sum())), (rows[ok], pos[ok])),   # columns local to the am block
                                  shape=(int(keep_row.sum()), n))
            ceiling = np.maximum(ag_mask[r_u[keep_row]] - cap[keep_row], 0.0)   # cell space left for simulated capacity
            if A.shape[0]:
                constrs = self.gurobi_model.addMConstr(A, am_vars, '<', np.asarray(ceiling, dtype=np.float64)).tolist()
                self.gurobi_model.setAttr(
                    'ConstrName', constrs,
                    [f"const_{am_name}_solvable_ub_{r}".replace(" ", "_") for r in r_u[keep_row]])

    def _add_demand_constraints(self):
        """Hard demand constraints: per-commodity quantity rows composed from the shared
        policy structure; equality where the DEMAND_BOUNDS lb==ub, else the SAME LHS row
        twice under '>' lb and '<' ub.
        """
        print("│   ├── Adding constraints for demand ...")

        print("│   ├── Adding <hard> demand constraints (equality where lb==ub, else lower + upper)...")
        structure = self._get_policy_structure()
        nvars = len(self._all_vars())
        ones_r = np.ones(self._input_data.ncells, dtype=np.float32)

        # Attach per-commodity quantity coefficients to the shared structure: per entry,
        # jc[c, cell] = Σ_p pr2cm[c, p] · q[m, cell, p] over the land use's active products.
        ncms = self._input_data.ncms
        per_c_groups = [[] for _ in range(ncms)]
        for s in structure:
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

        q_rows = [compose_row(per_c_groups[c], ones_r, nvars)
                  for c in range(ncms)]

        rows, senses, rhs, names = [], [], [], []
        for c_idx, c_name in enumerate(self._input_data.commodity_names):
            lb, ub = settings.DEMAND_BOUNDS[c_name]
            lim = self._input_data.limits['demand'][c_idx]
            if lb == ub:
                rows.append(q_rows[c_idx]); senses.append('=')
                rhs.append(lim * lb); names.append(f"demand_hard_bound_eq[{c_idx}]")
            else:
                rows.append(q_rows[c_idx]); senses.append('>')
                rhs.append(lim * lb); names.append(f"demand_hard_bound_lower[{c_idx}]")
                rows.append(q_rows[c_idx]); senses.append('<')
                rhs.append(lim * ub); names.append(f"demand_hard_bound_upper[{c_idx}]")

        block, rhs, scale = scale_rows(sparse.vstack(rows, format='csr'), rhs)   # row rescale, factors kept
        constrs = self.gurobi_model.addMConstr(
            block, self._all_vars(), np.array(senses), rhs).tolist()
        self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
        self.demand_constraints.extend(constrs)
        self.demand_scales.extend(scale.tolist())
        self.demand_q_block = sparse.vstack(q_rows, format='csr')   # per-commodity LHS (reporting)

    def _add_ghg_emissions_limit_constraints(self):
        """Hard GHG emissions cap: one global row over land-use, ag-management, non-ag and
        transition-delta emissions, Σ ghg · X ≤ limit − offland."""
        if settings.GHG_EMISSIONS_LIMITS == "off":
            print("│   ├── TURNING OFF GHG emissions constraints ...")
            return

        ghg_limit_raw = self._input_data.limits["ghg"]

        print(f"│   ├── Adding <hard> constraints for GHG emissions: {ghg_limit_raw:,.0f} tCO2e")
        # GHG coefficients on the shared policy structure + the transition-delta stream,
        # composed with an all-ones val_row (one global row). The offland_ghg constant
        # sits in the RHS: Σ ghg · X ≤ limit − offland.
        groups = attach_coeffs(self._get_policy_structure(), self._input_data.ag_g_mrj,
                               self._input_data.ag_man_g_mrj, self._input_data.non_ag_g_rk)
        # ... plus the transition-delta stream: one group per ag source (float32 gather).
        # Delta groups carry cells=0 placeholders — valid only under the all-ones
        # val_row below (the GHG row is global).
        a2a = self._input_data.flow_tables['a2a']   # table gather
        for si, src in enumerate(a2a['sources']):
            a, b = int(a2a['src_ptr'][si]), int(a2a['src_ptr'][si + 1])
            if a == b:
                continue
            carr = self._input_data.flow_ghg_ag2ag[src]             # [to_m, local_r, to_j]
            coeffs = carr[a2a['to_m'][a:b], a2a['local_r'][a:b], a2a['to_j'][a:b]]
            var_idx = (self.a2a_offset + np.arange(a, b)).astype(np.int32)
            groups.append(dict(cells=np.zeros(b - a, dtype=np.int32),
                               var=var_idx, w=np.ones(b - a, dtype=np.float32), c=coeffs))
        ones_r = np.ones(self._input_data.ncells, dtype=np.float32)
        row = compose_row(groups, ones_r, len(self._all_vars()))
        # offland_ghg arrives as a 1-element 1-D array (OFF_LAND_GHG_EMISSION_C row);
        # ravel to a length-1 RHS vector for the single row
        rhs = np.asarray(ghg_limit_raw - self._input_data.offland_ghg,
                         dtype=np.float64).ravel()
        row, rhs, scale = scale_rows(row, rhs)                            # row rescale, factor kept
        self.ghg_scale = float(scale[0])
        constrs = self.gurobi_model.addMConstr(
            row, self._all_vars(), '<', rhs).tolist()
        self.gurobi_model.setAttr('ConstrName', constrs, ["ghg_emissions_limit_ub"])
        self.ghg_constr = constrs[0]
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

        # The LHS is the shared bio contribution operator applied to GBF2_mask_area_r, which
        # is ZERO off-mask (off-mask terms have coefficient 0 and are dropped by the sub-floor
        # test in compose_row). One row.
        groups = self._get_bio_groups()
        row = compose_row(groups,self._input_data.GBF2_mask_area_r,
                          len(self._all_vars()))
        row, rhs, scale = scale_rows(row, [self._input_data.limits["GBF2"]])     # row rescale, factor kept
        self.bio_GBF2_scale = float(scale[0])
        constrs = self.gurobi_model.addMConstr(
            row, self._all_vars(), '>', rhs).tolist()
        self.gurobi_model.setAttr(
            'ConstrName', constrs, ["bio_GBF2_priority_degraded_area_limit"])
        self._record_bio_names('GBF2', ["bio_GBF2_priority_degraded_area_limit"], [()])
        self.bio_GBF2_constr = constrs[0]
        self.bio_GBF2_block = row

    def _add_GBF3_NVIS_constraints(self) -> None:
        if settings.GBF3_NVIS_TARGET == "off":
            print("│   │   ├── TURNING OFF constraints for biodiversity GBF 3 NVIS")
            return

        print("│   │   ├── Adding constraints for biodiversity GBF 3 NVIS...")
        pairs      = self._input_data.GBF3_NVIS_region_group            # list[(region, group)]
        v_limits   = self._input_data.limits["GBF3_NVIS"]               # xr [layer=(region, group)]
        val_matrix = self._input_data.GBF3_NVIS_pre_1750_area_vr        # xr [group, cell]
        reg_matrix = self._input_data.region_NRM_names_r                # np [cell]

        groups = self._get_bio_groups()
        nvars = len(self._all_vars())

        # Compose one CSR row per active pair. NOTE the GBF3 skip is `lb_raw < 0`
        # (NOT <= 0): a ZERO target still adds a row.
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
            rows.append(compose_row(groups,val_row, nvars))
            names.append(f"bio_GBF3_NVIS_limit_{region}_{group}".replace(" ", "_"))
            rhs.append(lb_raw)
            kept.append((region, group))

        if rows:
            block, rhs, scale = scale_rows(sparse.vstack(rows, format='csr'), rhs)   # row rescale, factors kept
            constrs = self.gurobi_model.addMConstr(
                block, self._all_vars(), '>', rhs).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self._record_bio_names('GBF3_NVIS', names, kept)
            self.bio_GBF3_NVIS_constrs = dict(zip(kept, constrs))
            self.bio_GBF3_NVIS_block, self.bio_GBF3_NVIS_block_pairs = block, kept
            self.bio_GBF3_NVIS_scales = dict(zip(kept, scale))
        print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")

    def _add_GBF4_SNES_constraints(self) -> None:
        if settings.GBF4_TARGET_SNES == 'off':
            print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 SNES...')
            return

        print("│   │   ├── Adding constraints for biodiversity GBF 4 SNES ...")
        pairs      = self._input_data.GBF4_SNES_region_species          # list[(region, species, presence)]
        v_limits   = self._input_data.limits["GBF4_SNES"]               # xr [layer=(region, species, presence)]
        val_matrix = self._input_data.GBF4_SNES_pre_1750_area_sr        # xr [layer=(species, presence), cell]
        reg_matrix = self._input_data.region_NRM_names_r                # np [cell]

        groups = self._get_bio_groups()
        nvars = len(self._all_vars())

        # Compose one CSR row per active key; skipped when the raw target is <= 0 or the
        # (region-masked) layer has no positive cell.
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
            rows.append(compose_row(groups,val_row, nvars))
            names.append(f"bio_GBF4_SNES_limit_{region}_{species}_{presence}".replace(" ", "_"))
            rhs.append(lb_raw)
            kept.append((region, species, presence))

        if rows:
            block, rhs, scale = scale_rows(sparse.vstack(rows, format='csr'), rhs)   # row rescale, factors kept
            constrs = self.gurobi_model.addMConstr(
                block, self._all_vars(), '>', rhs).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self._record_bio_names('GBF4_SNES', names, kept)
            self.bio_GBF4_SNES_constrs = dict(zip(kept, constrs))
            self.bio_GBF4_SNES_block, self.bio_GBF4_SNES_block_pairs = block, kept
            self.bio_GBF4_SNES_scales = dict(zip(kept, scale))
        print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")

    def _add_GBF4_ECNES_constraints(self) -> None:
        if settings.GBF4_TARGET_ECNES == 'off':
            print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 ECNES...')
            return

        print("│   │   ├── Adding constraints for biodiversity GBF 4 ECNES ...")
        pairs      = self._input_data.GBF4_ECNES_region_species         # list[(region, community, presence)]
        v_limits   = self._input_data.limits["GBF4_ECNES"]              # xr [layer=(region, community, presence)]
        val_matrix = self._input_data.GBF4_ECNES_pre_1750_area_sr       # xr [layer=(community, presence), cell]
        reg_matrix = self._input_data.region_NRM_names_r                # np [cell]

        groups = self._get_bio_groups()
        nvars = len(self._all_vars())

        # Compose one CSR row per active key; skipped when the raw target is <= 0 or the
        # (region-masked) layer has no positive cell.
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
            rows.append(compose_row(groups,val_row, nvars))
            names.append(f"bio_GBF4_ECNES_limit_{region}_{community}_{presence}".replace(" ", "_"))
            rhs.append(lb_raw)
            kept.append((region, community, presence))

        if rows:
            block, rhs, scale = scale_rows(sparse.vstack(rows, format='csr'), rhs)   # row rescale, factors kept
            constrs = self.gurobi_model.addMConstr(
                block, self._all_vars(), '>', rhs).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self._record_bio_names('GBF4_ECNES', names, kept)
            self.bio_GBF4_ECNES_constrs = dict(zip(kept, constrs))
            self.bio_GBF4_ECNES_block, self.bio_GBF4_ECNES_block_pairs = block, kept
            self.bio_GBF4_ECNES_scales = dict(zip(kept, scale))
        print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")

    def _add_GBF8_constraints(self) -> None:

        if settings.GBF8_TARGET == "off":
            print('│   │   ├── TURNING OFF constraints for biodiversity GBF 8 ...')
            return

        print("│   │   ├── Adding constraints for biodiversity GBF 8 ...")
        pairs      = self._input_data.GBF8_region_species               # list[(region, species)]
        v_limits   = self._input_data.limits["GBF8"]                    # xr [layer=(region, species)]
        val_matrix = self._input_data.GBF8_pre_1750_area_sr             # xr [species, cell]
        reg_matrix = self._input_data.region_NRM_names_r                # np [cell]

        groups = self._get_bio_groups()
        nvars = len(self._all_vars())

        # Compose one CSR row per active pair; skipped when the raw target is <= 0 or the
        # (region-masked) species layer has no positive cell.
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
            rows.append(compose_row(groups,val_row, nvars))
            names.append(f"bio_GBF8_limit_{region}_{species}".replace(" ", "_"))
            rhs.append(lb_raw)
            kept.append((region, species))

        if rows:
            block, rhs, scale = scale_rows(sparse.vstack(rows, format='csr'), rhs)   # row rescale, factors kept
            constrs = self.gurobi_model.addMConstr(
                block, self._all_vars(), '>', rhs).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self._record_bio_names('GBF8', names, kept)
            self.bio_GBF8_constrs = dict(zip(kept, constrs))
            self.bio_GBF8_block, self.bio_GBF8_block_pairs = block, kept
            self.bio_GBF8_scales = dict(zip(kept, scale))
        print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")

    def _record_bio_names(self, family: str, names: list[str], keys: list[tuple]) -> None:
        """Record name -> {family, region, item, presence} for rows just named (no Gurobi reads).

        The constraint NAME is all a Gurobi model carries, and it cannot be parsed back into its
        parts: `.replace(" ", "_")` makes the separator ambiguous ("Goulburn_Broken" vs the
        underscore before the community), and the arity differs by family — SNES/ECNES have a
        presence class, GBF3/GBF8 do not, GBF2 has no key at all. So each family records the
        mapping here, at its setAttr site, while the tuple is still known.
        """
        for name, key in zip(names, keys):
            row = {'family': family, 'region': None, 'item': None, 'presence': None}
            row.update(dict(zip(('region', 'item', 'presence'), key)))
            self._bio_index[name] = row

    def bio_constraint_index(self) -> dict:
        """{constraint_name: {family, region, item, presence}} for every biodiversity row built.

        Filled by the families as they name their rows (`_record_bio_names`), never read back from
        the model — so it needs no `update()`, and it still describes rows that
        `remove_constraints_by_name` has since dropped.
        """
        return self._bio_index

    def _add_regional_adoption_constraints(self) -> None:
        """Regional adoption caps as one addMConstr per family (ag / non-ag / non-ag sum).

        Each row: Σ real_area[r] · X over the region's cells ≤ cap. Coefficients gathered from
        the variable tables; the |coeff| ≥ SOLVER_COEFF_MIN drop kept for fidelity (a no-op on
        hectares). Non-ag caps carry the per-year relaxation (see below).
        """
        if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
            print("│   │   └── TURNING OFF constraints for regional adoption ...")
            return
        model = self.gurobi_model
        x_all = self._all_vars()
        real_area = self._input_data.real_area
        ag_tab, nonag_tab = self._input_data.ag_var_table, self._input_data.nonag_var_table

        def _add_family(rows, names, rhs):
            if not rows:
                return
            r_idx = np.concatenate([np.full(len(c), i) for i, (c, _) in enumerate(rows)])
            cols = np.concatenate([c for c, _ in rows]); vals = np.concatenate([v for _, v in rows])
            A = sparse.csr_matrix((vals, (r_idx, cols)), shape=(len(rows), len(x_all)))
            constrs = model.addMConstr(A, x_all, '<', np.asarray(rhs, dtype=np.float64)).tolist()
            model.setAttr('ConstrName', constrs, names)
            self.regional_adoption_constraints += constrs

        def _terms(tab, offset, sel):
            pos = np.flatnonzero(sel)
            return keep_terms(offset + pos, real_area[tab['r'][pos]].astype(np.float32))

        # ag land uses
        rows, names, rhs = [], [], []
        for reg_id, j, lu_name, reg_ind, reg_area_limit in self._input_data.limits["ag_regional_adoption"]:
            if len(reg_ind) == 0:
                print(f"│   │   │   ├── SKIPPING {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} (no cells at this resolution)")
                continue
            print(f"│   │   │   ├── Adding constraints for {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} <= {reg_area_limit:,.0f} HA...")
            rows.append(_terms(ag_tab, self.ag_offset, (ag_tab['j'] == j) & np.isin(ag_tab['r'], reg_ind)))
            names.append(f"reg_adopt_limit_ag_{lu_name}_{reg_id}".replace(" ", "_")); rhs.append(reg_area_limit)
        _add_family(rows, names, rhs)

        # Non-reversible plantings saturate the non-ag caps below, and last year's solved
        # areas become this year's exact lower bounds; float32 noise then puts the locked-in
        # floor a hair over the cap, which presolve rejects with NO tolerance (bound
        # propagation is exact). Grow the cap by 1e-6/yr RELATIVE so the RHS always recedes
        # ahead of the ratcheting floor (per-step increment ~5e-6 x cap vs float noise
        # ~2e-10 x cap). Cap erosion by 2050: ~3e-5 relative. Ag caps need no slack: ag is
        # reversible, so its floors never ratchet onto the cap.
        nonag_cap_relax = 1 + (self._input_data.target_year - settings.SIM_YEARS[0]) * 1e-6

        # per-(region, non-ag land use) caps ('on' mode)
        rows, names, rhs = [], [], []
        for reg_id, k, lu_name, reg_ind, reg_area_limit in (self._input_data.limits.get("non_ag_regional_adoption") or []):
            if len(reg_ind) == 0:
                print(f"│   │   │   ├── SKIPPING {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} (no cells at this resolution)")
                continue
            print(f"│   │   │   ├── Adding constraints for {lu_name} in {settings.REGIONAL_ADOPTION_ZONE} region {reg_id} <= {reg_area_limit:,.0f} HA...")
            rows.append(_terms(nonag_tab, self.nonag_offset, (nonag_tab['k'] == k) & np.isin(nonag_tab['r'], reg_ind)))
            names.append(f"reg_adopt_limit_non_ag_{lu_name}_{reg_id}".replace(" ", "_")); rhs.append(reg_area_limit * nonag_cap_relax)
        _add_family(rows, names, rhs)

        # SUM-of-non-ag caps ('NON_AG_CAP' mode): all non-ag land uses in a region together
        rows, names, rhs = [], [], []
        for reg_id, reg_ind, reg_area_limit in (self._input_data.limits.get("non_ag_regional_adoption_sum") or []):
            if len(reg_ind) == 0:
                print(f"│   │   │   ├── SKIPPING SUM-of-non-ag constraint for {settings.REGIONAL_ADOPTION_NON_AG_REGION} region {reg_id} (no cells at this resolution)")
                continue
            print(f"│   │   │   ├── Adding SUM-of-non-ag constraint for {settings.REGIONAL_ADOPTION_NON_AG_REGION} region {reg_id} <= {reg_area_limit:,.0f} HA...")
            rows.append(_terms(nonag_tab, self.nonag_offset, np.isin(nonag_tab['r'], reg_ind)))
            names.append(f"reg_adopt_limit_non_ag_sum_{reg_id}".replace(" ", "_")); rhs.append(reg_area_limit * nonag_cap_relax)
        _add_family(rows, names, rhs)

    def _add_water_usage_limit_constraints(self) -> None:

        if settings.WATER_LIMITS != "on":
            print("│   ├── TURNING OFF water usage constraints ...")
            return

        print("│   ├── Adding constraints for water usage limits...")


        # Water net-yield coefficients on the shared policy structure (they can be NEGATIVE; the
        # sub-floor drop in compose_row tests |q|). Each region's row = compose with that
        # region's 0/1 float32 indicator as the val_row: off-region terms give q = 0 and are
        # dropped, and 1.0f x c == c, so the drop test sees the raw coefficient.
        groups = attach_coeffs(self._get_policy_structure(), self._input_data.ag_w_mrj,
                               self._input_data.ag_man_w_mrj, self._input_data.non_ag_w_rk)
        nvars = len(self._all_vars())

        rows, names, rhs, regids = [], [], [], []
        for reg_idx, w_limit_raw in self._input_data.limits["water"].items():
            ind = self._input_data.water_region_indices[reg_idx]
            reg_name = self._input_data.water_region_names[reg_idx]
            print(f"│   │   ├── target (inside LUTO study area) is {w_limit_raw:15,.0f} ML for {reg_name}")
            indicator = np.zeros(self._input_data.ncells, dtype=np.float32)
            indicator[ind] = 1.0
            rows.append(compose_row(groups, indicator, nvars))
            names.append(f"water_yield_limit_{reg_name}".replace(" ", "_"))
            rhs.append(w_limit_raw)
            regids.append(reg_idx)

        if rows:
            block, rhs, scale = scale_rows(sparse.vstack(rows, format='csr'), rhs)   # row rescale, factors kept
            constrs = self.gurobi_model.addMConstr(
                block, self._all_vars(), '>', rhs).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self.water_limit_constraints.extend(constrs)
            self.water_block, self.water_block_regids = block, regids
            self.water_scales.extend(scale.tolist())

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

        # Each (state, type) row = the type's am-var structure entries with energy_r
        # coefficients, composed against an allowed-cells indicator (state region, ACT merged
        # into NSW, minus the per-type GBF2/MNES exclusion masks). RHS = target − existing
        # capacity. Row inclusion is a CELL-SET rule: a row exists iff at least one compatible
        # land use has eligible cells — even if every coefficient there is sub-floor.
        structure = self._get_policy_structure()
        nvars = len(self._all_vars())

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

                exist_power_mwh     = self._input_data.limits[f"renewable_{am}_exist"][reg_name]

                print(f"│   │   │   ├── target for {am} is {target_raw:5,.0f} MWh  (existing: {exist_power_mwh:5,.0f} MWh)")

                # Cell-set row-inclusion rule (NOT a coefficient test)
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
                          for s in structure
                          if s['kind'] == 'am' and s['am'] == am]
                rows.append(compose_row(groups, allowed, nvars))
                names.append(f"renewable_{am}_target_{reg_name}".replace(" ", "_"))
                rhs.append(target_raw - exist_power_mwh)                # raw MWh; row-rescaled below
                keys.append(f'{am}_{reg_name}')

        if rows:
            block, rhs, scale = scale_rows(sparse.vstack(rows, format='csr'), rhs)   # row rescale, factors kept
            constrs = self.gurobi_model.addMConstr(
                block, self._all_vars(), '>', rhs).tolist()
            self.gurobi_model.setAttr('ConstrName', constrs, names)     # bulk naming, one call
            self.renewable_constraints = dict(zip(keys, constrs))
            self.renewable_scales = dict(zip(keys, scale.tolist()))

    def _add_source_cap_constraints(self):
        """Source cap: a source cannot export more land than it holds.

            Σ_out D[src]  ≤  x_old[src]

        This BOUNDS the delta vars (some `flow_cost` entries are negative, so without it the
        objective could push a `D` to +∞ around a negative-cost cycle) and rules out
        "pass-through" (a source re-exporting land it imported). With the node-balance equality
        it gives an exact, bounded min-cost transition flow. RHS = `const` (base clipped into the
        [lb, ub] box) — the same quantity node-balance uses.

        ag source (fm,fj) at cell r:  Σ_to D_ag2ag[(fm,fj)][·,r,·] + Σ_k D_ag2nonag[(fm,fj)][k,r] ≤ const_ag[fm,r,fj]
        non-ag source k at cell r:    Σ_to D_nonag2ag[k][·,r,·]                                    ≤ const_nonag[r,k]

        Build: ONE group-by over (source, cell) of the a2a ∪ a2n edge tables —
        `np.unique(key, return_inverse)` gives the rows (source-major, then local_r; a
        (source, cell) with no arcs has no row) and the inverse is the row index of every arc.
        Coefficients exactly 1; one addMConstr per source family (ag, then non-ag).
        """
        print("│   ├── Adding source-cap (Σ out ≤ base) constraints...")
        model     = self.gurobi_model
        const_ag  = self.const_ag
        const_non = self.const_nonag
        ft        = self._input_data.flow_tables
        a2a, a2n, n2a = ft['a2a'], ft['a2n'], ft['n2a']
        stride    = self._input_data.ncells                        # local_r < ncells
        x_all     = self._all_vars()
        nvars     = len(x_all)
        assert a2a['sources'] == a2n['sources'], 'a2a / a2n must share the ag source order'

        # ag sources: rows keyed by (src, local_r) over a2a ∪ a2n
        key  = np.concatenate([a2a['src'].astype(np.int64) * stride + a2a['local_r'],
                               a2n['src'].astype(np.int64) * stride + a2n['local_r']])
        col  = np.concatenate([self.a2a_offset + np.arange(a2a['n'], dtype=np.int64),
                               self.a2n_offset + np.arange(a2n['n'], dtype=np.int64)])
        fm_a = np.concatenate([a2a['fm'], a2n['fm']]); fj_a = np.concatenate([a2a['fj'], a2n['fj']])
        r_a  = np.concatenate([a2a['r'], a2n['r']]);   lr_a = np.concatenate([a2a['local_r'], a2n['local_r']])
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        A_ag = sparse.csr_matrix((np.ones(key.size), (inv, col)), shape=(uniq.size, nvars))
        rhs_ag = const_ag[fm_a[first], r_a[first], fj_a[first]].astype(np.float64)
        names_ag = [f"srccap_a_{fm}_{fj}_{lr}" for fm, fj, lr in zip(fm_a[first], fj_a[first], lr_a[first])]
        c_ag = model.addMConstr(A_ag, x_all, '<', rhs_ag).tolist()
        model.setAttr('ConstrName', c_ag, names_ag)

        # non-ag sources: rows keyed by (src, local_r) over n2a
        key  = n2a['src'].astype(np.int64) * stride + n2a['local_r']
        col  = self.n2a_offset + np.arange(n2a['n'], dtype=np.int64)
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        A_non = sparse.csr_matrix((np.ones(key.size), (inv, col)), shape=(uniq.size, nvars))
        rhs_non = const_non[n2a['r'][first], n2a['fk'][first]].astype(np.float64)
        names_non = [f"srccap_n_{fk}_{lr}" for fk, lr in zip(n2a['fk'][first], n2a['local_r'][first])]
        c_non = model.addMConstr(A_non, x_all, '<', rhs_non).tolist() if uniq.size else []
        if c_non:
            model.setAttr('ConstrName', c_non, names_non)

        self.source_cap_block = sparse.vstack([A_ag, A_non]).tocsr() if uniq.size else A_ag
        self.source_cap_keys  = names_ag + names_non
        print(f"│   │   └── added {len(c_ag) + len(c_non):,} source-cap constraints")

    def _add_node_balance_constraints(self):
        """Node-balance equality: each LU's final area = base + inflows − outflows.

            X_ag[m,r,j]  = const_ag[m,r,j]  + Σ_in D[·→(m,j)] − Σ_out D[(m,j)→·]
            X_nonag[r,k] = const_nonag[r,k] + Σ_in D_ag2nonag[·→k] − Σ_out D_nonag2ag[k→·]

        This ties every delta to real per-LU land movement (a negative-cost arc cannot be used
        without moving land) and, with the source cap, gives an exact, bounded min-cost
        transition flow. "Staying" is the all-D=0 solution (X = const). `const` is the base
        clipped into the var's [lb, ub] box, so the stay point is feasible by construction.
        No non-ag→non-ag term exists.

        Build: one row per ag var (the ag table row order) then one per (non-ag land use,
        feasible cell) — every non-ag land use, enabled or not. Inflow arcs land on their
        TARGET's row via `col_ag` / `col_nonag` (−1), outflow arcs on their SOURCE's row (+1),
        X on its own row (+1): X − Σin + Σout = const. Rows of a land use with NO X var
        (disabled non-ag) carry the opposite sign (see `row_sign`). A source with no X var
        (banned dominant) has no row, so its outflow arcs are dropped.
        """
        print("│   └── Adding node-balance (X = base + Σin − Σout) constraints...")
        model     = self.gurobi_model
        const_ag  = self.const_ag
        const_non = self.const_nonag
        ft        = self._input_data.flow_tables
        a2a, a2n, n2a = ft['a2a'], ft['a2n'], ft['n2a']
        ag_tab = self._input_data.ag_var_table
        col_ag, col_nonag = self.ag_col, self.nonag_col
        n_ag = ag_tab['r'].size
        x_all     = self._all_vars()
        nvars     = len(x_all)

        # non-ag rows: one per (k, feasible cell) for EVERY non-ag land use k — including
        # disabled ones that own no X var: their rows are pure inflow guards,
        # −Σin + Σout = const, with no X column.
        feas = self._input_data.feasible_non_ag_cells
        k_b = np.concatenate([np.full(len(feas[k]), k, dtype=np.int64) for k in range(self._input_data.n_non_ag_lus)])
        r_b = np.concatenate([np.asarray(feas[k], dtype=np.int64) for k in range(self._input_data.n_non_ag_lus)])
        n_non = r_b.size
        row_nonag = np.full(col_nonag.shape, -1, dtype=np.int64)          # (k, r) -> balance row
        row_nonag[k_b, r_b] = n_ag + np.arange(n_non)

        # Row sign: a row WITHOUT an X var (disabled non-ag land use) is stored as
        # Σin − Σout = −const (+1 inflow, −1 outflow, RHS −const); rows with an X var are
        # X − Σin + Σout = const.
        x_col = col_nonag[k_b, r_b]
        row_sign = np.ones(n_ag + n_non, dtype=np.float64)
        row_sign[n_ag:] = np.where(x_col >= 0, 1.0, -1.0)

        rows, cols, vals = [], [], []

        def add(row, col, v):
            keep = row >= 0
            rows.append(row[keep].astype(np.int64)); cols.append(col[keep].astype(np.int64))
            vals.append(v * row_sign[row[keep]])

        # X terms: +1 on its own row (non-ag: only where the land use owns a var)
        add(np.arange(n_ag), self.ag_offset + np.arange(n_ag), 1.0)
        add(np.where(x_col >= 0, n_ag + np.arange(n_non), -1), self.nonag_offset + x_col, 1.0)
        # ag rows: inflows (a2a, n2a) −1 on the target's row; outflows (a2a, a2n) +1 on the source's row
        a2a_col = self.a2a_offset + np.arange(a2a['n']); a2n_col = self.a2n_offset + np.arange(a2n['n'])
        n2a_col = self.n2a_offset + np.arange(n2a['n'])
        add(col_ag[a2a['to_m'], a2a['to_j'], a2a['r']], a2a_col, -1.0)
        add(col_ag[n2a['to_m'], n2a['to_j'], n2a['r']], n2a_col, -1.0)
        add(col_ag[a2a['fm'], a2a['fj'], a2a['r']], a2a_col, 1.0)
        add(col_ag[a2n['fm'], a2n['fj'], a2n['r']], a2n_col, 1.0)
        # non-ag rows: inflows (a2n) −1; outflows (n2a) +1
        add(row_nonag[a2n['k'], a2n['r']], a2n_col, -1.0)
        add(row_nonag[n2a['fk'], n2a['r']], n2a_col, 1.0)

        A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n_ag + n_non, nvars))
        rhs = np.concatenate([const_ag[ag_tab['m'], ag_tab['r'], ag_tab['j']].astype(np.float64),
                              const_non[r_b, k_b].astype(np.float64) * row_sign[n_ag:]])
        names = ([f"bal_a_{m}_{j}_{r}" for m, j, r in zip(ag_tab['m'], ag_tab['j'], ag_tab['r'])]
                 + [f"bal_n_{k}_{r}" for k, r in zip(k_b, r_b)])
        constrs = model.addMConstr(A, x_all, '=', rhs).tolist()
        model.setAttr('ConstrName', constrs, names)
        self.node_balance_block = A
        self.node_balance_keys  = names
        print(f"│       └── added {len(constrs):,} node-balance constraints")

    def _setup_objective(self):
        """Objective obj · X: the economy block summed over its component rows, scaled to
        million AUD, then floored at SOLVER_COEFF_MIN — the last step of the coefficient
        contract (see ``row_builder.keep_terms``), applied here because the scaling can push a
        merged coefficient under the floor."""
        print(f"├── Setting up the objective function to {settings.OBJECTIVE}...")
        obj = np.asarray(self._input_data.obj_block.sum(axis=0)).ravel()        # (5 x n_dec) float32 block; disjoint vars per row
        obj = obj * (1.0 / 1e6)                                                 # raw AUD -> million AUD (reciprocal multiply, as gurobipy did)
        obj[np.abs(obj) < settings.SOLVER_COEFF_MIN] = 0.0                      # floor the merged, scaled coefficient
        self.obj_vec = obj

        X = gp.MVar.fromlist(self._all_vars()[:obj.size])                  # the decision vars (Var.index order); the range slacks after them carry no objective
        if settings.OBJECTIVE == "mincost":
            self.gurobi_model.setObjective(obj @ X, GRB.MINIMIZE)
        elif settings.OBJECTIVE == "maxprofit":
            self.gurobi_model.setObjective(obj @ X, GRB.MAXIMIZE)
        else:
            raise ValueError(f"Unknown objective: {settings.OBJECTIVE}")
        
        print(f"│   └── objective: {int((obj != 0).sum()):,} nonzero coefficients over {obj.size:,} variables")

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

        `bio_constraint_index()` is deliberately NOT invalidated — it was recorded at build time
        precisely so that dropped rows can be described afterwards.
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
        self.regional_adoption_constraints  = [c for c in self.regional_adoption_constraints if c.ConstrName not in doomed]
        self.demand_constraints = [c for c in self.demand_constraints if c.ConstrName not in doomed]

        if self.bio_GBF2_constr is not None and self.bio_GBF2_constr.ConstrName in doomed:
            self.bio_GBF2_constr = None
        if self.ghg_constr is not None and self.ghg_constr.ConstrName in doomed:
            self.ghg_constr = None

        self.gurobi_model.remove(
            [c for c in self.gurobi_model.getConstrs() if c.ConstrName in doomed])
        self.gurobi_model.update()

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

        # Agricultural results: ONE .X read of the ag block, scattered through the table
        # (float64 -> float32)
        tab = self._input_data.ag_var_table
        x_ag = np.asarray(self.ag_mvar.X, dtype=np.float64)
        dry = tab['m'] == 0
        X_dry_sol_rj[tab['r'][dry],  tab['j'][dry]]  = x_ag[dry]
        X_irr_sol_rj[tab['r'][~dry], tab['j'][~dry]] = x_ag[~dry]

        # Get non-agricultural results: ONE .X read of the block, scattered through the table
        # (disabled land uses have no rows and stay at the array's zeros)
        tab = self._input_data.nonag_var_table
        non_ag_X_sol_rk[tab['r'], tab['k']] = np.asarray(self.nonag_mvar.X, dtype=np.float64)

        # Ag-management results: ONE .X read of the block, scattered through the table.
        # Savanna eligibility is applied to BOTH lm here, while variable creation applied it
        # to dry only (get_am_var_table): irr savanna vars outside savanna_eligible_r are
        # reported as 0.
        tab = self._input_data.am_var_table
        x_am = np.asarray(self.am_mvar.X, dtype=np.float64)
        keep = np.ones(x_am.size, dtype=bool)
        if "Savanna Burning" in tab['am_list']:
            sav = tab['am_list'].index("Savanna Burning")
            keep &= ~((tab['am'] == sav) & (tab['m'] == 1)
                      & ~np.isin(tab['r'], self._input_data.savanna_eligible_r))
        for am_idx, am in enumerate(tab['am_list']):
            d = keep & (tab['am'] == am_idx) & (tab['m'] == 0)
            i = keep & (tab['am'] == am_idx) & (tab['m'] == 1)
            am_X_dry_sol_rj[am][tab['r'][d], tab['j'][d]] = x_am[d]
            am_X_irr_sol_rj[am][tab['r'][i], tab['j'][i]] = x_am[i]

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
        # one .X read per block, scattered per source slice through the edge tables
        ft = self._input_data.flow_tables
        a2a, a2n, n2a = ft['a2a'], ft['a2n'], ft['n2a']
        x_a2a = np.asarray(self.a2a_mvar.X, dtype=np.float32) if a2a['n'] else np.zeros(0, np.float32)
        x_a2n = np.asarray(self.a2n_mvar.X, dtype=np.float32) if a2n['n'] else np.zeros(0, np.float32)
        x_n2a = np.asarray(self.n2a_mvar.X, dtype=np.float32) if n2a['n'] else np.zeros(0, np.float32)
        for si, ((fm, fj), cells) in enumerate(self._input_data.ag_source_cells.items()):
            arr = np.zeros((self._input_data.nlms, len(cells), self._input_data.n_ag_lus), dtype=np.float32)
            a, b = int(a2a['src_ptr'][si]), int(a2a['src_ptr'][si + 1])
            arr[a2a['to_m'][a:b], a2a['local_r'][a:b], a2a['to_j'][a:b]] = x_a2a[a:b]
            dvar_D_ag2ag_mrj[(fm, fj)] = arr

            arr = np.zeros((len(cells), self._input_data.n_non_ag_lus), dtype=np.float32)
            a, b = int(a2n['src_ptr'][si]), int(a2n['src_ptr'][si + 1])
            arr[a2n['local_r'][a:b], a2n['k'][a:b]] = x_a2n[a:b]
            dvar_D_ag2nonag_rk[(fm, fj)] = arr
        for si, (fk, cells) in enumerate(self._input_data.nonag_source_cells.items()):
            arr = np.zeros((self._input_data.nlms, len(cells), self._input_data.n_ag_lus), dtype=np.float32)
            a, b = int(n2a['src_ptr'][si]), int(n2a['src_ptr'][si + 1])
            arr[n2a['to_m'][a:b], n2a['local_r'][a:b], n2a['to_j'][a:b]] = x_n2a[a:b]
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

        # Ag-management maps (one per option; options can stack): 1 where the cell's chosen
        # (lm, lu) carries the option at or above AGRICULTURAL_MANAGEMENT_USE_THRESHOLD.
        ammaps = {am: np.zeros(self._input_data.ncells, dtype=np.int8) for am in AG_MANAGEMENTS}
        r_ag = np.flatnonzero(lumap < settings.NON_AGRICULTURAL_LU_BASE_CODE)   # non-ag cells carry no option
        j_ag = lumap[r_ag].astype(np.int64)
        m_ag = lmmap[r_ag].astype(np.int64)
        for am, am_j_list in self._input_data.am2j.items():
            val = ag_man_X_mrj[am][m_ag, r_ag, j_ag]
            on = (val >= settings.AGRICULTURAL_MANAGEMENT_USE_THRESHOLD) & np.isin(j_ag, am_j_list)
            ammaps[am][r_ag[on]] = 1

        # ── Constraint LHS values at the solution: one mat-vec per stored block (report only) ──
        x_vals = np.asarray(self.gurobi_model.getAttr('X', self.gurobi_model.getVars()), dtype=np.float64)
        limits = self._input_data.limits

        obj_block = self._input_data.obj_block
        econ = dict(zip(OBJ_BLOCKS, obj_block @ x_vals[:obj_block.shape[1]]))   # per-block economy, raw AUD

        prod_data["Production"] = (
            self._block_lhs(self.demand_q_block, x_vals).tolist()                  # raw t
            if self.demand_q_block is not None else 0
        )
        prod_data["GHG"] = (
            float(self._block_lhs(self.ghg_block, x_vals, [self.ghg_scale])[0])
            + float(np.asarray(self._input_data.offland_ghg).ravel()[0])          # the block excludes the offland constant
            if self.ghg_block is not None else 0
        )
        prod_data["Water"] = (
            dict(zip(self.water_block_regids, self._block_lhs(self.water_block, x_vals, self.water_scales).tolist()))
            if self.water_block is not None else 0
        )
        prod_data["BIO (GBF2) value (ha)"] = (
            float(self._block_lhs(self.bio_GBF2_block, x_vals, [self.bio_GBF2_scale])[0])
            if self.bio_GBF2_block is not None else 0
        )
        prod_data["BIO (GBF3) NVIS value (ha)"] = (
            self._block_values(self.bio_GBF3_NVIS_block, self.bio_GBF3_NVIS_block_pairs, self.bio_GBF3_NVIS_scales, x_vals)
            if settings.GBF3_NVIS_TARGET != 'off' else 0
        )
        prod_data["BIO (GBF4) SNES value (ha)"] = (
            self._block_values(self.bio_GBF4_SNES_block, self.bio_GBF4_SNES_block_pairs, self.bio_GBF4_SNES_scales, x_vals)
            if settings.GBF4_TARGET_SNES != 'off' else 0
        )
        prod_data["BIO (GBF4) ECNES value (ha)"] = (
            self._block_values(self.bio_GBF4_ECNES_block, self.bio_GBF4_ECNES_block_pairs, self.bio_GBF4_ECNES_scales, x_vals)
            if settings.GBF4_TARGET_ECNES != 'off' else 0
        )
        prod_data["BIO (GBF8) value (ha)"] = (
            self._block_values(self.bio_GBF8_block, self.bio_GBF8_block_pairs, self.bio_GBF8_scales, x_vals)
            if settings.GBF8_TARGET != "off" else 0
        )

        def deviation(values: dict, limits_xr) -> list:
            """LHS − target per (region, item) key of a biodiversity family."""
            return [v - limits_xr.sel(dict(layer=k)).item() for k, v in values.items()]

        obj_val = {
            "ObjVal":                       self.gurobi_model.ObjVal if self.gurobi_model.Status == GRB.OPTIMAL else None,
            "Obj Economy":                  float(self.obj_vec @ x_vals[:self.obj_vec.size]),
            "Obj Penalties":                0,      # all constraints are hard: no penalty objective (key kept for the writers)
            'Economy (AUD) Ag':             float(econ['ag']),
            'Economy (AUD) Non-Ag Value':   float(econ['nonag']),
            'Economy (AUD) Ag-Man Value':   float(econ['am']),
            'Economy (AUD) Transition Ag':      float(econ['trans_ag']),       # ag->ag and non-ag->ag flow costs (negative)
            'Economy (AUD) Transition Non-Ag':  float(econ['trans_nonag']),    # ag->non-ag flow costs (negative)
            "Deviation Production (t)":     [prod_data["Production"][c] - limits['demand'][c] for c in range(self._input_data.ncms)],
            "Deviation Water (ML)":         ([prod_data["Water"][i] - lim for i, lim in limits['water'].items()]
                                             if settings.WATER_LIMITS == "on" else 0),
            "Deviation GHG (tCO2e)":        0,      # hard cap: no deviation (key kept for the writers)
            "Deviation BIO (GBF2) value (ha)":       ([prod_data["BIO (GBF2) value (ha)"] - limits['GBF2']]
                                                      if settings.GBF2_TARGET != "off" else 0),
            "Deviation BIO (GBF3) NVIS value (ha)":  (deviation(prod_data["BIO (GBF3) NVIS value (ha)"], limits['GBF3_NVIS'])
                                                      if settings.GBF3_NVIS_TARGET != "off" else 0),
            "Deviation BIO (GBF4) SNES value (ha)":  (deviation(prod_data["BIO (GBF4) SNES value (ha)"], limits['GBF4_SNES'])
                                                      if settings.GBF4_TARGET_SNES != 'off' else 0),
            "Deviation BIO (GBF4) ECNES value (ha)": (deviation(prod_data["BIO (GBF4) ECNES value (ha)"], limits['GBF4_ECNES'])
                                                      if settings.GBF4_TARGET_ECNES != 'off' else 0),
            "Deviation BIO (GBF8) value (ha)":       (deviation(prod_data["BIO (GBF8) value (ha)"], limits['GBF8'])
                                                      if settings.GBF8_TARGET != "off" else 0),
        }

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
            obj_val=obj_val,
        )

    @staticmethod
    def _block_lhs(block: sparse.csr_matrix, x_vals: np.ndarray, scales=None) -> np.ndarray:
        """Row values of a stored constraint block at the solution, in raw units: the stored
        (row-scaled) row times x, multiplied back by the row's scale factor when one is kept."""
        lhs = np.asarray(block @ x_vals[:block.shape[1]], dtype=np.float64)
        return lhs * np.asarray(scales, dtype=np.float64) if scales is not None else lhs

    def _block_values(self, block, keys: list, scales: dict, x_vals: np.ndarray) -> dict:
        """{key: raw-unit LHS} for a keyed family block; {} when the family added no rows."""
        if block is None:
            return {}
        return dict(zip(keys, self._block_lhs(block, x_vals, [scales[k] for k in keys]).tolist()))

