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
from typing import Any
from gurobipy import GRB
from scipy import sparse

from luto import tools
from luto.solvers import col_builder, row_builder
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

    def __init__(self, space: dict, rows: row_builder.RowInputs):
        """``space``: the column space of the step (col_builder.get_cols) — every unknown;
        ``rows``: the row side (row_builder.get_rows) — every coefficient stream and target."""
        self._space = space                         # the unified column space: every unknown as a labelled cube
        self._layout = space['layout']              # block offsets in Var.index order, n_dec, n_all
        self._rows = rows
        self._ncells = int(space['ag'].sizes['cell'])
        self._nlms = int(space['ag'].sizes['lm'])
        self._n_ag_lus = int(space['ag'].sizes['lu'])
        self._n_non_ag_lus = int(space['nonag'].sizes['nonag_lu'])
        self._agman2lu = space['am'].attrs['agman2lu']
        self.gurobi_model = gp.Model(f"LUTO {settings.VERSION}", env=gurenv)

        # --- decision-variable BLOCKS (one MVar per block of the column space) ---
        # For every block: the MVar and its global column offset (Var.index of element 0 — blocks
        # are created back to back in the space's layout order, so offsets chain arithmetically,
        # no model.update() needed). The column ids live in the space (`space[block].col`).
        self.ag_mvar = None                 # over space['ag']: the folded flow stream X_ag
        self.ag_offset = None
        self.nonag_mvar = None              # over space['nonag']
        self.nonag_offset = None
        self.am_mvar = None                 # over space['am']
        self.am_offset = None
        self.ag2ag_mvar = None              # over space['ag2ag'] (ag → ag), an arc list
        self.ag2ag_offset = None
        self.ag2nonag_mvar = None           # over space['ag2nonag'] (ag → non-ag)
        self.ag2nonag_offset = None
        self.nonag2ag_mvar = None           # over space['nonag2ag'] (non-ag → ag)
        self.nonag2ag_offset = None
        # accounting columns of the folded entries (X_acct where it differs from X_ag); None when nothing folds
        self.accounting_mvar = None
        self.accounting_offset = None
        self.accounting_link_constrs = []         # one exact equality per accounting column
        self.accounting_link_block = None
        self.cell_usage_slack_mvar = None   # the cell-usage range slacks (the last variables before any row)

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

        # --- the row blocks (row_builder.FAMILIES, keyed xr.Datasets with the Gurobi handles attached) ---
        self.blocks = {}                        # {family: block}; the flat attributes below are views published from these
        self._vars = None                       # _all_vars(): model.getVars() in Var.index order
        self._bio_index = {}                    # bio_constraint_index(): name -> {family, region, item, presence}, recorded when a bio block is added

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
        """The model: every unknown of the column space as variables, every row block of
        ``row_builder.FAMILIES`` as constraints, the objective."""
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
        self._setup_accounting_vars()
        self._setup_cell_usage_slack_vars()

    def _setup_constraints(self):
        """One loop: each family of ``row_builder.FAMILIES`` returns its row block(s) (None when
        off), each block goes to ``addMConstr`` as one call, is named, keeps its Gurobi handles,
        and the flat handle attributes are published from the blocks."""
        print("├── Adding the constraints...")
        model = self.gurobi_model
        model.update()                       # the ONE update before the first row: every variable exists
        self._vars = model.getVars()
        assert len(self._vars) == self._layout['n_all'], 'the model must hold exactly the columns of the space'
        for family in row_builder.FAMILIES:
            out = family(self._rows, self._space)
            for block in (out if isinstance(out, list) else [out]):
                if block is not None:
                    self._add_block(block)
        self._publish()

    def _setup_ag_vars(self):
        """The ag block: ONE addMVar over the existing entries of space['ag'], in column order (lu,
        lm, cell). lb = 0, ub from the cube (cleaned in col_builder: 0 ≤ base ≤ ub);
        ``space['ag'].col`` maps (lm, lu, cell) to Var.index (-1 = no variable)."""
        print("│   ├── setting up decision variables for agricultural land uses...")
        ag = self._space['ag']
        self.ag_offset = self._layout['ag']                              # recorded, not assumed: later blocks chain from it
        assert self.ag_offset == 0, 'the ag block must be the first block created'

        lu_idx, lm_idx, cell_idx = np.nonzero(ag['exists'].transpose('lu', 'lm', 'cell').values)   # column order: lu, lm, cell
        self.ag_mvar = self.gurobi_model.addMVar(
            cell_idx.size,
            lb=0.0,
            ub=ag['ub'].values[lm_idx, lu_idx, cell_idx].astype(np.float64),
            name="X_ag")
        self.gurobi_model.setAttr(
            'VarName',
            self.ag_mvar.tolist(),
            [f"X_ag_{'dry' if m == 0 else 'irr'}_{j}_{r}" for m, j, r in zip(lm_idx, lu_idx, cell_idx)]
        )

    def _setup_non_ag_vars(self):
        """The non-ag block: ONE addMVar over the existing entries of space['nonag'], in column
        order (k, cell). Bounds (collapse rule applied) come from the cube."""
        print("│   ├── setting up decision variables for non-agricultural land uses...")
        nonag = self._space['nonag']
        self.nonag_offset = self._layout['nonag']

        k_idx, cell_idx = np.nonzero(nonag['exists'].values)             # column order: k, cell
        self.nonag_mvar = self.gurobi_model.addMVar(
            cell_idx.size,
            lb=nonag['lb'].values[k_idx, cell_idx].astype(np.float64),
            ub=nonag['ub'].values[k_idx, cell_idx].astype(np.float64),
            name="X_non_ag")
        self.gurobi_model.setAttr(
            'VarName',
            self.nonag_mvar.tolist(),
            [f"X_non_ag_{k}_{r}" for k, r in zip(k_idx, cell_idx)]
        )

    def _setup_ag_management_variables(self):
        """The ag-management block: ONE addMVar over the existing entries of space['am'], in column
        order ((am, lu) slot, lm, cell). Bounds and the cell selection (GBF2 exclusion for
        renewables, savanna eligibility) come from the cube."""
        print("│   ├── setting up decision variables for agricultural management options...")
        am = self._space['am']
        self.am_offset = self._layout['am']

        slot_idx, lm_idx, cell_idx = np.nonzero(am['exists'].values)     # column order: (am, lu) slot, lm, cell
        self.am_mvar = self.gurobi_model.addMVar(
            cell_idx.size,
            lb=am['lb'].values[slot_idx, lm_idx, cell_idx].astype(np.float64),
            ub=np.ones(cell_idx.size, dtype=np.float64),
            name="X_ag_man")
        snake_of_slot = np.array([tools.am_name_snake_case(name) for name in am['am'].values], dtype=object)
        j_of_slot = am['j'].values
        self.gurobi_model.setAttr(
            'VarName',
            self.am_mvar.tolist(),
            [f"X_ag_man_{'dry' if m == 0 else 'irr'}_{snake_of_slot[slot]}_{j_of_slot[slot]}_{r}".replace(" ", "_")
             for slot, m, r in zip(slot_idx, lm_idx, cell_idx)]
        )

    def _setup_flow_vars(self):
        """The flow blocks: ONE addMVar per arc list (ag2ag, ag2nonag, nonag2ag) of the space. Arcs
        are in edge-table order (source in dict order, argwhere C-order within a source); the
        names come from the arc fields. A delta is a positive increment: no stay/diagonal var, the
        node-balance constant carries the base."""
        print("│   ├── setting up transition flow delta variables (D)...")
        ag2ag = self._space['ag2ag']
        ag2nonag = self._space['ag2nonag']
        nonag2ag = self._space['nonag2ag']
        self.ag2ag_offset = self._layout['ag2ag']
        self.ag2nonag_offset = self._layout['ag2nonag']
        self.nonag2ag_offset = self._layout['nonag2ag']
        n_ag2ag = ag2ag.attrs['n']
        n_ag2nonag = ag2nonag.attrs['n']
        n_nonag2ag = nonag2ag.attrs['n']

        self.ag2ag_mvar = self.gurobi_model.addMVar(n_ag2ag, lb=0.0, name="F_a2a")
        self.ag2nonag_mvar = self.gurobi_model.addMVar(n_ag2nonag, lb=0.0, name="F_a2n")
        self.nonag2ag_mvar = self.gurobi_model.addMVar(n_nonag2ag, lb=0.0, name="F_n2a")

        self.gurobi_model.setAttr(
            'VarName',
            self.ag2ag_mvar.tolist(),
            [f"F_a2a_{from_m}_{from_j}[{to_m},{local_r},{to_j}]" for from_m, from_j, to_m, local_r, to_j in
             zip(ag2ag['from_m'].values, ag2ag['from_j'].values, ag2ag['to_m'].values, ag2ag['local_r'].values, ag2ag['to_j'].values)]
        )
        self.gurobi_model.setAttr(
            'VarName',
            self.ag2nonag_mvar.tolist(),
            [f"F_a2n_{from_m}_{from_j}[{to_k},{local_r}]" for from_m, from_j, to_k, local_r in
             zip(ag2nonag['from_m'].values, ag2nonag['from_j'].values, ag2nonag['to_k'].values, ag2nonag['local_r'].values)]
        )
        self.gurobi_model.setAttr(
            'VarName',
            self.nonag2ag_mvar.tolist(),
            [f"F_n2a_{from_k}[{to_m},{local_r},{to_j}]" for from_k, to_m, local_r, to_j in
             zip(nonag2ag['from_k'].values, nonag2ag['to_m'].values, nonag2ag['local_r'].values, nonag2ag['to_j'].values)]
        )

        print(f"│   │   ├── ag2ag    : {n_ag2ag:,} delta vars")
        print(f"│   │   ├── ag2nonag : {n_ag2nonag:,} delta vars")
        print(f"│   │   ├── nonag2ag : {n_nonag2ag:,} delta vars")
        print(f"│   │   └── total    : {n_ag2ag + n_ag2nonag + n_nonag2ag:,} delta vars")

    def _setup_accounting_vars(self):
        """The accounting block: one column per entry of the accounting view X_acct that the θ fold
        makes different from its flow variable (a folded sliver and its receiving dominant); tied
        to X_ag by the exact linking rows (``row_builder.accounting_link_rows``). Everywhere else X_acct
        IS X_ag and no variable is created. Column order: dominants, then slivers."""
        print("│   └── setting up accounting variables for folded entries...")
        accounting = self._space['accounting']
        self.accounting_offset = self._layout['accounting']
        n_new = accounting.attrs['n_new']
        if n_new == 0:
            print("│       └── nothing folded: the accounting view is the flow stream")
            return
        lu_code = {name: j for j, name in enumerate(self._space['ag'].lu.values)}
        names = ([f"X_acct_{lm}_{lu_code[lu]}_{r}" for lm, lu, r in zip(accounting['dom_lm'].values, accounting['dom_lu'].values, accounting['dom_cell'].values)]
                 + [f"X_acct_{lm}_{lu_code[lu]}_{r}" for lm, lu, r in
                    zip(accounting['sliver_from_lm'].values, accounting['sliver_from_lu'].values, accounting['sliver_cell'].values)])
        self.accounting_mvar = self.gurobi_model.addMVar(n_new, lb=0.0, ub=1.0, name="X_acct")
        self.gurobi_model.setAttr('VarName', self.accounting_mvar.tolist(), names)
        print(f"│       └── {n_new:,} accounting vars ({accounting.attrs['n_dom']:,} dominants + {accounting.attrs['n_sliver']:,} slivers)")

    def _all_vars(self):
        """The model's Var list in Var.index order (materialised once), for addMConstr."""
        if self._vars is None:
            self.gurobi_model.update()
            self._vars = self.gurobi_model.getVars()
        return self._vars

    def _setup_cell_usage_slack_vars(self):
        """The cell-usage range slacks, as Gurobi's addRange creates them (lb 0, ub = hi − lo), one
        per cell of ``space['cell_usage']`` — the last variables before any row."""
        row_cells, lo, hi = row_builder.cell_usage_band(self._rows, self._space)
        self.cell_usage_slack_mvar = self.gurobi_model.addMVar(row_cells.size, lb=0.0, ub=hi - lo, name="Rg")
        self.gurobi_model.setAttr('VarName', self.cell_usage_slack_mvar.tolist(), [f"Rgconst_cell_usage_{cell}" for cell in row_cells])

    def _add_block(self, block) -> None:
        """One row block -> one ``addMConstr`` (against the full Var list, in Var.index order), the
        row names, the Gurobi handles attached to the block as ``constr``."""
        senses = block['sense'].values
        sense = str(senses[0]) if (senses == senses[0]).all() else np.asarray(senses, dtype='<U1')
        constrs = self.gurobi_model.addMConstr(block.attrs['A'], self._vars, sense, block['rhs'].values).tolist()
        self.gurobi_model.setAttr('ConstrName', constrs, block['name'].values.tolist())
        block['constr'] = (('row',), np.array(constrs, dtype=object))
        family = block.attrs['family']
        self.blocks[family] = block
        if family in ('GBF2', 'GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8'):
            # name -> {family, region, item, presence}: the NAME is all a Gurobi model carries and it
            # cannot be parsed back (spaces became "_"); recorded here, never read from the model,
            # and never invalidated so dropped rows can still be described.
            keys = [()] if family == 'GBF2' else row_builder.block_keys(block)
            for name, key in zip(block['name'].values, keys):
                row = {'family': family, 'region': None, 'item': None, 'presence': None}
                row.update(dict(zip(('region', 'item', 'presence'), key)))
                self._bio_index[name] = row
        print(f"│   │   {family}: {block.sizes['row']:,} row(s)")

    def _publish(self) -> None:
        """The flat handle / scale / block attributes that the dual readers (``tools.calc_shadow_price_*``,
        ``record_shadow_prices``), ``remove_constraints_by_name`` and ``solve()`` walk, as views of
        the row blocks. Re-run after any block changes."""
        B = self.blocks

        def constrs(f): return list(B[f]['constr'].values) if f in B else []
        def keys(f): return row_builder.block_keys(B[f]) if f in B else []
        def scale(f): return B[f]['scale'].values if f in B else np.array([])
        def level(f, lvl): return B[f].indexes['row'].get_level_values(lvl).tolist() if f in B else []
        def A(f): return B[f].attrs['A'] if f in B else None

        self.cell_usage_constraint_r = dict(zip(level('cell_usage', 'cell'), constrs('cell_usage')))
        self.cell_usage_block = A('cell_usage')
        self.accounting_link_constrs = constrs('accounting_link')
        self.accounting_link_block = A('accounting_link')
        self.ag_mgt_link_constraints_r = defaultdict(list)
        for c, r in zip(constrs('ag_mgt_link'), level('ag_mgt_link', 'cell')):
            self.ag_mgt_link_constraints_r[r].append(c)
        self.ag_mgt_link_block = A('ag_mgt_link')
        self.ag_mgt_adoption_constraints = constrs('ag_mgt_adoption')
        self.ag_mgt_adopt_block = A('ag_mgt_adoption')
        self.demand_constraints = constrs('demand')
        self.demand_scales = scale('demand').tolist()
        self.demand_q_block = B['demand'].attrs['q_block'] if 'demand' in B else None
        self.ghg_constr = constrs('ghg')[0] if constrs('ghg') else None
        self.ghg_scale = float(scale('ghg')[0]) if 'ghg' in B and B['ghg'].sizes['row'] else 1.0
        self.ghg_block = A('ghg')
        self.bio_GBF2_constr = constrs('GBF2')[0] if constrs('GBF2') else None
        self.bio_GBF2_scale = float(scale('GBF2')[0]) if 'GBF2' in B and B['GBF2'].sizes['row'] else 1.0
        self.bio_GBF2_block = A('GBF2')
        for fam in ('GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8'):
            setattr(self, f'bio_{fam}_constrs', dict(zip(keys(fam), constrs(fam))))
            setattr(self, f'bio_{fam}_scales', dict(zip(keys(fam), scale(fam))))
            setattr(self, f'bio_{fam}_block', A(fam))
            setattr(self, f'bio_{fam}_block_pairs', keys(fam))
        self.regional_adoption_constraints = (constrs('regional_adoption_ag') + constrs('regional_adoption_nonag')
                                              + constrs('regional_adoption_nonag_sum'))
        self.water_limit_constraints = constrs('water')
        self.water_scales = scale('water').tolist()
        self.water_block = A('water')
        self.water_block_regids = level('water', 'region')
        ren_keys = [f'{am}_{state}' for am, state in keys('renewable')]
        self.renewable_constraints = dict(zip(ren_keys, constrs('renewable')))
        self.renewable_scales = dict(zip(ren_keys, scale('renewable').tolist()))
        caps = [A(f) for f in ('source_cap_ag', 'source_cap_nonag') if f in B]
        self.source_cap_block = sparse.vstack(caps).tocsr() if len(caps) > 1 else (caps[0] if caps else None)
        self.source_cap_keys = [n for f in ('source_cap_ag', 'source_cap_nonag') if f in B for n in B[f]['name'].values.tolist()]
        self.node_balance_block = A('node_balance')
        self.node_balance_keys = B['node_balance']['name'].values.tolist() if 'node_balance' in B else []

    def bio_constraint_index(self) -> dict:
        """{constraint_name: {family, region, item, presence}} for every biodiversity row built —
        recorded when the block was added, never read back from the model, and still describing
        rows that ``remove_constraints_by_name`` has since dropped."""
        return self._bio_index

    def _setup_objective(self):
        """Objective obj · X: the economy block summed over its component rows, scaled to
        million AUD, then floored at SOLVER_COEFF_MIN — the last step of the coefficient
        contract (see ``row_builder``), applied here because the scaling can push a
        merged coefficient under the floor."""
        print(f"├── Setting up the objective function to {settings.OBJECTIVE}...")
        obj = np.asarray(self._rows.obj_block.sum(axis=0)).ravel()        # (5 x n_dec) float32 block; disjoint vars per row
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
        """Remove rows from the Gurobi model AND from the row blocks the dual readers walk.

        The infeasibility flow in `simulation.py` drops rows by name. Removing them only from the
        model leaves stale `Constr` handles in the blocks, and the first attribute read on one —
        `record_shadow_prices` reading `.Pi` after the next ACCEPTED solve — raises "Constr was
        removed from the model". So the rows are dropped from every block that holds one, the
        flat views are re-published, and only then are the rows removed from the model.
        `bio_constraint_index()` is deliberately NOT invalidated — it was recorded at build time
        precisely so that dropped rows can be described afterwards.
        """
        if not names:
            return
        doomed = np.asarray(sorted(set(names)), dtype=object)
        for family, block in list(self.blocks.items()):
            hit = np.isin(block['name'].values, doomed)
            if hit.any():
                keep = ~hit
                trimmed = block.isel(row=keep)
                trimmed.attrs = dict(block.attrs, A=block.attrs['A'][keep])
                self.blocks[family] = trimmed
        self._publish()
        doomed = set(names)
        self.gurobi_model.remove([c for c in self.gurobi_model.getConstrs() if c.ConstrName in doomed])
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

        # ── 1. the decision variables, scattered back through the space's column order (float64 -> float32) ──
        X_dry_sol_rj = np.zeros((self._ncells, self._n_ag_lus), dtype=np.float32)
        X_irr_sol_rj = np.zeros((self._ncells, self._n_ag_lus), dtype=np.float32)
        non_ag_X_sol_rk = np.zeros((self._ncells, self._n_non_ag_lus), dtype=np.float32)
        am_X_dry_sol_rj = {am: np.zeros((self._ncells, self._n_ag_lus), dtype=np.float32) for am in self._agman2lu}
        am_X_irr_sol_rj = {am: np.zeros((self._ncells, self._n_ag_lus), dtype=np.float32) for am in self._agman2lu}

        # agricultural: ONE .X read of the ag block
        lu_idx, lm_idx, cell_idx = np.nonzero(self._space['ag']['exists'].transpose('lu', 'lm', 'cell').values)   # column order
        x_ag = self.ag_mvar.X                                            # float64 ndarray
        is_dry = lm_idx == 0
        X_dry_sol_rj[cell_idx[is_dry],  lu_idx[is_dry]]  = x_ag[is_dry]
        X_irr_sol_rj[cell_idx[~is_dry], lu_idx[~is_dry]] = x_ag[~is_dry]

        # non-agricultural: ONE .X read of the block (disabled land uses have no columns and stay at zero)
        k_idx, cell_idx = np.nonzero(self._space['nonag']['exists'].values)
        non_ag_X_sol_rk[cell_idx, k_idx] = self.nonag_mvar.X

        # ag-management: ONE .X read of the block. Savanna eligibility is applied to BOTH lm here,
        # while variable creation applied it to dry only: irr savanna vars outside the eligible cells report 0.
        am_ds = self._space['am']
        slot_idx, lm_idx, cell_idx = np.nonzero(am_ds['exists'].values)
        am_of_col = am_ds['am'].values[slot_idx]
        j_of_col = am_ds['j'].values[slot_idx]
        x_am = self.am_mvar.X
        reported = ~((am_of_col == "Savanna Burning") & (lm_idx == 1) & ~np.isin(cell_idx, am_ds.attrs['savanna_eligible_r']))
        for am in am_ds.attrs['am_list']:
            dry_cols = reported & (am_of_col == am) & (lm_idx == 0)
            irr_cols = reported & (am_of_col == am) & (lm_idx == 1)
            am_X_dry_sol_rj[am][cell_idx[dry_cols], j_of_col[dry_cols]] = x_am[dry_cols]
            am_X_irr_sol_rj[am][cell_idx[irr_cols], j_of_col[irr_cols]] = x_am[irr_cols]

        ag_X_mrj = np.stack((X_dry_sol_rj, X_irr_sol_rj))                # fractional values preserved as-is
        ag_man_X_mrj = {am: np.stack((am_X_dry_sol_rj[am], am_X_irr_sol_rj[am])) for am in self._agman2lu}

        # ── 2. the transition deltas: the gross flows the objective charged, SOURCE-KEYED so reporting can
        #       attribute the true from → to flows. Leaf axes mirror the flow_cost dicts ([to_m, local_r, to_j]
        #       for ag targets, [local_r, k] for non-ag targets); local_r indexes the source's cell list.
        dvar_D_ag2ag_mrj    = {}   # (from_m, from_j) -> (NLMS, ncells_src, N_AG_LUS)
        dvar_D_ag2nonag_rk  = {}   # (from_m, from_j) -> (ncells_src, N_NON_AG_LUS)
        dvar_D_nonag2ag_mrj = {}   # from_k           -> (NLMS, ncells_k, N_AG_LUS)
        ag2ag = self._space['ag2ag']
        ag2nonag = self._space['ag2nonag']
        nonag2ag = self._space['nonag2ag']
        x_ag2ag = self.ag2ag_mvar.X.astype(np.float32) if ag2ag.attrs['n'] else np.zeros(0, np.float32)
        x_ag2nonag = self.ag2nonag_mvar.X.astype(np.float32) if ag2nonag.attrs['n'] else np.zeros(0, np.float32)
        x_nonag2ag = self.nonag2ag_mvar.X.astype(np.float32) if nonag2ag.attrs['n'] else np.zeros(0, np.float32)
        for src_idx, ((from_m, from_j), cells) in enumerate(self._space['sources']['ag'].items()):
            start = int(ag2ag.attrs['src_ptr'][src_idx])
            stop = int(ag2ag.attrs['src_ptr'][src_idx + 1])
            deltas = np.zeros((self._nlms, len(cells), self._n_ag_lus), dtype=np.float32)
            deltas[ag2ag['to_m'].values[start:stop], ag2ag['local_r'].values[start:stop], ag2ag['to_j'].values[start:stop]] = x_ag2ag[start:stop]
            dvar_D_ag2ag_mrj[(from_m, from_j)] = deltas

            start = int(ag2nonag.attrs['src_ptr'][src_idx])
            stop = int(ag2nonag.attrs['src_ptr'][src_idx + 1])
            deltas = np.zeros((len(cells), self._n_non_ag_lus), dtype=np.float32)
            deltas[ag2nonag['local_r'].values[start:stop], ag2nonag['to_k'].values[start:stop]] = x_ag2nonag[start:stop]
            dvar_D_ag2nonag_rk[(from_m, from_j)] = deltas
        for src_idx, (from_k, cells) in enumerate(self._space['sources']['nonag'].items()):
            start = int(nonag2ag.attrs['src_ptr'][src_idx])
            stop = int(nonag2ag.attrs['src_ptr'][src_idx + 1])
            deltas = np.zeros((self._nlms, len(cells), self._n_ag_lus), dtype=np.float32)
            deltas[nonag2ag['to_m'].values[start:stop], nonag2ag['local_r'].values[start:stop], nonag2ag['to_j'].values[start:stop]] = x_nonag2ag[start:stop]
            dvar_D_nonag2ag_mrj[from_k] = deltas

        # ── 3. the maps: land use, land management, ag-management options ──
        non_ag_dominates_r = non_ag_X_sol_rk.max(axis=1) > ag_X_mrj.max(axis=(0, 2))   # used for lumap/lmmap only
        lumap = ag_X_mrj.sum(axis=0).argmax(axis=1).astype("int8")
        lmmap = ag_X_mrj.sum(axis=2).argmax(axis=0).astype("int8")
        lumap[non_ag_dominates_r] = (
            non_ag_X_sol_rk[non_ag_dominates_r, :].argmax(axis=1)
            + settings.NON_AGRICULTURAL_LU_BASE_CODE
        )
        lmmap[non_ag_dominates_r] = 0                                    # all non-agricultural land uses are dryland
        # one map per option (options can stack): 1 where the cell's chosen (lm, lu) carries the option at or
        # above AGRICULTURAL_MANAGEMENT_USE_THRESHOLD; non-ag cells carry no option
        ammaps = {am: np.zeros(self._ncells, dtype=np.int8) for am in AG_MANAGEMENTS}
        ag_cells = np.flatnonzero(lumap < settings.NON_AGRICULTURAL_LU_BASE_CODE)
        chosen_j = lumap[ag_cells].astype(np.int64)
        chosen_m = lmmap[ag_cells].astype(np.int64)
        for am, lu_codes in self._agman2lu.items():
            adoption = ag_man_X_mrj[am][chosen_m, ag_cells, chosen_j]
            adopted = (adoption >= settings.AGRICULTURAL_MANAGEMENT_USE_THRESHOLD) & np.isin(chosen_j, lu_codes)
            ammaps[am][ag_cells[adopted]] = 1

        # ── 4. constraint LHS values at the solution: one mat-vec per stored block (report only) ──
        x_vals = np.asarray(self.gurobi_model.getAttr('X', self.gurobi_model.getVars()), dtype=np.float64)
        limits = self._rows.limits

        obj_block = self._rows.obj_block
        econ = dict(zip(row_builder.OBJ_BLOCKS, obj_block @ x_vals[:obj_block.shape[1]]))   # per-block economy, raw AUD

        prod_data["Production"] = (
            self._block_lhs(self.demand_q_block, x_vals).tolist()                  # raw t
            if self.demand_q_block is not None else 0
        )
        prod_data["GHG"] = (
            float(self._block_lhs(self.ghg_block, x_vals, [self.ghg_scale])[0])
            + float(np.asarray(self._rows.offland_ghg).ravel()[0])          # the block excludes the offland constant
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
            "Deviation Production (t)":     [prod_data["Production"][c] - limits['demand'][c] for c in range(self._rows.ncms)],
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
        lhs = block @ x_vals[:block.shape[1]]                            # float64 ndarray
        return lhs * np.asarray(scales, dtype=np.float64) if scales is not None else lhs   # scales: a list

    def _block_values(self, block, keys: list, scales: dict, x_vals: np.ndarray) -> dict:
        """{key: raw-unit LHS} for a keyed family block; {} when the family added no rows."""
        if block is None:
            return {}
        return dict(zip(keys, self._block_lhs(block, x_vals, [scales[k] for k in keys]).tolist()))

