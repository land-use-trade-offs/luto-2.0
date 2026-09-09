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
from luto.solvers import row_builder, row_inputs
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

    def __init__(self, cols: dict, rows: row_inputs.RowInputs, obj_block: sparse.csr_matrix):
        """The LP of one step: ``cols`` the column space (col_builder.get_cols) — every unknown;
        ``rows`` the coefficient streams and targets (row_inputs.get_rows); ``obj_block`` the economy
        coefficients (row_builder.get_obj_block), one row per row_builder.OBJ_BLOCKS component."""
        self._cols = cols                         # the column space: every unknown as one row of cols['table'], the wide id grids beside it
        self._rows = rows
        self._obj_block = obj_block
        self._ncells = int(cols['ag'].sizes['cell'])
        self._nlms = int(cols['ag'].sizes['lm'])
        self._n_ag_lus = int(cols['ag'].sizes['lu'])
        self._n_non_ag_lus = int(cols['nonag'].sizes['nonag_lu'])
        self._agman2lu = cols['table'].attrs['agman2lu']
        self.gurobi_model = gp.Model(f"LUTO {settings.VERSION}", env=gurenv)

        # --- the decision variables: ONE MVar over the column table, the blocks as views (slices) of it ---
        self.x = None                       # over cols['table']: every column, in Var.index order
        self.ag_mvar = None                 # the ag block: the ag land-use shares X_ag
        self.nonag_mvar = None              # the non-ag block
        self.am_mvar = None                 # the ag-management block
        self.ag2ag_mvar = None              # the ag → ag arcs
        self.ag2nonag_mvar = None           # the ag → non-ag arcs
        self.nonag2ag_mvar = None           # the non-ag → ag arcs
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

        # --- the row table (row_builder.stack_rows): every constraint as one row — its family, key fields,
        # rhs, sense, name, scale, active flag, the Gurobi handle (constr) and, after the solve, its raw-unit
        # value (lhs) — with the ONE A (rows x n_all, CSR) in its attrs. The flat attributes above are views
        # published from it (_publish); a dropped row is flagged inactive, the table never shrinks ---
        self.rows_table = None
        self._vars = None                       # model.getVars() in Var.index order (materialised once, for addMConstr)

    def formulate(self):
        """The model: every unknown of the column space as variables, every row block of
        ``row_builder.FAMILIES`` as constraints, the objective."""
        print("Setting up the model...")
        self._setup_vars()
        self._setup_constraints()
        self._setup_objective()

    def _setup_vars(self):
        """Every column of the space as ONE addMVar over the table — lb / ub per row, Var.index order =
        table order (ag | nonag | am | ag2ag | ag2nonag | nonag2ag | cell_usage) — the blocks as slices
        of it, the names from the table fields."""
        print("├── Setting up decision variables...")
        table = self._cols['table']
        lm_name = np.array(['dry', 'irr'])
        snake_of_slot = np.array([tools.am_name_snake_case(option) for option, lus in self._agman2lu.items() for _ in lus], dtype=object)   # the (option, lu) slots in slot order
        self.x = self.gurobi_model.addMVar(table.attrs['n_all'], lb=table['lb'].values, ub=table['ub'].values, name="X")

        names_of = {                                                                # the name of every column of a block, from its fields
            'ag':         lambda t: [f"X_ag_{lm_name[m]}_{j}_{r}" for m, j, r in zip(t['m'], t['j'], t['cell'])],
            'nonag':      lambda t: [f"X_non_ag_{k}_{r}" for k, r in zip(t['k'], t['cell'])],
            'am':         lambda t: [f"X_ag_man_{lm_name[m]}_{snake_of_slot[slot]}_{j}_{r}".replace(" ", "_")
                                     for slot, m, j, r in zip(t['slot'], t['m'], t['j'], t['cell'])],
            'ag2ag':      lambda t: [f"F_a2a_{from_m}_{from_j}[{m},{local_r},{j}]"
                                     for from_m, from_j, m, local_r, j in zip(t['from_m'], t['from_j'], t['m'], t['local_r'], t['j'])],
            'ag2nonag':   lambda t: [f"F_a2n_{from_m}_{from_j}[{k},{local_r}]"
                                     for from_m, from_j, k, local_r in zip(t['from_m'], t['from_j'], t['k'], t['local_r'])],
            'nonag2ag':   lambda t: [f"F_n2a_{from_k}[{m},{local_r},{j}]"
                                     for from_k, m, local_r, j in zip(t['from_k'], t['m'], t['local_r'], t['j'])],
            'cell_usage': lambda t: [f"Rgconst_cell_usage_{cell}" for cell in t['cell']],
        }
        mvar_of = dict(ag='ag_mvar', nonag='nonag_mvar', am='am_mvar', ag2ag='ag2ag_mvar', ag2nonag='ag2nonag_mvar',
                       nonag2ag='nonag2ag_mvar', cell_usage='cell_usage_slack_mvar')
        for block, block_rows in table.attrs['block_range'].items():
            rows = slice(*block_rows)
            fields = {field: table[field].values[rows] for field in ('m', 'j', 'k', 'slot', 'from_m', 'from_j', 'from_k', 'local_r', 'cell')}
            mvar = self.x[rows]
            setattr(self, mvar_of[block], mvar)
            self.gurobi_model.setAttr('VarName', mvar.tolist(), names_of[block](fields))
            print(f"│   {'└──' if block == 'cell_usage' else '├──'} {block:<10s} {rows.stop - rows.start:>12,} variables")

    def _setup_constraints(self):
        """The row table: each family of ``row_builder.FAMILIES`` returns its part(s) (None when off),
        ``stack_rows`` lays them back to back, ONE ``addMConstr`` adds every row (Constr.index = table
        order), the rows are named, the handles kept on the table, and the flat views published."""
        print("├── Adding the constraints...")
        model = self.gurobi_model
        model.update()                       # the ONE update before the first row: every variable exists
        self._vars = model.getVars()
        assert len(self._vars) == self._cols['table'].attrs['n_all'], 'the model must hold exactly the columns of the space'
        parts = []
        for family in row_builder.FAMILIES:
            out = family(self._rows, self._cols)
            parts += [part for part in (out if isinstance(out, list) else [out]) if part is not None]
        T = self.rows_table = row_builder.stack_rows(parts)
        constrs = model.addMConstr(T.attrs['A'], self._vars, np.asarray(T['sense'].values, dtype='<U1'), T['rhs'].values).tolist()
        model.setAttr('ConstrName', constrs, T['name'].values.tolist())
        T['constr'] = (('row',), np.array(constrs, dtype=object))
        for family, (start, stop) in T.attrs['family_range'].items():
            print(f"│   │   {family}: {stop - start:,} row(s)")
        self._publish()

    def _active_rows(self, family: str):
        """The family's ACTIVE rows as a boolean mask over the row table, None where the family was not built."""
        T = self.rows_table
        span = row_builder.family_rows(T, family)
        if span is None:
            return None
        mask = np.zeros(T.sizes['row'], dtype=bool)
        mask[span] = T['active'].values[span]
        return mask

    def _publish(self) -> None:
        """The flat handle / scale attributes that the dual readers (``tools.calc_shadow_price_*``,
        ``record_shadow_prices``) and ``solve()`` walk, as views of the row table's ACTIVE rows.
        Re-run after any row is dropped or restored."""
        T = self.rows_table
        options = self._cols['table'].attrs['options']

        def constrs(f): r = self._active_rows(f); return list(T['constr'].values[r]) if r is not None else []
        def keys(f): r = self._active_rows(f); return row_builder.keys_of(T, f, r) if r is not None else []
        def scale(f): r = self._active_rows(f); return T['scale'].values[r] if r is not None else np.array([])
        def level(f, field): r = self._active_rows(f); return row_builder.decode(T, field, r).tolist() if r is not None else []

        self.cell_usage_constraint_r = dict(zip(level('cell_usage', 'cell'), constrs('cell_usage')))
        self.ag_mgt_link_constraints_r = defaultdict(list)
        for c, r in zip(constrs('ag_mgt_link'), level('ag_mgt_link', 'cell')):
            self.ag_mgt_link_constraints_r[r].append(c)
        self.ag_mgt_adoption_constraints = constrs('ag_mgt_adoption')
        self.demand_constraints = constrs('demand')
        self.demand_scales = scale('demand').tolist()
        self.ghg_constr = constrs('ghg')[0] if constrs('ghg') else None
        self.ghg_scale = float(scale('ghg')[0]) if scale('ghg').size else 1.0
        self.bio_GBF2_constr = constrs('GBF2')[0] if constrs('GBF2') else None
        self.bio_GBF2_scale = float(scale('GBF2')[0]) if scale('GBF2').size else 1.0
        for fam in ('GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8'):
            setattr(self, f'bio_{fam}_constrs', dict(zip(keys(fam), constrs(fam))))
            setattr(self, f'bio_{fam}_scales', dict(zip(keys(fam), scale(fam))))
        self.regional_adoption_constraints = (constrs('regional_adoption_ag') + constrs('regional_adoption_nonag')
                                              + constrs('regional_adoption_nonag_sum'))
        self.water_limit_constraints = constrs('water')
        self.water_scales = scale('water').tolist()
        ren_keys = [f'{options[am_idx]}_{state}' for am_idx, state in keys('renewable')]
        self.renewable_constraints = dict(zip(ren_keys, constrs('renewable')))
        self.renewable_scales = dict(zip(ren_keys, scale('renewable').tolist()))

    def bio_constraint_index(self) -> dict:
        """{constraint_name: {family, region, item, presence}} for every biodiversity row built — read
        off the row table, which keeps its rows whether ``remove_constraints_by_name`` has dropped them or not."""
        T = self.rows_table
        index = {}
        for family in ('GBF2', 'GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8'):
            span = row_builder.family_rows(T, family)
            if span is None:
                continue
            for name, region, item, presence in zip(T['name'].values[span], *(row_builder.decode(T, field, span) for field in ('region', 'item', 'presence'))):
                index[name] = {'family': family, 'region': region, 'item': item, 'presence': presence}
        return index

    def _setup_objective(self):
        """Objective obj · X: the economy block summed over its component rows, scaled to
        million AUD, then floored at SOLVER_COEFF_MIN — the last step of the coefficient
        contract (see ``row_builder``), applied here because the scaling can push a
        merged coefficient under the floor."""
        print(f"├── Setting up the objective function to {settings.OBJECTIVE}...")
        obj = np.asarray(self._obj_block.sum(axis=0)).ravel()        # (5 x n_dec) float32 block; disjoint vars per row
        obj = obj * (1.0 / 1e6)                                                 # raw AUD -> million AUD (reciprocal multiply, as gurobipy did)
        obj[np.abs(obj) < settings.SOLVER_COEFF_MIN] = 0.0                      # floor the merged, scaled coefficient
        self.obj_vec = obj

        X = self.x[:obj.size]                                              # the decision columns; the range slacks after them carry no objective
        if settings.OBJECTIVE == "mincost":
            self.gurobi_model.setObjective(obj @ X, GRB.MINIMIZE)
        elif settings.OBJECTIVE == "maxprofit":
            self.gurobi_model.setObjective(obj @ X, GRB.MAXIMIZE)
        else:
            raise ValueError(f"Unknown objective: {settings.OBJECTIVE}")
        
        print(f"│   └── objective: {int((obj != 0).sum()):,} nonzero coefficients over {obj.size:,} variables")

    def remove_constraints_by_name(self, names) -> None:
        """Drop rows: flagged inactive on the row table (it never shrinks, so a dropped row stays
        describable and its ``lhs`` is still read at the solution), the flat views re-published
        (a stale `Constr` there would raise on the first `.Pi` read after the next accepted solve),
        then removed from the Gurobi model. The infeasibility flow in `simulation.py` drops rows this way."""
        if not names:
            return
        T = self.rows_table
        hit = np.isin(T['name'].values, np.asarray(sorted(set(names)), dtype=object)) & T['active'].values
        if not hit.any():
            return
        T['active'] = (('row',), T['active'].values & ~hit)
        self._publish()
        self.gurobi_model.remove(list(T['constr'].values[hit]))
        self.gurobi_model.update()

    def restore_constraints_by_name(self, names) -> None:
        """Put dropped rows back: added to the Gurobi model again from the row table (same row, rhs,
        sense and name; new handles, appended after the existing rows), flagged active, the views re-published."""
        if not names:
            return
        T = self.rows_table
        hit = np.isin(T['name'].values, np.asarray(sorted(set(names)), dtype=object)) & ~T['active'].values
        if not hit.any():
            return
        constrs = self.gurobi_model.addMConstr(T.attrs['A'][hit], self._vars, np.asarray(T['sense'].values[hit], dtype='<U1'), T['rhs'].values[hit]).tolist()
        self.gurobi_model.setAttr('ConstrName', constrs, T['name'].values[hit].tolist())
        handles = T['constr'].values.copy()
        handles[hit] = constrs
        T['constr'] = (('row',), handles)
        T['active'] = (('row',), T['active'].values | hit)
        self.gurobi_model.update()
        self._publish()

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

        # ── 1. the decision variables: ONE .X read of the table, scattered back through its fields (float64 -> float32) ──
        table = self._cols['table']
        rows_of = {block: slice(*block_rows) for block, block_rows in table.attrs['block_range'].items()}
        m, j, k, am_idx, local_r, cell = (table[field].values for field in ('m', 'j', 'k', 'am_idx', 'local_r', 'cell'))
        x_vals = self.x.X                                                # every column, float64

        X_dry_sol_rj = np.zeros((self._ncells, self._n_ag_lus), dtype=np.float32)
        X_irr_sol_rj = np.zeros((self._ncells, self._n_ag_lus), dtype=np.float32)
        non_ag_X_sol_rk = np.zeros((self._ncells, self._n_non_ag_lus), dtype=np.float32)
        am_X_dry_sol_rj = {am: np.zeros((self._ncells, self._n_ag_lus), dtype=np.float32) for am in self._agman2lu}
        am_X_irr_sol_rj = {am: np.zeros((self._ncells, self._n_ag_lus), dtype=np.float32) for am in self._agman2lu}

        # agricultural
        ag = rows_of['ag']
        is_dry = m[ag] == 0
        X_dry_sol_rj[cell[ag][is_dry],  j[ag][is_dry]]  = x_vals[ag][is_dry]
        X_irr_sol_rj[cell[ag][~is_dry], j[ag][~is_dry]] = x_vals[ag][~is_dry]

        # non-agricultural (disabled land uses have no columns and stay at zero)
        nonag = rows_of['nonag']
        non_ag_X_sol_rk[cell[nonag], k[nonag]] = x_vals[nonag]

        # ag-management. Savanna eligibility is applied to BOTH lm here, while variable creation applied
        # it to dry only: irr savanna vars outside the eligible cells report 0.
        am = rows_of['am']
        options = table.attrs['options']
        am_of_col = np.asarray(options, dtype=object)[am_idx[am]]
        reported = ~((am_of_col == "Savanna Burning") & (m[am] == 1) & ~np.isin(cell[am], table.attrs['savanna_eligible_r']))
        for option in options:
            dry_cols = reported & (am_of_col == option) & (m[am] == 0)
            irr_cols = reported & (am_of_col == option) & (m[am] == 1)
            am_X_dry_sol_rj[option][cell[am][dry_cols], j[am][dry_cols]] = x_vals[am][dry_cols]
            am_X_irr_sol_rj[option][cell[am][irr_cols], j[am][irr_cols]] = x_vals[am][irr_cols]

        ag_X_mrj = np.stack((X_dry_sol_rj, X_irr_sol_rj))                # fractional values preserved as-is
        ag_man_X_mrj = {am: np.stack((am_X_dry_sol_rj[am], am_X_irr_sol_rj[am])) for am in self._agman2lu}

        # ── 2. the transition deltas: the gross flows the objective charged, SOURCE-KEYED so reporting can
        #       attribute the true from → to flows. Leaf axes mirror the flow_cost dicts ([to_m, local_r, to_j]
        #       for ag targets, [local_r, k] for non-ag targets); local_r indexes the source's cell list.
        #       Each source's arcs are one run of the block's rows (src_ptr = the group bounds of the block sorted by source).
        dvar_D_ag2ag_mrj    = {}   # (from_m, from_j) -> (NLMS, ncells_src, N_AG_LUS)
        dvar_D_ag2nonag_rk  = {}   # (from_m, from_j) -> (ncells_src, N_NON_AG_LUS)
        dvar_D_nonag2ag_mrj = {}   # from_k           -> (NLMS, ncells_k, N_AG_LUS)
        x_arcs = x_vals.astype(np.float32)

        def src_rows(block, src_idx):
            src_ptr = table.attrs['src_ptr'][block]
            return slice(int(src_ptr[src_idx]), int(src_ptr[src_idx + 1]))

        for src_idx, ((from_m, from_j), cells) in enumerate(self._cols['sources']['ag'].items()):
            arcs = src_rows('ag2ag', src_idx)
            deltas = np.zeros((self._nlms, len(cells), self._n_ag_lus), dtype=np.float32)
            deltas[m[arcs], local_r[arcs], j[arcs]] = x_arcs[arcs]
            dvar_D_ag2ag_mrj[(from_m, from_j)] = deltas

            arcs = src_rows('ag2nonag', src_idx)
            deltas = np.zeros((len(cells), self._n_non_ag_lus), dtype=np.float32)
            deltas[local_r[arcs], k[arcs]] = x_arcs[arcs]
            dvar_D_ag2nonag_rk[(from_m, from_j)] = deltas
        for src_idx, (from_k, cells) in enumerate(self._cols['sources']['nonag'].items()):
            arcs = src_rows('nonag2ag', src_idx)
            deltas = np.zeros((self._nlms, len(cells), self._n_ag_lus), dtype=np.float32)
            deltas[m[arcs], local_r[arcs], j[arcs]] = x_arcs[arcs]
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

        # ── 4. every row's value at the solution: ONE mat-vec over the row table, back in raw units (row × scale),
        #       dropped rows included (how far a dropped row would be violated is the diagnosis); the report reads the active rows ──
        limits = self._rows.limits
        T = self.rows_table
        lhs = (T.attrs['A'] @ x_vals) * T['scale'].values
        T['lhs'] = (('row',), lhs)

        def values(family) -> dict:
            """{key: raw-unit lhs} over the family's active rows; {} when it added no rows."""
            rows = self._active_rows(family)
            return dict(zip(row_builder.keys_of(T, family, rows), lhs[rows].tolist())) if rows is not None else {}

        def value(family):
            """The single row's raw-unit lhs (ghg, GBF2), None when not active."""
            rows = self._active_rows(family)
            return float(lhs[rows][0]) if rows is not None and rows.any() else None

        obj_block = self._obj_block
        econ = dict(zip(row_builder.OBJ_BLOCKS, obj_block @ x_vals[:obj_block.shape[1]]))   # per-block economy, raw AUD

        q_block = T.attrs.get('demand_q_block')                            # demand's per-commodity LHS, unscaled
        ghg, gbf2 = value('ghg'), value('GBF2')
        prod_data["Production"] = (q_block @ x_vals).tolist() if q_block is not None else 0   # raw t
        prod_data["GHG"] = ghg + float(np.asarray(self._rows.offland_ghg).ravel()[0]) if ghg is not None else 0   # the row excludes the offland constant
        prod_data["Water"] = {region: v for (region,), v in values('water').items()} if self._active_rows('water') is not None else 0
        prod_data["BIO (GBF2) value (ha)"] = gbf2 if gbf2 is not None else 0
        prod_data["BIO (GBF3) NVIS value (ha)"] = values('GBF3_NVIS') if settings.GBF3_NVIS_TARGET != 'off' else 0
        prod_data["BIO (GBF4) SNES value (ha)"] = values('GBF4_SNES') if settings.GBF4_TARGET_SNES != 'off' else 0
        prod_data["BIO (GBF4) ECNES value (ha)"] = values('GBF4_ECNES') if settings.GBF4_TARGET_ECNES != 'off' else 0
        prod_data["BIO (GBF8) value (ha)"] = values('GBF8') if settings.GBF8_TARGET != "off" else 0

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

