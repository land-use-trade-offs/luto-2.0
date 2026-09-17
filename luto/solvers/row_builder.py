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

import numpy as np
import xarray as xr

from dataclasses import dataclass
from scipy import sparse

import luto.tools as tools

from luto import settings
from luto.solvers.col_builder import ColSide
from luto.solvers.row_inputs import EconomicInputs, RowInputs
from luto.solvers.row_table import make_part, stack_rows


# ═══════════════════════════ get_rows: the row space of one step ═══════════════════════════

@dataclass
class RowSide:
    """What the post-solve reads beside the row table."""
    q_block: sparse.csr_matrix      # (commodity x n_all) the per-commodity production row over the columns, UNSCALED — the demand rows are its rescaled copies; Production = q_block @ x


def get_rows(inputs: RowInputs, cols: xr.Dataset, side: ColSide) -> tuple[xr.Dataset, RowSide]:
    """The row space of one solve step"""

    # ── 1. the structural rows: what a cell's space, an ag-mgt option and the existing capacity allow ──
    parts = [
        add_renewable_ceiling(inputs, cols, side),                              # simulated + existing capacity share the cell
        add_cell_usage(inputs, cols, side),                                     # every cell's shares sum to its ag proportion
        add_ag_mgt_link(inputs, cols, side),                                    # an ag-mgt column cannot exceed its ag column ...
        add_ag_mgt_adoption(inputs, cols, side),                                # ... nor the option's adoption limit
    ]

    # ── 2. the demand rows: the production block, kept unscaled beside the table ──
    demand, q_block = add_demand(inputs, cols, side)
    parts.append(demand)

    # ── 3. the policy rows: a coefficient per column laid on the support, multiplied by the weights over the cells the target covers ──
    parts.append(add_ghg(inputs, cols, side))

    print("│   ├── Adding constraints for biodiversity...")
    bio_on = any(target != 'off' for target in (settings.GBF2_TARGET, settings.GBF3_NVIS_TARGET,
                                                settings.GBF4_TARGET_SNES, settings.GBF4_TARGET_ECNES, settings.GBF8_TARGET))
    bio_S = side.by_cell @ sparse.diags(bio_coeff(inputs, cols, side)) if bio_on else None   # the biodiversity contribution laid on the support ONCE: every GBF family's rows are W @ bio_S
    parts += [
        add_GBF2(inputs, cols, bio_S),
        add_GBF3_NVIS(inputs, cols, bio_S),
        add_GBF4_SNES(inputs, cols, bio_S),
        add_GBF4_ECNES(inputs, cols, bio_S),
        add_GBF8(inputs, cols, bio_S),
    ]

    # the non-ag caps recede 1e-6/yr RELATIVE, so the RHS always stays ahead of the ratcheting lower bound
    # non-reversible plantings create (last year's solved areas become this year's exact lower bounds, and
    # float32 noise then puts the locked-in floor a hair over a saturated cap, which presolve rejects with NO
    # tolerance). Ag caps need no slack: ag is reversible.
    relax = 1 + (inputs.target_year - settings.SIM_YEARS[0]) * 1e-6
    parts += [
        add_regional_adoption_ag(inputs, cols, side),
        add_regional_adoption_nonag(inputs, cols, side, relax),
        add_regional_adoption_nonag_sum(inputs, cols, side, relax),
        add_water(inputs, cols, side),
        add_renewable(inputs, cols, side),
    ]

    # ── 4. the flow rows: the arcs grouped by the source they leave, every column and arc looked up on the grid of the node it lands on ──
    parts += [
        add_source_cap_ag(cols, side),
        add_source_cap_nonag(cols, side),
        add_node_balance_ag(cols, side),
        add_node_balance_nonag(cols, side),
    ]

    # ── 5. the space: the parts back to back in the order above, and beside the table the unscaled production block ──
    return stack_rows([part for part in parts if part is not None]), RowSide(q_block=q_block)


# ═══════════════════════════ the coefficient contract: gather → weigh → contract ═══════════════════════════
#
# The model is one matrix, rows × cols. The columns are the column table; every family below produces ROWS of
# that matrix, always the same way: GATHER a coefficient per column — a block's wide input array masked by the mask the
# block was enumerated from is its run of the table, ``c[slice(*cols.attrs[<block>_range][key])] = input[mask]`` — lay
# then pick the columns of each ROW: a regional family (water, renewable) by the column's region label,
# ``side.region2col[layer] == code`` (the block-by-block union it stands for — the input filtered by ``region2cell``,
# the block's run by ``region2col`` — is hoisted into the one gather); a layer family (GBF2/3/4/8, cell usage, the ceiling) by laying the
# coefficient on the cell incidence and multiplying the family's WEIGHT ROWS over cells (``W @ (side.by_cell @
# diags(c))``: row i, column t = W[i, cell_t] · c_t, one float32 product per entry) — and CONTRACT the stacked block
# in one loop over its rows — the SOLVER_COEFF_MIN drop, then (policy families only) the geomean row rescale and
# the floor again.

def gather(cols: xr.Dataset, side: ColSide, ag_c_mrj, am_c_mrj: dict, nonag_c_rk) -> np.ndarray:
    """One family's coefficient at every column, float32 (zero off the accounting columns): the ag input (lm, cell, lu),
    the non-ag input (cell, nonag_lu) and, per (option, land use) slot, the option's (lm, cell) input at that land use,
    each masked onto its block's run of the table."""
    c = np.zeros(cols.attrs['n_all'], dtype=np.float32)
    c[slice(*cols.attrs['block_range']['ag'])]    = ag_c_mrj[side.valid_ag_mrj]
    c[slice(*cols.attrs['block_range']['nonag'])] = nonag_c_rk[side.valid_nonag_rk]
    for (option, j_idx), mask in side.valid_am.items():
        c[slice(*cols.attrs['am_range'][(option, j_idx)])] = am_c_mrj[option][:, :, j_idx][mask]
    return c


def weight_rows(weights, ncells: int) -> sparse.csr_matrix:
    """A family's weighting rows over cells as ONE sparse ``W`` (n_rows × ncells), float32, the nonzero support
    only — built row by row, never as a dense stack (GBF8 would be 10k rows × ncells)."""
    indptr = [0]
    indices = []
    data = []
    for weight_row in weights:
        weight_row = np.asarray(weight_row, dtype=np.float32)
        cells = np.flatnonzero(weight_row)                                 # NaN counts as nonzero: it reaches ``contract``, which drops it
        indices.append(cells)
        data.append(weight_row[cells])
        indptr.append(indptr[-1] + cells.size)
    return sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), np.asarray(indptr, dtype=np.int64)),
                             shape=(len(weights), ncells))


def contract(block: sparse.csr_matrix, rhs=None, rescale: bool = False) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """The coefficient contract as ONE loop over the rows of a stacked block: every entry with
    |a| < SOLVER_COEFF_MIN dropped (NaN too); when ``rescale`` — the policy families — row i and rhs i are
    divided by the geometric mean of max|row i| and |rhs i| over RESCALE_FACTOR (an exact LP transformation)
    and the scaled row floored again. Returns (block, rhs, scale): scale 1 where not rescaled,
    ``row × scale`` restores the raw row."""
    block = block.tocsr(copy=True)
    block.data[~(np.abs(block.data) >= settings.SOLVER_COEFF_MIN)] = 0.0   # the drop; NaN fails the test too
    block.eliminate_zeros()
    block.sort_indices()
    rhs = None if rhs is None else np.asarray(rhs, dtype=np.float64)
    if not rescale:
        return block, rhs, np.ones(block.shape[0], dtype=np.float64)

    nnz_per_row = np.diff(block.indptr)
    row_max = np.zeros(block.shape[0], dtype=np.float64)
    has_entries = nnz_per_row > 0
    if has_entries.any():
        row_max[has_entries] = np.maximum.reduceat(np.abs(block.data).astype(np.float64), block.indptr[:-1][has_entries])

    # sqrt(max|row| · |rhs|) / RESCALE_FACTOR lands max|row| and |rhs| symmetrically around RESCALE_FACTOR in
    # log space; where one side is absent, the side that exists sets the factor on its own
    rhs_max = np.abs(rhs)
    balanced = (row_max > 0.0) & (rhs_max > 0.0)
    scale = np.where(balanced, np.sqrt(row_max * rhs_max),
                     np.where(row_max > 0.0, row_max, settings.RESCALE_FACTOR)) / settings.RESCALE_FACTOR

    block.data = (block.data / np.repeat(scale, nnz_per_row)).astype(np.float32)
    block.data[np.abs(block.data) < settings.SOLVER_COEFF_MIN] = 0.0       # floor the scaled row
    block.eliminate_zeros()
    block.sort_indices()
    return block, rhs / scale, scale


def bio_coeff(inputs: RowInputs, cols: xr.Dataset, side: ColSide) -> np.ndarray:
    """The biodiversity contribution at every column: the three streams ``gather`` reads — a scalar per ag land
    use, a per-cell array per (option, land use), a scalar per non-ag land use — as broadcast VIEWS (no copy),
    gathered ONCE for all five GBF families."""
    nlms, ncells = inputs.nlms, inputs.ncells
    ag_j = np.asarray(inputs.biodiv_contr_ag_j, dtype=np.float32)
    ag_c_mrj = np.broadcast_to(ag_j[None, None, :], (nlms, ncells, ag_j.size))
    contr_nonag_k = inputs.biodiv_contr_non_ag_k
    n_k = max(contr_nonag_k) + 1 if len(contr_nonag_k) else 0
    nonag_k = np.array([contr_nonag_k.get(lu, 0.0) for lu in range(n_k)], dtype=np.float32)
    nonag_c_rk = np.broadcast_to(nonag_k[None, :], (ncells, n_k))
    am_c_mrj = {}
    for option, by_j_idx in inputs.biodiv_contr_ag_man.items():
        per_cell = np.stack([np.asarray(by_j_idx[j_idx], dtype=np.float32) for j_idx in range(len(by_j_idx))], axis=1)   # (cell, j_idx)
        am_c_mrj[option] = np.broadcast_to(per_cell[None, :, :], (nlms, ncells, per_cell.shape[1]))
    return gather(cols, side, ag_c_mrj, am_c_mrj, nonag_c_rk)


# ═══════════════════════════ get_obj: the objective coefficient of every column ═══════════════════════════

def get_obj(econ: EconomicInputs, cols: xr.Dataset, side: ColSide) -> xr.DataArray:
    """The objective coefficient of every column (on ``col``), as Gurobi takes it: the operating economics on
    the accounting columns and the transition costs, negated, on the arcs — raw AUD, float32 — through the
    coefficient contract (the SOLVER_COEFF_MIN drop), then scaled to million AUD and floored again, because
    the scaling can push a coefficient under the floor. Zero on the cell-usage slacks: they carry no cost."""
    # ── operating economics: one gather over the accounting columns (zero everywhere else) ──
    obj = gather(cols, side, econ.ag_obj_mrj, econ.ag_man_objs, econ.non_ag_obj_rk)

    # ── transition costs on the arcs: per source, its cost array masked is its run of the block ──
    for src, mask in side.valid_ag2ag.items():                               # ag → ag: the cost at [to_m, local_r, to_j]
        obj[slice(*cols.attrs['ag2ag_range'][src])] = -econ.flow_cost_ag2ag[src][mask]
    for src, mask in side.valid_nonag2ag.items():                            # non-ag → ag: the cost at [to_m, local_r, to_j]
        obj[slice(*cols.attrs['nonag2ag_range'][src])] = -econ.flow_cost_nonag2ag[src][mask]
    for src, mask in side.valid_ag2nonag.items():                            # ag → non-ag: the cost at [local_r, to_k]
        obj[slice(*cols.attrs['ag2nonag_range'][src])] = -econ.flow_cost_ag2nonag[src][mask]

    # ── the contract: the drop on the raw coefficient, the scaling to million AUD, the floor on the scaled one ──
    obj[~(np.abs(obj) >= settings.SOLVER_COEFF_MIN)] = 0.0                     # the drop; NaN fails the test too
    obj = obj * (1.0 / 1e6)                                                    # raw AUD -> million AUD (float32, a reciprocal multiply as gurobipy did)
    obj[np.abs(obj) < settings.SOLVER_COEFF_MIN] = 0.0                         # floor the scaled coefficient
    return xr.DataArray(obj.astype(np.float64), dims=('col',))


# ═══════════════════════════ the structural rows ═══════════════════════════

def add_renewable_ceiling(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Simulated and existing renewable capacity compete for the cell's space [0, ag_mask]: one row per
    (am, cell) with existing capacity, Σ_{m, j} X_am[am, m, j, r] ≤ max(ag_mask[r] − exist_r[r], 0) — the support
    at those cells, over the option's columns."""
    am_idx = cols['am_idx'].values
    ag_mask = inputs.ag_mask_proportion_r
    blocks = []
    rhs = []
    names = []
    key_am = []
    key_cell = []
    for option in cols.attrs['options']:
        if option not in settings.RENEWABLES_OPTIONS:
            continue
        am_name = tools.am_name_snake_case(option)
        exist_r = inputs.exist_renewable_solar_r if option == "Utility Solar PV" else inputs.exist_renewable_wind_r   # the total across ALL data years: the ceiling never decreases between periods, so lb(t) <= ceiling always holds
        on_option = (am_idx == cols.attrs['options'].index(option)).astype(np.float32)          # 1 on the option's columns
        has_option = side.by_cell @ on_option != 0                                            # the cells holding a column of the option ...
        row_cells = np.flatnonzero(has_option & (exist_r != 0))                                  # ... and existing capacity (none -> no ceiling row): one row each, ascending
        if not row_cells.size:
            continue
        blocks.append(side.by_cell[row_cells] @ sparse.diags(on_option))                     # the support at those cells, over the option's columns
        rhs.append(np.maximum(ag_mask[row_cells] - exist_r[row_cells], 0.0))                    # cell space left for simulated capacity
        names += [f"const_{am_name}_solvable_ub_{r}".replace(" ", "_") for r in row_cells]
        key_am += [cols.attrs['options'].index(option)] * row_cells.size
        key_cell.append(row_cells)
    if not blocks:
        return None
    A, rhs, _ = contract(sparse.vstack(blocks, format='csr'), np.concatenate(rhs))
    return make_part('renewable_ceiling', 'ag_mgt_ub', dict(am_idx=key_am, cell=np.concatenate(key_cell)), A, rhs, '<', names)


def add_cell_usage(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Every cell's ag + non-ag shares sum to its base-year agricultural proportion."""
    block_range = cols.attrs['block_range']
    row_cells = cols['cell'].values[slice(*block_range['cell_usage'])]   # one row per slack, in the slack block's (cell) order
    takes_space = np.zeros(cols.attrs['n_all'], dtype=np.float32)        # 1 on the ag, non-ag and slack columns: what a cell's shares sum over
    for block in ('ag', 'nonag', 'cell_usage'):
        takes_space[slice(*block_range[block])] = 1.0
    A = side.by_cell[row_cells] @ sparse.diags(takes_space)
    # Ranged, not ==: presolve folds the node-balance rows into this one and compares two constants summed
    # along different float32 paths (up to ~1.75x FeasibilityTol apart) with NO tolerance. The +-10x Ftol band
    # absorbs that; conservation still pins the cell total, so the band is not exploitable.
    hi = inputs.ag_mask_proportion_r[row_cells].astype(np.float64) + 10 * settings.FEASIBILITY_TOLERANCE   # the top of the band (widened before the band is applied)
    A, hi, _ = contract(A, hi)
    return make_part('cell_usage', 'cell_usage', dict(cell=row_cells), A, hi, '=',
                     [f"const_cell_usage_{cell}" for cell in row_cells])


def add_ag_mgt_link(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Ag-management variables cannot exceed their agricultural variable: one row per (am, land use, lm,
    cell) with an ag column — X_am − X_ag ≤ 0, or X_ag ≥ 0 where the am column does not exist. The rows are the
    ag columns of (m, j), read off the ag column-id grid; every am column sits on the row of its cell."""
    n_all   = cols.attrs['n_all']
    options = cols.attrs['options']
    cell    = cols['cell'].values
    col_m   = cols['m'].values
    row_idx = []
    col_idx = []
    vals = []
    senses = []
    names = []
    key_am = []
    key_lu = []
    key_lm = []
    key_cell = []
    n_rows = 0
    for (option, j_idx) in side.valid_am:
        j = inputs.agman2lu[option][j_idx]
        run = slice(*cols.attrs['am_range'][(option, j_idx)])
        slot_cols = np.arange(run.start, run.stop)                                          # the slot's am columns, dry then irr
        for m, lm in ((0, 'dry'), (1, 'irr')):
            ag_of_cell = side.col_ag_mjr[m, j]                                              # the ag column of (m, j) at every cell, -1 none
            has_ag = ag_of_cell >= 0
            ag_cols = ag_of_cell[has_ag]                                                    # one row each, cells ascending
            row_of_cell = np.cumsum(has_ag) - 1                                             # the row of every cell with an ag column
            am_cols = slot_cols[col_m[run] == m]                                            # the slot's am columns at m ...
            host = row_of_cell[cell[am_cols]]                                               # ... each on the row of the ag column it sits on (every am cell has one)
            has_am = np.zeros(ag_cols.size, dtype=bool)                                     # no am column: GBF2-excluded or savanna-ineligible cell
            has_am[host] = True
            # X_ag: −1 on the '<' rows (X_am − X_ag ≤ 0), +1 on the '>' rows (X_ag ≥ 0)
            row_idx.append(n_rows + np.arange(ag_cols.size))
            col_idx.append(ag_cols)
            vals.append(np.where(has_am, -1.0, 1.0))
            # X_am: +1 on its host's row
            row_idx.append(n_rows + host)
            col_idx.append(am_cols)
            vals.append(np.ones(am_cols.size))
            senses.append(np.where(has_am, '<', '>'))
            names += [f"const_ag_mam_{lm}_usage_{option}_{j}_{r}".replace(" ", "_") for r in cell[ag_cols]]
            key_am.append(np.full(ag_cols.size, options.index(option), dtype=np.int32))
            key_lu.append(np.full(ag_cols.size, j, dtype=np.int32))
            key_lm.append(np.full(ag_cols.size, m, dtype=np.int8))
            key_cell.append(cell[ag_cols])
            n_rows += ag_cols.size
    A, _, _ = contract(sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_rows, n_all)))
    return make_part('ag_mgt_link', 'ag_mgt_link',
                     dict(am_idx=np.concatenate(key_am), j=np.concatenate(key_lu), m=np.concatenate(key_lm), cell=np.concatenate(key_cell)),
                     A, np.zeros(n_rows), np.concatenate(senses), names)


def add_ag_mgt_adoption(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Adoption limits: one row per (am, land use), Σ am columns − limit · Σ ag columns ≤ 0
    (Σam ≤ limit · Σag with the RHS moved to the LHS); zero coefficients (limit = 0) are dropped. The ag columns
    of the land use are read off the ag column-id grid."""
    n_all   = cols.attrs['n_all']
    options = cols.attrs['options']
    row_idx = []
    col_idx = []
    vals = []
    names = []
    key_am = []
    key_lu = []
    for row, (option, j_idx) in enumerate(side.valid_am):
        j = inputs.agman2lu[option][j_idx]
        adoption_limit = float(np.float64(inputs.ag_man_limits[option][j]))
        am_cols = np.arange(*cols.attrs['am_range'][(option, j_idx)])                                                # the slot's am columns, both lm
        ag_of_cell = side.col_ag_mjr[:, j]                                                                           # (lm, cell): the ag columns of j, -1 none
        ag_cols = ag_of_cell[ag_of_cell >= 0]                                                                        # both lm, dry first, cells ascending
        row_idx += [np.full(am_cols.size, row), np.full(ag_cols.size, row)]
        col_idx += [am_cols, ag_cols]
        vals += [np.ones(am_cols.size), np.full(ag_cols.size, -adoption_limit)]
        names.append(f"const_ag_mam_adoption_limit_{option}_{j}".replace(" ", "_"))
        key_am.append(options.index(option))
        key_lu.append(j)
    n_rows = len(names)
    A, _, _ = contract(sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_rows, n_all)))
    return make_part('ag_mgt_adoption', 'ag_mgt_adopt', dict(am_idx=key_am, j=key_lu),
                     A, np.zeros(n_rows), '<', names)


# ═══════════════════════════ the demand rows ═══════════════════════════

def add_demand(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Hard demand constraints: one per-commodity quantity row over the accounting columns, used once under
    '=' where the DEMAND_BOUNDS lb == ub, else twice — under '>' lb and '<' ub. Returns the part and, beside
    it, the UNSCALED production block the post-solve reads (the demand rows are its rescaled copies)."""
    print("│   ├── Adding <hard> demand constraints (equality where lb==ub, else lower + upper)...")
    n_all      = cols.attrs['n_all']
    nlms       = inputs.nlms
    n_lu       = inputs.n_ag_lus
    n_nonag_lu = inputs.n_nonag_lus
    ncms       = inputs.ncms
    cell       = cols['cell'].values
    col_m      = cols['m'].values
    row_idx = []
    col_idx = []
    vals = []

    def put(commodity_coeffs: np.ndarray, columns: np.ndarray):
        """One (commodity × column) coefficient block into the COO lists — its nonzero support; the contract drops."""
        for c_idx in range(ncms):
            keep = commodity_coeffs[c_idx] != 0
            row_idx.append(np.full(int(keep.sum()), c_idx, dtype=np.int32))
            col_idx.append(columns[keep])
            vals.append(commodity_coeffs[c_idx][keep])

    # ── the per-commodity LHS (q_block): the ag columns per (m, land use) — off the ag column-id grid — and the
    #    ag-mgt columns per (slot, m) carry jc[c, cell] = Σ_p pr2cm[c, p] · q[m, cell, p] over the land use's
    #    active products; the non-ag columns per k — off the non-ag grid — carry non_ag_q_crk[c, cell, k] ──
    for lu in range(n_lu):
        active_p = np.where(inputs.lu2pr_pj[:, lu])[0]
        if not active_p.size:
            continue
        for lm in range(nlms):
            cells = np.flatnonzero(side.col_ag_mjr[lm, lu] >= 0)                         # the cells with an ag column of (lm, lu), ascending
            if cells.size:
                put(inputs.pr2cm_cp[:, active_p] @ inputs.ag_q_mrp[lm, cells, :][:, active_p].T, side.col_ag_mjr[lm, lu, cells])
    for (option, j_idx) in side.valid_am:
        lu = inputs.agman2lu[option][j_idx]
        active_p = np.where(inputs.lu2pr_pj[:, lu])[0]
        if not active_p.size:
            continue
        run = slice(*cols.attrs['am_range'][(option, j_idx)])
        for lm in range(nlms):
            group = np.arange(run.start, run.stop)[col_m[run] == lm]                    # the slot's am columns at lm
            if group.size:
                put(inputs.pr2cm_cp[:, active_p] @ inputs.ag_man_q_mrp[option][lm, cell[group], :][:, active_p].T, group)
    for lu in range(n_nonag_lu):
        cells = np.flatnonzero(side.col_nonag_kr[lu] >= 0)                                # the cells with a non-ag column of lu, ascending
        if cells.size:
            put(inputs.non_ag_q_crk[:, cells, lu], side.col_nonag_kr[lu, cells])
    q_block = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(ncms, n_all))
    q_block.sum_duplicates()
    q_block, _, _ = contract(q_block)                                    # the drop, unscaled: production reporting reads this block

    # ── the bound rows: the LHS row once under '=' where lb == ub, else twice under '>' lb and '<' ub ──
    lhs_row = []
    senses = []
    rhs = []
    names = []
    key_commodity = []
    key_bound = []
    for c_idx, c_name in enumerate(inputs.commodity_names):
        lb, ub = settings.DEMAND_BOUNDS[c_name]
        demand = inputs.limits['demand'][c_idx]
        bounds = [('eq', '=', lb)] if lb == ub else [('lower', '>', lb), ('upper', '<', ub)]
        for bound, sense, factor in bounds:
            lhs_row.append(c_idx)
            senses.append(sense)
            rhs.append(demand * factor)
            names.append(f"demand_hard_bound_{bound}[{c_idx}]")
            key_commodity.append(c_idx)
            key_bound.append(bound)
    A, rhs, scale = contract(q_block[lhs_row], rhs, rescale=True)        # row rescale, factors kept
    part = make_part('demand', 'demand', dict(commodity=key_commodity, bound=key_bound),
                     A, rhs, np.array(senses, dtype=object), names, scale)
    return part, q_block


# ═══════════════════════════ the policy rows ═══════════════════════════

def add_ghg(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Hard GHG emissions cap: one global row over land-use, ag-management, non-ag and
    transition-arc emissions, Σ ghg · X ≤ limit − offland."""
    if settings.GHG_EMISSIONS_LIMITS == "off":
        print("│   ├── TURNING OFF GHG emissions constraints ...")
        return None
    ghg_limit_raw = inputs.limits["ghg"]
    print(f"│   ├── Adding <hard> constraints for GHG emissions: {ghg_limit_raw:,.0f} tCO2e")
    
    n_all = cols.attrs['n_all']

    # land-use, ag-management and non-ag emissions on the accounting columns
    coeff = gather(cols, side, inputs.ag_g_mrj, inputs.ag_man_g_mrj, inputs.non_ag_g_rk)
    keep = coeff != 0                                                    # the nonzero support; the contract drops the rest
    col_idx = [np.flatnonzero(keep)]
    vals = [coeff[keep]]

    # transition emissions on the ag → ag arcs: per source, its emission array masked is its run of the block
    for src, mask in side.valid_ag2ag.items():
        arc_ghg = inputs.trans_ghg_ag2ag[src][mask]
        keep = arc_ghg != 0
        col_idx.append(np.arange(*cols.attrs['ag2ag_range'][src])[keep])
        vals.append(arc_ghg[keep])

    col_idx = np.concatenate(col_idx)
    vals = np.concatenate(vals)
    row = sparse.csr_matrix((vals, (np.zeros(col_idx.size, dtype=np.int32), col_idx)), shape=(1, n_all))
    row.sum_duplicates()
    row.sort_indices()
    rhs = np.asarray(ghg_limit_raw - inputs.offland_ghg, dtype=np.float64).ravel()   # offland_ghg: 1-element array
    A, rhs, scale = contract(row, rhs, rescale=True)                     # drop + row rescale, factor kept
    return make_part('ghg', 'ghg', {}, A, rhs, '<', ["ghg_emissions_limit_ub"], scale)


def add_GBF2(inputs: RowInputs, cols: xr.Dataset, bio_S: sparse.csr_matrix):
    """GBF2 priority degraded areas: GBF2_mask_area_r as the one weight row times the bio contribution laid on the
    support; the mask area is ZERO off-mask, so off-mask columns get no entry. One row."""
    if settings.GBF2_TARGET == "off":
        print("│   │   ├── TURNING OFF constraints for biodiversity GBF 2...")
        return None
    print(f'│   │   ├── Adding constraints for biodiversity GBF 2: {inputs.limits["GBF2"]:15,.0f}')
    A, rhs, scale = contract(weight_rows([inputs.GBF2_mask_area_r], inputs.ncells) @ bio_S, [inputs.limits["GBF2"]], rescale=True)
    return make_part('GBF2', 'bio_gbf2', {}, A, rhs, '>',
                     ["bio_GBF2_priority_degraded_area_limit"], scale)


def add_GBF3_NVIS(inputs: RowInputs, cols: xr.Dataset, bio_S: sparse.csr_matrix):
    """GBF3 major vegetation groups — NVIS groups, or IBRA bioregions when GBF3_NVIS_REGION_MODE selects them
    upstream: one row per (region, group) with a target and at least one cell."""
    if settings.GBF3_NVIS_TARGET == "off":
        print("│   │   ├── TURNING OFF constraints for biodiversity GBF 3 NVIS")
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 3 NVIS...")
    layers = inputs.GBF3_NVIS_pre_1750_area_vr                           # xr [group, cell]
    targets = inputs.limits["GBF3_NVIS"]
    region_of_cell = inputs.region_NRM_names_r
    weights = []
    rhs = []
    names = []
    kept = []
    for region, group in inputs.GBF3_NVIS_region_group:
        target = targets.sel(layer=(region, group)).item()
        if target < 0:                                                   # GBF3 still adds a row for a ZERO target
            continue
        weight_row = layers.sel(group=group, drop=True).data
        if region != "AUSTRALIA":                                        # NRM scope: mask the cells outside the region
            weight_row = np.where(region_of_cell == region, weight_row, 0)
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        rhs.append(target)
        names.append(f"bio_GBF3_NVIS_limit_{region}_{group}".replace(" ", "_"))
        kept.append((region, group))
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(inputs.GBF3_NVIS_region_group) - len(kept)} skipped")
    if not weights:
        return None
    A, rhs, scale = contract(weight_rows(weights, inputs.ncells) @ bio_S, rhs, rescale=True)
    return make_part('GBF3_NVIS', 'bio_nvis',
                     dict(region=[region for region, _ in kept], item=[group for _, group in kept]),
                     A, rhs, '>', names, scale)


def add_GBF4_SNES(inputs: RowInputs, cols: xr.Dataset, bio_S: sparse.csr_matrix):
    """GBF4 species of national environmental significance: one row per (region, species, presence) with a
    POSITIVE target and at least one cell."""
    if settings.GBF4_TARGET_SNES == 'off':
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 SNES...')
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 4 SNES ...")
    layers = inputs.GBF4_SNES_pre_1750_area_sr                           # xr [layer=(species, presence), cell]
    targets = inputs.limits["GBF4_SNES"]
    region_of_cell = inputs.region_NRM_names_r
    weights = []
    rhs = []
    names = []
    kept = []
    for region, species, presence in inputs.GBF4_SNES_region_species:
        target = targets.sel(layer=(region, species, presence)).item()
        if target <= 0:                                                  # GBF4 skips a ZERO target
            continue
        weight_row = layers.sel(layer=(species, presence), drop=True).values
        if region != "AUSTRALIA":                                        # NRM scope: mask the cells outside the region
            weight_row = np.where(region_of_cell == region, weight_row, 0)
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        rhs.append(target)
        names.append(f"bio_GBF4_SNES_limit_{region}_{species}_{presence}".replace(" ", "_"))
        kept.append((region, species, presence))
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(inputs.GBF4_SNES_region_species) - len(kept)} skipped")
    if not weights:
        return None
    A, rhs, scale = contract(weight_rows(weights, inputs.ncells) @ bio_S, rhs, rescale=True)
    return make_part('GBF4_SNES', 'bio_snes',
                     dict(region=[key[0] for key in kept], item=[key[1] for key in kept], presence=[key[2] for key in kept]),
                     A, rhs, '>', names, scale)


def add_GBF4_ECNES(inputs: RowInputs, cols: xr.Dataset, bio_S: sparse.csr_matrix):
    """GBF4 ecological communities of national environmental significance: one row per (region, community,
    presence) with a POSITIVE target and at least one cell."""
    if settings.GBF4_TARGET_ECNES == 'off':
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 ECNES...')
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 4 ECNES ...")
    layers = inputs.GBF4_ECNES_pre_1750_area_sr                          # xr [layer=(community, presence), cell]
    targets = inputs.limits["GBF4_ECNES"]
    region_of_cell = inputs.region_NRM_names_r
    weights = []
    rhs = []
    names = []
    kept = []
    for region, community, presence in inputs.GBF4_ECNES_region_species:
        target = targets.sel(layer=(region, community, presence)).item()
        if target <= 0:                                                  # GBF4 skips a ZERO target
            continue
        weight_row = layers.sel(layer=(community, presence), drop=True).values
        if region != "AUSTRALIA":                                        # NRM scope: mask the cells outside the region
            weight_row = np.where(region_of_cell == region, weight_row, 0)
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        rhs.append(target)
        names.append(f"bio_GBF4_ECNES_limit_{region}_{community}_{presence}".replace(" ", "_"))
        kept.append((region, community, presence))
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(inputs.GBF4_ECNES_region_species) - len(kept)} skipped")
    if not weights:
        return None
    A, rhs, scale = contract(weight_rows(weights, inputs.ncells) @ bio_S, rhs, rescale=True)
    return make_part('GBF4_ECNES', 'bio_ecnes',
                     dict(region=[key[0] for key in kept], item=[key[1] for key in kept], presence=[key[2] for key in kept]),
                     A, rhs, '>', names, scale)


def add_GBF8(inputs: RowInputs, cols: xr.Dataset, bio_S: sparse.csr_matrix):
    """GBF8 species conservation under climate change: one row per (region, species) with a POSITIVE target
    and at least one cell."""
    if settings.GBF8_TARGET == "off":
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 8 ...')
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 8 ...")
    layers = inputs.GBF8_pre_1750_area_sr                                # xr [species, cell]
    targets = inputs.limits["GBF8"]
    region_of_cell = inputs.region_NRM_names_r
    weights = []
    rhs = []
    names = []
    kept = []
    for region, species in inputs.GBF8_region_species:
        target = targets.sel(layer=(region, species)).item()
        if target <= 0:                                                  # GBF8 skips a ZERO target
            continue
        weight_row = layers.sel(species=species, drop=True).data
        if region != "AUSTRALIA":                                        # NRM scope: mask the cells outside the region
            weight_row = np.where(region_of_cell == region, weight_row, 0)
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        rhs.append(target)
        names.append(f"bio_GBF8_limit_{region}_{species}".replace(" ", "_"))
        kept.append((region, species))
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(inputs.GBF8_region_species) - len(kept)} skipped")
    if not weights:
        return None
    A, rhs, scale = contract(weight_rows(weights, inputs.ncells) @ bio_S, rhs, rescale=True)
    return make_part('GBF8', 'bio_gbf8',
                     dict(region=[region for region, _ in kept], item=[species for _, species in kept]),
                     A, rhs, '>', names, scale)


def add_regional_adoption_ag(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Per-(region, ag land use) caps ('on' mode): Σ real_area[r] · X_ag over the region's columns ≤ cap — the ag
    columns of the land use sitting in the cap's cells (each cap carries its own cell set: its region is whichever
    layer REGIONAL_ADOPTION_ZONE picks), each weighted by its cell's hectares. Hectares are NOT rescaled — the
    shadow-price reader assumes scale 1."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        print("│   │   └── TURNING OFF constraints for regional adoption ...")
        return None
    n_all = cols.attrs['n_all']
    in_ag = np.zeros(n_all, dtype=bool)
    in_ag[slice(*cols.attrs['block_range']['ag'])] = True
    j = cols['j'].values
    cell = cols['cell'].values
    hectares = inputs.real_area[cell].astype(np.float32)                 # the hectares a column's whole share stands for
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits["ag_regional_adoption"]:
        name = f"reg_adopt_limit_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(inputs.ncells, dtype=bool)                                    # the cap's cells ...
        in_region[reg_cells] = True
        on = in_ag & (j == lu_code) & in_region[cell]                                    # ... and the land use's ag columns in them
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(hectares[on])
        rhs.append(area_limit_ha)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not names:
        return None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, _ = contract(A, rhs)
    return make_part('regional_adoption_ag', 'adopt_ag',
                     dict(region=[reg_id for reg_id, _ in keys], j=[lu_code for _, lu_code in keys]),
                     A, rhs, '<', names)


def add_regional_adoption_nonag(inputs: RowInputs, cols: xr.Dataset, side: ColSide, relax: float):
    """Per-(region, non-ag land use) caps ('on' mode), with the per-year relaxation on the RHS — the non-ag columns
    of the land use sitting in the cap's cells, each weighted by its cell's hectares."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    n_all = cols.attrs['n_all']
    in_nonag = np.zeros(n_all, dtype=bool)
    in_nonag[slice(*cols.attrs['block_range']['nonag'])] = True
    k = cols['k'].values
    cell = cols['cell'].values
    hectares = inputs.real_area[cell].astype(np.float32)                 # the hectares a column's whole share stands for
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption") or []:
        name = f"reg_adopt_limit_non_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(inputs.ncells, dtype=bool)                                    # the cap's cells ...
        in_region[reg_cells] = True
        on = in_nonag & (k == lu_code) & in_region[cell]                                 # ... and the land use's non-ag columns in them
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(hectares[on])
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not names:
        return None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, _ = contract(A, rhs)
    return make_part('regional_adoption_nonag', 'adopt_nonag',
                     dict(region=[reg_id for reg_id, _ in keys], k=[lu_code for _, lu_code in keys]),
                     A, rhs, '<', names)


def add_regional_adoption_nonag_sum(inputs: RowInputs, cols: xr.Dataset, side: ColSide, relax: float):
    """SUM-of-non-ag caps ('NON_AG_CAP' mode): every non-ag land use in a region together, with the relaxation —
    every non-ag column sitting in the cap's cells (an NRM region or a state, per REGIONAL_ADOPTION_NON_AG_REGION),
    weighted by its cell's hectares."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    n_all = cols.attrs['n_all']
    in_nonag = np.zeros(n_all, dtype=bool)
    in_nonag[slice(*cols.attrs['block_range']['nonag'])] = True
    cell = cols['cell'].values
    hectares = inputs.real_area[cell].astype(np.float32)                 # the hectares a column's whole share stands for
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    keys = []
    for reg_id, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption_sum") or []:
        name = f"reg_adopt_limit_non_ag_sum_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(inputs.ncells, dtype=bool)                                    # the cap's cells ...
        in_region[reg_cells] = True
        on = in_nonag & in_region[cell]                                                  # ... and every non-ag column in them
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(hectares[on])
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append(reg_id)
    if not names:
        return None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, _ = contract(A, rhs)
    return make_part('regional_adoption_nonag_sum', 'nonag_cap', dict(region=keys), A, rhs, '<', names)


def add_water(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Water net-yield limits: one row per water region — the columns whose water-region label is the region, each
    carrying its net yield (which can be NEGATIVE; the contract's drop sees the raw coefficient). The block-by-block
    union behind this (the input filtered by ``region2cell``, the block's run by ``region2col``) is hoisted: gather the
    six blocks once, then mask that vector by the label per region — same numbers, one gather."""
    if settings.WATER_LIMITS != "on":
        print("│   ├── TURNING OFF water usage constraints ...")
        return None
    print("│   ├── Adding constraints for water usage limits...")
    n_all = cols.attrs['n_all']
    coeff = gather(cols, side, inputs.ag_w_mrj, inputs.ag_man_w_mrj, inputs.non_ag_w_rk)
    region_of_col = side.region2col['water_region'].values
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    region_ids = []
    for region_id, water_limit_raw in inputs.limits["water"].items():
        region_name = inputs.water_region_names[region_id]
        print(f"│   │   ├── target (inside LUTO study area) is {water_limit_raw:15,.0f} ML for {region_name}")
        on = (region_of_col == region_id) & (coeff != 0)                                  # the region's columns with a net yield
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(coeff[on])
        rhs.append(water_limit_raw)
        names.append(f"water_yield_limit_{region_name}".replace(" ", "_"))
        region_ids.append(region_id)
    if not names:
        return None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, scale = contract(A, rhs, rescale=True)
    return make_part('water', 'water', dict(region=region_ids), A, rhs, '>', names, scale)


def add_renewable(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """State-level renewable generation targets: one row per (state, type) — the type's ag-mgt columns whose state
    label is the state, outside the excluded cells, each carrying its cell's yield; RHS = target − existing capacity."""
    if not any(settings.RENEWABLES_OPTIONS.values()):
        print("│   ├── TURNING OFF renewable energy constraints ...")
        return None
    print("│   ├── Adding constraints for renewable energy production targets ...")
    re_types = {
        'Utility Solar PV': dict(energy_r=inputs.renewable_solar_r, gbf2_mask_idx=inputs.mask_gbf2_solar, mnes_mask_idx=inputs.mask_mnes_solar),
        'Onshore Wind':     dict(energy_r=inputs.renewable_wind_r,  gbf2_mask_idx=inputs.mask_gbf2_wind,  mnes_mask_idx=inputs.mask_mnes_wind),
    }
    region_state_name2idx = dict(inputs.region_state_name2idx)                # local copy: pop() must not mutate data's dict
    act_code = region_state_name2idx.pop('Australian Capital Territory')
    n_all        = cols.attrs['n_all']
    cell         = cols['cell'].values
    j            = cols['j'].values
    am_idx       = cols['am_idx'].values
    state_of_col = side.region2col['state'].values
    in_ag        = np.zeros(n_all, dtype=bool)
    in_ag[slice(*cols.attrs['block_range']['ag'])] = True

    # ── per type: its columns' yield, and the columns the exclusion masks keep out ──
    energy_of_type = {}
    excluded_of_type = {}
    for option, re_data in re_types.items():
        if option not in cols.attrs['options']:
            continue
        on_type = am_idx == cols.attrs['options'].index(option)
        energy_of_type[option] = np.where(on_type, re_data['energy_r'][cell], np.float32(0.0)).astype(np.float32)   # float32 yield per cell
        excluded_r = np.zeros(inputs.ncells, dtype=bool)
        if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
            excluded_r[re_data['gbf2_mask_idx']] = True
        if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
            excluded_r[re_data['mnes_mask_idx']] = True
        excluded_of_type[option] = excluded_r[cell]

    # ── one row per (state, type) with an allowed column of the type's host land uses ──
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    key_am = []
    key_state = []
    for state_name, state_code in region_state_name2idx.items():
        in_state = state_of_col == state_code
        if state_name == 'New South Wales':                              # ACT counts toward the NSW+ACT target
            in_state |= state_of_col == act_code
        print(f"│   │   ├── Adding renewable energy constraints for {state_name} ...")
        for am, re_data in re_types.items():
            if not settings.AG_MANAGEMENTS[am]:
                continue
            target_raw = inputs.limits[f"renewable_{am}"][state_name]
            exist_power_mwh = inputs.limits[f"renewable_{am}_exist"][state_name]
            print(f"│   │   │   ├── target for {am} is {target_raw:5,.0f} MWh  (existing: {exist_power_mwh:5,.0f} MWh)")
            allowed = in_state & ~excluded_of_type[am]                   # the state's columns outside the excluded cells
            # row-inclusion rule (NOT a coefficient test): the row exists iff an allowed cell holds an ag column of a
            # compatible land use — even if every coefficient there turns out to be sub-floor
            if not (allowed & in_ag & np.isin(j, inputs.agman2lu[am])).any():
                continue
            on = allowed & (energy_of_type[am] != 0)                     # the type's columns in the state, with a yield
            row_idx.append(np.full(int(on.sum()), len(names)))
            col_idx.append(np.flatnonzero(on))
            vals.append(energy_of_type[am][on])
            rhs.append(target_raw - exist_power_mwh)                     # raw MWh; row-rescaled below
            names.append(f"renewable_{am}_target_{state_name}".replace(" ", "_"))
            key_am.append(cols.attrs['options'].index(am))
            key_state.append(state_name)
    if not names:
        return None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, scale = contract(A, rhs, rescale=True)
    return make_part('renewable', 'renewable', dict(am_idx=key_am, state=key_state), A, rhs, '>', names, scale)


# ═══════════════════════════ the flow rows ═══════════════════════════
#
#   group the arcs by their source's place in the base grid   →  source cap:   Σ out ≤ base              (a ≤ row per source node)
#   look every column and arc up on the grid of its node       →  node balance: X = base + Σ in − Σ out   (an = row per node)
#   the inflow cap is not a row: X's own ub (the transition upper bound) and the cell-usage row bound it.

def add_source_cap_ag(cols: xr.Dataset, side: ColSide):
    """Source cap, ag sources: the arcs leaving an ag node (ag2ag ∪ ag2nonag) grouped by where their source sits in
    the (lm, lu, cell) grid; each group's arcs sum to at most the source's base share, Σ out ≤ base[from_m, from_j, r]."""
    # bounds the arc columns (some flow costs are negative) and rules out pass-through
    print("│   ├── Adding source-cap (Σ out ≤ base) constraints...")
    n_all       = cols.attrs['n_all']
    block_range = cols.attrs['block_range']
    arcs        = np.concatenate([np.arange(*block_range['ag2ag']), np.arange(*block_range['ag2nonag'])])   # the arcs leaving an ag node
    from_m      = cols['from_m'].values[arcs]
    from_j      = cols['from_j'].values[arcs]
    local_r     = cols['local_r'].values[arcs]
    cell        = cols['cell'].values[arcs]
    # one row per source, ascending in the (lm, lu, cell) grid; every arc leaving it gets a +1
    source = np.ravel_multi_index((from_m, from_j, cell), side.col_ag_mjr.shape)                     # the flat position of each arc's source in the grid
    _, first_arc, row_of_arc = np.unique(source, return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(first_arc.size, n_all))
    # the cap is the source's base on the table: a source is a holding above the noise floor, and the ag ub is raised
    # to that same floor, so every source has a column
    source_col = side.col_ag_mjr[from_m[first_arc], from_j[first_arc], cell[first_arc]]
    assert (source_col >= 0).all(), 'an ag source without an ag column: the source map and the ub floor disagree'
    rhs = cols['base'].values[source_col].astype(np.float64)
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_a_{m}_{j}_{r}" for m, j, r in zip(from_m[first_arc], from_j[first_arc], local_r[first_arc])]
    return make_part('source_cap_ag', 'flow_out',
                     dict(from_m=from_m[first_arc], from_j=from_j[first_arc], local_r=local_r[first_arc]),
                     A, rhs, '<', names)


def add_source_cap_nonag(cols: xr.Dataset, side: ColSide):
    """Source cap, non-ag sources: the arcs leaving a non-ag node (nonag2ag) grouped by where their source sits in
    the (nonag_lu, cell) grid; each group's arcs sum to at most the source's base share, Σ out ≤ base[from_k, r]."""
    n_all   = cols.attrs['n_all']
    arcs    = np.arange(*cols.attrs['block_range']['nonag2ag'])                                      # the arcs leaving a non-ag node
    if not arcs.size:
        return None
    from_k  = cols['from_k'].values[arcs]
    local_r = cols['local_r'].values[arcs]
    cell    = cols['cell'].values[arcs]
    source = np.ravel_multi_index((from_k, cell), side.col_nonag_kr.shape)                           # the flat position of each arc's source in the grid
    _, first_arc, row_of_arc = np.unique(source, return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(first_arc.size, n_all))
    # the cap is the source's base on the table: a source is a holding above the noise floor, and the non-ag ub is
    # raised to that same floor, so every source has a column
    source_col = side.col_nonag_kr[from_k[first_arc], cell[first_arc]]
    assert (source_col >= 0).all(), 'a non-ag source without a non-ag column: the source map and the ub floor disagree'
    rhs = cols['base'].values[source_col].astype(np.float64)
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_n_{k}_{r}" for k, r in zip(from_k[first_arc], local_r[first_arc])]
    return make_part('source_cap_nonag', 'flow_out', dict(from_k=from_k[first_arc], local_r=local_r[first_arc]), A, rhs, '<', names)


def add_node_balance_ag(cols: xr.Dataset, side: ColSide):
    """Node balance at the ag nodes, one row per ag column (the ag block, in its order):
    X_ag[m, r, j] = base_ag[m, r, j] + Σ in (ag2ag ∪ nonag2ag → (m, j)) − Σ out ((m, j) → ag2ag ∪ ag2nonag).
    Every column and arc finds the row of the ag node it lands on / leaves on the ag column-id grid; an entry whose
    node has no column (-1) is dropped. The inflow cap is X's own ub, not a row."""
    print("│   ├── Adding node-balance (X = base + Σin − Σout) constraints at the ag nodes...")
    n_all       = cols.attrs['n_all']
    m           = cols['m'].values
    j           = cols['j'].values
    from_m      = cols['from_m'].values
    from_j      = cols['from_j'].values
    cell        = cols['cell'].values
    block_range = cols.attrs['block_range']
    ag          = slice(*block_range['ag'])

    # ── the rows: one per ag column ──
    n_ag = ag.stop - ag.start
    ag_m, ag_j, ag_r = m[ag].astype(np.int64), j[ag].astype(np.int64), cell[ag].astype(np.int64)

    def ag_row(m_, j_, r_):
        """The row of the ag node (m, j, r): its column's place in the ag block, -1 where it has no column."""
        col = side.col_ag_mjr[m_, j_, r_]
        return np.where(col >= 0, col - ag.start, -1)

    # ── the entries: X on its own row, inflows −1 on the target's row, outflows +1 on the source's row ──
    row_idx = []
    col_idx = []
    vals = []

    def add(row, col, value):
        in_model = row >= 0                                              # no row (banned source / no X var): entry dropped
        row_idx.append(row[in_model].astype(np.int64))
        col_idx.append(col[in_model].astype(np.int64))
        vals.append(np.full(int(in_model.sum()), value))

    add(np.arange(n_ag), np.arange(ag.start, ag.stop), 1.0)              # X_ag on its own row
    arcs = np.arange(*block_range['ag2ag'])                              # ag → ag: in on the target's row, out of the source's
    add(ag_row(m[arcs], j[arcs], cell[arcs]), arcs, -1.0)
    add(ag_row(from_m[arcs], from_j[arcs], cell[arcs]), arcs, 1.0)
    arcs = np.arange(*block_range['ag2nonag'])                           # ag → non-ag: out of the source's row
    add(ag_row(from_m[arcs], from_j[arcs], cell[arcs]), arcs, 1.0)
    arcs = np.arange(*block_range['nonag2ag'])                           # non-ag → ag: in on the target's row
    add(ag_row(m[arcs], j[arcs], cell[arcs]), arcs, -1.0)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_ag, n_all))

    # ── rhs, names, keys ──
    A, rhs, _ = contract(A, cols['base'].values[ag].astype(np.float64))
    names = [f"bal_a_{m}_{j}_{r}" for m, j, r in zip(ag_m, ag_j, ag_r)]
    return make_part('node_balance_ag', 'flow_in', dict(m=ag_m, j=ag_j, cell=ag_r), A, rhs, '=', names)


def add_node_balance_nonag(cols: xr.Dataset, side: ColSide):
    """Node balance at the non-ag nodes, one row per non-ag column (the non-ag block, in its order; a disabled land
    use's column is fixed at zero, so its row only guards the inflow):
    X_nonag[r, k] = base_nonag[r, k] + Σ in (ag2nonag → k) − Σ out (k → nonag2ag).
    Every column and arc finds the row of the non-ag node it lands on / leaves on the non-ag column-id grid; an entry
    whose node has no column (-1) is dropped."""
    print("│   └── Adding node-balance (X = base + Σin − Σout) constraints at the non-ag nodes...")
    n_all       = cols.attrs['n_all']
    k           = cols['k'].values
    from_k      = cols['from_k'].values
    cell        = cols['cell'].values
    block_range = cols.attrs['block_range']
    nonag       = slice(*block_range['nonag'])

    # ── the rows: one per non-ag column ──
    n_nonag = nonag.stop - nonag.start
    if not n_nonag:
        return None
    nonag_k, nonag_r = k[nonag].astype(np.int64), cell[nonag].astype(np.int64)

    def nonag_row(k_, r_):
        """The row of the non-ag node (k, r): its column's place in the non-ag block, -1 where it has no column."""
        col = side.col_nonag_kr[k_, r_]
        return np.where(col >= 0, col - nonag.start, -1)

    # ── the entries: X on its own row, inflows −1 on the target's row, outflows +1 on the source's row ──
    row_idx = []
    col_idx = []
    vals = []

    def add(row, col, value):
        in_model = row >= 0                                              # no row (the node has no column): entry dropped
        row_idx.append(row[in_model].astype(np.int64))
        col_idx.append(col[in_model].astype(np.int64))
        vals.append(np.full(int(in_model.sum()), value))

    add(np.arange(n_nonag), np.arange(nonag.start, nonag.stop), 1.0)     # X_nonag on its own row
    arcs = np.arange(*block_range['ag2nonag'])                           # ag → non-ag: in on the target's row
    add(nonag_row(k[arcs], cell[arcs]), arcs, -1.0)
    arcs = np.arange(*block_range['nonag2ag'])                           # non-ag → ag: out of the source's row
    add(nonag_row(from_k[arcs], cell[arcs]), arcs, 1.0)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_nonag, n_all))

    # ── rhs, names, keys ──
    A, rhs, _ = contract(A, cols['base'].values[nonag].astype(np.float64))
    names = [f"bal_n_{k}_{r}" for k, r in zip(nonag_k, nonag_r)]
    return make_part('node_balance_nonag', 'flow_in', dict(k=nonag_k, cell=nonag_r), A, rhs, '=', names)
