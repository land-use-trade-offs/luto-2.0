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
        add_ag_mgt_link(cols, side),                                            # an ag-mgt column cannot exceed its ag column ...
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
    bio_S = side.support_rc @ sparse.diags(bio_coeff(inputs, cols)) if bio_on else None   # the biodiversity contribution laid on the support ONCE: every GBF family's rows are W @ bio_S
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
# that matrix, always the same way: GATHER a coefficient per column off the table's fields, lay it on the
# SUPPORT (``side.support_rc @ diags(c)``: the coefficient at (cell, column)) and multiply by the family's WEIGHT
# ROWS over cells (``W @ ...``: row i, column t = W[i, cell_t] · c_t, one float32 product per entry), and CONTRACT
# the stacked block in one loop over its rows — the SOLVER_COEFF_MIN drop, then (policy families only) the
# geomean row rescale and the floor again.

def gather(cols: xr.Dataset, ag_c_mrj, am_c_mrj: dict, nonag_c_rk) -> np.ndarray:
    """One family's coefficient at every column, float32 (zero off the accounting columns), read by what the
    column's fields say it is: ``k >= 0`` non-ag ``[cell, k]``; ``am_idx >= 0`` ag-mgt, its option's
    ``[m, cell, j_idx]``; else ag ``[m, cell, j]``."""
    n_terms = cols.attrs['n_terms']
    m = cols['m'].values[:n_terms]
    j = cols['j'].values[:n_terms]
    k = cols['k'].values[:n_terms]
    cell = cols['cell'].values[:n_terms]
    am_idx = cols['am_idx'].values[:n_terms]
    j_idx = cols['j_idx'].values[:n_terms]
    is_nonag, is_am = k >= 0, am_idx >= 0
    is_ag = ~is_nonag & ~is_am
    c = np.zeros(cols.attrs['n_all'], dtype=np.float32)
    c[:n_terms][is_ag] = ag_c_mrj[m[is_ag], cell[is_ag], j[is_ag]]
    c[:n_terms][is_nonag] = nonag_c_rk[cell[is_nonag], k[is_nonag]]
    for idx, option in enumerate(cols.attrs['options']):
        sel = am_idx == idx
        if sel.any():
            c[:n_terms][sel] = am_c_mrj[option][m[sel], cell[sel], j_idx[sel]]
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


def bio_coeff(inputs: RowInputs, cols: xr.Dataset) -> np.ndarray:
    """The biodiversity contribution at every column: the three streams ``gather`` reads — a scalar per ag land
    use, a per-cell array per (option, land use), a scalar per non-ag land use — as broadcast VIEWS (no copy),
    gathered ONCE for all five GBF families."""
    nlms, ncells = cols.attrs['nlms'], cols.attrs['ncells']
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
    return gather(cols, ag_c_mrj, am_c_mrj, nonag_c_rk)


# ═══════════════════════════ the am runs: the one group-by the ag-mgt rows share ═══════════════════════════

def am_slots(cols: xr.Dataset) -> tuple[list, np.ndarray]:
    """The (option, land use) slots in slot order, and the am block grouped by (slot, m) — ``ptr[slot · nlms + m]``
    bounds its run of table rows, cells ascending (a slot with no column has an empty run). The block is sorted by
    (slot, m, cell), so the runs are one ``searchsorted`` on that key."""
    am = slice(*cols.attrs['block_range']['am'])
    slots = [(option, lu) for option, lus in cols.attrs['agman2lu'].items() for lu in lus]
    nlms = cols.attrs['nlms']
    key = cols['slot'].values[am] * nlms + cols['m'].values[am]                                # ascending: the block is in (slot, m, cell) order
    return slots, am.start + np.searchsorted(key, np.arange(len(slots) * nlms + 1))


# ═══════════════════════════ get_obj: the objective coefficient of every column ═══════════════════════════

def get_obj(econ: EconomicInputs, cols: xr.Dataset, side: ColSide) -> xr.DataArray:
    """The objective coefficient of every column (on ``col``), as Gurobi takes it: the operating economics on
    the accounting columns and the transition costs, negated, on the arcs — raw AUD, float32 — through the
    coefficient contract (the SOLVER_COEFF_MIN drop), then scaled to million AUD and floored again, because
    the scaling can push a coefficient under the floor. Zero on the cell-usage slacks: they carry no cost."""
    m = cols['m'].values
    j = cols['j'].values
    k = cols['k'].values
    local_r = cols['local_r'].values

    # ── operating economics: one gather over the accounting columns (zero everywhere else) ──
    obj = gather(cols, econ.ag_obj_mrj, econ.ag_man_objs, econ.non_ag_obj_rk)

    # ── transition costs on the arcs with an ag target: per source run of the block, gathered from the source-keyed cost dicts ──
    for block, sources, flow_cost in (('ag2ag', side.sources_ag, econ.flow_cost_ag2ag),
                                      ('nonag2ag', side.sources_nonag, econ.flow_cost_nonag2ag)):
        src_ptr = cols.attrs['src_ptr'][block]
        for src, start, stop in zip(sources, src_ptr[:-1], src_ptr[1:]):
            run = slice(int(start), int(stop))
            obj[run] = -flow_cost[src][m[run], local_r[run], j[run]]

    # ── transition costs on the ag → non-ag arcs: the cost dict is keyed by target land use k ──
    src_ptr = cols.attrs['src_ptr']['ag2nonag']
    for src, start, stop in zip(side.sources_ag, src_ptr[:-1], src_ptr[1:]):
        run = slice(int(start), int(stop))
        cost_by_k = econ.flow_cost_ag2nonag[src]                               # {k: array(ncells_src)}
        to_k, cells, arc_obj = k[run], local_r[run], obj[run]                  # arc_obj: a view into obj
        for lu in np.unique(to_k):
            arc_obj[to_k == lu] = -cost_by_k[int(lu)][cells[to_k == lu]]

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
    for option in cols.attrs['agman2lu']:
        if option not in settings.RENEWABLES_OPTIONS:
            continue
        am_name = tools.am_name_snake_case(option)
        exist_r = inputs.exist_renewable_solar_r if option == "Utility Solar PV" else inputs.exist_renewable_wind_r   # the total across ALL data years: the ceiling never decreases between periods, so lb(t) <= ceiling always holds
        on_option = (am_idx == cols.attrs['options'].index(option)).astype(np.float32)          # 1 on the option's columns
        has_option = side.support_rc @ on_option != 0                                            # the cells holding a column of the option ...
        row_cells = np.flatnonzero(has_option & (exist_r != 0))                                  # ... and existing capacity (none -> no ceiling row): one row each, ascending
        if not row_cells.size:
            continue
        blocks.append(side.support_rc[row_cells] @ sparse.diags(on_option))                     # the support at those cells, over the option's columns
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
    A = side.support_rc[row_cells] @ sparse.diags(takes_space)
    # Ranged, not ==: presolve folds the node-balance rows into this one and compares two constants summed
    # along different float32 paths (up to ~1.75x FeasibilityTol apart) with NO tolerance. The +-10x Ftol band
    # absorbs that; conservation still pins the cell total, so the band is not exploitable.
    hi = inputs.ag_mask_proportion_r[row_cells].astype(np.float64) + 10 * settings.FEASIBILITY_TOLERANCE   # the top of the band (widened before the band is applied)
    A, hi, _ = contract(A, hi)
    return make_part('cell_usage', 'cell_usage', dict(cell=row_cells), A, hi, '=',
                     [f"const_cell_usage_{cell}" for cell in row_cells])


def add_ag_mgt_link(cols: xr.Dataset, side: ColSide):
    """Ag-management variables cannot exceed their agricultural variable: one row per (am, land use, lm,
    cell) with an ag column — X_am − X_ag ≤ 0, or X_ag ≥ 0 where the am column does not exist. The rows are the
    ag columns of (m, j), read off the ag column-id grid; every am column sits on the row of its cell."""
    n_all = cols.attrs['n_all']
    nlms = cols.attrs['nlms']
    cell = cols['cell'].values
    options = cols.attrs['options']
    slots, am_ptr = am_slots(cols)
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
    for slot, (option, j) in enumerate(slots):
        for m, lm in ((0, 'dry'), (1, 'irr')):
            ag_of_cell = side.col_ag_mjr[m, j]                                              # the ag column of (m, j) at every cell, -1 none
            has_ag = ag_of_cell >= 0
            ag_cols = ag_of_cell[has_ag]                                                    # one row each, cells ascending
            row_of_cell = np.cumsum(has_ag) - 1                                             # the row of every cell with an ag column
            am_cols = np.arange(am_ptr[slot * nlms + m], am_ptr[slot * nlms + m + 1])       # the slot's am columns at m ...
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
    n_all = cols.attrs['n_all']
    nlms = cols.attrs['nlms']
    slots, am_ptr = am_slots(cols)
    row_idx = []
    col_idx = []
    vals = []
    names = []
    key_am = []
    key_lu = []
    for row, (option, j) in enumerate(slots):
        adoption_limit = float(np.float64(inputs.ag_man_limits[option][j]))
        am_cols = np.arange(am_ptr[row * nlms], am_ptr[row * nlms + nlms])                                           # the slot's am columns, both lm (adjacent runs)
        ag_of_cell = side.col_ag_mjr[:, j]                                                                           # (lm, cell): the ag columns of j, -1 none
        ag_cols = ag_of_cell[ag_of_cell >= 0]                                                                        # both lm, dry first, cells ascending
        row_idx += [np.full(am_cols.size, row), np.full(ag_cols.size, row)]
        col_idx += [am_cols, ag_cols]
        vals += [np.ones(am_cols.size), np.full(ag_cols.size, -adoption_limit)]
        names.append(f"const_ag_mam_adoption_limit_{option}_{j}".replace(" ", "_"))
        key_am.append(cols.attrs['options'].index(option))
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
    n_all = cols.attrs['n_all']
    cell = cols['cell'].values
    nlms, n_lu, n_nonag_lu = cols.attrs['nlms'], cols.attrs['n_ag_lus'], cols.attrs['n_nonag_lus']
    slots, am_ptr = am_slots(cols)
    ncms = inputs.ncms
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
    for slot, (option, lu) in enumerate(slots):
        active_p = np.where(inputs.lu2pr_pj[:, lu])[0]
        if not active_p.size:
            continue
        for lm in range(nlms):
            group = np.arange(am_ptr[slot * nlms + lm], am_ptr[slot * nlms + lm + 1])
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
    m = cols['m'].values
    j = cols['j'].values
    local_r = cols['local_r'].values

    # land-use, ag-management and non-ag emissions on the accounting columns
    coeff = gather(cols, inputs.ag_g_mrj, inputs.ag_man_g_mrj, inputs.non_ag_g_rk)
    keep = coeff != 0                                                    # the nonzero support; the contract drops the rest
    col_idx = [np.flatnonzero(keep)]
    vals = [coeff[keep]]

    # transition emissions on the ag → ag arcs: per source run of the block, a float32 gather of the delta emissions
    src_ptr = cols.attrs['src_ptr']['ag2ag']
    for src, start, stop in zip(side.sources_ag, src_ptr[:-1], src_ptr[1:]):
        run = slice(int(start), int(stop))
        if run.stop == run.start:
            continue
        arc_ghg = inputs.trans_ghg_ag2ag[src][m[run], local_r[run], j[run]]
        keep = arc_ghg != 0
        col_idx.append(np.arange(run.start, run.stop)[keep])
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
    A, rhs, scale = contract(weight_rows([inputs.GBF2_mask_area_r], cols.attrs['ncells']) @ bio_S, [inputs.limits["GBF2"]], rescale=True)
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
    A, rhs, scale = contract(weight_rows(weights, cols.attrs['ncells']) @ bio_S, rhs, rescale=True)
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
    A, rhs, scale = contract(weight_rows(weights, cols.attrs['ncells']) @ bio_S, rhs, rescale=True)
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
    A, rhs, scale = contract(weight_rows(weights, cols.attrs['ncells']) @ bio_S, rhs, rescale=True)
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
    A, rhs, scale = contract(weight_rows(weights, cols.attrs['ncells']) @ bio_S, rhs, rescale=True)
    return make_part('GBF8', 'bio_gbf8',
                     dict(region=[region for region, _ in kept], item=[species for _, species in kept]),
                     A, rhs, '>', names, scale)


def add_regional_adoption_ag(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Per-(region, ag land use) caps ('on' mode): Σ real_area[r] · X_ag over the region's cells ≤ cap — the
    region's cell indicator as the weight row, times the land use's hectares laid on the support. Hectares are
    NOT rescaled — the shadow-price reader assumes scale 1."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        print("│   │   └── TURNING OFF constraints for regional adoption ...")
        return None
    ncells = cols.attrs['ncells']
    in_ag = np.zeros(cols.attrs['n_all'], dtype=bool)
    in_ag[slice(*cols.attrs['block_range']['ag'])] = True
    j = cols['j'].values
    hectares = inputs.real_area[cols['cell'].values].astype(np.float32)                 # the hectares a column's whole share stands for
    hectares_on_lu = {}                                                                  # {lu: its ag columns' hectares laid on the support}, built on first use
    blocks = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits["ag_regional_adoption"]:
        name = f"reg_adopt_limit_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        if lu_code not in hectares_on_lu:
            hectares_on_lu[lu_code] = side.support_rc @ sparse.diags(np.where(in_ag & (j == lu_code), hectares, np.float32(0.0)))
        in_region = np.zeros(ncells, dtype=np.float32)
        in_region[reg_cells] = 1.0
        blocks.append(weight_rows([in_region], ncells) @ hectares_on_lu[lu_code])
        rhs.append(area_limit_ha)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not blocks:
        return None
    A, rhs, _ = contract(sparse.vstack(blocks, format='csr'), rhs)
    return make_part('regional_adoption_ag', 'adopt_ag',
                     dict(region=[reg_id for reg_id, _ in keys], j=[lu_code for _, lu_code in keys]),
                     A, rhs, '<', names)


def add_regional_adoption_nonag(inputs: RowInputs, cols: xr.Dataset, side: ColSide, relax: float):
    """Per-(region, non-ag land use) caps ('on' mode), with the per-year relaxation on the RHS — the region's cell
    indicator times the land use's hectares laid on the support."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    ncells = cols.attrs['ncells']
    in_nonag = np.zeros(cols.attrs['n_all'], dtype=bool)
    in_nonag[slice(*cols.attrs['block_range']['nonag'])] = True
    k = cols['k'].values
    hectares = inputs.real_area[cols['cell'].values].astype(np.float32)                 # the hectares a column's whole share stands for
    hectares_on_lu = {}                                                                  # {lu: its non-ag columns' hectares laid on the support}, built on first use
    blocks = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption") or []:
        name = f"reg_adopt_limit_non_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        if lu_code not in hectares_on_lu:
            hectares_on_lu[lu_code] = side.support_rc @ sparse.diags(np.where(in_nonag & (k == lu_code), hectares, np.float32(0.0)))
        in_region = np.zeros(ncells, dtype=np.float32)
        in_region[reg_cells] = 1.0
        blocks.append(weight_rows([in_region], ncells) @ hectares_on_lu[lu_code])
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not blocks:
        return None
    A, rhs, _ = contract(sparse.vstack(blocks, format='csr'), rhs)
    return make_part('regional_adoption_nonag', 'adopt_nonag',
                     dict(region=[reg_id for reg_id, _ in keys], k=[lu_code for _, lu_code in keys]),
                     A, rhs, '<', names)


def add_regional_adoption_nonag_sum(inputs: RowInputs, cols: xr.Dataset, side: ColSide, relax: float):
    """SUM-of-non-ag caps ('NON_AG_CAP' mode): every non-ag land use in a region together, with the relaxation —
    the regions' cell indicators as the weight rows, times every non-ag column's hectares laid on the support."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    ncells = cols.attrs['ncells']
    in_nonag = np.zeros(cols.attrs['n_all'], dtype=bool)
    in_nonag[slice(*cols.attrs['block_range']['nonag'])] = True
    hectares = inputs.real_area[cols['cell'].values].astype(np.float32)                 # the hectares a column's whole share stands for
    regions = []
    rhs = []
    names = []
    keys = []
    for reg_id, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption_sum") or []:
        name = f"reg_adopt_limit_non_ag_sum_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(ncells, dtype=np.float32)
        in_region[reg_cells] = 1.0
        regions.append(in_region)
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append(reg_id)
    if not regions:
        return None
    hectares_on_nonag = side.support_rc @ sparse.diags(np.where(in_nonag, hectares, np.float32(0.0)))
    A, rhs, _ = contract(weight_rows(regions, ncells) @ hectares_on_nonag, rhs)
    return make_part('regional_adoption_nonag_sum', 'nonag_cap', dict(region=keys), A, rhs, '<', names)


def add_water(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """Water net-yield limits: one row per water region, the region's 0/1 float32 indicator as the
    weighting row over the accounting columns (off-region columns give q = 0 and are dropped)."""
    if settings.WATER_LIMITS != "on":
        print("│   ├── TURNING OFF water usage constraints ...")
        return None
    print("│   ├── Adding constraints for water usage limits...")
    coeff = gather(cols, inputs.ag_w_mrj, inputs.ag_man_w_mrj, inputs.non_ag_w_rk)
    weights = []
    rhs = []
    names = []
    region_ids = []
    for region_id, water_limit_raw in inputs.limits["water"].items():
        region_name = inputs.water_region_names[region_id]
        print(f"│   │   ├── target (inside LUTO study area) is {water_limit_raw:15,.0f} ML for {region_name}")
        indicator = np.zeros(cols.attrs['ncells'], dtype=np.float32)   # 1.0f x c == c, so the drop test sees the raw coefficient — which can be NEGATIVE
        indicator[inputs.water_region_indices[region_id]] = 1.0
        weights.append(indicator)
        rhs.append(water_limit_raw)
        names.append(f"water_yield_limit_{region_name}".replace(" ", "_"))
        region_ids.append(region_id)
    if not weights:
        return None
    A, rhs, scale = contract(weight_rows(weights, cols.attrs['ncells']) @ (side.support_rc @ sparse.diags(coeff)), rhs, rescale=True)
    return make_part('water', 'water', dict(region=region_ids), A, rhs, '>', names, scale)


def add_renewable(inputs: RowInputs, cols: xr.Dataset, side: ColSide):
    """State-level renewable generation targets: one row per (state, type) — the state's allowed-cells indicator
    as the weight row, times the type's yield laid on the support; RHS = target − existing capacity."""
    if not any(settings.RENEWABLES_OPTIONS.values()):
        print("│   ├── TURNING OFF renewable energy constraints ...")
        return None
    print("│   ├── Adding constraints for renewable energy production targets ...")
    re_types = {
        'Utility Solar PV': dict(energy_r=inputs.renewable_solar_r, gbf2_mask_idx=side.mask_gbf2_solar, mnes_mask_idx=side.mask_mnes_solar),
        'Onshore Wind':     dict(energy_r=inputs.renewable_wind_r,  gbf2_mask_idx=side.mask_gbf2_wind,  mnes_mask_idx=side.mask_mnes_wind),
    }
    region_state_name2idx = dict(inputs.region_state_name2idx)                # local copy: pop() must not mutate data's dict
    act_code = region_state_name2idx.pop('Australian Capital Territory')
    cell = cols['cell'].values
    am_idx = cols['am_idx'].values
    ncells = cols.attrs['ncells']

    # ── the yield per type laid on the support: energy_r at that type's ag-mgt columns, 0 on every other column ──
    energy_of_type = {}
    for option, re_data in re_types.items():
        if option in cols.attrs['options']:
            on_type = am_idx == cols.attrs['options'].index(option)
            energy_of_type[option] = side.support_rc @ sparse.diags(np.where(on_type, re_data['energy_r'][cell], np.float32(0.0)).astype(np.float32))   # float32 yield per cell

    # ── one row per (state, type) with an allowed cell that holds an ag column of a compatible land use ──
    has_ag_jr = (side.col_ag_mjr >= 0).any(axis=0)                       # (lu, cell): the cell has an ag column of the land use, either lm
    agman2lu = cols.attrs['agman2lu']
    blocks = []
    rhs = []
    names = []
    key_am = []
    key_state = []
    for state_name, state_code in region_state_name2idx.items():
        state_cells = np.where(inputs.region_state_r == state_code)[0]
        if state_name == 'New South Wales':                              # ACT counts toward the NSW+ACT target
            state_cells = np.union1d(state_cells, np.where(inputs.region_state_r == act_code)[0])
        print(f"│   │   ├── Adding renewable energy constraints for {state_name} ...")
        for am, re_data in re_types.items():
            if not settings.AG_MANAGEMENTS[am]:
                continue
            target_raw = inputs.limits[f"renewable_{am}"][state_name]
            exist_power_mwh = inputs.limits[f"renewable_{am}_exist"][state_name]
            print(f"│   │   │   ├── target for {am} is {target_raw:5,.0f} MWh  (existing: {exist_power_mwh:5,.0f} MWh)")
            allowed = np.zeros(ncells, dtype=np.float32)                 # the weighting row: 1 on the state's allowed cells
            allowed[state_cells] = 1.0
            if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS == True:
                allowed[re_data['gbf2_mask_idx']] = 0.0
            if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK == True:
                allowed[re_data['mnes_mask_idx']] = 0.0
            # cell-set row-inclusion rule (NOT a coefficient test): the row exists iff an allowed cell holds an ag
            # column of a compatible land use — even if every coefficient there turns out to be sub-floor
            if not (has_ag_jr[agman2lu[am]].any(axis=0) & (allowed > 0)).any():
                continue
            blocks.append(weight_rows([allowed], ncells) @ energy_of_type[am])
            rhs.append(target_raw - exist_power_mwh)                     # raw MWh; row-rescaled below
            names.append(f"renewable_{am}_target_{state_name}".replace(" ", "_"))
            key_am.append(cols.attrs['options'].index(am))
            key_state.append(state_name)
    if not blocks:
        return None
    A, rhs, scale = contract(sparse.vstack(blocks, format='csr'), rhs, rescale=True)
    return make_part('renewable', 'renewable', dict(am_idx=key_am, state=key_state), A, rhs, '>', names, scale)


# ═══════════════════════════ the flow rows ═══════════════════════════
#
#   group the arcs by their source's place in the base grid   →  source cap:   Σ out ≤ base              (a ≤ row per source node)
#   look every column and arc up on the grid of its node       →  node balance: X = base + Σ in − Σ out   (an = row per node)
#   the inflow cap is not a row: X's own ub (the transition upper bound) and the cell-usage row bound it.

def add_source_cap_ag(cols: xr.Dataset, side: ColSide):
    """Source cap, ag sources: the arcs leaving an ag node (ag2ag ∪ ag2nonag) grouped by where their source sits in
    the ag base grid; each group's arcs sum to at most its base share, Σ out ≤ base[from_m, from_j, r]."""
    # bounds the arc columns (some flow costs are negative) and rules out pass-through
    print("│   ├── Adding source-cap (Σ out ≤ base) constraints...")
    n_all = cols.attrs['n_all']
    block_range = cols.attrs['block_range']
    arcs = np.concatenate([np.arange(*block_range['ag2ag']), np.arange(*block_range['ag2nonag'])])   # the arcs leaving an ag node
    from_m = cols['from_m'].values[arcs]
    from_j = cols['from_j'].values[arcs]
    local_r = cols['local_r'].values[arcs]
    cell = cols['cell'].values[arcs]
    # one row per source, ascending in the base grid; every arc leaving it gets a +1
    source = np.ravel_multi_index((from_m, from_j, cell), side.ag_base_mjr.shape)                    # the flat position of each arc's source in the base grid
    _, first_arc, row_of_arc = np.unique(source, return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(first_arc.size, n_all))
    rhs = side.ag_base_mjr[from_m[first_arc], from_j[first_arc], cell[first_arc]].astype(np.float64)   # from the ag grid: a source with no X column still caps its outflow
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_a_{m}_{j}_{r}" for m, j, r in zip(from_m[first_arc], from_j[first_arc], local_r[first_arc])]
    return make_part('source_cap_ag', 'flow_out',
                     dict(from_m=from_m[first_arc], from_j=from_j[first_arc], local_r=local_r[first_arc]),
                     A, rhs, '<', names)


def add_source_cap_nonag(cols: xr.Dataset, side: ColSide):
    """Source cap, non-ag sources: the arcs leaving a non-ag node (nonag2ag) grouped by where their source sits in
    the non-ag base grid; Σ out ≤ base_nonag[from_k, r]."""
    n_all = cols.attrs['n_all']
    arcs = np.arange(*cols.attrs['block_range']['nonag2ag'])                                         # the arcs leaving a non-ag node
    if not arcs.size:
        return None
    from_k = cols['from_k'].values[arcs]
    local_r = cols['local_r'].values[arcs]
    cell = cols['cell'].values[arcs]
    source = np.ravel_multi_index((from_k, cell), side.nonag_base_kr.shape)                          # the flat position of each arc's source in the base grid
    _, first_arc, row_of_arc = np.unique(source, return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(first_arc.size, n_all))
    rhs = side.nonag_base_kr[from_k[first_arc], cell[first_arc]].astype(np.float64)
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_n_{k}_{r}" for k, r in zip(from_k[first_arc], local_r[first_arc])]
    return make_part('source_cap_nonag', 'flow_out', dict(from_k=from_k[first_arc], local_r=local_r[first_arc]), A, rhs, '<', names)


def add_node_balance_ag(cols: xr.Dataset, side: ColSide):
    """Node balance at the ag nodes, one row per ag column (the ag block, in its order):
    X_ag[m, r, j] = base_ag[m, r, j] + Σ in (ag2ag ∪ nonag2ag → (m, j)) − Σ out ((m, j) → ag2ag ∪ ag2nonag).
    Every column and arc finds the row of the ag node it lands on / leaves on the ag column-id grid; an entry whose
    node has no column (-1) is dropped. The inflow cap is X's own ub, not a row."""
    print("│   ├── Adding node-balance (X = base + Σin − Σout) constraints at the ag nodes...")
    n_all = cols.attrs['n_all']
    m = cols['m'].values
    j = cols['j'].values
    from_m = cols['from_m'].values
    from_j = cols['from_j'].values
    cell = cols['cell'].values
    block_range = cols.attrs['block_range']
    ag = slice(*block_range['ag'])

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
    """Node balance at the non-ag nodes, one row per (non-ag land use, feasible cell), X column or not:
    X_nonag[r, k] = base_nonag[r, k] + Σ in (ag2nonag → k) − Σ out (k → nonag2ag).
    Every column and arc finds the row of the non-ag node it lands on / leaves on the non-ag row grid; an entry
    whose node is not feasible (-1) is dropped."""
    print("│   └── Adding node-balance (X = base + Σin − Σout) constraints at the non-ag nodes...")
    n_all = cols.attrs['n_all']
    k = cols['k'].values
    from_k = cols['from_k'].values
    cell = cols['cell'].values
    block_range = cols.attrs['block_range']

    # ── the rows: one per (non-ag land use, feasible cell) — every feasible entry, enabled land use or not: k then cell ──
    nonag_k, nonag_r = np.nonzero(side.nonag_ub_kr > 0)
    nonag_k = nonag_k.astype(np.int64)
    nonag_r = nonag_r.astype(np.int64)
    n_nonag = nonag_r.size
    if not n_nonag:
        return None

    nonag_row_kr = np.full(side.nonag_ub_kr.shape, -1, dtype=np.int32)          # the row of the non-ag node (k, r), -1 where it is not feasible
    nonag_row_kr[nonag_k, nonag_r] = np.arange(n_nonag)

    row_sign = np.where(side.col_nonag_kr[nonag_k, nonag_r] >= 0, 1.0, -1.0)    # a disabled land use has no X column: its row is a pure inflow guard, Σin − Σout = −base

    # ── the entries: X on its own row, inflows −1 on the target's row, outflows +1 on the source's row ──
    row_idx = []
    col_idx = []
    vals = []

    def add(row, col, value):
        in_model = row >= 0                                              # no row (the node is not feasible): entry dropped
        row_idx.append(row[in_model].astype(np.int64))
        col_idx.append(col[in_model].astype(np.int64))
        vals.append(value * row_sign[row[in_model]])

    columns = np.arange(*block_range['nonag'])                           # X_nonag on its own row
    add(nonag_row_kr[k[columns], cell[columns]], columns, 1.0)
    arcs = np.arange(*block_range['ag2nonag'])                           # ag → non-ag: in on the target's row
    add(nonag_row_kr[k[arcs], cell[arcs]], arcs, -1.0)
    arcs = np.arange(*block_range['nonag2ag'])                           # non-ag → ag: out of the source's row
    add(nonag_row_kr[from_k[arcs], cell[arcs]], arcs, 1.0)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_nonag, n_all))

    # ── rhs, names, keys ──
    A, rhs, _ = contract(A, side.nonag_base_kr[nonag_k, nonag_r].astype(np.float64) * row_sign)
    names = [f"bal_n_{k}_{r}" for k, r in zip(nonag_k, nonag_r)]
    return make_part('node_balance_nonag', 'flow_in', dict(k=nonag_k, cell=nonag_r), A, rhs, '=', names)
