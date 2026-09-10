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
from luto.solvers.col_builder import ColSide, block_slice
from luto.solvers.row_inputs import EconomicInputs, RowInputs
from luto.solvers.row_table import make_part, stack_rows


# ═══════════════════════════ get_rows: the row space of one step ═══════════════════════════

@dataclass
class RowSide:
    """What the post-solve reads beside the row table."""
    q_block: sparse.csr_matrix      # (commodity x n_all) the per-commodity production row over the columns, UNSCALED — the demand rows are its rescaled copies; Production = q_block @ x


def get_rows(inputs: RowInputs, cols: xr.Dataset, side: ColSide) -> tuple[xr.Dataset, RowSide]:
    """The row space of one solve step"""

    # ── 1. what the rows join on: the node every column lands on / leaves, and the runs of the accounting blocks ──
    to_node, from_node  = node_ids(cols)                                        # (m, j, cell) or (k, cell) per column, -1 where n/a
    ag_ptr              = ag_runs(cols)                                         # the ag columns of (m, j): one run each, cells ascending
    slots, am_ptr       = am_slots(cols)                                        # the (option, land use) slots, and the am columns of (slot, m): one run each

    # ── 2. the structural rows: what a cell's space, an ag-mgt option and the existing capacity allow ──
    parts = [
        add_renewable_ceiling(inputs, cols),                                    # simulated + existing capacity share the cell
        add_cell_usage(inputs, cols),                                           # every cell's shares sum to its ag proportion
        add_ag_mgt_link(cols, to_node, ag_ptr, slots, am_ptr),                  # an ag-mgt column cannot exceed its ag column ...
        add_ag_mgt_adoption(inputs, cols, ag_ptr, slots, am_ptr),               # ... nor the option's adoption limit
    ]

    # ── 3. the demand rows: the production block, kept unscaled beside the table ──
    demand, q_block = add_demand(inputs, cols, ag_ptr, slots, am_ptr)
    parts.append(demand)

    # ── 4. the policy rows: a coefficient per accounting column, weighted over the cells the target covers ──
    parts.append(add_ghg(inputs, cols, side))

    print("│   ├── Adding constraints for biodiversity...")
    bio_on = any(target != 'off' for target in (settings.GBF2_TARGET, settings.GBF3_NVIS_TARGET,
                                                settings.GBF4_TARGET_SNES, settings.GBF4_TARGET_ECNES, settings.GBF8_TARGET))
    bio_c = bio_coeff(inputs, cols) if bio_on else None                         # the biodiversity contribution at every accounting column: all five GBF families weight this one stream
    parts += [
        add_GBF2(inputs, cols, bio_c),
        add_GBF3_NVIS(inputs, cols, bio_c),
        add_GBF4_SNES(inputs, cols, bio_c),
        add_GBF4_ECNES(inputs, cols, bio_c),
        add_GBF8(inputs, cols, bio_c),
    ]

    # the non-ag caps recede 1e-6/yr RELATIVE, so the RHS always stays ahead of the ratcheting lower bound
    # non-reversible plantings create (last year's solved areas become this year's exact lower bounds, and
    # float32 noise then puts the locked-in floor a hair over a saturated cap, which presolve rejects with NO
    # tolerance). Ag caps need no slack: ag is reversible.
    relax = 1 + (inputs.target_year - settings.SIM_YEARS[0]) * 1e-6
    parts += [
        add_regional_adoption_ag(inputs, cols),
        add_regional_adoption_nonag(inputs, cols, relax),
        add_regional_adoption_nonag_sum(inputs, cols, relax),
        add_water(inputs, cols),
        add_renewable(inputs, cols, side, ag_ptr),
    ]

    # ── 5. the flow rows: the arcs grouped by where they leave, every column joined on the node it lands on ──
    parts += [
        add_source_cap_ag(cols, side, from_node),
        add_source_cap_nonag(cols, side, from_node),
        add_node_balance(cols, side, to_node, from_node),
    ]

    # ── 6. the space: the parts back to back in the order above, and beside the table the unscaled production block ──
    return stack_rows([part for part in parts if part is not None]), RowSide(q_block=q_block)


# ═══════════════════════════ the coefficient contract: gather → compose → contract ═══════════════════════════
#
# The model is one matrix, rows × cols. The columns are the column table; every family below produces ROWS of
# that matrix, always the same way: GATHER a coefficient per column off the table's fields, COMPOSE it with the
# row's weights over cells into one sparse row (the support only), and CONTRACT the family's stacked block in
# one loop over its rows — the SOLVER_COEFF_MIN drop, then (policy families only) the geomean row rescale and
# the floor again.

def gather(cols: xr.Dataset, ag_c_mrj, am_c_mrj: dict, nonag_c_rk) -> np.ndarray:
    """One family's coefficient at every accounting column, float32, read by what the column's fields say it
    is: ``k >= 0`` non-ag ``[cell, k]``; ``am_idx >= 0`` ag-mgt, its option's ``[m, cell, j_idx]``; else ag ``[m, cell, j]``."""
    n_terms = cols.attrs['n_terms']
    m, j, k, cell, am_idx, j_idx = (cols[field].values[:n_terms] for field in ('m', 'j', 'k', 'cell', 'am_idx', 'j_idx'))
    is_nonag, is_am = k >= 0, am_idx >= 0
    is_ag = ~is_nonag & ~is_am
    c = np.empty(n_terms, dtype=np.float32)
    c[is_ag] = ag_c_mrj[m[is_ag], cell[is_ag], j[is_ag]]
    c[is_nonag] = nonag_c_rk[cell[is_nonag], k[is_nonag]]
    for idx, option in enumerate(cols.attrs['options']):
        sel = am_idx == idx
        if sel.any():
            c[sel] = am_c_mrj[option][m[sel], cell[sel], j_idx[sel]]
    return c


def compose(cols: xr.Dataset, c: np.ndarray, weights) -> sparse.csr_matrix:
    """The family's rows stacked: one CSR row per weighting row ``V`` over cells in ``weights``, entry
    ``q = V_i[cell_t] · c_t`` (float32) at accounting column t, the nonzero support only (not yet contracted)."""
    n_terms = cols.attrs['n_terms']
    term_cell = cols['cell'].values[:n_terms]
    term_col = np.arange(n_terms, dtype=np.int32)                         # the accounting columns are the first rows of the table: column = row
    by_cell_order, by_cell_ptr = cols.attrs['by_cell_order'], cols.attrs['by_cell_ptr']
    ncells = by_cell_ptr.size - 1
    nvars = cols.attrs['n_all']
    indptr = [0]
    indices = []
    data = []
    for weight_row in weights:
        weight_row = np.asarray(weight_row, dtype=np.float32)
        cells = np.flatnonzero(weight_row)
        if cells.size * 2 < ncells:                                       # sparse support: gather only its columns, through the accounting rows sorted by cell — cost ∝ the row's nonzero cells, not the model
            starts = by_cell_ptr[cells]
            counts = by_cell_ptr[cells + 1] - by_cell_ptr[cells]
            positions = np.repeat(starts - (np.cumsum(counts) - counts), counts) + np.arange(int(counts.sum()))
            term_idx = by_cell_order[positions]
            q = weight_row[term_cell[term_idx]] * c[term_idx]
            keep = q != 0                                                 # the nonzero support (NaN kept — ``contract`` drops it)
            kept_cols, kept_vals = term_col[term_idx][keep], q[keep]
        else:
            q = weight_row[term_cell] * c
            keep = q != 0
            kept_cols, kept_vals = term_col[keep], q[keep]
        order = np.argsort(kept_cols, kind='stable')
        indices.append(kept_cols[order])
        data.append(kept_vals[order])
        indptr.append(indptr[-1] + kept_cols.size)
    if not indices:
        return sparse.csr_matrix((0, nvars), dtype=np.float32)
    block = sparse.csr_matrix((np.concatenate(data).astype(np.float32), np.concatenate(indices).astype(np.int32),
                               np.asarray(indptr, dtype=np.int64)), shape=(len(weights), nvars))
    block.has_sorted_indices = True
    return block


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
    """The biodiversity contribution at every accounting column: the three streams ``gather`` reads — a scalar
    per ag land use, a per-cell array per (option, land use), a scalar per non-ag land use — as broadcast VIEWS
    (no copy), gathered ONCE for all five GBF families."""
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


# ═══════════════════════════ the nodes and the runs: what the rows group and join on ═══════════════════════════

def ag_node(cols: xr.Dataset, m, j, cell) -> np.ndarray:
    """The node id of an ag (m, j, cell): ``(m · n_lu + j) · ncells + cell`` — ascending in the ag block's own order."""
    n_lu, ncells = cols.attrs['n_ag_lus'], cols.attrs['ncells']
    return (np.asarray(m, dtype=np.int64) * n_lu + j) * ncells + cell


def nonag_node(cols: xr.Dataset, k, cell) -> np.ndarray:
    """The node id of a non-ag (k, cell): ``k · ncells + cell`` numbered after every ag node."""
    nlms, n_lu, ncells = cols.attrs['nlms'], cols.attrs['n_ag_lus'], cols.attrs['ncells']
    return nlms * n_lu * ncells + np.asarray(k, dtype=np.int64) * ncells + cell


def node_ids(cols: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Every column's nodes, int64, -1 where n/a: ``to_node`` — the (m, j, cell) an ag / ag-mgt / ag2ag /
    nonag2ag column lands on, the (k, cell) a non-ag / ag2nonag column lands on; ``from_node`` — the node
    an arc comes from. The flow rows group and join on these; the ag-mgt link joins an am column to its host."""
    m, j, k, from_m, from_j, from_k, cell = (cols[field].values for field in ('m', 'j', 'k', 'from_m', 'from_j', 'from_k', 'cell'))
    to_node = np.full(cell.size, -1, dtype=np.int64)
    from_node = np.full(cell.size, -1, dtype=np.int64)
    lands_nonag, lands_ag = k >= 0, (k < 0) & (j >= 0)
    to_node[lands_nonag] = nonag_node(cols, k[lands_nonag], cell[lands_nonag])
    to_node[lands_ag] = ag_node(cols, m[lands_ag], j[lands_ag], cell[lands_ag])
    leaves_nonag, leaves_ag = from_k >= 0, from_j >= 0
    from_node[leaves_nonag] = nonag_node(cols, from_k[leaves_nonag], cell[leaves_nonag])
    from_node[leaves_ag] = ag_node(cols, from_m[leaves_ag], from_j[leaves_ag], cell[leaves_ag])
    return to_node, from_node


def ag_runs(cols: xr.Dataset) -> np.ndarray:
    """The ag block grouped by (m, j) — ``ptr[m · n_lu + j] : ptr[m · n_lu + j + 1]`` is its run of table rows, cells ascending."""
    ag = block_slice(cols, 'ag')
    key = cols['m'].values[ag] * cols.attrs['n_ag_lus'] + cols['j'].values[ag]                 # ascending: the block is in (m, j, cell) order
    return ag.start + np.searchsorted(key, np.arange(cols.attrs['nlms'] * cols.attrs['n_ag_lus'] + 1))


def am_slots(cols: xr.Dataset) -> tuple[list, np.ndarray]:
    """The (option, land use) slots in slot order, and the am block grouped by (slot, m) — ``ptr[slot · nlms + m]``
    bounds its run of table rows, cells ascending (a slot with no column has an empty run)."""
    am = block_slice(cols, 'am')
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
    m, j, k, local_r = (cols[field].values for field in ('m', 'j', 'k', 'local_r'))
    obj = np.zeros(cols.attrs['n_all'], dtype=np.float32)

    # ── operating economics: one gather over the accounting columns ──
    n_terms = cols.attrs['n_terms']
    obj[:n_terms] = gather(cols, econ.ag_obj_mrj, econ.ag_man_objs, econ.non_ag_obj_rk)

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

def add_renewable_ceiling(inputs: RowInputs, cols: xr.Dataset):
    """Simulated and existing renewable capacity compete for the cell's space [0, ag_mask]: one row per
    (am, cell) with existing capacity, Σ_{m, j} X_am[am, m, j, r] ≤ max(ag_mask[r] − exist_r[r], 0)."""
    n_all = cols.attrs['n_all']
    am = block_slice(cols, 'am')
    am_idx, cell = cols['am_idx'].values, cols['cell'].values
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
        col_idx = am.start + np.flatnonzero(am_idx[am] == cols.attrs['options'].index(option))   # the option's columns ...
        cells, cell_of_col = np.unique(cell[col_idx], return_inverse=True)                       # ... grouped by cell, ascending
        existing_cap = exist_r[cells]
        keep_cell = existing_cap != 0                                    # no existing capacity -> no ceiling row
        n_rows = int(keep_cell.sum())
        if not n_rows:
            continue
        row_of_cell = np.full(cells.size, -1, dtype=np.int64)
        row_of_cell[keep_cell] = np.arange(n_rows)
        row_idx = row_of_cell[cell_of_col]
        in_row = row_idx >= 0
        blocks.append(sparse.csr_matrix((np.ones(int(in_row.sum())), (row_idx[in_row], col_idx[in_row])), shape=(n_rows, n_all)))
        rhs.append(np.maximum(ag_mask[cells[keep_cell]] - existing_cap[keep_cell], 0.0))   # cell space left for simulated capacity
        names += [f"const_{am_name}_solvable_ub_{r}".replace(" ", "_") for r in cells[keep_cell]]
        key_am += [cols.attrs['options'].index(option)] * n_rows
        key_cell.append(cells[keep_cell])
    if not blocks:
        return None
    A, rhs, _ = contract(sparse.vstack(blocks, format='csr'), np.concatenate(rhs))
    return make_part('renewable_ceiling', 'ag_mgt_ub', dict(am_idx=key_am, cell=np.concatenate(key_cell)), A, rhs, '<', names)


def add_cell_usage(inputs: RowInputs, cols: xr.Dataset):
    """Every cell's ag + non-ag shares sum to its base-year agricultural proportion: one row per cell
    with a slack column, stored the way Gurobi stores an addRange row (Σ X + slack = hi)."""
    n_all = cols.attrs['n_all']
    cell = cols['cell'].values
    ag, nonag, slack = (block_slice(cols, block) for block in ('ag', 'nonag', 'cell_usage'))
    row_cells = cell[slack]
    n_rows = row_cells.size
    row_of_cell = np.full(cols.attrs['ncells'], -1, dtype=np.int64)
    row_of_cell[row_cells] = np.arange(n_rows)
    columns = np.concatenate([np.arange(ag.start, ag.stop), np.arange(nonag.start, nonag.stop), np.arange(slack.start, slack.stop)])
    row_idx = row_of_cell[cell[columns]]
    in_row = row_idx >= 0
    A = sparse.csr_matrix((np.ones(int(in_row.sum())), (row_idx[in_row], columns[in_row])), shape=(n_rows, n_all))
    # Ranged, not ==: presolve folds the node-balance rows into this one and compares two constants summed
    # along different float32 paths (up to ~1.75x FeasibilityTol apart) with NO tolerance. The +-10x Ftol band
    # absorbs that; conservation still pins the cell total, so the band is not exploitable.
    hi = inputs.ag_mask_proportion_r[row_cells].astype(np.float64) + 10 * settings.FEASIBILITY_TOLERANCE   # the top of the band (widened before the band is applied)
    A, hi, _ = contract(A, hi)
    return make_part('cell_usage', 'cell_usage', dict(cell=row_cells), A, hi, '=',
                     [f"const_cell_usage_{cell}" for cell in row_cells])


def add_ag_mgt_link(cols: xr.Dataset, to_node: np.ndarray, ag_ptr: np.ndarray, slots: list, am_ptr: np.ndarray):
    """Ag-management variables cannot exceed their agricultural variable: one row per (am, land use, lm,
    cell) with an ag column — X_am − X_ag ≤ 0, or X_ag ≥ 0 where the am column does not exist."""
    n_all = cols.attrs['n_all']
    nlms, n_lu = cols.attrs['nlms'], cols.attrs['n_ag_lus']
    cell = cols['cell'].values
    ag = block_slice(cols, 'ag')
    ag_node_of_col = to_node[ag]                                         # ascending: the ag block is in (m, j, cell) order
    options = cols.attrs['options']
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
            run_start = ag_ptr[m * n_lu + j]
            ag_cols = np.arange(run_start, ag_ptr[m * n_lu + j + 1])                        # the ag columns of (m, j), cells ascending: one row each
            am_cols = np.arange(am_ptr[slot * nlms + m], am_ptr[slot * nlms + m + 1])       # the slot's am columns at m ...
            host = np.searchsorted(ag_node_of_col, to_node[am_cols]) - (run_start - ag.start)   # ... each joined to the ag column it sits on (its host node), as a row of this run
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


def add_ag_mgt_adoption(inputs: RowInputs, cols: xr.Dataset, ag_ptr: np.ndarray, slots: list, am_ptr: np.ndarray):
    """Adoption limits: one row per (am, land use), Σ am columns − limit · Σ ag columns ≤ 0
    (Σam ≤ limit · Σag with the RHS moved to the LHS); zero coefficients (limit = 0) are dropped."""
    n_all = cols.attrs['n_all']
    nlms, n_lu = cols.attrs['nlms'], cols.attrs['n_ag_lus']
    row_idx = []
    col_idx = []
    vals = []
    names = []
    key_am = []
    key_lu = []
    for row, (option, j) in enumerate(slots):
        adoption_limit = float(np.float64(inputs.ag_man_limits[option][j]))
        am_cols = np.arange(am_ptr[row * nlms], am_ptr[row * nlms + nlms])                                           # the slot's am columns, both lm (adjacent runs)
        ag_cols = np.concatenate([np.arange(ag_ptr[m * n_lu + j], ag_ptr[m * n_lu + j + 1]) for m in range(nlms)])   # the ag columns of j, both lm
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

def add_demand(inputs: RowInputs, cols: xr.Dataset, ag_ptr: np.ndarray, slots: list, am_ptr: np.ndarray):
    """Hard demand constraints: one per-commodity quantity row over the accounting columns, used once under
    '=' where the DEMAND_BOUNDS lb == ub, else twice — under '>' lb and '<' ub. Returns the part and, beside
    it, the UNSCALED production block the post-solve reads (the demand rows are its rescaled copies)."""
    print("│   ├── Adding <hard> demand constraints (equality where lb==ub, else lower + upper)...")
    n_all = cols.attrs['n_all']
    k, cell = cols['k'].values, cols['cell'].values
    nonag = block_slice(cols, 'nonag')
    nlms, n_lu = cols.attrs['nlms'], cols.attrs['n_ag_lus']
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

    # ── the per-commodity LHS (q_block): the ag columns per (m, land use) and the ag-mgt columns per (slot, m)
    #    carry jc[c, cell] = Σ_p pr2cm[c, p] · q[m, cell, p] over the land use's active products; the non-ag
    #    columns per k carry non_ag_q_crk[c, cell, k] ──
    for lu in range(n_lu):
        active_p = np.where(inputs.lu2pr_pj[:, lu])[0]
        if not active_p.size:
            continue
        for lm in range(nlms):
            group = np.arange(ag_ptr[lm * n_lu + lu], ag_ptr[lm * n_lu + lu + 1])
            if group.size:
                put(inputs.pr2cm_cp[:, active_p] @ inputs.ag_q_mrp[lm, cell[group], :][:, active_p].T, group)
    for slot, (option, lu) in enumerate(slots):
        active_p = np.where(inputs.lu2pr_pj[:, lu])[0]
        if not active_p.size:
            continue
        for lm in range(nlms):
            group = np.arange(am_ptr[slot * nlms + lm], am_ptr[slot * nlms + lm + 1])
            if group.size:
                put(inputs.pr2cm_cp[:, active_p] @ inputs.ag_man_q_mrp[option][lm, cell[group], :][:, active_p].T, group)
    for lu in np.unique(k[nonag]):
        group = nonag.start + np.flatnonzero(k[nonag] == lu)
        put(inputs.non_ag_q_crk[:, cell[group], lu], group)
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
    m, j, local_r = (cols[field].values for field in ('m', 'j', 'local_r'))

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


def add_GBF2(inputs: RowInputs, cols: xr.Dataset, bio_c: np.ndarray):
    """GBF2 priority degraded areas: the bio contribution at every accounting column weighted by
    GBF2_mask_area_r, which is ZERO off-mask (off-mask columns get coefficient 0 and are dropped). One row."""
    if settings.GBF2_TARGET == "off":
        print("│   │   ├── TURNING OFF constraints for biodiversity GBF 2...")
        return None
    print(f'│   │   ├── Adding constraints for biodiversity GBF 2: {inputs.limits["GBF2"]:15,.0f}')
    A, rhs, scale = contract(compose(cols, bio_c, [inputs.GBF2_mask_area_r]), [inputs.limits["GBF2"]], rescale=True)
    return make_part('GBF2', 'bio_gbf2', {}, A, rhs, '>',
                     ["bio_GBF2_priority_degraded_area_limit"], scale)


def add_GBF3_NVIS(inputs: RowInputs, cols: xr.Dataset, bio_c: np.ndarray):
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
    A, rhs, scale = contract(compose(cols, bio_c, weights), rhs, rescale=True)
    return make_part('GBF3_NVIS', 'bio_nvis',
                     dict(region=[region for region, _ in kept], item=[group for _, group in kept]),
                     A, rhs, '>', names, scale)


def add_GBF4_SNES(inputs: RowInputs, cols: xr.Dataset, bio_c: np.ndarray):
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
    A, rhs, scale = contract(compose(cols, bio_c, weights), rhs, rescale=True)
    return make_part('GBF4_SNES', 'bio_snes',
                     dict(region=[key[0] for key in kept], item=[key[1] for key in kept], presence=[key[2] for key in kept]),
                     A, rhs, '>', names, scale)


def add_GBF4_ECNES(inputs: RowInputs, cols: xr.Dataset, bio_c: np.ndarray):
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
    A, rhs, scale = contract(compose(cols, bio_c, weights), rhs, rescale=True)
    return make_part('GBF4_ECNES', 'bio_ecnes',
                     dict(region=[key[0] for key in kept], item=[key[1] for key in kept], presence=[key[2] for key in kept]),
                     A, rhs, '>', names, scale)


def add_GBF8(inputs: RowInputs, cols: xr.Dataset, bio_c: np.ndarray):
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
    A, rhs, scale = contract(compose(cols, bio_c, weights), rhs, rescale=True)
    return make_part('GBF8', 'bio_gbf8',
                     dict(region=[region for region, _ in kept], item=[species for _, species in kept]),
                     A, rhs, '>', names, scale)


def add_regional_adoption_ag(inputs: RowInputs, cols: xr.Dataset):
    """Per-(region, ag land use) caps ('on' mode): Σ real_area[r] · X_ag over the region's cells ≤ cap.
    Hectares are NOT rescaled — the shadow-price reader assumes scale 1."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        print("│   │   └── TURNING OFF constraints for regional adoption ...")
        return None
    ag = block_slice(cols, 'ag')
    ag_j, ag_r = cols['j'].values[ag], cols['cell'].values[ag]
    real_area = inputs.real_area
    rows = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits["ag_regional_adoption"]:
        name = f"reg_adopt_limit_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        selected = np.flatnonzero((ag_j == lu_code) & np.isin(ag_r, reg_cells))
        rows.append((ag.start + selected, real_area[ag_r[selected]].astype(np.float32)))
        rhs.append(area_limit_ha)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not rows:
        return None
    A, rhs, _ = contract(area_block(rows, cols.attrs['n_all']), rhs)
    return make_part('regional_adoption_ag', 'adopt_ag',
                     dict(region=[reg_id for reg_id, _ in keys], j=[lu_code for _, lu_code in keys]),
                     A, rhs, '<', names)


def add_regional_adoption_nonag(inputs: RowInputs, cols: xr.Dataset, relax: float):
    """Per-(region, non-ag land use) caps ('on' mode), with the per-year relaxation on the RHS."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    nonag = block_slice(cols, 'nonag')
    na_k, na_r = cols['k'].values[nonag], cols['cell'].values[nonag]
    real_area = inputs.real_area
    rows = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption") or []:
        name = f"reg_adopt_limit_non_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        selected = np.flatnonzero((na_k == lu_code) & np.isin(na_r, reg_cells))
        rows.append((nonag.start + selected, real_area[na_r[selected]].astype(np.float32)))
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not rows:
        return None
    A, rhs, _ = contract(area_block(rows, cols.attrs['n_all']), rhs)
    return make_part('regional_adoption_nonag', 'adopt_nonag',
                     dict(region=[reg_id for reg_id, _ in keys], k=[lu_code for _, lu_code in keys]),
                     A, rhs, '<', names)


def add_regional_adoption_nonag_sum(inputs: RowInputs, cols: xr.Dataset, relax: float):
    """SUM-of-non-ag caps ('NON_AG_CAP' mode): every non-ag land use in a region together, with the relaxation."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    nonag = block_slice(cols, 'nonag')
    na_r = cols['cell'].values[nonag]
    real_area = inputs.real_area
    rows = []
    rhs = []
    names = []
    keys = []
    for reg_id, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption_sum") or []:
        name = f"reg_adopt_limit_non_ag_sum_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        selected = np.flatnonzero(np.isin(na_r, reg_cells))
        rows.append((nonag.start + selected, real_area[na_r[selected]].astype(np.float32)))
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append(reg_id)
    if not rows:
        return None
    A, rhs, _ = contract(area_block(rows, cols.attrs['n_all']), rhs)
    return make_part('regional_adoption_nonag_sum', 'nonag_cap', dict(region=keys), A, rhs, '<', names)


def area_block(rows: list, n_all: int) -> sparse.csr_matrix:
    """The three regional-adoption families' rows — each already a (columns, hectares) pair — as one block,
    the nonzero support only."""
    support = [(row_cols[vals != 0], vals[vals != 0]) for row_cols, vals in rows]
    row_idx = np.concatenate([np.full(row_cols.size, row) for row, (row_cols, _) in enumerate(support)])
    return sparse.csr_matrix((np.concatenate([vals for _, vals in support]), (row_idx, np.concatenate([row_cols for row_cols, _ in support]))),
                             shape=(len(support), n_all))


def add_water(inputs: RowInputs, cols: xr.Dataset):
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
    A, rhs, scale = contract(compose(cols, coeff, weights), rhs, rescale=True)
    return make_part('water', 'water', dict(region=region_ids), A, rhs, '>', names, scale)


def add_renewable(inputs: RowInputs, cols: xr.Dataset, side: ColSide, ag_ptr: np.ndarray):
    """State-level renewable generation targets: one row per (state, type) — the type's ag-mgt columns
    weighted by an allowed-cells indicator, RHS = target − existing capacity."""
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
    n_terms = cols.attrs['n_terms']
    cell, am_idx = cols['cell'].values[:n_terms], cols['am_idx'].values[:n_terms]
    ncells = cols.attrs['ncells']

    # ── the coefficient per type: energy_r on that type's ag-mgt columns, 0 on every other accounting column ──
    coeff_of_type = {}
    for option, re_data in re_types.items():
        if option in cols.attrs['options']:
            on_type = am_idx == cols.attrs['options'].index(option)
            coeff_of_type[option] = np.where(on_type, re_data['energy_r'][cell], np.float32(0.0)).astype(np.float32)   # float32 yield per cell

    # ── one row per (state, type) with eligible cells ──
    n_lu = cols.attrs['n_ag_lus']
    cells_of_lu = {j: np.unique(np.concatenate([cell[ag_ptr[m * n_lu + j]:ag_ptr[m * n_lu + j + 1]] for m in range(cols.attrs['nlms'])]))
                   for j in range(n_lu)}                                  # the cells with an ag column of j, either lm
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
            # cell-set row-inclusion rule (NOT a coefficient test): a row exists iff some compatible land
            # use has eligible cells — even if every coefficient there turns out to be sub-floor
            has_cells = False
            for j in agman2lu[am]:
                eligible_cells = np.intersect1d(cells_of_lu[j], state_cells)
                if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS == True:
                    eligible_cells = np.setdiff1d(eligible_cells, re_data['gbf2_mask_idx'])
                if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK == True:
                    eligible_cells = np.setdiff1d(eligible_cells, re_data['mnes_mask_idx'])
                if eligible_cells.size:
                    has_cells = True
                    break
            if not has_cells:
                continue
            allowed = np.zeros(ncells, dtype=np.float32)                 # the weighting row: 1 on the state's allowed cells
            allowed[state_cells] = 1.0
            if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS == True:
                allowed[re_data['gbf2_mask_idx']] = 0.0
            if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK == True:
                allowed[re_data['mnes_mask_idx']] = 0.0
            blocks.append(compose(cols, coeff_of_type[am], [allowed]))
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
#   group the arcs by from_node                →  source cap:   Σ out ≤ base              (a ≤ row per source node)
#   join every column on its node, both roles  →  node balance: X = base + Σ in − Σ out   (an = row per node)
#   the inflow cap is not a row: X's own ub (the transition upper bound) and the cell-usage row bound it.

def add_source_cap_ag(cols: xr.Dataset, side: ColSide, from_node: np.ndarray):
    """Source cap, ag sources: the arcs leaving an ag node (``from_j >= 0``: ag2ag ∪ ag2nonag) grouped by
    ``from_node``; each group's arcs sum to at most its base share, Σ out ≤ base[from_m, r, from_j]."""
    # bounds the arc columns (some flow costs are negative) and rules out pass-through
    print("│   ├── Adding source-cap (Σ out ≤ base) constraints...")
    n_all = cols.attrs['n_all']
    arcs = np.flatnonzero(cols['from_j'].values >= 0)
    from_m, from_j, local_r, cell = (cols[field].values[arcs] for field in ('from_m', 'from_j', 'local_r', 'cell'))
    # one row per source node; every arc leaving it gets a +1
    nodes, first_arc, row_of_arc = np.unique(from_node[arcs], return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(nodes.size, n_all))
    rhs = side.ag_base_mjr[from_m[first_arc], from_j[first_arc], cell[first_arc]].astype(np.float64)   # from the ag grid: a source with no X column still caps its outflow
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_a_{m}_{j}_{r}" for m, j, r in zip(from_m[first_arc], from_j[first_arc], local_r[first_arc])]
    return make_part('source_cap_ag', 'flow_out',
                     dict(from_m=from_m[first_arc], from_j=from_j[first_arc], local_r=local_r[first_arc]),
                     A, rhs, '<', names)


def add_source_cap_nonag(cols: xr.Dataset, side: ColSide, from_node: np.ndarray):
    """Source cap, non-ag sources: the arcs leaving a non-ag node (``from_k >= 0``: nonag2ag) grouped by
    ``from_node``; Σ out ≤ base_nonag[r, from_k]."""
    n_all = cols.attrs['n_all']
    arcs = np.flatnonzero(cols['from_k'].values >= 0)
    if not arcs.size:
        return None
    from_k, local_r, cell = (cols[field].values[arcs] for field in ('from_k', 'local_r', 'cell'))
    nodes, first_arc, row_of_arc = np.unique(from_node[arcs], return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(nodes.size, n_all))
    rhs = side.nonag_base_kr[from_k[first_arc], cell[first_arc]].astype(np.float64)
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_n_{k}_{r}" for k, r in zip(from_k[first_arc], local_r[first_arc])]
    return make_part('source_cap_nonag', 'flow_out', dict(from_k=from_k[first_arc], local_r=local_r[first_arc]), A, rhs, '<', names)


def add_node_balance(cols: xr.Dataset, side: ColSide, to_node: np.ndarray, from_node: np.ndarray):
    """Node balance, X = base + Σ in − Σ out at every node (m, j, cell) or (k, cell): one row per ag
    column, then one per (non-ag land use, feasible cell). The inflow cap is X's own ub, not a row."""
    print("│   └── Adding node-balance (X = base + Σin − Σout) constraints...")
    n_all = cols.attrs['n_all']
    m, j, cell = (cols[field].values for field in ('m', 'j', 'cell'))
    ag, nonag = block_slice(cols, 'ag'), block_slice(cols, 'nonag')

    # ── the rows: one per ag column, then one per (non-ag land use, feasible cell) — X column or not — keyed by node, ascending ──
    n_ag = ag.stop - ag.start
    ag_m, ag_j, ag_r = m[ag].astype(np.int64), j[ag].astype(np.int64), cell[ag].astype(np.int64)
    nonag_k, nonag_r = np.nonzero(side.nonag_ub_kr > 0)       # every feasible entry, enabled land use or not: k then cell
    nonag_k = nonag_k.astype(np.int64)
    nonag_r = nonag_r.astype(np.int64)
    n_nonag = nonag_r.size
    row_keys = np.concatenate([to_node[ag], nonag_node(cols, nonag_k, nonag_r)])   # the ag block is in (m, j, cell) order, np.nonzero in (k, cell) order: ascending

    def row_of(node):
        """The row whose key is ``node`` (a join on the ascending row keys), -1 where no row has it."""
        if not row_keys.size:
            return np.full(node.size, -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(row_keys, node), row_keys.size - 1)
        return np.where(row_keys[pos] == node, pos, -1)

    nonag_cols = np.arange(nonag.start, nonag.stop)
    row_sign = np.ones(n_ag + n_nonag, dtype=np.float64)                 # a disabled land use has no X column: its row is a pure inflow guard, Σin − Σout = −base
    row_sign[n_ag:] = -1.0
    row_sign[row_of(to_node[nonag_cols])] = 1.0

    # ── the entries: X on its own row, inflows −1 on the target's row, outflows +1 on the source's row ──
    #     X_ag[m, r, j]  = base_ag[m, r, j]  + Σ in (ag2ag ∪ nonag2ag → (m, j)) − Σ out ((m, j) → ag2ag ∪ ag2nonag)
    #     X_nonag[r, k]  = base_nonag[r, k]  + Σ in (ag2nonag → k)              − Σ out (k → nonag2ag)
    row_idx = []
    col_idx = []
    vals = []

    def add(row, col, value):
        in_model = row >= 0                                              # no row (banned source / no X var): entry dropped
        row_idx.append(row[in_model].astype(np.int64))
        col_idx.append(col[in_model].astype(np.int64))
        vals.append(value * row_sign[row[in_model]])

    arcs = np.flatnonzero(from_node >= 0)
    add(np.arange(n_ag), np.arange(ag.start, ag.stop), 1.0)
    add(row_of(to_node[nonag_cols]), nonag_cols, 1.0)
    add(row_of(to_node[arcs]), arcs, -1.0)
    add(row_of(from_node[arcs]), arcs, 1.0)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_ag + n_nonag, n_all))

    # ── rhs, names, keys ──
    rhs = np.concatenate([cols['base'].values[ag].astype(np.float64),
                          side.nonag_base_kr[nonag_k, nonag_r].astype(np.float64) * row_sign[n_ag:]])
    names = ([f"bal_a_{m}_{j}_{r}" for m, j, r in zip(ag_m, ag_j, ag_r)] + [f"bal_n_{k}_{r}" for k, r in zip(nonag_k, nonag_r)])
    keys = dict(m=np.concatenate([ag_m, np.full(n_nonag, -1, dtype=np.int64)]),                  # an ag row carries its (m, j) node ...
                j=np.concatenate([ag_j, np.full(n_nonag, -1, dtype=np.int64)]),
                k=np.concatenate([np.full(n_ag, -1, dtype=np.int64), nonag_k]),                  # ... a non-ag row its k
                cell=np.concatenate([ag_r, nonag_r]))
    A, rhs, _ = contract(A, rhs)
    return make_part('node_balance', 'flow_in', keys, A, rhs, '=', names)
