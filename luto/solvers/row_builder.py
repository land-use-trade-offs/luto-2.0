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

from scipy import sparse

from luto import settings
import luto.tools as tools
from luto.solvers.row_inputs import EconomicInputs, RowInputs


# ═══════════════════════════ the coefficient contract: gather → compose → stack → contract ═══════════════════════════
#
# The model is one matrix, rows × cols. The columns are the column table (cols['table']); every family
# below produces ROWS of that matrix, always the same way: GATHER a coefficient per column off the
# table's fields, COMPOSE it with the row's weights over cells into one sparse row (the support only),
# STACK the family's rows into a block, and CONTRACT the block in one loop over its rows — the
# SOLVER_COEFF_MIN drop, then (policy families only) the geomean row rescale and the floor again.

def support(col: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The nonzero support of a row: (column, coefficient) where q != 0 (NaN included — ``contract`` drops it)."""
    keep = q != 0
    return col[keep], q[keep]


def block_slice(table: xr.Dataset, block: str) -> slice:
    """The rows of one block of the column table — its Var.index range (``attrs['block_range']`` holds the bounds)."""
    return slice(*table.attrs['block_range'][block])


def gather(table: xr.Dataset, ag_c_mrj, am_c_mrj: dict, nonag_c_rk) -> np.ndarray:
    """One family's coefficient at every scored column, float32, read by what the column's fields say it
    is: ``k >= 0`` non-ag ``[cell, k]``; ``am_idx >= 0`` ag-mgt, its option's ``[m, cell, j_idx]``; else ag ``[m, cell, j]``."""
    n_terms = table.attrs['n_terms']
    m, j, k, cell, am_idx, j_idx = (table[field].values[:n_terms] for field in ('m', 'j', 'k', 'cell', 'am_idx', 'j_idx'))
    is_nonag, is_am = k >= 0, am_idx >= 0
    is_ag = ~is_nonag & ~is_am
    c = np.empty(n_terms, dtype=np.float32)
    c[is_ag] = ag_c_mrj[m[is_ag], cell[is_ag], j[is_ag]]
    c[is_nonag] = nonag_c_rk[cell[is_nonag], k[is_nonag]]
    for idx, option in enumerate(table.attrs['options']):
        sel = am_idx == idx
        if sel.any():
            c[sel] = am_c_mrj[option][m[sel], cell[sel], j_idx[sel]]
    return c


def bio_streams(rows: RowInputs, cols: dict) -> tuple:
    """The biodiversity contribution as the three streams ``gather`` reads — a scalar per ag land use, a
    per-cell array per (option, land use), a scalar per non-ag land use — as broadcast VIEWS (no copy)."""
    nlms, ncells = cols['ag'].sizes['lm'], cols['ag'].sizes['cell']
    ag_j = np.asarray(rows.biodiv_contr_ag_j, dtype=np.float32)
    ag_c_mrj = np.broadcast_to(ag_j[None, None, :], (nlms, ncells, ag_j.size))
    contr_nonag_k = rows.biodiv_contr_non_ag_k
    n_k = max(contr_nonag_k) + 1 if len(contr_nonag_k) else 0
    nonag_k = np.array([contr_nonag_k.get(lu, 0.0) for lu in range(n_k)], dtype=np.float32)
    nonag_c_rk = np.broadcast_to(nonag_k[None, :], (ncells, n_k))
    am_c_mrj = {}
    for option, by_j_idx in rows.biodiv_contr_ag_man.items():
        per_cell = np.stack([np.asarray(by_j_idx[j_idx], dtype=np.float32) for j_idx in range(len(by_j_idx))], axis=1)   # (cell, j_idx)
        am_c_mrj[option] = np.broadcast_to(per_cell[None, :, :], (nlms, ncells, per_cell.shape[1]))
    return ag_c_mrj, am_c_mrj, nonag_c_rk


def compose_rows(cols: dict, c: np.ndarray, val_rows) -> sparse.csr_matrix:
    """The family's rows stacked: one CSR row per weighting row ``V`` over cells in ``val_rows``, entry
    ``q = V_i[cell_t] · c_t`` (float32) at scored column t, the nonzero support only (not yet contracted)."""
    table = cols['table']
    n_terms = table.attrs['n_terms']
    term_cell = table['cell'].values[:n_terms]
    term_col = np.arange(n_terms, dtype=np.int32)                         # the scored columns are the first rows of the table: column = row
    by_cell_order, by_cell_ptr = table.attrs['by_cell_order'], table.attrs['by_cell_ptr']
    ncells = by_cell_ptr.size - 1
    nvars = table.attrs['n_all']
    indptr = [0]
    indices = []
    data = []
    for val_row in val_rows:
        val_row = np.asarray(val_row, dtype=np.float32)
        cells = np.flatnonzero(val_row)
        if cells.size * 2 < ncells:                                   # sparse support: gather only its columns, through the scored rows sorted by cell — cost ∝ the row's nonzero cells, not the model
            starts = by_cell_ptr[cells]
            counts = by_cell_ptr[cells + 1] - by_cell_ptr[cells]
            positions = np.repeat(starts - (np.cumsum(counts) - counts), counts) + np.arange(int(counts.sum()))
            term_idx = by_cell_order[positions]
            kept_cols, kept_vals = support(term_col[term_idx], val_row[term_cell[term_idx]] * c[term_idx])
        else:
            kept_cols, kept_vals = support(term_col, val_row[term_cell] * c)
        order = np.argsort(kept_cols, kind='stable')
        indices.append(kept_cols[order])
        data.append(kept_vals[order])
        indptr.append(indptr[-1] + kept_cols.size)
    if not indices:
        return sparse.csr_matrix((0, nvars), dtype=np.float32)
    block = sparse.csr_matrix((np.concatenate(data).astype(np.float32), np.concatenate(indices).astype(np.int32),
                               np.asarray(indptr, dtype=np.int64)), shape=(len(val_rows), nvars))
    block.has_sorted_indices = True
    return block


def calc_geomean_scale(lhs_max: float, rhs_max: float) -> float:
    """One scale factor, ``sqrt(lhs_max * rhs_max) / RESCALE_FACTOR``: dividing both sides by it lands
    LHS_max and RHS symmetrically around RESCALE_FACTOR in log space."""
    if lhs_max > 0.0 and rhs_max > 0.0:
        return float(np.sqrt(lhs_max * rhs_max) / settings.RESCALE_FACTOR)
    ref = lhs_max if lhs_max > 0.0 else settings.RESCALE_FACTOR        # no RHS to balance against: LHS only
    return float(ref / settings.RESCALE_FACTOR)


def contract(block: sparse.csr_matrix, rhs=None, rescale: bool = False) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """The coefficient contract as ONE loop over the rows of a stacked block: every entry with
    |a| < SOLVER_COEFF_MIN dropped (NaN too); when ``rescale`` — the policy families — row i and rhs i
    are divided by ``calc_geomean_scale(max|row i|, |rhs i|)`` (an exact LP transformation) and the
    scaled row floored again. Returns (block, rhs, scale): scale 1 where not rescaled, ``row × scale``
    restores the raw row."""
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
    scale = np.fromiter((calc_geomean_scale(lhs_max, abs(rhs_i)) for lhs_max, rhs_i in zip(row_max, rhs)),
                        dtype=np.float64, count=block.shape[0])
    block.data = (block.data / np.repeat(scale, nnz_per_row)).astype(np.float32)
    block.data[np.abs(block.data) < settings.SOLVER_COEFF_MIN] = 0.0   # floor the scaled row
    block.eliminate_zeros()
    block.sort_indices()
    return block, rhs / scale, scale


# ═══════════════════════════ the objective: the economy coefficients as one block ═══════════════════════════

OBJ_BLOCKS = ('ag', 'am', 'nonag', 'trans_ag', 'trans_nonag')   # the rows of the objective block, in row order
OBJ_ROW = {name: row for row, name in enumerate(OBJ_BLOCKS)}    # the block row each economy term is charged to


def get_obj_block(econ: EconomicInputs, cols: dict) -> sparse.csr_matrix:
    """The economy coefficients (raw AUD) as a (5 x n_dec) block, one row per ``OBJ_BLOCKS`` component:
    the operating streams on their own columns, the transition costs negated on the arcs."""
    table = cols['table']
    m, j, k, local_r = (table[field].values for field in ('m', 'j', 'k', 'local_r'))
    parts = []                                                            # (block row, columns, values): the support of each part

    # ── operating economics: one gather over the scored columns, one part per block ──
    coeff = gather(table, econ.ag_obj_mrj, econ.ag_man_objs, econ.non_ag_obj_rk)   # float32, indexed by column
    for block in ('ag', 'am', 'nonag'):                                   # the block's own name IS its objective row
        span = block_slice(table, block)
        parts.append((OBJ_ROW[block], *support(np.arange(span.start, span.stop), coeff[span])))

    # ── transition costs on the arcs with an ag target: per source run of the block, gathered from the source-keyed cost dicts ──
    for block, sources, flow_cost in (('ag2ag', cols['sources']['ag'], econ.flow_cost_ag2ag),
                                      ('nonag2ag', cols['sources']['nonag'], econ.flow_cost_nonag2ag)):
        src_ptr = table.attrs['src_ptr'][block]
        for src, start, stop in zip(sources, src_ptr[:-1], src_ptr[1:]):
            run = slice(int(start), int(stop))
            parts.append((OBJ_ROW['trans_ag'], *support(np.arange(run.start, run.stop), -flow_cost[src][m[run], local_r[run], j[run]])))

    # ── transition costs on the ag → non-ag arcs: the cost dict is keyed by target land use k ──
    src_ptr = table.attrs['src_ptr']['ag2nonag']
    for src, start, stop in zip(cols['sources']['ag'], src_ptr[:-1], src_ptr[1:]):
        run = slice(int(start), int(stop))
        cost_by_k = econ.flow_cost_ag2nonag[src]                               # {k: array(ncells_src)}
        to_k = k[run]
        cells = local_r[run]
        arc_cost = np.empty(run.stop - run.start, dtype=np.float32)
        for lu in np.unique(to_k):
            arc_cost[to_k == lu] = cost_by_k[int(lu)][cells[to_k == lu]]
        parts.append((OBJ_ROW['trans_nonag'], *support(np.arange(run.start, run.stop), -arc_cost)))

    row_idx = np.concatenate([np.full(part_vals.size, block_row, dtype=np.int32) for block_row, _, part_vals in parts])
    col_idx = np.concatenate([part_cols for _, part_cols, _ in parts])
    vals = np.concatenate([part_vals for _, _, part_vals in parts])
    block = sparse.csr_matrix((vals, (row_idx, col_idx)), shape=(len(OBJ_BLOCKS), table.attrs['n_dec']))
    block.sum_duplicates()                                                # no column repeats; keeps CSR canonical
    block, _, _ = contract(block)                                         # the drop; the solver scales to million AUD and floors again
    return block


# ═══════════════════════════ the row table: every constraint as one row, the families back to back ═══════════════════════════
#
# The rows get what the columns have. Every family returns its rows as a PART (``make_part``) — an
# ``xr.Dataset`` on dim ``row`` with the family's key fields from ONE fixed schema (the union of every
# family's natural key: an int field is -1 where the family has none, a coded field carries its labels),
# per-row ``rhs`` (as stored in Gurobi), ``sense``, ``name`` (the ConstrName) and ``scale`` (the
# ``contract`` factor, 1 where not rescaled), and its block of A in ``attrs``. ``stack_rows`` lays the
# parts back to back into the ROW TABLE: dim ``row`` = Constr.index, the coded fields as int32 codes
# into ``attrs['vocab']``, ``attrs['family_range']`` = {family: (start, stop)} (the rows each family
# owns, as ``block_range`` does for the columns), ``attrs['keys']`` = {family: its key fields},
# ``attrs['A']`` the ONE CSR (rows × n_all), and ``active`` — a dropped row is flagged off, the table
# never shrinks. The solver adds ``constr`` (the Gurobi handle) after ``addMConstr`` and ``lhs`` (the
# raw-unit row value at the solution, dropped rows included) after the solve. Every generator is
# ``family(rows, cols) -> part | None`` (None = family off).

ROW_FIELDS_INT = ('cell', 'm', 'j', 'k', 'am_idx', 'from_m', 'from_j', 'from_k', 'local_r', 'commodity')   # -1 where n/a
ROW_FIELDS_CODED = ('family', 'group', 'region', 'item', 'presence', 'bound', 'state')                       # codes into attrs['vocab'][field]


def make_part(family: str, group: str, keys: dict, A: sparse.csr_matrix, rhs, sense, names, scale=None, **attrs) -> xr.Dataset:
    """One family's rows: ``keys`` = {field: labels per row} over the row schema (``ROW_FIELDS_INT`` as ints,
    ``ROW_FIELDS_CODED`` as labels); ``sense`` one character or one per row; ``A`` the family's block."""
    n_rows = A.shape[0]
    unknown = set(keys) - set(ROW_FIELDS_INT) - set(ROW_FIELDS_CODED)
    assert not unknown, f'{family}: key field(s) {unknown} are not in the row schema'
    fields = {field: (('row',), np.asarray(labels, dtype=np.int32 if field in ROW_FIELDS_INT else object)) for field, labels in keys.items()}
    sense = np.full(n_rows, sense, dtype=object) if isinstance(sense, str) else np.asarray(sense, dtype=object)
    return xr.Dataset(
        dict(**fields,
             rhs=(('row',), np.asarray(rhs, dtype=np.float64)),
             sense=(('row',), sense),
             name=(('row',), np.asarray(names, dtype=object)),
             scale=(('row',), np.ones(n_rows, dtype=np.float64) if scale is None else np.asarray(scale, dtype=np.float64))),
        attrs=dict(family=family, group=group, A=A, keys=list(keys), **attrs))


def stack_rows(parts: list) -> xr.Dataset:
    """The row table: the parts back to back in the order given (the model's row order), every field of
    the schema over every row, the coded fields encoded, the ONE A vstacked, every row active."""
    widths = [part.sizes['row'] for part in parts]
    bounds = np.cumsum([0, *widths])
    n_rows = int(bounds[-1])
    family_range = {part.attrs['family']: (int(start), int(stop)) for part, start, stop in zip(parts, bounds[:-1], bounds[1:])}

    def field(name, dtype, fill):
        """One field over the whole table: each part's array for it, or the fill where the part has no such field."""
        return np.concatenate([np.asarray(part[name].values if name in part else np.full(width, fill), dtype=dtype)
                               for part, width in zip(parts, widths)]) if parts else np.empty(0, dtype=dtype)

    vocab = {}
    fields = {name: field(name, np.int32, -1) for name in ROW_FIELDS_INT}
    for name in ROW_FIELDS_CODED:                                                         # labels -> codes, the vocabulary in order of first appearance
        code_of = {}
        codes = np.full(n_rows, -1, dtype=np.int32)
        for part, start, stop in zip(parts, bounds[:-1], bounds[1:]):
            if name in ('family', 'group'):                                               # one label per part
                codes[start:stop] = code_of.setdefault(part.attrs[name], len(code_of))
            elif name in part:                                                            # one label per row, on the parts that carry the field
                codes[start:stop] = [code_of.setdefault(label, len(code_of)) for label in part[name].values]
        vocab[name] = list(code_of)
        fields[name] = codes

    extra = {f"{part.attrs['family']}_{key}": value for part in parts for key, value in part.attrs.items()
             if key not in ('family', 'group', 'A', 'keys')}                              # a part's own attrs, family-prefixed (demand's q_block)
    return xr.Dataset(
        dict(**{name: (('row',), values) for name, values in fields.items()},
             rhs=(('row',), field('rhs', np.float64, np.nan)),
             sense=(('row',), field('sense', object, None)),
             name=(('row',), field('name', object, None)),
             scale=(('row',), field('scale', np.float64, 1.0)),
             active=(('row',), np.ones(n_rows, dtype=bool))),
        attrs=dict(A=sparse.vstack([part.attrs['A'] for part in parts], format='csr') if parts else None,
                   family_range=family_range,
                   keys={part.attrs['family']: part.attrs['keys'] for part in parts},
                   vocab=vocab,
                   **extra))


def family_rows(table: xr.Dataset, family: str) -> slice | None:
    """The rows one family owns (``attrs['family_range']``), None where the family was not built."""
    span = table.attrs['family_range'].get(family)
    return slice(*span) if span is not None else None


def decode(table: xr.Dataset, field: str, rows=None) -> np.ndarray:
    """A field's values at ``rows`` (a slice / mask / index array; every row by default) as labels: a coded
    field through its vocabulary (None where -1), an int field as it is."""
    values = table[field].values if rows is None else table[field].values[rows]
    if field not in ROW_FIELDS_CODED:
        return values
    labels = np.array([*table.attrs['vocab'][field], None], dtype=object)                 # -1 indexes the trailing None
    return labels[values]


def rows_where(table: xr.Dataset, **fields) -> np.ndarray:
    """A boolean mask over the table: the rows whose fields carry the given labels (``family='GBF8', region='AUSTRALIA'``)."""
    mask = np.ones(table.sizes['row'], dtype=bool)
    for field, label in fields.items():
        if field in ROW_FIELDS_CODED:
            vocab = table.attrs['vocab'][field]
            label = vocab.index(label) if label in vocab else -2                         # a label the table has never seen matches no row
        mask &= table[field].values == label
    return mask


def keys_of(table: xr.Dataset, family: str, rows=None) -> list:
    """The family's row keys as tuples (its key fields, decoded), at ``rows`` (its own rows by default), in row order."""
    rows = family_rows(table, family) if rows is None else rows
    columns = [decode(table, field, rows) for field in table.attrs['keys'][family]]
    n = table[table.attrs['keys'][family][0]].values[rows].size if columns else table['name'].values[rows].size
    return list(zip(*columns)) if columns else [()] * n


# ── the nodes and the runs: what the structural and flow rows group and join on ──────────────────

def ag_node(cols: dict, m, j, cell) -> np.ndarray:
    """The node id of an ag (m, j, cell): ``(m · n_lu + j) · ncells + cell`` — ascending in the ag block's own order."""
    n_lu, ncells = cols['ag'].sizes['lu'], cols['ag'].sizes['cell']
    return (np.asarray(m, dtype=np.int64) * n_lu + j) * ncells + cell


def nonag_node(cols: dict, k, cell) -> np.ndarray:
    """The node id of a non-ag (k, cell): ``k · ncells + cell`` numbered after every ag node."""
    nlms, n_lu, ncells = cols['ag'].sizes['lm'], cols['ag'].sizes['lu'], cols['ag'].sizes['cell']
    return nlms * n_lu * ncells + np.asarray(k, dtype=np.int64) * ncells + cell


def node_ids(cols: dict) -> tuple[np.ndarray, np.ndarray]:
    """Every column's nodes, int64, -1 where n/a: ``to_node`` — the (m, j, cell) an ag / ag-mgt / ag2ag /
    nonag2ag column lands on, the (k, cell) a non-ag / ag2nonag column lands on; ``from_node`` — the node
    an arc comes from. The flow rows group and join on these; the ag-mgt link joins an am column to its host."""
    table = cols['table']
    m, j, k, from_m, from_j, from_k, cell = (table[field].values for field in ('m', 'j', 'k', 'from_m', 'from_j', 'from_k', 'cell'))
    to_node = np.full(cell.size, -1, dtype=np.int64)
    from_node = np.full(cell.size, -1, dtype=np.int64)
    lands_nonag, lands_ag = k >= 0, (k < 0) & (j >= 0)
    to_node[lands_nonag] = nonag_node(cols, k[lands_nonag], cell[lands_nonag])
    to_node[lands_ag] = ag_node(cols, m[lands_ag], j[lands_ag], cell[lands_ag])
    leaves_nonag, leaves_ag = from_k >= 0, from_j >= 0
    from_node[leaves_nonag] = nonag_node(cols, from_k[leaves_nonag], cell[leaves_nonag])
    from_node[leaves_ag] = ag_node(cols, from_m[leaves_ag], from_j[leaves_ag], cell[leaves_ag])
    return to_node, from_node


def runs_of(sorted_key: np.ndarray, n_groups: int) -> np.ndarray:
    """The group bounds of a sorted key: ``ptr[g]:ptr[g+1]`` is the run of group g (empty where g is absent)."""
    return np.searchsorted(sorted_key, np.arange(n_groups + 1))


def ag_runs(cols: dict) -> np.ndarray:
    """The ag block grouped by (m, j) — ``ptr[m · n_lu + j]`` bounds its run of table rows, cells ascending."""
    table = cols['table']
    ag = block_slice(table, 'ag')
    return ag.start + runs_of(table['m'].values[ag] * cols['ag'].sizes['lu'] + table['j'].values[ag], cols['ag'].sizes['lm'] * cols['ag'].sizes['lu'])


def am_slots(cols: dict) -> tuple[list, np.ndarray]:
    """The (option, land use) slots in slot order, and the am block grouped by (slot, m) — ``ptr[slot · nlms + m]``
    bounds its run of table rows, cells ascending (a slot with no column has an empty run)."""
    table = cols['table']
    am = block_slice(table, 'am')
    pairs = [(option, lu) for option, lus in cols['table'].attrs['agman2lu'].items() for lu in lus]
    nlms = cols['ag'].sizes['lm']
    return pairs, am.start + runs_of(table['slot'].values[am] * nlms + table['m'].values[am], len(pairs) * nlms)


def row_of(row_keys: np.ndarray, node: np.ndarray) -> np.ndarray:
    """The row whose key is ``node`` (a join on ascending row keys), -1 where no row has it."""
    if not row_keys.size:
        return np.full(node.size, -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(row_keys, node), row_keys.size - 1)
    return np.where(row_keys[pos] == node, pos, -1)


# ── spine: the structural rows ────────────────────────────────────────────────────────────────

def renewable_ceiling_rows(rows: RowInputs, cols: dict):
    """Simulated and existing renewable capacity compete for the cell's space [0, ag_mask]: one row per
    (am, cell) with existing capacity, Σ_{m, j} X_am[am, m, j, r] ≤ max(ag_mask[r] − exist_r[r], 0)."""
    table = cols['table']
    n_all = table.attrs['n_all']
    am = block_slice(table, 'am')
    am_idx, cell = table['am_idx'].values, table['cell'].values
    ag_mask = rows.ag_mask_proportion_r
    parts = []
    names = []
    key_am = []
    key_cell = []
    rhs = []
    for option in cols['table'].attrs['agman2lu']:
        if option not in settings.RENEWABLES_OPTIONS:
            continue
        am_name = tools.am_name_snake_case(option)
        exist_r = rows.exist_renewable_solar_r if option == "Utility Solar PV" else rows.exist_renewable_wind_r   # the total across ALL data years: the ceiling never decreases between periods, so lb(t) <= ceiling always holds
        # the option's columns, grouped by cell
        col_idx = am.start + np.flatnonzero(am_idx[am] == table.attrs['options'].index(option))
        cells, cell_of_col = np.unique(cell[col_idx], return_inverse=True)   # the option's cells, ascending
        existing_cap = exist_r[cells]
        keep_cell = existing_cap != 0                                    # no existing capacity -> no ceiling row
        n_rows = int(keep_cell.sum())
        if not n_rows:
            continue
        row_of_cell = np.full(cells.size, -1, dtype=np.int64)
        row_of_cell[keep_cell] = np.arange(n_rows)
        row_idx = row_of_cell[cell_of_col]
        in_row = row_idx >= 0
        parts.append(sparse.csr_matrix((np.ones(int(in_row.sum())), (row_idx[in_row], col_idx[in_row])), shape=(n_rows, n_all)))
        rhs.append(np.maximum(ag_mask[cells[keep_cell]] - existing_cap[keep_cell], 0.0))   # cell space left for simulated capacity
        names += [f"const_{am_name}_solvable_ub_{r}".replace(" ", "_") for r in cells[keep_cell]]
        key_am += [table.attrs['options'].index(option)] * n_rows
        key_cell.append(cells[keep_cell])
    if not parts:
        return None
    A, rhs, _ = contract(sparse.vstack(parts, format='csr'), np.concatenate(rhs))
    return make_part('renewable_ceiling', 'ag_mgt_ub', dict(am_idx=key_am, cell=np.concatenate(key_cell)), A, rhs, '<', names)


def cell_usage_rows(rows: RowInputs, cols: dict):
    """Every cell's ag + non-ag shares sum to its base-year agricultural proportion: one row per cell
    with a slack column, stored the way Gurobi stores an addRange row (Σ X + slack = hi)."""
    table = cols['table']
    n_all = table.attrs['n_all']
    cell = table['cell'].values
    ag, nonag, slack = (block_slice(table, block) for block in ('ag', 'nonag', 'cell_usage'))
    row_cells = cell[slack]
    n_rows = row_cells.size
    ncells = cols['ag'].sizes['cell']
    row_of_cell = np.full(ncells, -1, dtype=np.int64)
    row_of_cell[row_cells] = np.arange(n_rows)
    columns = np.concatenate([np.arange(ag.start, ag.stop), np.arange(nonag.start, nonag.stop), np.arange(slack.start, slack.stop)])
    row_idx = row_of_cell[cell[columns]]
    in_row = row_idx >= 0
    A = sparse.csr_matrix((np.ones(int(in_row.sum())), (row_idx[in_row], columns[in_row])), shape=(n_rows, n_all))
    # Ranged, not ==: presolve folds the node-balance rows into this one and compares two constants summed
    # along different float32 paths (up to ~1.75x FeasibilityTol apart) with NO tolerance. The +-10x Ftol band
    # absorbs that; conservation still pins the cell total, so the band is not exploitable.
    hi = rows.ag_mask_proportion_r[row_cells].astype(np.float64) + 10 * settings.FEASIBILITY_TOLERANCE   # the top of the band (widened before the band is applied)
    A, hi, _ = contract(A, hi)
    return make_part('cell_usage', 'cell_usage', dict(cell=row_cells), A, hi, '=',
                      [f"const_cell_usage_{cell}" for cell in row_cells], n_skipped=int(ncells - n_rows))


def ag_mgt_link_rows(rows: RowInputs, cols: dict):
    """Ag-management variables cannot exceed their agricultural variable: one row per (am, land use, lm,
    cell) with an ag column — X_am − X_ag ≤ 0, or X_ag ≥ 0 where the am column does not exist."""
    table = cols['table']
    n_all = table.attrs['n_all']
    nlms, n_lu = cols['ag'].sizes['lm'], cols['ag'].sizes['lu']
    cell = table['cell'].values
    ag = block_slice(table, 'ag')
    to_node, _ = node_ids(cols)
    ag_node_of_col = to_node[ag]                                         # ascending: the ag block is in (m, j, cell) order
    ag_ptr = ag_runs(cols)
    pairs, am_ptr = am_slots(cols)
    options = table.attrs['options']
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
    for slot, (option, j) in enumerate(pairs):
        for m, lm in ((0, 'dry'), (1, 'irr')):
            ag_cols = np.arange(ag_ptr[m * n_lu + j], ag_ptr[m * n_lu + j + 1])         # the ag columns of (m, j), cells ascending: one row each
            am_cols = np.arange(am_ptr[slot * nlms + m], am_ptr[slot * nlms + m + 1])   # the slot's am columns at m ...
            host = np.searchsorted(ag_node_of_col, to_node[am_cols]) - ag_ptr[m * n_lu + j]   # ... each joined to the ag column it sits on (its host node), as a row of this run
            has_am = np.zeros(ag_cols.size, dtype=bool)                                 # no am column: GBF2-excluded or savanna-ineligible cell
            has_am[host] = True
            slot_rows = n_rows + np.arange(ag_cols.size)
            # X_ag: −1 on the '<' rows (X_am − X_ag ≤ 0), +1 on the '>' rows (X_ag ≥ 0)
            row_idx.append(slot_rows)
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


def ag_mgt_adoption_rows(rows: RowInputs, cols: dict):
    """Adoption limits: one row per (am, land use), Σ am columns − limit · Σ ag columns ≤ 0
    (Σam ≤ limit · Σag with the RHS moved to the LHS); zero coefficients (limit = 0) are dropped."""
    table = cols['table']
    n_all = table.attrs['n_all']
    nlms, n_lu = cols['ag'].sizes['lm'], cols['ag'].sizes['lu']
    ag_ptr = ag_runs(cols)
    pairs, am_ptr = am_slots(cols)
    row_idx = []
    col_idx = []
    vals = []
    names = []
    key_am = []
    key_lu = []
    for row, (option, j) in enumerate(pairs):
        adoption_limit = float(np.float64(rows.ag_man_limits[option][j]))
        am_cols = np.arange(am_ptr[row * nlms], am_ptr[row * nlms + nlms])                                # the slot's am columns, both lm (adjacent runs)
        ag_cols = np.concatenate([np.arange(ag_ptr[m * n_lu + j], ag_ptr[m * n_lu + j + 1]) for m in range(nlms)])   # the ag columns of j, both lm
        row_idx += [np.full(am_cols.size, row), np.full(ag_cols.size, row)]
        col_idx += [am_cols, ag_cols]
        vals += [np.ones(am_cols.size), np.full(ag_cols.size, -adoption_limit)]
        names.append(f"const_ag_mam_adoption_limit_{option}_{j}".replace(" ", "_"))
        key_am.append(table.attrs['options'].index(option))
        key_lu.append(j)
    n_rows = len(names)
    A, _, _ = contract(sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_rows, n_all)))
    return make_part('ag_mgt_adoption', 'ag_mgt_adopt', dict(am_idx=key_am, j=key_lu),
                      A, np.zeros(n_rows), '<', names)


# ── policy families: gather a coefficient per scored column, compose it with the row's weights over cells, contract with the rescale ──

def demand_rows(rows: RowInputs, cols: dict):
    """Hard demand constraints: one per-commodity quantity row over the scored columns, used once under
    '=' where the DEMAND_BOUNDS lb == ub, else twice — under '>' lb and '<' ub."""
    print("│   ├── Adding <hard> demand constraints (equality where lb==ub, else lower + upper)...")
    table = cols['table']
    n_all = table.attrs['n_all']
    k, cell = table['k'].values, table['cell'].values
    nonag = block_slice(table, 'nonag')
    nlms, n_lu = cols['ag'].sizes['lm'], cols['ag'].sizes['lu']
    ag_ptr = ag_runs(cols)                                               # the ag columns of (m, j): one run each
    pairs, am_ptr = am_slots(cols)                                       # the am columns of (slot, m): one run each
    ncms = rows.ncms
    row_idx = []
    col_idx = []
    vals = []

    def put(commodity_coeffs: np.ndarray, columns: np.ndarray):
        """One (commodity × column) coefficient block into the COO lists — its nonzero support; the contract drops."""
        for c_idx in range(ncms):
            kept_cols, kept_vals = support(columns, commodity_coeffs[c_idx])
            row_idx.append(np.full(kept_cols.size, c_idx, dtype=np.int32))
            col_idx.append(kept_cols)
            vals.append(kept_vals)

    # ── the per-commodity LHS (q_block, kept unscaled in attrs for production reporting): the ag columns
    #    per (m, land use) and the ag-mgt columns per (slot, m) carry jc[c, cell] = Σ_p pr2cm[c, p] · q[m, cell, p]
    #    over the land use's active products; the non-ag columns per k carry non_ag_q_crk[c, cell, k] ──
    for lu in range(n_lu):
        active_p = np.where(rows.lu2pr_pj[:, lu])[0]
        if not active_p.size:
            continue
        for lm in range(nlms):
            group = np.arange(ag_ptr[lm * n_lu + lu], ag_ptr[lm * n_lu + lu + 1])
            if group.size:
                put(rows.pr2cm_cp[:, active_p] @ rows.ag_q_mrp[lm, cell[group], :][:, active_p].T, group)
    for slot, (option, lu) in enumerate(pairs):
        active_p = np.where(rows.lu2pr_pj[:, lu])[0]
        if not active_p.size:
            continue
        for lm in range(nlms):
            group = np.arange(am_ptr[slot * nlms + lm], am_ptr[slot * nlms + lm + 1])
            if group.size:
                put(rows.pr2cm_cp[:, active_p] @ rows.ag_man_q_mrp[option][lm, cell[group], :][:, active_p].T, group)
    for lu in np.unique(k[nonag]):
        group = nonag.start + np.flatnonzero(k[nonag] == lu)
        put(rows.non_ag_q_crk[:, cell[group], lu], group)
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
    for c_idx, c_name in enumerate(rows.commodity_names):
        lb, ub = settings.DEMAND_BOUNDS[c_name]
        demand = rows.limits['demand'][c_idx]
        bounds = [('eq', '=', lb)] if lb == ub else [('lower', '>', lb), ('upper', '<', ub)]
        for bound, sense, factor in bounds:
            lhs_row.append(c_idx)
            senses.append(sense)
            rhs.append(demand * factor)
            names.append(f"demand_hard_bound_{bound}[{c_idx}]")
            key_commodity.append(c_idx)
            key_bound.append(bound)
    block, rhs, scale = contract(q_block[lhs_row], rhs, rescale=True)   # row rescale, factors kept
    return make_part('demand', 'demand', dict(commodity=key_commodity, bound=key_bound),
                      block, rhs, np.array(senses, dtype=object), names, scale, q_block=q_block)


def ghg_rows(rows: RowInputs, cols: dict):
    """Hard GHG emissions cap: one global row over land-use, ag-management, non-ag and
    transition-arc emissions, Σ ghg · X ≤ limit − offland."""
    if settings.GHG_EMISSIONS_LIMITS == "off":
        print("│   ├── TURNING OFF GHG emissions constraints ...")
        return None
    ghg_limit_raw = rows.limits["ghg"]
    print(f"│   ├── Adding <hard> constraints for GHG emissions: {ghg_limit_raw:,.0f} tCO2e")
    table = cols['table']
    n_all = table.attrs['n_all']
    m, j, local_r = (table[field].values for field in ('m', 'j', 'local_r'))
    # land-use, ag-management and non-ag emissions on the scored columns
    coeff = gather(table, rows.ag_g_mrj, rows.ag_man_g_mrj, rows.non_ag_g_rk)
    kept_cols, kept_vals = support(np.arange(coeff.size), coeff)
    col_idx = [kept_cols]
    vals = [kept_vals]
    # transition emissions on the ag → ag arcs: per source run of the block, a float32 gather of the delta emissions
    src_ptr = table.attrs['src_ptr']['ag2ag']
    for src, start, stop in zip(cols['sources']['ag'], src_ptr[:-1], src_ptr[1:]):
        run = slice(int(start), int(stop))
        if run.stop == run.start:
            continue
        kept_cols, kept_vals = support(np.arange(run.start, run.stop), rows.trans_ghg_ag2ag[src][m[run], local_r[run], j[run]])
        col_idx.append(kept_cols)
        vals.append(kept_vals)
    col_idx = np.concatenate(col_idx)
    vals = np.concatenate(vals)
    row = sparse.csr_matrix((vals, (np.zeros(col_idx.size, dtype=np.int32), col_idx)), shape=(1, n_all))
    row.sum_duplicates()
    row.sort_indices()
    rhs = np.asarray(ghg_limit_raw - rows.offland_ghg, dtype=np.float64).ravel()   # offland_ghg: 1-element array
    row, rhs, scale = contract(row, rhs, rescale=True)                   # drop + row rescale, factor kept
    return make_part('ghg', 'ghg', {}, row, rhs, '<', ["ghg_emissions_limit_ub"], scale)


def _bio_block(family, group, key_names, rows: RowInputs, cols: dict, pairs, v_limits, layer_of, skip_nonpositive: bool, name_of):
    """Shared body of the GBF families: one weighting row per active key (``layer_of(key)``, region-masked),
    composed with the biodiversity contribution and row-rescaled; keys with no target or no cell are skipped."""
    bio_c = gather(cols['table'], *bio_streams(rows, cols))
    reg_matrix = rows.region_NRM_names_r
    val_rows = []
    names = []
    rhs = []
    kept = []
    for key in pairs:
        lb_raw = v_limits.sel(dict(layer=key)).item()
        if (lb_raw <= 0) if skip_nonpositive else (lb_raw < 0):       # GBF4/8 skip a zero target; GBF3 still adds a row for one
            continue
        val_row = layer_of(key)
        region = key[0]
        if region != "AUSTRALIA":                                        # NRM scope: mask non-region cells
            val_row = np.where(reg_matrix == region, val_row, 0)
        if not (val_row > 0).any():
            continue
        val_rows.append(val_row)
        names.append(name_of(key))
        rhs.append(lb_raw)
        kept.append(key)
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")
    if not val_rows:
        return None
    block, rhs, scale = contract(compose_rows(cols, bio_c, val_rows), rhs, rescale=True)   # compose, then drop + row rescale, factors kept
    keys = {field: [key[i] for key in kept] for i, field in enumerate(key_names)}
    return make_part(family, group, keys, block, rhs, '>', names, scale)


def gbf2_rows(rows: RowInputs, cols: dict):
    """GBF2 priority degraded areas: the bio contribution at every scored column weighted by
    GBF2_mask_area_r, which is ZERO off-mask (off-mask columns get coefficient 0 and are dropped). One row."""
    if settings.GBF2_TARGET == "off":
        print("│   │   ├── TURNING OFF constraints for biodiversity GBF 2...")
        return None
    print(f'│   │   ├── Adding constraints for biodiversity GBF 2: {rows.limits["GBF2"]:15,.0f}')
    row = compose_rows(cols, gather(cols['table'], *bio_streams(rows, cols)), [rows.GBF2_mask_area_r])
    row, rhs, scale = contract(row, [rows.limits["GBF2"]], rescale=True)   # drop + row rescale, factor kept
    return make_part('GBF2', 'bio_gbf2', {}, row, rhs, '>',
                      ["bio_GBF2_priority_degraded_area_limit"], scale)


def gbf3_rows(rows: RowInputs, cols: dict):
    if settings.GBF3_NVIS_TARGET == "off":
        print("│   │   ├── TURNING OFF constraints for biodiversity GBF 3 NVIS")
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 3 NVIS...")
    val_matrix = rows.GBF3_NVIS_pre_1750_area_vr                        # xr [group, cell]
    return _bio_block('GBF3_NVIS', 'bio_nvis', ('region', 'item'), rows, cols, rows.GBF3_NVIS_region_group, rows.limits["GBF3_NVIS"],
                      lambda key: val_matrix.sel(group=key[1], drop=True).data, False,
                      lambda key: f"bio_GBF3_NVIS_limit_{key[0]}_{key[1]}".replace(" ", "_"))


def gbf4_snes_rows(rows: RowInputs, cols: dict):
    if settings.GBF4_TARGET_SNES == 'off':
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 SNES...')
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 4 SNES ...")
    val_matrix = rows.GBF4_SNES_pre_1750_area_sr                        # xr [layer=(species, presence), cell]
    return _bio_block('GBF4_SNES', 'bio_snes', ('region', 'item', 'presence'), rows, cols, rows.GBF4_SNES_region_species, rows.limits["GBF4_SNES"],
                      lambda key: val_matrix.sel(dict(layer=(key[1], key[2])), drop=True).values, True,
                      lambda key: f"bio_GBF4_SNES_limit_{key[0]}_{key[1]}_{key[2]}".replace(" ", "_"))


def gbf4_ecnes_rows(rows: RowInputs, cols: dict):
    if settings.GBF4_TARGET_ECNES == 'off':
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 ECNES...')
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 4 ECNES ...")
    val_matrix = rows.GBF4_ECNES_pre_1750_area_sr                       # xr [layer=(community, presence), cell]
    return _bio_block('GBF4_ECNES', 'bio_ecnes', ('region', 'item', 'presence'), rows, cols, rows.GBF4_ECNES_region_species, rows.limits["GBF4_ECNES"],
                      lambda key: val_matrix.sel(dict(layer=(key[1], key[2])), drop=True).values, True,
                      lambda key: f"bio_GBF4_ECNES_limit_{key[0]}_{key[1]}_{key[2]}".replace(" ", "_"))


def gbf8_rows(rows: RowInputs, cols: dict):
    if settings.GBF8_TARGET == "off":
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 8 ...')
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 8 ...")
    val_matrix = rows.GBF8_pre_1750_area_sr                             # xr [species, cell]
    return _bio_block('GBF8', 'bio_gbf8', ('region', 'item'), rows, cols, rows.GBF8_region_species, rows.limits["GBF8"],
                      lambda key: val_matrix.sel(species=key[1], drop=True).data, True,
                      lambda key: f"bio_GBF8_limit_{key[0]}_{key[1]}".replace(" ", "_"))


def _regional_adoption_family(family, group, key_names, caps, cols_all, r_all, sel_of, name_of, rhs_of, rows, cols):
    """One regional-adoption block: Σ real_area[r] · X over the region's cells ≤ cap per (region, land use)."""
    n_all = cols['table'].attrs['n_all']
    real_area = rows.real_area
    parts = []
    names = []
    rhs = []
    keys = []
    for cap in caps:
        region_cells = cap[-2]
        if len(region_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name_of(cap)} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name_of(cap)} <= {cap[-1]:,.0f} HA...")
        selected = np.flatnonzero(sel_of(cap, region_cells))
        parts.append(support(cols_all[selected], real_area[r_all[selected]].astype(np.float32)))
        names.append(name_of(cap))
        rhs.append(rhs_of(cap))
        keys.append(cap[:len(key_names)])
    if not parts:
        return None
    row_idx = np.concatenate([np.full(part_cols.size, row) for row, (part_cols, _) in enumerate(parts)])
    A = sparse.csr_matrix((np.concatenate([part_vals for _, part_vals in parts]), (row_idx, np.concatenate([part_cols for part_cols, _ in parts]))),
                          shape=(len(parts), n_all))
    A, rhs, _ = contract(A, rhs)                                          # the drop only: hectares are not rescaled (the shadow-price reader assumes scale 1)
    return make_part(family, group, {field: [key[i] for key in keys] for i, field in enumerate(key_names)},
                      A, rhs, '<', names)


def regional_adoption_ag_rows(rows: RowInputs, cols: dict):
    """Per-(region, ag land use) caps ('on' mode). Not rescaled (hectares)."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        print("│   │   └── TURNING OFF constraints for regional adoption ...")
        return None
    table = cols['table']
    ag = block_slice(table, 'ag')
    ag_j, ag_r = table['j'].values[ag], table['cell'].values[ag]
    ag_cols = np.arange(ag.start, ag.stop)
    return _regional_adoption_family(
        'regional_adoption_ag', 'adopt_ag', ('region', 'j'), rows.limits["ag_regional_adoption"], ag_cols, ag_r,
        lambda cap, reg_ind: (ag_j == cap[1]) & np.isin(ag_r, reg_ind),
        lambda cap: f"reg_adopt_limit_ag_{cap[2]}_{cap[0]}".replace(" ", "_"), lambda cap: cap[4], rows, cols)


def _nonag_cap_relax(rows: RowInputs) -> float:
    """Grow the non-ag caps by 1e-6/yr RELATIVE, so the RHS always recedes ahead of the ratcheting lower
    bound non-reversible plantings create. Ag caps need no slack: ag is reversible."""
    # Last year's solved areas become this year's exact lower bounds; float32 noise then puts the locked-in
    # floor a hair over a saturated cap, which presolve rejects with NO tolerance. Per-step increment
    # ~5e-6 x cap vs float noise ~2e-10 x cap; cap erosion by 2050 ~3e-5 relative.
    return 1 + (rows.target_year - settings.SIM_YEARS[0]) * 1e-6


def regional_adoption_nonag_rows(rows: RowInputs, cols: dict):
    """Per-(region, non-ag land use) caps ('on' mode), with the per-year relaxation."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    table = cols['table']
    nonag = block_slice(table, 'nonag')
    na_k, na_r = table['k'].values[nonag], table['cell'].values[nonag]
    na_cols = np.arange(nonag.start, nonag.stop)
    relax = _nonag_cap_relax(rows)
    return _regional_adoption_family(
        'regional_adoption_nonag', 'adopt_nonag', ('region', 'k'), rows.limits.get("non_ag_regional_adoption") or [], na_cols, na_r,
        lambda cap, reg_ind: (na_k == cap[1]) & np.isin(na_r, reg_ind),
        lambda cap: f"reg_adopt_limit_non_ag_{cap[2]}_{cap[0]}".replace(" ", "_"), lambda cap: cap[4] * relax, rows, cols)


def regional_adoption_nonag_sum_rows(rows: RowInputs, cols: dict):
    """SUM-of-non-ag caps ('NON_AG_CAP' mode): all non-ag land uses in a region together."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    table = cols['table']
    nonag = block_slice(table, 'nonag')
    na_r = table['cell'].values[nonag]
    na_cols = np.arange(nonag.start, nonag.stop)
    relax = _nonag_cap_relax(rows)
    return _regional_adoption_family(
        'regional_adoption_nonag_sum', 'nonag_cap', ('region',), rows.limits.get("non_ag_regional_adoption_sum") or [], na_cols, na_r,
        lambda cap, reg_ind: np.isin(na_r, reg_ind),
        lambda cap: f"reg_adopt_limit_non_ag_sum_{cap[0]}".replace(" ", "_"), lambda cap: cap[2] * relax, rows, cols)


def water_rows(rows: RowInputs, cols: dict):
    """Water net-yield limits: one row per water region, the region's 0/1 float32 indicator as the
    weighting row over the scored columns (off-region columns give q = 0 and are dropped)."""
    if settings.WATER_LIMITS != "on":
        print("│   ├── TURNING OFF water usage constraints ...")
        return None
    print("│   ├── Adding constraints for water usage limits...")
    coeff = gather(cols['table'], rows.ag_w_mrj, rows.ag_man_w_mrj, rows.non_ag_w_rk)
    val_rows = []
    names = []
    rhs = []
    region_ids = []
    for region_id, water_limit_raw in rows.limits["water"].items():
        region_name = rows.water_region_names[region_id]
        print(f"│   │   ├── target (inside LUTO study area) is {water_limit_raw:15,.0f} ML for {region_name}")
        indicator = np.zeros(cols['ag'].sizes['cell'], dtype=np.float32)   # 1.0f x c == c, so the drop test sees the raw coefficient — which can be NEGATIVE
        indicator[rows.water_region_indices[region_id]] = 1.0
        val_rows.append(indicator)
        names.append(f"water_yield_limit_{region_name}".replace(" ", "_"))
        rhs.append(water_limit_raw)
        region_ids.append(region_id)
    if not val_rows:
        return None
    block, rhs, scale = contract(compose_rows(cols, coeff, val_rows), rhs, rescale=True)   # compose, then drop + row rescale, factors kept
    return make_part('water', 'water', dict(region=region_ids), block, rhs, '>', names, scale)


def renewable_rows(rows: RowInputs, cols: dict):
    """State-level renewable generation targets: one row per (state, type) — the type's ag-mgt columns
    weighted by an allowed-cells indicator, RHS = target − existing capacity."""
    if not any(settings.RENEWABLES_OPTIONS.values()):
        print("│   ├── TURNING OFF renewable energy constraints ...")
        return None
    print("│   ├── Adding constraints for renewable energy production targets ...")
    re_types = {
        'Utility Solar PV': dict(energy_r=rows.renewable_solar_r, gbf2_mask_idx=cols['mask_gbf2_solar'], mnes_mask_idx=cols['mask_mnes_solar']),
        'Onshore Wind':     dict(energy_r=rows.renewable_wind_r,  gbf2_mask_idx=cols['mask_gbf2_wind'],  mnes_mask_idx=cols['mask_mnes_wind']),
    }
    region_state_name2idx = dict(rows.region_state_name2idx)                # local copy: pop() must not mutate data's dict
    act_code = region_state_name2idx.pop('Australian Capital Territory')
    table = cols['table']
    n_terms = table.attrs['n_terms']
    cell, am_idx = table['cell'].values[:n_terms], table['am_idx'].values[:n_terms]
    ncells = cols['ag'].sizes['cell']

    # ── the coefficient per type: energy_r on that type's ag-mgt columns, 0 on every other scored column ──
    coeff_of_type = {}
    for option, re_data in re_types.items():
        if option in table.attrs['options']:
            on_type = am_idx == table.attrs['options'].index(option)
            coeff_of_type[option] = np.where(on_type, re_data['energy_r'][cell], np.float32(0.0)).astype(np.float32)   # float32 yield per cell

    # ── one row per (state, type) with eligible cells ──
    ag_ptr = ag_runs(cols)
    n_lu = cols['ag'].sizes['lu']
    cells_of_lu = {j: np.unique(np.concatenate([cell[ag_ptr[m * n_lu + j]:ag_ptr[m * n_lu + j + 1]] for m in range(cols['ag'].sizes['lm'])]))
                   for j in range(n_lu)}                                  # the cells with an ag column of j, either lm
    agman2lu = cols['table'].attrs['agman2lu']
    parts = []
    names = []
    rhs = []
    key_am = []
    key_state = []
    for state_name, state_code in region_state_name2idx.items():
        state_cells = np.where(rows.region_state_r == state_code)[0]
        if state_name == 'New South Wales':                              # ACT counts toward the NSW+ACT target
            state_cells = np.union1d(state_cells, np.where(rows.region_state_r == act_code)[0])
        print(f"│   │   ├── Adding renewable energy constraints for {state_name} ...")
        for am, re_data in re_types.items():
            if not settings.AG_MANAGEMENTS[am]:
                continue
            target_raw = rows.limits[f"renewable_{am}"][state_name]
            exist_power_mwh = rows.limits[f"renewable_{am}_exist"][state_name]
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
            parts.append(compose_rows(cols, coeff_of_type[am], [allowed]))
            names.append(f"renewable_{am}_target_{state_name}".replace(" ", "_"))
            rhs.append(target_raw - exist_power_mwh)                     # raw MWh; row-rescaled below
            key_am.append(table.attrs['options'].index(am))
            key_state.append(state_name)
    if not parts:
        return None
    block, rhs, scale = contract(sparse.vstack(parts, format='csr'), rhs, rescale=True)   # drop + row rescale, factors kept
    return make_part('renewable', 'renewable', dict(am_idx=key_am, state=key_state),
                      block, rhs, '>', names, scale)


# ── the transition-flow rows: group-bys and joins on the nodes of the table (node_ids) ──
#    group the arcs by from_node             →  source cap:    Σ out ≤ base                 (a ≤ row per source node)
#    join every column on its node, both roles  →  node balance:  X = base + Σ in − Σ out   (an = row per node)
#    the inflow cap is not a row: X's own ub (the transition upper bound) and the cell-usage row bound it.

def source_cap_ag_rows(rows: RowInputs, cols: dict):
    """Source cap, ag sources: the arcs leaving an ag node (``from_j >= 0``: ag2ag ∪ ag2nonag) grouped by
    ``from_node``; each group's arcs sum to at most its base share, Σ out ≤ base[from_m, r, from_j]."""
    # bounds the arc columns (some flow costs are negative) and rules out pass-through
    print("│   ├── Adding source-cap (Σ out ≤ base) constraints...")
    table = cols['table']
    n_all = table.attrs['n_all']
    _, from_node = node_ids(cols)
    arcs = np.flatnonzero(table['from_j'].values >= 0)
    from_m, from_j, local_r, cell = (table[field].values[arcs] for field in ('from_m', 'from_j', 'local_r', 'cell'))
    # one row per source node; every arc leaving it gets a +1
    nodes, first_arc, row_of_arc = np.unique(from_node[arcs], return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(nodes.size, n_all))
    rhs = cols['ag']['base'].values[from_m[first_arc], from_j[first_arc], cell[first_arc]].astype(np.float64)   # from the ag grid: a source with no X column still caps its outflow
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_a_{m}_{j}_{r}" for m, j, r in zip(from_m[first_arc], from_j[first_arc], local_r[first_arc])]
    return make_part('source_cap_ag', 'flow_out', dict(from_m=from_m[first_arc], from_j=from_j[first_arc], local_r=local_r[first_arc]),
                      A, rhs, '<', names)


def source_cap_nonag_rows(rows: RowInputs, cols: dict):
    """Source cap, non-ag sources: the arcs leaving a non-ag node (``from_k >= 0``: nonag2ag) grouped by
    ``from_node``; Σ out ≤ base_nonag[r, from_k]."""
    table = cols['table']
    n_all = table.attrs['n_all']
    _, from_node = node_ids(cols)
    arcs = np.flatnonzero(table['from_k'].values >= 0)
    if not arcs.size:
        return None
    from_k, local_r, cell = (table[field].values[arcs] for field in ('from_k', 'local_r', 'cell'))
    nodes, first_arc, row_of_arc = np.unique(from_node[arcs], return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(nodes.size, n_all))
    rhs = cols['nonag']['base'].values[from_k[first_arc], cell[first_arc]].astype(np.float64)
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_n_{k}_{r}" for k, r in zip(from_k[first_arc], local_r[first_arc])]
    return make_part('source_cap_nonag', 'flow_out', dict(from_k=from_k[first_arc], local_r=local_r[first_arc]), A, rhs, '<', names)


def node_balance_rows(rows: RowInputs, cols: dict):
    """Node balance, X = base + Σ in − Σ out at every node (m, j, cell) or (k, cell): one row per ag
    column, then one per (non-ag land use, feasible cell). The inflow cap is X's own ub, not a row."""
    print("│   └── Adding node-balance (X = base + Σin − Σout) constraints...")
    table = cols['table']
    n_all = table.attrs['n_all']
    m, j, cell = (table[field].values for field in ('m', 'j', 'cell'))
    ag, nonag = block_slice(table, 'ag'), block_slice(table, 'nonag')
    to_node, from_node = node_ids(cols)

    # ── the rows: one per ag column, then one per (non-ag land use, feasible cell) — X column or not — keyed by node, ascending ──
    n_ag = ag.stop - ag.start
    ag_m, ag_j, ag_r = m[ag].astype(np.int64), j[ag].astype(np.int64), cell[ag].astype(np.int64)
    nonag_k, nonag_r = np.nonzero(cols['nonag']['ub'].values > 0)       # every feasible entry, enabled land use or not: k then cell
    nonag_k = nonag_k.astype(np.int64)
    nonag_r = nonag_r.astype(np.int64)
    n_nonag = nonag_r.size
    row_keys = np.concatenate([to_node[ag], nonag_node(cols, nonag_k, nonag_r)])   # the ag block is in (m, j, cell) order, np.nonzero in (k, cell) order: ascending
    row_sign = np.ones(n_ag + n_nonag, dtype=np.float64)                 # a disabled land use has no X column: its row is a pure inflow guard, Σin − Σout = −base
    nonag_cols = np.arange(nonag.start, nonag.stop)
    row_sign[n_ag:] = -1.0
    row_sign[row_of(row_keys, to_node[nonag_cols])] = 1.0

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
    add(row_of(row_keys, to_node[nonag_cols]), nonag_cols, 1.0)
    add(row_of(row_keys, to_node[arcs]), arcs, -1.0)
    add(row_of(row_keys, from_node[arcs]), arcs, 1.0)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_ag + n_nonag, n_all))

    # ── rhs, names, keys ──
    rhs = np.concatenate([table['base'].values[ag].astype(np.float64),
                          cols['nonag']['base'].values[nonag_k, nonag_r].astype(np.float64) * row_sign[n_ag:]])
    names = ([f"bal_a_{m}_{j}_{r}" for m, j, r in zip(ag_m, ag_j, ag_r)] + [f"bal_n_{k}_{r}" for k, r in zip(nonag_k, nonag_r)])
    keys = dict(m=np.concatenate([ag_m, np.full(n_nonag, -1, dtype=np.int64)]),                    # an ag row carries its (m, j) node ...
                j=np.concatenate([ag_j, np.full(n_nonag, -1, dtype=np.int64)]),
                k=np.concatenate([np.full(n_ag, -1, dtype=np.int64), nonag_k]),                  # ... a non-ag row its k
                cell=np.concatenate([ag_r, nonag_r]))
    A, rhs, _ = contract(A, rhs)
    return make_part('node_balance', 'flow_in', keys, A, rhs, '=', names)


def biodiversity_rows(rows: RowInputs, cols: dict):
    """The five GBF families in order, as a list of parts (None where a family is off)."""
    print("│   ├── Adding constraints for biodiversity...")
    return [gbf2_rows(rows, cols), gbf3_rows(rows, cols), gbf4_snes_rows(rows, cols), gbf4_ecnes_rows(rows, cols), gbf8_rows(rows, cols)]


# The model's row order: every family, in the order the rows are added to Gurobi (the solver's
# path depends on it — ceilings first, the flow rows last). A generator may return None (family
# off), a part, or a list of parts / Nones; stack_rows lays them into the row table in this order.
FAMILIES = (
    renewable_ceiling_rows,
    cell_usage_rows,
    ag_mgt_link_rows,
    ag_mgt_adoption_rows,
    demand_rows,
    ghg_rows,
    biodiversity_rows,
    regional_adoption_ag_rows,
    regional_adoption_nonag_rows,
    regional_adoption_nonag_sum_rows,
    water_rows,
    renewable_rows,
    source_cap_ag_rows,
    source_cap_nonag_rows,
    node_balance_rows,
)
