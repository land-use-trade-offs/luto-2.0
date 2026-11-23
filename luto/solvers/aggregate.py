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

"""CELL GROUPING (settings.CELL_GROUPING, default off): cells that are alike share one decision, so the model the solver
sees is smaller. A pure function of the step's tables, between ``get_obj`` and ``step.solve_step``:

    row_matrix_grouped = row_map @ row_matrix @ col_map      obj_grouped = col_map.T @ obj      x = col_map @ x_grouped

A BLOB is a set of cells with the same key and the same SVD bin (``get_blobs``):

    the key (exact)     the column types the cell holds, each with its base; the region of the cell in every layer of
                        ``col_support.region2cell``; Riparian Plantings' upper bound in natural-break bins; the set
                        of GBF rows the cell has a coefficient in (their footprints)
    the SVD (close)     the cell's coefficients per hectare in the objective and the wide rows, reduced to
                        settings.CELL_GROUPING_SVD_DIMS directions and binned at settings.CELL_GROUPING_TOLERANCE

A grouped column is one (blob, column type): every cell of the blob takes its share, so its coefficient in a wide row
and in the objective is the SUM over the blob's cells, its lb the largest and its ub the smallest of them. A per-cell
row (``PER_CELL_FAMILIES``) joins the rows of the blob's other cells that read the same once the columns are grouped
(same column types and coefficients, same rhs to ROUND_DECIMALS), as their average; rows that differ stay separate.
So every point feasible for the grouped model, mapped back, is feasible for the full one, and under maxprofit the
grouped optimum is a lower bound of the full one: grouping costs optimality, never feasibility. ``report_grouping``
checks it after every solve — every row of the full model at the mapped x.

It saves the solve only: the full tables are still built. Design, tests and limits: luto/solvers/README.md."""

import time
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from scipy import sparse

from luto import settings
from luto.solvers.row_builder import PER_CELL_FAMILIES, contract
from luto.solvers.tables import Tables


LOCATION_FIELDS = ('cell', 'local_r')                                                    # where a column is; every other integer field of the column table says what it is (its type)
SKETCH_WIDTH = 40                                                                        # the random sketch the SVD directions are taken from


class Grouping(NamedTuple):
    """The grouped tables of a step and the way back: ``x = col_map @ x_grouped``."""
    tables: Tables
    col_map: sparse.csr_matrix                  # cols × grouped cols, 0/1
    row_map: sparse.csr_matrix                  # grouped rows × rows, 1 / the number of rows joined
    blob_of_cell: np.ndarray                    # the blob of every cell
    seconds: float                              # spent grouping


# ═══════════════════════════ group_tables: the grouping of one step ═══════════════════════════

def group_tables(tables: Tables, area_r: np.ndarray, region2cell: xr.Dataset) -> Grouping:
    """The step's tables grouped: the blobs (``get_blobs``), the transform (``aggregate``), then the grouped wide rows'
    range narrowed when asked (``condition_grouped``)."""
    print('Grouping the cells...', flush=True)
    start = time.time()
    type_id = get_column_types(tables.cols)                                                # once: the blobs and the transform both read it
    blob_of_cell = get_blobs(tables, np.asarray(area_r, dtype=np.float64), region2cell, type_id)
    grouped, col_map, row_map = aggregate(tables, blob_of_cell, type_id)
    grouped = condition_grouped(grouped)
    seconds = time.time() - start
    sizes = pd.DataFrame({'': ['full', 'grouped'],
                          'rows': [tables.row_matrix.shape[0], grouped.row_matrix.shape[0]],
                          'columns': [tables.row_matrix.shape[1], grouped.row_matrix.shape[1]],
                          'nonzeros': [tables.row_matrix.nnz, grouped.row_matrix.nnz]})
    for line in sizes.to_markdown(index=False, tablefmt='psql', intfmt=',').split('\n'):
        print(f"│   {line}", flush=True)
    print(f"└── {tables.row_matrix.nnz / grouped.row_matrix.nnz:.1f}× fewer nonzeros, in {seconds:,.0f} s", flush=True)
    return Grouping(grouped, col_map, row_map, blob_of_cell, seconds)


def check_grouping_settings() -> None:
    """What the first version of the grouping cannot run with (raise), and what it runs with at a known cost (warn)."""
    if not settings.CELL_GROUPING:
        return
    if settings.BIO_ROW_GENERATION:
        raise ValueError('CELL_GROUPING cannot run with BIO_ROW_GENERATION: row generation has not been run on grouped tables.')
    if settings.ELASTIC_FAMILIES:
        raise ValueError('CELL_GROUPING cannot run with ELASTIC_FAMILIES: relaxed rows have not been run on grouped tables.')
    if settings.GBF8_TARGET != 'off':
        print('⚠ WARNING: CELL_GROUPING with GBF8 targets: every species row splits the blobs by its footprint, so little is '
              'left of the grouping (RES1, 73 species rows: 2.5x fewer nonzeros, no faster than the full model, 2.8 % loss).', flush=True)
    if any(crossover == 0 for _, _, crossover, _, _ in settings.RETRY_PARAMS):
        print('⚠ WARNING: CELL_GROUPING without crossover: a blob is whole only at a vertex, so an interior solution '
              'leaves most cells with fractional shares.', flush=True)


# ═══════════════════════════ the blobs: which cells share a decision ═══════════════════════════

def get_blobs(tables: Tables, area_r: np.ndarray, region2cell: xr.Dataset, type_id: np.ndarray) -> np.ndarray:
    """The blob of every cell: the same key (exact) and the same SVD bin (close). `type_id`: get_column_types(cols)."""
    row_matrix, rows, cols, obj = tables
    cell = cols['cell'].values.astype(np.int64)
    n_cells = area_r.size

    key = column_key(cols, type_id, cell, n_cells)                                       # the column types a cell holds, each with its base
    key = with_regions(key, region2cell)                                                 # the cell's region in every layer
    key = with_gbf_footprints(key, row_matrix, rows, cell, n_cells)                      # the GBF rows the cell has a coefficient in
    n_key = np.unique(key).size
    soft, n_directions, n_dims = svd_bins(row_matrix, rows, obj, cell, area_r, type_id)  # close: the SVD bin of the cell's coefficients
    key = key + soft

    blob_of_cell = np.unique(key, return_inverse=True)[1].ravel().astype(np.int32)
    print(f"│   ├── {n_cells:,} cells: {n_key:,} distinct keys, {int(blob_of_cell.max()) + 1:,} blobs with the SVD "
          f"({n_directions} directions of {n_dims:,} dimensions, tolerance {settings.CELL_GROUPING_TOLERANCE:g})", flush=True)
    return blob_of_cell


def column_key(cols: xr.Dataset, type_id: np.ndarray, cell: np.ndarray, n_cells: int) -> np.ndarray:
    """Every cell's key from its columns: the column types it holds, each with its base to ROUND_DECIMALS, and Riparian
    Plantings' upper bound in its bin (``riparian_ub_bins``) — summed over the cell's columns, so the order of the columns
    does not matter."""
    col_hash = mix(mix(type_id.astype(np.uint64)) ^ int_hash(np.round(cols['base'].values.astype(np.float64) * 10 ** settings.ROUND_DECIMALS)))
    riparian, bin_of_cap = riparian_ub_bins(cols)
    if bin_of_cap is not None:
        col_hash[riparian] = mix(col_hash[riparian] ^ int_hash(bin_of_cap))
    key = np.zeros(n_cells, dtype=np.uint64)
    np.add.at(key, cell, col_hash)                                                       # order-free over the cell's columns
    return key


def riparian_ub_bins(cols: xr.Dataset) -> tuple[np.ndarray, np.ndarray | None]:
    """(Riparian Plantings' columns, the natural-break bin of each one's upper bound), or (columns, None) with fewer than
    two distinct bounds. Every other upper bound follows from the cell's base (col_builder: the share its sources can
    move), and the base is in the key; Riparian Plantings' is each cell's own stream share. That is a fact of
    col_builder's bounds known here by name: a new upper bound that varies by cell would have to be added here too."""
    riparian = (cols['block'].values == 'nonag') & (cols['k'].values == list(settings.NON_AG_LAND_USES).index('Riparian Plantings'))
    caps = cols['ub'].values[riparian].astype(np.float64)
    n_bins = min(settings.CELL_GROUPING_RIPARIAN_BINS, np.unique(caps).size)
    if n_bins <= 1:
        return riparian, None
    centres = np.quantile(caps, (np.arange(n_bins) + 0.5) / n_bins)
    for _ in range(100):                                                                 # natural breaks: 1-D k-means
        bin_of_cap = np.searchsorted((centres[1:] + centres[:-1]) / 2, caps)
        updated = np.bincount(bin_of_cap, caps, n_bins) / np.maximum(np.bincount(bin_of_cap, minlength=n_bins), 1)
        if np.allclose(updated, centres):
            break
        centres = updated
    return riparian, np.searchsorted((centres[1:] + centres[:-1]) / 2, caps)


def with_regions(key: np.ndarray, region2cell: xr.Dataset) -> np.ndarray:
    """The key with the cell's region in every layer of the column support: a blob never mixes regions, so every
    regional row (water, renewable, regional caps) reads its cells exactly. For quality only: mixing regions stays exact."""
    for layer in region2cell.data_vars:
        key = mix(key ^ int_hash(pd.factorize(region2cell[layer].values)[0]))
    return key


def with_gbf_footprints(key: np.ndarray, row_matrix: sparse.csr_matrix, rows: xr.Dataset, cell: np.ndarray, n_cells: int) -> np.ndarray:
    """The key with the set of active GBF rows the cell has a coefficient in: a target over a mask, so a blob must not
    mix cells inside and outside it (without GBF2's footprint: 9.3 % objective loss at RES5)."""
    family = rows['family'].values.astype(str)
    gbf_rows = np.flatnonzero(np.char.startswith(family, 'GBF') & rows['active'].values)
    if not gbf_rows.size:
        return key
    footprint = row_matrix[gbf_rows].tocoo()
    inside = np.unique(footprint.row.astype(np.int64) * n_cells + cell[footprint.col])    # every (GBF row, cell) with a coefficient, once
    footprint_set = np.zeros(n_cells, dtype=np.uint64)
    np.add.at(footprint_set, inside % n_cells, int_hash(inside // n_cells + 1))            # the set of GBF rows a cell is in, order-free
    return mix(key ^ mix(footprint_set))


def svd_bins(row_matrix: sparse.csr_matrix, rows: xr.Dataset, obj: np.ndarray, cell: np.ndarray, area_r: np.ndarray,
             type_id: np.ndarray) -> tuple[np.ndarray, int, int]:
    """(every cell's SVD bin as a hash, the directions, the dimensions): the cell's coefficients per hectare, one
    dimension per (column type, wide row) and (column type, objective), reduced to settings.CELL_GROUPING_SVD_DIMS
    directions (a randomised SVD of the cells' signature) and binned at settings.CELL_GROUPING_TOLERANCE. Zeros when
    there is no dimension."""
    n_cells = area_r.size
    family = rows['family'].values.astype(str)
    wide_rows = np.flatnonzero(~np.isin(family, PER_CELL_FAMILIES) & rows['active'].values)
    wide = row_matrix[wide_rows].tocoo()
    priced = np.flatnonzero(obj != 0)
    entry_col = np.concatenate([wide.col, priced])
    entry_row = np.concatenate([wide.row.astype(np.int64), np.full(priced.size, wide.shape[0], dtype=np.int64)])
    entry_val = np.concatenate([wide.data.astype(np.float64), obj[priced].astype(np.float64)]) / area_r[cell[entry_col]]
    # a dimension per (column type, wide row or the objective) that has an entry, numbered in sorted order: the used ids
    # of the small (type x row) space ranked by a cumulative count, which is np.unique's numbering without its sort
    raw_dim = type_id[entry_col] * (wide.shape[0] + 1) + entry_row
    used = np.zeros(int(raw_dim.max()) + 1 if raw_dim.size else 0, dtype=bool)
    used[raw_dim] = True
    entry_dim = (np.cumsum(used) - 1)[raw_dim]
    n_dims = int(entry_dim.max()) + 1 if entry_dim.size else 0
    n_directions = min(settings.CELL_GROUPING_SVD_DIMS, n_dims)
    soft = np.zeros(n_cells, dtype=np.uint64)
    if not n_directions:
        return soft, n_directions, n_dims
    dim_scale = np.bincount(entry_dim, np.abs(entry_val)) / np.bincount(entry_dim)       # every dimension in units of its own mean size
    dim_scale[dim_scale == 0] = 1.0
    signature = sparse.csr_matrix(((entry_val / dim_scale[entry_dim]).astype(np.float32), (cell[entry_col], entry_dim)), shape=(n_cells, n_dims))
    sketch = signature @ np.random.default_rng(0).standard_normal((n_dims, min(SKETCH_WIDTH, n_dims)), dtype=np.float32)
    basis, _ = np.linalg.qr(signature @ (signature.T @ sketch))
    small = signature.T @ basis
    left, values, _ = np.linalg.svd(small.T @ small, hermitian=True)
    scores = (basis @ left) * np.sqrt(values)                                            # the cells on the leading directions, largest first
    bin_width = settings.CELL_GROUPING_TOLERANCE * np.sqrt(entry_val.size / n_cells / n_directions)   # in the signature's own units: the same meaning in every run
    for direction in range(n_directions):
        soft = mix(soft ^ int_hash(np.floor(scores[:, direction] / bin_width)))
    return soft, n_directions, n_dims


# ═══════════════════════════ the transform: the tables over blobs ═══════════════════════════

def aggregate(tables: Tables, blob_of_cell: np.ndarray, type_id: np.ndarray) -> tuple[Tables, sparse.csr_matrix, sparse.csr_matrix]:
    """The tables with every blob's cells sharing one share per column type: (grouped tables, col_map, row_map).
    `type_id`: get_column_types(cols)."""
    row_matrix, rows, cols, obj = tables
    row_matrix = row_matrix.tocsr()
    cell = cols['cell'].values.astype(np.int64)
    blob_of_cell = blob_of_cell.astype(np.int64)
    cols_grouped, col_map = group_columns(cols, cell, blob_of_cell, type_id)
    rows_grouped, row_map = group_rows(row_matrix, rows, cell, blob_of_cell, type_id)
    row_matrix_grouped = (row_map @ row_matrix.astype(np.float64) @ col_map).tocsr()
    row_matrix_grouped.eliminate_zeros()
    row_matrix_grouped.sort_indices()
    obj_grouped = col_map.T @ obj.astype(np.float64)
    return Tables(row_matrix_grouped, rows_grouped, cols_grouped, obj_grouped), col_map, row_map


def group_columns(cols: xr.Dataset, cell: np.ndarray, blob_of_cell: np.ndarray, type_id: np.ndarray) -> tuple[xr.Dataset, sparse.csr_matrix]:
    """(the grouped column table, col_map): one column per (blob, column type), lb the largest and ub the smallest of
    its cells'."""
    n_cols = cell.size
    n_types = int(type_id.max()) + 1
    first_col, new_col = number_by_first(blob_of_cell[cell] * n_types + type_id)
    n_new_cols = first_col.size
    col_map = sparse.csr_matrix((np.ones(n_cols), (np.arange(n_cols), new_col)), shape=(n_cols, n_new_cols))
    lb_grouped = np.full(n_new_cols, -np.inf)
    ub_grouped = np.full(n_new_cols, np.inf)
    np.maximum.at(lb_grouped, new_col, cols['lb'].values.astype(np.float64))
    np.minimum.at(ub_grouped, new_col, cols['ub'].values.astype(np.float64))
    # The cells of a blob hold the same base to ROUND_DECIMALS (the key), so a bound that follows the base (a locked-in
    # share: lb = ub = base) can differ between them in the next decimal, and the largest lb then passes the smallest ub by
    # that much. Within the margin the two meet at their midpoint — every cell's own bound holds to the margin, which
    # report_grouping measures; beyond it the blob mixes cells that cannot share a share, and the year stops.
    inverted = np.flatnonzero(lb_grouped > ub_grouped)
    if inverted.size:
        gap = lb_grouped[inverted] - ub_grouped[inverted]
        margin = 10 * settings.FEASIBILITY_TOLERANCE
        if gap.max() > margin:
            raise ValueError(f'Cell grouping: {int((gap > margin).sum()):,} grouped column(s) have a lower bound above their upper bound by more than '
                             f'{margin:g} (largest {gap.max():.3g}) — cells whose bounds exclude each other share a blob.')
        lb_grouped[inverted] = ub_grouped[inverted] = (lb_grouped[inverted] + ub_grouped[inverted]) / 2
        print(f"│   ├── {inverted.size:,} grouped column(s) whose cells' bounds cross by at most {gap.max():.3g}: set to the midpoint", flush=True)
    return cols.isel(col=first_col).assign(lb=(('col',), lb_grouped), ub=(('col',), ub_grouped)), col_map


def group_rows(row_matrix: sparse.csr_matrix, rows: xr.Dataset, cell: np.ndarray, blob_of_cell: np.ndarray,
               type_id: np.ndarray) -> tuple[xr.Dataset, sparse.csr_matrix]:
    """(the grouped row table, row_map): a wide row stays; a per-cell row (PER_CELL_FAMILIES) joins the rows of its blob
    that read the same over the grouped columns — the same column types with the same coefficients, family, sense, rhs
    to ROUND_DECIMALS and active flag — as their average."""
    n_rows = row_matrix.shape[0]
    family = rows['family'].values.astype(str)
    per_cell = np.isin(family, PER_CELL_FAMILIES)
    n_entries = np.diff(row_matrix.indptr)
    entry_row = np.repeat(np.arange(n_rows, dtype=np.int64), n_entries)
    row_cell = np.zeros(n_rows, dtype=np.int64)
    row_cell[n_entries > 0] = cell[row_matrix.indices[row_matrix.indptr[:-1][n_entries > 0]]]      # a per-cell row's cell: its first column's
    crossing = per_cell[entry_row] & (cell[row_matrix.indices] != row_cell[entry_row])
    if crossing.any():
        raise ValueError(f"Cell grouping: {int(crossing.sum()):,} entries of the per-cell families reach outside their row's cell "
                         f"({', '.join(pd.unique(family[entry_row[crossing]]))}).")
    del crossing

    reads = np.zeros(n_rows, dtype=np.uint64)                                            # what the row reads: its column types, each with its coefficient
    np.add.at(reads, entry_row, mix(mix(type_id[row_matrix.indices].astype(np.uint64)) ^ int_hash(np.round(row_matrix.data.astype(np.float64) * 10 ** settings.ROUND_DECIMALS))))
    del entry_row
    row_key = mix(np.arange(n_rows, dtype=np.uint64))                                    # a wide row: its own group
    per_cell_key = reads[per_cell]
    for part in (pd.factorize(family)[0], pd.factorize(rows['sense'].values)[0], np.round(rows['rhs'].values * 10 ** settings.ROUND_DECIMALS),
                 rows['active'].values, blob_of_cell[row_cell]):
        per_cell_key = mix(per_cell_key ^ int_hash(part[per_cell]))
    row_key[per_cell] = per_cell_key
    del reads, per_cell_key
    first_row, new_row = number_by_first(row_key)
    del row_key
    row_map = sparse.csr_matrix((1.0 / np.bincount(new_row)[new_row], (new_row, np.arange(n_rows))), shape=(first_row.size, n_rows))

    rows_grouped = rows.isel(row=first_row).assign(rhs=(('row',), row_map @ rows['rhs'].values.astype(np.float64)))
    # A grouped per-cell row is labelled with its blob's first cell, so that all the per-cell rows of a blob carry one
    # `cell`: bound propagation sums a cell's node-balance rows into the cell's total (row_bounds.conservation_rows)
    first_cell = np.full(int(blob_of_cell.max()) + 1, blob_of_cell.size, dtype=np.int64)
    np.minimum.at(first_cell, blob_of_cell, np.arange(blob_of_cell.size))
    grouped_cell = rows_grouped['cell'].values.copy()
    relabel = per_cell[first_row] & (grouped_cell >= 0)
    grouped_cell[relabel] = first_cell[blob_of_cell[grouped_cell[relabel]]]
    return rows_grouped.assign(cell=(('row',), grouped_cell)), row_map


def condition_grouped(tables: Tables) -> Tables:
    """The grouped tables with their wide rows rescaled again (settings.CELL_GROUPING_RESCALE, on by default; off, the
    tables come back as they are). A grouped column sums its blob's cells, so ``aggregate`` leaves the wide rows
    unscaled: [3e-7, 4e5] at RES1. The per-cell rows are never touched. An exact change of each row's unit: nothing is
    dropped (CELL_GROUPING_DROP_REL was retired on 2026-11-23: its drop broke the full model's GHG cap by 2 to 5 Mt,
    docs/FINDINGS.md, 20261123)."""
    if not settings.CELL_GROUPING_RESCALE:
        return tables
    row_matrix, rows, cols, obj = tables
    row_matrix = row_matrix.tocsr()
    wide = ~np.isin(rows['family'].values.astype(str), PER_CELL_FAMILIES)
    magnitude = np.abs(row_matrix.data)
    print(f"│   ├── grouped matrix |a| [{magnitude.min():.1e}, {magnitude.max():.1e}], {row_matrix.nnz:,} entries", flush=True)

    # ── contract the wide rows again: each row and its rhs divided by one factor, so they sit around RESCALE_FACTOR ──
    wide_rows, other_rows = np.flatnonzero(wide), np.flatnonzero(~wide)
    block, rhs_wide, factor = contract(row_matrix[wide_rows].tocsr(), rows['rhs'].values[wide_rows], rescale=True)
    stacked = sparse.vstack([row_matrix[other_rows], block]).tocsr()
    row_matrix = stacked[np.argsort(np.concatenate([other_rows, wide_rows]), kind='stable')].tocsr()   # back to the table's row order
    rhs = rows['rhs'].values.astype(np.float64).copy()
    rhs[wide_rows] = rhs_wide
    scale = rows['scale'].values.astype(np.float64).copy()
    scale[wide_rows] *= factor                                           # the readers restore raw units through it
    rows = rows.assign(rhs=(('row',), rhs), scale=(('row',), scale))
    print(f"│   ├── wide rows rescaled around {settings.RESCALE_FACTOR:g}", flush=True)

    row_matrix.sort_indices()
    magnitude = np.abs(row_matrix.data)
    print(f"│   ├── now |a| [{magnitude.min():.1e}, {magnitude.max():.1e}], {row_matrix.nnz:,} entries", flush=True)
    return Tables(row_matrix, rows, cols, obj)


# ═══════════════════════════ after the solve: the full model at the mapped x ═══════════════════════════

def report_grouping(grouping: Grouping, tables: Tables, x: np.ndarray, target_year: int, out_dir: str) -> None:
    """Every active row and every bound of the FULL model at the mapped x (``x = col_map @ x_grouped``): the excess per
    family in grouping_rows_<year>.csv (in the scaled row's own units), and the step's sizes in grouping_<year>.csv."""
    row_matrix, rows, cols, obj = tables
    activity = row_matrix @ x
    rhs = rows['rhs'].values
    sense = rows['sense'].values.astype(str)
    excess = np.where(sense == '<', activity - rhs, np.where(sense == '>', rhs - activity, np.abs(activity - rhs)))
    excess[~rows['active'].values] = 0.0
    tolerance = 10 * settings.FEASIBILITY_TOLERANCE
    family = rows['family'].values.astype(str)
    check = pd.DataFrame([{'family': name, 'rows': int((family == name).sum()), 'max_excess': float(excess[family == name].max()),
                           'rows_over_tolerance': int((excess[family == name] > tolerance).sum())} for name in pd.unique(family)])
    check.to_csv(f'{out_dir}/grouping_rows_{target_year}.csv', index=False)
    bound_excess = max(float((cols['lb'].values - x).max()), float((x - cols['ub'].values).max()))

    land = np.isin(cols['block'].values, ('ag', 'nonag'))                                 # a cell is whole when one land use holds all of it
    largest_share = np.zeros(grouping.blob_of_cell.size)
    np.maximum.at(largest_share, cols['cell'].values[land], x[land])
    n_fractional = int((largest_share < 1 - 1e-6).sum())
    cells_per_blob = np.bincount(grouping.blob_of_cell)

    grouped = grouping.tables
    pd.DataFrame([{
        'year': target_year, 'cells': grouping.blob_of_cell.size, 'blobs': cells_per_blob.size, 'blobs_of_one_cell': int((cells_per_blob == 1).sum()),
        'rows_full': row_matrix.shape[0], 'rows_grouped': grouped.row_matrix.shape[0],
        'columns_full': row_matrix.shape[1], 'columns_grouped': grouped.row_matrix.shape[1],
        'nonzeros_full': row_matrix.nnz, 'nonzeros_grouped': grouped.row_matrix.nnz,
        'grouping_seconds': round(grouping.seconds, 1), 'objective_full_model_at_mapped_x': float(obj.astype(np.float64) @ x),
        'max_row_excess': float(excess.max()), 'rows_over_tolerance': int((excess > tolerance).sum()), 'max_bound_excess': bound_excess,
        'cells_without_a_whole_land_use': n_fractional}]).to_csv(f'{out_dir}/grouping_{target_year}.csv', index=False)
    print(f"Year {target_year}: grouped solution mapped back to {grouping.blob_of_cell.size:,} cells — rows of the full model over "
          f"{tolerance:g}: {int((excess > tolerance).sum()):,} (worst {excess.max():.3g}, {family[int(excess.argmax())]}); "
          f"worst bound excess {bound_excess:.3g}; cells without a whole land use: {n_fractional:,}", flush=True)


# ═══════════════════════════ hashing and numbering ═══════════════════════════

def mix(values: np.ndarray) -> np.ndarray:
    """A 64-bit hash of every value (splitmix64), so that keys of many parts can be combined without a collision in practice."""
    hashed = values + np.uint64(0x9E3779B97F4A7C15)
    hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return hashed ^ (hashed >> np.uint64(31))


def int_hash(values: np.ndarray) -> np.ndarray:
    return mix(np.asarray(values).astype(np.int64).astype(np.uint64))


def number_by_first(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the first position of every distinct key, the group of every position), the groups numbered in the order they first
    appear — so grouped columns and rows keep the table's own order (the blocks and the families stay together).
    pd.factorize numbers them so by hashing (np.unique sorted 100 M keys at RES1); a position is a group's first exactly
    where its number exceeds every number before it."""
    group = pd.factorize(key)[0]
    first = np.ones(group.size, dtype=bool)
    first[1:] = group[1:] > np.maximum.accumulate(group)[:-1]
    return np.flatnonzero(first), group.astype(np.int64)


def get_column_types(cols: xr.Dataset) -> np.ndarray:
    """The type of every column — what it is: its block and every integer field of the column table but where it is
    (LOCATION_FIELDS) — as one integer, numbered in sorted order. Read off the table, so a field added to the column
    space is part of the type without a change here. pd.factorize with sort=True numbers them as np.unique did, by
    hashing instead of sorting every column."""
    fields = [pd.factorize(cols['block'].values)[0].astype(np.int64)]
    for name in cols.data_vars:
        if cols[name].dtype.kind == 'i' and name not in LOCATION_FIELDS:
            fields.append(cols[name].values.astype(np.int64) + 1)
    return pd.factorize(np.ravel_multi_index(fields, [int(values.max()) + 1 for values in fields]), sort=True)[0].astype(np.int64)
