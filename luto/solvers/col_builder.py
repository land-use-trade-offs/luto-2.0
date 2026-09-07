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
The COLUMN side of the model: every unknown of one solve step as a labelled ``xr.Dataset`` per
block, built from the base-year state alone (``get_cols``). The row side is ``row_builder``.
"""

import numpy as np
import pandas as pd
import xarray as xr

import luto.settings as settings
import luto.tools as tools
from luto.data import Data
import luto.economics.agricultural.transitions as ag_transition
import luto.economics.non_agricultural.transitions as non_ag_transition

BLOCK_ORDER = ('ag', 'nonag', 'am', 'ag2ag', 'ag2nonag', 'nonag2ag', 'acct', 'cell_usage')   # Var.index order of the blocks


# ═══════════════════════════ data: what decides existence, bounds and base (from the base-year state) ═══════════════════════════

# ── exclude matrix, transition bounds, ag-management lower bounds ──

def get_ag_x_mrj(data: Data, base_year):
    print('Getting agricultural exclude matrices...', flush = True)
    return ag_transition.get_to_ag_exclude_matrices(data, base_year)

def get_trans_ub_ag_mrj(data: Data, base_year: int) -> np.ndarray:
    """Ag target upper bound (ag2ag + nonag2ag), raised to the FOLDED base: a cell can always keep
    its base land use, and the solver's constant is the folded dvar (dominants carry their absorbed
    sub-θ mass). A real gap here (not float noise) means a base land use is banned by EXCLUDE/no-go;
    the raise keeps the lb <= base <= ub box coherent but does not create a variable for it."""
    print('Getting agricultural target upper bounds...', flush = True)
    ub = (
        ag_transition.get_ag2ag_ub(data, base_year)
        + non_ag_transition.get_nonag2ag_ub(data, base_year)
    ).astype(np.float32)
    base = ag_transition.get_folded_base_ag_dvar(data, base_year)
    return tools.clamp_dvar_bound(ub, np.maximum(base, 0.0), np.inf, 'Ag ub raised to base')

def get_trans_lb_ag_mrj(data: Data, base_year: int) -> np.ndarray:
    """Ag target lower bound (all zeros: the sliver pin is superseded by θ-folding), kept in [0, base]."""
    print('Getting agricultural target lower bounds...', flush = True)
    lb = ag_transition.get_ag2ag_lb(data, base_year)
    base = ag_transition.get_folded_base_ag_dvar(data, base_year)
    return tools.clamp_dvar_bound(lb, 0.0, np.maximum(base, 0.0), 'Ag lb clamped to [0,base]')

def get_trans_ub_nonag_rk(data: Data, base_year):
    print('Getting non-agricultural target upper bounds...', flush = True)
    base_dvar_nonag = (
        data.non_ag_dvars[base_year] if base_year != data.YR_CAL_BASE
        else np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32)
    )
    ub = non_ag_transition.get_non_ag_ub_matrices(
        data,
        base_dvar_nonag_rk=base_dvar_nonag,
        base_dvar_ag_mrj=ag_transition.get_folded_base_ag_dvar(data, base_year),   # solver-world identity
    )
    return tools.clamp_dvar_bound(ub, np.maximum(base_dvar_nonag, 0.0), np.inf, 'NonAg ub raised to base')

def get_trans_lb_nonag_rk(data: Data, base_year):
    print('Getting non-agricultural lower bound matrices...', flush = True)
    lb = non_ag_transition.get_non_ag_lb_matrices(data, base_year)
    base = (
        data.non_ag_dvars[base_year].astype(np.float32) if base_year != data.YR_CAL_BASE
        else np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32)
    )
    return tools.clamp_dvar_bound(lb, 0.0, np.maximum(base, 0.0), 'NonAg lb clamped to [0,base]')

def get_ag_man_lb_mrj(data: Data, base_year):
    print('Getting agricultural lower bound matrices...', flush = True)
    return ag_transition.get_lower_bound_agricultural_management_matrices(data, base_year)


# ── sources: the base-year holders of land (FROM-view) ──

def get_trans_source_ag(data: Data, base_year: int) -> dict:
    print('Getting agricultural source cells...', flush = True)
    return ag_transition.get_base_dvar_mj_cell_map(data, base_year)

def get_trans_source_nonag(data: Data, base_year: int) -> dict:
    print('Getting non-agricultural source cells...', flush = True)
    return non_ag_transition.get_base_nonag_dvar_k_cell_map(data, base_year)


# ── feasibility: which target entries, which cell-usage rows and which transition arcs exist ──

def get_trans_feasible_ag(ag_x_mrj: np.ndarray, trans_lb_ag_mrj: np.ndarray) -> dict:
    """{(m, j): cells} that get a target ag var: eligible by the exclude matrix, or with a stay floor."""
    print('Getting feasible agricultural cells...', flush = True)
    n_lms, _ncells, n_lus = ag_x_mrj.shape
    eligible = (ag_x_mrj > 0) | (trans_lb_ag_mrj > 0)
    return {
        (m, j): np.where(eligible[m, :, j])[0]
        for j in range(n_lus)
        for m in range(n_lms)
    }

def get_trans_feasible_nonag(trans_ub_nonag_rk: np.ndarray, threshold: float = 0.0) -> dict:
    """{k: cells} that get a target non-ag var (ub > threshold)."""
    print('Getting feasible non-agricultural cells...', flush = True)
    n_k = trans_ub_nonag_rk.shape[1]
    return {k: np.where(trans_ub_nonag_rk[:, k] > threshold)[0] for k in range(n_k)}

def get_cell_usage_feasible_r(trans_ub_ag_mrj: np.ndarray, trans_ub_nonag_rk: np.ndarray, ag_mask_r: np.ndarray) -> np.ndarray:
    """Cells (bool over r) that can meet the cell-usage equality Σ(ag + non-ag shares) = ag_mask.
    A cell with any ag var can always cover ag_mask (its sources can "stay"); a cell without one is
    limited by the sum of its non-ag upper bounds (no variables at all, or a capped non-ag option)."""
    print('Getting cells that can meet the cell-usage equality...', flush=True)
    has_any_ag_r = (trans_ub_ag_mrj > 0).any(axis=(0, 2))
    max_nonag_r  = trans_ub_nonag_rk.sum(axis=1)
    max_alloc_r  = np.where(has_any_ag_r, 1.0, max_nonag_r)
    return max_alloc_r >= ag_mask_r - 1e-6

def get_feasible_ag2ag_mrj(ag_x_mrj: np.ndarray, trans_source_ag: dict, T_ag2ag_reach_jj: np.ndarray) -> dict:
    """Ag2ag delta-var feasibility, SOURCE-KEYED like flow_cost_ag2ag:
    {(from_m, from_j): bool (NLMS, ncells_src, N_AG_LUS) [to_m, local_r, to_j]}, True where the
    target is eligible (its X var exists), T_MAT allows from_j -> to_j, and it is not the diagonal."""
    print('Getting feasible ag2ag delta-var targets...', flush = True)
    eligible = ag_x_mrj > 0
    result = {}
    for (from_m, from_j), cells in trans_source_ag.items():
        valid = eligible[:, cells, :] & T_ag2ag_reach_jj[from_j][None, None, :]     # (NLMS, ncells_src, N_AG)
        valid[from_m, :, from_j] = False                                            # staying is not a transition
        result[(from_m, from_j)] = valid
    return result

def get_feasible_nonag2ag_mrj(ag_x_mrj: np.ndarray, trans_source_nonag: dict, T_nonag2ag_reach_kj: np.ndarray) -> dict:
    """Nonag2ag delta-var feasibility, SOURCE-KEYED: {from_k: bool (NLMS, ncells_k, N_AG_LUS)
    [to_m, local_r, to_j]} — as get_feasible_ag2ag_mrj from the non-ag sources (e.g. reversible
    Destocked land back to ag); no diagonal to drop."""
    print('Getting feasible nonag2ag delta-var targets...', flush = True)
    eligible = ag_x_mrj > 0
    return {
        from_k: eligible[:, cells, :] & T_nonag2ag_reach_kj[from_k][None, None, :]   # (NLMS, ncells_k, N_AG)
        for from_k, cells in trans_source_nonag.items()
    }

def get_feasible_ag2nonag_rk(trans_ub_nonag_rk: np.ndarray, trans_source_ag: dict, T_ag2nonag_reach_jk: np.ndarray) -> dict:
    """Ag2nonag delta-var feasibility, SOURCE-KEYED: {(from_m, from_j): bool (ncells_src, N_NON_AG_LUS)
    [local_r, k]}. The target side gates on ``trans_ub_nonag_rk > 0`` — not raw T_MAT reach — because
    the non-ag ub carries extra zeroing caps (RP buffer, Destocked eligibility, no-go); a raw-reach
    gate would point deltas at targets with no X var."""
    print('Getting feasible ag2nonag delta-var targets...', flush = True)
    eligible = trans_ub_nonag_rk > 0
    return {
        (from_m, from_j): eligible[cells, :] & T_ag2nonag_reach_jk[from_j][None, :]   # (ncells_src, N_NONAG)
        for (from_m, from_j), cells in trans_source_ag.items()
    }

def get_table_flow(trans_source_ag: dict, trans_source_nonag: dict, feasible_ag2ag_mrj: dict,
                   feasible_ag2nonag_rk: dict, feasible_nonag2ag_mrj: dict) -> dict:
    """The transition-flow arcs as three edge tables (ag2ag, ag2nonag, nonag2ag): one row per delta
    variable, sources in ``trans_source_*`` order, arcs in ``np.argwhere`` C-order within a source.
    Per table: ``fields`` (int32 columns), ``n``, ``src_ptr`` (source s = rows src_ptr[s]:src_ptr[s+1])
    and ``sources`` (the source keys). ``local_r`` indexes the source's cell list (the axis of the
    source-keyed flow_cost / flow_ghg dicts), ``cell`` is the global cell; land never crosses cells."""
    print('Building the transition-flow edge tables...', flush = True)

    # ── ag → ag: source (from_m, from_j); arc fields (to_m, local_r, to_j) ──
    arc_rows, src_ptr = [], [0]
    for src_idx, ((from_m, from_j), valid) in enumerate(feasible_ag2ag_mrj.items()):
        to_m, local_r, to_j = np.argwhere(valid).T
        cell = np.asarray(trans_source_ag[(from_m, from_j)])[local_r]
        arc_rows.append(np.column_stack([np.full(cell.size, src_idx), np.full(cell.size, from_m), np.full(cell.size, from_j),
                                         local_r, cell, to_m, to_j]))
        src_ptr.append(src_ptr[-1] + cell.size)
    table = np.concatenate(arc_rows) if arc_rows else np.empty((0, 7), dtype=np.int32)
    fields = np.ascontiguousarray(table.T, dtype=np.int32)             # transposed: make each field one contiguous row
    ag2ag = dict(fields=dict(zip(('src', 'from_m', 'from_j', 'local_r', 'cell', 'to_m', 'to_j'), fields)),
                 n=fields.shape[1], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_ag2ag_mrj))

    # ── ag → non-ag: source (from_m, from_j); arc fields (local_r, to_k) ──
    arc_rows, src_ptr = [], [0]
    for src_idx, ((from_m, from_j), valid) in enumerate(feasible_ag2nonag_rk.items()):
        local_r, to_k = np.argwhere(valid).T
        cell = np.asarray(trans_source_ag[(from_m, from_j)])[local_r]
        arc_rows.append(np.column_stack([np.full(cell.size, src_idx), np.full(cell.size, from_m), np.full(cell.size, from_j),
                                         local_r, cell, to_k]))
        src_ptr.append(src_ptr[-1] + cell.size)
    table = np.concatenate(arc_rows) if arc_rows else np.empty((0, 6), dtype=np.int32)
    fields = np.ascontiguousarray(table.T, dtype=np.int32)
    ag2nonag = dict(fields=dict(zip(('src', 'from_m', 'from_j', 'local_r', 'cell', 'to_k'), fields)),
                    n=fields.shape[1], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_ag2nonag_rk))

    # ── non-ag → ag: source from_k; arc fields (to_m, local_r, to_j) ──
    arc_rows, src_ptr = [], [0]
    for src_idx, (from_k, valid) in enumerate(feasible_nonag2ag_mrj.items()):
        to_m, local_r, to_j = np.argwhere(valid).T
        cell = np.asarray(trans_source_nonag[from_k])[local_r]
        arc_rows.append(np.column_stack([np.full(cell.size, src_idx), np.full(cell.size, from_k), local_r, cell, to_m, to_j]))
        src_ptr.append(src_ptr[-1] + cell.size)
    table = np.concatenate(arc_rows) if arc_rows else np.empty((0, 6), dtype=np.int32)
    fields = np.ascontiguousarray(table.T, dtype=np.int32)
    nonag2ag = dict(fields=dict(zip(('src', 'from_k', 'local_r', 'cell', 'to_m', 'to_j'), fields)),
                    n=fields.shape[1], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_nonag2ag_mrj))

    print(f'    ag2ag {ag2ag["n"]:,} | ag2nonag {ag2nonag["n"]:,} | nonag2ag {nonag2ag["n"]:,} arcs', flush = True)
    return dict(ag2ag=ag2ag, ag2nonag=ag2nonag, nonag2ag=nonag2ag)


# ── masks: cell sets that restrict ag-management options ──

def get_savanna_eligible_r(data: Data) -> np.ndarray:
    return np.where(data.SAVBURN_ELIGIBLE == 1)[0]

def get_renewable_GBF2_mask_solar_idx(data: Data) -> np.ndarray:
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_GBF2_MASK_SOLAR)[0]

def get_renewable_GBF2_mask_wind_idx(data: Data) -> np.ndarray:
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_GBF2_MASK_WIND)[0]

def get_renewable_MNES_mask_solar_idx(data: Data) -> np.ndarray:
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_MNES_MASK_SOLAR)[0]

def get_renewable_MNES_mask_wind_idx(data: Data) -> np.ndarray:
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_MNES_MASK_WIND)[0]


# ═══════════════════════════ the column blocks: one Dataset per block, block-LOCAL ids (get_cols shifts them) ═══════════════════════════
#
# Every block holds ``exists`` (the select) and ``col`` (the column id, -1 = no variable) on its own
# dims, plus bounds / base where the variable has them. Ids are numbered in C-order over the dims
# that define the block's column order, so ``np.nonzero(exists)`` in that order IS the column list.

def ag_space(data: Data, trans_feasible_ag: dict, trans_lb_ag_mrj, trans_ub_ag_mrj, dvar_base_ag_mrj) -> xr.Dataset:
    """X_ag on (lm, lu, cell); column order = lu, lm, cell ascending."""
    exists_jmr = np.zeros((data.N_AG_LUS, data.NLMS, data.NCELLS), dtype=bool)   # (lu, lm, cell) = the column order
    for (m, j), cells in trans_feasible_ag.items():
        exists_jmr[j, m, cells] = True
    col_jmr = np.full(exists_jmr.shape, -1, dtype=np.int32)
    col_jmr[exists_jmr] = np.arange(exists_jmr.sum(), dtype=np.int32)              # ids in C-order over (lu, lm, cell)
    # the cubes are stored (lm, lu, cell); every transposed view is made contiguous so downstream gathers read whole rows
    to_lm_lu_cell = lambda cube_mrj: np.ascontiguousarray(np.asarray(cube_mrj).transpose(0, 2, 1))   # (m, r, j) -> (m, j, r)
    return xr.Dataset(
        dict(exists=(('lm', 'lu', 'cell'), np.ascontiguousarray(exists_jmr.transpose(1, 0, 2))),
             col   =(('lm', 'lu', 'cell'), np.ascontiguousarray(col_jmr.transpose(1, 0, 2))),
             lb    =(('lm', 'lu', 'cell'), to_lm_lu_cell(trans_lb_ag_mrj)),
             ub    =(('lm', 'lu', 'cell'), to_lm_lu_cell(trans_ub_ag_mrj)),
             base  =(('lm', 'lu', 'cell'), to_lm_lu_cell(dvar_base_ag_mrj))),
        coords=dict(lm=list(data.LANDMANS), lu=list(data.AGRICULTURAL_LANDUSES), cell=np.arange(data.NCELLS)),
        attrs=dict(n=int(exists_jmr.sum()))
    )


def nonag_space(data: Data, trans_feasible_nonag: dict, trans_lb_nonag_rk, trans_ub_nonag_rk, dvar_base_non_ag_rk) -> xr.Dataset:
    """X_non_ag on (nonag_lu, cell); column order = nonag_lu (enabled land uses only), cell ascending."""
    exists = np.zeros((data.N_NON_AG_LUS, data.NCELLS), dtype=bool)
    for k, lu_name in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        if settings.NON_AG_LAND_USES[lu_name] and k in trans_feasible_nonag:
            exists[k, trans_feasible_nonag[k]] = True
    col = np.full(exists.shape, -1, dtype=np.int32)
    col[exists] = np.arange(exists.sum(), dtype=np.int32)
    # collapse rule: where lb > 0 and ub is within 1% of lb, both bounds are pinned to the base value
    pinned = (trans_lb_nonag_rk > 0) & (np.abs(trans_ub_nonag_rk - trans_lb_nonag_rk) / np.where(trans_lb_nonag_rk > 0, trans_lb_nonag_rk, 1.0) < 0.01)
    return xr.Dataset(
        dict(exists=(('nonag_lu', 'cell'), exists),
             col   =(('nonag_lu', 'cell'), col),
             # (r, k) -> (k, r): contiguous after the transpose
             lb    =(('nonag_lu', 'cell'), np.ascontiguousarray(np.where(pinned, dvar_base_non_ag_rk, trans_lb_nonag_rk).T)),
             ub    =(('nonag_lu', 'cell'), np.ascontiguousarray(np.where(pinned, dvar_base_non_ag_rk, trans_ub_nonag_rk).T)),
             base  =(('nonag_lu', 'cell'), np.ascontiguousarray(np.asarray(dvar_base_non_ag_rk).T))),
        coords=dict(nonag_lu=list(data.NON_AGRICULTURAL_LANDUSES), cell=np.arange(data.NCELLS)),
        attrs=dict(n=int(exists.sum())))


def am_space(data: Data, trans_feasible_ag: dict, agman2lu: dict, renewable_GBF2_mask_solar_idx, renewable_GBF2_mask_wind_idx,
             savanna_eligible_r, ag_man_lb_mrj: dict) -> xr.Dataset:
    """X_ag_man on (slot, lm, cell), slot = MultiIndex (am, lu) over the enabled (option, land use)
    pairs in ``agman2lu`` order; column order = slot, lm, cell. An entry exists where the ag entry
    exists; renewables drop the GBF2-exclusion cells (both lm), savanna burning keeps only the
    savanna-eligible cells (dry). lb = ag_man_lb_mrj for non-reversible options, else 0; ub = 1."""
    pairs = [(am, lu_code) for am, lu_codes in agman2lu.items() for lu_code in lu_codes]
    ag_exists = np.zeros((data.NLMS, data.N_AG_LUS, data.NCELLS), dtype=bool)
    for (m, j), cells in trans_feasible_ag.items():
        ag_exists[m, j, cells] = True
    lb_sources = [np.asarray(ag_man_lb_mrj[am]) for am in agman2lu if not settings.AG_MANAGEMENTS_REVERSIBLE[am]]
    lb_dtype = np.result_type(*[source.dtype for source in lb_sources]) if lb_sources else np.float32   # the sources' own dtype
    exists = np.zeros((len(pairs), data.NLMS, data.NCELLS), dtype=bool)
    lb = np.zeros((len(pairs), data.NLMS, data.NCELLS), dtype=lb_dtype)
    savanna_mask = np.zeros(data.NCELLS, dtype=bool)
    savanna_mask[savanna_eligible_r] = True
    for slot, (am, j) in enumerate(pairs):
        slot_exists = ag_exists[:, j, :].copy()                                        # (lm, cell)
        if am in settings.RENEWABLES_OPTIONS:
            excluded_cells = renewable_GBF2_mask_solar_idx if am == "Utility Solar PV" else renewable_GBF2_mask_wind_idx
            slot_exists[:, excluded_cells] = False
        elif tools.am_name_snake_case(am) == "savanna_burning":
            slot_exists[0] &= savanna_mask                                             # dry only
        exists[slot] = slot_exists
        if not settings.AG_MANAGEMENTS_REVERSIBLE[am]:
            lb[slot] = np.asarray(ag_man_lb_mrj[am])[:, :, j]                          # (lm, cell) of land use j
    col = np.full(exists.shape, -1, dtype=np.int32)
    col[exists] = np.arange(exists.sum(), dtype=np.int32)
    slot_index = pd.MultiIndex.from_arrays([[am for am, _ in pairs], [data.AGRICULTURAL_LANDUSES[j] for _, j in pairs]], names=['am', 'lu'])
    coords = xr.Coordinates.from_pandas_multiindex(slot_index, 'slot').assign(lm=list(data.LANDMANS), cell=np.arange(data.NCELLS))
    return xr.Dataset(
        dict(exists=(('slot', 'lm', 'cell'), exists),
             col   =(('slot', 'lm', 'cell'), col),
             lb    =(('slot', 'lm', 'cell'), lb),
             j     =(('slot',), np.array([j for _, j in pairs], dtype=np.int32))),
        coords=coords, attrs=dict(n=int(exists.sum()), ub=1.0, am_list=list(agman2lu)))


def acct_space(ag: xr.Dataset, ag_fold_map: dict) -> xr.Dataset:
    """X_acct on (lm, lu, cell) plus the fold tables (dims ``sliver``, ``dominant``).

    ``col`` starts as a copy of ``ag.col`` (aliases). Every kept sliver (its dominant must own an ag
    column) and every receiving dominant gets its own acct-local column, flagged ``is_new``:
    dominants first (first-appearance order), then slivers (fold-map order). The linking rows read

        sliver:    X_acct[sliver] = c_k · X_ag[dom]  (+ X_ag[sliver] when the sliver owns an ag column)
        dominant:  X_acct[dom]    = (1 − Σ_k c_k) · X_ag[dom],   c_k = sliver / folded dominant (float32)

    ``*_ag_col`` are ag-local ids and ``*_acct_col`` acct-local; get_cols shifts both to Var.index."""
    col = ag['col'].values.copy()
    exists = ag['exists'].values.copy()
    is_new = np.zeros(col.shape, dtype=bool)
    lm_names = np.asarray(ag.lm.values, dtype=object)
    lu_names = np.asarray(ag.lu.values, dtype=object)
    n_lus, ncells = col.shape[1], col.shape[2]

    # ── the fold map, kept where the dominant owns an ag column (a banned dominant skips its slivers) ──
    cells       = np.asarray(ag_fold_map['cells'], dtype=np.int64)
    sliver_m    = np.asarray(ag_fold_map['from_m'], dtype=np.int64)
    sliver_j    = np.asarray(ag_fold_map['from_j'], dtype=np.int64)
    dom_m       = np.asarray(ag_fold_map['to_m'], dtype=np.int64)
    dom_j       = np.asarray(ag_fold_map['to_j'], dtype=np.int64)
    sliver_frac = np.asarray(ag_fold_map['vals']).astype(np.float64)
    dom_frac    = np.asarray(ag_fold_map['folded_dom']).astype(np.float64)
    dom_ag_col  = col[dom_m, dom_j, cells]
    keep = dom_ag_col >= 0
    cells, sliver_m, sliver_j, dom_m, dom_j = cells[keep], sliver_m[keep], sliver_j[keep], dom_m[keep], dom_j[keep]
    sliver_frac, dom_frac, dom_ag_col = sliver_frac[keep], dom_frac[keep], dom_ag_col[keep]
    n_slivers = cells.size
    fold_share = (sliver_frac / dom_frac).astype(np.float32)                  # c_k: float64 quotient, stored float32

    # ── the receiving dominants: unique (lm, lu, cell), in first-appearance order ──
    dom_key = (dom_m * n_lus + dom_j) * ncells + cells
    _, first_pos, dom_of_sliver = np.unique(dom_key, return_index=True, return_inverse=True)
    appearance = np.argsort(first_pos, kind='stable')                         # unique() sorts by key: re-rank by first appearance
    rank = np.empty(appearance.size, dtype=np.int64)
    rank[appearance] = np.arange(appearance.size)
    dom_of_sliver = rank[dom_of_sliver]                                       # per sliver: the row of its dominant
    first_pos = first_pos[appearance]
    n_dom = first_pos.size
    dom_rows_m, dom_rows_j, dom_rows_cell = dom_m[first_pos], dom_j[first_pos], cells[first_pos]

    # ── acct-local ids: dominants 0..n_dom-1, then the slivers ──
    dom_acct_col = np.arange(n_dom, dtype=np.int32)
    sliver_acct_col = (n_dom + np.arange(n_slivers)).astype(np.int32)
    sliver_ag_col = col[sliver_m, sliver_j, cells]                            # read BEFORE the overwrite: -1 = no own ag column
    col[dom_rows_m, dom_rows_j, dom_rows_cell] = dom_acct_col
    is_new[dom_rows_m, dom_rows_j, dom_rows_cell] = True
    col[sliver_m, sliver_j, cells] = sliver_acct_col
    is_new[sliver_m, sliver_j, cells] = True
    exists[sliver_m, sliver_j, cells] = True                                  # a sliver entry is an accounting entry even without an ag column
    dom_fold_share_sum = np.zeros(n_dom, dtype=np.float64)                    # Σ_k c_k per dominant
    np.add.at(dom_fold_share_sum, dom_of_sliver, fold_share.astype(np.float64))

    return xr.Dataset(
        dict(exists=(('lm', 'lu', 'cell'), exists), col=(('lm', 'lu', 'cell'), col), is_new=(('lm', 'lu', 'cell'), is_new),
             # the fold table: one row per kept sliver
             sliver_from_lm=(('sliver',), lm_names[sliver_m]), sliver_from_lu=(('sliver',), lu_names[sliver_j]), sliver_cell=(('sliver',), cells),
             sliver_to_lm=(('sliver',), lm_names[dom_m]), sliver_to_lu=(('sliver',), lu_names[dom_j]),
             sliver_fold_share=(('sliver',), fold_share), sliver_dom=(('sliver',), dom_of_sliver),
             sliver_dom_ag_col=(('sliver',), dom_ag_col.astype(np.int32)), sliver_ag_col=(('sliver',), sliver_ag_col.astype(np.int32)),
             sliver_acct_col=(('sliver',), sliver_acct_col),
             # one row per receiving dominant
             dom_lm=(('dominant',), lm_names[dom_rows_m]), dom_lu=(('dominant',), lu_names[dom_rows_j]), dom_cell=(('dominant',), dom_rows_cell),
             dom_ag_col=(('dominant',), dom_ag_col[first_pos].astype(np.int32)),
             dom_acct_col=(('dominant',), dom_acct_col), dom_fold_share_sum=(('dominant',), dom_fold_share_sum)),
        coords=dict(lm=ag.lm.values, lu=ag.lu.values, cell=ag.cell.values, sliver=np.arange(n_slivers), dominant=np.arange(n_dom)),
        attrs=dict(n_new=int(n_dom + n_slivers), n_dom=int(n_dom), n_sliver=int(n_slivers)))


def cell_usage_space(cell_usage_feasible_r) -> xr.Dataset:
    """The cell-usage range slacks on (cell): one per cell that gets a cell-usage row."""
    exists = np.asarray(cell_usage_feasible_r, dtype=bool)
    col = np.full(exists.shape, -1, dtype=np.int32)
    col[exists] = np.arange(exists.sum(), dtype=np.int32)
    return xr.Dataset(dict(exists=(('cell',), exists), col=(('cell',), col)),
                      coords=dict(cell=np.arange(exists.size)), attrs=dict(n=int(exists.sum())))


def columns(ds: xr.Dataset, dims: tuple, want: tuple | None = None) -> tuple:
    """The block's columns in column order: the index arrays of the dims in ``want`` (default: the
    last dim, the cell) followed by the global column ids."""
    idx = dict(zip(dims, np.nonzero(ds['exists'].transpose(*dims).values)))   # read once, a transposed view is enough
    col = ds['col'].values[tuple(idx[dim] for dim in ds['col'].dims)]
    return (*[idx[dim].astype(np.int32) for dim in (want or (dims[-1],))], col)


# ═══════════════════════════ get_cols: the column space of one step ═══════════════════════════

def get_cols(data: Data, base_year: int) -> dict:
    """The column space of one solve step from the base-year state alone: every unknown of the
    model as a labelled Dataset per block (``BLOCK_ORDER``) with global Var.index ids, plus what
    the rows and the post-solve read need about the columns —

        space['layout']    block offsets in Var.index order, n_dec (decision columns) and n_all (+ slacks)
        space['terms']     the coefficient support: one term per accounting entry / ag-mgt column / non-ag column
        space['feasible']  {'ag': {(m, j): cells}, 'nonag': {k: cells}} — the target-side feasibility
        space['sources']   {'ag': {(from_m, from_j): cells}, 'nonag': {k: cells}} — the base-year holders of land
        space['masks']     the renewable exclusion cell indices (GBF2 / MNES, solar / wind)
        space['am'].attrs  'agman2lu', 'savanna_eligible_r'
    """

    # ── 1. transition bounds and the base (TO-view) ──
    ag_x_mrj          = get_ag_x_mrj(data, base_year)                     # exclude matrix: which (m, j) a cell may become
    trans_ub_ag_mrj   = get_trans_ub_ag_mrj(data, base_year)              # ag target upper bound (ag2ag + nonag2ag)
    trans_lb_ag_mrj   = get_trans_lb_ag_mrj(data, base_year)              # ag target lower bound (zeros for now)
    trans_ub_nonag_rk = get_trans_ub_nonag_rk(data, base_year)
    trans_lb_nonag_rk = get_trans_lb_nonag_rk(data, base_year)
    ag_man_lb_mrj     = get_ag_man_lb_mrj(data, base_year)                # non-reversible options lock in last step's adoption
    # the base dvars are the node-balance "stay" constant: clipped into the cleaned [lb, ub] box so the
    # all-delta-zero stay point is feasible by construction (only bites on float noise, e.g. -1e-8 < lb = 0; reported)
    dvar_base_ag_mrj    = tools.clamp_dvar_bound(ag_transition.get_folded_base_ag_dvar(data, base_year), trans_lb_ag_mrj, trans_ub_ag_mrj, 'Ag base clipped to [lb,ub]')
    dvar_base_non_ag_rk = tools.clamp_dvar_bound(data.non_ag_dvars[base_year], trans_lb_nonag_rk, trans_ub_nonag_rk, 'NonAg base clipped to [lb,ub]')
    ag_fold_map         = ag_transition.get_ag_dvar_fold_map(data, base_year)   # which sub-θ slivers fold into which dominant

    # ── 2. sources (FROM-view): the base-year holders of land ──
    trans_source_ag    = get_trans_source_ag(data, base_year)             # cells holding each ag (from_m, from_j) source
    trans_source_nonag = get_trans_source_nonag(data, base_year)          # cells holding each non-ag source k

    # ── 3. feasibility: target entries, cell-usage rows and transition arcs ──
    trans_feasible_ag     = get_trans_feasible_ag(ag_x_mrj, trans_lb_ag_mrj)         # cells that get a target ag var
    trans_feasible_nonag  = get_trans_feasible_nonag(trans_ub_nonag_rk)              # cells that get a target non-ag var (ub > 0)
    cell_usage_feasible_r = get_cell_usage_feasible_r(trans_ub_ag_mrj, trans_ub_nonag_rk, data.AG_MASK_PROPORTION_R)
    # per-source reachability (T_MAT finite ⇒ allowed) decides which delta vars exist
    T_ag2ag_reach_jj    = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.AGRICULTURAL_LANDUSES).values)
    T_ag2nonag_reach_jk = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.NON_AGRICULTURAL_LANDUSES).values)
    T_nonag2ag_reach_kj = ~np.isnan(data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values)
    feasible_ag2ag_mrj    = get_feasible_ag2ag_mrj(ag_x_mrj, trans_source_ag, T_ag2ag_reach_jj)
    feasible_nonag2ag_mrj = get_feasible_nonag2ag_mrj(ag_x_mrj, trans_source_nonag, T_nonag2ag_reach_kj)
    feasible_ag2nonag_rk  = get_feasible_ag2nonag_rk(trans_ub_nonag_rk, trans_source_ag, T_ag2nonag_reach_jk)
    table_flow = get_table_flow(trans_source_ag, trans_source_nonag, feasible_ag2ag_mrj, feasible_ag2nonag_rk, feasible_nonag2ag_mrj)

    # ── 4. masks: the cell sets that restrict ag-management options ──
    savanna_eligible_r = get_savanna_eligible_r(data)
    masks = dict(gbf2_solar=get_renewable_GBF2_mask_solar_idx(data), gbf2_wind=get_renewable_GBF2_mask_wind_idx(data),
                 mnes_solar=get_renewable_MNES_mask_solar_idx(data), mnes_wind=get_renewable_MNES_mask_wind_idx(data))

    # ── 5. the blocks, with block-local ids ──
    ag = ag_space(data, trans_feasible_ag, trans_lb_ag_mrj, trans_ub_ag_mrj, dvar_base_ag_mrj)
    space = dict(
        ag         = ag,
        nonag      = nonag_space(data, trans_feasible_nonag, trans_lb_nonag_rk, trans_ub_nonag_rk, dvar_base_non_ag_rk),
        am         = am_space(data, trans_feasible_ag, data.AGMAN2LU, masks['gbf2_solar'], masks['gbf2_wind'], savanna_eligible_r, ag_man_lb_mrj),
        acct       = acct_space(ag, ag_fold_map),
        cell_usage = cell_usage_space(cell_usage_feasible_r),
    )
    for name in ('ag2ag', 'ag2nonag', 'nonag2ag'):                        # the arc lists as Datasets over 'arc'
        table = table_flow[name]
        fields = {field: (('arc',), values) for field, values in table['fields'].items()}
        fields['col'] = (('arc',), np.arange(table['n'], dtype=np.int32))
        space[name] = xr.Dataset(fields, coords=dict(arc=np.arange(table['n'])),
                                 attrs=dict(n=table['n'], block=name, src_ptr=table['src_ptr'], sources=table['sources']))

    # ── 6. layout: the offset of every block in Var.index order; every block-local id shifted to its global Var.index, in place ──
    n_cols = {block: space[block].attrs['n_new' if block == 'acct' else 'n'] for block in BLOCK_ORDER}
    layout, next_col = {}, 0
    for block in BLOCK_ORDER:
        layout[block] = next_col
        next_col += n_cols[block]
        if block == 'acct':
            layout['n_dec'] = next_col                                    # the decision columns, accounting block included
    layout['n_all'] = next_col                                            # + the cell-usage range slacks
    acct = space['acct']
    is_new = acct['is_new'].values
    aliases = (acct['col'].values >= 0) & ~is_new                         # an aliased accounting entry IS its ag column
    shifts = [(space[block]['col'].values, layout[block], None) for block in BLOCK_ORDER if block != 'acct']
    shifts += [(acct['col'].values, layout['ag'], aliases), (acct['col'].values, layout['acct'], is_new)]
    shifts += [(acct[fold_col].values, layout['ag'], None) for fold_col in ('sliver_dom_ag_col', 'sliver_ag_col', 'dom_ag_col')]
    shifts += [(acct[fold_col].values, layout['acct'], None) for fold_col in ('sliver_acct_col', 'dom_acct_col')]
    for col, offset, where in shifts:
        selected = (col >= 0) if where is None else where                 # -1 (no variable) never shifts
        col[selected] += offset
    print(f"    column space: ag {n_cols['ag']:,} | nonag {n_cols['nonag']:,} | am {n_cols['am']:,} | ag2ag {n_cols['ag2ag']:,} | "
          f"ag2nonag {n_cols['ag2nonag']:,} | nonag2ag {n_cols['nonag2ag']:,} | acct(new) {n_cols['acct']:,} "
          f"(= {acct.attrs['n_dom']:,} dominants + {acct.attrs['n_sliver']:,} slivers) | cell_usage {n_cols['cell_usage']:,} "
          f"-> n_dec {layout['n_dec']:,}, n_all {layout['n_all']:,}", flush=True)

    # ── 7. the coefficient support: one term per accounting entry (alias or own column), per ag-mgt column and per
    #       non-ag column — every unknown a policy coefficient can multiply, with its cell and its global column.
    #       Read by every policy family (row_builder.gather_coeffs / compose_rows) and by the objective.
    acct_lu, acct_lm, acct_cell = np.nonzero(acct['exists'].transpose('lu', 'lm', 'cell').values)   # column order: lu, lm, cell
    ag_terms = dict(m=acct_lm.astype(np.int32), j=acct_lu.astype(np.int32), r=acct_cell.astype(np.int32),
                    col=acct['col'].values[acct_lm, acct_lu, acct_cell])
    am = space['am']
    am_slot, am_lm, am_cell = np.nonzero(am['exists'].values)                                         # column order: slot, lm, cell
    am_list = list(am.attrs['am_list'])
    am_idx_of_slot = np.array([am_list.index(name) for name in am['am'].values], dtype=np.int32)      # slot -> index into am_list
    j_idx_of_slot = np.zeros(am.sizes['slot'], dtype=np.int32)                                        # slot -> position of its land use within the option
    for am_idx in range(len(am_list)):
        slots_of_option = np.flatnonzero(am_idx_of_slot == am_idx)
        j_idx_of_slot[slots_of_option] = np.arange(slots_of_option.size, dtype=np.int32)
    am_terms = dict(am_idx=am_idx_of_slot[am_slot], j_idx=j_idx_of_slot[am_slot], j=am['j'].values[am_slot],
                    m=am_lm.astype(np.int32), r=am_cell.astype(np.int32), col=am['col'].values[am_slot, am_lm, am_cell])
    nonag = space['nonag']
    nonag_k, nonag_cell = np.nonzero(nonag['exists'].values)                                           # column order: k, cell
    nonag_terms = dict(k=nonag_k.astype(np.int32), r=nonag_cell.astype(np.int32), col=nonag['col'].values[nonag_k, nonag_cell])
    term_cell = np.concatenate([ag_terms['r'], am_terms['r'], nonag_terms['r']]).astype(np.int32)   # the term order: ag | am | nonag
    term_col = np.concatenate([ag_terms['col'], am_terms['col'], nonag_terms['col']]).astype(np.int32)
    by_cell_order = np.argsort(term_cell, kind='stable')                                             # terms sorted by cell + CSR pointer over cells
    by_cell_ptr = np.searchsorted(term_cell[by_cell_order], np.arange(data.NCELLS + 1))

    # ── 8. what the rows and the post-solve read need besides the columns ──
    space['layout']   = layout
    space['terms']    = dict(ag=ag_terms, am=am_terms, nonag=nonag_terms, am_list=am_list, r=term_cell, col=term_col,
                             ncells=int(data.NCELLS), by_cell=(by_cell_order, by_cell_ptr))
    space['feasible'] = dict(ag=trans_feasible_ag, nonag=trans_feasible_nonag)
    space['sources']  = dict(ag=trans_source_ag, nonag=trans_source_nonag)
    space['masks']    = masks
    space['am'].attrs['agman2lu'] = data.AGMAN2LU
    space['am'].attrs['savanna_eligible_r'] = np.asarray(savanna_eligible_r)
    return space
