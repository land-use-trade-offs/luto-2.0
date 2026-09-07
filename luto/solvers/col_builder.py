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


# ═══════════════════════════ data: what decides existence, bounds and base (from the base-year state) ═══════════════════════════

# ── exclude matrix, transition bounds, ag-management lower bounds ──

def get_ag_eligible_mrj(data: Data, base_year: int) -> np.ndarray:
    """Bool (NLMS, NCELLS, N_AG_LUS): which ag (lm, lu) a cell may become (reachability × EXCLUDE × no-go)."""
    print('Getting agricultural target eligibility...', flush = True)
    return ag_transition.get_ag_eligible_mrj(data, base_year)

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


# ── feasibility: which cell-usage rows and which transition arcs exist ──
# (which target ENTRIES exist is the eligibility mask itself: ag_eligible_mrj for ag, trans_ub_nonag_rk > 0 for non-ag)

def get_cell_usage_feasible_r(trans_ub_ag_mrj: np.ndarray, trans_ub_nonag_rk: np.ndarray, ag_mask_r: np.ndarray) -> np.ndarray:
    """Cells (bool over r) that can meet the cell-usage equality Σ(ag + non-ag shares) = ag_mask.
    A cell with any ag var can always cover ag_mask (its sources can "stay"); a cell without one is
    limited by the sum of its non-ag upper bounds (no variables at all, or a capped non-ag option)."""
    print('Getting cells that can meet the cell-usage equality...', flush=True)
    has_any_ag_r = (trans_ub_ag_mrj > 0).any(axis=(0, 2))
    max_nonag_r  = trans_ub_nonag_rk.sum(axis=1)
    max_alloc_r  = np.where(has_any_ag_r, 1.0, max_nonag_r)
    return max_alloc_r >= ag_mask_r - 1e-6

def get_feasible_ag2ag_mrj(ag_eligible_mrj: np.ndarray, trans_source_ag: dict, T_ag2ag_reach_jj: np.ndarray) -> dict:
    """Ag2ag delta-var feasibility, SOURCE-KEYED like flow_cost_ag2ag:
    {(from_m, from_j): bool (NLMS, ncells_src, N_AG_LUS) [to_m, local_r, to_j]}, True where the
    target is eligible (its X var exists), T_MAT allows from_j -> to_j, and it is not the diagonal."""
    print('Getting feasible ag2ag delta-var targets...', flush = True)
    result = {}
    for (from_m, from_j), cells in trans_source_ag.items():
        valid = ag_eligible_mrj[:, cells, :] & T_ag2ag_reach_jj[from_j][None, None, :]   # (NLMS, ncells_src, N_AG)
        valid[from_m, :, from_j] = False                                            # staying is not a transition
        result[(from_m, from_j)] = valid
    return result

def get_feasible_nonag2ag_mrj(ag_eligible_mrj: np.ndarray, trans_source_nonag: dict, T_nonag2ag_reach_kj: np.ndarray) -> dict:
    """Nonag2ag delta-var feasibility, SOURCE-KEYED: {from_k: bool (NLMS, ncells_k, N_AG_LUS)
    [to_m, local_r, to_j]} — as get_feasible_ag2ag_mrj from the non-ag sources (e.g. reversible
    Destocked land back to ag); no diagonal to drop."""
    print('Getting feasible nonag2ag delta-var targets...', flush = True)
    return {
        from_k: ag_eligible_mrj[:, cells, :] & T_nonag2ag_reach_kj[from_k][None, None, :]   # (NLMS, ncells_k, N_AG)
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
    arc_rows = []
    src_ptr = [0]
    for src_idx, ((from_m, from_j), valid) in enumerate(feasible_ag2ag_mrj.items()):
        to_m, local_r, to_j = np.argwhere(valid).T
        cell = trans_source_ag[(from_m, from_j)][local_r]
        arc_rows.append(np.column_stack([np.full(cell.size, src_idx), np.full(cell.size, from_m), np.full(cell.size, from_j),
                                         local_r, cell, to_m, to_j]))
        src_ptr.append(src_ptr[-1] + cell.size)
    table = np.concatenate(arc_rows) if arc_rows else np.empty((0, 7), dtype=np.int32)
    fields = np.ascontiguousarray(table.T, dtype=np.int32)             # transposed: make each field one contiguous row
    ag2ag = dict(fields=dict(zip(('src', 'from_m', 'from_j', 'local_r', 'cell', 'to_m', 'to_j'), fields)),
                 n=fields.shape[1], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_ag2ag_mrj))

    # ── ag → non-ag: source (from_m, from_j); arc fields (local_r, to_k) ──
    arc_rows = []
    src_ptr = [0]
    for src_idx, ((from_m, from_j), valid) in enumerate(feasible_ag2nonag_rk.items()):
        local_r, to_k = np.argwhere(valid).T
        cell = trans_source_ag[(from_m, from_j)][local_r]
        arc_rows.append(np.column_stack([np.full(cell.size, src_idx), np.full(cell.size, from_m), np.full(cell.size, from_j),
                                         local_r, cell, to_k]))
        src_ptr.append(src_ptr[-1] + cell.size)
    table = np.concatenate(arc_rows) if arc_rows else np.empty((0, 6), dtype=np.int32)
    fields = np.ascontiguousarray(table.T, dtype=np.int32)
    ag2nonag = dict(fields=dict(zip(('src', 'from_m', 'from_j', 'local_r', 'cell', 'to_k'), fields)),
                    n=fields.shape[1], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_ag2nonag_rk))

    # ── non-ag → ag: source from_k; arc fields (to_m, local_r, to_j) ──
    arc_rows = []
    src_ptr = [0]
    for src_idx, (from_k, valid) in enumerate(feasible_nonag2ag_mrj.items()):
        to_m, local_r, to_j = np.argwhere(valid).T
        cell = trans_source_nonag[from_k][local_r]
        arc_rows.append(np.column_stack([np.full(cell.size, src_idx), np.full(cell.size, from_k), local_r, cell, to_m, to_j]))
        src_ptr.append(src_ptr[-1] + cell.size)
    table = np.concatenate(arc_rows) if arc_rows else np.empty((0, 6), dtype=np.int32)
    fields = np.ascontiguousarray(table.T, dtype=np.int32)
    nonag2ag = dict(fields=dict(zip(('src', 'from_k', 'local_r', 'cell', 'to_m', 'to_j'), fields)),
                    n=fields.shape[1], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_nonag2ag_mrj))

    print(f'    ag2ag {ag2ag["n"]:,} | ag2nonag {ag2nonag["n"]:,} | nonag2ag {nonag2ag["n"]:,} arcs', flush = True)
    return dict(ag2ag=ag2ag, ag2nonag=ag2nonag, nonag2ag=nonag2ag)


# ── masks: cell sets that restrict ag-management options ──

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

def ag_space(data: Data, ag_eligible_mrj, trans_ub_ag_mrj, dvar_base_ag_mrj) -> xr.Dataset:
    """X_ag on (lm, lu, cell): an entry exists where the cell may become that (lm, lu)
    (``ag_eligible_mrj``). Column order = lu, lm, cell ascending. lb = 0 (no ag lower bound: the
    θ fold superseded the sub-θ sliver pin), ub = the transition upper bound."""
    # the cubes are stored (lm, lu, cell); every transposed view is made contiguous so downstream gathers read whole rows
    to_lm_lu_cell = lambda cube_mrj: np.ascontiguousarray(cube_mrj.transpose(0, 2, 1))   # (m, r, j) -> (m, j, r)
    exists = to_lm_lu_cell(ag_eligible_mrj)
    
    col = np.full(exists.shape, -1, dtype=np.int32)
    col.transpose(1, 0, 2)[exists.transpose(1, 0, 2)] = np.arange(exists.sum(), dtype=np.int32)   # ids in C-order over (lu, lm, cell)
    
    return xr.Dataset(
        dict(exists=(('lm', 'lu', 'cell'), exists),
             col   =(('lm', 'lu', 'cell'), col),
             ub    =(('lm', 'lu', 'cell'), to_lm_lu_cell(trans_ub_ag_mrj)),
             base  =(('lm', 'lu', 'cell'), to_lm_lu_cell(dvar_base_ag_mrj))
        ),
        coords=dict(
            lm=list(data.LANDMANS), 
            lu=list(data.AGRICULTURAL_LANDUSES), 
            cell=np.arange(data.NCELLS)
        ),
        attrs=dict(n=int(exists.sum()))
    )


def nonag_space(data: Data, trans_lb_nonag_rk, trans_ub_nonag_rk, dvar_base_non_ag_rk, pin_cracks: bool = True) -> xr.Dataset:
    """X_non_ag on (nonag_lu, cell); column order = nonag_lu, cell ascending. ``feasible`` = ub > 0
    for EVERY land use (the node-balance rows need the disabled ones too); ``exists`` = feasible
    AND the land use is enabled (only those get a column)."""
    feasible = np.ascontiguousarray((trans_ub_nonag_rk > 0).T)                 # (r, k) -> (k, r)
    enabled = np.array([settings.NON_AG_LAND_USES[lu_name] for lu_name in data.NON_AGRICULTURAL_LANDUSES], dtype=bool)
    exists = feasible & enabled[:, None]
    
    col = np.full(exists.shape, -1, dtype=np.int32)
    col[exists] = np.arange(exists.sum(), dtype=np.int32)
    
    # collapse rule: where lb > 0 and ub is within 1% of lb, both bounds are pinned to the base value
    if pin_cracks:
        pinned = (trans_lb_nonag_rk > 0) & (np.abs(trans_ub_nonag_rk - trans_lb_nonag_rk) / np.where(trans_lb_nonag_rk > 0, trans_lb_nonag_rk, 1.0) < 0.01)
    else:
        pinned = np.zeros(trans_lb_nonag_rk.shape, dtype=bool)
    
    return xr.Dataset(
        dict(feasible=(('nonag_lu', 'cell'), feasible),
             exists=(('nonag_lu', 'cell'), exists),
             col   =(('nonag_lu', 'cell'), col),
             # (r, k) -> (k, r): contiguous after the transpose
             lb    =(('nonag_lu', 'cell'), np.ascontiguousarray(np.where(pinned, dvar_base_non_ag_rk, trans_lb_nonag_rk).T)),
             ub    =(('nonag_lu', 'cell'), np.ascontiguousarray(np.where(pinned, dvar_base_non_ag_rk, trans_ub_nonag_rk).T)),
             base  =(('nonag_lu', 'cell'), np.ascontiguousarray(dvar_base_non_ag_rk.T))
        ),
        coords=dict(
            nonag_lu=list(data.NON_AGRICULTURAL_LANDUSES), 
            cell=np.arange(data.NCELLS)
        ),
        attrs=dict(n=int(exists.sum()))
    )


def am_space(data: Data, ag_exists: np.ndarray, renewable_GBF2_mask_solar_idx, renewable_GBF2_mask_wind_idx, ag_man_lb_mrj: dict) -> xr.Dataset:
    """X_ag_man on (slot, lm, cell), slot = MultiIndex (am, lu) over the enabled (option, land use)
    pairs in ``data.AGMAN2LU`` order; column order = slot, lm, cell. An entry exists where the ag
    entry exists (``ag_exists``, the ag block's (lm, lu, cell) cube); renewables drop the
    GBF2-exclusion cells (both lm), savanna burning keeps only the savanna-eligible cells (dry).
    lb = ag_man_lb_mrj for non-reversible options, else 0; ub = 1."""
    pairs = [(am, lu_code) for am, lu_codes in data.AGMAN2LU.items() for lu_code in lu_codes]
    lb_sources = [ag_man_lb_mrj[am] for am in data.AGMAN2LU if not settings.AG_MANAGEMENTS_REVERSIBLE[am]]
    lb_dtype = np.result_type(*[source.dtype for source in lb_sources]) if lb_sources else np.float32   # the sources' own dtype
    
    exists = np.zeros((len(pairs), data.NLMS, data.NCELLS), dtype=bool)
    lb = np.zeros((len(pairs), data.NLMS, data.NCELLS), dtype=lb_dtype)
    savanna_mask = data.SAVBURN_ELIGIBLE == 1                                          # cells eligible for savanna burning
    
    for slot, (am, j) in enumerate(pairs):
        slot_exists = ag_exists[:, j, :].copy()                                        # (lm, cell)
        # exclude cells for renewable options
        if am in settings.RENEWABLES_OPTIONS:
            excluded_cells = renewable_GBF2_mask_solar_idx if am == "Utility Solar PV" else renewable_GBF2_mask_wind_idx
            slot_exists[:, excluded_cells] = False
        # exclude cells for savanna burning
        elif tools.am_name_snake_case(am) == "savanna_burning":
            slot_exists[0] &= savanna_mask                                             # dry only

        exists[slot] = slot_exists
        # set lower bounds for non-reversible options
        if not settings.AG_MANAGEMENTS_REVERSIBLE[am]:
            lb[slot] = ag_man_lb_mrj[am][:, :, j]                                      # (lm, cell) of land use j
    
    col = np.full(exists.shape, -1, dtype=np.int32)
    col[exists] = np.arange(exists.sum(), dtype=np.int32)
    
    slot_index = pd.MultiIndex.from_arrays([[am for am, _ in pairs], [data.AGRICULTURAL_LANDUSES[j] for _, j in pairs]], names=['am', 'lu'])
    coords = xr.Coordinates.from_pandas_multiindex(slot_index, 'slot').assign(lm=list(data.LANDMANS), cell=np.arange(data.NCELLS))
    return xr.Dataset(
        dict(exists=(('slot', 'lm', 'cell'), exists),
             col   =(('slot', 'lm', 'cell'), col),
             lb    =(('slot', 'lm', 'cell'), lb),
             j     =(('slot',), np.array([j for _, j in pairs], dtype=np.int32))
        ),
        coords=coords, 
        attrs=dict(
            n=int(exists.sum()), 
            ub=1.0, 
            am_list=list(data.AGMAN2LU), agman2lu=data.AGMAN2LU
        )
    )


def accounting_space(ag: xr.Dataset, ag_fold_map: dict) -> xr.Dataset:
    """The accounting columns: one per entry the θ fold makes different from its flow variable —
    every kept sliver (its dominant must own an ag column) and every receiving dominant — as two
    fold tables on dims ``sliver`` / ``dominant``. Everywhere else X_acct IS the ag column (the
    term stream of get_cols aliases it), so nothing is created when nothing folds.
    Accounting-local ids: dominants first (first-appearance order), then slivers (fold-map order).
    The linking rows read

        sliver:    X_acct[sliver] = fold_share · X_ag[dom]  (+ X_ag[sliver] when the sliver owns an ag column)
        dominant:  X_acct[dom]    = (1 − Σ fold_share) · X_ag[dom]

    where ``fold_share`` = the sliver's base-year fraction of the cell / its dominant's fraction after
    the fold (the share of the dominant's folded area that is really the sliver's land use).
    ``*_ag_col`` are ag-local ids and ``*_accounting_col`` accounting-local; get_cols shifts both to Var.index."""
    ag_col = ag['col'].values   # (lm, lu, cell) -> ag-local id (-1 = no ag column)
    lm_names = ag.lm.values
    lu_names = ag.lu.values
    n_lus = ag_col.shape[1]
    ncells = ag_col.shape[2]

    # ── the fold map (intp index arrays, float32 fractions), minus the slivers whose dominant has no ag column
    #    (its land use is banned at that cell in the target year: there is no X_ag[dom] to link to, and the
    #    folded land is force-converted with the dominant's own) ──
    keep        = ag_col[ag_fold_map['to_m'], ag_fold_map['to_j'], ag_fold_map['cells']] >= 0
    
    sliver_m    = ag_fold_map['from_m'][keep]
    sliver_j    = ag_fold_map['from_j'][keep]
    dom_m       = ag_fold_map['to_m'][keep]
    dom_j       = ag_fold_map['to_j'][keep]
    cells       = ag_fold_map['cells'][keep]
    sliver_frac = ag_fold_map['vals'][keep].astype(np.float64)
    dom_frac    = ag_fold_map['folded_dom'][keep].astype(np.float64)
    
    dom_ag_col  = ag_col[dom_m, dom_j, cells]
    n_slivers   = cells.size
    fold_share  = (sliver_frac / dom_frac).astype(np.float32)                   # the sliver's share of its dominant's folded area (float64 quotient, stored float32)

    # ── the receiving dominants: unique (lm, lu, cell), in first-appearance order ──
    dom_key = (dom_m * n_lus + dom_j) * ncells + cells                          # the shifted key of the dominant's (lm, lu, cell) tuple
    _, first_pos, dom_of_sliver = np.unique(dom_key, return_index=True, return_inverse=True)
    appearance = np.argsort(first_pos, kind='stable')                           # unique() sorts by key: re-rank by first appearance
    rank = np.empty(appearance.size, dtype=np.int64)
    rank[appearance] = np.arange(appearance.size)
    dom_of_sliver = rank[dom_of_sliver]                                         # per sliver: the row of its dominant
    first_pos = first_pos[appearance]
    n_dom = first_pos.size
    dom_rows_m = dom_m[first_pos]
    dom_rows_j = dom_j[first_pos]
    dom_rows_cell = cells[first_pos]

    # ── accounting-local ids: dominants 0..n_dom-1, then the slivers ──
    dom_accounting_col = np.arange(n_dom, dtype=np.int32)
    sliver_accounting_col = (n_dom + np.arange(n_slivers)).astype(np.int32)
    sliver_ag_col = ag_col[sliver_m, sliver_j, cells]                         # -1 = the sliver owns no ag column
    dom_fold_share_sum = np.zeros(n_dom, dtype=np.float64)                    # Σ fold_share over the slivers of each dominant
    np.add.at(dom_fold_share_sum, dom_of_sliver, fold_share.astype(np.float64))

    return xr.Dataset(
        dict(# the fold table: one row per kept sliver
             sliver_from_lm=(('sliver',), lm_names[sliver_m]),
             sliver_from_lu=(('sliver',), lu_names[sliver_j]),
             sliver_from_m=(('sliver',), sliver_m.astype(np.int32)),
             sliver_from_j=(('sliver',), sliver_j.astype(np.int32)),
             sliver_cell=(('sliver',), cells),
             sliver_to_lm=(('sliver',), lm_names[dom_m]),
             sliver_to_lu=(('sliver',), lu_names[dom_j]),
             sliver_fold_share=(('sliver',), fold_share),
             sliver_dom=(('sliver',), dom_of_sliver),
             sliver_dom_ag_col=(('sliver',), dom_ag_col.astype(np.int32)),
             sliver_ag_col=(('sliver',), sliver_ag_col.astype(np.int32)),
             sliver_accounting_col=(('sliver',), sliver_accounting_col),
             # one row per receiving dominant
             dom_lm=(('dominant',), lm_names[dom_rows_m]),
             dom_lu=(('dominant',), lu_names[dom_rows_j]),
             dom_cell=(('dominant',), dom_rows_cell),
             dom_ag_col=(('dominant',), dom_ag_col[first_pos].astype(np.int32)),
             dom_accounting_col=(('dominant',), dom_accounting_col),
             dom_fold_share_sum=(('dominant',), dom_fold_share_sum)),
        coords=dict(sliver=np.arange(n_slivers), dominant=np.arange(n_dom)),
        attrs=dict(n_new=int(n_dom + n_slivers), n_dom=int(n_dom), n_sliver=int(n_slivers))
    )


def cell_usage_space(cell_usage_feasible_r) -> xr.Dataset:
    """The cell-usage range slacks on (cell): one per cell that gets a cell-usage row."""
    exists = cell_usage_feasible_r
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
    model as a labelled Dataset per block (``block_order``) with global Var.index ids, plus what
    the rows and the post-solve read need about the columns —

        cols['layout']    block offsets in Var.index order, n_dec (decision columns) and n_all (+ slacks)
        cols['terms']     the coefficient support: one term per accounting entry / ag-mgt column / non-ag column
        cols['sources']   {'ag': {(from_m, from_j): cells}, 'nonag': {k: cells}} — the base-year holders of land
        cols['masks']     the renewable exclusion cell indices (GBF2 / MNES, solar / wind)
        cols['am'].attrs  'agman2lu', 'savanna_eligible_r'
    """

    # ── 1. transition bounds and the base (TO-view) ──
    ag_eligible_mrj   = get_ag_eligible_mrj(data, base_year)              # bool: which (m, j) a cell may become
    trans_ub_ag_mrj   = get_trans_ub_ag_mrj(data, base_year)              # ag target upper bound (ag2ag + nonag2ag); ag has no lower bound
    trans_ub_nonag_rk = get_trans_ub_nonag_rk(data, base_year)
    trans_lb_nonag_rk = get_trans_lb_nonag_rk(data, base_year)
    ag_man_lb_mrj     = get_ag_man_lb_mrj(data, base_year)                # non-reversible options lock in last step's adoption
    # the base dvars are the node-balance "stay" constant: clipped into the cleaned [lb, ub] box so the
    # all-delta-zero stay point is feasible by construction (only bites on float noise, e.g. -1e-8 < lb = 0; reported)
    dvar_base_ag_mrj    = tools.clamp_dvar_bound(ag_transition.get_folded_base_ag_dvar(data, base_year), 0.0, trans_ub_ag_mrj, 'Ag base clipped to [0,ub]')
    dvar_base_non_ag_rk = tools.clamp_dvar_bound(data.non_ag_dvars[base_year], trans_lb_nonag_rk, trans_ub_nonag_rk, 'NonAg base clipped to [lb,ub]')
    ag_fold_map         = ag_transition.get_ag_dvar_fold_map(data, base_year)   # which sub-θ slivers fold into which dominant

    # ── 2. sources (FROM-view): the base-year holders of land ──
    trans_source_ag    = get_trans_source_ag(data, base_year)             # cells holding each ag (from_m, from_j) source
    trans_source_nonag = get_trans_source_nonag(data, base_year)          # cells holding each non-ag source k

    # ── 3. feasibility: cell-usage rows and transition arcs (target entries: ag_eligible_mrj, trans_ub_nonag_rk > 0) ──
    cell_usage_feasible_r = get_cell_usage_feasible_r(trans_ub_ag_mrj, trans_ub_nonag_rk, data.AG_MASK_PROPORTION_R)
    # per-source reachability (T_MAT finite ⇒ allowed) decides which delta vars exist
    T_ag2ag_reach_jj    = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.AGRICULTURAL_LANDUSES).values)
    T_ag2nonag_reach_jk = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.NON_AGRICULTURAL_LANDUSES).values)
    T_nonag2ag_reach_kj = ~np.isnan(data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values)
    feasible_ag2ag_mrj    = get_feasible_ag2ag_mrj(ag_eligible_mrj, trans_source_ag, T_ag2ag_reach_jj)
    feasible_nonag2ag_mrj = get_feasible_nonag2ag_mrj(ag_eligible_mrj, trans_source_nonag, T_nonag2ag_reach_kj)
    feasible_ag2nonag_rk  = get_feasible_ag2nonag_rk(trans_ub_nonag_rk, trans_source_ag, T_ag2nonag_reach_jk)
    table_flow = get_table_flow(trans_source_ag, trans_source_nonag, feasible_ag2ag_mrj, feasible_ag2nonag_rk, feasible_nonag2ag_mrj)

    # ── 4. masks: the cell sets that restrict ag-management options ──
    masks = dict(
        gbf2_solar=get_renewable_GBF2_mask_solar_idx(data), 
        gbf2_wind=get_renewable_GBF2_mask_wind_idx(data),
        mnes_solar=get_renewable_MNES_mask_solar_idx(data), 
        mnes_wind=get_renewable_MNES_mask_wind_idx(data)
    )

    # ── 5. the blocks, with block-local ids ──
    ag = ag_space(data, ag_eligible_mrj, trans_ub_ag_mrj, dvar_base_ag_mrj)
    cols = dict(
        ag         = ag,
        nonag      = nonag_space(data, trans_lb_nonag_rk, trans_ub_nonag_rk, dvar_base_non_ag_rk),
        am         = am_space(data, ag['exists'].values, masks['gbf2_solar'], masks['gbf2_wind'], ag_man_lb_mrj),
        accounting = accounting_space(ag, ag_fold_map),
        cell_usage = cell_usage_space(cell_usage_feasible_r),
    )
    for name in ('ag2ag', 'ag2nonag', 'nonag2ag'):                        # the arc lists as Datasets over 'arc'
        table = table_flow[name]
        fields = {field: (('arc',), values) for field, values in table['fields'].items()}
        fields['col'] = (('arc',), np.arange(table['n'], dtype=np.int32))
        cols[name] = xr.Dataset(fields, coords=dict(arc=np.arange(table['n'])),
                                 attrs=dict(n=table['n'], block=name, src_ptr=table['src_ptr'], sources=table['sources']))

    # ── 6. layout: the offset of every block in Var.index order; every block-local id shifted to its global Var.index, in place ──
    block_order = ('ag', 'nonag', 'am', 'ag2ag', 'ag2nonag', 'nonag2ag', 'accounting', 'cell_usage')   # the Var.index order of the blocks
    n_cols = {block: cols[block].attrs['n_new' if block == 'accounting' else 'n'] for block in block_order}
    layout = {}
    next_col = 0
    for block in block_order:
        layout[block] = next_col
        next_col += n_cols[block]
        if block == 'accounting':
            layout['n_dec'] = next_col                                    # the decision columns, accounting block included
    layout['n_all'] = next_col                                            # + the cell-usage range slacks
    accounting = cols['accounting']
    shifts = [(cols[block]['col'].values, layout[block]) for block in block_order if block != 'accounting']
    shifts += [(accounting[fold_col].values, layout['ag']) for fold_col in ('sliver_dom_ag_col', 'sliver_ag_col', 'dom_ag_col')]
    shifts += [(accounting[fold_col].values, layout['accounting']) for fold_col in ('sliver_accounting_col', 'dom_accounting_col')]
    for col, offset in shifts:
        col[col >= 0] += offset                                           # -1 (no variable) never shifts
    print(f"    column space: ag {n_cols['ag']:,} | nonag {n_cols['nonag']:,} | am {n_cols['am']:,} | ag2ag {n_cols['ag2ag']:,} | "
          f"ag2nonag {n_cols['ag2nonag']:,} | nonag2ag {n_cols['nonag2ag']:,} | accounting(new) {n_cols['accounting']:,} "
          f"(= {accounting.attrs['n_dom']:,} dominants + {accounting.attrs['n_sliver']:,} slivers) | cell_usage {n_cols['cell_usage']:,} "
          f"-> n_dec {layout['n_dec']:,}, n_all {layout['n_all']:,}", flush=True)

    # ── 7. the coefficient support: one term per accounting entry, per ag-mgt column and per non-ag column — every
    #       unknown a policy coefficient can multiply, with its cell and its global column. Read by every policy
    #       family (row_builder.gather_coeffs / compose_rows) and by the objective.
    # the accounting view of the ag block: every ag entry, the folded ones redirected to their own accounting column
    # (the term index of an ag entry is its ag-local id: both are numbered in C-order over (lu, lm, cell))
    ag_lu, ag_lm, ag_cell = np.nonzero(ag['exists'].transpose('lu', 'lm', 'cell').values)
    ag_term_col = ag['col'].values[ag_lm, ag_lu, ag_cell].copy()
    ag_term_col[accounting['dom_ag_col'].values - layout['ag']] = accounting['dom_accounting_col'].values
    owns_ag_col = accounting['sliver_ag_col'].values >= 0
    ag_term_col[accounting['sliver_ag_col'].values[owns_ag_col] - layout['ag']] = accounting['sliver_accounting_col'].values[owns_ag_col]
    # a sliver without an ag column is an accounting entry of its own: appended to the stream
    ag_terms = dict(m=np.concatenate([ag_lm, accounting['sliver_from_m'].values[~owns_ag_col]]).astype(np.int32),
                    j=np.concatenate([ag_lu, accounting['sliver_from_j'].values[~owns_ag_col]]).astype(np.int32),
                    r=np.concatenate([ag_cell, accounting['sliver_cell'].values[~owns_ag_col]]).astype(np.int32),
                    col=np.concatenate([ag_term_col, accounting['sliver_accounting_col'].values[~owns_ag_col]]).astype(np.int32))
    am = cols['am']
    am_slot, am_lm, am_cell = np.nonzero(am['exists'].values)                                         # column order: slot, lm, cell
    am_list = list(am.attrs['am_list'])
    am_idx_of_slot = np.array([am_list.index(name) for name in am['am'].values], dtype=np.int32)      # slot -> index into am_list
    j_idx_of_slot = np.zeros(am.sizes['slot'], dtype=np.int32)                                        # slot -> position of its land use within the option
    for am_idx in range(len(am_list)):
        slots_of_option = np.flatnonzero(am_idx_of_slot == am_idx)
        j_idx_of_slot[slots_of_option] = np.arange(slots_of_option.size, dtype=np.int32)
    am_terms = dict(am_idx=am_idx_of_slot[am_slot], j_idx=j_idx_of_slot[am_slot], j=am['j'].values[am_slot],
                    m=am_lm.astype(np.int32), r=am_cell.astype(np.int32), col=am['col'].values[am_slot, am_lm, am_cell])
    nonag = cols['nonag']
    nonag_k, nonag_cell = np.nonzero(nonag['exists'].values)                                           # column order: k, cell
    nonag_terms = dict(k=nonag_k.astype(np.int32), r=nonag_cell.astype(np.int32), col=nonag['col'].values[nonag_k, nonag_cell])
    term_cell = np.concatenate([ag_terms['r'], am_terms['r'], nonag_terms['r']]).astype(np.int32)   # the term order: ag | am | nonag
    term_col = np.concatenate([ag_terms['col'], am_terms['col'], nonag_terms['col']]).astype(np.int32)
    by_cell_order = np.argsort(term_cell, kind='stable')                                             # terms sorted by cell + CSR pointer over cells
    by_cell_ptr = np.searchsorted(term_cell[by_cell_order], np.arange(data.NCELLS + 1))

    # ── 8. what the rows and the post-solve read need besides the columns ──
    cols['layout']   = layout
    cols['terms']    = dict(
        ag=ag_terms, 
        am=am_terms, 
        nonag=nonag_terms, 
        am_list=am_list, 
        r=term_cell, 
        col=term_col,
        ncells=int(data.NCELLS), 
        by_cell=(by_cell_order, by_cell_ptr)
    )
    cols['sources']  = dict(ag=trans_source_ag, nonag=trans_source_nonag)
    cols['masks']    = masks
    
    return cols
