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

# ── sources: the base-year holders of land (FROM-view) ──

def get_trans_source_ag(data: Data, base_year: int) -> dict:
    """{(from_m, from_j): cells} — the cells holding each ag (lm, lu) in the base year."""
    print('Getting agricultural source cells...', flush = True)
    return ag_transition.get_base_dvar_mj_cell_map(data, base_year)

def get_trans_source_nonag(data: Data, base_year: int) -> dict:
    """{from_k: cells} — the cells holding each non-ag land use in the base year."""
    print('Getting non-agricultural source cells...', flush = True)
    return non_ag_transition.get_base_nonag_dvar_k_cell_map(data, base_year)


# ── transition bounds: the ub / lb of every target entry (TO-view) ──

def get_trans_ub_ag_mrj(data: Data, base_year: int) -> np.ndarray:
    """Ag target upper bound (ag2ag + nonag2ag), raised to the folded base so a cell can always keep its base land use."""
    print('Getting agricultural target upper bounds...', flush = True)
    ub = (ag_transition.get_ag2ag_ub(data, base_year) + non_ag_transition.get_nonag2ag_ub(data, base_year)).astype(np.float32)
    base = ag_transition.get_folded_base_ag_dvar(data, base_year)
    return tools.clamp_dvar_bound(ub, np.maximum(base, 0.0), np.inf, 'Ag ub raised to base')

def get_trans_ub_nonag_rk(data: Data, base_year: int) -> np.ndarray:
    """Non-ag target upper bound, raised to the base so a cell can always keep its base land use."""
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

def get_trans_lb_nonag_rk(data: Data, base_year: int):
    """Non-ag target lower bound, clamped to [0, base]."""
    print('Getting non-agricultural lower bound matrices...', flush = True)
    lb = non_ag_transition.get_non_ag_lb_matrices(data, base_year)
    base = (
        data.non_ag_dvars[base_year].astype(np.float32) if base_year != data.YR_CAL_BASE
        else np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32)
    )
    return tools.clamp_dvar_bound(lb, 0.0, np.maximum(base, 0.0), 'NonAg lb clamped to [0,base]')

def get_trans_lb_ag_man_mrj(data: Data, base_year: int):
    """Ag-management lower bounds: the non-reversible options lock in the base-year adoption."""
    print('Getting agricultural lower bound matrices...', flush = True)
    return ag_transition.get_lower_bound_agricultural_management_matrices(data, base_year)



# ── feasibility: which target entries, transition arcs and cell-usage rows exist ──

def get_feasible_ag_mrj(data: Data, base_year: int) -> np.ndarray:
    """Bool (NLMS, NCELLS, N_AG_LUS): which ag (lm, lu) a cell may become (reachability × EXCLUDE × no-go)."""
    print('Getting feasible agricultural targets...', flush = True)
    return ag_transition.get_ag_eligible_mrj(data, base_year)

def get_feasible_ag2ag_mrj(feasible_ag_mrj: np.ndarray, trans_source_ag: dict, T_ag2ag_reach_jj: np.ndarray) -> dict:
    """{(from_m, from_j): bool (to_m, local_r, to_j)} — the ag targets each ag source may transition to (feasible, T_MAT-reachable, not itself)."""
    print('Getting feasible ag2ag delta-var targets...', flush = True)
    feasible_targets = {}
    for (from_m, from_j), source_cells in trans_source_ag.items():
        is_target_feasible = feasible_ag_mrj[:, source_cells, :] & T_ag2ag_reach_jj[from_j][None, None, :]   # (NLMS, ncells_src, N_AG)
        is_target_feasible[from_m, :, from_j] = False                                                   # staying is not a transition
        feasible_targets[(from_m, from_j)] = is_target_feasible
    return feasible_targets

def get_feasible_nonag2ag_mrj(feasible_ag_mrj: np.ndarray, trans_source_nonag: dict, T_nonag2ag_reach_kj: np.ndarray) -> dict:
    """{from_k: bool (to_m, local_r, to_j)} — the ag targets each non-ag source may transition to (feasible and T_MAT-reachable)."""
    print('Getting feasible nonag2ag delta-var targets...', flush = True)
    return {
        from_k: feasible_ag_mrj[:, source_cells, :] & T_nonag2ag_reach_kj[from_k][None, None, :]   # (NLMS, ncells_k, N_AG)
        for from_k, source_cells in trans_source_nonag.items()
    }

def get_feasible_ag2nonag_rk(trans_ub_nonag_rk: np.ndarray, trans_source_ag: dict, T_ag2nonag_reach_jk: np.ndarray) -> dict:
    """{(from_m, from_j): bool (local_r, to_k)} — the non-ag targets each ag source may transition to (ub > 0 and T_MAT-reachable)."""
    print('Getting feasible ag2nonag delta-var targets...', flush = True)
    feasible_nonag_rk = trans_ub_nonag_rk > 0
    return {
        (from_m, from_j): feasible_nonag_rk[source_cells, :] & T_ag2nonag_reach_jk[from_j][None, :]   # (ncells_src, N_NONAG)
        for (from_m, from_j), source_cells in trans_source_ag.items()
    }

def get_feasible_cell_usage_r(trans_ub_ag_mrj: np.ndarray, trans_ub_nonag_rk: np.ndarray, ag_mask_r: np.ndarray) -> np.ndarray:
    """Bool over cells: which can meet the cell-usage equality Σ(ag + non-ag shares) = ag_mask (any ag var, or enough non-ag ub)."""
    print('Getting cells that can meet the cell-usage equality...', flush=True)
    has_any_ag_r = (trans_ub_ag_mrj > 0).any(axis=(0, 2))
    max_nonag_r  = trans_ub_nonag_rk.sum(axis=1)
    max_alloc_r  = np.where(has_any_ag_r, 1.0, max_nonag_r)
    return max_alloc_r >= ag_mask_r - 1e-6


# ── masks: cell sets that restrict ag-management options ──

def get_mask_gbf2_solar(data: Data) -> np.ndarray:
    """Cell indices where solar is excluded by the GBF2 mask (empty when renewables or the exclusion are off)."""
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_GBF2_MASK_SOLAR)[0]

def get_mask_gbf2_wind(data: Data) -> np.ndarray:
    """Cell indices where wind is excluded by the GBF2 mask (empty when renewables or the exclusion are off)."""
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_GBF2_MASK_WIND)[0]

def get_mask_mnes_solar(data: Data) -> np.ndarray:
    """Cell indices where solar is excluded by the EPBC MNES mask (empty when renewables or the exclusion are off)."""
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_MNES_MASK_SOLAR)[0]

def get_mask_mnes_wind(data: Data) -> np.ndarray:
    """Cell indices where wind is excluded by the EPBC MNES mask (empty when renewables or the exclusion are off)."""
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_MNES_MASK_WIND)[0]


# ═══════════════════════════ the column blocks: one Dataset per block, block-LOCAL ids (get_cols shifts them) ═══════════════════════════

def ag_space(data: Data, feasible_ag_mrj: np.ndarray, trans_ub_ag_mrj: np.ndarray, dvar_base_ag_mrj: np.ndarray) -> xr.Dataset:
    """X_ag on (lm, lu, cell) = the column order: a column where the cell may become that (lm, lu); lb = 0, ub = the transition upper bound."""
    # the (m, r, j) cubes laid out on (m, j, r), contiguous so downstream gathers read whole rows
    has_col  = np.ascontiguousarray(feasible_ag_mrj.transpose(0, 2, 1))
    ub_mjr   = np.ascontiguousarray(trans_ub_ag_mrj.transpose(0, 2, 1))
    base_mjr = np.ascontiguousarray(dvar_base_ag_mrj.transpose(0, 2, 1))

    col_mjr = np.full(has_col.shape, -1, dtype=np.int32)
    col_mjr[has_col] = np.arange(has_col.sum(), dtype=np.int32)

    return xr.Dataset(
        dict(col_mjr =(('lm', 'lu', 'cell'), col_mjr),      # column id (-1 = no column); col_mjr >= 0 is the select for gp.Var creation
             ub_mjr  =(('lm', 'lu', 'cell'), ub_mjr),       # the transition upper bound (lb = 0 by default in gurobi)
             base_mjr=(('lm', 'lu', 'cell'), base_mjr)      # the node-balance constant (X = base + flow-in - flow-out); the flow-out sum has to be <= the base
        ),
        coords=dict(
            lm=list(data.LANDMANS),
            lu=list(data.AGRICULTURAL_LANDUSES),
            cell=np.arange(data.NCELLS)
        ),
        attrs=dict(n=int(has_col.sum()))
    )


def nonag_space(data: Data, trans_lb_nonag_rk: np.ndarray, trans_ub_nonag_rk: np.ndarray, dvar_base_nonag_rk: np.ndarray) -> xr.Dataset:
    """X_non_ag on (nonag_lu, cell): a column where ub > 0 and the land use is enabled; column order nonag_lu, cell."""
    # the (r, k) cubes laid out on (k, r), contiguous after the transpose
    lb_kr   = np.ascontiguousarray(trans_lb_nonag_rk.T)
    ub_kr   = np.ascontiguousarray(trans_ub_nonag_rk.T)
    base_kr = np.ascontiguousarray(dvar_base_nonag_rk.T)
    enabled = np.array([settings.NON_AG_LAND_USES[lu_name] for lu_name in data.NON_AGRICULTURAL_LANDUSES], dtype=bool)
    has_col = (ub_kr > 0) & enabled[:, None]

    col_kr = np.full(has_col.shape, -1, dtype=np.int32)
    col_kr[has_col] = np.arange(has_col.sum(), dtype=np.int32)

    return xr.Dataset(
        dict(col_kr =(('nonag_lu', 'cell'), col_kr),        # column id (-1 = no column); col_kr >= 0 is the select for gp.Var creation
             lb_kr  =(('nonag_lu', 'cell'), lb_kr),         # the transition lower bound
             ub_kr  =(('nonag_lu', 'cell'), ub_kr),         # the transition upper bound (> 0 = feasible, for EVERY land use, enabled or not: the node-balance rows need the disabled ones too)
             base_kr=(('nonag_lu', 'cell'), base_kr)        # the node-balance constant (X = base + flow-in - flow-out); the flow-out sum has to be <= the base
        ),
        coords=dict(
            nonag_lu=list(data.NON_AGRICULTURAL_LANDUSES),
            cell=np.arange(data.NCELLS)
        ),
        attrs=dict(n=int(has_col.sum()))
    )


def am_space(data: Data, ag_col_mjr: np.ndarray, mask_gbf2_solar: np.ndarray, mask_gbf2_wind: np.ndarray, trans_lb_ag_man_mrj: dict) -> xr.Dataset:
    """X_ag_man on (slot = (am, lu), lm, cell): a column where the ag column exists minus the renewable / savanna exclusions; column order slot, lm, cell."""
    pairs = [(am, lu_code) for am, lu_codes in data.AGMAN2LU.items() for lu_code in lu_codes]

    has_col = np.zeros((len(pairs), data.NLMS, data.NCELLS), dtype=bool)
    lb_smr = np.zeros((len(pairs), data.NLMS, data.NCELLS), dtype=np.float32)
    savanna_mask = data.SAVBURN_ELIGIBLE == 1                                          # cells eligible for savanna burning

    for slot, (am, j) in enumerate(pairs):
        slot_has_col = ag_col_mjr[:, j, :] >= 0                                        # (lm, cell): where the ag column exists
        # exclude cells for renewable options
        if am in settings.RENEWABLES_OPTIONS:
            excluded_cells = mask_gbf2_solar if am == "Utility Solar PV" else mask_gbf2_wind
            slot_has_col[:, excluded_cells] = False
        # exclude cells for savanna burning
        elif tools.am_name_snake_case(am) == "savanna_burning":
            slot_has_col[0] &= savanna_mask                                            # dry only
        # set lower bounds for non-reversible options
        if not settings.AG_MANAGEMENTS_REVERSIBLE[am]:
            lb_smr[slot] = trans_lb_ag_man_mrj[am][:, :, j]                            # (lm, cell) of land use j
        # assign the slot's column existence
        has_col[slot] = slot_has_col

    col_smr = np.full(has_col.shape, -1, dtype=np.int32)
    col_smr[has_col] = np.arange(has_col.sum(), dtype=np.int32)

    slot_index = pd.MultiIndex.from_arrays([[am for am, _ in pairs], [data.AGRICULTURAL_LANDUSES[j] for _, j in pairs]], names=['am', 'lu'])
    coords = xr.Coordinates.from_pandas_multiindex(slot_index, 'slot').assign(lm=list(data.LANDMANS), cell=np.arange(data.NCELLS))
    
    return xr.Dataset(
        dict(col_smr=(('slot', 'lm', 'cell'), col_smr),                                # column id (-1 = no column); col_smr >= 0 is the select for gp.Var creation
             lb_smr =(('slot', 'lm', 'cell'), lb_smr),                                 # the base-year adoption for non-reversible options, else 0 (ub = 1 for every column)
             j      =(('slot',), np.array([j for _, j in pairs], dtype=np.int32))      # the land-use code of each (am, lu) slot
        ),
        coords=coords,
        attrs=dict(
            n=int(has_col.sum()),
            agman2lu=data.AGMAN2LU,                                                    # {option: [land-use codes]}: the slot order
            savanna_eligible_r=np.flatnonzero(savanna_mask)                            # the solve read-back zeroes irr savanna columns outside these cells
        )
    )


def ag2ag_space(trans_source_ag: dict, feasible_ag2ag_mrj: dict) -> xr.Dataset:
    """The ag → ag arcs on (arc): one per feasible (source (from_m, from_j) → target (to_m, to_j)) at a cell, sources in trans_source_ag order (src_ptr bounds each)."""
    print('Building the ag2ag arc block...', flush = True)
    arc_rows = []
    src_ptr = [0]
    for src_idx, ((from_m, from_j), is_target_feasible) in enumerate(feasible_ag2ag_mrj.items()):
        to_m, local_r, to_j = np.nonzero(is_target_feasible)
        cell = trans_source_ag[(from_m, from_j)][local_r]
        arc_rows.append(np.column_stack([
            np.full(cell.size, src_idx), 
            np.full(cell.size, from_m), 
            np.full(cell.size, from_j),
            local_r, 
            cell, 
            to_m, 
            to_j]
        ))
        src_ptr.append(src_ptr[-1] + cell.size)
    
    table = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 7), dtype=np.int32)   # one arc per row; the fields are its columns
    
    return xr.Dataset(
        dict(src    =(('arc',), table[:, 0]),
             from_m =(('arc',), table[:, 1]),
             from_j =(('arc',), table[:, 2]),
             local_r=(('arc',), table[:, 3]),
             cell   =(('arc',), table[:, 4]),
             to_m   =(('arc',), table[:, 5]),
             to_j   =(('arc',), table[:, 6]),
             col    =(('arc',), np.arange(table.shape[0], dtype=np.int32))
        ),
        coords=dict(arc=np.arange(table.shape[0])),
        attrs=dict(n=table.shape[0], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_ag2ag_mrj))
    )


def ag2nonag_space(trans_source_ag: dict, feasible_ag2nonag_rk: dict) -> xr.Dataset:
    """The ag → non-ag arcs on (arc): one per feasible (source (from_m, from_j) → target to_k) at a cell, sources in trans_source_ag order."""
    print('Building the ag2nonag arc block...', flush = True)
    arc_rows = []
    src_ptr = [0]
    for src_idx, ((from_m, from_j), is_target_feasible) in enumerate(feasible_ag2nonag_rk.items()):
        local_r, to_k = np.nonzero(is_target_feasible)
        cell = trans_source_ag[(from_m, from_j)][local_r]
        arc_rows.append(np.column_stack([
            np.full(cell.size, src_idx), 
            np.full(cell.size, from_m), 
            np.full(cell.size, from_j),
            local_r, 
            cell, 
            to_k]
        ))
        src_ptr.append(src_ptr[-1] + cell.size)
    
    table = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 6), dtype=np.int32)   # one arc per row; the fields are its columns
    
    return xr.Dataset(
        dict(src    =(('arc',), table[:, 0]),
             from_m =(('arc',), table[:, 1]),
             from_j =(('arc',), table[:, 2]),
             local_r=(('arc',), table[:, 3]),
             cell   =(('arc',), table[:, 4]),
             to_k   =(('arc',), table[:, 5]),
             col    =(('arc',), np.arange(table.shape[0], dtype=np.int32))
        ),
        coords=dict(arc=np.arange(table.shape[0])),
        attrs=dict(n=table.shape[0], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_ag2nonag_rk))
    )


def nonag2ag_space(trans_source_nonag: dict, feasible_nonag2ag_mrj: dict) -> xr.Dataset:
    """The non-ag → ag arcs on (arc): one per feasible (source from_k → target (to_m, to_j)) at a cell, sources in trans_source_nonag order."""
    print('Building the nonag2ag arc block...', flush = True)
    arc_rows = []
    src_ptr = [0]
    for src_idx, (from_k, is_target_feasible) in enumerate(feasible_nonag2ag_mrj.items()):
        to_m, local_r, to_j = np.nonzero(is_target_feasible)
        cell = trans_source_nonag[from_k][local_r]
        arc_rows.append(np.column_stack([
            np.full(cell.size, src_idx), 
            np.full(cell.size, from_k), 
            local_r, 
            cell, 
            to_m, 
            to_j]
        ))
        src_ptr.append(src_ptr[-1] + cell.size)
    
    table = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 6), dtype=np.int32)   # one arc per row; the fields are its columns
    
    return xr.Dataset(
        dict(src    =(('arc',), table[:, 0]),
             from_k =(('arc',), table[:, 1]),
             local_r=(('arc',), table[:, 2]),
             cell   =(('arc',), table[:, 3]),
             to_m   =(('arc',), table[:, 4]),
             to_j   =(('arc',), table[:, 5]),
             col    =(('arc',), np.arange(table.shape[0], dtype=np.int32))
        ),
        coords=dict(arc=np.arange(table.shape[0])),
        attrs=dict(n=table.shape[0], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_nonag2ag_mrj))
    )


def fold_space(ag: xr.Dataset, ag_fold_map: dict) -> xr.Dataset:
    """The fold columns — one per receiver and per kept emitter — as two fold tables on (fold_receiver) / (fold_emitter); receivers numbered first."""
    ag_col_mjr = ag['col_mjr'].values   # (lm, lu, cell) -> ag-local id (-1 = no ag column)
    lm_names   = ag['lm'].values
    lu_names   = ag['lu'].values

    # ── the receivers: every entry that received an emitter AND owns an ag column (without one its land use is banned at that
    #    cell in the target year: no X_ag[receiver] to link to, so its emitters are dropped); ids in C-order over (lm, lu, cell) ──
    fold_applied_mjr = np.ascontiguousarray(ag_fold_map['fold_applied_mrj'].transpose(0, 2, 1)) & (ag_col_mjr >= 0)
    receiver_m, receiver_j, receiver_cell = np.nonzero(fold_applied_mjr)
    n_receiver = receiver_m.size
    receiver_row_mjr = np.full(fold_applied_mjr.shape, -1, dtype=np.int64)   # (lm, lu, cell) -> receiver row (-1 = not a kept receiver)
    receiver_row_mjr[fold_applied_mjr] = np.arange(n_receiver)

    # ── the emitters: those whose receiver is kept, in fold-map order ──
    receiver_of_emitter = receiver_row_mjr[ag_fold_map['to_m'], ag_fold_map['to_j'], ag_fold_map['cells']]
    keep                = receiver_of_emitter >= 0
    receiver_of_emitter = receiver_of_emitter[keep]
    emitter_m           = ag_fold_map['from_m'][keep]
    emitter_j           = ag_fold_map['from_j'][keep]
    to_m                = ag_fold_map['to_m'][keep]
    to_j                = ag_fold_map['to_j'][keep]
    cells               = ag_fold_map['cells'][keep]
    fold_share          = ag_fold_map['vals'][keep] / ag_fold_map['folded_dom'][keep]   # the emitter's share of its receiver's folded area (float32)
    n_emitter           = cells.size

    return xr.Dataset(
        dict(# one row per receiver
             receiver_lm=(('fold_receiver',), lm_names[receiver_m]),
             receiver_lu=(('fold_receiver',), lu_names[receiver_j]),
             receiver_cell=(('fold_receiver',), receiver_cell),
             receiver_ag_col=(('fold_receiver',), ag_col_mjr[receiver_m, receiver_j, receiver_cell]),
             receiver_fold_col=(('fold_receiver',), np.arange(n_receiver, dtype=np.int32)),   # fold-local ids: receivers 0..n_receiver-1 ...
             # the fold table: one row per kept emitter
             emitter_lm=(('fold_emitter',), lm_names[emitter_m]),
             emitter_lu=(('fold_emitter',), lu_names[emitter_j]),
             emitter_m=(('fold_emitter',), emitter_m.astype(np.int32)),
             emitter_j=(('fold_emitter',), emitter_j.astype(np.int32)),
             emitter_cell=(('fold_emitter',), cells),
             emitter_receiver_lm=(('fold_emitter',), lm_names[to_m]),
             emitter_receiver_lu=(('fold_emitter',), lu_names[to_j]),
             emitter_fold_share=(('fold_emitter',), fold_share),
             emitter_receiver=(('fold_emitter',), receiver_of_emitter),   # the row of its receiver
             emitter_receiver_ag_col=(('fold_emitter',), ag_col_mjr[to_m, to_j, cells]),
             emitter_ag_col=(('fold_emitter',), ag_col_mjr[emitter_m, emitter_j, cells]),   # -1 = the emitter owns no ag column
             emitter_fold_col=(('fold_emitter',), (n_receiver + np.arange(n_emitter)).astype(np.int32))   # ... then the emitters
        ),
        coords=dict(
            fold_emitter=np.arange(n_emitter), 
            fold_receiver=np.arange(n_receiver)
        ),
        attrs=dict(
            n=int(n_receiver + n_emitter), 
            n_receiver=int(n_receiver), 
            n_emitter=int(n_emitter)
        )
    )


def cell_usage_space(feasible_cell_usage_r) -> xr.Dataset:
    """The cell-usage range slacks on (cell): one per cell that gets a cell-usage row."""
    has_col = feasible_cell_usage_r
    col_r = np.full(has_col.shape, -1, dtype=np.int32)
    col_r[has_col] = np.arange(has_col.sum(), dtype=np.int32)
    return xr.Dataset(
        dict(col_r=(('cell',), col_r)),                                        # slack column id (-1 = no row, no slack); col_r >= 0 is the select for gp.Var creation
        coords=dict(cell=np.arange(has_col.size)),
        attrs=dict(n=int(has_col.sum()))
    )


def columns(col_grid: xr.DataArray, want: tuple | None = None) -> tuple:
    """A block's columns in column order from its column grid: the index arrays of the dims in ``want`` (default: the last dim) followed by the global column ids."""
    idx = dict(zip(col_grid.dims, np.nonzero(col_grid.values >= 0)))
    col = col_grid.values[tuple(idx.values())]
    return (*[idx[dim].astype(np.int32) for dim in (want or col_grid.dims[-1:])], col)


# ═══════════════════════════ get_cols: the column space of one step ═══════════════════════════

def get_cols(data: Data, base_year: int) -> dict:
    """The column space of one solve step: every unknown as a labelled Dataset per block with global Var.index ids, plus layout, terms, sources and masks."""

    # ── 1. sources (FROM-view): the base-year holders of land ──
    trans_source_ag    = get_trans_source_ag(data, base_year)             # cells holding each ag (from_m, from_j) source
    trans_source_nonag = get_trans_source_nonag(data, base_year)          # cells holding each non-ag source k

    # ── 2. transition bounds and the base (TO-view) ──
    trans_ub_ag_mrj     = get_trans_ub_ag_mrj(data, base_year)            # ag target upper bound (ag2ag + nonag2ag); ag has no lower bound
    trans_ub_nonag_rk   = get_trans_ub_nonag_rk(data, base_year)
    trans_lb_nonag_rk   = get_trans_lb_nonag_rk(data, base_year)
    trans_lb_ag_man_mrj = get_trans_lb_ag_man_mrj(data, base_year)        # non-reversible options lock in last step's adoption
    dvar_base_ag_mrj   = tools.clamp_dvar_bound(ag_transition.get_folded_base_ag_dvar(data, base_year), 0.0, trans_ub_ag_mrj, 'Ag base clipped to [0,ub]')
    dvar_base_nonag_rk = tools.clamp_dvar_bound(data.non_ag_dvars[base_year], trans_lb_nonag_rk, trans_ub_nonag_rk, 'NonAg base clipped to [lb,ub]')
    ag_fold_map        = ag_transition.get_ag_dvar_fold_map(data, base_year)   # which sub-θ emitters fold into which receiver

    # ── 3. feasibility: target entries (feasible_ag_mrj, trans_ub_nonag_rk > 0), transition arcs, cell-usage rows ──
    feasible_ag_mrj = get_feasible_ag_mrj(data, base_year)                # bool: which (m, j) a cell may become
    T_ag2ag_reach_jj    = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.AGRICULTURAL_LANDUSES).values)
    T_ag2nonag_reach_jk = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.NON_AGRICULTURAL_LANDUSES).values)
    T_nonag2ag_reach_kj = ~np.isnan(data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values)
    feasible_ag2ag_mrj    = get_feasible_ag2ag_mrj(feasible_ag_mrj, trans_source_ag, T_ag2ag_reach_jj)
    feasible_nonag2ag_mrj = get_feasible_nonag2ag_mrj(feasible_ag_mrj, trans_source_nonag, T_nonag2ag_reach_kj)
    feasible_ag2nonag_rk  = get_feasible_ag2nonag_rk(trans_ub_nonag_rk, trans_source_ag, T_ag2nonag_reach_jk)
    feasible_cell_usage_r = get_feasible_cell_usage_r(trans_ub_ag_mrj, trans_ub_nonag_rk, data.AG_MASK_PROPORTION_R)

    # ── 4. masks: the cell sets that restrict ag-management options ──
    mask_gbf2_solar = get_mask_gbf2_solar(data)
    mask_gbf2_wind  = get_mask_gbf2_wind(data)
    mask_mnes_solar = get_mask_mnes_solar(data)
    mask_mnes_wind  = get_mask_mnes_wind(data)

    # ── 5. the blocks in Var.index order, with block-local ids ──
    ag = ag_space(data, feasible_ag_mrj, trans_ub_ag_mrj, dvar_base_ag_mrj)
    cols = {
        'ag':         ag,
        'nonag':      nonag_space(data, trans_lb_nonag_rk, trans_ub_nonag_rk, dvar_base_nonag_rk),
        'am':         am_space(data, ag['col_mjr'].values, mask_gbf2_solar, mask_gbf2_wind, trans_lb_ag_man_mrj),
        'ag2ag':      ag2ag_space(trans_source_ag, feasible_ag2ag_mrj),
        'ag2nonag':   ag2nonag_space(trans_source_ag, feasible_ag2nonag_rk),
        'nonag2ag':   nonag2ag_space(trans_source_nonag, feasible_nonag2ag_mrj),
        'fold':       fold_space(ag, ag_fold_map),
        'cell_usage': cell_usage_space(feasible_cell_usage_r),
    }

    # ── 6. layout: the first Var.index of every block, in the order of `cols` ──
    layout = {}
    layout['ag']         = 0
    layout['nonag']      = layout['ag']         + cols['ag'].attrs['n']
    layout['am']         = layout['nonag']      + cols['nonag'].attrs['n']
    layout['ag2ag']      = layout['am']         + cols['am'].attrs['n']
    layout['ag2nonag']   = layout['ag2ag']      + cols['ag2ag'].attrs['n']
    layout['nonag2ag']   = layout['ag2nonag']   + cols['ag2nonag'].attrs['n']
    layout['fold']       = layout['nonag2ag']   + cols['nonag2ag'].attrs['n']
    layout['cell_usage'] = layout['fold']       + cols['fold'].attrs['n']          # fold.n = n_receiver + n_emitter; this mean only create vars on cells fold really happened
    
    layout['n_dec']      = layout['fold']       + cols['fold'].attrs['n']          # the decision columns: the objective block is built at this width (the slacks carry no cost)
    layout['n_all']      = layout['n_dec']      + cols['cell_usage'].attrs['n']    # the total columns: every constraint block (the rows) is built at this width

    # ── 7. every block-local id shifted to its global Var.index, in place ──
    def shift(col, offset):                                                         # -1 (no column) never shifts
        col[col >= 0] += offset

    fold = cols['fold']
    shift(cols['ag']['col_mjr'].values,           layout['ag'])
    shift(cols['nonag']['col_kr'].values,         layout['nonag'])
    shift(cols['am']['col_smr'].values,           layout['am'])
    shift(cols['ag2ag']['col'].values,            layout['ag2ag'])
    shift(cols['ag2nonag']['col'].values,         layout['ag2nonag'])
    shift(cols['nonag2ag']['col'].values,         layout['nonag2ag'])
    shift(cols['cell_usage']['col_r'].values,     layout['cell_usage'])
    shift(fold['receiver_ag_col'].values,         layout['ag'])                     # the fold tables' ag ids shift with the ag block ...
    shift(fold['emitter_ag_col'].values,          layout['ag'])
    shift(fold['emitter_receiver_ag_col'].values, layout['ag'])
    shift(fold['receiver_fold_col'].values,       layout['fold'])                   # ... their own ids with the fold block
    shift(fold['emitter_fold_col'].values,        layout['fold'])
    
    print(
        f"Column space: {layout['n_all']:,} columns = {layout['n_dec']:,} decision (n_dec) + {cols['cell_usage'].attrs['n']:,} cell-usage slacks\n"
        f"├── ag         {cols['ag'].attrs['n']:>12,}\n"
        f"├── nonag      {cols['nonag'].attrs['n']:>12,}\n"
        f"├── am         {cols['am'].attrs['n']:>12,}\n"
        f"├── ag2ag      {cols['ag2ag'].attrs['n']:>12,}\n"
        f"├── ag2nonag   {cols['ag2nonag'].attrs['n']:>12,}\n"
        f"├── nonag2ag   {cols['nonag2ag'].attrs['n']:>12,}\n"
        f"├── fold       {fold.attrs['n']:>12,}   ({fold.attrs['n_receiver']:,} receivers + {fold.attrs['n_emitter']:,} emitters)\n"
        f"└── cell_usage {cols['cell_usage'].attrs['n']:>12,}",
        flush=True
    )

    # ── 8. the coefficient support: one term per accounting entry, per ag-mgt column and per non-ag column — every
    ag_lm, ag_lu, ag_cell = np.nonzero(ag['col_mjr'].values >= 0)
    ag_term_col = ag['col_mjr'].values[ag_lm, ag_lu, ag_cell].copy()
    ag_term_col[fold['receiver_ag_col'].values - layout['ag']] = fold['receiver_fold_col'].values
    owns_ag_col = fold['emitter_ag_col'].values >= 0
    ag_term_col[fold['emitter_ag_col'].values[owns_ag_col] - layout['ag']] = fold['emitter_fold_col'].values[owns_ag_col]
    # an emitter without an ag column is an accounting entry of its own: appended to the stream
    ag_terms = dict(m=np.concatenate([ag_lm, fold['emitter_m'].values[~owns_ag_col]]).astype(np.int32),
                    j=np.concatenate([ag_lu, fold['emitter_j'].values[~owns_ag_col]]).astype(np.int32),
                    r=np.concatenate([ag_cell, fold['emitter_cell'].values[~owns_ag_col]]).astype(np.int32),
                    col=np.concatenate([ag_term_col, fold['emitter_fold_col'].values[~owns_ag_col]]).astype(np.int32))
    
    am = cols['am']
    am_slot, am_lm, am_cell = np.nonzero(am['col_smr'].values >= 0)                                         # column order: slot, lm, cell
    am_list = list(am.attrs['agman2lu'])
    am_idx_of_slot = np.array([am_list.index(name) for name in am['am'].values], dtype=np.int32)            # slot -> index into am_list
    j_idx_of_slot = np.zeros(am.sizes['slot'], dtype=np.int32)                                              # slot -> position of its land use within the option
    for am_idx in range(len(am_list)):
        slots_of_option = np.flatnonzero(am_idx_of_slot == am_idx)
        j_idx_of_slot[slots_of_option] = np.arange(slots_of_option.size, dtype=np.int32)
    am_terms = dict(am_idx=am_idx_of_slot[am_slot], j_idx=j_idx_of_slot[am_slot], j=am['j'].values[am_slot],
                    m=am_lm.astype(np.int32), r=am_cell.astype(np.int32), col=am['col_smr'].values[am_slot, am_lm, am_cell])
    
    nonag = cols['nonag']
    nonag_k, nonag_cell = np.nonzero(nonag['col_kr'].values >= 0)                                           # column order: k, cell
    nonag_terms = dict(k=nonag_k.astype(np.int32), r=nonag_cell.astype(np.int32), col=nonag['col_kr'].values[nonag_k, nonag_cell])
    term_cell = np.concatenate([ag_terms['r'], am_terms['r'], nonag_terms['r']]).astype(np.int32)           # the term order: ag | am | nonag
    term_col = np.concatenate([ag_terms['col'], am_terms['col'], nonag_terms['col']]).astype(np.int32)
    by_cell_order = np.argsort(term_cell, kind='stable')                                                    # terms sorted by cell + CSR pointer over cells
    by_cell_ptr = np.searchsorted(term_cell[by_cell_order], np.arange(data.NCELLS + 1))

    # ── 9. what the rows and the post-solve read need besides the columns ──
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
    cols['mask_gbf2_solar'] = mask_gbf2_solar
    cols['mask_gbf2_wind']  = mask_gbf2_wind
    cols['mask_mnes_solar'] = mask_mnes_solar
    cols['mask_mnes_wind']  = mask_mnes_wind
    
    return cols
