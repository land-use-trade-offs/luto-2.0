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

Two table shapes. The grid blocks (``ag``, ``nonag``, ``am``, ``cell_usage``) are WIDE tables: a
column-id grid over the block's dims (``col``, -1 = no column, the grid order is the column
order) with the attributes of each column (ub, lb, base) as grids of the same shape. The arc
blocks (``ag2ag``, ``ag2nonag``, ``nonag2ag``) are LONG tables: one row per arc column, its fields
the attributes of that column (from, to, local_r, cell, col), the rows sorted by source with
``attrs['src_ptr']`` marking where each source's run starts and ends. ``cols['table']`` is the
whole space as ONE long table on ``col`` = Var.index: one row per column in block order (``BLOCKS``),
its fields the attributes of that column (-1 where a field does not apply), ``attrs['block_ptr']``
marking where each block's run of rows starts and ends. ``cols['terms']`` is the table's first three
blocks (ag, non-ag, ag-mgt) in the order the policy families score them.
"""

import numpy as np
import pandas as pd
import xarray as xr

import luto.settings as settings
import luto.tools as tools
from luto.data import Data
import luto.economics.agricultural.transitions as ag_transition
import luto.economics.non_agricultural.transitions as non_ag_transition


BLOCKS = ('ag', 'nonag', 'am', 'ag2ag', 'ag2nonag', 'nonag2ag', 'cell_usage')   # the blocks of the space, in Var.index order


def block_slice(table: xr.Dataset, block: str) -> slice:
    """The rows of one block of the column table — its Var.index range (``attrs['block_ptr']`` holds the bounds)."""
    code = table.attrs['blocks'].index(block)
    return slice(int(table.attrs['block_ptr'][code]), int(table.attrs['block_ptr'][code + 1]))


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
    """Ag target upper bound (ag2ag + nonag2ag), raised to the base so a cell can always keep its base land use."""
    print('Getting agricultural target upper bounds...', flush = True)
    ub = (ag_transition.get_ag2ag_ub(data, base_year) + non_ag_transition.get_nonag2ag_ub(data, base_year)).astype(np.float32)
    return tools.clamp_dvar_bound(ub, np.maximum(data.ag_dvars[base_year], 0.0), np.inf, 'Ag ub raised to base')

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
        base_dvar_ag_mrj=data.ag_dvars[base_year],
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
        is_target_feasible = feasible_ag_mrj[:, source_cells, :] & T_ag2ag_reach_jj[from_j][None, None, :]      # (NLMS, ncells_src, N_AG)
        is_target_feasible[from_m, :, from_j] = False                                                           # staying is not a transition
        feasible_targets[(from_m, from_j)] = is_target_feasible
    return feasible_targets

def get_feasible_nonag2ag_mrj(feasible_ag_mrj: np.ndarray, trans_source_nonag: dict, T_nonag2ag_reach_kj: np.ndarray) -> dict:
    """{from_k: bool (to_m, local_r, to_j)} — the ag targets each non-ag source may transition to (feasible and T_MAT-reachable)."""
    print('Getting feasible nonag2ag delta-var targets...', flush = True)
    return {
        from_k: feasible_ag_mrj[:, source_cells, :] & T_nonag2ag_reach_kj[from_k][None, None, :]                # (NLMS, ncells_k, N_AG)
        for from_k, source_cells in trans_source_nonag.items()
    }

def get_feasible_ag2nonag_rk(trans_ub_nonag_rk: np.ndarray, trans_source_ag: dict, T_ag2nonag_reach_jk: np.ndarray) -> dict:
    """{(from_m, from_j): bool (local_r, to_k)} — the non-ag targets each ag source may transition to (ub > 0 and T_MAT-reachable)."""
    print('Getting feasible ag2nonag delta-var targets...', flush = True)
    feasible_nonag_rk = trans_ub_nonag_rk > 0
    return {
        (from_m, from_j): feasible_nonag_rk[source_cells, :] & T_ag2nonag_reach_jk[from_j][None, :]             # (ncells_src, N_NONAG)
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


# ═══════════════════════════ the column blocks: one Dataset per block, wide (grid) or long (arc) tables, block-LOCAL ids (get_cols shifts them) ═══════════════════════════

def ag_space(data: Data, feasible_ag_mrj: np.ndarray, trans_ub_ag_mrj: np.ndarray, dvar_base_ag_mrj: np.ndarray) -> xr.Dataset:
    """The ag columns as a wide table on (lm, lu, cell): a column where the cell may become that (lm, lu), its ub and base on the same grid; lb = 0."""
    # the (m, r, j) inputs laid out on the table's (m, j, r) grid, contiguous so downstream gathers read whole rows
    has_col  = np.ascontiguousarray(feasible_ag_mrj.transpose(0, 2, 1))
    ub_mjr   = np.ascontiguousarray(trans_ub_ag_mrj.transpose(0, 2, 1))
    base_mjr = np.ascontiguousarray(dvar_base_ag_mrj.transpose(0, 2, 1))

    col_mjr = np.full(has_col.shape, -1, dtype=np.int32)
    col_mjr[has_col] = np.arange(has_col.sum(), dtype=np.int32)

    return xr.Dataset(
        dict(col =(('lm', 'lu', 'cell'), col_mjr),      # column id (-1 = no column); col >= 0 is the select for gp.Var creation, grid order = column order
             ub  =(('lm', 'lu', 'cell'), ub_mjr),       # attribute: the transition upper bound (lb = 0 by default in gurobi)
             base=(('lm', 'lu', 'cell'), base_mjr)      # attribute: the node-balance constant (X = base + flow-in - flow-out); the flow-out sum has to be <= the base
        ),
        coords=dict(
            lm=list(data.LANDMANS),
            lu=list(data.AGRICULTURAL_LANDUSES),
            cell=np.arange(data.NCELLS)
        ),
        attrs=dict(n=int(has_col.sum()))
    )


def nonag_space(data: Data, trans_lb_nonag_rk: np.ndarray, trans_ub_nonag_rk: np.ndarray, dvar_base_nonag_rk: np.ndarray) -> xr.Dataset:
    """The non-ag columns as a wide table on (nonag_lu, cell): a column where ub > 0 and the land use is enabled, its lb / ub / base on the same grid."""
    # the (r, k) inputs laid out on the table's (k, r) grid, contiguous after the transpose
    lb_kr   = np.ascontiguousarray(trans_lb_nonag_rk.T)
    ub_kr   = np.ascontiguousarray(trans_ub_nonag_rk.T)
    base_kr = np.ascontiguousarray(dvar_base_nonag_rk.T)
    enabled = np.array([settings.NON_AG_LAND_USES[lu_name] for lu_name in data.NON_AGRICULTURAL_LANDUSES], dtype=bool)
    has_col = (ub_kr > 0) & enabled[:, None]

    col_kr = np.full(has_col.shape, -1, dtype=np.int32)
    col_kr[has_col] = np.arange(has_col.sum(), dtype=np.int32)

    return xr.Dataset(
        dict(col =(('nonag_lu', 'cell'), col_kr),        # column id (-1 = no column); col >= 0 is the select for gp.Var creation, grid order = column order
             lb  =(('nonag_lu', 'cell'), lb_kr),         # attribute: the transition lower bound
             ub  =(('nonag_lu', 'cell'), ub_kr),         # attribute: the transition upper bound (> 0 = feasible, for EVERY land use, enabled or not: the node-balance rows need the disabled ones too)
             base=(('nonag_lu', 'cell'), base_kr)        # attribute: the node-balance constant (X = base + flow-in - flow-out); the flow-out sum has to be <= the base
        ),
        coords=dict(
            nonag_lu=list(data.NON_AGRICULTURAL_LANDUSES),
            cell=np.arange(data.NCELLS)
        ),
        attrs=dict(n=int(has_col.sum()))
    )


def am_space(data: Data, ag_col_mjr: np.ndarray, mask_gbf2_solar: np.ndarray, mask_gbf2_wind: np.ndarray, trans_lb_ag_man_mrj: dict) -> xr.Dataset:
    """The ag-management columns as a wide table on (slot = (am, lu), lm, cell): a column where the ag column exists minus the renewable / savanna exclusions, its lb on the same grid."""
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
        dict(col=(('slot', 'lm', 'cell'), col_smr),                                # column id (-1 = no column); col >= 0 is the select for gp.Var creation, grid order = column order
             lb =(('slot', 'lm', 'cell'), lb_smr),                                 # attribute: the base-year adoption for non-reversible options, else 0 (ub = 1 for every column)
             j  =(('slot',), np.array([j for _, j in pairs], dtype=np.int32))      # the land-use code of each (am, lu) slot
        ),
        coords=coords,
        attrs=dict(
            n=int(has_col.sum()),
            agman2lu=data.AGMAN2LU,                                                    # {option: [land-use codes]}: the slot order
            savanna_eligible_r=np.flatnonzero(savanna_mask)                            # the solve read-back zeroes irr savanna columns outside these cells
        )
    )


def ag2ag_space(trans_source_ag: dict, feasible_ag2ag_mrj: dict) -> xr.Dataset:
    """The ag → ag arc columns as a long table on (arc): one row per feasible (from_m, from_j) → (to_m, to_j) transition at a cell, rows sorted by source in trans_source_ag order."""
    print('Building the ag2ag arc block...', flush = True)
    arc_rows = []                                            # one chunk of rows per source
    src_ptr = [0]                                            # where each source's run of rows starts / ends
    for (from_m, from_j), is_target_feasible in feasible_ag2ag_mrj.items():
        to_m, local_r, to_j = np.nonzero(is_target_feasible)
        cell = trans_source_ag[(from_m, from_j)][local_r]   # index of the cell in the global cell list
        arc_rows.append(np.column_stack([
            np.full(cell.size, from_m),
            np.full(cell.size, from_j),
            to_m,
            to_j,
            local_r,
            cell
        ]))
        
        src_ptr.append(src_ptr[-1] + cell.size)

    table = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 6), dtype=np.int32)   # one arc per row; the fields are its columns

    return xr.Dataset(
        dict(from_m =(('arc',), table[:, 0]),
             from_j =(('arc',), table[:, 1]),
             to_m   =(('arc',), table[:, 2]),
             to_j   =(('arc',), table[:, 3]),
             local_r=(('arc',), table[:, 4]),
             cell   =(('arc',), table[:, 5]),
             col    =(('arc',), np.arange(table.shape[0], dtype=np.int32))
        ),
        coords=dict(arc=np.arange(table.shape[0])),
        attrs=dict(n=table.shape[0], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_ag2ag_mrj))
    )


def ag2nonag_space(trans_source_ag: dict, feasible_ag2nonag_rk: dict) -> xr.Dataset:
    """The ag → non-ag arc columns as a long table on (arc): one row per feasible (from_m, from_j) → to_k transition at a cell, rows sorted by source in trans_source_ag order."""
    print('Building the ag2nonag arc block...', flush = True)
    arc_rows = []                                            # one chunk of rows per source
    src_ptr = [0]                                            # where each source's run of rows starts / ends
    for (from_m, from_j), is_target_feasible in feasible_ag2nonag_rk.items():
        local_r, to_k = np.nonzero(is_target_feasible)
        cell = trans_source_ag[(from_m, from_j)][local_r]   # index of the cell in the global cell list
        arc_rows.append(np.column_stack([
            np.full(cell.size, from_m),
            np.full(cell.size, from_j),
            to_k,
            local_r,
            cell
        ]))
        
        src_ptr.append(src_ptr[-1] + cell.size)

    table = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 5), dtype=np.int32)   # one arc per row; the fields are its columns

    return xr.Dataset(
        dict(from_m =(('arc',), table[:, 0]),
             from_j =(('arc',), table[:, 1]),
             to_k   =(('arc',), table[:, 2]),
             local_r=(('arc',), table[:, 3]),
             cell   =(('arc',), table[:, 4]),
             col    =(('arc',), np.arange(table.shape[0], dtype=np.int32))
        ),
        coords=dict(arc=np.arange(table.shape[0])),
        attrs=dict(n=table.shape[0], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_ag2nonag_rk))
    )


def nonag2ag_space(trans_source_nonag: dict, feasible_nonag2ag_mrj: dict) -> xr.Dataset:
    """The non-ag → ag arc columns as a long table on (arc): one row per feasible from_k → (to_m, to_j) transition at a cell, rows sorted by source in trans_source_nonag order."""
    print('Building the nonag2ag arc block...', flush = True)
    arc_rows = []                                            # one chunk of rows per source
    src_ptr = [0]                                            # where each source's run of rows starts / ends
    for from_k, is_target_feasible in feasible_nonag2ag_mrj.items():
        to_m, local_r, to_j = np.nonzero(is_target_feasible)
        cell = trans_source_nonag[from_k][local_r]   # index of the cell in the global cell list
        arc_rows.append(np.column_stack([
            np.full(cell.size, from_k),
            to_m,
            to_j,
            local_r,
            cell
        ]))
        
        src_ptr.append(src_ptr[-1] + cell.size)

    table = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 5), dtype=np.int32)   # one arc per row; the fields are its columns

    return xr.Dataset(
        dict(from_k =(('arc',), table[:, 0]),
             to_m   =(('arc',), table[:, 1]),
             to_j   =(('arc',), table[:, 2]),
             local_r=(('arc',), table[:, 3]),
             cell   =(('arc',), table[:, 4]),
             col    =(('arc',), np.arange(table.shape[0], dtype=np.int32))
        ),
        coords=dict(arc=np.arange(table.shape[0])),
        attrs=dict(n=table.shape[0], src_ptr=np.asarray(src_ptr, dtype=np.int64), sources=list(feasible_nonag2ag_mrj))
    )


def cell_usage_space(feasible_cell_usage_r) -> xr.Dataset:
    """The cell-usage slack columns as a wide table on (cell): a column per cell that gets a cell-usage row."""
    col_r = np.full(feasible_cell_usage_r.shape, -1, dtype=np.int32)
    col_r[feasible_cell_usage_r] = np.arange(feasible_cell_usage_r.sum(), dtype=np.int32)
    return xr.Dataset(
        dict(col=(('cell',), col_r)),                                          # slack column id (-1 = no row, no slack); col >= 0 is the select for gp.Var creation
        coords=dict(cell=np.arange(feasible_cell_usage_r.size)),
        attrs=dict(n=int(feasible_cell_usage_r.sum()))
    )


def table_space(blocks: dict, ag_mask_r: np.ndarray) -> xr.Dataset:
    """The whole space as ONE long table on (col = Var.index): the blocks' rows back to back in ``BLOCKS`` order, the fields of each column (-1 where n/a), its lb / ub / base."""
    ag, nonag, am, ag2ag, ag2nonag, nonag2ag, cell_usage = (blocks[name] for name in BLOCKS)

    # one dict of field arrays per block, in the block's column order (the grid order of a wide table, the row order of an arc table)
    m, j, r = np.nonzero(ag['col'].values >= 0)                                             # column order: lm, lu, cell
    ag_rows = dict(m=m, j=j, cell=r, ub=ag['ub'].values[m, j, r], base=ag['base'].values[m, j, r])
    k, r = np.nonzero(nonag['col'].values >= 0)                                             # column order: k, cell
    nonag_rows = dict(k=k, cell=r, lb=nonag['lb'].values[k, r], ub=nonag['ub'].values[k, r], base=nonag['base'].values[k, r])
    slot, m, r = np.nonzero(am['col'].values >= 0)                                          # column order: slot, lm, cell
    am_rows = dict(slot=slot, m=m, j=am['j'].values[slot], cell=r, lb=am['lb'].values[slot, m, r], ub=1.0)
    ag2ag_rows = dict(from_m=ag2ag['from_m'].values, from_j=ag2ag['from_j'].values, m=ag2ag['to_m'].values, j=ag2ag['to_j'].values,
                      local_r=ag2ag['local_r'].values, cell=ag2ag['cell'].values, ub=np.inf)
    ag2nonag_rows = dict(from_m=ag2nonag['from_m'].values, from_j=ag2nonag['from_j'].values, k=ag2nonag['to_k'].values,
                         local_r=ag2nonag['local_r'].values, cell=ag2nonag['cell'].values, ub=np.inf)
    nonag2ag_rows = dict(from_k=nonag2ag['from_k'].values, m=nonag2ag['to_m'].values, j=nonag2ag['to_j'].values,
                         local_r=nonag2ag['local_r'].values, cell=nonag2ag['cell'].values, ub=np.inf)
    slack_cells = np.flatnonzero(cell_usage['col'].values >= 0)
    band = 10 * settings.FEASIBILITY_TOLERANCE                                              # the cell-usage row is ranged: Σ shares ∈ [ag_mask − band, ag_mask + band]
    ag_mask = ag_mask_r[slack_cells].astype(np.float64)                                     # widened before the band is applied
    cell_usage_rows = dict(cell=slack_cells, ub=(ag_mask + band) - (ag_mask - band))        # the slack of a ranged row: lb 0, ub = hi − lo

    parts = [ag_rows, nonag_rows, am_rows, ag2ag_rows, ag2nonag_rows, nonag2ag_rows, cell_usage_rows]
    widths = [part['cell'].size for part in parts]
    block_ptr = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)                   # where each block's run of rows starts / ends
    for name, block, width, start in zip(BLOCKS, blocks.values(), widths, block_ptr):
        ids = block['col'].values
        assert block.attrs['n'] == width and np.array_equal(ids[ids >= 0], np.arange(start, start + width)), f'{name}: the ids must be its run of the table'

    def field(name, dtype, fill):
        """One field over the whole table: the block's array where it has the field, else the fill value."""
        return np.concatenate([np.broadcast_to(np.asarray(part.get(name, fill), dtype=dtype), width) for part, width in zip(parts, widths)])

    return xr.Dataset(
        dict(block  =(('col',), np.repeat(np.arange(len(BLOCKS), dtype=np.int8), widths)),  # the block code (attrs['blocks'] = names)
             m      =(('col',), field('m', np.int32, -1)),                                   # the ag (lm, lu) the column lands on: own (ag), host (am), TO fields (ag2ag, nonag2ag)
             j      =(('col',), field('j', np.int32, -1)),
             k      =(('col',), field('k', np.int32, -1)),                                   # the non-ag land use it lands on: own (nonag), TO field (ag2nonag)
             slot   =(('col',), field('slot', np.int32, -1)),                                # the (am, lu) slot of an ag-mgt column
             from_m =(('col',), field('from_m', np.int32, -1)),                              # where an arc comes from
             from_j =(('col',), field('from_j', np.int32, -1)),
             from_k =(('col',), field('from_k', np.int32, -1)),
             local_r=(('col',), field('local_r', np.int32, -1)),                             # the arc's cell in its source's cell list
             cell   =(('col',), field('cell', np.int32, -1)),                                # the cell (every column has one)
             lb     =(('col',), field('lb', np.float64, 0.0)),                               # the bounds of the column (gurobi stores double)
             ub     =(('col',), field('ub', np.float64, np.inf)),
             base   =(('col',), field('base', np.float32, 0.0))                              # the node-balance constant of an ag / non-ag column
        ),
        coords=dict(col=np.arange(block_ptr[-1])),
        attrs=dict(blocks=list(BLOCKS), block_ptr=block_ptr,
                   n_dec=int(block_ptr[-2]),                                                 # the decision columns end with the last nonag2ag arc: the objective is built at this width (the slacks carry no cost)
                   n_all=int(block_ptr[-1]))                                                 # every column: the rows are built at this width
    )


# ═══════════════════════════ get_cols: the column space of one step ═══════════════════════════

def get_cols(data: Data, base_year: int) -> dict:
    """The column space of one solve step: every unknown as a labelled Dataset per block (wide grid or long arc table) holding its actual Var.index ids, the whole space as one long table, plus the widths (layout), terms, sources and masks."""

    # ── 1. sources (FROM-view): the base-year holders of land ──
    trans_source_ag         = get_trans_source_ag(data, base_year)              # (from_m, from_j): global cell indices
    trans_source_nonag      = get_trans_source_nonag(data, base_year)           # from_k: global cell indices

    # ── 2. transition bounds and the base (TO-view) ──
    trans_ub_ag_mrj         = get_trans_ub_ag_mrj(data, base_year)              # upper bound of every ag target (ag2ag + nonag2ag), raised to the base; ag has no lower bound
    trans_ub_nonag_rk       = get_trans_ub_nonag_rk(data, base_year)            # upper bound of every non-ag target, raised to the base
    trans_lb_nonag_rk       = get_trans_lb_nonag_rk(data, base_year)            # lower bound of every non-ag target: irreversible non-ag land uses lock in their base-year holding
    trans_lb_ag_man_mrj     = get_trans_lb_ag_man_mrj(data, base_year)          # lower bound of every ag-mgt entry: non-reversible options lock in last step's adoption

    dvar_base_ag_mrj        = tools.clamp_dvar_bound(data.ag_dvars[base_year], 0.0, trans_ub_ag_mrj, 'Ag base clipped to [0,ub]')
    dvar_base_nonag_rk      = tools.clamp_dvar_bound(data.non_ag_dvars[base_year], trans_lb_nonag_rk, trans_ub_nonag_rk, 'NonAg base clipped to [lb,ub]')

    # ── 3. feasibility: target entries (feasible_ag_mrj, trans_ub_nonag_rk > 0), transition arcs, cell-usage rows ──
    T_ag2ag_reach_jj        = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.AGRICULTURAL_LANDUSES).values)       # T_MAT reachability: finite = the transition is allowed
    T_ag2nonag_reach_jk     = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.NON_AGRICULTURAL_LANDUSES).values)
    T_nonag2ag_reach_kj     = ~np.isnan(data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values)

    feasible_ag_mrj         = get_feasible_ag_mrj(data, base_year)                # bool: which (m, j) a cell may become
    feasible_ag2ag_mrj      = get_feasible_ag2ag_mrj(feasible_ag_mrj, trans_source_ag, T_ag2ag_reach_jj)
    feasible_nonag2ag_mrj   = get_feasible_nonag2ag_mrj(feasible_ag_mrj, trans_source_nonag, T_nonag2ag_reach_kj)
    feasible_ag2nonag_rk    = get_feasible_ag2nonag_rk(trans_ub_nonag_rk, trans_source_ag, T_ag2nonag_reach_jk)
    feasible_cell_usage_r   = get_feasible_cell_usage_r(trans_ub_ag_mrj, trans_ub_nonag_rk, data.AG_MASK_PROPORTION_R)

    # ── 4. masks: the cell sets that restrict ag-management options ──
    mask_gbf2_solar         = get_mask_gbf2_solar(data)
    mask_gbf2_wind          = get_mask_gbf2_wind(data)
    mask_mnes_solar         = get_mask_mnes_solar(data)
    mask_mnes_wind          = get_mask_mnes_wind(data)

    # ── 5. the blocks in Var.index order, each built with block-local ids ──
    ag = ag_space(data, feasible_ag_mrj, trans_ub_ag_mrj, dvar_base_ag_mrj)
    cols = {
        'ag':         ag,
        'nonag':      nonag_space(data, trans_lb_nonag_rk, trans_ub_nonag_rk, dvar_base_nonag_rk),
        'am':         am_space(data, ag['col'].values, mask_gbf2_solar, mask_gbf2_wind, trans_lb_ag_man_mrj),
        'ag2ag':      ag2ag_space(trans_source_ag, feasible_ag2ag_mrj),
        'ag2nonag':   ag2nonag_space(trans_source_ag, feasible_ag2nonag_rk),
        'nonag2ag':   nonag2ag_space(trans_source_nonag, feasible_nonag2ag_mrj),
        'cell_usage': cell_usage_space(feasible_cell_usage_r),
    }

    # ── 6. the blocks placed back to back: per block, its ids shifted in place to the actual Var.index (-1 never shifts),
    #       then the running count advanced by the block's width ──
    n_cols = 0                                                                      # the next free Var.index
    for block in cols.values():
        ids = block['col'].values
        ids[ids >= 0] += n_cols
        n_cols += block.attrs['n']

    # ── 7. the table: the whole space as one long table in Var.index order, the block bounds and widths as its attrs ──
    table = table_space(cols, data.AG_MASK_PROPORTION_R)
    layout = dict(n_dec=table.attrs['n_dec'], n_all=table.attrs['n_all'])

    print(f"Column space: {layout['n_all']:,} columns = {layout['n_dec']:,} decision (n_dec) + {cols['cell_usage'].attrs['n']:,} cell-usage slacks", flush=True)
    for block in cols:
        print(f"{'└──' if block == 'cell_usage' else '├──'} {block:<10s} {cols[block].attrs['n']:>12,}", flush=True)

    # ── 8. terms: the table's ag / ag-mgt / non-ag rows — one row per column with its attributes (m, j, r, ...) and its col —
    #       read by every policy family (row_builder.gather_coeffs / compose_rows) and by the objective (row_builder.get_obj_block) ──
    ag_rows = block_slice(table, 'ag')
    ag_terms = dict(
        m=table['m'].values[ag_rows],
        j=table['j'].values[ag_rows],
        r=table['cell'].values[ag_rows],
        col=table['col'].values[ag_rows].astype(np.int32)
    )

    am = cols['am']
    am_rows = block_slice(table, 'am')
    am_slot = table['slot'].values[am_rows]
    am_list = list(am.attrs['agman2lu'])
    am_idx_of_slot = np.array([am_list.index(name) for name in am['am'].values], dtype=np.int32)            # slot -> index into am_list
    j_idx_of_slot = np.zeros(am.sizes['slot'], dtype=np.int32)                                              # slot -> position of its land use within the option
    for am_idx in range(len(am_list)):
        slots_of_option = np.flatnonzero(am_idx_of_slot == am_idx)
        j_idx_of_slot[slots_of_option] = np.arange(slots_of_option.size, dtype=np.int32)
    am_terms = dict(
        am_idx=am_idx_of_slot[am_slot],
        j_idx=j_idx_of_slot[am_slot],
        j=table['j'].values[am_rows],
        m=table['m'].values[am_rows],
        r=table['cell'].values[am_rows],
        col=table['col'].values[am_rows].astype(np.int32)
    )

    nonag_rows = block_slice(table, 'nonag')
    nonag_terms = dict(k=table['k'].values[nonag_rows], r=table['cell'].values[nonag_rows], col=table['col'].values[nonag_rows].astype(np.int32))
    term_cell = np.concatenate([ag_terms['r'], am_terms['r'], nonag_terms['r']]).astype(np.int32)           # the row order of the terms table: ag | am | nonag
    term_col = np.concatenate([ag_terms['col'], am_terms['col'], nonag_terms['col']]).astype(np.int32)
    by_cell_order = np.argsort(term_cell, kind='stable')                                                    # the terms table re-sorted by cell ...
    by_cell_ptr = np.searchsorted(term_cell[by_cell_order], np.arange(data.NCELLS + 1))                     # ... and where each cell's run of rows starts / ends

    # ── 9. what the rows and the post-solve read need besides the columns ──
    cols['table']    = table
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
