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
    """{(from_m, from_j): {'cells': the source's cells, 'feasible': bool (to_m, local_r, to_j)}} — the ag targets each ag source may transition to (feasible, T_MAT-reachable, not itself); ``local_r`` indexes ``cells``."""
    print('Getting feasible ag2ag delta-var targets...', flush = True)
    feasible_targets = {}
    for (from_m, from_j), source_cells in trans_source_ag.items():
        is_target_feasible = feasible_ag_mrj[:, source_cells, :] & T_ag2ag_reach_jj[from_j][None, None, :]      # (NLMS, ncells_src, N_AG)
        is_target_feasible[from_m, :, from_j] = False                                                           # staying is not a transition
        feasible_targets[(from_m, from_j)] = dict(cells=source_cells, feasible=is_target_feasible)
    return feasible_targets

def get_feasible_nonag2ag_mrj(feasible_ag_mrj: np.ndarray, trans_source_nonag: dict, T_nonag2ag_reach_kj: np.ndarray) -> dict:
    """{from_k: {'cells': the source's cells, 'feasible': bool (to_m, local_r, to_j)}} — the ag targets each non-ag source may transition to (feasible and T_MAT-reachable); ``local_r`` indexes ``cells``."""
    print('Getting feasible nonag2ag delta-var targets...', flush = True)
    return {
        from_k: dict(
            cells=source_cells,
            feasible=feasible_ag_mrj[:, source_cells, :] & T_nonag2ag_reach_kj[from_k][None, None, :]
        ) # (NLMS, ncells_k, N_AG)
        for from_k, source_cells in trans_source_nonag.items()
    }

def get_feasible_ag2nonag_rk(trans_ub_nonag_rk: np.ndarray, trans_source_ag: dict, T_ag2nonag_reach_jk: np.ndarray) -> dict:
    """{(from_m, from_j): {'cells': the source's cells, 'feasible': bool (local_r, to_k)}} — the non-ag targets each ag source may transition to (ub > 0 and T_MAT-reachable); ``local_r`` indexes ``cells``."""
    print('Getting feasible ag2nonag delta-var targets...', flush = True)
    feasible_nonag_rk = trans_ub_nonag_rk > 0
    return {
        (from_m, from_j): dict(
            cells=source_cells,
            feasible=feasible_nonag_rk[source_cells, :] & T_ag2nonag_reach_jk[from_j][None, :]
        )  # (ncells_src, N_NONAG)
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

def ag_space(data: Data, feasible_ag_mrj: np.ndarray, trans_ub_ag_mrj: np.ndarray, dvar_base_ag_mrj: np.ndarray) -> tuple[xr.Dataset, dict]:
    """The ag columns: the wide id grid on (lm, lu, cell) the model indexes by (m, j, r), and the block's rows — its (lm, lu), cell, ub and base; lb = 0."""
    # the (m, r, j) inputs laid out on the table's (m, j, r) grid, contiguous so downstream gathers read whole rows
    has_col  = np.ascontiguousarray(feasible_ag_mrj.transpose(0, 2, 1))
    ub_mjr   = np.ascontiguousarray(trans_ub_ag_mrj.transpose(0, 2, 1))
    base_mjr = np.ascontiguousarray(dvar_base_ag_mrj.transpose(0, 2, 1))

    col_mjr = np.full(has_col.shape, -1, dtype=np.int32)
    col_mjr[has_col] = np.arange(has_col.sum(), dtype=np.int32)                    # grid order = column order

    # the column (gp.Var) view: where the ag exists
    m, j, r = np.nonzero(has_col)                                                  # the block's columns, in the order the ids were given
    rows = dict(
        m=m,
        j=j,
        cell=r,
        ub=ub_mjr[m, j, r],                                                        # attribute: the transition upper bound (lb = 0 by default in gurobi)
        base=base_mjr[m, j, r]                                                     # attribute: the node-balance constant (X = base + flow-in - flow-out)
    )
    
    # the row (constraint) view: cells of -1 are skipped since they are not attached to any gp.Var
    grid = xr.Dataset(
        dict(col =(('lm', 'lu', 'cell'), col_mjr),      # column id (-1 = no column); col >= 0 is the select for gp.Var creation, grid order = column order
             base=(('lm', 'lu', 'cell'), base_mjr)      # the source-cap rows read the base by (from_m, from_j, cell)
        ),
        coords=dict(
            lm=list(data.LANDMANS),
            lu=list(data.AGRICULTURAL_LANDUSES),
            cell=np.arange(data.NCELLS)
        )
    )
    return grid, rows


def nonag_space(data: Data, trans_lb_nonag_rk: np.ndarray, trans_ub_nonag_rk: np.ndarray, dvar_base_nonag_rk: np.ndarray) -> tuple[xr.Dataset, dict]:
    """The non-ag columns: the wide id grid on (nonag_lu, cell) the model indexes by (k, r), and the block's rows — its land use, cell, lb, ub and base."""
    # the (r, k) inputs laid out on the table's (k, r) grid, contiguous after the transpose
    lb_kr   = np.ascontiguousarray(trans_lb_nonag_rk.T)
    ub_kr   = np.ascontiguousarray(trans_ub_nonag_rk.T)
    base_kr = np.ascontiguousarray(dvar_base_nonag_rk.T)
    enabled = np.array([settings.NON_AG_LAND_USES[lu_name] for lu_name in data.NON_AGRICULTURAL_LANDUSES], dtype=bool)
    has_col = (ub_kr > 0) & enabled[:, None]

    col_kr = np.full(has_col.shape, -1, dtype=np.int32)
    col_kr[has_col] = np.arange(has_col.sum(), dtype=np.int32)                     # grid order = column order

    # the column (gp.Var) view: where the non-ag land use exists
    k, r = np.nonzero(has_col)                                                     # the block's columns, in the order the ids were given
    rows = dict(
        k=k,
        cell=r,
        lb=lb_kr[k, r],                                                            # attribute: the transition lower bound
        ub=ub_kr[k, r],                                                            # attribute: the transition upper bound
        base=base_kr[k, r]                                                         # attribute: the node-balance constant (X = base + flow-in - flow-out)
    )

    # the row (constraint) view: cells of -1 are skipped since they are not attached to any gp.Var
    grid = xr.Dataset(
        dict(col =(('nonag_lu', 'cell'), col_kr),        # column id (-1 = no column); col >= 0 is the select for gp.Var creation, grid order = column order
             ub  =(('nonag_lu', 'cell'), ub_kr),         # > 0 = feasible, for EVERY land use, enabled or not: the node-balance rows need the disabled ones too
             base=(('nonag_lu', 'cell'), base_kr)        # the source-cap rows read the base by (from_k, cell)
        ),
        coords=dict(
            nonag_lu=list(data.NON_AGRICULTURAL_LANDUSES),
            cell=np.arange(data.NCELLS)
        )
    )
    return grid, rows


def am_space(data: Data, ag_col_mjr: np.ndarray, mask_gbf2_solar: np.ndarray, mask_gbf2_wind: np.ndarray, trans_lb_ag_man_mrj: dict) -> tuple[xr.Dataset, dict]:
    """The ag-management columns: the wide id grid on (slot = (am, lu), lm, cell), and the block's rows — its slot, option, land use, the ag (lm, lu) it sits on, cell and lb; ub = 1."""
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
    col_smr[has_col] = np.arange(has_col.sum(), dtype=np.int32)                        # grid order = column order

    j_of_slot      = np.array([j for _, j in pairs], dtype=np.int32)                   # the land-use code of each (am, lu) slot
    am_idx_of_slot = np.array([option for option, lu_codes in enumerate(data.AGMAN2LU.values()) for _ in lu_codes], dtype=np.int32)     # the slot's option, as an index into agman2lu
    j_idx_of_slot  = np.array([position for lu_codes in data.AGMAN2LU.values() for position in range(len(lu_codes))], dtype=np.int32)   # the land use's position within its option: the last axis of the per-option effect arrays [m, r, j_idx]

    # the column (gp.Var) view: where the ag-mgt slot exists
    slot, m, r = np.nonzero(has_col)                                                   # the block's columns, in the order the ids were given
    rows = dict(
        slot=slot,
        am_idx=am_idx_of_slot[slot],
        j_idx=j_idx_of_slot[slot],
        m=m,
        j=j_of_slot[slot],
        cell=r,
        lb=lb_smr[slot, m, r],                                                         # attribute: the base-year adoption for non-reversible options, else 0
        ub=1.0
    )

    slot_index = pd.MultiIndex.from_arrays([[am for am, _ in pairs], [data.AGRICULTURAL_LANDUSES[j] for _, j in pairs]], names=['am', 'lu'])
    coords = xr.Coordinates.from_pandas_multiindex(slot_index, 'slot').assign(lm=list(data.LANDMANS), cell=np.arange(data.NCELLS))

    # the row (constraint) view: cells of -1 are skipped since they are not attached to any gp.Var
    grid = xr.Dataset(
        dict(col=(('slot', 'lm', 'cell'), col_smr),                                # column id (-1 = no column); col >= 0 is the select for gp.Var creation, grid order = column order
             j  =(('slot',), j_of_slot)                                            # the land-use code of each (am, lu) slot
        ),
        coords=coords,
        attrs=dict(
            agman2lu=data.AGMAN2LU,                                                    # {option: [land-use codes]}: the slot order
            savanna_eligible_r=np.flatnonzero(savanna_mask)                            # the solve read-back zeroes irr savanna columns outside these cells
        )
    )
    return grid, rows


def ag2ag_space(feasible_ag2ag_mrj: dict) -> tuple[dict, np.ndarray]:
    """The ag → ag arc columns as rows: one per feasible (from_m, from_j) → (to_m, to_j) transition at a cell, sorted by source in feasibility order; with the group bounds of each source's run."""
    print('Building the ag2ag arc block...', flush = True)
    arc_rows = []                                            # one chunk of rows per source
    src_ptr = [0]                                            # where each source's run of rows starts / ends
    for (from_m, from_j), source in feasible_ag2ag_mrj.items():
        to_m, local_r, to_j = np.nonzero(source['feasible'])
        cell = source['cells'][local_r]                      # index of the cell in the global cell list
        arc_rows.append(np.column_stack([
            np.full(cell.size, from_m),
            np.full(cell.size, from_j),
            to_m,
            to_j,
            local_r,
            cell
        ]))

        src_ptr.append(src_ptr[-1] + cell.size)

    arcs = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 6), dtype=np.int32)   # one arc per row; the fields are its columns

    # the column (gp.Var) view: one arc per feasible transition; no row view, nothing looks an arc up by its coordinates
    rows = dict(
        from_m =arcs[:, 0],
        from_j =arcs[:, 1],
        m      =arcs[:, 2],                                  # the ag (lm, lu) the arc lands on
        j      =arcs[:, 3],
        local_r=arcs[:, 4],                                  # the arc's cell in its SOURCE's frame: the per-source cost / GHG arrays are only ncells_src tall, and solve() scatters the arc's value back at [to_m, local_r, to_j]
        cell   =arcs[:, 5],                                  # the same cell in the global frame, which the demand / GHG / water / biodiversity rows weight by
        ub     =np.inf
    )
    return rows, np.asarray(src_ptr, dtype=np.int64)


def ag2nonag_space(feasible_ag2nonag_rk: dict) -> tuple[dict, np.ndarray]:
    """The ag → non-ag arc columns as rows: one per feasible (from_m, from_j) → to_k transition at a cell, sorted by source in feasibility order; with the group bounds of each source's run."""
    print('Building the ag2nonag arc block...', flush = True)
    arc_rows = []                                            # one chunk of rows per source
    src_ptr = [0]                                            # where each source's run of rows starts / ends
    for (from_m, from_j), source in feasible_ag2nonag_rk.items():
        local_r, to_k = np.nonzero(source['feasible'])
        cell = source['cells'][local_r]                      # index of the cell in the global cell list
        arc_rows.append(np.column_stack([
            np.full(cell.size, from_m),
            np.full(cell.size, from_j),
            to_k,
            local_r,
            cell
        ]))

        src_ptr.append(src_ptr[-1] + cell.size)

    arcs = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 5), dtype=np.int32)   # one arc per row; the fields are its columns

    # the column (gp.Var) view: one arc per feasible transition; no row view, nothing looks an arc up by its coordinates
    rows = dict(
        from_m =arcs[:, 0],
        from_j =arcs[:, 1],
        k      =arcs[:, 2],                                  # the non-ag land use the arc lands on
        local_r=arcs[:, 3],                                  # the arc's cell in its SOURCE's frame: the per-source cost arrays are only ncells_src tall, and solve() scatters the arc's value back at [local_r, to_k]
        cell   =arcs[:, 4],                                  # the same cell in the global frame, which the demand / GHG / water / biodiversity rows weight by
        ub     =np.inf
    )
    return rows, np.asarray(src_ptr, dtype=np.int64)


def nonag2ag_space(feasible_nonag2ag_mrj: dict) -> tuple[dict, np.ndarray]:
    """The non-ag → ag arc columns as rows: one per feasible from_k → (to_m, to_j) transition at a cell, sorted by source in feasibility order; with the group bounds of each source's run."""
    print('Building the nonag2ag arc block...', flush = True)
    arc_rows = []                                            # one chunk of rows per source
    src_ptr = [0]                                            # where each source's run of rows starts / ends
    for from_k, source in feasible_nonag2ag_mrj.items():
        to_m, local_r, to_j = np.nonzero(source['feasible'])
        cell = source['cells'][local_r]                      # index of the cell in the global cell list
        arc_rows.append(np.column_stack([
            np.full(cell.size, from_k),
            to_m,
            to_j,
            local_r,
            cell
        ]))

        src_ptr.append(src_ptr[-1] + cell.size)

    arcs = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 5), dtype=np.int32)   # one arc per row; the fields are its columns

    # the column (gp.Var) view: one arc per feasible transition; no row view, nothing looks an arc up by its coordinates
    rows = dict(
        from_k =arcs[:, 0],
        m      =arcs[:, 1],                                  # the ag (lm, lu) the arc lands on
        j      =arcs[:, 2],
        local_r=arcs[:, 3],                                  # the arc's cell in its SOURCE's frame: the per-source cost arrays are only ncells_k tall, and solve() scatters the arc's value back at [to_m, local_r, to_j]
        cell   =arcs[:, 4],                                  # the same cell in the global frame, which the demand / GHG / water / biodiversity rows weight by
        ub     =np.inf
    )
    return rows, np.asarray(src_ptr, dtype=np.int64)


def cell_usage_space(feasible_cell_usage_r: np.ndarray, ag_mask_r: np.ndarray) -> dict:
    """The cell-usage slack columns as rows: one per cell that gets a cell-usage row, its ub the width of that ranged row."""
    slack_cells = np.flatnonzero(feasible_cell_usage_r)
    band = 10 * settings.FEASIBILITY_TOLERANCE                                              # the cell-usage row is ranged: Σ shares ∈ [ag_mask − band, ag_mask + band]
    ag_mask = ag_mask_r[slack_cells].astype(np.float64)                                     # widened before the band is applied
    # the column (gp.Var) view: one slack per cell that gets a cell-usage row; no row view, the row finds its slack through the table
    return dict(
        cell=slack_cells,                                                                   # the slack of a ranged row: lb 0, ub = hi − lo
        ub=(ag_mask + band) - (ag_mask - band)
    )


def table_space(data: Data, blocks: dict, src_ptr: dict) -> xr.Dataset:
    """The whole space as ONE long table on (col = Var.index): the blocks' rows back to back, group after group in the order ``blocks`` declares them, the fields of each column (-1 where n/a), its lb / ub / base."""

    # the groups laid out one after another: their declared order IS the table's, so the widths below are its prefixes
    parts = {block: rows for group in blocks.values() for block, rows in group.items()}
    group_width = {group: sum(rows['cell'].size for rows in group_blocks.values()) for group, group_blocks in blocks.items()}

    # each block's run of the table: the rows [start, stop) it owns
    widths = [part['cell'].size for part in parts.values()]
    bounds = np.cumsum([0, *widths])
    block_range = {block: (int(start), int(stop)) for block, start, stop in zip(parts, bounds[:-1], bounds[1:])}

    n_all   = int(bounds[-1])                                    # every column: the rows are built at this width
    n_terms = group_width['scored']                              # the scored group leads, so its width IS the prefix a demand / GHG / water / biodiversity / renewable coefficient array is allocated at
    n_dec   = n_all - group_width['slack']                       # the slack group trails, so the objective stops where it starts

    # Ag block has no slot/am_idx/j_idx, fill -1 for those fields. Do the same for other blocks.
    def field(field_name, dtype, fill):
        """One field over the whole table: each block's array for it, or the fill where the block has no such field."""
        per_block = []
        for part, width in zip(parts.values(), widths):
            value = part.get(field_name, fill)                   # the block's own array, or the fill where the field does not apply to it
            value = np.asarray(value, dtype=dtype)               # one dtype for the whole field, whatever each block happened to store
            per_block.append(np.broadcast_to(value, width))      # a scalar stretches to the block's width (the fill, or a constant like ub = 1.0)
        return np.concatenate(per_block)                         # the blocks back to back: one value per column of the table

    # the scored columns (the ones demand, GHG, water, biodiversity and renewable rows multiply) re-sorted by cell, 
    # and where each cell's run starts / ends
    cell = field('cell', np.int32, -1)
    by_cell_order = np.argsort(cell[:n_terms], kind='stable')
    by_cell_ptr = np.searchsorted(cell[:n_terms][by_cell_order], np.arange(data.NCELLS + 1))

    return xr.Dataset(
        dict(m      =(('col',), field('m', np.int32, -1)),                                   # the ag (lm, lu) the column lands on: own (ag), host (am), TO fields (ag2ag, nonag2ag)
             j      =(('col',), field('j', np.int32, -1)),
             k      =(('col',), field('k', np.int32, -1)),                                   # the non-ag land use it lands on: own (nonag), TO field (ag2nonag)
             slot   =(('col',), field('slot', np.int32, -1)),                                # the (am, lu) slot of an ag-mgt column ...
             am_idx =(('col',), field('am_idx', np.int32, -1)),                              # ... its option (attrs['options'][am_idx] is the name) ...
             j_idx  =(('col',), field('j_idx', np.int32, -1)),                               # ... and its land use's position within the option (the last axis of the per-option effect arrays)
             from_m =(('col',), field('from_m', np.int32, -1)),                              # where an arc comes from
             from_j =(('col',), field('from_j', np.int32, -1)),
             from_k =(('col',), field('from_k', np.int32, -1)),
             local_r=(('col',), field('local_r', np.int32, -1)),                             # ... and the arc's cell in that SOURCE's frame (-1 off the arc blocks: only an arc lives in a source frame), where its cost / GHG coefficients are stored and its solved value is scattered back
             cell   =(('col',), cell),                                                       # the cell in the GLOBAL frame — every column has one, and the demand / GHG / water / biodiversity / renewable rows weight by it
             lb     =(('col',), field('lb', np.float64, 0.0)),                               # the bounds of the column (gurobi stores double)
             ub     =(('col',), field('ub', np.float64, np.inf)),
             base   =(('col',), field('base', np.float32, 0.0))                              # the node-balance constant of an ag / non-ag column
        ),
        attrs=dict(block_range=block_range,                                                  # {block: (start, stop)} — the rows each block owns, in the table's block order
                   options=list(data.AGMAN2LU),                                              # the ag-management options, in am_idx order
                   n_terms=n_terms,                                                          # the scored group: the table's first rows, the width a demand / GHG / water / biodiversity / renewable coefficient array is allocated at
                   n_dec=n_dec,                                                              # everything before the slack group: the objective is built at this width (a slack carries no cost)
                   n_all=n_all,                                                              # every column: the rows are built at this width
                   by_cell_order=by_cell_order, 
                   by_cell_ptr=by_cell_ptr,
                   src_ptr={name: block_range[name][0] + ptr for name, ptr in src_ptr.items()})    # per arc block, the group bounds of its rows sorted by source, as table rows
    )


# ═══════════════════════════ get_cols: the column space of one step ═══════════════════════════

def get_cols(data: Data, base_year: int) -> dict:
    """The column space of one solve step: every unknown as one row of the long table (``table``), the wide id grids of ag / nonag / am holding their actual Var.index ids, the sources and the masks."""

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

    # ── 5. the blocks and the table: every block's rows, plus the wide grids the model indexes, laid back to back in Var.index order ──
    ag_grid,    ag_rows     = ag_space(data, feasible_ag_mrj, trans_ub_ag_mrj, dvar_base_ag_mrj)
    nonag_grid, nonag_rows  = nonag_space(data, trans_lb_nonag_rk, trans_ub_nonag_rk, dvar_base_nonag_rk)
    am_grid,    am_rows     = am_space(data, ag_grid['col'].values, mask_gbf2_solar, mask_gbf2_wind, trans_lb_ag_man_mrj)

    ag2ag_rows,    ag2ag_src_ptr    = ag2ag_space(feasible_ag2ag_mrj)                # each source carries its own cells: local_r -> the global cell
    ag2nonag_rows, ag2nonag_src_ptr = ag2nonag_space(feasible_ag2nonag_rk)
    nonag2ag_rows, nonag2ag_src_ptr = nonag2ag_space(feasible_nonag2ag_mrj)
    
    cell_usage_rows                 = cell_usage_space(feasible_cell_usage_r, data.AG_MASK_PROPORTION_R)

    # the blocks, grouped by what they are FOR: the groups go into the table in this order, and n_terms / n_dec
    # are the group widths — move a block to another group and every count downstream follows it
    blocks = {
        'scored': dict(ag=ag_rows, nonag=nonag_rows, am=am_rows),                                   # the demand, GHG, water, biodiversity and renewable rows multiply these per cell (the first n_terms columns)
        'arcs':   dict(ag2ag=ag2ag_rows, ag2nonag=ag2nonag_rows, nonag2ag=nonag2ag_rows),           # charged per arc (transition cost in the objective, transition emissions in the GHG row), never per cell
        'slack':  dict(cell_usage=cell_usage_rows),                                                 # no cost at all: the objective stops where they start (n_dec)
    }

    # the group bounds of each arc block's rows, sorted by source, block-local: table_space raises them to table rows
    src_ptr = dict(ag2ag=ag2ag_src_ptr, ag2nonag=ag2nonag_src_ptr, nonag2ag=nonag2ag_src_ptr)

    table = table_space(data, blocks, src_ptr)                                      # the block bounds and widths land in its attrs

    # ── 6. the grids' ids raised in place to the actual Var.index (-1 never shifts): each grid's block is one run of the table ──
    for name, grid in (('ag', ag_grid), ('nonag', nonag_grid), ('am', am_grid)):
        start, stop = table.attrs['block_range'][name]
        ids = grid['col'].values
        ids[ids >= 0] += start
        assert np.array_equal(ids[ids >= 0], np.arange(start, stop)), f'{name}: the grid ids must be its run of the table'

    block_range = table.attrs['block_range']
    print(f"Column space: {table.attrs['n_all']:,} columns = {table.attrs['n_dec']:,} decision (n_dec) + "
          f"{table.attrs['n_all'] - table.attrs['n_dec']:,} cell-usage slacks", flush=True)
    for name, (start, stop) in block_range.items():
        print(f"{'└──' if name == list(block_range)[-1] else '├──'} {name:<10s} {stop - start:>12,}", flush=True)

    # ── 7. the space: the table, the wide grids the rows still index by (m, j, r) / (k, r) / (slot, m, r), the sources, the masks ──
    return dict(
        table=table,
        ag=ag_grid,
        nonag=nonag_grid,
        am=am_grid,
        sources=dict(ag=trans_source_ag, nonag=trans_source_nonag),
        mask_gbf2_solar=mask_gbf2_solar,
        mask_gbf2_wind=mask_gbf2_wind,
        mask_mnes_solar=mask_mnes_solar,
        mask_mnes_wind=mask_mnes_wind,
    )
