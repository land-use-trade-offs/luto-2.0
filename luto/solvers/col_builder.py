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

import luto.settings as settings
import luto.tools as tools
import luto.economics.agricultural.transitions as ag_transition
import luto.economics.non_agricultural.transitions as non_ag_transition

from luto.data import Data
from luto.solvers.row_inputs import get_mask_gbf2_solar, get_mask_gbf2_wind

# ═══════════════════════════ get_cols: the column space of one step ═══════════════════════════

@dataclass
class ColSupport:
    """Supporting data that maps the SPARSE column variables onto the DENSE input arrays, so the row side never infers 
    an index: every subset comes as a PAIR of handles — one slices the input (``valid_*``, ``region2cell``), its twin 
    slices the column table (``block_range`` / ``*_range``, ``region2col``) — and a constraint is input[handle] · x[twin].
    Beside them, the same pairs vectorised for the rows that need them at once: ``cell2col`` (every cell's variables,
    for the layer families) and ``ag_mrj2col`` / ``nonag_rk2col`` (the masks read backwards, position → column, for the
    rows that join variables by node)."""

    valid_ag_mrj: np.ndarray      # filter the INPUT to the ag entries that have a column, as a 1-D vector in column order     
    valid_nonag_rk: np.ndarray    # filter the INPUT to the non-ag entries that have a column, as a 1-D vector in column order 
    valid_am: dict                # filter an option's INPUT at one land use and one lm to the cells that have a column, per (option, j_idx, m)
    valid_ag2ag: dict             # filter a source's ag→ag INPUT to the arcs that have a column, per source                   
    valid_ag2nonag: dict          # filter a source's ag→non-ag INPUT to the arcs that have a column, per source               
    valid_nonag2ag: dict          # filter a source's non-ag→ag INPUT to the arcs that have a column, per source               
    region2cell: xr.Dataset       # filter the INPUT by region
    region2col: xr.Dataset        # filter the gp.Vars table by region
    # the three below are read by get_rows only; simulation frees them before the solve
    cell2col: sparse.csr_matrix   # (cell x col), 1 where a variable (ag/nonag/am/ag2ag/ag2nonag/nonag2ag) sits in the cell: for each cell, the variables in it
    ag_mrj2col: np.ndarray        # (lm, cell, lu) int32: for each ag position, the column index of its variable, -1 where it has none (no arcs, no am)
    nonag_rk2col: np.ndarray      # (cell, nonag_lu) int32: for each non-ag position, the column index of its variable, -1 where it has none


def get_cols(data: Data, base_year: int) -> tuple[xr.Dataset, ColSupport]:
    """The column space of one solve step: every unknown as one row of the long table ``cols`` (on ``col`` =
    Var.index), and beside it the ``ColSupport``: the handles the row side slices the inputs and the table with."""

    # ── 1. sources (FROM-view): the base-year holders of land ──
    trans_source_ag         = ag_transition.get_base_dvar_mj_cell_map(data, base_year)          # (from_m, from_j): global cell indices
    trans_source_nonag      = non_ag_transition.get_base_nonag_dvar_k_cell_map(data, base_year) # from_k: global cell indices

    # ── 2. transition bounds and the base (TO-view) ──
    trans_ub_ag_mrj         = get_trans_ub_ag_mrj(data, base_year)                              # upper bound of every ag target (ag2ag + nonag2ag), raised to the base; ag has no lower bound
    trans_ub_nonag_rk       = get_trans_ub_nonag_rk(data, base_year)                            # upper bound of every non-ag target, raised to the base
    trans_lb_nonag_rk       = get_trans_lb_nonag_rk(data, base_year)                            # lower bound of every non-ag target: irreversible non-ag land uses lock in their base-year holding
    trans_lb_ag_man_mrj     = ag_transition.get_lower_bound_agricultural_management_matrices(data, base_year)   # lower bound of every ag-mgt entry: non-reversible options lock in last step's adoption

    dvar_base_ag_mrj        = tools.clamp_dvar_bound(data.ag_dvars[base_year], 0.0, trans_ub_ag_mrj, 'Ag base clipped to [0,ub]')
    dvar_base_nonag_rk      = tools.clamp_dvar_bound(data.non_ag_dvars[base_year], trans_lb_nonag_rk, trans_ub_nonag_rk, 'NonAg base clipped to [lb,ub]')

    # ── 3. feasibility: an entry gets a column exactly where its upper bound is above zero ──
    feasible_ag_mrj         = trans_ub_ag_mrj > 0                                               # which ag (m, j) a cell may hold: reachable from a source here, or already held (the ub is raised to the base)

    # ── 4. the arcs: each source's own upper bound, above zero, on its own cells ──
    valid_ag2ag           = get_arc_ag2ag_src(data, base_year, trans_source_ag)               # {source: bool (to_m, local_r, to_j)}
    valid_nonag2ag        = get_arc_nonag2ag_src(data, base_year, trans_source_nonag)         # {source: bool (to_m, local_r, to_j)}
    valid_ag2nonag        = get_arc_ag2nonag_src(data, trans_source_ag, trans_ub_nonag_rk)    # {source: bool (local_r, to_k)}

    # ── 5. masks: the cells the renewable options get no column in (the row side reads the same cell sets off its inputs) ──
    mask_gbf2_solar         = get_mask_gbf2_solar(data)
    mask_gbf2_wind          = get_mask_gbf2_wind(data)

    # ── 6. the blocks and the table: every block's rows laid back to back in Var.index order, each block enumerated from its mask ──
    valid_ag_mrj,   ag_rows    = ag_space(feasible_ag_mrj, trans_ub_ag_mrj, dvar_base_ag_mrj)
    valid_nonag_rk, nonag_rows = nonag_space(data, trans_lb_nonag_rk, trans_ub_nonag_rk, dvar_base_nonag_rk)
    valid_am_smr,   am_rows    = am_space(data, feasible_ag_mrj, mask_gbf2_solar, mask_gbf2_wind, trans_lb_ag_man_mrj)

    ag2ag_rows                   = ag2ag_space(valid_ag2ag, trans_source_ag)              # each source carries its own cells: local_r -> the global cell
    ag2nonag_rows                = ag2nonag_space(valid_ag2nonag, trans_source_ag)
    nonag2ag_rows                = nonag2ag_space(valid_nonag2ag, trans_source_nonag)
    
    # the blocks, in the table's order: the three a per-cell coefficient accounts over (ag, nonag, am: the first
    # n_terms columns the demand, GHG, water, biodiversity and renewable rows read), then the arcs (charged per arc:
    # transition cost in the objective, transition emissions in the GHG row)
    blocks = dict(ag=ag_rows, nonag=nonag_rows, am=am_rows, ag2ag=ag2ag_rows, ag2nonag=ag2nonag_rows, nonag2ag=nonag2ag_rows)

    # the chunks inside the am and arc blocks — one per (option, land use, lm), one per source — by their widths: the
    # table lays them out as it lays the blocks out, and keeps their slice of it beside block_range
    slots    = [(option, j_idx) for option, lus in data.AGMAN2LU.items() for j_idx in range(len(lus))]
    valid_am = {(option, j_idx, m): valid_am_smr[s, m] for s, (option, j_idx) in enumerate(slots) for m in range(data.NLMS)}   # {(option, j_idx, m): bool (cell,)}, in the am block's (slot, lm) order
    chunks = {
        'am':       {key: int(mask.sum()) for key, mask in valid_am.items()},
        'ag2ag':    {src: int(mask.sum()) for src, mask in valid_ag2ag.items()},
        'ag2nonag': {src: int(mask.sum()) for src, mask in valid_ag2nonag.items()},
        'nonag2ag': {src: int(mask.sum()) for src, mask in valid_nonag2ag.items()},
    }

    table = table_space(blocks, chunks, list(data.AGMAN2LU), data.LANDMANS)                        # the block and chunk bounds land in its attrs

    block_range = table.attrs['block_range']
    print(f"Column space: {table.attrs['n_all']:,} columns = {table.attrs['n_terms']:,} accounting (n_terms) + "
          f"{table.attrs['n_all'] - table.attrs['n_terms']:,} arcs", flush=True)
    for name, span in block_range.items():
        print(f"{'└──' if name == list(block_range)[-1] else '├──'} {name:<10s} {span.stop - span.start:>12,}", flush=True)

    # ── 7. the space: the table, and beside it the handles — the mask every block was enumerated from, the region pair,
    #       the cell incidence and the position → column grids ──
    region2cell = cell_regions(data)                                                # the region layers on cell: what filters an input
    region2col  = region2cell.isel(cell=table['cell'].values).rename(cell='col')    # the same layers read at every column's cell: what filters the table

    # for each cell, which variables sit in it: the table's ``cell`` field as a (cell x col) matrix, 1 where they do
    cell     = table['cell'].values                                                 # the cell each gp Variable (column) sits in, one per column
    n_all    = table.attrs['n_all']                                                 # num of gp Variables (columns)
    cell2col = sparse.csr_matrix(
        (np.ones(n_all, dtype=np.float32), (cell, np.arange(n_all, dtype=np.int32))),
        shape=(data.NCELLS, n_all)
    )

    # for each mrj / rk position, which ag / non-ag variable it is: the column at a valid entry is its running count
    # within the block (the mask read backwards), -1 where it has none (no arcs, no am)
    ag           = block_range['ag']
    nonag        = block_range['nonag']
    ag_mrj2col   = np.where(valid_ag_mrj,   ag.start    + np.cumsum(valid_ag_mrj).reshape(valid_ag_mrj.shape)     - 1, -1).astype(np.int32)
    nonag_rk2col = np.where(valid_nonag_rk, nonag.start + np.cumsum(valid_nonag_rk).reshape(valid_nonag_rk.shape) - 1, -1).astype(np.int32)

    return table, ColSupport(
        valid_ag_mrj=valid_ag_mrj,
        valid_nonag_rk=valid_nonag_rk,
        valid_am=valid_am,
        valid_ag2ag=valid_ag2ag,
        valid_ag2nonag=valid_ag2nonag,
        valid_nonag2ag=valid_nonag2ag,
        region2cell=region2cell,
        region2col=region2col,
        cell2col=cell2col,
        ag_mrj2col=ag_mrj2col,
        nonag_rk2col=nonag_rk2col,
    )


# ═══════════════════════════ data: what decides existence, bounds and base (from the base-year state) ═══════════════════════════

# ── transition bounds: the ub / lb of every target entry (TO-view) ──

def get_base_nonag_rk(data: Data, base_year: int) -> np.ndarray:
    """The base-year non-ag holding (cell, nonag_lu): nothing is held in the base year itself, so all zeros there."""
    if base_year == data.YR_CAL_BASE:
        return np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32)
    return data.non_ag_dvars[base_year]

def get_trans_ub_ag_mrj(data: Data, base_year: int) -> np.ndarray:
    """Ag target upper bound (ag2ag + nonag2ag), raised to the base so a cell can always keep its base land use."""
    print('Getting agricultural target upper bounds...', flush=True)
    ub = (ag_transition.get_ag2ag_ub(data, base_year) + non_ag_transition.get_nonag2ag_ub(data, base_year))
    return tools.clamp_dvar_bound(ub, tools.get_base_held(data.ag_dvars[base_year]), np.inf, 'Ag ub raised to base')

def get_trans_ub_nonag_rk(data: Data, base_year: int) -> np.ndarray:
    """Non-ag target upper bound, raised to the base so a cell can always keep its base land use."""
    print('Getting non-agricultural target upper bounds...', flush=True)
    base_nonag_rk = get_base_nonag_rk(data, base_year)
    ub = non_ag_transition.get_non_ag_ub_matrices(
        data,
        base_dvar_nonag_rk=base_nonag_rk,
        base_dvar_ag_mrj=data.ag_dvars[base_year],
    )
    return tools.clamp_dvar_bound(ub, tools.get_base_held(base_nonag_rk), np.inf, 'NonAg ub raised to base')

def get_trans_lb_nonag_rk(data: Data, base_year: int) -> np.ndarray:
    """Non-ag target lower bound, clamped to [0, base]."""
    print('Getting non-agricultural lower bound matrices...', flush=True)
    base_nonag_rk = get_base_nonag_rk(data, base_year)
    lb = non_ag_transition.get_non_ag_lb_matrices(data, base_year)
    # capped by the base as held (not the noise-floor deadband the ub is raised to): a lock-in can never ask for
    # more than the cell holds, and the cap only ever lowers a lb
    return tools.clamp_dvar_bound(lb, 0.0, np.maximum(base_nonag_rk.astype(np.float32), 0.0), 'NonAg lb clamped to [0,base]')

# ── the arcs: each source's own upper bound, above zero, on its own cells ──

def get_arc_ag2ag_src(data: Data, base_year: int, trans_source_ag: dict) -> dict:
    """{(from_m, from_j): bool (to_m, local_r, to_j)} — where ONE ag source's own upper bound is above zero: its base
    share × its T_MAT reach row × EXCLUDE × no-go, its own entry dropped because staying is not a transition;
    ``local_r`` indexes the source's cells in ``trans_source_ag``."""
    print('Getting ag2ag arc upper bounds...', flush=True)
    arcs = {}
    for (from_m, from_j), src_cells in trans_source_ag.items():
        ub = ag_transition.get_ag2ag_ub_src(data, base_year, from_m, from_j, src_cells)   # (NLMS, ncells_src, N_AG)
        ub[from_m, :, from_j] = 0                                                         # staying is not a transition
        arcs[(from_m, from_j)] = ub > 0
    return arcs

def get_arc_nonag2ag_src(data: Data, base_year: int, trans_source_nonag: dict) -> dict:
    """{from_k: bool (to_m, local_r, to_j)} — where ONE non-ag source's own upper bound is above zero; ``local_r``
    indexes the source's cells in ``trans_source_nonag``."""
    print('Getting nonag2ag arc upper bounds...', flush=True)
    return {
        from_k: non_ag_transition.get_nonag2ag_ub_src(data, base_year, from_k, src_cells) > 0   # (NLMS, ncells_k, N_AG)
        for from_k, src_cells in trans_source_nonag.items()
    }

def get_arc_ag2nonag_src(data: Data, trans_source_ag: dict, trans_ub_nonag_rk: np.ndarray) -> dict:
    """{(from_m, from_j): bool (local_r, to_k)} — where the non-ag target upper bound survives an ag source's own
    T_MAT reach row on its cells; ``local_r`` indexes the source's cells in ``trans_source_ag``."""
    print('Getting ag2nonag arc upper bounds...', flush=True)
    reach_jk = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.NON_AGRICULTURAL_LANDUSES).values)   # (from_j, to_k): finite = the transition is allowed
    return {
        (from_m, from_j): (trans_ub_nonag_rk[src_cells, :] * reach_jk[from_j][None, :]) > 0   # (ncells_src, N_NONAG)
        for (from_m, from_j), src_cells in trans_source_ag.items()
    }

# ═══════════════════════════ the column blocks: every builder returns its block's ROWS as a field dict (table_space lays them back to back) ═══════════════════════════

def ag_space(feasible_ag_mrj: np.ndarray, trans_ub_ag_mrj: np.ndarray, dvar_base_ag_mrj: np.ndarray) -> tuple[np.ndarray, dict]:
    """The ag columns: the mask (lm, cell, lu) the block is enumerated from — the layout of every ag input, so ``x_mrj[mask]`` is the block's vector — and the block's rows — its (lm, lu), cell, ub and base; lb = 0 — in (lm, cell, lu) order."""
    valid = feasible_ag_mrj

    # the column (gp.Var) view: where the ag exists, in the inputs' own order
    m, r, j = np.nonzero(valid)                                                  # the block's columns: mask order = column order
    rows = dict(
        m=m,
        j=j,
        cell=r,
        ub=trans_ub_ag_mrj[m, r, j],                                               # attribute: the transition upper bound (lb = 0 by default in gurobi)
        base=dvar_base_ag_mrj[m, r, j]                                             # attribute: the node-balance constant (X = base + flow-in - flow-out)
    )
    return valid, rows


def nonag_space(data: Data, trans_lb_nonag_rk: np.ndarray, trans_ub_nonag_rk: np.ndarray, dvar_base_nonag_rk: np.ndarray) -> tuple[np.ndarray, dict]:
    """The non-ag columns: the mask (cell, nonag_lu) the block is enumerated from — the layout of every non-ag input, so ``x_rk[mask]`` is the block's vector — and the block's rows — its land use, cell, lb, ub and base — in (cell, lu) order."""
    valid = trans_ub_nonag_rk > 0                                                # every feasible entry, enabled land use or not
    enabled = np.array([settings.NON_AG_LAND_USES[lu_name] for lu_name in data.NON_AGRICULTURAL_LANDUSES], dtype=bool)

    # the column (gp.Var) view: where the non-ag land use is feasible, in the inputs' own order; a DISABLED land use's
    # column is fixed at zero — it exists so its node balance and source cap read the table like every other entry
    r, k = np.nonzero(valid)                                                     # the block's columns: mask order = column order
    open_ = enabled[k]
    rows = dict(
        k=k,
        cell=r,
        lb=np.where(open_, trans_lb_nonag_rk[r, k], np.float32(0.0)),             # attribute: the transition lower bound
        ub=np.where(open_, trans_ub_nonag_rk[r, k], np.float32(0.0)),             # attribute: the transition upper bound
        base=dvar_base_nonag_rk[r, k]                                              # attribute: the node-balance constant (X = base + flow-in - flow-out)
    )
    return valid, rows


def am_space(data: Data, feasible_ag_mrj: np.ndarray, mask_gbf2_solar: np.ndarray, mask_gbf2_wind: np.ndarray, trans_lb_ag_man_mrj: dict) -> tuple[np.ndarray, dict]:
    """The ag-management columns: the mask (slot, lm, cell) the block is enumerated from — per slot the (lm, cell) layout of an option's input at that slot's land use — and the block's rows — its slot, option, land use, the ag (lm, lu) it sits on, cell and lb; ub = 1 — in (slot = (am, lu), lm, cell) order."""
    slots = [(am, lu_code) for am, lu_codes in data.AGMAN2LU.items() for lu_code in lu_codes]   # the (option, land use) slots, in slot order

    valid = np.zeros((len(slots), data.NLMS, data.NCELLS), dtype=bool)
    lb_smr = np.zeros((len(slots), data.NLMS, data.NCELLS), dtype=np.float32)
    savanna_mask = data.SAVBURN_ELIGIBLE == 1                                          # cells eligible for savanna burning

    for slot, (am, j) in enumerate(slots):
        slot_valid = feasible_ag_mrj[:, :, j].copy()                                 # (lm, cell): where the ag column exists
        # exclude cells for renewable options
        if am in settings.RENEWABLES_OPTIONS:
            excluded_cells = mask_gbf2_solar if am == "Utility Solar PV" else mask_gbf2_wind
            slot_valid[:, excluded_cells] = False
        # exclude cells for savanna burning
        elif am == "Savanna Burning":
            slot_valid[0] &= savanna_mask                                            # dry only
        # set lower bounds for non-reversible options
        if not settings.AG_MANAGEMENTS_REVERSIBLE[am]:
            lb_smr[slot] = trans_lb_ag_man_mrj[am][:, :, j]                            # (lm, cell) of land use j
        # assign the slot's column existence
        valid[slot] = slot_valid

    j_of_slot      = np.array([j for _, j in slots], dtype=np.int32)                   # the land-use code of each (am, lu) slot
    am_idx_of_slot = np.array([option for option, lu_codes in enumerate(data.AGMAN2LU.values()) for _ in lu_codes], dtype=np.int32)     # the slot's option, as an index into agman2lu
    j_idx_of_slot  = np.array([position for lu_codes in data.AGMAN2LU.values() for position in range(len(lu_codes))], dtype=np.int32)   # the land use's position within its option: the last axis of the per-option effect arrays [m, r, j_idx]

    # the column (gp.Var) view: where the ag-mgt slot exists; nothing reads an am column by grid position
    slot, m, r = np.nonzero(valid)                                                   # the block's columns: mask order = column order
    return valid, dict(
        slot=slot,
        am_idx=am_idx_of_slot[slot],
        j_idx=j_idx_of_slot[slot],
        m=m,
        j=j_of_slot[slot],
        cell=r,
        lb=lb_smr[slot, m, r],                                                         # attribute: the base-year adoption for non-reversible options, else 0
        ub=1.0
    )


def ag2ag_space(valid_ag2ag: dict, trans_source_ag: dict) -> dict:
    """The ag → ag arc columns as rows: one per (from_m, from_j) → (to_m, to_j) arc at a cell, sorted by source in source-map order."""
    print('Building the ag2ag arc block...', flush=True)
    arc_rows = []                                            # one chunk of rows per source
    for (from_m, from_j), valid in valid_ag2ag.items():
        to_m, local_r, to_j = np.nonzero(valid)            # the source's run: mask order = column order
        cell = trans_source_ag[(from_m, from_j)][local_r]    # index of the cell in the global cell list
        arc_rows.append(np.column_stack([
            np.full(cell.size, from_m),
            np.full(cell.size, from_j),
            to_m,
            to_j,
            local_r,
            cell
        ]))

    arcs = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 6), dtype=np.int32)   # one arc per row; the fields are its columns

    # the column (gp.Var) view: one arc per positive upper bound; no row view, nothing looks an arc up by its coordinates
    rows = dict(
        from_m =arcs[:, 0],
        from_j =arcs[:, 1],
        m      =arcs[:, 2],                                  # the ag (lm, lu) the arc lands on
        j      =arcs[:, 3],
        local_r=arcs[:, 4],                                  # the arc's cell in its SOURCE's frame: the per-source cost / GHG arrays are only ncells_src tall, and solve() scatters the arc's value back at [to_m, local_r, to_j]
        cell   =arcs[:, 5],                                  # the same cell in the global frame, which the demand / GHG / water / biodiversity rows weight by
        ub     =np.inf
    )
    return rows


def ag2nonag_space(valid_ag2nonag: dict, trans_source_ag: dict) -> dict:
    """The ag → non-ag arc columns as rows: one per (from_m, from_j) → to_k arc at a cell, sorted by source in source-map order."""
    print('Building the ag2nonag arc block...', flush=True)
    arc_rows = []                                            # one chunk of rows per source
    for (from_m, from_j), valid in valid_ag2nonag.items():
        local_r, to_k = np.nonzero(valid)                  # the source's run: mask order = column order
        cell = trans_source_ag[(from_m, from_j)][local_r]    # index of the cell in the global cell list
        arc_rows.append(np.column_stack([
            np.full(cell.size, from_m),
            np.full(cell.size, from_j),
            to_k,
            local_r,
            cell
        ]))

    arcs = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 5), dtype=np.int32)   # one arc per row; the fields are its columns

    # the column (gp.Var) view: one arc per positive upper bound; no row view, nothing looks an arc up by its coordinates
    rows = dict(
        from_m =arcs[:, 0],
        from_j =arcs[:, 1],
        k      =arcs[:, 2],                                  # the non-ag land use the arc lands on
        local_r=arcs[:, 3],                                  # the arc's cell in its SOURCE's frame: the per-source cost arrays are only ncells_src tall, and solve() scatters the arc's value back at [local_r, to_k]
        cell   =arcs[:, 4],                                  # the same cell in the global frame, which the demand / GHG / water / biodiversity rows weight by
        ub     =np.inf
    )
    return rows


def nonag2ag_space(valid_nonag2ag: dict, trans_source_nonag: dict) -> dict:
    """The non-ag → ag arc columns as rows: one per from_k → (to_m, to_j) arc at a cell, sorted by source in source-map order."""
    print('Building the nonag2ag arc block...', flush=True)
    arc_rows = []                                            # one chunk of rows per source
    for from_k, valid in valid_nonag2ag.items():
        to_m, local_r, to_j = np.nonzero(valid)            # the source's run: mask order = column order
        cell = trans_source_nonag[from_k][local_r]           # index of the cell in the global cell list
        arc_rows.append(np.column_stack([
            np.full(cell.size, from_k),
            to_m,
            to_j,
            local_r,
            cell
        ]))

    arcs = np.concatenate(arc_rows).astype(np.int32) if arc_rows else np.empty((0, 5), dtype=np.int32)   # one arc per row; the fields are its columns

    # the column (gp.Var) view: one arc per positive upper bound; no row view, nothing looks an arc up by its coordinates
    rows = dict(
        from_k =arcs[:, 0],
        m      =arcs[:, 1],                                  # the ag (lm, lu) the arc lands on
        j      =arcs[:, 2],
        local_r=arcs[:, 3],                                  # the arc's cell in its SOURCE's frame: the per-source cost arrays are only ncells_k tall, and solve() scatters the arc's value back at [to_m, local_r, to_j]
        cell   =arcs[:, 4],                                  # the same cell in the global frame, which the demand / GHG / water / biodiversity rows weight by
        ub     =np.inf
    )
    return rows


def table_space(blocks: dict, chunks: dict, options: list, landmans: list) -> xr.Dataset:
    """The whole space as ONE long table on (col = Var.index): the blocks' rows back to back in the order ``blocks`` declares them, the fields of each column (-1 where n/a), its lb / ub / base; ``chunks`` = {block: {key: width}} the runs inside the am and arc blocks, kept as ``<block>_range`` = {key: slice(start, stop)} beside ``block_range``; ``options`` names the am_idx field and ``landmans`` the m field."""

    # each block's run of the table: the rows [start, stop) it owns
    widths = [rows['cell'].size for rows in blocks.values()]
    bounds = np.cumsum([0, *widths])
    block_range = {block: slice(int(start), int(stop)) for block, start, stop in zip(blocks, bounds[:-1], bounds[1:])}

    n_all   = int(bounds[-1])                         # every column: the rows are built at this width

    # each chunk's run of its block: the (option, land use) slots of the am block, the sources of an arc block, laid out in the order given
    chunk_range = {}
    for block, chunk_widths in chunks.items():
        chunk_bounds = block_range[block].start + np.cumsum([0, *chunk_widths.values()])
        chunk_range[f'{block}_range'] = {key: slice(int(start), int(stop)) for key, start, stop in zip(chunk_widths, chunk_bounds[:-1], chunk_bounds[1:])}
        assert chunk_bounds[-1] == block_range[block].stop, f'the {block} chunks do not fill the block'
    n_terms = blocks['ag']['cell'].size + blocks['nonag']['cell'].size + blocks['am']['cell'].size   # the three accounting blocks lead the table, so their width IS the prefix a demand / GHG / water / biodiversity / renewable coefficient array is allocated at

    # a field over the whole table: -1 (or the fill) on the blocks that have no such field, e.g. slot / am_idx / j_idx off the am block
    def field(field_name, dtype, fill):
        """One field over the whole table: each block's array for it, or the fill where the block has no such field."""
        per_block = []
        for rows, width in zip(blocks.values(), widths):
            value = rows.get(field_name, fill)                   # the block's own array, or the fill where the field does not apply to it
            value = np.asarray(value, dtype=dtype)               # one dtype for the whole field, whatever each block happened to store
            per_block.append(np.broadcast_to(value, width))      # a scalar stretches to the block's width (the fill, or a constant like ub = 1.0)
        return np.concatenate(per_block)                         # the blocks back to back: one value per column of the table

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
             cell   =(('col',), field('cell', np.int32, -1)),                                # the cell in the GLOBAL frame — every column has one, and the demand / GHG / water / biodiversity / renewable rows weight by it
             lb     =(('col',), field('lb', np.float32, 0.0)),                               # the bounds of the column (float32, as every input is; gurobi reads them as doubles)
             ub     =(('col',), field('ub', np.float32, np.inf)),
             base   =(('col',), field('base', np.float32, 0.0))                              # the node-balance constant of an ag / non-ag column
        ),
        attrs=dict(
            block_range=block_range,                                                         # {block: slice(start, stop)} — the rows each block owns, in the table's block order
            **chunk_range,                                                                   # am_range {(option, j_idx, m): slice(start, stop)}, ag2ag_range / ag2nonag_range / nonag2ag_range {source: slice(start, stop)} — the runs inside those blocks
            landmans=landmans,                                                               # the land-management names, in m order ('dry', 'irr'): what the m field means
            options=options,                                                                 # the ag-management options, in am_idx order: what the am_idx field means
            n_terms=n_terms,                                                                 # the ag, nonag and am blocks: the table's first rows, the width a demand / GHG / water / biodiversity / renewable coefficient array is allocated at
            n_all=n_all,                                                                     # every column: the rows and the objective are built at this width
        )
    )


def cell_regions(data: Data) -> xr.Dataset:
    """Every cell labelled with the region it sits in — one variable per region layer on ``cell``: the ``state`` and
    ``nrm`` codes and the ``water_region`` id — with the code → name maps in attrs. Read at the columns' cells it is
    ``region2col``; a region's cells are ``region2cell[layer] == code`` and its columns ``region2col[layer] == code``.
    (The regional-adoption caps carry their own cell set, because their region is whichever layer the settings pick,
    so they are not a layer here.) The ``ibra`` bioregion layer (-1 = no bioregion) is there only when the GBF3 targets
    are set per IBRA bioregion, the one mode that loads it."""
    ibra, ibra_name = {}, {}
    if settings.GBF3_NVIS_TARGET != 'off' and settings.GBF3_NVIS_REGION_MODE == 'IBRA_REG':
        ibra      = dict(ibra=(('cell',), np.asarray(data.REGION_IBRA_CODE).astype(np.int32)))
        ibra_name = dict(ibra_name=dict(enumerate(data.REGION_IBRA_NAMES)))
    return xr.Dataset(
        dict(state        =(('cell',), np.asarray(data.REGION_STATE_CODE).astype(np.int16)),
             nrm          =(('cell',), np.asarray(data.REGION_NRM_CODE).astype(np.int32)),
             water_region =(('cell',), np.asarray(data.WATER_REGION_ID).astype(np.int32)),
             **ibra),
        attrs=dict(state_name={code: name for name, code in data.REGION_STATE_NAME2CODE.items()},
                   nrm_name=dict(zip(np.asarray(data.REGION_NRM_CODE).tolist(), np.asarray(data.REGION_NRM_NAME).tolist())),
                   water_region_name=dict(data.WATER_REGION_NAMES),
                   **ibra_name),
    )
