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
The solution in the LUTO format: the raw x over the column table scattered back into the (m, r, j) / (r, k)
cubes and the source-keyed transition deltas, the maps, and the production data the writers read.
"""

import numpy as np
import xarray as xr

from dataclasses import dataclass
from typing import Any

import luto.settings as settings
from luto.settings import AG_MANAGEMENTS
from luto.solvers import row_builder
from luto.solvers.col_builder import ColSide
from luto.solvers.row_builder import RowSide
from luto.solvers.row_inputs import RowInputs


@dataclass
class SolverSolution:
    lumap: np.ndarray                                                     # the land use of every cell (non-ag codes offset by NON_AGRICULTURAL_LU_BASE_CODE)
    lmmap: np.ndarray                                                     # the land management of every cell (0 = dry, 1 = irr; non-ag cells are dry)
    ammaps: dict[str, np.ndarray]                                         # {option: 0/1 per cell} — 1 where the cell's chosen (lm, lu) carries the option at or above AGRICULTURAL_MANAGEMENT_USE_THRESHOLD
    ag_X_mrj: np.ndarray                                                  # the ag shares, float32 (NLMS, NCELLS, N_AG_LUS), fractional values as solved
    non_ag_X_rk: np.ndarray                                               # the non-ag shares, float32 (NCELLS, N_NON_AG_LUS); a disabled land use stays at zero
    ag_man_X_mrj: dict[str, np.ndarray]                                   # {option: the ag-mgt shares, float32 (NLMS, NCELLS, N_AG_LUS)}
    dvar_D_ag2ag_mrj: dict                                                # Solved ag->ag deltas, SOURCE-KEYED: {(from_m, from_j): ndarray(NLMS, ncells_src, N_AG_LUS) [to_m, local_r, to_j]} over the source's cells (get_base_dvar_mj_cell_map)
    dvar_D_ag2nonag_rk: dict                                              # Solved ag->nonag deltas, SOURCE-KEYED: {(from_m, from_j): ndarray(ncells_src, N_NON_AG_LUS) [local_r, k]}
    dvar_D_nonag2ag_mrj: dict                                             # Solved nonag->ag deltas, SOURCE-KEYED: {from_k: ndarray(NLMS, ncells_k, N_AG_LUS) [to_m, local_r, to_j]} (e.g. reversible Destocked back to ag; cells via get_base_nonag_dvar_k_cell_map)
    prod_data: dict[str, Any]                                             # what the writers read: 'Production' (raw t per commodity, unscaled) and 'GHG' (raw tCO2e, the off-land constant included)


def post_solve(x: np.ndarray, cols: xr.Dataset, col_side: ColSide, rows: xr.Dataset, row_side: RowSide, inputs: RowInputs) -> SolverSolution:
    """The LUTO-format solution of one step from the raw ``x`` (``LutoSolver.solve``): the column table says
    what every entry of x is, the col side which source each arc belongs to, the row side and the row table
    give the production data."""
    print("Collecting results...\n", flush=True)
    nlms, n_ag_lus, n_nonag_lus, ncells = (cols.attrs[key] for key in ('nlms', 'n_ag_lus', 'n_nonag_lus', 'ncells'))
    agman2lu = cols.attrs['agman2lu']

    # ── 1. the decision variables: x scattered back through the table's fields (float64 -> float32) ──
    rows_of = {block: slice(*block_rows) for block, block_rows in cols.attrs['block_range'].items()}
    m, j, k, am_idx, local_r, cell = (cols[field].values for field in ('m', 'j', 'k', 'am_idx', 'local_r', 'cell'))

    X_dry_sol_rj = np.zeros((ncells, n_ag_lus), dtype=np.float32)
    X_irr_sol_rj = np.zeros((ncells, n_ag_lus), dtype=np.float32)
    non_ag_X_sol_rk = np.zeros((ncells, n_nonag_lus), dtype=np.float32)
    am_X_dry_sol_rj = {am: np.zeros((ncells, n_ag_lus), dtype=np.float32) for am in agman2lu}
    am_X_irr_sol_rj = {am: np.zeros((ncells, n_ag_lus), dtype=np.float32) for am in agman2lu}

    # agricultural
    ag = rows_of['ag']
    is_dry = m[ag] == 0
    X_dry_sol_rj[cell[ag][is_dry],  j[ag][is_dry]]  = x[ag][is_dry]
    X_irr_sol_rj[cell[ag][~is_dry], j[ag][~is_dry]] = x[ag][~is_dry]

    # non-agricultural (disabled land uses have no columns and stay at zero)
    nonag = rows_of['nonag']
    non_ag_X_sol_rk[cell[nonag], k[nonag]] = x[nonag]

    # ag-management. Savanna eligibility is applied to BOTH lm here, while variable creation applied
    # it to dry only: irr savanna vars outside the eligible cells report 0.
    am = rows_of['am']
    options = cols.attrs['options']
    am_of_col = np.asarray(options, dtype=object)[am_idx[am]]
    reported = ~((am_of_col == "Savanna Burning") & (m[am] == 1) & ~np.isin(cell[am], cols.attrs['savanna_eligible_r']))
    for option in options:
        dry_cols = reported & (am_of_col == option) & (m[am] == 0)
        irr_cols = reported & (am_of_col == option) & (m[am] == 1)
        am_X_dry_sol_rj[option][cell[am][dry_cols], j[am][dry_cols]] = x[am][dry_cols]
        am_X_irr_sol_rj[option][cell[am][irr_cols], j[am][irr_cols]] = x[am][irr_cols]

    ag_X_mrj = np.stack((X_dry_sol_rj, X_irr_sol_rj))                    # fractional values preserved as-is
    ag_man_X_mrj = {am: np.stack((am_X_dry_sol_rj[am], am_X_irr_sol_rj[am])) for am in agman2lu}

    # ── 2. the transition deltas: the gross flows the objective charged, SOURCE-KEYED so reporting can
    #       attribute the true from → to flows. Leaf axes mirror the flow_cost dicts ([to_m, local_r, to_j]
    #       for ag targets, [local_r, k] for non-ag targets); local_r indexes the source's cell list.
    #       Each source's arcs are one run of the block's rows (src_ptr = the group bounds of the block sorted by source).
    dvar_D_ag2ag_mrj    = {}   # (from_m, from_j) -> (NLMS, ncells_src, N_AG_LUS)
    dvar_D_ag2nonag_rk  = {}   # (from_m, from_j) -> (ncells_src, N_NON_AG_LUS)
    dvar_D_nonag2ag_mrj = {}   # from_k           -> (NLMS, ncells_k, N_AG_LUS)
    x_arcs = x.astype(np.float32)

    def src_rows(block, src_idx):
        src_ptr = cols.attrs['src_ptr'][block]
        return slice(int(src_ptr[src_idx]), int(src_ptr[src_idx + 1]))

    for src_idx, ((from_m, from_j), cells) in enumerate(col_side.sources_ag.items()):
        arcs = src_rows('ag2ag', src_idx)
        deltas = np.zeros((nlms, len(cells), n_ag_lus), dtype=np.float32)
        deltas[m[arcs], local_r[arcs], j[arcs]] = x_arcs[arcs]
        dvar_D_ag2ag_mrj[(from_m, from_j)] = deltas

        arcs = src_rows('ag2nonag', src_idx)
        deltas = np.zeros((len(cells), n_nonag_lus), dtype=np.float32)
        deltas[local_r[arcs], k[arcs]] = x_arcs[arcs]
        dvar_D_ag2nonag_rk[(from_m, from_j)] = deltas
    for src_idx, (from_k, cells) in enumerate(col_side.sources_nonag.items()):
        arcs = src_rows('nonag2ag', src_idx)
        deltas = np.zeros((nlms, len(cells), n_ag_lus), dtype=np.float32)
        deltas[m[arcs], local_r[arcs], j[arcs]] = x_arcs[arcs]
        dvar_D_nonag2ag_mrj[from_k] = deltas

    # ── 3. the maps: land use, land management, ag-management options ──
    non_ag_dominates_r = non_ag_X_sol_rk.max(axis=1) > ag_X_mrj.max(axis=(0, 2))   # used for lumap/lmmap only
    lumap = ag_X_mrj.sum(axis=0).argmax(axis=1).astype("int8")
    lmmap = ag_X_mrj.sum(axis=2).argmax(axis=0).astype("int8")
    lumap[non_ag_dominates_r] = (
        non_ag_X_sol_rk[non_ag_dominates_r, :].argmax(axis=1)
        + settings.NON_AGRICULTURAL_LU_BASE_CODE
    )
    lmmap[non_ag_dominates_r] = 0                                        # all non-agricultural land uses are dryland
    # one map per option (options can stack): 1 where the cell's chosen (lm, lu) carries the option at or
    # above AGRICULTURAL_MANAGEMENT_USE_THRESHOLD; non-ag cells carry no option
    ammaps = {am: np.zeros(ncells, dtype=np.int8) for am in AG_MANAGEMENTS}
    ag_cells = np.flatnonzero(lumap < settings.NON_AGRICULTURAL_LU_BASE_CODE)
    chosen_j = lumap[ag_cells].astype(np.int64)
    chosen_m = lmmap[ag_cells].astype(np.int64)
    for am, lu_codes in agman2lu.items():
        adoption = ag_man_X_mrj[am][chosen_m, ag_cells, chosen_j]
        adopted = (adoption >= settings.AGRICULTURAL_MANAGEMENT_USE_THRESHOLD) & np.isin(chosen_j, lu_codes)
        ammaps[am][ag_cells[adopted]] = 1

    # ── 4. the production data the writers read: Production from the unscaled production block (raw t per
    #       commodity); GHG as the GHG row's raw-unit value (row × scale) plus the off-land constant the row
    #       excludes — 0 when the row is off, or dropped by the infeasibility flow ──
    prod_data = {"Production": (row_side.q_block @ x).tolist()}
    ghg = row_builder.family_rows(rows, 'ghg')
    ghg_active = rows['active'].values[ghg] if ghg is not None else np.zeros(0, dtype=bool)
    if ghg_active.any():
        row = ghg.start + int(np.flatnonzero(ghg_active)[0])              # the single GHG row
        prod_data["GHG"] = float((rows.attrs['A'][row] @ x)[0] * rows['scale'].values[row]) + float(np.asarray(inputs.offland_ghg).ravel()[0])
    else:
        prod_data["GHG"] = 0

    return SolverSolution(
        lumap=lumap,
        lmmap=lmmap,
        ammaps=ammaps,
        ag_X_mrj=ag_X_mrj,
        non_ag_X_rk=non_ag_X_sol_rk,
        ag_man_X_mrj=ag_man_X_mrj,
        dvar_D_ag2ag_mrj=dvar_D_ag2ag_mrj,
        dvar_D_ag2nonag_rk=dvar_D_ag2nonag_rk,
        dvar_D_nonag2ag_mrj=dvar_D_nonag2ag_mrj,
        prod_data=prod_data,
    )
