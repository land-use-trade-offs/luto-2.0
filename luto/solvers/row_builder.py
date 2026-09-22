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

import luto.tools as tools

from luto import settings
from luto.solvers.col_builder import ColSupport
from luto.solvers.row_inputs import EconomicInputs, RowInputs
from luto.solvers.row_table import FAMILIES, ROW_FILL, ROW_SCHEMA, make_part


# ═══════════════════════════ get_rows: the row space of one step ═══════════════════════════

def get_rows(inputs: RowInputs, cols: xr.Dataset, support: ColSupport) -> tuple[sparse.csr_matrix, xr.Dataset]:
    """The row space of one solve step: every family as its block of A and its table, in the model's row order — first
    the few, wide rows a target or a limit writes over the columns, then the many, narrow rows that say where land
    can move — stacked into the ONE matrix and the ONE row table."""

    # ── 1. the constraints: a coefficient per column laid on the support, summed over the cells the target covers ──
    parts = [
        get_demand(inputs, cols, support),
        get_ghg(inputs, cols, support),
    ]

    print("│   ├── Adding constraints for biodiversity...")
    bio_on = any(target != 'off' for target in (
        settings.GBF2_TARGET,
        settings.GBF3_NVIS_TARGET,
        settings.GBF4_TARGET_SNES,
        settings.GBF4_TARGET_ECNES,
        settings.GBF8_TARGET)
    )
    bio_S = support.cell2col @ sparse.diags(bio_coeff(inputs, cols, support)) if bio_on else None   # (cell x col) every column's biodiversity contribution in its cell's row, laid out ONCE for the five GBF families

    parts += [
        get_GBF2(inputs, bio_S),
        get_GBF3_NVIS(inputs, support, bio_S),
        get_GBF4_SNES(inputs, support, bio_S),
        get_GBF4_ECNES(inputs, support, bio_S),
        get_GBF8(inputs, support, bio_S),
        get_ag_mgt_adoption(inputs, cols, support),
        get_regional_adoption_ag(inputs, cols),
        get_regional_adoption_nonag(inputs, cols),
        get_regional_adoption_nonag_sum(inputs, cols),
        get_water(inputs, cols, support),
        get_renewable(inputs, cols, support),
    ]

    # ── 2. the structure: what a column can be at all — an ag-mgt column within its ag column, simulated + existing
    #       capacity within the cell, the arcs within the source they leave, every node at what it held plus what flowed ──
    parts += [
        get_ag_mgt_link(inputs, cols, support),
        get_renewable_ceiling(inputs, cols, support),
        get_source_cap_ag(cols, support),
        get_source_cap_nonag(cols, support),
        get_node_balance_ag(cols, support),
        get_node_balance_nonag(cols, support),
    ]

    # ── 3. the space: every family's block into the ONE matrix and its table into the ONE row table, in the order above
    #       (the model's row order); a family that is off returned (None, None) and is left out ──
    A = sparse.vstack([A for A, _ in parts if A is not None], format='csr')
    rows = xr.concat(
        [ROW_SCHEMA, *[table for _, table in parts if table is not None]],
        dim='row',
        fill_value=ROW_FILL,
        combine_attrs='no_conflicts'
    )

    # Embed the family ranges
    codes = rows['family'].values
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])                           # where the family code changes
    stops = np.r_[starts[1:], codes.size]
    rows.attrs['family_range'] = {FAMILIES[codes[start]]: slice(int(start), int(stop)) for start, stop in zip(starts, stops)}

    return A, rows


# ═══════════════════════════ the coefficient contract: gather → weigh → contract ═══════════════════════════

def gather(cols: xr.Dataset, support: ColSupport, ag_c_mrj, am_c_mrj: dict, nonag_c_rk,
           ag2ag_c: dict = None, ag2nonag_c: dict = None, nonag2ag_c: dict = None) -> np.ndarray:
    """
    Create a row of coefficients over the columns.
    """
    c = np.zeros(cols.attrs['n_all'], dtype=np.float32)
    
    # assign ag/nonag coefficients 
    c[cols.attrs['block_range']['ag']]    = ag_c_mrj[support.valid_ag_mrj]
    c[cols.attrs['block_range']['nonag']] = nonag_c_rk[support.valid_nonag_rk]
    
    # assign ag-man coefficients
    for (option, j_idx, m), mask in support.valid_am.items():
        c[cols.attrs['am_range'][(option, j_idx, m)]] = am_c_mrj[option][m, :, j_idx][mask]
    
    # assign ag2ag arc coefficients
    if ag2ag_c is not None:
        for src, mask in support.valid_ag2ag.items():
            c[cols.attrs['ag2ag_range'][src]] = ag2ag_c[src][mask]

    # assign ag2nonag arc coefficients
    if ag2nonag_c is not None:
        for src, mask in support.valid_ag2nonag.items():
            c[cols.attrs['ag2nonag_range'][src]] = ag2nonag_c[src][mask]

    # assign nonag2ag arc coefficients
    if nonag2ag_c is not None:
        for src, mask in support.valid_nonag2ag.items():
            c[cols.attrs['nonag2ag_range'][src]] = nonag2ag_c[src][mask]

    return c


def weight_rows(weights, ncells: int) -> sparse.csr_matrix:
    """A family's weighting rows over cells as ONE sparse ``W`` (n_rows × ncells), float32, the nonzero support
    only — built row by row, never as a dense stack (GBF8 would be 10k rows × ncells)."""
    indptr = [0]
    indices = []
    data = []
    for weight_row in weights:
        weight_row = np.asarray(weight_row, dtype=np.float32)
        cells = np.flatnonzero(weight_row)                                 # NaN counts as nonzero: it reaches ``contract``, which drops it
        indices.append(cells)
        data.append(weight_row[cells])
        indptr.append(indptr[-1] + cells.size)
    return sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), np.asarray(indptr, dtype=np.int64)),
                             shape=(len(weights), ncells))


def contract(block: sparse.csr_matrix, rhs=None, rescale: bool = False) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """
    1. Drop tiny coefficients (abs < SOLVER_COEFF_MIN) and eliminate zeros.
    2. Sort the indices of every row (Gurobi requires it).
    3. If ``rescale`` is True, geometrically rescale every row and the RHS.
    """
    
    block = block.tocsr(copy=True)          # copy to avoid modifying the original block in place
    # Drop tiny coefficients and eliminate zeros
    block.data[~(np.abs(block.data) >= settings.SOLVER_COEFF_MIN)] = 0.0   # the drop; NaN fails the test
    block.eliminate_zeros()
    # Sort the indices of every row
    block.sort_indices()
    
    # Skip rescaling if not requested
    rhs = None if rhs is None else np.asarray(rhs, dtype=np.float64)
    if not rescale:
        return block, rhs, np.ones(block.shape[0], dtype=np.float64)

    # the two magnitudes of every row: its largest coefficient and its rhs
    row_max = abs(block).max(axis=1).toarray().ravel().astype(np.float64)  # 0 for an empty row
    rhs_abs = np.abs(rhs)

    # the row's factor: the geometric mean of the two, so both land around RESCALE_FACTOR
    scale = np.sqrt(row_max * rhs_abs)
    scale[rhs_abs == 0] = row_max[rhs_abs == 0]                            # rhs = 0: the coefficients alone set it
    scale[row_max == 0] = settings.RESCALE_FACTOR                          # empty row: factor 1, the rhs is kept as it is
    scale /= settings.RESCALE_FACTOR

    # divide every entry by its row's factor
    row_of_entry = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr))   # the row each stored entry sits in
    block.data = (block.data / scale[row_of_entry]).astype(np.float32)
    block.data[np.abs(block.data) < settings.SOLVER_COEFF_MIN] = 0.0       # floor the scaled row
    block.eliminate_zeros()
    block.sort_indices()
    return block, rhs / scale, scale


def bio_coeff(inputs: RowInputs, cols: xr.Dataset, support: ColSupport) -> np.ndarray:
    """The biodiversity contribution at every column: the three streams ``gather`` reads — a scalar per ag land
    use, a per-cell array per (option, land use), a scalar per non-ag land use — as broadcast VIEWS (no copy),
    gathered ONCE for all five GBF families."""
    nlms, ncells = inputs.nlms, inputs.ncells
    ag_j = np.asarray(inputs.biodiv_contr_ag_j, dtype=np.float32)
    ag_c_mrj = np.broadcast_to(ag_j[None, None, :], (nlms, ncells, ag_j.size))
    nonag_k = np.array([inputs.biodiv_contr_non_ag_k.get(k, 0.0) for k in range(inputs.n_nonag_lus)], dtype=np.float32)
    nonag_c_rk = np.broadcast_to(nonag_k[None, :], (ncells, nonag_k.size))
    am_c_mrj = {}
    for option, by_j_idx in inputs.biodiv_contr_ag_man.items():
        per_cell = np.stack([np.asarray(by_j_idx[j_idx], dtype=np.float32) for j_idx in range(len(by_j_idx))], axis=1)   # (cell, j_idx)
        am_c_mrj[option] = np.broadcast_to(per_cell[None, :, :], (nlms, ncells, per_cell.shape[1]))
    return gather(cols, support, ag_c_mrj, am_c_mrj, nonag_c_rk)


# ═══════════════════════════ get_obj: the objective coefficient of every column ═══════════════════════════

def get_obj(econ: EconomicInputs, cols: xr.Dataset, support: ColSupport) -> xr.DataArray:
    """The objective coefficient of every column (on ``col``), as Gurobi takes it: the operating economics on
    the accounting columns and the transition costs, negated, on the arcs — raw AUD, float32 — through the
    coefficient contract (the SOLVER_COEFF_MIN drop), then scaled to million AUD and floored again, because
    the scaling can push a coefficient under the floor."""
    # ── one gather over the six blocks: the operating economics on the accounting columns, the transition costs on the arcs ──
    obj = gather(cols, support, econ.ag_obj_mrj, econ.ag_man_objs, econ.non_ag_obj_rk,
                 econ.flow_cost_ag2ag, econ.flow_cost_ag2nonag, econ.flow_cost_nonag2ag)
    arcs = slice(cols.attrs['n_terms'], cols.attrs['n_all'])                   # the arc blocks, everything after the accounting columns: a transition is a COST, so it enters negated
    obj[arcs] = -obj[arcs]

    # ── the contract: the drop on the raw coefficient, the scaling to million AUD, the floor on the scaled one ──
    obj[~(np.abs(obj) >= settings.SOLVER_COEFF_MIN)] = 0.0                     # the drop; NaN fails the test too
    obj = obj * (1.0 / 1e6)                                                    # raw AUD -> million AUD (float32, a reciprocal multiply as gurobipy did)
    obj[np.abs(obj) < settings.SOLVER_COEFF_MIN] = 0.0                         # floor the scaled coefficient
    return xr.DataArray(obj, dims=('col',))


# ═══════════════════════════ the structural rows ═══════════════════════════

def get_ag_mgt_link(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One row per am column: X_am − X_ag ≤ 0 (an ag-mgt column cannot exceed the ag column it sits on)."""
    n_all   = cols.attrs['n_all']
    options = cols.attrs['options']
    am_idx_all = []
    ag_idx_all = []
    names = []
    key_am = []
    key_lu = []
    key_lm = []
    key_cell = []
    for (option, j_idx, m), valid_am in support.valid_am.items():
        j = inputs.agman2lu[option][j_idx]
        lm = inputs.landmans[m]
        cells = np.flatnonzero(valid_am)                                                # the cells with an am column of (slot, m): one row each, ascending
        am_cols = np.r_[cols.attrs['am_range'][(option, j_idx, m)]]                # those am columns ...
        ag_cols = support.ag_mrj2col[m, cells, j]                                          # ... and the ag column each sits on
        am_idx_all.append(am_cols)
        ag_idx_all.append(ag_cols)
        names += [f"const_ag_man_{option}_usage_{lm}_{j}_{r}".replace(" ", "_") for r in cells]
        key_am.append(np.full(cells.size, options.index(option), dtype=np.int32))
        key_lu.append(np.full(cells.size, j, dtype=np.int32))
        key_lm.append(np.full(cells.size, m, dtype=np.int32))
        key_cell.append(cells)
    am_idx_all = np.concatenate(am_idx_all)
    ag_idx_all = np.concatenate(ag_idx_all)
    n_rows = am_idx_all.size
    rows = np.arange(n_rows)
    A = sparse.csr_matrix(
        (np.concatenate([np.ones(n_rows), -np.ones(n_rows)]), (np.concatenate([rows, rows]), np.concatenate([am_idx_all, ag_idx_all]))),
        shape=(n_rows, n_all)
    )                                         # +1 on X_am, −1 on X_ag
    A, _, _ = contract(A)
    return make_part(
        'ag_mgt_link',
        A,
        np.zeros(n_rows),
        '<',
        names,
        am_idx=np.concatenate(key_am), j=np.concatenate(key_lu), m=np.concatenate(key_lm), cell=np.concatenate(key_cell),
    )


def get_ag_mgt_adoption(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One row per (option, land use): Σ X_am − limit · Σ X_ag ≤ 0."""
    n_all   = cols.attrs['n_all']
    options = cols.attrs['options']
    row_idx = []
    col_idx = []
    vals = []
    names = []
    key_am = []
    key_lu = []
    slots = [(option, j_idx) for option, lus in inputs.agman2lu.items() for j_idx in range(len(lus))]
    for row, (option, j_idx) in enumerate(slots):
        j = inputs.agman2lu[option][j_idx]
        adoption_limit = float(inputs.ag_man_limits[option][j])
        am_cols = np.concatenate([np.r_[cols.attrs['am_range'][(option, j_idx, m)]] for m in range(inputs.nlms)])   # the slot's am columns, both lm, dry first
        ag_cols = support.ag_mrj2col[:, :, j][support.valid_ag_mrj[:, :, j]]                                               # the ag columns of j: both lm, dry first, cells ascending
        row_idx += [np.full(am_cols.size, row), np.full(ag_cols.size, row)]
        col_idx += [am_cols, ag_cols]
        vals += [np.ones(am_cols.size), np.full(ag_cols.size, -adoption_limit)]
        names.append(f"const_ag_man_adoption_limit_{option}_{j}".replace(" ", "_"))
        key_am.append(options.index(option))
        key_lu.append(j)
    n_rows = len(names)
    A, _, _ = contract(sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_rows, n_all)))
    return make_part('ag_mgt_adoption', A, np.zeros(n_rows), '<', names, am_idx=key_am, j=key_lu)


def get_renewable_ceiling(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One row per (renewable option, cell with existing capacity): Σ X_am ≤ max(ag_mask − existing, 0)."""
    am_idx = cols['am_idx'].values
    ag_mask = inputs.ag_mask_proportion_r
    blocks = []
    rhs = []
    names = []
    key_am = []
    key_cell = []
    for option_idx, option in enumerate(cols.attrs['options']):
        if option not in settings.RENEWABLES_OPTIONS:
            continue
        am_name = tools.am_name_snake_case(option)
        exist_r = inputs.exist_renewable_solar_r if option == "Utility Solar PV" else inputs.exist_renewable_wind_r   # the total across ALL data years: the ceiling never decreases between periods, so lb(t) <= ceiling always holds
        on_option = (am_idx == option_idx).astype(np.float32)                                   # 1 on the option's columns
        has_option = support.cell2col @ on_option != 0                                            # the cells holding a column of the option ...
        row_cells = np.flatnonzero(has_option & (exist_r != 0))                                  # ... and existing capacity (none -> no ceiling row): one row each, ascending
        if not row_cells.size:
            continue
        blocks.append(support.cell2col[row_cells] @ sparse.diags(on_option))                     # the variables in those cells, each column times its on_option: the option's columns
        rhs.append(np.maximum(ag_mask[row_cells] - exist_r[row_cells], 0.0))                    # cell space left for simulated capacity
        names += [f"const_{am_name}_solvable_ub_{r}".replace(" ", "_") for r in row_cells]
        key_am += [option_idx] * row_cells.size
        key_cell.append(row_cells)
    if not blocks:
        return None, None
    A, rhs, _ = contract(sparse.vstack(blocks, format='csr'), np.concatenate(rhs))
    return make_part('renewable_ceiling', A, rhs, '<', names, am_idx=key_am, cell=np.concatenate(key_cell))


# ═══════════════════════════ the demand rows ═══════════════════════════

def get_demand(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """
    Add `N_products` rows, each represeting a commodity's demand constraint.
    """
    print("│   ├── Adding <hard> demand constraints (equality where lb==ub, else lower + upper)...")
    n_all      = cols.attrs['n_all']
    nlms       = inputs.nlms
    n_lu_ag    = inputs.n_ag_lus
    n_lu_nonag = inputs.n_nonag_lus
    ncms       = inputs.ncms
    
    # Get the active products of every ag land use: {ag_idx: [product_idx, ...]} — the unallocated mod/nat are skipped
    active_p   = {lu: p for lu in range(n_lu_ag) if (p := np.flatnonzero(inputs.lu2pr_pj[:, lu])).size}
    
    # Containers for the production block
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

    # the production for ag
    for lu, p in active_p.items():
        for lm in range(nlms):
            cells = np.flatnonzero(support.valid_ag_mrj[lm, :, lu])
            # (commodity × products) @ (products × cell) = (commodity × cell)
            commodity_coeffs = inputs.pr2cm_cp[:, p] @ inputs.ag_q_mrp[lm, cells, :][:, p].T
            put(commodity_coeffs, support.ag_mrj2col[lm, cells, lu])

    # the production for ag-mgt
    for (option, j_idx, lm), valid_am in support.valid_am.items():
        lu = inputs.agman2lu[option][j_idx]
        if lu not in active_p:  # Unallocated mod/nat land has no products
            continue
        p = active_p[lu]
        cells = np.flatnonzero(valid_am)
        # (commodity × products) @ (products × cell) = (commodity × cell)
        commodity_coeffs = inputs.pr2cm_cp[:, p] @ inputs.ag_man_q_mrp[option][lm, cells, :][:, p].T
        put(commodity_coeffs, np.r_[cols.attrs['am_range'][(option, j_idx, lm)]])

    # the production for non-ag
    for lu in range(n_lu_nonag):
        cells = np.flatnonzero(support.valid_nonag_rk[:, lu])
        put(inputs.non_ag_q_crk[:, cells, lu], support.nonag_rk2col[cells, lu])
    
    q_block = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))),
        shape=(ncms, n_all)
    )
    q_block, _, _ = contract(q_block)

    # ── the bound rows ──
    commodity = []                                                       # the row's commodity: which production row it copies, and its label on the table
    senses = []
    rhs = []
    names = []
    key_bound = []
    for c_idx, c_name in enumerate(inputs.commodity_names):
        lb, ub = settings.DEMAND_BOUNDS[c_name]
        demand = inputs.limits['demand'][c_idx]
        bounds = [('eq', '=', lb)] if lb == ub else [('lower', '>', lb), ('upper', '<', ub)]
        for bound, sense, factor in bounds:
            commodity.append(c_idx)
            senses.append(sense)
            rhs.append(demand * factor)
            names.append(f"demand_{bound}_[{c_name}]".replace(" ", "_"))     # e.g. demand_hard_bound_eq_[sheep_meat]
            key_bound.append(bound)
    A, rhs, scale = contract(q_block[commodity], rhs, rescale=True)      # the commodity's production row under each of its bounds; row rescale, factors kept
    return make_part('demand', A, rhs, np.array(senses, dtype=object), names, scale, commodity=commodity, bound=key_bound)


# ═══════════════════════════ the policy rows ═══════════════════════════

def get_ghg(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One global row: Σ ghg · X over the ag, ag-mgt, non-ag columns and the ag → ag arcs ≤ limit − offland."""
    if settings.GHG_EMISSIONS_LIMITS == "off":
        print("│   ├── TURNING OFF GHG emissions constraints ...")
        return None, None
    ghg_limit_raw = inputs.limits["ghg"]
    print(f"│   ├── Adding <hard> constraints for GHG emissions: {ghg_limit_raw:,.0f} tCO2e")

    # land-use, ag-management and non-ag emissions on the accounting columns, transition emissions on the ag → ag arcs
    coeff = gather(cols, support, inputs.ag_g_mrj, inputs.ag_man_g_mrj, inputs.non_ag_g_rk, ag2ag_c=inputs.trans_ghg_ag2ag)
    row = sparse.csr_matrix(coeff[None, :])                              # the nonzero support; the contract drops the rest
    rhs = np.asarray(ghg_limit_raw - inputs.offland_ghg, dtype=np.float64).ravel()   # offland_ghg: 1-element array
    A, rhs, scale = contract(row, rhs, rescale=True)                     # drop + row rescale, factor kept
    return make_part('ghg', A, rhs, '<', ["ghg_emissions_limit_ub"], scale)


def get_GBF2(inputs: RowInputs, bio_S: sparse.csr_matrix):
    """One row: Σ_r mask_area[r] · (the bio contribution of cell r's columns) ≥ target."""
    if settings.GBF2_TARGET == "off":
        print("│   │   ├── TURNING OFF constraints for biodiversity GBF 2...")
        return None, None
    print(f'│   │   ├── Adding constraints for biodiversity GBF 2: {inputs.limits["GBF2"]:15,.0f}')
    A, rhs, scale = contract(weight_rows([inputs.GBF2_mask_area_r], inputs.ncells) @ bio_S, [inputs.limits["GBF2"]], rescale=True)
    return make_part('GBF2', A, rhs, '>', ["bio_GBF2_priority_degraded_area_limit"], scale)


def get_GBF3_NVIS(inputs: RowInputs, support: ColSupport, bio_S: sparse.csr_matrix):
    """One row per (region, group) with a target ≥ 0 and a cell: Σ_r area[group, r] · (the bio contribution of cell r's columns) ≥ target; IBRA bioregions in 'IBRA_REG' mode."""
    if settings.GBF3_NVIS_TARGET == "off":
        print("│   │   ├── TURNING OFF constraints for biodiversity GBF 3 NVIS")
        return None, None
    print("│   │   ├── Adding constraints for biodiversity GBF 3 NVIS...")
    layers = inputs.GBF3_NVIS_pre_1750_area_vr                           # xr [group, cell]
    targets = inputs.limits["GBF3_NVIS"]
    layer = 'ibra' if settings.GBF3_NVIS_REGION_MODE == 'IBRA_REG' else 'nrm'   # the region layer the targets are set on
    region_of_cell = support.region2cell[layer].values                     # the NRM / IBRA code of every cell
    region_code = {name: code for code, name in support.region2cell.attrs[f'{layer}_name'].items()}
    weights = []
    rhs = []
    names = []
    kept = []
    for region, group in inputs.GBF3_NVIS_region_group:
        target = targets.sel(layer=(region, group)).item()
        if target < 0:                                                   # GBF3 still adds a row for a ZERO target
            continue
        weight_row = layers.sel(group=group, drop=True).data
        if region != "AUSTRALIA":                                        # regional scope: mask the cells outside the region
            weight_row = np.where(region_of_cell == region_code[region], weight_row, 0)   # an unknown region raises
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        rhs.append(target)
        names.append(f"bio_GBF3_NVIS_limit_{region}_{group}".replace(" ", "_"))
        kept.append((region, group))
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(inputs.GBF3_NVIS_region_group) - len(kept)} skipped")
    if not weights:
        return None, None
    A, rhs, scale = contract(weight_rows(weights, inputs.ncells) @ bio_S, rhs, rescale=True)
    return make_part('GBF3_NVIS', A, rhs, '>', names, scale, region=[region for region, _ in kept], item=[group for _, group in kept])


def get_GBF4_SNES(inputs: RowInputs, support: ColSupport, bio_S: sparse.csr_matrix):
    """One row per (region, species, presence) with a target > 0 and a cell: Σ_r area[species, r] · (the bio contribution of cell r's columns) ≥ target."""
    if settings.GBF4_TARGET_SNES == 'off':
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 SNES...')
        return None, None
    print("│   │   ├── Adding constraints for biodiversity GBF 4 SNES ...")
    layers = inputs.GBF4_SNES_pre_1750_area_sr                           # xr [layer=(species, presence), cell]
    targets = inputs.limits["GBF4_SNES"]
    region_of_cell = support.region2cell['nrm'].values                     # the NRM code of every cell
    region_code = {name: code for code, name in support.region2cell.attrs['nrm_name'].items()}
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
            weight_row = np.where(region_of_cell == region_code[region], weight_row, 0)   # an unknown region raises
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        rhs.append(target)
        names.append(f"bio_GBF4_SNES_limit_{region}_{species}_{presence}".replace(" ", "_"))
        kept.append((region, species, presence))
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(inputs.GBF4_SNES_region_species) - len(kept)} skipped")
    if not weights:
        return None, None
    A, rhs, scale = contract(weight_rows(weights, inputs.ncells) @ bio_S, rhs, rescale=True)
    return make_part('GBF4_SNES', A, rhs, '>', names, scale,
                     region=[key[0] for key in kept], item=[key[1] for key in kept], presence=[key[2] for key in kept])


def get_GBF4_ECNES(inputs: RowInputs, support: ColSupport, bio_S: sparse.csr_matrix):
    """One row per (region, community, presence) with a target > 0 and a cell: Σ_r area[community, r] · (the bio contribution of cell r's columns) ≥ target."""
    if settings.GBF4_TARGET_ECNES == 'off':
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 ECNES...')
        return None, None
    print("│   │   ├── Adding constraints for biodiversity GBF 4 ECNES ...")
    layers = inputs.GBF4_ECNES_pre_1750_area_sr                          # xr [layer=(community, presence), cell]
    targets = inputs.limits["GBF4_ECNES"]
    region_of_cell = support.region2cell['nrm'].values                     # the NRM code of every cell
    region_code = {name: code for code, name in support.region2cell.attrs['nrm_name'].items()}
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
            weight_row = np.where(region_of_cell == region_code[region], weight_row, 0)   # an unknown region raises
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        rhs.append(target)
        names.append(f"bio_GBF4_ECNES_limit_{region}_{community}_{presence}".replace(" ", "_"))
        kept.append((region, community, presence))
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(inputs.GBF4_ECNES_region_species) - len(kept)} skipped")
    if not weights:
        return None, None
    A, rhs, scale = contract(weight_rows(weights, inputs.ncells) @ bio_S, rhs, rescale=True)
    return make_part('GBF4_ECNES', A, rhs, '>', names, scale,
                     region=[key[0] for key in kept], item=[key[1] for key in kept], presence=[key[2] for key in kept])


def get_GBF8(inputs: RowInputs, support: ColSupport, bio_S: sparse.csr_matrix):
    """One row per (region, species) with a target > 0 and a cell: Σ_r area[species, r] · (the bio contribution of cell r's columns) ≥ target."""
    if settings.GBF8_TARGET == "off":
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 8 ...')
        return None, None
    print("│   │   ├── Adding constraints for biodiversity GBF 8 ...")
    layers = inputs.GBF8_pre_1750_area_sr                                # xr [species, cell]
    targets = inputs.limits["GBF8"]
    region_of_cell = support.region2cell['nrm'].values                     # the NRM code of every cell
    region_code = {name: code for code, name in support.region2cell.attrs['nrm_name'].items()}
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
            weight_row = np.where(region_of_cell == region_code[region], weight_row, 0)   # an unknown region raises
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        rhs.append(target)
        names.append(f"bio_GBF8_limit_{region}_{species}".replace(" ", "_"))
        kept.append((region, species))
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(inputs.GBF8_region_species) - len(kept)} skipped")
    if not weights:
        return None, None
    A, rhs, scale = contract(weight_rows(weights, inputs.ncells) @ bio_S, rhs, rescale=True)
    return make_part('GBF8', A, rhs, '>', names, scale, region=[region for region, _ in kept], item=[species for _, species in kept])


def get_regional_adoption_ag(inputs: RowInputs, cols: xr.Dataset):
    """One row per (region, ag land use) cap: Σ hectares · X_ag over the cap's cells ≤ cap (hectares, NOT rescaled)."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        print("│   │   └── TURNING OFF constraints for regional adoption ...")
        return None, None
    n_all = cols.attrs['n_all']
    in_ag = np.zeros(n_all, dtype=bool)
    in_ag[cols.attrs['block_range']['ag']] = True
    j = cols['j'].values
    cell = cols['cell'].values
    hectares = inputs.real_area[cell].astype(np.float32)                 # the hectares a column's whole share stands for
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits["ag_regional_adoption"]:
        name = f"reg_adopt_limit_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(inputs.ncells, dtype=bool)                                    # the cap's cells ...
        in_region[reg_cells] = True
        on = in_ag & (j == lu_code) & in_region[cell]                                    # ... and the land use's ag columns in them
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(hectares[on])
        rhs.append(area_limit_ha)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not names:
        return None, None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, _ = contract(A, rhs)
    return make_part('regional_adoption_ag', A, rhs, '<', names, region=[reg_id for reg_id, _ in keys], j=[lu_code for _, lu_code in keys])


def get_regional_adoption_nonag(inputs: RowInputs, cols: xr.Dataset):
    """One row per (region, non-ag land use) cap: Σ hectares · X_nonag over the cap's cells ≤ cap · relax."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None, None
    # the caps recede 1e-6/yr RELATIVE, so the RHS always stays ahead of the ratcheting lower bound non-reversible
    # plantings create: last year's solved areas become this year's exact lower bounds, and float32 noise then puts the
    # locked-in floor a hair over a saturated cap, which presolve rejects with NO tolerance (ag caps need none: ag is reversible)
    relax = 1 + (inputs.target_year - settings.SIM_YEARS[0]) * 1e-6
    n_all = cols.attrs['n_all']
    in_nonag = np.zeros(n_all, dtype=bool)
    in_nonag[cols.attrs['block_range']['nonag']] = True
    k = cols['k'].values
    cell = cols['cell'].values
    hectares = inputs.real_area[cell].astype(np.float32)                 # the hectares a column's whole share stands for
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption") or []:
        name = f"reg_adopt_limit_non_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(inputs.ncells, dtype=bool)                                    # the cap's cells ...
        in_region[reg_cells] = True
        on = in_nonag & (k == lu_code) & in_region[cell]                                 # ... and the land use's non-ag columns in them
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(hectares[on])
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not names:
        return None, None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, _ = contract(A, rhs)
    return make_part('regional_adoption_nonag', A, rhs, '<', names, region=[reg_id for reg_id, _ in keys], k=[lu_code for _, lu_code in keys])


def get_regional_adoption_nonag_sum(inputs: RowInputs, cols: xr.Dataset):
    """One row per region cap ('NON_AG_CAP' mode): Σ hectares · X_nonag over EVERY non-ag land use in the cap's cells ≤ cap · relax."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None, None
    # the caps recede 1e-6/yr RELATIVE, so the RHS always stays ahead of the ratcheting lower bound non-reversible
    # plantings create: last year's solved areas become this year's exact lower bounds, and float32 noise then puts the
    # locked-in floor a hair over a saturated cap, which presolve rejects with NO tolerance (ag caps need none: ag is reversible)
    relax = 1 + (inputs.target_year - settings.SIM_YEARS[0]) * 1e-6
    n_all = cols.attrs['n_all']
    in_nonag = np.zeros(n_all, dtype=bool)
    in_nonag[cols.attrs['block_range']['nonag']] = True
    cell = cols['cell'].values
    hectares = inputs.real_area[cell].astype(np.float32)                 # the hectares a column's whole share stands for
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    keys = []
    for reg_id, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption_sum") or []:
        name = f"reg_adopt_limit_non_ag_sum_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(inputs.ncells, dtype=bool)                                    # the cap's cells ...
        in_region[reg_cells] = True
        on = in_nonag & in_region[cell]                                                  # ... and every non-ag column in them
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(hectares[on])
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append(reg_id)
    if not names:
        return None, None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, _ = contract(A, rhs)
    return make_part('regional_adoption_nonag_sum', A, rhs, '<', names, region=keys)


def get_water(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One row per water region: Σ net yield · X over the region's columns ≥ limit (a net yield can be negative)."""
    if settings.WATER_LIMITS != "on":
        print("│   ├── TURNING OFF water usage constraints ...")
        return None, None
    print("│   ├── Adding constraints for water usage limits...")
    n_all = cols.attrs['n_all']
    coeff = gather(cols, support, inputs.ag_w_mrj, inputs.ag_man_w_mrj, inputs.non_ag_w_rk)
    region_of_col = support.region2col['water_region'].values
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    region_ids = []
    for region_id, water_limit_raw in inputs.limits["water"].items():
        region_name = inputs.water_region_names[region_id]
        print(f"│   │   ├── target (inside LUTO study area) is {water_limit_raw:15,.0f} ML for {region_name}")
        on = (region_of_col == region_id) & (coeff != 0)                                  # the region's columns with a net yield
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(coeff[on])
        rhs.append(water_limit_raw)
        names.append(f"water_yield_limit_{region_name}".replace(" ", "_"))
        region_ids.append(region_id)
    if not names:
        return None, None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, scale = contract(A, rhs, rescale=True)
    return make_part('water', A, rhs, '>', names, scale, region=region_ids)


def get_renewable(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One row per (state, type): Σ yield · X_am over the type's columns in the state, outside the excluded cells ≥ target − existing."""
    if not any(settings.RENEWABLES_OPTIONS.values()):
        print("│   ├── TURNING OFF renewable energy constraints ...")
        return None, None
    print("│   ├── Adding constraints for renewable energy production targets ...")
    options = cols.attrs['options']
    re_types = {                                                         # the enabled types only: the options are the enabled ag managements
        option: re_data for option, re_data in {
            'Utility Solar PV': dict(energy_r=inputs.renewable_solar_r, gbf2_mask_idx=inputs.mask_gbf2_solar, mnes_mask_idx=inputs.mask_mnes_solar),
            'Onshore Wind':     dict(energy_r=inputs.renewable_wind_r,  gbf2_mask_idx=inputs.mask_gbf2_wind,  mnes_mask_idx=inputs.mask_mnes_wind),
        }.items() if option in options
    }
    state_code = {name: code for code, name in support.region2col.attrs['state_name'].items()}   # the states the rows are written for ...
    act_code = state_code.pop('Australian Capital Territory')                                # ... ACT folded into NSW below
    n_all        = cols.attrs['n_all']
    cell         = cols['cell'].values
    j            = cols['j'].values
    am_idx       = cols['am_idx'].values
    state_of_col = support.region2col['state'].values
    in_ag        = np.zeros(n_all, dtype=bool)
    in_ag[cols.attrs['block_range']['ag']] = True

    # ── per type: its columns' yield, and the columns the exclusion masks keep out ──
    energy_of_type = {}
    excluded_of_type = {}
    for option, re_data in re_types.items():
        on_type = am_idx == options.index(option)
        energy_of_type[option] = np.where(on_type, re_data['energy_r'][cell], np.float32(0.0)).astype(np.float32)   # float32 yield per cell
        excluded_r = np.zeros(inputs.ncells, dtype=bool)
        if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
            excluded_r[re_data['gbf2_mask_idx']] = True
        if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
            excluded_r[re_data['mnes_mask_idx']] = True
        excluded_of_type[option] = excluded_r[cell]

    # ── one row per (state, type) with an allowed column of the type's host land uses ──
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    key_am = []
    key_state = []
    for state_name, code in state_code.items():
        in_state = state_of_col == code
        if state_name == 'New South Wales':                              # ACT counts toward the NSW+ACT target
            in_state |= state_of_col == act_code
        print(f"│   │   ├── Adding renewable energy constraints for {state_name} ...")
        for am in re_types:
            target_raw = inputs.limits[f"renewable_{am}"][state_name]
            exist_power_mwh = inputs.limits[f"renewable_{am}_exist"][state_name]
            print(f"│   │   │   ├── target for {am} is {target_raw:5,.0f} MWh  (existing: {exist_power_mwh:5,.0f} MWh)")
            allowed = in_state & ~excluded_of_type[am]                   # the state's columns outside the excluded cells
            # row-inclusion rule (NOT a coefficient test): the row exists iff an allowed cell holds an ag column of a
            # compatible land use — even if every coefficient there turns out to be sub-floor
            if not (allowed & in_ag & np.isin(j, inputs.agman2lu[am])).any():
                continue
            on = allowed & (energy_of_type[am] != 0)                     # the type's columns in the state, with a yield
            row_idx.append(np.full(int(on.sum()), len(names)))
            col_idx.append(np.flatnonzero(on))
            vals.append(energy_of_type[am][on])
            rhs.append(target_raw - exist_power_mwh)                     # raw MWh; row-rescaled below
            names.append(f"renewable_{am}_target_{state_name}".replace(" ", "_"))
            key_am.append(options.index(am))
            key_state.append(state_name)
    if not names:
        return None, None
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), n_all))
    A, rhs, scale = contract(A, rhs, rescale=True)
    return make_part('renewable', A, rhs, '>', names, scale, am_idx=key_am, state=key_state)


# ═══════════════════════════ the flow rows ═══════════════════════════
#
#   group the arcs by their source's place in the base grid   →  source cap:   Σ out ≤ base              (a ≤ row per source node)
#   look every column and arc up on the grid of its node       →  node balance: X = base + Σ in − Σ out   (an = row per node)
#   the inflow cap is not a row: X's own ub (the transition upper bound) bounds it.
#   a cell's total is not a row either: every arc leaves one node of the cell and lands on another, so the cell's
#   node-balance rows sum to Σ X = Σ base (row_bounds forms that sum to bound a cell's columns together).

def get_source_cap_ag(cols: xr.Dataset, support: ColSupport):
    """One row per ag source (lm, lu, cell): Σ of the arcs leaving it (ag2ag ∪ ag2nonag) ≤ its base share."""
    # bounds the arc columns (some flow costs are negative) and rules out pass-through
    print("│   ├── Adding source-cap (Σ out ≤ base) constraints...")
    n_all       = cols.attrs['n_all']
    block_range = cols.attrs['block_range']
    arcs        = np.r_[block_range['ag2ag'], block_range['ag2nonag']]   # the arcs leaving an ag node
    if not arcs.size:
        return None, None
    from_m      = cols['from_m'].values[arcs]
    from_j      = cols['from_j'].values[arcs]
    local_r     = cols['local_r'].values[arcs]
    cell        = cols['cell'].values[arcs]
    # one row per source, ascending in (lm, lu, cell) order; every arc leaving it gets a +1
    nlms, ncells, n_lu_ag = support.ag_mrj2col.shape
    source = np.ravel_multi_index((from_m, from_j, cell), (nlms, n_lu_ag, ncells))                  # the flat position of each arc's source in (lm, lu, cell)
    _, first_arc, row_of_arc = np.unique(source, return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(first_arc.size, n_all))
    # the cap is the source's base on the table: a source is a holding above the noise floor, and the ag ub is raised
    # to that same floor, so every source has a column
    source_col = support.ag_mrj2col[from_m[first_arc], cell[first_arc], from_j[first_arc]]
    assert (source_col >= 0).all(), 'an ag source without an ag column: the source map and the ub floor disagree'
    rhs = cols['base'].values[source_col]
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_a_{m}_{j}_{r}" for m, j, r in zip(from_m[first_arc], from_j[first_arc], local_r[first_arc])]
    return make_part('source_cap_ag', A, rhs, '<', names, from_m=from_m[first_arc], from_j=from_j[first_arc], local_r=local_r[first_arc])


def get_source_cap_nonag(cols: xr.Dataset, support: ColSupport):
    """One row per non-ag source (nonag_lu, cell): Σ of the arcs leaving it (nonag2ag) ≤ its base share."""
    n_all   = cols.attrs['n_all']
    arcs    = np.r_[cols.attrs['block_range']['nonag2ag']]                                      # the arcs leaving a non-ag node
    if not arcs.size:
        return None, None
    from_k  = cols['from_k'].values[arcs]
    local_r = cols['local_r'].values[arcs]
    cell    = cols['cell'].values[arcs]
    ncells, n_k = support.nonag_rk2col.shape
    source = np.ravel_multi_index((from_k, cell), (n_k, ncells))                                # the flat position of each arc's source in (nonag_lu, cell): one row per source, ascending
    _, first_arc, row_of_arc = np.unique(source, return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(first_arc.size, n_all))
    # the cap is the source's base on the table: a source is a holding above the noise floor, and the non-ag ub is
    # raised to that same floor, so every source has a column
    source_col = support.nonag_rk2col[cell[first_arc], from_k[first_arc]]
    assert (source_col >= 0).all(), 'a non-ag source without a non-ag column: the source map and the ub floor disagree'
    rhs = cols['base'].values[source_col]
    A, rhs, _ = contract(A, rhs)
    names = [f"srccap_n_{k}_{r}" for k, r in zip(from_k[first_arc], local_r[first_arc])]
    return make_part('source_cap_nonag', A, rhs, '<', names, from_k=from_k[first_arc], local_r=local_r[first_arc])


def get_node_balance_ag(cols: xr.Dataset, support: ColSupport):
    """One row per ag column: X_ag = base + Σ in (ag2ag ∪ nonag2ag) − Σ out (ag2ag ∪ ag2nonag)."""
    print("│   ├── Adding node-balance (X = base + Σin − Σout) constraints at the ag nodes...")
    n_all       = cols.attrs['n_all']
    m           = cols['m'].values
    j           = cols['j'].values
    from_m      = cols['from_m'].values
    from_j      = cols['from_j'].values
    cell        = cols['cell'].values
    block_range = cols.attrs['block_range']
    ag          = block_range['ag']

    # ── the rows: one per ag column ──
    n_ag = ag.stop - ag.start
    ag_m, ag_j, ag_r = m[ag], j[ag], cell[ag]

    def ag_row(m_, j_, r_):
        """The row of the ag node (m, j, r): its column's place in the ag block, -1 where it has no column."""
        col = support.ag_mrj2col[m_, r_, j_]
        return np.where(col >= 0, col - ag.start, -1)

    # ── the entries: X on its own row, inflows −1 on the target's row, outflows +1 on the source's row ──
    row_idx = []
    col_idx = []
    vals = []

    def add(row, col, value):
        in_model = row >= 0                                              # no row (banned source / no X var): entry dropped
        row_idx.append(row[in_model].astype(np.int64))
        col_idx.append(col[in_model].astype(np.int64))
        vals.append(np.full(int(in_model.sum()), value))

    add(np.arange(n_ag), np.arange(ag.start, ag.stop), 1.0)              # X_ag on its own row
    arcs = np.r_[block_range['ag2ag']]                              # ag → ag: in on the target's row, out of the source's
    add(ag_row(m[arcs], j[arcs], cell[arcs]), arcs, -1.0)
    add(ag_row(from_m[arcs], from_j[arcs], cell[arcs]), arcs, 1.0)
    arcs = np.r_[block_range['ag2nonag']]                           # ag → non-ag: out of the source's row
    add(ag_row(from_m[arcs], from_j[arcs], cell[arcs]), arcs, 1.0)
    arcs = np.r_[block_range['nonag2ag']]                           # non-ag → ag: in on the target's row
    add(ag_row(m[arcs], j[arcs], cell[arcs]), arcs, -1.0)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_ag, n_all))

    # ── rhs, names, keys ──
    A, rhs, _ = contract(A, cols['base'].values[ag])
    names = [f"bal_a_{m}_{j}_{r}" for m, j, r in zip(ag_m, ag_j, ag_r)]
    return make_part('node_balance_ag', A, rhs, '=', names, m=ag_m, j=ag_j, cell=ag_r)


def get_node_balance_nonag(cols: xr.Dataset, support: ColSupport):
    """One row per non-ag column: X_nonag = base + Σ in (ag2nonag) − Σ out (nonag2ag)."""
    print("│   └── Adding node-balance (X = base + Σin − Σout) constraints at the non-ag nodes...")
    n_all       = cols.attrs['n_all']
    k           = cols['k'].values
    from_k      = cols['from_k'].values
    cell        = cols['cell'].values
    block_range = cols.attrs['block_range']
    nonag       = block_range['nonag']

    # ── the rows: one per non-ag column ──
    n_nonag = nonag.stop - nonag.start
    if not n_nonag:
        return None, None
    nonag_k, nonag_r = k[nonag], cell[nonag]

    def nonag_row(k_, r_):
        """The row of the non-ag node (k, r): its column's place in the non-ag block, -1 where it has no column."""
        col = support.nonag_rk2col[r_, k_]
        return np.where(col >= 0, col - nonag.start, -1)

    # ── the entries: X on its own row, inflows −1 on the target's row, outflows +1 on the source's row ──
    row_idx = []
    col_idx = []
    vals = []

    def add(row, col, value):
        in_model = row >= 0                                              # no row (the node has no column): entry dropped
        row_idx.append(row[in_model].astype(np.int64))
        col_idx.append(col[in_model].astype(np.int64))
        vals.append(np.full(int(in_model.sum()), value))

    add(np.arange(n_nonag), np.arange(nonag.start, nonag.stop), 1.0)     # X_nonag on its own row
    arcs = np.r_[block_range['ag2nonag']]                           # ag → non-ag: in on the target's row
    add(nonag_row(k[arcs], cell[arcs]), arcs, -1.0)
    arcs = np.r_[block_range['nonag2ag']]                           # non-ag → ag: out of the source's row
    add(nonag_row(from_k[arcs], cell[arcs]), arcs, 1.0)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_nonag, n_all))

    # ── rhs, names, keys ──
    A, rhs, _ = contract(A, cols['base'].values[nonag])
    names = [f"bal_n_{k}_{r}" for k, r in zip(nonag_k, nonag_r)]
    return make_part('node_balance_nonag', A, rhs, '=', names, k=nonag_k, cell=nonag_r)
