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

from scipy import sparse

import luto.tools as tools

from luto import settings
from luto.solvers.col_builder import ColSupport
from luto.solvers.row_inputs import EconomicInputs, RowInputs
from luto.solvers.row_table import ROW_FILL, ROW_SCHEMA, make_part


# ═══════════════════════════ get_rows: the row space of one step ═══════════════════════════

def get_rows(inputs: RowInputs, cols: xr.Dataset, support: ColSupport) -> tuple[sparse.csr_matrix, xr.Dataset]:
    """The row space of every family as its block of the row matrix and its table."""

    parts = [
        get_demand(inputs, cols, support),
        get_ghg(inputs, cols),
        get_ag_mgt_adoption(inputs, cols),
        get_regional_adoption_ag(inputs, cols),
        get_regional_adoption_nonag(inputs, cols),
        get_regional_adoption_nonag_sum(inputs, cols),
        get_water(inputs, cols, support),
        get_renewable(inputs, cols, support),
        get_ag_mgt_link(inputs, cols, support),
        get_renewable_ceiling(inputs, cols, support),
        get_source_cap_ag(cols, support),
        get_source_cap_nonag(cols, support),
        get_node_balance_ag(cols, support),
        get_node_balance_nonag(cols, support)
    ]


    # the biodiversity families, after the rest: where row generation appends its rows too
    bio_on = any(target != 'off' for target in (settings.GBF2_TARGET, settings.GBF3_NVIS_TARGET, settings.GBF4_TARGET_SNES,
                                                settings.GBF4_TARGET_ECNES, settings.GBF8_TARGET))
    if bio_on and settings.BIO_ROW_GENERATION:                             # the rows are generated during the solve (rowgen)
        print('│   ├── The biodiversity targets by row generation: no row built up front ...')
    elif bio_on:                                                           # every family: its targets, then the ONE builder
        bio_S = support.cell2col @ sparse.diags(bio_contribution(inputs, cols))   # (cell x column) @ (column x column) = (cell x column)
        parts += [build_GBF_rows('GBF2', get_GBF2_targets(inputs), bio_S)]
        parts += [build_GBF_rows('GBF3_NVIS', get_GBF3_NVIS_targets(inputs, support), bio_S)]
        parts += [build_GBF_rows('GBF4_SNES', get_GBF4_SNES_targets(inputs, support), bio_S)]
        parts += [build_GBF_rows('GBF4_ECNES', get_GBF4_ECNES_targets(inputs, support), bio_S)]
        parts += [build_GBF_rows('GBF8', get_GBF8_targets(inputs, support), bio_S)]


    # the space: every family's block into the ONE matrix and its table into the ONE row table, in the order above
    #       (the model's row order); a family that is off returned (None, None) and is left out ──
    row_matrix = sparse.vstack([block for block, _ in parts if block is not None], format='csr')
    rows = xr.concat(
        [ROW_SCHEMA, *[table for _, table in parts if table is not None]],
        dim='row',
        fill_value=ROW_FILL
    )
    print(f"│   └── Row space: {row_matrix.shape[0]:,} rows, {row_matrix.nnz:,} nonzeros", flush=True)

    return row_matrix, rows


# ═══════════════════════════ the coefficient contract: gather → weigh → contract ═══════════════════════════

def gather(cols: xr.Dataset, inputs: RowInputs, ag_coeff_mrj, am_coeff_mrj: dict, nonag_coeff_rk,
           ag2ag_coeff: dict = None, ag2nonag_coeff: dict = None, nonag2ag_coeff: dict = None) -> np.ndarray:
    """
    Create a row of coefficients over the columns: every column reads its block's input at its own fields.
    """
    block   = cols['block'].values
    m       = cols['m'].values
    j       = cols['j'].values
    k       = cols['k'].values
    am_idx  = cols['am_idx'].values
    j_idx   = cols['j_idx'].values
    from_m  = cols['from_m'].values
    from_j  = cols['from_j'].values
    from_k  = cols['from_k'].values
    local_r = cols['local_r'].values
    cell    = cols['cell'].values
    coeff   = np.zeros(cols.sizes['col'], dtype=np.float32)   # the row: one coefficient per column

    # assign ag/nonag coefficients
    on = block == 'ag'
    coeff[on] = ag_coeff_mrj[m[on], cell[on], j[on]]
    
    on = block == 'nonag'
    coeff[on] = nonag_coeff_rk[cell[on], k[on]]

    # assign ag-man coefficients
    am = np.flatnonzero(block == 'am')
    for option in inputs.agman2lu:
        on = am[am_idx[am] == inputs.am_options.index(option)]
        coeff[on] = am_coeff_mrj[option][m[on], cell[on], j_idx[on]]

    # assign ag2ag arc coefficients: a source's arcs at their (to_m, local_r, to_j)
    if ag2ag_coeff is not None:
        arcs = np.flatnonzero(block == 'ag2ag')
        for (src_m, src_j), coeff_src in ag2ag_coeff.items():
            on = arcs[(from_m[arcs] == src_m) & (from_j[arcs] == src_j)]
            coeff[on] = coeff_src[m[on], local_r[on], j[on]]

    # assign ag2nonag arc coefficients: a source's arcs at their (local_r, to_k)
    if ag2nonag_coeff is not None:
        arcs = np.flatnonzero(block == 'ag2nonag')
        for (src_m, src_j), coeff_src in ag2nonag_coeff.items():
            on = arcs[(from_m[arcs] == src_m) & (from_j[arcs] == src_j)]
            coeff[on] = coeff_src[local_r[on], k[on]]

    # assign nonag2ag arc coefficients: a source's arcs at their (to_m, local_r, to_j)
    if nonag2ag_coeff is not None:
        arcs = np.flatnonzero(block == 'nonag2ag')
        for src_k, coeff_src in nonag2ag_coeff.items():
            on = arcs[from_k[arcs] == src_k]
            coeff[on] = coeff_src[m[on], local_r[on], j[on]]

    return coeff


def weight_rows(weights, ncells: int) -> sparse.csr_matrix:
    """Apply weights to a row space.
    1. Used as a mask to turn on/off cells (GBF2 priority areas: the mask × each cell's area)
    2. Used as the area each cell holds of a target (GBF3 NVIS vegetation groups, GBF4 SNES species / ECNES
       communities, GBF8 species), zeroed outside the target's region
    Multiplied by the biodiversity contribution laid on the support (``W @ bio_S``), the rows go over the columns.
    """
    indptr = [0]
    indices = []
    data = []
    for weight_row in weights:
        weight_row = np.asarray(weight_row, dtype=np.float32)
        cells = np.flatnonzero(weight_row)                                 # NaN counts as nonzero: it reaches ``contract``, which drops it
        indices.append(cells)
        data.append(weight_row[cells])
        indptr.append(indptr[-1] + cells.size)
    return sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr, dtype=np.int64)),
        shape=(len(weights), ncells)
    )


def contract(block: sparse.csr_matrix, rhs=None, rescale: bool = False) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """
    1. Drop tiny coefficients (abs < SOLVER_COEFF_MIN) and eliminate zeros.
    2. Sort the indices of every row (Gurobi requires it).
    3. If ``rescale`` is True, geometrically rescale every row and the RHS.
    The block is modified in place (every caller hands over a freshly built block), and returned with float32 coefficients.
    """

    block = block.tocsr()                   # a CSR block is returned as it is: no copy
    # Drop tiny coefficients and eliminate zeros
    block.data[~(np.abs(block.data) >= settings.SOLVER_COEFF_MIN)] = 0.0   # the drop; NaN fails the test
    block.eliminate_zeros()
    # Sort the indices of every row
    block.sort_indices()
    
    # Skip rescaling if not requested
    rhs = None if rhs is None else np.asarray(rhs, dtype=np.float64)
    if not rescale:
        block.data = block.data.astype(np.float32, copy=False)             # float32 like the rescaled rows: one float64 block would upcast the whole stacked row_matrix
        return block, rhs, np.ones(block.shape[0], dtype=np.float64)

    # the two magnitudes of every row: its largest coefficient and its rhs
    has_entry = np.diff(block.indptr) > 0
    row_max = np.zeros(block.shape[0], dtype=np.float64)                    # 0 for an empty row
    row_max[has_entry] = np.maximum.reduceat(np.abs(block.data), block.indptr[:-1][has_entry])   # off the stored entries: no second matrix
    rhs_abs = np.abs(rhs)

    # the row's factor: the geometric mean of the two, so both land around RESCALE_FACTOR
    scale = np.sqrt(row_max * rhs_abs)
    scale[rhs_abs == 0] = row_max[rhs_abs == 0]                            # rhs = 0: the coefficients alone set it
    scale[row_max == 0] = settings.RESCALE_FACTOR                          # empty row: factor 1, the rhs is kept as it is
    scale /= settings.RESCALE_FACTOR

    # divide every entry by its row's factor
    row_of_entry = np.repeat(np.arange(block.shape[0]), np.diff(block.indptr))   # the row each stored entry sits in
    block.data /= scale[row_of_entry]                                      # in place, in the data's own precision ...
    block.data = block.data.astype(np.float32, copy=False)                 # ... then float32 (no copy when it already is)
    block.data[np.abs(block.data) < settings.SOLVER_COEFF_MIN] = 0.0       # floor the scaled row
    block.eliminate_zeros()
    block.sort_indices()
    return block, rhs / scale, scale


def bio_contribution(inputs: RowInputs, cols: xr.Dataset) -> np.ndarray:
    """Get the biodiversity contribution of every column"""
    # Ag contribution
    ag_j = np.asarray(inputs.biodiv_contr_ag_j, dtype=np.float32)
    ag_contr_mrj = np.broadcast_to(ag_j[None, None, :], (inputs.nlms, inputs.ncells, ag_j.size))
    # Non-ag contribution
    nonag_k = np.array([inputs.biodiv_contr_non_ag_k.get(k, 0.0) for k in range(inputs.n_nonag_lus)], dtype=np.float32)
    nonag_contr_rk = np.broadcast_to(nonag_k[None, :], (inputs.ncells, nonag_k.size))
    # Agricultural management contribution
    am_contr_mrj = {}
    for option, by_j_idx in inputs.biodiv_contr_ag_man.items():
        per_cell = np.stack([np.asarray(by_j_idx[j_idx], dtype=np.float32) for j_idx in range(len(by_j_idx))], axis=1)
        am_contr_mrj[option] = np.broadcast_to(per_cell[None, :, :], (inputs.nlms, inputs.ncells, per_cell.shape[1]))
    return gather(cols, inputs, ag_contr_mrj, am_contr_mrj, nonag_contr_rk)


# ═══════════════════════════ get_obj: the objective coefficient of every column ═══════════════════════════

def get_obj(econ: EconomicInputs, cols: xr.Dataset, inputs: RowInputs) -> np.ndarray:
    """The objective coefficient of every column (on ``col``), as Gurobi takes it: the operating economics on
    the accounting columns and the transition costs, negated, on the arcs — raw AUD, float32 — through the
    coefficient contract (the SOLVER_COEFF_MIN drop), then scaled to million AUD and floored again, because
    the scaling can push a coefficient under the floor."""
    # ── one gather over the six blocks: the operating economics on the accounting columns, the transition costs on the arcs ──
    obj = gather(cols, inputs, econ.ag_obj_mrj, econ.ag_man_objs, econ.non_ag_obj_rk,
                 econ.flow_cost_ag2ag, econ.flow_cost_ag2nonag, econ.flow_cost_nonag2ag)
    arcs = np.isin(cols['block'].values, ('ag2ag', 'ag2nonag', 'nonag2ag'))   # the arc blocks: a transition is a COST, so it enters negated
    obj[arcs] = -obj[arcs]

    # ── the contract: the drop on the raw coefficient, the scaling to million AUD, the floor on the scaled one ──
    obj[~(np.abs(obj) >= settings.SOLVER_COEFF_MIN)] = 0.0                     # the drop; NaN fails the test too
    obj = obj * (1.0 / 1e6)                                                    # raw AUD -> million AUD (float32, a reciprocal multiply as gurobipy did)
    obj[np.abs(obj) < settings.SOLVER_COEFF_MIN] = 0.0                         # floor the scaled coefficient
    return obj



def get_demand(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """
    Add `N_products` rows, each represeting a commodity's demand constraint.
    """
    print("│   ├── Adding demand constraints (equality where lb==ub, else lower + upper)...")

    # Get the active products of every ag land use: {ag_idx: [product_idx, ...]} — the unallocated mod/nat are skipped
    lu2pr = {lu: p for lu in range(inputs.n_ag_lus) if (p := np.flatnonzero(inputs.lu2pr_pj[:, lu])).size}
    
    # Containers for the production block in COO format.
    row_idx = []
    col_idx = []
    vals = []

    def put(commodity_coeffs: np.ndarray, cell2colspace_r: np.ndarray):
        """One (commodity × column) coefficient block into the COO lists — its nonzero support; the contract drops."""
        block = sparse.coo_matrix(commodity_coeffs)                      # row = the commodity, col = the cells indexes
        row_idx.append(block.row)
        col_idx.append(cell2colspace_r[block.col])                       # cell position -> column-space index
        vals.append(block.data)

    # the production for ag
    for lu, p in lu2pr.items():
        for lm in range(inputs.nlms):
            cells = np.flatnonzero(support.valid_ag_mrj[lm, :, lu])
            # (commodity × products) @ (products × cell) = (commodity × cell)
            commodity_coeffs = inputs.pr2cm_cp[:, p] @ inputs.ag_q_mrp[lm, cells, :][:, p].T
            cell2colspace_r = support.ag_mrj2col[lm, cells, lu]          # the column-space index of every cell in `cells`
            put(commodity_coeffs, cell2colspace_r)

    # the production for ag-mgt: per (option, land use, lm), its columns and their cells read off the table
    am     = np.flatnonzero(cols['block'].values == 'am')
    am_idx = cols['am_idx'].values[am]
    j_idx  = cols['j_idx'].values[am]
    am_m   = cols['m'].values[am]
    am_r   = cols['cell'].values[am]
    for option, lus in inputs.agman2lu.items():
        option_idx = inputs.am_options.index(option)                     # the option's fixed am_idx
        for lu_idx, lu in enumerate(lus):
            if lu not in lu2pr:  # Unallocated mod/nat land has no products
                continue
            p = lu2pr[lu]
            for lm in range(inputs.nlms):
                on = (am_idx == option_idx) & (j_idx == lu_idx) & (am_m == lm)
                cells = am_r[on]
                # (commodity × products) @ (products × cell) = (commodity × cell)
                commodity_coeffs = inputs.pr2cm_cp[:, p] @ inputs.ag_man_q_mrp[option][lm, cells, :][:, p].T
                cell2colspace_r = am[on]                                 # the column-space index of every cell in `cells`
                put(commodity_coeffs, cell2colspace_r)

    # the production for non-ag
    for lu in range(inputs.n_nonag_lus):
        cells = np.flatnonzero(support.valid_nonag_rk[:, lu])
        cell2colspace_r = support.nonag_rk2col[cells, lu]               # the column-space index of every cell in `cells`
        put(inputs.non_ag_q_crk[:, cells, lu], cell2colspace_r)
    
    q_block = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))),
        shape=(inputs.ncms, cols.sizes['col'])
    )
    q_block, _, _ = contract(q_block)

    # ── the bound rows ──
    commodity = []                                                       # the row's commodity: which production row it copies
    senses = []
    rhs = []
    names = []
    key_commodity = []
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
            key_commodity.append(c_name)
            key_bound.append(bound)
    row_matrix, rhs, scale = contract(q_block[commodity], rhs, rescale=True)  # the commodity's production row under each of its bounds; row rescale, factors kept
    return make_part('demand', row_matrix, rhs, np.array(senses, dtype=object), names, scale, demand_commodity=key_commodity, demand_bound=key_bound)



def get_ghg(inputs: RowInputs, cols: xr.Dataset):
    """One global row: Σ ghg · X over the ag, ag-mgt, non-ag columns and the ag → ag arcs ≤ limit − offland."""
    if settings.GHG_EMISSIONS_LIMITS == "off":
        print("│   ├── TURNING OFF GHG emissions constraints ...")
        return None, None
    print(f"│   ├── Adding constraints for GHG emissions: {inputs.limits['ghg']:,.0f} tCO2e")

    # land-use, ag-management and non-ag emissions on the accounting columns, transition emissions on the ag → ag arcs
    coeff = gather(cols, inputs, inputs.ag_g_mrj, inputs.ag_man_g_mrj, inputs.non_ag_g_rk, ag2ag_coeff=inputs.trans_ghg_ag2ag)
    row = sparse.csr_matrix(coeff[None, :])                              # the nonzero support; the contract drops the rest
    rhs = np.asarray(inputs.limits["ghg"] - inputs.offland_ghg, dtype=np.float64).ravel()   # offland_ghg: 1-element array
    row_matrix, rhs, scale = contract(row, rhs, rescale=True)            # drop + row rescale, factor kept
    return make_part('ghg', row_matrix, rhs, '<', ["ghg_emissions_limit_ub"], scale)


def get_bio_targets(inputs: RowInputs, support: ColSupport) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Every family's targets as ONE table (a ``family`` column added; a label column a family has not is None) and
    ONE (target × cell) weight matrix, in the model's row order — what row generation (``rowgen``) works from."""
    tables, weights = [], []
    for family, got in (('GBF2', get_GBF2_targets(inputs)),
                        ('GBF3_NVIS', get_GBF3_NVIS_targets(inputs, support)),
                        ('GBF4_SNES', get_GBF4_SNES_targets(inputs, support)),
                        ('GBF4_ECNES', get_GBF4_ECNES_targets(inputs, support)),
                        ('GBF8', get_GBF8_targets(inputs, support))):
        if got is None:
            continue
        targets, W = got
        tables.append(targets.assign(family=family))
        weights.append(W)
    assert tables, 'row generation with no biodiversity target'
    targets = pd.concat(tables, ignore_index=True)
    for field in ('region', 'GBF_target', 'GBF4_presence'):
        targets[field] = targets[field].astype(object).where(targets[field].notna(), None) if field in targets else None
    return targets, sparse.vstack(weights, format='csr')


def build_GBF_rows(family: str, family_targets, bio_S: sparse.csr_matrix):
    """A family's rows from its targets (those with `row`): W @ bio_S, contracted and rescaled row by row — the same for
    every family. Every column of the table beside `rhs`, `name` and `row` is a label of the rows. (None, None) for a
    family with no row."""
    if family_targets is None:
        return None, None
    targets, W = family_targets
    on = np.flatnonzero(targets['row'].values)                             # the targets the family builds a row for
    if on.size == 0:
        return None, None
    targets = targets.iloc[on]
    row_matrix, rhs, scale = contract(W[on] @ bio_S, targets['rhs'].values.astype(np.float64), rescale=True)
    labels = {field: targets[field].values for field in targets.columns if field not in ('rhs', 'name', 'row')}
    return make_part(family, row_matrix, rhs, '>', targets['name'].tolist(), scale, **labels)


def get_GBF2_targets(inputs: RowInputs):
    """One target: Σ_r mask_area[r] · (the bio contribution of cell r's columns) ≥ target."""
    if settings.GBF2_TARGET == "off":
        print("│   ├── TURNING OFF constraints for biodiversity GBF 2...")
        return None
    print(f'│   ├── Adding constraints for biodiversity GBF 2: target {inputs.limits["GBF2"]:,.0f} ha')
    targets = pd.DataFrame({'rhs': [inputs.limits["GBF2"]], 'name': ["bio_GBF2_priority_degraded_area_limit"], 'row': [True]})
    return targets, weight_rows([inputs.GBF2_mask_area_r], inputs.ncells)


def get_GBF3_NVIS_targets(inputs: RowInputs, support: ColSupport):
    """One target per (region, group) with a cell: W = the pre-1750 area of the group; IBRA bioregions in 'IBRA_REG' mode.
    A row is built where the level is ≥ 0 (GBF3 keeps a row at a ZERO level; a negative one is met by the land outside LUTO)."""
    if settings.GBF3_NVIS_TARGET == "off":
        print("│   ├── TURNING OFF constraints for biodiversity GBF 3 NVIS")
        return None
    print("│   ├── Adding constraints for biodiversity GBF 3 NVIS...")
    layer = 'ibra' if settings.GBF3_NVIS_REGION_MODE == 'IBRA_REG' else 'nrm'   # the region layer the targets are set on
    regions = set(support.region2cell[layer].values)   # the regions the layer knows
    weights = []
    records = []
    for region, group in inputs.GBF3_NVIS_region_group:
        level = inputs.limits["GBF3_NVIS"].sel(layer=(region, group)).item()
        weight_row = inputs.GBF3_NVIS_pre_1750_area_vr.sel(group=group, drop=True).data   # xr [group, cell]
        if region != "AUSTRALIA":                                        # regional scope: mask the cells outside the region
            if region not in regions:
                raise KeyError(f'target region {region!r} is not a region of the layer')
            weight_row = np.where(support.region2cell[layer].values == region, weight_row, 0)
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        records.append((region, group, level, f"bio_GBF3_NVIS_limit_{region}_{group}".replace(" ", "_"), level >= 0))
    print(f"│   │   └── {sum(r[-1] for r in records)} constraint(s) added, {len(inputs.GBF3_NVIS_region_group) - sum(r[-1] for r in records)} skipped (no cell, or a level the land outside LUTO meets)")
    if not weights:
        return None
    return pd.DataFrame(records, columns=['region', 'GBF_target', 'rhs', 'name', 'row']), weight_rows(weights, inputs.ncells)


def get_GBF4_SNES_targets(inputs: RowInputs, support: ColSupport):
    """One target per (region, species, presence) with a cell: W = the area of the species. A row is built where the level
    is > 0 (a level the land outside LUTO already meets gets none)."""
    if settings.GBF4_TARGET_SNES == 'off':
        print('│   ├── TURNING OFF constraints for biodiversity GBF 4 SNES...')
        return None
    print("│   ├── Adding constraints for biodiversity GBF 4 SNES ...")
    regions = set(support.region2cell['nrm'].values)   # the regions the layer knows
    weights = []
    records = []
    for region, species, presence in inputs.GBF4_SNES_region_species:
        level = inputs.limits["GBF4_SNES"].sel(layer=(region, species, presence)).item()
        weight_row = inputs.GBF4_SNES_pre_1750_area_sr.sel(layer=(species, presence), drop=True).values   # xr [layer=(species, presence), cell]
        if region != "AUSTRALIA":                                        # NRM scope: mask the cells outside the region
            if region not in regions:
                raise KeyError(f'target region {region!r} is not a region of the layer')
            weight_row = np.where(support.region2cell['nrm'].values == region, weight_row, 0)
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        records.append((region, species, presence, level, f"bio_GBF4_SNES_limit_{region}_{species}_{presence}".replace(" ", "_"), level > 0))
    print(f"│   │   └── {sum(r[-1] for r in records)} constraint(s) added, {len(inputs.GBF4_SNES_region_species) - sum(r[-1] for r in records)} skipped (no cell, or a level the land outside LUTO meets)")
    if not weights:
        return None
    return pd.DataFrame(records, columns=['region', 'GBF_target', 'GBF4_presence', 'rhs', 'name', 'row']), weight_rows(weights, inputs.ncells)


def get_GBF4_ECNES_targets(inputs: RowInputs, support: ColSupport):
    """One target per (region, community, presence) with a cell: W = the area of the community. A row is built where the level
    is > 0 (a level the land outside LUTO already meets gets none)."""
    if settings.GBF4_TARGET_ECNES == 'off':
        print('│   ├── TURNING OFF constraints for biodiversity GBF 4 ECNES...')
        return None
    print("│   ├── Adding constraints for biodiversity GBF 4 ECNES ...")
    regions = set(support.region2cell['nrm'].values)   # the regions the layer knows
    weights = []
    records = []
    for region, community, presence in inputs.GBF4_ECNES_region_species:
        level = inputs.limits["GBF4_ECNES"].sel(layer=(region, community, presence)).item()
        weight_row = inputs.GBF4_ECNES_pre_1750_area_sr.sel(layer=(community, presence), drop=True).values   # xr [layer=(community, presence), cell]
        if region != "AUSTRALIA":                                        # NRM scope: mask the cells outside the region
            if region not in regions:
                raise KeyError(f'target region {region!r} is not a region of the layer')
            weight_row = np.where(support.region2cell['nrm'].values == region, weight_row, 0)
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        records.append((region, community, presence, level, f"bio_GBF4_ECNES_limit_{region}_{community}_{presence}".replace(" ", "_"), level > 0))
    print(f"│   │   └── {sum(r[-1] for r in records)} constraint(s) added, {len(inputs.GBF4_ECNES_region_species) - sum(r[-1] for r in records)} skipped (no cell, or a level the land outside LUTO meets)")
    if not weights:
        return None
    return pd.DataFrame(records, columns=['region', 'GBF_target', 'GBF4_presence', 'rhs', 'name', 'row']), weight_rows(weights, inputs.ncells)


def get_GBF8_targets(inputs: RowInputs, support: ColSupport):
    """One target per (region, species) with a cell: W = suitability × LDS × area of the species. A row is built where
    the level is > 0 (a level the land outside LUTO already meets gets none — most species at a low level)."""
    if settings.GBF8_TARGET == "off":
        print('│   ├── TURNING OFF constraints for biodiversity GBF 8 ...')
        return None
    print("│   ├── Adding constraints for biodiversity GBF 8 ...")
    regions = set(support.region2cell['nrm'].values)   # the regions the layer knows
    weights = []
    records = []
    for region, species in inputs.GBF8_region_species:
        level = inputs.limits["GBF8"].sel(layer=(region, species)).item()
        weight_row = inputs.GBF8_pre_1750_area_sr.sel(species=species, drop=True).data   # xr [species, cell]
        if region != "AUSTRALIA":                                        # NRM scope: mask the cells outside the region
            if region not in regions:
                raise KeyError(f'target region {region!r} is not a region of the layer')
            weight_row = np.where(support.region2cell['nrm'].values == region, weight_row, 0)
        if not (weight_row > 0).any():
            continue
        weights.append(weight_row)
        records.append((region, species, level, f"bio_GBF8_limit_{region}_{species}".replace(" ", "_"), level > 0))
    print(f"│   │   └── {sum(r[-1] for r in records)} constraint(s) added, {len(inputs.GBF8_region_species) - sum(r[-1] for r in records)} skipped (no cell, or a level the land outside LUTO meets)")
    if not weights:
        return None
    return pd.DataFrame(records, columns=['region', 'GBF_target', 'rhs', 'name', 'row']), weight_rows(weights, inputs.ncells)


def get_ag_mgt_adoption(inputs: RowInputs, cols: xr.Dataset):
    """One row per (option, land use): Σ X_am − limit · Σ X_ag ≤ 0."""
    print("│   ├── Adding ag-management adoption-limit constraints...")
    am      = np.flatnonzero(cols['block'].values == 'am')
    ag      = np.flatnonzero(cols['block'].values == 'ag')
    am_idx  = cols['am_idx'].values[am]
    j_idx   = cols['j_idx'].values[am]
    ag_j    = cols['j'].values[ag]
    row_idx = []
    col_idx = []
    vals = []
    names = []
    key_am = []
    key_lu = []
    for option, lus in inputs.agman2lu.items():
        option_idx = inputs.am_options.index(option)                     # the option's fixed am_idx
        for lu_idx, j in enumerate(lus):
            row = len(names)
            adoption_limit = float(inputs.ag_man_limits[option][j])
            am_cols = am[(am_idx == option_idx) & (j_idx == lu_idx)]              # the slot's am columns, both lm, dry first
            ag_cols = ag[ag_j == j]                                               # the ag columns of j: both lm, dry first, cells ascending
            row_idx += [np.full(am_cols.size, row), np.full(ag_cols.size, row)]
            col_idx += [am_cols, ag_cols]
            vals += [np.ones(am_cols.size), np.full(ag_cols.size, -adoption_limit)]
            names.append(f"const_ag_man_adoption_limit_{option}_{j}".replace(" ", "_"))
            key_am.append(option_idx)
            key_lu.append(j)
    n_rows = len(names)
    row_matrix, _, _ = contract(sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_rows, cols.sizes['col'])))
    return make_part('ag_mgt_adoption', row_matrix, np.zeros(n_rows), '<', names, am_idx=key_am, j=key_lu)


def get_regional_adoption_ag(inputs: RowInputs, cols: xr.Dataset):
    """One row per (region, ag land use) cap ('individual_cap' mode): Σ hectares · X_ag over the cap's cells ≤ cap (hectares, NOT rescaled)."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        print("│   ├── TURNING OFF constraints for regional adoption (ag land uses) ...")
        return None, None
    print("│   ├── Adding constraints for regional adoption (ag land uses)...")
    in_ag = cols['block'].values == 'ag'
    hectares = inputs.real_area[cols['cell'].values].astype(np.float32)   # the hectares a column's whole share stands for
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits["ag_regional_adoption"]:
        name = f"reg_adopt_limit_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(inputs.ncells, dtype=bool)                                    # the cap's cells ...
        in_region[reg_cells] = True
        on = in_ag & (cols['j'].values == lu_code) & in_region[cols['cell'].values]   # ... and the land use's ag columns in them
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(hectares[on])
        rhs.append(area_limit_ha)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not names:
        return None, None
    row_matrix = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), cols.sizes['col']))
    row_matrix, rhs, _ = contract(row_matrix, rhs)
    return make_part('regional_adoption_ag', row_matrix, rhs, '<', names, region=[reg_id for reg_id, _ in keys], j=[lu_code for _, lu_code in keys])


def get_regional_adoption_nonag(inputs: RowInputs, cols: xr.Dataset):
    """One row per (region, non-ag land use) cap ('individual_cap' mode): Σ hectares · X_nonag over the cap's cells ≤ cap · relax."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        print("│   ├── TURNING OFF constraints for regional adoption (non-ag land uses) ...")
        return None, None
    print("│   ├── Adding constraints for regional adoption (non-ag land uses)...")
    # the caps recede 1e-6/yr RELATIVE, so the RHS always stays ahead of the ratcheting lower bound non-reversible
    # plantings create: last year's solved areas become this year's exact lower bounds, and float32 noise then puts the
    # locked-in floor a hair over a saturated cap, which presolve rejects with NO tolerance (ag caps need none: ag is reversible)
    relax = 1 + (inputs.target_year - settings.SIM_YEARS[0]) * 1e-6
    in_nonag = cols['block'].values == 'nonag'
    hectares = inputs.real_area[cols['cell'].values].astype(np.float32)   # the hectares a column's whole share stands for
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    keys = []
    for reg_id, lu_code, lu_name, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption") or []:
        name = f"reg_adopt_limit_non_ag_{lu_name}_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(inputs.ncells, dtype=bool)                                    # the cap's cells ...
        in_region[reg_cells] = True
        on = in_nonag & (cols['k'].values == lu_code) & in_region[cols['cell'].values]   # ... and the land use's non-ag columns in them
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(hectares[on])
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append((reg_id, lu_code))
    if not names:
        return None, None
    row_matrix = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), cols.sizes['col']))
    row_matrix, rhs, _ = contract(row_matrix, rhs)
    return make_part('regional_adoption_nonag', row_matrix, rhs, '<', names, region=[reg_id for reg_id, _ in keys], k=[lu_code for _, lu_code in keys])


def get_regional_adoption_nonag_sum(inputs: RowInputs, cols: xr.Dataset):
    """One row per region cap ('non_ag_sum_cap' mode): Σ hectares · X_nonag over EVERY non-ag land use in the cap's cells ≤ cap · relax."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        print("│   ├── TURNING OFF constraints for regional adoption (non-ag total) ...")
        return None, None
    print("│   ├── Adding constraints for regional adoption (non-ag total)...")
    # the caps recede 1e-6/yr RELATIVE, so the RHS always stays ahead of the ratcheting lower bound non-reversible
    # plantings create: last year's solved areas become this year's exact lower bounds, and float32 noise then puts the
    # locked-in floor a hair over a saturated cap, which presolve rejects with NO tolerance (ag caps need none: ag is reversible)
    relax = 1 + (inputs.target_year - settings.SIM_YEARS[0]) * 1e-6
    in_nonag = cols['block'].values == 'nonag'
    hectares = inputs.real_area[cols['cell'].values].astype(np.float32)   # the hectares a column's whole share stands for
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    keys = []
    for reg_id, reg_cells, area_limit_ha in inputs.limits.get("non_ag_regional_adoption_sum") or []:
        name = f"reg_adopt_limit_non_ag_sum_{reg_id}".replace(" ", "_")
        if len(reg_cells) == 0:
            print(f"│   │   ├── SKIPPING {name} (no cells at this resolution)")
            continue
        print(f"│   │   ├── Adding constraint {name} <= {area_limit_ha:,.0f} HA...")
        in_region = np.zeros(inputs.ncells, dtype=bool)                                    # the cap's cells ...
        in_region[reg_cells] = True
        on = in_nonag & in_region[cols['cell'].values]   # ... and every non-ag column in them
        row_idx.append(np.full(int(on.sum()), len(names)))
        col_idx.append(np.flatnonzero(on))
        vals.append(hectares[on])
        rhs.append(area_limit_ha * relax)
        names.append(name)
        keys.append(reg_id)
    if not names:
        return None, None
    row_matrix = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), cols.sizes['col']))
    row_matrix, rhs, _ = contract(row_matrix, rhs)
    return make_part('regional_adoption_nonag_sum', row_matrix, rhs, '<', names, region=keys)


def get_water(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One row per water region: Σ net yield · X over the region's columns ≥ limit (a net yield can be negative)."""
    if settings.WATER_LIMITS != "on":
        print("│   ├── TURNING OFF water usage constraints ...")
        return None, None
    print("│   ├── Adding constraints for water usage limits...")
    coeff = gather(cols, inputs, inputs.ag_w_mrj, inputs.ag_man_w_mrj, inputs.non_ag_w_rk)

    # ── the rows: one per region with a limit, in the limits' order ──
    region_ids = list(inputs.limits["water"].keys())
    region_names = [inputs.water_region_names[region_id] for region_id in region_ids]
    rhs = list(inputs.limits["water"].values())
    names = [f"water_yield_limit_{region_name}".replace(" ", "_") for region_name in region_names]

    # ── the entries: every column on its region's row ──
    row_of_cell = pd.Index(region_names).get_indexer(support.region2cell['water_region'].values)   # the row of every cell's region, -1 where it has none or no limit
    row_of_col = row_of_cell[cols['cell'].values]
    on = (row_of_col >= 0) & (coeff != 0)                                # the columns in a region with a limit, with a net yield
    for line in pd.DataFrame({
        'water region': region_names, 
        'target inside LUTO study area (ML)': rhs}
        ).to_markdown(index=False, tablefmt='psql', floatfmt=',.0f').split('\n'):
        print(f"│   │   {line}")
    if not names:
        return None, None
    row_matrix = sparse.csr_matrix((coeff[on], (row_of_col[on], np.flatnonzero(on))), shape=(len(names), cols.sizes['col']))
    row_matrix, rhs, scale = contract(row_matrix, rhs, rescale=True)
    return make_part('water', row_matrix, rhs, '>', names, scale, region=region_ids)


def get_renewable(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One row per (state, type): Σ yield · X_am over the type's columns in the state, outside the excluded cells ≥ target − existing."""
    if not any(settings.RENEWABLES_OPTIONS.values()):
        print("│   ├── TURNING OFF renewable energy constraints ...")
        return None, None
    print("│   ├── Adding constraints for renewable energy production targets ...")
    options = list(inputs.agman2lu)                                      # the enabled options
    re_types = {                                                         # the enabled types only: the options are the enabled ag managements
        option: re_data for option, re_data in {
            'Utility Solar PV': dict(energy_r=inputs.renewable_solar_r, gbf2_mask_idx=inputs.mask_gbf2_solar, mnes_mask_idx=inputs.mask_mnes_solar),
            'Onshore Wind':     dict(energy_r=inputs.renewable_wind_r,  gbf2_mask_idx=inputs.mask_gbf2_wind,  mnes_mask_idx=inputs.mask_mnes_wind),
        }.items() if option in options
    }
    states       = sorted({s for s in set(support.region2cell['state'].values) if isinstance(s, str)} - {'Australian Capital Territory'})   # the states the rows are written for, by name; ACT folded into NSW below
                                                                         # (a cell without a state is None in cell_regions, but xarray 2026.7 stores None as NaN: keep names only)
    in_ag        = cols['block'].values == 'ag'

    # ── per type: its columns' yield, and the columns the exclusion masks keep out ──
    energy_of_type = {}
    excluded_of_type = {}
    for option, re_data in re_types.items():
        on_type = cols['am_idx'].values == inputs.am_options.index(option)
        energy_of_type[option] = np.where(on_type, re_data['energy_r'][cols['cell'].values], np.float32(0.0)).astype(np.float32)   # float32 yield per cell
        excluded_r = np.zeros(inputs.ncells, dtype=bool)
        if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
            excluded_r[re_data['gbf2_mask_idx']] = True
        if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
            excluded_r[re_data['mnes_mask_idx']] = True
        excluded_of_type[option] = excluded_r[cols['cell'].values]

    # ── one row per (state, type) with an allowed column of the type's host land uses ──
    row_idx = []
    col_idx = []
    vals = []
    rhs = []
    names = []
    key_am = []
    key_state = []
    for state_name in states:
        in_state = (support.region2cell['state'].values == state_name)[cols['cell'].values]   # the state's columns: those in its cells
        if state_name == 'New South Wales':                              # ACT counts toward the NSW+ACT target
            in_state |= (support.region2cell['state'].values == 'Australian Capital Territory')[cols['cell'].values]
        print(f"│   │   ├── Adding renewable energy constraints for {state_name} ...")
        for am in re_types:
            print(f"│   │   │   ├── target for {am} is {inputs.limits[f'renewable_{am}'][state_name]:5,.0f} MWh  (existing: {inputs.limits[f'renewable_{am}_exist'][state_name]:5,.0f} MWh)")
            allowed = in_state & ~excluded_of_type[am]                   # the state's columns outside the excluded cells
            # row-inclusion rule (NOT a coefficient test): the row exists iff an allowed cell holds an ag column of a
            # compatible land use — even if every coefficient there turns out to be sub-floor
            if not (allowed & in_ag & np.isin(cols['j'].values, inputs.agman2lu[am])).any():
                continue
            on = allowed & (energy_of_type[am] != 0)                     # the type's columns in the state, with a yield
            row_idx.append(np.full(int(on.sum()), len(names)))
            col_idx.append(np.flatnonzero(on))
            vals.append(energy_of_type[am][on])
            rhs.append(inputs.limits[f"renewable_{am}"][state_name] - inputs.limits[f"renewable_{am}_exist"][state_name])   # raw MWh; row-rescaled below
            names.append(f"renewable_{am}_target_{state_name}".replace(" ", "_"))
            key_am.append(inputs.am_options.index(am))
            key_state.append(state_name)
    if not names:
        return None, None
    row_matrix = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(len(names), cols.sizes['col']))
    row_matrix, rhs, scale = contract(row_matrix, rhs, rescale=True)
    return make_part('renewable', row_matrix, rhs, '>', names, scale, am_idx=key_am, region=key_state)


# ═══════════════════════════ the structural rows ═══════════════════════════


def get_ag_mgt_link(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One row per am column: X_am − X_ag ≤ 0 (an ag-mgt column cannot exceed the ag column it sits on)."""
    print("│   ├── Adding ag-management link (X_am ≤ X_ag) constraints...")
    options  = inputs.am_options                                                        # the names of the am_idx codes
    am_cols  = np.flatnonzero(cols['block'].values == 'am')                            # one row per am column, in table order ...
    am_idx   = cols['am_idx'].values[am_cols]
    m        = cols['m'].values[am_cols]
    j        = cols['j'].values[am_cols]
    cells    = cols['cell'].values[am_cols]
    ag_cols  = support.ag_mrj2col[m, cells, j]                                          # ... and the ag column each sits on
    names    = [f"const_ag_man_{options[a]}_usage_{inputs.landmans[lm]}_{lu}_{r}".replace(" ", "_") for a, lm, lu, r in zip(am_idx, m, j, cells)]
    n_rows = am_cols.size
    rows = np.arange(n_rows)
    row_matrix = sparse.csr_matrix(
        (np.concatenate([np.ones(n_rows), -np.ones(n_rows)]), (np.concatenate([rows, rows]), np.concatenate([am_cols, ag_cols]))),
        shape=(n_rows, cols.sizes['col'])
    )                                         # +1 on X_am, −1 on X_ag
    row_matrix, _, _ = contract(row_matrix)
    return make_part(
        'ag_mgt_link',
        row_matrix,
        np.zeros(n_rows),
        '<',
        names,
        am_idx=am_idx, j=j, m=m, cell=cells,
    )


def get_renewable_ceiling(inputs: RowInputs, cols: xr.Dataset, support: ColSupport):
    """One row per (renewable option, cell with existing capacity): Σ X_am ≤ max(ag_mask − existing, 0)."""
    print("│   ├── Adding renewable ceiling (Σ X_am ≤ ag mask − existing) constraints...")
    blocks = []
    rhs = []
    names = []
    key_am = []
    key_cell = []
    for option in inputs.agman2lu:
        if option not in settings.RENEWABLES_OPTIONS:
            continue
        option_idx = inputs.am_options.index(option)                     # the option's fixed am_idx
        am_name = tools.am_name_snake_case(option)
        exist_r = inputs.exist_renewable_solar_r if option == "Utility Solar PV" else inputs.exist_renewable_wind_r   # the total across ALL data years: the ceiling never decreases between periods, so lb(t) <= ceiling always holds
        on_option = (cols['am_idx'].values == option_idx).astype(np.float32)   # 1 on the option's columns
        has_option = support.cell2col @ on_option != 0                                            # the cells holding a column of the option ...
        row_cells = np.flatnonzero(has_option & (exist_r != 0))                                  # ... and existing capacity (none -> no ceiling row): one row each, ascending
        if not row_cells.size:
            continue
        blocks.append(support.cell2col[row_cells] @ sparse.diags(on_option))                     # the variables in those cells, each column times its on_option: the option's columns
        rhs.append(np.maximum(inputs.ag_mask_proportion_r[row_cells] - exist_r[row_cells], 0.0))   # cell space left for simulated capacity
        names += [f"const_{am_name}_solvable_ub_{r}".replace(" ", "_") for r in row_cells]
        key_am += [option_idx] * row_cells.size
        key_cell.append(row_cells)
    if not blocks:
        return None, None
    row_matrix, rhs, _ = contract(sparse.vstack(blocks, format='csr'), np.concatenate(rhs))
    return make_part('renewable_ceiling', row_matrix, rhs, '<', names, am_idx=key_am, cell=np.concatenate(key_cell))


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
    print("│   ├── Adding source-cap (Σ out ≤ base) constraints at the ag sources...")
    arcs        = np.flatnonzero(np.isin(cols['block'].values, ('ag2ag', 'ag2nonag')))   # the arcs leaving an ag node
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
    row_matrix = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(first_arc.size, cols.sizes['col']))
    # the cap is the source's base on the table: a source is a holding above the noise floor, and the ag ub is raised
    # to that same floor, so every source has a column
    source_col = support.ag_mrj2col[from_m[first_arc], cell[first_arc], from_j[first_arc]]
    assert (source_col >= 0).all(), 'an ag source without an ag column: the source map and the ub floor disagree'
    rhs = cols['base'].values[source_col]
    row_matrix, rhs, _ = contract(row_matrix, rhs)
    names = [f"srccap_a_{m}_{j}_{r}" for m, j, r in zip(from_m[first_arc], from_j[first_arc], local_r[first_arc])]
    return make_part('source_cap_ag', row_matrix, rhs, '<', names, from_m=from_m[first_arc], from_j=from_j[first_arc], local_r=local_r[first_arc])


def get_source_cap_nonag(cols: xr.Dataset, support: ColSupport):
    """One row per non-ag source (nonag_lu, cell): Σ of the arcs leaving it (nonag2ag) ≤ its base share."""
    print("│   ├── Adding source-cap (Σ out ≤ base) constraints at the non-ag sources...")
    arcs    = np.flatnonzero(cols['block'].values == 'nonag2ag')                               # the arcs leaving a non-ag node
    if not arcs.size:
        return None, None
    from_k  = cols['from_k'].values[arcs]
    local_r = cols['local_r'].values[arcs]
    cell    = cols['cell'].values[arcs]
    ncells, n_k = support.nonag_rk2col.shape
    source = np.ravel_multi_index((from_k, cell), (n_k, ncells))                                # the flat position of each arc's source in (nonag_lu, cell): one row per source, ascending
    _, first_arc, row_of_arc = np.unique(source, return_index=True, return_inverse=True)
    row_matrix = sparse.csr_matrix((np.ones(arcs.size), (row_of_arc, arcs)), shape=(first_arc.size, cols.sizes['col']))
    # the cap is the source's base on the table: a source is a holding above the noise floor, and the non-ag ub is
    # raised to that same floor, so every source has a column
    source_col = support.nonag_rk2col[cell[first_arc], from_k[first_arc]]
    assert (source_col >= 0).all(), 'a non-ag source without a non-ag column: the source map and the ub floor disagree'
    rhs = cols['base'].values[source_col]
    row_matrix, rhs, _ = contract(row_matrix, rhs)
    names = [f"srccap_n_{k}_{r}" for k, r in zip(from_k[first_arc], local_r[first_arc])]
    return make_part('source_cap_nonag', row_matrix, rhs, '<', names, from_k=from_k[first_arc], local_r=local_r[first_arc])


def get_node_balance_ag(cols: xr.Dataset, support: ColSupport):
    """One row per ag column: X_ag = base + Σ in (ag2ag ∪ nonag2ag) − Σ out (ag2ag ∪ ag2nonag)."""
    print("│   ├── Adding node-balance (X = base + Σin − Σout) constraints at the ag nodes...")
    ag          = np.flatnonzero(cols['block'].values == 'ag')

    # ── the rows: one per ag column ──
    n_ag = ag.size
    ag_m, ag_j, ag_r = cols['m'].values[ag], cols['j'].values[ag], cols['cell'].values[ag]
    row_of_col = np.full(cols.sizes['col'], -1, dtype=np.int64)   # the row of every ag column, -1 off the ag block
    row_of_col[ag] = np.arange(n_ag)

    def ag_row(m_, j_, r_):
        """The row of the ag node (m, j, r): its column's row, -1 where it has no column."""
        col = support.ag_mrj2col[m_, r_, j_]
        return np.where(col >= 0, row_of_col[col], -1)

    # ── the entries: X on its own row, inflows −1 on the target's row, outflows +1 on the source's row ──
    row_idx = []
    col_idx = []
    vals = []

    def add(row, col, value):
        in_model = row >= 0                                              # no row (banned source / no X var): entry dropped
        row_idx.append(row[in_model].astype(np.int64))
        col_idx.append(col[in_model].astype(np.int64))
        vals.append(np.full(int(in_model.sum()), value))

    add(np.arange(n_ag), ag, 1.0)                                        # X_ag on its own row
    arcs = np.flatnonzero(cols['block'].values == 'ag2ag')   # ag → ag: in on the target's row, out of the source's
    add(ag_row(cols['m'].values[arcs], cols['j'].values[arcs], cols['cell'].values[arcs]), arcs, -1.0)
    add(ag_row(cols['from_m'].values[arcs], cols['from_j'].values[arcs], cols['cell'].values[arcs]), arcs, 1.0)
    arcs = np.flatnonzero(cols['block'].values == 'ag2nonag')   # ag → non-ag: out of the source's row
    add(ag_row(cols['from_m'].values[arcs], cols['from_j'].values[arcs], cols['cell'].values[arcs]), arcs, 1.0)
    arcs = np.flatnonzero(cols['block'].values == 'nonag2ag')   # non-ag → ag: in on the target's row
    add(ag_row(cols['m'].values[arcs], cols['j'].values[arcs], cols['cell'].values[arcs]), arcs, -1.0)
    row_matrix = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_ag, cols.sizes['col']))

    # ── rhs, names, keys ──
    row_matrix, rhs, _ = contract(row_matrix, cols['base'].values[ag])
    names = [f"bal_a_{m}_{j}_{r}" for m, j, r in zip(ag_m, ag_j, ag_r)]
    return make_part('node_balance_ag', row_matrix, rhs, '=', names, m=ag_m, j=ag_j, cell=ag_r)


def get_node_balance_nonag(cols: xr.Dataset, support: ColSupport):
    """One row per non-ag column: X_nonag = base + Σ in (ag2nonag) − Σ out (nonag2ag)."""
    print("│   ├── Adding node-balance (X = base + Σin − Σout) constraints at the non-ag nodes...")
    nonag       = np.flatnonzero(cols['block'].values == 'nonag')

    # ── the rows: one per non-ag column ──
    n_nonag = nonag.size
    if not n_nonag:
        return None, None
    nonag_k, nonag_r = cols['k'].values[nonag], cols['cell'].values[nonag]
    row_of_col = np.full(cols.sizes['col'], -1, dtype=np.int64)   # the row of every non-ag column, -1 off the non-ag block
    row_of_col[nonag] = np.arange(n_nonag)

    def nonag_row(k_, r_):
        """The row of the non-ag node (k, r): its column's row, -1 where it has no column."""
        col = support.nonag_rk2col[r_, k_]
        return np.where(col >= 0, row_of_col[col], -1)

    # ── the entries: X on its own row, inflows −1 on the target's row, outflows +1 on the source's row ──
    row_idx = []
    col_idx = []
    vals = []

    def add(row, col, value):
        in_model = row >= 0                                              # no row (the node has no column): entry dropped
        row_idx.append(row[in_model].astype(np.int64))
        col_idx.append(col[in_model].astype(np.int64))
        vals.append(np.full(int(in_model.sum()), value))

    add(np.arange(n_nonag), nonag, 1.0)                                  # X_nonag on its own row
    arcs = np.flatnonzero(cols['block'].values == 'ag2nonag')   # ag → non-ag: in on the target's row
    add(nonag_row(cols['k'].values[arcs], cols['cell'].values[arcs]), arcs, -1.0)
    arcs = np.flatnonzero(cols['block'].values == 'nonag2ag')   # non-ag → ag: out of the source's row
    add(nonag_row(cols['from_k'].values[arcs], cols['cell'].values[arcs]), arcs, 1.0)
    row_matrix = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_nonag, cols.sizes['col']))

    # ── rhs, names, keys ──
    row_matrix, rhs, _ = contract(row_matrix, cols['base'].values[nonag])
    names = [f"bal_n_{k}_{r}" for k, r in zip(nonag_k, nonag_r)]
    return make_part('node_balance_nonag', row_matrix, rhs, '=', names, k=nonag_k, cell=nonag_r)
