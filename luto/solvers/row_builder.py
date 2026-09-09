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
from dataclasses import dataclass
from typing import Any, Optional

from luto.data import Data
from luto import settings
import luto.tools as tools

import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.transitions as ag_transition
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.biodiversity as ag_biodiversity

import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.non_agricultural.biodiversity as non_ag_biodiversity
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.quantity as non_ag_quantity
import luto.economics.non_agricultural.transitions as non_ag_transition
import luto.economics.non_agricultural.revenue as non_ag_revenue


def drop(col: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step 1 of the coefficient contract: keep the terms with |q| >= SOLVER_COEFF_MIN (NaN is
    dropped). Returns (column, coefficient) of the kept terms."""
    keep = np.abs(q) >= settings.SOLVER_COEFF_MIN
    return col[keep], q[keep]


def block_slice(table: xr.Dataset, block: str) -> slice:
    """The rows of one block of the column table — its Var.index range (``attrs['block_range']`` holds the bounds)."""
    return slice(*table.attrs['block_range'][block])


def am_runs(table: xr.Dataset, by: str) -> list:
    """The am block grouped by ``by`` — ``'am_idx'`` (the option) or ``'slot'`` (the (am, lu) slot) — as runs
    of table rows: the block is sorted by slot and the slots by option, so every group present is one run.
    Returns the (start, stop) of each run; a group's fields are those of its first row."""
    am = block_slice(table, 'am')
    _, first = np.unique(table[by].values[am], return_index=True)
    bounds = am.start + np.append(first, am.stop - am.start)
    return list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))


def gather_coeffs(cols: dict, ag_c_mrj, am_c_mrj: dict, nonag_c_rk) -> np.ndarray:
    """One family's per-cell coefficient at every scored column — the table's ag | nonag | am rows, so
    the array is indexed by column — float32: ``ag_c_mrj[m, r, j]`` on the ag columns,
    ``nonag_c_rk[r, k]`` on the non-ag columns, ``am_c_mrj[am][m, r, j_idx]`` on the ag-mgt columns."""
    table = cols['table']
    m, j, k, cell, am_idx, j_idx = (table[field].values for field in ('m', 'j', 'k', 'cell', 'am_idx', 'j_idx'))
    ag, nonag = block_slice(table, 'ag'), block_slice(table, 'nonag')
    c = np.empty(table.attrs['n_terms'], dtype=np.float32)
    c[ag] = ag_c_mrj[m[ag], cell[ag], j[ag]]
    c[nonag] = nonag_c_rk[cell[nonag], k[nonag]]
    for start, stop in am_runs(table, 'am_idx'):
        run = slice(start, stop)
        c[run] = am_c_mrj[table.attrs['options'][am_idx[start]]][m[run], cell[run], j_idx[run]]
    return c


def gather_bio_coeffs(cols: dict, contr_ag_j, contr_am: dict, contr_nonag_k: dict) -> np.ndarray:
    """The biodiversity contribution at every scored column: a scalar per ag land use, a per-cell array
    per (am, land use), a scalar per non-ag land use — the shared C of the GBF families."""
    table = cols['table']
    j, k, cell, am_idx, j_idx = (table[field].values for field in ('j', 'k', 'cell', 'am_idx', 'j_idx'))
    ag, nonag = block_slice(table, 'ag'), block_slice(table, 'nonag')
    c = np.zeros(table.attrs['n_terms'], dtype=np.float32)
    c[ag] = contr_ag_j[j[ag]]                                                    # float32 per land use
    n_k = max(contr_nonag_k) + 1 if len(contr_nonag_k) else 0
    contr_by_k = np.array([contr_nonag_k.get(lu, 0.0) for lu in range(n_k)], dtype=np.float32)
    if nonag.stop > nonag.start:
        c[nonag] = contr_by_k[k[nonag]]
    for start, stop in am_runs(table, 'slot'):
        run = slice(start, stop)
        c[run] = np.asarray(contr_am[table.attrs['options'][am_idx[start]]][int(j_idx[start])], dtype=np.float32)[cell[run]]
    return c


def compose_rows(cols: dict, c: np.ndarray, val_rows) -> sparse.csr_matrix:
    """The family block: one CSR row per weighting row ``V`` in ``val_rows`` (each a float32 vector
    over cells), over the scored columns (the table's ag | nonag | am rows; ``c`` is indexed by column).
    Row i, column t: ``q = V_i[cell_t] · c_t`` (float32), dropped when |q| < SOLVER_COEFF_MIN. A row
    with a small support gathers only its columns, through the table's scored rows sorted by cell
    (cost ∝ the row's nonzero cells, not the model)."""
    table = cols['table']
    n_terms = table.attrs['n_terms']
    term_cell = table['cell'].values[:n_terms]
    term_col = np.arange(n_terms, dtype=np.int32)                         # the scored columns are the first rows of the table: column = row
    by_cell_order, by_cell_ptr = table.attrs['by_cell_order'], table.attrs['by_cell_ptr']
    ncells = by_cell_ptr.size - 1
    nvars = table.attrs['n_all']
    indptr = [0]
    indices = []
    data = []
    for val_row in val_rows:
        val_row = np.asarray(val_row, dtype=np.float32)
        cells = np.flatnonzero(val_row)
        if cells.size * 2 < ncells:                                   # sparse support: gather only its columns
            starts = by_cell_ptr[cells]
            counts = by_cell_ptr[cells + 1] - by_cell_ptr[cells]
            positions = np.repeat(starts - (np.cumsum(counts) - counts), counts) + np.arange(int(counts.sum()))
            term_idx = by_cell_order[positions]
            kept_cols, kept_vals = drop(term_col[term_idx], val_row[term_cell[term_idx]] * c[term_idx])
        else:
            kept_cols, kept_vals = drop(term_col, val_row[term_cell] * c)
        order = np.argsort(kept_cols, kind='stable')
        indices.append(kept_cols[order])
        data.append(kept_vals[order])
        indptr.append(indptr[-1] + kept_cols.size)
    if not indices:
        return sparse.csr_matrix((0, nvars), dtype=np.float32)
    block = sparse.csr_matrix((np.concatenate(data).astype(np.float32), np.concatenate(indices).astype(np.int32),
                               np.asarray(indptr, dtype=np.int64)), shape=(len(val_rows), nvars))
    block.has_sorted_indices = True
    return block


def calc_geomean_scale(lhs_max: float, rhs_max: float) -> float:
    """One scale factor as the geometric mean of LHS and RHS magnitudes, normalised by
    ``settings.RESCALE_FACTOR`` (RF): ``sqrt(lhs_max * rhs_max) / RF``. After dividing both sides
    by it, LHS_max and RHS land symmetrically around RF in log space. Falls back to LHS-only
    (``lhs_max / RF``) when ``rhs_max`` is zero."""
    if lhs_max > 0.0 and rhs_max > 0.0:
        return float(np.sqrt(lhs_max * rhs_max) / settings.RESCALE_FACTOR)
    ref = lhs_max if lhs_max > 0.0 else settings.RESCALE_FACTOR
    return float(ref / settings.RESCALE_FACTOR)


def scale_rows(block: sparse.csr_matrix, rhs: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Row rescaling on the COMPOSED block: row i and rhs i divided by ``calc_geomean_scale(max|row_i|,
    |rhs_i|)``, then the scaled row floored again (step 2 of the coefficient contract). An exact LP
    transformation. Returns ``(block, rhs, scale)``: row × scale restores the raw row; the block
    stays float32, ``rhs`` and ``scale`` are float64."""
    block = block.tocsr(copy=True)
    rhs = np.asarray(rhs, dtype=np.float64)
    nnz_per_row = np.diff(block.indptr)
    row_max = np.zeros(block.shape[0], dtype=np.float64)
    has_entries = nnz_per_row > 0
    if has_entries.any():
        row_max[has_entries] = np.maximum.reduceat(np.abs(block.data).astype(np.float64), block.indptr[:-1][has_entries])
    scale = np.fromiter((calc_geomean_scale(lhs_max, abs(rhs_i)) for lhs_max, rhs_i in zip(row_max, rhs)),
                        dtype=np.float64, count=block.shape[0])
    block.data = (block.data / np.repeat(scale, nnz_per_row)).astype(np.float32)
    block.data[np.abs(block.data) < settings.SOLVER_COEFF_MIN] = 0.0   # floor the scaled row
    block.eliminate_zeros()
    block.sort_indices()
    return block, rhs / scale, scale


# ═══════════════════════════ the row side: coefficient streams, targets, the objective ═══════════════════════════

OBJ_BLOCKS = ('ag', 'am', 'nonag', 'trans_ag', 'trans_nonag')   # rows of RowInputs.obj_block


@dataclass
class RowInputs:
    """The row side of one (base_year -> target_year) step: coefficient streams (raw units,
    float32), targets, regions, biodiversity layers, flow emissions and the objective block.
    What decides the COLUMNS (feasibility, bounds, base, sources, masks) lives on the space."""
    base_year: int                                                      # Base year of this solve step.
    target_year: int                                                    # Target year of this solve step.

    ag_g_mrj: np.ndarray                                                # Agricultural GHG emissions [m, r, j].
    ag_w_mrj: np.ndarray                                                # Agricultural water net yield [m, r, j].
    ag_q_mrp: np.ndarray                                                # Agricultural production quantity [m, r, p] (p = product, not land use).

    non_ag_g_rk: np.ndarray                                             # Non-agricultural GHG emissions [r, k].
    non_ag_w_rk: np.ndarray                                             # Non-agricultural water net yield [r, k].
    non_ag_q_crk: np.ndarray                                            # Non-agricultural production quantity [c, r, k].

    ag_man_g_mrj: dict                                                  # {am: GHG emission effect [m, r, j_idx]}.
    ag_man_w_mrj: dict                                                  # {am: water net-yield effect [m, r, j_idx]}.
    ag_man_q_mrp: dict                                                  # {am: production quantity effect [m, r, p]}.
    ag_man_limits: dict                                                 # {am: {j: adoption limit}}.

    renewable_solar_r: np.ndarray                                       # Renewable energy - solar yield matrix.
    renewable_wind_r: np.ndarray                                        # Renewable energy - wind yield matrix.
    exist_renewable_solar_r: np.ndarray                                 # Existing solar capacity converted to annual MWh per cell.
    exist_renewable_wind_r: np.ndarray                                  # Existing wind capacity converted to annual MWh per cell.
    
    region_state_r: np.ndarray                                          # Region state index for each cell.
    region_state_name2idx: dict[str, int]                               # Map of region state names to indices.
    region_NRM_names_r: np.ndarray                                      # Region NRM names for each cell.
    
    water_region_indices: dict[int, np.ndarray]                         # {region id: cell indices} of the water regions.
    water_region_names: dict[int, str]                                  # {region id: region name}.

    biodiv_contr_ag_j: np.ndarray                                       # Biodiversity contribution scale per agricultural land use (j).
    biodiv_contr_non_ag_k: dict[int, float]                             # Biodiversity contribution scale per non-agricultural land use (k).
    biodiv_contr_ag_man: dict[str, dict[int, np.ndarray]]               # Biodiversity contribution scale per ag-management option and land use, per cell.
    
    GBF2_mask_area_r: np.ndarray                                        # GBF2 priority-degraded-area mask × real area, per cell (r).
    GBF3_NVIS_pre_1750_area_vr: np.ndarray                              # GBF3 pre-1750 NVIS vegetation area, per group (v) and cell (r).
    GBF3_NVIS_region_group: list                                        # GBF3 constraint pairs - list[(region, group)].
    GBF4_SNES_pre_1750_area_sr: xr.DataArray                            # GBF4 SNES pre-1750 area [layer=(species, presence), cell]; region masking happens in the solver.
    GBF4_SNES_region_species: list                                      # GBF4 SNES constraint triplets - list[(region, species, presence)].
    GBF4_ECNES_pre_1750_area_sr: xr.DataArray                           # GBF4 ECNES pre-1750 area [layer=(community, presence), cell]; region masking happens in the solver.
    GBF4_ECNES_region_species: list                                     # GBF4 ECNES constraint triplets - list[(region, community, presence)].
    GBF8_pre_1750_area_sr: xr.DataArray                                 # GBF8 pre-1750 species area [species, cell].
    GBF8_region_species: list                                           # GBF8 constraint pairs - list[(region, species)].

    commodity_names: list[str]                                          # Commodity names (data.COMMODITIES order).
    offland_ghg: np.ndarray                                             # Target-year GHG emissions from off-land commodities (tCO2e); 0.0 when GHG limits are off.
    lu2pr_pj: np.ndarray                                                # Conversion matrix: product (p) × land use (j).
    pr2cm_cp: np.ndarray                                                # Conversion matrix: commodity (c) × product (p).
    limits: dict                                                        # Raw constraint targets for the target year (see get_limits).
    real_area: np.ndarray                                               # Area of each cell (ha), per cell (r).
    ag_mask_proportion_r: np.ndarray                                    # Base-year (2010) agricultural proportion of each cell (r).

    trans_ghg_ag2ag: dict                                               # {(from_m, from_j): ndarray[to_m, local_r, to_j]} transition emissions, raw tCO2e, float32.

    # ── Objective ──
    obj_block: sparse.csr_matrix                                        # Economy coefficients, raw AUD, (5 × n_dec) float32, one row per OBJ_BLOCKS component; see get_obj_block.

    @property
    def ncms(self):
        return len(self.commodity_names)

    def bio_coeffs(self, cols: dict) -> np.ndarray:
        """The biodiversity contribution at every scored column of the table (the shared C of the
        GBF families), gathered once per step."""
        if getattr(self, '_bio_c', None) is None:
            self._bio_c = gather_bio_coeffs(cols, self.biodiv_contr_ag_j, self.biodiv_contr_ag_man, self.biodiv_contr_non_ag_k)
        return self._bio_c
    


def get_ag_c_mrj(data: Data, target_index):
    print('Getting agricultural cost matrices...', flush = True)
    output = ag_cost.get_cost_matrices(data, target_index)
    return output.astype(np.float32)

def get_non_ag_c_rk(data: Data, ag_c_mrj: np.ndarray, lumap: np.ndarray, target_year):
    print('Getting non-agricultural cost matrices...', flush = True)
    output = non_ag_cost.get_cost_matrix(data, ag_c_mrj, lumap, target_year)
    return output.astype(np.float32)

def get_ag_r_mrj(data: Data, target_index):
    print('Getting agricultural revenue matrices...', flush = True)
    output = ag_revenue.get_rev_matrices(data, target_index)
    return output.astype(np.float32)

def get_non_ag_r_rk(data: Data, ag_r_mrj: np.ndarray, base_year: int, target_year: int):
    print('Getting non-agricultural revenue matrices...', flush = True)
    output = non_ag_revenue.get_rev_matrix(data, target_year, ag_r_mrj, data.lumaps[base_year])
    return output.astype(np.float32)

def get_ag_g_mrj(data: Data, target_index):
    print('Getting agricultural GHG emissions matrices...', flush = True)
    output = ag_ghg.get_ghg_matrices(data, target_index)
    return output.astype(np.float32)

def get_non_ag_g_rk(data: Data, ag_g_mrj, base_year):
    print('Getting non-agricultural GHG emissions matrices...', flush = True)
    output = non_ag_ghg.get_ghg_matrix(data, ag_g_mrj, data.lumaps[base_year])
    return output.astype(np.float32)

def get_ag_w_mrj(data: Data, target_index, water_dr_yield: Optional[np.ndarray] = None, water_sr_yield: Optional[np.ndarray] = None):
    print('Getting agricultural water net yield matrices based on historical water yield layers ...', flush = True)
    output = ag_water.get_water_net_yield_matrices(data, target_index, water_dr_yield, water_sr_yield)
    return output.astype(np.float32)

def get_w_region_indices(data: Data):
    if settings.WATER_LIMITS == 'off':
        return {}
    print('Getting water region indices...', flush = True)
    return data.WATER_REGION_INDEX_R

def get_w_region_names(data: Data):
    if settings.WATER_LIMITS == 'off':
        return {}
    print('Getting water region names...', flush = True)
    return data.WATER_REGION_NAMES

def get_ag_biodiv_contr_j(data: Data) -> dict[int, float]:
    print('Getting biodiversity degredation data for agricultural land uses...', flush = True)
    return ag_biodiversity.get_ag_biodiversity_contribution(data)

def get_non_ag_biodiv_impact_k(data: Data) -> dict[int, float]:
    print('Getting biodiversity benefits data for non-agricultural land uses...', flush = True)
    return non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data)

def get_ag_man_biodiv_impacts(data: Data, target_year: int) -> dict[str, dict[str, float]]:
    print('Getting biodiversity benefits data for agricultural management options...', flush = True)
    return ag_biodiversity.get_ag_management_biodiversity_contribution(data, target_year)

def get_GBF2_mask_area_r(data: Data) -> np.ndarray:
    if settings.GBF2_TARGET == "off":
        return np.empty(0)
    print('Getting GBF2 mask area layer...', flush = True)
    output = ag_biodiversity.get_GBF2_MASK_area(data)
    return output

def get_GBF3_NVIS_pre_1750_area_vr(data: Data):
    if settings.GBF3_NVIS_TARGET == "off":
        return np.empty(0)
    print('Getting GBF3 NVIS vegetation matrices...', flush = True)
    output = ag_biodiversity.get_GBF3_NVIS_matrices_vr(data)
    return output

def get_GBF3_NVIS_region_group(data: Data) -> dict[int,str]:
    if settings.GBF3_NVIS_TARGET == "off":
        return {}
    print('Getting GBF3 NVIS vegetation group names...', flush = True)
    return data.BIO_GBF3_NVIS_SEL

def get_GBF4_SNES_pre_1750_area_sr(data: Data) -> xr.DataArray:
    if settings.GBF4_TARGET_SNES == 'off':
        return np.empty(0)
    print('Getting GBF4 SNES species area matrices...', flush=True)
    return ag_biodiversity.get_GBF4_SNES_matrix_sr(data)

def get_GBF4_SNES_region_species(data: Data) -> list:
    if settings.GBF4_TARGET_SNES == 'off':
        return []
    print('Getting GBF4 SNES (region, species, presence) constraint triplets...', flush=True)
    return data.BIO_GBF4_SNES_SEL

def get_GBF4_ECNES_pre_1750_area_sr(data: Data) -> xr.DataArray:
    if settings.GBF4_TARGET_ECNES == 'off':
        return np.empty(0)
    print('Getting GBF4 ECNES community area matrices...', flush=True)
    return ag_biodiversity.get_GBF4_ECNES_matrix_sr(data)

def get_GBF4_ECNES_region_species(data: Data) -> list:
    if settings.GBF4_TARGET_ECNES == 'off':
        return []
    print('Getting GBF4 ECNES (region, community, presence) constraint triplets...', flush=True)
    return data.BIO_GBF4_ECNES_SEL

def get_GBF8_pre_1750_area_sr(data: Data, target_year: int) -> xr.DataArray:
    if settings.GBF8_TARGET == "off":
        return np.empty(0)
    print('Getting GBF8 species conservation area matrices...', flush=True)
    return ag_biodiversity.get_GBF8_matrix_sr(data, target_year)

def get_GBF8_region_species(data: Data) -> list:
    if settings.GBF8_TARGET == "off":
        return []
    print('Getting GBF8 (region, species) constraint pairs...', flush=True)
    return data.BIO_GBF8_SEL

def get_non_ag_w_rk(
    data: Data, 
    ag_w_mrj: np.ndarray, 
    base_year, 
    target_year, 
    water_dr_yield: Optional[np.ndarray] = None, 
    water_sr_yield: Optional[np.ndarray] = None
    ):
    print('Getting non-agricultural water yield matrices...', flush = True)
    yr_idx = target_year - data.YR_CAL_BASE
    output = non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj, data.lumaps[base_year], yr_idx, water_dr_yield, water_sr_yield)
    return output.astype(np.float32)

def get_ag_q_mrp(data: Data, target_index):
    print('Getting agricultural production quantity matrices...', flush = True)
    output = ag_quantity.get_quantity_matrices(data, target_index)
    return output.astype(np.float32)

def get_non_ag_q_crk(data: Data, ag_q_mrp: np.ndarray, base_year: int):
    print('Getting non-agricultural production quantity matrices...', flush = True)
    output = non_ag_quantity.get_quantity_matrix(data, ag_q_mrp, data.lumaps[base_year])
    return output.astype(np.float32)

def get_ag_t_mrj(data: Data, target_index, base_year):
    print('Getting agricultural transition cost matrices...', flush = True)
    # From-based flow-cost dict[(from_m, from_j)] -> ndarray(NLMS, ncells_src, N_AG_LUS), sliced per
    # source over each source's cells (the same cells `trans_source_ag` uses, so the solver
    # delta's local_r aligns with this dict's cell axis). Leaves are cast to float32 in get_input_data
    # with the other coefficient streams.
    mj_cell_map = ag_transition.get_base_dvar_mj_cell_map(data, base_year)
    return {
        (from_m, from_j): ag_transition.get_transition_matrices_ag2ag(data, target_index, from_m, from_j, cell_idx)
        for (from_m, from_j), cell_idx in mj_cell_map.items()
    }

def get_non_ag_t_rk(data: Data, base_year):
    # nonag→nonag transition cost. Currently a ZERO matrix — non-ag LUs are not allowed to transition
    # to other non-ag LUs (get_nonag2nonag_transition_matrix returns zeros). Kept as an explicit hook
    # so the objective wiring is ready if non-ag↔non-ag transitions are ever priced.
    print('Getting non-agricultural transition cost matrices...', flush = True)
    output = non_ag_transition.get_nonag2nonag_transition_matrix(data)
    return output

def get_potential_renewable_solar_r(data: Data, target_idx):
    print('Getting renewable energy - solar yield matrix...', flush = True)
    output = ag_quantity.get_quantity_renewable(data, 'Utility Solar PV', target_idx)
    return output

def get_potential_renewable_wind_r(data: Data, target_idx):
    print('Getting renewable energy - wind yield matrix...', flush = True)
    output = ag_quantity.get_quantity_renewable(data, 'Onshore Wind', target_idx)
    return output

def get_exist_renewable_fraction_solar_r(data: Data, yr_cal: int = None):
    print('Getting existing solar capacity fraction (all years, solver ceiling)...', flush=True)
    # Existing real-world capacity and LUTO-simulated capacity compete for the same
    # cell space [0, 1]. We lock the maximum existing fraction (cumulative 2000-2035)
    # in advance so that simulated + existing never exceeds 1 in any period.
    # Using all years (yr_cal=99999) keeps the ceiling fixed across solver calls,
    # preventing lb > ub when new real-world capacity enters mid-simulation.
    return ag_quantity.get_existing_renewable_dvar_fraction(data, 'Utility Solar PV', 99999)

def get_exist_renewable_fraction_wind_r(data: Data, yr_cal: int = None):
    print('Getting existing wind capacity fraction (all years, solver ceiling)...', flush=True)
    # Same rationale as solar: lock maximum existing fraction to prevent simulated + existing > 1.
    return ag_quantity.get_existing_renewable_dvar_fraction(data, 'Onshore Wind', 99999)

def get_exist_renewable_capacity_by_state_input(data: Data, yr_cal: int):
    print('Getting existing renewable capacity by state...', flush=True)
    return ag_quantity.get_exist_renewable_capacity_by_state(data, yr_cal)

def get_region_state_r(data: Data):
    print('Getting region state index for each cell...', flush = True)
    return data.REGION_STATE_CODE

def get_region_state_name2idx(data: Data):
    print('Getting map of region state names to indices...', flush = True)
    return data.REGION_STATE_NAME2CODE

def get_region_NRM_names_r(data: Data):
    print('Getting region NRM names for each cell...', flush = True)
    return data.REGION_NRM_NAME

def get_ag_man_c_mrj(data: Data, ag_c_mrj: np.ndarray, target_year):
    print('Getting agricultural management options\' cost effects...', flush = True)
    output = ag_cost.get_agricultural_management_cost_matrices(data, ag_c_mrj, target_year)
    return output

def get_ag_man_g_mrj(data: Data, target_index):
    print('Getting agricultural management options\' GHG emission effects...', flush = True)
    return ag_ghg.get_agricultural_management_ghg_matrices(data, target_index)

def get_ag_man_q_mrj(data: Data, target_index, ag_q_mrp: np.ndarray):
    print('Getting agricultural management options\' quantity effects...', flush = True)
    output = ag_quantity.get_agricultural_management_quantity_matrices(data, ag_q_mrp, target_index)
    return output

def get_ag_man_r_mrj(data: Data, target_index, ag_r_mrj: np.ndarray):
    print('Getting agricultural management options\' revenue effects...', flush = True)
    output = ag_revenue.get_agricultural_management_revenue_matrices(data, ag_r_mrj, target_index)
    return output

def get_ag_man_t_mrj(data: Data, target_index):
    print('Getting agricultural management options\' transition cost effects...', flush = True)
    output = ag_transition.get_agricultural_management_transition_matrices(data, target_index)
    return output

def get_ag_man_w_mrj(data: Data, target_index):
    print('Getting agricultural management options\' water yield effects...', flush = True)
    output = ag_water.get_agricultural_management_water_matrices(data, target_index)
    return output

def get_ag_man_limits(data: Data, target_index):
    print('Getting agricultural management options\' adoption limits...', flush = True)
    output = ag_transition.get_agricultural_management_adoption_limits(data, target_index)
    return output

def get_economic_mrj(
    ag_c_mrj: np.ndarray,
    ag_r_mrj: np.ndarray,
    non_ag_c_rk: np.ndarray,
    non_ag_r_rk: np.ndarray,
    non_ag_t_rk: np.ndarray,
    ag_man_c_mrj: dict[str, np.ndarray],
    ag_man_r_mrj: dict[str, np.ndarray],
    ag_man_t_mrj: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray|dict[str, np.ndarray]]:

    print('Getting base year economic matrix...', flush = True)

    # Land-use TRANSITION costs (ag2ag, ag2nonag, nonag2ag) are NOT baked here. They are charged in the
    # solver against the per-source delta vars via the source-keyed flow_cost dicts (Σ flow_cost·D).
    # get_economic_mrj is pure operating economics: revenue − (production cost). Two non-flow
    # terms remain: non_ag_t_rk (nonag→nonag — currently a ZERO matrix, since non-ag↛non-ag is disallowed;
    # kept as a hook if it's ever priced) and ag_man_t_mrj (ag-management adoption cost).
    if settings.OBJECTIVE == "maxprofit":
        # Pre-calculate profit (revenue minus cost) for each land use
        ag_obj_mrj = ag_r_mrj - ag_c_mrj
        non_ag_obj_rk = non_ag_r_rk - (non_ag_c_rk + non_ag_t_rk)

        # Get effects of alternative agr. management options (stored in a dict)
        ag_man_objs = {
            am: ag_man_r_mrj[am] - (ag_man_c_mrj[am] + ag_man_t_mrj[am])
            for am in settings.AG_MANAGEMENTS_TO_LAND_USES
        }

    elif settings.OBJECTIVE == "mincost":
        # Pre-calculate sum of production costs (land-use transition cost enters via flow_cost in the solver)
        ag_obj_mrj = ag_c_mrj
        non_ag_obj_rk = non_ag_c_rk + non_ag_t_rk

        # Store calculations for each agricultural management option in a dict
        ag_man_objs = {
            am: (ag_man_c_mrj[am] + ag_man_t_mrj[am])
            for am in settings.AG_MANAGEMENTS_TO_LAND_USES
        }

    else:
        raise ValueError("Unknown objective!")

    ag_obj_mrj = np.nan_to_num(ag_obj_mrj)
    non_ag_obj_rk = np.nan_to_num(non_ag_obj_rk)
    ag_man_objs = {am: np.nan_to_num(arr) for am, arr in ag_man_objs.items()}

    return [ag_obj_mrj, non_ag_obj_rk, ag_man_objs]

def get_limits(data: Data, yr_cal: int) -> dict[str, Any]:
    """
    Return raw (unscaled) constraint targets for the given calendar year.

    Keys returned depend on active settings:
      'demand', 'water', 'ghg',
      'renewable_Utility Solar PV', 'renewable_Onshore Wind',
      'renewable_Utility Solar PV_exist', 'renewable_Onshore Wind_exist',
      'GBF2', 'GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8',
      'ag_regional_adoption', 'non_ag_regional_adoption', 'non_ag_regional_adoption_sum'

    All values are raw (unscaled); the solver rescales each constraint row together with its
    target (``row_builder.scale_rows``).
    """
    print('Getting environmental limits...', flush = True)
    
    limits = {}
    
    # Clamped again here, not only in Data.__init__: a resumed run loads a pickled Data and never
    # re-runs __init__, so a checkpoint written before the clamp existed would still carry negatives.
    limits['demand'] = np.maximum(data.D_CY[yr_cal - data.YR_CAL_BASE], 0.0)
    
    if settings.WATER_LIMITS == 'on':
        limits['water'] = data.WATER_YIELD_TARGETS
        
    if settings.GHG_EMISSIONS_LIMITS != 'off':
        limits['ghg'] = data.GHG_TARGETS[yr_cal]
        
    if any(settings.RENEWABLES_OPTIONS.values()):
        renewable_targets = data.RENEWABLE_TARGETS.query('Year == @yr_cal').set_index('state')
        limits['renewable_Utility Solar PV'] = renewable_targets.query('tech == "Utility Solar"')['Renewable_Target_MWh'].to_dict()
        limits['renewable_Onshore Wind'] = renewable_targets.query('tech == "Wind"')['Renewable_Target_MWh'].to_dict()
        
        renewable_existing_capacity = get_exist_renewable_capacity_by_state_input(data, yr_cal)
        limits['renewable_Utility Solar PV_exist'] = {state: vals['Utility Solar PV'] for state, vals in renewable_existing_capacity.items()}
        limits['renewable_Onshore Wind_exist']     = {state: vals['Onshore Wind']     for state, vals in renewable_existing_capacity.items()}

    if settings.GBF2_TARGET != 'off':
        limits["GBF2"] = data.get_GBF2_target_for_yr_cal(yr_cal)

    if settings.GBF3_NVIS_TARGET != 'off':
        limits["GBF3_NVIS"] = data.get_GBF3_NVIS_limit_score_inside_LUTO_by_yr(yr_cal)

    if settings.GBF4_TARGET_SNES != 'off':
        limits["GBF4_SNES"] = data.get_GBF4_SNES_target_inside_LUTO_by_year(yr_cal)

    if settings.GBF4_TARGET_ECNES != 'off':
        limits["GBF4_ECNES"] = data.get_GBF4_ECNES_target_inside_LUTO_by_year(yr_cal)

    if settings.GBF8_TARGET != "off":
        limits["GBF8"] = data.get_GBF8_target_inside_LUTO_by_yr(yr_cal)

    if settings.REGIONAL_ADOPTION_CONSTRAINTS != 'off':
        ag_reg_adoption, non_ag_reg_adoption, non_ag_reg_adoption_sum = ag_transition.get_regional_adoption_limits(data, yr_cal)
        limits["ag_regional_adoption"] = ag_reg_adoption
        limits["non_ag_regional_adoption"] = non_ag_reg_adoption
        limits["non_ag_regional_adoption_sum"] = non_ag_reg_adoption_sum

    return limits

def get_obj_block(cols: dict, ag_obj_mrj: np.ndarray, non_ag_obj_rk: np.ndarray, ag_man_objs: dict,
                  flow_cost_ag2ag: dict, flow_cost_ag2nonag: dict, flow_cost_nonag2ag: dict) -> sparse.csr_matrix:
    """The economy coefficients (raw AUD) as a (5 x n_dec) sparse block, one row per ``OBJ_BLOCKS``
    component. Same contract as ``compose_rows`` (drop |q| < SOLVER_COEFF_MIN; the solver floors the
    scaled coefficient): the ag stream on the ag columns, the ag-mgt and non-ag streams on
    their columns, the transition costs negated on the arc columns. ``solve()`` reads the economy
    breakdown as ``obj_block @ x``."""
    AG, AM, NONAG, TRANS_AG, TRANS_NONAG = (OBJ_BLOCKS.index(name) for name in OBJ_BLOCKS)
    table = cols['table']
    m, j, k, local_r = (table[field].values for field in ('m', 'j', 'k', 'local_r'))
    parts = []                                                            # (block row, columns, values) after the drop

    # ── operating economics: one gather over the scored columns, one part per block ──
    coeff = gather_coeffs(cols, ag_obj_mrj, ag_man_objs, non_ag_obj_rk)   # float32, indexed by column
    for block_row, block in ((AG, 'ag'), (AM, 'am'), (NONAG, 'nonag')):
        rows = block_slice(table, block)
        parts.append((block_row, *drop(np.arange(rows.start, rows.stop), coeff[rows])))

    # ── transition costs on the arcs with an ag target: per source run of the block, gathered from the source-keyed cost dicts ──
    for block, sources, flow_cost, block_row in (('ag2ag', cols['sources']['ag'], flow_cost_ag2ag, TRANS_AG),
                                                 ('nonag2ag', cols['sources']['nonag'], flow_cost_nonag2ag, TRANS_AG)):
        src_ptr = table.attrs['src_ptr'][block]
        for src, start, stop in zip(sources, src_ptr[:-1], src_ptr[1:]):
            run = slice(int(start), int(stop))
            parts.append((block_row, *drop(np.arange(run.start, run.stop), -flow_cost[src][m[run], local_r[run], j[run]])))

    # ── transition costs on the ag → non-ag arcs: the cost dict is keyed by target land use k ──
    src_ptr = table.attrs['src_ptr']['ag2nonag']
    for src, start, stop in zip(cols['sources']['ag'], src_ptr[:-1], src_ptr[1:]):
        run = slice(int(start), int(stop))
        cost_by_k = flow_cost_ag2nonag[src]                               # {k: array(ncells_src)}
        to_k = k[run]
        cells = local_r[run]
        arc_cost = np.empty(run.stop - run.start, dtype=np.float32)
        for lu in np.unique(to_k):
            arc_cost[to_k == lu] = cost_by_k[int(lu)][cells[to_k == lu]]
        parts.append((TRANS_NONAG, *drop(np.arange(run.start, run.stop), -arc_cost)))

    row_idx = np.concatenate([np.full(part_vals.size, block_row, dtype=np.int32) for block_row, _, part_vals in parts])
    col_idx = np.concatenate([part_cols for _, part_cols, _ in parts])
    vals = np.concatenate([part_vals for _, _, part_vals in parts])
    block = sparse.csr_matrix((vals, (row_idx, col_idx)), shape=(len(OBJ_BLOCKS), table.attrs['n_dec']))
    block.sum_duplicates()                                                # no column repeats; keeps CSR canonical
    return block


def get_rows(data: Data, base_year: int, target_year: int, cols: dict) -> RowInputs:
    """The row side of one step: every coefficient stream and target the family methods read,
    plus the objective block over the column space."""

    target_index = target_year - data.YR_CAL_BASE
    ag_c_mrj     = get_ag_c_mrj(data, target_index)
    ag_r_mrj     = get_ag_r_mrj(data, target_index)

    # ── Transition costs — SOURCE-KEYED flow-cost dicts ──────────────
    # Sliced by base-year source ("(from_m, from_j)" for ag, "k" for non-ag) over each source's
    # base-year cells; the solver creates a matching delta var per (source, cell, target) and charges
    # Σ flow_cost·D in the objective. get_economic_mrj bakes no land-use transition cost.

    # ag→ag: dict[(from_m, from_j)] → ndarray(NLMS, ncells_src, N_AG_LUS)
    flow_cost_ag2ag = get_ag_t_mrj(data, target_index, base_year)

    # ag→ag transition GHG EMISSIONS (raw t CO2), source-keyed — the physical parallel of
    # flow_cost_ag2ag. The GHG constraint sums Σ flow_ghg·D (source-correct transition emissions).
    trans_ghg_ag2ag                 = ag_ghg.get_ghg_transition_emissions_from_base_year(data, base_year)

    # ag→nonag: the dispatcher gives dict[lu_name → dict[(from_m, from_j)]]; transposed to
    # dict[(from_m, from_j) → dict[k]] so the arcs loop ag sources first.
    flow_cost_ag2nonag = {}
    for lu_name, per_src in non_ag_transition.get_transition_matrix_ag2nonag(data, base_year, target_year).items():
        k = data.NON_AGRICULTURAL_LANDUSES.index(lu_name)
        for src, arr in per_src.items():
            flow_cost_ag2nonag.setdefault(src, {})[k] = arr

    # nonag→ag: the dispatcher gives dict[lu_name → dict[k]]; take the diagonal (cells in non-ag
    # land use k pay only k's own nonag→ag cost).
    flow_cost_nonag2ag = {}
    for lu_name, per_k in non_ag_transition.get_transition_matrix_nonag2ag(data, base_year, target_year).items():
        k = data.NON_AGRICULTURAL_LANDUSES.index(lu_name)
        if k in per_k:
            flow_cost_nonag2ag[k] = per_k[k]

    non_ag_c_rk                     = get_non_ag_c_rk(data, ag_c_mrj, data.lumaps[base_year], target_year)
    non_ag_r_rk                     = get_non_ag_r_rk(data, ag_r_mrj, base_year, target_year)
    non_ag_t_rk                     = get_non_ag_t_rk(data, base_year)

    ag_man_c_mrj                    = get_ag_man_c_mrj(data, ag_c_mrj, target_year)
    ag_man_r_mrj                    = get_ag_man_r_mrj(data, target_index, ag_r_mrj)
    ag_man_t_mrj                    = get_ag_man_t_mrj(data, target_index)
    
    ag_obj_mrj, non_ag_obj_rk,  ag_man_objs = get_economic_mrj(
        ag_c_mrj,
        ag_r_mrj,
        non_ag_c_rk,
        non_ag_r_rk,
        non_ag_t_rk,
        ag_man_c_mrj,
        ag_man_r_mrj,
        ag_man_t_mrj
    )
    

    ag_g_mrj                        = get_ag_g_mrj(data, target_index)
    ag_w_mrj                        = (
        get_ag_w_mrj(data, target_index) if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on' 
        else get_ag_w_mrj(data, target_index, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    )
    ag_q_mrp                        = get_ag_q_mrp(data, target_index)

    non_ag_g_rk                     = get_non_ag_g_rk(data, ag_g_mrj, base_year)
    non_ag_w_rk                     = (
        get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year)
        if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on'
        else get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    )
    non_ag_q_crk                    = get_non_ag_q_crk(data, ag_q_mrp, base_year)

    ag_man_g_mrj                    = get_ag_man_g_mrj(data, target_index)
    ag_man_w_mrj                    = get_ag_man_w_mrj(data, target_index)
    ag_man_q_mrp                    = get_ag_man_q_mrj(data, target_index, ag_q_mrp)
    ag_man_limits                   = get_ag_man_limits(data, target_index)
    
    renewable_solar_r               = get_potential_renewable_solar_r(data, target_index)
    renewable_wind_r                = get_potential_renewable_wind_r(data, target_index)
    exist_renewable_solar_r         = get_exist_renewable_fraction_solar_r(data, target_year)
    exist_renewable_wind_r          = get_exist_renewable_fraction_wind_r(data, target_year)

    region_state_r                  = get_region_state_r(data)
    region_state_name2idx           = get_region_state_name2idx(data)
    region_NRM_names_r              = get_region_NRM_names_r(data)
    
    water_region_indices            = get_w_region_indices(data)
    water_region_names              = get_w_region_names(data)
    
    biodiv_contr_ag_j               = get_ag_biodiv_contr_j(data)
    biodiv_contr_non_ag_k           = get_non_ag_biodiv_impact_k(data)
    biodiv_contr_ag_man             = get_ag_man_biodiv_impacts(data, target_year)

    GBF2_mask_area_r                = get_GBF2_mask_area_r(data)
    GBF3_NVIS_pre_1750_area_vr      = get_GBF3_NVIS_pre_1750_area_vr(data)
    GBF3_NVIS_region_group          = get_GBF3_NVIS_region_group(data)
    GBF4_SNES_pre_1750_area_sr      = get_GBF4_SNES_pre_1750_area_sr(data)
    GBF4_SNES_region_species        = get_GBF4_SNES_region_species(data)
    GBF4_ECNES_pre_1750_area_sr     = get_GBF4_ECNES_pre_1750_area_sr(data)
    GBF4_ECNES_region_species       = get_GBF4_ECNES_region_species(data)
    GBF8_pre_1750_area_sr           = get_GBF8_pre_1750_area_sr(data, target_year)
    GBF8_region_species             = get_GBF8_region_species(data)

    limits = get_limits(data, target_year)

    # The coefficient streams leave here raw and float32 (the DTYPE POLICY in row_builder):
    # every constraint block is row-rescaled in the solver (row_builder.scale_rows, per-row
    # factor kept there) and the objective is raw AUD / 1e6.
    ag_obj_mrj = ag_obj_mrj.astype(np.float32)
    non_ag_obj_rk = non_ag_obj_rk.astype(np.float32)
    ag_man_objs = {am: arr.astype(np.float32) for am, arr in ag_man_objs.items()}
    flow_cost_ag2ag    = {src: arr.astype(np.float32) for src, arr in flow_cost_ag2ag.items()}
    flow_cost_ag2nonag = {src: {k: arr.astype(np.float32) for k, arr in per_k.items()} for src, per_k in flow_cost_ag2nonag.items()}
    flow_cost_nonag2ag = {k: arr.astype(np.float32) for k, arr in flow_cost_nonag2ag.items()}
    ag_q_mrp = ag_q_mrp.astype(np.float32)
    non_ag_q_crk = non_ag_q_crk.astype(np.float32)
    ag_man_q_mrp = {am: arr.astype(np.float32) for am, arr in ag_man_q_mrp.items()}
    ag_g_mrj = ag_g_mrj.astype(np.float32)
    non_ag_g_rk = non_ag_g_rk.astype(np.float32)
    ag_man_g_mrj = {am: arr.astype(np.float32) for am, arr in ag_man_g_mrj.items()}
    trans_ghg_ag2ag = {src: arr.astype(np.float32) for src, arr in trans_ghg_ag2ag.items()}
    ag_w_mrj = ag_w_mrj.astype(np.float32)
    non_ag_w_rk = non_ag_w_rk.astype(np.float32)
    ag_man_w_mrj = {am: arr.astype(np.float32) for am, arr in ag_man_w_mrj.items()}
    renewable_solar_r = renewable_solar_r.astype(np.float32)
    renewable_wind_r = renewable_wind_r.astype(np.float32)

    offland_ghg = (
        data.OFF_LAND_GHG_EMISSION_C[target_index]                       # raw tCO2e (row-rescaled in the solver)
        if settings.GHG_EMISSIONS_LIMITS != 'off'
        else 0.0
    )

    # The objective block over the column space
    obj_block       = get_obj_block(cols, ag_obj_mrj, non_ag_obj_rk, ag_man_objs,
                                    flow_cost_ag2ag, flow_cost_ag2nonag, flow_cost_nonag2ag)

    return RowInputs(
        base_year=base_year,
        target_year=target_year,

        ag_g_mrj=ag_g_mrj,
        ag_w_mrj=ag_w_mrj,
        ag_q_mrp=ag_q_mrp,
        non_ag_g_rk=non_ag_g_rk,
        non_ag_w_rk=non_ag_w_rk,
        non_ag_q_crk=non_ag_q_crk,
        ag_man_g_mrj=ag_man_g_mrj,
        ag_man_w_mrj=ag_man_w_mrj,
        ag_man_q_mrp=ag_man_q_mrp,
        ag_man_limits=ag_man_limits,

        renewable_solar_r=renewable_solar_r,
        renewable_wind_r=renewable_wind_r,
        exist_renewable_solar_r=exist_renewable_solar_r,
        exist_renewable_wind_r=exist_renewable_wind_r,

        region_state_r=region_state_r,
        region_state_name2idx=region_state_name2idx,
        region_NRM_names_r=region_NRM_names_r,
        water_region_indices=water_region_indices,
        water_region_names=water_region_names,

        biodiv_contr_ag_j=biodiv_contr_ag_j,
        biodiv_contr_non_ag_k=biodiv_contr_non_ag_k,
        biodiv_contr_ag_man=biodiv_contr_ag_man,

        GBF2_mask_area_r=GBF2_mask_area_r,
        GBF3_NVIS_pre_1750_area_vr=GBF3_NVIS_pre_1750_area_vr,
        GBF3_NVIS_region_group=GBF3_NVIS_region_group,
        GBF4_SNES_pre_1750_area_sr=GBF4_SNES_pre_1750_area_sr,
        GBF4_SNES_region_species=GBF4_SNES_region_species,
        GBF4_ECNES_pre_1750_area_sr=GBF4_ECNES_pre_1750_area_sr,
        GBF4_ECNES_region_species=GBF4_ECNES_region_species,
        GBF8_pre_1750_area_sr=GBF8_pre_1750_area_sr,
        GBF8_region_species=GBF8_region_species,

        commodity_names=data.COMMODITIES,
        offland_ghg=offland_ghg,
        lu2pr_pj=data.LU2PR,
        pr2cm_cp=data.PR2CM,
        limits=limits,
        real_area=data.REAL_AREA,
        ag_mask_proportion_r=data.AG_MASK_PROPORTION_R,
        trans_ghg_ag2ag=trans_ghg_ag2ag,
        obj_block=obj_block,
    )


# ═══════════════════════════ row blocks: one generator per constraint family, in model order ═══════════════════════════
#
# A ROW BLOCK is one family's constraints as a keyed ``xr.Dataset`` over dim ``row``: a MultiIndex
# coordinate ``row`` (the family's natural key — (region, species), (cell,), (am, lu, lm, cell) …),
# per-row ``rhs`` (raw units), ``sense``, ``name`` (the Gurobi ConstrName) and ``scale`` (the
# row_builder.scale_rows factor, 1 where the family is not rescaled), and in ``attrs``: ``family``,
# ``group`` (tools.CONSTRAINT_GROUPS key), ``A`` (the CSR over Var.index, rows aligned with dim
# ``row``, row-scaled where ``scale`` != 1) and ``rescaled``. The solver adds ``constr`` (the Gurobi
# handle per row) after ``addMConstr`` and ``lhs`` (the raw-unit row value at the solution) after
# the solve. Every generator is ``family(rows, cols) -> Dataset | None`` (None = family off).


def make_block(family: str, group: str, keys: dict, A: sparse.csr_matrix, rhs, sense, names,
               scale=None, **attrs) -> xr.Dataset:
    """One family's rows as a keyed Dataset. ``keys`` = {level: array} (one array per key level,
    all of length n_rows); ``sense`` a single character or an array of characters."""
    n_rows = A.shape[0]
    row_index = pd.MultiIndex.from_arrays([np.asarray(level) for level in keys.values()], names=list(keys))
    coords = xr.Coordinates.from_pandas_multiindex(row_index, 'row')
    sense = np.full(n_rows, sense, dtype=object) if isinstance(sense, str) else np.asarray(sense, dtype=object)
    return xr.Dataset(
        dict(rhs=(('row',), np.asarray(rhs, dtype=np.float64)),
             sense=(('row',), sense),
             name=(('row',), np.asarray(names, dtype=object)),
             scale=(('row',), np.ones(n_rows, dtype=np.float64) if scale is None else np.asarray(scale, dtype=np.float64))),
        coords=coords,
        attrs=dict(family=family, group=group, A=A, rescaled=scale is not None, **attrs))


def block_keys(block: xr.Dataset) -> list:
    """The row keys of a block as tuples, in row order."""
    return list(block.indexes['row'])


# ── spine: the structural rows ────────────────────────────────────────────────────────────────

def renewable_ceiling_rows(rows: RowInputs, cols: dict):
    """Renewable ag-management options: simulated and existing capacity compete for the cell's
    space [0, ag_mask]. One row per (am, cell) with existing capacity,
    Σ_{m, j} X_am[am, m, j, r] ≤ max(ag_mask[r] − exist_r[r], 0). exist_r is the total across ALL
    data years, so the ceiling never decreases between periods (lb(t) ≤ ceiling always holds)."""
    n_all = cols['table'].attrs['n_all']
    am_ds = cols['am']
    am_col_smr = am_ds['col'].values
    am_of_slot = am_ds['am'].values
    ag_mask = rows.ag_mask_proportion_r
    parts = []
    names = []
    key_am = []
    key_cell = []
    rhs = []
    for am in am_ds.attrs['agman2lu']:
        if am not in settings.RENEWABLES_OPTIONS:
            continue
        am_name = tools.am_name_snake_case(am)
        exist_r = rows.exist_renewable_solar_r if am == "Utility Solar PV" else rows.exist_renewable_wind_r
        # the option's columns, grouped by cell
        option_col = am_col_smr[am_of_slot == am]                            # (slots of the option, lm, cell)
        _, _, col_cell = np.nonzero(option_col >= 0)
        col_idx = option_col[option_col >= 0]
        cells, cell_of_col = np.unique(col_cell, return_inverse=True)    # the option's cells, ascending
        existing_cap = exist_r[cells]
        keep_cell = existing_cap != 0                                    # no existing capacity -> no ceiling row
        n_rows = int(keep_cell.sum())
        if not n_rows:
            continue
        row_of_cell = np.full(cells.size, -1, dtype=np.int64)
        row_of_cell[keep_cell] = np.arange(n_rows)
        row_idx = row_of_cell[cell_of_col]
        in_row = row_idx >= 0
        parts.append(sparse.csr_matrix((np.ones(int(in_row.sum())), (row_idx[in_row], col_idx[in_row])), shape=(n_rows, n_all)))
        rhs.append(np.maximum(ag_mask[cells[keep_cell]] - existing_cap[keep_cell], 0.0))   # cell space left for simulated capacity
        names += [f"const_{am_name}_solvable_ub_{cell}".replace(" ", "_") for cell in cells[keep_cell]]
        key_am += [am] * n_rows
        key_cell.append(cells[keep_cell])
    if not parts:
        return None
    return make_block('renewable_ceiling', 'ag_mgt_ub', dict(am=np.array(key_am, dtype=object), cell=np.concatenate(key_cell)),
                      sparse.vstack(parts, format='csr'), np.concatenate(rhs), '<', names)


def cell_usage_rows(rows: RowInputs, cols: dict):
    """Every cell's ag + non-ag shares sum to its base-year agricultural proportion: group the table's
    ag, non-ag and slack columns by cell, one row per cell with a slack column (the cells that can meet
    it). Ranged, not ==: presolve folds the node-balance rows into this one and compares two constants
    summed along different float32 paths (up to ~1.75x FeasibilityTol apart) with NO tolerance; the
    ±10x Ftol band absorbs that, and conservation still pins the cell total, so the band is not
    exploitable. Stored the way Gurobi stores an addRange row, Σ X + slack = hi, the slack a column of
    the space (lb 0, ub = hi − lo)."""
    table = cols['table']
    n_all = table.attrs['n_all']
    cell = table['cell'].values
    ag, nonag, slack = (block_slice(table, block) for block in ('ag', 'nonag', 'cell_usage'))
    row_cells = cell[slack]
    n_rows = row_cells.size
    ncells = cols['ag'].sizes['cell']
    row_of_cell = np.full(ncells, -1, dtype=np.int64)
    row_of_cell[row_cells] = np.arange(n_rows)
    columns = np.concatenate([np.arange(ag.start, ag.stop), np.arange(nonag.start, nonag.stop), np.arange(slack.start, slack.stop)])
    row_idx = row_of_cell[cell[columns]]
    in_row = row_idx >= 0
    A = sparse.csr_matrix((np.ones(int(in_row.sum())), (row_idx[in_row], columns[in_row])), shape=(n_rows, n_all))
    hi = rows.ag_mask_proportion_r[row_cells].astype(np.float64) + 10 * settings.FEASIBILITY_TOLERANCE   # the top of the band (widened before the band is applied)
    return make_block('cell_usage', 'cell_usage', dict(cell=row_cells), A, hi, '=',
                      [f"const_cell_usage_{cell}" for cell in row_cells], n_skipped=int(ncells - n_rows))


def ag_mgt_link_rows(rows: RowInputs, cols: dict):
    """Ag-management variables cannot exceed the value of the agricultural variable: one row per
    (am, land use, lm, cell) with an ag column — the ag cube and the am cube aligned on (lm, cell).
    Where the am column exists the row is X_am − X_ag ≤ 0; where it does not (GBF2-excluded /
    savanna-ineligible cell) the row is X_ag ≥ 0 (sense '>'). Row order: (am, land use) slot, dry
    then irr, cells ascending."""
    n_all = cols['table'].attrs['n_all']
    ag_col_mjr = cols['ag']['col'].values
    am_ds = cols['am']
    am_col_smr = am_ds['col'].values
    am_of_slot = am_ds['am'].values
    j_of_slot = am_ds['j'].values
    am_list = list(am_ds.attrs['agman2lu'])
    row_idx = []
    col_idx = []
    vals = []
    senses = []
    names = []
    key_am = []
    key_lu = []
    key_lm = []
    key_cell = []
    n_rows = 0
    for slot in range(am_ds.sizes['slot']):
        am = am_of_slot[slot]
        j = int(j_of_slot[slot])
        for m, lm in ((0, 'dry'), (1, 'irr')):
            cells = np.flatnonzero(ag_col_mjr[m, j] >= 0)                # the ag columns of (m, j), cells ascending
            ag_cols = ag_col_mjr[m, j, cells]
            am_cols = am_col_smr[slot, m, cells]
            has_am = am_cols >= 0
            slot_rows = n_rows + np.arange(cells.size)
            # X_ag: −1 on the '<' rows (X_am − X_ag ≤ 0), +1 on the '>' rows (X_ag ≥ 0)
            row_idx.append(slot_rows)
            col_idx.append(ag_cols)
            vals.append(np.where(has_am, -1.0, 1.0))
            # X_am: +1 where the am column exists
            row_idx.append(slot_rows[has_am])
            col_idx.append(am_cols[has_am])
            vals.append(np.ones(int(has_am.sum())))
            senses.append(np.where(has_am, '<', '>'))
            names += [f"const_ag_mam_{lm}_usage_{am}_{j}_{cell}".replace(" ", "_") for cell in cells]
            key_am.append(np.full(cells.size, am_list.index(am), dtype=np.int32))
            key_lu.append(np.full(cells.size, j, dtype=np.int32))
            key_lm.append(np.full(cells.size, m, dtype=np.int8))
            key_cell.append(cells)
            n_rows += cells.size
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_rows, n_all))
    return make_block('ag_mgt_link', 'ag_mgt_link',                      # integer key levels (am = index into am_list, lm = 0 dry / 1 irr)
                      dict(am=np.concatenate(key_am), lu=np.concatenate(key_lu), lm=np.concatenate(key_lm), cell=np.concatenate(key_cell)),
                      A, np.zeros(n_rows), np.concatenate(senses), names)


def ag_mgt_adoption_rows(rows: RowInputs, cols: dict):
    """Adoption limits: one row per (am, land use), Σ am columns − limit · Σ ag columns ≤ 0
    (Σam ≤ limit · Σag with the RHS moved to the LHS); zero coefficients (limit = 0) are dropped."""
    n_all = cols['table'].attrs['n_all']
    ag_col_mjr = cols['ag']['col'].values
    am_ds = cols['am']
    am_col_smr = am_ds['col'].values
    am_of_slot = am_ds['am'].values
    j_of_slot = am_ds['j'].values
    row_idx = []
    col_idx = []
    vals = []
    names = []
    key_am = []
    key_lu = []
    for row, slot in enumerate(range(am_ds.sizes['slot'])):
        am = am_of_slot[slot]
        j = int(j_of_slot[slot])
        adoption_limit = float(np.float64(rows.ag_man_limits[am][j]))
        am_cols = am_col_smr[slot][am_col_smr[slot] >= 0]                # both lm, every cell with an am column
        ag_cols = ag_col_mjr[:, j][ag_col_mjr[:, j] >= 0]                # dry + irr feasible cells
        row_idx += [np.full(am_cols.size, row), np.full(ag_cols.size, row)]
        col_idx += [am_cols, ag_cols]
        vals += [np.ones(am_cols.size), np.full(ag_cols.size, -adoption_limit)]
        names.append(f"const_ag_mam_adoption_limit_{am}_{j}".replace(" ", "_"))
        key_am.append(am)
        key_lu.append(j)
    n_rows = len(names)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_rows, n_all))
    A.eliminate_zeros()
    return make_block('ag_mgt_adoption', 'ag_mgt_adopt', dict(am=np.array(key_am, dtype=object), lu=np.array(key_lu, dtype=np.int32)),
                      A, np.zeros(n_rows), '<', names)


# ── policy families ────────────────────────────────────────────────────────────────────────────

def demand_rows(rows: RowInputs, cols: dict):
    """Hard demand constraints: per-commodity quantity rows over the scored columns;
    equality where the DEMAND_BOUNDS lb==ub, else the SAME LHS row twice under '>' lb and '<' ub.

    Per (m, land use) group of the ag columns and per ((am, land use) slot, m) group of the
    ag-mgt columns, the commodity coefficients are ``jc[c, cell] = Σ_p pr2cm[c, p] · q[m, cell, p]``
    over the land use's active products; non-ag columns carry ``non_ag_q_crk[c, cell, k]``.
    ``attrs['q_block']`` keeps the unscaled per-commodity LHS (production reporting)."""
    print("│   ├── Adding <hard> demand constraints (equality where lb==ub, else lower + upper)...")
    table = cols['table']
    n_all = table.attrs['n_all']
    m, j, k, cell, am_idx = (table[field].values for field in ('m', 'j', 'k', 'cell', 'am_idx'))
    ag, nonag = block_slice(table, 'ag'), block_slice(table, 'nonag')
    ncms = rows.ncms
    row_idx = []
    col_idx = []
    vals = []

    def put(commodity_coeffs: np.ndarray, columns: np.ndarray):
        """One (commodity × column) coefficient block into the COO lists, dropped by the contract."""
        for c_idx in range(ncms):
            kept_cols, kept_vals = drop(columns, commodity_coeffs[c_idx])
            row_idx.append(np.full(kept_cols.size, c_idx, dtype=np.int32))
            col_idx.append(kept_cols)
            vals.append(kept_vals)

    # ── the per-commodity LHS (q_block): the ag columns per (m, land use), the ag-mgt columns per (slot, m), the non-ag columns per k ──
    for lu in range(cols['ag'].sizes['lu']):
        active_p = np.where(rows.lu2pr_pj[:, lu])[0]
        if not active_p.size:
            continue
        for lm in (0, 1):
            group = ag.start + np.flatnonzero((m[ag] == lm) & (j[ag] == lu))
            if group.size:
                put(rows.pr2cm_cp[:, active_p] @ rows.ag_q_mrp[lm, cell[group], :][:, active_p].T, group)
    for start, stop in am_runs(table, 'slot'):
        run = slice(start, stop)
        option = table.attrs['options'][am_idx[start]]
        active_p = np.where(rows.lu2pr_pj[:, j[start]])[0]
        if not active_p.size:
            continue
        for lm in (0, 1):
            group = run.start + np.flatnonzero(m[run] == lm)
            if group.size:
                put(rows.pr2cm_cp[:, active_p] @ rows.ag_man_q_mrp[option][lm, cell[group], :][:, active_p].T, group)
    for lu in np.unique(k[nonag]):
        group = nonag.start + np.flatnonzero(k[nonag] == lu)
        put(rows.non_ag_q_crk[:, cell[group], lu], group)
    q_block = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(ncms, n_all))
    q_block.sum_duplicates()
    q_block.sort_indices()

    # ── the bound rows: the LHS row once under '=' where lb == ub, else twice under '>' lb and '<' ub ──
    lhs_row = []
    senses = []
    rhs = []
    names = []
    key_commodity = []
    key_bound = []
    for c_idx, c_name in enumerate(rows.commodity_names):
        lb, ub = settings.DEMAND_BOUNDS[c_name]
        demand = rows.limits['demand'][c_idx]
        bounds = [('eq', '=', lb)] if lb == ub else [('lower', '>', lb), ('upper', '<', ub)]
        for bound, sense, factor in bounds:
            lhs_row.append(c_idx)
            senses.append(sense)
            rhs.append(demand * factor)
            names.append(f"demand_hard_bound_{bound}[{c_idx}]")
            key_commodity.append(c_idx)
            key_bound.append(bound)
    block, rhs, scale = scale_rows(q_block[lhs_row], rhs)                # row rescale, factors kept
    return make_block('demand', 'demand', dict(commodity=np.array(key_commodity, dtype=np.int32), bound=np.array(key_bound, dtype=object)),
                      block, rhs, np.array(senses, dtype=object), names, scale, q_block=q_block)


def ghg_rows(rows: RowInputs, cols: dict):
    """Hard GHG emissions cap: one global row over land-use, ag-management, non-ag and
    transition-arc emissions, Σ ghg · X ≤ limit − offland."""
    if settings.GHG_EMISSIONS_LIMITS == "off":
        print("│   ├── TURNING OFF GHG emissions constraints ...")
        return None
    ghg_limit_raw = rows.limits["ghg"]
    print(f"│   ├── Adding <hard> constraints for GHG emissions: {ghg_limit_raw:,.0f} tCO2e")
    table = cols['table']
    n_all = table.attrs['n_all']
    m, j, local_r = (table[field].values for field in ('m', 'j', 'local_r'))
    # land-use, ag-management and non-ag emissions on the scored columns
    coeff = gather_coeffs(cols, rows.ag_g_mrj, rows.ag_man_g_mrj, rows.non_ag_g_rk)
    kept_cols, kept_vals = drop(np.arange(coeff.size), coeff)
    col_idx = [kept_cols]
    vals = [kept_vals]
    # transition emissions on the ag → ag arcs: per source run of the block, a float32 gather of the delta emissions
    src_ptr = table.attrs['src_ptr']['ag2ag']
    for src, start, stop in zip(cols['sources']['ag'], src_ptr[:-1], src_ptr[1:]):
        run = slice(int(start), int(stop))
        if run.stop == run.start:
            continue
        kept_cols, kept_vals = drop(np.arange(run.start, run.stop), rows.trans_ghg_ag2ag[src][m[run], local_r[run], j[run]])
        col_idx.append(kept_cols)
        vals.append(kept_vals)
    col_idx = np.concatenate(col_idx)
    vals = np.concatenate(vals)
    row = sparse.csr_matrix((vals, (np.zeros(col_idx.size, dtype=np.int32), col_idx)), shape=(1, n_all))
    row.sum_duplicates()
    row.sort_indices()
    rhs = np.asarray(ghg_limit_raw - rows.offland_ghg, dtype=np.float64).ravel()   # offland_ghg: 1-element array
    row, rhs, scale = scale_rows(row, rhs)                                # row rescale, factor kept
    return make_block('ghg', 'ghg', dict(row_key=np.array(['ghg'], dtype=object)), row, rhs, '<', ["ghg_emissions_limit_ub"], scale)


def _bio_block(family, group, key_names, rows: RowInputs, cols: dict, pairs, v_limits, layer_of, skip_nonpositive: bool, name_of):
    """Shared body of the GBF families: one weighting row per active key (the region-masked layer),
    composed over the scored columns with the biodiversity contribution, then row-rescaled.
    ``layer_of(key)`` returns the layer over cells; ``skip_nonpositive`` = skip when the raw target
    is <= 0 (GBF4/8) vs < 0 (GBF3: a ZERO target still adds a row); rows whose region-masked
    layer has no positive cell are skipped."""
    bio_c = rows.bio_coeffs(cols)
    reg_matrix = rows.region_NRM_names_r
    val_rows = []
    names = []
    rhs = []
    kept = []
    for key in pairs:
        lb_raw = v_limits.sel(dict(layer=key)).item()
        if (lb_raw <= 0) if skip_nonpositive else (lb_raw < 0):
            continue
        val_row = layer_of(key)
        region = key[0]
        if region != "AUSTRALIA":                                        # NRM scope: mask non-region cells
            val_row = np.where(reg_matrix == region, val_row, 0)
        if not (val_row > 0).any():
            continue
        val_rows.append(val_row)
        names.append(name_of(key))
        rhs.append(lb_raw)
        kept.append(key)
    print(f"│   │   │   ├── {len(kept)} constraint(s) added, {len(pairs) - len(kept)} skipped")
    if not val_rows:
        return None
    block, rhs, scale = scale_rows(compose_rows(cols, bio_c, val_rows), rhs)   # compose + row rescale, factors kept
    keys = {lvl: np.array([k[i] for k in kept], dtype=object) for i, lvl in enumerate(key_names)}
    return make_block(family, group, keys, block, rhs, '>', names, scale)


def gbf2_rows(rows: RowInputs, cols: dict):
    """GBF2 priority degraded areas: the bio contribution at every scored column weighted by
    GBF2_mask_area_r, which is ZERO off-mask (off-mask columns get coefficient 0 and are dropped). One row."""
    if settings.GBF2_TARGET == "off":
        print("│   │   ├── TURNING OFF constraints for biodiversity GBF 2...")
        return None
    print(f'│   │   ├── Adding constraints for biodiversity GBF 2: {rows.limits["GBF2"]:15,.0f}')
    row = compose_rows(cols, rows.bio_coeffs(cols), [rows.GBF2_mask_area_r])
    row, rhs, scale = scale_rows(row, [rows.limits["GBF2"]])            # row rescale, factor kept
    return make_block('GBF2', 'bio_gbf2', dict(row_key=np.array(['gbf2'], dtype=object)), row, rhs, '>',
                      ["bio_GBF2_priority_degraded_area_limit"], scale)


def gbf3_rows(rows: RowInputs, cols: dict):
    if settings.GBF3_NVIS_TARGET == "off":
        print("│   │   ├── TURNING OFF constraints for biodiversity GBF 3 NVIS")
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 3 NVIS...")
    val_matrix = rows.GBF3_NVIS_pre_1750_area_vr                        # xr [group, cell]
    return _bio_block('GBF3_NVIS', 'bio_nvis', ('region', 'item'), rows, cols, rows.GBF3_NVIS_region_group, rows.limits["GBF3_NVIS"],
                      lambda key: val_matrix.sel(group=key[1], drop=True).data, False,
                      lambda key: f"bio_GBF3_NVIS_limit_{key[0]}_{key[1]}".replace(" ", "_"))


def gbf4_snes_rows(rows: RowInputs, cols: dict):
    if settings.GBF4_TARGET_SNES == 'off':
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 SNES...')
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 4 SNES ...")
    val_matrix = rows.GBF4_SNES_pre_1750_area_sr                        # xr [layer=(species, presence), cell]
    return _bio_block('GBF4_SNES', 'bio_snes', ('region', 'item', 'presence'), rows, cols, rows.GBF4_SNES_region_species, rows.limits["GBF4_SNES"],
                      lambda key: val_matrix.sel(dict(layer=(key[1], key[2])), drop=True).values, True,
                      lambda key: f"bio_GBF4_SNES_limit_{key[0]}_{key[1]}_{key[2]}".replace(" ", "_"))


def gbf4_ecnes_rows(rows: RowInputs, cols: dict):
    if settings.GBF4_TARGET_ECNES == 'off':
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 4 ECNES...')
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 4 ECNES ...")
    val_matrix = rows.GBF4_ECNES_pre_1750_area_sr                       # xr [layer=(community, presence), cell]
    return _bio_block('GBF4_ECNES', 'bio_ecnes', ('region', 'item', 'presence'), rows, cols, rows.GBF4_ECNES_region_species, rows.limits["GBF4_ECNES"],
                      lambda key: val_matrix.sel(dict(layer=(key[1], key[2])), drop=True).values, True,
                      lambda key: f"bio_GBF4_ECNES_limit_{key[0]}_{key[1]}_{key[2]}".replace(" ", "_"))


def gbf8_rows(rows: RowInputs, cols: dict):
    if settings.GBF8_TARGET == "off":
        print('│   │   ├── TURNING OFF constraints for biodiversity GBF 8 ...')
        return None
    print("│   │   ├── Adding constraints for biodiversity GBF 8 ...")
    val_matrix = rows.GBF8_pre_1750_area_sr                             # xr [species, cell]
    return _bio_block('GBF8', 'bio_gbf8', ('region', 'item'), rows, cols, rows.GBF8_region_species, rows.limits["GBF8"],
                      lambda key: val_matrix.sel(species=key[1], drop=True).data, True,
                      lambda key: f"bio_GBF8_limit_{key[0]}_{key[1]}".replace(" ", "_"))


def _regional_adoption_family(family, group, key_names, caps, cols_all, r_all, sel_of, name_of, rhs_of, rows, cols):
    """One regional-adoption block: Σ real_area[r] · X over the region's cells ≤ cap per (region, land use)."""
    n_all = cols['table'].attrs['n_all']
    real_area = rows.real_area
    parts = []
    names = []
    rhs = []
    keys = []
    for cap in caps:
        region_cells = cap[-2]
        if len(region_cells) == 0:
            print(f"│   │   │   ├── SKIPPING {name_of(cap)} (no cells at this resolution)")
            continue
        print(f"│   │   │   ├── Adding constraint {name_of(cap)} <= {cap[-1]:,.0f} HA...")
        selected = np.flatnonzero(sel_of(cap, region_cells))
        parts.append(drop(cols_all[selected], real_area[r_all[selected]].astype(np.float32)))
        names.append(name_of(cap))
        rhs.append(rhs_of(cap))
        keys.append(cap[:len(key_names)])
    if not parts:
        return None
    row_idx = np.concatenate([np.full(part_cols.size, row) for row, (part_cols, _) in enumerate(parts)])
    A = sparse.csr_matrix((np.concatenate([part_vals for _, part_vals in parts]), (row_idx, np.concatenate([part_cols for part_cols, _ in parts]))),
                          shape=(len(parts), n_all))
    return make_block(family, group, {level: np.array([key[i] for key in keys], dtype=object) for i, level in enumerate(key_names)},
                      A, rhs, '<', names)


def regional_adoption_ag_rows(rows: RowInputs, cols: dict):
    """Per-(region, ag land use) caps ('on' mode). Not rescaled (hectares)."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        print("│   │   └── TURNING OFF constraints for regional adoption ...")
        return None
    table = cols['table']
    ag = block_slice(table, 'ag')
    ag_j, ag_r = table['j'].values[ag], table['cell'].values[ag]
    ag_cols = np.arange(ag.start, ag.stop)
    return _regional_adoption_family(
        'regional_adoption_ag', 'adopt_ag', ('region', 'lu'), rows.limits["ag_regional_adoption"], ag_cols, ag_r,
        lambda cap, reg_ind: (ag_j == cap[1]) & np.isin(ag_r, reg_ind),
        lambda cap: f"reg_adopt_limit_ag_{cap[2]}_{cap[0]}".replace(" ", "_"), lambda cap: cap[4], rows, cols)


def _nonag_cap_relax(rows: RowInputs) -> float:
    """Non-reversible plantings saturate the non-ag caps, and last year's solved areas become
    this year's exact lower bounds; float32 noise then puts the locked-in floor a hair over the
    cap, which presolve rejects with NO tolerance. Grow the cap by 1e-6/yr RELATIVE so the RHS
    always recedes ahead of the ratcheting floor (per-step increment ~5e-6 x cap vs float noise
    ~2e-10 x cap). Cap erosion by 2050: ~3e-5 relative. Ag caps need no slack: ag is reversible."""
    return 1 + (rows.target_year - settings.SIM_YEARS[0]) * 1e-6


def regional_adoption_nonag_rows(rows: RowInputs, cols: dict):
    """Per-(region, non-ag land use) caps ('on' mode), with the per-year relaxation."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    table = cols['table']
    nonag = block_slice(table, 'nonag')
    na_k, na_r = table['k'].values[nonag], table['cell'].values[nonag]
    na_cols = np.arange(nonag.start, nonag.stop)
    relax = _nonag_cap_relax(rows)
    return _regional_adoption_family(
        'regional_adoption_nonag', 'adopt_nonag', ('region', 'lu'), rows.limits.get("non_ag_regional_adoption") or [], na_cols, na_r,
        lambda cap, reg_ind: (na_k == cap[1]) & np.isin(na_r, reg_ind),
        lambda cap: f"reg_adopt_limit_non_ag_{cap[2]}_{cap[0]}".replace(" ", "_"), lambda cap: cap[4] * relax, rows, cols)


def regional_adoption_nonag_sum_rows(rows: RowInputs, cols: dict):
    """SUM-of-non-ag caps ('NON_AG_CAP' mode): all non-ag land uses in a region together."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return None
    table = cols['table']
    nonag = block_slice(table, 'nonag')
    na_r = table['cell'].values[nonag]
    na_cols = np.arange(nonag.start, nonag.stop)
    relax = _nonag_cap_relax(rows)
    return _regional_adoption_family(
        'regional_adoption_nonag_sum', 'nonag_cap', ('region',), rows.limits.get("non_ag_regional_adoption_sum") or [], na_cols, na_r,
        lambda cap, reg_ind: np.isin(na_r, reg_ind),
        lambda cap: f"reg_adopt_limit_non_ag_sum_{cap[0]}".replace(" ", "_"), lambda cap: cap[2] * relax, rows, cols)


def water_rows(rows: RowInputs, cols: dict):
    """Water net-yield limits: one row per water region, the region's 0/1 float32 indicator as the
    weighting row over the scored columns (off-region columns give q = 0 and are dropped;
    1.0f x c == c, so the drop test sees the raw coefficient — which can be NEGATIVE)."""
    if settings.WATER_LIMITS != "on":
        print("│   ├── TURNING OFF water usage constraints ...")
        return None
    print("│   ├── Adding constraints for water usage limits...")
    coeff = gather_coeffs(cols, rows.ag_w_mrj, rows.ag_man_w_mrj, rows.non_ag_w_rk)
    val_rows = []
    names = []
    rhs = []
    region_ids = []
    for region_id, water_limit_raw in rows.limits["water"].items():
        region_name = rows.water_region_names[region_id]
        print(f"│   │   ├── target (inside LUTO study area) is {water_limit_raw:15,.0f} ML for {region_name}")
        indicator = np.zeros(cols['ag'].sizes['cell'], dtype=np.float32)
        indicator[rows.water_region_indices[region_id]] = 1.0
        val_rows.append(indicator)
        names.append(f"water_yield_limit_{region_name}".replace(" ", "_"))
        rhs.append(water_limit_raw)
        region_ids.append(region_id)
    if not val_rows:
        return None
    block, rhs, scale = scale_rows(compose_rows(cols, coeff, val_rows), rhs)   # compose + row rescale, factors kept
    return make_block('water', 'water', dict(region=np.array(region_ids)), block, rhs, '>', names, scale)


def renewable_rows(rows: RowInputs, cols: dict):
    """State-level renewable generation targets: one row per (state, type) — the type's ag-mgt
    columns with energy_r coefficients, weighted by an allowed-cells indicator (state region, ACT
    merged into NSW, minus the per-type GBF2/MNES exclusion masks). RHS = target − existing
    capacity. Row inclusion is a CELL-SET rule: a row exists iff at least one compatible land use
    has eligible cells — even if every coefficient there is sub-floor."""
    if not any(settings.RENEWABLES_OPTIONS.values()):
        print("│   ├── TURNING OFF renewable energy constraints ...")
        return None
    print("│   ├── Adding constraints for renewable energy production targets ...")
    re_types = {
        'Utility Solar PV': dict(energy_r=rows.renewable_solar_r, gbf2_mask_idx=cols['mask_gbf2_solar'], mnes_mask_idx=cols['mask_mnes_solar']),
        'Onshore Wind':     dict(energy_r=rows.renewable_wind_r,  gbf2_mask_idx=cols['mask_gbf2_wind'],  mnes_mask_idx=cols['mask_mnes_wind']),
    }
    region_state_name2idx = dict(rows.region_state_name2idx)                # local copy: pop() must not mutate data's dict
    act_code = region_state_name2idx.pop('Australian Capital Territory')
    table = cols['table']
    cell, am_idx = table['cell'].values, table['am_idx'].values
    ncells = cols['ag'].sizes['cell']

    # ── the coefficient per type: energy_r on that type's ag-mgt columns (the option's run of the table), 0 on every other scored column ──
    coeff_of_type = {}
    for start, stop in am_runs(table, 'am_idx'):
        option = table.attrs['options'][am_idx[start]]
        if option in re_types:
            coeff_of_type[option] = np.zeros(table.attrs['n_terms'], dtype=np.float32)
            coeff_of_type[option][start:stop] = re_types[option]['energy_r'][cell[start:stop]]   # float32 yield per cell

    # ── one row per (state, type) with eligible cells ──
    ag_has_col_mjr = cols['ag']['col'].values >= 0
    agman2lu = cols['am'].attrs['agman2lu']
    parts = []
    names = []
    rhs = []
    key_am = []
    key_state = []
    for state_name, state_code in region_state_name2idx.items():
        state_cells = np.where(rows.region_state_r == state_code)[0]
        if state_name == 'New South Wales':                              # ACT counts toward the NSW+ACT target
            state_cells = np.union1d(state_cells, np.where(rows.region_state_r == act_code)[0])
        print(f"│   │   ├── Adding renewable energy constraints for {state_name} ...")
        for am, re_data in re_types.items():
            if not settings.AG_MANAGEMENTS[am]:
                continue
            target_raw = rows.limits[f"renewable_{am}"][state_name]
            exist_power_mwh = rows.limits[f"renewable_{am}_exist"][state_name]
            print(f"│   │   │   ├── target for {am} is {target_raw:5,.0f} MWh  (existing: {exist_power_mwh:5,.0f} MWh)")
            # cell-set row-inclusion rule (NOT a coefficient test): some compatible land use must have eligible cells
            has_cells = False
            for j in agman2lu[am]:
                eligible_cells = np.intersect1d(np.flatnonzero(ag_has_col_mjr[:, j, :].any(axis=0)), state_cells)
                if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS == True:
                    eligible_cells = np.setdiff1d(eligible_cells, re_data['gbf2_mask_idx'])
                if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK == True:
                    eligible_cells = np.setdiff1d(eligible_cells, re_data['mnes_mask_idx'])
                if eligible_cells.size:
                    has_cells = True
                    break
            if not has_cells:
                continue
            allowed = np.zeros(ncells, dtype=np.float32)                 # the weighting row: 1 on the state's allowed cells
            allowed[state_cells] = 1.0
            if settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS == True:
                allowed[re_data['gbf2_mask_idx']] = 0.0
            if settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK == True:
                allowed[re_data['mnes_mask_idx']] = 0.0
            parts.append(compose_rows(cols, coeff_of_type[am], [allowed]))
            names.append(f"renewable_{am}_target_{state_name}".replace(" ", "_"))
            rhs.append(target_raw - exist_power_mwh)                     # raw MWh; row-rescaled below
            key_am.append(am)
            key_state.append(state_name)
    if not parts:
        return None
    block, rhs, scale = scale_rows(sparse.vstack(parts, format='csr'), rhs)   # row rescale, factors kept
    return make_block('renewable', 'renewable', dict(am=np.array(key_am, dtype=object), state=np.array(key_state, dtype=object)),
                      block, rhs, '>', names, scale)


# ── the transition-flow rows: group-bys over the table's arc blocks (ag2ag ∪ ag2nonag ∪ nonag2ag) ──
#    group by the FROM fields + cell  →  source cap:    Σ out ≤ base                 (a ≤ row per source node)
#    group by the node in BOTH roles  →  node balance:  X = base + Σ in − Σ out      (an = row per target column)
#    the inflow cap is not a row: X's own ub (the transition upper bound) and the cell-usage row bound it.

def source_cap_ag_rows(rows: RowInputs, cols: dict):
    """Source cap, ag sources: group the table's ag2ag ∪ ag2nonag rows by source node (from_m, from_j)
    and cell (``local_r``, the cell in the source's list); the group's arcs sum to at most the source's
    base share, Σ out ≤ base[from_m, r, from_j]. Every arc of the group gets +1; a (from, cell) with no
    arcs has no row. This BOUNDS the arc columns (some flow costs are negative) and rules out
    pass-through. The base comes from the ag grid: a source with no X column still caps its outflow."""
    print("│   ├── Adding source-cap (Σ out ≤ base) constraints...")
    table = cols['table']
    n_all = table.attrs['n_all']
    arcs = slice(block_slice(table, 'ag2ag').start, block_slice(table, 'ag2nonag').stop)   # the two blocks are adjacent runs of the table
    from_m, from_j, local_r, cell = (table[field].values[arcs] for field in ('from_m', 'from_j', 'local_r', 'cell'))
    n_lu, stride = cols['ag'].sizes['lu'], cols['ag'].sizes['cell']     # local_r < ncells
    key = (from_m.astype(np.int64) * n_lu + from_j) * stride + local_r  # the source node of every arc
    # one row per distinct key; every arc of the key gets a +1
    unique_keys, first_arc, row_of_arc = np.unique(key, return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(key.size), (row_of_arc, np.arange(arcs.start, arcs.stop))), shape=(unique_keys.size, n_all))
    rhs = cols['ag']['base'].values[from_m[first_arc], from_j[first_arc], cell[first_arc]].astype(np.float64)
    names = [f"srccap_a_{m}_{j}_{r}" for m, j, r in zip(from_m[first_arc], from_j[first_arc], local_r[first_arc])]
    return make_block('source_cap_ag', 'flow_out', dict(from_m=from_m[first_arc], from_j=from_j[first_arc], local_r=local_r[first_arc]),
                      A, rhs, '<', names)


def source_cap_nonag_rows(rows: RowInputs, cols: dict):
    """Source cap, non-ag sources: group the table's nonag2ag rows by (from_k, cell); Σ out ≤ base_nonag[r, from_k]."""
    table = cols['table']
    n_all = table.attrs['n_all']
    arcs = block_slice(table, 'nonag2ag')
    if arcs.stop == arcs.start:
        return None
    from_k, local_r, cell = (table[field].values[arcs] for field in ('from_k', 'local_r', 'cell'))
    stride = cols['ag'].sizes['cell']                                   # local_r < ncells
    key = from_k.astype(np.int64) * stride + local_r                    # the source node of every arc
    unique_keys, first_arc, row_of_arc = np.unique(key, return_index=True, return_inverse=True)
    A = sparse.csr_matrix((np.ones(key.size), (row_of_arc, np.arange(arcs.start, arcs.stop))), shape=(unique_keys.size, n_all))
    rhs = cols['nonag']['base'].values[from_k[first_arc], cell[first_arc]].astype(np.float64)
    names = [f"srccap_n_{k}_{r}" for k, r in zip(from_k[first_arc], local_r[first_arc])]
    return make_block('source_cap_nonag', 'flow_out', dict(from_k=from_k[first_arc], local_r=local_r[first_arc]), A, rhs, '<', names)


def node_balance_rows(rows: RowInputs, cols: dict):
    """Node balance: every target column is a node (m, j, cell) or (k, cell); group the table by the node
    in both roles — the X column (+1, its own node), the arcs whose TO fields (``m, j`` / ``k``) hit the
    node (−1 each, inflow), the arcs whose FROM fields hit it (+1 each, outflow):

        X_ag[m, r, j]  = base_ag[m, r, j]  + Σ in (ag2ag ∪ nonag2ag → (m, j)) − Σ out ((m, j) → ag2ag ∪ ag2nonag)
        X_nonag[r, k]  = base_nonag[r, k]  + Σ in (ag2nonag → k)              − Σ out (k → nonag2ag)

    One row per ag column (column order), then one per (non-ag land use, feasible cell) — every
    non-ag land use, enabled or not: a disabled one's row is a pure inflow guard with no X column,
    stored with the opposite sign (``row_sign``). A source with no X var (banned receiver) has no
    row, so its outflow arcs are dropped. The inflow cap is X's own ub, not a row."""
    print("│   └── Adding node-balance (X = base + Σin − Σout) constraints...")
    table = cols['table']
    n_all = table.attrs['n_all']
    m, j, k, from_m, from_j, from_k, cell = (table[field].values for field in ('m', 'j', 'k', 'from_m', 'from_j', 'from_k', 'cell'))
    ag, ag2ag, ag2nonag, nonag2ag = (block_slice(table, block) for block in ('ag', 'ag2ag', 'ag2nonag', 'nonag2ag'))
    n_lu, ncells = cols['ag'].sizes['lu'], cols['ag'].sizes['cell']

    # ── the rows: one per ag column, then one per (non-ag land use, feasible cell), each keyed by its node ──
    n_ag = ag.stop - ag.start
    ag_m, ag_j, ag_r = m[ag].astype(np.int64), j[ag].astype(np.int64), cell[ag].astype(np.int64)
    nonag_k, nonag_r = np.nonzero(cols['nonag']['ub'].values > 0)       # every feasible entry, enabled land use or not: k then cell
    nonag_k = nonag_k.astype(np.int64)
    nonag_r = nonag_r.astype(np.int64)
    n_nonag = nonag_r.size
    x_col_nonag = cols['nonag']['col'].values[nonag_k, nonag_r]         # -1: a disabled land use has no X column
    row_sign = np.ones(n_ag + n_nonag, dtype=np.float64)                 # rows without an X var: Σin − Σout = −base
    row_sign[n_ag:] = np.where(x_col_nonag >= 0, 1.0, -1.0)

    def ag_node(m_, j_, r_):
        return (m_.astype(np.int64) * n_lu + j_) * ncells + r_

    def nonag_node(k_, r_):
        return k_.astype(np.int64) * ncells + r_

    ag_key = ag_node(ag_m, ag_j, ag_r)                                   # ascending: the ag block is in (lm, lu, cell) order
    nonag_key = nonag_node(nonag_k, nonag_r)                             # ascending: np.nonzero order

    def row_of(keys, node, first_row):
        """The row whose node is ``node`` (a join on the ascending row keys), -1 where no row has it."""
        if not keys.size:
            return np.full(node.size, -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(keys, node), keys.size - 1)
        return np.where(keys[pos] == node, first_row + pos, -1)

    # ── the entries: X on its own row, inflows −1 on the target's row, outflows +1 on the source's row ──
    row_idx = []
    col_idx = []
    vals = []

    def add(row, col, value):
        in_model = row >= 0                                              # no row (banned source / no X var): entry dropped
        row_idx.append(row[in_model].astype(np.int64))
        col_idx.append(col[in_model].astype(np.int64))
        vals.append(value * row_sign[row[in_model]])

    add(np.arange(n_ag), np.arange(ag.start, ag.stop), 1.0)
    add(np.where(x_col_nonag >= 0, n_ag + np.arange(n_nonag), -1), x_col_nonag, 1.0)
    add(row_of(ag_key, ag_node(m[ag2ag], j[ag2ag], cell[ag2ag]), 0), np.arange(ag2ag.start, ag2ag.stop), -1.0)
    add(row_of(ag_key, ag_node(m[nonag2ag], j[nonag2ag], cell[nonag2ag]), 0), np.arange(nonag2ag.start, nonag2ag.stop), -1.0)
    add(row_of(ag_key, ag_node(from_m[ag2ag], from_j[ag2ag], cell[ag2ag]), 0), np.arange(ag2ag.start, ag2ag.stop), 1.0)
    add(row_of(ag_key, ag_node(from_m[ag2nonag], from_j[ag2nonag], cell[ag2nonag]), 0), np.arange(ag2nonag.start, ag2nonag.stop), 1.0)
    add(row_of(nonag_key, nonag_node(k[ag2nonag], cell[ag2nonag]), n_ag), np.arange(ag2nonag.start, ag2nonag.stop), -1.0)
    add(row_of(nonag_key, nonag_node(from_k[nonag2ag], cell[nonag2ag]), n_ag), np.arange(nonag2ag.start, nonag2ag.stop), 1.0)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(n_ag + n_nonag, n_all))

    # ── rhs, names, keys ──
    rhs = np.concatenate([table['base'].values[ag].astype(np.float64),
                          cols['nonag']['base'].values[nonag_k, nonag_r].astype(np.float64) * row_sign[n_ag:]])
    names = ([f"bal_a_{m}_{j}_{r}" for m, j, r in zip(ag_m, ag_j, ag_r)] + [f"bal_n_{k}_{r}" for k, r in zip(nonag_k, nonag_r)])
    keys = dict(kind=np.concatenate([np.zeros(n_ag, dtype=np.int8), np.ones(n_nonag, dtype=np.int8)]),   # 0 = ag row, 1 = non-ag row
                lm_or_k=np.concatenate([ag_m, nonag_k]),
                lu=np.concatenate([ag_j, np.full(n_nonag, -1, dtype=np.int64)]),
                cell=np.concatenate([ag_r, nonag_r]))
    return make_block('node_balance', 'flow_in', keys, A, rhs, '=', names)


def biodiversity_rows(rows: RowInputs, cols: dict):
    """The four GBF families in order, as a list of blocks (None where a family is off)."""
    print("│   ├── Adding constraints for biodiversity...")
    return [gbf2_rows(rows, cols), gbf3_rows(rows, cols), gbf4_snes_rows(rows, cols), gbf4_ecnes_rows(rows, cols), gbf8_rows(rows, cols)]


# The model's row order: every family, in the order the rows are added to Gurobi (the solver's
# path depends on it — ceilings first, the flow rows last). A generator may return None (family
# off), a block, or a list of blocks / Nones.
FAMILIES = (
    renewable_ceiling_rows,
    cell_usage_rows,
    ag_mgt_link_rows,
    ag_mgt_adoption_rows,
    demand_rows,
    ghg_rows,
    biodiversity_rows,
    regional_adoption_ag_rows,
    regional_adoption_nonag_rows,
    regional_adoption_nonag_sum_rows,
    water_rows,
    renewable_rows,
    source_cap_ag_rows,
    source_cap_nonag_rows,
    node_balance_rows,
)
