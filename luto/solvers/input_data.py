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
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional

from luto.data import Data
from luto import settings
import luto.tools as tools
from luto.solvers.row_builder import ag_acct_terms_by_mj, am_terms_by_key, nonag_terms_by_k, keep_terms

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


OBJ_BLOCKS = ('ag', 'am', 'nonag', 'trans_ag', 'trans_nonag')   # rows of SolverInputData.obj_block


@dataclass
class SolverInputData:
    """Everything the solver needs for one (base_year -> target_year) step: coefficient streams
    (raw units, float32), targets, the decision-variable tables and the objective block."""
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

    dvar_base_ag_mrj: np.ndarray                                        # Base-year ag decision variables [m, r, j], clipped into [lb, ub] (the node-balance constant).
    dvar_base_non_ag_rk: np.ndarray                                     # Base-year non-ag decision variables [r, k], clipped into [lb, ub].

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

    savanna_eligible_r: np.ndarray                                      # Cells eligible for savanna burning.
    renewable_GBF2_mask_solar_idx: np.ndarray                           # Index of GBF2 mask for solar renewable exclusion.
    renewable_GBF2_mask_wind_idx: np.ndarray                            # Index of GBF2 mask for wind renewable exclusion.
    renewable_MNES_mask_solar_idx: np.ndarray                           # Index of EPBC MNES mask for solar renewable exclusion.
    renewable_MNES_mask_wind_idx: np.ndarray                            # Index of EPBC MNES mask for wind renewable exclusion.

    commodity_names: list[str]                                          # Commodity names (data.COMMODITIES order).
    offland_ghg: np.ndarray                                             # Target-year GHG emissions from off-land commodities (tCO2e); 0.0 when GHG limits are off.
    lu2pr_pj: np.ndarray                                                # Conversion matrix: product (p) × land use (j).
    pr2cm_cp: np.ndarray                                                # Conversion matrix: commodity (c) × product (p).
    limits: dict                                                        # Raw constraint targets for the target year (see get_limits).
    desc2aglu: dict                                                     # {agricultural land-use description: code}.
    real_area: np.ndarray                                               # Area of each cell (ha), per cell (r).
    ag_mask_proportion_r: np.ndarray                                    # Base-year (2010) agricultural proportion of each cell (r).

    # ── Transition TARGETS (TO-view): which land use a cell may become, and its upper bound ──
    trans_ub_ag_mrj: np.ndarray                                         # Ag upper bound [m, r, j]: reachable share from ag2ag + nonag2ag sources.
    trans_feasible_ag: dict                                             # {(m, j): cell indices} that get an ag decision var.
    trans_ub_nonag_rk: np.ndarray                                       # Non-ag upper bound [r, k]: reachability, reversibility lock-in, RP/destock caps.
    trans_feasible_nonag: dict                                          # {k: cell indices} with trans_ub_nonag_rk > 0 (these get a non-ag decision var).

    # ── Transition SOURCES (FROM-view): the base-year holders of land; local_r in the flow tables indexes into these ──
    trans_source_ag: dict                                               # {(from_m, from_j): cell indices} of the ag sources.
    trans_source_nonag: dict                                            # {from_k: cell indices} of the non-ag sources.
    trans_ghg_ag2ag: dict                                               # {(from_m, from_j): ndarray[to_m, local_r, to_j]} transition emissions, raw tCO2e, float32.

    # ── DECISION-VARIABLE TABLES: one long table per MVar block, in the solver's block order (= Var.index) ──
    table_ag: dict                                                      # Ag decision vars X_ag, one row per var (m, j, r, lb, ub) + col[m, j, r] -> row; see get_table_ag.
    table_nonag: dict                                                   # Non-ag decision vars X_non_ag, one row per var (k, r, lb, ub) + col[k, r] -> row; see get_table_nonag.
    table_am: dict                                                      # Ag-management decision vars X_ag_man, one row per var (am, j_idx, j, m, r, lb, ub) + col[am][m, j_idx, r] -> row; see get_table_am.
    table_flow: dict                                                    # Transition-flow delta vars F_a2a / F_a2n / F_n2a: edge tables {'a2a', 'a2n', 'n2a'}, one row per var; see get_table_flow.
    var_layout: dict                                                    # Column offsets of the blocks above (ag, nonag, am, a2a, a2n, n2a) and n_dec, the total.
    table_ag_acct: dict                                                 # Accounting view X_acct = M · X_ag as a term table (term_row, term_col, term_w) over the ag block; see get_table_ag_acct.

    # ── Per-variable TERM dicts over global Var.index (built once; every constraint family and the objective read them) ──
    term_ag_acct: dict                                                  # {(m, j): {cells, var, w}} accounting-stream terms; see row_builder.ag_acct_terms_by_mj.
    term_am: dict                                                       # {(am_idx, j_idx, m): {cells, var, w=1}}; see row_builder.am_terms_by_key.
    term_nonag: dict                                                    # {k: {cells, var, w=1}}; see row_builder.nonag_terms_by_k.

    # ── Objective ──
    obj_block: sparse.csr_matrix                                        # Economy coefficients, raw AUD, (5 × n_dec) float32, one row per OBJ_BLOCKS component; see get_obj_block.

    @property
    def ncms(self):
        return len(self.commodity_names)
    
    @property
    def ncells(self):
        # Number of cells
        return self.ag_g_mrj.shape[1]
    
    @property
    def nlms(self):
        # Number of water managements
        return self.ag_g_mrj.shape[0]

    @property
    def n_ag_lus(self):
        # Number of Agricultural land-uses
        return self.ag_g_mrj.shape[2]

    @property
    def n_non_ag_lus(self):
        # Number of Non-Agricultural Land-uses
        return self.non_ag_g_rk.shape[1]

    @property
    def nprs(self):
        # Number of products
        return self.ag_q_mrp.shape[2]

    @cached_property
    def am2j(self):
        # Map of agricultural management options to land use codes
        return get_am2j(self.desc2aglu)


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
    # source over each source's dvar>θ cells (the same cells `trans_source_ag` uses, so the solver
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


def get_ag_x_mrj(data: Data, base_year):
    print('Getting agricultural exclude matrices...', flush = True)
    return ag_transition.get_to_ag_exclude_matrices(data, base_year)


def get_trans_feasible_ag(ag_x_mrj: np.ndarray, trans_lb_ag_mrj: np.ndarray) -> dict:
    print('Getting feasible agricultural cells...', flush = True)
    n_lms, _ncells, n_lus = ag_x_mrj.shape
    eligible = (ag_x_mrj > 0) | (trans_lb_ag_mrj > 0)
    return {
        (m, j): np.where(eligible[m, :, j])[0]
        for j in range(n_lus)
        for m in range(n_lms)
    }


def get_table_ag(trans_feasible_ag: dict, trans_lb_ag_mrj: np.ndarray, trans_ub_ag_mrj: np.ndarray) -> dict:
    """The ag decision variables as ONE LONG TABLE — one row per variable (select primitive).

    Row order is j outer, dry cells then irr cells; the solver's single ``addMVar`` over this
    table makes the row index the variable's position in the ag block. ``col[m, j, r]`` is the
    inverse of the select: the table row holding that variable, or -1 where no variable exists.
    """
    print('Building the ag decision-variable table...', flush = True)
    n_lms, ncells, n_lus = trans_lb_ag_mrj.shape
    m_parts, j_parts, r_parts = [], [], []
    for j in range(n_lus):
        for m in range(n_lms):
            cells = np.asarray(trans_feasible_ag[m, j], dtype=np.int32)
            m_parts.append(np.full(cells.size, m, dtype=np.int32))
            j_parts.append(np.full(cells.size, j, dtype=np.int32))
            r_parts.append(cells)
    m_col = np.concatenate(m_parts); j_col = np.concatenate(j_parts); r_col = np.concatenate(r_parts)
    col = np.full((n_lms, n_lus, ncells), -1, dtype=np.int32)
    col[m_col, j_col, r_col] = np.arange(r_col.size, dtype=np.int32)
    return dict(
        m=m_col, j=j_col, r=r_col,
        lb=trans_lb_ag_mrj[m_col, r_col, j_col].astype(np.float64),   # exact widen, same doubles addVar received
        ub=trans_ub_ag_mrj[m_col, r_col, j_col].astype(np.float64),
        col=col,
    )


def get_table_ag_acct(table_ag: dict, ag_fold_map: dict) -> dict:
    """The accounting stream X_acct as an explicit TERM TABLE over the ag block (the fold operator M).

    Starting from X_acct = X_ag (identity), each folded sliver k (cell r, sliver (from_m, from_j),
    receiver dominant (to_m, to_j), share c_k = slivers[k] / dominant_frac[k]) contributes

        X_acct[sliver]   += c_k · X_ag[dominant]     (sliver gains the live share)
        X_acct[dominant] -= c_k · X_ag[dominant]     (dominant loses it)

    and is skipped when the dominant has no ag var. As terms (acct_row, ag_col, w): one identity
    term per ag var, then per k in fold-map order a (+c_k) term on the sliver's row and a (-c_k)
    term on the dominant's row. Terms are NOT merged (a dominant with two slivers keeps
    [1, -c_1, -c_2] as three terms) so the sub-floor drop test downstream (|coefficient| <
    SOLVER_COEFF_MIN, tested before the fold weight — see ``row_builder.keep_terms``) runs on a
    single term's coefficient. `col[m, j, r]` is the acct row of an entry, -1 where it has no terms (no var and
    not a sliver of a var-holding dominant). Acct rows = the ag table rows in order, then
    sliver-only rows in first-appearance order. All size-0 fold arrays ⇒ pure identity
    (X_acct == X_ag).
    """
    print('Building the ag accounting-stream term table...', flush = True)
    col_ag = table_ag['col']
    n_lms, n_lus, ncells = col_ag.shape
    n_ag = table_ag['r'].size

    cells  = np.asarray(ag_fold_map['cells'], dtype=np.int64)
    from_m = np.asarray(ag_fold_map['from_m'], dtype=np.int64)
    from_j = np.asarray(ag_fold_map['from_j'], dtype=np.int64)
    to_m   = np.asarray(ag_fold_map['to_m'], dtype=np.int64)
    to_j   = np.asarray(ag_fold_map['to_j'], dtype=np.int64)
    slivers       = np.asarray(ag_fold_map['vals']).astype(np.float64)
    dominant_frac = np.asarray(ag_fold_map['folded_dom']).astype(np.float64)

    # identity block: acct row i == ag col i
    acct_m = [table_ag['m']]; acct_j = [table_ag['j']]; acct_r = [table_ag['r']]
    col = col_ag.copy()                                            # acct row of every ag var
    t_row = [np.arange(n_ag, dtype=np.int32)]
    t_col = [np.arange(n_ag, dtype=np.int32)]
    t_w   = [np.ones(n_ag, dtype=np.float64)]

    if cells.size:
        dom_col = col_ag[to_m, to_j, cells]                        # -1 ⇒ dominant has no var ⇒ skip k
        keep = dom_col >= 0
        cells, from_m, from_j, dom_col = cells[keep], from_m[keep], from_j[keep], dom_col[keep]
        c = slivers[keep] / dominant_frac[keep]                    # sliver's share of the dominant's live area (float64)

        # sliver acct rows: the sliver's own ag row, or a NEW acct row (first-appearance order)
        s_row = col[from_m, from_j, cells].astype(np.int64)
        new = s_row < 0
        if new.any():
            key = (from_m[new] * n_lus + from_j[new]) * ncells + cells[new]
            uniq, first = np.unique(key, return_index=True)
            order = np.argsort(first, kind='stable')               # first-appearance order
            uniq = uniq[order]
            new_rows = n_ag + np.arange(uniq.size, dtype=np.int64)
            um, rem = np.divmod(uniq, n_lus * ncells); uj, ur = np.divmod(rem, ncells)
            col[um, uj, ur] = new_rows
            acct_m.append(um.astype(np.int32)); acct_j.append(uj.astype(np.int32)); acct_r.append(ur.astype(np.int32))
            s_row[new] = col[from_m[new], from_j[new], cells[new]]

        t_row += [s_row.astype(np.int32),  dom_col.astype(np.int32)]   # sliver gains, dominant loses
        t_col += [dom_col.astype(np.int32), dom_col.astype(np.int32)]
        t_w   += [c, -c]

    return dict(
        m=np.concatenate(acct_m), j=np.concatenate(acct_j), r=np.concatenate(acct_r),
        col=col,
        term_row=np.concatenate(t_row), term_col=np.concatenate(t_col), term_w=np.concatenate(t_w).astype(np.float32),
        n_ag=n_ag,
    )


def get_table_nonag(trans_feasible_nonag: dict, trans_lb_nonag_rk: np.ndarray,
                        trans_ub_nonag_rk: np.ndarray, dvar_base_non_ag_rk: np.ndarray) -> dict:
    """The non-ag decision variables as ONE LONG TABLE — one row per variable (select primitive).

    Row order is enabled non-ag land uses in NON_AG_LAND_USES order, cells ascending within
    each; the solver's single ``addMVar`` makes the row index the variable's position in the
    non-ag block. Bounds apply the collapse rule (lb/ub within 1 % of a positive lb collapse to
    the base), computed in the arrays' own dtype and widened exactly. ``col[k, r]`` = table row
    or -1.
    """
    print('Building the non-ag decision-variable table...', flush = True)
    lb_n, ub_n = trans_lb_nonag_rk, trans_ub_nonag_rk
    collapse = (lb_n > 0) & (np.abs(ub_n - lb_n) / np.where(lb_n > 0, lb_n, 1.0) < 0.01)
    lb_eff = np.where(collapse, dvar_base_non_ag_rk, lb_n)
    ub_eff = np.where(collapse, dvar_base_non_ag_rk, ub_n)
    ncells, n_k = lb_n.shape
    k_parts, r_parts = [], []
    for k, k_name in enumerate(settings.NON_AG_LAND_USES):
        if not settings.NON_AG_LAND_USES[k_name]:
            continue
        cells = np.asarray(trans_feasible_nonag[k], dtype=np.int32)
        k_parts.append(np.full(cells.size, k, dtype=np.int32)); r_parts.append(cells)
    k_col = np.concatenate(k_parts) if k_parts else np.array([], dtype=np.int32)
    r_col = np.concatenate(r_parts) if r_parts else np.array([], dtype=np.int32)
    col = np.full((n_k, ncells), -1, dtype=np.int32)
    col[k_col, r_col] = np.arange(r_col.size, dtype=np.int32)
    return dict(k=k_col, r=r_col,
                lb=lb_eff[r_col, k_col].astype(np.float64),
                ub=ub_eff[r_col, k_col].astype(np.float64),
                col=col)


def get_trans_feasible_nonag(trans_ub_nonag_rk: np.ndarray, threshold: float = 0.0) -> dict:
    print('Getting feasible non-agricultural cells...', flush = True)
    n_k = trans_ub_nonag_rk.shape[1]
    return {k: np.where(trans_ub_nonag_rk[:, k] > threshold)[0] for k in range(n_k)}


def get_trans_source_ag(data: Data, base_year: int) -> dict:
    print('Getting agricultural source cells...', flush = True)
    return ag_transition.get_base_dvar_mj_cell_map(data, base_year)


def get_trans_source_nonag(data: Data, base_year: int) -> dict:
    print('Getting non-agricultural source cells...', flush = True)
    return non_ag_transition.get_base_nonag_dvar_k_cell_map(data, base_year)


def get_feasible_ag2ag_mrj(ag_x_mrj: np.ndarray, trans_source_ag: dict, T_ag2ag_reach_jj: np.ndarray) -> dict:
    """Ag2ag delta-var feasibility, SOURCE-KEYED like flow_cost_ag2ag:

        {(from_m, from_j): bool (NLMS, ncells_src, N_AG_LUS) [to_m, local_r, to_j]}

        feasible[to_m, local_r, to_j] = (ag_x_mrj[to_m, r, to_j] > 0)     target eligible (X var exists)
                                      ∧ T_MAT[from_j → to_j]              THIS source may make the move
                                      ∧ not the diagonal                  staying is not a transition

    One leaf per source over that source's dvar>θ cells (`trans_source_ag`, the same
    get_base_dvar_mj_cell_map slices that anchor the flow vars) — the solver adds one delta var per
    True entry, nothing more. `ag_x_mrj > 0` (exclude matrix: union reach × EXCLUDE × no-go) is the
    same quantity that creates the ag X vars, so every delta lands on an existing var.
    """
    print('Getting feasible ag2ag delta-var targets...', flush = True)
    eligible = ag_x_mrj > 0
    result = {}
    for (fm, fj), cells in trans_source_ag.items():
        valid = eligible[:, cells, :] & T_ag2ag_reach_jj[fj][None, None, :]     # (NLMS, ncells_src, N_AG)
        valid[fm, :, fj] = False                                                # drop the diagonal
        result[(fm, fj)] = valid
    return result


def get_feasible_nonag2ag_mrj(ag_x_mrj: np.ndarray, trans_source_nonag: dict, T_nonag2ag_reach_kj: np.ndarray) -> dict:
    """Nonag2ag delta-var feasibility, SOURCE-KEYED like flow_cost_nonag2ag:

        {from_k: bool (NLMS, ncells_k, N_AG_LUS) [to_m, local_r, to_j]}

    Same construction as get_feasible_ag2ag_mrj but from the non-ag sources (`trans_source_nonag`,
    e.g. reversible Destocked land converting back to ag). No diagonal to drop (cross-family).
    """
    print('Getting feasible nonag2ag delta-var targets...', flush = True)
    eligible = ag_x_mrj > 0
    return {
        fk: eligible[:, cells, :] & T_nonag2ag_reach_kj[fk][None, None, :]      # (NLMS, ncells_k, N_AG)
        for fk, cells in trans_source_nonag.items()
    }


def get_feasible_ag2nonag_rk(trans_ub_nonag_rk: np.ndarray, trans_source_ag: dict, T_ag2nonag_reach_jk: np.ndarray) -> dict:
    """Ag2nonag delta-var feasibility, SOURCE-KEYED like flow_cost_ag2nonag:

        {(from_m, from_j): bool (ncells_src, N_NON_AG_LUS) [local_r, k]}

    The target side gates on `trans_ub_nonag_rk > 0` — NOT raw T_MAT reach — because the non-ag ub
    carries extra zeroing caps (RP stream-buffer, Destocked eligibility, non-ag no-go): a raw-reach
    gate would point deltas at targets with no X var and land would vanish through the missing
    node-balance row. No diagonal to drop (cross-family).
    """
    print('Getting feasible ag2nonag delta-var targets...', flush = True)
    eligible = trans_ub_nonag_rk > 0
    return {
        (fm, fj): eligible[cells, :] & T_ag2nonag_reach_jk[fj][None, :]         # (ncells_src, N_NONAG)
        for (fm, fj), cells in trans_source_ag.items()
    }


def get_table_flow(trans_source_ag: dict, trans_source_nonag: dict, feasible_ag2ag_mrj: dict,
                    feasible_ag2nonag_rk: dict, feasible_nonag2ag_mrj: dict) -> dict:
    """The transition-flow system as EDGE TABLES — one row per delta variable (select primitive).

    Wide → long: the from-keyed dicts {(fm, fj): bool block[to_m, local_r, to_j]}
    become one table per sub-block whose every key is a column. Row order is sources in the
    feasibility dicts' insertion order (== ``trans_source_ag`` order), arcs in ``np.argwhere``
    C-order within a source; one ``addMVar`` per sub-block makes the row index the delta var's
    position in its block, and the variable names are generated from the columns.
    ``local_r`` is kept so the source-keyed ``flow_cost_*`` / ``flow_ghg_*``
    dicts gather unchanged; ``r`` = ``source_cells[local_r]`` is the global cell (the node key).
    ``src_ptr[s]:src_ptr[s+1]`` is source s's row range. Land never crosses cells: the
    destinations of an arc are land uses at the SAME cell.
    """
    print('Building the transition-flow edge tables...', flush = True)
    i32 = np.int32

    def cat(parts, dt=i32):
        return np.concatenate(parts).astype(dt) if parts else np.array([], dtype=dt)

    # ── ag → ag: keys (to_m, local_r, to_j) ─────────────────────────────────────
    src_c, fm_c, fj_c, lr_c, r_c, tm_c, tj_c, ptr = [], [], [], [], [], [], [], [0]
    for src, ((fm, fj), valid) in enumerate(feasible_ag2ag_mrj.items()):
        idx = np.argwhere(valid)                                    # (n, 3) C-order: to_m, local_r, to_j
        cells = np.asarray(trans_source_ag[(fm, fj)])
        src_c.append(np.full(len(idx), src)); fm_c.append(np.full(len(idx), fm)); fj_c.append(np.full(len(idx), fj))
        tm_c.append(idx[:, 0]); lr_c.append(idx[:, 1]); tj_c.append(idx[:, 2]); r_c.append(cells[idx[:, 1]])
        ptr.append(ptr[-1] + len(idx))
    a2a = dict(src=cat(src_c), fm=cat(fm_c), fj=cat(fj_c), local_r=cat(lr_c), r=cat(r_c),
               to_m=cat(tm_c), to_j=cat(tj_c), src_ptr=np.asarray(ptr, dtype=np.int64),
               sources=list(feasible_ag2ag_mrj))
    a2a['n'] = a2a['r'].size

    # ── ag → non-ag: keys (k, local_r), argwhere over (local_r, k) ──────────────
    src_c, fm_c, fj_c, lr_c, r_c, k_c, ptr = [], [], [], [], [], [], [0]
    for src, ((fm, fj), valid) in enumerate(feasible_ag2nonag_rk.items()):
        idx = np.argwhere(valid)                                    # (n, 2) C-order: local_r, k
        cells = np.asarray(trans_source_ag[(fm, fj)])
        src_c.append(np.full(len(idx), src)); fm_c.append(np.full(len(idx), fm)); fj_c.append(np.full(len(idx), fj))
        lr_c.append(idx[:, 0]); k_c.append(idx[:, 1]); r_c.append(cells[idx[:, 0]])
        ptr.append(ptr[-1] + len(idx))
    a2n = dict(src=cat(src_c), fm=cat(fm_c), fj=cat(fj_c), local_r=cat(lr_c), r=cat(r_c),
               k=cat(k_c), src_ptr=np.asarray(ptr, dtype=np.int64),
               sources=list(feasible_ag2nonag_rk))
    a2n['n'] = a2n['r'].size

    # ── non-ag → ag: keys (to_m, local_r, to_j) ─────────────────────────────────
    src_c, fk_c, lr_c, r_c, tm_c, tj_c, ptr = [], [], [], [], [], [], [0]
    for src, (fk, valid) in enumerate(feasible_nonag2ag_mrj.items()):
        idx = np.argwhere(valid)                                    # (n, 3): to_m, local_r, to_j
        cells = np.asarray(trans_source_nonag[fk])
        src_c.append(np.full(len(idx), src)); fk_c.append(np.full(len(idx), fk))
        tm_c.append(idx[:, 0]); lr_c.append(idx[:, 1]); tj_c.append(idx[:, 2]); r_c.append(cells[idx[:, 1]])
        ptr.append(ptr[-1] + len(idx))
    n2a = dict(src=cat(src_c), fk=cat(fk_c), local_r=cat(lr_c), r=cat(r_c),
               to_m=cat(tm_c), to_j=cat(tj_c), src_ptr=np.asarray(ptr, dtype=np.int64),
               sources=list(feasible_nonag2ag_mrj))
    n2a['n'] = n2a['r'].size
    print(f'    ag2ag {a2a["n"]:,} | ag2nonag {a2n["n"]:,} | nonag2ag {n2a["n"]:,} arcs', flush = True)
    return dict(a2a=a2a, a2n=a2n, n2a=n2a)


def get_trans_ub_ag_mrj(data: Data, base_year: int) -> np.ndarray:
    print('Getting agricultural target upper bounds...', flush = True)
    ub = (
        ag_transition.get_ag2ag_ub(data, base_year)
        + non_ag_transition.get_nonag2ag_ub(data, base_year)
    ).astype(np.float32)
    # A cell can always KEEP its base LU ⇒ ub must be ≥ base (exact Σfrac can land a hair below base
    # on float noise, e.g. 0.9999<1.0, which would break cell-usage saturation Σ X = ag_mask). Also ≥0.
    
    # NOTE: if a real gap is reported here (not float noise), some base land-use is banned by
    # EXCLUDE/no-go (e.g. a pre-reconciliation x_mrj.npy, or a no-go region overlapping the base map).
    # The raise does NOT let such cells keep their base LU — with ag_x_mrj=0 and lb=0 no X var exists
    # (get_trans_feasible_ag), so the solver force-converts that land; the raise only keeps the
    # lb <= base <= ub box coherent for bookkeeping (const clipping, has_any_ag_r).

    # FOLDED base: the solver's const/base is the folded dvar, so ub must cover THAT (dominant
    # entries carry their absorbed sub-θ mass; folded-away entries need no headroom).
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

def get_trans_lb_ag_mrj(data: Data, base_year: int) -> np.ndarray:
    print('Getting agricultural target lower bounds...', flush = True)
    lb = ag_transition.get_ag2ag_lb(data, base_year)      # all zeros — sliver pin superseded by θ-folding
    base = ag_transition.get_folded_base_ag_dvar(data, base_year)
    # lb must sit in [0, base] (never above the base it locks in).
    return tools.clamp_dvar_bound(lb, 0.0, np.maximum(base, 0.0), 'Ag lb clamped to [0,base]')

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
    output = ag_transition.get_lower_bound_agricultural_management_matrices(data, base_year)
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


def get_am2j(desc2aglu: dict) -> dict:
    """{am: [land-use codes]} for the ENABLED ag-management options, in settings order."""
    return {
        am: [desc2aglu[lu] for lu in am_lus]
        for am, am_lus in settings.AG_MANAGEMENTS_TO_LAND_USES.items()
        if settings.AG_MANAGEMENTS[am]
    }


def get_table_am(am2j: dict, nlms: int, ncells: int, trans_feasible_ag: dict,
                     renewable_GBF2_mask_solar_idx: np.ndarray, renewable_GBF2_mask_wind_idx: np.ndarray,
                     savanna_eligible_r: np.ndarray, ag_man_lb_mrj: dict) -> dict:
    """LONG table of the ag-management decision vars — one row per variable.

    Row order is am in ``am2j`` order (enabled only), j_idx, dry cells then irr cells.
    Cell selection: renewables drop the GBF2-exclusion cells from BOTH lm; savanna burning
    intersects ``savanna_eligible_r`` on DRY only (the post-solve read in ``solve()`` masks
    irr as well — keep the two in step). lb = 0 if the am is reversible else ``ag_man_lb_mrj[am][m, r, j]``
    (widened exactly); ub = 1. ``col[am][m, j_idx, r]`` = table row or -1; ``am_list`` =
    the am order; ``am`` column = index into it.
    """
    print('Building the ag-management decision-variable table...', flush = True)
    am_list = list(am2j)
    am_c, ji_c, j_c, m_c, r_c, lb_c = [], [], [], [], [], []
    col = {}
    for am_idx, (am, am_j_list) in enumerate(am2j.items()):
        col[am] = np.full((nlms, len(am_j_list), ncells), -1, dtype=np.int32)
        excl = None
        if am in settings.RENEWABLES_OPTIONS:
            excl = (renewable_GBF2_mask_solar_idx if am == "Utility Solar PV"
                    else renewable_GBF2_mask_wind_idx)
        for j_idx, j in enumerate(am_j_list):
            dry = trans_feasible_ag[0, j]
            irr = trans_feasible_ag[1, j]
            if excl is not None:
                if excl.size:
                    dry = np.setdiff1d(dry, excl); irr = np.setdiff1d(irr, excl)
            elif tools.am_name_snake_case(am) == "savanna_burning":
                dry = np.intersect1d(dry, savanna_eligible_r)               # dry only (see docstring)
            for m, cells in ((0, dry), (1, irr)):
                cells = np.asarray(cells, dtype=np.int32)
                am_c.append(np.full(cells.size, am_idx, dtype=np.int32))
                ji_c.append(np.full(cells.size, j_idx, dtype=np.int32))
                j_c.append(np.full(cells.size, j, dtype=np.int32))
                m_c.append(np.full(cells.size, m, dtype=np.int32))
                r_c.append(cells)
                lb_c.append(np.zeros(cells.size, dtype=np.float64) if settings.AG_MANAGEMENTS_REVERSIBLE[am]
                            else np.asarray(ag_man_lb_mrj[am])[m, cells, j].astype(np.float64))
    cat = lambda parts, dt: np.concatenate(parts) if parts else np.array([], dtype=dt)
    tab = dict(am=cat(am_c, np.int32), j_idx=cat(ji_c, np.int32), j=cat(j_c, np.int32),
               m=cat(m_c, np.int32), r=cat(r_c, np.int32), lb=cat(lb_c, np.float64),
               col=col, am_list=am_list)
    tab['ub'] = np.ones(tab['r'].size, dtype=np.float64)
    for am_idx, am in enumerate(am_list):
        sel = np.flatnonzero(tab['am'] == am_idx)
        col[am][tab['m'][sel], tab['j_idx'][sel], tab['r'][sel]] = sel.astype(np.int32)
    return tab


def get_var_layout(table_ag: dict, table_nonag: dict, table_am: dict, table_flow: dict) -> dict:
    """Column layout of the decision-variable vector, in the order the solver creates its
    MVar blocks (= Var.index): ag, nonag, am, a2a, a2n, n2a. Offsets chain by table size;
    ``n_dec`` is the total. The solver's range slacks come AFTER these and carry no objective."""
    ag = 0
    nonag = ag + table_ag['r'].size
    am = nonag + table_nonag['r'].size
    a2a = am + table_am['r'].size
    a2n = a2a + table_flow['a2a']['n']
    n2a = a2n + table_flow['a2n']['n']
    n_dec = n2a + table_flow['n2a']['n']
    return dict(ag=ag, nonag=nonag, am=am, a2a=a2a, a2n=a2n, n2a=n2a, n_dec=n_dec)


def get_obj_block(ag_obj_mrj: np.ndarray, non_ag_obj_rk: np.ndarray, ag_man_objs: dict,
                  term_ag_acct: dict, term_am: dict, am_list: list, term_nonag: dict,
                  table_flow: dict, flow_cost_ag2ag: dict, flow_cost_ag2nonag: dict,
                  flow_cost_nonag2ag: dict, var_layout: dict) -> sparse.csr_matrix:
    """The economy coefficients (raw AUD) as a (5 x n_dec) sparse block, one row per component
    (``OBJ_BLOCKS``: ag, am, nonag, trans_ag, trans_nonag), over ``var_layout``.

    Same coefficient contract as ``row_builder.compose_row`` (described on
    ``row_builder.keep_terms``): a per-cell coefficient is dropped when |q| < SOLVER_COEFF_MIN,
    tested BEFORE the fold weight; the survivors are multiplied by the fold weight in float32;
    a variable repeated by fold terms is merged into one coefficient with ``sum_duplicates``;
    the final floor of the merged coefficient happens in the solver, after scaling
    (``_setup_objective``). Transition costs are the negated flow costs on the
    positive-increment delta vars. The solver sums the rows, scales and floors them into the
    objective; ``solve()`` reads the per-block economy breakdown as ``obj_block @ x``.
    """
    lay = var_layout
    AG, AM, NONAG, TRANS_AG, TRANS_NONAG = (OBJ_BLOCKS.index(b) for b in OBJ_BLOCKS)
    terms = []                                       # (block row, var, value) — every term goes through keep_terms

    # ag ACCOUNTING stream: raw coeff × X_acct terms over the accounting support
    for (m, j), t in term_ag_acct.items():
        terms.append((AG, *keep_terms(t['var'], ag_obj_mrj[m, t['cells'], j], t['w'])))
    for (am_idx, j_idx, m), t in term_am.items():
        terms.append((AM, *keep_terms(t['var'], ag_man_objs[am_list[am_idx]][m, t['cells'], j_idx])))
    for k, t in term_nonag.items():
        terms.append((NONAG, *keep_terms(t['var'], non_ag_obj_rk[t['cells'], k])))

    # transition costs: per source slice of each edge table, gathered from the source-keyed cost dicts
    ft = table_flow
    for tab, offset, cost in ((ft['a2a'], lay['a2a'], flow_cost_ag2ag),
                              (ft['n2a'], lay['n2a'], flow_cost_nonag2ag)):
        for si, src in enumerate(tab['sources']):
            a, b = int(tab['src_ptr'][si]), int(tab['src_ptr'][si + 1])
            c = cost[src][tab['to_m'][a:b], tab['local_r'][a:b], tab['to_j'][a:b]]
            terms.append((TRANS_AG, *keep_terms(offset + np.arange(a, b), -c)))
    a2n = ft['a2n']
    for si, src in enumerate(a2n['sources']):
        a, b = int(a2n['src_ptr'][si]), int(a2n['src_ptr'][si + 1])
        cdict = flow_cost_ag2nonag[src]                                  # {k: array(ncells_src)}
        k, lr = a2n['k'][a:b], a2n['local_r'][a:b]
        c = np.empty(b - a, dtype=np.float32)
        for kk in np.unique(k):
            c[k == kk] = cdict[int(kk)][lr[k == kk]]
        terms.append((TRANS_NONAG, *keep_terms(lay['a2n'] + np.arange(a, b), -c)))

    rows = np.concatenate([np.full(v.size, r, dtype=np.int32) for r, _, v in terms])
    cols = np.concatenate([c for _, c, _ in terms])
    vals = np.concatenate([v for _, _, v in terms])
    block = sparse.csr_matrix((vals, (rows, cols)), shape=(len(OBJ_BLOCKS), lay['n_dec']))
    block.sum_duplicates()                                                # merge repeated variables (fold terms)
    return block


def get_input_data(data: Data, base_year: int, target_year: int) -> SolverInputData:
    """
    Using the given Data object, prepare a SolverInputData object for the solver.
    """

    target_index = target_year - data.YR_CAL_BASE
    ag_c_mrj     = get_ag_c_mrj(data, target_index)
    ag_r_mrj     = get_ag_r_mrj(data, target_index)

    # ── Transition costs — SOURCE-KEYED flow-cost dicts ──────────────
    # Sliced by base-year source ("(from_m, from_j)" for ag, "k" for non-ag) over each source's dvar>θ
    # cells; the solver creates a matching delta var per (source, cell, target) and charges
    # Σ flow_cost·D in the objective. get_economic_mrj bakes no land-use transition cost.

    # ag→ag: dict[(from_m, from_j)] → ndarray(NLMS, ncells_src, N_AG_LUS)
    flow_cost_ag2ag = get_ag_t_mrj(data, target_index, base_year)

    # ag→ag transition GHG EMISSIONS (raw t CO2), source-keyed — the physical parallel of
    # flow_cost_ag2ag. The GHG constraint sums Σ flow_ghg·D (source-correct transition emissions).
    trans_ghg_ag2ag                 = ag_ghg.get_ghg_transition_emissions_from_base_year(data, base_year)

    # Per-source transition reachability (T_MAT finite ⇒ allowed) — decides which delta vars exist.
    T_ag2ag_reach_jj    = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.AGRICULTURAL_LANDUSES).values)
    T_ag2nonag_reach_jk = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.NON_AGRICULTURAL_LANDUSES).values)
    T_nonag2ag_reach_kj = ~np.isnan(data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values)

    # ag→nonag: dispatcher gives dict[lu_name → dict[(fm,fj)]]; transpose to dict[(fm,fj) → dict[k]]
    # so the solver loops ag sources first.
    flow_cost_ag2nonag = {}
    for _lu_name, _per_src in non_ag_transition.get_transition_matrix_ag2nonag(
        data, base_year, target_year
    ).items():
        _k = data.NON_AGRICULTURAL_LANDUSES.index(_lu_name)
        for _src, _arr in _per_src.items():
            flow_cost_ag2nonag.setdefault(_src, {})[_k] = _arr

    # nonag→ag: dispatcher gives dict[lu_name → dict[k]]; take the diagonal (cells in non-ag LU k
    # pay only LU k's own nonag→ag cost).
    flow_cost_nonag2ag = {}
    for _lu_name, _per_k in non_ag_transition.get_transition_matrix_nonag2ag(
        data, base_year, target_year
    ).items():
        _k = data.NON_AGRICULTURAL_LANDUSES.index(_lu_name)
        if _k in _per_k:
            flow_cost_nonag2ag[_k] = _per_k[_k]

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
    ag_x_mrj                        = get_ag_x_mrj(data, base_year)          # exclude matrix: which (m, j) a cell may become
    trans_ub_ag_mrj                 = get_trans_ub_ag_mrj(data, base_year)        # TO-view ag target upper bound (ag2ag + nonag2ag)
    trans_lb_ag_mrj                 = get_trans_lb_ag_mrj(data, base_year)        # TO-view ag target lower bound (zeros for now)
    trans_source_ag                 = get_trans_source_ag(data, base_year)   # FROM-view: cells holding each ag (from_m,from_j) source
    trans_source_nonag              = get_trans_source_nonag(data, base_year)# FROM-view: cells holding each non-ag source k
    ag_q_mrp                        = get_ag_q_mrp(data, target_index)

    non_ag_g_rk                     = get_non_ag_g_rk(data, ag_g_mrj, base_year)
    non_ag_w_rk                     = (
        get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year)
        if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on'
        else get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    )
    trans_ub_nonag_rk               = get_trans_ub_nonag_rk(data, base_year)
    trans_feasible_nonag            = get_trans_feasible_nonag(trans_ub_nonag_rk) # cells that get a target non-ag var (ub > 0)
    non_ag_q_crk                    = get_non_ag_q_crk(data, ag_q_mrp, base_year)
    trans_lb_nonag_rk               = get_trans_lb_nonag_rk(data, base_year)

    ag_man_g_mrj                    = get_ag_man_g_mrj(data, target_index)
    ag_man_w_mrj                    = get_ag_man_w_mrj(data, target_index)
    ag_man_q_mrp                    = get_ag_man_q_mrj(data, target_index, ag_q_mrp)
    ag_man_limits                   = get_ag_man_limits(data, target_index)
    ag_man_lb_mrj                   = get_ag_man_lb_mrj(data, base_year)
    
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

    savanna_eligible_r              = get_savanna_eligible_r(data)
    renewable_GBF2_mask_solar_idx   = get_renewable_GBF2_mask_solar_idx(data)
    renewable_GBF2_mask_wind_idx    = get_renewable_GBF2_mask_wind_idx(data)
    renewable_MNES_mask_solar_idx   = get_renewable_MNES_mask_solar_idx(data)
    renewable_MNES_mask_wind_idx    = get_renewable_MNES_mask_wind_idx(data)

    limits = get_limits(data, target_year)

    # Derive target eligibility from ag_x_mrj so it matches the mask the solver reads (per-source θ;
    # was ag_lu2cells). Stay-floor cells (trans_lb_ag_mrj>0, sub-θ slivers locked in) are unioned in so
    # their var exists.
    trans_feasible_ag               = get_trans_feasible_ag(ag_x_mrj, trans_lb_ag_mrj)  # cells that get a target ag var
    table_ag        = get_table_ag(trans_feasible_ag, trans_lb_ag_mrj, trans_ub_ag_mrj)  # the ag block as a long table

    # Per-source delta-var feasibility — keyed/shaped like the flow_cost dicts; the solver adds one
    # delta var per True entry.
    feasible_ag2ag_mrj    = get_feasible_ag2ag_mrj(ag_x_mrj, trans_source_ag, T_ag2ag_reach_jj)
    feasible_nonag2ag_mrj = get_feasible_nonag2ag_mrj(ag_x_mrj, trans_source_nonag, T_nonag2ag_reach_kj)
    feasible_ag2nonag_rk  = get_feasible_ag2nonag_rk(trans_ub_nonag_rk, trans_source_ag, T_ag2nonag_reach_jk)
    table_flow      = get_table_flow(trans_source_ag, trans_source_nonag, feasible_ag2ag_mrj, feasible_ag2nonag_rk, feasible_nonag2ag_mrj)   # the flow system as edge tables

    # The coefficient streams leave here raw and float32 (the DTYPE POLICY in row_builder):
    # every constraint block is row-rescaled in the solver (row_builder.scale_rows, per-row
    # factor kept there) and the objective is raw AUD / 1e6.
    ag_obj_mrj, non_ag_obj_rk = ag_obj_mrj.astype(np.float32), non_ag_obj_rk.astype(np.float32)
    ag_man_objs = {am: v.astype(np.float32) for am, v in ag_man_objs.items()}
    flow_cost_ag2ag    = {s: v.astype(np.float32)     for s, v in flow_cost_ag2ag.items()}
    flow_cost_ag2nonag = {s: {k: a.astype(np.float32) for k, a in p.items()} for s, p in flow_cost_ag2nonag.items()}
    flow_cost_nonag2ag = {k: v.astype(np.float32)     for k, v in flow_cost_nonag2ag.items()}
    ag_q_mrp, non_ag_q_crk = ag_q_mrp.astype(np.float32), non_ag_q_crk.astype(np.float32)
    ag_man_q_mrp = {am: v.astype(np.float32) for am, v in ag_man_q_mrp.items()}
    ag_g_mrj, non_ag_g_rk = ag_g_mrj.astype(np.float32), non_ag_g_rk.astype(np.float32)
    ag_man_g_mrj = {am: v.astype(np.float32) for am, v in ag_man_g_mrj.items()}
    trans_ghg_ag2ag                 = {s: v.astype(np.float32) for s, v in trans_ghg_ag2ag.items()}
    ag_w_mrj, non_ag_w_rk = ag_w_mrj.astype(np.float32), non_ag_w_rk.astype(np.float32)
    ag_man_w_mrj = {am: v.astype(np.float32) for am, v in ag_man_w_mrj.items()}
    renewable_solar_r, renewable_wind_r = renewable_solar_r.astype(np.float32), renewable_wind_r.astype(np.float32)

    offland_ghg = (
        data.OFF_LAND_GHG_EMISSION_C[target_index]                       # raw tCO2e (row-rescaled in the solver)
        if settings.GHG_EMISSIONS_LIMITS != 'off'
        else 0.0
    )

    # ── The decision-variable tables ────────────────────────────────────────────────────────
    # Base dvars are the node-balance "stay" constant; clip them into the cleaned [lb, ub] box so the
    # all-delta=0 stay point is feasible by construction (fixes base's own float noise, e.g. -1e-8<lb=0).
    # Bounds were already clamped so lb ≤ base ≤ ub for real values — this only bites on noise. Reported.
    dvar_base_ag_mrj    = tools.clamp_dvar_bound(ag_transition.get_folded_base_ag_dvar(data, base_year), trans_lb_ag_mrj, trans_ub_ag_mrj, 'Ag base clipped to [lb,ub]')
    dvar_base_non_ag_rk = tools.clamp_dvar_bound(data.non_ag_dvars[base_year], trans_lb_nonag_rk, trans_ub_nonag_rk, 'NonAg base clipped to [lb,ub]')

    ag_fold_map     = ag_transition.get_ag_dvar_fold_map(data, base_year)                  # which sub-θ slivers fold into which dominant
    table_ag_acct   = get_table_ag_acct(table_ag, ag_fold_map)                          # X_acct = M · X_ag as a term table
    table_nonag     = get_table_nonag(trans_feasible_nonag, trans_lb_nonag_rk, trans_ub_nonag_rk, dvar_base_non_ag_rk)
    table_am        = get_table_am(get_am2j(data.DESC2AGLU), data.NLMS, data.NCELLS, trans_feasible_ag,
                                       renewable_GBF2_mask_solar_idx, renewable_GBF2_mask_wind_idx, savanna_eligible_r, ag_man_lb_mrj)
    var_layout      = get_var_layout(table_ag, table_nonag, table_am, table_flow)

    # Per-variable term dicts over global Var.index — built once, read by every constraint family
    # (row_builder.extract_groups / extract_structure) and by the objective block.
    term_ag_acct    = ag_acct_terms_by_mj(table_ag_acct, var_layout['ag'])
    term_am         = am_terms_by_key(table_am, var_layout['am'])
    term_nonag      = nonag_terms_by_k(table_nonag, var_layout['nonag'])
    obj_block       = get_obj_block(ag_obj_mrj, non_ag_obj_rk, ag_man_objs, term_ag_acct, term_am, table_am['am_list'],
                                  term_nonag, table_flow, flow_cost_ag2ag, flow_cost_ag2nonag, flow_cost_nonag2ag, var_layout)

    return SolverInputData(
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

        dvar_base_ag_mrj=dvar_base_ag_mrj,
        dvar_base_non_ag_rk=dvar_base_non_ag_rk,

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

        savanna_eligible_r=savanna_eligible_r,
        renewable_GBF2_mask_solar_idx=renewable_GBF2_mask_solar_idx,
        renewable_GBF2_mask_wind_idx=renewable_GBF2_mask_wind_idx,
        renewable_MNES_mask_solar_idx=renewable_MNES_mask_solar_idx,
        renewable_MNES_mask_wind_idx=renewable_MNES_mask_wind_idx,

        commodity_names=data.COMMODITIES,
        offland_ghg=offland_ghg,
        lu2pr_pj=data.LU2PR,
        pr2cm_cp=data.PR2CM,
        limits=limits,
        desc2aglu=data.DESC2AGLU,
        real_area=data.REAL_AREA,
        ag_mask_proportion_r=data.AG_MASK_PROPORTION_R,

        # transition targets (TO-view)
        trans_ub_ag_mrj=trans_ub_ag_mrj,
        trans_feasible_ag=trans_feasible_ag,
        trans_ub_nonag_rk=trans_ub_nonag_rk,
        trans_feasible_nonag=trans_feasible_nonag,
        # transition sources (FROM-view)
        trans_source_ag=trans_source_ag,
        trans_source_nonag=trans_source_nonag,
        trans_ghg_ag2ag=trans_ghg_ag2ag,
        # decision-variable tables, in the solver's block order
        table_ag=table_ag,
        table_nonag=table_nonag,
        table_am=table_am,
        table_flow=table_flow,
        var_layout=var_layout,
        table_ag_acct=table_ag_acct,
        # per-variable term dicts + objective
        term_ag_acct=term_ag_acct,
        term_am=term_am,
        term_nonag=term_nonag,
        obj_block=obj_block,
    )

