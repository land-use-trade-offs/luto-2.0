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
Data about transitions costs.
"""

import numpy as np
import luto.tools as tools
import luto.data as Data
import luto.economics.agricultural.ghg as ag_ghg

from functools import lru_cache
from luto import settings
from typing import Dict
from luto.data import Data
from luto.economics.agricultural.water import get_wreq_matrices



@lru_cache(maxsize=1)
def get_base_dvar_mj_cell_map(data: Data, base_year: int) -> dict:
    """{(from_m, from_j): cells} — the source cells of every base-year ag (lm, lu) above the ROUND_DECIMALS noise floor.
    One map (cached per (data, base_year)) feeds the solver's per-source slices, target eligibility and ghg.py."""
    noise = 10 ** (-settings.ROUND_DECIMALS)
    base_dvar_mrj = data.ag_dvars[base_year]
    return {
        (m, j): np.where(base_dvar_mrj[m, :, j] > noise)[0]
        for m in range(data.NLMS)
        for j in range(data.N_AG_LUS)
        if (base_dvar_mrj[m, :, j] > noise).any()
    }


def get_ag_eligible_mrj(data: Data, base_year: int) -> np.ndarray:
    """Bool (NLMS, NCELLS, N_AG_LUS): a cell may become (m, j) iff some base-year source there (ag or non-ag) reaches j through
    T_MAT and j is allowed by EXCLUDE / no-go. The column space creates an ag X var exactly where this is True."""
    # Lazy import to avoid the agricultural <-> non_agricultural transitions import cycle.
    from luto.economics.non_agricultural.transitions import get_base_nonag_dvar_k_cell_map

    mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
    k_cell_map  = get_base_nonag_dvar_k_cell_map(data, base_year)

    # Binary T_MAT allow/disallow matrices (finite → True, NaN → False)
    t_ag2ag_jj = ~np.isnan(
        data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )   # (N_AG_LUS_from, N_AG_LUS_to)
    t_nonag2ag_kj = ~np.isnan(
        data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )   # (N_NON_AG_LUS, N_AG_LUS_to)

    # Per-source reachability union: mark every target reachable from any present source at r.
    reach_rj = np.zeros((data.NCELLS, data.N_AG_LUS), dtype=bool)
    for (_fm, fj), cells in mj_cell_map.items():
        if cells.size:
            reach_rj[cells] |= t_ag2ag_jj[fj]
    for k, cells in k_cell_map.items():
        if cells.size:
            reach_rj[cells] |= t_nonag2ag_kj[k]

    # Spatial exclusion (EXCLUDE) and no-go zones
    allowed_mrj = data.EXCLUDE.astype(bool)                                   # astype copies: EXCLUDE itself is not touched
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            allowed_mrj[:, :, data.DESC2AGLU[no_go_desc]] &= np.asarray(no_go_x_r, dtype=bool)

    return allowed_mrj & reach_rj[np.newaxis, :, :]


def get_ag2ag_ub(data: Data, base_year: int) -> np.ndarray:
    """ag→ag target upper bound (NLMS, NCELLS, N_AG_LUS), fractional: the base-year share of every source LU that can reach to_j
    (T_MAT finite), × no-go × EXCLUDE. Ag-source component only; nonag2ag adds its own share in the combined ag ub."""
    ag_dvar = data.ag_dvars[base_year]                                                                  # (NLMS, NCELLS, N_AG_LUS)

    # Transition exclusion (T_MAT): binary allow/disallow per (from_j → to_j).
    t_ag2ag_jj = (~np.isnan(
        data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )).astype(np.float32)                                                                              # (from_j, to_j)
    
    # Reachable land share: sum the base-year fractions of every source LU that can reach to_j.
    ag_frac_rj    = ag_dvar.sum(axis=0)                                                                # (NCELLS, from_j)
    reach_frac_rj = (ag_frac_rj @ t_ag2ag_jj).astype(np.float32)                                       # (NCELLS, to_j)

    # No-go exclusion: user-defined LUs banned in specific regions.
    no_go = np.ones((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    # Spatial exclusion (data.EXCLUDE): LU never present in the SA2 region in 2010 → banned there.
    x_mrj = data.EXCLUDE.astype(np.float32)
    return (x_mrj * reach_frac_rj[np.newaxis, :, :] * no_go).astype(np.float32)


def get_transition_matrices_ag2ag(data: Data, yr_idx: int, from_m: int, from_j: int, cells=None, separate=False):
    """Per-source ag2ag cost primitive, unmasked: the amortised cost of leaving (from_m, from_j) for every (to_m, to_j) on `cells`,
    (NLMS, len(cells), N_AG_LUS) [to_m, r, to_j] = establishment + water-licence delta + GHG release × carbon price (separate=True → dict)."""
    yr_cal = data.YR_CAL_BASE + yr_idx
    if cells is None:
        cells = np.arange(data.NCELLS)
    n = len(cells)
    N_AG = data.N_AG_LUS
    area = data.REAL_AREA[cells]                                                    # (n,)

    # ── Establishment: source only enters via the T_MAT[from_j] row ──
    t_ij  = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values * data.TRANS_COST_MULTS[yr_cal]
    t_row = np.nan_to_num(t_ij[from_j]).astype(np.float32)                          # (N_AG,)
    e_rj  = tools.amortise(np.tile(t_row, (n, 1))) * area[:, None]                  # (n, N_AG)
    e_mrj = np.stack([e_rj, e_rj], axis=0).astype(np.float32)                       # (NLMS, n, N_AG)

    # ── Water licence delta (source-parameterised), amortised like Establishment above ──
    w_mrj       = get_wreq_matrices(data, yr_idx)                                   # <ML/cell> (lru_cached)
    w_raw_mrj   = tools.get_ag_to_ag_water_delta_matrix(data, from_m, from_j, cells, w_mrj, yr_idx)
    w_delta_mrj = tools.amortise(w_raw_mrj).astype(np.float32)

    # ── GHG release ($ = carbon price × raw emissions, amortised): source-parameterised ──
    price   = data.get_carbon_price_by_yr_idx(yr_idx)
    ghg_raw = ag_ghg.get_ghg_transition_emissions(data, from_m, from_j, cells, separate=True)   # raw t/cell
    ghg     = {k: tools.amortise(v * price).astype(np.float32) for k, v in ghg_raw.items()}

    if separate:
        return {'Establishment cost': e_mrj, 'Water license cost': w_delta_mrj, **ghg}
    return (e_mrj + w_delta_mrj + sum(ghg.values())).astype(np.float32)


def get_asparagopsis_effect_t_mrj(data: Data):
    """Zero transition-cost effect for Asparagopsis taxiformis: its establishment cost sits in the cost matrix."""
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_precision_agriculture_effect_t_mrj(data: Data):
    """Zero transition-cost effect for Precision Agriculture: its establishment cost sits in the cost matrix."""
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_ecological_grazing_effect_t_mrj(data: Data):
    """Zero transition-cost effect for Ecological Grazing: its establishment cost sits in the cost matrix."""
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_savanna_burning_effect_t_mrj(data: Data):
    """Zero transition-cost effect for Savanna Burning: its establishment cost sits in the cost matrix."""
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_agtech_ei_effect_t_mrj(data: Data):
    """Zero transition-cost effect for AgTech EI: its establishment cost sits in the cost matrix."""
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_biochar_effect_t_mrj(data: Data):
    """Zero transition-cost effect for Biochar: its establishment cost sits in the cost matrix."""
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_beef_hir_effect_t_mrj(data: Data):
    """Zero transition-cost effect for HIR - Beef."""
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_sheep_hir_effect_t_mrj(data: Data):
    """Zero transition-cost effect for HIR - Sheep."""
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_utility_solar_pv_effect_t_mrj(data: Data, yr_idx):
    """Zeros: the Utility Solar PV CAPEX is an amortised annual cost in cost.py (get_utility_solar_pv_effect_c_mrj)."""
    solar_lus = settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']
    return np.zeros((data.NLMS, data.NCELLS, len(solar_lus)), dtype=np.float32)


def get_onshore_wind_effect_t_mrj(data: Data, yr_idx):
    """Zeros: the Onshore Wind CAPEX is an amortised annual cost in cost.py (get_onshore_wind_effect_c_mrj)."""
    wind_lus = settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']
    return np.zeros((data.NLMS, data.NCELLS, len(wind_lus)), dtype=np.float32)


def get_agricultural_management_transition_matrices(data: Data, yr_idx) -> Dict[str, np.ndarray]:
    """{am: (NLMS, NCELLS, n_lus) transition-cost effect} for every ag-management option."""
    
    asparagopsis_data = get_asparagopsis_effect_t_mrj(data)                     
    precision_agriculture_data = get_precision_agriculture_effect_t_mrj(data)   
    eco_grazing_data = get_ecological_grazing_effect_t_mrj(data)                
    sav_burning_data = get_savanna_burning_effect_t_mrj(data)                   
    agtech_ei_data = get_agtech_ei_effect_t_mrj(data)                           
    biochar_data = get_biochar_effect_t_mrj(data)                               
    beef_hir_data = get_beef_hir_effect_t_mrj(data)                             
    sheep_hir_data = get_sheep_hir_effect_t_mrj(data)
    utility_solar_pv_data = get_utility_solar_pv_effect_t_mrj(data, yr_idx)
    onshore_wind_data = get_onshore_wind_effect_t_mrj(data, yr_idx)
                  
    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
        'Biochar': biochar_data,
        'HIR - Beef': beef_hir_data,
        'HIR - Sheep': sheep_hir_data,
        'Utility Solar PV': utility_solar_pv_data,
        'Onshore Wind': onshore_wind_data
    }


def get_asparagopsis_adoption_limits(data: Data, yr_idx):
    """{lu code: adoption limit} of Asparagopsis taxiformis for each of its land uses (0 when the option is off)."""
    if not settings.AG_MANAGEMENTS['Asparagopsis taxiformis']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']}
    
    asparagopsis_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']:
        j = data.DESC2AGLU[lu]
        asparagopsis_limits[j] = min(data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return asparagopsis_limits


def get_precision_agriculture_adoption_limit(data: Data, yr_idx):
    """{lu code: adoption limit} of Precision Agriculture for each of its land uses (0 when the option is off)."""
    if not settings.AG_MANAGEMENTS['Precision Agriculture']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']}
    
    prec_agr_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']:
        j = data.DESC2AGLU[lu]
        prec_agr_limits[j] = min(data.PRECISION_AGRICULTURE_DATA[settings.LU2TYPE[lu]].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return prec_agr_limits


def get_ecological_grazing_adoption_limit(data: Data, yr_idx):
    """{lu code: adoption limit} of Ecological Grazing for each of its land uses (0 when the option is off)."""
    if not settings.AG_MANAGEMENTS['Ecological Grazing']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']}
    
    eco_grazing_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']:
        j = data.DESC2AGLU[lu]
        eco_grazing_limits[j] = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, 'Feasible Adoption (%)']

    return eco_grazing_limits


def get_savanna_burning_adoption_limit(data: Data):
    """{lu code: adoption limit} of Savanna Burning for each of its land uses (1, or 0 when the option is off)."""
    if not settings.AG_MANAGEMENTS['Savanna Burning']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']}
    
    sav_burning_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']:
        j = data.DESC2AGLU[lu]
        sav_burning_limits[j] = 1

    return sav_burning_limits


def get_agtech_ei_adoption_limit(data: Data, yr_idx):
    """{lu code: adoption limit} of AgTech EI for each of its land uses (0 when the option is off)."""
    if not settings.AG_MANAGEMENTS['AgTech EI']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']}
    
    agtech_ei_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']:
        j = data.DESC2AGLU[lu]
        agtech_ei_limits[j] = min(data.AGTECH_EI_DATA[settings.LU2TYPE[lu]].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return agtech_ei_limits


def get_biochar_adoption_limit(data: Data, yr_idx):
    """{lu code: adoption limit} of Biochar for each of its land uses (0 when the option is off)."""
    if not settings.AG_MANAGEMENTS['Biochar']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']}
    
    biochar_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']:
        j = data.DESC2AGLU[lu]
        biochar_limits[j] = min(data.BIOCHAR_DATA[settings.LU2TYPE[lu]].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return biochar_limits


def get_beef_hir_adoption_limit(data: Data):
    """{lu code: adoption limit} of HIR - Beef for each of its land uses (1, or 0 when the option is off)."""
    if not settings.AG_MANAGEMENTS['HIR - Beef']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']}
    hir_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']:
        j = data.DESC2AGLU[lu]
        hir_limits[j] = 1

    return hir_limits


def get_sheep_hir_adoption_limit(data: Data):
    """{lu code: adoption limit} of HIR - Sheep for each of its land uses (1, or 0 when the option is off)."""
    if not settings.AG_MANAGEMENTS['HIR - Sheep']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']}
    hir_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']:
        j = data.DESC2AGLU[lu]
        hir_limits[j] = 1

    return hir_limits

def get_utility_solar_pv_adoption_limit(data: Data):
    """{lu code: adoption limit} of Utility Solar PV for each of its land uses (RENEWABLES_ADOPTION_LIMITS, or 0 when off)."""
    if not settings.AG_MANAGEMENTS['Utility Solar PV']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']}
    solar_pv_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']:
        j = data.DESC2AGLU[lu]
        solar_pv_limits[j] = settings.RENEWABLES_ADOPTION_LIMITS['Utility Solar PV']

    return solar_pv_limits

def get_onshore_wind_adoption_limit(data: Data):
    """{lu code: adoption limit} of Onshore Wind for each of its land uses (RENEWABLES_ADOPTION_LIMITS, or 0 when off)."""
    if not settings.AG_MANAGEMENTS['Onshore Wind']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']}
    
    wind_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']:
        j = data.DESC2AGLU[lu]
        wind_limits[j] = settings.RENEWABLES_ADOPTION_LIMITS['Onshore Wind']

    return wind_limits

def get_agricultural_management_adoption_limits(data: Data, yr_idx) -> Dict[str, dict]:
    """{am: {lu code: limit}} — the maximum fraction of each land use that may adopt each ag-management option."""
    ag_management_data = {}

    ag_management_data['Asparagopsis taxiformis'] = get_asparagopsis_adoption_limits(data, yr_idx)
    ag_management_data['Precision Agriculture'] = get_precision_agriculture_adoption_limit(data, yr_idx)
    ag_management_data['Ecological Grazing'] = get_ecological_grazing_adoption_limit(data, yr_idx)
    ag_management_data['Savanna Burning'] = get_savanna_burning_adoption_limit(data)
    ag_management_data['AgTech EI'] = get_agtech_ei_adoption_limit(data, yr_idx)
    ag_management_data['Biochar'] = get_biochar_adoption_limit(data, yr_idx)
    ag_management_data['HIR - Beef'] = get_beef_hir_adoption_limit(data)
    ag_management_data['HIR - Sheep'] = get_sheep_hir_adoption_limit(data)
    ag_management_data['Utility Solar PV'] = get_utility_solar_pv_adoption_limit(data)
    ag_management_data['Onshore Wind'] = get_onshore_wind_adoption_limit(data)
   
    return ag_management_data


def get_lower_bound_agricultural_management_matrices(data: Data, base_year) -> dict[str, dict]:
    """{am: (NLMS, NCELLS, N_AG_LUS) lower bound} for the next solve: the base-year adoption, clamped to its host ag dvar (the
    am ≤ ag link is a row, so a reported am can exceed ag by FeasibilityTol) and floor-truncated to ROUND_DECIMALS."""

    if base_year == data.YR_CAL_BASE or base_year not in data.ag_man_dvars:
        return {
            am: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
            for am in settings.AG_MANAGEMENTS_TO_LAND_USES
            if settings.AG_MANAGEMENTS[am]
        }

    ag_dvar = data.ag_dvars[base_year]                     # (NLMS, NCELLS, N_AG_LUS): the host ag land use of every am entry

    result = {}
    for am in settings.AG_MANAGEMENTS_TO_LAND_USES:
        if not settings.AG_MANAGEMENTS[am]:
            continue
        am_dvar = data.ag_man_dvars[base_year][am].astype(np.float32)
        am_dvar_true = tools.clamp_dvar_bound(am_dvar, 0.0, ag_dvar, f'Ag man lb clamped [{am}]')   # am cannot exceed its host ag
        am_lb = np.divide(
            np.floor(am_dvar_true * 10 ** settings.ROUND_DECIMALS),
            10 ** settings.ROUND_DECIMALS,
        )
        result[am] = am_lb.astype(np.float32)   # the int divisor promotes to float64; the bounds are float32 like every other cube

    return result


def get_regional_adoption_limits(data: Data, yr_cal: int):
    """Per-region adoption caps: (ag [reg_id, lu_code, lu_name, reg_ind, area_ha], non-ag [same], non-ag SUM [reg_id, reg_ind, area_ha]) —
    the per-LU lists in 'on' mode, the SUM list in 'NON_AG_CAP' mode, all three empty when 'off'."""
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return [], [], []

    ag_reg_adoption_constrs = []
    non_ag_reg_adoption_constrs = []

    # Per-LU (ag + non-ag) caps from xlsx — only populated in 'on' mode
    for reg_id, lu_name, area_limit_ha in data.get_regional_adoption_limit_ha_by_year(yr_cal):
        reg_ind = np.where(data.REGIONAL_ADOPTION_ZONES == reg_id)[0]

        if lu_name in data.DESC2AGLU:
            lu_code = data.DESC2AGLU[lu_name]
            ag_reg_adoption_constrs.append([reg_id, lu_code, lu_name, reg_ind, area_limit_ha])
        elif lu_name in data.DESC2NONAGLU:
            lu_code = data.DESC2NONAGLU[lu_name] - settings.NON_AGRICULTURAL_LU_BASE_CODE
            non_ag_reg_adoption_constrs.append([reg_id, lu_code, lu_name, reg_ind, area_limit_ha])
        else:
            raise ValueError(f"Regional adoption constraint exists for unrecognised land use: {lu_name}")

    # SUM-of-non-ag per-region cap — only populated in 'NON_AG_CAP' mode
    non_ag_reg_adoption_sum_constrs = [
        [reg_id, reg_ind, area_limit_ha]
        for reg_id, reg_ind, area_limit_ha
        in data.get_regional_adoption_non_ag_sum_limit_ha_by_year(yr_cal)
    ]

    return ag_reg_adoption_constrs, non_ag_reg_adoption_constrs, non_ag_reg_adoption_sum_constrs
