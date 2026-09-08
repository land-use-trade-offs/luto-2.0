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
from functools import lru_cache

import luto.tools as tools
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.ghg as ag_ghg

from luto import settings
from luto.data import Data
from luto.economics.agricultural.transitions import (
    get_base_dvar_mj_cell_map,
    get_transition_matrices_ag2ag,
)



# TODO: Ag to Non-Ag GHG transition costs are omitted; 
#   Need to think about whether to include them, and if so, 
#   how to calculate them (e.g., using ag_ghg.get_ghg_costs_from_ag_to_nonag()).
def get_env_plant_transitions_from_ag(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """{(from_m, from_j): $/cell/yr over that source's cells} to Environmental Plantings: amortised establishment + T_MAT
    transition + irrigation removal (from_m = irr); separate=True → {component: array} per source."""

    # Hoist invariants outside the per-combo loop
    t_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Environmental Plantings').values  # (N_AG_LUS,), NaN = prohibited
    est_mults  = data.EST_COST_MULTS[target_year]   # scalar
    irr_scalar = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]   # scalar

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs  = tools.amortise(data.EP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]).astype(np.float32)
        t_cost     = tools.amortise(np.nan_to_num(t_j[from_j]) * data.REAL_AREA[cell_idx]).astype(np.float32)
        w_rm_irrig = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1
            else np.zeros(len(cell_idx), dtype=np.float32)
        )

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        t_cost,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
            }
        else:
            result[(from_m, from_j)] = est_costs + t_cost + w_rm_irrig

    return result


def get_rip_plant_transitions_from_ag(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """{(from_m, from_j): $/cell/yr} to Riparian Plantings: establishment + T_MAT transition + irrigation removal + fencing."""
    t_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Riparian Plantings').values
    est_mults  = data.EST_COST_MULTS[target_year]
    irr_scalar = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs    = tools.amortise(data.RP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]).astype(np.float32)
        t_cost       = tools.amortise(np.nan_to_num(t_j[from_j]) * data.REAL_AREA[cell_idx]).astype(np.float32)
        w_rm_irrig   = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost = (data.RP_FENCING_LENGTH[cell_idx] * fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        t_cost,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + t_cost + w_rm_irrig + fencing_cost
    return result


def get_sheep_agroforestry_transitions_from_ag(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """{(from_m, from_j): $/cell/yr} to Sheep Agroforestry: AF_PROPORTION of (establishment + T_MAT to AF) plus the rest as
    T_MAT to Sheep - modified land, + irrigation removal + fencing."""
    t_af_j       = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Agroforestry').values
    t_sheep_j    = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values
    est_mults    = data.EST_COST_MULTS[target_year]
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.AF_FENCING_LENGTH_HA * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs           = (tools.amortise(data.AF_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]) * settings.AF_PROPORTION).astype(np.float32)
        ag_to_af_t          = (tools.amortise(np.nan_to_num(t_af_j[from_j]) * data.REAL_AREA[cell_idx]) * settings.AF_PROPORTION).astype(np.float32)
        ag_to_sheep_t       = (tools.amortise(np.nan_to_num(t_sheep_j[from_j]) * data.REAL_AREA[cell_idx]) * (1 - settings.AF_PROPORTION)).astype(np.float32)
        w_rm_irrig          = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost        = (fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        ag_to_af_t,
                'Transition cost (Ag2AF-Sheep)':      ag_to_sheep_t,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + ag_to_af_t + ag_to_sheep_t + w_rm_irrig + fencing_cost
    return result


def get_beef_agroforestry_transitions_from_ag(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """{(from_m, from_j): $/cell/yr} to Beef Agroforestry: AF_PROPORTION of (establishment + T_MAT to AF) plus the rest as
    T_MAT to Beef - modified land, + irrigation removal + fencing."""
    t_af_j       = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Agroforestry').values
    t_beef_j     = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values
    est_mults    = data.EST_COST_MULTS[target_year]
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.AF_FENCING_LENGTH_HA * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs    = (tools.amortise(data.AF_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]) * settings.AF_PROPORTION).astype(np.float32)
        ag_to_af_t   = (tools.amortise(np.nan_to_num(t_af_j[from_j]) * data.REAL_AREA[cell_idx]) * settings.AF_PROPORTION).astype(np.float32)
        ag_to_beef_t = (tools.amortise(np.nan_to_num(t_beef_j[from_j]) * data.REAL_AREA[cell_idx]) * (1 - settings.AF_PROPORTION)).astype(np.float32)
        w_rm_irrig   = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost = (fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        ag_to_af_t,
                'Transition cost (Ag2AF-Beef)':       ag_to_beef_t,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + ag_to_af_t + ag_to_beef_t + w_rm_irrig + fencing_cost
    return result


def get_carbon_plantings_block_from_ag(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """{(from_m, from_j): $/cell/yr} to Carbon Plantings (Block): establishment + T_MAT transition + irrigation removal."""
    t_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Carbon Plantings (Block)').values
    est_mults  = data.EST_COST_MULTS[target_year]
    irr_scalar = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs  = tools.amortise(data.CP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]).astype(np.float32)
        t_cost     = tools.amortise(np.nan_to_num(t_j[from_j]) * data.REAL_AREA[cell_idx]).astype(np.float32)
        w_rm_irrig = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        t_cost,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
            }
        else:
            result[(from_m, from_j)] = est_costs + t_cost + w_rm_irrig
    return result


def get_sheep_carbon_plantings_belt_from_ag(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """{(from_m, from_j): $/cell/yr} to Sheep Carbon Plantings (Belt): CP_BELT_PROPORTION of (establishment + T_MAT to CP)
    plus the rest as T_MAT to Sheep - modified land, + irrigation removal + fencing."""
    t_cp_j       = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Carbon Plantings (Belt)').values
    t_sheep_j    = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values
    est_mults    = data.EST_COST_MULTS[target_year]
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.CP_BELT_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs     = (tools.amortise(data.CP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]) * settings.CP_BELT_PROPORTION).astype(np.float32)
        ag_to_cp_t    = (tools.amortise(np.nan_to_num(t_cp_j[from_j]) * data.REAL_AREA[cell_idx]) * settings.CP_BELT_PROPORTION).astype(np.float32)
        ag_to_sheep_t = (tools.amortise(np.nan_to_num(t_sheep_j[from_j]) * data.REAL_AREA[cell_idx]) * (1 - settings.CP_BELT_PROPORTION)).astype(np.float32)
        w_rm_irrig    = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost  = (fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        ag_to_cp_t,
                'Transition cost (Ag2CP-Sheep)':      ag_to_sheep_t,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + ag_to_cp_t + ag_to_sheep_t + w_rm_irrig + fencing_cost
    return result


def get_beef_carbon_plantings_belt_from_ag(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """{(from_m, from_j): $/cell/yr} to Beef Carbon Plantings (Belt): CP_BELT_PROPORTION of (establishment + T_MAT to CP)
    plus the rest as T_MAT to Beef - modified land, + irrigation removal + fencing."""
    t_cp_j       = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Carbon Plantings (Belt)').values
    t_beef_j     = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values
    est_mults    = data.EST_COST_MULTS[target_year]
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.CP_BELT_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs    = (tools.amortise(data.CP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]) * settings.CP_BELT_PROPORTION).astype(np.float32)
        ag_to_cp_t   = (tools.amortise(np.nan_to_num(t_cp_j[from_j]) * data.REAL_AREA[cell_idx]) * settings.CP_BELT_PROPORTION).astype(np.float32)
        ag_to_beef_t = (tools.amortise(np.nan_to_num(t_beef_j[from_j]) * data.REAL_AREA[cell_idx]) * (1 - settings.CP_BELT_PROPORTION)).astype(np.float32)
        w_rm_irrig   = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost = (fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        ag_to_cp_t,
                'Transition cost (Ag2CP-Beef)':       ag_to_beef_t,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + ag_to_cp_t + ag_to_beef_t + w_rm_irrig + fencing_cost
    return result


def get_destocked_from_ag(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """{(from_m, from_j): $/cell/yr} to Destocked - natural land: T_MAT transition + HCAS-weighted removal cost (livestock-natural
    sources only, else 0) + irrigation removal."""
    t_j_raw      = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Destocked - natural land').values
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    HCAS_benefit_mult = {lu: 1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[lu] for lu in data.LU_LVSTK_NATURAL}

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        is_eligible  = not np.isnan(t_j_raw[from_j])   # NaN means non-lvstk-natural, zero cost
        hcas_mult    = HCAS_benefit_mult.get(from_j, 0.0)  # LU_LVSTK_NATURAL contains int codes

        t_cost       = tools.amortise(np.nan_to_num(t_j_raw[from_j]) * data.REAL_AREA[cell_idx]).astype(np.float32)
        removal_cost = tools.amortise(hcas_mult * data.EP_EST_COST_HA[cell_idx] * data.REAL_AREA[cell_idx]).astype(np.float32)
        est_costs    = t_cost + removal_cost

        w_rm_irrig   = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if (from_m == 1 and is_eligible) else np.zeros(len(cell_idx), dtype=np.float32)
        )

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
            }
        else:
            result[(from_m, from_j)] = est_costs + w_rm_irrig
    return result


def get_transition_matrix_ag2nonag(data: Data, base_year: int, target_year: int, separate: bool = False) -> dict:
    """{non-ag lu: {(from_m, from_j): $/cell/yr over the source's cells}} — every ag→non-ag lever on ONE shared source map
    (get_base_dvar_mj_cell_map), so the cost slices and the solver's per-source flow vars agree; BECCS reuses the EP costs."""
    mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
    return {
        'Environmental Plantings':       get_env_plant_transitions_from_ag(data, target_year, mj_cell_map, separate),
        'Riparian Plantings':            get_rip_plant_transitions_from_ag(data, target_year, mj_cell_map, separate),
        'Sheep Agroforestry':            get_sheep_agroforestry_transitions_from_ag(data, target_year, mj_cell_map, separate),
        'Beef Agroforestry':             get_beef_agroforestry_transitions_from_ag(data, target_year, mj_cell_map, separate),
        'Carbon Plantings (Block)':      get_carbon_plantings_block_from_ag(data, target_year, mj_cell_map, separate),
        'Sheep Carbon Plantings (Belt)': get_sheep_carbon_plantings_belt_from_ag(data, target_year, mj_cell_map, separate),
        'Beef Carbon Plantings (Belt)':  get_beef_carbon_plantings_belt_from_ag(data, target_year, mj_cell_map, separate),
        'BECCS':                         get_env_plant_transitions_from_ag(data, target_year, mj_cell_map, separate),
        'Destocked - natural land':      get_destocked_from_ag(data, target_year, mj_cell_map, separate),
    }



@lru_cache(maxsize=1)
def get_base_nonag_dvar_k_cell_map(data: Data, base_year: int, threshold: float = None) -> dict:
    """{k: cells} — the source cells of every base-year non-ag land use above `threshold` (default the ROUND_DECIMALS noise
    floor, the ag side's cutoff); every nonzero non-ag holding keeps its own flow vars (cached per (data, base_year))."""
    threshold = 10 ** (-settings.ROUND_DECIMALS) if threshold is None else threshold
    base_dvar_rk = data.non_ag_dvars[base_year]
    return {
        k: np.where(base_dvar_rk[:, k] > threshold)[0]
        for k in range(data.N_NON_AG_LUS)
        if (base_dvar_rk[:, k] > threshold).any()
    }


def get_nonag2ag_ub(data: Data, base_year: int) -> np.ndarray:
    """nonag→ag target upper bound (NLMS, NCELLS, N_AG_LUS), fractional: the base-year share of every non-ag source that can
    reach to_j (T_MAT finite), × no-go × EXCLUDE. Non-ag-source component only; the combined ag ub adds the ag share."""
    non_ag_dvar = data.non_ag_dvars[base_year]                          # (NCELLS, N_NON_AG_LUS)
    
    # Transition exclusion (T_MAT): binary allow per (nonag k → to_j).
    t_kj = (~np.isnan(
        data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )).astype(np.float32)                                               # (k, to_j)
    
    # Reachable land share: sum base-year fractions of every non-ag source that can reach to_j.
    reach_frac_rj = (non_ag_dvar @ t_kj).astype(np.float32)             # (NCELLS, to_j)

    # No-go exclusion: user-defined LUs banned in specific regions.
    no_go = np.ones((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    # Spatial exclusion (data.EXCLUDE).
    x_mrj = data.EXCLUDE.astype(np.float32)
    return (x_mrj * reach_frac_rj[np.newaxis, :, :] * no_go).astype(np.float32)


def get_nonag2ag_lb(data: Data, base_year: int) -> np.ndarray:
    """nonag→ag target lower bound (NLMS, NCELLS, N_AG_LUS): all zeros (placeholder)."""
    return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)


def get_env_plantings_to_ag(
    data: Data, target_year: int, k_cell_map: dict, separate=False
) -> dict:
    """{k: (NLMS, ncells_k, N_AG_LUS) $/cell/yr} from every non-ag source k over its cells: amortised T_MAT transition + water
    licence (dryland source → full target requirement, plus irrigation setup on lm = irr); separate=True → {component: array}."""

    # Hoist invariants outside the per-k loop
    base_ep_to_ag_t = data.EP2AG_TRANSITION_COSTS_HA * data.TRANS_COST_MULTS[target_year]  # (N_AG_LUS,)
    w_mrj = ag_water.get_wreq_matrices(data, target_year - data.YR_CAL_BASE)
    irr_scalar = settings.NEW_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]

    result = {}
    for k, cell_idx in k_cell_map.items():
        # Base transition cost: same for both lm slices (EP source is always dryland)
        base_cost_rj = tools.amortise(np.nan_to_num(base_ep_to_ag_t) * data.REAL_AREA[cell_idx, np.newaxis]).astype(np.float32)  # (n, N_AG_LUS)
        base_cost_mrj = np.stack([base_cost_rj, base_cost_rj], axis=0)  # (NLMS, n, N_AG_LUS)

        # Water license: source water req = 0 (nonag), so delta = full target req
        water_cost_mrj = np.nan_to_num(
            w_mrj[:, cell_idx, :]                          # (NLMS, n, N_AG_LUS)
            * data.WATER_LICENCE_PRICE[cell_idx, np.newaxis]  # (n, 1) → broadcasts
            * data.WATER_LICENSE_COST_MULTS[target_year]
            * settings.INCLUDE_WATER_LICENSE_COSTS
        ).astype(np.float32)

        # New irrigation setup: nonag source is always dryland → lm=1 always pays this
        new_irrig_r = tools.amortise(irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)  # (n,)
        water_cost_mrj[1] += new_irrig_r[:, np.newaxis]

        if separate:
            result[k] = {
                'Transition cost (Non-Ag2Ag)':    base_cost_mrj,
                'Water license cost (Non-Ag2Ag)': water_cost_mrj,
            }
        else:
            result[k] = base_cost_mrj + water_cost_mrj

    return result


def get_rip_plantings_to_ag(data: Data, base_year: int, target_year: int, separate=False):
    """Unused stub: Riparian Plantings → ag reuses the EP costs in get_transition_matrix_nonag2ag."""
    pass


def get_agroforestry_to_ag_base(data: Data, base_year: int, target_year: int, separate=False):
    """Unused stub: the agroforestry levers call get_env_plantings_to_ag directly."""
    pass


def get_sheep_agroforestry_to_ag(
    data: Data, target_year: int, k_cell_map: dict, separate=False
) -> dict:
    """{sheep_af_k: (NLMS, ncells_k, N_AG_LUS)} from Sheep Agroforestry over its source cells: AF_PROPORTION of the EP→ag cost
    plus the rest as the ag2ag cost from dry Sheep, both evaluated on the same cells."""
    sheep_af_k = data.NON_AGRICULTURAL_LANDUSES.index('Sheep Agroforestry')
    cell_idx   = k_cell_map.get(sheep_af_k, np.array([], dtype=int))
    yr_idx     = target_year - data.YR_CAL_BASE
    af_prop    = settings.AF_PROPORTION

    sheep_j             = tools.get_sheep_code(data)
    sheep_tcosts        = get_transition_matrices_ag2ag(data, yr_idx, 0, sheep_j, cell_idx, separate)
    agroforestry_tcosts = get_env_plantings_to_ag(data, target_year, {sheep_af_k: cell_idx}, separate)[sheep_af_k]

    if separate:
        all_keys = set(agroforestry_tcosts) | set(sheep_tcosts)
        zeros = np.zeros((data.NLMS, len(cell_idx), data.N_AG_LUS), dtype=np.float32)
        return {sheep_af_k: {key: (af_prop * agroforestry_tcosts.get(key, zeros) + (1 - af_prop) * sheep_tcosts.get(key, zeros)).astype(np.float32) for key in all_keys}}
    return {sheep_af_k: (af_prop * agroforestry_tcosts + (1 - af_prop) * sheep_tcosts).astype(np.float32)}


def get_beef_agroforestry_to_ag(
    data: Data, target_year: int, k_cell_map: dict, separate=False
) -> dict:
    """{beef_af_k: (NLMS, ncells_k, N_AG_LUS)} from Beef Agroforestry over its source cells: AF_PROPORTION of the EP→ag cost
    plus the rest as the ag2ag cost from dry Beef, both evaluated on the same cells."""
    beef_af_k = data.NON_AGRICULTURAL_LANDUSES.index('Beef Agroforestry')
    cell_idx  = k_cell_map.get(beef_af_k, np.array([], dtype=int))
    yr_idx    = target_year - data.YR_CAL_BASE
    af_prop   = settings.AF_PROPORTION

    beef_j              = tools.get_beef_code(data)
    beef_tcosts         = get_transition_matrices_ag2ag(data, yr_idx, 0, beef_j, cell_idx, separate)
    agroforestry_tcosts = get_env_plantings_to_ag(data, target_year, {beef_af_k: cell_idx}, separate)[beef_af_k]

    if separate:
        all_keys = set(agroforestry_tcosts) | set(beef_tcosts)
        zeros = np.zeros((data.NLMS, len(cell_idx), data.N_AG_LUS), dtype=np.float32)
        return {beef_af_k: {key: (af_prop * agroforestry_tcosts.get(key, zeros) + (1 - af_prop) * beef_tcosts.get(key, zeros)).astype(np.float32) for key in all_keys}}
    return {beef_af_k: (af_prop * agroforestry_tcosts + (1 - af_prop) * beef_tcosts).astype(np.float32)}


def get_carbon_plantings_block_to_ag(data: Data, base_year: int, target_year: int, separate=False):
    """Unused stub: Carbon Plantings (Block) → ag reuses the EP costs in get_transition_matrix_nonag2ag."""
    pass


def get_carbon_plantings_belt_to_ag_base(data: Data, base_year: int, target_year: int, separate=False) -> np.ndarray|dict:
    """Unused stub: the carbon-plantings belt levers call get_env_plantings_to_ag directly."""
    pass


def get_sheep_carbon_plantings_belt_to_ag(
    data: Data, target_year: int, k_cell_map: dict, separate=False
) -> dict:
    """{sheep_cpb_k: (NLMS, ncells_k, N_AG_LUS)} from Sheep Carbon Plantings (Belt) over its source cells: CP_BELT_PROPORTION
    of the EP→ag cost plus the rest as the ag2ag cost from dry Sheep, both evaluated on the same cells."""
    sheep_cpb_k = data.NON_AGRICULTURAL_LANDUSES.index('Sheep Carbon Plantings (Belt)')
    cell_idx    = k_cell_map.get(sheep_cpb_k, np.array([], dtype=int))
    yr_idx      = target_year - data.YR_CAL_BASE
    cp_prop     = settings.CP_BELT_PROPORTION

    sheep_j        = tools.get_sheep_code(data)
    sheep_tcosts   = get_transition_matrices_ag2ag(data, yr_idx, 0, sheep_j, cell_idx, separate)
    cp_belt_tcosts = get_env_plantings_to_ag(data, target_year, {sheep_cpb_k: cell_idx}, separate)[sheep_cpb_k]

    if separate:
        all_keys = set(cp_belt_tcosts) | set(sheep_tcosts)
        zeros = np.zeros((data.NLMS, len(cell_idx), data.N_AG_LUS), dtype=np.float32)
        return {sheep_cpb_k: {key: (cp_prop * cp_belt_tcosts.get(key, zeros) + (1 - cp_prop) * sheep_tcosts.get(key, zeros)).astype(np.float32) for key in all_keys}}
    return {sheep_cpb_k: (cp_prop * cp_belt_tcosts + (1 - cp_prop) * sheep_tcosts).astype(np.float32)}


def get_beef_carbon_plantings_belt_to_ag(
    data: Data, target_year: int, k_cell_map: dict, separate=False
) -> dict:
    """{beef_cpb_k: (NLMS, ncells_k, N_AG_LUS)} from Beef Carbon Plantings (Belt) over its source cells: CP_BELT_PROPORTION
    of the EP→ag cost plus the rest as the ag2ag cost from dry Beef, both evaluated on the same cells."""
    beef_cpb_k = data.NON_AGRICULTURAL_LANDUSES.index('Beef Carbon Plantings (Belt)')
    cell_idx   = k_cell_map.get(beef_cpb_k, np.array([], dtype=int))
    yr_idx     = target_year - data.YR_CAL_BASE
    cp_prop    = settings.CP_BELT_PROPORTION

    beef_j         = tools.get_beef_code(data)
    beef_tcosts    = get_transition_matrices_ag2ag(data, yr_idx, 0, beef_j, cell_idx, separate)
    cp_belt_tcosts = get_env_plantings_to_ag(data, target_year, {beef_cpb_k: cell_idx}, separate)[beef_cpb_k]

    if separate:
        all_keys = set(cp_belt_tcosts) | set(beef_tcosts)
        zeros = np.zeros((data.NLMS, len(cell_idx), data.N_AG_LUS), dtype=np.float32)
        return {beef_cpb_k: {key: (cp_prop * cp_belt_tcosts.get(key, zeros) + (1 - cp_prop) * beef_tcosts.get(key, zeros)).astype(np.float32) for key in all_keys}}
    return {beef_cpb_k: (cp_prop * cp_belt_tcosts + (1 - cp_prop) * beef_tcosts).astype(np.float32)}


def get_beccs_to_ag(data: Data, target_year, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """Unused stub: BECCS → ag reuses the EP costs in get_transition_matrix_nonag2ag."""
    pass
    

def get_destocked_to_ag_base(
    data: Data, target_year: int, cell_idx: np.ndarray, separate: bool = False
) -> dict:
    """Destocked→ag cost (NLMS, len(cell_idx), N_AG_LUS) $/cell/yr, amortised, unmasked: Destocked's own T_MAT row + water licence
    (dryland source, irrigation setup on lm = irr) + carbon release of natural-equivalent land (the ag2ag emissions of an Unallocated-natural source)."""
    yr_idx = target_year - data.YR_CAL_BASE

    # --- Transition cost: Destocked's OWN T_MAT row; dryland source (both target lm equal) ---
    t_j = data.T_MAT.sel(from_lu='Destocked - natural land', to_lu=data.AGRICULTURAL_LANDUSES).values  # (N_AG_LUS,)
    t_j = np.nan_to_num(t_j) * data.TRANS_COST_MULTS[target_year]
    trans_rj  = tools.amortise(t_j[np.newaxis, :] * data.REAL_AREA[cell_idx, np.newaxis]).astype(np.float32)  # (n, N_AG_LUS)
    trans_mrj = np.stack([trans_rj, trans_rj], axis=0)                                                        # (NLMS, n, N_AG_LUS)

    # --- Water license cost: destocked source dryland (req 0) → delta = full target req ---
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    water_mrj = np.nan_to_num(
        w_mrj[:, cell_idx, :]
        * data.WATER_LICENCE_PRICE[cell_idx, np.newaxis]
        * data.WATER_LICENSE_COST_MULTS[target_year]
        * settings.INCLUDE_WATER_LICENSE_COSTS
    ).astype(np.float32)
    new_irrig_r = tools.amortise(
        settings.NEW_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * data.REAL_AREA[cell_idx]
    ).astype(np.float32)
    water_mrj[1] += new_irrig_r[:, np.newaxis]

    # --- Carbon release: natural-equivalent land → modified/lvstk-natural (source-parameterised) ---
    unallocated_j = tools.get_unallocated_natural_land_code(data)
    ghg = ag_ghg.get_ghg_transition_emissions(data, 0, unallocated_j, cell_idx, separate=False)   # (NLMS, ncells, N_AG_LUS) t/cell
    carbon_mrj = tools.amortise(ghg * data.get_carbon_price_by_yr_idx(yr_idx)).astype(np.float32)

    if separate:
        return {
            'Transition cost (Non-Ag2Ag)':     trans_mrj,
            'Water license cost (Non-Ag2Ag)':  water_mrj,
            'Carbon release cost (Non-Ag2Ag)': carbon_mrj,
        }
    return trans_mrj + water_mrj + carbon_mrj


def get_destocked_to_ag(
    data: Data, target_year: int, k_cell_map: dict, separate: bool = False
) -> dict:
    """{destocked_k: get_destocked_to_ag_base(...)} over the Destocked source cells of `k_cell_map`."""
    destocked_k = data.NON_AGRICULTURAL_LANDUSES.index('Destocked - natural land')
    cell_idx    = k_cell_map.get(destocked_k, np.array([], dtype=int))
    return {destocked_k: get_destocked_to_ag_base(data, target_year, cell_idx, separate)}




def get_transition_matrix_nonag2ag(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """{non-ag lu: {k: (NLMS, ncells_k, N_AG_LUS)}} — every non-ag→ag lever on ONE shared source map (get_base_nonag_dvar_k_cell_map);
    EP / RP / CP-block / BECCS share the EP costs, and row_builder selects the per-source diagonal dict[k] for each lu."""
    k_cell_map = get_base_nonag_dvar_k_cell_map(data, base_year)
    ep_to_ag   = get_env_plantings_to_ag(data, target_year, k_cell_map, separate)
    return {
        'Environmental Plantings':       ep_to_ag,
        'Riparian Plantings':            ep_to_ag,
        'Sheep Agroforestry':            get_sheep_agroforestry_to_ag(data, target_year, k_cell_map, separate),
        'Beef Agroforestry':             get_beef_agroforestry_to_ag(data, target_year, k_cell_map, separate),
        'Carbon Plantings (Block)':      ep_to_ag,
        'Sheep Carbon Plantings (Belt)': get_sheep_carbon_plantings_belt_to_ag(data, target_year, k_cell_map, separate),
        'Beef Carbon Plantings (Belt)':  get_beef_carbon_plantings_belt_to_ag(data, target_year, k_cell_map, separate),
        'BECCS':                         ep_to_ag,
        'Destocked - natural land':      get_destocked_to_ag(data, target_year, k_cell_map, separate),
    }


def get_nonag2nonag_transition_matrix(data: Data) -> np.ndarray:
    """Non-ag → non-ag transition costs (NCELLS, N_NON_AG_LUS): all zeros, such transitions are not allowed."""
    return np.zeros((data.NCELLS, data.N_NON_AG_LUS)).astype(np.float32)




def get_non_ag_ub_matrices(data: Data, base_dvar_nonag_rk, base_dvar_ag_mrj) -> np.ndarray:
    """Non-ag target upper bound (NCELLS, N_NON_AG_LUS), fractional: the base-year share of every ag / non-ag source that can reach k
    (T_MAT finite), then no-go, the irreversible lock-in override, and the Destocked / Riparian physical caps (np.minimum)."""
    # 1. Transition exclusion (T_MAT), FRACTIONAL: reachable land share per non-ag target.
    t_jk = (~np.isnan(
        data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.NON_AGRICULTURAL_LANDUSES).values
    )).astype(np.float32)                                                                # (ag_from_j, nonag_k)
    t_kk = (~np.isnan(
        data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.NON_AGRICULTURAL_LANDUSES).values
    )).astype(np.float32)                                                                # (nonag_from_k, nonag_k)
    ag_frac_rj = base_dvar_ag_mrj.sum(axis=0)                                            # (NCELLS, ag_j)
    t_rk = (ag_frac_rj @ t_jk + base_dvar_nonag_rk @ t_kk).astype(np.float32)            # (NCELLS, nonag_k)

    # 2. No-go zones for non-ag LUs set UB=0 for the affected columns in the no-go regions.
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_NON_AG, data.NO_GO_LANDUSE_NON_AG):
            no_go_j = data.NON_AGRICULTURAL_LANDUSES.index(no_go_desc)
            t_rk[:, no_go_j] *= no_go_x_r

    # 3. Existing irreversible non-ag allocations forced to UB=1 so their lock-in (lb) survives.
    for k, k_name in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        if not settings.NON_AG_LAND_USES_REVERSIBLE[k_name]:
            t_rk[base_dvar_nonag_rk[:, k] >= settings.FEASIBILITY_TOLERANCE, k] = 1

    # 4. Destocked physical cap: UB ≤ livestock-natural ag fraction + already-destocked fraction.
    #    (min, not ×, because the reach is already fractional.)
    destock_k = data.NON_AGRICULTURAL_LANDUSES.index('Destocked - natural land')
    eligible_frac_r = (
        base_dvar_ag_mrj[:, :, data.LU_LVSTK_NATURAL].sum(axis=(0, 2))
        + base_dvar_nonag_rk[:, destock_k]
    ).astype(np.float32)
    eligible_frac_r[eligible_frac_r < settings.FEASIBILITY_TOLERANCE] = 0.0
    t_rk[:, destock_k] = np.minimum(t_rk[:, destock_k], eligible_frac_r)

    # 5. Riparian Plantings physical cap: UB ≤ stream-buffer fraction of the cell.
    RP_j = data.NON_AGRICULTURAL_LANDUSES.index('Riparian Plantings')
    t_rk[:, RP_j] = np.minimum(t_rk[:, RP_j], data.RP_PROPORTION)

    return t_rk.astype(np.float32)


def get_non_ag_lb_matrices(data: Data, base_year) -> np.ndarray:
    """Non-ag target lower bound (NCELLS, N_NON_AG_LUS): the floor-truncated base-year holding of every irreversible non-ag land
    use (zeros at the base year / before any allocation) — the lock-in floor the solver may not reduce."""

    if base_year == data.YR_CAL_BASE or base_year not in data.non_ag_dvars:
        return np.zeros((data.NCELLS, len(settings.NON_AG_LAND_USES))).astype(np.float32)

    prev_dvars = data.non_ag_dvars[base_year].astype(np.float32)
    scale = 10 ** settings.ROUND_DECIMALS

    # Only lock in non-reversible LUs; reversible LUs keep lb = 0.
    lb_rk = np.zeros_like(prev_dvars)
    for k, k_name in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        if not settings.NON_AG_LAND_USES_REVERSIBLE[k_name]:
            # Floor-truncate to prevent float32 upward rounding from inflating lb
            lb_rk[:, k] = np.floor(prev_dvars[:, k] * scale) / scale

    # NonAg lb can not exceed the available ag-mask proportion of the cell
    return tools.clamp_dvar_bound(
        lb_rk, 0.0, data.AG_MASK_PROPORTION_R[:, np.newaxis], 'NonAg lb capped to ag-mask'
    )
