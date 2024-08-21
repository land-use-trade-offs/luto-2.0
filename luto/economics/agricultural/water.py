# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
#
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>.

"""
Pure functions to calculate water net yield by lm, lu and water limits.
"""


from typing import Optional
import numpy as np
from collections import defaultdict

import luto.settings as settings
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.data import Data
from luto.economics.agricultural.quantity import get_yield_pot, lvs_veg_types
import luto.economics.non_agricultural.water as non_ag_water


def get_wreq_matrices(data: Data, yr_idx):
    """
    Return water requirement (water use by irrigation and livestock drinking water) matrices 
    by land management, cell, and land-use type.
    
    Parameters:
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
    
    Returns:
        numpy.ndarray: The w_mrj <unit: ML/cell> water requirement matrices, indexed (m, r, j).
    """
    
    # Stack water requirements data
    w_req_mrj = np.stack(( data.WREQ_DRY_RJ, data.WREQ_IRR_RJ ))    # <unit: ML/head|ha>
    
    # Covert water requirements units from ML/head to ML/ha
    for j, lu in enumerate(data.AGRICULTURAL_LANDUSES):
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)
            w_req_mrj[0, :, j] = w_req_mrj[0, :, j] * get_yield_pot(data, lvs, veg, 'dry', yr_idx)  # Water reqs depend on current stocking rate for drinking water
            w_req_mrj[1, :, j] = w_req_mrj[1, :, j] * get_yield_pot(data, lvs, veg, 'irr', 0)       # Water reqs depend on initial stocking rate for irrigation
    
    # Convert to ML per cell via REAL_AREA
    w_req_mrj *= data.REAL_AREA[:, np.newaxis]                      # <unit: ML/ha> * <unit: ha/cell> -> <unit: ML/cell>

    return w_req_mrj


def get_wyield_matrices(data: Data, yr_idx) -> np.ndarray:
    """
    Return water yield matrices by land management, cell, and land-use type.
    
    Parameters:
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
    
    Returns:
        numpy.ndarray: The w_mrj <unit: ML/cell> water yield matrices, indexed (m, r, j).
    """
    w_yield_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))

    w_yield_dr = data.get_water_dr_yield_for_yr_idx(yr_idx)
    w_yield_sr = data.get_water_sr_yield_for_yr_idx(yr_idx)
    w_yield_nl = data.get_water_nl_yield_for_yr_idx(yr_idx, w_yield_dr, w_yield_sr)

    
    for j in range(data.N_AG_LUS):
        if j in data.LU_SHALLOW_ROOTED:
            for m in range(data.NLMS):
                w_yield_mrj[m, :, j] = w_yield_sr * data.REAL_AREA

        elif j in data.LU_DEEP_ROOTED:
            for m in range(data.NLMS):
                w_yield_mrj[m, :, j] = w_yield_dr * data.REAL_AREA

        elif j in data.LU_NATURAL:
            for m in range(data.NLMS):
                w_yield_mrj[m, :, j] = w_yield_nl * data.REAL_AREA

        else:
            raise ValueError(
                f"Land use {j} ({data.AGLU2DESC[j]}) missing from all of "
                f"data.LU_SHALLOW_ROOTED, data.LU_DEEP_ROOTED, data.LU_NATURAL "
                f"(requires root definition)."
            )

    return w_yield_mrj


def get_water_ccimpact(data: Data, yr_idx) -> dict[int, float]:
    """
    Return water climate change (CC) impact matrices by land management, cell, and land-use type. 
    Note the impact comes from cells that LUTO does not look at.
    
    Parameters:
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
    
    Returns:
        dict[int, float]: <unit: ML/cell> dictionary of water CC impact amounts, keys being region IDs
    """
    yr_cal = data.YR_CAL_BASE + yr_idx

    if settings.WATER_REGION_DEF == 'River Region':
        ccimpact_array = data.RR_CCIMPACT

    elif settings.WATER_REGION_DEF == 'Drainage Division':
        ccimpact_array = data.DD_CCIMPACT

    else:
        raise ValueError(
            f"Invalid value for setting WATER_REGION_DEF: '{settings.WATER_REGION_DEF}' "
            f"(must be either 'River Region' or 'Drainage Division')."
        )
    
    return ccimpact_array.loc[yr_cal, :].to_dict()


def get_water_net_yield_matrices(data: Data, yr_idx):
    """
    Return water net yield matrices by land management, cell, and land-use type.
    The resulting array is used as the net yield w_mrj array in the input data of the solver.

    Parameters:
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
    
    Returns:
        numpy.ndarray: The w_mrj <unit: ML/cell> water net yield matrices, indexed (m, r, j).
    """
    return get_wyield_matrices(data, yr_idx) - get_wreq_matrices(data, yr_idx)


def get_asparagopsis_effect_w_mrj(data: Data, yr_idx):
    """
    Applies the effects of using asparagopsis to the water net yield data
    for all relevant agr. land uses.

    Args:
        data (object): The data object containing relevant information.
        w_mrj (ndarray, <unit:ML/cell>): The water net yield data for all land uses.
        yr_idx (int): The index of the year.

    Returns:
        ndarray <unit:ML/cell>: The updated water net yield data with the effects of using asparagopsis.

    Notes:
        Asparagopsis taxiformis has no effect on the water required.
    """
    if not settings.AG_MANAGEMENTS['Asparagopsis taxiformis']:
        return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))

    land_uses = AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, "Water Impacts"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in water used.
            # Since the effect applies to water use, it effects the net yield negatively.
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)

    return w_mrj_effect


def get_precision_agriculture_effect_w_mrj(data: Data, yr_idx):
    """
    Applies the effects of using precision agriculture to the water net yield data
    for all relevant agricultural land uses.

    Parameters:
    - data: The data object containing relevant information for the calculation.
    - w_mrj <unit:ML/cell>: The original water net yield data for different land uses.
    - yr_idx: The index representing the year for which the calculation is performed.

    Returns:
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data after applying precision agriculture effects.
    """
    if not settings.AG_MANAGEMENTS['Precision Agriculture']:
        return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))

    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each land use
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.PRECISION_AGRICULTURE_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in water used.
            # Since the effect applies to water use, it effects the net yield negatively.
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)

    return w_mrj_effect


def get_ecological_grazing_effect_w_mrj(data: Data, yr_idx):
    """
    Applies the effects of using ecological grazing to the water net yield data
    for all relevant agricultural land uses.

    Parameters:
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for different land uses.
    - yr_idx: The index of the year.

    Returns:
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data after applying ecological grazing effects.
    """
    if not settings.AG_MANAGEMENTS['Ecological Grazing']:
        return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))

    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each land use
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, "INPUT-wrt_water-required"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in water used.
            # Since the effect applies to water use, it effects the net yield negatively.
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)

    return w_mrj_effect


def get_savanna_burning_effect_w_mrj(data):
    """
    Applies the effects of using savanna burning to the water net yield data
    for all relevant agr. land uses.

    Savanna burning does not affect water usage, so return an array of zeros.

    Parameters:
    - data: The input data object containing information about land uses and water net yield.

    Returns:
    - An array of zeros with dimensions (NLMS, NCELLS, nlus), where:
        - NLMS: Number of land management systems
        - NCELLS: Number of cells
        - nlus: Number of land uses affected by savanna burning
    """
    if not settings.AG_MANAGEMENTS['Savanna Burning']:
        return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))

    nlus = len(AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning'])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_agtech_ei_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using AgTech EI to the water net yield data
    for all relevant agr. land uses.

    Parameters:
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns:
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data with AgTech EI effects applied.
    """
    if not settings.AG_MANAGEMENTS['AgTech EI']:
        return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.AGTECH_EI_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in water used.
            # Since the effect applies to water use, it effects the net yield negatively.
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)
            

    return w_mrj_effect


def get_agricultural_management_water_matrices(data: Data, yr_idx) -> dict[str, np.ndarray]:
    asparagopsis_data = (
        get_asparagopsis_effect_w_mrj(data, yr_idx) 
        if settings.AG_MANAGEMENTS['Asparagopsis taxiformis'] 
        else np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    )
    precision_agriculture_data = (
        get_precision_agriculture_effect_w_mrj(data, yr_idx) 
        if settings.AG_MANAGEMENTS['Precision Agriculture'] 
        else np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    )
    eco_grazing_data = (
        get_ecological_grazing_effect_w_mrj(data, yr_idx) 
        if settings.AG_MANAGEMENTS['Ecological Grazing'] 
        else np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    )
    sav_burning_data = (
        get_savanna_burning_effect_w_mrj(data) 
        if settings.AG_MANAGEMENTS['Savanna Burning'] 
        else np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    )
    agtech_ei_data = (
        get_agtech_ei_effect_w_mrj(data, yr_idx) 
        if settings.AG_MANAGEMENTS['AgTech EI'] 
        else np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    )

    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
    }


def calc_water_net_yield_for_region(
    region_ind: np.ndarray,
    am2j: dict[str, list[int]],
    ag_dvars: np.ndarray,
    non_ag_dvars: np.ndarray,
    ag_man_dvars: dict[str, np.ndarray],
    ag_w_mrj: np.ndarray,
    non_ag_w_rk: np.ndarray,
    ag_man_w_mrj: dict[str, np.ndarray],
    w_cc_impact_reg: float,
) -> float:
    ag_contr = (ag_w_mrj[:, region_ind, :] * ag_dvars[:, region_ind, :]).sum()
    non_ag_contr = (non_ag_w_rk[region_ind, :] * non_ag_dvars[region_ind, :]).sum()
    ag_man_contr = sum(
        (ag_man_w_mrj[am][:, region_ind, j_idx] * ag_man_dvars[am][:, region_ind, j]).sum()
        for am, am_j_list in am2j.items()
        for j_idx, j in enumerate(am_j_list)
    )
    return ag_contr + non_ag_contr + ag_man_contr + w_cc_impact_reg
    

def calc_water_net_yield_by_region_in_year_from_data(
    data: Data,
    yr_cal: int,
    ag_w_mrj: Optional[np.ndarray] = None,
    non_ag_w_rk: Optional[np.ndarray] = None,
    ag_man_w_mrj: Optional[dict[str, np.ndarray]] = None,
    w_cc_impact: Optional[dict[int, float]] = None,
) -> Optional[dict[int, float]]:
    """
    Gets net water yield in a given year if the year has a solution.
    """
    # Data for river regions or drainage divisions
    if settings.WATER_REGION_DEF == 'Drainage Division':
        region_limits = data.DRAINDIV_LIMITS
        region_id = data.DRAINDIV_ID
    elif settings.WATER_REGION_DEF == 'River Region':
        region_limits = data.RIVREG_LIMITS
        region_id = data.RIVREG_ID
    else:
        print(
            f"Incorrect option '{settings.WATER_REGION_DEF}' for WATER_REGION_DEF in settings "
            f"(Must be either 'Drainage Division' or 'River Region')."
        )

    # Ensure solution exists for the given yr_cal
    if any(
        [ yr_cal not in vars_array for vars_array in 
          [data.ag_dvars, data.non_ag_dvars, data.ag_man_dvars] ]
    ):
        raise ValueError(
            f"Cannot calculate water usage for year {yr_cal}: "
            f"no solution data available (has the simulation been run?)"
        )

    am2j = {
        am: [data.DESC2AGLU[lu] for lu in am_lus]
        for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items()
    }

    # Prepare water matrices for calculation
    yr_idx = yr_cal - data.YR_CAL_BASE
    ag_w_mrj = ag_w_mrj if ag_w_mrj is not None else get_water_net_yield_matrices(data, yr_idx)
    non_ag_w_rk = non_ag_w_rk if non_ag_w_rk is not None else non_ag_water.get_w_net_yield_matrix(
        data, ag_w_mrj, data.lumaps[yr_cal], yr_idx
    )
    ag_man_w_mrj = ag_man_w_mrj if ag_man_w_mrj is not None else get_agricultural_management_water_matrices(data, yr_idx)
    w_cc_impact = w_cc_impact if w_cc_impact is not None else get_water_ccimpact(data, yr_idx)
    
    # Calculate net yields
    net_yield_by_region = {}
    for region in region_limits:
        # Get indices of cells in region
        ind = np.flatnonzero(region_id == region).astype(np.int32)

        net_yield_by_region[region] = calc_water_net_yield_for_region(
            ind,
            am2j,
            data.ag_dvars[yr_cal],
            data.non_ag_dvars[yr_cal],
            data.ag_man_dvars[yr_cal],
            ag_w_mrj,
            non_ag_w_rk,
            ag_man_w_mrj,
            w_cc_impact[region],
        )

    return net_yield_by_region


def _get_historical_water_usage_by_regions(data: Data) -> dict[int, tuple[str, float, np.ndarray]]:
    """
    Gets historical water usage figures, split by regions.
    These figures are the basis for the net yield targets.

    Returns
    -------
    dict[int, ...]: A dictionary where the keys are region IDs and the values are tuples 
        containing (region name, historical yield, cells in region)

    """
    # Get historical yields of regions, stored in data.RIVREG_LIMITS and data.DRAINDIV_LIMITS
    baseline_reg_water_use_limits: dict[int, str] = {}
    if settings.WATER_REGION_DEF == 'River Region':
        baseline_reg_water_use_limits = data.RIVREG_LIMITS
        region_id = data.RIVREG_ID
        region_names = data.RIVREG_DICT

    elif settings.WATER_REGION_DEF == 'Drainage Division':
        baseline_reg_water_use_limits = data.DRAINDIV_LIMITS
        region_id = data.DRAINDIV_ID
        region_names = data.DRAINDIV_DICT

    else:
        raise ValueError(
            f"Invalid value for setting WATER_REGION_DEF: '{settings.WATER_REGION_DEF}' "
            f"(must be either 'River Region' or 'Drainage Division')."
        )
    
    # Net baseline water yield
    net_baseline_reg_water_yield = {}
    for reg in baseline_reg_water_use_limits.keys():
        # Get indices of cells in region
        ind = np.flatnonzero(region_id == reg).astype(np.int32)

        # Base the net yield target on the historical yield
        historical_yield = baseline_reg_water_use_limits[reg]

        net_baseline_reg_water_yield[reg] = (
            region_names[reg], historical_yield, ind
        )

    return net_baseline_reg_water_yield


def get_water_net_yield_limit_values(
    data: Data, yr_cal: int,
) -> dict[int, tuple[str, float, float, np.ndarray]]:
    """
    Return water net yield limits for regions (River Regions or Drainage Divisions as specified in luto.settings.py).
    
    Limits are year-specific, beginning at the 2010 net water yields and decreasing to the defined limits
    in the year given by settings.WATER_LIMITS_TARGET_YEAR.

    Parameters:
    - data: The data object containing the necessary input data.

    Returns:
    water_net_yield_limits: A dictionary of tuples containing the water use limits for each region\n
      region index:(
      - region name 
      - histircal water yield
      - lowest water yield allowed 
      - indices of cells in the region
      )

    Raises:
    - None

    """
    if any([not data.YR_CAL_BASE <= yr <= 2100 for yr in settings.WATER_YIELD_TARGETS]):
        raise ValueError(
            f"Setting WATER_YIELD_TARGETS must be defined for years between " 
            f"{data.YR_CAL_BASE} and 2100."
        )

    latest_target_year = max(settings.WATER_YIELD_TARGETS)
    if data.WATER_LIMITS_BY_YEAR: 
        return (
            data.WATER_LIMITS_BY_YEAR[latest_target_year] 
            if yr_cal >= latest_target_year
            else data.WATER_LIMITS_BY_YEAR[yr_cal]
        )
    
    # Limits do not yet exist and must be calculated
    historical_yields_dict = _get_historical_water_usage_by_regions(data)
    base_yr_net_yield_by_reg = calc_water_net_yield_by_region_in_year_from_data(data, data.YR_CAL_BASE)

    limits_by_region_year = defaultdict(dict)

    # Get the intervals for which gradients must be calculated
    if len(settings.WATER_YIELD_TARGETS) > 1:
        gradients_start_ends = [
            (y1, y2) for y1, y2 
            in zip(
                list(settings.WATER_YIELD_TARGETS.keys())[: -1],
                list(settings.WATER_YIELD_TARGETS.keys())[1: ],
            )
        ]
        first_target_year = gradients_start_ends[0][0]
        if first_target_year != data.YR_CAL_BASE:
            gradients_start_ends = [(data.YR_CAL_BASE, first_target_year)] + gradients_start_ends
    else:
        gradients_start_ends = [(data.YR_CAL_BASE, list(settings.WATER_YIELD_TARGETS.keys())[0])]

    for yr_start, yr_end in gradients_start_ends:
        n_years_cal = yr_end - yr_start + 1
        calc_years = np.linspace(yr_start, yr_end, n_years_cal).astype(int)

        for region, (name, hist_yield, ind) in historical_yields_dict.items():
            reg_target_yr_end = hist_yield * settings.WATER_YIELD_TARGETS[yr_end]
            if yr_start == data.YR_CAL_BASE:
                reg_target_yr_start = min(base_yr_net_yield_by_reg[region], reg_target_yr_end)
            else:
                saved_limit = limits_by_region_year[yr_start][region][2]
                reg_target_yr_start = min(saved_limit, reg_target_yr_end)

            yrs_targets = np.linspace(reg_target_yr_start, reg_target_yr_end, n_years_cal)
            for yr, limit in zip(calc_years, yrs_targets):
                limits_by_region_year[yr][region] = (name, hist_yield, limit, ind)
    
    # Save to data object to avoid re-calculating in the future.
    data.WATER_LIMITS_BY_YEAR = dict(limits_by_region_year)

    return (
        data.WATER_LIMITS_BY_YEAR[latest_target_year] if yr_cal >= latest_target_year
        else data.WATER_LIMITS_BY_YEAR[yr_cal]
    )
 


"""
Water logic *** a little outdated but maybe still useful ***

The limits are related to the pre-European inflows into rivers. As a proxy
for these inflows are used the flows that would result if all cells had
deeply-rooted vegetation. The values from 1985 are used for this as these
do not incorporate climate change corrections on rainfall. So the limit is
a _lower_ limit, it is a bottom, not a cap.

Performance relative to the cap is then composed of two parts:
    1. Water used for irrigation or as livestock drinking water, and
    2. Water retained in the soil by vegetation.
The former (1) is calculated using the water requirements (WR) data. This
water use effectively raises the lower limit, i.e. is added to it. The latter
is computed from the water yields data. The water yield data state the
inflows from each cell based on which type of vegetation (deeply or shallowly
rooted) and which SSP projection.

The first approach is to try to limit water stress to below 40% of the
pre-European inflows. This means the limit is to have _at least_ 60% of
the 1985 inflows if all cells had deeply rooted vegetation. If these 1985
inflows are called L, then inflows need to be >= .6L. Inflows are computed
using the water yield data based on the vegetation the simulation wants to
plant -- i.e. deeply or shallowly rooted, corresponding to trees and crops,
roughly. Subtracted from this is then the water use for irrigation. Since
plants do not fully use the irrigated water, some of the irrigation actually
also adds to the inflows. This fraction is the _complement_ of the irrigation
efficiency. So either the irrigation efficiency corrected water use is added
to the lower limit, or the complement of it (the irrigation running off) is
added to the inflow.
"""
