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

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import pairwise
from typing import Optional


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


def get_wyield_matrices(
    data: Data, yr_idx:int, 
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return water yield matrices for YR_CAL_BASE (2010) by land management, cell, and land-use type.

    Parameters:
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
        water_dr_yield (ndarray, <unit:ML/cell>): The water yield for deep-rooted vegetation.
        water_sr_yield (ndarray, <unit:ML/cell>): The water yield for shallow-rooted vegetation.

    Returns:
        numpy.ndarray: The w_mrj <unit: ML/cell> water yield matrices, indexed (m, r, j).
    """
    w_yield_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))

    w_yield_dr = data.WATER_YIELD_DR_FILE[yr_idx] if water_dr_yield is None else water_dr_yield
    w_yield_sr = data.WATER_YIELD_SR_FILE[yr_idx] if water_sr_yield is None else water_sr_yield
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


def get_water_net_yield_matrices(
    data: Data, 
    yr_idx, 
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Return water net yield matrices by land management, cell, and land-use type.
    The resulting array is used as the net yield w_mrj array in the input data of the solver.

    Parameters:
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
        water_dr_yield (ndarray, <unit:ML/cell>): The water yield for deep-rooted vegetation.
        water_sr_yield (ndarray, <unit:ML/cell>): The water yield for shallow-rooted vegetation.
        
    Notes:
        Provides the `water_dr_yield` or `water_sr_yield` will make the `yr_idx` useless because
        the water net yield will be calculated based on the provided water yield data regardless of the year.

    Returns:
        numpy.ndarray: The w_mrj <unit: ML/cell> water net yield matrices, indexed (m, r, j).
    """
    return get_wyield_matrices(data, yr_idx, water_dr_yield, water_sr_yield) - get_wreq_matrices(data, yr_idx)


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


def get_savanna_burning_effect_w_mrj(data: Data):
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

    nlus = len(AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning'])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_agtech_ei_effect_w_mrj(data: Data, yr_idx):
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


def get_biochar_effect_w_mrj(data: Data, yr_idx):
    """
    Applies the effects of using Biochar to the water net yield data
    for all relevant agr. land uses.

    Parameters:
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns:
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data with Biochar applied.
    """

    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Biochar']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.BIOCHAR_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)

    return w_mrj_effect


def get_agricultural_management_water_matrices(data: Data, yr_idx) -> dict[str, np.ndarray]:
    asparagopsis_data = (
        get_asparagopsis_effect_w_mrj(data, yr_idx)
        if settings.AG_MANAGEMENTS['Asparagopsis taxiformis']
        else np.zeros((data.NLMS, data.NCELLS, len(AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis'])))
    )
    precision_agriculture_data = (
        get_precision_agriculture_effect_w_mrj(data, yr_idx)
        if settings.AG_MANAGEMENTS['Precision Agriculture']
        else np.zeros((data.NLMS, data.NCELLS, len(AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture'])))
    )
    eco_grazing_data = (
        get_ecological_grazing_effect_w_mrj(data, yr_idx)
        if settings.AG_MANAGEMENTS['Ecological Grazing']
        else np.zeros((data.NLMS, data.NCELLS, len(AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing'])))
    )
    sav_burning_data = (
        get_savanna_burning_effect_w_mrj(data)
        if settings.AG_MANAGEMENTS['Savanna Burning']
        else np.zeros((data.NLMS, data.NCELLS, len(AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning'])))
    )
    agtech_ei_data = (
        get_agtech_ei_effect_w_mrj(data, yr_idx)
        if settings.AG_MANAGEMENTS['AgTech EI']
        else np.zeros((data.NLMS, data.NCELLS, len(AG_MANAGEMENTS_TO_LAND_USES['AgTech EI'])))
    )
    biochar_data = (
        get_biochar_effect_w_mrj(data, yr_idx)
        if settings.AG_MANAGEMENTS['Biochar']
        else np.zeros((data.NLMS, data.NCELLS, len(AG_MANAGEMENTS_TO_LAND_USES['Biochar'])))
    )

    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
        'Biochar': biochar_data,
    }


def get_water_outside_luto_study_area(data: Data, yr_cal:int) ->  dict[int, float]:
    """
    Return water yield from the outside regions of LUTO study area.

    Parameters:
        data (object): The data object containing the required data.
        yr_cal (int): The year for which the water yield is calculated.

    Returns:
        dict[int, dict[int, float]]: <unit: ML/cell> dictionary of water yield amounts.
            The first key is year and the second key is region ID.
    """
    if settings.WATER_REGION_DEF == 'River Region':
        water_yield_arr = data.WATER_OUTSIDE_LUTO_RR

    elif settings.WATER_REGION_DEF == 'Drainage Division':
        water_yield_arr = data.WATER_OUTSIDE_LUTO_DD

    else:
        raise ValueError(
            f"Invalid value for setting WATER_REGION_DEF: '{settings.WATER_REGION_DEF}' "
            f"(must be either 'River Region' or 'Drainage Division')."
        )

    return water_yield_arr.loc[yr_cal].to_dict()


def get_water_outside_luto_study_area_from_hist_level(data: Data) -> dict[int, float]:
    """
    Return water yield from the outside regions of LUTO study area based on historical levels.

    Parameters:
        data (object): The data object containing the required data.

    Returns:
        dict[int, float]: <unit: ML/cell> dictionary of water yield amounts.
    """
    if settings.WATER_REGION_DEF == 'River Region':
        water_yield_arr = data.WATER_OUTSIDE_LUTO_RR_HIST

    elif settings.WATER_REGION_DEF == 'Drainage Division':
        water_yield_arr = data.WATER_OUTSIDE_LUTO_DD_HIST

    else:
        raise ValueError(
            f"Invalid value for setting WATER_REGION_DEF: '{settings.WATER_REGION_DEF}' "
            f"(must be either 'River Region' or 'Drainage Division')."
        )

    return water_yield_arr


def calc_water_net_yield_for_region(
    region_ind: np.ndarray,
    am2j: dict[str, list[int]],
    ag_dvars: np.ndarray,
    non_ag_dvars: np.ndarray,
    ag_man_dvars: dict[str, np.ndarray],
    ag_w_mrj: np.ndarray,
    non_ag_w_rk: np.ndarray,
    ag_man_w_mrj: dict[str, np.ndarray],
    water_yield_outside_luto_study_area: float,
) -> float:
    '''
    This function calculates the net water yield for a given region. \n

    `Water_yield` = `ag_contr` + `non_ag_contr` + `ag_mam_contr` + `water_yield_outside_luto_study_area`
    '''
    
    ag_contr = (ag_w_mrj[:, region_ind, :] * ag_dvars[:, region_ind, :]).sum()
    non_ag_contr = (non_ag_w_rk[region_ind, :] * non_ag_dvars[region_ind, :]).sum()
    ag_man_contr = sum(
        (ag_man_w_mrj[am][:, region_ind, j_idx] * ag_man_dvars[am][:, region_ind, j]).sum()
        for am, am_j_list in am2j.items()
        for j_idx, j in enumerate(am_j_list)
    )
    return ag_contr + non_ag_contr + ag_man_contr + water_yield_outside_luto_study_area


def calc_water_net_yield_BASE_YR(data: Data) -> np.ndarray:
    """
    Calculate the water net yield for the base year (2010) for all regions.

    Parameters:
    - data: The data object containing the necessary input data.

    Returns:
    - water_net_yield: The water net yield for all regions.
    """
    if data.WATER_YIELD_RR_BASE_YR is not None:
        return data.WATER_YIELD_RR_BASE_YR
    
    # Get the water yield matrices
    w_mrj = get_water_net_yield_matrices(data, 0)
    
    # Get the ag decision variables
    ag_dvar_mrj = data.AG_L_MRJ

    # Multiply the water yield by the decision variables
    ag_w_r = np.einsum('mrj,mrj->r', w_mrj, ag_dvar_mrj)
    
    # Get water net yield for each region
    wny_inside_LUTO_regions = np.bincount(data.RIVREG_ID, ag_w_r)
    wny_inside_LUTO_regions = {region: wny for region, wny in enumerate(wny_inside_LUTO_regions) if region != 0}
    
    # Get water yield from outside the LUTO study area
    wny_outside_LUTO_regions = get_water_outside_luto_study_area_from_hist_level(data)
    
    return {
        region: wny_inside_LUTO_regions[region]  + wny_outside_LUTO_regions[region] 
        for region in wny_outside_LUTO_regions
    } 



def get_water_net_yield_limit_values(
    data: Data,
) -> dict[int, tuple[str, float, np.ndarray]]:
    """
    Return water net yield limits for regions (River Regions or Drainage Divisions as specified in luto.settings.py).

    Parameters:
    - data: The data object containing the necessary input data.

    Returns:
    water_net_yield_limits: A dictionary of tuples containing the water use limits for each region\n
      region index:(
      - region name
      - water yield limit
      - indices of cells in the region
      )

    Raises:
    - None

    """
    # Get the water limit from data to avoid recalculating
    if data.WATER_YIELD_LIMITS is not None:
        return data.WATER_YIELD_LIMITS
    
    # Get historical yields of regions, stored in data.RIVREG_LIMITS and data.DRAINDIV_LIMITS
    if settings.WATER_REGION_DEF == 'River Region':
        wny_region_hist = data.RIVREG_LIMITS
        region_id = data.RIVREG_ID
        region_names = data.RIVREG_DICT

    elif settings.WATER_REGION_DEF == 'Drainage Division':
        wny_region_hist = data.DRAINDIV_LIMITS
        region_id = data.DRAINDIV_ID
        region_names = data.DRAINDIV_DICT

    # Calculate the water yield limits for each region
    limits_by_region = {}
    for region, name in region_names.items():
        hist_yield = wny_region_hist[region]
        ind = np.flatnonzero(region_id == region).astype(np.int32)
        limit_hist_level = hist_yield * settings.WATER_YIELD_TARGET_AG_SHARE    # Water yield limit calculated as a proportial of historical level based on planetary boundary theory
        limits_by_region[region] = (name, limit_hist_level, ind)    

    # Save the results in data to avoid recalculating
    data.WATER_YIELD_LIMITS = limits_by_region
    
    return limits_by_region
    


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
