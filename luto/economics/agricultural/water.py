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
Pure functions to calculate water net yield by lm, lu and water limits.
"""

import numpy as np
import pandas as pd
import luto.settings as settings

from typing import Optional
from luto import tools
from luto.economics.agricultural.quantity import get_yield_pot, lvs_veg_types


def get_wreq_matrices(data, yr_idx):
    """
    Return water requirement (water use by irrigation and livestock drinking water) matrices
    by land management, cell, and land-use type.

    Parameters
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.

    Returns
        numpy.ndarray: The w_mrj <unit: ML/cell> water requirement matrices, indexed (m, r, j).
    """

    # Stack water requirements data
    w_req_mrj = np.stack((data.WREQ_DRY_RJ, data.WREQ_IRR_RJ))    # <unit: ML/head|ha>

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
    data, 
    yr_idx:int, 
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return water yield matrices for YR_CAL_BASE (2010) by land management, cell, and land-use type.

    Parameters
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
        water_dr_yield (ndarray, <unit:ML/cell>): The water yield for deep-rooted vegetation.
        water_sr_yield (ndarray, <unit:ML/cell>): The water yield for shallow-rooted vegetation.

    Returns
        numpy.ndarray: The w_mrj <unit: ML/cell> water yield matrices, indexed (m, r, j).
    """
    w_yield_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)

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
    data, 
    yr_idx, 
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Return water net yield matrices by land management, cell, and land-use type.
    The resulting array is used as the net yield w_mrj array in the input data of the solver.

    Parameters
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
        water_dr_yield (ndarray, <unit:ML/cell>): The water yield for deep-rooted vegetation.
        water_sr_yield (ndarray, <unit:ML/cell>): The water yield for shallow-rooted vegetation.
        
    Notes:
        Provides the `water_dr_yield` or `water_sr_yield` will make the `yr_idx` useless because
        the water net yield will be calculated based on the provided water yield data regardless of the year.

    Returns
        numpy.ndarray: The w_mrj <unit: ML/cell> water net yield matrices, indexed (m, r, j).
    """
    return get_wyield_matrices(data, yr_idx, water_dr_yield, water_sr_yield) - get_wreq_matrices(data, yr_idx)


def get_asparagopsis_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using asparagopsis to the water net yield data
    for all relevant agr. land uses.

    Args:
        data (object): The data object containing relevant information.
        w_mrj (ndarray, <unit:ML/cell>): The water net yield data for all land uses.
        yr_idx (int): The index of the year.

    Returns
        ndarray <unit:ML/cell>: The updated water net yield data with the effects of using asparagopsis.

    Notes:
        Asparagopsis taxiformis has no effect on the water required.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in water used.
            # Since the effect applies to water use, it effects the net yield negatively.
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1 - multiplier)

    return w_mrj_effect


def get_precision_agriculture_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using precision agriculture to the water net yield data
    for all relevant agricultural land uses.

    Parameters
    - data: The data object containing relevant information for the calculation.
    - w_mrj <unit:ML/cell>: The original water net yield data for different land uses.
    - yr_idx: The index representing the year for which the calculation is performed.

    Returns
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data after applying precision agriculture effects.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
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


def get_ecological_grazing_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using ecological grazing to the water net yield data
    for all relevant agricultural land uses.

    Parameters
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for different land uses.
    - yr_idx: The index of the year.

    Returns
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data after applying ecological grazing effects.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
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

    Parameters
    - data: The input data object containing information about land uses and water net yield.

    Returns
    - An array of zeros with dimensions (NLMS, NCELLS, nlus), where:
        - NLMS: Number of land management systems
        - NCELLS: Number of cells
        - nlus: Number of land uses affected by savanna burning
    """

    nlus = len(settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning'])
    return np.zeros((data.NLMS, data.NCELLS, nlus)).astype(np.float32)


def get_agtech_ei_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using AgTech EI to the water net yield data
    for all relevant agr. land uses.

    Parameters
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data with AgTech EI effects applied.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
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


def get_biochar_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using Biochar to the water net yield data
    for all relevant agr. land uses.

    Parameters
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data with Biochar applied.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']
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


def get_beef_hir_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using HIR to the water net yield data
    for the natural beef land use.

    Parameters:
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns:
    - w_mrj_effects <unit:ML/cell>: The updated water net yield data with Biochar applied.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']

    w_mrj_effects = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)
    base_w_req_mrj = np.stack(( data.WREQ_DRY_RJ, data.WREQ_IRR_RJ ))

    for lu_idx, lu in enumerate(land_uses):
        j = data.DESC2AGLU[lu]

        multiplier = settings.HIR_PRODUCTIVITY_CONTRIBUTION - 1 # Negative value, indicating reduction in water requirements

        # Reduce water requirements due to drop in yield potential (increase in net yield)
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)

            w_mrj_effects[0, :, lu_idx] = (
                multiplier * base_w_req_mrj[0, :, j] * get_yield_pot(data, lvs, veg, 'dry', yr_idx)
            )
            w_mrj_effects[1, :, lu_idx] = (
                multiplier * base_w_req_mrj[1, :, j] * get_yield_pot(data, lvs, veg, 'irr', 0)
            )

    return w_mrj_effects


def get_sheep_hir_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using HIR to the water net yield data
    for the natural sheep land use.

    Parameters:
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns:
    - w_mrj_effects <unit:ML/cell>: The updated water net yield data with Biochar applied.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']

    w_mrj_effects = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)
    base_w_req_mrj = np.stack(( data.WREQ_DRY_RJ, data.WREQ_IRR_RJ ))

    for lu_idx, lu in enumerate(land_uses):
        j = data.DESC2AGLU[lu]

        multiplier = settings.HIR_PRODUCTIVITY_CONTRIBUTION - 1 # Negative value, indicating reduction in water requirements

        # Reduce water requirements due to drop in yield potential (increase in net yield)
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)

            w_mrj_effects[0, :, lu_idx] = (
                multiplier * base_w_req_mrj[0, :, j] * get_yield_pot(data, lvs, veg, 'dry', yr_idx)
            )
            w_mrj_effects[1, :, lu_idx] = (
                multiplier * base_w_req_mrj[1, :, j] * get_yield_pot(data, lvs, veg, 'irr', 0)
            )

        # TODO - potential effects of increased water yield of HIR areas?

    return w_mrj_effects

def get_utility_solar_pv_effect_w_mrj(data, yr_idx):
    """
    Gets water use impacts of using Utility Solar PV

    Parameters
    - data: The input data object containing information about NLMS and NCELLS.

    Returns
    - new_b_mrj: A numpy array representing the water impacts of using Utility Solar PV.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    wreq_mrj = get_wreq_matrices(data, yr_idx)
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32) 
    
    if not settings.AG_MANAGEMENTS['Utility Solar PV']:
        return w_mrj_effect

    for lu_idx, lu in enumerate(land_uses):
        water_impact = data.RENEWABLE_BUNDLE_SOLAR.query('Year == @yr_cal and Commodity == @lu')['INPUT-wrt_water-required'].item()
        if water_impact != 1:
            j = lu_codes[lu_idx]
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (water_impact - 1)
    return w_mrj_effect

def get_onshore_wind_effect_w_mrj(data, yr_idx):
    """
    Gets water use impacts of using Onshore Wind

    Parameters
    - data: The input data object containing information about NLMS and NCELLS.

    Returns
    - new_b_mrj: A numpy array representing the water impacts of using Onshore Wind.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx
    
    # Set up the effects matrix
    wreq_mrj = get_wreq_matrices(data, yr_idx)
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)  
     
    if not settings.AG_MANAGEMENTS['Onshore Wind']:
        return w_mrj_effect
    
    for lu_idx, lu in enumerate(land_uses):
        water_impact = data.RENEWABLE_BUNDLE_WIND.query('Year == @yr_cal and Commodity == @lu')['INPUT-wrt_water-required'].item()
        if water_impact != 1:
            j = lu_codes[lu_idx]
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (water_impact - 1)
    return w_mrj_effect

def get_agricultural_management_water_matrices(data, yr_idx) -> dict[str, np.ndarray]:
    
    ag_mam_w_mrj ={}
    
    ag_mam_w_mrj['Asparagopsis taxiformis'] = get_asparagopsis_effect_w_mrj(data, yr_idx)           
    ag_mam_w_mrj['Precision Agriculture'] = get_precision_agriculture_effect_w_mrj(data, yr_idx)    
    ag_mam_w_mrj['Ecological Grazing'] = get_ecological_grazing_effect_w_mrj(data, yr_idx)          
    ag_mam_w_mrj['Savanna Burning'] = get_savanna_burning_effect_w_mrj(data)                        
    ag_mam_w_mrj['AgTech EI'] = get_agtech_ei_effect_w_mrj(data, yr_idx)                            
    ag_mam_w_mrj['Biochar'] = get_biochar_effect_w_mrj(data, yr_idx)                                
    ag_mam_w_mrj['HIR - Beef'] = get_beef_hir_effect_w_mrj(data, yr_idx)                            
    ag_mam_w_mrj['HIR - Sheep'] = get_sheep_hir_effect_w_mrj(data, yr_idx)             
    ag_mam_w_mrj['Utility Solar PV'] = get_utility_solar_pv_effect_w_mrj(data, yr_idx)
    ag_mam_w_mrj['Onshore Wind'] = get_onshore_wind_effect_w_mrj(data, yr_idx)             

    return ag_mam_w_mrj




def get_climate_change_impact_whole_region(data, yr_cal):
    '''
    Calculate the climate change impact on water yield change for Ag-land and Outside-LUTO regions.

    Note, the climate change impact on water here is calculated assuming the land-use remains constant since the starting year.

    Parameters
    ----------
    data : object
        The data object containing the necessary input data.
    yr_cal : int
        The calendar year for which to calculate the climate change impact.

    Returns
    -------
    dict
        A dictionary containing the climate change impact on water yield change for each region.
    '''

    yr_idx = yr_cal - data.YR_CAL_BASE
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.AG_L_MRJ).assign_coords(region_id=('cell', data.WATER_REGION_ID))

    # Get water yield change led by climate change
    ag_w_mrj_CCI = get_water_net_yield_matrices(data, yr_idx) -  get_water_net_yield_matrices(data, 0)
    wny_outside_luto_study_area_CCI = (
        np.array(list(data.WATER_OUTSIDE_LUTO_BY_CCI.loc[yr_cal].to_dict().values())) 
        - np.array(list(data.WATER_OUTSIDE_LUTO_BY_CCI.loc[data.YR_CAL_BASE].to_dict().values()))
    )

    CCI_impact = (
            (ag_w_mrj_CCI * ag_dvar_mrj).groupby('region_id').sum(['cell','lm', 'lu']) 
            + wny_outside_luto_study_area_CCI
        ).to_dataframe('Delta (ML)'
        ).reset_index(
        ).assign(name=lambda x: x['region_id'].map(data.WATER_REGION_NAMES), year=yr_cal)

    return CCI_impact


def get_water_delta_by_extreme_CCI_for_whole_region(data):
    """
    Get the extreme climate change impact on water yield change for the whole region.
    """
    water_delta_extreme_by_CCI = pd.DataFrame()
    for year in sorted(settings.SIM_YEARS):
        water_delta_extreme_by_CCI = pd.concat([
            water_delta_extreme_by_CCI,
            get_climate_change_impact_whole_region(data, year) 
        ])
        
    return water_delta_extreme_by_CCI.groupby('region_id')['Delta (ML)'].agg('min').to_dict()


def get_wny_inside_LUTO_by_CCI_for_base_yr(data):
    """
    Return water net yield for watershed regions at the BASE_YR.

    Parameters
        data (object): The data object containing the required data.

    Returns
        dict[int, float]: A dictionary with the following structure:
        - key: region ID
        - value: water net yield for this region (ML)
    """
    wny_inside_mrj = get_water_net_yield_matrices(data, 0)
    wny_base_yr_inside_r = np.einsum('mrj,mrj->r', wny_inside_mrj, data.AG_L_MRJ)
    wny_base_yr_inside_regions = {k:v for k,v in enumerate(np.bincount(data.WATER_REGION_ID, wny_base_yr_inside_r))}

    return wny_base_yr_inside_regions


def get_water_target_inside_LUTO_by_CCI(data):
    """
    Calculate the water net yield limit for each region based on historical levels.
    
    The limit is calculated as:
        - Water net yield limit for the whole region
        - Plus domestic water use
        - Minus water net yield from outside the LUTO study area
        
    Parameters
        data (object): The data object containing the required data.
        
    Returns
        dict[int, float]: A dictionary with the following structure:
        - key: region ID
        - value: water net yield limit for this region (ML)
    """

    wny_base_yr_outside_LUTO = data.WATER_OUTSIDE_LUTO_BY_CCI.loc[data.YR_CAL_BASE].to_dict()
    wny_base_yr_inside_LUTO = get_wny_inside_LUTO_by_CCI_for_base_yr(data)
    wny_extreme_delta = get_water_delta_by_extreme_CCI_for_whole_region(data)
    wreq_domestic = data.WATER_USE_DOMESTIC
    
    # Get inside LUTO targets based on historical level
    wny_inside_LUTO_targets = {}
    wny_relaxed_region_raw_targets = {}
    for reg_idx, hist_level in data.WATER_REGION_HIST_LEVEL.items():
        wny_inside_LUTO = wny_base_yr_inside_LUTO[reg_idx]
        wny_outside_LUTO = wny_base_yr_outside_LUTO[reg_idx]
        wreq_domestic = data.WATER_USE_DOMESTIC[reg_idx]    # positive values, indicating water requirements for domestic and industrial use
        CCI_extreme_stress = wny_extreme_delta[reg_idx]     # negative values, indicating water yield reductions by climate change
        
        wny_extreme_CCI = wny_inside_LUTO + wny_outside_LUTO - wreq_domestic + CCI_extreme_stress
        wny_hist_target = hist_level * settings.WATER_STRESS

        if wny_extreme_CCI < wny_hist_target:
            print(
                f"│   ├── Target ({settings.WATER_REGION_DEF}) relaxed to ({wny_extreme_CCI:12,.0f} ML) from ({wny_hist_target:12,.0f} ML) for {data.WATER_REGION_NAMES[reg_idx]}", flush=True
            )
            wny_inside_LUTO_targets[reg_idx] = wny_extreme_CCI - wny_outside_LUTO
            wny_relaxed_region_raw_targets[reg_idx] = wny_hist_target
        else:
            wny_inside_LUTO_targets[reg_idx] = wny_hist_target - wny_outside_LUTO

    return wny_inside_LUTO_targets, wny_relaxed_region_raw_targets


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
