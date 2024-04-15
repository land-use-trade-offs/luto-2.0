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
Pure functions to calculate water use by lm, lu.
"""


from typing import Dict
import numpy as np

import luto.settings as settings
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.data import Data, lvs_veg_types
from luto.economics.agricultural.quantity import get_yield_pot


def get_wreq_matrices(data: Data, yr_idx):
    """
    Return w_mrj water requirement matrices by land management, cell, and land-use type.
    
    Parameters:
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
    
    Returns:
        numpy.ndarray: The w_mrj <unit: ML/cell> water requirement matrices .
    """
    
    # Stack water requirements data
    w_mrj = np.stack(( data.WREQ_DRY_RJ, data.WREQ_IRR_RJ ))    # <unit: ML/head>
    
    # Covert water requirements units from ML/head to ML/ha
    for j, lu in enumerate(data.AGRICULTURAL_LANDUSES):
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)
            w_mrj[0, :, j] = w_mrj[0, :, j] * get_yield_pot(data, lvs, veg, 'dry', yr_idx)  # Water reqs depend on current stocking rate for drinking water
            w_mrj[1, :, j] = w_mrj[1, :, j] * get_yield_pot(data, lvs, veg, 'irr', 0)       # Water reqs depend on initial stocking rate for irrigation
    
    # Convert to ML per cell via REAL_AREA
    w_mrj *= data.REAL_AREA[:, np.newaxis]                      # <unit: ML/ha> * <unit: ha/cell> -> <unit: ML/cell>
    
    return w_mrj


def get_asparagopsis_effect_w_mrj(data: Data, w_mrj, yr_idx):
    """
    Applies the effects of using asparagopsis to the water requirements data
    for all relevant agr. land uses.

    Args:
        data (object): The data object containing relevant information.
        w_mrj (ndarray, <unit:ML/cell>): The water requirements data for all land uses.
        yr_idx (int): The index of the year.

    Returns:
        ndarray <unit:ML/cell>: The updated water requirements data with the effects of using asparagopsis.

    Notes:
        Asparagopsis taxiformis has no effect on the water required.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_w_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, "Water Impacts"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in quantity produced
            new_w_mrj[:, :, lu_idx] = w_mrj[:, :, j] * (multiplier - 1)

    return new_w_mrj


def get_precision_agriculture_effect_w_mrj(data: Data, w_mrj, yr_idx):
    """
    Applies the effects of using precision agriculture to the water requirements data
    for all relevant agricultural land uses.

    Parameters:
    - data: The data object containing relevant information for the calculation.
    - w_mrj <unit:ML/cell>: The original water requirements data for different land uses.
    - yr_idx: The index representing the year for which the calculation is performed.

    Returns:
    - new_w_mrj <unit:ML/cell>: The updated water requirements data after applying precision agriculture effects.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_w_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each land use
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.PRECISION_AGRICULTURE_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in quantity produced
            new_w_mrj[:, :, lu_idx] = w_mrj[:, :, j] * (multiplier - 1)

    return new_w_mrj


def get_ecological_grazing_effect_w_mrj(data: Data, w_mrj, yr_idx):
    """
    Applies the effects of using ecological grazing to the water requirements data
    for all relevant agricultural land uses.

    Parameters:
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water requirements data for different land uses.
    - yr_idx: The index of the year.

    Returns:
    - new_w_mrj <unit:ML/cell>: The updated water requirements data after applying ecological grazing effects.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_w_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each land use
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, "INPUT-wrt_water-required"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in quantity produced
            new_w_mrj[:, :, lu_idx] = w_mrj[:, :, j] * (multiplier - 1)

    return new_w_mrj


def get_savanna_burning_effect_w_mrj(data):
        """
        Applies the effects of using savanna burning to the water requirements data
        for all relevant agr. land uses.

        Savanna burning does not affect water usage, so return an array of zeros.

        Parameters:
        - data: The input data object containing information about land uses and water requirements.

        Returns:
        - An array of zeros with dimensions (NLMS, NCELLS, nlus), where:
            - NLMS: Number of land management systems
            - NCELLS: Number of cells
            - nlus: Number of land uses affected by savanna burning
        """
        nlus = len(AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning'])
        return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_agtech_ei_effect_w_mrj(data, w_mrj, yr_idx):
    """
    Applies the effects of using AgTech EI to the water requirements data
    for all relevant agr. land uses.

    Parameters:
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water requirements data for all land uses.
    - yr_idx: The index of the year.

    Returns:
    - new_w_mrj <unit:ML/cell>: The updated water requirements data with AgTech EI effects applied.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_w_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.AGTECH_EI_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in quantity produced
            new_w_mrj[:, :, lu_idx] = w_mrj[:, :, j] * (multiplier - 1)

    return new_w_mrj


def get_agricultural_management_water_matrices(data: Data, w_mrj, yr_idx) -> Dict[str, np.ndarray]:
    asparagopsis_data = get_asparagopsis_effect_w_mrj(data, w_mrj, yr_idx)
    precision_agriculture_data = get_precision_agriculture_effect_w_mrj(data, w_mrj, yr_idx)
    eco_grazing_data = get_ecological_grazing_effect_w_mrj(data, w_mrj, yr_idx)
    sav_burning_data = get_savanna_burning_effect_w_mrj(data)
    agtech_ei_data = get_agtech_ei_effect_w_mrj(data, w_mrj, yr_idx)

    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
    }


def get_wuse_limits(data):
    """
    Return water use limits for regions (River Regions or Drainage Divisions as specified in luto.settings.py).
    Currently set such that water limits are set at 2010 agricultural water requirements.

    Parameters:
    - data: The data object containing the necessary input data.

    Returns:
    - wuse_limits: A list of tuples containing the water use limits for each region. Each tuple contains the region ID,
      region name, water use limit, and the indices of cells in the region.

    Raises:
    - None

    """

    # Set up data for river regions or drainage divisions
    if settings.WATER_REGION_DEF == 'River Region':
        region_name = data.RIVREG_DICT
        region_limits = data.RIVREG_LIMITS
        region_id = data.RIVREG_ID
        
    elif settings.WATER_REGION_DEF == 'Drainage Division':
        region_name = data.DRAINDIV_DICT
        region_limits = data.DRAINDIV_LIMITS
        region_id = data.DRAINDIV_ID
        
    else:
        print('Incorrect option for WATER_REGION_DEF in settings')


    # Set up empty list to hold water use limits data
    wuse_limits = []
    
    # Calculate water use limits (ML) depending on which type of limits specified in settings
    if settings.WATER_LIMITS_TYPE == 'pct_ag':
        
        # Get water requirements of 2010 agriculture in ML per cell in mrj format for 2010.
        w_mrj = get_wreq_matrices(data, 0)  # 0 gets water requirements from base year (i.e., 2010)
        
        # Multiply by agricultural land use and management map in mrj format and sum for each grid cell
        w_lim_r = np.sum( w_mrj * data.AG_L_MRJ, axis = (0, 2) )
    
        # Loop through specified water regions
        for region in region_limits.keys():
            
            # Get indices of cells in region
            ind = np.flatnonzero(region_id == region).astype(np.int32)
    
            # Calculate the 2010 water requirements by agriculture for region.
            wuse_reg_limit = np.sum( w_lim_r[ind] )
    
            # Append to list
            wuse_limits.append( (region, region_name[region], wuse_reg_limit, ind) )
            
    
    elif settings.WATER_LIMITS_TYPE == 'water_stress':
        
        # Loop through specified water regions
        for region in region_limits.keys():
            
            # Get indices of cells in region
            ind = np.flatnonzero(region_id == region).astype(np.int32)
    
            # Retrieve the pre-calculated 2010 water use limit (ML) for each region.
            wuse_reg_limit = region_limits[region] * settings.WATER_STRESS_FRACTION   # np.sum( w_lim_r[ind] )
    
            # Append to list
            wuse_limits.append( (region, region_name[region], wuse_reg_limit, ind) )
            

    return wuse_limits




 


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
