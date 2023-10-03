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
from luto.economics.agricultural.quantity import get_yield_pot, lvs_veg_types
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


def get_wreq_matrices(data, yr_idx):
    """Return w_mrj water requirement matrices by land management, cell, and land-use type."""
    
    # Stack water requirements data
    w_mrj = np.stack(( data.WREQ_DRY_RJ, data.WREQ_IRR_RJ ))
    
    # Covert water requirements units from ML/head to ML/ha
    for j, lu in enumerate(data.AGRICULTURAL_LANDUSES):
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)
            w_mrj[0, :, j] = w_mrj[0, :, j] * get_yield_pot(data, lvs, veg, 'dry', yr_idx) # Water reqs depend on current stocking rate for drinking water
            w_mrj[1, :, j] = w_mrj[1, :, j] * get_yield_pot(data, lvs, veg, 'irr', 0)        # Water reqs depend on initial stocking rate for irrigation
    
    # Convert to ML per cell via REAL_AREA
    w_mrj *= data.REAL_AREA[:, np.newaxis]
    
    return w_mrj


def get_asparagopsis_effect_w_mrj(data, w_mrj, yr_idx):
    """
    Applies the effects of using asparagopsis to the water requirements data
    for all relevant agr. land uses.

    Asparagopsis taxiformis has no effect on the water required.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    year = 2010 + yr_idx

    # Set up the effects matrix
    new_w_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        j = lu_codes[lu_idx]
        multiplier = data.ASPARAGOPSIS_DATA[lu].loc[year, "Water_use"]
        if multiplier != 1:
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in quantity produced
            new_w_mrj[:, :, lu_idx] = w_mrj[:, :, j] * (multiplier - 1)

    return new_w_mrj


def get_precision_agriculture_effect_w_mrj(data, w_mrj, yr_idx):
    """
    Applies the effects of using precision agriculture to the water requirements data
    for all relevant agr. land uses.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    year = 2010 + yr_idx

    # Set up the effects matrix
    new_w_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        j = lu_codes[lu_idx]
        multiplier = data.PRECISION_AGRICULTURE_DATA[lu].loc[year, "Water_use"]
        if multiplier != 1:
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in quantity produced
            new_w_mrj[:, :, lu_idx] = w_mrj[:, :, j] * (multiplier - 1)

    return new_w_mrj


def get_ecological_grazing_effect_w_mrj(data, w_mrj, yr_idx):
    """
    Applies the effects of using ecological grazing to the water requirements data
    for all relevant agr. land uses.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    year = 2010 + yr_idx

    # Set up the effects matrix
    new_w_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        j = lu_codes[lu_idx]
        multiplier = data.ECOLOGICAL_GRAZING_DATA[lu].loc[year, "INPUT-wrt_water-required"]
        if multiplier != 1:
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in quantity produced
            new_w_mrj[:, :, lu_idx] = w_mrj[:, :, j] * (multiplier - 1)

    return new_w_mrj



def get_agricultural_management_water_matrices(data, w_mrj, yr_idx) -> Dict[str, np.ndarray]:
    asparagopsis_data = get_asparagopsis_effect_w_mrj(data, w_mrj, yr_idx)
    precision_agriculture_data = get_precision_agriculture_effect_w_mrj(data, w_mrj, yr_idx)
    eco_grazing_data = get_ecological_grazing_effect_w_mrj(data, w_mrj, yr_idx)

    ag_management_data = {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
    }

    return ag_management_data


def get_wuse_limits(data):
    """Return water use limits for regions (River Regions or Drainage Divisions as specified in settings.py.
       Currently set such that water limits are set at 2010 agricultural water requirements.
    """

    # Set up data for river regions or drainage divisions
    if settings.WATER_REGION_DEF == 'RR':
        regions = settings.WATER_RIVREGS
        region_id = data.RIVREG_ID
        
    elif settings.WATER_REGION_DEF == 'DD':
        regions = settings.WATER_DRAINDIVS
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
    
    elif settings.WATER_LIMITS_TYPE == 'water_stress':
        
        # Calculate water use limits as water yield under deep-rooted plants (i.e., native vegetation) x water stress percentage
        w_lim_r = data.WATER_YIELD_BASE_DR * settings.WATER_STRESS_FRACTION
        
        w_lim_r *= data.REAL_AREA
    
    
    # Loop through specified water regions
    for region in regions:
        
        # Get indices of cells in region
        ind = np.flatnonzero(region_id == region).astype(np.int32)

        # Calculate the 2010 water requiremnents by agriculture for region.
        wuse_reg_limit = np.sum( w_lim_r[ind] )

        # Append to list
        wuse_limits.append( (region, wuse_reg_limit, ind) )

    return wuse_limits





# def get_wyld_matrix( data # Data object or module.
#                     , yr_idx = None # Number of years post base-year ('YR_CAL_BASE').
#                     ):
#     """Return an rj matrix of the water yields, per cell, by land use."""

#     # If no year is provided, use the base yields of 1985.
#     if yr_idx is None:
#         yld_dr = data.WATER_YIELD_BASE_DR
#         yld_sr = data.WATER_YIELD_BASE_SR
#     else:
#         yld_dr = data.WATER_YIELD_NUNC_DR
#         yld_sr = data.WATER_YIELD_NUNC_SR

#     # Select the appropriate root depth for each land use.
#     cols = tuple(      yld_dr if 'natural' in lu
#                   else yld_sr
#                   for lu in data.LANDUSES )

#     # Stack the columns and convert from per ha to per cell.
#     return np.stack(cols, axis=1) * data.REAL_AREA[:, np.newaxis]

 


"""
Water logic

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
