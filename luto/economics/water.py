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


import numpy as np
import pandas as pd

from luto.economics.quantity import get_yield_pot, lvs_veg_types, get_quantity
import luto.settings as settings
from luto.tools import lumap2x_mrj


def get_aqreq_matrices( data ): # Data object or module.
    """Return w_mrj water requirement matrices by lm, cell and lu."""

    # Convert water requirements for LVSTK from per head to per hectare.
    AQ_REQ_LVSTK_DRY_RJ = data.AQ_REQ_LVSTK_DRY_RJ.copy()
    AQ_REQ_LVSTK_IRR_RJ = data.AQ_REQ_LVSTK_IRR_RJ.copy()
    for lu in data.LANDUSES:
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)
            j = data.LANDUSES.index(lu)
            AQ_REQ_LVSTK_DRY_RJ[:, j] *= get_yield_pot(data, lvs, veg, 'dry')
            AQ_REQ_LVSTK_IRR_RJ[:, j] *= get_yield_pot(data, lvs, veg, 'irr')

    # Each array has zeroes where the other has entries.
    # aq_req_dry_rj = ( np.nan_to_num(data.AQ_REQ_CROPS_DRY_RJ)
    #                 + np.nan_to_num(AQ_REQ_LVSTK_DRY_RJ) )
    aq_req_dry_rj = np.nan_to_num(AQ_REQ_LVSTK_DRY_RJ)
    aq_req_irr_rj = ( np.nan_to_num(data.AQ_REQ_CROPS_IRR_RJ)
                    + np.nan_to_num(AQ_REQ_LVSTK_IRR_RJ) )

    # Turn Ml/ha into Ml/cell.
    aq_req_dry_rj *= data.REAL_AREA[:, np.newaxis]
    aq_req_irr_rj *= data.REAL_AREA[:, np.newaxis]

    # Return as mrj stack.
    aq_req_mrj = np.stack((aq_req_dry_rj, aq_req_irr_rj))

    return aq_req_mrj


# def get_aqyld_matrix( data # Data object or module.
#                     , year = None # Number of years post base-year ('annum').
#                     ):
#     """Return an rj matrix of the water yields, per cell, by land use."""

#     # If no year is provided, use the base yields of 1985.
#     if year is None:
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


# def get_aqyld_matrices(data, year, mask=None):
#     """Return masked version of `get_aqyld_matrices()`."""
#     aqyld_rj = get_aqyld_matrix(data, year)
#     aqyld_mrj = np.stack((aqyld_rj, aqyld_rj))
#     if mask is None:
#         return aqyld_mrj
#     else:
#         return mask[:, np.newaxis] * aqyld_mrj



def get_aqreq_limits(data):
    """Return aqreq_mrj and water use targets for regions in settings.WATER_DRAINDIVS."""

    # Get the 2010 lumap + lmmap in decision variable format.
    X_mrj = lumap2x_mrj(data.LUMAP, data.LMMAP)

    aqreq_mrj = get_aqreq_matrices(data)

    aqreq_limits = []

    # Loop through specified drainage divisions
    for region in settings.WATER_DRAINDIVS:

        # Get indices of cells in region
        ind = np.flatnonzero(data.DRAINDIV_ID == region).astype(np.int32)

        # Calculate the 2010 water requiremnents by agriculture for region.
        aqreq_reg_limit = (aqreq_mrj[:, ind, :] * X_mrj[:, ind, :]).sum()

        # Append to list
        aqreq_limits.append((region, aqreq_reg_limit, ind))

    return aqreq_mrj, aqreq_limits


def get_target():

    # aquse_limits = [] # A list of water use limits by drainage division.
    # for region in np.unique(data.DRAINDIV_ID):  # 7 == MDB
    #     mask = np.where(data.DRAINDIV_ID == region, True, False)[data.mindices]
    #     basefrac = get_water_stress_basefrac(data, mask)
    #     stress = get_water_stress(data, target_index, mask)
    #     stresses.append((basefrac, stress))

    region = 7
    ddiv_ind = np.flatnonzero(data.DRAINDIV_ID == region).astype(np.int32)

    return ddiv_ind

# def get_water_stress(data, year, mask=None):
#     """Return tuple of (use, yld) for region `mask` in `year`."""

#     # Get the use and yield, ready for multiplication by X_mrj and summing.
#     use_year = get_aqreq_matrices(data, year, mask)
#     yld_year = get_aqyld_matrices(data, year, mask)

#     # Return the tuple.
#     return use_year, yld_year



# def _get_water_stress(data, year, mask=None):
#     """Return, by cell, how much the water yield is above the stress level."""
#     # Get the water yields -- disregarding irrigation but as mrj matrix.
#     aqyld_rj = get_aqyld_matrix(data, year)
#     aqyld_mrj = np.stack((aqyld_rj, aqyld_rj))

#     # Get the water requirements for irrigation and livestock drinking water.
#     aqreq_mrj = get_aqreq_matrices(data, year)

#     # Calculate the water stress threshold.
#     stresshold = ( settings.WATER_YIELD_STRESS_FRACTION
#                  * data.WATER_YIELD_BASE_DR[:, np.newaxis] )

#     # Net water yield is yield less requirements (use) less base level yields.
#     water_stress = ( aqyld_mrj # Water yields, dependent on land use.
#                    - aqreq_mrj # Water requirements for irr. and livestock.
#                    - stresshold ) # Yields below this level mean stress.

#     # Apply a mask if provided -- e.g. for catchment specific stress.
#     if mask is not None:
#         water_stress *= mask[:, np.newaxis]

#     return water_stress



# def get_water_stress_basefrac(data, mask=None):
#     """Return use / yld base fraction for region `mask`."""

#     # Get the 2010 lumap+lmmap in decision var format.
#     X_mrj = lumap2x_mrj(data.LUMAP, data.LMMAP)

#     # Calculate the 2010 use and the 1985 (pre-European proxy) yield.
#     use_base = (get_aqreq_matrices(data, 0, mask) * X_mrj).sum()
#     if mask is None:
#         yld_base = ( data.WATER_YIELD_BASE_DR
#                    * data.REAL_AREA
#                    ).sum()
#     else:
#         yld_base = ( data.WATER_YIELD_BASE_DR
#                    * data.REAL_AREA
#                    * mask
#                    ).sum()

#     # Return the water stress as the fraction of use over yield.
#     return use_base / yld_base




def get_water_totals(data, year, lumap, lmmap):
    """Return a data frame with water use totals."""

    # Get land-use and land management maps from sim object and slice to ag cells
    lumap = lumap[data.mindices]
    lmmap = lmmap[data.mindices]

    # Get the lumap+lmmap in decision variable format.
    X_mrj = lumap2x_mrj(lumap, lmmap)

    # Get 2010 water requirement in mrj format
    aqreq_mrj = get_aqreq_matrices(data)

    # Prepare a data frame.
    df = pd.DataFrame( columns=[ 'HR_DRAINDIV_NAME'
                               , 'TOT_WATER_REQ_YEAR_' + str(year) + '_ML' ] )

    # Loop, calculate and collect in data frame.
    for div in settings.WATER_DRAINDIVS:

        # Get indices of cells in region
        ind = np.flatnonzero(data.DRAINDIV_ID == div).astype(np.int32)

        # Calculate the water requirements by agriculture for region.
        aqreq_reg_limit = (aqreq_mrj[:, ind, :] * X_mrj[:, ind, :]).sum()

        df.loc[div] = ( data.DRAINDIV_DICT[div], aqreq_reg_limit)

    return df

"""
Water logic.

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
