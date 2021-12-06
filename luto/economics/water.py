#!/bin/env python3
#
# water.py - pure functions to calculate water use by lm, lu.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-11-15
# Last modified: 2021-12-06
#

import numpy as np

from luto.economics.quantity import get_yield_pot, lvs_veg_types, get_quantity
import luto.settings as settings

def get_aqreq_matrices( data # Data object or module.
                      , year # Number of years post base-year ('annum').
                      ):
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
    aq_req_dry_rj = data.AQ_REQ_CROPS_DRY_RJ + AQ_REQ_LVSTK_DRY_RJ
    aq_req_irr_rj = data.AQ_REQ_CROPS_IRR_RJ + AQ_REQ_LVSTK_IRR_RJ

    # Return as mrj stack.
    return np.stack((aq_req_dry_rj, aq_req_irr_rj))

def mask_aqrec_matrices(data, year, mask):
    """Return masked version of get_aqrec_matrices."""
    return mask[np.newaxis, :, np.newaxis] * get_aqrec_matrices(data, year)

def get_aqyld_matrix( data # Data object or module.
                    , year # Number of years post base-year ('annum').
                    ):
    """Return an rj matrix of the water yields, per cell, by land use."""
    cols = tuple(      data.WATER_YIELD_NUNC_DR  if 'natural' in lu
                  else data.WATER_YIELD_NUNC_SR
                  for lu in data.LANDUSES )
    return np.stack(cols, axis=1)

def get_water_stress(data, year, mask=None):
    """Return, by cell, how much the water yield is above the stress level."""
    # Get the water yields -- disregarding irrigation but as mrj matrix.
    aqyld_rj = get_aqyld_matrix(data, year)
    aqyld_mrj = np.stack((aqyld_rj, aqyld_rj))

    # Get the water requirements for irrigation and livestock drinking water.
    aqreq_mrj = get_aqreq_matrices(data, year)

    # Calculate the water stress threshold.
    stresshold = ( (1 - settings.WATER_YIELD_STRESS_FRACTION)
                 * data.WATER_YIELD_BASE_DR[:, np.newaxis] )

    # Net water yield is yield less requirements (use) less base level yields.
    water_stress = ( aqyld_mrj # Water yields, dependent on land use.
                   - aqreq_mrj # Water requirements for irr. and livestock.
                   - stresshold ) # Yields below this level mean stress.

    # Apply a mask if provided -- e.g. for catchment specific stress.
    if mask is not None:
        water_stress *= mask[:, np.newaxis]

    return water_stress

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
