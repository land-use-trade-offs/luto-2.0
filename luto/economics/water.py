#!/bin/env python3
#
# water.py - pure functions to calculate water use by lm, lu.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-11-15
# Last modified: 2021-11-15
#

import numpy as np

from luto.economics.quantity import get_yield_pot, lvs_veg_types, get_quantity

def get_aqreq_matrices( data # Data object or module.
                      , year # Number of years post base-year ('annum').
                      ):
    """Return w_mrj water requirement matrices by lm, cell and lu."""

    # This function returns water taken from catchments' environmental flows.
    # The total volume consists of:
    #     1. Water used for irrigation or as livestock drinking water, and
    #     2. Water retained in the soil by vegetation.
    # The former is calculated using the water requirements (WR) data, while
    # the latter comes from the water yields per hectare of land with either
    # deeply or shallowly rooted vegetation. The former is the baseline -- under
    # the interpretation that pre-European vegetation is deeply rooted.

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




