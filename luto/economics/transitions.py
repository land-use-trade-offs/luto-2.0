#!/bin/env python3
#
# transitions.py - data about transition costs.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-04-30
# Last modified: 2021-10-06
#

import os.path

import numpy as np
import pandas as pd
import numpy_financial as npf

from luto.economics.quantity import lvs_veg_types, get_yield_pot

def amortise(cost, rate=0.05, horizon=30):
    """Return amortised `cost` at `rate`interest over `horizon` years."""
    return -1 * npf.pmt(rate, horizon, pv=cost, fv=0, when='begin')

def get_exclude_matrices(data, lumap):
    """Return x_mrj exclude matrices."""
    # To be excluded based on SA2 data.
    x_sa2 = data.EXCLUDE

    # Raw transition-cost matrix is in AUD/Ha and lexigraphically ordered.
    t_ij = data.TMATRIX

    # Infer number of cells from lumap array.
    ncells = lumap.shape[0]

    # Transition costs to commodity j at cell r using present lumap (in AUD/ha).
    t_rj = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))

    # To be excluded based on disallowed switches.
    x_trn = np.where(np.isnan(t_rj), 0, 1)

    # Overal exclusion as elementwise, logical `and` of the exclude matrices.
    return x_sa2 * x_trn

def get_transition_matrices(data, year, lumap, lmmap):
    """Return t_mrj transition-cost matrices.

    A transition-cost matrix gives the cost of switching a certain cell r to
    a certain land-use j under a certain land-management m. The base costs are
    taken from the raw transition costs in the `data` module and additional
    costs are added depending on the land-management (e.g. costs of irrigation
    infrastructure). Switching costs further depend on both the current and the
    future land-use, so the present land-use map is needed.

    Parameters
    ----------

    data: object/module
        Data object or module with fields like in `luto.data`.
    year : int
        Number of years from base year, counting from zero.
    lumap : numpy.ndarray
        Present land-use map, i.e. highpos (shape=ncells, dtype=int).
    lmmap : numpy.ndarray
        Present land-management map (shape=ncells, dtype=int).

    Returns
    -------

    numpy.ndarray
        t_mrj transition-cost matrices. The m-slices correspond to the
        different land-management versions of the land-use `j` to switch _to_.
        With m==0 conventional dry-land, m==1 conventional irrigated.
    """

    # Raw transition-cost matrix is in AUD/Ha and lexigraphically ordered.
    t_ij = data.TMATRIX

    # Infer number land-uses and cells from t_ij and lumap matrices.
    nlus = t_ij.shape[0]
    ncells = lumap.shape[0]

    # Transition costs to commodity j at cell r using present lumap (in AUD/ha).
    t_rj = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))

    # Convert water requirements for LVSTK from per head to per hectare.
    AQ_REQ_LVSTK_DRY_RJ = data.AQ_REQ_LVSTK_DRY_RJ.copy()
    AQ_REQ_LVSTK_IRR_RJ = data.AQ_REQ_LVSTK_IRR_RJ.copy()
    for lu in data.LANDUSES:
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)
            j = data.LANDUSES.index(lu)
            AQ_REQ_LVSTK_DRY_RJ[:, j] *= get_yield_pot(data, lvs, veg, 'dry')
            AQ_REQ_LVSTK_IRR_RJ[:, j] *= get_yield_pot(data, lvs, veg, 'irr')

    # Switching to irr, from dry or irr land-use, may incur licence cost/refund.
    tdelta_toirr_rj = np.zeros((ncells, nlus))
    for r in range(ncells):
        lu = lumap[r]
        lm = lmmap[r]
        # IRR -> IRR / Difference with current licence paid or refunded.
        if lm == 1:
            # Net water requirements.
            aq_req_net = ( data.AQ_REQ_CROPS_IRR_RJ[r]
                         +      AQ_REQ_LVSTK_IRR_RJ[r]
                         - data.AQ_REQ_CROPS_IRR_RJ[r, lu]
                         -      AQ_REQ_LVSTK_IRR_RJ[r, lu] )
            # To pay: net water requirements x licence price.
            tdelta_toirr_rj[r] = aq_req_net * data.WATER_LICENCE_PRICE[r]

        # DRY -> IRR / Licence difference + infrastructure cost @10kAUD/ha.
        elif lm == 0:
            # Net water requirements.
            aq_req_net = ( data.AQ_REQ_CROPS_IRR_RJ[r]
                         +      AQ_REQ_LVSTK_IRR_RJ[r]
                         - data.AQ_REQ_CROPS_DRY_RJ[r, lu]
                         -      AQ_REQ_LVSTK_DRY_RJ[r, lu] )
            # To pay: net water requirements x licence price and 10kAUD.
            tdelta_toirr_rj[r] = aq_req_net * data.WATER_LICENCE_PRICE[r] + 10E3

        # ___ -> IRR / This case does not (yet) exist.
        else:
            raise ValueError("Unknown land management: %s." % lm)

    # Switching to dry, from dry or irr land-use, may incur licence cost/refund.
    tdelta_todry_rj = np.zeros((ncells, nlus))
    for r in range(ncells):
        lu = lumap[r]
        lm = lmmap[r]
        # IRR -> DRY / Licence difference plus additional costs at @3kAUD/ha.
        if lm == 1:
            # Net water requirements.
            aq_req_net = ( data.AQ_REQ_CROPS_DRY_RJ[r]
                         +      AQ_REQ_LVSTK_DRY_RJ[r]
                         - data.AQ_REQ_CROPS_IRR_RJ[r, lu]
                         -      AQ_REQ_LVSTK_IRR_RJ[r, lu] )
            # To pay: net water requirements x licence price and 3000.
            tdelta_todry_rj[r] = aq_req_net * data.WATER_LICENCE_PRICE[r] + 3000

        # DRY -> DRY / Licence difference costs.
        elif lm == 0:
            # Net water requirements.
            aq_req_net = ( data.AQ_REQ_CROPS_DRY_RJ[r]
                         +      AQ_REQ_LVSTK_DRY_RJ[r]
                         - data.AQ_REQ_CROPS_DRY_RJ[r, lu]
                         -      AQ_REQ_LVSTK_DRY_RJ[r, lu] )
            # To pay: net water requirements x licence price.
            tdelta_todry_rj[r] = aq_req_net * data.WATER_LICENCE_PRICE[r]

        # ___ -> IRR / This case does not (yet) exist.
        else:
            raise ValueError("Unknown land management: %s." % lm)

    # Transition costs are base + water costs and converted AUD/ha -> AUD/cell.
    t_rj_todry = (t_rj + tdelta_todry_rj) * data.REAL_AREA[:, np.newaxis]
    t_rj_toirr = (t_rj + tdelta_toirr_rj) * data.REAL_AREA[:, np.newaxis]

    # Transition costs are amortised.
    t_rj_todry = amortise(t_rj_todry)
    t_rj_toirr = amortise(t_rj_toirr)

    # Stack the t_rj matrices into one t_mrj array.
    t_mrj = np.stack((t_rj_todry, t_rj_toirr))

    return t_mrj
