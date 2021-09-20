#!/bin/env python3
#
# transitions.py - data about transition costs.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-04-30
# Last modified: 2021-09-20
#

import os.path

import numpy as np
import pandas as pd
import numpy_financial as npf

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
    a certain land-use j under a certain land-management. The base costs are
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

    # Areas in hectares for each cell, stacked for each land-use (identical).
    realarea_rj = np.stack((data.REAL_AREA,) * nlus, axis=1).astype(np.float64)

    # Total aq lic costs = aq req [Ml/ha] x area/cell [ha] x lic price [AUD/Ml].
    aqlic_rj = ( (data.WATER_REQUIRED * realarea_rj).T
                * data.WATER_LICENCE_PRICE ).T

    # Switching to irr, from and to an irr land-use, may incur licence costs.
    tdelta_toirr_rj = np.zeros((ncells, nlus))
    for r in range(ncells):
        lu = lumap[r]
        lm = lmmap[r]
        # IRR -> IRR / Difference with current licence paid or refunded.
        if lm == 1:
            tdelta_toirr_rj[r] = aqlic_rj[r] - aqlic_rj[r, lu]
        # DRY -> IRR / Licence cost + infrastructure cost @8kAUD/ha.
        else:
            tdelta_toirr_rj[r] = aqlic_rj[r] + (10**4 * realarea_rj[r])

    # Switching to dry, from and to an irr land-use, may incur licence refund.
    tdelta_todry_rj = np.zeros((ncells, nlus))
    for r in range(ncells):
        lu = lumap[r]
        lm = lmmap[r]
        # IRR -> DRY / Current licence refunded.
        if lm == 1:
            tdelta_todry_rj[r] = - aqlic_rj[r, lu]
        # DRY -> DRY / No additional costs.

    # Transition costs in AUD/ha converted to AUD per cell and amortised.
    t_rj_todry = amortise(t_rj * realarea_rj + tdelta_todry_rj)
    t_rj_toirr = amortise(t_rj * realarea_rj + tdelta_toirr_rj)

    # Stack the t_rj matrices into one t_mrj array.
    t_mrj = np.stack((t_rj_todry, t_rj_toirr))

    return t_mrj
