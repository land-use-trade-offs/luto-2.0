#!/bin/env python3
#
# transitions.py - data about transition costs.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-04-30
# Last modified: 2021-06-04
#

import os.path

import numpy as np
import pandas as pd
import numpy_financial as npf

import luto.data as data

def amortise(dollars, year):
    """Return amortised `dollars` for `year`. Interest 10% over 100 years."""
    return -1 * npf.pmt(0.05, 100, pv=dollars, fv=0, when='begin')

def get_transition_matrix(year, lumap):
    """Return t_rj transition-cost matrix for `year` and `lumap` in AUD/cell."""

    # Raw transition-cost matrix is in AUD/Ha. Set NaNs to zero.
    t_ij = np.nan_to_num(data.TCOSTMATRIX)

    # Infer number land-uses and cells from t_ij and lumap matrices.
    nlus = t_ij.shape[0]
    ncells = lumap.shape[0]

    # Transition costs to commodity j at cell r using present lumap (in AUD/ha).
    t_rj_audperha = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))

    # Areas in hectares for each cell, stacked for each land-use (identical).
    realarea_rj = np.stack((data.REAL_AREA,) * nlus, axis=1)

    # Total aq lic costs = aq req [Ml/ha] x area/cell [ha] x lic price [AUD/Ml].
    wr_rj = data.RAWEC['WR'].to_numpy() # Water required in Ml/ha.
    wp_rj = data.RAWEC['WP'].to_numpy() # Water licence price in AUD/Ml.
    aqlic_rj = np.nan_to_num(wr_rj * realarea_rj * wp_rj) # The total lic costs.

    # Switching from and to an _irr land-use can still incur licence costs.
    tdelta_rj = np.zeros((ncells, nlus))
    for r in range(ncells):
        j = lumap[r]
        if j % 2 == 1: # If it is currently an irrigated land use.
            # Difference with current licence is paid or refunded. TODO: TESTING.
            tdelta_rj[r] = aqlic_rj[r] - aqlic_rj[r, j]
            # # If licence cheaper than now, no pay. If not, pay difference.
            # tdelta_rj[r] = np.where( aqlic_rj[r] <= aqlic_rj[r, j]
                                   # , 0
                                   # , aqlic_rj[r] - aqlic_rj[r, j] )
        else: # If it is currently a non-irrigated land use.
            tdelta_rj[r] = aqlic_rj[r]

    # Transition costs to commodity j at cell r converted to AUD per cell.
    t_rj = t_rj_audperha * realarea_rj + tdelta_rj

    # Amortise the lump-sum transition cost.
    t_rj = amortise(t_rj, year)

    return t_rj
