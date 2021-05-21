#!/bin/env python3
#
# transitions.py - data about transition costs.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-04-30
# Last modified: 2021-05-21
#

import os.path

import numpy as np
import pandas as pd
import numpy_financial as npf

import luto.data as data

def amortise(dollars, year):
    """Return amortised `dollars` for `year`. Interest 10% over 100 years."""
    return -1 * npf.pmt(0.1, 100, pv=dollars, fv=0, when='begin')

def get_transition_matrix(year, lumap):
    """Return t_rj transition-cost matrix for `year` and `lumap` in AUD/cell."""

    # Raw transition-cost matrix is in AUD/Ha. Set NaNs to zero and amortise.
    t_ij = np.nan_to_num(amortise(data.TCOSTMATRIX, year))

    # Infer number land-uses and cells from t_ij and lumap matrices.
    nlus = t_ij.shape[0]
    ncells = lumap.shape[0]

    # Transition costs to commodity j at cell r using present lumap (in AUD/ha).
    t_rj_audperha = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))

    # Areas in hectares for each cell, stacked for each land-use (identical).
    realarea_rj = np.stack((data.REAL_AREA,) * nlus, axis=1)

    # Transition costs to commodity j at cell r converted to AUD per cell.
    t_rj = t_rj_audperha * realarea_rj

    return t_rj
