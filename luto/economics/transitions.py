#!/bin/env python3
#
# transitions.py - data about transition costs.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-04-30
# Last modified: 2021-05-03
#

import os.path

import numpy as np
import pandas as pd
import numpy_financial as npf

import luto.data as data

def amortise(dollars, year):
    """Return amortised `dollars` for `year`. Interest 10% over 100 years."""
    return -1 * npf.pmt(0.1, 100, pv=dollars, fv=0, when='begin')

# Raw transition-cost matrix has costs in AUD/Ha. As 121 Ha/cell, times 121.
t_ij_unamortised = 121 * data.TCOSTMATRIX

# Amortise with 10% interest over 100 years. NaN > GRB.INFINITY. Return as closure.
get_transition_matrix = lambda year: np.nan_to_num(amortise(t_ij_unamortised, year))
