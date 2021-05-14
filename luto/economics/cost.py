#!/bin/env python3
#
# cost.py - pure functions to calculate costs of commodities and alt. land uses.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-22
# Last modified: 2021-05-03
#

import numpy as np

import luto.data as data
from luto.economics.quantity import get_quantity_fn

def get_cost_fn():
    """Return function taking lu, year and returning corresponding cost."""

    # Get the quantities.
    q = get_quantity_fn()

    # Prepare the dictionary.
    costs = {} # Total costs.

    for lu in data.LANDUSES:

        # Fixed costs.
        fc = 0 # (Re-) set the fixed costs for each land use.
        for c in ['FLC', 'FOC', 'FDC']:
            fc += data.RAWEC[c, lu].copy()

        # Area costs.
        ac = data.RAWEC['AC', lu]

        # Turn QC into actual cost per quantity, i.e. divide by quantity.
        qc = data.RAWEC['QC', lu] / data.RAWEC['Q1', lu]

        # Water costs.
        if '_irr' in lu:
            wc = data.RAWEC['WC', lu]
        else:
            wc = 0

        # Total costs.
        costs[lu] = lambda lu, year: (fc + ( qc * q(lu, year)
                                           + ac
                                           + wc )*data.DIESEL_PRICE_PATH[year])

    return lambda lu, year: costs[lu](lu, year)

def get_cost_matrix(year):
    """Return c_rj matrix of unit costs per cell per lu as 2D Numpy array."""
    c_rj = np.zeros((data.NCELLS, data.NLUS))
    fn = get_cost_fn()
    for j, lu in enumerate(data.LANDUSES):
        c_rj[:, j] = fn(lu, year)
    # Make sure NaNs are larger than GRB.INFINITY.
    return np.nan_to_num(c_rj)
