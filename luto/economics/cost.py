#!/bin/env python3
#
# cost.py - pure functions to calculate costs of commodities and alt. land uses.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-22
# Last modified: 2021-07-15
#

import numpy as np

import luto.data as data
from luto.economics.quantity import get_quantity, get_quantity_ag


def get_cost_ag(lu, lm, year):
    """Return production cost [AUD/cell] of `lu`+`lm` in `year` as np array.

    `lu`: land use (e.g. 'winterCereals' or 'beef').
    `lm`: land management (e.g. 'dry', 'irr', 'org').
    `year`: number of years from base year, counting from zero.
    """

    # ------------ #
    # Fixed costs. #
    # ------------ #
    fc = 0
    for c in ['FLC', 'FOC', 'FDC']:
        fc += data.AGEC[c, lm, lu].copy()

    # ----------- #
    # Area costs.
    # ----------- #
    ac = data.AGEC['AC', lm, lu]

    # --------------- #
    # Quantity costs.
    # --------------- #

    # Turn QC into actual cost per quantity, i.e. divide by quantity.
    qc = data.AGEC['QC', lm, lu] / data.AGEC['Q1', lm, lu]

    # Multiply by quantity (w/ trends) for q-costs per ha. Divide for per cell.
    qc *= get_quantity_ag(lu, lm, year) / data.REAL_AREA

    # ------------ #
    # Water costs.
    # ------------ #
    if lm == 'irr':
        # Water delivery costs in AUD/ha.
        wc = data.AGEC['WC', lm, lu]
    else:
        wc = 0

    # ------------ #
    # Total costs.
    # ------------ #
    tc = fc + (qc + ac + wc) * data.DIESEL_PRICE_PATH[year]

    # Costs so far in AUD/ha. Now convert to AUD/cell.
    cost = tc * data.REAL_AREA

    return cost.values

def get_cost_matrix(lm, year):
    """Return c_rj matrix of costs/cell per lu under `lm` in `year`."""
    c_rj = np.zeros((data.NCELLS, len(data.RLANDUSES)))
    for j, lu in enumerate(data.RLANDUSES):
        c_rj[:, j] = get_cost_ag(lu, lm, year)
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(c_rj)

def get_cost(lu, year):
    """Return cost in AUD/cell of producing `lu` in `year` as 1D Numpy array."""

    # ------------ #
    # Fixed costs. #
    # ------------ #
    fc = 0
    for c in ['FLC', 'FOC', 'FDC']:
        fc += data.RAWEC[c, lu].copy()

    # ----------- #
    # Area costs.
    # ----------- #
    ac = data.RAWEC['AC', lu]

    # --------------- #
    # Quantity costs.
    # --------------- #

    # Turn QC into actual cost per quantity, i.e. divide by quantity.
    qc = data.RAWEC['QC', lu] / data.RAWEC['Q1', lu]

    # Multiply by quantity (w/ trends) for q-costs per ha. Divide for per cell.
    qc *= get_quantity(lu, year) / data.REAL_AREA

    # ------------ #
    # Water costs.
    # ------------ #
    if '_irr' in lu:
        # Water delivery costs in AUD/ha.
        wc = data.RAWEC['WC', lu]
    else:
        wc = 0

    # ------------ #
    # Total costs.
    # ------------ #
    tc = fc + (qc + ac + wc) * data.DIESEL_PRICE_PATH[year]

    # Costs so far in AUD/ha. Now convert to AUD/cell.
    cost = tc * data.REAL_AREA

    return cost.values

def get_cost_matrix_old(year):
    """Return c_rj matrix of unit costs per cell per lu as 2D Numpy array."""
    c_rj = np.zeros((data.NCELLS, data.NLUS))
    for j, lu in enumerate(data.LANDUSES):
        c_rj[:, j] = get_cost(lu, year)
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(c_rj)
