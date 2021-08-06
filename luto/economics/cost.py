#!/bin/env python3
#
# cost.py - pure functions to calculate costs of commodities and alt. land uses.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-22
# Last modified: 2021-08-06
#

import numpy as np

from luto.economics.quantity import get_quantity


def get_cost( data # Data object or module.
            , lu   # Land use.
            , lm   # Land management.
            , year # Number of years post base-year ('annum').
            ):
    """Return production cost [AUD/cell] of `lu`+`lm` in `year` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
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
    qc *= get_quantity(data, lu, lm, year) / data.REAL_AREA

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

def get_cost_matrix(data, lm, year):
    """Return c_rj matrix of costs/cell per lu under `lm` in `year`."""
    c_rj = np.zeros((data.NCELLS, len(data.LANDUSES)))
    for j, lu in enumerate(data.LANDUSES):
        c_rj[:, j] = get_cost(data, lu, lm, year)
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(c_rj)

def get_cost_matrices(data, year):
    """Return c_rmj matrix of costs per cell as 3D Numpy array."""
    return np.stack(tuple(get_cost_matrix(data, lm, year) for lm in data.LANDMANS))
