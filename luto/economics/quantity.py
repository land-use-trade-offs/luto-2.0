#!/bin/env python3
#
# quantity.py - pure functions for quantities of comm's and alt. land uses.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-26
# Last modified: 2021-06-01
#

import numpy as np

import luto.data as data

def get_quantity(lu, year):
    """Return yield in tonne/cell of `lu` in `year` as 1D Numpy array."""

    # Every crop or animal x irrigation status has its own yieldincrease.
    #
    # Animals live on land which incurs pasture damage if not irrigated.
    if any(s in lu for s in ['beef_dry', 'sheep_dry', 'dairy_dry']):
        quantity = ( data.RAWEC['Q1', lu].values
                   * data.AG_PASTURE_DAMAGE[year]
                   * data.YIELDINCREASE[lu][year] )
    # Crops grow on land which incurs dryland damage if not irrigated.
    elif '_dry' in lu:
        quantity = ( data.RAWEC['Q1', lu].values
                   * data.AG_DRYLAND_DAMAGE[year]
                   * data.YIELDINCREASE[lu][year] )
    # If the land is irrigated there is no such damage.
    else:
        quantity = ( data.RAWEC['Q1', lu].values
                   * data.YIELDINCREASE[lu][year] )

    # Quantities so far in tonnes/ha. Now convert to tonnes/cell.
    quantity *= data.REAL_AREA

    return quantity

def get_quantity_matrix(year):
    """Return q_rj matrix of quantities per cell per lu as 2D Numpy array."""
    q_rj = np.zeros((data.NCELLS, data.NLUS))
    for j, lu in enumerate(data.LANDUSES):
        q_rj[:, j] = get_quantity(lu, year)
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(q_rj)
