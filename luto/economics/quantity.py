#!/bin/env python3
#
# quantity.py - pure functions for quantities of comm's and alt. land uses.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-26
# Last modified: 2021-05-03
#

import numpy as np

import luto.data as data

def get_quantity_fn():
    """Return function taking lu, year and returning corresponding quantity."""

    # Prepare the dictionary.
    qovertime = {}

    for lu in data.LANDUSES:

        # Every crop or animal x irrigation status has its own yieldincrease.
        #
        # Animals live on land which incurs pasture damage if not irrigated.
        if any(s in lu for s in ['beef_dry', 'sheep_dry', 'dairy_dry']):
            qovertime[lu] = lambda lu, year: ( data.RAWEC['Q1', lu].values
                                             * data.AG_PASTURE_DAMAGE[year]
                                             * data.YIELDINCREASE[lu][year] )
        # Crops grow on land which incurs dryland damage if not irrigated.
        elif '_dry' in lu:
            qovertime[lu] = lambda lu, year: ( data.RAWEC['Q1', lu].values
                                             * data.AG_DRYLAND_DAMAGE[year]
                                             * data.YIELDINCREASE[lu][year] )
        # If the land is irrigated there is no such damage.
        else:
            qovertime[lu] = lambda lu, year: ( data.RAWEC['Q1', lu].values
                                             * data.YIELDINCREASE[lu][year] )

    return lambda lu, year: qovertime[lu](lu, year)

def get_quantity_matrix(year):
    """Return q_rj matrix of quantities per cell per lu as 2D Numpy array."""
    q_rj = np.zeros((data.NCELLS, data.NLUS))
    fn = get_quantity_fn()
    for j, lu in enumerate(data.LANDUSES):
        q_rj[:, j] = fn(lu, year)
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(q_rj)
