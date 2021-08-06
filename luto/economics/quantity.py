#!/bin/env python3
#
# quantity.py - pure functions for quantities of comm's and alt. land uses.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-26
# Last modified: 2021-08-06
#

import numpy as np


def get_quantity( data # Data object or module.
                , lu   # Land use.
                , lm   # Land management.
                , year # Number of years post base-year ('annum').
                ):
    """Return yield in tonne/cell of `lu`+`lm` in `year` as 1D Numpy array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'winterCereals' or 'beef').
    `lm`: land management (e.g. 'dry', 'irr', 'org').
    `year`: number of years from base year, counting from zero.
    """

    # Every crop or animal x irrigation status has its own yieldincrease.
    #
    # Animals live on land which incurs pasture damage if not irrigated.
    if lm == 'dry' and lu in ['beef', 'sheep', 'dairy']:
        quantity = ( data.AGEC['Q1', lm, lu].values
                   * data.AG_PASTURE_DAMAGE[year]
                   * data.YIELDINCREASE[lm, lu][year] )
    # Crops grow on land which incurs dryland damage if not irrigated.
    elif lm == 'dry':
        quantity = ( data.AGEC['Q1', lm, lu].values
                   * data.AG_DRYLAND_DAMAGE[year]
                   * data.YIELDINCREASE[lm, lu][year] )
    # If the land is irrigated there is no such damage.
    else:
        quantity = ( data.AGEC['Q1', lm, lu].values
                   * data.YIELDINCREASE[lm, lu][year] )

    # Quantities so far in tonnes/ha. Now convert to tonnes/cell.
    quantity *= data.REAL_AREA

    return quantity

def get_quantity_matrix(data, lm, year):
    """Return q_rj matrix of quantities per cell per lu as 2D Numpy array."""
    q_rj = np.zeros((data.NCELLS, len(data.LANDUSES)))
    for j, lu in enumerate(data.LANDUSES):
        q_rj[:, j] = get_quantity(data, lu, lm, year)
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(q_rj)

def get_quantity_matrices(data, year):
    """Return q_rmj matrix of quantities per cell as 3D Numpy array."""
    return np.stack(tuple( get_quantity_matrix(data, lm, year)
                           for lm in data.LANDMANS ))
