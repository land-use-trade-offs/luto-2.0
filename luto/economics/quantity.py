#!/bin/env python3
#
# quantity.py - pure functions for quantities of comm's and alt. land uses.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-26
# Last modified: 2021-08-16
#

import numpy as np


def get_yield_pot( data # Data object or module.
                 , lvstype # Livestock type (one of 'BEEF', 'SHEEP' or 'DAIRY')
                 , vegtype # Vegetation type (one of 'NVEG' or 'SOWN')
                 ):
    """Return the yield potential for livestock and vegetation type."""

    # Factors varying as a function of `lvstype`.
    dse_per_head = { 'BEEF': 8
                   , 'SHEEP': 1.5
                   , 'DAIRY': 17 }
    grassfed_factor = { 'BEEF': 0.85
                      , 'SHEEP': 0.85
                      , 'DAIRY': 0.65 }
    denominator = ( 365
                  * dse_per_head[lvstype]
                  * grassfed_factor[lvstype] )

    # Base potential.
    potential = data.FEED_REQ * data.PASTURE_KG_DM_HA / denominator

    # Multiply potential by appropriate SAFE_PUR.
    if vegtype == 'NVEG':
        potential *= data.SAFE_PUR_NVEG
    elif vegtype == 'SOWN':
        potential *= data.SAFE_PUR_SOWN
    else:
        raise KeyError("Vegetation type '%s' not identified." % vegtype)

    # Multiply potential by appropriate irrigation factor.
    if lm == 'irr':
        potential *= 2

    return potential

def get_quantity_crop( data # Data object or module.
                     , lu   # Land use.
                     , lm   # Land management.
                     , year # Number of years post base-year ('annum').
                     ):
    """Return crop yield [tonne/cell] of `lu`+`lm` in `year` as 1D Numpy array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'winterCereals' or 'beef').
    `lm`: land management (e.g. 'dry', 'irr', 'org').
    `year`: number of years from base year, counting from zero.
    """

    # Every crop x irrigation status has its own yieldincrease.
    #
    # Crops grow on land which incurs dryland damage if not irrigated.
    if lm == 'dry':
        quantity = ( data.AGEC_CROPS['Yield', lm, lu].values
                   * data.AG_DRYLAND_DAMAGE[year]
                   * data.YIELDINCREASE[lm, lu][year] )
    # If the land is irrigated there is no such damage.
    else:
        quantity = ( data.AGEC_CROPS['Yield', lm, lu].values
                   * data.YIELDINCREASE[lm, lu][year] )

    # Quantities so far in tonnes/ha. Now convert to tonnes/cell.
    quantity *= data.REAL_AREA

    return quantity

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
    # If it is a crop, it is known how to get the quantities.
    if lu in data.CROPS:
        return get_quantity_crops(data, lu, lm, year)
    # If it is livestock, it is known how to get the quantities.
    elif lu in data.LVSTK:
        return get_quantity_lvstk(data, lu, lm, year)
    # If it is none of the above, it is not known how to get the quantities.
    else:
        raise KeyError("Land use '%s' not found in data." % lu)

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
