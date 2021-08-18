#!/bin/env python3
#
# quantity.py - pure functions for quantities of comm's and alt. land uses.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-26
# Last modified: 2021-08-18
#

import numpy as np

from luto.economics.cost import lvs_veg_type

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

def get_quantity_lvstk( data # Data object or module.
                      , pr   # Livestock +  product like 'nveg-grazed wool').
                      , lm   # Land management.
                      , year # Number of years post base-year ('annum').
                      ):
    """Return lvstk yield [tonne/cell] of `pr`+`lm` in `year` as 1D Numpy array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `pr`: product (like 'wool from nveg grazing sheep').
    `lm`: land management (e.g. 'dry', 'irr', 'org').
    `year`: number of years from base year, counting from zero.
    """
    # Get livestock and vegetation type.
    lvstype, vegtype = lvs_veg_types(pr)

    # Get the yield potential.
    yield_pot = get_yield_pot(data, lvstype, vegtype)

    # Beef yields composed of just beef (Q1) and live exports (Q3).
    if lvstype == 'BEEF': # F1 * Q1 + F3 * Q3.
        quantity = ( ( data.AGEC_LVSTK['F1', lvstype]
                     * data.AGEC_LVSTK['Q1', lvstype] )
                   + ( data.AGEC_LVSTK['F3', lvstype]
                     * data.AGEC_LVSTK['Q3', lvstype] ) )
    # Sheep yields composed of sheep meat (Q1)+live exports (Q3) or wool (Q2).
    elif lvstype == 'SHEEP': # (F1 * Q1 + F3 * Q3) or F2 * Q2.
        quantity = ( ( data.AGEC_LVSTK['F1', lvstype]
                     * data.AGEC_LVSTK['Q1', lvstype] )
                   + ( data.AGEC_LVSTK['F3', lvstype]
                     * data.AGEC_LVSTK['Q3', lvstype] ) )
    # Dairy yields composed of just dairy (Q1).
    elif lvstype == 'DAIRY': # F1 * Q1.
        quantity = ( data.AGEC_LVSTK['F1', lvstype]
                   * data.AGEC_LVSTK['Q1', lvstype] )
    else:
        raise KeyError("Livestock type '%s' not identified." % lvstype)

    # Quantity is base quantity times the yield potential.
    quantity *= yield_pot

    # Quantities so far in tonnes/ha. Now convert to tonnes/cell.
    quantity *= data.REAL_AREA

    return quantity

def get_quantity_crop( data # Data object or module.
                     , pr   # Product -- equivalent to land use for crops.
                     , lm   # Land management.
                     , year # Number of years post base-year ('annum').
                     ):
    """Return crop yield [tonne/cell] of `pr`+`lm` in `year` as 1D Numpy array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `pr`: product -- equivalent to land use for crops (e.g. 'winterCereals').
    `lm`: land management (e.g. 'dry', 'irr', 'org').
    `year`: number of years from base year, counting from zero.
    """

    # Every crop x irrigation status has its own yieldincrease.
    #
    # Crops grow on land which incurs dryland damage if not irrigated.
    if lm == 'dry':
        quantity = ( data.AGEC_CROPS['Yield', lm, pr].to_numpy()
                   # TODO: New damage trends. * data.AG_DRYLAND_DAMAGE[year]
                   * data.YIELDINCREASE[lm, pr][year] )
    # If the land is irrigated there is no such damage.
    else:
        quantity = ( data.AGEC_CROPS['Yield', lm, pr].to_numpy()
                   * data.YIELDINCREASE[lm, pr][year] )

    # Quantities so far in tonnes/ha. Now convert to tonnes/cell.
    quantity *= data.REAL_AREA

    return quantity

def get_quantity( data # Data object or module.
                , pr   # The stuff yielded.
                , lm   # Land management.
                , year # Number of years post base-year ('annum').
                ):
    """Return yield in tonne/cell of `pr`+`lm` in `year` as 1D Numpy array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `pr`: product (like 'winterCereals' or 'wool').
    `lm`: land management (e.g. 'dry', 'irr', 'org').
    `year`: number of years from base year, counting from zero.
    """
    # If it is a crop, it is known how to get the quantities.
    if pr in data.CROPS:
        return get_quantity_crop(data, pr, lm, year)
    # If it is livestock, it is known how to get the quantities.
    elif pr in data.LVSTK: # TODO: Change into a list of livestock _products_.
        return get_quantity_lvstk(data, pr, lm, year)
    # If it is none of the above, it is not known how to get the quantities.
    else:
        raise KeyError("Land use '%s' not found in data." % pr)

def get_quantity_matrix(data, lm, year):
    """Return q_rj matrix of quantities per cell per pr as 2D Numpy array."""
    # TODO: LANDUSES to change to COMMODITIES
    q_rj = np.zeros((data.NCELLS, len(data.LANDUSES)))
    for j, pr in enumerate(data.LANDUSES):
        q_rj[:, j] = get_quantity(data, pr, lm, year)
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(q_rj)

def get_quantity_matrices(data, year):
    """Return q_rmj matrix of quantities per cell as 3D Numpy array."""
    return np.stack(tuple( get_quantity_matrix(data, lm, year)
                           for lm in data.LANDMANS ))
