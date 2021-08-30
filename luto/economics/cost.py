#!/bin/env python3
#
# cost.py - pure functions to calculate costs of commodities and alt. land uses.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-22
# Last modified: 2021-08-30
#

import numpy as np

from luto.economics.quantity import get_yield_pot, lvs_veg_types


def get_cost_crop( data # Data object or module.
                 , lu   # Land use.
                 , lm   # Land management.
                 , year # Number of years post base-year ('annum').
                 ):
    """Return crop prod. cost [AUD/cell] of `lu`+`lm` in `year` as np array.

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
        fc += data.AGEC_CROPS[c, lm, lu].copy()

    # ----------- #
    # Area costs. #
    # ----------- #
    ac = data.AGEC_CROPS['AC', lm, lu]

    # --------------- #
    # Quantity costs. #
    # --------------- #
    qc = data.AGEC_CROPS['QC', lm, lu]

    # TODO: below unnecessary but YIELDINCREASES perhaps need to be applied?
    # Turn QC into actual cost per quantity, i.e. divide by quantity.
    # qc = data.AGEC_CROPS['QC', lm, lu] / data.AGEC_CROPS['Yield', lm, lu]
    # Multiply by quantity (w/ trends) for q-costs per ha. Divide for per cell.
    # TODO: Is this step still required with the new data set? No, most likely
    # not. Better to apply the trend (YIELDINCREASE) separately here.
    #qc *= get_quantity(data, lu, lm, year) / data.REAL_AREA

    # ------------ #
    # Water costs. #
    # ------------ #
    if lm == 'irr':
        # Water delivery costs in AUD/ha.
        wp = data.AGEC_CROPS['WP', lm, lu]
    else:
        wp = 0

    # ------------ #
    # Total costs. #
    # ------------ #
    tc = fc + (qc + ac + wp) * data.DIESEL_PRICE_PATH[year]

    # Costs so far in AUD/ha. Now convert to AUD/cell.
    cost = tc * data.REAL_AREA

    return cost.to_numpy()

def get_cost_lvstk( data # Data object or module.
                  , lu   # Land use.
                  , lm   # Land management.
                  , year # Number of years post base-year ('annum').
                  ):
    """Return lvstk prod. cost [AUD/cell] of `lu`+`lm` in `year` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'winterCereals' or 'beef').
    `lm`: land management (e.g. 'dry', 'irr', 'org').
    `year`: number of years from base year, counting from zero.
    """
    # Get livestock and vegetation type.
    lvstype, vegtype = lvs_veg_types(lu)

    # Get the yield potential.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm)

    # Quantity and water costs.
    cost_qw = ( data.AGEC_LVSTK['QC', lvstype] # Quantity costs.
              + ( data.AGEC_LVSTK['WR_DRN', lvstype] # Drinking water required.
                * data.WATER_DELIVERY_PRICE ) # Cost of water delivery.
              ) * yield_pot # Yield potential.
    # Fixed and area costs.
    cost_fa = ( data.AGEC_LVSTK['AC', lvstype] # Area costs.
              + data.AGEC_LVSTK['FOC', lvstype] # Operational costs.
              + data.AGEC_LVSTK['FLC', lvstype] # Labour costs.
              + data.AGEC_LVSTK['FDC', lvstype] # Depreciation costs.
              )

    # TODO: if statement for irrigated on the cost_fa to multiply by 2.

    # Total costs are quantity plus fixed costs.
    cost = cost_qw + cost_fa

    # Costs so far in AUD/ha. Now convert to AUD/cell.
    cost *= data.REAL_AREA

    return cost.to_numpy()


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
    # If it is a crop, it is known how to get the costs.
    if lu in data.LU_CROPS:
        return get_cost_crop(data, lu, lm, year)
    # If it is livestock, it is known how to get the costs.
    elif lu in data.LU_LVSTK:
        return get_cost_lvstk(data, lu, lm, year)
    # If neither crop nor livestock but in LANDUSES it is unallocated land.
    elif lu in data.LANDUSES:
        return np.zeros(data.NCELLS)
    # If it is none of the above, it is not known how to get the costs.
    else:
        raise KeyError("Land use '%s' not found in data.LANDUSES" % lu)

def get_cost_matrix(data, lm, year):
    """Return c_rj matrix of costs/cell per lu under `lm` in `year`."""
    c_rj = np.zeros((data.NCELLS, len(data.LANDUSES)))
    for j, lu in enumerate(data.LANDUSES):
        c_rj[:, j] = get_cost(data, lu, lm, year)
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(c_rj)

def get_cost_matrices(data, year):
    """Return c_rmj matrix of costs per cell as 3D Numpy array."""
    return np.stack(tuple( get_cost_matrix(data, lm, year)
                           for lm in data.LANDMANS ))
