# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
# 
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# 
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>. 

"""
Pure functions to calculate costs of commodities and alt. land uses.
"""


import numpy as np

from luto.economics.quantity import get_yield_pot, lvs_veg_types, get_quantity


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

    # Fixed costs as FLC + FOC + FDC.
    fc = 0
    for c in ['FLC', 'FOC', 'FDC']:
        fc += data.AGEC_CROPS[c, lm, lu].copy()

    # Area costs.
    ac = data.AGEC_CROPS['AC', lm, lu]

    # Quantity costs as cost per tonne x tonne per cell / hectare per cell
    qc = ( data.AGEC_CROPS['QC', lm, lu]
         * get_quantity(data, lu.upper(), lm, year) # lu.upper() only for crops.
         / data.REAL_AREA )

    # Water costs as water required in Ml per hectare x delivery price per Ml.
    if lm == 'irr':
        wc = data.AGEC_CROPS['WR', lm, lu] * data.AGEC_CROPS['WP', lm, lu]
    elif lm == 'dry':
        wc = 0
    else: # Passed lm is neither `dry` nor `irr`.
        raise KeyError("Unknown %s land management. Check `lm` key." % lm)

    # ------------ #
    # Total costs. #
    # ------------ #
    tc = fc + (qc + ac + wc)

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

    # Get the yield potential, i.e. the total number of heads per hectare.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm)

    # Quantity costs as costs per head x heads per hectare.
    cost_q = data.AGEC_LVSTK['QC', lvstype] * yield_pot

    # Water costs as water requirements x delivery price x heads per hectare
    if lm == 'irr': # Irrigation water if required.
        WR_IRR = data.AGEC_LVSTK['WR_IRR', lvstype]
    elif lm == 'dry': # No irrigation water if not required.
        WR_IRR = 0
    else: # Passed lm is neither `dry` nor `irr`.
        raise KeyError("Unknown %s land management. Check `lm` key." % lm)
    WR_DRN = data.AGEC_LVSTK['WR_DRN', lvstype] # Drinking water required.
    cost_w = (WR_DRN + WR_IRR) * data.WATER_DELIVERY_PRICE * yield_pot

    # Fixed and area costs.
    cost_fa = ( data.AGEC_LVSTK['AC', lvstype] # Area costs.
              + data.AGEC_LVSTK['FOC', lvstype] # Operational costs.
              + data.AGEC_LVSTK['FLC', lvstype] # Labour costs.
              + data.AGEC_LVSTK['FDC', lvstype] ) # Depreciation costs.

    # Total costs are quantity + water + fixed + area costs.
    cost = cost_q + cost_w + cost_fa

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
