# Copyright 2023 Fjalar J. de Haan and Brett A. Bryan at Deakin University
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
Pure functions to calculate economic profit from land use.
"""

import numpy as np

from luto.economics.quantity import get_yield_pot, lvs_veg_types, get_quantity

def get_revenue_crop( data # Data object or module.
                 , lu   # Land use.
                 , lm   # Land management.
                 , year # Number of years post base-year ('annum').
                 ):
    """Return crop profit [AUD/cell] of `lu`+`lm` in `year` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals' or 'Beef - natural land').
    `lm`: land management (e.g. 'dry', 'irr').
    `year`: number of years from base year, counting from zero.
    """
    
    # Check if land-use exists in AGEC_CROPS (e.g., dryland Pears/Rice do not occur), if not return zeros
    if lu not in data.AGEC_CROPS['P1', lm].columns:
        rev_t = np.zeros((data.NCELLS))
        
    else: # Calculate the total revenue 
                  
        # Revenue in $ per cell (includes RESMULT via get_quantity)
        rev_t = ( data.AGEC_CROPS['P1', lm, lu]
                * get_quantity(data, lu.upper(), lm, year) )  # lu.upper() only for crops as needs to be in product format in get_quantity().
        
    # Return costs as numpy array.
    return rev_t


def get_revenue_lvstk( data # Data object or module.
                  , lu   # Land use.
                  , lm   # Land management.
                  , year # Number of years post base-year ('annum').
                  ):
    """Return lvstk prod. cost [AUD/cell] of `lu`+`lm` in `year` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals' or 'Beef - natural land').
    `lm`: land management (e.g. 'dry', 'irr').
    `year`: number of years from base year, counting from zero."""
    
    # Get livestock and vegetation type.
    lvstype, vegtype = lvs_veg_types(lu)

    # Get the yield potential, i.e. the total number of heads per hectare.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm)

    # Variable costs - quantity-dependent costs as costs per head x heads per hectare.
    costs_q = data.AGEC_LVSTK['QC', lvstype] * yield_pot

    # Variable costs - area-dependent costs per hectare.
    costs_a = data.AGEC_LVSTK['AC', lvstype]
    
    # Fixed costs
    costs_f = ( data.AGEC_LVSTK['FOC', lvstype]   # Fixed operating costs.
              + data.AGEC_LVSTK['FLC', lvstype]   # Fixed labour costs.
              + data.AGEC_LVSTK['FDC', lvstype] ) # Fixed depreciation costs.
    
    # Water costs in $/ha calculated as water requirements (ML/head) x heads per hectare x delivery price ($/ML)
    if lm == 'irr': # Irrigation water if required.
        WR_IRR = data.AGEC_LVSTK['WR_IRR', lvstype]
    elif lm == 'dry': # No irrigation water if not required.
        WR_IRR = 0
    else: # Passed lm is neither `dry` nor `irr`.
        raise KeyError("Unknown %s land management. Check `lm` key." % lm)
        
    WR_DRN = data.AGEC_LVSTK['WR_DRN', lvstype] # Drinking water required.
    costs_w = (WR_DRN + WR_IRR) * data.WATER_DELIVERY_PRICE * yield_pot

    # Total costs ($/ha) are variable (quantity + area) + fixed + water costs.
    costs_t = costs_q + costs_a + costs_f + costs_w

    # Costs so far in AUD/ha. Now convert to AUD/cell.
    costs_t *= data.REAL_AREA
              
    # Incorporate resfactor
    costs_t *= data.RESMULT
    
    # Return costs as numpy array.
    return costs_t.to_numpy()


def get_cost( data # Data object or module.
            , lu   # Land use.
            , lm   # Land management.
            , year # Number of years post base-year ('annum').
            ):
    """Return production cost [AUD/cell] of `lu`+`lm` in `year` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals').
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
    """Return c_mrj matrix of costs per cell as 3D Numpy array."""
    
    return np.stack(tuple( get_cost_matrix(data, lm, year)
                           for lm in data.LANDMANS )
                    ).astype(np.float32)
