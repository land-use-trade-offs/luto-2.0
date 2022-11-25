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
Pure functions for quantities of comm's and alt. land uses.
"""


import numpy as np
from scipy.interpolate import interp1d


def get_ccimpact(data, lu, lm, year):
    """Return climate change impact multiplier at (zero-based) year index."""
    
    # Check if land-use exists in CLIMATE_CHANGE_IMPACT (e.g., dryland Pears/Rice do not occur), if not return zeros
    if lu not in {t[0] for t in data.CLIMATE_CHANGE_IMPACT[lm].columns}:
        cci = np.zeros((data.NCELLS))
        
    else: # Calculate the quantities
        # Convert year index to calendar year.
        year += data.ANNUM
    
        # First make linear interpolation function.
        xs = sorted({t[2] for t in data.CLIMATE_CHANGE_IMPACT.columns})
        yys = data.CLIMATE_CHANGE_IMPACT[lm, lu]
        f = interp1d(xs, yys, kind = 'linear', fill_value = 'extrapolate')
        cci = f(year)
        
    # Return the interpolated values.                                        ################## Normalise ccimpact multipliers such that 2010 == 1
    return cci


def lvs_veg_types(lu):
    """Return livestock and vegetation types of the livestock land-use `lu`."""

    # Determine type of livestock.
    if 'beef' in lu.lower():
        lvstype = 'BEEF'
    elif 'sheep' in lu.lower():
        lvstype = 'SHEEP'
    elif 'dairy' in lu.lower():
        lvstype = 'DAIRY'
    else:
        raise KeyError("Livestock type '%s' not identified." % lu)

    # Determine type of vegetation.
    if 'natural' in lu.lower():
        vegtype = 'NATL'
    elif 'modified' in lu.lower():
        vegtype = 'MODL'
    else:
        raise KeyError("Vegetation type '%s' not identified." % lu)

    return lvstype, vegtype


def get_yield_pot( data # Data object or module.
                 , lvstype # Livestock type (one of 'BEEF', 'SHEEP' or 'DAIRY')
                 , vegtype # Vegetation type (one of 'NATL' or 'MODL')
                 , lm # Land-management type.
                 ):
    """Return the yield potential (head/ha) for livestock by land cover type."""

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
    if vegtype == 'NATL':
        potential *= data.SAFE_PUR_NATL
    elif vegtype == 'MODL':
        potential *= data.SAFE_PUR_MODL
    else:
        raise KeyError("Land cover type '%s' not identified." % vegtype)

    # Multiply potential by appropriate irrigation factor.
    if lm == 'irr':
        potential *= 2

    return potential


def get_quantity_lvstk( data # Data object or module.
                      , pr   # Livestock + product like 'SHEEP - MODIFIED LAND WOOL').
                      , lm   # Land management.
                      , year # Number of years post base-year ('annum').
                      ):
    """Return livestock yield of `pr`+`lm` in `year` as 1D Numpy array...
    
    `data`: data object/module -- assumes fields like in `luto.data`.
    `pr`: product (like 'wool from nveg grazing sheep').
    `lm`: land management (e.g. 'dry', 'irr', 'org').
    `year`: number of years from base year, counting from zero.
    
    ...in the following units:
    
    BEEF and SHEEP meat (1) and live exports (3) in tonnes of meat per cell,
    SHEEP wool (2) in tonnes per cell,
    DAIRY (1) in kilolitres milk per cell.
    """
    
    # Get livestock and land cover type.
    lvstype, vegtype = lvs_veg_types(pr)

    # Get the yield potential.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm)

    # Determine base quantity case-by-case.

    # Beef yields just beef (1) and live exports (3) (both in tonnes of meat per ha).
    if lvstype == 'BEEF': # (F1 * Q1) or (F3 * Q3).
        if 'MEAT' in pr:
            quantity = ( data.AGEC_LVSTK['F1', lvstype]
                       * data.AGEC_LVSTK['Q1', lvstype] )
        elif 'LEXP' in pr:
            quantity = ( data.AGEC_LVSTK['F3', lvstype]
                       * data.AGEC_LVSTK['Q3', lvstype] )
        else:
            raise KeyError("Unknown %s product. Check `pr` key." % lvstype)

    # Sheep yield sheep meat (1), wool (2) or live exports (3).
    elif lvstype == 'SHEEP': # (F1 * Q1), (F2 * Q2) or (F3 * Q3).
        if 'MEAT' in pr:
            quantity = ( data.AGEC_LVSTK['F1', lvstype]
                       * data.AGEC_LVSTK['Q1', lvstype] )
        elif 'WOOL' in pr:
            quantity = ( data.AGEC_LVSTK['F2', lvstype]
                       * data.AGEC_LVSTK['Q2', lvstype] )
        elif 'LEXP' in pr:
            quantity = ( data.AGEC_LVSTK['F3', lvstype]
                       * data.AGEC_LVSTK['Q3', lvstype] )
        else:
            raise KeyError("Unknown %s product. Check `pr` key." % lvstype)

    # Dairy yields just dairy (1, kilolitres of milk per ha).
    elif lvstype == 'DAIRY': # (F1 * Q1).
        if 'DAIRY' in pr: # No 'elif', just keeping structure consistent.
            quantity = ( data.AGEC_LVSTK['F1', lvstype]
                       * data.AGEC_LVSTK['Q1', lvstype] 
                       / 1000 ) # Convert to KL
        else:
            raise KeyError("Unknown %s product. Check `pr` key." % lvstype)

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
    """Return crop yield (tonne/cell) of `pr`+`lm` in `year` as 1D Numpy array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `pr`: product -- equivalent to land use for crops (e.g. 'winterCereals').
    `lm`: land management (e.g. 'dry', 'irr').
    `year`: number of years from base year, counting from zero.
    """
    
    # Check if land-use exists in AGEC_CROPS (e.g., dryland Pears/Rice do not occur), if not return zeros
    if pr not in data.AGEC_CROPS['Yield', lm].columns:
        quantity = np.zeros((data.NCELLS))
        
    else: # Calculate the quantities
        
        # Get the raw quantities in tonnes/ha from data.
        quantity = data.AGEC_CROPS['Yield', lm, pr].copy().to_numpy()
    
        # Convert to tonnes/cell.
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
    if pr in data.PR_CROPS:
        q = get_quantity_crop(data, pr.capitalize(), lm, year)
        
    # If it is livestock, it is known how to get the quantities.
    elif pr in data.PR_LVSTK:
        q = get_quantity_lvstk(data, pr, lm, year)
        
    # If it is none of the above, it is not known how to get the quantities.
    else:
        raise KeyError("Land use '%s' not found in data." % pr)

    # Apply yield increase multiplier.                                     ***************** Turned off yield modifiers for now
    # q *= data.YIELDINCREASE[lm, pr][year]

    # Apply climate change impact multiplier.
    # lu = data.PR2LU_DICT[pr]
    # q *= get_ccimpact(data, lu, lm, year)

    return q


def get_quantity_matrix(data, lm, year):
    """Return q_rp matrix of quantities per cell per pr as 2D Numpy array."""
    q_rp = np.zeros((data.NCELLS, len(data.PRODUCTS)))
    for j, pr in enumerate(data.PRODUCTS):
        q_rp[:, j] = get_quantity(data, pr, lm, year)
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(q_rp)


def get_quantity_matrices(data, year):
    """Return q_mrp matrix of quantities per cell as 3D Numpy array."""
    return np.stack(tuple( get_quantity_matrix(data, lm, year)
                           for lm in data.LANDMANS ))
