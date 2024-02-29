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

from typing import Dict
import numpy as np
import pandas as pd

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.data import Data, lvs_veg_types
from luto.economics.agricultural.quantity import get_yield_pot, get_quantity

def get_rev_crop( data: Data   # Data object.
                , lu           # Land use.
                , lm           # Land management.
                , yr_idx       # Number of years post base-year ('YR_CAL_BASE').
                ):
    """Return crop profit [AUD/cell] of `lu`+`lm` in `yr_idx` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals' or 'Beef - natural land').
    `lm`: land management (e.g. 'dry', 'irr').
    `yr_idx`: number of years from base year, counting from zero.
    """
    
    # Check if land-use exists in AGEC_CROPS (e.g., dryland Pears/Rice do not occur), if not return zeros
    if lu not in data.AGEC_CROPS['P1', lm].columns:
        rev_t = np.zeros((data.NCELLS))
        
    else:
        # Revenue in $ per cell (includes REAL_AREA via get_quantity)
        rev_t = ( data.AGEC_CROPS['P1', lm, lu]
                * get_quantity( data, lu.upper(), lm, yr_idx )  # lu.upper() only for crops as needs to be in product format in get_quantity().
                ).values
    
    # Return revenue as MultiIndexed DataFrame.
    return pd.DataFrame(rev_t, columns=pd.MultiIndex.from_product([[lu],[lm],['Revenue']]))

def get_rev_lvstk( data: Data   # Data object.
                 , lu           # Land use.
                 , lm           # Land management.
                 , yr_idx       # Number of years post base-year ('YR_CAL_BASE').
                 ):
    """Return livestock revenue [AUD/cell] of `lu`+`lm` in `yr_idx` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals' or 'Beef - natural land').
    `lm`: land management (e.g. 'dry', 'irr').
    `yr_idx`: number of years from base year, counting from zero."""
    
    # Get livestock and vegetation type.
    lvstype, vegtype = lvs_veg_types(lu)

    # Get the yield potential, i.e. the total number of heads per hectare.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)
    
    # Revenue in $ per cell (includes RESMULT via get_quantity)
    if lvstype == 'BEEF':

        # Get the revenue from meat and live exports. Set to zero if not produced.
        rev_meat = yield_pot * (                               # Stocking density (head/ha)
                            ( data.AGEC_LVSTK['F1', lvstype]   # Fraction of herd producing (0 - 1)
                            * data.AGEC_LVSTK['Q1', lvstype]   # Quantity produced per head (meat tonnes/head)
                            * data.AGEC_LVSTK['P1', lvstype] ) # Price per unit quantity ($/tonne of meat)
                            )
        
        rev_lexp = yield_pot * (  
                            ( data.AGEC_LVSTK['F3', lvstype]   # Fraction of herd producing (0 - 1)
                            * data.AGEC_LVSTK['Q3', lvstype]   # Quantity produced per head (animal weight tonnes/head)
                            * data.AGEC_LVSTK['P3', lvstype] ) # Price per unit quantity ($/tonne of animal)
                            )  
        
        # Set Wool and Milk to zero as they are not produced by beef cattle
        rev_wool = rev_milk = np.zeros((data.NCELLS))

    elif lvstype == 'SHEEP':    

        # Get the revenue from meat, wool and live exports. Set to zero if not produced.
        rev_meat = yield_pot * (  # Meat                           # Stocking density (head/ha)
                                ( data.AGEC_LVSTK['F1', lvstype]   # Fraction of herd producing (0 - 1)
                                * data.AGEC_LVSTK['Q1', lvstype]   # Quantity produced per head (meat tonnes/head)
                                * data.AGEC_LVSTK['P1', lvstype] ) # Price per unit quantity ($/tonne of meat)
                                )
        rev_wool = yield_pot * (  # Wool                           # Stocking density (head/ha) 
                                ( data.AGEC_LVSTK['F2', lvstype]   # Fraction of herd producing (0 - 1) 
                                * data.AGEC_LVSTK['Q2', lvstype]   # Quantity produced per head (wool tonnes/head)
                                * data.AGEC_LVSTK['P2', lvstype] ) # Price per unit quantity ($/tonne wool)
                                )   
        
        rev_lexp = yield_pot * (  # Live exports                   # Stocking density (head/ha)
                                ( data.AGEC_LVSTK['F3', lvstype]   # Fraction of herd producing (0 - 1) 
                                * data.AGEC_LVSTK['Q3', lvstype]   # Quantity produced per head (animal weight tonnes/head)
                                * data.AGEC_LVSTK['P3', lvstype] ) # Price per unit quantity ($/tonne of whole animal)
                                )
        
        # Set Milk to zero as it is not produced by sheep
        rev_milk = np.zeros((data.NCELLS)) # Set Milk to zero as it is not produced by sheep




    elif lvstype == 'DAIRY':

        # Get the revenue from milk. Set to zero if not produced.
        rev_milk = yield_pot * (  # Milk                           # Stocking density (head/ha)
                                ( data.AGEC_LVSTK['F1', lvstype]   # Fraction of herd producing (0 - 1) 
                                * data.AGEC_LVSTK['Q1', lvstype]   # Quantity produced per head (milk litres/head)
                                * data.AGEC_LVSTK['P1', lvstype] ) # Price per unit quantity ($/litre milk)
                                )
        
        # Set Meat, Wool and Live exports to zero
        rev_meat = rev_wool = rev_lexp = np.zeros((data.NCELLS)) 

    else:  # Livestock type is unknown.
        raise KeyError("Unknown %s livestock type. Check `lvstype`." % lvstype)   

    # Put the revenues into a MultiIndex DataFrame
    rev_seperate = pd.DataFrame(np.stack((rev_meat, rev_wool, rev_lexp, rev_milk), axis=1), 
                                columns=pd.MultiIndex.from_product([[lu], [lm], ['Meat', 'Wool', 'Live Exports', 'Milk']]))
    
    # Revenue so far in AUD/ha. Now convert to AUD/cell including resfactor.
    rev_seperate = rev_seperate * data.REAL_AREA.reshape(-1,1) # Convert to AUD/cell
    
    # Return revenue as numpy array.
    return rev_seperate


def get_rev( data: Data    # Data object.
            , lu           # Land use.
            , lm           # Land management.
            , yr_idx       # Number of years post base-year ('YR_CAL_BASE')
            ):
    """Return revenue from production [AUD/cell] of `lu`+`lm` in `yr_idx` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals').
    `lm`: land management (e.g. 'dry', 'irr').
    `yr_idx`: number of years from base year, counting from zero.
    """
    # If it is a crop, it is known how to get the revenue.
    if lu in data.LU_CROPS:
        return get_rev_crop(data, lu, lm, yr_idx)
    
    # If it is livestock, it is known how to get the revenue.
    elif lu in data.LU_LVSTK:
        return get_rev_lvstk(data, lu, lm, yr_idx)
    
    # If neither crop nor livestock but in LANDUSES it is unallocated land.
    elif lu in data.AGRICULTURAL_LANDUSES:
        return  pd.DataFrame(np.zeros((data.NCELLS, 1)),
                             columns=pd.MultiIndex.from_product([[lu],[lm],['Revenue']]))
    
    # If it is none of the above, it is not known how to get the revenue.
    else:
        raise KeyError("Land-use '%s' not found in data.LANDUSES" % lu)


def get_rev_matrix(data: Data, lm, yr_idx):
    """Return r_rj matrix of revenue/cell per lu under `lm` in `yr_idx`."""
    
    # Concatenate the revenue from each land use into a single Multiindex DataFrame.
    r_rjs = pd.concat([get_rev(data, lu, lm, yr_idx) for lu in data.AGRICULTURAL_LANDUSES], axis=1)
    r_rjs = r_rjs.fillna(0)
    return r_rjs


def get_rev_matrices(data: Data, yr_idx, aggregate:bool = True):
    """Return r_mrj matrix of revenue per cell as 3D Numpy array."""

    # Concatenate the revenue from each land management into a single Multiindex DataFrame.
    rev_rjms = pd.concat([get_rev_matrix(data, lm, yr_idx) for lm in data.LANDMANS], axis=1)

    if aggregate == True:
        j,m,s = rev_rjms.columns.levshape
        rev_rjm = rev_rjms.groupby(level=[0,1],axis=1).sum().values.reshape(-1,*[j,m])
        rev_mrj = np.einsum('rjm->mrj',rev_rjm)
        return rev_mrj
    
    elif aggregate == False:
        # Concatenate the revenue from each land management into a single Multiindex DataFrame.
        return rev_rjms
    else:
        raise KeyError("aggregate must be either True or False")


def get_asparagopsis_effect_r_mrj(data: Data, r_mrj, yr_idx):
    """
    Applies the effects of using asparagopsis to the revenue data
    for all relevant agr. land uses.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_r_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        j = lu_codes[lu_idx]
        multiplier = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in quantity produced
            new_r_mrj[:, :, lu_idx] = r_mrj[:, :, j] * (multiplier - 1)

    return new_r_mrj


def get_precision_agriculture_effect_r_mrj(data: Data, r_mrj, yr_idx):
    """
    Applies the effects of using precision agriculture to the revenue data
    for all relevant agr. land uses.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_r_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        j = lu_codes[lu_idx]
        multiplier = data.PRECISION_AGRICULTURE_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            new_r_mrj[:, :, lu_idx] = r_mrj[:, :, j] * (multiplier - 1)

    return new_r_mrj


def get_ecological_grazing_effect_r_mrj(data: Data, r_mrj, yr_idx):
    """
    Applies the effects of using ecologiacl grazing to the revenue data
    for all relevant agr. land uses.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_r_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        j = lu_codes[lu_idx]
        multiplier = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            new_r_mrj[:, :, lu_idx] = r_mrj[:, :, j] * (multiplier - 1)

    return new_r_mrj


def get_agricultural_management_revenue_matrices(data: Data, r_mrj, yr_idx) -> Dict[str, np.ndarray]:
    asparagopsis_data = get_asparagopsis_effect_r_mrj(data, r_mrj, yr_idx)
    precision_agriculture_data = get_precision_agriculture_effect_r_mrj(data, r_mrj, yr_idx)
    eco_grazing_data = get_ecological_grazing_effect_r_mrj(data, r_mrj, yr_idx)

    ag_management_data = {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
    }

    return ag_management_data
