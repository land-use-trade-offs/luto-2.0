# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.

 

"""
Pure functions to calculate economic profit from land use.
"""

import numpy as np
import pandas as pd
import luto.settings as settings 

from luto.data import Data
from luto.economics.agricultural.quantity import get_yield_pot, get_quantity, lvs_veg_types
from luto.economics.agricultural.ghg import get_savanna_burning_effect_g_mrj

def get_rev_crop( data:Data         # Data object.
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
    yr_cal = data.YR_CAL_BASE + yr_idx
    # Check if land-use exists in AGEC_CROPS (e.g., dryland Pears/Rice do not occur), if not return zeros
    if lu not in data.AGEC_CROPS['P1', lm].columns:
        rev_t = np.zeros((data.NCELLS)).astype(np.float32)
        
    else:
        rev_multiplier = 1
        if lu in data.CROP_PRICE_MULTIPLIERS.columns:
            rev_multiplier = data.CROP_PRICE_MULTIPLIERS.loc[data.YR_CAL_BASE + yr_idx, lu]
        
        elasticity_multiplier = 1
        if settings.DYNAMIC_PRICE:
            elasticity_multiplier = 1 + (data.DEMAND_DELTA.sel(YEAR=yr_cal, COMMODITY=lu.lower()) / data.DEMAND_ELASTICITY[lu.lower()]).values
            
        # Revenue in $ per cell (includes REAL_AREA via get_quantity)
        rev_t = ( data.AGEC_CROPS['P1', lm, lu] 
                * get_quantity( data, lu.upper(), lm, yr_idx )  # lu.upper() only for crops as needs to be in product format in get_quantity().
                * rev_multiplier
                * elasticity_multiplier
                ).values
    
    # Return revenue as MultiIndexed DataFrame.
    return pd.DataFrame(rev_t, columns=pd.MultiIndex.from_tuples([(lu, lm, 'Crop')]))

def get_rev_lvstk( data:Data   # Data object.
                 , lu           # Land use.
                 , lm           # Land management.
                 , yr_idx       # Number of years post base-year ('YR_CAL_BASE').
                 ):
    """Return livestock revenue [AUD/cell] of `lu`+`lm` in `yr_idx` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals' or 'Beef - natural land').
    `lm`: land management (e.g. 'dry', 'irr').
    `yr_idx`: number of years from base year, counting from zero."""
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    # Get livestock and vegetation type.
    lvstype, vegtype = lvs_veg_types(lu)

    # Get the yield potential, i.e. the total number of heads per hectare.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)
    
    elasticity_multiplier = {
        'beef lexp': 1,
        'beef meat': 1,
        'sheep lexp': 1,
        'sheep meat': 1,
        'sheep wool': 1,
        'dairy': 1,
    }
    # Get the dynamic price elasticity multiplier if required
    if settings.DYNAMIC_PRICE:
        elasticity_multiplier = {
            'beef lexp': 1 + (data.DEMAND_DELTA.sel(YEAR=yr_cal, COMMODITY='beef lexp') / data.DEMAND_ELASTICITY['beef lexp']).values,
            'beef meat': 1 + (data.DEMAND_DELTA.sel(YEAR=yr_cal, COMMODITY='beef meat') / data.DEMAND_ELASTICITY['beef meat']).values,
            'sheep lexp': 1 + (data.DEMAND_DELTA.sel(YEAR=yr_cal, COMMODITY='sheep lexp') / data.DEMAND_ELASTICITY['sheep lexp']).values,
            'sheep meat': 1 + (data.DEMAND_DELTA.sel(YEAR=yr_cal, COMMODITY='sheep meat') / data.DEMAND_ELASTICITY['sheep meat']).values,
            'sheep wool': 1 + (data.DEMAND_DELTA.sel(YEAR=yr_cal, COMMODITY='sheep wool') / data.DEMAND_ELASTICITY['sheep wool']).values,
            'dairy': 1 + (data.DEMAND_DELTA.sel(YEAR=yr_cal, COMMODITY='dairy') / data.DEMAND_ELASTICITY['dairy']).values,
        }
            

    # Revenue in $ per cell (includes RESMULT via get_quantity)
    if lvstype == 'BEEF':

        # Get the revenue from meat and live exports. Set to zero if not produced.
        rev_meat = yield_pot * (                               # Stocking density (head/ha)
                            ( data.AGEC_LVSTK['F1', lvstype]   # Fraction of herd producing (0 - 1)
                            * data.AGEC_LVSTK['Q1', lvstype]   # Quantity produced per head (meat tonnes/head)
                            * data.AGEC_LVSTK['P1', lvstype] ) # Price per unit quantity ($/tonne of meat)
                            * data.LVSTK_PRICE_MULTIPLIERS.loc[yr_cal, "BEEF P1"] # Multiplier for commodity price
                            * elasticity_multiplier['beef meat']  # Dynamic price elasticity multiplier
                            )

        rev_lexp = yield_pot * (  
                            ( data.AGEC_LVSTK['F3', lvstype]   # Fraction of herd producing (0 - 1)
                            * data.AGEC_LVSTK['Q3', lvstype]   # Quantity produced per head (animal weight tonnes/head)
                            * data.AGEC_LVSTK['P3', lvstype] ) # Price per unit quantity ($/tonne of animal)
                            * data.LVSTK_PRICE_MULTIPLIERS.loc[yr_cal, "BEEF P3"] # Multiplier for commodity price
                            * elasticity_multiplier['beef lexp']  # Dynamic price elasticity multiplier
                            )  

        # Set Wool and Milk to zero as they are not produced by beef cattle
        rev_wool = rev_milk = np.zeros((data.NCELLS)).astype(np.float32)

    elif lvstype == 'SHEEP':    

        # Get the revenue from meat, wool and live exports. Set to zero if not produced.
        rev_meat = yield_pot * (  # Meat                           # Stocking density (head/ha)
                                ( data.AGEC_LVSTK['F1', lvstype]   # Fraction of herd producing (0 - 1)
                                * data.AGEC_LVSTK['Q1', lvstype]   # Quantity produced per head (meat tonnes/head)
                                * data.AGEC_LVSTK['P1', lvstype] ) # Price per unit quantity ($/tonne of meat)
                                * data.LVSTK_PRICE_MULTIPLIERS.loc[yr_cal, "SHEEP P1"] # Multiplier for commodity price
                                * elasticity_multiplier['sheep meat']  # Dynamic price elasticity multiplier
                                )
        rev_wool = yield_pot * (  # Wool                           # Stocking density (head/ha) 
                                ( data.AGEC_LVSTK['F2', lvstype]   # Fraction of herd producing (0 - 1) 
                                * data.AGEC_LVSTK['Q2', lvstype]   # Quantity produced per head (wool tonnes/head)
                                * data.AGEC_LVSTK['P2', lvstype] ) # Price per unit quantity ($/tonne wool)
                                * data.LVSTK_PRICE_MULTIPLIERS.loc[yr_cal, "SHEEP P2"] # Multiplier for commodity price
                                * elasticity_multiplier['sheep wool']  # Dynamic price elasticity multiplier
                                )   

        rev_lexp = yield_pot * (  # Live exports                   # Stocking density (head/ha)
                                ( data.AGEC_LVSTK['F3', lvstype]   # Fraction of herd producing (0 - 1) 
                                * data.AGEC_LVSTK['Q3', lvstype]   # Quantity produced per head (animal weight tonnes/head)
                                * data.AGEC_LVSTK['P3', lvstype] ) # Price per unit quantity ($/tonne of whole animal)
                                * data.LVSTK_PRICE_MULTIPLIERS.loc[yr_cal, "SHEEP P3"] # Multiplier for commodity price
                                * elasticity_multiplier['sheep lexp']  # Dynamic price elasticity multiplier
                                )

        # Set Milk to zero as it is not produced by sheep
        rev_milk = np.zeros((data.NCELLS)).astype(np.float32) # Set Milk to zero as it is not produced by sheep


    elif lvstype == 'DAIRY':

        # Get the revenue from milk. Set to zero if not produced.
        rev_milk = yield_pot * (  # Milk                           # Stocking density (head/ha)
                                ( data.AGEC_LVSTK['F1', lvstype]   # Fraction of herd producing (0 - 1) 
                                * data.AGEC_LVSTK['Q1', lvstype]   # Quantity produced per head (milk litres/head)
                                * data.AGEC_LVSTK['P1', lvstype] ) # Price per unit quantity ($/litre milk)
                                * data.LVSTK_PRICE_MULTIPLIERS.loc[yr_cal, "DAIRY P1"] # Multiplier for commodity price
                                * elasticity_multiplier['dairy']  # Dynamic price elasticity multiplier
                                )

        # Set Meat, Wool and Live exports to zero
        rev_meat = rev_wool = rev_lexp = np.zeros((data.NCELLS)).astype(np.float32)

    else:  # Livestock type is unknown.
        raise KeyError(f"Unknown {lvstype} livestock type. Check `lvstype`.")   

    # Put the revenues into a MultiIndex DataFrame
    rev_seperate = pd.DataFrame(
        np.stack((rev_meat, rev_wool, rev_lexp, rev_milk), axis=1), 
        columns=pd.MultiIndex.from_product([[lu], [lm], ['Meat', 'Wool', 'Live Exports', 'Milk']])
    )

    # Revenue so far in AUD/ha. Now convert to AUD/cell including resfactor.
    rev_seperate = rev_seperate * data.REAL_AREA.reshape(-1, 1) # Convert to AUD/cell

    # Return revenue as numpy array.
    return rev_seperate


def get_rev( data:Data    # Data object.
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

    elif lu in data.LU_LVSTK:
        return get_rev_lvstk(data, lu, lm, yr_idx)

    elif lu in data.AGRICULTURAL_LANDUSES:
        return pd.DataFrame(
            np.zeros((data.NCELLS, 1)).astype(np.float32),
            columns=pd.MultiIndex.from_tuples([(lu, lm, 'Unallocated Land')])
        )
    else:
        raise KeyError(f"Land-use '{lu}' not found in data.LANDUSES")


def get_rev_matrix(data:Data, lm, yr_idx):
    """Return r_rj matrix of revenue/cell per lu under `lm` in `yr_idx`."""
    
    # Concatenate the revenue from each land use into a single Multiindex DataFrame.
    r_rjs = pd.concat(
        [get_rev(data, lu, lm, yr_idx) for lu in data.AGRICULTURAL_LANDUSES], 
        axis=1
    )
    r_rjs = r_rjs.fillna(0)
    return r_rjs


def get_rev_matrices(data:Data, yr_idx, aggregate:bool = True):
    """Return r_mrj matrix of revenue per cell as 3D Numpy array."""

    # Concatenate the revenue from each land management into a single Multiindex DataFrame.
    rev_rjms = pd.concat(
        [get_rev_matrix(data, lm, yr_idx) for lm in data.LANDMANS], 
        axis=1
    )

    if not aggregate:
        # Concatenate the revenue from each land management into a single Multiindex DataFrame.
        return rev_rjms
    
    df_jmr = rev_rjms.T.groupby(level=[0, 1]).sum()
    arr_jmr = df_jmr.values.reshape(*(list(df_jmr.index.levshape) + [-1]))
    return np.einsum('jmr->mrj', arr_jmr)


def get_commodity_prices(data: Data) -> np.ndarray:
    '''
    Get the prices of commodities in the base year. These prices will be used as multiplier
    to weight deviatios of commodity production from the target.
    '''
    
    commodity_lookup = {
        ('P1','BEEF'): 'beef meat',
        ('P3','BEEF'): 'beef lexp',
        ('P1','SHEEP'): 'sheep meat',
        ('P2','SHEEP'): 'sheep wool',
        ('P3','SHEEP'): 'sheep lexp',
        ('P1','DAIRY'): 'dairy',
    }

    commodity_prices = {}

    # Get the median price of each commodity
    for names, commodity in commodity_lookup.items():
        prices = np.nanpercentile(data.AGEC_LVSTK[names[0], names[1]], 50)
        prices = prices * 1000 if commodity == 'dairy' else prices # convert to per tonne for dairy
        commodity_prices[commodity] = prices

    # Get the median price of each crop; here need to use 'irr' because dry-Rice does exist in the data
    for name, col in data.AGEC_CROPS['P1','irr'].items():
        commodity_prices[name.lower()] = np.nanpercentile(col, 50)

    return np.array([commodity_prices[k] for k in data.COMMODITIES])


def get_asparagopsis_effect_r_mrj(data:Data, r_mrj, yr_idx):
    """
    Applies the effects of using asparagopsis to the revenue data
    for all relevant agr. land uses.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_r_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['Asparagopsis taxiformis']:
        return new_r_mrj

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in quantity produced
            new_r_mrj[:, :, lu_idx] = r_mrj[:, :, j] * (multiplier - 1)

    return new_r_mrj


def get_precision_agriculture_effect_r_mrj(data:Data, r_mrj, yr_idx):
    """
    Applies the effects of using precision agriculture to the revenue data
    for all relevant agr. land uses.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_r_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['Precision Agriculture']:
        return new_r_mrj

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.PRECISION_AGRICULTURE_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            j = lu_codes[lu_idx]
            new_r_mrj[:, :, lu_idx] = r_mrj[:, :, j] * (multiplier - 1)

    return new_r_mrj


def get_ecological_grazing_effect_r_mrj(data:Data, r_mrj, yr_idx):
    """
    Applies the effects of using ecologiacl grazing to the revenue data
    for all relevant agr. land uses.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_r_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['Ecological Grazing']:
        return new_r_mrj

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            j = lu_codes[lu_idx]
            new_r_mrj[:, :, lu_idx] = r_mrj[:, :, j] * (multiplier - 1)

    return new_r_mrj


def get_savanna_burning_effect_r_mrj(data:Data, yr_idx: int):
    """
    Applies the effects of using EDS savanna burning to the revenue data
    for all relevant agr. land uses.

    Since EDSSB has no effect on revenue, return an array of zeros.
    """
    ghg_effect = get_savanna_burning_effect_g_mrj(data)
    return ghg_effect * data.get_carbon_price_by_yr_idx(yr_idx)


def get_agtech_ei_effect_r_mrj(data:Data, r_mrj, yr_idx):
    """
    Applies the effects of using AgTech EI to the revenue data
    for all relevant agr. land uses.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_r_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['AgTech EI']:
        return new_r_mrj

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.AGTECH_EI_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            j = lu_codes[lu_idx]
            new_r_mrj[:, :, lu_idx] = r_mrj[:, :, j] * (multiplier - 1)

    return new_r_mrj


def get_biochar_effect_r_mrj(data:Data, r_mrj, yr_idx):
    """
    Applies the effects of using Biochar to the revenue data
    for all relevant agr. land uses.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_r_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['Biochar']:
        return new_r_mrj

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.BIOCHAR_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            j = lu_codes[lu_idx]
            new_r_mrj[:, :, lu_idx] = r_mrj[:, :, j] * (multiplier - 1)

    return new_r_mrj



def get_beef_hir_effect_r_mrj(data: Data, r_mrj):
    """
    Applies the effects of using HIR to the beef revenue data
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]

    # Set up the effects matrix
    r_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['HIR - Beef']:
        return r_mrj_effect
    
    # Update values in the new matrix    
    for lu_idx in range(len(land_uses)):
        j = lu_codes[lu_idx]
        r_mrj_effect[:, :, lu_idx] = r_mrj[:, :, j] * (settings.HIR_PRODUCTIVITY_CONTRIBUTION - 1)

    return r_mrj_effect


def get_sheep_hir_effect_r_mrj(data: Data, r_mrj):
    """
    Applies the effects of using HIR to the sheep revenue data
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]

    # Set up the effects matrix
    r_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['HIR - Sheep']:
        return r_mrj_effect
    
    # Update values in the new matrix    
    for lu_idx in range(len(land_uses)):
        j = lu_codes[lu_idx]
        r_mrj_effect[:, :, lu_idx] = r_mrj[:, :, j] * (settings.HIR_PRODUCTIVITY_CONTRIBUTION - 1)

    return r_mrj_effect


def get_agricultural_management_revenue_matrices(data:Data, r_mrj, yr_idx) -> dict[str, np.ndarray]:
    """
    Calculate the revenue matrices for different agricultural management practices.

    Args:
        data: The input data for revenue calculation.
        r_mrj: The value of r_mrj parameter.
        yr_idx: The index of the year.

    Returns
        A dictionary containing revenue matrices for different agricultural management practices.
        The keys of the dictionary represent the management practices, and the values are numpy arrays.

    """
    ag_mam_r_mrj = {}

    ag_mam_r_mrj['Asparagopsis taxiformis'] = get_asparagopsis_effect_r_mrj(data, r_mrj, yr_idx)           
    ag_mam_r_mrj['Precision Agriculture'] = get_precision_agriculture_effect_r_mrj(data, r_mrj, yr_idx)  
    ag_mam_r_mrj['Ecological Grazing'] = get_ecological_grazing_effect_r_mrj(data, r_mrj, yr_idx)          
    ag_mam_r_mrj['Savanna Burning'] = get_savanna_burning_effect_r_mrj(data, yr_idx)                       
    ag_mam_r_mrj['AgTech EI'] = get_agtech_ei_effect_r_mrj(data, r_mrj, yr_idx)                            
    ag_mam_r_mrj['Biochar'] = get_biochar_effect_r_mrj(data, r_mrj, yr_idx)                                
    ag_mam_r_mrj['HIR - Beef'] = get_beef_hir_effect_r_mrj(data, r_mrj)                                    
    ag_mam_r_mrj['HIR - Sheep'] = get_sheep_hir_effect_r_mrj(data, r_mrj)                                  

    return ag_mam_r_mrj
