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
Pure functions for calculating the production quantities of agricutlural commodities.
"""

from typing import Dict
import numpy as np
from scipy.interpolate import interp1d

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.data import Data, lvs_veg_types


def get_ccimpact(data: Data, lu, lm, yr_idx):
    """
    Return climate change impact multiplier at (zero-based) year index.

    Parameters:
    - data: The data object containing climate change impact data.
    - lu: The land-use for which the climate change impact is calculated.
    - lm: The land-management for which the climate change impact is calculated.
    - yr_idx: The zero-based index of the year for which the climate change impact is calculated.

    Returns:
    - The climate change impact multiplier at the specified year index.
    """

    # Check if land-use exists in CLIMATE_CHANGE_IMPACT (e.g., dryland Pears/Rice do not occur), if not return ones
    if lu not in {t[0] for t in data.CLIMATE_CHANGE_IMPACT[lm].columns}:
        return np.ones((data.NCELLS))

    # Convert year index to calendar year to match the climate impact data which is by calendar year.
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Interpolate climate change damage for lu, lm, and year for each cell using a linear function.
    xs = {t[2] for t in data.CLIMATE_CHANGE_IMPACT.columns}  # Returns set {2020, 2050, 2080}
    xs.add(2010)                                             # Adds the year 2010 and returns set {2010, 2020, 2050, 2080}
    xs = sorted(xs)                                          # Returns list and ensures sorted lowest to highest [2010, 2020, 2050, 2080]
    yys = data.CLIMATE_CHANGE_IMPACT[lm, lu].fillna(1)       # Grabs the column and replaces NaNs with ones to avoid issues with calculating water use limits
    yys.insert(0, '2010', 1)                                 # Insert a new column for 2010 with value of 1 to ensure no climate change impact at 2010
    yys = yys.astype(np.float32)

    # Create linear function f and interpolate climate change impact
    f = interp1d(xs, yys, kind='linear', fill_value='extrapolate')
    return f(yr_cal)


def get_yield_pot(data, lvstype, vegtype, lm, yr_idx):
    """
    Return the yield potential <unit: head/ha> for livestock by land cover type.

    Parameters:
    - data: Data object or module.
    - lvstype: Livestock type (one of 'BEEF', 'SHEEP' or 'DAIRY').
    - vegtype: Vegetation type (one of 'natural land' or 'modified land').
    - lm: Land-management type.
    - yr_idx: Number of years post 2010.

    Returns:
    - yield_pot: The yield potential <unit: head/ha>.
    """

    # Factors varying as a function of `lvstype`.
    dse_per_head = {'BEEF': 8, 'SHEEP': 1.5, 'DAIRY': 17}
    grassfed_factor = {'BEEF': 0.85, 'SHEEP': 0.85, 'DAIRY': 0.65}
    denominator = (365 * dse_per_head[lvstype] * grassfed_factor[lvstype])

    # Base potential.
    yield_pot = data.FEED_REQ * data.PASTURE_KG_DM_HA / denominator

    # Multiply potential by appropriate SAFE_PUR (safe pasture utilisation rate).
    if vegtype == 'natural land':
        yield_pot *= data.SAFE_PUR_NATL
    elif vegtype == 'modified land':
        yield_pot *= data.SAFE_PUR_MODL
    else:
        raise KeyError(f"Land cover type '{vegtype}' not identified.")

    # Multiply livestock yield potential by appropriate irrigation factor (i.e., 2).
    if lm == 'irr':
        yield_pot *= 2

    # Apply climate change yield impact multiplier. Essentially changes the number of head per hectare by a multiplier i.e., 1.2 = a 20% increase.
    lu = f'{lvstype.capitalize()} - {vegtype}'
    yield_pot *= get_ccimpact(data, lu, lm, yr_idx)

    # Here we can add a productivity multiplier for sustainable intensification to increase pasture growth and yield potential (i.e., head/ha)
    # yield_pot *= yield_mult  ***Still to do***

    return yield_pot


def get_quantity_lvstk(data: Data, pr, lm, yr_idx):
    """Return livestock yield of `pr`+`lm` in `yr_idx` as 1D Numpy array.

    Args:
        data (object/module): Data object or module.
        pr (str): Livestock + product like 'SHEEP - MODIFIED LAND WOOL'.
        lm (str): Land management.
        yr_idx (int): Number of years post base-year ('YR_CAL_BASE').

    Returns:
        numpy.ndarray: Livestock yield of `pr`+`lm` in `yr_idx` as 1D Numpy array.

    Units:
        - <unit: t/cell> BEEF and SHEEP meat (1) and live exports (3).
        - <unit: t/cell> SHEEP wool (2).
        - <unit: kilolitres/cell> DAIRY (1).
    """

    # Get livestock and land cover type.
    lvstype, vegtype = lvs_veg_types(pr)

    # Get the yield potential.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)

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
            raise KeyError(f"Unknown {lvstype} product. Check `pr` key.")

    elif lvstype == 'SHEEP': # (F1 * Q1), (F2 * Q2), (F3 * Q3).
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
            raise KeyError(f"Unknown {lvstype} product. Check `pr` key.")

    elif lvstype == 'DAIRY': # (F1 * Q1).
        if 'DAIRY' in pr: 
            quantity = ( data.AGEC_LVSTK['F1', lvstype]
                       * data.AGEC_LVSTK['Q1', lvstype] 
                       / 1000 ) # Convert to KL
        else:
            raise KeyError(f"Unknown {lvstype} product. Check `pr` key.")

    else:
        raise KeyError(f"Livestock type '{lvstype}' not identified.")

    # Quantity is base quantity times the yield potential. yield_pot includes climate change impacts.
    quantity *= yield_pot

    # Convert quantities in tonnes/ha to tonnes/cell including real_area and resfactor.
    quantity *= data.REAL_AREA

    return quantity


def get_quantity_crop(data: Data, pr, lm, yr_idx):
    """Return crop yield <unit: t/cell> of `pr`+`lm` in `yr_idx` as 1D Numpy array.

    Args:
        data (object/module): Data object or module.
        pr (str): Product -- equivalent to land use for crops.
        lm (str): Land management.
        yr_idx (int): Number of years post base-year ('YR_CAL_BASE').

    Returns:
        numpy.ndarray: 1D Numpy array containing crop yield <unit: t/cell>.

    Raises:
        None.

    Notes:
        - `data` assumes fields like in `luto.data`.
        - `pr` is the product equivalent to land use for crops (e.g., 'winterCereals').
        - `lm` is the land management (e.g., 'dry', 'irr').
        - `yr_idx` is the number of years from the base year, counting from zero.
    """
    
    # Check if land-use exists in AGEC_CROPS (e.g., dryland Pears/Rice do not occur), if not return zeros
    if pr not in data.AGEC_CROPS['Yield', lm].columns:
        quantity = np.zeros((data.NCELLS))
        
    else: # Calculate the quantities
        
        # Get the raw quantities in tonnes/ha from data.
        quantity = data.AGEC_CROPS['Yield', lm, pr].copy().to_numpy()
        
        # Apply climate change yield impact multiplier. Takes land use (lu) as input rather than product (pr) but lu == pr for crops
        quantity *= get_ccimpact(data, pr, lm, yr_idx)
    
        # Convert to tonnes per cell including real_area and resfactor.
        quantity *= data.REAL_AREA 

    return quantity


def get_quantity(data: Data, pr, lm, yr_idx):
    """Return yield <unit: t/cell> of `pr`+`lm` in `yr_idx` as 1D Numpy array.

    Args:
        data (object/module): Data object or module.
        pr (str): Product produced.
        lm (str): Land management.
        yr_idx (int): Number of years post base-year ('YR_CAL_BASE').

    Returns:
        numpy.ndarray: 1D Numpy array representing the yield <unit: t/cell>.

    Raises:
        KeyError: If the land use `pr` is not found in the data.

    Notes:
        - Assumes fields like in `luto.data`.
        - If it is a crop, it is known how to get the quantities.
        - Apply productivity increase multiplier by product. Essentially, this is a total factor productivity increase.
    """
    # If it is a crop, it is known how to get the quantities.
    if pr in data.PR_CROPS:
        q = get_quantity_crop(data, pr.capitalize(), lm, yr_idx)

    elif pr in data.PR_LVSTK:
        q = get_quantity_lvstk(data, pr, lm, yr_idx)

    else:
        raise KeyError(f"Land use '{pr}' not found in data.")

    # Apply productivity increase multiplier by product. Essentially, this is a total factor productivity increase.
    q *= data.BAU_PROD_INCR[lm, pr][yr_idx]

    return q


def get_quantity_matrix(data: Data, lm, yr_idx):
    """
    Return q_rp matrix of quantities per cell per product as 2D Numpy array.

    Parameters:
    - data: The data object containing information about cells and products.
    - lm: The lm object representing the land management.
    - yr_idx: The index of the year.

    Returns:
    - q_rp: A 2D Numpy array representing the quantities per cell per product.
    """

    q_rp = np.zeros((data.NCELLS, len(data.PRODUCTS)))
    for j, pr in enumerate(data.PRODUCTS):
        q_rp[:, j] = get_quantity(data, pr, lm, yr_idx)

    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(q_rp)


def get_quantity_matrices(data, yr_idx):
    """
    Return q_mrp matrix of quantities per cell as 3D Numpy array.

    Parameters:
    - data: The input data containing information about quantities.
    - yr_idx: The index of the year.

    Returns:
    - q_mrp: A 3D Numpy array representing the matrix of quantities per cell.
    """
    return np.stack(tuple( get_quantity_matrix(data, lm, yr_idx)
                           for lm in data.LANDMANS ))


def get_asparagopsis_effect_q_mrp(data: Data, q_mrp, yr_idx):
    """
    Applies the effects of using asparagopsis to the quantity data
    for all relevant agr. land uses.

    Parameters:
    - data: The data object containing relevant information.
    - q_mrp: The quantity data for all land uses and products.
    - yr_idx: The index of the year.

    Returns:
    - new_q_mrp: The updated quantity data after applying the effects of using asparagopsis.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_q_mrp = np.zeros((data.NLMS, data.NCELLS, data.NPRS)).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu, j in zip(land_uses, lu_codes):
        multiplier = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            # Apply to all products associated with land use
            for p in range(data.NPRS):
                if data.LU2PR[p, j]:
                    # The effect is: effect value = old value * multiplier - old value
                    # E.g. a multiplier of .95 means a 5% reduction in quantity produced
                    new_q_mrp[:, :, p] = q_mrp[:, :, p] * (multiplier - 1)

    return new_q_mrp


def get_precision_agriculture_effect_q_mrp(data: Data, q_mrp, yr_idx):
    """
    Applies the effects of using precision agriculture to the quantity data
    for all relevant agricultural land uses.

    Parameters:
    - data: The data object containing relevant information.
    - q_mrp: The quantity data for all land uses.
    - yr_idx: The index of the year.

    Returns:
    - new_q_mrp: The updated quantity data after applying precision agriculture effects.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_q_mrp = np.zeros((data.NLMS, data.NCELLS, data.NPRS)).astype(np.float32)

    # Update values in the new matrix    
    for lu, j in zip(land_uses, lu_codes):
        multiplier = data.PRECISION_AGRICULTURE_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            # Apply to all products associated with land use
            for p in range(data.NPRS):
                if data.LU2PR[p, j]:
                    new_q_mrp[:, :, p] = q_mrp[:, :, p] * (multiplier - 1)

    return new_q_mrp


def get_ecological_grazing_effect_q_mrp(data: Data, q_mrp, yr_idx):
    """
    Applies the effects of using ecological grazing to the quantity data
    for all relevant agricultural land uses.

    Parameters:
    - data: The data object containing relevant information.
    - q_mrp: The quantity data to be updated.
    - yr_idx: The index of the year to be used for calculations.

    Returns:
    - new_q_mrp: The updated quantity data after applying ecological grazing effects.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_q_mrp = np.zeros((data.NLMS, data.NCELLS, data.NPRS)).astype(np.float32)

    # Update values in the new matrix    
    for lu, j in zip(land_uses, lu_codes):
        multiplier = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            # Apply to all products associated with land use
            for p in range(data.NPRS):
                if data.LU2PR[p, j]:
                    new_q_mrp[:, :, p] = q_mrp[:, :, p] * (multiplier - 1)

    return new_q_mrp


def get_savanna_burning_effect_q_mrp(data):
    """
    Applies the effects of using EDS savanna burning to the quantity data
    for all relevant agr. land uses.

    Since EDSSB has no effect on quantity produced, return an array of zeros.

    Parameters:
    - data: The input data object containing information about the model

    Returns:
    - An array of zeros with shape (NLMS, NCELLS, NPRS)
    """
    return np.zeros((data.NLMS, data.NCELLS, data.NPRS))


def get_agtech_ei_effect_q_mrp(data, q_mrp, yr_idx):
    """
    Applies the effects of using AgTech EI to the quantity data
    for all relevant agr. land uses.

    Parameters:
    - data: The data object containing relevant information.
    - q_mrp: The quantity data to be updated.
    - yr_idx: The index of the year.

    Returns:
    - new_q_mrp: The updated quantity data
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_q_mrp = np.zeros((data.NLMS, data.NCELLS, data.NPRS)).astype(np.float32)

    # Update values in the new matrix    
    for lu, j in zip(land_uses, lu_codes):
        multiplier = data.AGTECH_EI_DATA[lu].loc[yr_cal, 'Productivity']
        if multiplier != 1:
            # Apply to all products associated with land use
            for p in range(data.NPRS):
                if data.LU2PR[p, j]:
                    new_q_mrp[:, :, p] = q_mrp[:, :, p] * (multiplier - 1)

    return new_q_mrp


def get_agricultural_management_quantity_matrices(data: Data, q_mrp, yr_idx) -> Dict[str, np.ndarray]:
    """
    Calculates the quantity matrices for different agricultural management practices.

    Args:
        data: The input data for the calculations.
        q_mrp: The 3D matix coresponding to water-supply, cell, and product.
        yr_idx: The 0-based year index.

    Returns:
        A dictionary containing the quantity matrices for different agricultural management practices.
        The keys of the dictionary represent the names of the practices, and the values are the corresponding quantity matrices.
    """
    asparagopsis_data = get_asparagopsis_effect_q_mrp(data, q_mrp, yr_idx)
    precision_agriculture_data = get_precision_agriculture_effect_q_mrp(data, q_mrp, yr_idx)
    eco_grazing_data = get_ecological_grazing_effect_q_mrp(data, q_mrp, yr_idx)
    sav_burning_data = get_savanna_burning_effect_q_mrp(data)
    agtech_ei_data = get_agtech_ei_effect_q_mrp(data, q_mrp, yr_idx)

    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
    }
