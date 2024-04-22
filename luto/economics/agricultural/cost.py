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



import itertools
import numpy as np
import pandas as pd

from luto.economics.agricultural.quantity import get_yield_pot, lvs_veg_types, get_quantity
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.data import Data, lvs_veg_types
from luto.economics.agricultural.quantity import get_yield_pot, get_quantity


def get_cost_crop(data: Data, lu, lm, yr_idx):
    """Return crop production cost <unit: $/cell> of `lu`+`lm` in `yr_idx` as np array.

    Args:
        data (object/module): Data object or module. Assumes fields like in `luto.data`.
        lu (str): Land use (e.g., 'Winter cereals' or 'Beef - natural land').
        lm (str): Land management (e.g., 'dry', 'irr').
        yr_idx (int): Number of years post base-year ('YR_CAL_BASE').

    Returns:
        pandas.DataFrame: Costs of the crop as a numpy array, with columns representing different cost types.

    Raises:
        KeyError: If the passed `lm` is neither 'dry' nor 'irr'.

    Notes:
        - If the land-use does not exist in AGEC_CROPS, zeros are returned.

    """
    
    # Check if land-use exists in AGEC_CROPS (e.g., dryland Pears/Rice do not occur), if not return zeros
    if lu not in data.AGEC_CROPS['AC', lm].columns:
        costs_t = np.zeros((data.NCELLS))
        # The column name is irrelevant and only used to make the out df the same shape as the rest of crops.
        return pd.DataFrame(costs_t,
                            columns=pd.MultiIndex.from_product([[lu], [lm], ['Area cost']]))

    else: # Calculate the total costs 
            
        # Variable costs (quantity costs and area costs)        
        # Quantity costs (calculated as cost per tonne x tonne per cell x resfactor)
        costs_q = ( data.AGEC_CROPS['QC', lm, lu]
                  * get_quantity(data, lu.upper(), lm, yr_idx) )  # lu.upper() only for crops as needs to be in product format in get_quantity().  

        # Area costs.
        costs_a = data.AGEC_CROPS['AC', lm, lu]

        # Fixed costs
        costs_f = ( data.AGEC_CROPS['FLC', lm, lu]    # Fixed labour costs.
                  + data.AGEC_CROPS['FOC', lm, lu]    # Fixed operating costs.
                  + data.AGEC_CROPS['FDC', lm, lu] )  # Fixed depreciation costs.

        # Water costs as water required in ML per hectare x delivery price per ML.
        if lm == 'irr':
            costs_w = data.AGEC_CROPS['WR', lm, lu] * data.AGEC_CROPS['WP', lm, lu]
        elif lm == 'dry':
            costs_w = 0
        else: # Passed lm is neither `dry` nor `irr`.
            raise KeyError(f"Unknown {lm} land management. Check `lm` key.")

        # Convert to $/cell including resfactor.
        # Quantity costs which has already been adjusted for REAL_AREA/resfactor via get_quantity
        costs_a, costs_f, costs_w = costs_a * data.REAL_AREA, costs_f * data.REAL_AREA, costs_w * data.REAL_AREA

        costs_t = np.stack([(costs_a), (costs_f), (costs_w), (costs_q)]).T


        # Return costs as numpy array.
        return pd.DataFrame(costs_t,
                            columns=pd.MultiIndex.from_product([[lu], [lm], ['Area cost', 'Fixed cost', 'Water cost', 'Quantity cost']]))


def get_cost_lvstk(data: Data, lu, lm, yr_idx):
    """Return lvstk prodution cost <unit: $/cell> of `lu`+`lm` in `yr_idx` as np array.

    Args:
        data (object/module): Data object or module.
        lu (str): Land use (e.g. 'Winter cereals' or 'Beef - natural land').
        lm (str): Land management (e.g. 'dry', 'irr').
        yr_idx (int): Number of years post base-year ('YR_CAL_BASE').

    Returns:
        pandas.DataFrame: Costs as numpy array, with columns representing different cost types.

    Raises:
        KeyError: If the passed `lm` is neither 'dry' nor 'irr'.

    """
    # Get livestock and vegetation type.
    lvstype, vegtype = lvs_veg_types(lu)

    # Get the yield potential, i.e. the total number of head per hectare.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)

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
        raise KeyError(f"Unknown {lm} land management. Check `lm` key.")

    # Water delivery costs equal drinking water plus irrigation water req per head * yield (head/ha)
    costs_w = (data.AGEC_LVSTK['WR_DRN', lvstype] + WR_IRR) * yield_pot
    costs_w *= data.WATER_DELIVERY_PRICE  # $/ha

    # Convert costs to $ per cell including resfactor.
    cost_a, cost_f, cost_w, cost_q = costs_a*data.REAL_AREA, costs_f*data.REAL_AREA,\
                                     costs_w*data.REAL_AREA, costs_q*data.REAL_AREA

    costs = np.stack([(cost_a), (cost_f), (cost_w), (cost_q)]).T

    # Return costs as numpy array.
    return  pd.DataFrame(costs,
                         columns=pd.MultiIndex.from_product([[lu], [lm], ['Area cost', 'Fixed cost', 'Water cost', 'Quantity cost']]))


def get_cost(data: Data, lu, lm, yr_idx):
    """Return production cost <unit: $/cell> of `lu`+`lm` in `yr_idx` as np array.

    Args:
        data (object/module): Data object or module. Assumes fields like in `luto.data`.
        lu (str): Land use (e.g. 'Winter cereals').
        lm (str): Land management (e.g. 'dry', 'irr', 'org').
        yr_idx (int): Number of years post base-year ('YR_CAL_BASE').

    Returns:
        np.array: Production cost <unit: $/cell> of `lu`+`lm` in `yr_idx`.

    Raises:
        KeyError: If land use `lu` is not found in `data.LANDUSES`.
    """
    # If it is a crop, it is known how to get the costs.
    if lu in data.LU_CROPS:
        return get_cost_crop(data, lu, lm, yr_idx)

    elif lu in data.LU_LVSTK:
        return get_cost_lvstk(data, lu, lm, yr_idx)

    elif lu in data.AGRICULTURAL_LANDUSES:
        return pd.DataFrame(np.zeros(data.NCELLS),
                            columns=pd.MultiIndex.from_product([[lu], [lm], ['Area cost']]))

    else:
        raise KeyError(f"Land use '{lu}' not found in any LANDUSES")


def get_cost_matrix(data: Data, lm, yr_idx):
    """
    Return agricultural c_rj matrix <unit: $/cell> per lu under `lm` in `yr_idx`.
    
    Parameters:
        data (DataFrame): The input data containing agricultural land use information.
        lm (str): The land use type.
        yr_idx (int): The index of the year.
    
    Returns:
        DataFrame: The cost matrix representing the costs per cell for each land use under the given land use type and year index.
    """
    
    cost = pd.concat([get_cost(data, lu, lm, yr_idx) for lu in data.AGRICULTURAL_LANDUSES], axis=1)
        
    # Make sure all NaNs are replaced by zeroes.
    return cost.fillna(0)


def get_cost_matrices(data: Data, yr_idx, aggregate=True):
    """
    Return agricultural c_mrj matrix <unit: $/cell> as 3D Numpy array.

    Parameters:
    - data: The input data containing information about land management.
    - yr_idx: The index of the year.
    - aggregate: A boolean value indicating whether to aggregate the cost matrices or not. Default is True.

    Returns:
    - If aggregate is True, returns a 3D Numpy array representing the aggregated cost matrix.
    - If aggregate is False, returns a pandas DataFrame representing the cost matrix.

    Raises:
    - ValueError: If the value of aggregate is neither True nor False.
    """
    
    # Concatenate the revenue from each land management into a single Multiindex DataFrame.
    cost_rjms = pd.concat([get_cost_matrix(data, lm, yr_idx) for lm in data.LANDMANS], axis=1)

    # Reorder the columns to match the multi-level dimension of r*jms.
    cost_rjms = cost_rjms.reindex(columns=pd.MultiIndex.from_product(cost_rjms.columns.levels), fill_value=0)

    if aggregate == True:
        j,m,s = cost_rjms.columns.levshape
        c_rjms = cost_rjms.values.reshape(-1,j,m,s)
        return np.einsum('rjms->mrj', c_rjms)
    elif aggregate == False:
        return cost_rjms

    else:
        raise ValueError("aggregate must be True or False")


def get_asparagopsis_effect_c_mrj(data: Data, yr_idx):
    """
    Applies the effects of using asparagopsis to the cost data
    for all relevant agricultural land uses.

    Parameters:
    - data: The data object containing relevant information.
    - yr_idx: The index of the year.

    Returns:
    - new_c_mrj: The updated cost matrix <unit: $/cell>.

    This function calculates the cost per cell for each relevant agricultural land use
    when using asparagopsis. It takes into account the yield potential, annual cost per
    animal, and the real area of the cell. The function returns the updated cost matrix.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_c_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix
    for lm in data.LANDMANS:
        m = 0 if lm == 'dry' else 1
        for lu_idx, lu in enumerate(land_uses):
            lvstype, vegtype = lvs_veg_types(lu)
            yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)
            cost_per_animal = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, 'Annual Cost Per Animal (A$2010/yr)']
            cost_per_cell = cost_per_animal * yield_pot * data.REAL_AREA

            new_c_mrj[m, :, lu_idx] = cost_per_cell

    return new_c_mrj


def get_precision_agriculture_effect_c_mrj(data: Data, yr_idx):
    """
    Applies the effects of using precision agriculture to the cost data
    for all relevant agr. land uses.

    Parameters:
    - data: The data object containing the necessary information.
    - yr_idx: The index of the year to calculate the effects for.

    Returns:
    - new_c_mrj: The updated cost data matrix <unit: $/cell>.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_c_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    for m in range(data.NLMS):
        for lu_idx, lu in enumerate(land_uses):
            cost_per_ha = data.PRECISION_AGRICULTURE_DATA[lu].loc[yr_cal, 'AnnCost_per_Ha']
            new_c_mrj[m, :, lu_idx] = cost_per_ha * data.REAL_AREA

    return new_c_mrj


def get_ecological_grazing_effect_c_mrj(data: Data, yr_idx):
    """
    Applies the effects of using ecological grazing to the cost data
    for all relevant agr. land uses.

    Parameters:
    - data: The data object containing the necessary input data.
    - yr_idx: The index of the year for which the effects are calculated.

    Returns:
    - new_c_mrj: The matrix containing the updated cost <unit: $/cell>.
    """

    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_c_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    for lu_idx, lu in enumerate(land_uses):
        lvstype, _ = lvs_veg_types(lu)

        # Get effects on operating costs
        operating_mult = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, 'Operating_cost_multiplier']
        operating_c_effect = data.AGEC_LVSTK['FOC', lvstype] * (operating_mult - 1) * data.REAL_AREA

        # Get effects on labour costs
        labour_mult = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, 'Labour_cost_mulitiplier']
        labour_c_effect = data.AGEC_LVSTK['FLC', lvstype] * (labour_mult - 1) * data.REAL_AREA

        # Combine for total cost effect
        total_c_effect = operating_c_effect + labour_c_effect

        for m in range(data.NLMS):
            new_c_mrj[m, :, lu_idx] = total_c_effect

    return new_c_mrj


def get_savanna_burning_effect_c_mrj(data: Data):
    """
    Applies the effects of using LDS Savanna Burning to the cost data
    for all relevant agr. land uses.

    Parameters:
    - data: The input data containing relevant information for calculations.

    Returns:
    - new_c_mrj: The modified cost data <unit: $/cell>.
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Savanna Burning"])
    new_c_mrj = np.zeros((data.NLMS, data.NCELLS, nlus))
    sav_burning_effect = data.SAVBURN_COST_HA * data.REAL_AREA

    big_number = 999999999999
    savburn_ineligible_cells = np.where(data.SAVBURN_ELIGIBLE == 0)[0]

    for m, j in itertools.product(range(data.NLMS), range(nlus)):
        new_c_mrj[m, :, j] = sav_burning_effect

        # TODO: build in hard constraints (ub for variables) instead of this temorary measure
        # to block certain cells from using Savanna Burning
        new_c_mrj[m, savburn_ineligible_cells, j] = big_number

    return new_c_mrj


def get_agtech_ei_effect_c_mrj(data: Data, yr_idx):
    """
    Applies the effects of using AgTech EI to the cost data
    for all relevant agr. land uses.

    Parameters:
    - data: The data object containing the necessary information.
    - yr_idx: The index of the year.

    Returns:
    - new_c_mrj: The updated cost data <unit: $/cell>.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_c_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    for m in range(data.NLMS):
        for lu_idx, lu in enumerate(land_uses):
            cost_per_ha = data.AGTECH_EI_DATA[lu].loc[yr_cal, 'AnnCost_per_Ha']
            new_c_mrj[m, :, lu_idx] = cost_per_ha * data.REAL_AREA

    return new_c_mrj


def get_agricultural_management_cost_matrices(data: Data, c_mrj, yr_idx):
    """
    Calculate the cost matrices for different agricultural management practices.

    Args:
        data (dict): The input data for cost calculations.
        c_mrj (float): The cost of marginal reduction in emissions.
        yr_idx (int): The index of the year.

    Returns:
        dict: A dictionary containing the cost matrices for different agricultural management practices.
            The keys are the names of the practices and the values are the corresponding cost matrices.
    """
    asparagopsis_data = get_asparagopsis_effect_c_mrj(data, yr_idx)
    precision_agriculture_data = get_precision_agriculture_effect_c_mrj(data, yr_idx)
    eco_grazing_data = get_ecological_grazing_effect_c_mrj(data, yr_idx)
    sav_burning_data = get_savanna_burning_effect_c_mrj(data)
    agtech_ei_data = get_agtech_ei_effect_c_mrj(data, yr_idx)

    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
    }
