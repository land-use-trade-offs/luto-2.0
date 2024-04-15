# Copyright 2023 Brett A. Bryan at Deakin University
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
Pure functions to calculate greenhouse gas emissions by lm, lu.
"""


import itertools
import numpy as np
import pandas as pd
import luto.tools as tools

from typing import Dict
from luto.economics.agricultural.quantity import get_yield_pot, lvs_veg_types
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


def get_ghg_crop(data, lu, lm, yr_idx, aggregate):
    """Return crop GHG emissions <unit: t/cell>  of `lu`+`lm` in `yr_idx` 
    as (np array|pd.DataFrame) depending on aggregate (True|False).

    Args:
        data (object/module): Data object or module. Assumes fields like in `luto.data`.
        lu (str): Land use (e.g. 'Winter cereals' or 'Beef - natural land').
        lm (str): Land management (e.g. 'dry', 'irr').
        yr_idx (int): Number of years from base year, counting from zero.
        aggregate (bool): True -> return GHG emission as np.array, False -> return GHG emission as pd.DataFrame.

    Returns:
        np.array or pd.DataFrame: Crop GHG emissions <unit: t/cell>  of `lu`+`lm` in `yr_idx`.

    Crop GHG emissions include:
        - 'CO2E_KG_HA_CHEM_APPL'
        - 'CO2E_KG_HA_CROP_MGT'
        - 'CO2E_KG_HA_CULTIV'
        - 'CO2E_KG_HA_FERT_PROD'
        - 'CO2E_KG_HA_HARVEST'
        - 'CO2E_KG_HA_IRRIG'
        - 'CO2E_KG_HA_PEST_PROD'
        - 'CO2E_KG_HA_SOIL'
        - 'CO2E_KG_HA_SOWING'
    """
    
    # Process GHG_crop only if the land-use (lu) and land management (lm) combination exists (e.g., dryland Pears/Rice do not occur)
    if lu in data.AGGHG_CROPS['CO2E_KG_HA_CHEM_APPL', lm].columns:

        # Get the data column {ghg_rs: r -> each pixel,  s -> each GHG source}
        ghg_rs = data.AGGHG_CROPS.loc[:, (slice(None), lm, lu)]

        # Convert kg CO2e per ha to tonnes. 
        ghg_rs /= 1000

        # Convert tonnes CO2 per ha to tonnes CO2 per cell including resfactor
        ghg_rs *= data.REAL_AREA[:, np.newaxis]

        # Convert to MultiIndex with levels [source, lm, lu]
        ghg_rs.columns = pd.MultiIndex.from_tuples([[col[0], lm, lu] for col in ghg_rs.columns])

        # Reset the dataframe index
        ghg_rs.reset_index(drop=True, inplace=True)

        # Return greenhouse gas emissions by individual source or summed over all sources (default)
        return ghg_rs if aggregate == False else ghg_rs.sum(axis=1).values



def get_ghg_lvstk( data        # Data object or module.
                 , lu          # Land use.
                 , lm          # Land management.
                 , yr_idx      # Number of years post base-year ('YR_CAL_BASE').
                 , aggregate): # GHG calculated as a total (for the solver) or by individual source (for writing outputs)
    """Return livestock GHG emissions <unit: t/cell>  of `lu`+`lm` in `yr_idx`
            as (np array|pd.DataFrame) depending on aggregate (True|False).

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals' or 'Beef - natural land').
    `lm`: land management (e.g. 'dry', 'irr').
    `yr_idx`: number of years from base year, counting from zero.
    `aggregate`: True -> return GHG emission as np.array 
                 False -> return GHG emission as pd.DataFrame.
    
    Livestock GHG emissions include:    
                  'CO2E_KG_HEAD_DUNG_URINE',
                  'CO2E_KG_HEAD_ELEC',
                  'CO2E_KG_HEAD_ENTERIC',
                  'CO2E_KG_HEAD_FODDER',
                  'CO2E_KG_HEAD_FUEL',
                  'CO2E_KG_HEAD_IND_LEACH_RUNOFF',
                  'CO2E_KG_HEAD_MANURE_MGT',
                  'CO2E_KG_HEAD_SEED',
    """
    
    lvstype, vegtype = lvs_veg_types(lu)

    # Get the yield potential, i.e. the total number of livestock head per hectare.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)

    # Get GHG emissions by source in kg CO2e per head of livestock. 
    # Note: ghg_rs (r -> each cell, s -> each GHG source)
    ghg_raw = data.AGGHG_LVSTK.loc[:, (lvstype, slice(None)) ]

    # Get the names for each GHG source
    ghg_name_s = [ i[1] for i in ghg_raw.columns ]

    # Calculate the GHG emissions (kgCO2/head * head/ha = kgCO/ha)
    ghg_rs = ghg_raw * yield_pot[:,np.newaxis]


    # Add pasture irrigation emissions.
    if lm == 'irr':
        ghg_lvstk_irr = data.AGGHG_IRRPAST
        ghg_lvstk_irr_cols = [i for i in ghg_lvstk_irr.columns if 'CO2E' in i]
        
        ghg_rs = pd.concat([ghg_rs, ghg_lvstk_irr[ghg_lvstk_irr_cols]], axis = 1)
        ghg_name_s += ghg_lvstk_irr_cols
        

    # Convert to tonnes of CO2e per ha. 
    ghg_rs = ghg_rs / 1000

    # Convert to tonnes CO2e per cell including resfactor
    ghg_rs *= data.REAL_AREA[:, np.newaxis]

    # Convert to MultiIndex with levels [source, lm, lu]
    ghg_rs = pd.DataFrame(ghg_rs)
    ghg_rs.columns = pd.MultiIndex.from_tuples( [(ghg, lm, lu) for ghg in ghg_name_s ])

    # Reset dataframe index
    ghg_rs.reset_index(drop = True, inplace = True)
    
    # Return the full dataframe if Aggregate == False otherwise return the sum over all GHG sources
    return ghg_rs if aggregate == False else ghg_rs.sum(axis = 1).values
       


def get_ghg(data, lu, lm, yr_idx, aggregate):
    """Return GHG emissions [tCO2e/cell] of `lu`+`lm` in `yr_idx` 
    as (np array|pd.DataFrame) depending on aggregate (True|False).

    Args:
        data (object/module): Data object or module. Assumes fields like in `luto.data`.
        lu (str): Land use (e.g. 'Winter cereals').
        lm (str): Land management (e.g. 'dry', 'irr').
        yr_idx (int): Number of years from base year, counting from zero.
        aggregate (bool): True -> return GHG emission as np.array, False -> return GHG emission as pd.DataFrame.

    Returns:
        np.array or pd.DataFrame: GHG emissions [tCO2e/cell] of `lu`+`lm` in `yr_idx`.

    Raises:
        KeyError: If land use `lu` is not found in `data.LANDUSES`.
    """

    # If it is a crop, it is known how to get GHG emissions.
    if lu in data.LU_CROPS:
        return get_ghg_crop(data, lu, lm, yr_idx, aggregate)
    elif lu in data.LU_LVSTK:
        return get_ghg_lvstk(data, lu, lm, yr_idx, aggregate)
    elif lu in data.AGRICULTURAL_LANDUSES:
        if aggregate:
            return np.zeros(data.NCELLS)
        else:
            return pd.DataFrame({('CO2E_KG_HA_CHEM_APPL', lm, lu): np.zeros(data.NCELLS)})
    else:
        raise KeyError(f"Land use '{lu}' not found in data.LANDUSES")



def get_ghg_matrix(data, lm, yr_idx, aggregate):
    """
    Return g_rj matrix <unit: t/cell> per lu under `lm` in `yr_idx`.

    Parameters:
    - data: The data object containing the necessary information.
    - lm: The land use model.
    - yr_idx: The index of the year.
    - aggregate: A boolean indicating whether to aggregate the results or not.

    Returns:
    - If `aggregate` is True, returns a numpy array of shape (NCELLS, len(data.AGRICULTURAL_LANDUSES)).
    - If `aggregate` is False, returns a pandas DataFrame with columns corresponding to each agricultural land use.

    """
    if aggregate == True: 
        g_rj = np.zeros((data.NCELLS, len(data.AGRICULTURAL_LANDUSES)))
        for j, lu in enumerate(data.AGRICULTURAL_LANDUSES):
            g_rj[:, j] = get_ghg(data, lu, lm, yr_idx, aggregate)
            
        # Make sure all NaNs are replaced by zeroes.
        g_rj = np.nan_to_num(g_rj)
    
        return g_rj
    
    elif aggregate == False:     
        return pd.concat([get_ghg(data, lu, lm, yr_idx, aggregate) 
                          for lu in data.AGRICULTURAL_LANDUSES],axis=1)
        


def get_ghg_matrices(data, yr_idx, aggregate=True):
    """
    Return g_mrj matrix <unit: t/cell> as 3D Numpy array.
    
    Parameters:
        data (object): The data object containing the necessary information.
        yr_idx (int): The index of the year.
        aggregate (bool, optional): Whether to aggregate the results. Defaults to True.
    
    Returns:
        numpy.ndarray or pandas.DataFrame: The GHG emissions matrix as a 3D Numpy array if aggregate is True,
        or as a pandas DataFrame if aggregate is False.
    """
    
    if aggregate == True:  
        return np.stack(
            tuple(
                get_ghg_matrix(data, lm, yr_idx, aggregate)
                for lm in data.LANDMANS
            )
        )
    elif aggregate == False:   
        return pd.concat([get_ghg_matrix(data, lu, yr_idx, aggregate) 
                          for lu in data.LANDMANS], axis=1)



def get_ghg_transition_penalties(data, lumap) -> np.ndarray:
    """
    Gets the one-off greenhouse gas penalties for transitioning natural land to
    modified land. The penalty represents the carbon that is emitted when
    clearing natural land.

    Parameters:
        data (object): The data object containing relevant information.
        lumap (1D array): The lumap object containing land use mapping.

    Returns:
        np.ndarray, <unit : t/ha>.
    """
    ncells, n_ag_lus = data.REAL_AREA.shape[0], len(data.AGRICULTURAL_LANDUSES)
    # Set up empty array of penalties
    penalties_rj = np.zeros((ncells, n_ag_lus), dtype=np.float32)
    natural_lu_cells = tools.get_ag_natural_lu_cells(data, lumap)

    # Calculate penalties and add to g_rj matrix
    penalties_r = (
          data.NATURAL_LAND_T_CO2_HA[natural_lu_cells]
        * data.REAL_AREA[natural_lu_cells]
    )
    for lu in data.LU_MODIFIED_LAND:
        penalties_rj[natural_lu_cells, lu] = penalties_r

    return np.stack([penalties_rj] * 2)



def get_ghg_limits(data, target):
    """
    Return greenhouse gas emissions limits in tonnes CO2e from year target.

    Parameters:
    - data: The data containing greenhouse gas emissions targets.
    - target: The target year for which the emissions limit is requested.

    Returns:
    - The greenhouse gas emissions limit in tonnes CO2e for the specified target year.
    """
    return data.GHG_TARGETS[target]




def get_asparagopsis_effect_g_mrj(data, yr_idx):
    """
    Applies the effects of using asparagopsis to the GHG data
    for all relevant agricultural land uses.

    Parameters:
    - data: The input data containing GHG and land use information.
    - yr_idx: The index of the year to calculate the effects for.

    Returns:
    - new_g_mrj: The matrix <unit: t/cell> containing the updated GHG data with the effects of using asparagopsis.

    Note: This function relies on other helper functions such as lvs_veg_types and get_yield_pot to calculate
    the reduction amount for each land use and management type.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix, taking into account the CH4 reduction of asparagopsis
    for lu_idx, lu in enumerate(land_uses):
        ch4_reduction_perc = 1 - data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, "CO2E_KG_HEAD_ENTERIC"]

        if ch4_reduction_perc != 0:
            for lm in data.LANDMANS:
                m = 0 if lm == 'irr' else 1
                # Subtract enteric fermentation emissions multiplied by reduction multiplier
                lvstype, vegtype = lvs_veg_types(lu)

                yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)

                reduction_amnt = (
                    data.AGGHG_LVSTK[lvstype, "CO2E_KG_HEAD_ENTERIC"].to_numpy()
                    * yield_pot
                    * ch4_reduction_perc
                    / 1000            # convert to tonnes
                    * data.REAL_AREA  # adjust for resfactor
                )
                new_g_mrj[m, :, lu_idx] = -reduction_amnt

    return new_g_mrj


def get_precision_agriculture_effect_g_mrj(data, yr_idx):
    """
    Applies the effects of using precision agriculture to the GHG data
    for all relevant agr. land uses.

    Parameters:
    - data: The input data containing the necessary information.
    - yr_idx: The index of the year to calculate the effects for.

    Returns:
    - new_g_mrj: The matrix <unit: t/cell> containing the updated GHG data after applying the effects of precision agriculture.
    """

    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix
    for lu_idx, lu in enumerate(land_uses):
        lu_data = data.PRECISION_AGRICULTURE_DATA[lu]

        for lm in data.LANDMANS:
            m = 0 if lm == 'dry' else 1
            for co2e_type in [
                'CO2E_KG_HA_CHEM_APPL',
                'CO2E_KG_HA_CROP_MGT',
                'CO2E_KG_HA_PEST_PROD',
                'CO2E_KG_HA_SOIL'
            ]:
                # Check if land-use/land management combination exists (e.g., dryland Pears/Rice do not occur), if not use zeros
                if lu not in data.AGGHG_CROPS[data.AGGHG_CROPS.columns[0][0], lm].columns:
                    continue

                reduction_perc = 1 - lu_data.loc[yr_cal, co2e_type]

                if reduction_perc != 0:
                    reduction_amnt = (
                        np.nan_to_num(data.AGGHG_CROPS[co2e_type, lm, lu].to_numpy(), 0) 
                        * reduction_perc
                        / 1000            # convert to tonnes
                        * data.REAL_AREA  # adjust for resfactor
                    )
                    new_g_mrj[m, :, lu_idx] -= reduction_amnt

    if np.isnan(new_g_mrj).any():
        raise ValueError("Error in data: NaNs detected in agricultural management options' GHG effect matrix.")

    return new_g_mrj


def get_ecological_grazing_effect_g_mrj(data, yr_idx):
    """
    Applies the effects of using ecological grazing to the GHG data
    for all relevant agricultural land uses.

    Parameters:
    - data: The input data containing relevant information for calculations.
    - yr_idx: The index of the year for which the calculations are performed.

    Returns:
    - new_g_mrj: The matrix <unit: t/cell> containing the updated GHG data after applying ecological grazing effects.
    """

    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix
    for lu_idx, lu in enumerate(land_uses):
        lu_data = data.ECOLOGICAL_GRAZING_DATA[lu]

        for lm in data.LANDMANS:
            m = 0 if lm == 'dry' else 1
            # Subtract leach runoff carbon benefit
            leach_reduction_perc = 1 - lu_data.loc[yr_cal, 'CO2E_KG_HEAD_IND_LEACH_RUNOFF']
            if leach_reduction_perc != 0:
                lvstype, vegtype = lvs_veg_types(lu)
                yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)

                leach_reduction_amnt = (
                    data.AGGHG_LVSTK[lvstype, 'CO2E_KG_HEAD_IND_LEACH_RUNOFF'].to_numpy()
                    * yield_pot       # convert to HAs
                    * leach_reduction_perc
                    / 1000            # convert to tonnes
                    * data.REAL_AREA  # adjust for resfactor
                )
                new_g_mrj[m, :, lu_idx] -= leach_reduction_amnt

            # Subtract soil carbon benefit
            soil_multiplier = lu_data.loc[yr_cal, 'IMPACTS_soil_carbon'] - 1
            if soil_multiplier != 0:
                soil_reduction_amnt = (
                    data.SOIL_CARBON_AVG_T_CO2_HA
                    * soil_multiplier
                    * data.REAL_AREA  # adjust for resfactor
                )
                new_g_mrj[m, :, lu_idx] -= soil_reduction_amnt

    return new_g_mrj


def get_savanna_burning_effect_g_mrj(data, g_mrj):
    """
    Applies the effects of using ecological grazing to the GHG data
    for all relevant agr. land uses.

    Parameters:
    - data: The input data containing relevant information.
    - g_mrj: The ecological grazing factor.

    Returns:
    - sb_g_mrj: The GHG data <unit: t/cell> with the effects of ecological grazing applied.
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Savanna Burning"])
    sb_g_mrj = np.zeros((data.NLMS, data.NCELLS, nlus))

    for m, j in itertools.product(range(data.NLMS), range(nlus)):
        sb_g_mrj[m, :, j] = -data.SAVBURN_TOTAL_TCO2E_HA * data.REAL_AREA

    return sb_g_mrj


def get_agtech_ei_effect_g_mrj(data, yr_idx):
    """
    Applies the effects of using AgTech EI to the GHG data
    for all relevant agr. land uses.

    Parameters:
    - data: The input data containing the necessary information.
    - yr_idx: The index of the year to calculate the effects for.

    Returns:
    - new_g_mrj: The matrix <unit: t/cell> containing the updated GHG data after applying the AgTech EI effects.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix
    for lu_idx, lu in enumerate(land_uses):
        lu_data = data.AGTECH_EI_DATA[lu]

        for lm in data.LANDMANS:
            m = 0 if lm == 'dry' else 1
            for co2e_type in [
                'CO2E_KG_HA_CHEM_APPL',
                'CO2E_KG_HA_CROP_MGT',
                'CO2E_KG_HA_PEST_PROD',
                'CO2E_KG_HA_SOIL'
            ]:
                # Check if land-use/land management combination exists (e.g., dryland Pears/Rice do not occur), if not use zeros
                if lu not in data.AGGHG_CROPS[data.AGGHG_CROPS.columns[0][0], lm].columns:
                    continue

                reduction_perc = 1 - lu_data.loc[yr_cal, co2e_type]

                if reduction_perc != 0:
                    reduction_amnt = (
                        np.nan_to_num(data.AGGHG_CROPS[co2e_type, lm, lu].to_numpy(), 0) 
                        * reduction_perc
                        / 1000            # convert to tonnes
                        * data.REAL_AREA  # adjust for resfactor
                    )
                    new_g_mrj[m, :, lu_idx] -= reduction_amnt

            # Subtract extra 'CO2e_KG_HA_IRRIG' carbon for irrigated land uses
            if m == 1:
                if lu not in data.AGGHG_CROPS[data.AGGHG_CROPS.columns[0][0], lm].columns:
                    continue

                # Columns names for irrig. CO2e are inconsistent across sheets
                irrig_co2e_col = 'CO2e_KG_HA_IRRIG'
                if 'CO2E_KG_HA_IRRIG' in lu_data.columns:
                    irrig_co2e_col = 'CO2E_KG_HA_IRRIG'

                reduction_perc = 1 - lu_data.loc[yr_cal, irrig_co2e_col]

                if reduction_perc != 0:
                    reduction_amnt = (
                        np.nan_to_num(data.AGGHG_CROPS['CO2E_KG_HA_IRRIG', lm, lu].to_numpy(), 0) 
                        * reduction_perc
                        / 1000            # convert to tonnes
                        * data.REAL_AREA  # adjust for resfactor
                    )
                    new_g_mrj[m, :, lu_idx] -= reduction_amnt

    return new_g_mrj


def get_agricultural_management_ghg_matrices(data, g_mrj, yr_idx) -> Dict[str, np.ndarray]:
    """
    Calculate the greenhouse gas (GHG) matrices for different agricultural management practices.

    Args:
        data: The input data for the calculations.
        g_mrj: The g_mrj parameter.
        yr_idx: The year index.

    Returns:
        A dictionary containing the GHG matrices <unit: t/cell> for different agricultural management practices.
        The keys of the dictionary represent the management practices, and the values are numpy arrays.

    """
    asparagopsis_data = get_asparagopsis_effect_g_mrj(data, yr_idx)
    precision_agriculture_data = get_precision_agriculture_effect_g_mrj(data, yr_idx)
    eco_grazing_data = get_ecological_grazing_effect_g_mrj(data, yr_idx)
    sav_burning_ghg_impact = get_savanna_burning_effect_g_mrj(data, g_mrj)
    agtech_ei_ghg_impact = get_agtech_ei_effect_g_mrj(data, yr_idx)

    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_ghg_impact,
        'AgTech EI': agtech_ei_ghg_impact,
    }
