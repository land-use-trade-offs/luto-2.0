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

import numpy as np

from luto import settings
from luto.data import Data
import luto.tools as tools
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.transitions as ag_transitions


def get_env_plant_transitions_from_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Calculate the transition costs for transitioning from agricultural land to environmental plantings.

    Args:
        data (object): The data object containing relevant information.
        yr_idx (int): The index of the year.
        lumap (np.ndarray): The land use map.
        lmmap (np.ndarray): The water supply map.
        separate (bool, optional): Whether to return separate costs or the total cost. Defaults to False.

    Returns
        np.ndarray|dict: The transition costs as either a numpy array or a dictionary, depending on the value of `separate`.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.EP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_r[~cells] = 0.0
    
    # Transition costs
    ag_to_ep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Environmental Plantings').values
    ag_to_ep_t_r = np.vectorize(dict(enumerate(ag_to_ep_j)).get, otypes=['float32'])(lumap)
    ag_to_ep_t_r = np.nan_to_num(ag_to_ep_t_r)
    ag_to_ep_t_r = tools.amortise(ag_to_ep_t_r * data.REAL_AREA)
    ag_to_ep_t_r[~cells] = 0.0
    
    # Water costs; Assume EP is dryland
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    if separate:
        return {'Establishment cost (Ag2Non-Ag)': est_costs_r,
                'Transition cost (Ag2Non-Ag)': ag_to_ep_t_r, 
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r}
    else:   
        return est_costs_r + ag_to_ep_t_r + w_rm_irrig_cost_r


def get_rip_plant_transitions_from_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to riparian plantings for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.RP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_r[~cells] = 0.0
    est_costs_r *= data.RP_PROPORTION  # Apply proportion of riparian plantings
    
    # Transition costs
    ag_to_ep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Riparian Plantings').values
    ag_to_ep_t_r = np.vectorize(dict(enumerate(ag_to_ep_j)).get, otypes=['float32'])(lumap)
    ag_to_ep_t_r = np.nan_to_num(ag_to_ep_t_r)
    ag_to_ep_t_r = tools.amortise(ag_to_ep_t_r * data.REAL_AREA)
    ag_to_ep_t_r[~cells] = 0.0
    ag_to_ep_t_r *= data.RP_PROPORTION  # Apply proportion of riparian plantings
    
    
    # Water costs; Assume riparian plantings are dryland
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0
    w_rm_irrig_cost_r *= data.RP_PROPORTION  # Apply proportion of riparian plantings

    # Fencing costs
    fencing_cost_r = (
        data.RP_FENCING_LENGTH 
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[yr_cal]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    fencing_cost_r *= data.RP_PROPORTION  # Apply proportion of riparian plantings
    
    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_ep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_r + ag_to_ep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_sheep_agroforestry_transitions_from_ag(
    data: Data, yr_idx, lumap, lmmap, separate=False
):
    """
    Get the base transition costs from agricultural land uses to Sheep Agroforestry for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.AF_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_r[~cells] = 0.0
    est_costs_r *= settings.AF_PROPORTION  
    
    # Transition costs
    ag_to_agroforestry_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Agroforestry').values
    ag_to_agroforestry_t_r = np.vectorize(dict(enumerate(ag_to_agroforestry_j)).get, otypes=['float32'])(lumap)
    ag_to_agroforestry_t_r = np.nan_to_num(ag_to_agroforestry_t_r)
    ag_to_agroforestry_t_r = tools.amortise(ag_to_agroforestry_t_r * data.REAL_AREA)
    ag_to_agroforestry_t_r[~cells] = 0.0
    ag_to_agroforestry_t_r *= settings.AF_PROPORTION
    
    ag_to_sheep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values    # Only consider ag to sheep-modified land here; Ag to sheep-natural is handled in the destocked-natural module
    ag_to_sheep_t_r = np.vectorize(dict(enumerate(ag_to_sheep_j)).get, otypes=['float32'])(lumap)
    ag_to_sheep_t_r = np.nan_to_num(ag_to_sheep_t_r)
    ag_to_sheep_t_r = tools.amortise(ag_to_sheep_t_r * data.REAL_AREA)
    ag_to_sheep_t_r[~cells] = 0.0
    ag_to_sheep_t_r *= (1 - settings.AF_PROPORTION)
    
    # Water costs; Assume AF is dryland so no need to multiply by AF_PROPORTION here
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    # Fencing costs
    fencing_cost_r = (
        settings.AF_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[yr_cal]
        * data.REAL_AREA 
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    fencing_cost_r *= settings.AF_PROPORTION
    
    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_agroforestry_t_r,
            'Transition cost (Ag2AF-Sheep)': ag_to_sheep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_r + ag_to_agroforestry_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r
    

def get_beef_agroforestry_transitions_from_ag(
    data: Data, yr_idx, lumap, lmmap, separate=False
):
    """
    Get the base transition costs from agricultural land uses to Beef Agroforestry for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.AF_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_r[~cells] = 0.0
    est_costs_r *= settings.AF_PROPORTION  
    
    # Transition costs
    ag_to_agroforestry_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Agroforestry').values
    ag_to_agroforestry_t_r = np.vectorize(dict(enumerate(ag_to_agroforestry_j)).get, otypes=['float32'])(lumap)
    ag_to_agroforestry_t_r = np.nan_to_num(ag_to_agroforestry_t_r)
    ag_to_agroforestry_t_r = tools.amortise(ag_to_agroforestry_t_r * data.REAL_AREA)
    ag_to_agroforestry_t_r[~cells] = 0.0
    ag_to_agroforestry_t_r *= settings.AF_PROPORTION
    
    ag_to_beef_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values    # Only consider ag to beef-modified land here; Ag to beef-natural is handled in the destocked-natural module
    ag_to_beef_t_r = np.vectorize(dict(enumerate(ag_to_beef_j)).get, otypes=['float32'])(lumap)
    ag_to_beef_t_r = np.nan_to_num(ag_to_beef_t_r)
    ag_to_beef_t_r = tools.amortise(ag_to_beef_t_r * data.REAL_AREA)
    ag_to_beef_t_r[~cells] = 0.0
    ag_to_beef_t_r *= (1 - settings.AF_PROPORTION)
    
    # Water costs; Assume AF is dryland so no need to multiply by AF_PROPORTION here
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0
    
    # Fencing costs
    fencing_cost_r = (
        settings.AF_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[data.YR_CAL_BASE + yr_idx]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    fencing_cost_r *= settings.AF_PROPORTION
    
    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_agroforestry_t_r,
            'Transition cost (Ag2AF-Beef)': ag_to_beef_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_r + ag_to_agroforestry_t_r + ag_to_beef_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_carbon_plantings_block_from_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to carbon plantings (block) for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_CP_r[~cells] = 0.0

    # Transition costs
    ag_to_cp_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Carbon Plantings (Block)').values
    ag_to_cp_t_r = np.vectorize(dict(enumerate(ag_to_cp_j)).get, otypes=['float32'])(lumap)
    ag_to_cp_t_r = np.nan_to_num(ag_to_cp_t_r)
    ag_to_cp_t_r = tools.amortise(ag_to_cp_t_r * data.REAL_AREA)
    ag_to_cp_t_r[~cells] = 0.0

    # Water costs; Assume CP is dryland
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_CP_r,
            'Transition cost (Ag2Non-Ag)': ag_to_cp_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r
        }
    else:
        return est_costs_CP_r + ag_to_cp_t_r + w_rm_irrig_cost_r




def get_sheep_carbon_plantings_belt_from_ag(
    data: Data, yr_idx, lumap, lmmap, separate=False
):
    """
    Get the transition costs from agricultural land uses to Sheep Carbon Plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_CP_r[~cells] = 0.0
    est_costs_CP_r *= settings.CP_BELT_PROPORTION

    # Transition costs
    ag_to_cp_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Carbon Plantings (Belt)').values
    ag_to_cp_t_r = np.vectorize(dict(enumerate(ag_to_cp_j)).get, otypes=['float32'])(lumap)
    ag_to_cp_t_r = np.nan_to_num(ag_to_cp_t_r)
    ag_to_cp_t_r = tools.amortise(ag_to_cp_t_r * data.REAL_AREA)
    ag_to_cp_t_r[~cells] = 0.0
    ag_to_cp_t_r *= settings.CP_BELT_PROPORTION
    
    ag_to_sheep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values    # Only consider sheep-modified land here; Ag to sheep-natural is handled in the destocked-natural module
    ag_to_sheep_t_r = np.vectorize(dict(enumerate(ag_to_sheep_j)).get, otypes=['float32'])(lumap)
    ag_to_sheep_t_r = np.nan_to_num(ag_to_sheep_t_r)
    ag_to_sheep_t_r = tools.amortise(ag_to_sheep_t_r * data.REAL_AREA)
    ag_to_sheep_t_r[~cells] = 0.0
    ag_to_sheep_t_r *= (1 - settings.CP_BELT_PROPORTION)  
    
    # Water costs; Assume CP is dryland
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    fencing_cost_r = (
        settings.CP_BELT_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[data.YR_CAL_BASE + yr_idx]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    fencing_cost_r *= settings.CP_BELT_PROPORTION

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_CP_r,
            'Transition cost (Ag2Non-Ag)': ag_to_cp_t_r,
            'Transition cost (Ag2CP-Sheep)': ag_to_sheep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_CP_r + ag_to_cp_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_beef_carbon_plantings_belt_from_ag(
    data: Data,  yr_idx, lumap, lmmap, separate=False
):
    """
    Get the base transition costs from agricultural land uses to Beef Carbon Plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_CP_r[~cells] = 0.0
    est_costs_CP_r *= settings.CP_BELT_PROPORTION

    # Transition costs
    ag_to_cp_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Carbon Plantings (Belt)').values
    ag_to_cp_t_r = np.vectorize(dict(enumerate(ag_to_cp_j)).get, otypes=['float32'])(lumap)
    ag_to_cp_t_r = np.nan_to_num(ag_to_cp_t_r)
    ag_to_cp_t_r = tools.amortise(ag_to_cp_t_r * data.REAL_AREA)
    ag_to_cp_t_r[~cells] = 0.0
    ag_to_cp_t_r *= settings.CP_BELT_PROPORTION
    
    ag_to_sheep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values    # Only consider beef-modified land here; Ag to beef-natural is handled in the destocked-natural module
    ag_to_sheep_t_r = np.vectorize(dict(enumerate(ag_to_sheep_j)).get, otypes=['float32'])(lumap)
    ag_to_sheep_t_r = np.nan_to_num(ag_to_sheep_t_r)
    ag_to_sheep_t_r = tools.amortise(ag_to_sheep_t_r * data.REAL_AREA)
    ag_to_sheep_t_r[~cells] = 0.0
    ag_to_sheep_t_r *= (1 - settings.CP_BELT_PROPORTION)  
    
    # Water costs; Assume CP is dryland
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    fencing_cost_r = (
        settings.CP_BELT_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[data.YR_CAL_BASE + yr_idx]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    fencing_cost_r *= settings.CP_BELT_PROPORTION

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_CP_r,
            'Transition cost (Ag2Non-Ag)': ag_to_cp_t_r,
            'Transition cost (Ag2CP-Beef)': ag_to_sheep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_CP_r + ag_to_cp_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_beccs_from_ag(data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to carbon plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """

    return get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)



def get_destocked_from_ag(
    data: Data, yr_idx, lumap, lmmap, separate=False
) -> np.ndarray | dict:
    """
    Get transition costs from agricultural land uses to destocked land for each cell.

    Returns
    -------
    if separate == False:
        np.ndarray
            1-D array, indexed by cell.
    if separate == True:
        dict
            Separated dictionary of transition cost arrays.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, data.LU_LVSTK_NATURAL)
    
    # Establishment costs; If destocking brings 30% of bio/GHG benefits, then it takes 30% of establishment costs as Environmental Plantings
    HCAS_benefit_mult = {lu:1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[lu] for lu in data.LU_LVSTK_NATURAL}
    est_costs_r = np.vectorize(HCAS_benefit_mult.get, otypes=[np.float32])(lumap) * data.EP_EST_COST_HA
    est_costs_r = np.nan_to_num(est_costs_r)
    est_costs_r = tools.amortise(est_costs_r * data.REAL_AREA)
    
    # # Transition costs
    # ag2destock_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Destocked - natural land').values
    # ag_to_destock_t_r = np.vectorize(dict(enumerate(ag2destock_j)).get, otypes=['float32'])(lumap)
    # ag_to_destock_t_r = np.nan_to_num(ag_to_destock_t_r)
    # ag_to_destock_t_r = tools.amortise(ag_to_destock_t_r * data.REAL_AREA)
    # ag_to_destock_t_r[~cells] = 0.0
    
    # Water costs; Assume destocked land is dryland
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0
    
    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r
        }
    else:
        return est_costs_r + w_rm_irrig_cost_r


def get_transition_matrix_ag2nonag(
    data: Data,
    yr_idx: int,
    lumap: np.ndarray,
    lmmap: np.ndarray,
    separate: bool = False
) -> np.ndarray|dict:
    """
    Get the matrix containing transition costs from agricultural land uses to non-agricultural land uses.

    Parameters
    ----------
    data : object
        The data object containing information about the model.
    yr_idx : int
        The index of the year.
    lumap : dict
        The land use map.
    lmmap : dict
        The land management map.
    ag_t_mrj: np.ndarray | dict
        Agricultural transitions: should be the ag_t_mrj array if separate==False and the separated agricultural
        tranistions dictionary if separate==True
    separate : bool, optional
        If True, return a dictionary containing the transition costs for each non-agricultural land use.
        If False, return a 2-D array indexed by (r, k) where r is cell and k is non-agricultural land usage.

    Returns
    -------
    np.ndarray or dict
        If separate is False, returns a 2-D array indexed by (r, k) where r is cell and k is non-agricultural land usage.
        If separate is True, returns a dictionary containing the transition costs for each non-agricultural land use.
    """

    env_plant_transitions_from_ag = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate) 
    rip_plant_transitions_from_ag = get_rip_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate) 
    sheep_agroforestry_transitions_from_ag = get_sheep_agroforestry_transitions_from_ag(data, yr_idx, lumap, lmmap, separate) 
    beef_agroforestry_transitions_from_ag = get_beef_agroforestry_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    carbon_plantings_block_transitions_from_ag = get_carbon_plantings_block_from_ag(data, yr_idx, lumap, lmmap, separate)            
    sheep_carbon_plantings_belt_transitions_from_ag = get_sheep_carbon_plantings_belt_from_ag(data, yr_idx, lumap, lmmap, separate)        
    beef_carbon_plantings_belt_transitions_from_ag = get_beef_carbon_plantings_belt_from_ag(data, yr_idx, lumap, lmmap, separate)         
    beccs_transitions_from_ag = get_beccs_from_ag(data, yr_idx, lumap, lmmap, separate)
    destocked_from_ag = get_destocked_from_ag(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # IMPORTANT: The order of the keys in the dictionary must match the order of the non-agricultural land uses
        return {
            'Environmental Plantings': env_plant_transitions_from_ag,
            'Riparian Plantings': rip_plant_transitions_from_ag,
            'Sheep Agroforestry': sheep_agroforestry_transitions_from_ag,
            'Beef Agroforestry': beef_agroforestry_transitions_from_ag,
            'Carbon Plantings (Block)': carbon_plantings_block_transitions_from_ag,
            'Sheep Carbon Plantings (Belt)': sheep_carbon_plantings_belt_transitions_from_ag,
            'Beef Carbon Plantings (Belt)': beef_carbon_plantings_belt_transitions_from_ag,
            'BECCS': beccs_transitions_from_ag,
            'Destocked - natural land': destocked_from_ag,
        }
        
    else:
        return np.array([
            env_plant_transitions_from_ag,
            rip_plant_transitions_from_ag,
            sheep_agroforestry_transitions_from_ag,
            beef_agroforestry_transitions_from_ag,
            carbon_plantings_block_transitions_from_ag,
            sheep_carbon_plantings_belt_transitions_from_ag,
            beef_carbon_plantings_belt_transitions_from_ag,
            beccs_transitions_from_ag,
            destocked_from_ag,
        ]).T.astype(np.float32)


# TODO: Need to check the logic of transition cost, espcially the water cost.
def get_env_plantings_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from environmental plantings to agricultural land uses for each cell.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    l_mrj = tools.lumap2ag_l_mrj(lumap, lmmap)
    l_mrj_not = np.logical_not(l_mrj)           # This ensures the lu remains the same has 0 cost

    # Get base transition costs: add cost of installing irrigation
    base_ep_to_ag_t = data.EP2AG_TRANSITION_COSTS_HA * data.TRANS_COST_MULTS[yr_cal]

    # Get the agricultural cells, and the env-ag can not happen on these cells
    ag_cells, _ = tools.get_ag_and_non_ag_cells(lumap)

    # Get water license price and costs of installing/removing irrigation where appropriate
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
    w_delta_mrj[:, ag_cells, :] = 0

    # Reshape and amortise upfront costs to annualised costs
    base_ep_to_ag_t_mrj = np.broadcast_to(base_ep_to_ag_t, (data.NLMS, data.NCELLS, base_ep_to_ag_t.shape[0]))
    base_ep_to_ag_t_mrj = tools.amortise(base_ep_to_ag_t_mrj).copy()
    base_ep_to_ag_t_mrj[:, ag_cells, :] = 0

    if separate:
        return {'Non-Ag2Ag Transition cost':np.nan_to_num(np.einsum('mrj,mrj,r->mrj', base_ep_to_ag_t_mrj, l_mrj_not, data.REAL_AREA)), 
                'Non-Ag2Ag Water license cost': np.nan_to_num(np.einsum('mrj,mrj,r->mrj', w_delta_mrj, l_mrj_not, data.REAL_AREA))}
        
    # Add cost of water license and cost of installing/removing irrigation where relevant (pre-amortised)
    ep_to_ag_t_mrj = (base_ep_to_ag_t_mrj + w_delta_mrj) * l_mrj_not * data.REAL_AREA[np.newaxis, :, np.newaxis]
    return np.nan_to_num(ep_to_ag_t_mrj) 


def get_rip_plantings_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from riparian plantings to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    if separate:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    else:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)


def get_agroforestry_to_ag_base(data: Data, yr_idx, lumap, lmmap, separate) -> np.ndarray|dict:
    """
    Get transition costs from agroforestry to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    if separate:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    else:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)


def get_sheep_to_ag_base(data: Data, yr_idx: int, lumap, separate=False) -> np.ndarray|dict:
    """
    Get sheep contribution to transition costs to agricultural land uses.
    Used for getting transition costs for Sheep Agroforestry and CP (Belt).

    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    ------
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    sheep_j = tools.get_sheep_code(data)

    all_sheep_lumap = (np.ones(data.NCELLS) * sheep_j).astype(np.int8)
    all_dry_lmmap = np.zeros(data.NCELLS).astype(np.float32)
    l_mrj = tools.lumap2ag_l_mrj(all_sheep_lumap, all_dry_lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    t_ij = data.AG_TMATRIX * data.TRANS_COST_MULTS[yr_cal]
    x_mrj = ag_transitions.get_to_ag_exclude_matrices(data, all_sheep_lumap)

    # Calculate sheep contribution to transition costs
    # Establishment costs
    ag_cells = tools.get_ag_cells(lumap)

    e_rj = np.zeros((data.NCELLS, data.N_AG_LUS)).astype(np.float32)
    e_rj[ag_cells, :] = t_ij[all_sheep_lumap[ag_cells]]

    e_rj = tools.amortise(e_rj) * data.REAL_AREA[:, np.newaxis]
    e_rj_dry = np.einsum('rj,r->rj', e_rj, all_sheep_lumap == 0)
    e_rj_irr = np.einsum('rj,r->rj', e_rj, all_dry_lmmap == 1)
    e_mrj = np.stack([e_rj_dry, e_rj_irr], axis=0)
    e_mrj = np.einsum('mrj,mrj,mrj->mrj', e_mrj, x_mrj, l_mrj_not)

    # Water license cost
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
    w_delta_mrj = np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not)

    # Carbon costs
    ghg_t_mrj = ag_ghg.get_ghg_transition_emissions(data, all_sheep_lumap)               # <unit: t/ha>      
    ghg_t_mrj_cost = tools.amortise(ghg_t_mrj * data.get_carbon_price_by_yr_idx(yr_idx))     
    ghg_t_mrj_cost = np.einsum('mrj,mrj,mrj->mrj', ghg_t_mrj_cost, x_mrj, l_mrj_not)

    # Ensure transition costs are zero for all agricultural cells 
    e_mrj[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)
    w_delta_mrj[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)
    ghg_t_mrj_cost[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)

    if separate:
        return {
            'Non-Ag2Ag Establishment cost': np.nan_to_num(e_mrj), 
            'Water license cost': np.nan_to_num(w_delta_mrj), 
            'GHG emissions cost': np.nan_to_num(ghg_t_mrj_cost)
        }
    
    else:
        return np.nan_to_num(e_mrj + w_delta_mrj + ghg_t_mrj_cost)


def get_beef_to_ag_base(data: Data, yr_idx, lumap, separate) -> np.ndarray|dict:
    """
    Get beef contribution to transition costs to agricultural land uses.
    Used for getting transition costs for Beef Agroforestry and CP (Belt).

    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    beef_j = tools.get_beef_code(data)

    all_beef_lumap = (np.ones(data.NCELLS) * beef_j).astype(np.int8)
    all_dry_lmmap = np.zeros(data.NCELLS).astype(np.float32)
    l_mrj = tools.lumap2ag_l_mrj(all_beef_lumap, all_dry_lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    t_ij = data.AG_TMATRIX * data.TRANS_COST_MULTS[yr_cal]
    x_mrj = ag_transitions.get_to_ag_exclude_matrices(data, all_beef_lumap)

    # Calculate sheep contribution to transition costs
    # Establishment costs
    ag_cells = tools.get_ag_cells(lumap)

    e_rj = np.zeros((data.NCELLS, data.N_AG_LUS)).astype(np.float32)
    e_rj[ag_cells, :] = t_ij[all_beef_lumap[ag_cells]]

    e_rj = tools.amortise(e_rj) * data.REAL_AREA[:, np.newaxis]
    e_mrj = np.stack([e_rj] * data.NLMS, axis=0)
    e_mrj = np.einsum('mrj,mrj,mrj->mrj', e_mrj, x_mrj, l_mrj_not)

    # Water license cost
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
    w_delta_mrj = np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not)

    # Carbon costs
    ghg_t_mrj = ag_ghg.get_ghg_transition_emissions(data, all_beef_lumap)               # <unit: t/ha>      
    ghg_t_mrj_cost = tools.amortise(ghg_t_mrj * data.get_carbon_price_by_yr_idx(yr_idx))     
    ghg_t_mrj_cost = np.einsum('mrj,mrj,mrj->mrj', ghg_t_mrj_cost, x_mrj, l_mrj_not)

    beef_af_cells = tools.get_beef_agroforestry_cells(lumap)
    non_beef_af_cells = np.array([r for r in range(data.NCELLS) if r not in beef_af_cells])

    # Ensure transition costs are zero for all agricultural cells 
    e_mrj[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)
    w_delta_mrj[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)
    ghg_t_mrj_cost[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)

    if separate:
        return {
            'Non-Ag2Ag Establishment cost': np.nan_to_num(e_mrj), 
            'Non-Ag2Ag Water license cost': np.nan_to_num(w_delta_mrj), 
            'Non-Ag2Ag GHG emissions cost': np.nan_to_num(ghg_t_mrj_cost)
        }
    
    else:
        t_mrj = e_mrj + w_delta_mrj + ghg_t_mrj_cost
        # Set all costs for non-beef-agroforestry cells to zero
        t_mrj[:, non_beef_af_cells, :] = 0
        return np.nan_to_num(t_mrj)


def get_sheep_agroforestry_to_ag(
    data: Data, yr_idx, lumap, lmmap, agroforestry_x_r, separate=False
) -> np.ndarray|dict:
    """
    Get transition costs of Sheep Agroforestry to all agricultural land uses.

    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    sheep_tcosts = get_sheep_to_ag_base(data, yr_idx, lumap, separate)
    agroforestry_tcosts = get_agroforestry_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in agroforestry_tcosts.items():
            combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * agroforestry_x_r

        for key, array in sheep_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - agroforestry_x_r)

        return combined_costs
    
    else:
        sheep_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                sheep_contr[m, :, j] = (1 - agroforestry_x_r) * sheep_tcosts[m, :, j]

        agroforestry_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                agroforestry_contr[m, :, j] = agroforestry_x_r * agroforestry_tcosts[m, :, j]

        return sheep_contr + agroforestry_contr


def get_beef_agroforestry_to_ag(
    data: Data, yr_idx, lumap, lmmap, agroforestry_x_r, separate=False
) -> np.ndarray|dict:
    """
    Get transition costs of Beef Agroforestry to all agricultural land uses.
    
    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    beef_tcosts = get_beef_to_ag_base(data, yr_idx, lumap, separate)
    agroforestry_tcosts = get_agroforestry_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in agroforestry_tcosts.items():
            combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * agroforestry_x_r

        for key, array in beef_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - agroforestry_x_r)

        return combined_costs
    
    else:
        beef_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                beef_contr[m, :, j] = (1 - agroforestry_x_r) * beef_tcosts[m, :, j]

        agroforestry_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                agroforestry_contr[m, :, j] = agroforestry_x_r * agroforestry_tcosts[m, :, j]

        return beef_contr + agroforestry_contr


def get_carbon_plantings_block_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False):
    """
    Get transition costs from carbon plantings (block) to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)


def get_carbon_plantings_belt_to_ag_base(data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from carbon plantings (belt) to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)


def get_sheep_carbon_plantings_belt_to_ag(
    data: Data, yr_idx, lumap, lmmap, cp_belt_x_r, separate
) -> np.ndarray|dict:
    """
    Get transition costs of Sheep Carbon Plantings (Belt) to all agricultural land uses.
    
    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    sheep_tcosts = get_sheep_to_ag_base(data, yr_idx, lumap, separate)
    cp_belt_tcosts = get_carbon_plantings_belt_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in cp_belt_tcosts.items():
            combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * cp_belt_x_r

        for key, array in sheep_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - cp_belt_x_r)

        return combined_costs
    
    else:
        sheep_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                sheep_contr[m, :, j] = (1 - cp_belt_x_r) * sheep_tcosts[m, :, j]

        cp_belt_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                cp_belt_contr[m, :, j] = cp_belt_x_r * cp_belt_tcosts[m, :, j]

        return sheep_contr + cp_belt_contr
    

def get_beef_carbon_plantings_belt_to_ag(
    data: Data, yr_idx, lumap, lmmap, cp_belt_x_r, separate
) -> np.ndarray|dict:
    """
    Get transition costs of Beef Carbon Plantings (Belt) to all agricultural land uses.
    
    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    beef_tcosts = get_beef_to_ag_base(data, yr_idx, lumap, separate)
    cp_belt_tcosts = get_carbon_plantings_belt_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in cp_belt_tcosts.items():
            combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * cp_belt_x_r

        for key, array in beef_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - cp_belt_x_r)

        return combined_costs
    
    else:
        beef_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                beef_contr[m, :, j] = (1 - cp_belt_x_r) * beef_tcosts[m, :, j]

        cp_belt_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                cp_belt_contr[m, :, j] = cp_belt_x_r * cp_belt_tcosts[m, :, j]

        return beef_contr + cp_belt_contr


def get_beccs_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from BECCS to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    if separate:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    else:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)
    

def get_destocked_to_ag(data: Data, yr_idx: int, lumap: np.ndarray, separate: bool = False) -> np.ndarray:
    """
    Get transition costs from destocked land to agricultural land uses for each cell.
    Transition costs are based on the transition costs of unallocated natural land to agricultural land.
    
    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    unallocated_j = tools.get_unallocated_natural_land_code(data)
    all_unallocated_lumap = (np.ones(data.NCELLS) * unallocated_j).astype(np.int8)
    all_dry_lmmap = (np.zeros(data.NCELLS)).astype(np.int8)

    destocked_cells = tools.get_destocked_land_cells(lumap)
    if destocked_cells.size == 0 and separate == False:
        return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    
    # Get transition costs from destocked cells by using transition costs from unallocated land
    unallocated_t_mrj = ag_transitions.get_transition_matrices_ag2ag(
        data, yr_idx, all_unallocated_lumap, all_dry_lmmap, separate=separate
    )

    if separate == False:
        destocked_t_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
        destocked_t_mrj[:, destocked_cells, :] = unallocated_t_mrj[:, destocked_cells, :]
        return destocked_t_mrj
    
    elif separate == True:
        sep_destocked_trans = {k: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)) for k in unallocated_t_mrj}
        if destocked_cells.size == 0:
            return sep_destocked_trans

        for k, v in unallocated_t_mrj.items():
            sep_destocked_trans[k][:, destocked_cells, :] = v[:, destocked_cells, :]
        return sep_destocked_trans

    raise ValueError(
        f"Incorrect value for 'separate' when calling get_destocked_from_ag: {separate}. "
        f"should be either True or False."
    )


def get_to_ag_transition_matrix(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get the matrix containing transition costs from non-agricultural land uses to agricultural land uses.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    yr_idx : int
        The index of the year.
    lumap : dict
        The land use mapping dictionary.
    lmmap : dict
        The land management mapping dictionary.
    separate : bool, optional
        If True, returns a dictionary of transition matrices for each land use category.
        If False, returns a single aggregated transition matrix.

    Returns
    -------
    np.ndarray or dict
        If `separate` is True, returns a dictionary of transition matrices, where the keys are the land use categories.
        If `separate` is False, returns a single aggregated transition matrix.

    """
    
    non_ag_to_agr_t_matrices = {lu: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32) for lu in settings.NON_AG_LAND_USES}

    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_ag_to_agr_t_matrices['Environmental Plantings'] = get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    non_ag_to_agr_t_matrices['Riparian Plantings'] = get_rip_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    non_ag_to_agr_t_matrices['Sheep Agroforestry'] = get_sheep_agroforestry_to_ag(data, yr_idx, lumap, lmmap, agroforestry_x_r, separate)
    non_ag_to_agr_t_matrices['Beef Agroforestry'] = get_beef_agroforestry_to_ag(data, yr_idx, lumap, lmmap, agroforestry_x_r, separate)
    non_ag_to_agr_t_matrices['Carbon Plantings (Block)'] = get_carbon_plantings_block_to_ag(data, yr_idx, lumap, lmmap, separate)
    non_ag_to_agr_t_matrices['Sheep Carbon Plantings (Belt)'] = get_sheep_carbon_plantings_belt_to_ag(data, yr_idx, lumap, lmmap, cp_belt_x_r, separate)
    non_ag_to_agr_t_matrices['Beef Carbon Plantings (Belt)'] = get_beef_carbon_plantings_belt_to_ag(data, yr_idx, lumap, lmmap, cp_belt_x_r, separate)
    non_ag_to_agr_t_matrices['BECCS'] = get_beccs_to_ag(data, yr_idx, lumap, lmmap, separate)
    non_ag_to_agr_t_matrices['Destocked - natural land'] = get_destocked_to_ag(data, yr_idx, lumap, separate)

    if separate:
        # Note: The order of the keys in the dictionary must match the order of the non-agricultural land uses
        return non_ag_to_agr_t_matrices
            
    non_ag_to_agr_t_matrices = list(non_ag_to_agr_t_matrices.values())
    return np.add.reduce(non_ag_to_agr_t_matrices)


def get_non_ag_transition_matrix(data: Data) -> np.ndarray:
    """
    Get the matrix that contains transition costs for non-agricultural land uses. 
    There are no transition costs for non-agricultural land uses, therefore the matrix is filled with zeros.
    
    Parameters
        data (object): The data object containing information about the model.
    
    Returns
        np.ndarray: The transition cost matrix, filled with zeros.
    """
    return np.zeros((data.NCELLS, data.N_NON_AG_LUS)).astype(np.float32)



def get_to_non_ag_exclude_matrices(data: Data, lumap) -> np.ndarray:
    """
    Get the non-agricultural exclusions matrix.

    Parameters
    ----------
    data : object
        The data object containing information about the model.
    lumap : object
        The lumap object containing land usage mapping information.

    Returns
    -------
    np.ndarray
        A 2-D array indexed by (r, k) where r is the cell and k is the non-agricultural land usage.

    Notes
    -----
    This function calculates the non-agricultural exclusions matrix by combining several exclusion matrices
    related to different non-agricultural land uses. The resulting matrix is a concatenation of these matrices
    along the k indexing.
    """

    # Get transition costs for to_non_ag 2D array (r, k)
    t_ik = data.T_MAT.loc[:,data.NON_AGRICULTURAL_LANDUSES].copy()
    lumap2desc = np.vectorize(data.ALLLU2DESC.get, otypes=[str])
    ag_cells, non_ag_cells = tools.get_ag_and_non_ag_cells(lumap)                            
    
    t_rk = np.ones((data.NCELLS, len(data.NON_AGRICULTURAL_LANDUSES))).astype(np.float32)    # Empty ones_rj array to be filled with transition flag (1 allow, 0 not allow)
    t_rk[ag_cells, :] = t_ik[lumap[ag_cells]]                                                # For ag cells in the base year lumap, get transition cost (np.nan is not-allow) for them
    t_rk[non_ag_cells, :] *= t_ik.sel(from_lu=lumap2desc(lumap[non_ag_cells]))               # For non-ag cells in the base year lumap, get transition cost (np.nan is not-allow) for them
    t_rk[non_ag_cells, :] *= t_ik.sel(from_lu=lumap2desc(data.LUMAP[non_ag_cells]))          # For non-ag cells, find its ag status in BASE_YR (2010), then get transition cost based on these 2010-ag status
    t_rk = np.where(np.isnan(t_rk), 0, 1).astype(np.int8)  

    # No-go exclusion; user-defined layer specifying which land-use are not disallowd at where
    no_go_x_rk = np.ones((data.NCELLS, data.N_NON_AG_LUS))  
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_NON_AG, data.NO_GO_LANDUSE_NON_AG):
            no_go_j = data.NON_AGRICULTURAL_LANDUSES.index(no_go_desc)   # Get the index of the non-agricultural land use
            no_go_x_rk[:, no_go_j] = no_go_x_r
            
    # Assign non-ag maximum land-use proportions
    no_go_x_rk = (t_rk * no_go_x_rk).astype(np.float32)
    no_go_x_rk[:, 1] *= data.RP_PROPORTION                  # Riparian Plantings can not exceed its proportion to the cell

    return no_go_x_rk


def get_lower_bound_non_agricultural_matrices(data: Data, base_year) -> np.ndarray:
    """
    Get the non-agricultural lower bound matrix.

    Returns
    -------
    2-D array, indexed by (r,k) where r is the cell and k is the non-agricultural land usage.
    """

    if base_year == data.YR_CAL_BASE or base_year not in data.non_ag_dvars:
        return np.zeros((data.NCELLS, len(settings.NON_AG_LAND_USES))).astype(np.float32)
        
    # return np.divide(
    #     np.floor(data.non_ag_dvars[base_year].astype(np.float32) * 10 ** settings.ROUND_DECMIALS),
    #     10 ** settings.ROUND_DECMIALS,
    # )
    
    return data.non_ag_dvars[base_year].astype(np.float32)
