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
from luto.data import Data, lumap2ag_l_mrj
import luto.tools as tools
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.transitions as ag_transitions
from luto.settings import NON_AG_LAND_USES


def get_env_plant_transitions_from_ag(data: Data, yr_idx, lumap, w_license_cost_r, w_rm_irrig_cost_r, separate=False) -> np.ndarray|dict:
    """
    Calculate the transition costs for transitioning from agricultural land to environmental plantings.

    Args:
        data (object): The data object containing relevant information.
        yr_idx (int): The index of the year.
        lumap (np.ndarray): The land use map.
        w_license_cost_r (np.ndarray): The water license costs.
        w_rm_irrig_cost_r (np.ndarray): The costs of removing irrigation.
        separate (bool, optional): Whether to return separate costs or the total cost. Defaults to False.

    Returns
        np.ndarray|dict: The transition costs as either a numpy array or a dictionary, depending on the value of `separate`.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Establishment costs
    est_costs_r = tools.amortise(data.EP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_r[tools.get_env_plantings_cells(lumap)] = 0.0
    
    # Transition costs
    base_ag_to_ep_t_r = np.vectorize(dict(enumerate(data.AG2EP_TRANSITION_COSTS_HA)).get, otypes=['float32'])(lumap)
    base_ag_to_ep_t_r = tools.amortise(base_ag_to_ep_t_r * data.REAL_AREA)
    base_ag_to_ep_t_r = np.nan_to_num(base_ag_to_ep_t_r)
    base_ag_to_ep_t_r[tools.get_env_plantings_cells(lumap)] = 0.0
    
    # Waster costs; Assume EP is dryland, and no 'REMOVE_IRRIG_COST' for EP
    w_license_cost_r[tools.get_env_plantings_cells(lumap)] = 0.0
    w_rm_irrig_cost_r[tools.get_env_plantings_cells(lumap)] = 0.0
    w_cost_r = w_license_cost_r + w_rm_irrig_cost_r

    if separate:
        return {'Establishment cost (Ag2Non-Ag)': est_costs_r,
                'Transition cost (Ag2Non-Ag)': base_ag_to_ep_t_r, 
                'Water license cost (Ag2Non-Ag)': w_license_cost_r,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r}
    else:   
        return est_costs_r + base_ag_to_ep_t_r + w_cost_r


def get_rip_plant_transitions_from_ag(data: Data, base_costs_r, yr_idx, lumap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to riparian plantings for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_costs_r_copy = base_costs_r.copy()     # Copy the transition costs so we do not modify the original values

    fencing_cost_r = (
        data.RP_FENCING_LENGTH 
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[data.YR_CAL_BASE + yr_idx]
        * data.REAL_AREA
    ).astype(np.float32)
    
    if separate:
        base_costs_r_copy.update({'Fencing cost (Ag2Non-Ag)':fencing_cost_r})
        return base_costs_r_copy
    else:
        return base_costs_r_copy + fencing_cost_r


def get_agroforestry_transitions_from_ag_base(data: Data, base_costs_r, yr_idx, lumap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to agroforestry for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_costs_r_copy = base_costs_r.copy()   # Copy the transition costs so we do not modify the original values

    fencing_cost_r = (
        settings.AF_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[data.YR_CAL_BASE + yr_idx]
        * data.REAL_AREA 
    ).astype(np.float32)
    
    
    if separate:
        base_costs_r_copy.update({'Fencing cost (Ag2Non-Ag)':fencing_cost_r})
        return base_costs_r_copy
    else:
        return base_costs_r_copy + fencing_cost_r
    

def get_sheep_agroforestry_transitions_from_ag(
    data: Data, agroforestry_x_r, agroforestry_costs, ag_t_costs, lumap, separate=False
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
    
    sheep_j = tools.get_sheep_code(data)

    if separate:
        # Copy the transition costs so we do not modify the original values
        ag_cost = ag_t_costs.copy()
        non_ag_cost = agroforestry_costs.copy()
        # Combine and return separated costs
        for key, array in non_ag_cost.items():
            non_ag_cost.update({key: array * agroforestry_x_r})
        for key, array in ag_cost.items():
            ag_cost.update({key: array[0, :, sheep_j] * (1 - agroforestry_x_r)})

        return {**non_ag_cost, **ag_cost}

    else:
        sheep_costs_r = ag_t_costs[0, :, sheep_j]                   # Assume sheep is dryland under sheep-agroforestry  
        t_r = (sheep_costs_r * (1 - agroforestry_x_r)) + (agroforestry_costs * agroforestry_x_r)

        return t_r.astype(np.float32)
    

def get_beef_agroforestry_transitions_from_ag(
    data: Data, agroforestry_x_r, agroforestry_costs, ag_t_costs, lumap, separate=False
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
    
    beef_j = tools.get_beef_code(data)

    if separate:
        # Copy the transition costs so we do not modify the original values
        ag_cost = ag_t_costs.copy()
        non_ag_cost = agroforestry_costs.copy()
        # Combine and return separated costs
        for key, array in non_ag_cost.items():
            non_ag_cost.update({key: array * agroforestry_x_r})
        for key, array in ag_cost.items():
            ag_cost.update({key: array[0, :, beef_j] * (1 - agroforestry_x_r)})

        return {**non_ag_cost, **ag_cost}

    else:
        beef_costs_r = ag_t_costs[0, :, beef_j]    
        t_r = (beef_costs_r * (1 - agroforestry_x_r)) + (agroforestry_costs * agroforestry_x_r) 

        return t_r.astype(np.float32)


def get_carbon_plantings_block_from_ag(data: Data, yr_idx, lumap, w_license_cost_r, w_rm_irrig_cost_r, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to carbon plantings (block) for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Establishment costs for each cell
    est_costs_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal] ).astype(np.float32)
    est_costs_r[tools.get_carbon_plantings_block_cells(lumap)] = 0.0
    
    # Transition costs
    base_ag_to_cp_t_j = np.vectorize(dict(enumerate(data.AG2EP_TRANSITION_COSTS_HA)).get, otypes=['float32'])(lumap).astype(np.float32)
    base_ag_to_cp_t_j = tools.amortise(base_ag_to_cp_t_j * data.REAL_AREA).copy()
    base_ag_to_cp_t_j = np.nan_to_num(base_ag_to_cp_t_j)
    base_ag_to_cp_t_j[tools.get_carbon_plantings_block_cells(lumap)] = 0.0

    # Water costs (pre-amortised)
    w_license_cost_r[tools.get_carbon_plantings_block_cells(lumap)] = 0.0
    w_rm_irrig_cost_r[tools.get_carbon_plantings_block_cells(lumap)] = 0.0
    w_cost_r = w_license_cost_r + w_rm_irrig_cost_r

    if separate:
        return {'Establishment cost (Ag2Non-Ag)': est_costs_r,
                'Transition cost (Ag2Non-Ag)':base_ag_to_cp_t_j, 
                'Water license cost (Ag2Non-Ag)': w_license_cost_r,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r
                }
    else:
        return est_costs_r + base_ag_to_cp_t_j + w_cost_r


def get_carbon_plantings_belt_from_ag_base(data: Data, base_costs_r, yr_idx, lumap, separate) -> np.ndarray|dict:
    """
    Get the base transition costs from agricultural land uses to carbon plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    
    base_costs_r_copy = base_costs_r.copy()         # Copy the transition costs so we do not modify the original values

    fencing_cost_r = (
        settings.CP_BELT_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[data.YR_CAL_BASE + yr_idx]
        * data.REAL_AREA
    ).astype(np.float32)
    
    
    if separate:
        base_costs_r_copy.update({'Fencing cost (Ag2Non-Ag)':fencing_cost_r})
        return base_costs_r_copy
    else:
        return base_costs_r_copy + fencing_cost_r



def get_sheep_carbon_plantings_belt_from_ag(
    data: Data, cp_belt_x_r, cp_belt_costs, ag_t_costs, lumap, separate=False
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
    
    sheep_j = tools.get_sheep_code(data)

    if separate:
        # Copy the transition costs so we do not modify the original values
        ag_cost = ag_t_costs.copy()
        non_ag_cost = cp_belt_costs.copy()
        # Combine and return separated costs
        for key, array in non_ag_cost.items():
            non_ag_cost.update({key: array * cp_belt_x_r})
        for key, array in ag_cost.items():
            ag_cost.update({key: array[0, :, sheep_j] * (1 - cp_belt_x_r)})

        return {**non_ag_cost, **ag_cost}

    else:
        
        sheep_costs_r = ag_t_costs[0, :, sheep_j]                   # Assume sheep is dryland under sheep-agroforestry
        t_r = (sheep_costs_r * (1 - cp_belt_x_r)) + (cp_belt_costs * cp_belt_x_r)

        return t_r.astype(np.float32)


def get_beef_carbon_plantings_belt_from_ag(
    data: Data, cp_belt_x_r, cp_belt_costs, ag_t_costs, lumap, separate=False
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
    
    beef_j = tools.get_beef_code(data)

    if separate:
        # Copy the transition costs so we do not modify the original values
        ag_cost = ag_t_costs.copy()
        non_ag_cost = cp_belt_costs.copy()
        # Combine and return separated costs
        for key, array in non_ag_cost.items():
            non_ag_cost.update({key: array * cp_belt_x_r})
        for key, array in ag_cost.items():
            ag_cost.update({key: array[0, :, beef_j] * (1 - cp_belt_x_r)})
            
        return {**non_ag_cost, **ag_cost}

    else:
        beef_costs_r = ag_t_costs[0, :, beef_j]          # Assume beef is dryland under beef-agroforestry
        t_r = (beef_costs_r * (1 - cp_belt_x_r)) + (cp_belt_costs * cp_belt_x_r)

        return t_r


def get_beccs_from_ag(data, yr_idx, lumap, w_license_cost_r, w_rm_irrig_cost_r, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to carbon plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """

    return get_env_plant_transitions_from_ag(data, yr_idx, lumap, w_license_cost_r, w_rm_irrig_cost_r, separate)



def get_destocked_from_ag(data: Data, ag_t_mrj: np.ndarray) -> np.ndarray:
    """
    Get transition costs from agricultural land uses to destocked land for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    unallocated_j = tools.get_unallocated_natural_land_code(data)
    destocked_t_mrj = ag_t_mrj[0, :, unallocated_j]
    return destocked_t_mrj


def get_from_ag_transition_matrix(
    data: Data,
    yr_idx: int,
    base_year: int,
    lumap: np.ndarray,
    lmmap: np.ndarray,
    ag_t_mrj: np.ndarray,
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
    separate : bool, optional
        If True, return a dictionary containing the transition costs for each non-agricultural land use.
        If False, return a 2-D array indexed by (r, k) where r is cell and k is non-agricultural land usage.

    Returns
    -------
    np.ndarray or dict
        If separate is False, returns a 2-D array indexed by (r, k) where r is cell and k is non-agricultural land usage.
        If separate is True, returns a dictionary containing the transition costs for each non-agricultural land use.
    """
    
    ag_t_costs = ag_transitions.get_transition_matrices(data, yr_idx, base_year, separate)
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)
    w_license_cost_r, w_rm_irrig_cost_r = tools.get_ag_to_non_ag_water_delta_matrix(data, yr_idx, lumap, lmmap)
    
    env_plant_transitions_from_ag = get_env_plant_transitions_from_ag(data, yr_idx, lumap, w_license_cost_r, w_rm_irrig_cost_r, separate)                           # Base EP transition 
    rip_plant_transitions_from_ag = get_rip_plant_transitions_from_ag(data, env_plant_transitions_from_ag, yr_idx, lumap, separate)                                 # Base EP transition plus RF fencing costs
    agroforestry_costs = get_agroforestry_transitions_from_ag_base(data, env_plant_transitions_from_ag, yr_idx, lumap, separate)                                    # Base EP transition plus AF fencing costs
    
    sheep_agroforestry_transitions_from_ag = get_sheep_agroforestry_transitions_from_ag(data, agroforestry_x_r, agroforestry_costs, ag_t_costs, lumap, separate)    # Base EP transition plus RF fencing costs + sheep grazing
    beef_agroforestry_transitions_from_ag = get_beef_agroforestry_transitions_from_ag( data, agroforestry_x_r, agroforestry_costs, ag_t_costs, lumap, separate)     # Base EP transition plus RF fencing costs + beef grazing
    
    carbon_plantings_block_transitions_from_ag = get_carbon_plantings_block_from_ag(data, yr_idx, lumap, w_license_cost_r, w_rm_irrig_cost_r, separate)             # Base CP transition 
    cp_belt_costs = get_carbon_plantings_belt_from_ag_base(data, carbon_plantings_block_transitions_from_ag, yr_idx, lumap, separate)                               # Base CP transition plus CP fencing costs
    
    sheep_carbon_plantings_belt_transitions_from_ag = get_sheep_carbon_plantings_belt_from_ag(data, cp_belt_x_r, cp_belt_costs, ag_t_costs, lumap, separate)        # Base CP transition plus CP fencing costs + sheep grazing
    beef_carbon_plantings_belt_transitions_from_ag = get_beef_carbon_plantings_belt_from_ag( data, cp_belt_x_r, cp_belt_costs, ag_t_costs, lumap, separate)         # Base CP transition plus CP fencing costs + beef grazing
    
    beccs_transitions_from_ag = get_beccs_from_ag(data, yr_idx, lumap, w_license_cost_r, w_rm_irrig_cost_r, separate)                                               # Base EP transition (the same)
    destocked_from_ag = get_destocked_from_ag(data, ag_t_mrj)

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
            'Destocked - natural land': destocked_from_ag
        }
        
    else:
        # Stack the transition matrices into a single 2D array (r, k)
        return np.array([
            env_plant_transitions_from_ag,
            rip_plant_transitions_from_ag,
            sheep_agroforestry_transitions_from_ag,
            beef_agroforestry_transitions_from_ag,
            carbon_plantings_block_transitions_from_ag,
            sheep_carbon_plantings_belt_transitions_from_ag,
            beef_carbon_plantings_belt_transitions_from_ag,
            beccs_transitions_from_ag,
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
    l_mrj = lumap2ag_l_mrj(lumap, lmmap)
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
    l_mrj = lumap2ag_l_mrj(all_sheep_lumap, all_dry_lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    t_ij = data.AG_TMATRIX * data.TRANS_COST_MULTS[yr_cal]
    x_mrj = ag_transitions.get_exclude_matrices(data, all_sheep_lumap)

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
    ghg_t_mrj = ag_ghg.get_ghg_transition_penalties(data, all_sheep_lumap)               # <unit: t/ha>      
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
    l_mrj = lumap2ag_l_mrj(all_beef_lumap, all_dry_lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    t_ij = data.AG_TMATRIX * data.TRANS_COST_MULTS[yr_cal]
    x_mrj = ag_transitions.get_exclude_matrices(data, all_beef_lumap)

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
    ghg_t_mrj = ag_ghg.get_ghg_transition_penalties(data, all_beef_lumap)               # <unit: t/ha>      
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
    

def get_destocked_to_ag(data: Data, yr_idx: int, lumap: np.ndarray) -> np.ndarray:
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
    if destocked_cells.size == 0:
        return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    
    # Get transition costs from destocked cells by using transition costs from unallocated land
    unallocated_t_mrj = ag_transitions.get_transition_matrices_from_maps(data, yr_idx, all_unallocated_lumap, all_dry_lmmap)

    destocked_t_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    destocked_t_mrj[:, destocked_cells, :] = unallocated_t_mrj[:, destocked_cells, :]
    return destocked_t_mrj


def get_to_ag_transition_matrix(data: Data, yr_idx, lumap, lmmap, ag_t_mrj: np.ndarray, separate=False) -> np.ndarray|dict:
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
    
    non_ag_to_agr_t_matrices = {lu: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32) for lu in NON_AG_LAND_USES}

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
    non_ag_to_agr_t_matrices['Destocked - natural land'] = get_destocked_to_ag(data, yr_idx, lumap)

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


def get_exclusions_environmental_plantings(data: Data, lumap) -> np.ndarray:
    """
    Get the exclusion array for the environmental plantings land use.

    Parameters
    - data: The data object containing information about land use transitions.
    - lumap: The land use map.

    Returns
    - exclude: The exclusion array where 0 represents excluded land uses and 1 represents allowed land uses.
    """
    # Get (agricultural) land uses that cannot transition to environmental plantings
    excluded_ag_lus_cells = np.where(np.isnan(data.AG2EP_TRANSITION_COSTS_HA))[0]

    # Create the exclude array as having 0 for every cell that has an excluded land use and 1 otherwise.
    exclude = np.where(np.isin(lumap, excluded_ag_lus_cells), 0, 1)

    # Ensure other non-agricultural land uses are excluded
    exclude[tools.get_non_ag_natural_lu_cells(data, lumap)] = 0

    # Ensure cells being used for environmental plantings may retain that LU
    exclude[tools.get_env_plantings_cells(lumap)] = 1

    return exclude


def get_exclusions_riparian_plantings(data: Data, lumap) -> np.ndarray:
    """
    Get the exclusion array for the Riparian plantings land use.
    
    This function calculates and returns a 1-D array indexed by r that represents how much Riparian Plantings (RP) land use can be utilized.
    
    Parameters
        data (DataFrame): The data containing information about the land use.
        lumap (array-like): The land use map.
        
    Returns
        np.ndarray: The exclusion array for Riparian Plantings land use.
    """
    exclude = (data.RP_PROPORTION).astype(np.float32)

    # Exclude all cells used for natural land uses
    # TODO - this means natural LU cells cannot transition to agriculture/RP splits, even though
    # they may transition to agriculture without the RP portion.
    exclude *= tools.get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for riparian plantings may retain that LU
    rp_cells = tools.get_riparian_plantings_cells(lumap)
    exclude[rp_cells] = data.RP_PROPORTION[rp_cells]

    return exclude


def get_exclusions_sheep_agroforestry(
    data: Data, ag_x_mrj: np.ndarray, lumap: np.ndarray
) -> np.ndarray:
    """
    Calculate exclusions for sheep and agroforestry land use transitions.

    Args:
        data (Data): The data object containing information about the model.
        ag_x_mrj (np.ndarray): The agroforestry land use matrix.
        lumap (np.ndarray): The land use map.

    Returns
        np.ndarray: An array of exclusions indicating which cells cannot utilize both agroforestry and sheep.

    """
    sheep_j = tools.get_sheep_code(data)
    sheep_x_r = ag_x_mrj[0, :, sheep_j]
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)

    exclusions = np.zeros(data.NCELLS).astype(np.float32)

    # Block cells that can't utilise both agroforestry and sheep - natural land.
    intersect = np.intersect1d(np.nonzero(sheep_x_r)[0], np.nonzero(agroforestry_x_r)[0])
    exclusions[intersect] = 1
    return exclusions


def get_exclusions_beef_agroforestry(
    data: Data, ag_x_mrj: np.ndarray, lumap: np.ndarray
) -> np.ndarray:
    """
    Calculate exclusions for cells that cannot utilize both agroforestry and beef.

    Args:
        data (Data): The data object containing relevant information.
        ag_x_mrj (np.ndarray): The ag_x_mrj array.
        lumap (np.ndarray): The lumap array.

    Returns
        np.ndarray: An array of exclusions, where 1 represents cells that cannot utilize both agroforestry and beef.
    """
    beef_j = tools.get_beef_code(data)
    beef_x_r = ag_x_mrj[0, :, beef_j]
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)

    exclusions = np.zeros(data.NCELLS).astype(np.float32)

    # Block cells that can't utilise both agroforestry and beef - natural land.
    intersect = np.intersect1d(np.nonzero(beef_x_r)[0], np.nonzero(agroforestry_x_r)[0])
    exclusions[intersect] = 1
    return exclusions


def get_exclusions_carbon_plantings_block(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much carbon plantings (block) can possibly 
    be done at each cell.

    Parameters
    - data: The data object containing information about the cells.
    - lumap: The land use map.

    Returns
    - exclude: A 1-D numpy array
    """
    exclude = np.ones(data.NCELLS)
    exclude *= tools.get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for carbon plantings (block) may retain that LU
    exclude[tools.get_carbon_plantings_block_cells(lumap)] = 1

    return exclude


def get_exclusions_sheep_carbon_plantings_belt(
    data: Data, ag_x_mrj: np.ndarray, lumap: np.ndarray
) -> np.ndarray:
    """
    Calculate exclusions for sheep and carbon plantings belt.

    Args:
        data (Data): The data object containing relevant information.
        ag_x_mrj (np.ndarray): The ag_x_mrj array.
        lumap (np.ndarray): The lumap array.

    Returns
        np.ndarray: The exclusions array.

    """
    sheep_j = tools.get_sheep_code(data)
    sheep_x_r = ag_x_mrj[0, :, sheep_j]
    cp_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    exclusions = np.zeros(data.NCELLS).astype(np.float32)

    # Block cells that can't utilise both agroforestry and sheep - natural land.
    intersect = np.intersect1d(np.nonzero(sheep_x_r)[0], np.nonzero(cp_x_r)[0])
    exclusions[intersect] = 1
    return exclusions


def get_exclusions_beef_carbon_plantings_belt(
    data: Data, ag_x_mrj: np.ndarray, lumap: np.ndarray
) -> np.ndarray:
    """
    Calculate exclusions for cells that cannot utilize both agroforestry and beef.

    Parameters
        data (Data): The data object containing necessary information.
        ag_x_mrj (np.ndarray): The agroforestry matrix.
        lumap (np.ndarray): The land use map.

    Returns
        np.ndarray: An array of exclusions, where 1 represents cells that cannot utilize both agroforestry and beef.
    """
    beef_j = tools.get_beef_code(data)
    beef_x_r = ag_x_mrj[0, :, beef_j]
    cp_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    exclusions = np.zeros(data.NCELLS).astype(np.float32)

    # Block cells that can't utilise both agroforestry and beef - natural land.
    intersect = np.intersect1d(np.nonzero(beef_x_r)[0], np.nonzero(cp_x_r)[0])
    exclusions[intersect] = 1
    return exclusions


def get_exclusions_beccs(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much BECCS can possibly 
    be done at each cell.

    Parameters
    - data: The data object containing BECCS costs and other relevant information.
    - lumap: The land use map object.

    Returns
    - exclude: A 1-D array
    """
    exclude = np.zeros(data.NCELLS).astype(np.float32)

    # All cells with NaN BECCS data should be excluded from eligibility
    beccs_cells = np.argwhere(~np.isnan(data.BECCS_COSTS_AUD_HA_YR))[:, 0]
    exclude[beccs_cells] = 1

    # Exclude all cells used for natural land uses
    exclude *= tools.get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for BECCS may retain that LU
    exclude[tools.get_beccs_cells(lumap)] = 1

    return exclude


def get_exclusions_destocked(data: Data, lumap: np.ndarray):
    """
    Return a 1-D array indexed by r that represents how much of a cell may be used for destocked land.

    Parameters:
    - data: The data object containing BECCS costs and other relevant information.
    - lumap: The land use map object.

    Returns:
    - exclude: A 1-D array
    """
    destocked_x_r = np.zeros(data.NCELLS).astype(np.int8)
    
    sheep_j = tools.get_natural_sheep_code
    sheep_cells = tools.get_cells_using_ag_landuse(lumap, sheep_j)
    destocked_x_r[sheep_cells] = 1

    beef_j = tools.get_natural_beef_code
    beef_cells = tools.get_cells_using_ag_landuse(lumap, beef_j)
    destocked_x_r[beef_cells] = 1

    return destocked_x_r


def get_exclude_matrices(data: Data, ag_x_mrj, lumap) -> np.ndarray:
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
    non_ag_x_matrices = {lu: np.zeros(data.NCELLS).astype(np.float32) for lu in NON_AG_LAND_USES}

    # Environmental plantings exclusions. Note: the order must be consistent with the NON_AG_LAND_USES order.
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_ag_x_matrices['Environmental Plantings'] = get_exclusions_environmental_plantings(data, lumap)
    if NON_AG_LAND_USES['Riparian Plantings']:
        non_ag_x_matrices['Riparian Plantings'] = get_exclusions_riparian_plantings(data, lumap)
    if NON_AG_LAND_USES['Sheep Agroforestry']:
        non_ag_x_matrices['Sheep Agroforestry'] = get_exclusions_sheep_agroforestry(data, ag_x_mrj, lumap)
    if NON_AG_LAND_USES['Beef Agroforestry']:
        non_ag_x_matrices['Beef Agroforestry'] = get_exclusions_beef_agroforestry(data, ag_x_mrj, lumap)
    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_ag_x_matrices['Carbon Plantings (Block)'] = get_exclusions_carbon_plantings_block(data, lumap)
    if NON_AG_LAND_USES['Sheep Carbon Plantings (Belt)']:
        non_ag_x_matrices['Sheep Carbon Plantings (Belt)'] = get_exclusions_sheep_carbon_plantings_belt(data, ag_x_mrj, lumap)
    if NON_AG_LAND_USES['Beef Carbon Plantings (Belt)']:
        non_ag_x_matrices['Beef Carbon Plantings (Belt)'] = get_exclusions_beef_carbon_plantings_belt(data, ag_x_mrj, lumap)
    if NON_AG_LAND_USES['BECCS']:
        non_ag_x_matrices['BECCS'] = get_exclusions_beccs(data, lumap)
    if NON_AG_LAND_USES['Destocked - natural land']:
        non_ag_x_matrices['Destocked - natural land'] = get_exclusions_destocked(data, lumap)

    if settings.EXCLUDE_NO_GO_LU:
        no_go_regions = data.NO_GO_REGION_NON_AG
        for count, desc in enumerate(data.NO_GO_LANDUSE_NON_AG):
            if desc in non_ag_x_matrices.keys():
                non_ag_x_matrices[desc] *= no_go_regions[count]

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_ag_x_matrices = [array.reshape((data.NCELLS, 1)) for array in non_ag_x_matrices.values()]
    return np.concatenate(non_ag_x_matrices, axis=1).astype(np.float32)


def get_lower_bound_non_agricultural_matrices(data: Data, base_year) -> np.ndarray:
    """
    Get the non-agricultural lower bound matrix.

    Returns
    -------
    2-D array, indexed by (r,k) where r is the cell and k is the non-agricultural land usage.
    """

    if base_year == data.YR_CAL_BASE or base_year not in data.non_ag_dvars:
        return np.zeros((data.NCELLS, len(NON_AG_LAND_USES))).astype(np.float32)
        
    return np.divide(
        np.floor(data.non_ag_dvars[base_year].astype(np.float32) * 10 ** settings.ROUND_DECMIALS),
        10 ** settings.ROUND_DECMIALS,
    )
