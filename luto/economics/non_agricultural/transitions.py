import numpy as np

from luto import settings
from luto.data import Data, lumap2ag_l_mrj
import luto.tools as tools
import luto.economics.agricultural.water as ag_water


def get_env_plant_transitions_from_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Calculate the transition costs for transitioning from agricultural land to environmental plantings.

    Args:
        data (object): The data object containing relevant information.
        yr_idx (int): The index of the year.
        lumap (np.ndarray): The land use map.
        lmmap (np.ndarray): The land management map.
        separate (bool, optional): Whether to return separate costs or the total cost. Defaults to False.

    Returns:
        np.ndarray|dict: The transition costs as either a numpy array or a dictionary, depending on the value of `separate`.
    """
    
    base_ag_to_ep_t = data.AG2EP_TRANSITION_COSTS_HA
    l_mrj = lumap2ag_l_mrj(lumap, lmmap)
    base_ag_to_ep_t_mrj = np.broadcast_to(base_ag_to_ep_t, (data.NLMS, data.NCELLS, base_ag_to_ep_t.shape[0]))

    # Amortise base costs to be annualised
    base_ag_to_ep_t_mrj = tools.amortise(base_ag_to_ep_t_mrj)

    # Add cost of water license and cost of installing/removing irrigation where relevant (pre-amortised)
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_water_delta_matrix(w_mrj, l_mrj, data)
    ag_to_ep_t_mrj = base_ag_to_ep_t_mrj + w_delta_mrj

    # Get raw transition costs for each cell to transition to environmental plantings
    ag2ep_transitions_r = np.nansum(l_mrj * ag_to_ep_t_mrj, axis=(0, 2))   # Here multiply by l_mrj to force the ag-env transition can only happen on ag cells

    # Add establishment costs for each cell
    est_costs_r = data.EP_EST_COST_HA

    # Amortise establishment costs  to be annualised
    est_costs_r = tools.amortise(est_costs_r)


    if separate:
        return {'Transition cost': np.einsum('mrj,mrj,r->mrj', base_ag_to_ep_t_mrj, l_mrj, data.REAL_AREA), 
                'Establishment cost': np.einsum('r,mrj,r->mrj', est_costs_r, l_mrj, data.REAL_AREA),
                'Water license cost': np.einsum('mrj,mrj,r->mrj', w_delta_mrj, l_mrj, data.REAL_AREA)}
        
    ag2ep_transitions_r += est_costs_r
    return ag2ep_transitions_r * data.REAL_AREA


def get_rip_plant_transitions_from_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to riparian plantings for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_costs = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    fencing_cost = data.RP_FENCING_LENGTH * data.REAL_AREA * settings.RIPARIAN_PLANTINGS_FENCING_COST_PER_M
    
    if separate:
        l_mrj = lumap2ag_l_mrj(lumap, lmmap)
        base_costs.update({'Fencing cost':np.einsum('r,mrj->mrj', fencing_cost, l_mrj)})
        return base_costs
    else:
        return base_costs + fencing_cost


def get_agroforestry_transitions_from_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to agroforestry for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_costs = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    fencing_cost = settings.AF_FENCING_LENGTH * data.REAL_AREA * settings.AGROFORESTRY_FENCING_COST_PER_M
    
    if separate:
        l_mrj = lumap2ag_l_mrj(lumap, lmmap)
        base_costs.update({'Fencing cost':np.einsum('r,mrj->mrj', fencing_cost, l_mrj)})
        return base_costs
    else:
        return base_costs + fencing_cost


def get_carbon_plantings_block_from_ag(data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to carbon plantings (block) for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_ag_to_cp_t = data.AG2EP_TRANSITION_COSTS_HA
    l_mrj = lumap2ag_l_mrj(lumap, lmmap)
    base_ag_to_cp_t_mrj = np.broadcast_to(base_ag_to_cp_t, (2, data.NCELLS, base_ag_to_cp_t.shape[0]))

    # Amortise base costs to be annualised
    base_ag_to_cp_t_mrj = tools.amortise(base_ag_to_cp_t_mrj)

    # Add cost of water license and cost of installing/removing irrigation where relevant (pre-amortised)
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_water_delta_matrix(w_mrj, l_mrj, data)
    ag_to_cp_t_mrj = base_ag_to_cp_t_mrj + w_delta_mrj

    # Get raw transition costs for each cell to transition to carbon plantings
    ag2cp_transitions_r = np.nansum(l_mrj * ag_to_cp_t_mrj, axis=(0, 2))

    # Add establishment costs for each cell
    est_costs_r = data.CP_EST_COST_HA

    # Amortise establishment costs  to be annualised
    est_costs_r = tools.amortise(est_costs_r)
    ag2cp_transitions_r += est_costs_r

    if separate:
        return {'Transition cost':np.einsum('mrj,mrj,r->mrj', base_ag_to_cp_t_mrj, l_mrj, data.REAL_AREA), 
                'Establishment cost': np.einsum('r,mrj,r->mrj', est_costs_r, l_mrj, data.REAL_AREA),
                'Water license cost': np.einsum('mrj,mrj,r->mrj', w_delta_mrj, l_mrj, data.REAL_AREA)}
    else:
        return ag2cp_transitions_r * data.REAL_AREA


def get_carbon_plantings_belt_from_ag(data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to carbon plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_costs = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    fencing_cost = settings.CP_BELT_FENCING_LENGTH * data.REAL_AREA * settings.CARBON_PLANTINGS_BELT_FENCING_COST_PER_M
    
    if separate:
        l_mrj = lumap2ag_l_mrj(lumap, lmmap)
        base_costs.update({'Fencing cost':np.einsum('r,mrj->mrj', fencing_cost, l_mrj)})
        return base_costs
    else:
        return base_costs + fencing_cost


def get_beccs_from_ag(data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to carbon plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    if separate:
        return get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    else:
        return get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap)


def get_from_ag_transition_matrix(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
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
    env_plant_transitions_from_ag_r = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    rip_plant_transitions_from_ag_r = get_rip_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    agroforestry_transitions_from_ag_r = get_agroforestry_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    carbon_plantings_block_transitions_from_ag_r = get_carbon_plantings_block_from_ag(data, yr_idx, lumap, lmmap, separate)
    carbon_plantings_belt_transitions_from_ag_r = get_carbon_plantings_belt_from_ag(data, yr_idx, lumap, lmmap, separate)
    beccs_transitions_from_ag_r = get_beccs_from_ag(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # IMPORTANT: The order of the keys in the dictionary must match the order of the non-agricultural land uses
        return {'Environmental Plantings': env_plant_transitions_from_ag_r,
                'Riparian Plantings': rip_plant_transitions_from_ag_r,
                'Agroforestry': agroforestry_transitions_from_ag_r,
                'Carbon Plantings (Block)': carbon_plantings_block_transitions_from_ag_r,
                'Carbon Plantings (Belt)': carbon_plantings_belt_transitions_from_ag_r,
                'BECCS': beccs_transitions_from_ag_r}
        
    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    ag_to_non_agr_t_matrices = [
        env_plant_transitions_from_ag_r.reshape((data.NCELLS, 1)),
        rip_plant_transitions_from_ag_r.reshape((data.NCELLS, 1)),
        agroforestry_transitions_from_ag_r.reshape((data.NCELLS, 1)),
        carbon_plantings_block_transitions_from_ag_r.reshape((data.NCELLS, 1)),
        carbon_plantings_belt_transitions_from_ag_r.reshape((data.NCELLS, 1)),
        beccs_transitions_from_ag_r.reshape((data.NCELLS, 1)),
    ]
    return np.concatenate(ag_to_non_agr_t_matrices, axis=1)


def get_env_plantings_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from environmental plantings to agricultural land uses for each cell.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    # Get base transition costs: add cost of installing irrigation
    base_ep_to_ag_t = data.EP2AG_TRANSITION_COSTS_HA

    # Get the agricultural cells, and the env-ag can not happen on these cells
    ag_cells, _ = tools.get_ag_and_non_ag_cells(lumap)

    # Get water license price and costs of installing/removing irrigation where appropriate
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    l_mrj = lumap2ag_l_mrj(lumap, lmmap)
    w_delta_mrj = tools.get_water_delta_matrix(w_mrj, l_mrj, data)
    w_delta_mrj[:, ag_cells, :] = 0

    # Reshape and amortise upfront costs to annualised costs
    base_ep_to_ag_t_mrj = np.broadcast_to(base_ep_to_ag_t, (2, data.NCELLS, base_ep_to_ag_t.shape[0]))
    base_ep_to_ag_t_mrj = tools.amortise(base_ep_to_ag_t_mrj).copy()
    base_ep_to_ag_t_mrj[:, ag_cells, :] = 0

    if separate:
        return {'Transition cost':np.einsum('mrj,mrj,r->mrj', base_ep_to_ag_t_mrj, l_mrj, data.REAL_AREA), 
                'Water license cost': np.einsum('mrj,mrj,r->mrj', w_delta_mrj, l_mrj, data.REAL_AREA)}
        
    # Add cost of water license and cost of installing/removing irrigation where relevant (pre-amortised)
    ep_to_ag_t_mrj = base_ep_to_ag_t_mrj + w_delta_mrj
    return ep_to_ag_t_mrj * data.REAL_AREA[np.newaxis, :, np.newaxis]


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


def get_agroforestry_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
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


def get_carbon_plantings_block_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False):
    """
    Get transition costs from carbon plantings (block) to agricultural land uses for each cell.
    
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


def get_carbon_plantings_belt_to_ag(data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from carbon plantings (belt) to agricultural land uses for each cell.
    
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
    
    env_plant_transitions_to_ag_mrj = get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    rip_plant_transitions_to_ag_mrj = get_rip_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    agroforestry_transitions_to_ag_mrj = get_agroforestry_to_ag(data, yr_idx, lumap, lmmap, separate)
    carbon_plantings_block_transitions_to_ag_mrj = get_carbon_plantings_block_to_ag(data, yr_idx, lumap, lmmap, separate)
    carbon_plantings_belt_transitions_to_ag_mrj =  get_carbon_plantings_belt_to_ag(data, yr_idx, lumap, lmmap, separate)
    beccs_transitions_to_ag_mrj = get_beccs_to_ag(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Note: The order of the keys in the dictionary must match the order of the non-agricultural land uses
        return {'Environmental Plantings' : env_plant_transitions_to_ag_mrj, 
                'Riparian Plantings': rip_plant_transitions_to_ag_mrj,
                'Agroforestry': agroforestry_transitions_to_ag_mrj,
                'Carbon Plantings (Block)': carbon_plantings_block_transitions_to_ag_mrj, 
                'Carbon Plantings (Belt)': carbon_plantings_belt_transitions_to_ag_mrj, 
                'BECCS':beccs_transitions_to_ag_mrj}
        
    # Reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    ag_to_non_agr_t_matrices = [
        env_plant_transitions_to_ag_mrj,
        rip_plant_transitions_to_ag_mrj,
        agroforestry_transitions_to_ag_mrj,
        carbon_plantings_block_transitions_to_ag_mrj,
        carbon_plantings_belt_transitions_to_ag_mrj,
        beccs_transitions_to_ag_mrj,
    ]
    return np.add.reduce(ag_to_non_agr_t_matrices)


def get_non_ag_transition_matrix(data: Data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get the matrix that contains transition costs for non-agricultural land uses. 
    There are no transition costs for non-agricultural land uses, therefore the matrix is filled with zeros.
    
    Parameters:
        data (object): The data object containing information about the model.
        yr_idx (int): The index of the year for which to calculate the transition costs.
        lumap (dict): A dictionary mapping land use codes to land use names.
        lmmap (dict): A dictionary mapping land market codes to land market names.
    
    Returns:
        np.ndarray: The transition cost matrix, filled with zeros.
    """
    return np.zeros((data.NCELLS, data.N_NON_AG_LUS))



def get_exclusions_for_excluding_all_natural_cells(data: Data, lumap) -> np.ndarray:
    """
    A number of non-agricultural land uses can only be applied to cells that
    don't already utilise a natural land use. This function gets the exclusion
    matrix for all such non-ag land uses, returning an array valued 0 at the 
    indices of cells that use natural land uses, and 1 everywhere else.

    Parameters:
    - data: The data object containing information about the cells.
    - lumap: The land use map.

    Returns:
    - exclude: An array of shape (NCELLS,) with values 0 at the indices of cells
               that use natural land uses, and 1 everywhere else.
    """
    exclude = np.ones(data.NCELLS)

    natural_lu_cells = tools.get_ag_and_non_ag_natural_lu_cells(data, lumap)
    exclude[natural_lu_cells] = 0

    return exclude


def get_exclusions_environmental_plantings(data: Data, lumap) -> np.ndarray:
    """
    Get the exclusion array for the environmental plantings land use.

    Parameters:
    - data: The data object containing information about land use transitions.
    - lumap: The land use map.

    Returns:
    - exclude: The exclusion array where 0 represents excluded land uses and 1 represents allowed land uses.
    """
    # Get (agricultural) land uses that cannot transition to environmental plantings
    excluded_ag_lus_cells = np.where(np.isnan(data.AG2EP_TRANSITION_COSTS_HA))[0]

    # Create the exclude array as having 0 for every cell that has an excluded land use and 1 otherwise.
    exclude = (~np.isin(lumap, excluded_ag_lus_cells)).astype(int)

    # Ensure other non-agricultural land uses are excluded
    exclude[tools.get_non_ag_natural_lu_cells(data, lumap)] = 0

    # Ensure cells being used for environmental plantings may retain that LU
    exclude[tools.get_env_plantings_cells(lumap)] = 1

    return exclude


def get_exclusions_riparian_plantings(data: Data, lumap) -> np.ndarray:
    """
    Get the exclusion array for the Riparian plantings land use.
    
    This function calculates and returns a 1-D array indexed by r that represents how much Riparian Plantings (RP) land use can be utilized.
    
    Parameters:
        data (DataFrame): The data containing information about the land use.
        lumap (array-like): The land use map.
        
    Returns:
        np.ndarray: The exclusion array for Riparian Plantings land use.
    """
    exclude = (data.RP_PROPORTION).astype(np.float32)

    # Exclude all cells used for natural land uses
    # TODO - this means natural LU cells cannot transition to agriculture/RP splits, even though
    # they may transition to agriculture without the RP portion. Think about this before merging.
    exclude *= get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for riparian plantings may retain that LU
    exclude[tools.get_riparian_plantings_cells(lumap)] = 1

    return exclude

    
def get_exclusions_agroforestry(data: Data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much riparian plantings can possibly 
    be done at each cell.

    Parameters:
    - data: The data object containing information about the landscape.
    - lumap: The land use map.

    Returns:
    - exclude: A 1-D array.
    """
    exclude = (np.ones(data.NCELLS) * settings.AF_PROPORTION).astype(np.float32)

    # Exclude all cells used for natural land uses
    exclude *= get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for agroforestry may retain that LU
    exclude[tools.get_agroforestry_cells(lumap)] = 1

    return exclude


def get_exclusions_carbon_plantings_block(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much carbon plantings (block) can possibly 
    be done at each cell.

    Parameters:
    - data: The data object containing information about the cells.
    - lumap: The land use map.

    Returns:
    - exclude: A 1-D numpy array
    """
    exclude = np.ones(data.NCELLS)
    exclude *= get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for carbon plantings (block) may retain that LU
    exclude[tools.get_carbon_plantings_block_cells(lumap)] = 1

    return exclude


def get_exclusions_carbon_plantings_belt(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much carbon plantings (belt) can possibly 
    be done at each cell.

    Parameters:
    - data: The data object containing information about the cells.
    - lumap: The land use map.

    Returns:
    - exclude: A 1-D array
    """
    exclude = (np.ones(data.NCELLS) * settings.CP_BELT_PROPORTION).astype(np.float32)

    # Exclude all cells used for natural land uses
    exclude *= get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for carbon plantings (belt) may retain that LU
    exclude[tools.get_carbon_plantings_belt_cells(lumap)] = 1

    return exclude


def get_exclusions_beccs(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much BECCS can possibly 
    be done at each cell.

    Parameters:
    - data: The data object containing BECCS costs and other relevant information.
    - lumap: The land use map object.

    Returns:
    - exclude: A 1-D array
    """
    exclude = np.zeros(data.NCELLS)

    # All cells with NaN BECCS data should be excluded from eligibility
    beccs_cells = np.argwhere(~np.isnan(data.BECCS_COSTS_AUD_HA_YR))[:, 0]
    exclude[beccs_cells] = 1

    # Exclude all cells used for natural land uses
    exclude *= get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for BECCS may retain that LU
    exclude[tools.get_beccs_cells(lumap)] = 1

    return exclude


def get_exclude_matrices(data: Data, lumap) -> np.ndarray:
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
    # Environmental plantings exclusions
    env_plant_exclusions = get_exclusions_environmental_plantings(data, lumap)
    rip_plant_exclusions = get_exclusions_riparian_plantings(data, lumap)
    agroforestry_exclusions = get_exclusions_agroforestry(data, lumap)
    carbon_plantings_block_exclusions = get_exclusions_carbon_plantings_block(data, lumap)
    carbon_plantings_belt_exclusions = get_exclusions_carbon_plantings_belt(data, lumap)
    beccs_exclusions = get_exclusions_beccs(data, lumap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_ag_x_matrices = [
        env_plant_exclusions.reshape((data.NCELLS, 1)),
        rip_plant_exclusions.reshape((data.NCELLS, 1)),
        agroforestry_exclusions.reshape((data.NCELLS, 1)),
        carbon_plantings_block_exclusions.reshape((data.NCELLS, 1)),
        carbon_plantings_belt_exclusions.reshape((data.NCELLS, 1)),
        beccs_exclusions.reshape((data.NCELLS, 1)),
    ]

    # Stack list and return to get x_rk
    return np.concatenate(non_ag_x_matrices, axis=1).astype(np.float32)
