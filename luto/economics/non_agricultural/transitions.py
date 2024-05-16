import numpy as np

from luto import settings
from luto.data import Data, lumap2ag_l_mrj
import luto.tools as tools
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.transitions as ag_transitions
from luto.settings import NON_AG_LAND_USES


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
    fencing_cost = data.RP_FENCING_LENGTH * data.REAL_AREA * settings.FENCING_COST_PER_M
    
    if separate:
        l_mrj = lumap2ag_l_mrj(lumap, lmmap)
        base_costs.update({'Fencing cost':np.einsum('r,mrj->mrj', fencing_cost, l_mrj)})
        return base_costs
    else:
        return base_costs + fencing_cost


def get_agroforestry_transitions_from_ag_base(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to agroforestry for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_costs = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    fencing_cost = settings.AF_FENCING_LENGTH * data.REAL_AREA * settings.FENCING_COST_PER_M
    
    if separate:
        l_mrj = lumap2ag_l_mrj(lumap, lmmap)
        base_costs.update({'Fencing cost':np.einsum('r,mrj->mrj', fencing_cost, l_mrj)})
        return base_costs
    else:
        return base_costs + fencing_cost
    

def get_sheep_agroforestry_transitions_from_ag(
    data: Data, agroforestry_x_r, agroforestry_costs, ag_t_costs, yr_idx, base_year, lumap, lmmap, separate=False
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
    
    
    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in agroforestry_costs.items():
            combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * agroforestry_x_r

        for key, array in ag_t_costs.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - agroforestry_x_r)

        return combined_costs

    else:
        sheep_j = tools.get_sheep_natural_land_code(data)
        sheep_costs_r = ag_t_costs[0, :, sheep_j]        
    
        sheep_contr = sheep_costs_r * (1 - agroforestry_x_r)
        cp_belt_contr = agroforestry_costs * agroforestry_x_r
        t_r = sheep_contr + cp_belt_contr

        # Set all non-agricultural land to have zero
        non_ag_cells = tools.get_non_ag_cells(lumap)
        t_r[non_ag_cells] = 0

        return t_r
    

def get_beef_agroforestry_transitions_from_ag(
    data: Data, agroforestry_x_r, agroforestry_costs, ag_t_costs, yr_idx, base_year, lumap, lmmap, separate=False
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
    agroforestry_costs = get_agroforestry_transitions_from_ag_base(data, yr_idx, lumap, lmmap, separate)
    ag_t_costs = ag_transitions.get_transition_matrices(data, yr_idx, base_year, data.lumaps, data.lmmaps, separate)
    
    if separate:
        # Combine and return separated costs
        # Agroforestry keys: 'Transition cost', 'Establishment cost', 'Water license cost', 'Fencing cost'
        combined_costs = {}
        for key, array in agroforestry_costs.items():
            combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * agroforestry_x_r

        # Beef cost keys: 'Establishment cost', 'Water license cost', 'GHG emissions cost'
        for key, array in ag_t_costs.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - agroforestry_x_r)

        return combined_costs

    else:
        beef_j = tools.get_beef_natural_land_code(data)
        beef_costs_r = ag_t_costs[0, :, beef_j]        
    
        beef_contr = beef_costs_r * (1 - agroforestry_x_r)
        agroforestry_contr = agroforestry_costs * agroforestry_x_r
        t_r = beef_contr + agroforestry_contr

        # Set all non-agricultural land to have zero
        non_ag_cells = tools.get_non_ag_cells(lumap)
        t_r[non_ag_cells] = 0

        return t_r


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


def get_carbon_plantings_belt_from_ag_base(data, yr_idx, lumap, lmmap, separate) -> np.ndarray|dict:
    """
    Get the base transition costs from agricultural land uses to carbon plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    base_costs = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    fencing_cost = settings.CP_BELT_FENCING_LENGTH * data.REAL_AREA * settings.FENCING_COST_PER_M
    
    if separate:
        l_mrj = lumap2ag_l_mrj(lumap, lmmap)
        base_costs.update({'Fencing cost':np.einsum('r,mrj->mrj', fencing_cost, l_mrj)})
        return base_costs
    else:
        return base_costs + fencing_cost


def get_sheep_carbon_plantings_belt_from_ag(
    data: Data, cp_belt_x_r, cp_belt_costs, ag_t_costs, yr_idx, base_year, lumap, lmmap, separate=False
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
    
    ag_t_costs = ag_transitions.get_transition_matrices(data, yr_idx, base_year, data.lumaps, data.lmmaps, separate)
    
    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in cp_belt_costs.items():
            combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * cp_belt_x_r

        for key, array in ag_t_costs.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - cp_belt_x_r)

        return combined_costs

    else:
        sheep_j = tools.get_sheep_natural_land_code(data)
        sheep_costs_r = ag_t_costs[0, :, sheep_j]        
    
        sheep_contr = sheep_costs_r * (1 - cp_belt_x_r)
        cp_belt_contr = cp_belt_costs * cp_belt_x_r
        t_r = sheep_contr + cp_belt_contr

        # Set all non-agricultural land to have zero
        non_ag_cells = tools.get_non_ag_cells(lumap)
        t_r[non_ag_cells] = 0

        return t_r


def get_beef_carbon_plantings_belt_from_ag(
    data: Data, cp_belt_x_r, cp_belt_costs, ag_t_costs, yr_idx, base_year, lumap, lmmap, separate=False
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
    cp_belt_costs = get_carbon_plantings_belt_from_ag_base(data, yr_idx, lumap, lmmap, separate)
    ag_t_costs = ag_transitions.get_transition_matrices(data, yr_idx, base_year, data.lumaps, data.lmmaps, separate)
    
    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in cp_belt_costs.items():
            combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * cp_belt_x_r

        for key, array in ag_t_costs.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - cp_belt_x_r)

        return combined_costs

    else:
        beef_j = tools.get_beef_natural_land_code(data)
        beef_costs_r = ag_t_costs[0, :, beef_j]
    
        beef_contr = beef_costs_r * (1 - cp_belt_x_r)
        cp_belt_contr = cp_belt_costs * cp_belt_x_r
        t_r = beef_contr + cp_belt_contr

        # Set all non-agricultural land to have zero
        non_ag_cells = tools.get_non_ag_cells(lumap)
        t_r[non_ag_cells] = 0

        return t_r


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


def get_from_ag_transition_matrix(data: Data, yr_idx, base_year, lumap, lmmap, separate=False) -> np.ndarray|dict:
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
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    agroforestry_costs = get_agroforestry_transitions_from_ag_base(data, yr_idx, lumap, lmmap, separate)
    ag_t_costs = ag_transitions.get_transition_matrices(data, yr_idx, base_year, data.lumaps, data.lmmaps, separate)
    cp_belt_costs = get_carbon_plantings_belt_from_ag_base(data, yr_idx, lumap, lmmap, separate)

    env_plant_transitions_from_ag = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    rip_plant_transitions_from_ag = get_rip_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)
    sheep_agroforestry_transitions_from_ag = get_sheep_agroforestry_transitions_from_ag(
        data, agroforestry_x_r, agroforestry_costs, ag_t_costs, yr_idx, base_year, lumap, lmmap, separate
    )
    beef_agroforestry_transitions_from_ag = get_beef_agroforestry_transitions_from_ag(
        data, agroforestry_x_r, agroforestry_costs, ag_t_costs, yr_idx, base_year, lumap, lmmap, separate
    )
    carbon_plantings_block_transitions_from_ag = get_carbon_plantings_block_from_ag(data, yr_idx, lumap, lmmap, separate)
    sheep_carbon_plantings_belt_transitions_from_ag = get_sheep_carbon_plantings_belt_from_ag(
        data, cp_belt_x_r, cp_belt_costs, ag_t_costs, yr_idx, base_year, lumap, lmmap, separate
    )
    beef_carbon_plantings_belt_transitions_from_ag = get_beef_carbon_plantings_belt_from_ag(
        data, cp_belt_x_r, cp_belt_costs, ag_t_costs, yr_idx, base_year, lumap, lmmap, separate
    )
    beccs_transitions_from_ag = get_beccs_from_ag(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # IMPORTANT: The order of the keys in the dictionary must match the order of the non-agricultural land uses
        return {'Environmental Plantings': env_plant_transitions_from_ag,
                'Riparian Plantings': rip_plant_transitions_from_ag,
                'Sheep Agroforestry': sheep_agroforestry_transitions_from_ag,
                'Beef Agroforestry': beef_agroforestry_transitions_from_ag,
                'Carbon Plantings (Block)': carbon_plantings_block_transitions_from_ag,
                'Sheep Carbon Plantings (Belt)': sheep_carbon_plantings_belt_transitions_from_ag,
                'Beef Carbon Plantings (Belt)': beef_carbon_plantings_belt_transitions_from_ag,
                'BECCS': beccs_transitions_from_ag}
        
    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    ag_to_non_agr_t_matrices = [
        env_plant_transitions_from_ag.reshape((data.NCELLS, 1)),
        rip_plant_transitions_from_ag.reshape((data.NCELLS, 1)),
        sheep_agroforestry_transitions_from_ag.reshape((data.NCELLS, 1)),
        beef_agroforestry_transitions_from_ag.reshape((data.NCELLS, 1)),
        carbon_plantings_block_transitions_from_ag.reshape((data.NCELLS, 1)),
        sheep_carbon_plantings_belt_transitions_from_ag.reshape((data.NCELLS, 1)),
        beef_carbon_plantings_belt_transitions_from_ag.reshape((data.NCELLS, 1)),
        beccs_transitions_from_ag.reshape((data.NCELLS, 1)),
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


def get_sheep_to_ag_base(data: Data, yr_idx, lumap, separate=False) -> np.ndarray:
    """
    
    """
    sheep_j = tools.get_sheep_natural_land_code(data)

    all_sheep_lumap = (np.ones(data.NCELLS) * sheep_j).astype(np.int8)
    all_dry_lmmap = np.zeros(data.NCELLS)
    l_mrj = lumap2ag_l_mrj(all_sheep_lumap, all_dry_lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    t_ij = data.AG_TMATRIX
    x_mrj = ag_transitions.get_exclude_matrices(data, all_sheep_lumap)

    # Calculate sheep contribution to transition costs
    # Establishment costs
    sheep_af_cells = tools.get_sheep_agroforestry_cells(lumap)

    e_rj = np.zeros((data.NCELLS, data.N_AG_LUS))
    e_rj[sheep_af_cells, :] = t_ij[all_sheep_lumap[sheep_af_cells]]

    e_rj = tools.amortise(e_rj) * data.REAL_AREA[:, np.newaxis]
    e_rj_dry = np.einsum('rj,r->rj', e_rj, all_sheep_lumap == 0)
    e_rj_irr = np.einsum('rj,r->rj', e_rj, all_dry_lmmap == 1)
    e_mrj = np.stack([e_rj_dry, e_rj_irr], axis=0)
    e_mrj = np.einsum('mrj,mrj,mrj->mrj', e_mrj, x_mrj, l_mrj_not)

    # Water license cost
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_water_delta_matrix(w_mrj, l_mrj, data)
    w_delta_mrj = np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not)

    # Carbon costs
    ghg_t_mrj = ag_ghg.get_ghg_transition_penalties(data, all_sheep_lumap)               # <unit: t/ha>      
    ghg_t_mrj_cost = tools.amortise(ghg_t_mrj * settings.CARBON_PRICE_PER_TONNE)     
    ghg_t_mrj_cost = np.einsum('mrj,mrj,mrj->mrj', ghg_t_mrj_cost, x_mrj, l_mrj_not)

    if separate:
        return {'Establishment cost': e_mrj, 'Water license cost': w_delta_mrj, 'GHG emissions cost': ghg_t_mrj_cost}
    
    else:
        return e_mrj + w_delta_mrj + ghg_t_mrj_cost


def get_beef_to_ag_base(data: Data, yr_idx, lumap, separate) -> np.ndarray:
    """
    
    """
    beef_j = tools.get_beef_natural_land_code(data)

    all_beef_lumap = (np.ones(data.NCELLS) * beef_j).astype(np.int8)
    all_dry_lmmap = np.zeros(data.NCELLS)
    l_mrj = lumap2ag_l_mrj(all_beef_lumap, all_dry_lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    t_ij = data.AG_TMATRIX
    x_mrj = ag_transitions.get_exclude_matrices(data, all_beef_lumap)

    # Calculate sheep contribution to transition costs
    # Establishment costs
    e_rj = np.zeros((data.NCELLS, data.N_AG_LUS))
    e_rj = t_ij[beef_j, :]
    e_rj = tools.amortise(e_rj) * data.REAL_AREA[:, np.newaxis]
    e_mrj = np.stack([e_rj] * 2, axis=0)
    e_mrj = np.einsum('mrj,mrj,mrj->mrj', e_mrj, x_mrj, l_mrj_not)

    # Water license cost
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_water_delta_matrix(w_mrj, l_mrj, data)
    w_delta_mrj = np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not)

    # Carbon costs
    ghg_t_mrj = ag_ghg.get_ghg_transition_penalties(data, all_beef_lumap)               # <unit: t/ha>      
    ghg_t_mrj_cost = tools.amortise(ghg_t_mrj * settings.CARBON_PRICE_PER_TONNE)     
    ghg_t_mrj_cost = np.einsum('mrj,mrj,mrj->mrj', ghg_t_mrj_cost, x_mrj, l_mrj_not)

    beef_af_cells = tools.get_beef_agroforestry_cells(lumap)
    non_beef_af_cells = np.array([r for r in range(data.NCELLS) if r not in beef_af_cells])

    if separate:
        return {'Establishment cost': e_mrj, 'Water license cost': w_delta_mrj, 'GHG emissions cost': ghg_t_mrj_cost}
    
    else:
        t_mrj = e_mrj + w_delta_mrj + ghg_t_mrj_cost
        # Set all costs for non-beef-agroforestry cells to zero
        t_mrj[:, non_beef_af_cells, :] = 0
        return t_mrj


def get_sheep_agroforestry_to_ag(
    data: Data, yr_idx, lumap, lmmap, agroforestry_x_r, separate
) -> np.ndarray:
    """
    
    """
    sheep_tcosts = get_sheep_to_ag_base(data, yr_idx, lumap, separate)
    agroforestry_tcosts = get_agroforestry_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in agroforestry_tcosts.items():
            combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * agroforestry_x_r

        for key, array in sheep_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - agroforestry_x_r)

        return combined_costs
    
    else:
        sheep_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                sheep_contr[m, :, j] = (1 - agroforestry_x_r) * sheep_tcosts[m, :, j]

        agroforestry_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                agroforestry_contr[m, :, j] = agroforestry_x_r * agroforestry_tcosts[m, :, j]

        return sheep_contr + agroforestry_contr


def get_beef_agroforestry_to_ag(
    data: Data, yr_idx, lumap, lmmap, agroforestry_x_r, separate
) -> np.ndarray:
    """
    
    """
    beef_tcosts = get_beef_to_ag_base(data, yr_idx, lumap, separate)
    agroforestry_tcosts = get_agroforestry_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in agroforestry_tcosts.items():
            combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * agroforestry_x_r

        for key, array in beef_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - agroforestry_x_r)

        return combined_costs
    
    else:
        beef_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                beef_contr[m, :, j] = (1 - agroforestry_x_r) * beef_tcosts[m, :, j]

        agroforestry_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
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
) -> np.ndarray:
    """
    
    """
    sheep_tcosts = get_sheep_to_ag_base(data, yr_idx, lumap, separate)
    cp_belt_tcosts = get_carbon_plantings_belt_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in cp_belt_tcosts.items():
            combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * cp_belt_x_r

        for key, array in sheep_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - cp_belt_x_r)

        return combined_costs
    
    else:
        sheep_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                sheep_contr[m, :, j] = (1 - cp_belt_x_r) * sheep_tcosts[m, :, j]

        cp_belt_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                cp_belt_contr[m, :, j] = cp_belt_x_r * cp_belt_tcosts[m, :, j]

        return sheep_contr + cp_belt_contr
    

def get_beef_carbon_plantings_belt_to_ag(
    data: Data, yr_idx, lumap, lmmap, cp_belt_x_r, separate
) -> np.ndarray:
    """
    
    """
    beef_tcosts = get_beef_to_ag_base(data, yr_idx, lumap, separate)
    cp_belt_tcosts = get_carbon_plantings_belt_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in cp_belt_tcosts.items():
            combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * cp_belt_x_r

        for key, array in beef_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - cp_belt_x_r)

        return combined_costs
    
    else:
        beef_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                beef_contr[m, :, j] = (1 - cp_belt_x_r) * beef_tcosts[m, :, j]

        cp_belt_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
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
    
    if separate:
        non_ag_to_agr_t_matrices = {lu: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)) for lu in NON_AG_LAND_USES}
    else:
        non_ag_to_agr_t_matrices = {lu: np.zeros(data.NCELLS) for lu in NON_AG_LAND_USES}

    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_ag_to_agr_t_matrices['Environmental Plantings'] = get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    if NON_AG_LAND_USES['Riparian Plantings']:
        non_ag_to_agr_t_matrices['Riparian Plantings'] = get_rip_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    if NON_AG_LAND_USES['Sheep Agroforestry']:
        non_ag_to_agr_t_matrices['Sheep Agroforestry'] = get_sheep_agroforestry_to_ag(data, yr_idx, lumap, lmmap, agroforestry_x_r, separate)
    if NON_AG_LAND_USES['Beef Agroforestry']:
        non_ag_to_agr_t_matrices['Beef Agroforestry'] = get_beef_agroforestry_to_ag(data, yr_idx, lumap, lmmap, agroforestry_x_r, separate)
    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_ag_to_agr_t_matrices['Carbon Plantings (Block)'] = get_carbon_plantings_block_to_ag(data, yr_idx, lumap, lmmap, separate)
    if NON_AG_LAND_USES['Sheep Carbon Plantings (Belt)']:
        non_ag_to_agr_t_matrices['Sheep Carbon Plantings (Belt)'] = get_sheep_carbon_plantings_belt_to_ag(data, yr_idx, lumap, lmmap, cp_belt_x_r, separate)
    if NON_AG_LAND_USES['Beef Carbon Plantings (Belt)']:
        non_ag_to_agr_t_matrices['Beef Carbon Plantings (Belt)'] = get_beef_carbon_plantings_belt_to_ag(data, yr_idx, lumap, lmmap, cp_belt_x_r, separate)
    if NON_AG_LAND_USES['BECCS']:
        non_ag_to_agr_t_matrices['BECCS'] = get_beccs_to_ag(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Note: The order of the keys in the dictionary must match the order of the non-agricultural land uses
        return non_ag_to_agr_t_matrices
        
    non_ag_to_agr_t_matrices = list(non_ag_to_agr_t_matrices.values())
    return np.add.reduce(non_ag_to_agr_t_matrices)


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
    exclude *= tools.get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for riparian plantings may retain that LU
    rp_cells = tools.get_riparian_plantings_cells(lumap)
    exclude[rp_cells] = data.RP_PROPORTION[rp_cells]

    return exclude


def get_exclusions_sheep_agroforestry(
    data: Data, ag_x_mrj: np.ndarray, lumap: np.ndarray
) -> np.ndarray:
    sheep_j = tools.get_sheep_natural_land_code(data)
    sheep_x_r = ag_x_mrj[0, :, sheep_j]
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)

    exclusions = np.zeros(data.NCELLS)

    # Block cells that can't utilise both agroforestry and sheep - natural land.
    intersect = np.intersect1d(np.nonzero(sheep_x_r)[0], np.nonzero(agroforestry_x_r)[0])
    exclusions[intersect] = 1
    return exclusions


def get_exclusions_beef_agroforestry(
    data: Data, ag_x_mrj: np.ndarray, lumap: np.ndarray
) -> np.ndarray:
    beef_j = tools.get_beef_natural_land_code(data)
    beef_x_r = ag_x_mrj[0, :, beef_j]
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)

    exclusions = np.zeros(data.NCELLS)

    # Block cells that can't utilise both agroforestry and beef - natural land.
    intersect = np.intersect1d(np.nonzero(beef_x_r)[0], np.nonzero(agroforestry_x_r)[0])
    exclusions[intersect] = 1
    return exclusions


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
    exclude *= tools.get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for carbon plantings (block) may retain that LU
    exclude[tools.get_carbon_plantings_block_cells(lumap)] = 1

    return exclude


def get_exclusions_sheep_carbon_plantings_belt(
    data: Data, ag_x_mrj: np.ndarray, lumap: np.ndarray
) -> np.ndarray:
    sheep_j = tools.get_sheep_natural_land_code(data)
    sheep_x_r = ag_x_mrj[0, :, sheep_j]
    cp_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    exclusions = np.zeros(data.NCELLS)

    # Block cells that can't utilise both agroforestry and sheep - natural land.
    intersect = np.intersect1d(np.nonzero(sheep_x_r)[0], np.nonzero(cp_x_r)[0])
    exclusions[intersect] = 1
    return exclusions


def get_exclusions_beef_carbon_plantings_belt(
    data: Data, ag_x_mrj: np.ndarray, lumap: np.ndarray
) -> np.ndarray:
    beef_j = tools.get_beef_natural_land_code(data)
    beef_x_r = ag_x_mrj[0, :, beef_j]
    cp_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    exclusions = np.zeros(data.NCELLS)

    # Block cells that can't utilise both agroforestry and beef - natural land.
    intersect = np.intersect1d(np.nonzero(beef_x_r)[0], np.nonzero(cp_x_r)[0])
    exclusions[intersect] = 1
    return exclusions


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
    exclude *= tools.get_exclusions_for_excluding_all_natural_cells(data, lumap)

    # Ensure cells being used for BECCS may retain that LU
    exclude[tools.get_beccs_cells(lumap)] = 1

    return exclude


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
    non_ag_x_matrices = {lu: np.zeros(data.NCELLS) for lu in NON_AG_LAND_USES}

    # Environmental plantings exclusions
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

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_ag_x_matrices = [array.reshape((data.NCELLS, 1)) for array in non_ag_x_matrices.values()]
    return np.concatenate(non_ag_x_matrices, axis=1).astype(np.float32)


def get_lower_bound_non_agricultural_matrices(data: Data, yr) -> np.ndarray:
    """
    Get the non-agricultural lower bound matrix.

    Returns
    -------
    2-D array, indexed by (r,k) where r is the cell and k is the non-agricultural land usage.
    """

    if yr not in data.non_ag_dvars:
        return np.zeros((data.NCELLS, len(NON_AG_LAND_USES)))
        
    return data.non_ag_dvars[yr].astype(np.float32).round(decimals=settings.LB_ROUND_DECMIALS)
