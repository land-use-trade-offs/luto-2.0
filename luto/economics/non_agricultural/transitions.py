import numpy as np
import luto.tools as tools
import luto.economics.agricultural.water as ag_water


def get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get transition costs from agricultural land uses to environmental plantings for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_ag_to_ep_t = data.AG2EP_TRANSITION_COSTS
    l_mrj = tools.lumap2ag_l_mrj(lumap, lmmap)
    base_ag_to_ep_t_mrj = np.broadcast_to(base_ag_to_ep_t, (2, data.NCELLS, base_ag_to_ep_t.shape[0]))

    # Amortise base costs to be annualised
    base_ag_to_ep_t_mrj = tools.amortise(base_ag_to_ep_t_mrj)

    # Add cost of water license and cost of removing irrigation where relevant (pre-amortised)
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_water_delta_matrix(w_mrj, l_mrj, data)
    ag_to_ep_t_mrj = base_ag_to_ep_t_mrj + w_delta_mrj

    # Get raw transition costs for each cell to transition to environmental plantings
    ag2ep_transitions_r = np.nansum(l_mrj * ag_to_ep_t_mrj, axis=(0, 2))

    # Add establishment costs for each cell
    est_costs_r = data.EP_EST_COST_HA

    # Amortise establishment costs  to be annualised
    est_costs_r = tools.amortise(est_costs_r)
    ag2ep_transitions_r += est_costs_r

    return ag2ep_transitions_r * data.REAL_AREA


def get_from_ag_transition_matrix(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get the matrix containing transition costs from agricultural land uses to non-agricultural land uses.

    Returns
    -------
    2-D array, indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """
    env_plant_transitions_from_ag_r = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    ag_to_non_agr_t_matrices = [
        env_plant_transitions_from_ag_r.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(ag_to_non_agr_t_matrices, axis=1)


def get_env_plantings_to_ag(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get transition costs from environmental plantings to agricultural land uses for each cell.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    # Get base transition costs: add cost of installing irrigation
    base_ep_to_ag_t = data.EP2AG_TRANSITION_COSTS_HA

    # Get water license price and costs of removing irrigation where appropriate
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    l_mrj = tools.lumap2ag_l_mrj(lumap, lmmap)
    w_delta_mrj = tools.get_water_delta_matrix(w_mrj, l_mrj, data)

    # Reshape and amortise upfront costs to annualised costs
    base_ep_to_ag_t_mrj = np.broadcast_to(base_ep_to_ag_t, (2, data.NCELLS, base_ep_to_ag_t.shape[0]))
    base_ep_to_ag_t_mrj = tools.amortise(base_ep_to_ag_t_mrj)

    ep_to_ag_t_mrj = base_ep_to_ag_t_mrj + w_delta_mrj

    # Apply costs only to non-agricultural cells.
    ag_cells, _ = tools.get_ag_and_non_ag_cells(lumap)
    ep_to_ag_t_mrj[:, ag_cells, :] = 0

    return ep_to_ag_t_mrj * data.REAL_AREA[np.newaxis, :, np.newaxis]


def get_to_ag_transition_matrix(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get the matrix containing transition costs from non-agricultural land uses to agricultural land uses.

    Returns
    -------
    3-D array, indexed by (m, r, j).
    """
    env_plant_transitions_to_ag_mrj = get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)

    # Reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    ag_to_non_agr_t_matrices = [
        env_plant_transitions_to_ag_mrj,
    ]

    # Element-wise sum each mrj-indexed matrix to get the final transition matrix
    return np.add.reduce(ag_to_non_agr_t_matrices)


def get_exclusions_environmental_plantings(data, lumap) -> np.ndarray:
    """
    Get an array of cells that cannot be transitioned to environmental plantings.
    """
    # Get (agricultural) land uses that cannot transition to environmental plantings
    excluded_ag_lus = np.where(np.isnan(data.AG2EP_TRANSITION_COSTS[0]))[0]
    # Return an array with 0 for every cell that has an excluded land use and 1 otherwise.
    return (~np.isin(lumap, excluded_ag_lus)).astype(int)


def get_exclude_matrices(data, lumap) -> np.ndarray:
    """
    Get the non-agricultural exclusions matrix.

    Returns
    -------
    2-D array, indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """
    # Environmental plantings exclusions
    env_plant_exclusions = get_exclusions_environmental_plantings(data, lumap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_ag_x_matrices = [
        env_plant_exclusions.reshape((data.NCELLS, 1)),
    ]

    # Stack list and return to get x_rk
    return np.concatenate(non_ag_x_matrices, axis=1).astype(bool)
