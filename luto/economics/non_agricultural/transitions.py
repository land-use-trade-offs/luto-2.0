import numpy as np
import luto.tools as tools


def get_env_plant_transitions_from_ag(data, lumap, lmmap) -> np.ndarray:
    """
    Get transition costs from agricultural land uses to environmental plantings for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    # Get raw transition costs for each cell to transition to environmental plantings
    ag2ep_transitions_r = (
          data.AG_L_MRJ[0, :, :] @ data.AG2EP_TRANSITION_COSTS[0, :]  # agricultural dry land uses contribution
        + data.AG_L_MRJ[1, :, :] @ data.AG2EP_TRANSITION_COSTS[1, :]  # agricultural irrigated land uses contribution
    )
    # Amortise upfront costs to annualised costs and converted to $ per cell via REAL_AREA
    ag2ep_transitions_r = tools.amortise(ag2ep_transitions_r) * data.REAL_AREA

    # add establishment costs for each cell
    est_costs_r = data.EP_EST_COST_HA
    est_costs_r = tools.amortise(est_costs_r) * data.REAL_AREA
    ag2ep_transitions_r += est_costs_r

    return ag2ep_transitions_r


def get_from_ag_transition_matrix(data, lumap, lmmap) -> np.ndarray:
    """
    Get the matrix containing transition costs from agricultural land uses to non-agricultural land uses.

    Returns
    -------
    2-D array, indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """
    env_plant_transitions_from_ag_r = get_env_plant_transitions_from_ag(data, lumap, lmmap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    ag_to_non_agr_t_matrices = [
        env_plant_transitions_from_ag_r.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(ag_to_non_agr_t_matrices, axis=1)


def get_env_plantings_to_ag(data, lumap) -> np.ndarray:
    """
    Get transition costs from environmental plantings to agricultural land uses for each cell.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    n_ag_lms, ncells, n_ag_lus = data.AG_L_MRJ.shape
    ep2ag_transitions_mrj = np.zeros((n_ag_lms, ncells, n_ag_lus))

    _, non_ag_cells = tools.get_ag_and_non_ag_cells(lumap)
    env_plant_cells = non_ag_cells  # If additional non-agricultural land uses are added, this line must be updated.

    # apply environmental plantings to agriculture transitions for each cell by
    # repeating the transitions vector for each EP cell
    env_plant_transitions = np.repeat(
        data.EP2AG_TRANSITION_COSTS_HA[:, :, np.newaxis], env_plant_cells.size, axis=2
    ).reshape((n_ag_lms, env_plant_cells.size, n_ag_lus))

    ep2ag_transitions_mrj[:, env_plant_cells, :] = env_plant_transitions

    # Amortise upfront costs to annualised costs and converted to $ per cell via REAL_AREA
    ep2ag_transitions_mrj = tools.amortise(ep2ag_transitions_mrj) * data.REAL_AREA[np.newaxis, :, np.newaxis]

    return ep2ag_transitions_mrj


def get_to_ag_transition_matrix(data, lumap) -> np.ndarray:
    """
    Get the matrix containing transition costs from non-agricultural land uses to agricultural land uses.

    Returns
    -------
    3-D array, indexed by (m, r, j).
    """
    env_plant_transitions_to_ag_mrj = get_env_plantings_to_ag(data, lumap)

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
    return np.concatenate(non_ag_x_matrices, axis=1)
