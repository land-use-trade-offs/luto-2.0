import numpy as np
from luto import settings
import luto.tools as tools
import luto.economics.agricultural.water as ag_water
from luto.non_ag_landuses import NON_AG_LAND_USES


def get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get transition costs from agricultural land uses to environmental plantings for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_ag_to_ep_t = data.AG2EP_TRANSITION_COSTS_HA
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


def get_rip_plant_transitions_from_ag(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get transition costs from agricultural land uses to riparian plantings for each cell.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_costs = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap)
    fencing_cost = data.RP_FENCING_LENGTH * data.REAL_AREA * settings.RIPARIAN_PLANTINGS_FENCING_COST_PER_M
    return base_costs + fencing_cost


def get_agroforestry_transitions_from_ag(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get transition costs from agricultural land uses to agroforestry for each cell.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    base_costs = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap)
    fencing_cost = settings.AF_FENCING_LENGTH * data.REAL_AREA * settings.AGROFORESTRY_FENCING_COST_PER_M
    return base_costs + fencing_cost


def get_from_ag_transition_matrix(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get the matrix containing transition costs from agricultural land uses to non-agricultural land uses.

    Returns
    -------
    2-D array, indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """
    env_plant_transitions_from_ag_r = get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap)
    rip_plant_transitions_from_ag_r = get_rip_plant_transitions_from_ag(data, yr_idx, lumap, lmmap)
    agroforestry_transitions_from_ag_r = get_agroforestry_transitions_from_ag(data, yr_idx, lumap, lmmap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    ag_to_non_agr_t_matrices = [
        env_plant_transitions_from_ag_r.reshape((data.NCELLS, 1)),
        rip_plant_transitions_from_ag_r.reshape((data.NCELLS, 1)),
        agroforestry_transitions_from_ag_r.reshape((data.NCELLS, 1)),
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


def get_rip_plantings_to_ag(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get transition costs from riparian plantings to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)


def get_agroforestry_to_ag(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get transition costs from agroforestry to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)


def get_to_ag_transition_matrix(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get the matrix containing transition costs from non-agricultural land uses to agricultural land uses.

    Returns
    -------
    3-D array, indexed by (m, r, j).
    """

    ag_to_non_agr_t_matrices = {use: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        ag_to_non_agr_t_matrices['Environmental Plantings'] = get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)

    if NON_AG_LAND_USES['Riparian Plantings']:
        ag_to_non_agr_t_matrices['Riparian Plantings'] = get_rip_plantings_to_ag(data, yr_idx, lumap, lmmap)

    if NON_AG_LAND_USES['Agroforestry']:
        ag_to_non_agr_t_matrices['Agroforestry'] = get_agroforestry_to_ag(data, yr_idx, lumap, lmmap)

    ag_to_non_agr_t_matrices = list(ag_to_non_agr_t_matrices.values())

    # Element-wise sum each mrj-indexed matrix to get the final transition matrix
    return np.add.reduce(ag_to_non_agr_t_matrices)


def get_non_ag_transition_matrix(data, yr_idx, lumap, lmmap) -> np.ndarray:
    """
    Get the matrix that contains transition costs for non-agricultural land uses.
    That is, the cost of transitioning between non-agricultural land uses.
    """
    t_rk = np.zeros((data.NCELLS, data.N_NON_AG_LUS))

    # Currently, non of the non-agricultural land uses may transition between each other.
    # Thus, transition costs need not be considered.
    return t_rk


def get_exclusions_environmental_plantings(data, lumap) -> np.ndarray:
    """
    Get the exclusion array for the environmental plantings land use.
    """
    # Get (agricultural) land uses that cannot transition to environmental plantings
    excluded_ag_lus_cells = np.where(np.isnan(data.AG2EP_TRANSITION_COSTS_HA))[0]

    # Create the exclude array as having 0 for every cell that has an excluded land use and 1 otherwise.
    exclude = (~np.isin(lumap, excluded_ag_lus_cells)).astype(int)

    # Ensure other non-agricultural land uses are excluded
    rip_plantings_cells = tools.get_riparian_plantings_cells(lumap)
    exclude[rip_plantings_cells] = 0
    agroforestry_cells = tools.get_agroforestry_cells(lumap)
    exclude[agroforestry_cells] = 0

    return exclude


def get_exclusions_riparian_plantings(data, lumap) -> np.ndarray:
    """
    Get the exclusion array for the Riparian plantings land use.
    Return a 1-D array indexed by r that represents how much RP can be utilised.
    """
    exclude = (data.RP_PROPORTION).astype(np.float32)

    # Exclude all cells used for natural land uses
    # TODO - this means natural LU cells cannot transition to agriculture/RP splits, even though
    # they may transition to agriculture without the RP portion. Think about this before merging.
    natural_lu_cells = tools.get_natural_lu_cells(data, lumap)
    exclude[natural_lu_cells] = 0

    # Ensure other non-agricultural land uses are excluded
    env_plantings_cells = tools.get_env_plantings_cells(lumap)
    exclude[env_plantings_cells] = 0
    agroforestry_cells = tools.get_agroforestry_cells(lumap)
    exclude[agroforestry_cells] = 0

    return exclude

    
def get_exclusions_agroforestry(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much agroforestry can possibly 
    be done at each cell.
    """
    exclude = (np.ones(data.NCELLS) * settings.AF_PROPORTION).astype(np.float32)

    # Exclude all cells used for natural land uses
    natural_lu_cells = tools.get_natural_lu_cells(data, lumap)
    exclude[natural_lu_cells] = 0

    # Ensure other non-agricultural land uses are excluded
    env_plantings_cells = tools.get_env_plantings_cells(lumap)
    exclude[env_plantings_cells] = 0
    rip_plantings_cells = tools.get_riparian_plantings_cells(lumap)
    exclude[rip_plantings_cells] = 0

    return exclude


def get_exclude_matrices(data, lumap) -> np.ndarray:
    """
    Get the non-agricultural exclusions matrix.

    Returns
    -------
    2-D array, indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """

    non_ag_x_matrices = {use: np.zeros((data.NCELLS, 1)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_ag_x_matrices['Environmental Plantings'] = get_exclusions_environmental_plantings(data, lumap).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_ag_x_matrices['Riparian Plantings'] = get_exclusions_riparian_plantings(data, lumap).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Agroforestry']:
        non_ag_x_matrices['Agroforestry'] = get_exclusions_agroforestry(data, lumap).reshape((data.NCELLS, 1))

    non_ag_x_matrices = list(non_ag_x_matrices.values())

    # Stack list and return to get x_rk
    return np.concatenate(non_ag_x_matrices, axis=1).astype(np.float32)


def get_lower_bound_matrices(data, non_ag_dvars, index) -> np.ndarray:
    """
    Get the non-agricultural lower bound matrix.

    Returns
    -------
    2-D array, indexed by (r,k) where r is the cell and k is the non-agricultural land usage.
    """

    non_ag_lb_rk = non_ag_dvars.get(index, np.empty(0))

    if not non_ag_lb_rk.size:
        return np.zeros((data.NCELLS, len([lu for lu in NON_AG_LAND_USES if NON_AG_LAND_USES[lu]])))
    return non_ag_lb_rk.astype(np.float32)
