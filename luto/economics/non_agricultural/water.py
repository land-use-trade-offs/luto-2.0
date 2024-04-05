import numpy as np
from luto.non_ag_landuses import NON_AG_LAND_USES


def get_wreq_matrix_env_planting(data) -> np.ndarray:
    """
    Get water requirements vector of environmental plantings.

    To get the water requirements of environmental plantings, subtract the deep-rooted
    water yields from the shallow-rooted water yields in the data. This represents
    how much water would be used if the area was reforested.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return (data.WATER_YIELD_BASE_SR - data.WATER_YIELD_BASE) * data.REAL_AREA


def get_wreq_matrix_rip_planting(data) -> np.ndarray:
    """
    Get water requirements vector of riparian plantings.

    To get the water requirements of riparian plantings, subtract the deep-rooted
    water yields from the shallow-rooted water yields in the data.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_wreq_matrix_env_planting(data)


def get_wreq_matrix_agroforestry(data) -> np.ndarray:
    """
    Get water requirements vector of agroforestry.

    To get the water requirements of agroforestry, subtract the deep-rooted
    water yields from the shallow-rooted water yields in the data.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_wreq_matrix_env_planting(data)


def get_wreq_matrix_carbon_plantings_block(data) -> np.ndarray:
    """
    Get water requirements vector of carbon plantings (block arrangement).

    Note: this is the same as for environmental plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_wreq_matrix_env_planting(data)


def get_wreq_matrix_carbon_plantings_belt(data) -> np.ndarray:
    """
    Get water requirements vector of carbon plantings (belt arrangement).

    Note: this is the same as for environmental plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_wreq_matrix_env_planting(data)


def get_wreq_matrix_beccs(data) -> np.ndarray:
    """
    Get water requirements vector of BECCS.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_wreq_matrix_env_planting(data)


def get_wreq_matrix(data) -> np.ndarray:
    """
    Get the water requirements matrix for all non-agricultural land uses.

    Returns
    -------
    np.ndarray
        Indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """

    non_agr_wreq_matrices = {use: np.zeros((data.NCELLS, 1)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_agr_wreq_matrices['Environmental Plantings'] = get_wreq_matrix_env_planting(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_agr_wreq_matrices['Riparian Plantings'] = get_wreq_matrix_rip_planting(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Agroforestry']:
        non_agr_wreq_matrices['Agroforestry'] = get_wreq_matrix_agroforestry(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Belt)']:
        non_agr_wreq_matrices['Carbon Plantings (Belt)'] = get_wreq_matrix_carbon_plantings_belt(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_agr_wreq_matrices['Carbon Plantings (Block)'] = get_wreq_matrix_carbon_plantings_block(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['BECCS']:
        non_agr_wreq_matrices['BECCS'] = get_wreq_matrix_beccs(data).reshape((data.NCELLS, 1))

    non_agr_wreq_matrices = list(non_agr_wreq_matrices.values())

    return np.concatenate(non_agr_wreq_matrices, axis=1)
