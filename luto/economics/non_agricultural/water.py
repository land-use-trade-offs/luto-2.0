import numpy as np

from luto.data import Data


def get_wreq_matrix_env_planting(data: Data) -> np.ndarray:
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


def get_wreq_matrix_rip_planting(data: Data) -> np.ndarray:
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


def get_wreq_matrix_agroforestry(data: Data) -> np.ndarray:
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


def get_wreq_matrix(data: Data) -> np.ndarray:
    """
    Get the water requirements matrix for all non-agricultural land uses.

    Returns
    -------
    np.ndarray
        Indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """
    env_plant_wreq_matrix = get_wreq_matrix_env_planting(data)
    rip_plant_wreq_matrix = get_wreq_matrix_rip_planting(data)
    agroforestry_wreq_matrix = get_wreq_matrix_agroforestry(data)
    carbon_plantings_block_wreq_matrix = get_wreq_matrix_carbon_plantings_block(data)
    carbon_plantings_belt_wreq_matrix = get_wreq_matrix_carbon_plantings_belt(data)
    beccs_wreq_matrix = get_wreq_matrix_beccs(data)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_ag_wreq_matrices = [
        env_plant_wreq_matrix.reshape((data.NCELLS, 1)),
        rip_plant_wreq_matrix.reshape((data.NCELLS, 1)),
        agroforestry_wreq_matrix.reshape((data.NCELLS, 1)),
        carbon_plantings_block_wreq_matrix.reshape((data.NCELLS, 1)),
        carbon_plantings_belt_wreq_matrix.reshape((data.NCELLS, 1)),
        beccs_wreq_matrix.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_ag_wreq_matrices, axis=1)
