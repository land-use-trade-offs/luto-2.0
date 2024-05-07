import numpy as np
from luto.settings import NON_AG_LAND_USES

from luto.data import Data


def get_wreq_matrix_env_planting(data: Data) -> np.ndarray:
    """
    Get water requirements vector of environmental plantings.

    To get the water requirements of environmental plantings, subtract the baseline 
    water yields from the shallow-rooted water yields in the data. This represents
    how much water would be used if modified (i.e., cleared) land area was reforested
    with native pre-European vegetation communities (WATER_YIELD_BASE). Pre-European 
    communities include both deep-rooted (i.e., forests/woodlands) and shallow-rooted 
    communities - natural vegetation is not all deep-rooted.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return (data.WATER_YIELD_BASE_SR - data.WATER_YIELD_BASE) * data.REAL_AREA


def get_wreq_matrix_carbon_plantings_block(data) -> np.ndarray:
    """
    Get water requirements vector of carbon plantings (block arrangement).

    To get the water requirements of carbon plantings, subtract the deep-rooted 
    water yields from the shallow-rooted water yields in the data. This represents
    how much water would be used if modified (i.e., cleared) land area was reforested
    with wall-to-wall deep-rooted tree species (WATER_YIELD_BASE_DR). The assumption 
    here is that plantations are all deep-rooted and hence use more water.
    
    Returns
    -------
    1-D array, indexed by cell.
    """
    # return get_wreq_matrix_env_planting(data)
    return (data.WATER_YIELD_BASE_SR - data.WATER_YIELD_BASE_DR) * data.REAL_AREA


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

    To get the water requirements of agroforestry, subtract the baseline
    water yields from the shallow-rooted water yields in the data.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_wreq_matrix_env_planting(data)


def get_wreq_matrix_carbon_plantings_belt(data) -> np.ndarray:
    """
    Get water requirements vector of carbon plantings (belt arrangement).

    Note: this is the same as for carbon plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_wreq_matrix_carbon_plantings_block(data)


def get_wreq_matrix_beccs(data) -> np.ndarray:
    """
    Get water requirements vector of BECCS.

    Note: this is the same as for carbon plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_wreq_matrix_carbon_plantings_block(data)


def get_wreq_matrix(data: Data) -> np.ndarray:
    """
    Get the water requirements matrix for all non-agricultural land uses.

    Parameters
    ----------
    data : object
        The data object containing necessary information for calculating the water requirements.

    Returns
    -------
    np.ndarray
        The water requirements matrix for all non-agricultural land uses.
        Indexed by (r, k) where r is the cell index and k is the non-agricultural land usage index.
    """

    non_agr_wreq_matrices = {use: np.zeros((data.NCELLS, 1)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_agr_wreq_matrices['Environmental Plantings'] = get_wreq_matrix_env_planting(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_agr_wreq_matrices['Riparian Plantings'] = get_wreq_matrix_rip_planting(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Agroforestry']:
        non_agr_wreq_matrices['Agroforestry'] = get_wreq_matrix_agroforestry(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_agr_wreq_matrices['Carbon Plantings (Block)'] = get_wreq_matrix_carbon_plantings_block(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Belt)']:
        non_agr_wreq_matrices['Carbon Plantings (Belt)'] = get_wreq_matrix_carbon_plantings_belt(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['BECCS']:
        non_agr_wreq_matrices['BECCS'] = get_wreq_matrix_beccs(data).reshape((data.NCELLS, 1))

    non_agr_wreq_matrices = list(non_agr_wreq_matrices.values())

    return np.concatenate(non_agr_wreq_matrices, axis=1)
