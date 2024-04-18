import numpy as np

from luto.data import Data
import luto.settings as settings
from luto.non_ag_landuses import NON_AG_LAND_USES


def get_cost_env_plantings(data: Data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of environmental plantings for each cell. 1-D array Indexed by cell.
    """
    return settings.ENV_PLANTING_COST_PER_HA_PER_YEAR * data.REAL_AREA


def get_cost_rip_plantings(data: Data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of riparian plantings for each cell. 1-D array Indexed by cell.
    """
    return settings.RIPARIAN_PLANTING_COST_PER_HA_PER_YEAR * data.REAL_AREA


def get_cost_agroforestry(data: Data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of agroforestry for each cell. 1-D array Indexed by cell.
    """
    return settings.AGROFORESTRY_COST_PER_HA_PER_YEAR * data.REAL_AREA


def get_cost_carbon_plantings_block(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Cost of carbon plantings (block arrangement) for each cell. 1-D array Indexed by cell.
    """
    return settings.CARBON_PLANTING_BLOCK_COST_PER_HA_PER_YEAR * data.REAL_AREA


def get_cost_carbon_plantings_belt(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of carbon plantings (belt arrangement) for each cell. 1-D array Indexed by cell.
    """
    return settings.CARBON_PLANTING_BELT_COST_PER_HA_PER_YEAR * data.REAL_AREA


def get_cost_beccs(data: Data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of BECCS for each cell. 1-D array Indexed by cell.
    """
    return np.nan_to_num(data.BECCS_COSTS_AUD_HA_YR) * data.REAL_AREA


def get_cost_matrix(data: Data):
    """
    Returns non-agricultural c_rk matrix of costs per cell and land use.

    Parameters:
    - data: The input data containing information about the cells and land use.

    Returns:
    - cost_matrix: A 2D numpy array of costs per cell and land use.
    """

    non_agr_c_matrices = {use: np.zeros((data.NCELLS, 1)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_agr_c_matrices['Environmental Plantings'] = get_cost_env_plantings(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_agr_c_matrices['Riparian Plantings'] = get_cost_rip_plantings(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Agroforestry']:
        non_agr_c_matrices['Agroforestry'] = get_cost_agroforestry(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Belt)']:
        non_agr_c_matrices['Carbon Plantings (Belt)'] = get_cost_carbon_plantings_belt(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_agr_c_matrices['Carbon Plantings (Block)'] = get_cost_carbon_plantings_block(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['BECCS']:
        non_agr_c_matrices['BECCS'] = get_cost_beccs(data).reshape((data.NCELLS, 1))

    non_agr_c_matrices = list(non_agr_c_matrices.values())
    return np.concatenate(non_agr_c_matrices, axis=1)