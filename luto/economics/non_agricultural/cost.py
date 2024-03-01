import numpy as np

import luto.settings as settings
from luto.non_ag_landuses import NON_AG_LAND_USES


def get_cost_env_plantings(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Cost of environmental plantings for each cell. 1-D array Indexed by cell.
    """
    return settings.ENV_PLANTING_COST_PER_HA_PER_YEAR * data.REAL_AREA


def get_cost_rip_plantings(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Cost of environmental plantings for each cell. 1-D array Indexed by cell.
    """
    return settings.RIPARIAN_PLANTING_COST_PER_HA_PER_YEAR * data.REAL_AREA


def get_cost_agroforestry(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Cost of environmental plantings for each cell. 1-D array Indexed by cell.
    """
    return settings.AGROFORESTRY_COST_PER_HA_PER_YEAR * data.REAL_AREA


def get_cost_matrix(data):
    """
    Returns non-agricultural c_rk matrix of costs per cell and land use.
    """

    non_agr_c_matrices = {use: np.zeros(data.NCELLS, 1) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_agr_c_matrices['Environmental Plantings'] = get_cost_env_plantings(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_agr_c_matrices['Riparian Plantings'] = get_cost_rip_plantings(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Agroforestry']:
        non_agr_c_matrices['Agroforestry'] = get_cost_agroforestry(data).reshape((data.NCELLS, 1))

    non_agr_c_matrices = list(non_agr_c_matrices.values())
    return np.concatenate(non_agr_c_matrices, axis=1)