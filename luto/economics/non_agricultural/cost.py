import numpy as np

from luto.data import Data
import luto.settings as settings


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
    """
    env_plantings_costs = get_cost_env_plantings(data)
    rip_plantings_costs = get_cost_rip_plantings(data)
    agroforestry_costs = get_cost_agroforestry(data)
    carbon_plantings_block_costs = get_cost_carbon_plantings_block(data)
    carbon_plantings_belt_costs = get_cost_carbon_plantings_belt(data)
    beccs_costs = get_cost_beccs(data)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_c_matrices = [
        env_plantings_costs.reshape((data.NCELLS, 1)),
        rip_plantings_costs.reshape((data.NCELLS, 1)),
        agroforestry_costs.reshape((data.NCELLS, 1)),
        carbon_plantings_block_costs.reshape((data.NCELLS, 1)),
        carbon_plantings_belt_costs.reshape((data.NCELLS, 1)),
        beccs_costs.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_c_matrices, axis=1)