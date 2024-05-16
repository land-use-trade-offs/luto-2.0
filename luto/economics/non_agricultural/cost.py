import numpy as np

from luto.data import Data
import luto.settings as settings
from luto.settings import NON_AG_LAND_USES
from luto import tools


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


def get_cost_agroforestry_base(data: Data) -> np.ndarray:
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


def get_cost_sheep_agroforestry(
    data: Data, 
    ag_c_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_c_mrj: agricultural cost matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_natural_land_code(data)

    # Only use the dryland version of natural land sheep
    sheep_cost = ag_c_mrj[0, :, sheep_j]
    base_agroforestry_cost = get_cost_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_cost * agroforestry_x_r
    sheep_contr = sheep_cost * (1 - agroforestry_x_r)
    return agroforestry_contr + sheep_contr


def get_cost_beef_agroforestry(
    data: Data, 
    ag_c_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_c_mrj: agricultural cost matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_natural_land_code(data)

    # Only use the dryland version of natural land sheep
    beef_cost = ag_c_mrj[0, :, beef_j]
    base_agroforestry_cost = get_cost_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_cost * agroforestry_x_r
    beef_contr = beef_cost * (1 - agroforestry_x_r)
    return agroforestry_contr + beef_contr


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


def get_cost_carbon_plantings_belt_base(data) -> np.ndarray:
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


def get_cost_sheep_carbon_plantings_belt(
    data: Data, 
    ag_c_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_c_mrj: agricultural cost matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_natural_land_code(data)

    # Only use the dryland version of natural land sheep
    sheep_cost = ag_c_mrj[0, :, sheep_j]
    base_cp_cost = get_cost_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_cost * cp_belt_x_r
    sheep_contr = sheep_cost * (1 - cp_belt_x_r)
    return cp_contr + sheep_contr


def get_cost_beef_carbon_plantings_belt(
    data: Data, 
    ag_c_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_c_mrj: agricultural cost matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_natural_land_code(data)

    # Only use the dryland version of natural land sheep
    beef_cost = ag_c_mrj[0, :, beef_j]
    base_cp_cost = get_cost_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_cost * cp_belt_x_r
    beef_contr = beef_cost * (1 - cp_belt_x_r)
    return cp_contr + beef_contr



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


def get_cost_matrix(data: Data, ag_c_mrj: np.ndarray, lumap):
    """
    Returns non-agricultural c_rk matrix of costs per cell and land use.

    Parameters:
    - data: The input data containing information about the cells and land use.

    Returns:
    - cost_matrix: A 2D numpy array of costs per cell and land use.
    """
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    non_agr_c_matrices = {use: np.zeros((data.NCELLS, 1)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_agr_c_matrices['Environmental Plantings'] = get_cost_env_plantings(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_agr_c_matrices['Riparian Plantings'] = get_cost_rip_plantings(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Sheep Agroforestry']:
        non_agr_c_matrices['Sheep Agroforestry'] = get_cost_sheep_agroforestry(data, ag_c_mrj, agroforestry_x_r).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Beef Agroforestry']:
        non_agr_c_matrices['Beef Agroforestry'] = get_cost_beef_agroforestry(data, ag_c_mrj, agroforestry_x_r).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_agr_c_matrices['Carbon Plantings (Block)'] = get_cost_carbon_plantings_block(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Sheep Carbon Plantings (Belt)']:
        non_agr_c_matrices['Sheep Carbon Plantings (Belt)'] = get_cost_sheep_carbon_plantings_belt(data, ag_c_mrj, cp_belt_x_r).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Beef Carbon Plantings (Belt)']:
        non_agr_c_matrices['Beef Carbon Plantings (Belt)'] = get_cost_beef_carbon_plantings_belt(data, ag_c_mrj, cp_belt_x_r).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['BECCS']:
        non_agr_c_matrices['BECCS'] = get_cost_beccs(data).reshape((data.NCELLS, 1))

    non_agr_c_matrices = list(non_agr_c_matrices.values())
    return np.concatenate(non_agr_c_matrices, axis=1)