import numpy as np
from luto.non_ag_landuses import NON_AG_LAND_USES
import luto.settings as settings


def get_rev_env_plantings(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        The cost of environmental plantings for each cell. A 1-D array indexed by cell.
    """
    # Multiply carbon reduction by carbon price for each cell and adjust for resfactor.
    return data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA * settings.CARBON_PRICE_PER_TONNE


def get_rev_rip_plantings(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        The cost of riparian plantings for each cell. A 1-D array indexed by cell.
    """
    return get_rev_env_plantings(data)


def get_rev_agroforestry(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        The cost of agroforestry for each cell. A 1-D array indexed by cell.
    """
    return get_rev_env_plantings(data)


def get_rev_carbon_plantings_block(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        The cost of carbon plantings (block) for each cell. A 1-D array indexed by cell.
    """
    # Multiply carbon reduction by carbon price for each cell and adjust for resfactor.
    return data.CP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA * settings.CARBON_PRICE_PER_TONNE


def get_rev_carbon_plantings_belt(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        The cost of carbon plantings (belt) for each cell. A 1-D array indexed by cell.
    """
    # Multiply carbon reduction by carbon price for each cell and adjust for resfactor.
    return data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA * settings.CARBON_PRICE_PER_TONNE


def get_rev_matrix(data) -> np.ndarray:
    """

    """
    non_agr_rev_matrices = {use: np.zeros((data.NCELLS, 1)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_agr_rev_matrices['Environmental Plantings'] = get_rev_env_plantings(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_agr_rev_matrices['Riparian Plantings'] = get_rev_rip_plantings(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Agroforestry']:
        non_agr_rev_matrices['Agroforestry'] = get_rev_agroforestry(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Belt)']:
        non_agr_rev_matrices['Carbon Plantings (Belt)'] = get_rev_carbon_plantings_belt(data).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_agr_rev_matrices['Carbon Plantings (Block)'] = get_rev_carbon_plantings_block(data).reshape((data.NCELLS, 1))

    non_agr_rev_matrices = list(non_agr_rev_matrices.values())

    return np.concatenate(non_agr_rev_matrices, axis=1)