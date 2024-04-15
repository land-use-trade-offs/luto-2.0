import numpy as np

import luto.settings as settings
from luto.data import Data


def get_rev_env_plantings(data: Data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        The revenue produced by environmental plantings for each cell. A 1-D array indexed by cell.
    """
    # Multiply carbon reduction by carbon price for each cell and adjust for resfactor.
    return data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA * settings.CARBON_PRICE_PER_TONNE


def get_rev_rip_plantings(data: Data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        The revenue produced by riparian plantings for each cell. A 1-D array indexed by cell.
    """
    return get_rev_env_plantings(data)


def get_rev_agroforestry(data: Data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        The revenue produced by agroforestry for each cell. A 1-D array indexed by cell.
    """
    return get_rev_env_plantings(data)


def get_rev_carbon_plantings_block(data: Data) -> np.ndarray:
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


def get_rev_carbon_plantings_belt(data: Data) -> np.ndarray:
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


def get_rev_beccs(data: Data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
    """
    base_rev = np.nan_to_num(data.BECCS_REV_AUD_HA_YR) * data.REAL_AREA
    return base_rev + np.nan_to_num(data.BECCS_TCO2E_HA_YR) * data.REAL_AREA * settings.CARBON_PRICE_PER_TONNE


def get_rev_matrix(data: Data) -> np.ndarray:
    """
    Gets the matrix containing the revenue produced by each non-agricultural land use for each cell.

    Parameters:
        data (Data): The data object containing the necessary information.

    Returns:
        np.ndarray.
    """
    env_plantings_rev_matrix = get_rev_env_plantings(data)
    rip_plantings_rev_matrix = get_rev_rip_plantings(data)
    agroforestry_rev_matrix = get_rev_agroforestry(data)
    carbon_plantings_block_rev_matrix = get_rev_carbon_plantings_block(data)
    carbon_plantings_belt_rev_matrix = get_rev_carbon_plantings_belt(data)
    beccs_rev_matrix = get_rev_beccs(data)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_rev_matrices = [
        env_plantings_rev_matrix.reshape((data.NCELLS, 1)),
        rip_plantings_rev_matrix.reshape((data.NCELLS, 1)),
        agroforestry_rev_matrix.reshape((data.NCELLS, 1)),
        carbon_plantings_block_rev_matrix.reshape((data.NCELLS, 1)),
        carbon_plantings_belt_rev_matrix.reshape((data.NCELLS, 1)),
        beccs_rev_matrix.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_rev_matrices, axis=1)