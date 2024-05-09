import numpy as np


def get_quantity_env_plantings(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Indexed by (c, r): represents the quantity commodity c produced by cell r
        if used for environmental plantings.
        A matrix of zeros because environmental plantings doesn't produce anything.
    """
    return np.zeros((data.NCMS, data.NCELLS))


def get_quantity_rip_plantings(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Indexed by (c, r): represents the quantity commodity c produced by cell r
        if used for riparian plantings.
        A matrix of zeros because Riparian Plantings doesn't produce anything.
    """

    return np.zeros((data.NCMS, data.NCELLS))


def get_quantity_agroforestry(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Indexed by (c, r): represents the quantity commodity c produced by cell r
        if used for agroforestry.
        A matrix of zeros because agroforestry doesn't produce anything.
    """

    return np.zeros((data.NCMS, data.NCELLS))


def get_quantity_carbon_plantings_block(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Indexed by (c, r): represents the quantity commodity c produced by cell r
        if used for carbon plantings (block).
        A matrix of zeros because carbon plantings doesn't produce anything.
    """
    return np.zeros((data.NCMS, data.NCELLS))


def get_quantity_carbon_plantings_belt(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Indexed by (c, r): represents the quantity commodity c produced by cell r
        if used for carbon plantings (belt).
        A matrix of zeros because carbon plantings doesn't produce anything.
    """
    return np.zeros((data.NCMS, data.NCELLS))


def get_quantity_beccs(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Indexed by (c, r): represents the quantity commodity c produced by cell r
        if used for BECCS.
        A matrix of zeros because BECCS doesn't produce anything.
    """
    return np.zeros((data.NCMS, data.NCELLS))


def get_quantity_matrix(data) -> np.ndarray:
    """
    Get the non-agricultural quantity matrix q_crk.
    Values represent the yield of each commodity c from the cell r when using
    the non-agricultural land use k.

    Parameters:
    - data: The input data containing information about the land use and commodities.

    Returns:
    - np.ndarray: The non-agricultural quantity matrix q_crk.
    """
    env_plantings_quantity_matrix = get_quantity_env_plantings(data)
    rip_plantings_quantity_matrix = get_quantity_rip_plantings(data)
    agroforestry_quantity_matrix = get_quantity_agroforestry(data)
    carbon_plantings_block_quantity_matrix = get_quantity_carbon_plantings_block(data)
    carbon_plantings_belt_quantity_matrix = get_quantity_carbon_plantings_belt(data)
    beccs_quantity_matrix = get_quantity_beccs(data)

    # reshape each matrix to be indexed (c, r, k) and concatenate on the k indexing
    non_agr_quantity_matrices = [
        env_plantings_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        rip_plantings_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        agroforestry_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        carbon_plantings_block_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        carbon_plantings_belt_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        beccs_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_quantity_matrices, axis=2)