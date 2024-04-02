import numpy as np


def get_quantity_env_plantings(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

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
    data: object/module
        Data object or module with fields like in `luto.data`.

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
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Indexed by (c, r): represents the quantity commodity c produced by cell r
        if used for riparian plantings.
        A matrix of zeros because riparian plantings doesn't produce anything.
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
    """
    env_plantings_quantity_matrix = get_quantity_env_plantings(data)
    rip_plantings_quantity_matrix = get_quantity_rip_plantings(data)
    agroforestry_quantity_matrix = get_quantity_agroforestry(data)
    beccs_quantity_matrix = get_quantity_beccs(data)

    # reshape each matrix to be indexed (c, r, k) and concatenate on the k indexing
    non_agr_quantity_matrices = [
        env_plantings_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        rip_plantings_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        agroforestry_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        beccs_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_quantity_matrices, axis=2)