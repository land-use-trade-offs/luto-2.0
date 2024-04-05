import numpy as np
from luto.non_ag_landuses import NON_AG_LAND_USES

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


def get_quantity_matrix(data) -> np.ndarray:
    """
    Get the non-agricultural quantity matrix q_crk.
    Values represent the yield of each commodity c from the cell r when using
        the non-agricultural land use k.
    """

    non_agr_quantity_matrices = {use: np.zeros((data.NCMS, data.NCELLS, 1)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_agr_quantity_matrices['Environmental Plantings'] = get_quantity_env_plantings(data).reshape((data.NCMS, data.NCELLS, 1))

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_agr_quantity_matrices['Riparian Plantings'] = get_quantity_rip_plantings(data).reshape((data.NCMS, data.NCELLS, 1))

    if NON_AG_LAND_USES['Agroforestry']:
        non_agr_quantity_matrices['Agroforestry'] = get_quantity_agroforestry(data).reshape((data.NCMS, data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Belt)']:
        non_agr_quantity_matrices['Carbon Plantings (Belt)'] = get_quantity_carbon_plantings_belt(data).reshape((data.NCMS, data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_agr_quantity_matrices['Carbon Plantings (Block)'] = get_quantity_carbon_plantings_block(data).reshape((data.NCMS, data.NCELLS, 1))

    non_agr_quantity_matrices = list(non_agr_quantity_matrices.values())

    return np.concatenate(non_agr_quantity_matrices, axis=2)