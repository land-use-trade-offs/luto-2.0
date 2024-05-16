import numpy as np

from luto import tools


def get_sheep_q_cr(data, ag_q_mrp: np.ndarray) -> np.ndarray:
    """
    Gets the matrix containing the commodities produced by sheep (natural land) 
    """
    sheep_j = tools.get_sheep_natural_land_code(data)

    sheep_p = []
    for p in range(data.NPRS):
        if data.LU2PR[p, sheep_j]:
            sheep_p.append(p)

    sheep_q_cr = np.zeros((data.NCMS, data.NCELLS))
    for p in sheep_p:
        for c in range(data.NCMS):
            if data.PR2CM[c, p]:
                sheep_q_cr[c, :] += ag_q_mrp[0, :, p]

    return sheep_q_cr
        

def get_beef_q_cr(data, ag_q_mrp: np.ndarray) -> np.ndarray:
    """
    Gets the matrix containing the commodities produced by beef (natural land) 
    """
    beef_j = tools.get_beef_natural_land_code(data)

    beef_p = []
    for p in range(data.NPRS):
        if data.LU2PR[p, beef_j]:
            beef_p.append(p)

    beef_q_cr = np.zeros((data.NCMS, data.NCELLS))
    for p in beef_p:
        for c in range(data.NCMS):
            if data.PR2CM[c, p]:
                beef_q_cr[c, :] += ag_q_mrp[0, :, p]

    return beef_q_cr


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


def get_quantity_agroforestry_base(data) -> np.ndarray:
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


def get_quantity_sheep_agroforestry(
    data,
    ag_q_mrp: np.ndarray, 
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
    Numpy array indexed by (c, r)
    """
    sheep_quantity_cr = get_sheep_q_cr(data, ag_q_mrp)    
    base_agroforestry_quantity_cr = get_quantity_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_quantity_cr
    for c in range(data.NCMS):
        agroforestry_contr[c, :] *= agroforestry_x_r

    sheep_contr = sheep_quantity_cr
    for c in range(data.NCMS):
        sheep_contr[c, :] *= (1 - agroforestry_x_r)

    return agroforestry_contr + sheep_contr


def get_quantity_beef_agroforestry(
    data, 
    ag_q_mrp: np.ndarray, 
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
    Numpy array indexed by (c, r)
    """
    beef_quantity_cr = get_beef_q_cr(data, ag_q_mrp)    
    base_agroforestry_quantity_cr = get_quantity_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_quantity_cr
    for c in range(data.NCMS):
        agroforestry_contr[c, :] *= agroforestry_x_r

    beef_contr = beef_quantity_cr
    for c in range(data.NCMS):
        beef_contr[c, :] *= (1 - agroforestry_x_r)

    return agroforestry_contr + beef_contr



def get_quantity_sheep_agroforestry(
    data, 
    ag_q_mrp: np.ndarray, 
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
    Numpy array indexed by (c, r)
    """
    sheep_quantity_cr = get_sheep_q_cr(data, ag_q_mrp)    
    base_agroforestry_quantity_cr = get_quantity_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_quantity_cr
    for c in range(data.NCMS):
        agroforestry_contr[c, :] *= agroforestry_x_r

    sheep_contr = sheep_quantity_cr
    for c in range(data.NCMS):
        sheep_contr[c, :] *= (1 - agroforestry_x_r)

    return agroforestry_contr + sheep_contr


def get_quantity_beef_agroforestry(
    data, 
    ag_q_mrp: np.ndarray, 
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
    Numpy array indexed by (c, r)
    """
    beef_quantity_cr = get_beef_q_cr(data, ag_q_mrp)    
    base_agroforestry_quantity_cr = get_quantity_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_quantity_cr
    for c in range(data.NCMS):
        agroforestry_contr[c, :] *= agroforestry_x_r

    beef_contr = beef_quantity_cr
    for c in range(data.NCMS):
        beef_contr[c, :] *= (1 - agroforestry_x_r)

    return agroforestry_contr + beef_contr



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


def get_quantity_carbon_plantings_belt_base(data) -> np.ndarray:
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


def get_quantity_sheep_carbon_plantings_belt(
    data, 
    ag_q_mrp: np.ndarray, 
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
    Numpy array indexed by (c, r)
    """
    sheep_quantity_cr = get_sheep_q_cr(data, ag_q_mrp)    
    base_cp_quantity_cr = get_quantity_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_quantity_cr
    for c in range(data.NCMS):
        cp_contr[c, :] *= cp_belt_x_r

    sheep_contr = sheep_quantity_cr
    for c in range(data.NCMS):
        sheep_contr *= (1 - cp_belt_x_r)

    return cp_contr + sheep_contr


def get_quantity_beef_carbon_plantings_belt(
    data, 
    ag_q_mrp: np.ndarray, 
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
    Numpy array indexed by (c, r)
    """
    beef_quantity_cr = get_beef_q_cr(data, ag_q_mrp)    
    base_cp_quantity_cr = get_quantity_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_quantity_cr
    for c in range(data.NCMS):
        cp_contr[c, :] *= cp_belt_x_r

    beef_contr = beef_quantity_cr
    for c in range(data.NCMS):
        beef_contr *= (1 - cp_belt_x_r)

    return cp_contr + beef_contr


def get_quantity_sheep_carbon_plantings_belt(
    data, 
    ag_q_mrp: np.ndarray, 
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
    Numpy array indexed by (c, r)
    """
    sheep_quantity_cr = get_sheep_q_cr(data, ag_q_mrp)    
    base_cp_quantity_cr = get_quantity_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_quantity_cr
    for c in range(data.NCMS):
        cp_contr[c, :] *= cp_belt_x_r

    sheep_contr = sheep_quantity_cr
    for c in range(data.NCMS):
        sheep_contr *= (1 - cp_belt_x_r)

    return cp_contr + sheep_contr


def get_quantity_beef_carbon_plantings_belt(
    data, 
    ag_q_mrp: np.ndarray, 
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
    Numpy array indexed by (c, r)
    """
    beef_quantity_cr = get_beef_q_cr(data, ag_q_mrp)    
    base_cp_quantity_cr = get_quantity_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_quantity_cr
    for c in range(data.NCMS):
        cp_contr[c, :] *= cp_belt_x_r

    beef_contr = beef_quantity_cr
    for c in range(data.NCMS):
        beef_contr *= (1 - cp_belt_x_r)

    return cp_contr + beef_contr


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


def get_quantity_matrix(data, ag_q_mrp: np.ndarray, lumap: np.ndarray) -> np.ndarray:
    """
    Get the non-agricultural quantity matrix q_crk.
    Values represent the yield of each commodity c from the cell r when using
    the non-agricultural land use k.

    Parameters:
    - data: The input data containing information about the land use and commodities.

    Returns:
    - np.ndarray: The non-agricultural quantity matrix q_crk.
    """
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    env_plantings_quantity_matrix = get_quantity_env_plantings(data)
    rip_plantings_quantity_matrix = get_quantity_rip_plantings(data)
    sheep_agroforestry_quantity_matrix = get_quantity_sheep_agroforestry(data, ag_q_mrp, agroforestry_x_r)
    beef_agroforestry_quantity_matrix = get_quantity_sheep_agroforestry(data, ag_q_mrp, agroforestry_x_r)
    carbon_plantings_block_quantity_matrix = get_quantity_carbon_plantings_block(data)
    sheep_carbon_plantings_belt_quantity_matrix = get_quantity_sheep_carbon_plantings_belt(data, ag_q_mrp, cp_belt_x_r)
    beef_carbon_plantings_belt_quantity_matrix = get_quantity_beef_carbon_plantings_belt(data, ag_q_mrp, cp_belt_x_r)
    beccs_quantity_matrix = get_quantity_beccs(data)

    # reshape each matrix to be indexed (c, r, k) and concatenate on the k indexing
    non_agr_quantity_matrices = [
        env_plantings_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        rip_plantings_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        sheep_agroforestry_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        beef_agroforestry_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        carbon_plantings_block_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        sheep_carbon_plantings_belt_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        beef_carbon_plantings_belt_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
        beccs_quantity_matrix.reshape((data.NCMS, data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_quantity_matrices, axis=2)