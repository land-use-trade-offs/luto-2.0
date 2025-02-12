import numpy as np

from luto.data import Data
from luto import tools
from luto.settings import (
    ENV_PLANTING_BIODIVERSITY_BENEFIT, 
    CARBON_PLANTING_BLOCK_BIODIV_BENEFIT, 
    CARBON_PLANTING_BELT_BIODIV_BENEFIT, 
    RIPARIAN_PLANTING_BIODIV_BENEFIT,
    AGROFORESTRY_BIODIV_BENEFIT,
    BECCS_BIODIVERSITY_BENEFIT,
    )


def get_biodiv_environmental_plantings(data: Data) -> np.ndarray:
    return data.BIODIV_SCORE_RAW_WEIGHTED * data.REAL_AREA * ENV_PLANTING_BIODIVERSITY_BENEFIT


def get_biodiv_riparian_plantings(data: Data) -> np.ndarray:
    return data.BIODIV_SCORE_RAW_WEIGHTED * data.REAL_AREA * RIPARIAN_PLANTING_BIODIV_BENEFIT


def get_biodiv_agroforestry_base(data: Data) -> np.ndarray:
    return data.BIODIV_SCORE_RAW_WEIGHTED * data.REAL_AREA * AGROFORESTRY_BIODIV_BENEFIT


def get_biodiv_sheep_agroforestry(
    data: Data, 
    ag_b_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_b_mrj: agricultural biodiversity matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_biodiv = ag_b_mrj[0, :, sheep_j]
    base_agroforestry_biodiv = get_biodiv_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_biodiv * agroforestry_x_r
    sheep_contr = sheep_biodiv * (1 - agroforestry_x_r)
    return agroforestry_contr + sheep_contr


def get_biodiv_beef_agroforestry(
    data: Data, 
    ag_b_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_b_mrj: agricultural biodiversity matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_biodiv = ag_b_mrj[0, :, beef_j]
    base_agroforestry_biodiv = get_biodiv_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_biodiv * agroforestry_x_r
    beef_contr = beef_biodiv * (1 - agroforestry_x_r)
    return agroforestry_contr + beef_contr


def get_biodiv_carbon_plantings_block(data: Data) -> np.ndarray:
    return data.BIODIV_SCORE_RAW_WEIGHTED * data.REAL_AREA * CARBON_PLANTING_BLOCK_BIODIV_BENEFIT


def get_biodiv_carbon_plantings_belt_base(data: Data) -> np.ndarray:
    return data.BIODIV_SCORE_RAW_WEIGHTED * data.REAL_AREA * CARBON_PLANTING_BELT_BIODIV_BENEFIT


def get_biodiv_sheep_carbon_plantings_belt(
    data: Data, 
    ag_b_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_b_mrj: agricultural biodiversity matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_biodiv = ag_b_mrj[0, :, sheep_j]
    base_cp_biodiv = get_biodiv_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_biodiv * cp_belt_x_r
    sheep_contr = sheep_biodiv * (1 - cp_belt_x_r)
    return cp_contr + sheep_contr


def get_biodiv_beef_carbon_plantings_belt(
    data: Data, 
    ag_b_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_b_mrj: agricultural biodiversity matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_biodiv = ag_b_mrj[0, :, beef_j]
    base_cp_biodiv = get_biodiv_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_biodiv * cp_belt_x_r
    beef_contr = beef_biodiv * (1 - cp_belt_x_r)
    return cp_contr + beef_contr


def get_biodiv_beccs(data: Data):
    return data.BIODIV_SCORE_RAW_WEIGHTED * data.REAL_AREA * BECCS_BIODIVERSITY_BENEFIT


def get_breq_matrix(data: Data, ag_b_mrj: np.ndarray, lumap: np.ndarray):
    """
    Returns non-agricultural c_rk matrix of costs per cell and land use.

    Parameters:
    - data: The input data object containing necessary information.

    Returns:
    - numpy.ndarray: The non-agricultural c_rk matrix of costs per cell and land use.
    """
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    env_plantings_biodiv = get_biodiv_environmental_plantings(data)
    rip_plantings_biodiv = get_biodiv_riparian_plantings(data)
    sheep_agroforestry_biodiv = get_biodiv_sheep_agroforestry(data, ag_b_mrj, agroforestry_x_r)
    beef_agroforestry_biodiv = get_biodiv_beef_agroforestry(data, ag_b_mrj, agroforestry_x_r)
    carbon_plantings_block_biodiv = get_biodiv_carbon_plantings_block(data)
    sheep_carbon_plantings_belt_biodiv = get_biodiv_sheep_carbon_plantings_belt(data, ag_b_mrj, cp_belt_x_r)
    beef_carbon_plantings_belt_biodiv = get_biodiv_beef_carbon_plantings_belt(data, ag_b_mrj, cp_belt_x_r)
    beccs_biodiv = get_biodiv_beccs(data)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_c_matrices = [
        env_plantings_biodiv.reshape((data.NCELLS, 1)),
        rip_plantings_biodiv.reshape((data.NCELLS, 1)),
        sheep_agroforestry_biodiv.reshape((data.NCELLS, 1)),
        beef_agroforestry_biodiv.reshape((data.NCELLS, 1)),
        carbon_plantings_block_biodiv.reshape((data.NCELLS, 1)),
        sheep_carbon_plantings_belt_biodiv.reshape((data.NCELLS, 1)),
        beef_carbon_plantings_belt_biodiv.reshape((data.NCELLS, 1)),
        beccs_biodiv.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_c_matrices, axis=1)


def get_major_vegetation_matrices(data: Data) -> np.ndarray:
    """
    Get the matrix containing the contribution of each cell/non-ag. land use combination 
    to each major vegetation group.

    Returns:
    - Array indexed by (r, k, v) containing the contributions.
    """
    mvg_rk = {
        v: np.zeros((data.NCELLS, data.N_NON_AG_LUS)).astype(np.float32) 
        for v in range(data.N_NVIS_CLASSES)
    }
    for k in range(len(data.NON_AG_LU_ENV_PLANTINGS)):
        for v in range(data.N_NVIS_CLASSES):
            mvg_rk[v][:, k] = data.NVIS_PRE_GR[v]

    return mvg_rk