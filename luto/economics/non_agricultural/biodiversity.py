import numpy as np

from luto.data import Data
from luto.settings import (
    ENV_PLANTING_BIODIVERSITY_BENEFIT, 
    CARBON_PLANTING_BLOCK_BIODIV_BENEFIT, 
    CARBON_PLANTING_BELT_BIODIV_BENEFIT, 
    RIPARIAN_PLANTING_BIODIV_BENEFIT,
    AGROFORESTRY_BIODIV_BENEFIT,
    BECCS_BIODIVERSITY_BENEFIT,
    )


def get_biodiv_environmental_plantings(data: Data) -> np.ndarray:
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * ENV_PLANTING_BIODIVERSITY_BENEFIT


def get_biodiv_riparian_plantings(data) -> np.ndarray:
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * RIPARIAN_PLANTING_BIODIV_BENEFIT


def get_biodiv_agroforestry(data) -> np.ndarray:
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * AGROFORESTRY_BIODIV_BENEFIT


def get_biodiv_carbon_plantings_block(data) -> np.ndarray:
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * CARBON_PLANTING_BLOCK_BIODIV_BENEFIT


def get_biodiv_carbon_plantings_belt(data) -> np.ndarray:
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * CARBON_PLANTING_BELT_BIODIV_BENEFIT


def get_biodiv_beccs(data):
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * BECCS_BIODIVERSITY_BENEFIT


def get_breq_matrix(data):
    """
    Returns non-agricultural c_rk matrix of costs per cell and land use.

    Parameters:
    - data: The input data object containing necessary information.

    Returns:
    - numpy.ndarray: The non-agricultural c_rk matrix of costs per cell and land use.
    """
    env_plantings_biodiv = get_biodiv_environmental_plantings(data)
    rip_plantings_biodiv = get_biodiv_riparian_plantings(data)
    agroforestry_biodiv = get_biodiv_agroforestry(data)
    carbon_plantings_block_biodiv = get_biodiv_carbon_plantings_block(data)
    carbon_plantings_belt_biodiv = get_biodiv_carbon_plantings_belt(data)
    beccs_biodiv = get_biodiv_beccs(data)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_c_matrices = [
        env_plantings_biodiv.reshape((data.NCELLS, 1)),
        rip_plantings_biodiv.reshape((data.NCELLS, 1)),
        agroforestry_biodiv.reshape((data.NCELLS, 1)),
        carbon_plantings_block_biodiv.reshape((data.NCELLS, 1)),
        carbon_plantings_belt_biodiv.reshape((data.NCELLS, 1)),
        beccs_biodiv.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_c_matrices, axis=1)