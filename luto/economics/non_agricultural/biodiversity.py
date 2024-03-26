import numpy as np
from luto.settings import REFORESTATION_BIODIVERSITY_BENEFIT


def get_biodiv_environmental_plantings(data) -> np.ndarray:
    return data.BIODIV_SCORE_WEIGHTED_LDS_BURNING * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT


def get_biodiv_riparian_plantings(data) -> np.ndarray:
    return data.BIODIV_SCORE_WEIGHTED_LDS_BURNING * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT


def get_biodiv_agroforestry(data) -> np.ndarray:
    return data.BIODIV_SCORE_WEIGHTED_LDS_BURNING * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT


def get_biodiv_savanna_burning(data) -> np.ndarray:
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT


def get_breq_matrix(data) -> np.ndarray:
    """
    Get the biodiversity requirements matrix for all non-agricultural land uses.

    Returns
    -------
    np.ndarray
        Indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """
    env_plantings_biodiv = get_biodiv_environmental_plantings(data)
    rip_plantings_biodiv = get_biodiv_riparian_plantings(data)
    agroforestry_biodiv = get_biodiv_agroforestry(data)
    sav_burning_biodiv = get_biodiv_savanna_burning(data)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_c_matrices = [
        env_plantings_biodiv.reshape((data.NCELLS, 1)),
        rip_plantings_biodiv.reshape((data.NCELLS, 1)),
        agroforestry_biodiv.reshape((data.NCELLS, 1)),
        sav_burning_biodiv.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_c_matrices, axis=1)
