import numpy as np
from luto.settings import REFORESTATION_BIODIVERSITY_BENEFIT, CARBON_PLANTINGS_BIODIV_MULTIPLIER


def get_biodiv_environmental_plantings(data):
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT


def get_biodiv_riparian_plantings(data):
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT


def get_biodiv_agroforestry(data):
    return data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT


def get_biodiv_carbon_plantings_block(data, yr_idx):
    base_biodiv = data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT
    return np.multiply(base_biodiv, (1 + CARBON_PLANTINGS_BIODIV_MULTIPLIER) ** yr_idx)


def get_biodiv_carbon_plantings_belt(data, yr_idx):
    base_biodiv = data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT
    return np.multiply(base_biodiv, (1 + CARBON_PLANTINGS_BIODIV_MULTIPLIER) ** yr_idx)


def get_breq_matrix(data, yr_idx):
    """
    Returns non-agricultural c_rk matrix of costs per cell and land use.
    """
    env_plantings_biodiv = get_biodiv_environmental_plantings(data)
    rip_plantings_biodiv = get_biodiv_riparian_plantings(data)
    agroforestry_biodiv = get_biodiv_agroforestry(data)
    carbon_plantings_block_biodiv = get_biodiv_carbon_plantings_block(data, yr_idx)
    carbon_plantings_belt_biodiv = get_biodiv_carbon_plantings_belt(data, yr_idx)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_c_matrices = [
        env_plantings_biodiv.reshape((data.NCELLS, 1)),
        rip_plantings_biodiv.reshape((data.NCELLS, 1)),
        agroforestry_biodiv.reshape((data.NCELLS, 1)),
        carbon_plantings_block_biodiv.reshape((data.NCELLS, 1)),
        carbon_plantings_belt_biodiv.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_c_matrices, axis=1)