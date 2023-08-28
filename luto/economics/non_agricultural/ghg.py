import numpy as np


def get_ghg_reduction_env_plantings(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Greenhouse gas emissions of environmental plantings for each cell.
        Since environmental plantings reduces carbon in the air, each value will be <= 0.
        1-D array Indexed by cell.
    """
    # Tonnes of CO2e per ha, adjusted for resfactor
    return -data.EP_BLOCK_AVG_T_C02_HA * data.REAL_AREA


def get_ghg_matrix(data) -> np.ndarray:
    """
    Get the g_rk matrix containing non-agricultural greenhouse gas emissions.
    """
    env_plantings_ghg_matrix = get_ghg_reduction_env_plantings(data)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_ghg_matrices = [
        env_plantings_ghg_matrix.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_ghg_matrices, axis=1)