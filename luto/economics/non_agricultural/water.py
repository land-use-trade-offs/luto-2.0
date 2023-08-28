import numpy as np


def get_wreq_matrix_env_planting(data) -> np.ndarray:
    """
    Get water requirements vector of environmental plantings.

    To get the water requirements of environmental plantings, subtract the deep-rooted
    water yields from the shallow-rooted water yields in the data. This represents
    how much water would be used if the area was reforested.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return data.WATER_YIELD_BASE_DIFF * data.REAL_AREA


def get_wreq_matrix(data) -> np.ndarray:
    """
    Get the water requirements matrix for all non-agricultural land uses.

    Returns
    -------
    np.ndarray
        Indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """
    env_plant_wreq_matrix = get_wreq_matrix_env_planting(data)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_ag_wreq_matrices = [
        env_plant_wreq_matrix.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_ag_wreq_matrices, axis=1)
