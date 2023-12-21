import numpy as np


def get_breq_matrix(data) -> np.ndarray:
    """
    Get the water requirements matrix for all non-agricultural land uses.

    Returns
    -------
    np.ndarray
        Indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """

    return np.zeros((data.NCELLS, data.N_NON_AG_LUS))