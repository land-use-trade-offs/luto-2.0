import numpy as np


def get_breq_matrix(data) -> np.ndarray:
    """
    Get the biodiversity requirements matrix for all non-agricultural land uses.

    Returns
    -------
    np.ndarray
        Indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """
    b_rk = np.zeros((data.NCELLS, data.N_NON_AG_LUS))

    for k in range(len(data.NON_AG_LU_NATURAL)):
        b_rk[:, k] = data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA

    return b_rk