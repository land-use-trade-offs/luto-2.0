import numpy as np

from luto.data import Data
from luto.settings import REFORESTATION_BIODIVERSITY_BENEFIT


def get_breq_matrix(data: Data) -> np.ndarray:
    """
    Get the biodiversity requirements matrix for all non-agricultural land uses.

    Returns
    -------
    np.ndarray
        Indexed by (r, k) where r is cell and k is non-agricultural land usage.
    """
    b_rk = np.zeros((data.NCELLS, data.N_NON_AG_LUS))

    # The current assumption is that all non-agricultural land uses contribute fully to biodiversity
    for k in range(len(data.NON_AG_LU_NATURAL)):
        b_rk[:, k] = data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * REFORESTATION_BIODIVERSITY_BENEFIT

    return b_rk