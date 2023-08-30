import numpy as np

import luto.settings as settings


def get_rev_env_plantings(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        The cost of environmental plantings for each cell. A 1-D array indexed by cell.
    """
    # Multiply carbon reduction by carbon price for each cell and adjust for resfactor.
    return np.zeros(data.NCELLS)
    # return data.EP_BLOCK_AVG_T_C02_HA * data.REAL_AREA * settings.CARBON_PRICE_PER_TONNE


def get_rev_matrix(data) -> np.ndarray:
    """

    """
    env_plantings_rev_matrix = get_rev_env_plantings(data)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_rev_matrices = [
        env_plantings_rev_matrix.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_rev_matrices, axis=1)