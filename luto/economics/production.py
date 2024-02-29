import numpy as np

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.data import Data
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.non_agricultural.quantity as non_ag_quantity


def get_production(
    data: Data, 
    yr_cal: int, 
    ag_X_mrj: np.ndarray,
    non_ag_X_rk: np.ndarray,
    ag_man_X_mrj: np.ndarray
) -> np.ndarray:
    """Return total production of commodities for a specific year...

    'data' is a Data object, 'yr_cal' is calendar year, and X_mrj 
    is land-use and land management in mrj decision-variable format.

    Can return base year production (e.g., year = 2010) or can return production for 
    a simulated year if one exists (i.e., year = 2030) check sim.info()).

    Includes the impacts of land-use change, productivity increases, and 
    climate change on yield."""

    # Calculate year index (i.e., number of years since 2010)
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the quantity of each commodity produced by agricultural land uses
    # Get the quantity matrices. Quantity array of shape m, r, p
    ag_q_mrp = ag_quantity.get_quantity_matrices(data, yr_idx)

    # Convert map of land-use in mrj format to mrp format
    ag_X_mrp = np.stack([ag_X_mrj[:, :, j] for p in range(data.NPRS)
                         for j in range(data.N_AG_LUS)
                         if data.LU2PR[p, j]
                         ], axis=2)

    # Sum quantities in product (PR/p) representation.
    ag_q_p = np.sum(ag_q_mrp * ag_X_mrp, axis=(0, 1), keepdims=False)

    # Transform quantities to commodity (CM/c) representation.
    ag_q_c = [sum(ag_q_p[p] for p in range(data.NPRS) if data.PR2CM[c, p])
              for c in range(data.NCMS)]

    # Get the quantity of each commodity produced by non-agricultural land uses
    # Quantity matrix in the shape of c, r, k
    q_crk = non_ag_quantity.get_quantity_matrix(data)
    non_ag_q_c = [(q_crk[c, :, :] * non_ag_X_rk).sum()
                  for c in range(data.NCMS)]

    # Get quantities produced by agricultural management options
    j2p = {j: [p for p in range(data.NPRS) if data.LU2PR[p, j]]
           for j in range(data.N_AG_LUS)}
    ag_man_q_mrp = ag_quantity.get_agricultural_management_quantity_matrices(
        data, ag_q_mrp, yr_idx)
    ag_man_q_c = np.zeros(data.NCMS)

    for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():
        am_j_list = [data.DESC2AGLU[lu] for lu in am_lus]
        current_ag_man_X_mrp = np.zeros(ag_q_mrp.shape, dtype=np.float32)

        for j in am_j_list:
            for p in j2p[j]:
                current_ag_man_X_mrp[:, :, p] = ag_man_X_mrj[am][:, :, j]

        ag_man_q_p = np.sum(
            ag_man_q_mrp[am] * current_ag_man_X_mrp, axis=(0, 1), keepdims=False)

        for c in range(data.NCMS):
            ag_man_q_c[c] += sum(ag_man_q_p[p]
                                 for p in range(data.NPRS) if data.PR2CM[c, p])

    # Return total commodity production as numpy array.
    total_q_c = [ag_q_c[c] + non_ag_q_c[c] + ag_man_q_c[c]
                 for c in range(data.NCMS)]
    return np.array(total_q_c)
