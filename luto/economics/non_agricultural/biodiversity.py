# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from luto.data import Data
from luto import tools
from luto.settings import (
    BIO_CONTRIBUTION_ENV_PLANTING, 
    BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK, 
    BIO_CONTRIBUTION_CARBON_PLANTING_BELT, 
    BIO_CONTRIBUTION_RIPARIAN_PLANTING,
    BIO_CONTRIBUTION_AGROFORESTRY,
    BIO_CONTRIBUTION_BECCS,
    AF_PROPORTION,
    CP_BELT_PROPORTION,
)


def get_biodiv_environmental_plantings(data: Data) -> np.ndarray:
    return data.BIO_CONNECTIVITY_RAW * data.REAL_AREA * BIO_CONTRIBUTION_ENV_PLANTING


def get_biodiv_riparian_plantings(data: Data) -> np.ndarray:
    return data.BIO_CONNECTIVITY_RAW * data.REAL_AREA * BIO_CONTRIBUTION_RIPARIAN_PLANTING


def get_biodiv_agroforestry_base(data: Data) -> np.ndarray:
    return data.BIO_CONNECTIVITY_RAW * data.REAL_AREA * BIO_CONTRIBUTION_AGROFORESTRY


def get_biodiv_sheep_agroforestry(
    data: Data, 
    ag_b_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_b_mrj: agricultural biodiversity matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_biodiv = ag_b_mrj[0, :, sheep_j]
    base_agroforestry_biodiv = get_biodiv_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_biodiv * agroforestry_x_r
    sheep_contr = sheep_biodiv * (1 - agroforestry_x_r)
    return agroforestry_contr + sheep_contr


def get_biodiv_beef_agroforestry(
    data: Data, 
    ag_b_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_b_mrj: agricultural biodiversity matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_biodiv = ag_b_mrj[0, :, beef_j]
    base_agroforestry_biodiv = get_biodiv_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_biodiv * agroforestry_x_r
    beef_contr = beef_biodiv * (1 - agroforestry_x_r)
    return agroforestry_contr + beef_contr


def get_biodiv_carbon_plantings_block(data: Data) -> np.ndarray:
    return data.BIO_CONNECTIVITY_RAW * data.REAL_AREA * BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK


def get_biodiv_carbon_plantings_belt_base(data: Data) -> np.ndarray:
    return data.BIO_CONNECTIVITY_RAW * data.REAL_AREA * BIO_CONTRIBUTION_CARBON_PLANTING_BELT


def get_biodiv_sheep_carbon_plantings_belt(
    data: Data, 
    ag_b_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_b_mrj: agricultural biodiversity matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_biodiv = ag_b_mrj[0, :, sheep_j]
    base_cp_biodiv = get_biodiv_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_biodiv * cp_belt_x_r
    sheep_contr = sheep_biodiv * (1 - cp_belt_x_r)
    return cp_contr + sheep_contr


def get_biodiv_beef_carbon_plantings_belt(
    data: Data, 
    ag_b_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_b_mrj: agricultural biodiversity matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_biodiv = ag_b_mrj[0, :, beef_j]
    base_cp_biodiv = get_biodiv_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_biodiv * cp_belt_x_r
    beef_contr = beef_biodiv * (1 - cp_belt_x_r)
    return cp_contr + beef_contr


def get_biodiv_beccs(data: Data):
    """
    Parameters
    ------
    data: Data object.

    Returns
    ------
    Numpy array indexed by r
    """
    return data.BIO_CONNECTIVITY_RAW * data.REAL_AREA * BIO_CONTRIBUTION_BECCS


def get_biodiv_destocked_land(data: Data, lumap: np.ndarray):
    """
    Parameters
    ------
    data: Data object.
    ag_b_mrj: agricultural biodiversity matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    destock_b_contr = np.zeros(data.NCELLS)
    
    from_lu = data.DESC2AGLU['Unallocated - natural land']
    for to_lu in data.LU_LVSTK_NATURAL:
        destock_b_contr[lumap == from_lu] = (
            data.BIO_CONNECTIVITY_RAW[lumap == from_lu] 
            * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[to_lu])
            * data.REAL_AREA[lumap == from_lu]
        )
    return destock_b_contr


def get_breq_matrix(data: Data, ag_b_mrj: np.ndarray, lumap: np.ndarray):
    """
    Returns non-agricultural c_rk matrix of costs per cell and land use.

    Parameters
    - data: The input data object containing necessary information.

    Returns
    - numpy.ndarray: The non-agricultural c_rk matrix of costs per cell and land use.
    """
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    env_plantings_biodiv = get_biodiv_environmental_plantings(data)
    rip_plantings_biodiv = get_biodiv_riparian_plantings(data)
    sheep_agroforestry_biodiv = get_biodiv_sheep_agroforestry(data, ag_b_mrj, agroforestry_x_r)
    beef_agroforestry_biodiv = get_biodiv_beef_agroforestry(data, ag_b_mrj, agroforestry_x_r)
    carbon_plantings_block_biodiv = get_biodiv_carbon_plantings_block(data)
    sheep_carbon_plantings_belt_biodiv = get_biodiv_sheep_carbon_plantings_belt(data, ag_b_mrj, cp_belt_x_r)
    beef_carbon_plantings_belt_biodiv = get_biodiv_beef_carbon_plantings_belt(data, ag_b_mrj, cp_belt_x_r)
    beccs_biodiv = get_biodiv_beccs(data)
    destocked_biodiv = get_biodiv_destocked_land(data, lumap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_b_matrices = [
        env_plantings_biodiv.reshape((data.NCELLS, 1)),
        rip_plantings_biodiv.reshape((data.NCELLS, 1)),
        sheep_agroforestry_biodiv.reshape((data.NCELLS, 1)),
        beef_agroforestry_biodiv.reshape((data.NCELLS, 1)),
        carbon_plantings_block_biodiv.reshape((data.NCELLS, 1)),
        sheep_carbon_plantings_belt_biodiv.reshape((data.NCELLS, 1)),
        beef_carbon_plantings_belt_biodiv.reshape((data.NCELLS, 1)),
        beccs_biodiv.reshape((data.NCELLS, 1)),
        destocked_biodiv.reshape((data.NCELLS, 1)),
    ]

    return np.concatenate(non_agr_b_matrices, axis=1)


def get_non_ag_lu_biodiv_contribution(data: Data) -> dict[int, float]:
    return {
        # Environmental plantings
        0: BIO_CONTRIBUTION_ENV_PLANTING,
        # Riparian plantings
        1: BIO_CONTRIBUTION_RIPARIAN_PLANTING,
        # Sheep agroforestry
        2: (
            AF_PROPORTION * BIO_CONTRIBUTION_AGROFORESTRY
            + (1 - AF_PROPORTION) * (data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[tools.get_sheep_code(data)])
        ),
        # Beef agroforestry
        3: (
            AF_PROPORTION * BIO_CONTRIBUTION_AGROFORESTRY
            + (1 - AF_PROPORTION) * (data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[tools.get_beef_code(data)])
        ),
        # Carbon plantings (block)
        4: BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK,
        # Sheep carbon plantings (belt)
        5: (
            CP_BELT_PROPORTION * BIO_CONTRIBUTION_AGROFORESTRY
            + (1 - CP_BELT_PROPORTION) * (data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[tools.get_sheep_code(data)])
        ),
        # Beef carbon plantings (belt)
        6: (
            CP_BELT_PROPORTION * BIO_CONTRIBUTION_AGROFORESTRY
            + (1 - CP_BELT_PROPORTION) * (data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[tools.get_beef_code(data)])
        ),
        # BECCS
        7: BIO_CONTRIBUTION_BECCS,
        # Destocked land
        8: data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[tools.get_unallocated_natural_land_code(data)],
    }