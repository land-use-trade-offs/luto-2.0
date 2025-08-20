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
import luto.data as Data

from typing import Optional
from luto import tools


def get_w_net_yield_env_planting(
    data: Data, 
    yr_idx: int, 
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Get water yields vector of environmental plantings.

    Environmental plantings refer to restoring land to its pre-European vegetation state. 
    So we need to use the `get_water_nl_yield_for_yr_idx` to consider both deep-rooted and 
    shallow-rooted water yields under the pre-European vegetation state.

    Returns
    -------
    1-D array, indexed by cell.
    
    Notes
    -----
    If `water_dr_yield` and `water_sr_yield` are provided, the function will calculate
    the water yields regardless of the `yr_idx`.
    """
    w_yield_dr = data.WATER_YIELD_DR_FILE[yr_idx] if water_dr_yield is None else water_dr_yield
    w_yield_sr = data.WATER_YIELD_SR_FILE[yr_idx] if water_sr_yield is None else water_sr_yield
    w_yield_nl = data.get_water_nl_yield_for_yr_idx(yr_idx, w_yield_dr, w_yield_sr)
    wyield = w_yield_nl * data.REAL_AREA
    return wyield


def get_w_net_yield_carbon_plantings_block(
    data: Data, 
    yr_idx: int, 
    water_dr_yield: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Get water yields vector of carbon plantings (block arrangement).

    Carbon plantings are pure deep-rooted systems.

    Returns
    -------
    1-D array, indexed by cell.
    """
    w_yield_dr = data.WATER_YIELD_DR_FILE[yr_idx] if water_dr_yield is None else water_dr_yield
    wyield = w_yield_dr * data.REAL_AREA
    return wyield


def get_w_net_yield_rip_planting(
    data: Data, 
    yr_idx: int, 
    water_dr_yield: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Get water yields vector of riparian plantings.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_w_net_yield_env_planting(data, yr_idx, water_dr_yield)


def get_w_net_yield_agroforestry_base(
    data: Data, 
    yr_idx: int, 
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Get water yields vector of agroforestry.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_w_net_yield_env_planting(data, yr_idx, water_dr_yield, water_sr_yield)


def get_w_net_yield_sheep_agroforestry(
    data: Data, 
    ag_w_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray,
    yr_idx: int,
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Parameters
    ------
    data object.
    ag_w_mrj: agricultural water requirements matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_w_net_yield = ag_w_mrj[0, :, sheep_j]
    base_agroforestry_w_net_yield = get_w_net_yield_agroforestry_base(data, yr_idx, water_dr_yield, water_sr_yield)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_w_net_yield * agroforestry_x_r
    sheep_contr = sheep_w_net_yield * (1 - agroforestry_x_r)
    return agroforestry_contr + sheep_contr


def get_w_net_yield_beef_agroforestry(
    data: Data, 
    ag_w_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray,
    yr_idx: int,
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Parameters
    ------
    data object.
    ag_w_mrj: agricultural water yield matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_w_net_yield = ag_w_mrj[0, :, beef_j]
    base_agroforestry_w_net_yield = get_w_net_yield_agroforestry_base(data, yr_idx, water_dr_yield, water_sr_yield)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_w_net_yield * agroforestry_x_r
    beef_contr = beef_w_net_yield * (1 - agroforestry_x_r)
    return agroforestry_contr + beef_contr


def get_w_net_yield_carbon_plantings_belt_base(data, yr_idx: int, water_dr_yield: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get water requirments vector of carbon plantings (belt arrangement).

    Note: this is the same as for carbon plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_w_net_yield_carbon_plantings_block(data, yr_idx, water_dr_yield)


def get_w_net_yield_sheep_carbon_plantings_belt(
    data: Data, 
    ag_w_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray,
    yr_idx: int,
    water_dr_yield: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Parameters
    ------
    data object.
    ag_w_mrj: agricultural water requirements matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_w_net_yield = ag_w_mrj[0, :, sheep_j]
    base_cp_w_net_yield = get_w_net_yield_carbon_plantings_belt_base(data, yr_idx, water_dr_yield)

    # Calculate contributions and return the sum
    cp_contr = base_cp_w_net_yield * cp_belt_x_r
    sheep_contr = sheep_w_net_yield * (1 - cp_belt_x_r)
    return cp_contr + sheep_contr


def get_w_net_yield_beef_carbon_plantings_belt(
    data: Data, 
    ag_w_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray,
    yr_idx: int,
    water_dr_yield: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Parameters
    ------
    data object.
    ag_w_mrj: agricultural water requirements matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_w_net_yield = ag_w_mrj[0, :, beef_j]
    base_cp_w_net_yield = get_w_net_yield_carbon_plantings_belt_base(data, yr_idx, water_dr_yield)

    # Calculate contributions and return the sum
    cp_contr = base_cp_w_net_yield * cp_belt_x_r
    beef_contr = beef_w_net_yield * (1 - cp_belt_x_r)
    return cp_contr + beef_contr


def get_w_net_yield_beccs(data, yr_idx: int, water_dr_yield: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get water requirments vector of BECCS.

    Note: this is the same as for carbon plantings.

    Returns
    -------
    1-D array, indexed by cell.
    """
    return get_w_net_yield_carbon_plantings_block(data, yr_idx, water_dr_yield)


def get_w_net_yield_destocked(data, ag_w_mrj):
    unallocated_j = tools.get_unallocated_natural_land_code(data)
    return ag_w_mrj[0, :, unallocated_j]


def get_w_net_yield_matrix(
    data: Data,
    ag_w_mrj: np.ndarray,
    lumap: np.ndarray,
    yr_idx: int,
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Get the water yields matrix for all non-agricultural land uses.

    Parameters
    ----------
    data : object
        The data object containing necessary information for calculating the water yields.

    Returns
    -------
    np.ndarray
        The water yields matrix for all non-agricultural land uses.
        Indexed by (r, k) where r is the cell index and k is the non-agricultural land usage index.
    """
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    non_agr_yield_matrices = [
        get_w_net_yield_env_planting(data, yr_idx, water_dr_yield, water_sr_yield)                     ,
        get_w_net_yield_rip_planting(data, yr_idx, water_dr_yield)                                     ,
        get_w_net_yield_sheep_agroforestry(data, ag_w_mrj, agroforestry_x_r, yr_idx, water_dr_yield, water_sr_yield) ,
        get_w_net_yield_beef_agroforestry(data, ag_w_mrj, agroforestry_x_r, yr_idx, water_dr_yield, water_sr_yield)  ,
        get_w_net_yield_carbon_plantings_block(data, yr_idx, water_dr_yield)                           ,
        get_w_net_yield_sheep_carbon_plantings_belt(data, ag_w_mrj, cp_belt_x_r, yr_idx, water_dr_yield)             ,
        get_w_net_yield_beef_carbon_plantings_belt(data, ag_w_mrj, cp_belt_x_r, yr_idx, water_dr_yield)              ,
        get_w_net_yield_beccs(data, yr_idx, water_dr_yield)                                                   ,
        get_w_net_yield_destocked(data, ag_w_mrj)                                                             
    ]

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    return np.concatenate([
        arr.reshape((data.NCELLS, 1)) for arr in non_agr_yield_matrices
    ], axis=1)
