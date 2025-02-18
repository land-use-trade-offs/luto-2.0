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
import luto.settings as settings
from luto.settings import NON_AG_LAND_USES
from luto import tools


def get_cost_env_plantings(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of environmental plantings for each cell. 1-D array Indexed by cell.
    """
    cost_per_ha_per_year = (
        settings.EP_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR * data.MAINT_COST_MULTS[yr_cal]
        - settings.EP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR
    )
    return cost_per_ha_per_year * data.REAL_AREA


def get_cost_rip_plantings(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of riparian plantings for each cell. 1-D array Indexed by cell.
    """
    cost_per_ha_per_year = (
        settings.RP_ANNUAL_MAINTENNANCE_COST_PER_HA_PER_YEAR * data.MAINT_COST_MULTS[yr_cal]
        - settings.RP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR
    )
    return cost_per_ha_per_year * data.REAL_AREA 


def get_cost_agroforestry_base(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of agroforestry for each cell. 1-D array Indexed by cell.
    """
    cost_per_ha_per_year = (
        settings.AF_ANNUAL_MAINTENNANCE_COST_PER_HA_PER_YEAR * data.MAINT_COST_MULTS[yr_cal]
        - settings.AF_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR
    )
    return cost_per_ha_per_year * data.REAL_AREA


def get_cost_sheep_agroforestry(
    data: Data,
    yr_cal: int,
    ag_c_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_c_mrj: agricultural cost matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_cost = ag_c_mrj[0, :, sheep_j]
    base_agroforestry_cost = get_cost_agroforestry_base(data, yr_cal)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_cost * agroforestry_x_r
    sheep_contr = sheep_cost * (1 - agroforestry_x_r)
    return agroforestry_contr + sheep_contr


def get_cost_beef_agroforestry(
    data: Data,
    yr_cal: int,
    ag_c_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_c_mrj: agricultural cost matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_cost = ag_c_mrj[0, :, beef_j]
    base_agroforestry_cost = get_cost_agroforestry_base(data, yr_cal)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_cost * agroforestry_x_r
    beef_contr = beef_cost * (1 - agroforestry_x_r)
    return agroforestry_contr + beef_contr


def get_cost_carbon_plantings_block(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        Cost of carbon plantings (block arrangement) for each cell. 1-D array Indexed by cell.
    """
    cost_per_ha_per_year = (
        settings.CP_BLOCK_ANNUAL_MAINTENNANCE_COST_PER_HA_PER_YEAR * data.MAINT_COST_MULTS[yr_cal]
        - settings.CP_BLOCK_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR
    )
    return cost_per_ha_per_year * data.REAL_AREA


def get_cost_carbon_plantings_belt_base(data: Data, yr_cal) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of carbon plantings (belt arrangement) for each cell. 1-D array Indexed by cell.
    """
    cost_per_ha_per_year = (
        settings.CP_BELT_ANNUAL_MAINTENNANCE_COST_PER_HA_PER_YEAR * data.MAINT_COST_MULTS[yr_cal]
        - settings.CP_BELT_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR
    )
    return cost_per_ha_per_year * data.REAL_AREA


def get_cost_sheep_carbon_plantings_belt(
    data: Data,
    yr_cal: int,
    ag_c_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_c_mrj: agricultural cost matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_cost = ag_c_mrj[0, :, sheep_j]
    base_cp_cost = get_cost_carbon_plantings_belt_base(data, yr_cal)

    # Calculate contributions and return the sum
    cp_contr = base_cp_cost * cp_belt_x_r
    sheep_contr = sheep_cost * (1 - cp_belt_x_r)
    return cp_contr + sheep_contr


def get_cost_beef_carbon_plantings_belt(
    data: Data,
    yr_cal: int,
    ag_c_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_c_mrj: agricultural cost matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_cost = ag_c_mrj[0, :, beef_j]
    base_cp_cost = get_cost_carbon_plantings_belt_base(data, yr_cal)

    # Calculate contributions and return the sum
    cp_contr = base_cp_cost * cp_belt_x_r
    beef_contr = beef_cost * (1 - cp_belt_x_r)
    return cp_contr + beef_contr


def get_cost_beccs(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        Cost of BECCS for each cell. 1-D array Indexed by cell.
    """
    return np.nan_to_num(data.BECCS_COSTS_AUD_HA_YR) * data.BECCS_COST_MULTS[yr_cal] * data.REAL_AREA


def get_cost_matrix(data: Data, ag_c_mrj: np.ndarray, lumap, yr_cal):
    """
    Returns non-agricultural c_rk matrix of costs per cell and land use.

    Parameters:
    - data: The input data containing information about the cells and land use.

    Returns:
    - cost_matrix: A 2D numpy array of costs per cell and land use.
    """
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    non_agr_c_matrices = {use: np.zeros((data.NCELLS, 1)) for use in NON_AG_LAND_USES}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    if NON_AG_LAND_USES['Environmental Plantings']:
        non_agr_c_matrices['Environmental Plantings'] = get_cost_env_plantings(data, yr_cal).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Riparian Plantings']:
        non_agr_c_matrices['Riparian Plantings'] = get_cost_rip_plantings(data, yr_cal).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Sheep Agroforestry']:
        non_agr_c_matrices['Sheep Agroforestry'] = get_cost_sheep_agroforestry(data, yr_cal, ag_c_mrj, agroforestry_x_r).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Beef Agroforestry']:
        non_agr_c_matrices['Beef Agroforestry'] = get_cost_beef_agroforestry(data, yr_cal, ag_c_mrj, agroforestry_x_r).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Carbon Plantings (Block)']:
        non_agr_c_matrices['Carbon Plantings (Block)'] = get_cost_carbon_plantings_block(data, yr_cal).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Sheep Carbon Plantings (Belt)']:
        non_agr_c_matrices['Sheep Carbon Plantings (Belt)'] = get_cost_sheep_carbon_plantings_belt(data, yr_cal, ag_c_mrj, cp_belt_x_r).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['Beef Carbon Plantings (Belt)']:
        non_agr_c_matrices['Beef Carbon Plantings (Belt)'] = get_cost_beef_carbon_plantings_belt(data, yr_cal, ag_c_mrj, cp_belt_x_r).reshape((data.NCELLS, 1))

    if NON_AG_LAND_USES['BECCS']:
        non_agr_c_matrices['BECCS'] = get_cost_beccs(data, yr_cal).reshape((data.NCELLS, 1))

    non_agr_c_matrices = list(non_agr_c_matrices.values())
    return np.concatenate(non_agr_c_matrices, axis=1)