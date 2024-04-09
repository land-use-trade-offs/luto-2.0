# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
#
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>.

"""
Data about transitions costs.
"""

import numpy as np
from typing import Dict

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.economics.agricultural.water import get_wreq_matrices
import luto.economics.agricultural.ghg as ag_ghg
from luto import settings
import luto.tools as tools


def get_exclude_matrices(data, base_year: int, lumaps: Dict[int, np.ndarray]):
    """Return x_mrj exclude matrices.

    An exclude matrix indicates whether switching land-use for a certain cell r 
    with land-use i to all other land-uses j under all land management types 
    (i.e., dryland, irrigated) m is possible. 

    Parameters
    ----------

    data: object/module
        Data object or module with fields like in `luto.data`.
    base_year: int
        Current base year of the solve.
    lumaps: dict[str, numpy.ndarray]
        All previously generated land-use maps (shape = ncells, dtype=int).

        
    Returns
    -------

    numpy.ndarray
        x_mrj exclude matrix. The m-slices correspond to the
        different land-management versions of the land-use `j` to switch _to_.
        With m==0 conventional dryland, m==1 conventional irrigated.
    """    
    # Boolean exclusion matrix based on SA2/NLUM agricultural land-use data (in mrj structure).
    # Effectively, this ensures that in any SA2 region the only combinations of land-use and land management
    # that can occur in the future are those that occur in 2010 (i.e., YR_CAL_BASE)
    x_mrj = data.EXCLUDE

    # Raw transition-cost matrix is in $/ha and lexicographically ordered by land-use (shape = 28 x 28).
    t_ij = data.AG_TMATRIX

    lumap = lumaps[base_year]
    lumap_2010 = lumaps[2010]

    # Get all agricultural and non-agricultural cells
    ag_cells, non_ag_cells = tools.get_ag_and_non_ag_cells(lumap)
    
    # Transition costs from current land-use to all other land-uses j using current land-use map (in $/ha).
    t_rj = np.zeros((data.NCELLS, len(data.AGRICULTURAL_LANDUSES)))
    t_rj[ag_cells, :] = t_ij[lumap[ag_cells]]
    
    # For non-agricultural cells, use the original 2010 solve's LUs to determine what LUs are possible for a cell
    t_rj[non_ag_cells, :] = t_ij[lumap_2010[non_ag_cells]]

    # To be excluded based on disallowed switches as specified in transition cost matrix i.e., where t_rj is NaN.
    t_rj = np.where(np.isnan(t_rj), 0, 1)

    # Overall exclusion as elementwise, logical `and` of the 0/1 exclude matrices.
    x_mrj = (x_mrj * t_rj).astype(np.int8)
    
    return x_mrj


def get_transition_matrices(data, yr_idx, base_year, lumaps, lmmaps, separate=False):
    """
    Calculate the transition matrices for land-use and land management transitions.

    Args:
        data (object): The data object containing the necessary input data.
        yr_idx (int): The index of the current year.
        base_year (int): The base year for the transition calculations.
        lumaps (dict): A dictionary of land-use maps for each year.
        lmmaps (dict): A dictionary of land management maps for each year.
        separate (bool, optional): Whether to return separate cost matrices for each cost component. 
                                   Defaults to False.

    Returns:
        numpy.ndarray or dict: The transition matrices for land-use and land management transitions.
                               If `separate` is False, returns a numpy array representing the total costs.
                               If `separate` is True, returns a dictionary with separate cost matrices for
                               establishment costs, Water license cost, and carbon releasing costs.
    """
    
    lumap = lumaps[base_year]
    lmmap = lmmaps[base_year]

    # Return l_mrj (Boolean) for current land-use and land management
    l_mrj = tools.lumap2ag_l_mrj(lumap, lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    # Get the exclusion matrix
    x_mrj = get_exclude_matrices(data, base_year, lumaps)

    ag_cells, _ = tools.get_ag_and_non_ag_cells(lumap)

    n_ag_lms, ncells, n_ag_lus = data.AG_L_MRJ.shape

    # -------------------------------------------------------------- #
    # Establishment costs (upfront, amortised to annual, per cell).  #
    # -------------------------------------------------------------- #

    # Raw transition-cost matrix is in $/ha and lexigraphically ordered (shape: land-use x land-use).
    t_ij = data.AG_TMATRIX

    # Non-irrigation related transition costs for cell r to change to land-use j calculated based on lumap (in $/ha).
    # Only consider for cells currently being used for agriculture.
    e_rj = np.zeros((ncells, n_ag_lus))
    e_rj[ag_cells, :] = t_ij[lumap[ag_cells]]

    # Amortise upfront costs to annualised costs and converted to $ per cell via REAL_AREA
    e_rj = tools.amortise(e_rj) * data.REAL_AREA[:, np.newaxis]

    # Separate the establishment costs into dryland and irrigated land management types
    e_rj_dry = np.einsum('rj,r->rj', e_rj, lmmap == 0)
    e_rj_irr = np.einsum('rj,r->rj', e_rj, lmmap == 1)
    e_mrj = np.stack([e_rj_dry, e_rj_irr], axis=0)
    
    # Update the cost matrix with exclude matrices; the transition cost for a cell that remain the same is 0.
    e_mrj = np.einsum('mrj,mrj,mrj->mrj', e_mrj, x_mrj, l_mrj_not)

    # -------------------------------------------------------------- #
    # Water license cost (upfront, amortised to annual, per cell).  #
    # -------------------------------------------------------------- #

    w_mrj = get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_water_delta_matrix(w_mrj, l_mrj, data)
    w_delta_mrj = np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not)  

    # -------------------------------------------------------------- #
    # Carborn costs of transitioning cells.                          #
    # -------------------------------------------------------------- #

    # apply the cost of carbon released by transitioning modified land to natural land
    ghg_t_mrj = ag_ghg.get_ghg_transition_penalties(data, lumap)
    ghg_t_mrj_cost = tools.amortise(ghg_t_mrj * settings.CARBON_PRICE_PER_TONNE)
    ghg_t_mrj_cost = np.einsum('mrj,mrj,mrj->mrj', ghg_t_mrj_cost, x_mrj, l_mrj_not) 

    # -------------------------------------------------------------- #
    # Total costs.                                                   #
    # -------------------------------------------------------------- #
    
    if separate:
        return {'Establishment cost': e_mrj, 'Water license cost': w_delta_mrj, 'Carborn resleasing cost': ghg_t_mrj_cost}  
    else:
        t_mrj = e_mrj + w_delta_mrj + ghg_t_mrj_cost
        return t_mrj


def get_asparagopsis_effect_t_mrj(data):
    """
    Gets the transition costs of asparagopsis taxiformis, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_precision_agriculture_effect_t_mrj(data):
    """
    Gets the effects on transition costs of asparagopsis taxiformis, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_ecological_grazing_effect_t_mrj(data):
    """
    Gets the effects on transition costs of ecological grazing, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_savanna_burning_effect_t_mrj(data):
    """
    Gets the effects on transition costs of savanna burning, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_agtech_ei_effect_t_mrj(data):
    """
    Gets the effects on transition costs of AgTech EI, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_agricultural_management_transition_matrices(data, t_mrj, yr_idx) -> Dict[str, np.ndarray]:
    asparagopsis_data = get_asparagopsis_effect_t_mrj(data)
    precision_agriculture_data = get_precision_agriculture_effect_t_mrj(data)
    eco_grazing_data = get_ecological_grazing_effect_t_mrj(data)
    sav_burning_data = get_savanna_burning_effect_t_mrj(data)
    agtech_ei_data = get_agtech_ei_effect_t_mrj(data)

    ag_management_data = {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
    }

    return ag_management_data


def get_asparagopsis_adoption_limits(data, yr_idx):
    """
    Gets the adoption limit of Asparagopsis taxiformis for each possible land use.
    """
    asparagopsis_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']:
        j = data.DESC2AGLU[lu]
        asparagopsis_limits[j] = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, 'Technical_Adoption']

    return asparagopsis_limits


def get_precision_agriculture_adoption_limit(data, yr_idx):
    """
    Gets the adoption limit of precision agriculture for each possible land use.
    """
    prec_agr_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']:
        j = data.DESC2AGLU[lu]
        prec_agr_limits[j] = data.PRECISION_AGRICULTURE_DATA[lu].loc[yr_cal, 'Technical_Adoption']

    return prec_agr_limits


def get_ecological_grazing_adoption_limit(data, yr_idx):
    """
    Gets the adoption limit of ecological grazing for each possible land use.
    """
    eco_grazing_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']:
        j = data.DESC2AGLU[lu]
        eco_grazing_limits[j] = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, 'Feasible Adoption (%)']

    return eco_grazing_limits


def get_savanna_burning_adoption_limit(data):
    """
    Gets the adoption limit of Savanna Burning for each possible land use
    """
    sav_burning_limits = {}
    for lu in AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']:
        j = data.DESC2AGLU[lu]
        sav_burning_limits[j] = 1

    return sav_burning_limits


def get_agtech_ei_adoption_limit(data, yr_idx):
    """
    Gets the adoption limit of AgTech EI for each possible land use.
    """
    agtech_ei_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']:
        j = data.DESC2AGLU[lu]
        agtech_ei_limits[j] = data.AGTECH_EI_DATA[lu].loc[yr_cal, 'Technical_Adoption']

    return agtech_ei_limits


def get_agricultural_management_adoption_limits(data, yr_idx) -> Dict[str, dict]:
    """
    An adoption limit represents the maximum percentage of cells (for each land use) that can utilise
    each agricultural management option.
    """
    asparagopsis_limits = get_asparagopsis_adoption_limits(data, yr_idx)
    precision_agriculture_limits = get_precision_agriculture_adoption_limit(data, yr_idx)
    eco_grazing_limits = get_ecological_grazing_adoption_limit(data, yr_idx)
    savanna_burning_limits = get_savanna_burning_adoption_limit(data)
    agtech_ei_limits = get_agtech_ei_adoption_limit(data, yr_idx)

    adoption_limits = {
        'Asparagopsis taxiformis': asparagopsis_limits,
        'Precision Agriculture': precision_agriculture_limits,
        'Ecological Grazing': eco_grazing_limits,
        'Savanna Burning': savanna_burning_limits,
        'AgTech EI': agtech_ei_limits,
    }

    return adoption_limits

