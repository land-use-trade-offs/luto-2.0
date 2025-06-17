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



"""
Data about transitions costs.
"""

import numpy as np
from typing import Dict

from luto.data import Data
from luto.settings import AG_MANAGEMENTS, AG_MANAGEMENTS_TO_LAND_USES
from luto.economics.agricultural.water import get_wreq_matrices
import luto.economics.agricultural.ghg as ag_ghg
from luto import settings
import luto.tools as tools


def get_to_ag_exclude_matrices(data: Data, lumap: np.ndarray):
    """Return x_mrj exclude matrices.

    An exclude matrix indicates whether switching land-use for a certain cell r
    with land-use i to all other land-uses j under all land management types
    (i.e., dryland, irrigated) m is possible.

    Parameters
    ----------

    data: Data object.
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
    # Spatial exclusoin; if a land-use never exists in a SA2 region in 2010, it will be disallowed in this region forever
    x_mrj = data.EXCLUDE.copy().astype(np.int8)

    # Transition exclusion; Whether a landuse can be converted to another one (np.nan indicates NOT-ALLOW)
    t_ij = data.T_MAT.loc[:,data.AGRICULTURAL_LANDUSES].copy()                                          # Lexicographical transition matix (2D, all-lus to ag-lus).
    ag_cells, non_ag_cells = tools.get_ag_and_non_ag_cells(lumap)                                       # Get ag and non-ag index from base year lumap
    lumap2desc = np.vectorize(data.ALLLU2DESC.get, otypes=[str])
    
    t_rj = np.ones((data.NCELLS, len(data.AGRICULTURAL_LANDUSES))).astype(np.float32)       # Empty ones_rj array to be filled with transition flag (1 allow, 0 not allow)
    t_rj[ag_cells, :] = t_ij[lumap[ag_cells]]                                               # For ag cells in the base year lumap, get transition cost (np.nan is not-allow) for them
    t_rj[non_ag_cells, :] *= t_ij.sel(from_lu=lumap2desc(lumap[non_ag_cells]))              # For non-ag cells in the base year lumap, get transition cost (np.nan is not-allow) for them
    t_rj[non_ag_cells, :] *= t_ij.sel(from_lu=lumap2desc(data.LUMAP[non_ag_cells]))         # For non-ag cells, find its ag status in BASE_YR (2010), then get transition cost based on these 2010-ag status
    t_rj = np.where(np.isnan(t_rj), 0, 1).astype(np.int8)     

    # No-go exclusion; user-defined layer specifying which land-use are not disallowd at where
    no_go_x_mrj = np.ones_like(data.AG_L_MRJ)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go_j = data.DESC2AGLU[no_go_desc]
            no_go_x_mrj[:,:,no_go_j] = no_go_x_r

    return (x_mrj * t_rj * no_go_x_mrj).astype(np.int8)


def get_transition_matrices_ag2ag(data: Data, yr_idx: int, lumap: np.ndarray, lmmap: np.ndarray, separate=False):
    """
    Calculate the transition matrices for land-use and land management transitions.
    Args:
        data (Data object): The data object containing the necessary input data.
        yr_idx (int): The index of the current year.
        lumap (np.ndarray): Land use map of the base year for the transitions.
        lmmap (np.ndarray): Land management map of the base year for the transitions.
        separate (bool, optional): Whether to return separate cost matrices for each cost component.
                                   Defaults to False.
    Returns:
            numpy.ndarray or dict: The transition matrices for land-use and land management transitions.
                               If `separate` is False, returns a numpy array representing the total costs.
                               If `separate` is True, returns a dictionary with separate cost matrices for
                               establishment costs, Water license cost, and carbon releasing costs.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Return l_mrj (Boolean) for current land-use and land management
    l_mrj = tools.lumap2ag_l_mrj(lumap, lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    # Get the exclusion matrix
    x_mrj = get_to_ag_exclude_matrices(data, lumap)

    ag_cells, _ = tools.get_ag_and_non_ag_cells(lumap)

    n_ag_lms, ncells, n_ag_lus = data.AG_L_MRJ.shape

    # -------------------------------------------------------------- #
    # Establishment costs (upfront, amortised to annual, per cell).  #
    # -------------------------------------------------------------- #

    # Raw transition-cost matrix is in $/ha and lexigraphically ordered (shape: land-use x land-use).
    t_ij = data.AG_TMATRIX * data.TRANS_COST_MULTS[yr_cal]

    # Non-irrigation related transition costs for cell r to change to land-use j calculated based on lumap (in $/ha).
    # Only consider for cells currently being used for agriculture.
    e_rj = np.zeros((ncells, n_ag_lus)).astype(np.float32)
    e_rj[ag_cells, :] = t_ij[lumap[ag_cells]]

    # Amortise upfront costs to annualised costs and converted to $ per cell via REAL_AREA
    e_rj = tools.amortise(e_rj) * data.REAL_AREA[:, np.newaxis]

    # Repeat the establishment costs into dryland and irrigated land management types
    e_mrj = np.stack([e_rj, e_rj], axis=0)

    # Update the cost matrix with exclude matrices; the transition cost for a cell that remain the same is 0.
    e_mrj = np.einsum('mrj,mrj,mrj->mrj', e_mrj, x_mrj, l_mrj_not).astype(np.float32)
    e_mrj = np.nan_to_num(e_mrj)

    # -------------------------------------------------------------- #
    # Water license cost (upfront, amortised to annual, per cell).   #
    # -------------------------------------------------------------- #

    w_mrj = get_wreq_matrices(data, yr_idx)                                     # <unit: ML/cell>
    w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
    w_delta_mrj = np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not).astype(np.float32)

    # -------------------------------------------------------------- #
    # Carbon costs of transitioning cells.                           #
    # -------------------------------------------------------------- #

    # Apply the cost of carbon released by transitioning natural land to modified land
    ghg_transition = ag_ghg.get_ghg_transition_emissions(data, lumap, separate=True)        # <unit: t/ha>
    
    # ghg_transition *= data.REAL_AREA[:, np.newaxis]                                       # <unit: $/cell>  TODO: check if this is needed
    
    ghg_transition = {
        k:np.einsum('mrj,mrj,mrj->mrj', v, x_mrj, l_mrj_not).astype(np.float32)             # No GHG penalty for cells that remain the same, or are prohibited from transitioning
        for k, v in ghg_transition.items()
    }
    
    ghg_transition = {
        k:tools.amortise(v * data.get_carbon_price_by_yr_idx(yr_idx))                       # Amortise the GHG penalties
        for k,v in ghg_transition.items()
    }
    
    ghg_t_types = ghg_transition.keys()
    ghg_t_smrj = np.stack([ghg_transition[t] for t in ghg_t_types], axis=0)                 # s: ghg_t_types, m: land management, r: cell, j: land use
    ghg_t_mrj = np.einsum('smrj->mrj', ghg_t_smrj)

    # -------------------------------------------------------------- #
    # Total costs.                                                   #
    # -------------------------------------------------------------- #

    if separate:
        return {'Establishment cost': e_mrj, 'Water license cost': w_delta_mrj, **ghg_transition}
    else:
        t_mrj = e_mrj + w_delta_mrj + ghg_t_mrj
        return t_mrj


def get_transition_matrices_ag2ag_from_base_year(data: Data, yr_idx, base_year, separate=False):
    """
    Calculate the transition matrices for land-use and land management transitions.
    Args:
        data (Data object): The data object containing the necessary input data.
        yr_idx (int): The index of the current year.
        base_year (int): The base year for the transition calculations.
        separate (bool, optional): Whether to return separate cost matrices for each cost component.
                                   Defaults to False.
    Returns:
        numpy.ndarray or dict: The transition matrices for land-use and land management transitions.
                               If `separate` is False, returns a numpy array representing the total costs.
                               If `separate` is True, returns a dictionary with separate cost matrices for
                               establishment costs, Water license cost, and carbon releasing costs.
    """
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    return get_transition_matrices_ag2ag(data, yr_idx, lumap, lmmap, separate)
    


def get_asparagopsis_effect_t_mrj(data: Data):
    """
    Gets the transition costs of asparagopsis taxiformis, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_precision_agriculture_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of asparagopsis taxiformis, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_ecological_grazing_effect_t_mrj(data: Data):
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


def get_biochar_effect_t_mrj(data):
    """
    Gets the effects on transition costs of Biochar, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Biochar']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_beef_hir_effect_t_mrj(data):
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_sheep_hir_effect_t_mrj(data):
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_agricultural_management_transition_matrices(data: Data, t_mrj, yr_idx) -> Dict[str, np.ndarray]:
    
    asparagopsis_data = get_asparagopsis_effect_t_mrj(data)                     
    precision_agriculture_data = get_precision_agriculture_effect_t_mrj(data)   
    eco_grazing_data = get_ecological_grazing_effect_t_mrj(data)                
    sav_burning_data = get_savanna_burning_effect_t_mrj(data)                   
    agtech_ei_data = get_agtech_ei_effect_t_mrj(data)                           
    biochar_data = get_biochar_effect_t_mrj(data)                               
    beef_hir_data = get_beef_hir_effect_t_mrj(data)                             
    sheep_hir_data = get_sheep_hir_effect_t_mrj(data)                           

    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
        'Biochar': biochar_data,
        'HIR - Beef': beef_hir_data,
        'HIR - Sheep': sheep_hir_data,
    }


def get_asparagopsis_adoption_limits(data: Data, yr_idx):
    """
    Gets the adoption limit of Asparagopsis taxiformis for each possible land use.
    """
    asparagopsis_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']:
        j = data.DESC2AGLU[lu]
        asparagopsis_limits[j] = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, 'Technical_Adoption']

    return asparagopsis_limits


def get_precision_agriculture_adoption_limit(data: Data, yr_idx):
    """
    Gets the adoption limit of precision agriculture for each possible land use.
    """
    prec_agr_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']:
        j = data.DESC2AGLU[lu]
        prec_agr_limits[j] = data.PRECISION_AGRICULTURE_DATA[lu].loc[yr_cal, 'Technical_Adoption']

    return prec_agr_limits


def get_ecological_grazing_adoption_limit(data: Data, yr_idx):
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


def get_biochar_adoption_limit(data, yr_idx):
    """
    Gets the adoption limit of Biochar for each possible land use.
    """
    biochar_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in AG_MANAGEMENTS_TO_LAND_USES['Biochar']:
        j = data.DESC2AGLU[lu]
        biochar_limits[j] = data.BIOCHAR_DATA[lu].loc[yr_cal, 'Technical_Adoption']

    return biochar_limits


def get_beef_hir_adoption_limit(data: Data):
    """
    Gets the adoption limit of HIR - Beef for each possible land use.
    """
    hir_limits = {}
    for lu in AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']:
        j = data.DESC2AGLU[lu]
        hir_limits[j] = 1

    return hir_limits


def get_sheep_hir_adoption_limit(data: Data):
    """
    Gets the adoption limit of HIR - Sheep for each possible land use.
    """
    hir_limits = {}
    for lu in AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']:
        j = data.DESC2AGLU[lu]
        hir_limits[j] = 1

    return hir_limits


def get_agricultural_management_adoption_limits(data: Data, yr_idx) -> Dict[str, dict]:
    """
    An adoption limit represents the maximum percentage of cells (for each land use) that can utilise
    each agricultural management option.
    """
    ag_management_data = {}

    ag_management_data['Asparagopsis taxiformis'] = get_asparagopsis_adoption_limits(data, yr_idx)          if AG_MANAGEMENTS['Asparagopsis taxiformis'] else {data.DESC2AGLU[lu]: 0 for lu in AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']}
    ag_management_data['Precision Agriculture'] = get_precision_agriculture_adoption_limit(data, yr_idx)    if AG_MANAGEMENTS['Precision Agriculture'] else {data.DESC2AGLU[lu]: 0 for lu in AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']}
    ag_management_data['Ecological Grazing'] = get_ecological_grazing_adoption_limit(data, yr_idx)          if AG_MANAGEMENTS['Ecological Grazing'] else {data.DESC2AGLU[lu]: 0 for lu in AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']}
    ag_management_data['Savanna Burning'] = get_savanna_burning_adoption_limit(data)                        if AG_MANAGEMENTS['Savanna Burning'] else {data.DESC2AGLU[lu]: 0 for lu in AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']}
    ag_management_data['AgTech EI'] = get_agtech_ei_adoption_limit(data, yr_idx)                            if AG_MANAGEMENTS['AgTech EI'] else {data.DESC2AGLU[lu]: 0 for lu in AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']}
    ag_management_data['Biochar'] = get_biochar_adoption_limit(data, yr_idx)                                if AG_MANAGEMENTS['Biochar'] else {data.DESC2AGLU[lu]: 0 for lu in AG_MANAGEMENTS_TO_LAND_USES['Biochar']}
    ag_management_data['HIR - Beef'] = get_beef_hir_adoption_limit(data)                                    if AG_MANAGEMENTS['HIR - Beef'] else {data.DESC2AGLU[lu]: 0 for lu in AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']}
    ag_management_data['HIR - Sheep'] = get_sheep_hir_adoption_limit(data)                                  if AG_MANAGEMENTS['HIR - Sheep'] else {data.DESC2AGLU[lu]: 0 for lu in AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']}

    return ag_management_data


def get_lower_bound_agricultural_management_matrices(data: Data, base_year) -> dict[str, dict]:
    """
    Gets the lower bound for the agricultural land use of the current years optimisation.
    """

    if base_year == data.YR_CAL_BASE or base_year not in data.non_ag_dvars:
        return {
            am: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
            for am in AG_MANAGEMENTS_TO_LAND_USES
            if AG_MANAGEMENTS[am]
        }

    return {
        am: np.divide(
            np.floor(data.ag_man_dvars[base_year][am].astype(np.float32) * 10 ** settings.ROUND_DECMIALS)
            , 10 ** settings.ROUND_DECMIALS
        )
        for am in AG_MANAGEMENTS_TO_LAND_USES
        if AG_MANAGEMENTS[am]
    }


def get_regional_adoption_limits(data: Data, yr_cal: int):
    if settings.REGIONAL_ADOPTION_CONSTRAINTS != "on":
        return None, None
    
    ag_reg_adoption_constrs = []
    non_ag_reg_adoption_constrs = []

    for reg_id, lu_name, area_limit_ha in data.get_regional_adoption_limit_ha_by_year(yr_cal):
        reg_ind = np.where(data.REGIONAL_ADOPTION_ZONES == reg_id)[0]

        if lu_name in data.DESC2AGLU:
            lu_code = data.DESC2AGLU[lu_name]
            ag_reg_adoption_constrs.append([reg_id, lu_code, lu_name, reg_ind, area_limit_ha])

        elif lu_name in data.DESC2NONAGLU:
            lu_code = data.DESC2NONAGLU[lu_name] - settings.NON_AGRICULTURAL_LU_BASE_CODE
            non_ag_reg_adoption_constrs.append([reg_id, lu_code, lu_name, reg_ind, area_limit_ha])

        else:
            raise ValueError(f"Regional adoption constraint exists for unrecognised land use: {lu_name}")

    return ag_reg_adoption_constrs, non_ag_reg_adoption_constrs
