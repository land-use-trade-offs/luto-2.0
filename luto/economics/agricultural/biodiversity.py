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
Pure functions to calculate biodiversity by lm, lu.
"""


import itertools
import numpy as np

from luto import settings
from luto import tools
from luto.data import Data


################################ Functions below calculate biodiversity matrices for the bio-objective of the solover ################################
def get_bio_quality_score_mrj(data:Data):
    """
    Return b_mrj biodiversity score matrices by land management, cell, and land-use type.

    Parameters
    - data: The data object containing information about land management, cells, and land-use types.

    Returns
    - np.ndarray.
    """
    b_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)

    for j in range(data.N_AG_LUS):
        b_mrj[:, :, j] = (
            data.BIO_QUALITY_LDS -                                                     # Biodiversity score after Late Dry Season (LDS) burning
            (data.BIO_QUALITY_RAW * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[j]))    # Biodiversity degradation for land-use j
        ) * data.REAL_AREA    
    
    return b_mrj


def get_asparagopsis_effect_b_mrj(data:Data):
    """
    Gets biodiversity impacts of using Asparagopsis taxiformis (no effect)

    Parameters
    - data: The input data object containing NLMS and NCELLS attributes.

    Returns
    - An array of zeros with shape (data.NLMS, data.NCELLS, nlus).
    """
    nlus = len(settings.AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"])
    return np.zeros((data.NLMS, data.NCELLS, nlus), dtype=np.float32)


def get_precision_agriculture_effect_b_mrj(data:Data):
    """
    Gets biodiversity impacts of using Precision Agriculture (no effect)

    Parameters
    - data: The input data object containing NLMS and NCELLS information

    Returns
    - An array of zeros with shape (data.NLMS, data.NCELLS, nlus)
    """
    nlus = len(settings.AG_MANAGEMENTS_TO_LAND_USES["Precision Agriculture"])
    return np.zeros((data.NLMS, data.NCELLS, nlus), dtype=np.float32)


def get_ecological_grazing_effect_b_mrj(data:Data):
    """
    Gets biodiversity impacts of using Ecological Grazing (no effect)

    Parameters
    - data: The input data object containing information about NLMS and NCELLS.

    Returns
    - An array of zeros with shape (NLMS, NCELLS, nlus)
    """
    nlus = len(settings.AG_MANAGEMENTS_TO_LAND_USES["Ecological Grazing"])
    return np.zeros((data.NLMS, data.NCELLS, nlus), dtype=np.float32)


def get_savanna_burning_effect_b_mrj(data:Data):
    """
    Gets biodiversity impacts of using Savanna Burning.

    Land that can perform Savanna Burning but does not is penalised for not doing so.
    Thus, add back in the penalised amount to get the positive effect of Savanna
    Burning on biodiversity.

    Parameters
    - data: The input data containing information about land management and biodiversity.

    Returns
    - new_b_mrj: A numpy array representing the biodiversity impacts of using Savanna Burning.
    """
    nlus = len(settings.AG_MANAGEMENTS_TO_LAND_USES["Savanna Burning"])
    new_b_mrj = np.zeros((data.NLMS, data.NCELLS, nlus), dtype=np.float32)

    eds_sav_burning_biodiv_benefits = np.where(
        data.SAVBURN_ELIGIBLE, 
        data.BIO_QUALITY_RAW * (1 - settings.BIO_CONTRIBUTION_LDS) * data.REAL_AREA, 
        0
    ).astype(np.float32)
    

    for m, j in itertools.product(range(data.NLMS), range(nlus)):
        new_b_mrj[m, :, j] = eds_sav_burning_biodiv_benefits

    return new_b_mrj


def get_agtech_ei_effect_b_mrj(data:Data):
    """
    Gets biodiversity impacts of using AgTech EI (no effect)

    Parameters
    - data: The input data object containing information about NLMS and NCELLS.

    Returns
    - An array of zeros with shape (NLMS, NCELLS, nlus)
    """
    nlus = len(settings.AG_MANAGEMENTS_TO_LAND_USES["AgTech EI"])
    return np.zeros((data.NLMS, data.NCELLS, nlus), dtype=np.float32)


def get_biochar_effect_b_mrj(data:Data, ag_b_mrj: np.ndarray, yr_idx):
    """
    Gets biodiversity impacts of using Biochar. If the ag_b_mrj comes from bio_quality,
    then the biodiversity impact is calculated for the whole Australia. If the ag_b_mrj
    comes from bio_GBF2, then the biodiversity impact is calculated only for the 'priority degraded areas'.

    Parameters
    - data: The input data object containing information about NLMS and NCELLS.
    - ag_b_mrj: A numpy array representing the biodiversity scores by land management, cell, and land-use type.
    - yr_idx: The index of the target year for which the biodiversity impacts are to be

    Returns
    - new_b_mrj: A numpy array representing the biodiversity impacts of using Biochar.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    b_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['Biochar']:
        return b_mrj_effect

    for lu_idx, lu in enumerate(land_uses):
        biodiv_impact = data.BIOCHAR_DATA[lu].loc[yr_cal, 'Biodiversity_impact']

        if biodiv_impact != 1:
            j = lu_codes[lu_idx]
            b_mrj_effect[:, :, lu_idx] = ag_b_mrj[:, :, j] * (biodiv_impact - 1)
            
    return b_mrj_effect



def get_beef_hir_effect_b_mrj(data: Data, ag_b_mrj: np.ndarray) -> np.ndarray:
    """
    Gets biodiversity impacts of using HIR on beef. If the ag_b_mrj comes from bio_quality,
    then the biodiversity impact is calculated for the whole Australia. If the ag_b_mrj
    comes from bio_GBF2, then the biodiversity impact is calculated only for the 'priority degraded areas'.

    Parameters
    - data: The input data object containing information about NLMS and NCELLS.
    - ag_b_mrj: A numpy array representing the biodiversity scores by land management, cell, and land-use type.
    - yr_idx: The index of the target year for which the biodiversity impacts are to be calculated.
    
    Returns
    - b_mrj_effect: A numpy array representing the biodiversity impacts of using HIR on beef.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    b_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    unallocated_j = tools.get_unallocated_natural_land_code(data)
    # HIR's biodiversity contribution is based on that of unallocated land 
    unallocated_b_mr = ag_b_mrj[:, :, unallocated_j]

    for idx, lu_code in enumerate(lu_codes):
        b_mrj_effect[:, :, idx] = (
            unallocated_b_mr * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[lu_code])     # The proportional gap of biodiversity between beef-natural to full-natural
            - (1 - settings.HIR_CEILING_PERCENTAGE)                                     # HIR achienves HIR_CEILING_PERCENTAGE% of full-natural biodiversity
        )

    return b_mrj_effect


def get_sheep_hir_effect_b_mrj(data: Data, ag_b_mrj: np.ndarray) -> np.ndarray:
    """
    Gets biodiversity impacts of using HIR on sheep. If the ag_b_mrj comes from bio_quality,
    then the biodiversity impact is calculated for the whole Australia. If the ag_b_mrj
    comes from bio_GBF2, then the biodiversity impact is calculated only for the 'priority degraded areas'.

    Parameters
    - data: The input data object containing information about NLMS and NCELLS.
    - ag_b_mrj: A numpy array representing the biodiversity scores by land management, cell, and land-use type.
    - yr_idx: The index of the target year for which the biodiversity impacts are to be calculated.
    
    Returns
    - b_mrj_effect: A numpy array representing the biodiversity impacts of using HIR on sheep.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']
    lu_codes = [data.DESC2AGLU[lu] for lu in land_uses]
    b_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    unallocated_j = tools.get_unallocated_natural_land_code(data)
    # HIR's biodiversity contribution is based on that of unallocated land 
    unallocated_b_mr = ag_b_mrj[:, :, unallocated_j]

    for idx, lu_code in enumerate(lu_codes):
        b_mrj_effect[:, :, idx] = (
            unallocated_b_mr * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[lu_code])     # The proportional gap of biodiversity between beef-natural to full-natural
            - (1 - settings.HIR_CEILING_PERCENTAGE)                                     # HIR achienves HIR_CEILING_PERCENTAGE% of full-natural biodiversity
        )
        
    return b_mrj_effect


def get_ag_mgt_biodiversity_matrices(data:Data, ag_b_mrj: np.ndarray, yr_idx: int):
    """
    Calculate the biodiversity matrices for different agricultural management practices.

    Parameters
    - data: The input data used for calculations.

    Returns
    A dictionary containing the biodiversity matrices for different agricultural management practices.
    The keys of the dictionary represent the management practices, and the values represent the corresponding biodiversity matrices.
    """

    asparagopsis_data = get_asparagopsis_effect_b_mrj(data)                       
    precision_agriculture_data = get_precision_agriculture_effect_b_mrj(data)     
    eco_grazing_data = get_ecological_grazing_effect_b_mrj(data)                   
    sav_burning_data = get_savanna_burning_effect_b_mrj(data)                       
    agtech_ei_data = get_agtech_ei_effect_b_mrj(data)                               
    biochar_data = get_biochar_effect_b_mrj(data, ag_b_mrj, yr_idx)                 
    beef_hir_data = get_beef_hir_effect_b_mrj(data, ag_b_mrj)                       
    sheep_hir_data = get_sheep_hir_effect_b_mrj(data, ag_b_mrj)                     

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




################################ Functions below calculate biodiversity matrices for the bio-constraints of the solover ################################

def get_GBF2_MASK_area(data:Data) -> np.ndarray:
    return data.BIO_GBF2_MASK * data.REAL_AREA

def get_GBF3_NVIS_matrices_vr(data:Data) -> np.ndarray:
    return data.NVIS_LAYERS_LDS * data.REAL_AREA


def get_GBF4_SNES_matrix_sr(data:Data) -> np.ndarray:
    """
    Gets the SNES contributions  matrix.
    
    Returns
    -------
    np.ndarray
        indexed (s, r) where s is species (independent of species conversation limits) and r is cell
    """
    return np.where(
        data.SAVBURN_ELIGIBLE,
        data.BIO_GBF4_SPECIES_LAYERS * data.REAL_AREA * settings.BIO_CONTRIBUTION_LDS,
        data.BIO_GBF4_SPECIES_LAYERS * data.REAL_AREA
    ).astype(np.float32)
    

def get_GBF4_ECNES_matrix_sr(data:Data) -> np.ndarray:
    """
    Gets the ECNES contributions  matrix.
    
    Returns
    -------
    np.ndarray
        indexed (s, r) where s is species (independent of species conversation limits) and r is cell
    """
    return np.where(
        data.SAVBURN_ELIGIBLE,
        data.BIO_GBF4_COMUNITY_LAYERS * data.REAL_AREA * settings.BIO_CONTRIBUTION_LDS,
        data.BIO_GBF4_COMUNITY_LAYERS * data.REAL_AREA
    ).astype(np.float32)
    


def get_GBF8_matrix_sr(data:Data, target_year: int):
    return np.where(
        data.SAVBURN_ELIGIBLE,
        data.get_GBF8_bio_layers_by_yr(target_year) * data.REAL_AREA * settings.BIO_CONTRIBUTION_LDS,
        data.get_GBF8_bio_layers_by_yr(target_year) * data.REAL_AREA
    )


def get_ag_biodiversity_contribution(data:Data) -> np.ndarray:
    """
    Return b_j biodiversity contribution matrices by land-use type.

    Parameters
    - data: The data object containing information about land management, cells, and land-use types.

    Returns
    - np.ndarray.
    """
    return np.array(list(data.BIO_HABITAT_CONTRIBUTION_LOOK_UP.values()))


def get_ag_management_biodiversity_contribution(
    data:Data,
    yr_cal: int,
) -> dict[str, dict[int, np.ndarray]]:
    
    am_contr_dict = {}
    
    if settings.AG_MANAGEMENTS['Asparagopsis taxiformis']:
        am_contr_dict['Asparagopsis taxiformis'] = {
            j_idx: np.zeros(data.NCELLS).astype(np.float32)
            for j_idx, lu in enumerate(settings.AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis'])
        }
    if settings.AG_MANAGEMENTS['Precision Agriculture']:
        am_contr_dict['Precision Agriculture'] = {
            j_idx: np.zeros(data.NCELLS).astype(np.float32)
            for j_idx, lu in enumerate(settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture'])
        }
    if settings.AG_MANAGEMENTS['Ecological Grazing']:
        am_contr_dict['Ecological Grazing'] = {
            j_idx: np.zeros(data.NCELLS).astype(np.float32)
            for j_idx, lu in enumerate(settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing'])
        }
    if settings.AG_MANAGEMENTS['Savanna Burning']:
        am_contr_dict['Savanna Burning'] = {
            j_idx: np.where(data.SAVBURN_ELIGIBLE, (1 - settings.BIO_CONTRIBUTION_LDS), 0).astype(np.float32)
            for j_idx, lu in enumerate(settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning'])
        }
    if settings.AG_MANAGEMENTS['AgTech EI']:
        am_contr_dict['AgTech EI'] = {
            j_idx: np.zeros(data.NCELLS).astype(np.float32)
            for j_idx, lu in enumerate(settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI'])
        }
    if settings.AG_MANAGEMENTS['Biochar']:
        am_contr_dict['Biochar'] = {
            j_idx: (data.BIOCHAR_DATA[lu].loc[yr_cal, 'Biodiversity_impact'] - 1) * np.ones(data.NCELLS).astype(np.float32)
            for j_idx, lu in enumerate(settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar'])
        }
    if settings.AG_MANAGEMENTS['HIR - Beef']:
        am_contr_dict['HIR - Beef'] = {
            j_idx: (
                np.ones(data.NCELLS).astype(np.float32) 
                * (
                    (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[tools.get_natural_beef_code(data)])  # The proportional gap of biodiversity between beef-natural to full-natural
                    - (1 - settings.HIR_CEILING_PERCENTAGE)                                         # The gap that HIR's biodiversity contribution to full-natural biodiversity 
                )   
            )
            for j_idx, lu in enumerate(settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef'])
        }
    if settings.AG_MANAGEMENTS['HIR - Sheep']:
        am_contr_dict['HIR - Sheep'] = {
            j_idx: (
                np.ones(data.NCELLS).astype(np.float32)
                * (
                    (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[tools.get_natural_beef_code(data)])  # The proportional gap of biodiversity between beef-natural to full-natural
                    - (1 - settings.HIR_CEILING_PERCENTAGE)                                         # The gap that HIR's biodiversity contribution to full-natural biodiversity 
                )  
            )
            for j_idx, lu in enumerate(settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep'])
        }
    
    return am_contr_dict
