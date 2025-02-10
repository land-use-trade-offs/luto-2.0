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
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.data import Data


def get_breq_matrices(data: Data):
    """
    Return b_mrj biodiversity score matrices by land management, cell, and land-use type.

    Parameters:
    - data: The data object containing information about land management, cells, and land-use types.

    Returns:
    - np.ndarray.
    """
    b_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))

    for j in range(data.N_AG_LUS):
        b_mrj[:, :, j] = (
            data.BIODIV_RAW_WEIGHTED_LDS -                                                      # Biodiversity score after Late Dry Season (LDS) burning
            (data.BIODIV_SCORE_RAW_WEIGHTED * (1 - data.BIODIV_HABITAT_DEGRADE_LOOK_UP[j]))     # Biodiversity degradation for land-use j
        ) * data.REAL_AREA    
    
    return b_mrj


def get_asparagopsis_effect_b_mrj(data: Data):
    """
    Gets biodiversity impacts of using Asparagopsis taxiformis (no effect)

    Parameters:
    - data: The input data object containing NLMS and NCELLS attributes.

    Returns:
    - An array of zeros with shape (data.NLMS, data.NCELLS, nlus).
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_precision_agriculture_effect_b_mrj(data: Data):
    """
    Gets biodiversity impacts of using Precision Agriculture (no effect)

    Parameters:
    - data: The input data object containing NLMS and NCELLS information

    Returns:
    - An array of zeros with shape (data.NLMS, data.NCELLS, nlus)
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Precision Agriculture"])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_ecological_grazing_effect_b_mrj(data: Data):
    """
    Gets biodiversity impacts of using Ecological Grazing (no effect)

    Parameters:
    - data: The input data object containing information about NLMS and NCELLS.

    Returns:
    - An array of zeros with shape (NLMS, NCELLS, nlus)
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Ecological Grazing"])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_savanna_burning_effect_b_mrj(data: Data):
    """
    Gets biodiversity impacts of using Savanna Burning.

    Land that can perform Savanna Burning but does not is penalised for not doing so.
    Thus, add back in the penalised amount to get the positive effect of Savanna
    Burning on biodiversity.

    Parameters:
    - data: The input data containing information about land management and biodiversity.

    Returns:
    - new_b_mrj: A numpy array representing the biodiversity impacts of using Savanna Burning.
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Savanna Burning"])
    new_b_mrj = np.zeros((data.NLMS, data.NCELLS, nlus))

    eds_sav_burning_biodiv_benefits = np.where( data.SAVBURN_ELIGIBLE, 
                                                (1 - settings.LDS_BIODIVERSITY_VALUE) * data.BIODIV_SCORE_RAW_WEIGHTED * data.REAL_AREA, 
                                                0
                                              )
    

    for m, j in itertools.product(range(data.NLMS), range(nlus)):
        new_b_mrj[m, :, j] = eds_sav_burning_biodiv_benefits

    return new_b_mrj


def get_agtech_ei_effect_b_mrj(data: Data):
    """
    Gets biodiversity impacts of using AgTech EI (no effect)

    Parameters:
    - data: The input data object containing information about NLMS and NCELLS.

    Returns:
    - An array of zeros with shape (NLMS, NCELLS, nlus)
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["AgTech EI"])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_biochar_effect_b_mrj(data: Data, ag_b_mrj: np.ndarray, yr_idx):
    """
    Gets biodiversity impacts of using Biochar

    Parameters:
    - data: The input data object containing information about NLMS and NCELLS.

    Returns:
    - new_b_mrj: A numpy array representing the biodiversity impacts of using Biochar.
    """
    land_uses = AG_MANAGEMENTS_TO_LAND_USES['Biochar']
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


def get_agricultural_management_biodiversity_matrices(data: Data, ag_b_mrj: np.ndarray, yr_idx: int):
    """
    Calculate the biodiversity matrices for different agricultural management practices.

    Parameters:
    - data: The input data used for calculations.

    Returns:
    A dictionary containing the biodiversity matrices for different agricultural management practices.
    The keys of the dictionary represent the management practices, and the values represent the corresponding biodiversity matrices.
    """

    asparagopsis_data = get_asparagopsis_effect_b_mrj(data) if settings.AG_MANAGEMENTS['Asparagopsis taxiformis'] else 0
    precision_agriculture_data = get_precision_agriculture_effect_b_mrj(data) if settings.AG_MANAGEMENTS['Precision Agriculture'] else 0
    eco_grazing_data = get_ecological_grazing_effect_b_mrj(data) if settings.AG_MANAGEMENTS['Ecological Grazing'] else 0
    sav_burning_data = get_savanna_burning_effect_b_mrj(data) if settings.AG_MANAGEMENTS['Savanna Burning'] else 0
    agtech_ei_data = get_agtech_ei_effect_b_mrj(data) if settings.AG_MANAGEMENTS['AgTech EI'] else 0
    biochar_data = get_biochar_effect_b_mrj(data, ag_b_mrj, yr_idx) if settings.AG_MANAGEMENTS['Biochar'] else 0

    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
        'Biochar': biochar_data,
    }


def get_biodiversity_limits(data: Data, yr_cal: int):
    """
    Calculate the biodiversity limits for a given year used as a constraint.

    The biodiversity score target timeline is specified in data.BIODIV_GBF_TARGET_2.

    Parameters:
    - data: The data object containing relevant information.
    - yr_cal: The calendar year for which to calculate the biodiversity limits.

    Returns:
    - The biodiversity limit for the given year.

    """

    return data.BIODIV_GBF_TARGET_2[yr_cal]
