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


def get_major_vegetation_matrices(data: Data) -> np.ndarray:
    """
    Get the matrix containing the contribution of each cell/ag. land use combination 
    to each major vegetation group.

    Returns:
    - Array indexed by (m, r, j, v) containing the contributions.
    """
    mvg_mrjv = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS, data.N_MVG_CLASSES))
    for m in range(data.NLMS):
        for j in data.LU_NATURAL:
            mvg_mrjv[m, :, j, :] = data.MAJOR_VEGETATION_GROUPS_RV
            for v in range(data.N_MVG_CLASSES):
                mvg_mrjv[m, :, j, v] *= data.REAL_AREA  # Account for cells' proportional contributions based on cell size

    return mvg_mrjv


def get_major_vegetation_group_limits(data: Data, yr_cal: int) -> tuple[dict[int, float], dict[int, str]]:
    """
    Gets the correct major vegetation group targets for the given year (yr_cal).

    Returns:
    - dict[int, float]
        A dictionary indexed by veg group (v) with the target for each group.
    - dict[int, str]
        A dictionary mapping of veg group index to name
    """
    if yr_cal >= settings.MAJOR_VEG_GROUP_TARGET_YEAR:
        limits = data.MVG_PROP_FINAL_TARGETS
    
    elif yr_cal <= data.YR_CAL_BASE:
        limits = data.MVG_PROP_TARGETS_BY_YEAR[data.YR_CAL_BASE]

    else:
        limits = data.MVG_PROP_TARGETS_BY_YEAR[yr_cal]

    ra_sum = data.REAL_AREA.sum()
    for v in limits:
        limits[v] *= ra_sum

    return limits, data.MAJOR_VEG_GROUP_NAMES


def get_major_vegetation_contrs_outside_study_area(data: Data) -> dict[int, float]:
    """
    Gets a dictionary mapping each major vegetation group (v) the contribution of
    land outside LUTO's study area that applies to it.
    """
    props = data.MVG_PROP_OUTSIDE_STUDY_AREA
    for v in props:
        props[v] *= data.REAL_AREA
    return props
