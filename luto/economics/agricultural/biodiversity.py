"""
Pure functions to calculate biodiversity by lm, lu.
"""

from typing import Dict
import numpy as np

from luto import settings
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


def get_non_penalty_land_uses(data) -> list[int]:
    """
    Return a list of land uses that contribute to biodiversity output without penalty.
    """
    return list(set(data.LU_NATURAL) - set(data.LU_LVSTK_INDICES))  # returns [23], the index of Unallocated - natural land index


def get_livestock_natural_land_land_uses(data) -> list[int]:
    """
    Return a list of land uses that contribute to biodiversity but are penalised as per the 
    BIODIV_LIVESTOCK_IMPACT setting (i.e., livestock on natural land).
    """
    return list(set(data.LU_NATURAL) & set(data.LU_LVSTK_INDICES)) # returns [2, 6, 15], Beef, Dairy, Sheep on natural land 


def get_breq_matrices(data):
    """Return b_mrj biodiversity score matrices by land management, cell, and land-use type."""
    b_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))

    biodiv_non_penalty_lus = get_non_penalty_land_uses(data)
    livestock_nat_land_lus = get_livestock_natural_land_land_uses(data)

    for j in biodiv_non_penalty_lus:
        b_mrj[:, :, j] = data.BIODIV_SCORE_RAW * data.REAL_AREA

    # if settings.BIODIV_LIVESTOCK_IMPACT > 0:
    for j in livestock_nat_land_lus:
        b_mrj[:, :, j] = data.BIODIV_SCORE_WEIGHTED_LDS_BURNING * data.REAL_AREA * (1 - settings.BIODIV_LIVESTOCK_IMPACT)
    
    return b_mrj


def get_asparagopsis_effect_b_mrj(data):
    """
    Gets biodiversity impacts of using Asparagopsis taxiformis (no effect)
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_precision_agriculture_effect_b_mrj(data):
    """
    Gets biodiversity impacts of using Precision Agriculture (no effect)
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Precision Agriculture"])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_ecological_grazing_effect_b_mrj(data):
    """
    Gets biodiversity impacts of using Ecological Grazing (no effect)
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Ecological Grazing"])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_savanna_burning_effect_b_mrj(data):
    """
    Gets biodiversity impacts of using Savanna Burning.

    Land that can perform Savanna Burning but does not is penalised for not doing so.
    Thus, add back in the penalised amount to get the positive effect of Savanna
    Burning on biodiversity.
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["Savanna Burning"])
    new_b_mrj = np.zeros((data.NLMS, data.NCELLS, nlus))

    eds_sav_burning_biodiv_benefits = (1 - settings.LDS_BIODIVERSITY_VALUE) * data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA

    for m in range(data.NLMS):
        for j in range(nlus):
            new_b_mrj[m, :, j] = eds_sav_burning_biodiv_benefits

    return new_b_mrj


def get_agtech_ei_effect_b_mrj(data):
    """
    Gets biodiversity impacts of using AgTech EI (no effect)
    """
    nlus = len(AG_MANAGEMENTS_TO_LAND_USES["AgTech EI"])
    return np.zeros((data.NLMS, data.NCELLS, nlus))


def get_agricultural_management_biodiversity_matrices(data):
    asparagopsis_data = get_asparagopsis_effect_b_mrj(data)
    precision_agriculture_data = get_precision_agriculture_effect_b_mrj(data)
    eco_grazing_data = get_ecological_grazing_effect_b_mrj(data)
    sav_burning_data = get_savanna_burning_effect_b_mrj(data)
    agtech_ei_data = get_agtech_ei_effect_b_mrj(data)

    ag_management_data = {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
    }

    return ag_management_data


def get_base_year_biodiversity_score(data):
    """
    Gets the biodiversity score of the base year (2010).
    """
    biodiv_non_penalty_lus = get_non_penalty_land_uses(data)
    livestock_nat_land_lus = get_livestock_natural_land_land_uses(data)

    non_penalty_cells_2010 = np.isin(data.LUMAP, np.array(list(biodiv_non_penalty_lus))).astype(int)
    livestock_cells_2010 = np.isin(data.LUMAP, np.array(list(livestock_nat_land_lus))).astype(int)

    # Apply penalties for livestock land uses
    biodiv_2010_non_pen_score = (non_penalty_cells_2010 * data.BIODIV_SCORE_RAW * data.REAL_AREA).sum()
    biodiv_2010_pen_score = (1 - settings.BIODIV_LIVESTOCK_IMPACT) * (
        livestock_cells_2010 * data.BIODIV_SCORE_RAW * data.REAL_AREA
    ).sum()

    return biodiv_2010_non_pen_score + biodiv_2010_pen_score


def get_biodiversity_limits(data, yr_cal):
    """
    Biodiversity score must hit data.TOTAL_BIODIV_TARGET_SCORE by 
    settings.BIODIV_TARGET_ACHIEVEMENT_YEAR, beginning in 2010. 

    Assume that the biodiversity score may increase linearly over this time.
    """
    biodiv_score_2010 = get_base_year_biodiversity_score(data)

    no_years_to_reach_limit = settings.BIODIV_TARGET_ACHIEVEMENT_YEAR - data.YR_CAL_BASE

    biodiv_target_score = data.TOTAL_BIODIV_TARGET_SCORE
    
    if biodiv_target_score < biodiv_score_2010:
        # In the case that the 2010 biodiversity score exceeds the calculated
        # target score, use the 2010 score as the target instead.
        biodiv_target_score = biodiv_score_2010

    if yr_cal >= settings.BIODIV_TARGET_ACHIEVEMENT_YEAR:
        # For each year after the target achievement year, the limit is equal
        # to that of the target achievement year.
        return biodiv_target_score

    biodiv_targets_each_year = np.linspace(
        biodiv_score_2010, biodiv_target_score, no_years_to_reach_limit + 1
    )

    return biodiv_targets_each_year[yr_cal - data.YR_CAL_BASE]
