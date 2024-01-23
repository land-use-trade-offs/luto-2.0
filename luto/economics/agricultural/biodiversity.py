"""
Pure functions to calculate biodiversity by lm, lu.
"""

from typing import Dict
import numpy as np

from luto import settings


def get_non_penalty_land_uses(data) -> list[int]:
    """
    Return a list of land uses that contribute to biodiversity output without penalty.
    """
    return list(set(data.LU_NATURAL) - set(data.LU_LVSTK))


def get_livestock_land_uses(data) -> list[int]:
    """
    Return a list of land uses that contribute to biodiversity but are penalised as per the 
    BIODIV_LIVESTOCK_IMPACT setting.
    """
    return list(set(data.LU_NATURAL) & set(data.LU_LVSTK))


def get_breq_matrices(data):
    """Return b_mrj biodiversity score matrices by land management, cell, and land-use type."""
    b_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))

    biodiv_non_penalty_lus = get_non_penalty_land_uses(data)
    livestock_lus = get_livestock_land_uses(data)

    for j in biodiv_non_penalty_lus:
        b_mrj[:, :, j] = data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA

    if settings.BIODIV_LIVESTOCK_IMPACT > 0:
        for j in livestock_lus:
            b_mrj[:, :, j] = data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA * (1 - settings.BIODIV_LIVESTOCK_IMPACT)
    
    return b_mrj


def get_asparagopsis_effect_b_mrj(data) -> np.ndarray:
    """
    Gets biodiversity impacts of using Asparagopsis taxiformis (no effect)
    """
    return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))


def get_precision_agriculture_effect_b_mrj(data) -> np.ndarray:
    """
    Gets biodiversity impacts of using Precision Agriculture (no effect)
    """
    return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))


def get_ecological_grazing_effect_b_mrj(data) -> np.ndarray:
    """
    Gets biodiversity impacts of using Ecological Grazing (no effect)
    """
    return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))


def get_agricultural_management_biodiversity_matrices(data, b_mrj: np.ndarray) -> dict[str, np.ndarray]:
    asparagopsis_data = get_asparagopsis_effect_b_mrj(data)
    precision_agriculture_data = get_precision_agriculture_effect_b_mrj(data)
    eco_grazing_data = get_ecological_grazing_effect_b_mrj(data)

    ag_management_data = {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
    }

    return ag_management_data


def get_base_year_biodiversity_score(data) -> float:
    """
    Gets the biodiversity score of the base year (2010).
    """
    biodiv_non_penalty_lus = get_non_penalty_land_uses(data)
    livestock_lus = get_livestock_land_uses(data)

    non_penalty_cells_2010 = np.isin(data.LUMAP, np.array(list(biodiv_non_penalty_lus))).astype(int)
    livestock_cells_2010 = np.isin(data.LUMAP, np.array(list(livestock_lus))).astype(int)

    # Apply penalties for livestock land uses
    biodiv_2010_non_pen_score = (non_penalty_cells_2010 * data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA).sum()
    biodiv_2010_pen_score = (1 - settings.BIODIV_LIVESTOCK_IMPACT) * (
        livestock_cells_2010 * data.BIODIV_SCORE_WEIGHTED * data.REAL_AREA
    ).sum()

    biodiv_score_2010 = biodiv_2010_non_pen_score + biodiv_2010_pen_score
    return biodiv_score_2010


def get_biodiversity_limits(data, yr_idx) -> float:
    """
    Biodiversity score must hit settings.TOTAL_BIODIV_TARGET_SCORE by 
    settings.BIODIV_TARGET_ACHIEVEMENT_YEAR, beginning in 2010. 

    Assume that the biodiversity score may increase linearly over this time.
    """
    biodiv_score_2010 = get_base_year_biodiversity_score(data)

    no_years_to_reach_limit = settings.BIODIV_TARGET_ACHIEVEMENT_YEAR - 2010
    biodiv_targets_each_year = np.linspace(
        biodiv_score_2010, data.TOTAL_BIODIV_TARGET_SCORE, no_years_to_reach_limit + 1
    )
    return biodiv_targets_each_year[yr_idx - 2010]
