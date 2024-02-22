import numpy as np
from luto import settings


def get_percentage_cost_mask(m, r, x_mrj_mask, costs_mrj):
    """
    Exclude the least profitable settings.LAND_USAGE_CULL_PERCENTAGE of land usage options for a given
    land management / cell.
    """
    # only consider costs that are relevant based on the exclusion matrix
    allowed_costs = costs_mrj[m, r, :][x_mrj_mask[m, r, :]]
    if len(allowed_costs) == 0:
        # this cell / land management pair has no valid land use options
        return None

    sorted_costs = np.sort(allowed_costs)
    include_percentage = 1 - settings.LAND_USAGE_CULL_PERCENTAGE
    max_land_use_options = max(
        round(include_percentage * len(allowed_costs)),
        1,  # there should always be at least one option
    )
    max_cost = sorted_costs[max_land_use_options - 1]

    # modify exclusion mask to only include costs that are below the threshold
    cost_include_mask = costs_mrj[m, r, :] <= max_cost
    return cost_include_mask


def get_absolute_cost_mask(m, r, x_mrj_mask, costs_mrj):
    """
    Include only the settings.MAX_LAND_USES_PER_CELL most profitable land usage options for a given
    land management / cell.
    """
    # only consider costs that are relevant based on the exclusion matrix
    allowed_costs = costs_mrj[m, r, :][x_mrj_mask[m, r, :]]
    if len(allowed_costs) < settings.MAX_LAND_USES_PER_CELL:
        # this cell / land management pair already has less than max_land_uses
        return None

    sorted_costs = np.sort(allowed_costs)
    max_cost = sorted_costs[settings.MAX_LAND_USES_PER_CELL - 1]

    # modify exclusion mask to only include costs that are below the threshold
    cost_include_mask = costs_mrj[m, r, :] <= max_cost
    return cost_include_mask


def apply_agricultural_land_use_culling(x_mrj, c_mrj, t_mrj, r_mrj):
    """
    Refine the exclude matrix to cull unprofitable land uses based on the settings.CULL_MODE setting.
    This function modifies the x_mrj matrix in-place.

    Args:
        x_mrj (np.ndarray): The 'exclude' matrix returned by `get_exclude_matrices`. This will
            be modified in-place by this function.
        c_mrj (np.ndarray): The 'cost' matrix.
        t_mrj (np.ndarray): The 'transition' matrix.
        r_mrj (np.ndarray): The 'revenue' matrix.
    """

    if settings.CULL_MODE == "none":
        return

    x_mrj_mask = x_mrj.astype(bool)
    costs_mrj = (c_mrj + t_mrj) - r_mrj
    for r in range(costs_mrj.shape[1]):
        # Apply cost masks for every land management option
        for m in range(costs_mrj.shape[0]):
            if settings.CULL_MODE == "absolute":
                cost_include_mask = get_absolute_cost_mask(
                    m,
                    r,
                    x_mrj_mask,
                    costs_mrj,
                )
            elif settings.CULL_MODE == "percentage":
                cost_include_mask = get_percentage_cost_mask(
                    m, r, x_mrj_mask, costs_mrj
                )
            else:
                raise ValueError(f"Unknown settings.CULL_MODE={settings.CULL_MODE}")

            if cost_include_mask is None:
                continue

            x_mrj[m, r, :] = x_mrj[m, r, :] & cost_include_mask
