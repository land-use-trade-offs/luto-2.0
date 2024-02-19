import numpy as np

from luto.settings import CULL_MODE

# The {MAX_LAND_USES_PER_CELL}, and {LAND_USAGE_CULL_PERCENTAGE} settings are 
# mutually exclusive. Here we use a try-except block to set the value of the
# settings if they are not set in the settings.py file.
try:
    from luto.settings import MAX_LAND_USES_PER_CELL 
except ImportError: 
    MAX_LAND_USES_PER_CELL = None
    
try:
    from luto.settings import LAND_USAGE_CULL_PERCENTAGE
except ImportError:
    LAND_USAGE_CULL_PERCENTAGE = None


def get_percentage_cost_mask(m, r, x_mrj_mask, costs_mrj):
    """
    Exclude the least profitable LAND_USAGE_CULL_PERCENTAGE of land usage options for a given
    land management / cell.
    """
    # only consider costs that are relevant based on the exclusion matrix
    allowed_costs = costs_mrj[m, r, :][x_mrj_mask[m, r, :]]
    if len(allowed_costs) == 0:
        # this cell / land management pair has no valid land use options
        return None

    sorted_costs = np.sort(allowed_costs)
    include_percentage = 1 - LAND_USAGE_CULL_PERCENTAGE
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
    Include only the MAX_LAND_USES_PER_CELL most profitable land usage options for a given
    land management / cell.
    """
    # only consider costs that are relevant based on the exclusion matrix
    allowed_costs = costs_mrj[m, r, :][x_mrj_mask[m, r, :]]
    if len(allowed_costs) < MAX_LAND_USES_PER_CELL:
        # this cell / land management pair already has less than max_land_uses
        return None

    sorted_costs = np.sort(allowed_costs)
    max_cost = sorted_costs[MAX_LAND_USES_PER_CELL - 1]

    # modify exclusion mask to only include costs that are below the threshold
    cost_include_mask = costs_mrj[m, r, :] <= max_cost
    return cost_include_mask


def apply_agricultural_land_use_culling(x_mrj, c_mrj, t_mrj, r_mrj):
    """
    Refine the exclude matrix to cull unprofitable land uses based on the CULL_MODE setting.
    This function modifies the x_mrj matrix in-place.

    Args:
        x_mrj (np.ndarray): The 'exclude' matrix returned by `get_exclude_matrices`. This will
            be modified in-place by this function.
        c_mrj (np.ndarray): The 'cost' matrix.
        t_mrj (np.ndarray): The 'transition' matrix.
        r_mrj (np.ndarray): The 'revenue' matrix.
    """

    if CULL_MODE == "none":
        return

    print(f"Culling using {CULL_MODE=}...")
    if CULL_MODE == "percentage":
        print(f"    {LAND_USAGE_CULL_PERCENTAGE=}")
    elif CULL_MODE == "absolute":
        print(f"    {MAX_LAND_USES_PER_CELL=}")

    x_mrj_mask = x_mrj.astype(bool)
    costs_mrj = (c_mrj + t_mrj) - r_mrj
    for r in range(costs_mrj.shape[1]):
        # Apply cost masks for every land management option
        for m in range(costs_mrj.shape[0]):
            if CULL_MODE == "absolute":
                cost_include_mask = get_absolute_cost_mask(
                    m,
                    r,
                    x_mrj_mask,
                    costs_mrj,
                )
            elif CULL_MODE == "percentage":
                cost_include_mask = get_percentage_cost_mask(
                    m, r, x_mrj_mask, costs_mrj
                )
            else:
                raise ValueError(f"Unknown CULL_MODE={CULL_MODE}")

            if cost_include_mask is None:
                continue

            x_mrj[m, r, :] = x_mrj[m, r, :] & cost_include_mask
