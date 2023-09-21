import math
import random
from typing import Any
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from luto.economics import land_use_culling


def _generate_flat_mrj_matrix(values: list[Any]):
    """
    Generates a 3D matrix with shape (1, 1, len(values)).
    Acts as a mock of an MRJ matrix with m = r = 0
    """
    return np.array([[values]])


@given(
    st.lists(st.floats(-150, 150), min_size=28, max_size=28),
    st.integers(min_value=3, max_value=16),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_percentage_cost_mask(
    land_use_costs: list[float],
    num_valid: int,
    land_use_cull_percentage: float,
):
    """
    Ensure that getting the percentage cost mask works as expected

    land_use_costs: randomly generated cost metric for land usage options
    num_valid: the number of land usage options that are considered 'valid' to start with
        - used to construct a dummy x_mrj_mask
    land_usage_percentage: the percentage of land usage options to cull
    """
    # generate cost metric matrix
    costs_mrj = _generate_flat_mrj_matrix(land_use_costs)

    # generate random x_mrj_mask based on num_valid
    is_valid_land_use = [True for _ in range(num_valid)] + [
        False for _ in range(len(land_use_costs) - num_valid)
    ]
    random.shuffle(is_valid_land_use)
    x_mrj_mask = _generate_flat_mrj_matrix(is_valid_land_use)

    # determine which costs should be excluded
    num_costs_to_exclude = math.ceil(land_use_cull_percentage * num_valid)
    costs_to_consider = [
        cost for i, cost in enumerate(land_use_costs) if is_valid_land_use[i]
    ]
    costs_to_exclude = sorted(costs_to_consider, reverse=True)[0:num_costs_to_exclude]

    # get and validate the cost mask with the given percentage settings
    with patch(
        "luto.economics.land_use_culling.LAND_USAGE_CULL_PERCENTAGE",
        land_use_cull_percentage,
    ):
        cost_mask = land_use_culling.get_percentage_cost_mask(
            0, 0, x_mrj_mask, costs_mrj
        )

    for i, cost_value in enumerate(land_use_costs):
        is_culled = not cost_mask[i]
        if is_culled:
            assert (cost_value in costs_to_exclude) or not is_valid_land_use[i]


@given(
    st.lists(st.floats(-150, 150), min_size=28, max_size=28),
    st.integers(min_value=3, max_value=16),
    st.integers(min_value=3, max_value=16),
)
def test_absolute_cost_mask(
    land_use_costs: list[float],
    num_valid: int,
    max_land_uses: int,
):
    """
    Ensure that getting the absolute cost mask modifies the x_mrj matrix as expected

    land_use_costs: randomly generated cost metric for land usage options
    num_valid: the number of land usage options that are considered 'valid' to start with
        - used to construct a dummy x_mrj_mask
    max_land_uses: the maximum number of land use options that should not be culled
    """
    # generate cost metric matrix
    costs_mrj = _generate_flat_mrj_matrix(land_use_costs)

    # generate random x_mrj_mask based on num_valid
    is_valid_land_use = [True for _ in range(num_valid)] + [
        False for _ in range(len(land_use_costs) - num_valid)
    ]
    random.shuffle(is_valid_land_use)
    x_mrj_mask = _generate_flat_mrj_matrix(is_valid_land_use)

    # determine which costs should be excluded
    num_costs_to_exclude = max(num_valid - max_land_uses, 0)
    costs_to_consider = [
        cost for i, cost in enumerate(land_use_costs) if is_valid_land_use[i]
    ]
    costs_to_exclude = sorted(costs_to_consider, reverse=True)[0:num_costs_to_exclude]

    # get and validate the cost mask with the given percentage settings
    with patch(
        "luto.economics.land_use_culling.MAX_LAND_USES_PER_CELL",
        max_land_uses,
    ):
        cost_mask = land_use_culling.get_absolute_cost_mask(0, 0, x_mrj_mask, costs_mrj)

    if max_land_uses > num_valid:
        assert cost_mask is None
        return

    for i, cost_value in enumerate(land_use_costs):
        is_culled = not cost_mask[i]
        if is_culled:
            assert (cost_value in costs_to_exclude) or not is_valid_land_use[i]
