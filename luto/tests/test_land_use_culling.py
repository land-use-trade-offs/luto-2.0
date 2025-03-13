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

import math
import random
from typing import Any
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
import pytest

from luto.economics import land_use_culling

MAX_M = 2
MAX_R = 400
MAX_J = 28


def _generate_mock_mrj_matrix(
    values: list[Any], m: int = 0, r: int = 0, max_m: int = MAX_M, max_r: int = MAX_R
) -> np.ndarray:
    """
    Generates a matrix with shape (max_m, max_r, len(values)), and unpacks the
    `values` into the matrix at coordinates [m, r, :].

    Acts as a mock of an `mrj` matrix.
    """
    matrix = np.zeros((max_m, max_r, len(values))).astype(np.float32)
    matrix[m, r, :] = np.array(values)
    return matrix


@given(
    st.lists(st.floats(-150, 150), min_size=MAX_J, max_size=MAX_J),
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
    land_use_cull_percentage: the percentage of land usage options to cull
    """
    m = random.randint(0, MAX_M - 1)
    r = random.randint(0, MAX_R - 1)

    # generate cost metric matrix
    costs_mrj = _generate_mock_mrj_matrix(land_use_costs, m=m, r=r)

    # generate random x_mrj_mask based on num_valid
    is_valid_land_use = [True for _ in range(num_valid)] + [
        False for _ in range(len(land_use_costs) - num_valid)
    ]
    random.shuffle(is_valid_land_use)
    x_mrj_mask = _generate_mock_mrj_matrix(is_valid_land_use, m=m, r=r).astype(bool)

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
            m, r, x_mrj_mask, costs_mrj
        )

    for i, cost_value in enumerate(land_use_costs):
        is_culled = not cost_mask[i]
        if is_culled:
            assert (cost_value in costs_to_exclude) or not is_valid_land_use[i]


@given(
    st.lists(st.floats(-150, 150), min_size=MAX_J, max_size=MAX_J),
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
    m = random.randint(0, MAX_M - 1)
    r = random.randint(0, MAX_R - 1)

    # generate cost metric matrix
    costs_mrj = _generate_mock_mrj_matrix(land_use_costs, m=m, r=r)

    # generate random x_mrj_mask based on num_valid
    is_valid_land_use = [True for _ in range(num_valid)] + [
        False for _ in range(len(land_use_costs) - num_valid)
    ]
    random.shuffle(is_valid_land_use)
    x_mrj_mask = _generate_mock_mrj_matrix(is_valid_land_use, m=m, r=r).astype(bool)

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
        cost_mask = land_use_culling.get_absolute_cost_mask(m, r, x_mrj_mask, costs_mrj)

    if max_land_uses > num_valid:
        assert cost_mask is None
        return

    for i, cost_value in enumerate(land_use_costs):
        is_culled = not cost_mask[i]
        if is_culled:
            assert (cost_value in costs_to_exclude) or not is_valid_land_use[i]


@pytest.mark.parametrize(
    ("cull_mode", "cull_param", "cull_param_value"),
    [
        ("none", None, None),
        ("percentage", "LAND_USAGE_CULL_PERCENTAGE", 0.3),
        ("absolute", "MAX_LAND_USES_PER_CELL", 8),
    ],
)
def test_apply_agricultural_land_use_culling(
    cull_mode: str,
    cull_param: str,
    cull_param_value: Any,
):
    """
    Basic smoke test for the 'apply_agricultural_land_use_culling' function.

    If cull_mode is 'none', ensure the x_mrj matrix is not modified.
    Otherwise, ensure that the number of x_mrj values that are 1 is reduced.
    """
    m = random.randint(0, MAX_M - 1)
    r = random.randint(0, MAX_R - 1)

    x_mrj = _generate_mock_mrj_matrix([1 for _ in range(MAX_J)], m=m, r=r).astype(int)
    old_x_mrj = x_mrj.copy()

    cost_values = [random.random() for _ in range(MAX_J)]
    c_mrj = t_mrj = r_mrj = _generate_mock_mrj_matrix(cost_values, m=m, r=r)

    with (
        patch("luto.economics.land_use_culling.CULL_MODE", cull_mode),
        patch(f"luto.economics.land_use_culling.{cull_param}", cull_param_value),
    ):
        land_use_culling.apply_agricultural_land_use_culling(x_mrj, c_mrj, t_mrj, r_mrj)

    if cull_mode == "none":
        assert sum(x_mrj[m, r, :]) == MAX_J
        assert (x_mrj == old_x_mrj).all()
    else:
        assert sum(x_mrj[m, r, :]) < MAX_J
        assert (x_mrj != old_x_mrj).any()
