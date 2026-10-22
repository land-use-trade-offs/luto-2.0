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

import xarray as xr

# import luto settings
from luto.settings import (
    INPUT_DIR, 
    SCENARIO, 
    DIET_DOM, 
    DIET_GLOB,
    CONVERGENCE, 
    IMPORT_TREND, 
    WASTE, 
    FEED_EFFICIENCY,
    EGGS_AVG_WEIGHT
)


def get_demand_xr(egg_weight=EGGS_AVG_WEIGHT) -> xr.Dataset:
    """
    The demand under the running scenario: one variable per quantity (DOMESTIC, EXPORTS, IMPORTS, FEED, PRODUCTION)
    over (Commodity, year), eggs converted from count to tonnes.

    Args:
        egg_weight (int, optional): The weight of each egg in grams. Defaults to settings.EGGS_AVG_WEIGHT.

    Returns
        xr.Dataset: The demand of every commodity and year.
    """
    with xr.open_dataset(f'{INPUT_DIR}/demand_projections.nc') as demand_ds:
        demand = (
            demand_ds
            .sel(SCENARIO=SCENARIO, DIET_DOM=DIET_DOM, DIET_GLOB=DIET_GLOB, CONVERGENCE=CONVERGENCE,
                 IMPORT_TREND=IMPORT_TREND, WASTE=WASTE, FEED_EFFICIENCY=FEED_EFFICIENCY)
            .reset_coords(drop=True)
            .load()
        )

    # Convert eggs from count to tonnes
    return xr.where(demand['Commodity'] == 'eggs', demand * egg_weight / 1000 / 1000, demand)