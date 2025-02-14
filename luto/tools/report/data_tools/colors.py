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

import itertools

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.tools.report.data_tools.parameters import (LANDUSE_ALL_RAW,
                                                     COMMODITIES_ALL,
                                                     GHG_NAMES, 
                                                     LANDUSE_ALL_RENAMED)


COLORS = [
    "#8085e9",
    "#f15c80",
    "#e4d354",
    "#2b908f",
    "#f45b5b",
    "#7cb5ec",
    "#434348",
    "#90ed7d",
    "#f7a35c",
    "#91e8e1",
]

LANDUSE_CATEGORIES_COLORS = dict(zip(['Agricultural Landse', 'Agricultural Management','Non-Agricultural Landuse'], itertools.cycle(COLORS)))
LANDUSE_ALL_COLORS = dict(zip(LANDUSE_ALL_RENAMED, itertools.cycle(COLORS)))
COMMODITIES_ALL_COLORS = dict(zip(COMMODITIES_ALL, itertools.cycle(COLORS)))
GHG_NAMES_COLORS = dict(zip(GHG_NAMES.values(), itertools.cycle(COLORS)))
AG_MANAGEMENTS_COLORS = dict(zip(AG_MANAGEMENTS_TO_LAND_USES.keys(), itertools.cycle(COLORS)))

WATER_SUPPLY_COLORS = dict(zip(['Dryland', 'Irrigated'], itertools.cycle(COLORS)))

DEMAND_TYEPS_COLORS = dict(zip(['Domestic', 'Exports', 'Feed', 'Imports', 'Production'], itertools.cycle(COLORS)))
DEMAND_CATEGORIES_COLORS = dict(zip(['Off-land', 'On-land'], itertools.cycle(COLORS)))




