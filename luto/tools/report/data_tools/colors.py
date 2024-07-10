import itertools
from luto.tools.report.data_tools.parameters import LANDUSE_ALL_RAW,COMMODITIES_ALL,GHG_NAMES
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES

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
LANDUSE_ALL_COLORS = dict(zip(LANDUSE_ALL_RAW, itertools.cycle(COLORS)))
COMMODITIES_ALL_COLORS = dict(zip(COMMODITIES_ALL, itertools.cycle(COLORS)))
GHG_NAMES_COLORS = dict(zip(GHG_NAMES.values(), itertools.cycle(COLORS)))
AG_MANAGEMENTS_COLORS = dict(zip(AG_MANAGEMENTS_TO_LAND_USES.keys(), itertools.cycle(COLORS)))

WATER_SUPPLY_COLORS = dict(zip(['Dryland', 'Irrigated'], itertools.cycle(COLORS)))

DEMAND_TYEPS_COLORS = dict(zip(['Domestic', 'Exports', 'Feed', 'Imports', 'Production'], itertools.cycle(COLORS)))
DEMAND_CATEGORIES_COLORS = dict(zip(['Off-land', 'On-land'], itertools.cycle(COLORS)))




