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

import luto.settings as settings


# Get the root directory of the data
YR_BASE = 2010

LU_CROPS = [
    'Apples','Citrus','Cotton','Grapes','Hay','Nuts','Other non-cereal crops',
    'Pears','Plantation fruit','Rice','Stone fruit','Sugar','Summer cereals',
    'Summer legumes','Summer oilseeds','Tropical stone fruit','Vegetables',
    'Winter cereals','Winter legumes','Winter oilseeds'
]

LU_LVSTKS = [
    'Beef - natural land','Dairy - natural land','Sheep - natural land',
    'Beef - modified land','Dairy - modified land','Sheep - modified land'
]

LU_UNALLOW = ['Unallocated - modified land', 'Unallocated - natural land']


COMMIDOTY_GROUP = {
   "Beef meat": "Animal Products",
   "Sheep meat": "Animal Products",
   "Dairy": "Animal Products",
   "Beef live export": "Animal Products",
   "Sheep live export": "Animal Products",
   "Sheep wool": "Animal Products",
   "Apples": "Fruits",
   "Citrus": "Fruits",
   "Grapes": "Fruits",
   "Plantation fruit": "Fruits",
   "Stone fruit": "Fruits",
   "Tropical stone fruit": "Fruits",
   "Pears": "Fruits",
   "Nuts": "Fruits",
   "Summer cereals": "Grains/Legumes",
   "Winter cereals": "Grains/Legumes",
   "Rice": "Grains/Legumes",
   "Summer legumes": "Grains/Legumes",
   "Winter legumes": "Grains/Legumes",
   "Vegetables": "Crops",
   "Summer oilseeds": "Crops",
   "Winter oilseeds": "Crops",
   "Cotton": "Crops",
   "Sugar": "Crops",
   "Other non-cereal crops": "Crops",
   "Hay": "Hay",
   "Aquaculture": "Off-land Products",
   "Chicken": "Off-land Products",
   "Eggs": "Off-land Products",
   "Pork": "Off-land Products"
}


COMMODITIES_ALL = [
    'Apples','Beef live export','Beef meat','Citrus','Cotton','Dairy','Grapes',
    'Hay','Nuts','Other non-cereal crops', 'Pears', 'Plantation fruit',
    'Rice', 'Sheep live export', 'Sheep meat', 'Sheep wool', 'Stone fruit', 'Sugar',
    'Summer cereals', 'Summer legumes', 'Summer oilseeds', 'Tropical stone fruit',
    'Vegetables','Winter cereals','Winter legumes','Winter oilseeds',
    'Aquaculture', 'Chicken', 'Eggs', 'Pork'
]




# Define land use code for am and non-ag land uses
AM_SELECT = [i for i in settings.AG_MANAGEMENTS if settings.AG_MANAGEMENTS[i]]
AM_DESELECT = [i for i in settings.AG_MANAGEMENTS if not settings.AG_MANAGEMENTS[i]]
AM_MAP_CODES = {i:(AM_SELECT.index(i) + 1) for i in AM_SELECT}

NON_AG_SELECT = [i for i in settings.NON_AG_LAND_USES if settings.NON_AG_LAND_USES[i]]
NON_AG_DESELECT = [i for i in settings.NON_AG_LAND_USES if not settings.NON_AG_LAND_USES[i]]
NON_AG_MAP_CODES = {i:(NON_AG_SELECT.index(i) + 1) for i in NON_AG_SELECT}

AM_NON_AG_CODES = {**AM_MAP_CODES, **NON_AG_MAP_CODES}
AM_NON_AG_REMOVED_DESC = AM_DESELECT + NON_AG_DESELECT



# Define the renaming of the Agricultural-Managment and Non-Agricultural 
RENAME_AM = {
    "Asparagopsis taxiformis": "Methane reduction (livestock)",
    "Precision Agriculture": "Agricultural technology (fertiliser)", 
    "Ecological Grazing": "Regenerative agriculture (livestock)", 
    "Savanna Burning": "Early dry-season savanna burning",
    "AgTech EI": "Agricultural technology (energy)",
    'Biochar': "Biochar (soil amendment)",
    'HIR - Beef': "Human-induced regeneration (Beef)",
    'HIR - Sheep': "Human-induced regeneration (Sheep)",
    'Utility Solar PV': "Utility Solar PV",
    'Onshore Wind': "Onshore wind"
}

RENAME_NON_AG = {
    "Environmental Plantings": "Environmental plantings (mixed species)",
    "Riparian Plantings": "Riparian buffer restoration (mixed species)",
    "Sheep Agroforestry": "Agroforestry (mixed species + sheep)",
    "Beef Agroforestry": "Agroforestry (mixed species + beef)",
    "Carbon Plantings (Block)": "Carbon plantings (monoculture)",
    "Sheep Carbon Plantings (Belt)": "Farm forestry (hardwood timber + sheep)",
    "Beef Carbon Plantings (Belt)": "Farm forestry (hardwood timber + beef)",
    "BECCS": "BECCS (Bioenergy with Carbon Capture and Storage)",
    "Destocked - natural land": "Destocked - natural land",
}

RENAME_AM_NON_AG = {**RENAME_AM, **RENAME_NON_AG}

# Read the land uses from the file
with open(f'{settings.INPUT_DIR}/ag_landuses.csv') as f:
    AG_LANDUSE = [line.strip() for line in f]
  
    
# Get the non-agricultural land uses raw names
NON_AG_LANDUSE_RAW = list(settings.NON_AG_LAND_USES.keys())
NON_AG_LANDUSE_RAW = [i for i in NON_AG_LANDUSE_RAW if settings.NON_AG_LAND_USES[i]]


# Merge the land uses
LANDUSE_ALL_RAW = AG_LANDUSE + NON_AG_LANDUSE_RAW
LANDUSE_ALL_RENAMED = ['Agricultural Management', 'ALL'] + AG_LANDUSE + list(RENAME_NON_AG.values())  + ['Outside LUTO study area'] 


# Define the land use groups
GROUP_LU = {
    'Beef': [
        'Beef - natural land',
        'Beef - modified land',
    ],
    'Carbon-focussed sequestration': [
        'Carbon plantings (monoculture)',
        'BECCS (Bioenergy with Carbon Capture and Storage)',
        'Farm forestry (hardwood timber + sheep)',
        'Farm forestry (hardwood timber + beef)',
    ],
    'Cereals': [
        'Summer cereals',
        'Winter cereals',
    ],
    'Dairy': [
        'Dairy - natural land',
        'Dairy - modified land',
    ],
    'Fruit, vegetables, nuts, legumes': [
        'Apples',
        'Citrus',
        'Grapes',
        'Nuts',
        'Pears',
        'Plantation fruit',
        'Stone fruit',
        'Summer legumes',
        'Tropical stone fruit',
        'Vegetables',
        'Winter legumes',
    ],
    'Hay': [
        'Hay',
    ],
    'Nature positive sequestration': [
        'Environmental plantings (mixed species)',
        'Riparian buffer restoration (mixed species)',
        'Agroforestry (mixed species + sheep)',
        'Agroforestry (mixed species + beef)',
    ],
    'Oilseeds': [
        'Summer oilseeds',
        'Winter oilseeds',
    ],
    'Other crops': [
        'Cotton',
        'Other non-cereal crops',
        'Rice',
        'Sugar',
    ],
    'Sheep': [
        'Sheep - natural land',
        'Sheep - modified land',
    ],
    'Unallocated - modified land': [
        'Unallocated - modified land',
    ],
    'Unallocated - natural land': [
        'Unallocated - natural land',
    ],
    'Human-induced regeneration': [
        'Destocked - natural land',
    ],
}




# Define the GHG categories
GHG_NAMES = {
    # Agricultural Land-use
    'TCO2E_CHEM_APPL': 'Chemical Application',
    'TCO2E_CROP_MGT': 'Crop Management',
    'TCO2E_CULTIV': 'Cultivation',
    'TCO2E_FERT_PROD': 'Fertiliser production',
    'TCO2E_HARVEST': 'Harvesting',
    'TCO2E_IRRIG': 'Irrigation',
    'TCO2E_PEST_PROD': 'Pesticide production',
    'TCO2E_SOWING': 'Sowing',
    'TCO2E_ELEC': 'Electricity Use livestock',
    'TCO2E_FODDER': 'Fodder production',
    'TCO2E_FUEL': 'Fuel Use livestock',
    'TCO2E_IND_LEACH_RUNOFF': 'Agricultural soils: Indirect leaching and runoff',
    'TCO2E_MANURE_MGT': 'Livestock Manure Management (biogenic)',
    'TCO2E_SEED': 'Pasture Seed production',
    'TCO2E_SOIL': 'Agricultural soils: Direct Soil Emissions (biogenic)',
    'TCO2E_DUNG_URINE': 'Agricultural soils: Animal production, dung and urine',
    'TCO2E_ENTERIC': 'Livestock Enteric Fermentation (biogenic)',
    # Agricultural Management
    'TCO2E_Asparagopsis taxiformis': 'Asparagopsis taxiformis', 
    'TCO2E_Precision Agriculture': 'Precision Agriculture',
    'TCO2E_Ecological Grazing': 'Ecological Grazing',
    # Non-Agricultural Land-use
    'TCO2E_Agroforestry': 'Agroforestry', 
    'TCO2E_Environmental Plantings': 'Environmental Plantings',
    'TCO2E_Riparian Plantings': 'Riparian Plantings',
    'TCO2E_Carbon Plantings (Belt)': 'Carbon Plantings (Belt)',
    'TCO2E_Carbon Plantings (Block)': 'Carbon Plantings (Block)',
    'TCO2E_BECCS': 'BECCS',
    'TCO2E_Savanna Burning': 'Savanna Burning',
    'TCO2E_AgTech EI': 'AgTech EI',
}

GHG_CATEGORY = {
    'Agricultural soils: Animal production, dung and urine': {"CH4":0.5,"CO2":0.5},
    'Livestock Enteric Fermentation (biogenic)':{'CH4':1},
    'Agricultural soils: Direct Soil Emissions (biogenic)':{"N2O":1},
    
    'Asparagopsis taxiformis':{'Asparagopsis taxiformis':1},
    'Precision Agriculture':{'Precision Agriculture':1},
    'Ecological Grazing':{'Ecological Grazing':1}
}




# ============================================================
# Chart-only colors (used for charts/reports, not map rendering)
# ============================================================
COLORS_PLOT = {

    # --- General ---
    'ALL':                                      "#E8E8E8",
    'Outside LUTO study area':                  "#E8E8E8",

    # --- Rank ---
    '1-10':                                     "#ff8f5e",
    '11-20':                                    "#d5e5a3",
    '>=21':                                     "#91e8e1",
    'N.A.':                                     "#E8E8E8",

    # --- Water (chart colors — distinct from map COLOR_LM) ---
    'Dryland':                                  "#f7a35c",
    'Irrigated':                                "#7cb5ec",

    # --- Land-use types ---
    'Agricultural Land-use':                    "#7EB87A",
    'Agricultural Management':                  "#D5F100",
    'Non-Agricultural Land-use':                "#8C8888",

    # --- GHG overview ---
    'Unallocated natural to modified':          "#C8A876",
    'Unallocated natural to livestock natural': "#A8C8A8",
    'Livestock natural to modified':            "#D4A0A0",
    'Off-land emissions':                       "#4ECDC4",
    'Net emissions':                            "#45B7D1",
    'GHG emission limit':                       "#2C2B2B",

    # --- Generic summary ---
    'Total':                                    "#888888",
    'Target (%)':                               "#040404",

    # --- Economics ---
    'Agricultural Land-use (revenue)':          "#7EB87A",
    'Agricultural Management (revenue)':        "#D5F100",
    'Non-Agricultural Land-use (revenue)':      "#A0A0A0",
    'Agricultural Land-use (cost)':             "#4A8040",
    'Agricultural Management (cost)':           "#A3C400",
    'Non-Agricultural Land-use (cost)':         "#707070",
    'Transition cost (Ag2Ag)':                  "#FF6B6B",
    'Transition cost (Ag2Non-Ag)':              "#FF9F43",
    'Transition cost (Non-Ag2Ag)':              "#FFC3A0",
    'Profit':                                   "#232424",

    # --- Cost breakdown ---
    'Area cost':                                '#f45b5b',
    'Fixed depreciation cost':                  '#7cb5ec',
    'Fixed labour cost':                        '#434348',
    'Fixed operating cost':                     '#90ed7d',
    'Quantity cost':                            '#f7a35c',
    'Water cost':                               '#91e8e1',

    # --- Production demand ---
    'Domestic':                                 "#7EB87A",
    'Exports':                                  "#7298C7",
    'Imports':                                  "#E79029",
    'Feed':                                     "#BEB678",

    # --- Commodity groups ---
    'Animal Products':                          "#408DD5",
    'Fruits':                                   "#A63634",
    'Grains/Legumes':                           "#A76D5F",
    'Crops':                                    "#D4BEA6",
    'Off-land Products':                        "#434348",

    # --- Commodity aggregates ---
    'Beef':                                     '#8b2205',
    'Dairy':                                    '#BED2FF',
    'Sheep':                                    '#00b0e0',
    'Cereals':                                  '#f5c13d',
    'Oilseeds':                                 '#e0d47a',
    'Hay':                                      '#C3CADA',
    'Fruit, vegetables, nuts, legumes':         '#9f2cb6',
    'Other crops':                              '#d9abe1',
    'Carbon-focussed sequestration':            '#73ff73',
    'Nature positive sequestration':            '#006f53',
    'Human-induced regeneration':               '#7ece7e',

    # --- Production aggregates ---
    'Meat':                                     '#8085e9',
    'Milk':                                     '#f15c80',
    'Wool':                                     '#e4d354',
    'Live Exports':                             '#c7a97b',
    'Crop':                                     '#2b908f',

    # --- Commodities (on-land livestock) ---
    'Beef meat':                                '#408DD5',
    'Beef live export':                         '#D8B6B4',
    'Sheep meat':                               '#FCB819',
    'Sheep live export':                        '#A88571',
    'Sheep wool':                               '#C8D3DB',

    # --- Commodities (off-land) ---
    'Aquaculture':                              '#49482A',
    'Chicken':                                  '#D6A193',
    'Eggs':                                     '#f15c80',
    'Pork':                                     '#434348',

    # --- GHG sources (agricultural land-use) ---
    'TCO2E_CHEM_APPL':                          '#8085e9',
    'TCO2E_CROP_MGT':                           '#f15c80',
    'TCO2E_CULTIV':                             '#e4d354',
    'TCO2E_FERT_PROD':                          '#2b908f',
    'TCO2E_HARVEST':                            '#f45b5b',
    'TCO2E_IRRIG':                              '#7cb5ec',
    'TCO2E_PEST_PROD':                          '#434348',
    'TCO2E_SOWING':                             '#90ed7d',
    'TCO2E_ELEC':                               '#f7a35c',
    'TCO2E_FODDER':                             '#91e8e1',
    'TCO2E_FUEL':                               '#8085e9',
    'TCO2E_IND_LEACH_RUNOFF':                   '#f15c80',
    'TCO2E_MANURE_MGT':                         '#e4d354',
    'TCO2E_SEED':                               '#2b908f',
    'TCO2E_SOIL':                               '#f45b5b',
    'TCO2E_DUNG_URINE':                         '#7cb5ec',
    'TCO2E_ENTERIC':                            '#434348',

    # --- GHG sources (agricultural management) ---
    'TCO2E_Asparagopsis taxiformis':            '#90ed7d',
    'TCO2E_Precision Agriculture':              '#f7a35c',
    'TCO2E_Ecological Grazing':                 '#91e8e1',
    'TCO2E_Savanna Burning':                    '#434348',
    'TCO2E_AgTech EI':                          '#90ed7d',

    # --- GHG sources (non-agricultural land-use) ---
    'TCO2E_Agroforestry':                       '#8085e9',
    'TCO2E_Environmental Plantings':            '#f15c80',
    'TCO2E_Riparian Plantings':                 '#e4d354',
    'TCO2E_Carbon Plantings (Belt)':            '#2b908f',
    'TCO2E_Carbon Plantings (Block)':           '#f45b5b',
    'TCO2E_BECCS':                              '#7cb5ec',
}
COLORS_PLOT = {RENAME_AM_NON_AG.get(k, k): v for k, v in COLORS_PLOT.items()}
COLORS_PLOT = {GHG_NAMES.get(k, k): v for k, v in COLORS_PLOT.items()}


# ============================================================
# Map layer color dicts  —  {lu_code: (lu_desc, hex_color)}
# These drive both pixel rendering (code_codes) and map legend display.
# ============================================================

# Agricultural land uses — lumap_colors.csv
COLOR_AG = {
     0: ('Apples',                      '#a63634'),
     1: ('Beef - modified land',        '#b09c83'),
     2: ('Beef - natural land',         '#d4bea6'),
     3: ('Citrus',                      '#e79029'),
     4: ('Cotton',                      '#c3cada'),
     5: ('Dairy - modified land',       '#9c9d13'),
     6: ('Dairy - natural land',        '#beb678'),
     7: ('Grapes',                      '#7298c7'),
     8: ('Hay',                         '#c47646'),
     9: ('Nuts',                        '#6f4328'),
    10: ('Other non-cereal crops',      '#bf8d7e'),
    11: ('Pears',                       '#ad5d44'),
    12: ('Plantation fruit',            '#a76d5f'),
    13: ('Rice',                        '#7f7969'),
    14: ('Sheep - modified land',       '#f0c662'),
    15: ('Sheep - natural land',        '#e2bd76'),
    16: ('Stone fruit',                 '#704228'),
    17: ('Sugar',                       '#bc8463'),
    18: ('Summer cereals',              '#dedfb7'),
    19: ('Summer legumes',              '#757448'),
    20: ('Summer oilseeds',             '#d8b6b4'),
    21: ('Tropical stone fruit',        '#408dd5'),
    22: ('Unallocated - modified land', "#706F6F"),
    23: ('Unallocated - natural land',  "#afb6bb"),
    24: ('Vegetables',                  '#a88571'),
    25: ('Winter cereals',              '#fcb819'),
    26: ('Winter legumes',              '#49482a'),
    27: ('Winter oilseeds',             '#d6a193'),
}


# Agricultural Management — ammap_colors.csv
COLOR_AM = {
    -1: ('Non-Agricultural Land',   '#e1e1e1'),
     0: ('No management',           '#cccccc'),
     1: ('Asparagopsis taxiformis', '#00a9e6'),
     2: ('Precision Agriculture',   '#e69800'),
     3: ('Ecological Grazing',      '#00a884'),
     4: ('Savanna Burning',         '#d69dbc'),
     5: ('AgTech EI',               '#343434'),
     6: ('Biochar',                 '#8B6914'),
     7: ('HIR - Beef',              '#68ba77'),
     8: ('HIR - Sheep',             '#4f8079'),
     9: ('Utility Solar PV',        '#FFD966'),
    10: ('Onshore Wind',            '#7EB8D4'),
}


# Non-agricultural land uses — non_ag_colors.csv
COLOR_NON_AG = {
      -1: ('Non-Agricultural Land',           '#e1e1e1'),
       0: ('Other Agricultural Land',         '#ffebbe'),
     100: ('Environmental Plantings',         '#7dba5a'),
     101: ('Riparian Plantings',              '#3a86c8'),
     102: ('Sheep Agroforestry',              '#b57fd4'),
     103: ('Beef Agroforestry',               '#e07b54'),
     104: ('Carbon Plantings (Block)',        '#2d6a4f'),
     105: ('Sheep Carbon Plantings (Belt)',   '#52b788'),
     106: ('Beef Carbon Plantings (Belt)',    '#95d5b2'),
     107: ('BECCS',                           '#888fa3'),
     108: ('Destocked - natural land',        '#c9b45a'),
}


# Grouped land-use overview map — lumap_colors_grouped.csv
# Each lu_code maps to its display group name and color.
COLOR_LUMAP = {
      -1: ('Non-Agricultural Land',       '#e1e1e1'),
      -2: ('Non-Agricultural Land',       '#e1e1e1'),
       0: ('Crops',                       '#7a8ef5'),
       3: ('Crops',                       '#7a8ef5'),
       4: ('Crops',                       '#7a8ef5'),
       7: ('Crops',                       '#7a8ef5'),
       8: ('Crops',                       '#7a8ef5'),
       9: ('Crops',                       '#7a8ef5'),
      10: ('Crops',                       '#7a8ef5'),
      11: ('Crops',                       '#7a8ef5'),
      12: ('Crops',                       '#7a8ef5'),
      13: ('Crops',                       '#7a8ef5'),
      16: ('Crops',                       '#7a8ef5'),
      17: ('Crops',                       '#7a8ef5'),
      18: ('Crops',                       '#7a8ef5'),
      19: ('Crops',                       '#7a8ef5'),
      20: ('Crops',                       '#7a8ef5'),
      21: ('Crops',                       '#7a8ef5'),
      24: ('Crops',                       '#7a8ef5'),
      25: ('Crops',                       '#7a8ef5'),
      26: ('Crops',                       '#7a8ef5'),
      27: ('Crops',                       '#7a8ef5'),
       1: ('Livestock',                   '#ffa77f'),
       2: ('Livestock',                   '#ffa77f'),
       5: ('Livestock',                   '#ffa77f'),
       6: ('Livestock',                   '#ffa77f'),
      14: ('Livestock',                   '#ffa77f'),
      15: ('Livestock',                   '#ffa77f'),
      22: ('Unallocated - modified land', '#00e6a9'),
      23: ('Unallocated - natural land',  '#ffebbe'),
     100: ('Environmental plantings',     '#778f45'),
     101: ('Riparian Plantings',          '#005ce6'),
     102: ('Sheep Agroforestry',          '#c500ff'),
     103: ('Beef Agroforestry',           '#ff0000'),
     104: ('Carbon Plantings (Block)',    '#267300'),
     105: ('Carbon Plantings (Block)',    '#267300'),
     106: ('Carbon Plantings (Block)',    '#267300'),
     107: ('BECCS',                       '#686868'),
     108: ('Destocked - natural land',    '#e0e048'),
}

# Land management — lm_colors.csv
COLOR_LM = {
    -1: ('Non-Agricultural Land', '#cccccc'),
     0: ('Dryland',               '#ffebbe'),
     1: ('Irrigated Land',        '#ff73df'),
}


# ============================================================
# Reconstruct COLORS by merging all sub-dicts.
# COLORS_PLOT is applied last so chart-specific values win on
# key conflicts (e.g. Dryland chart vs map color).
# ============================================================
def _code_to_desc_hex(code_dict, rename_map=None):
    """Return {lu_desc: hex_color} from a {lu_code: (lu_desc, hex_color)} dict."""
    if rename_map:
        return {rename_map.get(desc, desc): hex_c for _, (desc, hex_c) in code_dict.items()}
    return {desc: hex_c for _, (desc, hex_c) in code_dict.items()}


COLORS = {
    **_code_to_desc_hex(COLOR_LUMAP, RENAME_AM_NON_AG),
    **_code_to_desc_hex(COLOR_LM),
    **_code_to_desc_hex(COLOR_AG),
    **_code_to_desc_hex(COLOR_AM, RENAME_AM_NON_AG),
    **_code_to_desc_hex(COLOR_NON_AG, RENAME_AM_NON_AG),
    **COLORS_PLOT,  # chart-specific overrides win on duplicate keys
}

COLORS_FLOAT = {
    0: '#E1E1E1FF',  # no-data grey
    -1: '#00000000',  # outside study area transparent 

    # 1-50: Blue tones 
     1: '#000E2BFF',
     2: '#00122FFF',
     3: '#001433FF',
     4: '#001637FF',
     5: '#011A3FFF',
     6: '#011C43FF',
     7: '#011E47FF',
     8: '#01204BFF',
     9: '#01224FFF',
    10: '#012453FF',
    11: '#022657FF',
    12: '#02285BFF',
    13: '#022A60FF',
    14: '#022D64FF',
    15: '#032F68FF',
    16: '#03316CFF',
    17: '#043370FF',
    18: '#043574FF',
    19: '#053778FF',
    20: '#063A7DFF',
    21: '#083D81FF',
    22: '#0A4085FF',
    23: '#0C4389FF',
    24: '#0E468DFF',
    25: '#104991FF',
    26: '#134C94FF',
    27: '#164F97FF',
    28: '#19529AFF',
    29: '#1C559DFF',
    30: '#1F58A0FF',
    31: '#225BA3FF',
    32: '#255EA6FF',
    33: '#2861A9FF',
    34: '#2B64ACFF',
    35: '#2E67AFFF',
    36: '#316AB2FF',
    37: '#346DB5FF',
    38: '#3770B8FF',
    39: '#3A73BBFF',
    40: '#3D76BEFF',
    41: '#4179C1FF',
    42: '#447CC4FF',
    43: '#477FC7FF',
    44: '#4A82CAFF',
    45: '#4D85CDFF',
    46: '#5088D0FF',
    47: '#538BD3FF',
    48: '#568ED6FF',
    49: '#5A91D9FF',
    50: '#5D94DCFF',

    # 51-100: Yellow->Red tones (mapped from COLORS_FLOAT_POSITIVE keys 0->100)
    51: '#FFF6B0FF',
    52: '#FFF6B7FF',
    53: '#FFF3B0FF',
    54: '#FFF0A9FF',
    55: '#FFEEA2FF',
    56: '#FEEB9BFF',
    57: '#FEE795FF',
    58: '#FEE48DFF',
    59: '#FEE186FF',
    60: '#FEDF84FF',
    61: '#FEDC7DFF',
    62: '#FED875FF',
    63: '#FED26FFF',
    64: '#FECC68FF',
    65: '#FEC661FF',
    66: '#FEC05BFF',
    67: '#FEBC57FF',
    68: '#FEB650FF',
    69: '#FEB34EFF',
    70: '#FDB04BFF',
    71: '#FDAA48FF',
    72: '#FDA446FF',
    73: '#FD9E43FF',
    74: '#FD9941FF',
    75: '#FD933EFF',
    76: '#FC8C3BFF',
    77: '#FC8238FF',
    78: '#FC7836FF',
    79: '#FC7434FF',
    80: '#FC6E33FF',
    81: '#FC6430FF',
    82: '#FC5A2DFF',
    83: '#FC502AFF',
    84: '#F94828FF',
    85: '#F53F26FF',
    86: '#F03623FF',
    87: '#EC2D21FF',
    88: '#E8251FFF',
    89: '#E41D1CFF',
    90: '#DF171CFF',
    91: '#D9131EFF',
    92: '#D30F20FF',
    93: '#CD0B21FF',
    94: '#C60623FF',
    95: '#C00225FF',
    96: '#B90026FF',
    97: '#AF0026FF',
    98: '#A60026FF',
    99: '#960026FF',
   100: '#800026FF',
}
 
 
 
 