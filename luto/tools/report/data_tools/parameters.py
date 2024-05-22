import luto.settings as settings


# Get the root directory of the data
YR_BASE = 2010

# Define crop-lvstk land uses
LU_CROPS = ['Apples','Citrus','Cotton','Grapes','Hay','Nuts','Other non-cereal crops',
            'Pears','Plantation fruit','Rice','Stone fruit','Sugar','Summer cereals',
            'Summer legumes','Summer oilseeds','Tropical stone fruit','Vegetables',
            'Winter cereals','Winter legumes','Winter oilseeds']

LVSTK_NATURAL = ['Beef - natural land','Dairy - natural land','Sheep - natural land']

LVSTK_MODIFIED = ['Beef - modified land','Dairy - modified land','Sheep - modified land']

LU_LVSTKS = LVSTK_NATURAL + LVSTK_MODIFIED

LU_UNALLOW = ['Unallocated - modified land','Unallocated - natural land']


LU_NATURAL = ['Beef - natural land',
              'Dairy - natural land',
              'Sheep - natural land',
              'Unallocated - natural land']


# Define the commodity categories
COMMODITIES_ON_LAND = ['Apples','Beef live export','Beef meat','Citrus','Cotton','Dairy','Grapes',
                       'Hay','Nuts','Other non-cereal crops', 'Pears', 'Plantation fruit',
                       'Rice', 'Sheep lexp', 'Sheep meat', 'Sheep wool', 'Stone fruit', 'Sugar',
                       'Summer cereals', 'Summer legumes', 'Summer oilseeds', 'Tropical stone fruit',
                       'Vegetables','Winter cereals','Winter legumes','Winter oilseeds']

COMMODITIES_OFF_LAND = ['Aquaculture', 'Chicken', 'Eggs', 'Pork' ]

COMMODITIES_ALL = COMMODITIES_ON_LAND + COMMODITIES_OFF_LAND



# Define the file name patterns for each category
GHG_FNAME2TYPE = {'GHG_emissions_separate_agricultural_landuse': 'Agricultural Landuse',
                  'GHG_emissions_separate_agricultural_management': 'Agricultural Management',
                  'GHG_emissions_separate_no_ag_reduction': 'Non-Agricultural Landuse',
                  'GHG_emissions_separate_transition_penalty': 'Transition Penalty',
                  'GHG_emissions_offland_commodity': 'Offland Commodity',}


AG_LANDUSE_MERGE_LANDTYPE = ['Apples', 'Beef', 'Citrus', 'Cotton', 'Dairy', 'Grapes', 'Hay', 'Nuts', 'Other non-cereal crops',
                             'Pears', 'Plantation fruit', 'Rice', 'Sheep', 'Stone fruit', 'Sugar', 'Summer cereals',
                             'Summer legumes', 'Summer oilseeds', 'Tropical stone fruit', 'Unallocated - modified land', 
                             'Unallocated - natural land', 'Vegetables', 'Winter cereals', 'Winter legumes', 'Winter oilseeds']


# Define the renaming of the Agricultural-Managment and Non-Agricultural 
RENAME_AM_NON_AG = {
    # Agricultural Management
    "Asparagopsis taxiformis": "Methane reduction (livestock)",
    "Precision Agriculture": "Agricultural technology (fertiliser)", 
    "Ecological Grazing": "Regenerative agriculture (livestock)", 
    "Savanna Burning": "Early dry-season savanna burning",
    "AgTech EI": "Agricultural technology (energy)",
    # Non-Agricultural Landuse
    "Environmental Plantings": "Environmental plantings (mixed species)",
    "Riparian Plantings": "Riparian buffer restoration (mixed species)",
    "Sheep Agroforestry": "Agroforestry (mixed species + sheep)",
    "Beef Agroforestry": "Agroforestry (mixed species + beef)",
    "Carbon Plantings (Block)": "Carbon plantings (monoculture)",
    "Sheep Carbon Plantings (Belt)": "Farm forestry (hardwood timber + sheep)",
    "Beef Carbon Plantings (Belt)": "Farm forestry (hardwood timber + beef)",
    "BECCS": "BECCS (Bioenergy with Carbon Capture and Storage)"
}

# Read the land uses from the file
with open(f'{settings.INPUT_DIR}/ag_landuses.csv') as f:
    AG_LANDUSE = [line.strip() for line in f]

NON_AG_LANDUSE = list(settings.NON_AG_LAND_USES.keys())

# Rename the land uses
NON_AG_LANDUSE = [RENAME_AM_NON_AG.get(item, item) for item in NON_AG_LANDUSE]

# Merge the land uses
LANDUSE_ALL = AG_LANDUSE + NON_AG_LANDUSE



# Define the GHG categories
GHG_NAMES = {
    # Agricultural Landuse
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
    # Non-Agricultural Landuse
    'TCO2E_Agroforestry': 'Agroforestry', 
    'TCO2E_Environmental Plantings': 'Environmental Plantings',
    'TCO2E_Riparian Plantings': 'Riparian Plantings',
    'TCO2E_Carbon Plantings (Belt)': 'Carbon Plantings (Belt)',
    'TCO2E_Carbon Plantings (Block)': 'Carbon Plantings (Block)',
    'TCO2E_BECCS': 'BECCS',
    'TCO2E_Savanna Burning': 'Savanna Burning',
    'TCO2E_AgTech EI': 'AgTech EI',
}

GHG_CATEGORY = {'Agricultural soils: Animal production, dung and urine': {"CH4":0.5,"CO2":0.5},
                'Livestock Enteric Fermentation (biogenic)':{'CH4':1},
                'Agricultural soils: Direct Soil Emissions (biogenic)':{"N2O":1},
                
                'Asparagopsis taxiformis':{'Asparagopsis taxiformis':1},
                'Precision Agriculture':{'Precision Agriculture':1},
                'Ecological Grazing':{'Ecological Grazing':1}}


