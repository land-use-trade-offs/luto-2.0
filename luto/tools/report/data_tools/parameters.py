
# Get the root directory of the data
YR_BASE = 2010

# Define crop-lvstk land uses
LU_CROPS = ['Apples','Citrus','Cotton','Grapes','Hay','Nuts','Other non-cereal crops',
            'Pears','Plantation fruit','Rice','Stone fruit','Sugar','Summer cereals',
            'Summer legumes','Summer oilseeds','Tropical stone fruit','Vegetables',
            'Winter cereals','Winter legumes','Winter oilseeds']

LU_LVSTKS = ['Beef - modified land','Beef - natural land','Dairy - modified land',
             'Dairy - natural land','Sheep - modified land','Sheep - natural land']

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

# Read the land uses from the file
with open('input/ag_landuses.csv') as f:
    AG_LANDUSE = [line.strip() for line in f]

with open('input/non_ag_landuses.csv') as f:
    NON_AG_LANDUSE = [line.strip() for line in f]

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
}

GHG_CATEGORY = {'Agricultural soils: Animal production, dung and urine': {"CH4":0.5,"CO2":0.5},
                'Livestock Enteric Fermentation (biogenic)':{'CH4':1},
                'Agricultural soils: Direct Soil Emissions (biogenic)':{"N2O":1},
                
                'Asparagopsis taxiformis':{'Asparagopsis taxiformis':1},
                'Precision Agriculture':{'Precision Agriculture':1},
                'Ecological Grazing':{'Ecological Grazing':1}}


