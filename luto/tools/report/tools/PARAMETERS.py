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



# Define the file name patterns for each category
GHG_FNAME2TYPE = {'GHG_emissions_separate_agricultural_landuse': 'Agricultural Landuse',
                  'GHG_emissions_separate_agricultural_management': 'Agricultural Management',
                  'GHG_emissions_separate_no_ag_reduction': 'Non-Agricultural Landuse',
                  'GHG_emissions_separate_transition_penalty': 'Transition Penalty'}

# Define all land uses for each category
AG_LANDUSE = ['Apples', 'Beef - modified land', 'Beef - natural land', 'Citrus', 'Cotton', 'Dairy - modified land', 
              'Dairy - natural land', 'Grapes', 'Hay', 'Nuts', 'Other non-cereal crops', 'Pears', 'Plantation fruit', 
              'Rice', 'Sheep - modified land', 'Sheep - natural land', 'Stone fruit', 'Sugar', 'Summer cereals', 
              'Summer legumes', 'Summer oilseeds', 'Tropical stone fruit', 'Unallocated - modified land', 
              'Unallocated - natural land', 'Vegetables', 'Winter cereals', 'Winter legumes', 'Winter oilseeds']

NON_AG_LANDUSE = ['Environmental Plantings']


# Define the GHG categories

GHG_NAMES = {
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

    'TCO2E_Asparagopsis taxiformis': 'Asparagopsis taxiformis', 
    'TCO2E_Precision Agriculture': 'Precision Agriculture',
    'TCO2E_Ecological Grazing': 'Ecological Grazing',
}

GHG_CATEGORY = {'Agricultural soils: Animal production, dung and urine': {"CH4":0.5,"CO2":0.5},
                'Livestock Enteric Fermentation (biogenic)':{'CH4':1},
                'Agricultural soils: Direct Soil Emissions (biogenic)':{"N2O":1},
                
                'Asparagopsis taxiformis':{'Asparagopsis taxiformis':1},
                'Precision Agriculture':{'Precision Agriculture':1},
                'Ecological Grazing':{'Ecological Grazing':1}}


# Text to look for in adding tags
NOTEBOOK_META_DICT = {
    "# HIDDEN": "remove-cell",  # Remove the whole cell
    "# NO CODE": "remove-input",  # Remove only the input
    "# HIDE CODE": "hide-input"  # Hide the input w/ a button to show
}


# The category and base corespondences for proper names
CAT2NAME = {'lumap': 'Agricultural Land-use all category',
            'lumap_separate_AgriculturalLand-use': 'Agricultural Land-use single category',
            'ammap': 'Agricultural Management',
            'lumap_separate_Non-AgriculturalLand-use': 'Non-Agricultural Land-use ',
            'lmmap': 'Dry and Irrigated Land-use'}

BASE2NAME = {'AgriculturalLand-use_00_Apples_color4band': 'Apples',
            'AgriculturalLand-use_01_Beef-modifiedland_color4band': 'Beef - modified land',
            'AgriculturalLand-use_02_Beef-naturalland_color4band': 'Beef - natural land',
            'AgriculturalLand-use_03_Citrus_color4band': 'Citrus',
            'AgriculturalLand-use_04_Cotton_color4band': 'Cotton',
            'AgriculturalLand-use_05_Dairy-modifiedland_color4band': 'Dairy - modified land',
            'AgriculturalLand-use_06_Dairy-naturalland_color4band': 'Dairy - natural land',
            'AgriculturalLand-use_07_Grapes_color4band': 'Grapes',
            'AgriculturalLand-use_08_Hay_color4band': 'Hay',
            'AgriculturalLand-use_09_Nuts_color4band': 'Nuts',
            'AgriculturalLand-use_10_Othernon-cerealcrops_color4band': 'Other non-cereal crops',
            'AgriculturalLand-use_11_Pears_color4band': 'Pears',
            'AgriculturalLand-use_12_Plantationfruit_color4band': 'Plantation fruit',
            'AgriculturalLand-use_13_Rice_color4band': 'Rice',
            'AgriculturalLand-use_14_Sheep-modifiedland_color4band': 'Sheep - modified land',
            'AgriculturalLand-use_15_Sheep-naturalland_color4band': 'Sheep - natural land',
            'AgriculturalLand-use_16_Stonefruit_color4band': 'Stone fruit',
            'AgriculturalLand-use_17_Sugar_color4band': 'Sugar',
            'AgriculturalLand-use_18_Summercereals_color4band': 'Summer cereals',
            'AgriculturalLand-use_19_Summerlegumes_color4band': 'Summer legumes',
            'AgriculturalLand-use_20_Summeroilseeds_color4band': 'Summer oilseeds',
            'AgriculturalLand-use_21_Tropicalstonefruit_color4band': 'Tropical stone fruit',
            'AgriculturalLand-use_22_Unallocated-modifiedland_color4band': 'Unallocated - modified land',
            'AgriculturalLand-use_23_Unallocated-naturalland_color4band': 'Unallocated - natural land',
            'AgriculturalLand-use_24_Vegetables_color4band': 'Vegetables',
            'AgriculturalLand-use_25_Wintercereals_color4band': 'Winter cereals',
            'AgriculturalLand-use_26_Winterlegumes_color4band': 'Winter legumes',
            'AgriculturalLand-use_27_Winteroilseeds_color4band': 'Winter oilseeds',
            
            'Non-AgriculturalLand-use_00_EnvironmentalPlantings_color4band': 'Environmental Plantings',
            
            'ammap_asparagopsis_taxiformis_color4band': 'Asparagopsis taxiformis',
            'ammap_ecological_grazing_color4band': 'Ecological Grazing',
            'ammap_precision_agriculture_color4band': 'Precision Agriculture',
            
            'lumap_color4band': 'Land-use Map',
            'lmmap_color4band': 'Dry Irriagted Land-use Map'}