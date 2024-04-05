# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
# 
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# 
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>. 


import os, time
import rasterio

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from luto.economics.agricultural.quantity import lvs_veg_types
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.non_ag_landuses import NON_AG_LAND_USES
from luto.settings import (
    INPUT_DIR, 
    RESFACTOR, 
    CO2_FERT, 
    SOC_AMORTISATION, 
    NON_AGRICULTURAL_LU_BASE_CODE, 
    RISK_OF_REVERSAL, 
    FIRE_RISK, 
    CONNECTIVITY_WEIGHTING, 
    BIODIV_TARGET, 
    SSP, 
    RCP, 
    SCENARIO, 
    DIET_DOM, 
    DIET_GLOB, 
    CONVERGENCE, 
    IMPORT_TREND, 
    WASTE, 
    FEED_EFFICIENCY, 
    GHG_LIMITS_TYPE,
    GHG_LIMITS,
    GHG_LIMITS_FIELD,
    RIPARIAN_PLANTINGS_BUFFER_WIDTH,
    RIPARIAN_PLANTINGS_TORTUOSITY_FACTOR,
    BIODIV_LIVESTOCK_IMPACT,
    LDS_BIODIVERSITY_VALUE,
    OFF_LAND_COMMODITIES,
    EGGS_AVG_WEIGHT
)

print('\n' + time.strftime("%Y-%m-%d %H:%M:%S"), 'Beginning data initialisation...')


###############################################################
# Agricultural economic data.                                                 
###############################################################

# Load the agro-economic data (constructed using dataprep.py).
AGEC_CROPS = pd.read_hdf( os.path.join(INPUT_DIR, 'agec_crops.h5') )
AGEC_LVSTK = pd.read_hdf( os.path.join(INPUT_DIR, 'agec_lvstk.h5') )

#Load greenhouse gas emissions from agriculture
AGGHG_CROPS = pd.read_hdf( os.path.join(INPUT_DIR, 'agGHG_crops.h5') )
AGGHG_LVSTK = pd.read_hdf( os.path.join(INPUT_DIR, 'agGHG_lvstk.h5') )
AGGHG_IRRPAST = pd.read_hdf( os.path.join(INPUT_DIR, 'agGHG_irrpast.h5') )

# Raw transition cost matrix. In AUD/ha and ordered lexicographically.
AG_TMATRIX = np.load(os.path.join(INPUT_DIR, 'ag_tmatrix.npy'))

# Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
EXCLUDE = np.load(os.path.join(INPUT_DIR, 'x_mrj.npy'))



###############################################################
# Miscellaneous parameters.                                                 
###############################################################

# Derive NCELLS (number of spatial cells) from AGEC.
NCELLS, = AGEC_CROPS.index.shape

# The base year, i.e. where year index yr_idx == 0.
YR_CAL_BASE = 2010



###############################################################
# Set up lists of land-uses, commodities etc. 
###############################################################

# Read in lexicographically ordered list of land-uses.
AGRICULTURAL_LANDUSES = pd.read_csv((os.path.join(INPUT_DIR, 'ag_landuses.csv')), header = None)[0].to_list()
NON_AGRICULTURAL_LANDUSES = [non_ag_land_use for non_ag_land_use in NON_AG_LAND_USES]

NONAGLU2DESC = dict(zip(range(NON_AGRICULTURAL_LU_BASE_CODE, 
                             NON_AGRICULTURAL_LU_BASE_CODE + len(NON_AGRICULTURAL_LANDUSES)),
                       NON_AGRICULTURAL_LANDUSES))

DESC2NONAGLU = {value: key for key, value in NONAGLU2DESC.items()}

# Get number of land-uses
N_AG_LUS = len(AGRICULTURAL_LANDUSES)
N_NON_AG_LUS = len(NON_AGRICULTURAL_LANDUSES)

# Construct land-use index dictionary (distinct from LU_IDs!)
AGLU2DESC = {i: lu for i, lu in enumerate(AGRICULTURAL_LANDUSES)}
DESC2AGLU = {value: key for key, value in AGLU2DESC.items()}
AGLU2DESC[-1] = 'Non-agricultural land'

# Some useful sub-sets of the land uses.
LU_CROPS = [ lu for lu in AGRICULTURAL_LANDUSES if 'Beef' not in lu
                                                and 'Sheep' not in lu
                                                and 'Dairy' not in lu
                                                and 'Unallocated' not in lu
                                                and 'Non-agricultural' not in lu ]
LU_LVSTK = [ lu for lu in AGRICULTURAL_LANDUSES if 'Beef' in lu
                                                or 'Sheep' in lu
                                                or 'Dairy' in lu ]
LU_UNALL = [ lu for lu in AGRICULTURAL_LANDUSES if 'Unallocated' in lu ]
LU_NATURAL = [
    DESC2AGLU["Beef - natural land"],
    DESC2AGLU["Dairy - natural land"],
    DESC2AGLU["Sheep - natural land"],
    DESC2AGLU["Unallocated - natural land"],
]
LU_MODIFIED_LAND = [DESC2AGLU[lu] for lu in AGRICULTURAL_LANDUSES if DESC2AGLU[lu] not in LU_NATURAL]

LU_CROPS_INDICES = [AGRICULTURAL_LANDUSES.index(lu) for lu in AGRICULTURAL_LANDUSES if lu in LU_CROPS]
LU_LVSTK_INDICES = [AGRICULTURAL_LANDUSES.index(lu) for lu in AGRICULTURAL_LANDUSES if lu in LU_LVSTK]
LU_UNALL_INDICES = [AGRICULTURAL_LANDUSES.index(lu) for lu in AGRICULTURAL_LANDUSES if lu in LU_UNALL]

NON_AG_LU_NATURAL = [ 
    DESC2NONAGLU.get("Environmental Plantings"),
    DESC2NONAGLU.get("Riparian Plantings"),
    DESC2NONAGLU.get("Agroforestry"),
    DESC2NONAGLU.get("Carbon Plantings (Block)"),
    DESC2NONAGLU.get("Carbon Plantings (Belt)"),
    DESC2NONAGLU.get("BECCS"),
]
NON_AG_LU_NATURAL = [n for n in NON_AG_LU_NATURAL if n]

# Derive land management types from AGEC.
LANDMANS = {t[1] for t in AGEC_CROPS.columns} # Set comp., unique entries.
LANDMANS = list(LANDMANS) # Turn into list.
LANDMANS.sort() # Ensure lexicographic order.

# Get number of land management types
NLMS = len(LANDMANS)

# List of products. Everything upper case to avoid mistakes.
PR_CROPS = [s.upper() for s in LU_CROPS]
PR_LVSTK = [ s.upper() + ' ' + p
             for s in LU_LVSTK if 'DAIRY' not in s.upper()
             for p in ['LEXP', 'MEAT'] ]
PR_LVSTK += [s.upper() for s in LU_LVSTK if 'DAIRY' in s.upper()]
PR_LVSTK += [s.upper() + ' WOOL' for s in LU_LVSTK if 'SHEEP' in s.upper()]
PRODUCTS = PR_CROPS + PR_LVSTK
PRODUCTS.sort() # Ensure lexicographic order.

# Get number of products
NPRS = len(PRODUCTS)


# Some land-uses map to multiple products -- a dict and matrix to capture this.
# Crops land-uses and crop products are one-one. Livestock is more complicated.
LU2PR_DICT = {key: [key.upper()] if key in LU_CROPS else [] for key in AGRICULTURAL_LANDUSES}
for lu in LU_LVSTK:
    for PR in PR_LVSTK:
        if lu.upper() in PR:
            LU2PR_DICT[lu] = LU2PR_DICT[lu] + [PR]

# A reverse dictionary for convenience.
PR2LU_DICT = {}
for key, val in LU2PR_DICT.items():
    for pr in val:
        PR2LU_DICT[pr] = key

def dict2matrix(d, fromlist, tolist):
    """Return 0-1 matrix mapping 'from-vectors' to 'to-vectors' using dict d."""
    A = np.zeros((len(tolist), len(fromlist)), dtype=np.int8)
    for j, jstr in enumerate(fromlist):
        for istr in d[jstr]:
            i = tolist.index(istr)
            A[i, j] = True
    return A

LU2PR = dict2matrix(LU2PR_DICT, AGRICULTURAL_LANDUSES, PRODUCTS)


# List of commodities. Everything lower case to avoid mistakes.
# Basically collapse 'NATURAL LAND' and 'MODIFIED LAND' products and remove duplicates.
COMMODITIES = { ( s.replace(' - NATURAL LAND', '')
                   .replace(' - MODIFIED LAND', '')
                   .lower() )
                for s in PRODUCTS }
COMMODITIES = list(COMMODITIES)
COMMODITIES.sort()
CM_CROPS = [s for s in COMMODITIES if s in [k.lower() for k in LU_CROPS]]

# Get number of commodities
NCMS = len(COMMODITIES)


# Some commodities map to multiple products -- dict and matrix to capture this.
# Crops commodities and products are one-one. Livestock is more complicated.
CM2PR_DICT = { key.lower(): [key.upper()] if key in CM_CROPS else []
               for key in COMMODITIES }
for key, value in CM2PR_DICT.items():
    if len(key.split())==1:
        head = key.split()[0]
        tail = 0
    else:
        head = key.split()[0]
        tail = key.split()[1]
    for PR in PR_LVSTK:
        if tail==0 and head.upper() in PR:
            CM2PR_DICT[key] = CM2PR_DICT[key] + [PR]
        elif (head.upper()) in PR and (tail.upper() in PR):
            CM2PR_DICT[key] = CM2PR_DICT[key] + [PR]
        else:
            ... # Do nothing, this should be a crop.

PR2CM = dict2matrix(CM2PR_DICT, COMMODITIES, PRODUCTS).T # Note the transpose.



###############################################################
# Agricultural management options data.
###############################################################

# Asparagopsis taxiformis data
asparagopsis_file = os.path.join(INPUT_DIR, '20231101_Bundle_MR.xlsx')
ASPARAGOPSIS_DATA = {}
ASPARAGOPSIS_DATA['Beef - natural land'] = pd.read_excel( asparagopsis_file, sheet_name='MR bundle (ext cattle)', index_col='Year' )
ASPARAGOPSIS_DATA['Beef - modified land'] = pd.read_excel( asparagopsis_file, sheet_name='MR bundle (int cattle)', index_col='Year' )
ASPARAGOPSIS_DATA['Sheep - natural land'] = pd.read_excel( asparagopsis_file, sheet_name='MR bundle (sheep)', index_col='Year' )
ASPARAGOPSIS_DATA['Sheep - modified land'] = ASPARAGOPSIS_DATA['Sheep - natural land']
ASPARAGOPSIS_DATA['Dairy - natural land'] = pd.read_excel( asparagopsis_file, sheet_name='MR bundle (dairy)', index_col='Year' )
ASPARAGOPSIS_DATA['Dairy - modified land'] = ASPARAGOPSIS_DATA["Dairy - natural land"]

# Precision agriculture data
prec_agr_file = os.path.join(INPUT_DIR, '20231101_Bundle_AgTech_NE.xlsx')
PRECISION_AGRICULTURE_DATA = {}
int_cropping_data = pd.read_excel( prec_agr_file, sheet_name='AgTech NE bundle (int cropping)', index_col='Year' )
cropping_data = pd.read_excel( prec_agr_file, sheet_name='AgTech NE bundle (cropping)', index_col='Year' )
horticulture_data = pd.read_excel( prec_agr_file, sheet_name='AgTech NE bundle (horticulture)', index_col='Year' )

for lu in ['Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds',
           'Winter cereals', 'Winter legumes', 'Winter oilseeds']:
    # Cropping land uses
    PRECISION_AGRICULTURE_DATA[lu] = cropping_data

for lu in ['Cotton', 'Other non-cereal crops', 'Rice', 'Sugar', 'Vegetables']:
    # Intensive Cropping land uses
    PRECISION_AGRICULTURE_DATA[lu] = int_cropping_data

for lu in ['Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears', 
           'Plantation fruit', 'Stone fruit', 'Tropical stone fruit']:
    # Horticulture land uses
    PRECISION_AGRICULTURE_DATA[lu] = horticulture_data

# Ecological grazing data
eco_grazing_file = os.path.join(INPUT_DIR, '20231107_ECOGRAZE_Bundle.xlsx')
ECOLOGICAL_GRAZING_DATA = {}
ECOLOGICAL_GRAZING_DATA['Beef - modified land'] = pd.read_excel( eco_grazing_file, sheet_name='Ecograze bundle (ext cattle)', index_col='Year' )
ECOLOGICAL_GRAZING_DATA['Sheep - modified land'] = pd.read_excel( eco_grazing_file, sheet_name='Ecograze bundle (sheep)', index_col='Year' )
ECOLOGICAL_GRAZING_DATA['Dairy - modified land'] = pd.read_excel( eco_grazing_file, sheet_name='Ecograze bundle (dairy)', index_col='Year' )

# Load soil carbon data, convert C to CO2e (x 44/12), and average over years
SOIL_CARBON_AVG_T_CO2_HA = pd.read_hdf( os.path.join(INPUT_DIR, 'soil_carbon_t_ha.h5') ).to_numpy(dtype = np.float32) * (44 / 12) / SOC_AMORTISATION



###############################################################
# Non-agricultural data.
###############################################################

# Load plantings economic data
EP_EST_COST_HA = pd.read_hdf( os.path.join(INPUT_DIR, 'ep_est_cost_ha.h5') ).to_numpy(dtype = np.float32)
CP_EST_COST_HA = pd.read_hdf( os.path.join(INPUT_DIR, 'cp_est_cost_ha.h5') ).to_numpy(dtype = np.float32)


# Load fire risk data (reduced carbon sequestration by this amount)
fr_df = pd.read_hdf( os.path.join(INPUT_DIR, 'fire_risk.h5') )
fr_dict = {'low': 'FD_RISK_PERC_5TH', 'med': 'FD_RISK_MEDIAN', 'high': 'FD_RISK_PERC_95TH'}
fire_risk = fr_df[fr_dict[FIRE_RISK]]

# Load environmental plantings (block) GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
ep_df = pd.read_hdf( os.path.join(INPUT_DIR, 'ep_block_avg_t_co2_ha_yr.h5') )
EP_BLOCK_AVG_T_CO2_HA = (
                         ( ep_df.EP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL) )
                         + ep_df.EP_BLOCK_BG_AVG_T_CO2_HA_YR
                        ).to_numpy(dtype = np.float32)

# Load environmental plantings (belt) GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
ep_df = pd.read_hdf( os.path.join(INPUT_DIR, 'ep_belt_avg_t_co2_ha_yr.h5') )
EP_BELT_AVG_T_CO2_HA = (
                         ( ep_df.EP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL) )
                         + ep_df.EP_BELT_BG_AVG_T_CO2_HA_YR
                        ).to_numpy(dtype = np.float32)

# Load environmental plantings (riparian) GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
ep_df = pd.read_hdf( os.path.join(INPUT_DIR, 'ep_rip_avg_t_co2_ha_yr.h5') )
EP_RIP_AVG_T_CO2_HA = (
                         ( ep_df.EP_RIP_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL) )
                         + ep_df.EP_RIP_BG_AVG_T_CO2_HA_YR
                        ).to_numpy(dtype = np.float32)

# Load carbon plantings (block) GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
cp_df = pd.read_hdf( os.path.join(INPUT_DIR, 'cp_block_avg_t_co2_ha_yr.h5') )
CP_BLOCK_AVG_T_CO2_HA = (
                         ( cp_df.CP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL) )
                         + cp_df.CP_BLOCK_BG_AVG_T_CO2_HA_YR
                        ).to_numpy(dtype = np.float32)

# Load farm forestry [i.e. carbon plantings (belt)] GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
cp_df = pd.read_hdf( os.path.join(INPUT_DIR, 'cp_belt_avg_t_co2_ha_yr.h5') )
CP_BELT_AVG_T_CO2_HA = (
                         ( cp_df.CP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL) )
                         + cp_df.CP_BELT_BG_AVG_T_CO2_HA_YR
                        ).to_numpy(dtype = np.float32)

# Agricultural land use to plantings raw transition costs:
AG2EP_TRANSITION_COSTS_HA = np.load( os.path.join(INPUT_DIR, 'ag_to_ep_tmatrix.npy') )  # shape: (28,)

# EP to agricultural land use transition costs:
EP2AG_TRANSITION_COSTS_HA = np.load( os.path.join(INPUT_DIR, 'ep_to_ag_tmatrix.npy') )  # shape: (28,)



###############################################################
# Spatial layers. 
###############################################################

# NLUM mask.
with rasterio.open( os.path.join(INPUT_DIR, 'NLUM_2010-11_mask.tif') ) as rst:
    NLUM_MASK = rst.read(1)

# Actual hectares per cell, including projection corrections.
REAL_AREA = pd.read_hdf(os.path.join(INPUT_DIR, 'real_area.h5')).to_numpy()

# Initial (2010) land-use map, mapped as lexicographic land-use class indices.
LUMAP = pd.read_hdf(os.path.join(INPUT_DIR, 'lumap.h5')).to_numpy()

# Initial (2010) land management map.
LMMAP = pd.read_hdf(os.path.join(INPUT_DIR, 'lmmap.h5')).to_numpy()

# Initial (2010) agricutural management maps - no cells are used for alternative agricultural management options.
# Includes a separate AM map for each agricultural management option, because they can be stacked.
AMMAP_DICT = {am: np.zeros(NCELLS).astype('int8') for am in AG_MANAGEMENTS_TO_LAND_USES}

# Load stream length data in metres of stream per cell
STREAM_LENGTH = pd.read_hdf(os.path.join(INPUT_DIR, 'stream_length_m_cell.h5')).to_numpy()

# Calculate the proportion of the area of each cell within stream buffer (convert REAL_AREA from ha to m2 and divide m2 by m2)
RP_PROPORTION = ( (2 * RIPARIAN_PLANTINGS_BUFFER_WIDTH * STREAM_LENGTH) / (REAL_AREA * 10000) ).astype(np.float32)

# Calculate the length of fencing required for each cell in per hectare terms for riparian plantings
RP_FENCING_LENGTH = ( (2 * RIPARIAN_PLANTINGS_TORTUOSITY_FACTOR * STREAM_LENGTH) / REAL_AREA ).astype(np.float32)



###############################################################
# Masking and spatial coarse graining.                                                 
###############################################################

# Set resfactor multiplier
RESMULT = RESFACTOR ** 2

# Mask out non-agricultural, non-environmental plantings land (i.e., -1) from lumap (True means included cells. Boolean dtype.)
MASK_LU_CODE = -1
LUMASK = LUMAP != MASK_LU_CODE  

# Return combined land-use and resfactor mask
if RESFACTOR > 1:
    
    # Create resfactor mask for spatial coarse-graining.
    rf_mask = NLUM_MASK.copy()
    nonzeroes = np.nonzero(rf_mask)
    rf_mask[::RESFACTOR, ::RESFACTOR] = 0
    resmask = np.where(rf_mask[nonzeroes] == 0, True, False)

    # Superimpose resfactor mask upon land-use map mask (Boolean).
    MASK = LUMASK * resmask
    
elif RESFACTOR == 1:
    MASK = LUMASK
    
else: 
    raise KeyError('RESFACTOR setting invalid')

# Create a mask indices array for subsetting arrays
MINDICES = np.where(MASK)[0].astype(np.int32)


        
###############################################################
# Water data.                                                 
###############################################################

# Water requirements by land use -- LVSTK.
wreq_lvstk_dry = pd.DataFrame()
wreq_lvstk_irr = pd.DataFrame()

# The rj-indexed arrays have zeroes where j is not livestock.
for lu in AGRICULTURAL_LANDUSES:
    if lu in LU_LVSTK:
        # First find out which animal is involved.
        animal, _ = lvs_veg_types(lu)
        # Water requirements per head are for drinking and irrigation.
        wreq_lvstk_dry[lu] = AGEC_LVSTK['WR_DRN', animal]
        wreq_lvstk_irr[lu] = ( AGEC_LVSTK['WR_DRN', animal] + AGEC_LVSTK['WR_IRR', animal] )
    else:
        wreq_lvstk_dry[lu] = 0.0
        wreq_lvstk_irr[lu] = 0.0

# Water requirements by land use -- CROPS.
wreq_crops_irr = pd.DataFrame()

# The rj-indexed arrays have zeroes where j is not a crop.
for lu in AGRICULTURAL_LANDUSES:
    if lu in LU_CROPS:
        wreq_crops_irr[lu] = AGEC_CROPS['WR', 'irr', lu]
    else:
        wreq_crops_irr[lu] = 0.0

# Add together as they have nans where not lvstk/crops
WREQ_DRY_RJ = np.nan_to_num(wreq_lvstk_dry.to_numpy(dtype = np.float32))
WREQ_IRR_RJ = np.nan_to_num(wreq_crops_irr.to_numpy(dtype = np.float32)) + \
              np.nan_to_num(wreq_lvstk_irr.to_numpy(dtype = np.float32))

# Spatially explicit costs of a water licence per ML.
WATER_LICENCE_PRICE = np.nan_to_num( pd.read_hdf(os.path.join(INPUT_DIR, 'water_licence_price.h5')).to_numpy() )

# Spatially explicit costs of water delivery per ML.
WATER_DELIVERY_PRICE = np.nan_to_num( pd.read_hdf(os.path.join(INPUT_DIR, 'water_delivery_price.h5')).to_numpy() )

# River regions.
RIVREG_ID = pd.read_hdf(os.path.join(INPUT_DIR, 'rivreg_id.h5')).to_numpy() # River region ID mapped.
rr = pd.read_hdf(os.path.join(INPUT_DIR, 'rivreg_lut.h5'))
RIVREG_DICT = dict(zip(rr.HR_RIVREG_ID, rr.HR_RIVREG_NAME))   # River region ID to Name lookup table
RIVREG_LIMITS = dict(zip(rr.HR_RIVREG_ID, rr.WATER_YIELD_HIST_BASELINE_ML))   # River region ID and water use limits

# Drainage divisions
DRAINDIV_ID = pd.read_hdf(os.path.join(INPUT_DIR, 'draindiv_id.h5')).to_numpy() # Drainage div ID mapped.
dd = pd.read_hdf(os.path.join(INPUT_DIR, 'draindiv_lut.h5'))
DRAINDIV_DICT = dict(zip(dd.HR_DRAINDIV_ID, dd.HR_DRAINDIV_NAME))   # Drainage div ID to Name lookup table
DRAINDIV_LIMITS = dict(zip(dd.HR_DRAINDIV_ID, dd.WATER_YIELD_HIST_BASELINE_ML))   # Drainage div ID and water use limits

# Water yields -- run off from a cell into catchment by deep-rooted, shallow-rooted, and natural vegetation types
water_yield_base = pd.read_hdf(os.path.join( INPUT_DIR, 'water_yield_baselines.h5' ))
# WATER_YIELD_BASE_DR = water_yield_base['WATER_YIELD_HIST_DR_ML_HA'].to_numpy(dtype = np.float32)
WATER_YIELD_BASE_SR = water_yield_base['WATER_YIELD_HIST_SR_ML_HA'].to_numpy(dtype = np.float32)
WATER_YIELD_BASE = water_yield_base['WATER_YIELD_HIST_BASELINE_ML_HA'].to_numpy(dtype = np.float32)


fname_dr = os.path.join(INPUT_DIR, 'water_yield_ssp' + SSP + '_2010-2100_dr_ml_ha.h5')
fname_sr = os.path.join(INPUT_DIR, 'water_yield_ssp' + SSP + '_2010-2100_sr_ml_ha.h5')

# wy_dr_file = h5py.File(fname_dr, 'r')
# wy_sr_file = h5py.File(fname_sr, 'r')

# Water yields for current year -- placeholder slice for year zero. ############### Better to slice off a year as the file is >2 GB  TODO
# WATER_YIELD_NUNC_DR = wy_dr_file[list(wy_dr_file.keys())[0]][0]                   # This might go in the simulation module where year is specified to save loading into memory
# WATER_YIELD_NUNC_SR = wy_sr_file[list(wy_sr_file.keys())[0]][0]



###############################################################
# Carbon sequestration by trees data.
###############################################################

# Load the remnant vegetation carbon data.
rem_veg = pd.read_hdf(os.path.join(INPUT_DIR, 'natural_land_t_co2_ha.h5')).to_numpy(dtype = np.float32)
rem_veg = np.squeeze(rem_veg) # Remove extraneous extra dimension

# Discount by fire risk.
NATURAL_LAND_T_CO2_HA = rem_veg * (fire_risk.to_numpy() / 100) 



###############################################################
# Climate change impact data.
###############################################################

CLIMATE_CHANGE_IMPACT = pd.read_hdf(os.path.join(INPUT_DIR, 'climate_change_impacts_' + RCP + '_CO2_FERT_' + CO2_FERT.upper() + '.h5'))



###############################################################
# Livestock related data.
###############################################################

FEED_REQ = np.nan_to_num( pd.read_hdf(os.path.join(INPUT_DIR, 'feed_req.h5')).to_numpy() )
PASTURE_KG_DM_HA = pd.read_hdf(os.path.join(INPUT_DIR, 'pasture_kg_dm_ha.h5')).to_numpy()
SAFE_PUR_NATL = pd.read_hdf(os.path.join(INPUT_DIR, 'safe_pur_natl.h5')).to_numpy()
SAFE_PUR_MODL = pd.read_hdf(os.path.join(INPUT_DIR, 'safe_pur_modl.h5')).to_numpy()



###############################################################
# Productivity data.
###############################################################

# Yield increases.
fpath = os.path.join(INPUT_DIR, "yieldincreases_bau2022.csv")
BAU_PROD_INCR = pd.read_csv(fpath, header = [0,1]).astype(np.float32)



###############################################################
# Demand data.
###############################################################

# Load demand data (actual production (tonnes, ML) by commodity) - from demand model     
dd = pd.read_hdf(os.path.join(INPUT_DIR, 'demand_projections.h5') )

# Select the demand data under the running scenario
DEMAND_DATA = dd.loc[(SCENARIO, 
                      DIET_DOM, 
                      DIET_GLOB, 
                      CONVERGENCE,
                      IMPORT_TREND, 
                      WASTE, 
                      FEED_EFFICIENCY)].copy()

# Convert eggs from count to tonnes
DEMAND_DATA.loc['eggs'] = DEMAND_DATA.loc['eggs'] * EGGS_AVG_WEIGHT / 1000 / 1000

# Get the off-land commodities
DEMAND_OFFLAND = DEMAND_DATA.loc[DEMAND_DATA.query("COMMODITY in @OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()

# Remove off-land commodities
DEMAND_C = DEMAND_DATA.loc[DEMAND_DATA.query("COMMODITY not in @OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()

# Convert to numpy array of shape (91, 26)
DEMAND_C = DEMAND_C.to_numpy(dtype = np.float32).T


###############################################################
# Carbon emissions from off-land commodities.
###############################################################

# Read the greenhouse gas intensity data
off_land_ghg_intensity = pd.read_csv(f'{INPUT_DIR}/agGHG_lvstk_off_land.csv')
# Split the Emission Source column into two columns
off_land_ghg_intensity[['Emission Type', 'Emission Source']] = off_land_ghg_intensity['Emission Source'].str.extract(r'^(.*?)\s*\((.*?)\)')

# Get the emissions from the off-land commodities
demand_offland_long = DEMAND_OFFLAND.stack().reset_index()
demand_offland_long = demand_offland_long.rename(columns={ 0: 'DEMAND (tonnes)'})

# Merge the demand and GHG intensity, and calculate the total GHG emissions
off_land_ghg_emissions = demand_offland_long.merge(off_land_ghg_intensity, on='COMMODITY')
off_land_ghg_emissions['Total GHG Emissions (tCO2e)'] = off_land_ghg_emissions.eval('`DEMAND (tonnes)` * `Emission Intensity [ kg CO2eq / kg ]`')

# Keep only the relevant columns
OFF_LAND_GHG_EMISSION = off_land_ghg_emissions[['YEAR',
                                                'COMMODITY',
                                                'Emission Type', 
                                                'Emission Source',
                                                'Total GHG Emissions (tCO2e)']]

# Get the GHG constraints for luto, shape is (91, 1)
OFF_LAND_GHG_EMISSION_C = OFF_LAND_GHG_EMISSION.groupby(['YEAR']).sum(numeric_only=True).values




###############################################################
# GHG targets data.
###############################################################

# If GHG_LIMITS_TYPE == 'file' then import the Excel spreadsheet and import the results to a python dictionary {year: target (tCO2e), ...}
if GHG_LIMITS_TYPE == 'file':
    GHG_TARGETS = pd.read_excel(os.path.join(INPUT_DIR, 'GHG_targets.xlsx'), sheet_name = 'Data', index_col = 'YEAR')
    GHG_TARGETS = GHG_TARGETS[GHG_LIMITS_FIELD].to_dict()

# If GHG_LIMITS_TYPE == 'dict' then import the Excel spreadsheet and import the results to a python dictionary {year: target (tCO2e), ...}
elif GHG_LIMITS_TYPE == 'dict':
    
    # Create a dictionary to hold the GHG target data
    GHG_TARGETS = {} # pd.DataFrame(columns = ['TOTAL_GHG_TCO2E'])
    
    # # Create linear function f and interpolate
    f = interp1d(list(GHG_LIMITS.keys()), list(GHG_LIMITS.values()), kind = 'linear', fill_value = 'extrapolate')
    keys = range(2010, 2101)
    for yr in range(2010, 2101):
        GHG_TARGETS[yr] = f(yr)
 
    
 
###############################################################
# Savanna burning data.
###############################################################

# Read in the dataframe
savburn_df = pd.read_hdf(os.path.join(INPUT_DIR, 'cell_savanna_burning.h5') )

# Load the columns as numpy arrays
SAVBURN_ELIGIBLE = savburn_df.ELIGIBLE_AREA.to_numpy()               # 1 = areas eligible for early dry season savanna burning under the ERF, 0 = ineligible
SAVBURN_AVEM_CH4_TCO2E_HA = savburn_df.SAV_AVEM_CH4_TCO2E_HA.to_numpy()  # Avoided emissions - methane
SAVBURN_AVEM_N2O_TCO2E_HA = savburn_df.SAV_AVEM_N2O_TCO2E_HA.to_numpy()  # Avoided emissions - nitrous oxide
SAVBURN_SEQ_CO2_TCO2E_HA = savburn_df.SAV_SEQ_CO2_TCO2E_HA.to_numpy()    # Additional carbon sequestration - carbon dioxide
SAVBURN_TOTAL_TCO2E_HA = savburn_df.AEA_TOTAL_TCO2E_HA.to_numpy()        # Total emissions abatement from EDS savanna burning

# Cost per hectare in dollars
SAVBURN_COST_HA = 2 


 
###############################################################
# Biodiversity data.
###############################################################
"""
Kunming-Montreal Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
Ensure that by 2030 at least 30 per cent of areas of degraded terrestrial, inland water, and coastal and marine ecosystems are under effective restoration, in order to enhance biodiversity and ecosystem functions and services, ecological integrity and connectivity.
"""
# Load biodiversity data
biodiv_priorities = pd.read_hdf(os.path.join(INPUT_DIR, 'biodiv_priorities.h5') )

# Get the Zonation output score between 0 and 1
BIODIV_SCORE_RAW = biodiv_priorities['BIODIV_PRIORITY_SSP' + SSP].to_numpy(dtype = np.float32)

# Get the natural area connectivity score between 0 and 1 (1 is highly connected, 0 is not connected)
conn_score = biodiv_priorities['NATURAL_AREA_CONNECTIVITY'].to_numpy(dtype = np.float32)

# Calculate weighted biodiversity score
BIODIV_SCORE_WEIGHTED = BIODIV_SCORE_RAW - (BIODIV_SCORE_RAW * (1 - conn_score) * CONNECTIVITY_WEIGHTING)

# Calculate total biodiversity target score as the quality-weighted sum of biodiv raw score over the study area 
biodiv_value_current = ( np.isin(LUMAP, 23) * BIODIV_SCORE_RAW +                                         # Biodiversity value of Unallocated - natural land 
                         np.isin(LUMAP, [2, 6, 15]) * BIODIV_SCORE_RAW * (1 - BIODIV_LIVESTOCK_IMPACT)   # Biodiversity value of livestock on natural land 
                       ) * np.where(SAVBURN_ELIGIBLE, LDS_BIODIVERSITY_VALUE, 1) * REAL_AREA             # Reduce biodiversity value of area eligible for savanna burning 

biodiv_value_target = ( ( np.isin(LUMAP, [2, 6, 15, 23]) * BIODIV_SCORE_RAW * REAL_AREA - biodiv_value_current ) +  # On natural land calculate the difference between the raw biodiversity score and the current score
                          np.isin(LUMAP, LU_MODIFIED_LAND) * BIODIV_SCORE_RAW * REAL_AREA                           # Calculate raw biodiversity score of modified land
                      ) * BIODIV_TARGET                                                                             # Multiply by biodiversity target to get the additional biodiversity score required to achieve the target
                        
# Sum the current biodiversity value and the addition biodiversity score required to meet the target
TOTAL_BIODIV_TARGET_SCORE = biodiv_value_current.sum() + biodiv_value_target.sum()                         

"""
TOTAL_BIODIV_TARGET_SCORE = ( 
                              np.isin(LUMAP, 23) * BIODIV_SCORE_RAW * REAL_AREA +                                                   # Biodiversity value of Unallocated - natural land 
                              np.isin(LUMAP, [2, 6, 15]) * BIODIV_SCORE_RAW * (1 - BIODIV_LIVESTOCK_IMPACT) * REAL_AREA +           # Biodiversity value of livestock on natural land 
                              np.isin(LUMAP, [2, 6, 15]) * BIODIV_SCORE_RAW * BIODIV_LIVESTOCK_IMPACT * BIODIV_TARGET * REAL_AREA + # Add 30% improvement to the degraded part of livestock on natural land
                              np.isin(LUMAP, LU_MODIFIED_LAND) * BIODIV_SCORE_RAW * BIODIV_TARGET * REAL_AREA                       # Add 30% improvement to modified land
                            ).sum() 
"""

 
###############################################################
# BECCS data.
###############################################################

# Load dataframe
beccs_df = pd.read_hdf(os.path.join(INPUT_DIR, 'cell_BECCS_df.h5') )

# Capture as numpy arrays
BECCS_COSTS_AUD_HA_YR = beccs_df['BECCS_COSTS_AUD_HA_YR'].to_numpy()
BECCS_REV_AUD_HA_YR = beccs_df['BECCS_REV_AUD_HA_YR'].to_numpy()
BECCS_TCO2E_HA_YR = beccs_df['BECCS_TCO2E_HA_YR'].to_numpy()
BECCS_MWH_HA_YR = beccs_df['BECCS_MWH_HA_YR'].to_numpy()

