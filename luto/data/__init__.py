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


import os

import pandas as pd
import numpy as np
import rasterio
import h5py

from luto.settings import INPUT_DIR, OUTPUT_DIR, SSP, RCP, RESFACTOR
from luto.economics.quantity import lvs_veg_types
from luto.tools import lumap2l_mrj, get_production



###############################################################
# Agricultural economic data.                                                 
###############################################################

# Load the agro-economic data (constructed using dataprep.py).
AGEC_CROPS = pd.read_hdf( os.path.join(INPUT_DIR, 'agec_crops.h5') )
AGEC_LVSTK = pd.read_hdf( os.path.join(INPUT_DIR, 'agec_lvstk.h5') )

#Load greenhouse gas emissions from agriculture
AGGHG_CROPS = pd.read_hdf( os.path.join(INPUT_DIR, 'agGHG_crops.h5') )
AGGHG_LVSTK = pd.read_hdf( os.path.join(INPUT_DIR, 'agGHG_lvstk.h5') )



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
LANDUSES = pd.read_csv((os.path.join(INPUT_DIR, 'landuses.csv')), header = None)[0].to_list()

# Get number of land-uses
NLUS = len(LANDUSES)

# Construct land-use index dictionary (distinct from LU_IDs!)
LU2DESC = {i: lu for i, lu in enumerate(LANDUSES)}
LU2DESC[-1] = 'Non-agricultural land'

# Some useful sub-sets of the land uses.
LU_CROPS = [ lu for lu in LANDUSES if 'Beef' not in lu
                                  and 'Sheep' not in lu
                                  and 'Dairy' not in lu
                                  and 'Unallocated' not in lu
                                  and 'Non-agricultural' not in lu ]
LU_LVSTK = [ lu for lu in LANDUSES if 'Beef' in lu
                                   or 'Sheep' in lu
                                   or 'Dairy' in lu ]
LU_UNALL = [ lu for lu in LANDUSES if 'Unallocated' in lu ]

LU_CROPS_INDICES = [LANDUSES.index(lu) for lu in LANDUSES if lu in LU_CROPS]
LU_LVSTK_INDICES = [LANDUSES.index(lu) for lu in LANDUSES if lu in LU_LVSTK]
LU_UNALL_INDICES = [LANDUSES.index(lu) for lu in LANDUSES if lu in LU_UNALL]

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
LU2PR_DICT = {key: [key.upper()] if key in LU_CROPS else [] for key in LANDUSES}
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

LU2PR = dict2matrix(LU2PR_DICT, LANDUSES, PRODUCTS)


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



###############################################################
# Masking and spatial coarse graining.                                                 
###############################################################

# Set resfactor multiplier
RESMULT = RESFACTOR ** 2

# Mask out non-agricultural land (i.e., -1) from lumap (True means included cells. Boolean dtype.)
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
for lu in LANDUSES:
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
for lu in LANDUSES:
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
RIVREG_DICT = dict(pd.read_hdf(os.path.join(INPUT_DIR, 'rivreg_lut.h5')))   # River region ID to Name lookup table

# Drainage divisions
DRAINDIV_ID = pd.read_hdf(os.path.join(INPUT_DIR, 'draindiv_id.h5')).to_numpy() # Drainage div ID mapped.
DRAINDIV_DICT = dict(pd.read_hdf(os.path.join(INPUT_DIR, 'draindiv_lut.h5')))   # Drainage div ID to Name lookup table

# Water yields -- run off from a cell into catchment by deep-rooted and shallow-rooted vegetation type.
water_yield_base = pd.read_hdf(os.path.join( INPUT_DIR, 'water_yield_baselines.h5' ))
WATER_YIELD_BASE_DR = water_yield_base['WATER_YIELD_HIST_DR_ML_HA'].to_numpy()
WATER_YIELD_BASE_SR = water_yield_base['WATER_YIELD_HIST_SR_ML_HA'].to_numpy()

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

# Load the carbon data.
REMNANT_VEG_T_CO2_HA = pd.read_hdf( os.path.join(INPUT_DIR, 'natural_land_t_co2_ha.h5') )



###############################################################
# Climate change impact data.
###############################################################

CLIMATE_CHANGE_IMPACT = pd.read_hdf(os.path.join(INPUT_DIR, 'climate_change_impacts_' + RCP + '.h5'))



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

# Load demand deltas (multipliers on 2010 production by commodity)
DEMAND_DELTAS_C = np.load(os.path.join(INPUT_DIR, 'demand_deltas_c.npy') )



###############################################################
# Other data.
###############################################################

# Raw transition cost matrix. In AUD/ha and ordered lexicographically.
TMATRIX = np.load(os.path.join(INPUT_DIR, 'tmatrix.npy'))

# Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
EXCLUDE = np.load(os.path.join(INPUT_DIR, 'x_mrj.npy'))

    
    