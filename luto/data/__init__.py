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

from luto.settings import INPUT_DIR, OUTPUT_DIR
import luto.settings as settings
from luto.data.economic import exclude
from luto.economics.quantity import lvs_veg_types

# Load the agro-economic data (constructed w/ fns from data.economic module).
fpath = os.path.join(INPUT_DIR, "agec-crops-c9.hdf5")
AGEC_CROPS = pd.read_hdf(fpath, 'agec_crops')
fpath = os.path.join(INPUT_DIR, "agec-lvstk-c9.hdf5")
AGEC_LVSTK = pd.read_hdf(fpath, 'agec_lvstk')

# Derive NCELLS (number of spatial cells) from AGEC.
NCELLS, = AGEC_CROPS.index.shape

# Read in lexicographically ordered list of land uses.
LANDUSES = np.load(os.path.join(INPUT_DIR, 'landuses.npy')).tolist()
LANDUSES.sort() # Ensure lexicographic order - should be superfluous.
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

# Derive LANDMANS (land-managements) from AGEC.
LANDMANS = {t[1] for t in AGEC_CROPS.columns} # Set comp., unique entries.
LANDMANS = list(LANDMANS) # Turn into list.
LANDMANS.sort() # Ensure lexicographic order.
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
NPRS = len(PRODUCTS)

# Some land-uses map to multiple products -- a dict and matrix to capture this.
# Crops land-uses and products are one-one. Livestock is more complicated.
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
    A = np.zeros((len(tolist), len(fromlist)), dtype=np.int)
    for j, jstr in enumerate(fromlist):
        for istr in d[jstr]:
            i = tolist.index(istr)
            A[i, j] = True
    return A

LU2PR = dict2matrix(LU2PR_DICT, LANDUSES, PRODUCTS)

# List of commodities. Everything lower case to avoid mistakes.
# Basically collapse 'nveg' and 'sown' products and remove duplicates.
COMMODITIES = { ( s.replace(' - NATURAL LAND', '')
                   .replace(' - MODIFIED LAND', '')
                   .lower() )
                for s in PRODUCTS }
COMMODITIES = list(COMMODITIES)
COMMODITIES.sort()
CM_CROPS = [s for s in COMMODITIES if s in [k.lower() for k in LU_CROPS]]
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

# NLUM mask.
fpath = os.path.join(INPUT_DIR, 'NLUM_2010-11_mask.tif')
with rasterio.open(fpath) as rst:
    NLUM_MASK = rst.read(1)

# Actual hectares per cell, including projection corrections.
REAL_AREA = np.load(os.path.join(INPUT_DIR, 'real-area.npy'))

# Initial (2010) land-use map.
LUMAP = np.load(os.path.join(INPUT_DIR, 'lumap.npy'))

# Initial (2010) land-man map.
LMMAP = np.load(os.path.join(INPUT_DIR, 'lmmap.npy'))

# The base year, i.e. year == 0.
ANNUM = 2010

# ----------- #
# Water data. #
# ----------- #

# Water requirements by land use -- LVSTK.
aq_req_lvstk_dry = pd.DataFrame()
aq_req_lvstk_irr = pd.DataFrame()

# The rj-indexed arrays have zeroes where j is not livestock.
for lu in LANDUSES:
    if lu in LU_LVSTK:
        # First find out which animal is involved.
        animal, _ = lvs_veg_types(lu)
        # Water requirements per head are for drinking and irrigation.
        aq_req_lvstk_dry[lu] = AGEC_LVSTK['WR_DRN', animal]
        aq_req_lvstk_irr[lu] = ( AGEC_LVSTK['WR_DRN', animal]
                               + AGEC_LVSTK['WR_IRR', animal] )
    else:
        aq_req_lvstk_dry[lu] = 0.0
        aq_req_lvstk_irr[lu] = 0.0

# Water requirements by land use -- CROPS.
aq_req_crops_irr = pd.DataFrame()

# The rj-indexed arrays have zeroes where j is not a crop.
for lu in LANDUSES:
    if lu in LU_CROPS:
        aq_req_crops_irr[lu] = AGEC_CROPS['WR', 'irr', lu]
    else:
        aq_req_crops_irr[lu] = 0.0

AQ_REQ_CROPS_DRY_RJ = np.zeros((NCELLS, NLUS), dtype=np.int8)
AQ_REQ_CROPS_IRR_RJ = aq_req_crops_irr.to_numpy()
AQ_REQ_LVSTK_DRY_RJ = aq_req_lvstk_dry.to_numpy()
AQ_REQ_LVSTK_IRR_RJ = aq_req_lvstk_irr.to_numpy()

# Spatially explicit costs of a water licence per Ml.
WATER_LICENCE_PRICE = np.load(os.path.join( INPUT_DIR
                                           , 'water-licence-price.npy') )

# Spatially explicit costs of water delivery per Ml.
WATER_DELIVERY_PRICE = np.load(os.path.join( INPUT_DIR
                                           , 'water-delivery-price.npy') )

# River regions.
rivregs = pd.read_hdf(os.path.join(INPUT_DIR, 'rivregs.hdf5'))
RIVREGS = rivregs['HR_RIVREG_ID'].to_numpy() # River region ids as integers.
RIVREG_DICT = dict( rivregs.groupby('HR_RIVREG_ID')
                          .first()['HR_RIVREG_NAME'] )

# Drainage divisions
draindivs = pd.read_hdf(os.path.join(INPUT_DIR, 'draindivs.hdf5'))
DRAINDIVS = draindivs['HR_DRAINDIV_ID'].to_numpy() # Drainage div ids as ints.
DRAINDIV_DICT = dict( draindivs.groupby('HR_DRAINDIV_ID')
                              .first()['HR_DRAINDIV_NAME'] )

# Water yields -- run off from a cell into catchment by vegetation type.
water_yield_base = pd.read_hdf(os.path.join( INPUT_DIR
                                           , 'water-yield-baselines.hdf5' ))
WATER_YIELD_BASE_DR = water_yield_base['WATER_YIELD_DR_ML_HA'].to_numpy()
WATER_YIELD_BASE_SR = water_yield_base['WATER_YIELD_SR_ML_HA'].to_numpy()

fname_dr = os.path.join( INPUT_DIR
         , 'Water_yield_GCM-Ensemble_ssp245_2010-2100_DR_ML_HA_mean.h5' )
fname_sr = os.path.join( INPUT_DIR
         , 'Water_yield_GCM-Ensemble_ssp245_2010-2100_SR_ML_HA_mean.h5' )

wy_dr_file = h5py.File(fname_dr, 'r')
wy_sr_file = h5py.File(fname_sr, 'r')

WATER_YIELDS_DR = wy_dr_file[list(wy_dr_file.keys())[0]][:]
WATER_YIELDS_SR = wy_sr_file[list(wy_sr_file.keys())[0]][:]

# Water yields for current year -- placeholder with year zero.
WATER_YIELD_NUNC_DR = WATER_YIELDS_DR[0]
WATER_YIELD_NUNC_SR = WATER_YIELDS_SR[0]


# ---------------------------------------------------------------------------- #
# Climate change impact data.                                                  #
# ---------------------------------------------------------------------------- #

fname = 'climate-change-impacts-' + settings.RCP + '.hdf5'
fpath = os.path.join(INPUT_DIR, fname)
CLIMATE_CHANGE_IMPACT = pd.read_hdf(fpath)

# ----------------------- #
# Livestock related data. #
# ----------------------- #

FEED_REQ = np.load(os.path.join(INPUT_DIR, 'feed-req.npy'))
PASTURE_KG_DM_HA = np.load(os.path.join(INPUT_DIR, 'pasture-kg-dm-ha.npy'))
SAFE_PUR_NATL = np.load(os.path.join(INPUT_DIR, 'safe-pur-natl.npy'))
SAFE_PUR_MODL = np.load(os.path.join(INPUT_DIR, 'safe-pur-modl.npy'))

# ---------------------------------- #
# Temporal and spatio-temporal data. #
# ---------------------------------- #

# Yield increases.
fpath = os.path.join(INPUT_DIR, "yieldincreases-bau2022.csv")
YIELDINCREASE = pd.read_csv(fpath, header=[0,1])

# --------------- #
# All other data. #
# --------------- #

# Raw transition cost matrix. In AUD/ha and ordered lexicographically.
TMATRIX = np.load(os.path.join(INPUT_DIR, 'tmatrix.npy'))

# Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
EXCLUDE = np.load(os.path.join(INPUT_DIR, 'x-mrj.npy'))
