#!/bin/env python3
#
# __init__.py
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-22
# Last modified: 2021-08-31
#

import os

import pandas as pd
import numpy as np

from luto.settings import INPUT_DIR, OUTPUT_DIR
from luto.data.economic import exclude

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

# Some useful sub-sets of the land uses.
LU_CROPS = [ lu for lu in LANDUSES if 'Beef' not in lu
                                  and 'Sheep' not in lu
                                  and 'Dairy' not in lu
                                  and 'Unallocated' not in lu
                                  and 'Non-agricultural' not in lu ]
LU_LVSTK = [ lu for lu in LANDUSES if 'Beef' in lu
                                   or 'Sheep' in lu
                                   or 'Dairy' in lu ]

# Derive LANDMANS (land-managements) from AGEC.
LANDMANS = {t[1] for t in AGEC_CROPS.columns} # Set comp., unique entries.
LANDMANS = list(LANDMANS) # Turn into list.
LANDMANS.sort() # Ensure lexicographic order.
NLMS = len(LANDMANS)

# List of products. Everything upper case to avoid mistakes.
PR_CROPS = [s.upper() for s in LU_CROPS]
PR_LVSTK = [ s.upper() + ' ' + p
             for s in LU_LVSTK if 'DAIRY' not in s.upper()
             for p in ['LIVEXPORT', 'DOMCONSUM'] ]
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

# Water requirements by land use.
wr_lvstk = pd.DataFrame()
for lu in LU_LVSTK:
    if 'beef' in lu.lower():
        animal = 'BEEF'
    if 'sheep' in lu.lower():
        animal = 'SHEEP'
    if 'dairy' in lu.lower():
        animal = 'DAIRY'
    wr_lvstk[lu] = AGEC_LVSTK['WR_IRR', animal]
wr_crops = AGEC_CROPS['WR', 'irr']
wr_rj = pd.concat([wr_crops, wr_lvstk], axis=1)
for lu in LANDUSES:
    if lu not in LU_CROPS and lu not in LU_LVSTK:
        wr_rj[lu] = 0
wr_rj.sort_index(axis=1, inplace=True)

WATER_REQUIRED = np.nan_to_num(wr_rj)

WATER_LICENCE_PRICE = np.load(os.path.join( INPUT_DIR
                                           , 'water-licence-price.npy') )
WATER_DELIVERY_PRICE = np.load(os.path.join( INPUT_DIR
                                           , 'water-delivery-price.npy') )
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
fpath = os.path.join(INPUT_DIR, "yieldincreases-c9.hdf5")
YIELDINCREASE = pd.read_hdf(fpath, 'yieldincreases')

# Climate damages to pastures and dryland as NYEARS x NCELLS shaped bricks.
# AG_PASTURE_DAMAGE = np.load(os.path.join(INPUT_DIR, 'ag-pasture-damage.npy'))
# AG_DRYLAND_DAMAGE = np.load(os.path.join(INPUT_DIR, 'ag-dryland-damage.npy'))

# Price paths.
price_paths = pd.read_csv(os.path.join(INPUT_DIR, 'pricepaths.csv'))
DIESEL_PRICE_PATH = price_paths['diesel_price_path']


# --------------- #
# All other data. #
# --------------- #

# Raw transition cost matrix. In AUD/ha and ordered lexicographically.
TMATRIX = np.load(os.path.join(INPUT_DIR, 'tmatrix.npy'))

# Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
EXCLUDE = np.load(os.path.join(INPUT_DIR, 'x-mrj.npy'))
