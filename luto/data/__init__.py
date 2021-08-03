#!/bin/env python3
#
# __init__.py
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-22
# Last modified: 2021-08-03
#

import os.path

import pandas as pd
import numpy as np

from luto.data.economic import exclude

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

# Load the agro-economic data (constructed w/ fns from data.economic module).
fpath = os.path.join(INPUT_DIR, "agec-c9.hdf5")
AGEC = pd.read_hdf(fpath, 'agec')

# Derive NCELLS (number of spatial cells) from AGEC.
NCELLS, = AGEC.index.shape

# Derive LANDUSES (land-uses) set from AGEC.
LANDUSES = {t[2] for t in AGEC.columns}
NLUS = len(LANDUSES)

# Derive LANDMANS (land-managements) set from AGEC.
LANDMANS = {t[1] for t in AGEC.columns}
NLMS = len(LANDMANS)

# Actual hectares per cell, including projection corrections.
REAL_AREA = np.load(os.path.join(INPUT_DIR, 'real-area.npy'))

# ---------------------------------- #
# Temporal and spatio-temporal data. #
# ---------------------------------- #

# Yield increases.
fpath = os.path.join(INPUT_DIR, "yieldincreases-c9.hdf5")
YIELDINCREASE = pd.read_hdf(fpath, 'yieldincreases')

# Climate damages to pastures and dryland as NYEARS x NCELLS shaped bricks.
AG_PASTURE_DAMAGE = np.load(os.path.join(INPUT_DIR, 'ag-pasture-damage.npy'))
AG_DRYLAND_DAMAGE = np.load(os.path.join(INPUT_DIR, 'ag-dryland-damage.npy'))

# Price paths.
price_paths = pd.read_csv(os.path.join(INPUT_DIR, 'pricepaths.csv'))
DIESEL_PRICE_PATH = price_paths['diesel_price_path']


# --------------- #
# All other data. #
# --------------- #

# Raw transition cost matrix. In AUD/ha and ordered lexicographically.
fpath = os.path.join(INPUT_DIR, 'tmatrix.csv')
TMATRIX = pd.read_csv(fpath, index_col=0)
TMATRIX = TMATRIX.sort_index(axis='index').sort_index(axis='columns')

# Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
x_mrj = exclude(AGEC)

lulist = { 'Apples'
         , 'Citrus'
         , 'Cotton'
         , 'Grapes'
         , 'Hay'
         , 'Nuts'
         , 'Other non-cereal crops'
         , 'Pears'
         , 'Plantation fruit'
         , 'Rice'
         , 'Stone fruit'
         , 'Sugar'
         , 'Summer cereals'
         , 'Summer legumes'
         , 'Summer oilseeds'
         , 'Tropical stone fruit'
         , 'Vegetables'
         , 'Winter cereals'
         , 'Winter legumes'
         , 'Winter oilseeds'
         }
