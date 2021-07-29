#!/bin/env python3
#
# __init__.py
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-22
# Last modified: 2021-07-29
#

import os.path

import pandas as pd
import numpy as np

# This declaration before imports to avoid complaints. TODO better solution.
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

# ------------- #
# Spatial data. #
# ------------- #

import luto.data.spatial as spatial
from luto.data.economic import exclude
# ------------------- #
# Agro-economic data. #
# ------------------- #

# Raw (spatial-) economic data.
fpath = os.path.join(INPUT_DIR, "col-lu-irr-plugged.feather")
RAWEC = spatial.read_data_feather(fpath)

fpath = os.path.join(INPUT_DIR, "agec-c9.hdf5")
AGEC = pd.read_hdf(fpath, 'agec')

# Actual hectares per cell, including projection corrections.
REAL_AREA = np.load(os.path.join(INPUT_DIR, 'realArea.npy'))

# ---------------------------------- #
# Temporal and spatio-temporal data. #
# ---------------------------------- #

import luto.data.temporal as temporal

# Yield increases.
fpath = os.path.join(INPUT_DIR, "yieldincreases-c9.hdf5")
YIELDINCREASE = pd.read_hdf(fpath, 'yieldincreases')

# Legacy
YIELDINCREASES = temporal.yieldincrease

# Climate-change damage bricks (NYEARS x NCELLS).
AG_DRYLAND_DAMAGE = temporal.ag_dryland_damage
AG_PASTURE_DAMAGE = temporal.ag_pasture_damage

# Price paths.
DIESEL_PRICE_PATH = temporal.diesel_price_path

# --------------- #
# All other data. #
# --------------- #

# Raw transition cost matrix.
TCOSTMATRIX = np.load(os.path.join(INPUT_DIR, 'tmatrix-audperha.npy'))

# Raw transition cost matrix. In AUD/ha and ordered lexicographically.
fpath = os.path.join(INPUT_DIR, 'tmatrix.csv')
TMATRIX = pd.read_csv(fpath, index_col=0)
TMATRIX = TMATRIX.sort_index(axis='index').sort_index(axis='columns')

# Boolean x_rj matrix identifying allowed land uses j for each cell r.
x_rj = spatial.wherelu(RAWEC).values.astype(np.int8)

# Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
x_mrj = exclude(AGEC)

# List of land uses - i.e. land-use x irrigation status combinations.
if spatial.landuses == temporal.landuses:
    LANDUSES = spatial.landuses
else:
    print( "Data inconsistency: "
           "different land-use lists in luto.spatial and luto.temporal."
         )

# Reduced land-uses list, i.e. only land-uses proper, sans irrigation status.
RLANDUSES = [ lu.replace('_dry', '')
              for j, lu in enumerate(LANDUSES)
              if j % 2 ==0 ]

# Land-management types.
LANDMANS = [ 'dry', 'irr' ]
NLMS = len(LANDMANS)

# Number of land uses.
NLUS = len(LANDUSES)

# Number of cells in spatial domain.
NCELLS = spatial.cells.shape[0]
