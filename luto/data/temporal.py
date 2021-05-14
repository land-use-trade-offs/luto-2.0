#!/bin/env python3
#
# temporal.py - sub module to load/prepare time-series data, including bricks.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-04-16
# Last modified: 2021-05-14
#

import importlib
import os.path

import pandas as pd
import numpy as np

from luto.data import INPUT_DIR

# Yield increases for each land-use x irrigation status combination.
yieldincrease = pd.read_csv(os.path.join(INPUT_DIR, 'yieldincreases.csv'))
landuses = yieldincrease.columns.to_list()[1:] # Leave out the 'YEAR' column.

# Climate damages to pastures and dryland as NYEARS x NCELLS shaped bricks.
ag_pasture_damage = np.load(os.path.join(INPUT_DIR, 'ag-pasture-damage.npy'))
ag_dryland_damage = np.load(os.path.join(INPUT_DIR, 'ag-dryland-damage.npy'))

# Price paths.
price_paths = pd.read_csv(os.path.join(INPUT_DIR, 'pricepaths.csv'))
diesel_price_path = price_paths['diesel_price_path']
