#!/bin/env python3
#
# settings.py - neoLUTO settings.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-08-04
# Last modified: 2021-11-26
#

import os

# ---------------------------------------------------------------------------- #
# Directories.                                                                 #
# ---------------------------------------------------------------------------- #

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
DATA_DIR = '../../data/neoluto-data/new-data-and-domain'

# ---------------------------------------------------------------------------- #
# Parameters.                                                                  #
# ---------------------------------------------------------------------------- #

# Environmental constraint settings. In general there is some sort of 'cap'
# ('hard', 'soft' or 'none') and an optional requirement to further minimise.

# Water:
WATER_CONSTRAINT_TYPE = 'hard' # or 'soft' or 'none'.
WATER_CONSTRAINT_MINIMISE = False # or True. Whether to also minimise.
WATER_CONSTRAINT_WEIGHT = 1.0 # Minimisation weight in objective function.
WATER_YIELD_STRESS_FRACTION = 0.4 # Water stress if yields below this fraction.

