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

"""
LUTO 2.0 settings.
"""


import os
import pandas as pd


############### Set some Spyder options
pd.set_option('display.width', 470)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.float_format', '{:,.4f}'.format)

# ---------------------------------------------------------------------------- #
# Directories.                                                                 #
# ---------------------------------------------------------------------------- #

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
DATA_DIR = 'input'

# ---------------------------------------------------------------------------- #
# Parameters.                                                                  #
# ---------------------------------------------------------------------------- #

# Optionally coarse-grain spatial domain (faster runs useful for testing)
RESFACTOR = 1             # set to 1 to run at full spatial resolution
# SAMPLING = 'linear'     # Converts non-nodata cells to 1D array and selects every n'th cell for modelling
SAMPLING = 'quadratic'    # Selects cell from every n x n block from 2D array (raster map) for modelling (better as more regularly spaced)

# How does the model run over time
STYLE = 'snapshot'       # runs for target year only
# STYLE = 'timeseries'   # runs each year from base year to target year

# Penalty in objective function
PENALTY = 1000000

# Set Gurobi parameters

# Select Gurobi algorithm used to solve continuous models or the initial root relaxation of a MIP model.
# Set solve method. Default is automatic. Dual simplex uses less memory.
SOLVE_METHOD = 1

""" SOLVE METHODS
 'automatic':                       -1
 'primal simplex':                   0
 'dual simplex':                     1
 'barrier':                          2
 'concurrent':                       3
 'deterministic concurrent':         4
 'deterministic concurrent simplex': 5
"""

# Print detailed output to screen
VERBOSE = 0

# Environmental constraint settings. In general there is some sort of 'cap'
# ('hard', 'soft' or 'none') and an optional requirement to further minimise.

# Water:
WATER_CONSTRAINT_TYPE = 'hard' # or 'soft' or None.
WATER_CONSTRAINT_MINIMISE = False # or True. Whether to also minimise.
WATER_CONSTRAINT_WEIGHT = 1.0 # Minimisation weight in objective function.
WATER_YIELD_STRESS_FRACTION = 0.4 # 3.0 # Water stress if yields below this fraction.
WATER_DRAINDIVS = list(range(1, 14, 1)) # List of drainage divisions to take into account e.g., [1, 2].

""" DRAINAGE DIVISIONS
 1: 'Tanami-Timor Sea Coast',
 2: 'South Western Plateau',
 3: 'South West Coast',
 4: 'Tasmania',
 5: 'South East Coast (Victoria)',
 6: 'South Australian Gulf',
 7: 'Murray-Darling Basin',
 8: 'Pilbara-Gascoyne',
 9: 'North Western Plateau',
 10: 'South East Coast (NSW)',
 11: 'Carpentaria Coast',
 12: 'Lake Eyre Basin',
 13: 'North East Coast'
"""

# Climate change assumptions. Options include '126', '245', '370', '585'
SSP = '245' # 'rcp4p5' # Representative Concentration Pathway string identifier.
RCP = 'rcp' + SSP[1] + 'p' + SSP[2]

# Economic parameters
DISCOUNT_RATE = 0.05     # 0.05 = 5% pa.
AMORTISATION_PERIOD = 30 # years


