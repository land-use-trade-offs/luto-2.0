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
WATER_CONSTRAINT_TYPE = 'hard' # or 'soft' or None.
WATER_CONSTRAINT_MINIMISE = False # or True. Whether to also minimise.
WATER_CONSTRAINT_WEIGHT = 1.0 # Minimisation weight in objective function.
WATER_YIELD_STRESS_FRACTION = 0.4 # 3.0 # Water stress if yields below this fraction.
WATER_DRAINDIVS = ['Murray-Darling Basin'] # Drainage divisions to take into account.

# Climate change assumptions.
RCP = 'rcp4p5' # Representative Concentration Pathway string identifier.

