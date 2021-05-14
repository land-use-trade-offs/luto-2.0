#!/bin/env python3
#
# temporal.py - sub module to load and prepare time-series data.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-04-16
# Last modified: 2021-04-30
#

import importlib
import os.path

import pandas as pd
import numpy as np

from luto.data import INPUT_DIR

yieldincrease = pd.read_csv(os.path.join(INPUT_DIR, 'yieldincreases.csv'))

landuses = yieldincrease.columns.to_list()[1:] # Leave out the 'YEAR' column.
