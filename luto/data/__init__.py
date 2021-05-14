#!/bin/env python3
#
# __init__.py
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-03-22
# Last modified: 2021-05-12
#

import os.path

import pandas as pd
import numpy as np

# This declaration before imports to avoid complaints. TODO better solution.
INPUT_DIR = 'input'
# INPUT_DIR = '../../data/neoluto-cost-data'

import luto.data.spatial as spatial
import luto.data.temporal as temporal

TCOSTMATRIX = np.load(os.path.join(INPUT_DIR, 'tmatrix-audperha.npy'))

# Available fields. TODO: Include a __all__ variable for safe exporting.

# Raw (spatial-) economic data.
# fpath = os.path.join(INPUT_DIR, "col-lu-irr.feather")
fpath = os.path.join(INPUT_DIR, "col-lu-irr-plugged.feather")
RAWEC = spatial.read_data_feather(fpath)

# Boolean x_rj matrix identifying allowed land uses j for each cell r.
x_rj = spatial.wherelu(RAWEC).values.astype(np.int8)

# List of land uses - i.e. combinations of what is on the land and irrigation status.
if spatial.landuses == temporal.landuses:
    LANDUSES = spatial.landuses
else:
    print( "Data inconsistency: "
           "different land-use lists in luto.spatial and luto.temporal."
         )

# Reduced land-uses list, contains only land-uses propersans irrigation status.
RLANDUSES = [ lu.replace('_dry', '')
              for j, lu in enumerate(LANDUSES)
              if j % 2 ==0 ]

# Number of land uses.
NLUS = len(LANDUSES)

# Number of cells in spatial domain.
NCELLS = spatial.cells.shape[0]

# Yield increases.
YIELDINCREASE = temporal.yieldincrease

def init_ano(scenario_number):
    """Initialise data to an ANO scenario."""

    # Globals to (re-)initialise.
    global aer
    global scene
    global cost
    global ncells

    global AG_DRYLAND_DAMAGE
    global AG_PASTURE_DAMAGE
    global DIESEL_PRICE_PATH

    # Import other data from luto.economics
    import luto.economics as ec
    del ec.spatial
    globals().update(ec.__dict__)

    # Derive the number of cells from a typical array.
    ncells = ARRAY_BF_G.size,

    # Economic Returns and Costs.
    from luto.scenario import Scenario
    from luto.spatial import Spatial
    from luto.economics.aer import AER, Costs
    scene = Scenario(scenario_number, Spatial())
    aer = AER(scene)
    cost = Costs(scene)
    AG_DRYLAND_DAMAGE = scene.ag_dryland_damage
    AG_PASTURE_DAMAGE = scene.ag_pasture_damage
    DIESEL_PRICE_PATH = scene.diesel_price_path










