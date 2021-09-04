#!/bin/env python3
#
# plotmap.py - to plot neoLUTO spatial arrays.
#
# Based on code by: Brett Bryan (b.bryan@deakin.edu.au)
# Adaptation: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-09-01
# Last modified: 2021-09-04
#

import os.path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd
import numpy as np

import rasterio

import luto.settings as settings

fpath = os.path.join(settings.INPUT_DIR, 'NLUM_2010-11_mask.tif')
with rasterio.open(fpath) as rst:
    nlum_mask = rst.read(1)
    mask2D = np.full(nlum_mask.shape, -2)
    # mask2D = np.zeros(nlum_mask.shape)
    nonzeroes = np.nonzero(nlum_mask)

def plotmap(lumap, labels=None):
    themap = mask2D.copy()
    themap[nonzeroes] = lumap

    # Land uses and their number. For colour map and legend.
    lus = np.unique(lumap)
    nlus, = lus.shape

    cmap = matplotlib.colors.ListedColormap(np.random.rand(nlus, 3))
    im = plt.imshow(themap, cmap=cmap, resample=False)

    # Build the legend.
    colours = [im.cmap(im.norm(lu)) for lu in lus]
    if labels is None:
        patches = [ mpatches.Patch( color=colours[i]
                                  , label="Land Use {lu}".format(lu=lus[i]) )
                    for i in range(len(lus)) ]
    else:
        patches = [ mpatches.Patch( color=colours[i]
                                  , label="{lu}".format(lu=labels[lus[i]]) )
                    for i in range(len(lus)) ]
    plt.legend( handles=patches
              , bbox_to_anchor=(1.05, 1)
              , loc=2
              , borderaxespad=0 )

    # Finally.
    plt.show()

