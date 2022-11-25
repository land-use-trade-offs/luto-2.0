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
To plot neoLUTO spatial arrays.

Based on code by: Brett Bryan (b.bryan@deakin.edu.au)
Adaptation: Fjalar de Haan (f.dehaan@deakin.edu.au)
Colour scheme by: Carla Archibald (c.archibald@deakin.edu.au)
"""


import os.path
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import numpy as np

import rasterio

import luto.settings as settings

colours = { 'Non-agricultural land': '#666b65'
          , 'Unallocated - natural land': '#c3cada'
          , 'Unallocated - modified land': '#a8a8b4'
          , 'Winter cereals': '#fcb819'
          , 'Summer cereals': '#f0c662'
          , 'Rice': '#7f7969'
          , 'Winter legumes': '#49482a'
          , 'Summer legumes': '#757448'
          , 'Winter oilseeds': '#d6a193'
          , 'Summer oilseeds': '#d8b6b4'
          , 'Sugar': '#bc8463'
          , 'Hay': '#c47646'
          , 'Cotton': '#c3cada'
          , 'Other non-cereal crops': '#bf8d7e'
          , 'Vegetables': '#a88571'
          , 'Citrus': '#e79029'
          , 'Apples': '#a63634'
          , 'Pears': '#ad5d44'
          , 'Stone fruit': '#704228'
          , 'Tropical stone fruit': '#408dd5'
          , 'Nuts': '#6f4328'
          , 'Plantation fruit': '#a76d5f'
          , 'Grapes': '#7298c7'
          , 'Dairy - natural land': '#beb678'
          , 'Beef - natural land': '#d4bea6'
          , 'Sheep - natural land': '#e2bd76'
          , 'Dairy - modified land': '#9c9d13'
          , 'Beef - modified land': '#b09c83'
          , 'Sheep - modified land': '#f0c662'
          , 'Water bodies, cities etc.': '#FFFFFF'
          }

id2desc = { -2: 'Water bodies, cities etc.'
          , -1: 'Non-agricultural land'
          , 0: 'Apples'
          , 1: 'Beef - modified land'
          , 2: 'Beef - natural land'
          , 3: 'Citrus'
          , 4: 'Cotton'
          , 5: 'Dairy - modified land'
          , 6: 'Dairy - natural land'
          , 7: 'Grapes'
          , 8: 'Hay'
          , 9: 'Nuts'
          , 10: 'Other non-cereal crops'
          , 11: 'Pears'
          , 12: 'Plantation fruit'
          , 13: 'Rice'
          , 14: 'Sheep - modified land'
          , 15: 'Sheep - natural land'
          , 16: 'Stone fruit'
          , 17: 'Sugar'
          , 18: 'Summer cereals'
          , 19: 'Summer legumes'
          , 20: 'Summer oilseeds'
          , 21: 'Tropical stone fruit'
          , 22: 'Unallocated - modified land'
          , 23: 'Unallocated - natural land'
          , 24: 'Vegetables'
          , 25: 'Winter cereals'
          , 26: 'Winter legumes'
          , 27: 'Winter oilseeds'
          }

clist = [colours[key] for key in list(id2desc.values())]

fpath = os.path.join(settings.INPUT_DIR, 'NLUM_2010-11_mask.tif')
with rasterio.open(fpath) as rst:
    nlum_mask = rst.read(1)
    mask2D = np.full(nlum_mask.shape, -2)
    nonzeroes = np.nonzero(nlum_mask)

def plotmap(lumap, labels=True):

    # Reconstitute the 2D array.
    themap = mask2D.copy()
    themap[nonzeroes] = lumap
    themap += 2 # Shift all lu-codes by two so the colour list starts at zero.

    # Land uses and their number. For colour maps and legends.
    lus = np.unique(themap)
    nlus, = lus.shape

    # Build the legend.
    if labels: # Use the land-use list colour scheme and labels.
        cmap = matplotlib.colors.ListedColormap(clist)
        im = plt.imshow(themap, cmap=cmap, resample=False)
        patches = [ mpatches.Patch( color=cmap(i)
                                  , label="{l} ({code})".format( l=lu
                                                               , code=i ) )
                    for i, lu in enumerate(list(id2desc.values())) ]
    else: # Use random colour scheme and labels.
        # cmap = matplotlib.colors.ListedColormap(np.random.rand(nlus, 3))
        cmap = matplotlib.colors.ListedColormap(clist)
        im = plt.imshow(themap, cmap=cmap, resample=False)
        patches = [ mpatches.Patch( color=cmap(i)
                                  , label="LU {code}".format(code=i) )
                    for i in range(len(lus)) ]

    # Attach legend to plot.
    plt.legend( handles=patches
              , loc='lower left'
              , bbox_to_anchor=(0, 0)
              , borderaxespad=0
              , ncol=3
              , fontsize='xx-small'
              , frameon=False
              )

    # Finally.
    plt.show()

def _plotmap(lumap):

    # Reconstitute the 2D array.
    themap = mask2D.copy()
    themap[nonzeroes] = lumap

    def clr(lu): return mcolors.hex2color(desc2col[id2desc[lu]])
    colourise = np.vectorize(clr)

    themapc = np.transpose(colourise(themap), (1, 2, 0))

    # Land uses and their number. For colour maps and legends.
    lus = np.unique(themap)
    nlus, = lus.shape

    # Build the legend.
    im = plt.imshow(themapc, interpolation='none')

    # Get the colours.
    colours = [im.cmap(lu) for lu in lus]

    # Make patches of each colour.
    patches = [ mpatches.Patch( color=mcolors.hex2color(desc2col[id2desc[lus[i]]])
                              , label="{lu} ({l})".format( lu=id2desc[lus[i]]
                                                         , l=lus[i] ) )
                for i in range(nlus) ]


    # Attach legend to plot.
    plt.legend( handles=patches
              , loc='lower left'
              , bbox_to_anchor=(0, 0)
              , borderaxespad=0
              , ncol=3
              , fontsize='xx-small'
              , frameon=False
              )

    # Finally.
    plt.show()

if __name__ == '__main__':
    labels = True
    lumap = np.load(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == '--random-legend':
        labels=False
    plotmap(lumap, labels=labels)
