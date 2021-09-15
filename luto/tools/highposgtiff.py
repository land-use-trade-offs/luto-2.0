#!/bin/env python3
#
# highposgtiff.py - to make GeoTIFFs from highpos files.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Based on code by: Brett Bryan (b.bryan@deakin.edu.au)
#
# Created: 2021-06-25
# Last modified: 2021-09-15
#

import os.path

import pandas as pd
import numpy as np

import rasterio
from rasterio import features
from geopandas import GeoDataFrame
from shapely.geometry import Point

import luto.settings as settings

def write_lumap_gtiff(lumap, fname, nodata=-9999):
    """Write a GeoTiff based on the input lumap and the NLUM mask."""

    # Open the file, distill the NLUM mask and get the meta data.
    fpath_src = os.path.join(settings.INPUT_DIR, 'NLUM_2010-11_mask.tif')
    with rasterio.open(fpath_src) as src:
        NLUM_mask = src.read(1) == 1
        meta = src.meta.copy()

    # These keys set afresh when exporting the GeoTiff.
    for key in ('dtype', 'nodata'):
        meta.pop(key)
    meta.update(compress = 'lzw', driver = 'GTiff')

    # Reconstitute the 2D map using the NLUM mask and the lumap array.
    array_2D = np.zeros(NLUM_mask.shape, dtype=np.float32) + nodata
    nonzeroes = np.nonzero(NLUM_mask)
    array_2D[nonzeroes] = lumap

    # Write the GeoTiff.
    with rasterio.open( fname
                      , 'w+'
                      , dtype = 'float32'
                      , nodata = nodata
                      , **meta
                      ) as dst:
        dst.write_band(1, array_2D)
