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
To make GeoTIFFs from highpos files.
"""


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
