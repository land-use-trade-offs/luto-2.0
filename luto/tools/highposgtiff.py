#!/bin/env python3
#
# highposgtiff.py - to make GeoTIFFs from highpos files.
#
# Original author: Carla Archibald (c.archibald@deakin.edu.au)
# Adaptation: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-06-25
# Last modified: 2021-06-25
#

import os.path

import pandas as pd
import numpy as np

import rasterio
from rasterio import features
from geopandas import GeoDataFrame
from shapely.geometry import Point

import luto.data as data

def write_highpos_gtiff(highpos, fname):
    """Write a GeoTIFF `fname.tif` from `highpos` Numpy array.

    The following files are assumed to live in data.INPUT_DIR:

        xy-lutoid.csv    - Concordance of LUTO cell ids with X, Y coordinates.
        nlum-raster.tif  - NLUM raster.
    """

    # Convert to pandas dataframe
    highpos = np.load(highpos)
    highpos = pd.DataFrame(highpos)
    highpos.columns = ['LU_Class']

    # Create SID column which corresponds to teh LUTO cell id
    highpos['SID'] = highpos.index

    # Read in LUTO cell ID concordance
    LUTO_sid_xy = pd.read_csv(os.path.join(data.INPUT_DIR, 'xy-lutoid.csv'))

    # Join high_pos with LUTO cell ID concordance
    highpos_xy = pd.merge( left=LUTO_sid_xy
                         , right=highpos
                         , left_on='SID'
                         , right_on='SID' )

    # Convert pandas to geopandas dataframe
    geometry = [Point(xy) for xy in zip(highpos_xy.X, highpos_xy.Y)]
    df = highpos_xy.drop(['X', 'Y', 'SID'], axis=1)
    gdf = GeoDataFrame(df, crs="EPSG:4283", geometry=geometry)
    gdf.plot(column='LU_Class')

    # Set up raster meta using NLUM raster
    rst = rasterio.open(os.path.join(data.INPUT_DIR, 'nlum-raster.tif'))
    meta = rst.meta.copy()
    meta.update(compress='lzw')

    with rasterio.open(fname + '.tif', 'w+', **meta) as out:
        out_arr = out.read(1)
        # Create a generator of geom, value pairs to use in rasterizing
        shapes = ( (geom,value)
                 for geom, value in zip(gdf.geometry, gdf.LU_Class) )
        burned = features.rasterize( shapes=shapes
                                   , fill=0
                                   , out=out_arr
                                   , transform=out.transform )
        out.write_band(1, burned)




