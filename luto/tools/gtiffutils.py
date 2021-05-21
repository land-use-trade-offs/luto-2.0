#!/bin/env python3
#
# gtiffutils.py - module providing convenience methods to convert to GeoTIFF.
#
# Based on modLUTO's Spatial class, in turn based on CSIRO's ANO version of
# the LUTO model.
#
# Author: Fjalar de Haan, f.dehaan@deakin.edu.au
# Created: 2020-07-10
# Last modified: 2021-05-21
#

import os.path
import sys
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GDT_Byte, GDT_Float32

import luto.data as data

driver = gdal.GetDriverByName("GTiff")

# Some constants.
res_factor = 1.0
nodata = 255

# Set template to enable recovery of 2D.
atemplate = np.load(os.path.join(data.INPUT_DIR, 'aTemplate.npy'))
mask_all = np.where( atemplate == nodata
                   , True
                   , False ).astype(np.bool)
mask_rows = list(range( 0
                      , mask_all.shape[0]
                      , int(res_factor**0.5) ))
mask_cols = list(range( 0
                      , mask_all.shape[1]
                      , int(res_factor**0.5) ))
MASK = mask_all[mask_rows, :][:, mask_cols]

# Projection information and dimensions of grid for image creation.
cell_size = 0.01 * (res_factor**0.5)
nrows = MASK.shape[0]
ncols = MASK.shape[1]
transform = ( 113.385
            , cell_size
            , 0.0
            , -11.215
            , 0.0
            , -cell_size )

projection = ( 'GEOGCS["GDA94"'
                    ',DATUM["Geocentric_Datum_of_Australia_1994"'
                            ',SPHEROID["GRS 1980"'
                                    ',6378137'
                                    ',298.257222101'
                                    ',AUTHORITY["EPSG","7019"]]'
                            ',TOWGS84[0,0,0,0,0,0,0]'
                            ',AUTHORITY["EPSG","6283"]]'
                    ',PRIMEM["Greenwich"'
                            ',0'
                            ',AUTHORITY["EPSG","8901"]]'
                    ',UNIT["degree"'
                            ',0.0174532925199433'
                            ',AUTHORITY["EPSG","9122"]]'
                    ',AUTHORITY["EPSG","4283"]]' )


def highpos2gtiff(highpos, out_name):
    """Write an integer GeoTIFF to out_name.tif from numpy highpos."""

    ind = np.cumsum(MASK == False) - 1
    ind.shape = MASK.shape
    ind = np.where(MASK == False, highpos[ind], 255)
    name = data.OUTPUT_DIR + '/' + ('%s.tif' % out_name)

    out_dataset_n = driver.Create( name
                                 , ncols
                                 , nrows
                                 , 1
                                 , GDT_Byte
                                 , ['COMPRESS=LZW'] )
    out_dataset_n.SetGeoTransform(transform)
    out_dataset_n.SetProjection(projection)

    out_band1 = out_dataset_n.GetRasterBand(1)
    out_band1.SetNoDataValue(255)
    out_band1.WriteArray(ind,0,0)

    del out_dataset_n

def to_tiff_float32(highpos, out_name):
    """Write a float32 GeoTIFF to out_name.tif from numpy highpos."""

    ind = np.cumsum(MASK == False) - 1
    ind.shape = MASK.shape
    ind = np.where(MASK == False, highpos[ind], -3.4028235e+38)
    name = data.OUTPUT_DIR + '/' + ('%s.tif' % out_name)

    try:  # Delete if exists.
        os.remove(name)
    except:
        pass

    out_dataset_n = driver.Create(name,
                                        ncols,
                                        nrows,
                                        1,
                                        GDT_Float32,
                                        ['COMPRESS=LZW'])
    out_dataset_n.SetGeoTransform(transform)
    out_dataset_n.SetProjection(projection)

    out_band1 = out_dataset_n.GetRasterBand(1)
    out_band1.SetNoDataValue(-3.4028235e+38)
    out_band1.WriteArray(ind,0,0)

    del out_dataset_n

if __name__ == "__main__":
    highpos = np.load(sys.argv[1])
    to_tiff_int(highpos, sys.argv[2])


