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
To make 2D GeoTIFF files from 1D numpy arrays based on a 2D array study area mask.
By Fjalar de Haan and Brett Bryan (b.bryan@deakin.edu.au).
"""

import os.path
import numpy as np
try:
    import rasterio
except:
    from osgeo import gdal
    import rasterio
from scipy.interpolate import NearestNDInterpolator
import luto.settings as settings
from luto.data import Data


def create_2d_map(data: Data, map_, filler) -> np.ndarray:
    """
    Create a 2D map by converting it back to full resolution if necessary and
    putting the excluded land-use and land management types back in the array.

    Args:
        sim: The simulation object.
        map_: The input map.
        filler: The filler value for excluded land-use and land management types.

    Returns:
        np.ndarray: The created 2D map.
    """
    
    # First convert back to full resolution 2D array if resfactor is > 1.
    if settings.RESFACTOR > 1:
        map_ = uncoursify(data, map_)

    # Then put the excluded land-use and land management types back in the array.
    map_ = reconstitute(map_, data.LUMASK, filler = filler)
    
    return map_



def reconstitute(map_, mask, filler=-1):
    """
    Reconstitutes a map based on a spatial mask.

    Args:
        map_ (numpy.ndarray): The input map.
        mask (numpy.ndarray): The spatial mask.
        filler (int, optional): The value to fill in the non-masked cells. Defaults to -1.

    Returns:
        numpy.ndarray: The reconstituted map.

    Raises:
        ValueError: If the map is not of the right shape.

    """
    if map_.shape[0] == mask.sum():
        indices = np.cumsum(mask) - 1
        return np.where(mask, map_[indices], filler)
    elif map_.shape != mask.shape:
        raise ValueError("Map not of right shape.")
    else:
        return np.where(mask, map_, filler)


def uncoursify(data: Data, lxmap):
    """
    Uncoursify the map by interpolating missing values based on known indices.

    Parameters:
    sim (object): The simulation object.
    lxmap (ndarray): The map containing the values to be uncoursified.

    Returns:
    ndarray: The uncoursified map.

    """
    # Arrays with all x, y -coordinates on the larger map.
    allindices = np.nonzero(data.NLUM_MASK)
    
    # Arrays with x, y -coordinates on the larger map of entries in lxmap.
    knownindices = tuple(np.stack(allindices)[:, data.MASK])
    
    # Instantiate an interpolation function.
    f = NearestNDInterpolator(knownindices, lxmap)
    
    # The uncoursified map is obtained by interpolating the missing values.
    return f(allindices).astype(np.float16) 



def write_gtiff(array_1D, fname, nodata=-9999):
    """
    Write a GeoTIFF file from a 1D numpy array.

    Parameters:
        array_1D (numpy.ndarray): The 1D numpy array containing the data to be written.
        fname (str): The output file name.
        nodata (int or float, optional): The nodata value to be used in the GeoTIFF file. Default is -9999.

    Returns:
        None
    """

    # Open the file, distill the NLUM study area mask and get the meta data.
    fpath_src = os.path.join(settings.INPUT_DIR, 'NLUM_2010-11_mask.tif')
    with rasterio.open(fpath_src) as src:
        NLUM_mask = src.read(1) == 1
        meta = src.meta.copy()

    # These keys set afresh when exporting the GeoTiff.
    for key in ('dtype', 'nodata'):
        meta.pop(key)
    meta.update(compress='lzw', driver='GTiff')

    # Reconstitute the 2D map using the NLUM mask and the numpy array.
    array_2D = np.zeros(NLUM_mask.shape, dtype=np.float32) + nodata
    nonzeroes = np.nonzero(NLUM_mask)
    array_2D[nonzeroes] = array_1D

    # Write the GeoTiff.
    with rasterio.open(fname, 'w+', dtype='float32', nodata=nodata, **meta) as dst:
        dst.write_band(1, array_2D)
