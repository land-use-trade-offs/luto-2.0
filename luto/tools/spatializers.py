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


import os.path
import rasterio
import numpy as np
import luto.settings as settings

from affine import Affine
from luto.data import Data
from scipy.ndimage import distance_transform_edt


def create_2d_map(data: Data, map_:np.ndarray=None, filler:int=-1, nodata:int=-9999) -> np.ndarray:
    """
    Create a 2D map based on the given data and map.

    Args:
        data (Data): The input data used to create the map.
        map_ (np.ndarray, 1D array): The initial map to be modified.
        filler (int): The value used to indicate "Non-Agricultural Land".
        nodata (int): The value used to indicate "No Data".

    Returns:
        np.ndarray: The created 2D map.

    """
    
    if settings.RESFACTOR > 1:
        map_ = get_coarse2D_map(data, map_, filler, nodata)
        if settings.WRITE_FULL_RES_MAPS:
            # Fill the "Non-Agriculture land" with nearst "Ag land"
            map_ = replace_with_nearest(map_, filler)
            map_ = upsample_array(data, map_, factor=settings.RESFACTOR, filler=filler, nodata=nodata)
            
    return map_



def get_coarse2D_map(data, map_:np.ndarray, filler:int, nodata:int)-> np.ndarray:
    """
    Generate a coarse 2D map based on the input data.

    Args:
        data (Data): The input data used to create the map.
        map_ (np.ndarray, 1D array): The initial map used to create a 2D map_.
        filler (int): The value used to indicate "Non-Agricultural Land".
        nodata (int): The value used to indicate "No Data".

    Returns:
        The generated coarse 2D map.

    """
    # Get the dimensions of the dense 2D map and the coarse 2D map.
    dense_2d_height, dense_2d_width = data.NLUM_MASK.shape
    coarse_2d_height = (dense_2d_height // settings.RESFACTOR) + 1 if (dense_2d_height % settings.RESFACTOR != 0) else (dense_2d_height // settings.RESFACTOR)
    coarse_2d_width = (dense_2d_width // settings.RESFACTOR) + 1 if (dense_2d_width % settings.RESFACTOR != 0) else (dense_2d_width // settings.RESFACTOR)

    # Create the coarse 2D map.
    coarse_2d_map = np.zeros((coarse_2d_height,coarse_2d_width)) + filler
    coarse_2d_map[data.MASK_IDX_2D_SPARSE[0], data.MASK_IDX_2D_SPARSE[1]] = map_
    
    # Apply the NLUM mask to add no-data to the coarse 2D map.
    coarse_2d_NLUM_MASK = data.NLUM_MASK[::settings.RESFACTOR, ::settings.RESFACTOR]
    coarse_2d_map = np.where(coarse_2d_NLUM_MASK, coarse_2d_map, nodata)
    
    return coarse_2d_map



def upsample_array(data, map_:np.ndarray, factor:int, filler:int, nodata:int) -> np.ndarray:
    """
    Upsamples the given array based on the provided map and factor.

    Parameters:
    data (object): The input data to derive the original dense_2D shape from NLUM mask.
    map_ (2D, np.ndarray): The map used for upsampling.
    factor (int): The upsampling factor.
    filler (int): The value used to indicate "Non-Ag land".
    nondata (int): The value used to indicate "No Data".

    Returns:
    np.ndarray: The upsampled array.
    """
    dense_2D_shape = data.NLUM_MASK.shape
    dense_2D_map = np.repeat(np.repeat(map_, factor, axis=0), factor, axis=1)       # Simply repeate each cell by `factor` times at every row/col direction
    dense_2D_map = dense_2D_map[0:dense_2D_shape[0], 0:dense_2D_shape[1]]           # Make sure it has the same shape as the original full-res 2D map
    filler_mask = data.LUMAP_2D != data.MASK_LU_CODE
    dense_2D_map = np.where(filler_mask, dense_2D_map, filler)                      # Apply the LU mask to the dense 2D map.
    dense_2D_map = np.where(data.NLUM_MASK, dense_2D_map, nodata)                   # Apply the NLUM mask to the dense 2D map.
    return dense_2D_map



def replace_with_nearest(array: np.ndarray, filler: int) -> np.ndarray:
    """
    Replaces invalid values in the input array with the nearest non-filler values.

    Parameters:
        array (np.ndarray): The input array.
        filler (int): The value to be considered as invalid.

    Returns:
        np.ndarray: The array with invalid values replaced by the nearest non-invalid values.
    """
    # Create a mask for invalid values
    mask = (array == filler)
    
    # Perform distance transform on the mask
    _, nearest_indices = distance_transform_edt(mask, return_indices=True)
    
    # Replace the invalid values with the nearest non-invalid values
    array[mask] = array[tuple(nearest_indices[:, mask])]
    
    return array




def write_gtiff(map_:np.ndarray, fname:str, nodata=-9999):
    """
    Write a GeoTiff file with the given map data.

    Parameters:
    map_ (np.ndarray): The map data to be written as a GeoTiff.
    fname (str): The file name (including path) of the output GeoTiff file.
    nodata (int, optional): The nodata value to be used in the GeoTiff file. Default is -9999.

    Returns:
    None
    """

    # Open the file, distill the NLUM study area mask and get the meta data.
    fpath_src = os.path.join(settings.INPUT_DIR, 'NLUM_2010-11_mask.tif')
    
    with rasterio.open(fpath_src) as src:
        meta = src.meta.copy()
        trans = list(src.transform)
        trans[0] = trans[0] * settings.RESFACTOR if settings.WRITE_FULL_RES_MAPS else trans[0]
        trans[4] = trans[4] * settings.RESFACTOR if settings.WRITE_FULL_RES_MAPS else trans[4]
        trans = Affine(*trans)
        meta.update(compress='lzw', driver='GTiff')

    # These keys set afresh when exporting the GeoTiff.
    for key in ('dtype', 'nodata'):
        meta.pop(key)
    
    # Write the GeoTiff.
    with rasterio.open(fname, 'w+', dtype='float32', nodata=nodata, **meta) as dst:
        dst.write_band(1, map_)


# def reconstitute(map_, mask, filler=-1):
#     """
#     Reconstitutes a map based on a spatial mask.

#     Args:
#         map_ (numpy.ndarray): The input map.
#         mask (numpy.ndarray): The spatial mask.
#         filler (int, optional): The value to fill in the non-masked cells. Defaults to -1.

#     Returns:
#         numpy.ndarray: The reconstituted map.

#     Raises:
#         ValueError: If the map is not of the right shape.

#     """
#     if map_.shape[0] == mask.sum():
#         indices = np.cumsum(mask) - 1
#         return np.where(mask, map_[indices], filler)
#     elif map_.shape != mask.shape:
#         raise ValueError("Map not of right shape.")
#     else:
#         return np.where(mask, map_, filler)


# def uncoursify(data: Data, lxmap):
#     """
#     Uncoursify the map by interpolating missing values based on known indices.

#     Parameters:
#     data (object): Data object.
#     lxmap (ndarray): The map containing the values to be uncoursified.

#     Returns:
#     ndarray: The uncoursified map.

#     """
#     # Arrays with all x, y -coordinates on the larger map.
#     allindices = np.nonzero(data.NLUM_MASK)
    
#     # Arrays with x, y -coordinates on the larger map of entries in lxmap.
#     knownindices = tuple(np.stack(allindices)[:, data.MASK])
    
#     # Instantiate an interpolation function.
#     f = NearestNDInterpolator(knownindices, lxmap)
    
#     # The uncoursified map is obtained by interpolating the missing values.
#     return f(allindices)