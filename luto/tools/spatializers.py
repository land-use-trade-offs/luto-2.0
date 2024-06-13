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
            map_ = np.where(map_ == -9999, -1, map_)            # Replace the nodata with filler
            map_ = replace_with_nearest(map_, filler)           # Replace the filler with the nearest non-filler value
            map_ = upsample_array(data, map_, factor=settings.RESFACTOR, filler=filler, nodata=nodata)
        return map_    
    else: 
        LUMAP_FullRes_2D = np.full(data.NLUM_MASK.shape, -9999).astype(np.float32) 
        # Get the full resolution LUMAP_2D at the begining year, with -1 as Non-Agricultural Land, and -9999 as NoData
        np.place(LUMAP_FullRes_2D, data.NLUM_MASK, data.LUMAP_NO_RESFACTOR) 
        # Fill the LUMAP_FullRes_2D with map_ sequencialy by the row-col order of 1s in (LUMAP_FullRes_2D >=0) 
        np.place(LUMAP_FullRes_2D, LUMAP_FullRes_2D >=0, map_)
        return LUMAP_FullRes_2D                       
                
    



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
    
    # Fill the 1D map to the 2D map_resfactored.
    map_resfactored = data.LUMAP_2D_RESFACTORED.copy().astype(np.float32)
    np.place(map_resfactored, map_resfactored >=0, map_)    

    return map_resfactored
    



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



def replace_with_nearest(map_: np.ndarray, filler: int) -> np.ndarray:
    """
    Replaces invalid values in the input array with the nearest non-filler values.

    Parameters:
        map_ (np.ndarray, 2D): The input array.
        filler (int): The value to be considered as invalid.

    Returns:
        np.ndarray (2D): The array with invalid values replaced by the nearest non-invalid values.
    """
    # Create a mask for invalid values
    mask = (map_ == filler)
    # Perform distance transform on the mask
    _, nearest_indices = distance_transform_edt(mask, return_indices=True)
    # Replace the invalid values with the nearest non-invalid values
    map_[mask] = map_[tuple(nearest_indices[:, mask])]
    
    return map_




def write_gtiff(map_:np.ndarray, fname:str, nodata=-9999, data:Data=Data):
    """
    Write a GeoTiff file with the given map data.

    Parameters:
    map_ (np.ndarray, 2D): The map data to be written as a GeoTiff.
    fname (str): The file name (including path) of the output GeoTiff file.
    nodata (int, optional): The nodata value to be used in the GeoTiff file. Default is -9999.
    data (Data): The data object containing the GeoTiff metadata. Default is Data.

    Returns:
    None
    """
    
    # Write the GeoTiff.
    with rasterio.open(fname, 'w+', **data.GEO_META) as dst:
        dst.write_band(1, map_)


