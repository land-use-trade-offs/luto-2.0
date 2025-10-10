# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import luto.settings as settings


def create_2d_map(data, map_:np.ndarray=None) -> np.ndarray:
    """
    Create a 2D map based on the given data and map.

    Args:
        data (Data): The input data used to create the map.
        map_ (np.ndarray, 1D array): The initial map to be modified.

    Returns:
        np.ndarray: The created 2D map.

    """
    
    if settings.RESFACTOR > 1:
        return get_coarse2D_map(data, map_)    
    else: 
        return get_fullres2D_map(data, map_)                       
                


def get_fullres2D_map(data, map_:np.ndarray)-> np.ndarray:
    """
    Returns the full resolution 2D map by filling the 1D `map_` to the 2D `NLUM_MASK`.

    Args:
        `data`(Data): The data object containing the NLUM_MASK and LUMAP_NO_RESFACTOR arrays.
        `map_`(np.ndarray): The 1D np.ndarray to be filled into the `NLUM_MASK` array.

    Returns:
        np.ndarray : The restored 2D full resolution land-use map.
    """
    LUMAP_FullRes_2D = np.full(data.NLUM_MASK.shape, data.NODATA).astype(np.float32) 
    # Get the full resolution LUMAP_2D_RESFACTORED at the begining year, with -1 as Non-Agricultural Land, and -9999 as NoData
    np.place(LUMAP_FullRes_2D, data.NLUM_MASK, data.LUMAP_NO_RESFACTOR) 
    # Fill the LUMAP_FullRes_2D with map_ sequencialy by the row-col order of 1s in (LUMAP_FullRes_2D >=0) 
    np.place(LUMAP_FullRes_2D, LUMAP_FullRes_2D >=0, map_)
    return LUMAP_FullRes_2D



def get_coarse2D_map(data, map_:np.ndarray)-> np.ndarray:
    """
    Generate a coarse 2D map based on the input data.

    Args:
        `data` (Data): The input data used to create the map.
        `map_` (np.ndarray): The initial 1D map used to create a 2D map.
        
    Returns
        np.ndarray: The generated coarse 2D map.

    """
    
    # Fill the 1D map to the 2D map_resfactored.
    map_resfactored = data.LUMAP_2D_RESFACTORED.copy().astype(np.float32)
    np.place(map_resfactored, (map_resfactored != data.MASK_LU_CODE) & (map_resfactored != data.NODATA), map_.astype(np.float32))                    
    return map_resfactored
    

def upsample_array(data, map_:np.ndarray, factor:int) -> np.ndarray:
    """
    Upsamples the given array based on the provided map and factor.

    Parameters
    data (object): The input data to derive the original dense_2D shape from NLUM mask.
    map_ (2D, np.ndarray): The map used for upsampling.
    factor (int): The upsampling factor.

    Returns
    np.ndarray: The upsampled array.
    """
    dense_2D_shape = data.NLUM_MASK.shape
    dense_2D_map = np.repeat(np.repeat(map_, factor, axis=0), factor, axis=1)       # Simply repeate each cell by `factor` times at every row/col direction  
    
    # Adjust the dense_2D_map size if it exceeds the original shape
    if dense_2D_map.shape[0] > dense_2D_shape[0] or dense_2D_map.shape[1] > dense_2D_shape[1]:
        dense_2D_map = dense_2D_map[:dense_2D_shape[0], :dense_2D_shape[1]]
    
    # Pad the array if necessary
    if dense_2D_map.shape[0] < dense_2D_shape[0] or dense_2D_map.shape[1] < dense_2D_shape[1]:
        pad_height = dense_2D_shape[0] - dense_2D_map.shape[0]
        pad_width = dense_2D_shape[1] - dense_2D_map.shape[1]
        dense_2D_map = np.pad(
            dense_2D_map, 
            pad_width=((0, pad_height), (0, pad_width)), 
            mode='edge'
        )   
        
    return dense_2D_map

        