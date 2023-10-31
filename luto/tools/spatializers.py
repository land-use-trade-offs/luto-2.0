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

def create_2d_map(sim, map_, filler) -> np.ndarray:
    """Recreates a full resolution 2D array from masked and resfactored 1D array"""
    
    # First convert back to full resolution 2D array if resfactor is > 1.
    if settings.RESFACTOR > 1:
        map_ = uncoursify(sim, map_)
        
    # Then put the excluded land-use and land management types back in the array.
    map_ = reconstitute(map_, sim.data.LUMASK, filler = filler)
    
    return map_



def recreate_2D_maps(sim, yr_cal):
    """Recreates a full resolution 2D array for lumap, lmmap and ammaps"""

    # Put the excluded land-use and land management types back in the array.
    lumap = create_2d_map(sim, sim.lumaps[yr_cal], filler = sim.data.MASK_LU_CODE)
    lmmap = create_2d_map(sim, sim.lmmaps[yr_cal], filler = 0)
    ammaps = {am: create_2d_map(sim, ammap, filler=0) for am, ammap in sim.ammaps[yr_cal].items()}
    
    return lumap, lmmap, ammaps




def reconstitute(map_, mask, filler = -1):
    """Return lumap, lmmap or ammap reconstituted to original 2D size and spatial domain.
       Add Non-Agricultural, Non-Environmental Plantings Land as -1"""
    
    # Check that the number of cells in input map is full resolution. If so, then reconstitute 2D array
    if map_.shape[0] == mask.sum():
        indices = np.cumsum(mask) - 1
        return np.where(mask, map_[indices], filler)
    
    # If the map is 2D but not the right shape then the map is filled out differently.
    elif map_.shape != mask.shape: # Use Uncoursify to return full size map.
        raise ValueError("Map not of right shape.")
        
    else: # If the map is the same shape as the spatial mask
        return np.where(mask, map_, filler)


def uncoursify(sim, lxmap):
    """Recreates a full resolution 2D array from resfactored low resolution 1D array"""
    
    # Arrays with all x, y -coordinates on the larger map.
    allindices = np.nonzero(sim.data.NLUM_MASK)
    
    # Arrays with x, y -coordinates on the larger map of entries in lxmap.
    knownindices = tuple(np.stack(allindices)[:, sim.data.MASK])
    
    # Instantiate an interpolation function.
    f = NearestNDInterpolator(knownindices, lxmap)
    
    # The uncoursified map is obtained by interpolating the missing values.
    return f(allindices).astype(np.int8)



def write_gtiff(array_1D, fname, nodata = -9999):
    """Write a 2D GeoTiff based on an input 1D numpy array and the study area mask."""

    # Open the file, distill the NLUM study area mask and get the meta data.
    fpath_src = os.path.join(settings.INPUT_DIR, 'NLUM_2010-11_mask.tif')
    with rasterio.open(fpath_src) as src:
        NLUM_mask = src.read(1) == 1
        meta = src.meta.copy()

    # These keys set afresh when exporting the GeoTiff.
    for key in ('dtype', 'nodata'):
        meta.pop(key)
    meta.update(compress = 'lzw', driver = 'GTiff')

    # Reconstitute the 2D map using the NLUM mask and the numpy array.
    array_2D = np.zeros(NLUM_mask.shape, dtype=np.float32) + nodata
    nonzeroes = np.nonzero(NLUM_mask)
    array_2D[nonzeroes] = array_1D

    # Write the GeoTiff.
    with rasterio.open( fname
                      , 'w+'
                      , dtype = 'float32'
                      , nodata = nodata
                      , **meta
                      ) as dst:
        dst.write_band(1, array_2D)
