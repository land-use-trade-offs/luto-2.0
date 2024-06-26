import numpy as np
import xarray as xr
import rioxarray as rxr

from itertools import product
from joblib import Parallel, delayed
from rasterio.enums import Resampling

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.tools.spatializers import upsample_array



def bincount_avg(mask_arr, weight_arr, low_res_xr: xr.DataArray=None):
    """
    Calculate the average of weighted values based on bin counts.

    Parameters:
    - mask_arr (2D, xarray.DataArray): Array containing the mask values. 
    - weight_arr (2D, xarray.DataArray, >0 values are valid): Array containing the weight values.
    - low_res_xr (2D, xarray.DataArray): Low-resolution array containing 
        `y`, `x`, `CRS`, and `transform` to restore the bincount stats.

    Returns:
    - bin_avg (xarray.DataArray): Array containing the average values based on bin counts.
    """
    bin_sum = np.bincount(mask_arr.values.flatten(), weights=weight_arr.values.flatten(), minlength=low_res_xr.size)
    bin_occ = np.bincount(mask_arr.values.flatten(), minlength=low_res_xr.size)

    # Take values up to the last valid index, because the index of `low_res_xr.size + 1` indicates `NODATA`
    bin_sum = bin_sum[:low_res_xr.size + 1]     
    bin_occ = bin_occ[:low_res_xr.size + 1]     

    # Calculate the average value of each bin, ignoring division by zero (which will be nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_avg = (bin_sum / bin_occ).reshape(low_res_xr.shape).astype(np.float32)
        bin_avg = np.nan_to_num(bin_avg)
        
    # Restore the bincount stats to the low-resolution array    
    bin_avg = xr.DataArray(
        bin_avg, 
        dims=low_res_xr.dims, 
        coords=low_res_xr.coords)

    # Expand the dimensions of the bin_avg array to match the original weight_arr
    append_dims = {dim: weight_arr[dim] for dim in weight_arr.dims if dim not in bin_avg.dims}
    bin_avg = bin_avg.expand_dims(append_dims)
    
    # Write the CRS and transform to the output array
    bin_avg = bin_avg.rio.write_crs(low_res_xr.rio.crs)
    bin_avg = bin_avg.rio.write_transform(low_res_xr.rio.transform())
    
    return bin_avg


def get_coarse2D_map(data, map_:np.ndarray)-> np.ndarray:
    """
    Generate a coarse 2D map based on the input data.

    Args:
        data (Data): The input data used to create the map.
        map_ (np.ndarray): The initial 1D map used to create a 2D map.
        
    Returns:
        np.ndarray: The generated coarse 2D map.

    """
    
    # Fill the 1D map to the 2D map_resfactored.
    map_resfactored = data.LUMAP_2D_RESFACTORED.copy().astype(np.float32)
    np.place(map_resfactored, (map_resfactored != data.MASK_LU_CODE) & (map_resfactored != data.NODATA), map_) 
    return map_resfactored


def get_id_map_by_upsample_reproject(low_res_map, high_res_map, low_res_crs, low_res_trans):
    """
    Upsamples and reprojects a low-resolution map to match the resolution and 
    coordinate reference system (CRS) of a high-resolution map.
    
    Parameters:
        low_res_map (2D xarray.DataArray): The low-resolution map to upsample and reproject. Should at least has the affine transformation.
        high_res_map (2D xarray.DataArray): The high-resolution map to match the resolution and CRS to. Must have CRS and affine transformation.
        low_res_crs (str): The CRS of the low-resolution map.
        low_res_trans (affine.Affine): The affine transformation of the low-resolution map.
    
    Returns:
        xarray.DataArray: The upsampled and reprojected map with the same CRS and resolution as the high-resolution map.
    """
    low_res_id_map = np.arange(low_res_map.size).reshape(low_res_map.shape)
    low_res_id_map = xr.DataArray(
        low_res_id_map, 
        dims=['y', 'x'], 
        coords={'y': low_res_map.coords['y'], 'x': low_res_map.coords['x']})
    
    low_res_id_map = low_res_id_map.rio.write_crs(low_res_crs)
    low_res_id_map = low_res_id_map.rio.write_transform(low_res_trans)
    low_res_id_map = low_res_id_map.rio.reproject_match(
        high_res_map, 
        Resampling = Resampling.nearest, 
        nodata=low_res_map.size + 1).chunk('auto')
    
    return low_res_id_map



def match_lumap_biomap(
    data, 
    map_:np.ndarray, 
    res_factor:int, 
    lumap_tempelate:str='data/NLUM_2010-11_mask.tif', 
    biomap_tempelate:str='data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif')-> xr.DataArray:
    """
    Matches the map_ to the same projection and resolution as of biomap templates.
    
    If the map_ is not in the same resolution as the biomap, it will be upsampled to (1km x 1km) first.
    
    The resampling method is set to average, because the map_ is assumed to be the decision variables (float), 
    representing the percentage of a given land-use within the cell.

    Parameters:
    - data (Data): The data object containing necessary information.
    - map_ (1D, np.ndarray): The map to be matched.
    - res_factor (int): The resolution factor.
    - lumap_tempelate (str): The path to the lumap template file. Default is 'data/NLUM_2010-11_mask.tif'.
    - biomap_tempelate (str): The path to the biomap template file. Default is 'data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif'.

    Returns:
    - xr.DataArray: The matched map.

    """
    
    NLUM = rxr.open_rasterio(lumap_tempelate, chunks='auto').squeeze('band').drop_vars('band')
    bio_map = rxr.open_rasterio(biomap_tempelate, chunks='auto').squeeze('band').drop_vars('band')
    bio_map = bio_map.rio.write_crs(NLUM.rio.crs)
    
    if res_factor > 1:   
        map_ = get_coarse2D_map(data, map_)
        map_ = upsample_array(data, map_, res_factor)
    else:
        empty_map = np.full(data.NLUM_MASK.shape, data.NODATA).astype(np.float32) 
        np.place(empty_map, data.NLUM_MASK, data.LUMAP_NO_RESFACTOR) 
        np.place(empty_map, empty_map >=0, map_)
        map_ = empty_map
        
    map_ = xr.DataArray(map_, dims=('y','x'), coords={'y': NLUM['y'], 'x': NLUM['x']})
    map_ = map_.where(map_>=0, 0)
    map_ = map_.rio.write_crs(NLUM.rio.crs)
    map_ = map_.rio.write_transform(NLUM.rio.transform())  
    
    bio_id_map = get_id_map_by_upsample_reproject(bio_map, NLUM, NLUM.rio.crs, bio_map.rio.transform())
    map_ = bincount_avg(bio_id_map, map_,  bio_map)
    map_ = map_.where(map_ != map_.rio.nodata, 0)
    return map_


def ag_dvar_to_bio_map(data, ag_dvar, res_factor, max_workers):
    """
    Reprojects and matches agricultural land cover variables to biodiversity maps.

    Parameters:
    - data (Data object): The Data class object of LUTO.
    - ag_dvar (xarray.Dataset): The agricultural land cover variables.
    - res_factor (int): The resolution factor for matching.
    - max_workers (int): The maximum number of parallel workers.

    Returns:
    - xarray.Dataset: The combined dataset of reprojected and matched variables.
    """

    # wrapper function for parallel processing
    def reproject_match_dvar(ag_dvar, lm, lu, res_factor):
        map_ = ag_dvar.sel(lm=lm, lu=lu)
        map_ = match_lumap_biomap(data, map_, res_factor)
        map_ = map_.expand_dims({'lm': [lm], 'lu': [lu]})
        return map_

    tasks = [delayed(reproject_match_dvar)(ag_dvar, lm, lu, res_factor) 
            for lm,lu in product(ag_dvar['lm'].values, ag_dvar['lu'].values)]

    para_obj = Parallel(n_jobs=min(len(tasks), max_workers), return_as='generator')
    return xr.combine_by_coords([i for i in para_obj(tasks)])


def am_dvar_to_bio_map(data, am_dvar, res_factor, max_workers):
    """
    Reprojects and matches agricultural land cover variables to biodiversity maps.

    Parameters:
    - data (Data object): The Data class object of LUTO.
    - am_dvar (xarray.Dataset): The agricultural land cover variables.
    - res_factor (int): The resolution factor for matching.
    - max_workers (int): The maximum number of parallel workers.

    Returns:
    - xarray.Dataset: The combined dataset of reprojected and matched variables.
    """

    # wrapper function for parallel processing
    def reproject_match_dvar(am_dvar, am, lm, lu, res_factor):
        map_ = am_dvar.sel(lm=lm, lu=lu)
        map_ = match_lumap_biomap(data, map_, res_factor)
        map_ = map_.expand_dims({'am':[am], 'lm': [lm], 'lu': [lu]})
        return map_

    tasks = [delayed(reproject_match_dvar)(am_dvar, am, lm, lu, res_factor) 
            for am,lm,lu in product(am_dvar['am'].values, am_dvar['lm'].values, am_dvar['lu'].values)]

    para_obj = Parallel(n_jobs=min(len(tasks), max_workers), return_as='generator')
    return xr.combine_by_coords([i for i in para_obj(tasks)]).reindex(am=AG_MANAGEMENTS_TO_LAND_USES.keys())



def non_ag_dvar_to_bio_map(data, non_ag_dvar, res_factor, max_workers):
    """
    Reprojects and matches agricultural land cover variables to biodiversity maps.

    Parameters:
    - data (Data object): The Data class object of LUTO.
    - non_ag_dvar (xarray.Dataset): The agricultural land cover variables.
    - res_factor (int): The resolution factor for matching.
    - max_workers (int): The maximum number of parallel workers.

    Returns:
    - xarray.Dataset: The combined dataset of reprojected and matched variables.
    """

    # wrapper function for parallel processing
    def reproject_match_dvar(non_ag_dvar, lu, res_factor):
        map_ = non_ag_dvar.sel(lu=lu)
        map_ = match_lumap_biomap(data, map_, res_factor)
        map_ = map_.expand_dims({'lu': [lu]})
        return map_

    tasks = [delayed(reproject_match_dvar)(non_ag_dvar, lu, res_factor) 
            for lu in non_ag_dvar['lu'].values]

    para_obj = Parallel(n_jobs=min(len(tasks), max_workers), return_as='generator')
    return xr.combine_by_coords([i for i in para_obj(tasks)])
