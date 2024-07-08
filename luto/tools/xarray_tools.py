import os
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sparse
import xarray as xr
import rioxarray as rxr
import geopandas as gpd

import luto.settings as settings

from itertools import product
from rasterio.enums import Resampling
from rasterio.features import shapes

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.data import Data
from luto.tools.spatializers import get_coarse2D_map, upsample_array



def ag_to_xr(data:Data, yr_cal:int):
    """
    Convert agricultural data to xarray DataArray.

    Parameters:
        data (Data): The input data object.
        yr_cal (int): The year calendar.

    Returns:
        xr.DataArray: The converted xarray DataArray.
    """

    dvar = data.ag_dvars[yr_cal]
    ag_dvar_xr = xr.DataArray(
        dvar, 
        dims=('lm', 'cell', 'lu'),
        coords={
            'lm': data.LANDMANS,
            'cell': np.arange(dvar.shape[1]),
            'lu': data.AGRICULTURAL_LANDUSES
        }   
    )
    return ag_dvar_xr.reindex(lu=data.AGRICULTURAL_LANDUSES)


def am_to_xr(data:Data, yr_cal:int):
    """
    Convert agricultural management data to xarray format.

    Parameters:
        data (Data): The input data object.
        yr_cal (int): The year calendar.

    Returns:
        xr.Dataset: The xarray dataset containing the converted data.
    """

    am_dvar = data.ag_man_dvars[yr_cal]
    
    am_dvars = []
    for am in am_dvar.keys():  
        am_dvar_xr = xr.DataArray(
            am_dvar[am], 
            dims=('lm', 'cell', 'lu'),
            coords={
                'lm': data.LANDMANS,
                'cell': np.arange(am_dvar[am].shape[1]),
                'lu': data.AGRICULTURAL_LANDUSES
            }   
        )
        am_dvar_xr = am_dvar_xr.expand_dims({'am':[am]})
        am_dvars.append(am_dvar_xr)
            
    return xr.combine_by_coords(am_dvars).reindex(am=AG_MANAGEMENTS_TO_LAND_USES.keys(), lu=data.AGRICULTURAL_LANDUSES, lm=data.LANDMANS)



def non_ag_to_xr(data:Data, yr_cal:int):
    """
    Convert non-agricultural land use data to xarray DataArray.

    Args:
        data (Data): The data object containing non-agricultural land use data.
        yr_cal (int): The year calendar.

    Returns:
        xr.DataArray: The xarray DataArray representing the non-agricultural land use data.
    """
    dvar = data.non_ag_dvars[yr_cal]
    non_ag_dvar_xr = xr.DataArray(
        dvar, 
        dims=('cell', 'lu'),
        coords={
            'cell': np.arange(dvar.shape[0]),
            'lu': data.NON_AGRICULTURAL_LANDUSES}   
    )
    return non_ag_dvar_xr.reindex(lu=data.NON_AGRICULTURAL_LANDUSES)



def bincount_avg(bin_arr, weight_arr, low_res_xr: xr.DataArray=None):
    """
    Calculate the average of weighted values based on bin counts.

    Parameters:
    - bin_arr (2D, xarray.DataArray): Array containing the mask values. 
    - weight_arr (2D, xarray.DataArray, >0 values are valid): Array containing the weight values.
    - low_res_xr (2D, xarray.DataArray): Low-resolution array containing 
        `y`, `x`, `CRS`, and `transform` to restore the bincount stats.

    Returns:
    - bin_avg (xarray.DataArray): Array containing the average values based on bin counts.
    """
    # Get the valide cells
    valid_mask = weight_arr > 0
    
    # Flatten arries
    bin_flatten = bin_arr.values[valid_mask]
    weights_flatten = weight_arr.values[valid_mask]
    
    bin_occ = np.bincount(bin_flatten, minlength=low_res_xr.size)
    bin_sum = np.bincount(bin_flatten, weights=weights_flatten, minlength=low_res_xr.size)
    
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



def match_lumap_biomap(
    data:Data, 
    map_:np.ndarray, 
    res_factor:int, 
    bio_id_path:str=f'{settings.INPUT_DIR}/bio_id_map.nc',
    lumap_tempelate:str=f'{settings.INPUT_DIR}/NLUM_2010-11_mask.tif', 
    biomap_tempelate:str=f'{settings.INPUT_DIR}/bio_mask.nc')-> xr.DataArray:
    """
    Matches the lumap and biomap data based on the given parameters.

    Args:
        data (Data): The data object.
        map_ (np.ndarray): The map array.
        res_factor (int): The resolution factor.
        bio_id_path (str, optional): The path to the bio id map. Defaults to f'{settings.INPUT_DIR}/bio_id_map.nc'.
        lumap_tempelate (str, optional): The path to the lumap template. Defaults to f'{settings.INPUT_DIR}/NLUM_2010-11_mask.tif'.
        biomap_tempelate (str, optional): The path to the biomap template. Defaults to f'{settings.INPUT_DIR}/bio_mask.nc'.

    Returns:
        xr.DataArray: The matched map array.
    """
    # Read lumap, bio_map, and bio_id_map
    NLUM = rxr.open_rasterio(lumap_tempelate, chunks='auto').squeeze('band').drop_vars('band')
    bio_mask_ds = xr.open_dataset(f'{settings.INPUT_DIR}/bio_mask.nc', decode_coords="all")
    bio_map = bio_mask_ds['data']
    bio_map['spatial_ref'] = bio_mask_ds['spatial_ref']
    bio_id_map = xr.open_dataset(bio_id_path, chunks='auto')['data']
        
    if res_factor > 1:   
        map_ = get_coarse2D_map(data, map_)
        map_ = upsample_array(data, map_, res_factor)
    else:
        empty_map = np.full(data.NLUM_MASK.shape, data.NODATA).astype(np.float32) 
        np.place(empty_map, data.NLUM_MASK, data.LUMAP_NO_RESFACTOR) 
        np.place(empty_map, empty_map >=0, map_.data.todense())
        map_ = empty_map
        
    map_ = xr.DataArray(map_, dims=('y','x'), coords={'y': NLUM['y'], 'x': NLUM['x']})
    map_ = map_.where(map_>=0, 0)
    map_ = map_.rio.write_crs(NLUM.rio.crs)
    map_ = map_.rio.write_transform(NLUM.rio.transform())  
    map_ = bincount_avg(bio_id_map, map_,  bio_map)
    map_ = map_.where(map_ != map_.rio.nodata, 0)
    return map_



def ag_dvar_to_bio_map(data, ag_dvar, res_factor, workers=5):
    """
    Reprojects and matches agricultural land cover variables to biodiversity maps.

    Parameters:
    - data (Data object): The Data class object of LUTO.
    - ag_dvar (xarray.Dataset): The agricultural land cover variables.
    - res_factor (int): The resolution factor for matching.
    - workers (int, optional): The number of parallel workers to use for processing. Default is 5.

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
    para_obj = Parallel(n_jobs=workers, return_as='generator')
    out_arr = xr.combine_by_coords([i for i in para_obj(tasks)])
    
    # Convert to sparse array to save memory
    out_arr.values = sparse.COO.from_numpy(out_arr.values)
    
    return out_arr


def am_dvar_to_bio_map(data, am_dvar, res_factor, workers=5):
    """
    Reprojects and matches agricultural land cover variables to biodiversity maps.

    Parameters:
    - data (Data object): The Data class object of LUTO.
    - am_dvar (xarray.Dataset): The agricultural land cover variables.
    - res_factor (int): The resolution factor for matching.
    - workers (int, optional): The number of parallel workers to use for processing. Default is 5.

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
    para_obj = Parallel(n_jobs=workers, return_as='generator')
    out_arr = xr.combine_by_coords([i for i in para_obj(tasks)])
    
    # Convert to sparse array to save memory
    out_arr.values = sparse.COO.from_numpy(out_arr.values)
    
    return out_arr



def non_ag_dvar_to_bio_map(data, non_ag_dvar, res_factor, workers=5):
    """
    Reprojects and matches agricultural land cover variables to biodiversity maps.

    Parameters:
    - data (Data object): The Data class object of LUTO.
    - non_ag_dvar (xarray.Dataset): The agricultural land cover variables.
    - res_factor (int): The resolution factor for matching.
    - workers (int, optional): The number of parallel workers to use for processing. Default is 5.

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
    para_obj = Parallel(n_jobs=workers, return_as='generator')
    out_arr = xr.combine_by_coords([i for i in para_obj(tasks)])
    
    # Convert to sparse array to save memory
    out_arr.values = sparse.COO.from_numpy(out_arr.values)
    
    return out_arr


# Calculate the biodiversity condition
def calc_bio_hist_sum(bio_nc_path:str, base_yr:int=1990):
    """
    Calculate the sum of historic cells for a given biodiversity dataset.

    Parameters:
    - bio_nc_path (str): The file path to the biodiversity dataset in NetCDF format.
    - base_yr (int): The base year for calculating the sum. Default is 1990.

    Returns:
    - bio_xr_hist_sum_species (xarray.DataArray): The sum of historic cells for each species.

    """
    bio_xr_raw = xr.open_dataset(bio_nc_path, chunks='auto')['data']
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint64'}}
    bio_xr_mask = xr.open_dataset(f'{settings.INPUT_DIR}/bio_mask.nc', chunks='auto')['data'].astype(np.bool_)
    
    # Get the sum of all historic cells, !!! important !!! need to mask out the cells that are not in the bio_mask 
    if os.path.exists(f'{settings.INPUT_DIR}/bio_xr_hist_sum_species.nc'):
        bio_xr_hist_sum_species = xr.open_dataarray(f'{settings.INPUT_DIR}/bio_xr_hist_sum_species.nc', chunks='auto') 
    else:
        bio_xr_hist_sum_species = (bio_xr_raw.sel(year=base_yr).astype('uint64') * bio_xr_mask).sum(['x', 'y']).compute()
        bio_xr_hist_sum_species.to_netcdf(f'{settings.INPUT_DIR}/bio_xr_hist_sum_species.nc', mode='w', encoding=encoding, engine='h5netcdf')   
    return bio_xr_hist_sum_species


# Calculate the biodiversity contribution of each species
def calc_bio_score_species(bio_nc_path:str, bio_xr_hist_sum_species: xr.DataArray):
    """
    Calculate the contribution of each cell to the total biodiversity score for a given species.

    Parameters:
    bio_nc_path (str): The file path to the biodiversity data in NetCDF format.
    bio_xr_hist_sum_species (xr.DataArray): The historical sum of biodiversity scores for the species.

    Returns:
    xr.DataArray: The contribution of each cell to the total biodiversity score as a percentage.
    """
    bio_xr_raw = xr.open_dataset(bio_nc_path, chunks='auto')['data']
    # Calculate the contribution of each cell (%) to the total biodiversity
    bio_xr_contribution_species = (bio_xr_raw / bio_xr_hist_sum_species).astype(np.float32) * 100   
    return bio_xr_contribution_species


# Helper functions to interpolate biodiversity scores to given years
def interp_by_year(ds, year:list[int]):
    return ds.interp(year=year, method='linear', kwargs={'fill_value': 'extrapolate'}).astype(np.float32).compute()

def interp_bio_species_to_shards(bio_contribution_species:xr.DataArray, interp_year:list[int], n_shards=settings.THREADS):
    """
    Interpolates the bio_contribution_species data to shards based on the given interpolation years.
    Each shard is a subset from `bio_contribution_species` by spliting the species dimension into `n_shards` parts.
    
    By splitting the data into shards, we can reduce the memory use.

    Parameters:
    - bio_contribution_species (xr.DataArray): The input data array containing bio contribution species.
    - interp_year (list[int]): The list of years to interpolate the data to.
    - n_shards (int): The number of shards to split the data into.

    Returns:
    - list[xr.DataArray]: A list of interpolated data arrays for each shard and year combination.
    """
    chunks_species = np.array_split(range(bio_contribution_species['species'].size), n_shards)
    bio_xr_chunks = [bio_contribution_species.isel(species=idx) for idx in chunks_species]
    return [interp_by_year(chunk, [yr]) for chunk in bio_xr_chunks for yr in interp_year]


def interp_bio_group_to_shards(bio_contribution_group, interp_year):
    """
    Interpolates the biodiversity contribution group to shards for the given interpolation years.

    Parameters:
    - bio_contribution_group (list): List of biodiversity contribution values for a group.
    - interp_year (list): List of years for interpolation.

    """
    return [interp_by_year(bio_contribution_group, [year]) for year in interp_year]


# Helper functions to calculate the biodiversity contribution scores
def xr_to_df(shard:xr.DataArray, dvar_ag:xr.DataArray, dvar_am:xr.DataArray, dvar_non_ag:xr.DataArray):
    """
    Convert xarray DataArrays to a pandas DataFrame.

    Parameters:
    - shard (xr.DataArray): The input data to be converted. It can be either a list of xarray DataArrays or a single xarray DataArray.
    - dvar_ag (xr.DataArray): The xarray DataArray representing the agricultural variable.
    - dvar_am (xr.DataArray): The xarray DataArray representing the agricultural margin variable.
    - dvar_non_ag (xr.DataArray): The xarray DataArray representing the non-agricultural variable.

    Returns:
    - pd.DataFrame: The converted pandas DataFrame containing the contribution scores for each land use type.
    """

    # Compute the shard first to avoid duplicate computation
    shard_xr = shard.compute()

    # Calculate the biodiversity contribution scores
    tmp_ag = (shard_xr * dvar_ag).compute().sum(['x', 'y'])
    tmp_am = (shard_xr * dvar_am).compute().sum(['x', 'y'])
    tmp_non_ag = (shard_xr * dvar_non_ag).compute().sum(['x', 'y'])
    
    # Convert to dense array
    tmp_ag.data = tmp_ag.data.todense()
    tmp_am.data = tmp_am.data.todense()
    tmp_non_ag.data = tmp_non_ag.data.todense()
    
    # Convert to dataframe
    tmp_ag_df = tmp_ag.to_dataframe(name='contribution_%').reset_index()
    tmp_am_df = tmp_am.to_dataframe(name='contribution_%').reset_index()
    tmp_non_ag_df = tmp_non_ag.to_dataframe(name='contribution_%').reset_index()
    
    # Add land use type
    tmp_ag_df['lu_type'] = 'ag'
    tmp_am_df['lu_type'] = 'am'
    tmp_non_ag_df['lu_type'] = 'non_ag'
    
    return pd.concat([tmp_ag_df, tmp_am_df, tmp_non_ag_df], ignore_index=True)


def calc_bio_score_by_yr(ag_dvar:xr.DataArray, am_dvar:xr.DataArray, non_ag_dvar:xr.DataArray, bio_shards:list[xr.DataArray]):
    """
    Calculate the bio score by multipling `dvar` with `bio_contribution`. 
    The `dvar` is a 2D land-use map, indicating the percentage of each land use type in each cell.
    The `bio_contribution` is a 3D array (group * y * x), indicating the contribution of each cell to the total biodiversity score for each species.

    Parameters:
    - ag_dvar (xr.DataArray): The agricultural data variable.
    - am_dvar (xr.DataArray): The agricultural management data variable.
    - non_ag_dvar (xr.DataArray): The non-agricultural data variable.
    - bio_shards (list[xr.DataArray]): The list of bio shards.

    Returns:
    - out (pd.DataFrame): The concatenated DataFrame with bio scores, excluding the 'spatial_ref' column.
    """
    
    out = [xr_to_df(bio_score, ag_dvar, am_dvar, non_ag_dvar) for bio_score in bio_shards]
    out = pd.concat(out, ignore_index=True)
    return out.drop(columns=['spatial_ref'])