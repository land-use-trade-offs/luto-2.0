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


def match_lumap_biomap(data, dvar, bio_id_path:str=f'{settings.INPUT_DIR}/bio_id_map.nc'):

    # Read the bio id map
    bio_id_map = xr.open_dataset(bio_id_path)['data']
    
    # Rechunk the dvar, where the 'cell' dimension keeps as its raw size
    dim_size = {k: 1 for k in dvar.dims}
    dim_size['cell'] = -1
    dvar = dvar.chunk(dim_size)
    
    # Get the template for the upsampled dvar and bincount_avg
    coords = dict(dvar.coords)

    # Remove the 'cell' dimension
    del coords['cell']
    del dim_size['cell']
    upsample_template = get_upsample_template(coords, dim_size)
    bin_template = get_bincount_template(coords, dim_size)

    # Apply the function using map_blocks
    dvar = dvar.map_blocks(
        upsample_dvar, 
        kwargs={'data':data, 'res_factor':settings.RESFACTOR, 'x':upsample_template['x'], 'y':upsample_template['y']},
        template=upsample_template)

    dvar = dvar.map_blocks(
        bincount_avg,
        kwargs={'bin_arr':bio_id_map, 'y':bin_template.y, 'x':bin_template.x},
        template=bin_template)
    
    return dvar.compute()
  
  
    
def get_upsample_template(coords, dim_size, lumap_tempelate:str=f'{settings.INPUT_DIR}/NLUM_2010-11_mask.tif'):
    # Read reference maps
    NLUM = rxr.open_rasterio(lumap_tempelate, chunks='auto').squeeze('band').drop_vars('band')
    NLUM = NLUM.drop_vars('spatial_ref') 
    NLUM.attrs = {}
    return NLUM.expand_dims(coords).chunk(dim_size)



def get_bincount_template(coords, dim_size, biomap_tempelate:str=f'{settings.INPUT_DIR}/bio_mask.nc'):
    bio_map = xr.open_dataset(biomap_tempelate, chunks='auto')['data']
    return bio_map.expand_dims(coords).chunk(dim_size)


def bincount_avg(weight_arr, bin_arr, y, x):
    """
    Calculate the average value of each bin based on the weighted values.

    Parameters:
    - weight_arr (xarray.DataArray): Array of weighted values.
    - bin_arr (xarray.DataArray): Array of bin values.
    - y (numpy.ndarray): Array of y coordinates.
    - x (numpy.ndarray): Array of x coordinates.

    Returns:
    - xarray.DataArray: Array of average values for each bin.

    """
    # Get the coords of the map_ except for 'x' and 'y'
    coords = dict(weight_arr.coords)
    del coords['x']
    del coords['y']
    
    # Get the valide cells
    valid_mask = weight_arr > 0
    out_shape = (y.size, x.size)
    bin_size = int(y.size * x.size)
    
    # Flatten arries
    bin_flatten = bin_arr.values[valid_mask.squeeze(coords.keys())]
    weights_flatten = weight_arr.values[valid_mask]
    
    bin_occ = np.bincount(bin_flatten, minlength=bin_size)
    bin_sum = np.bincount(bin_flatten, weights=weights_flatten, minlength=bin_size)
  

    # Calculate the average value of each bin, ignoring division by zero (which will be nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_avg = (bin_sum / bin_occ).reshape(out_shape).astype(np.float32)
        bin_avg = np.nan_to_num(bin_avg)
        

    # Expand the dimensions of the bin_avg array to match the original weight_arr
    bin_avg = xr.DataArray(bin_avg, dims=('y', 'x'), coords={'y': y, 'x': x})
    bin_avg = bin_avg.expand_dims(coords)
    
    return bin_avg


def upsample_dvar(
    map_:np.ndarray,
    data:Data, 
    res_factor:int,
    x:np.ndarray,
    y:np.ndarray)-> xr.DataArray:

    # Get the coords of the map_
    coords = dict(map_.coords)
    del coords['cell']
    
    # Up sample the arr as RESFACTOR=1
    if res_factor > 1:   
        map_ = get_coarse2D_map(data, map_)
        map_ = upsample_array(data, map_, res_factor)
    else:
        empty_map = np.full(data.NLUM_MASK.shape, data.NODATA).astype(np.float32) 
        np.place(empty_map, data.NLUM_MASK, data.LUMAP_NO_RESFACTOR) 
        np.place(empty_map, empty_map >=0, map_)
        map_ = empty_map
    
    # Convert to xarray
    map_ = xr.DataArray(map_, dims=('y','x'), coords={'y': y, 'x': x})
    map_ = map_.expand_dims(coords)
    return map_



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