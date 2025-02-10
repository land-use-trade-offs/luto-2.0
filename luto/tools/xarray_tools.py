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

import os
import numpy as np
import pandas as pd
import xarray as xr
import luto.settings as settings


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
    
    # # Convert to dense array
    # tmp_ag.data = tmp_ag.data.todense()
    # tmp_am.data = tmp_am.data.todense()
    # tmp_non_ag.data = tmp_non_ag.data.todense()
    
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
    - out (pd.DataFrame): The concatenated DataFrame with bio scores.
    """
    
    out = [xr_to_df(bio_score, ag_dvar, am_dvar, non_ag_dvar) for bio_score in bio_shards]
    return pd.concat(out, ignore_index=True)