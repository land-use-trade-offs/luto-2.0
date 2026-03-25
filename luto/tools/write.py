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
import re
import json
import shutil
import threading
import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray as cfxr
import rasterio.features
import geopandas as gpd

from shapely.geometry import shape
from joblib import Parallel, delayed
from collections import defaultdict

from luto import settings
from luto import tools
from luto.data import Data

from luto.tools.Manual_jupyter_books.helpers import arr_to_xr
from luto.tools.report.data_tools.parameters import GHG_NAMES
from luto.tools.report.create_report_layers import save_report_layer
from luto.tools.report.create_report_data import save_report_data

import luto.economics.agricultural.quantity as ag_quantity                  # ag production already calculated in solver, imported but skip here                 
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.transitions as ag_transitions
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.biodiversity as ag_biodiversity

import luto.economics.non_agricultural.quantity as non_ag_quantity          # non-ag production already calculated in solver, imported but skip here
import luto.economics.non_agricultural.revenue as non_ag_revenue
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.transitions as non_ag_transitions
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.non_agricultural.biodiversity as non_ag_biodiversity




# ── Write parameters ─────────────────────────────────────────────────────────

MAX_CELL_MAGNITUDE = (f := lambda: defaultdict(f))()
# Arbitrary-depth nested defaultdict.
# Innermost leaf: list of scalar values accumulated across workers/years.

# Quantiles to get a robust estimate of the magnitude for setting colorbar limits in the report.
# This elinimates extreme values calculates using vanilla min/max.
MIN_P, MAX_P = 0.005, 0.995

def _get_mag(arr: xr.DataArray) -> list:
    """Return [MIN_P-quantile, MAX_P-quantile] of non-zero values via numpy (avoids MultiIndex quantile bug)."""
    vals = arr.where(arr != 0).compute().values.ravel()
    return [float(np.nanquantile(vals, MIN_P)), float(np.nanquantile(vals, MAX_P))]


def _mag_merge(contributions):
    """
    Merge per-worker magnitude contributions into `MAX_CELL_MAGNITUDE`.

    contributions: {top_key: {sub_key: tuple_or_dict}}
    At each leaf, the tuple is extended into a flat list stored under '_vals'.
    """
    def _apply(node, val):
        if isinstance(val, dict):
            for k, v in val.items():
                _apply(node[k], v)
        else:
            if '_vals' not in node:
                node['_vals'] = []
            node['_vals'].extend(val)

    for top_key, val in contributions.items():
        _apply(MAX_CELL_MAGNITUDE[top_key], val)








# ── Shared helpers ────────────────────────────────────────────────────────────

def add_all(da, dims):
    """Prepend an ALL-aggregate slice along dim."""
    for dim in dims:
        ds = da.sum(dim=dim, keepdims=True).assign_coords({dim: ['ALL']})
        da = xr.concat([ds, da], dim=dim)
    return da


def to_region_and_aus_df(da, group_dims, yr_cal):
    """Aggregate xarray to region-level DataFrame; return (AUS+region combined, AUS only).
    group_dims must include 'region' as the first element."""
    aus_dims = [d for d in group_dims if d != 'region'] + ['Year']
    region = (
        da.groupby('region').sum(dim='cell')
        .to_dataframe('Value ($)').reset_index()
        .groupby(group_dims)[['Value ($)']].sum().reset_index()
        .assign(Year=yr_cal)
        .query('abs(`Value ($)`) > 1')
    )
    aus = (
        region.groupby(aus_dims).sum().reset_index()
        .assign(region='AUSTRALIA')
        .query('abs(`Value ($)`) > 1')
    )
    return pd.concat([aus, region]), aus


def _bio_to_region_and_aus_df(da, group_dims, value_name, base_score, yr_cal):
    """
    Aggregate xarray to region-level DataFrame; return (AUS+region combined, AUS only).
    group_dims must include 'region' as the first element.
    """
    aus_dims = [d for d in group_dims if d != 'region'] + ['Year']
    region = (
        da.groupby('region')
        .sum(dim='cell')
        .to_dataframe(value_name).reset_index()
        .groupby(group_dims)[[value_name]]
        .sum()
        .reset_index()
        .assign(Year=yr_cal)
        .eval(f'Relative_Contribution_Percentage = (`{value_name}` / {base_score}) * 100')
        .query(f'abs(`{value_name}`) > 1')
    )
    aus = (
        region.groupby(aus_dims)
        .sum()
        .reset_index()
        .assign(region='AUSTRALIA')
        .query(f'abs(`{value_name}`) > 1')
    )
    return pd.concat([aus, region]), aus

def save2nc(in_xr: xr.DataArray, save_path: str):
    chunks = {dim: (size if dim == 'cell' else 1) for dim, size in in_xr.sizes.items()}
    ds = cfxr.encode_multi_index_as_compress(
        in_xr.chunk(chunks).to_dataset(name='data'), 'layer'
    )
    ds.to_netcdf(
        save_path,
        encoding={'data': {'dtype': 'float32', 'zlib': True, 'complevel': 4, 'chunksizes': list(chunks.values())}},
    )
    
def save_csv(df, rename_map, filepath):
    (df.rename(columns=rename_map)
       .infer_objects(copy=False)
       .replace({'dry': 'Dryland', 'irr': 'Irrigated'})
       .to_csv(filepath, index=False))


def valid_layers_na(df):
    if df.empty:
        return pd.MultiIndex.from_tuples([('ALL',)], names=['lu'])
    return pd.MultiIndex.from_frame(df[['lu']]).sort_values()


def process_cost_chunks(trans_xr, data, yr_cal, chunk_size, groupby_cols, value_col):
    """
    Process large xarray in chunks and aggregate to DataFrame.This is because the input array
    is a huge intermediate array that consumes a lot of memory. By mannually select each chunk,
    we can limit the size of the intermediate array.

    Memory usage at RESFACTOR=13 for trans_xr:
    - Without manual chunking (trans_xr): ~5 GB
    - With manual chunking:  ~70 MB

    Args:
        trans_xr: xarray DataArray to process
        data: Data object with NCELLS attribute
        yr_cal: Calendar year
        chunk_size: Number of cells per chunk
        groupby_cols: List of column names for groupby operation
        value_col: Name of value column

    Returns:
        List of DataFrames, one per chunk
    """
    trans_dfs = []
    for i in range(0, data.NCELLS, chunk_size):
        end_idx = min(i + chunk_size, data.NCELLS)
        cell_slice = slice(i, end_idx)
        chunk_arr = trans_xr.isel(cell=cell_slice).compute()

        df_region = chunk_arr.groupby('region'
            ).sum(dim='cell'
            ).to_dataframe(value_col
            ).reset_index(
            ).groupby(groupby_cols
            )[value_col
            ].sum(
            ).reset_index(
            ).query(f'abs(`{value_col}`) > 1'
            ).assign(Year=yr_cal, chunk_idx=i//chunk_size)

        trans_dfs.append(df_region)

    return trans_dfs


# ── Config / Orchestration ────────────────────────────────────────────────────

def write_outputs(data: Data):
    """Write outputs using dynamic timestamp from read_timestamp."""
    timestamp = tools.read_timestamp()
    out_dir = f'{settings.OUTPUT_DIR}/{timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}'

    @tools.LogToFile(f"{out_dir}/LUTO_RUN_")
    def _write_outputs():
        stop_event = threading.Event()
        memory_thread = threading.Thread(target=tools.log_memory_usage, args=(out_dir, 'a', 1, stop_event))
        memory_thread.start()
        try:
            write_data(data)
            print("Data writing complete, now creating report...\n")
            create_report(data)
        except Exception as e:
            print(f"An error occurred while writing outputs: {e}")
            raise
        finally:
            stop_event.set()
            memory_thread.join()

    return _write_outputs()



def write_data(data: Data):
    years = [yr for yr in settings.SIM_YEARS if yr <= data.last_year]
    paths = [f"{data.path}/out_{yr}" for yr in years]
    write_settings(data.path)

    # DVars must be written first as other outputs depend on them
    dvar_jobs = [delayed(write_dvar_and_mosaic_map)(data, yr, path_yr) for yr, path_yr in zip(years, paths)]
    for result in Parallel(n_jobs=min(len(years), settings.WRITE_THREADS), return_as="generator")(dvar_jobs):
        print(result)

    # Other outputs can be written in any order, but we run them in parallel to speed up the process. 
    #   We also include the area transition start/end output as a job to ensure it runs after the 
    #   dvars are written (it depends on the dvars of the last year).
    jobs = [delayed(write_area_transition_start_end)(data, f'{data.path}/out_{years[-1]}', years[-1])]
    for yr, path_yr in zip(years, paths):
        jobs += write_output_single_year(data, yr, path_yr)
    jobs = [job for job in jobs if job is not None]  # None means task is skipped

    # We use joblib.Parallel to run the jobs in parallel, but we set n_jobs=1 if WRITE_PARALLEL is False
    #   to run them sequentially (this is useful for debugging or if there are memory issues with parallelization).
    num_jobs = min(len(jobs), settings.WRITE_THREADS) if settings.WRITE_PARALLEL else 1
    for result in Parallel(n_jobs=num_jobs, return_as='generator_unordered')(jobs):
        if isinstance(result, tuple):
            msg, mag = result
            _mag_merge(mag)
            print(msg)
        else:
            print(result)
            
    # After all jobs are done, we collapse the MAX_CELL_MAGNITUDE dict to get the final maximum magnitudes 
    # for each variable, which can be used for setting colorbar limits in the report.
    def _defaultdict_to_dict(d):
        if isinstance(d, defaultdict):
            if '_vals' in d:
                return [0.0 if np.isnan(v) else float(v) for v in d['_vals']]
            return {k: _defaultdict_to_dict(v) for k, v in d.items()}
        return d

    with open(os.path.join(data.path, 'max_cell_magnitudes.json'), 'w') as f:
        json.dump(_defaultdict_to_dict(MAX_CELL_MAGNITUDE), f, indent=2)
  
  

def write_settings(path):
    pattern = re.compile(r"^(\s*[A-Z].*?)\s*=")
    all_settings = {k: getattr(settings, k) for k in dir(settings) if k.isupper()}
    with open('luto/settings.py') as f:
        order = [m[1].strip() for line in f if (m := pattern.match(line))]
        ordered = {k: all_settings[k] for k in order if k in all_settings}
    with open(os.path.join(path, 'model_run_settings.txt'), 'w') as f:
        f.writelines(f'{k}:{v}\n' for k, v in ordered.items())



def create_report(data: Data):
    out_dir = f"{settings.OUTPUT_DIR}/{tools.read_timestamp()}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"

    @tools.LogToFile(f"{out_dir}/LUTO_RUN_", mode='a')
    def _create_report():
        print('Creating report...')
        print('├── Copying report template...')
        shutil.copytree('luto/tools/report/VUE_modules', f"{data.path}/DATA_REPORT", dirs_exist_ok=True)
        print('├── Creating chart data...')
        save_report_data(data.path)
        print('├── Creating map data...')
        save_report_layer(data.path)
        print('└── Report created successfully!')

    return _create_report()



def write_output_single_year(data: Data, yr_cal, path_yr):
    """Wrap write tasks for a single year"""

    if not os.path.isdir(path_yr):
        os.mkdir(path_yr)
        
    tasks = [
        delayed(write_dvar_area)(data, yr_cal, path_yr),
        delayed(write_crosstab)(data, yr_cal, path_yr),
        delayed(write_quantity)(data, yr_cal, path_yr),
        delayed(write_economics)(data, yr_cal, path_yr),
        delayed(write_transition_ag2ag)(data, yr_cal, path_yr),
        delayed(write_transition_ag2nonag)(data, yr_cal, path_yr),
        delayed(write_transition_nonag2ag)(data, yr_cal, path_yr),
        delayed(write_water)(data, yr_cal, path_yr),
        delayed(write_ghg)(data, yr_cal, path_yr),
        delayed(write_biodiversity_quality_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF2_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF3_NVIS_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF3_IBRA_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF4_SNES_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF4_ECNES_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF8_scores_groups)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF8_scores_species)(data, yr_cal, path_yr)
    ]

    return tasks



# ── DVAR ─────────────────────────────────────────────────────────────────────

def write_dvar_and_mosaic_map(data: Data, yr_cal, path):

    ag_map = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    non_ag_map = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    am_map = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    
    ag_mask = ag_map.sum(['lm','lu']) > 0.001
    am_mask = am_map.sum(['am','lm', 'lu']) > 0.001
    non_ag_mask = non_ag_map.sum('lu') > 0.001

    ag_map = add_all(ag_map, ['lm'])
    am_map = add_all(am_map, ['lm', 'lu'])

    lm_map = data.lmmaps[yr_cal].astype(bool) # has to be boolean for the '~' operator to work
    lu_map = data.lumaps[yr_cal]
    
    ag_map_argmax_ALL = ag_map.sum('lm').argmax(dim='lu').expand_dims(lm=['ALL']).astype(np.float32)
    ag_map_argmax_dry = ag_map_argmax_ALL.where(~lm_map).drop_vars('lm').assign_coords(lm=['dry']).astype(np.float32)
    ag_map_argmax_irr = ag_map_argmax_ALL.where(lm_map).drop_vars('lm').assign_coords(lm=['irr']).astype(np.float32)
    ag_map_argmax = xr.concat([ag_map_argmax_ALL, ag_map_argmax_dry, ag_map_argmax_irr], dim='lm').astype(np.float32)
    ag_map_argmax = ag_map_argmax.expand_dims(lu=['ALL'])
    ag_map_argmax = xr.where(ag_mask.values[None, None, :], ag_map_argmax, np.nan)
    
    am_argmax_ALL = am_map.sum(['lm','lu']).argmax(dim='am').expand_dims(lm=['ALL']).astype(np.float32)
    am_argmax_dry = am_argmax_ALL.where(~lm_map).drop_vars('lm').assign_coords(lm=['dry']).astype(np.float32)
    am_argmax_irr = am_argmax_ALL.where(lm_map).drop_vars('lm').assign_coords(lm=['irr']).astype(np.float32)
    am_argmax_lm = xr.concat([am_argmax_ALL, am_argmax_dry, am_argmax_irr], dim='lm')
    am_argmax_lu = xr.concat([
        am_argmax_lm.where(lu_map == lu_code).expand_dims(lu=[lu_desc])
        for lu_code, lu_desc in data.AGLU2DESC.items()
        if lu_code != -1 # Exclude NoData (cells outside LUTO study area)
    ], dim='lu')
    am_argmax = xr.concat([am_argmax_lu.sum('lu', skipna=True).expand_dims(lu=['ALL']), am_argmax_lu], dim='lu')
    am_argmax = am_argmax.expand_dims(am=['ALL'])
    am_argmax = xr.where(am_mask.values[None,None,None,:], am_argmax, np.nan)
    

    non_ag_map_argmax = non_ag_map.argmax(dim='lu') + settings.NON_AGRICULTURAL_LU_BASE_CODE
    non_ag_map_argmax = xr.where(non_ag_mask, non_ag_map_argmax, np.nan)
    non_ag_map_argmax = non_ag_map_argmax.expand_dims(lu=['ALL']).astype(np.float32)
    
    ag_map_cat = xr.concat([ag_map_argmax, ag_map], dim='lu')
    non_ag_map_cat = xr.concat([non_ag_map_argmax, non_ag_map], dim='lu')
    am_map_cat = xr.concat([am_argmax, am_map], dim='am')
    
    ag_map_stack = ag_map_cat.stack(layer=['lm','lu'])
    non_ag_map_stack = non_ag_map_cat.stack(layer=['lu'])
    am_map_stack = am_map_cat.stack(layer=['am','lm','lu'])
    
    valid_layers_ag = (ag_map_stack.sum('cell') > 0.001).to_dataframe('valid').query('valid == True').index
    valid_layers_non_ag = (non_ag_map_stack.sum('cell') > 0.001).to_dataframe('valid').query('valid == True').index
    valid_layers_am = (am_map_stack.sum('cell') > 0.001).to_dataframe('valid').query('valid == True').index

    save2nc(ag_map_stack.sel(layer=valid_layers_ag), os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))
    save2nc(non_ag_map_stack.sel(layer=valid_layers_non_ag), os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))
    save2nc(am_map_stack.sel(layer=valid_layers_am), os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))

    lumap_xr_ALL= xr.DataArray(data.lumaps[yr_cal].astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})
    lumap_xr_dry = xr.DataArray(np.where(~lm_map, lumap_xr_ALL, np.nan).astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})
    lumap_xr_irr = xr.DataArray(np.where(lm_map, lumap_xr_ALL, np.nan).astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})
    
    lumap_xr = xr.concat([
        lumap_xr_ALL.expand_dims(lm=['ALL']),
        lumap_xr_dry.expand_dims(lm=['dry']),
        lumap_xr_irr.expand_dims(lm=['irr'])
    ], dim='lm').astype(np.float32)
        
    save2nc(lumap_xr.stack(layer=['lm']), os.path.join(path, f'xr_map_lumap_{yr_cal}.nc'))
    
    
    xr.Dataset({
        'layer':arr_to_xr(data, lumap_xr_ALL.astype(np.float32))
    }).to_netcdf(os.path.join(path, f'xr_map_template_{yr_cal}.nc'))

    return f"Mosaic maps written for year {yr_cal}"

def write_dvar_area(data: Data, yr_cal, path):

    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).assign_coords({'region': ('cell', data.REGION_NRM_NAME)}
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    non_ag_rj = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).assign_coords({'region': ('cell', data.REGION_NRM_NAME)}
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    am_dvar_mrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).assign_coords({'region': ('cell', data.REGION_NRM_NAME)}
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})


    real_area_r = xr.DataArray(data.REAL_AREA.astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})

    area_ag = (ag_dvar_mrj * real_area_r)
    area_non_ag = (non_ag_rj * real_area_r)
    area_am = (am_dvar_mrj * real_area_r)

    area_ag = add_all(area_ag, ['lm'])
    area_am = add_all(area_am, ['lm', 'lu'])

    df_ag_area_region = area_ag.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).assign(Year=yr_cal
        ).query('abs(`Area (ha)`) > 1')
    df_non_ag_area_region = area_non_ag.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).assign(Year=yr_cal
        ).query('abs(`Area (ha)`) > 1')
    df_am_area_region = area_am.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).assign(Year=yr_cal
        ).query('abs(`Area (ha)`) > 1')

    df_ag_area_AUS = area_ag.sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).query('abs(`Area (ha)`) > 1')
    df_non_ag_area_AUS = area_non_ag.sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).query('abs(`Area (ha)`) > 1')
    df_am_area_AUS = area_am.sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).query('abs(`Area (ha)`) > 1')

    pd.concat([df_ag_area_AUS, df_ag_area_region]
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply'}
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'area_agricultural_landuse_{yr_cal}.csv'), index = False)
    pd.concat([df_non_ag_area_AUS, df_non_ag_area_region]
        ).rename(columns={'lu': 'Land-use'}
        ).to_csv(os.path.join(path, f'area_non_agricultural_landuse_{yr_cal}.csv'), index = False)
    pd.concat([df_am_area_AUS, df_am_area_region]
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply', 'am': 'Type'}
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'area_agricultural_management_{yr_cal}.csv'), index = False)


    # ==================== Agricultural Area ====================
    # Get valid data layers
    valid_ag_layers = pd.MultiIndex.from_frame(df_ag_area_AUS[['lm', 'lu']]).sort_values()
    area_ag_valid_layers = area_ag.stack(layer=['lm', 'lu']).sel(layer=valid_ag_layers)

    # Get mosaic and filter
    ag_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc')), 'layer'
        )['data'].sel(lu='ALL', lm='ALL')
    ag_mosaic_area = ag_mosaic.where(
            area_ag.sum(dim='lu').transpose('cell', ...)
        ).expand_dims(lu=['ALL'])

    # Stack mosaic and filter by valid lm (NOT lu since mosaic has lu='ALL' only)
    ag_mosaic_area_stack = ag_mosaic_area.stack(layer=['lm', 'lu'])
    ag_mosaic_area_stack = ag_mosaic_area_stack.sel(
        layer=ag_mosaic_area_stack['layer']['lm'].isin(valid_ag_layers.get_level_values('lm'))
    )

    # Combine valid layers from data and mosaic
    area_ag_cat = xr.concat([ag_mosaic_area_stack, area_ag_valid_layers], dim='layer').drop_vars('region')


    # ==================== Non-Agricultural Area ====================
    # Get valid data layers (NonAg: lu dimension only)
    valid_non_ag_layers = pd.MultiIndex.from_frame(df_non_ag_area_AUS[['lu']]).sort_values()

    if df_non_ag_area_AUS['Area (ha)'].abs().sum() < 1e-3:
        area_non_ag_cat = xr.DataArray(
            np.zeros((1, data.NCELLS), dtype=np.float32),
            dims=['lu', 'cell'],
            coords={'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['lu'])

    else:
        # Stack and select valid data layers
        area_non_ag_valid_layers = area_non_ag.stack(layer=['lu']).sel(layer=valid_non_ag_layers)

        # Get valid mosaic layers
        non_ag_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc')), 'layer'
        )['data']

        non_ag_mosaic = non_ag_mosaic.sel(lu='ALL').expand_dims('lu').stack(layer=['lu'])

        # Combine valid layers from dvar and mosaic
        area_non_ag_cat = xr.concat([non_ag_mosaic, area_non_ag_valid_layers], dim='layer').drop_vars('region')


    # ==================== Agricultural Management Area ====================
    # Get valid data layers (Am: am → lm → lu dimension order)
    valid_am_layers = pd.MultiIndex.from_frame(df_am_area_AUS[['am', 'lm', 'lu']]).sort_values()

    if df_am_area_AUS['Area (ha)'].abs().sum() < 1e-3:
        area_am_cat = xr.DataArray(
            np.zeros((1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['am', 'lm', 'lu', 'cell'],
            coords={'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['am', 'lm', 'lu'])

    else:
        # Stack and select valid data layers
        area_am_valid_layers = area_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_am_layers)

        # Get mosaic and filter
        am_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc')), 'layer'
        )['data'].sel(am='ALL', lm='ALL', lu='ALL')

        # Filter mosaic where data exists, then expand am dimension
        am_mosaic_area = am_mosaic.where(
            area_am.sum('am').transpose('cell', ...)
        ).expand_dims('am')

        # Stack mosaic and filter by lm and lu (NOT am since mosaic has am='ALL' only)
        am_mosaic_area_stack = am_mosaic_area.stack(layer=['am', 'lm', 'lu'])
        am_mosaic_area_stack = am_mosaic_area_stack.sel(
            layer=(
                am_mosaic_area_stack['layer']['lm'].isin(valid_am_layers.get_level_values('lm')) &
                am_mosaic_area_stack['layer']['lu'].isin(valid_am_layers.get_level_values('lu'))
            )
        )

        # Combine valid layers from data and mosaic
        area_am_cat = xr.concat([area_am_valid_layers, am_mosaic_area_stack], dim='layer').drop_vars('region')

    # Save to netcdf with valid layers
    save2nc(area_ag_cat, os.path.join(path, f'xr_area_agricultural_landuse_{yr_cal}.nc'))
    save2nc(area_non_ag_cat, os.path.join(path, f'xr_area_non_agricultural_landuse_{yr_cal}.nc'))
    save2nc(area_am_cat, os.path.join(path, f'xr_area_agricultural_management_{yr_cal}.nc'))
    
    
    # Records cell magnitudes
    area_magnitudes = {
        'area': {
            'ag':     _get_mag(area_ag_cat),
            'non_ag': _get_mag(area_non_ag_cat),
            'am':     _get_mag(area_am_cat),
        }
    }
    
    return (f"Decision variable areas written for year {yr_cal}", area_magnitudes)




# ── Quantity ─────────────────────────────────────────────────────────────────

def write_quantity(data: Data, yr_cal: int, path: str) -> np.ndarray:
    """Write commodity production quantities for a specific year.

    Covers: quantity comparison summary CSV, and per-category spatial NetCDF/CSV outputs.
    'yr_cal' is calendar year. Includes impacts of land-use change, productivity
    increases, and climate change on yield.
    """

    # ==================== Total / Comparison Summary ====================

    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]

    if yr_cal > data.YR_CAL_BASE:
        assert data.YR_CAL_BASE <= yr_cal_sim_pre < yr_cal, f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"
        prod_base = np.array(data.prod_data[yr_cal_sim_pre]['Production'])
        prod_targ = np.array(data.prod_data[yr_cal]['Production'])
        demands = data.D_CY[yr_idx]
        pd.DataFrame({
            'Commodity': [i[0].capitalize() + i[1:] for i in data.COMMODITIES],
            'Prod_base_year (tonnes, KL)': prod_base,
            'Prod_targ_year (tonnes, KL)': prod_targ,
            'Demand (tonnes, KL)': demands,
            'Abs_diff (tonnes, KL)': prod_targ - demands,
            'Prop_diff (%)': (prod_targ / demands) * 100,
            'Year': yr_cal,
        }).to_csv(os.path.join(path, f'quantity_comparison_{yr_cal}.csv'), index=False)

    # ==================== Separate Spatial Outputs ====================

    # Get the commodity quantity dataarrays (sptial layers, (tonnes/KL)/(cell))
    ag_q_mrc, non_ag_p_rc, am_p_amrc = data.get_actual_production_lyr(yr_cal)

    ag_q_mrc  = add_all(ag_q_mrc,  ['lm'])
    am_p_amrc = add_all(am_p_amrc, ['lm', 'Commodity'])


    # ==================== Region Level Aggregation ====================

    ag_q_mrc_df_region = (ag_q_mrc
        .groupby('region')
        .sum('cell')
        .to_dataframe('Production (t/KL)')
        .reset_index().assign(Year=yr_cal, Type='Agricultural')
        .query('abs(`Production (t/KL)`) > 1'))
    non_ag_p_rc_df_region = (non_ag_p_rc
        .groupby('region')
        .sum('cell')
        .to_dataframe('Production (t/KL)')
        .reset_index().assign(Year=yr_cal, Type='Non_Agricultural')
        .query('abs(`Production (t/KL)`) > 1'))
    am_p_amrc_df_region = (am_p_amrc
        .groupby('region')
        .sum('cell')
        .to_dataframe('Production (t/KL)')
        .reset_index().assign(Year=yr_cal, Type='Agricultural_Management')
        .query('abs(`Production (t/KL)`) > 1'))

    ag_q_mrc_df_AUS = ag_q_mrc_df_region.groupby(['lm', 'Commodity', 'Year']
        ).sum(
        ).assign(region='AUSTRALIA', Type='Agricultural'
        ).reset_index(
        ).query('abs(`Production (t/KL)`) > 1')
    non_ag_p_rc_df_AUS = non_ag_p_rc_df_region.groupby(['Commodity', 'Year']
        ).sum(
        ).assign(region='AUSTRALIA', Type='Non_Agricultural'
        ).reset_index(
        ).query('abs(`Production (t/KL)`) > 1')
    am_p_amrc_df_AUS = am_p_amrc_df_region.groupby(['am', 'lm', 'Commodity', 'Year']
        ).sum(
        ).assign(region='AUSTRALIA', Type='Agricultural_Management'
        ).reset_index(
        ).query('abs(`Production (t/KL)`) > 1')
        

    # Save the production dataframes to csv
    quantity_df_AUS = pd.concat([ag_q_mrc_df_AUS, non_ag_p_rc_df_AUS, am_p_amrc_df_AUS], ignore_index=True)
    quantity_df_region = pd.concat([ag_q_mrc_df_region, non_ag_p_rc_df_region, am_p_amrc_df_region], ignore_index=True)
    
    quantity_df = pd.concat([quantity_df_AUS, quantity_df_region]
            ).rename(columns={'lm':'Water_supply'}
            ).infer_objects(copy=False
            ).replace({'dry':'Dryland', 'irr':'Irrigated'})
            
            
    quantity_df.to_csv(os.path.join(path, f'quantity_production_t_separate_{yr_cal}.csv'), index=False)
    
    
    # ==================== Agricultural: Stack, Mosaic, Save ====================
    valid_ag_layers = pd.MultiIndex.from_frame(ag_q_mrc_df_AUS[['lm', 'Commodity']]).sort_values()
    ag_q_mrc_stack = ag_q_mrc.stack(layer=['lm','Commodity']).sel(layer=valid_ag_layers)

    ag_mosaic = cfxr.decode_compress_to_multi_index(
        xr.load_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc')), 'layer')['data'
        ].sel(lu='ALL', lm='ALL').rename({'lu':'Commodity'})
    ag_mosaic_valid = ag_mosaic.where(ag_q_mrc.sum('Commodity').transpose('cell', ...)).expand_dims('Commodity')
    ag_mosaic_stack = ag_mosaic_valid.stack(layer=['lm','Commodity'])
    ag_mosaic_stack = ag_mosaic_stack.sel(
        layer=ag_mosaic_stack['layer']['lm'].isin(valid_ag_layers.get_level_values('lm'))
    )

    ag_q_mrc_cat_stack = xr.concat([ag_mosaic_stack, ag_q_mrc_stack], dim='layer').drop_vars('region').compute()
    
    
    # ==================== Non-Agricultural: Stack, Mosaic, Save ====================
    valid_non_ag_layers = pd.MultiIndex.from_frame(non_ag_p_rc_df_AUS[['Commodity']]).sort_values()

    if non_ag_p_rc_df_AUS['Production (t/KL)'].abs().sum() < 1e-3:
        non_ag_p_rc_cat_stack = xr.DataArray(
            np.zeros((1, data.NCELLS), dtype=np.float32),
            dims=['Commodity', 'cell'],
            coords={'Commodity': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['Commodity'])
    else:
        non_ag_p_rc_stack = non_ag_p_rc.stack(layer=['Commodity']).sel(layer=valid_non_ag_layers)

        non_ag_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc')), 'layer')['data'
            ].sel(lu='ALL').rename({'lu':'Commodity'})
        non_ag_mosaic_stack = non_ag_mosaic.expand_dims('Commodity').stack(layer=['Commodity'])

        non_ag_p_rc_cat_stack = xr.concat([non_ag_mosaic_stack, non_ag_p_rc_stack], dim='layer').drop_vars('region').compute()


    # ==================== Agricultural Management: Stack, Mosaic, Save ====================
    valid_am_layers = pd.MultiIndex.from_frame(am_p_amrc_df_AUS[['am', 'lm', 'Commodity']]).sort_values()

    if am_p_amrc_df_AUS['Production (t/KL)'].abs().sum() < 1e-3:
        am_p_amrc_cat_stack = xr.DataArray(
            np.zeros((1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['am', 'lm', 'Commodity', 'cell'],
            coords={'am': ['ALL'], 'lm': ['ALL'], 'Commodity': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['am','lm','Commodity'])

    else:
        am_p_amrc_stack = am_p_amrc.stack(layer=['am','lm','Commodity']).sel(layer=valid_am_layers)

        am_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc')), 'layer')['data'
            ].sel(am='ALL', lm='ALL').sel(lu='ALL').rename({'lu':'Commodity'})
        am_mosaic_valid = am_mosaic.where(am_p_amrc.sum('am').transpose('cell', ...)).expand_dims('am')
        am_mosaic_stack = am_mosaic_valid.stack(layer=['am','lm','Commodity'])
        am_mosaic_stack = am_mosaic_stack.sel(
            layer=(
                am_mosaic_stack['layer']['Commodity'].isin(valid_am_layers.get_level_values('Commodity')) &
                am_mosaic_stack['layer']['lm'].isin(valid_am_layers.get_level_values('lm'))
            )
        )

        am_p_amrc_cat_stack = xr.concat([am_mosaic_stack, am_p_amrc_stack], dim='layer').drop_vars('region').compute()

    save2nc(ag_q_mrc_cat_stack, os.path.join(path, f'xr_quantities_agricultural_{yr_cal}.nc'))
    save2nc(non_ag_p_rc_cat_stack, os.path.join(path, f'xr_quantities_non_agricultural_{yr_cal}.nc'))
    save2nc(am_p_amrc_cat_stack, os.path.join(path, f'xr_quantities_agricultural_management_{yr_cal}.nc'))


    # ==================== Sum (Ag + Am + NonAg): Stack and Save ====================
    # Sum ag + am (summed over 'am'); non_ag assigned to lm='dry' to avoid double counting
    am_sum_mrc = am_p_amrc.sel(lm=['dry', 'irr'], Commodity=[c for c in am_p_amrc.coords['Commodity'].values if c != 'ALL']).sum('am')
    non_ag_as_dry = non_ag_p_rc.expand_dims('lm').assign_coords(lm=['dry']).reindex(lm=['dry', 'irr'], fill_value=0)

    sum_dry_irr = (ag_q_mrc.sel(lm=['dry', 'irr'])
                   + am_sum_mrc.reindex_like(ag_q_mrc.sel(lm=['dry', 'irr']), fill_value=0)
                   + non_ag_as_dry.reindex_like(ag_q_mrc.sel(lm=['dry', 'irr']), fill_value=0))

    sum_all = sum_dry_irr.sum('lm', keepdims=True).assign_coords(lm=['ALL'])
    sum_mrc = xr.concat([sum_all, sum_dry_irr], dim='lm')

    # Float layers: per-commodity production sums
    sum_mrc_stack = sum_mrc.stack(layer=['lm', 'Commodity'])

    # Mosaic layers: use xr_map_lumap for Commodity='ALL' (categorical land-use map with lm splits)
    lumap_mosaic = cfxr.decode_compress_to_multi_index(
        xr.load_dataset(os.path.join(path, f'xr_map_lumap_{yr_cal}.nc')), 'layer')['data'].unstack('layer')
    sum_mosaic = lumap_mosaic.expand_dims('Commodity').assign_coords(Commodity=['ALL'])
    sum_mosaic_stack = sum_mosaic.stack(layer=['lm', 'Commodity'])

    sum_cat_stack = xr.concat([sum_mosaic_stack, sum_mrc_stack], dim='layer').drop_vars('region').compute()
    save2nc(sum_cat_stack, os.path.join(path, f'xr_quantities_sum_{yr_cal}.nc'))


    # Record max cell value for report generation later (e.g., for setting colorbar limits)
    prod_magnitudes = {
        'ag':     {cm: _get_mag(ag_q_mrc.sel(Commodity=cm)) for cm in data.COMMODITIES},
        'non_ag': {cm: _get_mag(non_ag_p_rc.sel(Commodity=cm)) for cm in data.COMMODITIES},
        'am':     {cm: _get_mag(am_p_amrc.sel(Commodity=cm)) for cm in data.COMMODITIES},
        'sum':    {cm: _get_mag(sum_mrc.sel(Commodity=cm)) for cm in data.COMMODITIES},
    }
    
    commodity_magnitudes = {'production': {}}
    for cm in data.COMMODITIES:
        vals = [*prod_magnitudes['ag'][cm], *prod_magnitudes['non_ag'][cm], *prod_magnitudes['am'][cm], *prod_magnitudes['sum'][cm]]
        commodity_magnitudes['production'][cm] = [i for i in vals if not np.isnan(i)]  # Filter out None values (in case some categories don't produce certain commodities)
    
    return (
        f"Separate quantity production written for year {yr_cal}", 
        commodity_magnitudes
    )



# ── Economics ────────────────────────────────────────────────────────────────

def write_economics(data: Data, yr_cal, path):
    """Calculate agricultural, agricultural management, and non-agricultural revenue, cost, and profit.
    Also produces a Sum profit layer combining all three categories."""

    yr_idx = yr_cal - data.YR_CAL_BASE
    chunk  = {'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}

    if yr_idx == 0:
        yr_cal_sim_pre = None
    else:
        yr_cal_sim_pre = sorted(list(data.lumaps.keys()))[sorted(list(data.lumaps.keys())).index(yr_cal) - 1]


    # ==================== Agricultural Economics ====================

    ag_dvar_mrj = (
        tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
        .chunk(chunk)
        .assign_coords(region=('cell', data.REGION_NRM_NAME))
    )

    ag_rev_df_rjms = ag_revenue.get_rev_matrices(data, yr_idx, aggregate=False)
    ag_cost_df_rjms = ag_cost.get_cost_matrices(data, yr_idx, aggregate=False)
    ag_rev_df_rjms.columns.names = ['lu', 'lm', 'source']
    ag_cost_df_rjms.columns.names = ['lu', 'lm', 'source']

    if yr_cal_sim_pre is not None:
        ag2ag_mrj = ag_transitions.get_transition_matrices_ag2ag_from_base_year(
            data, yr_idx, yr_cal_sim_pre, separate=True
        )
        nonag2ag_mrj = non_ag_transitions.get_transition_matrix_nonag2ag(
            data, yr_idx, data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre], separate=True
        )
    else:
        ag2ag_mrj = {'Establishment cost': np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)}
        nonag2ag_mrj = {'Environmental Plantings': {'Transition cost (NonAg2Ag)': np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)}}

    ag_rev_rjms = xr.DataArray(
        ag_rev_df_rjms.values.astype(np.float32),
        dims=['cell', 'layer'],
        coords={'cell': range(data.NCELLS), 'layer': ag_rev_df_rjms.columns}
    ).chunk(chunk).unstack('layer')

    ag_cost_rjms = xr.DataArray(
        ag_cost_df_rjms.values.astype(np.float32),
        dims=['cell', 'layer'],
        coords={'cell': range(data.NCELLS), 'layer': ag_cost_df_rjms.columns}
    ).chunk(chunk).unstack('layer')

    ag2ag_smrj = xr.DataArray(
        np.stack(list(ag2ag_mrj.values())),
        dims=['source', 'lm', 'cell', 'lu'],
        coords={
            'source': list(ag2ag_mrj.keys()),
            'lm': data.LANDMANS,
            'cell': range(data.NCELLS),
            'lu': data.AGRICULTURAL_LANDUSES
        }
    ).chunk(chunk)

    nonag2ag_k_v = [(k, _k) for k, v in nonag2ag_mrj.items() for _k, _v in v.items()]
    nonag2ag_smrj = xr.DataArray(
        np.zeros((len(nonag2ag_k_v), data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32),
        dims=['layer', 'lm', 'cell', 'lu'],
        coords={
            'layer': pd.MultiIndex.from_tuples(nonag2ag_k_v, names=['from_lu', 'source']),
            'lm': data.LANDMANS,
            'cell': range(data.NCELLS),
            'lu': data.AGRICULTURAL_LANDUSES
        }
    ).chunk(chunk).unstack('layer', fill_value=0)

    profit_ag = (
        ag_rev_rjms.sum('source')
        - ag_cost_rjms.sum('source')
        - ag2ag_smrj.sum('source')
        - nonag2ag_smrj.sum(['source', 'from_lu'])
    )

    xr_ag_rev      = ag_dvar_mrj * ag_rev_rjms
    xr_ag_cost     = ag_dvar_mrj * ag_cost_rjms
    xr_ag2ag_cost  = ag_dvar_mrj * ag2ag_smrj
    xr_nonag2ag_cost = ag_dvar_mrj * nonag2ag_smrj
    xr_profit_ag   = ag_dvar_mrj * profit_ag

    # Prepend ALL-aggregate slices (must be after multiplication to avoid double counting)
    xr_ag_rev        = add_all(xr_ag_rev,        ['lu', 'lm', 'source'])
    xr_ag_cost       = add_all(xr_ag_cost,       ['lu', 'lm', 'source'])
    xr_ag2ag_cost    = add_all(xr_ag2ag_cost,    ['lu', 'lm', 'source'])
    xr_nonag2ag_cost = add_all(xr_nonag2ag_cost, ['lu', 'lm', 'source'])
    xr_nonag2ag_cost = add_all(xr_nonag2ag_cost, ['from_lu'])
    xr_profit_ag     = add_all(xr_profit_ag,     ['lm', 'lu'])

    ag_rev_jms,       ag_rev_jms_AUS       = to_region_and_aus_df(xr_ag_rev,        ['region', 'lu', 'lm', 'source'], yr_cal)
    ag_cost_jms,      ag_cost_jms_AUS      = to_region_and_aus_df(xr_ag_cost,       ['region', 'lu', 'lm', 'source'], yr_cal)
    ag2ag_cost_jms,   ag2ag_cost_jms_AUS   = to_region_and_aus_df(xr_ag2ag_cost,   ['region', 'lu', 'lm', 'source'], yr_cal)
    nonag2ag_cost_jms, nonag2ag_cost_jms_AUS = to_region_and_aus_df(xr_nonag2ag_cost, ['region', 'from_lu', 'lm', 'source'], yr_cal)
    profit_ag_jms,    profit_ag_jms_AUS    = to_region_and_aus_df(xr_profit_ag,     ['region', 'lu', 'lm'], yr_cal)

    if nonag2ag_cost_jms.empty:
        nonag2ag_cost_jms = pd.DataFrame({
            'from_lu': ['ALL'], 'lm': ['ALL'], 'source': ['ALL'],
            'Year': [yr_cal], 'region': ['AUSTRALIA'], 'Value ($)': [0.0]
        })

    save_csv(ag_rev_jms,          {'lu': 'Land-use',           'lm': 'Water_supply', 'source': 'Type'}, os.path.join(path, f'economics_ag_revenue_{yr_cal}.csv'))
    save_csv(ag_cost_jms,         {'lu': 'Land-use',           'lm': 'Water_supply', 'source': 'Type'}, os.path.join(path, f'economics_ag_cost_{yr_cal}.csv'))
    save_csv(ag2ag_cost_jms,      {'lu': 'To_Land-use',        'lm': 'Water_supply', 'source': 'Type'}, os.path.join(path, f'economics_ag_transition_Ag2Ag_{yr_cal}.csv'))
    save_csv(nonag2ag_cost_jms,   {'from_lu': 'From_Land-use', 'lm': 'Water_supply', 'source': 'Type'}, os.path.join(path, f'economics_ag_transition_NonAg2Ag_{yr_cal}.csv'))
    save_csv(profit_ag_jms,       {'lu': 'Land-use',           'lm': 'Water_supply', 'source': 'Type'}, os.path.join(path, f'economics_ag_profit_{yr_cal}.csv'))

    valid_rev_layers          = pd.MultiIndex.from_frame(ag_rev_jms_AUS[['lm', 'source', 'lu']]).sort_values()
    valid_cost_layers         = pd.MultiIndex.from_frame(ag_cost_jms_AUS[['lm', 'source', 'lu']]).sort_values()
    valid_ag2ag_cost_layers   = pd.MultiIndex.from_frame(ag2ag_cost_jms_AUS[['lm', 'source', 'lu']]).sort_values()
    valid_nonag2ag_cost_layers = pd.MultiIndex.from_frame(nonag2ag_cost_jms_AUS[['lm', 'source', 'from_lu']]).sort_values()
    valid_profit_ag_layers    = pd.MultiIndex.from_frame(profit_ag_jms_AUS[['lm', 'lu']]).sort_values()

    if len(valid_nonag2ag_cost_layers) == 0:
        valid_nonag2ag_cost_layers = pd.MultiIndex.from_tuples([('ALL', 'ALL', 'ALL')], names=['lm', 'source', 'from_lu'])
        xr_nonag2ag_cost = xr_nonag2ag_cost.sel(lm=['ALL'], source=['ALL'], from_lu=['ALL'])

    ag_rev_valid_layers        = xr_ag_rev.stack(layer=['lm', 'source', 'lu']).sel(layer=valid_rev_layers).drop_vars('region')
    ag_cost_valid_layers       = xr_ag_cost.stack(layer=['lm', 'source', 'lu']).sel(layer=valid_cost_layers).drop_vars('region')
    ag2ag_cost_valid_layers    = xr_ag2ag_cost.stack(layer=['lm', 'source', 'lu']).sel(layer=valid_ag2ag_cost_layers).drop_vars('region')
    nonag2ag_cost_valid_layers = xr_nonag2ag_cost.stack(layer=['lm', 'source', 'from_lu']).sel(layer=valid_nonag2ag_cost_layers).drop_vars('region')
    profit_ag_valid_layers     = xr_profit_ag.stack(layer=['lm', 'lu']).sel(layer=valid_profit_ag_layers).drop_vars('region')

    save2nc(ag_rev_valid_layers,         os.path.join(path, f'xr_economics_ag_revenue_{yr_cal}.nc'))
    save2nc(ag_cost_valid_layers,        os.path.join(path, f'xr_economics_ag_cost_{yr_cal}.nc'))
    save2nc(ag2ag_cost_valid_layers,     os.path.join(path, f'xr_economics_ag_transition_Ag2Ag_{yr_cal}.nc'))
    save2nc(nonag2ag_cost_valid_layers,  os.path.join(path, f'xr_economics_ag_transition_NonAg2Ag_{yr_cal}.nc'))
    save2nc(profit_ag_valid_layers,      os.path.join(path, f'xr_economics_ag_profit_{yr_cal}.nc'))


    # ==================== Agricultural Management Economics ====================

    am_dvar_mrj = (
        tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
        .chunk(chunk)
        .assign_coords(region=('cell', data.REGION_NRM_NAME))
    )

    ag_rev_mrj  = ag_revenue.get_rev_matrices(data, yr_idx)
    ag_cost_mrj = ag_cost.get_cost_matrices(data, yr_idx)
    am_revenue_mat = tools.am_mrj_to_xr(data, ag_revenue.get_agricultural_management_revenue_matrices(data, ag_rev_mrj, yr_idx))
    am_cost_mat    = tools.am_mrj_to_xr(data, ag_cost.get_agricultural_management_cost_matrices(data, ag_cost_mrj, yr_idx))
    am_trans_mat   = tools.am_mrj_to_xr(data, ag_transitions.get_agricultural_management_transition_matrices(data, yr_idx))

    xr_revenue_am = am_dvar_mrj * am_revenue_mat
    xr_cost_am    = am_dvar_mrj * am_cost_mat
    xr_trans_am   = am_dvar_mrj * am_trans_mat
    xr_profit_am  = xr_revenue_am - (xr_cost_am + xr_trans_am)

    xr_revenue_am = add_all(xr_revenue_am, ['lm', 'lu', 'am'])
    xr_cost_am    = add_all(xr_cost_am, ['lm', 'lu', 'am'])
    xr_trans_am   = add_all(xr_trans_am, ['lm', 'lu', 'am'])
    xr_profit_am  = add_all(xr_profit_am, ['lm', 'lu', 'am'])

    revenue_am_df, revenue_am_df_AUS = to_region_and_aus_df(xr_revenue_am, ['region', 'am', 'lm', 'lu'], yr_cal)
    cost_am_df,    cost_am_df_AUS    = to_region_and_aus_df(xr_cost_am,    ['region', 'am', 'lm', 'lu'], yr_cal)
    trans_am_df,   trans_am_df_AUS   = to_region_and_aus_df(xr_trans_am,   ['region', 'am', 'lm', 'lu'], yr_cal)
    profit_am_df,  profit_am_df_AUS  = to_region_and_aus_df(xr_profit_am,  ['region', 'am', 'lm', 'lu'], yr_cal)

    rename_map_am = {'lu': 'Land-use', 'lm': 'Water_supply', 'am': 'Management Type'}
    save_csv(revenue_am_df, rename_map_am, os.path.join(path, f'economics_am_revenue_{yr_cal}.csv'))
    save_csv(cost_am_df,    rename_map_am, os.path.join(path, f'economics_am_cost_{yr_cal}.csv'))
    save_csv(trans_am_df,   rename_map_am, os.path.join(path, f'economics_am_transition_{yr_cal}.csv'))
    save_csv(profit_am_df,  rename_map_am, os.path.join(path, f'economics_am_profit_{yr_cal}.csv'))

    if revenue_am_df_AUS.empty:
        valid_layers_rev_am = pd.MultiIndex.from_tuples([('ALL', 'ALL', 'ALL')], names=['am', 'lm', 'lu'])
    else:
        valid_layers_rev_am = pd.MultiIndex.from_frame(revenue_am_df_AUS[['am', 'lm', 'lu']]).sort_values()
    if cost_am_df_AUS.empty:
        valid_layers_cost_am = pd.MultiIndex.from_tuples([('ALL', 'ALL', 'ALL')], names=['am', 'lm', 'lu'])
    else:
        valid_layers_cost_am = pd.MultiIndex.from_frame(cost_am_df_AUS[['am', 'lm', 'lu']]).sort_values()
    if trans_am_df_AUS.empty:
        valid_layers_trans_am = pd.MultiIndex.from_tuples([('ALL', 'ALL', 'ALL')], names=['am', 'lm', 'lu'])
    else:
        valid_layers_trans_am = pd.MultiIndex.from_frame(trans_am_df_AUS[['am', 'lm', 'lu']]).sort_values()
    if profit_am_df_AUS.empty:
        valid_layers_profit_am = pd.MultiIndex.from_tuples([('ALL', 'ALL', 'ALL')], names=['am', 'lm', 'lu'])
    else:
        valid_layers_profit_am = pd.MultiIndex.from_frame(profit_am_df_AUS[['am', 'lm', 'lu']]).sort_values()

    valid_layers_stack_rev_am = xr_revenue_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_layers_rev_am).drop_vars('region')
    valid_layers_stack_cost_am = xr_cost_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_layers_cost_am).drop_vars('region')
    valid_layers_stack_transition_am = xr_trans_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_layers_trans_am).drop_vars('region')
    valid_layers_stack_profit_am = xr_profit_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_layers_profit_am).drop_vars('region')

    save2nc(valid_layers_stack_rev_am,        os.path.join(path, f'xr_economics_am_revenue_{yr_cal}.nc'))
    save2nc(valid_layers_stack_cost_am,       os.path.join(path, f'xr_economics_am_cost_{yr_cal}.nc'))
    save2nc(valid_layers_stack_transition_am, os.path.join(path, f'xr_economics_am_transition_{yr_cal}.nc'))
    save2nc(valid_layers_stack_profit_am,     os.path.join(path, f'xr_economics_am_profit_{yr_cal}.nc'))


    # ==================== Non-Agricultural Economics ====================

    non_ag_dvar = (
        tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
        .chunk(chunk)
        .assign_coords(region=('cell', data.REGION_NRM_NAME))
    )

    ag_rev_mrj  = ag_revenue.get_rev_matrices(data, yr_idx)
    ag_cost_mrj = ag_cost.get_cost_matrices(data, yr_idx)
    non_ag_rev_mat       = tools.non_ag_rk_to_xr(data, non_ag_revenue.get_rev_matrix(data, yr_cal, ag_rev_mrj, data.lumaps[yr_cal]))
    non_ag_cost_mat      = tools.non_ag_rk_to_xr(data, non_ag_cost.get_cost_matrix(data, ag_cost_mrj, data.lumaps[yr_cal], yr_cal))
    nonag2nonag_mat = tools.non_ag_rk_to_xr(data, non_ag_transitions.get_non_ag_to_non_ag_transition_matrix(data))

    if yr_cal_sim_pre is None:
        ag2nonag_mat = xr.DataArray(
            np.zeros((data.NCELLS, len(data.NON_AGRICULTURAL_LANDUSES)), dtype=np.float32),
            dims=['cell', 'lu'],
            coords={'cell': range(data.NCELLS), 'lu': data.NON_AGRICULTURAL_LANDUSES}
        )
    else:
        ag2nonag_mat = tools.non_ag_rk_to_xr(data, non_ag_transitions.get_transition_matrix_ag2nonag(
            data, yr_idx, data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre]
        ).astype(np.float32))

    non_ag_profit_mat = non_ag_rev_mat - (non_ag_cost_mat + nonag2nonag_mat + ag2nonag_mat)

    xr_revenue_non_ag = non_ag_dvar * non_ag_rev_mat
    xr_cost_non_ag    = non_ag_dvar * non_ag_cost_mat
    xr_nonag2nonag    = non_ag_dvar * nonag2nonag_mat
    xr_ag2nonag       = non_ag_dvar * ag2nonag_mat
    xr_non_ag_profit  = non_ag_dvar * non_ag_profit_mat

    xr_revenue_non_ag = add_all(xr_revenue_non_ag, ['lu'])
    xr_cost_non_ag    = add_all(xr_cost_non_ag,    ['lu'])
    xr_nonag2nonag    = add_all(xr_nonag2nonag,    ['lu'])
    xr_ag2nonag       = add_all(xr_ag2nonag,       ['lu'])
    xr_non_ag_profit  = add_all(xr_non_ag_profit,  ['lu'])

    revenue_na_df,       revenue_na_df_AUS       = to_region_and_aus_df(xr_revenue_non_ag, ['region', 'lu'], yr_cal)
    cost_na_df,          cost_na_df_AUS          = to_region_and_aus_df(xr_cost_non_ag,    ['region', 'lu'], yr_cal)
    t_nonag2nonag_df,    t_nonag2nonag_df_AUS    = to_region_and_aus_df(xr_nonag2nonag,    ['region', 'lu'], yr_cal)
    t_ag2nonag_df,       t_ag2nonag_df_AUS       = to_region_and_aus_df(xr_ag2nonag,       ['region', 'lu'], yr_cal)
    profit_na_df,        profit_na_df_AUS        = to_region_and_aus_df(xr_non_ag_profit,  ['region', 'lu'], yr_cal)

    rename_map_na = {'lu': 'Land-use'}
    save_csv(revenue_na_df,    rename_map_na, os.path.join(path, f'economics_non_ag_revenue_{yr_cal}.csv'))
    save_csv(cost_na_df,       rename_map_na, os.path.join(path, f'economics_non_ag_cost_{yr_cal}.csv'))
    save_csv(t_nonag2nonag_df, rename_map_na, os.path.join(path, f'economics_non_ag_transition_NonAg2NonAg_{yr_cal}.csv'))
    save_csv(t_ag2nonag_df,    rename_map_na, os.path.join(path, f'economics_non_ag_transition_Ag2NonAg_{yr_cal}.csv'))
    save_csv(profit_na_df,     rename_map_na, os.path.join(path, f'economics_non_ag_profit_{yr_cal}.csv'))

    vl_rev_na          = valid_layers_na(revenue_na_df_AUS)
    vl_cost_na         = valid_layers_na(cost_na_df_AUS)
    vl_t_nonag2nonag   = valid_layers_na(t_nonag2nonag_df_AUS)
    vl_t_ag2nonag      = valid_layers_na(t_ag2nonag_df_AUS)
    vl_profit_na       = valid_layers_na(profit_na_df_AUS)

    vl_stack_rev_na        = xr_revenue_non_ag.stack(layer=['lu']).sel(layer=vl_rev_na).drop_vars('region').compute()
    vl_stack_cost_na       = xr_cost_non_ag.stack(layer=['lu']).sel(layer=vl_cost_na).drop_vars('region').compute()
    vl_stack_t_nonag2nonag = xr_nonag2nonag.stack(layer=['lu']).sel(layer=vl_t_nonag2nonag).drop_vars('region').compute()
    vl_stack_t_ag2nonag    = xr_ag2nonag.stack(layer=['lu']).sel(layer=vl_t_ag2nonag).drop_vars('region').compute()
    vl_stack_profit_na     = xr_non_ag_profit.stack(layer=['lu']).sel(layer=vl_profit_na).drop_vars('region').compute()

    save2nc(vl_stack_rev_na,        os.path.join(path, f'xr_economics_non_ag_revenue_{yr_cal}.nc'))
    save2nc(vl_stack_cost_na,       os.path.join(path, f'xr_economics_non_ag_cost_{yr_cal}.nc'))
    save2nc(vl_stack_t_nonag2nonag, os.path.join(path, f'xr_economics_non_ag_transition_NonAg2NonAg_{yr_cal}.nc'))
    save2nc(vl_stack_t_ag2nonag,    os.path.join(path, f'xr_economics_non_ag_transition_Ag2NonAg_{yr_cal}.nc'))
    save2nc(vl_stack_profit_na,     os.path.join(path, f'xr_economics_non_ag_profit_{yr_cal}.nc'))


    # ==================== Sum Profit (Ag + Am + NonAg) ====================
    # Use pre-ALL profit arrays: ag has (lm, lu, cell), am has (am, lm, lu, cell), nonag has (lu, cell)
    # xr_profit_ag before add_all is ag_dvar_mrj * profit_ag => dims (lm, lu, cell) with region coord
    # We need the raw (pre-ALL) profits, so recompute from the pre-ALL vars
    raw_profit_ag = (ag_dvar_mrj * profit_ag).drop_vars('region')                          # (lm, lu, cell)
    raw_profit_am = (xr_revenue_am - (xr_cost_am + xr_trans_am))                           # already has ALL dims
    # Use pre-ALL am profit: select non-ALL am, sum over am => (lm, lu, cell)
    raw_profit_am_pre = (am_dvar_mrj * (am_revenue_mat - (am_cost_mat + am_trans_mat)))     # (am, lm, lu, cell)
    am_sum_profit = raw_profit_am_pre.sel(lm=['dry', 'irr']).sum('am').drop_vars('region')  # (lm, lu, cell)

    # NonAg profit: assign to lm='dry', append nonag land uses to ag land uses
    raw_profit_nonag = (non_ag_dvar * non_ag_profit_mat).drop_vars('region')                # (lu, cell)
    nonag_as_dry = raw_profit_nonag.expand_dims('lm').assign_coords(lm=['dry']).reindex(lm=['dry', 'irr'], fill_value=0)

    # Combine: ag land uses get ag+am; nonag land uses get nonag only (appended)
    ag_lus = list(data.AGRICULTURAL_LANDUSES)
    nonag_lus = list(data.NON_AGRICULTURAL_LANDUSES)

    # ag + am profit for ag land uses
    ag_plus_am = raw_profit_ag.sel(lm=['dry', 'irr']) + am_sum_profit.reindex(lu=ag_lus, fill_value=0)

    # Concat ag land uses and nonag land uses along lu
    sum_dry_irr = xr.concat([ag_plus_am, nonag_as_dry], dim='lu')

    # Add ALL lm aggregate
    sum_all_lm = sum_dry_irr.sum('lm', keepdims=True).assign_coords(lm=['ALL'])
    sum_profit = xr.concat([sum_all_lm, sum_dry_irr], dim='lm')

    # Add ALL lu aggregate (sum over all land uses)
    sum_all_lu = sum_profit.sum('lu', keepdims=True).assign_coords(lu=['ALL'])
    sum_profit = xr.concat([sum_all_lu, sum_profit], dim='lu')

    # Stack and save
    sum_profit_stack = sum_profit.stack(layer=['lm', 'lu']).compute()
    save2nc(sum_profit_stack, os.path.join(path, f'xr_economics_sum_profit_{yr_cal}.nc'))


    # ==================== Record Cell Magnitudes ====================
    magnitudes = {
        'Economics_ag': {
            'ag_revenue':    _get_mag(ag_rev_valid_layers),
            'ag_cost':       _get_mag(ag_cost_valid_layers),
            'ag2ag_cost':    _get_mag(ag2ag_cost_valid_layers),
            'non_ag2ag_cost':_get_mag(nonag2ag_cost_valid_layers),
            'profit_ag':     _get_mag(profit_ag_valid_layers),
        },
        'Economics_am': {
            'am_revenue':   _get_mag(valid_layers_stack_rev_am),
            'am_cost':      _get_mag(valid_layers_stack_cost_am),
            'am_transition':_get_mag(valid_layers_stack_transition_am),
            'am_profit':    _get_mag(valid_layers_stack_profit_am),
        },
        'Economics_non_ag': {
            'non_ag_revenue':        _get_mag(vl_stack_rev_na),
            'non_ag_cost':           _get_mag(vl_stack_cost_na),
            'nonag2nonag_cost':      _get_mag(vl_stack_t_nonag2nonag),
            'ag2nonag_cost':         _get_mag(vl_stack_t_ag2nonag),
            'non_ag_profit':         _get_mag(vl_stack_profit_na),
        },
        'Economics_sum': {
            'sum_profit':    _get_mag(sum_profit_stack),
        },
    }
    return (f"Economics (Ag + Am + NonAg + Sum) written for year {yr_cal}", magnitudes)



# ── Transitions ──────────────────────────────────────────────────────────────

def write_transition_ag2ag(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Calculate transition cost."""

    # Set up
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE
    chunk_size = min(settings.WRITE_CHUNK_SIZE, data.NCELLS)

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre
    
    if yr_cal == data.YR_CAL_BASE:
        base_lumap = base_lmmap = l_mrj = l_mrj_not = x_mrj = None
    else:
        base_lumap = data.lumaps[yr_cal_sim_pre]
        base_lmmap = data.lmmaps[yr_cal_sim_pre]
        l_mrj = tools.lumap2ag_l_mrj(base_lumap, base_lmmap)
        l_mrj_not = np.logical_not(l_mrj)
        x_mrj = ag_transitions.get_to_ag_exclude_matrices(data, base_lumap)

    # Get the decision variables for agricultural land-use
    ag_dvar_mrj_target = tools.ag_mrj_to_xr(
        data, 
        data.ag_dvars[yr_cal]
    ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    
    ag_dvar_mrj_base = tools.ag_mrj_to_xr(
        data, 
        (tools.lumap2ag_l_mrj(data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre]))
    )

    ag_dvar_mrj_target = ag_dvar_mrj_target.rename({'lm': 'To-water-supply', 'lu': 'To-land-use'}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': chunk_size})

    ag_dvar_mrj_base = ag_dvar_mrj_base.rename({'lm': 'From-water-supply', 'lu': 'From-land-use'}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': chunk_size})
        
        
    # ==================== Transitions - Area ====================

    if yr_idx == 0:
        # If it's the first year, we assume no transition cost (i.e., all land remains the same)
        ag_trans_mat = xr.DataArray(
            np.zeros_like(data.AG_L_MRJ).astype(np.float32),
            coords={
                'To-water-supply': data.LANDMANS,
                'cell': range(data.NCELLS),
                'To-land-use': data.AGRICULTURAL_LANDUSES,
            }
        )
    else:
        # Get the transition area matrices for agricultural land-uses
        ag_trans_mat = xr.DataArray(
            np.einsum('r,mrj,mrj->mrj', data.REAL_AREA, l_mrj_not, x_mrj).astype(np.float32),
            coords={
                'To-water-supply': data.LANDMANS,
                'cell': range(data.NCELLS),
                'To-land-use': data.AGRICULTURAL_LANDUSES,
            }
        )
        
    xr_ag_trans_area = ag_dvar_mrj_base * ag_dvar_mrj_target * ag_trans_mat

    xr_ag_trans_area = add_all(xr_ag_trans_area, dims=['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'])

    # Calculate total transition area by region and land-use (for report generation later)
    area_dfs = process_cost_chunks(
        xr_ag_trans_area, data, yr_cal, chunk_size,
        groupby_cols=['region', 'From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'],
        value_col='Transition Area (ha)'
    )

    transition_area_region = pd.concat(area_dfs, ignore_index=True)
    transition_area_AUS = transition_area_region.groupby(['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply']
        )['Transition Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('`Transition Area (ha)` > 1') # Skip transitions under 1 ha at national level


    # Save transition area to csv
    pd.concat([transition_area_region, transition_area_AUS]
        ).to_csv(os.path.join(path, f'transition_ag2ag_area_{yr_cal}.csv'), index=False)

    # Stack array and save to netcdf for later use in report (e.g., for setting colorbar limits)
    valid_trans_area_layers = pd.MultiIndex.from_frame(transition_area_AUS[['From-water-supply', 'To-water-supply', 'From-land-use', 'To-land-use']]).sort_values()
    transition_area_stacked = xr_ag_trans_area.stack({'layer': ['From-water-supply', 'To-water-supply', 'From-land-use', 'To-land-use']}
        ).sel(layer=valid_trans_area_layers
        ).drop_vars('region'
        ).compute()
        
    save2nc(transition_area_stacked, os.path.join(path, f'xr_transition_ag2ag_area_{yr_cal}.nc'))


    # ==================== Transitions - Cost ====================
    if yr_idx == 0:
        ag_transitions_cost_mat = {'Establishment cost': np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)}
    else:
        # Get the transition cost matrices for agricultural land-use
        #   l_mrj_not and x_mrj are already considered in the `get_transition_matrices_ag2ag_from_base_year`
        ag_transitions_cost_mat = ag_transitions.get_transition_matrices_ag2ag_from_base_year(data, yr_idx, yr_cal_sim_pre, separate=True)

    ag_transitions_cost_mat = xr.DataArray(
        np.stack(list(ag_transitions_cost_mat.values())).astype(np.float32),
        coords={
            'Type': list(ag_transitions_cost_mat.keys()),
            'To-water-supply': data.LANDMANS,
            'cell': range(data.NCELLS),
            'To-land-use': data.AGRICULTURAL_LANDUSES
        }
    )

    # Use xr.dot() to contract To-water-supply without broadcasting:
    #   Plain `*` would first expand ag_dvar_mrj_target × ag_transitions_cost_mat into a full
    #   [To-water-supply, To-land-use, Type, cell] intermediate array, then sum — allocating that
    #   array even though it is immediately discarded. xr.dot() fuses the multiply-and-sum into
    #   one pass so the large intermediate never exists in memory.
    #   Similarly, From-water-supply is summed on ag_dvar_mrj_base before the multiply to keep
    #   the operand size small.
    cost_xr = (
        ag_dvar_mrj_base.sum(dim='From-water-supply')
        * xr.dot(ag_dvar_mrj_target, ag_transitions_cost_mat, dims=['To-water-supply'])
    )

    cost_xr = add_all(cost_xr, ['From-land-use', 'To-land-use', 'Type'])

    # Get transition cost by region and land-use; This is for report generation later (e.g., for setting colorbar limits)
    cost_dfs = process_cost_chunks(
        cost_xr, data, yr_cal, chunk_size,
        groupby_cols=['region', 'Type', 'From-land-use', 'To-land-use'],
        value_col='Cost ($)'
    )

    cost_df_region = pd.concat(cost_dfs, ignore_index=True
        ).groupby(['region', 'From-land-use', 'To-land-use', 'Year', 'Type']
        )['Cost ($)'
        ].sum(
        ).reset_index()
    cost_df_AUS = cost_df_region.groupby(['From-land-use', 'To-land-use', 'Type', 'Year']
        )['Cost ($)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Cost ($)`) > 1e4') # Skip transitions under $10,000 at national level

    # Write to csv
    pd.concat([cost_df_region, cost_df_AUS]
        ).to_csv(os.path.join(path, f'transition_ag2ag_cost_{yr_cal}.csv'), index=False)

    # Get valid data layers
    valid_layers_transition = pd.MultiIndex.from_frame(
        cost_df_AUS[['From-land-use', 'To-land-use', 'Type']]
    ).sort_values()

    cost_xr_stacked = cost_xr.stack({'layer': ['From-land-use', 'To-land-use', 'Type']}
        ).drop_vars('region'
        ).sel(layer=valid_layers_transition
        ).compute()


    # Save the compact filtered array
    save2nc(cost_xr_stacked, os.path.join(path, f'xr_transition_ag2ag_cost_{yr_cal}.nc'))
    
    
    
    # ==================== Transitions - GHG ====================

    if yr_cal == data.YR_CAL_BASE:
        ghg_t_smrj_values = np.zeros_like(data.AG_L_MRJ).astype(np.float32)[np.newaxis]
        ghg_t_types = ['Unallocated natural to modified']
    else:
        ghg_t_dict = ag_ghg.get_ghg_transition_emissions(data, data.lumaps[yr_cal_sim_pre], separate=True)
        ghg_t_smrj_values = (np.stack(list(ghg_t_dict.values()), axis=0) * l_mrj_not * x_mrj).astype(np.float32)
        ghg_t_types = list(ghg_t_dict.keys())

    ghg_t_smrj = xr.DataArray(
        ghg_t_smrj_values,
        dims=['Type', 'To-water-supply', 'cell', 'To-land-use'],
        coords={
            'Type': ghg_t_types,
            'To-water-supply': data.LANDMANS,
            'cell': range(data.NCELLS),
            'To-land-use': data.AGRICULTURAL_LANDUSES
        }
    )

    # Calculate GHG emissions for transition penalties (collapse water dims via xr.dot, same rationale as cost)
    xr_ghg_transition = (
        ag_dvar_mrj_base.sum(dim='From-water-supply')
        * xr.dot(ag_dvar_mrj_target, ghg_t_smrj, dims=['To-water-supply'])
    )

    xr_ghg_transition = add_all(xr_ghg_transition, ['From-land-use', 'To-land-use', 'Type'])

    # Get transition GHG emissions by region and land-use; This is for report generation later (e.g., for setting colorbar limits)
    ghg_dfs = process_cost_chunks(
        xr_ghg_transition, data, yr_cal, chunk_size,
        groupby_cols=['region', 'Type', 'From-land-use', 'To-land-use'],
        value_col='Value (t CO2e)'
    )

    ghg_df_region = pd.concat(ghg_dfs, ignore_index=True
        ).groupby(['region', 'From-land-use', 'To-land-use', 'Year', 'Type']
        )['Value (t CO2e)'
        ].sum(
        ).reset_index()
    ghg_df_AUS = ghg_df_region.groupby(['From-land-use', 'To-land-use', 'Type', 'Year']
        )['Value (t CO2e)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Value (t CO2e)`) > 1e-3') # Skip transitions under 1 t CO2e at national level

    # Write to csv
    pd.concat([ghg_df_AUS, ghg_df_region]
        ).infer_objects(copy=False
        ).to_csv(os.path.join(path, f'transition_ag2ag_ghg_{yr_cal}.csv'), index=False)

    # Get valid data layers (before renaming/replacing)
    valid_transition_layers = pd.MultiIndex.from_frame(ghg_df_AUS[['From-land-use', 'To-land-use', 'Type']]).sort_values()
    transition_valid_layers = xr_ghg_transition.stack(layer=['From-land-use', 'To-land-use', 'Type']).sel(layer=valid_transition_layers).drop_vars('region')
    save2nc(transition_valid_layers, os.path.join(path, f'xr_transition_ag2ag_ghg_{yr_cal}.nc'))



    # ==================== Transitions - Water ====================
    if yr_cal == data.YR_CAL_BASE:
        w_delta_mrj = xr.DataArray(
            np.zeros_like(data.AG_L_MRJ).astype(np.float32),
            dims=['To-water-supply', 'cell', 'To-land-use'],
            coords={
                'To-water-supply': data.LANDMANS,
                'cell': range(data.NCELLS),
                'To-land-use': data.AGRICULTURAL_LANDUSES
            }
        )
    else:
        w_mrj = ag_water.get_wreq_matrices(data, yr_idx)                                                 
        w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
        w_delta_mrj = xr.DataArray(
            np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not).astype(np.float32),
            dims=['To-water-supply', 'cell', 'To-land-use'],
            coords={
                'To-water-supply': data.LANDMANS,
                'cell': range(data.NCELLS),
                'To-land-use': data.AGRICULTURAL_LANDUSES
            }
        )
        
    # Calculate water requirement changes for transition penalties
    xr_water_transition = (
        ag_dvar_mrj_base
        * ag_dvar_mrj_target
        * w_delta_mrj
    )
    
    xr_water_transition = add_all(xr_water_transition, ['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'])
    
    # Get transition water requirement changes by region and land-use; This is for report generation later (e.g., for setting colorbar limits)
    water_dfs = process_cost_chunks(
        xr_water_transition, data, yr_cal, chunk_size,
        groupby_cols=['region', 'From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'],
        value_col='Water Requirement Change (ML)'
    )
    
    # Concat chunks and aggregate to region level; 
    #   Then aggregate to national level; 
    #   Flip the water requirement to water yield change for easier 
    #   interpolation in report (i.e., requirement decrease = yield increase);
    water_df_region = pd.concat(water_dfs, ignore_index=True
        ).groupby(['region', 'From-land-use', 'To-land-use', 'Year', 'From-water-supply', 'To-water-supply']
        )['Water Requirement Change (ML)'
        ].sum(
        ).reset_index()
    water_df_region['Water Yield Change (ML)'] = -water_df_region['Water Requirement Change (ML)']
    water_df_region = water_df_region.drop(columns='Water Requirement Change (ML)')
    water_df_AUS = water_df_region.groupby(['From-land-use', 'To-land-use', 'Year', 'From-water-supply', 'To-water-supply']
        )['Water Yield Change (ML)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA')
    water_df_AUS = water_df_AUS.loc[water_df_AUS['Water Yield Change (ML)'].abs() > 1e3] # Skip transitions under 1,000 ML at national level
        
    # Save to csv 
    pd.concat([water_df_region, water_df_AUS]
        ).to_csv(os.path.join(path, f'transition_ag2ag_water_{yr_cal}.csv'), index=False)
    
    # Get valid data layers (before renaming/replacing)
    valid_water_transition_layers = pd.MultiIndex.from_frame(water_df_AUS[['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply']]).sort_values()
    water_transition_valid_layers = (
        xr_water_transition
        .stack(layer=['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'])
        .sel(layer=valid_water_transition_layers)
        .drop_vars('region')
    )
    save2nc(water_transition_valid_layers, os.path.join(path, f'xr_transition_ag2ag_water_{yr_cal}.nc'))
    
    
    
    # ==================== Transitions - Bio ====================
    '''
    Only consider GBF2 for now. Will add more if requested.
    '''
    # TODO: complete bio transition after introducing the bio transition matrices module.
    
    
    return f"Agricultural to agricultural transition changes written for year {yr_cal}"



def write_transition_ag2nonag(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Calculate transition cost."""

    # Set up
    chunk_size = min(settings.WRITE_CHUNK_SIZE, data.NCELLS)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre
    
    if yr_cal == data.YR_CAL_BASE:
        base_lumap = l_rk = l_rk_not = x_rk = None
    else:
        base_lumap = data.lumaps[yr_cal_sim_pre]
        l_rk = tools.lumap2non_ag_l_mk(base_lumap, data.N_NON_AG_LUS)
        l_rk_not = np.logical_not(l_rk)
        x_rk = non_ag_transitions.get_to_non_ag_exclude_matrices(data, base_lumap)

    # Get the non-agricultural decision variable
    ag_dvar_base = tools.ag_mrj_to_xr(data, (tools.lumap2ag_l_mrj(data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre]))
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).rename({'lm': 'From-water-supply', 'lu': 'From-land-use'}
        ).chunk({'cell': chunk_size})
    non_ag_dvar_target = tools.non_ag_rk_to_xr(data, tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).rename({'lu': 'To-land-use'}
        ).chunk({'cell': chunk_size})

    
    
    # ==================== Transitions - Area ====================
    if yr_idx == 0:
        non_ag_transitions_area_mat = xr.DataArray(
            np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32),
            coords={
                'cell': range(data.NCELLS),
                'To-land-use': data.NON_AGRICULTURAL_LANDUSES
            }
        )
    else:
        non_ag_transitions_area_mat = xr.DataArray(
            np.einsum('r,rk,rk->rk', data.REAL_AREA, l_rk_not, x_rk).astype(np.float32),
            coords={
                'cell': range(data.NCELLS),
                'To-land-use': data.NON_AGRICULTURAL_LANDUSES
            }
        )
    
    # Calculate transition area, then expand dimension (has to be after multiplication to avoid double counting)
    non_ag_transitions_area = ag_dvar_base * non_ag_transitions_area_mat * non_ag_dvar_target
    non_ag_transitions_area = add_all(non_ag_transitions_area, ['From-water-supply', 'From-land-use', 'To-land-use'])
    
    # Get transition area by region and land-use; This is for report generation later (e.g., for setting colorbar limits)
    area_dfs = process_cost_chunks(
        non_ag_transitions_area, data, yr_cal, chunk_size, 
        groupby_cols=['region', 'From-water-supply', 'From-land-use', 'To-land-use'],
        value_col='Transition Area (ha)'
    )
    
    area_df_region = pd.concat(area_dfs, ignore_index=True).groupby(
        ['region', 'From-water-supply', 'From-land-use', 'To-land-use', 'Year']
        )['Transition Area (ha)'
        ].sum(
        ).reset_index()
    area_df_AUS = area_df_region.groupby(['From-water-supply', 'From-land-use', 'To-land-use', 'Year']
        )['Transition Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('`Transition Area (ha)` > 1') # Skip transitions under 1 ha at national level
        
        
    # Save to csv
    pd.concat([area_df_region, area_df_AUS]
        ).to_csv(os.path.join(path, f'transition_ag2nonag_area_{yr_cal}.csv'), index=False)
    
    # Get valid data layers (before renaming/replacing)
    valid_layers_transition = pd.MultiIndex.from_frame(
        area_df_AUS[['From-water-supply', 'From-land-use', 'To-land-use']]
    ).sort_values()
    
    valid_layers_stack_area = non_ag_transitions_area.stack({'layer': ['From-water-supply', 'From-land-use', 'To-land-use']}
        ).sel(layer=valid_layers_transition).drop_vars('region').compute()
    
    save2nc(valid_layers_stack_area, os.path.join(path, f'xr_transition_ag2nonag_area_{yr_cal}.nc'))
        
    

    # ==================== Transitions - Cost ====================
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {
            k:{'Transition cost':np.zeros(data.NCELLS).astype(np.float32)}
            for k in settings.NON_AG_LAND_USES.keys()
        }
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_transition_matrix_ag2nonag(
            data, yr_idx, data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre], separate=True
        )

    non_ag_transitions_flat = {}
    for lu, sub_dict in non_ag_transitions_cost_mat.items():
        for source, arr in sub_dict.items():
            non_ag_transitions_flat[(lu, source)] = arr
            
    non_ag_transitions_flat = xr.DataArray(
        np.stack(list(non_ag_transitions_flat.values())).astype(np.float32),
        coords={
            'lu_source': pd.MultiIndex.from_tuples(list(non_ag_transitions_flat.keys()), names= ('To-land-use', 'Cost-type')),
            'cell': range(data.NCELLS),
        }
    )

    # Compute in chunks and aggregate to DataFrame; This is to reduce memory usage
    cost_xr = (
        ag_dvar_base.sum(dim='From-water-supply')       # Colapse water supply dimension to reduce mem
        * non_ag_transitions_flat.unstack('lu_source') 
        * non_ag_dvar_target
    )
    
    cost_xr = add_all(cost_xr, ['From-land-use', 'To-land-use', 'Cost-type'])


    # Get transition cost by region and land-use
    cost_dfs = process_cost_chunks(
        cost_xr, data, yr_cal, chunk_size,
        groupby_cols=['region', 'From-land-use', 'To-land-use', 'Cost-type'],
        value_col='Cost ($)'
    )

    cost_df_region = pd.concat(cost_dfs, ignore_index=True).groupby(
        ['region', 'From-land-use', 'To-land-use', 'Cost-type', 'Year']
        )['Cost ($)'
        ].sum(
        ).reset_index()
    cost_df_AUS = cost_df_region.groupby(['From-land-use', 'To-land-use', 'Cost-type', 'Year']
        )['Cost ($)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Cost ($)`) > 1000') # Skip transitions under $1,000 at national level
        
    pd.concat([cost_df_AUS, cost_df_region]
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_ag2nonag_cost_{yr_cal}.csv'), index=False)
        
    # Get valid data layers (before renaming/replacing)
    valid_layers_transition = pd.MultiIndex.from_frame(
        cost_df_AUS[['From-land-use', 'To-land-use', 'Cost-type']]
    ).sort_values()
    
    valid_layers_stack_cost = cost_xr.stack({'layer': ['From-land-use', 'To-land-use', 'Cost-type']}
        ).sel(layer=valid_layers_transition).drop_vars('region').compute()
    
    save2nc(valid_layers_stack_cost, os.path.join(path, f'xr_transition_ag2nonag_cost_{yr_cal}.nc'))
    
    
    
    # TODO: add GHG, Water, Bio transitions after introducing their transition matrices.
    


    return f"Agricultural to non-agricultural transition written for year {yr_cal}"





def write_transition_nonag2ag(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Calculate transition cost."""

    # Set up
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    # Get the decision variables for agricultural land-use
    nonag_dvar = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).rename({'lu': 'From-land-use'}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    ag_dvar = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).rename({'lm': 'To-water-supply', 'lu': 'To-land-use'}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
        
    # ==================== Transitions - Area ====================
    '''
    NonAg to Ag transition is currently prohibited in the model, so the transition area is zero.
    We skip the area calculation and directly create a zero array for the cost calculation below.
    We keep the code here as a placeholder for future when the transition is allowed.
    '''
    
    area_df = pd.DataFrame({
        'region': ['AUSTRALIA', 'ACT'],
        'From-water-supply': ['ALL', 'ALL'],
        'From-land-use': ['ALL', 'ALL'],
        'To-land-use': ['ALL', 'ALL'],
        'Transition Area (ha)': [0, 0],
        'Year': [yr_cal, yr_cal]
    })
    
    area_df.to_csv(os.path.join(path, f'transition_nonag2ag_area_{yr_cal}.csv'), index=False)
    
    area_xr = xr.DataArray(
        np.zeros((1, 1, data.NCELLS, 1)).astype(np.float32),
        coords={
            'From-water-supply': ['ALL'],
            'From-land-use': ['ALL'],
            'cell': range(data.NCELLS),
            'To-land-use': ['ALL']
        }
    )
    area_xr_stack = area_xr.stack({'layer': ['From-water-supply', 'From-land-use', 'To-land-use']})
    save2nc(area_xr_stack, os.path.join(path, f'xr_transition_nonag2ag_area_{yr_cal}.nc'))


    # ==================== Transitions - Cost ====================
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {
            k:{'Transition cost (Non-Ag2Ag)':np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)}
            for k in settings.NON_AG_LAND_USES.keys()
        }
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_transition_matrix_nonag2ag(
            data,
            yr_idx,
            data.lumaps[yr_cal_sim_pre],
            data.lmmaps[yr_cal_sim_pre],
            separate=True
        )


    non_ag_transitions_flat = {}
    for lu, sub_dict in non_ag_transitions_cost_mat.items():
        for source, arr in sub_dict.items():
            non_ag_transitions_flat[(lu, source)] = arr.sum(0) # Sum over `lm` dimension
            
    non_ag_transitions_flat = xr.DataArray(
        np.stack(list(non_ag_transitions_flat.values())).astype(np.float32),
        coords={
            'lu_source': pd.MultiIndex.from_tuples(
                list(non_ag_transitions_flat.keys()),
                names= ('From-land-use', 'Cost-type')
            ),
            'cell': range(data.NCELLS),
            'To-land-use': data.AGRICULTURAL_LANDUSES
        }
    ).unstack('lu_source').chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    
    # Compute transition cost
    cost_xr = xr.dot(ag_dvar, nonag_dvar, dim=['To-water-supply']) * non_ag_transitions_flat 
    
    cost_xr = add_all(cost_xr, ['From-land-use', 'To-land-use', 'Cost-type'])
    
    
    
    #   !!! cost_xr is zero for now
    #   !!! so only selecting a chunk to get the stats
    chunk_size = min(settings.WRITE_CHUNK_SIZE, data.NCELLS)
    cost_df_region = cost_xr.isel(cell=slice(0, chunk_size)
        ).groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Cost ($)'
        ).reset_index(
        ).groupby(['region', 'From-land-use', 'To-land-use', 'Cost-type'], dropna=False
        )['Cost ($)'].sum(
        ).reset_index(
        ).assign(Year=yr_cal)
        
    cost_df_AUS = cost_df_region.groupby(['From-land-use', 'To-land-use', 'Cost-type'],
        )['Cost ($)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA', Year=yr_cal)
        
    # Save to csv
    pd.concat([cost_df_AUS, cost_df_region]
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_nonag2ag_cost_{yr_cal}.csv'), index=False)
        
    '''
    NoAg to Ag are currently all zeros, so we skip below calculation.
    '''
    valid_layers_transition = pd.MultiIndex.from_frame(
        cost_df_AUS[['From-land-use', 'Cost-type']].iloc[:1] # Only take the first row to avoid empty layers, since all costs are zero for now
    ).sort_values()
    
    cost_xr_stacked = cost_xr.stack({
        'layer': ['From-land-use', 'Cost-type']
    }).drop_vars('region').sel(layer=valid_layers_transition).compute()
    
    # Save valid layers 
    save2nc(cost_xr_stacked, os.path.join(path, f'xr_transition_nonag2ag_cost_{yr_cal}.nc'))

    return f"Non-agricultural to agricultural transition written for year {yr_cal}"



def write_area_transition_start_end(data: Data, path, yr_cal_end):

    yr_cal_start = data.YR_CAL_BASE
    real_area_r = xr.DataArray(data.REAL_AREA.astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})

    # Get the decision variables for the start year
    ag_dvar_base_mrj = tools.ag_mrj_to_xr(data, tools.lumap2ag_l_mrj(data.lumaps[yr_cal_start], data.lmmaps[yr_cal_start])
        ).assign_coords({'region': ('cell', data.REGION_NRM_NAME)}
        ).rename({'lu':'From-land-use', 'lm':'From-water-supply'}
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})

    ag_dvar_target_mrj = tools.ag_mrj_to_xr(
        data, tools.lumap2ag_l_mrj(data.lumaps[yr_cal_end], data.lmmaps[yr_cal_end])
        ).rename({'lu':'To-land-use', 'lm':'To-water-supply'}
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})

    non_ag_dvar_target_rk = tools.non_ag_rk_to_xr(
        data, data.non_ag_dvars[yr_cal_end]
        ).rename({'lu':'To-land-use'}
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})

    xr_ag2ag = ag_dvar_base_mrj * ag_dvar_target_mrj * real_area_r
    xr_ag2non_ag = ag_dvar_base_mrj * non_ag_dvar_target_rk * real_area_r

    xr_ag2ag     = add_all(xr_ag2ag,     ['From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use'])
    xr_ag2non_ag = add_all(xr_ag2non_ag, ['From-water-supply', 'From-land-use', 'To-land-use'])


    # ==================== Chunk Level Aggregation ====================
    '''
    Process both ag2ag and ag2non_ag transitions in a single loop to reduce memory usage.
        This is because the `xr_ag2ag` and `xr_ag2non_ag` are huge intermediate arrays that consume a lot of memory.
        By manually selecting each chunk, we can limit the size of the intermediate array.

    '''
    chunk_size = min(settings.WRITE_CHUNK_SIZE, data.NCELLS)
    transition_ag2ag_dfs = []
    transition_ag2non_ag_dfs = []

    for i in range(0, data.NCELLS, chunk_size):
        end_idx = min(i + chunk_size, data.NCELLS)
        cell_slice = slice(i, end_idx)

        # Process ag2ag chunk
        chunk_arr_ag2ag = xr_ag2ag.isel(cell=cell_slice).compute()
        transition_df_ag2ag = chunk_arr_ag2ag.groupby('region'
            ).sum(dim='cell'
            ).to_dataframe('Area (ha)'
            ).reset_index(
            ).groupby(['region', 'From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
            )['Area (ha)'
            ].sum(
            ).reset_index(
            ).query('abs(`Area (ha)`) > 0.01'
            ).assign(chunk_idx=i//chunk_size)
        transition_ag2ag_dfs.append(transition_df_ag2ag)

        # Process ag2non_ag chunk
        chunk_arr_ag2non_ag = xr_ag2non_ag.isel(cell=cell_slice).compute()
        transition_df_ag2non_ag = chunk_arr_ag2non_ag.groupby('region'
            ).sum(dim='cell'
            ).to_dataframe('Area (ha)'
            ).reset_index(
            ).groupby(['region', 'From-water-supply', 'From-land-use', 'To-land-use']
            )['Area (ha)'
            ].sum(
            ).reset_index(
            ).query('abs(`Area (ha)`) > 0.01'
            ).assign(chunk_idx=i//chunk_size)
        transition_ag2non_ag_dfs.append(transition_df_ag2non_ag)

    # Combine all chunks df for ag2ag
    transition_ag2ag = pd.concat(transition_ag2ag_dfs, ignore_index=True
        ).groupby(['region', 'From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
        )['Area (ha)'
        ].sum(
        ).reset_index()

    transition_ag2ag_AUS = transition_ag2ag.groupby(['From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
        )['Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Area (ha)`) > 1')  # Skip transitions under 1 ha at national level

    # Combine all chunks df for ag2non_ag
    transition_ag2non_ag = pd.concat(transition_ag2non_ag_dfs, ignore_index=True
        ).groupby(['region', 'From-water-supply', 'From-land-use', 'To-land-use']
        )['Area (ha)'
        ].sum(
        ).reset_index()

    transition_ag2non_ag_AUS = transition_ag2non_ag.groupby(['From-water-supply', 'From-land-use', 'To-land-use']
        )['Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Area (ha)`) > 1')  # Skip transitions under 1 ha at national level

    # Write the transition matrix to a csv file
    pd.concat([transition_ag2ag, transition_ag2ag_AUS]
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_matrix_ag2ag_start_end.csv'), index=False)

    pd.concat([transition_ag2non_ag, transition_ag2non_ag_AUS]
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_matrix_ag2non_ag_start_end.csv'), index=False)


    # ==================== Stack Array, Get Valid Layers for ag2ag ====================
    '''
    We manually loop through chunks to save stacked array to reduce memory usage.
        The materializing of stacked arrays requires a lot of memory.
    '''

    # Get valid data layers for ag2ag
    valid_layers_ag2ag = pd.MultiIndex.from_frame(
        transition_ag2ag_AUS[['From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']]
    ).sort_values()

    xr_ag2ag_stacked = xr_ag2ag.stack({
        'layer': ['From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
    }).sel(layer=valid_layers_ag2ag)

    # Materialize the filtered array by looping through chunks
    xr_ag2ag_filtered_array = xr.DataArray(
        np.zeros((data.NCELLS, len(valid_layers_ag2ag)), dtype=np.float32),
        coords={
            'cell': range(data.NCELLS),
            'layer': valid_layers_ag2ag
        }
    )

    for i in range(0, data.NCELLS, chunk_size):
        end_idx = min(i + chunk_size, data.NCELLS)
        cell_slice = slice(i, end_idx)
        xr_ag2ag_filtered_array[cell_slice, :] = xr_ag2ag_stacked.isel(cell=cell_slice)

    # Save the compact filtered array
    save2nc(xr_ag2ag_filtered_array, os.path.join(path, f'xr_transition_ag2ag_area_start_end.nc'))


    # ==================== Stack Array, Get Valid Layers for ag2non_ag ====================

    # Get valid data layers for ag2non_ag
    valid_layers_ag2non_ag = pd.MultiIndex.from_frame(
        transition_ag2non_ag_AUS[['From-water-supply', 'From-land-use', 'To-land-use']]
    ).sort_values()

    xr_ag2non_ag_stacked = xr_ag2non_ag.stack({
        'layer': ['From-water-supply', 'From-land-use', 'To-land-use']
    }).sel(layer=valid_layers_ag2non_ag)

    # Materialize the filtered array by looping through chunks
    xr_ag2non_ag_filtered_array = xr.DataArray(
        np.zeros((data.NCELLS, len(valid_layers_ag2non_ag)), dtype=np.float32),
        coords={
            'cell': range(data.NCELLS),
            'layer': valid_layers_ag2non_ag
        }
    )

    for i in range(0, data.NCELLS, chunk_size):
        end_idx = min(i + chunk_size, data.NCELLS)
        cell_slice = slice(i, end_idx)
        xr_ag2non_ag_filtered_array[cell_slice, :] = xr_ag2non_ag_stacked.isel(cell=cell_slice)

    # Save the compact filtered array
    save2nc(xr_ag2non_ag_filtered_array, os.path.join(path, f'xr_transition_area_ag2non_ag_start_end.nc'))
    
    # Record maximum cell magnitude for this transition period for later use in scaling the transition area in the visualization
    return (f"Area transition matrix written from year {data.YR_CAL_BASE} to {yr_cal_end}", {
        'transition_area': {
            'ag2ag':     (yr_cal_end, xr_ag2ag_filtered_array.max().item()),
            'ag2non_ag': (yr_cal_end, xr_ag2non_ag_filtered_array.max().item()),
        }
    })



def write_crosstab(data: Data, yr_cal, path):
    """Write out land-use and production data"""

    if yr_cal == data.YR_CAL_BASE:
        return "Skip land-use transition calculation for the base year."

    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]
    
    # Check if yr_cal_sim_pre meets the requirement
    assert yr_cal_sim_pre >= data.YR_CAL_BASE and yr_cal_sim_pre < yr_cal,\
        f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"

    lumap_pre = data.lumaps[yr_cal_sim_pre]
    lumap = data.lumaps[yr_cal]
    
    crosstab_region = pd.crosstab(lumap_pre,  [lumap, data.REGION_NRM_NAME], values=data.REAL_AREA, aggfunc=lambda x:x.sum(), margins = False
        ).unstack(
        ).reset_index(
        ).rename(
            columns={
                'row_0': 'From-land-use', 
                'NRM_NAME': 'region', 
                'col_0':'To-land-use', 
                'col_1': 'region',
                0: 'Area (ha)'
            }
        ).dropna(
        ).infer_objects(copy=False
        ).replace({'From-land-use': data.ALLLU2DESC, 'To-land-use': data.ALLLU2DESC})
    crosstab_AUS = crosstab_region.groupby(['From-land-use', 'To-land-use']        
        )['Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA')
        
    crosstab = pd.concat([crosstab_AUS, crosstab_region], ignore_index=True)
        
        
    switches = (crosstab.groupby(['region', 'From-land-use'])['Area (ha)'].sum() - crosstab.groupby(['region', 'To-land-use'])['Area (ha)'].sum()
        ).reset_index(
        ).rename(columns={'index':'Landuse'}
        ).query('abs(`Area (ha)`) > 100') # Skip switches under 100 ha
    
    
    crosstab['Year'] = yr_cal
    switches['Year'] = yr_cal

    crosstab.to_csv(os.path.join(path, f'crosstab-lumap_{yr_cal}.csv'), index=False)
    switches.to_csv(os.path.join(path, f'switches-lumap_{yr_cal}.csv'), index=False)

    return f"Land-use cross-tabulation and switches written for year {yr_cal}"




# ── GHG ──────────────────────────────────────────────────────────────────────

def write_ghg(data: Data, yr_cal: int, path: str):
    """Write all GHG emissions outputs to NetCDF and CSV files.

    Covers: total/limit summary, off-land commodity, agricultural land-use,
    non-agricultural land-use, agricultural management, land-use transition
    penalties, and cross-category sum.
    """
    yr_idx = yr_cal - data.YR_CAL_BASE

    # ==================== Total / Limit Summary ====================

    ghg_limits = 0 if settings.GHG_EMISSIONS_LIMITS == 'off' else data.GHG_TARGETS[yr_cal]
    if yr_cal >= data.YR_CAL_BASE + 1:
        ghg_emissions = data.prod_data[yr_cal]['GHG']
    else:
        ghg_emissions = (ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True) * data.ag_dvars[settings.SIM_YEARS[0]]).sum()
    pd.DataFrame({
        'Variable': ['GHG_EMISSIONS_LIMIT_TCO2e', 'GHG_EMISSIONS_TCO2e'],
        'Emissions (t CO2e)': [ghg_limits, ghg_emissions],
        'Year': yr_cal,
    }).to_csv(os.path.join(path, f'GHG_emissions_{yr_cal}.csv'), index=False)

    # ==================== Off-land Commodity ====================

    offland_ghg = data.OFF_LAND_GHG_EMISSION.query(f'YEAR == {yr_cal}').rename(columns={'YEAR': 'Year'})
    offland_ghg.to_csv(os.path.join(path, f'GHG_emissions_offland_commodity_{yr_cal}.csv'), index=False)

    # ==================== Agricultural Land-use ====================

    ag_g_xr = xr.Dataset(ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=False)
        ).rename({'dim_0': 'cell'})
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})

    mindex = pd.MultiIndex.from_tuples(ag_g_xr.data_vars.keys(), names=['GHG_source', 'lm', 'lu'])
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, 'variable')
    ag_g_rsmj = ag_g_xr.to_dataarray().assign_coords(mindex_coords).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}).unstack()
    ag_g_rsmj['GHG_source'] = ag_g_rsmj['GHG_source'].to_series().infer_objects(copy=False).replace(GHG_NAMES)

    ghg_e = ag_g_rsmj * ag_dvar_mrj
    ghg_e = add_all(ghg_e, ['lm', 'GHG_source', 'lu'])

    ghg_df_region = ghg_e.groupby('region'
        ).sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).assign(Year=yr_cal, Type='Agricultural Land-use'
        ).query('abs(`Value (t CO2e)`) > 1e-3')
    ghg_df_AUS = ghg_e.sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).assign(Year=yr_cal, Type='Agricultural Land-use', region='AUSTRALIA'
        ).query('abs(`Value (t CO2e)`) > 1e-3')
    pd.concat([ghg_df_AUS, ghg_df_region]
        ).infer_objects(copy=False
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).rename(columns={'lu': 'Land-use', 'lm': 'Water_supply', 'GHG_source': 'Source'}
        ).to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_landuse_{yr_cal}.csv'), index=False)

    valid_ghg_layers = pd.MultiIndex.from_frame(ghg_df_AUS[['lm', 'GHG_source', 'lu']]).sort_values()
    valid_layers_stack_ghg = ghg_e.stack(layer=['lm', 'GHG_source', 'lu']).sel(layer=valid_ghg_layers).drop_vars('region').compute()
    save2nc(valid_layers_stack_ghg, os.path.join(path, f'xr_GHG_ag_{yr_cal}.nc'))

    # ==================== Non-Agricultural Land-use ====================

    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    non_ag_g_rk = tools.non_ag_rk_to_xr(
        data,
        non_ag_ghg.get_ghg_matrix(data, ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True), data.lumaps[yr_cal])
    )

    xr_ghg_non_ag = non_ag_dvar_rk * non_ag_g_rk
    xr_ghg_non_ag = add_all(xr_ghg_non_ag, ['lu'])

    ghg_df_region = xr_ghg_non_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).assign(Year=yr_cal, Type='Non-Agricultural Land-use'
        ).query('abs(`Value (t CO2e)`) > 1e-3')
    ghg_df_AUS = xr_ghg_non_ag.sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).assign(Year=yr_cal, Type='Non-Agricultural Land-use', region='AUSTRALIA'
        ).query('abs(`Value (t CO2e)`) > 1e-3')
    pd.concat([ghg_df_AUS, ghg_df_region]
        ).rename(columns={'lu': 'Land-use'}
        ).to_csv(os.path.join(path, f'GHG_emissions_separate_no_ag_reduction_{yr_cal}.csv'), index=False)

    valid_non_ag_ghg_layers = pd.MultiIndex.from_frame(ghg_df_AUS[['lu']]).sort_values()
    if ghg_df_AUS['Value (t CO2e)'].abs().sum() < 1e-3:
        xr_ghg_non_ag_cat = xr.DataArray(
            np.zeros((1, data.NCELLS), dtype=np.float32),
            dims=['lu', 'cell'],
            coords={'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['lu'])
    else:
        xr_ghg_non_ag_cat = xr_ghg_non_ag.stack(layer=['lu']).sel(layer=valid_non_ag_ghg_layers).drop_vars('region').compute()
    save2nc(xr_ghg_non_ag_cat, os.path.join(path, f'xr_GHG_non_ag_{yr_cal}.nc'))

    # ==================== Agricultural Management ====================

    ag_man_dvar_mrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    ag_man_g_mrj = tools.am_mrj_to_xr(data, ag_ghg.get_agricultural_management_ghg_matrices(data, yr_idx))

    xr_ghg_ag_man = ag_man_dvar_mrj * ag_man_g_mrj
    xr_ghg_ag_man = add_all(xr_ghg_ag_man, ['lm', 'lu', 'am'])

    ghg_df_region = xr_ghg_ag_man.groupby('region'
        ).sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).assign(Year=yr_cal, Type='Agricultural Management'
        ).query('abs(`Value (t CO2e)`) > 1e-3')
    ghg_df_AUS = xr_ghg_ag_man.sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).assign(Year=yr_cal, Type='Agricultural Management', region='AUSTRALIA'
        ).query('abs(`Value (t CO2e)`) > 1e-3')
    pd.concat([ghg_df_AUS, ghg_df_region]
        ).infer_objects(copy=False
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).rename(columns={'lm': 'Water_supply', 'lu': 'Land-use', 'am': 'Agricultural Management Type'}
        ).to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_management_{yr_cal}.csv'), index=False)

    valid_am_ghg_layers = pd.MultiIndex.from_frame(ghg_df_AUS[['am', 'lm', 'lu']]).sort_values()
    if ghg_df_AUS['Value (t CO2e)'].abs().sum() < 1e-3:
        valid_layers_stack_am_ghg = xr.DataArray(
            np.zeros((1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['am', 'lm', 'lu', 'cell'],
            coords={'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['am', 'lm', 'lu'])
    else:
        valid_layers_stack_am_ghg = xr_ghg_ag_man.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_am_ghg_layers).drop_vars('region').compute()
    save2nc(valid_layers_stack_am_ghg, os.path.join(path, f'xr_GHG_ag_management_{yr_cal}.nc'))

    # ==================== Transition Penalty ====================

    transition_magnitudes = []
    if yr_cal != data.YR_CAL_BASE:
        simulated_year_list = sorted(list(data.lumaps.keys()))
        yr_idx_sim = simulated_year_list.index(yr_cal)
        yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]

        ghg_t_dict = ag_ghg.get_ghg_transition_emissions(data, data.lumaps[yr_cal_sim_pre], separate=True)
        ghg_t_smrj = xr.DataArray(
            np.stack(list(ghg_t_dict.values()), axis=0).astype(np.float32),
            dims=['Type', 'lm', 'cell', 'lu'],
            coords={
                'Type': list(ghg_t_dict.keys()),
                'lm': data.LANDMANS,
                'cell': range(data.NCELLS),
                'lu': data.AGRICULTURAL_LANDUSES
            }
        )

        xr_ghg_transition = ghg_t_smrj * ag_dvar_mrj
        xr_ghg_transition = add_all(xr_ghg_transition, ['lm', 'Type'])

        ghg_df_region = xr_ghg_transition.groupby('region'
            ).sum('cell'
            ).to_dataframe('Value (t CO2e)'
            ).reset_index(
            ).assign(Year=yr_cal
            ).query('abs(`Value (t CO2e)`) > 1e-3')
        ghg_df_AUS = xr_ghg_transition.sum('cell'
            ).to_dataframe('Value (t CO2e)'
            ).reset_index(
            ).assign(Year=yr_cal, region='AUSTRALIA'
            ).query('abs(`Value (t CO2e)`) > 1e-3')
        pd.concat([ghg_df_AUS, ghg_df_region]
            ).infer_objects(copy=False
            ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
            ).rename(columns={'lu': 'Land-use', 'lm': 'Water_supply'}
            ).to_csv(os.path.join(path, f'GHG_emissions_separate_transition_penalty_{yr_cal}.csv'), index=False)

        valid_transition_layers = pd.MultiIndex.from_frame(ghg_df_AUS[['Type', 'lm', 'lu']]).sort_values()
        transition_valid_layers = xr_ghg_transition.stack(layer=['Type', 'lm', 'lu']).sel(layer=valid_transition_layers).drop_vars('region').compute()
        save2nc(transition_valid_layers, os.path.join(path, f'xr_transition_GHG_{yr_cal}.nc'))
        transition_magnitudes = _get_mag(transition_valid_layers)

    # ==================== Sum (Ag + Am + NonAg + Transition) ====================

    # Ag: sum over GHG_source to remove source dim → (lm=['dry','irr'], lu, cell)
    ag_lus = [l for l in ghg_e.coords['lu'].values if l != 'ALL']
    ghg_sources = [s for s in ghg_e.coords['GHG_source'].values if s != 'ALL']
    ghg_pre_ag = ghg_e.sel(lm=['dry', 'irr'], GHG_source=ghg_sources, lu=ag_lus).sum('GHG_source')

    # Am: sum over am dim → (lm=['dry','irr'], lu, cell); same lu coords as Ag
    am_types = [a for a in xr_ghg_ag_man.coords['am'].values if a != 'ALL']
    ghg_pre_am = (xr_ghg_ag_man.sel(am=am_types, lm=['dry', 'irr'], lu=ag_lus).sum('am')
                  if am_types else xr.zeros_like(ghg_pre_ag))

    # NonAg: no water dim — assign lm='dry', fill lm='irr' with 0 → (lm=['dry','irr'], lu, cell)
    non_ag_lus = [l for l in xr_ghg_non_ag.coords['lu'].values if l != 'ALL']
    ghg_pre_nonag = (xr_ghg_non_ag.sel(lu=non_ag_lus)
                     .expand_dims('lm').assign_coords(lm=['dry'])
                     .reindex(lm=['dry', 'irr'], fill_value=0))

    # Transition: dims are (Type, lm, cell, lu) — lm/lu are already destination water/lu.
    # Sum over Type (other dim) → (lm=['dry','irr'], lu, cell). Zero for base year.
    if yr_cal != data.YR_CAL_BASE:
        trans_types = [t for t in xr_ghg_transition.coords['Type'].values if t != 'ALL']
        ghg_pre_transition = xr_ghg_transition.sel(Type=trans_types, lm=['dry', 'irr'], lu=ag_lus).sum('Type')
    else:
        ghg_pre_transition = xr.zeros_like(ghg_pre_ag)

    # Concat: ag+am+transition for ag LUs, nonag for nonag LUs
    sum_ghg_dry_irr = xr.concat([ghg_pre_ag + ghg_pre_am + ghg_pre_transition, ghg_pre_nonag], dim='lu')

    # Add ALL lm aggregate
    sum_ghg_all_lm = sum_ghg_dry_irr.sum('lm', keepdims=True).assign_coords(lm=['ALL'])
    sum_ghg = xr.concat([sum_ghg_all_lm, sum_ghg_dry_irr], dim='lm')

    # Add ALL lu aggregate
    sum_ghg_all_lu = sum_ghg.sum('lu', keepdims=True).assign_coords(lu=['ALL'])
    sum_ghg = xr.concat([sum_ghg_all_lu, sum_ghg], dim='lu')

    sum_ghg_stack = sum_ghg.stack(layer=['lm', 'lu']).drop_vars('region').compute()
    save2nc(sum_ghg_stack, os.path.join(path, f'xr_GHG_sum_{yr_cal}.nc'))

    magnitudes = {
        'ghg_emission': {
            'ag':         _get_mag(valid_layers_stack_ghg),
            'non_ag':     _get_mag(xr_ghg_non_ag_cat),
            'ag_man':     _get_mag(valid_layers_stack_am_ghg),
            'transition': transition_magnitudes,
            'sum':        _get_mag(sum_ghg_stack),
        }
    }
    return (f"GHG emissions written for year {yr_cal}", magnitudes)





# ── Water ────────────────────────────────────────────────────────────────────

def write_water(data: Data, yr_cal, path):
    """ Water yield is written to disk no matter if `WATER_LIMITS` is on or off. """

    yr_idx = yr_cal - data.YR_CAL_BASE
    region2code = {v: k for k, v in data.WATER_REGION_NAMES.items()}

    # Get the decision variables
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    non_ag_dvar_rj = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    am_dvar_mrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
        
    # Get water target and domestic use
    w_limit_inside_luto = xr.DataArray(
        np.array(list(data.WATER_YIELD_TARGETS.values()), dtype=np.float32),
        dims=['region_water'],
        coords={'region_water': list(data.WATER_YIELD_TARGETS.keys())}
    )
    domestic_water_use = xr.DataArray(
        np.array(list(data.WATER_USE_DOMESTIC.values()), dtype=np.float32),
        dims=['region_water'],
        coords={'region_water': list(data.WATER_USE_DOMESTIC.keys())}
    )

    # ==================== Get Water Yield without CCI ====================

    # Get water yield matrix
    if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on':
        ag_w_mrj = tools.ag_mrj_to_xr(
            data,
            ag_water.get_water_net_yield_matrices(data, yr_idx)
        )
        non_ag_w_rk = tools.non_ag_rk_to_xr(
            data,
            non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj.values, data.lumaps[yr_cal], yr_idx)
        )
        ag_man_w_mrj = tools.am_mrj_to_xr(  # Ag-man water yield only related to water requirement, that not affected by climate change
            data,
            ag_water.get_agricultural_management_water_matrices(data, yr_idx)
        )
    elif settings.WATER_CLIMATE_CHANGE_IMPACT == 'off':
        ag_w_mrj = tools.ag_mrj_to_xr(
            data,
            ag_water.get_water_net_yield_matrices(data, yr_idx, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
        )
        non_ag_w_rk = tools.non_ag_rk_to_xr(
            data,
            non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj.values, data.lumaps[yr_cal], yr_idx, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
        )
        ag_man_w_mrj = tools.am_mrj_to_xr(  # Ag-man water yield only related to water requirement, that not affected by climate change
            data,
            ag_water.get_agricultural_management_water_matrices(data, yr_idx)
        )
    else:
        raise ValueError("Invalid setting for WATER_CLIMATE_CHANGE_IMPACT, only 'on' or 'off' allowed.")

    # Calculate water net yield inside LUTO study region
    xr_ag_wny = ag_dvar_mrj * ag_w_mrj
    xr_non_ag_wny = non_ag_dvar_rj * non_ag_w_rk
    xr_am_wny = ag_man_w_mrj * am_dvar_mrj

    xr_ag_wny     = add_all(xr_ag_wny,     ['lm', 'lu'])
    xr_non_ag_wny = add_all(xr_non_ag_wny, ['lu'])
    xr_am_wny     = add_all(xr_am_wny,     ['lm', 'lu', 'am'])

    ag_wny = xr_ag_wny.groupby('region_water'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Land-use'
        ).infer_objects(copy=False
        ).replace({'region_water': data.WATER_REGION_NAMES})
    non_ag_wny = xr_non_ag_wny.groupby('region_water'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Non-Agricultural Land-use', lm='dry'
        ).infer_objects(copy=False
        ).replace({'region_water': data.WATER_REGION_NAMES})
    am_wny = xr_am_wny.groupby('region_water'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Management'
        ).infer_objects(copy=False
        ).replace({'region_water': data.WATER_REGION_NAMES})
    wny_inside_luto = pd.concat([ag_wny, non_ag_wny, am_wny], ignore_index=True
        ).assign(Year=yr_cal
        ).rename(columns={
            'region_water': 'Region',
            'lu':'Landuse',
            'am':'Agricultural Management',
            'lm':'Water Supply'}
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).dropna(axis=0, how='all')

    wny_inside_luto.to_csv(os.path.join(path, f'water_yield_separate_watershed_{yr_cal}.csv'), index=False)


    # ==================== Get Water Yield outside LUTO Study Region ====================
    wny_outside_luto_study_area = xr.DataArray(
        np.array(list(data.WATER_OUTSIDE_LUTO_BY_CCI.loc[data.YR_CAL_BASE].to_dict().values()), dtype=np.float32),
        dims=['region_water'],
        coords={'region_water': list(data.WATER_REGION_INDEX_R.keys())},
    )


    # ==================== Get Water Yield Change (delta) under CCI ====================

    # Get CCI matrix
    if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on':
        ag_w_mrj_base = tools.ag_mrj_to_xr(data, ag_water.get_water_net_yield_matrices(data, 0))
        ag_w_mrj_base = add_all(ag_w_mrj_base, ['lm'])
        wny_outside_luto_study_area_base = np.array(list(data.WATER_OUTSIDE_LUTO_BY_CCI.loc[data.YR_CAL_BASE].to_dict().values()))
    elif settings.WATER_CLIMATE_CHANGE_IMPACT == 'off':
        ag_w_mrj_base = tools.ag_mrj_to_xr(data, ag_water.get_water_net_yield_matrices(data, 0, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR))
        ag_w_mrj_base = add_all(ag_w_mrj_base, ['lm'])
        wny_outside_luto_study_area_base = np.array(list(data.WATER_OUTSIDE_LUTO_HIST.values()))

    ag_w_mrj_CCI = ag_w_mrj - ag_w_mrj_base
    wny_outside_luto_study_area_CCI = wny_outside_luto_study_area - wny_outside_luto_study_area_base



    # Calculate water net yield (delta) under CCI; 
    #   we use BASE_YEAR (2010) dvar_mrj to calculate CCI, 
    #   because the CCI calculated with base year (previouse year) 
    #   dvar_mrj includes wny From-land-use change
    xr_ag_dvar_BASE = tools.ag_mrj_to_xr(data, data.AG_L_MRJ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME))
    xr_ag_dvar_BASE = add_all(xr_ag_dvar_BASE, ['lm'])

    xr_ag_wny_CCI = xr_ag_dvar_BASE * ag_w_mrj_CCI


    # Get the CCI impact (delta)
    CCI_impact = (
        xr_ag_wny_CCI.groupby('region_water').sum(['cell','lm', 'lu']) 
        + wny_outside_luto_study_area_CCI
    )

    # ==================== Organise Water Yield Components ====================

    # Water net yield for watershed regions
    wny_inside_luto_sum = wny_inside_luto\
        .query('`Water Supply` != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby('Region')[['Water Net Yield (ML)']]\
        .sum()
    wny_inside_luto_sum = xr.DataArray(
        wny_inside_luto_sum['Water Net Yield (ML)'].values.astype(np.float32),
        dims=['region_water'],
        coords={'region_water': [region2code[i] for i in wny_inside_luto_sum.index.values]}
    )
    wny_watershed_sum = wny_inside_luto_sum + wny_outside_luto_study_area - domestic_water_use  # CCI delta already include in the wny_inside_luto_sum

    w_limit_region = w_limit_inside_luto + wny_outside_luto_study_area - domestic_water_use     # CCI delta already include in the w_limit_inside_luto

    water_other_records = xr.Dataset(
            {   
                'Water yield inside LUTO (ML)': wny_inside_luto_sum,
                'Water yield outside LUTO (ML)': wny_outside_luto_study_area,
                'Climate Change Impact (ML)': CCI_impact,
                'Domestic Water Use (ML)': domestic_water_use,
                'Water Net Yield (ML)': wny_watershed_sum,
                'Water Yield Limit (ML)': w_limit_region,
            },
        ).to_dataframe(
        ).reset_index(
        ).rename(columns={'region_water': 'Region'}
        ).infer_objects(copy=False
        ).replace({'Region': data.WATER_REGION_NAMES}
        ).assign(Year=yr_cal)
        
    water_other_records.to_csv(os.path.join(path, f'water_yield_limits_and_public_land_{yr_cal}.csv'), index=False)

    # Water yield for NRM region (use add_all'd arrays so Water Supply="ALL" is included)
    ag_wny = xr_ag_wny.groupby('region_NRM'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Land-use'
        ).infer_objects(copy=False
        ).replace({'region_NRM': data.WATER_REGION_NAMES})
    non_ag_wny = xr_non_ag_wny.groupby('region_NRM'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Non-Agricultural Land-use', lm='dry'
        ).infer_objects(copy=False
        ).replace({'region_NRM': data.WATER_REGION_NAMES})
    am_wny = xr_am_wny.groupby('region_NRM'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Management'
        ).infer_objects(copy=False
        ).replace({'region_NRM': data.WATER_REGION_NAMES})
    wny_NRM = pd.concat([ag_wny, non_ag_wny, am_wny], ignore_index=True
        ).assign(Year=yr_cal
        ).rename(columns={
            'region_water': 'Region',
            'lu':'Landuse',
            'am':'Agricultural Management',
            'lm':'Water Supply'}
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).dropna(axis=0, how='all')

    wny_NRM.to_csv(os.path.join(path, f'water_yield_separate_NRM_{yr_cal}.csv'), index=False)


    xr_ag_wny_cat = xr_ag_wny.stack(layer=['lm', 'lu']).drop_vars(['region_water', 'region_NRM']).compute()
    xr_non_ag_wny_cat = xr_non_ag_wny.stack(layer=['lu']).drop_vars(['region_water', 'region_NRM']).compute()
    xr_am_wny_cat = xr_am_wny.stack(layer=['am', 'lm', 'lu']).drop_vars(['region_water', 'region_NRM']).compute()

    save2nc(xr_ag_wny_cat, os.path.join(path, f'xr_water_yield_ag_{yr_cal}.nc'))
    save2nc(xr_non_ag_wny_cat, os.path.join(path, f'xr_water_yield_non_ag_{yr_cal}.nc'))
    save2nc(xr_am_wny_cat, os.path.join(path, f'xr_water_yield_ag_management_{yr_cal}.nc'))

    # --- Sum water yield (Ag + Am + NonAg) ---
    ag_lus = [lu for lu in xr_ag_wny.coords['lu'].values if lu != 'ALL']
    nonag_lus = [lu for lu in xr_non_ag_wny.coords['lu'].values if lu != 'ALL']
    am_non_all = [am for am in xr_am_wny.coords['am'].values if am != 'ALL']

    raw_wny_ag = xr_ag_wny.sel(lm=['dry', 'irr'], lu=ag_lus)
    am_sum_wny = xr_am_wny.sel(am=am_non_all, lm=['dry', 'irr']).sum('am').sel(lu=ag_lus)
    nonag_as_dry = (
        xr_non_ag_wny.sel(lu=nonag_lus)
        .expand_dims('lm').assign_coords(lm=['dry'])
        .reindex(lm=['dry', 'irr'], fill_value=0)
    )

    ag_plus_am_wny = raw_wny_ag + am_sum_wny
    sum_wny_dry_irr = xr.concat([ag_plus_am_wny, nonag_as_dry], dim='lu')

    sum_wny_all_lm = sum_wny_dry_irr.sum('lm', keepdims=True).assign_coords(lm=['ALL'])
    sum_wny = xr.concat([sum_wny_all_lm, sum_wny_dry_irr], dim='lm')

    sum_wny_all_lu = sum_wny.sum('lu', keepdims=True).assign_coords(lu=['ALL'])
    sum_wny = xr.concat([sum_wny_all_lu, sum_wny], dim='lu')

    xr_sum_wny_cat = sum_wny.stack(layer=['lm', 'lu']).drop_vars(['region_water', 'region_NRM']).compute()
    save2nc(xr_sum_wny_cat, os.path.join(path, f'xr_water_yield_sum_{yr_cal}.nc'))


    # ==================== Write Original Targets for Relaxed Watershed Regions ====================
    water_relaxed_region_raw_targets = pd.DataFrame(
        [[k, v, data.WATER_REGION_NAMES[k]] for k, v in data.WATER_RELAXED_REGION_RAW_TARGETS.items()], 
        columns=['Region Id', 'Target', 'Region Name']
    )
    water_relaxed_region_raw_targets['Year'] = yr_cal
    water_relaxed_region_raw_targets.to_csv(os.path.join(path, f'water_yield_relaxed_region_raw_{yr_cal}.csv'), index=False)

    return (
        f"Water yield data written for year {yr_cal}",
        {
            'water_yield': {
                'ag': (xr_ag_wny_cat.min().item(), xr_ag_wny_cat.max().item()),
                'non_ag': (xr_non_ag_wny_cat.min().item(), xr_non_ag_wny_cat.max().item()),
                'am': (xr_am_wny_cat.min().item(), xr_am_wny_cat.max().item()),
                'sum': (xr_sum_wny_cat.min().item(), xr_sum_wny_cat.max().item()),
            }
        }
    )



# ── Biodiversity ─────────────────────────────────────────────────────────────

def write_biodiversity_quality_scores(data: Data, yr_cal, path):
    ''' Biodiversity overall quality scores are always written to disk. '''

    # Set up
    yr_idx_previouse = sorted(data.lumaps.keys()).index(yr_cal) - 1
    yr_cal_previouse = sorted(data.lumaps.keys())[yr_idx_previouse]
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the biodiversity scores b_mrj
    bio_ag_priority_mrj =  tools.ag_mrj_to_xr(data, ag_biodiversity.get_bio_quality_score_mrj(data))   
    bio_am_priority_amrj = tools.am_mrj_to_xr(data, ag_biodiversity.get_ag_mgt_biodiversity_matrices(data, bio_ag_priority_mrj.values, yr_idx))
    bio_non_ag_priority_rk = tools.non_ag_rk_to_xr(data, non_ag_biodiversity.get_breq_matrix(data,bio_ag_priority_mrj.values, data.lumaps[yr_cal_previouse]))

    if yr_cal == data.YR_CAL_BASE: # this means now is the base year, hence no ag-man and non-ag applied
        bio_am_priority_amrj *= 0.0
        bio_non_ag_priority_rk *= 0.0


    # Get the decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    ag_mam_dvar_mrj =  tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))


    # Calculate the biodiversity scores under pre-1750 conditions
    base_yr_score = bio_ag_priority_mrj.sel(lu='Unallocated - natural land', lm='dry').sum().item()
    
    # Calculate xarray biodiversity scores
    xr_priority_ag = ag_dvar_mrj * bio_ag_priority_mrj
    xr_priority_non_ag = non_ag_dvar_rk * bio_non_ag_priority_rk
    xr_priority_am = ag_mam_dvar_mrj * bio_am_priority_amrj
    xr_priority_all = xr.concat(
        [   xr_priority_ag.sum(dim=['lm', 'lu']).expand_dims({'Type': ['ag']}),
            xr_priority_non_ag.sum(dim=['lu']).expand_dims({'Type': ['non-ag']}),
            xr_priority_am.sum(dim=['am', 'lu', 'lm']).expand_dims({'Type': ['ag-man']})
        ], dim='Type'
    )
    
    xr_priority_ag = add_all(xr_priority_ag, dims=['lm', 'lu'])
    xr_priority_non_ag = add_all(xr_priority_non_ag, dims=['lu'])
    xr_priority_am = add_all(xr_priority_am, dims=['am', 'lm', 'lu'])
    xr_priority_all = add_all(xr_priority_all, dims=['Type'])
    
    priority_ag_df, priority_ag_df_AUS = _bio_to_region_and_aus_df(
        xr_priority_ag,
        group_dims=['region', 'lm', 'lu'],
        value_name='Area Weighted Score (ha)',
        base_score=base_yr_score,
        yr_cal=yr_cal
    )
    priority_non_ag_df, priority_non_ag_df_AUS = _bio_to_region_and_aus_df(
        xr_priority_non_ag,
        group_dims=['region', 'lu'],
        value_name='Area Weighted Score (ha)',
        base_score=base_yr_score,
        yr_cal=yr_cal
    )
    priority_am_df, priority_am_df_AUS = _bio_to_region_and_aus_df(
        xr_priority_am,
        group_dims=['region', 'am', 'lm', 'lu'],
        value_name='Area Weighted Score (ha)',
        base_score=base_yr_score,
        yr_cal=yr_cal
    )
    priority_all_df, priority_all_df_AUS = _bio_to_region_and_aus_df(
        xr_priority_all,
        group_dims=['region', 'Type'],
        value_name='Area Weighted Score (ha)',
        base_score=base_yr_score,
        yr_cal=yr_cal
    )
    
    # Create zeros values to fill the first row for am/non-ag if their df is empty
    if priority_non_ag_df.empty:
        priority_non_ag_df = pd.DataFrame({
            'region': ['AUSTRALIA'],
            'Year': [yr_cal],
            'lu': ['Environmental Plantings'],
            'Area Weighted Score (ha)': [0.0],
            'Relative_Contribution_Percentage': [0.0]
        })
    if priority_am_df.empty:
        priority_am_df = pd.DataFrame({
            'region': ['AUSTRALIA'],
            'Year': [yr_cal],
            'am': ['ALL'],
            'lm': ['dry'],
            'lu': ['Apples'],
            'Area Weighted Score (ha)': [0.0],
            'Relative_Contribution_Percentage': [0.0]
        })


    # Save the biodiversity scores
    pd.concat([
        priority_ag_df.assign(Type = "Agricultural Land-use"), 
        priority_non_ag_df.assign(Type = "Non-Agricultural Land-use"), 
        priority_am_df.assign(Type = "Agricultural Management"), 
        ], axis=0
        ).rename(columns={
            'lu':'Landuse',
            'lm':'Water_supply',
            'am':'Agricultural Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Base Year Level (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'biodiversity_overall_priority_scores_{yr_cal}.csv'), index=False)
        
    priority_all_df.rename(columns={
        'lu':'Landuse',
        'lm':'Water_supply',
        'am':'Agricultural Management',
        'Relative_Contribution_Percentage':'Contribution Relative to Base Year Level (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'biodiversity_overall_priority_scores_all_{yr_cal}.csv'), index=False)
        
        
        
    

    # ==================== Ag Valid Layers ====================
    valid_ag_layers = pd.MultiIndex.from_frame(priority_ag_df_AUS[['lm', 'lu']]).sort_values()
    valid_layers_stack_ag = xr_priority_ag.stack(layer=['lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()

    # ==================== Non-Ag Valid Layers ====================
    valid_non_ag_layers = pd.MultiIndex.from_frame(priority_non_ag_df_AUS[['lu']]).sort_values()

    if priority_non_ag_df_AUS['Area Weighted Score (ha)'].abs().sum() < 1:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, data.NCELLS), dtype=np.float32),
            dims=['lu', 'cell'],
            coords={'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['lu'])
    else:
        valid_layers_stack_non_ag = xr_priority_non_ag.stack(layer=['lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()

    # ==================== Ag Management Valid Layers ====================
    valid_am_layers = pd.MultiIndex.from_frame(priority_am_df_AUS[['am', 'lm', 'lu']]).sort_values()

    if priority_am_df_AUS['Area Weighted Score (ha)'].abs().sum() < 1:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['am', 'lm', 'lu', 'cell'],
            coords={'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['am', 'lm', 'lu'])
    else:
        valid_layers_stack_am = xr_priority_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()
        
    # Quality 'All' layer (for the overall priority score)
    valid_all_layers = pd.MultiIndex.from_frame(priority_all_df_AUS[['Type']]).sort_values()
    valid_layers_stack_all = xr_priority_all.stack(layer=['Type']).sel(layer=valid_all_layers).drop_vars('region').compute()
    

    save2nc(valid_layers_stack_ag,     os.path.join(path, f'xr_biodiversity_overall_priority_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_overall_priority_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am,     os.path.join(path, f'xr_biodiversity_overall_priority_ag_management_{yr_cal}.nc'))
    save2nc(valid_layers_stack_all,    os.path.join(path, f'xr_biodiversity_overall_priority_all_{yr_cal}.nc'))
    

    magnitudes = {
        'bio_quality': {
            'ag':     _get_mag(valid_layers_stack_ag),
            'non_ag': _get_mag(valid_layers_stack_non_ag),
            'am':     _get_mag(valid_layers_stack_am),
            'all':    _get_mag(valid_layers_stack_all),
        }
    }
    return (f"Biodiversity overall priority scores written for year {yr_cal}", magnitudes)



def write_biodiversity_GBF2_scores(data: Data, yr_cal, path):
    ''' Biodiversity GBF2 only being written to disk when `BIODIVERSITY_TARGET_GBF_2` is not 'off' '''

    # Do nothing if biodiversity limits are off and no need to report
    if settings.BIODIVERSITY_TARGET_GBF_2 == 'off':
        return 'Skipped: Biodiversity GBF2 scores not written as `BIODIVERSITY_TARGET_GBF_2` is set to "off"'

        
    # Unpack the ag managements and land uses
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get the priority degraded areas 
    GBF2_MASK_area_ha = ag_biodiversity.get_GBF2_MASK_area(data)
    
    # Get the priority degraded areas score
    priority_degraded_area_score_r = xr.DataArray(
        GBF2_MASK_area_ha.astype(np.float32),
        dims=['cell'],
        coords={'cell':range(data.NCELLS)}
    )

    # Get the impacts of each ag/non-ag/am to vegetation matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ajr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]).astype(np.float32),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS)}
    ).unstack()

    # Get the total area of the priority degraded areas
    total_priority_degraded_area = GBF2_MASK_area_ha.sum()

    # Calculate xarray biodiversity GBF2 scores
    xr_gbf2_ag = priority_degraded_area_score_r * ag_impact_j * ag_dvar_mrj
    xr_gbf2_non_ag = priority_degraded_area_score_r * non_ag_impact_k * non_ag_dvar_rk
    xr_gbf2_am = priority_degraded_area_score_r * am_impact_ajr * am_dvar_amrj

    xr_gbf2_ag     = add_all(xr_gbf2_ag,     ['lm', 'lu'])
    xr_gbf2_non_ag = add_all(xr_gbf2_non_ag, ['lu'])
    xr_gbf2_am     = add_all(xr_gbf2_am,     ['lm', 'lu', 'am'])

    GBF2_score_ag, GBF2_score_ag_AUS = _bio_to_region_and_aus_df(
        xr_gbf2_ag, group_dims=['region', 'lm', 'lu'],
        value_name='Area Weighted Score (ha)', base_score=total_priority_degraded_area, yr_cal=yr_cal)
    GBF2_score_non_ag, GBF2_score_non_ag_AUS = _bio_to_region_and_aus_df(
        xr_gbf2_non_ag, group_dims=['region', 'lu'],
        value_name='Area Weighted Score (ha)', base_score=total_priority_degraded_area, yr_cal=yr_cal)
    GBF2_score_am, GBF2_score_am_AUS = _bio_to_region_and_aus_df(
        xr_gbf2_am, group_dims=['region', 'am', 'lm', 'lu'],
        value_name='Area Weighted Score (ha)', base_score=total_priority_degraded_area, yr_cal=yr_cal)

    GBF2_score_ag = GBF2_score_ag.assign(Type='Agricultural Land-use')
    GBF2_score_non_ag = GBF2_score_non_ag.assign(Type='Non-Agricultural Land-use')
    GBF2_score_am = GBF2_score_am.assign(Type='Agricultural Management')
        
    # Fill nan to empty dataframes
    if GBF2_score_ag.empty:
        GBF2_score_ag.loc[0] = 0
        GBF2_score_ag = GBF2_score_ag.astype({'Type':str, 'lu':str,'Year':'int'})
        GBF2_score_ag.loc[0, ['Type', 'lu' ,'Year']] = ['Agricultural Land-use', 'Apples', yr_cal]

    if GBF2_score_non_ag.empty:
        GBF2_score_non_ag.loc[0] = 0
        GBF2_score_non_ag = GBF2_score_non_ag.astype({'Type':str, 'lu':str,'Year':'int'})
        GBF2_score_non_ag.loc[0, ['Type', 'lu' ,'Year']] = ['Agricultural Management', 'Apples', yr_cal]

    if GBF2_score_am.empty:
        GBF2_score_am.loc[0] = 0
        GBF2_score_am = GBF2_score_am.astype({'Type':str, 'lu':str,'Year':'int'})
        GBF2_score_am.loc[0, ['Type', 'lu' ,'Year']] = ['Non-Agricultural Land-use', 'Environmental Plantings', yr_cal]
        
    # Save to disk  
    df = pd.concat([
            GBF2_score_ag,
            GBF2_score_non_ag,
            GBF2_score_am], axis=0
        ).assign( Priority_Target=(data.get_GBF2_target_for_yr_cal(yr_cal) / total_priority_degraded_area) * 100,
        ).rename(columns={
            'lu':'Landuse',
            'lm':'Water_supply',
            'am':'Agricultural Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)',
            'Priority_Target':'Priority Target (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        )
    df.to_csv(os.path.join(path, f'biodiversity_GBF2_priority_scores_{yr_cal}.csv'), index=False)


    # ------------------------- Vectorize GBF2 mask to GeoJSON -------------------------
    
    geojson_js_path = f'{data.path}/DATA_REPORT/data/geo/biodiversity_GBF2_mask.js'
    os.makedirs(os.path.dirname(geojson_js_path), exist_ok=True)
    if not os.path.exists(geojson_js_path):
        mask_2d_da = arr_to_xr(data, data.BIO_GBF2_MASK)
        mask_2d_np = np.where(np.isnan(mask_2d_da.values), 0, mask_2d_da.values).astype(np.uint8)

        # Vectorize using rasterio.features.shapes with the model's CRS and transform
        transform = data.GEO_META['transform']
        crs = data.GEO_META['crs']
        pixel_area = abs(transform.a * transform.e)
        min_area = 10 * pixel_area  # drop isolated patches smaller than 10 pixels
        polygons = [
            shape(geom)
            for geom, val in rasterio.features.shapes(mask_2d_np, transform=transform)
            if val == 1 and shape(geom).area >= min_area
        ]

        # Dissolve, smooth in projected CRS (EPSG:3577 Australian Albers, metres), then simplify
        # smooth_d in metres: approx 2× the pixel width converted from degrees to metres
        smooth_d = abs(transform.a) * 111_000 * 2  # 1 degree ≈ 111 km
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs).dissolve().to_crs('EPSG:3577')
        gdf['geometry'] = gdf.buffer(smooth_d).buffer(-smooth_d)
        gdf = gdf.to_crs('EPSG:4326')
        gdf['geometry'] = gdf.simplify(tolerance=0.05, preserve_topology=True)

        # Write as a JS window variable directly to VUE_modules; copytree carries it into DATA_REPORT
        geojson_dict = json.loads(gdf.to_json())
        with open(geojson_js_path, 'w', encoding='utf-8') as f:
            f.write(f'window.BIO_GBF2_MASK = {json.dumps(geojson_dict)};\n')



    # ------------------------- Stack array, get valid layers -------------------------

    # ---- Ag valid layers ----
    valid_ag_layers = pd.MultiIndex.from_frame(GBF2_score_ag_AUS[['lm', 'lu']]).sort_values()
    valid_layers_stack_ag = xr_gbf2_ag.stack(layer=['lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()

    # ---- Non-ag valid layers ----
    valid_non_ag_layers = pd.MultiIndex.from_frame(GBF2_score_non_ag_AUS[['lu']]).sort_values()

    if GBF2_score_non_ag_AUS.empty:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, data.NCELLS), dtype=np.float32),
            dims=['lu', 'cell'],
            coords={'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['lu'])

    else:
        valid_layers_stack_non_ag = xr_gbf2_non_ag.stack(layer=['lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()

    # ---- Ag management valid layers ----
    valid_am_layers = pd.MultiIndex.from_frame(GBF2_score_am_AUS[['am', 'lm', 'lu']]).sort_values()

    if GBF2_score_am_AUS.empty:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['am', 'lm', 'lu', 'cell'],
            coords={'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['am', 'lm', 'lu'])

    else:
        valid_layers_stack_am = xr_gbf2_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()

    # min/max should calculated using array without appending mosaic layers
    save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF2_priority_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF2_priority_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF2_priority_ag_management_{yr_cal}.nc'))

    # --- Sum GBF2 (Ag + Am + NonAg) — Type dimension matches Quality Sum pattern ---
    xr_gbf2_all = xr.concat(
        [   xr_gbf2_ag.sum(dim=['lm', 'lu']).expand_dims({'Type': ['ag']}),
            xr_gbf2_non_ag.sum(dim=['lu']).expand_dims({'Type': ['non-ag']}),
            xr_gbf2_am.sum(dim=['am', 'lm', 'lu']).expand_dims({'Type': ['ag-man']})
        ], dim='Type'
    )
    xr_gbf2_all = add_all(xr_gbf2_all, dims=['Type'])
    xr_sum_gbf2_cat = xr_gbf2_all.stack(layer=['Type']).drop_vars('region').compute()
    save2nc(xr_sum_gbf2_cat, os.path.join(path, f'xr_biodiversity_GBF2_priority_sum_{yr_cal}.nc'))

    return f"Biodiversity GBF2 priority scores written for year {yr_cal}"



def write_biodiversity_GBF3_NVIS_scores(data: Data, yr_cal: int, path) -> None:
    ''' Biodiversity GBF3 (NVIS) only being written to disk when `BIODIVERSITY_TARGET_GBF_3_NVIS` is not 'off' '''

    # Do nothing if biodiversity limits are off and no need to report
    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS == 'off':
        return "Skipped: Biodiversity GBF3 NVIS scores not written as `BIODIVERSITY_TARGET_GBF_3_NVIS` is set to 'off'"


    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get vegetation matrices for the year
    vegetation_score_vr = xr.DataArray(
        ag_biodiversity.get_GBF3_NVIS_matrices_vr(data).astype(np.float32),
        dims=['group','cell'],
        coords={'group':list(data.BIO_GBF3_NVIS_ID2DESC.values()),  'cell':range(data.NCELLS)}
    ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS), 'group': 1})

    # Get the impacts of each ag/non-ag/am to vegetation matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]).astype(np.float32),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS)}
    ).unstack()

    # Get the base year biodiversity scores
    veg_base_score_score = pd.DataFrame({
            'group': data.BIO_GBF3_NVIS_ID2DESC.values(),
            'BASE_OUTSIDE_SCORE': data.BIO_GBF3_NVIS_BASELINE_OUTSIDE_LUTO,
            'BASE_TOTAL_SCORE': data.BIO_GBF3_NVIS_BASELINE_AUSTRALIA,
            'TARGET_INSIDE_SCORE': data.get_GBF3_NVIS_limit_score_inside_LUTO_by_yr(yr_cal)}
        ).eval('Target_by_Percent = (TARGET_INSIDE_SCORE + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100')

    # Calculate xarray biodiversity GBF3 scores
    xr_gbf3_ag = vegetation_score_vr * ag_impact_j * ag_dvar_mrj
    xr_gbf3_am = vegetation_score_vr * am_impact_amr * am_dvar_amrj
    xr_gbf3_non_ag = vegetation_score_vr * non_ag_impact_k * non_ag_dvar_rk

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_gbf3_ag     = add_all(xr_gbf3_ag,     ['lm', 'lu'])
    xr_gbf3_non_ag = add_all(xr_gbf3_non_ag, ['lu'])
    xr_gbf3_am     = add_all(xr_gbf3_am,     ['lm', 'lu', 'am'])

    # Regional level aggregation
    GBF3_score_ag_region = xr_gbf3_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(veg_base_score_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal)

    GBF3_score_am_region = xr_gbf3_am.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(veg_base_score_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)

    GBF3_score_non_ag_region = xr_gbf3_non_ag.groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(veg_base_score_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)

    # Australia level aggregation
    GBF3_score_ag_AUS = xr_gbf3_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(veg_base_score_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    GBF3_score_am_AUS = xr_gbf3_am.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(veg_base_score_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')

    GBF3_score_non_ag_AUS = xr_gbf3_non_ag.sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(veg_base_score_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    # Combine regional and Australia level data
    GBF3_score_ag = pd.concat([GBF3_score_ag_region, GBF3_score_ag_AUS], axis=0)
    GBF3_score_am = pd.concat([GBF3_score_am_region, GBF3_score_am_AUS], axis=0)
    GBF3_score_non_ag = pd.concat([GBF3_score_non_ag_region, GBF3_score_non_ag_AUS], axis=0)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    veg_base_score_score = veg_base_score_score.assign(
            Type='Outside LUTO study area',
            Year=yr_cal,
            lu='Outside LUTO study area',
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    pd.concat([
        GBF3_score_ag,
        GBF3_score_am,
        GBF3_score_non_ag,
        veg_base_score_score],axis=0
        ).rename(columns={
            'lu':'Landuse',
            'lm':'Water_supply',
            'am':'Agricultural Management',
            'group':'Vegetation Group',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF3_NVIS_scores_{yr_cal}.csv'), index=False)

    # ------------------------- Stack array, get valid layers -------------------------

    # ---- Ag valid layers (include group in layer so each parallel task gets 1D cell data) ----
    valid_ag_layers = pd.MultiIndex.from_frame(GBF3_score_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'lm', 'lu']]).sort_values()
    valid_layers_stack_ag = xr_gbf3_ag.stack(layer=['group', 'lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()

    # ---- Non-ag valid layers ----
    valid_non_ag_layers = pd.MultiIndex.from_frame(GBF3_score_non_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'lu']]).sort_values()

    if GBF3_score_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, 1, data.NCELLS), dtype=np.float32),
            dims=['group', 'lu', 'cell'],
            coords={'group': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['group', 'lu'])

    else:
        valid_layers_stack_non_ag = xr_gbf3_non_ag.stack(layer=['group', 'lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()

    # ---- Ag management valid layers ----
    valid_am_layers = pd.MultiIndex.from_frame(GBF3_score_am_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'am', 'lm', 'lu']]).sort_values()

    if GBF3_score_am_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['group', 'am', 'lm', 'lu', 'cell'],
            coords={'group': ['ALL'], 'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['group', 'am', 'lm', 'lu'])

    else:
        valid_layers_stack_am = xr_gbf3_am.stack(layer=['group', 'am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()

    # min/max should calculated using array without appending mosaic layers
    save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF3_NVIS_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF3_NVIS_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF3_NVIS_ag_management_{yr_cal}.nc'))

    return f"Biodiversity GBF3 scores written for year {yr_cal}"



def write_biodiversity_GBF3_IBRA_scores(data: Data, yr_cal: int, path) -> None:
    ''' Biodiversity GBF3 (IBRA) only being written to disk when `BIODIVERSITY_TARGET_GBF_3_IBRA` is not 'off' '''

    # Do nothing if biodiversity limits are off and no need to report
    if settings.BIODIVERSITY_TARGET_GBF_3_IBRA == 'off':
        return "Skipped: Biodiversity GBF3 IBRA scores not written as `BIODIVERSITY_TARGET_GBF_3_IBRA` is set to 'off'"


    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get IBRA bioregion matrices for the year
    bioregion_score_vr = xr.DataArray(
        ag_biodiversity.get_GBF3_IBRA_matrices_vr(data).astype(np.float32),
        dims=['group','cell'],
        coords={'group':list(data.BIO_GBF3_IBRA_ID2DESC.values()),  'cell':range(data.NCELLS)}
    ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS), 'group': 1})

    # Get the impacts of each ag/non-ag/am to bioregion matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]).astype(np.float32),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS)}
    ).unstack()

    # Get the base year biodiversity scores
    bioregion_base_score = pd.DataFrame({
            'group': data.BIO_GBF3_IBRA_ID2DESC.values(),
            'BASE_OUTSIDE_SCORE': data.BIO_GBF3_IBRA_BASELINE_OUTSIDE_LUTO,
            'BASE_TOTAL_SCORE': data.BIO_GBF3_IBRA_BASELINE_AUSTRALIA,
            'TARGET_INSIDE_SCORE': data.get_GBF3_IBRA_limit_score_inside_LUTO_by_yr(yr_cal)}
        ).eval('Target_by_Percent = (TARGET_INSIDE_SCORE + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100')

    # Calculate xarray biodiversity GBF3 IBRA scores
    xr_gbf3_ibra_ag = bioregion_score_vr * ag_impact_j * ag_dvar_mrj
    xr_gbf3_ibra_am = bioregion_score_vr * am_impact_amr * am_dvar_amrj
    xr_gbf3_ibra_non_ag = bioregion_score_vr * non_ag_impact_k * non_ag_dvar_rk

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_gbf3_ibra_ag     = add_all(xr_gbf3_ibra_ag,     ['lm', 'lu'])
    xr_gbf3_ibra_non_ag = add_all(xr_gbf3_ibra_non_ag, ['lu'])
    xr_gbf3_ibra_am     = add_all(xr_gbf3_ibra_am,     ['lm', 'lu', 'am'])

    # Regional level aggregation
    GBF3_IBRA_score_ag_region = xr_gbf3_ibra_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(bioregion_base_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal)

    GBF3_IBRA_score_am_region = xr_gbf3_ibra_am.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(bioregion_base_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)

    GBF3_IBRA_score_non_ag_region = xr_gbf3_ibra_non_ag.groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(bioregion_base_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)

    # Australia level aggregation
    GBF3_IBRA_score_ag_AUS = xr_gbf3_ibra_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(bioregion_base_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    GBF3_IBRA_score_am_AUS = xr_gbf3_ibra_am.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(bioregion_base_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')

    GBF3_IBRA_score_non_ag_AUS = xr_gbf3_ibra_non_ag.sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(bioregion_base_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    # Combine regional and Australia level data
    GBF3_IBRA_score_ag = pd.concat([GBF3_IBRA_score_ag_region, GBF3_IBRA_score_ag_AUS], axis=0)
    GBF3_IBRA_score_am = pd.concat([GBF3_IBRA_score_am_region, GBF3_IBRA_score_am_AUS], axis=0)
    GBF3_IBRA_score_non_ag = pd.concat([GBF3_IBRA_score_non_ag_region, GBF3_IBRA_score_non_ag_AUS], axis=0)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    bioregion_base_score = bioregion_base_score.assign(
            Type='Outside LUTO study area',
            Year=yr_cal,
            lu='Outside LUTO study area',
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    pd.concat([
        GBF3_IBRA_score_ag,
        GBF3_IBRA_score_am,
        GBF3_IBRA_score_non_ag,
        bioregion_base_score],axis=0
        ).rename(columns={
            'lu':'Landuse',
            'lm':'Water_supply',
            'am':'Agricultural Management',
            'group':'IBRA Bioregion',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF3_IBRA_scores_{yr_cal}.csv'), index=False)

    # ------------------------- Stack array, get valid layers -------------------------

    # ---- Ag valid layers (include group in layer so each parallel task gets 1D cell data) ----
    valid_ag_layers = pd.MultiIndex.from_frame(GBF3_IBRA_score_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'lm', 'lu']]).sort_values()
    valid_layers_stack_ag = xr_gbf3_ibra_ag.stack(layer=['group', 'lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()

    # ---- Non-ag valid layers ----
    valid_non_ag_layers = pd.MultiIndex.from_frame(GBF3_IBRA_score_non_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'lu']]).sort_values()

    if GBF3_IBRA_score_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, 1, data.NCELLS), dtype=np.float32),
            dims=['group', 'lu', 'cell'],
            coords={'group': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['group', 'lu'])

    else:
        valid_layers_stack_non_ag = xr_gbf3_ibra_non_ag.stack(layer=['group', 'lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()

    # ---- Ag management valid layers ----
    valid_am_layers = pd.MultiIndex.from_frame(GBF3_IBRA_score_am_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'am', 'lm', 'lu']]).sort_values()

    if GBF3_IBRA_score_am_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['group', 'am', 'lm', 'lu', 'cell'],
            coords={'group': ['ALL'], 'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['group', 'am', 'lm', 'lu'])

    else:
        valid_layers_stack_am = xr_gbf3_ibra_am.stack(layer=['group', 'am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()

    # min/max should calculated using array without appending mosaic layers
    save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF3_IBRA_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF3_IBRA_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF3_IBRA_ag_management_{yr_cal}.nc'))

    return f"Biodiversity GBF3 IBRA scores written for year {yr_cal}"



def write_biodiversity_GBF4_SNES_scores(data: Data, yr_cal: int, path) -> None:
    ''' Biodiversity GBF4 SNES only being written to disk when `BIODIVERSITY_TARGET_GBF_4_SNES` is 'on' '''

    if not settings.BIODIVERSITY_TARGET_GBF_4_SNES == "on":
        return "Skipped: Biodiversity GBF4 SNES scores not written as `BIODIVERSITY_TARGET_GBF_4_SNES` is set to 'off'"

    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get the biodiversity scores for the year
    bio_snes_sr = xr.DataArray(
        ag_biodiversity.get_GBF4_SNES_matrix_sr(data).astype(np.float32),
        dims=['species','cell'],
        coords={'species':data.BIO_GBF4_SNES_SEL_ALL, 'cell':np.arange(data.NCELLS)}
    )

    # Apply habitat contribution from ag/am/non-ag land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]).astype(np.float32),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    ).unstack()

    # Get the base year biodiversity scores
    bio_snes_scores = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_SNES.csv')
    idx_row = [bio_snes_scores.query('SCIENTIFIC_NAME == @i').index[0] for i in data.BIO_GBF4_SNES_SEL_ALL]
    idx_all_score = [bio_snes_scores.columns.get_loc(f'BASELINE_LEVEL_ALL_AUSTRALIA_{col}') for col in data.BIO_GBF4_PRESENCE_SNES_SEL]
    idx_outside_score =  [bio_snes_scores.columns.get_loc(f'BASEYEAR_SCORE_OUT_LUTO_NATURAL_{col}') for col in data.BIO_GBF4_PRESENCE_SNES_SEL]

    base_yr_score = pd.DataFrame({
            'species': data.BIO_GBF4_SNES_SEL_ALL,
            'BASE_TOTAL_SCORE': [bio_snes_scores.iloc[row, col] for row, col in zip(idx_row, idx_all_score)],
            'BASE_OUTSIDE_SCORE': [bio_snes_scores.iloc[row, col] for row, col in zip(idx_row, idx_outside_score)],
            'TARGET_INSIDE_SCORE': data.get_GBF4_SNES_target_inside_LUTO_by_year(yr_cal)}
    ).eval('Target_by_Percent = (TARGET_INSIDE_SCORE + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100')

    # Calculate the biodiversity scores
    # Calculate xarray biodiversity GBF4 SNES scores
    xr_gbf4_snes_ag = bio_snes_sr * ag_impact_j * ag_dvar_mrj
    xr_gbf4_snes_am = bio_snes_sr * am_impact_amr * am_dvar_amrj
    xr_gbf4_snes_non_ag = bio_snes_sr * non_ag_impact_k * non_ag_dvar_rk

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_gbf4_snes_ag     = add_all(xr_gbf4_snes_ag,     ['lm', 'lu'])
    xr_gbf4_snes_non_ag = add_all(xr_gbf4_snes_non_ag, ['lu'])
    xr_gbf4_snes_am     = add_all(xr_gbf4_snes_am,     ['lm', 'lu', 'am'])

    # Regional level aggregation
    GBF4_score_ag_region = xr_gbf4_snes_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal)

    GBF4_score_am_region = xr_gbf4_snes_am.groupby('region'
        ).sum('cell').to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)

    GBF4_score_non_ag_region = xr_gbf4_snes_non_ag.groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)

    # Australia level aggregation
    GBF4_score_ag_AUS = xr_gbf4_snes_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    GBF4_score_am_AUS = xr_gbf4_snes_am.sum('cell').to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')

    GBF4_score_non_ag_AUS = xr_gbf4_snes_non_ag.sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    # Combine regional and Australia level data
    GBF4_score_ag = pd.concat([GBF4_score_ag_region, GBF4_score_ag_AUS], axis=0)
    GBF4_score_am = pd.concat([GBF4_score_am_region, GBF4_score_am_AUS], axis=0)
    GBF4_score_non_ag = pd.concat([GBF4_score_non_ag_region, GBF4_score_non_ag_AUS], axis=0)


    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal, lu='Outside LUTO study area'
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    pd.concat([
            GBF4_score_ag,
            GBF4_score_am,
            GBF4_score_non_ag,
            base_yr_score], axis=0
        ).rename(columns={
            'lu':'Landuse',
            'lm':'Water_supply',
            'am':'Agricultural Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)',
            'Target_by_Percent':'Target by Percent (%)'}).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF4_SNES_scores_{yr_cal}.csv'), index=False)

    # ------------------------- Stack array, get valid layers -------------------------

    # ---- Ag valid layers (include species in layer so each parallel task gets 1D cell data) ----
    valid_ag_layers = pd.MultiIndex.from_frame(GBF4_score_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'lm', 'lu']]).sort_values()
    valid_layers_stack_ag = xr_gbf4_snes_ag.stack(layer=['species', 'lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()

    # ---- Non-ag valid layers ----
    valid_non_ag_layers = pd.MultiIndex.from_frame(GBF4_score_non_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'lu']]).sort_values()

    if GBF4_score_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, 1, data.NCELLS), dtype=np.float32),
            dims=['species', 'lu', 'cell'],
            coords={'species': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['species', 'lu'])

    else:
        valid_layers_stack_non_ag = xr_gbf4_snes_non_ag.stack(layer=['species', 'lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()

    # ---- Ag management valid layers ----
    valid_am_layers = pd.MultiIndex.from_frame(GBF4_score_am_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'am', 'lm', 'lu']]).sort_values()

    if GBF4_score_am_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['species', 'am', 'lm', 'lu', 'cell'],
            coords={'species': ['ALL'], 'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['species', 'am', 'lm', 'lu'])

    else:
        valid_layers_stack_am = xr_gbf4_snes_am.stack(layer=['species', 'am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()

    # min/max should calculated using array without appending mosaic layers
    save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF4_SNES_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF4_SNES_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF4_SNES_ag_management_{yr_cal}.nc'))

    return f"Biodiversity GBF4 SNES scores written for year {yr_cal}"





def write_biodiversity_GBF4_ECNES_scores(data: Data, yr_cal: int, path) -> None:
    ''' Biodiversity GBF4 ECNES only being written to disk when `BIODIVERSITY_TARGET_GBF_4_ECNES` is 'on' '''
    
    if not settings.BIODIVERSITY_TARGET_GBF_4_ECNES == "on":
        return "Skipped: Biodiversity GBF4 ECNES scores not written as `BIODIVERSITY_TARGET_GBF_4_ECNES` is set to 'off'"

    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    
    # Get the biodiversity scores for the year
    bio_ecnes_sr = xr.DataArray(
        ag_biodiversity.get_GBF4_ECNES_matrix_sr(data).astype(np.float32),
        dims=['species','cell'],
        coords={'species':data.BIO_GBF4_ECNES_SEL_ALL, 'cell':np.arange(data.NCELLS)}
    ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS), 'species': 1})

    # Apply habitat contribution from ag/am/non-ag land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]).astype(np.float32),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)
        }
    ).unstack()

    # Get the base year biodiversity scores
    bio_ecnes_scores = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_ECNES.csv')
    idx_row = [bio_ecnes_scores.query('COMMUNITY == @i').index[0] for i in data.BIO_GBF4_ECNES_SEL_ALL]
    idx_all_score = [bio_ecnes_scores.columns.get_loc(f'BASELINE_LEVEL_ALL_AUSTRALIA_{col}') for col in data.BIO_GBF4_PRESENCE_ECNES_SEL]
    idx_outside_score = [bio_ecnes_scores.columns.get_loc(f'BASEYEAR_SCORE_OUT_LUTO_NATURAL_{col}') for col in data.BIO_GBF4_PRESENCE_ECNES_SEL]

    base_yr_score = pd.DataFrame({
        'species': data.BIO_GBF4_ECNES_SEL_ALL,
        'BASE_TOTAL_SCORE': [bio_ecnes_scores.iloc[row, col] for row, col in zip(idx_row, idx_all_score)],
        'BASE_OUTSIDE_SCORE': [bio_ecnes_scores.iloc[row, col] for row, col in zip(idx_row, idx_outside_score)],
        'TARGET_INSIDE_SCORE': data.get_GBF4_ECNES_target_inside_LUTO_by_year(yr_cal)
    }).eval('Target_by_Percent = (TARGET_INSIDE_SCORE + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100')

    # Calculate the biodiversity scores
    # Calculate xarray biodiversity GBF4 ECNES scores
    xr_gbf4_ecnes_ag = bio_ecnes_sr * ag_impact_j * ag_dvar_mrj
    xr_gbf4_ecnes_am = bio_ecnes_sr * am_impact_amr * am_dvar_amrj
    xr_gbf4_ecnes_non_ag = bio_ecnes_sr * non_ag_impact_k * non_ag_dvar_rk

    xr_gbf4_ecnes_ag     = add_all(xr_gbf4_ecnes_ag,     ['lm', 'lu'])
    xr_gbf4_ecnes_non_ag = add_all(xr_gbf4_ecnes_non_ag, ['lu'])
    xr_gbf4_ecnes_am     = add_all(xr_gbf4_ecnes_am,     ['lm', 'lu', 'am'])

    GBF4_score_ag_region = xr_gbf4_ecnes_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal)

    GBF4_score_am_region = xr_gbf4_ecnes_am.groupby('region'
        ).sum('cell').to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)

    GBF4_score_non_ag_region = xr_gbf4_ecnes_non_ag.groupby('region'
        ).sum(['cell']).to_dataframe('Area Weighted Score (ha)').reset_index(
        ).merge(base_yr_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)

    GBF4_score_ag_AUS = xr_gbf4_ecnes_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    GBF4_score_am_AUS = xr_gbf4_ecnes_am.sum('cell').to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')

    GBF4_score_non_ag_AUS = xr_gbf4_ecnes_non_ag.sum(['cell']).to_dataframe('Area Weighted Score (ha)').reset_index(
        ).merge(base_yr_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    # Combine regional and Australia level data
    GBF4_score_ag = pd.concat([GBF4_score_ag_region, GBF4_score_ag_AUS], axis=0)
    GBF4_score_am = pd.concat([GBF4_score_am_region, GBF4_score_am_AUS], axis=0)
    GBF4_score_non_ag = pd.concat([GBF4_score_non_ag_region, GBF4_score_non_ag_AUS], axis=0)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal, lu='Outside LUTO study area'
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')
    
    pd.concat([
            GBF4_score_ag,
            GBF4_score_am,
            GBF4_score_non_ag,
            base_yr_score], axis=0
        ).rename(columns={
            'lu':'Landuse',
            'lm':'Water_supply',
            'am':'Agricultural Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)',
            'Target_by_Percent': 'Target by Percent (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF4_ECNES_scores_{yr_cal}.csv'), index=False)

    

    # ==================== Ag Valid Layers ====================
    valid_ag_layers = pd.MultiIndex.from_frame(GBF4_score_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'lm', 'lu']]).sort_values()
    valid_layers_stack_ag = xr_gbf4_ecnes_ag.stack(layer=['species', 'lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()

    # ==================== Non-Ag Valid Layers ====================
    valid_non_ag_layers = pd.MultiIndex.from_frame(GBF4_score_non_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'lu']]).sort_values()

    if GBF4_score_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, 1, data.NCELLS), dtype=np.float32),
            dims=['species', 'lu', 'cell'],
            coords={'species': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['species', 'lu'])

    else:
        valid_layers_stack_non_ag = xr_gbf4_ecnes_non_ag.stack(layer=['species', 'lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()

    # ==================== Ag Management Valid Layers ====================
    valid_am_layers = pd.MultiIndex.from_frame(GBF4_score_am_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'am', 'lm', 'lu']]).sort_values()

    if GBF4_score_am_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['species', 'am', 'lm', 'lu', 'cell'],
            coords={'species': ['ALL'], 'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['species', 'am', 'lm', 'lu'])

    else:
        valid_layers_stack_am = xr_gbf4_ecnes_am.stack(layer=['species', 'am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()

    # min/max should calculated using array without appending mosaic layers
    save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF4_ECNES_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF4_ECNES_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF4_ECNES_ag_management_{yr_cal}.nc'))
    
    return f"Biodiversity GBF4 ECNES scores written for year {yr_cal}"




def write_biodiversity_GBF8_scores_groups(data: Data, yr_cal, path):
    ''' Biodiversity GBF8 groups only being written to disk when `BIODIVERSITY_TARGET_GBF_8` is 'on' '''
    
    # Do nothing if biodiversity limits are off and no need to report
    if not settings.BIODIVERSITY_TARGET_GBF_8 == 'on':
        return "Skipped: Biodiversity GBF8 groups scores not written as `BIODIVERSITY_TARGET_GBF_8` is set to 'off'"

        
    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get biodiversity scores for selected species
    bio_scores_sr = xr.DataArray(
        (data.get_GBF8_bio_layers_by_yr(yr_cal, level='group') * data.REAL_AREA[None,:]).astype(np.float32),
        dims=['group','cell'],
        coords={
            'group': data.BIO_GBF8_GROUPS_NAMES,
            'cell': np.arange(data.NCELLS)}
    ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS), 'group': 1})  # Chunking to save mem use

    # Get the habitat contribution for ag/non-ag/am land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]).astype(np.float32),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    ).unstack()

    # Get the base year biodiversity scores
    base_yr_score = pd.DataFrame({
            'group': data.BIO_GBF8_GROUPS_NAMES,
            'BASE_OUTSIDE_SCORE': data.get_GBF8_score_outside_natural_LUTO_by_yr(yr_cal, level='group'),
            'BASE_TOTAL_SCORE': data.BIO_GBF8_BASELINE_SCORE_GROUPS['HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA']}
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    # Calculate GBF8 scores for groups
    # Calculate xarray biodiversity GBF8 group scores
    xr_gbf8_groups_ag = bio_scores_sr * ag_impact_j * ag_dvar_mrj
    xr_gbf8_groups_am = am_dvar_amrj * bio_scores_sr * am_impact_amr
    xr_gbf8_groups_non_ag = non_ag_dvar_rk * bio_scores_sr * non_ag_impact_k

    xr_gbf8_groups_ag     = add_all(xr_gbf8_groups_ag,     ['lm', 'lu'])
    xr_gbf8_groups_non_ag = add_all(xr_gbf8_groups_non_ag, ['lu'])
    xr_gbf8_groups_am     = add_all(xr_gbf8_groups_am,     ['lm', 'lu', 'am'])

    GBF8_scores_groups_ag_region = xr_gbf8_groups_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal)
        
    GBF8_scores_groups_am_region = xr_gbf8_groups_am.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)
        
    GBF8_scores_groups_non_ag_region = xr_gbf8_groups_non_ag.groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)

    GBF8_scores_groups_ag_AUS = xr_gbf8_groups_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')
        
    GBF8_scores_groups_am_AUS = xr_gbf8_groups_am.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')
        
    GBF8_scores_groups_non_ag_AUS = xr_gbf8_groups_non_ag.sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    # Combine regional and Australia level data
    GBF8_scores_groups_ag = pd.concat([GBF8_scores_groups_ag_region, GBF8_scores_groups_ag_AUS], axis=0)
    GBF8_scores_groups_am = pd.concat([GBF8_scores_groups_am_region, GBF8_scores_groups_am_AUS], axis=0)
    GBF8_scores_groups_non_ag = pd.concat([GBF8_scores_groups_non_ag_region, GBF8_scores_groups_non_ag_AUS], axis=0)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal) 

    pd.concat([
        GBF8_scores_groups_ag, 
        GBF8_scores_groups_am, 
        GBF8_scores_groups_non_ag,
        base_yr_score], axis=0
        ).rename(columns={
            'group': 'Group',
            'lu': 'Landuse',
            'lm': 'Water_supply',
            'am': 'Agricultural Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF8_groups_scores_{yr_cal}.csv'), index=False)

    

    # ==================== Ag Valid Layers ====================
    valid_ag_layers = pd.MultiIndex.from_frame(GBF8_scores_groups_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'lm', 'lu']]).sort_values()
    valid_layers_stack_ag = xr_gbf8_groups_ag.stack(layer=['group', 'lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()

    # ==================== Non-Ag Valid Layers ====================
    valid_non_ag_layers = pd.MultiIndex.from_frame(GBF8_scores_groups_non_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'lu']]).sort_values()

    if GBF8_scores_groups_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, 1, data.NCELLS), dtype=np.float32),
            dims=['group', 'lu', 'cell'],
            coords={'group': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['group', 'lu'])

    else:
        valid_layers_stack_non_ag = xr_gbf8_groups_non_ag.stack(layer=['group', 'lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()

    # ==================== Ag Management Valid Layers ====================
    valid_am_layers = pd.MultiIndex.from_frame(GBF8_scores_groups_am_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'am', 'lm', 'lu']]).sort_values()

    if GBF8_scores_groups_am_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['group', 'am', 'lm', 'lu', 'cell'],
            coords={'group': ['ALL'], 'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['group', 'am', 'lm', 'lu'])

    else:
        valid_layers_stack_am = xr_gbf8_groups_am.stack(layer=['group', 'am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()

    # min/max should calculated using array without appending mosaic layers
    save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF8_groups_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF8_groups_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF8_groups_ag_management_{yr_cal}.nc'))
    
    return f"Biodiversity GBF8 groups scores written for year {yr_cal}"




def write_biodiversity_GBF8_scores_species(data: Data, yr_cal, path):
    ''' Biodiversity GBF8 species only being written to disk when `BIODIVERSITY_TARGET_GBF_8` is 'on' and selected species are provided '''

    if settings.BIODIVERSITY_TARGET_GBF_8 != 'on':
        return "Skipped: Biodiversity GBF8 species scores not written as `BIODIVERSITY_TARGET_GBF_8` is set to 'off'"
    if len(data.BIO_GBF8_SEL_SPECIES) == 0:
        return "Skipped: Biodiversity GBF8 species scores not written as no selected species provided in `BIO_GBF8_SEL_SPECIES`"

    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    
    # Get biodiversity scores for selected species
    bio_scores_sr = xr.DataArray(
        (data.get_GBF8_bio_layers_by_yr(yr_cal, level='species') * data.REAL_AREA[None, :]).astype(np.float32),
        dims=['species', 'cell'],
        coords={
            'species': data.BIO_GBF8_SEL_SPECIES,
            'cell': np.arange(data.NCELLS)}
    ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS), 'species': 1})  # Chunking to save mem use

    # Get the habitat contribution for ag/non-ag/am land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]).astype(np.float32),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    ).unstack()

    # Get the base year biodiversity scores
    base_yr_score = pd.DataFrame({
            'species': data.BIO_GBF8_SEL_SPECIES,
            'BASE_OUTSIDE_SCORE': data.get_GBF8_score_outside_natural_LUTO_by_yr(yr_cal),
            'BASE_TOTAL_SCORE': data.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES['HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA'],
            'TARGET_INSIDE_SCORE': data.get_GBF8_target_inside_LUTO_by_yr(yr_cal),}
        ).eval('Target_by_Percent = (TARGET_INSIDE_SCORE + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100')

    # Calculate GBF8 scores for species
    # Calculate xarray biodiversity GBF8 species scores
    xr_gbf8_species_ag = bio_scores_sr * ag_impact_j * ag_dvar_mrj
    xr_gbf8_species_am = am_dvar_amrj * bio_scores_sr * am_impact_amr
    xr_gbf8_species_non_ag = non_ag_dvar_rk * bio_scores_sr * non_ag_impact_k

    xr_gbf8_species_ag     = add_all(xr_gbf8_species_ag,     ['lm', 'lu'])
    xr_gbf8_species_non_ag = add_all(xr_gbf8_species_non_ag, ['lu'])
    xr_gbf8_species_am     = add_all(xr_gbf8_species_am,     ['lm', 'lu', 'am'])

    GBF8_scores_species_ag_region = xr_gbf8_species_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal)

    GBF8_scores_species_am_region = xr_gbf8_species_am.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)

    GBF8_scores_species_non_ag_region = xr_gbf8_species_non_ag.groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)

    GBF8_scores_species_ag_AUS = xr_gbf8_species_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    GBF8_scores_species_am_AUS = xr_gbf8_species_am.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')

    GBF8_scores_species_non_ag_AUS = xr_gbf8_species_non_ag.sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    # Combine regional and Australia level data
    GBF8_scores_species_ag = pd.concat([GBF8_scores_species_ag_region, GBF8_scores_species_ag_AUS], axis=0)
    GBF8_scores_species_am = pd.concat([GBF8_scores_species_am_region, GBF8_scores_species_am_AUS], axis=0)
    GBF8_scores_species_non_ag = pd.concat([GBF8_scores_species_non_ag_region, GBF8_scores_species_non_ag_AUS], axis=0)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal, lu='Outside LUTO study area'
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    pd.concat([
        GBF8_scores_species_ag,
        GBF8_scores_species_am,
        GBF8_scores_species_non_ag,
        base_yr_score], axis=0
        ).rename(columns={
            'species': 'Species',
            'lu': 'Landuse',
            'lm': 'Water_supply',
            'am': 'Agricultural Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF8_species_scores_{yr_cal}.csv'), index=False)

    

    # ==================== Ag Valid Layers ====================
    valid_ag_layers = pd.MultiIndex.from_frame(GBF8_scores_species_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'lm', 'lu']]).sort_values()
    valid_layers_stack_ag = xr_gbf8_species_ag.stack(layer=['species', 'lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()

    # ==================== Non-Ag Valid Layers ====================
    valid_non_ag_layers = pd.MultiIndex.from_frame(GBF8_scores_species_non_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'lu']]).sort_values()

    if GBF8_scores_species_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, 1, data.NCELLS), dtype=np.float32),
            dims=['species', 'lu', 'cell'],
            coords={'species': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['species', 'lu'])

    else:
        valid_layers_stack_non_ag = xr_gbf8_species_non_ag.stack(layer=['species', 'lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()

    # ==================== Ag Management Valid Layers ====================
    valid_am_layers = pd.MultiIndex.from_frame(GBF8_scores_species_am_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'am', 'lm', 'lu']]).sort_values()

    if GBF8_scores_species_am_AUS['Area Weighted Score (ha)'].abs().sum() < 1e-3:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['species', 'am', 'lm', 'lu', 'cell'],
            coords={'species': ['ALL'], 'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['species', 'am', 'lm', 'lu'])

    else:
        valid_layers_stack_am = xr_gbf8_species_am.stack(layer=['species', 'am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()

    # min/max should calculated using array without appending mosaic layers
    save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF8_species_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF8_species_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF8_species_ag_management_{yr_cal}.nc'))
    
    return f"Biodiversity GBF8 species scores written for year {yr_cal}"






