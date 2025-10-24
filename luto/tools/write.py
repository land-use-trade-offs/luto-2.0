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
import shutil
import threading
import numpy as np
import pandas as pd
import xarray as xr

from joblib import Parallel, delayed

from luto import settings
from luto import tools
from luto.data import Data

from luto.tools.Manual_jupyter_books.helpers import arr_to_xr
from luto.tools.report.data_tools.parameters import GHG_NAMES
from luto.tools.report.create_report_layers import save_report_layer
from luto.tools.report.create_report_data import save_report_data

import luto.economics.agricultural.quantity as ag_quantity                      
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.transitions as ag_transitions
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.biodiversity as ag_biodiversity

import luto.economics.non_agricultural.quantity as non_ag_quantity              
import luto.economics.non_agricultural.revenue as non_ag_revenue
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.transitions as non_ag_transitions
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.non_agricultural.biodiversity as non_ag_biodiversity


def write_outputs(data: Data):
    """Write outputs using dynamic timestamp from read_timestamp."""
    
    # Generate path using read_timestamp each time this function is called
    current_timestamp = tools.read_timestamp()
    log_path = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}/LUTO_RUN_"
    
    @tools.LogToFile(log_path)
    def _write_outputs():
        # Start recording memory usage
        stop_event = threading.Event()
        memory_thread = threading.Thread(
            target=tools.log_memory_usage, 
            args=(f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}", 'a', 1, stop_event)
        )
        memory_thread.start()
        try:
            write_data(data)
            create_report(data)
        except Exception as e:
            print(f"An error occurred while writing outputs: {e}")
            raise e
        finally:
            # Ensure the memory logging thread is stopped
            stop_event.set()
            memory_thread.join()
    
    return _write_outputs()



def write_data(data: Data):
    years = [i for i in settings.SIM_YEARS if i<=data.last_year]
    paths = [f"{data.path}/out_{yr}" for yr in years]
    write_settings(data.path)
    
    # Has to write dvars first because some other writings depend on dvars
    jobs = [delayed(write_dvar_and_mosaic_map)(data, yr, path_yr) for (yr, path_yr) in zip(years, paths)]
    Parallel(n_jobs=min(len(jobs), settings.WRITE_THREADS))(jobs)
    
    # Wrap each year's writing tasks into a job
    jobs = [delayed(write_area_transition_start_end)(data, f'{data.path}/out_{years[-1]}', years[-1])]
    for (yr, path_yr) in zip(years, paths): jobs += write_output_single_year(data, yr, path_yr)  
    jobs = [job for job in jobs if job is not None] # None means a task is skipped
    num_jobs = (min(len(jobs), settings.WRITE_THREADS) if settings.PARALLEL_WRITE else 1)

    # Execute jobs in parallel and print outputs as they complete
    for out in Parallel(n_jobs=num_jobs, return_as='generator')(jobs):
        print(out)


def write_settings(path):
    with open('luto/settings.py', 'r') as file:
        lines = file.readlines()
        parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=")
        settings_order = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]
        settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
        settings_dict = {i: settings_dict[i] for i in settings_order if i in settings_dict}
    with open(os.path.join(path, 'model_run_settings.txt'), 'w') as f:
        f.writelines(f'{k}:{v}\n' for k, v in settings_dict.items())
    return "Settings written successfully"



def create_report(data: Data):
    """Create report using dynamic timestamp from read_timestamp."""
    
    # Generate path using read_timestamp each time this function is called
    current_timestamp = tools.read_timestamp()
    save_dir = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"
    log_path = f"{save_dir}/LUTO_RUN_"
    
    @tools.LogToFile(log_path, mode='a')
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
   
   
def save2nc(in_xr:xr.DataArray, save_path:str, calc_valid_layers:bool=True):
    
    # Ensure dask chunking: keep 'cell' chunks as full length, others as 1
    chunks_size = dict(zip(in_xr.dims, in_xr.shape))
    chunks_size = {k: (v if k == 'cell' else 1) for k, v in chunks_size.items()}
    in_xr = in_xr.chunk(chunks_size)
 
    encoding = {'data':{
        'dtype': 'float32',
        'zlib': True,
        'complevel': 4,
        'chunksizes': list(chunks_size.values())
    }}
    in_xr.name = 'data'
    in_xr = in_xr.drop_vars(set(in_xr.coords) - set(in_xr.dims))
    
    if calc_valid_layers:
        # Identify valid layers (i.e., layers with non-zero values)
        valid_df = in_xr.sum(['cell'], skipna=True).to_dataframe('ly_sum').query('abs(ly_sum) > 1e-3')
        loop_sel = valid_df.index.to_frame().to_dict('records')
        in_xr.attrs['valid_layers'] = str(loop_sel)
        
        # Add min and max attributes, 
        #  excluding 'ALL' layers because they are index (used to create mosaic maps) 
        #  rather than actual data
        data_lys = in_xr
        
        if 'am' in data_lys.dims and 'ALL' in data_lys['am'].values:
            data_lys = data_lys.drop_sel(am='ALL')
        if 'lu' in data_lys.dims and 'ALL' in data_lys['lu'].values:
            data_lys = data_lys.drop_sel(lu='ALL')
        
        min_val = float(data_lys.min().values)
        if abs(min_val) < 1e-3: min_val = 0.0
        max_val = float(data_lys.max().values)
        in_xr.attrs['min_max'] = (min_val, max_val)
    
    in_xr.astype('float32').to_netcdf(save_path, encoding=encoding)



def write_output_single_year(data: Data, yr_cal, path_yr):
    """Wrap write tasks for a single year"""

    if not os.path.isdir(path_yr):
        os.mkdir(path_yr)
        
    tasks = [
        delayed(write_dvar_area)(data, yr_cal, path_yr),
        delayed(write_crosstab)(data, yr_cal, path_yr),
        delayed(write_quantity)(data, yr_cal, path_yr),
        delayed(write_quantity_separate)(data, yr_cal, path_yr),
        delayed(write_revenue_cost_ag)(data, yr_cal, path_yr),
        delayed(write_revenue_cost_ag_man)(data, yr_cal, path_yr),
        delayed(write_revenue_cost_non_ag)(data, yr_cal, path_yr),
        delayed(write_transition_cost_ag2ag)(data, yr_cal, path_yr),
        delayed(write_transition_cost_ag2nonag)(data, yr_cal, path_yr),
        delayed(write_transition_cost_nonag2ag)(data, yr_cal, path_yr),
        delayed(write_transition_cost_apply_ag_man)(data),
        delayed(write_water)(data, yr_cal, path_yr),
        delayed(write_ghg)(data, yr_cal, path_yr),
        delayed(write_ghg_separate)(data, yr_cal, path_yr),
        delayed(write_ghg_offland_commodity)(data, yr_cal, path_yr),
        delayed(write_biodiversity_overall_quality_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF2_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF3_NVIS_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF3_IBRA_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF4_SNES_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF4_ECNES_scores)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF8_scores_groups)(data, yr_cal, path_yr),
        delayed(write_biodiversity_GBF8_scores_species)(data, yr_cal, path_yr)
    ]

    return tasks



def write_dvar_and_mosaic_map(data: Data, yr_cal, path):
        
    # Dvar maps
    ag_map = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]).chunk({'cell': min(1024, data.NCELLS)})
    non_ag_map = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]).chunk({'cell': min(1024, data.NCELLS)})
    am_map = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).chunk({'cell': min(1024, data.NCELLS)})
    
    ag_mask = ag_map.sum(['lm','lu']) > 0.001
    am_mask = am_map.sum(['am','lm', 'lu']) > 0.001
    non_ag_mask = non_ag_map.sum('lu') > 0.001

    # Expand dimension
    ag_map = xr.concat([ag_map.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), ag_map], dim='lm')
    am_map = xr.concat([am_map.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), am_map], dim='lm')
    am_map = xr.concat([am_map.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), am_map], dim='lu')

    # Mosaic maps
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
    
    # Concat dvar and mosaic map
    ag_map_cat = xr.concat([ag_map_argmax, ag_map], dim='lu')
    non_ag_map_cat = xr.concat([non_ag_map_argmax, non_ag_map], dim='lu')
    am_map_cat = xr.concat([am_argmax, am_map], dim='am')

    save2nc(ag_map_cat, os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))
    save2nc(non_ag_map_cat, os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))
    save2nc(am_map_cat, os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))

    # Write landuse mosaic map
    lumap_xr_ALL= xr.DataArray(data.lumaps[yr_cal], dims=['cell'], coords={'cell': range(data.NCELLS)})
    lumap_xr_dry = xr.DataArray(np.where(~lm_map, lumap_xr_ALL, np.nan), dims=['cell'], coords={'cell': range(data.NCELLS)})
    lumap_xr_irr = xr.DataArray(np.where(lm_map, lumap_xr_ALL, np.nan), dims=['cell'], coords={'cell': range(data.NCELLS)})
    
    lumap_xr = xr.concat([
        lumap_xr_ALL.expand_dims(lm=['ALL']),
        lumap_xr_dry.expand_dims(lm=['dry']),
        lumap_xr_irr.expand_dims(lm=['irr'])
    ], dim='lm').astype(np.float32)
    
    lumap_xr.attrs['valid_layers'] = str([{'lm':lm} for lm in lumap_xr['lm'].values])
    
    save2nc(lumap_xr, os.path.join(path, f'xr_map_lumap_{yr_cal}.nc'))
    
    
    # Save an template to get geo spatial reference later
    xr.Dataset({
        'layer':arr_to_xr(data, lumap_xr_ALL.astype(np.float32))
    }).to_netcdf(os.path.join(path, f'xr_map_template_{yr_cal}.nc'))

    return f"Mosaic maps written for year {yr_cal}"



def write_quantity(data: Data, yr_cal, path):
    
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE
    yr_idx_sim = sorted(list(data.lumaps.keys())).index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]

    # Calculate data for quantity comparison between base year and target year
    if yr_cal > data.YR_CAL_BASE:
        # Check if yr_cal_sim_pre meets the requirement
        assert data.YR_CAL_BASE <= yr_cal_sim_pre < yr_cal, f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"

        # Get commodity production quantities produced in base year and target year
        prod_base = np.array(data.prod_data[yr_cal_sim_pre]['Production'])
        prod_targ = np.array(data.prod_data[yr_cal]['Production'])
        demands = data.D_CY[yr_idx]  # Get commodity demands for target year

        # Calculate differences
        abs_diff = prod_targ - demands
        prop_diff = (prod_targ / demands) * 100

        # Create pandas dataframe
        df = pd.DataFrame({
            'Commodity': [i[0].capitalize() + i[1:] for i in data.COMMODITIES],
            'Prod_base_year (tonnes, KL)': prod_base,
            'Prod_targ_year (tonnes, KL)': prod_targ,
            'Demand (tonnes, KL)': demands,
            'Abs_diff (tonnes, KL)': abs_diff,
            'Prop_diff (%)': prop_diff
        })

        # Save files to disk
        df['Year'] = yr_cal
        df.to_csv(os.path.join(path, f'quantity_comparison_{yr_cal}.csv'), index=False)

    return f"Quantity comparison written for year {yr_cal}"


        
def write_quantity_separate(data: Data, yr_cal: int, path: str) -> np.ndarray:
    """
    Return total production of commodities for a specific year...

    'yr_cal' is calendar year

    Can return base year production (e.g., year = 2010) or can return production for
    a simulated year if one exists (i.e., year = 2030).

    Includes the impacts of land-use change, productivity increases, and
    climate change on yield.
    """
    if yr_cal == data.YR_CAL_BASE:
        ag_X_mrj = data.AG_L_MRJ
        non_ag_X_rk = data.NON_AG_L_RK
        ag_man_X_mrj = data.AG_MAN_L_MRJ_DICT
        
    else: # In this case, the dvars are already appended from the solver
        ag_X_mrj = data.ag_dvars[yr_cal]
        non_ag_X_rk = data.non_ag_dvars[yr_cal]
        ag_man_X_mrj = data.ag_man_dvars[yr_cal]

    # Calculate year index (i.e., number of years since 2010)
    yr_idx = yr_cal - data.YR_CAL_BASE
    lumap = data.lumaps[yr_cal]

    # Convert np.array to xr.DataArray; Chunk the data to reduce memory usage
    ag_X_mrj_xr = tools.ag_mrj_to_xr(data, ag_X_mrj).chunk({'cell': min(4096, data.NCELLS)})
    non_ag_X_rk_xr = tools.non_ag_rk_to_xr(data, non_ag_X_rk).chunk({'cell': min(4096, data.NCELLS)})
    ag_man_X_mrj_xr = tools.am_mrj_to_xr(data, ag_man_X_mrj).chunk({'cell': min(4096, data.NCELLS)})


    # Convert LU2PR and PR2CM to xr.DataArray 
    lu2pr_xr = xr.DataArray(
        data.LU2PR.astype(bool),
        dims=['product', 'lu'],
        coords={
            'product': data.PRODUCTS,
            'lu': data.AGRICULTURAL_LANDUSES
        },
    )

    pr2cm_xr = xr.DataArray(
        data.PR2CM.astype(bool),
        dims=['Commodity', 'product'],
        coords={
            'Commodity': data.COMMODITIES,
            'product': data.PRODUCTS
        },
    )

    # Get commodity matrices
    ag_q_mrp_xr = xr.DataArray(
        ag_quantity.get_quantity_matrices(data, yr_idx),
        dims=['lm','cell','product'],
        coords={
            'lm': data.LANDMANS,
            'cell': range(data.NCELLS),
            'product': data.PRODUCTS
        },
    ).assign_coords(
        region=('cell', data.REGION_NRM_NAME),
    )

    non_ag_crk_xr = xr.DataArray(
        non_ag_quantity.get_quantity_matrix(data, ag_q_mrp_xr, lumap),
        dims=['Commodity', 'cell', 'lu'],
        coords={
            'Commodity': data.COMMODITIES,
            'cell': range(data.NCELLS),
            'lu': data.NON_AGRICULTURAL_LANDUSES
        },
    ).assign_coords(
        region=('cell', data.REGION_NRM_NAME),
    )

    ag_man_q_mrp_xr = xr.DataArray(
        np.stack(list(ag_quantity.get_agricultural_management_quantity_matrices(data, ag_q_mrp_xr, yr_idx).values())),
        dims=['am', 'lm', 'cell', 'product'],
        coords={
            'am': data.AG_MAN_DESC,
            'lm': data.LANDMANS,
            'cell': np.arange(data.NCELLS),
            'product': data.PRODUCTS
        }
    ).assign_coords(
        region=('cell', data.REGION_NRM_NAME),
    )

    # Calculate the commodity production (BEFORE dimension expansion to avoid double counting)
    ag_q_mrc = (((ag_X_mrj_xr * lu2pr_xr).sum(dim=['lu']) * ag_q_mrp_xr) * pr2cm_xr).sum(dim='product')
    non_ag_p_rc = (non_ag_X_rk_xr * non_ag_crk_xr).sum(dim=['lu'])
    am_p_amrc = (((ag_man_X_mrj_xr * lu2pr_xr).sum(['lu']) * ag_man_q_mrp_xr) * pr2cm_xr).sum('product')

    # Expand dimension (has to be after calculation to avoid double counting)
    ag_q_mrc = xr.concat([ag_q_mrc.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), ag_q_mrc], dim='lm')
    am_p_amrc = xr.concat([am_p_amrc.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), am_p_amrc], dim='lm')
    am_p_amrc = xr.concat([am_p_amrc.sum(dim='Commodity', keepdims=True).assign_coords(Commodity=['ALL']), am_p_amrc], dim='Commodity')

    # Regional level aggregation
    ag_q_mrc_df_region = ag_q_mrc.groupby('region'
        ).sum('cell'
        ).to_dataframe('Production (t/KL)'
        ).reset_index(
        ).assign(Type='Agricultural'
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply'}
        ).assign(Year=yr_cal
        ).replace({'dry':'Dryland', 'irr':'Irrigated'})
    non_ag_p_rc_df_region = non_ag_p_rc.groupby('region'
        ).sum('cell'
        ).to_dataframe('Production (t/KL)'
        ).assign(Year=yr_cal, Type='Non-Agricultural'
        ).reset_index()
    am_p_amrc_df_region = am_p_amrc.groupby('region'
        ).sum('cell'
        ).to_dataframe('Production (t/KL)'
        ).reset_index(
        ).assign(Type='Agricultural Management'
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply'}
        ).assign(Year=yr_cal
        ).replace({'dry':'Dryland', 'irr':'Irrigated'})
    
    # Australia level aggregation
    ag_q_mrc_df_AUS = ag_q_mrc.sum('cell'
        ).to_dataframe('Production (t/KL)'
        ).reset_index(
        ).assign(Type='Agricultural'
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply'}
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).replace({'dry':'Dryland', 'irr':'Irrigated'})
    non_ag_p_rc_df_AUS = non_ag_p_rc.sum('cell'
        ).to_dataframe('Production (t/KL)'
        ).assign(Year=yr_cal, Type='Non-Agricultural', region='AUSTRALIA'
        ).reset_index()
    am_p_amrc_df_AUS = am_p_amrc.sum('cell'
        ).to_dataframe('Production (t/KL)'
        ).reset_index(
        ).assign(Type='Agricultural Management'
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply'}
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).replace({'dry':'Dryland', 'irr':'Irrigated'})
    
    # Save the production dataframes to csv
    quantity_df_AUS = pd.concat([ag_q_mrc_df_AUS, non_ag_p_rc_df_AUS, am_p_amrc_df_AUS], ignore_index=True).query('abs(`Production (t/KL)`) > 1')
    quantity_df_region = pd.concat([ag_q_mrc_df_region, non_ag_p_rc_df_region, am_p_amrc_df_region], ignore_index=True).query('abs(`Production (t/KL)`) > 1')
    pd.concat([quantity_df_AUS, quantity_df_region]).to_csv(os.path.join(path, f'quantity_production_t_separate_{yr_cal}.csv'), index=False)
    
    # Append the mosaic layers and save to netcdf
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL']).rename({'lu':'Commodity'})
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL']).rename({'lu':'Commodity'})
    am_mosaic = xr.open_dataarray(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'), chunks='auto').sel(am=['ALL'], lu='ALL', drop=True)
    am_mosaic.attrs = {}
    
    # Get the commodity mosaic map
    commodity_lumap = am_p_amrc.sel(lm='ALL').sum('am', skipna=True).argmax('Commodity')
    
    commodity_ammap = xr.concat([
        am_mosaic.where(commodity_lumap == c_code).expand_dims(Commodity=[c_desc])
        for c_code, c_desc in enumerate(data.COMMODITIES)
    ], dim='Commodity')
    
    commodity_ammap = xr.concat([
        am_mosaic.expand_dims(Commodity=['ALL']),
        commodity_ammap
    ], dim='Commodity')

    

    ag_q_mrc_cat = xr.concat([ag_mosaic, ag_q_mrc], dim='Commodity')
    non_ag_p_rc_cat = xr.concat([non_ag_mosaic, non_ag_p_rc], dim='Commodity')
    am_p_amrc_cat = xr.concat([commodity_ammap, am_p_amrc], dim='am')
    
    save2nc(ag_q_mrc_cat, os.path.join(path, f'xr_quantities_agricultural_{yr_cal}.nc'))
    save2nc(non_ag_p_rc_cat, os.path.join(path, f'xr_quantities_non_agricultural_{yr_cal}.nc'))
    save2nc(am_p_amrc_cat, os.path.join(path, f'xr_quantities_agricultural_management_{yr_cal}.nc'))

    return f"Separate quantity production written for year {yr_cal}"


def write_revenue_cost_ag(data: Data, yr_cal, path):
    """Calculate agricultural revenue. Takes a simulation object, a target calendar
       year (e.g., 2030), and an output path as input."""

    yr_idx = yr_cal - data.YR_CAL_BASE
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]).chunk({'cell': min(1024, data.NCELLS)})

    # Get agricultural revenue/cost for year in mrjs format
    ag_rev_df_rjms = ag_revenue.get_rev_matrices(data, yr_idx, aggregate=False)
    ag_cost_df_rjms = ag_cost.get_cost_matrices(data, yr_idx, aggregate=False)
    ag_rev_rjms = ag_rev_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_rev_df_rjms.columns.levels), fill_value=0).values.reshape(-1, *ag_rev_df_rjms.columns.levshape)
    ag_cost_rjms = ag_cost_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_cost_df_rjms.columns.levels), fill_value=0).values.reshape(-1, *ag_cost_df_rjms.columns.levshape)

    # Convert the ag_rev_rjms and ag_cost_rjms to xarray DataArray, 
    # and assign region names to the cell dimension
    ag_rev_rjms = xr.DataArray(
            ag_rev_rjms,
            dims=['cell', 'lu', 'lm', 'source'],
            coords={
                'cell': range(data.NCELLS),
                'lu': data.AGRICULTURAL_LANDUSES,
                'lm': data.LANDMANS,
                'source': ag_rev_df_rjms.columns.levels[2]
            }
        ).assign_coords(
            region = ('cell', data.REGION_NRM_NAME),
        )
    ag_cost_rjms = xr.DataArray(
            ag_cost_rjms,
            dims=['cell', 'lu', 'lm', 'source'],
            coords={
                'cell': range(data.NCELLS),
                'lu': data.AGRICULTURAL_LANDUSES,
                'lm': data.LANDMANS,
                'source': ag_cost_df_rjms.columns.levels[2]
            }
        ).assign_coords(
            region = ('cell', data.REGION_NRM_NAME),
        )

    # Multiply the ag_dvar_mrj with the ag_rev_mrj to get the ag_rev_jm
    xr_ag_rev = ag_dvar_mrj * ag_rev_rjms
    xr_ag_cost = ag_dvar_mrj * ag_cost_rjms

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_ag_rev = xr.concat([xr_ag_rev.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_ag_rev], dim='lm')
    xr_ag_rev = xr.concat([xr_ag_rev.sum(dim='source', keepdims=True).assign_coords(source=['ALL']), xr_ag_rev], dim='source')
    xr_ag_cost = xr.concat([xr_ag_cost.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_ag_cost], dim='lm')
    xr_ag_cost = xr.concat([xr_ag_cost.sum(dim='source', keepdims=True).assign_coords(source=['ALL']), xr_ag_cost], dim='source')

    # Regional level aggregation
    ag_rev_jms_region = xr_ag_rev.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year=yr_cal)
    ag_rev_jms_AUS = xr_ag_rev.sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year=yr_cal, region='AUSTRALIA')
        
    ag_cost_jms_region = xr_ag_cost.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year=yr_cal)
    ag_cost_jms_AUS = xr_ag_cost.sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year=yr_cal, region='AUSTRALIA')
   
    # Save to disk
    ag_rev_jms = pd.concat([ag_rev_jms_AUS, ag_rev_jms_region]
        ).rename(columns={
            'lu': 'Land-use',
            'lm': 'Water_supply',
            'source': 'Type'
        }).replace({
            'dry': 'Dryland',
            'irr': 'Irrigated'
        })
    ag_cost_jms = pd.concat([ag_cost_jms_AUS, ag_cost_jms_region]
        ).rename(columns={
            'lu': 'Land-use',
            'lm': 'Water_supply',
            'source': 'Type'
        }).replace({
            'dry': 'Dryland',
            'irr': 'Irrigated'
        })
        
    ag_rev_jms.to_csv(os.path.join(path, f'revenue_ag_{yr_cal}.csv'), index=False)
    ag_cost_jms.to_csv(os.path.join(path, f'cost_ag_{yr_cal}.csv'), index=False)
    
    # Append mosaic layer, then save to netcdf
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_ag_rev_cat = xr.concat([ag_mosaic, xr_ag_rev], dim='lu')
    xr_ag_cost_cat = xr.concat([ag_mosaic, xr_ag_cost], dim='lu')
    
    save2nc(xr_ag_rev_cat, os.path.join(path, f'xr_revenue_ag_{yr_cal}.nc'))
    save2nc(xr_ag_cost_cat, os.path.join(path, f'xr_cost_ag_{yr_cal}.nc'))

    return f"Agricultural revenue and cost written for year {yr_cal}"



def write_revenue_cost_ag_man(data: Data, yr_cal, path):
    """Calculate agricultural management revenue and cost."""

    
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the ag-man dvars
    am_dvar_mrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).assign_coords(region = ('cell', data.REGION_NRM_NAME),
        ).chunk({'cell': min(1024, data.NCELLS)})

    # Get the revenue/cost matrices for each agricultural land-use
    ag_rev_mrj = ag_revenue.get_rev_matrices(data, yr_idx)
    ag_cost_mrj = ag_cost.get_cost_matrices(data, yr_idx)
    am_revenue_mat = tools.am_mrj_to_xr(data, ag_revenue.get_agricultural_management_revenue_matrices(data, ag_rev_mrj, yr_idx))
    am_cost_mat = tools.am_mrj_to_xr(data, ag_cost.get_agricultural_management_cost_matrices(data, ag_cost_mrj, yr_idx))
    
    # Multiply the am_dvar_mrj with the am_revenue_mat to get the revenue and cost
    xr_revenue_am = am_dvar_mrj * am_revenue_mat
    xr_cost_am = am_dvar_mrj * am_cost_mat

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_revenue_am = xr.concat([xr_revenue_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_revenue_am], dim='lm')
    xr_revenue_am = xr.concat([xr_revenue_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_revenue_am], dim='lu')
    xr_cost_am = xr.concat([xr_cost_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_cost_am], dim='lm')
    xr_cost_am = xr.concat([xr_cost_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_cost_am], dim='lu')

    # Regional level aggregation
    revenue_am_df_region = xr_revenue_am.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year = yr_cal
        )
    cost_am_df_region = xr_cost_am.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year = yr_cal)

    # Australia level aggregation
    revenue_am_df_AUS = xr_revenue_am.sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year = yr_cal, region='AUSTRALIA'
        )
    cost_am_df_AUS = xr_cost_am.sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year = yr_cal, region='AUSTRALIA')
        
    revenue_am_df = pd.concat([revenue_am_df_AUS, revenue_am_df_region]
        ).rename(
            columns={
                'lu': 'Land-use',
                'lm': 'Water_supply',
                'am': 'Management Type'
        }).replace({'dry': 'Dryland', 'irr': 'Irrigated'})
    cost_am_df = pd.concat([cost_am_df_AUS, cost_am_df_region]
        ).rename(
            columns={
                'lu': 'Land-use',
                'lm': 'Water_supply',
                'am': 'Management Type'
        }).replace({'dry': 'Dryland', 'irr': 'Irrigated'})

    # Save to disk
    revenue_am_df.to_csv(os.path.join(path, f'revenue_agricultural_management_{yr_cal}.csv'), index=False)
    cost_am_df.to_csv(os.path.join(path, f'cost_agricultural_management_{yr_cal}.csv'), index=False)

    # Append mosaic layer, then save to netcdf
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_revenue_am_cat = xr.concat([am_mosaic, xr_revenue_am], dim='am')
    xr_cost_am_cat = xr.concat([am_mosaic, xr_cost_am], dim='am')

    save2nc(xr_revenue_am_cat, os.path.join(path, f'xr_revenue_agricultural_management_{yr_cal}.nc'))
    save2nc(xr_cost_am_cat, os.path.join(path, f'xr_cost_agricultural_management_{yr_cal}.nc'))
    
    return f"Agricultural Management revenue and cost written for year {yr_cal}"


def write_revenue_cost_non_ag(data: Data, yr_cal, path):
    """Calculate non_agricultural cost. """

        
    yr_idx = yr_cal - data.YR_CAL_BASE

    non_ag_dvar = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).assign_coords( region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(1024, data.NCELLS)})

    # Get the non-agricultural revenue/cost matrices
    non_ag_rev_mat = tools.non_ag_rk_to_xr(
        data, 
        non_ag_revenue.get_rev_matrix(data, yr_cal, ag_revenue.get_rev_matrices(data, yr_idx), data.lumaps[yr_cal]) 
    )   
    non_ag_cost_mat = tools.non_ag_rk_to_xr(
        data, 
        non_ag_cost.get_cost_matrix(data, ag_cost.get_cost_matrices(data, yr_idx), data.lumaps[yr_cal], yr_cal)    
    )   

    xr_revenue_non_ag = non_ag_dvar * non_ag_rev_mat
    xr_cost_non_ag = non_ag_dvar * non_ag_cost_mat

    # Regional level aggregation
    rev_non_ag_df_region = xr_revenue_non_ag.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year=yr_cal
        ).rename(columns={'lu': 'Land-use'})
    cost_non_ag_df_region = xr_cost_non_ag.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year=yr_cal
        ).rename(columns={'lu': 'Land-use'})

    # Australia level aggregation
    rev_non_ag_df_AUS = xr_revenue_non_ag.sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).rename(columns={'lu': 'Land-use'})
    cost_non_ag_df_AUS = xr_cost_non_ag.sum(dim='cell'
        ).to_dataframe('Value ($)'
        ).reset_index(
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).rename(columns={'lu': 'Land-use'})

    # Save to disk
    pd.concat([rev_non_ag_df_AUS, rev_non_ag_df_region]).to_csv(os.path.join(path, f'revenue_non_ag_{yr_cal}.csv'), index = False)
    pd.concat([cost_non_ag_df_AUS, cost_non_ag_df_region]).to_csv(os.path.join(path, f'cost_non_ag_{yr_cal}.csv'), index = False)

    # Append mosaic layer, then save to netcdf
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_revenue_non_ag_cat = xr.concat([non_ag_mosaic, xr_revenue_non_ag], dim='lu')
    xr_cost_non_ag_cat = xr.concat([non_ag_mosaic, xr_cost_non_ag], dim='lu')

    save2nc(xr_revenue_non_ag_cat, os.path.join(path, f'xr_revenue_non_ag_{yr_cal}.nc'))
    save2nc(xr_cost_non_ag_cat, os.path.join(path, f'xr_cost_non_ag_{yr_cal}.nc'))

    return f"Non-agricultural revenue and cost written for year {yr_cal}"



def write_transition_cost_ag2ag(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Calculate transition cost."""

    
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    # Get the decision variables for agricultural land-use
    ag_dvar_mrj_target = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]).assign_coords(region=('cell', data.REGION_NRM_NAME))
    ag_dvar_mrj_base = tools.ag_mrj_to_xr(data, (tools.lumap2ag_l_mrj(data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre])))

    ag_dvar_mrj_target = ag_dvar_mrj_target.rename({'lm': 'To water-supply', 'lu': 'To land-use'}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(1024, data.NCELLS)})

    ag_dvar_mrj_base = ag_dvar_mrj_base.rename({'lm': 'From water-supply', 'lu': 'From land-use'}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(1024, data.NCELLS)})

    # Get the transition cost matrices for agricultural land-use
    if yr_idx == 0:
        ag_transitions_cost_mat = {'Establishment cost': np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)}
    else:
        # Get the transition cost matrices for agricultural land-use
        ag_transitions_cost_mat = ag_transitions.get_transition_matrices_ag2ag_from_base_year(data, yr_idx, yr_cal_sim_pre, separate=True)
        
    ag_transitions_cost_mat = xr.DataArray(
        np.stack(list(ag_transitions_cost_mat.values())),
        coords={
            'Type': list(ag_transitions_cost_mat.keys()),
            'To water-supply': data.LANDMANS,
            'cell': range(data.NCELLS),
            'To land-use': data.AGRICULTURAL_LANDUSES
        }
    )

    cost_xr = ag_dvar_mrj_base * ag_dvar_mrj_target * ag_transitions_cost_mat
    cost_df_region = cost_xr.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Cost ($)'
        ).reset_index(
        ).assign(Year=yr_cal
        ).query('abs(`Cost ($)`) > 0')
    cost_df_AUS = cost_xr.sum(dim='cell'
        ).to_dataframe('Cost ($)'
        ).reset_index(
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).query('abs(`Cost ($)`) > 0')
    cost_df = pd.concat([cost_df_AUS, cost_df_region]).reset_index(drop=True)
                

    # Save the cost DataFrames
    cost_df = cost_df.replace({'dry':'Dryland', 'irr':'Irrigated'})
    cost_df.to_csv(os.path.join(path, f'cost_transition_ag2ag_{yr_cal}.csv'), index=False)
    
    save2nc(cost_xr, os.path.join(path, f'xr_cost_transition_ag2ag_{yr_cal}.nc'))

    return f"Agricultural to agricultural transition cost written for year {yr_cal}"




def write_transition_cost_ag2nonag(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Calculate transition cost."""

    
    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    # Get the non-agricultural decision variable
    ag_dvar_base = tools.ag_mrj_to_xr(data, (tools.lumap2ag_l_mrj(data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre]))
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_target = tools.non_ag_rk_to_xr(data, tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    ag_dvar_base = ag_dvar_base.rename({'lm': 'From water-supply', 'lu': 'From land-use'}).chunk({'cell': min(1024, data.NCELLS)})
    non_ag_dvar_target = non_ag_dvar_target.rename({'lu': 'To land-use'}).chunk({'cell': min(1024, data.NCELLS)})


    # Get the transition cost matirces for Non-Agricultural Land-use
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
        np.stack(list(non_ag_transitions_flat.values())),
        coords={
            'lu_source': pd.MultiIndex.from_tuples(
                list(non_ag_transitions_flat.keys()),
                names= ('To land-use', 'Cost type')
            ),
            'cell': range(data.NCELLS),
        }
    )

    cost_xr = ag_dvar_base * non_ag_transitions_flat.unstack('lu_source') * non_ag_dvar_target
    cost_df_region = cost_xr.groupby('region'
            ).sum(dim='cell'
            ).to_dataframe('Cost ($)'
            ).reset_index(
            ).assign(Year=yr_cal
            ).query('abs(`Cost ($)`) > 0')
    cost_df_AUS = cost_xr.sum(dim='cell'
            ).to_dataframe('Cost ($)'
            ).reset_index(
            ).assign(Year=yr_cal, region='AUSTRALIA'
            ).query('abs(`Cost ($)`) > 0')
    cost_df = pd.concat([cost_df_AUS, cost_df_region]).reset_index(drop=True)
    cost_df = cost_df.replace({'dry':'Dryland', 'irr':'Irrigated'})    
    cost_df.to_csv(os.path.join(path, f'cost_transition_ag2non_ag_{yr_cal}.csv'), index=False)
    
    save2nc(cost_xr, os.path.join(path, f'xr_transition_cost_ag2non_ag_{yr_cal}.nc'))
    
    return f"Agricultural to non-agricultural transition cost written for year {yr_cal}"



def write_transition_cost_apply_ag_man(data: Data):
    
    # The agricultural management transition cost are all zeros, so skip the calculation here
    # am_cost = ag_transitions.get_agricultural_management_transition_matrices(data)
    
    return "Agricultural Management transition cost processing completed"


def write_transition_cost_nonag2ag(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Calculate transition cost."""

    
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    # Get the decision variables for agricultural land-use
    ag_dvar = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get the transition cost matrices for Non-Agricultural Land-use
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

    cost_dfs = []
    for from_lu_desc, from_lu_idx in data.DESC2NONAGLU.items():
        for to_lu, to_lu_idx  in data.DESC2AGLU.items():
            for to_lm_idx, to_lm in enumerate(data.LANDMANS):
                for cost_type in non_ag_transitions_cost_mat[from_lu_desc].keys():
                    
                    
                    from_lu_cells = data.lumaps[yr_cal_sim_pre] == from_lu_idx          # Get the land-use index of the from land-use (r)
                    to_lu_cells = data.lumaps[yr_cal] == to_lu_idx                      # Get the land-use index of the to land-use (r*)
                    to_lm_cells = data.lmmaps[yr_cal] == to_lm_idx                      # Get the land-management index of the from land-management (r)
                    trans_cells =  from_lu_cells & to_lu_cells & to_lm_cells            # Get the land-use index of the from land-use (r*)
                    
                    if trans_cells.sum() == 0:
                        cost_dfs.append(
                            pd.DataFrame(
                                [{
                                    'region': data.REGION_NRM_NAME.iloc[0],
                                    'From land-use': from_lu_desc,
                                    'To land-use': to_lu,
                                    'To water-supply': to_lm,
                                    'Cost type': cost_type,
                                    'Cost ($)': 0,
                                    'Year': yr_cal
                                }]
                            )
                        )
                    else:
                        arr_dvar = ag_dvar[to_lm_idx, trans_cells, to_lu_idx]
                        arr_trans = non_ag_transitions_cost_mat[from_lu_desc][cost_type][to_lm_idx, trans_cells, to_lu_idx]
                        
                        cost_dfs.append(
                            (arr_dvar * arr_trans).groupby('region'
                            ).sum(dim='cell'
                            ).to_dataframe('Cost ($)'
                            ).reset_index(
                            ).rename(columns={'lu': 'To land-use', 'lm': 'To water-supply'}
                            ).assign(**{
                                'From land-use': from_lu_desc,
                                'Cost type': cost_type,
                                'Year': yr_cal
                            })
                        )
                    

    cost_df = pd.concat(cost_dfs, axis=0)
    cost_df = cost_df.replace({'dry':'Dryland', 'irr':'Irrigated'})
    cost_df.to_csv(os.path.join(path, f'cost_transition_non_ag2_ag_{yr_cal}.csv'), index=False)

    return f"Non-agricultural to agricultural transition cost written for year {yr_cal}"



def write_dvar_area(data: Data, yr_cal, path):
    
    # Get dvars
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).assign_coords({'region': ('cell', data.REGION_NRM_NAME)}
        ).chunk({'cell': min(1024, data.NCELLS)})
    non_ag_rj = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).assign_coords({'region': ('cell', data.REGION_NRM_NAME)}
        ).chunk({'cell': min(1024, data.NCELLS)})
    am_dvar_mrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).assign_coords({'region': ('cell', data.REGION_NRM_NAME)}
        ).chunk({'cell': min(1024, data.NCELLS)})

    
    # Calculate the real area in hectares
    real_area_r = xr.DataArray(data.REAL_AREA, dims=['cell'], coords={'cell': range(data.NCELLS)})

    area_ag = (ag_dvar_mrj * real_area_r)
    area_non_ag = (non_ag_rj * real_area_r)
    area_am = (am_dvar_mrj * real_area_r)
    
    # Expand dimension (has to be after multiplication to avoid double counting)
    area_ag = xr.concat([area_ag.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), area_ag], dim='lm')
    area_am = xr.concat([area_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), area_am], dim='lm')
    area_am = xr.concat([area_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), area_am], dim='lu')


    # Region level aggregation
    df_ag_area_region = area_ag.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply'}
        ).assign(Year=yr_cal
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area (ha)`) > 1e-6')
    df_non_ag_area_region = area_non_ag.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).rename(columns={'lu': 'Land-use'}
        ).assign(Year=yr_cal
        ).query('abs(`Area (ha)`) > 1e-6')
    df_am_area_region = area_am.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply', 'am': 'Type'}
        ).assign(Year=yr_cal
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area (ha)`) > 1e-6')
        
    # Australia level aggregation
    df_ag_area_AUS = area_ag.sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply'}
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area (ha)`) > 1e-6')
    df_non_ag_area_AUS = area_non_ag.sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).rename(columns={'lu': 'Land-use'}
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).query('abs(`Area (ha)`) > 1e-6')
    df_am_area_AUS = area_am.sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).rename(columns={'lu': 'Land-use', 'lm':'Water_supply', 'am': 'Type'}
        ).assign(Year=yr_cal, region='AUSTRALIA'
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area (ha)`) > 1e-6')
                                 

    pd.concat([df_ag_area_AUS, df_ag_area_region]).to_csv(os.path.join(path, f'area_agricultural_landuse_{yr_cal}.csv'), index = False)
    pd.concat([df_non_ag_area_AUS, df_non_ag_area_region]).to_csv(os.path.join(path, f'area_non_agricultural_landuse_{yr_cal}.csv'), index = False)
    pd.concat([df_am_area_AUS, df_am_area_region]).to_csv(os.path.join(path, f'area_agricultural_management_{yr_cal}.csv'), index = False)

    # Append the mosaic layer, then save to netcdf
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    area_ag_cat = xr.concat([ag_mosaic, area_ag], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    area_non_ag_cat = xr.concat([non_ag_mosaic, area_non_ag], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    area_am_cat = xr.concat([am_mosaic, area_am], dim='am')

    save2nc(area_ag_cat, os.path.join(path, f'xr_area_agricultural_landuse_{yr_cal}.nc'))
    save2nc(area_non_ag_cat, os.path.join(path, f'xr_area_non_agricultural_landuse_{yr_cal}.nc'))
    save2nc(area_am_cat, os.path.join(path, f'xr_area_agricultural_management_{yr_cal}.nc'))

    return f"Decision variable areas written for year {yr_cal}"



def write_area_transition_start_end(data: Data, path, yr_cal_end):

    
    # Get the end year
    yr_cal_start = data.YR_CAL_BASE

    real_area_r = xr.DataArray(data.REAL_AREA, dims=['cell'], coords={'cell': range(data.NCELLS)})

    # Get the decision variables for the start year
    ag_dvar_base_mrj = tools.ag_mrj_to_xr(data, tools.lumap2ag_l_mrj(data.lumaps[yr_cal_start], data.lmmaps[yr_cal_start])
        ).assign_coords({'region': ('cell', data.REGION_NRM_NAME)}
        ).rename({'lu':'From Land-use', 'lm':'From Water_supply'}
        ).chunk({'cell': min(1024, data.NCELLS)})

    ag_dvar_target_mrj = tools.ag_mrj_to_xr(
        data, tools.lumap2ag_l_mrj(data.lumaps[yr_cal_start], data.lmmaps[yr_cal_start])
        ).rename({'lu':'To Land-use', 'lm':'To Water_supply'}
        ).chunk({'cell': min(1024, data.NCELLS)})
        
    non_ag_dvar_target_rk = tools.non_ag_rk_to_xr(
        data, data.non_ag_dvars[yr_cal_end]
        ).rename({'lu':'To Land-use'}
        ).chunk({'cell': min(1024, data.NCELLS)})
        
    xr_ag2ag = ag_dvar_base_mrj * ag_dvar_target_mrj * real_area_r
    xr_ag2non_ag = ag_dvar_base_mrj * non_ag_dvar_target_rk * real_area_r
        
    transition_ag2ag = xr_ag2ag.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).groupby(['region', 'From Land-use', 'To Land-use']
        ).sum(
        ).reset_index(
        ).filter(['region', 'From Land-use', 'To Land-use', 'Area (ha)'])
    transition_ag2non_ag = xr_ag2non_ag.groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Area (ha)'
        ).reset_index(
        ).groupby(['region', 'From Land-use', 'To Land-use']
        ).sum(
        ).reset_index(
        ).filter(['region', 'From Land-use', 'To Land-use', 'Area (ha)'])

    # Write the transition matrix to a csv file
    pd.concat([transition_ag2ag, transition_ag2non_ag]
        ).to_csv(os.path.join(path, f'transition_matrix_start_end.csv'), index=False)
    
    save2nc(xr_ag2ag, os.path.join(path, f'xr_transition_area_ag2ag_start_end.nc'))
    save2nc(xr_ag2non_ag, os.path.join(path, f'xr_transition_area_ag2non_ag_start_end.nc'))

    return f"Area transition matrix written from year {data.YR_CAL_BASE} to {yr_cal_end}"





def write_crosstab(data: Data, yr_cal, path):
    """Write out land-use and production data"""

    if yr_cal > data.YR_CAL_BASE:

    
        simulated_year_list = sorted(list(data.lumaps.keys()))
        yr_idx_sim = simulated_year_list.index(yr_cal)
        yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]
        
        # Check if yr_cal_sim_pre meets the requirement
        assert yr_cal_sim_pre >= data.YR_CAL_BASE and yr_cal_sim_pre < yr_cal,\
            f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"

        lumap_pre = data.lumaps[yr_cal_sim_pre]
        lumap = data.lumaps[yr_cal]
        
        crosstab = pd.crosstab(lumap_pre,  [lumap, data.REGION_NRM_NAME], values=data.REAL_AREA, aggfunc=lambda x:x.sum(), margins = False
            ).unstack(
            ).reset_index(
            ).rename(
                columns={
                    'row_0': 'From land-use', 
                    'NRM_NAME': 'region', 
                    'col_0':'To land-use', 
                    0: 'Area (ha)'
                }
            ).dropna(
            ).replace({'From land-use': data.ALLLU2DESC, 'To land-use': data.ALLLU2DESC})
            
        switches = (crosstab.groupby('From land-use')['Area (ha)'].sum() - crosstab.groupby('To land-use')['Area (ha)'].sum()
            ).reset_index(
            ).rename(columns={'index':'Landuse'})
        
        
        crosstab['Year'] = yr_cal
        switches['Year'] = yr_cal
   
        crosstab.to_csv(os.path.join(path, f'crosstab-lumap_{yr_cal}.csv'), index=False)
        switches.to_csv(os.path.join(path, f'switches-lumap_{yr_cal}.csv'), index=False)

    return f"Land-use cross-tabulation and switches written for year {yr_cal}"




def write_ghg(data: Data, yr_cal, path):
    """GHG is written to disk no matter if GHG_EMISSIONS_LIMITS is 'off' or 'on'"""
 
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get GHG emissions limits used as constraints in model
    ghg_limits = 0 if settings.GHG_EMISSIONS_LIMITS == 'off' else data.GHG_TARGETS[yr_cal]

    # Get GHG emissions from model
    if yr_cal >= data.YR_CAL_BASE + 1:
        ghg_emissions = data.prod_data[yr_cal]['GHG']
    else:
        ghg_emissions = (ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True) * data.ag_dvars[settings.SIM_YEARS[0]]).sum()

    # Save GHG emissions to file
    df = pd.DataFrame({
        'Variable':['GHG_EMISSIONS_LIMIT_TCO2e','GHG_EMISSIONS_TCO2e'],
        'Emissions (t CO2e)':[ghg_limits, ghg_emissions]
        })
    df['Year'] = yr_cal
    df.to_csv(os.path.join(path, f'GHG_emissions_{yr_cal}.csv'), index=False)
    
    if settings.GHG_EMISSIONS_LIMITS == 'off':
        return 'WARNING: GHG emissions (total) calculated as `GHG_EMISSIONS_LIMITS` is set to "off"'
    else:
        return f"GHG emissions written for year {yr_cal}"





def write_ghg_separate(data: Data, yr_cal, path):
    '''
    GHG is written to disk no matter if GHG_EMISSIONS_LIMITS is 'off' or 'on'
    '''

    # Convert calendar year to year index.
    yr_idx = yr_cal - data.YR_CAL_BASE

    # -------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural landuse #
    # -------------------------------------------------------#

    # Get the ghg_df
    ag_g_xr = xr.Dataset(ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=False)
        ).rename({'dim_0':'cell'})
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(1024, data.NCELLS)})

    mindex = pd.MultiIndex.from_tuples(ag_g_xr.data_vars.keys(), names=['GHG_source', 'lm', 'lu'])
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, 'variable')
    ag_g_rsmj = ag_g_xr.to_dataarray().assign_coords(mindex_coords).chunk({'cell': min(1024, data.NCELLS)}).unstack()
    ag_g_rsmj['GHG_source'] = ag_g_rsmj['GHG_source'].to_series().replace(GHG_NAMES)
    
    # Calculate GHG emissions
    ghg_e = ag_g_rsmj * ag_dvar_mrj

    # Expand dimension (has to be after multiplication to avoid double counting)
    ghg_e = xr.concat([ghg_e.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), ghg_e], dim='lm')
    ghg_e = xr.concat([ghg_e.sum(dim='GHG_source', keepdims=True).assign_coords(GHG_source=['ALL']), ghg_e], dim='GHG_source')

    # Regional level aggregation
    ghg_df_region = ghg_e.groupby('region'
        ).sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).rename(columns={'lu':'Land-use', 'lm':'Water_supply', 'GHG_source':'Source'}
        ).assign(Year=yr_cal, Type='Agricultural land-use',
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Value (t CO2e)`) > 1e-3') 
    
    # Australia level aggregation
    ghg_df_AUS = ghg_e.sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).rename(columns={'lu':'Land-use', 'lm':'Water_supply', 'GHG_source':'Source'}
        ).assign(Year=yr_cal, Type='Agricultural land-use', region='AUSTRALIA'
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Value (t CO2e)`) > 1e-3') 
    
    # Save table to disk
    pd.concat([ghg_df_AUS, ghg_df_region]).to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_landuse_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    ghg_e_cat = xr.concat([ag_mosaic, ghg_e], dim='lu')

    save2nc(ghg_e_cat, os.path.join(path, f'xr_GHG_ag_{yr_cal}.nc'))


    # -----------------------------------------------------------#
    # Get greenhouse gas emissions from Non-Agricultural Land-use #
    # -----------------------------------------------------------#
    
    # Get the non_ag GHG reduction
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(1024, data.NCELLS)})

    non_ag_g_rk = tools.non_ag_rk_to_xr(
        data,
        non_ag_ghg.get_ghg_matrix(
            data, 
            ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True), 
            data.lumaps[yr_cal]
        )
    )

    # Calculate GHG emissions for non-agricultural land use
    xr_ghg_non_ag = non_ag_dvar_rk * non_ag_g_rk

    # Regional level aggregation
    ghg_df_region = xr_ghg_non_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).rename(columns={'lu': 'Land-use'}
        ).assign(Year=yr_cal, Type='Non-Agricultural Land-use'
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).query('abs(`Value (t CO2e)`) > 1e-3') 
        
    # Australia level aggregation
    ghg_df_AUS = xr_ghg_non_ag.sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).rename(columns={'lu': 'Land-use'}
        ).assign(Year=yr_cal, Type='Non-Agricultural Land-use', region='AUSTRALIA'
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).query('abs(`Value (t CO2e)`) > 1e-3') 
        
    # Save table to disk
    pd.concat([ghg_df_AUS, ghg_df_region]).to_csv(os.path.join(path, f'GHG_emissions_separate_no_ag_reduction_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_ghg_non_ag_cat = xr.concat([non_ag_mosaic, xr_ghg_non_ag], dim='lu')

    # Save xarray data to netCDF
    save2nc(xr_ghg_non_ag_cat, os.path.join(path, f'xr_GHG_non_ag_{yr_cal}.nc'))
    
    
    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural management          #
    # -------------------------------------------------------------------#

    # Get the ag_man_g_mrj
    ag_man_dvar_mrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(1024, data.NCELLS)})

    ag_man_g_mrj = tools.am_mrj_to_xr(
        data, 
        ag_ghg.get_agricultural_management_ghg_matrices(data, yr_idx)
    )

    # Calculate GHG emissions for agricultural management
    xr_ghg_ag_man = ag_man_dvar_mrj * ag_man_g_mrj

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_ghg_ag_man = xr.concat([xr_ghg_ag_man.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_ghg_ag_man], dim='lm')
    xr_ghg_ag_man = xr.concat([xr_ghg_ag_man.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_ghg_ag_man], dim='lu')

    # Regional level aggregation
    ghg_df_region = xr_ghg_ag_man.groupby('region'
        ).sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).rename(columns={'lm': 'Water_supply', 'lu': 'Land-use', 'am': 'Agricultural Management Type'}
        ).assign(Year=yr_cal, Type='Agricultural Management'
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).query('abs(`Value (t CO2e)`) > 1e-3') 
        
    # Australia level aggregation
    ghg_df_AUS = xr_ghg_ag_man.sum('cell'
        ).to_dataframe('Value (t CO2e)'
        ).reset_index(
        ).rename(columns={'lm': 'Water_supply', 'lu': 'Land-use', 'am': 'Agricultural Management Type'}
        ).assign(Year=yr_cal, Type='Agricultural Management', region='AUSTRALIA'
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).query('abs(`Value (t CO2e)`) > 1e-3') 
        
    # Save table to disk
    pd.concat([ghg_df_AUS, ghg_df_region]).to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_management_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_ghg_ag_man_cat = xr.concat([am_mosaic, xr_ghg_ag_man], dim='am')

    # Save xarray data to netCDF
    save2nc(xr_ghg_ag_man_cat, os.path.join(path, f'xr_GHG_ag_management_{yr_cal}.nc'))


    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from landuse transformation penalties #
    # -------------------------------------------------------------------#

    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx_sim = simulated_year_list.index(yr_cal)

    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    if yr_cal == data.YR_CAL_BASE:
        pass
    else:
        yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]
        ghg_t_dict = ag_ghg.get_ghg_transition_emissions(data, data.lumaps[yr_cal_sim_pre], separate=True)
        ghg_t_smrj = xr.DataArray(
            np.stack(list(ghg_t_dict.values()), axis=0),
            dims=['Type', 'lm', 'cell', 'lu'],
            coords={
                'Type': list(ghg_t_dict.keys()),
                'lm': data.LANDMANS,
                'cell': range(data.NCELLS),
                'lu': data.AGRICULTURAL_LANDUSES
            }
        )

        # Calculate GHG emissions for transition penalties
        xr_ghg_transition = ghg_t_smrj * ag_dvar_mrj
        
        # Regional level aggregation
        ghg_df_region = xr_ghg_transition.groupby('region'
            ).sum('cell'
            ).to_dataframe('Value (t CO2e)'
            ).reset_index(
            ).rename(columns={'lu': 'Land-use', 'lm': 'Water_supply'}
            ).assign(Year=yr_cal
            ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
            ).query('abs(`Value (t CO2e)`) > 1e-3')

        # Australia level aggregation
        ghg_df_AUS = xr_ghg_transition.sum('cell'
            ).to_dataframe('Value (t CO2e)'
            ).reset_index(
            ).rename(columns={'lu': 'Land-use', 'lm': 'Water_supply'}
            ).assign(Year=yr_cal, region='AUSTRALIA'
            ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
            ).query('abs(`Value (t CO2e)`) > 1e-3')

        # Save table to disk
        pd.concat([ghg_df_AUS, ghg_df_region]).to_csv(os.path.join(path, f'GHG_emissions_separate_transition_penalty_{yr_cal}.csv'), index=False)
        
        # Save xarray data to netCDF
        save2nc(xr_ghg_transition, os.path.join(path, f'xr_transition_GHG_{yr_cal}.nc'))
    
    if settings.GHG_EMISSIONS_LIMITS == 'off':
        return 'WARNING: GHG emissions calculated as `GHG_EMISSIONS_LIMITS` is set to "off"'
    else:
        return f"Separate GHG emissions written for year {yr_cal}"




def write_ghg_offland_commodity(data: Data, yr_cal, path):
    """Off-land commodity GHG emissions are written to disk no matter if GHG_EMISSIONS_LIMITS is 'off' or 'on'"""

    offland_ghg = data.OFF_LAND_GHG_EMISSION.query(f'YEAR == {yr_cal}').rename(columns={'YEAR':'Year'})
    offland_ghg.to_csv(os.path.join(path, f'GHG_emissions_offland_commodity_{yr_cal}.csv'), index = False)
    
    if settings.GHG_EMISSIONS_LIMITS == 'off':
        return 'WARNING: Off-land commodity GHG emissions calculate as `GHG_EMISSIONS_LIMITS` is set to "off"'
    else:
        return f"Off-land commodity GHG emissions written for year {yr_cal}"




def write_water(data: Data, yr_cal, path):
    """ Water yield is written to disk no matter if `WATER_LIMITS` is on or off. """

    yr_idx = yr_cal - data.YR_CAL_BASE
    region2code = {v: k for k, v in data.WATER_REGION_NAMES.items()}

    # Get the decision variables
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(1024, data.NCELLS)})
    non_ag_dvar_rj = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(1024, data.NCELLS)})
    am_dvar_mrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(1024, data.NCELLS)})
        
    # Get water target and domestic use
    w_limit_inside_luto = xr.DataArray(
        list(data.WATER_YIELD_TARGETS.values()),
        dims=['region_water'],
        coords={'region_water': list(data.WATER_YIELD_TARGETS.keys())}
    )
    domestic_water_use = xr.DataArray(
        list(data.WATER_USE_DOMESTIC.values()),
        dims=['region_water'],
        coords={'region_water': list(data.WATER_USE_DOMESTIC.keys())}
    )

    # ------------------------------- Get water yield without CCI -----------------------------------

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

    # Expand dimension (has to be after calculation to avoid double counting)
    xr_ag_wny = xr.concat([xr_ag_wny.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_ag_wny], dim='lm')
    xr_am_wny = xr.concat([xr_am_wny.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_am_wny], dim='lm')
    xr_am_wny = xr.concat([xr_am_wny.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_am_wny], dim='lu')

    ag_wny = xr_ag_wny.groupby('region_water'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Landuse'
        ).replace({'region_water': data.WATER_REGION_NAMES})
    non_ag_wny = xr_non_ag_wny.groupby('region_water'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Non-Agricultural Land-use'
        ).replace({'region_water': data.WATER_REGION_NAMES})
    am_wny = xr_am_wny.groupby('region_water'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Management'
        ).replace({'region_water': data.WATER_REGION_NAMES})
    wny_inside_luto = pd.concat([ag_wny, non_ag_wny, am_wny], ignore_index=True
        ).assign(Year=yr_cal
        ).rename(columns={
            'region_water': 'Region',
            'lu':'Landuse',
            'am':'Agri-Management',
            'lm':'Water Supply'}
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).dropna(axis=0, how='all')
        
    wny_inside_luto.to_csv(os.path.join(path, f'water_yield_separate_watershed_{yr_cal}.csv'), index=False)


    # ------------------------------- Get water yield outside LUTO study region -----------------------------------
    wny_outside_luto_study_area = xr.DataArray(
        list(data.WATER_OUTSIDE_LUTO_BY_CCI.loc[data.YR_CAL_BASE].to_dict().values()),
        dims=['region_water'],
        coords={'region_water': list(data.WATER_REGION_INDEX_R.keys())},
    )


    # ------------------------------- Get water yield change (delta) under CCI -----------------------------------

    # Get CCI matrix
    if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on':
        ag_w_mrj_base = tools.ag_mrj_to_xr(data, ag_water.get_water_net_yield_matrices(data, 0))
        ag_w_mrj_base = xr.concat([ag_w_mrj_base.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), ag_w_mrj_base], dim='lm')
        wny_outside_luto_study_area_base = np.array(list(data.WATER_OUTSIDE_LUTO_BY_CCI.loc[data.YR_CAL_BASE].to_dict().values()))
    elif settings.WATER_CLIMATE_CHANGE_IMPACT == 'off':
        ag_w_mrj_base = tools.ag_mrj_to_xr(data, ag_water.get_water_net_yield_matrices(data, 0, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR))
        ag_w_mrj_base = xr.concat([ag_w_mrj_base.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), ag_w_mrj_base], dim='lm')
        wny_outside_luto_study_area_base = np.array(list(data.WATER_OUTSIDE_LUTO_HIST.values()))

    ag_w_mrj_CCI = ag_w_mrj - ag_w_mrj_base
    wny_outside_luto_study_area_CCI = wny_outside_luto_study_area - wny_outside_luto_study_area_base



    # Calculate water net yield (delta) under CCI; 
    #   we use BASE_YEAR (2010) dvar_mrj to calculate CCI, 
    #   because the CCI calculated with base year (previouse year) 
    #   dvar_mrj includes wny from land-use change
    xr_ag_dvar_BASE = tools.ag_mrj_to_xr(data, data.AG_L_MRJ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME))
    xr_ag_dvar_BASE = xr.concat([xr_ag_dvar_BASE.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_ag_dvar_BASE], dim='lm')

    xr_ag_wny_CCI = xr_ag_dvar_BASE * ag_w_mrj_CCI


    # Get the CCI impact (delta)
    CCI_impact = (
        xr_ag_wny_CCI.groupby('region_water').sum(['cell','lm', 'lu']) 
        + wny_outside_luto_study_area_CCI
    )

    # ------------------------------- Organise water yield components -----------------------------------

    # Water net yield for watershed regions
    wny_inside_luto_sum = wny_inside_luto\
        .query('`Water Supply` != "ALL" and `Agri-Management` != "ALL"')\
        .groupby('Region')[['Water Net Yield (ML)']]\
        .sum()
    wny_inside_luto_sum = xr.DataArray(
        wny_inside_luto_sum['Water Net Yield (ML)'].values, 
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
        ).replace({'Region': data.WATER_REGION_NAMES}
        ).assign(Year=yr_cal)
        
    water_other_records.to_csv(os.path.join(path, f'water_yield_limits_and_public_land_{yr_cal}.csv'), index=False)

    # Water yield for NRM region
    ag_wny = (ag_w_mrj * ag_dvar_mrj
        ).groupby('region_NRM'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Landuse'
        ).replace({'region_NRM': data.WATER_REGION_NAMES})
    non_ag_wny = (non_ag_w_rk * non_ag_dvar_rj
        ).groupby('region_NRM'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Non-Agricultural Land-use'
        ).replace({'region_NRM': data.WATER_REGION_NAMES})
    am_wny = (am_dvar_mrj * ag_man_w_mrj
        ).groupby('region_NRM'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Management'
        ).replace({'region_NRM': data.WATER_REGION_NAMES})
    wny_NRM = pd.concat([ag_wny, non_ag_wny, am_wny], ignore_index=True
        ).assign(Year=yr_cal
        ).rename(columns={
            'region_water': 'Region',
            'lu':'Landuse',
            'am':'Agri-Management',
            'lm':'Water Supply'}
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).dropna(axis=0, how='all')
        
    wny_NRM.to_csv(os.path.join(path, f'water_yield_separate_NRM_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_ag_wny_cat = xr.concat([ag_mosaic, xr_ag_wny], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_non_ag_wny_cat = xr.concat([non_ag_mosaic, xr_non_ag_wny], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_am_wny_cat = xr.concat([am_mosaic, xr_am_wny], dim='am')

    save2nc(xr_ag_wny_cat, os.path.join(path, f'xr_water_yield_ag_{yr_cal}.nc'))
    save2nc(xr_non_ag_wny_cat, os.path.join(path, f'xr_water_yield_non_ag_{yr_cal}.nc'))
    save2nc(xr_am_wny_cat, os.path.join(path, f'xr_water_yield_ag_management_{yr_cal}.nc'))


    # ------------ Write the original targets for watershed regions being relaxed under CCI -----------------
    water_relaxed_region_raw_targets = pd.DataFrame(
        [[k, v, data.WATER_REGION_NAMES[k]] for k, v in data.WATER_RELAXED_REGION_RAW_TARGETS.items()], 
        columns=['Region Id', 'Target', 'Region Name']
    )
    water_relaxed_region_raw_targets['Year'] = yr_cal
    water_relaxed_region_raw_targets.to_csv(os.path.join(path, f'water_yield_relaxed_region_raw_{yr_cal}.csv'), index=False)

    return f"Water yield data written for year {yr_cal}"


def write_biodiversity_overall_quality_scores(data: Data, yr_cal, path):
    ''' Biodiversity overall quality scores are always written to disk. '''
    
    yr_idx_previouse = sorted(data.lumaps.keys()).index(yr_cal) - 1
    yr_cal_previouse = sorted(data.lumaps.keys())[yr_idx_previouse]
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the biodiversity scores b_mrj
    bio_ag_priority_mrj =  tools.ag_mrj_to_xr(data, ag_biodiversity.get_bio_quality_score_mrj(data))   
    bio_am_priority_amrj = tools.am_mrj_to_xr(data, ag_biodiversity.get_ag_mgt_biodiversity_matrices(data, bio_ag_priority_mrj.values, yr_idx))
    bio_non_ag_priority_rk = tools.non_ag_rk_to_xr(data, non_ag_biodiversity.get_breq_matrix(data,bio_ag_priority_mrj.values, data.lumaps[yr_cal_previouse]))

    if yr_idx_previouse < 0: # this means now is the base year, hence no ag-man and non-ag applied
        bio_am_priority_amrj *= 0.0
        bio_non_ag_priority_rk *= 0.0


    # Get the decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    ag_mam_dvar_mrj =  tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))


    # Calculate the biodiversity scores under pre-1750 conditions
    base_yr_score = bio_ag_priority_mrj.sel(lu='Unallocated - natural land', lm='dry').sum().item()
    
    # Calculate xarray biodiversity scores
    xr_priority_ag = ag_dvar_mrj * bio_ag_priority_mrj
    xr_priority_non_ag = non_ag_dvar_rk * bio_non_ag_priority_rk
    xr_priority_am = ag_mam_dvar_mrj * bio_am_priority_amrj
    
    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_priority_ag = xr.concat([xr_priority_ag.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_priority_ag], dim='lm')
    xr_priority_am = xr.concat([xr_priority_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_priority_am], dim='lm')
    xr_priority_am = xr.concat([xr_priority_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_priority_am], dim='lu')
    
    priority_ag_region = (xr_priority_ag
        ).groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:( (x['Area Weighted Score (ha)'] / base_yr_score) * 100) 
        ).assign(Type='Agricultural Landuse', Year=yr_cal)
    priority_ag_AUS = (xr_priority_ag
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:( (x['Area Weighted Score (ha)'] / base_yr_score) * 100) 
        ).assign(Type='Agricultural Landuse', Year=yr_cal, region='AUSTRALIA')

    priority_non_ag_region = (xr_priority_non_ag
        ).groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:( x['Area Weighted Score (ha)'] / base_yr_score * 100)
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)
    priority_non_ag_AUS = (xr_priority_non_ag
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:( x['Area Weighted Score (ha)'] / base_yr_score * 100)
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    priority_am_region = (xr_priority_am
        ).groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:( x['Area Weighted Score (ha)'] / base_yr_score * 100)
        ).dropna(
        ).assign(Type='Agricultural Management', Year=yr_cal)
    priority_am_AUS = (xr_priority_am
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:( x['Area Weighted Score (ha)'] / base_yr_score * 100)
        ).dropna(
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')


    # Save the biodiversity scores
    df = pd.concat([
        priority_ag_region, 
        priority_ag_AUS, 
        priority_non_ag_region, 
        priority_non_ag_AUS, 
        priority_am_region, 
        priority_am_AUS], axis=0
        ).rename(columns={
            'lu':'Landuse',
            'lm':'Water_supply',
            'am':'Agri-Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Base Year Level (%)'}
        ).reset_index(drop=True
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'biodiversity_overall_priority_scores_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_priority_ag_cat = xr.concat([ag_mosaic, xr_priority_ag], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_priority_non_ag_cat = xr.concat([non_ag_mosaic, xr_priority_non_ag], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_priority_am_cat = xr.concat([am_mosaic, xr_priority_am], dim='am')

    # Save xarray data to netCDF
    save2nc(xr_priority_ag_cat, os.path.join(path, f'xr_biodiversity_overall_priority_ag_{yr_cal}.nc'))
    save2nc(xr_priority_non_ag_cat, os.path.join(path, f'xr_biodiversity_overall_priority_non_ag_{yr_cal}.nc'))
    save2nc(xr_priority_am_cat, os.path.join(path, f'xr_biodiversity_overall_priority_ag_management_{yr_cal}.nc'))
    
    return f"Biodiversity overall priority scores written for year {yr_cal}"




def write_biodiversity_GBF2_scores(data: Data, yr_cal, path):
    ''' Biodiversity GBF2 only being written to disk when `BIODIVERSITY_TARGET_GBF_2` is not 'off' '''

    # Do nothing if biodiversity limits are off and no need to report
    if settings.BIODIVERSITY_TARGET_GBF_2 == 'off':
        return 'Skipped: Biodiversity GBF2 scores not written as `BIODIVERSITY_TARGET_GBF_2` is set to "off"'

        
    # Unpack the ag managements and land uses
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get the priority degraded areas 
    GBF2_MASK_area_ha = ag_biodiversity.get_GBF2_MASK_area(data)
    
    # Get the priority degraded areas score
    priority_degraded_area_score_r = xr.DataArray(
        GBF2_MASK_area_ha,
        dims=['cell'],
        coords={'cell':range(data.NCELLS)}
    )

    # Get the impacts of each ag/non-ag/am to vegetation matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ajr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
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

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_gbf2_ag = xr.concat([xr_gbf2_ag.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf2_ag], dim='lm')
    xr_gbf2_am = xr.concat([xr_gbf2_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf2_am], dim='lm')
    xr_gbf2_am = xr.concat([xr_gbf2_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_gbf2_am], dim='lu')

    # Regional level aggregation
    GBF2_score_ag_region = xr_gbf2_ag.groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:((x['Area Weighted Score (ha)'] / total_priority_degraded_area) * 100)
        ).assign(Type='Agricultural Landuse', Year=yr_cal)
    GBF2_score_non_ag_region = xr_gbf2_non_ag.groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:(x['Area Weighted Score (ha)'] / total_priority_degraded_area * 100)
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)  
    GBF2_score_am_region = xr_gbf2_am.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).assign(Relative_Contribution_Percentage = lambda x:(x['Area Weighted Score (ha)'] / total_priority_degraded_area * 100)
        ).assign(Type='Agricultural Management', Year=yr_cal)
        
    # Australia level aggregation
    GBF2_score_ag_AUS = xr_gbf2_ag.sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:((x['Area Weighted Score (ha)'] / total_priority_degraded_area) * 100)
        ).assign(Type='Agricultural Landuse', Year=yr_cal, region='AUSTRALIA')
    GBF2_score_non_ag_AUS = xr_gbf2_non_ag.sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:(x['Area Weighted Score (ha)'] / total_priority_degraded_area * 100)
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')  
    GBF2_score_am_AUS = xr_gbf2_am.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).assign(Relative_Contribution_Percentage = lambda x:(x['Area Weighted Score (ha)'] / total_priority_degraded_area * 100)
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')
        
    # Combine regional and Australia level data
    GBF2_score_ag = pd.concat([GBF2_score_ag_region, GBF2_score_ag_AUS], axis=0)
    GBF2_score_non_ag = pd.concat([GBF2_score_non_ag_region, GBF2_score_non_ag_AUS], axis=0)
    GBF2_score_am = pd.concat([GBF2_score_am_region, GBF2_score_am_AUS], axis=0)
        
    # Fill nan to empty dataframes
    if GBF2_score_ag.empty:
        GBF2_score_ag.loc[0] = 0
        GBF2_score_ag = GBF2_score_ag.astype({'Type':str, 'lu':str,'Year':'int'})
        GBF2_score_ag.loc[0, ['Type', 'lu' ,'Year']] = ['Agricultural Landuse', 'Apples', yr_cal]

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
            'am':'Agri-Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)',
            'Priority_Target':'Priority Target (%)'}
        ).reset_index(drop=True
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        )
    df.to_csv(os.path.join(path, f'biodiversity_GBF2_priority_scores_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf2_ag_cat = xr.concat([ag_mosaic, xr_gbf2_ag], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf2_non_ag_cat = xr.concat([non_ag_mosaic, xr_gbf2_non_ag], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_gbf2_am_cat = xr.concat([am_mosaic, xr_gbf2_am], dim='am')

    # Save xarray data to netCDF
    save2nc(xr_gbf2_ag_cat, os.path.join(path, f'xr_biodiversity_GBF2_priority_ag_{yr_cal}.nc'))
    save2nc(xr_gbf2_non_ag_cat, os.path.join(path, f'xr_biodiversity_GBF2_priority_non_ag_{yr_cal}.nc'))
    save2nc(xr_gbf2_am_cat, os.path.join(path, f'xr_biodiversity_GBF2_priority_ag_management_{yr_cal}.nc'))
    
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
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    
    # Get vegetation matrices for the year
    vegetation_score_vr = xr.DataArray(
        ag_biodiversity.get_GBF3_NVIS_matrices_vr(data),
        dims=['group','cell'],
        coords={'group':list(data.BIO_GBF3_NVIS_ID2DESC.values()),  'cell':range(data.NCELLS)}
    ).chunk({'cell': min(1024, data.NCELLS), 'group': 1})

    # Get the impacts of each ag/non-ag/am to vegetation matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
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
    xr_gbf3_ag = xr.concat([xr_gbf3_ag.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf3_ag], dim='lm')
    xr_gbf3_am = xr.concat([xr_gbf3_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf3_am], dim='lm')
    xr_gbf3_am = xr.concat([xr_gbf3_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_gbf3_am], dim='lu')

    # Regional level aggregation
    GBF3_score_ag_region = xr_gbf3_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(veg_base_score_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)

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
        ).assign(Type='Agricultural Landuse', Year=yr_cal, region='AUSTRALIA')

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
            'am':'Agri-Management',
            'group':'Vegetation Group',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF3_NVIS_scores_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf3_ag_cat = xr.concat([ag_mosaic, xr_gbf3_ag], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf3_non_ag_cat = xr.concat([non_ag_mosaic, xr_gbf3_non_ag], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_gbf3_am_cat = xr.concat([am_mosaic, xr_gbf3_am], dim='am')

    # Save xarray data to netCDF
    save2nc(xr_gbf3_ag_cat, os.path.join(path, f'xr_biodiversity_GBF3_NVIS_ag_{yr_cal}.nc'))
    save2nc(xr_gbf3_non_ag_cat, os.path.join(path, f'xr_biodiversity_GBF3_NVIS_non_ag_{yr_cal}.nc'))
    save2nc(xr_gbf3_am_cat, os.path.join(path, f'xr_biodiversity_GBF3_NVIS_ag_management_{yr_cal}.nc'))
    
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
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get IBRA bioregion matrices for the year
    bioregion_score_vr = xr.DataArray(
        ag_biodiversity.get_GBF3_IBRA_matrices_vr(data),
        dims=['group','cell'],
        coords={'group':list(data.BIO_GBF3_IBRA_ID2DESC.values()),  'cell':range(data.NCELLS)}
    ).chunk({'cell': min(1024, data.NCELLS), 'group': 1})

    # Get the impacts of each ag/non-ag/am to bioregion matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
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
    xr_gbf3_ibra_ag = xr.concat([xr_gbf3_ibra_ag.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf3_ibra_ag], dim='lm')
    xr_gbf3_ibra_am = xr.concat([xr_gbf3_ibra_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf3_ibra_am], dim='lm')
    xr_gbf3_ibra_am = xr.concat([xr_gbf3_ibra_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_gbf3_ibra_am], dim='lu')

    # Regional level aggregation
    GBF3_IBRA_score_ag_region = xr_gbf3_ibra_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(bioregion_base_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)

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
        ).assign(Type='Agricultural Landuse', Year=yr_cal, region='AUSTRALIA')

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
            'am':'Agri-Management',
            'group':'IBRA Bioregion',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF3_IBRA_scores_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf3_ibra_ag_cat = xr.concat([ag_mosaic, xr_gbf3_ibra_ag], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf3_ibra_non_ag_cat = xr.concat([non_ag_mosaic, xr_gbf3_ibra_non_ag], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_gbf3_ibra_am_cat = xr.concat([am_mosaic, xr_gbf3_ibra_am], dim='am')

    # Save xarray data to netCDF
    save2nc(xr_gbf3_ibra_ag_cat, os.path.join(path, f'xr_biodiversity_GBF3_IBRA_ag_{yr_cal}.nc'))
    save2nc(xr_gbf3_ibra_non_ag_cat, os.path.join(path, f'xr_biodiversity_GBF3_IBRA_non_ag_{yr_cal}.nc'))
    save2nc(xr_gbf3_ibra_am_cat, os.path.join(path, f'xr_biodiversity_GBF3_IBRA_ag_management_{yr_cal}.nc'))

    return f"Biodiversity GBF3 IBRA scores written for year {yr_cal}"



def write_biodiversity_GBF4_SNES_scores(data: Data, yr_cal: int, path) -> None:
    ''' Biodiversity GBF4 SNES only being written to disk when `BIODIVERSITY_TARGET_GBF_4_SNES` is 'on' '''

    if not settings.BIODIVERSITY_TARGET_GBF_4_SNES == "on":
        return "Skipped: Biodiversity GBF4 SNES scores not written as `BIODIVERSITY_TARGET_GBF_4_SNES` is set to 'off'"

    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    
    # Get the biodiversity scores for the year
    bio_snes_sr = xr.DataArray(
        ag_biodiversity.get_GBF4_SNES_matrix_sr(data),
        dims=['species','cell'],
        coords={'species':data.BIO_GBF4_SNES_SEL_ALL, 'cell':np.arange(data.NCELLS)}
    )

    # Apply habitat contribution from ag/am/non-ag land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    ).unstack()

    # Get the base year biodiversity scores
    bio_snes_scores = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_SNES.csv')
    idx_row = [bio_snes_scores.query('SCIENTIFIC_NAME == @i').index[0] for i in data.BIO_GBF4_SNES_SEL_ALL]
    idx_all_score = [bio_snes_scores.columns.get_loc(f'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_{col}') for col in data.BIO_GBF4_PRESENCE_SNES_SEL]
    idx_outside_score =  [bio_snes_scores.columns.get_loc(f'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_{col}') for col in data.BIO_GBF4_PRESENCE_SNES_SEL]

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
    xr_gbf4_snes_ag = xr.concat([xr_gbf4_snes_ag.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf4_snes_ag], dim='lm')
    xr_gbf4_snes_am = xr.concat([xr_gbf4_snes_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf4_snes_am], dim='lm')
    xr_gbf4_snes_am = xr.concat([xr_gbf4_snes_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_gbf4_snes_am], dim='lu')

    # Regional level aggregation
    GBF4_score_ag_region = xr_gbf4_snes_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)
        
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
        ).assign(Type='Agricultural Landuse', Year=yr_cal, region='AUSTRALIA')
        
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
            'am':'Agri-Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)',
            'Target_by_Percent':'Target by Percent (%)'}).reset_index(drop=True
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF4_SNES_scores_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf4_snes_ag_cat = xr.concat([ag_mosaic, xr_gbf4_snes_ag], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf4_snes_non_ag_cat = xr.concat([non_ag_mosaic, xr_gbf4_snes_non_ag], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_gbf4_snes_am_cat = xr.concat([am_mosaic, xr_gbf4_snes_am], dim='am')

    # Save xarray data to netCDF
    save2nc(xr_gbf4_snes_ag_cat, os.path.join(path, f'xr_biodiversity_GBF4_SNES_ag_{yr_cal}.nc'))
    save2nc(xr_gbf4_snes_non_ag_cat, os.path.join(path, f'xr_biodiversity_GBF4_SNES_non_ag_{yr_cal}.nc'))
    save2nc(xr_gbf4_snes_am_cat, os.path.join(path, f'xr_biodiversity_GBF4_SNES_ag_management_{yr_cal}.nc'))
    
    return f"Biodiversity GBF4 SNES scores written for year {yr_cal}"





def write_biodiversity_GBF4_ECNES_scores(data: Data, yr_cal: int, path) -> None:
    ''' Biodiversity GBF4 ECNES only being written to disk when `BIODIVERSITY_TARGET_GBF_4_ECNES` is 'on' '''
    
    if not settings.BIODIVERSITY_TARGET_GBF_4_ECNES == "on":
        return "Skipped: Biodiversity GBF4 ECNES scores not written as `BIODIVERSITY_TARGET_GBF_4_ECNES` is set to 'off'"

    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    
    # Get the biodiversity scores for the year
    bio_ecnes_sr = xr.DataArray(
        ag_biodiversity.get_GBF4_ECNES_matrix_sr(data),
        dims=['species','cell'],
        coords={'species':data.BIO_GBF4_ECNES_SEL_ALL, 'cell':np.arange(data.NCELLS)}
    ).chunk({'cell': min(1024, data.NCELLS), 'species': 1})

    # Apply habitat contribution from ag/am/non-ag land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)
        }
    ).unstack()

    # Get the base year biodiversity scores
    bio_ecnes_scores = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_ECNES.csv')
    idx_row = [bio_ecnes_scores.query('COMMUNITY == @i').index[0] for i in data.BIO_GBF4_ECNES_SEL_ALL]
    idx_all_score = [bio_ecnes_scores.columns.get_loc(f'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_{col}') for col in data.BIO_GBF4_PRESENCE_ECNES_SEL]
    idx_outside_score = [bio_ecnes_scores.columns.get_loc(f'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_{col}') for col in data.BIO_GBF4_PRESENCE_ECNES_SEL]

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

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_gbf4_ecnes_ag = xr.concat([xr_gbf4_ecnes_ag.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf4_ecnes_ag], dim='lm')
    xr_gbf4_ecnes_am = xr.concat([xr_gbf4_ecnes_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf4_ecnes_am], dim='lm')
    xr_gbf4_ecnes_am = xr.concat([xr_gbf4_ecnes_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_gbf4_ecnes_am], dim='lu')

    # Regional level aggregation
    GBF4_score_ag_region = xr_gbf4_ecnes_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)

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

    # Australia level aggregation
    GBF4_score_ag_AUS = xr_gbf4_ecnes_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal, region='AUSTRALIA')

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
            'am':'Agri-Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)',
            'Target_by_Percent': 'Target by Percent (%)'}
        ).reset_index(drop=True
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF4_ECNES_scores_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf4_ecnes_ag_cat = xr.concat([ag_mosaic, xr_gbf4_ecnes_ag], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf4_ecnes_non_ag_cat = xr.concat([non_ag_mosaic, xr_gbf4_ecnes_non_ag], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_gbf4_ecnes_am_cat = xr.concat([am_mosaic, xr_gbf4_ecnes_am], dim='am')

    # Save xarray data to netCDF
    save2nc(xr_gbf4_ecnes_ag_cat, os.path.join(path, f'xr_biodiversity_GBF4_ECNES_ag_{yr_cal}.nc'))
    save2nc(xr_gbf4_ecnes_non_ag_cat, os.path.join(path, f'xr_biodiversity_GBF4_ECNES_non_ag_{yr_cal}.nc'))
    save2nc(xr_gbf4_ecnes_am_cat, os.path.join(path, f'xr_biodiversity_GBF4_ECNES_ag_management_{yr_cal}.nc'))
    
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
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get biodiversity scores for selected species
    bio_scores_sr = xr.DataArray(
        data.get_GBF8_bio_layers_by_yr(yr_cal, level='group') * data.REAL_AREA[None,:],
        dims=['group','cell'],
        coords={
            'group': data.BIO_GBF8_GROUPS_NAMES,
            'cell': np.arange(data.NCELLS)}
    ).chunk({'cell': min(1024, data.NCELLS), 'group': 1})  # Chunking to save mem use

    # Get the habitat contribution for ag/non-ag/am land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
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

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_gbf8_groups_ag = xr.concat([xr_gbf8_groups_ag.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf8_groups_ag], dim='lm')
    xr_gbf8_groups_am = xr.concat([xr_gbf8_groups_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf8_groups_am], dim='lm')
    xr_gbf8_groups_am = xr.concat([xr_gbf8_groups_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_gbf8_groups_am], dim='lu')

    # Regional level aggregation
    GBF8_scores_groups_ag_region = xr_gbf8_groups_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)
        
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

    # Australia level aggregation
    GBF8_scores_groups_ag_AUS = xr_gbf8_groups_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal, region='AUSTRALIA')
        
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
            'am': 'Agri-Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF8_groups_scores_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf8_groups_ag_cat = xr.concat([ag_mosaic, xr_gbf8_groups_ag], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf8_groups_non_ag_cat = xr.concat([non_ag_mosaic, xr_gbf8_groups_non_ag], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_gbf8_groups_am_cat = xr.concat([am_mosaic, xr_gbf8_groups_am], dim='am')

    # Save xarray data to netCDF
    save2nc(xr_gbf8_groups_ag_cat, os.path.join(path, f'xr_biodiversity_GBF8_groups_ag_{yr_cal}.nc'))
    save2nc(xr_gbf8_groups_non_ag_cat, os.path.join(path, f'xr_biodiversity_GBF8_groups_non_ag_{yr_cal}.nc'))
    save2nc(xr_gbf8_groups_am_cat, os.path.join(path, f'xr_biodiversity_GBF8_groups_ag_management_{yr_cal}.nc'))
    
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
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]
        ).chunk({'cell': min(1024, data.NCELLS)}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    
    # Expand dimension
    ag_dvar_mrj = xr.concat([ag_dvar_mrj.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), ag_dvar_mrj], dim='lm')
    am_dvar_amrj = xr.concat([am_dvar_amrj.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), am_dvar_amrj], dim='lm')
    am_dvar_amrj = xr.concat([am_dvar_amrj.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), am_dvar_amrj], dim='lu')

    # Get biodiversity scores for selected species
    bio_scores_sr = xr.DataArray(
        data.get_GBF8_bio_layers_by_yr(yr_cal, level='species') * data.REAL_AREA[None, :],
        dims=['species', 'cell'],
        coords={
            'species': data.BIO_GBF8_SEL_SPECIES,
            'cell': np.arange(data.NCELLS)}
    ).chunk({'cell': min(1024, data.NCELLS), 'species': 1})  # Chunking to save mem use

    # Get the habitat contribution for ag/non-ag/am land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    ).unstack()

    # Expand lu dimension
    am_impact_amr = xr.concat([am_impact_amr.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), am_impact_amr], dim='lu')

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

    # Expand dimension (has to be after multiplication to avoid double counting)
    xr_gbf8_species_ag = xr.concat([xr_gbf8_species_ag.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf8_species_ag], dim='lm')
    xr_gbf8_species_am = xr.concat([xr_gbf8_species_am.sum(dim='lm', keepdims=True).assign_coords(lm=['ALL']), xr_gbf8_species_am], dim='lm')
    xr_gbf8_species_am = xr.concat([xr_gbf8_species_am.sum(dim='lu', keepdims=True).assign_coords(lu=['ALL']), xr_gbf8_species_am], dim='lu')

    # Regional level aggregation
    GBF8_scores_species_ag_region = xr_gbf8_species_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)

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

    # Australia level aggregation
    GBF8_scores_species_ag_AUS = xr_gbf8_species_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal, region='AUSTRALIA')

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
            'am': 'Agri-Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF8_species_scores_{yr_cal}.csv'), index=False)

    # Append the 'ALL' dimension from dvar file
    ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf8_species_ag_cat = xr.concat([ag_mosaic, xr_gbf8_species_ag], dim='lu')
    non_ag_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))['data'].sel(lu=['ALL'])
    xr_gbf8_species_non_ag_cat = xr.concat([non_ag_mosaic, xr_gbf8_species_non_ag], dim='lu')
    am_mosaic = xr.open_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))['data'].sel(am=['ALL'])
    xr_gbf8_species_am_cat = xr.concat([am_mosaic, xr_gbf8_species_am], dim='am')

    # Save xarray data to netCDF
    save2nc(xr_gbf8_species_ag_cat, os.path.join(path, f'xr_biodiversity_GBF8_species_ag_{yr_cal}.nc'))
    save2nc(xr_gbf8_species_non_ag_cat, os.path.join(path, f'xr_biodiversity_GBF8_species_non_ag_{yr_cal}.nc'))
    save2nc(xr_gbf8_species_am_cat, os.path.join(path, f'xr_biodiversity_GBF8_species_ag_management_{yr_cal}.nc'))
    
    return f"Biodiversity GBF8 species scores written for year {yr_cal}"






