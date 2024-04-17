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
Writes model output and statistics to files.
"""




import os, re
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

from luto import settings
from luto import tools
from luto.data import Data
from luto.tools.spatializers import *
from luto.tools.compmap import *

import luto.economics.agricultural.quantity as ag_quantity                      # ag_quantity has already been calculated and stored in <sim.prod_data>
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.transitions as ag_transitions
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.biodiversity as ag_biodiversity

import luto.economics.non_agricultural.quantity as non_ag_quantity              # non_ag_quantity is all zeros, so skip the calculation here
import luto.economics.non_agricultural.revenue as non_ag_revenue
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.transitions as non_ag_transitions
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.non_agricultural.biodiversity as non_ag_biodiversity

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES

from luto.tools.report.create_report_data import save_report_data
from luto.tools.report.create_html import data2html
from luto.tools.report.create_static_maps import TIF2MAP


timestamp_write = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

def write_outputs(data: Data):
    # Write the model outputs to file
    write_data(data)
    # Move the log files to the output directory
    write_logs(data)
    

@tools.LogToFile(f"{settings.OUTPUT_DIR}/write_{timestamp_write}")
def write_data(data: Data):

    # Write model run settings
    if not data.path:
        raise ValueError(
            "Cannot write outputs: 'path' attribute of Data object has not been set "
            "(has the simulation been run?)"
        )
    write_settings(data.path)

    # Get the years to write
    years = sorted(list(data.lumaps.keys()))
    paths = [f"{data.path}/out_{yr}" for yr in years]

    ###############################################################
    #                     Create raw outputs                      #
    ###############################################################

    # Write the area transition between base-year and target-year 
    write_area_transition_start_end(data,f'{data.path}/out_{years[-1]}')
    write_ghg_offland_commodity(data, f'{data.path}/out_{years[-1]}')

    # Write outputs for each year
    jobs = [delayed(write_output_single_year)(data, yr, path_yr, None) for (yr, path_yr) in zip(years, paths)]

    # Write the area/quantity comparison between base-year and target-year for the timeseries mode
    begin_end_path = f"{data.path}/begin_end_compare_{years[0]}_{years[-1]}"
    jobs += [delayed(write_output_single_year)(data, years[-1], f"{begin_end_path}/out_{years[-1]}", years[0])] if settings.MODE == 'timeseries' else None

    # Parallel write the outputs for each year
    num_jobs = min(len(jobs), settings.WRITE_THREADS) if settings.PARALLEL_WRITE else 1   # Use the minimum between jobs_num and threads for parallel writing
    Parallel(n_jobs=num_jobs, prefer='threads')(jobs)
    
    # Copy the base-year outputs to the path_begin_end_compare
    shutil.copytree(f"{data.path}/out_{years[0]}", f"{begin_end_path}/out_{years[0]}", dirs_exist_ok = True) if settings.MODE == 'timeseries' else None

    # Create the report HTML and png maps
    TIF2MAP(data.path) if settings.WRITE_OUTPUT_GEOTIFFS else None
    save_report_data(data.path)
    data2html(data.path)
    
    



def write_logs(data: Data):
    # Move the log files to the output directory
    logs = [f"{settings.OUTPUT_DIR}/run_{data.timestamp_sim}_stdout.log",
            f"{settings.OUTPUT_DIR}/run_{data.timestamp_sim}_stderr.log",
            f"{settings.OUTPUT_DIR}/write_{timestamp_write}_stdout.log",
            f"{settings.OUTPUT_DIR}/write_{timestamp_write}_stderr.log"]
    
    [shutil.move(log, f"{data.path}/{os.path.basename(log)}") for log in logs if os.path.exists(log)]



def write_output_single_year(data: Data, yr_cal, path_yr, yr_cal_sim_pre=None):
    """Write outputs for simulation 'sim', calendar year, demands d_c, and path"""
    
    years = sorted(list(data.lumaps.keys()))
    
    if not os.path.isdir(path_yr):
        os.mkdir(path_yr)

    # # Write the decision variables, land-use and land management maps
    if settings.WRITE_OUTPUT_GEOTIFFS:
        write_files(data, yr_cal, path_yr)
        write_files_separate(data, yr_cal, path_yr)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CAUTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # The area here was calculated from lumap/lmmap, which {maby not accurate !!!} 
    # compared to the area calculated from dvars
    write_crosstab(data, yr_cal, path_yr, yr_cal_sim_pre)
    

    # Write the reset outputs
    write_dvar_area(data, yr_cal, path_yr)
    write_quantity(data, yr_cal, path_yr, yr_cal_sim_pre)
    write_revenue_cost_ag(data, yr_cal, path_yr)
    write_revenue_cost_ag_management(data, yr_cal, path_yr)
    write_revenue_cost_non_ag(data, yr_cal, path_yr)
    write_cost_transition(data, yr_cal, path_yr)
    write_water(data, yr_cal, path_yr)
    write_ghg(data, yr_cal, path_yr)
    write_ghg_separate(data, yr_cal, path_yr)
    write_biodiversity(data, yr_cal, path_yr)
    write_biodiversity_separate(data, yr_cal, path_yr)
    
    print(f"Finished writing {yr_cal} out of {years[0]}-{years[-1]} years\n")
    
       

def write_settings(path):
    # sourcery skip: extract-method, swap-nested-ifs, use-named-expression
    """Write model run settings"""

    # Open the settings.py file
    with open('luto/settings.py', 'r') as file:
        lines = file.readlines()

        # Regex patterns that matches variable assignments from settings
        parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=")
        settings_order = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]

        # Reorder the settings dictionary to match the order in the settings.py file
        settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
        settings_dict = {i: settings_dict[i] for i in settings_order if i in settings_dict}

        # Set unused variables to None
        settings_dict['GHG_LIMITS_FIELD'] = 'None' if settings.GHG_LIMITS_TYPE == 'dict' else None
        settings_dict['GHG_LIMITS'] = 'None' if settings.GHG_LIMITS_TYPE == 'file' else None

        settings_dict['LAND_USAGE_CULL_PERCENTAGE'] = 'None' if settings.CULL_MODE in ['absolute', 'none'] else None
        settings_dict['MAX_LAND_USES_PER_CELL'] = 'None' if settings.CULL_MODE in ['percentage', 'none'] else None

        settings_dict['WATER_STRESS_FRACTION'] = 'None' if settings.WATER_USE_LIMITS == 'on' and settings.WATER_LIMITS_TYPE == 'pct_ag' else None
        settings_dict['WATER_USE_REDUCTION_PERCENTAGE'] = 'None' if settings.WATER_USE_LIMITS == 'on' and settings.WATER_LIMITS_TYPE == 'water_stress' else None

    # Write the settings to a file
    with open(os.path.join(path, 'model_run_settings.txt'), 'w') as f:
        f.writelines(f'{k}:{v}\n' for k, v in settings_dict.items())



def write_files(data: Data, yr_cal, path):
    """Writes numpy arrays and geotiffs to file"""
    
    print('Writing numpy arrays and geotiff outputs')
    
    # Save raw agricultural decision variables (float array).
    ag_X_mrj_fname = f'ag_X_mrj_{yr_cal}.npy'
    np.save(os.path.join(path, ag_X_mrj_fname), data.ag_dvars[yr_cal].astype(np.float16))
    
    # Save raw non-agricultural decision variables (float array).
    non_ag_X_rk_fname = f'non_ag_X_rk_{yr_cal}.npy'
    np.save(os.path.join(path, non_ag_X_rk_fname), data.non_ag_dvars[yr_cal].astype(np.float16))

    # Save raw agricultural management decision variables (float array).
    for am in AG_MANAGEMENTS_TO_LAND_USES:
        snake_case_am = tools.am_name_snake_case(am)
        am_X_mrj_fname = f'ag_man_X_mrj_{snake_case_am}_{yr_cal}.npy'
        np.save(os.path.join(path, am_X_mrj_fname), data.ag_man_dvars[yr_cal][am].astype(np.float16))
    
    # Write out raw numpy arrays for land-use and land management
    lumap_fname = f'lumap_{yr_cal}.npy'
    lmmap_fname = f'lmmap_{yr_cal}.npy'
    
    np.save(os.path.join(path, lumap_fname), data.lumaps[yr_cal])
    np.save(os.path.join(path, lmmap_fname), data.lmmaps[yr_cal])
    
    
    # Get the Agricultural Management applied to each pixel
    ag_man_dvar = np.stack([np.einsum('mrj -> r', v) for _,v in data.ag_man_dvars[yr_cal].items()]).T   # (r, am)
    ag_man_dvar_mask = ag_man_dvar.sum(1) > 0.01            # Meaning that they have at least 1% of agricultural management applied
    ag_man_dvar = np.argmax(ag_man_dvar, axis=1) + 1        # Start from 1
    ag_man_dvar_argmax = np.where(ag_man_dvar_mask, ag_man_dvar, 0)


    # Get the non-agricultural landuse for each pixel
    non_ag_dvar = data.non_ag_dvars[yr_cal]                 # (r, k)
    non_ag_dvar_mask = non_ag_dvar.sum(1) > 0.01            # Meaning that they have at least 1% of non-agricultural landuse applied
    non_ag_dvar = np.argmax(non_ag_dvar, axis=1) + settings.NON_AGRICULTURAL_LU_BASE_CODE    # Start from 100
    non_ag_dvar_argmax = np.where(non_ag_dvar_mask, non_ag_dvar, 0)

    # Put the excluded land-use and land management types back in the array.
    lumap = create_2d_map(data, data.lumaps[yr_cal], filler=data.MASK_LU_CODE)
    lmmap = create_2d_map(data, data.lmmaps[yr_cal], filler=data.MASK_LU_CODE)
    ammap = create_2d_map(data, ag_man_dvar_argmax, filler=data.MASK_LU_CODE)
    non_ag = create_2d_map(data, non_ag_dvar_argmax, filler=data.MASK_LU_CODE)

    lumap_fname = f'lumap_{yr_cal}.tiff'
    lmmap_fname = f'lmmap_{yr_cal}.tiff'
    ammap_fname = f'ammap_{yr_cal}.tiff'
    non_ag_fname = f'non_ag_{yr_cal}.tiff'

    write_gtiff(lumap, os.path.join(path, lumap_fname))
    write_gtiff(lmmap, os.path.join(path, lmmap_fname))
    write_gtiff(ammap, os.path.join(path, ammap_fname))
    write_gtiff(non_ag, os.path.join(path, non_ag_fname))



def write_files_separate(data: Data, yr_cal, path, ammap_separate=False):

    '''Write raw decision variables to separate GeoTiffs'''
 
    # Collapse the land management dimension (m -> [dry, irr])
    ag_dvar_rj = np.einsum('mrj -> rj', data.ag_dvars[yr_cal])   # To compute the landuse map
    ag_dvar_rm = np.einsum('mrj -> rm', data.ag_dvars[yr_cal])   # To compute the land management (dry/irr) map
    non_ag_rk = np.einsum('rk -> rk', data.non_ag_dvars[yr_cal]) # Do nothing, just for code consistency
    ag_man_rj_dict = {am: np.einsum('mrj -> rj', ammap) for am, ammap in data.ag_man_dvars[yr_cal].items()}

    # Get the desc2dvar table. 
    ag_dvar_map = tools.map_desc_to_dvar_index('Ag_LU', data.DESC2AGLU, ag_dvar_rj)
    non_ag_dvar_map = tools.map_desc_to_dvar_index('Non-Ag_LU', {v:k for k,v in enumerate(data.NON_AGRICULTURAL_LANDUSES)}, non_ag_rk)
    lm_dvar_map = tools.map_desc_to_dvar_index('Land_Mgt', {v:k for k,v in enumerate(data.LANDMANS)},ag_dvar_rm)
    
    # Get the desc2dvar table for agricultural management
    ag_man_maps = [
        tools.map_desc_to_dvar_index(am, {desc: data.DESC2AGLU[desc] for desc in AG_MANAGEMENTS_TO_LAND_USES[am]}, am_dvar.sum(1)[:, np.newaxis])
        if ammap_separate else
        tools.map_desc_to_dvar_index('Ag_Mgt', {am: 0}, am_dvar.sum(1)[:, np.newaxis])
        for am, am_dvar in ag_man_rj_dict.items()
    ]
    ag_man_map = pd.concat(ag_man_maps)
    
    # Combine the desc2dvar table for agricultural land-use, agricultural management, and non-agricultural land-use
    desc2dvar_df = pd.concat([ag_dvar_map, ag_man_map, non_ag_dvar_map, lm_dvar_map])

    # Export to GeoTiff
    lucc_separate_dir = os.path.join(path, 'lucc_separate')
    os.makedirs(lucc_separate_dir, exist_ok=True)
    for _, row in desc2dvar_df.iterrows():
        category = row['Category']
        dvar_idx = row['dvar_idx']
        desc = row['lu_desc']
        dvar = create_2d_map(data, row['dvar'], filler=data.MASK_LU_CODE)
        fname = f'{category}_{dvar_idx:02}_{desc}_{yr_cal}.tiff'
        lucc_separate_path = os.path.join(lucc_separate_dir, fname)
        write_gtiff(dvar, lucc_separate_path)



def write_quantity(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE
    yr_idx_sim = sorted(list(data.lumaps.keys())).index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

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
            'Commodity': data.COMMODITIES,
            'Prod_base_year (tonnes, KL)': prod_base,
            'Prod_targ_year (tonnes, KL)': prod_targ,
            'Demand (tonnes, KL)': demands,
            'Abs_diff (tonnes, KL)': abs_diff,
            'Prop_diff (%)': prop_diff
        })

        # Save files to disk
        df.to_csv(os.path.join(path, f'quantity_comparison_{yr_cal}.csv'), index=False)

        # Write the production of each year to disk
        production_years = pd.DataFrame({yr_cal: data.prod_data[yr_cal]['Production']})
        production_years.insert(0, 'Commodity', data.COMMODITIES)
        production_years.to_csv(os.path.join(path, f'quantity_production_kt_{yr_cal}.csv'), index=False)

    # --------------------------------------------------------------------------------------------
    # NOTE:Non-agricultural production are all zeros, therefore skip the calculation
    # --------------------------------------------------------------------------------------------
        


def write_revenue_cost_ag(data: Data, yr_cal, path):
    """Calculate agricultural revenue. Takes a simulation object, a target calendar 
       year (e.g., 2030), and an output path as input."""

    yr_idx = yr_cal - data.YR_CAL_BASE
    ag_dvar_mrj = data.ag_dvars[yr_cal]
    print('Writing agricultural revenue_cost outputs')

    # Get agricultural revenue/cost for year in mrjs format
    ag_rev_df_rjms = ag_revenue.get_rev_matrices(data, yr_idx, aggregate=False)
    ag_cost_df_rjms = ag_cost.get_cost_matrices(data, yr_idx, aggregate=False)

    # Expand the original df with zero values to convert it to a **mrjs** array
    ag_rev_rjms = ag_rev_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_rev_df_rjms.columns.levels), fill_value=0).values.reshape(-1, *ag_rev_df_rjms.columns.levshape)
    ag_cost_rjms = ag_cost_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_cost_df_rjms.columns.levels), fill_value=0).values.reshape(-1, *ag_cost_df_rjms.columns.levshape)

    # Multiply the ag_dvar_mrj with the ag_rev_mrj to get the ag_rev_jm
    ag_rev_jms = np.einsum('mrj,rjms -> jms', ag_dvar_mrj, ag_rev_rjms)
    ag_cost_jms = np.einsum('mrj,rjms -> jms', ag_dvar_mrj, ag_cost_rjms)

    # Put the ag_rev_jms into a dataframe
    df_rev = pd.DataFrame(ag_rev_jms.reshape(ag_rev_jms.shape[0],-1), 
                          columns=pd.MultiIndex.from_product(ag_rev_df_rjms.columns.levels[1:]),
                          index=ag_rev_df_rjms.columns.levels[0])
    
    df_cost = pd.DataFrame(ag_cost_jms.reshape(ag_cost_jms.shape[0],-1),    
                           columns=pd.MultiIndex.from_product(ag_cost_df_rjms.columns.levels[1:]),
                           index=ag_cost_df_rjms.columns.levels[0])
    
    # Reformat the revenue/cost matrix into a long dataframe
    df_rev = df_rev.melt(ignore_index=False).reset_index()
    df_rev.columns = ['Land-use', 'Water_supply', 'Type', 'Value ($)']
    df_cost = df_cost.melt(ignore_index=False).reset_index()
    df_cost.columns = ['Land-use', 'Water_supply', 'Type', 'Value ($)']

    # Save to file
    df_rev.to_csv(os.path.join(path, f'revenue_agricultural_commodity_{yr_cal}.csv'))
    df_cost.to_csv(os.path.join(path, f'cost_agricultural_commodity_{yr_cal}.csv'))
    

def write_revenue_cost_ag_management(data: Data, yr_cal, path):
    """Calculate agricultural management revenue and cost."""
    
    print('Writing agricultural management revenue_cost outputs')

    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the revenue/cost matirces for each agricultural land-use
    ag_rev_mrj = ag_revenue.get_rev_matrices(data, yr_idx)
    ag_cost_mrj = ag_cost.get_cost_matrices(data, yr_idx)

    # Get the revenuecost matrices for each agricultural management
    am_revenue_mat = ag_revenue.get_agricultural_management_revenue_matrices(data, ag_rev_mrj, yr_idx)
    am_cost_mat = ag_cost.get_agricultural_management_cost_matrices(data, ag_cost_mrj, yr_idx)

    revenue_am_dfs = []
    cost_am_dfs = []
    # Loop through the agricultural managements
    for am, am_desc in AG_MANAGEMENTS_TO_LAND_USES.items():
        # Get the land use codes for the agricultural management
        am_code = [data.DESC2AGLU[desc] for desc in am_desc]

        # Get the revenue/cost matrix for the agricultural management
        am_rev = np.nan_to_num(am_revenue_mat[am])  # Replace NaNs with 0
        am_cost = np.nan_to_num(am_cost_mat[am])  # Replace NaNs with 0

        # Get the decision variable for each agricultural management
        am_dvar = data.ag_man_dvars[yr_cal][am][:, :, am_code]

        # Multiply the decision variable by revenue matrix
        am_rev_yr = np.einsum('mrj,mrj->jm', am_dvar, am_rev)
        am_cost_yr = np.einsum('mrj,mrj->jm', am_dvar, am_cost)

        # Reformat the revenue/cost matrix into a dataframe
        am_rev_yr_df = pd.DataFrame(am_rev_yr, columns=data.LANDMANS)
        am_rev_yr_df['Land-use'] = am_desc
        am_rev_yr_df = am_rev_yr_df.melt(id_vars='Land-use', value_vars=data.LANDMANS, var_name='Water',
                                         value_name='Value (AU$)')
        am_rev_yr_df['year'] = yr_cal
        am_rev_yr_df['Management Type'] = am

        am_cost_yr_df = pd.DataFrame(am_cost_yr, columns=data.LANDMANS)
        am_cost_yr_df['Land-use'] = am_desc
        am_cost_yr_df = am_cost_yr_df.melt(id_vars='Land-use', value_vars=data.LANDMANS, var_name='Water',
                                          value_name='Value (AU$)')
        am_cost_yr_df['year'] = yr_cal
        am_cost_yr_df['Management Type'] = am

        # Store the revenue/cost dataframes
        revenue_am_dfs.append(am_rev_yr_df)
        cost_am_dfs.append(am_cost_yr_df)

    # Concatenate the revenue/cost dataframes
    revenue_am_df = pd.concat(revenue_am_dfs)
    cost_am_df = pd.concat(cost_am_dfs)

    revenue_am_df.to_csv(os.path.join(path, f'revenue_agricultural_management_{yr_cal}.csv'), index=False)
    cost_am_df.to_csv(os.path.join(path, f'cost_agricultural_management_{yr_cal}.csv'), index=False)
    
    
    
def write_cost_transition(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Calculate transition cost."""
    
    print('Writing transition cost outputs')
        
    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    # Get index of yr_cal in timeseries (e.g., if yr_cal is 2050 then yr_idx = 40)
    yr_idx = yr_cal - data.YR_CAL_BASE
    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre


    # Get the decision variables for agricultural land-use
    ag_dvar = data.ag_dvars[yr_cal]                          # (m,r,j)
    # Get the non-agricultural decision variable
    non_ag_dvar = data.non_ag_dvars[yr_cal]                  # (r,k)
    

    #---------------------------------------------------------------------
    #              Agricultural land-use transition costs
    #---------------------------------------------------------------------
    
    # Get the transition cost matrices for agricultural land-use
    if yr_idx == 0:
        base_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
        ag_transitions_cost_mat = {k: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))  
                                for k in ['Establishment cost', 'Water license cost', 'Carborn releasing cost']}
    else:
        # Get the base_year mrj matirx 
        base_mrj = tools.lumap2ag_l_mrj(data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre])
        # Get the transition cost matrices for agricultural land-use
        ag_transitions_cost_mat = ag_transitions.get_transition_matrices(data, 
                                                                        yr_idx, 
                                                                        yr_cal_sim_pre,
                                                                        data.lumaps,
                                                                        data.lmmaps,
                                                                        separate = True)

    cost_dfs = []
    # Convert the transition cost matrices to a DataFrame
    for lu_desc, lu_idx in data.DESC2AGLU.items():
        for cost_type in ag_transitions_cost_mat.keys():
            
            base_lu_arr = base_mrj[:, :, lu_idx]                                      # Get the base land-use array                       (m,r)
            arr = np.nan_to_num(ag_transitions_cost_mat[cost_type])                   # Get the transition cost matrix                    (m,r,j)
            arr = np.einsum('mr,mrj,mrj->mj', base_lu_arr, arr, ag_dvar)              # Multiply by decision variables
            
            arr_df = pd.DataFrame(arr.flatten(), 
                            index=pd.MultiIndex.from_product([data.LANDMANS, data.AGRICULTURAL_LANDUSES], 
                            names=['Water Supply', 'To land-use']), 
                            columns=['Cost ($)']).reset_index()
            arr_df.insert(0, 'Type', cost_type)
            arr_df.insert(1, 'year', yr_cal)
            arr_df.insert(2, 'From land-use', lu_desc)
            cost_dfs.append(arr_df)

    # Save the cost DataFrames 
    cost_df = pd.concat(cost_dfs, axis=0)
    cost_df.to_csv(os.path.join(path, f'cost_transition_ag2ag_{yr_cal}.csv'), index=False)




    #---------------------------------------------------------------------
    #              Agricultural management transition costs
    #---------------------------------------------------------------------

    # The agricultural management transition cost are all zeros, so skip the calculation here
    # am_cost = ag_transitions.get_agricultural_management_transition_matrices(sim.data) 




    #--------------------------------------------------------------------
    #              Non-agricultural land-use transition costs (from ag to non-ag)
    #--------------------------------------------------------------------

    # Get the transition cost matirces for non-agricultural land-use
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {k:{'Transition cost':np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))}
                                   for k in data.NON_AGRICULTURAL_LANDUSES}
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_from_ag_transition_matrix(data,
                                                                                       yr_idx,
                                                                                       data.lumaps[yr_cal],
                                                                                       data.lmmaps[yr_cal],
                                                                                       separate=True)

    cost_dfs = []
    for idx,non_ag_type in enumerate(non_ag_transitions_cost_mat):
        for cost_type in non_ag_transitions_cost_mat[non_ag_type]:

            arr = non_ag_transitions_cost_mat[non_ag_type][cost_type]          # Get the transition cost matrix 
            arr = np.einsum('mrj,r->mj', arr, non_ag_dvar[:,idx])              # Multiply the transition cost matrix by the cost of non-agricultural land-use


            arr_df = pd.DataFrame(arr.flatten(), 
                                index=pd.MultiIndex.from_product([data.LANDMANS, data.AGRICULTURAL_LANDUSES],names=['Water supply', 'From land-use']), 
                                columns=['Cost ($)']).reset_index()
            arr_df.insert(0, 'To land-use', non_ag_type)
            arr_df.insert(1, 'Cost type', cost_type)
            arr_df.insert(2, 'year', yr_cal)
            cost_dfs.append(arr_df)

    # Save the cost DataFrames
    cost_df = pd.concat(cost_dfs, axis=0)
    cost_df.to_csv(os.path.join(path, f'cost_transition_ag2non_ag_{yr_cal}.csv'), index=False)




    #--------------------------------------------------------------------
    #              Non-agricultural land-use transition costs (from non-ag to ag)
    #--------------------------------------------------------------------

    # Get the transition cost matirces for non-agricultural land-use
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {k:{'Transition cost':np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))}
                                        for k in data.NON_AGRICULTURAL_LANDUSES}
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_to_ag_transition_matrix(data,
                                                                                    yr_idx,
                                                                                    data.lumaps[yr_cal],
                                                                                    data.lmmaps[yr_cal],
                                                                                    separate=True)

    cost_dfs = []
    for non_ag_type in non_ag_transitions_cost_mat:
        for cost_type in non_ag_transitions_cost_mat[non_ag_type]:

            arr = non_ag_transitions_cost_mat[non_ag_type][cost_type]          # Get the transition cost matrix 
            arr = np.einsum('mrj,mrj->mj', arr, ag_dvar)                       # Multiply the transition cost matrix by the cost of non-agricultural land-use


            arr_df = pd.DataFrame(arr.flatten(), 
                                index=pd.MultiIndex.from_product([data.LANDMANS, data.AGRICULTURAL_LANDUSES],names=['Water supply', 'To land-use']), 
                                columns=['Cost ($)']).reset_index()
            arr_df.insert(0, 'From land-use', non_ag_type)
            arr_df.insert(1, 'Cost type', cost_type)
            arr_df.insert(2, 'year', yr_cal)
            cost_dfs.append(arr_df)

    # Save the cost DataFrames
    cost_df = pd.concat(cost_dfs, axis=0)
    cost_df.to_csv(os.path.join(path, f'cost_transition_non_ag2_ag_{yr_cal}.csv'), index=False)


def write_revenue_cost_non_ag(data: Data, yr_cal, path):
    """Calculate non_agricultural cost. """

    print('Writing non agricultural management cost outputs')
    non_ag_dvar = data.non_ag_dvars[yr_cal]

    # Get the non-agricultural revenue/cost matrices
    non_ag_rev_mat = non_ag_revenue.get_rev_matrix(data)    # rk
    non_ag_cost_mat = non_ag_cost.get_cost_matrix(data)     # rk

    # Replace nan with 0
    non_ag_rev_mat = np.nan_to_num(non_ag_rev_mat)
    non_ag_cost_mat = np.nan_to_num(non_ag_cost_mat)

    # Calculate the non-agricultural revenue and cost
    rev_non_ag = np.einsum('rk,rk->k', non_ag_dvar, non_ag_rev_mat)
    cost_non_ag = np.einsum('rk,rk->k', non_ag_dvar, non_ag_cost_mat)

    # Reformat the revenue/cost matrix into a dataframe
    rev_non_ag_df = pd.DataFrame(rev_non_ag.reshape(-1,1), columns=['Value AU$'])
    rev_non_ag_df['year'] = yr_cal
    rev_non_ag_df['Land-use'] = data.NON_AGRICULTURAL_LANDUSES

    cost_non_ag_df = pd.DataFrame(cost_non_ag.reshape(-1,1), columns=['Value AU$'])
    cost_non_ag_df['year'] = yr_cal
    cost_non_ag_df['Land-use'] = data.NON_AGRICULTURAL_LANDUSES

    # Save to disk
    rev_non_ag_df.to_csv(os.path.join(path, f'revenue_non_ag_{yr_cal}.csv'), index = False)
    cost_non_ag_df.to_csv(os.path.join(path, f'cost_non_ag_{yr_cal}.csv'), index = False)
    
    
    

def write_dvar_area(data: Data, yr_cal, path):
        
    # Reprot the process
    print('Writing area calculated from dvars to {path}')
    
    # Get the decision variables for the year, multiply them by the area of each pixel, 
    # and sum over the landuse dimension (j/k)
    ag_area = np.einsum('mrj,r -> mj', data.ag_dvars[yr_cal], data.REAL_AREA)  
    non_ag_area = np.einsum('rk,r -> k', data.non_ag_dvars[yr_cal], data.REAL_AREA) 
    ag_man_area_dict = {am: np.einsum('mrj,r -> mj', ammap, data.REAL_AREA) 
                        for am, ammap in data.ag_man_dvars[yr_cal].items()}

    # Agricultural landuse
    df_ag_area = pd.DataFrame(ag_area.reshape(-1), 
                                index=pd.MultiIndex.from_product([[yr_cal],
                                                                data.LANDMANS,
                                                                data.AGRICULTURAL_LANDUSES],
                                                                names=['Year', 'Water','Land use']),
                                columns=['Area (ha)']).reset_index()
    # Non-agricultural landuse
    df_non_ag_area = pd.DataFrame(non_ag_area.reshape(-1),
                                index=pd.MultiIndex.from_product([[yr_cal],
                                                                ['dry'],
                                                                data.NON_AGRICULTURAL_LANDUSES],
                                                                names=['Year', 'Water', 'Land use']),
                                columns=['Area (ha)']).reset_index()

    # Agricultural management
    am_areas = []
    for am, am_arr in ag_man_area_dict.items():
        df_am_area = pd.DataFrame(am_arr.reshape(-1),
                                index=pd.MultiIndex.from_product([[yr_cal],
                                                                [am],
                                                                data.LANDMANS,
                                                                data.AGRICULTURAL_LANDUSES],
                                                                names=['Year', 'Type', 'Water','Land use']),
                                columns=['Area (ha)']).reset_index()
        am_areas.append(df_am_area)
    
    # Concatenate the dataframes
    df_am_area = pd.concat(am_areas)

    # Save to file
    df_ag_area.to_csv(os.path.join(path, f'area_agricultural_landuse_{yr_cal}.csv'), index = False)
    df_non_ag_area.to_csv(os.path.join(path, f'area_non_agricultural_landuse_{yr_cal}.csv'), index = False)
    df_am_area.to_csv(os.path.join(path, f'area_agricultural_management_{yr_cal}.csv'), index = False)


def write_area_transition_start_end(data: Data, path):
        
    print(f'Save transition matrix for start year to end year to {path}\n')
    
    # Get all years from sim
    years = sorted(data.ag_dvars.keys())

    # Get the end year
    yr_cal_end = years[-1]

    # Get the decision variables for the start year
    dvar_base = data.ag_dvars[data.YR_CAL_BASE]

    # Calculate the transition matrix for agricultural land uses (start) to agricultural land uses (end)
    transitions_ag2ag = []
    for lu_idx, lu in enumerate(data.AGRICULTURAL_LANDUSES):
        dvar_target = data.ag_dvars[yr_cal_end][:,:,lu_idx]
        trans = np.einsum('mrj, mr, r -> j', dvar_base, dvar_target, data.REAL_AREA)
        trans_df = pd.DataFrame({lu:trans.flatten()}, index=data.AGRICULTURAL_LANDUSES)
        transitions_ag2ag.append(trans_df)
    transition_ag2ag = pd.concat(transitions_ag2ag, axis=1)

    # Calculate the transition matrix for agricultural land uses (start) to non-agricultural land uses (end)
    trainsitions_ag2non_ag = []
    for lu_idx, lu in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        dvar_target = data.non_ag_dvars[yr_cal_end][:,lu_idx]
        trans = np.einsum('mrj, r, r -> j', dvar_base, dvar_target, data.REAL_AREA)
        trans_df = pd.DataFrame({lu:trans.flatten()}, index=data.AGRICULTURAL_LANDUSES)
        trainsitions_ag2non_ag.append(trans_df)
    transition_ag2non_ag = pd.concat(trainsitions_ag2non_ag, axis=1)

    # Concatenate the two transition matrices
    transition = pd.concat([transition_ag2ag, transition_ag2non_ag], axis=1)
    transition = transition.stack().reset_index()
    transition.columns = ['From land-use','To land-use','Area (ha)']
    
    # Write the transition matrix to a csv file
    transition.to_csv(os.path.join(path, f'transition_matrix_{data.YR_CAL_BASE}_{yr_cal_end}.csv'), index=False)



def write_crosstab(data: Data, yr_cal, path, yr_cal_sim_pre=None): 
    """Write out land-use and production data"""
        
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre
    

    # Only perform the calculation if the yr_cal is not the base year
    if yr_cal > data.YR_CAL_BASE:

        # Check if yr_cal_sim_pre meets the requirement
        assert yr_cal_sim_pre >= data.YR_CAL_BASE and yr_cal_sim_pre < yr_cal,\
            f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"

        print('Writing production outputs')

        # LUS = ['Non-agricultural land'] + data.AGRICULTURAL_LANDUSES + data.NON_AGRICULTURAL_LANDUSES
        ctlu, swlu = lumap_crossmap( data.lumaps[yr_cal_sim_pre]
                                   , data.lumaps[yr_cal]
                                   , data.AGRICULTURAL_LANDUSES
                                   , data.NON_AGRICULTURAL_LANDUSES
                                   , data.REAL_AREA)
        
        ctlm, swlm = lmmap_crossmap( data.lmmaps[yr_cal_sim_pre]
                                   , data.lmmaps[yr_cal]
                                   , data.REAL_AREA
                                   , data.LANDMANS)

        cthp, swhp = crossmap_irrstat( data.lumaps[yr_cal_sim_pre]
                                     , data.lmmaps[yr_cal_sim_pre]
                                     , data.lumaps[yr_cal], data.lmmaps[yr_cal]
                                     , data.AGRICULTURAL_LANDUSES
                                     , data.NON_AGRICULTURAL_LANDUSES
                                     , data.REAL_AREA)
        

        ctass = {}
        swass = {}
        for am in AG_MANAGEMENTS_TO_LAND_USES:        
            ctas, swas = crossmap_amstat( am
                                        , data.lumaps[yr_cal_sim_pre]
                                        , data.ammaps[yr_cal_sim_pre][am]
                                        , data.lumaps[yr_cal]
                                        , data.ammaps[yr_cal][am]
                                        , data.AGRICULTURAL_LANDUSES
                                        , data.NON_AGRICULTURAL_LANDUSES
                                        , data.REAL_AREA)
            ctass[am] = ctas
            swass[am] = swas
            
        ctlu.to_csv(os.path.join(path, f'crosstab-lumap_{yr_cal}.csv'))
        ctlm.to_csv(os.path.join(path, f'crosstab-lmmap_{yr_cal}.csv'))
        swlu.to_csv(os.path.join(path, f'switches-lumap_{yr_cal}.csv'))
        swlm.to_csv(os.path.join(path, f'switches-lmmap_{yr_cal}.csv'))
        cthp.to_csv(os.path.join(path, f'crosstab-irrstat_{yr_cal}.csv'))
        swhp.to_csv(os.path.join(path, f'switches-irrstat_{yr_cal}.csv'))
        
        for am in AG_MANAGEMENTS_TO_LAND_USES:
            am_snake_case = tools.am_name_snake_case(am).replace("_", "-")
            ctass[am].to_csv(os.path.join(path, f'crosstab-amstat-{am_snake_case}_{yr_cal}.csv'))
            swass[am].to_csv(os.path.join(path, f'switches-amstat-{am_snake_case}_{yr_cal}.csv'))



def write_water(data: Data, yr_cal, path):
    """Calculate water use totals. Takes a simulation object, a numeric
       target calendar year (e.g., 2030), and an output path as input."""

    print('Writing water outputs')

    # Convert calendar year to year index.
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get water use for year in mrj format
    ag_w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    non_ag_w_rk = non_ag_water.get_wreq_matrix(data)
    ag_man_w_mrj = ag_water.get_agricultural_management_water_matrices(data, ag_w_mrj, yr_idx)

    # Prepare a data frame.
    df = pd.DataFrame( columns=[ 'REGION_ID'
                                , 'REGION_NAME'
                                , 'WATER_USE_LIMIT_ML'
                                , 'TOT_WATER_REQ_ML'
                                , 'ABS_DIFF_ML'
                                , 'PROPORTION_%'  ] )

    # Get 2010 water use limits used as constraints in model
    wuse_limits = ag_water.get_wuse_limits(data)

    # Set up data for river regions or drainage divisions
    if settings.WATER_REGION_DEF == 'Drainage Division':
        region_limits = data.DRAINDIV_LIMITS
        region_id = data.DRAINDIV_ID
        # regions = settings.WATER_DRAINDIVS
        region_dict = data.DRAINDIV_DICT

    elif settings.WATER_REGION_DEF == 'River Region':
        region_limits = data.RIVREG_LIMITS
        region_id = data.RIVREG_ID
        # regions = settings.WATER_RIVREGS
        region_dict = data.RIVREG_DICT

    else:
        print('Incorrect option for WATER_REGION_DEF in settings')

    # Loop through specified water regions
    df_water_seperate_dfs = []
    for i, region in enumerate(region_limits.keys()):
        
        # Get indices of cells in region
        ind = np.flatnonzero(region_id == region).astype(np.int32)

        # Calculate water requirements by agriculture for year and region.

        index_levels = ['Landuse Type', 'Landuse', 'Irrigation',  'Water Use (ML)']

        # Agricultural water use
        ag_mrj = ag_w_mrj[:, ind, :] * data.ag_dvars[yr_cal][:, ind, :]   
        ag_jm = np.einsum('mrj->jm', ag_mrj)                             

        ag_df = pd.DataFrame(ag_jm.reshape(-1).tolist(),
                            index=pd.MultiIndex.from_product([['Agricultural Landuse'],
                                                            data.AGRICULTURAL_LANDUSES,
                                                            data.LANDMANS
                                                            ])).reset_index()

        ag_df.columns = index_levels


        # Non-agricultural water use
        non_ag_rk = non_ag_w_rk[ind, :] * data.non_ag_dvars[yr_cal][ind, :]  # Non-agricultural contribution
        non_ag_k = np.einsum('rk->k', non_ag_rk)                            # Sum over cells

        non_ag_df = pd.DataFrame(non_ag_k, 
                                index= pd.MultiIndex.from_product([['Non-agricultural Landuse'],
                                                                    data.NON_AGRICULTURAL_LANDUSES ,
                                                                    ['dry']  # non-agricultural land is always dry
                                                                    ])).reset_index()

        non_ag_df.columns = index_levels



        # Agricultural management water use
        AM_dfs = []
        for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():  # Agricultural managements contribution

            am_j = np.array([data.DESC2AGLU[lu] for lu in am_lus])

            # Water requirements for each agricultural management in array format
            am_mrj = ag_man_w_mrj[am][:, ind, :]\
                            * data.ag_man_dvars[yr_cal][am][:, ind[:,np.newaxis], am_j] 

            am_jm = np.einsum('mrj->jm', am_mrj)

            # Water requirements for each agricultural management in long dataframe format
            df_am = pd.DataFrame(am_jm.reshape(-1).tolist(),
                                index=pd.MultiIndex.from_product([['Agricultural Management'],
                                                                am_lus,
                                                                data.LANDMANS
                                                                ])).reset_index()
            df_am.columns = index_levels

            # Add to list of dataframes
            AM_dfs.append(df_am)

        # Combine all AM dataframes
        AM_df = pd.concat(AM_dfs)

        # Combine all dataframes
        df_region = pd.concat([ag_df, non_ag_df, AM_df])
        df_region.insert(0, 'region', region_dict[region])
        df_region.insert(0, 'year', yr_cal)

        # Add to list of dataframes
        df_water_seperate_dfs.append(df_region)


        # Calculate water use limits and actual water use
        wul = wuse_limits[i][2]
        wreq_reg = df_region['Water Use (ML)'].sum()

        # Calculate absolute and proportional difference between water use target and actual water use
        abs_diff = wreq_reg - wul
        prop_diff = (wreq_reg / wul) * 100 if wul > 0 else np.nan
        # Add to dataframe
        df.loc[i] = ( region
                    , region_dict[region]
                    , wul
                    , wreq_reg 
                    , abs_diff
                    , prop_diff )

    # Write to CSV with 2 DP
    df = df.drop(columns=['REGION_ID']).set_index('REGION_NAME').stack().reset_index()
    df.columns = ['REGION_NAME', 'Variable', 'Value (ML)']
    df.to_csv( os.path.join(path, f'water_demand_vs_use_{yr_cal}.csv'), index=False)

    # Write the separate water use to CSV
    df_water_seperate = pd.concat(df_water_seperate_dfs)
    df_water_seperate.to_csv( os.path.join(path, f'water_demand_vs_use_separate_{yr_cal}.csv'), index=False)
    
    
    

def write_ghg(data: Data, yr_cal, path):
    """Calculate total GHG emissions from on-land agricultural sector. 
        Takes a simulation object, a target calendar year (e.g., 2030), 
        and an output path as input."""

    print('Writing GHG outputs' )
    
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get GHG emissions limits used as constraints in model
    ghg_limits = ag_ghg.get_ghg_limits(data, yr_cal)

    # Get GHG emissions from model
    if yr_cal >= data.YR_CAL_BASE + 1:
        ghg_emissions = data.prod_data[yr_cal]['GHG Emissions']
    else:
        ghg_emissions = (ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True) * data.ag_dvars[data.YR_CAL_BASE]).sum()

    # Save GHG emissions to file
    df = pd.DataFrame({
        'Variable':['GHG_EMISSIONS_LIMIT_TCO2e','GHG_EMISSIONS_TCO2e'], 
        'Emissions (t CO2e)':[ghg_limits, ghg_emissions]
        })
    df.to_csv(os.path.join(path, f'GHG_emissions_{yr_cal}.csv'), index=False)





def write_biodiversity(data: Data, yr_cal, path):
    """
    Write biodiversity info for a given year ('yr_cal'), simulation ('sim')
    and output path ('path').
    """

    print('Writing biodiversity outputs')

    # Get limits used as constraints in model
    biodiv_limit = ag_biodiversity.get_biodiversity_limits(data, yr_cal)

    # Get GHG emissions from model
    if yr_cal >= data.YR_CAL_BASE + 1:
        biodiv_score = data.prod_data[yr_cal]['Biodiversity']
    else:
        # Limits are based on the 2010 biodiversity scores, with the base year
        # biodiversity score equal to the 
        biodiv_score = ag_biodiversity.get_base_year_biodiversity_score(data)

    # Add to dataframe
    df = pd.DataFrame({
            'Variable':['Biodiversity score limit',
                        'Solve biodiversity score'], 
            'Score':[biodiv_limit, biodiv_score]
            })
    
    # Save to file
    df.to_csv(os.path.join(path, f'biodiversity_{yr_cal}.csv'), index = False)
    
    
    
    
def write_biodiversity_separate(data: Data, yr_cal, path):
    
    print('Writing biodiversity_separate outputs')

    # Get the biodiversity scores b_mrj
    ag_biodiv_mrj = ag_biodiversity.get_breq_matrices(data)
    am_biodiv_mrj = ag_biodiversity.get_agricultural_management_biodiversity_matrices(data)
    non_ag_biodiv_rk = non_ag_biodiversity.get_breq_matrix(data)

    # Get the decision variables for the year
    ag_dvar_mrj = data.ag_dvars[yr_cal]
    ag_mam_dvar_mrj =  data.ag_man_dvars[yr_cal]
    non_ag_dvar_rk = data.non_ag_dvars[yr_cal]

    # Multiply the decision variables with the biodiversity scores
    ag_biodiv_jm = np.einsum('mrj,mrj -> jm', ag_biodiv_mrj, ag_dvar_mrj)
    non_ag_biodiv_k = np.einsum('rk,rk -> k', non_ag_biodiv_rk, non_ag_dvar_rk)

    # Get the biodiversity scores for agricultural landuse
    AG_df = pd.DataFrame(ag_biodiv_jm.reshape(-1),
                        index=pd.MultiIndex.from_product([['Agricultural Landuse'],
                                                        ['Agricultural Landuse'],
                                                        data.AGRICULTURAL_LANDUSES,
                                                        data.LANDMANS,
                                                        ])).reset_index()

    # Get the biodiversity scores for non-agricultural landuse
    NON_AG_df = pd.DataFrame(non_ag_biodiv_k.reshape(-1),
                            index=pd.MultiIndex.from_product([['Non-Agricultural Landuse'],
                                                            ['Non-Agricultural Landuse'],
                                                            data.NON_AGRICULTURAL_LANDUSES,
                                                            ['dry'],
                                                            ])).reset_index()

    # Get the biodiversity scores for agricultural management
    AM_dfs = []
    for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():  # Agricultural managements contribution
        
        # Slice the arrays with the agricultural management land uses
        am_j = np.array([data.DESC2AGLU[lu] for lu in am_lus])
        am_dvar = ag_mam_dvar_mrj[am][:,:,am_j]
        am_biodiv = am_biodiv_mrj[am]
        
        # Biodiversity score for each agricultural management in array format                    
        am_jm = np.einsum('mrj,mrj -> jm', am_dvar, am_biodiv)
        
        # Water requirements for each agricultural management in long dataframe format
        df_am = pd.DataFrame(am_jm.reshape(-1),
                            index=pd.MultiIndex.from_product([['Agricultural Management'],
                                                            [am],
                                                            am_lus,
                                                            data.LANDMANS
                                                            ])).reset_index()

        
        # Add to list of dataframes
        AM_dfs.append(df_am)
        
    # Combine all AM dataframes
    AM_df = pd.concat(AM_dfs)


    # Combine all dataframes
    biodiv_df = pd.concat([AG_df, NON_AG_df, AM_df])
    biodiv_df.columns = ['Landuse type','Landuse subtype', 'Landuse', 'Land management', 'Biodiversity score']
    biodiv_df.insert(0, 'Year', yr_cal)

    # Write to file
    biodiv_df.to_csv(os.path.join(path, f'biodiversity_separate_{yr_cal}.csv'), index=False)    
      
    
  
def write_ghg_separate(data: Data, yr_cal, path):

    print('Writing GHG emissions_Separate to {path}')

    # Convert calendar year to year index.
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the landuse descriptions for each validate cell (i.e., 0 -> Apples)
    lu_desc_map = {**data.AGLU2DESC,**data.NONAGLU2DESC}
    lu_desc = [lu_desc_map[x] for x in data.lumaps[yr_cal]]

    # -------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural landuse #
    # -------------------------------------------------------#

    # Get the ghg_df
    ag_g_df = ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=False)

    GHG_cols = []
    for col in ag_g_df.columns:
        # Get the index of each column
        s,m,j = [ag_g_df.columns.levels[i].get_loc(col[i]) for i in range(len(col))]
        # Get the GHG emissions
        ghg_col = np.nan_to_num(ag_g_df.loc[slice(None), col])
        # Get the dvar coresponding to the (m,j) dimension
        dvar = data.ag_dvars[yr_cal][m,:,j]
        # Multiply the GHG emissions by the dvar
        ghg_e = (ghg_col * dvar).sum()
        # Create a dataframe with the GHG emissions
        ghg_col = pd.DataFrame([ghg_e], index=pd.MultiIndex.from_tuples([col]))

        GHG_cols.append(ghg_col)
        
    # Concatenate the GHG emissions
    ghg_df = pd.concat(GHG_cols).reset_index()
    ghg_df.columns = ['Source','Water','Landuse','GHG Emissions (t)']

    # Pivot the dataframe
    ghg_df = ghg_df.pivot(index='Landuse', columns=['Water','Source'], values='GHG Emissions (t)')

    # Rename the columns
    ghg_df.columns = pd.MultiIndex.from_tuples([['Agricultural Landuse'] + list(col) for col in ghg_df.columns])
    column_rename = [(i[0],i[1],i[2].replace('CO2E_KG_HA','TCO2E')) for i in ghg_df.columns]
    column_rename = [(i[0],i[1],i[2].replace('CO2E_KG_HEAD','TCO2E')) for i in column_rename]
    ghg_df.columns = pd.MultiIndex.from_tuples(column_rename)
    
    
    # Add the sum of each column
    ghg_df.loc['SUM'] = ghg_df.sum(axis=0)
    ghg_df['SUM'] = ghg_df.sum(axis=1)
    ghg_df = ghg_df.fillna(0)                                       

    # Save table to disk
    ghg_df.to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_landuse_{yr_cal}.csv'))



    # -----------------------------------------------------------#
    # Get greenhouse gas emissions from non-agricultural landuse #
    # -----------------------------------------------------------#
    
    # Get the non_ag GHG reduction
    non_ag_g_rk = non_ag_ghg.get_ghg_matrix(data)
    
    # Multiply with decision variable to get the GHG in yr_cal
    non_ag_g_rk = non_ag_g_rk * data.non_ag_dvars[yr_cal]
    
    # get the non_ag GHG reduction on dry/irri land
    non_ag_g_rk_dry = np.einsum('rk,r -> rk', non_ag_g_rk, (data.LMMAP != 1))
    non_ag_g_rk_irri = np.einsum('rk,r -> rk', non_ag_g_rk, (data.LMMAP == 1))
    non_ag_g_mrk = np.array([non_ag_g_rk_dry,non_ag_g_rk_irri])        # mrk
    non_ag_g_rmk = np.swapaxes(non_ag_g_mrk,0,1)                       # rmk
    
    # Summarize the array as a df
    non_ag_g_rk_summary = tools.summarize_ghg_separate_df(
        non_ag_g_rmk,
        (['Non_Agricultural Landuse'], data.LANDMANS, [f'TCO2E_{non_ag}' for non_ag in data.NON_AGRICULTURAL_LANDUSES]),
        lu_desc
    )
    
    # Save table to disk
    non_ag_g_rk_summary.to_csv(os.path.join(path, f'GHG_emissions_separate_no_ag_reduction_{yr_cal}.csv'))
                        

    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from landuse transformation penalties #
    # -------------------------------------------------------------------#
    
    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    
    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    
    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    if yr_cal == data.YR_CAL_BASE:
        ghg_t = np.zeros(data.ag_dvars[yr_cal].shape, dtype=np.bool_)
    else:
        yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]
        ghg_t = ag_ghg.get_ghg_transition_penalties(data, data.lumaps[yr_cal_sim_pre])
    
    
    # Get the GHG emissions from lucc-convertion compared to the previous year
    ghg_t_separate = np.einsum('mrj,mrj -> rmj', data.ag_dvars[yr_cal], ghg_t)         

    # Summarize the array as a df
    ghg_t_separate_summary = tools.summarize_ghg_separate_df(ghg_t_separate,(['Deforestation'], 
                                                                       data.LANDMANS,
                                                                       [f"TCO2E_{i}" for i in data.AGRICULTURAL_LANDUSES]),
                                                             lu_desc)

    
    
    # Save table to disk
    ghg_t_separate_summary.to_csv(os.path.join(path, f'GHG_emissions_separate_transition_penalty_{yr_cal}.csv'))
    
    
    
    
    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural management          #
    # -------------------------------------------------------------------#
     
    ag_g_mrj = ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True)
    
    # Get the ag_man_g_mrj
    ag_man_g_mrj = ag_ghg.get_agricultural_management_ghg_matrices(data, ag_g_mrj, yr_idx)
    
    ag_ghg_arrays = []
    for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():
        
        # Get the lucc_code for this the agricultural management in this loop
        am_j = np.array([data.DESC2AGLU[lu] for lu in am_lus]) 
    
        # Get the GHG emission from agricultural management, then reshape it to starte with row (r) dimension
        am_ghg_mrj = ag_man_g_mrj[am] * data.ag_man_dvars[yr_cal][am][:, :, am_j]              # mrj 
        am_ghg_rm = np.einsum('mrj -> rm',am_ghg_mrj)                                         # rm
    
        # Summarize the df by calculating the total value of each column
        ag_ghg_arrays.append(am_ghg_rm)
    
    
    # Concat all summary tables
    ag_ghg_summary = np.stack(ag_ghg_arrays)                                           # srm
    ag_ghg_summary = np.einsum('srm -> rms',ag_ghg_summary)                            # rms
    
    # Summarize the array as a df
    ag_ghg_summary_df = tools.summarize_ghg_separate_df(ag_ghg_summary,( ['Agricultural Management']
                                                                , data.LANDMANS
                                                                , [f"TCO2E_{i}" for i in AG_MANAGEMENTS_TO_LAND_USES.keys()]),
                                                       lu_desc)
        
    # Save table to disk
    ag_ghg_summary_df.to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_management_{yr_cal}.csv'))
    
    
    
    

def write_ghg_offland_commodity(data: Data, path, yr_cal):
    """Write out offland commodity GHG emissions"""
    
    print('Writing offland commodity GHG\n')

    # Get the offland commodity data
    offland_ghg = data.OFF_LAND_GHG_EMISSION.query(f'year == {yr_cal}')
    
    # Save to disk
    offland_ghg.to_csv(os.path.join(path, f'GHG_emissions_offland_commodity_{yr_cal}.csv'), index = False)
    
    
