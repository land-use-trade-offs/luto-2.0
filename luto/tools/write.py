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

from luto import tools
from luto.tools.spatializers import *
from luto.tools.compmap import *
import luto.settings as settings
from luto.data import Data

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
    if settings.MODE == 'timeseries':
        # Write the target-year outputs to the path_begin_end_compare
        jobs += [
            delayed(write_output_single_year)(
                data, years[-1], f"{begin_end_path}/out_{years[-1]}", years[0]
            )
        ]

    # Parallel write the outputs for each year
    num_jobs = min(len(jobs), settings.WRITE_THREADS) if settings.PARALLEL_WRITE else 1   # Use the minimum between jobs_num and threads for parallel writing
    Parallel(n_jobs=num_jobs, prefer='threads')(jobs)
    
    # Copy the base-year outputs to the path_begin_end_compare
    shutil.copytree(f"{data.path}/out_{years[0]}", f"{begin_end_path}/out_{years[0]}", dirs_exist_ok = True)

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
    
    for log in logs:
        if os.path.exists(log):
            # Move the file to the output directory
            shutil.move(log, f"{data.path}/{os.path.basename(log)}")



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

        settings_order = []
        for line in lines:
            match = parameter_reg.match(line)
            if match:
                settings_order.append(match[1].strip())


        # Reorder the settings dictionary to match the order in the settings.py file
        settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
        settings_dict = {i: settings_dict[i] for i in settings_order if i in settings_dict}

        # Some variables are mutually exclusive, 
        # so we set the unsed variable to None here.
        if settings.GHG_LIMITS_TYPE == 'dict': 
            settings_dict['GHG_LIMITS_FIELD'] = 'None'
        elif settings.GHG_LIMITS_TYPE == 'file':
            settings_dict['GHG_LIMITS'] = 'None'

        if settings.CULL_MODE == 'absolute':
            settings_dict['LAND_USAGE_CULL_PERCENTAGE'] = 'None'
        elif settings.CULL_MODE == 'percentage':
            settings_dict['MAX_LAND_USES_PER_CELL'] = 'None'
        elif settings.CULL_MODE == 'none':
            settings_dict['LAND_USAGE_CULL_PERCENTAGE'] = 'None'
            settings_dict['MAX_LAND_USES_PER_CELL'] = 'None'

        if settings.WATER_USE_LIMITS == 'on':
            if settings.WATER_LIMITS_TYPE == 'pct_ag':
                settings_dict['WATER_STRESS_FRACTION'] = 'None'
            elif settings.WATER_LIMITS_TYPE == 'water_stress':
                settings_dict['WATER_USE_REDUCTION_PERCENTAGE'] = 'None'

    # Write the settings to a file
    with open(os.path.join(path, 'model_run_settings.txt'), 'w') as f:
        for k, v in settings_dict.items():
            f.write(f'{k}:{v}\n')



def write_files(data: Data, yr_cal, path):
    """Writes numpy arrays and geotiffs to file"""
    
    print(f'Writing numpy arrays and geotiff outputs to {path}')
    
    # Save raw agricultural decision variables (float array).
    ag_X_mrj_fname = 'ag_X_mrj' + '_' + str(yr_cal) + '.npy'
    np.save(os.path.join(path, ag_X_mrj_fname), data.ag_dvars[yr_cal].astype(np.float16))
    
    # Save raw non-agricultural decision variables (float array).
    non_ag_X_rk_fname = 'non_ag_X_rk' + '_' + str(yr_cal) + '.npy'
    np.save(os.path.join(path, non_ag_X_rk_fname), data.non_ag_dvars[yr_cal].astype(np.float16))

    # Save raw agricultural management decision variables (float array).
    for am in AG_MANAGEMENTS_TO_LAND_USES:
        snake_case_am = tools.am_name_snake_case(am)
        am_X_mrj_fname = 'ag_man_X_mrj' + snake_case_am + '_' + str(yr_cal) + ".npy"
        np.save(os.path.join(path, am_X_mrj_fname), data.ag_man_dvars[yr_cal][am].astype(np.float16))
    
    # Write out raw numpy arrays for land-use and land management
    lumap_fname = 'lumap' + '_' + str(yr_cal) + '.npy'
    lmmap_fname = 'lmmap' + '_' + str(yr_cal) + '.npy'
    
    np.save(os.path.join(path, lumap_fname), data.lumaps[yr_cal])
    np.save(os.path.join(path, lmmap_fname), data.lmmaps[yr_cal])
    
    
    
    # Get the Agricultural Management applied to each pixel
    ag_man_dvar = np.stack([np.einsum('mrj -> r', v) for _,v in data.ag_man_dvars[yr_cal].items()]).T   # (r, am)
    ag_man_dvar_mask = ag_man_dvar.sum(1) > 0.01            # Meaning that they have at least 1% of agricultural management applied
    # Get the maximum index of the agricultural management applied to the valid pixel
    ag_man_dvar = np.argmax(ag_man_dvar, axis=1) + 1        # Start from 1
    # Let the pixels that were all zeros in the original array to be 0
    ag_man_dvar_argmax = np.where(ag_man_dvar_mask, ag_man_dvar, 0)
    

    # Get the non-agricultural landuse for each pixel
    non_ag_dvar = data.non_ag_dvars[yr_cal]                  # (r, k)
    non_ag_dvar_mask = non_ag_dvar.sum(1) > 0.01            # Meaning that they have at least 1% of non-agricultural landuse applied
    # Get the maximum index of the non-agricultural landuse applied to the valid pixel
    non_ag_dvar = np.argmax(non_ag_dvar, axis=1) + settings.NON_AGRICULTURAL_LU_BASE_CODE    # Start from 100
    # Let the pixels that were all zeros in the original array to be 0
    non_ag_dvar_argmax = np.where(non_ag_dvar_mask, non_ag_dvar, 0)
 
    
    # Put the excluded land-use and land management types back in the array.
    lumap = create_2d_map(data, data.lumaps[yr_cal], filler = data.MASK_LU_CODE)
    lmmap = create_2d_map(data, data.lmmaps[yr_cal], filler = data.MASK_LU_CODE)
    ammap = create_2d_map(data, ag_man_dvar_argmax, filler = data.MASK_LU_CODE)
    non_ag = create_2d_map(data, non_ag_dvar_argmax, filler = data.MASK_LU_CODE)
    
    lumap_fname = 'lumap' + '_' + str(yr_cal) + '.tiff'
    lmmap_fname = 'lmmap' + '_' + str(yr_cal) + '.tiff'
    ammap_fname = 'ammap' + '_' + str(yr_cal) + '.tiff'
    non_ag_fname = 'non_ag' + '_' + str(yr_cal) + '.tiff'
    
    write_gtiff(lumap, os.path.join(path, lumap_fname))
    write_gtiff(lmmap, os.path.join(path, lmmap_fname)) 
    write_gtiff(ammap, os.path.join(path, ammap_fname))  
    write_gtiff(non_ag, os.path.join(path, non_ag_fname))        



def write_files_separate(data: Data, yr_cal, path, ammap_separate=False):

    # Write raw decision variables to separate GeoTiffs
    # 
    # Here we use decision variables to create TIFFs rather than directly creating them from 
    #       {data.lu/lmmaps, data.ammaps}, because the decision variables is {np.float32} and 
    #       the data.lxmap is {np.int8}. 
    # 
    # In this way, if partial land-use is allowed (i.e, one pixel is determined to be 50% of apples and 50% citrus),
    #       we can handle the fractional land-use successfully.

    
    # 1) Collapse the land management dimension (m -> [dry, irr])
    #    i.e., mrj -> rj
    ag_dvar_rj = np.einsum('mrj -> rj', data.ag_dvars[yr_cal])   # To compute the landuse map
    ag_dvar_rm = np.einsum('mrj -> rm', data.ag_dvars[yr_cal])   # To compute the land management (dry/irr) map
    non_ag_rk = np.einsum('rk -> rk', data.non_ag_dvars[yr_cal]) # Do nothing, just for code consistency
    ag_man_rj_dict = {am: np.einsum('mrj -> rj', ammap) for am, ammap in data.ag_man_dvars[yr_cal].items()}
    

    # 2) Get the desc2dvar table. 
    #    desc is the land-use description, dvar is the decision variable corresponding to desc
    ag_dvar_map = tools.map_desc_to_dvar_index('Ag_LU', 
                                               data.DESC2AGLU, 
                                               ag_dvar_rj)

    non_ag_dvar_map = tools.map_desc_to_dvar_index('Non-Ag_LU',
                                                   {v:k for k,v in dict(list(enumerate(data.NON_AGRICULTURAL_LANDUSES))).items()},
                                                    non_ag_rk)
    
    lm_dvar_map = tools.map_desc_to_dvar_index('Land_Mgt',
                                                {i[1]:i[0] for i in enumerate(data.LANDMANS)},
                                                ag_dvar_rm)
    
    # Get the desc2dvar table for agricultural management
    ag_man_maps = []
    for am,am_dvar in ag_man_rj_dict.items():
        desc2idx = {desc: data.DESC2AGLU[desc] for desc in  AG_MANAGEMENTS_TO_LAND_USES[am]}
        # Check if need to separate the agricultural management into different land uses
        if ammap_separate == True:
            am_map = tools.map_desc_to_dvar_index(am, desc2idx, am_dvar)
        else:
            am_dvar = am_dvar.sum(1)[:,np.newaxis]
            am_map = tools.map_desc_to_dvar_index('Ag_Mgt', {am:0}, am_dvar)
        ag_man_maps.append(am_map)
        
    ag_man_map = pd.concat(ag_man_maps)
    
    
    # Combine the desc2dvar table for agricultural land-use, agricultural management, and non-agricultural land-use
    desc2dvar_df = pd.concat([ag_dvar_map, ag_man_map, non_ag_dvar_map, lm_dvar_map])

    lucc_separate_dir =  os.path.join(path, 'lucc_separate')
    if not os.path.isdir(lucc_separate_dir):
        os.mkdir(lucc_separate_dir)
    
    # 3) Export to GeoTiff
    for _,row in desc2dvar_df.iterrows():
        # Get the Category, land-use desc, and dvar
        category = row['Category']
        dvar_idx = row['dvar_idx']
        desc = row['lu_desc']

        # reconsititude the dvar to 2d
        dvar = row['dvar']
        dvar = create_2d_map(data, dvar, filler = data.MASK_LU_CODE) # fill the missing values with data.MASK_LU_CODE  

        # Create output file name
        fname = f'{category}_{dvar_idx:02}_{desc}_{yr_cal}.tiff'
        lucc_separate_path = os.path.join(lucc_separate_dir, fname)

        # Write to GeoTiff
        write_gtiff(dvar, lucc_separate_path)



def write_quantity(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    
    timestamp = data.timestamp_sim

    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    
    # Get index of yr_cal in timeseries (e.g., if yr_cal is 2050 then yr_idx = 40)
    yr_idx = yr_cal - data.YR_CAL_BASE
    
    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    
    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    
    # Calculate data for quantity comparison between base year and target year
    if yr_cal > data.YR_CAL_BASE:

        # Check if yr_cal_sim_pre meets the requirement
        assert yr_cal_sim_pre >= data.YR_CAL_BASE and yr_cal_sim_pre < yr_cal,\
            f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"
        
        # Get commodity production quantities produced in base year and target year
        prod_base = np.array(data.prod_data[yr_cal_sim_pre]['Production'])
        prod_targ = np.array(data.prod_data[yr_cal]['Production'])

        demands = data.D_CY[yr_idx]                        # Get commodity demands for target year
        abs_diff = prod_targ - demands                     # Diff between target year production and demands in absolute terms (i.e. tonnes etc)
        prop_diff = ( prod_targ / demands ) * 100          # Target year production as a proportion of demands (%)
        
        # Write to pandas dataframe
        df = pd.DataFrame()
        df['Commodity'] = data.COMMODITIES
        df['Prod_base_year (tonnes, KL)'] = prod_base
        df['Prod_targ_year (tonnes, KL)'] = prod_targ
        df['Demand (tonnes, KL)'] = demands
        df['Abs_diff (tonnes, KL)'] = abs_diff
        df['Prop_diff (%)'] = prop_diff

        # Save files to disk      
        df.to_csv(os.path.join(path, f'quantity_comparison_{timestamp}.csv'), index = False)

        # Write the production of each year to disk
        production_years = pd.DataFrame({yr:data.prod_data[yr]['Production'] for yr in data.prod_data.keys()})
        production_years.insert(0,'Commodity',data.COMMODITIES)
        production_years.to_csv(os.path.join(path, f'quantity_production_kt_{timestamp}.csv'), index = False)

    # --------------------------------------------------------------------------------------------
    # NOTE:Non-agricultural production are all zeros, therefore skip the calculation
    # --------------------------------------------------------------------------------------------
        


def write_revenue_cost_ag(data: Data, yr_cal, path):
    """Calculate agricultural revenue. Takes a simulation object, a target calendar 
       year (e.g., 2030), and an output path as input."""

    timestamp = data.timestamp_sim

    print(f'Writing agricultural revenue outputs to {path}' )

    # Convert calendar year to year index.
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the ag_dvar_mrj in the yr_cal
    ag_dvar_mrj = data.ag_dvars[yr_cal]

    
    # Get agricultural revenue/cost for year in mrjs format. Note the s stands for sources:
    # E.g., Sources for crops only contains ['Revenue'], 
    #    but sources for livestock includes ['Meat', 'Wool', 'Live Exports', 'Milk']
    ag_rev_df_rjms = ag_revenue.get_rev_matrices(data, yr_idx, aggregate=False)
    ag_cost_df_rjms = ag_cost.get_cost_matrices(data, yr_idx, aggregate=False)

    # Expand the original df with zero values to convert it to a **mrjs** array
    ag_rev_df_rjms = ag_rev_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_rev_df_rjms.columns.levels), fill_value=0)
    ag_rev_rjms = ag_rev_df_rjms.values.reshape(-1, *ag_rev_df_rjms.columns.levshape)

    ag_cost_df_rjms = ag_cost_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_cost_df_rjms.columns.levels), fill_value=0)
    ag_cost_rjms = ag_cost_df_rjms.values.reshape(-1, *ag_cost_df_rjms.columns.levshape)


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
    
    # Add the SUM column to the last row and column
    df_rev.loc['SUM'] = df_rev.sum(axis=0)
    df_rev['SUM'] = df_rev.sum(axis=1)

    df_cost.loc['SUM'] = df_cost.sum(axis=0)
    df_cost['SUM'] = df_cost.sum(axis=1)

    # Save to file
    df_rev.to_csv(os.path.join(path, f'revenue_agricultural_commodity_{timestamp}.csv'))
    df_cost.to_csv(os.path.join(path, f'cost_agricultural_commodity_{timestamp}.csv'))
    

def write_revenue_cost_ag_management(data: Data, yr_cal, path):
    """Calculate agricultural management revenue and cost."""
    
    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = data.timestamp_sim

    print(f'Writing agricultural management revenue and cost outputs to {path}')

    # Convert calendar year to year index.
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
    for am in AG_MANAGEMENTS_TO_LAND_USES:
        
        # Get the land use codes for the agricultural management
        am_desc = AG_MANAGEMENTS_TO_LAND_USES[am]
        am_code = [data.DESC2AGLU[desc] for desc in am_desc]
        
        # Get the revenue/cost matrix for the agricultural management
        am_rev = np.nan_to_num(am_revenue_mat[am]) # Replace NaNs with 0
        am_cost = np.nan_to_num(am_cost_mat[am])   # Replace NaNs with 0
        
        # Get the decision variable for each agricultural management
        am_dvar = data.ag_man_dvars[yr_cal][am][:,:,am_code]
        
        # Multiply the decision variable by revenue matrix
        am_rev_yr = np.einsum('mrj,mrj->jm', am_dvar, am_rev)
        am_cost_yr = np.einsum('mrj,mrj->jm', am_dvar, am_cost)
        
        # Reformat the revenue/cost matrix into a dataframe
        am_rev_yr_df = pd.DataFrame(am_rev_yr, columns=data.LANDMANS)
        am_rev_yr_df['Land-use'] = am_desc
        am_rev_yr_df = am_rev_yr_df.melt(id_vars='Land-use',value_vars=data.LANDMANS,var_name='Water',value_name='Value (AU$)')
        am_rev_yr_df['year'] = yr_cal
        am_rev_yr_df['Management Type'] = am
        
        am_cost_yr_df = pd.DataFrame(am_cost_yr, columns=data.LANDMANS)
        am_cost_yr_df['Land-use'] = am_desc
        am_cost_yr_df = am_cost_yr_df.melt(id_vars='Land-use',value_vars=data.LANDMANS,var_name='Water',value_name='Value (AU$)')
        am_cost_yr_df['year'] = yr_cal
        am_cost_yr_df['Management Type'] = am
        
        # Store the revenue/cost dataframes
        revenue_am_dfs.append(am_rev_yr_df)
        cost_am_dfs.append(am_cost_yr_df)   

    # Concatenate the revenue/cost dataframes    
    revenue_am_df = pd.concat(revenue_am_dfs)
    cost_am_df = pd.concat(cost_am_dfs)
    
    revenue_am_df.to_csv(os.path.join(path, f'revenue_agricultural_management_{timestamp}.csv'), index = False)
    cost_am_df.to_csv(os.path.join(path, f'cost_agricultural_management_{timestamp}.csv'), index = False)
    
    
    
def write_cost_transition(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Calculate transition cost."""
    
    print(f'Writing transition cost outputs to {path}')
    
    timestamp = data.timestamp_sim
    
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
    cost_df.to_csv(os.path.join(path, f'cost_transition_ag2ag_{timestamp}.csv'), index=False)




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
    cost_df.to_csv(os.path.join(path, f'cost_transition_ag2non_ag_{timestamp}.csv'), index=False)




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
    cost_df.to_csv(os.path.join(path, f'cost_transition_non_ag2_ag_{timestamp}.csv'), index=False)


def write_revenue_cost_non_ag(data: Data, yr_cal, path):
    """Calculate non_agricultural cost. """

    timestamp = data.timestamp_sim

    print(f'Writing non agricultural management cost outputs to {path}')
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
    rev_non_ag_df.to_csv(os.path.join(path, f'revenue_non_ag_{timestamp}.csv'), index = False)
    cost_non_ag_df.to_csv(os.path.join(path, f'cost_non_ag_{timestamp}.csv'), index = False)
    
    
    

def write_dvar_area(data: Data, yr_cal, path):
    
    timestamp = data.timestamp_sim
    
    # Reprot the process
    print(f'Writing area calculated from dvars to {path}')
    

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
    df_ag_area.to_csv(os.path.join(path, f'area_agricultural_landuse_{timestamp}.csv'), index = False)
    df_non_ag_area.to_csv(os.path.join(path, f'area_non_agricultural_landuse_{timestamp}.csv'), index = False)
    df_am_area.to_csv(os.path.join(path, f'area_agricultural_management_{timestamp}.csv'), index = False)


def write_area_transition_start_end(data: Data, path):
    
    # Append the yr_cal to timestamp as prefix
    timestamp = data.timestamp_sim 
    
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
    
    # Write the transition matrix to a csv file
    transition.to_csv(os.path.join(path, f'transition_matrix_{data.YR_CAL_BASE}_{yr_cal_end}_{timestamp}.csv'))



def write_crosstab(data: Data, yr_cal, path, yr_cal_sim_pre=None): 
    """Write out land-use and production data"""
    
    timestamp = data.timestamp_sim
    
    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    
    # Get index of yr_cal in timeseries (e.g., if yr_cal is 2050 then yr_idx = 40)
    yr_idx = yr_cal - data.YR_CAL_BASE
    
    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    
    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre
    

    # Only perform the calculation if the yr_cal is not the base year
    if yr_cal > data.YR_CAL_BASE:

        # Check if yr_cal_sim_pre meets the requirement
        assert yr_cal_sim_pre >= data.YR_CAL_BASE and yr_cal_sim_pre < yr_cal,\
            f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"

        print(f'Writing production outputs to {path}')

        # LUS = ['Non-agricultural land'] + data.AGRICULTURAL_LANDUSES + data.NON_AGRICULTURAL_LANDUSES
        ctlu, swlu = lumap_crossmap( data.lumaps[yr_cal_sim_pre]
                                   , data.lumaps[yr_cal]
                                   , data.AGRICULTURAL_LANDUSES
                                   , data.NON_AGRICULTURAL_LANDUSES
                                   , data.REAL_AREA)
        
        ctlm, swlm = lmmap_crossmap( data.lmmaps[yr_cal_sim_pre]
                                   , data.lmmaps[yr_cal]
                                   , data.REAL_AREA)

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
            
        ctlu.to_csv(os.path.join(path, f'crosstab-lumap_{timestamp}.csv'))
        ctlm.to_csv(os.path.join(path, f'crosstab-lmmap_{timestamp}.csv'))
        swlu.to_csv(os.path.join(path, f'switches-lumap_{timestamp}.csv'))
        swlm.to_csv(os.path.join(path, f'switches-lmmap_{timestamp}.csv'))
        cthp.to_csv(os.path.join(path, f'crosstab-irrstat_{timestamp}.csv'))
        swhp.to_csv(os.path.join(path, f'switches-irrstat_{timestamp}.csv'))
        
        for am in AG_MANAGEMENTS_TO_LAND_USES:
            am_snake_case = tools.am_name_snake_case(am).replace("_", "-")
            ctass[am].to_csv(os.path.join(path, f'crosstab-amstat-{am_snake_case}_{timestamp}.csv'))
            swass[am].to_csv(os.path.join(path, f'switches-amstat-{am_snake_case}_{timestamp}.csv'))



def write_water(data: Data, yr_cal, path):
    """Calculate water use totals. Takes a simulation object, a numeric
       target calendar year (e.g., 2030), and an output path as input."""

    timestamp = data.timestamp_sim

    print(f'Writing water outputs to {path}')

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
    df.to_csv( os.path.join(path, f'water_demand_vs_use_{timestamp}.csv')
             , index = False
             , float_format = '{:0,.2f}'.format)

    # Write the separate water use to CSV
    df_water_seperate = pd.concat(df_water_seperate_dfs)
    df_water_seperate.to_csv( os.path.join(path, f'water_demand_vs_use_separate_{timestamp}.csv')
                            , index = False)
    
    
    

def write_ghg(data: Data, yr_cal, path):
    """Calculate total GHG emissions from on-land agricultural sector. 
        Takes a simulation object, a target calendar year (e.g., 2030), 
        and an output path as input."""

    timestamp = data.timestamp_sim

    print(f'Writing GHG outputs to {path}' )
    

    yr_idx = yr_cal - data.YR_CAL_BASE
        
    # Prepare a data frame.
    df = pd.DataFrame( columns=[ 'GHG_EMISSIONS_LIMIT_TCO2e'
                               , 'GHG_EMISSIONS_TCO2e' ] )

    # Get GHG emissions limits used as constraints in model
    ghg_limits = ag_ghg.get_ghg_limits(data, yr_cal)

    # Get GHG emissions from model
    if yr_cal >= data.YR_CAL_BASE + 1:
        ghg_emissions = data.prod_data[yr_cal]['GHG Emissions']
    else:
        ghg_emissions = (ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True) * data.ag_dvars[data.YR_CAL_BASE]).sum()
        
    # Add to dataframe
    df.loc[0] = ("{:,.0f}".format(ghg_limits), "{:,.0f}".format(ghg_emissions))
    
    # Save to file
    df.to_csv(os.path.join(path, f'GHG_emissions_{timestamp}.csv'), index = False)


def write_biodiversity(data: Data, yr_cal, path):
    """
    Write biodiversity info for a given year ('yr_cal'), simulation ('sim')
    and output path ('path').
    """

    timestamp = data.timestamp_sim

    print(f'Writing biodiversity outputs to {path}')


    yr_idx = yr_cal - data.YR_CAL_BASE

    df = pd.DataFrame( columns=[ 'Biodiversity score limit'
                               , f'Solve biodiversity score ({yr_cal})' ] )
    
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
    df.loc[0] = ("{:,.0f}".format(biodiv_limit), "{:,.0f}".format(biodiv_score))
    
    # Save to file
    df.to_csv(os.path.join(path, f'biodiversity_{timestamp}.csv'), index = False)
    
    
def write_biodiversity_separate(data: Data, yr_cal, path):
    
    timestamp = data.timestamp_sim

    print(f'Writing biodiversity_separate outputs to {path}')


    # Get the biodiversity scores b_mrj
    yr_idx = yr_cal - data.YR_CAL_BASE
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
    biodiv_df.to_csv(os.path.join(path, 'biodiversity_separate_' + timestamp + '.csv'), index=False)    
      
    
  
def write_ghg_separate(data: Data, yr_cal, path):


    timestamp = data.timestamp_sim

    print(f'Writing GHG emissions_Separate to {path}')

        
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
    ghg_df.to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_landuse_{timestamp}.csv'))



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
    non_ag_g_rk_summary.to_csv(os.path.join(path, f'GHG_emissions_separate_no_ag_reduction_{timestamp}.csv'))
                        

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
    ghg_t_separate_summary.to_csv(os.path.join(path, f'GHG_emissions_separate_transition_penalty_{timestamp}.csv'))
    
    
    
    
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
    ag_ghg_summary_df.to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_management_{timestamp}.csv'))
    

def write_ghg_offland_commodity(data: Data, path):
    """Write out offland commodity GHG emissions"""
    
    print(f'Writing offland commodity GHG to {path}\n')

    # Append the yr_cal to timestamp as prefix
    timestamp = data.timestamp_sim

    # Get the offland commodity data
    offland_ghg = data.OFF_LAND_GHG_EMISSION
    
    # Save to disk
    offland_ghg.to_csv(os.path.join(path, f'GHG_emissions_offland_commodity_{timestamp}.csv'), index = False)
    
    
