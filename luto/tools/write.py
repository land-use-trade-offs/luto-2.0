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

import os, math
import shutil
from datetime import datetime
import time
import numpy as np
import pandas as pd
from itertools import product

import luto.settings as settings
from luto import tools
from luto.tools.spatializers import *
from luto.tools.compmap import *
import luto.economics.agricultural.water as ag_water
import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.cost as ag_cost
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.agricultural.biodiversity as ag_biodiversity

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


def get_path(sim):
    """Create a folder for storing outputs and return folder name."""

    # Get date and time
    path = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Get the years to write
    yr_all = sorted(list(sim.lumaps.keys()))

    # Add some shorthand details about the model run
    post = '_' + settings.DEMAND_CONSTRAINT_TYPE + \
           '_' + settings.OBJECTIVE + \
           '_RF' + str(settings.RESFACTOR) + \
           '_P1e' + str(int(math.log10(settings.PENALTY))) + \
           '_' + str(yr_all[0]) + '-' + str(yr_all[-1]) + \
           '_' + settings.MODE + \
           '_' + str(int(sim.data.GHG_TARGETS[yr_all[-1]] / 1e6)) + 'Mt'

    # Create path name
    path = 'output/' + path + post

    # Get all paths
    paths = [path] \
            + [f"{path}/out_{yr}" for yr in yr_all] \
            + [f"{path}/out_{yr}/lucc_separate" for yr in yr_all[1:]]  # Skip creating lucc_separate for base year

    # Add the path for the comparison between base-year and target-year if in the timeseries mode
    if settings.MODE == 'timeseries':
        path_begin_end_compare = f"{path}/begin_end_compare_{yr_all[0]}_{yr_all[-1]}"
        paths = paths \
                + [path_begin_end_compare] \
                + [f"{path_begin_end_compare}/out_{yr_all[0]}",
                   f"{path_begin_end_compare}/out_{yr_all[-1]}",
                   f"{path_begin_end_compare}/out_{yr_all[-1]}/lucc_separate"]

    # Create all paths
    for p in paths:
        if not os.path.exists(p):
            os.mkdir(p)

    return path


def write_outputs(sim, path):
    # Write model run settings
    write_settings(path)

    # Get the years to write
    years = sorted(list(sim.lumaps.keys()))
    paths = [f"{path}/out_{yr}" for yr in years]

    # Write outputs for each year
    for idx, (yr, path_yr) in enumerate(zip(years, paths)):
        write_output_single_year(sim, yr, path_yr, yr_cal_sim_pre=None)
        print(f"Finished writing {yr} out of {years[0]}-{years[-1]} years\n")

    # Write the area/quantity comparison between base-year and target-year for the timeseries mode
    if settings.MODE == 'timeseries':
        begin_end_path = f"{path}/begin_end_compare_{years[0]}_{years[-1]}"
        # 1) Simply copy the base-year outputs to the path_begin_end_compare
        shutil.copytree(f"{path}/out_{years[0]}", f"{begin_end_path}/out_{years[0]}", dirs_exist_ok=True)
        # 2) Write the target-year outputs to the path_begin_end_compare
        write_output_single_year(sim, years[-1], f"{begin_end_path}/out_{years[-1]}", yr_cal_sim_pre=years[0])
        print(f"Finished writing {years[0]}-{years[-1]} comparison\n")

    ###############################################################
    #               Fire up the report script                     #
    ###############################################################

    # Use sub process to run the report script
    import subprocess

    # 1) Clean up the output CSVs
    result = subprocess.run(['python', 'luto/tools/report/create_report_data.py', '-p', path], capture_output=True,
                            text=True)
    print("\nError occurred:", result.stderr) if result.returncode != 0 else print("\nReport data:", result.stdout)

    # 2) Create the report
    result = subprocess.run(['python', 'luto/tools/report/create_html.py', '-p', path], capture_output=True, text=True)
    print("\nError occurred:", result.stderr) if result.returncode != 0 else print("\nReport HTML:", result.stdout)


def write_output_single_year(sim, yr_cal, path_yr, yr_cal_sim_pre=None):
    """Write outputs for simulation 'sim', calendar year, demands d_c, and path"""
    if not os.path.isdir(path_yr):
        os.mkdir(path_yr)

    # Write the decision variables, land-use and land management maps
    # write_files(sim, yr_cal, path_yr)
    # write_files_separate(sim, yr_cal, path_yr)

    # Write the crosstab and switches, and the quantity comparison
    write_crosstab(sim, yr_cal, path_yr, yr_cal_sim_pre)
    write_quantity(sim, yr_cal, path_yr, yr_cal_sim_pre)

    # Write the water and GHG outputs
    write_ag_revenue_cost(sim, yr_cal, path_yr)
    write_ag_management_cost(sim, yr_cal, path_yr)
    write_non_ag_cost(sim, yr_cal, path_yr)
    write_water(sim, yr_cal, path_yr)
    write_ghg(sim, yr_cal, path_yr)
    write_ghg_separate(sim, yr_cal, path_yr)
    write_biodiversity(sim, yr_cal, path_yr)

def write_outputs_snapshot(sim, path):
    # Write model run settings
    write_settings(path)

    # Get the years to write
    years = sorted(list(sim.lumaps.keys()))
    paths = [f"{path}/out_{yr}" for yr in years]

    # Write outputs for each year
    for idx, (yr, path_yr) in enumerate(zip(years, paths)):
        write_output_single_year_snapshot(sim, yr, path_yr, yr_cal_sim_pre=None)
        print(f"Finished writing {yr} out of {years[0]}-{years[-1]} years\n")

    # Write the area/quantity comparison between base-year and target-year for the timeseries mode
    if settings.MODE == 'timeseries':
        begin_end_path = f"{path}/begin_end_compare_{years[0]}_{years[-1]}"
        # 1) Simply copy the base-year outputs to the path_begin_end_compare
        shutil.copytree(f"{path}/out_{years[0]}", f"{begin_end_path}/out_{years[0]}", dirs_exist_ok=True)
        # 2) Write the target-year outputs to the path_begin_end_compare
        write_output_single_year(sim, years[-1], f"{begin_end_path}/out_{years[-1]}", yr_cal_sim_pre=years[0])
        print(f"Finished writing {years[0]}-{years[-1]} comparison\n")

    ###############################################################
    #               Fire up the report script                     #
    ###############################################################

    # Use sub process to run the report script
    import subprocess

    # 1) Clean up the output CSVs
    result = subprocess.run(['python', 'luto/tools/report/create_report_data.py', '-p', path], capture_output=True,
                            text=True)
    print("\nError occurred:", result.stderr) if result.returncode != 0 else print("\nReport data:", result.stdout)

    # 2) Create the report
    result = subprocess.run(['python', 'luto/tools/report/create_html.py', '-p', path], capture_output=True, text=True)
    print("\nError occurred:", result.stderr) if result.returncode != 0 else print("\nReport HTML:", result.stdout)


def write_output_single_year_snapshot(sim, yr_cal, path_yr, yr_cal_sim_pre=None):
    """Write outputs for simulation 'sim', calendar year, demands d_c, and path"""
    if not os.path.isdir(path_yr):
        os.mkdir(path_yr)

    # Write the decision variables, land-use and land management maps
    write_files(sim, yr_cal, path_yr)
    write_files_separate(sim, yr_cal, path_yr)

    # Write the crosstab and switches, and the quantity comparison
    write_crosstab(sim, yr_cal, path_yr, yr_cal_sim_pre)
    write_quantity(sim, yr_cal, path_yr, yr_cal_sim_pre)

    # Write the water and GHG outputs
    write_ag_revenue_cost(sim, yr_cal, path_yr)
    write_ag_management_cost(sim, yr_cal, path_yr)
    write_non_ag_cost(sim, yr_cal, path_yr)
    write_water(sim, yr_cal, path_yr)
    write_ghg(sim, yr_cal, path_yr)
    write_ghg_separate(sim, yr_cal, path_yr)
    write_biodiversity(sim, yr_cal, path_yr)

def write_settings(path):
    """Write model run settings"""

    with open(os.path.join(path, 'model_run_settings.txt'), 'w') as f:
        f.write('LUTO version %s\n' % settings.VERSION)
        f.write('SSP %s, RCP %s\n' % (settings.SSP, settings.RCP))
        f.write('DISCOUNT_RATE: %s\n' % settings.DISCOUNT_RATE)
        f.write('AMORTISATION_PERIOD: %s\n' % settings.AMORTISATION_PERIOD)
        f.write('CO2_FERT: %s\n' % settings.CO2_FERT)
        f.write('ENV_PLANTING_COST_PER_HA_PER_YEAR: %s\n' % settings.ENV_PLANTING_COST_PER_HA_PER_YEAR)
        f.write('CARBON_PRICE_PER_TONNE: %s\n' % settings.CARBON_PRICE_PER_TONNE)
        f.write('AGRICULTURAL_MANAGEMENT_USE_THRESHOLD: %s\n' % settings.AGRICULTURAL_MANAGEMENT_USE_THRESHOLD)
        f.write('SOC_AMORTISATION: %s\n' % settings.SOC_AMORTISATION)
        f.write('RESFACTOR: %s\n' % settings.RESFACTOR)
        f.write('MODE: %s\n' % settings.MODE)
        f.write('OBJECTIVE: %s\n' % settings.OBJECTIVE)
        f.write('DEMAND_CONSTRAINT_TYPE: %s\n' % settings.DEMAND_CONSTRAINT_TYPE)
        f.write('SOLVE_METHOD: %s\n' % settings.SOLVE_METHOD)
        f.write('PENALTY: %s\n' % settings.PENALTY)
        f.write('OPTIMALITY_TOLERANCE: %s\n' % settings.OPTIMALITY_TOLERANCE)
        f.write('THREADS: %s\n' % settings.THREADS)

        f.write('CULL_MODE: %s\n' % settings.CULL_MODE)
        if settings.CULL_MODE == 'absolute':
            f.write('MAX_LAND_USES_PER_CELL: %s\n' % settings.MAX_LAND_USES_PER_CELL)
        elif settings.CULL_MODE == 'percentage':
            f.write('LAND_USAGE_CULL_PERCENTAGE: %s\n' % settings.LAND_USAGE_CULL_PERCENTAGE)

        f.write('WATER_USE_LIMITS: %s\n' % settings.WATER_USE_LIMITS)
        if settings.WATER_USE_LIMITS == 'on':
            f.write('WATER_LIMITS_TYPE: %s\n' % settings.WATER_LIMITS_TYPE)
            if settings.WATER_LIMITS_TYPE == 'pct_ag':
                f.write('WATER_USE_REDUCTION_PERCENTAGE: %s\n' % settings.WATER_USE_REDUCTION_PERCENTAGE)
            elif settings.WATER_LIMITS_TYPE == 'water_stress':
                f.write('WATER_STRESS_FRACTION: %s\n' % settings.WATER_STRESS_FRACTION)
            f.write('WATER_REGION_DEF: %s\n' % settings.WATER_REGION_DEF)

        f.write('GHG_EMISSIONS_LIMITS: %s\n' % settings.GHG_EMISSIONS_LIMITS)
        if settings.GHG_EMISSIONS_LIMITS == 'on':
            f.write('GHG_LIMITS_TYPE: %s\n' % settings.GHG_LIMITS_TYPE)
            if settings.GHG_LIMITS_TYPE == 'tonnes':
                f.write('GHG_LIMITS: %s\n' % settings.GHG_LIMITS)
            elif settings.GHG_LIMITS_TYPE == 'file':
                f.write('GHG_LIMITS: from file GHG_targets.xlsx in input folder')


def write_files(sim, yr_cal, path):
    """Writes numpy arrays and geotiffs to file"""

    print('Writing numpy arrays and geotiff outputs to', path)

    # Save raw agricultural decision variables (float array).
    ag_X_mrj_fname = 'ag_X_mrj' + '_' + str(yr_cal) + '.npy'
    np.save(os.path.join(path, ag_X_mrj_fname), sim.ag_dvars[yr_cal].astype(np.float16))

    # Save raw non-agricultural decision variables (float array).
    non_ag_X_rk_fname = 'non_ag_X_rk' + '_' + str(yr_cal) + '.npy'
    np.save(os.path.join(path, non_ag_X_rk_fname), sim.non_ag_dvars[yr_cal].astype(np.float16))

    # Save raw agricultural management decision variables (float array).
    for am in AG_MANAGEMENTS_TO_LAND_USES:
        snake_case_am = tools.am_name_snake_case(am)
        am_X_mrj_fname = 'ag_man_X_mrj' + snake_case_am + '_' + str(yr_cal) + ".npy"
        np.save(os.path.join(path, am_X_mrj_fname), sim.ag_man_dvars[yr_cal][am].astype(np.float16))

    # Write out raw numpy arrays for land-use and land management
    lumap_fname = 'lumap' + '_' + str(yr_cal) + '.npy'
    lmmap_fname = 'lmmap' + '_' + str(yr_cal) + '.npy'

    np.save(os.path.join(path, lumap_fname), sim.lumaps[yr_cal])
    np.save(os.path.join(path, lmmap_fname), sim.lmmaps[yr_cal])

    # Get the Agricultural Management applied to each pixel
    ag_man_dvar = np.stack([np.einsum('mrj -> r', v) for _, v in sim.ag_man_dvars[yr_cal].items()]).T  # (r, am)
    ag_man_dvar_mask = ag_man_dvar.sum(1) > 0  # Meaning that they have some agricultural management applied
    # Get the maximum index of the agricultural management applied to the valid pixel
    ag_man_dvar = np.argmax(ag_man_dvar, axis=1) + 1  # Start from 1
    # Let the pixels that were all zeros in the original array to be 0
    ag_man_dvar_argmax = np.where(ag_man_dvar_mask, ag_man_dvar, 0)

    # Get the non-agricultural landuse for each pixel
    non_ag_dvar = sim.non_ag_dvars[yr_cal]  # (r, k)
    non_ag_dvar_mask = non_ag_dvar.sum(1) > 0  # Meaning that they have some non-agricultural landuse applied
    # Get the maximum index of the non-agricultural landuse applied to the valid pixel
    non_ag_dvar = np.argmax(non_ag_dvar, axis=1) + sim.data.NON_AGRICULTURAL_LU_BASE_CODE  # Start from 100
    # Let the pixels that were all zeros in the original array to be 0
    non_ag_dvar_argmax = np.where(non_ag_dvar_mask, non_ag_dvar, 0)

    # Put the excluded land-use and land management types back in the array.
    lumap = create_2d_map(sim, sim.lumaps[yr_cal], filler=sim.data.MASK_LU_CODE)
    lmmap = create_2d_map(sim, sim.lmmaps[yr_cal], filler=sim.data.MASK_LU_CODE)
    ammap = create_2d_map(sim, ag_man_dvar_argmax, filler=sim.data.MASK_LU_CODE)
    non_ag = create_2d_map(sim, non_ag_dvar_argmax, filler=sim.data.MASK_LU_CODE)

    lumap_fname = 'lumap' + '_' + str(yr_cal) + '.tiff'
    lmmap_fname = 'lmmap' + '_' + str(yr_cal) + '.tiff'
    ammap_fname = 'ammap' + '_' + str(yr_cal) + '.tiff'
    non_ag_fname = 'non_ag' + '_' + str(yr_cal) + '.tiff'

    write_gtiff(lumap, os.path.join(path, lumap_fname))
    write_gtiff(lmmap, os.path.join(path, lmmap_fname))
    write_gtiff(ammap, os.path.join(path, ammap_fname))
    write_gtiff(non_ag, os.path.join(path, non_ag_fname))


def write_files_separate(sim, yr_cal, path, ammap_separate=False):
    # Write raw decision variables to separate GeoTiffs
    #
    # Here we use decision variables to create TIFFs rather than directly creating them from
    #       {sim.lu/lmmaps, sim.ammaps}, because the decision variables is {np.float32} and
    #       the sim.lxmap is {np.int8}.
    #
    # In this way, if partial land-use is allowed (i.e, one pixel is determined to be 50% of apples and 50% citrus),
    #       we can handle the fractional land-use successfully.

    # Skip writing if the yr_cal is the base year
    if yr_cal == sim.bdata.YR_CAL_BASE: return

    # 1) Collapse the land management dimension (m -> [dry, irr])
    #    i.e., mrj -> rj
    ag_dvar_rj = np.einsum('mrj -> rj', sim.ag_dvars[yr_cal])  # To compute the landuse map
    ag_dvar_rm = np.einsum('mrj -> rm', sim.ag_dvars[yr_cal])  # To compute the land management (dry/irr) map
    non_ag_rk = np.einsum('rk -> rk', sim.non_ag_dvars[yr_cal])  # Do nothing, just for code consistency
    ag_man_rj_dict = {am: np.einsum('mrj -> rj', ammap) for am, ammap in sim.ag_man_dvars[yr_cal].items()}

    # 2) Get the desc2dvar table.
    #    desc is the land-use description, dvar is the decision variable corresponding to desc
    ag_dvar_map = tools.map_desc_to_dvar_index('Ag_LU',
                                               sim.data.DESC2AGLU,
                                               ag_dvar_rj)

    non_ag_dvar_map = tools.map_desc_to_dvar_index('Non-Ag_LU',
                                                   {v: k for k, v in
                                                    dict(list(enumerate(sim.data.NON_AGRICULTURAL_LANDUSES))).items()},
                                                   non_ag_rk)

    lm_dvar_map = tools.map_desc_to_dvar_index('Land_Mgt',
                                               {i[1]: i[0] for i in enumerate(sim.data.LANDMANS)},
                                               ag_dvar_rm)

    # Get the desc2dvar table for agricultural management
    ag_man_maps = []
    for am, am_dvar in ag_man_rj_dict.items():
        desc2idx = {desc: sim.data.DESC2AGLU[desc] for desc in AG_MANAGEMENTS_TO_LAND_USES[am]}
        # Check if need to separate the agricultural management into different land uses
        if ammap_separate == True:
            am_map = tools.map_desc_to_dvar_index(am, desc2idx, am_dvar)
        else:
            am_dvar = am_dvar.sum(1)[:, np.newaxis]
            am_map = tools.map_desc_to_dvar_index('Ag_Mgt', {am: 0}, am_dvar)
        ag_man_maps.append(am_map)

    ag_man_map = pd.concat(ag_man_maps)

    # Combine the desc2dvar table for agricultural land-use, agricultural management, and non-agricultural land-use
    desc2dvar_df = pd.concat([ag_dvar_map, ag_man_map, non_ag_dvar_map, lm_dvar_map])

    # 3) Export to GeoTiff
    for _, row in desc2dvar_df.iterrows():
        # Get the Category, land-use desc, and dvar
        category = row['Category']
        dvar_idx = row['dvar_idx']
        desc = row['lu_desc']

        # reconsititude the dvar to 2d
        dvar = row['dvar']
        dvar = create_2d_map(sim, dvar,
                             filler=sim.data.MASK_LU_CODE)  # fill the missing values with sim.data.MASK_LU_CODE

        # Create output file name
        fname = f'{category}_{dvar_idx:02}_{desc}_{yr_cal}.tiff'

        # Write to GeoTiff
        write_gtiff(dvar, os.path.join(path, 'lucc_separate', fname))


def write_quantity(sim, yr_cal, path, yr_cal_sim_pre=None):
    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Append the yr_cal to timestamp as prefix
    timestamp = str(yr_cal) + '_' + timestamp

    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(sim.lumaps.keys()))

    # Get index of yr_cal in timeseries (e.g., if yr_cal is 2050 then yr_idx = 40)
    yr_idx = yr_cal - sim.data.YR_CAL_BASE

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)

    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    # Calculate data for quantity comparison between base year and target year
    if yr_cal > sim.data.YR_CAL_BASE:

        # Check if yr_cal_sim_pre meets the requirement
        assert yr_cal_sim_pre >= sim.data.YR_CAL_BASE and yr_cal_sim_pre < yr_cal, \
            f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {sim.data.YR_CAL_BASE} and < {yr_cal}"

        # Get commodity production quantities produced in base year
        if yr_cal_sim_pre == sim.data.YR_CAL_BASE:  # if base year is 2010
            prod_base = sim.data.PROD_2010_C
        else:  # if timeseries and base year is > 2010
            prod_base = np.array(sim.prod_data[yr_cal_sim_pre]['Production'])

        prod_targ = np.array(sim.prod_data[yr_cal]['Production'])  # Get commodity quantities produced in target year
        demands = sim.data.D_CY[yr_idx - 1]  # Get commodity demands for target year
        abs_diff = prod_targ - demands  # Diff between target year production and demands in absolute terms (i.e. tonnes etc)
        prop_diff = (prod_targ / demands) * 100  # Target year production as a proportion of demands (%)

        # Write to pandas dataframe
        df = pd.DataFrame()
        df['Commodity'] = sim.data.COMMODITIES
        df['Prod_base_year (tonnes, KL)'] = prod_base
        df['Prod_targ_year (tonnes, KL)'] = prod_targ
        df['Demand (tonnes, KL)'] = demands
        df['Abs_diff (tonnes, KL)'] = abs_diff
        df['Prop_diff (%)'] = prop_diff

        # Save files to disk
        df.to_csv(os.path.join(path, f'quantity_comparison_{timestamp}.csv'), index=False)

        # Write the production of each year to disk
        production_years = pd.DataFrame({yr: sim.prod_data[yr]['Production'] for yr in sim.prod_data.keys()})
        production_years.insert(0, 'Commodity', sim.bdata.COMMODITIES)
        production_years.to_csv(os.path.join(path, f'quantity_production_kt_{timestamp}.csv'), index=False)


def write_ag_revenue_cost(sim, yr_cal, path):
    """Calculate agricultural revenue. Takes a simulation object, a target calendar
       year (e.g., 2030), and an output path as input."""

    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Append the yr_cal to timestamp as prefix
    timestamp = str(yr_cal) + '_' + timestamp

    print('Writing agricultural revenue outputs to', path)

    # Convert calendar year to year index.
    yr_idx = yr_cal - sim.data.YR_CAL_BASE

    # Get the ag_dvar_mrj in the yr_cal
    ag_dvar_mrj = sim.ag_dvars[yr_cal]

    # Get agricultural revenue/cost for year in mrjs format. Note the s stands for sources:
    # E.g., Sources for crops only contains ['Revenue'],
    #    but sources for livestock includes ['Meat', 'Wool', 'Live Exports', 'Milk']
    ag_rev_df_rjms = ag_revenue.get_rev_matrices(sim.data, yr_idx, aggregate=False)
    ag_cost_df_rjms = ag_cost.get_cost_matrices(sim.data, yr_idx, aggregate=False)

    # Expand the original df with zero values to convert it to a **mrjs** array
    ag_rev_df_rjms = ag_rev_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_rev_df_rjms.columns.levels),
                                            fill_value=0)
    ag_rev_rjms = ag_rev_df_rjms.values.reshape(-1, *ag_rev_df_rjms.columns.levshape)

    ag_cost_df_rjms = ag_cost_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_cost_df_rjms.columns.levels),
                                              fill_value=0)
    ag_cost_rjms = ag_cost_df_rjms.values.reshape(-1, *ag_cost_df_rjms.columns.levshape)

    # Multiply the ag_dvar_mrj with the ag_rev_mrj to get the ag_rev_jm
    ag_rev_jms = np.einsum('mrj,rjms -> jms', ag_dvar_mrj, ag_rev_rjms)
    ag_cost_jms = np.einsum('mrj,rjms -> jms', ag_dvar_mrj, ag_cost_rjms)

    # Put the ag_rev_jms into a dataframe
    df_rev = pd.DataFrame(ag_rev_jms.reshape(ag_rev_jms.shape[0], -1),
                          columns=pd.MultiIndex.from_product(ag_rev_df_rjms.columns.levels[1:]),
                          index=ag_rev_df_rjms.columns.levels[0])

    df_cost = pd.DataFrame(ag_cost_jms.reshape(ag_cost_jms.shape[0], -1),
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


def write_ag_management_cost(sim, yr_cal, path):
    """Calculate agricultural mangement cost."""

    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Append the yr_cal to timestamp as prefix
    timestamp = str(yr_cal) + '_' + timestamp

    print('Writing non agricultural mangement cost outputs to', path)
    # Convert calendar year to year index.
    yr_idx = yr_cal - sim.bdata.YR_CAL_BASE
    ag_man_dvar = np.stack([np.einsum('mrj -> r', v) for _, v in sim.ag_man_dvars[
        yr_cal].items()]).T  # 0 Asparagopsis taxiformis,1 Precision Agriculture,2 Ecological Grazing

    cost_asparagopsis_lm = np.sum(ag_cost.get_asparagopsis_effect_c_mrj(sim.data, yr_idx),
                                  axis=2)  # Cost of ag_management in dry(0) and irr(1)
    cost_asparagopsis_irr = np.sum(ag_man_dvar[:, 0] * cost_asparagopsis_lm[1:] * sim.lmmaps[yr_cal])
    cost_asparagopsis_dry = np.sum(ag_man_dvar[:, 0] * cost_asparagopsis_lm[0:] * (1 - sim.lmmaps[yr_cal]))
    cost_asparagopsis = cost_asparagopsis_irr + cost_asparagopsis_dry

    cost_precision_agriculture_lm = np.sum(ag_cost.get_precision_agriculture_effect_c_mrj(sim.data, yr_idx),
                                           axis=2)  # Cost of ag_management in dry(0) and irr(1)
    cost_precision_agriculture_irr = np.sum(ag_man_dvar[:, 1] * cost_precision_agriculture_lm[1:] * sim.lmmaps[yr_cal])
    cost_precision_agriculture_dry = np.sum(
        ag_man_dvar[:, 1] * cost_precision_agriculture_lm[0:] * (1 - sim.lmmaps[yr_cal]))
    cost_precision_agriculture = cost_precision_agriculture_irr + cost_precision_agriculture_dry

    cost_ecological_grazing_lm = np.sum(ag_cost.get_ecological_grazing_effect_c_mrj(sim.data, yr_idx),
                                        axis=2)  # Cost of ag_management in dry(0) and irr(1)
    cost_ecological_grazing_lm = np.nan_to_num(cost_ecological_grazing_lm)
    cost_ecological_grazing_irr = np.sum(ag_man_dvar[:, 2] * cost_ecological_grazing_lm[1:] * sim.lmmaps[yr_cal])
    cost_ecological_grazing_dry = np.sum(ag_man_dvar[:, 2] * cost_ecological_grazing_lm[0:] * (1 - sim.lmmaps[yr_cal]))
    cost_ecological_grazing = cost_ecological_grazing_irr + cost_ecological_grazing_dry

    df_cost = pd.DataFrame({
        "asparagopsis": [cost_asparagopsis],
        "precision agriculture": [cost_precision_agriculture],
        "ecological grazing": [cost_ecological_grazing]
    })
    # Save to file
    df_cost.to_csv(os.path.join(path, f'cost_agricultural_mangement_{timestamp}.csv'))


def write_non_ag_cost(sim, yr_cal, path):
    """Calculate agricultural revenue. Takes a simulation object, a target calendar
       year (e.g., 2030), and an output path as input."""

    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Append the yr_cal to timestamp as prefix
    timestamp = str(yr_cal) + '_' + timestamp

    print('Writing non agricultural mangement cost outputs to', path)

    non_ag_dvar = sim.non_ag_dvars[yr_cal]

    cost_env_plantings = np.sum(non_ag_dvar[:, 0] * non_ag_cost.get_cost_env_plantings(sim.data))
    cost_rip_plantings = np.sum(non_ag_dvar[:, 1] * non_ag_cost.get_cost_rip_plantings(sim.data))
    cost_agroforestry = np.sum(non_ag_dvar[:, 2] * non_ag_cost.get_cost_agroforestry(sim.data))
    df_cost = pd.DataFrame({
        "Environmental Plantings": [cost_env_plantings],
        "Riparian Plantings": [cost_rip_plantings],
        "Agroforestry": [cost_agroforestry]
    })
    # Save to file
    df_cost.to_csv(os.path.join(path, f'cost_non_ag_{timestamp}.csv'))


def write_crosstab(sim, yr_cal, path, yr_cal_sim_pre=None):
    """Write out land-use and production data"""

    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Append the yr_cal to timestamp as prefix
    timestamp = str(yr_cal) + '_' + timestamp

    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(sim.lumaps.keys()))

    # Get index of yr_cal in timeseries (e.g., if yr_cal is 2050 then yr_idx = 40)
    yr_idx = yr_cal - sim.data.YR_CAL_BASE

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)

    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    # Only perform the calculation if the yr_cal is not the base year
    if yr_cal > sim.data.YR_CAL_BASE:

        # Check if yr_cal_sim_pre meets the requirement
        assert yr_cal_sim_pre >= sim.data.YR_CAL_BASE and yr_cal_sim_pre < yr_cal, \
            f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {sim.data.YR_CAL_BASE} and < {yr_cal}"

        print('Writing production outputs to', path)

        # LUS = ['Non-agricultural land'] + sim.data.AGRICULTURAL_LANDUSES + sim.data.NON_AGRICULTURAL_LANDUSES
        ctlu, swlu = lumap_crossmap(sim.lumaps[yr_cal_sim_pre]
                                    , sim.lumaps[yr_cal]
                                    , sim.data.AGRICULTURAL_LANDUSES
                                    , sim.data.NON_AGRICULTURAL_LANDUSES
                                    , sim.data.REAL_AREA)

        ctlm, swlm = lmmap_crossmap(sim.lmmaps[yr_cal_sim_pre]
                                    , sim.lmmaps[yr_cal]
                                    , sim.data.REAL_AREA)

        cthp, swhp = crossmap_irrstat(sim.lumaps[yr_cal_sim_pre]
                                      , sim.lmmaps[yr_cal_sim_pre]
                                      , sim.lumaps[yr_cal], sim.lmmaps[yr_cal]
                                      , sim.data.AGRICULTURAL_LANDUSES
                                      , sim.data.NON_AGRICULTURAL_LANDUSES
                                      , sim.data.REAL_AREA)

        # ctams = {}
        # swams = {}
        ctass = {}
        swass = {}
        for am in AG_MANAGEMENTS_TO_LAND_USES:
            # ctam, swam = ammap_crossmap(sim.ammaps[yr_pre][am], sim.ammaps[yr_cal][am], am)
            # ctams[am] = ctam
            # swams[am] = swam

            ctas, swas = crossmap_amstat(am
                                         , sim.lumaps[yr_cal_sim_pre]
                                         , sim.ammaps[yr_cal_sim_pre][am]
                                         , sim.lumaps[yr_cal]
                                         , sim.ammaps[yr_cal][am]
                                         , sim.data.AGRICULTURAL_LANDUSES
                                         , sim.data.NON_AGRICULTURAL_LANDUSES
                                         , sim.data.REAL_AREA)
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
            # ctams[am].to_csv(os.path.join(path, f'crosstab-{am_snake_case}-ammap_{timestamp}.csv'))
            # swams[am].to_csv(os.path.join(path, f'switches-{am_snake_case}-ammap_{timestamp}.csv'))

            ctass[am].to_csv(os.path.join(path, f'crosstab-amstat-{am_snake_case}_{timestamp}.csv'))
            swass[am].to_csv(os.path.join(path, f'switches-amstat-{am_snake_case}_{timestamp}.csv'))


def write_water(sim, yr_cal, path):
    """Calculate water use totals. Takes a simulation object, a numeric
       target calendar year (e.g., 2030), and an output path as input."""

    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Append the yr_cal to timestamp as prefix
    timestamp = str(yr_cal) + '_' + timestamp

    print('Writing water outputs to', path)

    # Convert calendar year to year index.
    yr_idx = yr_cal - sim.data.YR_CAL_BASE

    # Get water use for year in mrj format
    ag_w_mrj = ag_water.get_wreq_matrices(sim.data, yr_idx)
    non_ag_w_rk = non_ag_water.get_wreq_matrix(sim.data)
    ag_man_w_mrj = ag_water.get_agricultural_management_water_matrices(sim.data, ag_w_mrj, yr_idx)

    # Prepare a data frame.
    df = pd.DataFrame(columns=['REGION_ID'
        , 'REGION_NAME'
        , 'WATER_USE_LIMIT_ML'
        , 'TOT_WATER_REQ_ML'
        , 'ABS_DIFF_ML'
        , 'PROPORTION_%'])

    # Get 2010 water use limits used as constraints in model
    wuse_limits = ag_water.get_wuse_limits(sim.data)

    # Set up data for river regions or drainage divisions
    if settings.WATER_REGION_DEF == 'RR':
        region_limits = sim.data.RIVREG_LIMITS
        region_id = sim.data.RIVREG_ID
        # regions = settings.WATER_RIVREGS
        region_dict = sim.data.RIVREG_DICT

    elif settings.WATER_REGION_DEF == 'DD':
        region_limits = sim.data.DRAINDIV_LIMITS
        region_id = sim.data.DRAINDIV_ID
        # regions = settings.WATER_DRAINDIVS
        region_dict = sim.data.DRAINDIV_DICT

    else:
        print('Incorrect option for WATER_REGION_DEF in settings')

    # Loop through specified water regions
    df_water_seperate_dfs = []
    for i, region in enumerate(region_limits.keys()):

        # Get indices of cells in region
        ind = np.flatnonzero(region_id == region).astype(np.int32)

        # Calculate water requirements by agriculture for year and region.

        index_levels = ['Landuse Type', 'Landuse', 'Irrigation', 'Water Use (ML)']

        # Agricultural water use
        ag_mrj = ag_w_mrj[:, ind, :] * sim.ag_dvars[yr_cal][:, ind, :]
        ag_jm = np.einsum('mrj->jm', ag_mrj)

        ag_df = pd.DataFrame(ag_jm.reshape(-1).tolist(),
                             index=pd.MultiIndex.from_product([['Agricultural Landuse'],
                                                               sim.data.AGRICULTURAL_LANDUSES,
                                                               sim.data.LANDMANS
                                                               ])).reset_index()

        ag_df.columns = index_levels

        # Non-agricultural water use
        non_ag_rk = non_ag_w_rk[ind, :] * sim.non_ag_dvars[yr_cal][ind, :]  # Non-agricultural contribution
        non_ag_k = np.einsum('rk->k', non_ag_rk)  # Sum over cells

        non_ag_df = pd.DataFrame(non_ag_k,
                                 index=pd.MultiIndex.from_product([['Non-agricultural Landuse'],
                                                                   sim.data.NON_AGRICULTURAL_LANDUSES,
                                                                   ['dry']  # non-agricultural land is always dry
                                                                   ])).reset_index()

        non_ag_df.columns = index_levels

        # Agricultural management water use
        AM_dfs = []
        for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():  # Agricultural managements contribution

            am_j = np.array([sim.data.DESC2AGLU[lu] for lu in am_lus])

            # Water requirements for each agricultural management in array format
            am_mrj = ag_man_w_mrj[am][:, ind, :] \
                     * sim.ag_man_dvars[yr_cal][am][:, ind[:, np.newaxis], am_j]

            am_jm = np.einsum('mrj->jm', am_mrj)

            # Water requirements for each agricultural management in long dataframe format
            df_am = pd.DataFrame(am_jm.reshape(-1).tolist(),
                                 index=pd.MultiIndex.from_product([['Agricultural Management'],
                                                                   am_lus,
                                                                   sim.data.LANDMANS
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
        wul = wuse_limits[i][1]
        wreq_reg = df_region['Water Use (ML)'].sum()

        # Calculate absolute and proportional difference between water use target and actual water use
        abs_diff = wreq_reg - wul
        if wul > 0:
            prop_diff = (wreq_reg / wul) * 100
        else:
            prop_diff = np.nan

        # Add to dataframe
        df.loc[i] = (region
                     , region_dict[region]
                     , wul
                     , wreq_reg
                     , abs_diff
                     , prop_diff)

    # Write to CSV with 2 DP
    df.to_csv(os.path.join(path, f'water_demand_vs_use_{timestamp}.csv')
              , index=False
              , float_format='{:0,.2f}'.format)

    # Write the separate water use to CSV
    df_water_seperate = pd.concat(df_water_seperate_dfs)
    df_water_seperate.to_csv(os.path.join(path, f'water_demand_vs_use_separate_{timestamp}.csv')
                             , index=False)


def write_ghg(sim, yr_cal, path):
    """Calculate total GHG emissions. Takes a simulation object, a target calendar
       year (e.g., 2030), and an output path as input."""

    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Append the yr_cal to timestamp as prefix
    timestamp = str(yr_cal) + '_' + timestamp

    print('Writing GHG outputs to', path)

    yr_idx = yr_cal - sim.data.YR_CAL_BASE

    # Prepare a data frame.
    df = pd.DataFrame(columns=['GHG_EMISSIONS_LIMIT_TCO2e'
        , 'GHG_EMISSIONS_TCO2e'])

    # Get GHG emissions limits used as constraints in model
    ghg_limits = ag_ghg.get_ghg_limits(sim.data, yr_cal)

    # Get GHG emissions from model
    if yr_cal >= sim.data.YR_CAL_BASE + 1:
        ghg_emissions = sim.prod_data[yr_cal]['GHG Emissions']
    else:
        ghg_emissions = (ag_ghg.get_ghg_matrices(sim.data, yr_idx, aggregate=True) * sim.ag_dvars[
            sim.data.YR_CAL_BASE]).sum()

    # Add to dataframe
    df.loc[0] = ("{:,.0f}".format(ghg_limits), "{:,.0f}".format(ghg_emissions))

    # Save to file
    df.to_csv(os.path.join(path, f'GHG_emissions_{timestamp}.csv'), index=False)


def write_biodiversity(sim, yr_cal, path):
    """
    Write biodiversity info for a given year ('yr_cal'), simulation ('sim')
    and output path ('path').
    """
    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Append the yr_cal to timestamp as prefix
    timestamp = str(yr_cal) + '_' + timestamp

    print('Writing biodiversity outputs to', path)

    yr_idx = yr_cal - sim.data.YR_CAL_BASE

    df = pd.DataFrame(columns=['Biodiversity score limit'
        , f'Solve biodiversity score ({yr_cal})'])

    # Get limits used as constraints in model
    biodiv_limit = ag_biodiversity.get_biodiversity_limits(sim.data, yr_cal)

    # Get GHG emissions from model
    if yr_cal >= sim.data.YR_CAL_BASE + 1:
        biodiv_score = sim.prod_data[yr_cal]['Biodiversity']
    else:
        # Limits are based on the 2010 biodiversity scores, with the base year
        # biodiversity score equal to the
        biodiv_score = ag_biodiversity.get_base_year_biodiversity_score(sim.data)

    # Add to dataframe
    df.loc[0] = ("{:,.0f}".format(biodiv_limit), "{:,.0f}".format(biodiv_score))

    # Save to file
    df.to_csv(os.path.join(path, f'biodiversity_{timestamp}.csv'), index=False)


def write_ghg_separate(sim, yr_cal, path):
    # Notation explain
    #       r: The pixel/row dimens.
    #       j: The landuse
    #       k: The non-agricultural landuse
    #       m: Land management [dry, irri]
    #       s: The sources of origin. Such as [Chemical_CO2, Electric_CO2] for {Agricultural landuse},

    # Get the timestamp so each CSV in the timeseries mode has a unique name
    timestamp = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')

    # Append the yr_cal to timestamp as prefix
    timestamp = str(yr_cal) + '_' + timestamp

    # Convert calendar year to year index.
    yr_idx = yr_cal - sim.data.YR_CAL_BASE

    # Get the landuse descriptions for each validate cell (i.e., 0 -> Apples)
    lu_desc_map = {**sim.data.AGLU2DESC, **sim.data.NONAGLU2DESC}
    lu_desc = [lu_desc_map[x] for x in sim.lumaps[yr_cal]]

    # -------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural landuse #
    # -------------------------------------------------------#

    # 1-1) Get the ghg_df
    ag_g_df = ag_ghg.get_ghg_matrices(sim.data, yr_idx, aggregate=False)

    # Expand the original df with zero values to convert it to a a **mrj** array
    origin_type = ag_g_df.columns.levels[0]
    ghg_sources = ag_g_df.columns.levels[1]
    lm_type = sim.data.LANDMANS
    lu_type = sim.data.AGRICULTURAL_LANDUSES

    df_col_product = list(product(*[origin_type, ghg_sources, lm_type, lu_type]))

    ag_g_df = ag_g_df.reindex(columns=df_col_product, fill_value=0)

    # Reshape the df to the shape of [rosmj]
    #       r: row, i.e, each valid pixel
    #       o: origin, i.e, each origin [crop, lvstk, unallocated]
    #       s: GHG source [chemical fertilizer, electricity, ...]
    #       m: land management: [dry,irr]
    #       j: land use [Apples, Beef, ...]
    ag_g_df = ag_g_df.to_numpy(na_value=0).reshape(ag_g_df.shape[0], *ag_g_df.columns.levshape)

    # 1-2) Get the ag_g_mrjs, which will be used to compute GHG_agricultural_landuse
    ag_g_mrjs = np.einsum('rosmj -> mrjs', ag_g_df)  # mrjs

    # 1-3) Get the ag_g_mrj, which will be used to compute GHG_agricultural_management
    ag_g_mrj = np.einsum('mrjs -> mrj', ag_g_mrjs)  # mrj

    # Use einsum to do the multiplication,
    # easy to understand, don't need to worry too much about the dimensionality
    ag_dvar_mrj = sim.ag_dvars[yr_cal]  # mrj
    GHG_emission_separate = np.einsum('mrjs,mrj -> rms', ag_g_mrjs, ag_dvar_mrj)  # rms

    # Summarize the array as a df
    GHG_emission_separate_summary = tools.summarize_ghg_separate_df(
        GHG_emission_separate,
        (['Agricultural Landuse'], lm_type, ghg_sources),
        lu_desc
    )

    # Change "KG_HA/HEAD" to "TCO2E"
    column_rename = [(i[0], i[1], i[2].replace('CO2E_KG_HA', 'TCO2E')) for i in GHG_emission_separate_summary.columns]
    column_rename = [(i[0], i[1], i[2].replace('CO2E_KG_HEAD', 'TCO2E')) for i in column_rename]
    GHG_emission_separate_summary.columns = pd.MultiIndex.from_tuples(column_rename)

    # Save table to disk
    GHG_emission_separate_summary.to_csv(
        os.path.join(path, f'GHG_emissions_separate_agricultural_landuse_{timestamp}.csv'))

    # -----------------------------------------------------------#
    # Get greenhouse gas emissions from non-agricultural landuse #
    # -----------------------------------------------------------#

    # Get the non_ag GHG reduction
    non_ag_g_rk = non_ag_ghg.get_ghg_matrix(sim.data)

    # Multiply with decision variable to get the GHG in yr_cal
    non_ag_g_rk = non_ag_g_rk * sim.non_ag_dvars[yr_cal]

    # get the non_ag GHG reduction on dry/irri land
    non_ag_g_rk_dry = np.einsum('rk,r -> rk', non_ag_g_rk, (sim.data.LMMAP != 1))
    non_ag_g_rk_irri = np.einsum('rk,r -> rk', non_ag_g_rk, (sim.data.LMMAP == 1))
    non_ag_g_mrk = np.array([non_ag_g_rk_dry, non_ag_g_rk_irri])  # mrk
    non_ag_g_rmk = np.swapaxes(non_ag_g_mrk, 0, 1)  # rmk

    # Summarize the array as a df
    non_ag_g_rk_summary = tools.summarize_ghg_separate_df(
        non_ag_g_rmk,
        (['Non_Agricultural Landuse'], sim.data.LANDMANS,
         [f'TCO2E_{non_ag}' for non_ag in sim.data.NON_AGRICULTURAL_LANDUSES]),
        lu_desc
    )

    # Save table to disk
    non_ag_g_rk_summary.to_csv(os.path.join(path, f'GHG_emissions_separate_no_ag_reduction_{timestamp}.csv'))

    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from landuse transformation penalties #
    # -------------------------------------------------------------------#

    # Get the lucc transition penalty data (mrj) between target and base (2010) year
    ghg_t_2010 = ag_ghg.get_ghg_transition_penalties(sim.data, sim.lumaps[2010])  # mrj

    # Get the GHG emissions from lucc-convertion compared to the base year (2010)
    ghg_t_separate = np.einsum('mrj,mrj -> rmj', sim.ag_dvars[yr_cal], ghg_t_2010)  # rmj

    # Summarize the array as a df
    ghg_t_separate_summary = tools.summarize_ghg_separate_df(ghg_t_separate, (['Transition Penalty'],
                                                                              sim.data.LANDMANS,
                                                                              [f"TCO2E_{i}" for i in
                                                                               sim.data.AGRICULTURAL_LANDUSES]),
                                                             lu_desc)

    # Save table to disk
    ghg_t_separate_summary.to_csv(os.path.join(path, f'GHG_emissions_separate_transition_penalty_{timestamp}.csv'))

    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural management          #
    # -------------------------------------------------------------------#

    # 3) Get the ag_man_g_mrj
    ag_man_g_mrj = ag_ghg.get_agricultural_management_ghg_matrices(sim.data, ag_g_mrj, yr_idx)

    ag_ghg_arrays = []
    for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():
        # Get the lucc_code for this the agricultural management in this loop
        am_j = np.array([sim.data.DESC2AGLU[lu] for lu in am_lus])

        # Get the GHG emission from agricultural management, then reshape it to starte with row (r) dimension
        am_ghg_mrj = ag_man_g_mrj[am] * sim.ag_man_dvars[yr_cal][am][:, :, am_j]  # mrj
        am_ghg_rm = np.einsum('mrj -> rm', am_ghg_mrj)  # rm

        # Summarize the df by calculating the total value of each column
        ag_ghg_arrays.append(am_ghg_rm)

    # Concat all summary tables
    ag_ghg_summary = np.stack(ag_ghg_arrays)  # srm
    ag_ghg_summary = np.einsum('srm -> rms', ag_ghg_summary)  # rms

    # Summarize the array as a df
    ag_ghg_summary_df = tools.summarize_ghg_separate_df(ag_ghg_summary, (['Agricultural Management']
                                                                         , sim.data.LANDMANS
                                                                         , [f"TCO2E_{i}" for i in
                                                                            AG_MANAGEMENTS_TO_LAND_USES.keys()]),
                                                        lu_desc)

    # Save table to disk
    ag_ghg_summary_df.to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_management_{timestamp}.csv'))




