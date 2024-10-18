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
Pure helper functions and other tools.
"""



import re
import sys
import time
import os.path
import traceback
import functools
import shutil

import pandas as pd
import numpy as np
import numpy_financial as npf
import luto.settings as settings

from typing import Tuple
from datetime import datetime
from joblib import Parallel, delayed

from luto.tools.report.create_html import data2html
from luto.tools.report.create_report_data import save_report_data
from luto.tools.report.create_static_maps import TIF2MAP
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.tools.report.data_tools import get_all_files


def amortise(cost, rate=settings.DISCOUNT_RATE, horizon=settings.AMORTISATION_PERIOD):
    """Return NPV of future `cost` amortised to annual value at discount `rate` over `horizon` years."""
    if settings.AMORTISE_UPFRONT_COSTS: return -1 * npf.pmt(rate, horizon, pv=cost, fv=0, when='begin')
    else: return cost


def move_html(path_from, path_to):
    if os.path.exists(path_from):
        shutil.move(path_from, path_to)


def report_on_path(path:str, remake_map=True, remake_csv=True):
    
    if remake_csv:
        if os.path.exists(f"{path}/DATA_REPORT/data"):
            [os.remove(f"{path}/DATA_REPORT/data/{i}") for i in os.listdir(f"{path}/DATA_REPORT/data") if os.path.isfile(f"{path}/DATA_REPORT/data/{i}")]
        save_report_data(path)

    if remake_map:
        map_dir = f"{path}/DATA_REPORT/data/Map_data"
        if os.path.exists(map_dir):
            [os.remove(f'{map_dir}/{i}') for i in os.listdir(map_dir) if len(os.listdir(map_dir)) > 0]
        else:
            os.mkdir(map_dir)
            
        # Make the maps
        TIF2MAP(path)
        map_html = get_all_files(path).query('base_ext == ".html" and year_types != "begin_end_year"')
        tasks = [delayed(move_html)(row['path'], map_dir) for _,row in map_html.iterrows()]
        worker = min(settings.WRITE_THREADS, len(tasks)) if len(tasks) > 0 else 1
        Parallel(n_jobs=worker)(tasks)
            
    # Copy the html files to the the report index HTML
    data2html(path)


def lumap2ag_l_mrj(lumap, lmmap):
    """
    Return land-use maps in decision-variable (X_mrj) format.
    Where 'm' is land mgt, 'r' is cell, and 'j' is agricultural land-use.

    Cells used for non-agricultural land uses will have value 0 for all agricultural
    land uses, i.e. all r.
    """
    # Set up a container array of shape m, r, j.
    x_mrj = np.zeros((2, lumap.shape[0], 28), dtype=bool)   # TODO - remove 2

    # Populate the 3D land-use, land mgt mask.
    for j in range(28):
        # One boolean map for each land use.
        jmap = np.where(lumap == j, True, False).astype(bool)
        # Keep only dryland version.
        x_mrj[0, :, j] = np.where(lmmap == False, jmap, False)
        # Keep only irrigated version.
        x_mrj[1, :, j] = np.where(lmmap == True, jmap, False)

    return x_mrj.astype(bool)


def lumap2non_ag_l_mk(lumap, num_non_ag_land_uses: int):
    """
    Convert the land-use map to a decision variable X_rk, where 'r' indexes cell and
    'k' indexes non-agricultural land use.

    Cells used for agricultural purposes have value 0 for all k.
    """
    base_code = settings.NON_AGRICULTURAL_LU_BASE_CODE
    non_ag_lu_codes = list(range(base_code, base_code + num_non_ag_land_uses))

    # Set up a container array of shape r, k.
    x_rk = np.zeros((lumap.shape[0], num_non_ag_land_uses), dtype=bool)

    for i,k in enumerate(non_ag_lu_codes):
        kmap = np.where(lumap == k, True, False)
        x_rk[:, i] = kmap

    return x_rk.astype(bool)


def get_base_am_vars(ncells, ncms, n_ag_lus):
    """
    Get the 2010 agricultural management option vars.
    It is assumed that no agricultural management options were used in 2010, 
    so get zero arrays in the correct format.
    """
    return {
        am: np.zeros((ncms, ncells, n_ag_lus))
        for am in AG_MANAGEMENTS_TO_LAND_USES
    }


def get_ag_and_non_ag_cells(lumap) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the index of cells based on whether that cell is used for agricultural
    land, given the lumap.

    Returns
    -------
    ( np.ndarray, np.ndarray )
        Two numpy arrays containing the split cell index.
    """
    non_ag_base = settings.NON_AGRICULTURAL_LU_BASE_CODE
    all_cells = np.array(range(lumap.shape[0]))

    # get all agricultural and non agricultural cells
    non_agricultural_cells = np.nonzero(lumap >= non_ag_base)[0]
    agricultural_cells = np.nonzero(
        ~np.isin(all_cells, non_agricultural_cells)
    )[0]

    return agricultural_cells, non_agricultural_cells


def get_env_plantings_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used for environmental plantings
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 0)[0]


def get_riparian_plantings_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used for riparian plantings
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 1)[0]


def get_sheep_agroforestry_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used for riparian plantings
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 2)[0]


def get_beef_agroforestry_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used for riparian plantings
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 3)[0]


def get_agroforestry_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used that currently use agroforestry (either sheep or beef)
    """
    agroforestry_lus = [settings.NON_AGRICULTURAL_LU_BASE_CODE + 2, settings.NON_AGRICULTURAL_LU_BASE_CODE + 3]
    return np.nonzero(np.isin(lumap, agroforestry_lus))[0]


def get_carbon_plantings_block_cells(lumap) -> np.ndarray:
    """
    Get an array with all cells being used for carbon plantings (block)
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 4)[0]


def get_sheep_carbon_plantings_belt_cells(lumap) -> np.ndarray:
    """
    Get an array with all cells being used for sheep carbon plantings (belt)
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 5)[0]


def get_beef_carbon_plantings_belt_cells(lumap) -> np.ndarray:
    """
    Get an array with all cells being used for beef carbon plantings (belt)
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 6)[0]


def get_carbon_plantings_belt_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used that currently use carbon plantings belt (either sheep or beef)
    """

    cp_belt_lus = [settings.NON_AGRICULTURAL_LU_BASE_CODE + 5, settings.NON_AGRICULTURAL_LU_BASE_CODE + 6]
    return np.nonzero(np.isin(lumap, cp_belt_lus))[0]


def get_beccs_cells(lumap) -> np.ndarray:
    """
    Get an array with all cells being used for carbon plantings (block)
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 7)[0]


def get_ag_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for agricultural natural land uses.
    """
    return np.nonzero(np.isin(lumap, data.LU_NATURAL))[0]


def get_non_ag_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for non-agricultural natural land uses.
    """
    return np.nonzero(np.isin(lumap, data.NON_AG_LU_NATURAL))[0]


def get_ag_and_non_ag_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for natural land uses, both agricultural and non-agricultural.
    """
    return np.nonzero(np.isin(lumap, data.LU_NATURAL + data.NON_AG_LU_NATURAL))[0]


def get_ag_cells(lumap) -> np.ndarray:
    """
    Get an array containing the index of all non-agricultural cells
    """
    return np.nonzero(lumap < settings.NON_AGRICULTURAL_LU_BASE_CODE)[0]


def get_non_ag_cells(lumap) -> np.ndarray:
    """
    Get an array containing the index of all non-agricultural cells
    """
    return np.nonzero(lumap >= settings.NON_AGRICULTURAL_LU_BASE_CODE)[0]


def timethis(function, *args, **kwargs):
    """Generic wrapper to time functions."""

    # Start the wall clock.
    start = time.time()
    start_time = time.localtime()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", start_time)

    # Call the function.
    return_value = function(*args, **kwargs)

    # Stop the wall clock.
    stop = time.time()
    stop_time = time.localtime()
    stop_time_str = time.strftime("%Y-%m-%d %H:%M:%S", stop_time)

    print()
    print(f"Start time: {start_time_str}")
    print(f"Stop time: {stop_time_str}")
    print("Elapsed time: %d seconds." % (stop - start))

    return return_value


def mergeorderly(dict1, dict2):
    """Return merged dictionary with keys in alphabetic order."""
    list1 = list(dict1.keys())
    list2 = list(dict2.keys())
    lst = sorted(list1 + list2)
    return {key: dict1[key] if key in list1 else dict2[key] for key in lst}


def crosstabulate(before, after, labels):
    """Return a cross-tabulation matrix as a Pandas DataFrame."""

    # The base DataFrame
    ct = pd.crosstab(before, after, margins=True)

    # Replace the numbered row labels with corresponding entries in `labels`.
    rows = [labels[int(i)] for i in ct.index if i != 'All']
    # The sum row was excluded, now add it back.
    rows.append('All')

    # Replace the numbered column labels with corresponding entries in `labels`.
    cols = [labels[int(i)] for i in ct.columns if i != 'All']
    # The sum column was excluded, now add it back.
    cols.append('All')

    # Replace the row and column labels.
    ct.index = rows
    ct.columns = cols

    return ct


def ctabnpystocsv(before, after, labels):
    """Write cross-tabulation of `before` and `after` from .npys to .csv."""

    # Load the .npy arrays.
    pre = np.load(before)
    post = np.load(after)

    # Get the file names as strings without extension.
    prename = os.path.splitext(os.path.basename(before))[0]
    postname = os.path.splitext(os.path.basename(after))[0]
    dirname = os.path.dirname(before)

    # Produce the cross tabulation and write the file to dir of first file.
    ct = crosstabulate(pre, post, labels)
    ct.to_csv(os.path.join(prename + str(2) + postname + ".csv"))


def inspect(lumap, highpos, d_j, q_rj, c_rj, landuses):

    # Prepare the pandas.DataFrame.
    columns = ["Pre Cells [#]", "Pre Costs [AUD]", "Pre Deviation [units]", "Pre Deviation [%]", "Post Cells [#]", "Post Costs [AUD]", "Post Deviation [units]", "Post Deviation [%]",
               "Delta Total Cells [#]", "Delta Total Cells [%]", "Delta Costs [AUD]", "Delta Costs [%]", "Delta Moved Cells [#]", "Delta Deviation [units]", "Delta Deviation [%]"]
    index = landuses
    df = pd.DataFrame(index=index, columns=columns)

    precost = 0
    postcost = 0

    # Reduced land-use list with half of the land-uses.
    lus = [lu for lu in landuses if '_dry' in lu]

    for j, lu in enumerate(lus):
        k = 2*j
        prewhere_dry = np.where(lumap == k, 1, 0)
        prewhere_irr = np.where(lumap == k+1, 1, 0)
        precount_dry = prewhere_dry.sum()
        precount_irr = prewhere_irr.sum()
        prequantity_dry = prewhere_dry @ q_rj.T[k]
        prequantity_irr = prewhere_irr @ q_rj.T[k+1]

        precost_dry = prewhere_dry @ c_rj.T[k]
        precost_irr = prewhere_irr @ c_rj.T[k+1]

        postwhere_dry = np.where(highpos == k, 1, 0)
        postwhere_irr = np.where(highpos == k+1, 1, 0)
        postcount_dry = postwhere_dry.sum()
        postcount_irr = postwhere_irr.sum()
        postquantity_dry = postwhere_dry @ q_rj.T[k]
        postquantity_irr = postwhere_irr @ q_rj.T[k+1]

        postcost_dry = postwhere_dry @ c_rj.T[k]
        postcost_irr = postwhere_irr @ c_rj.T[k+1]

        predeviation = prequantity_dry + prequantity_irr - d_j[j]
        predevfrac = predeviation / d_j[j]

        postdeviation = postquantity_dry + postquantity_irr - d_j[j]
        postdevfrac = postdeviation / d_j[j]

        df.loc[lu] = [precount_dry, precost_dry, predeviation, 100 * predevfrac, postcount_dry, postcost_dry, postdeviation, 100 * postdevfrac, postcount_dry - precount_dry, 100 * (postcount_dry - precount_dry) / precount_dry, postcost_dry - precost_dry, 100 * (postcost_dry - precost_dry) / precost_dry, np.sum(postwhere_dry - (prewhere_dry*postwhere_dry)), np.abs(postdeviation) - np.abs(predeviation), 100 * (np.abs(postdeviation) - np.abs(predeviation)
                                                                                                                                                                                                                                                                                                                                                                                                                            / predeviation)]

        lu_irr = lu.replace('_dry', '_irr')
        df.loc[lu_irr] = [precount_irr, precost_irr, predeviation, 100 * predevfrac, postcount_irr, postcost_irr, postdeviation, 100 * postdevfrac, postcount_irr - precount_irr, 100 * (postcount_irr - precount_irr) / precount_irr, postcost_irr - precost_irr, 100 * (postcost_irr - precost_irr) / precost_irr, np.sum(postwhere_irr - (prewhere_irr*postwhere_irr)), np.abs(postdeviation) - np.abs(predeviation), 100 * (np.abs(postdeviation) - np.abs(predeviation)
                                                                                                                                                                                                                                                                                                                                                                                                                                / predeviation)]

        precost += prewhere_dry @ c_rj.T[k] + prewhere_irr @ c_rj.T[k+1]
        postcost += postwhere_dry @ c_rj.T[k] + postwhere_irr @ c_rj.T[k+1]

    df.loc['total'] = [df[col].sum() for col in df.columns]

    return df


def get_water_delta_matrix(w_mrj, l_mrj, data, yr_idx):
    """
    Gets the water delta matrix ($/cell) that applies the cost of installing/removing irrigation to
    base transition costs. Includes the costs of water license fees.

    Parameters:
    - w_mrj (numpy.ndarray, <unit:ML/cell>): Water requirements matrix for target year.
    - l_mrj (numpy.ndarray): Land-use and land management matrix for the base_year.
    - data (object): Data object containing necessary information.

    Returns:
    - w_delta_mrj (numpy.ndarray, <unit:$/cell>).
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    
    # Get water requirements from current agriculture, converting water requirements for LVSTK from ML per head to ML per cell (inc. REAL_AREA).
    # Sum total water requirements of current land-use and land management
    w_r = (w_mrj * l_mrj).sum(axis=0).sum(axis=1)

    # Net water requirements calculated as the diff in water requirements between current land-use and all other land-uses j.
    w_net_mrj = w_mrj - w_r[:, np.newaxis]

    # Water license cost calculated as net water requirements (ML/cell) x licence price ($/ML).
    w_delta_mrj = w_net_mrj * data.WATER_LICENCE_PRICE[:, np.newaxis] * data.WATER_LICENSE_COST_MULTS[yr_cal] * settings.INCLUDE_WATER_LICENSE_COSTS

    # When land-use changes from dryland to irrigated add <settings.NEW_IRRIG_COST> per hectare for establishing irrigation infrastructure
    new_irrig = (
        settings.NEW_IRRIG_COST
        * data.IRRIG_COST_MULTS[yr_cal]
        * data.REAL_AREA[:, np.newaxis]  # <unit:$/cell>
    )
    w_delta_mrj[1] = np.where(l_mrj[0], w_delta_mrj[1] + new_irrig, w_delta_mrj[1])

    # When land-use changes from irrigated to dryland add <settings.REMOVE_IRRIG_COST> per hectare for removing irrigation infrastructure
    remove_irrig = (
        settings.REMOVE_IRRIG_COST
        * data.IRRIG_COST_MULTS[yr_cal]
        * data.REAL_AREA[:, np.newaxis]  # <unit:$/cell>
    )
    w_delta_mrj[0] = np.where(l_mrj[1], w_delta_mrj[0] + remove_irrig, w_delta_mrj[0])
    

    # Amortise upfront costs to annualised costs
    w_delta_mrj = amortise(w_delta_mrj)
    return w_delta_mrj  # <unit:$/cell>


def am_name_snake_case(am_name):
    """Get snake_case version of the AM name"""
    return am_name.lower().replace(' ', '_')


def get_exclusions_for_excluding_all_natural_cells(data, lumap) -> np.ndarray:
    """
    A number of non-agricultural land uses can only be applied to cells that
    don't already utilise a natural land use. This function gets the exclusion
    matrix for all such non-ag land uses, returning an array valued 0 at the 
    indices of cells that use natural land uses, and 1 everywhere else.

    Parameters:
    - data: The data object containing information about the cells.
    - lumap: The land use map.

    Returns:
    - exclude: An array of shape (NCELLS,) with values 0 at the indices of cells
               that use natural land uses, and 1 everywhere else.
    """
    exclude = np.ones(data.NCELLS)

    natural_lu_cells = get_ag_and_non_ag_natural_lu_cells(data, lumap)
    exclude[natural_lu_cells] = 0

    return exclude


def get_exclusions_agroforestry_base(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much agroforestry can possibly 
    be done at each cell.

    Parameters:
    - data: The data object containing information about the landscape.
    - lumap: The land use map.

    Returns:
    - exclude: A 1-D array.
    """
    exclude = (np.ones(data.NCELLS) * settings.AF_PROPORTION).astype(np.float32)

    # Ensure cells being used for agroforestry may retain that LU
    exclude[get_agroforestry_cells(lumap)] = settings.AF_PROPORTION

    return exclude


def get_exclusions_carbon_plantings_belt_base(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much carbon plantings (belt) can possibly 
    be done at each cell.

    Parameters:
    - data: The data object containing information about the cells.
    - lumap: The land use map.

    Returns:
    - exclude: A 1-D array
    """
    exclude = (np.ones(data.NCELLS) * settings.CP_BELT_PROPORTION).astype(np.float32)

    # Ensure cells being used for carbon plantings (belt) may retain that LU
    exclude[get_carbon_plantings_belt_cells(lumap)] = settings.CP_BELT_PROPORTION

    return exclude


def get_sheep_code(data):
    """
    Get the land use code (j) for 'Sheep - modified land'
    """
    return data.DESC2AGLU['Sheep - modified land']


def get_beef_code(data):
    """
    Get the land use code (j) for 'Beef - modified land'
    """
    return data.DESC2AGLU['Beef - modified land']


# function to create mapping table between lu_desc and dvar index
def map_desc_to_dvar_index(category: str,
                           desc2idx: dict,
                           dvar_arr: np.ndarray):
    '''Input:
        category: str, the category of the dvar, e.g., 'Agriculture/Non-Agriculture',
        desc2idx: dict, the mapping between lu_desc and dvar index, e.g., {'Apples': 0 ...},
        dvar_arr: np.ndarray, the dvar array with shape (r,{j|k}), where r is the number of pixels,
                  and {j|k} is the dimension of ag-landuses or non-ag-landuses.

    Return:
        pd.DataFrame, with columns of ['Category','lu_desc','dvar_idx','dvar'].'''

    df = pd.DataFrame({'Category': category,
                       'lu_desc': desc2idx.keys(),
                       'dvar_idx': desc2idx.values()})

    df['dvar'] = [dvar_arr[:, j] for j in df['dvar_idx']]

    return df.reindex(columns=['Category', 'lu_desc', 'dvar_idx', 'dvar'])


def get_out_resfactor(dvar_path:str):
    # Get the number of pixels in the output decision variable
    dvar = np.load(dvar_path)
    dvar_size = dvar.shape[1]

    # Get the number of pixels in full resolution LUMASK
    full_lumap = pd.read_hdf(os.path.join(settings.INPUT_DIR, "lumap.h5")).to_numpy() 
    full_size = (full_lumap > 0).sum()

    # Return the resfactor
    return round((full_size/dvar_size)**0.5)


def read_dvars(yr, df_yr: pd.DataFrame) -> tuple:
    '''Read the dvars and maps from the dataframe containing the paths to the files in a given year.'''
    
    # Agricultrual management dvars/maps need to be read separately as dictionary
    am_dvars = {}
    ammaps ={}
    for am in AG_MANAGEMENTS_TO_LAND_USES:
        am_fname =  am.split(' ')
        am_fname = '_'.join([s.lower() for s in am_fname])
        am_fname = f"ag_man_X_mrj_{am_fname}"
        
        am_dvar = np.load(df_yr.query('base_name == @am_fname')['path'].iloc[0])
        ammap = am_dvar.sum(0).sum(1)               # mrj to r
        ammap = ammap >= settings.AGRICULTURAL_MANAGEMENT_USE_THRESHOLD
        
        am_dvars[am] = am_dvar
        ammaps[am] = ammap
    
    return int(yr),(np.load(df_yr.query('base_name == "lumap"')['path'].iloc[0]),
                    np.load(df_yr.query('base_name == "lmmap"')['path'].iloc[0]),
                    ammaps,
                    np.load(df_yr.query('base_name == "ag_X_mrj"')['path'].iloc[0]),
                    np.load(df_yr.query('base_name == "non_ag_X_rk"')['path'].iloc[0]),
                    am_dvars)

    
def calc_water(
    data, 
    ind:np.ndarray, 
    ag_w_mrj:np.ndarray, 
    non_ag_w_rk:np.ndarray, 
    ag_man_w_mrj:np.ndarray, 
    ag_dvar:np.ndarray, 
    non_ag_dvar:np.ndarray, 
    am_dvar:np.ndarray):
    
    '''
    Note:
        This function is only used for the `write_water` in the `luto.tools.write` module.
        Calculate water yields for year given the index. 
    
    Return:
    - pd.DataFrame, the water yields for year and region.
    '''
    
    # Calculate water yields for year and region.
    index_levels = ['Landuse Type', 'Landuse', 'Water_supply',  'Water Net Yield (ML)']

    # Agricultural contribution
    ag_mrj = ag_w_mrj[:, ind, :] * ag_dvar[:, ind, :]   
    ag_jm = np.einsum('mrj->jm', ag_mrj)                             
    ag_df = pd.DataFrame(
        ag_jm.reshape(-1).tolist(),
        index=pd.MultiIndex.from_product(
            [['Agricultural Landuse'],
                data.AGRICULTURAL_LANDUSES,
                data.LANDMANS])).reset_index()
    ag_df.columns = index_levels

    # Non-agricultural contribution
    non_ag_rk = non_ag_w_rk[ind, :] * non_ag_dvar[ind, :]  # Non-agricultural contribution
    non_ag_k = np.einsum('rk->k', non_ag_rk)                             # Sum over cells
    non_ag_df = pd.DataFrame(
        non_ag_k, 
        index= pd.MultiIndex.from_product([
                ['Non-agricultural Landuse'],
                settings.NON_AG_LAND_USES.keys() ,
                ['dry']  # non-agricultural land is always dry
    ])).reset_index()
    non_ag_df.columns = index_levels

    # Agricultural managements contribution
    AM_dfs = []
    for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():  # Agricultural managements contribution
        am_j = np.array([data.DESC2AGLU[lu] for lu in am_lus])
        am_mrj = ag_man_w_mrj[am][:, ind, :] * am_dvar[am][:, ind, :][:, :, am_j] 
        am_jm = np.einsum('mrj->jm', am_mrj)
        # water yields for each agricultural management in long dataframe format
        df_am = pd.DataFrame(
            am_jm.reshape(-1).tolist(),
            index=pd.MultiIndex.from_product([
                ['Agricultural Management'],
                am_lus,
                data.LANDMANS
                ])).reset_index()
        df_am.columns = index_levels
        AM_dfs.append(df_am)
    AM_df = pd.concat(AM_dfs)
    
    return pd.concat([ag_df, non_ag_df, AM_df])


class LogToFile:
    def __init__(self, log_path, mode:str='w'):
        self.log_path_stdout = f"{log_path}_stdout.log"
        self.log_path_stderr = f"{log_path}_stderr.log"
        self.mode = mode

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Open files for writing here, ensuring they're only created upon function call
            with open(self.log_path_stdout, self.mode) as file_stdout, open(self.log_path_stderr, self.mode) as file_stderr:
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                try:
                    sys.stdout = self.StreamToLogger(file_stdout, original_stdout)
                    sys.stderr = self.StreamToLogger(file_stderr, original_stderr)
                    return func(*args, **kwargs)
                except Exception as e:
                    # Capture the full traceback
                    exc_info = traceback.format_exc()
                    # Log the traceback to stderr log before re-raising the exception
                    sys.stderr.write(exc_info + '\n')
                    raise  # Re-raise the caught exception to propagate it
                finally:
                    # Reset stdout and stderr
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
        return wrapper

    class StreamToLogger(object):
        def __init__(self, file, orig_stream=None):
            self.file = file
            self.orig_stream = orig_stream

        def write(self, buf):
            if buf.strip():  # Only prepend timestamp to non-newline content
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                formatted_buf = f"{timestamp} - {buf}"
            else:
                formatted_buf = buf  # If buf is just a newline/whitespace, don't prepend timestamp
            
            if self.orig_stream:
                self.orig_stream.write(formatted_buf)  # Write to the original stream if it exists
            self.file.write(formatted_buf)  # Write to the log file

        def flush(self):
            self.file.flush()  # Ensure content is written to disk

        def __init__(self, file, orig_stream=None):
            self.file = file
            self.orig_stream = orig_stream

        def write(self, buf):
            if buf.strip():  # Check if buf is not just whitespace/newline
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                formatted_buf = f"{timestamp} - {buf}"
            else:
                formatted_buf = buf  # If buf is just a newline/whitespace, don't prepend timestamp

            # Write to the original stream if it exists
            if self.orig_stream:
                self.orig_stream.write(formatted_buf)
            
            # Write to the log file
            self.file.write(formatted_buf)

        def flush(self):
            # Ensure content is written to disk
            self.file.flush()