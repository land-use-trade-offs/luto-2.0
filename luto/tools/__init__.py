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



"""
Pure helper functions and other tools.
"""

import re
import sys
import os.path
import time
import traceback
import functools
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import numpy as np
import psutil
import xarray as xr
import numpy_financial as npf
import matplotlib.patches as patches

from typing import Tuple
from datetime import datetime
from matplotlib import pyplot as plt

import luto.settings as settings
import luto.economics.agricultural.water as ag_water
import luto.economics.non_agricultural.water as non_ag_water


def write_timestamp():
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    timestamp_path = os.path.join(settings.OUTPUT_DIR, '.timestamp')
        
    with open(timestamp_path, 'w') as f: f.write(timestamp)
    return timestamp

def read_timestamp():
    timestamp_path = os.path.join(settings.OUTPUT_DIR, '.timestamp')
    if os.path.exists(timestamp_path):
        with open(timestamp_path, 'r') as f: timestamp = f.read()
    else:
        raise FileNotFoundError(f"Timestamp file not found at {timestamp_path}")
    return timestamp


def amortise(cost, rate=settings.DISCOUNT_RATE, horizon=settings.AMORTISATION_PERIOD):
    """Return NPV of future `cost` amortised to annual value at discount `rate` over `horizon` years."""
    if settings.AMORTISE_UPFRONT_COSTS: 
        return -1 * npf.pmt(rate, horizon, pv=cost, fv=0, when='begin')
    else: 
        return cost


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
    Get an array with cells that currently use agroforestry (either sheep or beef)
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


def get_destocked_land_cells(lumap) -> np.ndarray:
    """
    Get an array with all destocked land cells
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 8)[0]


def get_unallocated_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for unallocated natural land uses.
    """
    return np.nonzero(np.isin(lumap, data.DESC2AGLU["Unallocated - natural land"]))[0]

def get_lvstk_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for livestock natural land uses.
    """
    return np.nonzero(np.isin(lumap, data.LU_LVSTK_NATURAL))[0]


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
    Get an array containing the index of all agricultural cells
    """
    return np.nonzero(lumap < settings.NON_AGRICULTURAL_LU_BASE_CODE)[0]


def get_non_ag_cells(lumap) -> np.ndarray:
    """
    Get an array containing the index of all non-agricultural cells
    """
    return np.nonzero(lumap >= settings.NON_AGRICULTURAL_LU_BASE_CODE)[0]


def get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx):
    """
    Gets the water delta matrix ($/cell) that applies the cost of installing/removing irrigation to
    base transition costs. Includes the costs of water license fees.

    Parameters
     w_mrj (numpy.ndarray, <unit:ML/cell>): Water requirements matrix for target year.
     l_mrj (numpy.ndarray): Land-use and land management matrix for the base_year.
     data (object): Data object containing necessary information.

    Returns
     w_delta_mrj (numpy.ndarray, <unit:$/cell>).
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

def get_ag_to_non_ag_water_delta_matrix(data, yr_idx, lumap, lmmap)->tuple[np.ndarray, np.ndarray]:
    """
    Gets the water delta matrix ($/cell) that applies the cost of installing/removing irrigation to
    base transition costs. Includes the costs of water license fees.
    
    Parameters
     data (object): Data object containing necessary information.
     yr_idx (int): Index of the target year.
     lumap (numpy.ndarray): Land-use map.
     lmmap (numpy.ndarray): Land management map.
    
    Returns
     w_rm_irrig_cost_r (numpy.ndarray) : Cost of removing irrigation for each cell.
     
     
    """
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    l_mrj = lumap2ag_l_mrj(lumap, lmmap)
    non_ag_cells = get_non_ag_cells(lumap)
    
    w_req_mrj = ag_water.get_wreq_matrices(data, yr_idx).astype(np.float32)     # <unit: ML/CELL>
    w_req_r = (w_req_mrj * l_mrj).sum(axis=0).sum(axis=1)
    w_yield_r = non_ag_water.get_w_net_yield_env_planting(data, yr_idx)  # <unit: ML/CELL>
    w_delta_r = - (w_req_r + w_yield_r)
    
    w_license_cost_r = w_delta_r * data.WATER_LICENCE_PRICE * data.WATER_LICENSE_COST_MULTS[yr_cal] * settings.INCLUDE_WATER_LICENSE_COSTS     # <unit: $/CELL>
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA                   # <unit: $/CELL>

    return w_rm_irrig_cost_r


def am_name_snake_case(am_name):
    """Get snake_case version of the AM name"""
    return am_name.lower().replace(' ', '_')


def get_exclusions_for_excluding_all_natural_cells(data, lumap) -> np.ndarray:
    """
    A number of non-agricultural land uses can only be applied to cells that
    don't already utilise a natural land use. This function gets the exclusion
    matrix for all such non-ag land uses, returning an array valued 0 at the 
    indices of cells that use natural land uses, and 1 everywhere else.

    Parameters
     data: The data object containing information about the cells.
     lumap: The land use map.

    Returns
     exclude: An array of shape (NCELLS,) with values 0 at the indices of cells
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

    Parameters
     data: The data object containing information about the landscape.
     lumap: The land use map.

    Returns
     exclude: A 1-D array.
    """
    exclude = (np.ones(data.NCELLS) * settings.AF_PROPORTION).astype(np.float32)

    # Ensure cells being used for agroforestry may retain that LU
    exclude[get_agroforestry_cells(lumap)] = settings.AF_PROPORTION

    return exclude


def get_exclusions_carbon_plantings_belt_base(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much carbon plantings (belt) can possibly 
    be done at each cell.

    Parameters
     data (Data): The data object containing information about the cells.
     lumap (np.ndarray): The land use map.

    Returns
     exclude: A 1-D array
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


def get_natural_sheep_code(data):
    """
    Get the land use code (j) for 'Sheep - natural land'
    """
    return data.DESC2AGLU['Sheep - natural land']


def get_natural_beef_code(data):
    """
    Get the land use code (j) for 'Beef - modified land'
    """
    return data.DESC2AGLU['Beef - natural land']


def get_unallocated_natural_land_code(data):
    """
    Get the land use code (j) for 'Unallocated - natural land'
    """
    return data.DESC2AGLU['Unallocated - natural land']


def get_cells_using_ag_landuse(lumap: np.ndarray, j: int) -> np.ndarray:
    """
    Gets the cells in the given 'lumap' using the land use indexed by 'j'
    """
    return np.where(lumap == j)[0]


def ag_mrj_to_xr(data, arr: np.ndarray, threshold: float = 0.01) -> xr.DataArray:
    """Convert agricultural dvar array to xarray DataArray with automatic masking.

    Masks out cells where the sum across all land uses is less than 0.01.
    """
    xr_arr = xr.DataArray(
        arr,
        dims=['lm', 'cell', 'lu'],
        coords={'lm': data.LANDMANS,
                'cell': np.arange(data.NCELLS),
                'lu': data.AGRICULTURAL_LANDUSES}
    ).astype(np.float32)

    # Mask out cells with very small values
    ag_mask = (abs(xr_arr.sum(['lu','lm'])) > threshold).values
    xr_arr = xr_arr.where(ag_mask[None,:,None], 0)

    return xr_arr

def non_ag_rk_to_xr(data, arr: np.ndarray, threshold: float = 0.01) -> xr.DataArray:
    """Convert non-agricultural dvar array to xarray DataArray with automatic masking.

    Masks out cells where the sum across all land uses is less than 0.01.
    """
    xr_arr = xr.DataArray(
        arr,
        dims=['cell', 'lu'],
        coords={'cell': np.arange(data.NCELLS),
                'lu': data.NON_AGRICULTURAL_LANDUSES}
    ).astype(np.float32)

    # Mask out cells with very small values
    non_ag_mask = (abs(xr_arr.sum('lu')) > threshold).values
    xr_arr = xr_arr.where(non_ag_mask[..., None], 0)

    return xr_arr

def am_mrj_to_xr(data, am_mrj_dict: dict, threshold: float = 0.01) -> xr.DataArray:
    """Convert agricultural management dvar dict to xarray DataArray with automatic masking.

    Masks out cells where the sum across all agricultural management types is less than 0.01.
    """
    arr = np.zeros((data.N_AG_MANS, data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)

    for am_idx, (am, lu_names) in enumerate(data.AG_MAN_LU_DESC.items()):
        lu_idxs = [data.DESC2AGLU[lu] for lu in lu_names]
        src = am_mrj_dict[am]

        if src.shape[-1] == len(lu_idxs):
            for j, li in enumerate(lu_idxs):
                arr[am_idx, :, :, li] = src[:, :, j]
        else:
            src_lu_idxs = [data.DESC2AGLU[i] for i in settings.AG_MANAGEMENTS_TO_LAND_USES[am]]
            for j, li in enumerate(lu_idxs):
                arr[am_idx, :, :, li] = src[:, :, src_lu_idxs[j]]

    # Mask out cells with very small values
    cell_sum = np.abs(arr).sum(axis=(0, 1, 3))
    arr[:, :, cell_sum <= threshold, :] = 0

    return xr.DataArray(
        arr,
        dims=['am', 'lm', 'cell', 'lu'],
        coords={'am': data.AG_MAN_DESC,
                'lm': data.LANDMANS,
                'cell': np.arange(data.NCELLS),
                'lu': data.AGRICULTURAL_LANDUSES}
    )


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


def plot_t_mat(t_mat:xr.DataArray):
    
    '''
    Plot the transition matrix with hatched rectangles for NaN values.
    
    Parameters
    ----------
    t_mat : xr.DataArray
        The transition matrix to plot.
        
    '''
 
    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot with imshow for correct alignment
    im = ax.imshow(t_mat.values, cmap='viridis', origin='upper')

    # Set tick positions and labels
    ax.set_xticks(np.arange(len(t_mat.coords['to_lu'])))
    ax.set_yticks(np.arange(len(t_mat.coords['from_lu'])))
    ax.set_xticklabels(t_mat.coords['to_lu'].values, rotation=90)
    ax.set_yticklabels(t_mat.coords['from_lu'].values)

    # Move x labels to top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Draw hatched rectangles over NaNs
    nrows, ncols = t_mat.shape
    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(t_mat[i, j]):
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, hatch='////', fill=False, edgecolor='gray', linewidth=0.0)
                ax.add_patch(rect)

def set_path() -> str:
        """Create a folder for storing outputs and return folder name."""
        years = [i for i in settings.SIM_YEARS]
        path = f"{settings.OUTPUT_DIR}/{read_timestamp()}_RF{settings.RESFACTOR}_{years[0]}-{years[-1]}"
        paths = [path] + [f"{path}/out_{yr}" for yr in years]
        
        for p in paths:
            if not os.path.exists(p):
                os.mkdir(p)
  

class _TeeIO:
    """Write to both an original stream and a log file, adding timestamps."""

    _ts_re = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - ')

    def __init__(self, orig_stream, file):
        self._orig = orig_stream
        self._file = file

    def write(self, buf):
        if buf.strip() and not self._ts_re.match(buf):
            buf = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {buf}"
        self._file.write(buf)
        self._orig.write(buf)

    def flush(self):
        self._file.flush()


class LogToFile:
    def __init__(self, log_path, mode: str = 'a'):
        self.log_path_stdout = f"{log_path}_stdout.log"
        self.log_path_stderr = f"{log_path}_stderr.log"
        self.mode = mode
        os.makedirs(os.path.dirname(self.log_path_stdout), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path_stderr), exist_ok=True)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with (
                open(self.log_path_stdout, self.mode) as f_out,
                open(self.log_path_stderr, self.mode) as f_err,
                redirect_stdout(_TeeIO(sys.stdout, f_out)),
                redirect_stderr(_TeeIO(sys.stderr, f_err)),
            ):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    sys.stderr.write(traceback.format_exc() + '\n')
                    raise
        return wrapper
            
            

def log_memory_usage(output_dir=settings.OUTPUT_DIR, mode='a', interval=1, stop_event=None):
    '''
    Log the memory usage of the current process to a file with enhanced accuracy.
    Parameters
        output_dir (str): The directory to save the memory log file.
        mode (str): The mode to open the file. Default is 'a' (append).
        interval (int): The interval in seconds to log the memory usage.
    '''
    
    with open(f'{output_dir}/RES_{settings.RESFACTOR}_mem_log.txt', mode=mode) as file:
        while not stop_event.is_set():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            process = psutil.Process(os.getpid())
            
            # Get working set memory (most accurate) - ensure consistency across all processes
            memory_info = process.memory_info()
            
            # Check if working set is available on this system
            has_wset = hasattr(memory_info, 'wset')
            
            if has_wset:
                wset_memory = memory_info.wset
            else:
                wset_memory = memory_info.rss
            
            # Include child processes using the SAME metric type
            children = process.children(recursive=True)
            if children:
                for child in children:
                    try:
                        child_memory_info = child.memory_info()
                        if has_wset and hasattr(child_memory_info, 'wset'):
                            wset_memory += child_memory_info.wset
                        else:
                            # Use RSS for consistency if wset not available
                            wset_memory += child_memory_info.rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            
            # Write working set memory info (most accurate)
            wset_gb = wset_memory / (1024 * 1024 * 1024)
            
            file.write(f'{timestamp}\t{wset_gb:.3f}\n')
            file.flush()
            time.sleep(interval)

