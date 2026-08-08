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
import gurobipy as gp
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


def clamp_dvar_bound(arr: np.ndarray, lo, hi, name: str) -> np.ndarray:
    """Return clip(arr, lo, hi) as float32, REPORTING entries changed beyond the ROUND_DECIMALS
    noise threshold. `lo`/`hi` may be scalars or same-shape arrays. Shared by the dvar bound/base
    builders in solvers/input_data.py and the ag/non-ag transition lb builders — all dvar-bound
    cleaning goes through here, explicitly and logged, rather than silently min/max'd."""
    out = np.clip(arr, lo, hi).astype(np.float32)
    thr = 10 ** (-settings.ROUND_DECIMALS)
    chg = np.abs(out - arr) > thr
    if np.any(chg):
        gap = np.abs(out - arr)[chg]
        print(f"  └── {name}: clamped {int(chg.sum())} cells, max gap={gap.max():.2e}, mean gap={gap.mean():.2e}", flush=True)
    return out


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
    agricultural_cells = np.nonzero(~np.isin(all_cells, non_agricultural_cells))[0]

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


def get_ag_to_ag_water_delta_matrix(data, from_m, from_j, cells, w_mrj, yr_idx) -> np.ndarray:
    """Source-parameterised water-licence delta ($/cell): transitioning FROM (from_m, from_j) TO every
    target (to_m, to_j) on `cells` — (target req − source req) × licence price, plus the dry↔irr
    irrigation setup/teardown. Returns (NLMS, len(cells), N_AG_LUS), RAW (un-amortised) upfront cost —
    the caller amortises explicitly (see transitions.py).
    """
    yr_cal   = data.YR_CAL_BASE + yr_idx
    area     = data.REAL_AREA[cells]
    w_target = w_mrj[:, cells, :] * settings.INCLUDE_WATER_LICENSE_COSTS
    w_base   = w_mrj[from_m, cells, from_j]
    w_cost   = (w_target - w_base[None, :, None]) * data.WATER_LICENCE_PRICE[cells, None] * data.WATER_LICENSE_COST_MULTS[yr_cal]
    if from_m == 0:    # was dryland → irrigation setup when switching to irrigated (m=1)
        w_cost[1] += settings.NEW_IRRIG_COST    * data.IRRIG_COST_MULTS[yr_cal] * area[:, None]
    else:              # was irrigated → teardown when switching to dryland (m=0)
        w_cost[0] += settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal] * area[:, None]
    return w_cost.astype(np.float32)


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
        self._orig.flush()


class LogToFile:
    _active: set = set()  # paths currently being logged; prevents double-open on nested calls

    def __init__(self, log_path, mode: str = 'a'):
        self.log_path_stdout = f"{log_path}_stdout.log"
        self.log_path_stderr = f"{log_path}_stderr.log"
        self.mode = mode
        os.makedirs(os.path.dirname(self.log_path_stdout), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path_stderr), exist_ok=True)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.log_path_stdout in LogToFile._active:
                return func(*args, **kwargs)
            LogToFile._active.add(self.log_path_stdout)
            try:
                with (
                    open(self.log_path_stdout, self.mode, encoding='utf-8') as f_out,
                    open(self.log_path_stderr, self.mode, encoding='utf-8') as f_err,
                    redirect_stdout(_TeeIO(sys.stdout, f_out)),
                    redirect_stderr(_TeeIO(sys.stderr, f_err)),
                ):
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        sys.stderr.write(traceback.format_exc() + '\n')
                        raise
            finally:
                LogToFile._active.discard(self.log_path_stdout)
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


# ---------------------------------------------------------------------------
# Shadow-price recording
#
# After an ACCEPTED (OPTIMAL) solve we read each constraint's dual (Constr.Pi)
# straight from the model — no re-solve. Constr.Pi is only a valid basic dual
# when the accepted solve produced a simplex basis (simplex, or barrier WITH
# crossover). A barrier-only solve (Crossover=0, or a concurrent solve that
# finished on the barrier) leaves no basis and Pi is unreliable, so the
# orchestrator probes the basis ONCE (CBasis on one held constraint) and skips
# the whole year rather than emit numbers that silently depend on which
# RETRY_PARAMS attempt happened to win.
#
# Real shadow price = Pi * scale['Economy'] / scale[constraint]:
#   - scale['Economy'] un-scales the (rescaled) objective back to AUD.
#   - scale[constraint] un-scales the (rescaled) RHS back to its real unit.
# Model sense is MAXIMIZE: a binding ``>=`` target gives Pi <= 0 (relaxing it by
# one unit costs objective); a binding ``<=`` cap (GHG, regional adoption) gives
# Pi >= 0. Regional-adoption constraints are not rescaled, so scale = 1.
#
# ``shadow_price`` is per real unit (AUD/ha, AUD/ML, AUD/tCO2e, …) so its magnitude
# is NOT comparable across constraint families. ``shadow_price_AUD`` normalises for
# cross-family comparison: it is the marginal value at the constraint's own target,
# shadow_price * real_RHS = Pi * scale['Economy'] * constr.RHS (the constraint scale
# cancels since real_RHS = constr.RHS * scale[constraint]). Sign follows the binding
# direction; take the absolute value to rank "which constraint costs the model most".
# Both are only meaningful for HARD constraints, so the soft forms of Water/GHG/Demand
# (``*_CONSTRAINT_TYPE == 'soft'``) are skipped entirely — a soft dual is just the
# configured penalty weight, not a scarcity price.
#
# Each ``calc_shadow_price_*`` is a pure calculation returning a DataFrame (no
# file IO, no try/except); ``record_shadow_prices`` orchestrates and writes.
# ---------------------------------------------------------------------------

def calc_shadow_price_GBF2(luto_solver, input_data, target_year) -> pd.DataFrame:
    """GBF2 priority-degraded-area constraint shadow price (AUD per real ha of target)."""
    if settings.GBF2_TARGET == "off":
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    Ss = float(input_data.scale_factors["GBF2"])
    constr = luto_solver.bio_GBF2_constrs
    if not isinstance(constr, gp.Constr):   # not built, or dropped by the infeasibility flow
        return pd.DataFrame()
    pi = float(constr.Pi)
    return pd.DataFrame([{
        "year": target_year, "constraint": "GBF2", "region": "Australia",
        "item": "", "presence": "", "pi_rescaled": pi, "scale": Ss,
        "shadow_price": pi * So / Ss, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "ha",
    }])


def calc_shadow_price_GBF3_NVIS(luto_solver, input_data, target_year) -> pd.DataFrame:
    """GBF3 NVIS vegetation-group constraint shadow prices (AUD per real ha of target)."""
    if settings.GBF3_NVIS_TARGET == "off":
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    ss = input_data.scale_factors["GBF3_NVIS"]
    rows = []
    for (region, group), constr in luto_solver.bio_GBF3_NVIS_constrs.items():
        Ss = float(ss.sel(layer=(region, group)).item())
        pi = float(constr.Pi)
        rows.append({
            "year": target_year, "constraint": "GBF3_NVIS", "region": region,
            "item": group, "presence": "", "pi_rescaled": pi, "scale": Ss,
            "shadow_price": pi * So / Ss, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "ha",
        })
    return pd.DataFrame(rows)


def calc_shadow_price_GBF4_SNES(luto_solver, input_data, target_year) -> pd.DataFrame:
    """GBF4 SNES species constraint shadow prices (AUD per real ha of target)."""
    if settings.GBF4_TARGET_SNES == "off":
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    ss = input_data.scale_factors["GBF4_SNES"]
    rows = []
    for (region, species, presence), constr in luto_solver.bio_GBF4_SNES_constrs.items():
        Ss = float(ss.sel(layer=(region, species, presence)).item())
        pi = float(constr.Pi)
        rows.append({
            "year": target_year, "constraint": "GBF4_SNES", "region": region,
            "item": species, "presence": presence, "pi_rescaled": pi, "scale": Ss,
            "shadow_price": pi * So / Ss, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "ha",
        })
    return pd.DataFrame(rows)


def calc_shadow_price_GBF4_ECNES(luto_solver, input_data, target_year) -> pd.DataFrame:
    """GBF4 ECNES ecological-community constraint shadow prices (AUD per real ha of target)."""
    if settings.GBF4_TARGET_ECNES == "off":
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    ss = input_data.scale_factors["GBF4_ECNES"]
    rows = []
    for (region, community, presence), constr in luto_solver.bio_GBF4_ECNES_constrs.items():
        Ss = float(ss.sel(layer=(region, community, presence)).item())
        pi = float(constr.Pi)
        rows.append({
            "year": target_year, "constraint": "GBF4_ECNES", "region": region,
            "item": community, "presence": presence, "pi_rescaled": pi, "scale": Ss,
            "shadow_price": pi * So / Ss, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "ha",
        })
    return pd.DataFrame(rows)


def calc_shadow_price_GBF8(luto_solver, input_data, target_year) -> pd.DataFrame:
    """GBF8 species-conservation constraint shadow prices (AUD per real ha of target)."""
    if settings.GBF8_TARGET != "on":
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    ss = input_data.scale_factors["GBF8"]
    rows = []
    for (region, species), constr in luto_solver.bio_GBF8_constrs.items():
        Ss = float(ss.sel(layer=(region, species)).item())
        pi = float(constr.Pi)
        rows.append({
            "year": target_year, "constraint": "GBF8", "region": region,
            "item": species, "presence": "", "pi_rescaled": pi, "scale": Ss,
            "shadow_price": pi * So / Ss, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "ha",
        })
    return pd.DataFrame(rows)


def calc_shadow_price_Water(luto_solver, input_data, target_year) -> pd.DataFrame:
    """Per-region water-yield constraint shadow prices (AUD per real ML of target).

    Only the hard form gives a clean scarcity price; soft mode
    (``WATER_CONSTRAINT_TYPE == 'soft'``) records nothing — its dual is just the slack penalty.
    """
    if settings.WATER_LIMITS != "on" or settings.WATER_CONSTRAINT_TYPE == "soft":
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    Ss = float(input_data.scale_factors["Water"])
    rows = []
    for constr in luto_solver.water_limit_constraints:
        pi = float(constr.Pi)
        rows.append({
            "year": target_year, "constraint": "Water", "region": "",
            "item": constr.ConstrName, "presence": "", "pi_rescaled": pi, "scale": Ss,
            "shadow_price": pi * So / Ss, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "ML",
        })
    return pd.DataFrame(rows)


def calc_shadow_price_GHG(luto_solver, input_data, target_year) -> pd.DataFrame:
    """GHG-emissions constraint shadow price (AUD per real tCO2e of target).

    Only the hard ``<=`` upper bound gives a clean scarcity price; soft mode
    (``GHG_CONSTRAINT_TYPE == 'soft'``) records nothing — its dual is just the objective's
    penalty rate on the deviation var.
    """
    if settings.GHG_EMISSIONS_LIMITS == "off" or settings.GHG_CONSTRAINT_TYPE == "soft":
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    Ss = float(input_data.scale_factors["GHG"])
    constr = luto_solver.ghg_consts_ub
    if constr is None:                      # not built, or dropped by the infeasibility flow
        return pd.DataFrame()
    pi = float(constr.Pi)
    return pd.DataFrame([{
        "year": target_year, "constraint": "GHG", "region": "",
        "item": constr.ConstrName, "presence": "", "pi_rescaled": pi, "scale": Ss,
        "shadow_price": pi * So / Ss, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "tCO2e",
    }])


def calc_shadow_price_Demand(luto_solver, input_data, target_year) -> pd.DataFrame:
    """Per-commodity production/demand constraint shadow prices (AUD per real tonne of demand).

    Only hard bounds (``DEMAND_CONSTRAINT_TYPE == 'hard'``) give a clean marginal price; soft mode
    records nothing — its dual is just the commodity-price penalty rate on the deviation var
    ``V[c]``. ``presence`` holds the bound kind (eq/lower/upper) so a commodity's paired hard bounds
    stay distinguishable.
    """
    if settings.DEMAND_CONSTRAINT_TYPE == "soft":
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    Ss = float(input_data.scale_factors["Demand"])
    commodities = input_data.commodity_names
    rows = []
    for constr in luto_solver.demand_penalty_constraints:
        name = constr.ConstrName                        # e.g. demand_hard_bound_lower[3]
        m = re.search(r"\[(\d+)\]", name)
        commodity = commodities[int(m.group(1))] if m else name
        bound = name.split("_bound_")[-1].split("[")[0]  # eq / lower / upper
        pi = float(constr.Pi)
        rows.append({
            "year": target_year, "constraint": "Demand", "region": "",
            "item": commodity, "presence": bound, "pi_rescaled": pi, "scale": Ss,
            "shadow_price": pi * So / Ss, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "t",
        })
    return pd.DataFrame(rows)


def calc_shadow_price_Renewable(luto_solver, input_data, target_year) -> pd.DataFrame:
    """State-level renewable-generation-target shadow prices (AUD per real MWh of target).

    Solar and wind are rescaled independently, so the scale is looked up per type.
    """
    if not any(settings.RENEWABLES_OPTIONS.values()):
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    rows = []
    for key, constr in luto_solver.renewable_constraints.items():
        # key == f"{am}_{reg_name}"; both parts can contain spaces, so match the known am prefix.
        am = next((a for a in ("Utility Solar PV", "Onshore Wind") if key.startswith(a + "_")), None)
        Ss = float(input_data.scale_factors[am]) if am is not None else float("nan")
        reg = key[len(am) + 1:] if am is not None else key
        pi = float(constr.Pi)
        rows.append({
            "year": target_year, "constraint": am or "Renewable", "region": reg,
            "item": "", "presence": "", "pi_rescaled": pi, "scale": Ss,
            "shadow_price": pi * So / Ss, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "MWh",
        })
    return pd.DataFrame(rows)


def calc_shadow_price_Regional_Adoption(luto_solver, input_data, target_year) -> pd.DataFrame:
    """Regional adoption area-cap shadow prices (AUD per real ha of cap).

    These constraints carry no RHS rescale, so scale = 1 and shadow_price = Pi * scale['Economy'].
    """
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return pd.DataFrame()
    So = float(input_data.scale_factors["Economy"])
    rows = []
    for constr in luto_solver.regional_adoption_constrs:
        pi = float(constr.Pi)
        rows.append({
            "year": target_year, "constraint": "Regional_Adoption", "region": "",
            "item": constr.ConstrName, "presence": "", "pi_rescaled": pi, "scale": 1.0,
            "shadow_price": pi * So, "shadow_price_AUD": pi * So * float(constr.RHS), "unit": "ha",
        })
    return pd.DataFrame(rows)


def record_shadow_prices(luto_solver, input_data, target_year, out_dir) -> None:
    """Compute every active constraint's shadow prices and write one CSV for the year.

    Probes the simplex basis once (barrier-only solves have unreliable duals → skip the year),
    then concatenates the per-constraint calculators into ``shadow_prices_{target_year}.csv``.
    The file is written fresh per year, so a resume/re-run simply overwrites the year's file.
    """
    # Grab one constraint we already hold to probe the basis — avoids `model.getConstrs()`, which
    # builds a Python list of *every* constraint (millions of per-cell constraints at full res).
    gbf2 = luto_solver.bio_GBF2_constrs
    probe = gbf2 if (gbf2 and not isinstance(gbf2, dict)) else None    # single Constr once GBF2 is on
    for coll in (
        luto_solver.bio_GBF3_NVIS_constrs, luto_solver.bio_GBF4_SNES_constrs,
        luto_solver.bio_GBF4_ECNES_constrs, luto_solver.bio_GBF8_constrs,
        luto_solver.renewable_constraints,
    ):
        if probe is None and coll:
            probe = next(iter(coll.values()))
    for coll in (luto_solver.water_limit_constraints, luto_solver.regional_adoption_constrs,
                 luto_solver.ghg_consts_soft, luto_solver.demand_penalty_constraints):
        if probe is None and coll:
            probe = coll[0]
    probe = probe or luto_solver.ghg_consts_ub
    if probe is None:
        print(f"No active constraints to record shadow prices for {target_year}.")
        return

    # Constr.Pi is a clean basic dual only when the accepted solve left a simplex basis; CBasis
    # raises GurobiError on a barrier-only solve (no basis) → duals unreliable, skip the year.
    try:
        _ = probe.CBasis
    except gp.GurobiError:
        print(f"Skipping shadow prices for {target_year}: accepted solve has no simplex basis "
              f"(barrier-only) — duals would be unreliable.")
        return

    # Each calculator returns rows for its active constraints, or a column-less empty frame.
    df = pd.concat(
        [calc(luto_solver, input_data, target_year) for calc in (
            calc_shadow_price_GBF2,
            calc_shadow_price_GBF3_NVIS,
            calc_shadow_price_GBF4_SNES,
            calc_shadow_price_GBF4_ECNES,
            calc_shadow_price_GBF8,
            calc_shadow_price_Water,
            calc_shadow_price_GHG,
            calc_shadow_price_Demand,
            calc_shadow_price_Renewable,
            calc_shadow_price_Regional_Adoption,
        )],
        ignore_index=True,
    )

    df.to_csv(f"{out_dir}/shadow_prices_{target_year}.csv", index=False)
    print(f"Recorded {len(df)} shadow prices for {target_year} -> {out_dir}/shadow_prices_{target_year}.csv")

