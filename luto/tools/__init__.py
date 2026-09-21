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

import numpy as np
import psutil
import xarray as xr
import numpy_financial as npf
import matplotlib.patches as patches

from datetime import datetime
from matplotlib import pyplot as plt

import luto.settings as settings


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
    builders in solvers/col_builder.py and the ag/non-ag transition lb builders — all dvar-bound
    cleaning goes through here, explicitly and logged, rather than silently min/max'd."""
    out = np.clip(arr, lo, hi).astype(np.float32)
    thr = 10 ** (-settings.ROUND_DECIMALS)
    chg = np.abs(out - arr) > thr
    if np.any(chg):
        gap = np.abs(out - arr)[chg]
        print(f"  └── {name}: clamped {int(chg.sum())} cells, max gap={gap.max():.2e}, mean gap={gap.mean():.2e}", flush=True)
    return out


def get_base_held(dvar: np.ndarray) -> np.ndarray:
    """The base-year holding the model recognises: a share at or below the ROUND_DECIMALS noise floor is dropped,
    exactly as the source maps drop it. Nothing under the floor is a source, so no arc can move it — and it must
    not raise a target's upper bound or a node's base either. Negatives (never expected) go to zero with it.

    A deadband, not a clip: clamp_dvar_bound above bounds a range and leaves a sliver where it is, this zeroes it.
    Both key off the same ROUND_DECIMALS floor, and every dvar that reaches the column space passes through one
    or the other."""
    return np.where(dvar > 10 ** (-settings.ROUND_DECIMALS), dvar, 0).astype(np.float32)


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


def am_name_snake_case(am_name):
    """Get snake_case version of the AM name"""
    return am_name.lower().replace(' ', '_')


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

