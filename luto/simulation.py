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
To maintain state and handle iteration and data-view changes. This module
functions as a singleton class. It is intended to be the _only_ part of the
model that has 'global' varying state.
"""

import gzip
import os
import time
import dill
import threading
import time

from datetime import datetime
from joblib import Parallel, delayed

import luto.settings as settings

from luto.data import Data
from luto import tools
from luto.solvers.input_data import get_input_data
from luto.solvers.solver import LutoSolver
from luto.tools.create_task_runs.helpers import log_memory_usage
from luto.tools.report.data_tools import get_all_files
from luto.tools.write import write_outputs

# Get date and time
timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


@tools.LogToFile(f"{settings.OUTPUT_DIR}/run_{timestamp}")
def load_data(base_year: int | None = None) -> Data:
    """
    Load the Data object containing all required data to run a LUTO simulation.
    """
    memory_thread = threading.Thread(target=log_memory_usage, daemon=True)
    memory_thread.start()
    
    return Data(timestamp=timestamp, base_year=base_year)

@tools.LogToFile(f"{settings.OUTPUT_DIR}/run_{timestamp}", 'a')
def run( data: Data, base_year: int, target: int) -> None:
    """
    Run the simulation.
    Parameters:
        'data' is a Data object, and 'base' and 'target' are the base and target years for the whole simulation.
    """
    memory_thread = threading.Thread(target=log_memory_usage, daemon=True)
    memory_thread.start()
    
    # Set Data object's path and create output directories
    data.set_path(base_year, target)

    # Run the simulation up to `year` sequentially.
    if settings.MODE == 'timeseries':
        if len(data.D_CY.shape) != 2:
            raise ValueError( "Demands need to be a time series array of shape (years, commodities) and years > 0." )
        if target - base_year > data.D_CY.shape[0]:
            raise ValueError( "Not enough years in demands time series.")

        steps = target - base_year
        solve_timeseries(data, steps, base_year, target)

    elif settings.MODE == 'snapshot':
        # If demands is a time series, choose the appropriate entry.
        solve_snapshot(data, base_year, target)

    else:
        raise ValueError(f"Unkown MODE: {settings.MODE}.")
    
    # Save the Data object to disk
    write_outputs(data)


def solve_timeseries(data: Data, steps: int, base: int, target: int):
    print('\n')
    print(f"Running LUTO {settings.VERSION} timeseries from {base} to {target} at resfactor {settings.RESFACTOR}.", flush=True)

    for s in range(steps):
        print( "-------------------------------------------------")
        print( f"Running for year {base + s + 1}"   )
        print( "-------------------------------------------------\n" )
        start_time = time.time()

        input_data = get_input_data(data, base + s, base + s + 1)
        d_c = data.D_CY[s + 1]

        if s == 0:
            luto_solver = LutoSolver(input_data, d_c, target)
            luto_solver.formulate()

        if s > 0:
            old_ag_x_mrj = luto_solver._input_data.ag_x_mrj.copy()
            old_ag_man_lb_mrj = luto_solver._input_data.ag_man_lb_mrj.copy()
            old_non_ag_x_rk = luto_solver._input_data.non_ag_x_rk.copy()
            old_non_ag_lb_rk = luto_solver._input_data.non_ag_lb_rk.copy()

            luto_solver.update_formulation(
                input_data=input_data,
                d_c=d_c,
                old_ag_x_mrj=old_ag_x_mrj,
                old_ag_man_lb_mrj=old_ag_man_lb_mrj,
                old_non_ag_x_rk=old_non_ag_x_rk,
                old_non_ag_lb_rk=old_non_ag_lb_rk,
                old_lumap=data.lumaps[base + s - 1],
                current_lumap=data.lumaps[base + s],
                old_lmmap=data.lmmaps[base + s - 1],
                current_lmmap=data.lmmaps[base + s],
            )

        solution = luto_solver.solve()

        yr = base + s + 1
        data.add_lumap(yr, solution.lumap)
        data.add_lmmap(yr, solution.lmmap)
        data.add_ammaps(yr, solution.ammaps)
        data.add_ag_dvars(yr, solution.ag_X_mrj)
        data.add_non_ag_dvars(yr, solution.non_ag_X_rk)
        data.add_ag_man_dvars(yr, solution.ag_man_X_mrj)
        data.add_obj_vals(yr, solution.obj_val)


        for data_type, prod_data in solution.prod_data.items():
            data.add_production_data(yr, data_type, prod_data)

        print(f'Processing for {base + s + 1} completed in {round(time.time() - start_time)} seconds\n\n' )


def solve_snapshot(data: Data, base: int, target: int):
    if base < 2010 or base >= target:
        raise ValueError("Base year must be >= 2010 and less than the target year.")

    if len(data.D_CY.shape) == 2:
        d_c = data.D_CY[ target - data.YR_CAL_BASE ]       # Demands needs to be a timeseries from 2010 to target year
    else:
        d_c = data.D_CY

    print('\n')
    print( f"Running LUTO {settings.VERSION} snapshot for {target} at resfactor {settings.RESFACTOR}" )
    print( "-------------------------------------------------" )
    print( f"Running for year {target}" )
    print( "-------------------------------------------------" )

    start_time = time.time()
    input_data = get_input_data(data, base, target)
    luto_solver = LutoSolver(input_data, d_c, target)
    luto_solver.formulate()

    solution = luto_solver.solve()
    data.add_lumap(target, solution.lumap)
    data.add_lmmap(target, solution.lmmap)
    data.add_ammaps(target, solution.ammaps)
    data.add_ag_dvars(target, solution.ag_X_mrj)
    data.add_non_ag_dvars(target, solution.non_ag_X_rk)
    data.add_ag_man_dvars(target, solution.ag_man_X_mrj)
    data.add_obj_vals(target, solution.obj_val)

    for data_type, prod_data in solution.prod_data.items():
        data.add_production_data(target, data_type, prod_data)

    print(f'Processing for {target} completed in {round(time.time() - start_time)} seconds\n\n')


def save_data_to_disk(data: Data, path: str, compress_level=9) -> None:
    """Save the Data object to disk with gzip compression.
    Arguments:
        data: `Data` object.
        path: Path to save the Data object.
        compress_level: Compression level for gzip compression.
    """
    # Save with gzip compression
    with gzip.open(path, 'wb', compresslevel=compress_level) as f:
        dill.dump(data, f)
    

def load_data_from_disk(path: str) -> Data:
    """Load the Data object from disk.
    
    Arguments:
        path: Path to the Data object.

    Raises:
        ValueError: if the resolution factor from the data object does not match the settings.RESFACTOR.

    Returns:
        Data: `Data` object.
    """
    # Load the data object with gzip compression
    with gzip.open(path, 'rb') as f:
        data = dill.load(f)
    
    # Check if the resolution factor from the data object matches the settings.RESFACTOR
    if int(data.RESMULT ** 0.5) != settings.RESFACTOR:
        raise ValueError(f'Resolution factor from data loading ({int(data.RESMULT ** 0.5)}) does not match it of settings ({settings.RESFACTOR})!')

    # Update the timestamp
    data.timestamp_sim = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    
    return data
  