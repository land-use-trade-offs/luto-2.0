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
To maintain state and handle iteration and data-view changes. This module
functions as a singleton class. It is intended to be the _only_ part of the
model that has 'global' varying state.
"""

import os
import time
import dill

from datetime import datetime
from joblib import Parallel, delayed

import luto.settings as settings
from luto import tools
from luto.settings import NON_AG_LAND_USES
from luto.data import Data, get_base_am_vars, lumap2ag_l_mrj, lumap2non_ag_l_mk
from luto.solvers.input_data import get_input_data
from luto.solvers.solver import LutoSolver
from luto.tools.report.data_tools import get_all_files

# Get date and time
timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

@tools.LogToFile(f"{settings.OUTPUT_DIR}/run_{timestamp}")
def load_data() -> Data:
    """
    Load the Data object containing all required data to run a LUTO simulation.
    """
    return Data(timestamp=timestamp)

@tools.LogToFile(f"{settings.OUTPUT_DIR}/run_{timestamp}", 'a')
def run( data: Data, base: int, target: int) -> None:
    """
    Run the simulation.
    Parameters:
        'data' is a Data object, and 'base' and 'target' are the base and target years for the whole simulation.
    """

    # Set Data object's path and create output directories
    data.set_path(base, target)

    # Run the simulation up to `year` sequentially.
    if settings.MODE == 'timeseries':
        if len(data.D_CY.shape) != 2:
            raise ValueError( "Demands need to be a time series array of shape (years, commodities) and years > 0." )
        if target - base > data.D_CY.shape[0]:
            raise ValueError( "Not enough years in demands time series.")

        steps = target - base
        solve_timeseries(data, steps, base, target)

    elif settings.MODE == 'snapshot':
        # If demands is a time series, choose the appropriate entry.
        solve_snapshot(data, base, target)

    else:
        raise ValueError(f"Unkown MODE: {settings.MODE}.")


def solve_timeseries(data: Data, steps: int, base: int, target: int):
    print('\n')
    print(f"\nRunning LUTO {settings.VERSION} timeseries from {base} to {target} at resfactor {settings.RESFACTOR}, starting at {time.ctime()}.", flush=True)

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
        data.add_obj_vals(yr, solution.obj_val_sum)

        if settings.CALC_BIODIVERSITY_CONTRIBUTION:
            print(f'Reproject decision variables...')
            data.add_ag_dvars_xr(yr, solution.ag_X_mrj)
            data.add_am_dvars_xr(yr, solution.ag_man_X_mrj)
            data.add_non_ag_dvars_xr(yr, solution.non_ag_X_rk)

        for data_type, prod_data in solution.prod_data.items():
            data.add_production_data(yr, data_type, prod_data)

        print(f'Processing for {base + s + 1} completed in {round(time.time() - start_time)} seconds\n\n' )


def solve_snapshot(data: Data, base: int, target: int):
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
    data.add_obj_vals(target, solution.obj_val_sum)

    if settings.CALC_BIODIVERSITY_CONTRIBUTION:
        print(f'Reproject decision variables...')
        data.add_ag_dvars_xr(target, solution.ag_X_mrj)
        data.add_am_dvars_xr(target, solution.ag_man_X_mrj)
        data.add_non_ag_dvars_xr(target, solution.non_ag_X_rk)

    for data_type, prod_data in solution.prod_data.items():
        data.add_production_data(target, data_type, prod_data)

    print(f'Processing for {target} completed in {round(time.time() - start_time)} seconds\n\n')


def save_data_to_disk(data: Data, path:str) -> None:
    """Save the Data object to disk.
    Arguments:
        data: `Data` object.
        path: Path to save the Data object.
    """
    # Save
    with open(path, 'wb') as f: dill.dump(data, f)
    

def load_data_from_disk(path:str) -> Data:
    """Load the Data object from disk.
    
    Arguments:
        path: Path to the Data object.

    Raises:
        ValueError: if the resolution factor from the data object does not match the settings.RESFACTOR.

    Returns:
        Data: `Data` object.
    """
    # Load the data object
    with open(path, 'rb') as f: 
        data = dill.load(f)
    
    # Check if the resolution factor from the data object matches the settings.RESFACTOR
    if int(data.RESMULT ** 0.5) != settings.RESFACTOR: 
        raise ValueError(f'Resolution factor from data loading ({int(data.RESMULT ** 0.5)}) does not match it of settings ({settings.RESFACTOR})!')

    # Update the timestamp
    data.timestamp_sim = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    
    return data
    

def read_dvars(output_dir:str)-> None:
    '''
    Read the output decision variables from the output directory and add them to the Data object.
    '''
    # Read all output files;
    files = get_all_files(output_dir)
    dvar_path = files.query('category == "ag_X_mrj"').iloc[0]['path']
    # Check if the output resolution is the same as the settings.RESFACTOR
    output_res = tools.get_out_resfactor(dvar_path)
    if output_res != settings.RESFACTOR:
        raise ValueError(f'Please change the `settings.RESFACTOR` ({settings.RESFACTOR}) to to be the same as output files ({output_res}).')

    # Loading data from
    data = load_data()

    # Save reading dvars as a delayed task
    tasks = [delayed(tools.read_dvars)(yr, files.query('base_ext == ".npy" and Year == @yr and year_types == "single_year"'))
            for yr in sorted(files['Year'].unique())]

    # Run the tasks in parallel to add the dvars to the data object
    print(f'Reading decision variables from existing output directory...\n   {output_dir}')
    for yr,dvar in Parallel(n_jobs=min(len(tasks), 10), return_as='generator')(tasks):
        data.add_lumap(yr, dvar[0])
        data.add_lmmap(yr, dvar[1])
        data.add_ammaps(yr, dvar[2])
        data.add_ag_dvars(yr, dvar[3])
        data.add_non_ag_dvars(yr, dvar[4])
        data.add_ag_man_dvars(yr, dvar[5])

    # Remove the log file, because we only read the dvars
    os.remove(f"{settings.OUTPUT_DIR}/run_{timestamp}_stderr.log")
    os.remove(f"{settings.OUTPUT_DIR}/run_{timestamp}_stdout.log")

    return data