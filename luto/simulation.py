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

import time
from datetime import datetime

import luto.settings as settings
from luto import tools
from luto.non_ag_landuses import NON_AG_LAND_USES
from luto.data import Data, get_base_am_vars, lumap2ag_l_mrj, lumap2non_ag_l_mk
from luto.economics.production import get_production
from luto.solvers.input_data import SolverInputData, get_input_data
from luto.solvers.solver import LutoSolver

# Get date and time
timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


def load_data() -> Data:
    """
    Load the Data object containing all required data to run a LUTO simulation.
    """
    data = Data(timestamp=timestamp)

    # Calculate base year production figures
    print(f"Calculating base year ({data.YR_CAL_BASE}) production data...", end = " ", flush = True)
    yr_cal_base_prod_data = get_production( 
        data,
        data.YR_CAL_BASE,
        lumap2ag_l_mrj(data.LUMAP, data.LMMAP),
        lumap2non_ag_l_mk(data.LUMAP, len(NON_AG_LAND_USES.keys())),
        get_base_am_vars(data.NCELLS, data.NLMS, data.N_AG_LUS),
    )
    data.add_production_data(data.YR_CAL_BASE, "Production", yr_cal_base_prod_data)
    print("Done.")

    # Apply resfactor for solve and set up initial data for base year
    data.apply_resfactor()
    data.add_base_year_data_to_containers()

    return data


def solve_timeseries(data: Data, steps: int, base: int, target: int):
    print('\n')
    print(f"\nRunning LUTO {settings.VERSION} timeseries from {base} to {target} at resfactor {settings.RESFACTOR}, starting at {time.ctime()}.", flush=True)

    for s in range(steps):
        print( "-------------------------------------------------")
        print( f"Running for year {base + s + 1}"   )
        print( "-------------------------------------------------\n" )
        start_time = time.time()

        input_data = get_input_data(data, base + s, base + s + 1)
        d_c = data.D_CY[s]
        
        if s == 0:
            luto_solver = LutoSolver(input_data, d_c)
            luto_solver.formulate()

        if s > 0:
            old_ag_x_mrj = luto_solver._input_data.ag_x_mrj.copy()
            old_non_ag_x_rk = luto_solver._input_data.non_ag_x_rk.copy()

            luto_solver.update_formulation(
                input_data=input_data,
                d_c=d_c,
                old_ag_x_mrj=old_ag_x_mrj,
                old_non_ag_x_rk=old_non_ag_x_rk,
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
    luto_solver = LutoSolver(input_data, d_c)
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

@tools.LogToFile(f"{settings.OUTPUT_DIR}/run_{timestamp}")
def run( data: Data, base: int, target: int) -> None:
    """
    Run the simulation.
    Parameters:
        'data' is a Data object, and 'base' and 'target' are the base and target years.
    """
    
    # Set Data object's path and create output directories
    data.set_path(base, target)

    # Run the simulation up to `year` sequentially.         *** Not sure that timeseries mode is working ***
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
