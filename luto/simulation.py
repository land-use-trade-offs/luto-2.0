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

from luto.tools import calc_major_vegetation_group_ag_area_for_year, calc_species_ag_area_for_year
from luto.economics.agricultural.biodiversity import get_major_vegetation_matrices
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.biodiversity as ag_biodiversity


# Get date and time
timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


@tools.LogToFile(f"{settings.OUTPUT_DIR}/run_{timestamp}")
def load_data() -> Data:
    """
    Load the Data object containing all required data to run a LUTO simulation.
    """
    memory_thread = threading.Thread(target=log_memory_usage, daemon=True)
    memory_thread.start()
    
    return Data(timestamp=timestamp)


@tools.LogToFile(f"{settings.OUTPUT_DIR}/run_{timestamp}", 'a')
def run(
    data: Data, 
    base_year: int | None = None, 
    target_year: int | None = None, 
    *,
    step_size: int | None = None,
    years: list[int] | None = None,
) -> None:
    """
    Run the simulation.

    Parameters:
        - data: is a Data object which is previously loaded using load_data(),
        - base_year: optional argument base year for the simulation,
        - target_year: optional target year for the simulation,
        - step size: optional argument to specify the increments in which solves are run in the simulation,
        - years: optional argument to specify the years for which the simulation is run.

    Some Functionality Notes:
        - If neither 'step size' nor 'years' are provided, the simulation is run sequentially from base to target year.
        - 'Years' essentially replaces base_year, target_year and step_size to allow for more flexibility so these
          should not be provided alongside 'years'.
    """
    validate_simulation_years_settings(data.YR_CAL_BASE, base_year, target_year, step_size, years)

    if not step_size:
        step_size = 1

    if base_year != data.YR_CAL_BASE:
        populate_containers_dynamic_base_year(data, base_year)

    memory_thread = threading.Thread(target=log_memory_usage, daemon=True)
    memory_thread.start()
    
    # Set Data object's path and create output directories
    data.set_path(base_year, target_year, step_size, years)

    # Run the simulation up to `year` sequentially.
    if settings.MODE == 'timeseries':
        if len(data.D_CY.shape) != 2:
            raise ValueError( "Demands need to be a time series array of shape (years, commodities) and years > 0." )
        if target_year - base_year > data.D_CY.shape[0]:
            raise ValueError( "Not enough years in demands time series.")

        if years:
            years_to_run = years
        else:
            years_to_run = list(range(base_year, target_year + 1, step_size))
            if target_year not in years_to_run:
                years_to_run.append(target_year)

        solve_timeseries(data, years_to_run)

    elif settings.MODE == 'snapshot':
        # If demands is a time series, choose the appropriate entry.
        solve_snapshot(data, base_year, target_year)

    else:
        raise ValueError(f"Unkown MODE: {settings.MODE}.")
    
    # Save the Data object to disk
    write_outputs(data)


def validate_simulation_years_settings(
    data_yr_cal_base: int,
    base_year: int | None = None, 
    target_year: int | None = None, 
    step_size: int | None = None,
    years: list[int] | None = None,
) -> None:
    """
    Validates the years settings given to run a simulation.
    Raises errors or warnings if the settings don't adhere to the following rules:
    - if 'years' list is provided, it must have at least two years in it (the first year is the solve's base year)
    - if 'years' list is provided, all other years settings should not be specified 
    - if 'years' list is provided, it must be in ascending order
    - the 'years' list cannot be specified for snapshot solves
    - 'step_size' argument will be ignored for snapshot solves
    - 'base_year' must be greater than 2010 and less than 'target_year'
    """
    if years:
        if len(years) < 2:
            raise ValueError("'years' must be populated with at least two years (inclusive of a base year).")

        if step_size or base_year or target_year:
            raise ValueError("'years' cannot be specified with step_size, base_year or target_year.")
        
        if years != sorted(years):
            raise ValueError("'years' must be in ascending order.")
        
        if settings.MODE == 'snapshot':
            raise ValueError("'years' cannot be specified for snapshot solves.")

    if settings.MODE == 'snapshot' and step_size:
        print(f"WARNING: Step size: {step_size} is ignored when settings.MODE is 'snapshot'.")

    if not years:
        if not base_year or not target_year:
            raise ValueError("Both base and target years, or 'years' argument, must be specified.")

    if base_year < data_yr_cal_base or base_year >= target_year:
        raise ValueError(f"Base year must be >= {data_yr_cal_base} and less than the target year.")


def populate_containers_dynamic_base_year(
    data: Data,
    base_year: int,
) -> None:
    """
    If the base year parsed to run() is not the same as data.YR_CAL_BASE
    then
        - Calculate and add data for the base year for containers. If this is not possible,
        - Copy the data from the most recent year to the base year container.
    """
    year_before_base_year = max([y for y in list(data.lumaps.keys()) if y < base_year])

    data.lumaps[base_year] = data.lumaps[year_before_base_year]
    data.lmmaps[base_year] = data.lmmaps[year_before_base_year]
    data.ammaps[base_year] = data.ammaps[year_before_base_year]
    data.ag_dvars[base_year] = data.ag_dvars[year_before_base_year]
    data.non_ag_dvars[base_year] = data.non_ag_dvars[year_before_base_year]
    data.ag_man_dvars[base_year] = data.ag_man_dvars[year_before_base_year]

    # Populate production containers
    if year_before_base_year == data.YR_CAL_BASE:
        ghg_emission_data = (
            ag_ghg.get_ghg_matrices(data, 0, aggregate=True) 
            * data.ag_dvars[data.YR_CAL_BASE]
        ).sum()
        biodiversity_data = (ag_biodiversity.get_breq_matrices(data) * data.ag_dvars[data.YR_CAL_BASE]).sum()
        major_vegetation_data = calc_major_vegetation_group_ag_area_for_year(
            get_major_vegetation_matrices(data), 
            data.lumaps[data.YR_CAL_BASE],
            data.BIODIV_HABITAT_DEGRADE_LOOK_UP,
        )
        species_conservation_data = calc_species_ag_area_for_year(
            ag_biodiversity.get_species_conservation_matrix(data, data.YR_CAL_BASE),
            data.lumaps[year_before_base_year],
            data.BIODIV_HABITAT_DEGRADE_LOOK_UP,
        )
        
    else:
        ghg_emission_data = data.prod_data[year_before_base_year]["GHG Emissions"]
        biodiversity_data = data.prod_data[year_before_base_year]["Biodiversity"]
        major_vegetation_data = data.prod_data[year_before_base_year]["Major Vegetation Groups"]
        species_conservation_data = data.prod_data[year_before_base_year]["Species Conservation"]

    data.add_production_data(base_year, "Production", data.prod_data[year_before_base_year]["Production"])
    data.add_production_data(base_year, "GHG Emissions", ghg_emission_data)
    data.add_production_data(base_year, "Biodiversity", biodiversity_data)
    data.add_production_data(base_year, "Major Vegetation Groups", major_vegetation_data)
    data.add_production_data(base_year, "Species Conservation", species_conservation_data)    


def solve_timeseries(data: Data, years_to_run: list[int]) -> None:
    print('\n')
    print(f"Running LUTO {settings.VERSION} timeseries from {years_to_run[0]} to {years_to_run[-1]} at resfactor {settings.RESFACTOR}.", flush=True)

    final_year = years_to_run[-1]

    for s in range(len(years_to_run) - 1):
        base_year = years_to_run[s]
        target_year = years_to_run[s + 1]

        print( "-------------------------------------------------")
        print( f"Running for year {target_year}"   )
        print( "-------------------------------------------------\n" )
        start_time = time.time()

        input_data = get_input_data(data, base_year, target_year)
        d_c = data.D_CY[s + 1]

        if s == 0:
            luto_solver = LutoSolver(input_data, d_c, final_year)
            luto_solver.formulate()

        if s > 0:
            prev_base_year = years_to_run[s - 1]

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
                old_lumap=data.lumaps[prev_base_year],
                current_lumap=data.lumaps[base_year],
                old_lmmap=data.lmmaps[prev_base_year],
                current_lmmap=data.lmmaps[base_year],
            )

        solution = luto_solver.solve()

        yr = target_year
        data.add_lumap(yr, solution.lumap)
        data.add_lmmap(yr, solution.lmmap)
        data.add_ammaps(yr, solution.ammaps)
        data.add_ag_dvars(yr, solution.ag_X_mrj)
        data.add_non_ag_dvars(yr, solution.non_ag_X_rk)
        data.add_ag_man_dvars(yr, solution.ag_man_X_mrj)
        data.add_obj_vals(yr, solution.obj_val)


        for data_type, prod_data in solution.prod_data.items():
            data.add_production_data(yr, data_type, prod_data)

        print(f'Processing for {target_year} completed in {round(time.time() - start_time)} seconds\n\n' )


def solve_snapshot(data: Data, base_year: int, target_year: int):
    if len(data.D_CY.shape) == 2:
        d_c = data.D_CY[target_year - data.YR_CAL_BASE]       # Demands needs to be a timeseries from 2010 to target year
    else:
        d_c = data.D_CY

    print('\n')
    print(f"Running LUTO {settings.VERSION} snapshot for {target_year} at resfactor {settings.RESFACTOR}")
    print("-------------------------------------------------")
    print(f"Running for year {target_year}")
    print("-------------------------------------------------")

    start_time = time.time()
    input_data = get_input_data(data, base_year, target_year)
    luto_solver = LutoSolver(input_data, d_c, target_year)
    luto_solver.formulate()

    solution = luto_solver.solve()
    data.add_lumap(target_year, solution.lumap)
    data.add_lmmap(target_year, solution.lmmap)
    data.add_ammaps(target_year, solution.ammaps)
    data.add_ag_dvars(target_year, solution.ag_X_mrj)
    data.add_non_ag_dvars(target_year, solution.non_ag_X_rk)
    data.add_ag_man_dvars(target_year, solution.ag_man_X_mrj)
    data.add_obj_vals(target_year, solution.obj_val)

    for data_type, prod_data in solution.prod_data.items():
        data.add_production_data(target_year, data_type, prod_data)

    print(f'Processing for {target_year} completed in {round(time.time() - start_time)} seconds\n\n')


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
  