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

import time
import threading

from gurobipy import GRB
import joblib
from luto import settings
from luto.data import Data
from luto.solvers.input_data import get_input_data
from luto.solvers.solver import LutoSolver
from luto.tools.inspect_iis import analyze_iis
from luto.tools.write import write_outputs
from luto.tools import (
    LogToFile,
    log_memory_usage,
    set_path,
    write_timestamp,
    read_timestamp
)




def load_data() -> Data:
    """
    Load the Data object containing all required data to run a LUTO simulation.
    """
    
    # Generate new timestamp each time and apply decorator dynamically
    current_timestamp = write_timestamp()
    save_dir = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"
    log_path = f"{save_dir}/LUTO_RUN_"
    set_path()
    
    # Apply the LogToFile decorator dynamically
    @LogToFile(log_path)
    def _load_data():
        # Thread to log memory usage
        stop_event = threading.Event()
        memory_thread = threading.Thread(target=log_memory_usage, args=(save_dir, 'w', 1, stop_event))
        memory_thread.start()
        
        try:
            data = Data()
            data.timestamp = read_timestamp()
            data.path = save_dir
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            raise e
        finally:
            # Ensure the memory logging thread is stopped
            stop_event.set()
            memory_thread.join()

        return data
    
    return _load_data()


def run(
    data: Data, 
) -> None:
    """
    Run the simulation.
    """
    
    # Generate new timestamp each time and apply decorator dynamically
    current_timestamp = read_timestamp()
    save_dir = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"
    log_path = f"{save_dir}/LUTO_RUN_"
    
    # Apply the LogToFile decorator dynamically
    @LogToFile(log_path)
    def _run():
        # Get the years to run
        years = sorted(settings.SIM_YEARS).copy()
        
        # Start recording memory usage
        stop_event = threading.Event()
        memory_thread = threading.Thread(target=log_memory_usage, args=(save_dir, 'a', 1, stop_event))
        memory_thread.start()
        
        try:
            print('\n')
            print(f"Running LUTO {settings.VERSION} between {years[0]} - {years[-1]} at RES-{settings.RESFACTOR}, total {len(years)} runs!\n", flush=True)
            # Insert the base year at the beginning of the years list if not already present
            if data.YR_CAL_BASE not in years: 
                years.insert(0, data.YR_CAL_BASE)
            # Solve and write outputs
            solve_timeseries(data, years)
            save_data_to_disk(data, f"{save_dir}/Data_RES{settings.RESFACTOR}.lz4")
            write_outputs(data)
        except Exception as e:
            print(f"An error occurred during the simulation: {e}")
            raise e
        finally:
            # Ensure the memory logging thread is stopped
            stop_event.set()
            memory_thread.join()
    
    return _run()


def solve_timeseries(data: Data, years_to_run: list[int]) -> None:

    for step in range(len(years_to_run) - 1):
        base_year = years_to_run[step]
        target_year = years_to_run[step + 1]

        print( "-------------------------------------------------")
        print( f"Running for year {target_year}"   )
        print( "-------------------------------------------------\n")
        
        start_time = time.time()
        input_data = get_input_data(data, base_year, target_year)

        # if step == 0:
        #     luto_solver = LutoSolver(input_data, d_c)
        #     luto_solver.formulate()

        # if step > 0:
        #     prev_base_year = years_to_run[step - 1]

        #     old_ag_x_mrj = luto_solver._input_data.ag_x_mrj.copy()
        #     old_ag_man_lb_mrj = luto_solver._input_data.ag_man_lb_mrj.copy()
        #     old_non_ag_x_rk = luto_solver._input_data.non_ag_x_rk.copy()
        #     old_non_ag_lb_rk = luto_solver._input_data.non_ag_lb_rk.copy()

        #     luto_solver.update_formulation(
        #         input_data=input_data,
        #         d_c=d_c,
        #         old_ag_x_mrj=old_ag_x_mrj,
        #         old_ag_man_lb_mrj=old_ag_man_lb_mrj,
        #         old_non_ag_x_rk=old_non_ag_x_rk,
        #         old_non_ag_lb_rk=old_non_ag_lb_rk,
        #         old_lumap=data.lumaps[prev_base_year],
        #         current_lumap=data.lumaps[base_year],
        #         old_lmmap=data.lmmaps[prev_base_year],
        #         current_lmmap=data.lmmaps[base_year],
        #     )

        luto_solver = LutoSolver(input_data)
        luto_solver.formulate()
        solution = luto_solver.solve()
        
        data.last_year = target_year 

        data.add_lumap(target_year, solution.lumap)
        data.add_lmmap(target_year, solution.lmmap)
        data.add_ammaps(target_year, solution.ammaps)
        data.add_ag_dvars(target_year, solution.ag_X_mrj)
        data.add_non_ag_dvars(target_year, solution.non_ag_X_rk)
        data.add_ag_man_dvars(target_year, solution.ag_man_X_mrj)
        data.add_obj_vals(target_year, solution.obj_val)

        for data_type, prod_data in solution.prod_data.items():
            data.add_production_data(target_year, data_type, prod_data)
            

        print(f'Processing for {target_year} completed in {round(time.time() - start_time)} seconds\n\n' )
        
        if luto_solver.gurobi_model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            print('!' * 100)
            print(f"Warning: Gurobi solver did not find an optimal/suboptimal solution for year {target_year}. Status: {luto_solver.gurobi_model.Status}")
            print(f'Warning: The results are still written to disk, but will not be optimal.')
            print('!' * 100)

            # Save model and compute IIS for debugging
            model_path = f"{data.path}/debug_model_{base_year}_{target_year}.mps"
            luto_solver.gurobi_model.write(model_path)
            print(f"Saved Gurobi model to {model_path}")

            if luto_solver.gurobi_model.Status == GRB.INFEASIBLE:
                print("Computing IIS (Irreducible Inconsistent Subsystem)...")
                luto_solver.gurobi_model.computeIIS()
                iis_path = f"{data.path}/debug_model_{base_year}_{target_year}.ilp"
                luto_solver.gurobi_model.write(iis_path)
                print(f"Analyzed IIS and saved to {iis_path}")
                analyze_iis(iis_path, data)

            print('\n')
            break



def save_data_to_disk(data: Data, path: str, compress_level=3) -> None:
    """Save using joblib - faster and more memory efficient."""
    print(f'Saving data to {path}...')
    
    # joblib is optimized for large numpy/scipy data
    joblib.dump(data, path, compress=('lz4', compress_level))
    
    
def load_data_from_disk(path: str) -> Data:
    """Load the Data object from disk.

    Arguments:
        path: Path to the Data object.

    Raises:
        ValueError: if the resolution factor from the data object does not match the settings.RESFACTOR.

    Returns
        Data: `Data` object.
    """
    
    # Generate new timestamp each time and apply decorator dynamically
    current_timestamp = write_timestamp()
    save_dir = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"
    log_path = f"{save_dir}/LUTO_RUN_"
    
    set_path()
    
    # Apply the LogToFile decorator dynamically
    @LogToFile(log_path, 'w')
    def _load_data():
        print(f"Loading data from {path}...\n")

        # Load joblib-compressed file
        data = joblib.load(path)
        data.timestamp = read_timestamp()
        data.path = save_dir

        # Check if the resolution factor from the data object matches the settings.RESFACTOR
        if int(data.RESMULT ** 0.5) != settings.RESFACTOR:
            raise ValueError(f'Resolution factor from data loading ({int(data.RESMULT ** 0.5)}) does not match it of settings ({settings.RESFACTOR})!')
        
        return data
    
    return _load_data()

