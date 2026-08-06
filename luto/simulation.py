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

import os
import re
import time
import threading
from pathlib import Path

import numpy as np
import pandas as pd

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
    read_timestamp,
    record_shadow_prices,
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
    data: Data | None = None,
    do_analyze_iis: bool = settings.DO_IIS,
    do_report: bool = settings.WRITE_OUTPUTS,
    checkpoint_dir: str | None = None,
) -> Data:
    """
    Run the simulation.

    Parameters
    ----------
    data : Data or None
        Loaded simulation data. Must be provided; if a checkpoint exists it will
        replace this object internally, but a freshly loaded Data is still needed
        to determine the year sequence and base year.
    do_analyze_iis : bool, default False
        If True, infeasible per-year solves trigger ``computeIIS()`` +
        ``analyze_iis()`` and write a debug .ilp file alongside the run output.
        Task runs typically pass ``settings.DO_IIS`` so this can be controlled
        as a grid_search parameter.
    do_report : bool, default True
        If True, write outputs at the end of the run. Set to False to skip output
        writing (e.g. when doing a quick test run or debugging IIS infeasibility).
    checkpoint_dir : str or None, default None
        If provided, enables checkpoint mode. After each successfully solved year
        a ``data_<year>.lz4`` file is written to this directory. On re-run, the
        latest valid checkpoint is loaded and the simulation resumes from the
        next unsolved year. Useful for long NCI jobs that may be wall-time killed.
    """
    
    if (data is None) and (checkpoint_dir is None):
        raise ValueError("Either `data` must be provided or `checkpoint_dir` must be set to enable checkpoint loading.")

    # Generate new timestamp each time and apply decorator dynamically
    current_timestamp = read_timestamp()
    save_dir = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"
    log_path = f"{save_dir}/LUTO_RUN_"

    # Apply the LogToFile decorator dynamically
    @LogToFile(log_path)
    def _run():

        years = sorted(settings.SIM_YEARS).copy()

        # Use active_data to avoid Python closure scoping issues: assigning to
        # `data` inside a nested function would make Python treat it as local
        # throughout _run(), causing UnboundLocalError on non-checkpoint paths.
        active_data = data
        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
        resume_from_year = None

        if checkpoint_path is not None:
            print(f"Checkpoint mode enabled: {checkpoint_path}")
            files = sorted(f for f in checkpoint_path.iterdir() if re.match(r'data_\d{4}\.lz4', f.name))
            if files:
                checkpoint_file = files[-1]
                resume_from_year = int(checkpoint_file.stem.split("_")[1])
                active_data = joblib.load(str(checkpoint_file))
                active_data.timestamp = read_timestamp()
                active_data.path = save_dir
                # load_data_from_disk()'s set_path() normally pre-creates out_<yr> dirs
                # for write_outputs; checkpoint resume bypasses that, so do it here too.
                set_path()
                print(f"Resuming from checkpoint (year {resume_from_year}): {checkpoint_file}")
            elif data is None:
                raise ValueError(
                    f"No checkpoint files found in '{checkpoint_path}' and no `data` was provided; "
                    "cannot start simulation."
                )
            else:
                print(f"No valid checkpoint found in '{checkpoint_path}'; starting from {years[0]}.")

        # active_data is guaranteed non-None from here on
        if active_data.YR_CAL_BASE not in years:
            years.insert(0, active_data.YR_CAL_BASE)

        if resume_from_year is not None:
            years_to_run = years[years.index(resume_from_year):]
            print(f"Resuming simulation from {resume_from_year} to {years[-1]}.")
        else:
            years_to_run = years

        # Start recording memory usage
        stop_event = threading.Event()
        memory_thread = threading.Thread(target=log_memory_usage, args=(save_dir, 'a', 1, stop_event))
        memory_thread.start()

        try:
            print('\n')
            print(f"Running LUTO {settings.VERSION} between {years[0]} - {years[-1]} at RES-{settings.RESFACTOR}, total {len(years) - 1} runs!\n", flush=True)

            if len(years_to_run) > 1:
                solve_timeseries(active_data, years_to_run, do_analyze_iis, Path(save_dir))

            # Save final data and write outputs
            save_data_to_disk(active_data, f"{save_dir}/Data_RES{settings.RESFACTOR}.lz4")
            if do_report:
                write_outputs(active_data)
        except Exception as e:
            print(f"An error occurred during the simulation: {e}")
            raise e
        finally:
            # Ensure the memory logging thread is stopped
            stop_event.set()
            memory_thread.join()

        return active_data

    return _run()


def solve_timeseries(
    data: Data,
    years_to_run: list[int],
    do_analyze_iis: bool,
    checkpoint_path: Path | None = None,
) -> None:

    # Save the base-year state before any solving so a retry can re-attempt the
    # first target year. Skipped on resume (file already exists from a prior run).
    if checkpoint_path is not None:
        base_ckpt = checkpoint_path / f"data_{years_to_run[0]}.lz4"
        if not base_ckpt.exists():
            save_data_to_disk(data, str(base_ckpt))
            print(f"Saved base checkpoint for year {years_to_run[0]}: {base_ckpt}")

    for step in range(len(years_to_run) - 1):
        base_year = years_to_run[step]
        target_year = years_to_run[step + 1]

        print( "-------------------------------------------------")
        print( f"Running for year {target_year}"   )
        print( "-------------------------------------------------\n")

        start_time = time.time()
        input_data = get_input_data(data, base_year, target_year)
        data.last_year = target_year

        # Retry loop with escalating numerical settings. settings.RETRY_PARAMS is a list
        # of (NumericFocus, Method, Crossover) tuples tried in order; only GRB.OPTIMAL
        # is accepted — any other status falls through to the failure path.
        nf_attempts = list(settings.RETRY_PARAMS)
        accepted = False
        solution = None
        status = None
        luto_solver = LutoSolver(input_data)
        luto_solver.formulate()

        # Persist any biodiversity rows proven unreachable at build time. Written BEFORE the solve on
        # purpose: the record is most valuable exactly when the year goes on to fail, and the
        # `accepted` guard below would throw it away.
        if luto_solver.bio_unreachable_constrs:
            out_dir = f"{data.path}/out_{target_year}"
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(luto_solver.bio_unreachable_constrs).to_csv(
                f"{out_dir}/unreachable_bio_constraints_{target_year}.csv", index=False)

        for nf, method, crossover, presolve, barhomogenous in nf_attempts:
            print(f"Trying NumericFocus={nf}, Method={method}, Crossover={crossover}, Presolve={presolve}, BarHomogeneous={barhomogenous} for year {target_year}...")
            luto_solver.gurobi_model.Params.NumericFocus    = nf
            luto_solver.gurobi_model.Params.Method          = method
            luto_solver.gurobi_model.Params.Crossover       = crossover
            luto_solver.gurobi_model.Params.Presolve        = presolve
            luto_solver.gurobi_model.Params.BarHomogeneous  = barhomogenous
            solution = luto_solver.solve()
            status = luto_solver.gurobi_model.Status

            if solution is not None and status == GRB.OPTIMAL:
                print(f"Optimal solution found with NumericFocus={nf}, Method={method}")
                accepted = True
                break

            print(f"Non-optimal status {status} with NumericFocus={nf}, Method={method}; retrying with next attempt if available.")

        # Only commit results when a solve was accepted. `solution` is None whenever Gurobi
        # stored no solution (SolCount == 0), so these writes MUST stay inside the guard —
        # outside it they would raise on None and re-strand the diagnostics below, which is
        # the bug this guard exists to prevent.
        if accepted:
            data.add_lumap(target_year, solution.lumap)
            data.add_lmmap(target_year, solution.lmmap)
            data.add_ammaps(target_year, solution.ammaps)
            data.add_ag_dvars(target_year, solution.ag_X_mrj)
            data.add_delta_dvars_ag2ag(target_year, solution.dvar_D_ag2ag_mrj)
            data.add_non_ag_dvars(target_year, solution.non_ag_X_rk)
            data.add_delta_dvars_ag2nonag(target_year, solution.dvar_D_ag2nonag_rk)
            data.add_delta_dvars_nonag2ag(target_year, solution.dvar_D_nonag2ag_mrj)
            data.add_ag_man_dvars(target_year, solution.ag_man_X_mrj)
            data.add_obj_vals(target_year, solution.obj_val)

            for data_type, prod_data in solution.prod_data.items():
                data.add_production_data(target_year, data_type, prod_data)

            record_shadow_prices(luto_solver, input_data, target_year, f"{data.path}/out_{target_year}")

        if checkpoint_path is not None and accepted:
            final_path = checkpoint_path / f"data_{target_year}.lz4"
            save_data_to_disk(data, str(final_path))
            for old in checkpoint_path.iterdir():
                if re.match(r'data_\d{4}\.lz4', old.name) and old != final_path:
                    old.unlink()
            print(f"Saved checkpoint for year {target_year}: {final_path}")

        print(f'Processing for {target_year} completed in {round(time.time() - start_time)} seconds\n\n' )

        if not accepted:
            print('!' * 100)

            status_msgs = {
                GRB.INFEASIBLE:  "INFEASIBLE",
                GRB.INF_OR_UNBD: "INFEASIBLE OR UNBOUNDED — set `BARHOMOGENOUS`=1 to distinguish",
                GRB.UNBOUNDED:   "UNBOUNDED — check objective coefficients and variable bounds",
                GRB.NUMERIC:     "NUMERICAL ISSUES — consider adjusting tolerances or `NumericFocus`",
                GRB.SUBOPTIMAL:  "SUBOPTIMAL — constraints may not be fully satisfied",
            }
            print(f"Solver status for year {target_year}: {status_msgs.get(status, f'unexpected status {status}')}")

            if status == GRB.INFEASIBLE:
                model_path = f"{data.path}/debug_model_{base_year}_{target_year}.mps"
                luto_solver.gurobi_model.write(model_path)
                print(f"Saved model to {model_path}")
                if do_analyze_iis:
                    print("Computing IIS...")
                    luto_solver.gurobi_model.computeIIS()
                    iis_path = f"{data.path}/debug_model_{base_year}_{target_year}.ilp"
                    luto_solver.gurobi_model.write(iis_path)
                    print(f"IIS saved to {iis_path}")
                    analyze_iis(iis_path, data)

            print('!' * 100)
            print('\n')
            break



def save_data_to_disk(data: Data, path: str, compress_level=3) -> None:
    """Save using joblib with atomic tmp→rename to prevent partial writes."""
    print(f'Saving data to {path}...')
    tmp = Path(f"{path}.tmp")
    joblib.dump(data, str(tmp), compress=('lz4', compress_level))
    # Write to .tmp first, then rename atomically (os.replace → POSIX rename()).
    # If the job is killed mid-write, only the .tmp is left partial; the final
    # .lz4 is never created until the write completes successfully.
    os.replace(tmp, path)


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
