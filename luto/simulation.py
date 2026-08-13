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
import joblib
import pandas as pd

from contextlib import contextmanager
from pathlib import Path
from gurobipy import GRB

from luto import settings
from luto.data import Data
from luto.solvers.input_data import get_input_data
from luto.solvers.solver import LutoSolver
from luto.solvers.tools import feasibility_spectrum, resolve_infeasibility, group_of
from luto.tools.write import write_outputs
from luto.tools import (
    LogToFile,
    log_memory_usage,
    set_path,
    write_timestamp,
    read_timestamp,
    record_shadow_prices,
)


# Checkpoint files are named data_<year>.lz4. The same pattern drives both resume
# discovery and stale-checkpoint rotation.
CHECKPOINT_RE = re.compile(r'data_\d{4}\.lz4')

# Human-readable meaning of each non-optimal Gurobi status, for the failure banner.
SOLVER_STATUS_MSGS = {
    GRB.INFEASIBLE:  "INFEASIBLE",
    GRB.INF_OR_UNBD: "INFEASIBLE OR UNBOUNDED — set `BARHOMOGENOUS`=1 to distinguish",
    GRB.UNBOUNDED:   "UNBOUNDED — check objective coefficients and variable bounds",
    GRB.NUMERIC:     "NUMERICAL ISSUES — consider adjusting tolerances or `NumericFocus`",
    GRB.SUBOPTIMAL:  "SUBOPTIMAL — constraints may not be fully satisfied",
}


def _run_dir(timestamp: str) -> str:
    """Timestamped output directory that holds every artefact of one run."""
    return (f"{settings.OUTPUT_DIR}/{timestamp}"
            f"_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}")


@contextmanager
def _memory_log(save_dir: str, mode: str):
    """Sample process memory to <save_dir>/RES_*_mem_log.txt while the block runs."""
    stop_event = threading.Event()
    thread = threading.Thread(target=log_memory_usage, args=(save_dir, mode, 1, stop_event))
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()


# ---------------------------------------------------------------------------- #
# Entry points                                                                 #
# ---------------------------------------------------------------------------- #

def load_data() -> Data:
    """
    Load the Data object containing all required data to run a LUTO simulation.
    """
    save_dir = _run_dir(write_timestamp())    # new timestamp: this starts a new run
    set_path()

    @LogToFile(f"{save_dir}/LUTO_RUN_")
    def _load_data():
        # The explicit print puts the failure in the tee'd log; the traceback itself
        # only surfaces after LogToFile has detached from stdout/stderr.
        try:
            with _memory_log(save_dir, 'w'):
                data = Data()
                data.timestamp = read_timestamp()
                data.path = save_dir
                return data
        except Exception as e:
            print(f"An error occurred while loading data: {e}", flush=True)
            raise

    return _load_data()


def run(
    data: Data | None = None,
    do_report: bool = settings.WRITE_OUTPUTS,
    checkpoint_dir: str | None = None,
) -> Data:
    """
    Run the simulation.

    Parameters
    ----------
    data : Data or None
        Loaded simulation data. Required unless `checkpoint_dir` holds a checkpoint
        to resume from, in which case the checkpointed Data replaces it.
    do_report : bool, default True
        If True, write outputs at the end of the run. Set to False to skip output
        writing (e.g. when doing a quick test run or debugging IIS infeasibility).
    checkpoint_dir : str or None, default None
        Where to LOOK for a ``data_<year>.lz4`` checkpoint to resume from; the run
        continues at the first unsolved year. Checkpoints themselves are always
        written into the run's own output directory (see `solve_timeseries`) — on a
        resumed run the reused timestamp makes that the same directory. Useful for
        long NCI jobs that may be wall-time killed.
    """
    if data is None and checkpoint_dir is None:
        raise ValueError("Either `data` must be provided or `checkpoint_dir` must be set to enable checkpoint loading.")

    save_dir = _run_dir(read_timestamp())    # reuse the timestamp written at load time

    @LogToFile(f"{save_dir}/LUTO_RUN_")
    def _run(active_data: Data | None) -> Data:
        resume_year = None
        if checkpoint_dir is not None:
            active_data, resume_year = load_latest_checkpoint(Path(checkpoint_dir), active_data, save_dir)

        years = sorted(settings.SIM_YEARS)
        if active_data.YR_CAL_BASE not in years:
            years.insert(0, active_data.YR_CAL_BASE)

        if resume_year is not None:
            years_to_run = years[years.index(resume_year):]
            print(f"Resuming simulation from {resume_year} to {years[-1]}.", flush=True)
        else:
            years_to_run = years

        try:
            with _memory_log(save_dir, 'a'):
                print('\n', flush=True)
                print(f"Running LUTO {settings.VERSION} between {years[0]} - {years[-1]} at RES-{settings.RESFACTOR}, total {len(years) - 1} runs!\n", flush=True)

                if len(years_to_run) > 1:
                    solve_timeseries(active_data, years_to_run, Path(save_dir))

                save_data_to_disk(active_data, f"{save_dir}/Data_RES{settings.RESFACTOR}.lz4")
                if do_report:
                    write_outputs(active_data)
        except Exception as e:
            print(f"An error occurred during the simulation: {e}", flush=True)
            raise

        return active_data

    return _run(data)


def load_data_from_disk(path: str) -> Data:
    """Load the Data object from disk.

    Arguments:
        path: Path to the Data object.

    Raises:
        ValueError: if the resolution factor from the data object does not match the settings.RESFACTOR.

    Returns
        Data: `Data` object.
    """
    save_dir = _run_dir(write_timestamp())    # new timestamp: this starts a new run
    set_path()

    @LogToFile(f"{save_dir}/LUTO_RUN_", 'w')
    def _load_data():
        print(f"Loading data from {path}...\n", flush=True)

        data = joblib.load(path)
        data.timestamp = read_timestamp()
        data.path = save_dir

        if int(data.RESMULT ** 0.5) != settings.RESFACTOR:
            raise ValueError(f'Resolution factor from data loading ({int(data.RESMULT ** 0.5)}) does not match it of settings ({settings.RESFACTOR})!')

        return data

    return _load_data()


# ---------------------------------------------------------------------------- #
# Time-series solve                                                            #
# ---------------------------------------------------------------------------- #

def solve_timeseries(
    data: Data,
    years_to_run: list[int],
    checkpoint_path: Path | None = None,
) -> None:
    """Solve each consecutive (base, target) year pair, checkpointing after each success.

    Stops at the first year that cannot be solved. `checkpoint_path` is where the
    ``data_<year>.lz4`` checkpoints are written (the run's own output directory when
    called from `run()`); pass None to disable checkpointing.
    """
    # Save the base-year state before any solving so a retry can re-attempt the
    # first target year. Skipped on resume (file already exists from a prior run).
    if checkpoint_path is not None:
        base_ckpt = checkpoint_path / f"data_{years_to_run[0]}.lz4"
        if not base_ckpt.exists():
            save_data_to_disk(data, str(base_ckpt))
            print(f"Saved base checkpoint for year {years_to_run[0]}: {base_ckpt}", flush=True)

    for base_year, target_year in zip(years_to_run[:-1], years_to_run[1:]):
        print( "-------------------------------------------------", flush=True)
        print( f"Running for year {target_year}"   , flush=True)
        print( "-------------------------------------------------\n", flush=True)

        start_time = time.time()
        input_data = get_input_data(data, base_year, target_year)
        data.last_year = target_year

        luto_solver = LutoSolver(input_data)
        luto_solver.formulate()

        # Save the model to disk BEFORE solving (see save_model_to_disk for why).
        save_model_to_disk(luto_solver.gurobi_model, data.path, base_year, target_year)
        # Drop constraints that cannot hold even alone, BEFORE solving. Dropped rows
        # are recorded in out_<year>/dropped_constraints_<year>.csv.
        drop_unreachable_before_solve(luto_solver, data, target_year)

        accepted, solution, status = solve_with_retries(luto_solver, data, target_year)

        if accepted:
            store_solution(data, target_year, solution)
            record_shadow_prices(luto_solver, input_data, target_year, f"{data.path}/out_{target_year}")
            if checkpoint_path is not None:
                save_checkpoint(data, checkpoint_path, target_year)

        print(f'Processing for {target_year} completed in {round(time.time() - start_time)} seconds\n\n' , flush=True)

        if not accepted:
            print('!' * 100, flush=True)
            print(f"Solver status for year {target_year}: {SOLVER_STATUS_MSGS.get(status, f'unexpected status {status}')}", flush=True)
            print('!' * 100, flush=True)
            print('\n', flush=True)
            break


def solve_with_retries(luto_solver: LutoSolver, data: Data, target_year: int):
    """Run the RETRY_PARAMS ladder against the current model. Returns (accepted, solution, status).

    settings.RETRY_PARAMS is a list of (NumericFocus, Method, Crossover, Presolve,
    BarHomogeneous) tuples tried in order; only GRB.OPTIMAL is accepted.

    A failed attempt is treated as a CONFLICT first and a numerical problem second: diagnose
    and drop, then re-solve with the same configuration. Only when nothing more can be
    dropped does the next RETRY_PARAMS entry get tried.

    That ordering matters in wall-clock. Falling straight through to the next configuration
    sends a genuinely infeasible model into the dual-simplex rung, which has been measured
    diverging for 35 min+ without terminating — so the diagnosis that would have explained
    the failure in minutes never gets reached. Diagnosing first costs one restricted IIS.
    """
    accepted, solution, status = False, None, None
    for params in settings.RETRY_PARAMS:
        accepted, solution, status = solve_attempt(luto_solver, target_year, *params)
        while not accepted and diagnose_and_drop_conflict(luto_solver, data, target_year):
            accepted, solution, status = solve_attempt(luto_solver, target_year, *params)
        if accepted:
            break
    return accepted, solution, status


def solve_attempt(luto_solver, target_year, nf, method, crossover, presolve, barhomogenous):
    """One RETRY_PARAMS attempt against the current model. Returns (accepted, solution, status)."""
    print(f"Trying NumericFocus={nf}, Method={method}, Crossover={crossover}, Presolve={presolve}, BarHomogeneous={barhomogenous} for year {target_year}...", flush=True)
    luto_solver.gurobi_model.Params.NumericFocus    = nf
    luto_solver.gurobi_model.Params.Method          = method
    luto_solver.gurobi_model.Params.Crossover       = crossover
    luto_solver.gurobi_model.Params.Presolve        = presolve
    luto_solver.gurobi_model.Params.BarHomogeneous  = barhomogenous

    solution = luto_solver.solve()
    status = luto_solver.gurobi_model.Status
    if solution is not None and status == GRB.OPTIMAL:
        print(f"Optimal solution found with NumericFocus={nf}, Method={method}", flush=True)
        return True, solution, status

    print(f"Non-optimal status {status} with NumericFocus={nf}, Method={method}; retrying with next attempt if available.", flush=True)
    return False, solution, status


def store_solution(data: Data, target_year: int, solution) -> None:
    """Copy the accepted solver solution into the Data singleton."""
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


# ---------------------------------------------------------------------------- #
# Infeasibility handling                                                       #
# ---------------------------------------------------------------------------- #

def drop_unreachable_before_solve(luto_solver: LutoSolver, data: Data, target_year: int) -> list:
    """One pre-solve feasibility SPECTRUM: provable infeasibility and knife-edge thinness are the
    same question at different tightenings, asked on one shared probe copy (tools.py).

        eps = 0      an IIS is a PROOF — the least-valued droppable row is surrendered per round
                     until feasible. Catches rows impossible alone (NE Buloke) AND joint conflicts
                     (SNES × cap) before any production rung runs. Termination guarantee, not an
                     optimisation: a jointly-infeasible model can send a rung into `Numerical
                     trouble` → Gurobi's internal simplex fallback → divergence that never
                     terminates, so the post-failure IIS is unreachable (R2_SNES_T1525_cap10,
                     2026-08-09, deterministic to the digit).
        eps > 0      rows with relative headroom below 1e-6/1e-4/1e-2. Below
                     KNIFE_EDGE_DROP_BELOW (droppable groups only) they are removed — inside that
                     margin the production solve cannot distinguish them from infeasible, and both
                     observed stall classes trace to exactly such rows. The 1e-2 band is recorded
                     as early warning (lock-in ratchets: under-1% today is thinner next year).

    Non-droppable groups (the cap, GBF2, ...) are NEVER removed however thin — when the cap
    itself is the thin row, the IIS names its droppable partner, and dropping the partner
    relieves the edge. Analysis note: filter dropped_constraints CSVs on `action` — 'DROPPED'
    (proven) and 'DROPPED_KNIFE_EDGE' (inside numerical noise, margin in `headroom_lt`) left the
    model; 'KNIFE_EDGE' rows stayed in.
    """
    if not (settings.DROP_UNREACHABLE_CONSTRAINTS and settings.INFEASIBILITY_DIAGNOSIS_GROUPS):
        return []

    print("├── Pre-solve feasibility spectrum (provable infeasibility → knife-edge census)...", flush=True)
    spec = feasibility_spectrum(
        luto_solver.gurobi_model,
        keep_groups=settings.INFEASIBILITY_DIAGNOSIS_GROUPS,
        droppable=settings.DROP_UNREACHABLE_CONSTRAINTS)

    # Proven drops. Removal goes through the solver so the bookkeeping dicts stay in sync — a
    # stale Constr would crash `record_shadow_prices` after the accepted solve. Records are
    # written BEFORE the solve on purpose: they matter most when the year still goes on to fail.
    if spec['dropped']:
        luto_solver.remove_constraints_by_name(spec['dropped'])
        record_dropped([{'group': group_of(n), 'constraint': n, 'action': 'DROPPED'}
                        for n in spec['dropped']],
                       luto_solver, data, target_year, 'pre_solve')

    if spec['status'] == 'INFEASIBLE_UNRESOLVABLE':
        print("├── conflict among non-droppable rows — nothing more can be given up; the ladder "
              "will run and the year will fail loudly if it cannot solve", flush=True)
        record_dropped([{'group': None, 'constraint': None, 'action': 'UNRESOLVABLE'}],
                       luto_solver, data, target_year, 'pre_solve')
        return spec['dropped']

    threshold = getattr(settings, 'KNIFE_EDGE_DROP_BELOW', 1e-4)
    droppable = set(settings.DROP_UNREACHABLE_CONSTRAINTS)
    to_drop = {n: eps for n, eps in spec['edge'].items()
               if eps <= threshold and group_of(n) in droppable}
    to_keep = {n: eps for n, eps in spec['edge'].items() if n not in to_drop}

    if to_drop:
        print(f"├── dropping {len(to_drop)} knife-edge row(s) with relative headroom "
              f"<= {threshold:g} (numerically indistinguishable from infeasible):", flush=True)
        for n, eps in sorted(to_drop.items(), key=lambda kv: kv[1]):
            print(f"│       [{group_of(n)}] headroom<{eps:g}  {n}", flush=True)
        luto_solver.remove_constraints_by_name(list(to_drop))
        record_dropped([{'group': group_of(n), 'constraint': n,
                         'action': 'DROPPED_KNIFE_EDGE', 'headroom_lt': eps}
                        for n, eps in to_drop.items()],
                       luto_solver, data, target_year, 'pre_solve')
    if to_keep:
        print(f"├── {len(to_keep)} thin row(s) recorded as knife-edge, kept in the model:", flush=True)
        for n, eps in sorted(to_keep.items(), key=lambda kv: kv[1]):
            print(f"│       [{group_of(n)}] headroom<{eps:g}  {n}", flush=True)
        record_dropped([{'group': group_of(n), 'constraint': n, 'action': 'KNIFE_EDGE',
                         'headroom_lt': eps}
                        for n, eps in to_keep.items()],
                       luto_solver, data, target_year, 'pre_solve')

    return spec['dropped'] + list(to_drop)


def diagnose_and_drop_conflict(luto_solver: LutoSolver, data: Data, target_year: int) -> bool:
    """After the ladder fails: ask the IIS what conflicts, drop it, and say whether to retry.

    Returns True when something was dropped — the caller should put the ladder back on the reduced
    model — and False when there is nothing left to give up, which ends the year.
    """
    if not settings.INFEASIBILITY_DIAGNOSIS_GROUPS:
        return False

    print("├── Not optimal — diagnosing the conflict...", flush=True)
    resolution = resolve_infeasibility(
        luto_solver.gurobi_model,
        droppable=settings.DROP_UNREACHABLE_CONSTRAINTS,
        keep_groups=settings.INFEASIBILITY_DIAGNOSIS_GROUPS)

    if not resolution['dropped']:
        print(f"├── {resolution['status']} — nothing droppable in the conflict; "
              f"giving up on {target_year}", flush=True)
        return False

    print(f"├── dropping {len(resolution['dropped'])} row(s) and re-solving {target_year}:", flush=True)
    for n in resolution['dropped']:
        print(f"│       [{group_of(n)}] {n}", flush=True)

    # Removes from the model AND the solver's bookkeeping dicts — a stale Constr left in those
    # would crash `record_shadow_prices` after the accepted solve.
    luto_solver.remove_constraints_by_name(resolution['dropped'])
    record_dropped([{'group': group_of(n), 'constraint': n, 'action': 'DROPPED'}
                     for n in resolution['dropped']],
                    luto_solver, data, target_year, 'post_solve')
    return True


def record_dropped(records, luto_solver, data, target_year, stage) -> None:
    """Append dropped-constraint records to out_<year>/dropped_constraints_<year>.csv.

    Re-attaches family / region / item / presence from the solver's own index, because the
    constraint name cannot be parsed back into them (spaces became underscores, and the arity
    differs by family). Appends rather than overwrites: a year can drop rows in BOTH the pre-solve
    per-group test and the post-failure IIS, and the first record must survive the second.
    """
    if records is None or (hasattr(records, 'empty') and records.empty) or len(records) == 0:
        return
    df = pd.DataFrame(records)
    index = luto_solver.bio_constraint_index()
    parts = pd.DataFrame(
        [index.get(n, {'family': None, 'region': None, 'item': None, 'presence': None})
         for n in df['constraint']],
        index=df.index)
    df = pd.concat([df, parts], axis=1).assign(year=target_year, stage=stage)
    # Canonical column set: appends from different stages carry different keys (feasible_solve has
    # round/iis_size, the knife-edge census has headroom_lt) and a CSV append with a different
    # column set from the existing header silently misaligns the file. Absent keys become blanks.
    df = df.reindex(columns=['year', 'stage', 'group', 'constraint', 'action', 'round',
                             'iis_size', 'headroom_lt', 'family', 'region', 'item', 'presence'])

    out_dir = f"{data.path}/out_{target_year}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/dropped_constraints_{target_year}.csv"
    df.to_csv(path, mode='a' if os.path.exists(path) else 'w', header=not os.path.exists(path), index=False)


# ---------------------------------------------------------------------------- #
# Persistence (checkpoints, model dumps, Data serialisation)                   #
# ---------------------------------------------------------------------------- #

def load_latest_checkpoint(
    checkpoint_path: Path,
    data: Data | None,
    save_dir: str,
) -> tuple[Data, int | None]:
    """Load the newest data_<year>.lz4 in `checkpoint_path`, falling back to `data`.

    Returns (data, resume_year); resume_year is None when starting fresh. Raises when
    there is neither a checkpoint nor a `data` object to fall back on.
    """
    print(f"Checkpoint mode enabled: {checkpoint_path}", flush=True)
    files = sorted(f for f in checkpoint_path.iterdir() if CHECKPOINT_RE.fullmatch(f.name))

    if not files:
        if data is None:
            raise ValueError(
                f"No checkpoint files found in '{checkpoint_path}' and no `data` was provided; "
                "cannot start simulation."
            )
        print(f"No valid checkpoint found in '{checkpoint_path}'; starting from {min(settings.SIM_YEARS)}.", flush=True)
        return data, None

    checkpoint_file = files[-1]
    resume_year = int(checkpoint_file.stem.split("_")[1])
    data = joblib.load(str(checkpoint_file))
    data.timestamp = read_timestamp()
    data.path = save_dir
    # load_data()'s set_path() normally pre-creates the out_<yr> dirs that
    # write_outputs expects; checkpoint resume bypasses load_data(), so do it here.
    set_path()
    print(f"Resuming from checkpoint (year {resume_year}): {checkpoint_file}", flush=True)
    return data, resume_year


def save_checkpoint(data: Data, checkpoint_path: Path, target_year: int) -> None:
    """Write data_<target_year>.lz4 and delete older checkpoints — only the latest is kept."""
    final_path = checkpoint_path / f"data_{target_year}.lz4"
    save_data_to_disk(data, str(final_path))
    # `match` here on purpose, unlike the fullmatch in load_latest_checkpoint: the prefix form also
    # sweeps up any `data_<year>.lz4.tmp` orphaned by an earlier killed write, which is exactly what
    # rotation should do. Discovery must be strict; cleanup should be greedy.
    for old in checkpoint_path.iterdir():
        if CHECKPOINT_RE.match(old.name) and old != final_path:
            old.unlink()
    print(f"Saved checkpoint for year {target_year}: {final_path}", flush=True)


def save_model_to_disk(gurobi_model, out_dir: str, base_year: int, target_year: int) -> None:
    """Write the year's Gurobi model to MPS, replacing the previous year's file.

    Written BEFORE the solve, unconditionally — the same reasoning as the unreachable-rows CSV. The
    model is most valuable exactly when the year does not finish, and the cases that most need a
    post-mortem are the ones that never reach the failure branch at all: a wall-time kill, an OOM,
    or INF_OR_UNBD where Gurobi cannot even classify what went wrong.

    Only the latest year is kept. At ~800 MB per model there is no point accumulating one per year,
    and the interesting model is always the one that just failed. Written to a temp name and
    renamed, so an interrupted write cannot leave a truncated file that looks valid.

    Plain .mps, NOT .mps.gz/.bz2 — this Gurobi build ships without compression codecs and any
    compressed extension raises "Unable to write to file". Run_Archive.zip compresses it anyway.

    Constraint names matter here: MPS is whitespace-delimited, so a single name containing a space
    makes Gurobi discard EVERY name in the file and emit c0, c1, c2 ... which makes the artefact
    useless for attributing a failure. Every `name=` in solver.py must therefore stay
    space-free (see the `.replace(" ", "_")` on the regional-adoption rows).
    """
    dest = Path(out_dir) / f"debug_model_{base_year}_{target_year}.mps"
    # The temp name must KEEP the .mps extension: Gurobi picks the writer from the extension, so
    # "....mps.tmp" fails with "Unknown file type" and the save is silently lost.
    tmp = dest.with_name(f"{dest.stem}.tmp{dest.suffix}")
    try:
        os.makedirs(out_dir, exist_ok=True)
        gurobi_model.write(str(tmp))
        # `tmp` must be excluded here: it ends in .mps (Gurobi picks its writer from the extension,
        # so it has to) which means this very glob matches the file just written, and deleting it
        # makes the rename below fail with "cannot find the file specified".
        for stale in Path(out_dir).glob("debug_model_*.mps"):
            if stale not in (dest, tmp):
                stale.unlink()
        tmp.replace(dest)
        print(f"Saved model to {dest} ({dest.stat().st_size / 1e6:,.0f} MB)", flush=True)
    except Exception as exc:
        # Never let a diagnostic artefact take the run down with it.
        print(f"WARNING: could not save model to {dest}: {exc}", flush=True)
        tmp.unlink(missing_ok=True)


def save_data_to_disk(data: Data, path: str, compress_level=3) -> None:
    """Save using joblib with atomic tmp→rename to prevent partial writes."""
    print(f'Saving data to {path}...', flush=True)
    tmp = Path(f"{path}.tmp")
    joblib.dump(data, str(tmp), compress=('lz4', compress_level))
    # Write to .tmp first, then rename atomically (os.replace → POSIX rename()).
    # If the job is killed mid-write, only the .tmp is left partial; the final
    # .lz4 is never created until the write completes successfully.
    os.replace(tmp, path)
