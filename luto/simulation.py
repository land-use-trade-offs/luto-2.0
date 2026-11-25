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

import re
import gc
import time
import threading
import joblib
import lz4.frame
import numpy as np
import pandas as pd

from contextlib import contextmanager
from pathlib import Path
from gurobipy import GRB

import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.non_agricultural.ghg as non_ag_ghg

from luto import settings
from luto.data import Data
from luto.solvers.aggregate import check_grouping_settings, group_tables, report_grouping
from luto.solvers.col_builder import get_cols
from luto.solvers.rowgen import RowGen
from luto.solvers.step import solve_step
from luto.solvers.row_inputs import RowInputs, get_economics, get_row_inputs
from luto.solvers.row_builder import get_rows, get_obj
from luto.solvers.row_bounds import STATUS, drop_redundant_rows, get_row_bounds, report_row_bounds
from luto.solvers.post_solve import post_solve
from luto.solvers.reports import record_shadow_prices, report_shortfall, save_model_to_disk
from luto.solvers.tables import Tables, elastic_absorbs
from luto.tools.write_output import write_outputs
from luto.tools import  LogToFile, log_memory_usage, set_path, write_timestamp, read_timestamp


# A run's checkpoint is two files in its folder (2026-11-23):
#   data_<year>.lz4      the whole `Data`, written ONCE, before the first solve of the run (<year> = where the run starts);
#   solution_<year>.lz4  what solving writes on `Data` (the attributes `Data.SOLVED_STATE` names: the per-year containers
#                        and the last solved year), rewritten after every solved year, the latest kept.
# Almost all of `Data` is inputs that never change after loading: rewriting it every year cost ~3 min at RES1.
DATA_RE = re.compile(r'data_\d{4}\.lz4')
SOLUTION_RE = re.compile(r'solution_\d{4}\.lz4')

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
        writing (e.g. when doing a quick test run or debugging an infeasible year).
    checkpoint_dir : str or None, default None
        Where to LOOK for a checkpoint to resume from (``data_<year>.lz4`` and the
        latest ``solution_<year>.lz4``); the run continues at the first unsolved year. Checkpoints themselves are always
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
        if years[0] < active_data.YR_CAL_BASE:        # nothing else stops it: the first step would run backwards from the base year
            raise ValueError(f"SIM_YEARS starts in {years[0]}, before the base year YR_CAL_BASE = {active_data.YR_CAL_BASE}.")
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

        # The lz4 stream opened and closed here: joblib.load(path) leaves its own decompressor wrapper to the garbage
        # collector after closing the file under it, and Python then prints "Exception ignored ... I/O operation on
        # closed file" (harmless, same data either way)
        with lz4.frame.open(path, 'rb') as stream:
            data = joblib.load(stream)
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
    ``data_<year>.lz4`` / ``solution_<year>.lz4`` checkpoint is written (the run's own output directory when
    called from `run()`); pass None to disable checkpointing.
    """
    # ═══ the base-year checkpoint: saved before any solving, so a retry can re-attempt the first target year ═══
    #     (skipped on resume: the file already exists from a prior run)
    if checkpoint_path is not None and not any(DATA_RE.fullmatch(f.name) for f in checkpoint_path.iterdir()):
        save_data_to_disk(data, str(checkpoint_path / f"data_{years_to_run[0]}.lz4"))

    check_grouping_settings()                                                               # settings.CELL_GROUPING: what it cannot run with
    carry = set()                                                                            # row generation: the targets whose rows the next step starts with

    for base_year, target_year in zip(years_to_run[:-1], years_to_run[1:]):
        solved, carry = solve_year(data, base_year, target_year, carry, checkpoint_path)
        gc.collect()                                                                          # the year's tables are unreachable now: free them before the next year builds its own
        if not solved:
            break


def solve_year(data: Data, base_year: int, target_year: int, carry: set, checkpoint_path: Path | None) -> tuple[bool, set]:
    """One step, base_year -> target_year: build the tables, solve, store the solution, checkpoint. (solved, the row
    generation's carry for the next step). Every table, array and solver the step builds is local here, so it is freed
    when the step returns — not kept until the next step overwrites it (about 90 GB at RES1, 2026-11-23)."""
    print( "-------------------------------------------------", flush=True)
    print( f"Running for year {target_year}"   , flush=True)
    print( "-------------------------------------------------\n", flush=True)
    start_time = time.time()
    out_dir = f"{data.path}/out_{target_year}"

    # ─────────────── 1. the columns: the unknowns ───────────────
    cols, col_support = get_cols(data, base_year)                                       # the column table, and what the row side reads beside it

    # ─────────────── 2. the rows: the constraints ───────────────
    inputs = get_row_inputs(data, base_year, target_year)                               # the coefficient streams and targets
    row_matrix, rows = get_rows(inputs, cols, col_support)                              # the matrix, and the row table on the same rows

    # ─────────────── 3. row generation: the biodiversity targets and their screen (settings.BIO_ROW_GENERATION) ───────────────
    #     built here, while the column support still holds cell2col / ag_mrj2col
    rowgen = None
    bio_on = any(target != 'off' for target in (settings.GBF2_TARGET, settings.GBF3_NVIS_TARGET, settings.GBF4_TARGET_SNES,
                                                settings.GBF4_TARGET_ECNES, settings.GBF8_TARGET))
    if settings.BIO_ROW_GENERATION and bio_on:
        rowgen = RowGen(data, inputs, cols, col_support, base_year, target_year, carry)
    col_support.cell2col = col_support.ag_mrj2col = col_support.nonag_rk2col = None     # freed before the solve (GBs at RES1); the masks and the regions stay for post_solve
    if rowgen is not None and rowgen.blocked.any() and settings.BOUND_PROP_ON_IMPOSSIBLE == 'stop':
        print('!' * 100, flush=True)
        print(f"Year {target_year}: {int(rowgen.blocked.sum()):,} hard biodiversity target(s) cannot be met at any feasible point (the screen), so no solve can succeed. "
              f"Stopping before the model is built.", flush=True)
        print('!' * 100, flush=True)
        return False, carry

    # ─────────────── 4. the objective ───────────────
    obj = get_obj(get_economics(data, base_year, target_year), cols, inputs)            # million AUD; the economy streams (~300 MB at RES5, ~7 GB at RES1) die with the call

    # ─────────────── 5. the tables the solver takes: settings.CELL_GROUPING groups alike cells into blobs ───────────────
    #     (the grouped solver's x is mapped back to every cell below)
    tables = Tables(row_matrix, rows, cols, obj)
    grouping = group_tables(tables, data.REAL_AREA.values, col_support.region2cell) if settings.CELL_GROUPING else None
    solved = tables if grouping is None else grouping.tables

    # ─────────────── 6. the pre-solve diagnosis: bound propagation over the rows of the tables the solver takes ───────────────
    #     On the grouped tables under CELL_GROUPING (2026-11-23): ~10 times fewer rows than the full model, and the model
    #     actually solved. Nothing is missed: a grouped column's box lies inside its cells' (lb the largest, ub the
    #     smallest), so a row impossible over the full box is impossible over the grouped one; a row impossible only
    #     when grouped is one the grouped solve would fail on. The cost: a year it stops has built its economics and blobs.
    bounds = get_row_bounds(solved.row_matrix, solved.rows, solved.cols)               # every row's interval over the column box, and its verdict
    drop_redundant_rows(solved.rows, bounds, settings.BOUND_PROP_DROP_FAMILIES)         # opt-in per family: rows every point of the box satisfies never reach the solver
    report_row_bounds(solved.rows, bounds, target_year, out_dir)                        # the log table, bound_report_<year>.csv, bound_preflight_<year>.csv
    impossible = (bounds['status'].values == STATUS.index('impossible')) | (bounds['status_implied'].values == STATUS.index('impossible'))
    n_impossible = int((impossible & ~elastic_absorbs(solved.rows, bounds)).sum())     # an elastic row whose slack can reach the rhs does not stop the year
    if n_impossible and settings.BOUND_PROP_ON_IMPOSSIBLE == 'stop':
        print('!' * 100, flush=True)
        print(f"Year {target_year}: {n_impossible:,} row(s) cannot hold at any point the column box and the rows' own bounds allow, so no solve can succeed "
              f"(see {out_dir}/bound_report_{target_year}.csv). Stopping before the model is built.", flush=True)
        print('!' * 100, flush=True)
        return False, carry

    # ─────────────── 7. the solve: the tables are the state, a solver is built from them for every solve ───────────────
    on_first_model = (lambda model: save_model_to_disk(model, data.path, base_year, target_year)) if settings.WRITE_DEBUG_MPS else None   # the debug MPS, BEFORE the first solve (off by default: ~5 GB a year)
    accepted, x_solved, status, luto_solver = solve_step(solved, inputs, rowgen, target_year, out_dir, on_first_model=on_first_model)
    if grouping is None:
        cols, obj = luto_solver.cols, luto_solver.obj                                    # the final tables: with the shortfall columns, the dropped rows off

    if not accepted:
        print(f'Processing for {target_year} completed in {round(time.time() - start_time)} seconds\n\n' , flush=True)
        print('!' * 100, flush=True)
        print(f"Solver status for year {target_year}: {SOLVER_STATUS_MSGS.get(status, f'unexpected status {status}')}", flush=True)
        print('!' * 100, flush=True)
        print('\n', flush=True)
        return False, carry

    # ─────────────── 8. the solution: stored, reported, checkpointed ───────────────
    x = x_solved if grouping is None else grouping.col_map @ x_solved                    # over the full column table: every cell of a blob takes the blob's share
    if grouping is not None:
        report_grouping(grouping, tables, x, target_year, out_dir)                       # every row of the full model at the mapped x; grouping_<year>.csv
    solution = post_solve(x, cols, col_support, inputs)                                    # the LUTO 1-D format
    slack = cols['block'].values == 'slack'                                               # the elastic rows' shortfall columns (settings.ELASTIC_FAMILIES)
    penalty = float(obj[slack] @ x[slack]) if slack.any() else 0.0                      # million AUD: the penalty in the objective, bookkeeping not money
    if penalty:
        print(f"Year {target_year}: objective {luto_solver.gurobi_model.ObjVal:,.2f} includes the elastic penalty {penalty:,.2f} (million AUD); "
              f"stored without it: {luto_solver.gurobi_model.ObjVal - penalty:,.2f}", flush=True)
    store_solution(data, target_year, solution, luto_solver.gurobi_model.ObjVal - penalty, inputs)
    data.last_year = target_year                                                            # only a solved and stored year: the writers report through it

    record_shadow_prices(luto_solver, target_year, out_dir)
    if not settings.ELASTIC_DROP_SHORT:                                                   # under the two-pass step, pass 1's shortfall was written before the drop
        report_shortfall(x_solved, luto_solver.rows, target_year, out_dir)
    if rowgen is not None:
        rowgen.report(luto_solver.rows, out_dir)
        carry = rowgen.carry_next(luto_solver.rows)

    if checkpoint_path is not None:
        save_checkpoint(data, checkpoint_path, target_year)

    luto_solver.gurobi_model.dispose()                                                      # the model's memory is Gurobi's (C), freed now rather than when Python collects it
    print(f'Processing for {target_year} completed in {round(time.time() - start_time)} seconds\n\n' , flush=True)
    return True, carry


def store_solution(data: Data, target_year: int, solution, obj_val: float, inputs: RowInputs) -> None:
    """Copy the accepted solution (the LUTO format, ``post_solve``) and the objective value into the Data singleton, and
    put Production and GHG beside them — both computed from the stored dvars, every share counted."""
    data.add_lumap(target_year, solution.lumap)
    data.add_lmmap(target_year, solution.lmmap)
    data.add_ammaps(target_year, solution.ammaps)
    data.add_ag_dvars(target_year, solution.ag_X_mrj)
    data.add_delta_dvars_ag2ag(target_year, solution.dvar_D_ag2ag_mrj)
    data.add_non_ag_dvars(target_year, solution.non_ag_X_rk)
    data.add_delta_dvars_ag2nonag(target_year, solution.dvar_D_ag2nonag_rk)
    data.add_delta_dvars_nonag2ag(target_year, solution.dvar_D_nonag2ag_mrj)
    data.add_ag_man_dvars(target_year, solution.ag_man_X_mrj)
    data.add_obj_vals(target_year, obj_val)

    # GHG from the stored dvars and the step's own GHG inputs: Σ ghg · share over the ag, ag-mgt and non-ag land, the
    # transition emissions on the ag → ag flows, plus the off-land constant — what the GHG row summed (the row's contract
    # dropped its sub-floor coefficients; float32 dot products, 1e-7 relative of the float64 sum at RES50)
    ghg = float(np.dot(inputs.ag_g_mrj.ravel(), solution.ag_X_mrj.ravel()))
    for non_ag_g_rk in (inputs.non_ag_g_rk_ag, inputs.non_ag_g_rk_lulucf):                # both rows' streams: the net total
        ghg += float(np.dot(non_ag_g_rk.ravel(), solution.non_ag_X_rk.ravel()))
    for am, lus in inputs.agman2lu.items():                                                 # the enabled options; the GHG effect is [m, r, j_idx] over the option's land uses
        for j_idx, j in enumerate(lus):
            for ag_man_g_mrj in (inputs.ag_man_g_mrj_ag, inputs.ag_man_g_mrj_lulucf):
                ghg += float(np.dot(ag_man_g_mrj[am][:, :, j_idx].ravel(), solution.ag_man_X_mrj[am][:, :, j].ravel()))
    ghg_transition = 0.0
    for src, g in inputs.trans_ghg_ag2ag.items():                                           # [to_m, local_r, to_j], as the flows are keyed
        ghg_transition += float(np.dot(g.ravel(), solution.dvar_D_ag2ag_mrj[src].ravel()))
    ghg += ghg_transition
    ghg += inputs.offland_ghg
    data.add_production_data(target_year, 'GHG', ghg)
    data.add_production_data(target_year, 'GHG_TRANSITION', ghg_transition)                 # the row's transition term; write_ghg checks its output against it
    if settings.GHG_EMISSIONS_LIMITS != 'off':
        data.add_production_data(target_year, 'GHG_split', get_ghg_split(data, target_year, solution, inputs))

    # Production from the stored dvars, the way the base year's is (data.py, at load): t / KL per commodity — every share
    # counted (threshold 0: the map clean-up that drops a cell's slivers under 1 % would leave the total 0.1–0.8 % under
    # the demand the solver met)
    # the quantities are the step's own (RowInputs): what its demand rows were built from, not computed a second time
    potential = data.label_production_layers(inputs.ag_q_mrp, inputs.non_ag_q_crk, inputs.ag_man_q_mrp)
    ag_mrc, non_ag_rc, am_amrc = data.get_actual_production_lyr(target_year, threshold=0.0, potential=potential)
    production = (ag_mrc.sum(['cell', 'lm']) + non_ag_rc.sum(['cell']) + am_amrc.sum(['cell', 'am', 'lm'])).compute().values
    data.add_production_data(target_year, 'Production', production)


def get_ghg_split(data: Data, target_year: int, solution, inputs: RowInputs) -> pd.DataFrame:
    """The two GHG rows of the step, agriculture ('AG') and net LULUCF ('LULUCF'), from the stored dvars: one line per
    named sub-term (tCO2e), the exogenous constants on each row's left-hand side, the row's total and its series.

    ``value_t`` of the 'total' lines is the row's left-hand side; ``rhs_t`` is what it is compared with. In a year
    whose role is not 'BINDING' the row was not imposed, and ``deviation_t`` is the benchmark deviation (for AG also
    tested against the NIR tolerance in ``within_tolerance``)."""
    t = inputs.limits['ghg']
    lines = []                                                              # (row, term, component, sub_term, value_t)

    # term 1: the ag land uses (all on AG)
    energy = float(np.dot(ag_ghg.get_irrpast_energy_ghg_matrices(data).ravel(), solution.ag_X_mrj.ravel()))
    stubble = float(np.dot(ag_ghg.get_stubble_ghg_matrices(data).ravel(), solution.ag_X_mrj.ravel()))
    total_1 = float(np.dot(inputs.ag_g_mrj.ravel(), solution.ag_X_mrj.ravel()))
    lines += [('AG', 1, 'Agricultural land uses', 'Crop and livestock emissions', total_1 - energy - stubble),
              ('AG', 1, 'Agricultural land uses', 'Irrigated pasture energy and lifecycle fields', energy),
              ('AG', 1, 'Agricultural land uses', 'Stubble-grazing sheep', stubble)]

    # term 2: the ag-management options, sub-term by sub-term
    am_parts = inputs.ag_man_g_parts                                       # computed for the step's GHG rows (row_inputs)
    for am, lus in inputs.agman2lu.items():
        for name, arr in am_parts[am].items():
            value = sum(float(np.dot(arr[:, :, j_idx].ravel(), solution.ag_man_X_mrj[am][:, :, j].ravel())) for j_idx, j in enumerate(lus))
            lines.append((ag_ghg.AG_MAN_GHG_SUBTERMS[am][name], 2, am, name, value))

    # term 3: the non-ag land uses, sub-term by sub-term
    nonag_parts = inputs.non_ag_g_parts                                     # computed for the step's GHG rows (row_inputs)
    for k, lu in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        for name, arr in nonag_parts[lu].items():
            lines.append((non_ag_ghg.NON_AG_GHG_SUBTERMS[lu][name], 3, lu, name, float(np.dot(arr, solution.non_ag_X_rk[:, k]))))

    # term 4: the ag -> ag transitions (all on LULUCF)
    trans = sum(float(np.dot(g.ravel(), solution.dvar_D_ag2ag_mrj[src].ravel())) for src, g in inputs.trans_ghg_ag2ag.items())
    lines.append((ag_ghg.TRANSITION_GHG_SIDE, 4, 'Land-use transitions', 'Natural land carbon released', trans))

    # term 5 and the exogenous constants
    lines += [('AG', 5, 'Off-land commodities', 'Livestock emissions (enteric, manure)', inputs.offland_ghg - inputs.offland_ghg_energy),
              ('AG', 5, 'Off-land commodities', 'On-farm energy CO2 (direct, embedded)', inputs.offland_ghg_energy),
              ('AG', 'exogenous', 'Exogenous agriculture', 'series_agriculture_exogenous.csv (none built: 0)' if t['AG_EXO_t'] == 0 else 'series_agriculture_exogenous.csv', t['AG_EXO_t']),
              ('LULUCF', 'exogenous', 'Exogenous LULUCF', 'LULUCF_EXO (series_lulucf_exogenous.csv)', t['LULUCF_EXO_t']),
              ('LULUCF', 'baseline', 'Plantings baseline', 'Baseline new-plantings removals, subtracted (plantings_baseline.csv)', -t['PLANTINGS_BASELINE_t']),
              ('AG', 'baseline', 'HIR baseline', 'Existing HIR projects destocking, subtracted (ag_mgt_caps)', -t['HIR_BASELINE_AG_t']),
              ('LULUCF', 'baseline', 'HIR baseline', 'Existing HIR projects regrowth, subtracted (ag_mgt_caps)', -t['HIR_BASELINE_LULUCF_t'])]

    df = pd.DataFrame(lines, columns=['row', 'term', 'component', 'sub_term', 'value_t'])
    df['term'] = df['term'].astype(str)

    # the rows' totals against their series
    binding_year = t['role'] == 'BINDING'
    target = {'AG': t['AG_t'], 'LULUCF': t['LULUCF_MOD_t'] + t['LULUCF_EXO_t']}
    totals = []
    for row in ('AG', 'LULUCF'):
        imposed = binding_year and row not in settings.GHG_BENCHMARK_ROWS     # a benchmark row is reported, not imposed
        lhs = float(df.loc[df['row'] == row, 'value_t'].sum())
        tol = t['AG_TOLERANCE_t'] if row == 'AG' else np.nan
        totals.append(dict(row=row, term='total', component='Row total (left-hand side)', sub_term='', value_t=lhs,
                           rhs_t=target[row], role=t['role'], imposed=imposed, deviation_t=lhs - target[row],
                           slack_t=target[row] - lhs if imposed else np.nan, tolerance_t=tol,
                           within_tolerance=(abs(lhs - target[row]) <= tol) if (not imposed and not np.isnan(tol)) else None))
    df = pd.concat([df, pd.DataFrame(totals)], ignore_index=True)
    df.insert(0, 'Year', target_year)

    for r in totals:
        what = (f"slack {r['slack_t']:,.0f}" if r['imposed']
                else f"benchmark deviation {r['deviation_t']:,.0f}" + ('' if np.isnan(r['tolerance_t']) else
                     f" (tolerance ±{r['tolerance_t']:,.0f}: {'within' if r['within_tolerance'] else 'OUTSIDE'})"))
        print(f"Year {target_year}: GHG {r['row']} row {r['value_t']:,.0f} tCO2e vs series {r['rhs_t']:,.0f} ({'ENFORCED' if r['role'] == 'BINDING' else r['role']}); {what}", flush=True)   # the series' role 'BINDING' = the limit is enforced this year; whether it is tight is the slack
    return df


# ---------------------------------------------------------------------------- #
# Persistence (checkpoints, model dumps, Data serialisation)                   #
# ---------------------------------------------------------------------------- #

def load_latest_checkpoint(
    checkpoint_path: Path,
    data: Data | None,
    save_dir: str,
) -> tuple[Data, int | None]:
    """Load the run's `Data` (data_<year>.lz4) and put the latest solved state (solution_<year>.lz4) back on it, falling
    back to `data` when the folder holds no checkpoint.

    Returns (data, resume_year); resume_year is None when starting fresh, the year of the latest solution file when
    there is one, else the year of the data file (nothing solved since it was written). Raises when there is neither a
    checkpoint nor a `data` object to fall back on.
    """
    print(f"Checkpoint mode enabled: {checkpoint_path}", flush=True)
    data_files = sorted(f for f in checkpoint_path.iterdir() if DATA_RE.fullmatch(f.name))
    solution_files = sorted(f for f in checkpoint_path.iterdir() if SOLUTION_RE.fullmatch(f.name))

    if not data_files:
        if data is None:
            raise ValueError(
                f"No checkpoint files found in '{checkpoint_path}' and no `data` was provided; "
                "cannot start simulation."
            )
        print(f"No valid checkpoint found in '{checkpoint_path}'; starting from {min(settings.SIM_YEARS)}.", flush=True)
        return data, None

    data_file = data_files[-1]
    resume_year = int(data_file.stem.split("_")[1])
    with lz4.frame.open(data_file, 'rb') as stream:                       # opened and closed here, as in load_data_from_disk
        data = joblib.load(stream)
    if solution_files:
        solution_file = solution_files[-1]
        with lz4.frame.open(solution_file, 'rb') as stream:
            solved = joblib.load(stream)
        for name in data.SOLVED_STATE:                                    # every attribute solving writes (declared in Data.__init__)
            setattr(data, name, solved[name])
        resume_year = int(solution_file.stem.split("_")[1])
    data.timestamp = read_timestamp()
    data.path = save_dir
    # load_data()'s set_path() normally pre-creates the out_<yr> dirs that
    # write_outputs expects; checkpoint resume bypasses load_data(), so do it here.
    set_path()
    print(f"Resuming from checkpoint (year {resume_year}): {data_file.name}"
          + (f" + {solution_files[-1].name}" if solution_files else ""), flush=True)
    return data, resume_year


def save_checkpoint(data: Data, checkpoint_path: Path, target_year: int) -> None:
    """Write solution_<target_year>.lz4 (what solving writes on `Data`: the attributes `data.SOLVED_STATE` names, with
    every solved year so far) and delete the older solution files: only the latest is kept. The `Data` file of the run
    is left alone."""
    final_path = checkpoint_path / f"solution_{target_year}.lz4"
    save_data_to_disk({name: getattr(data, name) for name in data.SOLVED_STATE}, str(final_path))
    # `match` here on purpose, unlike the fullmatch in load_latest_checkpoint: the prefix form also
    # sweeps up any `solution_<year>.lz4.tmp` orphaned by an earlier killed write, which is exactly what
    # rotation should do. Discovery must be strict; cleanup should be greedy.
    for old in checkpoint_path.iterdir():
        if SOLUTION_RE.match(old.name) and old != final_path:
            old.unlink()


def save_data_to_disk(data, path: str, compress_level=3) -> None:
    """Save `data` (the `Data` object, or a checkpoint's dict of solved state) using joblib with atomic tmp→rename to
    prevent partial writes."""
    start_time = time.time()
    tmp = Path(f"{path}.tmp")
    joblib.dump(data, str(tmp), compress=('lz4', compress_level))
    # Write to .tmp first, then rename atomically (os.replace → POSIX rename()).
    # If the job is killed mid-write, only the .tmp is left partial; the final
    # .lz4 is never created until the write completes successfully.
    Path(tmp).replace(path)
    print(f'Saved data to {path} in {round(time.time() - start_time)} seconds', flush=True)
