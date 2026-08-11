# Skill: Retry Task Runs with Adjusted Gurobi Parameters

This skill retries task runs where a year returned a non-optimal solver status (INFEASIBLE,
NUMERIC, SUBOPTIMAL).

**There are two paths, and picking the wrong one wastes hours.**

| | **A — live in-place resume** | **B — archive rebuild** |
|---|---|---|
| when | the run is **still on Gadi**, finished or not: `Run_<name>/output/<ts>/data_<year>.lz4` exists | the run dir is gone or you need a *different* configuration in a clean task dir |
| cost | seconds — no unzip, no copy, no new task dir | ~10 min per run of unzipping + resubmission |
| keeps | the same run directory, same output dir, all years already solved | starts a parallel task dir |
| how | patch `Run_<name>/luto/settings.py`, `qdel` if running, then `bash cmd.sh` in the run dir | Steps 1–5 below |

Path A is almost always what you want when a run is sitting in the task dir — including a run
that is **still executing** and heading for trouble. Jump to
[Path A](#path-a--resume-a-live-run-in-place); Steps 1–5 document Path B.

---

## Path A — resume a live run in place

Every run directory is self-contained and already knows how to resume itself.
`python_script.py` looks for a `data_<year>.lz4` under `output/*/`, and when it finds one it
**skips `load_data()`** and calls `sim.run(data=None, checkpoint_dir=...)`, which loads the newest
checkpoint and continues from there into the same output directory.

```bash
cd /g/data/jk53/jinzhu/LUTO/Custom_runs/<TASK>/Run_<name>

# 1. verify there IS a checkpoint, and see which year it will resume from.
#    The newest one is the last year that SOLVED — see the section below.
ls -la output/*/data_*.lz4

# 2. patch whatever setting was wrong. This has NO effect on a running process —
#    Python imported settings.py at startup — so it only matters for the relaunch.
sed -i "s|^INFEASIBILITY_DIAGNOSIS_GROUPS=.*|INFEASIBILITY_DIAGNOSIS_GROUPS=[...]|" luto/settings.py

# 3. if the job is still in the queue, kill it FIRST and wait for it to leave
qdel <jobid>; while qstat <jobid> >/dev/null 2>&1; do sleep 15; done

# 4. relaunch. cmd.sh re-reads task_param.py (and redo_param.py if present) and qsubs.
bash cmd.sh
```

⚠ **Never `qdel` before checking step 1.** With no checkpoint the relaunch starts from the base
year and every solved year is lost.

⚠ **Editing `luto/settings.py` mid-run changes nothing** for the process already running. The
module is imported once at startup. Patch-then-relaunch, or the run keeps the old behaviour.

Override resources for the relaunch (a longer walltime, a different queue) by writing
`redo_param.py` next to `task_param.py` — `cmd.sh` sources it after `task_param.py`, so it wins.

### When to interrupt a run that has not failed yet

If a configuration bug is found *while* runs are executing, restarting is worth it only for runs
with real work left:

* run has **finished solving** (all `SIM_YEARS` done, now writing outputs) — **leave it alone.**
  The solver settings no longer matter and a restart destroys hours of report writing.
* run is on its **last year** — marginal. A restart costs that year's partial attempt; leaving it
  risks a stall that costs the same plus the wasted walltime.
* run has **two or more years left** — restart. A stall burns the remaining walltime, and you end
  up resuming from the same checkpoint anyway, just later.

---

## When checkpoint-based retry actually fixes the infeasible year

The retry is only useful if the checkpoint sitting in the archive is from the **year
before** the infeasible one. Whether that holds depends on which version of
`simulation.py` created the archive:

| `simulation.py` version | Checkpoint saved when | Archive contains | Retry fixes infeasible year? |
|---|---|---|---|
| **Old** (unconditional save) | Every year, optimal or not | Infeasible year's state | **No** — infeasible year is the base, not re-solved |
| **New** (save only on optimal) | Last good year only | Year before infeasibility | **Yes** — infeasible year becomes the first target |

The new behaviour (`accepted`-gated checkpoint save + pre-loop base checkpoint) is in
`simulation.py:solve_timeseries`. If the archive was created by old code, a full re-run
from scratch is the only way to fix the infeasible year.

---

## Step 1: Check which year went infeasible and why

```bash
# Scan all PBS stdout logs for solver status messages
for run in /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>/Run_G*/; do
    name=$(basename $run)
    pbs_out=$(ls $run/run_G*.o* 2>/dev/null | head -1)
    echo "=== $name ==="
    grep -i "infeasib\|non-optimal\|Solver status" "$pbs_out" 2>/dev/null
done
```

Check the IIS `.err` files if IIS jobs were submitted — "Cannot compute IIS on a
feasible model" means the infeasibility was **numerical** (false infeasible), not
structural. A structurally infeasible model requires constraint relaxation, not solver
tuning.

---

## Step 2: Choose the right `RETRY_PARAMS`

Edit `BASE_GRID["RETRY_PARAMS"]` in `retry_create_task.py`. The default setting that
resolves numerical false-infeasibility:

```python
# Dual simplex as third attempt — avoids false-INFEASIBLE from the homogeneous
# barrier (BARHOMOGENOUS=1), which declares infeasibility too aggressively when
# the feasible region is tight. Dual simplex uses a different code path.
"RETRY_PARAMS": [(0, 2, 0), (3, 2, 0), (3, 1, 0)],
```

Alternative if you suspect barrier stagnation rather than false-infeasibility:

```python
# Barrier with auto crossover — forces a vertex solution from the interior point,
# resolving stagnation at ill-conditioned termination. Can be slow on large models.
"RETRY_PARAMS": [(3, 2, -1)],
```

Method reference: `-1`=auto, `0`=primal simplex, `1`=dual simplex, `2`=barrier,
`3`=concurrent, `4`=deterministic concurrent.

---

## Step 3: Write `retry_create_task.py`

Place it under `jinzhu_inspect_code/<Iteration>/retry_create_task.py`. It mirrors
`create_tasks.py` in structure but `main()` does three things instead of creating
fresh run folders:

1. Builds the settings template (same `BASE_GRID` + `RUN_OVERRIDES` pattern).
2. Unzips each `Run_Archive.zip` from `SOURCE_DIR` into `TASK_DIR/Run_G000X/` — skips
   any run that already has `output/*/data_*.lz4` (idempotent).
3. Calls `create_task_runs(overwrite=True)` to overlay fresh `luto/` source and write
   new `settings.py` + `task_param.py` without submitting.

```python
SOURCE_DIR = Path("/g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>")
TASK_DIR   = Path("/g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>_retry")
```

Key points:
- `create_task_runs(overwrite=True)` calls `create_run_folders`, which copies the full
  `luto/` source tree but excludes `output/` (it is in `EXCLUDE_DIRS`), so unzipped
  checkpoints are never touched.
- The per-run `settings.py` written by `write_settings` overrides `RETRY_PARAMS` with
  the value from `BASE_GRID`, regardless of what `luto/settings.py` says.

```python
def main():
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    template = build(TASK_DIR)

    archives = {p.parent.name: p for p in sorted(SOURCE_DIR.glob("Run_G*/Run_Archive.zip"))}
    print(f"\nFound {len(archives)} archives in {SOURCE_DIR.name}\n")

    run_cols = [c for c in template.columns if c != "Name"]
    for col in run_cols:
        run_dir      = TASK_DIR / col
        archive_path = archives.get(col)
        print(f"=== {col} ===")
        if archive_path is None:
            print(f"  [WARN] No archive found — skipping.\n"); continue

        lz4_files = sorted(run_dir.glob("output/*/data_*.lz4")) if run_dir.exists() else []
        if lz4_files:
            print(f"  Already unzipped ({lz4_files[-1].relative_to(run_dir)}) — skipping.\n"); continue

        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Unzipping: {archive_path.relative_to(SOURCE_DIR.parent)}")
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(run_dir)

        lz4_files = sorted(run_dir.glob("output/*/data_*.lz4"))
        if lz4_files:
            print(f"  Checkpoint: {lz4_files[-1].relative_to(run_dir)}\n")
        else:
            print(f"  [WARN] No checkpoint found — run will start from scratch.\n")

    print("Writing updated settings ...\n")
    create_task_runs(str(TASK_DIR), template, mode="cluster", n_workers=4, overwrite=True)

    print(f"\nDone. To submit:\n  cd {TASK_DIR}\n  python run_all.py")
```

---

## Step 4: Run the script

```bash
cd /g/data/jk53/jinzhu/LUTO/luto-2.0
python jinzhu_inspect_code/<Iteration>/retry_create_task.py
```

Verify checkpoint files were found for each run:

```bash
for run in /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>_retry/Run_G*/; do
    echo "$(basename $run): $(ls $run/output/*/data_*.lz4 2>/dev/null || echo 'NO CHECKPOINT')"
done
```

---

## Step 5: Submit via `run_all.py`

```bash
cd /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>_retry
python run_all.py --dry-run   # preview — shows checkpoint year per run
python run_all.py             # submit
```

`run_all.py` classifies run dirs and resumes those with a checkpoint lz4 and no
`Run_Archive.zip` (runs that are `finished` are skipped). It inherits `MEM`, `NCPUS`,
`TIME`, `QUEUE` from each run's `task_param.py`; override with `--mem`, `--ncpus`,
`--time`, `--queue` if needed.

---

## How checkpoint resume re-solves the failed year

The guarantee that makes both paths work: **the newest `data_<year>.lz4` is always the last year
that SOLVED**, never the one that failed. `simulation.py:solve_timeseries` saves only when the
solve is `accepted`:

```python
accepted, solution, status = solve_with_retries(luto_solver, data, target_year)
if accepted:
    store_solution(...); record_shadow_prices(...)
    if checkpoint_path is not None:
        save_checkpoint(data, checkpoint_path, target_year)   # ONLY on success
if not accepted:
    break                                                     # stop the timeseries
```

```
Before loop:  saves data_{years_to_run[0]}.lz4   (only if file absent — skipped on resume)

Loop step 0:  base=2030, target=2035 → solve 2035
  → OPTIMAL:  save data_2035.lz4, delete data_2030.lz4
  → FAIL:     do NOT save → data_2030.lz4 survives, loop breaks

resume loads data_2030.lz4   (load_latest_checkpoint: newest matching data_\d{4}\.lz4)
  → years_to_run = [2030, 2035, 2040, ...]
  → step 0: base=2030, target=2035  ← re-solves the failed year ✓
```

So `ls output/*/data_*.lz4` doubles as a status read: exactly one file, named for the last good
year. A run stopped on 2045 shows `data_2040.lz4`, and relaunching re-attempts 2045.

Both entry points use the same code — `python_script.py` passes `checkpoint_dir` for a live run,
`run_all.py` auto-detects it for a rebuilt one. `load_latest_checkpoint` sorts by filename and
takes the last, then `run()` restores the original output-directory timestamp so the resumed years
land beside the ones already written rather than in a new folder.

Edge case — fails on the very first target year (e.g. 2020 on a fresh run):
- Pre-loop saves `data_2010.lz4` (base year, before any solve)
- 2020 fails → `data_2010.lz4` survives
- `run_all.py` resumes from 2010 → re-solves 2020 ✓

---

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| "Cannot compute IIS on a feasible model" in IIS `.err` | False infeasibility — model is feasible with default tolerances | Use `RETRY_PARAMS` with dual simplex `(3, 1, 0)` |
| Retry still shows infeasible year in results | Archive created by old code — contains infeasible year's checkpoint, not prior year | Full re-run from scratch with new `RETRY_PARAMS` |
| No checkpoint found after unzip | Archive only contains `Data_RES*.lz4` (final state), not `data_YEAR.lz4` | Run was created with very old code; full re-run needed |
| `run_all.py` classifies run as `finished` | `Run_Archive.zip` present in the unzipped dir | Old archive created a zip-inside-zip; remove `Run_Archive.zip` from `TASK_DIR/Run_G*` |
| Patched `luto/settings.py` but behaviour unchanged | The process imported settings at startup; editing the file mid-run does nothing | `qdel` and `bash cmd.sh` — see Path A |
| Relaunch starts from the base year, losing solved years | No `data_<year>.lz4` in `output/*/` when it was relaunched | Nothing to recover. Always `ls output/*/data_*.lz4` **before** `qdel` |
| `NO_CONFLICT_FOUND — nothing droppable`, then a simplex stall with the objective frozen near 1e36 | The real conflict involves a group **not** in `INFEASIBILITY_DIAGNOSIS_GROUPS`, so the pre-solve spectrum certified the model feasible. The log says so: *"the cause involves a group excluded by keep_groups"* | Add the missing group(s) and resume. `flow_in`/`flow_out` (the transition-flow rows) are the usual omission — targets unreachable *through land movement* are invisible without them. Compare against the repo default rather than a hardcoded list |
