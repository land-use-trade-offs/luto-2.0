# Skill: Submit Task Runs on Windows

This skill documents the end-to-end workflow for launching LUTO2 task runs locally on Windows. It uses `run_all.py` as a local concurrency manager (analogous to the PBS scheduler on Linux clusters).

---

## Overview

Each task run lives in a `Run_G*` subdirectory. `run_all.py` launches `python_script.py` inside each subdirectory up to a configurable concurrency limit, logs stdout/stderr to `run.log`, and archives output on completion.

```
<task_root>/
  run_all.py                         ← concurrency launcher
  merged_grid_search_parameters_unique.csv
  merged_grid_search_template.csv
  Run_G0001/
    python_script.py                 ← simulation entry point
    luto/                            ← frozen snapshot of luto source
    luto/settings.py                 ← run-specific settings
    run.log                          ← created at launch
    Run_Archive.zip                  ← created after sim completes
  Run_G0002/ ...
  Report_Data/
    Run_G0001.zip                    ← DATA_REPORT content only
    Run_G0002.zip ...
```

---

## Setup: Deploy `run_all.py` to the task root

`run_all.py` is tracked in the repo at [luto/tools/create_task_runs/bash_scripts/run_all.py](../../luto/tools/create_task_runs/bash_scripts/run_all.py), alongside the `python_script.py` template. It must be **copied once** into the task root before use — it is never committed inside a `Run_G*` directory.

```powershell
# Copy from the repo into the task root (run once per task batch)
copy F:\Users\jinzhu\Documents\luto-2.0\luto\tools\create_task_runs\bash_scripts\run_all.py `
     N:\LUF-Modelling\LUTO2_JZ\TEMP\<task_root>\run_all.py
```

Or from within the task root:

```powershell
copy F:\Users\jinzhu\Documents\luto-2.0\luto\tools\create_task_runs\bash_scripts\run_all.py .
```

`python_script.py` inside each `Run_G*` dir is already stamped there by `create_grid_search_tasks.py` (it copies from the same `bash_scripts/` directory at task-creation time). If you fix `python_script.py` in the repo after runs are created, copy the updated file into each run dir before launching.

---

## Step 1: Navigate to the task root

```powershell
cd N:\LUF-Modelling\LUTO2_JZ\TEMP\RE_runs
```

---

## Step 2: Activate the conda environment

```powershell
conda activate luto
```

---

## Step 3: Launch runs with `run_all.py`

### Run all `Run_G*` dirs, 2 at a time (default)

```bash
python run_all.py
```

### Run all with higher concurrency

```bash
python run_all.py --max 4
```

### Run specific dirs only

```bash
python run_all.py Run_G0001 Run_G0003
```

### Run sequentially (useful for debugging)

```bash
python run_all.py --max 1
```

### Override `INPUT_DIR` in every run's `luto/settings.py`

```bash
python run_all.py --input_dir "N:/LUF-Modelling/LUTO2_JZ/input"
```

### Combine: specific runs + custom input dir

```bash
python run_all.py Run_G0002 --input_dir "N:/LUF-Modelling/LUTO2_JZ/input"
```

---

## Step 4: Monitor progress

Each run writes stdout/stderr to `Run_G000X/run.log`. Tail a single run's log in PowerShell:

```powershell
Get-Content Run_G0001\run.log -Wait
```

Or watch all logs simultaneously (one terminal per run):

```powershell
Get-ChildItem Run_G*\run.log | ForEach-Object { Start-Process powershell -ArgumentList "-NoExit", "Get-Content '$($_.FullName)' -Wait" }
```

`run_all.py` also prints live status to its own console:

```
[start]  Run_G0001  (pid 12345)  → ...\Run_G0001\run.log
[done]   Run_G0001  ✓
[failed] Run_G0002  ✗  (exit 1)  see ...\Run_G0002\run.log
```

---

## Step 5: Check results

After each run completes, `python_script.py` automatically:

1. Creates `Run_G000X/Run_Archive.zip` — full simulation output (lz4 data + non-report files)
2. Creates `Report_Data/Run_G000X.zip` — DATA_REPORT content only (for the reporting dashboard)
3. Deletes all unpacked output from the run directory (only the zip files remain)

To confirm a run succeeded, check that both zips exist and are non-zero:

```powershell
Get-ChildItem Run_G0001\Run_Archive.zip, Report_Data\Run_G0001.zip | Select-Object Name, Length
```

---

## Step 6: Diagnose failures

1. Open `Run_G000X/run.log` and scroll to the bottom for the traceback.
2. Common issues:
   - **`INPUT_DIR` wrong** — patch with `--input_dir` flag or edit `Run_G000X/luto/settings.py` directly.
   - **Out of memory** — reduce `--max` concurrency so fewer runs overlap.
   - **Settings mismatch** — if source files changed after the run dir was created, copy the fixed files into `Run_G000X/luto/` before relaunching (same pattern as `redo_failed_write` skill).

To relaunch only failed runs after fixing the issue:

```bash
python run_all.py Run_G0002 Run_G0005
```

---

## Key design notes

- `python_script.py` calls `os.chdir()` to its own directory at startup — all relative paths in `luto/settings.py` resolve correctly without any `sys.path` manipulation.
- Each `Run_G*` dir contains a **frozen snapshot** of `luto/` source code at the time the task was created. If you fix a bug in the main `luto-2.0` repo, copy the affected file(s) into each run's `luto/` before relaunching.
- `run_all.py` uses `sys.executable` (the currently active conda Python) — always activate the `luto` environment before running.
- `run_all.py` polls for completions every 5 seconds; it is safe to `Ctrl+C` and relaunch (already-completed runs will skip since they lack `Run_Archive.zip` triggers — though you must pass specific run names to avoid re-running finished runs).
