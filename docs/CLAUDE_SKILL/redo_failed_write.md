# Skill: Redo Failed Write Outputs

This skill re-runs `write_outputs` for task runs that completed simulation successfully (lz4 saved) but failed during the write phase. It creates a `redo_write.py` + `redo_write.pbs` in each failed run dir and submits PBS jobs.

## Key design principles

- **`os.chdir(run_dir)` is the only setup needed.** Each run dir has its own `luto/` copy, `luto/settings.py` (with correct `RESFACTOR`), and `input/` symlink. Changing to the run dir makes all relative paths resolve correctly — no `sys.path` manipulation or settings patching required.
- **Copy fixed source files into each run's `luto/` before submitting.** Each run dir has a frozen snapshot of the source code at submission time. If `write.py` or other files were fixed after the original run, you must copy the updated files into `Run_G000X/luto/` — otherwise the run will use the old broken code.
- **PBS logs go to `redo_write.stdout` / `redo_write.stderr` in the run dir.** Use `#PBS -o` and `#PBS -e` with absolute paths since `#PBS -d` is not supported on this cluster (PBS 2024.1).

---

## Step 1: Identify which source files were fixed

Check what changed in `luto-2.0` since the runs were submitted. Typically this means:

```bash
git -C /g/data/jk53/jinzhu/LUTO/luto-2.0 diff --name-only HEAD~3 HEAD
```

Common files fixed for write errors:
- `luto/tools/write.py`
- `luto/economics/agricultural/quantity.py`

---

## Step 2: Copy fixed files into each run's `luto/` directory

For every run that has a saved lz4 (simulation completed), copy the fixed files from `luto-2.0` into the run's own `luto/` copy:

```bash
for run in /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>/Run_G*/; do
    lz4=$(ls $run/output/*/Data_RES*.lz4 2>/dev/null | head -1)
    [ -z "$lz4" ] && continue
    cp /g/data/jk53/jinzhu/LUTO/luto-2.0/luto/tools/write.py         $run/luto/tools/write.py
    cp /g/data/jk53/jinzhu/LUTO/luto-2.0/luto/economics/agricultural/quantity.py \
                                                                        $run/luto/economics/agricultural/quantity.py
    echo "Copied to $(basename $run)"
done
```

Add or remove files from the copy list depending on what was actually fixed.

---

## Step 3: Run `submit_redo_write.py`

Each iteration has a `submit_redo_write.py` at the iteration root (e.g. `Custom_runs/LUF_Fifth_iteration/submit_redo_write.py`). It:

1. Scans all `Run_G*` dirs
2. Skips runs still running (no lz4 yet)
3. Skips runs that already have a `redo_write.py` (already submitted)
4. Skips runs with no write error in the log
5. Creates `redo_write.py` + `redo_write.pbs` and submits via `qsub`

```bash
cd /g/data/jk53/jinzhu/LUTO/luto-2.0
python /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>/submit_redo_write.py
```

The script is safe to re-run after remaining simulations finish — it will pick up newly failed runs while skipping already-submitted ones.

---

## What `redo_write.py` looks like

```python
import os

RUN_DIR = '/g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>/Run_G000X'
LZ4_PATH = '/path/to/Data_RES5.lz4'

os.chdir(RUN_DIR)

import joblib
from luto.tools.write import write_data, create_report

data = joblib.load(LZ4_PATH)
print(f"Loaded data, data.path = {data.path}")

write_data(data)
print("write_data complete")

create_report(data)
print("create_report complete")
```

**Why `joblib.load` + `write_data`/`create_report` instead of `sim.load_data_from_disk` + `write_outputs`:**

- `sim.load_data_from_disk` calls `write_timestamp()` and sets `data.path = new_timestamped_dir`, so output is written to a *new* directory instead of the existing one.
- `write_outputs` also reads `tools.read_timestamp()` to build `out_dir` for its log — fine for first runs, but on a redo we want logs in the existing dir.
- Using `joblib.load` directly preserves `data.path` (set when the run was originally saved), so `write_data` and `create_report` both write into the **existing** output directory.

`os.chdir(run_dir)` is the only setup. It ensures:
- `input/` resolves to the run's own input data
- `luto/settings.py` (with correct `RESFACTOR`, `SIM_YEARS`, etc.) is imported from the run dir
- The run's own `luto/` copy (now containing the fixed files from Step 2) is used

---

## What `redo_write.pbs` looks like

```bash
#!/bin/bash
#PBS -N redo_write_Run_G000X
#PBS -q normalsr
#PBS -l storage=scratch/jk53+gdata/jk53
#PBS -l ncpus=${NCPUS}   # match original run's settings_bash.py
#PBS -l mem=${MEM}        # match original run's settings_bash.py
#PBS -l jobfs=100GB
#PBS -l walltime=04:00:00

exec > /path/to/Run_G000X/redo_write.stdout 2>&1

source ~/.bashrc
conda activate luto

cd /path/to/Run_G000X
python /path/to/Run_G000X/redo_write.py
```

Notes:
- **Always match the original run's `NCPUS` and `MEM` from `luto/settings_bash.py`** — `joblib.load` loads the full data object into RAM, so write-only jobs need the same memory as the original simulation. Under-allocating causes OOM kills.
- **Use `exec > ... 2>&1` for live logs** — `#PBS -o/-e` directives buffer output until the job ends; shell-level redirection writes immediately so you can `tail -f redo_write.stdout` while the job runs.
- Walltime `04:00:00` is sufficient (write-only, no simulation)
- `#PBS -d` is NOT supported on this cluster — use `cd` in the script body instead

To read the correct resource values:
```bash
grep -E "^export (MEM|NCPUS|QUEUE)" /path/to/Run_G000X/luto/settings_bash.py
```

---

## Step 4: Monitor and check results

```bash
qstat -u jw6041 | grep redo_writ
```

Once jobs finish, check for new output dirs:
```bash
ls /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>/Run_G0001/output/
```

Check stderr if a job failed:
```bash
cat /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>/Run_G0001/redo_write.stderr
```

---

## Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `"not all values found in index 'am'"` | Old `write.py` used by run | Copy fixed `write.py` into run's `luto/tools/` (Step 2) |
| `FileNotFoundError: input/ag_landuses.csv` | Wrong working dir | Ensure `os.chdir(run_dir)` is before any luto import |
| `RESFACTOR mismatch` | luto-2.0 `settings.py` used instead of run's | Ensure `os.chdir(run_dir)` so run's own `luto/settings.py` is imported |
| PBS logs in wrong dir | `#PBS -d` not supported | Use `#PBS -o/-e` with absolute paths + `cd` in script body |
| Output written to new timestamped dir | Used `sim.load_data_from_disk` which calls `write_timestamp()` and overwrites `data.path` | Use `joblib.load` directly so `data.path` stays as the existing output dir |
| `ValueError: place: mask and data must be the same size` (RF=1) | Old `helpers/__init__.py` has broken intermediate `arr_fulllen` step | Copy fixed `luto/tools/Manual_jupyter_books/helpers/__init__.py` into run's dir (Step 2) |