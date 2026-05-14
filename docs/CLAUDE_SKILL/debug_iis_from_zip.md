# Skill: Debug IIS from Run Archive Zips

This skill submits PBS jobs to compute and analyze the IIS (Irreducible Infeasible Subsystem) for infeasible LUTO2 runs archived as `Run_Archive.zip` files. Each zip contains a `debug_model_*.mps` (the Gurobi model), a `Data_RES*.lz4` (the serialized LUTO data object), and `luto/settings.py` (the run's settings).

## Zip Entry Patterns (fixed, auto-detected)

- MPS: `debug_model_*.mps`
- Data: `Data_RES*.lz4`
- Settings: `luto/settings.py` (extracted to `run_dir/settings.py`)

All outputs (`.mps`, `.lz4`, `settings.py`, `.ilp`, PBS `.sh`/`.out`/`.err`) land in the run directory (zip's parent).

## Important: Walltime

IIS computation on a ~3M-variable LUTO model can take well over 10 minutes. Set `WALLTIME` generously (e.g. `"06:00:00"`). Jobs killed by SIGTERM (exit status 271) indicate a walltime breach — increase it and resubmit.

---

## Worker: `debug_iis(zip_path)`

Create as a throwaway script (not committed). Takes only the zip path.

```python
"""
Worker for IIS debugging from a Run_Archive.zip.

Steps:
  1. Extract the MPS and Data_RES*.lz4 from the zip into the run dir
  2. Load model with Gurobi, compute IIS, write .ilp to the run dir
  3. Load LUTO data from the extracted lz4 and run analyze_iis()
"""

import importlib.util
import sys
import zipfile
import gurobipy as gp
from pathlib import Path


def _find_entry(names: list[str], pattern: str) -> str:
    matches = [e for e in names if Path(e).match(pattern)]
    if not matches:
        raise FileNotFoundError(f"No entry matching '{pattern}' found in the zip.")
    return matches[0]


def _extract_entry(zf: zipfile.ZipFile, entry: str, dest: Path):
    print(f"Extracting {entry} -> {dest} ...", flush=True)
    with zf.open(entry) as src, dest.open("wb") as dst:
        while True:
            chunk = src.read(64 * 1024 * 1024)  # 64 MB chunks
            if not chunk:
                break
            dst.write(chunk)
    print(f"Done. Size: {dest.stat().st_size / 1e9:.2f} GB", flush=True)


def debug_iis(zip_path: str):
    """
    Parameters
    ----------
    zip_path : Path to Run_Archive.zip (e.g. '.../Run_G0002/Run_Archive.zip')

    Both the MPS (debug_model_*.mps) and data (Data_RES*.lz4) are auto-detected
    from the zip. The MPS, lz4, settings.py, and ILP are all written into the
    same directory as the zip.
    """
    run_dir = Path(zip_path).resolve().parent

    # ── Step 1: Extract MPS, lz4 data, and settings ───────────────────────────
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        mps_entry = _find_entry(names, "debug_model_*.mps")
        mps_path = run_dir / Path(mps_entry).name
        _extract_entry(zf, mps_entry, mps_path)

        lz4_entry = _find_entry(names, "Data_RES*.lz4")
        lz4_path = run_dir / Path(lz4_entry).name
        _extract_entry(zf, lz4_entry, lz4_path)

        settings_path = run_dir / "settings.py"
        _extract_entry(zf, "luto/settings.py", settings_path)

    # ── Step 2: Load model, compute IIS, write ILP ───────────────────────────
    ilp_path = mps_path.with_suffix(".ilp")

    print("Loading model ...", flush=True)
    model = gp.read(str(mps_path))
    print(f"Model: {model.NumVars} vars, {model.NumConstrs} constraints", flush=True)

    print("Computing IIS ...", flush=True)
    model.computeIIS()
    model.write(str(ilp_path))
    print(f"IIS saved to {ilp_path}", flush=True)

    # ── Step 3: Load LUTO data from disk and analyze IIS ─────────────────────
    # Inject the archived settings.py so RESFACTOR matches the lz4 data
    print(f"Loading archived settings from {settings_path} ...", flush=True)
    spec = importlib.util.spec_from_file_location("luto.settings", settings_path)
    archived_settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived_settings)
    sys.modules["luto.settings"] = archived_settings

    import luto.simulation as sim
    print("Loading LUTO data from disk ...", flush=True)
    data = sim.load_data_from_disk(str(lz4_path))

    from luto.tools.inspect_iis import analyze_iis
    analyze_iis(str(ilp_path), data)
```

---

## Batch PBS Submission Script

Create as a throwaway script (not committed). Update `RUNS` for your target runs, then run with `python <script>.py`.

```python
"""
Submit PBS jobs to run IIS analysis on infeasible MPS models stored inside zip archives.
"""

import subprocess
from pathlib import Path

LUTO_ROOT = Path("/g/data/jk53/jinzhu/LUTO/luto-2.0")
PROJECT = "jk53"
QUEUE = "normalsr"
NCPUS = 16
MEM = "64GB"
WALLTIME = "06:00:00"  # IIS on ~3M-var models can take hours — set generously

BASE = Path("/g/data/jk53/jinzhu/LUTO/Custom_runs/LUF_Fourth_iteration")

RUNS = [
    BASE / "Run_G0002/Run_Archive.zip",
    BASE / "Run_G0003/Run_Archive.zip",
    # ... add more runs
]


def _pbs_script(zip_path: Path, run_name: str) -> str:
    run_dir = zip_path.parent
    return (
        f"#!/bin/bash\n"
        f"#PBS -N iis_{run_name}\n"
        f"#PBS -q {QUEUE}\n"
        f"#PBS -l storage=scratch/{PROJECT}+gdata/{PROJECT}\n"
        f"#PBS -l ncpus={NCPUS}\n"
        f"#PBS -l mem={MEM}\n"
        f"#PBS -l walltime={WALLTIME}\n"
        f"#PBS -o {run_dir}/iis_{run_name}.out\n"
        f"#PBS -e {run_dir}/iis_{run_name}.err\n"
        f"\n"
        f"conda run --no-capture-output -n luto --cwd {LUTO_ROOT} python -c "
        f"\"from jinzhu_inspect_code.debug_iis_worker import debug_iis; debug_iis('{zip_path}')\"\n"
    )


def submit(zip_path: Path):
    run_name = zip_path.parent.name
    pbs_path = zip_path.parent / f"iis_{run_name}.sh"
    pbs_path.write_text(_pbs_script(zip_path, run_name))

    result = subprocess.run(["qsub", pbs_path], capture_output=True, text=True)
    job_id = result.stdout.strip()
    if result.returncode == 0:
        print(f"Submitted {run_name}: job {job_id}")
    else:
        print(f"ERROR submitting {run_name}: {result.stderr.strip()}")


if __name__ == "__main__":
    for zip_path in RUNS:
        submit(zip_path)
```

---

## PBS Configuration

| Setting | Value |
|---------|-------|
| Queue | `normalsr` |
| CPUs | 16 |
| Memory | 64 GB |
| Walltime | 06:00:00 |
| Project | `jk53` |

## Output Files per Run (all in `{run_dir}/`)

| File | Description |
|------|-------------|
| `debug_model_YYYY_YYYY.mps` | Extracted Gurobi model |
| `Data_RES5.lz4` | Extracted LUTO data object |
| `settings.py` | Extracted run settings (ensures RESFACTOR matches lz4) |
| `debug_model_YYYY_YYYY.ilp` | Computed IIS constraints |
| `iis_Run_GXXXX.sh` | PBS job script |
| `iis_Run_GXXXX.out` / `.err` | PBS logs with `analyze_iis()` output |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Exit status 271 (SIGTERM) | Walltime exceeded during IIS computation | Increase `WALLTIME` and resubmit |
| `RESFACTOR mismatch` ValueError | Wrong `settings.py` loaded | Ensure archived `luto/settings.py` is in zip and extracted correctly |
