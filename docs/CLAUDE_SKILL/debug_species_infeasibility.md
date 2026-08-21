# Skill: Debug GBF4 Species / Community Numerical & Infeasibility Issues

This skill diagnoses which GBF4 SNES species or ECNES ecological communities cause
solver failure (NUMERIC, INFEASIBLE, TIME_LIMIT) in a specific year transition.
It works by loading the live checkpoint, building the full model **once per worker**
with all non-GBF4 constraints active, then adding a **single** region–species
constraint and solving. This catches both structural infeasibility (target > achievable
area) and numerical breakdown (ill-conditioned coefficient rows).

## When to use

- A run fails with `GRB.NUMERIC` or `GRB.INFEASIBLE` at a specific year and GBF4
  SNES or ECNES constraints are active
- You want to identify which species to add to `GBF4_SNES_EXCLUDE_REGION_SPECIES`
  or `GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES`
- Applies to any `USER_DEFINED` or NRM-mode SNES/ECNES run

## Key concept

For each `(region, species/community, presence)` triplet:
- Build the full model with GBF4 disabled (all other constraints: GHG, water, GBF2/3,
  land budget, renewables, etc. remain active)
- Add **one** constraint: `sum(val_vector × dvar) >= lb_rescale`
- Solve with barrier (15-min cap per worker)
- Record: `status`, `tightness`, `coeff_ratio`, `n_cells`, solve time

**Tightness** = `avail / lb_rescale` where `avail = val_vector[ind].sum()` (rescaled
available area) and `lb_rescale = lb_raw / scale_factor`. Both are in rescaled solver
space.
- `tightness < 1` → structurally infeasible (available area < target even if every
  eligible cell is fully converted)
- `tightness ≥ 1` but solver returns NUMERIC/TIME_LIMIT → numerically ill-conditioned

**coeff_ratio** = `bare_coeff_max / bare_coeff_min` — measures the within-row
coefficient spread. High ratio (> 30) with few cells (< 20) is the primary indicator
of a numerically problematic constraint.

## Setup

Create a fresh inspection directory with the following layout, then copy the scripts
below into it:

```
debug_gbf4/
    data/
        data_YYYY.lz4    ← checkpoint from the last successful year
        settings.py      ← settings file copied from the failing run
        results/         ← auto-created; one result_NNN.json per worker
    log/                 ← auto-created; launcher + worker logs
    launch.py
    worker.py
    collect.py
    submit_launch.sh
```

Edit the `# ── CONFIG ──` block at the top of `launch.py` and `worker.py`:
- `LUTO_DIR`: path to the luto-2.0 source tree
- `CHECKPOINT`: filename of the saved checkpoint (e.g. `data_2040.lz4`)
- `BASE_YEAR` / `TARGET_YEAR`: the failing year transition (e.g. `2040` / `2045`)
- `PBS_PROJECT`: your NCI project code (e.g. `jk53`)
- `PBS_STORAGE`: storage flags (e.g. `scratch/jk53+gdata/jk53`)

## Steps

### Step 1: Submit the launcher

```bash
bash submit_launch.sh
```

Watch the launcher log once it starts:
```bash
tail -f log/launch.out
```

The launcher enumerates all SNES and ECNES `(region, species, presence)` triplets,
saves `data/species_list.json`, and submits exactly N individual PBS worker jobs.

### Step 2: Wait for workers

Each worker builds the full model (all non-GBF4 constraints), adds one constraint,
and solves. Worker logs appear at job completion (PBS buffers `-o`/`-e`).
Check job status with `qstat -u $USER`.

### Step 3: Collect results

```bash
conda run -n luto python collect.py
```

Writes `data/region_species_test_results.csv` and prints a status summary.

### Step 4: Interpret results

Key columns: `status_str`, `tightness`, `coeff_ratio`, `n_cells`.

| Pattern | Diagnosis |
|---------|-----------|
| any non-OPTIMAL, `tightness < 1` | Structurally infeasible — available area < target |
| `NUMERIC` or `TIME_LIMIT`, `coeff_ratio > 30`, `n_cells < 20` | Numerically ill-conditioned row |
| `TIME_LIMIT`, `tightness >> 1`, many cells | Feasible but slow under combined constraint pressure |
| `SKIP_NEG_TARGET` | Negative target — base year already exceeds threshold, safe to ignore |

### Step 5: Update exclusion lists

Add problem species to settings (both the run's `luto/settings.py` **and** the
master `luto/settings.py`):

```python
GBF4_SNES_EXCLUDE_REGION_SPECIES = [
    ...existing entries...,
    # reason + date
    ('North East', 'Pomaderris subplicata'),
]
GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES = [
    ...existing entries...,
    ('Goulburn Broken', 'Some Community Name'),
]
```

> **Checkpoint exclusion pitfall**: exclusions are applied in `data.py` at `Data()`
> construction time. A saved checkpoint already has the species baked in. Apply
> exclusions post-load by filtering `input_data.GBF4_SNES_region_species` directly:
>
> ```python
> snes_excl = set(settings.GBF4_SNES_EXCLUDE_REGION_SPECIES)
> input_data.GBF4_SNES_region_species = [
>     (r, s, p) for r, s, p in input_data.GBF4_SNES_region_species
>     if (r, s) not in snes_excl
> ]
> ```

### Step 6: Patch existing run archives (if retrying)

```bash
OLD="('North East', 'Zieria citriodora')]"
NEW="('North East', 'Zieria citriodora'), ('North East', 'Pomaderris subplicata')]"
for f in /path/to/runs/Run_G*/luto/settings.py; do
    sed -i "s/$OLD/$NEW/" "$f"
done
```

---

## Iterative multi-year workflow

When validating a full simulation across multiple year transitions (e.g.
2010→2050 in 5-year steps), species exclusions discovered for one transition
often do **not** carry over cleanly to the next — GBF4 targets escalate each
period, so a species that is feasible at 2045 may become infeasible by 2050.
Repeat the scan/exclude/retest loop **per transition**, chaining checkpoints
forward:

1. **Per-transition test directory**: create `Check_<scenario>_<base>_<target>/`
   mirroring the layout in [Setup](#setup) (`data/`, `log/`, scan scripts,
   `script_2_fullsolve.py`).
2. **Checkpoint input**: copy `data_<base>.lz4` from the previous transition's
   test directory (or generate one — see "Generating a missing checkpoint"
   below).
3. **Run the scan** using the Windows-local variant —
   `script_1_launch.py` / `script_1_worker.py` / `script_1_run_workers.py` /
   `script_1_collect.py` — same logic as `launch.py` / `worker.py` /
   `collect.py` above, but `script_1_run_workers.py` drives a
   `subprocess.Popen` pool with `N_WORK` parallel local workers (each given a
   `PBS_ARRAY_INDEX` env var) instead of submitting `qsub` jobs. Resume support:
   workers skip targets whose `result_{idx:03d}.json` already exists.
4. **Cross-reference before adding exclusions**: from
   `region_species_test_results_<target>.csv`, collect the non-optimal
   `(region, name)` tuples (`status_str not in (OPTIMAL, SKIP_NEG_TARGET,
   SKIP_EMPTY)`), then diff against the exclusion lists already present in
   `settings.py`. Only append tuples that are genuinely new — many will already
   be excluded from earlier transitions, since the same species often re-fails
   at later years too.
5. **Apply new exclusions to both**:
   - the master `luto/settings.py` (production settings), and
   - this test directory's `data/settings.py`

   Append with a dated comment block identifying the transition and test
   directory, e.g.
   `# --- 2045→2050 exclusions (Check_..._2045_2050 per-species test, <date>) ---`,
   and annotate each tuple with its `status_str`, `tightness`, and `n_cells`
   from the scan for future reference.
6. **Re-run `script_2_fullsolve.py`** with the combined exclusion set. If it
   reaches `OPTIMAL`, it saves `data_<target>.lz4` (see checkpoint pattern
   below) — use this as the input checkpoint for the **next** transition's test
   directory.
7. **If still non-OPTIMAL** after adding all newly-identified species, repeat
   from step 3 with a fresh scan (consider a `_v2` test directory) — the
   combined exclusion set can shift which species become tight or infeasible.
8. Treat **both `INFEASIBLE` and `TIME_LIMIT`** results as exclusion
   candidates — `TIME_LIMIT` in isolation often indicates a species that only
   becomes infeasible when combined with all the others in the full model.

### Generating a missing checkpoint

If no `data_<base>.lz4` checkpoint exists for the transition you need (e.g. the
production archive only saved every other year), generate one from
`script_2_fullsolve.py` for the *prior* transition by capturing the
`SolverSolution` and writing the checkpoint — mirroring the production pattern
in `simulation.py`'s `solve_timeseries()`:

```python
from luto.simulation import save_data_to_disk

solution = solver.solve()   # NOT model.optimize() — solve() returns SolverSolution

if solution is not None and model.Status == GRB.OPTIMAL:
    data.add_lumap(target_year, solution.lumap)
    data.add_lmmap(target_year, solution.lmmap)
    data.add_ammaps(target_year, solution.ammaps)
    data.add_ag_dvars(target_year, solution.ag_X_mrj)
    data.add_non_ag_dvars(target_year, solution.non_ag_X_rk)
    data.add_ag_man_dvars(target_year, solution.ag_man_X_mrj)
    data.add_obj_vals(target_year, solution.obj_val)
    for data_type, prod_data in solution.prod_data.items():
        data.add_production_data(target_year, data_type, prod_data)
    data.last_year = target_year
    save_data_to_disk(data, os.path.join(DATA_DIR, f"data_{target_year}.lz4"))
```

### RETRY_PARAMS unpacking (5-tuples)

`script_2_fullsolve.py` retries with `settings.RETRY_PARAMS` — a list of
5-tuples `(NumericFocus, Method, Crossover, Presolve, BarHomogeneous)`, iterated
directly with **no** special "first attempt" prepended:

```python
retry_params = getattr(settings, "RETRY_PARAMS", [])
for attempt, (numeric_focus, method, crossover, presolve, bar_homogeneous) in enumerate(retry_params):
    model.Params.Method         = method
    model.Params.NumericFocus   = numeric_focus
    model.Params.Crossover      = crossover
    model.Params.Presolve       = presolve
    model.Params.BarHomogeneous = bar_homogeneous

    solution = solver.solve()
    if model.Status == GRB.OPTIMAL:
        break
    elif model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        break  # don't burn remaining retries on a proven infeasibility
```

---

## Script: `submit_launch.sh`

```bash
#!/bin/bash
# Submit the launcher as a PBS job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"

SCRIPT_PBS=$(mktemp)
cat << EOF > "${SCRIPT_PBS}"
#!/bin/bash
#PBS -N gbf4_launch
#PBS -q normalsr
#PBS -P jk53
#PBS -l storage=scratch/jk53+gdata/jk53
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l walltime=00:30:00
#PBS -r y
#PBS -l wd
#PBS -o /dev/null
#PBS -j oe

exec >> "${LOG_DIR}/launch.out" 2>&1
PYTHONUNBUFFERED=1 conda run -n luto python -u "${SCRIPT_DIR}/launch.py"
EOF

JOB_ID=\$(qsub "\${SCRIPT_PBS}")
rm "\${SCRIPT_PBS}"
echo "Launcher submitted: \${JOB_ID}"
echo "Watch: tail -f ${LOG_DIR}/launch.out"
echo "After all workers finish: conda run -n luto python ${SCRIPT_DIR}/collect.py"
```

---

## Script: `launch.py`

```python
#!/usr/bin/env python3
"""
Enumerate all GBF4 SNES/ECNES region-species targets and submit one PBS worker per target.
Edit the CONFIG block below before running.
"""

import sys, os, json, subprocess, importlib.util, joblib

# ── CONFIG ────────────────────────────────────────────────────────────────────
LUTO_DIR    = "/g/data/jk53/jinzhu/LUTO/luto-2.0"
CHECKPOINT  = "data_2040.lz4"   # filename inside data/
BASE_YEAR   = 2040
TARGET_YEAR = 2045
PBS_PROJECT = "jk53"
PBS_STORAGE = "scratch/jk53+gdata/jk53"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
LOG_DIR     = os.path.join(SCRIPT_DIR, "log")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

sys.path.insert(0, LUTO_DIR)
import luto.settings as settings

spec = importlib.util.spec_from_file_location("run_settings", os.path.join(DATA_DIR, "settings.py"))
run_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_settings)
for attr in dir(run_settings):
    if not attr.startswith("_"):
        setattr(settings, attr, getattr(run_settings, attr))

from luto.solvers.input_data import get_input_data

print(f"Loading {CHECKPOINT} ...", flush=True)
data = joblib.load(os.path.join(DATA_DIR, CHECKPOINT))

print(f"Building input_data for {BASE_YEAR}→{TARGET_YEAR} ...", flush=True)
input_data = get_input_data(data, BASE_YEAR, TARGET_YEAR)

all_targets = []
for region, species, presence in input_data.GBF4_SNES_region_species:
    lb_raw = input_data.limits["GBF4_SNES"].sel(dict(layer=(region, species, presence))).item()
    all_targets.append({"idx": len(all_targets), "type": "SNES",  "region": region, "name": species,   "presence": presence, "lb_raw": lb_raw})
for region, community, presence in input_data.GBF4_ECNES_region_species:
    lb_raw = input_data.limits["GBF4_ECNES"].sel(dict(layer=(region, community, presence))).item()
    all_targets.append({"idx": len(all_targets), "type": "ECNES", "region": region, "name": community, "presence": presence, "lb_raw": lb_raw})

N = len(all_targets)
print(f"Total targets: {N}  (SNES: {sum(1 for t in all_targets if t['type']=='SNES')}, "
      f"ECNES: {sum(1 for t in all_targets if t['type']=='ECNES')})", flush=True)

with open(os.path.join(DATA_DIR, "species_list.json"), "w") as f:
    json.dump(all_targets, f, indent=2)

WORKER = os.path.join(SCRIPT_DIR, "worker.py")
WRAPPER = os.path.join(SCRIPT_DIR, "_worker_wrapper.sh")
with open(WRAPPER, "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"PYTHONUNBUFFERED=1 conda run -n luto python -u {WORKER}\n")
os.chmod(WRAPPER, 0o755)

print(f"\nSubmitting {N} worker jobs ...", flush=True)
for t in all_targets:
    idx = t["idx"]
    r = subprocess.run(
        ["qsub",
         "-N", f"gbf4_{idx}",
         "-q", "normalsr",
         "-P", PBS_PROJECT,
         "-l", f"storage={PBS_STORAGE}",
         "-l", "ncpus=16",
         "-l", "mem=64GB",
         "-l", "walltime=01:00:00",
         "-l", "wd", "-r", "y",
         "-v", f"PBS_ARRAY_INDEX={idx},SCRIPT_DIR={SCRIPT_DIR}",
         "-o", f"{LOG_DIR}/worker_{idx}.out",
         "-e", f"{LOG_DIR}/worker_{idx}.err",
         WRAPPER],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  [{idx}] qsub FAILED: {r.stderr.strip()}", flush=True)
    else:
        print(f"  [{idx}] {t['type']:5s} {t['name'][:40]:40s} [{t['region']}] → {r.stdout.strip()}", flush=True)

print(f"\nDone. {N} jobs submitted.", flush=True)
print(f"After all finish: conda run -n luto python {SCRIPT_DIR}/collect.py", flush=True)
```

---

## Script: `worker.py`

```python
#!/usr/bin/env python3
"""
PBS worker: test ONE GBF4 SNES/ECNES region-species constraint in isolation.
Edit the CONFIG block below before running.
"""

import sys, os, json, time, importlib.util, joblib, numpy as np
from gurobipy import GRB

# ── CONFIG ────────────────────────────────────────────────────────────────────
LUTO_DIR    = "/g/data/jk53/jinzhu/LUTO/luto-2.0"
CHECKPOINT  = "data_2040.lz4"   # filename inside data/
BASE_YEAR   = 2040
TARGET_YEAR = 2045
# ─────────────────────────────────────────────────────────────────────────────

idx = int(os.environ.get("PBS_ARRAY_INDEX", sys.argv[1] if len(sys.argv) > 1 else "0"))

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"[idx={idx}] Starting worker", flush=True)

sys.path.insert(0, LUTO_DIR)
import luto.settings as settings

spec = importlib.util.spec_from_file_location("run_settings", os.path.join(DATA_DIR, "settings.py"))
run_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_settings)
for attr in dir(run_settings):
    if not attr.startswith("_"):
        setattr(settings, attr, getattr(run_settings, attr))

from luto.solvers.input_data import get_input_data
from luto.solvers.solver import LutoSolver

print(f"[idx={idx}] Loading {CHECKPOINT} ...", flush=True)
data = joblib.load(os.path.join(DATA_DIR, CHECKPOINT))

print(f"[idx={idx}] Building input_data for {BASE_YEAR}→{TARGET_YEAR} ...", flush=True)
input_data = get_input_data(data, BASE_YEAR, TARGET_YEAR)

# Build flat targets list (GBF4 must be enabled in settings so triplets are populated)
all_targets = []
for region, species, presence in input_data.GBF4_SNES_region_species:
    lb_raw = input_data.limits["GBF4_SNES"].sel(dict(layer=(region, species, presence))).item()
    all_targets.append(("SNES", region, species, presence, lb_raw))
for region, community, presence in input_data.GBF4_ECNES_region_species:
    lb_raw = input_data.limits["GBF4_ECNES"].sel(dict(layer=(region, community, presence))).item()
    all_targets.append(("ECNES", region, community, presence, lb_raw))

print(f"[idx={idx}] Total targets: {len(all_targets)}", flush=True)

if idx >= len(all_targets):
    print(f"[idx={idx}] Index out of range — exiting.", flush=True)
    sys.exit(0)

typ, region, name, presence, lb_raw = all_targets[idx]
print(f"[idx={idx}] Target: {typ}  {name} ({presence})  [{region}]  target={lb_raw:,.0f}", flush=True)

# Build base model with GBF4 disabled
settings.GBF4_TARGET_SNES  = "off"
settings.GBF4_TARGET_ECNES = "off"

print(f"[idx={idx}] Formulating base model ...", flush=True)
solver = LutoSolver(input_data)
solver.formulate()
model  = solver.gurobi_model

model.Params.Method       = 2       # barrier
model.Params.Crossover    = -1
model.Params.NumericFocus = 1
model.Params.BarConvTol   = 1e-4
model.Params.TimeLimit    = 900     # 15-min cap per test
model.Params.OutputFlag   = 1

# Compute constraint row metrics
reg_matrix = input_data.region_NRM_names_r
if typ == "SNES":
    val_matrix    = input_data.GBF4_SNES_pre_1750_area_sr
    scale_factors = input_data.scale_factors["GBF4_SNES"]
    val_vector    = val_matrix.sel(dict(layer=(name, presence)), drop=True).values
    sf            = scale_factors.sel(dict(layer=(region, name, presence))).item()
else:
    val_matrix    = input_data.GBF4_ECNES_pre_1750_area_sr
    scale_factors = input_data.scale_factors["GBF4_ECNES"]
    val_vector    = val_matrix.sel(dict(layer=(name, presence)), drop=True).values
    sf            = scale_factors.sel(dict(layer=(region, name, presence))).item()

lb_rescale = lb_raw / sf

if region == "Australia":
    ind = np.where(val_vector > 0)[0]
else:
    ind = np.intersect1d(np.where(val_vector > 0)[0], np.where(reg_matrix == region)[0])

n_cells     = ind.size
avail_ha    = float(val_vector[ind].sum()) if n_cells > 0 else 0.0
tightness   = avail_ha / lb_rescale if lb_rescale > 0 else float("inf")  # <1 infeasible, ≥1 feasible
bare_min    = float(val_vector[ind].min()) if n_cells > 0 else 0.0
bare_max    = float(val_vector[ind].max()) if n_cells > 0 else 0.0
coeff_ratio = bare_max / bare_min if bare_min > 0 else float("inf")  # ill-conditioning indicator

print(
    f"[idx={idx}] n_cells={n_cells}  avail={avail_ha:,.0f}  "
    f"tightness={tightness:.4f}  coeff=[{bare_min:.3e},{bare_max:.3e}]  ratio={coeff_ratio:.1f}",
    flush=True,
)

GUROBI_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.TIME_LIMIT: "TIME_LIMIT", GRB.NUMERIC: "NUMERIC", GRB.SUBOPTIMAL: "SUBOPTIMAL",
}

result = dict(
    idx=idx, type=typ, region=region, name=name, presence=presence,
    n_cells=n_cells, target_ha=lb_raw, avail_ha=avail_ha,
    tightness=tightness, bare_coeff_min=bare_min, bare_coeff_max=bare_max,
    coeff_ratio=coeff_ratio, lb_rescale=lb_rescale, scale_factor=sf,
    status=None, status_str=None, solve_s=None,
)

if lb_raw <= 0:
    result.update(status=-1, status_str="SKIP_NEG_TARGET", solve_s=0)
elif n_cells == 0:
    result.update(status=-2, status_str="SKIP_EMPTY", solve_s=0)
else:
    if tightness < 1.0:
        print(f"[idx={idx}] WARNING: tightness < 1 ({tightness:.4f}) — target exceeds available area!", flush=True)

    model.addConstr(
        solver._build_biodiv_contr_expr(val_vector, ind) >= lb_rescale,
        name=f"test_{typ}_{region}_{name}_{presence}".replace(" ", "_"),
    )
    model.update()

    print(f"[idx={idx}] Solving ...", flush=True)
    t0 = time.time()
    model.optimize()
    elapsed = time.time() - t0

    status     = model.Status
    status_str = GUROBI_STATUS.get(status, f"STATUS_{status}")
    print(f"[idx={idx}] → {status_str} in {elapsed:.1f}s", flush=True)
    result.update(status=status, status_str=status_str, solve_s=round(elapsed, 1))

out_file = os.path.join(RESULTS_DIR, f"result_{idx:03d}.json")
with open(out_file, "w") as f:
    json.dump(result, f, indent=2)
print(f"[idx={idx}] Saved: {out_file}", flush=True)
```

---

## Script: `collect.py`

```python
#!/usr/bin/env python3
"""Aggregate per-worker JSON results into a CSV and print a summary."""

import os, csv, json, glob
from collections import Counter

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "data", "results")
OUT_CSV     = os.path.join(SCRIPT_DIR, "data", "region_species_test_results.csv")

files   = sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.json")))
results = [json.load(open(f)) for f in files]
results.sort(key=lambda r: r["idx"])

print(f"Found {len(results)} result files.")

fieldnames = [
    "idx", "type", "region", "name", "presence",
    "n_cells", "target_ha", "avail_ha", "tightness",
    "bare_coeff_min", "bare_coeff_max", "coeff_ratio",
    "lb_rescale", "scale_factor", "status", "status_str", "solve_s",
]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

print(f"CSV written: {OUT_CSV}\n")

print("Status summary:")
for s, c in Counter(r["status_str"] for r in results).most_common():
    print(f"  {s:25s}: {c}")

non_opt = [r for r in results if r["status_str"] not in ("OPTIMAL", "SKIP_NEG_TARGET", "SKIP_EMPTY")]
if non_opt:
    print(f"\nNon-optimal ({len(non_opt)}) — sorted by tightness desc:")
    non_opt.sort(key=lambda r: -r["tightness"])
    for r in non_opt:
        print(
            f"  [{r['status_str']:12s}]  {r['type']:5s}  n_cells={r['n_cells']:>5}  "
            f"tightness={r['tightness']:.4f}  ratio={r.get('coeff_ratio', float('nan')):6.1f}  "
            f"{r['name']} ({r['presence']}) [{r['region']}]"
        )
```

---

## Known pitfalls

### Tightness formula

`tightness = avail / lb_rescale` — **both in rescaled solver space**.
`val_vector` comes from `input_data.GBF4_SNES_pre_1750_area_sr` which is already
rescaled by `rescale_solver_input_data()`. Do **not** use `lb_raw / avail` — that
mixes raw and rescaled units and gives inverted results.

### Exclusions not applied from checkpoints

`GBF4_SNES_EXCLUDE_REGION_SPECIES` is filtered in `data.py` during `Data()` init.
A saved `data_YYYY.lz4` checkpoint bypasses this. Apply post-load filtering when
using checkpoints in diagnostic or retry scripts (see Step 5).

### PBS log buffering

Worker logs only appear after the job finishes. To get live logs for the launcher,
redirect inside the PBS script body with `exec >> log 2>&1` and set
`#PBS -o /dev/null -j oe` (as done in `submit_launch.sh` above).

### Windows-local execution (no PBS)

See the [Windows-local variant](#windows-local-variant-no-pbs) section below for
the full script set and run commands.

### NUMERIC vs INFEASIBLE distinction

- `NUMERIC`: the barrier blew up before proving anything — the problem may be infeasible
  but ill-conditioned constraints prevented Gurobi from finding the proof. Removing the
  offending constraint often lets Gurobi prove `INFEASIBLE` cleanly on the remainder.
- `INFEASIBLE`: Gurobi produced an infeasibility certificate — a definitive mathematical
  result. Check whether this is from a different constraint (compound effect) before
  concluding the excluded species was the sole cause.

---

## Windows-local variant (no PBS)

Drop-in replacement for the `submit_launch.sh` / `launch.py` / `worker.py` /
`collect.py` PBS scripts when running on a local Windows machine (no NCI
access). Same diagnostic logic — only the job-submission mechanism changes
(`subprocess.Popen` pool instead of `qsub`).

### Directory layout

```
Check_<scenario>_<base>_<target>/
    data/
        data_<BASE>.lz4      ← checkpoint from the previous transition
        settings.py          ← settings file copied from the run under test
        results/             ← auto-created; one result_NNN.json per worker
        species_list.json    ← auto-created by script_1_launch.py
    log/                      ← auto-created; launcher + worker + run_workers logs
    script_1_launch.py
    script_1_worker.py
    script_1_run_workers.py
    script_1_collect.py
    script_2_fullsolve.py    ← full re-solve with exclusions (see Step 6)
```

Edit the `# -- CONFIG --` / hardcoded constants at the top of each script:
- `LUTO_DIR`: path to the luto-2.0 source tree (e.g. `"F:/Users/jinzhu/Documents/luto-2.0"`)
- `BASE_YEAR` / `TARGET_YEAR`: the transition under test (e.g. `2045` / `2050`),
  reflected in the checkpoint filename `data_<BASE_YEAR>.lz4` and in
  `OUT_CSV = "region_species_test_results_<TARGET_YEAR>.csv"` in
  `script_1_collect.py`
- `N_WORK` in `script_1_run_workers.py`: number of parallel local workers
  (6 was used for a 32-thread workstation; each worker formulates a full
  Gurobi model, so don't oversubscribe memory/threads)

### Running it

All commands wrap `conda run -n luto` inside `powershell -Command "..."` —
`conda` is not directly recognized by the PowerShell tool otherwise. Redirect
output with `*> log\<name>.out` on the *outer* call so it captures stdout from
the inner `conda run`.

```bash
# 1. Enumerate targets -> data/species_list.json
powershell -Command "conda run -n luto python -u script_1_launch.py" *> log/launch.out

# 2. Run all workers locally, N_WORK in parallel, with resume support
powershell -Command "conda run -n luto python -u script_1_run_workers.py" *> log/run_workers.out

# 3. Collect results -> data/region_species_test_results_<TARGET_YEAR>.csv
powershell -Command "conda run -n luto python -u script_1_collect.py" *> log/collect.out

# 4. After updating exclusions in settings.py, full re-solve
powershell -Command "conda run -n luto python -u script_2_fullsolve.py" *> log/fullsolve.out
```

Steps 2–4 are long-running (minutes to hours) — launch with `run_in_background`
and poll the `.out` log rather than waiting synchronously. `.out` log files
written by PowerShell redirection (`*>`) are UTF-16 encoded; read them with a
tool that handles encoding (the `Read` tool works, raw `cat`/`type` shows
spaced-out characters).

### Script: `script_1_launch.py`

```python
#!/usr/bin/env python3
"""
Launcher: enumerate exact region-species targets for BASE_YEAR->TARGET_YEAR,
save species_list.json. Windows-local version of the Gadi PBS launch.py.
"""

import sys, os, json, importlib.util, joblib

# -- CONFIG --------------------------------------------------------------------
LUTO_DIR    = "F:/Users/jinzhu/Documents/luto-2.0"
BASE_YEAR   = 2045
TARGET_YEAR = 2050
# --------------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
LOG_DIR     = os.path.join(SCRIPT_DIR, "log")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

sys.path.insert(0, LUTO_DIR)
import luto.settings as settings

spec = importlib.util.spec_from_file_location("run_settings", os.path.join(DATA_DIR, "settings.py"))
run_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_settings)
for attr in dir(run_settings):
    if not attr.startswith("_"):
        setattr(settings, attr, getattr(run_settings, attr))

from luto.solvers.input_data import get_input_data

print(f"Loading data_{BASE_YEAR}.lz4 ...", flush=True)
data = joblib.load(os.path.join(DATA_DIR, f"data_{BASE_YEAR}.lz4"))

print(f"Building input_data for {BASE_YEAR}->{TARGET_YEAR} ...", flush=True)
input_data = get_input_data(data, BASE_YEAR, TARGET_YEAR)

all_targets = []
for region, species, presence in input_data.GBF4_SNES_region_species:
    lb_raw = input_data.limits["GBF4_SNES"].sel(dict(layer=(region, species, presence))).item()
    all_targets.append({"idx": len(all_targets), "type": "SNES",  "region": region, "name": species,   "presence": presence, "lb_raw": lb_raw})
for region, community, presence in input_data.GBF4_ECNES_region_species:
    lb_raw = input_data.limits["GBF4_ECNES"].sel(dict(layer=(region, community, presence))).item()
    all_targets.append({"idx": len(all_targets), "type": "ECNES", "region": region, "name": community, "presence": presence, "lb_raw": lb_raw})

N = len(all_targets)
print(f"\nExact target count: {N}  (SNES: {sum(1 for t in all_targets if t['type']=='SNES')}, "
      f"ECNES: {sum(1 for t in all_targets if t['type']=='ECNES')})", flush=True)

with open(os.path.join(DATA_DIR, "species_list.json"), "w") as f:
    json.dump(all_targets, f, indent=2)
print(f"Species list saved: {os.path.join(DATA_DIR, 'species_list.json')}", flush=True)

print("\nNow run the workers locally with:", flush=True)
print(f"  conda run -n luto python {os.path.join(SCRIPT_DIR, 'script_1_run_workers.py')}", flush=True)
print("After all finish, collect results with:", flush=True)
print(f"  conda run -n luto python {os.path.join(SCRIPT_DIR, 'script_1_collect.py')}", flush=True)
```

### Script: `script_1_run_workers.py`

```python
#!/usr/bin/env python3
"""
Run script_1_worker.py for every target in species_list.json, locally, in
parallel batches of N_WORK processes (Windows replacement for PBS array
submission).

Usage:
  conda run -n luto python script_1_run_workers.py
"""

import os, sys, json, subprocess, time

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
LOG_DIR     = os.path.join(SCRIPT_DIR, "log")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
WORKER      = os.path.join(SCRIPT_DIR, "script_1_worker.py")

N_WORK = 6

with open(os.path.join(DATA_DIR, "species_list.json")) as f:
    all_targets = json.load(f)

# Skip targets that already have a result on disk (resume support)
todo = []
for t in all_targets:
    idx = t["idx"]
    out_file = os.path.join(RESULTS_DIR, f"result_{idx:03d}.json")
    if os.path.exists(out_file):
        print(f"  [idx={idx}] result exists, skipping")
        continue
    todo.append(idx)

print(f"\n{len(todo)} / {len(all_targets)} targets to run, N_WORK={N_WORK}\n", flush=True)

python_exe = sys.executable
running = {}  # idx -> (Popen, out_fh, err_fh)

i = 0
while i < len(todo) or running:
    while len(running) < N_WORK and i < len(todo):
        idx = todo[i]
        i += 1
        env = os.environ.copy()
        env["PBS_ARRAY_INDEX"] = str(idx)
        out_fh = open(os.path.join(LOG_DIR, f"worker_{idx}.out"), "w")
        err_fh = open(os.path.join(LOG_DIR, f"worker_{idx}.err"), "w")
        print(f"  [idx={idx}] launching ...", flush=True)
        proc = subprocess.Popen(
            [python_exe, "-u", WORKER],
            cwd=SCRIPT_DIR, env=env, stdout=out_fh, stderr=err_fh,
        )
        running[idx] = (proc, out_fh, err_fh)

    done = []
    for idx, (proc, out_fh, err_fh) in running.items():
        ret = proc.poll()
        if ret is not None:
            out_fh.close()
            err_fh.close()
            print(f"  [idx={idx}] finished, returncode={ret}", flush=True)
            done.append(idx)
    for idx in done:
        del running[idx]

    if running:
        time.sleep(2)

print("\nAll workers finished.", flush=True)
print(f"Collect results with:\n  conda run -n luto python {os.path.join(SCRIPT_DIR, 'script_1_collect.py')}", flush=True)
```

### Script: `script_1_worker.py`

Identical diagnostic logic to `worker.py` above (same constraint construction,
tightness/coeff_ratio calculation, and result JSON schema), with two
differences:

1. `BASE_YEAR`/`TARGET_YEAR` are hardcoded constants matching the launcher.
2. `idx` comes from the `PBS_ARRAY_INDEX` env var set by
   `script_1_run_workers.py` (same mechanism as the PBS variant — no code
   change needed):

```python
idx = int(os.environ.get("PBS_ARRAY_INDEX", sys.argv[1] if len(sys.argv) > 1 else "0"))
```

Copy `worker.py`, rename to `script_1_worker.py`, and replace the
`CHECKPOINT`/`BASE_YEAR`/`TARGET_YEAR` config constants with this
transition's values (e.g. `data_2045.lz4`, `2045`, `2050`). Everything else —
model formulation, `_build_biodiv_contr_expr`, status mapping, JSON output —
is unchanged.

### Script: `script_1_collect.py`

Identical to `collect.py` above, except `OUT_CSV` is suffixed with
`TARGET_YEAR` so results from successive transitions don't overwrite each
other:

```python
OUT_CSV = os.path.join(SCRIPT_DIR, "data", f"region_species_test_results_{TARGET_YEAR}.csv")
```
