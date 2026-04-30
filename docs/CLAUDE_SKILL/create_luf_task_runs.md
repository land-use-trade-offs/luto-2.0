# Skill: Creating Task Runs for Multiple LUF Scenarios

This skill documents the end-to-end workflow for creating, merging, and submitting a new batch of LUTO2 scenario runs for a LUF (Land Use Futures) iteration. It covers: writing per-group `create_tasks_*.py` task scripts (each with its own inline `GRID` dict), the orchestrating `merge_unique_parameters.py`, verifying settings alignment, and submitting to the cluster.

---

## Overview

Each LUF iteration lives in two places:

| Location | Purpose |
|---|---|
| `jinzhu_inspect_code/<Iteration>/` | Script source — `create_tasks_*.py` + `merge_unique_parameters.py` |
| `<Custom_runs>/<Iteration>/` | Output — generated CSVs, `_overrides.json`, and submitted run folders |

The current design has **two pieces**:

1. **`create_tasks_*.py`** — one script per scenario group. Each contains its own plain `GRID = {...}` dict (single-element lists for the baseline) and exposes `build_<group>(task_root_dir, overrides=None)` plus a CLI `--task-dir / --overrides` so the merge script can drive it via subprocess. Variant scripts (e.g. sensitivity) define a list of variant overrides applied on top of `GRID`.
2. **`merge_unique_parameters.py`** — single configuration entry point. You set `TASK_DIR` (one path), `OVERRIDES` (cluster + working settings shared by every group, including `DO_IIS`), and `TASK_SCRIPTS` (list of `(group_name, script_filename)`). The merge script orchestrates the per-group subprocesses, merges CSVs into globally unique runs, and (optionally) submits.

**No shared `_common.py`** — keeping each `GRID` inline in its own script means `merged_grid_search_parameters_unique.csv` reflects exactly the dict the user wrote, with no hidden indirection. The trade-off is that scenario-baseline duplication across scripts must be kept in sync manually (or by copy-paste from the CORE script).

The workflow has two steps:
1. Write one `create_tasks_*.py` per scenario group, each with its own `GRID` dict (the variant scripts copy CORE's `GRID` and apply per-variant overrides on top)
2. Configure `merge_unique_parameters.py` (paths, `OVERRIDES`, `TASK_SCRIPTS`); run with `SUBMIT=False` to inspect, then `SUBMIT=True` to submit

---

## Step 1: Write `create_tasks_core.py` — the baseline `GRID`

The CORE script holds the canonical baseline `GRID = {...}` dict and produces a single run. Other scripts copy this dict and layer variant overrides on top.

**Cluster + working settings (`MEM`, `NCPUS`, `TIME`, `QUEUE`, `OBJECTIVE`, `RESFACTOR`, `SIM_YEARS`, `WRITE_PARALLEL`, `WRITE_THREADS`, `DO_IIS`) are intentionally NOT in `GRID`** — they live in `OVERRIDES` inside the merge script so they can be edited in one place without touching the scenario logic.

```python
"""<Iteration> Core scenario — single run (baseline for sensitivity OAT).

Plain ``GRID`` dict; cluster + working settings (MEM/NCPUS/TIME/QUEUE/RESFACTOR/
SIM_YEARS/WRITE_*/OBJECTIVE/DO_IIS) are supplied by ``OVERRIDES`` in
merge_unique_parameters.py and applied on top before generating CSVs.
"""
import argparse
import json
from pathlib import Path

from luto.tools.create_task_runs.helpers import (
    get_settings_df,
    get_grid_search_param_df,
    get_grid_search_settings_df,
)


# Region scope (example — customize per iteration)
NECMA_GBCMA_NRMS = ['North East', 'Goulburn Broken']


# ----------------------------------------------------------------------
# CORE grid (single-element lists -> exactly 1 run with no variant overrides)
# Cluster + working settings excluded — see OVERRIDES in merge_unique_parameters.py.
# ----------------------------------------------------------------------
GRID = {
        # --------------- Scenarios ---------------
        'SSP': ['245'],
        'CARBON_EFFECTS_WINDOW': [60],
        'RISK_OF_REVERSAL': [0],
        'FIRE_RISK': ['med'],
        'CONVERGENCE': [2050],
        'CO2_FERT': ['off'],
        'APPLY_DEMAND_MULTIPLIERS': [True],
        'PRODUCTIVITY_TREND': ['BAU'],

        # --------------- Economics ---------------
        'DYNAMIC_PRICE': [True],
        'AMORTISE_UPFRONT_COSTS': [False],
        'DISCOUNT_RATE': [0.07],
        'AMORTISATION_PERIOD': [30],
        'CARBON_PRICE_COSTANT': [0],
        'BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR': [100],
        'SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR': [100],
        'HIR_CEILING_PERCENTAGE': [0.8],

        # --------------- Target deviation weight ---------------
        'SOLVER_WEIGHT_DEMAND': [1],
        'SOLVER_WEIGHT_GHG': [1],
        'SOLVER_WEIGHT_WATER': [1],

        # --------------- Social license ---------------
        'EXCLUDE_NO_GO_LU': [False],
        'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],
        'REGIONAL_ADOPTION_NON_AG_CAP': [15],
        'REGIONAL_ADOPTION_NON_AG_REGION': ['NRM'],
        'REGIONAL_ADOPTION_ZONE': ['NRM_CODE'],

        # --------------- GHG settings ---------------
        'GHG_EMISSIONS_LIMITS': ['low'],
        'CARBON_PRICES_FIELD': ['CONSTANT'],
        'GHG_CONSTRAINT_TYPE': ['hard'],
        'USE_GHG_SCOPE_1': [True],

        # --------------- Water constraints ---------------
        'WATER_REGION_DEF': ['River Region'],
        'WATER_LIMITS': ['on'],
        'WATER_CONSTRAINT_TYPE': ['hard'],
        'WATER_STRESS': [0.6],
        'WATER_CLIMATE_CHANGE_IMPACT': ['on'],
        'LIVESTOCK_DRINKING_WATER': [1],
        'INCLUDE_WATER_LICENSE_COSTS': [1],

        # --------------- Biodiversity overall ---------------
        'BIO_QUALITY_LAYER': ['Suitability'],
        'CONTRIBUTION_PERCENTILE': ['USER_DEFINED'],
        'CONNECTIVITY_SOURCE': ['NCI'],
        'CONNECTIVITY_LB': [0.7],

        # --------------- Biodiversity contribution ---------------
        'BIO_CONTRIBUTION_LDS': [0.75],
        'BIO_CONTRIBUTION_ENV_PLANTING': [0.7],
        'BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK': [0.12],
        'BIO_CONTRIBUTION_CARBON_PLANTING_BELT': [0.12],
        'BIO_CONTRIBUTION_RIPARIAN_PLANTING': [1.0],
        'BIO_CONTRIBUTION_AGROFORESTRY': [0.7],
        'BIO_CONTRIBUTION_BECCS': [0],
        'BIO_CONTRIBUTION_DESTOCKING': [0.75],

        # --------------- GBF 2 ---------------
        'BIODIVERSITY_TARGET_GBF_2': ['high'],
        'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [15],
        'GBF2_CONSTRAINT_TYPE': ['hard'],

        # --------------- GBF 3 / 4 / 8 (customize per iteration) ---------------
        'BIODIVERSITY_TARGET_GBF_3_NVIS': ['high'],
        'GBF3_NVIS_TARGET_CLASS': ['MVS'],
        'GBF3_NVIS_REGION_MODE': ['NRM'],
        'GBF3_NVIS_SELECTED_REGIONS': [NECMA_GBCMA_NRMS],
        'BIODIVERSITY_TARGET_GBF_3_IBRA': ['off'],
        'BIODIVERSITY_TARGET_GBF_4_SNES': ['on'],
        'GBF4_SNES_REGION_MODE': ['NRM'],
        'GBF4_SNES_SELECTED_REGIONS': [NECMA_GBCMA_NRMS],
        'BIODIVERSITY_TARGET_GBF_4_ECNES': ['on'],
        'GBF4_ECNES_REGION_MODE': ['NRM'],
        'GBF4_ECNES_SELECTED_REGIONS': [NECMA_GBCMA_NRMS],
        'BIODIVERSITY_TARGET_GBF_8': ['off'],

        # --------------- Renewable energy ---------------
        'RENEWABLES_OPTIONS': [{'Utility Solar PV': False, 'Onshore Wind': False}],

        # --------------- Objective function weights ---------------
        'SOLVE_WEIGHT_ALPHA': [1],
        'SOLVE_WEIGHT_BETA': [0.5],

        # --------------- Ag management ---------------
        'AG_MANAGEMENTS': [{
            'Asparagopsis taxiformis': True, 'Precision Agriculture': True,
            'Ecological Grazing': False, 'Savanna Burning': True,
            'AgTech EI': True, 'Biochar': True,
            'HIR - Beef': True, 'HIR - Sheep': True,
            'Utility Solar PV': False, 'Onshore Wind': False,
        }],
        'AG_MANAGEMENTS_REVERSIBLE': [{
            'Asparagopsis taxiformis': True, 'Precision Agriculture': True,
            'Ecological Grazing': True, 'Savanna Burning': True,
            'AgTech EI': True, 'Biochar': True,
            'HIR - Beef': False, 'HIR - Sheep': False,
            'Utility Solar PV': False, 'Onshore Wind': False,
        }],

        # --------------- Non-agricultural land uses ---------------
        'NON_AG_LAND_USES': [{
            'Environmental Plantings': True, 'Riparian Plantings': True,
            'Sheep Agroforestry': True, 'Beef Agroforestry': True,
            'Carbon Plantings (Block)': True,
            'Sheep Carbon Plantings (Belt)': True,
            'Beef Carbon Plantings (Belt)': True,
            'BECCS': False, 'Destocked - natural land': True,
        }],
        'NON_AG_LAND_USES_REVERSIBLE': [{
            'Environmental Plantings': False, 'Riparian Plantings': False,
            'Sheep Agroforestry': False, 'Beef Agroforestry': False,
            'Carbon Plantings (Block)': False,
            'Sheep Carbon Plantings (Belt)': False,
            'Beef Carbon Plantings (Belt)': False,
            'BECCS': False, 'Destocked - natural land': True,
        }],

        # --------------- Dietary ---------------
        'DIET_DOM': ['BAU'],
        'DIET_GLOB': ['BAU'],
        'WASTE': [1],
        'FEED_EFFICIENCY': ['BAU'],
        'IMPORT_TREND': ['Trend'],
}


def build_core(task_root_dir, overrides=None):
    """Generate CORE grid_search CSVs under ``task_root_dir``."""
    task_root_dir = Path(task_root_dir)
    task_root_dir.mkdir(parents=True, exist_ok=True)

    grid = dict(GRID)
    if overrides:
        grid.update(overrides)

    default_settings_df = get_settings_df(str(task_root_dir))
    grid_search_param_df = get_grid_search_param_df(grid)
    grid_search_param_df.to_csv(task_root_dir / 'grid_search_parameters.csv', index=False)
    print(f'  Core scenario: {len(grid_search_param_df)} run(s)  (expect 1)')
    get_grid_search_settings_df(str(task_root_dir), default_settings_df, grid_search_param_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', required=True)
    parser.add_argument('--overrides', default=None,
                        help='Path to JSON file with single-list overrides.')
    args = parser.parse_args()

    overrides = json.loads(Path(args.overrides).read_text()) if args.overrides else None
    build_core(args.task_dir, overrides=overrides)
```

**CRITICAL:** `AG_MANAGEMENTS` Solar/Wind flags must mirror `RENEWABLES_OPTIONS` — `write_settings` writes flat literals and does NOT re-evaluate. Sync them manually whenever toggling renewables.

---

## Step 2: Write variant scripts (sensitivity, progressive, ...)

Each variant script copies CORE's `GRID` verbatim at the top of the file (so each script's CSVs reflect exactly what's in that script — no hidden imports), then defines a list of variant overrides applied on top of `GRID`. The merge script invokes each via `conda run` with `--task-dir` + `--overrides`.

### 2a. SENSITIVITY script (one-at-a-time variants)

Each variant is a single-list override applied on top of `GRID` (and on top of the merge script's `OVERRIDES`):

```python
"""<Iteration> Sensitivity (One-At-A-Time from CORE)."""
import argparse
import json
from pathlib import Path
import pandas as pd

from luto.tools.create_task_runs.helpers import (
    get_settings_df,
    get_grid_search_param_df,
    get_grid_search_settings_df,
)


# --- Copy CORE GRID here verbatim from create_tasks_core.py ---
NECMA_GBCMA_NRMS = ['North East', 'Goulburn Broken']
GRID = {
    # ... same body as create_tasks_core.GRID ...
}


# Each entry: (label, {param: [value]}) — applied on top of GRID.
SENSITIVITY_OVERRIDES = [
    ('ghg_high',         {'GHG_EMISSIONS_LIMITS': ['high']}),
    ('water_50',         {'WATER_STRESS': [0.5]}),
    ('water_70',         {'WATER_STRESS': [0.7]}),
    ('ssp126_rcp26',     {'SSP': ['126']}),
    ('gbf2_cut_0',       {'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [0]}),
    ('gbf4_australia',   {
        'GBF4_SNES_REGION_MODE':  ['Australia'],
        'GBF4_ECNES_REGION_MODE': ['Australia'],
    }),
    ('social_5',         {'REGIONAL_ADOPTION_CONSTRAINTS': ['NON_AG_CAP'],
                          'REGIONAL_ADOPTION_NON_AG_CAP': [5]}),
    # ... add more variants
]


def build_sensitivity(task_root_dir, overrides=None):
    """Generate SENSITIVITY grid_search CSVs.

    ``overrides`` is applied on top of CORE for *every* variant before the
    variant's own override (used for cluster/working settings).
    """
    task_root_dir = Path(task_root_dir)
    task_root_dir.mkdir(parents=True, exist_ok=True)
    default_settings_df = get_settings_df(str(task_root_dir))

    rows = []
    for label, variant_overrides in SENSITIVITY_OVERRIDES:
        grid = dict(GRID)
        if overrides:
            grid.update(overrides)
        grid.update(variant_overrides)
        df = get_grid_search_param_df(grid)
        if len(df) != 1:
            raise RuntimeError(
                f"Override '{label}' produced {len(df)} rows (expected 1). "
                "Each override must contain single-element lists only."
            )
        df.insert(0, 'sensitivity_label', label)
        rows.append(df)

    grid_search_param_df = pd.concat(rows, ignore_index=True)
    grid_search_param_df['run_idx'] = range(1, len(grid_search_param_df) + 1)
    grid_search_param_df.to_csv(task_root_dir / 'grid_search_parameters.csv', index=False)

    print(f'  Sensitivity runs: {len(grid_search_param_df)}  '
          f'(expect {len(SENSITIVITY_OVERRIDES)})')
    settings_param_df = grid_search_param_df.drop(columns=['sensitivity_label'])
    get_grid_search_settings_df(str(task_root_dir), default_settings_df, settings_param_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', required=True)
    parser.add_argument('--overrides', default=None)
    args = parser.parse_args()

    overrides = json.loads(Path(args.overrides).read_text()) if args.overrides else None
    build_sensitivity(args.task_dir, overrides=overrides)
```

### 2b. PROGRESSIVE feasibility ladder (optional)

When a CORE scenario combines several aggressive constraints that may render the model infeasible (e.g. tight water + multiple GBF targets restricted to small NRMs), add a **progressive ladder** group: a sequence of runs ordered easiest -> hardest along orthogonal dimensions, so the first failure pinpoints the binding constraint.

Pattern:

```python
WATER_REGIONS = [
    ('drain', 'Drainage Division'),    # looser
    ('river', 'River Region'),         # tighter (= CORE)
]
BIO_REGION_SCOPES = [
    ('aus', {'GBF3_NVIS_REGION_MODE': ['Australia'],
             'GBF4_SNES_REGION_MODE': ['Australia'],
             'GBF4_ECNES_REGION_MODE': ['Australia']}),
    ('nrm', {}),                       # = CORE NRM-restricted
]
BIO_TARGET_SETS = [
    ('nvis',            {'BIODIVERSITY_TARGET_GBF_4_ECNES': ['off'],
                         'BIODIVERSITY_TARGET_GBF_4_SNES':  ['off']}),
    ('nvis_ecnes',      {'BIODIVERSITY_TARGET_GBF_4_ECNES': ['on'],
                         'BIODIVERSITY_TARGET_GBF_4_SNES':  ['off']}),
    ('nvis_ecnes_snes', {'BIODIVERSITY_TARGET_GBF_4_ECNES': ['on'],
                         'BIODIVERSITY_TARGET_GBF_4_SNES':  ['on']}),
]

PROGRESSIVE_OVERRIDES = []
for wlbl, wdef in WATER_REGIONS:
    for rlbl, rov in BIO_REGION_SCOPES:
        for blbl, bov in BIO_TARGET_SETS:
            ov = {'WATER_REGION_DEF': [wdef]}
            ov.update(rov); ov.update(bov)
            PROGRESSIVE_OVERRIDES.append((f'{wlbl}_{rlbl}_{blbl}', ov))
```

The `build_progressive` function is structurally identical to `build_sensitivity` — iterate over `PROGRESSIVE_OVERRIDES`, copy `GRID`, apply variant override, concat. The final ladder run (e.g. `river_nrm_nvis_ecnes_snes`) equals CORE, so it doubles as a CORE smoke test.

To enable IIS diagnostics on every ladder run, set `'DO_IIS': [True]` in the merge script's `OVERRIDES` (applies globally) — or add `'DO_IIS': [True]` to a specific variant override (then `DO_IIS` will appear as a varying column in `merged_grid_search_parameters_unique.csv`).

### 2c. Common sensitivity patterns

| Sensitivity | Override dict |
|---|---|
| **GHG higher ambition** | `{'GHG_EMISSIONS_LIMITS': ['high']}` |
| **Water stress** | `{'WATER_STRESS': [0.5 / 0.7 / 0.8]}` |
| **Climate** | `{'SSP': ['126' / '370']}` |
| **GBF2 cut** | `{'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [0 / 10 / 20]}` |
| **GBF4 Australia-wide** | `{'GBF4_SNES_REGION_MODE': ['Australia'], 'GBF4_ECNES_REGION_MODE': ['Australia']}` |
| **Social licence** | `{'REGIONAL_ADOPTION_CONSTRAINTS': ['NON_AG_CAP'], 'REGIONAL_ADOPTION_NON_AG_CAP': [5/10/20/30]}` |
| **Renewables ON** | `{'RENEWABLES_OPTIONS': [{'Utility Solar PV': True, 'Onshore Wind': True}], 'AG_MANAGEMENTS': [{...solar/wind: True}]}` (must override both — see CRITICAL note above) |

### 2d. Removing redundant permutations

When a script does Cartesian-product variants (rare in OAT designs), use a `duplicate_runs` filter to drop logically-equivalent rows. Example: when `BIODIVERSITY_TARGET_GBF_2 == 'off'`, all `GBF2_*_CUT` values produce the same run.

```python
duplicate_runs = {
    'BIODIVERSITY_TARGET_GBF_2': ('off', 'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT'),
}

rm_idx = []
for _, row in df.iterrows():
    for k, (off_val, dup_col) in duplicate_runs.items():
        if row[k] == off_val and str(row[dup_col]) != str(grid[dup_col][0]):
            rm_idx.append(row['run_idx'])
df = df[~df['run_idx'].isin(rm_idx)]
```

---

## Step 3: Write `merge_unique_parameters.py`

One merge script per iteration. **All cluster + working settings live here in `OVERRIDES`** so you only edit one file to change MEM/NCPUS/SIM_YEARS/etc.

```python
"""<Iteration> merge script.

Single entry point: sets paths + OVERRIDES, runs each create_tasks_*.py via
conda subprocess (passing --task-dir + --overrides JSON), merges per-group
CSVs into globally unique runs, and optionally submits.

Cross-platform: works identically on Windows and NCI (Linux). Paths are
auto-derived from this file's location; TASK_DIR picks a per-OS default
and can be overridden via the LUTO_TASK_DIR env var.
"""
import os
import sys
import json
import shutil
import platform
import subprocess
from pathlib import Path
import pandas as pd


# ============================================================
# CELL 1 — Configuration
# ============================================================
SUBMIT = False   # <-- True to submit after inspecting CSVs

# --------------- Paths (auto cross-platform) ---------------
SCRIPTS_DIR = Path(__file__).resolve().parent       # .../jinzhu_inspect_code/<Iteration>
LUTO_ROOT   = SCRIPTS_DIR.parents[1]                # .../luto-2.0

# TASK_DIR — output root. Override via LUTO_TASK_DIR env var; otherwise
# pick a sensible per-OS default.
_DEFAULT_TASK_DIRS = {
    "Windows": Path("F:/Users/jinzhu/Documents/Custome_run/<Iteration>"),
    "Linux":   Path("/g/data/jk53/jinzhu/LUTO/Custom_runs/<Iteration>"),
    "Darwin":  Path.home() / "Custom_runs" / "<Iteration>",
}
TASK_DIR = Path(os.environ.get(
    "LUTO_TASK_DIR",
    _DEFAULT_TASK_DIRS.get(platform.system(), Path.cwd() / "<Iteration>"),
))

CONDA_ENV = os.environ.get("LUTO_CONDA_ENV", "luto")

# Per-group task scripts. Output dir for each = TASK_DIR / GROUP_NAME.
TASK_SCRIPTS = [
    ("CORE",        "create_tasks_core.py"),
    ("SENSITIVITY", "create_tasks_sensitivity.py"),
]

# Cluster + working settings applied on top of every group's GRID.
# Each value MUST be a single-element list.
OVERRIDES = {
    # --- Cluster job settings ---
    'MEM':                     ['128GB'],
    'WRITE_REPORT_MAX_MEM_GB': [80],
    'NCPUS':                   [32],
    'TIME':                    ['12:00:00'],
    'QUEUE':                   ['normalsr'],

    # --- Working settings ---
    'OBJECTIVE':      ['maxprofit'],
    'RESFACTOR':      [5],
    'SIM_YEARS':      [[2020, 2025, 2030, 2035, 2040, 2045, 2050]],
    'WRITE_PARALLEL': [True],
    'WRITE_THREADS':  [4],
    'DO_IIS':         [False],   # set True to enable IIS diagnostics for all runs
}


# ============================================================
# CELL 2 — Run each create_tasks_*.py via subprocess
# ============================================================
TASK_DIR.mkdir(parents=True, exist_ok=True)

overrides_json = TASK_DIR / "_overrides.json"
overrides_json.write_text(json.dumps(OVERRIDES, indent=2))

# List-form subprocess args (no shell=True) -> identical on Windows + Linux.
# Fall back to current Python if `conda` isn't on PATH.
_use_conda = shutil.which("conda") is not None

GROUPS = {}
print(f"=== Step 1: Generating per-group CSVs ({platform.system()}) ===")
env = {**os.environ, "PYTHONPATH": str(LUTO_ROOT)}
for group_name, script in TASK_SCRIPTS:
    group_dir = TASK_DIR / group_name
    GROUPS[group_name] = group_dir
    script_path = SCRIPTS_DIR / script

    if _use_conda:
        cmd = ["conda", "run", "-n", CONDA_ENV, "--no-capture-output",
               "python", str(script_path),
               "--task-dir", str(group_dir),
               "--overrides", str(overrides_json)]
    else:
        cmd = [sys.executable, str(script_path),
               "--task-dir", str(group_dir),
               "--overrides", str(overrides_json)]

    print(f"  Running {script} -> {group_dir}")
    subprocess.run(cmd, check=True, cwd=str(SCRIPTS_DIR), env=env)


# ============================================================
# CELL 3 — Merge per-group CSVs
# ============================================================
OUT_UNIQUE   = TASK_DIR / "merged_grid_search_parameters_unique.csv"
OUT_TEMPLATE = TASK_DIR / "merged_grid_search_template.csv"

params_dfs, template_dfs, global_idx = [], [], 1
for group_name, task_root in GROUPS.items():
    p_path  = task_root / "grid_search_parameters.csv"
    t_path  = task_root / "grid_search_template.csv"
    ns_path = task_root / "non_str_val.txt"
    if any(not p.exists() for p in [p_path, t_path, ns_path]):
        print(f"  [WARNING] Missing CSVs for '{group_name}' — skipping")
        continue

    params = pd.read_csv(p_path)
    n = len(params)
    params.insert(0, "scenario_group", group_name)
    params.insert(1, "global_run_idx", range(global_idx, global_idx + n))
    params["local_run_idx"] = params["run_idx"]
    params_dfs.append(params)

    tmpl = pd.read_csv(t_path)
    rename_map = {c: f"Run_G{global_idx + i:04d}"
                  for i, c in enumerate(c for c in tmpl.columns if c != "Name")}
    tmpl = tmpl.rename(columns=rename_map)
    extra = pd.DataFrame({
        "Name": ["scenario_group", "global_run_idx"],
        **{new: [group_name, global_idx + i]
           for i, new in enumerate(rename_map.values())}
    })
    tmpl = pd.concat([extra, tmpl], ignore_index=True)
    template_dfs.append(tmpl)

    global_idx += n

merged_params = pd.concat(params_dfs, ignore_index=True)
vary_cols = ["scenario_group", "global_run_idx", "local_run_idx"] + [
    c for c in merged_params.columns
    if c not in ("scenario_group", "global_run_idx", "local_run_idx", "run_idx")
    and merged_params[c].nunique(dropna=False) > 1
]
merged_unique = merged_params[vary_cols]

merged_template = template_dfs[0]
for t in template_dfs[1:]:
    merged_template = merged_template.merge(t, on="Name", how="outer")

merged_unique.to_csv(OUT_UNIQUE, index=False)
merged_template.to_csv(OUT_TEMPLATE, index=False)
print(f"\nMerged {len(merged_params)} runs across {len(params_dfs)} groups")


# ============================================================
# CELL 4 — Submit
# ============================================================
if SUBMIT:
    from luto.tools.create_task_runs.helpers import create_task_runs

    non_str_src = next(iter(GROUPS.values())) / "non_str_val.txt"
    non_str_dst = TASK_DIR / "non_str_val.txt"
    if not non_str_dst.exists() or not non_str_dst.samefile(non_str_src):
        shutil.copy(non_str_src, non_str_dst)

    create_task_runs(str(TASK_DIR), merged_template, mode='single',
                     n_workers=1, max_concurrent_tasks=200)
else:
    print("\nSUBMIT=False — set SUBMIT=True to submit.")
```

### Single-group case

If only one group exists (e.g. a one-off run): set `TASK_SCRIPTS = [("MyRun", "create_tasks_myrun.py")]`. The merge logic still works — outer-join loop runs zero times, `vary_cols` will only contain the index columns (everything is fixed), but `merged_grid_search_template.csv` retains all settings.

---

## Step 4: Verify settings alignment with `settings.py`

Before generating CSVs, audit each `create_tasks_*.py`'s `GRID` dict against `luto/settings.py`. The most common stale items are **new settings added after the iteration was forked** — typically renewable energy keys, plus the `DO_IIS` flag.

```bash
grep -E "RENEWABLE_EXISTING_END_YEAR|RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS|RENEWABLE_TARGET_SCENARIO_TARGETS|INSTALL_CAPACITY_MW_HA|DO_IIS" jinzhu_inspect_code/<Iteration>/create_tasks_core.py
```

If anything in `settings.py` is missing from the script's `GRID`, add it. Any setting expected to be the same across all runs (cluster, working, `DO_IIS`) goes into the merge script's `OVERRIDES` instead so it doesn't need to be repeated in every variant script.

### Settings that are intentionally different from `settings.py` defaults

These are scenario spec overrides — do NOT "fix" them to match the defaults:

| Setting | `settings.py` default | LUF task scripts |
|---|---|---|
| `IMPORT_TREND` | `'Static'` | `'Trend'` |
| `DYNAMIC_PRICE` | `False` | `True` |
| `CARBON_EFFECTS_WINDOW` | `50` | `60` |
| `GHG_EMISSIONS_LIMITS` | `'high'` | `'low'` (core) |
| `BIO_CONTRIBUTION_ENV_PLANTING` | `0.8` | `0.7` |
| `BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK/BELT` | `0.1` | `0.12` |
| `BIO_CONTRIBUTION_RIPARIAN_PLANTING` | `1.2` | `1.0` |
| `BIO_CONTRIBUTION_AGROFORESTRY` | `0.75` | `0.7` |
| `BIO_CONTRIBUTION_DESTOCKING` | `None` | `0.75` |

---

## Step 5: Generate CSVs and inspect

```bash
# Linux / NCI
conda run -n luto python jinzhu_inspect_code/<Iteration>/merge_unique_parameters.py

# Windows (from luto conda env)
python jinzhu_inspect_code/<Iteration>/merge_unique_parameters.py
```

This subprocess-runs every `create_tasks_*.py` (passing `--task-dir` + `--overrides _overrides.json`), then writes per-group:
- `grid_search_parameters.csv` — all permutations with `run_idx`
- `grid_search_template.csv` — full settings, one column per run
- `non_str_val.txt` — setting names needing `eval()` (non-string types)

And to `TASK_DIR`:
- `_overrides.json` — frozen snapshot of the merge script's `OVERRIDES`
- `merged_grid_search_parameters_unique.csv` — varying columns + `scenario_group` + `global_run_idx`
- `merged_grid_search_template.csv` — all runs with globally unique `Run_G0001`, `Run_G0002`, ...

Inspect before submitting:
```python
import pandas as pd
df = pd.read_csv(r"F:\Users\jinzhu\Documents\Custome_run\<Iteration>\merged_grid_search_parameters_unique.csv")
print(df.head())
```

---

## Step 6: Submit runs

Set `SUBMIT = True` and re-run the merge script.

- **Cluster (NCI):** use `mode='cluster'` in `create_task_runs` — submits PBS jobs via `qsub`. Monitor with `qstat -u jw6041`.
- **Windows (single mode):** `mode='single'` deposits `python_script.py` in each run dir without executing. Run each manually, e.g. `python Custom_runs/<Iteration>/Run_G0001/python_script.py`. See [submit_task_runs_windows.md](submit_task_runs_windows.md) for `run_all.py`.

`create_task_runs` then:
1. Creates `Run_G0001/`, `Run_G0002/`, ... under `TASK_DIR`
2. Copies `luto/` source + creates `input/` symlink in each
3. Writes the run-specific `luto/settings.py` from the template column
4. Submits PBS jobs (cluster mode) or stages scripts (single mode)

---

## Checklist for a new iteration

- [ ] Copy `create_tasks_core.py` from the previous iteration; update the docstring + `GRID` baseline + any region scope (e.g. `NECMA_GBCMA_NRMS`)
- [ ] Verify the CORE `GRID` carries every scenario-relevant key in `luto/settings.py` (esp. new renewable energy keys); leave cluster / working / `DO_IIS` to `OVERRIDES`
- [ ] Copy variant scripts (`create_tasks_sensitivity.py`, `create_tasks_progressive.py`, ...) — copy CORE's `GRID` body verbatim into each, adjust the `*_OVERRIDES` list of variants
- [ ] Configure `merge_unique_parameters.py`:
  - [ ] `TASK_DIR` → new iteration output root
  - [ ] `OVERRIDES` → cluster (MEM/NCPUS/TIME/QUEUE) + working (RESFACTOR/SIM_YEARS/`DO_IIS`/...) settings
  - [ ] `TASK_SCRIPTS` → list every `create_tasks_*.py`
  - [ ] `CONDA_ENV` → `luto` (or override)
- [ ] Run merge with `SUBMIT=False` and inspect `merged_grid_search_parameters_unique.csv` — verify it contains the parameters you expect to vary, and *only* those
- [ ] Set `SUBMIT=True` and re-run
