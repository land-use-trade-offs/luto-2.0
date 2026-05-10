# Skill: Creating Task Runs for Multiple LUF Scenarios

This skill documents the end-to-end workflow for creating, merging, and submitting a new batch of LUTO2 scenario runs for a LUF (Land Use Futures) iteration. It covers: writing per-group `create_tasks_*.py` task scripts (each **fully standalone** with all settings baked in), the orchestrating `merge_unique_parameters.py` (which only merges and optionally patches cluster settings), verifying settings alignment, and submitting to the cluster.

---

## Overview

Each LUF iteration lives in two places:

| Location | Purpose |
|---|---|
| `jinzhu_inspect_code/<Iteration>/` | Script source — `create_tasks_*.py` + `merge_unique_parameters.py` |
| `<Custom_runs>/<Iteration>/` | Output — generated CSVs and submitted run folders |

The design has **two pieces**:

1. **`create_tasks_*.py`** — one script per scenario group. Each is **fully self-contained**: its `GRID = {...}` holds *all* settings including cluster job settings (`MEM`, `NCPUS`, `TIME`, `QUEUE`, `RESFACTOR`, `SIM_YEARS`, `WRITE_*`, `DO_IIS`, `OBJECTIVE`). The script can be run standalone with just `--task-dir` and produces correct CSVs without any external input. Variant/sensitivity scripts extend this with a `SENSITIVITY_OVERRIDES` list and process each variant **independently** (no DataFrame concatenation across variants) to avoid NaN.
2. **`merge_unique_parameters.py`** — single configuration entry point. You set `TASK_DIR` and `TASK_SCRIPTS`. The merge script runs each task script, merges CSVs into globally unique runs, and (optionally) applies `SYSTEM_OVERRIDES` — a small dict that patches cluster/system rows in the merged template after the fact (useful for batch switches like `RESFACTOR` or `DO_IIS` without editing each task script).

**No `OVERRIDES` JSON injection** — cluster settings live in each task script's `GRID`. The merge script never passes `--overrides` to child scripts.

**No shared `_common.py`** — keeping each `GRID` inline in its own script means `merged_grid_search_parameters_unique.csv` reflects exactly the dict the user wrote, with no hidden indirection. The trade-off is that scenario-baseline duplication across scripts must be kept in sync manually (or by copy-paste from the CORE script).

The workflow:
1. Write one `create_tasks_*.py` per scenario group, each with its own complete `GRID` (cluster settings + scenario settings). Variant scripts copy CORE's `GRID` and define per-variant overrides on top.
2. Configure `merge_unique_parameters.py` (paths, `TASK_SCRIPTS`, optional `SYSTEM_OVERRIDES`); run with `SUBMIT=False` to inspect, then `SUBMIT=True` to submit.

---

## Step 1: Write `create_tasks_core.py` — the baseline `GRID`

The CORE script holds the canonical baseline `GRID = {...}` dict and produces a single run. Other scripts copy this dict and layer variant overrides on top.

**All settings live in `GRID`** — cluster job settings (`MEM`, `NCPUS`, `TIME`, `QUEUE`), working settings (`OBJECTIVE`, `RESFACTOR`, `SIM_YEARS`, `WRITE_*`, `DO_IIS`), and scenario settings. The script runs standalone.

```python
"""<Iteration> Core scenario — single run (baseline for sensitivity OAT).

Standalone script: all settings (model + cluster) live in GRID below.
Run directly with just --task-dir; no external overrides needed.
"""
import argparse
from pathlib import Path

from luto.tools.create_task_runs.helpers import (
    get_settings_df,
    get_grid_search_param_df,
    get_grid_search_settings_df,
)


# Region scope (example — customize per iteration)
NECMA_GBCMA_NRMS = ['North East', 'Goulburn Broken']


# ----------------------------------------------------------------------
# CORE grid (single-element lists -> exactly 1 run)
# All cluster + working + scenario settings live here — fully self-contained.
# ----------------------------------------------------------------------
GRID = {
        # --------------- Cluster job settings ---------------
        'MEM':                     ['128GB'],
        'WRITE_REPORT_MAX_MEM_GB': [80],
        'NCPUS':                   [32],
        'TIME':                    ['12:00:00'],
        'QUEUE':                   ['normalsr'],

        # --------------- Working settings ---------------
        'OBJECTIVE':      ['maxprofit'],
        'RESFACTOR':      [5],
        'SIM_YEARS':      [[2020, 2025, 2030, 2035, 2040, 2045, 2050]],
        'WRITE_PARALLEL': [True],
        'WRITE_THREADS':  [4],
        'DO_IIS':         [False],
        'WRITE_OUTPUTS':  [True],

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


def build_core(task_root_dir):
    """Generate CORE grid_search CSVs under ``task_root_dir``."""
    task_root_dir = Path(task_root_dir)
    task_root_dir.mkdir(parents=True, exist_ok=True)

    default_settings_df = get_settings_df(str(task_root_dir))
    grid_search_param_df = get_grid_search_param_df(GRID)
    grid_search_param_df.to_csv(task_root_dir / 'grid_search_parameters.csv', index=False)
    print(f'  Core scenario: {len(grid_search_param_df)} run(s)  (expect 1)')
    get_grid_search_settings_df(str(task_root_dir), default_settings_df, grid_search_param_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', required=True)
    args = parser.parse_args()
    build_core(args.task_dir)
```

**CRITICAL:** `AG_MANAGEMENTS` Solar/Wind flags must mirror `RENEWABLES_OPTIONS` — `write_settings` writes flat literals and does NOT re-evaluate. Sync them manually whenever toggling renewables.

---

## Step 2: Write variant scripts (sensitivity, progressive, ...)

Each variant script copies CORE's `GRID` verbatim at the top (including cluster settings), defines a list of variant overrides, and processes **each variant independently** using `update_settings`. This avoids the NaN-in-template bug that arises when different variants have different parameter keys and their DataFrames are concatenated before calling `get_grid_search_settings_df`.

### 2a. SENSITIVITY script (one-at-a-time variants)

```python
"""<Iteration> Sensitivity (One-At-A-Time from CORE).

Standalone script: all settings (model + cluster) live in GRID below.
Run directly with just --task-dir; no external overrides needed.
"""
import argparse
from pathlib import Path
import pandas as pd

from luto.tools.create_task_runs.helpers import (
    get_settings_df,
    get_grid_search_param_df,
    update_settings,
)


# --- Copy CORE GRID here verbatim from create_tasks_core.py ---
NECMA_GBCMA_NRMS = ['North East', 'Goulburn Broken']
GRID = {
    # ... same body as create_tasks_core.GRID (including cluster settings) ...
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


def build_sensitivity(task_root_dir):
    """Generate SENSITIVITY grid_search CSVs under ``task_root_dir``.

    Each variant is processed in isolation: settings start from settings.py
    defaults, then only that variant's keys are updated.  This prevents NaN
    from appearing in the template when different variants have different
    parameter keys (e.g. variant A doesn't set BIODIVERSITY_GBF_4_TARGET_DICT_*
    but variant B does — concatenating their DataFrames would insert NaN for A,
    which eval() cannot handle).
    """
    task_root_dir = Path(task_root_dir)
    task_root_dir.mkdir(parents=True, exist_ok=True)
    default_settings_df = get_settings_df(str(task_root_dir))
    defaults = default_settings_df.set_index('Name')['Default_run'].to_dict()

    all_param_rows = []
    all_run_series = []

    for run_idx, (label, variant_overrides) in enumerate(SENSITIVITY_OVERRIDES, start=1):
        # Build a complete grid for this variant — every key is explicit, no NaN possible.
        grid = {**GRID, **variant_overrides}
        df = get_grid_search_param_df(grid)
        if len(df) != 1:
            raise RuntimeError(
                f"Override '{label}' produced {len(df)} rows (expected 1). "
                "Each override must contain single-element lists only."
            )
        df.insert(0, 'sensitivity_label', label)
        df['run_idx'] = run_idx
        all_param_rows.append(df)

        # Build complete settings for this variant in isolation.
        settings_dict = dict(defaults)
        settings_dict.update(df.drop(columns=['sensitivity_label', 'run_idx']).iloc[0].to_dict())
        settings_dict = update_settings(settings_dict, f'run_{run_idx:04}')
        all_run_series.append(pd.Series(settings_dict, name=f'Run_{run_idx:04}'))

    # All series are complete — pd.concat produces a NaN-free template.
    template = pd.concat(all_run_series, axis=1).reset_index(names='Name')
    template.to_csv(task_root_dir / 'grid_search_template.csv', index=False)

    params = pd.concat(all_param_rows, ignore_index=True)
    params.to_csv(task_root_dir / 'grid_search_parameters.csv', index=False)

    vary_cols = [c for c in params.columns if params[c].nunique(dropna=False) > 1]
    params[vary_cols].to_csv(task_root_dir / 'grid_search_parameters_unique.csv', index=False)

    print(f'  Sensitivity runs: {len(params)}  (expect {len(SENSITIVITY_OVERRIDES)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', required=True)
    args = parser.parse_args()
    build_sensitivity(args.task_dir)
```

### Why `update_settings` instead of `get_grid_search_settings_df`

`get_grid_search_settings_df` takes the entire param DataFrame and calls `settings_dict.update(row.to_dict())` per row. When variants have different column sets, the concatenated DataFrame has NaN in rows not present in every variant. Those NaN values then **overwrite** the `settings.py` default, producing literal `nan` in the template CSV. `eval('nan')` → `NameError` at submission time.

The fix: use `update_settings` per variant on a complete `{**GRID, **variant_overrides}` dict — NaN can never appear because every key is explicitly set.

> **Rule:** use `get_grid_search_settings_df` only when all runs share an identical column set (e.g. a pure Cartesian grid). For OAT sensitivity with different keys per variant, always use the per-variant `update_settings` pattern above.

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

The `build_progressive` function is structurally identical to `build_sensitivity` above — use the same per-variant-independent `update_settings` pattern. The final ladder run (e.g. `river_nrm_nvis_ecnes_snes`) equals CORE, so it doubles as a CORE smoke test.

To enable IIS diagnostics on every ladder run, add `'DO_IIS': [True]` to a specific variant override (it then appears as a varying column in `merged_grid_search_parameters_unique.csv`), or put it in the CORE `GRID` and rely on `SYSTEM_OVERRIDES` in the merge script to flip it for the progressive group.

### 2c. Common sensitivity patterns

| Sensitivity | Override dict |
|---|---|
| **GHG higher ambition** | `{'GHG_EMISSIONS_LIMITS': ['high']}` |
| **Water stress** | `{'WATER_STRESS': [0.5 / 0.7 / 0.8]}` |
| **Climate** | `{'SSP': ['126' / '370']}` |
| **GBF2 cut** | `{'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [0 / 10 / 20]}` |
| **GBF4 Australia-wide** | `{'GBF4_SNES_REGION_MODE': ['Australia'], 'GBF4_ECNES_REGION_MODE': ['Australia']}` |
| **GBF4 dict targets** | `{'BIODIVERSITY_GBF_4_TARGET_SOURCE_SNES': ['dict'], 'BIODIVERSITY_GBF_4_TARGET_DICT_SNES': [{'2030': 30, '2050': 50, '2100': 50}]}` |
| **Social licence** | `{'REGIONAL_ADOPTION_CONSTRAINTS': ['NON_AG_CAP'], 'REGIONAL_ADOPTION_NON_AG_CAP': [5/10/20/30]}` |
| **Renewables ON** | `{'RENEWABLES_OPTIONS': [{'Utility Solar PV': True, 'Onshore Wind': True}], 'AG_MANAGEMENTS': [{...solar/wind: True}]}` (must override both — see CRITICAL note above) |
| **Bio contribution sharpened** | `{'BIO_CONTRIBUTION_DESTOCKING': ['GAP'], 'BIO_CONTRIBUTION_RIPARIAN_PLANTING': [1.2], ...}` — use `'GAP'` string sentinel (not `None`) for destocking to avoid CSV round-trip NaN |

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

One merge script per iteration. **Cluster + working settings already live in each task script's `GRID`** — the merge script's only job is to run the scripts, merge their CSVs, and optionally patch system settings in the merged template before submitting.

```python
"""<Iteration> merge script.

Single entry point:
  1) Run each create_tasks_*.py via subprocess (just --task-dir; each script
     is fully standalone with all settings in its own GRID).
  2) Merge per-group CSVs into one global parameters/template pair.
  3) Optionally apply SYSTEM_OVERRIDES to patch cluster/system rows in the
     merged template (e.g. switch RESFACTOR for a test run, flip DO_IIS).
  4) Optionally submit all merged runs.

Cross-platform: works identically on Windows and NCI (Linux).
"""
import os
import sys
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
    "Windows": Path("F:/Users/jinzhu/Documents/Custom_runs/<Iteration>"),
    "Linux":   Path("/g/data/jk53/jinzhu/LUTO/Custom_runs/<Iteration>"),
    "Darwin":  Path.home() / "Custom_runs" / "<Iteration>",
}
TASK_DIR = Path(os.environ.get(
    "LUTO_TASK_DIR",
    _DEFAULT_TASK_DIRS.get(platform.system(), Path.cwd() / "<Iteration>"),
))

CONDA_ENV = os.environ.get("LUTO_CONDA_ENV", "luto")

# Per-group task scripts. Output dir for each = TASK_DIR / GROUP_NAME.
# Each script is standalone — all settings (model + cluster) live inside it.
TASK_SCRIPTS = [
    ("CORE",        "create_tasks_core.py"),
    ("SENSITIVITY", "create_tasks_sensitivity.py"),
]

# Optional: patch specific rows in the merged template after merge.
# Use to override system/cluster settings across ALL runs without editing
# each task script (e.g. quick test run, enabling IIS, changing queue).
# Leave empty {} when each task script's GRID already has correct values.
SYSTEM_OVERRIDES = {
    # 'RESFACTOR':      10,      # uncomment for quick test run
    # 'DO_IIS':         False,
    # 'MEM':            '256GB',
}


# ============================================================
# CELL 2 — Run each create_tasks_*.py to generate per-group CSVs
# ============================================================
TASK_DIR.mkdir(parents=True, exist_ok=True)

_use_conda = os.environ.get("LUTO_USE_CONDA", "0") == "1" and shutil.which("conda") is not None

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
               "--task-dir", str(group_dir)]
        use_shell = platform.system() == "Windows"
    else:
        cmd = [sys.executable, str(script_path),
               "--task-dir", str(group_dir)]
        use_shell = False

    print(f"  Running {script} -> {group_dir}")
    subprocess.run(cmd, check=True, cwd=str(SCRIPTS_DIR), env=env, shell=use_shell)


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

# Apply SYSTEM_OVERRIDES — patches cluster/system rows across all runs.
if SYSTEM_OVERRIDES:
    run_cols = [c for c in merged_template.columns if c != 'Name']
    for key, val in SYSTEM_OVERRIDES.items():
        mask = merged_template['Name'] == key
        if mask.any():
            merged_template.loc[mask, run_cols] = str(val)
        else:
            print(f"  [WARNING] SYSTEM_OVERRIDES key '{key}' not found in template")
    print(f"  Applied {len(SYSTEM_OVERRIDES)} SYSTEM_OVERRIDES to merged template")

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

Before generating CSVs, audit each `create_tasks_*.py`'s `GRID` dict against `luto/settings.py`. The most common stale items are **new settings added after the iteration was forked** — typically renewable energy keys and new biodiversity keys.

```bash
grep -E "RENEWABLE_EXISTING_END_YEAR|RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS|RENEWABLE_TARGET_SCENARIO_TARGETS|INSTALL_CAPACITY_MW_HA|DO_IIS" jinzhu_inspect_code/<Iteration>/create_tasks_core.py
```

If anything in `settings.py` is missing from the script's `GRID`, add it to `GRID`. All settings — scenario, cluster, and working — belong in `GRID` in the task script.

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
| `BIO_CONTRIBUTION_DESTOCKING` | `'GAP'` | `0.75` (use `'GAP'` sentinel when the sensitivity variant wants lookup-table difference) |

---

## Step 5: Generate CSVs and inspect

```bash
# Linux / NCI
conda run -n luto python jinzhu_inspect_code/<Iteration>/merge_unique_parameters.py

# Windows (from luto conda env)
python jinzhu_inspect_code/<Iteration>/merge_unique_parameters.py
```

This subprocess-runs every `create_tasks_*.py` (passing only `--task-dir`), then writes per-group:
- `grid_search_parameters.csv` — all permutations with `run_idx`
- `grid_search_template.csv` — full settings, one column per run (**NaN-free**)
- `non_str_val.txt` — setting names needing `eval()` (non-string types)

And to `TASK_DIR`:
- `merged_grid_search_parameters_unique.csv` — varying columns + `scenario_group` + `global_run_idx`
- `merged_grid_search_template.csv` — all runs with globally unique `Run_G0001`, `Run_G0002`, ...

Inspect before submitting:
```python
import pandas as pd
df = pd.read_csv(r"F:\Users\jinzhu\Documents\Custom_run\<Iteration>\merged_grid_search_parameters_unique.csv")
print(df.head())

# Verify NaN-free template
tmpl = pd.read_csv(r"...\merged_grid_search_template.csv")
assert tmpl.isnull().sum().sum() == 0, "Template has NaN — check variant key coverage"
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
- [ ] Verify the CORE `GRID` carries every scenario-relevant key in `luto/settings.py` (esp. new biodiversity dict keys, renewable energy keys); cluster + working settings also live here
- [ ] Copy variant scripts (`create_tasks_sensitivity.py`, `create_tasks_progressive.py`, ...) — copy CORE's `GRID` body verbatim into each, adjust the `*_OVERRIDES` list of variants; use the `update_settings` per-variant pattern (not `get_grid_search_settings_df` on a concatenated DataFrame)
- [ ] Configure `merge_unique_parameters.py`:
  - [ ] `TASK_DIR` → new iteration output root
  - [ ] `TASK_SCRIPTS` → list every `create_tasks_*.py`
  - [ ] `SYSTEM_OVERRIDES` → leave `{}` for normal runs; populate only to batch-patch cluster settings without editing each script (e.g. `{'RESFACTOR': 10}` for a test run)
  - [ ] `CONDA_ENV` → `luto` (or override)
- [ ] Run merge with `SUBMIT=False` and inspect `merged_grid_search_parameters_unique.csv` — verify it contains the parameters you expect to vary, and *only* those
- [ ] Assert zero NaN in `merged_grid_search_template.csv`
- [ ] Set `SUBMIT=True` and re-run

  - [ ] `CONDA_ENV` → `luto` (or override)
- [ ] Run merge with `SUBMIT=False` and inspect `merged_grid_search_parameters_unique.csv` — verify it contains the parameters you expect to vary, and *only* those
- [ ] Set `SUBMIT=True` and re-run
