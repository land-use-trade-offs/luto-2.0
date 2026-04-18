# Skill: Creating Task Runs for Multiple LUF Scenarios

This skill documents the end-to-end workflow for creating, merging, and submitting a new batch of LUTO2 scenario runs for a LUF (Land Use Futures) iteration. It covers: writing `create_tasks_*.py` scripts, writing `merge_unique_parameters.py`, verifying settings alignment, and submitting to the cluster.

---

## Overview

Each LUF iteration lives in two places:

| Location | Purpose |
|---|---|
| `jinzhu_inspect_code/LUF_Nth_iteration/` | Script source — `create_tasks_*.py` + `merge_unique_parameters.py` |
| `/g/data/jk53/jinzhu/LUTO/Custom_runs/LUF_Nth_iteration/` | Output — generated CSVs and submitted run folders |

The workflow has three steps:
1. Write one `create_tasks_*.py` per scenario group
2. Write `merge_unique_parameters.py` to orchestrate all groups
3. Run merge (inspect CSVs), then re-run with `SUBMIT=True`

---

## Step 1: Write `create_tasks_*.py` for each scenario group

Each script defines a `grid_search` dict of parameter lists. The Cartesian product of all lists becomes the set of runs. Single-element lists are fixed values; multi-element lists are the axes of variation.

### File layout

```
jinzhu_inspect_code/LUF_Nth_iteration/
  create_tasks_core.py
  create_tasks_sensitivity_ecnes.py
  create_tasks_sensitivity_regional.py
  create_tasks_sensitivity_gbf2_area.py
  create_tasks_sensitivity_lower_productivity.py
  create_tasks_sensitivity_alt_productivity.py
  merge_unique_parameters.py
```

### Script skeleton

```python
import numpy as np
import pandas as pd
from luto.tools.create_task_runs.helpers import (
    get_settings_df,
    get_grid_search_param_df,
    get_grid_search_settings_df,
)

# One-line description of what makes this group different from core.
TASK_ROOT_DIR = "/g/data/jk53/jinzhu/LUTO/Custom_runs/LUF_Nth_iteration/<GroupName>"

grid_search = {
    # --- PBS settings ---
    'MEM': ['64GB'],
    'WRITE_REPORT_MAX_MEM_GB': [64],
    'NCPUS': [16],
    'TIME': ['12:00:00'],
    'QUEUE': ['normalsr'],

    # --- Working settings ---
    'OBJECTIVE': ['maxprofit'],
    'RESFACTOR': [5],
    'SIM_YEARS': [range(2020, 2051, 5)],
    'WRITE_THREADS': [4],

    # --- Model settings (see "Standard base block" below) ---
    ...
}

duplicate_runs = {
    'BIODIVERSITY_TARGET_GBF_2': ('off', 'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT'),
}

if __name__ == '__main__':
    default_settings_df = get_settings_df(TASK_ROOT_DIR)
    grid_search_param_df = get_grid_search_param_df(grid_search)

    rm_idx = []
    for idx, row in grid_search_param_df.iterrows():
        for k, v in duplicate_runs.items():
            if (row[k] == v[0]) and (str(row[v[1]]) != str(grid_search[v[1]][0])):
                rm_idx.append(row['run_idx'])

    grid_search_param_df = grid_search_param_df[~grid_search_param_df['run_idx'].isin(rm_idx)]
    grid_search_param_df.to_csv(f'{TASK_ROOT_DIR}/grid_search_parameters.csv', index=False)
    print(f'Removed {len(set(rm_idx))} unnecessary runs!')

    get_grid_search_settings_df(TASK_ROOT_DIR, default_settings_df, grid_search_param_df)
```

### Standard base block (copy into every script, then change only the axes)

This is the current LUF core scenario settings as of 2026-04 (Sixth iteration). Copy this verbatim into every `create_tasks_*.py` and modify only the lines that differ for that group.

```python
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
    'HIR_CEILING_PERCENTAGE': [0.7, 0.8, 0.9],

    # --------------- Target deviation weight ---------------
    'SOLVER_WEIGHT_DEMAND': [1],
    'SOLVER_WEIGHT_GHG': [1],
    'SOLVER_WEIGHT_WATER': [1],

    # --------------- Social license ---------------
    'EXCLUDE_NO_GO_LU': [False],
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],
    'REGIONAL_ADOPTION_NON_AG_UNIFORM': [5],
    'REGIONAL_ADOPTION_ZONE': ['NRM_CODE'],

    # --------------- GHG settings ---------------
    'GHG_EMISSIONS_LIMITS': ['low'],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'USE_GHG_SCOPE_1': [True],

    # --------------- Water constraints ---------------
    'WATER_REGION_DEF': ['Drainage Division'],
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

    # --------------- Biodiversity contribution parameters ---------------
    'BIO_CONTRIBUTION_LDS': [0.75],
    'BIO_CONTRIBUTION_ENV_PLANTING': [0.7],
    'BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK': [0.12],
    'BIO_CONTRIBUTION_CARBON_PLANTING_BELT': [0.12],
    'BIO_CONTRIBUTION_RIPARIAN_PLANTING': [1.0],
    'BIO_CONTRIBUTION_AGROFORESTRY': [0.7],
    'BIO_CONTRIBUTION_BECCS': [0],
    'BIO_CONTRIBUTION_DESTOCKING': [0.75],

    # --------------- Biodiversity settings - GBF 2 ---------------
    'BIODIVERSITY_TARGET_GBF_2': ['high'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [15, 20],
    'GBF2_CONSTRAINT_TYPE': ['hard'],

    # --------------- Biodiversity settings - GBF 3 ---------------
    'BIODIVERSITY_TARGET_GBF_3_NVIS': ['off'],
    'BIODIVERSITY_TARGET_GBF_3_IBRA': ['off'],

    # --------------- Biodiversity settings - GBF 4 ---------------
    'BIODIVERSITY_TARGET_GBF_4_SNES': ['off'],
    'BIODIVERSITY_TARGET_GBF_4_ECNES': ['off'],

    # --------------- Biodiversity settings - GBF 8 ---------------
    'BIODIVERSITY_TARGET_GBF_8': ['off'],

    # --------------- Renewable energy ---------------
    'RENEWABLES_OPTIONS': [{'Utility Solar PV': False, 'Onshore Wind': False}],
    'RENEWABLE_EXISTING_END_YEAR': [2030],
    'RENEWABLE_TARGET_SCENARIO_TARGETS': ['Gladstone - Core'],
    'RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS': ['step_change'],

    # --------------- Objective function weights ---------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.5],

    # --------------- Ag management ---------------
    'AG_MANAGEMENTS': [{
        'Asparagopsis taxiformis': True,
        'Precision Agriculture': True,
        'Ecological Grazing': False,
        'Savanna Burning': True,
        'AgTech EI': True,
        'Biochar': True,
        'HIR - Beef': True,
        'HIR - Sheep': True,
        'Utility Solar PV': False,
        'Onshore Wind': False,
    }],
    'AG_MANAGEMENTS_REVERSIBLE': [{
        'Asparagopsis taxiformis': True,
        'Precision Agriculture': True,
        'Ecological Grazing': True,
        'Savanna Burning': True,
        'AgTech EI': True,
        'Biochar': True,
        'HIR - Beef': False,
        'HIR - Sheep': False,
        'Utility Solar PV': False,
        'Onshore Wind': False,
    }],

    # --------------- Non-agricultural land uses ---------------
    'NON_AG_LAND_USES': [{
        'Environmental Plantings': True,
        'Riparian Plantings': True,
        'Sheep Agroforestry': True,
        'Beef Agroforestry': True,
        'Carbon Plantings (Block)': True,
        'Sheep Carbon Plantings (Belt)': True,
        'Beef Carbon Plantings (Belt)': True,
        'BECCS': False,
        'Destocked - natural land': True,
    }],
    'NON_AG_LAND_USES_REVERSIBLE': [{
        'Environmental Plantings': False,
        'Riparian Plantings': False,
        'Sheep Agroforestry': False,
        'Beef Agroforestry': False,
        'Carbon Plantings (Block)': False,
        'Sheep Carbon Plantings (Belt)': False,
        'Beef Carbon Plantings (Belt)': False,
        'BECCS': False,
        'Destocked - natural land': True,
    }],

    # --------------- Dietary ---------------
    'DIET_DOM': ['BAU'],
    'DIET_GLOB': ['BAU'],
    'WASTE': [1],
    'FEED_EFFICIENCY': ['BAU'],
    'IMPORT_TREND': ['Trend'],
```

### Common sensitivity patterns

| Sensitivity | What to change from base block |
|---|---|
| **ECNES** | `'BIODIVERSITY_TARGET_GBF_4_ECNES': ['on']` |
| **HIR Ceiling** | `'HIR_CEILING_PERCENTAGE': [0.7, 0.8, 0.9]` — already a core axis; apply consistently across all groups |
| **Regional** | `'REGIONAL_ADOPTION_CONSTRAINTS': ['NON_AG_UNIFORM']`, `'REGIONAL_ADOPTION_NON_AG_UNIFORM': [5, 10]` (15% dropped in Sixth iteration) |
| **GBF2 Area** | `'CONTRIBUTION_PERCENTILE': ['AG_UNIFORM']`, `'AG_UNIFORM_BIO_CONTRIBUTION': [0]`, `BIO_CONTRIBUTION_ENV_PLANTING/RIPARIAN/DESTOCKING: [1.0]`, `BIO_CONTRIBUTION_CARBON_PLANTING_*: [0.0]`, `BIO_CONTRIBUTION_AGROFORESTRY: [1.0, 0.0]`, `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT: [10, 15, 20]` |
| **Lower Productivity** | `AG_MANAGEMENTS` with `'Precision Agriculture': False, 'AgTech EI': False` |
| **Alt Productivity** | `'PRODUCTIVITY_TREND': ['MEDIUM', 'HIGH']` |
| **Higher Ambition** | `'GHG_EMISSIONS_LIMITS': ['high']`, higher `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT` |
| **Renewables ON** | `RENEWABLES_OPTIONS` Solar/Wind `True`, `AG_MANAGEMENTS` Solar/Wind `True`, add `EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS`, `RENEWABLE_GBF2_CUT_SOLAR/WIND`, `EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK`, `RENEWABLE_EPBC_MNES_CUT_SOLAR/WIND` |

### `duplicate_runs` — removing redundant permutations

`duplicate_runs` removes combinations that are logically equivalent. The standard one:

```python
duplicate_runs = {
    # When GBF2 is 'off', all GBF2_CUT values produce the same run — keep only the first
    'BIODIVERSITY_TARGET_GBF_2': ('off', 'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT'),
}
```

### Critical: `AG_MANAGEMENTS` must mirror `RENEWABLES_OPTIONS`

`write_settings` in `helpers.py` writes each setting as a flat literal — it does **not** re-execute Python. So `AG_MANAGEMENTS` in the run copy will NOT automatically derive from `RENEWABLES_OPTIONS`. You must keep them in sync manually:

```python
# Renewables OFF — both dicts have Solar/Wind = False
'RENEWABLES_OPTIONS': [{'Utility Solar PV': False, 'Onshore Wind': False}],
'AG_MANAGEMENTS': [{ ..., 'Utility Solar PV': False, 'Onshore Wind': False }],

# Renewables ON — both dicts have Solar/Wind = True
'RENEWABLES_OPTIONS': [{'Utility Solar PV': True, 'Onshore Wind': True}],
'AG_MANAGEMENTS': [{ ..., 'Utility Solar PV': True, 'Onshore Wind': True }],
```

---

## Step 2: Write `merge_unique_parameters.py`

One merge script per iteration. Template:

```python
SUBMIT = False   # set True when ready to submit

import os, shutil, subprocess
import pandas as pd

LUTO_ROOT   = "/g/data/jk53/jinzhu/LUTO/luto-2.0"
SCRIPTS_DIR = f"{LUTO_ROOT}/jinzhu_inspect_code/LUF_Nth_iteration"   # MUST match actual dir name

CREATE_SCRIPTS = [
    "create_tasks_core.py",
    "create_tasks_sensitivity_ecnes.py",
    # ... one entry per create_tasks_*.py
]

# Keys must match TASK_ROOT_DIR in each create_tasks_*.py
GROUPS = {
    "Core":               "/g/data/jk53/jinzhu/LUTO/Custom_runs/LUF_Nth_iteration/Core",
    "Sensitivity_ECNES":  "/g/data/jk53/jinzhu/LUTO/Custom_runs/LUF_Nth_iteration/Sensitivity_ECNES",
    # ...
}

OUTPUT_DIR = "/g/data/jk53/jinzhu/LUTO/Custom_runs/LUF_Nth_iteration"
```

**Common mistake:** `SCRIPTS_DIR` must use the actual directory name on disk (e.g. `LUF_4th_iteration`), not a long-form name (e.g. `LUF_Fourth_iteration`). This caused `CalledProcessError: exit status 2` in the 4th iteration.

The merge logic (global run indexing, outer-join template, `Run_G0001` naming) is identical across all iterations — copy it verbatim from any existing `merge_unique_parameters.py`.

### Single-group case

When there is only one scenario group (e.g. a standalone run like `Sachi_run`), the merge script handles it without any special logic:

- **`OUTPUT_DIR` equals the single group's `TASK_ROOT_DIR`** — set them to the same path. The group's CSVs are already in the right place; the merged outputs overwrite them in place.
- **`non_str_val.txt` copy** — `os.path.samefile(src, dst)` returns `True`, so the copy is correctly skipped.
- **Outer-join loop** — runs zero times (only one template), which is fine: `merged_template = template_dfs[0]`.
- **`vary_cols` filter** — if all parameters are fixed, most columns won't satisfy `nunique() > 1`. This is expected; the merged unique CSV will only contain the index columns. The full settings are still in `merged_grid_search_template.csv`.

Single-group `GROUPS` example:

```python
GROUPS = {
    "MyRun": "/g/data/jk53/jinzhu/LUTO/Custom_runs/MyRun",
}
OUTPUT_DIR = "/g/data/jk53/jinzhu/LUTO/Custom_runs/MyRun"   # same as the single group
```

---

## Step 3: Verify settings alignment with `settings.py`

Before generating CSVs, check that all scripts are current with `luto/settings.py`. The most common stale items are **new renewable energy settings** added to `settings.py` after the scripts were written.

Quick audit command:
```bash
for f in jinzhu_inspect_code/LUF_Nth_iteration/create_tasks_*.py; do
  echo "=== $(basename $f) ==="
  grep -E "RENEWABLE_EXISTING_END_YEAR|RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS|RENEWABLE_TARGET_SCENARIO_TARGETS" "$f"
done
```

Expected output for each script:
```
'RENEWABLE_EXISTING_END_YEAR': [2030],
'RENEWABLE_TARGET_SCENARIO_TARGETS': ['Gladstone - Core'],
'RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS': ['step_change'],
```

If any are missing, add them to the renewable energy section of the affected scripts.

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

## Step 4: Generate CSVs and inspect

```bash
cd /g/data/jk53/jinzhu/LUTO/luto-2.0
conda run -n luto python jinzhu_inspect_code/LUF_Nth_iteration/merge_unique_parameters.py
```

This runs all `create_tasks_*.py` scripts and writes to each group's `TASK_ROOT_DIR`:
- `grid_search_parameters.csv` — all permutations with `run_idx`
- `grid_search_template.csv` — full settings for every run (one column per run)
- `non_str_val.txt` — setting names that need `eval()` (non-string types)

And writes to `OUTPUT_DIR`:
- `merged_grid_search_parameters_unique.csv` — varying columns only, with `scenario_group` + `global_run_idx`
- `merged_grid_search_template.csv` — all runs with globally unique `Run_G0001`, `Run_G0002`, ...

Inspect before submitting:
```python
import pandas as pd
df = pd.read_csv("/g/data/jk53/jinzhu/LUTO/Custom_runs/LUF_Nth_iteration/merged_grid_search_parameters_unique.csv")
print(df[['scenario_group', 'global_run_idx', 'HIR_CEILING_PERCENTAGE', 'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT']])
```

---

## Step 5: Submit to cluster

Set `SUBMIT = True` in `merge_unique_parameters.py` and re-run:

```bash
conda run -n luto python jinzhu_inspect_code/LUF_Nth_iteration/merge_unique_parameters.py
```

This calls `create_task_runs(OUTPUT_DIR, merged_template, mode='cluster', max_concurrent_tasks=200)` which:
1. Creates `Run_G0001/`, `Run_G0002/`, ... folders under `OUTPUT_DIR`
2. Copies `luto/` source + creates `input/` symlink in each
3. Writes the run-specific `luto/settings.py` from the template column
4. Submits PBS jobs via `qsub`

Monitor:
```bash
qstat -u jw6041
```

---

## Checklist for a new iteration

- [ ] Copy the latest `create_tasks_core.py` from the previous iteration as a starting point
- [ ] Update `TASK_ROOT_DIR` path to the new iteration name
- [ ] Update the varying axes in `grid_search` (e.g. `HIR_CEILING_PERCENTAGE`, `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT`)
- [ ] Copy and modify for each sensitivity group
- [ ] Write `merge_unique_parameters.py` with correct `SCRIPTS_DIR` (match actual dir name on disk)
- [ ] Verify all renewable energy settings present: `RENEWABLE_EXISTING_END_YEAR`, `RENEWABLE_TARGET_SCENARIO_TARGETS`, `RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS`
- [ ] Run merge with `SUBMIT=False`, inspect CSVs
- [ ] Set `SUBMIT=True` and re-run
