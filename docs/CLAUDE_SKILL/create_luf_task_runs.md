# Skill: Creating Task Runs for Multiple LUF Scenarios

This skill documents the standard workflow for creating and launching a batch of LUTO2 scenario runs locally on Windows. The canonical pattern is a **single self-contained `create_tasks.py` script** that generates CSVs, creates `Run_G*` directories, and copies `run_all.py` — all in one shot.

---

## Overview

```
jinzhu_inspect_code/<Iteration>/
  create_tasks.py          ← single script: BASE_GRID + RUN_OVERRIDES + __main__

<Custom_runs>/<Iteration>/
  run_all.py               ← copied by create_tasks.py
  grid_search_parameters.csv
  grid_search_parameters_unique.csv
  grid_search_template.csv
  non_str_val.txt
  Run_G0001/
    python_script.py       ← simulation entry point (copied by create_task_runs)
    luto/                  ← frozen snapshot of luto source
    luto/settings.py       ← run-specific settings
    run.log                ← created at launch by run_all.py
  Run_G0002/ ...
```

---

## Step 1: Write `create_tasks.py`

One script per batch. It has four parts:

1. **Top-level constants** — `LUTO_DIR`, `TASK_DIR` hardcoded as `Path` objects
2. **`BASE_GRID`** — settings common to all runs (cluster + working + scenario)
3. **`RUN_OVERRIDES`** — list of `(run_label, scenario_group, overrides_dict)` tuples, one per run
4. **`build()` + `__main__`** — generates CSVs, creates `Run_G*` dirs, copies `run_all.py`

### Full template

```python
"""<Batch description> — N-run grid.

Runs vary <axes>. BIO_QUALITY_LAYER = '<value>'.

Scenario groups:
  <GROUP_A>  — <description>  (Runs 01-XX)
  <GROUP_B>  — <description>  (Runs XX-YY)
"""
import shutil
from pathlib import Path
import pandas as pd

from luto.tools.create_task_runs.helpers import (
    get_settings_df,
    get_grid_search_param_df,
    update_settings,
    create_task_runs,
)

LUTO_DIR = Path(r'F:\Users\jinzhu\Documents\luto-2.0')
TASK_DIR = Path(r'F:\Users\jinzhu\Documents\Custom_runs\<Iteration>')


# ----------------------------------------------------------------------
# BASE GRID — settings common to all runs.
# ----------------------------------------------------------------------
BASE_GRID = {
    # --------------- Cluster job settings ---------------
    'MEM':                     ['128GB'],
    'WRITE_REPORT_MAX_MEM_GB': [120],
    'NCPUS':                   [32],
    'TIME':                    ['12:00:00'],
    'QUEUE':                   ['normalsr'],

    # --------------- Working settings ---------------
    'OBJECTIVE':      ['maxprofit'],
    'RESFACTOR':      [5],
    'SIM_YEARS':      [[2010, 2020, 2025, 2030, 2035, 2040, 2045, 2050]],
    'WRITE_PARALLEL': [True],
    'WRITE_THREADS':  [12],
    'DO_IIS':         [False],
    'WRITE_OUTPUTS':  [True],

    # --------------- Scenarios ---------------
    'SSP':                          ['245'],
    'CARBON_EFFECTS_WINDOW':        [60],
    'RISK_OF_REVERSAL':             [0],
    'FIRE_RISK':                    ['med'],
    'CONVERGENCE':                  [2050],
    'CO2_FERT':                     ['off'],
    'APPLY_DEMAND_MULTIPLIERS':     [True],
    'PRODUCTIVITY_TREND':           ['BAU'],

    # --------------- Economics ---------------
    'DYNAMIC_PRICE':                                 [True],
    'AMORTISE_UPFRONT_COSTS':                        [False],
    'DISCOUNT_RATE':                                 [0.07],
    'AMORTISATION_PERIOD':                           [30],
    'CARBON_PRICE_COSTANT':                          [0],
    'BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR':     [100],
    'SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR':    [100],
    'HIR_CEILING_PERCENTAGE':                        [0.8],

    # --------------- Target deviation weights ---------------
    'SOLVER_WEIGHT_DEMAND':         [1],
    'SOLVER_WEIGHT_GHG':            [1],
    'SOLVER_WEIGHT_WATER':          [1],

    # --------------- Social licence ---------------
    'EXCLUDE_NO_GO_LU':                  [False],
    'REGIONAL_ADOPTION_CONSTRAINTS':     ['off'],
    'REGIONAL_ADOPTION_NON_AG_CAP':      [15],
    'REGIONAL_ADOPTION_NON_AG_REGION':   ['NRM'],
    'REGIONAL_ADOPTION_ZONE':            ['NRM_CODE'],

    # --------------- GHG settings ---------------
    'CARBON_PRICES_FIELD':          ['CONSTANT'],
    'GHG_CONSTRAINT_TYPE':          ['hard'],
    'USE_GHG_SCOPE_1':              [True],

    # --------------- Water constraints ---------------
    'WATER_REGION_DEF':             ['River Region'],
    'WATER_LIMITS':                 ['on'],
    'WATER_CONSTRAINT_TYPE':        ['hard'],
    'WATER_STRESS':                 [0.6],
    'WATER_CLIMATE_CHANGE_IMPACT':  ['on'],
    'LIVESTOCK_DRINKING_WATER':     [1],
    'INCLUDE_WATER_LICENSE_COSTS':  [1],

    # --------------- Biodiversity overall ---------------
    'BIO_QUALITY_LAYER':            ['MNES_likely'],
    'CONTRIBUTION_PERCENTILE':      ['USER_DEFINED'],
    'CONNECTIVITY_SOURCE':          ['NCI'],
    'CONNECTIVITY_LB':              [0.7],

    # --------------- Biodiversity contribution parameters ---------------
    'BIO_CONTRIBUTION_LDS':                   [0.75],
    'BIO_CONTRIBUTION_ENV_PLANTING':          [0.7],
    'BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK': [0.12],
    'BIO_CONTRIBUTION_CARBON_PLANTING_BELT':  [0.12],
    'BIO_CONTRIBUTION_RIPARIAN_PLANTING':     [1.0],
    'BIO_CONTRIBUTION_AGROFORESTRY':          [0.7],
    'BIO_CONTRIBUTION_BECCS':                 [0],
    'BIO_CONTRIBUTION_DESTOCKING':            [0.75],

    # --------------- GBF 2 ---------------
    'BIODIVERSITY_TARGET_GBF_2':    ['high'],
    'GBF2_CONSTRAINT_TYPE':         ['hard'],

    # --------------- GBF 3 ---------------
    'BIODIVERSITY_TARGET_GBF_3_NVIS':    ['off'],
    'BIODIVERSITY_TARGET_GBF_3_IBRA':    ['off'],

    # --------------- GBF 4 ---------------
    'BIODIVERSITY_TARGET_GBF_4_SNES':    ['on'],
    'GBF4_SNES_REGION_MODE':             ['Australia'],
    'BIODIVERSITY_TARGET_GBF_4_ECNES':   ['on'],
    'GBF4_ECNES_REGION_MODE':            ['Australia'],

    # --------------- GBF 8 ---------------
    'BIODIVERSITY_TARGET_GBF_8':         ['off'],

    # --------------- Renewable energy ---------------
    'RENEWABLES_OPTIONS': [{'Utility Solar PV': True, 'Onshore Wind': True}],
    'RENEWABLE_TARGET_SCENARIO_TARGETS':      ['Gladstone - Core'],
    'RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS': ['step_change'],

    # --------------- Objective weights ---------------
    'SOLVE_WEIGHT_ALPHA':   [1],
    'SOLVE_WEIGHT_BETA':    [0.5],

    # --------------- Ag management ---------------
    'AG_MANAGEMENTS': [{
        'Asparagopsis taxiformis': True,
        'Precision Agriculture':   True,
        'Ecological Grazing':      False,
        'Savanna Burning':         True,
        'AgTech EI':               True,
        'Biochar':                 True,
        'HIR - Beef':              True,
        'HIR - Sheep':             True,
        'Utility Solar PV':        True,
        'Onshore Wind':            True,
    }],
    'AG_MANAGEMENTS_REVERSIBLE': [{
        'Asparagopsis taxiformis': True,
        'Precision Agriculture':   True,
        'Ecological Grazing':      True,
        'Savanna Burning':         True,
        'AgTech EI':               True,
        'Biochar':                 True,
        'HIR - Beef':              False,
        'HIR - Sheep':             False,
        'Utility Solar PV':        False,
        'Onshore Wind':            False,
    }],

    # --------------- Non-agricultural land uses ---------------
    'NON_AG_LAND_USES': [{
        'Environmental Plantings':       True,
        'Riparian Plantings':            True,
        'Sheep Agroforestry':            True,
        'Beef Agroforestry':             True,
        'Carbon Plantings (Block)':      True,
        'Sheep Carbon Plantings (Belt)': True,
        'Beef Carbon Plantings (Belt)':  True,
        'BECCS':                         False,
        'Destocked - natural land':      True,
    }],
    'NON_AG_LAND_USES_REVERSIBLE': [{
        'Environmental Plantings':       False,
        'Riparian Plantings':            False,
        'Sheep Agroforestry':            False,
        'Beef Agroforestry':             False,
        'Carbon Plantings (Block)':      False,
        'Sheep Carbon Plantings (Belt)': False,
        'Beef Carbon Plantings (Belt)':  False,
        'BECCS':                         False,
        'Destocked - natural land':      True,
    }],

    # --------------- Dietary ---------------
    'DIET_DOM':         ['BAU'],
    'DIET_GLOB':        ['BAU'],
    'WASTE':            [1],
    'FEED_EFFICIENCY':  ['BAU'],
    'IMPORT_TREND':     ['Trend'],
}


# ----------------------------------------------------------------------
# RUN_OVERRIDES — one tuple per run.
# Format: (run_label, scenario_group, {param: [single_value]})
# Every key in an override replaces the corresponding BASE_GRID value.
# ----------------------------------------------------------------------
RUN_OVERRIDES = [
    # ---- GROUP_A ----
    ('Run_G0001', 'GROUP_A', {
        'GHG_EMISSIONS_LIMITS':                         ['low'],
        'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT':  [15],
        # ... other varying params ...
    }),
    ('Run_G0002', 'GROUP_A', {
        'GHG_EMISSIONS_LIMITS':                         ['low'],
        'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT':  [15],
        # ...
    }),

    # ---- GROUP_B ----
    ('Run_G0003', 'GROUP_B', {
        'GHG_EMISSIONS_LIMITS':                         ['high'],
        'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT':  [20],
        # ...
    }),
]


def build(task_root_dir):
    """Generate grid_search CSVs for all runs under task_root_dir.
    Returns the template DataFrame for passing directly to create_task_runs.
    """
    task_root_dir = Path(task_root_dir)
    task_root_dir.mkdir(parents=True, exist_ok=True)
    default_settings_df = get_settings_df(str(task_root_dir))
    defaults = default_settings_df.set_index('Name')['Default_run'].to_dict()

    all_param_rows = []
    all_run_series = []

    for run_idx, (run_label, scenario_group, run_overrides) in enumerate(RUN_OVERRIDES, start=1):
        grid = {**BASE_GRID, **run_overrides}
        df = get_grid_search_param_df(grid)
        if len(df) != 1:
            raise RuntimeError(
                f"Run '{run_label}' produced {len(df)} rows (expected 1). "
                "All override lists must have exactly one element."
            )
        df.insert(0, 'run_label', run_label)
        df.insert(1, 'scenario_group', scenario_group)
        df['run_idx'] = run_idx
        all_param_rows.append(df)

        settings_dict = dict(defaults)
        settings_dict.update(df.drop(columns=['run_label', 'scenario_group', 'run_idx']).iloc[0].to_dict())
        settings_dict = update_settings(settings_dict, f'run_G{run_idx:04}')
        all_run_series.append(pd.Series(settings_dict, name=f'Run_G{run_idx:04}'))

    template = pd.concat(all_run_series, axis=1).reset_index(names='Name')
    template.to_csv(task_root_dir / 'grid_search_template.csv', index=False)

    params = pd.concat(all_param_rows, ignore_index=True)
    params.to_csv(task_root_dir / 'grid_search_parameters.csv', index=False)

    vary_cols = ['run_label', 'scenario_group'] + [
        c for c in params.columns
        if c not in ('run_label', 'scenario_group', 'run_idx')
        and params[c].nunique(dropna=False) > 1
    ]
    params[vary_cols + ['run_idx']].to_csv(
        task_root_dir / 'grid_search_parameters_unique.csv', index=False
    )

    print(f'Runs: {len(params)}  (expect {len(RUN_OVERRIDES)})')
    return template


if __name__ == '__main__':
    template = build(TASK_DIR)
    create_task_runs(str(TASK_DIR), template, mode='single', n_workers=4)
    shutil.copy(
        LUTO_DIR / 'luto/tools/create_task_runs/bash_scripts/run_all.py',
        TASK_DIR / 'run_all.py',
    )
    print(f'Copied run_all.py → {TASK_DIR}')
```

---

## Step 2: What `create_tasks.py` produces

Running `python create_tasks.py` does three things in sequence:

1. **`build()`** — writes three CSVs to `TASK_DIR`:
   - `grid_search_parameters.csv` — all runs with `run_label`, `scenario_group`, `run_idx`
   - `grid_search_parameters_unique.csv` — only the columns that vary across runs
   - `grid_search_template.csv` — full settings, one column per run (NaN-free)

2. **`create_task_runs(..., mode='single')`** — for each `Run_G*` column in the template:
   - Creates `TASK_DIR/Run_G0001/`, `Run_G0002/`, ... directories
   - Copies a frozen snapshot of the entire `luto/` source into each
   - Writes the run-specific `luto/settings.py` from that column
   - Copies `python_script.py` into each run dir

3. **`shutil.copy`** — puts `run_all.py` at `TASK_DIR/` root

---

## Step 3: Launch runs with `run_all.py`

```powershell
cd F:\Users\jinzhu\Documents\Custom_runs\<Iteration>
conda activate luto
python run_all.py           # 2 runs at a time (default)
python run_all.py --max 4   # 4 concurrent
python run_all.py Run_G0001 Run_G0003   # specific runs only
```

`run_all.py` globs `Run_G*`, launches `python_script.py` in each up to `--max` concurrently, and writes stdout/stderr to `Run_G*/run.log`. See [submit_task_runs_windows.md](submit_task_runs_windows.md) for monitoring and failure recovery.

---

## Key design notes

### Run naming: always `Run_G{idx:04}`
- `update_settings` job name: `f'run_G{run_idx:04}'`
- `pd.Series` column name: `f'Run_G{run_idx:04}'`
- `run_all.py` globs `Run_G*` — these must match.
- The `run_label` field in `RUN_OVERRIDES` (e.g. `'Run_RE0001'`) is only a human-readable tag stored in the CSV; it does **not** affect the on-disk directory name. The directory name comes from the `pd.Series` name (`Run_G{idx:04}`), so **always keep the `build()` internals as `run_G` regardless of the batch's labelling convention**.

### `RUN_OVERRIDES` vs Cartesian grid
`RUN_OVERRIDES` defines runs **explicitly** — each tuple is one exact run. This is the right choice when:
- Runs aren't a simple Cartesian product (e.g. GHG=low only gets GBF2=15%, GHG=high only gets GBF2=20%)
- You want each run labelled and documented with a scenario narrative

For a true Cartesian product (every combination of N axes), use `get_grid_search_param_df` with multi-value lists instead and remove `RUN_OVERRIDES`.

### `AG_MANAGEMENTS` must mirror `RENEWABLES_OPTIONS`
`write_settings` writes flat literals and does NOT re-evaluate. When renewables are ON in `RENEWABLES_OPTIONS`, `AG_MANAGEMENTS` must also have `'Utility Solar PV': True, 'Onshore Wind': True`. Keep them in sync manually in both `BASE_GRID` and any override that toggles renewables.

### Unspecified settings: fall back to source `luto/settings.py`

**If a setting is not explicitly listed in `BASE_GRID` or `RUN_OVERRIDES`, do NOT invent a value — leave it out of the grid entirely.** The `build()` function seeds every run's `settings_dict` from `get_settings_df()`, which reads the live `luto/settings.py` defaults. Omitted keys therefore inherit the source LUTO value automatically.

Only add a key to `BASE_GRID` when you need to **override** the source default for this batch. The table below lists the known intentional overrides used in standard task runs:

### Settings intentionally different from `settings.py` defaults

| Setting | `settings.py` default | Standard task value |
|---|---|---|
| `IMPORT_TREND` | `'Static'` | `'Trend'` |
| `DYNAMIC_PRICE` | `False` | `True` |
| `CARBON_EFFECTS_WINDOW` | `50` | `60` |
| `GHG_EMISSIONS_LIMITS` | `'high'` | `'low'` (core) |
| `BIO_CONTRIBUTION_ENV_PLANTING` | `0.8` | `0.7` |
| `BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK/BELT` | `0.1` | `0.12` |
| `BIO_CONTRIBUTION_RIPARIAN_PLANTING` | `1.2` | `1.0` |
| `BIO_CONTRIBUTION_AGROFORESTRY` | `0.75` | `0.7` |

Do NOT "fix" these to match defaults — they are scenario spec overrides.

### Common `RUN_OVERRIDES` patterns

| Axis | Override keys |
|---|---|
| GHG higher ambition | `'GHG_EMISSIONS_LIMITS': ['high']` |
| GBF2 cut level | `'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [20]` |
| RE exclusion off | `'EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS': [False], 'RENEWABLE_GBF2_CUT_WIND': [0], 'RENEWABLE_GBF2_CUT_SOLAR': [0]` |
| RE GBF2 cut | `'EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS': [True], 'RENEWABLE_GBF2_CUT_WIND': [20], 'RENEWABLE_GBF2_CUT_SOLAR': [20]` |
| RE EPBC MNES cut | `'EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK': [True], 'RENEWABLE_EPBC_MNES_CUT_SOLAR': [10], 'RENEWABLE_EPBC_MNES_CUT_WIND': [10]` |
| Social licence | `'REGIONAL_ADOPTION_CONSTRAINTS': ['NON_AG_CAP'], 'REGIONAL_ADOPTION_NON_AG_CAP': [5]` |
| GBF4 NRM-restricted | `'GBF4_SNES_REGION_MODE': ['NRM'], 'GBF4_SNES_SELECTED_REGIONS': [['North East', 'Goulburn Broken']]` |

---

## Checklist for a new batch

- [ ] Copy `create_tasks.py` from a recent iteration; update docstring, `TASK_DIR`, `BASE_GRID` baseline — **only include settings that differ from source `luto/settings.py`; omit everything else so it inherits the live default**
- [ ] Set `BIO_QUALITY_LAYER` in `BASE_GRID` to the correct value (`'Suitability'` or `'MNES_likely'`)
- [ ] Verify `RENEWABLES_OPTIONS` and `AG_MANAGEMENTS` Solar/Wind flags are in sync
- [ ] Define `RUN_OVERRIDES` — one tuple per run, all override lists are single-element
- [ ] Run `python create_tasks.py` — check console output for expected run count
- [ ] Inspect `grid_search_parameters_unique.csv` — verify only the expected axes vary
- [ ] Assert NaN-free template: `pd.read_csv('grid_search_template.csv').isnull().sum().sum() == 0`
- [ ] Launch: `cd <TASK_DIR> && conda activate luto && python run_all.py --max 2`
