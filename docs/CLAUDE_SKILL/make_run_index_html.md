# Skill: Generate Run Index HTML for a Task Run Directory

Creates a self-contained interactive `index.html` for any grid-search / sensitivity task run directory.

**Sources used:**
- `merged_grid_search_parameters_unique.csv` — GEP (grid-search experiment parameters): the subset of settings that vary across runs
- `merged_grid_search_template.csv` — full settings snapshot for every run (154 rows × N runs); used to show all fixed settings in the detail panel
- PBS stdout logs (`Run_G####/run_*.o*`) — scanned to determine model feasibility status per run

**Output:** `<task_root_dir>/index.html` — two-panel interactive page:
- **Left panel**: filterable table (scenario group, productivity, ECNES, regional constraints, status), one row per run, red diagonal stripe on infeasible rows
- **Right panel**: click a row → full settings detail with fixed/varying highlighted, AG_MANAGEMENTS mini-table, infeasibility status

---

## When to Use

- After a grid search or sensitivity run batch completes (or is in progress — partial logs work fine)
- To get a quick overview of which runs are feasible, what parameters each uses, and to share with collaborators as a single HTML file

---

## Input File Formats

### `merged_grid_search_parameters_unique.csv`

Row-per-run CSV. Required columns:

| Column | Description |
|--------|-------------|
| `scenario_group` | Group label (e.g. `Core`, `Sensitivity_ECNES`, `Sensitivity_Regional`, …) |
| `global_run_idx` | Integer; maps to `Run_G####` folder name (zero-padded to 4 digits) |
| `local_run_idx` | Integer index within the scenario group |
| `PRODUCTIVITY_TREND` | `BAU` / `MEDIUM` / `HIGH` |
| `REGIONAL_ADOPTION_CONSTRAINTS` | `off` / `NON_AG_CAP` |
| `REGIONAL_ADOPTION_NON_AG_CAP` | Threshold % (integer or float) |
| `CONTRIBUTION_PERCENTILE` | `USER_DEFINED` / `AG_UNIFORM` |
| `BIO_CONTRIBUTION_LDS` | Float (e.g. `0.75`, `0.8`, `0.85`) |
| `BIO_CONTRIBUTION_ENV_PLANTING` | Float |
| `BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK` | Float |
| `BIO_CONTRIBUTION_CARBON_PLANTING_BELT` | Float |
| `BIO_CONTRIBUTION_AGROFORESTRY` | Float |
| `BIO_CONTRIBUTION_DESTOCKING` | Float |
| `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT` | Integer (e.g. `10`, `15`, `20`) |
| `BIODIVERSITY_TARGET_GBF_4_ECNES` | `on` / `off` |
| `AG_MANAGEMENTS` | Python dict literal string |

All other columns in this file are ignored by the script.

### `merged_grid_search_template.csv`

Wide-format: rows = setting names, columns = `Name` + one column per `Run_G####`.
Used only for the detail panel (right side) — shows **all** 154 settings for the selected run,
with rows that vary across runs highlighted in yellow.

### PBS log files (cluster)

Located at `<task_root_dir>/Run_G####/run_####.o<jobid>`.
The script parses three log patterns to classify solver outcomes per year:
- `Trying NumericFocus=N for year YYYY` — marks which year+NF level is being attempted
- `Infeasible model` — that attempt failed
- `Feasible solution found with NumericFocus=N` — that attempt succeeded

A year is **infeasible** if no attempt ever succeeded. A year is **retried** if any attempt
used `NumericFocus > 0` — this applies to **both** recovered years (which also succeeded) and
still-infeasible years (which failed even after retry). The `retried_years` dict records
`{year: max_NumericFocus_used}` in both cases, so the Retries column shows retry effort even
for ultimately infeasible runs.

### `run.log` files (Windows / local runs)

Located at `<task_root_dir>/Run_G####/run.log` — produced by `run_all.py` on Windows.
Parse these patterns (in priority order):

| Pattern | Outcome |
|---------|--------|
| `ValueError:` or `Traceback (most recent call last)` | **ERROR** — data loading crash before solver |
| `Solver status for year YYYY: INFEASIBLE` | **INFEASIBLE** — collect all matching years |
| `Optimal solution found` or `OPTIMAL` | **OPTIMAL** |
| none of the above | **UNKNOWN** (run incomplete / still running) |

ERROR runs fail during `Data()` construction — common cause for ECNES/SNES species runs:
`ValueError: No valid GBF4 SNES NRM targets found for regions ['North East']` means that
species has no entry in `BIODIVERSITY_GBF4_TARGET_SNES_NRM.csv` for that NRM region.
These must be re-run after fixing the target CSV, not debugged as infeasibility.

Sample `scan_feasibility` for Windows `run.log`:
```python
def scan_feasibility(task_root):
    """
    Returns dict: run_id -> {'status': 'OPTIMAL'|'INFEASIBLE'|'ERROR'|'UNKNOWN',
                              'infeasible_years': list[int],
                              'error_msg': str or None}
    """
    import os, re, glob
    results = {}
    for rd in sorted(glob.glob(os.path.join(task_root, "Run_G*"))):
        run_id = os.path.basename(rd)
        log_path = os.path.join(rd, "run.log")
        if not os.path.exists(log_path):
            results[run_id] = {'status': 'NO_LOG', 'infeasible_years': [], 'error_msg': None}
            continue
        content = open(log_path, encoding='utf-8', errors='replace').read()
        # 1. Crash / data loading error
        if 'ValueError' in content or 'Traceback (most recent call last)' in content:
            m = re.search(r'ValueError: (.+)', content)
            results[run_id] = {'status': 'ERROR', 'infeasible_years': [],
                               'error_msg': m.group(1).strip() if m else 'Unknown error'}
            continue
        # 2. Infeasible solver
        inf_years = sorted(set(int(m.group(1)) for m in re.finditer(
            r'Solver status for year (\d{4}):\s*INFEASIBLE', content)))
        if inf_years:
            results[run_id] = {'status': 'INFEASIBLE', 'infeasible_years': inf_years, 'error_msg': None}
        elif 'Optimal solution found' in content or 'OPTIMAL' in content:
            results[run_id] = {'status': 'OPTIMAL', 'infeasible_years': [], 'error_msg': None}
        else:
            results[run_id] = {'status': 'UNKNOWN', 'infeasible_years': [], 'error_msg': None}
    return results
```

---

## Script

Save to `jinzhu_inspect_code/make_run_index.py` (or run inline). Takes one argument: `task_root_dir`.

```python
#!/usr/bin/env python3
"""
Generate index.html for a LUTO2 grid-search task run directory.

Usage:
    python make_run_index.py /path/to/Custom_runs/MyRun
    # or call make_index(task_root_dir) from Python
"""
import os, re, glob, csv, ast, json, sys


def scan_infeasibility(task_root_dir):
    """
    Returns dict: run_id -> (infeasible: bool, infeasible_years: list[int],
                              retried_years: dict[int -> int])
    - infeasible: True only if at least one year has NO successful retry
    - infeasible_years: years with no successful retry
    - retried_years: {year: max_numeric_focus_used} for years that were retried
      (NumericFocus > 0) — includes both recovered years AND still-infeasible years
      that were retried

    Parses log patterns:
      "Trying NumericFocus=N for year YYYY"  -- start of attempt
      "Infeasible model"                      -- attempt failed
      "Feasible solution found with NumericFocus=N" -- attempt succeeded
    """
    results = {}
    for rd in sorted(glob.glob(os.path.join(task_root_dir, "Run_G*"))):
        run_id = os.path.basename(rd)
        infeasible_years = []
        retried_years = {}

        for f in sorted(glob.glob(os.path.join(rd, "*.o*"))):
            try:
                content = open(f).read()
                if 'Infeasible model' not in content and 'INFEASIBLE' not in content:
                    continue

                lines = content.split('\n')
                current_year = None
                current_nf = 0
                year_attempts = {}

                for line in lines:
                    m = re.search(r'Trying NumericFocus=(\d+) for year (\d{4})', line)
                    if m:
                        current_nf = int(m.group(1))
                        current_year = int(m.group(2))
                        continue

                    m2 = re.search(r'Running for year (\d{4})', line)
                    if m2:
                        current_year = int(m2.group(1))
                        current_nf = 0
                        continue

                    if ('Infeasible model' in line or 'INFEASIBLE' in line) and current_year is not None:
                        year_attempts.setdefault(current_year, []).append((current_nf, 'infeasible'))
                        continue

                    mf = re.search(r'Feasible solution found with NumericFocus=(\d+)', line)
                    if mf and current_year is not None:
                        nf_used = int(mf.group(1))
                        year_attempts.setdefault(current_year, []).append((nf_used, 'feasible'))
                        continue

                for year, attempts in year_attempts.items():
                    outcomes = [o for _, o in attempts]
                    if 'feasible' in outcomes:
                        if 'infeasible' in outcomes:
                            max_nf = max(nf for nf, _ in attempts)
                            if year not in retried_years or retried_years[year] < max_nf:
                                retried_years[year] = max_nf
                    else:
                        if year not in infeasible_years:
                            infeasible_years.append(year)
                        # Also record retry attempts even though they failed
                        max_nf = max(nf for nf, _ in attempts)
                        if max_nf > 0:
                            if year not in retried_years or retried_years[year] < max_nf:
                                retried_years[year] = max_nf

            except Exception:
                pass

        infeasible_years = sorted(infeasible_years)
        infeasible = len(infeasible_years) > 0
        results[run_id] = (infeasible, infeasible_years, retried_years)
    return results


def load_unique_params(task_root_dir):
    """Returns list of dicts from merged_grid_search_parameters_unique.csv."""
    path = os.path.join(task_root_dir, "merged_grid_search_parameters_unique.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def load_template(task_root_dir):
    """
    Returns (rows, varying_names, run_ids) from merged_grid_search_template.csv.
    rows: list of dicts {Name, Run_G0001, Run_G0002, ...}
    varying_names: set of Name values that differ across runs
    run_ids: ordered list of Run_G#### column names
    """
    path = os.path.join(task_root_dir, "merged_grid_search_template.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    run_ids = [k for k in rows[0].keys() if k != 'Name']
    varying_names = set()
    for r in rows:
        vals = set(r[c] for c in run_ids)
        if len(vals) > 1:
            varying_names.add(r['Name'])
    return rows, varying_names, run_ids


def build_runs_js(unique_rows, infeasibility, template_rows, template_varying, template_run_ids):
    """Build the JS RUNS array string."""
    # Build template lookup: name -> {run_id -> value}
    tmpl = {r['Name']: {c: r[c] for c in template_run_ids} for r in template_rows}

    parts = []
    for r in unique_rows:
        gidx = int(r['global_run_idx'])
        run_id = f"Run_G{gidx:04d}"
        inf, inf_years, retried_years = infeasibility.get(run_id, (False, [], {}))

        # AG_MANAGEMENTS as proper JS object
        try:
            ag_dict = ast.literal_eval(r['AG_MANAGEMENTS'])
        except Exception:
            ag_dict = {}
        ag_js = json.dumps(ag_dict)

        # Full settings for this run from template
        full_settings = {name: tmpl[name][run_id] for name in tmpl if run_id in tmpl[name]}
        full_settings_js = json.dumps(full_settings)

        # retried_years keys must be strings for JSON
        retried_years_js = json.dumps({str(k): v for k, v in retried_years.items()})

        js = (
            f'  {{ run_id:"{run_id}", scenario_group:"{r["scenario_group"]}", '
            f'global_run_idx:{gidx}, local_run_idx:{r["local_run_idx"]}, '
            f'PRODUCTIVITY_TREND:"{r["PRODUCTIVITY_TREND"]}", '
            f'REGIONAL_ADOPTION_CONSTRAINTS:"{r["REGIONAL_ADOPTION_CONSTRAINTS"]}", '
            f'REGIONAL_ADOPTION_NON_AG_CAP:{r["REGIONAL_ADOPTION_NON_AG_CAP"]}, '
            f'CONTRIBUTION_PERCENTILE:"{r["CONTRIBUTION_PERCENTILE"]}", '
            f'BIO_CONTRIBUTION_LDS:{r["BIO_CONTRIBUTION_LDS"]}, '
            f'BIO_CONTRIBUTION_ENV_PLANTING:{r["BIO_CONTRIBUTION_ENV_PLANTING"]}, '
            f'BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK:{r["BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK"]}, '
            f'BIO_CONTRIBUTION_CARBON_PLANTING_BELT:{r["BIO_CONTRIBUTION_CARBON_PLANTING_BELT"]}, '
            f'BIO_CONTRIBUTION_AGROFORESTRY:{r["BIO_CONTRIBUTION_AGROFORESTRY"]}, '
            f'BIO_CONTRIBUTION_DESTOCKING:{r["BIO_CONTRIBUTION_DESTOCKING"]}, '
            f'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT:{r["GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT"]}, '
            f'BIODIVERSITY_TARGET_GBF_4_ECNES:"{r["BIODIVERSITY_TARGET_GBF_4_ECNES"]}", '
            f'AG_MANAGEMENTS:{ag_js}, '
            f'full_settings:{full_settings_js}, '
            f'infeasible:{"true" if inf else "false"}, '
            f'infeasible_years:{json.dumps(inf_years)}, '
            f'retried_years:{retried_years_js} }}'
        )
        parts.append(js)
    return "[\n" + ",\n".join(parts) + "\n]"


def make_index(task_root_dir):
    task_name = os.path.basename(task_root_dir.rstrip('/'))
    print(f"Loading parameters for {task_name}...")
    unique_rows = load_unique_params(task_root_dir)
    template_rows, template_varying, template_run_ids = load_template(task_root_dir)

    print(f"Scanning infeasibility ({len(unique_rows)} runs)...")
    infeasibility = scan_infeasibility(task_root_dir)

    n_inf = sum(1 for v in infeasibility.values() if v[0])
    n_retried = sum(1 for v in infeasibility.values() if not v[0] and v[2])
    n_opt = len(infeasibility) - n_inf - n_retried
    print(f"  {n_opt} optimal, {n_retried} retried (recovered), {n_inf} infeasible")

    runs_js = build_runs_js(unique_rows, infeasibility, template_rows, template_varying, template_run_ids)

    # Scenario groups for filter dropdown
    groups = []
    seen = set()
    for r in unique_rows:
        g = r['scenario_group']
        if g not in seen:
            seen.add(g)
            groups.append(g)
    groups_options = '\n'.join(
        f'      <option value="{g}">{g}</option>' for g in groups
    )

    # Varying setting names for JS (used to highlight yellow in detail panel)
    varying_js = json.dumps(sorted(template_varying))

    # Run count summary line
    from collections import Counter
    group_counts = Counter(r['scenario_group'] for r in unique_rows)
    summary = ' &middot; '.join(f'{g} ({n})' for g, n in group_counts.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LUTO2 Run Index — {task_name}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px; background: #f4f6f9; color: #2d3748;
    height: 100vh; overflow: hidden; display: flex; flex-direction: column;
  }}
  header {{
    background: linear-gradient(135deg, #1a2a4a 0%, #2c3e50 100%);
    color: #fff; padding: 12px 20px; flex-shrink: 0; border-bottom: 3px solid #3498db;
  }}
  header h1 {{ font-size: 16px; font-weight: 700; letter-spacing: 0.5px; }}
  header p  {{ font-size: 11px; color: #a0b4cc; margin-top: 2px; }}

  .filter-bar {{
    background: #e8f0fb; border-bottom: 1px solid #c8d4e0;
    padding: 6px 14px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; flex-shrink: 0;
  }}
  .filter-bar label {{ font-size: 11px; font-weight: 600; color: #4a5568; display: flex; align-items: center; gap: 5px; }}
  .filter-bar select {{ font-size: 11px; padding: 2px 6px; border: 1px solid #b0bec5; border-radius: 3px; background: #fff; }}
  .filter-stats {{ margin-left: auto; font-size: 11px; color: #5a6a7e; font-weight: 600; }}

  .layout {{ display: flex; flex: 1; overflow: hidden; }}

  .left-panel {{
    width: 62%; display: flex; flex-direction: column;
    border-right: 2px solid #c8d4e0; background: #fff; overflow: hidden;
  }}
  .panel-header {{
    padding: 7px 14px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px; color: #5a6a7e;
    background: #f0f4f8; border-bottom: 1px solid #dde3ea; flex-shrink: 0;
  }}
  .left-scroll {{ overflow-y: auto; flex: 1; }}

  #index-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  #index-table thead th {{
    position: sticky; top: 0; background: #2c3e50; color: #e8edf2;
    padding: 7px 8px; text-align: left; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap;
    z-index: 2; border-right: 1px solid #3d5166;
  }}
  #index-table thead th:last-child {{ border-right: none; }}
  #index-table tbody tr {{ cursor: pointer; transition: filter 0.1s; border-bottom: 1px solid rgba(0,0,0,0.06); }}
  #index-table tbody tr:hover {{ filter: brightness(0.96); }}
  #index-table tbody tr.selected {{ box-shadow: inset 3px 0 0 #2980b9; }}
  #index-table tbody tr.selected td {{ font-weight: 600; }}
  #index-table tbody tr.hidden {{ display: none; }}
  #index-table td {{ padding: 5px 8px; vertical-align: middle; white-space: nowrap; }}

  /* Row colours per scenario group — generated dynamically by JS */
  .row-infeasible {{ position: relative; }}
  .row-infeasible::after {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      135deg, transparent, transparent 4px,
      rgba(220,53,69,0.08) 4px, rgba(220,53,69,0.08) 8px
    );
    pointer-events: none;
  }}
  .run-idx-cell {{
    font-family: 'Courier New', Courier, monospace;
    font-weight: 700; font-size: 13px; color: #1a3a5c; text-align: center; min-width: 76px;
  }}

  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 700; }}
  .badge-infeasible {{ background: #dc3545; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 700; }}
  .badge-optimal    {{ background: #28a745; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 700; }}
  .infeasible-years {{
    display: inline-block; font-family: 'Courier New', Courier, monospace;
    font-size: 10px; color: #721c24; background: #fef0f0;
    border: 1px solid #f5c6cb; border-radius: 3px; padding: 1px 5px; margin-left: 4px;
  }}
  .retried-years {{
    display: inline-block; font-family: 'Courier New', Courier, monospace;
    font-size: 10px; color: #7d4800; background: #fff3e0;
    border: 1px solid #ffd580; border-radius: 3px; padding: 1px 5px; margin-left: 4px;
  }}
  .badge-ecnes-on  {{ background: #dc3545; color: #fff; padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 700; }}
  .badge-ecnes-off {{ background: #e9ecef; color: #6c757d; padding: 1px 5px; border-radius: 3px; font-size: 10px; }}

  .right-panel {{ width: 38%; display: flex; flex-direction: column; background: #fff; overflow: hidden; }}
  .right-header {{ padding: 8px 14px; background: #f0f4f8; border-bottom: 1px solid #dde3ea; flex-shrink: 0; }}
  .right-header h2 {{ font-size: 13px; font-weight: 700; color: #1a2a4a; }}
  .right-header p  {{ font-size: 10px; color: #7a8a9a; margin-top: 2px; }}
  .right-scroll {{ overflow-y: auto; flex: 1; }}
  .placeholder {{ display: flex; align-items: center; justify-content: center; height: 100%; color: #a0adb8; font-size: 14px; gap: 8px; }}

  #detail-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  #detail-table tr.section-header td {{
    background: #2c3e50; color: #d0dcea; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.9px; padding: 5px 10px;
  }}
  #detail-table tr.setting-row td {{ padding: 5px 10px; vertical-align: top; border-bottom: 1px solid #edf0f3; }}
  #detail-table tr.setting-row:hover td {{ background: #f7f9fb; }}
  #detail-table tr.setting-row td:first-child {{
    font-weight: 600; color: #3a4a5e; width: 52%; font-size: 11px; white-space: nowrap;
  }}
  #detail-table tr.setting-row td:last-child {{ color: #2d3748; word-break: break-word; font-size: 11px; }}
  #detail-table tr.setting-row.varied td {{ background: #fffacd; }}
  #detail-table tr.setting-row.varied:hover td {{ background: #fff5a0; }}

  .mini-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  .mini-table td {{ padding: 2px 6px !important; border: none !important; border-bottom: 1px solid #eee !important; background: transparent !important; vertical-align: middle !important; }}
  .mini-table tr:last-child td {{ border-bottom: none !important; }}
  .mini-table td:first-child {{ font-size: 11px !important; color: #4a5568 !important; font-weight: 500 !important; width: 70% !important; white-space: normal !important; }}
  .badge-mini-true  {{ background: #c3e6cb; color: #155724; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }}
  .badge-mini-false {{ background: #d6d8db; color: #495057; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }}
  .val-code {{ font-family: 'Courier New', Courier, monospace; font-size: 11px; background: #f0f3f6; padding: 1px 4px; border-radius: 3px; }}

  .left-scroll::-webkit-scrollbar, .right-scroll::-webkit-scrollbar {{ width: 6px; }}
  .left-scroll::-webkit-scrollbar-track, .right-scroll::-webkit-scrollbar-track {{ background: #f0f4f8; }}
  .left-scroll::-webkit-scrollbar-thumb, .right-scroll::-webkit-scrollbar-thumb {{ background: #b0bec5; border-radius: 3px; }}
</style>
</head>
<body>

<header>
  <h1>LUTO2 Run Index &mdash; {task_name}</h1>
  <p>{len(unique_rows)} runs total &nbsp;&middot;&nbsp; {summary} &nbsp;&middot;&nbsp;
     Click a row to view all settings &nbsp;&middot;&nbsp; Yellow = varies across runs &nbsp;&middot;&nbsp; Red stripe = infeasible</p>
</header>

<div class="filter-bar">
  <label>Scenario Group:
    <select id="filter-group">
      <option value="">All</option>
{groups_options}
    </select>
  </label>
  <label>Productivity:
    <select id="filter-prod">
      <option value="">All</option>
      <option value="BAU">BAU</option>
      <option value="MEDIUM">MEDIUM</option>
      <option value="HIGH">HIGH</option>
    </select>
  </label>
  <label>ECNES:
    <select id="filter-ecnes">
      <option value="">All</option>
      <option value="on">on</option>
      <option value="off">off</option>
    </select>
  </label>
  <label>Regional Constraints:
    <select id="filter-regional">
      <option value="">All</option>
      <option value="off">off</option>
      <option value="NON_AG_CAP">NON_AG_CAP</option>
    </select>
  </label>
  <label>Status:
    <select id="filter-status">
      <option value="">All</option>
      <option value="0">Optimal only</option>
      <option value="1">Infeasible only</option>
    </select>
  </label>
  <label>Retries:
    <select id="filter-retried">
      <option value="">All</option>
      <option value="1">Has retries</option>
      <option value="0">No retries</option>
    </select>
  </label>
  <span class="filter-stats" id="filter-stats">{len(unique_rows)} / {len(unique_rows)} shown</span>
</div>

<div class="layout">
  <div class="left-panel">
    <div class="panel-header">Run Index</div>
    <div class="left-scroll">
      <table id="index-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Scenario Group</th>
            <th>Local #</th>
            <th>Productivity</th>
            <th>Regional Adopt.</th>
            <th>Non-AG Uniform</th>
            <th>ECNES</th>
            <th>BIO_LDS</th>
            <th>GBF2 Cut%</th>
            <th>Status</th>
            <th>Retries</th>
          </tr>
        </thead>
        <tbody id="index-tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="right-panel">
    <div class="right-header" id="right-header">
      <h2>Settings Detail</h2>
      <p>Select a run from the left panel</p>
    </div>
    <div class="right-scroll">
      <div class="placeholder" id="placeholder">
        <span style="font-size:20px;">&#8592;</span>
        Click a row to view full settings
      </div>
      <table id="detail-table" style="display:none;"></table>
    </div>
  </div>
</div>

<script>
// ── Data ──
const RUNS = {runs_js};
const VARYING_SETTINGS = {varying_js};
const VARYING_SET = new Set(VARYING_SETTINGS);

// ── Colour palette for scenario groups (auto-assigned) ──
const GROUP_PALETTE = [
  {{ bg: '#eaf4ff', badge: '#2980b9' }},  // blue
  {{ bg: '#fdecea', badge: '#c0392b' }},  // red
  {{ bg: '#fef9e7', badge: '#e6ac00' }},  // yellow
  {{ bg: '#e8f8f0', badge: '#27ae60' }},  // green
  {{ bg: '#fff3e0', badge: '#e67e22' }},  // orange
  {{ bg: '#f3e8ff', badge: '#8e44ad' }},  // purple
  {{ bg: '#e0f7fa', badge: '#00838f' }},  // teal
  {{ bg: '#fce4ec', badge: '#ad1457' }},  // pink
];
const groupNames = [...new Set(RUNS.map(r => r.scenario_group))];
const groupStyle = {{}};
groupNames.forEach((g, i) => {{ groupStyle[g] = GROUP_PALETTE[i % GROUP_PALETTE.length]; }});

// Inject row-background CSS rules dynamically
const styleEl = document.createElement('style');
groupNames.forEach(g => {{
  const key = g.toLowerCase().replace(/[ _-]/g, '_');
  styleEl.textContent += `.row-${{key}} {{ background: ${{groupStyle[g].bg}}; }}\\n`;
}});
document.head.appendChild(styleEl);

// ── Badge helpers ──
function scenarioBadge(s) {{
  const st = groupStyle[s] || {{ badge: '#888' }};
  return `<span style="background:${{st.badge}};color:#fff;padding:2px 7px;border-radius:3px;font-size:10px;font-weight:700">${{s.replace('Sensitivity_','Sens.')}}</span>`;
}}
function prodBadge(p) {{
  const map = {{ BAU:'#dee2e6,#495057', MEDIUM:'#cce5ff,#004085', HIGH:'#d4edda,#155724' }};
  const [bg, fg] = (map[p] || '#eee,#333').split(',');
  return `<span style="background:${{bg}};color:${{fg}};padding:1px 5px;border-radius:3px;font-size:10px;font-weight:700">${{p}}</span>`;
}}
function ecnesBadge(v) {{
  return v === 'on'
    ? `<span class="badge-ecnes-on">on</span>`
    : `<span class="badge-ecnes-off">off</span>`;
}}
function regionalBadge(v) {{
  return v === 'off'
    ? `<span class="badge" style="background:#e9ecef;color:#6c757d;font-size:10px">off</span>`
    : `<span class="badge" style="background:#17a2b8;color:#fff;font-size:10px">${{v}}</span>`;
}}
function statusBadge(r) {{
  if (r.infeasible) {{
    const ys = r.infeasible_years.join(', ');
    return `<span class="badge-infeasible">INFEASIBLE</span><span class="infeasible-years">${{ys}}</span>`;
  }}
  return `<span class="badge-optimal">Optimal</span>`;
}}
function retriesCellHtml(r) {{
  if (!r.retried_years || Object.keys(r.retried_years).length === 0) return '';
  const parts = Object.entries(r.retried_years).map(([y, nf]) => `${{y}}↑NF${{nf}}`);
  return `<span class="retried-years">${{parts.join(', ')}}</span>`;
}}

// ── Build table ──
function buildTable() {{
  const tbody = document.getElementById('index-tbody');
  RUNS.forEach((r, i) => {{
    const tr = document.createElement('tr');
    const sgKey = r.scenario_group.toLowerCase().replace(/[ _-]/g, '_');
    const hasRetry = r.retried_years && Object.keys(r.retried_years).length > 0;
    tr.className = `row-${{sgKey}}${{r.infeasible ? ' row-infeasible' : ''}}`;
    tr.dataset.group = r.scenario_group;
    tr.dataset.prod = r.PRODUCTIVITY_TREND;
    tr.dataset.ecnes = r.BIODIVERSITY_TARGET_GBF_4_ECNES;
    tr.dataset.regional = r.REGIONAL_ADOPTION_CONSTRAINTS;
    tr.dataset.infeasible = r.infeasible ? '1' : '0';
    tr.dataset.retried = hasRetry ? '1' : '0';

    tr.innerHTML = `
      <td class="run-idx-cell">${{r.run_id}}</td>
      <td>${{scenarioBadge(r.scenario_group)}}</td>
      <td style="text-align:center;font-size:11px;color:#5a6a7e">${{r.local_run_idx}}</td>
      <td>${{prodBadge(r.PRODUCTIVITY_TREND)}}</td>
      <td>${{regionalBadge(r.REGIONAL_ADOPTION_CONSTRAINTS)}}</td>
      <td style="text-align:center;font-size:11px">${{r.REGIONAL_ADOPTION_NON_AG_CAP}}%</td>
      <td>${{ecnesBadge(r.BIODIVERSITY_TARGET_GBF_4_ECNES)}}</td>
      <td style="text-align:center;font-size:11px">${{r.BIO_CONTRIBUTION_LDS}}</td>
      <td style="text-align:center;font-size:11px">${{r.GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT}}%</td>
      <td>${{statusBadge(r)}}</td>
      <td>${{retriesCellHtml(r)}}</td>
    `;
    tr.addEventListener('click', () => selectRun(i));
    tbody.appendChild(tr);
  }});
  updateStats();
}}

// ── Filtering ──
function applyFilters() {{
  const fg = document.getElementById('filter-group').value;
  const fp = document.getElementById('filter-prod').value;
  const fe = document.getElementById('filter-ecnes').value;
  const fr = document.getElementById('filter-regional').value;
  const fs = document.getElementById('filter-status').value;
  const fret = document.getElementById('filter-retried').value;
  document.querySelectorAll('#index-tbody tr').forEach(tr => {{
    let hide = false;
    if (fg && tr.dataset.group !== fg) hide = true;
    if (fp && tr.dataset.prod !== fp) hide = true;
    if (fe && tr.dataset.ecnes !== fe) hide = true;
    if (fr && tr.dataset.regional !== fr) hide = true;
    if (fs !== '' && tr.dataset.infeasible !== fs) hide = true;
    if (fret !== '' && tr.dataset.retried !== fret) hide = true;
    tr.classList.toggle('hidden', hide);
  }});
  updateStats();
}}
function updateStats() {{
  const total = RUNS.length;
  const shown = document.querySelectorAll('#index-tbody tr:not(.hidden)').length;
  const inf = document.querySelectorAll('#index-tbody tr:not(.hidden)[data-infeasible="1"]').length;
  const ret = document.querySelectorAll('#index-tbody tr:not(.hidden)[data-retried="1"]').length;
  let msg = `${{shown}} / ${{total}} shown`;
  if (inf > 0) msg += `  ·  ${{inf}} infeasible`;
  if (ret > 0) msg += `  ·  ${{ret}} with retries`;
  document.getElementById('filter-stats').textContent = msg;
}}
['filter-group','filter-prod','filter-ecnes','filter-regional','filter-status','filter-retried'].forEach(id =>
  document.getElementById(id).addEventListener('change', applyFilters)
);

// ── Detail panel ──
function selectRun(i) {{
  document.querySelectorAll('#index-tbody tr').forEach(tr => tr.classList.remove('selected'));
  document.querySelectorAll('#index-tbody tr')[i].classList.add('selected');

  const r = RUNS[i];
  const hdr = document.getElementById('right-header');
  hdr.querySelector('h2').textContent = `${{r.run_id}}  ·  ${{r.scenario_group}}  (local #${{r.local_run_idx}})`;
  const statusP = hdr.querySelector('p');
  if (r.infeasible) {{
    statusP.textContent = `INFEASIBLE — year(s): ${{r.infeasible_years.join(', ')}}`;
    statusP.style.color = '#dc3545';
  }} else {{
    const hasRetry = r.retried_years && Object.keys(r.retried_years).length > 0;
    const retrySuffix = hasRetry
      ? ` (retried: ${{Object.entries(r.retried_years).map(([y,nf])=>`${{y}} NF=${{nf}}`).join(', ')}})`
      : '';
    statusP.textContent = `Optimal — all years solved${{retrySuffix}}`;
    statusP.style.color = '#28a745';
  }}

  document.getElementById('placeholder').style.display = 'none';
  const dt = document.getElementById('detail-table');
  dt.style.display = '';
  dt.innerHTML = '';

  function section(title) {{
    const tr = document.createElement('tr');
    tr.className = 'section-header';
    tr.innerHTML = `<td colspan="2">${{title}}</td>`;
    dt.appendChild(tr);
  }}
  function row(label, value, varied=false) {{
    const tr = document.createElement('tr');
    tr.className = `setting-row${{varied ? ' varied' : ''}}`;
    tr.innerHTML = `<td>${{label}}</td><td>${{value}}</td>`;
    dt.appendChild(tr);
  }}
  function valFmt(v) {{
    if (v === undefined || v === null || v === '') return '<span class="val-na">—</span>';
    return `<span class="val-code">${{String(v).replace(/</g,'&lt;').replace(/>/g,'&gt;')}}</span>`;
  }}

  // ── Identity + Status ──
  section('Identity & Status');
  row('run_id', `<span class="val-code">${{r.run_id}}</span>`);
  row('Scenario Group', scenarioBadge(r.scenario_group), true);
  row('Global run index', r.global_run_idx);
  row('Local run index (within group)', r.local_run_idx);
  row('Status', statusBadge(r));
  if (r.infeasible && r.infeasible_years.length) {{
    row('Infeasible year(s)', `<span style="font-family:monospace;color:#dc3545">${{r.infeasible_years.join(', ')}}</span>`);
  }}
  if (r.retried_years && Object.keys(r.retried_years).length > 0) {{
    const parts = Object.entries(r.retried_years).map(([y, nf]) =>
      `<span style="font-family:monospace">${{y}}</span> <span style="font-size:10px;color:#666">(solved with NumericFocus=${{nf}})</span>`
    );
    row('Retried year(s)', parts.join('<br>'));
  }}

  // ── GEP: varying parameters ──
  section('Grid-Search Parameters (GEP — varying across runs)');
  row('PRODUCTIVITY_TREND', prodBadge(r.PRODUCTIVITY_TREND), true);
  row('REGIONAL_ADOPTION_CONSTRAINTS', regionalBadge(r.REGIONAL_ADOPTION_CONSTRAINTS), true);
  row('REGIONAL_ADOPTION_NON_AG_CAP', `${{r.REGIONAL_ADOPTION_NON_AG_CAP}}%`, true);
  row('CONTRIBUTION_PERCENTILE', valFmt(r.CONTRIBUTION_PERCENTILE), true);
  row('BIO_CONTRIBUTION_LDS', r.BIO_CONTRIBUTION_LDS, true);
  row('BIO_CONTRIBUTION_ENV_PLANTING', r.BIO_CONTRIBUTION_ENV_PLANTING, true);
  row('BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK', r.BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK, true);
  row('BIO_CONTRIBUTION_CARBON_PLANTING_BELT', r.BIO_CONTRIBUTION_CARBON_PLANTING_BELT, true);
  row('BIO_CONTRIBUTION_AGROFORESTRY', r.BIO_CONTRIBUTION_AGROFORESTRY, true);
  row('BIO_CONTRIBUTION_DESTOCKING', r.BIO_CONTRIBUTION_DESTOCKING, true);
  row('GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT', `${{r.GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT}}%`, true);
  row('BIODIVERSITY_TARGET_GBF_4_ECNES', ecnesBadge(r.BIODIVERSITY_TARGET_GBF_4_ECNES), true);

  // AG_MANAGEMENTS mini-table
  const amEntries = Object.entries(r.AG_MANAGEMENTS);
  const miniRows = amEntries.map(([k,v]) =>
    `<tr><td>${{k}}</td><td><span class="badge-mini-${{v?'true':'false'}}">${{v?'ON':'OFF'}}</span></td></tr>`
  ).join('');
  const variedAM = amEntries.some(([,v]) => !v);
  row('AG_MANAGEMENTS', `<table class="mini-table"><tbody>${{miniRows}}</tbody></table>`, variedAM);

  // ── Full settings from template (fixed settings) ──
  const skipInGEP = new Set([
    'scenario_group','global_run_idx','run_idx','local_run_idx','JOB_NAME',
    'AG_MANAGEMENTS','BIODIVERSITY_TARGET_GBF_4_ECNES','BIO_CONTRIBUTION_LDS',
    'BIO_CONTRIBUTION_ENV_PLANTING','BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK',
    'BIO_CONTRIBUTION_CARBON_PLANTING_BELT','BIO_CONTRIBUTION_AGROFORESTRY',
    'BIO_CONTRIBUTION_DESTOCKING','GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT',
    'CONTRIBUTION_PERCENTILE','PRODUCTIVITY_TREND',
    'REGIONAL_ADOPTION_CONSTRAINTS','REGIONAL_ADOPTION_NON_AG_CAP',
  ]);
  const fullEntries = Object.entries(r.full_settings)
    .filter(([k]) => !skipInGEP.has(k))
    .sort(([a],[b]) => a.localeCompare(b));

  section('All Other Settings (from settings template — yellow = varies)');
  fullEntries.forEach(([k, v]) => {{
    row(k, valFmt(v), VARYING_SET.has(k));
  }});
}}

buildTable();
</script>
</body>
</html>"""

    out_path = os.path.join(task_root_dir, "index.html")
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Written: {out_path}")
    return out_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python make_run_index.py <task_root_dir>")
        sys.exit(1)
    make_index(sys.argv[1])
```

---

## Usage

```bash
# From command line
python jinzhu_inspect_code/make_run_index.py /g/data/.../Custom_runs/MyRun

# From Python / notebook
import sys
sys.path.insert(0, '/g/data/jk53/jinzhu/LUTO/luto-2.0/jinzhu_inspect_code')
from make_run_index import make_index
make_index("/g/data/.../Custom_runs/MyRun")
```

Opens `<task_root_dir>/index.html` in a browser.

---

## Assumptions / Limitations

- **Run folder names** must be `Run_G####` (zero-padded 4-digit `global_run_idx`). This is the standard created by `create_grid_search_tasks.py`.
- **PBS logs** must be `run_*.o<jobid>` inside each run folder. If a run hasn't started yet or has no log, it appears as "Optimal" (no infeasibility detected) — check the filter-stats count against expected completed runs.
- **Infeasibility detection** parses `Trying NumericFocus=N for year YYYY`, `Infeasible model`, and `Feasible solution found with NumericFocus=N` to determine per-year outcomes. **Status column** reflects the final outcome only: `Optimal` or `INFEASIBLE`. Years that initially failed but recovered at a higher NumericFocus appear in the separate **Retries column** (e.g. `2045↑NF2`). A "Retries" filter dropdown lets you find runs that needed retries. The detail panel also lists retried year(s) and the NumericFocus level used.
- **Scenario group colours** are auto-assigned from an 8-colour palette in order of first appearance. Up to 8 groups get distinct colours; beyond that colours cycle.
- **Filter dropdowns** are hardcoded for the standard LUTO2 GEP columns. If your run uses different column names, edit the `filter-bar` HTML section and the `applyFilters()` JS function accordingly.
- The script is self-contained: copy `make_run_index.py` to `jinzhu_inspect_code/` — no extra dependencies beyond the Python standard library.

---

## Windows / Local Run Adaptations

### Colocated output directory
For Windows local runs (`run_all.py`), the run directories are inside the task root itself —
`TASK_ROOT` and `OUT_DIR` are the **same path**. Place `make_index.py` inside the task root
and set both constants to the same directory:
```python
TASK_ROOT = r"F:\path\to\Custom_runs\MyRun"
OUT_DIR   = TASK_ROOT  # index.html written alongside Run_G#### folders
```

### ECNES/SNES species-per-run CSV schema
The standard GEP columns (`PRODUCTIVITY_TREND`, `BIO_CONTRIBUTION_*`, etc.) **do not exist**
in ECNES/SNES species-isolation runs. The `merged_grid_search_parameters_unique.csv` has a
different schema:

| Column | Description |
|--------|-------------|
| `scenario_group` | `ECNES` or `SNES` |
| `global_run_idx` | Maps to `Run_G####` |
| `local_run_idx` | Index within ECNES or SNES group |
| `label` | Filesystem-safe slug of species/community name |
| `BIODIVERSITY_TARGET_GBF_4_ECNES` | `on` / `off` |
| `BIODIVERSITY_TARGET_GBF_4_SNES` | `on` / `off` |
| `GBF4_ECNES_INCLUDE_COMMUNITIES` | Python list literal, e.g. `['Alpine Sphagnum Bogs...']` |
| `GBF4_SNES_INCLUDE_SPECIES` | Python list literal, e.g. `['Acacia phasmoides']` |
| `GBF4_ECNES_SELECTED_REGIONS` | Python list literal of NRM region names |
| `GBF4_SNES_SELECTED_REGIONS` | Python list literal of NRM region names |
| `GBF3_NVIS_SELECTED_REGIONS` | Python list literal (same regions as target species) |

For this schema, write a **custom** `make_index.py` placed in `OUT_DIR`. The table columns
should be: Run, Type (ECNES/SNES badge), Local #, Species/Community (italic), NRM Regions
(tag chips), Status. Filter bar: Type dropdown, Status dropdown, free-text species search input.

Key differences from the standard script:
- Use `ast.literal_eval()` to parse the Python list columns
- Add an **ERROR** status badge (orange) for runs that crashed in `Data.__init__`
- The detail panel branches on `scenario_group`: show communities + ECNES regions for ECNES
  runs, show species + SNES regions for SNES runs
- No `merged_grid_search_template.csv` detail panel needed (all runs share the same baseline)

See `F:\Users\jinzhu\Documents\Custom_runs\NECMA_ECNES_SNES_20260502_RR\make_index.py`
for a working example of this pattern.