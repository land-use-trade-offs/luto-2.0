# Task Run Plots

Step-by-step guide for writing plot scripts against grid search / sensitivity task run outputs.

---

## Directory structure

```
/g/data/.../Custom_runs/<run_name>/
  grid_search_parameters.csv          # all runs × all params
  grid_search_parameters_unique.csv   # only params that vary across runs
  Run_0001/                           # run working dirs (may be archived as zip inside)
  Report_Data/
    Run_0001.zip                      # DATA_REPORT zips — the chart data source
    Run_0002.zip
    ...
```

---

## Primary approach: use `process_task_root_dirs`

The helper in `luto/tools/create_task_runs/helpers.py` loads all runs into a single
long-format DataFrame. **Use this first** — it handles parallelism, zip parsing, and
parameter joining automatically.

```python
import sys
sys.path.insert(0, '/g/data/jk53/jinzhu/LUTO/luto-2.0')

from luto.tools.create_task_runs.helpers import process_task_root_dirs

task_root_dir = "/g/data/.../Custom_runs/<run_name>"
report_data = process_task_root_dirs(task_root_dir)
```

### Output columns

| Column | Description |
|--------|-------------|
| `region` | `"AUSTRALIA"`, NRM region name, or state |
| `name` | Series name (land use, ag-mgt, GHG source, etc.) |
| `year` | Simulation year (int) |
| `value` | Raw value in the units of the `Type` |
| `Type` | Data category (see table below) |
| `ag_mgt` | Agricultural management type (only for `Area_ag_man_ha`) |
| `water_supply` | Land use name (only for `Area_ag_man_ha`) |
| `run_idx` | Run number |
| *param columns* | All varying parameters from `grid_search_parameters_unique.csv` |

### `Type` values

| Type | Source file | Units | Notes |
|------|-------------|-------|-------|
| `Area_broad_category_ha` | `Area_overview_1_Land-use.js` | ha | Ag / NonAg / AgMgt broad categories |
| `Area_non_ag_lu_ha` | `Area_NonAg.js` | ha | Individual non-ag land uses |
| `Area_ag_man_ha` | `Area_Am.js` | ha | Ag management options; has extra `ag_mgt` + `water_supply` columns |
| `Economic_AUD` | `Economics_overview_sum.js` | AUD | Revenue, cost, profit components |
| `GHG_tCO2e` | `GHG_overview_sum.js` | tCO2e | Emissions + net + limit |
| `Production_deviation_percent` | `Production_overview_AUS_achive_percent.js` | % | Deviation from demand target |
| `Bio_relative_to_PRE1750_percent` | `BIO_GBF2_overview_sum.js` | % | GBF2 biodiversity relative to pre-1750 |

### Filter to Australia-only (most common)

```python
df = report_data.query('region == "AUSTRALIA"').copy()
```

---

## Recipes

### Area of a specific non-ag land use over time

```python
df_env = (
    report_data
    .query('region == "AUSTRALIA" and Type == "Area_non_ag_lu_ha"')
    .query('name == "Environmental plantings (mixed species)"')
    .eval('value_Mha = value / 1e6')
)
# df_env columns: year, value_Mha, run_idx, <param columns>
```

### Area of a specific ag-management option over time

```python
df_savanna = (
    report_data
    .query('region == "AUSTRALIA" and Type == "Area_ag_man_ha"')
    .query('ag_mgt == "ALL" and water_supply == "ALL"')   # national total
    .query('name == "Early dry-season savanna burning"')
    .eval('value_Mha = value / 1e6')
)
```

### GHG net emissions

```python
df_ghg_net = (
    report_data
    .query('region == "AUSTRALIA" and Type == "GHG_tCO2e"')
    .query('name == "Net emissions"')
    .eval('value_MtCO2e = value / 1e6')
)
```

### Economics profit

```python
df_profit = (
    report_data
    .query('region == "AUSTRALIA" and Type == "Economic_AUD"')
    .query('name == "Profit"')
    .eval('value_Bbn = value / 1e9')
)
```

---

## Secondary approach: read raw JS files from zip

Use this when you need data **not** covered by `process_task_root_dirs`
(e.g. `Area_Ag.js`, `Water_overview_NRM_sum.js`, `BIO_quality_overview_sum.js`, etc.).

### Parse a JS data file

All `DATA_REPORT/data/*.js` files are JS variable assignments, not pure JSON:

```python
import zipfile, json

def read_js(zip_path, filename):
    with zipfile.ZipFile(zip_path) as z:
        if f"DATA_REPORT/data/{filename}" not in z.namelist():
            return {}
        with z.open(f"DATA_REPORT/data/{filename}") as f:
            raw = f.read().decode("utf-8")
            return json.loads(raw.split("=")[1][:-2])
```

### Nesting structures

| File pattern | Nesting | How to navigate |
|---|---|---|
| `*_overview_*.js`, `Area_NonAg.js` | `region → [records]` | `data["AUSTRALIA"]` → list |
| `Area_Ag.js`, `Economics_Ag_*.js` | `region → water_supply → [records]` | `data["AUSTRALIA"]["ALL"]` |
| `Area_Am.js`, `Economics_Am_*.js` | `region → ag_mgt → water_supply → [records]` | `data["AUSTRALIA"]["ALL"]["ALL"]` |

Each `record` has: `{"name": "...", "data": [[year, value], ...]}`

### Loop over all runs

```python
import pandas as pd

params_df = pd.read_csv(f"{task_root_dir}/grid_search_parameters_unique.csv")

records = []
for _, row in params_df.iterrows():
    run_idx = int(row["run_idx"])
    zip_path = f"{task_root_dir}/Report_Data/Run_{run_idx:04d}.zip"

    try:
        data = read_js(zip_path, "Water_overview_NRM_sum.js")
    except Exception as e:
        print(f"Run_{run_idx:04d}: {e}")
        continue

    for record in data.get("AUSTRALIA", []):
        for year, value in record["data"]:
            records.append({
                "run_idx": run_idx,
                "name": record["name"],
                "year": int(year),       # cast — int64 is not JSON-serialisable
                "value": float(value),   # cast — float64 is not JSON-serialisable
                **{k: row[k] for k in ["MY_PARAM"]},
            })

df = pd.DataFrame(records)
```

---

## Build an ECharts HTML web app

ECharts from CDN is the preferred interactive output format. The pattern:
1. Build a Python `datasets` list (one entry per sensitivity value / series)
2. Colour by a viridis gradient
3. Embed as JSON in a self-contained HTML file

### Viridis colour helper

```python
def viridis_hex(t):
    stops = [
        (0.0,  (68,  1,  84)),
        (0.25, (59,  82, 139)),
        (0.5,  (33, 145, 140)),
        (0.75, (94, 201,  98)),
        (1.0,  (253, 231,  37)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            r, g, b = [int(c0[j] + f * (c1[j] - c0[j])) for j in range(3)]
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#000000"
```

### Build datasets

```python
import json

PARAM_COL = "MY_PARAM"   # the sensitivity parameter column name

years = [int(y) for y in sorted(df["year"].unique())]
param_vals = [float(v) for v in sorted(df[PARAM_COL].unique())]
n = len(param_vals) - 1

datasets = []
for i, pval in enumerate(param_vals):
    grp = df[df[PARAM_COL] == pval].sort_values("year")
    year_to_val = dict(zip(grp["year"], grp["value_Mha"]))
    color = viridis_hex(i / n if n > 0 else 0)
    datasets.append({
        "label": f"{pval:.2f}",
        "data": [year_to_val.get(y) for y in years],
        "color": color,
    })
```

### Write HTML

```python
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>My sensitivity plot</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {{ margin: 0; background: #f8f8f8; font-family: sans-serif; }}
  #chart {{ width: 100%; height: 100vh; }}
</style>
</head>
<body>
<div id="chart"></div>
<script>
const years = {json.dumps(years)};
const rawSeries = {json.dumps(datasets)};

const series = rawSeries.map(ds => ({{
  name: ds.label,
  type: 'line',
  data: ds.data,
  itemStyle: {{ color: ds.color }},
  lineStyle: {{ color: ds.color, width: 1.5 }},
  symbolSize: 5,
}}));

const chart = echarts.init(document.getElementById('chart'));
chart.setOption({{
  title: {{
    text: 'My metric over time',
    subtext: '{PARAM_COL} sensitivity',
    left: 'center',
  }},
  tooltip: {{
    trigger: 'axis',
    valueFormatter: v => v != null ? v.toFixed(2) + ' Mha' : 'N/A',
  }},
  legend: {{
    type: 'scroll',
    orient: 'vertical',
    right: 10,
    top: 60,
    bottom: 20,
  }},
  grid: {{ left: 60, right: 160, top: 80, bottom: 50 }},
  xAxis: {{
    type: 'category',
    data: years,
    name: 'Year',
    nameLocation: 'middle',
    nameGap: 30,
  }},
  yAxis: {{
    type: 'value',
    name: 'Value (million ha)',
    nameLocation: 'middle',
    nameGap: 45,
  }},
  dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', bottom: 5 }}],
  series,
}});

window.addEventListener('resize', () => chart.resize());
</script>
</body>
</html>
"""

out_path = f"{task_root_dir}/my_plot.html"
with open(out_path, "w") as f:
    f.write(html)
print(f"Saved {out_path}")
```

---

## Key conventions

- **Save scripts** to `jinzhu_inspect_code/`, not inside the source tree
- **Output HTML** to the task run root dir
- Always cast to native Python types before `json.dumps()`: `int(year)`, `float(value)` — pandas `int64`/`float64` are not JSON-serialisable
- Use `"ALL"` keys to get national/aggregated values when navigating raw JS nesting
- `grid_search_parameters_unique.csv` only has columns that **vary** across runs — use it as the loop driver
- To invert a parameter for display (e.g. penalty → contribution): apply `1 - val` when building records, before grouping or plotting
