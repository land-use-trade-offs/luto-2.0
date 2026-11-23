# LUTO2: The Land-Use Trade-Offs Model Version 2.0

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/github/v/release/land-use-trade-offs/luto-2.0?label=Version&color=green)](https://github.com/land-use-trade-offs/luto-2.0/releases/latest)

## Introduction
The Land Use Trade-Offs model v2 (LUTO2) is an integrated land systems model designed to simulate the optimal spatial arrangement of land use and land management decisions over time in Australia. It aims to achieve climate and biodiversity targets without compromising economic growth, food production or water security. The model is implemented as a Python package, offering users the flexibility to run interactively or to execute batch processes through scripted automation.

LUTO2 was developed through a collaboration between Deakin University and Climateworks Centre, with research contributions from CSIRO. The model is a cornerstone of Climateworks’ Land Use Futures program, which supports Australia’s transition to sustainable food and land systems. The technical development of LUTO2 is led by Professor Brett Bryan at Deakin University. LUTO2 continues the approach to land-use change modelling of its predecessor, the original LUTO, which was developed by CSIRO from 2010 - 2015 (see also Pedigree, below) and published under the GNU GPLv3 in 2021.

## Pedigree
LUTO2 builds on the approach and pedigree of nearly two decades of land-use modelling expertise starting with the original LUTO model. The original LUTO model was developed by CSIRO for the Australian National Outlook in 2015 and was groundbreaking for quantifying and projecting land use changes and their sustainability impacts in Australia, illustrated by its published works in *Nature* in 2015 and 2017.

LUTO2 represents a generational leap in sophistication and functionality for national-scale land-use change modelling in Australia. Both LUTO versions are optimisation models but different commercial solvers are used (CPLEX in original LUTO, GUROBI in LUTO2). The spatial domains are different in extent, with LUTO2's being nearly 5 times as large. The data requirements to run LUTO2 are consequently different and heavier. There is no backwards compatibility whatsoever.

The original LUTO model is available online and should be cited as:
> Bryan, Brett; Nolan, Martin; Stock, Florian; Graham, Paul; Dunstall, Simon; Ernst, Andreas; Connor, Jeff (2021): Land Use Trade-Offs (LUTO) Model. v1. CSIRO. Software Collection. https://doi.org/10.25919/y8ee-sk45.

This new version represents an entirely new model featuring a complete rewrite of the codebase and comprehensive upgrades to data and functionality. Enhancements to the original model include extended spatial coverage and timespan (2010 to 2100), a complete refresh of input data, additional land-use options and sustainability indicators and management solutions, the ability to model demand-side solutions, and additional environmental indicators and reporting. Due to LUTO2’s model complexity, the computational requirements to run the model are far more intensive.

LUTO2’s modelling approach, indicators and solutions have been guided by extensive stakeholder consultation (documented here: https://doi.org/10.1007/s11625-024-01574-7) following principles of participatory model co-design.

## Authors
Coordinating lead author: **Bryan, B.A.**  

Lead authors (in order of contribution): **Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H.**  

Other significant contributors (in alphabetical order): **Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., Thiruvady, D.R.**

## Documentation
Documentation, including instructions on how to set up and run LUTO2, can be found at `docs/luto2-overview.pdf`.

LUTO2 comes with a full diagram to illustrate its data preparation, workflow, and code logics. The diagram link can be found in this link.  
*Replace with updated documentation currently in preparation when ready.*

Developer-facing reference material lives alongside the code:

| File | Covers |
|------|--------|
| `CLAUDE.md` | Repository map, conventions, and pointers into the themed docs below |
| `docs/CLAUDE_SETUP.md` | Environment setup, running the model, settings, memory profiling |
| `docs/CLAUDE_ARCHITECTURE.md` | Simulation engine, economics modules, solver integration |
| `docs/CLAUDE_GBF2.md` | GBF2 priority degraded areas: mask, targets, constraint, reporting |
| `docs/CLAUDE_OUTPUT.md` | NetCDF output format, map layers, write pipeline |
| `docs/CLAUDE_VUE_REPORTING.md` | Vue.js reporting interface and data hierarchies |
| `docs/CLAUDE_SKILL/` | Step-by-step guides for recurring tasks (task runs, retries, infeasibility debugging) |
| `docs/FINDINGS.md` | Running log of investigations: solver numerics, transition-cost audits, performance profiling |

## Project Structure

The LUTO2 codebase is organized into the following structure:

```
luto/                                    # Main package directory
├── data.py                              # Core data management and loading
├── simulation.py                        # Main simulation engine (incl. checkpoint/resume)
├── settings.py                          # Configuration parameters
├── dataprep.py                          # Data preprocessing utilities
├── economics/                           # Economic models and calculations
│   ├── agricultural/                    # Agricultural economics modules
│   │   ├── biodiversity.py              # Biodiversity calculations
│   │   ├── cost.py                      # Cost calculations
│   │   ├── ghg.py                       # GHG emissions calculations
│   │   ├── quantity.py                  # Production quantity calculations
│   │   ├── revenue.py                   # Revenue calculations (includes dynamic pricing)
│   │   ├── transitions.py               # Land use transition costs
│   │   └── water.py                     # Water yield calculations
│   ├── non_agricultural/                # Non-agricultural economics modules
│   │   ├── biodiversity.py              # Non-ag biodiversity impacts
│   │   ├── cost.py                      # Non-ag establishment costs
│   │   ├── ghg.py                       # Non-ag GHG calculations
│   │   ├── quantity.py                  # Non-ag production quantities
│   │   ├── revenue.py                   # Non-ag revenue streams
│   │   ├── transitions.py               # Non-ag transition costs
│   │   └── water.py                     # Non-ag water impacts
│   └── off_land_commodity/              # Off-land commodity economics
├── solvers/                             # The optimization model: a row table over a column table
│   ├── col_builder.py                   # The column table: every decision variable, its bounds and fields
│   ├── row_inputs.py                    # The coefficient streams and targets the rows read
│   ├── row_builder.py                   # The row table: every constraint family, and the objective
│   ├── row_table.py                     # How the rows are stored and queried
│   ├── row_bounds.py                    # Pre-solve bound check: every row's verdict, the bound report
│   ├── solver.py                        # GUROBI solver interface (LutoSolver)
│   ├── post_solve.py                    # The solution back into the LUTO format
│   └── tools.py                         # Shadow prices
└── tools/                               # Utility tools and scripts
    ├── __init__.py                      # Shared helpers, shadow-price recording
    ├── create_task_runs/                # Task execution and batch processing
    │   ├── bash_scripts/                # PBS/local launcher scripts
    │   │   ├── cmd.sh                   # PBS job script (fresh run + checkpoint resume)
    │   │   ├── python_script.py         # Per-run entry point; archives results
    │   │   └── run_all.py               # Submit/launch all runs (PBS or local Windows)
    │   ├── create_grid_search_tasks.py  # Grid search task generation
    │   ├── create_grid_search_plots.py  # Grid search result plotting
    │   ├── helpers.py                   # Task run utilities
    │   └── parameters.py                # Task run parameters
    ├── report/                          # Reporting and visualization system
    │   ├── VUE_modules/                 # Vue.js 3 interactive reporting dashboard
    │   │   ├── assets/                  # Shapefiles and styling assets (NRM, state, AEMO REZ)
    │   │   ├── components/              # Reusable Vue components
    │   │   ├── data/                    # Chart data, map layers and geometry
    │   │   │   ├── chart_option/        # Chart configuration options
    │   │   │   ├── geo/                 # Geographic boundary data (NRM, state, REZ)
    │   │   │   └── map_layers/          # Map layer data (split per dimension combo)
    │   │   ├── dataTransform/           # Data transformation scripts
    │   │   ├── lib/                     # JavaScript libraries (Vue, Leaflet, Highcharts)
    │   │   ├── routes/                  # Vue router configuration
    │   │   ├── services/                # Data and map services
    │   │   ├── views/                   # Vue view components (11 modules)
    │   │   ├── index.html               # Main HTML entry point
    │   │   └── index.js                 # Vue application entry
    │   ├── data_tools/                  # Data processing for reports
    │   ├── create_report_data.py        # Generate chart data files
    │   └── create_report_layers.py      # Generate map layer files
    ├── mem_monitor.py                   # Memory profiling decorator and live plot
    └── write.py                         # Output writing functions

input/                                   # Input data directory (requires separate download)
output/                                  # Simulation outputs with interactive HTML reports
docs/                                    # Documentation files
requirements.yml                         # Python package dependencies (conda environment spec)
pyproject.toml                           # Project configuration
```

## Memory Profiling and Monitoring

LUTO2 includes a built-in memory monitoring tool (`luto.tools.mem_monitor`) for tracking memory usage with live visualization. This is particularly useful for optimizing memory-intensive functions and identifying memory bottlenecks.

### Using the Memory Monitor as a Decorator

The **recommended way** to monitor memory usage is using the `@trace_mem_usage` decorator. It automatically handles starting, monitoring, and cleanup:

```python
from luto.tools.mem_monitor import trace_mem_usage

@trace_mem_usage
def my_expensive_function(data):
    """This function's memory usage will be automatically monitored."""
    result = process_large_data(data)
    return result

# Usage - monitoring happens automatically
result = my_expensive_function(my_data)
```


## Troubleshooting

### Common Issues

**Memory Errors:**
- Raise `RESFACTOR` — it is the single biggest lever on memory use
- Lower `WRITE_REPORT_MAX_MEM_MB` so fewer parallel write workers are spawned, and turn off `WRITE_GBF4_SNES` / `WRITE_GBF4_ECNES` / `WRITE_GBF3_NVIS` if those layers are not needed
- Close other applications during simulation
- Use the memory monitor to identify memory-intensive operations

**GUROBI License Issues:**
- Verify your license file location
- Check license expiration date
- Ensure your license supports the required model size

**Data Loading Errors:**
- Verify all required input files are present in `/input/`
- Check file permissions
- Ensure sufficient disk space

**Non-optimal Solves (INFEASIBLE / NUMERIC):**
- A year is only accepted when Gurobi returns `GRB.OPTIMAL`; the attempts in `settings.RETRY_PARAMS` are tried in order first
- The default is ONE attempt, the homogeneous barrier. It solves the grouped RES1 models on which the plain barrier ends in "Numerical trouble", but on a highly degenerate model it can report a false INFEASIBLE; such a year stops. Re-run it from its checkpoint with another attempt in `RETRY_PARAMS` (e.g. the plain barrier, `(0, 2, -1, -1, -1)`)
- A genuinely infeasible year is usually a handful of targets that cannot be met on the available land. Start with the bound report every year writes before its model is built — see [Diagnosing Infeasible Years](#diagnosing-infeasible-years)

### Diagnosing Infeasible Years

Every constraint row is a linear expression over decision variables whose bounds are already known. Before each year's model is built, `luto/solvers/row_bounds.py` works out the range each row's left-hand side can take and compares it with the row's target — first over the variable bounds alone, then under the bounds the rows themselves imply (the land-use shares of a cell cannot together exceed its agricultural area; a transition cannot move more land than its source holds). Every row, in every constraint family, gets a verdict:

| Verdict | Meaning | What happens |
|---|---|---|
| `impossible` | No feasible point can meet the row | Reported with its shortfall in physical units. With `BOUND_PROP_ON_IMPOSSIBLE = 'stop'` (default) the run stops at that year before the model is built, since no solve could succeed |
| `redundant` | Every point within the variable bounds meets the row | Reported. Dropped before the build if its family is listed in `BOUND_PROP_DROP_FAMILIES` — exact: the solution does not change, and the row appears in the shadow prices at 0 with `dropped = True` |
| `tight` | The row can be met only at the extreme of its bounds (a knife-edge) | Reported for the target families (GBF2/3/4/8, water, GHG, demand, renewable, regional adoption) |
| `near_redundant`, `straddle` | Met everywhere up to the tolerance; nothing proven | Counted in the run log |

**Outputs** in `out_<year>/`, written before the solve — so they exist even when the year fails:
- `bound_report_<year>.csv` — one line per impossible, redundant or tight row: `family`, `name` (the constraint name in the saved model), the row's labels (`region` — the NRM / IBRA region, water region id, state or regional cap the row is about; `GBF_target` = the vegetation group / species / community a GBF3 / GBF4 / GBF8 row targets; `GBF4_presence`; `demand_commodity`; `demand_bound` = eq / lower / upper; `am_idx` = the renewable option of a renewable row), `rhs_raw` (the target), `lo_raw` / `hi_raw` (the range over the variable bounds), `lo_implied_raw` / `hi_implied_raw` (the range under the implied bounds), `best_raw` (how far the target is met at the most favourable reachable point — negative is the shortfall), `worst_raw`, `unit` (ha, ML, tCO2e, t, MWh), `status` / `status_implied`, `dropped`
- `bound_preflight_<year>.csv` — for each variable block: variables that no constraint touches, and NaN or inverted bounds
- The run log — a table of rows per family and verdict, and one line per impossible row with its shortfall

The check needs no solver — it is arithmetic on the constraint matrix and the variable bounds — so the report can be reproduced and shared without a Gurobi licence.

**What it cannot see.** It proves facts about one row at a time. Targets that are each achievable on their own but compete for the same cells (several species sharing habitat, a water limit against a planting target) are not detected, so "nothing proven" does not mean feasible. When the solver still reports INFEASIBLE:
- Let the suspect families fall short instead of failing: list them in `ELASTIC_FAMILIES` and re-run the year from its checkpoint. Every row that cannot be met together with the rest of the year is then named in `out_<year>/shortfall_<year>.csv` with the amount it misses (with `ELASTIC_DROP_SHORT`, pass 1 minimises only the shortfall: a row still short there cannot be met together with the hard rows)
- Work on the model space itself: a year's model is the column table `cols`, the row table `rows` and one scipy matrix `row_matrix` (plus `obj`), plain numpy / scipy / xarray. A family or a region is a mask on the row table (`rows['family'].values == name`, `rows['region']`), a cell's columns a mask on `cols['cell']`, so any subset of rows can be switched off (`active`) or examined (activity, ranges, coefficients) without a solver and without parsing names. There is no IIS tooling: an IIS is a statement about the model handed to it and was found to mislead (docs/FINDINGS.md, 20260810)

The settings (`BOUND_PROP_REL_TOL`, `BOUND_PROP_DROP_FAMILIES`, `BOUND_PROP_ON_IMPOSSIBLE`) are under [Solver Configuration](#solver-configuration).


### Getting Help

1. Check the documentation in `docs/luto2-overview.pdf`
2. Review `LUTO_RUN__stdout.log` and `LUTO_RUN__stderr.log` in the run directory
3. Contact the development team: **b.bryan@deakin.edu.au**
4. Submit issues on GitHub: [github.com/land-use-trade-offs/luto-2.0](https://github.com/land-use-trade-offs/luto-2.0)

## System Requirements

**Minimum Requirements:**
- Python 3.14 (pinned in `requirements.yml`)
- 16 GB RAM at `RESFACTOR >= 10`; 32 GB or more for `RESFACTOR = 5`. Full resolution (`RESFACTOR = 1`) is an HPC workload: with `CELL_GROUPING = True` a year takes about 1 h and the run peaks at about 290 GB (2020–2050, default targets, no report, 2026-11-18); the write/report phase at RES1 is not yet tested with grouping.
- 50 GB available disk space for input data and outputs
- GUROBI optimization solver license (academic licenses available); `gurobipy` is pinned to 13.0.3
- We are working on support for other solving engines, such as IBM CPLEX and NVIDIA cuOpt (see [Other solving engines](#other-solving-engines-in-progress)). GUROBI is the only engine the model solves with today

**Supported Operating Systems:**
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+)

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/land-use-trade-offs/luto-2.0.git
cd luto-2.0
```

### 2. Set Up Environment

```bash
# Create and activate the LUTO environment from requirements.yml
conda env create -f requirements.yml
conda activate luto
```

### 3. Configure the Solver
LUTO2 currently solves with GUROBI. Follow these steps:
```bash
# 1) Set up your GUROBI license (academic license available at gurobi.com)
# 2) Place your gurobi.lic file in the appropriate directory
```

#### Other solving engines (in progress)

We are working on letting a run solve with engines other than GUROBI:

- **IBM CPLEX**, a commercial LP solver on the CPU and the engine the original LUTO used. The `cplex` and `docplex`
  packages are already in `requirements.yml`; the native runtime needs a licensed IBM ILOG CPLEX Optimization
  Studio 22.2 installation.
- **NVIDIA cuOpt**, an open-source, GPU-accelerated solver. Tested (cuOpt 26.08): its GPU barrier is 2.5 – 7× faster
  than Gurobi on models with few wide rows, but it fails on models with GBF8 species rows and on the default grouped
  RES1 model, under every linear system it offers, and its own crossover would take about a day. Not reliable for
  LUTO today; every test is listed in `docs/FINDINGS.md` (20261119). Gurobi's own GPU method (PDHG) has not converged
  on any LUTO model.

The model build is what makes this practical. Each year's LP is handed to the solver as plain arrays: the column
table (bounds), one scipy CSR matrix `A`, the row table (rhs, sense) and the objective vector. So a new engine only
needs a small adapter in `luto/solvers/solver.py`; the model code (`col_builder`, `row_builder`, `row_bounds`,
`post_solve`) doesn't change. A saved `debug_model_*.mps` file already loads in other engines, which is how we compare
them. Nothing in the model selects an engine yet: there is no engine setting, and every run goes through GUROBI.

### 4. Obtain Input Data
The LUTO2 input database is approximately 40 GB and contains sensitive data. 
Please contact **b.bryan@deakin.edu.au** to request access to the input dataset.


## Running LUTO2

### Basic Simulation
```python
import luto.simulation as sim

# Load input data and settings
data = sim.load_data()

# Run simulation with default parameters
results = sim.run(data=data)
```

### Advanced Configuration

Several modules read settings at import time, so the reliable way to configure a scenario is to edit `luto/settings.py` — which is exactly what the batch tooling does, writing one settings file per run. If you patch `luto.settings` from a script instead, do it before importing `luto.simulation`:

```python
import luto.settings as settings

settings.RESFACTOR = 10                                 # 10 makes the spatial resolution to ~10km.
settings.SIM_YEARS = list(range(2020, 2051, 5))

settings.WATER_LIMITS = 'on'                            # 'on' or 'off'.
settings.GHG_EMISSIONS_LIMITS = 'low'                   # 'off', 'low', 'medium', or 'high'

settings.GBF2_TARGET = 'high'                           # 'off', 'low', 'medium', or 'high'
settings.GBF3_NVIS_TARGET = 'off'                       # 'off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED'
settings.GBF3_NVIS_REGION_MODE = 'NRM'                  # 'AUSTRALIA', 'NRM', or 'IBRA_REG'
settings.GBF4_TARGET_SNES = 'off'                       # 'off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED'
settings.GBF4_TARGET_ECNES = 'off'                      # 'off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED'
settings.GBF8_TARGET = 'off'                            # 'off', 'medium', 'high', or 'CSV_DEFINED'

settings.DYNAMIC_PRICE = True                           # Demand elasticity-based dynamic pricing

settings.RENEWABLES_OPTIONS = {'Utility Solar PV': True, 'Onshore Wind': True}       # Enable renewable energy types
settings.RENEWABLE_TARGET_SCENARIO_TARGETS = 'Gladstone - Core'                      # Generation target scenario
settings.RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS = 'step_change'                      # Spatial layer scenario

import luto.simulation as sim

data = sim.load_data()
sim.run(data=data)
```

IBRA bioregion targets run through the NVIS stream — select them with `GBF3_NVIS_REGION_MODE = 'IBRA_REG'`. There is no separate IBRA target setting.

### Checkpointing and Resume

Every run checkpoints into its own output folder, as two files: `data_<year>.lz4`, the whole `Data`, written once before the first solve (`<year>` is where the run starts), and `solution_<year>.lz4`, what solving has written on `Data` so far (the attributes `Data.SOLVED_STATE` names), rewritten whenever a year solves to optimality with only the latest kept. The inputs never change after loading, so only the small file is rewritten each year (RES1: 4 s and 0.2 GB, against about 4 minutes and 4.3 GB for the whole `Data`). Pass the run's folder as `checkpoint_dir` to resume from the latest solved year instead of starting over:

```python
sim.run(data=None, checkpoint_dir='output/<run timestamp>_RF1_2010-2050')
```

A year is only accepted (and checkpointed) when the solver returns `GRB.OPTIMAL`. If a solve attempt fails, the next attempt listed in `settings.RETRY_PARAMS` is tried; the default lists one, so a failed year stops the run and can be resumed from its checkpoint.

### Batch and Grid-Search Runs

Multi-scenario runs are generated and launched from `luto/tools/create_task_runs/`:

```bash
# 1) Define the grid and write one directory per scenario
python luto/tools/create_task_runs/create_grid_search_tasks.py

# 2) Submit them — PBS on HPC, or as local processes on Windows
python luto/tools/create_task_runs/bash_scripts/run_all.py
```

Each run directory carries its own `luto/settings.py`, so scenarios are fully reproducible. `run_all.py` handles both fresh runs and checkpoint resumes; see `docs/CLAUDE_SKILL/create_task_runs.md`.

### Viewing Results
Results are saved in a run directory named `output/<timestamp>_RF<resfactor>_<first_year>-<last_year>/`:

1. **Interactive HTML Dashboard:** 
   ```
   /output/<run_dir>/DATA_REPORT/index.html
   ```
   A Vue.js 3 based interactive dashboard featuring:
   - **Multi-module Analysis:** Area, Economics, GHG, Production, Water, Biodiversity, Renewable, Transition
   - **Region-level Switching:** Australia, state, and NRM views of the same data
   - **Progressive Data Selection:** Region → Category → Water/AgMgt → Landuse hierarchies
   - **Dual Visualization:** Charts (Highcharts) and Maps (Leaflet) for all data types
   - **Dynamic Filtering:** Responsive dropdowns with cascading selection updates
   - **Export Capabilities:** Chart and map export functionality
   - **11 Specialized Views:** Individual modules for detailed analysis

2. **Raw Data Outputs:**
   - **NetCDF Files:** Spatial datasets (`.nc`) for each year and variable
   - **CSV Files:** Tabular data summaries for regional analysis
   - **Shadow Prices:** `out_<year>/shadow_prices_<year>.csv` — the dual value of every binding constraint (GHG, water, demand, GBF2/3/4/8, renewable, regional adoption), reported both per real unit and normalised to AUD; rows dropped before the build as redundant are listed with price 0 and `dropped = True`
   - **Bound Reports:** `out_<year>/bound_report_<year>.csv` and `bound_preflight_<year>.csv` — the verdict of the pre-solve bound check for every impossible, redundant and tight row, with targets and shortfalls in physical units, written before the model is built (see [Diagnosing Infeasible Years](#diagnosing-infeasible-years))

3. **Execution Logs:** 
   - `LUTO_RUN__stdout.log`: Standard output logs
   - `LUTO_RUN__stderr.log`: Error and warning logs
   - Memory usage logs for performance monitoring

## Configuration

LUTO2 behavior can be customized through the `luto.settings` module. Key parameters include:

### Core Simulation Parameters
- `SIM_YEARS`: Simulation time period (default: 2020-2050 in 5-year steps)
- `SCENARIO`: Shared Socioeconomic Pathway (SSP1-SSP5)
- `RCP`: Representative Concentration Pathway (e.g., 'rcp4p5')
- `OBJECTIVE`: Optimization objective ('maxprofit' or 'mincost')

### Demand Constraints
- Every constraint is hard: production is forced into per-commodity bounds (there is no soft, penalised option)
- `DEMAND_BOUNDS`: Per-commodity `[lower, upper]` multipliers applied to the demand target. Most commodities are pinned at `[1.0, 1.0]`; sheep wool is relaxed because meat and wool are co-produced in biologically fixed ratios

### Environmental Constraints
- `GHG_EMISSIONS_LIMITS`: Greenhouse gas emission targets ('off', 'low', 'medium', 'high')
- `WATER_LIMITS`: Whether to enforce water yield constraints ('on' or 'off')
- `CARBON_EFFECTS_WINDOW`: Years for carbon accumulation averaging (50, 60, 70, 80, or 90 years)
  - Determines the time period over which carbon sequestration is averaged
  - Must match available ages in NetCDF input data
  - Default: 60 years (based on S-curve carbon accumulation pattern)
- `GBF2_TARGET`: Global Biodiversity Framework Target 2 ('off', 'low', 'medium', 'high')
- `GBF3_NVIS_TARGET`: Conservation targets for vegetation groups ('off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED')
- `GBF3_NVIS_REGION_MODE`: Spatial framing of the GBF3 targets — 'AUSTRALIA', 'NRM', or 'IBRA_REG'. IBRA bioregion targets are served by this mode; there is no separate IBRA setting
- `GBF4_TARGET_SNES`: Species of National Environmental Significance ('off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED')
- `GBF4_TARGET_ECNES`: Ecological Communities of National Environmental Significance ('off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED')
- `GBF4_SNES_TARGETS_OVERRIDE` / `GBF4_SNES_CAP_MARGIN`: Per-species target overrides, and the safety margin subtracted from each species' attainable level to keep a feasibility buffer
- `GBF8_TARGET`: Species climate-suitability targets ('off', 'medium', 'high', or 'CSV_DEFINED')

### Renewable Energy Constraints
- `RENEWABLES_OPTIONS`: Dict of renewable energy types and whether each is enabled (e.g., `{'Utility Solar PV': True, 'Onshore Wind': True}`). Set values to `False` to disable individual types.
- `RENEWABLE_TARGET_SCENARIO_TARGETS`: Generation target scenario. Valid values: `'AEMO 2026 ISP - Accelerated Transition'`, `'AEMO 2026 ISP - Slower Growth'`, `'AEMO 2026 ISP - Step Change'`, `'Gladstone - BESS Sensitivity'`, `'Gladstone - Core'`
- `RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS`: Spatial layer scenario. Valid values: `'step_change'`, `'accelerated_transition'`, `'ANU_transmission_T3'`, `'ANU_transmission_T5'`, `'ANU_transmission_T10'`
- `RE_TARGET_LEVEL`: Spatial level for targets ('STATE' or 'NRM'; only STATE currently supported)
- `INSTALL_CAPACITY_MW_HA`: Per-hectare generation capacity (MW/ha) for each renewable type
- `RENEWABLES_ADOPTION_LIMITS`: Maximum fraction of compatible land available for each renewable type (default: 1.0)
- `EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS`: Prevent renewable installation on high-biodiversity GBF2-masked cells (default: True)
- `RENEWABLE_GBF2_CUT_SOLAR` / `RENEWABLE_GBF2_CUT_WIND`: Biodiversity area coverage % threshold for GBF2 exclusion (default: 20)
- `EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK`: Prevent renewable installation on EPBC MNES high-priority cells (default: True)
- `RENEWABLE_EPBC_MNES_CUT_SOLAR` / `RENEWABLE_EPBC_MNES_CUT_WIND`: MNES priority rank % threshold for EPBC exclusion (default: 10)

> **Existing capacity reporting**: Pre-simulation real-world solar/wind installations are automatically included in all output reports as `lu='Existing Capacity'` — visible in area, economics, production, and map layers without any extra configuration.

### Land Use Options
- `NON_AG_LAND_USES`: Enable/disable non-agricultural land uses (Environmental Plantings, Carbon Plantings, etc.)
- `AG_MANAGEMENTS`: Enable/disable agricultural management practices (Precision Agriculture, Biochar, Utility Solar PV, Onshore Wind, etc.)
- `EXCLUDE_NO_GO_LU`: Whether to exclude certain land uses from specific areas

### Economic Parameters
- `DYNAMIC_PRICE`: Enable demand elasticity-based dynamic pricing (default: True)
- `CARBON_PRICES_FIELD`: Carbon pricing scenario ('Default', 'CONSTANT', etc.)
- `AMORTISE_UPFRONT_COSTS`: Whether to amortize establishment costs (default: False)
- `DISCOUNT_RATE`: Discount rate for economic calculations (default: 7%)
- `AMORTISATION_PERIOD`: Period for cost amortization in years (default: 30)
- `TRANSITION_COST_MULT`: Scenario multiplier on land-use transition costs (1 = baseline, <1 cheaper switching, >1 higher barrier)
- `TECH_ADOPT_MULT`: Scenario multiplier on technical adoption ceilings (Asparagopsis, Precision Ag, AgTech EI, Biochar)

### Transition Flow Model
Transitions are modelled as explicit per-source delta flows: each cell's land is tracked back to the land uses it came from, and only the land that actually moves is charged. The model is exact: every nonzero land-use fraction of a cell in the base year (above the `ROUND_DECIMALS` noise floor) is its own source with its own flow variables, for agricultural and non-agricultural land uses alike. There is no dial that merges small fractions into a cell's dominant land use.

### Solver Configuration
- `THREADS`: Number of parallel threads for optimization (default: 32)
- `RETRY_PARAMS`: Ordered list of solve attempts, each a `(NumericFocus, Method, Crossover, Presolve, BarHomogeneous)` tuple. The algorithm is chosen here — there is no standalone `SOLVE_METHOD` setting. The default is one attempt, `(0, 2, -1, -1, 1)`: barrier with crossover and presolve automatic and the homogeneous algorithm ON (2026-11-18; the plain barrier failed on the grouped RES1 2025 model). There is no simplex fallback: on a model this size it runs for hours
- `SCALE_FLAG`: Gurobi's own scaling of rows and columns (default 2, geometric). It works on top of the row rescale (`RESCALE_FACTOR`); with cell grouping a column sums its blob's cells, so the columns differ by the blob's size, and geometric scaling cut the barrier iterations by 35 % on the grouped RES1 model
- `CELL_GROUPING`: Cells that are alike share one decision, so the solver sees a much smaller model (default `False`). At RES1 about 4.2 M cells become about 250 k blobs and the solved model has about 9× fewer nonzeros, about the size of an ungrouped RES5 model. A blob is cells with the same starting land use and options, the same regions and GBF target footprints, and close coefficients (an SVD of the objective and the wide rows: `CELL_GROUPING_SVD_DIMS`, `CELL_GROUPING_TOLERANCE`; `CELL_GROUPING_RIPARIAN_BINS`). The solution is mapped back to every cell and checked against the full model each year (`out_<year>/grouping_<year>.csv`, `grouping_rows_<year>.csv`): it is always feasible there, and the cost is optimality (within 0.12 % of the ungrouped run at RES5, 2020–2050). It shortens the solve only — the full tables are still built — needs crossover, and cannot be combined with `BIO_ROW_GENERATION` or `ELASTIC_FAMILIES`; with GBF8 targets most of the speed-up is lost. Design: `luto/solvers/README.md`
- `FEASIBILITY_TOLERANCE` / `OPTIMALITY_TOLERANCE` / `BARRIER_CONVERGENCE_TOLERANCE`: Solver tolerances. `ROUND_DECIMALS` and the near-zero bound snapping threshold are derived from `FEASIBILITY_TOLERANCE`
- `RESCALE_FACTOR`: Target magnitude (1e3) that solver input arrays are rescaled to for numerical stability
- No coefficient floor: `SOLVER_COEFF_MIN` (1e-4) was retired on 2026-11-23; every coefficient reaches Gurobi as built (rows rescaled, nothing dropped)
- `BOUND_PROP_REL_TOL`: Relative margin (default 1e-6) of the pre-solve bound check (see [Diagnosing Infeasible Years](#diagnosing-infeasible-years)). A row is called impossible or redundant only beyond max(10 × `FEASIBILITY_TOLERANCE`, `BOUND_PROP_REL_TOL` × the magnitude of the row's terms); a borderline row is kept
- `BOUND_PROP_DROP_FAMILIES`: Constraint families whose provably redundant rows are dropped before the model is built (default `[]`; e.g. `['GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8']`). Exact — the solution does not change — and every row is still reported
- `BOUND_PROP_ON_IMPOSSIBLE`: `'stop'` (default) ends the run at a year with a provably impossible row, before its model is built; `'solve'` reports the row and solves anyway
- `VERBOSE`: Control solver output verbosity

### Output Control
- `RESFACTOR`: Spatial resolution factor (1 = full resolution, >1 = coarser)
- `WRITE_OUTPUTS`: Whether to write outputs at the end of a run (set False for quick tests and infeasibility debugging)
- `WRITE_REPORT_MAX_MEM_MB`: Memory budget (MB) for report generation. Parallel-write worker counts are derived from this budget and each task's measured peak
- `WRITE_CHUNK_SIZE`: Chunk size used while writing NetCDF outputs
- `WRITE_GBF3_NVIS` / `WRITE_GBF4_SNES` / `WRITE_GBF4_ECNES`: Per-layer toggles for the expensive biodiversity outputs ('on' or 'off'). SNES alone adds roughly five hours to the write phase

Refer to `luto/settings.py` for a complete list of configurable parameters and detailed descriptions.

## Data Formats

### Carbon Sequestration Data (NetCDF)

LUTO2 uses NetCDF format with xarray for carbon sequestration data, replacing the previous HDF5/pandas format. This provides better performance, compression, and flexibility.

**Key Features:**
- **Format**: NetCDF (.nc) files with dimensions: `age` × `cell`
- **Available ages**: 50, 60, 70, 80, 90 years (selected from full carbon accumulation timeseries)
- **Components**: Trees (aboveground biomass), Debris (litter), Soil (belowground)
- **Compression**: zlib level 5 with chunking for efficient storage and loading
- **File naming**: `tCO2_ha_{type}.nc` (e.g., `tCO2_ha_ep_block.nc` for Environmental Plantings Block)

**Planting Types:**
- Environmental Plantings: Block, Belt, Riparian (ep_block, ep_belt, ep_rip)
- Carbon Plantings: Block, Belt (cp_block, cp_belt)
- Human-Induced Regeneration: Block, Riparian (hir_block, hir_rip)

**Carbon Calculation:**
The model loads NetCDF data at the age specified by `CARBON_EFFECTS_WINDOW` setting:
- Aboveground carbon (Trees + Debris) is discounted by fire risk and reversal risk
- Belowground carbon (Soil) is not risk-discounted
- Total sequestration is averaged over the carbon effects window to get annual rate

**Example:**
```python
settings.CARBON_EFFECTS_WINDOW = 50  # Use 50-year carbon accumulation data
# Model will load NetCDF data at age=50 and average to get annual sequestration rate
```

For more technical details, see `docs/CLAUDE_OUTPUT.md`.

## Copyright
Copyright 2024-now **Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.**  

Copyright 2021-2023 **Fjalar J. de Haan and Brett A. Bryan, Deakin University.** (see `CITATION.cff`).

## License
LUTO2 is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the **Free Software Foundation**, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **without any warranty**; without even the implied warranty of **merchantability** or **fitness for a particular purpose**. See the **GNU General Public License** for more details.

You should have received a copy of the **GNU General Public License** along with this program. If not, see <https://www.gnu.org/licenses/>.

## Citation
> Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R. (2025). The Land-Use Trade-Offs Model Version 2 (LUTO2): an integrated land system model for Australia. Software Collection. https://github.com/land-use-trade-offs/luto-2.0

## Contributing

We welcome contributions to LUTO2! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed modifications.

## Acknowledgments

LUTO2 was developed through a collaboration between:
- **Deakin University** - Centre for Integrative Ecology
- **Climateworks Centre** - Land Use Futures program
- **CSIRO** - Research contributions

This work is supported by funding from various Australian research councils and industry partners. We acknowledge the traditional custodians of the lands on which this research was conducted.

