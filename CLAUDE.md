# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LUTO2 is the Land-Use Trade-Offs Model Version 2, an integrated land systems optimization model for Australia. It simulates optimal spatial arrangement of land use and management decisions to achieve climate and biodiversity targets while maintaining economic productivity. The model uses GUROBI optimization solver and processes large spatial datasets.

## Documentation Structure

The LUTO2 documentation is split into themed files for better memory efficiency. **Read the relevant documentation file based on your current task**:

### 📁 [docs/CLAUDE_SETUP.md](docs/CLAUDE_SETUP.md)

**Read this when working on:**

- Environment setup and dependencies
- Running tests or simulations
- Configuring model parameters (settings.py)
- Setting up GUROBI license
- Performance optimization and memory management
- Memory profiling with `@trace_mem_usage` decorator
- **xr.dot() optimization** (CRITICAL: use `xr.dot()` instead of broadcasting for memory efficiency)

### 📁 [docs/CLAUDE_ARCHITECTURE.md](docs/CLAUDE_ARCHITECTURE.md)

**Read this when working on:**

- Core simulation engine (simulation.py, data.py)
- Economic modules (agricultural, non-agricultural, off-land)
- Solver integration (GUROBI, optimization)
- Biodiversity calculations (GBF framework)
- Data flow and preprocessing (dataprep.py)
- Dynamic pricing and demand elasticity

### 📁 [docs/CLAUDE_GBF2.md](docs/CLAUDE_GBF2.md)

**Read this when working on:**

- GBF2 priority degraded areas logic (mask construction, Zonation performance curve)
- `BIO_GBF2_BASE_YR`, `BIO_GBF2_MASK`, savanna-burning LDS correction
- `get_GBF2_target_for_yr_cal()` — baseline / base-year / restoration-fraction interpolation
- `_add_GBF2_constraints()` in solver.py — how the hard/soft constraint is built
- `write_biodiversity_GBF2_scores()` in write.py — denominator, ag/non-ag/am numerators, `Relative_Contribution_Percentage` formula and why it sums to ~30%
- Settings: `GBF2_TARGET`, `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT`, `BIO_CONTRIBUTION_LDS`, `GBF2_CONSTRAINT_TYPE`

### 📁 [docs/CLAUDE_OUTPUT.md](docs/CLAUDE_OUTPUT.md)

**Read this when working on:**

- NetCDF output format and structure
- Mosaic layer generation (write.py)
- **Valid layers implementation pattern** (memory/disk optimization)
- save2nc() optimization
- create_report_layers.py workflow
- Carbon sequestration data format
- Data transformation pipeline (1D→2D→EPSG:3857→RGBA→base64)
- Dimension hierarchies (Ag, Am, NonAg, GHG, Economics)

### 📁 [docs/CLAUDE_VUE_REPORTING.md](docs/CLAUDE_VUE_REPORTING.md)

**Read this when working on:**

- Vue.js 3 reporting interface
- Progressive selection pattern
- Cascade watcher implementation
- Data hierarchies for all modules (Area, Economics, GHG, Production, Water, Biodiversity, DVAR)
- Chart vs Map data structures
- Special cases (Economics dual map-types, GHG Ag Source level, Biodiversity multi-metric, Water Am chart series-by-AgMgt)
- File structure (views, data, services, routes)

### 📁 [docs/CLAUDE_SKILL/](docs/CLAUDE_SKILL/)

**Step-by-step skill guides for common tasks:**

- [adding_sum_tab.md](docs/CLAUDE_SKILL/adding_sum_tab.md): Adding a "Sum" tab (Ag + Am + NonAg) — covers write.py, report data/layers, Vue services, and view wiring
- [debug_species_infeasibility.md](docs/CLAUDE_SKILL/debug_species_infeasibility.md): Debug SNES/ECNES species infeasibility — build MPS from checkpoint, submit per-species maximisation jobs, identify which species targets are physically unachievable
- [debug_iis_from_zip.md](docs/CLAUDE_SKILL/debug_iis_from_zip.md): Debug IIS from Run_Archive.zip — extract MPS + lz4, compute IIS with Gurobi, analyze via PBS jobs
- [fakedata_inspection.md](docs/CLAUDE_SKILL/fakedata_inspection.md): Use `fakedata` as a lightweight `data.py` substitute for inspecting arrays and prototyping spatial helpers without loading full simulation inputs
- [task_run_plots.md](docs/CLAUDE_SKILL/task_run_plots.md): Extract data from task run Report_Data zips and build interactive ECharts HTML plots for sensitivity/grid search analysis
- [make_run_index_html.md](docs/CLAUDE_SKILL/make_run_index_html.md): Generate a self-contained interactive `index.html` for any task run directory — reads GEP params from `merged_grid_search_parameters_unique.csv`, full settings from `merged_grid_search_template.csv`, and model status from PBS stdout logs
- [iis_to_story.md](docs/CLAUDE_SKILL/iis_to_story.md): Translate IIS analysis summaries, ILP files, and PBS stdout logs into plain-language story tables grouped by scenario — diagnoses numerical stagnation vs rounding artefacts vs structural infeasibility
- [redo_failed_write.md](docs/CLAUDE_SKILL/redo_failed_write.md): Re-run write_outputs for runs that completed simulation but failed during write — copy fixed source files into each run's luto/ dir, then submit PBS jobs via submit_redo_write.py
- [create_task_runs.md](docs/CLAUDE_SKILL/create_task_runs.md): Create and submit multi-scenario task runs — write create_tasks.py with lean BASE_GRID (intentional overrides only), RUN_OVERRIDES, generate CSVs, and submit to cluster
- [patch_existing_renewable_capacity.md](docs/CLAUDE_SKILL/patch_existing_renewable_capacity.md): Inject real-world existing renewable capacity as `lu='Existing Capacity'` into output xarrays before `add_all` — covers write_dvar_and_mosaic_map, write_dvar_area, write_economics, and write_renewable_production
- [submit_task_runs_windows.md](docs/CLAUDE_SKILL/submit_task_runs_windows.md): Launch LUTO2 task runs locally on Windows via `run_all.py` — concurrency control, log monitoring, result verification, and failure recovery
- [retry_task_runs.md](docs/CLAUDE_SKILL/retry_task_runs.md): Retry runs with non-optimal solver status (false-INFEASIBLE, NUMERIC) — unzip archives, patch RETRY_PARAMS, resubmit via run_all.py; covers when checkpoint-based retry fixes the infeasible year vs when a full re-run is needed

## Diagnostic Tools (`luto/tests/`)

- **`find_infeasible_ecnes.py`**: Diagnoses infeasible GBF4 ECNES biodiversity constraints in a saved Gurobi MPS model. Removes all ECNES constraints, then tests each individually (in parallel via joblib) by maximizing its LHS to check if the target is achievable. Usage: `from luto.tests.find_infeasible_ecnes import find_infeasible_ecnes; base_model, results = find_infeasible_ecnes("path/to/model.mps")`

## Quick Reference

### Common Development Commands

```bash
# Testing
python -m pytest

# Run simulation
python -c "import luto.simulation as sim; data = sim.load_data(); sim.run(data=data)"

# Batch processing
python luto/tools/create_task_runs/create_grid_search_tasks.py
```

## Architecture Overview

### Core Modules

- **`luto/simulation.py`**: Main simulation engine and state management singleton
- **`luto/data.py`**: Core data management, loading, and spatial data structures
- **`luto/settings.py`**: Configuration parameters for all model aspects
- **`luto/solvers/`**: Optimization solver interface and input data preparation
  - `solver.py`: GUROBI solver wrapper (LutoSolver class)
    - Biodiversity constraint methods: `_add_GBF2_constraints()`, `_add_GBF3_NVIS_constraints()`, `_add_GBF4_SNES_constraints()`, `_add_GBF4_ECNES_constraints()`, `_add_GBF8_constraints()`. IBRA bioregion targets have **no separate constraint method** — they run through `_add_GBF3_NVIS_constraints()` when `GBF3_NVIS_REGION_MODE = 'IBRA_REG'`.
    - Renewable energy constraint method: `_add_renewable_energy_constraints()` — enforces state-level solar and wind generation targets
  - `col_builder.py`: the COLUMN side — `get_cols(data, base_year)` builds the column space (every unknown of the step as a labelled `xr.Dataset` per block, from the base-year state alone: exclude matrix, transition bounds, feasibility, sources, arc lists)
  - `row_builder.py`: the ROW side — `get_rows(data, base_year, target_year, space)` prepares the coefficient streams and targets (`RowInputs`) and one generator per constraint family returns its rows as a keyed `xr.Dataset` block (`FAMILIES`, in model order); `get_obj_block` is the objective
    - Biodiversity data attributes use `*_pre_1750_area_*` naming (e.g., `GBF3_NVIS_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`)
    - **No input rescaling** (2026-09-03): the coefficient streams reach the solver raw (float32). Every constraint block is row-rescaled in the solver by `row_builder.scale_rows` — per row, scale = geometric mean of max|row| and |RHS| over `RESCALE_FACTOR` (`calc_geomean_scale`), row and RHS divided by it, stage-4 floor on the scaled row — and the factor is kept on the solver (`demand_scales`, `water_scales`, `ghg_scale`, `renewable_scales`, `bio_GBF2_scale`, `bio_*_scales`) for the post-solve breakdown and `tools.calc_shadow_price_*` (So = 1e6: the objective is raw AUD / 1e6). Row scaling is an exact LP transformation; the gate compares models in RESTORED space (rows × their factor).
    - `SOLVER_COEFF_MIN` (1e-4): Universal minimum coefficient threshold, applied by the builders as a two-step contract (`row_builder.compose_rows` for every constraint family, `row_builder.get_obj_block` + `_setup_objective` for the objective vector): (1) a coefficient `q = V[cell] · c` (float32) is dropped when `|q| < SOLVER_COEFF_MIN` (`row_builder.drop`); (2) after row rescaling the scaled coefficient is floored again (the objective is scaled `× (1/1e6)` before its floor). Every policy coefficient multiplies an ag / ag-mgt / non-ag column directly (the transition model is exact, there is no fold and no accounting layer). Chosen empirically: 1e-3 caused ~3% economic loss; 1e-4 retains meaningful small coefficients while keeping the matrix ratio at 1e8.

### Economic Modules

- **`luto/economics/agricultural/`**: Agricultural land use economics
  - Revenue, cost, quantity, water, biodiversity, GHG calculations
  - Transition costs between agricultural land uses
  - **Renewable energy** effects integrated across all economics modules (cost, revenue, quantity, water, biodiversity, transitions)
  - **Dynamic pricing** (`revenue.py`): Demand elasticity-based price adjustments
    - Calculates commodity price multipliers based on supply-demand dynamics
    - Uses elasticity coefficients and demand deltas from 2010 baseline
    - Applied to crops and livestock (beef, sheep, dairy) when `DYNAMIC_PRICE` enabled
  - **Biodiversity module** (`biodiversity.py`): GBF (Global Biodiversity Framework) calculations
    - `get_GBF2_MASK_area()`: Returns GBF2 priority degraded areas (mask × real area)
    - `get_GBF3_NVIS_matrices_vr()`: NVIS vegetation layer matrices for GBF3 (also serves IBRA layers, selected by `GBF3_NVIS_REGION_MODE`)
    - `get_GBF4_SNES_matrix_sr()`, `get_GBF4_ECNES_matrix_sr()`: Species/Ecological Community NES matrices
    - Variable naming convention: `*_pre_1750_area_*` for baseline biodiversity area matrices
- **`luto/economics/non_agricultural/`**: Non-agricultural land use economics
  - Environmental plantings, riparian plantings, agroforestry, carbon plantings, BECCS, destocked natural land
- **`luto/economics/off_land_commodity/`**: Off-land commodity economics

### Data Processing

- **`luto/dataprep.py`**: Data preprocessing utilities
  - **Carbon sequestration data**: Migrated from HDF5/pandas to NetCDF/xarray format
  - Saves tree planting carbon data at specific ages (50, 60, 70, 80, 90 years)
  - Uses compressed NetCDF encoding with chunking for efficient storage
  - Format: `tCO2_ha_{type}.nc` where type is ep_block, ep_belt, ep_rip, cp_block, cp_belt, hir_block, hir_rip
- **`luto/tools/spatializers.py`**: Spatial data processing and upsampling
- **`luto/tools/write.py`**: Output writing and file generation

### Utilities

- **`luto/tools/create_task_runs/`**: Batch processing and grid search utilities
- **`luto/tools/report/`**: Report generation and visualization
  - `data_tools/`: Data processing for reports
  - `map_tools/`: Spatial visualization utilities
- **`luto/helpers.py`**: General utility functions

## Key Configuration Parameters

### Core Settings (`luto/settings.py`)

- `VERSION`: Model version identifier (current: '2.3')
- `SSP`: Shared Socioeconomic Pathway code (e.g., '245' for SSP2-RCP4.5)
- `SCENARIO`: Auto-derived from SSP (e.g., 'SSP2')
- `RCP`: Auto-derived from SSP (e.g., 'rcp4p5')
- `SIM_YEARS`: Simulation time periods (default: 2010-2050 in 5-year steps; 2010 is base year)
- `RESFACTOR`: Spatial resolution factor (1 = full resolution, >1 = coarser)
- `OBJECTIVE`: Optimization objective ('maxprofit' or 'mincost')

### Scenario Settings

- `DIET_DOM`: Domestic diet option ('BAU', 'FLX', 'VEG', 'VGN')
- `DIET_GLOB`: Global diet option (varies by year)
- `CONVERGENCE`: Dietary transformation target year (2050 or 2100)
- `IMPORT_TREND`: Import trend assumption ('Static' or 'Trend')
- `WASTE`: Waste multiplier (1 or 0.5)
- `FEED_EFFICIENCY`: Livestock feed efficiency ('BAU' or 'High')
- `APPLY_DEMAND_MULTIPLIERS`: Enable demand scenario effects (True/False)
- `AG_YIELD_MULT`: Agricultural yield multiplier (default: 1.15 = 15% increase)
- `CO2_FERT`: CO2 fertilization effects ('on' or 'off')

### Economic Settings

- `DYNAMIC_PRICE`: Enable demand elasticity-based dynamic pricing (default: False)
- `AMORTISE_UPFRONT_COSTS`: Whether to amortize establishment costs (default: False)
- `DISCOUNT_RATE`: Discount rate for economic calculations (default: 0.07)
- `AMORTISATION_PERIOD`: Period for cost amortization in years (default: 30)

### Environmental Constraints

- `GHG_EMISSIONS_LIMITS`: Greenhouse gas targets ('off', 'low', 'medium', 'high')
- `GHG_CONSTRAINT_TYPE`: Hard or soft GHG constraint ('hard' or 'soft')
- `WATER_LIMITS`: Water yield constraints ('on' or 'off')
- `WATER_CONSTRAINT_TYPE`: Hard or soft water constraint ('hard' or 'soft')
- `CARBON_EFFECTS_WINDOW`: Years for carbon accumulation averaging (50, 60, 70, 80, or 90 years)
  - Must match available NetCDF data ages in input files
  - Determines annual sequestration rate by averaging total CO2 over this period
  - Default: 50 years (follows S-curve logic with rapid early accumulation)
- `BIODIVERSITY_TARGET_GBF_*`: Global Biodiversity Framework targets
  - `GBF2_TARGET`: Priority degraded areas restoration ('off', 'low', 'medium', 'high')
  - `GBF2_CONSTRAINT_TYPE`: Hard or soft GBF2 constraint ('hard' or 'soft')
  - `GBF3_NVIS_TARGET`: NVIS vegetation group targets ('off', 'medium', 'high', 'CSV_DEFINED')
  - `GBF3_NVIS_REGION_MODE`: 'AUSTRALIA', 'NRM', or 'IBRA_REG' (IBRA bioregion targets are handled through the NVIS stream — there is no separate `BIODIVERSITY_TARGET_GBF_3_IBRA` setting or IBRA constraint method)
  - `GBF4_TARGET_SNES`: Species NES targets ('off', 'medium', 'high', 'SPECIFIED', or 'CSV_DEFINED'; levels apply uniform presets from `GBF4_SNES_TARGETS_DICT` to ALL species, GBF2-style; 'CSV_DEFINED' keeps CSV targets and filters to species with TARGET_LEVEL_2030 > 0; **'SPECIFIED'** = the same species as CSV_DEFINED with **region-specific** uniform levels — `GBF4_SNES_SEL_REGION_TARGETS` is then a dict `{region: {year: pct}}` whose keys select the regions (`{'AUSTRALIA': {...}}` at national scope); a plain list in the other modes)
  - `GBF4_TARGET_ECNES`: Ecological Community NES targets (same semantics with `GBF4_ECNES_TARGETS_DICT` / `GBF4_ECNES_SEL_REGION_TARGETS`)
  - `GBF3_NVIS_TARGET` also accepts 'SPECIFIED' with `GBF3_NVIS_SEL_REGION_TARGETS` as `{region: {year: pct}}`
  - `GBF4_SNES_MIN_AREA_HA` / `GBF4_ECNES_MIN_AREA_HA` / `GBF3_NVIS_MIN_AREA_HA` (100): every (region, item) whose `IN_LUTO_HA` is below the threshold is dropped in `data.py` (LHS ≈ 0 → structurally infeasible). These replaced the hand-written `GBF4_SNES_EXCLUDE_REGION_SPECIES` / `GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES` / `GBF3_NVIS_EXCLUDE_REGION_GROUPS` lists (removed 2026-08-26)
  - `GBF8_TARGET`: Species conservation targets ('off', 'medium', 'high', or 'USER_DEFINED'; levels apply uniform presets from `GBF8_TARGETS_DICT` to ALL ~10.6k species; 'USER_DEFINED' = former 'on', reads hand-filled USER_DEFINED_TARGET_PERCENT_* CSV columns)

### Renewable Energy Settings

- `RENEWABLES_OPTIONS`: Dict controlling which renewable energy types are enabled, e.g. `{'Utility Solar PV': True, 'Onshore Wind': True}`. Set values to `False` to disable individual types. Also drives the corresponding `AG_MANAGEMENTS` entries.
- `RENEWABLE_TARGET_SCENARIO_TARGETS`: Generation target scenario (one of: 'AEMO 2026 ISP - Accelerated Transition', 'AEMO 2026 ISP - Slower Growth', 'AEMO 2026 ISP - Step Change', 'Gladstone - BESS Sensitivity', 'Gladstone - Core')
- `RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS`: Spatial layer scenario (one of: 'step_change', 'accelerated_transition', 'ANU_transmission_T3', 'ANU_transmission_T5', 'ANU_transmission_T10')
- `RE_TARGET_LEVEL`: Spatial level for constraints ('STATE' or 'NRM'; only STATE currently supported)
- `INSTALL_CAPACITY_MW_HA`: Per-hectare capacity (MW/ha) per renewable type
- `EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS`: Exclude renewables from high-biodiversity GBF2 cells (default: True)
- `EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK`: Exclude renewables from EPBC MNES high-priority cells (default: True)
- `RENEWABLES_ADOPTION_LIMITS`: Maximum adoption fraction per type (default: 1.0 for both)
- Both renewable types are registered as non-reversible agricultural management options in `AG_MANAGEMENTS`
- Compatible land uses differ: Solar PV excludes Hay; Wind includes Hay and horticulture crops

### Solver Configuration

- `SOLVE_METHOD`: GUROBI algorithm (default: 2 for barrier method)
- `THREADS`: Parallel threads for optimization (default: min(32, cpu_count))
- `FEASIBILITY_TOLERANCE`: Primal feasibility tolerance (1e-6). Also the precision granule: `ROUND_DECIMALS` is derived from it, and the near-zero bound snap threshold is `FEASIBILITY_TOLERANCE * 10`
- `OPTIMALITY_TOLERANCE`: Optimality tolerance (default: 1e-2)
- `BARRIER_CONVERGENCE_TOLERANCE`: Barrier method convergence (default: 1e-5)
- `RESCALE_FACTOR`: Target magnitude of the per-row rescale (default: 1e3) — `row_builder.scale_rows` lands max|row| and |RHS| symmetrically around it
- `SOLVER_COEFF_MIN`: Universal minimum coefficient threshold (default: 1e-4). Applied by the builders (`row_builder.compose_rows` for every constraint family, `get_obj_block` + `_setup_objective` for the objective vector) as a two-step contract: coefficient `q = V[cell] · c` dropped when `|q| < SOLVER_COEFF_MIN` (`row_builder.drop`), scaled coefficient floored again after row rescaling. Applies to Economy, Biodiversity-quality, GHG, Water, Renewable, GBF2/3/4/8, Demand/Quantity, and Regional Adoption limits. Chosen empirically: 1e-3 caused ~3% economic loss; 1e-4 retains meaningful small coefficients while keeping the matrix range ratio at 1e8 (well within Gurobi's safe zone). `RESCALE_ZERO_THRESHOLD` was removed — post-rescale zeroing is superseded by this universal filter.

### Output Writing Configuration

- `WRITE_REPORT_MAX_MEM_GB`: Max memory for report generation (default: 64)
- `WRITE_CHUNK_SIZE`: Chunk size for NetCDF writing (default: 4096)

## Data Flow

1. **Data Loading**: `luto.data.Data` class loads spatial datasets from `/input/`
   - Loads demand scenarios and elasticity coefficients for dynamic pricing
   - Calculates demand deltas (change from 2010 baseline) for price adjustments
   - **Carbon data**: Loads NetCDF files using xarray, selects data at `CARBON_EFFECTS_WINDOW` age
   - Carbon sequestration components: Trees + Debris (aboveground, risk-discounted) + Soil (belowground)
   - **Renewable energy data**: Loads targets (CSV), electricity prices (separate CSV per type: solar, wind), spatial layers (NetCDF), and bundle parameters (CSV)
2. **Preprocessing**: `dataprep.py` processes raw data into model-ready formats
   - Copies demand elasticity data from source to input directory
   - **Carbon data preparation**: Converts 3D timeseries to NetCDF format with age dimension
   - Selects specific ages (50, 60, 70, 80, 90 years) for carbon accumulation data
   - Applies chunked compression (zlib level 5) for efficient storage
3. **Economic Calculations**: Economics modules calculate costs, revenues, transitions, biodiversity impacts
   - Revenue calculations apply demand elasticity multipliers when `DYNAMIC_PRICE` enabled
   - Elasticity multipliers computed as: `1 + (demand_delta / demand_elasticity)`
4. **Solver Input**: `solvers/col_builder.get_cols` (the unknowns) and `solvers/row_builder.get_rows` (the coefficient streams and targets) prepare the model data; the solver takes `(space, rows)`
   - Biodiversity matrices: GBF2 mask areas, GBF3 NVIS layers (NVIS or IBRA, per `GBF3_NVIS_REGION_MODE`), GBF4 SNES/ECNES matrices, GBF8 species data
   - Renewable energy: Solar/wind yield arrays (`renewable_solar_r`, `renewable_wind_r`), state region mapping, raw targets
   - No input rescaling: the streams reach the solver raw; every constraint block is row-rescaled by `row_builder.scale_rows` (factor kept on the solver) and the objective is raw AUD / 1e6. Every constraint family is composed by `row_builder.compose_rows` and the objective by `row_builder.get_obj_block`; both drop `|q| < SOLVER_COEFF_MIN` (`row_builder.drop`) and floor the scaled coefficient.
   - **Row blocks** (`row_builder.FAMILIES`): every constraint family is a generator `family(rows, space) -> xr.Dataset | None` returning its rows as ONE keyed block — a MultiIndex coordinate `row` (the family's natural key: `(region, item)`, `(cell,)`, `(am, lu, lm, cell)`, …), per-row `rhs`, `sense`, `name`, `scale`, and `attrs['A']` (the CSR over Var.index, row-scaled where `scale` != 1). `LutoSolver.formulate` creates ONE `addMVar` over the column table (the blocks as slices of it), then loops the families: one `addMConstr` per block, names, the Gurobi handles attached as `constr`, and the flat handle / scale attributes the shadow-price readers walk are published from the blocks (`_publish`). `solve()` reads the solution back through the space and the blocks.
   - **The column space** (`luto/solvers/col_builder.py`): every unknown of the model as ONE row of a long table, `cols['table']` — an `xr.Dataset` on `col` = Var.index, the rows in block order `BLOCKS` (`ag | nonag | am | ag2ag | ag2nonag | nonag2ag | cell_usage`, `attrs['block_ptr']` = the group bounds of the table sorted by block, `block_slice(table, name)` = one block's rows) — with the fields of each column, −1 where a field does not apply: `block`, `m, j` (the ag (lm, lu) the column lands on: own for ag, host for am, the TO fields of ag2ag / nonag2ag), `k` (own for nonag, the TO field of ag2nonag), `slot`, `am_idx`, `j_idx` (the (am, lu) slot of an am column, its option as an index into `attrs['options']`, its land use's position within the option), `from_m, from_j, from_k, local_r` (where an arc comes from), `cell` (every column), `lb` / `ub` (float64) and `base` (float32, the node-balance constant). Attrs: `n_terms` (the scored columns: the first three blocks, so a coefficient array indexed by column covers them), `n_dec` (the decision columns; the objective's width), `n_all` (every column; the rows' width), `by_cell_order` / `by_cell_ptr` (the scored rows re-sorted by cell, for the sparse-support gather in `compose_rows`) and `src_ptr` (per arc block, the group bounds of its rows sorted by source, as table rows). The am block reads from the table alone (`row_builder.am_runs` groups it by option or slot). Beside the table the space keeps the wide id grids the rows still index by (m, j, r) / (k, r) / (slot, m, r) — `cols['ag']` (lm, lu, cell; `ub` / `base`), `cols['nonag']` (nonag_lu, cell; `lb` / `ub` / `base`, every land use enabled or not: the node-balance rows need the disabled ones), `cols['am']` (slot = MultiIndex (am, lu), lm, cell; `lb`, per-slot `j` / `am_idx` / `j_idx`, attrs `agman2lu` / `savanna_eligible_r`) — plus `cols['sources']` and the four `mask_*` arrays. The module is ordered data-first: the `get_*` getters (sources, transition bounds, feasibility, masks), then the block builders `ag_space / nonag_space / am_space / ag2ag_space / ag2nonag_space / nonag2ag_space / cell_usage_space` (block-local ids; the arc builders return long tables with `src_ptr`), then `table_space` (the blocks' rows back to back) and `get_cols` — one function in numbered sections that calls the getters, builds the blocks, places them back to back (each at the next free Var.index) and returns the table with the grids. `data.AGMAN2LU` ({option: [land-use codes]}, a `Data` property) fixes the ag-management slot order. The row families read the table: the policy families gather one float32 coefficient per scored column (`row_builder.gather_coeffs` / `gather_bio_coeffs`, the am block per option or slot run via `am_runs`) and multiply it by a weighting row over cells (`compose_rows`); the objective (`get_obj_block`) is the same gather plus the transition costs per source run of the arc blocks; the flow rows are group-bys over the table — source cap groups the arc rows by (from fields, cell), node balance joins every X column and every arc (TO fields and FROM fields) on its node (m, j, cell) / (k, cell). The solver creates ONE `addMVar` over the table (`lb` / `ub` per row, the block views as slices, the names from the fields) and reads the solution back through it. **The transition model is exact** (2026-09-08): every nonzero (lm, lu) fraction of a cell in the base year above the `ROUND_DECIMALS` noise floor is its own source (`data.ag_dvars[base_year]` read as it is; `transitions.get_base_dvar_mj_cell_map`); the former θ fold (`EXACT_REACHABILITY_MIN_FRACTION`, the fold block, the `acct_link_*` rows) is gone — a census showed it touched ≤ 0.02 % of the entries per year at RES5 (`docs/FINDINGS.md`, 20260908).
5. **Optimization**: `solvers/solver.py` runs GUROBI optimization with biodiversity and renewable energy constraints
6. **Output Generation**: `tools/write.py` writes results to `/output/`
   - Biodiversity outputs: GBF2/3/4/8 scores, species impacts, vegetation group restoration

## Output Structure

Results saved in `/output/<timestamp>/`:

- `DATA_REPORT/REPORT_HTML/index.html`: Interactive dashboard
- NetCDF files: Spatial outputs (xarray format)
- CSV files: Data tables
- Logs: Execution logs and metrics

## Important Conventions

### Memory Optimization: xr.dot() Pattern (CRITICAL)

**ALWAYS use `xr.dot()` instead of broadcasting for array operations:**

```python
import xarray as xr
# Load NetCDF and select specific age from CARBON_EFFECTS_WINDOW setting
ds = xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_block.nc"))
ds = ds.sel(age=settings.CARBON_EFFECTS_WINDOW, cell=self.MASK)

# Calculate total sequestration with risk discounting
total_co2 = (
    (ds.EP_BLOCK_TREES_TOT_T_CO2_HA + ds.EP_BLOCK_DEBRIS_TOT_T_CO2_HA)
    * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)  # Aboveground with risk discount
    + ds.EP_BLOCK_SOIL_TOT_T_CO2_HA  # Belowground (no risk discount)
).values / settings.CARBON_EFFECTS_WINDOW  # Average over window
```

### Risk Discounting

- **Aboveground carbon** (Trees + Debris): Discounted by fire risk and reversal risk
- **Belowground carbon** (Soil): No risk discounting applied
- Formula: `(AG_carbon × fire_risk% × (1 - RISK_OF_REVERSAL)) + BG_carbon`

### Migration Notes

- **Old format**: HDF5 files with pandas DataFrames, separate AG/BG columns
- **New format**: NetCDF files with xarray Datasets, separate component variables
- **Advantages**: Better compression, faster subsetting, age dimension flexibility, xarray integration
- **CARBON_EFFECTS_WINDOW**: Must be one of [50, 60, 70, 80, 90] to match available data ages

## Biodiversity Module Naming Conventions

The biodiversity module follows consistent naming conventions for GBF (Global Biodiversity Framework) variables:

### Variable Naming Pattern

- **Pre-1750 baseline areas**: Use `*_pre_1750_area_*` suffix
  - Examples: `GBF3_NVIS_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`, `GBF8_pre_1750_area_sr`
  - These represent baseline biodiversity area matrices before land use changes

### Function Naming Pattern

- **GBF constraint methods**: Use `_add_GBF{N}_{TYPE}_constraints()` format
  - Examples: `_add_GBF2_constraints()`, `_add_GBF3_NVIS_constraints()`, `_add_GBF4_SNES_constraints()`, `_add_GBF4_ECNES_constraints()`, `_add_GBF8_constraints()`
  - Maintain consistency between method names and GBF target types

### Data Structure Indices

- `v, r`: Vegetation group / bioregion (v) × cell (r) - used for GBF3 NVIS and IBRA data
- `s, r`: Species/community (s) × cell (r) - used for GBF4 and GBF8 data
- `r`: Cell only - used for GBF2 mask data

### Key GBF Modules

1. **GBF2**: Priority degraded areas restoration
   - Function: `get_GBF2_MASK_area(data)` returns mask × real area
2. **GBF3 NVIS / IBRA**: NVIS major vegetation group targets, or IBRA bioregion targets
   - Function: `get_GBF3_NVIS_matrices_vr(data)` returns the layers for both; `GBF3_NVIS_REGION_MODE` ('AUSTRALIA', 'NRM', or 'IBRA_REG') selects NVIS vs IBRA. There is no separate IBRA function, attribute, setting, or constraint method.
4. **GBF4**: Species and Ecological Community NES
   - SNES: `get_GBF4_SNES_matrix_sr(data)`
   - ECNES: `get_GBF4_ECNES_matrix_sr(data)`
5. **GBF8**: Species conservation
   - Function: `get_GBF8_species_matrices_sr(data, target_year)`

## Renewable Energy Module

The renewable energy module (REM) introduces solar and wind energy generation as agricultural management options, enabling optimization of land use to meet state-level renewable energy targets.

### Architecture

Renewable energy types (Utility Solar PV, Onshore Wind) are implemented as agricultural management options (`AG_MANAGEMENTS`). Each type has effects across all economics modules:

- **`quantity.py`**: `get_quantity_renewable(data, re_type, yr_idx)` — core yield calculation (MWh per cell). Yield = `MW_HA_HR × capacity% × (1 - distribution_loss%) × 8760 × REAL_AREA`
- **`revenue.py`**: `get_utility_solar_pv_effect_r_mrj()` / `get_onshore_wind_effect_r_mrj()` — agricultural revenue change + electricity revenue (quantity × state-level price)
- **`cost.py`**: `get_utility_solar_pv_effect_c_mrj()` / `get_onshore_wind_effect_c_mrj()` — O&M cost multiplier on base ag costs + operational costs from spatial layers
- **`transitions.py`**: `get_utility_solar_pv_effect_t_mrj()` / `get_onshore_wind_effect_t_mrj()` — upfront installation CAPEX (not amortized)
- **`biodiversity.py`**: `get_utility_solar_pv_effect_b_mrj()` / `get_onshore_wind_effect_b_mrj()` — biodiversity compatibility impacts from bundle data
- **`water.py`**: `get_utility_solar_pv_effect_w_mrj()` / `get_onshore_wind_effect_w_mrj()` — water requirement impacts

### Solver Constraints

`_add_renewable_energy_constraints()` in `solver.py` enforces state-level generation targets:

- Separate constraints for solar and wind per state
- Uses `renewable_solar_r` / `renewable_wind_r` yield arrays from `row_builder.get_rows`
- Targets from `RENEWABLE_TARGETS` CSV, filtered by year and scenario
- Row-rescaled in the solver like every other family (`row_builder.scale_rows`; factor per (type, state) row in `renewable_scales`)

### Data Loading (`data.py`)

- `RENEWABLE_TARGETS`: State-level generation targets (TWh → MWh) by year, scenario, and product
- `SOLAR_PRICES`: State-level solar electricity prices (AUD/MWh) by year
- `WIND_PRICES`: State-level wind electricity prices (AUD/MWh) by year
- `RENEWABLE_LAYERS`: NetCDF spatial layers with installation cost, operation cost, capacity %, and distribution loss %
- `RENEWABLE_BUNDLE_SOLAR` / `RENEWABLE_BUNDLE_WIND`: Parameters per land use (productivity, revenue, O&M multiplier, biodiversity compatibility, water requirements)
- `REGION_STATE_CODE` / `REGION_STATE_NAME2CODE`: State mapping for state-level constraint aggregation

### Input Files Required

| File | Format | Description |
|------|--------|-------------|
| `renewable_targets.csv` | CSV | Year, STATE, SCENARIO, PRODUCT, Renewable_Target_TWh |
| `renewable_price_AUD_MWh_solar.csv` | CSV | Year, State, Price_AUD_per_MWh (solar) |
| `renewable_price_AUD_MWh_wind.csv` | CSV | Year, State, Price_AUD_per_MWh (wind) |
| `renewable_energy_bundle.csv` | CSV | Year, Commodity, Lever, Productivity, Revenue, OM_Cost_Multiplier, Biodiversity_compatability, INPUT-wrt_water-required |
| `renewable_energy_layers_1D.nc` | NetCDF | Spatial layers: Cost_of_install_AUD_kw, Cost_of_operation_AUD_kw, capacity_factor_multiplier, distribution_loss_factor_multiplier |

### Compatible Land Uses

- **Utility Solar PV**: Unallocated - modified land, Beef/Sheep/Dairy - modified land, Summer/Winter cereals/legumes/oilseeds
- **Onshore Wind**: All Solar PV land uses + Hay, Cotton, Other non-cereal crops, Rice, Sugar, Vegetables

### Key Design Notes

- Both types are **non-reversible** once installed (`AG_MANAGEMENTS_REVERSIBLE = False`)
- Adoption limits enforced via existing `const_ag_mam_adoption_limit` solver constraints
- State-level pricing: electricity revenue uses state-specific prices mapped via `REGION_STATE_CODE`
- Effects follow standard pattern: `base_value × (multiplier - 1)` for additive impacts
- **GHG effects return zeros**: No direct on-farm GHG impact; displacement benefits handled externally via AusTIMES energy model
- **Per-row rescaling**: each (type, state) target row carries its own factor (`renewable_scales`); the yield arrays themselves are raw
- **ACT excluded**: Australian Capital Territory skipped in state-level constraints
- **`write_renewable_economics` deleted**: Superseded by xarray injection in `write_economics` (cost/revenue/profit) and `write_renewable_production` (MWh). Old function had a broken parallel task-list call and patched DataFrames post-hoc, causing `KeyError` in `xr.stack(...).sel(layer=valid_layers)`.
- **Existing capacity injection pattern**: Patch the result xarray of `dvar × mat` (after multiplication) with `lu='Existing Capacity'` **before** `add_all` — never patch the dvar arrays or the DataFrame. `lm='dry'` carries real values; `lm='irr'` is zeros to avoid double-counting. See skill: [patch_existing_renewable_capacity.md](docs/CLAUDE_SKILL/patch_existing_renewable_capacity.md).
- **`return_cells=True`**: `get_utility_solar_pv_existing_cost_by_region` and `get_onshore_wind_existing_cost_by_region` in `cost.py` accept `return_cells=True` to return per-cell `{'opex_r': DataArray[cell], 'capex_r': DataArray[cell]}` before any regional groupby — used by `write_economics` for xarray injection.

## Vue.js Reporting System Architecture

The LUTO reporting system uses Vue.js 3 with a progressive selection pattern for data visualization.

### Naming Patterns

- **Biodiversity variables**: `*_pre_1750_area_*` for baseline matrices
- **GBF functions**: `_add_GBF{N}_{TYPE}_constraints()`, `get_GBF{N}_*()`
- **Carbon files**: `tCO2_ha_{ep,cp,hir}_{block,belt,rip}.nc`

### NetCDF Dimensions

- **Ag**: `lm[ALL,dry,irr] → lu[ALL,...] → year → cell`
- **Am**: `am[ALL,...] → lm[ALL,dry,irr] → lu[ALL,...] → year → cell`
- **NonAg**: `lu[ALL,...] → year → cell`

### Map Layer Split-File Pattern

Map layers are split into per-combo files — **not** a single nested JS object. `create_report_layers.py` calls `_write_split_by_combo()` which writes:

- `<prefix>__index.js` → `window["<prefix>__index"] = { dims: [...], tree: { dim1val: [dim2vals...], ... } }`
- `<prefix>__<safe(d1)>__<safe(d2)>….js` → `window["..."] = { 2020: {leaf}, 2025: {leaf}, ... }`

**MapService entries** use `{ indexPath, indexName, layerPrefix }` — not `{ path, name }`. Exception: `mask` GeoJSON overlays keep `{ path, name }`.

Views load on demand via `createMapLayerLoader(VIEW_NAME)` from `helpers.js`:

- `ensureComboLayer(layerPrefix, [dim1, …])` — loads the combo file, releases the previous for GC
- `selectMapData = computed(() => currentLayerData.value?.[year] ?? {})` — same expression in every view

### JSON Output Hierarchies (Chart)

Chart JSON files end at `[series array]`. See [CLAUDE_VUE_REPORTING.md](docs/CLAUDE_VUE_REPORTING.md) for the full per-module table.

**Chart JSON (Time Series)** — ends at `[series array]`:

- **Ag**: `region → lm → lu` (standard); `region → lm → source → [series(name=LU)]` (GHG)
- **Am**: `region → lm → lu → [series(name=AgMgt)]`; source removed from Am in GHG/Water
- **NonAg**: `region → [series(name=LU)]`

**`source` dimension** appears only in **Ag** for GHG (emission type) and Economics (cost/revenue type). Am no longer has a source level.

**Valid Layers Pattern** — two approaches:

- **Economics** (revenue/cost/profit/transitions): `ALL` = dvar mosaic (categorical) — load dvar, filter, concat
- **GHG / Biodiversity / Water / Production**: `ALL` = sum aggregate — `xr.concat([data.sum('dim'), data], 'dim')` before stacking

**Greyscale ramp for unselected NRMs** — biodiversity write functions for GBF3 NVIS / GBF4 SNES / GBF4 ECNES attach an `is_selected` cell coord (boolean) to the **source** xarray (e.g. `vegetation_score_vr`, `bio_snes_sr`, `bio_ecnes_sr`) so it propagates automatically through all downstream arithmetic. Always attached — all-ones in Australia mode, NRM-union mask in NRM mode. The render-side `map2base64` maps unselected non-zero cells through palette codes 151-200 (grey ramp). CSV/AUS aggregation applies `.where(is_selected_da)` on the fly (no separate masked-variant arrays). GBF2 (mask-based) and GBF8 (climate-driven) are excluded — no spatial restriction to grey out.

See [CLAUDE_OUTPUT.md](docs/CLAUDE_OUTPUT.md) for detailed examples.

### Vue.js Progressive Selection Hierarchies

- **Standard Full**: Category → AgMgt → Water → Landuse
- **Biodiversity**: Metric → Category → AgMgt → Water → (Species) → Landuse
- **NonAg Simplified**: Category → Landuse
- **DVAR Simplified**: Category → Landuse/AgMgt → Year
- **Economics Extended**: Category → MapType → (AgMgt) → Water → (Source) → Landuse

## Getting Started

1. **New to the project?** Start with [CLAUDE_SETUP.md](docs/CLAUDE_SETUP.md) for environment setup
2. **Working on core model logic?** See [CLAUDE_ARCHITECTURE.md](docs/CLAUDE_ARCHITECTURE.md)
3. **Working on output generation?** See [CLAUDE_OUTPUT.md](docs/CLAUDE_OUTPUT.md)
4. **Working on the reporting UI?** See [CLAUDE_VUE_REPORTING.md](docs/CLAUDE_VUE_REPORTING.md)

**Remember**: Only read the documentation file relevant to your current task to minimize memory usage!
