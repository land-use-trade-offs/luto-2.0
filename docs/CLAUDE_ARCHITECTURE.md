# LUTO2 Architecture Overview

This document describes the core architecture, modules, and data flow of LUTO2.

## Core Modules

- **`luto/simulation.py`**: Main simulation engine and state management singleton
- **`luto/data.py`**: Core data management, loading, and spatial data structures
- **`luto/settings.py`**: Configuration parameters for all model aspects
- **`luto/solvers/`**: Optimization solver interface and input data preparation
  - `solver.py`: GUROBI solver wrapper (LutoSolver class)
    - Biodiversity constraint methods: `_add_GBF2_constraints()`, `_add_GBF3_NVIS_constraints()`, `_add_GBF4_SNES_constraints()`, `_add_GBF4_ECNES_constraints()`, `_add_GBF8_constraints()`. IBRA bioregion targets have **no separate constraint method** — they run through `_add_GBF3_NVIS_constraints()` when `GBF3_NVIS_REGION_MODE = 'IBRA_REG'`.
    - Renewable energy constraint method: `_add_renewable_energy_constraints()` — enforces state-level solar and wind generation targets
    - Hard/soft constraint flexibility: `GHG_CONSTRAINT_TYPE`, `WATER_CONSTRAINT_TYPE`, `GBF2_CONSTRAINT_TYPE`
    - Two-stream transition/accounting model (see "Theta-Fold Transition Model" below): `_setup_ag_accounting_vars()` re-expresses the folded decision vars for correct accounting.
  - `input_data.py`: Prepares optimization model input data
    - Biodiversity data attributes use `*_pre_1750_area_*` naming (e.g., `GBF3_NVIS_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`). IBRA reuses the NVIS attribute — there is no `GBF3_IBRA_pre_1750_area_vr`.
    - Renewable energy data: `renewable_solar_r`, `renewable_wind_r` yield arrays; `region_state_r` mapping
    - **No input rescaling** (2026-09-03): the coefficient streams reach the solver raw (float32). Every constraint block is row-rescaled in the solver by `row_builder.scale_rows` — per row, scale = geometric mean of max|row| and |RHS| over `RESCALE_FACTOR` (`calc_geomean_scale`), row and RHS divided by it, stage-4 floor on the scaled row — and the factor is kept on the solver (`demand_scales`, `water_scales`, `ghg_scale`, `renewable_scales`, `bio_GBF2_scale`, `bio_*_scales`) for the post-solve breakdown and `tools.calc_shadow_price_*` (So = 1e6: the objective is raw AUD / 1e6). Row scaling is an exact LP transformation; the gate compares models in RESTORED space (rows × their factor).
    - `SOLVER_COEFF_MIN` (1e-4): Universal minimum coefficient threshold, applied by the array-path builders as a four-stage contract (`row_builder.compose_row` for every constraint family, `_setup_objective` for the objective vector): (1) the per-cell coefficient is dropped when `|q| < SOLVER_COEFF_MIN`, tested BEFORE the fold weight; (2) kept terms get `q × w` in double; (3) duplicate variables (fold terms only) are merged with `sum_duplicates`; (4) the merged coefficient is floored again (the objective is scaled `× scale × (1/1e6)` after the merge, before its floor). Chosen empirically: 1e-3 caused ~3% economic loss; 1e-4 retains meaningful small coefficients while keeping the matrix ratio at 1e8.
    - No per-family scale factors: `input_data.scale_factors` was removed with the input rescaling; every factor is per constraint row, on the solver.

## Economic Modules

### Agricultural Economics (`luto/economics/agricultural/`)
- Revenue, cost, quantity, water, biodiversity, GHG, transitions calculations
- Each module has 10 agricultural management effect functions (one per AM type)
- **Renewable energy** effects integrated across all economics modules (cost, revenue, quantity, water, biodiversity, transitions, GHG)
  - `get_quantity_renewable(data, re_type, yr_idx)`: Core yield calculation (MWh per cell)
  - Revenue: electricity price × quantity + ag revenue change via productivity multiplier
  - Cost: O&M cost multiplier on base ag costs + operational costs from spatial layers
  - Transitions: upfront installation CAPEX (not amortized)
  - GHG: returns zeros (displacement handled externally via AusTIMES)
- **Dynamic pricing** (`revenue.py`): Demand elasticity-based price adjustments
  - Calculates commodity price multipliers based on supply-demand dynamics
  - Uses elasticity coefficients and demand deltas from 2010 baseline
  - Applied to crops and livestock (beef, sheep, dairy) when `DYNAMIC_PRICE` enabled
- **Biodiversity module** (`biodiversity.py`): GBF (Global Biodiversity Framework) calculations
  - `get_GBF2_MASK_area()`: Returns GBF2 priority degraded areas (mask × real area)
  - `get_GBF3_NVIS_matrices_vr()`: NVIS vegetation layer matrices for GBF3 (also serves IBRA layers, selected by `GBF3_NVIS_REGION_MODE`)
  - `get_GBF4_SNES_matrix_sr()`, `get_GBF4_ECNES_matrix_sr()`: Species/Ecological Community NES matrices
  - `get_GBF8_matrix_sr(data, target_year)`: Species conservation matrices
  - Variable naming convention: `*_pre_1750_area_*` for baseline biodiversity area matrices
- **Agricultural Management options** (10 types): Asparagopsis taxiformis, Precision Agriculture, Ecological Grazing, Savanna Burning, AgTech EI, Biochar, HIR-Beef, HIR-Sheep, Utility Solar PV, Onshore Wind

### Non-Agricultural Economics (`luto/economics/non_agricultural/`)
- Environmental plantings, riparian plantings, sheep/beef agroforestry, carbon plantings (block/belt), BECCS, destocked natural land
- Revenue, cost, quantity, water, biodiversity, GHG, transitions calculations
- No agricultural management options (simpler structure)

### Off-Land Commodity (`luto/economics/off_land_commodity/`)
- Off-land commodity economics (pork, chicken, eggs, aquaculture)

## Data Processing Modules

### Preprocessing (`luto/dataprep.py`)
- Data preprocessing utilities
- **Carbon sequestration data**: Migrated from HDF5/pandas to NetCDF/xarray format
- Saves tree planting carbon data at specific ages (50, 60, 70, 80, 90 years)
- Uses compressed NetCDF encoding with chunking for efficient storage
- Format: `tCO2_ha_{type}.nc` where type is ep_block, ep_belt, ep_rip, cp_block, cp_belt, hir_block, hir_rip

### Spatial Processing (`luto/tools/spatializers.py`)
- Spatial data processing and upsampling

### Output Writing (`luto/tools/write.py`)
- Outputs model results as xarray datasets in NetCDF format
- Uses hierarchical dimension structure for progressive selection in reporting UI
- See [CLAUDE_OUTPUT.md](CLAUDE_OUTPUT.md) for detailed NetCDF format documentation

### Report Generation (`luto/tools/report/`)
- `data_tools/`: Data processing utilities for report generation
  - `__init__.py`: Shared helper functions (array_to_base64, tuple_dict_to_nested, etc.)
  - `parameters.py`: Configuration parameters and name mappings
- `create_report_data.py`: Generates chart data JSON files for Vue.js dashboard
- `create_report_layers.py`: Converts NetCDF to map layer JSON files
  - **Function signature**: `save_report_layer(data_path: str)` - takes output path, not Data object
- `map_tools/`: Spatial visualization utilities
- See [CLAUDE_VUE_REPORTING.md](CLAUDE_VUE_REPORTING.md) for Vue.js system details

### Utilities (`luto/helpers.py`)
- General utility functions

### Batch Processing (`luto/tools/create_task_runs/`)
- Batch processing and grid search utilities

## Data Flow

1. **Data Loading**: `luto.data.Data` class loads spatial datasets from `/input/`
   - Loads demand scenarios and elasticity coefficients for dynamic pricing
   - Calculates demand deltas (change from 2010 baseline) for price adjustments
   - **Carbon data**: Loads NetCDF files using xarray, selects data at `CARBON_EFFECTS_WINDOW` age
   - Carbon sequestration components: Trees + Debris (aboveground, risk-discounted) + Soil (belowground)
   - **Renewable energy data**: Loads targets (CSV), electricity prices (separate CSVs: solar, wind), spatial layers (NetCDF), bundle parameters (CSV)
   - **Biodiversity data**: GBF2 masks, GBF3 NVIS/IBRA layers, GBF4 SNES/ECNES species data, GBF8 conservation data

2. **Preprocessing**: `dataprep.py` processes raw data into model-ready formats
   - Copies demand elasticity data from source to input directory
   - **Carbon data preparation**: Converts 3D timeseries to NetCDF format with age dimension
   - Selects specific ages (50, 60, 70, 80, 90 years) for carbon accumulation data
   - Applies chunked compression (zlib level 5) for efficient storage

3. **Economic Calculations**: Economics modules calculate costs, revenues, transitions, biodiversity impacts
   - Revenue calculations apply demand elasticity multipliers when `DYNAMIC_PRICE` enabled
   - Elasticity multipliers computed as: `1 + (demand_delta / demand_elasticity)`
   - Renewable energy: electricity yield, revenue, cost, biodiversity effects across all economics modules

4. **Solver Input**: `solvers/input_data.py` prepares optimization model data
   - Biodiversity matrices: GBF2 mask areas, GBF3 NVIS layers (NVIS or IBRA, per `GBF3_NVIS_REGION_MODE`), GBF4 SNES/ECNES matrices, GBF8 species data
   - Renewable energy: Solar/wind yield arrays (`renewable_solar_r`, `renewable_wind_r`), state region mapping, raw targets
   - No input rescaling: constraint blocks are row-rescaled in the solver (`row_builder.scale_rows`, factor kept per row); the objective is raw AUD / 1e6
   - Per-variable term dicts (`term_ag_acct`, `term_am`, `term_nonag`, global Var.index) are built once in `get_input_data` and shared by every constraint family (`row_builder.extract_groups` / `extract_structure` + `attach_coeffs`) and the objective block
   - `LutoSolver` method order follows `formulate()`: variables → spine rows (cell usage, ag-mgt link, adoption, renewable ceilings) → policy rows (demand, GHG) → bio rows (GBF2/3/4/8) → regional adoption → water, renewables → flow rows (source cap, node balance) → objective; then `remove_constraints_by_name` and `solve()`. Constraint handles: `demand_constraints`, `water_limit_constraints`, `renewable_constraints`, `ghg_constr`, `bio_GBF2_constr` (single `Constr` or `None`), `bio_*_constrs` dicts, `regional_adoption_constraints`, `ag_mgt_adoption_constraints`, `ag_mgt_link_constraints_r`, `cell_usage_constraint_r`

5. **Optimization**: `solvers/solver.py` runs GUROBI optimization with biodiversity, renewable energy, and environmental constraints
   - Hard/soft constraint flexibility for GHG, water, GBF2
   - Soft constraints add deviation penalties (`_setup_deviation_penalties()`): demand, GHG, water, biodiversity
   - Objective: `obj_economy × (1 - SOLVE_WEIGHT_BETA) ± obj_penalties × SOLVE_WEIGHT_BETA`. `SOLVE_WEIGHT_BETA` is the **only** economy-vs-penalty knob — the former per-target `SOLVER_WEIGHT_DEMAND/GHG/WATER` weights were removed.
   - The sub-`SOLVER_COEFF_MIN` floor on merged coefficients is stage 4 of every family's `compose_row` and of the objective vector; no post-build sweep exists.

6. **Output Generation**: `tools/write.py` writes results to `/output/`
   - **Two-stage writing process**: Decision variables and mosaic maps written first (stage 1), then all other outputs (stage 2)
   - Stage 1 uses `write_dvar_and_mosaic_map()` which combines dvar and mosaic generation in a single function
   - Mosaic maps are concatenated directly to dvar arrays before saving (optimizes file I/O)
   - Biodiversity outputs: GBF2/3/4/8 scores, species impacts, vegetation group restoration
   - Transition reporting is rebuilt on the solved per-source **delta flows** (`data.delta_dvars_ag2ag[yr_cal]`), giving exact from→to attribution rather than a `base × target × cost` approximation
   - **Per-constraint shadow prices**: after each accepted (OPTIMAL) solve, `record_shadow_prices()` (in `luto/tools/__init__.py`, called from `simulation.py`) reads each constraint's dual (`Constr.Pi`) and writes a shadow-price DataFrame per constraint family (GBF2, GBF3_NVIS, GBF4_SNES, GBF4_ECNES, GBF8, Water, GHG, Demand, Renewable, Regional Adoption) into each `out_<year>/` dir. Columns include `shadow_price` (per real unit, e.g. AUD/ha) and `shadow_price_AUD` (normalised, comparable across families)
   - Parallel output writing with joblib (concurrency auto-determined by `WRITE_REPORT_MAX_MEM_MB`; `get_n_jobs()` budgets by true per-worker cost)

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
   - Constraint type: hard or soft (configurable via `GBF2_CONSTRAINT_TYPE`)
2. **GBF3 NVIS / IBRA**: NVIS major vegetation group targets, or IBRA bioregion targets
   - Function: `get_GBF3_NVIS_matrices_vr(data)` returns the layers for both
   - Settings: `GBF3_NVIS_TARGET_CLASS` ('NVIS_MVG' or 'NVIS_MVS'); `GBF3_NVIS_REGION_MODE` ('AUSTRALIA', 'NRM', or 'IBRA_REG') selects NVIS vs IBRA. There is no separate IBRA function, attribute, setting, or constraint method.
3. **GBF4**: Species and Ecological Community NES
   - SNES: `get_GBF4_SNES_matrix_sr(data)`
   - ECNES: `get_GBF4_ECNES_matrix_sr(data)`
4. **GBF8**: Species conservation
   - Function: `get_GBF8_matrix_sr(data, target_year)`

### Mask Proportion Strategy (`AG_MASK_PROPORTION_R`)

When `RESFACTOR > 1`, each coarsened cell may only partially overlap the LUTO study area. `AG_MASK_PROPORTION_R` (defined in `data.py` as `AG_L_MRJ.sum(0).sum(1)`) captures the fraction of each coarsened cell that is inside LUTO. Whether a biodiversity constraint needs this correction depends on how its area coefficients are computed:

**Needs `AG_MASK_PROPORTION_R`:**
- **GBF2** — `BIO_GBF2_MASK` is a **binary mask** (`bio_quality_raw >= threshold`), which is `True/False` for the entire coarsened cell regardless of partial coverage. So `BIO_GBF2_MASK * REAL_AREA` overstates the area for boundary cells. The mask proportion is applied in:
  - `get_GBF2_MASK_area()` → `BIO_GBF2_MASK * REAL_AREA * AG_MASK_PROPORTION_R`
  - `BIO_GBF2_BASE_YR` einsum result → `* AG_MASK_PROPORTION_R`
  - `get_GBF2_target_for_yr_cal()` baseline sum → `* AG_MASK_PROPORTION_R`

**Does NOT need `AG_MASK_PROPORTION_R`:**
- **GBF3 NVIS/IBRA, GBF4 SNES/ECNES, GBF8** — Their layer arrays (`GBF3_NVIS_LAYERS_LDS`, `GBF4_SNES_LAYERS_SEL`, etc.) are built via `get_resfactored_average_fraction()`, which coarsens by computing `mean()` over all RESFACTOR² subcells (including zeros outside LUTO). A boundary cell with 7/25 subcells in LUTO gets fraction 7/25. Multiplied by `REAL_AREA` (= cell_area × RESFACTOR²), this correctly yields `7 × cell_area`. The partial-cell correction is already implicit in the fractional layer values.

**Rule of thumb:** If the constraint coefficient is a **binary mask** or scalar per coarsened cell, multiply by `AG_MASK_PROPORTION_R`. If it comes from `get_resfactored_average_fraction()`, the correction is already built in.

## Renewable Energy Module

The renewable energy module (REM) introduces solar and wind energy generation as agricultural management options.

### Architecture

Renewable energy types (Utility Solar PV, Onshore Wind) are implemented as non-reversible agricultural management options (`AG_MANAGEMENTS`). Each type has effects across all economics modules:

- **`quantity.py`**: `get_quantity_renewable(data, re_type, yr_idx)` — MWh per cell = `MW_HA_HR × capacity% × (1 - distribution_loss%) × 8760 × REAL_AREA`
- **`revenue.py`**: Electricity revenue (quantity × state-level price) + ag revenue change via productivity multiplier
- **`cost.py`**: O&M cost multiplier on base ag costs + operational costs from spatial layers
- **`transitions.py`**: Upfront installation CAPEX (not amortized)
- **`biodiversity.py`**: Biodiversity compatibility impacts from bundle data
- **`water.py`**: Water requirement impacts
- **`ghg.py`**: Returns zeros (displacement handled externally via AusTIMES)

### Solver Constraints

`_add_renewable_energy_constraints()` in `solver.py` enforces state-level generation targets:
- Separate constraints for solar and wind per state (ACT excluded)
- Uses `renewable_solar_r` / `renewable_wind_r` yield arrays from `input_data.py`
- Per-row rescaling: each (type, state) row carries its own factor (`renewable_scales`)

### Data Loading (`data.py`)

- `RENEWABLE_TARGETS`: State-level generation targets (TWh → MWh) by year, scenario, product
- `SOLAR_PRICES` / `WIND_PRICES`: Separate state-level electricity prices (AUD/MWh)
- `RENEWABLE_LAYERS`: NetCDF spatial layers (install cost, operation cost, capacity %, distribution loss %)
- `RENEWABLE_BUNDLE_SOLAR` / `RENEWABLE_BUNDLE_WIND`: Parameters per land use

## Theta-Fold Transition Model & Accounting Stream

Transition costs use a **fold-into-dominant (θ)** model with a **two-stream** formulation in `solver.py`:

- **Decision / flow stream (`dvar_flow`)** carries the *folded* composition: within each cell, every sub-θ land-use sliver is merged into that cell's **dominant** land use, so a single scalar variable represents "how much of this cell stays in its original composition". This keeps the transition matrix small and well-conditioned.
- **Accounting stream (`dvar_account`)**, built by `_setup_ag_accounting_vars()`, **un-folds** that scalar back into each true land use as a constant-ratio `LinExpr`, so profit / water / GHG / GBF / production are scored against the real per-land-use fractions rather than the folded dominant.

**Mental model** — a cell is a fixed-composition bundle scaled by one scalar. If a cell is 0.7 Beef + 0.3 Apple, folding merges Apple into the dominant Beef so one variable `X_Beef` (mass 1.0) represents the whole cell; each land use is then a constant ratio of it (`Apple = 0.3/1.0 · X_Beef`). Reducing `X_Beef` shrinks both fractions proportionally — the 7:3 composition ratio is preserved, only the scale changes.

**Coefficient-floor consequence**: because accounting terms are `coeff × X_acct` where a folded-sliver `X_acct` entry is a weighted sum of the dominant's variable (weights ~1/RESFACTOR²), a floored-and-kept `coeff` can distribute into a *sub-floor* product on the dominant var. Stage 4 of the coefficient contract floors the MERGED coefficient in every composed row (`row_builder.compose_row`) and in the objective vector (`_setup_objective`), which drops these (see `docs/FINDINGS.md`, 20260721, for the original diagnosis).

Transition **reporting** (`write.py`) is rebuilt on the solved per-source delta flows (`data.delta_dvars_ag2ag[yr_cal]` etc.), giving exact from→to attribution.

## Simulation Flow

```
load_data() → Data() initialization
    ↓
run(data) → solve_timeseries(data, years=sorted(SIM_YEARS))   # default 2020, 2025, …, 2050
    ↓
    For each year pair (base→target):
        ├── get_input_data(data, base_yr, target_yr) → SolverInputData
        ├── LutoSolver(input_data).formulate()
        │   ├── _setup_vars()             # incl. _setup_ag_accounting_vars() (accounting stream)
        │   ├── _setup_constraints()
        │   └── _setup_objective()       # _setup_economy_objective() -> obj_block (5 x n_vars); summed, scaled, floored (stage 4)
        ├── solve() → SolverSolution
        ├── record_shadow_prices(...) → out_<year>/ (per-constraint duals)
        └── Store results: lumaps, lmmaps, ag_dvars, non_ag_dvars, ag_man_dvars, delta_dvars_ag2ag
    ↓
    save_data_to_disk(data) [joblib + lz4]
    ↓
    write_outputs(data) → write_data() + create_report()
```

### Key Data Structures

| Index | Dimension | Count | Description |
|-------|-----------|-------|-------------|
| m | Land Management | 2 | Dryland (0), Irrigated (1) |
| r | Cell | ~100K-7M | Spatial cell index (depends on RESFACTOR) |
| j | Agricultural Land-Use | 28 | Crop/livestock types |
| k | Non-Agricultural Land-Use | 9 | Environmental plantings, agroforestry, etc. |
| p | Product | 40+ | Individual crop/livestock products |
| c | Commodity | 20+ | Aggregated commodity categories |
| v | Vegetation/Bioregion | Variable | GBF3 NVIS/IBRA groups |
| s | Species/Community | Variable | GBF4/GBF8 indices |
