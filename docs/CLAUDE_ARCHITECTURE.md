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
    - Exact transition flow model (see "Transition Flow Model" below): one source per nonzero base-year (lm, lu, cell) entry, per-source delta variables, node-balance and source-cap rows.
  - `col_builder.py`: the COLUMN side — `get_cols(data, base_year) -> (cols, col_side)` builds the column space from the base-year state: `cols`, the long table (every unknown as one row, its fields, `lb` / `ub` / `base`; the row side adds `obj`), and beside it a `ColSide` — the SUPPORT the row side reads by position (`support_rc`, the sparse cell × col incidence the sum families multiply with; the `col_ag_mjr` / `col_nonag_kr` column-id grids, −1 = no column, the join families look up), the wide `base` / `ub` grids for entries with no column, the two source maps, the four renewable masks
  - `row_inputs.py`: the row side's INPUT DATA, purely downstream of the economics modules — it never sees the column space. `get_row_inputs(data, base_year, target_year)` returns every coefficient stream, target and layer as ONE `RowInputs` (no wrapper layer; the setting that turns a family off is read at the call); `get_economics(...)` returns the objective's own streams as an `EconomicInputs`, loaded separately because it is ~300 MB at RES5 and wanted only while the objective is built
  - `row_builder.py`: the row side's ROW GENERATION, the mirror of `col_builder.py` — `get_rows(inputs, cols, col_side) -> (rows, row_side)` at the top, one function in five numbered sections whose body IS the model's row order, everything it calls below it in that order. Every sum family builds rows of the one matrix (rows × cols) the same way: `gather` → weigh (`weight_rows(W) @ (col_side.support_rc @ diags(c))`) → `contract` (drop, then the policy rescale + floor, one loop over the stacked rows); the join families look their nodes up on the column-id grids (−1 = no column / no row), and section 3 lays the biodiversity contribution on the support once (`bio_S`); each family is an `add_*` written out in full (19 of them, no lambda-taking wrappers) returning a part (`row_table.make_part`), and `row_table.stack_rows` lays them into the row table, a `RowSide` (the unscaled production block) beside it; `get_obj(econ, cols, col_side)` is the objective coefficient of every column (million AUD, dropped and floored), attached by the caller as `cols['obj']`.
  - `row_table.py`: the row table's storage (`make_part`, `stack_rows`) and its queries (`family_rows`, `decode`, `rows_where`, `keys_of`) — split out of `row_builder.py` on 2026-09-10 so that file holds only `get_rows`, `get_obj` and the families. `LutoSolver(cols, rows)` is A x T — the two tables and nothing else; `solve()` returns the raw x; `post_solve.post_solve(x, cols, col_side, rows, row_side, inputs)` turns it into the LUTO format (`SolverSolution`: the maps, the dvars, `prod_data` = Production / GHG)
    - Biodiversity data attributes use `*_pre_1750_area_*` naming (e.g., `GBF3_NVIS_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`). IBRA reuses the NVIS attribute — there is no `GBF3_IBRA_pre_1750_area_vr`.
    - Renewable energy data: `renewable_solar_r`, `renewable_wind_r` yield arrays; `region_state_r` mapping
    - **No input rescaling** (2026-09-03): the coefficient streams reach the solver raw (float32). Every constraint block is row-rescaled by `row_builder.contract` — per row, scale = geometric mean of max|row| and |RHS| over `RESCALE_FACTOR`, row and RHS divided by it, stage-4 floor on the scaled row — and the factor is kept on the row table (`scale`); the `solvers/tools.calc_shadow_price_*` readers read it per row straight off the table (So = 1e6: the objective is raw AUD / 1e6). Row scaling is an exact LP transformation; the gate compares models in RESTORED space (rows × their factor).
    - `SOLVER_COEFF_MIN` (1e-4): Universal minimum coefficient threshold, applied by `row_builder.contract` to every family's stacked block (and by `row_builder.get_obj` to the objective coefficients, which it then scales `× (1/1e6)` and floors again): (1) an entry is dropped when `|a| < SOLVER_COEFF_MIN` (NaN too); (2) after the row rescale the scaled coefficient is floored again. Every policy coefficient multiplies an ag / ag-mgt / non-ag column directly. Chosen empirically: 1e-3 caused ~3% economic loss; 1e-4 retains meaningful small coefficients while keeping the matrix ratio at 1e8.
    - No per-family scale factors: every factor is per constraint row, carried on the row table (`scale`); the shadow-price readers (`solvers/tools.calc_shadow_price_*`) walk the table's active rows per family — no per-family views.

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

4. **Solver Input**: `col_builder.get_cols` → `(cols, col_side)`; `row_inputs.get_row_inputs` → `inputs`; `row_builder.get_obj(get_economics(...), cols, col_side)` → `cols['obj']`; `row_builder.get_rows(inputs, cols, col_side)` → `(rows, row_side)`; the solver takes the two tables, `LutoSolver(cols, rows)`
   - Biodiversity matrices: GBF2 mask areas, GBF3 NVIS layers (NVIS or IBRA, per `GBF3_NVIS_REGION_MODE`), GBF4 SNES/ECNES matrices, GBF8 species data
   - Renewable energy: Solar/wind yield arrays (`renewable_solar_r`, `renewable_wind_r`), state region mapping, raw targets
   - No input rescaling: constraint blocks are row-rescaled on the row side (`row_builder.contract`, factor kept per row on the row table); the objective is raw AUD / 1e6 (`row_builder.get_obj`)
   - The column space (`col_builder.py`): every unknown as one row of the long table `cols` (on `col` = Var.index, the blocks `ag | nonag | am | ag2ag | ag2nonag | nonag2ag | cell_usage` back to back, `attrs['block_range']` = {block: (start, stop)}, the rows each block owns; fields `m, j, k, slot, am_idx, j_idx, from_m, from_j, from_k, local_r, cell` with −1 where a field does not apply, `lb` / `ub` / `base` per column and, from the row side, `obj`; attrs `n_terms` / `n_dec` / `n_all`, the sizes `nlms` / `n_ag_lus` / `n_nonag_lus` / `ncells`, `src_ptr` per arc block, `agman2lu` / `savanna_eligible_r`), beside a `ColSide` (`col_side`), the support the row side reads by position, derived from the finished table (`col_support`): `support_rc` (the cell × col incidence), the column-id grids `col_ag_mjr` / `col_nonag_kr` (−1 = no column; no am grid), the value grids `ag_base_mjr`, `nonag_ub_kr` / `nonag_base_kr` for entries with NO column, plus `sources_ag` / `sources_nonag` and the four `mask_*`. Built once in `get_cols` (getters first, then the block builders — each returns its block's ROWS, `ag_space` / `nonag_space` their grid beside it and the arc builders their `src_ptr` — then `table_space`, which lays the rows back to back group by group (`accounting | arcs | cell_use`), so `n_terms` and `n_dec` are group widths rather than positions). Every sum family multiplies the support (`row_builder.gather`, then `W @ (support_rc @ diags(c))`), the join families look their nodes up on the id grids, source cap groups the arcs by their source's position in the base grid, and the objective (`get_obj`) is the same gather plus the arc costs. Every policy row scores the ag columns directly: there is no accounting layer.
   - `LutoSolver(cols, rows)` is A x T and nothing else. `formulate()` = `_setup_vars` (ONE `addMVar` over the column table, the names from the fields) → `_setup_constraints` (ONE `addMConstr` over the row table, the names, the handles kept on `rows['constr']`) → `_setup_objective` (`cols['obj'] @ x`); then `remove_constraints_by_name` / `restore_constraints_by_name` (flag on the table + remove / re-add) and `solve()` → the raw x. The row order is the row table's (`row_builder.get_rows`): ceilings → cell usage → ag-mgt link → adoption → demand → GHG → GBF2/3/4/8 → regional adoption → water → renewables → source cap → node balance. No constraint-handle attributes on the solver: the row table holds every handle and scale, and the `solvers/tools.calc_shadow_price_*` readers walk its active rows per family

5. **Optimization**: `solvers/solver.py` runs GUROBI optimization with biodiversity, renewable energy, and environmental constraints
   - Hard/soft constraint flexibility for GHG, water, GBF2
   - Soft constraints add deviation penalties (`_setup_deviation_penalties()`): demand, GHG, water, biodiversity
   - Objective: `obj_economy × (1 - SOLVE_WEIGHT_BETA) ± obj_penalties × SOLVE_WEIGHT_BETA`. `SOLVE_WEIGHT_BETA` is the **only** economy-vs-penalty knob — the former per-target `SOLVER_WEIGHT_DEMAND/GHG/WATER` weights were removed.
   - The sub-`SOLVER_COEFF_MIN` floor on scaled coefficients is the last step of `row_builder.contract` on every family's stacked block and of the objective vector; no post-build sweep exists. The flow rows (source cap, node balance) and the cell-usage / ag-mgt link rows are structural ±1 rows: they pass through `contract` without the rescale, and the drop is a no-op on them or rescaled.

6. **Output Generation**: `tools/write.py` writes results to `/output/`
   - **Two-stage writing process**: Decision variables and mosaic maps written first (stage 1), then all other outputs (stage 2)
   - Stage 1 uses `write_dvar_and_mosaic_map()` which combines dvar and mosaic generation in a single function
   - Mosaic maps are concatenated directly to dvar arrays before saving (optimizes file I/O)
   - Biodiversity outputs: GBF2/3/4/8 scores, species impacts, vegetation group restoration
   - Transition reporting is rebuilt on the solved per-source **delta flows** (`data.delta_dvars_ag2ag[yr_cal]`), giving exact from→to attribution rather than a `base × target × cost` approximation
   - **Per-constraint shadow prices**: after each accepted (OPTIMAL) solve, `record_shadow_prices()` (in `luto/solvers/tools.py`, called from `simulation.py`) reads each constraint's dual (`Constr.Pi`) and writes a shadow-price DataFrame per constraint family (GBF2, GBF3_NVIS, GBF4_SNES, GBF4_ECNES, GBF8, Water, GHG, Demand, Renewable, Regional Adoption) into each `out_<year>/` dir. Columns include `shadow_price` (per real unit, e.g. AUD/ha) and `shadow_price_AUD` (normalised, comparable across families)
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
- Uses `renewable_solar_r` / `renewable_wind_r` yield arrays from `row_inputs.get_rows`
- Per-row rescaling: each (type, state) row carries its own factor (`renewable_scales`)

### Data Loading (`data.py`)

- `RENEWABLE_TARGETS`: State-level generation targets (TWh → MWh) by year, scenario, product
- `SOLAR_PRICES` / `WIND_PRICES`: Separate state-level electricity prices (AUD/MWh)
- `RENEWABLE_LAYERS`: NetCDF spatial layers (install cost, operation cost, capacity %, distribution loss %)
- `RENEWABLE_BUNDLE_SOLAR` / `RENEWABLE_BUNDLE_WIND`: Parameters per land use

## Transition Flow Model (exact)

Transitions are explicit per-source delta flows. The base-year ag dvar `data.ag_dvars[base_year]` is used as it is: every nonzero (lm, lu) fraction of a cell above the `ROUND_DECIMALS` noise floor (1e-6) is a **source** (`get_base_dvar_mj_cell_map`: `{(from_m, from_j): cells}`), and so is every nonzero non-ag fraction (`get_base_nonag_dvar_k_cell_map`). For each source the column space holds one delta variable per feasible (cell, target) arc (`ag2ag / ag2nonag / nonag2ag`), the source-cap rows bound Σ out ≤ base, and the node-balance rows define every land-use column as base + Σ in − Σ out. Policy rows (profit, water, GHG, GBF, production) score the ag / ag-mgt / non-ag columns directly.

There is no fold. The former θ dial (`EXACT_REACHABILITY_MIN_FRACTION`), which merged sub-θ fractions into the cell's dominant land use and undid the merge with an accounting layer (`X_acct_*` columns, `acct_link_*` rows), was removed on 2026-09-08: a census of the RES5 2020→2050 trajectory showed the fold catching 12–69 of ~350k nonzero entries per year (≤ 0.02 %), so the exact model was already the model being solved (see `docs/FINDINGS.md`, 20260908). Model size scales with the number of nonzero base-year entries, not with the number of cells.

Transition **reporting** (`write.py`) is rebuilt on the solved per-source delta flows (`data.delta_dvars_ag2ag[yr_cal]` etc.), giving exact from→to attribution.

## Simulation Flow

```
load_data() → Data() initialization
    ↓
run(data) → solve_timeseries(data, years=sorted(SIM_YEARS))   # default 2020, 2025, …, 2050
    ↓
    For each year pair (base→target):
        ├── col_builder.get_cols(data, base_yr) → (cols, col_side);  row_inputs.get_row_inputs(data, base_yr, target_yr) → inputs
        ├── cols['obj'] = row_builder.get_obj(row_inputs.get_economics(...), cols, col_side);  row_builder.get_rows(inputs, cols, col_side) → (rows, row_side)
        ├── LutoSolver(cols, rows).formulate()      # A x T: ONE addMVar over the column table; ONE addMConstr over the row table
        │   ├── _setup_vars()             # the column table -> gp.MVar, names from the fields
        │   ├── _setup_constraints()      # the row table -> addMConstr; handles on rows['constr']
        │   └── _setup_objective()       # cols['obj'] @ x
        ├── solve() → x;  post_solve.post_solve(x, cols, col_side, rows, row_side, inputs) → SolverSolution
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
