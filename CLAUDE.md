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
    - Biodiversity constraint methods: `_add_GBF2_constraints()`, `_add_GBF3_NVIS_constraints()`, `_add_GBF3_IBRA_constraints()`, `_add_GBF4_SNES_constraints()`, `_add_GBF4_ECNES_constraints()`, `_add_GBF8_constraints()`
    - Renewable energy constraint method: `_add_renewable_energy_constraints()` — enforces state-level solar and wind generation targets
  - `input_data.py`: Prepares optimization model input data
    - Biodiversity data attributes use `*_pre_1750_area_*` naming (e.g., `GBF3_NVIS_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`)
    - `rescale_solver_input_data()`: **In-place** rescaling of arrays to magnitude 0-1e3 for numerical stability

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
    - `get_GBF3_NVIS_matrices_vr()`: NVIS vegetation layer matrices for GBF3
    - `get_GBF3_IBRA_matrices_vr()`: IBRA bioregion layer matrices for GBF3
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
  - `BIODIVERSITY_TARGET_GBF_2`: Priority degraded areas restoration ('off', 'low', 'medium', 'high')
  - `GBF2_CONSTRAINT_TYPE`: Hard or soft GBF2 constraint ('hard' or 'soft')
  - `BIODIVERSITY_TARGET_GBF_3_NVIS`: NVIS vegetation group targets ('off', 'medium', 'high', 'USER_DEFINED')
  - `BIODIVERSITY_TARGET_GBF_3_IBRA`: IBRA bioregion targets ('off', 'medium', 'high', 'USER_DEFINED')
  - `BIODIVERSITY_TARGET_GBF_4_SNES`: Species NES (National Environmental Significance) ('on' or 'off')
  - `BIODIVERSITY_TARGET_GBF_4_ECNES`: Ecological Community NES ('on' or 'off')
  - `BIODIVERSITY_TARGET_GBF_8`: Species conservation targets ('on' or 'off')

### Renewable Energy Settings
- `RENEWABLE_ENERGY_CONSTRAINTS`: Enable renewable energy generation targets ('on' or 'off')
- `RENEWABLES_OPTIONS`: Renewable energy types: `['Utility Solar PV', 'Onshore Wind']`
- `RENEWABLE_TARGET_SCENARIO`: Target scenario ('CNS25 - Accelerated Transition' or 'CNS25 - Current Targets')
- `RE_TARGET_LEVEL`: Spatial level for constraints ('STATE' or 'NRM'; only STATE currently supported)
- `INSTALL_CAPACITY_MW_HA`: Per-hectare capacity (MW/ha) per renewable type
- `RENEWABLES_ADOPTION_LIMITS`: Maximum adoption fraction per type (default: 1.0 for both)
- Both renewable types are registered as non-reversible agricultural management options in `AG_MANAGEMENTS`
- Compatible land uses differ: Solar PV excludes Hay; Wind includes Hay and horticulture crops

### Solver Configuration
- `SOLVE_METHOD`: GUROBI algorithm (default: 2 for barrier method)
- `THREADS`: Parallel threads for optimization (default: min(32, cpu_count))
- `FEASIBILITY_TOLERANCE`: Solver tolerance (default: 1e-2, relaxed from 1e-6)
- `OPTIMALITY_TOLERANCE`: Optimality tolerance (default: 1e-2)
- `BARRIER_CONVERGENCE_TOLERANCE`: Barrier method convergence (default: 1e-5)
- `RESCALE_FACTOR`: Rescaling magnitude for numerical stability (default: 1e3)
- `SOLVER_WEIGHT_DEMAND`: Demand deviation weight in objective (default: 1)
- `SOLVER_WEIGHT_GHG`: GHG deviation weight in objective (default: 1)
- `SOLVER_WEIGHT_WATER`: Water deviation weight in objective (default: 1)

### Output Writing Configuration
- `WRITE_PARALLEL`: Enable parallel output writing (default: True)
- `WRITE_THREADS`: Number of parallel write threads (default: min(6, cpu_count))
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
4. **Solver Input**: `solvers/input_data.py` prepares optimization model data
   - Biodiversity matrices: GBF2 mask areas, GBF3 NVIS layers, GBF3 IBRA layers, GBF4 SNES/ECNES matrices, GBF8 species data
   - Renewable energy: Solar/wind yield arrays (`renewable_solar_r`, `renewable_wind_r`), state region mapping, rescaled targets
   - Data rescaling: Arrays rescaled in-place to 0-1e3 magnitude for numerical stability
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
  - Examples: `_add_GBF2_constraints()`, `_add_GBF3_NVIS_constraints()`, `_add_GBF3_IBRA_constraints()`, `_add_GBF4_SNES_constraints()`
  - Maintain consistency between method names and GBF target types

### Data Structure Indices
- `v, r`: Vegetation group / bioregion (v) × cell (r) - used for GBF3 NVIS and IBRA data
- `s, r`: Species/community (s) × cell (r) - used for GBF4 and GBF8 data
- `r`: Cell only - used for GBF2 mask data

### Key GBF Modules
1. **GBF2**: Priority degraded areas restoration
   - Function: `get_GBF2_MASK_area(data)` returns mask × real area
2. **GBF3 NVIS**: NVIS major vegetation group targets
   - Function: `get_GBF3_NVIS_matrices_vr(data)` returns vegetation layers
3. **GBF3 IBRA**: IBRA bioregion targets
   - Function: `get_GBF3_IBRA_matrices_vr(data)` returns bioregion layers
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
- Uses `renewable_solar_r` / `renewable_wind_r` yield arrays from `input_data.py`
- Targets from `RENEWABLE_TARGETS` CSV, filtered by year and scenario
- Rescaled for numerical stability (same pattern as biodiversity constraints)

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
- **Separate rescaling**: Solar and wind yield arrays are rescaled independently (`Renewable_Solar` / `Renewable_Wind` scale factors)
- **ACT excluded**: Australian Capital Territory skipped in state-level constraints

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

### JSON Output Hierarchies (Map vs Chart)

Map and Chart JSON files have different dimension hierarchies. See [CLAUDE_VUE_REPORTING.md](docs/CLAUDE_VUE_REPORTING.md) for the full per-module table.

**Map JSON (Spatial Layers)** — ends at `year → {img_str, bounds, min_max}`:
- **Ag**: `lm → lu → year` (standard); `lm → source → lu → year` (GHG/Economics)
- **Am**: `am → lm → lu → year` (standard); no source in Am for GHG
- **NonAg**: `lu → year`

**Chart JSON (Time Series)** — ends at `[series array]`:
- **Ag**: `region → lm → lu` (standard); `region → lm → source → [series(name=LU)]` (GHG)
- **Am**: `region → lm → lu → [series(name=AgMgt)]`; source removed from Am in GHG/Water
- **NonAg**: `region → [series(name=LU)]`

**`source` dimension** appears only in **Ag** for GHG (emission type) and Economics (cost/revenue type). Am no longer has a source level.

**Valid Layers Pattern** — two approaches:
- **Economics** (revenue/cost/profit/transitions): `ALL` = dvar mosaic (categorical) — load dvar, filter, concat
- **GHG / Biodiversity / Water / Production**: `ALL` = sum aggregate — `xr.concat([data.sum('dim'), data], 'dim')` before stacking

See [CLAUDE_OUTPUT.md](docs/CLAUDE_OUTPUT.md) for detailed examples.

### Vue.js Progressive Selection Hierarchies

- **Standard Full**: Category → AgMgt → Water → Landuse
- **Biodiversity**: Metric → Category → AgMgt → Water → Landuse
- **NonAg Simplified**: Category → Landuse
- **DVAR Simplified**: Category → Landuse/AgMgt → Year
- **Economics Extended**: Category → MapType → (AgMgt) → Water → (Source) → Landuse

## Getting Started

1. **New to the project?** Start with [CLAUDE_SETUP.md](docs/CLAUDE_SETUP.md) for environment setup
2. **Working on core model logic?** See [CLAUDE_ARCHITECTURE.md](docs/CLAUDE_ARCHITECTURE.md)
3. **Working on output generation?** See [CLAUDE_OUTPUT.md](docs/CLAUDE_OUTPUT.md)
4. **Working on the reporting UI?** See [CLAUDE_VUE_REPORTING.md](docs/CLAUDE_VUE_REPORTING.md)

**Remember**: Only read the documentation file relevant to your current task to minimize memory usage!
