# LUTO2 Architecture Overview

This document describes the core architecture, modules, and data flow of LUTO2.

## Core Modules

- **`luto/simulation.py`**: Main simulation engine and state management singleton
- **`luto/data.py`**: Core data management, loading, and spatial data structures
- **`luto/settings.py`**: Configuration parameters for all model aspects
- **`luto/solvers/`**: Optimization solver interface and input data preparation
  - `solver.py`: GUROBI solver wrapper (LutoSolver class)
    - Biodiversity constraint methods: `_add_GBF2_constraints()`, `_add_GBF3_NVIS_constraints()`, `_add_GBF3_IBRA_constraints()`, `_add_GBF4_SNES_constraints()`, `_add_GBF4_ECNES_constraints()`, `_add_GBF8_constraints()`
    - Renewable energy constraint method: `_add_renewable_energy_constraints()` — enforces state-level solar and wind generation targets
    - Hard/soft constraint flexibility: `GHG_CONSTRAINT_TYPE`, `WATER_CONSTRAINT_TYPE`, `GBF2_CONSTRAINT_TYPE`
  - `input_data.py`: Prepares optimization model input data
    - Biodiversity data attributes use `*_pre_1750_area_*` naming (e.g., `GBF3_NVIS_pre_1750_area_vr`, `GBF3_IBRA_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`)
    - Renewable energy data: `renewable_solar_r`, `renewable_wind_r` yield arrays; `region_state_r` mapping
    - `rescale_solver_input_data()`: **In-place** rescaling of arrays to magnitude 0-1e3 for numerical stability
    - Separate rescaling for: Economy, Demand, Biodiversity, GHG, Renewable_Solar, Renewable_Wind, Water, GBF2, GBF3_NVIS, GBF3_IBRA, GBF4_SNES, GBF4_ECNES, GBF8

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
  - `get_GBF3_NVIS_matrices_vr()`: NVIS vegetation layer matrices for GBF3
  - `get_GBF3_IBRA_matrices_vr()`: IBRA bioregion layer matrices for GBF3
  - `get_GBF4_SNES_matrix_sr()`, `get_GBF4_ECNES_matrix_sr()`: Species/Ecological Community NES matrices
  - `get_GBF8_matrix_sr()`: Species conservation matrices
  - Variable naming convention: `*_pre_1750_area_*` for baseline biodiversity area matrices
- **Agricultural management options** (10 types): Asparagopsis taxiformis, Precision Agriculture, Ecological Grazing, Savanna Burning, AgTech EI, Biochar, HIR-Beef, HIR-Sheep, Utility Solar PV, Onshore Wind

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
   - Biodiversity matrices: GBF2 mask areas, GBF3 NVIS layers, GBF3 IBRA layers, GBF4 SNES/ECNES matrices, GBF8 species data
   - Renewable energy: Solar/wind yield arrays (`renewable_solar_r`, `renewable_wind_r`), state region mapping, rescaled targets
   - Data rescaling: Arrays rescaled in-place to 0-1e3 magnitude for numerical stability (13 separate scale factors)

5. **Optimization**: `solvers/solver.py` runs GUROBI optimization with biodiversity, renewable energy, and environmental constraints
   - Hard/soft constraint flexibility for GHG, water, GBF2
   - Penalty variables (V: demand, E: GHG, W: water) for soft constraints
   - Multi-objective: economy + biodiversity terms with configurable weights

6. **Output Generation**: `tools/write.py` writes results to `/output/`
   - **Two-stage writing process**: Decision variables and mosaic maps written first (stage 1), then all other outputs (stage 2)
   - Stage 1 uses `write_dvar_and_mosaic_map()` which combines dvar and mosaic generation in a single function
   - Mosaic maps are concatenated directly to dvar arrays before saving (optimizes file I/O)
   - Biodiversity outputs: GBF2/3/4/8 scores, species impacts, vegetation group restoration
   - Parallel output writing with joblib (configurable via `WRITE_PARALLEL` and `WRITE_THREADS`)

## Biodiversity Module Naming Conventions

The biodiversity module follows consistent naming conventions for GBF (Global Biodiversity Framework) variables:

### Variable Naming Pattern
- **Pre-1750 baseline areas**: Use `*_pre_1750_area_*` suffix
  - Examples: `GBF3_NVIS_pre_1750_area_vr`, `GBF3_IBRA_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`, `GBF8_pre_1750_area_sr`
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
   - Constraint type: hard or soft (configurable via `GBF2_CONSTRAINT_TYPE`)
2. **GBF3 NVIS**: NVIS major vegetation group targets
   - Function: `get_GBF3_NVIS_matrices_vr(data)` returns vegetation layers
   - Settings: `GBF3_NVIS_TARGET_CLASS` ('MVG' or 'MVS')
3. **GBF3 IBRA**: IBRA bioregion targets
   - Function: `get_GBF3_IBRA_matrices_vr(data)` returns bioregion layers
   - Settings: `GBF3_IBRA_TARGET_CLASS` ('IBRA_Regions' or 'IBRA_Sub_regions')
4. **GBF4**: Species and Ecological Community NES
   - SNES: `get_GBF4_SNES_matrix_sr(data)`
   - ECNES: `get_GBF4_ECNES_matrix_sr(data)`
5. **GBF8**: Species conservation
   - Function: `get_GBF8_species_matrices_sr(data, target_year)`

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
- Separate rescaling: `Renewable_Solar` and `Renewable_Wind` scale factors

### Data Loading (`data.py`)

- `RENEWABLE_TARGETS`: State-level generation targets (TWh → MWh) by year, scenario, product
- `SOLAR_PRICES` / `WIND_PRICES`: Separate state-level electricity prices (AUD/MWh)
- `RENEWABLE_LAYERS`: NetCDF spatial layers (install cost, operation cost, capacity %, distribution loss %)
- `RENEWABLE_BUNDLE_SOLAR` / `RENEWABLE_BUNDLE_WIND`: Parameters per land use

## Simulation Flow

```
load_data() → Data() initialization
    ↓
run(data) → solve_timeseries(data, years=[2010, 2015, ..., 2050])
    ↓
    For each year pair (base→target):
        ├── get_input_data(data, base_yr, target_yr) → SolverInputData
        ├── LutoSolver(input_data).formulate()
        │   ├── _setup_vars()
        │   ├── _setup_constraints()
        │   └── _setup_objective()
        ├── solve() → SolverSolution
        └── Store results: lumaps, lmmaps, ag_dvars, non_ag_dvars, ag_man_dvars
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
