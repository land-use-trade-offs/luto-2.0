# LUTO2 Architecture Overview

This document describes the core architecture, modules, and data flow of LUTO2.

## Core Modules

- **`luto/simulation.py`**: Main simulation engine and state management singleton
- **`luto/data.py`**: Core data management, loading, and spatial data structures
- **`luto/settings.py`**: Configuration parameters for all model aspects
- **`luto/solvers/`**: Optimization solver interface and input data preparation
  - `solver.py`: GUROBI solver wrapper (LutoSolver class)
    - Biodiversity constraint methods: `_add_GBF2_constraints()`, `_add_GBF3_NVIS_constraints()`, `_add_GBF4_SNES_constraints()`, `_add_GBF4_ECNES_constraints()`, `_add_GBF8_constraints()`
  - `input_data.py`: Prepares optimization model input data
    - Biodiversity data attributes use `*_pre_1750_area_*` naming (e.g., `GBF3_NVIS_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`)
    - `rescale_solver_input_data()`: **In-place** rescaling of arrays to magnitude 0-1e3 for numerical stability

## Economic Modules

### Agricultural Economics (`luto/economics/agricultural/`)
- Revenue, cost, quantity, water, biodiversity, GHG calculations
- Transition costs between agricultural land uses
- **Dynamic pricing** (`revenue.py`): Demand elasticity-based price adjustments
  - Calculates commodity price multipliers based on supply-demand dynamics
  - Uses elasticity coefficients and demand deltas from 2010 baseline
  - Applied to crops and livestock (beef, sheep, dairy) when `DYNAMIC_PRICE` enabled
- **Biodiversity module** (`biodiversity.py`): GBF (Global Biodiversity Framework) calculations
  - `get_GBF2_MASK_area()`: Returns GBF2 priority degraded areas (mask × real area)
  - `get_GBF3_NVIS_matrices_vr()`: NVIS vegetation layer matrices for GBF3
  - `get_GBF4_SNES_matrix_sr()`, `get_GBF4_ECNES_matrix_sr()`: Species/Ecological Community NES matrices
  - Variable naming convention: `*_pre_1750_area_*` for baseline biodiversity area matrices

### Non-Agricultural Economics (`luto/economics/non_agricultural/`)
- Environmental plantings, carbon plantings, etc.

### Off-Land Commodity (`luto/economics/off_land_commodity/`)
- Off-land commodity economics

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

2. **Preprocessing**: `dataprep.py` processes raw data into model-ready formats
   - Copies demand elasticity data from source to input directory
   - **Carbon data preparation**: Converts 3D timeseries to NetCDF format with age dimension
   - Selects specific ages (50, 60, 70, 80, 90 years) for carbon accumulation data
   - Applies chunked compression (zlib level 5) for efficient storage

3. **Economic Calculations**: Economics modules calculate costs, revenues, transitions, biodiversity impacts
   - Revenue calculations apply demand elasticity multipliers when `DYNAMIC_PRICE` enabled
   - Elasticity multipliers computed as: `1 + (demand_delta / demand_elasticity)`

4. **Solver Input**: `solvers/input_data.py` prepares optimization model data
   - Biodiversity matrices: GBF2 mask areas, GBF3 NVIS layers, GBF4 SNES/ECNES matrices, GBF8 species data
   - Data rescaling: Arrays rescaled in-place to 0-1e3 magnitude for numerical stability

5. **Optimization**: `solvers/solver.py` runs GUROBI optimization with biodiversity constraints

6. **Output Generation**: `tools/write.py` writes results to `/output/`
   - **Two-stage writing process**: Decision variables and mosaic maps written first (stage 1), then all other outputs (stage 2)
   - Stage 1 uses `write_dvar_and_mosaic_map()` which combines dvar and mosaic generation in a single function
   - Mosaic maps are concatenated directly to dvar arrays before saving (optimizes file I/O)
   - Biodiversity outputs: GBF2/3/4/8 scores, species impacts, vegetation group restoration

## Biodiversity Module Naming Conventions

The biodiversity module follows consistent naming conventions for GBF (Global Biodiversity Framework) variables:

### Variable Naming Pattern
- **Pre-1750 baseline areas**: Use `*_pre_1750_area_*` suffix
  - Examples: `GBF3_NVIS_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`, `GBF8_pre_1750_area_sr`
  - These represent baseline biodiversity area matrices before land use changes

### Function Naming Pattern
- **GBF constraint methods**: Use `_add_GBF{N}_{TYPE}_constraints()` format
  - Examples: `_add_GBF2_constraints()`, `_add_GBF3_NVIS_constraints()`, `_add_GBF4_SNES_constraints()`
  - Maintain consistency between method names and GBF target types

### Data Structure Indices
- `v, r`: Vegetation group (v) × cell (r) - used for GBF3 NVIS data
- `s, r`: Species/community (s) × cell (r) - used for GBF4 and GBF8 data
- `r`: Cell only - used for GBF2 mask data

### Key GBF Modules
1. **GBF2**: Priority degraded areas restoration
   - Function: `get_GBF2_MASK_area(data)` returns mask × real area
2. **GBF3**: NVIS major vegetation group targets
   - Function: `get_GBF3_NVIS_matrices_vr(data)` returns vegetation layers
3. **GBF4**: Species and Ecological Community NES
   - SNES: `get_GBF4_SNES_matrix_sr(data)`
   - ECNES: `get_GBF4_ECNES_matrix_sr(data)`
4. **GBF8**: Species conservation
   - Function: `get_GBF8_species_matrices_sr(data, target_year)`
