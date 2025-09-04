# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LUTO2 is the Land-Use Trade-Offs Model Version 2, an integrated land systems optimization model for Australia. It simulates optimal spatial arrangement of land use and management decisions to achieve climate and biodiversity targets while maintaining economic productivity. The model uses GUROBI optimization solver and processes large spatial datasets.

## Development Environment Setup

### Environment Creation
```bash
# Create and activate conda environment
conda env create -f luto/tools/create_task_runs/bash_scripts/conda_env.yml
conda activate luto

# Install additional pip dependencies (gurobipy, numpy_financial, tables)
pip install gurobipy==11.0.2 numpy_financial==1.0.0 tables==3.9.2
```

### GUROBI License
LUTO2 requires a GUROBI optimization solver license. Academic licenses are available at gurobi.com. Place your `gurobi.lic` file in the appropriate directory as specified by GUROBI documentation.

### Input Data
The model requires approximately 40GB of input data that must be obtained separately by contacting b.bryan@deakin.edu.au. Input data goes in the `/input/` directory.

## Common Development Commands

### Testing
```bash
# Run all tests from repository root
python -m pytest

# Run tests with specific patterns
python -m pytest luto/tests/
```

### Running Simulations
```python
# Basic simulation
import luto.simulation as sim
data = sim.load_data()
results = sim.run(data=data)

# With custom settings
import luto.settings as settings
settings.RESFACTOR = 10  # Coarser spatial resolution
settings.SIM_YEARS = [2010, 2020, 2030]
data = sim.load_data()
sim.run(data=data)
```

### Batch Processing
```bash
# Use tools for creating and managing batch runs
python luto/tools/create_task_runs/create_grid_search_tasks.py
```

## Architecture Overview

### Core Modules
- **`luto/simulation.py`**: Main simulation engine and state management singleton
- **`luto/data.py`**: Core data management, loading, and spatial data structures
- **`luto/settings.py`**: Configuration parameters for all model aspects
- **`luto/solvers/`**: Optimization solver interface and input data preparation
  - `solver.py`: GUROBI solver wrapper (LutoSolver class)
  - `input_data.py`: Prepares optimization model input data

### Economic Modules
- **`luto/economics/agricultural/`**: Agricultural land use economics
  - Revenue, cost, quantity, water, biodiversity, GHG calculations
  - Transition costs between agricultural land uses
- **`luto/economics/non_agricultural/`**: Non-agricultural land use economics
  - Environmental plantings, carbon plantings, etc.
- **`luto/economics/off_land_commodity/`**: Off-land commodity economics

### Data Processing
- **`luto/dataprep.py`**: Data preprocessing utilities
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
- `SIM_YEARS`: Simulation time periods (default: 2020-2050 in 5-year steps)
- `RESFACTOR`: Spatial resolution factor (1 = full resolution, >1 = coarser)
- `SCENARIO`: Shared Socioeconomic Pathway (SSP1-SSP5)
- `RCP`: Representative Concentration Pathway (e.g., 'rcp4p5')
- `OBJECTIVE`: Optimization objective ('maxprofit' or 'mincost')

### Environmental Constraints
- `GHG_EMISSIONS_LIMITS`: Greenhouse gas targets ('off', 'low', 'medium', 'high')
- `WATER_LIMITS`: Water yield constraints ('on' or 'off')
- `BIODIVERSITY_TARGET_GBF_*`: Various Global Biodiversity Framework targets

### Solver Configuration
- `SOLVE_METHOD`: GUROBI algorithm (default: 2 for barrier method)
- `THREADS`: Parallel threads for optimization
- `FEASIBILITY_TOLERANCE`: Solver tolerance settings

## Data Flow

1. **Data Loading**: `luto.data.Data` class loads spatial datasets from `/input/`
2. **Preprocessing**: `dataprep.py` processes raw data into model-ready formats
3. **Economic Calculations**: Economics modules calculate costs, revenues, transitions
4. **Solver Input**: `solvers/input_data.py` prepares optimization model data
5. **Optimization**: `solvers/solver.py` runs GUROBI optimization
6. **Output Generation**: `tools/write.py` writes results to `/output/`

## Output Structure

Results are saved in `/output/<timestamp>/`:
- `DATA_REPORT/REPORT_HTML/index.html`: Interactive HTML dashboard
- Spatial outputs: GeoTIFF files for mapping
- Data tables: CSV files with numerical results
- Logs: Execution logs and performance metrics

## Memory and Performance

- Minimum 16GB RAM (32GB recommended for large simulations)
- Model complexity requires substantial computational resources
- Use `RESFACTOR > 1` for testing and development to reduce memory usage
- Monitor memory usage with built-in logging utilities

## Testing Framework

- Uses pytest with hypothesis for property-based testing
- Tests focus on robustness of core functionality
- Run tests before making significant changes to ensure model integrity

## Vue.js Reporting System Architecture

The LUTO reporting system uses Vue.js 3 with a progressive selection pattern for data visualization.

### Progressive Selection Pattern

All reporting views follow the progressive selection pattern:

1. **Data Loading**: Use `chartRegister`/`mapRegister` from `DataService`/`MapService`
2. **Progressive Buttons**: Dynamic buttons generated from data structure keys
3. **Cascading Watchers**: Downstream selections auto-update when upstream changes
4. **Reactive Data**: `selectMapData`/`selectChartData` computed properties
5. **Data Readiness**: `mapReady`/`chartReady` computed validation

### Complete Data Structure Hierarchies

#### AREA MODULE
- **Chart Data**:
  - `Area_Ag`: `Region → Water → [series]`
  - `Area_Am`: `Region → AgMgt → Water → [series]`
  - `Area_NonAg`: `Region → [series]` (simplified, no Water level)
- **Map Data**: `Water → Landuse → Year → {img_str, bounds, min_max}`

#### ECONOMICS MODULE (Special Case - Complex Validation Required)
- **Chart Data (SINGLE FILES containing BOTH Cost & Revenue)**:
  - `Economics_Ag`: `Region → "ALL" → "ALL" → [mixed series array]` (aggregated, no Water/Landuse selection needed)
  - `Economics_Am`: `Region → "ALL" → "ALL" → [mixed series array]` (aggregated, no AgMgt selection needed)
  - `Economics_overview_Non_Ag`: `Region → [mixed series array]` (simplified)
  - **Dual Series Structure**: Cost (`id: null`) + Revenue (`id: name`) mixed in same array
  - **Chart Independence**: Cost/Revenue button does NOT affect chart data access - always shows both
- **Map Data (SEPARATE FILES for Cost vs Revenue)**:
  - `map_cost_Ag`: `Water → Landuse → Year → {img_str, bounds, min_max}`
  - `map_cost_Am`: `AgMgt → Water → Landuse → Year → {img_str, bounds, min_max}`
  - `map_cost_NonAg`: `Landuse → Year → {img_str, bounds, min_max}` (simplified)
  - `map_revenue_Ag`: `Water → Landuse → Year → {img_str, bounds, min_max}` 
  - `map_revenue_Am`: `AgMgt → Water → Landuse → Year → {img_str, bounds, min_max}`
  - `map_revenue_NonAg`: `Landuse → Year → {img_str, bounds, min_max}` (simplified)
- **Critical Implementation Details**:
  - **Different AgMgt Categories**: Cost and Revenue have DIFFERENT AgMgt categories (MAP data only):
    - **Cost AgMgt**: `"ALL"`, `"Agricultural technology (energy)"`, `"Agricultural technology (fertiliser)"`, `"Biochar (soil amendment)"`, `"Early dry-season savanna burning"`, `"Human-induced regeneration (Beef)"`, `"Human-induced regeneration (Sheep)"`, `"Methane reduction (livestock)"`
    - **Revenue AgMgt**: `"ALL"`, `"Agricultural technology (energy)"`, `"Agricultural technology (fertiliser)"`, `"Biochar (soil amendment)"`, `"Human-induced regeneration (Beef)"`, `"Human-induced regeneration (Sheep)"`, `"Methane reduction (livestock)"` (missing `"Early dry-season savanna burning"`)
  - **Chart vs Map Structure Mismatch**: Ag Mgt chart data is aggregated while map data uses AgMgt hierarchy
  - **Validation Required**: Must validate AgMgt selection exists in current Cost/Revenue data (MAP only)
  - **Combined Watcher**: Watch both `[selectCostRevenue, selectCategory]` with immediate: true
  - **Selection Reset**: Reset AgMgt selection if it doesn't exist in new data structure
  - **Safe Access**: Use optional chaining (`?.`) in selectMapData with fallback `|| {}`
  - **Chart Access**: Both Ag and Ag Mgt charts use `chartData["ALL"]["ALL"]` (no selections needed)
- **UI Pattern**: Cost/Revenue buttons affect MAP selection ONLY, charts ALWAYS show both cost & revenue series

#### GHG MODULE
- **Chart Data**:
  - `GHG_Ag`: `Region → "ALL" → Water → [series]` (no Landuse breakdown)
  - `GHG_Am`: `Region → AgMgt → Water → [series]` (has AgMgt breakdown)
  - `GHG_NonAg`: `Region → [series]` (simplified)
- **Map Data**: 
  - `map_GHG_Ag`: `Water → Landuse → Year → {img_str, bounds, min_max}`
  - `map_GHG_Am`: `AgMgt → Water → Landuse → Year → {img_str, bounds, min_max}` (AgMgt first, then Water)
  - `map_GHG_NonAg`: `Landuse → Year → {img_str, bounds, min_max}` (simplified)

#### PRODUCTION MODULE
- **Chart Data**:
  - `Production_Ag`: `Region → Water → [series]`
  - `Production_Am`: `Region → AgMgt → Water → [series]`
  - `Production_NonAg`: `Region → [series]` (simplified)
- **Map Data**: `map_quantities_*` follows standard Water → Landuse → Year pattern

#### WATER MODULE
- **Chart Data**:
  - `Water_Ag_NRM`: `Region → Water → [series]` (water property included)
  - `Water_Am_NRM`: `Region → AgMgt → Water → [series]`
  - `Water_NonAg_NRM`: `Region → [series]` (simplified)
- **Map Data**: 
  - `map_water_yield_Ag`: `Water → Landuse → Year → {img_str, bounds, min_max}`
  - `map_water_yield_Am`: `AgMgt → Water → Landuse → Year → {img_str, bounds, min_max}` (AgMgt first, then Water)
  - `map_water_yield_NonAg`: `Landuse → Year → {img_str, bounds, min_max}` (simplified)

#### BIODIVERSITY MODULE
- **Chart Data**:
  - `BIO_GBF2_overview_1_Type`: `Region → [series]` (simplified overview - Agricultural Landuse, Agricultural Management, Non-Agricultural land-use)
  - `BIO_GBF2_split_Ag_1_Landuse`: `Region → [series]` (simplified, no Water/AgMgt levels)
  - `BIO_GBF2_split_Am_1_Landuse`: `Region → [series]` (simplified, no Water/AgMgt levels) 
  - `BIO_GBF2_split_Am_2_Agri-Management`: `Region → [series]` with AgMgt categories: `"ALL"`, `"Early dry-season savanna burning"`, `"Human-induced regeneration (Beef)"`, `"Human-induced regeneration (Sheep)"`
  - `BIO_GBF2_split_NonAg_1_Landuse`: `Region → [series]` (simplified)
  - `BIO_quality_overview_1_Type`: `Region → [series]` (simplified overview)
  - `BIO_quality_split_*`: Similar structure to GBF2 files
- **Map Data**: 
  - `map_bio_GBF2_Ag`: `Water → Landuse → Year → {img_str, bounds, min_max}` (standard pattern)
  - `map_bio_GBF2_Am`: `Water → Landuse → Year → {img_str, bounds, min_max}` (standard pattern)
  - `map_bio_GBF2_NonAg`: `Landuse → Year → {img_str, bounds, min_max}` (simplified, no Water level)
  - `map_bio_overall_*`: Similar structures for overview maps

#### DVAR MODULE (Decision Variables - Map-Only Module)
- **Map Data (Simplified Hierarchy)**:
  - `map_dvar_Ag`: `Landuse → Year → {img_str, bounds, min_max}` (direct landuse access)
  - `map_dvar_Am`: `AgMgt → Year → {img_str, bounds, min_max}` (direct agmgt access)  
  - `map_dvar_NonAg`: `Landuse → Year → {img_str, bounds, min_max}` (direct landuse access)
  - `map_dvar_mosaic`: Contains overview categories:
    - `"Land-use"`: `Year → {img_str, bounds, min_max}`
    - `"Water-supply"`: `Year → {img_str, bounds, min_max}` 
    - `"Agricultural Land-use"`: `Year → {img_str, bounds, min_max}`
    - `"Agricultural Management"`: `Year → {img_str, bounds, min_max}`
    - `"Non-agricultural Land-use"`: `Year → {img_str, bounds, min_max}`
- **Composite Structure**: Map.js creates combined structure:
  - Categories: `"Land-use"`, `"Water-supply"`, `"Ag"`, `"Ag Mgt"`, `"Non-Ag"`
  - Each category combines "ALL" from mosaic + individual items from specific files
  - Final hierarchy: `Category → Landuse/AgMgt → Year → {img_str, bounds, min_max}`

### Key Patterns

#### Progressive Selection Hierarchies
1. **Standard Full**: Category → AgMgt → Water → Landuse
2. **Standard Simple**: Category → Water → Landuse  
3. **NonAg Simplified**: Category → Landuse (no Water/AgMgt levels)
4. **DVAR Simplified**: Category → Landuse/AgMgt → Year (map-only, direct access)

#### Water Level Options
- **Ag/AgMgt**: `"ALL"`, `"Dryland"`, `"Irrigated"`
- **NonAg**: No Water level (simplified structure)

#### AgMgt Options (where applicable)
- `"ALL"`, `"AgTech EI"`, `"Asparagopsis taxiformis"`, `"Biochar"`, `"Precision Agriculture"`

#### Map vs Chart Data
- **Charts**: Always end with array of series objects `[{name, data, type, color}]`
- **Maps**: Always end with object `{img_str: "base64...", bounds: [...], min_max: [...]}`

### Implementation Guidelines

1. **Data Validation**: Always check data readiness at each hierarchy level before accessing
2. **Progressive Watchers**: Use cascading watchers that clear downstream selections when upstream changes
3. **Special Cases**:
   - Economics: Handle dual Cost/Revenue series in same array
   - NonAg: Handle simplified structures without Water/AgMgt levels
   - Biodiversity: Mixed structures - most use simplified `Region → [series]`, but `BIO_*_Am_2_Agri-Management` files have AgMgt categories; map data follows standard patterns with some NonAg files simplified
4. **UI Conditions**: Use proper `v-if` conditions based on category selections
5. **Data Access**: Use optional chaining (`?.`) for safe property access

### File Structure
- **Views**: `/luto/tools/report/VUE_modules/views/` - Main view components
- **Chart Data**: `/luto/tools/report/VUE_modules/data/` - Chart data files (68 total)
- **Map Data**: `/luto/tools/report/VUE_modules/data/map_layers/` - Map layer files
- **Services**: `/luto/tools/report/VUE_modules/services/` - DataService/MapService registrations
- **Routes**: `/luto/tools/report/VUE_modules/routes/route.js` - Vue router configuration