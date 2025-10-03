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
    - Biodiversity constraint methods: `_add_GBF2_constraints()`, `_add_GBF3_NVIS_constraints()`, `_add_GBF4_SNES_constraints()`, `_add_GBF4_ECNES_constraints()`, `_add_GBF8_constraints()`
  - `input_data.py`: Prepares optimization model input data
    - Biodiversity data attributes use `*_pre_1750_area_*` naming (e.g., `GBF3_NVIS_pre_1750_area_vr`, `GBF4_SNES_pre_1750_area_sr`)
    - `rescale_solver_input_data()`: **In-place** rescaling of arrays to magnitude 0-1e3 for numerical stability

### Economic Modules
- **`luto/economics/agricultural/`**: Agricultural land use economics
  - Revenue, cost, quantity, water, biodiversity, GHG calculations
  - Transition costs between agricultural land uses
  - **Biodiversity module** (`biodiversity.py`): GBF (Global Biodiversity Framework) calculations
    - `get_GBF2_MASK_area()`: Returns GBF2 priority degraded areas (mask × real area)
    - `get_GBF3_NVIS_matrices_vr()`: NVIS vegetation layer matrices for GBF3
    - `get_GBF4_SNES_matrix_sr()`, `get_GBF4_ECNES_matrix_sr()`: Species/Ecological Community NES matrices
    - Variable naming convention: `*_pre_1750_area_*` for baseline biodiversity area matrices
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
- `BIODIVERSITY_TARGET_GBF_*`: Global Biodiversity Framework targets
  - `BIODIVERSITY_TARGET_GBF_2`: Priority degraded areas restoration ('off' or percentage target)
  - `BIODIVERSITY_TARGET_GBF_3_NVIS`: NVIS vegetation group targets ('off' or percentage target)
  - `BIODIVERSITY_TARGET_GBF_4_SNES`: Species NES (National Environmental Significance) ('on' or 'off')
  - `BIODIVERSITY_TARGET_GBF_4_ECNES`: Ecological Community NES ('on' or 'off')
  - `BIODIVERSITY_TARGET_GBF_8`: Species conservation targets ('on' or 'off')

### Solver Configuration
- `SOLVE_METHOD`: GUROBI algorithm (default: 2 for barrier method)
- `THREADS`: Parallel threads for optimization
- `FEASIBILITY_TOLERANCE`: Solver tolerance settings

## Data Flow

1. **Data Loading**: `luto.data.Data` class loads spatial datasets from `/input/`
2. **Preprocessing**: `dataprep.py` processes raw data into model-ready formats
3. **Economic Calculations**: Economics modules calculate costs, revenues, transitions, biodiversity impacts
4. **Solver Input**: `solvers/input_data.py` prepares optimization model data
   - Biodiversity matrices: GBF2 mask areas, GBF3 NVIS layers, GBF4 SNES/ECNES matrices, GBF8 species data
   - Data rescaling: Arrays rescaled in-place to 0-1e3 magnitude for numerical stability
5. **Optimization**: `solvers/solver.py` runs GUROBI optimization with biodiversity constraints
6. **Output Generation**: `tools/write.py` writes results to `/output/`
   - Biodiversity outputs: GBF2/3/4/8 scores, species impacts, vegetation group restoration

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

## Vue.js Reporting System Architecture

The LUTO reporting system uses Vue.js 3 with a progressive selection pattern for data visualization.

### Progressive Selection Pattern

All reporting views follow the progressive selection pattern:

1. **Data Loading**: Use `chartRegister`/`mapRegister` from `ChartService`/`MapService`
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

#### BIODIVERSITY MODULE (Dynamic/Conditional Loading)
- **Dynamic ChartData Structure**: Biodiversity data is conditionally loaded based on scenario settings
  - **Conditional Loading Logic**: Only load GBF scripts when corresponding targets are not 'off':
    - `BIODIVERSITY_TARGET_GBF_2 !== 'off'` → loads GBF2 data
    - `BIODIVERSITY_TARGET_GBF_3_NVIS !== 'off'` → loads GBF3 data
    - `BIODIVERSITY_TARGET_GBF_4_SNES !== 'off'` → loads GBF4 (SNES) data
    - `BIODIVERSITY_TARGET_GBF_4_ECNES !== 'off'` → loads GBF4 (ECNES) data
    - `BIODIVERSITY_TARGET_GBF_8 !== 'off'` → loads GBF8 (SPECIES & GROUP) data
- **Dynamic ChartData Construction**: Base structure created first, then GBF data added conditionally:
  ```javascript
  // Base structure always includes Quality data
  ChartData.value['Biodiversity'] = {
    'Quality': window[chartOverview_bio_quality['name']]
  };
  // Then conditionally add GBF data based on scenario settings
  if (runScenario.value['BIODIVERSITY_TARGET_GBF_2'] !== 'off') {
    ChartData.value['Biodiversity']['GBF2'] = window[chartOverview_bio_GBF2['name']];
  }
  // ... similar pattern for GBF3, GBF4, GBF8
  ```
- **Chart Data** (when loaded):
  - `BIO_quality_overview_1_Type`: `Region → [series]` (always loaded - simplified overview)
  - `BIO_GBF2_overview_1_Type`: `Region → [series]` (conditional - Agricultural Landuse, Agricultural Management, Non-Agricultural Land-use)
  - `BIO_GBF2_split_Ag_1_Landuse`: `Region → [series]` (conditional - simplified, no Water/AgMgt levels)
  - `BIO_GBF2_split_Am_1_Landuse`: `Region → [series]` (conditional - simplified, no Water/AgMgt levels)
  - `BIO_GBF2_split_Am_2_Agri-Management`: `Region → [series]` (conditional - with AgMgt categories: `"ALL"`, `"Early dry-season savanna burning"`, `"Human-induced regeneration (Beef)"`, `"Human-induced regeneration (Sheep)"`)
  - `BIO_GBF2_split_NonAg_1_Landuse`: `Region → [series]` (conditional - simplified)
  - `BIO_GBF3_*`, `BIO_GBF4_*`, `BIO_GBF8_*`: Similar structures for other GBF targets (conditional loading)
- **Map Data**:
  - `map_bio_quality_*`: Always available (quality data always loaded)
  - `map_bio_GBF2_Ag`: `Water → Landuse → Year → {img_str, bounds, min_max}` (conditional - standard pattern)
  - `map_bio_GBF2_Am`: `Water → Landuse → Year → {img_str, bounds, min_max}` (conditional - standard pattern)
  - `map_bio_GBF2_NonAg`: `Landuse → Year → {img_str, bounds, min_max}` (conditional - simplified, no Water level)
  - `map_bio_GBF3_*`, `map_bio_GBF4_*`, `map_bio_GBF8_*`: Similar structures for other GBF targets (conditional loading)
- **Implementation Notes**:
  - **Script Loading Order**: Conditional GBF scripts loaded after base scripts but before ChartData construction
  - **Error Handling**: Views must handle cases where expected GBF data may not be available
  - **UI Adaptation**: Biodiversity view buttons/options should adapt to available data structure
  - **Memory Optimization**: Only loads necessary data files based on scenario configuration

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
    - `"Non-Agricultural Land-use"`: `Year → {img_str, bounds, min_max}`
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
2. **Progressive Watchers**: Use the standardized cascade pattern (see "Progressive Selection Cascade Watchers" section below)
   - Follow the exact watcher implementation pattern from Area.js
   - Never manually clear selections - let the cascade pattern handle it automatically
3. **Special Cases**:
   - Economics: Handle dual Cost/Revenue series in same array with combined watcher pattern
   - NonAg: Handle simplified structures without Water/AgMgt levels
   - Biodiversity: **Dynamic/Conditional Loading** - GBF data conditionally loaded based on scenario settings; Quality data always available; mixed structures where most use simplified `Region → [series]`, but `BIO_*_Am_2_Agri-Management` files have AgMgt categories; map data follows standard patterns with some NonAg files simplified; views must adapt to potentially missing GBF data
4. **UI Conditions**: Use proper `v-if` conditions based on category selections
5. **Data Access**: Use optional chaining (`?.`) for safe property access
6. **Code Consistency**: All views must follow the same cascade watcher pattern for maintainability

### Progressive Selection Cascade Watchers

All Vue.js reporting views implement a standardized cascade pattern for progressive selection that automatically handles downstream option updates when upstream selections change.

#### Core Pattern (Area.js, Biodiversity.js, GHG.js, Production.js, Water.js)

**Watch Order**: `selectCategory` → `selectWater` → `selectAgMgt` → `selectLanduse`

```javascript
// 1. Category watcher handles ALL downstream cascading
watch(selectCategory, (newCategory, oldCategory) => {
  // Save previous selections before switching
  if (oldCategory) {
    if (oldCategory === "Ag") {
      previousSelections.value["Ag"] = { water: selectWater.value, landuse: selectLanduse.value };
    } else if (oldCategory === "Ag Mgt") {
      previousSelections.value["Ag Mgt"] = { agMgt: selectAgMgt.value, water: selectWater.value, landuse: selectLanduse.value };
    } else if (oldCategory === "Non-Ag") {
      previousSelections.value["Non-Ag"] = { landuse: selectLanduse.value };
    }
  }

  // Handle ALL downstream variables with cascading pattern
  if (newCategory === "Ag Mgt") {
    availableAgMgt.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]] || {});
    const prevAgMgt = previousSelections.value["Ag Mgt"].agMgt;
    selectAgMgt.value = (prevAgMgt && availableAgMgt.value.includes(prevAgMgt)) ? prevAgMgt : (availableAgMgt.value[0] || '');
    
    availableWater.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][selectAgMgt.value] || {});
    const prevWater = previousSelections.value["Ag Mgt"].water;
    selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
    
    availableLanduse.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][selectAgMgt.value][selectWater.value] || {});
    const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
    selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
  } else if (newCategory === "Ag") {
    availableWater.value = Object.keys(window[mapRegister["Ag"]["name"]] || {});
    const prevWater = previousSelections.value["Ag"].water;
    selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
    
    availableLanduse.value = Object.keys(window[mapRegister["Ag"]["name"]][selectWater.value] || {});
    const prevLanduse = previousSelections.value["Ag"].landuse;
    selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
  } else if (newCategory === "Non-Ag") {
    availableLanduse.value = Object.keys(window[mapRegister["Non-Ag"]["name"]] || {});
    const prevLanduse = previousSelections.value["Non-Ag"].landuse;
    selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
  }
});

// 2. Water watcher handles downstream landuse cascading
watch(selectWater, (newWater) => {
  // Save current water selection
  if (selectCategory.value === "Ag") {
    previousSelections.value["Ag"].water = newWater;
  } else if (selectCategory.value === "Ag Mgt") {
    previousSelections.value["Ag Mgt"].water = newWater;
  }

  // Handle ALL downstream variables
  if (selectCategory.value === "Ag") {
    availableLanduse.value = Object.keys(window[mapRegister["Ag"]["name"]][newWater] || {});
    const prevLanduse = previousSelections.value["Ag"].landuse;
    selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
  } else if (selectCategory.value === "Ag Mgt") {
    availableLanduse.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][selectAgMgt.value][newWater] || {});
    const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
    selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
  }
});

// 3. AgMgt watcher handles downstream water + landuse cascading  
watch(selectAgMgt, (newAgMgt) => {
  // Save current agMgt selection
  if (selectCategory.value === "Ag Mgt") {
    previousSelections.value["Ag Mgt"].agMgt = newAgMgt;
    
    // Handle ALL downstream variables with cascading pattern
    availableWater.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][newAgMgt] || {});
    const prevWater = previousSelections.value["Ag Mgt"].water;
    selectWater.value = (prevWater && availableWater.value.includes(prevWater)) ? prevWater : (availableWater.value[0] || '');
    
    availableLanduse.value = Object.keys(window[mapRegister["Ag Mgt"]["name"]][newAgMgt][selectWater.value] || {});
    const prevLanduse = previousSelections.value["Ag Mgt"].landuse;
    selectLanduse.value = (prevLanduse && availableLanduse.value.includes(prevLanduse)) ? prevLanduse : (availableLanduse.value[0] || 'ALL');
  }
});

// 4. Landuse watcher only saves selection (no downstream)
watch(selectLanduse, (newLanduse) => {
  // Save current landuse selection
  if (selectCategory.value === "Ag") {
    previousSelections.value["Ag"].landuse = newLanduse;
  } else if (selectCategory.value === "Ag Mgt") {
    previousSelections.value["Ag Mgt"].landuse = newLanduse;
  } else if (selectCategory.value === "Non-Ag") {
    previousSelections.value["Non-Ag"].landuse = newLanduse;
  }
});
```

#### Special Pattern (Economics.js)

Economics uses a combined watcher pattern due to its dual Cost/Revenue structure:

```javascript
// Combined watcher for Cost/Revenue + Category changes
watch([selectCostRevenue, selectCategory], ([newCostRevenue, newCategory], [oldCostRevenue, oldCategory]) => {
  if (!newCategory) return;

  // Save previous selections before switching (only when category changes)
  if (oldCategory && oldCategory !== newCategory) {
    // ... save previous selections
  }
  
  // Handle cascading based on current Cost/Revenue selection
  if (newCategory === "Ag Mgt") {
    const currentMapData = window[mapRegister[newCostRevenue]["Ag Mgt"]["name"]];
    // ... cascade all downstream selections using currentMapData
  }
  // ... other categories
}, { immediate: true });
```

#### Key Principles

1. **No Manual Clearing**: NEVER manually clear arrays or selections (e.g., `availableAgMgt.value = []`)
   - The progressive pattern handles this automatically
   - Manual clearing creates unnecessary complexity

2. **Cascading Flow**: Each watcher handles ALL its downstream selections
   - `selectCategory` → handles AgMgt, Water, Landuse
   - `selectAgMgt` → handles Water, Landuse  
   - `selectWater` → handles Landuse
   - `selectLanduse` → no downstream (just saves)

3. **Previous Selection Memory**: Always try to restore previous valid selections
   ```javascript
   const prevSelection = previousSelections.value[category].field;
   selectField.value = (prevSelection && availableOptions.includes(prevSelection)) 
     ? prevSelection 
     : (availableOptions[0] || 'ALL');
   ```

4. **Data Structure Consistency**: Use the map data structure as the source of truth for available options
   ```javascript
   availableOptions.value = Object.keys(window[mapRegister[category]["name"]] || {});
   ```

#### Benefits

- **Maintainable**: Consistent pattern across all views
- **Memory Efficient**: No unnecessary data structures or operations
- **User Friendly**: Preserves user selections when switching between categories
- **Robust**: Handles edge cases with fallback selections
- **Clean**: Eliminates complex conditional clearing logic

### File Structure
- **Views**: `/luto/tools/report/VUE_modules/views/` - Main view components
- **Chart Data**: `/luto/tools/report/VUE_modules/data/` - Chart data files (68 total)
- **Map Data**: `/luto/tools/report/VUE_modules/data/map_layers/` - Map layer files
- **Services**: `/luto/tools/report/VUE_modules/services/` - ChartService/MapService registrations
- **Routes**: `/luto/tools/report/VUE_modules/routes/route.js` - Vue router configuration