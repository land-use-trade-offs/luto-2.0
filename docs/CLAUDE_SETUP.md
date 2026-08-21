# LUTO2 Development Setup & Configuration

This document covers environment setup, common commands, and configuration parameters.

## Development Environment Setup

### Environment Creation
```bash
# Create and activate conda environment from requirements.yml
conda env create -f requirements.yml
conda activate luto

# Note: All dependencies including gurobipy, numpy_financial, and tables are included in requirements.yml
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

## Key Configuration Parameters

### Core Settings (`luto/settings.py`)
- `VERSION`: Model version identifier (current: '2.3')
- `SSP`: Shared Socioeconomic Pathway code (e.g., '245' for SSP2-RCP4.5)
- `SCENARIO`: Auto-derived from SSP (e.g., 'SSP2')
- `RCP`: Auto-derived from SSP (e.g., 'rcp4p5')
- `SIM_YEARS`: Simulation time periods (default: `list(range(2020, 2051, 5))` = 2020-2050 in 5-year steps)
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
- `GHG_CONSTRAINT_TYPE`: Hard or soft constraint ('hard' or 'soft')
- `WATER_LIMITS`: Water yield constraints ('on' or 'off')
- `WATER_CONSTRAINT_TYPE`: Hard or soft constraint ('hard' or 'soft')
- `WATER_CLIMATE_CHANGE_IMPACT`: Apply climate change to water yields ('on' or 'off')
- `WATER_STRESS`: Historical yield requirement fraction (default: 0.6 = 60%)
- `CARBON_EFFECTS_WINDOW`: Years for carbon accumulation averaging (50, 60, 70, 80, or 90 years)
  - Must match available NetCDF data ages in input files
  - Determines annual sequestration rate by averaging total CO2 over this period
  - Default: 50 years (follows S-curve logic with rapid early accumulation)
- `BIODIVERSITY_TARGET_GBF_*`: Global Biodiversity Framework targets
  - `GBF2_TARGET`: Priority degraded areas restoration ('off', 'low', 'medium', 'high')
  - `GBF2_CONSTRAINT_TYPE`: Hard or soft constraint ('hard' or 'soft')
  - `GBF3_NVIS_TARGET`: NVIS vegetation group targets ('off', 'medium', 'high', 'USER_DEFINED')
  - `GBF3_NVIS_TARGET_CLASS`: Layer class ('NVIS_MVG' or 'NVIS_MVS'); also selects the class for IBRA layers when `GBF3_NVIS_REGION_MODE = 'IBRA_REG'`
  - `GBF3_NVIS_REGION_MODE`: 'AUSTRALIA', 'NRM', or 'IBRA_REG' (IBRA bioregion targets are handled through the NVIS stream — there is no separate `BIODIVERSITY_TARGET_GBF_3_IBRA` setting or IBRA constraint method)
  - `GBF4_TARGET_SNES`: Species NES targets ('off', 'USER_DEFINED', or 'dict')
  - `GBF4_TARGET_ECNES`: Ecological Community NES targets ('off', 'USER_DEFINED', or 'dict')
  - `GBF4_SNES_TARGETS_OVERRIDE`: dict letting a few species carry a different target from the rest (empty = no override)
  - `GBF4_SNES_CAP_MARGIN`: safety margin (percentage points, default 2.0) subtracted from each species' `ATTAINABLE_LEVEL` when clamping an interpolated SNES target, to keep a feasibility buffer (effective cap = `ATTAINABLE_LEVEL - GBF4_SNES_CAP_MARGIN`)
  - `GBF8_TARGET`: Species conservation targets ('on' or 'off')

### Renewable Energy Settings
- `RENEWABLES_OPTIONS`: Dict controlling which renewable energy types are enabled, e.g. `{'Utility Solar PV': True, 'Onshore Wind': True}`. Set values to `False` to disable individual types. Also drives the corresponding `AG_MANAGEMENTS` entries.
- `RENEWABLE_TARGET_SCENARIO_TARGETS`: Generation-target scenario (e.g. 'Gladstone - Core', 'Gladstone - BESS Sensitivity', 'AEMO 2026 ISP - Step Change', 'AEMO 2026 ISP - Accelerated Transition', 'AEMO 2026 ISP - Slower Growth')
- `RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS`: Spatial-layer scenario (e.g. 'step_change', 'accelerated_transition', 'ANU_transmission_T3/T5/T10')
- `RE_TARGET_LEVEL`: Spatial level for constraints ('STATE' or 'NRM'; only STATE currently supported)
- `INSTALL_CAPACITY_MW_HA`: Per-hectare capacity (MW/ha) per renewable type
- `RENEWABLES_ADOPTION_LIMITS`: Maximum adoption fraction per type (default: 1.0 for both)

### Solver Configuration
- Gurobi `Method` (algorithm; 2 = barrier) is not a standalone setting — it is the 2nd element of each `RETRY_PARAMS` tuple (default first attempt uses barrier). There is no `SOLVE_METHOD` setting.
- `THREADS`: Parallel threads for optimization (default: 32)
- `FEASIBILITY_TOLERANCE`: Solver feasibility tolerance (default: 1e-6)
- `OPTIMALITY_TOLERANCE`: Optimality tolerance (default: 1e-2)
- `BARRIER_CONVERGENCE_TOLERANCE`: Barrier method convergence (default: 1e-5)
- `RESCALE_FACTOR`: Rescaling magnitude for numerical stability (default: 1e3)
- `SOLVER_COEFF_MIN`: Universal minimum coefficient threshold (default: 1e-4). The `_qsum(coeffs, gurobi_vars)` helper in `solver.py` is called by **all** constraint and objective builders; any term whose absolute value falls below this threshold is dropped before entering Gurobi. Applies to Economy, Biodiversity-quality, GHG, Water, Renewable, GBF2/3/4/8, Demand/Quantity, and Regional Adoption limits. Chosen empirically: 1e-3 caused ~3% economic loss; 1e-4 retains meaningful small coefficients while keeping the matrix range ratio at 1e8 (well within Gurobi's safe zone). `RESCALE_ZERO_THRESHOLD` was removed — post-rescale zeroing is superseded by this universal filter.
  - `_qsum` floors coefficients as each term is *built*, but some coefficients are created **downstream** — e.g. the accounting term `coeff × X_acct`, where a folded-sliver `X_acct` is a `LinExpr` with `~1/RESFACTOR²` weights, distributes a kept `coeff` into sub-floor products. A single post-build sweep, `_floor_assembled_matrix(model)`, runs after `_setup_objective()` and drops `|coeff| < SOLVER_COEFF_MIN` from both the assembled constraint matrix (`getA()`) **and** the objective vector (`v.Obj`), which `getA()` can't reach. See `docs/FINDINGS.md` (20260721).

### Output Writing Configuration
- `WRITE_REPORT_MAX_MEM_MB`: Max memory (MB) for report generation (default: `64 * 1024` = 65536). Parallel-write `n_jobs` is budgeted from this: `get_n_jobs(peak_mb)` subtracts the parent `Data` object's live RSS, then divides the remaining budget by each worker's `peak_mb + ~500 MB` overhead (true per-worker cost, not a plain floor-division).
- `WRITE_CHUNK_SIZE`: Chunk size for NetCDF writing (default: 4096)

### No-Go Areas & Regional Adoption
- `EXCLUDE_NO_GO_LU`: Enforce no-go area constraints (True/False)
- `REGIONAL_ADOPTION_CONSTRAINTS`: Regional adoption limits ('off', 'on', 'NON_AG_CAP')
- `REGIONAL_ADOPTION_ZONE`: Zone type ('ABARES_AAGIS', 'LGA_CODE', 'NRM_CODE', 'IBRA_ID', 'SLA_5DIGIT')
- `REGIONAL_ADOPTION_NON_AG_CAP`: uniform sum-of-non-ag cap % per region (only under `NON_AG_CAP`)
- `REGIONAL_ADOPTION_NON_AG_CAP_REGIONS`: scope of the cap — `[]` caps all regions (default); a list of region names caps only those, leaving the rest uncapped
- `REGIONAL_ADOPTION_NON_AG_CAP_OVERRIDE`: dict of per-region cap % that override the uniform `REGIONAL_ADOPTION_NON_AG_CAP` for named regions (applied within the `_REGIONS` scope)

### Land-Use Culling
- `CULL_MODE`: Land-use culling mode ('absolute', 'percentage', 'none')
- `MAX_LAND_USES_PER_CELL`: Max land uses per cell if absolute culling (default: 12)
- `LAND_USAGE_CULL_PERCENTAGE`: Culling percentage if percentage mode (default: 0.15)

## Memory and Performance

### System Requirements
- Minimum 16GB RAM (32GB recommended for large simulations)
- Model complexity requires substantial computational resources
- Use `RESFACTOR > 1` for testing and development to reduce memory usage

### Memory Monitoring Tool

LUTO2 includes a built-in memory profiling tool (`luto.tools.mem_monitor`) with live Plotly visualization. This is essential for optimizing memory-intensive operations and identifying bottlenecks.

#### Using `@trace_mem_usage` Decorator (Recommended)

The decorator automatically manages the full monitoring lifecycle:

```python
from luto.tools.mem_monitor import trace_mem_usage

@trace_mem_usage
def write_quantity_separate(data, sim, year):
    """Memory usage is automatically tracked with live visualization."""
    # Your memory-intensive code here
    pass

# Usage - monitoring happens automatically
write_quantity_separate(data, sim, 2030)
```

**Features:**
- ✅ Automatic start/stop lifecycle management
- ✅ Live-updating Plotly visualization in Jupyter notebooks
- ✅ Graceful exception handling (monitoring stops even if function fails)
- ✅ Tracks delta memory from baseline (Working Set/RSS)
- ✅ Reports peak, final memory, and execution duration

**Advanced Usage:**

```python
# Customize plot update interval (default: 0.1s)
@trace_mem_usage(update_interval=0.5)
def slower_refresh(data):
    return process(data)

# Return memory statistics with function result
@trace_mem_usage(return_data=True)
def get_memory_stats(data):
    return process(data)

result, stats = get_memory_stats(data)
print(f"Peak: {stats['peak_memory_mb']:.2f} MB")
print(f"Duration: {stats['duration']:.2f}s")
```

#### Manual Monitoring

For monitoring multiple operations or interactive development:

```python
from luto.tools.mem_monitor import start_memory_monitor, stop_memory_monitor

start_memory_monitor(update_interval=0.1)  # Starts with live plot
# Run your code while plot updates automatically
operation1()
operation2()
stop_memory_monitor()  # Shows final summary and statistics
```

**When to Use:**
- Monitoring sequential operations
- Interactive Jupyter notebook development
- Custom profiling workflows

#### Example Output

```
Starting memory trace for: write_quantity_separate
------------------------------------------------------------
Memory monitoring started (baseline: 1234.56 MB)
Live plot active in background. Run your code normally.
[...live plot updates automatically...]

Monitoring stopped.
Duration: 45.23s | Peak: 2048.12 MB | Final: 1567.89 MB

Function 'write_quantity_separate' completed successfully.
```

#### Implementation Location

- **Module**: `luto/tools/mem_monitor.py`
- **Key Functions**:
  - `trace_mem_usage()`: Decorator for automatic monitoring
  - `start_memory_monitor()`: Manual start
  - `stop_memory_monitor()`: Manual stop with statistics

## Memory-Efficient Array Operations with xr.dot()

### Critical Optimization: Use xr.dot() Instead of Broadcasting

When working with large xarray DataArrays in LUTO2, **always use `xr.dot()` for element-wise multiplication followed by summation** instead of broadcasting operations. This is critical for memory efficiency and performance.

#### Why xr.dot() is Essential

**Broadcasting creates intermediate arrays:**
```python
# BAD - Creates large intermediate broadcasted array before summing
result = (matrix_A * matrix_B).sum(dim=['lu'])

# Intermediate steps:
# 1. matrix_A * matrix_B → Creates full broadcasted array (uses lots of memory!)
# 2. .sum(dim=['lu'])    → Reduces to final result
```

**xr.dot() computes directly:**
```python
# GOOD - Computes dot product directly without intermediate arrays
result = xr.dot(matrix_A, matrix_B, dims=['lu'])

# Single optimized operation - minimal memory overhead
```

#### Performance Impact

For typical LUTO2 operations with ~25,000 cells:
- **Memory savings**: 50-80% reduction in peak memory usage
- **Speed improvement**: 2-4x faster execution
- **Identical results**: Numerically exact (difference = 0.00e+00)

#### Usage Patterns in LUTO2

**Pattern 1: Simple dot product**
```python
# Before
commodity_production = (land_use * commodity_matrix).sum(dim=['lu'])

# After
commodity_production = xr.dot(land_use, commodity_matrix, dims=['lu'])
```

**Pattern 2: Chained operations**
```python
# Before
result = (((matrix_A * matrix_B).sum(dim=['lu']) * matrix_C) * matrix_D).sum(dim='product')

# After
result = ((xr.dot(matrix_A, matrix_B, dims=['lu']) * matrix_C) * matrix_D).sum(dim='product')
```

**Pattern 3: Multiple dimensions**
```python
# Before
profit = (profit_combo * decision_vars).sum(dim=['am', 'lm', 'lu'])

# After
profit = xr.dot(profit_combo, decision_vars, dims=['am', 'lm', 'lu'])
```

#### Where this applies

This is a **standing convention** for any large-array `(a * b).sum(dim=...)` reduction in `write.py` and the report data tools — apply it whenever you add or refactor such a reduction. (Do not rely on hard-coded line numbers here: `write.py` has been substantially rewritten — notably the transition reporting, which now aggregates solved per-source delta flows rather than a `base × target × cost` Cartesian product — so exact call sites move.)

**Special case - Dimension elimination.** When a product would create a full Cartesian intermediate over two water-supply dimensions, sum each dimension separately instead:
```python
# Instead of the huge intermediate: base[From-ws, ...] * target[To-ws, ...] * cost[To-ws, ...]
ag_base_no_ws = ag_dvar_mrj_base.sum(dim='From-water-supply')
target_cost_product = xr.dot(ag_dvar_mrj_target, ag_transitions_cost_mat, dims=['To-water-supply'])
cost_xr = ag_base_no_ws * target_cost_product
```

#### When to Use xr.dot()

Use `xr.dot()` whenever you have:
1. Element-wise multiplication between DataArrays
2. Followed by `.sum()` over one or more dimensions
3. Working with large arrays (>10,000 cells)

**Rule of thumb:** If you see `(a * b).sum(dim=...)`, replace with `xr.dot(a, b, dims=...)`

#### Validation

Always validate that xr.dot() produces identical results:
```python
# Test equivalence
broadcast_result = (matrix_A * matrix_B).sum(dim=['lu'])
dot_result = xr.dot(matrix_A, matrix_B, dims=['lu'])

# Verify
max_diff = abs(broadcast_result - dot_result).max()
print(f"Max difference: {float(max_diff):.2e}")  # Should be 0.00e+00
```

## Testing Framework

- Uses pytest with hypothesis for property-based testing
- Tests focus on robustness of core functionality
- Run tests before making significant changes to ensure model integrity
