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
- `SIM_YEARS`: Simulation time periods (default: 2020-2050 in 5-year steps)
- `RESFACTOR`: Spatial resolution factor (1 = full resolution, >1 = coarser)
- `SCENARIO`: Shared Socioeconomic Pathway (SSP1-SSP5)
- `RCP`: Representative Concentration Pathway (e.g., 'rcp4p5')
- `OBJECTIVE`: Optimization objective ('maxprofit' or 'mincost')

### Economic Settings
- `DYNAMIC_PRICE`: Enable demand elasticity-based dynamic pricing (default: False)
- `AMORTISE_UPFRONT_COSTS`: Whether to amortize establishment costs (default: False)
- `DISCOUNT_RATE`: Discount rate for economic calculations (default: 0.07)
- `AMORTISATION_PERIOD`: Period for cost amortization in years (default: 30)

### Environmental Constraints
- `GHG_EMISSIONS_LIMITS`: Greenhouse gas targets ('off', 'low', 'medium', 'high')
- `WATER_LIMITS`: Water yield constraints ('on' or 'off')
- `CARBON_EFFECTS_WINDOW`: Years for carbon accumulation averaging (50, 60, 70, 80, or 90 years)
  - Must match available NetCDF data ages in input files
  - Determines annual sequestration rate by averaging total CO2 over this period
  - Default: 50 years (follows S-curve logic with rapid early accumulation)
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

#### Implementation Locations

Key locations where xr.dot() is used in LUTO2:

- [write.py:424](luto/tools/write.py#L424) - Commodity production calculations
- [write.py:652](luto/tools/write.py#L652) - Agricultural profit aggregation
- [write.py:758](luto/tools/write.py#L758) - Non-agricultural profit aggregation
- [write.py:840](luto/tools/write.py#L840) - Agricultural management profit
- [write.py:1266](luto/tools/write.py#L1266) - Ag-to-ag transition costs (dimension elimination)
- [write.py:1987](luto/tools/write.py#L1987) - GHG emissions calculation

**Special case - Dimension elimination** (line 1266):
```python
# Instead of creating full Cartesian product of water-supply dimensions:
# cost_xr = base[From-ws, ...] * target[To-ws, ...] * cost[To-ws, ...]
# Which creates huge intermediate: [From-ws × To-ws × ...] (~5GB)

# Sum each water-supply dimension separately:
ag_base_no_ws = ag_dvar_mrj_base.sum(dim='From-water-supply')
target_cost_product = xr.dot(ag_dvar_mrj_target, ag_transitions_cost_mat, dims=['To-water-supply'])
cost_xr = ag_base_no_ws * target_cost_product
# Result: [From-lu × To-lu × Type] (~1.6GB, 68% memory reduction)
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
