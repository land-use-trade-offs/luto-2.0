# LUTO2 Development Setup & Configuration

This document covers environment setup, common commands, and configuration parameters.

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

- Minimum 16GB RAM (32GB recommended for large simulations)
- Model complexity requires substantial computational resources
- Use `RESFACTOR > 1` for testing and development to reduce memory usage
- Monitor memory usage with built-in logging utilities

## Testing Framework

- Uses pytest with hypothesis for property-based testing
- Tests focus on robustness of core functionality
- Run tests before making significant changes to ensure model integrity
