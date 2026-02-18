# LUTO2: The Land-Use Trade-Offs Model Version 2.0

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/Version-2.0-green.svg)](https://github.com/land-use-trade-offs/luto-2.0)

## Introduction
The Land Use Trade-Offs model v2 (LUTO2) is an integrated land systems model designed to simulate the optimal spatial arrangement of land use and land management decisions over time in Australia. It aims to achieve climate and biodiversity targets without compromising economic growth, food production or water security. The model is implemented as a Python package, offering users the flexibility to run interactively or to execute batch processes through scripted automation.

LUTO2 was developed through a collaboration between Deakin University and Climateworks Centre, with research contributions from CSIRO. The model is a cornerstone of Climateworks’ Land Use Futures program, which supports Australia’s transition to sustainable food and land systems. The technical development of LUTO2 is led by Professor Brett Bryan at Deakin University. LUTO2 continues the approach to land-use change modelling of its predecessor, the original LUTO, which was developed by CSIRO from 2010 - 2015 (see also Pedigree, below) and published under the GNU GPLv3 in 2021.

## Pedigree
LUTO2 builds on the approach and pedigree of nearly two decades of land-use modelling expertise starting with the original LUTO model. The original LUTO model was developed by CSIRO for the Australian National Outlook in 2015 and was groundbreaking for quantifying and projecting land use changes and their sustainability impacts in Australia, illustrated by its published works in *Nature* in 2015 and 2017.

LUTO2 represents a generational leap in sophistication and functionality for national-scale land-use change modelling in Australia. Both LUTO versions are optimisation models but different commercial solvers are used (CPLEX in original LUTO, GUROBI in LUTO2). The spatial domains are different in extent, with LUTO2's being nearly 5 times as large. The data requirements to run LUTO2 are consequently different and heavier. There is no backwards compatibility whatsoever.

The original LUTO model is available online and should be cited as:
> Bryan, Brett; Nolan, Martin; Stock, Florian; Graham, Paul; Dunstall, Simon; Ernst, Andreas; Connor, Jeff (2021): Land Use Trade-Offs (LUTO) Model. v1. CSIRO. Software Collection. https://doi.org/10.25919/y8ee-sk45.

This new version represents an entirely new model featuring a complete rewrite of the codebase and comprehensive upgrades to data and functionality. Enhancements to the original model include extended spatial coverage and timespan (2010 to 2100), a complete refresh of input data, additional land-use options and sustainability indicators and management solutions, the ability to model demand-side solutions, and additional environmental indicators and reporting. Due to LUTO2’s model complexity, the computational requirements to run the model are far more intensive.

LUTO2’s modelling approach, indicators and solutions have been guided by extensive stakeholder consultation (documented here: https://doi.org/10.1007/s11625-024-01574-7) following principles of participatory model co-design.

## Authors
Coordinating lead author: **Bryan, B.A.**  

Lead authors (in order of contribution): **Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H.**  

Other significant contributors (in alphabetical order): **Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., Thiruvady, D.R.**

## Documentation
Documentation, including instructions on how to set up and run LUTO2, can be found at `docs/luto2-overview.pdf`.

LUTO2 comes with a full diagram to illustrate its data preparation, workflow, and code logics. The diagram link can be found in this link.  
*Replace with updated documentation currently in preparation when ready.*

## Project Structure

The LUTO2 codebase is organized into the following structure:

```
luto/                                    # Main package directory
├── data.py                              # Core data management and loading
├── simulation.py                        # Main simulation engine
├── settings.py                          # Configuration parameters
├── dataprep.py                          # Data preprocessing utilities
├── helpers.py                           # Utility functions
├── economics/                           # Economic models and calculations
│   ├── agricultural/                    # Agricultural economics modules
│   │   ├── biodiversity.py              # Biodiversity calculations
│   │   ├── cost.py                      # Cost calculations
│   │   ├── ghg.py                       # GHG emissions calculations
│   │   ├── quantity.py                  # Production quantity calculations
│   │   ├── revenue.py                   # Revenue calculations (includes dynamic pricing)
│   │   ├── transitions.py               # Land use transition costs
│   │   └── water.py                     # Water yield calculations
│   ├── non_agricultural/                # Non-agricultural economics modules
│   │   ├── biodiversity.py              # Non-ag biodiversity impacts
│   │   ├── cost.py                      # Non-ag establishment costs
│   │   ├── ghg.py                       # Non-ag GHG calculations
│   │   ├── quantity.py                  # Non-ag production quantities
│   │   ├── revenue.py                   # Non-ag revenue streams
│   │   ├── transitions.py               # Non-ag transition costs
│   │   └── water.py                     # Non-ag water impacts
│   ├── off_land_commodity/              # Off-land commodity economics
│   └── land_use_culling.py              # Land use optimization culling
├── solvers/                             # Optimization solvers and algorithms
│   ├── input_data.py                    # GUROBI solver input preparation
│   └── solver.py                        # GUROBI solver interface
├── tests/                               # Unit and integration tests
└── tools/                               # Utility tools and scripts
    ├── create_task_runs/                # Task execution and batch processing
    │   ├── bash_scripts/                # Shell scripts and conda environment
    │   ├── create_grid_search_tasks.py  # Grid search task generation
    │   ├── helpers.py                   # Task run utilities
    │   └── parameters.py                # Task run parameters
    ├── Manual_jupyter_books/            # Documentation notebooks
    │   ├── helpers/                     # Notebook helper functions
    │   └── asset/                       # Notebook assets and data descriptions
    ├── report/                          # Reporting and visualization system
    │   ├── VUE_modules/                 # Vue.js 3 interactive reporting dashboard
    │   │   ├── components/              # Reusable Vue components
    │   │   ├── data/                    # Chart data files
    │   │   │   ├── chart_option/        # Chart configuration options
    │   │   │   ├── geo/                 # Geographic boundary data
    │   │   │   └── map_layers/          # Map layer data
    │   │   ├── dataTransform/           # Data transformation scripts
    │   │   ├── lib/                     # JavaScript libraries (Vue, Leaflet, Highcharts)
    │   │   ├── routes/                  # Vue router configuration
    │   │   ├── services/                # Data and map services
    │   │   ├── views/                   # Vue view components (11 modules)
    │   │   ├── index.html               # Main HTML entry point
    │   │   └── index.js                 # Vue application entry
    │   ├── Assets/                      # Color schemes and styling assets
    │   ├── data_tools/                  # Data processing for reports
    │   ├── map_tools/                   # Spatial visualization utilities
    │   ├── create_report_data.py        # Generate chart data files
    │   └── create_report_layers.py      # Generate map layer files
    ├── plotmap.py                       # Mapping utilities
    ├── spatializers.py                  # Spatial data processing and upsampling
    └── write.py                         # Output writing functions
    
input/                                   # Input data directory (requires separate download)
output/                                  # Simulation outputs with interactive HTML reports
docs/                                    # Documentation files
requirements.yml                        # Python package dependencies (conda environment spec)
pyproject.toml                           # Project configuration
```

## Memory Profiling and Monitoring

LUTO2 includes a built-in memory monitoring tool (`luto.tools.mem_monitor`) for tracking memory usage with live visualization. This is particularly useful for optimizing memory-intensive functions and identifying memory bottlenecks.

### Using the Memory Monitor as a Decorator

The **recommended way** to monitor memory usage is using the `@trace_mem_usage` decorator. It automatically handles starting, monitoring, and cleanup:

```python
from luto.tools.mem_monitor import trace_mem_usage

@trace_mem_usage
def my_expensive_function(data):
    """This function's memory usage will be automatically monitored."""
    result = process_large_data(data)
    return result

# Usage - monitoring happens automatically
result = my_expensive_function(my_data)
```


## Troubleshooting

### Common Issues

**Memory Errors:**
- Ensure you have at least 32 GB RAM available
- Close other applications during simulation
- Consider running smaller scenarios first
- Use the memory monitor to identify memory-intensive operations

**GUROBI License Issues:**
- Verify your license file location
- Check license expiration date
- Ensure your license supports the required model size

**Data Loading Errors:**
- Verify all required input files are present in `/input/`
- Check file permissions
- Ensure sufficient disk space


### Getting Help

1. Check the documentation in `docs/luto2-overview.pdf`
2. Review log files in `/output/<run_dir>/logs/`
3. Contact the development team: **b.bryan@deakin.edu.au**
4. Submit issues on GitHub: [github.com/land-use-trade-offs/luto-2.0](https://github.com/land-use-trade-offs/luto-2.0)

## System Requirements

**Minimum Requirements:**
- Python 3.10 or higher
- 16 GB RAM (32 GB recommended for large simulations)
- 50 GB available disk space for input data and outputs
- GUROBI optimization solver license (academic licenses available)

**Supported Operating Systems:**
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+, CentOS 7+)

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/land-use-trade-offs/luto-2.0.git
cd luto-2.0
```

### 2. Set Up Environment

```bash
# Create and activate the LUTO environment from requirements.yml
conda env create -f requirements.yml
conda activate luto
```

### 3. Configure GUROBI Solver
LUTO2 requires GUROBI for optimization. Follow these steps:
```bash
# 1) Set up your GUROBI license (academic license available at gurobi.com)
# 2) Place your gurobi.lic file in the appropriate directory
```

### 4. Obtain Input Data
The LUTO2 input database is approximately 40 GB and contains sensitive data. 
Please contact **b.bryan@deakin.edu.au** to request access to the input dataset.


## Running LUTO2

### Basic Simulation
```python
import luto.simulation as sim

# Load input data and settings
data = sim.load_data()

# Run simulation with default parameters
results = sim.run(data=data)
```

### Advanced Configuration
```python
import luto.simulation as sim
import luto.settings as settings

# Customize simulation settings
settings.RESFACTOR = 10                                 # 10 makes the spatial resolution to ~10km. 
settings.SIM_YEARS = [2010, 2020, 2030, 2040, 2050]

settings.WATER_LIMITS = 'on'                            # 'on' or 'off'.
settings.GHG_EMISSIONS_LIMITS = 'high'                  # 'off', 'low', 'medium', or 'high'
settings.BIODIVERSITY_TARGET_GBF_2 = 'high'             # 'off', 'low', 'medium', or 'high'
settings.BIODIVERSITY_TARGET_GBF_3_NVIS = 'off'         # 'off', 'medium', 'high', or 'USER_DEFINED'
settings.BIODIVERSITY_TARGET_GBF_3_IBRA = 'off'         # 'off', 'medium', 'high', or 'USER_DEFINED'
settings.BIODIVERSITY_TARGET_GBF_4_SNES = 'off'         # 'on' or 'off'
settings.BIODIVERSITY_TARGET_GBF_4_ECNES = 'off'        # 'on' or 'off'
settings.BIODIVERSITY_TARGET_GBF_8 = 'off'              # 'on' or 'off'

settings.DYNAMIC_PRICE = False                          # Enable demand elasticity-based dynamic pricing

settings.RENEWABLE_ENERGY_CONSTRAINTS = 'on'             # Enable renewable energy targets
settings.RENEWABLE_TARGET_SCENARIO = 'CNS25 - Accelerated Transition'  # Target scenario

# Load data with custom parameters
data = sim.load_data()

# Run simulation
sim.run(data=data)
```

### Viewing Results
After execution, results are saved in the `/output/<timestamp>/` directory:

1. **Interactive HTML Dashboard:** 
   ```
   /output/<run_dir>/DATA_REPORT/index.html
   ```
   A Vue.js 3 based interactive dashboard featuring:
   - **Multi-module Analysis:** Area, Economics, GHG, Production, Water, Biodiversity
   - **Progressive Data Selection:** Region → Category → Water/AgMgt → Landuse hierarchies
   - **Dual Visualization:** Charts (Highcharts) and Maps (Leaflet) for all data types
   - **Dynamic Filtering:** Responsive dropdowns with cascading selection updates
   - **Export Capabilities:** Chart and map export functionality
   - **11 Specialized Views:** Individual modules for detailed analysis

2. **Raw Data Outputs:**
   - **NetCDF Files:** Spatial datasets (`.nc`) for each year and variable
   - **CSV Files:** Tabular data summaries for regional analysis

3. **Execution Logs:** 
   - `LUTO_RUN__stdout.log`: Standard output logs
   - `LUTO_RUN__stderr.log`: Error and warning logs
   - Memory usage logs for performance monitoring

## Configuration

LUTO2 behavior can be customized through the `luto.settings` module. Key parameters include:

### Core Simulation Parameters
- `SIM_YEARS`: Simulation time period (default: 2020-2050 in 5-year steps)
- `SCENARIO`: Shared Socioeconomic Pathway (SSP1-SSP5)
- `RCP`: Representative Concentration Pathway (e.g., 'rcp4p5')
- `OBJECTIVE`: Optimization objective ('maxprofit' or 'mincost')

### Environmental Constraints
- `GHG_EMISSIONS_LIMITS`: Greenhouse gas emission targets ('off', 'low', 'medium', 'high')
- `WATER_LIMITS`: Whether to enforce water yield constraints ('on' or 'off')
- `CARBON_EFFECTS_WINDOW`: Years for carbon accumulation averaging (50, 60, 70, 80, or 90 years)
  - Determines the time period over which carbon sequestration is averaged
  - Must match available ages in NetCDF input data
  - Default: 50 years (based on S-curve carbon accumulation pattern)
- `BIODIVERSITY_TARGET_GBF_2`: Global Biodiversity Framework Target 2 ('off', 'low', 'medium', 'high')
- `BIODIVERSITY_TARGET_GBF_3_NVIS`: Conservation targets for NVIS vegetation types ('off', 'medium', 'high', 'USER_DEFINED')
- `BIODIVERSITY_TARGET_GBF_3_IBRA`: Conservation targets for IBRA bioregions ('off', 'medium', 'high', 'USER_DEFINED')
- `BIODIVERSITY_TARGET_GBF_4_SNES`: Species of National Environmental Significance ('on' or 'off')
- `BIODIVERSITY_TARGET_GBF_4_ECNES`: Ecological Communities of National Environmental Significance ('on' or 'off')
- `BIODIVERSITY_TARGET_GBF_8`: Species and group targets ('on' or 'off')

### Renewable Energy Constraints
- `RENEWABLE_ENERGY_CONSTRAINTS`: Enable/disable renewable energy generation targets ('on' or 'off')
- `RENEWABLES_OPTIONS`: List of renewable energy types (`['Utility Solar PV', 'Onshore Wind']`)
- `RENEWABLE_TARGET_SCENARIO`: Target scenario ('CNS25 - Accelerated Transition' or 'CNS25 - Current Targets')
- `RE_TARGET_LEVEL`: Spatial level for targets ('STATE' or 'NRM')
- `INSTALL_CAPACITY_MW_HA`: Per-hectare generation capacity (MW/ha) for each renewable type
- `RENEWABLES_ADOPTION_LIMITS`: Maximum fraction of compatible land available for each renewable type (default: 1.0)

### Land Use Options
- `NON_AG_LAND_USES`: Enable/disable non-agricultural land uses (Environmental Plantings, Carbon Plantings, etc.)
- `AG_MANAGEMENTS`: Enable/disable agricultural management practices (Precision Agriculture, Biochar, Utility Solar PV, Onshore Wind, etc.)
- `EXCLUDE_NO_GO_LU`: Whether to exclude certain land uses from specific areas

### Economic Parameters
- `DYNAMIC_PRICE`: Enable demand elasticity-based dynamic pricing (default: False)
- `CARBON_PRICES_FIELD`: Carbon pricing scenario ('Default', 'CONSTANT', etc.)
- `AMORTISE_UPFRONT_COSTS`: Whether to amortize establishment costs (default: False)
- `DISCOUNT_RATE`: Discount rate for economic calculations (default: 7%)
- `AMORTISATION_PERIOD`: Period for cost amortization in years (default: 30)

### Solver Configuration
- `SOLVE_METHOD`: GUROBI algorithm selection (default: 2 for barrier method)
- `THREADS`: Number of parallel threads for optimization
- `FEASIBILITY_TOLERANCE`: Solver tolerance settings
- `VERBOSE`: Control solver output verbosity

### Output Control
- `WRITE_PARALLEL`: Use parallel processing for output generation
- `RESFACTOR`: Spatial resolution factor (1 = full resolution, >1 = coarser)

Refer to `luto/settings.py` for a complete list of configurable parameters and detailed descriptions.

## Data Formats

### Carbon Sequestration Data (NetCDF)

LUTO2 uses NetCDF format with xarray for carbon sequestration data, replacing the previous HDF5/pandas format. This provides better performance, compression, and flexibility.

**Key Features:**
- **Format**: NetCDF (.nc) files with dimensions: `age` × `cell`
- **Available ages**: 50, 60, 70, 80, 90 years (selected from full carbon accumulation timeseries)
- **Components**: Trees (aboveground biomass), Debris (litter), Soil (belowground)
- **Compression**: zlib level 5 with chunking for efficient storage and loading
- **File naming**: `tCO2_ha_{type}.nc` (e.g., `tCO2_ha_ep_block.nc` for Environmental Plantings Block)

**Planting Types:**
- Environmental Plantings: Block, Belt, Riparian (ep_block, ep_belt, ep_rip)
- Carbon Plantings: Block, Belt (cp_block, cp_belt)
- Human-Induced Regeneration: Block, Riparian (hir_block, hir_rip)

**Carbon Calculation:**
The model loads NetCDF data at the age specified by `CARBON_EFFECTS_WINDOW` setting:
- Aboveground carbon (Trees + Debris) is discounted by fire risk and reversal risk
- Belowground carbon (Soil) is not risk-discounted
- Total sequestration is averaged over the carbon effects window to get annual rate

**Example:**
```python
settings.CARBON_EFFECTS_WINDOW = 50  # Use 50-year carbon accumulation data
# Model will load NetCDF data at age=50 and average to get annual sequestration rate
```

For more technical details, see the "Carbon Sequestration Data Format" section in `CLAUDE.md`.

## Copyright
Copyright 2024-now **Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.**  

Copyright 2021-2023 **Fjalar J. de Haan and Brett A. Bryan, Deakin University.** (see `CITATION.cff`).

## License
LUTO2 is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the **Free Software Foundation**, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **without any warranty**; without even the implied warranty of **merchantability** or **fitness for a particular purpose**. See the **GNU General Public License** for more details.

You should have received a copy of the **GNU General Public License** along with this program. If not, see <https://www.gnu.org/licenses/>.

## Citation
> Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., van Schoten, N., Hadjikakou, M., Sanson, J., Zyngier, R., Marcos-Martinez, R., Navarro, J., Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R. (2025). The Land-Use Trade-Offs Model Version 2 (LUTO2): an integrated land system model for Australia. Software Collection. https://github.com/land-use-trade-offs/luto-2.0

## Contributing

We welcome contributions to LUTO2! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed modifications.

## Acknowledgments

LUTO2 was developed through a collaboration between:
- **Deakin University** - Centre for Integrative Ecology
- **Climateworks Centre** - Land Use Futures program
- **CSIRO** - Research contributions

This work is supported by funding from various Australian research councils and industry partners. We acknowledge the traditional custodians of the lands on which this research was conducted.

