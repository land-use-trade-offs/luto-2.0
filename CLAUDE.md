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
- Special cases (Economics dual series, Biodiversity conditional loading)
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

### Key File Locations

- **Core**: `luto/simulation.py`, `luto/data.py`, `luto/settings.py`
- **Solvers**: `luto/solvers/solver.py`, `luto/solvers/input_data.py`
- **Economics**: `luto/economics/agricultural/`, `luto/economics/non_agricultural/`
- **Output**: `luto/tools/write.py`, `luto/tools/report/create_report_layers.py`
- **Vue.js**: `luto/tools/report/VUE_modules/views/`, `luto/tools/report/VUE_modules/data/`

### Data Flow Summary

1. **Load** (`data.py`) → 2. **Preprocess** (`dataprep.py`) → 3. **Economics** (economic modules) → 4. **Solver Input** (`input_data.py`) → 5. **Optimize** (`solver.py`) → 6. **Output** (`write.py`)

### Output Structure

Results saved in `/output/<timestamp>/`:
- `DATA_REPORT/REPORT_HTML/index.html`: Interactive dashboard
- NetCDF files: Spatial outputs (xarray format)
- CSV files: Data tables
- Logs: Execution logs and metrics

## Important Conventions

### Naming Patterns

- **Biodiversity variables**: `*_pre_1750_area_*` for baseline matrices
- **GBF functions**: `_add_GBF{N}_{TYPE}_constraints()`, `get_GBF{N}_*()`
- **Carbon files**: `tCO2_ha_{ep,cp,hir}_{block,belt,rip}.nc`

### NetCDF Dimensions

- **Ag**: `lm[ALL,dry,irr] → lu[ALL,...] → year → cell`
- **Am**: `am[ALL,...] → lm[ALL,dry,irr] → lu[ALL,...] → year → cell`
- **NonAg**: `lu[ALL,...] → year → cell`

### JSON Output Hierarchies (Map vs Chart)

**IMPORTANT**: Map and Chart JSON files have different dimension hierarchies:

**Map JSON (Spatial Layers)**:
- **Ag**: `lm → lu → source (if applicable for GHG/Economics) → year`
- **Am**: `am → lm → lu → source (if applicable) → year`
- **NonAg**: `lu → year`

**Chart JSON (Time Series)**:
- **Ag**: `region → lm → lu` (array of series)
- **Am**: `region → lm → lu → source (if applicable) → am` (array of series)
- **NonAg**: `region → lu` (array of series)

**Key Difference**: Map JSON places `source` before `year`, while Chart JSON places `source` before the final series array (Am only). See [CLAUDE_OUTPUT.md](docs/CLAUDE_OUTPUT.md) for detailed examples.

### Vue.js Progressive Selection Hierarchies

- **Standard Full**: Category → AgMgt → Water → Landuse
- **Standard Simple**: Category → Water → Landuse
- **NonAg Simplified**: Category → Landuse
- **DVAR Simplified**: Category → Landuse/AgMgt → Year

## Getting Started

1. **New to the project?** Start with [CLAUDE_SETUP.md](docs/CLAUDE_SETUP.md) for environment setup
2. **Working on core model logic?** See [CLAUDE_ARCHITECTURE.md](docs/CLAUDE_ARCHITECTURE.md)
3. **Working on output generation?** See [CLAUDE_OUTPUT.md](docs/CLAUDE_OUTPUT.md)
4. **Working on the reporting UI?** See [CLAUDE_VUE_REPORTING.md](docs/CLAUDE_VUE_REPORTING.md)

**Remember**: Only read the documentation file relevant to your current task to minimize memory usage!
