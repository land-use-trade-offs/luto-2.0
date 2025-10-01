# Repository Guidelines

## Project Overview
LUTO 2.0 (LUTO2) is an integrated land systems optimisation model for Australia. It explores spatial land-use decisions that meet climate and biodiversity targets while maintaining economic outcomes. The codebase is a complete rewrite of the original CSIRO LUTO model and runs GUROBI-backed optimisation over large national datasets. Development is led by Deakin University's Centre for Integrative Ecology in collaboration with Climateworks Centre and CSIRO, with releases distributed under GPLv3.

## Data & Inputs
- Required inputs (~40GB) cover biophysical baselines, demand scenarios, and sustainability indicators; contact b.bryan@deakin.edu.au for access and place the packages under `input/`.
- Follow the sequence documented in `DataFlow.md`: raw assets -> staging scripts -> `luto/dataprep.py` transformations -> `luto.data.Data` loading -> solver assembly -> `luto/tools/write.py` outputs.
- Respect risk discounting, sign conventions, and units noted in the flow doc; keep bulky intermediate artefacts in `output/` rather than the repo.
- Use `settings.RESFACTOR > 1` when experimenting to downsample spatial resolution and control memory usage. Production runs typically assume >=32GB RAM; monitor `LUTO_RUN__stderr.log` for solver diagnostics.

## Simulation & Optimisation Workflow
- `luto/simulation.py` orchestrates end-to-end runs: it pulls `settings`, loads data via `luto.data.Data`, prepares inputs through `luto/solvers/input_data.py`, and invokes `luto/solvers/solver.py` (GUROBI).
- Economics subpackages (`luto/economics/agricultural`, `non_agricultural`, `off_land_commodity`) compute revenues, costs, water, biodiversity, and transition penalties before the optimiser is built.
- Tuning knobs live in `luto/settings.py`. Key levers include `SIM_YEARS`, `SCENARIO`, `RCP`, `OBJECTIVE`, emissions and biodiversity targets, and solver parameters such as `SOLVE_METHOD` and `THREADS`.
- Outputs populate timestamped folders under `output/`: NetCDF and CSV datasets, interactive Vue dashboard at `DATA_REPORT/REPORT_HTML/index.html`, and execution logs (`LUTO_RUN__stdout.log`, `LUTO_RUN__stderr.log`).

## Reporting Pipeline
- Vue 3 modules in `luto/tools/report/` implement a progressive selection pattern defined in `CLAUDE.md`.
- Watchers cascade selections (Category -> AgMgt -> Water -> Landuse) while preserving previous valid choices; never clear arrays manually.
- Map data (`map_*` files) and chart data (`*.json` under `data/`) follow hierarchical structures documented per module (Area, Economics, GHG, Production, Water, Biodiversity).
- Cost vs Revenue map layers maintain distinct AgMgt vocabularies; guard selection resets accordingly and use optional chaining with safe fallbacks when traversing nested registries.

## Project Structure & Module Organization
`luto/` holds the core package: `simulation.py` coordinates runs, `data.py` loads large spatial inputs, `settings.py` centralises knobs, and `solvers/` converts scenarios into GUROBI models. Economics subpackages mirror land use domains documented in `DataFlow.md`. Utilities under `luto/tools/` cover spatialisation, reporting, and task templates; the Vue reporting pipeline in `luto/tools/report/` follows the progressive selection pattern outlined above. Keep bulky data in `input/` and run artefacts in `output/`, with specs and diagrams living in `docs/` and `DataFlow.md`.

## Build, Test, and Development Commands
Create an isolated environment before editing:
- `conda env create -f luto/tools/create_task_runs/bash_scripts/conda_env.yml && conda activate luto`
- `pip install gurobipy==11.0.2 numpy_financial==1.0.0 tables==3.9.2` if not captured by conda
- `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt` for a pure pip workflow
Set `GRB_LICENSE_FILE` to your GUROBI licence path. Use `python -m pytest` or `python -m pytest luto/tests/test_land_use_culling.py -k <keyword>` for targeted runs. Batch scenarios via `python luto/tools/create_task_runs/create_grid_search_tasks.py`, and lower `settings.RESFACTOR` when you need lighter exploratory runs.

## Coding Style & Naming Conventions
Adhere to PEP 8 with four-space indentation. Modules favour explicit type hints and docstrings that describe array shapes or table schemas; match that style when touching NumPy or pandas heavy code. Globals stay UPPER_SNAKE_CASE, functions and variables in `snake_case`, classes in `CamelCase`. Configuration literals belong in `settings.py` so solvers and data prep stay in sync.

## Testing Guidelines
pytest with Hypothesis powers the existing suite; mirror the property based patterns in `luto/tests/test_land_use_culling.py`. Name files `test_<module>.py`, keep scenarios narrow, and patch GUROBI touchpoints for determinism. Run `python -m pytest --maxfail=1 --disable-warnings` before publishing. Guard new tests with fixtures or skips when external data in `input/` is required.

## Commit & Pull Request Guidelines
Recent history relies on Conventional Commits (`feat:`, `fix:`, `chore:`). Summarise modelling impacts, note solver or data dependencies, and include before or after metrics when optimisation behaviour shifts. Pull requests should link issues, highlight any updates to `DataFlow.md` or documentation, attach relevant solver logs or screenshots, and flag reviewers across modelling, data, and reporting tracks.

## Model & Data Flow Notes
When altering pipelines, trace the sequence documented in `DataFlow.md`: raw biophysical assets, staging scripts, `dataprep.py` transforms, `Data` loads, solver assembly, and `tools/write.py` outputs. Respect the risk discounting and sign conventions detailed there. Reporting outputs must keep map versus chart hierarchies consistent with the Vue guidance so progressive selection watchers stay valid.

## Operational Checklist
- Confirm GUROBI licensing and environment activation before invoking solvers.
- Verify input bundles, `settings.py` toggles, and `RESFACTOR` before long runs.
- Inspect `output/<timestamp>/LUTO_RUN__stderr.log` for infeasibilities or performance warnings.
- Regenerate Vue assets only through the provided reporting scripts to keep watcher patterns intact.
