# ag_mgt_caps_2021

`ag_mgt_caps/` built on the 2020-21 land-use map: the existing ACCU HIR projects as base-year adoption for 2020-21 runs.
Only the HIR rows of `caps.csv` depend on the map (project area on the option's land use / land-use area); the project
cells, projects, scaling and manifest equal those of `ag_mgt_caps/`. Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/ag_mgt_caps.py`, from the repository root (step 7 of the order in `scripts/README.md`, after
`livestock_map/`):

    python input/Nick_build/scripts/ag_mgt_caps.py --out input/Nick_build/ag_mgt_caps_2021 --input input
        --lumap input/Nick_build/livestock_map/lumap.nc --base-year 2021 --cache input/Nick_build/ag_mgt_caps_cache

The map is refused unless its content sha256 is the 2021 map's (`LUMAP_CONTENT_SHA` in the script).

## Sources

As `ag_mgt_caps/README.md`. Input from another dataset: `livestock_map/lumap.nc`.

## Outputs

| File | Read by | Content |
|---|---|---|
| `caps.csv` | evidence (HIR caps on the 2021 map) | cap fraction by option, land use and year |
| `hir_project_cells.csv`, `hir_projects.csv`, `hir_scaling.csv` | LUTO (`HIR_BASE_YEAR_PROJECTS`, 2020-21 runs: cells, and the baseline FY of the scaling file); `hir_layer.py --caps` (cells, projects) | existing HIR project cells, projects, scaling |
| `cea_manifest.csv` | evidence | CEA files with sha256 |
| `provenance.csv` | check | sha256 of the script, sources, the map (content) and every output; `hir_layer.py` checks its register pin |
