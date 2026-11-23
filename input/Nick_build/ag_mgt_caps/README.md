# ag_mgt_caps

Observed ag-management adoption caps and the existing ACCU HIR project cells, on the 2010 land-use map. Data not in git;
rebuild with the script below. The same script on the 2021 map writes `ag_mgt_caps_2021/`.

## Built by

`input/Nick_build/scripts/ag_mgt_caps.py`, from the repository root (step 1 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/ag_mgt_caps.py --out input/Nick_build/ag_mgt_caps --input input
        --cache input/Nick_build/ag_mgt_caps_cache

`--input` is LUTO's input folder (NLUM_2010-11_mask.tif, lumap.nc, real_area.nc, ag_landuses.csv). The map is checked
against the 2010 map's content sha256.

## Sources

Downloaded at run time (or read from `ag_mgt_caps_cache/`) and sha256-checked against pins in the script:

| Source | Release / version | Link |
|---|---|---|
| CER ACCU Scheme project register (CSV) | retrieved 2026-09-28; live file, raw file pinned | https://cer.gov.au/document/accu-scheme-project-register-0 |
| CER Carbon Estimation Area mapping files of the HIR projects | as listed in `cea_manifest.csv` (sha256 per file) | CER project pages |
| ABS 4627.0 Land Management and Farming in Australia, Table 1 Australia | 2016-17 (released 2018-06-26) | ABS 4627.0 |

The live register no longer matches the pin; `ag_mgt_caps_cache/` holds the pinned copy, so the build runs from the cache.

## Outputs

| File | Read by | Content |
|---|---|---|
| `caps.csv` | LUTO (`AG_MANAGEMENT_OBSERVED_CAPS`) | cap fraction by option, land use and year |
| `hir_project_cells.csv`, `hir_scaling.csv` | LUTO (`HIR_BASELINE`); `livestock_map.py --hir-cells` | share of each cell inside each mapped HIR project; scale for unmapped projects |
| `hir_projects.csv`, `cea_manifest.csv` | evidence | HIR projects; CEA files with sha256 |
| `provenance.csv` | check | sha256 of the script, sources, the map (content) and every output |
