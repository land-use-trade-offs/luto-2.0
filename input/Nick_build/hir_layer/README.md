# hir_layer

LUTO's HIR regrowth rate layer: observed CER issuance per CEA hectare, grossed up, by NRM region.
Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/hir_layer.py`, from the repository root (step 8 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/hir_layer.py --input input/Nick_build/inherited_2010
        --lumap input/Nick_build/livestock_map/lumap.h5 --base-year 2021 --caps input/Nick_build/ag_mgt_caps_2021
        --out input/Nick_build/hir_layer --cache input/Nick_build/_cache [--offline] [--live-check] [--source NAME=PATH]

Peak about 2.4 GB on the full grid (October 2026); needs a machine with at least 8 GB of RAM.

## Sources

Downloaded at run time and sha256-checked against a pin in the script.

| Source | Release / version | Link |
|---|---|---|
| Clean Energy Regulator ACCU Scheme project register (CSV): HIR method projects, registration dates, nominated permanence period, ACCU issuance by financial year | retrieved 2026-09-28; live file CER updates without a version stamp, raw file byte-pinned, cached copy is the reproduction source, `--live-check` stops if the live file differs | https://cer.gov.au/document/accu-scheme-project-register-0 |

Inputs from other datasets: `ag_mgt_caps_2021/hir_project_cells.csv` and `hir_projects.csv` (`ag_mgt_caps.py --base-year 2021`,
step 7, the 2021-map caps); its `provenance.csv` must record the same register pin. `livestock_map/lumap.h5` (`--lumap`,
the base-year map; used only by the eligibility report).
Inherited 2010-build files (`inherited_2010/`): `REGION_NRM_r.h5`, `REGION_STATE_r.h5`, `real_area.h5`,
`NLUM_2010-11_mask.tif` (cell count check), `ag_landuses.csv` (land-use ids). The output is therefore not built from
public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `hir_rate_layer_gross.nc` | LUTO (`HIR_RATE_FILE`) | one variable per column: HIR_RATE, HIR_ELIGIBLE, HIR_RATE_ISSUANCE |
| `hir_rate_layer_gross.h5` | evidence | the same values, 2010-base file layout (key `hir_rate_layer_gross`, float32) |
| `hir_projects_rates.csv` | evidence | per-project table of issuance and rates (written once, read back before the region step) |
| `hir_region_rates_gross.csv` | evidence | gross rate by NRM region |
| `hir_region_rates.csv`, `hir_state_rates.csv` | evidence | issuance-basis rates by region and state |
| `hir_eligible_beef_sheep_natural_by_state.csv` | evidence | HIR-eligible Beef and Sheep natural land by state on the `--lumap` map |
| `provenance.csv` | check | sha256 of the script, every source and input, every output (content hashes for HDF5) |

The window FY2020-FY2024 and the pooling of regions with fewer than 3 projects to the state rate are analysis choices with no
published source. Method: `docs/BASE2021_INPUTS_METHODS.md`.
