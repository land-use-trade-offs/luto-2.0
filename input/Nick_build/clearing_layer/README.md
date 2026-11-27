# clearing_layer

Land-clearing emissions layer: NVIS forest share x NIR primary-clearing emission per hectare.
Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/clearing_layer.py`, from the repository root (step 9 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/clearing_layer.py --input input/Nick_build/inherited_2010
        --out input/Nick_build/clearing_layer --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Peak about 1.4 GB on the full grid (October 2026); needs a machine with at least 8 GB of RAM.

## Sources

Downloaded at run time and sha256-checked against a pin in the script.

| Source | Release / version | Link |
|---|---|---|
| DCCEEW NIR 2024 activity table, LULUCF (Tables 6-14: national and state forest conversion, primary and re-clearing area and direct emissions by financial year; CC BY 4.0) | NIR 2024 | https://www.dcceew.gov.au/sites/default/files/documents/activity-table-1990-2024-lulucf.xlsx |

Inputs from other datasets: none.
Inherited 2010-build files (`inherited_2010/`): `bio_GBF3_NVIS_MVG.nc` and `bio_GBF3_NVIS_MVG_groups.npy` (NVIS major
vegetation group percent cover per cell; origin not recorded), `REGION_STATE_r.h5` (state of each cell). The output is
therefore not built from public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `clearing_layer.nc` | none yet (no reader on jinzhu) | one variable per column: CLEAR_DIRECT, CLEAR_DIRECT_MIXED, F_FOREST, F_SPARSE |
| `clearing_layer.h5` | evidence | the same values, 2010-base file layout (key `clearing_layer`, float32) |
| `nir_rates_pooled_fy2017_2022.csv` | evidence | NIR primary and all-clearing emission per hectare, pooled FY2016-17 to FY2021-22, by state and national |
| `rates_used.csv` | evidence | the rate applied to each state (ACT takes NSW; other territories national; VIC and TAS flagged) |
| `provenance.csv` | check | sha256 of the script, the source and inputs, every output (content hashes for HDF5) |

FY2022-23 and FY2023-24 copy FY2021-22 in the source and are excluded (checked, recorded in provenance).
Method: `docs/BASE2021_INPUTS_METHODS.md`.
