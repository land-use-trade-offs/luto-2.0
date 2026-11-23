# ghg_factors

2021-base agricultural GHG emission factors: livestock per head, off-land, crop and irrigated-pasture SOIL, crop
residues, field burning and rice CH4. AR5 CO2e, FY2020-21. Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/ghg_factors.py`, from the repository root (step 6 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/ghg_factors.py --input input/Nick_build/inherited_2010 --nlum-map input/Nick_build/nlum_map
        --map-dir input/Nick_build/livestock_map --x-mrj input/Nick_build/eligibility/x_mrj.npy
        --out input/Nick_build/ghg_factors --cache input/Nick_build/_cache [--offline] [--live-check]
        [--source NAME=PATH] [--source-dir DIR]

Peak 73.2 GiB on the full grid with the crop fields, 24 min (19 October 2026 build); needs a machine with at least
128 GB of RAM.

## Sources

Downloaded at run time and sha256-checked against pins in the script. Release as pinned.

| Source | Release / version | Link |
|---|---|---|
| National Greenhouse Accounts API (NIR 2024), `StatesOutputTimeSeries`: FY2020-21 emissions by state of the livestock, off-land (swine, poultry), fertiliser and crop-field nodes; 827 raw responses pinned one by one | NIR 2024 | https://greenhouseaccounts.climatechange.gov.au/api/StatesOutputTimeSeries |
| DCCEEW NIR 2024 activity table, agriculture livestock (state populations by class, FY2020-21) | NIR 2024 | https://www.dcceew.gov.au/sites/default/files/documents/activity-table-1990-2024-agriculture-livestock-national.xlsx |
| DCCEEW NIR 2024 activity table, agriculture fertiliser (N mass by state and system; reported rates only, no factor uses it) | NIR 2024 | https://www.dcceew.gov.au/sites/default/files/documents/activity-table-1990-2024-agriculture-fertiliser.xlsx |
| NIR 2024 Volume 2, Table A5.5.8.1 (crop production areas by state and fertiliser system): FY2020-21 rows hand-entered in the script's reference table with PDF page and PDF sha256; the script does not read the PDF | NIR 2024 | https://www.dcceew.gov.au/sites/default/files/documents/national-inventory-report-2024-vol-2.pdf |

Inputs from other datasets: `nlum_map/sa2_2021_cells.npy`, `livestock_map/{lumap,lmmap}.h5`, `eligibility/x_mrj.npy`.
Inherited 2010-build files (`inherited_2010/`): `agGHG_lvstk.h5`, `agGHG_crops.h5`, `agGHG_irrpast.h5`,
`agGHG_lvstk_off_land.csv` (file layouts, NaN masks, every unchanged field, and the energy and lifecycle values),
`demand_projections.h5` (FY2021 off-land production), `real_area.h5`, `ag_landuses.csv`, `state_id.npy`,
`NLUM_2010-11_mask.tif`. The outputs are therefore not built from public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `agGHG_lvstk.nc` | LUTO (`INPUT_SET='2021'`); `stubble_dse.py` (`--input`) | `AGGHG_LVSTK` over (column, cell), levels animal and field |
| `agGHG_irrpast.nc` | LUTO (`INPUT_SET='2021'`) | one variable per column |
| `agGHG_crops.h5` | `crop_economics.py` (`--ghg-crops`, interim input) | 2010-base file layout with the three crop fields appended; energy and lifecycle fields interim-filled |
| `agGHG_lvstk.h5`, `agGHG_irrpast.h5` | `compare_inputs.py` | the same values as the `.nc` files, 2010-base file layout |
| `agGHG_lvstk_off_land.csv` | `compare_inputs.py` | pork, chicken and egg enteric and manure rows; text layout of the 2010-base file |
| `ghg_crop_rates.csv`, `ghg_crop_field_rates.csv` | `compare_inputs.py` (rates); field rates evidence | crop and irrigated-pasture rates by system and state; crop-field rates by field, group and state |
| `ghg_crop_field_placed.csv` | evidence | crop-field tonnes placed on the final map against NIR, by field and state |
| `interim_fill_energy_lifecycle.csv` | evidence | every interim fill |
| `provenance.csv` | check | sha256 of the script, every source and input, every output (content hashes for HDF5) |

There is no `agGHG_crops.nc` here: `crop_economics.py` writes it with the final file.

Interim items: blank energy and lifecycle fields of `agGHG_crops.h5` and `agGHG_irrpast.h5` on eligible cells take the
area-weighted state mean (national where the state has none); to be rebuilt; reported only, off every constraint row.
Method: `docs/BASE2021_INPUTS_METHODS.md`.
