# nlum_map

The NLUM v7.1 land-use class allocation on the LUTO grid for 2010-11 (control), 2015-16 and 2020-21, with the SA2 2021
code of each cell. Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/nlum_map.py`, from the repository root (step 2 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/nlum_map.py --input input/Nick_build/inherited_2010 --out input/Nick_build/nlum_map
        --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Peak about 9.9 GB in the main process and 30.3 GB across the process tree with 48 worker processes (October 2026);
needs a machine with at least 64 GB of RAM, or a lower `--workers`.

## Sources

Downloaded at run time and sha256-checked against pins in the script.

| Source | Release / version | Link |
|---|---|---|
| ABARES NLUM v7.1 250 m ALUMV8 rasters and class tables, 2010-11, 2015-16, 2020-21 (CC BY 4.0) | release 2026-08-14, DOI 10.25814/n0z4-5y98 | https://www.agriculture.gov.au/sites/default/files/documents/ (`NLUM_v7_1_250m_ALUMV8_<year>_alb_20260814.zip`) |
| ABARES NLUM v7.1 agricultural commodity probability surfaces, 2010-11, 2015-16, 2020-21 (CC BY 4.0) | release 2026-08-14 | https://www.agriculture.gov.au/sites/default/files/documents/ (`NLUM_v7_1_250m_AgProbabilitySurfaces_<year>_geo_20260814.zip`) |
| ABS ASGS Edition 3 SA2 2021 boundaries, GDA94 (CC BY 4.0) | July 2021 to June 2026 | https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip |

Inputs from other datasets: none.
Inherited 2010-build files (`inherited_2010/`): `NLUM_2010-11_mask.tif`, `real_area.h5`, `lumap.h5`, `lmmap.h5`,
`state_id.npy`. The outputs are therefore not built from public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `xwalk_definition.csv` | `eligibility.py` | crosswalk from ALUM v8 tertiary class to LUTO (lu, lm) |
| `sa2_2021_cells.npy` | `livestock_map.py`, `eligibility.py`, `ghg_factors.py`, `crop_economics.py`, `livestock_economics.py`, `cci_fill.py`, `compare_inputs.py` | int32 SA2_2021 code per LUTO cell |
| `class_alloc_<Y>_lumap.npy` (int16) | `livestock_map.py` (2020_21), `compare_inputs.py` (2010_11, 2020_21) | land use per cell, Y = 2010_11, 2015_16, 2020_21 |
| `class_alloc_<Y>_lmmap.npy` (int8) | `livestock_map.py` (2020_21), `eligibility.py` (all three), `compare_inputs.py` (2010_11, 2020_21) | land management per cell; 2015_16 is built only for its irrigation flag |
| `class_alloc_<Y>_category.npy` (int8) | `livestock_map.py` (2020_21), `compare_inputs.py` (2010_11, 2020_21) | class category per cell (codes in `CATEGORY`) |
| `provenance.csv` | check | sha256 of the script, every source and input, every output |

Livestock typing of grazing cells is not done here (`livestock_map.py`). Method: `docs/BASE2021_INPUTS_METHODS.md`.
