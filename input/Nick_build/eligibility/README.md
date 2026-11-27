# eligibility

2021-base land-use eligibility (`x_mrj`) and its zone layers. Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/eligibility.py`, from the repository root (step 5 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/eligibility.py --input input/Nick_build/inherited_2010 --nlum-map input/Nick_build/nlum_map
        --map-dir input/Nick_build/livestock_map --out input/Nick_build/eligibility --cache input/Nick_build/_cache
        [--offline] [--source NAME=PATH]

Peak about 6.0 GB on the full grid (October 2026); needs a machine with at least 16 GB of RAM.

## Sources

Downloaded at run time and sha256-checked against pins in the script.

| Source | Release / version | Link |
|---|---|---|
| ABARES NLUM v7.1 250 m ALUMV8 2020-21 raster and class table (CC BY 4.0) | release 2026-08-14 | https://www.agriculture.gov.au/sites/default/files/documents/NLUM_v7_1_250m_ALUMV8_2020_21_alb_20260814.zip |
| ABARES NLUM v7.1 2020-21 agricultural probability surfaces (CC BY 4.0) | release 2026-08-14 | https://www.agriculture.gov.au/sites/default/files/documents/NLUM_v7_1_250m_AgProbabilitySurfaces_2020_21_geo_20260814.zip |
| ABS ASGS Edition 3 SA2 2021 boundaries, GDA94 (CC BY 4.0) | July 2021 to June 2026 | https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip |
| SILO gridded monthly rainfall, 0.05 degree (Queensland Government, CC BY 4.0) | 1961-2020, one file per year | https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/monthly_rain/ (`<year>.monthly_rain.nc`) |
| Australian Irrigation Areas v1A, vector (ABARES / NLWRA, CC BY 3.0 AU) | v1A | https://data.gov.au/data/dataset/3c934692-b670-4608-8f74-ee36583f74c1/resource/4b926ea7-3453-4194-b402-3e4b7dc76fd6/download/aia__r9ab__00211a02e_sgeo___.zip |

Inputs from other datasets: `nlum_map/xwalk_definition.csv`, `nlum_map/sa2_2021_cells.npy`,
`nlum_map/class_alloc_<Y>_lmmap.npy` (Y = 2010_11, 2015_16, 2020_21), `livestock_map/lumap.h5`, `livestock_map/lmmap.h5`.
Inherited 2010-build files (`inherited_2010/`): `NLUM_2010-11_mask.tif` (the cell grid), `lumap.h5` (the 2010 footprint,
a reported column of the occurrence table only). The outputs are therefore not built from public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `x_mrj.nc` | LUTO (`INPUT_SET='2021'`) | `X_MRJ` (int8, marked bool) over (lm, cell, lu), labels dry/irr and the 28 land uses |
| `x_mrj.npy` | `ghg_factors.py`, `crop_economics.py`, `cci_fill.py`, `compare_inputs.py` | bool, 2 x cells x 28, C order |
| `xmrj_zone_layers.h5`, `xmrj_zone_layers.nc` | evidence | CELL_ID index; the two gates, the SILO 1961-2000 reference gate, both April-November means, AIA_V1A |
| `occurrence_2020_21.csv` | evidence | the thresholded occurrence table (net pixel count at least 4, 25 ha) |
| `provenance.csv` | check | sha256 of the script, every source and input, every output (content hashes for HDF5) |

Method: `docs/BASE2021_INPUTS_METHODS.md`.
