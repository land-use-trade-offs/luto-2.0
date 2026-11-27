# livestock_map

2021-base land-use map with livestock typing, the stubble rate and the per-type stocking calibration k.
Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/livestock_map.py`, from the repository root (step 4 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/livestock_map.py --input input/Nick_build/inherited_2010 --nlum-map input/Nick_build/nlum_map
        --pasture input/Nick_build/pasture_productivity/pasture_kg_dm_ha.h5 --hir-cells input/Nick_build/ag_mgt_caps
        --out input/Nick_build/livestock_map --cache input/Nick_build/_cache [--offline] [--live-check] [--source NAME=PATH]

Peak about 2.0 GB on the full grid (October 2026); needs a machine with at least 8 GB of RAM.

## Sources

Downloaded at run time and sha256-checked against pins in the script.

| Source | Release / version | Link |
|---|---|---|
| ABS 7121.0 Agricultural Commodities, Australia, ASGS Ed 3 data cube (CC BY 4.0) | 2020-21 | https://www.abs.gov.au/statistics/industry/agriculture/agricultural-commodities-australia/2020-21/AGCDCASGS202021.xlsx |
| DCCEEW NIR 2024 activity table, agriculture livestock (CC BY 4.0) | NIR 2024 | https://www.dcceew.gov.au/sites/default/files/documents/activity-table-1990-2024-agriculture-livestock-national.xlsx |
| National Greenhouse Accounts API (NIR 2024), enteric CH4 of dairy cattle, pasture beef cattle and six sheep classes, FY2020-21; raw responses pinned one by one | NIR 2024 | https://greenhouseaccounts.climatechange.gov.au/api/StatesOutputTimeSeries |
| ABS ASGS Edition 3 SA2 2021 boundaries, GDA94 (CC BY 4.0) | July 2021 to June 2026 | https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA94.zip |
| ABARES NLUM v7.1 250 m ALUM v8 2020-21 (feedlot area, ALUM 5.2.2, by SA2; the file `nlum_map.py` pins) | release 2026-08-14 | https://www.agriculture.gov.au/sites/default/files/documents/NLUM_v7_1_250m_ALUMV8_2020_21_alb_20260814.zip |

Inputs from other datasets: `nlum_map/` (class allocation, `sa2_2021_cells.npy`), `pasture_productivity/pasture_kg_dm_ha.h5`,
`ag_mgt_caps/hir_project_cells.csv` and `hir_scaling.csv` (with `--hir-cells`; 2010 map, step 1).
Inherited 2010-build files (`inherited_2010/`): `real_area.h5`, `safe_pur_natl.h5`, `safe_pur_modl.h5`, `state_id.npy`,
`NLUM_2010-11_mask.tif`, and `lumap.h5` / `lmmap.h5` as the file layout (dtype, name, index) of the outputs of the same name.
The outputs are therefore not built from public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `lumap.nc`, `lmmap.nc` | LUTO (`INPUT_SET='2021'`) | 2021-base land use and land management per cell |
| `lvstk_k.nc` | LUTO (`LVSTK_K_FILE`) | stocking calibration k per cell, variables BEEF, SHEEP, DAIRY |
| `stubble_dse_ha.nc` | `stubble_dse.py` (`--raw`) | stubble rate per cell, DSE/ha |
| `lumap.h5`, `lmmap.h5` | `eligibility.py`, `ghg_factors.py`, `crop_economics.py`, `livestock_economics.py`, `cci_fill.py`, `water_licence_price.py`, `hir_layer.py` (`--lumap`, `lumap.h5` only), `compare_inputs.py` | 2010-base file layout |
| `lvstk_k.h5`, `stubble_dse_ha.h5` | `livestock_economics.py` (k), `compare_inputs.py` | 2010-base file layout (`STUBBLE_DSE_HA` float32; BEEF, SHEEP, DAIRY float32) |
| `livestock_dse_ratings.csv`, `livestock_recon_state_type.csv`, `livestock_recon_sa2.csv` | `livestock_economics.py` (all three); `compare_inputs.py` (ratings, state and type); LUTO (`INPUT_SET='2021'`, ratings) | DSE per head by type; NIR head reconciliation by state and type, and by SA2 (town-SA2 pooling flag, target and k before pooling, feedlot head) |
| `livestock_hir_recalibration.csv` | evidence | per SA2 and type with HIR area: HIR capacity share, k without and with HIR, head moved, band flags |
| `provenance.csv` | check | sha256 of the script, every source and input, every output (content hashes for HDF5) |

No `feed_req.h5` is written. Method: `docs/BASE2021_INPUTS_METHODS.md`.
