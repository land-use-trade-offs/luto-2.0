# crop_economics

2020-21 base crop economics and crop energy and lifecycle GHG fields, per hectare of cropping land, FY2021 dollars.
Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/crop_economics.py`, from the repository root (step 10 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/crop_economics.py --input input/Nick_build/inherited_2010 --nlum-map input/Nick_build/nlum_map
        --map-dir input/Nick_build/livestock_map --x-mrj input/Nick_build/eligibility/x_mrj.npy
        --ghg-crops input/Nick_build/ghg_factors/agGHG_crops.h5 --out input/Nick_build/crop_economics
        --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Peak 130.3 GiB on the full grid (130.5 GiB in the 22 October 2026 build), 54 min with the NetCDF copies (22 October 2026);
needs a machine with at least 256 GB of RAM.

## Sources

Downloaded at run time and sha256-checked against pins in the script. Release as pinned.

| Source | Release / version | Link |
|---|---|---|
| ABS 6401.0 Consumer Price Index (A2325846C, all groups, quarterly) | August 2026 | https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/aug-2026/CPI-Quarterly.zip |
| ABS 7121.0 Agricultural Commodities (state files; SA2 files for FY2015-16 and FY2020-21) | FY2015-16 to FY2021-22 | https://www.abs.gov.au/statistics/industry/agriculture/agricultural-commodities-australia/ |
| ABS 7503.0 Value of Agricultural Commodities Produced (state files; SA2 files for FY2015-16 and FY2020-21) | FY2015-16 to FY2021-22 | https://www.abs.gov.au/statistics/industry/agriculture/value-agricultural-commodities-produced-australia/ |
| ABS Australian Agriculture data cubes: Broadacre Crops, Horticulture, Livestock | 2022-23 to 2024-25 | https://www.abs.gov.au/statistics/industry/agriculture/australian-agriculture- (`broadacre-crops/`, `horticulture/`, `livestock/` + year) |
| ABS 4618.0 Water Use on Australian Farms | FY2016-17 to FY2020-21 | https://www.abs.gov.au/statistics/industry/agriculture/water-use-australian-farms/ |
| ABARES Australian Crop Report, state tables | September 2026 | https://daff.ent.sirsidynix.net.au/client/en_AU/search/asset/1038523/3/__AustCropRrt20260901_v1.0.0.xlsx |
| ABARES Farm Data Portal, state and national historical files | as pinned (sha256 in the script) | https://www.agriculture.gov.au/sites/default/files/documents/fdp-state-historical.csv and `fdp-national-historical.csv` |
| ABS 1270.0.55.001 ASGS Volume 1, SA2 boundaries, GDA94 (FY2015-16 SA2 yield index) | July 2011 | https://www.abs.gov.au/AUSSTATS/subscriber.nsf/log?openagent&1270055001_sa2_2011_aust_shape.zip&1270.0.55.001&Data%20Cubes&7130A5514535C5FCCA257801000D3FBD&0&July%202011&23.12.2010&Latest |

Inputs from other datasets: `nlum_map/sa2_2021_cells.npy`, `livestock_map/{lumap,lmmap}.h5`, `eligibility/x_mrj.npy`,
`ghg_factors/agGHG_crops.h5` (interim).
Inherited 2010-build files (`inherited_2010/`): `agec_crops.h5` (layout, cost-to-revenue ratios and cost shares of the
ratio-group crops, cost relativities of the FDP-group crops, irrigated to dry yield ratio, water relativities within a
group), `agGHG_crops.h5` (levels of the energy and lifecycle fields), `water_delivery_price.h5` (WP), `lumap.h5` and
`lmmap.h5` (2010 map, cells preferred for those levels), `real_area.h5`, `state_id.npy`, `ag_landuses.csv`,
`NLUM_2010-11_mask.tif`. The outputs are therefore not built from public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `agec_crops.nc` | LUTO (`INPUT_SET='2021'`) | `AGEC_CROPS` over (column, cell), levels field, lm, lu |
| `agGHG_crops.nc` | LUTO (`INPUT_SET='2021'`) | `AGGHG_CROPS` over (column, cell), levels field, lm, lu |
| `agec_crops.h5`, `agGHG_crops.h5` | `compare_inputs.py` | the same values, 2010-base file layout plus the (`*`, dry, Pears) columns |
| `crop_production_window.csv` | `compare_inputs.py` | production and value by crop and state, FY2017-25 window |
| `crop_yield_calibration.csv`, `crop_water_calibration.csv` | `compare_inputs.py` | K by crop and state, with calibration labels |
| `interim_items_crop_economics.csv` | evidence | INTERIM and inferred items |
| `provenance.csv` | check | sha256 of the script, every source and input, every output (content hashes for HDF5 and NetCDF) |

Interim items: ratio-group costs (13 crops) pending cost budgets; water delivery price (WP) pending the ACCC reports;
IRRIG field on 2010-basis levels. Method: `docs/BASE2021_INPUTS_METHODS.md`.
