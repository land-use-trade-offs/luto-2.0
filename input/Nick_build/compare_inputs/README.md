# compare_inputs

Comparison tables of the 2010-base inputs, the 2021-base products and public reference data. Builds no model input;
the tables are the evidence cited by `docs/BASE2021_INPUTS_METHODS.md`. Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/compare_inputs.py`, from the repository root (step 16 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/compare_inputs.py --input input/Nick_build/inherited_2010 --base2021 input/Nick_build
        --out input/Nick_build/compare_inputs --cache input/Nick_build/_cache [--offline] [--live-check]
        [--source NAME=PATH] [--source-dir DIR]

Peak about 15 GB on the full grid (October 2026); needs a machine with at least 32 GB of RAM.

## Sources

Reference data, downloaded at run time and sha256-checked against pins in the script.

| Source | Release / version | Link |
|---|---|---|
| National Greenhouse Accounts API (NIR 2024), `StatesOutputTimeSeries`: livestock, fertiliser and off-land nodes, Australia and the eight states; 454 raw responses pinned one by one | NIR 2024 | https://greenhouseaccounts.climatechange.gov.au/api/StatesOutputTimeSeries |
| DCCEEW NIR 2024 activity table, agriculture livestock (state populations, FY2010-11 and FY2020-21) | NIR 2024 | https://www.dcceew.gov.au/sites/default/files/documents/activity-table-1990-2024-agriculture-livestock-national.xlsx |
| ABS 7121.0 Agricultural Commodities (state livestock head 2020-21) | 2015-16 to 2020-21 | https://www.abs.gov.au/statistics/industry/agriculture/agricultural-commodities-australia/ |
| ABS 7503.0 Value of Agricultural Commodities Produced (Tasmanian "Other oilseeds") | 2015-16 to 2020-21 | https://www.abs.gov.au/statistics/industry/agriculture/value-agricultural-commodities-produced-australia/ |
| ABS 4618.0 Water Use on Australian Farms (ASGS Edition 3 and Murray-Darling Basin cubes) | 2020-21 | https://www.abs.gov.au/statistics/industry/agriculture/water-use-australian-farms/ |

Inputs from other datasets (`--base2021`, `input/Nick_build`, one folder per script): `nlum_map/` (`class_alloc_<Y>_*.npy`,
`sa2_2021_cells.npy`), `pasture_productivity/pasture_kg_dm_ha.h5`, `livestock_map/` (`lumap.h5`, `lmmap.h5`, `lvstk_k.h5`,
`stubble_dse_ha.h5`, `livestock_dse_ratings.csv`, `livestock_recon_state_type.csv`), `eligibility/x_mrj.npy`,
`ghg_factors/` (`agGHG_lvstk.h5`, `agGHG_irrpast.h5`, `agGHG_lvstk_off_land.csv`, `ghg_crop_rates.csv`),
`crop_economics/` (`agGHG_crops.h5`, `agec_crops.h5`, `crop_production_window.csv`, `crop_yield_calibration.csv`,
`crop_water_calibration.csv`), `livestock_economics/` (`agec_lvstk.h5`, `water_delivery_price.h5`,
`livestock_water_per_head.csv`).
Inherited 2010-build files (`inherited_2010/`): `lumap.h5`, `lmmap.h5`, `real_area.h5`, `state_id.npy`,
`NLUM_2010-11_mask.tif`, `ag_landuses.csv`, `safe_pur_natl.h5`, `safe_pur_modl.h5`, `pasture_kg_dm_ha.h5`, `feed_req.h5`,
`agGHG_crops.h5`, `agGHG_lvstk.h5`, `agGHG_irrpast.h5`, `agGHG_lvstk_off_land.csv`, `demand_projections.h5`, `x_mrj.npy`,
`draindiv_id.h5`, `draindiv_lut.h5`.

## Outputs

CSV with one `#` header comment row (what is compared, sources, units, labels), then the data.

| File | Read by | Content |
|---|---|---|
| `t1_livestock_head_state_type.csv` | evidence | head by state and type, 2021-base map, against NIR FY2020-21 and ABS 7121.0 2020-21 |
| `t2_landuse_area_state_class.csv` | evidence | area by LUTO class and state: 2010 map, NLUM 2010-11 control, NLUM 2020-21 before typing, 2021-base map |
| `t3a_ghg_fy2021_rows.csv`, `t3b_ghg_fy2021_category_state.csv`, `t3c_ghg_fy2011.csv` | evidence | emissions by NIR category, state and row against NIR (AR5), split into activity and factor parts; FY2010-11 on the 2010 map |
| `t4a_crops_by_crop_state.csv`, `t4b_crops_named_rows.csv`, `t4c_tas_other_oilseeds.csv`, `t4d_sown_share_group_state.csv` | evidence | crop production and value against the ABS window means; named rows; Tasmanian Other oilseeds; sown share by crop group and state |
| `t5_water_drainage_division.csv` | evidence | irrigation water by drainage division against ABS 4618.0 2020-21 |
| `t6a_pasture_productivity_state.csv`, `t6b_eligibility_state_lm_lu.csv` | evidence | pasture productivity 2021 against 2010 by state; x_mrj eligible area by (lm, lu) and state, with cells gained and lost |
| `compare_inputs_summary.txt` | evidence | key national numbers, each with the files it is computed from |
| `provenance.csv` | check | sources, inherited and 2021-base inputs (sha256), per-table inputs, outputs |

Method: `docs/BASE2021_INPUTS_METHODS.md`.
