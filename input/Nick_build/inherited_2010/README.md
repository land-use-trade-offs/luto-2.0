# inherited_2010

Frozen copies of 38 old-format 2010-build input files (`.h5`, `.npy`, `.tif`, `.csv`, two `.nc`), as LUTO's
`dataprep.py` wrote them before the inputs moved to NetCDF. The 2021-base scripts read them through `--input`. Not built
here and not in git. Porting the scripts' readers to the current NetCDF inputs is planned as a separate change; this
folder goes once that is done.

Source: the input folder of the 2026-10-22 build (`dataprep.py` output on the pre-NetCDF branch). `SHA256SUMS.csv` lists
each file's sha256; every file that a script records under `inherited` in its `provenance.csv` matches.

| File | Read by |
|---|---|
| `NLUM_2010-11_mask.tif` | nlum_map, pasture_productivity, livestock_map, eligibility, ghg_factors, hir_layer, crop_economics, livestock_economics, compare_inputs |
| `real_area.h5` | nlum_map, livestock_map, ghg_factors, hir_layer, crop_economics, livestock_economics, cci_fill, water_licence_price, water_domestic, compare_inputs |
| `lumap.h5`, `lmmap.h5` | nlum_map, livestock_map, crop_economics, compare_inputs (`lumap.h5` also eligibility) |
| `state_id.npy` | nlum_map, livestock_map, ghg_factors, crop_economics, livestock_economics, cci_fill, compare_inputs |
| `ag_landuses.csv` | ghg_factors, hir_layer, crop_economics, cci_fill, compare_inputs |
| `safe_pur_natl.h5`, `safe_pur_modl.h5` | livestock_map, livestock_economics, compare_inputs |
| `agec_crops.h5` | crop_economics |
| `agec_lvstk.h5` | livestock_economics |
| `agGHG_crops.h5` | ghg_factors, crop_economics, compare_inputs |
| `agGHG_lvstk.h5`, `agGHG_irrpast.h5`, `agGHG_lvstk_off_land.csv`, `demand_projections.h5` | ghg_factors, compare_inputs |
| `water_delivery_price.h5` | crop_economics, livestock_economics |
| `yieldincreases_bau2022.csv` | livestock_economics |
| `climate_change_impacts_<rcp>_CO2_FERT_<ON/OFF>.h5` (8) | cci_fill |
| `REGION_STATE_r.h5` | hir_layer, clearing_layer, water_licence_price, water_domestic |
| `REGION_NRM_r.h5` | hir_layer |
| `bio_GBF3_NVIS_MVG.nc`, `bio_GBF3_NVIS_MVG_groups.npy` | clearing_layer |
| `rivreg_id.h5`, `rivreg_lut.h5`, `draindiv_id.h5`, `draindiv_lut.h5` | water_licence_price, water_domestic (`draindiv_*` also compare_inputs) |
| `water_licence_price.h5` | water_licence_price |
| `Water_Use_Domestic.csv` | water_domestic |
| `pasture_kg_dm_ha.h5`, `feed_req.h5`, `x_mrj.npy` | compare_inputs |
