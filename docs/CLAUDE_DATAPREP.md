# LUTO2 dataprep: how `input/` is produced, and what depends on the base map

Scope: `luto/dataprep.py` (`create_new_dataset`), its upstream sources on the N: share, and the dependency map you need before regenerating inputs for a new base map. Line numbers: branch jinzhu at 72d30427. Companion to [CLAUDE_DATA.md](CLAUDE_DATA.md).

## 1. Shape of the pipeline

1. Upstream (not in this repo): scripts under `N:\Data-Master\LUTO_2.0_input_data\Scripts\` (e.g. `2_assemble_agricultural_data.py`, `4_assemble_biophysical_data.py`, a `build.ipynb`, and a "script 5_4" that dataprep:213 credits for the biodiversity performance curves) produce the raw snapshots below. These were located on N: in an earlier investigation; from a machine without N: mounted they are unreachable.
2. `dataprep.create_new_dataset(refresh=False)` copies raw snapshots from N: into `settings.RAW_DATA` (`'../raw_data'`, relative to the working directory) (dataprep:81-107), copies about 60 files straight into `INPUT_DIR` (dataprep:97-226), and computes the remaining runtime inputs (dataprep:232-1291).
3. `input/` as delivered in 2026-09 arrived as a zip (`transfer_*.zip`); `raw_data/` is not part of it.

N: source directories (dataprep:47-68): `LUTO_2.0_input_data/Input_data/{1D_Parameter_Timeseries, 2D_Spatial_Snapshot, 3D_Spatial_Timeseries, 4D_Spatial_SSP_Timeseries}`, `LUF-Modelling/fdh-archive/...`, `Profit_map`, `Food_demand_AU/au.food.demand/Outputs`, `Demand_elasticity`, `Water/Water_account`, `Regional_adoption_and_Social_license`, `National_Landuse_Map`, `BECCS/From_CSIRO/20211124_as_submitted`, `Habitat_condition_assessment_system`, `Biodiversity/DCCEEW/SNES_ECNES/Processed`, `NVIS/Processed`, `Biodiversity/Environmental-suitability/...`, `Biodiversity/DCCEEW/RHI`, `Renewable Energy/processed`, `AG 2050`.

## 2. Raw snapshots dataprep needs (from `2D_Spatial_Snapshot` unless noted)

| Raw file | Copied at | Used for |
|---|---|---|
| `cell_LU_mapping.h5` (`lmap`) | :86 | the base map: `LU_ID_LUTO` -> lumap, `IRRIGATION` -> lmmap, `SA2_ID` -> state_id and the SA2 concordance, `CELL_ID` |
| `cell_zones_df.h5` (`zones`) | :87 | regions, river regions, drainage divisions, `CELL_HA` -> real_area, IBRA layers, BECCS join |
| `cell_livestock_data.h5` (`lvstk`) | :88 | agec_lvstk, feed_req, pasture, safe_pur, water delivery price |
| `cell_biophysical_df.h5` (`bioph`) | :93 | soil carbon, fire risk, stream length, establishment costs, water yield baselines, biodiversity rank/connectivity, natural land carbon |
| `SA2_crop_data.h5`, `SA2_crop_GHG_data.h5`, `SA2_livestock_GHG_data.h5`, `SA2_irrigated_pasture_GHG_data.h5`, `SA2_climate_damage_mult.h5` | :89-94 | joined to cells on `SA2_ID` -> agec_crops, agGHG_*, climate_change_impacts_* |
| `NLUM_SPREAD_LU_ID_Mapped_Concordance.h5` (Profit_map) | :84 | x_mrj eligibility |
| `tmatrix-cat2lus.csv`, `transitions_costs_20251002.xlsx` (fdh-archive) | :81-82 | transition matrices |
| `All_LUTO_demand_scenarios_with_convergences.csv` | :102 | demand_projections.h5 |
| `df_info_best_grid_20211116.pkl` (BECCS) | :107 | cell_BECCS_df.h5 |

## 3. The SA2 join

`concordance = lmap[['CELL_ID', 'SA2_ID']]` (dataprep:320) is the only bridge between SA2 tables and cells. Merges: climate impacts :947, agec_crops :995, agGHG_crops :1050, agGHG_lvstk :1131, agGHG_irrpast :1156, all `how='left'` on `SA2_ID`. There is no spatial fallback: if the `SA2_ID` vintage in `cell_LU_mapping.h5` (ASGS 2011 today) differs from the SA2 tables, cells get NaN silently. Any new base map must either carry the same SA2 code vintage or come with re-keyed SA2 tables.

## 4. Dependency map: what a new `cell_LU_mapping.h5` forces you to regenerate

Same grid assumed (6,956,407 cells, same order). Every YES/PARTIAL row was traced to an explicit `lmap` / `lumap` / `concordance` reference in dataprep.py.

| Output | Write line | Depends on the map because |
|---|---|---|
| `state_id.npy` | 311 | `lmap['SA2_ID']` |
| `lumap.h5` | 340 | `lmap['LU_ID_LUTO']` |
| `lmmap.h5` | 355 | `lmap['IRRIGATION']` |
| `x_mrj.npy` | 608 | concordance merge (548-552) plus reconciliation against lumap/lmmap (600-605) |
| `regional_adoption_zones.xlsx` | 411-413 | BASE_LANDUSE_AREA_PERCENT computed by looping over `lumap` (398-404) |
| `water_yield_outside_LUTO_study_area_2010_2100_{dd,rr}_ml.h5`, `..._hist_1970_2000.h5` | 711-712, 742-748 | `idx_outside_LUTO = (lumap == -1)` (345) |
| `water_yield_natural_land_2010_2100_{dd,rr}_ml.h5` | 719-720 | indexed by `lmap['CELL_ID']` (698) |
| `BIODIVERSITY_GBF2_TOP_RANK_CELL_BIO_SCORES_AND_TARGET.csv` | 872-876 | degradation lookup on `lumap` (809), `idx_inside_LUTO` (823); this is the GBF2 base-year score |
| `climate_change_impacts_*.h5` (16 files) | 976-978 | SA2 join |
| `agec_crops.h5` | 1019 | SA2 join |
| `agGHG_crops.h5`, `agGHG_lvstk.h5`, `agGHG_irrpast.h5` | 1078, 1153, 1159 | SA2 join |

Map-independent (regenerate only if their own raw source changes): all `shutil.copyfile` rows (97-226), `cell_savanna_burning.h5` (119), the seven `tCO2_ha_*.nc` (124-137), the eight SSP water-yield h5 (141-156), transition matrices (258, 270-275, 486-502), `REGION_*` and `regional_adoption_zones.h5` (360-379), `real_area.h5` (422), IBRA nc (455-462), livestock helpers (510-528), `water_licence_price.h5`, `soil_carbon_t_ha.h5` (532-540), `water_yield_baselines.h5` (617), river/drainage ids and luts (632-647), biodiversity rank h5 (764-771), `stream_length_m_cell.h5`, `natural_land_t_co2_ha.h5`, `fire_risk.h5`, `ep/cp_est_cost_ha.h5` (886-922), `agec_lvstk.h5` (1039), `agGHG_lvstk_off_land.csv` (1209), `demand_projections.h5` (1240), `cell_BECCS_df.h5` (1291).

Caveat on the "independent" rows: they assume `cell_zones_df.h5`, `cell_biophysical_df.h5` and `cell_livestock_data.h5` stay row-aligned with `cell_LU_mapping.h5`. Nothing in dataprep checks that; verify `CELL_ID` equality yourself.

## 5. Things in dataprep that will bite

- `dataprep.py:1088` reads the hard-coded relative path `'input/real_area.h5'` instead of `outpath`.
- `dataprep.py:789` reads `INPUT_DIR/BIODIVERSITY_GBF2_conservation_performance.xlsx`, which nothing in the repo produces. It must already be in `input/` (copied by hand or by an older dataprep). The runtime reads the sibling `Biodiversity_conserve_performance.xlsx` (copied at :177) instead.
- `bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv` is copied from `HABITAT_CONDITION.csv` (:174) then rescaled in place (:776-783). The 2026-09 bundle's copy has a `CSV_DEFINED` column where `settings.HCAS_CONTRIBUTION_PERCENTILE = 'USER_DEFINED'` expects `USER_DEFINED`; the bundle and HEAD are not from the same dataprep run.
- `refresh=True` deletes everything in `INPUT_DIR` except `.gitignore` (:76). Do not run it against a shared checkout.
- dataprep writes only into `RAW_DATA` and `INPUT_DIR`. It never writes to N:. Keep it that way.
