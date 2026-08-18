# DataFlow.md — LUTO2 input-data tracing map

**Purpose**: trace every array the solver consumes back to the raw dataset it came from, and forward
again. The map is anchored on `luto/solvers/input_data.py` — the single place where all model inputs
converge into one `SolverInputData` object — because that is the only complete inventory of what the
optimisation actually sees.

**Verified against**: commit `4ea251d9` (working tree, 2026-08-12). Line numbers are navigation
anchors, not contracts — re-grep the symbol if a number has drifted.

---

## Table of Contents

- [1. The eight layers](#1-the-eight-layers)
- [2. Path constants](#2-path-constants)
- [3. Upstream script inventory (`N:` preprocessing)](#3-upstream-script-inventory-n-preprocessing)
- [4. Master index — `SolverInputData` field → provenance](#4-master-index--solverinputdata-field--provenance)
  - [4.1 Economy](#41-economy)
  - [4.2 GHG](#42-ghg)
  - [4.3 Water](#43-water)
  - [4.4 Biodiversity — quality score](#44-biodiversity--quality-score)
  - [4.5 Biodiversity — GBF targets](#45-biodiversity--gbf-targets)
  - [4.6 Quantity / demand](#46-quantity--demand)
  - [4.7 Transitions, bounds, feasibility](#47-transitions-bounds-feasibility)
  - [4.8 Renewable energy](#48-renewable-energy)
  - [4.9 Regions, masks, bookkeeping](#49-regions-masks-bookkeeping)
  - [4.10 Limits (RHS)](#410-limits-rhs)
- [5. `input/` file inventory → raw provenance](#5-input-file-inventory--raw-provenance)
- [6. Detailed trace — reforestation carbon (EP / CP / HIR)](#6-detailed-trace--reforestation-carbon-ep--cp--hir)
- [7. Detailed trace — non-agricultural GHG functions](#7-detailed-trace--non-agricultural-ghg-functions)
- [8. The last mile — rescaling and coefficient filtering](#8-the-last-mile--rescaling-and-coefficient-filtering)
- [9. Issue register](#9-issue-register)
  - [9.1 RESOLVED — carbon-pool caps applied to the wrong pools](#91-resolved--carbon-pool-caps-applied-to-the-wrong-pools)
  - [9.2 OPEN — no re-run trigger linking FullCAM refreshes to `script_9`](#92-open--no-re-run-trigger-linking-fullcam-refreshes-to-script_9)
  - [9.3 OPEN — dead IBRA path in the GBF3 stream](#93-open--dead-ibra-path-in-the-gbf3-stream)
  - [9.4 OPEN — `cell_savanna_burning.h5` has no traceable origin](#94-open--cell_savanna_burningh5-has-no-traceable-origin)
  - [9.5 OPEN — `no_go_areas/` is a toy dataset feeding real constraints](#95-open--no_go_areas-is-a-toy-dataset-feeding-real-constraints)
  - [9.6 OPEN — `input/` files nothing reads](#96-open--input-files-nothing-reads)

---

## 1. The eight layers

Every number in the model travels the same eight-layer chain. When tracing, name the layer you are
at — most confusion comes from conflating L1 (the `N:` script that *built* a file) with L2
(`dataprep.py`, which usually only *copies or subsets* it).

| Layer | Where | What happens |
|-------|-------|--------------|
| **L0** Raw source | `N:/Data-Master/...`, `N:/LUF-Modelling/...` | External datasets: FullCAM, ABARES/ABS, NLUM, DCCEEW, CSIRO, NVIS, AEMO |
| **L1** Upstream script | `N:/Data-Master/LUTO_2.0_input_data/Scripts/script_*.py` | Heavy spatial assembly → `Input_data/{1D,2D,3D,4D}_*` intermediates |
| **L2** `luto/dataprep.py` | repo | `create_new_dataset()` — copy, subset, unit-convert, reshape into `input/` |
| **L3** `input/<file>` | repo `input/` | The model's on-disk contract: `.h5`, `.nc`, `.npy`, `.npz`, `.csv`, `.xlsx`, `.tif` |
| **L4** `luto/data.py` | repo | `Data.__init__` loads + masks (`where=self.MASK` / `.isel(cell=self.MASK)`), applies risk/CPI/climate adjustments → `Data` attributes |
| **L5** `luto/economics/**` | repo | Per-theme matrices: cost, revenue, quantity, GHG, water, biodiversity, transitions |
| **L6** `luto/solvers/input_data.py` | repo | `get_*` builders + rescaling → `SolverInputData` |
| **L7** `luto/solvers/solver.py` | repo | `_qsum()` coefficient filter → Gurobi constraint/objective terms |

Two rules that hold throughout:

- **Masking happens at L4, once.** `self.MASK` (from `NLUM_2010-11_mask.tif` + `RESFACTOR`) is applied
  at load. Everything downstream is indexed by the compact cell index `r` of length `data.NCELLS`.
- **Per-hectare → per-cell happens at L5**, by `× data.REAL_AREA`. If an array is still per-hectare at
  L6 it is a bug.

---

## 2. Path constants

L1→L2 source roots, all defined in `luto/dataprep.py:47-68`:

| Variable | Path |
|----------|------|
| `luto_1D_inpath` | `N:/Data-Master/LUTO_2.0_input_data/Input_data/1D_Parameter_Timeseries/` |
| `luto_2D_inpath` | `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/` |
| `luto_3D_inpath` | `N:/Data-Master/LUTO_2.0_input_data/Input_data/3D_Spatial_Timeseries/` |
| `luto_4D_inpath` | `N:/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/` |
| `fdh_inpath` | `N:/LUF-Modelling/fdh-archive/data/neoluto-data/new-data-and-domain/` |
| `profit_map_inpath` | `N:/Data-Master/Profit_map/` |
| `demand_scenarios_inpath` | `N:/LUF-Modelling/Food_demand_AU/au.food.demand/Outputs` |
| `demand_elasticity_inpath` | `N:/Data-Master/Demand_elasticity` |
| `water_domestic_use` | `N:/Data-Master/Water/Water_account/` |
| `no_go_areas` | `N:/Data-Master/Regional_adoption_and_Social_license/` |
| `nlum_inpath` | `N:/Data-Master/National_Landuse_Map/` |
| `BECCS_inpath` | `N:/Data-Master/BECCS/From_CSIRO/20211124_as_submitted/` |
| `GHG_off_land_inpath` | `N:/LUF-Modelling/Food_demand_AU/au.food.demand/Inputs/Off_land_GHG_emissions` |
| `bio_HACS_inpath` | `N:/Data-Master/Habitat_condition_assessment_system/Data/Processed/` |
| `bio_GBF2_inpath`, `bio_GBF4_inpath`, `bio_NES_Zonation_inpath` | `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/` |
| `bio_GBF3_NVIS_inpath` | `N:/Data-Master/NVIS/Processed` |
| `bio_GBF8_inpath` | `N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km_to_NetCDF/` |
| `bio_RHI_Zonation_inpath` | `N:/Data-Master/Biodiversity/DCCEEW/RHI (Relative Habitat Importance)` |
| `renewable_energy_inpath` | `N:/Data-Master/Renewable Energy/processed` |
| `ag_yield_trend` | `N:/Data-Master/AG 2050/` |

Two L2 destinations: `raw_data` (= `RAW_DATA`, staging for files `dataprep` will process) and
`outpath` (= `INPUT_DIR`, i.e. `input/`, the L3 contract). A file copied straight to `outpath` has
**no** L2 transformation — trace it directly to its L1 script.

---

## 3. Upstream script inventory (`N:` preprocessing)

`N:/Data-Master/LUTO_2.0_input_data/Scripts/` — L1. Note the `script_N_` prefix; the older names
(`4_assemble_biophysical_data.py`) are gone.

| Script | Produces (L1 output) |
|--------|----------------------|
| `script_1_assemble_zones_data.py` | `2D/cell_zones_df.h5` — cell area, irrigation potential, river region / drainage division IDs, NRM & state codes |
| `script_2_assemble_agricultural_data.py` | `2D/cell_LU_mapping.h5`, `cell_livestock_data.h5`, `SA2_crop_data.h5`, `SA2_crop_GHG_data.h5`, `SA2_livestock_GHG_data.h5`, `SA2_irrigated_pasture_GHG_data.h5`, `SA2_off_land_commodity_data.h5` |
| `script_3_agriculture_climate_damage.py` | `2D/SA2_climate_damage_mult.h5` |
| `script_4_assemble_biophysical_data.py` | `2D/cell_biophysical_df.h5` — natural-land carbon stocks (L153-192), fire/drought risk percentiles (L633-651), EP/CP establishment costs, soil carbon, water price, growing-season rainfall |
| `script_5_0_SNES_ECNES_selected.py` | SNES/ECNES species shortlist |
| `script_5_1_assemble_biodiversity_data.py` | `bio_DCCEEW_{SNES,ECNES}_sparse.npz` + `_species.npy`, `bio_NES_Zonation.nc` |
| `script_5_2_get_NVIS_SNES_ECNES_targets_by_regions.py` | `BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv`, `bio_DCCEEW_{SNES,ECNES}_target_ALL_REGIONS.csv` |
| `script_5_3_get_Zonation_layers.py` | Zonation priority layers |
| `script_5_4_get_Zonation_performance_curves.py` | `Biodiversity_conserve_performance.xlsx`, `bio_RHI_Zonation.nc` |
| `script_6_water_yield_modelling.py` | `4D/Water_yield_GCM-Ensemble_ssp{126,245,370,585}_2010-2100_{DR,SR}_ML_HA_mean.h5` |
| `script_7_assemble_additional_land_use_sieve_data.py` | land-use sieve layers feeding the `x_mrj` exclusion build |
| `script_8_assemble_ag_yield_gap_data.py` | agricultural yield-gap data |
| `script_9_reforestation_carbon_data.py` | `3D/tCO2_ha_{ep_block,ep_rip,ep_belt,cp_block,cp_belt,hir_block,hir_rip}.nc` |
| `script_10_1_REM_get_tables_inputs.py` | `renewable_targets.csv`, `renewable_price_AUD_MWh_{solar,wind}.csv`, `renewable_energy_bundle.csv` |
| `script_10_2_REM_get_existing_capacity.py` | `renewable_existing_capacity_MW_1D.nc`, `renewable_existing_capacity_area_fraction_1D.nc` |
| `script_10_3_REM_get_align_input_layers.py` | `renewable_energy_layers_1D.nc` |

`2D/cell_savanna_burning.h5` has **no producing script** in `Scripts/` — it arrives in
`2D_Spatial_Snapshot/` from elsewhere. Flag if you need its provenance.

---

## 4. Master index — `SolverInputData` field → provenance

Field order follows the dataclass (`input_data.py:57-160`). "Builder" is the `get_*` in
`input_data.py`; "L5 entry" is the economics function it delegates to; "Key `data.` attributes" are
the L4 attributes that carry actual data (structural attributes like `NCELLS`, `NLMS`, `DESC2AGLU`
are omitted — they are shape metadata, not payload).

### 4.1 Economy

`economic_contr_mrj` is a 3-tuple `(ag_obj_mrj, non_ag_obj_rk, ag_man_objs)`; under `maxprofit` it is
revenue − cost, under `mincost` it is cost. **Land-use transition costs are not baked in** — they are
charged in the solver against per-source delta vars (`input_data.py:639-643`).

| Field | Builder (line) | L5 entry | Key `data.` attributes | L3 files |
|-------|----------------|----------|------------------------|----------|
| `economic_contr_mrj` (ag part) | `get_ag_c_mrj` 209 / `get_ag_r_mrj` 221 → `get_economic_mrj` 626 | `ag_cost.get_cost_matrices`, `ag_revenue.get_rev_matrices` | `AGEC_CROPS`, `AGEC_LVSTK`, `CROP_PRICE_MULTIPLIERS`, `LVSTK_PRICE_MULTIPLIERS`, `AC/QC/FOC/FLC/FDC/WP_COST_MULTS`, `WATER_DELIVERY_PRICE`, `SAVBURN_COST_HA`, `REAL_AREA` | `agec_crops.h5`, `agec_lvstk.h5`, `ag_price_multipliers.xlsx`, `cost_multipliers.xlsx`, `water_delivery_price.h5`, `cell_savanna_burning.h5`, `real_area.h5` |
| `economic_contr_mrj` (non-ag part) | `get_non_ag_c_rk` 215 / `get_non_ag_r_rk` 227 | `non_ag_cost.get_cost_matrix`, `non_ag_revenue.get_rev_matrix` | `BECCS_COSTS_AUD_HA_YR`, `BECCS_REV_AUD_HA_YR`, `BECCS_TCO2E_HA_YR`, `BECCS_{COST,REV}_MULTS`, `MAINT_COST_MULTS`, `{EP,CP}_{BLOCK,BELT,RIP}_AVG_T_CO2_HA_PER_YR` | `cell_BECCS_df.h5`, `cost_multipliers.xlsx`, `tCO2_ha_*.nc` |
| `economic_contr_mrj` (am part) | `get_ag_man_c_mrj` 579 / `get_ag_man_r_mrj` 596 / `get_ag_man_t_mrj` 602 | `ag_cost.get_agricultural_management_cost_matrices`, etc. | `ASPARAGOPSIS_DATA`, `ECOLOGICAL_GRAZING_DATA`, `PRECISION_AGRICULTURE_DATA`, `AGTECH_EI_DATA`, `BIOCHAR_DATA`, `RENEWABLE_BUNDLE_{SOLAR,WIND}`, `RENEWABLE_LAYERS`, `SOLAR_PRICES`, `WIND_PRICES` | `20260317_Bundle_{MR,AgTech_NE,AgTech_EI}.xlsx`, `20231107_ECOGRAZE_Bundle.xlsx`, `20260401_Bundle_BC.xlsx`, `renewable_energy_bundle.csv`, `renewable_energy_layers_1D.nc`, `renewable_price_AUD_MWh_{solar,wind}.csv` |
| `economic_prices` | `get_commodity_prices_target_yr` 676 | `ag_revenue.get_commodity_prices` | `AGEC_CROPS`, `AGEC_LVSTK`, price multipliers, demand-elasticity multipliers when `DYNAMIC_PRICE` | as above + `demand_elasticity.csv` |
| `economic_target_yr_carbon_price` | `get_target_yr_carbon_price` 684 | — (direct) | `CARBON_PRICES` | `carbon_prices.xlsx` |

### 4.2 GHG

| Field | Builder (line) | L5 entry | Key `data.` attributes | L3 files |
|-------|----------------|----------|------------------------|----------|
| `ag_g_mrj` | `get_ag_g_mrj` 233 | `ag_ghg.get_ghg_matrices` | `AGGHG_CROPS`, `AGGHG_LVSTK`, `AGGHG_IRRPAST`, `SOIL_CARBON_AVG_T_CO2_HA_PER_YR`, `SAVBURN_ELIGIBLE`, `SAVBURN_TOTAL_TCO2E_HA`, `CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR`, `BIO_HABITAT_CONTRIBUTION_LOOK_UP` | `agGHG_crops.h5`, `agGHG_lvstk.h5`, `agGHG_irrpast.h5`, `soil_carbon_t_ha.h5`, `cell_savanna_burning.h5`, `natural_land_t_co2_ha.h5`, `fire_risk.h5`, `bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv` |
| `non_ag_g_rk` | `get_non_ag_g_rk` 239 | `non_ag_ghg.get_ghg_matrix` | `{EP,CP}_*_AVG_T_CO2_HA_PER_YR`, `BECCS_TCO2E_HA_YR`, `CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR`, `BIO_HABITAT_CONTRIBUTION_LOOK_UP`, `LU_LVSTK_NATURAL` | `tCO2_ha_*.nc`, `cell_BECCS_df.h5`, `natural_land_t_co2_ha.h5`, `fire_risk.h5` — full trace in §6-§7 |
| `ag_man_g_mrj` | `get_ag_man_g_mrj` 585 | `ag_ghg.get_agricultural_management_ghg_matrices` | the five AM bundle dicts | AM bundle `.xlsx` files |
| `flow_ghg_ag2ag` | inline 1071 | `ag_ghg.get_ghg_transition_emissions_from_base_year` | `CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR`, `BIO_HABITAT_CONTRIBUTION_LOOK_UP` | `natural_land_t_co2_ha.h5` |
| `offland_ghg` | inline 1330 | — (direct, `/ GHG scale`) | `OFF_LAND_GHG_EMISSION_C` | `agGHG_lvstk_off_land.csv` |

### 4.3 Water

`WATER_CLIMATE_CHANGE_IMPACT = 'off'` swaps the climate-projected yield layers for the historical
ones (`WATER_YIELD_HIST_DR/SR`) at `input_data.py:1119-1122` and `1132-1136`.

| Field | Builder (line) | L5 entry | Key `data.` attributes | L3 files |
|-------|----------------|----------|------------------------|----------|
| `ag_w_mrj` | `get_ag_w_mrj` 245 | `ag_water.get_water_net_yield_matrices` | `WATER_YIELD_DR_FILE`, `WATER_YIELD_SR_FILE`, `WREQ_DRY_RJ`, `WREQ_IRR_RJ`, `WATER_REGION_ID`, `WATER_REGION_HIST_LEVEL`, `WATER_OUTSIDE_LUTO_BY_CCI`, `WATER_USE_DOMESTIC` | `water_yield_ssp{SSP}_2010-2100_{dr,sr}_ml_ha.h5`, `water_yield_baselines.h5`, `water_yield_outside_LUTO_*.h5`, `water_yield_natural_land_*.h5`, `Water_Use_Domestic.csv`, `rivreg_{id,lut}.h5` / `draindiv_{id,lut}.h5` |
| `non_ag_w_rk` | `get_non_ag_w_rk` 340 | `non_ag_water.get_w_net_yield_matrix` | `WATER_YIELD_DR_FILE`, `WATER_YIELD_SR_FILE`, `REAL_AREA` | same yield layers |
| `ag_man_w_mrj` | `get_ag_man_w_mrj` 608 | `ag_water.get_agricultural_management_water_matrices` | AM bundles + `RENEWABLE_BUNDLE_{SOLAR,WIND}` | AM bundle `.xlsx`, `renewable_energy_bundle.csv` |
| `water_region_indices` / `water_region_names` | 250 / 256 | — (direct) | `WATER_REGION_INDEX_R`, `WATER_REGION_NAMES` | `rivreg_*.h5` or `draindiv_*.h5` per `WATER_REGION_DEF` |

`WATER_LIMITS = 'off'` short-circuits both region getters to `{}`.

### 4.4 Biodiversity — quality score

| Field | Builder (line) | L5 entry | Key `data.` attributes | L3 files |
|-------|----------------|----------|------------------------|----------|
| `ag_b_mrj` | `get_ag_b_mrj` 263 | `ag_biodiversity.get_bio_quality_score_mrj` | `BIO_QUALITY_RAW`, `BIO_QUALITY_LDS`, `BIO_HABITAT_CONTRIBUTION_LOOK_UP`, `SAVBURN_ELIGIBLE` (the quality layers are built in `data.py` from the priority-rank layer × `CONNECTIVITY_SCORE`) | `bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.h5`, `bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv`, `cell_savanna_burning.h5` |
| `non_ag_b_rk` | `get_non_ag_b_rk` 354 | `non_ag_biodiversity.get_breq_matrix` | `BIO_QUALITY_RAW`, `BIO_HABITAT_CONTRIBUTION_LOOK_UP`, `LU_LVSTK_NATURAL` | as above |
| `ag_man_b_mrj` | `get_ag_man_b_mrj` 614 | `ag_biodiversity.get_ag_mgt_biodiversity_matrices` | AM bundles, `RENEWABLE_BUNDLE_*` | AM bundle `.xlsx`, `renewable_energy_bundle.csv` |
| `biodiv_contr_ag_j` | `get_ag_biodiv_contr_j` 269 | `ag_biodiversity.get_ag_biodiversity_contribution` | `BIO_HABITAT_CONTRIBUTION_LOOK_UP` | `bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv` (HCAS `HABITAT_CONDITION.csv`) |
| `biodiv_contr_non_ag_k` | `get_non_ag_biodiv_impact_k` 274 | `non_ag_biodiversity.get_non_ag_lu_biodiv_contribution` | same | same |
| `biodiv_contr_ag_man` | `get_ag_man_biodiv_impacts` 279 | `ag_biodiversity.get_ag_management_biodiversity_contribution` | AM bundles | AM bundle `.xlsx` |

### 4.5 Biodiversity — GBF targets

Every GBF field is gated: when its setting is `'off'` the builder returns `np.empty(0)` / `{}` / `[]`
and the whole stream costs nothing. The `*_region_*` companions are the constraint key lists.

| Field | Builder (line) | L5 entry | Key `data.` attributes | L3 files |
|-------|----------------|----------|------------------------|----------|
| `GBF2_mask_area_r` | `get_GBF2_mask_area_r` 283 | `ag_biodiversity.get_GBF2_MASK_area` | `BIO_GBF2_MASK`, `BIO_GBF2_MASK_LDS`, `BIO_GBF2_BASE_YR`, `REAL_AREA` | `bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.h5`, `Biodiversity_conserve_performance.xlsx`, `BIODIVERSITY_GBF2_TOP_RANK_CELL_BIO_SCORES_AND_TARGET.csv` |
| `GBF3_NVIS_pre_1750_area_vr` | `get_GBF3_NVIS_pre_1750_area_vr` 290 | `ag_biodiversity.get_GBF3_NVIS_matrices_vr` | `GBF3_NVIS_LAYERS_LDS` ← `get_NVIS_sparse_array` (`data.py:1651-1670`) | `bio_GBF3_{GBF3_NVIS_TARGET_CLASS}_sparse.npz` + `_groups.npy` |
| `GBF3_NVIS_region_group` | 297 | — (direct) | `BIO_GBF3_NVIS_SEL` ← `get_NVIS_targets_df` (`data.py:1672-1714`) | `BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv` |

**`GBF3_NVIS_REGION_MODE` does not swap the layer file.** The spatial layers always come from
`bio_GBF3_{GBF3_NVIS_TARGET_CLASS}_sparse.npz` (`data.py:1664`). The mode — `AUSTRALIA`, `NRM` or
`IBRA_REG` — is a filter on the **targets CSV**: `get_NVIS_targets_df` queries
`region_level == '{GBF3_NVIS_REGION_MODE}'` alongside `sheet_name == '{TARGET_CLASS}'` and
`resfactor == {RESFACTOR}` (`data.py:1700-1706`), and `AUSTRALIA` additionally collapses every
regional row to a single `'AUSTRALIA'` label so the solver bypasses NRM masking. There is no separate
IBRA layer, attribute or constraint method — see §9.3.
| `GBF4_SNES_pre_1750_area_sr` | 303 | `ag_biodiversity.get_GBF4_SNES_matrix_sr` | `GBF4_SNES_LAYERS_LDS` | `bio_GBF4_SNES_sparse.npz`, `bio_GBF4_SNES_sparse_species.npy` |
| `GBF4_SNES_region_species` | 309 | — (direct) | `BIO_GBF4_SNES_SEL` | `BIODIVERSITY_GBF4_TARGET_SNES.csv` |
| `GBF4_ECNES_pre_1750_area_sr` | 315 | `ag_biodiversity.get_GBF4_ECNES_matrix_sr` | `GBF4_ECNES_LAYERS_LDS` | `bio_GBF4_ECNES_sparse.npz`, `bio_GBF4_ECNES_sparse_species.npy` |
| `GBF4_ECNES_region_species` | 321 | — (direct) | `BIO_GBF4_ECNES_SEL` | `BIODIVERSITY_GBF4_TARGET_ECNES.csv` |
| `GBF8_pre_1750_area_sr` | 327 | `ag_biodiversity.get_GBF8_matrix_sr` | `BIO_GBF8_SEL_SPECIES`, `BIO_GBF8_GROUPS_LAYER` | `bio_GBF8_ssp{SSP}_EnviroSuit.nc`, `..._group.nc` |
| `GBF8_region_species` | 333 | — (direct) | `BIO_GBF8_SEL`, `BIO_GBF8_BASELINE_SCORE_GROUPS`, `BIO_GBF8_OUTSDIE_LUTO_SCORE_GROUPS` | `BIODIVERSITY_GBF8_{SCORES,TARGET}{,_group}.csv` |

### 4.6 Quantity / demand

| Field | Builder (line) | L5 entry | Key `data.` attributes | L3 files |
|-------|----------------|----------|------------------------|----------|
| `ag_q_mrp` | `get_ag_q_mrp` 360 | `ag_quantity.get_quantity_matrices` | `AGEC_CROPS`, `AGEC_LVSTK`, `FEED_REQ`, `PASTURE_KG_DM_HA`, `SAFE_PUR_NATL`, `SAFE_PUR_MODL`, `CLIMATE_CHANGE_IMPACT`, `PRODUCTIVITY_MUL_*` | `agec_*.h5`, `feed_req.h5`, `pasture_kg_dm_ha.h5`, `safe_pur_{natl,modl}.h5`, `climate_change_impacts_{rcp}_{on,off}.h5`, `yieldincreases_bau2022.csv`, `yieldincreases_ag_2050.xlsx` |
| `non_ag_q_crk` | `get_non_ag_q_crk` 366 | `non_ag_quantity.get_quantity_matrix` | `LU2PR`, `PR2CM` (structural — non-ag LUs produce no commodity) | — |
| `ag_man_q_mrp` | `get_ag_man_q_mrj` 590 | `ag_quantity.get_agricultural_management_quantity_matrices` | AM bundles | AM bundle `.xlsx` |
| `commodity_names` | inline 1307 | — | `COMMODITIES` | derived from `ag_landuses.csv` |
| `limits['demand']` | `get_limits` 817 | — | `D_CY` ← `DEMAND_DATA` × AusTIMES multipliers | `demand_projections.h5`, `AusTIMES_demand_multiplier.xlsx`, `demand_elasticity.csv` |

### 4.7 Transitions, bounds, feasibility

The transition system is **source-keyed**: costs and feasibility are sliced per base-year source
(`(from_m, from_j)` for ag, `k` for non-ag), and the solver creates one delta var per feasible
`(source, cell, target)`.

| Field | Builder (line) | L5 entry | Key `data.` attributes | L3 files |
|-------|----------------|----------|------------------------|----------|
| `ag_x_mrj` | `get_ag_x_mrj` 394 | `ag_transition.get_to_ag_exclude_matrices` | `EXCLUDE` (← `x_mrj.npy`), `T_MAT`, `NO_GO_{LANDUSE,REGION}_AG` | `x_mrj.npy`, `ag_tmatrix.npy`, `no_go_areas/` |
| `flow_cost_ag2ag` | `get_ag_t_mrj` 372 | `ag_transition.get_transition_matrices_ag2ag` | `T_MAT`, `TRANS_COST_MULTS`, `AG_TMATRIX`, `WATER_LICENCE_PRICE`, `IRRIG_COST_MULTS`, `REGIONAL_ADOPTION_ZONES` | `ag_tmatrix.npy`, `transition_cost_clearing_forest.npz`, `cost_multipliers.xlsx`, `water_licence_price.h5`, `regional_adoption_zones.h5` |
| `flow_cost_ag2nonag` | inline 1080-1086 | `non_ag_transition.get_transition_matrix_ag2nonag` | `EP_EST_COST_HA`, `RP_EST_COST_HA`, `AF_EST_COST_HA`, `CP_EST_COST_HA`, `AG2EP_TRANSITION_COSTS_HA`, `AG_TO_DESTOCKED_NATURAL_COSTS_HA`, `RP_FENCING_LENGTH`, `EST/FENCE/IRRIG_COST_MULTS` | `ep_est_cost_ha.h5`, `cp_est_cost_ha.h5`, `ag_to_ep_tmatrix.npy`, `ag_to_destock_tmatrix.npy`, `stream_length_m_cell.h5`, `cost_multipliers.xlsx` |
| `flow_cost_nonag2ag` | inline 1090-1096 | `non_ag_transition.get_transition_matrix_nonag2ag` | `EP2AG_TRANSITION_COSTS_HA`, `T_MAT` | `ep_to_ag_tmatrix.npy` |
| `dvar_ub_ag` / `dvar_lb_ag` | 484 / 517 | `ag_transition.get_ag2ag_{ub,lb}` + `non_ag_transition.get_nonag2ag_ub` | `T_MAT`, `EXCLUDE`, base dvars | `ag_tmatrix.npy`, `x_mrj.npy` |
| `dvar_ub_nonag` / `dvar_lb_nonag` | 504 / 524 | `non_ag_transition.get_non_ag_{ub,lb}_matrices` | `RP_PROPORTION`, `LU_LVSTK_NATURAL`, `NO_GO_*_NON_AG`, reversibility flags | `stream_length_m_cell.h5`, `no_go_areas/` |
| `feasible_*` (4 fields) | 399, 410, 426, 450, 466 | — (pure logic over `ag_x_mrj`, `dvar_ub_nonag`, `T_MAT` reach) | `T_MAT` | `ag_tmatrix.npy` |
| `ag_source_cells` / `nonag_source_cells` | 416 / 421 | `ag_transition.get_base_dvar_mj_cell_map`, `non_ag_transition.get_base_nonag_dvar_k_cell_map` | base-year dvars | — (runtime state) |
| `ag_man_limits` | `get_ag_man_limits` 620 | `ag_transition.get_agricultural_management_adoption_limits` | AM bundles | AM bundle `.xlsx` |
| `ag_man_lb_mrj` | `get_ag_man_lb_mrj` 533 | `ag_transition.get_lower_bound_agricultural_management_matrices` | base-year AM dvars | — (runtime state) |
| `dvar_base_ag_mrj` / `dvar_base_non_ag_rk` | inline 1346-1347 | `ag_transition.get_folded_base_ag_dvar`, `data.non_ag_dvars` | `lumaps`, `lmmaps` | `lumap.h5`, `lmmap.h5` (base year only; later years are runtime state) |
| `ag_fold_map`, `acct_cells_mrj` | 1310, 1316-1324 | `ag_transition.get_ag_dvar_fold_map` | θ-fold bookkeeping | — (runtime) |

### 4.8 Renewable energy

| Field | Builder (line) | L5 entry | Key `data.` attributes | L3 files |
|-------|----------------|----------|------------------------|----------|
| `renewable_solar_r` | `get_potential_renewable_solar_r` 538 | `ag_quantity.get_quantity_renewable(data, 'Utility Solar PV', ...)` | `RENEWABLE_LAYERS`, `RENEWABLE_BUNDLE_SOLAR`, `REAL_AREA` | `renewable_energy_layers_1D.nc`, `renewable_energy_bundle.csv` |
| `renewable_wind_r` | 543 | same, `'Onshore Wind'` | `RENEWABLE_LAYERS`, `RENEWABLE_BUNDLE_WIND` | as above |
| `exist_renewable_solar_r` / `_wind_r` | 548 / 557 | `ag_quantity.get_existing_renewable_dvar_fraction(..., 99999)` | `RENEWABME_EXISTING_DVAR_FRACTION_{SOLAR,WIND}` | `renewable_existing_capacity_area_fraction_1D.nc` |
| `limits['renewable_*']` | `get_limits` 825-832 | `ag_quantity.get_exist_renewable_capacity_by_state` | `RENEWABLE_TARGETS`, `RENEWABLE_EXISTING_CAPACITY_LAYER_{SOLAR,WIND}_MWH_CELL` | `renewable_targets.csv`, `renewable_existing_capacity_MW_1D.nc` |
| `renewable_MNES_mask_{solar,wind}_idx` | 787 / 793 | — (direct) | `RENEWABLE_MNES_MASK_{SOLAR,WIND}` | `renewable_QLD_EPBC_MNES_prioritization.nc` + `_performance.csv` |
| `renewable_GBF2_mask_{solar,wind}_idx` | 775 / 781 | — (direct) | `RENEWABLE_GBF2_MASK_{SOLAR,WIND}` | GBF2 mask inputs (§4.5) |

The existing-capacity ceiling deliberately uses `yr_cal=99999` (all years) so simulated + existing
never exceeds 1 in any period (`input_data.py:550-554`).

### 4.9 Regions, masks, bookkeeping

| Field | Builder (line) | `data.` attribute | L3 file |
|-------|----------------|-------------------|---------|
| `region_state_r`, `region_state_name2idx` | 566, 570 | `REGION_STATE_CODE`, `REGION_STATE_NAME2CODE` | `REGION_STATE_r.h5` ← `cell_zones_df.h5` |
| `region_NRM_names_r` | 574 | `REGION_NRM_NAME` | `REGION_NRM_r.h5` ← `cell_zones_df.h5` |
| `savanna_eligible_r` | 765 | `SAVBURN_ELIGIBLE` | `cell_savanna_burning.h5` |
| `GBF2_mask_idx` | 769 | `BIO_GBF2_MASK_LDS` | see §4.5 |
| `real_area` | 1339 | `REAL_AREA` | `real_area.h5` ← `cell_zones_df['CELL_HA']` |
| `ag_mask_proportion_r` | 1340 | `AG_MASK_PROPORTION_R` | derived from `lumap.h5` |
| `lu2pr_pj`, `pr2cm_cp`, `desc2aglu` | 1336-1338 | `LU2PR`, `PR2CM`, `DESC2AGLU` | derived from `ag_landuses.csv` |
| `base_yr_prod` | 1298-1305 | 6 base-year aggregates re-derived at `target_index=0` and cached on `data` | (recomputes the L5 matrices above) |
| `scale_factors` | 1283-1296 | rescaling bands — see §8 | — |

### 4.10 Limits (RHS)

`get_limits` (799-855) returns **raw, unscaled** targets; the solver divides each by its
`scale_factors` entry inline.

| Key | Source attribute | L3 file | Gate |
|-----|------------------|---------|------|
| `demand` | `D_CY` | `demand_projections.h5`, `AusTIMES_demand_multiplier.xlsx` | always |
| `water` | `WATER_YIELD_TARGETS` | water yield baselines/outside-LUTO layers | `WATER_LIMITS == 'on'` |
| `ghg` | `GHG_TARGETS[yr]` | `GHG_targets.xlsx` | `GHG_EMISSIONS_LIMITS != 'off'` |
| `renewable_*` | `RENEWABLE_TARGETS` (TWh→MWh) | `renewable_targets.csv` | any `RENEWABLES_OPTIONS` |
| `GBF2` | `get_GBF2_target_for_yr_cal` | `Biodiversity_conserve_performance.xlsx` | `GBF2_TARGET != 'off'` |
| `GBF3_NVIS` | `get_GBF3_NVIS_limit_score_inside_LUTO_by_yr` | `BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv` | `GBF3_NVIS_TARGET != 'off'` |
| `GBF4_SNES` / `GBF4_ECNES` | `get_GBF4_{SNES,ECNES}_target_inside_LUTO_by_year` | `BIODIVERSITY_GBF4_TARGET_{SNES,ECNES}.csv` | respective setting `!= 'off'` |
| `GBF8` | `get_GBF8_target_inside_LUTO_by_yr` | `BIODIVERSITY_GBF8_TARGET{,_group}.csv` | `GBF8_TARGET == 'on'` |
| `ag_regional_adoption`, `non_ag_regional_adoption{,_sum}` | `ag_transition.get_regional_adoption_limits` | `regional_adoption_zones.h5/.xlsx` | `REGIONAL_ADOPTION_CONSTRAINTS != 'off'` |

---

## 5. `input/` file inventory → raw provenance

The L3 contract, with its L4 consumer and L2/L1/L0 origin. "copy" in the L2 column means
`dataprep.py` does no transformation — trace straight through to L1.

### Structural / land use

| `input/` file | `data.py` | L2 (`dataprep.py`) | L1 / L0 origin |
|---------------|-----------|--------------------|----------------|
| `NLUM_2010-11_mask.tif` | 146 | copy 111 | `nlum_inpath` (National Landuse Map) |
| `ag_landuses.csv` | 227 | copy 112 | `nlum_inpath` |
| `lumap.h5` | 143 | built 340 | `raw_data/cell_LU_mapping.h5` ← script_2 |
| `lmmap.h5` | 442 | built 355 | `raw_data/cell_LU_mapping.h5` ← script_2 |
| `real_area.h5` | 435 | built 422 (`zones['CELL_HA']`) | `cell_zones_df.h5` ← script_1 |
| `x_mrj.npy` | 898 | built 545-608 (SA2 concordance pivot × rainfall ≥175 mm × irrigation potential, then OR'd with the observed 2010 map) | `NLUM_SPREAD_LU_ID_Mapped_Concordance.h5` + `cell_biophysical_df.h5` + `cell_zones_df.h5` |
| `state_id.npy` | **not read** | built 311 | `cell_zones_df.h5` (superseded by `REGION_STATE_r.h5`) |
| `REGION_NRM_r.h5` / `REGION_STATE_r.h5` | 517 / 524 | built 360 / 363 | `cell_zones_df.h5` ← script_1 |
| `regional_adoption_zones.h5` / `.xlsx` | 597 / 601 | built 379 / 411 | `cell_zones_df.h5` |
| `no_go_areas/` | 561 | copytree 163 | `no_go_areas` (toy dataset — flagged as such in `dataprep.py:56`) |

### Agricultural economics & production

| `input/` file | `data.py` | L2 | L1 / L0 |
|---------------|-----------|----|---------|
| `agec_crops.h5` | 212 | built 1019 | `SA2_crop_data.h5` ← script_2 |
| `agec_lvstk.h5` | 213 | built 1039 | `cell_livestock_data.h5` ← script_2 |
| `agGHG_crops.h5` | 885 | built 1078 | `SA2_crop_GHG_data.h5` ← script_2 |
| `agGHG_lvstk.h5` | 886 | built 1153 | `SA2_livestock_GHG_data.h5` ← script_2 |
| `agGHG_irrpast.h5` | 887 | built 1159 | `SA2_irrigated_pasture_GHG_data.h5` ← script_2 |
| `agGHG_lvstk_off_land.csv` | 1305 | built 1209 | `GHG_off_land_inpath` |
| `feed_req.h5` | 646 | built 510 | `cell_livestock_data.h5` |
| `pasture_kg_dm_ha.h5` | 648 | built 514 | `cell_livestock_data.h5` |
| `safe_pur_natl.h5` / `safe_pur_modl.h5` | 651 / 654 | built 517 / 520 | `cell_livestock_data.h5` |
| `soil_carbon_t_ha.h5` | 711 | built 540 (`SOC_T_HA_TOP_30CM`) | `cell_biophysical_df.h5` ← script_4 |
| `climate_change_impacts_{rcp}_{on,off}.h5` | 486 | built 939-981 (parallel) | `SA2_climate_damage_mult.h5` ← script_3 |
| `yieldincreases_bau2022.csv` | 827 | copy 97 | `fdh_inpath` |
| `yieldincreases_ag_2050.xlsx` | 839 | copy 98 | `ag_yield_trend` (ABARES) |
| `ag_price_multipliers.xlsx` | 216-217 | copy 116 | `luto_1D_inpath` |
| `cost_multipliers.xlsx` | 411-425 | copy 117 | `luto_1D_inpath` |

### Transitions

| `input/` file | `data.py` | L2 | L1 / L0 |
|---------------|-----------|----|---------|
| `ag_tmatrix.npy` | 891 | built 486 | `transitions_costs_20251002.xlsx` + `tmatrix_cat2lus.csv` (`fdh_inpath`) |
| `ag_to_ep_tmatrix.npy` | 972 | built 502 | same |
| `ep_to_ag_tmatrix.npy` | 977 | built 494 | same |
| `ag_to_destock_tmatrix.npy` | 892 | built 258 | `transitions_costs_20251002.xlsx` |
| `transition_cost_clearing_forest.npz` | 1002 | built 271 | `transitions_costs_20251002.xlsx` |
| `ep_est_cost_ha.h5` / `cp_est_cost_ha.h5` | 918 / 921 | built 919 / 922 — **CPI-adjusted 2021→2010 AUD (×99.2/118.8)**, `dataprep.py:913-916` | `cell_biophysical_df.h5` ← script_4 |
| `stream_length_m_cell.h5` | 863 | built 886 | `cell_biophysical_df.h5` |

### Carbon

| `input/` file | `data.py` | L2 | L1 / L0 |
|---------------|-----------|----|---------|
| `tCO2_ha_{ep_block,ep_rip,ep_belt,cp_block,cp_belt,hir_block,hir_rip}.nc` | 932-969 | subset 122-137 — ages `[50,60,70,80,90]` only, recompressed | script_9 ← FullCAM `carbonstock_RES_1_*` NetCDFs + `HIR_NFMR_AC_*.npy` |
| `natural_land_t_co2_ha.h5` | 1155 | built 897-901 | `cell_biophysical_df.h5` L153-192 ← script_4 (ERF max-aboveground-biomass rasters) |
| `fire_risk.h5` | 924 | built 908-910 | `cell_biophysical_df.h5` L633-651 ← `N:/Data-Master/Fire_drought_risk/ep_CO2_percentage.csv` |
| `cell_BECCS_df.h5` | 1641 | built 1244-1291 | `df_info_best_grid_20211116.pkl` (CSIRO, Lei Gao) |
| `carbon_prices.xlsx` | 1337 | copy 115 | `luto_1D_inpath` |
| `GHG_targets.xlsx` | 1353 | copy 114 | `luto_1D_inpath/GHG_targets_20260223_2010-2060.xlsx` |
| `cell_savanna_burning.h5` | 1366 | re-saved 119 | `luto_2D_inpath` — **no producing script in `Scripts/`** |

### Water

| `input/` file | `data.py` | L2 | L1 / L0 |
|---------------|-----------|----|---------|
| `water_yield_ssp{126,245,370,585}_2010-2100_{dr,sr}_ml_ha.h5` | 1117-1118 | built 141-156 (transposed out of the 4D HDF5) | script_6 |
| `water_yield_baselines.h5` | 1105 | built 617 | script_6 outputs |
| `water_yield_outside_LUTO_study_area_hist_1970_2000.h5` | 1122 | built 743 | script_6 |
| `water_yield_outside_LUTO_study_area_2010_2100_{dd,rr}_ml.h5` | 2200 / 2220 | built 711-712 | script_6 |
| `water_yield_natural_land_2010_2100_{dd,rr}_ml.h5` | 2204 / 2224 | built 719-720 | script_6 |
| `rivreg_id.h5` / `rivreg_lut.h5` | 2194 / 2196 | built 635 / 632 | `cell_zones_df.h5` ← script_1 |
| `draindiv_id.h5` / `draindiv_lut.h5` | 2214 / 2216 | built 647 / 644 | `cell_zones_df.h5` |
| `water_licence_price.h5` | 1095 | built 532 (`WATER_PRICE_ML_ABARES`) | `cell_biophysical_df.h5` |
| `water_delivery_price.h5` | 1100 | built 528 (`lvstk['WP']`) | `cell_livestock_data.h5` |
| `Water_Use_Domestic.csv`, `Water_Use_Agriculture_ML.csv` | 1125 | copy 159-160 | `water_domestic_use` (Water Account) |

### Biodiversity

| `input/` file | `data.py` | L2 | L1 / L0 |
|---------------|-----------|----|---------|
| `bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv` | 1394 | copy 174 (renamed from `HABITAT_CONDITION.csv`) | HCAS processed |
| `bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.h5` | 1399, 2791 | built 765 | HCAS / Zonation |
| `Biodiversity_conserve_performance.xlsx` | 1446 | copy 177 | script_5_4 |
| `BIODIVERSITY_GBF2_TOP_RANK_CELL_BIO_SCORES_AND_TARGET.csv` | — (diagnostic) | built 873 | derived in `dataprep` |
| `bio_GBF3_NVIS_{MVG,MVS}_sparse.npz`, `..._groups.npy` | 1665 | copy 180-183 | `bio_GBF3_NVIS_inpath` (NVIS 7.0 pre-1750) |
| `BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv` | 1701 | copy 184 | script_5_2 |
| `bio_GBF3_IBRA_{Regions,SubRegions}.nc` | **not read** | built 455-460 | IBRA layers — produced but never loaded (§9.3) |
| `bio_GBF4_{SNES,ECNES}_sparse.npz`, `..._sparse_species.npy` | 1773 / 1800 | copy 187-190 | script_5_1 (DCCEEW SNES/ECNES) |
| `BIODIVERSITY_GBF4_TARGET_{SNES,ECNES}.csv` | 2306 / 2361 | copy 191-192 | script_5_2 |
| `bio_GBF8_ssp{SSP}_EnviroSuit{,_group}.nc` | 1608 / 1631 | copy 195-202 | `bio_GBF8_inpath` (environmental suitability, 5 km) |
| `BIODIVERSITY_GBF8_{SCORES,TARGET}{,_group}.csv` | 1609-1622 | copy 204-207 | `bio_GBF8_inpath` |
| `bio_NES_Zonation.nc` | 2800 | copy 210 | script_5_1 / 5_3 |
| `bio_RHI_Zonation.nc` | 2809 | copy 215 | script_5_4 (DCCEEW RHI) |

### Renewables

| `input/` file | `data.py` | L2 | L1 |
|---------------|-----------|----|----|
| `renewable_targets.csv` | 731 | copy 218 | script_10_1 (AEMO ISP scenarios) |
| `renewable_energy_bundle.csv` | 725 | copy 220 | script_10_1 |
| `renewable_price_AUD_MWh_{solar,wind}.csv` | 737-738 | copy 221-222 | script_10_1 |
| `renewable_energy_layers_1D.nc` | 744 | copy 219 | script_10_3 |
| `renewable_existing_capacity_{MW,area_fraction}_1D.nc` | (existing-capacity attrs) | copy 225-226 | script_10_2 |
| `renewable_QLD_EPBC_MNES_prioritization.nc` / `_performance.csv` | 1487 / 1488 | copy 223-224 | `renewable_energy_inpath` |

### Demand & AM bundles

| `input/` file | `data.py` | L2 | L1 / L0 |
|---------------|-----------|----|---------|
| `demand_projections.h5` | 1249 | built 1240 | `All_LUTO_demand_scenarios_with_convergences.csv` (`demand_scenarios_inpath`) |
| `demand_elasticity.csv` | 1289 | copy 103 | `demand_elasticity_inpath/20260311_values for runs.csv` |
| `AusTIMES_demand_multiplier.xlsx` | 1214 | copy 104 | `luto_1D_inpath/20260220_CNS25 Pathways AusTIMES data.xlsx` |
| `20260317_Bundle_MR.xlsx` | 672-675 | copy 167 | `luto_1D_inpath` (Asparagopsis) |
| `20231107_ECOGRAZE_Bundle.xlsx` | 681-683 | copy 170 | `luto_1D_inpath` |
| `20260317_Bundle_AgTech_NE.xlsx` | 689-691 | copy 168 | `luto_1D_inpath` (precision agriculture) |
| `20260317_Bundle_AgTech_EI.xlsx` | 697-699 | copy 169 | `luto_1D_inpath` |
| `20260401_Bundle_BC.xlsx` | 705-706 | copy 171 | `luto_1D_inpath` (biochar) |

**Demand-multiplier gate** (`data.py:1203-1229`): the AusTIMES multipliers belong to a carbon
pathway. `GHG_EMISSIONS_LIMITS = 'off'` maps to `None` in `GHG_TARGETS_DICT` and has no sheet, so the
workbook is read only to borrow its year index and commodity columns, then every multiplier is
forced to `1.0`. The frame keeps its shape however the carbon constraint is set.

---

## 6. Detailed trace — reforestation carbon (EP / CP / HIR)

The carbon pipeline is the most-changed part of the model and the one most often mis-remembered: it
is **NetCDF/xarray, not HDF5/pandas**, and the attributes carry a `_PER_YR` suffix.

```
FullCAM REST API 2025 GeoTIFFs
  → script_9_reforestation_carbon_data.py        (cap, CO2e conversion, riparian burn-in, species blend)
  → 3D_Spatial_Timeseries/tCO2_ha_*.nc           (age 0-90 × 6,956,407 cells)
  → dataprep.py:122-137                          (subset ages [50,60,70,80,90], recompress)
  → input/tCO2_ha_*.nc
  → data.py:929-969                              (select CARBON_EFFECTS_WINDOW, mask, risk-discount, annualise)
  → data.{EP_BLOCK,EP_BELT,EP_RIP,CP_BLOCK,CP_BELT}_AVG_T_CO2_HA_PER_YR
  → non_agricultural/ghg.py + revenue.py         (× -1, × REAL_AREA)
  → non_ag_g_rk / non_ag_obj_rk
```

**L1 — `script_9_reforestation_carbon_data.py`**

| Stage | Lines | What it does |
|-------|-------|--------------|
| Load | 120-122, 199-214, 273-276 | EP: `carbonstock_RES_1_specId_7_specCat_{BlockES,Water,BeltH}.nc`. CP: Mallee (`specId_23`) + E. globulus (`specId_8`). HIR: `HIR_NFMR_AC_{gt,lt}_500mm{,_rip}.npy` |
| Gap-fill | 124-128, 218-223 | `fill_nan_nearest_2d` per year per C-pool |
| Eglob-Belt proxy | 204-214 | FullCAM v2024 Eglob-Belt is unreliable → `Eglob_Belt = Eglob_Block × (EP_Belt / EP_Block)` |
| Riparian burn-in | 136, 279-280 | `block = (1 − rip_area_prop)·block + rip_area_prop·rip` |
| Species blend | 233-237 | Mallee→E. globulus ramp over 550-650 mm rainfall; HIR ramps over 450-550 mm (283-285) |
| Cap + CO2e | 139-149, 240-246, 289-295 | caps `max_tree_C=1500`, `max_debris_C=300`, `max_soil_C=500` t C/ha, then `× 44/12`. **The EP/CP caps were paired with the wrong pools until 2026-08-12** — see §9.1 |
| Soil as change | 156-183, 249-267, 298-312 | soil variable is written as `array[age] − array[age=0]` — sequestration *after* planting, excluding the pre-existing stock |
| Write | 161-184, 254-267, 303-315 | `tCO2_ha_*.nc`, dims `(age, cell)`, vars `{PREFIX}_{TREES,DEBRIS,SOIL}_T_CO2_HA` |
| GeoTIFF export | 327-355 | Side artefact, **not** on the model's data path: `save_dataset_as_multiband_tiffs` writes 18 multiband tifs (91 bands, float32, LZW, nodata −99) to `N:/Data-Master/FullCAM/Output_TOT_CO2_HA_GeoTiffs/` — EP block/rip/belt, CP block/belt and HIR **block** only (HIR rip is not exported). For visualisation/QA; nothing in `luto/` reads them. Refreshed 2026-08-13 alongside the cap fix; they had been stale since 2025-10-09. |

**L2 — `dataprep.py:122-137`**: `sel(age=[50,60,70,80,90])`, `assign_coords(cell=range(6956407))`,
re-encode zlib level 5. No unit change.

**L4 — `data.py:923-969`**:

```python
fr_df     = pd.read_hdf(".../fire_risk.h5", where=self.MASK)          # 924
fire_risk = fr_df[{"low":"FD_RISK_PERC_5TH","med":"FD_RISK_MEDIAN","high":"FD_RISK_PERC_95TH"}[settings.FIRE_RISK]]

ds = xr.open_dataset(".../tCO2_ha_ep_block.nc") \
       .sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK)   # 932
self.EP_BLOCK_AVG_T_CO2_HA_PER_YR = (
    (ds['EP_BLOCK_TREES_T_CO2_HA'] + ds['EP_BLOCK_DEBRIS_T_CO2_HA'])
    * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)     # aboveground: risk-discounted
    + ds['EP_BLOCK_SOIL_T_CO2_HA']                            # belowground: undiscounted
).values / settings.CARBON_EFFECTS_WINDOW                     # cumulative → annual rate
```

Identical shape for `EP_BELT` (939-945), `EP_RIP` (947-953), `CP_BLOCK` (955-961), `CP_BELT`
(963-969). `.sel(age=...).load().isel(cell=self.MASK)` is deliberate — label-based `.sel(cell=...)`
on a 6.9 M cell axis is pathologically slow (`data.py:930-931`).

Current settings: `CARBON_EFFECTS_WINDOW = 60`, `FIRE_RISK = 'med'`, `RISK_OF_REVERSAL = 0`
(`settings.py:70, 84, 92`). `CARBON_EFFECTS_WINDOW` must be one of `[50,60,70,80,90]` — the ages the
L2 subset kept.

**Unallocated natural land** takes a different route (`data.py:1140-1160`):

```python
nat_land_CO2 = pd.read_hdf(".../natural_land_t_co2_ha.h5", where=self.MASK)
self.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR = np.array(
    nat_land_CO2['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA']
    - nat_land_CO2['NATURAL_LAND_AGB_DEBRIS_TCO2_HA'] * (100 - fire_risk) / 100   # minus fire DAMAGE
) / settings.CARBON_EFFECTS_WINDOW
```

The `/ CARBON_EFFECTS_WINDOW` lives **here**, not in `ghg.py` — see §7.

`HIR` layers (`tCO2_ha_hir_{block,rip}.nc`) are prepared through L2 but are not currently loaded in
`data.py`; the HIR mask was retired on 2026-06-16 (`dataprep.py:904-906`).

---

## 7. Detailed trace — non-agricultural GHG functions

`luto/economics/non_agricultural/ghg.py`. Sign convention throughout: **negative = sequestration**.
Every function multiplies by `data.REAL_AREA` (per-ha → per-cell); none re-applies risk discounting
or annualisation, both of which already happened at L4.

| Function | Lines | Formula | Source attribute |
|----------|-------|---------|------------------|
| `get_ghg_env_plantings` | 29-51 | `-EP_BLOCK_AVG_T_CO2_HA_PER_YR × REAL_AREA` | `tCO2_ha_ep_block.nc` |
| `get_ghg_rip_plantings` | 54-75 | `-EP_RIP_AVG_T_CO2_HA_PER_YR × REAL_AREA` | `tCO2_ha_ep_rip.nc` |
| `get_ghg_agroforestry_base` | 78-93 | `-EP_BELT_AVG_T_CO2_HA_PER_YR × REAL_AREA` | `tCO2_ha_ep_belt.nc` |
| `get_ghg_sheep_agroforestry` | 96-130 | `base_af × x_r + ag_g_mrj[0,:,sheep_j] × (1 − x_r)` | + `agGHG_lvstk.h5` |
| `get_ghg_beef_agroforestry` | 133-167 | same with `beef_j` | + `agGHG_lvstk.h5` |
| `get_ghg_carbon_plantings_block` | 170-192 | `-CP_BLOCK_AVG_T_CO2_HA_PER_YR × REAL_AREA` | `tCO2_ha_cp_block.nc` |
| `get_ghg_carbon_plantings_belt_base` | 195-210 | `-CP_BELT_AVG_T_CO2_HA_PER_YR × REAL_AREA` | `tCO2_ha_cp_belt.nc` |
| `get_ghg_sheep_carbon_plantings_belt` | 213-247 | `base_cp × x_r + sheep × (1 − x_r)` | + `agGHG_lvstk.h5` |
| `get_ghg_beef_carbon_plantings_belt` | 250-284 | same with `beef_j` | + `agGHG_lvstk.h5` |
| `get_ghg_beccs` | 287-308 | `-np.nan_to_num(BECCS_TCO2E_HA_YR) × REAL_AREA` | `cell_BECCS_df.h5` |
| `get_ghg_destocked_land` | 311-348 | per base-year livestock-natural LU: `CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR × (habitat_contr[from_lu] − 1) × REAL_AREA` | `natural_land_t_co2_ha.h5`, `fire_risk.h5`, `bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv` |
| `get_ghg_matrix` | 352-398 | assembles all nine into `(r, k)` | — |

**Exclusion (mixing) proportions** — `luto/tools/__init__.py`:

- `get_exclusions_agroforestry_base` (351-368): `np.ones(NCELLS) × settings.AF_PROPORTION`, where
  `AF_PROPORTION = AGROFORESTRY_ROW_WIDTH / (ROW_WIDTH + ROW_SPACING)` (`settings.py:631`).
- `get_exclusions_carbon_plantings_belt_base` (371-388): same shape with `CP_BELT_PROPORTION`
  (`settings.py:614`).

**Destocked land — two corrections to earlier versions of this document:**

1. There is **no** `/ settings.CARBON_EFFECTS_WINDOW` in `ghg.py`. The annualisation moved into
   `data.py:1160` when the attribute was renamed to `..._PER_YR`. Do not divide twice.
2. The sign is **negative** for a sequestration benefit, matching every other non-ag function:
   `BIO_HABITAT_CONTRIBUTION_LOOK_UP` is normalised so unallocated natural land = 1 and livestock-on-
   natural < 1 (`data.py:1420-1433`), so `(contribution − 1) < 0`.

`get_ghg_matrix` keys the dict by **land-use display name** (`'Environmental Plantings'`,
`'Destocked - natural land'`, …) at lines 377-385, while the `aggregate=False` DataFrames carry
SCREAMING_SNAKE column names (`'ENV_PLANTINGS'`, `'DESTOCKED_LAND'`, …). Both naming systems are
live; do not assume they match.

---

## 8. The last mile — rescaling and coefficient filtering

Between L6 and L7 the arrays are rescaled for numerical conditioning. This is the only place where
input magnitudes change without a physical reason, so it is worth knowing when reading solver logs.

**Rescaling** (`input_data.py:1198-1296`). Each band is rescaled independently:

| Band | Function | Paired RHS |
|------|----------|------------|
| `Economy` | `rescale_lhs` | — (objective) |
| `Demand` | `rescale_lhs_rhs` | `limits['demand']` |
| `Biodiversity` | `rescale_lhs` | — |
| `GHG` | `rescale_lhs_rhs` | `limits['ghg']` (skipped, scale 1.0, when GHG off) |
| `Water` | `rescale_lhs_rhs` | `limits['water']` (skipped when `WATER_LIMITS != 'on'`) |
| `GBF2` | `rescale_lhs_rhs` | `limits['GBF2']` |
| `GBF3_NVIS`, `GBF4_SNES`, `GBF4_ECNES`, `GBF8` | `rescale_lhs_rhs_region_species` | per-(region, species/group) targets |
| `Utility Solar PV`, `Onshore Wind` | `rescale_lhs_rhs` | per-state MWh targets |

The source-keyed dicts are rescaled explicitly (1203-1205, 1217) because `rescale_lhs` only walks one
dict level. `limits` itself stays **raw** — the solver divides by `scale_factors[...]` inline.

**Coefficient filter** (L7): `_qsum(coeffs, gurobi_vars)` in `solver.py` is called by *every*
constraint and objective builder and drops any term with `|coeff| < settings.SOLVER_COEFF_MIN`
(1e-4). This supersedes the removed `RESCALE_ZERO_THRESHOLD` post-rescale zeroing — there is no
zeroing in `input_data.py` any more.

---

## 9. Issue register

Things found while tracing this map that are wrong, unverifiable, or dead. One has been fixed; the
rest are **flagged, not changed** — each needs a decision from whoever owns that data or code.

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| [9.1](#91-resolved--carbon-pool-caps-applied-to-the-wrong-pools) | EP/CP carbon-pool caps applied to the wrong pools | changed model inputs | ✅ **RESOLVED** 2026-08-12/13 |
| [9.2](#92-open--no-re-run-trigger-linking-fullcam-refreshes-to-script_9) | No re-run trigger linking FullCAM refreshes to `script_9` | silent 4-month data staleness | ⚠️ **OPEN** — process gap |
| [9.3](#93-open--dead-ibra-path-in-the-gbf3-stream) | Dead IBRA path (`get_GBF3_IBRA_matrices_vr`) | would `AttributeError` if called; misleads readers | ⚠️ **OPEN** — needs a decision |
| [9.4](#94-open--cell_savanna_burningh5-has-no-traceable-origin) | `cell_savanna_burning.h5` has no producing script | provenance untraceable | ⚠️ **OPEN** — needs an owner |
| [9.5](#95-open--no_go_areas-is-a-toy-dataset-feeding-real-constraints) | `no_go_areas/` is a self-described toy dataset | feeds real exclusion logic | ⚠️ **OPEN** — verify intent |
| [9.6](#96-open--input-files-nothing-reads) | Four `input/` files nothing reads | wasted `dataprep` time and disk | ⚠️ **OPEN** — safe to prune after checking |

---

### 9.1 RESOLVED — carbon-pool caps applied to the wrong pools

Caps were paired with the wrong pools in `script_9`, for EP and CP (HIR was correct). **Verified
against the data**, not inferred. Fixed, regenerated and propagated over 2026-08-12/13; the full
audit trail is kept below because it explains why the shipped carbon numbers changed.

The `VARIABLE` axis of the FullCAM source NetCDFs is `['DEBRIS_C_HA', 'SOIL_C_HA', 'TREE_C_HA']`
(read directly from `carbonstock_RES_1_specId_{7,23}_specCat_BlockES.nc`, dims
`(y, x, YEAR, VARIABLE)`, shape `(3364, 4071, 91, 3)`). After
`.data[np.nonzero(NLUM_mask)].transpose(1, 2, 0)` the array is `(age, VARIABLE, cell)`, so axis-1
index **0 = DEBRIS, 1 = SOIL, 2 = TREES**. The output `Dataset` block named them correctly; the cap
block did not (line numbers below are pre-fix — the corrected blocks are now at 142-152 and 246-252):

| axis-1 index | pool | cap intended | cap applied |
|---|---|---|---|
| 0 | DEBRIS | `max_debris_C` = 300 | `max_tree_C` = **1500** |
| 1 | SOIL | `max_soil_C` = 500 | `max_debris_C` = **300** |
| 2 | TREES | `max_tree_C` = 1500 | `max_soil_C` = **500** |

The **pre-fix** `input/tCO2_ha_*.nc` files showed the signature unambiguously. Maxima at age 60, as
shipped before 2026-08-12:

| file | TREES max | DEBRIS max | SOIL max |
|---|---|---|---|
| `ep_block` | **1833.33** (= 500 × 44/12, clipped) | 1568.30 (over the intended 1100) | 958.99 |
| `ep_rip` | **1833.33** | 1924.86 | 962.00 |
| `ep_belt` | **1833.33** | 2128.60 | 962.00 |
| `cp_block` | **1833.33** | 2072.45 | 996.30 |
| `cp_belt` | **1833.33** | 2833.11 | 1011.09 |
| `hir_block` | 5500.00 (= 1500 × 44/12) ✓ | 1100.00 (= 300 × 44/12) ✓ | 1165.26 |
| `hir_rip` | 5500.00 ✓ | 1100.00 ✓ | 1783.17 |

Every EP/CP tree layer pinned at exactly `500 × 44/12 = 1833.33` t CO2e/ha — one third of the intended
ceiling — while debris ran past its intended 1100 t CO2e/ha. HIR pinned at exactly the right two
values, confirming both the diagnosis and that the HIR block (axis order trees/debris/soil, per
`script_9:272`) was sound.

**Magnitude.** In `ep_block`, cells clipped at the tree cap: 6,108 at age 50 rising to 7,818 at age 90
(≈0.09-0.11% of the 6.96 M cell grid). Against the raw FullCAM layer at YEAR 2070 (age 60): tree
carbon reaches 2,088.8 t C/ha, 7,656 cells exceed 500 t C/ha but only **11** exceed the intended
1,500 — so ~7,645 cells are being clipped that should not be. The clipped group averages 703.5 t C/ha
raw, i.e. roughly 200 t C/ha ≈ 745 t CO2e/ha of sequestration discarded per affected cell. Raw debris
peaks at 606 t C/ha with 542 cells over the intended 300 cap, so the debris error runs the other way
but is far smaller.

Net effect: EP/CP tree sequestration is **understated in the highest-biomass cells** and debris is
mildly overstated.

**Fix applied** to `script_9_reforestation_carbon_data.py` (2026-08-12): the 15 cap lines for
`ep_block`, `ep_rip`, `ep_belt`, `cp_block`, `cp_belt` now pair index 0→`max_debris_C`,
1→`max_soil_C`, 2→`max_tree_C`, with a comment recording the `VARIABLE` axis order above each block.
The HIR block is unchanged — its arrays are ordered trees/debris/soil and its caps already matched.

**Regenerated 2026-08-12.** The EP/CP half of `script_9` was re-run (~35 min) and the five rebuilt
layers verified: `TREES` now tops out at 5500.00, `DEBRIS` at 1100.00 and `SOIL` below 1833.33 t
CO2e/ha — the same signature HIR already had. HIR was **not** regenerated: the fix does not touch it
and its inputs are unchanged since Oct 2025. The pre-fix layers were kept as a rollback until the new
data had been verified at both the 3D and `input/` level, then deleted (2026-08-13).

**Propagated to `input/` 2026-08-12.** The `dataprep.py:122-137` copy block was run standalone (all
seven layers, ~3 min) rather than the full `create_new_dataset()`, which would have wiped and rebuilt
the entire `input/` directory. Verified through the exact access pattern `data.py:932` uses
(`.sel(age=settings.CARBON_EFFECTS_WINDOW)`, currently 60): all seven files carry `age` coords
`[50, 60, 70, 80, 90]`, and every pool sits within its intended cap — `TREES` 5500.00, `DEBRIS`
1100.00, `SOIL` ≤1833.33 t CO2e/ha. **The correction is now live for simulation runs.**

**Measured effect on the regenerated data** (`ep_block`, age 60, old vs new): 2,738,204 cells changed,
but **2,737,660 of them are non-agricultural** — essentially the entire area outside the ag estate.
Only **544 ag cells** changed, against the 539 predicted from the cap analysis. Over ag land the EP-block
carbon pool rises **+0.04%** (CP-block +0.02%); the much larger whole-grid figure (EP +1.86%) is
dominated by cells the model cannot use.

**Side finding — the shipped layers were 4 months stale.** The old `tCO2_ha_*.nc` were built
2026-02-26, but the FullCAM sources (`carbonstock_RES_1_*.nc`) were refreshed 2026-06-02/05. The
regeneration therefore also picked up that newer vintage. This is why the non-ag cell count changed so
widely — the nearest-neighbour gap-fill differs in the nodata region between vintages. Worth a standing
check that `script_9` is re-run whenever FullCAM outputs are refreshed.

**Marginal vs average.** The aggregate effect is small because the mis-cap concentrates in
high-rainfall forest mostly outside the agricultural estate. The *marginal* effect can still matter:
those ~540 ag cells each gain roughly 750 t CO2e/ha, so any sitting near the economic margin become
markedly more attractive for plantings. Tightening the debris cap (300 instead of 1500, affecting 542
cells with raw debris >300 t C/ha) pushes the other way and partly offsets.

**Soil cap resolved too.** Earlier analysis could not tell whether it bound, because soil is stored as
a change from age 0. With the cap corrected from 300 to its intended 500 t C/ha, soil maxima rose from
~959-1011 to 1517-1728 t CO2e/ha across the five layers — so it *was* binding.

**Derived artefacts refreshed** (2026-08-13): the 18 multiband GeoTIFFs in
`N:/Data-Master/FullCAM/Output_TOT_CO2_HA_GeoTiffs/` were rebuilt from the corrected layers and
verified (91 bands, float32, LZW, nodata −99; band 61 reproduces the age-60 NetCDF slice). They are
visualisation/QA only — nothing in `luto/` reads them — so this changed no model result. They had been
stale since 2025-10-09.

---

### 9.2 OPEN — no re-run trigger linking FullCAM refreshes to `script_9`

The shipped `tCO2_ha_*.nc` were built 2026-02-26; the FullCAM sources they derive from
(`carbonstock_RES_1_*.nc`) were refreshed 2026-06-02/05. The model therefore ran for four months on
carbon layers that silently lagged their inputs, and the GeoTIFFs lagged by ten months. Nothing in the
pipeline detects this — the staleness is invisible unless someone compares file timestamps by hand.

**Impact**: silent, unbounded. The June refresh moved values across ~2.7 M cells; it happened to land
almost entirely outside the ag estate this time, but nothing guarantees that.

**Action**: make `script_9` → `dataprep` → GeoTIFF re-run routine whenever FullCAM outputs change, or
add a timestamp assertion (source mtime ≤ output mtime) at the top of `script_9` so a stale build
fails loudly. The same exposure applies to every L1 script in §3, not just `script_9`.

---

### 9.3 OPEN — dead IBRA path in the GBF3 stream

`ag_biodiversity.get_GBF3_IBRA_matrices_vr` (`biodiversity.py:388-397`) returns
`data.GBF3_IBRA_LAYERS_LDS * data.REAL_AREA`, but **no such attribute is ever set** — `IBRA` appears
in `data.py` only in the `GBF3_NVIS_REGION_MODE` validation (lines 1693-1697). Nothing calls the
function (`input_data.py:294` calls `get_GBF3_NVIS_matrices_vr` in every mode). The two
`bio_GBF3_IBRA_*.nc` files that `dataprep.py:455-460` writes are likewise never read.

**Impact**: would raise `AttributeError` the moment anything called it. More corrosively, its presence
implies `IBRA_REG` mode loads different spatial layers — it does not (see §4.5).

**Action**: either wire it up or delete the function together with the `dataprep` writes.

---

### 9.4 OPEN — `cell_savanna_burning.h5` has no traceable origin

Read straight out of `2D_Spatial_Snapshot/` at `dataprep.py:119`, but no script in
`N:/Data-Master/LUTO_2.0_input_data/Scripts/` produces it.

**Impact**: it feeds `SAVBURN_ELIGIBLE` and `SAVBURN_TOTAL_TCO2E_HA` — live inputs to ag GHG, ag
biodiversity and the savanna-burning land use — and cannot be rebuilt or audited from the repo.

**Action**: find the owner and record the provenance in §3, or bring its build into `Scripts/`.

---

### 9.5 OPEN — `no_go_areas/` is a toy dataset feeding real constraints

Labelled "just a toy example dataset" in `dataprep.py:56`, yet it drives `NO_GO_LANDUSE_AG`,
`NO_GO_REGION_AG` and the non-ag equivalents through `ag_transition` / `non_ag_transition`.

**Impact**: placeholder exclusions silently shape which land uses are reachable in which regions.

**Action**: confirm whether the toy data is still in play. If it is, either replace it with the real
layer or make the placeholder explicit at load time rather than in a `dataprep` comment.

---

### 9.6 OPEN — `input/` files nothing reads

Written by `dataprep` but never loaded by `data.py`: `state_id.npy`,
`bio_GBF3_IBRA_{Regions,SubRegions}.nc`, `tCO2_ha_hir_{block,rip}.nc`, `Water_Use_Agriculture_ML.csv`.

**Impact**: minor — wasted `dataprep` time and disk on every refresh. Listed mainly so a future reader
does not mistake them for live inputs.

**Action**: prune after confirming no external tooling (report scripts, notebooks under `luto/tools/`)
depends on them. Note `tCO2_ha_hir_*` is kept deliberately — the HIR mask was retired 2025-06-16
(`dataprep.py:904-906`) and the layers may return.
