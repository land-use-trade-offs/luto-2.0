# LUTO2 input data: what each file is and how `data.py` loads it

Scope: the runtime inputs under `settings.INPUT_DIR` (`input/`), the cell grid they share, and how `luto/data.py` reduces them to the study area. Line numbers refer to branch jinzhu at 72d30427 (2026-09). Everything marked "observed" was read from the files in `input/` on 2026-09-18; anything else is from reading code.

Related pages: [CLAUDE_TIME_AXIS.md](CLAUDE_TIME_AXIS.md) (base year and year indexing), [CLAUDE_DATAPREP.md](CLAUDE_DATAPREP.md) (how `input/` is produced), [CLAUDE_BASEYEAR_RUN.md](CLAUDE_BASEYEAR_RUN.md) (base-year-only runs and write functions).

## 1. The grid everything sits on

- The National Land Use Map (NLUM) 2010-11 raster `NLUM_2010-11_mask.tif`: 3364 rows x 4071 cols, EPSG:4283, 0.01 degree cells, bounds (112.925, -43.655, 153.635, -10.015). Value 1 on 6,956,407 land cells, 0 (nodata) on 6,738,437 (observed).
- Every cell-level runtime input is stored as a 1D vector (or an array with a `cell` axis) of exactly 6,956,407 entries, in raster row-major order of the land cells. There are no pre-masked or pre-coarsened cell files (observed for every h5/npy/npz/nc under `input/`; listing in `docs/FINDINGS.md` is not required, see the memo evidence path at the end).
- Two files are the exception and live on their own ~0.0417 degree grid (978 x 808): `bio_GBF8_ssp*_EnviroSuit.nc` and `bio_GBF8_ssp*_EnviroSuit_group.nc`. `data.py:2508-2516` bilinearly interpolates them onto the study-cell centroids, so they are grid-independent.

## 2. The base map (what "2010" is inside the model)

| File | What it holds | Where loaded | Notes |
|---|---|---|---|
| `lumap.h5` | integer land-use code per land cell, -1 for non-agricultural / outside study | `data.py:143` -> `LUMAP_NO_RESFACTOR` | masked to study cells at :156-190; registered as the base-year map by `add_lumap(YR_CAL_BASE, ...)` at :460 |
| `lmmap.h5` | irrigation flag per cell (0 dry, 1 irrigated) | `data.py:442` -> `LMMAP_NO_RESFACTOR`, masked :463 | registered at :464 |
| `NLUM_2010-11_mask.tif` | 2D land/ocean layout | `data.py:146-152` -> `NLUM_MASK`, `LUMAP_2D_FULLRES` | source of `self.MASK`; reopened at :566 to rasterise the no-go shapefiles |
| derived `AG_L_MRJ` | base-year agricultural decision variables (m = water supply, r = cell, j = land use) | `data.py:442-444`, `get_exact_resfactored_lumap_mrj()` | this is the object the solver, `write_dvar_area` and `write_quantity` treat as the base year. At RESFACTOR > 1 it carries within-block land-use fractions, so it is not the same as `lumap[cell] * RESMULT` (see section 5) |
| `x_mrj.npy` | boolean eligibility (2, 6,956,407, 28): may land use j occur in cell r under water supply m | `data.py:898-908` -> `self.EXCLUDE` | not economic data. Built by dataprep from the NLUM/SPREAD concordance and reconciled so every observed land use is eligible. `x_mrj_pre_reconcile_backup.npy` is the pre-reconciliation copy |
| `real_area.h5` | hectares per cell | `data.py:435` | `REAL_AREA = REAL_AREA_NO_RESFACTOR[MASK] * RESMULT`; `NCELLS = REAL_AREA.shape[0]` (:436-439) |

`ag_landuses.csv` (28 names, alphabetical, codes 0-27) gives the meaning of the lumap codes. Livestock land uses are split into "modified land" and "natural land" variants; "Unallocated - natural land" (code 23) is the largest class by far.

## 3. MASK and RESFACTOR

- `LUMASK = LUMAP_NO_RESFACTOR != MASK_LU_CODE` (`data.py:156`): True where a real land use exists.
- RESFACTOR == 1: `MASK = LUMASK` (:190).
- RESFACTOR > 1: the 2D LUMASK is coarsened into RESFACTOR x RESFACTOR blocks, the block-centre cell is kept, and `MASK` is a 1D boolean over the 6,956,407 land cells that is True at kept centres (:161-184). `RESMULT = RESFACTOR**2` scales per-cell areas.
- Observed cell counts: RESFACTOR 10 -> NCELLS 49,027 (RESMULT 100); RESFACTOR 5 -> NCELLS 186,648 (RESMULT 25).
- Load mechanisms, all equivalent in effect (pick the cells where MASK is True):
  - `pd.read_hdf(..., where=self.MASK)`: the agec, agGHG, climate, water-yield, soil, fire, BECCS, region, biodiversity-rank tables.
  - `xarray ... .isel(cell=self.MASK)` or `.values[self.MASK]`: renewable layers, tCO2 sequestration files, Zonation layers, `real_area`, `lmmap`.
  - Resfactor helpers `get_resfactored_sum` / `get_resfactored_average_fraction` (:1910-1966): quantities that must be aggregated over a block rather than sampled at its centre (`x_mrj`, existing renewable capacity).
  - Full length, never masked: `LUMAP_NO_RESFACTOR`, `LMMAP_NO_RESFACTOR`, the 2D full-res arrays.

## 4. Agricultural economic and yield data

| File | Loaded | Columns (observed) | Origin |
|---|---|---|---|
| `agec_crops.h5` | `data.py:212`, where=MASK | MultiIndex (field, lm, lu): fields AC, QC, FOC, FLC, FDC, WP, WR, P1, Yield; lm {dry, irr}; 20 crop land uses; 342 columns | SA2-level crop economics spread to cells via the `SA2_ID` join in dataprep (dataprep:995) |
| `agec_lvstk.h5` | `data.py:213`, where=MASK | MultiIndex (field, livestock): AC, QC, FOC, FLC, FDC, F1-F3, Q1-Q3, P1-P3, WR_DRN, WR_IRR x {BEEF, DAIRY, SHEEP}; 39 columns | per-cell from `cell_livestock_data.h5` (dataprep:1039) |
| `ag_price_multipliers.xlsx` | :216-217 | sheets AGEC_CROPS, AGEC_LVSTK; Year 2010..2100; 1.0 at 2010 | file dated 2024-06 |
| `cost_multipliers.xlsx` | :410 | 15 sheets, Year 2010..2100; 1.0 at 2010 | file dated 2024-06 |
| `feed_req.h5`, `pasture_kg_dm_ha.h5`, `safe_pur_natl.h5`, `safe_pur_modl.h5` | :646-655 | livestock carrying-capacity inputs | `cell_livestock_data.h5` |
| `water_licence_price.h5`, `water_delivery_price.h5` | :1095-1100 | $/ML | `cell_biophysical_df.h5`, `cell_livestock_data.h5` |
| `agGHG_crops.h5`, `agGHG_lvstk.h5`, `agGHG_irrpast.h5` | :885-887 | CO2E_KG_HA_* by (source, lm, lu); CO2E_KG_HEAD_* by (livestock, indicator); irrpast has an `SA2_ID` column | SA2 GHG tables spread via `SA2_ID` |

Field abbreviations: AC area cost ($/ha), QC quantity cost ($/t), FOC/FLC/FDC fixed operating / labour / depreciation cost, WP water price, WR water requirement (ML/ha), P price, Q quantity per head, F feed requirement. Consumers: `luto/economics/agricultural/{cost,revenue,quantity}.py`.

Vintage: none of the h5 files carry a collection date. The 1.0 rows at 2010 in the multiplier tables and `YR_CAL_BASE = 2010` are the only in-repo statements of the economic base year. Raw upstream snapshots (`SA2_crop_data.h5`, `cell_livestock_data.h5`) are not runtime inputs and are not under `input/`; see CLAUDE_DATAPREP.md.

## 5. Two ways to measure base-year area, and why they differ at coarse RESFACTOR

- `write_dvar_area` uses `ag_dvars[YR_CAL_BASE] = AG_L_MRJ` times `REAL_AREA`. At RESFACTOR 10 the national total is 464.06 Mha; at RESFACTOR 5 it is 464.59 Mha (observed).
- `lumap[cell]` grouped over `REAL_AREA` gives 539.33 Mha at RESFACTOR 10 and 513.38 Mha at RESFACTOR 5 (observed). The gap is almost entirely "Unallocated - natural land" and shrinks as RESFACTOR -> 1, because `AG_L_MRJ` carries exact within-block fractions while the lumap code is the block centre.
- Use the dvar-based figure (the csv from `write_dvar_area`) when comparing runs.

## 6. Non-spatial tables worth knowing

| File | Loaded | Shape / content (observed) |
|---|---|---|
| `demand_projections.h5` | :1239 | (38,400, 455). Row MultiIndex SCENARIO x DIET_DOM x DIET_GLOB x CONVERGENCE x IMPORT_TREND x WASTE x FEED_EFFICIENCY x COMMODITY (30). Columns (series, YEAR): DOMESTIC, EXPORTS, IMPORTS, FEED, PRODUCTION x 2010..2100 |
| `demand_elasticity.csv` | :1279 | supply and demand elasticities per commodity |
| `AusTIMES_demand_multiplier.xlsx` | :1208 | multipliers 2020..2060; applying them truncates the demand horizon to 2060 |
| `climate_change_impacts_<rcp>_CO2_FERT_<ON/OFF>.h5` (8 files) | :487 | cell table, columns (lm, lu, year) with year in {2020, 2050, 2080}; cell-level but listed here because of the year axis |
| `water_yield_ssp<SSP>_2010-2100_{dr,sr}_ml_ha.h5` | :1113-1118 | cell table, 91 columns labelled 0..90 (= 2010..2100 by position only) |
| `yieldincreases_bau2022.csv` | :826 | (100, 68), rows = 2010..2109 by position, row 0 = 1.0 |
| `yieldincreases_ag_2050.xlsx` | :838 | sheets low/medium/high/very_high, real Year column 2010..2050 |
| `GHG_targets.xlsx`, `carbon_prices.xlsx` | :1344, :1328 | Year 2010..2100 |
| `bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv` | :1384 | 28 rows; columns lu, PERCENTILE_10/25/50/75/90, CSV_DEFINED, AG_UNIFORM. Note: `settings.HCAS_CONTRIBUTION_PERCENTILE = 'USER_DEFINED'` expects a `USER_DEFINED` column that the 2026-09 input bundle does not have (it has `CSV_DEFINED`). Loading fails at `data.py:1417` until the setting or the file is aligned |
| `Biodiversity_conserve_performance.xlsx` | :1436 | conservation performance curves by sheet (SNES/ECNES/MNES likely, RHI, ssp*) |
| `BIODIVERSITY_GBF2_conservation_performance.xlsx` | not read at runtime | read only by dataprep:789; produced by an upstream N: script, not by the repo |

## 7. Files not read by `data.py`

`state_id.npy` (6,956,407 int8, dataprep output), `x_mrj_pre_reconcile_backup.npy`, `Biodiversity_conserve_performance.xlsx` is read but `BIODIVERSITY_GBF2_*` is not, `_cache/` (GBF8 caches at RESFACTOR 5/10/50; no code under `luto/` writes them), the older dated `*_Bundle_*.xlsx` (only the 2026 bundles and `20231107_ECOGRAZE_Bundle.xlsx` are loaded, :670-703), and `transfer_*.zip` if present (the delivery archive `input/` was extracted from).

## 8. Where the evidence is

Shapes, column dumps and the base-year aggregate tables behind this page are in the 2026-09-18 migration memo (`C:\scratch\luto_mig\memo.md` on optimus-nc, evidence under `C:\scratch\luto_mig\inspect\`). If that path is gone, re-derive with h5py `visititems` over `input/*.h5` and `xarray.open_dataset(engine='h5netcdf')` over `input/*.nc`.
