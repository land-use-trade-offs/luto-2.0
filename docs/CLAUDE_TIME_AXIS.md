# LUTO2 time axis: YR_CAL_BASE, year indexing and the places that assume 2010

Scope: how calendar years map onto input rows/columns, which code paths use labels and which use positions, and what breaks if the base year moves. Line numbers: branch jinzhu at 72d30427. Companion to [CLAUDE_DATA.md](CLAUDE_DATA.md).

## 1. The base year constant

- `self.YR_CAL_BASE = 2010` is hard-coded at `data.py:128` ("the base year, i.e. where year index yr_idx == 0"). There is no setting for it.
- `yr_idx = yr_cal - YR_CAL_BASE` (`data.py:2006`) is the generic year offset.
- Base-year state is registered under the calendar key: `add_ag_dvars(YR_CAL_BASE, AG_L_MRJ)` :444, `add_lumap` :460, `add_lmmap` :464, `add_ammaps` :472, `add_non_ag_dvars` :479, `add_ag_man_dvars` :1168. `data.lumaps`, `data.ag_dvars` etc. are dicts keyed by calendar year.
- `settings.SIM_YEARS` (default `range(2020, 2051, 5)`, `settings.py:151`) lists target years. `simulation.run` inserts `YR_CAL_BASE` if absent (`simulation.py:166-167`) and solves each consecutive pair. With the default settings the first solve step is the ten-year jump 2010 -> 2020.
- `data.last_year` starts as `None` (`data.py:110`) and is set only after a solve step (`simulation.py:229`). See CLAUDE_BASEYEAR_RUN.md for the consequence.

## 2. Inputs with real year labels (safe under a base-year change)

| Input | Where | Label form |
|---|---|---|
| `demand_projections.h5` | `data.py:1239` | column level YEAR = 2010..2100 (int). Labels are kept in `DEMAND_C.columns` until :1268 |
| `yieldincreases_ag_2050.xlsx` (PRODUCTIVITY_TREND != 'BAU') | :838-844, `index_col=0` | Year 2010..2050 |
| `ag_price_multipliers.xlsx`, `cost_multipliers.xlsx` | :216-217, :410, `index_col='Year'` | Year 2010..2100 |
| `GHG_targets.xlsx`, `carbon_prices.xlsx` | :1344, :1328 | Year |
| `climate_change_impacts_*.h5` | :487 | column level year in {2020, 2050, 2080}; `CLIMATE_CHANGE_IMPACT_xr` is a labelled DataArray (:493-511) |
| renewable layers, existing capacity | :744-799 | `year` coordinate (2030.., 2000..) |
| `AusTIMES_demand_multiplier.xlsx` | :1208 | Year 2020..2060 |

## 3. Positional sites (assume row 0 or index 0 == 2010)

These four were verified against the files on disk. Each is a silent misalignment, not an error, if YR_CAL_BASE changes and the files do not.

| # | Site | Mechanism | On-disk evidence |
|---|---|---|---|
| 1 | Demand `D_CY` | `D_CY = DEMAND_C.to_numpy().T` drops the year labels (:1268); `D_CY_xr` re-labels with `YR_CAL_BASE + arange` (:1272); `solvers/input_data.py:817` indexes `D_CY[yr_cal - YR_CAL_BASE]` | file has real labels 2010..2100 (91 columns per series) |
| 2 | Water yield SSP files | `WATER_YIELD_DR_FILE[yr_idx]`, `_SR_FILE[yr_idx]` (:2744, :2748; `economics/agricultural/water.py:90-91`; `non_agricultural/water.py:49-50, 70`) | 91 columns labelled 0..90, no year metadata anywhere in the h5 |
| 3 | Productivity trend, BAU path | `productivity_trend.index + YR_CAL_BASE` (:828) | csv has 100 rows, no year column, row 0 = 1.0 exactly, cumulative growth since row 0 |
| 4 | Climate-change impact base | `quantity.py:88-92` inserts the literal year 2010 with multiplier 1 (as a string column `'2010'`) then interpolates; `data.py:500-511` prepends a `YR_CAL_BASE` slice equal to 1 | file years are {2020, 2050, 2080}; there is no 2010 column, the 2020 column is a real (non-1) multiplier |

Consequence of site 4 under a 2020 base with the file unchanged: base-year production would be observed 2020 yields times the 2020 climate multiplier, i.e. climate impact counted twice.

**Status: sites 1-4 and off-land GHG now read by year label** (branch `rebase-2021`). `D_CY_xr` keeps `DEMAND_C.columns` and is read with `.sel(year=)` (`row_inputs.get_limits`, `write.write_quantity`); `OFF_LAND_GHG_EMISSION_C` is a Series by `YEAR`, read with `.loc[target_year]`; the water yield files are read through `Data.get_water_yield_file_row(yr_cal)` over `WATER_YIELD_FILE_YEARS` (2010-2100, from the file name); the BAU productivity csv is labelled from 2010; the climate files anchor on `Data.CLIMATE_CHANGE_IMPACT_ANCHOR_YEAR` (2010), not `YR_CAL_BASE`. At the 2010 base every read returns the same values as before. Moving the base still leaves the renormalisation question for sites 3 and 4 (multipliers relative to 2010, not to the new base) open.

## 4. Dynamic pricing depends on the base year twice

`data.py:2140-2158` builds the price elasticity multiplier from (a) `BASE_YR_production_t` (production implied by the base-year dvars, :1195) and (b) `D_CY_xr.sel(year=YR_CAL_BASE)`. Changing the base map changes (a); changing the base year changes (b). With `settings.DYNAMIC_PRICE = True` (default) price trajectories therefore move after a base swap even when `demand_projections.h5` is untouched. Compare against a `DYNAMIC_PRICE = False` run before attributing a solve difference to anything else.

## 5. Demand versus production at the 2010 base (reference)

Observed at RESFACTOR 10 on the 2010 map: `BASE_YR_production_t` is below the 2010 PRODUCTION demand series for 24 of 26 commodities. Largest gaps: pears -38%, other non-cereal crops -14%, plantation fruit -14%, apples -13%, vegetables -11%. Winter cereals and cotton are at parity. This is the starting point for any demand-vs-production sanity check.

## 6. Other 2010 literals and comments

Cosmetic, but they mislead readers: `data.py:128` comment, :492 comment ("YRS_CAL_BASE (2010) is not included"), :1267 comment "(91, 26)" (true of `D_CY`, not of the file), `quantity.py:88-92` comments, `dataprep.py:786` "BASE YEAR (2010)". `GHG_targets.xlsx` Targets sheet is anchored on "2010 emissions".

## 7. If the base year is ever moved

Two ways to fix sites 1-4: reslice the input files to start at the new base, or switch the lookups to calendar labels (the `CLIMATE_CHANGE_IMPACT_xr` pattern). The 2026-09-18 migration memo lays out both options per site with a recommendation for the label approach; the decision is Nick's. Whatever is chosen, the AM lower-bound guard at `economics/agricultural/transitions.py:649-654` (returns zeros when `base_year == YR_CAL_BASE`, ignoring stored dvars) must change in the same commit as any injected base-year AM dvars.
