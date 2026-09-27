# LUTO2 base-year-only runs, write functions and environment gotchas

Scope: what happens when `SIM_YEARS` contains only the base year, how to get the base-year aggregates out anyway, how to read the output csvs, and the environment traps hit on a Windows workstation in 2026-09. Line numbers: branch jinzhu at 72d30427. Companions: [CLAUDE_DATA.md](CLAUDE_DATA.md), [CLAUDE_TIME_AXIS.md](CLAUDE_TIME_AXIS.md), [CLAUDE_OUTPUT.md](CLAUDE_OUTPUT.md).

## 1. What a base-year-only run does (observed)

Settings: `SIM_YEARS = [2010]`, RESFACTOR 10 and 5, everything else default.

- `sim.load_data()` succeeds: 38 min at RESFACTOR 10 and 27 min at RESFACTOR 5 when `input/` is read over a network share (I/O bound, not cell bound); peak RSS 6.1 GB and 7.0 GB.
- `sim.run(data=data)` logs "Running LUTO 2.3 between 2010 - 2010 at RES-10, total 0 runs!", never touches Gurobi, saves `Data_RES<rf>.lz4` into the run folder, then `write_outputs` fails:

```
File "luto/tools/write.py", line 455, in write_data
    years = [yr for yr in settings.SIM_YEARS if yr <= data.last_year]
TypeError: '<=' not supported between instances of 'int' and 'NoneType'
```

`data.last_year` is `None` until a solve step sets it (`simulation.py:229`). So a base-year-only run does not write `out_<year>/`. Do not expect csvs from it until this is fixed (a one-line guard in `write_data`, or set `last_year = YR_CAL_BASE` when there are zero steps).

## 2. Getting base-year aggregates anyway (no source change)

Reload the saved object and call the writers directly for `yr_cal = YR_CAL_BASE`:

```python
import luto.simulation as sim
from luto.tools import write as W
data = sim.load_data_from_disk(r"<run>/Data_RES10.lz4")   # checks RESMULT against settings.RESFACTOR
out = r"C:\scratch\e1_extract"
W.write_dvar_and_mosaic_map(data, 2010, out)   # must run first: writes xr_dvar_ag_2010.nc that the next two read
W.write_dvar_area(data, 2010, out)
W.write_quantity(data, 2010, out)
W.write_economics(data, 2010, out)
W.write_ghg(data, 2010, out)
W.write_water(data, 2010, out)
```

Order matters: `write_dvar_area` (`write.py:765`) and `write_quantity` (:935) load `xr_dvar_ag_<yr>.nc`, which only `write_dvar_and_mosaic_map` (:661) produces. Base-year attributes on the object: `BASE_YR_production_t` (:1195, per `data.COMMODITIES`), `lumaps[2010]`, `lmmaps[2010]`, `ag_dvars[2010]`, `NCELLS`, `MASK`, `REAL_AREA`, `D_CY[0]`. There is no `data.lumap`; use the year-keyed dicts.

## 3. Reading the regional csvs

Every `*_<yr>.csv` from `write_economics`, `write_dvar_area`, `write_quantity`, `write_ghg` has columns like `Land-use, Water_supply, Type, Year, region, Value ($), region_level`. Facts that are easy to get wrong:

- `region_level` takes the values `region_state` and `region_NRM` only. The national row is `region == 'AUSTRALIA'` inside `region_level == 'region_state'`. The eight state rows sum to it.
- `Land-use`, `Water_supply`, `Type` and `Source` each include an `ALL` row. Summing a column without filtering double- or triple-counts. Filter to the single `ALL` combination you want (or exclude `ALL` and sum the parts; both agree to rounding).
- `GHG_emissions_<yr>.csv` is the two-line headline (limit and total). `water_yield_limits_and_public_land_<yr>.csv` is one row per drainage division (13).
- At the base year the am and non-ag economics/area files have zero rows.

## 4. Reference base-year aggregates, 2010 map (observed 2026-09-18)

National, from the writers above. Use these as the "before" column when the base map or base year changes. Full tables, per land use and per commodity, are in the migration memo (`C:\scratch\luto_mig\memo.md` on optimus-nc).

| Aggregate | RESFACTOR 10 | RESFACTOR 5 |
|---|---|---|
| NCELLS | 49,027 | 186,648 |
| Production total, `BASE_YR_production_t` (t or KL) | 125,125,067 | 127,933,490 |
| Ag revenue ($) | 58,237,444,000 | 59,849,056,000 |
| Ag cost ($) | 29,057,528,000 | 29,970,094,000 |
| Ag profit ($) | 29,179,929,000 | 29,878,964,000 |
| Area by land use total, `write_dvar_area` (ha) | 464,056,327 | 464,591,103 |
| of which irrigated (ha) | 2,083,664 | 2,092,021 |
| GHG_EMISSIONS_TCO2e | 64,538,696 | 65,838,388 |
| Water yield inside LUTO (ML) | 234,380,498 | 233,009,948 |
| Water net yield (ML) | 395,891,049 | 394,520,500 |

Area is RESFACTOR-stable to 0.1-0.4%; production, revenue, cost and GHG move 2-3% between RESFACTOR 10 and 5; small horticulture commodities move most (pears +27%). Compare runs at matched RESFACTOR. These numbers came from Python 3.13 / numpy 2.5 / pandas 2.3, not the pinned env in `requirements.yml`; re-take them there before using as fixtures.

## 5. Environment traps (Windows workstation)

- Console encoding: `data.py:133` and neighbours print box-drawing characters. Under cp1252 this raises `UnicodeEncodeError` inside the `LogToFile` wrapper. Set `PYTHONIOENCODING=utf-8` (and `PYTHONUTF8=1`).
- `settings.HCAS_CONTRIBUTION_PERCENTILE = 'USER_DEFINED'` (default) fails with `KeyError: 'USER_DEFINED'` at `data.py:1417` against the 2026-09 input bundle, whose `bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv` has `CSV_DEFINED` instead. `'50'` is the nearest available column (identical except the three natural-land livestock rows). Affects biodiversity scoring only.
- xarray on `.nc` inputs: with numpy >= 2.4 the pip `netCDF4` wheel may raise "numpy.dtype size changed". Use `engine='h5netcdf'` for inspection, or the pinned env.
- `sim.load_data_from_disk` refuses an object whose `RESMULT` does not match `settings.RESFACTOR` (`simulation.py:370`). Point `PYTHONPATH` at a checkout whose settings match, or `joblib.load` directly.
- Running from a copied checkout needs the repo root on `PYTHONPATH` (`import luto` is not installed as a package).
- Gurobi: base-year-only runs never call the solver, so a pip `gurobipy` trial licence is enough for them. Anything with a solve step needs a real licence.
- Never point `OUTPUT_DIR` inside a shared checkout; run folders are timestamped under `OUTPUT_DIR` and include the lz4 Data object (0.9-1.1 GB at RESFACTOR 10/5).
- No pytest suite exists in the tree despite references in `CLAUDE_SETUP.md`; `luto/tests/` holds diagnostic tools.

## 6. Two scratch scripts that worked

`run_e1.py`: `import luto.simulation as sim; data = sim.load_data(); sim.run(data=data)` with timing prints. `extract_e1.py`: section 2 above plus prints of `BASE_YR_production_t`, area grouped by `lumaps[2010]`, and `D_CY[0]`. Both live in `C:\scratch\luto_mig\` on optimus-nc as of 2026-09-18.
