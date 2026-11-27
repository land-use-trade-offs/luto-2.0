# water_licence_price

Water licence (entitlement) price layer, $/ML by river region and state, FY2020-21 in FY2021 dollars.
Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/water_licence_price.py`, from the repository root (step 14 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/water_licence_price.py --input input/Nick_build/inherited_2010
        --map-dir input/Nick_build/livestock_map --out input/Nick_build/water_licence_price
        --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Memory and run time: not stated in the script.

## Sources

Fetched at run time and sha256-checked against pins in the script; the values used are hand-checked entries of the
script's reference table, each with its page. Release as pinned.

| Source | Release / version | Link |
|---|---|---|
| Ricardo (formerly Aither, now WSP) Water Markets Report: Table 4 (p.27, southern Murray-Darling Basin entitlement VWAP by trading zone and class) and northern Basin general security VWAPs (p.33); no licence stated | 2020-21 review and 2021-22 outlook | https://www.ricardo.com/media/z5aplhxw/2021-ricardo-water-markets-report.pdf |
| Queensland Permanent Water Trading Annual Report, surface water, supplemented supply (weighted average price by scheme and priority group); Internet Archive copy, the department host refuses scripted clients | 1 Jul 2020 to 30 Jun 2021 | https://web.archive.org/web/2023id_/https://www.rdmw.qld.gov.au/__data/assets/pdf_file/0006/1590045/pwtr-supplemented-annual-2020-21.pdf |
| Tasmanian Irrigation: Scottsdale 2020-21 charge sheet; Sassafras Wesley Vale Augmentation offer; Greater Meander offer (primary-sale list prices, no licence stated) | 2020-21; April 2022; May 2022 | https://www.tasmanianirrigation.com.au/source-assets/ (`map-data/scottsdale/downloads/Annual-Charges-2020-21_SIS.pdf`, `map-data/SWISA-Water-Sales-One-Page-Summary.pdf`, `map-data/3.-GMIS-Water-Sales-Scheme-Offer.pdf`) |
| NSW Water Trade dashboard, crosstab "71Q Data (Table)" (CC BY 3.0 AU) | registration dates June 2020 to October 2026; downloaded by hand 2026-10-19 | https://tableau.dpie.nsw.gov.au/t/DPIEExternal/views/WaterTradeDashboard/WaterTradeSnapshot |
| ABS 6401.0 Consumer Price Index (A2325846C, all groups; CPI for Tasmanian prices dated outside FY2021) | August 2026 | https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/aug-2026/CPI-Quarterly.zip |

The NSW 71Q trade table has no scripted download (Tableau view). It is read from the shared cache (`_cache/`); on a first
run it is supplied with `--source nsw_71q=PATH`.

Inputs from other datasets: `livestock_map/lumap.h5` and `lmmap.h5` (`--map-dir`; comparison weights only).
Inherited 2010-build files (`inherited_2010/`): `rivreg_id.h5`, `rivreg_lut.h5`, `draindiv_id.h5`, `draindiv_lut.h5`,
`REGION_STATE_r.h5`, `real_area.h5`, `water_licence_price.h5` (the interim layer: its key, index, dtype and series name are
the output layout; its values are the comparison only).

## Outputs

| File | Read by | Content |
|---|---|---|
| `water_licence_price.nc` | LUTO (`INPUT_SET='2021'`) | one variable, named as the series (`WATER_PRICE_ML_ABARES`, kept so the file is a drop-in; the values are not ABARES') |
| `water_licence_price.h5` | evidence | the same values, 2010-base file layout (key `/water_licence_price`, int16) |
| `water_licence_price_zones.csv` | evidence | every (river region, state) pair: cells, area, irrigated area on the 2021-base map, rule, zone, price |
| `water_licence_price_compare.csv` | evidence | the layer against the interim by drainage division and state |
| `water_licence_price_nsw71q.csv` | evidence | FY2020-21 VWAP by NSW water source and licence category, with the zone each feeds or is compared with |
| `provenance.csv` | check | sha256 of the script, every source and input, every output (content hashes for HDF5 and NetCDF) |

Pairs with no public price take the northern proxy (2,869 $/ML) in northern Basin river regions, else the median of public
prices outside the Basin (checked against 1,442 $/ML); both labelled INFERRED. Method: `docs/BASE2021_INPUTS_METHODS.md`.
