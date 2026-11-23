# livestock_economics

2021-base livestock economics (prices, costs, outputs and water per head, FY2021 dollars), the interim water delivery
price, and the 2010 productivity table rewritten with a flat wool per-head trend. Data not in git; rebuild with the
script below.

## Built by

`input/Nick_build/scripts/livestock_economics.py`, from the repository root (step 11 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/livestock_economics.py --input input/Nick_build/inherited_2010
        --livestock-map input/Nick_build/livestock_map --pasture input/Nick_build/pasture_productivity/pasture_kg_dm_ha.h5
        --sa2 input/Nick_build/nlum_map/sa2_2021_cells.npy --out input/Nick_build/livestock_economics
        --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Peak about 6.2 GB on the full grid (October 2026); needs a machine with at least 16 GB of RAM.

## Sources

Downloaded at run time and sha256-checked against pins in the script. Release as pinned unless stated.

| Source | Release / version | Link |
|---|---|---|
| ABS 6401.0 Consumer Price Index (A2325846C, all groups) | August 2026 | https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/aug-2026/CPI-Quarterly.zip |
| ABS 7215.0 Livestock products: beef and veal, lamb and mutton carcase tonnes (Australia) | June 2026 | https://www.abs.gov.au/statistics/industry/agriculture/livestock-products/jun-2026/ (`7215009`, `7215010`, `7215011`, `7215012` `.xlsx`) |
| ABARES Agricultural commodities statistical tables 8 and 13 (live cattle and sheep exports) | September 2026 | https://daff.ent.sirsidynix.net.au/client/en_AU/search/asset/1038521/3 |
| DAFF live animal export statistics (exports by state of export port), pinned on the rows and columns used, not the file bytes | 2016-2021 file (data as at 31 Jul 2021); 2021-2026 file (data as at 31 Aug 2026, updated monthly) | https://www.agriculture.gov.au/sites/default/files/documents/ (`all-livestock-exports-2016-2021.xlsx`, `all-livestock-exports-2021-2026.xlsx`) |
| ABS 7503.0 Value of Agricultural Commodities Produced; ABS Australian Agriculture: livestock | FY2017-22; FY2023-25 | https://www.abs.gov.au/statistics/industry/agriculture/value-agricultural-commodities-produced-australia/ ; https://www.abs.gov.au/statistics/industry/agriculture/australian-agriculture-livestock/ |
| ABARES Farm Data Portal, state and national historical files | as pinned (sha256 in the script) | https://www.agriculture.gov.au/sites/default/files/documents/ (`fdp-state-historical.csv`, `fdp-national-historical.csv`) |
| ABS 4618.0 Water Use on Australian Farms | FY2017-21 | https://www.abs.gov.au/statistics/industry/agriculture/water-use-australian-farms/ |
| AWI AWPFC wool production forecast reports (fetched and pinned only; values used are hand-checked reference-table entries; all rights reserved, not for redeposit) | August 2020, August 2021 (amended), September 2022, August 2023, August 2024, August 2025 | https://www.wool.com/globalassets/wool/market-intelligence/wool-production-forecasts/ |
| Dairy Australia, The Australian Dairy Industry In Focus (fetched and pinned only; all rights reserved, not for redeposit) | 2025 | https://assets.dairyaustralia.com.au/api/public/content/In-Focus-2025 |
| FutureBeef, "Water requirements for cattle" (drinking-water constants in the reference table; not fetched, terms unclear) | page modified 2026-06-17 | https://futurebeef.com.au/resources/water-requirements/ |

Inputs from other datasets: `livestock_map/` (`lumap.h5`, `lmmap.h5`, `lvstk_k.h5`, `livestock_recon_state_type.csv`,
`livestock_recon_sa2.csv`, `livestock_dse_ratings.csv`), `pasture_productivity/pasture_kg_dm_ha.h5`,
`nlum_map/sa2_2021_cells.npy`.
Inherited 2010-build files (`inherited_2010/`): `state_id.npy`, `real_area.h5`, `safe_pur_natl.h5`, `safe_pur_modl.h5`,
`NLUM_2010-11_mask.tif`, `agec_lvstk.h5` (column list and layout only), `water_delivery_price.h5` (the interim price
values), `yieldincreases_bau2022.csv` (the productivity input, file sha256 pinned). The outputs are therefore not built
from public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `agec_lvstk.nc` | LUTO (`INPUT_SET='2021'`); `stubble_dse.py` (`--input`) | `AGEC_LVSTK` over (column, cell), levels field and animal |
| `water_delivery_price.nc` | LUTO (`INPUT_SET='2021'`); `stubble_dse.py` (`--input`) | `WP` over cell; interim |
| `agec_lvstk.h5`, `water_delivery_price.h5` | `compare_inputs.py` | the same values, 2010-base file layout |
| `yieldincreases_bau2022.csv` | LUTO (`INPUT_SET='2021'`) | the 2010 file with every wool column (dry and irrigated, modified and natural sheep land) set to 1 in every year; other bytes unchanged |
| `livestock_prices.csv`, `livestock_outputs_per_head.csv`, `livestock_costs_per_head.csv` | evidence | national prices; output per head; cost per head |
| `livestock_water_per_head.csv` | `compare_inputs.py` | WR_DRN and WR_IRR per head |
| `receipts_share_report.csv` | evidence | share of livestock receipts the modelled products cover, by state (unverified-cause flag for NT) |
| `interim_items_livestock_economics.csv` | evidence | INTERIM and inferred items |
| `provenance.csv` | check | sha256 of the script, every source and input, every output (content hashes for HDF5 and NetCDF) |

Interim items: `water_delivery_price` is the 2010-build values with the state median on blank cells (national median
where a state has none); the water licence price is built by `water_licence_price.py`. The flat wool trend is a change to
default behaviour (the 2010 file fell 0.77 per cent a year dryland and 1.53 irrigated from 2010, source not recorded).
Method: `docs/BASE2021_INPUTS_METHODS.md`.
