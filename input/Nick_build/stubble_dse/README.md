# stubble_dse

The stubble carrying rate LUTO reads (`STUBBLE_DSE_FILE`): the raw rate of `livestock_map/` with the no-sheep-data rule
applied (0 on cells without sheep per-head data; no NaN). Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/stubble_dse.py`, from the repository root (step 15 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/stubble_dse.py --raw input/Nick_build/livestock_map/stubble_dse_ha.nc
        --input <folder> --out input/Nick_build/stubble_dse

`--input` must hold `agec_lvstk.nc`, `agGHG_lvstk.nc` and `water_delivery_price.nc`. For the 2020-21 build these sit in
two dataset folders, so `--input` is a local staging folder of links to `livestock_economics/agec_lvstk.nc`,
`livestock_economics/water_delivery_price.nc` and `ghg_factors/agGHG_lvstk.nc`.

## Sources

No downloads. Inputs from other datasets: `livestock_map/stubble_dse_ha.nc` (raw rate), `livestock_economics/agec_lvstk.nc`,
`livestock_economics/water_delivery_price.nc`, `ghg_factors/agGHG_lvstk.nc`.

## Outputs

| File | Read by | Content |
|---|---|---|
| `stubble_dse_ha.nc` | LUTO (`STUBBLE_DSE_FILE`) | stubble carrying rate, DSE per ha, annualised, every full-resolution land cell |
| `provenance.csv` | check | sha256 of the script, every input and the output; cells set to 0 |
