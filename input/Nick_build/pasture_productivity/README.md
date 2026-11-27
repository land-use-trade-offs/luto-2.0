# pasture_productivity

LUTO's pasture productivity layer (kg DM/ha) from AussieGRASS total pasture growth, FY2016 to FY2025.
Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/pasture_productivity.py`, from the repository root (step 3 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/pasture_productivity.py --input input/Nick_build/inherited_2010
        --out input/Nick_build/pasture_productivity --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Peak about 1.2 GB on the full grid (October 2026); needs a machine with at least 8 GB of RAM.

## Sources

Downloaded at run time and checked against pins in the script. The LongPaddock grids are pinned on the decoded grid
(dtype, shape, transform, CRS, nodata, pixel bytes), not the file bytes; the raw-file sha256 is recorded, not enforced.

| Source | Release / version | Link |
|---|---|---|
| Queensland Government LongPaddock AussieGRASS national grids, `12months.growth.tot` (kg DM/ha, July to June), ERDAS Imagine in zip (CC BY 4.0) | issues 2016-06 to 2025-06 (ten grids) | https://data.longpaddock.qld.gov.au/AussieGRASS2/AUS/ (`<yyyy>06/<yyyy>06.12months.growth.tot.aus.img.zip`) |

Inputs from other datasets: none.
Inherited 2010-build files (`inherited_2010/`): `NLUM_2010-11_mask.tif` (the cell grid). The output is therefore not
built from public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `pasture_kg_dm_ha.nc` | LUTO (`INPUT_SET='2021'`) | `PASTURE_KG_DM_HA` over `cell` |
| `pasture_kg_dm_ha.h5` | `livestock_map.py` (`--pasture`), `livestock_economics.py`, `compare_inputs.py` | the same values, 2010-base file layout (pandas Series, int16, key `pasture_kg_dm_ha`) |
| `provenance.csv` | check | sha256 of the script, every source and input, every output (content hashes for HDF5); lists each nearest-filled cell with a tie, the tied pixels and the one chosen |

Method: `docs/BASE2021_INPUTS_METHODS.md`.
