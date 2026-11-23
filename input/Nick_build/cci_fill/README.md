# cci_fill

The eight climate-change-impact multiplier files with missing values filled on the 2021-base map.
Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/cci_fill.py`, from the repository root (step 12 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/cci_fill.py --input input/Nick_build/inherited_2010 --livestock-map input/Nick_build/livestock_map
        --x-mrj input/Nick_build/eligibility/x_mrj.npy --sa2 input/Nick_build/nlum_map/sa2_2021_cells.npy
        --out input/Nick_build/cci_fill

Peak 8.7 GiB on the full grid, 69 min, both with the NetCDF copies (22 October 2026 build); needs a machine with at least
16 GB of RAM.

## Sources

None public. The multipliers cannot be regenerated from a public source, so they are filled, not rebuilt.

Inputs from other datasets: `livestock_map/{lumap,lmmap}.h5`, `eligibility/x_mrj.npy`, `nlum_map/sa2_2021_cells.npy`.
Inherited 2010-build files (`inherited_2010/`): the eight `climate_change_impacts_*.h5` of the same names (the values
filled), `real_area.h5`, `state_id.npy`, `ag_landuses.csv`. The outputs are therefore not built from public sources alone.

Fill rule: a NaN on a target cell (cells allocated to, or eligible for, the (lm, lu) column) takes the real-area-weighted
mean of that column over the non-NaN cells of the same SA2_2021, else the same state, else national. Other values are
unchanged (bit-identical).

## Outputs

| File | Read by | Content |
|---|---|---|
| `climate_change_impacts_<rcp>_CO2_FERT_<ON or OFF>.nc` (eight files) | LUTO (`INPUT_SET='2021'`), one per run, by `YIELD_RCP` and `CO2_FERT` | `CLIMATE_CHANGE_IMPACT` over (column, cell), levels lm, lu, year |
| `climate_change_impacts_<rcp>_CO2_FERT_<ON or OFF>.h5` (eight files) | evidence | the same values, 2010-base file layout (141 float32 columns, same order) |
| `cci_fill_by_column.csv`, `cci_fill_summary.csv` | evidence | cells filled by column; summary |
| `provenance.csv` | check | sha256 of the script, every input, every output (content hashes for HDF5 and NetCDF) |

No columns are added: (dry, Pears) has none, and LUTO returns ones for a (lm, lu) without columns.
Method: `docs/BASE2021_INPUTS_METHODS.md`.
