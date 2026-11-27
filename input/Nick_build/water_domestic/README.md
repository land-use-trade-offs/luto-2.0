# water_domestic

2021-base domestic and industrial water use by water region (ML), the 2010-11 spatial pattern scaled to the ABS Water
Account 2020-21. Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/water_domestic.py`, from the repository root (step 13 of the order in `scripts/README.md`):

    python input/Nick_build/scripts/water_domestic.py --input input/Nick_build/inherited_2010 --out input/Nick_build/water_domestic
        --cache input/Nick_build/_cache [--offline] [--live-check] [--source NAME=PATH]

Peak memory about 1 GB.

## Sources

Downloaded at run time and sha256-checked against pins in the script.

| Source | Release / version | Link |
|---|---|---|
| ABS Water Account, Australia (cat. 4610.0), 46100DO014: physical water supply and use by industry, water type, Australia and state/territory, 2014-15 to 2020-21 (ML) | 2020-21, released 2022-10-20 | https://www.abs.gov.au/statistics/environment/environmental-accounts/water-account-australia/2020-21/46100DO014_202021.csv |
| ABS Water Account, Australia, 46100DO001 Table 1.4 (water consumption, Australia; the published national check) | 2020-21 | https://www.abs.gov.au/statistics/environment/environmental-accounts/water-account-australia/2020-21/46100DO001_202021.xlsx |

Inputs from other datasets: none.
Inherited 2010-build files (`inherited_2010/`): `Water_Use_Domestic.csv` (the 2010-11 file this replaces: ABS Water Account
2010-11, households plus ANZSIC divisions B to S, by drainage division and river region; sha256 pinned), `draindiv_id.h5`,
`rivreg_id.h5`, `draindiv_lut.h5`, `rivreg_lut.h5`, `REGION_STATE_r.h5`, `real_area.h5`. The output is therefore not built
from public sources alone.

## Outputs

| File | Read by | Content |
|---|---|---|
| `Water_Use_Domestic.csv` | LUTO (`INPUT_SET='2021'`) | layout, row order and line endings of the 2010-11 file |
| `water_domestic_state_derivation.csv` | evidence | every term of Q (2020-21 use by state) with its basis |
| `water_domestic_scale_by_state.csv` | evidence | 2010-11 total, Q and scale factor by state |
| `water_domestic_by_region.csv` | evidence | 2010-11 and 2021 values by region |
| `provenance.csv` | check | sha256 of the script, every source and input, every output |

Suppressed (np) ABS cells are the residual of the published state total, or an estimate split by each state's published
agriculture, forestry and fishing self-extracted use (labelled estimated in the derivation file). The spatial pattern is
the 2010-11 file's, which is 0.9838 of the ABS 2010-11 households plus B to S total (5,889,627 against 5,986,324 ML); the
96.7 GL gap is unattributed. Method: `docs/BASE2021_INPUTS_METHODS.md`.
