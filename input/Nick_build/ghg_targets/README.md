# ghg_targets

The split GHG target series LUTO constrains against when `GHG_EMISSIONS_LIMITS` is not 'off': an agriculture row and a
net LULUCF row, each with its modelled and exogenous parts. AR5 GWP100, t CO2e. Data not in git; rebuild with the script below.

## Built by

`input/Nick_build/scripts/ghg_targets.py`, from the repository root:

    python input/Nick_build/scripts/ghg_targets.py --out input/Nick_build/ghg_targets
        --scope-matrix input/Nick_build/ghg_targets/scope_matrix.csv [--cache DIR]

Packages: pandas, openpyxl. Output is deterministic (sorted rows, fixed float format, LF line endings, no timestamps).

## Sources

Downloaded at run time and sha256-checked against pins in the script (raw files and the canonical text of the values):

| Source | Release / version | Link |
|---|---|---|
| Australia's National Greenhouse Accounts API, NIR 2024 (sectors 3 and 4, all gases and by gas) | NIR 2024, as pinned (sha256 in the script) | https://greenhouseaccounts.climatechange.gov.au/api/ |
| DCCEEW, Australia's emissions projections 2025, chart-data workbook, Baseline scenario | 2025 edition, as pinned (sha256 in the script) | https://www.dcceew.gov.au/sites/default/files/documents/australias-emissions-projections-2025-chart-data.xlsx |

NIR for FY2010 to FY2024, DCCEEW Baseline for FY2025 to FY2040, held flat at FY2040 after it.

Hand-made input: `scope_matrix.csv` (which inventory nodes LUTO models). It sits in this ignored folder for now.

## Outputs

| File | Read by | Content |
|---|---|---|
| `series_agriculture.csv`, `series_agriculture_exogenous.csv` | LUTO (`load_ghg_targets`) | agriculture row: target and the part LUTO does not model |
| `series_lulucf_modelled.csv`, `series_lulucf_exogenous.csv` | LUTO | net LULUCF row: modelled and exogenous parts |
| `plantings_baseline.csv` | LUTO | plantings baseline rate and anchor year |
| `tolerances.csv` | LUTO | AG benchmark tolerance |
| `budget.csv` | LUTO (reported, not imposed) | cumulative LULUCF budget |
| `ag_not_modelled.csv`, `series_components_long.csv`, `sum_check.csv` | evidence | agriculture not modelled; every component by year; component sums against totals |
| `provenance.csv` | check | sha256 of the script, sources, `scope_matrix.csv` and every output |
