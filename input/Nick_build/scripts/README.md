# Nick_build/scripts

The code that builds LUTO's externally built inputs. Everything in this folder is version-controlled; everything
else in `input/Nick_build/` (the built data, download caches) is ignored by git (`input/.gitignore`).

```
input/Nick_build/
  scripts/            tracked: the build code, and any hand-made table or config it reads
  ghg_targets/        ignored: built by scripts/ghg_targets.py
  ag_mgt_caps/        ignored: built by scripts/ag_mgt_caps.py
  ag_mgt_caps_cache/  ignored: downloads
```

## Rules

1. **Data-processing code goes here, never in `luto/`.** One script per dataset, writing into its own folder
   `input/Nick_build/<dataset>/`. A hand-made input the script reads (a lookup table, a config) also goes in
   `scripts/`, not in the ignored data folder.
2. **No `luto` imports.** A build script runs on its own from the repository root; `luto/` reads only the finished
   files, through the paths in `luto/settings.py` / `luto/data.py`.
3. **Every output folder holds a `provenance.csv`** written by the script: the script's sha256, each source's URL or
   path and sha256, and the sha256 of every file written. The data is not in git, so this file is the only record of
   which code and sources built it.
4. **Never commit built data or downloads.** Large or rebuilt files stay in the ignored folders.
5. **Packages:** list any package a script needs that the model does not (below), rather than adding it to the
   model's `requirements.yml` unannounced.
6. **Handing over data:** build locally; the data is copied to Gadi by hand (`rsync` without `--inplace`), and the
   two copies' `provenance.csv` should match.

## Scripts

| Script | Builds | Run (from the repository root) | Third-party packages |
|---|---|---|---|
| `ghg_targets.py` | `ghg_targets/`: the split GHG target series (NIR 2024, DCCEEW 2025) | `python input/Nick_build/scripts/ghg_targets.py --out input/Nick_build/ghg_targets --scope-matrix input/Nick_build/ghg_targets/scope_matrix.csv [--cache DIR]` | pandas, openpyxl |
| `ag_mgt_caps.py` | `ag_mgt_caps/`: observed ag-management adoption caps, existing HIR project cells (CER register, CEA files, ABS 4627.0) | `python input/Nick_build/scripts/ag_mgt_caps.py --out input/Nick_build/ag_mgt_caps --input input --cache input/Nick_build/ag_mgt_caps_cache` | numpy, pandas, xarray, geopandas, rasterio, shapely, xlrd (pandas reads the ABS `.xls` with it) |
| `stubble_dse.py` | `stubble_dse/`: the stubble carrying rate LUTO reads (`STUBBLE_DSE_FILE`): Nick's raw rate, 0 on cells without sheep per-head data | `python input/Nick_build/scripts/stubble_dse.py --raw <Nick's raw file> --input input --out input/Nick_build/stubble_dse` | numpy, xarray |
| `crop_economics.py` | `crop_economics/`: 2021-base crop economics and crop energy and lifecycle GHG fields | `python input/Nick_build/scripts/crop_economics.py --input input/Nick_build/inherited_2010 --nlum-map input/Nick_build/nlum_map --map-dir input/Nick_build/livestock_map --x-mrj input/Nick_build/eligibility/x_mrj.npy --ghg-crops input/Nick_build/ghg_factors/agGHG_crops.h5 --out input/Nick_build/crop_economics --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]` | numpy, pandas, PyTables, openpyxl, xlrd, rasterio, geopandas, scipy, xarray, netCDF4 |
| `livestock_economics.py` | `livestock_economics/`: 2021-base livestock economics, interim water delivery price, flat-wool productivity table | `python input/Nick_build/scripts/livestock_economics.py --input input/Nick_build/inherited_2010 --livestock-map input/Nick_build/livestock_map --pasture input/Nick_build/pasture_productivity/pasture_kg_dm_ha.h5 --sa2 input/Nick_build/nlum_map/sa2_2021_cells.npy --out input/Nick_build/livestock_economics --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]` | numpy, pandas, PyTables, openpyxl, xlrd, rasterio, scipy, xarray, netCDF4 |

`stubble_dse/` is not built: Nick's raw stubble rate is not here yet. `ag_mgt_caps/` is not built at present: the CER register no longer matches the script's sha256 pin.
