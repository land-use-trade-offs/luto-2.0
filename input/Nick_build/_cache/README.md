# _cache

Shared download cache of the 2021-base scripts (all but `cci_fill.py`): every raw file and API response they read, stored
by content as `<first two hex digits>/<sha256>`, with `index.csv` (sha256, url, bytes, name). Scripts read from here and,
with `--offline`, fail on a miss instead of downloading. Not in git (about 1.7 GB).

One cache, not one per dataset: several scripts read the same files (SA2 boundaries, NLUM rasters, ABS tables).

Files with no scripted download, kept only here: the NSW 71Q trade table read by `water_licence_price.py`.

`ag_mgt_caps.py` keeps its own name-keyed cache in `ag_mgt_caps_cache/`.
