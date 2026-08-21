# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.


import os
import re
import glob
import json
import base64
import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray as cfxr
import rioxarray  # .rio accessor on xarray objects
from rasterio.io import MemoryFile

from joblib import delayed, Parallel
from luto import settings
from luto.tools.report.data_tools import (
    get_all_files,
    build_map_legend,
    rename_reorder_hierarchy,
)

from luto.tools.report.data_tools.parameters import (
    COLOR_AG,
    COLOR_AM,
    COLOR_NON_AG,
    COLOR_LUMAP,
)



def safe_key(s: str) -> str:
    """Convert arbitrary string to a safe filename/JS-variable suffix."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', str(s)).strip('_')


def build_tree(combos: list) -> list | dict:
    """Recursively build a nested tree from combo tuples; the last level becomes a list."""
    if not combos:
        return []
    if len(combos[0]) == 1:
        seen = dict.fromkeys(c[0] for c in combos)
        vals = list(seen)
        return ['ALL', *sorted(v for v in vals if v != 'ALL')] if 'ALL' in seen else sorted(vals)
    groups: dict = {}
    for combo in combos:
        groups.setdefault(combo[0], []).append(combo[1:])
    sorted_keys = (['ALL'] if 'ALL' in groups else []) + sorted(k for k in groups if k != 'ALL')
    return {k: build_tree(groups[k]) for k in sorted_keys}



def write_geotiff_index(flat_data: dict, dim_names: list, prefix: str) -> None:
    """Write ``<prefix>__index.js`` with the dimension tree and per-combo display attrs.

    flat_data : { (dim1, …, dimN, year): data_dict }
    data_dict : {'tif_b64': str, 'intOrFloat': ..., 'min_max'|'legendKey': ...}
    tif_b64 is excluded from the index — it lives in the per-combo JS files.
    """
    base_name = os.path.basename(prefix)
    os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)

    combo_map: dict = {}
    for tp, data in flat_data.items():
        combo = tp[:-1]
        combo_map[combo] = {k: v for k, v in data.items() if k != 'tif_b64'}

    tree = build_tree(list(combo_map.keys()))
    combo_attrs = {
        '__'.join(safe_key(v) for v in combo): attrs
        for combo, attrs in combo_map.items()
    }

    index = {'dims': dim_names, 'tree': tree, 'combos': combo_attrs}
    index_name = f'{base_name}__index'
    with open(f'{prefix}__index.js', 'w') as f:
        f.write(f'window["{index_name}"] = ')
        json.dump(index, f, separators=(',', ':'), indent=2)
        f.write(';\n')


def write_split_by_combo_geotiff(flat_data: dict, dim_names: list, prefix: str) -> None:
    """Write one JS file per unique combo (all years) and an index JS file.

    Each per-combo JS file contains the base64-encoded GeoTIFF and display attrs
    for every year, following the same split-file pattern the Vue already uses:
      window["map_area_Ag__Dryland__Wheat"] = {
          2010: {tif_b64: "...", intOrFloat: "float", min_max: [0, 500]},
          2025: {tif_b64: "...", ...},
      };
    """
    base_name = os.path.basename(prefix)
    tif_dir = os.path.dirname(prefix)
    os.makedirs(tif_dir, exist_ok=True)

    combos: dict = {}
    for tp, data in flat_data.items():
        combo, year = tp[:-1], tp[-1]
        combos.setdefault(combo, {})[year] = data

    for combo, year_data in combos.items():
        var_suffix = '__'.join(safe_key(v) for v in combo)
        js_filename = f'{base_name}__{var_suffix}.js'
        var_name = js_filename.removesuffix('.js')
        with open(os.path.join(tif_dir, js_filename), 'w') as f:
            f.write(f'window["{var_name}"] = ')
            json.dump(year_data, f, separators=(',', ':'))
            f.write(';\n')

    write_geotiff_index(flat_data, dim_names, prefix)



def map2geotiff(
    index_merc: np.ndarray,    # (H, W) int32; ≥0 = 1D cell index, -1 = nodata
    geo_meta_merc: dict,       # rasterio write kwargs for the EPSG:3857 grid
    arr_sel: xr.DataArray,     # 1D DataArray[cell] — raw data values
    isInt: bool,
    hierarchy_tp: tuple,
    layer_magnitude,           # (global_min, global_max) for float layers; None for int
    legend_key: str | None,
    real_area_1d: np.ndarray,               # 1D float32[cell] — actual ha per cell
) -> tuple:
    """Convert a 1D cell array to GeoTIFF bytes in EPSG:3857.

    For float layers, each cell value is divided by the cell's real area (ha) so
    the GeoTIFF stores actual per-ha values.  ``index_merc`` is memmapped by
    joblib across workers; ``real_area_1d`` is small enough to be pickled.
    """
    H, W = int(index_merc.shape[-2]), int(index_merc.shape[-1])
    nodata_val = geo_meta_merc['nodata']
    dtype = 'int16' if isInt else 'float32'

    arr_2d = np.full((H, W), nodata_val, dtype=dtype)
    valid = index_merc >= 0
    src_vals = arr_sel.values[index_merc[valid]]

    if isInt:
        # Source is float32; NaN/inf can't cast to int16 — map them to nodata first.
        src_f = np.asarray(src_vals, dtype=np.float32)
        src_f[~np.isfinite(src_f)] = nodata_val
        arr_2d[valid] = src_f.astype(np.int16)
    else:
        cell_area = real_area_1d[index_merc[valid]]
        src_vals = src_vals / np.where(cell_area > 0, cell_area, np.nan)
        arr_2d[valid] = src_vals

    meta = {**geo_meta_merc, 'dtype': dtype}
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dst:
            dst.write(arr_2d[np.newaxis])   # rasterio expects (bands, H, W)
        tif_bytes = memfile.read()

    if isInt:
        attrs: dict = {'intOrFloat': 'int', 'legendKey': legend_key}
    else:
        global_min, global_max = layer_magnitude
        # min_max: colorbar labels — use RF²×121 (≈ max cell area) so bounded
        # layers display ≤1 and the legend matches prior behaviour.
        min_max = (
            round(global_min / settings.RESFACTOR ** 2 / 121, 2),
            round(global_max / settings.RESFACTOR ** 2 / 121, 2),
        )
        # raw_min_max: pixel normalisation — use mean real area to match the
        # per-ha values actually stored in the GeoTIFF (raw / cell_real_area).
        mean_area = float(np.nanmean(real_area_1d))
        per_ha_min = round(global_min / mean_area, 4)
        per_ha_max = round(global_max / mean_area, 4)
        attrs = {'intOrFloat': 'float',
                 'min_max':     list(min_max),
                 'raw_min_max': [per_ha_min, per_ha_max]}

    return hierarchy_tp, tif_bytes, attrs



def build_year_meta(year: int, base_dir: str) -> tuple:
    """Build EPSG:3857 raster metadata for one simulation year.

    Returns (index_merc, geo_meta_merc, real_area_1d):
      index_merc    — (H, W) int32; cell index ≥0, -1 = nodata
      geo_meta_merc — rasterio write kwargs for the reprojected grid
      real_area_1d  — 1D float32[cell] ha per cell, or None if file missing
    """
    ds_template = os.path.join(base_dir, f'xr_map_template_{year}.nc')
    with xr.load_dataset(ds_template) as ds:
        crs   = ds['spatial_ref'].attrs['crs_wkt']
        valid = ds['layer'].values >= 0
        arr   = np.full(ds['layer'].shape, -1.0, dtype=np.float32)
        arr[valid] = np.arange(valid.sum(), dtype=np.float32)
        rxr_merc = (
            ds['layer'].copy(data=arr)
            .rio.write_nodata(-1.0)
            .rio.write_crs(crs)
            .rio.reproject('EPSG:3857', nodata=-1.0)
        )

    index_merc = np.round(rxr_merc.values).astype(np.int32)
    geo_meta_merc = {
        'driver': 'GTiff', 'count': 1,
        'crs':       rxr_merc.rio.crs,
        'transform': rxr_merc.rio.transform(),
        'width':     int(rxr_merc.shape[-1]),
        'height':    int(rxr_merc.shape[-2]),
        'nodata':    -9999.0, 'compress': 'deflate', 'predictor': 2,
    }

    real_area_path = os.path.join(base_dir, f'xr_area_real_area_ha_{year}.nc')
    real_area_1d = None
    if os.path.exists(real_area_path):
        with xr.open_dataset(real_area_path) as ra_ds:
            real_area_1d = ra_ds['data'].values.squeeze().astype(np.float32)

    return index_merc, geo_meta_merc, real_area_1d



def get_map2json(
    files_df: pd.DataFrame,
    legend_int_level: dict | str,
    float_magnitude: tuple | dict,
    save_path: str,
    legend_key_int: str | None = None,
) -> None:
    """Convert all layers in *files_df* to GeoTIFF files (EPSG:3857).

    For each unique (combo × year) one JS file is written per combo (all years)
    plus a companion ``<prefix>__index.js`` with the dimension tree and per-combo
    display attrs (intOrFloat, min_max / legendKey) for the Vue UI.
    """
    # Per-worker memory: float32 2D output grid (EPSG:3857 is ~7× the 1D cell count).
    ncells = 5_000_000 // (settings.RESFACTOR ** 2)
    mem_per_worker = ncells * 7 * 4 / 1e6
    cpu_cap = 61 if os.name == 'nt' else (os.cpu_count() or 1)
    workers = min(cpu_cap, max(1, int(settings.WRITE_REPORT_MAX_MEM_MB // mem_per_worker)))

    def get_legend_params(sel):
        if legend_int_level is None:
            isInt = False
        elif isinstance(legend_int_level, dict):
            isInt = legend_int_level.items() <= sel.items()
        elif isinstance(legend_int_level, str):
            isInt = legend_int_level in sel.keys()
        else:
            raise ValueError('legend_int_level must be None, a dict, or a str')
        if isInt:
            return True, None, legend_key_int
        magnitude = float_magnitude[sel['Commodity']] if isinstance(float_magnitude, dict) else float_magnitude
        return False, magnitude, None

    prefix = save_path.removesuffix('.js')

    flat_data: dict = {}
    tasks: list = []
    dim_names: list = []

    # Accumulate all (year × layer) tasks across every file before launching workers.
    # This keeps all workers busy across years instead of going idle between year files.
    for _, row in files_df.iterrows():
        xr_arr = cfxr.decode_compress_to_multi_index(xr.open_dataset(row['path'], chunks={}), 'layer')['data']
        valid_layers = xr_arr['layer'].to_index().to_frame().to_dict(orient='records')

        if len(valid_layers) == 0:
            print(f'│   ├── No valid layers found in {row["base_name"]}_{row["Year"]}, skipping.')
            continue

        dim_names = list(rename_reorder_hierarchy(valid_layers[0]).keys())
        index_merc, geo_meta_merc, real_area_1d = build_year_meta(int(row['Year']), os.path.dirname(row['path']))

        # GBF NRM-mode layers carry an is_selected coord marking which cells belong to the
        # selected NRM region; prebuilt once per file rather than reconstructed per layer.
        is_selected_arr = (
            np.asarray(xr_arr['is_selected'].values, dtype=bool)
            if 'is_selected' in xr_arr.coords else None
        )
        is_selected_coords = (
            {'is_selected': ('cell', is_selected_arr)} if is_selected_arr is not None else None
        )

        for sel in valid_layers:
            isInt, layer_magnitude, legend_key = get_legend_params(sel)
            renamed = rename_reorder_hierarchy(sel)
            hierarchy_tp = tuple(list(renamed.values()) + [int(row['Year'])])
            vals = xr_arr.sel(**sel).values.copy()
            # Transition layers (From-lu × To-lu MultiIndex) sometimes leave a residual
            # size-1 non-cell dimension after .sel(); squeeze then reshape to guarantee 1-D.
            vals = np.asarray(vals).squeeze()
            if vals.ndim != 1:
                vals = vals.reshape(-1)
            arr_sel = xr.DataArray(vals, dims=('cell',), coords=is_selected_coords)
            tasks.append(delayed(map2geotiff)(
                index_merc, geo_meta_merc, arr_sel, isInt, hierarchy_tp, layer_magnitude, legend_key,
                real_area_1d,
            ))

        xr_arr.close()

    if not tasks:
        return

    # threads: map2geotiff is NumPy + Rasterio (both release the GIL), so threads get true
    # parallelism with zero pickling cost. Standard layers have few tasks (2–30 per file),
    # so heap-allocator contention between threads is negligible.
    for hierarchy_tp, tif_bytes, attrs in Parallel(
        n_jobs=workers, prefer='threads'
    )(tasks):
        flat_data[hierarchy_tp] = {'tif_b64': base64.b64encode(tif_bytes).decode(), **attrs}

    write_split_by_combo_geotiff(flat_data, dim_names, prefix)
        


def find_chunks_dirs(raw_data_dir: str, base_pattern: str, years: list) -> dict:
    """Return {year: chunks_dir_path} for every year that has a chunks directory."""
    result = {}
    for yr in years:
        cdir = os.path.join(raw_data_dir, f'out_{yr}', f'{base_pattern}_{yr}_chunks')
        if os.path.isdir(cdir):
            result[yr] = cdir
    return result


def write_split_by_combo_paged(
    combo_sp_data: dict,    # {combo_tp: {sp: {year: {tif_b64, ...}}}}
    page_start: int,
    page_end: int,
    prefix: str,
) -> None:
    """Write one JS file per combo for a single page of a paged (NVIS/SNES/ECNES) layer.

    Mirrors write_split_by_combo_geotiff for the paged case.

    JS filename:  {prefix}__{combo}_{page_start}_{page_end}.js
    JS var name:  {prefix}__{combo}               (no page suffix — Vue overwrites per page)
    JS content:   window["prefix__combo"] = {
                    "Species A": {2020: {tif_b64, ...}, 2025: {...}},
                    "Species B": {...},
                  }
    """
    base_name = os.path.basename(prefix)
    tif_dir   = os.path.dirname(prefix)
    os.makedirs(tif_dir, exist_ok=True)

    for combo_tp, sp_year_data in combo_sp_data.items():
        var_suffix  = '__'.join(safe_key(v) for v in combo_tp)
        var_name    = f'{base_name}__{var_suffix}'
        js_filename = f'{var_name}_{page_start}_{page_end}.js'
        with open(os.path.join(tif_dir, js_filename), 'w') as f:
            f.write(f'window["{var_name}"] = ')
            json.dump(sp_year_data, f, separators=(',', ':'))
            f.write(';\n')


def write_paged_index(combo_attrs: dict, dim_names: list, manifest: dict,
                      prefix: str, chunk_num: int = 1) -> None:
    """Write the __index.js for a paged (NVIS/SNES/ECNES) layer set.

    chunk_num must match the value used in get_map2json_paged so that each
    page entry's [start, end] maps to an existing JS file.

    Index format::

        window["<base>__index"] = {
          dims:     ["lm", "lu"],          // no species dim
          tree:     {"Dryland": ["Nuts"]},
          combos:   {"Dryland__Nuts": {intOrFloat, min_max, raw_min_max}},
          pages:    {"0": {start:0, end:20, species:[...]}, ...},
          pageSize: 20,
          paged:    true
        }
    """
    base_name = os.path.basename(prefix)
    os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)

    tree = build_tree(list(combo_attrs.keys()))
    combos_out = {
        '__'.join(safe_key(v) for v in combo): {k: v for k, v in attrs.items() if k != 'tif_b64'}
        for combo, attrs in combo_attrs.items()
    }

    # Build cumulative species list from manifest (chunk order).
    sorted_chunk_keys = sorted(manifest.keys(), key=int)
    cumulative = 0
    chunk_starts = {}
    chunk_species = {}
    for idx_str in sorted_chunk_keys:
        chunk_starts[int(idx_str)] = cumulative
        chunk_species[int(idx_str)] = manifest[idx_str]
        cumulative += len(manifest[idx_str])

    # Merge chunk_num chunks per page so [start, end] matches the JS filenames.
    pages_out = {}
    page_idx = 0
    for group_start_pos in range(0, len(sorted_chunk_keys), chunk_num):
        group_keys = [int(sorted_chunk_keys[i]) for i in range(group_start_pos, min(group_start_pos + chunk_num, len(sorted_chunk_keys)))]
        page_start = chunk_starts[group_keys[0]]
        last_key   = group_keys[-1]
        page_end   = chunk_starts[last_key] + len(chunk_species[last_key])
        species    = [sp for k in group_keys for sp in chunk_species[k]]
        pages_out[str(page_idx)] = {'start': page_start, 'end': page_end, 'species': species}
        page_idx += 1

    batch_size = len(manifest['0']) if manifest else 10
    index = {
        'dims':     dim_names,
        'tree':     tree,
        'combos':   combos_out,
        'pages':    pages_out,
        'pageSize': batch_size * chunk_num,
        'paged':    True,
    }
    index_name = f'{base_name}__index'
    with open(f'{prefix}__index.js', 'w') as f:
        f.write(f'window["{index_name}"] = ')
        json.dump(index, f, separators=(',', ':'), indent=2)
        f.write(';\n')


def get_map2json_paged(
    chunks_dirs: dict,        # {year: path_to_chunks_dir}
    float_magnitude: tuple,
    save_path: str,
    chunk_num: int = 1,
) -> None:
    """Convert chunked NCs (NVIS/SNES/ECNES) to paged JS layer files.

    chunk_num controls how many chunk NC files are merged into one JS output.
    Each chunk NC holds N species (default 10). chunk_num=10 → 100 species per JS file.

    JS filename:  {prefix}__{combo}_{page_start}_{page_end}.js
    JS var name:  {prefix}__{combo}               (no page suffix — Vue overwrites per page)
    JS content:   window["prefix__combo"] = {
                    "Species A": {2020: {tif_b64, intOrFloat, min_max, raw_min_max}, ...},
                    "Species B": {...},
                  }
    """
    if not chunks_dirs:
        return

    ncells = 5_000_000 // (settings.RESFACTOR ** 2)
    mem_per_worker = ncells * 7 * 4 / 1e6
    cpu_cap = 61 if os.name == 'nt' else (os.cpu_count() or 1)
    workers = min(cpu_cap, max(1, int(settings.WRITE_REPORT_MAX_MEM_MB // mem_per_worker)))

    prefix = save_path.removesuffix('.js')

    # Pre-build EPSG:3857 metadata once per year.
    year_meta: dict = {}
    for year, chunks_dir in chunks_dirs.items():
        year_meta[year] = build_year_meta(year, os.path.dirname(chunks_dir))

    # Read manifest from the first year's chunks dir.
    first_dir  = next(iter(chunks_dirs.values()))
    chunk_files_first = sorted(glob.glob(os.path.join(first_dir, 'chunk_*.nc')))
    if not chunk_files_first:
        return
    with open(os.path.join(first_dir, 'manifest.json')) as f:
        manifest = json.load(f)   # {"0": [sp1, sp2, ...], "1": [...], ...}

    # Introspect dimension names from the first chunk (species-like dim excluded).
    SPECIES_DIMS = {'species', 'group', 'community'}
    with xr.open_dataset(chunk_files_first[0]) as ds:
        sample_da = cfxr.decode_compress_to_multi_index(ds, 'layer')['data']
        sample_layers = sample_da['layer'].to_index().to_frame().to_dict(orient='records')
    species_dim = next((k for k in sample_layers[0] if k in SPECIES_DIMS), 'species')
    sample_no_sp = {k: v for k, v in sample_layers[0].items() if k != species_dim}
    dim_names = list(rename_reorder_hierarchy(sample_no_sp).keys())

    chunk_indices = sorted(
        int(re.search(r'chunk_(\d+)', os.path.basename(f)).group(1))
        for f in chunk_files_first
    )
    # Pre-compute cumulative page starts from the manifest (agnostic to batch size).
    cumulative = 0
    page_starts: dict = {}
    for idx_str in sorted(manifest.keys(), key=int):
        page_starts[int(idx_str)] = cumulative
        cumulative += len(manifest[idx_str])

    all_combo_attrs: dict = {}   # {combo_tuple: attrs}  — for the index JS

    for group_start_pos in range(0, len(chunk_indices), chunk_num):
        group = chunk_indices[group_start_pos : group_start_pos + chunk_num]
        page_start = page_starts[group[0]]
        last_idx   = group[-1]
        page_end   = page_starts[last_idx] + len(manifest[str(last_idx)])

        # Accumulate all (chunk × year) tasks for this page before launching workers.
        chunk_data: dict = {}   # {(combo_tp, sp): {year: {tif_b64, ...}}}
        tasks: list = []

        for chunk_idx in group:
            for year, chunks_dir in sorted(chunks_dirs.items()):
                nc_path = os.path.join(chunks_dir, f'chunk_{chunk_idx:06d}.nc')
                if not os.path.exists(nc_path):
                    continue

                index_merc, geo_meta_merc, real_area_1d = year_meta[year]

                xr_arr = cfxr.decode_compress_to_multi_index(
                    xr.open_dataset(nc_path, chunks={}), 'layer'
                )['data']
                valid_layers = xr_arr['layer'].to_index().to_frame().to_dict(orient='records')
                is_selected_arr = (
                    np.asarray(xr_arr['is_selected'].values, dtype=bool)
                    if 'is_selected' in xr_arr.coords else None
                )
                is_selected_coords = (
                    {'is_selected': ('cell', is_selected_arr)}
                    if is_selected_arr is not None else None
                )

                for sel in valid_layers:
                    sp_val    = sel[species_dim]
                    sel_no_sp = {k: v for k, v in sel.items() if k != species_dim}
                    renamed   = rename_reorder_hierarchy(sel_no_sp)
                    combo_tp  = tuple(renamed.values())
                    hierarchy_tp = (*combo_tp, sp_val, int(year))

                    vals = xr_arr.sel(**sel).values.copy()
                    vals = np.asarray(vals).squeeze()
                    if vals.ndim != 1:
                        vals = vals.reshape(-1)
                    arr_sel = xr.DataArray(vals, dims=('cell',), coords=is_selected_coords)
                    tasks.append(delayed(map2geotiff)(
                        index_merc, geo_meta_merc, arr_sel, False,
                        hierarchy_tp, float_magnitude, None, real_area_1d,
                    ))

                xr_arr.close()

        # loky (processes): paged layers submit 1000+ tasks per page, each allocating a large
        # arr_2d buffer. Under threads, all workers share one heap — the OS allocator serialises
        # concurrent malloc/free calls, causing heavy lock contention that outweighs pickling
        # savings. Loky workers have isolated heaps so allocations never contend.
        # max_nbytes memmaps index_merc (≥200 MB) to avoid pickling it per task.
        for hierarchy_tp, tif_bytes, attrs in Parallel(
            n_jobs=workers, max_nbytes=1_000_000
        )(tasks):
            combo_tp = hierarchy_tp[:-2]
            sp_val   = hierarchy_tp[-2]
            year_val = hierarchy_tp[-1]
            key = (combo_tp, sp_val)
            chunk_data.setdefault(key, {})[year_val] = {
                'tif_b64': base64.b64encode(tif_bytes).decode(),
                **attrs,
            }
            all_combo_attrs[combo_tp] = attrs   # last-write wins — all years share attrs

        combo_sp_data: dict = {}   # {combo_tp: {sp: {year: data}}}
        for (combo_tp, sp_val), year_data in chunk_data.items():
            combo_sp_data.setdefault(combo_tp, {})[sp_val] = year_data

        write_split_by_combo_paged(combo_sp_data, page_start, page_end, prefix)

    write_paged_index(all_combo_attrs, dim_names, manifest, prefix, chunk_num=chunk_num)


def save_report_layer(raw_data_dir: str):

    SAVE_DIR = f'{raw_data_dir}/DATA_REPORT/data'

    # Load the cell magnitudes for all layers, which will be used to set colorbar limits in the report
    with open(f'{raw_data_dir}/max_cell_magnitudes.json', 'r') as f:
        cell_magnitudes = json.load(f)

    # Create the directory if it does not exist
    if not os.path.exists(f'{SAVE_DIR}/map_layers'):
        os.makedirs(f'{SAVE_DIR}/map_layers', exist_ok=True)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir).query('category == "xarray_layer"')
    files['Year'] = files['Year'].astype(int)
    files = files.query(f'Year.isin({sorted(settings.SIM_YEARS)})')
    
    


    ####################################################
    #                    0) Legend                     #
    ####################################################

    legend_registry = {
        'ag':     build_map_legend(COLOR_AG),
        'am':     build_map_legend(COLOR_AM),
        'non_ag': build_map_legend(COLOR_NON_AG),
        'lumap':  build_map_legend(COLOR_LUMAP),
    }
    
    with open(f'{SAVE_DIR}/map_layers/legend_registry.js', 'w') as _f:
        _f.write('window["legend_registry"] = ')
        json.dump(legend_registry, _f, separators=(',', ':'))
        _f.write(';\n')
        
    print('│   ├── Legend registry saved.')
    

    ####################################################
    #                   1) LUMAP Layer                 #
    ####################################################
    lumap_ly = files.query('base_name == "xr_map_lumap"')
    get_map2json(lumap_ly, 'lm', None, f'{SAVE_DIR}/map_layers/map_dvar_lumap.js', legend_key_int='lumap')
    print('│   ├── LUMAP layer saved.')


 
    ####################################################
    #                   2) Dvar Layer                  #
    ####################################################
    
    dvar_min_max = (0, 1)

    dvar_ag = files.query('base_name == "xr_dvar_ag"')
    get_map2json(dvar_ag, {'lu':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_Ag.js', legend_key_int='ag')
    print('│   ├── Dvar Ag layer saved.')

    dvar_am = files.query('base_name == "xr_dvar_am"')
    get_map2json(dvar_am, {'am':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_Am.js', legend_key_int='am')
    print('│   ├── Dvar Am layer saved.')

    dvar_nonag = files.query('base_name == "xr_dvar_non_ag"')
    get_map2json(dvar_nonag, {'lu':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_NonAg.js', legend_key_int='non_ag')
    print('│   ├── Dvar Non-Ag layer saved.')
    
    
    
    ####################################################
    #                   3) Area Layer                  #
    ####################################################

    files_area = files.query('base_name.str.contains("area")')
    
    area_magnitudes = (
        *cell_magnitudes['area']['ag'], 
        *cell_magnitudes['area']['non_ag'], 
        *cell_magnitudes['area']['am']
    )
    
    area_min_max = (min(area_magnitudes), max(area_magnitudes))
    

    area_ag = files_area.query('base_name == "xr_area_agricultural_landuse"')
    get_map2json(area_ag, {'lu':'ALL'}, area_min_max, f'{SAVE_DIR}/map_layers/map_area_Ag.js', legend_key_int='ag')
    print('│   ├── Area Ag layer saved.')
    
    # files_df,legend_int,legend_int_level,legend_float,float_magnitude,save_path = area_ag, legend_ag, {'lu':'ALL'}, legend_float, area_min_max, f'{SAVE_DIR}/map_layers/map_area_Ag.js'
    
    area_nonag = files_area.query('base_name == "xr_area_non_agricultural_landuse"')
    get_map2json(area_nonag, {'lu':'ALL'}, area_min_max, f'{SAVE_DIR}/map_layers/map_area_NonAg.js', legend_key_int='non_ag')
    print('│   ├── Area Non-Ag layer saved.')

    area_am = files_area.query('base_name == "xr_area_agricultural_management"')
    get_map2json(area_am, {'am':'ALL'}, area_min_max, f'{SAVE_DIR}/map_layers/map_area_Am.js', legend_key_int='am')
    print('│   ├── Area Am layer saved.')


    ####################################################
    #                  4) Biodiversity                 #
    ####################################################

    files_bio = files.query('base_name.str.contains("biodiversity")')
    
    bio_quality_vals = [v for lst in cell_magnitudes.get('bio_quality', {}).values() for v in lst]
    gbf2_vals        = [v for lst in cell_magnitudes.get('biodiversity_GBF2',       {}).values() for v in lst]
    gbf3_vals        = [v for lst in cell_magnitudes.get('biodiversity_GBF3',       {}).values() for v in lst]
    gbf4_snes_vals   = [v for lst in cell_magnitudes.get('biodiversity_GBF4_SNES',  {}).values() for v in lst]
    gbf4_ecnes_vals  = [v for lst in cell_magnitudes.get('biodiversity_GBF4_ECNES', {}).values() for v in lst]
    gbf8_vals        = [v for lst in cell_magnitudes.get('biodiversity_GBF8',       {}).values() for v in lst]
    
    bio_min_max        = (min(bio_quality_vals),    max(bio_quality_vals))   # bio_quality always has values 
    gbf2_min_max       = (min(gbf2_vals),           max(gbf2_vals))          if gbf2_vals          else bio_min_max
    gbf3_min_max       = (min(gbf3_vals),           max(gbf3_vals))          if gbf3_vals          else bio_min_max
    gbf4_snes_min_max  = (min(gbf4_snes_vals),      max(gbf4_snes_vals))     if gbf4_snes_vals     else bio_min_max
    gbf4_ecnes_min_max = (min(gbf4_ecnes_vals),     max(gbf4_ecnes_vals))    if gbf4_ecnes_vals    else bio_min_max
    gbf8_min_max       = (min(gbf8_vals),           max(gbf8_vals))          if gbf8_vals          else bio_min_max
    
    
    # Quality layers
    bio_overall_ag = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag"')
    get_map2json(bio_overall_ag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_Ag.js')
    print('│   ├── Biodiversity Overall Ag layer saved.')

    bio_overall_nonag = files_bio.query('base_name == "xr_biodiversity_overall_priority_non_ag"')
    get_map2json(bio_overall_nonag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_NonAg.js')
    print('│   ├── Biodiversity Overall Non-Ag layer saved.')

    bio_overall_am = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag_management"')
    get_map2json(bio_overall_am, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_Am.js')
    print('│   ├── Biodiversity Overall Am layer saved.')

    bio_overall_all = files_bio.query('base_name == "xr_biodiversity_overall_priority_all"')
    get_map2json(bio_overall_all, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_All.js')
    print('│   ├── Biodiversity Overall All layer saved.')


    # GBF2
    if settings.GBF2_TARGET != 'off':
        bio_GBF2_ag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag"')
        get_map2json(bio_GBF2_ag, None, gbf2_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Ag.js')
        print('│   ├── Biodiversity GBF2 Ag layer saved.')

        bio_GBF2_am = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag_management"')
        get_map2json(bio_GBF2_am, None, gbf2_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Am.js')
        print('│   ├── Biodiversity GBF2 Am layer saved.')

        bio_GBF2_nonag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_non_ag"')
        get_map2json(bio_GBF2_nonag, None, gbf2_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_NonAg.js')
        print('│   ├── Biodiversity GBF2 Non-Ag layer saved.')

        bio_GBF2_sum = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_sum"')
        get_map2json(bio_GBF2_sum, None, gbf2_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Sum.js')
        print('│   ├── Biodiversity GBF2 Sum layer saved.')

    # GBF3-NVIS (chunked)
    if settings.WRITE_GBF3_NVIS != 'off':
        _nvis_types = [
            ('xr_biodiversity_GBF3_NVIS_ag',             'map_bio_GBF3_NVIS_Ag',    'Biodiversity GBF3_NVIS Ag'),
            ('xr_biodiversity_GBF3_NVIS_ag_management',  'map_bio_GBF3_NVIS_Am',    'Biodiversity GBF3_NVIS Am'),
            ('xr_biodiversity_GBF3_NVIS_non_ag',         'map_bio_GBF3_NVIS_NonAg', 'Biodiversity GBF3_NVIS Non-Ag'),
            ('xr_biodiversity_GBF3_NVIS_sum',            'map_bio_GBF3_NVIS_Sum',   'Biodiversity GBF3_NVIS Sum'),
        ]
        for base_pat, map_name, label in _nvis_types:
            cdirs = find_chunks_dirs(raw_data_dir, base_pat, sorted(settings.SIM_YEARS))
            get_map2json_paged(cdirs, gbf3_min_max, f'{SAVE_DIR}/map_layers/{map_name}.js', chunk_num=10)
            print(f'│   ├── {label} layer saved.')

    # GBF4-SNES (chunked)
    if settings.WRITE_GBF4_SNES != 'off':
        _snes_types = [
            ('xr_biodiversity_GBF4_SNES_ag',            'map_bio_GBF4_SNES_Ag',    'Biodiversity GBF4_SNES Ag'),
            ('xr_biodiversity_GBF4_SNES_ag_management', 'map_bio_GBF4_SNES_Am',    'Biodiversity GBF4_SNES Am'),
            ('xr_biodiversity_GBF4_SNES_non_ag',        'map_bio_GBF4_SNES_NonAg', 'Biodiversity GBF4_SNES Non-Ag'),
            ('xr_biodiversity_GBF4_SNES_sum',           'map_bio_GBF4_SNES_Sum',   'Biodiversity GBF4_SNES Sum'),
        ]
        for base_pat, map_name, label in _snes_types:
            cdirs = find_chunks_dirs(raw_data_dir, base_pat, sorted(settings.SIM_YEARS))
            get_map2json_paged(cdirs, gbf4_snes_min_max, f'{SAVE_DIR}/map_layers/{map_name}.js', chunk_num=10)
            print(f'│   ├── {label} layer saved.')

    # GBF4-ECNES (chunked)
    if settings.WRITE_GBF4_ECNES != 'off':
        _ecnes_types = [
            ('xr_biodiversity_GBF4_ECNES_ag',            'map_bio_GBF4_ECNES_Ag',    'Biodiversity GBF4_ECNES Ag'),
            ('xr_biodiversity_GBF4_ECNES_ag_management', 'map_bio_GBF4_ECNES_Am',    'Biodiversity GBF4_ECNES Am'),
            ('xr_biodiversity_GBF4_ECNES_non_ag',        'map_bio_GBF4_ECNES_NonAg', 'Biodiversity GBF4_ECNES Non-Ag'),
            ('xr_biodiversity_GBF4_ECNES_sum',           'map_bio_GBF4_ECNES_Sum',   'Biodiversity GBF4_ECNES Sum'),
        ]
        for base_pat, map_name, label in _ecnes_types:
            cdirs = find_chunks_dirs(raw_data_dir, base_pat, sorted(settings.SIM_YEARS))
            get_map2json_paged(cdirs, gbf4_ecnes_min_max, f'{SAVE_DIR}/map_layers/{map_name}.js', chunk_num=10)
            print(f'│   ├── {label} layer saved.')

    # GBF8
    if settings.GBF8_TARGET != 'off':
        bio_GBF8_ag = files_bio.query('base_name == "xr_biodiversity_GBF8_species_ag"')
        get_map2json(bio_GBF8_ag, None, gbf8_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_Ag.js')
        print('│   ├── Biodiversity GBF8 Ag layer saved.')

        bio_GBF8_am = files_bio.query('base_name == "xr_biodiversity_GBF8_species_ag_management"')
        get_map2json(bio_GBF8_am, None, gbf8_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_Am.js')
        print('│   ├── Biodiversity GBF8 Am layer saved.')

        bio_GBF8_nonag = files_bio.query('base_name == "xr_biodiversity_GBF8_species_non_ag"')
        get_map2json(bio_GBF8_nonag, None, gbf8_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_NonAg.js')
        print('│   ├── Biodiversity GBF8 Non-Ag layer saved.')

        bio_GBF8_ag_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_ag"')
        get_map2json(bio_GBF8_ag_group, None, gbf8_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_Ag.js')
        print('│   ├── Biodiversity GBF8 Groups Ag layer saved.')

        bio_GBF8_am_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_ag_management"')
        get_map2json(bio_GBF8_am_group, None, gbf8_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_Am.js')
        print('│   ├── Biodiversity GBF8 Groups Am layer saved.')

        bio_GBF8_nonag_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_non_ag"')
        get_map2json(bio_GBF8_nonag_group, None, gbf8_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_NonAg.js')
        print('│   ├── Biodiversity GBF8 Groups Non-Ag layer saved.')

    


    ####################################################
    #                    5) Economics                  #
    ####################################################
    
    ecnomic_magnitudes_ag = (
        *cell_magnitudes['Economics_ag']['ag_revenue'],
        *cell_magnitudes['Economics_ag']['ag_cost'],
        *cell_magnitudes['Economics_ag']['profit_ag']
    )
    ecnomic_magnitudes_nonag = (
        *cell_magnitudes['Economics_am']['am_revenue'],
        *cell_magnitudes['Economics_am']['am_cost'],
        *cell_magnitudes['Economics_am']['am_profit']
    )
    ecnomic_magnitudes_am = (
        *cell_magnitudes['Economics_non_ag']['non_ag_revenue'],
        *cell_magnitudes['Economics_non_ag']['non_ag_cost'],
        *cell_magnitudes['Economics_non_ag']['non_ag_profit']
    )
    economic_magnitudes_transition = (
        *cell_magnitudes['Economics_ag']['ag2ag_cost'],
        *cell_magnitudes['Economics_ag']['non_ag2ag_cost'],
        # Am transition is all zeros (CAPEX moved to cost matrix), so excluded here
        *cell_magnitudes['Economics_non_ag']['nonag2nonag_cost'],
        *cell_magnitudes['Economics_non_ag']['ag2nonag_cost']
    )
    
    economic_min_max_ag = (min(ecnomic_magnitudes_ag), max(ecnomic_magnitudes_ag))
    economic_min_max_nonag = (min(ecnomic_magnitudes_nonag), max(ecnomic_magnitudes_nonag))
    economic_min_max_am = (min(ecnomic_magnitudes_am), max(ecnomic_magnitudes_am))
    economic_min_max_transition = (min(economic_magnitudes_transition), max(economic_magnitudes_transition))

    # ---------------- Profit ----------------
    profit_ag = files.query('base_name == "xr_economics_ag_profit"')
    get_map2json(profit_ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_profit.js')
    print('│   ├── Economics Ag profit layer saved.')

    profit_am = files.query('base_name == "xr_economics_am_profit"')
    get_map2json(profit_am, None, economic_min_max_am, f'{SAVE_DIR}/map_layers/map_economics_Am_profit.js')
    print('│   ├── Economics Am profit layer saved.')

    profit_nonag = files.query('base_name == "xr_economics_non_ag_profit"')
    get_map2json(profit_nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_profit.js')
    print('│   ├── Economics NonAg profit layer saved.')

    # Sum profit (Ag + Am + NonAg)
    economic_magnitudes_sum = cell_magnitudes.get('Economics_sum', {}).get('sum_profit', [0.0, 0.0])
    economic_min_max_sum = (min(economic_magnitudes_sum), max(economic_magnitudes_sum))
    profit_sum = files.query('base_name == "xr_economics_sum_profit"')
    get_map2json(profit_sum, None, economic_min_max_sum, f'{SAVE_DIR}/map_layers/map_economics_Sum_profit.js')
    print('│   ├── Economics Sum profit layer saved.')


    # ---------------- Revenue ----------------
    revenue_ag = files.query('base_name == "xr_economics_ag_revenue"')
    get_map2json(revenue_ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_revenue.js')
    print('│   ├── Economics Ag revenue layer saved.')

    revenue_am = files.query('base_name == "xr_economics_am_revenue"')
    get_map2json(revenue_am, None, economic_min_max_am, f'{SAVE_DIR}/map_layers/map_economics_Am_revenue.js')
    print('│   ├── Economics Am revenue layer saved.')
    
    revenue_nonag = files.query('base_name == "xr_economics_non_ag_revenue"')
    get_map2json(revenue_nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_revenue.js')
    print('│   ├── Economics NonAg revenue layer saved.')


    # ---------------- Cost ----------------
    cost_ag = files.query('base_name == "xr_economics_ag_cost"')
    get_map2json(cost_ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_cost.js')
    print('│   ├── Economics Ag cost layer saved.')

    cost_am = files.query('base_name == "xr_economics_am_cost"')
    get_map2json(cost_am, None, economic_min_max_am, f'{SAVE_DIR}/map_layers/map_economics_Am_cost.js')
    print('│   ├── Economics Am cost layer saved.')

    cost_nonag = files.query('base_name == "xr_economics_non_ag_cost"')
    get_map2json(cost_nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_cost.js')
    print('│   ├── Economics NonAg cost layer saved.')


    # ---------------- Transition Cost ----------------
    cost_trans_ag2ag = files.query('base_name == "xr_economics_ag_transition_Ag2Ag"')
    get_map2json(cost_trans_ag2ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_transition_ag2ag.js')
    print('│   ├── Economics Ag transition ag2ag layer saved.')

    cost_trans_nonag2ag = files.query('base_name == "xr_economics_ag_transition_NonAg2Ag"')
    get_map2json(cost_trans_nonag2ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_transition_nonag2ag.js')
    print('│   ├── Economics Ag transition nonag2ag layer saved.')

    cost_trans_ag2nonag = files.query('base_name == "xr_economics_non_ag_transition_Ag2NonAg"')
    get_map2json(cost_trans_ag2nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_transition_ag2non_ag.js')
    print('│   ├── Economics NonAg transition ag2nonag layer saved.')

    cost_trans_nonag2nonag = files.query('base_name == "xr_economics_non_ag_transition_NonAg2NonAg"')
    get_map2json(cost_trans_nonag2nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_transition_nonag2nonag.js')
    print('│   ├── Economics NonAg transition nonag2nonag layer saved.')

    # AgMgt has 0 transition cost, so skipping
    # cost_ag_man = files.query('base_name == "xr_economics_am_transition"')

    # Non-Ag to Ag transition cost is not allowed, so skipping
    # cost_trans_nonag2ag = files.query('base_name == "xr_economics_non_ag_transition_NonAg2NonAg"')
    
    


    ####################################################
    #                    6) GHG                        #
    ####################################################

    files_ghg = files.query('base_name.str.contains("GHG")')
    
    ghg_magnitudes = (
        *cell_magnitudes['ghg_emission']['ag'],
        *cell_magnitudes['ghg_emission']['non_ag'],
        *cell_magnitudes['ghg_emission']['ag_man'],
        # *cell_magnitudes['ghg_emission']['transition'],
        *cell_magnitudes['ghg_emission'].get('sum', []),
    )

    ghg_min_max = (min(ghg_magnitudes), max(ghg_magnitudes))

    ghg_ag = files_ghg.query('base_name == "xr_GHG_ag"')
    get_map2json(ghg_ag, None, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_Ag.js')
    print('│   ├── GHG Ag layer saved.')

    ghg_am = files_ghg.query('base_name == "xr_GHG_ag_management"')
    get_map2json(ghg_am, None, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_Am.js')
    print('│   ├── GHG Am layer saved.')

    ghg_nonag = files_ghg.query('base_name == "xr_GHG_non_ag"')
    get_map2json(ghg_nonag, None, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_NonAg.js')
    print('│   ├── GHG Non-Ag layer saved.')

    ghg_sum = files_ghg.query('base_name == "xr_GHG_sum"')
    get_map2json(ghg_sum, None, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_Sum.js')
    print('│   ├── GHG Sum layer saved.')



    ####################################################
    #                  7) Quantities                   #
    ####################################################

    files_quantities = files.query('base_name.str.contains("quantities")')
    
    prod_magnitudes = cell_magnitudes['production']
    prod_min_max = {k: (min(v), max(v)) for k, v in prod_magnitudes.items()}

    quantities_ag = files_quantities.query('base_name == "xr_quantities_agricultural"')
    get_map2json(quantities_ag, {'Commodity':'ALL'}, prod_min_max, f'{SAVE_DIR}/map_layers/map_quantities_Ag.js', legend_key_int='ag')
    print('│   ├── Quantities Ag layer saved.')

    quantities_am = files_quantities.query('base_name == "xr_quantities_agricultural_management"')
    get_map2json(quantities_am, {'Commodity':'ALL'}, prod_min_max, f'{SAVE_DIR}/map_layers/map_quantities_Am.js', legend_key_int='am')
    print('│   ├── Quantities Am layer saved.')

    quantities_nonag = files_quantities.query('base_name == "xr_quantities_non_agricultural"')
    get_map2json(quantities_nonag, {'Commodity':'ALL'}, prod_min_max, f'{SAVE_DIR}/map_layers/map_quantities_NonAg.js', legend_key_int='non_ag')
    print('│   ├── Quantities Non-Ag layer saved.')

    quantities_sum = files_quantities.query('base_name == "xr_quantities_sum"')
    get_map2json(quantities_sum, {'Commodity':'ALL'}, prod_min_max, f'{SAVE_DIR}/map_layers/map_quantities_Sum.js', legend_key_int='ag')
    print('│   ├── Quantities Sum layer saved.')



    ####################################################
    #                8) Transitions                    #
    ####################################################

    # Transitions Area (Ag2Ag)
    files_transition_area_ag2ag = files.query('base_name == "xr_transition_ag2ag_area"')
    trans_area_ag2ag_magnitudes = cell_magnitudes['transition_area']['ag2ag']
    trans_area_ag2ag_min_max = (min(trans_area_ag2ag_magnitudes), max(trans_area_ag2ag_magnitudes))
    get_map2json(files_transition_area_ag2ag, None, trans_area_ag2ag_min_max, f'{SAVE_DIR}/map_layers/map_transition_area_ag2ag.js')
    print('│   ├── Transition Area (Ag2Ag) layer saved.')


    # Transitions Area (Ag2NonAg)
    files_transition_area_ag2nonag = files.query('base_name == "xr_transition_ag2nonag_area"')
    trans_area_ag2non_ag_magnitudes = cell_magnitudes['transition_area']['ag2non_ag']
    trans_area_ag2non_ag_min_max = (min(trans_area_ag2non_ag_magnitudes), max(trans_area_ag2non_ag_magnitudes))
    get_map2json(files_transition_area_ag2nonag, None, trans_area_ag2non_ag_min_max, f'{SAVE_DIR}/map_layers/map_transition_area_ag2nonag.js')
    print('│   ├── Transition Area (Ag2NonAg) layer saved.')


    # Transitions Cost (Ag2Ag)
    files_transition_cost_ag2ag = files.query('base_name == "xr_transition_ag2ag_cost"')
    trans_cost_ag2ag_magnitudes = cell_magnitudes['Economics_ag']['ag2ag_cost']
    trans_cost_ag2ag_min_max = (min(trans_cost_ag2ag_magnitudes), max(trans_cost_ag2ag_magnitudes))
    get_map2json(files_transition_cost_ag2ag, None, trans_cost_ag2ag_min_max, f'{SAVE_DIR}/map_layers/map_transition_cost_ag2ag.js')
    print('│   ├── Transition Cost (Ag2Ag) layer saved.')


    # Transitions Cost (Ag2NonAg)
    files_transition_cost_ag2nonag = files.query('base_name == "xr_transition_ag2nonag_cost"')
    trans_cost_ag2nonag_magnitudes = cell_magnitudes['Economics_non_ag']['ag2nonag_cost']
    trans_cost_ag2nonag_min_max = (min(trans_cost_ag2nonag_magnitudes), max(trans_cost_ag2nonag_magnitudes))
    get_map2json(files_transition_cost_ag2nonag, None, trans_cost_ag2nonag_min_max, f'{SAVE_DIR}/map_layers/map_transition_cost_ag2nonag.js')
    print('│   ├── Transition Cost (Ag2NonAg) layer saved.')




    ####################################################
    #               9) Water Yield                     #
    ####################################################

    files_water = files.query('base_name.str.contains("water_yield")')
    
    water_magnitudes = (
        *cell_magnitudes['water_yield']['ag'],
        *cell_magnitudes['water_yield']['non_ag'],
        *cell_magnitudes['water_yield']['am'],
        *cell_magnitudes['water_yield']['sum'],
    )

    water_min_max = (min(water_magnitudes), max(water_magnitudes))

    water_sum = files_water.query('base_name == "xr_water_yield_sum"')
    get_map2json(water_sum, None, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_Sum.js')
    print('│   ├── Water Yield Sum layer saved.')

    water_ag = files_water.query('base_name == "xr_water_yield_ag"')
    get_map2json(water_ag, None, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_Ag.js')
    print('│   ├── Water Yield Ag layer saved.')

    water_am = files_water.query('base_name == "xr_water_yield_ag_management"')
    get_map2json(water_am, None, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_Am.js')
    print('│   ├── Water Yield Am layer saved.')

    water_nonag = files_water.query('base_name == "xr_water_yield_non_ag"')
    get_map2json(water_nonag, None, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_NonAg.js')
    print('│   ├── Water Yield Non-Ag layer saved.')


    ####################################################
    #             10) Renewable Energy                 #
    ####################################################

    files_renewable = files.query('base_name == "xr_renewable_energy"')

    if not files_renewable.empty:
        re_magnitudes = cell_magnitudes.get('renewable_energy', [0.0, 0.0])
        re_min_max = (min(re_magnitudes), max(re_magnitudes))

        get_map2json(files_renewable, None, re_min_max, f'{SAVE_DIR}/map_layers/map_renewable_energy_Am.js')
        print('│   ├── Renewable Energy Am layer saved.')

    files_renewable_exist = files.query('base_name == "xr_renewable_existing_dvar"')

    if not files_renewable_exist.empty:
        exist_magnitudes = cell_magnitudes.get('renewable_existing_dvar', [0.0, 1.0])
        exist_min_max = (min(exist_magnitudes), max(exist_magnitudes))

        get_map2json(files_renewable_exist, None, exist_min_max, f'{SAVE_DIR}/map_layers/map_renewable_existing_dvar_Am.js')
        print('│   ├── Renewable Existing Dvar Am layer saved.')
        
        
        
        



