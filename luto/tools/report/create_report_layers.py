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


from pathlib import Path
import os
import re
import json
import base64
import zlib
import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray as cfxr
import rioxarray  # .rio accessor on xarray objects

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
    LUMAP_GROUP,
    LUMAP_GROUP_HEX,
    RENAME_AM_NON_AG,
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
    data_dict : {'cells_b64': str, 'intOrFloat': ..., 'min_max'|'legendKey': ...}
    cells_b64 is excluded from the index — it lives in the per-combo JS files.
    """
    base_name = Path(prefix).name
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    combo_map: dict = {}
    for tp, data in flat_data.items():
        combo = tp[:-1]
        combo_map[combo] = {k: v for k, v in data.items() if k != 'cells_b64'}

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


def write_split_by_combo_geotiff(flat_data: dict, dim_names: list, prefix: str, collapse_years: bool = False) -> None:
    """Write one JS file per unique combo (all years) and an index JS file.

    Each per-combo JS file contains the base64-encoded cell vector (map2cells) and display attrs
    for every year, following the same split-file pattern the Vue already uses:
      window["map_area_Ag__Dryland__Wheat"] = {
          2010: {cells_b64: "...", intOrFloat: "float", min_max: [0, 500]},
          2025: {cells_b64: "...", ...},
      };
    """
    base_name = Path(prefix).name
    tif_dir = Path(prefix).parent
    Path(tif_dir).mkdir(parents=True, exist_ok=True)

    combos: dict = {}
    for tp, data in flat_data.items():
        combo, year = tp[:-1], tp[-1]
        combos.setdefault(combo, {})[year] = data

    for combo, year_data in combos.items():
        # `collapse_years`: a layer whose cells are the same in every year is stored once, under 'ALL'
        if collapse_years and len(year_data) > 1 and len({leaf['cells_b64'] for leaf in year_data.values()}) == 1:
            year_data = {'ALL': next(iter(year_data.values()))}
        var_suffix = '__'.join(safe_key(v) for v in combo)
        js_filename = f'{base_name}__{var_suffix}.js'
        var_name = js_filename.removesuffix('.js')
        with open(Path(tif_dir) / js_filename, 'w') as f:
            f.write(f'window["{var_name}"] = ')
            json.dump(year_data, f, separators=(',', ':'))
            f.write(';\n')

    write_geotiff_index(flat_data, dim_names, prefix)



#: A cell of a float layer is stored as one signed byte: 0 = no value, ±1 … ±CELL_LEVELS = the value as a share of
#: the layer's own range (the positive side of `raw_min_max[1]`, the negative side of `raw_min_max[0]`), a value
#: beyond the range clipped to it; CELL_NODATA = not a number (drawn transparent).
CELL_LEVELS = 127
CELL_NODATA = -128


def map2cells(
    arr_sel: xr.DataArray,     # 1D DataArray[cell] — raw data values
    isInt: bool,
    hierarchy_tp: tuple,
    layer_magnitude,           # (global_min, global_max) for float layers; None for int
    legend_key: str | None,
    real_area_1d: np.ndarray,               # 1D float32[cell] — actual ha per cell
    per_ha: bool = True,                    # False: the values are intensive already (%, ML/ha), stored as they are
) -> tuple:
    """Pack a 1D cell array for the report: the land cells only, in the model's cell order, deflated.

    The 2D picture is not stored per layer: the report holds ONE raster of cell numbers (`map_cell_index.js`,
    `write_cell_index`) and the browser paints every layer through it. A layer is therefore the same 1D vector
    the model works with, in the smallest type a map needs:

      int layers (land-use codes)   int16, the codes as they are (-9999 = no data)
      float layers                  int8, each side of zero scaled to the layer's range (CELL_LEVELS); the browser
                                    restores value = byte / CELL_LEVELS x raw_min_max[side], to within one level
                                    (0.8 % of the range). Float layers are divided by the cell's real area (ha)
                                    first unless `per_ha` is False.
    Returns (hierarchy_tp, the deflated bytes, the layer's display attrs).
    """
    src_vals = np.asarray(arr_sel.values, dtype=np.float32)

    if isInt:
        cells = np.where(np.isfinite(src_vals), src_vals, -9999).astype('<i2')
        attrs: dict = {'intOrFloat': 'int', 'legendKey': legend_key}
    else:
        global_min, global_max = layer_magnitude
        if per_ha:
            src_vals = src_vals / np.where(real_area_1d > 0, real_area_1d, np.nan)
            # min_max: colorbar labels — use RF²×121 (≈ max cell area) so bounded
            # layers display ≤1 and the legend matches prior behaviour.
            min_max = (
                round(global_min / settings.RESFACTOR ** 2 / 121, 2),
                round(global_max / settings.RESFACTOR ** 2 / 121, 2),
            )
            # raw_min_max: the range the cells are scaled to — use mean real area to match the
            # per-ha values (raw / cell_real_area).
            mean_area = float(np.nanmean(real_area_1d))
            raw_min_max = (float(global_min / mean_area), float(global_max / mean_area))
        else:
            # Intensive values: the labels and the cells share the layer's own range
            min_max = raw_min_max = (float(global_min), float(global_max))
            min_max = tuple(round(value, 4) for value in min_max)

        low, high = min(raw_min_max[0], 0.0), max(raw_min_max[1], 0.0)
        finite = np.isfinite(src_vals)
        values = np.where(finite, src_vals, 0)
        # ceil: a value that is not zero never lands on 0, the byte of "no value"
        positive = np.ceil(np.clip(values / high, 0, 1) * CELL_LEVELS) if high > 0 else np.zeros_like(values)
        negative = np.ceil(np.clip(values / low, 0, 1) * CELL_LEVELS) if low < 0 else np.zeros_like(values)
        cells = np.where(values > 0, positive, np.where(values < 0, -negative, 0))
        cells = np.where(finite, cells, CELL_NODATA).astype('i1')
        attrs = {'intOrFloat': 'float',
                 'min_max':     list(min_max),
                 'raw_min_max': [float(f'{low:.6g}'), float(f'{high:.6g}')]}

    return hierarchy_tp, zlib.compress(cells.tobytes(), 6), attrs



def write_cell_index(index_merc: np.ndarray, geo_meta_merc: dict, save_dir: str, name: str = 'map_cell_index') -> None:
    """Write `map_cell_index.js`, the raster of the report every layer of the run is painted through: for each pixel
    of the EPSG:3857 grid the number of the cell it shows (-1 = outside the study area), with the grid's extent.
    The input layers are on a grid of their own (the input cache has its own resolution): `name` says which."""
    transform = geo_meta_merc['transform']
    height, width = int(index_merc.shape[-2]), int(index_merc.shape[-1])
    content = {
        'width': width, 'height': height,
        'xmin': transform.c, 'xmax': transform.c + transform.a * width,
        'ymax': transform.f, 'ymin': transform.f + transform.e * height,
        'ncells': int(index_merc.max()) + 1,
        'levels': CELL_LEVELS, 'nodata': CELL_NODATA,
        'index_b64': base64.b64encode(zlib.compress(np.ascontiguousarray(index_merc.reshape(height, width), dtype='<i4').tobytes(), 6)).decode(),
    }
    with open(f'{save_dir}/{name}.js', 'w') as f:
        f.write(f'window["{name}"] = ')
        json.dump(content, f, separators=(',', ':'))
        f.write(';\n')



def build_year_meta(year: int, base_dir: str) -> tuple:
    """Build EPSG:3857 raster metadata for one simulation year.

    Returns (index_merc, geo_meta_merc, real_area_1d):
      index_merc    — (H, W) int32; cell index ≥0, -1 = nodata
      geo_meta_merc — rasterio write kwargs for the reprojected grid
      real_area_1d  — 1D float32[cell] ha per cell, or None if file missing
    """
    ds_template = Path(base_dir) / f'xr_map_template_{year}.nc'
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

    real_area_path = Path(base_dir) / f'xr_area_real_area_ha_{year}.nc'
    real_area_1d = None
    if Path(real_area_path).exists():
        with xr.open_dataset(real_area_path) as ra_ds:
            real_area_1d = ra_ds['data'].values.squeeze().astype(np.float32)

    return index_merc, geo_meta_merc, real_area_1d



def get_map2json(
    files_df: pd.DataFrame,
    legend_int_level: dict | str,
    float_magnitude: tuple | dict,
    save_path: str,
    legend_key_int: str | None = None,
    per_ha: bool = True,
    collapse_years: bool = False,
) -> None:
    """Pack all layers in *files_df* as 1D cell vectors (map2cells), painted by the browser through the
    report's cell-number raster (write_cell_index).

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
        if isinstance(float_magnitude, dict):
            # Keyed by any dim VALUE of the combo (Commodity for production, backend for the
            # bio-quality maps, ...): the first sel value that is a key wins.
            # A dict keyed by the whole combo (the input layers: (input, lm, lu)) gives every layer its own range.
            key = tuple(sel.values()) if tuple(sel.values()) in float_magnitude else next((v for v in sel.values() if v in float_magnitude), None)
            magnitude = float_magnitude[key]
        else:
            magnitude = float_magnitude
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
        index_merc, geo_meta_merc, real_area_1d = build_year_meta(int(row['Year']), Path(row['path']).parent)

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
            # A layer is one value per cell. A file whose layer index leaves another dimension over (the NonAg2Ag
            # transition cost keeps its 29 to-land-uses beside `layer`) is summed over it: the layer is then the
            # total its labels name. (Flattening such an array, as before 2026-11-12, scrambled the map.)
            layer_arr = xr_arr.sel(**sel)
            layer_arr = layer_arr.sum([dim for dim in layer_arr.dims if dim != 'cell'])
            vals = np.asarray(layer_arr.values).reshape(-1).copy()
            arr_sel = xr.DataArray(vals, dims=('cell',), coords=is_selected_coords)
            tasks.append(delayed(map2cells)(
                arr_sel, isInt, hierarchy_tp, layer_magnitude, legend_key, real_area_1d, per_ha,
            ))

        xr_arr.close()

    if not tasks:
        return

    # threads: map2cells is NumPy + Rasterio (both release the GIL), so threads get true
    # parallelism with zero pickling cost. Standard layers have few tasks (2–30 per file),
    # so heap-allocator contention between threads is negligible.
    for hierarchy_tp, cell_bytes, attrs in Parallel(
        n_jobs=workers, prefer='threads'
    )(tasks):
        flat_data[hierarchy_tp] = {'cells_b64': base64.b64encode(cell_bytes).decode(), **attrs}

    write_split_by_combo_geotiff(flat_data, dim_names, prefix, collapse_years)
        


def get_lumap_legend_columns() -> dict:
    """The entries of the land-use map's legend, in its two columns: the agricultural groups and the land outside
    the model's area ('land'), and EVERY non-agricultural land use ('non_ag'), enabled or not — `enabled` is False
    where settings.NON_AG_LAND_USES turns every land use of the entry off (the report greys its name)."""
    def display(group):
        return RENAME_AM_NON_AG.get(group, group)

    ag_groups = {LUMAP_GROUP[lu] for lu in LUMAP_GROUP if lu not in settings.NON_AG_LAND_USES}
    land = [group for group in LUMAP_GROUP_HEX if group in ag_groups]           # in the order of LUMAP_GROUP_HEX
    non_ag: dict = {}
    for lu, enabled in settings.NON_AG_LAND_USES.items():
        entry = non_ag.setdefault(LUMAP_GROUP[lu], {'name': display(LUMAP_GROUP[lu]), 'color': LUMAP_GROUP_HEX[LUMAP_GROUP[lu]], 'enabled': False})
        entry['enabled'] = entry['enabled'] or bool(enabled)
    return {
        'land': [{'name': display(group), 'color': LUMAP_GROUP_HEX[group], 'enabled': True} for group in land]
                + [{'name': COLOR_LUMAP[-1][0], 'color': COLOR_LUMAP[-1][1], 'enabled': True}],
        'non_ag': list(non_ag.values()),
    }



def get_lumap_group_layers(dvar_ag: pd.DataFrame, dvar_nonag: pd.DataFrame, save_path: str) -> None:
    """Write one float layer per entry of the land-use map's legend: the share of each cell (0-1) under the land
    uses of the entry (LUMAP_GROUP: 'Cropping' is the sum of every cropping land use, both irrigation statuses), so a click on a
    legend entry of the Home map can show where that entry is, and how much of each cell it takes.

    The layers are named as the legend names them; an entry the run turned off has none."""
    shares: dict = {}                                           # year -> legend name -> 1D share of the cell
    for files_df, sel_all in [(dvar_ag, {'lm': 'ALL'}), (dvar_nonag, {})]:
        for _, row in files_df.iterrows():
            xr_arr = cfxr.decode_compress_to_multi_index(xr.open_dataset(row['path'], chunks={}), 'layer')['data']
            year_shares = shares.setdefault(int(row['Year']), {})
            for sel in xr_arr['layer'].to_index().to_frame().to_dict(orient='records'):
                if sel['lu'] == 'ALL' or any(sel[dim] != value for dim, value in sel_all.items()):
                    continue
                group = LUMAP_GROUP[sel['lu']]
                name = RENAME_AM_NON_AG.get(group, group)
                values = np.nan_to_num(np.asarray(xr_arr.sel(**sel).values, dtype=np.float32).reshape(-1))
                year_shares[name] = year_shares.get(name, 0) + values
            xr_arr.close()

    # Every entry the run has turned on gets a layer in every year, all zero where the run never used it
    # (`empty`): a click on the legend then shows that there is none, instead of doing nothing.
    columns = get_lumap_legend_columns()
    enabled = [entry['name'] for column in columns.values() for entry in column if entry['enabled'] and entry['name'] != COLOR_LUMAP[-1][0]]
    flat_data: dict = {}
    for year, year_shares in shares.items():
        if not year_shares:
            continue
        zeros = np.zeros_like(next(iter(year_shares.values())))
        for name in enabled:
            values = year_shares.get(name, zeros)
            hierarchy_tp, cell_bytes, attrs = map2cells(
                xr.DataArray(values, dims=('cell',)), False, (name, year), (0, 1), None, None, per_ha=False)
            flat_data[hierarchy_tp] = {'cells_b64': base64.b64encode(cell_bytes).decode(), **attrs, 'empty': not bool(np.any(values > 0))}

    if flat_data:
        write_split_by_combo_geotiff(flat_data, ['lu'], save_path.removesuffix('.js'))



def find_chunks_dirs(raw_data_dir: str, base_pattern: str, years: list) -> dict:
    """Return {year: chunks_dir_path} for every year that has a chunks directory."""
    result = {}
    for yr in years:
        cdir = Path(raw_data_dir) / f'out_{yr}' / f'{base_pattern}_{yr}_chunks'
        if Path(cdir).is_dir():
            result[yr] = cdir
    return result


def write_split_by_combo_paged(
    combo_sp_data: dict,    # {combo_tp: {sp: {year: {cells_b64, ...}}}}
    page_start: int,
    page_end: int,
    prefix: str,
) -> None:
    """Write one JS file per combo for a single page of a paged (NVIS/SNES/ECNES) layer.

    Mirrors write_split_by_combo_geotiff for the paged case.

    JS filename:  {prefix}__{combo}_{page_start}_{page_end}.js
    JS var name:  {prefix}__{combo}               (no page suffix — Vue overwrites per page)
    JS content:   window["prefix__combo"] = {
                    "Species A": {2020: {cells_b64, ...}, 2025: {...}},
                    "Species B": {...},
                  }
    """
    base_name = Path(prefix).name
    tif_dir   = Path(prefix).parent
    Path(tif_dir).mkdir(parents=True, exist_ok=True)

    for combo_tp, sp_year_data in combo_sp_data.items():
        var_suffix  = '__'.join(safe_key(v) for v in combo_tp)
        var_name    = f'{base_name}__{var_suffix}'
        js_filename = f'{var_name}_{page_start}_{page_end}.js'
        with open(Path(tif_dir) / js_filename, 'w') as f:
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
    base_name = Path(prefix).name
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    tree = build_tree(list(combo_attrs.keys()))
    combos_out = {
        '__'.join(safe_key(v) for v in combo): {k: v for k, v in attrs.items() if k != 'cells_b64'}
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
                    "Species A": {2020: {cells_b64, intOrFloat, min_max, raw_min_max}, ...},
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
        year_meta[year] = build_year_meta(year, Path(chunks_dir).parent)

    # Read manifest from the first year's chunks dir.
    first_dir  = next(iter(chunks_dirs.values()))
    chunk_files_first = sorted(Path(first_dir).glob('chunk_*.nc'))
    if not chunk_files_first:
        return
    with open(Path(first_dir) / 'manifest.json') as f:
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
        int(re.search(r'chunk_(\d+)', Path(f).name).group(1))
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
        chunk_data: dict = {}   # {(combo_tp, sp): {year: {cells_b64, ...}}}
        tasks: list = []

        for chunk_idx in group:
            for year, chunks_dir in sorted(chunks_dirs.items()):
                nc_path = Path(chunks_dir) / f'chunk_{chunk_idx:06d}.nc'
                if not Path(nc_path).exists():
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
                    tasks.append(delayed(map2cells)(
                        arr_sel, False, hierarchy_tp, float_magnitude, None, real_area_1d,
                    ))

                xr_arr.close()

        # loky (processes): paged layers submit 1000+ tasks per page, each allocating a large
        # arr_2d buffer. Under threads, all workers share one heap — the OS allocator serialises
        # concurrent malloc/free calls, causing heavy lock contention that outweighs pickling
        # savings. Loky workers have isolated heaps so allocations never contend.
        # max_nbytes memmaps index_merc (≥200 MB) to avoid pickling it per task.
        for hierarchy_tp, cell_bytes, attrs in Parallel(
            n_jobs=workers, max_nbytes=1_000_000
        )(tasks):
            combo_tp = hierarchy_tp[:-2]
            sp_val   = hierarchy_tp[-2]
            year_val = hierarchy_tp[-1]
            key = (combo_tp, sp_val)
            chunk_data.setdefault(key, {})[year_val] = {
                'cells_b64': base64.b64encode(cell_bytes).decode(),
                **attrs,
            }
            all_combo_attrs[combo_tp] = attrs   # last-write wins — all years share attrs

        combo_sp_data: dict = {}   # {combo_tp: {sp: {year: data}}}
        for (combo_tp, sp_val), year_data in chunk_data.items():
            combo_sp_data.setdefault(combo_tp, {})[sp_val] = year_data

        write_split_by_combo_paged(combo_sp_data, page_start, page_end, prefix)

    write_paged_index(all_combo_attrs, dim_names, manifest, prefix, chunk_num=chunk_num)


# The inputs whose layers share one colour range over every land use and both irrigation states: runoff (ML/ha) and
# its change with the climate, and the climate multiplier on yield (a ratio). The others keep a range per land use:
# their values differ by orders of magnitude between land uses (cost, revenue, emissions, water requirement).
INPUTS_ONE_RANGE = ['Runoff', 'Climate change impact on runoff', 'Climate change']


def get_input_layer_ranges(files_input: pd.DataFrame) -> dict:
    """The colour range of every input layer (xr_input_layers_<year>.nc), shared by its years so they compare: the
    0.5 % and 99.5 % quantiles of its non-zero cells, opened to include zero. A range per (input, lm, lu): one
    range for a whole input would be set by its dearest land use (horticulture) and leave the others pale. The parts
    of an input share the range of the whole, and the inputs of INPUTS_ONE_RANGE have one range for all their layers.
    {(input, lm, lu): (low, high)}"""
    samples, layer_keys, names = {}, set(), ['input', 'lm', 'option', 'lu']
    for path in files_input['path']:
        with xr.open_dataset(path) as dataset:
            layers = cfxr.decode_compress_to_multi_index(dataset, 'layer')['data'].load()
        names = list(layers['layer'].to_index().names)
        for position, key in enumerate(layers['layer'].to_index()):
            values = layers.values[position]
            label = dict(zip(names, key))
            layer_keys.add(tuple(key))
            samples.setdefault((label['input'], label['lm'], label['lu']), []).append(values[np.isfinite(values) & (values != 0)])
    def quantile_range(parts):
        values = np.concatenate(parts)
        low, high = (np.quantile(values, [0.005, 0.995]) if values.size else (0.0, 0.0))
        return (min(float(low), 0.0), max(float(high), 0.0))

    ranges = {key: quantile_range(parts) for key, parts in samples.items()}

    # An input whose land uses are the same quantity on the same scale (INPUTS_ONE_RANGE) takes ONE range, from all
    # its layers together, so its land uses and its dryland / irrigated maps compare.
    for name in INPUTS_ONE_RANGE:
        keys = [key for key in samples if key[0] == name]
        if keys:
            shared = quantile_range([part for key in keys for part in samples[key]])
            ranges.update({key: shared for key in keys})

    # A part of an input ('Agricultural cost: quantity' of 'Agricultural cost') takes the range of the whole for the
    # same (lm, lu), so the maps of the parts and of their total read on one colour bar and compare.
    inputs = {key[0] for key in ranges}
    for (name, lm, lu) in list(ranges):
        whole = name.split(': ')[0]
        if whole != name and whole in inputs and (whole, lm, lu) in ranges:
            ranges[(name, lm, lu)] = ranges[(whole, lm, lu)]

    # Every layer by its own labels, in the order of its index: the options of a layer share its range, so they compare
    return {key: ranges[(label['input'], label['lm'], label['lu'])]
            for key in layer_keys for label in [dict(zip(names, key))]}


def save_input_layers(raw_dir: str, save_dir: str):
    """The input layers of the input cache (write_input.build_input_cache), from the files of `raw_dir`
    (out_<year>/xr_input_layers_<theme>_<year>.nc) into `save_dir`/map_layers: input -> lm -> option -> lu, on the
    grid of the cache (map_input_cell_index.js). The values are intensive (per ha, %, 0 / 1), so they are stored as
    they are; a layer that is the same in every year is stored once."""
    files = get_all_files(raw_dir).query('category == "xarray_layer"')
    files['Year'] = files['Year'].astype(int)

    first_template = files.query('base_name == "xr_map_template"').sort_values('Year').iloc[0]
    index_merc, geo_meta_merc, _ = build_year_meta(int(first_template['Year']), Path(first_template['path']).parent)
    write_cell_index(index_merc, geo_meta_merc, f'{save_dir}/map_layers', name='map_input_cell_index')

    # One theme at a time: with every option a theme holds thousands of layers
    files_input = files.query('base_name.str.startswith("xr_input_layers")')
    input_min_max = get_input_layer_ranges(files_input)
    get_map2json(files_input, None, input_min_max, f'{save_dir}/map_layers/map_in.js', per_ha=False, collapse_years=True)   # a short name: it starts every layer's file name
    print('Input layers saved.', flush=True)


def save_report_layer(raw_data_dir: str):

    SAVE_DIR = f'{raw_data_dir}/DATA_REPORT/data'

    # Load the cell magnitudes for all layers, which will be used to set colorbar limits in the report
    with open(f'{raw_data_dir}/max_cell_magnitudes.json', 'r') as f:
        cell_magnitudes = json.load(f)

    # Create the directory if it does not exist
    if not Path(f'{SAVE_DIR}/map_layers').exists():
        Path(f'{SAVE_DIR}/map_layers').mkdir(parents=True, exist_ok=True)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir).query('category == "xarray_layer"')
    files['Year'] = files['Year'].astype(int)
    files = files.query(f'Year.isin({sorted(settings.SIM_YEARS)})')
    
    


    ####################################################
    #                    0) Legend                     #
    ####################################################

    legend_registry = {
        'ag':     build_map_legend(COLOR_AG),
        'am':     build_map_legend(COLOR_AM, RENAME_AM_NON_AG),
        'non_ag': build_map_legend(COLOR_NON_AG, RENAME_AM_NON_AG),
        'lumap':  {**build_map_legend(COLOR_LUMAP, RENAME_AM_NON_AG), 'columns': get_lumap_legend_columns()},
    }
    # The land-use legend in the order of its columns (build_map_legend lists the entries by map code)
    lumap_order = [entry['name'] for column in legend_registry['lumap']['columns'].values() for entry in column]
    lumap_legend = legend_registry['lumap']['legend']
    legend_registry['lumap']['legend'] = {name: lumap_legend[name] for name in lumap_order if name in lumap_legend}
    
    with open(f'{SAVE_DIR}/map_layers/legend_registry.js', 'w') as _f:
        _f.write('window["legend_registry"] = ')
        json.dump(legend_registry, _f, separators=(',', ':'))
        _f.write(';\n')
        
    print('│   ├── Legend registry saved.')


    ####################################################
    #     0b) The cell-number raster of the report     #
    ####################################################
    # Every layer is a 1D vector of cells (map2cells); the browser paints it through this one raster.
    first_template = files.query('base_name == "xr_map_template"').sort_values('Year').iloc[0]
    index_merc, geo_meta_merc, _ = build_year_meta(int(first_template['Year']), Path(first_template['path']).parent)
    write_cell_index(index_merc, geo_meta_merc, f'{SAVE_DIR}/map_layers')
    print('│   ├── Cell index raster saved.')
    

    ####################################################
    #                   1) LUMAP Layer                 #
    ####################################################
    lumap_ly = files.query('base_name == "xr_map_lumap"')
    get_map2json(lumap_ly, 'lm', None, f'{SAVE_DIR}/map_layers/map_dvar_lumap.js', legend_key_int='lumap')
    print('│   ├── LUMAP layer saved.')


 
    ####################################################
    #                   2) Dvar Layer                  #
    ####################################################
    
    # A dvar layer is the share of the cell under a land use (0-1): an intensive value, stored as it is
    # (per_ha=False). Divided by the cell's area, as an extensive layer is, its colour-bar labels rounded to 0 - 0.
    dvar_min_max = (0, 1)

    dvar_ag = files.query('base_name == "xr_dvar_ag"')
    get_map2json(dvar_ag, {'lu':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_Ag.js', legend_key_int='ag', per_ha=False)
    print('│   ├── Dvar Ag layer saved.')

    dvar_am = files.query('base_name == "xr_dvar_am"')
    get_map2json(dvar_am, {'am':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_Am.js', legend_key_int='am', per_ha=False)
    print('│   ├── Dvar Am layer saved.')

    dvar_nonag = files.query('base_name == "xr_dvar_non_ag"')
    get_map2json(dvar_nonag, {'lu':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_NonAg.js', legend_key_int='non_ag', per_ha=False)
    print('│   ├── Dvar Non-Ag layer saved.')

    get_lumap_group_layers(dvar_ag, dvar_nonag, f'{SAVE_DIR}/map_layers/map_dvar_lumap_share.js')
    print('│   ├── Land-use share layers (one per legend entry of the land-use map) saved.')
    
    
    
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
    
    
    # Per-backend scale for the quality maps: RHI carries native 0-100 units while every other
    # quality backend runs 0-1, so ONE shared (min, max) washes every non-RHI backend into the
    # bottom of the colour ramp. Quantiles come from the 'all' files so each backend keeps a
    # single scale across the Ag / NonAg / Am / All views.
    bio_backend_min_max = {}
    for _p in files_bio.query('base_name == "xr_biodiversity_overall_priority_all"')['path']:
        _arr = cfxr.decode_compress_to_multi_index(xr.open_dataset(_p, chunks={}), 'layer')['data']
        for _backend in _arr['layer'].to_index().unique(level='backend'):
            _vals = np.asarray(_arr.sel(backend=_backend).values).ravel()
            _vals = _vals[(_vals != 0) & np.isfinite(_vals)]
            if _vals.size:
                _lo, _hi = float(np.nanquantile(_vals, 0.005)), float(np.nanquantile(_vals, 0.995))
                _prev = bio_backend_min_max.get(str(_backend))
                bio_backend_min_max[str(_backend)] = ((min(_lo, _prev[0]), max(_hi, _prev[1]))
                                                      if _prev else (_lo, _hi))
        _arr.close()
    bio_backend_min_max = bio_backend_min_max or bio_min_max

    # Quality layers
    bio_overall_ag = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag"')
    get_map2json(bio_overall_ag, None, bio_backend_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_Ag.js')
    print('│   ├── Biodiversity Overall Ag layer saved.')

    bio_overall_nonag = files_bio.query('base_name == "xr_biodiversity_overall_priority_non_ag"')
    get_map2json(bio_overall_nonag, None, bio_backend_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_NonAg.js')
    print('│   ├── Biodiversity Overall Non-Ag layer saved.')

    bio_overall_am = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag_management"')
    get_map2json(bio_overall_am, None, bio_backend_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_Am.js')
    print('│   ├── Biodiversity Overall Am layer saved.')

    bio_overall_all = files_bio.query('base_name == "xr_biodiversity_overall_priority_all"')
    get_map2json(bio_overall_all, None, bio_backend_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_All.js')
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
    # The range is shared per commodity across Ag / Ag Mgt / Non-Ag; only Ag Mgt (a change in
    # production) can go below 0, so the other layers start their colour bar at 0.
    prod_min_max_nonneg = {k: (max(0, lo), hi) for k, (lo, hi) in prod_min_max.items()}

    quantities_ag = files_quantities.query('base_name == "xr_quantities_agricultural"')
    get_map2json(quantities_ag, {'Commodity':'ALL'}, prod_min_max_nonneg, f'{SAVE_DIR}/map_layers/map_quantities_Ag.js', legend_key_int='ag')
    print('│   ├── Quantities Ag layer saved.')

    quantities_am = files_quantities.query('base_name == "xr_quantities_agricultural_management"')
    get_map2json(quantities_am, {'Commodity':'ALL'}, prod_min_max, f'{SAVE_DIR}/map_layers/map_quantities_Am.js', legend_key_int='am')
    print('│   ├── Quantities Am layer saved.')

    quantities_nonag = files_quantities.query('base_name == "xr_quantities_non_agricultural"')
    get_map2json(quantities_nonag, {'Commodity':'ALL'}, prod_min_max_nonneg, f'{SAVE_DIR}/map_layers/map_quantities_NonAg.js', legend_key_int='non_ag')
    print('│   ├── Quantities Non-Ag layer saved.')

    quantities_sum = files_quantities.query('base_name == "xr_quantities_sum"')
    get_map2json(quantities_sum, {'Commodity':'ALL'}, prod_min_max_nonneg, f'{SAVE_DIR}/map_layers/map_quantities_Sum.js', legend_key_int='ag')
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



    ####################################################
    #        11) Input layers (the Home page)          #
    ####################################################
    # Not built here: the inputs are no product of the run. They come from the input cache, which
    # create_report_data.copy_input_cache copies into the report (write_input.ensure_input_cache).
