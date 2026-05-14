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
import json
import base64
import hashlib
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



# 2/3 of the NTFS per-component filename limit (255 chars).
# Any generated .tif filename longer than this is hashed and recorded in nameWarp.js.
FILENAME_WRAP_THRESHOLD = 170


def safe_key(s: str) -> str:
    """Convert arbitrary string to a safe filename/JS-variable suffix."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', str(s)).strip('_')


def _build_tree(combos: list) -> list | dict:
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
    return {k: _build_tree(groups[k]) for k in sorted_keys}



def _write_geotiff_index(flat_data: dict, dim_names: list, prefix: str) -> None:
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

    tree = _build_tree(list(combo_map.keys()))
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


def _write_split_by_combo_geotiff(flat_data: dict, dim_names: list, prefix: str, name_warp: dict) -> None:
    """Write one JS file per unique combo (all years) and an index JS file.

    Each per-combo JS file contains the base64-encoded GeoTIFF and display attrs
    for every year, following the same split-file pattern the Vue already uses:
      window["map_area_Ag__Dryland__Wheat"] = {
          2010: {tif_b64: "...", intOrFloat: "float", min_max: [0, 500]},
          2025: {tif_b64: "...", ...},
      };
    Long filenames (JS or TIF) are hashed and recorded in name_warp.
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
        if len(js_filename) > FILENAME_WRAP_THRESHOLD:
            hash12 = hashlib.md5(js_filename.encode()).hexdigest()[:12]
            wrapped = f'{base_name}__{hash12}.js'
            name_warp[wrapped] = js_filename
            js_filename = wrapped
        var_name = js_filename.removesuffix('.js')
        with open(os.path.join(tif_dir, js_filename), 'w') as f:
            f.write(f'window["{var_name}"] = ')
            json.dump(year_data, f, separators=(',', ':'))
            f.write(';\n')

    _write_geotiff_index(flat_data, dim_names, prefix)



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
    
    

def get_map2json(
    files_df: pd.DataFrame,
    legend_int_level: dict | str,
    float_magnitude: tuple | dict,
    save_path: str,
    legend_key_int: str | None = None,
    name_warp: dict | None = None,
) -> None:
    """Convert all layers in *files_df* to GeoTIFF files (EPSG:3857).

    For each unique (combo × year) one ``<prefix>__<combo>__<year>.tif`` file is
    written to the same directory as *save_path*.  Filenames that exceed
    FILENAME_WRAP_THRESHOLD chars are hashed and recorded in the module-level
    ``_name_warp`` dict; ``save_report_layer`` writes that dict to nameWarp.js.
    A companion ``<prefix>__index.js`` records the dimension tree and per-combo
    display attrs (intOrFloat, min_max / legendKey) for the Vue UI.
    """
    # Per-worker memory: float32 2D output grid (EPSG:3857 is ~7× the 1D cell count).
    ncells = 5_000_000 // (settings.RESFACTOR ** 2)
    mem_per_worker = ncells * 7 * 4 / 1e9
    workers = min(60, max(1, int(settings.WRITE_REPORT_MAX_MEM_GB // mem_per_worker)))

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

    flat_data: dict = {}   # { hierarchy_tp: {tif_b64, intOrFloat, min_max/legendKey} }

    for _, row in files_df.iterrows():
        xr_arr = cfxr.decode_compress_to_multi_index(xr.open_dataset(row['path'], chunks={}), 'layer')['data']
        ds_template = f'{os.path.dirname(row["path"])}/xr_map_template_{row["Year"]}.nc'
        valid_layers = xr_arr['layer'].to_index().to_frame().to_dict(orient='records')

        if len(valid_layers) == 0:
            print(f'│   ├── No valid layers found in {row["base_name"]}_{row["Year"]}, skipping.')
            continue
        
        # dim_names is the same for all layers in the file, so get it once here for the index.js file.
        dim_names = list(rename_reorder_hierarchy(valid_layers[0]).keys())

        # Build EPSG:3857 cell-index map once per year.
        # index_merc[h,w] = 1D LUTO cell index, or -1 for nodata.
        # joblib memmaps this array across workers (max_nbytes=1e6).
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

        # Load real cell area (ha) for per-ha normalisation of float layers.
        real_area_path = os.path.join(
            os.path.dirname(row['path']),
            f'xr_area_real_area_ha_{row["Year"]}.nc'
        )
        if os.path.exists(real_area_path):
            with xr.open_dataset(real_area_path) as ra_ds:
                real_area_1d = ra_ds['data'].values.squeeze().astype(np.float32)
        else:
            real_area_1d = None
        geo_meta_merc = {
            'driver': 'GTiff', 'count': 1,
            'crs':       rxr_merc.rio.crs,
            'transform': rxr_merc.rio.transform(),
            'width':     int(rxr_merc.shape[-1]),
            'height':    int(rxr_merc.shape[-2]),
            'nodata':    -9999.0, 'compress': 'deflate', 'predictor': 2,
        } 

        # GBF NRM-mode layers carry an is_selected coord marking which cells belong to the
        # selected NRM region; prebuilt once per file rather than reconstructed per layer.
        is_selected_arr    = (
            np.asarray(xr_arr['is_selected'].values, dtype=bool)
            if 'is_selected' in xr_arr.coords else None
        )
        is_selected_coords = (
            {'is_selected': ('cell', is_selected_arr)} if is_selected_arr is not None else None
        )

        # Batch to avoid OOM on large layer counts (e.g. GBF4 ECNES ~2700 layers).
        for batch in (valid_layers[i:i+workers] for i in range(0, len(valid_layers), workers)):
            tasks = []
            for sel in batch:
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

            for hierarchy_tp, tif_bytes, attrs in Parallel(
                n_jobs=workers, return_as='generator', max_nbytes=1_000_000
            )(tasks):
                flat_data[hierarchy_tp] = {'tif_b64': base64.b64encode(tif_bytes).decode(), **attrs}

        xr_arr.close()

    if not flat_data:
        return
    _write_split_by_combo_geotiff(flat_data, dim_names, prefix, name_warp)
        


def save_report_layer(raw_data_dir: str):
    name_warp: dict[str, str] = {}

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
    #               0) Legend and Name Warp            #
    ####################################################
    # Get legend info (used only for legend_registry.js)

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
    get_map2json(lumap_ly, 'lm', None, f'{SAVE_DIR}/map_layers/map_dvar_lumap.js', legend_key_int='lumap', name_warp=name_warp)
    print('│   ├── LUMAP layer saved.')


 
    ####################################################
    #                   2) Dvar Layer                  #
    ####################################################
    
    dvar_min_max = (0, 1)

    dvar_ag = files.query('base_name == "xr_dvar_ag"')
    get_map2json(dvar_ag, {'lu':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_Ag.js', legend_key_int='ag', name_warp=name_warp)
    print('│   ├── Dvar Ag layer saved.')
    
    files_df,legend_int_level,float_magnitude,save_path,legend_key_int,name_warp = (
        dvar_ag, {'lu':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_Ag.js', 'ag', name_warp
    )

    dvar_am = files.query('base_name == "xr_dvar_am"')
    get_map2json(dvar_am, {'am':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_Am.js', legend_key_int='am', name_warp=name_warp)
    print('│   ├── Dvar Am layer saved.')

    dvar_nonag = files.query('base_name == "xr_dvar_non_ag"')
    get_map2json(dvar_nonag, {'lu':'ALL'}, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_NonAg.js', legend_key_int='non_ag', name_warp=name_warp)
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
    get_map2json(area_ag, {'lu':'ALL'}, area_min_max, f'{SAVE_DIR}/map_layers/map_area_Ag.js', legend_key_int='ag', name_warp=name_warp)
    print('│   ├── Area Ag layer saved.')
    
    # files_df,legend_int,legend_int_level,legend_float,float_magnitude,save_path = area_ag, legend_ag, {'lu':'ALL'}, legend_float, area_min_max, f'{SAVE_DIR}/map_layers/map_area_Ag.js'
    
    area_nonag = files_area.query('base_name == "xr_area_non_agricultural_landuse"')
    get_map2json(area_nonag, {'lu':'ALL'}, area_min_max, f'{SAVE_DIR}/map_layers/map_area_NonAg.js', legend_key_int='non_ag', name_warp=name_warp)
    print('│   ├── Area Non-Ag layer saved.')

    area_am = files_area.query('base_name == "xr_area_agricultural_management"')
    get_map2json(area_am, {'am':'ALL'}, area_min_max, f'{SAVE_DIR}/map_layers/map_area_Am.js', legend_key_int='am', name_warp=name_warp)
    print('│   ├── Area Am layer saved.')


    ####################################################
    #                  4) Biodiversity                 #
    ####################################################

    files_bio = files.query('base_name.str.contains("biodiversity")')
    
    bio_magnitudes = (
        *cell_magnitudes['bio_quality']['ag'],
        *cell_magnitudes['bio_quality']['non_ag'],
        *cell_magnitudes['bio_quality']['am'],
        *cell_magnitudes['bio_quality']['all']
    )
    
    bio_min_max = (min(bio_magnitudes), max(bio_magnitudes))
    
    
    # Quality layers
    bio_overall_ag = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag"')
    get_map2json(bio_overall_ag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_Ag.js', name_warp=name_warp)
    print('│   ├── Biodiversity Overall Ag layer saved.')

    bio_overall_nonag = files_bio.query('base_name == "xr_biodiversity_overall_priority_non_ag"')
    get_map2json(bio_overall_nonag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_NonAg.js', name_warp=name_warp)
    print('│   ├── Biodiversity Overall Non-Ag layer saved.')

    bio_overall_am = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag_management"')
    get_map2json(bio_overall_am, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_Am.js', name_warp=name_warp)
    print('│   ├── Biodiversity Overall Am layer saved.')

    bio_overall_all = files_bio.query('base_name == "xr_biodiversity_overall_priority_all"')
    get_map2json(bio_overall_all, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_All.js', name_warp=name_warp)
    print('│   ├── Biodiversity Overall All layer saved.')


    # GBF2
    if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':
        bio_GBF2_ag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag"')
        get_map2json(bio_GBF2_ag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Ag.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF2 Ag layer saved.')

        bio_GBF2_am = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag_management"')
        get_map2json(bio_GBF2_am, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Am.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF2 Am layer saved.')

        bio_GBF2_nonag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_non_ag"')
        get_map2json(bio_GBF2_nonag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_NonAg.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF2 Non-Ag layer saved.')

        bio_GBF2_sum = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_sum"')
        get_map2json(bio_GBF2_sum, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Sum.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF2 Sum layer saved.')

    # TODO: need to create cell magnitude and determine colorbar limit for these GBF layers.
    # Now just use the quality magnitude, which could be incorrect for specific GBFs.

    # GBF3-NVIS (NRM aggregation mode)
    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS != 'off' and settings.GBF3_NVIS_REGION_MODE != 'IBRA':
        bio_GBF3_NVIS_ag = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_ag"')
        get_map2json(bio_GBF3_NVIS_ag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_Ag.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF3_NVIS Ag layer saved.')

        bio_GBF3_NVIS_am = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_ag_management"')
        get_map2json(bio_GBF3_NVIS_am, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_Am.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF3_NVIS Am layer saved.')

        bio_GBF3_NVIS_nonag = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_non_ag"')
        get_map2json(bio_GBF3_NVIS_nonag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_NonAg.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF3_NVIS Non-Ag layer saved.')

        bio_GBF3_NVIS_sum = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_sum"')
        get_map2json(bio_GBF3_NVIS_sum, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_Sum.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF3_NVIS Sum layer saved.')

    # GBF3-IBRA aggregation mode — writes to GBF3_NVIS map filenames
    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS != 'off' and settings.GBF3_NVIS_REGION_MODE == 'IBRA':
        bio_GBF3_IBRA_ag = files_bio.query('base_name == "xr_biodiversity_GBF3_IBRA_ag"')
        get_map2json(bio_GBF3_IBRA_ag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_Ag.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF3_IBRA Ag layer saved.')

        bio_GBF3_IBRA_am = files_bio.query('base_name == "xr_biodiversity_GBF3_IBRA_ag_management"')
        get_map2json(bio_GBF3_IBRA_am, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_Am.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF3_IBRA Am layer saved.')

        bio_GBF3_IBRA_nonag = files_bio.query('base_name == "xr_biodiversity_GBF3_IBRA_non_ag"')
        get_map2json(bio_GBF3_IBRA_nonag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_NonAg.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF3_IBRA Non-Ag layer saved.')

    # GBF4-SNES
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES != 'off':
        bio_GBF4_SNES_ag = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_ag"')
        get_map2json(bio_GBF4_SNES_ag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_Ag.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF4_SNES Ag layer saved.')

        bio_GBF4_SNES_am = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_ag_management"')
        get_map2json(bio_GBF4_SNES_am, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_Am.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF4_SNES Am layer saved.')

        bio_GBF4_SNES_nonag = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_non_ag"')
        get_map2json(bio_GBF4_SNES_nonag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_NonAg.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF4_SNES Non-Ag layer saved.')

        bio_GBF4_SNES_sum = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_sum"')
        get_map2json(bio_GBF4_SNES_sum, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_Sum.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF4_SNES Sum layer saved.')

    # GBF4_ECNES
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES != 'off':
        bio_GBF4_ECNES_ag = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_ag"')
        get_map2json(bio_GBF4_ECNES_ag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_Ag.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF4_ECNES Ag layer saved.')

        bio_GBF4_ECNES_am = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_ag_management"')
        get_map2json(bio_GBF4_ECNES_am, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_Am.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF4_ECNES Am layer saved.')

        bio_GBF4_ECNES_nonag = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_non_ag"')
        get_map2json(bio_GBF4_ECNES_nonag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_NonAg.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF4_ECNES Non-Ag layer saved.')

        bio_GBF4_ECNES_sum = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_sum"')
        get_map2json(bio_GBF4_ECNES_sum, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_Sum.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF4_ECNES Sum layer saved.')

    # GBF8
    if settings.BIODIVERSITY_TARGET_GBF_8 != 'off':
        bio_GBF8_ag = files_bio.query('base_name == "xr_biodiversity_GBF8_species_ag"')
        get_map2json(bio_GBF8_ag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_Ag.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF8 Ag layer saved.')

        bio_GBF8_am = files_bio.query('base_name == "xr_biodiversity_GBF8_species_ag_management"')
        get_map2json(bio_GBF8_am, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_Am.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF8 Am layer saved.')

        bio_GBF8_nonag = files_bio.query('base_name == "xr_biodiversity_GBF8_species_non_ag"')
        get_map2json(bio_GBF8_nonag, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_NonAg.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF8 Non-Ag layer saved.')

        bio_GBF8_ag_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_ag"')
        get_map2json(bio_GBF8_ag_group, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_Ag.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF8 Groups Ag layer saved.')

        bio_GBF8_am_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_ag_management"')
        get_map2json(bio_GBF8_am_group, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_Am.js', name_warp=name_warp)
        print('│   ├── Biodiversity GBF8 Groups Am layer saved.')

        bio_GBF8_nonag_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_non_ag"')
        get_map2json(bio_GBF8_nonag_group, None, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_NonAg.js', name_warp=name_warp)
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
    get_map2json(profit_ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_profit.js', name_warp=name_warp)
    print('│   ├── Economics Ag profit layer saved.')

    profit_am = files.query('base_name == "xr_economics_am_profit"')
    get_map2json(profit_am, None, economic_min_max_am, f'{SAVE_DIR}/map_layers/map_economics_Am_profit.js', name_warp=name_warp)
    print('│   ├── Economics Am profit layer saved.')

    profit_nonag = files.query('base_name == "xr_economics_non_ag_profit"')
    get_map2json(profit_nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_profit.js', name_warp=name_warp)
    print('│   ├── Economics NonAg profit layer saved.')

    # Sum profit (Ag + Am + NonAg)
    economic_magnitudes_sum = cell_magnitudes.get('Economics_sum', {}).get('sum_profit', [0.0, 0.0])
    economic_min_max_sum = (min(economic_magnitudes_sum), max(economic_magnitudes_sum))
    profit_sum = files.query('base_name == "xr_economics_sum_profit"')
    get_map2json(profit_sum, None, economic_min_max_sum, f'{SAVE_DIR}/map_layers/map_economics_Sum_profit.js', name_warp=name_warp)
    print('│   ├── Economics Sum profit layer saved.')


    # ---------------- Revenue ----------------
    revenue_ag = files.query('base_name == "xr_economics_ag_revenue"')
    get_map2json(revenue_ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_revenue.js', name_warp=name_warp)
    print('│   ├── Economics Ag revenue layer saved.')

    revenue_am = files.query('base_name == "xr_economics_am_revenue"')
    get_map2json(revenue_am, None, economic_min_max_am, f'{SAVE_DIR}/map_layers/map_economics_Am_revenue.js', name_warp=name_warp)
    print('│   ├── Economics Am revenue layer saved.')
    
    revenue_nonag = files.query('base_name == "xr_economics_non_ag_revenue"')
    get_map2json(revenue_nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_revenue.js', name_warp=name_warp)
    print('│   ├── Economics NonAg revenue layer saved.')


    # ---------------- Cost ----------------
    cost_ag = files.query('base_name == "xr_economics_ag_cost"')
    get_map2json(cost_ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_cost.js', name_warp=name_warp)
    print('│   ├── Economics Ag cost layer saved.')

    cost_am = files.query('base_name == "xr_economics_am_cost"')
    get_map2json(cost_am, None, economic_min_max_am, f'{SAVE_DIR}/map_layers/map_economics_Am_cost.js', name_warp=name_warp)
    print('│   ├── Economics Am cost layer saved.')

    cost_nonag = files.query('base_name == "xr_economics_non_ag_cost"')
    get_map2json(cost_nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_cost.js', name_warp=name_warp)
    print('│   ├── Economics NonAg cost layer saved.')


    # ---------------- Transition Cost ----------------
    cost_trans_ag2ag = files.query('base_name == "xr_economics_ag_transition_Ag2Ag"')
    get_map2json(cost_trans_ag2ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_transition_ag2ag.js', name_warp=name_warp)
    print('│   ├── Economics Ag transition ag2ag layer saved.')

    cost_trans_nonag2ag = files.query('base_name == "xr_economics_ag_transition_NonAg2Ag"')
    get_map2json(cost_trans_nonag2ag, None, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_transition_nonag2ag.js', name_warp=name_warp)
    print('│   ├── Economics Ag transition nonag2ag layer saved.')

    cost_trans_ag2nonag = files.query('base_name == "xr_economics_non_ag_transition_Ag2NonAg"')
    get_map2json(cost_trans_ag2nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_transition_ag2non_ag.js', name_warp=name_warp)
    print('│   ├── Economics NonAg transition ag2nonag layer saved.')

    cost_trans_nonag2nonag = files.query('base_name == "xr_economics_non_ag_transition_NonAg2NonAg"')
    get_map2json(cost_trans_nonag2nonag, None, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_transition_nonag2nonag.js', name_warp=name_warp)
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
    get_map2json(ghg_ag, None, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_Ag.js', name_warp=name_warp)
    print('│   ├── GHG Ag layer saved.')

    ghg_am = files_ghg.query('base_name == "xr_GHG_ag_management"')
    get_map2json(ghg_am, None, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_Am.js', name_warp=name_warp)
    print('│   ├── GHG Am layer saved.')

    ghg_nonag = files_ghg.query('base_name == "xr_GHG_non_ag"')
    get_map2json(ghg_nonag, None, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_NonAg.js', name_warp=name_warp)
    print('│   ├── GHG Non-Ag layer saved.')

    ghg_sum = files_ghg.query('base_name == "xr_GHG_sum"')
    get_map2json(ghg_sum, None, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_Sum.js', name_warp=name_warp)
    print('│   ├── GHG Sum layer saved.')



    ####################################################
    #                  7) Quantities                   #
    ####################################################

    files_quantities = files.query('base_name.str.contains("quantities")')
    
    prod_magnitudes = cell_magnitudes['production']
    prod_min_max = {k: (min(v), max(v)) for k, v in prod_magnitudes.items()}

    quantities_ag = files_quantities.query('base_name == "xr_quantities_agricultural"')
    get_map2json(quantities_ag, {'Commodity':'ALL'}, prod_min_max, f'{SAVE_DIR}/map_layers/map_quantities_Ag.js', legend_key_int='ag', name_warp=name_warp)
    print('│   ├── Quantities Ag layer saved.')

    quantities_am = files_quantities.query('base_name == "xr_quantities_agricultural_management"')
    get_map2json(quantities_am, {'Commodity':'ALL'}, prod_min_max, f'{SAVE_DIR}/map_layers/map_quantities_Am.js', legend_key_int='am', name_warp=name_warp)
    print('│   ├── Quantities Am layer saved.')

    quantities_nonag = files_quantities.query('base_name == "xr_quantities_non_agricultural"')
    get_map2json(quantities_nonag, {'Commodity':'ALL'}, prod_min_max, f'{SAVE_DIR}/map_layers/map_quantities_NonAg.js', legend_key_int='non_ag', name_warp=name_warp)
    print('│   ├── Quantities Non-Ag layer saved.')

    quantities_sum = files_quantities.query('base_name == "xr_quantities_sum"')
    get_map2json(quantities_sum, {'Commodity':'ALL'}, prod_min_max, f'{SAVE_DIR}/map_layers/map_quantities_Sum.js', legend_key_int='ag', name_warp=name_warp)
    print('│   ├── Quantities Sum layer saved.')



    ####################################################
    #                8) Transitions                    #
    ####################################################

    # Transitions Area (Ag2Ag)
    files_transition_area_ag2ag = files.query('base_name == "xr_transition_ag2ag_area"')
    trans_area_ag2ag_magnitudes = cell_magnitudes['transition_area']['ag2ag']
    trans_area_ag2ag_min_max = (min(trans_area_ag2ag_magnitudes), max(trans_area_ag2ag_magnitudes))
    get_map2json(files_transition_area_ag2ag, None, trans_area_ag2ag_min_max, f'{SAVE_DIR}/map_layers/map_transition_area_ag2ag.js', name_warp=name_warp)
    print('│   ├── Transition Area (Ag2Ag) layer saved.')


    # Transitions Area (Ag2NonAg)
    files_transition_area_ag2nonag = files.query('base_name == "xr_transition_ag2nonag_area"')
    trans_area_ag2non_ag_magnitudes = cell_magnitudes['transition_area']['ag2non_ag']
    trans_area_ag2non_ag_min_max = (min(trans_area_ag2non_ag_magnitudes), max(trans_area_ag2non_ag_magnitudes))
    get_map2json(files_transition_area_ag2nonag, None, trans_area_ag2non_ag_min_max, f'{SAVE_DIR}/map_layers/map_transition_area_ag2nonag.js', name_warp=name_warp)
    print('│   ├── Transition Area (Ag2NonAg) layer saved.')


    # Transitions Cost (Ag2Ag)
    files_transition_cost_ag2ag = files.query('base_name == "xr_transition_ag2ag_cost"')
    trans_cost_ag2ag_magnitudes = cell_magnitudes['Economics_ag']['ag2ag_cost']
    trans_cost_ag2ag_min_max = (min(trans_cost_ag2ag_magnitudes), max(trans_cost_ag2ag_magnitudes))
    get_map2json(files_transition_cost_ag2ag, None, trans_cost_ag2ag_min_max, f'{SAVE_DIR}/map_layers/map_transition_cost_ag2ag.js', name_warp=name_warp)
    print('│   ├── Transition Cost (Ag2Ag) layer saved.')


    # Transitions Cost (Ag2NonAg)
    files_transition_cost_ag2nonag = files.query('base_name == "xr_transition_ag2nonag_cost"')
    trans_cost_ag2nonag_magnitudes = cell_magnitudes['Economics_non_ag']['ag2nonag_cost']
    trans_cost_ag2nonag_min_max = (min(trans_cost_ag2nonag_magnitudes), max(trans_cost_ag2nonag_magnitudes))
    get_map2json(files_transition_cost_ag2nonag, None, trans_cost_ag2nonag_min_max, f'{SAVE_DIR}/map_layers/map_transition_cost_ag2nonag.js', name_warp=name_warp)
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
    get_map2json(water_sum, None, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_Sum.js', name_warp=name_warp)
    print('│   ├── Water Yield Sum layer saved.')

    water_ag = files_water.query('base_name == "xr_water_yield_ag"')
    get_map2json(water_ag, None, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_Ag.js', name_warp=name_warp)
    print('│   ├── Water Yield Ag layer saved.')

    water_am = files_water.query('base_name == "xr_water_yield_ag_management"')
    get_map2json(water_am, None, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_Am.js', name_warp=name_warp)
    print('│   ├── Water Yield Am layer saved.')

    water_nonag = files_water.query('base_name == "xr_water_yield_non_ag"')
    get_map2json(water_nonag, None, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_NonAg.js', name_warp=name_warp)
    print('│   ├── Water Yield Non-Ag layer saved.')


    ####################################################
    #             10) Renewable Energy                 #
    ####################################################

    files_renewable = files.query('base_name == "xr_renewable_energy"')

    if not files_renewable.empty:
        re_magnitudes = cell_magnitudes.get('renewable_energy', [0.0, 0.0])
        re_min_max = (min(re_magnitudes), max(re_magnitudes))

        get_map2json(files_renewable, None, re_min_max, f'{SAVE_DIR}/map_layers/map_renewable_energy_Am.js', name_warp=name_warp)
        print('│   ├── Renewable Energy Am layer saved.')

    files_renewable_exist = files.query('base_name == "xr_renewable_existing_dvar"')

    if not files_renewable_exist.empty:
        exist_magnitudes = cell_magnitudes.get('renewable_existing_dvar', [0.0, 1.0])
        exist_min_max = (min(exist_magnitudes), max(exist_magnitudes))

        get_map2json(files_renewable_exist, None, exist_min_max, f'{SAVE_DIR}/map_layers/map_renewable_existing_dvar_Am.js', name_warp=name_warp)
        print('│   ├── Renewable Existing Dvar Am layer saved.')
        
        
        
        
    ####################################################
    #               11) Name Warp                      #
    ####################################################
    
    with open(f'{SAVE_DIR}/map_layers/nameWarp.js', 'w') as f:
        f.write('window["nameWarp"] = ')
        json.dump(name_warp, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
    print(f'│   └── nameWarp.js saved ({len(name_warp)} wrapped filename{"s" if len(name_warp) != 1 else ""}).')

    

