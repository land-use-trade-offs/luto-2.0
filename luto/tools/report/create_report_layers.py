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
import json
import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray as cfxr

from joblib import delayed, Parallel
from luto import settings
from luto.tools.report.data_tools import (
    array_to_base64,
    get_all_files,
    build_map_legend,
    hex_color_to_numeric,
    rename_reorder_hierarchy,
    tuple_dict_to_nested,
)

from luto.tools.report.data_tools.parameters import (
    COLOR_AG,
    COLOR_AM, 
    COLOR_NON_AG, 
    COLOR_LUMAP, 
    COLORS_FLOAT
)




def map2base64(
    ds_template:str, 
    arr_sel:xr.DataArray, 
    isInt:bool, 
    legend:dict, 
    hierarchy_tp:tuple,
    layer_magnitude:int,
) -> tuple:

    # Get an template rio-xarray, it will be used to convert 1D array to its 2D map format
    with xr.load_dataset(ds_template) as rxr_ds:
        rxr_arr = rxr_ds['layer'].astype('float32')         # Use float32 to allow NaN values
        rxr_crs = rxr_ds['spatial_ref'].attrs['crs_wkt']

    # Set legend and metadata based on type
    if isInt:
        # layer_magnitude is None in this case
        img_attrs = {
            'intOrFloat': 'int',
            'legend': legend['legend'],
        }
    else:
        # Map values to color codes: positives → red (51–100), negatives → blue (0–50), zeros → grey (101)
        global_min, global_max = layer_magnitude
        vals = arr_sel.values.copy()
        codes = np.full(vals.shape, 101, dtype=np.int8)  # default: no-data grey

        if global_max > 0:
            codes[vals > 0] = np.clip(51 + (vals[vals > 0] / global_max) * 49, 51, 100)
        if global_min < 0:
            codes[vals < 0] = np.clip(50 + (vals[vals < 0] / abs(global_min)) * 50, 0, 50)

        arr_sel.values = codes
        min_max = (global_min, global_max)

        arr_sel = arr_sel.astype(np.float32)
        img_attrs = {
            'intOrFloat': 'float',
            'legend': legend['legend'],
            'min_max': min_max,
        }

    # Convert the 1D array to a 2D array
    np.place(rxr_arr.data, rxr_arr.data >= 0, arr_sel.values)     # Negative values in the template are outside LUTO area
    rxr_arr = xr.where(rxr_arr < 0, np.nan, rxr_arr)              # Set negative values to NaN, which will be transparent in the final map
    # Get the bounds
    rxr_arr = rxr_arr.rio.write_crs(rxr_crs)
    bbox = rxr_arr.rio.bounds()
    bbox = [[bbox[1], bbox[0]],[bbox[3], bbox[2]]]
    # Reproject to Web Mercator EPSG:3857; because Leaflet uses EPSG:3857 to display maps
    rxr_arr = rxr_arr.rio.reproject('EPSG:3857')                # To Mercator with Nearest Neighbour
    rxr_arr = np.nan_to_num(rxr_arr, nan=-1).astype('int16')    # Use -1 to flag nodata pixels
    # Convert the 1D array to a RGBA array
    color_dict = legend['code_colors']  # {lu_code: (R,G,B,A)}, -1 → transparent
    # Render to 4-band RGBA array
    arr_4band = np.zeros((rxr_arr.shape[0], rxr_arr.shape[1], 4), dtype='uint8')
    for k, v in color_dict.items():
        arr_4band[rxr_arr == k] = v

    return hierarchy_tp, {**array_to_base64(arr_4band), 'bounds':bbox, **img_attrs}
    
    

def get_map2json(
    files_df:pd.DataFrame, 
    legend_int:dict, 
    legend_int_level:dict|str, 
    legend_float:dict, 
    float_magnitude:int,
    save_path:str, 
    ) -> None:
    
    # Determine number of workers based on max memory setting
    #    The MEM for RESFACTOR=13 is ~0.5 GB, so scale accordingly
    workers = (settings.WRITE_REPORT_MAX_MEM_GB) // (0.5 * (13/settings.RESFACTOR)**2)
    workers = max(4, int(workers))      # At least 4 workers
    workers = min(workers, 32)          # At most 32 workers
    
    # Loop through each year
    tasks = []
    for _,row in files_df.iterrows():
        
        xr_arr = cfxr.decode_compress_to_multi_index(xr.load_dataset(row['path'], chunks={}), 'layer')['data']
        ds_template = f'{os.path.dirname(row['path'])}/xr_map_template_{row['Year']}.nc'
        valid_layers = xr_arr['layer'].to_index().to_frame().to_dict(orient='records')
        
        for sel in valid_layers:
            
            # Select the array for this layer
            arr_sel = xr_arr.sel(**sel)
            sel_rename = rename_reorder_hierarchy(sel)
            hierarchy_tp = tuple(list(sel_rename.values()) + [row['Year']])

            # Determine if this layer should use integer legend
            if legend_int_level is None:
                isInt = False
            elif isinstance(legend_int_level, dict):
                isInt = legend_int_level.items() <= sel.items()
            elif isinstance(legend_int_level, str):
                isInt = legend_int_level in sel.keys()
            else:
                raise ValueError('legend_int_level must be None, a dict, or a str')

            # Set legend and metadata based on type
            if isInt:
                legend = legend_int
                layer_magnitude = None
            else:
                legend = legend_float
                layer_magnitude = float_magnitude
            
            tasks.append(
                delayed(map2base64)(ds_template, arr_sel, isInt, legend, hierarchy_tp, layer_magnitude)
            )    
            
    # Gather results and save to JSON
    output = {}
    for res in Parallel(n_jobs=workers, return_as='generator')(tasks):
        hierarchy_tp, val_dict = res
        output[hierarchy_tp] = val_dict
        
    # To nested dict
    output = tuple_dict_to_nested(output)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        filename = os.path.basename(save_path).replace('.js', '')
        f.write(f'window["{filename}"] = ')
        json.dump(output, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        


def save_report_layer(raw_data_dir:str):
    
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
    
    # Get legend info
    legend_ag = build_map_legend(COLOR_AG)
    legend_am = build_map_legend(COLOR_AM)
    legend_non_ag = build_map_legend(COLOR_NON_AG)
    legend_lumap =  build_map_legend(COLOR_LUMAP)
    legend_float = {
        'legend': COLORS_FLOAT,
        'code_colors': {code: hex_color_to_numeric(hex_c) for code, hex_c in COLORS_FLOAT.items()}
    }

    
    ####################################################
    #                   1) LUMAP Layer                 #
    ####################################################
    lumap_ly = files.query('base_name == "xr_map_lumap"')
    get_map2json(lumap_ly, legend_lumap, 'lm', None, None, f'{SAVE_DIR}/map_layers/map_dvar_lumap.js')
    print('│   ├── LUMAP layer saved.')


 
    ####################################################
    #                   2) Dvar Layer                  #
    ####################################################
    
    dvar_min_max = (0, 1)

    dvar_ag = files.query('base_name == "xr_dvar_ag"')
    get_map2json(dvar_ag, legend_ag, {'lu':'ALL'}, legend_float, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_Ag.js')
    print('│   ├── Dvar Ag layer saved.')

    dvar_am = files.query('base_name == "xr_dvar_am"')
    get_map2json(dvar_am, legend_am, {'am':'ALL'}, legend_float, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_Am.js')
    print('│   ├── Dvar Am layer saved.')

    dvar_nonag = files.query('base_name == "xr_dvar_non_ag"')
    get_map2json(dvar_nonag, legend_non_ag, {'lu':'ALL'}, legend_float, dvar_min_max, f'{SAVE_DIR}/map_layers/map_dvar_NonAg.js')
    print('│   ├── Dvar Non-Ag layer saved.')
    
    
    
    ####################################################
    #                   3) Area Layer                  #
    ####################################################

    files_area = files.query('base_name.str.contains("area")')
    
    area_magnitudes = (
        cell_magnitudes['area']['ag'], 
        cell_magnitudes['area']['non_ag'], 
        *cell_magnitudes['area']['am'].values()
    )
    
    area_min_max = (min(area_magnitudes), max(area_magnitudes))
    

    area_ag = files_area.query('base_name == "xr_area_agricultural_landuse"')
    get_map2json(area_ag, legend_ag, {'lu':'ALL'}, legend_float, area_min_max, f'{SAVE_DIR}/map_layers/map_area_Ag.js')
    print('│   ├── Area Ag layer saved.')
    
    # files_df,legend_int,legend_int_level,legend_float,float_magnitude,save_path = area_ag, legend_ag, {'lu':'ALL'}, legend_float, area_min_max, f'{SAVE_DIR}/map_layers/map_area_Ag.js'



    area_nonag = files_area.query('base_name == "xr_area_non_agricultural_landuse"')
    get_map2json(area_nonag, legend_non_ag, {'lu':'ALL'}, legend_float, area_min_max, f'{SAVE_DIR}/map_layers/map_area_NonAg.js')
    print('│   ├── Area Non-Ag layer saved.')

    area_am = files_area.query('base_name == "xr_area_agricultural_management"')
    get_map2json(area_am, legend_am, {'am':'ALL'}, legend_float, area_min_max, f'{SAVE_DIR}/map_layers/map_area_Am.js')
    print('│   ├── Area Am layer saved.')


    ####################################################
    #                  4) Biodiversity                 #
    ####################################################

    files_bio = files.query('base_name.str.contains("biodiversity")')
    
    bio_magnitudes = (
        cell_magnitudes['bio_quality']['ag'],
        cell_magnitudes['bio_quality']['non_ag'],
        *cell_magnitudes['bio_quality']['am'].values(),
        cell_magnitudes['bio_quality']['all']
    )
    
    bio_min_max = (min(bio_magnitudes), max(bio_magnitudes))
    
    
    # Quality layers
    bio_overall_ag = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag"')
    get_map2json(bio_overall_ag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_Ag.js')
    print('│   ├── Biodiversity Overall Ag layer saved.')

    bio_overall_nonag = files_bio.query('base_name == "xr_biodiversity_overall_priority_non_ag"')
    get_map2json(bio_overall_nonag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_NonAg.js')
    print('│   ├── Biodiversity Overall Non-Ag layer saved.')

    bio_overall_am = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag_management"')
    get_map2json(bio_overall_am, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_Am.js')
    print('│   ├── Biodiversity Overall Am layer saved.')

    bio_overall_all = files_bio.query('base_name == "xr_biodiversity_overall_priority_all"')
    get_map2json(bio_overall_all, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_overall_All.js')
    print('│   ├── Biodiversity Overall All layer saved.')


    # GBF2
    if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':
        bio_GBF2_ag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag"')
        get_map2json(bio_GBF2_ag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Ag.js')
        print('│   ├── Biodiversity GBF2 Ag layer saved.')

        bio_GBF2_am = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag_management"')
        get_map2json(bio_GBF2_am, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Am.js')
        print('│   ├── Biodiversity GBF2 Am layer saved.')

        bio_GBF2_nonag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_non_ag"')
        get_map2json(bio_GBF2_nonag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF2_NonAg.js')
        print('│   ├── Biodiversity GBF2 Non-Ag layer saved.')
        
    # TODO: need to create cell magnitude and determine colorbar limit for these GBF layers.
    # Now just use the quality magnitude, which could be incorrect for specific GBFs.

    # GBF3-NVIS
    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS != 'off':
        bio_GBF3_NVIS_ag = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_ag"')
        get_map2json(bio_GBF3_NVIS_ag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_Ag.js')
        print('│   ├── Biodiversity GBF3_NVIS Ag layer saved.')

        bio_GBF3_NVIS_am = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_ag_management"')
        get_map2json(bio_GBF3_NVIS_am, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_Am.js')
        print('│   ├── Biodiversity GBF3_NVIS Am layer saved.')

        bio_GBF3_NVIS_nonag = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_non_ag"')
        get_map2json(bio_GBF3_NVIS_nonag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_NonAg.js')
        print('│   ├── Biodiversity GBF3_NVIS Non-Ag layer saved.')

    # GBF3-IBRA
    if settings.BIODIVERSITY_TARGET_GBF_3_IBRA != 'off':
        bio_GBF3_IBRA_ag = files_bio.query('base_name == "xr_biodiversity_GBF3_IBRA_ag"')
        get_map2json(bio_GBF3_IBRA_ag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_IBRA_Ag.js')
        print('│   ├── Biodiversity GBF3_IBRA Ag layer saved.')

        bio_GBF3_IBRA_am = files_bio.query('base_name == "xr_biodiversity_GBF3_IBRA_ag_management"')
        get_map2json(bio_GBF3_IBRA_am, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_IBRA_Am.js')
        print('│   ├── Biodiversity GBF3_IBRA Am layer saved.')

        bio_GBF3_IBRA_nonag = files_bio.query('base_name == "xr_biodiversity_GBF3_IBRA_non_ag"')
        get_map2json(bio_GBF3_IBRA_nonag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF3_IBRA_NonAg.js')
        print('│   ├── Biodiversity GBF3_IBRA Non-Ag layer saved.')

    # GBF4-SNES
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES != 'off':
        bio_GBF4_SNES_ag = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_ag"')
        get_map2json(bio_GBF4_SNES_ag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_Ag.js')
        print('│   ├── Biodiversity GBF4_SNES Ag layer saved.')

        bio_GBF4_SNES_am = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_ag_management"')
        get_map2json(bio_GBF4_SNES_am, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_Am.js')
        print('│   ├── Biodiversity GBF4_SNES Am layer saved.')

        bio_GBF4_SNES_nonag = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_non_ag"')
        get_map2json(bio_GBF4_SNES_nonag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_NonAg.js')
        print('│   ├── Biodiversity GBF4_SNES Non-Ag layer saved.')

    # GBF4_ECNES
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES != 'off':
        bio_GBF4_ECNES_ag = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_ag"')
        get_map2json(bio_GBF4_ECNES_ag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_Ag.js')
        print('│   ├── Biodiversity GBF4_ECNES Ag layer saved.')

        bio_GBF4_ECNES_am = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_ag_management"')
        get_map2json(bio_GBF4_ECNES_am, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_Am.js')
        print('│   ├── Biodiversity GBF4_ECNES Am layer saved.')

        bio_GBF4_ECNES_nonag = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_non_ag"')
        get_map2json(bio_GBF4_ECNES_nonag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_NonAg.js')
        print('│   ├── Biodiversity GBF4_ECNES Non-Ag layer saved.')

    # GBF8
    if settings.BIODIVERSITY_TARGET_GBF_8 != 'off':
        bio_GBF8_ag = files_bio.query('base_name == "xr_biodiversity_GBF8_species_ag"')
        get_map2json(bio_GBF8_ag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_Ag.js')
        print('│   ├── Biodiversity GBF8 Ag layer saved.')

        bio_GBF8_am = files_bio.query('base_name == "xr_biodiversity_GBF8_species_ag_management"')
        get_map2json(bio_GBF8_am, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_Am.js')
        print('│   ├── Biodiversity GBF8 Am layer saved.')

        bio_GBF8_nonag = files_bio.query('base_name == "xr_biodiversity_GBF8_species_non_ag"')
        get_map2json(bio_GBF8_nonag, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_NonAg.js')
        print('│   ├── Biodiversity GBF8 Non-Ag layer saved.')

        bio_GBF8_ag_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_ag"')
        get_map2json(bio_GBF8_ag_group, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_Ag.js')
        print('│   ├── Biodiversity GBF8 Groups Ag layer saved.')

        bio_GBF8_am_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_ag_management"')
        get_map2json(bio_GBF8_am_group, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_Am.js')
        print('│   ├── Biodiversity GBF8 Groups Am layer saved.')

        bio_GBF8_nonag_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_non_ag"')
        get_map2json(bio_GBF8_nonag_group, None, None, legend_float, bio_min_max, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_NonAg.js')
        print('│   ├── Biodiversity GBF8 Groups Non-Ag layer saved.')

    


    ####################################################
    #                    5) Economics                  #
    ####################################################
    
    ecnomic_magnitudes_ag = (
        *cell_magnitudes['Economics_ag']['ag_revenue'].values(),
        *cell_magnitudes['Economics_ag']['ag_cost'].values(),
        *cell_magnitudes['Economics_ag']['profit_ag'].values()
    )
    ecnomic_magnitudes_nonag = (
        *cell_magnitudes['Economics_am']['am_revenue'].values(),
        *cell_magnitudes['Economics_am']['am_cost'].values(),
        *cell_magnitudes['Economics_am']['am_profit'].values()
    )
    ecnomic_magnitudes_am = (
        cell_magnitudes['Economics_non_ag']['non_ag_revenue'],
        cell_magnitudes['Economics_non_ag']['non_ag_cost'],
        cell_magnitudes['Economics_non_ag']['non_ag_profit']
    )
    economic_magnitudes_transition = (
        *cell_magnitudes['Economics_ag']['ag2ag_cost'].values(),
        *cell_magnitudes['Economics_ag']['non_ag2ag_cost'].values(),
        *cell_magnitudes['Economics_am']['am_transition'].values(),
        cell_magnitudes['Economics_non_ag']['non_ag_to_non_ag_cost'],
        cell_magnitudes['Economics_non_ag']['non_ag_to_ag_cost']
    )
    
    economic_min_max_ag = (min(ecnomic_magnitudes_ag), max(ecnomic_magnitudes_ag))
    economic_min_max_nonag = (min(ecnomic_magnitudes_nonag), max(ecnomic_magnitudes_nonag))
    economic_min_max_am = (min(ecnomic_magnitudes_am), max(ecnomic_magnitudes_am))
    economic_min_max_transition = (min(economic_magnitudes_transition), max(economic_magnitudes_transition))

    # ---------------- Profit ----------------
    profit_ag = files.query('base_name == "xr_economics_ag_profit"')
    get_map2json(profit_ag, None, None, legend_float, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_profit.js')
    print('│   ├── Economics Ag profit layer saved.')

    profit_am = files.query('base_name == "xr_economics_am_profit"')
    get_map2json(profit_am, None, None, legend_float, economic_min_max_am, f'{SAVE_DIR}/map_layers/map_economics_Am_profit.js')
    print('│   ├── Economics Am profit layer saved.')

    profit_nonag = files.query('base_name == "xr_economics_non_ag_profit"')
    get_map2json(profit_nonag, None, None, legend_float, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_profit.js')
    print('│   ├── Economics NonAg profit layer saved.')


    # ---------------- Revenue ----------------
    revenue_ag = files.query('base_name == "xr_economics_ag_revenue"')
    get_map2json(revenue_ag, None, None, legend_float, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_revenue.js')
    print('│   ├── Economics Ag revenue layer saved.')

    revenue_am = files.query('base_name == "xr_economics_am_revenue"')
    get_map2json(revenue_am, None, None, legend_float, economic_min_max_am, f'{SAVE_DIR}/map_layers/map_economics_Am_revenue.js')
    print('│   ├── Economics Am revenue layer saved.')
    
    revenue_nonag = files.query('base_name == "xr_economics_non_ag_revenue"')
    get_map2json(revenue_nonag, None, None, legend_float, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_revenue.js')
    print('│   ├── Economics NonAg revenue layer saved.')


    # ---------------- Cost ----------------
    cost_ag = files.query('base_name == "xr_economics_ag_cost"')
    get_map2json(cost_ag, None, None, legend_float, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_cost.js')
    print('│   ├── Economics Ag cost layer saved.')

    cost_am = files.query('base_name == "xr_economics_am_cost"')
    get_map2json(cost_am, None, None, legend_float, economic_min_max_am, f'{SAVE_DIR}/map_layers/map_economics_Am_cost.js')
    print('│   ├── Economics Am cost layer saved.')

    cost_nonag = files.query('base_name == "xr_economics_non_ag_cost"')
    get_map2json(cost_nonag, None, None, legend_float, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_NonAg_cost.js')
    print('│   ├── Economics NonAg cost layer saved.')


    # ---------------- Transition Cost ----------------
    cost_trans_ag2ag = files.query('base_name == "xr_economics_ag_transition_ag2ag"')
    get_map2json(cost_trans_ag2ag, None, None, legend_float, economic_min_max_ag, f'{SAVE_DIR}/map_layers/map_economics_Ag_transition_ag2ag.js')
    print('│   ├── Economics Ag transition ag2ag layer saved.')

    cost_trans_ag2nonag = files.query('base_name == "xr_economics_ag_transition_non_ag2ag"')
    get_map2json(cost_trans_ag2nonag, None, None, legend_float, economic_min_max_nonag, f'{SAVE_DIR}/map_layers/map_economics_Ag_transition_ag2nonag.js')
    print('│   ├── Economics Ag transition ag2nonag layer saved.')

    # AgMgt has 0 transition cost, so skipping
    # cost_ag_man = files.query('base_name == "xr_economics_am_transition"')

    # Non-Ag to Ag transition cost is not allowed, so skipping
    # cost_trans_nonag2ag = files.query('base_name == "xr_economics_non_ag_transition_non_ag2ag"')
    
    


    ####################################################
    #                    6) GHG                        #
    ####################################################

    files_ghg = files.query('base_name.str.contains("GHG")')
    
    ghg_magnitudes = (
        *cell_magnitudes['ghg_emission']['ag'].values(),
        *cell_magnitudes['ghg_emission']['non_ag'].values(),
        *cell_magnitudes['ghg_emission']['ag_man'].values(),
        *cell_magnitudes['ghg_emission']['transition'].values(),
    )
    
    ghg_min_max = (min(ghg_magnitudes), max(ghg_magnitudes))

    ghg_ag = files_ghg.query('base_name == "xr_GHG_ag"')
    get_map2json(ghg_ag, None, None, legend_float, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_Ag.js')
    print('│   ├── GHG Ag layer saved.')

    ghg_am = files_ghg.query('base_name == "xr_GHG_ag_management"')
    get_map2json(ghg_am, None, None, legend_float, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_Am.js')
    print('│   ├── GHG Am layer saved.')

    ghg_nonag = files_ghg.query('base_name == "xr_GHG_non_ag"')
    get_map2json(ghg_nonag, None, None, legend_float, ghg_min_max, f'{SAVE_DIR}/map_layers/map_GHG_NonAg.js')
    print('│   ├── GHG Non-Ag layer saved.')



    ####################################################
    #                  7) Quantities                   #
    ####################################################

    files_quantities = files.query('base_name.str.contains("quantities")')

    quantities_ag = files_quantities.query('base_name == "xr_quantities_agricultural"')
    get_map2json(quantities_ag, legend_ag, {'Commodity':'ALL'}, legend_float, f'{SAVE_DIR}/map_layers/map_quantities_Ag.js')
    print('│   ├── Quantities Ag layer saved.')

    quantities_am = files_quantities.query('base_name == "xr_quantities_agricultural_management"')
    get_map2json(quantities_am, legend_am, {'am':'ALL'}, legend_float, f'{SAVE_DIR}/map_layers/map_quantities_Am.js')
    print('│   ├── Quantities Am layer saved.')

    quantities_nonag = files_quantities.query('base_name == "xr_quantities_non_agricultural"')
    get_map2json(quantities_nonag, legend_non_ag, {'Commodity':'ALL'}, legend_float, f'{SAVE_DIR}/map_layers/map_quantities_NonAg.js')
    print('│   ├── Quantities Non-Ag layer saved.')



    ####################################################
    #                8) Transition Cost                #
    ####################################################

    # files_transition = files.query('base_name.str.contains("transition")')

    # transition_cost = files_transition.query('base_name == "xr_transition_cost_ag2non_ag"')
    # get_map2json(transition_cost, f'{SAVE_DIR}/map_layers/map_transition_cost.js')
    # print('     ... Transition Cost layer saved.')

    # transition_ghg = files_transition.query('base_name == "xr_transition_GHG"')
    # get_map2json(transition_ghg, f'{SAVE_DIR}/map_layers/map_transition_GHG.js')
    # print('     ... Transition GHG layer saved.')



    ####################################################
    #               9) Water Yield                    #
    ####################################################

    files_water = files.query('base_name.str.contains("water_yield")')
    
    water_magnitudes = (
        *cell_magnitudes['water_yield']['ag'].values(),
        *cell_magnitudes['water_yield']['non_ag'].values(),
        *cell_magnitudes['water_yield']['ag_man'].values(),
    )

    water_min_max = (min(water_magnitudes), max(water_magnitudes))

    water_ag = files_water.query('base_name == "xr_water_yield_ag"')
    get_map2json(water_ag, None, None, legend_float, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_Ag.js')
    print('│   ├── Water Yield Ag layer saved.')

    water_am = files_water.query('base_name == "xr_water_yield_ag_management"')
    get_map2json(water_am, None, None, legend_float, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_Am.js')
    print('│   ├── Water Yield Am layer saved.')

    water_nonag = files_water.query('base_name == "xr_water_yield_non_ag"')
    get_map2json(water_nonag, None, None, legend_float, water_min_max, f'{SAVE_DIR}/map_layers/map_water_yield_NonAg.js')
    print('│   └── Water Yield Non-Ag layer saved.')