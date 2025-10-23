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

from joblib import delayed, Parallel

from luto import settings
from luto.tools.report.data_tools import (
    array_to_base64, 
    get_all_files, 
    get_map_legend,
    hex_color_to_numeric, 
    rename_reorder_hierarchy, 
    tuple_dict_to_nested
)



def map2base64(ds_template:str, arr_lyr:xr.DataArray, min_max:tuple[float,float], isInt:bool, legend_dict:dict, hierarchy_tp:tuple) -> tuple:

    # Get an template rio-xarray, it will be used to convert 1D array to its 2D map format
    with xr.open_dataset(ds_template) as rxr_ds:
        rxr_arr = rxr_ds['layer'].astype('float32')         # Use float32 to allow NaN values
        rxr_crs = rxr_ds['spatial_ref'].attrs['crs_wkt']

    img_attrs = {}
    if isInt:
        img_attrs = {
            'intOrFloat': 'int',
            'legend': legend_dict['legend']
        }
    else:
        # Normalize the layer to 0-100 as integer
        min_val, max_val = min_max
        arr_lyr.values = (arr_lyr - min_val) / (max_val - min_val) * 100
        arr_lyr = arr_lyr.astype(np.float32)
        img_attrs = {
            'intOrFloat': 'float',
            'legend': legend_dict['legend'],
            'min_max': min_max
        }
        

    # Convert the 1D array to a 2D array
    np.place(rxr_arr.data, rxr_arr.data>=0, arr_lyr.values)     # Negative values in the template are outside LUTO area
    rxr_arr = xr.where(rxr_arr<0, np.nan, rxr_arr)              # Set negative values to NaN, which will be transparent in the final map
    # Get the bounds
    rxr_arr = rxr_arr.rio.write_crs(rxr_crs)
    bbox = rxr_arr.rio.bounds()
    bbox = [[bbox[1], bbox[0]],[bbox[3], bbox[2]]]
    # Reproject to Web Mercator EPSG:3857; because Leaflet uses EPSG:3857 to display maps
    rxr_arr = rxr_arr.rio.reproject('EPSG:3857')                # To Mercator with Nearest Neighbour
    rxr_arr = np.nan_to_num(rxr_arr, nan=-1).astype('int16')    # Use -1 to flag nodata pixels
    # Convert the 1D array to a RGBA array
    color_csv = pd.read_csv(legend_dict['color_csv'])
    color_csv['color_numeric'] = color_csv['lu_color_HEX'].apply(hex_color_to_numeric)
    color_dict = color_csv.set_index('lu_code')['color_numeric'].to_dict()
    color_dict[-1] = (0,0,0,0)                                  # Nodata pixels are transparent
    arr_4band = np.zeros((rxr_arr.shape[0], rxr_arr.shape[1], 4), dtype='uint8')
    for k, v in color_dict.items():
        arr_4band[rxr_arr == k] = v

    return hierarchy_tp, {**array_to_base64(arr_4band), 'bounds':bbox, **img_attrs}
    
    

def get_map_float(
    files_df:pd.DataFrame, 
    legend_int:dict, 
    legend_int_level:dict|str, 
    legend_float:dict, 
    save_path:str, 
    workers:int=min(settings.WRITE_THREADS, 16)
    ) -> None:

    # Loop through each year
    tasks = []
    for _,row in files_df.iterrows():
        xr_arr = xr.open_dataarray(row['path'])
        chunks = {k:1 for k in xr_arr.dims}
        chunks.update({'cell':-1})
        xr_arr = xr_arr.chunk(chunks)  # the cell dimension is full size, all other dimensions are 1
        min_max = eval(xr_arr.attrs['valid_layers'])
        
        _year = row['Year']
        ds_template = f'{os.path.dirname(row['path'])}/xr_map_template_{_year}.nc'
        
        # Loop and skip empty layers, the valid layers are precalculated and stored as attribute
        for sel in eval(xr_arr.attrs['valid_layers']):
            
            if isinstance(legend_int_level, dict):
                if legend_int_level.items() <= sel.items():
                    legend = legend_int
                    arrIsInt = True
                else:
                    legend = legend_float
                    arrIsInt = False
                    
            elif isinstance(legend_int_level, str):
                if legend_int_level in sel.keys():
                    legend = legend_int
                    arrIsInt = True
                else:
                    legend = legend_float
                    arrIsInt = False
            else:
                raise ValueError('legend_int_level must be either a dict or a str')
            
            arr_sel = xr_arr.sel(**sel) 
            sel_rename = rename_reorder_hierarchy(sel)
            hierarchy_tp = tuple(list(sel_rename.values()) + [_year])

            tasks.append(
                delayed(map2base64)(ds_template, arr_sel, min_max, arrIsInt, legend, hierarchy_tp)
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

    # Create the directory if it does not exist
    if not os.path.exists(f'{SAVE_DIR}/map_layers'):
        os.makedirs(f'{SAVE_DIR}/map_layers', exist_ok=True)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir).query('category == "xarray_layer"')
    files['Year'] = files['Year'].astype(int)
    files = files.query(f'Year.isin({sorted(settings.SIM_YEARS)})')
    
    # Get legend info
    colors_legend = get_map_legend()
    colors_legend_ag = colors_legend['ag']
    colors_legend_am = colors_legend['am']
    colors_legend_non_ag = colors_legend['non_ag']
    colors_legend_lumap = colors_legend['lumap']
    colors_legend_float = colors_legend['float']

    
    ####################################################
    #                   1) LUMAP Layer                 #
    ####################################################
    lumap_ly = files.query('base_name == "xr_map_lumap"')
    get_map_float(lumap_ly, colors_legend_lumap, 'lm', None, f'{SAVE_DIR}/map_layers/map_dvar_lumap.js')


 
    ####################################################
    #                   2) Dvar Layer                  #
    ####################################################
    
    dvar_ag = files.query('base_name == "xr_dvar_ag"')
    get_map_float(dvar_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_dvar_Ag.js')
    
    dvar_am = files.query('base_name == "xr_dvar_am"')
    get_map_float(dvar_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_dvar_Am.js')
    
    dvar_nonag = files.query('base_name == "xr_dvar_non_ag"')
    get_map_float(dvar_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_dvar_NonAg.js')
    
    
    
    ####################################################
    #                   3) Area Layer                  #
    ####################################################

    files_area = files.query('base_name.str.contains("area")')

    area_ag = files_area.query('base_name == "xr_area_agricultural_landuse"')
    get_map_float(area_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_area_Ag.js')

    area_am = files_area.query('base_name == "xr_area_agricultural_management"')
    get_map_float(area_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_area_Am.js')

    area_nonag = files_area.query('base_name == "xr_area_non_agricultural_landuse"')
    get_map_float(area_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_area_NonAg.js')



    ####################################################
    #                  4) Biodiversity                 #
    ####################################################

    files_bio = files.query('base_name.str.contains("biodiversity")')
    
    # Overall priority
    bio_overall_ag = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag"')
    get_map_float(bio_overall_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_overall_Ag.js')

    bio_overall_am = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag_management"')
    get_map_float(bio_overall_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_overall_Am.js')

    bio_overall_nonag = files_bio.query('base_name == "xr_biodiversity_overall_priority_non_ag"')
    get_map_float(bio_overall_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_overall_NonAg.js')


    # GBF2
    if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':
        bio_GBF2_ag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag"')
        get_map_float(bio_GBF2_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Ag.js')

        bio_GBF2_am = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag_management"')
        get_map_float(bio_GBF2_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Am.js')

        bio_GBF2_nonag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_non_ag"')
        get_map_float(bio_GBF2_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF2_NonAg.js')
        
    # GBF3-NVIS
    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS != 'off':
        bio_GBF3_NVIS_ag = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_ag"')
        get_map_float(bio_GBF3_NVIS_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_Ag.js')

        bio_GBF3_NVIS_am = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_ag_management"')
        get_map_float(bio_GBF3_NVIS_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_Am.js')

        bio_GBF3_NVIS_nonag = files_bio.query('base_name == "xr_biodiversity_GBF3_NVIS_non_ag"')
        get_map_float(bio_GBF3_NVIS_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF3_NVIS_NonAg.js')
        
    # GBF3-IBRA
    if settings.BIODIVERSITY_TARGET_GBF_3_IBRA != 'off':
        bio_GBF3_IBRA_ag = files_bio.query('base_name == "xr_biodiversity_GBF3_IBRA_ag"')
        get_map_float(bio_GBF3_IBRA_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF3_IBRA_Ag.js')

        bio_GBF3_IBRA_am = files_bio.query('base_name == "xr_biodiversity_GBF3_IBRA_ag_management"')
        get_map_float(bio_GBF3_IBRA_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF3_IBRA_Am.js')

        bio_GBF3_IBRA_nonag = files_bio.query('base_name == "xr_biodiversity_GBF3_IBRA_non_ag"')
        get_map_float(bio_GBF3_IBRA_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF3_IBRA_NonAg.js')
    
    # GBF4-SNES
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES != 'off':
        bio_GBF4_SNES_ag = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_ag"')
        get_map_float(bio_GBF4_SNES_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_Ag.js')

        bio_GBF4_SNES_am = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_ag_management"')
        get_map_float(bio_GBF4_SNES_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_Am.js')

        bio_GBF4_SNES_nonag = files_bio.query('base_name == "xr_biodiversity_GBF4_SNES_non_ag"')
        get_map_float(bio_GBF4_SNES_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF4_SNES_NonAg.js')
        
    # GBF4_ECNES
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES != 'off':
        bio_GBF4_ECNES_ag = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_ag"')
        get_map_float(bio_GBF4_ECNES_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_Ag.js')

        bio_GBF4_ECNES_am = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_ag_management"')
        get_map_float(bio_GBF4_ECNES_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_Am.js')

        bio_GBF4_ECNES_nonag = files_bio.query('base_name == "xr_biodiversity_GBF4_ECNES_non_ag"')
        get_map_float(bio_GBF4_ECNES_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF4_ECNES_NonAg.js')
        
    # GBF8
    if settings.BIODIVERSITY_TARGET_GBF_8 != 'off':
        bio_GBF8_ag = files_bio.query('base_name == "xr_biodiversity_GBF8_species_ag"')
        get_map_float(bio_GBF8_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF8_Ag.js')

        bio_GBF8_am = files_bio.query('base_name == "xr_biodiversity_GBF8_species_ag_management"')
        get_map_float(bio_GBF8_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF8_Am.js')

        bio_GBF8_nonag = files_bio.query('base_name == "xr_biodiversity_GBF8_species_non_ag"')
        get_map_float(bio_GBF8_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF8_NonAg.js')

        bio_GBF8_ag_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_ag"')
        get_map_float(bio_GBF8_ag_group, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_Ag.js')

        bio_GBF8_am_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_ag_management"')
        get_map_float(bio_GBF8_am_group, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_Am.js')

        bio_GBF8_nonag_group = files_bio.query('base_name == "xr_biodiversity_GBF8_groups_non_ag"')
        get_map_float(bio_GBF8_nonag_group, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_bio_GBF8_groups_NonAg.js')

    


    ####################################################
    #                    5) Cost                       #
    ####################################################

    files_cost = files.query('base_name.str.contains("cost")')

    cost_ag = files_cost.query('base_name == "xr_cost_ag"')
    get_map_float(cost_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_cost_Ag.js')

    cost_am = files_cost.query('base_name == "xr_cost_agricultural_management"')
    get_map_float(cost_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_cost_Am.js')

    cost_nonag = files_cost.query('base_name == "xr_cost_non_ag"')
    get_map_float(cost_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_cost_NonAg.js')

    # cost_transition = files_cost.query('base_name == "xr_cost_transition_ag2ag"')
    # get_map_float(cost_transition, f'{SAVE_DIR}/map_layers/map_cost_transition.js')



    ####################################################
    #                    6) GHG                        #
    ####################################################

    files_ghg = files.query('base_name.str.contains("GHG")')

    ghg_ag = files_ghg.query('base_name == "xr_GHG_ag"')
    get_map_float(ghg_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_GHG_Ag.js')

    ghg_am = files_ghg.query('base_name == "xr_GHG_ag_management"')
    get_map_float(ghg_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_GHG_Am.js')

    ghg_nonag = files_ghg.query('base_name == "xr_GHG_non_ag"')
    get_map_float(ghg_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_GHG_NonAg.js')



    ####################################################
    #                  7) Quantities                   #
    ####################################################

    files_quantities = files.query('base_name.str.contains("quantities")')

    quantities_ag = files_quantities.query('base_name == "xr_quantities_agricultural"')
    get_map_float(quantities_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_quantities_Ag.js')

    quantities_am = files_quantities.query('base_name == "xr_quantities_agricultural_management"')
    get_map_float(quantities_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_quantities_Am.js')

    quantities_nonag = files_quantities.query('base_name == "xr_quantities_non_agricultural"')
    get_map_float(quantities_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_quantities_NonAg.js')



    ####################################################
    #                   8) Revenue                     #
    ####################################################

    files_revenue = files.query('base_name.str.contains("revenue")')

    revenue_ag = files_revenue.query('base_name == "xr_revenue_ag"')
    get_map_float(revenue_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_revenue_Ag.js')

    revenue_am = files_revenue.query('base_name == "xr_revenue_agricultural_management"')
    get_map_float(revenue_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_revenue_Am.js')

    revenue_nonag = files_revenue.query('base_name == "xr_revenue_non_ag"')
    get_map_float(revenue_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_revenue_NonAg.js')



    ####################################################
    #                9) Transition Cost                #
    ####################################################

    # files_transition = files.query('base_name.str.contains("transition")')

    # transition_cost = files_transition.query('base_name == "xr_transition_cost_ag2non_ag"')
    # get_map_float(transition_cost, f'{SAVE_DIR}/map_layers/map_transition_cost.js')

    # transition_ghg = files_transition.query('base_name == "xr_transition_GHG"')
    # get_map_float(transition_ghg, f'{SAVE_DIR}/map_layers/map_transition_GHG.js')



    ####################################################
    #               10) Water Yield                    #
    ####################################################

    files_water = files.query('base_name.str.contains("water_yield")')

    water_ag = files_water.query('base_name == "xr_water_yield_ag"')
    get_map_float(water_ag, colors_legend_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_water_yield_Ag.js')

    water_am = files_water.query('base_name == "xr_water_yield_ag_management"')
    get_map_float(water_am, colors_legend_am, {'am':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_water_yield_Am.js')

    water_nonag = files_water.query('base_name == "xr_water_yield_non_ag"')
    get_map_float(water_nonag, colors_legend_non_ag, {'lu':'ALL'}, colors_legend_float, f'{SAVE_DIR}/map_layers/map_water_yield_NonAg.js')