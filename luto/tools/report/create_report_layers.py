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


import json
import os
import base64
import numpy as np
import pandas as pd
import xarray as xr

from io import BytesIO
from PIL import Image
from joblib import delayed, Parallel

from luto import settings
from luto.data import Data
from luto.tools.report.data_tools import get_all_files
from luto.tools.report.data_tools.parameters import RENAME_AM_NON_AG



def tuple_dict_to_nested(flat_dict):
    nested_dict = {}
    for key_tuple, value in flat_dict.items():
        current_level = nested_dict
        for key in key_tuple[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]
        current_level[key_tuple[-1]] = value
    return nested_dict
        
        
def hex_color_to_numeric(hex:str) -> tuple:
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4, 6)) 


def array_to_base64(arr_4band: np.ndarray, bbox: list, min_max:list) -> dict:

    # Create PIL Image from RGBA array
    image = Image.fromarray(arr_4band, 'RGBA')
    
    # Convert to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        'img_str': 'data:image/png;base64,' + img_str,
        'bounds': [
            [bbox[1], bbox[0]],
            [bbox[3], bbox[2]]
        ],
        'min_max': min_max
    }



def map2base64(rxr_path:str, arr_lyr:xr.DataArray, attrs:tuple) -> dict|None:

        # Get an template rio-xarray, it will be used to convert 1D array to its 2D map format
        with xr.open_dataset(rxr_path) as rxr_ds:
            rxr_arr = rxr_ds['__xarray_dataarray_variable__']
            rxr_crs = rxr_ds['spatial_ref'].attrs['crs_wkt']

        # Skip if the layer is empty
        if arr_lyr.sum() == 0:
            return 

        # Normalize the layer
        min_val = np.nanmin(arr_lyr.values)
        max_val = np.nanmax(arr_lyr.values)
        arr_lyr.values = (arr_lyr - min_val) / (max_val - min_val)

        # Convert the 1D array to a 2D array
        np.place(rxr_arr.data, rxr_arr.data>=0, arr_lyr.data)  # Set negative values to NaN
        rxr_arr = xr.where(rxr_arr<0, np.nan, rxr_arr)
        rxr_arr = rxr_arr.rio.write_crs(rxr_crs)

        # Get bounding box, then reproject to Mercator
        bbox = rxr_arr.rio.bounds()
        rxr_arr = rxr_arr.rio.reproject('EPSG:3857') # To Mercator with Nearest Neighbour

        # Convert layer to integer; after this 0 is nodata, -100 is outside LUTO area
        rxr_arr.values *= 100
        rxr_arr = np.where(np.isnan(rxr_arr), 0, rxr_arr).astype('int32')

        # Convert the 1D array to a RGBA array
        color_dict = pd.read_csv('luto/tools/report/VUE_modules/assets/float_img_colors.csv')
        if max_val == 0:
            color_dict.loc[range(100), 'lu_color_HEX'] = color_dict.loc[range(100), 'lu_color_HEX'].values[::-1]
            min_val, max_val = max_val, min_val
        color_dict['color_numeric'] = color_dict['lu_color_HEX'].apply(hex_color_to_numeric)
        color_dict = color_dict.set_index('lu_code')['color_numeric'].to_dict()

        color_dict[0] = (0,0,0,0)    # Nodata pixels are transparent

        arr_4band = np.zeros((rxr_arr.shape[0], rxr_arr.shape[1], 4), dtype='uint8')
        for k, v in color_dict.items():
            arr_4band[rxr_arr == k] = v

        # Generate base64 and overlay info
        return attrs, array_to_base64(arr_4band, bbox, [float(max_val), float(min_val)])
    
    

def get_map_obj(data:Data, files_df:pd.DataFrame, save_path:str, workers:int=settings.WRITE_THREADS) -> dict:

    # Get an template rio-xarray, it will be used to convert 1D array to its 2D map format
    template_xr = f'{data.path}/out_{sorted(settings.SIM_YEARS)[0]}/xr_map_lumap_{sorted(settings.SIM_YEARS)[0]}.nc'
    
    # Get dim info
    with xr.open_dataarray(files_df.iloc[0]['path']) as arr_eg:
        loop_dims = set(arr_eg.dims) - set(['cell'])
        
        dim_vals = pd.MultiIndex.from_product(
            [arr_eg[dim].values for dim in loop_dims],
            names=loop_dims
        ).to_list()

        loop_sel = [dict(zip(loop_dims, val)) for val in dim_vals]
        
    # Loop through each year
    task = []
    for _,row in files_df.iterrows():
        xr_arr = xr.load_dataarray(row.path)
        _year = row['Year']
        
        for sel in loop_sel:
            arr_sel = xr_arr.sel(**sel) 
            
            # Rename keys; also serve as reordering the keys
            sel_rename = {}
            if 'am' in sel:
                sel_rename['am'] = RENAME_AM_NON_AG.get(sel['am'], sel['am'])
            if 'lm' in sel:
                sel_rename['lm'] =  {'irr': 'Irrigated', 'dry': 'Dryland'}.get(sel['lm'], sel['lm'])
            if 'lu' in sel:
                sel_rename['lu'] = RENAME_AM_NON_AG.get(sel['lu'], sel['lu'])
            if 'Commodity' in sel:
                commodity = sel['Commodity'].capitalize()
                sel_rename['Commodity'] = {
                    'Sheep lexp': 'Sheep live export',
                    'Beef lexp': 'Beef live export'
                }.get(commodity, commodity)
                
            task.append(
                delayed(map2base64)(template_xr, arr_sel, tuple(list(sel_rename.values()) + [_year]))
            )    
            
    # Gather results and save to JSON
    output = {}
    for res in Parallel(n_jobs=workers, return_as='generator')(task):
        if res is None:continue
        attr, val = res
        output[attr] = val
        
    # To nested dict
    output = tuple_dict_to_nested(output)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        filename = os.path.basename(save_path).replace('.js', '')
        f.write(f'window["{filename}"] = ')
        json.dump(output, f, separators=(',', ':'), indent=2)
        f.write(';\n')




def save_report_layer(data:Data, raw_data_dir:str):
    """
    Saves the report data in the specified directory.

    Parameters
    ----------
    data (Data): The Data object containing the metadata and settings.
    raw_data_dir (str): The directory where the raw data is stored.
    
    Returns
    -------
    None
    """
    
    SAVE_DIR = f'{raw_data_dir}/DATA_REPORT/data'
    years = sorted(settings.SIM_YEARS)

    # Create the directory if it does not exist
    if not os.path.exists(f'{SAVE_DIR}/map_layers'):
        os.makedirs(f'{SAVE_DIR}/map_layers', exist_ok=True)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir).query('category == "xarray_layer"')
    files['Year'] = files['Year'].astype(int)
    files = files.query('Year.isin(@years)')
    
    
    
    ####################################################
    #                   1) Area Layer                  #
    ####################################################

    files_area = files.query('base_name.str.contains("area")')

    area_ag = files_area.query('base_name == "xr_area_agricultural_landuse"')
    get_map_obj(data, area_ag, f'{SAVE_DIR}/map_layers/map_area_Ag.js')

    area_am = files_area.query('base_name == "xr_area_agricultural_management"')
    get_map_obj(data, area_am, f'{SAVE_DIR}/map_layers/map_area_Am.js')

    area_nonag = files_area.query('base_name == "xr_area_non_agricultural_landuse"')
    get_map_obj(data, area_nonag, f'{SAVE_DIR}/map_layers/map_area_NonAg.js')

    ####################################################
    #                  2) Biodiversity                 #
    ####################################################

    files_bio = files.query('base_name.str.contains("biodiversity")')

    # GBF2
    bio_GBF2_ag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag"')
    get_map_obj(data, bio_GBF2_ag, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Ag.js')

    bio_GBF2_am = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag_management"')
    get_map_obj(data, bio_GBF2_am, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Am.js')

    bio_GBF2_nonag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_non_ag"')
    get_map_obj(data, bio_GBF2_nonag, f'{SAVE_DIR}/map_layers/map_bio_GBF2_NonAg.js')

    # Overall priority
    bio_overall_ag = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag"')
    get_map_obj(data, bio_overall_ag, f'{SAVE_DIR}/map_layers/map_bio_overall_Ag.js')

    bio_overall_am = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag_management"')
    get_map_obj(data, bio_overall_am, f'{SAVE_DIR}/map_layers/map_bio_overall_Am.js')

    bio_overall_nonag = files_bio.query('base_name == "xr_biodiversity_overall_priority_non_ag"')
    get_map_obj(data, bio_overall_nonag, f'{SAVE_DIR}/map_layers/map_bio_overall_NonAg.js')

    ####################################################
    #                    3) Cost                       #
    ####################################################

    files_cost = files.query('base_name.str.contains("cost")')

    cost_ag = files_cost.query('base_name == "xr_cost_ag"')
    get_map_obj(data, cost_ag, f'{SAVE_DIR}/map_layers/map_cost_Ag.js')

    cost_am = files_cost.query('base_name == "xr_cost_agricultural_management"')
    get_map_obj(data, cost_am, f'{SAVE_DIR}/map_layers/map_cost_Am.js')

    cost_nonag = files_cost.query('base_name == "xr_cost_non_ag"')
    get_map_obj(data, cost_nonag, f'{SAVE_DIR}/map_layers/map_cost_NonAg.js')

    # cost_transition = files_cost.query('base_name == "xr_cost_transition_ag2ag"')
    # get_map_obj(data, cost_transition, f'{SAVE_DIR}/map_layers/map_cost_transition.js')

    ####################################################
    #                    4) GHG                        #
    ####################################################

    files_ghg = files.query('base_name.str.contains("GHG")')

    ghg_ag = files_ghg.query('base_name == "xr_GHG_ag"')
    get_map_obj(data, ghg_ag, f'{SAVE_DIR}/map_layers/map_GHG_Ag.js')

    ghg_am = files_ghg.query('base_name == "xr_GHG_ag_management"')
    get_map_obj(data, ghg_am, f'{SAVE_DIR}/map_layers/map_GHG_Am.js')

    ghg_nonag = files_ghg.query('base_name == "xr_GHG_non_ag"')
    get_map_obj(data, ghg_nonag, f'{SAVE_DIR}/map_layers/map_GHG_NonAg.js')

    ####################################################
    #                  5) Quantities                   #
    ####################################################

    files_quantities = files.query('base_name.str.contains("quantities")')

    quantities_ag = files_quantities.query('base_name == "xr_quantities_agricultural"')
    get_map_obj(data, quantities_ag, f'{SAVE_DIR}/map_layers/map_quantities_Ag.js')

    quantities_am = files_quantities.query('base_name == "xr_quantities_agricultural_management"')
    get_map_obj(data, quantities_am, f'{SAVE_DIR}/map_layers/map_quantities_Am.js')

    quantities_nonag = files_quantities.query('base_name == "xr_quantities_non_agricultural"')
    get_map_obj(data, quantities_nonag, f'{SAVE_DIR}/map_layers/map_quantities_NonAg.js')

    ####################################################
    #                   6) Revenue                     #
    ####################################################

    files_revenue = files.query('base_name.str.contains("revenue")')

    revenue_ag = files_revenue.query('base_name == "xr_revenue_ag"')
    get_map_obj(data, revenue_ag, f'{SAVE_DIR}/map_layers/map_revenue_Ag.js')

    revenue_am = files_revenue.query('base_name == "xr_revenue_agricultural_management"')
    get_map_obj(data, revenue_am, f'{SAVE_DIR}/map_layers/map_revenue_Am.js')

    revenue_nonag = files_revenue.query('base_name == "xr_revenue_non_ag"')
    get_map_obj(data, revenue_nonag, f'{SAVE_DIR}/map_layers/map_revenue_NonAg.js')

    ####################################################
    #                7) Transition Cost                #
    ####################################################

    # files_transition = files.query('base_name.str.contains("transition")')

    # transition_cost = files_transition.query('base_name == "xr_transition_cost_ag2non_ag"')
    # get_map_obj(data, transition_cost, f'{SAVE_DIR}/map_layers/map_transition_cost.js')

    # transition_ghg = files_transition.query('base_name == "xr_transition_GHG"')
    # get_map_obj(data, transition_ghg, f'{SAVE_DIR}/map_layers/map_transition_GHG.js')

    ####################################################
    #                8) Water Yield                    #
    ####################################################

    files_water = files.query('base_name.str.contains("water_yield")')

    water_ag = files_water.query('base_name == "xr_water_yield_ag"')
    get_map_obj(data, water_ag, f'{SAVE_DIR}/map_layers/map_water_yield_Ag.js')

    water_am = files_water.query('base_name == "xr_water_yield_ag_management"')
    get_map_obj(data, water_am, f'{SAVE_DIR}/map_layers/map_water_yield_Am.js')

    water_nonag = files_water.query('base_name == "xr_water_yield_non_ag"')
    get_map_obj(data, water_nonag, f'{SAVE_DIR}/map_layers/map_water_yield_NonAg.js')