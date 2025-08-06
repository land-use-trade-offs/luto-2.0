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
from tqdm.auto import tqdm
from collections import defaultdict

from luto import settings
from luto.data import Data
from luto.tools import start_memory_monitor, stop_memory_monitor
from luto.tools.report.data_tools import get_all_files
from luto.tools.Manual_jupyter_books.helpers import arr_to_xr
from luto.tools.report.map_tools import hex_color_to_numeric



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



def map2base64(data:Data, arr_lyr:xr.DataArray, attrs:tuple) -> dict|None:
        
        # Skip if the layer is empty
        if arr_lyr.sum() == 0:
            return

        # Normalize the layer
        min_val = np.nanmin(arr_lyr.values)
        max_val = np.nanmax(arr_lyr.values)
        arr_lyr.values = (arr_lyr - min_val) / (max_val - min_val)

        # Convert the 1D array to a 2D array
        arr_lyr = arr_to_xr(data, arr_lyr)
        bbox = arr_lyr.rio.bounds()
        arr_lyr = arr_lyr.rio.reproject('EPSG:3857') # To Mercator with Nearest Neighbour

        # Convert layer to integer; after this 0 is nodata, -100 is outside LUTO area
        arr_lyr.values *= 100
        arr_lyr = np.where(np.isnan(arr_lyr), 0, arr_lyr).astype('int32')

        # Convert the 1D array to a RGBA array
        color_dict = pd.read_csv('luto/tools/report/Assets/float_img_colors.csv')
        if max_val == 0:
            color_dict.loc[range(100), 'lu_color_HEX'] = color_dict.loc[range(100), 'lu_color_HEX'].values[::-1]
            min_val, max_val = max_val, min_val
        color_dict['color_numeric'] = color_dict['lu_color_HEX'].apply(hex_color_to_numeric)
        color_dict = color_dict.set_index('lu_code')['color_numeric'].to_dict()
        
        color_dict[0] = (0,0,0,0)    # Nodata pixcels are trasparent
                    
        arr_4band = np.zeros((arr_lyr.shape[0], arr_lyr.shape[1], 4), dtype='uint8')
        for k, v in color_dict.items():
            arr_4band[arr_lyr == k] = v

        # Generate base64 and overlay info
        return attrs, array_to_base64(arr_4band, bbox, [float(max_val), float(min_val)])
    
    
def tuple_dict_to_nested(flat_dict):
            nested = {}
            for key_tuple, value in flat_dict.items():
                current = nested
                for key in key_tuple[:-1]:
                    current = current.setdefault(key, {})
                current[key_tuple[-1]] = value
            return nested
    
    
    
def get_map_obj(data:Data, files_df:pd.DataFrame, save_path:str, workers:int=-1) -> dict:
        
    # Get Permute info
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
        xr_arr = xr.open_dataarray(row.path)
        chunk_size = {dim:1 for dim in xr_arr.dims if dim != 'cell'}
        chunk_size.update({'cell': data.NCELLS})  
        xr_arr = xr_arr.chunk(chunk_size)
        _year = row['Year']
        for sel in loop_sel:
            arr_sel = xr_arr.sel(**sel)                        
            task.append(
                delayed(map2base64)(data, arr_sel, tuple(list(sel.values()) + [_year]))
            )

    # Gather results and save to JSON
    output = {}
    for res in tqdm(Parallel(n_jobs=workers, return_as='generator')(task), total=len(task)):
        if res is None:continue
        attr, val = res
        output[attr] = val
        
    if not os.path.exists(save_path):
        os.makedirs
        
        
    with open(save_path, 'w') as f:
        json.dump(tuple_dict_to_nested(output), f, indent=2)



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
    if not os.path.exists(f'{SAVE_DIR}/map_metrics'):
        os.makedirs(f'{SAVE_DIR}/map_metrics', exist_ok=True)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir).query('category == "xarray_layer"')
    files['Year'] = files['Year'].astype(int)
    files = files.query('Year.isin(@years)')
    
    
    
    ####################################################
    #                   1) Area Layer                  #
    ####################################################

    files_area = files.query('base_name.str.contains("area")')

    area_ag = files_area.query('base_name == "xr_area_agricultural_landuse"')
    get_map_obj(data, area_ag, f'{SAVE_DIR}/map_metrics/area_Ag.json')

    area_am = files_area.query('base_name == "xr_area_agricultural_management"')
    get_map_obj(data, area_am, f'{SAVE_DIR}/map_metrics/area_Am.json')

    area_nonag = files_area.query('base_name == "xr_area_non_agricultural_landuse"')
    get_map_obj(data, area_nonag, f'{SAVE_DIR}/map_metrics/area_NonAg.json')

    ####################################################
    #                  2) Biodiversity                 #
    ####################################################

    files_bio = files.query('base_name.str.contains("biodiversity")')

    # GBF2
    bio_GBF2_ag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag"')
    get_map_obj(data, bio_GBF2_ag, f'{SAVE_DIR}/map_metrics/bio_GBF2_Ag.json')

    bio_GBF2_am = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag_management"')
    get_map_obj(data, bio_GBF2_am, f'{SAVE_DIR}/map_metrics/bio_GBF2_Am.json')

    bio_GBF2_nonag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_non_ag"')
    get_map_obj(data, bio_GBF2_nonag, f'{SAVE_DIR}/map_metrics/bio_GBF2_NonAg.json')

    # Overall priority
    bio_overall_ag = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag"')
    get_map_obj(data, bio_overall_ag, f'{SAVE_DIR}/map_metrics/bio_overall_Ag.json')

    bio_overall_am = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag_management"')
    get_map_obj(data, bio_overall_am, f'{SAVE_DIR}/map_metrics/bio_overall_Am.json')

    bio_overall_nonag = files_bio.query('base_name == "xr_biodiversity_overall_priority_non_ag"')
    get_map_obj(data, bio_overall_nonag, f'{SAVE_DIR}/map_metrics/bio_overall_NonAg.json')

    ####################################################
    #                    3) Cost                       #
    ####################################################

    files_cost = files.query('base_name.str.contains("cost")')

    cost_ag = files_cost.query('base_name == "xr_cost_ag"')
    get_map_obj(data, cost_ag, f'{SAVE_DIR}/map_metrics/cost_Ag.json')

    cost_am = files_cost.query('base_name == "xr_cost_agricultural_management"')
    get_map_obj(data, cost_am, f'{SAVE_DIR}/map_metrics/cost_Am.json')

    cost_nonag = files_cost.query('base_name == "xr_cost_non_ag"')
    get_map_obj(data, cost_nonag, f'{SAVE_DIR}/map_metrics/cost_NonAg.json')

    # cost_transition = files_cost.query('base_name == "xr_cost_transition_ag2ag"')
    # get_map_obj(data, cost_transition, f'{SAVE_DIR}/map_metrics/cost_transition.json')

    ####################################################
    #                    4) GHG                        #
    ####################################################

    files_ghg = files.query('base_name.str.contains("GHG")')

    ghg_ag = files_ghg.query('base_name == "xr_GHG_ag"')
    get_map_obj(data, ghg_ag, f'{SAVE_DIR}/map_metrics/GHG_Ag.json')

    ghg_am = files_ghg.query('base_name == "xr_GHG_ag_management"')
    get_map_obj(data, ghg_am, f'{SAVE_DIR}/map_metrics/GHG_Am.json')

    ghg_nonag = files_ghg.query('base_name == "xr_GHG_non_ag"')
    get_map_obj(data, ghg_nonag, f'{SAVE_DIR}/map_metrics/GHG_NonAg.json')

    ####################################################
    #                  5) Quantities                   #
    ####################################################

    files_quantities = files.query('base_name.str.contains("quantities")')

    quantities_ag = files_quantities.query('base_name == "xr_quantities_agricultural"')
    get_map_obj(data, quantities_ag, f'{SAVE_DIR}/map_metrics/quantities_Ag.json')

    quantities_am = files_quantities.query('base_name == "xr_quantities_agricultural_management"')
    get_map_obj(data, quantities_am, f'{SAVE_DIR}/map_metrics/quantities_Am.json')

    quantities_nonag = files_quantities.query('base_name == "xr_quantities_non_agricultural"')
    get_map_obj(data, quantities_nonag, f'{SAVE_DIR}/map_metrics/quantities_NonAg.json')

    ####################################################
    #                   6) Revenue                     #
    ####################################################

    files_revenue = files.query('base_name.str.contains("revenue")')

    revenue_ag = files_revenue.query('base_name == "xr_revenue_ag"')
    get_map_obj(data, revenue_ag, f'{SAVE_DIR}/map_metrics/revenue_Ag.json')

    revenue_am = files_revenue.query('base_name == "xr_revenue_agricultural_management"')
    get_map_obj(data, revenue_am, f'{SAVE_DIR}/map_metrics/revenue_Am.json')

    revenue_nonag = files_revenue.query('base_name == "xr_revenue_non_ag"')
    get_map_obj(data, revenue_nonag, f'{SAVE_DIR}/map_metrics/revenue_NonAg.json')

    ####################################################
    #                7) Transition Cost                #
    ####################################################

    # files_transition = files.query('base_name.str.contains("transition")')

    # transition_cost = files_transition.query('base_name == "xr_transition_cost_ag2non_ag"')
    # get_map_obj(data, transition_cost, f'{SAVE_DIR}/map_metrics/transition_cost.json')

    # transition_ghg = files_transition.query('base_name == "xr_transition_GHG"')
    # get_map_obj(data, transition_ghg, f'{SAVE_DIR}/map_metrics/transition_GHG.json')

    ####################################################
    #                8) Water Yield                    #
    ####################################################

    files_water = files.query('base_name.str.contains("water_yield")')

    water_ag = files_water.query('base_name == "xr_water_yield_ag"')
    get_map_obj(data, water_ag, f'{SAVE_DIR}/map_metrics/water_yield_Ag.json')

    water_am = files_water.query('base_name == "xr_water_yield_ag_management"')
    get_map_obj(data, water_am, f'{SAVE_DIR}/map_metrics/water_yield_Am.json')

    water_nonag = files_water.query('base_name == "xr_water_yield_non_ag"')
    get_map_obj(data, water_nonag, f'{SAVE_DIR}/map_metrics/water_yield_NonAg.json')