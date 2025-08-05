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
from luto.tools.Manual_jupyter_books.helpers import arr_to_xr
from luto.tools.report.map_tools import hex_color_to_numeric



def array_to_base64(arr_4band: np.ndarray, bbox: list) -> dict:

    # Create PIL Image from RGBA array
    image = Image.fromarray(arr_4band, 'RGBA')
    
    # Convert to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        'img_str': 'data:image/png;base64,' + img_str,
        'bbox': bbox,
        'width': arr_4band.shape[1],
        'height': arr_4band.shape[0]
    }



def map2base64(data:Data, arr_lyr:xr.DataArray, attrs:dict) -> dict|None:

        arr_lyr = arr_lyr.compute()
        
        # Skip if the layer is empty
        if arr_lyr.sum() == 0:
            return

        # Convert the 1D array to a 2D array
        arr_lyr = arr_to_xr(data, arr_lyr)
        bbox = arr_lyr.rio.bounds()
        arr_lyr = arr_lyr.rio.reproject('EPSG:3857') # To Mercator with Nearest Neighbour

        # Normalize the layer
        positive_mask = (arr_lyr > 0).values
        min_val = 0 if  arr_lyr.values[positive_mask].min() < 1e-3 else arr_lyr.values[positive_mask].min()
        max_val = arr_lyr.values[positive_mask].max()
        arr_lyr.values[positive_mask] = (arr_lyr.values[positive_mask] - min_val) / (max_val - min_val)
        
        # Convert layer to integer; after this 0 is nodata, -100 is outside LUTO area
        arr_lyr.values *= 100
        arr_lyr = np.where(np.isnan(arr_lyr), 0, arr_lyr).astype('int32')

        # Convert the 1D array to a RGBA array
        color_dict = pd.read_csv('luto/tools/report/Assets/float_img_colors.csv')
        color_dict['color_numeric'] = color_dict['lu_color_HEX'].apply(hex_color_to_numeric)
        color_dict = color_dict.set_index('lu_code')['color_numeric'].to_dict()
                    
        arr_4band = np.zeros((arr_lyr.shape[0], arr_lyr.shape[1], 4), dtype='uint8')
        for k, v in color_dict.items():
            arr_4band[arr_lyr == k] = v

        # Generate base64 and overlay info
        return attrs, array_to_base64(arr_4band, bbox)
    
    
    
def get_map_obj(data:Data, files_df:pd.DataFrame, save_path:str) -> dict:
        
        # Loop through each year
        task = []
        for _,row in files_df.iterrows():
            
            # Keep the cell dimension as the only chunked dimension
            xr_arr = xr.open_dataarray(row.path)
            chunk_size = {dim:1 for dim in xr_arr.dims if dim != 'cell'}
            chunk_size.update({'cell': data.NCELLS})  
            xr_arr = xr_arr.chunk(chunk_size)
            _year = row['Year']
            
            # Permute the selections
            loop_dims = set(xr_arr.dims) - set(['cell'])
            
            dim_vals = pd.MultiIndex.from_product(
                [xr_arr[dim].values for dim in loop_dims],
                names=loop_dims
            ).to_list()

            loop_sel = [dict(zip(loop_dims, val)) for val in dim_vals]

            # Parallel processing to convert each map to base64
            for sel in loop_sel:
                arr_sel = xr_arr.sel(**sel)
                task.append(
                    delayed(map2base64)(data, arr_sel, {'key':"_".join(sel.values()), 'year': _year})
                )

        # Gather results and save to JSON
        results = Parallel(n_jobs=-1)(task)
        results = [res for res in results if res is not None]
        
        output = {}
        for attr, val in results:
            output.setdefault(attr['key'], {})
            output[attr['key']][attr['year']] = val
            
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)



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
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir).query('category == "xarray_layer"')
    files['Year'] = files['Year'].astype(int)
    files = files.query('Year.isin(@years)')
    
    
    
    ####################################################
    #                   1) Area Layer                  #
    ####################################################
    
    files_area = files.query('base_name.str.contains("area")')

    area_ag = files_area.query(f'base_name.str.contains("agricultural_landuse")')
    get_map_obj(data, area_ag, f'{SAVE_DIR}/map_metrics/area_Ag.json')
    
    area_am = files_area.query(f'base_name.str.contains("agricultural_management")')
    get_map_obj(data, area_am, f'{SAVE_DIR}/map_metrics/area_Am.json')
    
    area_nonag = files_area.query(f'base_name.str.contains("non_agricultural_landuse")')
    get_map_obj(data, area_nonag, f'{SAVE_DIR}/map_metrics/area_NonAg.json')


