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
import base64
import numpy as np
import pandas as pd
import xarray as xr

from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

from luto import settings
from luto.data import Data
from luto.tools.report.data_tools import get_all_files
from luto.tools.Manual_jupyter_books.helpers import arr_to_xr
from luto.tools.report.map_tools import hex_color_to_numeric



def array_to_base64(arr_4band: np.ndarray, filename: str) -> dict:
    """
    Convert 4-band RGBA array to base64 PNG string
    
    Args:
        arr_4band: numpy array of shape (height, width, 4) with RGBA values
        filename: str, the name of the file to save the image as
    
    Returns:
        dict with base64 string and bbox info
    """
    # Create PIL Image from RGBA array
    image = Image.fromarray(arr_4band, 'RGBA')
    
    # Save to BytesIO buffer as PNG
    image.save(filename)



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
    from rasterio.enums import Resampling

    for _,row in files_area.iterrows():
        xr_arr = xr.load_dataarray(row.path)
        loop_dims = set(xr_arr.dims) - set(['cell'])
        xr_arr = xr_arr.stack(stacked_dim=loop_dims)
        
        for stack_dim in xr_arr['stacked_dim'].values:
            arr_lyr = arr_to_xr(data, xr_arr.sel(stacked_dim=stack_dim))
            bbox = arr_lyr.rio.bounds()
            arr_lyr = arr_lyr.rio.reproject('EPSG:3857') # To Mercator with Nearest Neighbour


            # Normalize the layer
            positive_mask = (arr_lyr > 0).values
            min_val = 0 if  arr_lyr.values[positive_mask].min() < 1e-3 else arr_lyr.values[positive_mask].min()
            max_val = arr_lyr.values[positive_mask].max()
            arr_lyr.values[positive_mask] = (arr_lyr.values[positive_mask] - min_val) / (max_val - min_val)
            
            # Convert layer to integer; after this 0 is nodata, -100 is outside LUTO area
            arr_lyr.values *= 100
            arr_lyr = np.where(np.isnan(arr_lyr), 0, arr_lyr)

            # Convert the 1D array to a RGBA array
            color_dict = pd.read_csv('luto/tools/report/Assets/float_img_colors.csv')
            color_dict['color_numeric'] = color_dict['lu_color_HEX'].apply(hex_color_to_numeric)
            color_dict = color_dict.set_index('lu_code')['color_numeric'].to_dict()
                        
            arr_4band = np.zeros((arr_lyr.shape[0], arr_lyr.shape[1], 4), dtype='uint8')
            for k, v in color_dict.items():
                arr_4band[arr_lyr == k] = v
                
                
            array_to_base64(arr_4band, f'{SAVE_DIR}/map_png/{('_').join(stack_dim)}_area.png')
            plt.imshow(arr_4band)
                
            

 
 
 
 
 
 
 