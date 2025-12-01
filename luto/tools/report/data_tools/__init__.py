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
import base64
import numpy as np
import pandas as pd

from io import BytesIO
from PIL import Image

from luto import settings
from luto.tools.report.data_tools.parameters import RENAME_AM_NON_AG, YR_BASE


def extract_dtype_from_path(path):
    """
    Extracts the data type and year type from a given file path.

    Args:
        path (str): The file path.

    Returns
        tuple: A tuple containing the year type and data type extracted from the file path.
    """
    # Define the output categories and its corresponding file patterns
    f_cat = {
            # decision variables (npy files)
            'ag_X_mrj':['ag_X_mrj'],
            'ag_man_X_mrj':['ag_man_X_mrj'],
            'non_ag_X_rk':['non_ag_X_rk'],
            # CSVs
            'GHG':['GHG'],
            'water':['water'],
            'cross_table':['crosstab','switches'],
            'area':['area'],
            'transition_matrix':['transition_matrix'],
            'quantity':['quantity'],
            'revenue':['revenue'],
            'cost':['cost'],
            'profit':['profit'],
            'biodiversity':['biodiversity'],
            # Maps (GeoTIFFs)
            'ammap':['ammap'],
            'lumap':['lumap'],
            'lmmap':['lmmap'],
            'non_ag':['non_ag'],
            'Ag_LU':['Ag_LU'], 
            'Ag_Mgt':['Ag_Mgt'],
            'Land_Mgt':['Land_Mgt'],
            'Non-Ag':['Non-Ag'],
            # Metrics xarrays
            'xarray_layer':['xr_'],
    }

    # Get the base name of the file path
    base_name = os.path.basename(path)

    # Check the file type
    for ftype, fpat in f_cat.items():

        search_result = []
        for pat in fpat:
            # Registry to check the start of the base name
            reg = re.compile(fr'^{pat}')
            search_result.append(bool(reg.search(base_name)))

        # If any of the patterns are found, break the loop
        if any(search_result): 
            break
        else:
            ftype = 'Unknown'

    return ftype



def get_all_files(data_root):
    """
    Retrieve a list of file paths from the specified data root directory.

    Args:
        data_root (str): The root directory to search for files.

    Returns
        pandas.DataFrame: A DataFrame containing the file paths, along with 
        additional columns for year, year types, category, base name, and 
        base extension.
    """
    file_paths = []

    # Walk through the folder and its subfolders
    for foldername, _, filenames in os.walk(data_root):
        for filename in filenames:
            # Create the full path to the file by joining the foldername and filename
            file_path = os.path.join(foldername, filename)
            # Append the file path to the list
            file_paths.append(file_path)

    # Only filepath containing "out_" are valid paths
    file_paths = sorted([i for i in file_paths if 'out_' in i])

    # Get the year from the file name
    file_paths = pd.DataFrame({'path':file_paths})
    file_paths.insert(0, 'Year', [re.compile(r'out_(\d{4})').findall(i)[0] for i in file_paths['path']])

    # Try to get the year type and category from the file path
    f_cats = [extract_dtype_from_path(i) for i in file_paths['path']]

    # Append the year type and category to the file paths
    file_paths.insert(1, 'category', f_cats)

    # Get the base name and extension of the file path
    file_paths[['base_name','base_ext']] = [os.path.splitext(os.path.basename(i)) for i in file_paths['path']]
    file_paths = file_paths.reindex(columns=['Year','category','base_name','base_ext','path'])

    # Remove the datatime stamp <YYYY_MM_DD__HH_mm_SS> from the base_name
    file_paths['base_name'] = file_paths['base_name'].apply(lambda x: re.sub(r'_\d{4}','',x))
    
    # Report the unknown files
    unknown_files = file_paths.query('category == "Unknown"')
    if not unknown_files.empty:
        print(f"Unknown files found: \n\t{'\n\t'.join(unknown_files['path'].tolist())}")
        
    # Remove rows with category = 'Unknown'
    file_paths = file_paths.query('category != "Unknown"').reset_index(drop=True)
    

    return file_paths


def array_to_base64(arr_4band: np.ndarray) -> dict:
    image = Image.fromarray(arr_4band, 'RGBA')
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {'img_str': 'data:image/png;base64,' + img_str}


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
        
        
def hex_color_to_numeric(hex: str) -> tuple:
    hex = hex.lstrip('#')
    if len(hex) == 6:
        hex = hex + 'FF'  # Add full opacity if alpha is not provided
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4, 6))


def rename_reorder_hierarchy(sel: dict) -> dict:
    '''
    Rename and reorder the dimensions. The order is finally used as the hierarchy in the map JSON files.
    
    The order is:
    1. 'am' (Agricultural Management)
    2. 'lm' (Water Supply)
    3. Other dimensions:
        - such as "Source" for GHG, or 'Type' for economic data.
        - Or profit/revenue/cost data.
    4. 'lu' (Land-use) has to be the lowest level in the hierarchy.
    '''
    sel_rename = {}
    # 1: 'am'
    if 'am' in sel:
        sel_rename['am'] = RENAME_AM_NON_AG.get(sel['am'], sel['am'])
    # 2: 'lm'
    if 'lm' in sel:
        sel_rename['lm'] =  {'irr': 'Irrigated', 'dry': 'Dryland'}.get(sel['lm'], sel['lm'])
        
    # 3-1: Commodity dimensions
    if 'Commodity' in sel:
        commodity = sel['Commodity'].capitalize()
        sel_rename['Commodity'] = {
            'All': 'ALL',
            'Sheep lexp': 'Sheep live export',
            'Beef lexp': 'Beef live export'
        }.get(commodity, commodity)

    # 3-2: Profit/Revenue/Cost
    leftover_keys = set(sel.keys()) - set(sel_rename.keys()) - {'lu'}
    for key in leftover_keys:
        sel_rename[key] = {
            'Operation-cost': 'Cost (operation)',
            'Transition-cost-ag2ag': 'Cost (trans Ag2Ag)',
            'Transition-cost-ag2nonag': 'Cost (trans Ag2NonAg)',
            'Transition-cost-nonag2ag': 'Cost (trans NonAg2Ag)',
            'Transition-cost-agMgt': 'Cost (trans AgMgt)',
        }.get(sel[key], sel[key])
        
    # 4 last: 'lu'
    if 'lu' in sel:
        sel_rename['lu'] = RENAME_AM_NON_AG.get(sel['lu'], sel['lu'])
        
        
    return sel_rename


def get_map_legend() -> dict:

    color_csvs = {
        'lumap': 'luto/tools/report/VUE_modules/assets/lumap_colors_grouped.csv',
        'lm': 'luto/tools/report/VUE_modules/assets/lm_colors.csv',
        'ag': 'luto/tools/report/VUE_modules/assets/lumap_colors.csv',
        'non_ag': 'luto/tools/report/VUE_modules/assets/non_ag_colors.csv',
        'am': 'luto/tools/report/VUE_modules/assets/ammap_colors.csv',
        'float': 'luto/tools/report/VUE_modules/assets/float_img_colors.csv'
    }
    
    # Remove land-uses and ag managements that are not used, this excludes them from showing in the map legend
    rm_lus = [i for i in settings.NON_AG_LAND_USES if not settings.NON_AG_LAND_USES[i]]
    rm_ams = [i for i in settings.AG_MANAGEMENTS if not settings.AG_MANAGEMENTS[i]]
    rm_items = rm_lus + rm_ams
    
    return {
        'lumap': {
            'color_csv': color_csvs['lumap'], 
            'legend': {
                RENAME_AM_NON_AG.get(k,k):v for k,v in pd.read_csv(color_csvs['lumap']).set_index('lu_desc')['lu_color_HEX'].to_dict().items() 
                if k not in rm_items
            }
        },
        'lm': {
            'color_csv': color_csvs['lm'],
            'legend': pd.read_csv(color_csvs['lm']).set_index('lu_desc')['lu_color_HEX'].to_dict()
        },
        'ag': {
            'color_csv': color_csvs['ag'],
            'legend': pd.read_csv(color_csvs['ag']).set_index('lu_desc')['lu_color_HEX'].to_dict()
        },
        'non_ag': {
            'color_csv': color_csvs['non_ag'],
            'legend': {
                RENAME_AM_NON_AG.get(k,k):v for k,v in pd.read_csv(color_csvs['non_ag']).set_index('lu_desc')['lu_color_HEX'].to_dict().items() 
                if k not in rm_items
            }
        },
        'am': {
            'color_csv': color_csvs['am'],
            'legend': {
                RENAME_AM_NON_AG.get(k,k):v for k,v in pd.read_csv(color_csvs['am']).set_index('lu_desc')['lu_color_HEX'].to_dict().items() 
                if k not in rm_items
            }
        },
        'float': {
            'color_csv': color_csvs['float'],
            'legend': pd.read_csv(color_csvs['float']).set_index('lu_code')['lu_color_HEX'].to_dict()
        }
    }
    
    

