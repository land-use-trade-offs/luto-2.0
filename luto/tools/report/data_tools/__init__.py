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
from luto.tools.report.data_tools.parameters import RENAME_AM_NON_AG


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
            # CSVs
            'GHG':['GHG'],
            'water':['water'],
            'cross_table':['crosstab','switches'],
            'area':['area'],
            'transition_matrix':['transition_matrix'],
            'quantity':['quantity'],
            'economics_ag':['economics_ag_'],
            'economics_am':['economics_am_'],
            'economics_non_ag':['economics_non_ag_'],
            'transition':['transition_ag2ag_', 'transition_ag2nonag_', 'transition_nonag2ag_'],
            'biodiversity':['biodiversity'],
            'renewable':['renewable_energy_'],
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
        
    # 3: 'species' or 'group' (biodiversity GBF4/GBF8 — top level in hierarchy)
    if 'species' in sel:
        sel_rename['species'] = sel['species']
    if 'group' in sel:
        sel_rename['group'] = sel['group']
        
    # 4-1: Commodity dimensions
    if 'Commodity' in sel:
        commodity = sel['Commodity'].capitalize()
        sel_rename['Commodity'] = {
            'All': 'ALL',
            'Sheep lexp': 'Sheep live export',
            'Beef lexp': 'Beef live export'
        }.get(commodity, commodity)
        
    # 4-2: Profit/Revenue/Cost/Source
    leftover_keys = set(sel.keys()) - set(sel_rename.keys()) - {'lu', 'from_lu', 'species', 'group'}
    for key in leftover_keys:
        sel_rename[key] = {
            'Operation-cost': 'Cost (operation)',
            'Transition-cost-ag2ag': 'Cost (trans Ag2Ag)',
            'Transition-cost-ag2nonag': 'Cost (trans Ag2NonAg)',
            'Transition-cost-nonag2ag': 'Cost (trans NonAg2Ag)',
            'Transition-cost-agMgt': 'Cost (trans AgMgt)',
        }.get(sel[key], sel[key])

    # 5 last: 'lu' or 'from_lu' (treated identically as the land-use selection level)
    if 'lu' in sel:
        sel_rename['lu'] = RENAME_AM_NON_AG.get(sel['lu'], sel['lu'])
    elif 'from_lu' in sel:
        sel_rename['from_lu'] = RENAME_AM_NON_AG.get(sel['from_lu'], sel['from_lu'])

    return sel_rename


def build_map_legend(code_dict) -> dict:
    """Build map legend dicts from the Python color sub-dicts in parameters.py.

    Input: code_dict is a dict of the form {code: (description, hex_color)}, e.g. COLOR_AG.
    Output: dict with two sub-dicts:
     - 'legend': {description: hex_color} for legend rendering
     - 'code_colors': {code: (R, G, B, A)} for pixel rendering, with nodata (-1) as transparent (0, 0, 0, 0).
    """
    # Items disabled in settings are excluded from legend and pixel rendering
    rm_items = set(
        [lu for lu, enabled in settings.NON_AG_LAND_USES.items() if not enabled] +
        [am for am, enabled in settings.AG_MANAGEMENTS.items()    if not enabled]
    )

    legend = {}
    code_colors = {}
    for code, (desc, hex_c) in code_dict.items():
        if desc in rm_items:
            continue
        legend[desc] = hex_c
        code_colors[code] = hex_color_to_numeric(hex_c)
    code_colors[-1] = (0, 0, 0, 0)  # nodata / outside study area → transparent
    return {'legend': legend, 'code_colors': code_colors}


    

