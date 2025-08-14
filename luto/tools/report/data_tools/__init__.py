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
import pandas as pd

from luto.tools.report.data_tools.parameters import YR_BASE


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
        print(f"Unknown files found: {unknown_files['path'].tolist()}")
        
    # Remove rows with category = 'Unknown'
    file_paths = file_paths.query('category != "Unknown"').reset_index(drop=True)
    

    return file_paths


