import os
import re
import pandas as pd

from luto.tools.report.data_tools.parameters import YR_BASE


def extract_dtype_from_path(path):
    """
    Extracts the data type and year type from a given file path.

    Args:
        path (str): The file path.

    Returns:
        tuple: A tuple containing the year type and data type extracted from the file path.
    """
    # Define the output categories and its corresponding file patterns
    f_cat = {
            # decision variables (npy files)
            'ag_X_mrj':['ag_X_mrj'],
            'ag_man_X_mrj':['ag_man_X_mrj_asparagopsis_taxiformis',
                            'ag_man_X_mrj_ecological_grazing',
                            'ag_man_X_mrj_precision_agriculture',
                            'ag_man_X_mrj_savanna_burning',
                            'ag_man_X_mrj_agtech_ei'],
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


    # Check if this comes from the begin_end_compare folder
    yr_type = 'begin_end_year' if 'begin_end_compare' in path else 'single_year'
    return yr_type, ftype



def get_all_files(data_root):
    """
    Retrieve a list of file paths from the specified data root directory.

    Args:
        data_root (str): The root directory to search for files.

    Returns:
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
    yr_types, f_cats = zip(*[extract_dtype_from_path(i) for i in file_paths['path']])


    # Append the year type and category to the file paths
    file_paths.insert(1, 'year_types', yr_types)
    file_paths.insert(2, 'category', f_cats)

    # Get the base name and extension of the file path
    file_paths[['base_name','base_ext']] = [os.path.splitext(os.path.basename(i)) for i in file_paths['path']]
    file_paths = file_paths.reindex(columns=['Year','year_types','category','base_name','base_ext','path'])
    
    # Remove the datatime stamp <YYYY_MM_DD__HH_mm_SS> from the base_name
    file_paths['base_name'] = file_paths['base_name'].apply(lambda x: re.sub(r'_\d{4}','',x))
    
    # Report the unknown files
    unknown_files = file_paths.query('category == "Unknown"')
    if not unknown_files.empty:
        print(f"Unknown files found: {unknown_files['path'].tolist()}")
        
    # Remove rows with category = 'Unknown'
    file_paths = file_paths.query('category != "Unknown"').reset_index(drop=True)
    

    return file_paths



def get_quantity_df(in_dfs):
    """
    Concatenates and transforms a list of dataframes into a single dataframe with quantity information.

    Args:
        in_dfs (pandas.DataFrame): A dataframe containing information about input dataframes.

    Returns:
        pandas.DataFrame: A dataframe with concatenated and transformed quantity information.

    """
    all_dfs = []
    for idx,row in in_dfs.iterrows():

        # read the df
        df = pd.read_csv(row['path'])

        # get the df for the year of the loop
        df_this_yr = df[['Commodity','Prod_targ_year (tonnes, KL)']].copy()             
        df_this_yr['Year'] = row['Year']

        # check if this is the first row, then get the df for base year
        if idx == 0:
            df_base_yr = df[['Commodity','Prod_base_year (tonnes, KL)']].copy()         
                                                                                        
            # add the year to the df
            df_base_yr['Year'] = YR_BASE
            
            # reanme the columns so it can be concatenated with df_this_yr
            df_base_yr.columns = df_this_yr.columns
            
            all_dfs.append(df_base_yr)

        # append df
        all_dfs.append(df_this_yr)


    # concatenate all the dfs, and made a column with the unit of Million tonnes
    all_df = pd.concat(all_dfs).reset_index(drop=True)
    all_df['Prod_targ_year (tonnes, ML)'] = all_df['Prod_targ_year (tonnes, KL)']/1e6

    # rename the commodity column so that the first letter is capitalised
    all_df['Commodity'] = all_df['Commodity'].str.capitalize()
    
    return all_df



