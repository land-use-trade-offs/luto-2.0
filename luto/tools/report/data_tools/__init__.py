import sys
import os
import re
import pandas as pd

from luto.tools.report.data_tools.parameters import LU_CROPS, LU_LVSTKS, YR_BASE, COMMODITIES_OFF_LAND, COMMODITIES_ALL
from luto.tools.report.data_tools.helper_func import df_wide2long, get_GHG_category, merge_LVSTK_UAALLOW

# Set up working directory to the root of the report folder
if __name__ == '__main__':
    os.chdir('..')

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
            'dvar':['npy'],
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
            # Maps
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
            # Registry of chekcing the start of the base name
            reg = re.compile(fr'^{pat}')
            search_result.append(bool(reg.search(base_name)))
        
        # If any of the patterns are found, break the loop
        if any(search_result): break


    # Check if this comes from the begin_end_compare folder
    if 'begin_end_compare' in path:
        yr_type = 'begin_end_year'
    else:
        yr_type = 'single_year'
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
    for foldername, subfolders, filenames in os.walk(data_root):
        for filename in filenames:
            # Create the full path to the file by joining the foldername and filename
            file_path = os.path.join(foldername, filename)
            # Append the file path to the list
            file_paths.append(file_path)

    # remove log files and sort the files
    file_paths = sorted([i for i in file_paths if 'out_' in i])

    # Get the year and the run number from the file name
    file_paths = pd.DataFrame({'path':file_paths})
    file_paths.insert(0, 'year', [re.compile(r'out_(\d{4})').findall(i)[0] for i in file_paths['path']])

    yr_types, f_cats = zip(*[extract_dtype_from_path(i) for i in file_paths['path']])
    file_paths.insert(1, 'year_types', yr_types)
    file_paths.insert(2, 'category', f_cats)

    file_paths[['base_name','base_ext']] = [os.path.splitext(os.path.basename(i)) for i in file_paths['path']]
    file_paths = file_paths.reindex(columns=['year','year_types','category','base_name','base_ext','path'])

    file_paths['year'] = file_paths['year'].astype(int)
    
    # remove the datatime stamp from the base_name
    file_paths['base_name'] = file_paths['base_name'].apply(lambda x: re.sub(r'_\d{4}.*\d{2}','',x))

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
        df_this_yr['year'] = row['year']

        # check if this is the first row, then get the df for base year
        if idx == 0:
            df_base_yr = df[['Commodity','Prod_base_year (tonnes, KL)']].copy()         
                                                                                        
            # add the year to the df
            df_base_yr['year'] = YR_BASE
            
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


def get_ag_rev_cost_df(files_df:pd.DataFrame,in_type:str):
    
    """
    Given a DataFrame containing information about files, this function returns a DataFrame of revenue or cost (in billion dollars) for each year in the input DataFrame.

    Parameters:
    files_df (pandas.DataFrame): A DataFrame containing at least the columns 'year' and 'path', where 'year' is the year of the data and 'path' is the file path to the data file.
    in_type (str): Either 'Revenue' or 'Cost', depending on which type of data to return.

    Returns:
    pandas.DataFrame: A DataFrame of revenue or cost (in billion dollars) for each year in the input DataFrame.
    """
    
    # in_type = 'Revenue' or 'Cost', otherwise raise error
    if in_type not in ['revenue','cost']:
        raise ValueError('in_type must be either "revenue" or "cost"')
    
    path_df = files_df.query('category == @in_type and year_types == "single_year"').reset_index(drop=True) 
    
    # Remove the non-ag and ag-mam from the path_df
    remove_list = ['cost_non_ag', 
                   'cost_agricultural_management',
                   'revenue_non_ag', 
                   'revenue_agricultural_management']
    path_df = path_df[~path_df['base_name'].str.contains('|'.join(remove_list))]
       
    dfs =[]  
    for _,row in path_df.iterrows():
        df = df_wide2long(row['path'])
        # insert year as a column
        df.insert(0, 'year', row['year'])
        dfs.append(df)
    
    # concatenate all the dfs, remove rows with value of 0    
    out_df = pd.concat(dfs).reset_index(drop=True)
    out_df = out_df.query('`value (billion)` != 0')
    
    # sort the df by year, Source
    out_df = out_df.sort_values(['year','Source']).reset_index(drop=True)
    
    # merge the Source and Type columns
    out_df['Source_type'] = out_df.apply(lambda x: x['Source'] + ' ' + x['Type'], axis=1)
    
    # manually change the Source_type to remove "Revenue" from "Crop Revenue", and change "Dairy Milk" to "Dairy"
    out_df['Source_type'] = out_df['Source_type'].str.replace(' Revenue','')
    out_df['Source_type'] = out_df['Source_type'].str.replace('Dairy  Milk','Dairy')
    
    # add the column to show if this is a corp or livestock
    lvstk = [i.split(' - ')[0] for i in LU_LVSTKS]
    out_df['crop_lvstk'] = out_df['Source'].apply(lambda x: 'Crop' if any(x == s for s in LU_CROPS) else 'Livestock')

    return out_df



def get_AREA_lu(df):
    """
    Returns a pandas DataFrame containing processed data on land use and area (in km2) for each year in the input DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing at least the columns 'year' and 'path', where 'year' is the year of the data and 'path' is the file path to the data file.

    Returns:
    pandas.DataFrame: A DataFrame containing processed data on land use and area (in km2) for each year in the input DataFrame.
    """
    area_df = []
    for idx, (_, row) in enumerate(df.iterrows()):
        year = row['year']
        file_path = row['path']
        df = pd.read_csv(file_path, index_col=0)
        column_names = ['Land use', 'Area (km2)']
        
        # Function to process a single row
        if idx == 0:
            # Process the first year
            data_row = df.iloc[:, -1].reset_index().query('(index != "Total") & (index != "All")')
            data_row.columns = column_names
            data_row.insert(0, 'Year', YR_BASE)
            data_row = merge_LVSTK_UAALLOW(data_row)
            area_df.append(data_row)
            
            # Process the last row for the year
            data_row = df.iloc[-1, :].reset_index().query('(index != "Total") & (index != "All")')
            data_row.columns = column_names
            data_row.insert(0, 'Year', year)
            data_row = merge_LVSTK_UAALLOW(data_row)  
            area_df.append(data_row)
            
        else:
            # Process the last row for the year
            data_row = df.iloc[-1, :].reset_index().query('(index != "Total") & (index != "All")')
            data_row.columns = column_names
            data_row.insert(0, 'Year', year)
            data_row = merge_LVSTK_UAALLOW(data_row)
            area_df.append(data_row)

    return pd.concat(area_df).reset_index(drop=True)

def get_ag_dvar_area(area_dvar_paths):

    ag_dvar_dfs = area_dvar_paths.query('base_name.str.contains("area_agricultural_landuse")').reset_index(drop=True)
    ag_dvar_area = pd.concat([pd.read_csv(path) for path in ag_dvar_dfs['path']], ignore_index=True)
    # ag_dvar_area = merge_LVSTK_UAALLOW(ag_dvar_area)
    ag_dvar_area['Area (million km2)'] = ag_dvar_area['Area (ha)'] / 100 / 1e6
    ag_dvar_area['Type'] = 'Agricultural landuse'

    non_ag_dvar_dfs = area_dvar_paths.query('base_name.str.contains("area_non_agricultural_landuse")').reset_index(drop=True)
    non_ag_dvar_area = pd.concat([pd.read_csv(path) for path in non_ag_dvar_dfs['path']], ignore_index=True)
    non_ag_dvar_area['Area (million km2)'] = non_ag_dvar_area['Area (ha)'] / 100 / 1e6
    non_ag_dvar_area['Type'] = 'Non-agricultural landuse'

    area_dvar = pd.concat([ag_dvar_area, non_ag_dvar_area], ignore_index=True) 
    
    return area_dvar

def get_AREA_lm(df):
    """
    Processes a DataFrame of irrigation data and returns a concatenated DataFrame of processed data.

    Args:
        df (pandas.DataFrame): A DataFrame containing irrigation data.

    Returns:
        pandas.DataFrame: A concatenated DataFrame of processed data.
    """
    area_df = []
    for idx, (_, row) in enumerate(df.iterrows()):
        year = row['year']
        file_path = row['path']
        df = pd.read_csv(file_path, index_col=0)
        
        column_names = ['Irrigation', 'Area (km2)']

        # Define a processing function for irrigation data
        def process_irrigation(row):
            return row.replace({'0': 'Dryland', '1': 'Irrigated'})
        
        # Function to process a single row
        if idx == 0:
            # Process the first year
            data_row = df.iloc[:, -1].reset_index().query('(index != "Total") & (index != "All")')
            data_row.columns = column_names
            data_row = process_irrigation(data_row)
            data_row.insert(0, 'Year', YR_BASE)
            area_df.append(data_row)
            
            # Process the last row for the year
            data_row = df.iloc[-1, :].reset_index().query('(index != "Total") & (index != "All")')
            data_row.columns = column_names
            data_row = process_irrigation(data_row)
            data_row.insert(0, 'Year', year)
            area_df.append(data_row)
        else:
            # Process the last row for the year
            data_row = df.iloc[-1, :].reset_index().query('(index != "Total") & (index != "All")')
            data_row.columns = column_names
            data_row = process_irrigation(data_row)
            data_row.insert(0, 'Year', year)    
            area_df.append(data_row)

    return pd.concat(area_df).reset_index(drop=True)



def get_AREA_am(df):
    """
    Given a pandas DataFrame `df` containing information about land use and area in different years,
    this function returns a new DataFrame containing the area of each land use category in each year.
    The input DataFrame `df` should have the following columns:
    - 'path': the file path of the data file
    - 'year': the year of the data
    - 'Area prior [ km2 ]': the area of each land use category before any changes
    - 'Area after [ km2 ]': the area of each land use category after any changes
    - 'Land use': the name of each land use category
    """
    
    # read all the switches
    switches = []
    for idx,(_,row) in enumerate(df.iterrows()):

        path = row['path']
        year = row['year']
        file_path = row['path']
        df = pd.read_csv(file_path,index_col=0)

        # check if this is the first row
        if idx == 0:
            first_year = df[['Area prior [ km2 ]']].reset_index(names='Land use').query('`Land use` != "Total"')
            first_year.columns = ['Land use','Area (km2)']
            first_year.insert(0,'Year',YR_BASE)
            switches.append(first_year)
        
        # get the last row, which is the area in the year
        last_row = df[['Area after [ km2 ]']].reset_index(names='Land use').query('`Land use` != "Total"')
        last_row.columns = ['Land use','Area (km2)']
        last_row.insert(0,'Year',year)
        switches.append(last_row)

    return pd.concat(switches).reset_index(drop=True)



def get_begin_end_df(files,timeseries = False):
    
    """
    Given a DataFrame containing information about file paths, this function returns two DataFrames:
    - begin_end_df_area: the area (in km2) of each land use category in each year
    - begin_end_df_pct: the percentage change of each land use category in each year
    The input DataFrame `files` should have the following columns:
    - 'path': the file path of the data file
    - 'year': the year of the data
    - 'year_types': the year type of the data
    - 'base_name': the base name of the data file
    - 'base_ext': the base extension of the data file
    """
    
    # read the cross table between start and end year
    if timeseries:
        begin_end_df = files.query('''year_types == "begin_end_year" and base_name.str.contains("crosstab-lumap")''').reset_index(drop=True)
    else:
        begin_end_df = files.query(''' base_name.str.contains("crosstab-lumap")''').reset_index(drop=True)
    
    begin_end_df = pd.read_csv(begin_end_df['path'][0], index_col=0)

    # get the total area in the begin year
    area_begin = begin_end_df['Total [km2]']

    # remove the last row and column
    begin_end_df_area = begin_end_df.iloc[:-1,:-1]

    # divide the begain area so we get the percentage change
    begin_end_df_pct = begin_end_df_area.divide(area_begin, axis=0) * 100

    # fill nan with 0
    begin_end_df_pct = begin_end_df_pct.fillna(0)
    
    return begin_end_df_area, begin_end_df_pct


def get_GHG_emissions_by_crop_lvstk_df(GHG_emissions_long):
    GHG_crop_lvstk_total = GHG_emissions_long.groupby(['Year','Land use category','Land category']).sum()['Quantity (Mt CO2e)'].reset_index()
    GHG_crop_lvstk_total['Landuse_land_cat'] = GHG_crop_lvstk_total.apply(lambda x: (x['Land use category'] + ' - ' + x['Land category']) 
                                    if (x['Land use category'] != x['Land category']) else x['Land use category'], axis=1)
    
    return GHG_crop_lvstk_total




def get_water_df(water_dfs):
    dfs = []
    for _,row in water_dfs.iterrows():
        df = pd.read_csv(row['path'], index_col=0).reset_index(drop=True)
        # insert year column
        df.insert(0, 'year', row['year'])
        df['TOT_WATER_REQ_ML'] = df['TOT_WATER_REQ_ML'].str.replace(',','').astype(float)
        df['WATER_USE_LIMIT_ML'] = df['WATER_USE_LIMIT_ML'].str.replace(',','').astype(float)
        df['ABS_DIFF_ML'] = df['ABS_DIFF_ML'].str.replace(',','').astype(float)
        dfs.append(df)

    return pd.concat(dfs, axis=0).reset_index(drop=True)