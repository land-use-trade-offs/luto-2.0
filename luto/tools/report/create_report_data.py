
import os
import json
import re
import shutil
import pandas as pd
import numpy as np
import luto.settings as settings
from joblib import Parallel, delayed

from luto.economics.off_land_commodity import get_demand_df
from luto.tools.report.data_tools import   get_all_files, get_quantity_df        
from luto.tools.report.data_tools.helper_func import select_years

from luto.tools.report.data_tools.colors import LANDUSE_ALL_COLORS, COMMODITIES_ALL_COLORS                                                                             
from luto.tools.report.data_tools.parameters import (AG_LANDUSE, 
                                                     COMMODITIES_ALL,
                                                     COMMODITIES_OFF_LAND, 
                                                     GHG_CATEGORY, 
                                                     GHG_NAMES, 
                                                     LANDUSE_ALL,
                                                     LU_CROPS, 
                                                     LU_NATURAL,
                                                     LVSTK_MODIFIED, 
                                                     LVSTK_NATURAL, 
                                                     NON_AG_LANDUSE, 
                                                     RENAME_AM_NON_AG)


def save_report_data(raw_data_dir:str):
    """
    Saves the report data in the specified directory.

    Parameters:
    raw_data_dir (str): The directory path where the raw output data is.

    Returns:
    None
    """
    
    # Set the save directory    
    SAVE_DIR = f'{raw_data_dir}/DATA_REPORT/data'
    
    # Create the directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir)
    
    # Set the years to be int
    files['Year'] = files['Year'].astype(int)
    
    # Select the years to reduce the column number to 
    # avoid cluttering in the multi-level axis graphing
    years = sorted(files['Year'].unique().tolist())
    years_select = select_years(years)
    
    

    ####################################################
    #                    1) Area Change                #
    ####################################################

    area_dvar_paths = files.query('category == "area" and year_types == "single_year"').reset_index(drop=True)
    
    ag_dvar_dfs = area_dvar_paths.query('base_name.str.contains("area_agricultural_landuse")').reset_index(drop=True)
    ag_dvar_area = pd.concat([pd.read_csv(path) for path in ag_dvar_dfs['path']], ignore_index=True)
    ag_dvar_area['Area (million km2)'] = ag_dvar_area['Area (ha)'] / 100 / 1e6
    ag_dvar_area['Type'] = 'Agricultural landuse'

    non_ag_dvar_dfs = area_dvar_paths.query('base_name.str.contains("area_non_agricultural_landuse")').reset_index(drop=True)
    non_ag_dvar_area = pd.concat([pd.read_csv(path) for path in non_ag_dvar_dfs['path']], ignore_index=True)
    non_ag_dvar_area['Area (million km2)'] = non_ag_dvar_area['Area (ha)'] / 100 / 1e6
    non_ag_dvar_area['Type'] = 'Non-agricultural landuse'

    area_dvar = pd.concat([ag_dvar_area, non_ag_dvar_area], ignore_index=True)
    area_dvar = area_dvar.replace(RENAME_AM_NON_AG)

    # Plot_1-1: Total Area (km2)
    lu_area_dvar = area_dvar.groupby(['Year','Land-use']).sum(numeric_only=True).reset_index()
    lu_area_dvar = lu_area_dvar\
        .groupby('Land-use')[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    lu_area_dvar.columns = ['name','data']
    lu_area_dvar['type'] = 'column'
    lu_area_dvar['sort_index'] = lu_area_dvar['name'].apply(lambda x: LANDUSE_ALL.index(x))
    lu_area_dvar = lu_area_dvar.sort_values('sort_index').drop('sort_index',axis=1)
    lu_area_dvar['color'] = lu_area_dvar['name'].apply(lambda x: LANDUSE_ALL_COLORS[x])
    
    lu_area_dvar.to_json(f'{SAVE_DIR}/area_1_total_area_wide.json',orient='records')




    # Plot_1-2: Total Area (km2) by Water_supply
    lm_dvar_area = area_dvar.groupby(['Year','Water_supply']).sum(numeric_only=True).reset_index()
    lm_dvar_area['Water_supply'] = lm_dvar_area['Water_supply'].replace({'dry': 'Dryland', 'irr': 'Irrigated'})

    lm_dvar_area = lm_dvar_area\
        .groupby(['Water_supply'])[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    lm_dvar_area.columns = ['name','data']
    lm_dvar_area['type'] = 'column'
    lm_dvar_area.to_json(f'{SAVE_DIR}/area_2_Water_supply_area_wide.json',orient='records')




    # Plot_1-3: Area (km2) by Non-Agricultural land use
    non_ag_dvar_area = area_dvar.query('Type == "Non-agricultural landuse"').reset_index(drop=True)
    non_ag_dvar_area = non_ag_dvar_area\
        .groupby(['Land-use'])[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    non_ag_dvar_area.columns = ['name','data']
    non_ag_dvar_area['type'] = 'column'
    non_ag_dvar_area.to_json(f'{SAVE_DIR}/area_3_non_ag_lu_area_wide.json',orient='records')



    # Plot_1-4: Area (km2) by Agricultural management
    am_dvar_dfs = area_dvar_paths.query('base_name.str.contains("area_agricultural_management")').reset_index(drop=True)
    am_dvar_area = pd.concat([pd.read_csv(path) for path in am_dvar_dfs['path']], ignore_index=True)
    am_dvar_area['Area (million km2)'] = am_dvar_area['Area (ha)'] / 100 / 1e6
    am_dvar_area = am_dvar_area.replace(RENAME_AM_NON_AG)

    am_dvar_area_type = am_dvar_area.groupby(['Year','Type']).sum(numeric_only=True).reset_index()

    am_dvar_area_type = am_dvar_area_type\
        .groupby(['Type'])[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    am_dvar_area_type.columns = ['name','data']
    am_dvar_area_type['type'] = 'column'
    am_dvar_area_type.to_json(f'{SAVE_DIR}/area_4_am_total_area_wide.json',orient='records')



    # Plot_1-5: Agricultural management Area (km2) by Land use
    am_dvar_area_lu = am_dvar_area.groupby(['Year','Land-use']).sum(numeric_only=True).reset_index()

    am_dvar_area_lu = am_dvar_area_lu\
        .groupby(['Land-use'])[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    am_dvar_area_lu.columns = ['name','data']
    am_dvar_area_lu['type'] = 'column'
    am_dvar_area_lu['color'] = am_dvar_area_lu['name'].apply(lambda x: LANDUSE_ALL_COLORS[x])
    am_dvar_area_lu.to_json(f'{SAVE_DIR}/area_5_am_lu_area_wide.json',orient='records')


    # Plot_1-6/7: Area (km2) Transition by Land use
    transition_path = files.query('category =="transition_matrix"')
    transition_df_area = pd.read_csv(transition_path['path'].values[0], index_col=0).reset_index()
    transition_df_area['Area (km2)'] = transition_df_area['Area (ha)'] / 100   
    transition_df_area = transition_df_area.replace(RENAME_AM_NON_AG)

    # Get the total area of each land use
    transition_mat = transition_df_area.pivot(index='From land-use', columns='To land-use', values='Area (km2)')
    transition_mat = transition_mat.reindex(columns=LANDUSE_ALL, fill_value=0)
    total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
    
    # Calculate the percentage of each land use
    transition_df_pct = transition_mat / total_area_from * 100
    transition_df_pct = transition_df_pct.fillna(0).replace([np.inf, -np.inf], 0)

    # Add the total area to the transition matrix
    transition_mat['SUM'] = transition_mat.sum(axis=1)
    transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

    heat_area = transition_mat.style.background_gradient(cmap='Oranges', 
                                                            axis=1, 
                                                            subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]).format('{:,.0f}')
    
    heat_pct = transition_df_pct.style.background_gradient(cmap='Oranges', 
                                                        axis=1,
                                                        vmin=0, 
                                                        vmax=100).format('{:,.2f}')


    heat_area_html = heat_area.to_html()
    heat_pct_html = heat_pct.to_html()

    # Replace 0.00 with - in the html
    heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)
    heat_pct_html = re.sub(r'(?<!\d)0.00(?!\d)', '-', heat_pct_html)

    # Save the html
    with open(f'{SAVE_DIR}/area_6_begin_end_area.html', 'w') as f:
        f.write(heat_area_html)
        
    with open(f'{SAVE_DIR}/area_7_begin_end_pct.html', 'w') as f:
        f.write(heat_pct_html)


    ####################################################
    #                   2) Demand                      #
    ####################################################
    
    # Get the demand data
    DEMAND_DATA_long = get_demand_df()

    # Reorder the columns to match the order in COMMODITIES_ALL
    DEMAND_DATA_long = DEMAND_DATA_long.reindex(COMMODITIES_ALL, level=1).reset_index()
    DEMAND_DATA_long = DEMAND_DATA_long.replace(RENAME_AM_NON_AG)

    # Add columns for on-land and off-land commodities
    DEMAND_DATA_long['on_off_land'] = DEMAND_DATA_long['COMMODITY'].apply(
        lambda x: 'On-land' if x not in COMMODITIES_OFF_LAND else 'Off-land')

    # Convert quanlity to million tonnes
    DEMAND_DATA_long['Quantity (tonnes, ML)'] = DEMAND_DATA_long['Quantity (tonnes, ML)'] / 1e6
    DEMAND_DATA_long = DEMAND_DATA_long.query('Year.isin(@years)')
    DEMAND_DATA_long.loc[:,'COMMODITY'] = DEMAND_DATA_long['COMMODITY'].str.replace('Beef lexp','Beef live export')
    DEMAND_DATA_long_filter_year = DEMAND_DATA_long.query(f'Year.isin({years_select})')

    # Plot_2_1: {Total} for 'Domestic', 'Exports', 'Feed', 'Imports', 'Production'(Tonnes) 
    DEMAND_DATA_type = DEMAND_DATA_long_filter_year.groupby(['Year','Type']).sum(numeric_only=True).reset_index()
    DEMAND_DATA_type_wide = DEMAND_DATA_type\
                                .groupby('Type')[['Year','Quantity (tonnes, ML)']]\
                                .apply(lambda x:list(map(list,zip(x['Year'],x['Quantity (tonnes, ML)']))))\
                                .reset_index()  
                                
    DEMAND_DATA_type_wide.columns = ['name','data']
    DEMAND_DATA_type_wide['type'] = 'column'
                                
    DEMAND_DATA_type_wide.to_json(f'{SAVE_DIR}/production_1_demand_type_wide.json', orient='records')


    # Plot_2_2: {ON/OFF land} for 'Domestic', 'Exports', 'Feed', 'Imports', 'Production'(Tonnes) 
    DEMAND_DATA_on_off = DEMAND_DATA_long_filter_year.groupby(['Year','Type','on_off_land']).sum(numeric_only=True).reset_index()
    DEMAND_DATA_on_off = DEMAND_DATA_on_off.sort_values(['on_off_land','Year','Type'])
    
    DEMAND_DATA_on_off_series = DEMAND_DATA_on_off[['on_off_land','Quantity (tonnes, ML)']]\
                                .groupby('on_off_land')[['on_off_land','Quantity (tonnes, ML)']]\
                                .apply(lambda x: x['Quantity (tonnes, ML)'].tolist())\
                                .reset_index()
                                
    DEMAND_DATA_on_off_series.columns = ['name','data']
    DEMAND_DATA_on_off_series['type'] = 'column'
    
    
    DEMAND_DATA_on_off_categories = DEMAND_DATA_on_off.query('on_off_land == "On-land"')[['Year','Type']]\
                                    .groupby('Year')[['Year','Type']]\
                                    .apply(lambda x:x['Type'].tolist())\
                                    .reset_index()
                                    
    DEMAND_DATA_on_off_categories.columns = ['name','categories']
        
    
    
    # Combine the JSON objects into one dictionary
    combined_json = {
        'series': json.loads(DEMAND_DATA_on_off_series.to_json(orient='records')),
        'categories': json.loads(DEMAND_DATA_on_off_categories.to_json(orient='records'))
    }

    # Convert the dictionary to a JSON string
    combined_json_str = json.dumps(combined_json)  
    with open(f'{SAVE_DIR}/production_2_demand_on_off_wide.json', 'w') as outfile:
        outfile.write(combined_json_str)                          
        
        
    
    
    # Plot_2_3: {Commodity} 'Domestic', 'Exports', 'Feed', 'Imports', 'Production' (Tonnes)
    DEMAND_DATA_commodity = DEMAND_DATA_long_filter_year.groupby(['Year','Type','COMMODITY']).sum(numeric_only=True).reset_index()
    DEMAND_DATA_commodity = DEMAND_DATA_commodity.sort_values(['COMMODITY','Year','Type'])
    
    DEMAND_DATA_commodity_series = DEMAND_DATA_commodity[['COMMODITY', 'Quantity (tonnes, ML)']]\
                                    .groupby('COMMODITY')[['Quantity (tonnes, ML)']]\
                                    .apply(lambda x: x['Quantity (tonnes, ML)'].tolist())\
                                    .reset_index()
                                    
    DEMAND_DATA_commodity_series.columns = ['name','data']
    DEMAND_DATA_commodity_series['type'] = 'column' 
    DEMAND_DATA_commodity_series['color'] = DEMAND_DATA_commodity_series['name'].apply(lambda x: COMMODITIES_ALL_COLORS[x])
    
    DEMAND_DATA_commodity_series = DEMAND_DATA_commodity_series.set_index('name').reindex(COMMODITIES_ALL).reset_index()
    DEMAND_DATA_commodity_series = DEMAND_DATA_commodity_series.dropna()
    
    DEMAND_DATA_commodity_categories = DEMAND_DATA_commodity.query('COMMODITY == "Apples"')[['Year','Type']]\
                                        .groupby('Year')[['Year','Type']]\
                                        .apply(lambda x: x['Type'].tolist())\
                                        .reset_index()
                                        
    DEMAND_DATA_commodity_categories.columns = ['name','categories']
    
    combined_json = {   
        'series': json.loads(DEMAND_DATA_commodity_series.to_json(orient='records')),
        'categories': json.loads(DEMAND_DATA_commodity_categories.to_json(orient='records'))
    }
    
    combined_json_str = json.dumps(combined_json)
    with open(f'{SAVE_DIR}/production_3_demand_commodity.json', 'w') as outfile:
        outfile.write(combined_json_str)




    # Plot_2-4_(1-2): Domestic On/Off land commodities (Million Tonnes)
    for idx,on_off_land in enumerate(DEMAND_DATA_long['on_off_land'].unique()):
        DEMAND_DATA_on_off_commodity = DEMAND_DATA_long.query('on_off_land == @on_off_land and Type == "Domestic" ')
        
        DEMAND_DATA_on_off_commodity_wide = DEMAND_DATA_on_off_commodity\
            .groupby(['COMMODITY'])[['Year','Quantity (tonnes, ML)']]\
            .apply(lambda x: list(zip(x['Year'],x['Quantity (tonnes, ML)'])))\
            .reset_index()
            
        DEMAND_DATA_on_off_commodity_wide.columns = ['name','data']
        DEMAND_DATA_on_off_commodity_wide['type'] = 'column'
        DEMAND_DATA_on_off_commodity_wide['color'] = DEMAND_DATA_on_off_commodity_wide['name'].apply(lambda x: COMMODITIES_ALL_COLORS[x])

        DEMAND_DATA_on_off_commodity_wide.to_json(f'{SAVE_DIR}/production_4_{idx+1}_demand_domestic_{on_off_land}_commodity.json', orient='records')
        
        

    # Plot_2-5_(1-4): Commodities for 'Exports','Feed','Imports','Production' (Million Tonnes)
    for idx,Type in enumerate(DEMAND_DATA_long['Type'].unique()):
        if Type == 'Domestic':
            continue
        DEMAND_DATA_commodity = DEMAND_DATA_long.query('Type == @Type')
        DEMAND_DATA_commodity_wide = DEMAND_DATA_commodity\
            .groupby(['COMMODITY'])[['Year','Quantity (tonnes, ML)']]\
            .apply(lambda x: list(zip(x['Year'],x['Quantity (tonnes, ML)'])))\
            .reset_index()
            
        DEMAND_DATA_commodity_wide.columns = ['name','data']
        DEMAND_DATA_commodity_wide['type'] = 'column'
        DEMAND_DATA_commodity_wide['color'] = DEMAND_DATA_commodity_wide['name'].apply(lambda x: COMMODITIES_ALL_COLORS[x])
        
        DEMAND_DATA_commodity_wide = DEMAND_DATA_commodity_wide.set_index('name').reindex(COMMODITIES_ALL).reset_index()
        DEMAND_DATA_commodity_wide = DEMAND_DATA_commodity_wide.dropna()
        
        DEMAND_DATA_commodity_wide.to_json(f'{SAVE_DIR}/production_5_{idx+1}_demand_{Type}_commodity.json', orient='records')
        
    # Plot_2-5-6: Production (LUTO outputs, Million Tonnes)
    quantity_csv_paths = files.query('category == "quantity" and base_name == "quantity_comparison" and year_types == "single_year"').reset_index(drop=True)
    quantity_df = get_quantity_df(quantity_csv_paths)
    quantity_df['Commodity'] = quantity_df['Commodity'].str.replace('Beef lexp','Beef live export')
    
    quantity_df_wide = quantity_df\
        .groupby(['Commodity'])[['Year','Prod_targ_year (tonnes, ML)']]\
        .apply(lambda x: list(map(list,zip(x['Year'],x['Prod_targ_year (tonnes, ML)']))))\
        .reset_index()
        
    quantity_df_wide.columns = ['name','data']
    quantity_df_wide['type'] = 'column'
    quantity_df_wide['color'] = quantity_df_wide['name'].apply(lambda x: COMMODITIES_ALL_COLORS[x])
    
    quantity_df_wide = quantity_df_wide.set_index('name').reindex(COMMODITIES_ALL).reset_index()
    quantity_df_wide = quantity_df_wide.dropna()
    
    quantity_df_wide.to_json(f'{SAVE_DIR}/production_5_6_demand_Production_commodity_from_LUTO.json', orient='records')    





    ####################################################
    #                  3) Economics                    #
    ####################################################
    
    
    # Get the revenue and cost data
    revenue_ag_df = files.query('category == "revenue" and base_name == "revenue_agricultural_commodity" and year_types != "begin_end_year"').reset_index(drop=True)
    revenue_ag_df = pd.concat([pd.read_csv(path) for path in revenue_ag_df['path']], ignore_index=True)
    revenue_ag_df = revenue_ag_df.replace({'Revenue':'Crop'})
    revenue_ag_df['Value (billion)'] = revenue_ag_df['Value ($)'] / 1e9
    revenue_ag_df = revenue_ag_df.replace(RENAME_AM_NON_AG)
    
    revenue_am_df = files.query('category == "revenue" and base_name == "revenue_agricultural_management" and year_types != "begin_end_year"').reset_index(drop=True)
    revenue_am_df = pd.concat([pd.read_csv(path) for path in revenue_am_df['path']], ignore_index=True)
    revenue_am_df['Value (billion)'] = revenue_am_df['Value ($)'] / 1e9
    revenue_am_df = revenue_am_df.replace(RENAME_AM_NON_AG)
    
    revenue_non_ag_df = files.query('category == "revenue" and base_name == "revenue_non_ag" and year_types != "begin_end_year"').reset_index(drop=True)
    revenue_non_ag_df = pd.concat([pd.read_csv(path) for path in revenue_non_ag_df['path']], ignore_index=True)
    revenue_non_ag_df['Value (billion)'] = revenue_non_ag_df['Value ($)'] / 1e9
    revenue_non_ag_df = revenue_non_ag_df.replace(RENAME_AM_NON_AG)
    
    cost_ag_df = files.query('category == "cost" and base_name == "cost_agricultural_commodity" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_ag_df = pd.concat([pd.read_csv(path) for path in cost_ag_df['path']], ignore_index=True)
    cost_ag_df['Value (billion)'] = cost_ag_df['Value ($)'] * -1 / 1e9
    cost_ag_df = cost_ag_df.replace(RENAME_AM_NON_AG)

    cost_am_df = files.query('category == "cost" and base_name == "cost_agricultural_management" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_am_df = pd.concat([pd.read_csv(path) for path in cost_am_df['path']], ignore_index=True)
    cost_am_df['Value (billion)'] = cost_am_df['Value ($)'] * -1 / 1e9
    cost_am_df = cost_am_df.replace(RENAME_AM_NON_AG)
    
    cost_non_ag_df = files.query('category == "cost" and base_name == "cost_non_ag" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_non_ag_df = pd.concat([pd.read_csv(path) for path in cost_non_ag_df['path']], ignore_index=True)
    cost_non_ag_df['Value (billion)'] = cost_non_ag_df['Value ($)'] * -1 / 1e9
    cost_non_ag_df = cost_non_ag_df.replace(RENAME_AM_NON_AG)
    
    cost_transition_ag2ag_df = files.query('category == "cost" and base_name == "cost_transition_ag2ag" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_transition_ag2ag_df = pd.concat([pd.read_csv(path) for path in cost_transition_ag2ag_df['path']], ignore_index=True)
    cost_transition_ag2ag_df['Value (billion)'] = cost_transition_ag2ag_df['Cost ($)'] * -1 / 1e9
    cost_transition_ag2ag_df = cost_transition_ag2ag_df.replace(RENAME_AM_NON_AG)
    
    cost_transition_ag2non_ag_df = files.query('category == "cost" and base_name == "cost_transition_ag2non_ag" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_transition_ag2non_ag_df = pd.concat([pd.read_csv(path) for path in cost_transition_ag2non_ag_df['path']], ignore_index=True)
    cost_transition_ag2non_ag_df['Value (billion)'] = cost_transition_ag2non_ag_df['Cost ($)'] * -1 / 1e9
    cost_transition_ag2non_ag_df = cost_transition_ag2non_ag_df.replace(RENAME_AM_NON_AG)
    
    cost_transition_non_ag2ag_df = files.query('base_name == "cost_transition_non_ag2_ag" and year_types != "begin_end_year"')
    cost_transition_non_ag2ag_df = pd.concat([pd.read_csv(path) for path in cost_transition_non_ag2ag_df['path']], ignore_index=True)
    cost_transition_non_ag2ag_df['Value (billion)'] = cost_transition_non_ag2ag_df['Cost ($)'] * -1 / 1e9
    cost_transition_non_ag2ag_df = cost_transition_non_ag2ag_df.replace(RENAME_AM_NON_AG)
    

    # Plot_3-1: Revenue and Cost data for all types (Billion $)
    revenue_ag_sum = revenue_ag_df.groupby(['Year']).sum(numeric_only=True).reset_index()
    revenue_ag_sum.insert(1,'Type','Agricultural land-use (revenue)')
    
    revenue_am_sum = revenue_am_df.groupby(['Year']).sum(numeric_only=True).reset_index()
    revenue_am_sum.insert(1,'Type','Agricultural management (revenue)')
    
    revenue_non_ag_sum = revenue_non_ag_df.groupby(['Year']).sum(numeric_only=True).reset_index()
    revenue_non_ag_sum.insert(1,'Type','Non-agricultural land-use (revenue)')
    
    cost_ag_sum = cost_ag_df.groupby(['Year']).sum(numeric_only=True).reset_index()
    cost_ag_sum.insert(1,'Type','Agricultural land-use (cost)')
    
    cost_am_sum = cost_am_df.groupby(['Year']).sum(numeric_only=True).reset_index()
    cost_am_sum.insert(1,'Type','Agricultural management (cost)')
    
    cost_non_ag_sum = cost_non_ag_df.groupby(['Year']).sum(numeric_only=True).reset_index()
    cost_non_ag_sum.insert(1,'Type','Non-agricultural land-use (cost)')
    
    cost_transition_ag2ag_sum = cost_transition_ag2ag_df.groupby(['Year']).sum(numeric_only=True).reset_index()
    cost_transition_ag2ag_sum.insert(1,'Type','Transition cost')
    
    cost_transition_ag2non_sum = cost_transition_ag2non_ag_df.groupby(['Year']).sum(numeric_only=True).reset_index()
    cost_transition_ag2non_sum.insert(1,'Type','Transition cost')
    
    cost_transition_non_ag2ag_sum = cost_transition_non_ag2ag_df.groupby(['Year']).sum(numeric_only=True).reset_index()
    cost_transition_non_ag2ag_sum.insert(1,'Type','Transition cost')
    
    
    rev_cost_all = pd.concat([revenue_ag_sum,revenue_am_sum,revenue_non_ag_sum,
                            cost_ag_sum,cost_am_sum,cost_non_ag_sum,
                            cost_transition_ag2ag_sum,cost_transition_ag2non_sum,
                            cost_transition_non_ag2ag_sum],axis=0)
    rev_cost_all = rev_cost_all.groupby(['Year','Type']).sum(numeric_only=True).reset_index()

    rev_cost_net = rev_cost_all.groupby(['Year']).sum(numeric_only=True).reset_index()
    rev_cost_net['Type'] = 'Profit'
    
    rev_cost_all_wide = rev_cost_all\
        .groupby(['Type'])[['Year','Value (billion)']]\
        .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
        .reset_index()
    rev_cost_all_wide.columns = ['name','data']
    rev_cost_all_wide['type'] = 'column'
        
    rev_cost_net_wide = rev_cost_net\
        .groupby(['Type'])[['Year','Value (billion)']]\
        .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
        .reset_index()
    rev_cost_net_wide.columns = ['name','data']
    rev_cost_net_wide['type'] = 'spline'
    
    rev_cost_wide_json = pd.concat([rev_cost_all_wide,rev_cost_net_wide],axis=0)
    
    # Define the specific order
    order = ['Agricultural land-use (revenue)', 
             'Agricultural management (revenue)', 
             'Non-agricultural land-use (revenue)',
             'Agricultural land-use (cost)', 
             'Agricultural management (cost)', 
             'Non-agricultural land-use (cost)',
             'Transition cost','Profit']
    rev_cost_wide_json = rev_cost_wide_json.set_index('name').reindex(order).reset_index()
    
    rev_cost_wide_json.to_json(f'{SAVE_DIR}/economics_0_rev_cost_all_wide.json', orient='records')
    
    
    

    # Plot_3-1: Revenue for Agricultural land-use (Billion Dollars)
    keep_cols = ['Year', 'Value (billion)', 'Value ($)']
    loop_cols = revenue_ag_df.columns.difference(keep_cols)

    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = revenue_ag_df[take_cols].groupby(['Year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.groupby(col)[['Year','Value (billion)']]\
                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
                    .reset_index()
                    
        df_wide.columns = ['name','data']
        df_wide['type'] = 'column'
        
        df_wide.to_json(f'{SAVE_DIR}/economics_1_ag_revenue_{idx+1}_{col}_wide.json', orient='records')



    # Plot_3-2: Cost for Agricultural land-use (Billion Dollars)
    keep_cols = ['Year', 'Value (billion)', 'Value ($)']
    loop_cols = cost_ag_df.columns.difference(keep_cols)
    
    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = cost_ag_df[take_cols].groupby(['Year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.groupby(col)[['Year','Value (billion)']]\
                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
                    .reset_index()
        df_wide.columns = ['name','data']
        df_wide['type'] = 'column'
        # save to disk
        df_wide.to_json(f'{SAVE_DIR}/economics_2_ag_cost_{idx+1}_{col}_wide.json', orient='records')


    # # Plot_3-3: Revenue and Cost data (Billion Dollars)
    # rev_cost_compare = get_rev_cost(revenue_ag_df,cost_ag_df)
    # rev_cost_compare = rev_cost_compare.sort_values(['Year'])
    # rev_cost_compare['rev_low'] = 0
    
    # rev_cost_compare_rev = rev_cost_compare[['rev_low','Revenue (billion)']].copy()
    # rev_cost_compare_rev.columns = ['low','high']
    # rev_cost_compare_rev_records = {'name' : 'Revenue',
    #                                 'data': list(map(list,zip(rev_cost_compare['rev_low'],rev_cost_compare['Revenue (billion)'])))}
    
    
    # rev_cost_compare_cost = rev_cost_compare[['Profit (billion)','Revenue (billion)']].copy()
    # rev_cost_compare_cost.columns = ['low','high']
    # rev_cost_compare_cost_records = {'name' : 'Cost',
    #                                 'data': list(map(list,zip(rev_cost_compare['Profit (billion)'],rev_cost_compare['Revenue (billion)'])))}
    
    # rev_cost_compare_records = {'categories': [str(i) for i in rev_cost_compare['Year'].unique()],
    #                             'series': [rev_cost_compare_rev_records,rev_cost_compare_cost_records]}
    

    # with open(f'{SAVE_DIR}/economics_3_rev_cost_all.json', 'w') as outfile:
    #     outfile.write(json.dumps(rev_cost_compare_records))
    
    
    
    # Plot_3-4: Revenue for Agricultural Management (Billion $)
    keep_cols = ['Year', 'Value (billion)', 'Value ($)']
    loop_cols = revenue_am_df.columns.difference(keep_cols)
    
    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = revenue_am_df[take_cols].groupby(['Year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.groupby(col)[['Year','Value (billion)']]\
                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
                    .reset_index()
        df_wide.columns = ['name','data']
        df_wide['type'] = 'column'
        # save to disk
        df_wide.to_json(f'{SAVE_DIR}/economics_4_am_revenue_{idx+1}_{col}_wide.json', orient='records')



    # Plot_3-5: Cost for Agricultural Management (Billion $)
    keep_cols = ['Year', 'Value (billion)','Value ($)']
    loop_cols = cost_am_df.columns.difference(keep_cols)
    
    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = cost_am_df[take_cols].groupby(['Year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.groupby(col)[['Year','Value (billion)']]\
                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
                    .reset_index()
        df_wide.columns = ['name','data']
        df_wide['type'] = 'column'
        # save to disk
        df_wide.to_json(f'{SAVE_DIR}/economics_5_am_cost_{idx+1}_{col}_wide.json', orient='records')
        
        
    # Plot_3-6: Revenue for Non-Agricultural land-use (Billion $)
    keep_cols = ['Year', 'Value (billion)','Value ($)']
    loop_cols = revenue_non_ag_df.columns.difference(keep_cols)
    
    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = revenue_non_ag_df[take_cols].groupby(['Year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.groupby(col)[['Year','Value (billion)']]\
                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
                    .reset_index()
        df_wide.columns = ['name','data']
        df_wide['type'] = 'column'
        # save to disk
        df_wide.to_json(f'{SAVE_DIR}/economics_6_non_ag_revenue_{idx+1}_{col}_wide.json', orient='records')
        
        
    # Plot_3-7: Cost for Non-Agricultural land-use (Billion $)
    keep_cols = ['Year', 'Value (billion)','Value ($)']
    loop_cols = cost_non_ag_df.columns.difference(keep_cols)
    
    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = cost_non_ag_df[take_cols].groupby(['Year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.groupby(col)[['Year','Value (billion)']]\
                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
                    .reset_index()
        df_wide.columns = ['name','data']
        df_wide['type'] = 'column'
        # save to disk
        df_wide.to_json(f'{SAVE_DIR}/economics_7_non_ag_cost_{idx+1}_{col}_wide.json', orient='records')
    
    
    # Plot_3-8: Transition cost for Ag to Ag (Billion $)
    keep_cols = ['Year', 'Value (billion)','Cost ($)']
    loop_cols = cost_transition_ag2ag_df.columns.difference(keep_cols)
    
    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = cost_transition_ag2ag_df[take_cols].groupby(['Year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.groupby(col)[['Year','Value (billion)']]\
                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
                    .reset_index()
        df_wide.columns = ['name','data']
        df_wide['type'] = 'column'
        # save to disk
        df_wide.to_json(f'{SAVE_DIR}/economics_8_transition_ag2ag_cost_{idx+1}_{col}_wide.json', orient='records')
        
    
    # Save the transition matrix cost
    cost_transition_ag2ag_trans_mat = cost_transition_ag2ag_df.groupby(['Year','From land-use', 'To land-use']).sum(numeric_only=True).reset_index()
    cost_transition_ag2ag_trans_mat = cost_transition_ag2ag_trans_mat.set_index(['Year','From land-use', 'To land-use'])
    cost_transition_ag2ag_trans_mat = cost_transition_ag2ag_trans_mat\
                                        .reindex(index = pd.MultiIndex.from_product([years, AG_LANDUSE, AG_LANDUSE], 
                                                 names = ['Year','From land-use', 'To land-use'])).reset_index()
    
    cost_transition_ag2ag_trans_mat['idx_from'] = cost_transition_ag2ag_trans_mat['From land-use']\
                                                    .apply(lambda x: AG_LANDUSE.index(x))
    cost_transition_ag2ag_trans_mat['idx_to'] = cost_transition_ag2ag_trans_mat['To land-use']\
                                                    .apply(lambda x: AG_LANDUSE.index(x))
                                                    
    cost_transition_ag2ag_trans_mat_data = cost_transition_ag2ag_trans_mat\
                                    .groupby(['Year'])[['idx_from','idx_to', 'Value (billion)']]\
                                    .apply(lambda x: list(map(list,zip(x['idx_from'],x['idx_to'],x['Value (billion)']))))\
                                    .reset_index()                             
    cost_transition_ag2ag_trans_mat_data.columns = ['Year','data']
    
    cost_transition_ag2ag_trans_mat_json = {'categories': AG_LANDUSE,
                                            'series': json.loads(cost_transition_ag2ag_trans_mat_data.to_json(orient='records'))}
    
    with open(f'{SAVE_DIR}/economics_8_transition_ag2ag_cost_5_transition_matrix.json', 'w') as outfile:
        outfile.write(json.dumps(cost_transition_ag2ag_trans_mat_json))
                                    
                                    
                                    
                                        
                                
    # Plot_3-9: Transition cost for Ag to Non-Ag (Billion $)
    keep_cols = ['Year', 'Value (billion)','Cost ($)']
    loop_cols = cost_transition_ag2non_ag_df.columns.difference(keep_cols)
    
    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = cost_transition_ag2non_ag_df[take_cols].groupby(['Year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.groupby(col)[['Year','Value (billion)']]\
                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
                    .reset_index()
        df_wide.columns = ['name','data']
        df_wide['type'] = 'column'
        # save to disk
        df_wide.to_json(f'{SAVE_DIR}/economics_9_transition_ag2non_cost_{idx+1}_{col}_wide.json', orient='records')
        
        
    # Get the transition matrix cost
    cost_transition_ag2non_ag_trans_mat = cost_transition_ag2non_ag_df\
                                            .groupby(['Year','From land-use', 'To land-use'])\
                                            .sum(numeric_only=True).reset_index()
                                            
    cost_transition_ag2non_ag_trans_mat = cost_transition_ag2non_ag_trans_mat\
                                           .set_index(['Year','From land-use', 'To land-use'])\
                                           .reindex(index = pd.MultiIndex.from_product([years, AG_LANDUSE, NON_AG_LANDUSE],
                                                    names = ['Year','From land-use', 'To land-use'])).reset_index()
                                           
    cost_transition_ag2non_ag_trans_mat['idx_from'] = cost_transition_ag2non_ag_trans_mat['From land-use']\
                                                    .apply(lambda x: AG_LANDUSE.index(x))
    cost_transition_ag2non_ag_trans_mat['idx_to']  = cost_transition_ag2non_ag_trans_mat['To land-use']\
                                                    .apply(lambda x: NON_AG_LANDUSE.index(x))
                                                    

    cost_transition_ag2non_ag_trans_mat_data = cost_transition_ag2non_ag_trans_mat\
                                    .groupby(['Year'])[['idx_from','idx_to', 'Value (billion)']]\
                                    .apply(lambda x: list(map(list,zip(x['idx_from'],x['idx_to'],x['Value (billion)']))))\
                                    .reset_index()                            
    cost_transition_ag2non_ag_trans_mat_data.columns = ['Year','data']
    
    
    cost_transition_ag2non_ag_trans_mat_json = {'categories_from': AG_LANDUSE,
                                                'categories_to': NON_AG_LANDUSE,
                                                'series': json.loads(cost_transition_ag2non_ag_trans_mat_data.to_json(orient='records'))}
    
    
    with open(f'{SAVE_DIR}/economics_9_transition_ag2non_cost_5_transition_matrix.json', 'w') as outfile:
        outfile.write(json.dumps(cost_transition_ag2non_ag_trans_mat_json))
    

        
    # Plot_3-10: Transition cost for Non-Ag to Ag (Billion $)
    keep_cols = ['Year', 'Value (billion)','Cost ($)']
    loop_cols = cost_transition_non_ag2ag_df.columns.difference(keep_cols)
    
    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = cost_transition_non_ag2ag_df[take_cols].groupby(['Year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.groupby(col)[['Year','Value (billion)']]\
                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (billion)']))))\
                    .reset_index()
        df_wide.columns = ['name','data']
        df_wide['type'] = 'column'
        # save to disk
        df_wide.to_json(f'{SAVE_DIR}/economics_10_transition_non_ag2ag_cost_{idx+1}_{col}_wide.json', orient='records')





    ####################################################
    #                       4) GHGs                    #
    ####################################################
    if settings.GHG_EMISSIONS_LIMITS == 'on':
        GHG_files_onland = files.query('category == "GHG" and base_name.str.contains("GHG_emissions_separate") and year_types != "begin_end_year"').reset_index(drop=True)
        GHG_files_onland = pd.concat([pd.read_csv(path) for path in GHG_files_onland['path']], ignore_index=True)
        GHG_files_onland['CO2_type'] = GHG_files_onland['CO2_type'].replace(GHG_NAMES)
        GHG_files_onland['Value (Mt CO2e)'] = GHG_files_onland['Value (t CO2e)'] / 1e6
        GHG_files_onland = GHG_files_onland.replace(RENAME_AM_NON_AG)
        
        def get_landuse_type(x):
            if x in LU_CROPS:
                return 'Crop'
            elif x in LVSTK_NATURAL:
                return 'Livestock - natural land'
            elif x in LVSTK_MODIFIED:
                return 'Livestock - modified land'
            else:
                return 'Unallocated land'
            
        GHG_files_onland['Land-use type'] = GHG_files_onland['Land-use'].apply(get_landuse_type)

        # Read the off-land GHG emissions
        GHG_off_land = files.query('category == "GHG" and base_name == "GHG_emissions_offland_commodity" and year_types != "begin_end_year"').reset_index(drop=True)
        GHG_off_land = pd.concat([pd.read_csv(path) for path in GHG_off_land['path']], ignore_index=True)
        GHG_off_land['Value (Mt CO2e)'] = GHG_off_land['Total GHG Emissions (tCO2e)'] / 1e6
        GHG_off_land['COMMODITY'] = GHG_off_land['COMMODITY'].apply(lambda x: x[0].capitalize() + x[1:])
        GHG_off_land['Emission Source'] = GHG_off_land['Emission Source'].replace({'CO2': 'Carbon Dioxide (CO2)',
                                                                                'CH4': 'Methane (CH4)',
                                                                                'N2O': 'Nitrous Oxide (N2O)'})
        
        # Plot_4-1: GHG of cumulative emissions (Mt)
        Emission_onland = GHG_files_onland.groupby('Year')['Value (Mt CO2e)'].sum(numeric_only = True).reset_index()
        Emission_onland = Emission_onland[['Year','Value (Mt CO2e)']]
        
        Emission_offland = GHG_off_land.groupby('Year').sum(numeric_only=True).reset_index()
        Emission_offland = Emission_offland[['Year','Value (Mt CO2e)']]
        Emission_offland['Type'] = 'Off-land Commodity'
        
        Net_emission = pd.concat([Emission_onland,Emission_offland],axis=0)
        Net_emission = Net_emission.groupby(['Year']).sum(numeric_only = True).reset_index()
        
        Cumsum_emissions = Net_emission.copy()
        Cumsum_emissions['Cumulative GHG emissions (Mt)'] = Cumsum_emissions.cumsum()['Value (Mt CO2e)']
        Cumsum_emissions = Cumsum_emissions[['Year','Cumulative GHG emissions (Mt)']]
        
        Cumsum_emissions_json = [{'data': list(map(list,zip(Cumsum_emissions['Year'],Cumsum_emissions['Cumulative GHG emissions (Mt)']))),
                                'type' : 'column'}]
        
        with open(f'{SAVE_DIR}/GHG_1_cunsum_emission_Mt.json', 'w') as outfile:
            json.dump(Cumsum_emissions_json, outfile)
        

        # Plot_4-2: GHG from individual emission sectors (Mt)
        GHG_files_wide_onland = GHG_files_onland[['Year','Type','Value (Mt CO2e)']]
        GHG_files_wide_offland = Emission_offland[['Year','Type','Value (Mt CO2e)']]
        
        GHG_files_wide = pd.concat([GHG_files_wide_onland,GHG_files_wide_offland],axis=0)
        GHG_files_wide = GHG_files_wide.groupby(['Type','Year']).sum(numeric_only=True).reset_index()
        
        
        GHG_files_wide = GHG_files_wide\
            .groupby(['Type'])[['Year','Value (Mt CO2e)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
            .reset_index()
            
        GHG_files_wide.columns = ['name','data'] 
        GHG_files_wide['type'] = 'column'
        
        GHG_files_wide.loc[-1] = ['Net emissions', list(map(list,zip(Net_emission['Year'],Net_emission['Value (Mt CO2e)']))), 'line']
        GHG_files_wide.to_json(f'{SAVE_DIR}/GHG_2_individual_emission_Mt.json',orient='records')



        # Plot_4-3: GHG emission (Mt) for on-land
        GHG_agricultural = GHG_files_onland.query('Type == "Agricultural Landuse"').copy()
        
        GHG_CO2 = GHG_agricultural.query('~CO2_type.isin(@GHG_CATEGORY.keys())').copy()
        GHG_CO2['GHG Category'] = 'CO2'

        GHG_nonCO2 = GHG_agricultural.query('CO2_type.isin(@GHG_CATEGORY.keys())').copy()
        GHG_nonCO2['GHG Category'] = GHG_nonCO2['CO2_type'].apply(lambda x: GHG_CATEGORY[x].keys())
        GHG_nonCO2['Multiplier'] = GHG_nonCO2['CO2_type'].apply(lambda x: GHG_CATEGORY[x].values())
        GHG_nonCO2 = GHG_nonCO2.explode(['GHG Category','Multiplier']).reset_index(drop=True)
        GHG_nonCO2['Value (Mt CO2e)'] = GHG_nonCO2['Value (Mt CO2e)'] * GHG_nonCO2['Multiplier']
        GHG_nonCO2 = GHG_nonCO2.drop(columns=['Multiplier'])
        
        dfs = [GHG_CO2.dropna(axis=1, how='all'),GHG_nonCO2.dropna(axis=1, how='all')]
        GHG_ag_emissions_long = pd.concat(dfs,axis=0).reset_index(drop=True)
        GHG_ag_emissions_long['Value (Mt CO2e)'] = GHG_ag_emissions_long['Value (t CO2e)'] / 1e6
        GHG_ag_emissions_long['GHG Category'] = GHG_ag_emissions_long['GHG Category'].replace({'CH4': 'Methane (CH4)', 
                                                                                        'N2O': 'Nitrous Oxide (N2O)', 
                                                                                        'CO2': 'Carbon Dioxide (CO2)'})
        
        


        # Plot_4-3-1: Agricultural Emission (on-land) by crop/lvstk sectors (Mt)
        GHG_crop_lvstk_total = GHG_ag_emissions_long\
                                        .groupby(['Year','Type','Land-use type'])\
                                        .sum(numeric_only=True)[['Value (Mt CO2e)']]\
                                        .reset_index()
                                        

        GHG_Ag_emission_total_crop_lvstk_wide = GHG_crop_lvstk_total\
                                                    .groupby(['Land-use type'])[['Year','Value (Mt CO2e)']]\
                                                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
                                                    .reset_index()
                                                    
        GHG_Ag_emission_total_crop_lvstk_wide.columns = ['name','data']
        GHG_Ag_emission_total_crop_lvstk_wide['type'] = 'column'
        GHG_Ag_emission_total_crop_lvstk_wide.to_json(f'{SAVE_DIR}/GHG_4_3_1_crop_lvstk_emission_Mt.json',orient='records')



        # Plot_4-3-2: Agricultural Emission (on-land) by dry/Water_supply  (Mt)
        GHG_Ag_emission_total_dry_irr = GHG_ag_emissions_long.groupby(['Year','Water_supply']).sum()['Value (Mt CO2e)'].reset_index()
        
        GHG_Ag_emission_total_dry_irr_wide = GHG_Ag_emission_total_dry_irr\
                                                .groupby(['Water_supply'])[['Year','Value (Mt CO2e)']]\
                                                .apply(lambda x: list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
                                                .reset_index()
                                                
        GHG_Ag_emission_total_dry_irr_wide.columns = ['name','data']
        GHG_Ag_emission_total_dry_irr_wide['type'] = 'column'
        GHG_Ag_emission_total_dry_irr_wide.to_json(f'{SAVE_DIR}/GHG_4_3_2_dry_irr_emission_Mt.json',orient='records')
        
        


        # Plot_4-3-3: Agricultural Emission (on-land) by GHG type sectors (Mt)
        GHG_Ag_emission_total_GHG_type = GHG_ag_emissions_long.groupby(['Year','GHG Category']).sum()['Value (Mt CO2e)'].reset_index()
        
        GHG_Ag_emission_total_GHG_type_wide = GHG_Ag_emission_total_GHG_type\
                                                .groupby(['GHG Category'])[['Year','Value (Mt CO2e)']]\
                                                .apply(lambda x: list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
                                                .reset_index()
                                                
        GHG_Ag_emission_total_GHG_type_wide.columns = ['name','data']
        GHG_Ag_emission_total_GHG_type_wide['type'] = 'column'
        GHG_Ag_emission_total_GHG_type_wide.to_json(f'{SAVE_DIR}/GHG_4_3_3_category_emission_Mt.json',orient='records')


        # Plot_4-3-4: Agricultural Emission (on-land) by Sources (Mt)
        GHG_Ag_emission_total_Source = GHG_ag_emissions_long.groupby(['Year','CO2_type']).sum()['Value (Mt CO2e)'].reset_index()
        
        GHG_Ag_emission_total_Source_wide = GHG_Ag_emission_total_Source\
                                                .groupby(['CO2_type'])[['Year','Value (Mt CO2e)']]\
                                                .apply(lambda x: list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
                                                .reset_index()
                                                
        GHG_Ag_emission_total_Source_wide.columns = ['name','data']
        GHG_Ag_emission_total_Source_wide['type'] = 'column'
        GHG_Ag_emission_total_Source_wide.to_json(f'{SAVE_DIR}/GHG_4_3_4_sources_emission_Mt.json',orient='records')
        

        # Plot_4-3-5: GHG emission (on-land) in start and end years (Mt)
        start_year,end_year = GHG_ag_emissions_long['Year'].min(),GHG_ag_emissions_long['Year'].max() 

        GHG_lu_lm = GHG_ag_emissions_long\
                .groupby(['Year','Type','Land-use','Water_supply'])\
                .sum()['Value (Mt CO2e)']\
                .reset_index()
                
        GHG_lu_lm_df_start = GHG_lu_lm.query('Year == @start_year').reset_index(drop=True)
        GHG_lu_lm_df_end = GHG_lu_lm.query('Year == @end_year').reset_index(drop=True)

        GHG_lu_lm_df_begin_end = pd.concat([GHG_lu_lm_df_start,GHG_lu_lm_df_end],axis=0)
        GHG_lu_lm_df_begin_end = GHG_lu_lm_df_begin_end.sort_values(['Water_supply','Land-use','Year'])
        
        GHG_lu_lm_df_begin_end_category = GHG_lu_lm_df_begin_end.query('Water_supply == "Dryland"')\
                                            .groupby('Land-use')[['Year','Land-use']]\
                                            .apply(lambda x: x['Year'].tolist())\
                                            .reset_index()                                   
        GHG_lu_lm_df_begin_end_category.columns = ['name','categories']
        
        GHG_lu_lm_df_begin_end_series = GHG_lu_lm_df_begin_end[['Water_supply','Value (Mt CO2e)']]\
                                            .groupby('Water_supply')[['Water_supply','Value (Mt CO2e)']]\
                                            .apply(lambda x: list(map(list,zip(x['Water_supply'], x['Value (Mt CO2e)']))))\
                                            .reset_index()
        GHG_lu_lm_df_begin_end_series.columns = ['name','data']
        GHG_lu_lm_df_begin_end_series['type'] = 'column'
        
        
        GHG_lu_lm_df_begin_end_json = {'categories': json.loads(GHG_lu_lm_df_begin_end_category.to_json(orient='records')),
                                        'series': json.loads(GHG_lu_lm_df_begin_end_series.to_json(orient='records'))}
        
        with open(f'{SAVE_DIR}/GHG_4_3_5_lu_lm_emission_Mt_wide.json', 'w') as outfile:
            json.dump(GHG_lu_lm_df_begin_end_json, outfile)
        


        # Plot_4-3-6: GHG emission (on-land) in the target year (Mt)
        GHG_lu_source = GHG_ag_emissions_long\
                        .groupby(['Year','Land-use','Water_supply','CO2_type'])\
                        .sum()['Value (Mt CO2e)']\
                        .reset_index()
                
        GHG_lu_source_target_yr = GHG_lu_source.query(f'Year == {end_year}')

        GHG_lu_source_nest = GHG_lu_source_target_yr\
                                .groupby(['CO2_type','Land-use'])\
                                .sum(numeric_only=True)[['Value (Mt CO2e)']]\
                                .reset_index()
                                
        GHG_lu_source_nest_dict = GHG_lu_source_nest\
                                    .groupby('CO2_type')[['Land-use','Value (Mt CO2e)']]\
                                    .apply(lambda x: x[['Land-use','Value (Mt CO2e)']].to_dict(orient='records'))\
                                    .reset_index()
                                    
        GHG_lu_source_nest_dict.columns = ['name','data']
        GHG_lu_source_nest_dict['data'] = GHG_lu_source_nest_dict['data']\
            .apply(lambda x: [{'name': i['Land-use'], 'value': i['Value (Mt CO2e)']} for i in x])
        
        GHG_lu_source_nest_dict.to_json(f'{SAVE_DIR}/GHG_4_3_6_lu_source_emission_Mt.json',orient='records')

                
            
        # Plot_4-3-7: GHG emission (off-land) by commodity (Mt)
        GHG_off_land_commodity = GHG_off_land\
            .groupby(['Year','COMMODITY'])\
            .sum(numeric_only=True)['Value (Mt CO2e)'].reset_index()
            
        GHG_off_land_commodity_json = GHG_off_land_commodity\
            .groupby('COMMODITY')[['Year','Value (Mt CO2e)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
            .reset_index()
            
        GHG_off_land_commodity_json.columns = ['name','data']
        GHG_off_land_commodity_json['type'] = 'column'
        GHG_off_land_commodity_json.to_json(f'{SAVE_DIR}/GHG_4_3_7_off_land_commodity_emission_Mt.json',orient='records')
        
        
        
        # Plot_4-3-8: GHG emission (off-land) by sources (Mt)
        GHG_off_land_sources = GHG_off_land\
            .groupby(['Year','Emission Source'])\
            .sum(numeric_only=True)['Value (Mt CO2e)'].reset_index()
            
        GHG_off_land_sources_json = GHG_off_land_sources\
            .groupby('Emission Source')[['Year','Value (Mt CO2e)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
            .reset_index()
            
        GHG_off_land_sources_json.columns = ['name','data']
        GHG_off_land_sources_json['type'] = 'column'
        GHG_off_land_sources_json.to_json(f'{SAVE_DIR}/GHG_4_3_8_off_land_sources_emission_Mt.json',orient='records')
        
        
        
        
        # Plot_4-3-9: GHG emission (off-land) by Emission Type
        GHG_off_land_type = GHG_off_land\
            .groupby(['Year','Emission Type'])\
            .sum(numeric_only=True)['Value (Mt CO2e)'].reset_index()
            
        GHG_off_land_type_json = GHG_off_land_type\
            .groupby('Emission Type')[['Year','Value (Mt CO2e)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
            .reset_index()
            
        GHG_off_land_type_json.columns = ['name','data']
        GHG_off_land_type_json['type'] = 'column'
        
        GHG_off_land_type_json.to_json(f'{SAVE_DIR}/GHG_4_3_9_off_land_type_emission_Mt.json',orient='records')



        # Plot_4-4: GHG abatement by Non-Agrilcultural sector (Mt)
        Non_ag_reduction_long = GHG_files_onland.query('Type == "Non-Agricultural land-use"').reset_index(drop=True)
                                                    
        Non_ag_reduction_source = Non_ag_reduction_long.groupby(['Year','Type','Land-use'])\
            .sum(numeric_only=True)['Value (Mt CO2e)'].reset_index()
        
        # Fill the missing years with 0 values    
        fill_years = set(years) - set(Non_ag_reduction_source['Year'].unique())
        Non_ag_reduction_source = pd.concat([Non_ag_reduction_source, pd.DataFrame({'Year': list(fill_years), 'CO2_type':'Agroforestry','Value (Mt CO2e)': 0})], axis=0)
            
        Non_ag_reduction_source_wide = Non_ag_reduction_source\
                                            .groupby(['Land-use'])[['Year','Value (Mt CO2e)']]\
                                            .apply(lambda x: list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
                                            .reset_index()
                                            
        Non_ag_reduction_source_wide.columns = ['name','data']
        Non_ag_reduction_source_wide['type'] = 'column'
        
        Non_ag_reduction_source_wide.to_json(f'{SAVE_DIR}/GHG_4_4_ag_reduction_source_wide_Mt.json',orient='records')



        # Plot_4-5: GHG reductions by Agricultural managements (Mt)
        Ag_man_sequestration_long = GHG_files_onland.query('Type == "Agricultural Management"').reset_index(drop=True)

        # Plot_4-5-1: GHG reductions by Agricultural managements in total (Mt)
        Ag_man_sequestration_total = Ag_man_sequestration_long.groupby(['Year','Agricultural Management Type']).sum()['Value (Mt CO2e)'].reset_index()
        
        Ag_man_sequestration_total_wide = Ag_man_sequestration_total\
                                            .groupby(['Agricultural Management Type'])[['Year','Value (Mt CO2e)']]\
                                            .apply(lambda x: list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
                                            .reset_index()
                                            
        Ag_man_sequestration_total_wide.columns = ['name','data']
        Ag_man_sequestration_total_wide['type'] = 'column'
        Ag_man_sequestration_total_wide.to_json(f'{SAVE_DIR}/GHG_4_5_1_GHG_ag_man_df_wide_Mt.json',orient='records')


        # Plot_4-5-2: GHG reductions by Agricultural managements in subsector (Mt)
        Ag_man_sequestration_crop_lvstk_wide = Ag_man_sequestration_long.groupby(['Year','Type','Land-use type']).sum()['Value (Mt CO2e)'].reset_index()
        
        Ag_man_sequestration_crop_lvstk_wide = Ag_man_sequestration_crop_lvstk_wide\
                                                .groupby(['Land-use type'])[['Year','Value (Mt CO2e)']]\
                                                .apply(lambda x: list(map(list,zip(x['Year'],x['Value (Mt CO2e)']))))\
                                                .reset_index()
                                                
        Ag_man_sequestration_crop_lvstk_wide.columns = ['name','data']
        Ag_man_sequestration_crop_lvstk_wide['type'] = 'column'
        Ag_man_sequestration_crop_lvstk_wide.to_json(f'{SAVE_DIR}/GHG_4_5_2_GHG_ag_man_GHG_crop_lvstk_df_wide_Mt.json',orient='records')
        


        # Plot_4-5-3: GHG reductions by Agricultural managements in subsector (Mt)
        Ag_man_sequestration_dry_irr_total = Ag_man_sequestration_long.groupby(['Year','Water_supply']).sum()['Value (Mt CO2e)'].reset_index()
        
        Ag_man_sequestration_dry_irr_wide = Ag_man_sequestration_dry_irr_total\
                                            .groupby(['Water_supply'])[['Year','Value (Mt CO2e)']]\
                                            .apply(lambda x: list(map(list,zip(x['Year'], x['Value (Mt CO2e)']))))\
                                            .reset_index()
        Ag_man_sequestration_dry_irr_wide.columns = ['name','data']
        Ag_man_sequestration_dry_irr_wide['type'] = 'column'
        Ag_man_sequestration_dry_irr_wide.to_json(f'{SAVE_DIR}/GHG_4_5_3_GHG_ag_man_dry_irr_df_wide_Mt.json',orient='records')
    
    
    
    


    ####################################################
    #                     5) Water                     #
    ####################################################
    
    if settings.WATER_NET_YIELD_LIMITS == 'on':
        water_df_total = files.query('category == "water" and year_types == "single_year" and ~base_name.str.contains("separate")').reset_index(drop=True)
        water_df_total = pd.concat([pd.read_csv(path) for path in water_df_total['path']], ignore_index=True)
        
        water_df_separate = files.query('category == "water" and year_types == "single_year" and base_name.str.contains("separate")').reset_index(drop=True)
        water_df_separate = pd.concat([pd.read_csv(path) for path in water_df_separate['path']], ignore_index=True)
        water_df_separate = water_df_separate.replace(RENAME_AM_NON_AG)
        

        # Plot_5-1: Water net yield compared to limite (%)
        water_df_total_pct = water_df_total.query('Variable == "PROPORTION_LIMIT_%" ')
        water_df_total_pct_wide = water_df_total_pct\
                                    .groupby(['REGION_NAME'])[['Year','Value (ML)']]\
                                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (ML)']))))\
                                    .reset_index()
                                    
        water_df_total_pct_wide.columns = ['name','data']
        water_df_total_pct_wide['type'] = 'spline'
        water_df_total_pct_wide.to_json(f'{SAVE_DIR}/water_1_percent_of_limit.json',orient='records')


        # Plot_5-2: Water net yield compared to limite (ML)
        water_df_total_yield = water_df_total.query('Variable == "TOT_WATER_NET_YIELD_ML" ')
        water_df_total_yield_wide = water_df_total_yield\
                                    .groupby(['REGION_NAME'])[['Year','Value (ML)']]\
                                    .apply(lambda x: list(map(list,zip(x['Year'],x['Value (ML)']))))\
                                    .reset_index()

        water_df_total_yield_wide.columns = ['name','data']
        water_df_total_yield_wide['type'] = 'spline'
        water_df_total_yield_wide.to_json(f'{SAVE_DIR}/water_2_yield_to_limit.json',orient='records')                        
        
        
        
        # Plot_5-3: Water net yield by sector (ML)
        water_df_separate_lu_type = water_df_separate.groupby(['Year','Landuse Type']).sum(numeric_only=True)[['Water Net Yield (ML)']].reset_index()
        water_df_net = water_df_separate.groupby('Year').sum(numeric_only=True).reset_index()

        water_df_separate_lu_type = water_df_separate_lu_type\
            .groupby(['Landuse Type'])[['Landuse Type','Year','Water Net Yield (ML)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Water Net Yield (ML)']))))\
            .reset_index()
            
        water_df_separate_lu_type.columns = ['name','data']
        water_df_separate_lu_type['type'] = 'column'

        water_df_separate_lu_type.loc[len(water_df_separate_lu_type)] = ['Net Volume', list(map(list,zip(water_df_net['Year'],water_df_net['Water Net Yield (ML)']))), 'line']

        water_df_separate_lu_type.to_json(f'{SAVE_DIR}/water_3_net_yield_by_sector.json',orient='records')



        # Plot_5-4: Water net yield by landuse (ML)
        water_df_seperate_lu = water_df_separate.groupby(['Year','Landuse']).sum()[['Water Net Yield (ML)']].reset_index()
        
        water_df_seperate_lu_wide = water_df_seperate_lu\
                                        .groupby(['Landuse'])[['Year','Water Net Yield (ML)']]\
                                        .apply(lambda x: list(map(list,zip(x['Year'],x['Water Net Yield (ML)']))))\
                                        .reset_index()
                                        
        water_df_seperate_lu_wide.columns = ['name','data']
        water_df_seperate_lu_wide['type'] = 'column'
        water_df_seperate_lu_wide = water_df_seperate_lu_wide.set_index('name').reindex(LANDUSE_ALL).reset_index()
        
        
        water_df_seperate_lu_wide.to_json(f'{SAVE_DIR}/water_4_net_yield_by_landuse.json',orient='records')
        
        
        # Plot_5-5: Water net yield by Water_supply (ML)
        water_df_seperate_irr = water_df_separate.groupby(['Year','Water_supply']).sum()[['Water Net Yield (ML)']].reset_index()
        
        water_df_seperate_irr_wide = water_df_seperate_irr\
                                        .groupby(['Water_supply'])[['Year','Water Net Yield (ML)']]\
                                        .apply(lambda x: list(map(list,zip(x['Year'],x['Water Net Yield (ML)']))))\
                                        .reset_index()
                                        
        water_df_seperate_irr_wide.columns = ['name','data']
        water_df_seperate_irr_wide['type'] = 'column'
        water_df_seperate_irr_wide.to_json(f'{SAVE_DIR}/water_5_net_yield_by_Water_supply.json',orient='records')
    



    #########################################################
    #                   6) Biodiversity                     #
    #########################################################
    
    # ---------------- Biodiversity priority score  ----------------
    # get biodiversity dataframe
    bio_paths = files.query('category == "biodiversity" and year_types == "single_year" and base_name == "biodiversity_separate"').reset_index(drop=True)
    bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
    bio_df['Biodiversity score (million)'] = bio_df['Biodiversity score'] / 1e6
    bio_df = bio_df.replace(RENAME_AM_NON_AG)

    # Filter out landuse that are related to biodiversity
    bio_lucc = LU_NATURAL + NON_AG_LANDUSE


    # Plot_6-1: Biodiversity total by category
    bio_df_category = bio_df.groupby(['Year','Landuse type']).sum(numeric_only=True).reset_index()
    bio_df_category = bio_df_category\
        .groupby('Landuse type')[['Year','Biodiversity score (million)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Biodiversity score (million)']))))\
        .reset_index()
        
    bio_df_category.columns = ['name','data']
    bio_df_category['type'] = 'column'
    bio_df_category.to_json(f'{SAVE_DIR}/biodiversity_1_total_score_by_category.json',orient='records')


    # Plot_6-2: Biodiversity total by Water_supply
    bio_df_Water_supply = bio_df.groupby(['Year','Land management']).sum(numeric_only=True).reset_index()
    bio_df_Water_supply = bio_df_Water_supply\
        .groupby('Land management')[['Year','Biodiversity score (million)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Biodiversity score (million)']))))\
        .reset_index()
        
    bio_df_Water_supply.columns = ['name','data']
    bio_df_Water_supply['type'] = 'column'
    bio_df_Water_supply.to_json(f'{SAVE_DIR}/biodiversity_2_total_score_by_Water_supply.json',orient='records')


    # Plot_6-3: Biodiversity total by landuse
    bio_df_landuse = bio_df.groupby(['Year','Landuse']).sum(numeric_only=True).reset_index()
    bio_df_landuse = bio_df_landuse.query('Landuse in @bio_lucc')

    bio_df_landuse = bio_df_landuse\
        .groupby('Landuse')[['Year','Biodiversity score (million)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Biodiversity score (million)']))))\
        .reset_index()
        
    bio_df_landuse['sort_index'] = bio_df_landuse['Landuse'].apply(lambda x: bio_lucc.index(x))
    bio_df_landuse = bio_df_landuse.sort_values('sort_index').drop('sort_index',axis=1)

    bio_df_landuse.columns = ['name','data']
    bio_df_landuse['type'] = 'column'
    bio_df_landuse.to_json(f'{SAVE_DIR}/biodiversity_3_total_score_by_landuse.json',orient='records')

    # Plot_6-4: Natural landuse area (million)
    natural_land_area = area_dvar.groupby(['Year','Land-use']).sum(numeric_only=True).reset_index()
    natural_land_area = natural_land_area.query('`Land-use` in @bio_lucc').copy()

    # million km2 to million ha
    natural_land_area.loc[:,'Area (million ha)'] = natural_land_area['Area (ha)'] / 1e6
    natural_land_area = natural_land_area\
        .groupby('Land-use')[['Year','Area (million ha)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million ha)']))))\
        .reset_index()
        
    natural_land_area['sort_index'] = natural_land_area['Land-use'].apply(lambda x: bio_lucc.index(x))
    natural_land_area = natural_land_area.sort_values(['sort_index']).drop('sort_index',axis=1)

    natural_land_area.columns = ['name','data']
    natural_land_area['type'] = 'column'
    natural_land_area.to_json(f'{SAVE_DIR}/biodiversity_4_natural_land_area.json',orient='records')
    
    
    # ---------------- Biodiversity contribution score  ----------------
    bio_paths = files.query('category == "biodiversity" and year_types == "single_year" and base_name == "biodiversity_contribution"').reset_index(drop=True)
    bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
    bio_df = bio_df.replace(RENAME_AM_NON_AG)                   # Rename the landuse
    
    bio_df['lm'] = bio_df['lm'].replace({
        'dry':'Dryland', 
        'irr':'Irrigated',
        np.nan: 'Dryland'})    # Replace `nan` to 'Dryland'
    
    bio_df['lu_type'] = bio_df['lu_type'].replace({
        'ag': 'Agricultural Landuse',
        'non_ag': 'Non-Agricultural Landuse',
        'am': 'Agricultural Management'})   
    
    bio_df['group'] = bio_df['group'].replace({
        'all_species': 'All species', 
        'amphibians': 'Amphibians',
        'birds': 'Birds',
        'mammals': 'Mammals',
        'plants': 'Plants',
        'reptiles': 'Reptiles'})
 
    
    # Plot_6-5: Biodiversity contribution score by group
    bio_df_species_group = bio_df.groupby(['group','year']).sum(numeric_only=True).reset_index()
    
    bio_df_species_group = bio_df_species_group\
        .groupby('group')[['year','contribution_%']]\
        .apply(lambda x:list(map(list,zip(x['year'],x['contribution_%']))))\
        .reset_index()
        
    bio_df_species_group.columns = ['name','data']
    bio_df_species_group['type'] = 'spline'
    bio_df_species_group.to_json(f'{SAVE_DIR}/biodiversity_5_contribution_score_by_group.json',orient='records')




    # Plot_6-6: Biodiversity contribution score by landuse broader type
    bio_df_landuse_type_broad = bio_df.query('group == "All species"').groupby(['lu_type','year']).sum(numeric_only=True).reset_index()
    bio_df_landuse_type_broad = bio_df_landuse_type_broad\
        .groupby('lu_type')[['year','contribution_%']]\
        .apply(lambda x:list(map(list,zip(x['year'],x['contribution_%']))))\
        .reset_index()
        
    bio_df_landuse_type_broad.columns = ['name','data']
    bio_df_landuse_type_broad['type'] = 'column'
    bio_df_landuse_type_broad.to_json(f'{SAVE_DIR}/biodiversity_6_contribution_score_by_landuse_type_broad.json',orient='records')
    
    
    # Plot_6-7: Biodiversity contribution score by specific landuse type
    bio_df_landuse_type_specific = bio_df.query('group == "All species"').groupby(['lu','year']).sum(numeric_only=True).reset_index()
    bio_df_landuse_type_specific = bio_df_landuse_type_specific\
        .groupby('lu')[['year','contribution_%']]\
        .apply(lambda x:list(map(list,zip(x['year'],x['contribution_%']))))\
        .reset_index()
        
    bio_df_landuse_type_specific.columns = ['name','data']
    bio_df_landuse_type_specific['type'] = 'column'
    bio_df_landuse_type_specific['sort_index'] = bio_df_landuse_type_specific['name'].apply(lambda x: LANDUSE_ALL.index(x))
    bio_df_landuse_type_specific = bio_df_landuse_type_specific.sort_values(['sort_index']).drop('sort_index',axis=1)
    bio_df_landuse_type_specific.to_json(f'{SAVE_DIR}/biodiversity_7_contribution_score_by_landuse_type_specific.json',orient='records')
    
    

    
    
    #########################################################
    #                         7) Maps                       #
    #########################################################
    map_files = files.query('base_ext == ".html" and year_types != "begin_end_year"')
    map_save_dir = f"{SAVE_DIR}/Map_data/"
    
    # Create the directory to save map_html if it does not exist
    if  not os.path.exists(map_save_dir):
        os.makedirs(map_save_dir)
    
    # Function to move a file from one location to another if the file exists
    def move_html(path_from, path_to):
        if os.path.exists(path_from):
            shutil.move(path_from, path_to)
    
    # Move the map files to the save directory
    tasks = [delayed(move_html)(row['path'], map_save_dir)
                for _,row in map_files.iterrows()]
    
    worker = min(settings.WRITE_THREADS, len(tasks)) if len(tasks) > 0 else 1
    
    Parallel(n_jobs=worker)(tasks)
    
    


    #########################################################
    #              Report success info                      #
    #########################################################

    print('Report data created successfully!\n')