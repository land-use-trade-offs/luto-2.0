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
import json
import re
import shutil
import pandas as pd
import numpy as np
import luto.settings as settings
from joblib import Parallel, delayed

from luto.economics.off_land_commodity import get_demand_df
from luto.tools.report.data_tools import get_all_files
from luto.tools.report.data_tools.helper_func import select_years

from luto.tools.report.data_tools.parameters import (
    AG_LANDUSE, 
    COMMODITIES_ALL,
    COMMODITIES_OFF_LAND, 
    GHG_CATEGORY, 
    GHG_NAMES,
    LANDUSE_ALL_RENAMED,
    LU_CROPS, 
    LVSTK_MODIFIED, 
    LVSTK_NATURAL, 
    RENAME_NON_AG,
    RENAME_AM_NON_AG
)


def save_report_data(raw_data_dir:str):
    """
    Saves the report data in the specified directory.

    Parameters
    raw_data_dir (str): The directory path where the raw output data is.

    Returns
    None
    """
    # Set the save directory
    SAVE_DIR = f'{raw_data_dir}/DATA_REPORT/data'
    
    # Select the years to reduce the column number to avoid cluttering in the multi-level axis graphing
    years = sorted(settings.SIM_YEARS)
    years_select = select_years(years)
    
    # Create the directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir).reset_index(drop=True)
    files['Year'] = files['Year'].astype(int)
    files = files.query('Year.isin(@years)')
    
    # The land-use groupings to combine the land-use into a single category
    lu_group = pd.read_csv('luto/tools/report/Assets/lu_group.csv')
    lu_group_expand = lu_group.set_index(['Category', 'color_HEX']).apply(lambda x: x.str.split(', ').explode()).reset_index()
    




    ####################################################
    #                    1) Area Change                #
    ####################################################

    area_dvar_paths = files.query('category == "area" and year_types == "single_year"').reset_index(drop=True)
    
    ag_dvar_dfs = area_dvar_paths.query('base_name == "area_agricultural_landuse"').reset_index(drop=True)
    ag_dvar_area = pd.concat([pd.read_csv(path) for path in ag_dvar_dfs['path']], ignore_index=True)
    ag_dvar_area['Type'] = 'Agricultural landuse'
    ag_dvar_area['Area (ha)'] = ag_dvar_area['Area (ha)'].round(2)

    non_ag_dvar_dfs = area_dvar_paths.query('base_name == "area_non_agricultural_landuse"').reset_index(drop=True)
    non_ag_dvar_area = pd.concat([pd.read_csv(path) for path in non_ag_dvar_dfs['path'] if not pd.read_csv(path).empty], ignore_index=True)
    non_ag_dvar_area['Type'] = 'Non-agricultural landuse'
    non_ag_dvar_area['Area (ha)'] = non_ag_dvar_area['Area (ha)'].round(2)

    am_dvar_dfs = area_dvar_paths.query('base_name == "area_agricultural_management"').reset_index(drop=True)
    am_dvar_area = pd.concat([pd.read_csv(path) for path in am_dvar_dfs['path'] if not pd.read_csv(path).empty], ignore_index=True)
    am_dvar_area = am_dvar_area.replace(RENAME_AM_NON_AG)
    am_dvar_area['Area (ha)'] = am_dvar_area['Area (ha)'].round(2)

    area_dvar = pd.concat([ag_dvar_area, non_ag_dvar_area], ignore_index=True)
    area_dvar = area_dvar.replace(RENAME_AM_NON_AG)
    
    # Add the category and color
    area_dvar = area_dvar.merge(lu_group_expand, left_on='Land-use', right_on='Land-use', how='left')
    area_dvar = area_dvar.rename(columns={
        'Category': 'category_name',
        'color_HEX': 'category_HEX_color'
    })
    
    
    
    # Total Area (ha) split by different subcategories
    group_cols = ['Water_supply', 'Land-use', 'Type', 'category_name']
    
    for idx, col in enumerate(group_cols):

        df_AUS = area_dvar\
            .groupby(['Year', col])[['Area (ha)']]\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Area (ha)']]\
            .apply(lambda x: list(map(list,zip(x['Year'], x['Area (ha)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        
        df_region = area_dvar\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Area (ha)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Area (ha)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        if col == "Land-use":
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif 'commodity' in col.lower():
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            out_dict[region] = df.to_dict(orient='records')


        with open(f'{SAVE_DIR}/Area_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
            


    # Agricultural management Area (ha) Land use
    group_cols = ['Type', 'Water_supply', 'Land-use']
    
    for idx, col in enumerate(group_cols):

        df_AUS = am_dvar_area\
            .groupby(['Year', col])[['Area (ha)']]\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Area (ha)']]\
            .apply(lambda x: list(map(list,zip(x['Year'], x['Area (ha)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        
        df_region = am_dvar_area\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Area (ha)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Area (ha)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        if col == "Land-use":
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif 'commodity' in col.lower():
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Area_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
            
            


    # Transition areas
    transition_path = files.query('category =="transition_matrix"')
    transition_df_area = pd.read_csv(transition_path['path'].values[0], index_col=0).reset_index() 
    transition_df_area['Area (km2)'] = transition_df_area['Area (ha)'] / 100   
    transition_df_area = transition_df_area.replace(RENAME_AM_NON_AG)

    transition_regions = list(transition_df_area.groupby('region'))
    transition_regions += [
        (
            'AUSTRALIA', 
            transition_df_area.groupby(['From Land-use', 'To Land-use'])[['Area (km2)']].sum().reset_index()
        )
    ]
    
    
    out_dict = {}
    for (region, df) in transition_regions:
        transition_mat = df.pivot(index='From Land-use', columns='To Land-use', values='Area (km2)')
        transition_mat = transition_mat.reindex(index=AG_LANDUSE, columns=LANDUSE_ALL_RENAMED)
        transition_mat = transition_mat.fillna(0)
        total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
        
        transition_df_pct = transition_mat / total_area_from * 100
        transition_df_pct = transition_df_pct.fillna(0).replace([np.inf, -np.inf], 0)

        transition_mat['SUM'] = transition_mat.sum(axis=1)
        transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

        heat_area = transition_mat.style.background_gradient(
            cmap='Oranges',
            axis=1,
            subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
        ).format('{:,.0f}')

        heat_pct = transition_df_pct.style.background_gradient(
            cmap='Oranges',
            axis=1,
            vmin=0,
            vmax=100
        ).format('{:,.2f}')

        heat_area_html = heat_area.to_html()
        heat_pct_html = heat_pct.to_html()

        # Replace '0.00' with '-' in the html
        heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)
        heat_pct_html = re.sub(r'(?<!\d)0.00(?!\d)', '-', heat_pct_html)

        out_dict[region] = {
            'area': heat_area_html,
            'pct': heat_pct_html
        }
        
    with open(f'{SAVE_DIR}/Area_transition.json', 'w') as f:
        json.dump(out_dict, f)


    ####################################################
    #                   2) Demand                      #
    ####################################################
    
    # Get the demand data
    DEMAND_DATA_long = get_demand_df().query(f'Year.isin({years_select})')
    DEMAND_DATA_long['Quantity (tonnes, KL)'] = DEMAND_DATA_long['Quantity (tonnes, KL)'].round(2)
    
    # Rename the commodities index for Beef and Sheep live export
    DEMAND_DATA_long = DEMAND_DATA_long.rename(
        index={'Beef lexp': 'Beef live export', 'Sheep lexp': 'Sheep live export'}, level=1
    )

    # Reorder the columns to match the order in COMMODITIES_ALL
    DEMAND_DATA_long = DEMAND_DATA_long.reindex(COMMODITIES_ALL, level=1
        ).reset_index(
        ).replace(RENAME_AM_NON_AG)

    # Add columns for on-land and off-land commodities
    DEMAND_DATA_long['on_off_land'] = DEMAND_DATA_long['Commodity'].apply(
        lambda x: 'On-land' if x not in COMMODITIES_OFF_LAND else 'Off-land'
    )


    # Plot_2_1: {Total} for 'Domestic', 'Exports', 'Feed', 'Imports', 'Production'(Tonnes) 
    DEMAND_DATA_type_AUS = DEMAND_DATA_long.groupby(['Year','Type']
        ).sum(numeric_only=True
        ).reset_index()
    DEMAND_DATA_type_AUS = DEMAND_DATA_type_AUS\
        .groupby('Type')[['Year','Quantity (tonnes, KL)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Quantity (tonnes, KL)']))))\
        .reset_index()
    
    DEMAND_DATA_type_AUS.columns = ['name','data']
    DEMAND_DATA_type_AUS['type'] = 'column'
    DEMAND_DATA_type_AUS.to_json(f'{SAVE_DIR}/production_1_demand_type_wide_AUSTRALIA.json', orient='records')


    # Plot_2_2: {ON/OFF land} for 'Domestic', 'Exports', 'Feed', 'Imports', 'Production'(Tonnes)
    # For Australia (national level)
    DEMAND_DATA_on_off_AUS = DEMAND_DATA_long.groupby(['Year','Type','on_off_land']).sum(numeric_only=True).reset_index()
    DEMAND_DATA_on_off_AUS = DEMAND_DATA_on_off_AUS.sort_values(['on_off_land','Year','Type'])
    
    DEMAND_DATA_on_off_series_AUS = DEMAND_DATA_on_off_AUS[['on_off_land','Quantity (tonnes, KL)']]\
        .groupby('on_off_land')[['on_off_land','Quantity (tonnes, KL)']]\
        .apply(lambda x: x['Quantity (tonnes, KL)'].tolist())\
        .reset_index()
    DEMAND_DATA_on_off_series_AUS.columns = ['name','data']
    DEMAND_DATA_on_off_series_AUS['type'] = 'column'
    
    DEMAND_DATA_on_off_categories_AUS = DEMAND_DATA_on_off_AUS.query('on_off_land == "On-land"')[['Year','Type']]\
        .groupby('Year')[['Year','Type']]\
        .apply(lambda x:x['Type'].tolist())\
        .reset_index()
    DEMAND_DATA_on_off_categories_AUS.columns = ['name','categories']
    
    combined_json_AUS = {
        'series': json.loads(DEMAND_DATA_on_off_series_AUS.to_json(orient='records')),
        'categories': json.loads(DEMAND_DATA_on_off_categories_AUS.to_json(orient='records'))
    }
    
    # Convert the dictionary to a JSON string and save
    combined_json_str_AUS = json.dumps(combined_json_AUS)  
    with open(f'{SAVE_DIR}/production_2_demand_on_off_wide_AUSTRALIA.json', 'w') as outfile:
        outfile.write(combined_json_str_AUS)
    
    
    # Plot_2_3: {Commodity} 'Domestic', 'Exports', 'Feed', 'Imports', 'Production' (Tonnes)
    # For Australia (national level)
    DEMAND_DATA_commodity_AUS = DEMAND_DATA_long.groupby(['Year','Type','Commodity']
        ).sum(numeric_only=True
        ).reset_index()
    DEMAND_DATA_commodity_AUS = DEMAND_DATA_commodity_AUS.sort_values(['Commodity','Year','Type'])
    
    DEMAND_DATA_commodity_series_AUS = DEMAND_DATA_commodity_AUS[['Commodity', 'Quantity (tonnes, KL)']]\
        .groupby('Commodity')[['Quantity (tonnes, KL)']]\
        .apply(lambda x: x['Quantity (tonnes, KL)'].tolist())\
        .reset_index()
                                    
    DEMAND_DATA_commodity_series_AUS.columns = ['name','data']
    DEMAND_DATA_commodity_series_AUS['type'] = 'column' 
    DEMAND_DATA_commodity_series_AUS = DEMAND_DATA_commodity_series_AUS.set_index('name').reindex(COMMODITIES_ALL).reset_index()
    DEMAND_DATA_commodity_series_AUS = DEMAND_DATA_commodity_series_AUS.dropna()
    
    DEMAND_DATA_commodity_categories_AUS = DEMAND_DATA_commodity_AUS.query('Commodity == "Apples"')[['Year','Type']]\
        .groupby('Year')[['Year','Type']]\
        .apply(lambda x: x['Type'].tolist())\
        .reset_index()
    DEMAND_DATA_commodity_categories_AUS.columns = ['name','categories']
    
    combined_json_AUS = {   
        'series': json.loads(DEMAND_DATA_commodity_series_AUS.to_json(orient='records')),
        'categories': json.loads(DEMAND_DATA_commodity_categories_AUS.to_json(orient='records'))
    }
    
    combined_json_str_AUS = json.dumps(combined_json_AUS)
    with open(f'{SAVE_DIR}/production_3_demand_commodity_AUSTRALIA.json', 'w') as outfile:
        outfile.write(combined_json_str_AUS)
    


    # Plot_2-4_(1-2): Domestic On/Off land commodities
    # For Australia (national level)
    for idx,on_off_land in enumerate(DEMAND_DATA_long['on_off_land'].unique()):
        
        DEMAND_DATA_on_off_commodity_AUS = DEMAND_DATA_long.query('on_off_land == @on_off_land and Type == "Domestic" ')
        DEMAND_DATA_on_off_commodity_wide_AUS = DEMAND_DATA_on_off_commodity_AUS\
            .groupby(['Commodity'])[['Year','Quantity (tonnes, KL)']]\
            .apply(lambda x: list(zip(x['Year'],x['Quantity (tonnes, KL)'])))\
            .reset_index()
    
        DEMAND_DATA_on_off_commodity_wide_AUS.columns = ['name','data']
        DEMAND_DATA_on_off_commodity_wide_AUS['type'] = 'column'
        DEMAND_DATA_on_off_commodity_wide_AUS.to_json(
            f'{SAVE_DIR}/production_4_{idx+1}_demand_domestic_{on_off_land}_commodity_AUSTRALIA.json',
            orient='records'
        )
        
        

    # Plot_2-5_(2-5): Commodities for 'Exports','Feed','Imports','Production'
    # For Australia (national level)
    for idx,Type in enumerate(DEMAND_DATA_long['Type'].unique()):
        if Type == 'Domestic':
            continue
        
        DEMAND_DATA_commodity_AUS = DEMAND_DATA_long.query('Type == @Type')
        DEMAND_DATA_commodity_wide_AUS = DEMAND_DATA_commodity_AUS\
            .groupby(['Commodity'])[['Year','Quantity (tonnes, KL)']]\
            .apply(lambda x: list(zip(x['Year'],x['Quantity (tonnes, KL)'])))\
            .reset_index()
        
        DEMAND_DATA_commodity_wide_AUS.columns = ['name','data']
        DEMAND_DATA_commodity_wide_AUS['type'] = 'column'
        DEMAND_DATA_commodity_wide_AUS.to_json(
            f'{SAVE_DIR}/production_5_{idx+1}_demand_{Type}_commodity_AUSTRALIA.json',
            orient='records'
        )



    # Plot_2-5-6: Production (LUTO outputs)
    filter_str = '''
        category == "quantity" 
        and base_name == "quantity_production_t_separate" 
        and year_types == "single_year"
    '''.replace('\n','').replace('    ','')
    
    quantity_csv_paths = files.query(filter_str).reset_index(drop=True).query('Year.isin(@years)')
    quantity_df = pd.concat(
        [pd.read_csv(path).assign(Year=Year) for Year,path in quantity_csv_paths[['Year','path']].values.tolist()],
        ignore_index=True
    )
    quantity_df['Commodity'] = quantity_df['Commodity'].str.capitalize()
    quantity_df = quantity_df.replace({'Sheep lexp': 'Sheep live export', 'Beef lexp': 'Beef live export'})
    quantity_df['Production (t/KL)'] = quantity_df['Production (t/KL)'].round(2)
    
    # For Australia (national level)
    quantity_df_wide_AUS = quantity_df\
        .groupby(['Commodity'])[['Year','Production (t/KL)']]\
        .apply(lambda x: list(map(list,zip(x['Year'],x['Production (t/KL)']))))\
        .reset_index()\
        .assign(region='AUSTRALIA')
    quantity_df_wide_AUS = quantity_df_wide_AUS.set_index('Commodity').reindex(COMMODITIES_ALL).reset_index()
    quantity_df_wide_AUS = quantity_df_wide_AUS.dropna()
    
    quantity_df_wide_region = quantity_df.groupby(['region', 'Commodity'])[['Year','Production (t/KL)']]\
        .apply(lambda x: list(map(list,zip(x['Year'],x['Production (t/KL)']))))\
        .reset_index()
    
    quantity_df_wide = pd.concat([quantity_df_wide_AUS, quantity_df_wide_region], axis=0, ignore_index=True)

    
    for region, df in quantity_df_wide.groupby('region'):
        df = df.drop('region', axis=1)
        df.columns = ['name','data']
        df['type'] = 'column'
        df.to_json(
            f'{SAVE_DIR}/production_5_6_demand_Production_commodity_from_LUTO_{region.replace("/", "_")}.json',
            orient='records'
        )


    # Plot_2-7: Demand achievement in the final target year (%)
    filter_str = '''
        category == "quantity"
        and base_name == "quantity_comparison"
        and year_types == "single_year"
    '''.replace('\n','').replace('    ',' ')
    
    quantify_diff = files.query(filter_str).reset_index(drop=True)
    quantify_diff = pd.concat([pd.read_csv(path) for path in quantify_diff['path']], ignore_index=True)
    quantify_diff = quantify_diff.replace({'Sheep lexp': 'Sheep live export', 'Beef lexp': 'Beef live export'})
    quantify_diff = quantify_diff[['Year','Commodity','Prop_diff (%)']].rename(columns={'Prop_diff (%)': 'Demand Achievement (%)'})

    # For Australia (national level)
    # Remove rows where Demand Achievement (%) is 100 across all years
    mask_AUS = quantify_diff.groupby('Commodity'
        )['Demand Achievement (%)'
        ].transform(lambda x: abs(round(x) - 100) > 0.01)
    quantify_diff_AUS = quantify_diff[mask_AUS].copy()

    quantify_diff_wide_AUS = quantify_diff_AUS\
        .groupby(['Commodity'])[['Year','Demand Achievement (%)']]\
        .apply(lambda x: list(map(list,zip(x['Year'],x['Demand Achievement (%)']))))\
        .reset_index()
        
    # Set attributes
    quantify_diff_wide_AUS['type'] = 'spline'
    quantify_diff_wide_AUS['showInLegend'] = True
    quantify_diff_wide_AUS.columns = ['name','data', 'type', 'showInLegend']
    quantify_diff_wide_AUS.to_json(f'{SAVE_DIR}/production_6_demand_achievement_commodity_AUSTRALIA.json', orient='records')





    ####################################################
    #                  3) Economics                    #
    ####################################################
    
    # Get the revenue and cost data
    revenue_ag_df = files.query('base_name == "revenue_agricultural_commodity" and year_types != "begin_end_year"').reset_index(drop=True)
    revenue_ag_df = pd.concat([pd.read_csv(path) for path in revenue_ag_df['path']], ignore_index=True)
    revenue_ag_df = revenue_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural land-use (revenue)')

    revenue_am_df = files.query('base_name == "revenue_agricultural_management" and year_types != "begin_end_year"').reset_index(drop=True)
    revenue_am_df = pd.concat([pd.read_csv(path) for path in revenue_am_df['path']], ignore_index=True)
    revenue_am_df = revenue_am_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural management (revenue)')

    revenue_non_ag_df = files.query('base_name == "revenue_non_ag" and year_types != "begin_end_year"').reset_index(drop=True)
    revenue_non_ag_df = pd.concat([pd.read_csv(path) for path in revenue_non_ag_df['path']], ignore_index=True)
    revenue_non_ag_df = revenue_non_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Non-agricultural land-use (revenue)')

    cost_ag_df = files.query('base_name == "cost_agricultural_commodity" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_ag_df = pd.concat([pd.read_csv(path) for path in cost_ag_df['path']], ignore_index=True)
    cost_ag_df = cost_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural land-use (cost)')
    cost_ag_df['Value ($)'] = cost_ag_df['Value ($)'] * -1

    cost_am_df = files.query('base_name == "cost_agricultural_management" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_am_df = pd.concat([pd.read_csv(path) for path in cost_am_df['path']], ignore_index=True)
    cost_am_df = cost_am_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural management (cost)')
    cost_am_df['Value ($)'] = cost_am_df['Value ($)'] * -1

    cost_non_ag_df = files.query('base_name == "cost_non_ag" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_non_ag_df = pd.concat([pd.read_csv(path) for path in cost_non_ag_df['path']], ignore_index=True)
    cost_non_ag_df = cost_non_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Non-agricultural land-use (cost)')
    cost_non_ag_df['Value ($)'] = cost_non_ag_df['Value ($)'] * -1

    cost_transition_ag2ag_df = files.query('base_name == "cost_transition_ag2ag" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_transition_ag2ag_df = pd.concat([pd.read_csv(path) for path in cost_transition_ag2ag_df['path'] if not pd.read_csv(path).empty], ignore_index=True)
    cost_transition_ag2ag_df = cost_transition_ag2ag_df.replace(RENAME_AM_NON_AG).assign(Source='Transition cost (Ag2Ag)')
    cost_transition_ag2ag_df['Value ($)'] = cost_transition_ag2ag_df['Cost ($)']  * -1
    

    cost_transition_ag2non_ag_df = files.query('base_name == "cost_transition_ag2non_ag" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_transition_ag2non_ag_df = pd.concat([pd.read_csv(path) for path in cost_transition_ag2non_ag_df['path'] if not pd.read_csv(path).empty], ignore_index=True)
    cost_transition_ag2non_ag_df = cost_transition_ag2non_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Transition cost (Ag2Non-Ag)')
    cost_transition_ag2non_ag_df['Value ($)'] = cost_transition_ag2non_ag_df['Cost ($)'] * -1

    cost_transition_non_ag2ag_df = files.query('base_name == "cost_transition_non_ag2_ag" and year_types != "begin_end_year"').reset_index(drop=True)
    cost_transition_non_ag2ag_df = pd.concat([pd.read_csv(path) for path in cost_transition_non_ag2ag_df['path'] if not pd.read_csv(path).empty], ignore_index=True)
    cost_transition_non_ag2ag_df = cost_transition_non_ag2ag_df.replace(RENAME_AM_NON_AG).assign(Source='Transition cost (Non-Ag2Ag)').dropna(subset=['Cost ($)'])
    cost_transition_non_ag2ag_df['Value ($)'] = cost_transition_non_ag2ag_df['Cost ($)'] * -1

    economics_df = pd.concat(
            [
                revenue_ag_df, 
                revenue_am_df, 
                revenue_non_ag_df,
                cost_ag_df, 
                cost_am_df, 
                cost_non_ag_df,
                cost_transition_ag2ag_df, 
                cost_transition_ag2non_ag_df,
                # cost_transition_non_ag2ag_df
            ], axis=0
        ).reset_index(drop=True
        ).round({'Value ($)': 2})



    # Plot_3-0: Revenue and Cost data for all types 
    # For Australia (national level)
    economics_df_AUS = economics_df.groupby(['Year','Source']
        ).sum(numeric_only=True
        ).reset_index()

    economics_df_AUS_wide = economics_df_AUS\
        .groupby(['Source'])[['Year','Value ($)']]\
        .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
        .reset_index()
        
    economics_df_AUS_wide.columns = ['name','data']
    economics_df_AUS_wide['type'] = 'column'
    economics_df_AUS_wide['region'] = 'AUSTRALIA'
    economics_df_AUS_wide.loc[len(economics_df_AUS_wide)] = [
        'Profit', 
        economics_df_AUS.groupby('Year')['Value ($)'].sum().reset_index().values.tolist(), 
        'spline', 
        'AUSTRALIA'
    ]


    rev_cost_net_region = economics_df.groupby(['Year','Source','region']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index()
        
    rev_cost_net_wide_region = pd.DataFrame()
    for region, df in rev_cost_net_region.groupby('region'):
        df_col = df.groupby(['Source'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()
        df_col.columns = ['name','data']
        df_col['type'] = 'column'
        
        df_col.loc[len(df_col)] = [
            'Profit',
            df.groupby(['Year'])[['Value ($)']].sum(numeric_only=True).reset_index().values.tolist(),
            'spline',
        ]
        df_col['region'] = region
        rev_cost_net_wide_region = pd.concat([rev_cost_net_wide_region, df_col])

    

    # Define the specific order
    order = [
        'Agricultural land-use (revenue)', 
        'Agricultural management (revenue)', 
        'Non-agricultural land-use (revenue)',
        'Agricultural land-use (cost)', 
        'Agricultural management (cost)', 
        'Non-agricultural land-use (cost)',
        'Transition cost (Ag2Ag)',
        'Transition cost (Ag2Non-Ag)',
        'Transition cost (Non-Ag2Ag)',
        'Profit'
    ]

    rev_cost_wide_json = pd.concat([economics_df_AUS_wide, rev_cost_net_wide_region], axis=0).reset_index(drop=True)
    rev_cost_wide_json['name_order'] = rev_cost_wide_json['name'].map({name: i for i, name in enumerate(order)})
    rev_cost_wide_json = rev_cost_wide_json.sort_values(['region', 'name_order']).drop(columns=['name_order']).reset_index(drop=True)

    # Save to disk
    for region,df in rev_cost_wide_json.groupby('region'):
        df = df.drop(columns='region')
        df.columns = ['name','data','type']
        df.to_json(f'{SAVE_DIR}/economics_0_rev_cost_all_wide_{region.replace("/", "_")}.json', orient='records')


    

    # Plot_3-1: Revenue for Agricultural land-use 
    keep_cols = ['Year', 'region', 'Value ($)', 'Source']
    group_cols = ['Land-use', 'Type', 'Water_supply']
    
    for idx, col in enumerate(group_cols):
        
        take_cols = keep_cols + [col]
        
        df_AUS = revenue_ag_df[take_cols]\
            .groupby(['Year', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'], x['Value ($)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        
        df_region = revenue_ag_df[take_cols]\
            .groupby(['Year', 'region', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        if col == "Land-use":
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            # save to disk
            df.to_json(f'{SAVE_DIR}/economics_1_ag_revenue_{idx+1}_{col}_wide_{region.replace("/", "_")}.json', orient='records')




    # Plot_3-2: Cost for Agricultural land-use 
    cost_ag_df['Value ($)'] = cost_ag_df['Value ($)'] * -1  # Convert from negative to positive
    keep_cols = ['Year', 'region', 'Value ($)', 'Source']
    group_cols = ['Land-use', 'Type', 'Water_supply']
    
    for idx,col in enumerate(group_cols):
        
        take_cols = keep_cols + [col]
        
        df_AUS = cost_ag_df[take_cols]\
            .groupby(['Year', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby(col)[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        df_region = cost_ag_df[take_cols]\
            .groupby(['Year', 'region', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()\
            .assign(type='column')
        df_region_wide.columns = ['name', 'region','data','type']
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        if col == "Land-use":
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            # save to disk
            df.to_json(f'{SAVE_DIR}/economics_2_ag_cost_{idx+1}_{col}_wide_{region.replace("/", "_")}.json', orient='records')


    
    # Plot_3-4: Revenue for Agricultural Management 
    keep_cols = ['Year', 'region', 'Value ($)', 'Source']
    group_cols = ['Land-use', 'Management Type', 'Water_supply']
    
    for idx,col in enumerate(group_cols):
        take_cols = keep_cols + [col]
        
        df_AUS = revenue_am_df[take_cols]\
            .groupby(['Year', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        df_region = revenue_am_df[take_cols]\
            .groupby(['Year', 'region', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        if col == "Land-use":
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            # save to disk
            df.to_json(f'{SAVE_DIR}/economics_4_am_revenue_{idx+1}_{col}_wide_{region.replace("/", "_")}.json', orient='records')



    # Plot_3-5: Cost for Agricultural Management 
    cost_am_df['Value ($)'] = cost_am_df['Value ($)'] * -1  # Convert from negative to positive
    keep_cols = ['Year', 'region', 'Value ($)', 'Source']
    group_cols = ['Land-use', 'Management Type', 'Water_supply']
    
    for idx,col in enumerate(group_cols):
        take_cols = keep_cols + [col]
        
        df_AUS = cost_am_df[take_cols]\
            .groupby(['Year', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        df_region = cost_am_df[take_cols]\
            .groupby(['Year', 'region', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        if col == "Land-use":
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            # save to disk
            df.to_json(f'{SAVE_DIR}/economics_5_am_cost_{idx+1}_{col}_wide_{region.replace("/", "_")}.json', orient='records')
        
        
        
        
    # Plot_3-6: Revenue for Non-Agricultural land-use 
    keep_cols = ['Year', 'region', 'Value ($)', 'Source']
    group_cols = ['Land-use']
    
    for idx,col in enumerate(group_cols):
        take_cols = keep_cols + [col]
        
        df_AUS = revenue_non_ag_df[take_cols]\
            .groupby(['Year', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        df_region = revenue_non_ag_df[take_cols]\
            .groupby(['Year', 'region', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        if col == "Land-use":
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            # save to disk
            df.to_json(f'{SAVE_DIR}/economics_6_non_ag_revenue_{idx+1}_{col}_wide_{region.replace("/", "_")}.json', orient='records')
        
        
        
        
    # Plot_3-7: Cost for Non-Agricultural land-use 
    cost_non_ag_df['Value ($)'] = cost_non_ag_df['Value ($)'] * -1  # Convert from negative to positive
    keep_cols = ['Year', 'region', 'Value ($)', 'Source']
    group_cols = ['Land-use']
    
    for idx,col in enumerate(group_cols):
        take_cols = keep_cols + [col]
        
        df_AUS = cost_non_ag_df[take_cols]\
            .groupby(['Year', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        df_region = cost_non_ag_df[take_cols]\
            .groupby(['Year', 'region', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        if col == "Land-use":
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            # save to disk
            df.to_json(f'{SAVE_DIR}/economics_7_non_ag_cost_{idx+1}_{col}_wide_{region.replace("/", "_")}.json', orient='records')
    
    
    
    
    # Plot_3-8: Transition cost for Ag2Ag 
    cost_transition_ag2ag_df['Value ($)'] = cost_transition_ag2ag_df['Value ($)'] * -1  # Convert from negative to positive
    keep_cols = ['Year', 'region', 'Value ($)']
    group_cols = ['Type', 'From land-use', 'To land-use', 'Source']
    
    for idx,col in enumerate(group_cols):
        take_cols = keep_cols + [col]
        
        df_AUS = cost_transition_ag2ag_df[take_cols]\
            .groupby(['Year', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        df_region = cost_transition_ag2ag_df[take_cols]\
            .groupby(['Year', 'region', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            # save to disk
            df.to_json(f'{SAVE_DIR}/economics_8_transition_ag2ag_cost_{idx+1}_{col}_wide_{region.replace("/", "_")}.json', orient='records')
        
        
        
    
    # Transition cost matrix for AUSTRALIA
    cost_transition_ag2ag_trans_mat_AUS = cost_transition_ag2ag_df\
        .groupby(['Year','From land-use', 'To land-use'])\
        .sum(numeric_only=True)\
        .reset_index()
    cost_transition_ag2ag_trans_mat_AUS = cost_transition_ag2ag_trans_mat_AUS\
        .set_index(['Year','From land-use', 'To land-use'])\
        .query('abs(`Value ($)`) > 0')
    cost_transition_ag2ag_trans_mat_AUS = cost_transition_ag2ag_trans_mat_AUS\
        .reindex(
            index = pd.MultiIndex.from_product([years, AG_LANDUSE, AG_LANDUSE], 
            names = ['Year','From land-use', 'To land-use'])
        ).reset_index()

    cost_transition_ag2ag_trans_mat_AUS['idx_from'] = cost_transition_ag2ag_trans_mat_AUS['From land-use']\
        .apply(lambda x: AG_LANDUSE.index(x))
    cost_transition_ag2ag_trans_mat_AUS['idx_to'] = cost_transition_ag2ag_trans_mat_AUS['To land-use']\
        .apply(lambda x: AG_LANDUSE.index(x))
                                                    
    cost_transition_ag2ag_trans_mat_AUS_data = cost_transition_ag2ag_trans_mat_AUS\
        .groupby(['Year'])[['idx_from','idx_to', 'Value ($)']]\
        .apply(lambda x: list(map(list,zip(x['idx_from'],x['idx_to'],x['Value ($)']))))\
        .reset_index()                             
    cost_transition_ag2ag_trans_mat_AUS_data.columns = ['Year','data']
    
    cost_transition_ag2ag_trans_mat_AUS_json = {
        'categories': AG_LANDUSE,
        'series': json.loads(cost_transition_ag2ag_trans_mat_AUS_data.to_json(orient='records'))
    }

    with open(f'{SAVE_DIR}/economics_8_transition_ag2ag_cost_4_transition_matrix_AUSTRALIA.json', 'w') as outfile:
        outfile.write(json.dumps(cost_transition_ag2ag_trans_mat_AUS_json))
        
        
        
    # Transition cost matrix for each region
    cost_transition_ag2ag_trans_mat_region = cost_transition_ag2ag_df\
        .groupby(['Year','From land-use', 'To land-use', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()
    cost_transition_ag2ag_trans_mat_region = cost_transition_ag2ag_trans_mat_region\
        .set_index(['Year','From land-use', 'To land-use', 'region'])\
        .query('abs(`Value ($)`) > 0')
    cost_transition_ag2ag_trans_mat_region = cost_transition_ag2ag_trans_mat_region\
        .reindex(
            index = pd.MultiIndex.from_product([years, AG_LANDUSE, AG_LANDUSE, cost_transition_ag2ag_df['region'].unique()],
            names = ['Year','From land-use', 'To land-use', 'region'])
        ).reset_index()

    cost_transition_ag2ag_trans_mat_region['idx_from'] = cost_transition_ag2ag_trans_mat_region['From land-use']\
        .apply(lambda x: AG_LANDUSE.index(x))
    cost_transition_ag2ag_trans_mat_region['idx_to'] = cost_transition_ag2ag_trans_mat_region['To land-use']\
        .apply(lambda x: AG_LANDUSE.index(x))

    for region,df in cost_transition_ag2ag_trans_mat_region.groupby('region'):
        df = df.groupby('Year')[['idx_from', 'idx_to', 'Value ($)']]\
            .apply(lambda x: x[['idx_from','idx_to', 'Value ($)']].values.tolist())\
            .reset_index()
        df.columns = ['Year','data']
        
        region_json = {
            'categories': AG_LANDUSE,
            'series': json.loads(df.to_json(orient='records'))
        }
        
        with open(f'{SAVE_DIR}/economics_8_transition_ag2ag_cost_4_transition_matrix_{region.replace("/", "_")}.json', 'w') as outfile:
            outfile.write(json.dumps(region_json))



    # Plot_3-9: Transition cost for Ag2Non-Ag 
    cost_transition_ag2non_ag_df['Value ($)'] = cost_transition_ag2non_ag_df['Value ($)'] * -1  # Convert from negative to positive
    keep_cols = ['Year', 'region', 'Value ($)']
    group_cols = ['Cost type', 'From land-use', 'To land-use', 'Source']
    
    for idx,col in enumerate(group_cols):
        take_cols = keep_cols + [col]
        
        df_AUS = cost_transition_ag2non_ag_df[take_cols]\
            .groupby(['Year', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        df_region = cost_transition_ag2non_ag_df[take_cols]\
            .groupby(['Year', 'region', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            # save to disk
            df.to_json(f'{SAVE_DIR}/economics_9_transition_ag2non_cost_{idx+1}_{col}_wide_{region.replace("/", "_")}.json', orient='records')
        
        
    # Transition cost matrix for AUSTRALIA
    cost_transition_ag2non_ag_trans_mat_AUS = cost_transition_ag2non_ag_df\
        .groupby(['Year','From land-use', 'To land-use'])\
        .sum(numeric_only=True)\
        .reset_index()
    cost_transition_ag2non_ag_trans_mat_AUS = cost_transition_ag2non_ag_trans_mat_AUS\
        .set_index(['Year','From land-use', 'To land-use'])\
        .query('abs(`Value ($)`) > 0')
    cost_transition_ag2non_ag_trans_mat_AUS = cost_transition_ag2non_ag_trans_mat_AUS\
        .reindex(
            index = pd.MultiIndex.from_product([years, AG_LANDUSE, RENAME_NON_AG.values()],
            names = ['Year','From land-use', 'To land-use'])
        ).reset_index()

    cost_transition_ag2non_ag_trans_mat_AUS['idx_from'] = cost_transition_ag2non_ag_trans_mat_AUS['From land-use']\
        .apply(lambda x: AG_LANDUSE.index(x))
    cost_transition_ag2non_ag_trans_mat_AUS['idx_to'] = cost_transition_ag2non_ag_trans_mat_AUS['To land-use']\
        .apply(lambda x: list(RENAME_NON_AG.values()).index(x))
                                                    
    cost_transition_ag2non_ag_trans_mat_AUS_data = cost_transition_ag2non_ag_trans_mat_AUS\
        .groupby(['Year'])[['idx_from','idx_to', 'Value ($)']]\
        .apply(lambda x: list(map(list,zip(x['idx_from'],x['idx_to'],x['Value ($)']))))\
        .reset_index()                             
    cost_transition_ag2non_ag_trans_mat_AUS_data.columns = ['Year','data']
    
    cost_transition_ag2non_ag_trans_mat_AUS_json = {
        'categories_from': AG_LANDUSE,
        'categories_to': list(RENAME_NON_AG.values()),
        'series': json.loads(cost_transition_ag2non_ag_trans_mat_AUS_data.to_json(orient='records'))
    }

    with open(f'{SAVE_DIR}/economics_9_transition_ag2non_cost_4_transition_matrix_AUSTRALIA.json', 'w') as outfile:
        outfile.write(json.dumps(cost_transition_ag2non_ag_trans_mat_AUS_json))
        
        
        
    # Transition cost matrix for each region
    cost_transition_ag2non_ag_trans_mat_region = cost_transition_ag2non_ag_df\
        .groupby(['Year','From land-use', 'To land-use', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()
    cost_transition_ag2non_ag_trans_mat_region = cost_transition_ag2non_ag_trans_mat_region\
        .set_index(['Year','From land-use', 'To land-use', 'region'])\
        .query('abs(`Value ($)`) > 0')
    cost_transition_ag2non_ag_trans_mat_region = cost_transition_ag2non_ag_trans_mat_region\
        .reindex(
            index = pd.MultiIndex.from_product([years, AG_LANDUSE, RENAME_NON_AG.values(), cost_transition_ag2non_ag_df['region'].unique()],
            names = ['Year','From land-use', 'To land-use', 'region'])
        ).reset_index()

    cost_transition_ag2non_ag_trans_mat_region['idx_from'] = cost_transition_ag2non_ag_trans_mat_region['From land-use']\
        .apply(lambda x: AG_LANDUSE.index(x))
    cost_transition_ag2non_ag_trans_mat_region['idx_to'] = cost_transition_ag2non_ag_trans_mat_region['To land-use']\
        .apply(lambda x: list(RENAME_NON_AG.values()).index(x))

    for region,df in cost_transition_ag2non_ag_trans_mat_region.groupby('region'):
        df = df.groupby('Year')[['idx_from', 'idx_to', 'Value ($)']]\
            .apply(lambda x: x[['idx_from','idx_to', 'Value ($)']].values.tolist())\
            .reset_index()
        df.columns = ['Year','data']
        
        region_json = {
            'categories_from': AG_LANDUSE,
            'categories_to': list(RENAME_NON_AG.values()),
            'series': json.loads(df.to_json(orient='records'))
        }
        
        with open(f'{SAVE_DIR}/economics_9_transition_ag2non_cost_4_transition_matrix_{region.replace("/", "_")}.json', 'w') as outfile:
            outfile.write(json.dumps(region_json))
    

        
    # Plot_3-10: Transition cost for Non-Ag to Ag 
    cost_transition_non_ag2ag_df['Value ($)'] = cost_transition_non_ag2ag_df['Value ($)'] * -1  # Convert from negative to positive
    keep_cols = ['Year', 'region', 'Value ($)']
    group_cols = ['Cost type', 'From land-use', 'To land-use', 'Source']
    
    for idx,col in enumerate(group_cols):
        take_cols = keep_cols + [col]
        
        df_AUS = cost_transition_non_ag2ag_df[take_cols]\
            .groupby(['Year', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        df_region = cost_transition_non_ag2ag_df[take_cols]\
            .groupby(['Year', 'region', col])\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: list(map(list,zip(x['Year'],x['Value ($)']))))\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            df.columns = ['name','data','type']
            # save to disk
            df.to_json(f'{SAVE_DIR}/economics_10_transition_non_ag2ag_cost_{idx+1}_{col}_wide_{region.replace("/", "_")}.json', orient='records')
        
    # Transition cost matrix for AUSTRALIA
    cost_transition_non_ag2ag_trans_mat_AUS = cost_transition_non_ag2ag_df\
        .groupby(['Year','From land-use', 'To land-use'])\
        .sum(numeric_only=True)\
        .reset_index()
    cost_transition_non_ag2ag_trans_mat_AUS = cost_transition_non_ag2ag_trans_mat_AUS\
        .set_index(['Year','From land-use', 'To land-use'])\
        .query('abs(`Value ($)`) > 0')
    cost_transition_non_ag2ag_trans_mat_AUS = cost_transition_non_ag2ag_trans_mat_AUS\
        .reindex(
            index = pd.MultiIndex.from_product([years, RENAME_NON_AG.values(), AG_LANDUSE],
            names = ['Year','From land-use', 'To land-use'])
        ).reset_index()

    cost_transition_non_ag2ag_trans_mat_AUS['idx_from'] = cost_transition_non_ag2ag_trans_mat_AUS['From land-use']\
        .apply(lambda x: list(RENAME_NON_AG.values()).index(x))
    cost_transition_non_ag2ag_trans_mat_AUS['idx_to'] = cost_transition_non_ag2ag_trans_mat_AUS['To land-use']\
        .apply(lambda x: AG_LANDUSE.index(x))
                                                    
    cost_transition_non_ag2ag_trans_mat_AUS_data = cost_transition_non_ag2ag_trans_mat_AUS\
        .groupby(['Year'])[['idx_from','idx_to', 'Value ($)']]\
        .apply(lambda x: list(map(list,zip(x['idx_from'],x['idx_to'],x['Value ($)']))))\
        .reset_index()                             
    cost_transition_non_ag2ag_trans_mat_AUS_data.columns = ['Year','data']
    
    cost_transition_non_ag2ag_trans_mat_AUS_json = {
        'categories_from': list(RENAME_NON_AG.values()),
        'categories_to': AG_LANDUSE,
        'series': json.loads(cost_transition_non_ag2ag_trans_mat_AUS_data.to_json(orient='records'))
    }

    with open(f'{SAVE_DIR}/economics_10_transition_non_ag2ag_cost_4_transition_matrix_AUSTRALIA.json', 'w') as outfile:
        outfile.write(json.dumps(cost_transition_non_ag2ag_trans_mat_AUS_json))
        
        
        
    # Transition cost matrix for each region
    cost_transition_non_ag2ag_trans_mat_region = cost_transition_non_ag2ag_df\
        .groupby(['Year','From land-use', 'To land-use', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()
    cost_transition_non_ag2ag_trans_mat_region = cost_transition_non_ag2ag_trans_mat_region\
        .set_index(['Year','From land-use', 'To land-use', 'region'])\
        .query('abs(`Value ($)`) > 0')
    cost_transition_non_ag2ag_trans_mat_region = cost_transition_non_ag2ag_trans_mat_region\
        .reindex(
            index = pd.MultiIndex.from_product([years, RENAME_NON_AG.values(), AG_LANDUSE, cost_transition_non_ag2ag_df['region'].unique()],
            names = ['Year','From land-use', 'To land-use', 'region'])
        ).reset_index()

    cost_transition_non_ag2ag_trans_mat_region['idx_from'] = cost_transition_non_ag2ag_trans_mat_region['From land-use']\
        .apply(lambda x: list(RENAME_NON_AG.values()).index(x))
    cost_transition_non_ag2ag_trans_mat_region['idx_to'] = cost_transition_non_ag2ag_trans_mat_region['To land-use']\
        .apply(lambda x: AG_LANDUSE.index(x))

    for region,df in cost_transition_non_ag2ag_trans_mat_region.groupby('region'):
        df = df.groupby('Year')[['idx_from', 'idx_to', 'Value ($)']]\
            .apply(lambda x: x[['idx_from','idx_to', 'Value ($)']].values.tolist())\
            .reset_index()
        df.columns = ['Year','data']
        
        region_json = {
            'categories_from': list(RENAME_NON_AG.values()),
            'categories_to': AG_LANDUSE,
            'series': json.loads(df.to_json(orient='records'))
        }
        
        with open(f'{SAVE_DIR}/economics_10_transition_non_ag2ag_cost_4_transition_matrix_{region.replace("/", "_")}.json', 'w') as outfile:
            outfile.write(json.dumps(region_json))                                              
                                                





    ####################################################
    #                       4) GHGs                    #
    ####################################################
    if settings.GHG_EMISSIONS_LIMITS != 'off':
        filter_str = '''
        category == "GHG" 
        and base_name.str.contains("GHG_emissions") 
        and year_types != "begin_end_year"
        '''.replace('\n', ' ').replace('  ', ' ')

        GHG_files = files.query(filter_str).reset_index(drop=True)

        GHG_ag = GHG_files.query('base_name.str.contains("agricultural_landuse")').reset_index(drop=True)
        GHG_ag = pd.concat([pd.read_csv(path) for path in GHG_ag['path']], ignore_index=True)
        GHG_ag = GHG_ag.replace(GHG_NAMES)
        
        GHG_non_ag = GHG_files.query('base_name.str.contains("no_ag_reduction")').reset_index(drop=True)
        GHG_non_ag = pd.concat([pd.read_csv(path) for path in GHG_non_ag['path'] if not pd.read_csv(path).empty], ignore_index=True)
        GHG_non_ag = GHG_non_ag.replace(RENAME_AM_NON_AG)
        
        GHG_ag_man = GHG_files.query('base_name.str.contains("agricultural_management")').reset_index(drop=True)
        GHG_ag_man = pd.concat([pd.read_csv(path) for path in GHG_ag_man['path'] if not pd.read_csv(path).empty], ignore_index=True)
        GHG_ag_man = GHG_ag_man.replace(RENAME_AM_NON_AG)
        
        GHG_transition = GHG_files.query('base_name.str.contains("transition_penalty")').reset_index(drop=True)
        GHG_transition = pd.concat([pd.read_csv(path) for path in GHG_transition['path'] if not pd.read_csv(path).empty], ignore_index=True)
        GHG_transition = GHG_transition.replace(RENAME_AM_NON_AG)
        for yr in years:
            if not yr in GHG_transition['Year'].unique():
                GHG_transition = pd.concat([
                    GHG_transition, 
                    pd.DataFrame([{
                        'Year': yr, 
                        'Value (t CO2e)': 0,
                        'Type': 'Unallocated natural to modified',
                        'region': 'Avon',
                    }]
                )], ignore_index=True)

        GHG_off_land = GHG_files.query('base_name.str.contains("offland_commodity")')
        GHG_off_land = pd.concat([pd.read_csv(path) for path in GHG_off_land['path']], ignore_index=True)
        GHG_off_land['Value (t CO2e)'] = GHG_off_land['Total GHG Emissions (tCO2e)']
        GHG_off_land['Commodity'] = GHG_off_land['COMMODITY'].apply(lambda x: x[0].capitalize() + x[1:])
        GHG_off_land = GHG_off_land.drop(columns=['COMMODITY', 'Total GHG Emissions (tCO2e)'])
        GHG_off_land['Emission Source'] = GHG_off_land['Emission Source']\
            .replace({
                'CO2': 'Carbon Dioxide (CO2)',
                'CH4': 'Methane (CH4)',
                'N2O': 'Nitrous Oxide (N2O)'
            })
            
            
        def get_landuse_type(x):
            if x in LU_CROPS:
                return 'Crop'
            elif x in LVSTK_NATURAL:
                return 'Livestock - natural land'
            elif x in LVSTK_MODIFIED:
                return 'Livestock - modified land'
            else:
                return 'Unallocated land'
            
        GHG_land = pd.concat([GHG_ag, GHG_non_ag, GHG_ag_man, GHG_transition], axis=0).query('abs(`Value (t CO2e)`) > 0').reset_index(drop=True)
        GHG_land['Land-use type'] = GHG_land['Land-use'].apply(get_landuse_type)
        net_land = GHG_land.groupby('Year')[['Value (t CO2e)']].sum(numeric_only=True).reset_index()


        GHG_limit = GHG_files.query('base_name == "GHG_emissions"')
        GHG_limit = pd.concat([pd.read_csv(path) for path in GHG_limit['path']], ignore_index=True)
        GHG_limit = GHG_limit.query('Variable == "GHG_EMISSIONS_LIMIT_TCO2e"')
        GHG_limit['Value (t CO2e)'] = GHG_limit['Emissions (t CO2e)']
        GHG_limit_wide = list(map(list,zip(GHG_limit['Year'],GHG_limit['Value (t CO2e)'])))
        
        

        # Plot_4-2: GHG from individual emission sectors (Mt)
        net_offland_AUS = GHG_off_land.groupby('Year')[['Value (t CO2e)']].sum(numeric_only=True).reset_index()
        net_offland_AUS_wide = net_offland_AUS[['Year','Value (t CO2e)']].values.tolist()
        
        net_land_AUS_wide = GHG_land\
            .groupby(['Year','Type'])[['Value (t CO2e)']]\
            .sum(numeric_only=True)\
            .reset_index()
        net_land_AUS_wide = net_land_AUS_wide\
            .groupby(['Type'])[['Year','Value (t CO2e)']]\
            .apply(lambda x:x[['Year', 'Value (t CO2e)']].values.tolist())\
            .reset_index()
        net_land_AUS_wide.columns = ['name','data']
        net_land_AUS_wide['type'] = 'column'
            
        net_land_AUS_wide.loc[len(net_land_AUS_wide)] = [
            'Off-land emissions', 
            net_offland_AUS_wide, 
            'column'
        ]
        net_land_AUS_wide.loc[len(net_land_AUS_wide)] = [
            'Net emissions',
            list(zip(years, (net_land['Value (t CO2e)'] + net_offland_AUS['Value (t CO2e)']))),
            'spline'
        ]
        net_land_AUS_wide.loc[len(net_land_AUS_wide)] = [
            'GHG emission limit',
            GHG_limit_wide,
            'spline'
        ]

        
        order_GHG = [
            'Agricultural land-use',
            'Agricultural Management',
            'Non-Agricultural land-use',
            'Off-land emissions',
            'Unallocated natural to modified',
            'Unallocated natural to livestock natural',
            'Livestock natural to modified',
            'Net emissions',
            'GHG emission limit'
        ]
        net_land_AUS_wide['name_order'] = net_land_AUS_wide['name'].apply(lambda x: order_GHG.index(x))
        net_land_AUS_wide = net_land_AUS_wide.sort_values('name_order').drop(columns=['name_order'])
        net_land_AUS_wide.to_json(f'{SAVE_DIR}/GHG_2_individual_emission_Mt.json', orient='records')
        
        

        for region,df in GHG_land.groupby('region'):
            df_reg = df\
                .groupby(['Year','Type'])[['Value (t CO2e)']]\
                .sum(numeric_only=True)\
                .reset_index()
            df_reg = df_reg\
                .groupby(['Type'])[['Year','Value (t CO2e)']]\
                .apply(lambda x:x[['Year', 'Value (t CO2e)']].values.tolist())\
                .reset_index()
            df_reg.columns = ['name','data']
            df_reg['type'] = 'column'
            
            df_reg.loc[len(df_reg)] = [
                'Net emissions', 
                list(zip(years, (df.groupby('Year')['Value (t CO2e)'].sum().values))),
                'spline'
            ]

            df_reg['name_order'] = df_reg['name'].apply(lambda x: order_GHG.index(x))
            df_reg = df_reg.sort_values('name_order').drop(columns=['name_order'])
            df_reg.to_json(f'{SAVE_DIR}/GHG_2_individual_emission_Mt_{region.replace("/", "_")}.json', orient='records')




        # Plot_4-3: GHG emission for agricultural land-use  
        GHG_ag = GHG_land.query('Type == "Agricultural land-use"') 
        GHG_CO2 = GHG_ag.query('~Source.isin(@GHG_CATEGORY.keys())').copy()
        GHG_CO2['GHG Category'] = 'CO2'

        GHG_nonCO2 = GHG_ag.query('Source.isin(@GHG_CATEGORY.keys())').copy()
        GHG_nonCO2['GHG Category'] = GHG_nonCO2['Source'].apply(lambda x: GHG_CATEGORY[x].keys())
        GHG_nonCO2['Multiplier'] = GHG_nonCO2['Source'].apply(lambda x: GHG_CATEGORY[x].values())
        GHG_nonCO2 = GHG_nonCO2.explode(['GHG Category','Multiplier']).reset_index(drop=True)
        GHG_nonCO2['Value (t CO2e)'] = GHG_nonCO2['Value (t CO2e)'] * GHG_nonCO2['Multiplier']
        GHG_nonCO2 = GHG_nonCO2.drop(columns=['Multiplier'])

        GHG_ag_emissions_long = pd.concat([GHG_CO2, GHG_nonCO2], axis=0).reset_index(drop=True)
        GHG_ag_emissions_long['GHG Category'] = GHG_ag_emissions_long['GHG Category']\
            .replace({
                'CH4': 'Methane (CH4)', 
                'N2O': 'Nitrous Oxide (N2O)', 
                'CO2': 'Carbon Dioxide (CO2)'
            })
            
            
        keep_cols = ['Year', 'region', 'Value (t CO2e)', 'Type', 'Agricultural Management Type']
        group_cols = ['GHG Category', 'Land-use', 'Land-use type', 'Source', 'Water_supply']
        for idx, col in enumerate(group_cols):
            
            take_cols = keep_cols + [col]
            
            df_AUS = GHG_ag_emissions_long[take_cols]\
                .groupby(['Year', col])[['Value (t CO2e)']]\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (t CO2e)']]\
                .apply(lambda x: list(map(list,zip(x['Year'], x['Value (t CO2e)']))))\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name','data','region']
            df_AUS_wide['type'] = 'column'
            
            
            df_region = GHG_ag_emissions_long[take_cols]\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (t CO2e)']]\
                .apply(lambda x: list(map(list,zip(x['Year'],x['Value (t CO2e)']))))\
                .reset_index()
            df_region_wide.columns = ['name', 'region','data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            if col == "Land-use":
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

            for region, df in df_wide.groupby('region'):
                df = df.drop('region', axis=1)
                df.columns = ['name','data','type']
                # save to disk
                df.to_json(f'{SAVE_DIR}/GHG_3_{idx+1}_{col.replace(" ", "_")}_{region.replace("/", "_")}.json', orient='records')

     
            
        # Plot_4-4: GHG emission (off-land) by commodity
        keep_cols = ['Year', 'Value (t CO2e)']
        group_cols = ['Emission Type', 'Emission Source', 'Commodity']
        for idx, col in enumerate(group_cols):
            
            take_cols = keep_cols + [col]
            
            df_AUS = GHG_off_land[take_cols]\
                .groupby(['Year', col])[['Value (t CO2e)']]\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (t CO2e)']]\
                .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
                .reset_index()
                
            df_AUS_wide.columns = ['name','data']
            df_AUS_wide['type'] = 'column'
            df_AUS_wide.to_json(f'{SAVE_DIR}/GHG_4_{idx+1}_{col.replace(" ", "_")}.json', orient='records')
    


        # Plot_4-5: GHG abatement by Non-Agrilcultural sector
        Non_ag_reduction_long = GHG_land.query('Type == "Non-Agricultural land-use"').reset_index(drop=True)
        
        keep_cols = ['Year', 'region', 'Value (t CO2e)', 'Type', 'Source','Agricultural Management Type', 'Land-use type', 'Water_supply']
        group_cols = ['Land-use']
        for idx, col in enumerate(group_cols):
            
            take_cols = keep_cols + [col]
            
            df_AUS = Non_ag_reduction_long[take_cols]\
                .groupby(['Year', col])[['Value (t CO2e)']]\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (t CO2e)']]\
                .apply(lambda x: list(map(list,zip(x['Year'], x['Value (t CO2e)']))))\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name','data','region']
            df_AUS_wide['type'] = 'column'
            
            
            df_region = Non_ag_reduction_long[take_cols]\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (t CO2e)']]\
                .apply(lambda x: list(map(list,zip(x['Year'],x['Value (t CO2e)']))))\
                .reset_index()
            df_region_wide.columns = ['name', 'region','data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            if col == "Land-use":
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

            for region, df in df_wide.groupby('region'):
                df = df.drop('region', axis=1)
                df.columns = ['name','data','type']
                df.to_json(f'{SAVE_DIR}/GHG_5_{idx+1}_{col.replace(" ", "_")}_{region.replace("/", "_")}.json', orient='records')

     

        # Plot_4-6: GHG reductions by Agricultural managements (Mt)
        Ag_man_sequestration_long = GHG_land.query('Type == "Agricultural Management"').reset_index(drop=True)
        
        keep_cols = ['Year', 'region', 'Value (t CO2e)', 'Type', 'Source']
        group_cols = ['Land-use', 'Land-use type', 'Agricultural Management Type', 'Water_supply']
        for idx, col in enumerate(group_cols):
            
            take_cols = keep_cols + [col]
            
            df_AUS = Ag_man_sequestration_long[take_cols]\
                .groupby(['Year', col])[['Value (t CO2e)']]\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (t CO2e)']]\
                .apply(lambda x: list(map(list,zip(x['Year'], x['Value (t CO2e)']))))\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name','data','region']
            df_AUS_wide['type'] = 'column'
            
            
            df_region = Ag_man_sequestration_long[take_cols]\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (t CO2e)']]\
                .apply(lambda x: list(map(list,zip(x['Year'],x['Value (t CO2e)']))))\
                .reset_index()
            df_region_wide.columns = ['name', 'region','data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            if col == "Land-use":
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

            for region, df in df_wide.groupby('region'):
                df = df.drop('region', axis=1)
                df.columns = ['name','data','type']
                df.to_json(f'{SAVE_DIR}/GHG_6_{idx+1}_{col.replace(" ", "_")}_{region.replace("/", "_")}.json', orient='records')

        


    ####################################################
    #                     5) Water                     #
    ####################################################
    
    water_files = files.query('category == "water" and year_types == "single_year"').reset_index(drop=True)

    water_net_yield_water_region = water_files.query('base_name == "water_yield_separate"')
    water_net_yield_water_region = pd.concat([pd.read_csv(path) for path in water_net_yield_water_region['path']], ignore_index=True)
    water_net_yield_water_region = water_net_yield_water_region\
        .replace(RENAME_AM_NON_AG)\
        .query('`Water Net Yield (ML)` > 0')\
        .rename(columns={'Water Net Yield (ML)': 'Value (ML)'})


    water_net_yield_NRM_region = water_files.query('base_name == "water_yield_separate_NRM"')
    water_net_yield_NRM_region = pd.concat([pd.read_csv(path) for path in water_net_yield_NRM_region['path']], ignore_index=True)
    water_net_yield_NRM_region = water_net_yield_NRM_region\
        .replace(RENAME_AM_NON_AG)\
        .query('`Water Net Yield (ML)` > 0')\
        .rename(columns={'Water Net Yield (ML)': 'Value (ML)'})


    hist_and_public_wny_water_region = water_files.query('base_name == "water_yield_limits_and_public_land"')
    hist_and_public_wny_water_region = pd.concat([pd.read_csv(path) for path in hist_and_public_wny_water_region['path']], ignore_index=True)
 
    water_outside_LUTO = hist_and_public_wny_water_region[['Year','Region', 'Water yield outside LUTO (ML)']].rename(
        columns={'Water yield outside LUTO (ML)': 'Value (ML)'}
    )
    water_climate_change_impact = hist_and_public_wny_water_region[['Year','Region', 'Climate Change Impact (ML)']].rename(
        columns={'Climate Change Impact (ML)': 'Value (ML)'}
    )
    water_domestic_use = hist_and_public_wny_water_region[['Year','Region', 'Domestic Water Use (ML)']].rename(
        columns={'Domestic Water Use (ML)': 'Value (ML)'}
    ).eval('`Value (ML)` = -`Value (ML)`')  # Domestic water use is negative, indicating a water loss (consumption)
    water_yield_limit = hist_and_public_wny_water_region[['Year','Region', 'Water Yield Limit (ML)']].rename(
        columns={'Water Yield Limit (ML)': 'Value (ML)'}
    )
    water_net_yield = hist_and_public_wny_water_region[['Year','Region', 'Water Net Yield (ML)']].rename(
        columns={'Water Net Yield (ML)': 'Value (ML)'}
    )



    # Plot_5-1: Water total yield by broad categories (ML)
    water_inside_LUTO_sum = water_net_yield_water_region\
        .groupby(['Year','Type'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()
    water_inside_LUTO_sum_wide = water_inside_LUTO_sum\
        .groupby('Type')[['Year','Value (ML)']]\
        .apply(lambda x: list(map(list,zip(x['Year'],x['Value (ML)']))))\
        .reset_index()
    water_inside_LUTO_sum_wide.columns = ['name','data']
    water_inside_LUTO_sum_wide['type'] = 'column'
    
        
    water_outside_LUTO_total = water_outside_LUTO\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .reset_index()[['Year','Value (ML)']].values.tolist()
    water_CCI = water_climate_change_impact\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .reset_index()[['Year','Value (ML)']].values.tolist()
    water_domestic = water_domestic_use\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .reset_index()[['Year','Value (ML)']].values.tolist()
    water_net_yield_sum = water_net_yield\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .reset_index()[['Year','Value (ML)']].values.tolist()
    water_limit = water_yield_limit\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .reset_index()[['Year','Value (ML)']].values.tolist()
  
    water_yield_df_AUS = water_inside_LUTO_sum_wide.copy()
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Outside LUTO Study Area', water_outside_LUTO_total,  'column']
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Climate Change Impact', water_CCI,  'column']
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Domestic Water Use', water_domestic,  'column']
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Water Net Yield', water_net_yield_sum, 'spline']
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Water Limit', water_limit, 'spline']

    water_yield_df_AUS.to_json(f'{SAVE_DIR}/water_1_water_net_use_by_broader_category_AUSTRALIA.json', orient='records')
    

    # Plot_5-2: Water net yield by specific land-use (ML)
    water_inside_LUTO_lu_sum = water_net_yield_water_region\
        .groupby(['Year','Landuse'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()
            
    water_inside_LUTO_lu_sum_wide = water_inside_LUTO_lu_sum\
        .groupby(['Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()\
        .set_index('Landuse')\
        .reindex(LANDUSE_ALL_RENAMED)\
        .reset_index()
        
    water_inside_LUTO_lu_sum_wide.columns = ['name','data']
    water_inside_LUTO_lu_sum_wide['type'] = 'column'
    water_inside_LUTO_lu_sum_wide.to_json(f'{SAVE_DIR}/water_2_water_net_yield_by_specific_landuse.json', orient='records')
    
    
    # Plot_5-3: Water net yield by region (ML)
    water_yield_region = {}
    for reg_name in water_net_yield['Region'].unique():

        water_inside_LUTO_region = water_net_yield.query('Region == @reg_name').copy()
        water_inside_yield_wide = water_inside_LUTO_region[['Year','Value (ML)']].values.tolist()

        water_outside_LUTO_region = water_outside_LUTO.query('Region == @reg_name').copy()
        water_outside_yield_wide = water_outside_LUTO_region[['Year','Value (ML)']].values.tolist()

        water_CCI = water_climate_change_impact.query('Region == @reg_name').copy()
        water_CCI_wide = water_CCI[['Year','Value (ML)']].values.tolist()
        
        water_domestic = water_domestic_use.query('Region == @reg_name').copy()
        water_domestic_wide = water_domestic[['Year','Value (ML)']].values.tolist()

        water_net_yield_sum = water_net_yield.query('Region == @reg_name').copy()
        water_net_yield_sum_wide = water_net_yield_sum[['Year','Value (ML)']].values.tolist()

        water_limit = water_yield_limit.query('Region == @reg_name').copy()
        water_limit_wide = water_limit[['Year','Value (ML)']].values.tolist()


        water_df = pd.DataFrame([
            ['Water Yield Inside LUTO Study Area', water_inside_yield_wide, 'column', None],
            ['Water Yield Outside LUTO Study Area', water_outside_yield_wide, 'column', None],
            ['Climate Change Impact', water_CCI_wide, 'column', None],
            ['Domestic Water Use', water_domestic_wide, 'column', None],
            ['Water Net Yield', water_net_yield_sum_wide, 'spline', None],
            ['Water Limit', water_limit_wide, 'spline', 'black']],  columns=['name','data','type','color']
        )

        water_yield_region[reg_name] = water_df.to_dict(orient='records')

    with open(f'{SAVE_DIR}/water_3_water_net_yield_by_region.json', 'w') as outfile:
        json.dump(water_yield_region, outfile)

        



    #########################################################
    #                   6) Biodiversity                     #
    #########################################################
    
    
    # ---------------- Biodiversity priority total score  ----------------
    filter_str = '''
        category == "biodiversity"
        and year_types == "single_year"
        and base_name == "biodiversity_overall_priority_scores"
    '''.strip().replace('\n','')
    
    bio_paths = files.query(filter_str).reset_index(drop=True)
    bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
    bio_df = bio_df.replace(RENAME_AM_NON_AG)
    
    # Plot_BIO_priority_1: Biodiversity total score by Type
    bio_df_type = bio_df.groupby(['Year','Type']).sum(numeric_only=True).reset_index()
    bio_df_type = bio_df_type\
        .groupby('Type')[['Year','Contribution Relative to Base Year Level (%)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Base Year Level (%)']))))\
        .reset_index()
        
    bio_df_type.columns = ['name','data']
    bio_df_type['type'] = 'column'
    bio_df_type.to_json(f'{SAVE_DIR}/biodiversity_priority_1_total_score_by_type.json', orient='records')
    
    
    # Plot_BIO_priority_2: Biodiversity total score by landuse
    bio_df_landuse = bio_df.groupby(['Year','Landuse']).sum(numeric_only=True).reset_index()
    bio_df_landuse = bio_df_landuse\
        .groupby('Landuse')[['Year','Contribution Relative to Base Year Level (%)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Base Year Level (%)']))))\
        .reset_index()
        
    bio_df_landuse.columns = ['name','data']
    bio_df_landuse['type'] = 'column'
    bio_df_landuse = bio_df_landuse.set_index('name').reindex(LANDUSE_ALL_RENAMED).reset_index()
    bio_df_landuse.to_json(f'{SAVE_DIR}/biodiversity_priority_2_total_score_by_landuse.json', orient='records')
    
    
    # Plot_BIO_priority_3: Biodiversity total score by Agricultural Management
    bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()
    bio_df_am = bio_df_am.groupby(['Year','Agri-Management']).sum(numeric_only=True).reset_index()
    
    bio_df_am = bio_df_am\
        .groupby('Agri-Management')[['Year','Contribution Relative to Base Year Level (%)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Base Year Level (%)']))))\
        .reset_index()
        
    bio_df_am.columns = ['name','data']
    bio_df_am['type'] = 'column'
    bio_df_am.to_json(f'{SAVE_DIR}/biodiversity_priority_3_total_score_by_agri_management.json', orient='records')
    
    
    # Plot_BIO_priority_4: Biodiversity total score by Non-Agricultural Land-use
    bio_df_non_ag = bio_df.query('Type == "Non-Agricultural land-use"').copy()
    bio_df_non_ag = bio_df_non_ag.groupby(['Year','Landuse']).sum(numeric_only=True).reset_index()
    
    bio_df_non_ag = bio_df_non_ag\
        .groupby('Landuse')[['Year','Contribution Relative to Base Year Level (%)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Base Year Level (%)']))))\
        .reset_index()
        
    bio_df_non_ag.columns = ['name','data']
    bio_df_non_ag['type'] = 'column'
    bio_df_non_ag.to_json(f'{SAVE_DIR}/biodiversity_priority_4_total_score_by_non_agri_landuse.json', orient='records')
    
    
        
    # ---------------- (GBF2) Biodiversity priority score  ----------------
    if settings.BIODIVERSITY_TARGET_GBF_2 == 'off':
        pass
    else:
        # get biodiversity dataframe
        filter_str = '''
            category == "biodiversity" 
            and year_types == "single_year" 
            and base_name == "biodiversity_GBF2_priority_scores"
        '''.strip('').replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)
        

        # Plot_GBF2_1: Biodiversity total by Type
        bio_df_target = bio_df.groupby(['Year'])[['Priority Target (%)']].agg('first').reset_index()
        bio_df_target_json = list(map(list,zip(bio_df_target['Year'],bio_df_target['Priority Target (%)'])))
        
        bio_df_type = bio_df.groupby(['Year','Type']).sum(numeric_only=True).reset_index()
        bio_df_type = bio_df_type\
            .groupby('Type')[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()

        bio_df_type.columns = ['name','data']
        bio_df_type['type'] = 'column'
        bio_df_type.loc[len(bio_df_type)] = ['Priority Target (%)', bio_df_target_json, 'spline']
        bio_df_type.to_json(f'{SAVE_DIR}/biodiversity_GBF2_1_total_score_by_type.json', orient='records')
        
        
        
        # Plot_GBF2_2: Biodiversity total by landuse
        bio_df_landuse = bio_df\
            .query('Type == "Agricultural land-use"')\
            .groupby(['Year','Landuse'])\
            .sum(numeric_only=True).reset_index()
        bio_df_landuse = bio_df_landuse\
            .groupby('Landuse')[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_landuse.columns = ['name','data']
        bio_df_landuse['type'] = 'column'
        bio_df_landuse = bio_df_landuse.set_index('name').reindex(LANDUSE_ALL_RENAMED).reset_index()
        
        bio_df_landuse.to_json(f'{SAVE_DIR}/biodiversity_GBF2_2_total_score_by_landuse.json', orient='records')
        
        
 
        # Plot_GBF2_3: Biodiversity total by Agricultural Management
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()
        bio_df_am = bio_df_am.groupby(['Year','Agri-Management']).sum(numeric_only=True).reset_index()
        
        bio_df_am = bio_df_am\
            .groupby('Agri-Management')[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_am.columns = ['name','data']
        bio_df_am['type'] = 'column'
        bio_df_am.to_json(f'{SAVE_DIR}/biodiversity_GBF2_3_total_score_by_agri_management.json', orient='records')
        
        
        
        
        # Plot_GBF2_4: Biodiversity total by Non-Agricultural Land-use
        bio_df_non_ag = bio_df.query('Type == "Non-Agricultural land-use"').copy()
        bio_df_non_ag = bio_df_non_ag.groupby(['Year','Landuse']).sum(numeric_only=True).reset_index()
        
        bio_df_non_ag = bio_df_non_ag\
            .groupby('Landuse')[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_non_ag.columns = ['name','data']
        bio_df_non_ag['type'] = 'column'
        bio_df_non_ag.to_json(f'{SAVE_DIR}/biodiversity_GBF2_4_total_score_by_non_agri_landuse.json', orient='records')
        
            
            
    # ---------------- (GBF3) Biodiversity Major Vegetation Group score  ----------------
    if settings.BIODIVERSITY_TARGET_GBF_3 == 'off':
        pass
    else:
        filter_str = '''
            category == "biodiversity" 
            and year_types == "single_year" 
            and base_name.str.contains("biodiversity_GBF3")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)
        
        
        # Plot_GBF3_1: Biodiversity contribution score (group) total
        bio_df_group = bio_df.groupby(['Vegetation Group','Year']).sum(numeric_only=True).reset_index()
        bio_df_group = bio_df_group\
            .groupby(['Vegetation Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_group.columns = ['name','data']
        bio_df_group['type'] = 'spline'
        bio_df_group.to_json(f'{SAVE_DIR}/biodiversity_GBF3_1_contribution_group_score_total.json', orient='records')
        
        
        # Plot_GBF3_2: Biodiversity contribution score (group) by Type
        bio_df_group_type_sum = bio_df\
            .groupby(['Year','Type','Vegetation Group'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_df_group_type_sum = bio_df_group_type_sum\
            .groupby(['Type','Vegetation Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_group_type_records = []
        for idx,df in bio_df_group_type_sum.groupby('Vegetation Group'):
            df = df.drop('Vegetation Group',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            bio_df_group_type_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF3_2_contribution_group_score_by_type.json', 'w') as outfile:
            json.dump(bio_df_group_type_records, outfile)
            
            
        # Plot_GBF3_3: Biodiversity contribution score (group) by landuse
        bio_group_lu_sum = bio_df\
            .query('Type == "Agricultural land-use"')\
            .groupby(['Year','Landuse','Vegetation Group'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('`Contribution Relative to Pre-1750 Level (%)` >1')
            
        bio_group_lu_sum = bio_group_lu_sum\
            .groupby(['Landuse','Vegetation Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_group_records = []
        for idx,df in bio_group_lu_sum.groupby('Vegetation Group'):
            df = df.drop('Vegetation Group',axis=1)
            df.columns = ['name','data']
            df = df.set_index('name').reindex(LANDUSE_ALL_RENAMED).reset_index().dropna()
            df['type'] = 'column'
            bio_df_group_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF3_3_contribution_group_score_by_landuse.json', 'w') as outfile:
            json.dump(bio_df_group_records, outfile)
            
            
        # Plot_GBF3_4: Biodiversity contribution score (group) by agricultural management
        bio_group_am_sum = bio_df\
            .query('Type == "Agricultural Management"')\
            .groupby(['Year','Agri-Management','Vegetation Group'])\
            .sum(numeric_only=True)\
            .reset_index()
                
        bio_group_am_sum = bio_group_am_sum\
            .groupby(['Agri-Management','Vegetation Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_group_records = []
        for idx,df in bio_group_am_sum.groupby('Vegetation Group'):
            df = df.drop('Vegetation Group',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            bio_df_group_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF3_4_contribution_group_score_by_agri_management.json', 'w') as outfile:
            json.dump(bio_df_group_records, outfile)
            
            
        # Plot_GBF3_5: Biodiversity contribution score (group) by non-Agricultural land-use
        bio_group_non_ag_sum = bio_df\
            .query('Type == "Non-Agricultural land-use"')\
            .groupby(['Year','Landuse','Vegetation Group'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_group_non_ag_sum = bio_group_non_ag_sum\
            .groupby(['Landuse','Vegetation Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_group_records = []
        for idx,df in bio_group_non_ag_sum.groupby('Vegetation Group'):
            df = df.drop('Vegetation Group',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            bio_df_group_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF3_5_contribution_group_score_by_non_agri_landuse.json', 'w') as outfile:
            json.dump(bio_df_group_records, outfile)
            
            
            
            
    # ---------------- (GBF4) Biodiversity National Significance score  ----------------
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES == 'on':
        
        # Get biodiversity dataframe
        filter_str = '''
            category == "biodiversity" 
            and year_types == "single_year" 
            and base_name.str.contains("biodiversity_GBF4_SNES_scores")
        '''.strip().replace('\n', '')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)  # Rename the landuse
        
        
        # Plot_GBF4_SNES_1: Biodiversity contribution score for each species
        bio_df_species = bio_df.groupby(['Year','species']).sum(numeric_only=True).reset_index()
        bio_df_species = bio_df_species\
            .groupby(['species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_species.columns = ['name','data']
        bio_df_species['type'] = 'spline'
        bio_df_species.to_json(f'{SAVE_DIR}/biodiversity_GBF4_SNES_1_contribution_species_score_total.json', orient='records')
        
        
        # Plot_GBF4_SNES_2: Biodiversity contribution score (species) by Type
        bio_df_species_type_sum = bio_df\
            .groupby(['Year','Type','species'])\
            .sum(numeric_only=True)\
            .reset_index()

        bio_df_species_type_sum = bio_df_species_type_sum\
            .groupby(['Type','species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x: list(map(list, zip(x['Year'], x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()

        bio_df_species_type_records = []
        for idx, df in bio_df_species_type_sum.groupby('species'):
            df = df.drop('species', axis=1)
            df.columns = ['name', 'data']
            df['type'] = 'column'
            bio_df_species_type_records.append({'name': idx, 'data': df.to_dict(orient='records')})

        with open(f'{SAVE_DIR}/biodiversity_GBF4_SNES_2_contribution_species_score_by_type.json', 'w') as outfile:
            json.dump(bio_df_species_type_records, outfile)
        
        
        # Plot_GBF4_SNES_3: Biodiversity contribution score (species) by landuse
        bio_species_lu_sum = bio_df\
            .query('Type == "Agricultural land-use"')\
            .groupby(['Year','Landuse','species'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_species_lu_sum = bio_species_lu_sum\
            .groupby(['Landuse','species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_species_records = []
        for idx, df in bio_species_lu_sum.groupby('species'):
            df = df.drop('species', axis=1)
            df.columns = ['name', 'data']
            df['type'] = 'column'
            df = df.set_index('name').reindex(LANDUSE_ALL_RENAMED).reset_index().dropna()
            bio_df_species_records.append({'name': idx, 'data': df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF4_SNES_3_contribution_species_score_by_landuse.json', 'w') as outfile:
            json.dump(bio_df_species_records, outfile)
        
        
        # Plot_GBF4_SNES_4: Biodiversity contribution score (species) by agricultural management
        bio_species_am_sum = bio_df\
            .groupby(['Year','Agri-Management','species'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_species_am_sum = bio_species_am_sum\
            .groupby(['Agri-Management','species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_species_records = []
        for idx, df in bio_species_am_sum.groupby('species'):
            df = df.drop('species', axis=1)
            df.columns = ['name', 'data']
            df['type'] = 'column'
            bio_df_species_records.append({'name': idx, 'data': df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF4_SNES_4_contribution_species_score_by_agri_management.json', 'w') as outfile:
            json.dump(bio_df_species_records, outfile)
        
        
        # Plot_GBF4_SNES_5: Biodiversity contribution score (species) by non-Agricultural land-use
        bio_species_non_ag_sum = bio_df\
            .query('Type == "Non-Agricultural land-use"')\
            .groupby(['Year','Landuse','species'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_species_non_ag_sum = bio_species_non_ag_sum\
            .groupby(['Landuse','species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_species_records = []
        for idx, df in bio_species_non_ag_sum.groupby('species'):
            df = df.drop('species', axis=1)
            df.columns = ['name', 'data']
            df['type'] = 'column'
            bio_df_species_records.append({'name': idx, 'data': df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF4_SNES_5_contribution_species_score_by_non_agri_landuse.json', 'w') as outfile:
            json.dump(bio_df_species_records, outfile)
            
            
            
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES == 'on':
        # Get biodiversity dataframe
        filter_str = '''
            category == "biodiversity" 
            and year_types == "single_year" 
            and base_name.str.contains("biodiversity_GBF4_ECNES_scores")
        '''.strip().replace('\n', '')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)
        
        
        # Plot_GBF4_ECNES_1: Biodiversity contribution score for each species
        bio_df_species = bio_df.groupby(['Year','species']).sum(numeric_only=True).reset_index()
        bio_df_species = bio_df_species\
            .groupby(['species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x: list(map(list, zip(x['Year'], x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
        bio_df_species.columns = ['name', 'data']
        bio_df_species['type'] = 'spline'
        bio_df_species.to_json(f'{SAVE_DIR}/biodiversity_GBF4_ECNES_1_contribution_species_score_total.json', orient='records')
        
        
        # Plot_GBF4_ECNES_2: Biodiversity contribution score (species) by Type
        bio_df_species_type_sum = bio_df\
            .groupby(['Year','Type','species'])\
            .sum(numeric_only=True)\
            .reset_index()

        bio_df_species_type_sum = bio_df_species_type_sum\
            .groupby(['Type','species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x: list(map(list, zip(x['Year'], x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()

        bio_df_species_type_records = []
        for idx, df in bio_df_species_type_sum.groupby('species'):
            df = df.drop('species', axis=1)
            df.columns = ['name', 'data']
            df['type'] = 'column'
            bio_df_species_type_records.append({'name': idx, 'data': df.to_dict(orient='records')})

        with open(f'{SAVE_DIR}/biodiversity_GBF4_ECNES_2_contribution_species_score_by_type.json', 'w') as outfile:
            json.dump(bio_df_species_type_records, outfile)

        # Plot_GBF4_ECNES_3: Biodiversity contribution score (species) by agri landuse
        bio_species_lu_sum = bio_df\
            .query('Type == "Agricultural land-use"')\
            .groupby(['Year','Landuse','species'])\
            .sum(numeric_only=True)\
            .reset_index()

        bio_species_lu_sum = bio_species_lu_sum\
            .groupby(['Landuse','species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x: list(map(list, zip(x['Year'], x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()

        bio_df_species_records = []
        for idx, df in bio_species_lu_sum.groupby('species'):
            df = df.drop('species', axis=1)
            df.columns = ['name', 'data']
            df['type'] = 'column'
            df = df.set_index('name').reindex(LANDUSE_ALL_RENAMED).reset_index().dropna()
            bio_df_species_records.append({'name': idx, 'data': df.to_dict(orient='records')})

        with open(f'{SAVE_DIR}/biodiversity_GBF4_ECNES_3_contribution_species_score_by_landuse.json', 'w') as outfile:
            json.dump(bio_df_species_records, outfile)

        # Plot_GBF4_ECNES_4: Biodiversity contribution score (species) by agricultural management
        bio_species_am_sum = bio_df\
            .groupby(['Year','Agri-Management','species'])\
            .sum(numeric_only=True)\
            .reset_index()

        bio_species_am_sum = bio_species_am_sum\
            .groupby(['Agri-Management','species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x: list(map(list, zip(x['Year'], x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()

        bio_df_species_records = []
        for idx, df in bio_species_am_sum.groupby('species'):
            df = df.drop('species', axis=1)
            df.columns = ['name', 'data']
            df['type'] = 'column'
            bio_df_species_records.append({'name': idx, 'data': df.to_dict(orient='records')})

        with open(f'{SAVE_DIR}/biodiversity_GBF4_ECNES_4_contribution_species_score_by_agri_management.json', 'w') as outfile:
            json.dump(bio_df_species_records, outfile)

        # Plot_GBF4_ECNES_5: Biodiversity contribution score (species) by non-Agricultural land-use
        bio_species_non_ag_sum = bio_df\
            .query('Type == "Non-Agricultural land-use"')\
            .groupby(['Year','Landuse','species'])\
            .sum(numeric_only=True)\
            .reset_index()

        bio_species_non_ag_sum = bio_species_non_ag_sum\
            .groupby(['Landuse','species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x: list(map(list, zip(x['Year'], x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()

        bio_df_species_records = []
        for idx, df in bio_species_non_ag_sum.groupby('species'):
            df = df.drop('species', axis=1)
            df.columns = ['name', 'data']
            df['type'] = 'column'
            bio_df_species_records.append({'name': idx, 'data': df.to_dict(orient='records')})

        with open(f'{SAVE_DIR}/biodiversity_GBF4_ECNES_5_contribution_species_score_by_non_agri_landuse.json', 'w') as outfile:
            json.dump(bio_df_species_records, outfile)
        
        
        
    # ---------------- (GBF8) Biodiversity suitability under differen climate change  ----------------
    
    # 1) Biodiversity suitability scores (GBF8) by group
    if settings.BIODIVERSITY_TARGET_GBF_8 == 'on':
        
        # Get biodiversity dataframe
        filter_str = '''
            category == "biodiversity" 
            and year_types == "single_year" 
            and base_name.str.contains("biodiversity_GBF8_groups_scores")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)                   # Rename the landuse

        # Plot_GBF8_1: Biodiversity contribution score (group) total
        bio_df_group = bio_df.groupby(['Group','Year']).sum(numeric_only=True).reset_index()
        
        bio_df_group = bio_df_group\
            .groupby(['Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_group.columns = ['name','data']
        bio_df_group['type'] = 'spline'
        bio_df_group.to_json(f'{SAVE_DIR}/biodiversity_GBF8_1_contribution_group_score_total.json', orient='records')
        
        
        # Plot_GBF8_2: Biodiversity contribution score (group) by Type
        bio_df_group_type_sum = bio_df\
            .groupby(['Year','Type','Group'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_df_group_type_sum = bio_df_group_type_sum\
            .groupby(['Type','Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_group_type_records = []
        for idx,df in bio_df_group_type_sum.groupby('Group'):
            df = df.drop('Group',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            bio_df_group_type_records.append({'name':idx,'data':df.to_dict(orient='records')})
        
        with open(f'{SAVE_DIR}/biodiversity_GBF8_2_contribution_group_score_by_type.json', 'w') as outfile:
            json.dump(bio_df_group_type_records, outfile)
        
        
        
        # Plot_GBF8_3: Biodiversity contribution score (group) by landuse
        bio_group_lu_sum = bio_df\
            .query('Type == "Agricultural land-use"')\
            .groupby(['Year','Landuse','Group'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('`Contribution Relative to Pre-1750 Level (%)` >1')
        
        bio_group_lu_sum = bio_group_lu_sum\
            .groupby(['Landuse','Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()

        bio_df_group_records = []
        for idx,df in bio_group_lu_sum.groupby('Group'):
            df = df.drop('Group',axis=1)
            df.columns = ['name','data']
            df = df.set_index('name').reindex(LANDUSE_ALL_RENAMED).reset_index().dropna()
            df['type'] = 'column'
            bio_df_group_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF8_3_contribution_group_score_by_landuse.json', 'w') as outfile:
            json.dump(bio_df_group_records, outfile)
            
        
        # Plot_GBF8_4: Biodiversity contribution score (group) by agricultural management
        bio_group_am_sum = bio_df\
            .groupby(['Year','Agri-Management','Group'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_group_am_sum = bio_group_am_sum\
            .groupby(['Agri-Management','Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_group_records = []
        for idx,df in bio_group_am_sum.groupby('Group'):
            df = df.drop('Group',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            bio_df_group_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF8_4_contribution_group_score_by_agri_management.json', 'w') as outfile:
            json.dump(bio_df_group_records, outfile)
            
            
        # Plot_GBF8_5: Biodiversity contribution score (group) by non-Agricultural land-use
        bio_group_non_ag_sum = bio_df\
            .query('Type == "Non-Agricultural land-use"')\
            .groupby(['Year','Landuse','Group'])\
            .sum(numeric_only=True)\
            .reset_index()
                
        bio_group_non_ag_sum = bio_group_non_ag_sum\
            .groupby(['Landuse','Group'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_group_records = []
        for idx,df in bio_group_non_ag_sum.groupby('Group'):
            df = df.drop('Group',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            bio_df_group_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF8_5_contribution_group_score_by_non_agri_landuse.json', 'w') as outfile:
            json.dump(bio_df_group_records, outfile)
            
            
            
        # Plot Species level biodiversity contribution score 
        filter_str = '''
            category == "biodiversity" 
            and year_types == "single_year" 
            and base_name.str.contains("biodiversity_GBF8_species_scores")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)                   # Rename the landuse

        # Plot_GBF8_6: Biodiversity contribution score (species) total
        bio_df_species = bio_df.groupby(['Species','Year']).sum(numeric_only=True).reset_index()
        
        bio_df_species = bio_df_species\
            .groupby(['Species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_species.columns = ['name','data']
        bio_df_species['type'] = 'spline'
        bio_df_species.to_json(f'{SAVE_DIR}/biodiversity_GBF8_6_contribution_species_score_total.json', orient='records')
        
        
        
        # Plot_GBF8_7: Biodiversity contribution score (species) by Type
        bio_df_species_type_sum = bio_df\
            .groupby(['Year','Type','Species'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_df_species_type_sum = bio_df_species_type_sum\
            .groupby(['Type','Species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_species_type_records = []
        for idx,df in bio_df_species_type_sum.groupby('Species'):
            df = df.drop('Species',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            bio_df_species_type_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF8_7_contribution_species_score_by_type.json', 'w') as outfile:
            json.dump(bio_df_species_type_records, outfile)
            
            
        # Plot_GBF8_8: Biodiversity contribution score (species) by agri landuse
        bio_species_lu_sum = bio_df\
            .query('Type == "Agricultural land-use"')\
            .groupby(['Year','Landuse','Species'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_species_lu_sum = bio_species_lu_sum\
            .groupby(['Landuse','Species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_species_records = []
        for idx,df in bio_species_lu_sum.groupby('Species'):
            df = df.drop('Species',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            df = df.set_index('name').reindex(AG_LANDUSE).reset_index()
            bio_df_species_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF8_8_contribution_species_score_by_landuse.json', 'w') as outfile:
            json.dump(bio_df_species_records, outfile)
            
            
        # Plot_GBF8_9: Biodiversity contribution score (species) by agricultural management
        bio_species_am_sum = bio_df\
            .groupby(['Year','Agri-Management','Species'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_species_am_sum = bio_species_am_sum\
            .groupby(['Agri-Management','Species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_species_records = []
        for idx,df in bio_species_am_sum.groupby('Species'):
            df = df.drop('Species',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            bio_df_species_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF8_9_contribution_species_score_by_agri_management.json', 'w') as outfile:
            json.dump(bio_df_species_records, outfile)
            
            
        # Plot_GBF8_10: Biodiversity contribution score (species) by non-agricultural landuse
        bio_species_non_ag_sum = bio_df\
            .query('Type == "Non-Agricultural land-use"')\
            .groupby(['Year','Landuse','Species'])\
            .sum(numeric_only=True)\
            .reset_index()
            
        bio_species_non_ag_sum = bio_species_non_ag_sum\
            .groupby(['Landuse','Species'])[['Year','Contribution Relative to Pre-1750 Level (%)']]\
            .apply(lambda x:list(map(list,zip(x['Year'],x['Contribution Relative to Pre-1750 Level (%)']))))\
            .reset_index()
            
        bio_df_species_records = []
        for idx,df in bio_species_non_ag_sum.groupby('Species'):
            df = df.drop('Species',axis=1)
            df.columns = ['name','data']
            df['type'] = 'column'
            bio_df_species_records.append({'name':idx,'data':df.to_dict(orient='records')})
            
        with open(f'{SAVE_DIR}/biodiversity_GBF8_10_contribution_species_score_by_non_agri_landuse.json', 'w') as outfile:
            json.dump(bio_df_species_records, outfile)
        
        
 
    
    
    #########################################################
    #                         7) Maps                       #
    #########################################################
    map_files = files.query('base_ext == ".html" and year_types != "begin_end_year"')
    map_save_dir = f"{SAVE_DIR}/Map_data/"
    
    # Create the directory to save map_html if it does not exist
    if  not os.path.exists(map_save_dir):
        os.makedirs(map_save_dir)
        
    # Remove any existing map files in the save directory
    if os.path.exists(map_save_dir):
        for file in os.listdir(map_save_dir):
            file_path = os.path.join(map_save_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # Function to move a file from one location to another if the file exists
    def move_html(path_from, path_to):
        if os.path.exists(path_from):
            shutil.move(path_from, path_to)
    
    # Move the map files to the save directory
    tasks = [
        delayed(move_html)(row['path'], map_save_dir)
        for _,row in map_files.iterrows()
    ]
    
    worker = min(settings.WRITE_THREADS, len(tasks)) if len(tasks) > 0 else 1
    
    Parallel(n_jobs=worker)(tasks)
    
    


    #########################################################
    #              Report success info                      #
    #########################################################

    print('Report data created successfully!\n')