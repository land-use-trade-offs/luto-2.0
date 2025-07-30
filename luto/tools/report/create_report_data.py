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
from luto import settings
from joblib import Parallel, delayed

from luto.economics.off_land_commodity import get_demand_df
from luto.tools.report.data_tools import get_all_files
from luto.tools.report.data_tools.helper_func import select_years

from luto.tools.report.data_tools.parameters import (
    AG_LANDUSE,
    COLORS,
    COLORS_RANK,
    COLORS_AM_NONAG,
    COLORS_COMMODITIES,
    COLORS_ECONOMY_TYPE,
    COLORS_GHG,
    COLORS_LM,
    COLORS_LU,
    COMMODITIES_ALL,
    COMMODITIES_OFF_LAND,
    GHG_CATEGORY,
    GHG_NAMES,
    LANDUSE_ALL_RENAMED,
    LU_CROPS,
    LU_LVSTKS,
    LU_UNALLOW,
    RENAME_AM_NON_AG,
    RENAME_NON_AG,
    SPATIAL_MAP_DICT
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
    
    # Function to get rank color based on value
    def get_rank_color(x):
        if x in [None, np.nan, 'N.A.']:
            return COLORS_RANK['N.A.']
        elif x <= 10:
            return COLORS_RANK['1-10']
        elif x <= 20:
            return COLORS_RANK['11-20']
        else:
            return COLORS_RANK['>=21']

    def get_rank_area_type(x:pd.Series) -> str:
        if not x['Type'] is np.nan:
            return 'Agricultural Management'

        if x['Land-use'] in LU_CROPS + LU_LVSTKS:
            return 'Agricultural Landuse'
        elif x['Land-use'] in LU_UNALLOW:
            return 'Unallocated land'
        elif x['Land-use'] in RENAME_NON_AG.values():
            return 'Non-Agricultural Landuse'
        else:
            return 'Unknown'
        
        

    ####################################################
    #                    1) Area Change                #
    ####################################################
    area_dvar_paths = files.query('category == "area"').reset_index(drop=True)
    
    ag_dvar_dfs = area_dvar_paths.query('base_name == "area_agricultural_landuse"').reset_index(drop=True)
    ag_dvar_area = pd.concat([pd.read_csv(path) for path in ag_dvar_dfs['path']], ignore_index=True)
    ag_dvar_area['Source'] = 'Agricultural Landuse'
    ag_dvar_area['Area (ha)'] = ag_dvar_area['Area (ha)'].round(2)

    non_ag_dvar_dfs = area_dvar_paths.query('base_name == "area_non_agricultural_landuse"').reset_index(drop=True)
    non_ag_dvar_area = pd.concat([pd.read_csv(path) for path in non_ag_dvar_dfs['path'] if not pd.read_csv(path).empty], ignore_index=True)
    non_ag_dvar_area['Land-use'] = non_ag_dvar_area['Land-use'].replace(RENAME_NON_AG)
    non_ag_dvar_area['Source'] = 'Non-Agricultural Landuse'
    non_ag_dvar_area['Area (ha)'] = non_ag_dvar_area['Area (ha)'].round(2)

    am_dvar_dfs = area_dvar_paths.query('base_name == "area_agricultural_management"').reset_index(drop=True)
    am_dvar_area = pd.concat([pd.read_csv(path) for path in am_dvar_dfs['path'] if not pd.read_csv(path).empty], ignore_index=True)
    am_dvar_area = am_dvar_area.replace(RENAME_AM_NON_AG)
    am_dvar_area['Source'] = 'Agricultural Management'
    am_dvar_area['Area (ha)'] = am_dvar_area['Area (ha)'].round(2)
    

    lu_group_raw = pd.read_csv('luto/tools/report/Assets/lu_group.csv')
    colors_lu_category = lu_group_raw.set_index('Category')['color_HEX'].to_dict()
    lu_group = lu_group_raw.set_index(['Category', 'color_HEX'])\
        .apply(lambda x: x.str.split(', ').explode())\
        .reset_index()
        
    # -------------------- Area ranking --------------------
    area_ranking_raw = pd.concat([ag_dvar_area, non_ag_dvar_area, am_dvar_area])\
         .assign(Area_type=lambda x: x.apply(get_rank_area_type, axis=1))
        
    area_ranking_type_region = area_ranking_raw\
        .query('Area_type != "Unallocated land"')\
        .groupby(['Year', 'region', 'Area_type'])[['Area (ha)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Area_type', 'Area (ha)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'Area_type']).cumcount() + 1)\
        .assign(Percent=lambda x: x['Area (ha)'] / x.groupby(['Year', 'Area_type'])['Area (ha)'].transform('sum') * 100)\
        .round({'Percent': 2, 'Area (ha)': 2})
    area_ranking_type_AUS = area_ranking_raw\
        .query('Area_type != "Unallocated land"')\
        .groupby(['Year', 'Area_type'])[['Area (ha)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Area_type', 'Area (ha)'], ascending=[True, True, False])\
        .assign(Rank='N.A.', Percent=100, region='AUSTRALIA')\
        .round({'Percent': 2, 'Area (ha)': 2})
        
    area_ranking_total = pd.concat([ag_dvar_area, non_ag_dvar_area, am_dvar_area])\
        .assign(Area_type=lambda x: x.apply(get_rank_area_type, axis=1))\
        .groupby(['Year', 'region'])[["Area (ha)"]]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Area (ha)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount() + 1)\
        .assign(Area_type='Total')\
        .assign(Percent=lambda x: x['Area (ha)'] / x.groupby(['Year'])['Area (ha)'].transform('sum') * 100)\
        .round({'Percent': 2, 'Area (ha)': 2})
    area_ranking_AUS = pd.concat([ag_dvar_area, non_ag_dvar_area, am_dvar_area])\
        .assign(Area_type=lambda x: x.apply(get_rank_area_type, axis=1))\
        .groupby(['Year'])[["Area (ha)"]]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Area (ha)'], ascending=[True, False])\
        .assign(Rank='N.A.', Area_type='Total', Percent=100, region='AUSTRALIA')\
        .round({'Percent': 2, 'Area (ha)': 2})

    area_ranking = pd.concat([area_ranking_type_region, area_ranking_type_AUS, area_ranking_total, area_ranking_AUS], ignore_index=True)
    area_ranking = area_ranking.set_index(['Year', 'region', 'Area_type'])\
        .reindex(
            index=pd.MultiIndex.from_product(
                [years, area_ranking['region'].unique(), area_ranking['Area_type'].unique()],
                names=['Year', 'region', 'Area_type']), fill_value=None )\
        .reset_index()\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))
        

    out_dict = {}
    for (region, area_type), df in area_ranking.groupby(['region', 'Area_type']):
        if region not in out_dict:
            out_dict[region] = {}
        if area_type not in out_dict[region]:
            out_dict[region][area_type] = {}

        df = df.drop('region', axis=1)
        out_dict[region][area_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][area_type]['Percent'] = df.set_index('Year')['Percent'].replace({np.nan: 0}).to_dict()
        out_dict[region][area_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][area_type]['value'] = df.set_index('Year')['Area (ha)'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else 0).to_dict()

    with open(f'{SAVE_DIR}/Area_ranking.json', 'w') as f:
        json.dump(out_dict, f, indent=2)
        

    # -------------------- Area overview --------------------
    area_ag_nonag = pd.concat([ag_dvar_area, non_ag_dvar_area], ignore_index=True)
    area_ag_nonag = area_ag_nonag.replace(RENAME_AM_NON_AG)    
    area_ag_nonag = area_ag_nonag.merge(lu_group, left_on='Land-use', right_on='Land-use', how='left')
    
    group_cols = ['Land-use', 'Category']
    
    for idx, col in enumerate(group_cols):

        df_AUS = area_ag_nonag\
            .groupby(['Year', col])[['Area (ha)']]\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        
        df_region = area_ag_nonag\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Land-use":
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LU[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water_supply':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LM[x])
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Category':
            df_wide['color'] = df_wide['name'].apply(lambda x: colors_lu_category[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            out_dict[region] = df.to_dict(orient='records')

        with open(f'{SAVE_DIR}/Area_overview_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
    
    
    area_df = pd.concat([ag_dvar_area, non_ag_dvar_area, am_dvar_area], ignore_index=True)
    group_cols = ['Source']
    for _, col in enumerate(group_cols):

        df_AUS = area_df\
            .groupby(['Year', col])[['Area (ha)']]\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        
        df_region = area_df\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Land-use":
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LU[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water_supply':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LM[x])
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Category':
            df_wide['color'] = df_wide['name'].apply(lambda x: colors_lu_category[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            out_dict[region] = df.to_dict(orient='records')

        with open(f'{SAVE_DIR}/Area_overview_3_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
            
    
    # -------------------- Area by Agricultural land --------------------
    group_cols = ['Land-use', 'Water_supply']
    for idx, col in enumerate(group_cols):

        df_AUS = ag_dvar_area\
            .groupby(['Year', col])[['Area (ha)']]\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        
        df_region = ag_dvar_area\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Land-use":
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LU[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water_supply':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LM[x])
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Category':
            df_wide['color'] = df_wide['name'].apply(lambda x: colors_lu_category[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Area_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)


    # -------------------- Area by Agricultural Management Area (ha) Land use --------------------
    group_cols = ['Type', 'Water_supply', 'Land-use']
    
    for idx, col in enumerate(group_cols):

        df_AUS = am_dvar_area\
            .groupby(['Year', col])[['Area (ha)']]\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
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
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Land-use":
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LU[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water_supply':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LM[x])
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Category':
            df_wide['color'] = df_wide['name'].apply(lambda x: colors_lu_category[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Area_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
            
            
    # -------------------- Area by Non-Agricultural Landuse --------------------
    group_cols = ['Land-use']
    
    for idx, col in enumerate(group_cols):

        df_AUS = non_ag_dvar_area\
            .groupby(['Year', col])[['Area (ha)']]\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        
        df_region = non_ag_dvar_area\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Land-use":
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_AM_NONAG[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water_supply':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LM[x])
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Category':
            df_wide['color'] = df_wide['name'].apply(lambda x: colors_lu_category[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Area_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
            
            


    # -------------------- Transition areas (start-end) --------------------
    transition_path = files.query('category =="transition_matrix"')
    transition_df_region = pd.read_csv(transition_path['path'].values[0], index_col=0).reset_index() 
    transition_df_region = transition_df_region.replace(RENAME_AM_NON_AG)

    transition_df_AUS = transition_df_region.groupby(['From Land-use', 'To Land-use'])[['Area (ha)']].sum().reset_index()
    transition_df_AUS['region'] = 'AUSTRALIA'

    transition_df = pd.concat([transition_df_AUS, transition_df_region], ignore_index=True)


    out_dict = {}
    for (region, df) in transition_df.groupby('region'):
        out_dict[region] = {}
        
        transition_mat = df.pivot(index='From Land-use', columns='To Land-use', values='Area (ha)')
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

        out_dict[region]['area'] = heat_area_html
        out_dict[region]['pct'] = heat_pct_html

    with open(f'{SAVE_DIR}/Area_transition_start_end.json', 'w') as f:
        json.dump(out_dict, f, indent=2)
        
        
    # -------------------- Transition areas (year-to-year) --------------------
    transition_path = files.query('base_name =="crosstab-lumap"')
    transition_df_region = pd.concat([pd.read_csv(path) for path in transition_path['path']], ignore_index=True)
    transition_df_region = transition_df_region.replace(RENAME_AM_NON_AG)

    transition_df_AUS = transition_df_region.groupby(['Year', 'From land-use', 'To land-use'])[['Area (ha)']].sum().reset_index()
    transition_df_AUS['region'] = 'AUSTRALIA'

    transition_df = pd.concat([transition_df_AUS, transition_df_region], ignore_index=True)
    
    out_dict = {region: {'area': {}, 'pct':{}} for region in transition_df['region'].unique()}
    for (year, region), df in transition_df.groupby(['Year', 'region']):
        
        transition_mat = df.pivot(index='From land-use', columns='To land-use', values='Area (ha)')
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

        out_dict[region]['area'][str(year)] = heat_area_html
        out_dict[region]['pct'][str(year)] = heat_pct_html
        
    with open(f'{SAVE_DIR}/Area_transition_year_to_year.json', 'w') as f:
        json.dump(out_dict, f, indent=2)
        
        

    ####################################################
    #                   2) Demand                      #
    ####################################################
    
    demand_files = files.query('category == "quantity"')
    
    quantity_LUTO = demand_files\
        .query('base_name == "quantity_production_t_separate"')\
        .reset_index(drop=True)\
        .query('Year.isin(@years)')
    quantity_LUTO = pd.concat(
            [pd.read_csv(path).assign(Year=Year) for Year,path in quantity_LUTO[['Year','path']].values.tolist()],
            ignore_index=True)\
        .assign(Commodity= lambda x: x['Commodity'].str.capitalize())\
        .replace({'Sheep lexp': 'Sheep live export', 'Beef lexp': 'Beef live export'})\
        .round({'`Production (t/KL)`': 2})
        
        
    DEMAND_DATA_long = get_demand_df()\
        .replace({'Beef lexp': 'Beef live export', 'Sheep lexp': 'Sheep live export'})\
        .set_index(['Commodity', 'Type', 'Year'])\
        .reindex(COMMODITIES_ALL, level=0)\
        .reset_index()\
        .replace(RENAME_AM_NON_AG)\
        .assign(on_off_land=lambda x: np.where(x['Commodity'].isin(COMMODITIES_OFF_LAND), 'Off-land', 'On-land'))
    

    # -------------------- Demand --------------------
    group_cols = ['Type', 'on_off_land', 'Commodity']

    for idx, col in enumerate(group_cols):

        if col == 'Type':
            _df = DEMAND_DATA_long.query(f'Year.isin({years_select})')
        else:
            _df = DEMAND_DATA_long.query(f'Year.isin({years})')
            
        df_AUS = _df\
            .groupby(['Year', col])[['Quantity (tonnes, KL)']]\
            .sum()\
            .reset_index()\
            .round({'Quantity (tonnes, KL)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Quantity (tonnes, KL)']]\
            .apply(lambda x: x[['Year','Quantity (tonnes, KL)']].values.tolist())\
            .reset_index()
        df_AUS_wide.columns = ['name','data']
        df_AUS_wide['type'] = 'column'
 
        if col == "Land-use":
            df_AUS_wide['color'] = df_AUS_wide['name'].apply(lambda x: COLORS_LU[x])
            df_AUS_wide['name_order'] = df_AUS_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_AUS_wide = df_AUS_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col.lower() == 'commodity':
            df_AUS_wide['color'] = df_AUS_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
            df_AUS_wide['name_order'] = df_AUS_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_AUS_wide = df_AUS_wide.sort_values('name_order').drop(columns=['name_order'])
 
        df_AUS_wide.to_json(f'{SAVE_DIR}/Production_demand_{idx+1}_{col.replace(" ", "_")}.json', orient='records')


    # -------------------- Production limit. --------------------
    demand_limit = DEMAND_DATA_long.query('Type == "Domestic" and on_off_land == "On-land" and Year.isin(@years)')
    demand_limit_wide = demand_limit.groupby(['Commodity', 'Year'])[['Quantity (tonnes, KL)']]\
        .sum()\
        .reset_index()\
        .groupby('Commodity')[['Year','Quantity (tonnes, KL)']]\
        .apply(lambda x: x[['Year','Quantity (tonnes, KL)']].values.tolist())\
        .reset_index()
    demand_limit_wide.columns = ['name','data']
    demand_limit_wide['type'] = 'column'
    demand_limit_wide.to_json(f'{SAVE_DIR}/Production_demand_4_Limit.json', orient='records')



    # -------------------- Sum of commodity production --------------------
    group_cols = ['Commodity', 'Type']
    for idx, col in enumerate(group_cols):

        df_AUS = quantity_LUTO\
            .groupby(['Year', col])[['Production (t/KL)']]\
            .sum()\
            .reset_index()\
            .round({'Production (t/KL)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Production (t/KL)']]\
            .apply(lambda x: x[['Year','Production (t/KL)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name','data','region']
        df_AUS_wide['type'] = 'column'
        
        
        df_region = quantity_LUTO\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Production (t/KL)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Production (t/KL)']]\
            .apply(lambda x: x[['Year','Production (t/KL)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region','data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        if col == "Land-use":
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LU[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Production_sum_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)




    # -------------------- Demand achievement in the final target year (%) --------------------
    quantity_diff = demand_files.query('base_name == "quantity_comparison"').reset_index(drop=True)
    quantity_diff = pd.concat([pd.read_csv(path) for path in quantity_diff['path']], ignore_index=True)
    quantity_diff = quantity_diff.replace({'Sheep lexp': 'Sheep live export', 'Beef lexp': 'Beef live export'})
    quantity_diff = quantity_diff[['Year','Commodity','Prop_diff (%)']].rename(columns={'Prop_diff (%)': 'Demand Achievement (%)'})

    mask_AUS = quantity_diff.groupby('Commodity'
        )['Demand Achievement (%)'
        ].transform(lambda x: abs(round(x) - 100) > 0.01)
    quantity_diff_AUS = quantity_diff[mask_AUS].copy()
    quantity_diff_wide_AUS = quantity_diff_AUS\
        .groupby(['Commodity'])[['Year','Demand Achievement (%)']]\
        .apply(lambda x: list(map(list,zip(x['Year'],x['Demand Achievement (%)']))))\
        .reset_index()
        
    quantity_diff_wide_AUS['type'] = 'line'
    quantity_diff_wide_AUS.columns = ['name','data', 'type']

    quantity_diff_wide_AUS_data = {
        'AUSTRALIA': quantity_diff_wide_AUS.to_dict(orient='records')
    }
    with open(f'{SAVE_DIR}/Production_achive_percent.json', 'w') as f:
        json.dump(quantity_diff_wide_AUS_data, f)    
    
    
    # -------------------- Commodity production for ag, non-ag, and agricultural management --------------------
    for idx, _type in enumerate(quantity_LUTO['Type'].unique()):
        _df = quantity_LUTO.query(f'Type == "{_type}"').query('`Production (t/KL)` > 0').copy()

        group_cols = ['Commodity']
        for col in group_cols:

            df_AUS = _df\
                .groupby(['Year', col])[['Production (t/KL)']]\
                .sum()\
                .reset_index()\
                .round({'Production (t/KL)': 2})
            df_AUS_wide = df_AUS.groupby([col])[['Year','Production (t/KL)']]\
                .apply(lambda x: x[['Year','Production (t/KL)']].values.tolist())\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name','data','region']
            df_AUS_wide['type'] = 'column'
            
            
            df_region = _df\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .round({'Production (t/KL)': 2})
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Production (t/KL)']]\
                .apply(lambda x: x[['Year','Production (t/KL)']].values.tolist())\
                .reset_index()
            df_region_wide.columns = ['name', 'region','data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            if col == "Land-use":
                df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LU[x])
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col.lower() == 'commodity':
                df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
                df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

            out_dict = {}
            for region, df in df_wide.groupby('region'):
                df = df.drop('region', axis=1)
                out_dict[region] = df.to_dict(orient='records')
                
            with open(f'{SAVE_DIR}/Production_LUTO_{idx+1}_{_type.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
            
            
    



    ####################################################
    #                  3) Economics                    #
    ####################################################
    
    # -------------------- Get the revenue and cost data --------------------
    revenue_ag_df = files.query('base_name == "revenue_agricultural_commodity"').reset_index(drop=True)
    revenue_ag_df = pd.concat([pd.read_csv(path) for path in revenue_ag_df['path']], ignore_index=True)
    revenue_ag_df = revenue_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural land-use (revenue)')
    
    cost_ag_df = files.query('base_name == "cost_agricultural_commodity"').reset_index(drop=True)
    cost_ag_df = pd.concat([pd.read_csv(path) for path in cost_ag_df['path']], ignore_index=True)
    cost_ag_df = cost_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural land-use (cost)')
    cost_ag_df['Value ($)'] = cost_ag_df['Value ($)'] * -1          # Convert cost to negative value
    
    revenue_am_df = files.query('base_name == "revenue_agricultural_management"').reset_index(drop=True)
    revenue_am_df = pd.concat([pd.read_csv(path) for path in revenue_am_df['path']], ignore_index=True)
    revenue_am_df = revenue_am_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural Management (revenue)')
    
    cost_am_df = files.query('base_name == "cost_agricultural_management"').reset_index(drop=True)
    cost_am_df = pd.concat([pd.read_csv(path) for path in cost_am_df['path']], ignore_index=True)
    cost_am_df = cost_am_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural Management (cost)')
    cost_am_df['Value ($)'] = cost_am_df['Value ($)'] * -1          # Convert cost to negative value

    revenue_non_ag_df = files.query('base_name == "revenue_non_ag"').reset_index(drop=True)
    revenue_non_ag_df = pd.concat([pd.read_csv(path) for path in revenue_non_ag_df['path']], ignore_index=True)
    revenue_non_ag_df = revenue_non_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Non-agricultural land-use (revenue)')

    cost_non_ag_df = files.query('base_name == "cost_non_ag"').reset_index(drop=True)
    cost_non_ag_df = pd.concat([pd.read_csv(path) for path in cost_non_ag_df['path']], ignore_index=True)
    cost_non_ag_df = cost_non_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Non-agricultural land-use (cost)')
    cost_non_ag_df['Value ($)'] = cost_non_ag_df['Value ($)'] * -1  # Convert cost to negative value
    
    net_ag_df = revenue_ag_df.groupby('Year')[['Value ($)']].sum().reset_index()
    net_ag_df['Value ($)'] = net_ag_df['Value ($)'] + cost_ag_df.groupby('Year')[['Value ($)']].sum().reset_index()['Value ($)']
    
    net_am_df = revenue_am_df.groupby('Year')[['Value ($)']].sum().reset_index()
    net_am_df['Value ($)'] = net_am_df['Value ($)'] + cost_am_df.groupby('Year')[['Value ($)']].sum().reset_index()['Value ($)']
    
    net_non_ag_df = revenue_non_ag_df.groupby('Year')[['Value ($)']].sum().reset_index()
    net_non_ag_df['Value ($)'] = net_non_ag_df['Value ($)'] + cost_non_ag_df.groupby('Year')[['Value ($)']].sum().reset_index()['Value ($)']
    
    cost_transition_ag2ag_df = files.query('base_name == "cost_transition_ag2ag"').reset_index(drop=True)
    cost_transition_ag2ag_df = pd.concat([pd.read_csv(path) for path in cost_transition_ag2ag_df['path'] if not pd.read_csv(path).empty], ignore_index=True)
    cost_transition_ag2ag_df = cost_transition_ag2ag_df.replace(RENAME_AM_NON_AG).assign(Source='Transition cost (Ag2Ag)')
    cost_transition_ag2ag_df['Value ($)'] = cost_transition_ag2ag_df['Cost ($)']  * -1          # Convert cost to negative value

    cost_transition_ag2non_ag_df = files.query('base_name == "cost_transition_ag2non_ag"').reset_index(drop=True)
    cost_transition_ag2non_ag_df = pd.concat([pd.read_csv(path) for path in cost_transition_ag2non_ag_df['path'] if not pd.read_csv(path).empty], ignore_index=True)
    cost_transition_ag2non_ag_df = cost_transition_ag2non_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Transition cost (Ag2Non-Ag)')
    cost_transition_ag2non_ag_df['Value ($)'] = cost_transition_ag2non_ag_df['Cost ($)'] * -1   # Convert cost to negative value

    cost_transition_non_ag2ag_df = files.query('base_name == "cost_transition_non_ag2_ag"').reset_index(drop=True)
    cost_transition_non_ag2ag_df = pd.concat([pd.read_csv(path) for path in cost_transition_non_ag2ag_df['path'] if not pd.read_csv(path).empty], ignore_index=True)
    cost_transition_non_ag2ag_df = cost_transition_non_ag2ag_df.replace(RENAME_AM_NON_AG).assign(Source='Transition cost (Non-Ag2Ag)').dropna(subset=['Cost ($)'])
    cost_transition_non_ag2ag_df['Value ($)'] = cost_transition_non_ag2ag_df['Cost ($)'] * -1   # Convert cost to negative value

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
                cost_transition_non_ag2ag_df
            ]
        ).round({'Value ($)': 2}
        ).query('abs(`Value ($)`) > 0'
        ).reset_index(drop=True) 
        
    order = [
        'Agricultural land-use (revenue)', 
        'Agricultural Management (revenue)', 
        'Non-agricultural land-use (revenue)',
        'Agricultural land-use (cost)', 
        'Agricultural Management (cost)', 
        'Non-agricultural land-use (cost)',
        'Transition cost (Ag2Ag)',
        'Transition cost (Ag2Non-Ag)',
        'Transition cost (Non-Ag2Ag)',
        'Profit'
    ]


    # -------------------- Economic ranking --------------------
    revenue_df_region = pd.concat([revenue_ag_df, revenue_am_df, revenue_non_ag_df]
        ).query('`Value ($)` >= 0'
        ).groupby(['Year', 'region']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank=lambda x: x.groupby(['Year']).cumcount() + 1
        ).assign(Percent=lambda x: x['Value ($)'] / x.groupby(['Year'])['Value ($)'].transform('sum') * 100
        ).assign(Source='Revenue'
        ).round({'Value ($)': 2, 'Percent': 2})
    revenue_df_AUS = pd.concat([revenue_ag_df, revenue_am_df, revenue_non_ag_df]
        ).query('`Value ($)` >= 0'
        ).groupby(['Year']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank='N.A.', Percent=100, Source='Revenue', region='AUSTRALIA'
        ).round({'Value ($)': 2, 'Percent': 2})

    cost_df_region = pd.concat([cost_ag_df, cost_am_df, cost_non_ag_df]
        ).query('`Value ($)` < 0'
        ).groupby(['Year', 'region']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).assign(**{'Value ($)': lambda x: abs(x['Value ($)'])}
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank=lambda x: x.groupby(['Year']).cumcount() + 1
        ).assign(Percent=lambda x: x['Value ($)'] / x.groupby('Year')['Value ($)'].transform('sum') * 100
        ).assign(Source='Cost'
        ).round({'Value ($)': 2, 'Percent': 2})
    cost_df_AUS = pd.concat([cost_ag_df, cost_am_df, cost_non_ag_df]
        ).query('`Value ($)` < 0'
        ).groupby(['Year']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).assign(**{'Value ($)': lambda x: abs(x['Value ($)'])}
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank='N.A.', Percent=100, Source='Cost', region='AUSTRALIA'
        ).round({'Value ($)': 2, 'Percent': 2})
 

    ranking_df = pd.concat([revenue_df_region, revenue_df_AUS, cost_df_region, cost_df_AUS])
    ranking_df = ranking_df.set_index(['region', 'Source', 'Year']
        ).reindex(
            index=pd.MultiIndex.from_product(
                [ranking_df['region'].unique(), ranking_df['Source'].unique(), ranking_df['Year'].unique()],
                names=['region', 'Source', 'Year']), fill_value=None
        ).reset_index(
        ).assign(color= lambda x: x['Rank'].map(get_rank_color))
        


    out_dict = {}
    for (region, source), df in ranking_df.groupby(['region', 'Source']):
        if region not in out_dict:
            out_dict[region] = {}
        if not source in out_dict[region]:
            out_dict[region][source] = {}
        
        df = df.drop(columns='region')
        out_dict[region][source]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['Percent'] = df.set_index('Year')['Percent'].replace({np.nan: 0}).to_dict()
        out_dict[region][source]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['value'] = df.set_index('Year')['Value ($)'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else 0).to_dict()

    with open(f'{SAVE_DIR}/Economics_ranking.json', 'w') as f:
        json.dump(out_dict, f, indent=2)
        

    # -------------------- Economy overview --------------------
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
        'line', 
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
            'line',
        ]
        df_col['region'] = region
        rev_cost_net_wide_region = pd.concat([rev_cost_net_wide_region, df_col])


    rev_cost_wide_json = pd.concat([economics_df_AUS_wide, rev_cost_net_wide_region], axis=0).reset_index(drop=True)
    rev_cost_wide_json['name_order'] = rev_cost_wide_json['name'].map({name: i for i, name in enumerate(order)})
    rev_cost_wide_json = rev_cost_wide_json.sort_values(['region', 'name_order']).drop(columns=['name_order']).reset_index(drop=True)


    out_dict = {}
    for region,df in rev_cost_wide_json.groupby('region'):
        df = df.drop(columns='region')
        df.columns = ['name','data','type']
        out_dict[region] = df.to_dict(orient='records')
        
    with open(f'{SAVE_DIR}/Economics_overview.json', 'w') as f:
        json.dump(out_dict, f)
        
 
 
 
    # -------------------- Economics for ag --------------------
    revenue_ag_df = revenue_ag_df.assign(Rev_Cost='Revenue')
    cost_ag_df = cost_ag_df.assign(Rev_Cost='Cost')

    economics_ag = pd.concat([revenue_ag_df, cost_ag_df]
        ).round({'Value ($)': 2}
        ).query('abs(`Value ($)`) > 0'
        ).reset_index(drop=True)

    group_cols = ['Land-use', 'Type', 'Water_supply']

    for idx, col in enumerate(group_cols):
        df_AUS = economics_ag\
            .groupby(['Year', 'Rev_Cost', col])[['Value ($)']]\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col, 'Rev_Cost'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'Rev_Cost', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = economics_ag\
            .groupby(['Year', 'Rev_Cost', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region', 'Rev_Cost'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region','Rev_Cost', 'data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Land-use":
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
            df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)
        elif col == 'Water_supply':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
            df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
            df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
            df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)
        elif col == 'Type':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_ECONOMY_TYPE[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: list(COLORS_ECONOMY_TYPE.keys()).index(x) if x in COLORS_ECONOMY_TYPE else -1)
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])


        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region','Rev_Cost'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Economics_split_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
    


    # -------------------- Economics for ag-management --------------------
    revenue_am_df = revenue_am_df.assign(Rev_Cost='Revenue')
    cost_am_df = cost_am_df.assign(Rev_Cost='Cost')

    economics_am = pd.concat([revenue_am_df, cost_am_df]
        ).round({'Value ($)': 2}
        ).query('abs(`Value ($)`) > 0'
        ).reset_index(drop=True)


    group_cols = ['Management Type', 'Water_supply', 'Land-use']

    for idx, col in enumerate(group_cols):
        df_AUS = economics_am\
            .groupby(['Year', 'Rev_Cost', col])[['Value ($)']]\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col, 'Rev_Cost'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'Rev_Cost', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = economics_am\
            .groupby(['Year', 'Rev_Cost', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region', 'Rev_Cost'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region','Rev_Cost', 'data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Land-use":
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
            df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)
            df_wide.loc[df_wide['name'] == 'Unallocated - natural land', 'linkedTo'] = None
        elif col == 'Water_supply':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
            df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
            df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)
        elif col == 'Management Type':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
            df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)
            df_wide.loc[df_wide['name'] == 'Early dry-season savanna burning', 'linkedTo'] = None
        
        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region','Rev_Cost'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Economics_split_AM_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)


    # -------------------- Economics for non-agriculture --------------------
    revenue_non_ag_df = revenue_non_ag_df.assign(Rev_Cost='Revenue')
    cost_non_ag_df = cost_non_ag_df.assign(Rev_Cost='Cost')

    economics_non_ag = pd.concat([revenue_non_ag_df, cost_non_ag_df]
        ).round({'Value ($)': 2}
        ).query('abs(`Value ($)`) > 0'
        ).reset_index(drop=True)


    group_cols = ['Land-use']

    for idx, col in enumerate(group_cols):
        df_AUS = economics_non_ag\
            .groupby(['Year', 'Rev_Cost', col])[['Value ($)']]\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col, 'Rev_Cost'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'Rev_Cost', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = economics_non_ag\
            .groupby(['Year', 'Rev_Cost', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region', 'Rev_Cost'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region','Rev_Cost', 'data']
        df_region_wide['type'] = 'column'
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
        df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)
        df_wide.loc[df_wide['name'].isin([
            'Environmental plantings (mixed species)',
            'Riparian buffer restoration (mixed species)',
            'Carbon plantings (monoculture)',
            'Destocked - natural land'
            ]), 'linkedTo'] = None

        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region','Rev_Cost'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        with open(f'{SAVE_DIR}/Economics_split_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)


    # -------------------- Transition cost for Ag2Ag --------------------
    cost_transition_ag2ag_df['Value ($)'] = cost_transition_ag2ag_df['Value ($)'] * -1  # Convert from negative to positive
    group_cols = ['Type', 'From land-use', 'To land-use']
    
    for idx, col in enumerate(group_cols):
        df_AUS = cost_transition_ag2ag_df\
            .groupby(['Year', col])[['Value ($)']]\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = cost_transition_ag2ag_df\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region', 'data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Economics_transition_split_ag2ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
      

    # -------------------- Transition cost matrix for Ag2Ag --------------------
    cost_transition_ag2ag_trans_mat_AUS = cost_transition_ag2ag_df\
        .groupby(['Year','From land-use', 'To land-use'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value ($)': 2})\
        .query('abs(`Value ($)`) > 0')\
        .assign(region='AUSTRALIA')

    cost_transition_ag2ag_trans_mat_region_df = cost_transition_ag2ag_df\
        .groupby(['Year','From land-use', 'To land-use', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value ($)': 2})
        
        
    cost_transition_ag2ag_trans_mat = pd.concat([
        cost_transition_ag2ag_trans_mat_AUS,
        cost_transition_ag2ag_trans_mat_region_df
    ])


    out_dict_area = {}
    for (region,year),df in cost_transition_ag2ag_trans_mat.groupby(['region', 'Year']):
        
        out_dict_area.setdefault(region, {})
        
        transition_mat = df.pivot(index='From land-use', columns='To land-use', values='Value ($)')
        transition_mat = transition_mat.reindex(index=AG_LANDUSE, columns=AG_LANDUSE)
        transition_mat = transition_mat.fillna(0)
        total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
        
        transition_mat['SUM'] = transition_mat.sum(axis=1)
        transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

        heat_area = transition_mat.style.background_gradient(
            cmap='Oranges',
            axis=1,
            subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
        ).format('{:,.0f}')

        heat_area_html = heat_area.to_html()
        heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)

        out_dict_area[region][str(year)] = rf'{heat_area_html}'

    with open(f'{SAVE_DIR}/Economics_transition_mat_ag2ag.json', 'w', encoding='utf-8') as f:
        json.dump(out_dict_area, f, ensure_ascii=False, indent=2)






    # -------------------- Transition cost for Ag2Non-Ag --------------------
    cost_transition_ag2non_ag_df['Value ($)'] = cost_transition_ag2non_ag_df['Value ($)'] * -1  # Convert from negative to positive
    group_cols = ['Cost type', 'From land-use', 'To land-use']
    
    for idx, col in enumerate(group_cols):
        df_AUS = cost_transition_ag2non_ag_df\
            .groupby(['Year', col])[['Value ($)']]\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = cost_transition_ag2non_ag_df\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region', 'data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Economics_transition_split_Ag2NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
        
  
    # -------------------- Transition cost matrix for Ag2Non-Ag --------------------
    cost_transition_ag2nonag_trans_mat_AUS = cost_transition_ag2non_ag_df\
        .groupby(['Year','From land-use', 'To land-use'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value ($)': 2})\
        .assign(region='AUSTRALIA')

    cost_transition_ag2nonag_trans_mat_region_df = cost_transition_ag2non_ag_df\
        .groupby(['Year','From land-use', 'To land-use', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value ($)': 2})
        
        
    cost_transition_ag2nonag_trans_mat = pd.concat([
        cost_transition_ag2nonag_trans_mat_AUS,
        cost_transition_ag2nonag_trans_mat_region_df
    ])


    out_dict_area = {}
    for (region,year),df in cost_transition_ag2nonag_trans_mat.groupby(['region', 'Year']):
        
        out_dict_area.setdefault(region, {})
        
        transition_mat = df.pivot(index='From land-use', columns='To land-use', values='Value ($)')
        transition_mat = transition_mat.reindex(index=AG_LANDUSE, columns=RENAME_NON_AG.values())
        transition_mat = transition_mat.fillna(0)
        total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
        
        transition_mat['SUM'] = transition_mat.sum(axis=1)
        transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

        heat_area = transition_mat.style.background_gradient(
            cmap='Oranges',
            axis=1,
            subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
        ).format('{:,.0f}')

        heat_area_html = heat_area.to_html()
        heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)

        out_dict_area[region][str(year)] = rf'{heat_area_html}'

    with open(f'{SAVE_DIR}/Economics_transition_mat_ag2nonag.json', 'w', encoding='utf-8') as f:
        json.dump(out_dict_area, f, ensure_ascii=False, indent=2)
    
    
    
    

    # -------------------- Transition cost for Non-Ag to Ag --------------------
    cost_transition_non_ag2ag_df['Value ($)'] = cost_transition_non_ag2ag_df['Value ($)'] * -1  # Convert from negative to positive
    group_cols = ['Cost type', 'From land-use', 'To land-use']
    
    for idx, col in enumerate(group_cols):
        df_AUS = cost_transition_non_ag2ag_df\
            .groupby(['Year', col])[['Value ($)']]\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = cost_transition_non_ag2ag_df\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .round({'Value ($)': 2})
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
            .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region', 'data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Economics_transition_split_NonAg2Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
    
    
    
    # -------------------- Transition cost matrix for Non-Ag to Ag --------------------
    cost_transition_nonag2ag_trans_mat_AUS = cost_transition_non_ag2ag_df\
        .groupby(['Year','From land-use', 'To land-use'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value ($)': 2})\
        .assign(region='AUSTRALIA')

    cost_transition_nonag2ag_trans_mat_region_df = cost_transition_non_ag2ag_df\
        .groupby(['Year','From land-use', 'To land-use', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value ($)': 2})
        
        
    cost_transition_nonag2ag_trans_mat = pd.concat([
        cost_transition_nonag2ag_trans_mat_AUS,
        cost_transition_nonag2ag_trans_mat_region_df
    ])


    out_dict_area = {}
    for (region,year),df in cost_transition_nonag2ag_trans_mat.groupby(['region', 'Year']):
        
        out_dict_area.setdefault(region, {})
        
        transition_mat = df.pivot(index='From land-use', columns='To land-use', values='Value ($)')
        transition_mat = transition_mat.reindex(index=RENAME_NON_AG.values(), columns=AG_LANDUSE)
        transition_mat = transition_mat.fillna(0)
        total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
        
        transition_mat['SUM'] = transition_mat.sum(axis=1)
        transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

        heat_area = transition_mat.style.background_gradient(
            cmap='Oranges',
            axis=1,
            subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
        ).format('{:,.0f}')

        heat_area_html = heat_area.to_html()
        heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)

        out_dict_area[region][str(year)] = rf'{heat_area_html}'

    with open(f'{SAVE_DIR}/Economics_transition_mat_nonag2ag.json', 'w', encoding='utf-8') as f:
        json.dump(out_dict_area, f, ensure_ascii=False, indent=2)




    ####################################################
    #                       4) GHGs                    #
    ####################################################
    if settings.GHG_EMISSIONS_LIMITS != 'off':
        filter_str = '''
        category == "GHG" 
        and base_name.str.contains("GHG_emissions") 
       
        '''.replace('\n', ' ').replace('  ', ' ')

        GHG_files = files.query(filter_str).reset_index(drop=True)

        GHG_ag = GHG_files.query('base_name.str.contains("agricultural_landuse")').reset_index(drop=True)
        GHG_ag = pd.concat([pd.read_csv(path) for path in GHG_ag['path']], ignore_index=True)
        GHG_ag = GHG_ag.replace(GHG_NAMES).round({'Value (t CO2e)': 2})
        
        GHG_non_ag = GHG_files.query('base_name.str.contains("no_ag_reduction")').reset_index(drop=True)
        GHG_non_ag = pd.concat([pd.read_csv(path) for path in GHG_non_ag['path'] if not pd.read_csv(path).empty], ignore_index=True)
        GHG_non_ag = GHG_non_ag.replace(RENAME_AM_NON_AG).round({'Value (t CO2e)': 2})
        
        GHG_ag_man = GHG_files.query('base_name.str.contains("agricultural_management")').reset_index(drop=True)
        GHG_ag_man = pd.concat([pd.read_csv(path) for path in GHG_ag_man['path'] if not pd.read_csv(path).empty], ignore_index=True)
        GHG_ag_man = GHG_ag_man.replace(RENAME_AM_NON_AG).round({'Value (t CO2e)': 2})
        
        GHG_transition = GHG_files.query('base_name.str.contains("transition_penalty")').reset_index(drop=True)
        GHG_transition = pd.concat([pd.read_csv(path) for path in GHG_transition['path'] if not pd.read_csv(path).empty], ignore_index=True)
        GHG_transition = GHG_transition.replace(RENAME_AM_NON_AG).round({'Value (t CO2e)': 2})

        GHG_off_land = GHG_files.query('base_name.str.contains("offland_commodity")')
        GHG_off_land = pd.concat([pd.read_csv(path) for path in GHG_off_land['path']], ignore_index=True).round({'Value (t CO2e)': 2})
        GHG_off_land['Value (t CO2e)'] = GHG_off_land['Total GHG Emissions (tCO2e)']
        GHG_off_land['Commodity'] = GHG_off_land['COMMODITY'].apply(lambda x: x[0].capitalize() + x[1:])
        GHG_off_land = GHG_off_land.drop(columns=['COMMODITY', 'Total GHG Emissions (tCO2e)'])
        GHG_off_land['Emission Source'] = GHG_off_land['Emission Source']\
            .replace({
                'CO2': 'Carbon Dioxide (CO2)',
                'CH4': 'Methane (CH4)',
                'N2O': 'Nitrous Oxide (N2O)'
            })
   
        GHG_land = pd.concat([GHG_ag, GHG_non_ag, GHG_ag_man, GHG_transition], axis=0).query('abs(`Value (t CO2e)`) > 0').reset_index(drop=True)
        GHG_land['Land-use type'] = GHG_land['Land-use'].apply(lu_group.set_index('Land-use')['Category'].to_dict().get)
        net_land = GHG_land.groupby('Year')[['Value (t CO2e)']].sum(numeric_only=True).reset_index()


        GHG_limit = GHG_files.query('base_name == "GHG_emissions"')
        GHG_limit = pd.concat([pd.read_csv(path) for path in GHG_limit['path']], ignore_index=True)
        GHG_limit = GHG_limit.query('Variable == "GHG_EMISSIONS_LIMIT_TCO2e"').copy()
        GHG_limit['Value (t CO2e)'] = GHG_limit['Emissions (t CO2e)']
        GHG_limit_wide = list(map(list,zip(GHG_limit['Year'],GHG_limit['Value (t CO2e)'])))
        
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
   

        # -------------------- GHG from individual emission sectors --------------------
        net_offland_AUS = GHG_off_land.groupby('Year')[['Value (t CO2e)']].sum(numeric_only=True).reset_index()
        net_offland_AUS_wide = net_offland_AUS[['Year','Value (t CO2e)']].values.tolist()
        
        net_land_AUS_wide = GHG_land\
            .groupby(['Year','Type'])[['Value (t CO2e)']]\
            .sum(numeric_only=True)\
            .reset_index()\
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
            'line'
        ]
        net_land_AUS_wide.loc[len(net_land_AUS_wide)] = [
            'GHG emission limit',
            GHG_limit_wide,
            'line'
        ]

 
        net_land_AUS_wide['name_order'] = net_land_AUS_wide['name'].apply(lambda x: order_GHG.index(x))
        net_land_AUS_wide = net_land_AUS_wide.sort_values('name_order').drop(columns=['name_order'])
        
        GHG_AUS = {'AUSTRALIA': json.loads(net_land_AUS_wide.to_json(orient='records'))}
        
        GHG_region = {}
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
                'line'
            ]

            df_reg['name_order'] = df_reg['name'].apply(lambda x: order_GHG.index(x))
            df_reg = df_reg.sort_values('name_order').drop(columns=['name_order'])
            GHG_region[region] = json.loads(df_reg.to_json(orient='records'))
            
            
        GHG_json = {**GHG_AUS,  **GHG_region}
        with open(f'{SAVE_DIR}/GHG_overview.json', 'w') as f:
            json.dump(GHG_json, f)
            
            
            
            
        # -------------------- GHG ranking --------------------
        GHG_rank_emission_region = GHG_land\
            .query('`Value (t CO2e)`>= 0')\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (t CO2e)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount() + 1)\
            .assign(Percent=lambda x: x['Value (t CO2e)'] / x.groupby('Year')['Value (t CO2e)'].transform('sum') * 100)\
            .assign(Type='GHG emissions')
        GHG_rank_emission_AUS = GHG_land\
            .query('`Value (t CO2e)`>= 0')\
            .groupby(['Year'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (t CO2e)'], ascending=[True, False])\
            .assign(Rank='N.A.', Percent=100, Type='GHG emissions', region='AUSTRALIA')

        GHG_rank_sequestration_region = GHG_land\
            .query('`Value (t CO2e)` < 0')\
            .assign(**{'Value (t CO2e)': lambda x: abs(x['Value (t CO2e)'])})\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (t CO2e)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount() + 1)\
            .assign(Percent=lambda x: x['Value (t CO2e)'] / x.groupby('Year')['Value (t CO2e)'].transform('sum') * 100)\
            .assign(Type='GHG sequestrations')
        GHG_rank_sequestration_AUS = GHG_land\
            .query('`Value (t CO2e)` < 0')\
            .assign(**{'Value (t CO2e)': lambda x: abs(x['Value (t CO2e)'])})\
            .groupby(['Year'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (t CO2e)'], ascending=[True, False])\
            .assign(Rank='N.A.', Area_type='Total', Percent=100, region='AUSTRALIA')\
            .assign(Type='GHG sequestrations')

        GHG_rank = pd.concat([
            GHG_rank_emission_region, 
            GHG_rank_emission_AUS,
            GHG_rank_sequestration_region, 
            GHG_rank_sequestration_AUS
            ], axis=0, ignore_index=True).reset_index(drop=True)
        GHG_rank = GHG_rank\
            .reset_index(drop=True)\
            .round({'Percent': 2, 'Value (t CO2e)':2})\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))\
            .set_index(['region', 'Year', 'Type'])\
            .reindex(
                index=pd.MultiIndex.from_product(
                    [GHG_rank['region'].unique(), GHG_rank['Year'].unique(), GHG_rank['Type'].unique()],
                    names=['region', 'Year', 'Type']), fill_value=None
            ).reset_index()\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))
     

        out_dict = {}
        for (region, e_type), df in GHG_rank.groupby(['region', 'Type']):
            if region not in out_dict:
                out_dict[region] = {}
            if e_type not in out_dict[region]:
                out_dict[region][e_type] = {}

            df = df.drop(columns='region')
            out_dict[region][e_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            out_dict[region][e_type]['Percent'] = df.set_index('Year')['Percent'].replace({np.nan: 0}).to_dict()
            out_dict[region][e_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            out_dict[region][e_type]['value'] = df.set_index('Year')['Value (t CO2e)'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else 0).to_dict()

        with open(f'{SAVE_DIR}/GHG_ranking.json', 'w') as f:
            json.dump(out_dict, f, indent=2)



        # -------------------- GHG emission for agricultural land-use --------------------
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
            
            
        group_cols = ['GHG Category', 'Land-use', 'Land-use type', 'Source', 'Water_supply']
        
        for idx, col in enumerate(group_cols):
            df_AUS = GHG_ag_emissions_long\
                .groupby(['Year', col])[['Value (t CO2e)']]\
                .sum()\
                .reset_index()
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (t CO2e)']]\
                .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name', 'data','region']
            df_AUS_wide['type'] = 'column'

            df_region = GHG_ag_emissions_long\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (t CO2e)']]\
                .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
                .reset_index()
            df_region_wide.columns = ['name', 'region', 'data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            
            if col == "Land-use":
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Water_supply':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
            elif col.lower() == 'commodity':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Agricultural Management Type':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            elif col == 'Source':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_GHG[x['name']], axis=1)
            
            
            out_dict = {}
            for region, df in df_wide.groupby('region'):
                df = df.drop(['region'], axis=1)
                out_dict[region] = df.to_dict(orient='records')
                
            with open(f'{SAVE_DIR}/GHG_split_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)

     
            
        # -------------------- GHG emission (off-land) by commodity --------------------
        group_cols = ['Emission Type', 'Emission Source', 'Commodity']

        for idx, col in enumerate(group_cols):
            df_AUS = GHG_off_land\
                .groupby(['Year', col])[['Value (t CO2e)']]\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (t CO2e)']]\
                .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name', 'data','region']
            df_AUS_wide['type'] = 'column'   
            
            out_dict = {}
            for region, df in df_AUS_wide.groupby('region'):
                df = df.drop(['region'], axis=1)
                out_dict[region] = df.to_dict(orient='records')
                
            with open(f'{SAVE_DIR}/GHG_split_off_land_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
    


        # -------------------- GHG abatement by Non-Agricultural sector --------------------
        Non_ag_reduction_long = GHG_land.query('Type == "Non-Agricultural land-use"').reset_index(drop=True)
        Non_ag_reduction_long['Value (t CO2e)'] = Non_ag_reduction_long['Value (t CO2e)'] * -1  # Convert from negative to positive
        group_cols = ['Land-use']
        for idx, col in enumerate(group_cols):
            df_AUS = Non_ag_reduction_long\
                .groupby(['Year', col])[['Value (t CO2e)']]\
                .sum()\
                .reset_index()
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (t CO2e)']]\
                .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name', 'data','region']
            df_AUS_wide['type'] = 'column'

            df_region = Non_ag_reduction_long\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (t CO2e)']]\
                .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
                .reset_index()
            df_region_wide.columns = ['name', 'region', 'data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            
            df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            
            out_dict = {}
            for region, df in df_wide.groupby('region'):
                df = df.drop(['region'], axis=1)
                out_dict[region] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/GHG_split_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)



        # -------------------- GHG reductions by Agricultural Managements --------------------
        Ag_man_sequestration_long = GHG_land.query('Type == "Agricultural Management"').reset_index(drop=True)
        Ag_man_sequestration_long['Value (t CO2e)'] = Ag_man_sequestration_long['Value (t CO2e)'] * -1  # Convert from negative to positive
        group_cols = ['Land-use', 'Land-use type', 'Agricultural Management Type', 'Water_supply']
        for idx, col in enumerate(group_cols):
            df_AUS = Ag_man_sequestration_long\
                .groupby(['Year', col])[['Value (t CO2e)']]\
                .sum()\
                .reset_index()
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (t CO2e)']]\
                .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name', 'data','region']
            df_AUS_wide['type'] = 'column'

            df_region = Ag_man_sequestration_long\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .round({'Value (t CO2e)': 2})
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (t CO2e)']]\
                .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
                .reset_index()
            df_region_wide.columns = ['name', 'region', 'data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            
            if col == "Land-use":
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Water_supply':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
            elif col.lower() == 'commodity':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Agricultural Management Type':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            
            
            out_dict = {}
            for region, df in df_wide.groupby('region'):
                df = df.drop(['region'], axis=1)
                out_dict[region] = df.to_dict(orient='records')
                
            with open(f'{SAVE_DIR}/GHG_split_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)

        


    ####################################################
    #                     5) Water                     #
    ####################################################
    
    water_files = files.query('category == "water"').reset_index(drop=True)

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
        columns={'Domestic Water Use (ML)': 'Value (ML)'})
    water_domestic_use['Value (ML)'] *= -1  # Domestic water use is negative, indicating a water loss (consumption)
    
    water_yield_limit = hist_and_public_wny_water_region[['Year','Region', 'Water Yield Limit (ML)']].rename(
        columns={'Water Yield Limit (ML)': 'Value (ML)'}
    )
    water_net_yield = hist_and_public_wny_water_region[['Year','Region', 'Water Net Yield (ML)']].rename(
        columns={'Water Net Yield (ML)': 'Value (ML)'}
    )


    # -------------------- Water yield overview for Australia --------------------
    water_inside_LUTO_sum = water_net_yield_water_region\
        .groupby(['Year','Type'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .round({'Value (ML)': 2})\
        .reset_index()
    water_inside_LUTO_sum_wide = water_inside_LUTO_sum\
        .groupby('Type')[['Year','Value (ML)']]\
        .apply(lambda x: list(map(list,zip(x['Year'],x['Value (ML)']))))\
        .reset_index()\
        .round({'Value (ML)': 2})
    water_inside_LUTO_sum_wide.columns = ['name','data']
    water_inside_LUTO_sum_wide['type'] = 'column'
    
        
    water_outside_LUTO_total = water_outside_LUTO\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .round({'Value (ML)': 2})\
        .reset_index()[['Year','Value (ML)']].values.tolist()
    water_CCI = water_climate_change_impact\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .round({'Value (ML)': 2})\
        .reset_index()[['Year','Value (ML)']].values.tolist()
    water_domestic = water_domestic_use\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .round({'Value (ML)': 2})\
        .reset_index()[['Year','Value (ML)']].values.tolist()
    water_net_yield_sum = water_net_yield\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .round({'Value (ML)': 2})\
        .reset_index()[['Year','Value (ML)']].values.tolist()
    water_limit = water_yield_limit\
        .groupby('Year')\
        .sum(numeric_only=True)\
        .round({'Value (ML)': 2})\
        .reset_index()[['Year','Value (ML)']].values.tolist()
  
    water_yield_df_AUS = water_inside_LUTO_sum_wide.copy()
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Outside LUTO Study Area', water_outside_LUTO_total,  'column']
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Climate Change Impact', water_CCI,  'column']
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Domestic Water Use', water_domestic,  'column']
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Water Net Yield', water_net_yield_sum, 'line']
    water_yield_df_AUS.loc[len(water_yield_df_AUS)] = ['Water Limit', water_limit, 'line']

    water_yield_df_AUS.to_json(f'{SAVE_DIR}/Water_overview_AUSTRALIA.json', orient='records')
    

    # -------------------- Water yield overview for Australia by landuse --------------------
    water_inside_LUTO_lu_sum = water_net_yield_water_region\
        .groupby(['Year','Landuse'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value (ML)': 2})\
            
    water_inside_LUTO_lu_sum_wide = water_inside_LUTO_lu_sum\
        .groupby(['Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()\
        .set_index('Landuse')\
        .reindex(LANDUSE_ALL_RENAMED)\
        .reset_index()\
        .round({'Value (ML)': 2})\
        
    water_inside_LUTO_lu_sum_wide.columns = ['name','data']
    water_inside_LUTO_lu_sum_wide['type'] = 'column'
    water_inside_LUTO_lu_sum_wide.to_json(f'{SAVE_DIR}/Water_overview_landuse.json', orient='records')


    # -------------------- Water yield overview for Australia by watershed region --------------------
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
            ['Water Net Yield', water_net_yield_sum_wide, 'line', None],
            ['Water Limit', water_limit_wide, 'line', 'black']],  columns=['name','data','type','color']
        )

        water_yield_region[reg_name] = water_df.to_dict(orient='records')

    with open(f'{SAVE_DIR}/Water_overview_by_watershed_region.json', 'w') as outfile:
        json.dump(water_yield_region, outfile)
        
        
        
    # -------------------- Water yield ranking --------------------
    water_ranking_type_region = water_net_yield_NRM_region\
        .groupby(['Year', 'region_NRM', 'Type'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Type', 'Value (ML)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'Type']).cumcount() + 1)\
        .assign(Percent=lambda x: x['Value (ML)'] / x.groupby(['Year', 'Type'])['Value (ML)'].transform('sum') * 100)
    water_ranking_type_AUS = water_net_yield_NRM_region\
        .groupby(['Year', 'Type'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Type', 'Value (ML)'], ascending=[True, True, False])\
        .assign(Rank='N.A.', Percent=100, region_NRM='AUSTRALIA')
        
        
    water_ranking_net_region = water_net_yield_NRM_region\
        .groupby(['Year', 'region_NRM'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (ML)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby('Year').cumcount() + 1)\
        .assign(Percent=lambda x: x['Value (ML)'] / x.groupby('Year')['Value (ML)'].transform('sum') * 100)\
        .assign(Type='Total')
    water_ranking_net_AUS = water_net_yield_NRM_region\
        .groupby(['Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (ML)'], ascending=[True, False])\
        .assign(Rank='N.A.', Percent=100, Type='Total', region_NRM='AUSTRALIA')
        
    
    water_ranking = pd.concat([
        water_ranking_type_region, 
        water_ranking_type_AUS, 
        water_ranking_net_region, 
        water_ranking_net_AUS], axis=0, ignore_index=True).reset_index(drop=True)
    water_ranking = water_ranking\
        .set_index(['region_NRM', 'Year', 'Type'])\
        .reindex(
            index=pd.MultiIndex.from_product(
                [water_ranking['region_NRM'].unique(), water_ranking['Year'].unique(), water_ranking['Type'].unique()],
                names=['region_NRM', 'Year', 'Type']), fill_value=None)\
        .reset_index()\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))\
        .round({'Percent': 2, 'Value (ML)': 2})

 
    out_dict = {}
    for (region, w_type), df in water_ranking.groupby(['region_NRM', 'Type']):
        if region not in out_dict:
            out_dict[region] = {}
        if w_type not in out_dict[region]:
            out_dict[region][w_type] = {}

        df = df.drop(columns='region_NRM')
        out_dict[region][w_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][w_type]['Percent'] = df.set_index('Year')['Percent'].replace({np.nan: 0}).to_dict()
        out_dict[region][w_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][w_type]['value'] = df.set_index('Year')['Value (ML)'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else 0).to_dict()

    with open(f'{SAVE_DIR}/Water_ranking.json', 'w') as f:
        json.dump(out_dict, f, indent=2)



    # -------------------- Water yield overview by NRM region --------------------
    group_cols = ['Landuse', 'Type']
    out_dict = {region: [] for region in water_net_yield_NRM_region['region_NRM'].unique()}
    out_dict['AUSTRALIA'] = water_yield_df_AUS.to_dict(orient='records')
    for idx, col in enumerate(group_cols):

        df_region = water_net_yield_NRM_region\
            .groupby(['Year', 'region_NRM', col])\
            .sum()\
            .reset_index()\
            .round({'Value (ML)': 2})
        df_region_wide = df_region.groupby([col, 'region_NRM'])[['Year','Value (ML)']]\
            .apply(lambda x: x[['Year', 'Value (ML)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region_NRM', 'data']
        df_region_wide['type'] = 'column'
        if col == 'Landuse':
            df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
            df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])
        
        
        for region, df in df_region_wide.groupby('region_NRM'):
            df = df.drop(['region_NRM'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Water_overview_MRN_region_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)



    # -------------------- Water yield for agricultural landuse by NRM region --------------------
    water_ag = water_net_yield_NRM_region.query('Type == "Agricultural Landuse"').copy()
    group_cols = ['Landuse', 'Water Supply']
    
    for idx, col in enumerate(group_cols):

        df_region = water_ag\
            .groupby(['Year', 'region_NRM', col])\
            .sum()\
            .reset_index()\
            .round({'Value (ML)': 2})
        df_region_wide = df_region.groupby([col, 'region_NRM'])[['Year','Value (ML)']]\
            .apply(lambda x: x[['Year', 'Value (ML)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region_NRM', 'data']
        df_region_wide['type'] = 'column'
        
        if col == 'Landuse':
            df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
            df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water Supply':
            df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
        
        out_dict = {}
        for region, df in df_region_wide.groupby('region_NRM'):
            df = df.drop(['region_NRM'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Water_split_Ag_MRN_region_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
            
            
    # -------------------- Water yield for agricultural management by NRM region --------------------
    water_am = water_net_yield_NRM_region.query('Type == "Agricultural Management"').copy()
    group_cols = ['Water Supply', 'Landuse', 'Agri-Management']
    
    for idx, col in enumerate(group_cols):

        df_region = water_am\
            .groupby(['Year', 'region_NRM', col])\
            .sum()\
            .reset_index()\
            .round({'Value (ML)': 2})
        df_region_wide = df_region.groupby([col, 'region_NRM'])[['Year','Value (ML)']]\
            .apply(lambda x: x[['Year', 'Value (ML)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region_NRM', 'data']
        df_region_wide['type'] = 'column'
        
        if col == 'Landuse':
            df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
            df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water Supply':
            df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
        elif col.lower() == 'Agri-Management':
            df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: RENAME_AM_NON_AG.index(x))
            df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])
        
        out_dict = {}
        for region, df in df_region_wide.groupby('region_NRM'):
            df = df.drop(['region_NRM'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Water_split_Am_MRN_region_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
            
            
    # -------------------- Water yield for non-agricultural landuse by NRM region --------------------
    water_nonag = water_net_yield_NRM_region.query('Type == "Non-Agricultural Landuse"').copy()
    group_cols = ['Landuse']
    
    for idx, col in enumerate(group_cols):

        df_region = water_nonag\
            .groupby(['Year', 'region_NRM', col])\
            .sum()\
            .reset_index()\
            .round({'Value (ML)': 2})
        df_region_wide = df_region.groupby([col, 'region_NRM'])[['Year','Value (ML)']]\
            .apply(lambda x: x[['Year', 'Value (ML)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region_NRM', 'data']
        df_region_wide['type'] = 'column'
        
        if col == 'Landuse':
            df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_region_wide.groupby('region_NRM'):
            df = df.drop(['region_NRM'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/Water_split_NonAg_MRN_region_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
        



    #########################################################
    #                   6) Biodiversity                     #
    #########################################################

    filter_str = '''
        category == "biodiversity"
       
        and base_name == "biodiversity_overall_priority_scores"
    '''.strip().replace('\n','')
    
    bio_paths = files.query(filter_str).reset_index(drop=True)
    bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])\
        .replace(RENAME_AM_NON_AG)\
        .rename(columns={'Contribution Relative to Base Year Level (%)': 'Value (%)'})\
        .query('`Value (%)` > 1e-6')\
        .round({'Value (%)': 6})
        
        
        
    # ---------------- Biodiversity ranking ----------------
    bio_rank_type_region = bio_df\
        .query('`Value (%)`>= 0')\
        .groupby(['Year', 'region', 'Type'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Type', 'Value (%)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'Type']).cumcount() + 1)\
        .assign(Percent=lambda x: x['Value (%)'] / x.groupby(['Year', 'Type'])['Value (%)'].transform('sum') * 100)\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))\
        .round({'Percent': 2, 'Area Weighted Score (ha)': 2})
    bio_rank_type_AUS = bio_df\
        .query('`Value (%)`>= 0')\
        .groupby(['Year', 'Type'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Type', 'Value (%)'], ascending=[True, True, False])\
        .assign(Rank='N.A.', Percent=100, region='AUSTRALIA')\
        .round({'Percent': 2, 'Area Weighted Score (ha)': 2})
        
    bio_rank_total_region = bio_df\
        .query('`Value (%)`>= 0')\
        .groupby(['Year', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount() + 1)\
        .assign(Percent=lambda x: x['Value (%)'] / x.groupby(['Year'])['Value (%)'].transform('sum') * 100)\
        .assign(Type='Total')\
        .round({'Percent': 2, 'Area Weighted Score (ha)': 2})
    bio_rank_total_AUS = bio_df\
        .query('`Value (%)`>= 0')\
        .groupby(['Year'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
        .assign(Rank='N.A.', Percent=100, Type='Total', region='AUSTRALIA')\
        .round({'Percent': 2, 'Area Weighted Score (ha)': 2})


    bio_rank = pd.concat([
        bio_rank_type_region,
        bio_rank_total_region,
        bio_rank_type_AUS,
        bio_rank_total_AUS], axis=0, ignore_index=True).reset_index(drop=True)
    bio_rank = bio_rank\
        .set_index(['region', 'Year', 'Type'])\
        .reindex(
            index=pd.MultiIndex.from_product(
                [bio_rank['region'].unique(), bio_rank['Year'].unique(), bio_rank['Type'].unique()],
                names=['region', 'Year', 'Type']),fill_value=None)\
        .reset_index()\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))

            
    out_dict = {}
    for (region, b_type), df in bio_rank.groupby(['region', 'Type']):
        if region not in out_dict:
            out_dict[region] = {}
        if b_type not in out_dict[region]:
            out_dict[region][b_type] = {}

        df = df.drop(columns='region')
        out_dict[region][b_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][b_type]['Percent'] = df.set_index('Year')['Percent'].replace({np.nan: 0}).to_dict()
        out_dict[region][b_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][b_type]['value'] = df.set_index('Year')['Area Weighted Score (ha)'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else 0).to_dict()

        
    with open(f'{SAVE_DIR}/Biodiversity_ranking.json', 'w') as f:
        json.dump(out_dict, f, indent=2)



    # ---------------- Biodiversity quality overview  ----------------
    group_cols = ['Type']
    for idx, col in enumerate(group_cols):
        df_AUS = bio_df\
            .groupby(['Year', col])[['Value (%)']]\
            .sum()\
            .reset_index()
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = bio_df\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .query('`Value (%)` > 0')
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region', 'data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Landuse":
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water_supply':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Agricultural Management Type':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
        
        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        with open(f'{SAVE_DIR}/BIO_quality_overview_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
    
    
    
    # ---------------- Biodiversity quality by Agricultural Landuse  ----------------
    bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()
    group_cols = ['Landuse']
    for idx, col in enumerate(group_cols):
        df_AUS = bio_df_ag\
            .groupby(['Year', col])[['Value (%)']]\
            .sum()\
            .reset_index()
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = bio_df_ag\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .query('`Value (%)` > 0')
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region', 'data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Landuse":
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water_supply':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Agricultural Management Type':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
        
        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/BIO_quality_split_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
            
    # ---------------- Biodiversity quality by Agricultural Management  ----------------
    bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()
    group_cols = ['Landuse','Agri-Management']
    for idx, col in enumerate(group_cols):
        df_AUS = bio_df_am\
            .groupby(['Year', col])[['Value (%)']]\
            .sum()\
            .reset_index()
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = bio_df_am\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .query('`Value (%)` > 0')
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region', 'data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Landuse":
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water_supply':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Agricultural Management Type':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
        
        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/BIO_quality_split_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)


    # ---------------- Biodiversity quality by Non-Agricultural  ----------------
    bio_df_nonag = bio_df.query('Type == "Non-Agricultural land-use"').copy()
    group_cols = ['Landuse']
    for idx, col in enumerate(group_cols):
        df_AUS = bio_df_nonag\
            .groupby(['Year', col])[['Value (%)']]\
            .sum()\
            .reset_index()
        df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()\
            .assign(region='AUSTRALIA')
        df_AUS_wide.columns = ['name', 'data','region']
        df_AUS_wide['type'] = 'column'

        df_region = bio_df_nonag\
            .groupby(['Year', 'region', col])\
            .sum()\
            .reset_index()\
            .query('`Value (%)` > 0')
        df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_region_wide.columns = ['name', 'region', 'data']
        df_region_wide['type'] = 'column'
        
        
        df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
        if col == "Landuse":
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Water_supply':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
        elif col.lower() == 'commodity':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
            df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Agricultural Management Type':
            df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
        
        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')
            
        with open(f'{SAVE_DIR}/BIO_quality_split_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
            json.dump(out_dict, f)
    
    
        
    
    if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':

        filter_str = '''
            category == "biodiversity" 
            
            and base_name == "biodiversity_GBF2_priority_scores"
        '''.strip('').replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .query('`Value (%)` > 1e-6')\
            .round(6)
        

        # ---------------- (GBF2) overview  ----------------
        bio_df_target = bio_df.groupby(['Year'])[['Priority Target (%)']].agg('first').reset_index()
        bio_df_target = bio_df_target[['Year','Priority Target (%)']].values.tolist()

        group_cols = ['Type']
        for idx, col in enumerate(group_cols):
            df_AUS = bio_df\
                .groupby(['Year', col])[['Value (%)']]\
                .sum()\
                .reset_index()
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name', 'data','region']
            df_AUS_wide['type'] = 'column'
            df_AUS_wide.loc[len(df_AUS_wide)] = ['Target (%)', bio_df_target, 'line', None]

            df_region = bio_df\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .query('`Value (%)` > 0')
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                .reset_index()
            df_region_wide.columns = ['name', 'region', 'data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            
            if col == "Landuse":
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Water_supply':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
            elif col.lower() == 'commodity':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Agricultural Management Type':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            
            
            out_dict = {}
            for region, df in df_wide.groupby('region'):
                df = df.drop(['region'], axis=1)
                out_dict[region] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF2_overview_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)


        # ---------------- (GBF2) Agricultural Landuse  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()

        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            df_AUS = bio_df_ag\
                .groupby(['Year', col])[['Value (%)']]\
                .sum()\
                .reset_index()
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name', 'data','region']
            df_AUS_wide['type'] = 'column'

            df_region = bio_df_ag\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .query('`Value (%)` > 0')
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                .reset_index()
            df_region_wide.columns = ['name', 'region', 'data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            
            if col == "Landuse":
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Water_supply':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
            elif col.lower() == 'commodity':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Agricultural Management Type':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            
            
            out_dict = {}
            for region, df in df_wide.groupby('region'):
                df = df.drop(['region'], axis=1)
                out_dict[region] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF2_split_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
                
        # ---------------- (GBF2) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        group_cols = ['Landuse', 'Agri-Management']
        for idx, col in enumerate(group_cols):
            df_AUS = bio_df_am\
                .groupby(['Year', col])[['Value (%)']]\
                .sum()\
                .reset_index()
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name', 'data','region']
            df_AUS_wide['type'] = 'column'

            df_region = bio_df_am\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .query('`Value (%)` > 0')
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                .reset_index()
            df_region_wide.columns = ['name', 'region', 'data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            
            if col == "Landuse":
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Water_supply':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
            elif col.lower() == 'commodity':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Agricultural Management Type':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            
            
            out_dict = {}
            for region, df in df_wide.groupby('region'):
                df = df.drop(['region'], axis=1)
                out_dict[region] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF2_split_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
                
                
        # ---------------- (GBF2) Agricultural Management  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural land-use"').copy()

        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            df_AUS = bio_df_nonag\
                .groupby(['Year', col])[['Value (%)']]\
                .sum()\
                .reset_index()
            df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                .reset_index()\
                .assign(region='AUSTRALIA')
            df_AUS_wide.columns = ['name', 'data','region']
            df_AUS_wide['type'] = 'column'

            df_region = bio_df_nonag\
                .groupby(['Year', 'region', col])\
                .sum()\
                .reset_index()\
                .query('`Value (%)` > 0')
            df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                .reset_index()
            df_region_wide.columns = ['name', 'region', 'data']
            df_region_wide['type'] = 'column'
            
            
            df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
            
            if col == "Landuse":
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Water_supply':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
            elif col.lower() == 'commodity':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            elif col == 'Agricultural Management Type':
                df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
            
            
            out_dict = {}
            for region, df in df_wide.groupby('region'):
                df = df.drop(['region'], axis=1)
                out_dict[region] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF2_split_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
        
        
 
        
        
            
            
    
    if settings.BIODIVERSITY_TARGET_GBF_3 != 'off':
        filter_str = '''
            category == "biodiversity" 
            
            and base_name.str.contains("biodiversity_GBF3")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path, low_memory=False) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Vegetation Group': 'species'})\
            .query('`Value (%)` > 1e-6')\
            .round(6)
        
        
        # ---------------- (GBF3) Overview  ----------------
        bio_df_target = bio_df.groupby(['Year', 'species'])[['Target_by_Percent']].agg('first').reset_index()

        group_cols = ['Type']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df['species'].unique():
                target_species = bio_df_target.query('species == @species')[['Year', 'Target_by_Percent']].values.tolist()
                scores_species = bio_df.query('species == @species')
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'
                df_AUS_wide.loc[len(df_AUS_wide)] = ['Target (%)', target_species, 'AUSTRALIA',  'line']

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF3_overview_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
                
                
                
        # ---------------- (GBF3) Agricultural Landuse  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()
        
        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_ag['species'].unique():
                scores_species = bio_df_ag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF3_split_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)


        # ---------------- (GBF3) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        group_cols = ['Landuse', 'Agri-Management']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_am['species'].unique():
                scores_species = bio_df_am.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF3_split_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)


        # ---------------- (GBF3) Non-agricultural management  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural land-use"').copy()

        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_nonag['species'].unique():
                scores_species = bio_df_nonag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF3_split_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
            
            
            
            
    
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES == 'on':
        
        # -------------------- Get biodiversity dataframe --------------------
        filter_str = '''
            category == "biodiversity" 
            
            and base_name.str.contains("biodiversity_GBF4_SNES_scores")
        '''.strip().replace('\n', '')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .query('`Value (%)` > 1e-6')\
            .round(6)
        
        # ---------------- (GBF4 SNES) Overview  ----------------
        bio_df_target = bio_df.groupby(['Year', 'species'])[['Target by Percent (%)']].agg('first').reset_index()

        group_cols = ['Type']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df['species'].unique():
                target_species = bio_df_target.query('species == @species')[['Year', 'Target by Percent (%)']].values.tolist()
                scores_species = bio_df.query('species == @species')                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'
                df_AUS_wide.loc[len(df_AUS_wide)] = ['Target (%)', target_species, 'AUSTRALIA',  'line']

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF4_SNES_overview_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
                
                
                
        # ---------------- (GBF4 SNES) Agricultural Landuse  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()
        
        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_ag['species'].unique():
                scores_species = bio_df_ag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF4_SNES_split_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)


        # ---------------- (GBF4 SNES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        group_cols = ['Landuse', 'Agri-Management']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_am['species'].unique():
                scores_species = bio_df_am.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF4_SNES_split_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)

        # ---------------- (GBF4 SNES) Non-agricultural management  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural land-use"').copy()

        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_nonag['species'].unique():
                scores_species = bio_df_nonag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF4_SNES_split_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
            
            
            
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES == 'on':
        # -------------------- Get biodiversity dataframe --------------------
        filter_str = '''
            category == "biodiversity" 
            
            and base_name.str.contains("biodiversity_GBF4_ECNES_scores")
        '''.strip().replace('\n', '')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .query('`Value (%)` > 1e-6')\
            .round(6)
        
        
        # ---------------- (GBF4 ECNES) Overview  ----------------
        bio_df_target = bio_df.groupby(['Year', 'species'])[['Target by Percent (%)']].agg('first').reset_index()

        group_cols = ['Type']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df['species'].unique():
                target_species = bio_df_target.query('species == @species')[['Year', 'Target by Percent (%)']].values.tolist()
                scores_species = bio_df.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'
                df_AUS_wide.loc[len(df_AUS_wide)] = ['Target (%)', target_species, 'AUSTRALIA',  'line']

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF4_ECNES_overview_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
                
                
                
        # ---------------- (GBF4 ECNES) Agricultural Landuse  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()
        
        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_ag['species'].unique():
                scores_species = bio_df_ag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF4_ECNES_split_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)


        # ---------------- (GBF4 ECNES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        group_cols = ['Landuse', 'Agri-Management']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_am['species'].unique():
                scores_species = bio_df_am.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF4_ECNES_split_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)

        # ---------------- (GBF4 ECNES) Non-agricultural management  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural land-use"').copy()

        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_nonag['species'].unique():
                scores_species = bio_df_nonag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF4_ECNES_split_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
        
        
        

    if settings.BIODIVERSITY_TARGET_GBF_8 == 'on':
        
        filter_str = '''
            category == "biodiversity" 
            
            and base_name.str.contains("biodiversity_GBF8_species_scores")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Species':'species'})\
            .query('`Value (%)` > 1e-6')\
            .round(6)
        
        # ---------------- (GBF8 SPECIES) Overview  ----------------
        bio_df_target = bio_df.groupby(['Year', 'species'])[['Target_by_Percent']].agg('first').reset_index()

        group_cols = ['Type']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df['species'].unique():
                target_species = bio_df_target.query('species == @species')[['Year', 'Target_by_Percent']].values.tolist()
                scores_species = bio_df.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'
                df_AUS_wide.loc[len(df_AUS_wide)] = ['Target (%)', target_species, 'AUSTRALIA',  'line']

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF8_SPECIES_overview_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
                
                
                
        # ---------------- (GBF8 SPECIES) Agricultural Landuse  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()
        
        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_ag['species'].unique():
                scores_species = bio_df_ag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF8_SPECIES_split_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)


        # ---------------- (GBF8 SPECIES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        group_cols = ['Landuse', 'Agri-Management']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_am['species'].unique():
                scores_species = bio_df_am.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF8_SPECIES_split_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)

        # ---------------- (GBF8 SPECIES) Non-agricultural management  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural land-use"').copy()

        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_nonag['species'].unique():
                scores_species = bio_df_nonag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF8_SPECIES_split_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
        
        
        
        
        # -------------------- Get biodiversity dataframe --------------------
        filter_str = '''
            category == "biodiversity" 
            
            and base_name.str.contains("biodiversity_GBF8_groups_scores")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Group':'species'})\
            .query('`Value (%)` > 1e-6')\
            .round(6)

        # ---------------- (GBF8 GROUP) Overview  ----------------
        group_cols = ['Type']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df['species'].unique():
                scores_species = bio_df.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF8_GROUP_overview_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
                
                
                
        # ---------------- (GBF8 GROUP) Agricultural Landuse  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()
        
        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_ag['species'].unique():
                scores_species = bio_df_ag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF8_GROUP_split_Ag_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)


        # ---------------- (GBF8 GROUP) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        group_cols = ['Landuse', 'Agri-Management']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_am['species'].unique():
                scores_species = bio_df_am.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF8_GROUP_split_Am_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)

        # ---------------- (GBF8 GROUP) Non-agricultural management  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural land-use"').copy()

        group_cols = ['Landuse']
        for idx, col in enumerate(group_cols):
            out_dict = {region: {} for region in bio_df['region'].unique()}
            out_dict['AUSTRALIA'] = {}
            for species in bio_df_nonag['species'].unique():
                scores_species = bio_df_nonag.query('species == @species')
                
                
                df_AUS = scores_species\
                    .groupby(['Year', col])[['Value (%)']]\
                    .sum()\
                    .reset_index()
                df_AUS_wide = df_AUS.groupby([col])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()\
                    .assign(region='AUSTRALIA')
                df_AUS_wide.columns = ['name', 'data','region']
                df_AUS_wide['type'] = 'column'

                df_region = scores_species\
                    .groupby(['Year', 'region', col])\
                    .sum()\
                    .reset_index()\
                    .query('`Value (%)` > 1e-6')
                df_region_wide = df_region.groupby([col, 'region'])[['Year','Value (%)']]\
                    .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
                    .reset_index()
                df_region_wide.columns = ['name', 'region', 'data']
                df_region_wide['type'] = 'column'
                
                
                df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
                
                if col == "Landuse":
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Water_supply':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_LM[x['name']], axis=1)
                elif col.lower() == 'commodity':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_COMMODITIES[x['name']], axis=1)
                    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
                    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                elif col == 'Agricultural Management Type':
                    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

                for region, df in df_wide.groupby('region'):
                    df = df.drop(['region'], axis=1)
                    out_dict[region][species] = df.to_dict(orient='records')

            with open(f'{SAVE_DIR}/BIO_GBF8_GROUP_split_NonAg_{idx+1}_{col.replace(" ", "_")}.json', 'w') as f:
                json.dump(out_dict, f)
                
  

    
    
    #########################################################
    #                         7) Maps                       #
    #########################################################
    map_files = files.query('base_ext == ".html"')
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
    # Supporting information               
    #########################################################       
    with open(f'{raw_data_dir}/model_run_settings.txt', 'r', encoding='utf-8') as src_file:
        settings_dict = {i.split(':')[0].strip(): ''.join(i.split(':')[1:]).strip() for i in src_file.readlines()}
        settings_dict = [{'parameter': k, 'val': v} for k, v in settings_dict.items()]
        
    with open(f'{settings.OUTPUT_DIR}/RES_{settings.RESFACTOR}_mem_log.txt', 'r', encoding='utf-8') as src_file:
        mem_logs = src_file.readlines()
        mem_logs = [i.split('\t') for i in mem_logs]
        mem_logs = [{'time': i[0], 'mem (GB)': i[1].strip()} for i in mem_logs]
        mem_logs_df = pd.DataFrame(mem_logs)
        mem_logs_df['time'] = pd.to_datetime(mem_logs_df['time'], format='%Y-%m-%d %H:%M:%S')
        mem_logs_df['time'] = mem_logs_df['time'].astype('int64') // 10**6  # convert to milliseconds
        mem_logs_df['mem (GB)'] = mem_logs_df['mem (GB)'].astype(float)
        mem_logs_obj = [{
            'name': f'Memory Usage (RES {settings.RESFACTOR})',
            'data': mem_logs_df.values.tolist()
        }]

    supporting = {
        'model_run_settings': settings_dict,
        'years': years,
        'colors': COLORS,
        'colors_ranking': COLORS_RANK,
        'mem_logs': mem_logs_obj,
        'RENAME_AM_NON_AG': RENAME_AM_NON_AG,
        'SPATIAL_MAP_DICT': SPATIAL_MAP_DICT
    }
    
    with open(f"{SAVE_DIR}/Supporting_info.json", 'w') as f:
        json.dump(supporting, f, indent=2)
    


    #########################################################
    # Report success info                     
    #########################################################

    print('Report data created successfully!\n')