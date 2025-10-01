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
import pandas as pd
import numpy as np
from luto import settings

from luto.economics.off_land_commodity import get_demand_df
from luto.tools.report.data_tools import get_all_files

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
    COMMIDOTY_GROUP,
    GHG_CATEGORY,
    GHG_NAMES,
    LANDUSE_ALL_RENAMED,
    LU_CROPS,
    LU_LVSTKS,
    LU_UNALLOW,
    RENAME_AM_NON_AG,
    RENAME_NON_AG,
)


def save_report_data(raw_data_dir:str):
    """
    Saves the report data in the specified directory.
    """
    # Set the save directory
    SAVE_DIR = f'{raw_data_dir}/DATA_REPORT/data'
    years = sorted(settings.SIM_YEARS)

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

        
    def format_with_suffix(x):
        if pd.isna(x) or x == 0:
            return "0"
        suffixes = ['', 'K', 'M', 'B', 'T']
        # Determine the appropriate suffix
        magnitude = 0
        while abs(x) >= 1000 and magnitude < len(suffixes)-1:
            magnitude += 1
            x /= 1000.0
        # Format with 2 significant digits
        if abs(x) < 100:
            formatted = f"{x:.2f}"
        else:
            formatted = f"{int(round(x))}"
        return f"{formatted} {suffixes[magnitude]}"
    
    
    # Land-use group and colors
    lu_group_raw = pd.read_csv('luto/tools/report/VUE_modules/assets/lu_group.csv')
    colors_lu_category = lu_group_raw.set_index('Category')['color_HEX'].to_dict()
    colors_lu_category.update({'Agri-Management': "#D5F100"})
    lu_group = lu_group_raw.set_index(['Category', 'color_HEX'])\
        .apply(lambda x: x.str.split(', ').explode())\
        .reset_index()
    
        

    ####################################################
    #                    1) Area Change                #
    ####################################################
    area_dvar_paths = files.query('category == "area"').reset_index(drop=True)
    
    ag_dvar_dfs = area_dvar_paths.query('base_name == "area_agricultural_landuse"').reset_index(drop=True)
    ag_dvar_area = pd.concat([pd.read_csv(path) for path in ag_dvar_dfs['path']], ignore_index=True)
    ag_dvar_area['Source'] = 'Agricultural Landuse'
    ag_dvar_area['Category'] = ag_dvar_area['Land-use'].apply(lu_group.set_index('Land-use')['Category'].to_dict().get)
    ag_dvar_area['Area (ha)'] = ag_dvar_area['Area (ha)'].round(2)
    ag_dvar_area_non_all = ag_dvar_area.query('Water_supply != "ALL"').copy()

    non_ag_dvar_dfs = area_dvar_paths.query('base_name == "area_non_agricultural_landuse"').reset_index(drop=True)
    non_ag_dvar_area = pd.concat([pd.read_csv(path) for path in non_ag_dvar_dfs['path'] if not pd.read_csv(path).empty], ignore_index=True)
    non_ag_dvar_area['Land-use'] = non_ag_dvar_area['Land-use'].replace(RENAME_NON_AG)
    non_ag_dvar_area['Category'] = non_ag_dvar_area['Land-use'].apply(lu_group.set_index('Land-use')['Category'].to_dict().get)
    non_ag_dvar_area['Source'] = 'Non-Agricultural Land-use'
    non_ag_dvar_area['Area (ha)'] = non_ag_dvar_area['Area (ha)'].round(2)

    am_dvar_dfs = area_dvar_paths.query('base_name == "area_agricultural_management"').reset_index(drop=True)
    am_dvar_area = pd.concat([pd.read_csv(path) for path in am_dvar_dfs['path'] if not pd.read_csv(path).empty], ignore_index=True)
    am_dvar_area = am_dvar_area.replace(RENAME_AM_NON_AG)
    am_dvar_area['Source'] = 'Agricultural Management'
    am_dvar_area['Area (ha)'] = am_dvar_area['Area (ha)'].round(2)
    am_dvar_area_non_all = am_dvar_area.query('Water_supply != "ALL" and Type != "ALL"').copy()
    

    # -------------------- Area ranking --------------------
    area_ranking_raw = pd.concat([ag_dvar_area_non_all, non_ag_dvar_area, am_dvar_area_non_all])
    
    area_ranking_type = area_ranking_raw\
        .groupby(['Year', 'region', 'Source'])[['Area (ha)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Source', 'Area (ha)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'Source']).cumcount())\
        .round({'Area (ha)': 2})
         
    area_ranking_total = area_ranking_raw\
        .groupby(['Year', 'region'])[["Area (ha)"]]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Area (ha)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount(), Source='Total')\
        .round({'Area (ha)': 2})
        
    area_ranking = pd.concat([area_ranking_type, area_ranking_total], ignore_index=True)\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))

        

    out_dict = {}
    for (region, source), df in area_ranking.groupby(['region', 'Source']):
        df = df.drop(['region'], axis=1)
        
        if region not in out_dict:
            out_dict[region] = {}
        if source not in out_dict[region]:
            out_dict[region][source] = {}

        out_dict[region][source]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['value'] = df.set_index('Year')['Area (ha)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Area_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        

    # -------------------- Area overview --------------------
    
    area_df = pd.concat([
        ag_dvar_area_non_all, 
        non_ag_dvar_area, 
        am_dvar_area_non_all.assign(**{'Land-use':'Agri-Management', 'Category':'Agri-Management'})
        ], ignore_index=True)
    
    group_cols = ['Land-use', 'Category', 'Source']
    for idx, col in enumerate(group_cols):
 
        df_region = area_df\
            .groupby(['Year', 'region', col])[['Area (ha)']]\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_wide = df_region.groupby([col, 'region'])[['Year','Area (ha)']]\
            .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region','data']
        df_wide['type'] = 'column'

        if col == "Land-use":
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LU[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Category':
            df_wide['color'] = df_wide['name'].apply(lambda x: colors_lu_category[x])
        elif col == 'Source':
            df_wide['name_order'] = df_wide['name'].apply(lambda x: ['Agricultural Management', 'Agricultural Landuse', 'Non-Agricultural Land-use'].index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'Area_overview_{idx+1}_{col.replace(" ", "_")}'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
    
    
    # -------------------- Area by Agricultural land --------------------
    df_wide = ag_dvar_area\
        .groupby(['region', 'Water_supply', 'Land-use'])[['Year','Area (ha)']]\
        .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
        .reset_index()

    df_wide.columns = ['region', 'water', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LU[x])

    out_dict = {}
    for (region, water), df in df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = []
        out_dict[region][water] = df.to_dict(orient='records')
        
    filename = 'Area_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Area by Agricultural Management Area (ha) Land use --------------------
    df_wide = am_dvar_area\
        .groupby(['region', 'Type', 'Water_supply', 'Land-use'])[['Year','Area (ha)']]\
        .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['region', '_type', 'water', 'name', 'data']
    df_wide['type'] = 'column'
    
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_LU[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, _type, water), df in df_wide.groupby(['region', '_type', 'water']):
        df = df.drop(['region', 'water', '_type'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if _type not in out_dict[region]:
            out_dict[region][_type] = {}
        if water not in out_dict[region][_type]:
            out_dict[region][_type][water] = {}
            
        out_dict[region][_type][water] = df.to_dict(orient='records')
        
    filename = f'Area_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
            
            
    # -------------------- Area by Non-Agricultural Land-use --------------------
    df_wide = non_ag_dvar_area\
        .groupby(['region', 'Land-use'])[['Year', 'Area (ha)']]\
        .apply(lambda x: x[['Year','Area (ha)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['region', 'name', 'data']
    df_wide['type'] = 'column'

    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_AM_NONAG[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
   
    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop('region', axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
    filename = 'Area_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
            
            


    # # -------------------- Transition areas (start-end) --------------------
    # transition_path = files.query('category =="transition_matrix"')
    # transition_df_region = pd.read_csv(transition_path['path'].values[0], index_col=0).reset_index() 
    # transition_df_region = transition_df_region.replace(RENAME_AM_NON_AG)

    # transition_df_AUS = transition_df_region.groupby(['From Land-use', 'To Land-use'])[['Area (ha)']].sum().reset_index()
    # transition_df_AUS['region'] = 'AUSTRALIA'

    # transition_df = pd.concat([transition_df_AUS, transition_df_region], ignore_index=True)


    # out_dict = {}
    # for (region, df) in transition_df.groupby('region'):
    #     out_dict[region] = {}
        
    #     transition_mat = df.pivot(index='From Land-use', columns='To Land-use', values='Area (ha)')
    #     transition_mat = transition_mat.reindex(index=AG_LANDUSE, columns=LANDUSE_ALL_RENAMED)
    #     transition_mat = transition_mat.fillna(0)
    #     total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
        
    #     transition_df_pct = transition_mat / total_area_from * 100
    #     transition_df_pct = transition_df_pct.fillna(0).replace([np.inf, -np.inf], 0)

    #     transition_mat['SUM'] = transition_mat.sum(axis=1)
    #     transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

    #     heat_area = transition_mat.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
    #     ).format('{:,.0f}')

    #     heat_pct = transition_df_pct.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         vmin=0,
    #         vmax=100
    #     ).format('{:,.2f}')

    #     heat_area_html = heat_area.to_html()
    #     heat_pct_html = heat_pct.to_html()

    #     # Replace '0.00' with '-' in the html
    #     heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)
    #     heat_pct_html = re.sub(r'(?<!\d)0.00(?!\d)', '-', heat_pct_html)

    #     out_dict[region]['area'] = heat_area_html
    #     out_dict[region]['pct'] = heat_pct_html

    # filename = 'Area_transition_start_end'
    # with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
    #     f.write(f'window["{filename}"] = ')
    #     json.dump(out_dict, f, separators=(',', ':'), indent=2)
    #     f.write(';\n')
        
        
    # # -------------------- Transition areas (year-to-year) --------------------
    # transition_path = files.query('base_name =="crosstab-lumap"')
    # transition_df_region = pd.concat([pd.read_csv(path) for path in transition_path['path']], ignore_index=True)
    # transition_df_region = transition_df_region.replace(RENAME_AM_NON_AG)

    # transition_df_AUS = transition_df_region.groupby(['Year', 'From land-use', 'To land-use'])[['Area (ha)']].sum().reset_index()
    # transition_df_AUS['region'] = 'AUSTRALIA'

    # transition_df = pd.concat([transition_df_AUS, transition_df_region], ignore_index=True)
    
    # out_dict = {region: {'area': {}, 'pct':{}} for region in transition_df['region'].unique()}
    # for (year, region), df in transition_df.groupby(['Year', 'region']):
        
    #     transition_mat = df.pivot(index='From land-use', columns='To land-use', values='Area (ha)')
    #     transition_mat = transition_mat.reindex(index=AG_LANDUSE, columns=LANDUSE_ALL_RENAMED)
    #     transition_mat = transition_mat.fillna(0)
    #     total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
        
    #     transition_df_pct = transition_mat / total_area_from * 100
    #     transition_df_pct = transition_df_pct.fillna(0).replace([np.inf, -np.inf], 0)

    #     transition_mat['SUM'] = transition_mat.sum(axis=1)
    #     transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

    #     heat_area = transition_mat.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
    #     ).format('{:,.0f}')

    #     heat_pct = transition_df_pct.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         vmin=0,
    #         vmax=100
    #     ).format('{:,.2f}')

    #     heat_area_html = heat_area.to_html()
    #     heat_pct_html = heat_pct.to_html()

    #     # Replace '0.00' with '-' in the html
    #     heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)
    #     heat_pct_html = re.sub(r'(?<!\d)0.00(?!\d)', '-', heat_pct_html)

    #     out_dict[region]['area'][str(year)] = heat_area_html
    #     out_dict[region]['pct'][str(year)] = heat_pct_html
        
    # filename = 'Area_transition_year_to_year'
    # with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
    #     f.write(f'window["{filename}"] = ')
    #     json.dump(out_dict, f, separators=(',', ':'), indent=2)
    #     f.write(';\n')
        
        

    ####################################################
    #                   2) Production                  #
    ####################################################
    
    quantity_df = files.query('base_name == "quantity_production_t_separate"')
    quantity_df = pd.concat([pd.read_csv(path) for path in quantity_df['path']])\
        .assign(Commodity = lambda x: x['Commodity'].str.capitalize())\
        .replace({'Sheep lexp': 'Sheep live export', 'Beef lexp': 'Beef live export'})\
        .assign(group = lambda x: x['Commodity'].map(COMMIDOTY_GROUP.get))\
        .replace(RENAME_AM_NON_AG)\
        .query('Year.isin(@years) and abs(`Production (t/KL)`) > 1')\
        .round({'`Production (t/KL)`': 2})
        
    quantity_ag = quantity_df.query('Type == "Agricultural"').copy()    
    quantity_am = quantity_df.query('Type == "Agricultural Management"').copy()
    quantity_non_ag = quantity_df.query('Type == "Non-Agricultural"').copy()

    quantity_ag_non_all = quantity_ag.query('Water_supply != "ALL"').copy()
    quantity_am_non_all = quantity_am.query('Water_supply != "ALL" and am != "ALL"').copy()
    

    # -------------------- Demand --------------------
    
    DEMAND_DATA = get_demand_df()\
        .query('Year.isin(@years) and abs(`Quantity (tonnes, KL)`) > 1')\
        .replace({'Beef lexp': 'Beef live export', 'Sheep lexp': 'Sheep live export'})\
        .set_index(['Commodity', 'Type', 'Year'])\
        .reindex(COMMODITIES_ALL, level=0)\
        .reset_index()\
        .replace(RENAME_AM_NON_AG)\
        .assign(group = lambda x: x['Commodity'].map(COMMIDOTY_GROUP.get))
    
    # Convert imports to negative values, making it below zero in the stacked column chart
    DEMAND_DATA_long = DEMAND_DATA.query('Type != "Production" ')
    DEMAND_DATA_long.loc[DEMAND_DATA_long['Type'] == 'Imports', 'Quantity (tonnes, KL)'] *= -1
    
    DEMAND_target = DEMAND_DATA.query('Type == "Production"')




    # -------------------- Ranking --------------------
    
    quantity_rank = pd.concat([quantity_ag_non_all, quantity_non_ag, quantity_am_non_all])\
        .groupby(['Year', 'region', 'group'])[['Production (t/KL)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'group', 'Production (t/KL)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'group']).cumcount())\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))\
        .assign(Year=lambda x: x['Year'].astype(int))\
        .round({'Production (t/KL)': 2})

    out_dict = {}
    for (region, group), df in quantity_rank.groupby(['region', 'group']):
        df = df.drop(['region'], axis=1)
        
        if region not in out_dict:
            out_dict[region] = {}
        if group not in out_dict[region]:
            out_dict[region][group] = {}

        out_dict[region][group]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][group]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][group]['value'] = df.set_index('Year')['Production (t/KL)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Production_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    

    # -------------------- Overview --------------------
    
    # sum
    demand_type_wide = DEMAND_DATA_long\
        .groupby(['Year', 'Type'])[['Quantity (tonnes, KL)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Quantity (tonnes, KL)': 2})\
        .groupby(['Type'])[['Year', 'Quantity (tonnes, KL)']]\
        .apply(lambda x: x[['Year', 'Quantity (tonnes, KL)']].values.tolist())\
        .reset_index()
    demand_type_wide.columns = ['name', 'data']
    demand_type_wide['type'] = 'column'
        
    out_dict = {'AUSTRALIA': demand_type_wide.to_dict(orient='records')}
        
    filename = 'Production_overview_demand_type'
    with open(fr'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    # seperate plot data
    for _type in ['Domestic', 'Exports', 'Imports', 'Feed']:
        demand_group = DEMAND_DATA_long\
            .query('Type == @_type')\
            .groupby(['Year', 'Type', 'group'])[['Quantity (tonnes, KL)']]\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Quantity (tonnes, KL)': 2})\
            .groupby(['Type', 'group'])[['Year', 'Quantity (tonnes, KL)']]\
            .apply(lambda x: x[['Year', 'Quantity (tonnes, KL)']].values.tolist())\
            .reset_index()
        
        demand_group = demand_group.drop(columns=['Type'])
        demand_group.columns = ['name', 'data']
        demand_group['type'] = 'column'
        
        out_dict = {'AUSTRALIA': demand_group.to_dict(orient='records')}
        
        filename = f'Production_overview_{_type}'
        with open(fr'{SAVE_DIR}\{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


    # -------------------- Overview: Australia production achievement (%) --------------------
    quantity_diff = files.query('base_name == "quantity_comparison"').reset_index(drop=True)
    quantity_diff = pd.concat([pd.read_csv(path) for path in quantity_diff['path']], ignore_index=True)
    quantity_diff = quantity_diff.replace({'Sheep lexp': 'Sheep live export', 'Beef lexp': 'Beef live export'})
    quantity_diff = quantity_diff[['Year','Commodity','Prop_diff (%)']].rename(columns={'Prop_diff (%)': 'Demand Achievement (%)'})

    mask_AUS = quantity_diff.groupby('Commodity'
        )['Demand Achievement (%)'
        ].transform(lambda x: abs(round(x) - 100) > 0.01)
    quantity_diff_AUS = quantity_diff[mask_AUS].copy()
    quantity_diff_wide_AUS = quantity_diff_AUS\
        .groupby(['Commodity'])[['Year','Demand Achievement (%)']]\
        .apply(lambda x: x[['Year','Demand Achievement (%)']].values.tolist())\
        .reset_index()
        
    quantity_diff_wide_AUS['type'] = 'line'
    quantity_diff_wide_AUS.columns = ['name','data', 'type']

    quantity_diff_wide_AUS_data = {
        'AUSTRALIA': quantity_diff_wide_AUS.to_dict(orient='records')
    }
    filename = 'Production_overview_AUS_achive_percent'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(quantity_diff_wide_AUS_data, f, separators=(',', ':'), indent=2)
        f.write(';\n')    
    
    
    
    
    # -------------------- Commodity production for ag --------------------
    df_wide = quantity_ag_non_all\
        .groupby(['region', 'Water_supply', 'Commodity'])[['Year','Production (t/KL)']]\
        .apply(lambda x: x[['Year','Production (t/KL)']].values.tolist())\
        .reset_index()

    df_wide.columns = ['region', 'water', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water), df in df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water] = df.to_dict(orient='records')
        
    filename = f'Production_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # -------------------- Commodity production for ag-man --------------------
    df_wide = quantity_am_non_all\
        .groupby(['region', 'am', 'Water_supply', 'Commodity'])[['Year','Production (t/KL)']]\
        .apply(lambda x: x[['Year','Production (t/KL)']].values.tolist())\
        .reset_index()

    df_wide.columns = ['region', '_type', 'water', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, _type, water), df in df_wide.groupby(['region', '_type', 'water']):
        df = df.drop(['region', '_type', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if _type not in out_dict[region]:
            out_dict[region][_type] = {}
        if water not in out_dict[region][_type]:
            out_dict[region][_type][water] = {}
        out_dict[region][_type][water] = df.to_dict(orient='records')
        
    filename = f'Production_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # -------------------- Commodity production for non-ag --------------------
    df_wide = quantity_non_ag\
        .groupby(['region', 'Commodity'])[['Year','Production (t/KL)']]\
        .apply(lambda x: x[['Year','Production (t/KL)']].values.tolist())\
        .reset_index()

    df_wide.columns = ['region', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS_COMMODITIES[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
    filename = f'Production_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
            
            
    



    ####################################################
    #                  3) Economics                    #
    ####################################################
    
    # -------------------- Get the revenue and cost data --------------------
    revenue_ag_df = files.query('base_name == "revenue_ag"').reset_index(drop=True)
    revenue_ag_df = pd.concat([pd.read_csv(path) for path in revenue_ag_df['path']], ignore_index=True)
    revenue_ag_df = revenue_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural land-use (revenue)')
    revenue_ag_df_non_all = revenue_ag_df.query('Water_supply != "ALL" and Type != "ALL"')
    
    cost_ag_df = files.query('base_name == "cost_ag"').reset_index(drop=True)
    cost_ag_df = pd.concat([pd.read_csv(path) for path in cost_ag_df['path']], ignore_index=True)
    cost_ag_df = cost_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural land-use (cost)')
    cost_ag_df['Value ($)'] = cost_ag_df['Value ($)'] * -1          # Convert cost to negative value
    cost_ag_df_non_all = cost_ag_df.query('Water_supply != "ALL" and Type != "ALL"')
    
    revenue_am_df = files.query('base_name == "revenue_agricultural_management"').reset_index(drop=True)
    revenue_am_df = pd.concat([pd.read_csv(path) for path in revenue_am_df['path']], ignore_index=True)
    revenue_am_df = revenue_am_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural Management (revenue)')
    revenue_am_df_non_all = revenue_am_df.query('Water_supply != "ALL" and `Management Type` != "ALL"')
    
    cost_am_df = files.query('base_name == "cost_agricultural_management"').reset_index(drop=True)
    cost_am_df = pd.concat([pd.read_csv(path) for path in cost_am_df['path']], ignore_index=True)
    cost_am_df = cost_am_df.replace(RENAME_AM_NON_AG).assign(Source='Agricultural Management (cost)')
    cost_am_df['Value ($)'] = cost_am_df['Value ($)'] * -1          # Convert cost to negative value
    cost_am_df_non_all = cost_am_df.query('Water_supply != "ALL" and `Management Type` != "ALL"')

    revenue_non_ag_df = files.query('base_name == "revenue_non_ag"').reset_index(drop=True)
    revenue_non_ag_df = pd.concat([pd.read_csv(path) for path in revenue_non_ag_df['path']], ignore_index=True)
    revenue_non_ag_df = revenue_non_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Non-Agricultural Land-use (revenue)')

    cost_non_ag_df = files.query('base_name == "cost_non_ag"').reset_index(drop=True)
    cost_non_ag_df = pd.concat([pd.read_csv(path) for path in cost_non_ag_df['path']], ignore_index=True)
    cost_non_ag_df = cost_non_ag_df.replace(RENAME_AM_NON_AG).assign(Source='Non-Agricultural Land-use (cost)')
    cost_non_ag_df['Value ($)'] = cost_non_ag_df['Value ($)'] * -1  # Convert cost to negative value
    
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
        ).query('abs(`Value ($)`) > 1e-4'
        ).reset_index(drop=True)
        
    economics_df_non_all = pd.concat(
            [
                revenue_ag_df_non_all, 
                revenue_am_df_non_all, 
                revenue_non_ag_df,
                cost_ag_df_non_all, 
                cost_am_df_non_all, 
                cost_non_ag_df,
                cost_transition_ag2ag_df, 
                cost_transition_ag2non_ag_df,
                cost_transition_non_ag2ag_df
            ]
        ).round({'Value ($)': 2}
        ).query('abs(`Value ($)`) > 1e-4'
        ).reset_index(drop=True) 
        
    order = [
        'Agricultural land-use (revenue)', 
        'Agricultural Management (revenue)', 
        'Non-Agricultural Land-use (revenue)',
        'Agricultural land-use (cost)', 
        'Agricultural Management (cost)', 
        'Non-Agricultural Land-use (cost)',
        'Transition cost (Ag2Ag)',
        'Transition cost (Ag2Non-Ag)',
        'Transition cost (Non-Ag2Ag)',
        'Profit'
    ]


    # -------------------- Economic ranking --------------------
    revenue_df = pd.concat([revenue_ag_df_non_all, revenue_am_df_non_all, revenue_non_ag_df]
        ).groupby(['Year', 'region']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank=lambda x: x.groupby(['Year']).cumcount()
        ).assign(Source='Revenue')
    cost_df = pd.concat(
        [
            cost_ag_df_non_all, 
            cost_am_df_non_all, 
            cost_non_ag_df,
            cost_transition_ag2ag_df, 
            cost_transition_ag2non_ag_df, 
            cost_transition_non_ag2ag_df
        ]
        ).groupby(['Year', 'region']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).assign(**{'Value ($)': lambda x: abs(x['Value ($)'])}
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank=lambda x: x.groupby(['Year']).cumcount()
        ).assign(Source='Cost')
    profit_df = revenue_df.merge(
        cost_df, on=['Year', 'region'], suffixes=('_revenue', '_cost')
        ).assign(**{'Value ($)': lambda x: x['Value ($)_revenue'] - x['Value ($)_cost']}
        ).drop(columns=['Value ($)_revenue', 'Value ($)_cost']
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank=lambda x: x.groupby(['Year']).cumcount()
        ).assign(Source='Total')

    ranking_df = pd.concat([revenue_df, cost_df, profit_df]).assign(color= lambda x: x['Rank'].map(get_rank_color))
        

    out_dict = {}
    for (region, source), df in ranking_df.groupby(['region', 'Source']):
        if region not in out_dict:
            out_dict[region] = {}
        if not source in out_dict[region]:
            out_dict[region][source] = {}
        
        df = df.drop(columns='region')
        out_dict[region][source]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['value'] = df.set_index('Year')['Value ($)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Economics_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        

    # -------------------- Economy overview --------------------

    # Overview: sum of revenue, cost, and profit by region
    rev_cost_net_region = economics_df_non_all.groupby(['region', 'Source', 'Year']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index()
        
    dfs = []
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
        dfs.append(df_col)

    rev_cost_wide_json = pd.concat(dfs, ignore_index=True)
    rev_cost_wide_json['name_order'] = rev_cost_wide_json['name'].map({name: i for i, name in enumerate(order)})
    rev_cost_wide_json = rev_cost_wide_json.sort_values(['region', 'name_order']).drop(columns=['name_order']).reset_index(drop=True)


    out_dict = {}
    for region,df in rev_cost_wide_json.groupby('region'):
        df = df.drop(columns='region')
        df.columns = ['name','data','type']
        out_dict[region] = df.to_dict(orient='records')
        
    filename = 'Economics_overview_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    
    
    # Overview: ag cost/revenue by type
    economics_ag = pd.concat([revenue_ag_df_non_all, cost_ag_df_non_all])\
        .query('abs(`Value ($)`) > 1')\
        .groupby(['region', 'Type','Year'])['Value ($)']\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value ($)': 2})
  
    
    df_wide = economics_ag\
        .groupby(['region', 'Type'])[['Year', 'Value ($)']]\
        .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
        .reset_index()
        
    df_wide.columns = ['region', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_ECONOMY_TYPE[x['name']], axis=1)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: list(COLORS_ECONOMY_TYPE.keys()).index(x) if x in COLORS_ECONOMY_TYPE else -1)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])


    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop('region', axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
    filename = f'Economics_overview_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
    
    # Overview: ag-man cost/revenue by type
    economics_am = pd.concat(
        [
            revenue_am_df_non_all.assign(Rev_Cost='Revenue'), 
            cost_am_df_non_all.assign(Rev_Cost='Cost')
        ]
        ).query('abs(`Value ($)`) > 1'
        ).round({'Value ($)': 2}
        ).groupby(['region', 'Management Type', 'Rev_Cost', 'Year'])[['Value ($)']
        ].sum(
        ).reset_index()
    
    df_wide = economics_am.groupby(['region', 'Management Type', 'Rev_Cost'])[[ 'Year', 'Value ($)']]\
        .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
        .reset_index()
        
    df_wide.columns = ['region', 'name', 'Rev_Cost', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
    df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
    df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)
    df_wide.loc[df_wide['name'] == 'Early dry-season savanna burning', 'linkedTo'] = None

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region', 'Rev_Cost'], axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
    filename = f'Economics_overview_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Overview: non-ag cost/revenue by type
    economics_non_ag = pd.concat([
            revenue_non_ag_df.assign(Rev_Cost='Revenue'), 
            cost_non_ag_df.assign(Rev_Cost='Cost')]
        ).query('abs(`Value ($)`) > 1'
        ).round({'Value ($)': 2}
        ).groupby(['region', 'Land-use', 'Rev_Cost', 'Year'])[['Value ($)']
        ].sum(
        ).reset_index()
    
    df_wide = economics_non_ag.groupby(['region', 'Land-use', 'Rev_Cost'])[['Year','Value ($)']]\
        .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['region', 'name', 'Rev_Cost', 'data']
    df_wide['type'] = 'column'
    
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

    filename = f'Economics_overview_Non_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
 
 
 
 
    # -------------------- Economics for ag --------------------
    revenue_ag_df = revenue_ag_df.assign(Rev_Cost='Revenue')
    cost_ag_df = cost_ag_df.assign(Rev_Cost='Cost')

    economics_ag = pd.concat([revenue_ag_df, cost_ag_df]
        ).round({'Value ($)': 2}
        ).query('abs(`Value ($)`) > 1'
        ).reset_index(drop=True)

    df_wide = economics_ag\
        .groupby(['region', 'Type', 'Water_supply', 'Rev_Cost', 'Land-use'])[['Year','Value ($)']]\
        .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
        .reset_index()\
        .round({'Value ($)': 2})

    df_wide.columns = ['region', '_type', 'water', 'Rev_Cost', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
    df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
    df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)


    out_dict = {}
    for (region, _type, water), df in df_wide.groupby(['region', '_type', 'water']):
        df = df.drop(['region', '_type', 'water', 'Rev_Cost'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if _type not in out_dict[region]:
            out_dict[region][_type] = {}
        if water not in out_dict[region][_type]:
            out_dict[region][_type][water] = {}
        out_dict[region][_type][water] = df.to_dict(orient='records')
        
    filename = 'Economics_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    


    # -------------------- Economics for ag-management --------------------
    revenue_am_df = revenue_am_df.assign(Rev_Cost='Revenue')
    cost_am_df = cost_am_df.assign(Rev_Cost='Cost')

    economics_am = pd.concat([revenue_am_df, cost_am_df]
        ).round({'Value ($)': 2}
        ).query('abs(`Value ($)`) > 1'
        ).reset_index(drop=True)

    df_wide = economics_am\
        .groupby(['region', 'Management Type', 'Water_supply', 'Land-use', 'Rev_Cost'])[['Year', 'Value ($)']]\
        .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
        .reset_index()\
        .round({'Value ($)': 2})
  
    df_wide.columns = ['region', '_type', 'water', 'name', 'Rev_Cost', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    df_wide['id'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Revenue' else None, axis=1)
    df_wide['linkedTo'] = df_wide.apply(lambda x: x['name'] if x['Rev_Cost'] == 'Cost' else None, axis=1)
    df_wide.loc[df_wide['name'] == 'Early dry-season savanna burning', 'linkedTo'] = None

    
    out_dict = {}
    for (region,_type,water), df in df_wide.groupby(['region', '_type', 'water']):
        df = df.drop(['region', '_type', 'water','Rev_Cost'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if _type not in out_dict[region]:
            out_dict[region][_type] = {}
        if water not in out_dict[region][_type]:
            out_dict[region][_type][water] = {}
        out_dict[region][_type][water] = df.to_dict(orient='records')
        
    filename = f'Economics_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Economics for non-agriculture --------------------

    # This is the same as the "Economics_overview_Non_Ag" 




    # # -------------------- Transition cost for Ag2Ag --------------------
    # cost_transition_ag2ag_df['Value ($)'] = cost_transition_ag2ag_df['Value ($)'] * -1  # Convert from negative to positive
    # group_cols = ['Type', 'From land-use', 'To land-use']
    
    # for idx, col in enumerate(group_cols):
    #     df_AUS = cost_transition_ag2ag_df\
    #         .groupby(['Year', col])[['Value ($)']]\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()\
    #         .assign(region='AUSTRALIA')
    #     df_AUS_wide.columns = ['name', 'data','region']
    #     df_AUS_wide['type'] = 'column'

    #     df_region = cost_transition_ag2ag_df\
    #         .groupby(['Year', 'region', col])\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()
    #     df_region_wide.columns = ['name', 'region', 'data']
    #     df_region_wide['type'] = 'column'
        
        
    #     df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
    #     out_dict = {}
    #     for region, df in df_wide.groupby('region'):
    #         df = df.drop(['region'], axis=1)
    #         out_dict[region] = df.to_dict(orient='records')
            
    #     filename = f'Economics_transition_split_ag2ag_{idx+1}_{col.replace(" ", "_")}'
    #     with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
    #         f.write(f'window["{filename}"] = ')
    #         json.dump(out_dict, f, separators=(',', ':'), indent=2)
    #         f.write(';\n')
      

    # # -------------------- Transition cost matrix for Ag2Ag --------------------
    # cost_transition_ag2ag_trans_mat_AUS = cost_transition_ag2ag_df\
    #     .groupby(['Year','From land-use', 'To land-use'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})\
    #     .query('abs(`Value ($)`) > 1e-4')\
    #     .assign(region='AUSTRALIA')

    # cost_transition_ag2ag_trans_mat_region_df = cost_transition_ag2ag_df\
    #     .groupby(['Year','From land-use', 'To land-use', 'region'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})
        
        
    # cost_transition_ag2ag_trans_mat = pd.concat([
    #     cost_transition_ag2ag_trans_mat_AUS,
    #     cost_transition_ag2ag_trans_mat_region_df
    # ])


    # out_dict_area = {}
    # for (region,year),df in cost_transition_ag2ag_trans_mat.groupby(['region', 'Year']):
        
    #     out_dict_area.setdefault(region, {})
        
    #     transition_mat = df.pivot(index='From land-use', columns='To land-use', values='Value ($)')
    #     transition_mat = transition_mat.reindex(index=AG_LANDUSE, columns=AG_LANDUSE)
    #     transition_mat = transition_mat.fillna(0)
    #     total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
        
    #     transition_mat['SUM'] = transition_mat.sum(axis=1)
    #     transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

    #     heat_area = transition_mat.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
    #     ).format('{:,.0f}')

    #     heat_area_html = heat_area.to_html()
    #     heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)

    #     out_dict_area[region][str(year)] = rf'{heat_area_html}'

    # filename = 'Economics_transition_mat_ag2ag'
    # with open(f'{SAVE_DIR}/{filename}.js', 'w', encoding='utf-8') as f:
    #     f.write(f'window["{filename}"] = ')
    #     json.dump(out_dict_area, f, separators=(',', ':'), indent=2)
    #     f.write(';\n')






    # # -------------------- Transition cost for Ag2Non-Ag --------------------
    # cost_transition_ag2non_ag_df['Value ($)'] = cost_transition_ag2non_ag_df['Value ($)'] * -1  # Convert from negative to positive
    # group_cols = ['Cost type', 'From land-use', 'To land-use']
    
    # for idx, col in enumerate(group_cols):
    #     df_AUS = cost_transition_ag2non_ag_df\
    #         .groupby(['Year', col])[['Value ($)']]\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()\
    #         .assign(region='AUSTRALIA')
    #     df_AUS_wide.columns = ['name', 'data','region']
    #     df_AUS_wide['type'] = 'column'

    #     df_region = cost_transition_ag2non_ag_df\
    #         .groupby(['Year', 'region', col])\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()
    #     df_region_wide.columns = ['name', 'region', 'data']
    #     df_region_wide['type'] = 'column'
        
        
    #     df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
    #     out_dict = {}
    #     for region, df in df_wide.groupby('region'):
    #         df = df.drop(['region'], axis=1)
    #         out_dict[region] = df.to_dict(orient='records')
            
    #     filename = f'Economics_transition_split_Ag2NonAg_{idx+1}_{col.replace(" ", "_")}'
    #     with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
    #         f.write(f'window["{filename}"] = ')
    #         json.dump(out_dict, f, separators=(',', ':'), indent=2)
    #         f.write(';\n')
        
  
    # # -------------------- Transition cost matrix for Ag2Non-Ag --------------------
    # cost_transition_ag2nonag_trans_mat_AUS = cost_transition_ag2non_ag_df\
    #     .groupby(['Year','From land-use', 'To land-use'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})\
    #     .assign(region='AUSTRALIA')

    # cost_transition_ag2nonag_trans_mat_region_df = cost_transition_ag2non_ag_df\
    #     .groupby(['Year','From land-use', 'To land-use', 'region'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})
        
        
    # cost_transition_ag2nonag_trans_mat = pd.concat([
    #     cost_transition_ag2nonag_trans_mat_AUS,
    #     cost_transition_ag2nonag_trans_mat_region_df
    # ])


    # out_dict_area = {}
    # for (region,year),df in cost_transition_ag2nonag_trans_mat.groupby(['region', 'Year']):
        
    #     out_dict_area.setdefault(region, {})
        
    #     transition_mat = df.pivot(index='From land-use', columns='To land-use', values='Value ($)')
    #     transition_mat = transition_mat.reindex(index=AG_LANDUSE, columns=RENAME_NON_AG.values())
    #     transition_mat = transition_mat.fillna(0)
    #     total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
        
    #     transition_mat['SUM'] = transition_mat.sum(axis=1)
    #     transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

    #     heat_area = transition_mat.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
    #     ).format('{:,.0f}')

    #     heat_area_html = heat_area.to_html()
    #     heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)

    #     out_dict_area[region][str(year)] = rf'{heat_area_html}'

    # filename = 'Economics_transition_mat_ag2nonag'
    # with open(f'{SAVE_DIR}/{filename}.js', 'w', encoding='utf-8') as f:
    #     f.write(f'window["{filename}"] = ')
    #     json.dump(out_dict_area, f, separators=(',', ':'), indent=2)
    #     f.write(';\n')
    
    
    
    

    # # -------------------- Transition cost for Non-Ag to Ag --------------------
    # cost_transition_non_ag2ag_df['Value ($)'] = cost_transition_non_ag2ag_df['Value ($)'] * -1  # Convert from negative to positive
    # group_cols = ['Cost type', 'From land-use', 'To land-use']
    
    # for idx, col in enumerate(group_cols):
    #     df_AUS = cost_transition_non_ag2ag_df\
    #         .groupby(['Year', col])[['Value ($)']]\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()\
    #         .assign(region='AUSTRALIA')
    #     df_AUS_wide.columns = ['name', 'data','region']
    #     df_AUS_wide['type'] = 'column'

    #     df_region = cost_transition_non_ag2ag_df\
    #         .groupby(['Year', 'region', col])\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()
    #     df_region_wide.columns = ['name', 'region', 'data']
    #     df_region_wide['type'] = 'column'
        
        
    #     df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)
        
    #     out_dict = {}
    #     for region, df in df_wide.groupby('region'):
    #         df = df.drop(['region'], axis=1)
    #         out_dict[region] = df.to_dict(orient='records')
            
    #     filename = f'Economics_transition_split_NonAg2Ag_{idx+1}_{col.replace(" ", "_")}'
    #     with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
    #         f.write(f'window["{filename}"] = ')
    #         json.dump(out_dict, f, separators=(',', ':'), indent=2)
    #         f.write(';\n')
    
    
    
    # # -------------------- Transition cost matrix for Non-Ag to Ag --------------------
    # cost_transition_nonag2ag_trans_mat_AUS = cost_transition_non_ag2ag_df\
    #     .groupby(['Year','From land-use', 'To land-use'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})\
    #     .assign(region='AUSTRALIA')

    # cost_transition_nonag2ag_trans_mat_region_df = cost_transition_non_ag2ag_df\
    #     .groupby(['Year','From land-use', 'To land-use', 'region'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})
        
        
    # cost_transition_nonag2ag_trans_mat = pd.concat([
    #     cost_transition_nonag2ag_trans_mat_AUS,
    #     cost_transition_nonag2ag_trans_mat_region_df
    # ])


    # out_dict_area = {}
    # for (region,year),df in cost_transition_nonag2ag_trans_mat.groupby(['region', 'Year']):
        
    #     out_dict_area.setdefault(region, {})
        
    #     transition_mat = df.pivot(index='From land-use', columns='To land-use', values='Value ($)')
    #     transition_mat = transition_mat.reindex(index=RENAME_NON_AG.values(), columns=AG_LANDUSE)
    #     transition_mat = transition_mat.fillna(0)
    #     total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)
        
    #     transition_mat['SUM'] = transition_mat.sum(axis=1)
    #     transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

    #     heat_area = transition_mat.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
    #     ).format('{:,.0f}')

    #     heat_area_html = heat_area.to_html()
    #     heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)

    #     out_dict_area[region][str(year)] = rf'{heat_area_html}'

    # filename = 'Economics_transition_mat_nonag2ag'
    # with open(f'{SAVE_DIR}/{filename}.js', 'w', encoding='utf-8') as f:
    #     f.write(f'window["{filename}"] = ')
    #     json.dump(out_dict_area, f, separators=(',', ':'), indent=2)
    #     f.write(';\n')




    ####################################################
    #                       4) GHGs                    #
    ####################################################
    '''GHG is written to disk no matter if GHG_EMISSIONS_LIMITS is 'off' or 'on' '''

    filter_str = '''
    category == "GHG" 
    and base_name.str.contains("GHG_emissions") 
    '''.replace('\n', ' ').replace('  ', ' ')

    GHG_files = files.query(filter_str).reset_index(drop=True)

    GHG_ag = GHG_files.query('base_name.str.contains("agricultural_landuse")').reset_index(drop=True)
    GHG_ag = pd.concat([pd.read_csv(path) for path in GHG_ag['path']], ignore_index=True)
    GHG_ag = GHG_ag.replace(GHG_NAMES).round({'Value (t CO2e)': 2})
    GHG_ag_non_all = GHG_ag.query('Water_supply != "ALL" and Source != "ALL"').reset_index(drop=True)
    
    GHG_non_ag = GHG_files.query('base_name.str.contains("no_ag_reduction")').reset_index(drop=True)
    GHG_non_ag = pd.concat([pd.read_csv(path) for path in GHG_non_ag['path'] if not pd.read_csv(path).empty], ignore_index=True)
    GHG_non_ag = GHG_non_ag.replace(RENAME_AM_NON_AG).round({'Value (t CO2e)': 2})
    
    GHG_ag_man = GHG_files.query('base_name.str.contains("agricultural_management")').reset_index(drop=True)
    GHG_ag_man = pd.concat([pd.read_csv(path) for path in GHG_ag_man['path'] if not pd.read_csv(path).empty], ignore_index=True)
    GHG_ag_man = GHG_ag_man.replace(RENAME_AM_NON_AG).round({'Value (t CO2e)': 2})
    GHG_ag_man_non_all = GHG_ag_man.query('Water_supply != "ALL" and `Agricultural Management Type` != "ALL"').reset_index(drop=True)
    
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

    GHG_land = pd.concat([GHG_ag, GHG_non_ag, GHG_ag_man, GHG_transition], axis=0)\
        .query('abs(`Value (t CO2e)`) > 1')\
        .reset_index(drop=True)
    GHG_land_non_all = pd.concat([GHG_ag_non_all, GHG_non_ag, GHG_ag_man_non_all, GHG_transition], axis=0)\
        .query('abs(`Value (t CO2e)`) > 1')\
        .reset_index(drop=True)   
        
    GHG_land['Land-use type'] = GHG_land['Land-use'].apply(lu_group.set_index('Land-use')['Category'].to_dict().get)
    GHG_land_non_all['Land-use type'] = GHG_land_non_all['Land-use'].apply(lu_group.set_index('Land-use')['Category'].to_dict().get)

    net_offland_AUS = GHG_off_land.groupby('Year')[['Value (t CO2e)']].sum(numeric_only=True).reset_index()
    net_offland_AUS_wide = net_offland_AUS[['Year','Value (t CO2e)']].values.tolist()


    GHG_limit = GHG_files.query('base_name == "GHG_emissions"')
    GHG_limit = pd.concat([pd.read_csv(path) for path in GHG_limit['path']], ignore_index=True)
    GHG_limit = GHG_limit.query('Variable == "GHG_EMISSIONS_LIMIT_TCO2e"').copy()
    GHG_limit['Value (t CO2e)'] = GHG_limit['Emissions (t CO2e)']
    GHG_limit_wide = list(map(list,zip(GHG_limit['Year'],GHG_limit['Value (t CO2e)'])))
    
    order_GHG = [
        'Agricultural land-use',
        'Agricultural Management',
        'Non-Agricultural Land-use',
        'Off-land emissions',
        'Unallocated natural to modified',
        'Unallocated natural to livestock natural',
        'Livestock natural to modified',
        'Net emissions',
        'GHG emission limit'
    ]


    # -------------------- GHG overview --------------------

    # sum
    GHG_region = {}
    for region,df in GHG_land_non_all.groupby('region'):
        df_reg = df\
            .groupby(['Year','Type'])[['Value (t CO2e)']]\
            .sum(numeric_only=True)\
            .reset_index()\
            .groupby(['Type'])[['Year','Value (t CO2e)']]\
            .apply(lambda x:x[['Year', 'Value (t CO2e)']].values.tolist())\
            .reset_index()
        df_reg.columns = ['name','data']
        df_reg['type'] = 'column'

        if region == "AUSTRALIA":
            df_reg.loc[len(df_reg)] = ['Off-land emissions', net_offland_AUS_wide,  'column']
            df_reg.loc[len(df_reg)] = ['GHG emission limit', GHG_limit_wide, 'line']
            df_reg.loc[len(df_reg)] = ['Net emissions', 
                list(zip(years, (df.groupby('Year')['Value (t CO2e)'].sum().values + GHG_off_land.groupby('Year')['Value (t CO2e)'].sum()))),
                'line'
            ]
        else:
            df_reg.loc[len(df_reg)] = [
                'Net emissions', 
                list(zip(years, (df.groupby('Year')['Value (t CO2e)'].sum().values))),
                'line'
            ]
                

        df_reg['name_order'] = df_reg['name'].apply(lambda x: order_GHG.index(x))
        df_reg = df_reg.sort_values('name_order').drop(columns=['name_order'])
        GHG_region[region] = json.loads(df_reg.to_json(orient='records'))


    filename = 'GHG_overview_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(GHG_region, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
        
    # Ag
    GHG_ag_non_all_wide = GHG_ag_non_all\
        .groupby(['region','Land-use','Year'])[['Value (t CO2e)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .groupby(['region','Land-use'])[['Year','Value (t CO2e)']]\
        .apply(lambda x:x[['Year', 'Value (t CO2e)']].values.tolist())\
        .reset_index()
        
    GHG_ag_non_all_wide.columns = ['region', 'name','data']
    GHG_ag_non_all_wide['type'] = 'column'
    
    out_dict = {}
    for region,df in GHG_ag_non_all_wide.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')
    

    filename = 'GHG_overview_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Am
    GHG_ag_man_non_all_wide = GHG_ag_man_non_all\
        .groupby(['region', 'Agricultural Management Type', 'Year'])[['Value (t CO2e)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .groupby(['region', 'Agricultural Management Type'])[['Year','Value (t CO2e)']]\
        .apply(lambda x:x[['Year', 'Value (t CO2e)']].values.tolist())\
        .reset_index()
        
    GHG_ag_man_non_all_wide.columns = ['region', 'name','data']
    GHG_ag_man_non_all_wide['type'] = 'column'
    
    out_dict = {}
    for region, df in GHG_ag_man_non_all_wide.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')
    

    filename = 'GHG_overview_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Non-Ag
    GHG_non_ag_wide = GHG_non_ag\
        .groupby(['region','Land-use','Year'])[['Value (t CO2e)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .groupby(['region','Land-use'])[['Year','Value (t CO2e)']]\
        .apply(lambda x:x[['Year', 'Value (t CO2e)']].values.tolist())\
        .reset_index()
        
    GHG_non_ag_wide.columns = ['region','name','data']
    GHG_non_ag_wide['type'] = 'column'
    
    out_dict = {}
    for region,df in GHG_non_ag_wide.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')
    

    filename = 'GHG_overview_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    


    # -------------------- GHG ranking --------------------
    GHG_rank_emission_region = GHG_land_non_all\
        .query('`Value (t CO2e)` > 0')\
        .groupby(['Year', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (t CO2e)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
        .assign(Type='GHG emissions')
        
    GHG_rank_emission_region.loc[
        GHG_rank_emission_region['region'] == 'AUSTRALIA', 
        'Value (t CO2e)'] += GHG_off_land.groupby('Year')['Value (t CO2e)'].sum().values
    
    GHG_rank_sequestration_region = GHG_land_non_all\
        .query('`Value (t CO2e)` < 0')\
        .assign(**{'Value (t CO2e)': lambda x: abs(x['Value (t CO2e)'])})\
        .groupby(['Year', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (t CO2e)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
        .assign(Type='GHG sequestrations')
    GHG_rank_region_net = GHG_rank_emission_region\
        .merge(GHG_rank_sequestration_region, on=['Year', 'region'], how='outer', suffixes=('_emission', '_sequestration'))\
        .assign(**{'Value (t CO2e)': lambda x: x['Value (t CO2e)_emission'] - x['Value (t CO2e)_sequestration']})\
        .assign(Type='Total')


    GHG_rank = pd.concat([
        GHG_rank_emission_region, 
        GHG_rank_sequestration_region, 
        GHG_rank_region_net,
        ], axis=0, ignore_index=True).reset_index(drop=True)\
        .round({'Value (t CO2e)':2})\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))
    

    out_dict = {}
    for (region, e_type), df in GHG_rank.groupby(['region', 'Type']):
        if region not in out_dict:
            out_dict[region] = {}
        if e_type not in out_dict[region]:
            out_dict[region][e_type] = {}

        df = df.drop(columns='region')
        out_dict[region][e_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][e_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][e_type]['value'] = df.set_index('Year')['Value (t CO2e)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'GHG_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        



    # -------------------- GHG by agricultural land-use --------------------
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

    df_wide = GHG_ag_emissions_long\
        .groupby(['region', 'Source', 'Water_supply', 'Land-use', 'Year'])[['Value (t CO2e)']]\
        .sum()\
        .reset_index()\
        .round({'Value (t CO2e)': 2})
    df_wide = df_wide.groupby(['region', 'Source', 'Water_supply', 'Land-use'])[['Year','Value (t CO2e)']]\
        .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['region', 'source', 'water', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, source, water), df in df_wide.groupby(['region', 'source', 'water']):
        df = df.drop(['region', 'source', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if source not in out_dict[region]:
            out_dict[region][source] = {}
        if water not in out_dict[region][source]:
            out_dict[region][source][water] = {}
        out_dict[region][source][water] = df.to_dict(orient='records')
        
    filename = 'GHG_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- GHG by Non-Agricultural --------------------
    Non_ag_reduction_long = GHG_land.query('Type == "Non-Agricultural Land-use"').reset_index(drop=True)
    Non_ag_reduction_long['Value (t CO2e)'] *= -1  # Convert from negative to positive
    
    df_region = Non_ag_reduction_long\
        .groupby(['Year', 'region', 'Land-use'])[['Value (t CO2e)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value (t CO2e)': 2})
    df_wide = df_region.groupby(['Land-use', 'region'])[['Year','Value (t CO2e)']]\
        .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['name', 'region', 'data']
    df_wide['type'] = 'column'
    
    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
    
    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

    filename = f'GHG_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- GHG by Agricultural Managements --------------------
    Ag_man_sequestration_long = GHG_land.query('Type == "Agricultural Management"').reset_index(drop=True)
    Ag_man_sequestration_long['Value (t CO2e)'] = Ag_man_sequestration_long['Value (t CO2e)'] * -1  # Convert from negative to positive
    group_cols = ['Land-use', 'Land-use type', 'Agricultural Management Type', 'Water_supply']

    df_region = Ag_man_sequestration_long\
        .groupby(['region', 'Agricultural Management Type', 'Water_supply', 'Land-use', 'Year'])[['Value (t CO2e)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value (t CO2e)': 2})
    df_wide = df_region.groupby(['region', 'Agricultural Management Type', 'Water_supply', 'Land-use'])[['Year','Value (t CO2e)']]\
        .apply(lambda x: x[['Year', 'Value (t CO2e)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['region', '_type', 'water', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region,_type,water), df in df_wide.groupby(['region', '_type', 'water']):
        df = df.drop(['region', '_type', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if _type not in out_dict[region]:
            out_dict[region][_type] = {}
        if water not in out_dict[region][_type]:
            out_dict[region][_type][water] = {}
        out_dict[region][_type][water] = df.to_dict(orient='records')
        
    filename = 'GHG_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

        


    ####################################################
    #                     5) Water                     #
    ####################################################
    
    water_files = files.query('category == "water"').reset_index(drop=True)
    
    ############ Watershed level  ##############

    water_net_yield_watershed_region = water_files.query('base_name == "water_yield_separate_watershed"')
    water_net_yield_watershed_region = pd.concat([pd.read_csv(path) for path in water_net_yield_watershed_region['path']], ignore_index=True)
    water_net_yield_watershed_AUS = water_net_yield_watershed_region\
        .groupby(['Water Supply',  'Landuse', 'Type', 'Agri-Management', 'Year'], dropna=False)[['Water Net Yield (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(Region='AUSTRALIA')
    water_net_yield_watershed = pd.concat([water_net_yield_watershed_region, water_net_yield_watershed_AUS], ignore_index=True)
    water_net_yield_watershed = water_net_yield_watershed\
        .replace(RENAME_AM_NON_AG)\
        .query('abs(`Water Net Yield (ML)`) > 1e-4')\
        .rename(columns={'Water Net Yield (ML)': 'Value (ML)'})
    water_net_yield_watershed_non_all = water_net_yield_watershed\
        .query('`Water Supply` != "ALL" and `Agri-Management` != "ALL"')\
        .reset_index(drop=True)


    hist_and_public_wny_water_region = water_files.query('base_name == "water_yield_limits_and_public_land"')
    hist_and_public_wny_water_region = pd.concat([pd.read_csv(path) for path in hist_and_public_wny_water_region['path']], ignore_index=True)
    
    water_outside_LUTO = hist_and_public_wny_water_region[['Year','Region', 'Water yield outside LUTO (ML)']].rename(
        columns={'Water yield outside LUTO (ML)': 'Value (ML)'} )
    water_outside_LUTO = pd.concat([
        water_outside_LUTO,
        water_outside_LUTO.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_climate_change_impact = hist_and_public_wny_water_region[['Year','Region', 'Climate Change Impact (ML)']].rename(
        columns={'Climate Change Impact (ML)': 'Value (ML)'})
    water_climate_change_impact = pd.concat([
        water_climate_change_impact,
        water_climate_change_impact.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_domestic_use = hist_and_public_wny_water_region[['Year','Region', 'Domestic Water Use (ML)']]\
        .rename(columns={'Domestic Water Use (ML)': 'Value (ML)'})
    water_domestic_use['Value (ML)'] *= -1  # Domestic water use is negative, indicating a water loss (consumption)
    water_domestic_use = pd.concat([
        water_domestic_use,
        water_domestic_use.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_yield_limit = hist_and_public_wny_water_region[['Year','Region', 'Water Yield Limit (ML)']].rename(
        columns={'Water Yield Limit (ML)': 'Value (ML)'})
    water_yield_limit = pd.concat([
        water_yield_limit,
        water_yield_limit.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_net_yield = hist_and_public_wny_water_region[['Year','Region', 'Water Net Yield (ML)']].rename(
        columns={'Water Net Yield (ML)': 'Value (ML)'})
    water_net_yield = pd.concat([
        water_net_yield,
        water_net_yield.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_targets_before_relaxation = water_files.query('base_name == "water_yield_relaxed_region_raw"')
    water_targets_before_relaxation = pd.concat([pd.read_csv(path) for path in water_targets_before_relaxation['path']])\
        .drop(columns=['Region Id'])\
        .rename(columns={'Region Name': 'Region', 'Target': 'Value (ML)'})


    # -------------------- Water yield overview --------------------
    water_inside_LUTO_wide = water_net_yield_watershed_non_all\
        .groupby(['Region', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .round({'Value (ML)': 2})\
        .reset_index()\
        .groupby(['Region'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_outside_LUTO_wide = water_outside_LUTO\
        .groupby(['Region'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_CCI_wide = water_climate_change_impact\
        .groupby('Region')[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_domestic_wide = water_domestic_use\
        .groupby('Region')[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_net_yield_wide = water_net_yield\
        .groupby('Region')[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_limit_wide = water_yield_limit\
        .groupby('Region')[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    
    # -------------------- Water yield overview --------------------
    water_yield_region = {}
    for reg_name in water_net_yield['Region'].unique():
        
        water_inside = water_inside_LUTO_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_outside = water_outside_LUTO_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_CCI = water_CCI_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_domestic = water_domestic_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_net_yield_sum = water_net_yield_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_limit = water_limit_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        
        water_df = pd.DataFrame([
            ['Water Yield Inside LUTO Study Area', water_inside, 'column', None, None],
            ['Water Yield Outside LUTO Study Area', water_outside, 'column', None, None],
            ['Climate Change Impact', water_CCI, 'column', None, None],
            ['Domestic Water Use', water_domestic, 'column', None, None],
            ['Water Net Yield', water_net_yield_sum, 'line', None, None],
            ['Water Limit (model)', water_limit, 'line', 'black', None],
        ],
            columns=['name', 'data','type','color','dashStyle']
        )
        
        # Add historical water limit if it exists for this region
        if reg_name in water_targets_before_relaxation['Region'].values:
            raw_targets = water_targets_before_relaxation.query('`Region` == @reg_name')[['Year','Value (ML)']].values.tolist()
            water_df.loc[len(water_df)] = ['Water Limit (historical level)', raw_targets, 'line', '#2176cc', 'Dash']

        water_yield_region[reg_name] = water_df.to_dict(orient='records')
        
    filename = 'Water_overview_watershed'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as outfile:
        outfile.write(f'window["{filename}"] = ')
        json.dump(water_yield_region, outfile, separators=(',', ':'), indent=2)
        outfile.write(';\n')
        
        
    ############ NRM region level  ##############  
       
    water_net_yield_NRM_region_region = water_files.query('base_name == "water_yield_separate_NRM"')
    water_net_yield_NRM_region_region = pd.concat([pd.read_csv(path) for path in water_net_yield_NRM_region_region['path']], ignore_index=True)
    water_net_yield_NRM_region_AUS = water_net_yield_NRM_region_region\
        .groupby(['Water Supply', 'Landuse',  'Type', 'Agri-Management', 'Year'], dropna=False)[['Water Net Yield (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(region_NRM='AUSTRALIA')
    water_net_yield_NRM_region = pd.concat([water_net_yield_NRM_region_region, water_net_yield_NRM_region_AUS])\
        .replace(RENAME_AM_NON_AG)\
        .query('abs(`Water Net Yield (ML)`) > 1e-4')\
        .rename(columns={'Water Net Yield (ML)': 'Value (ML)'})
    water_net_yield_NRM_region_non_all = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and `Agri-Management` != "ALL"')\
        .reset_index(drop=True)


    # -------------------- Water yield ranking by NRM --------------------
    water_ranking_type = water_net_yield_NRM_region_non_all\
        .groupby(['region_NRM', 'Type', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Type', 'Value (ML)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year','Type']).cumcount())
        
    water_ranking_total = water_net_yield_NRM_region_non_all\
        .groupby(['region_NRM', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (ML)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby('Year').cumcount())\
        .assign(Type='Total')
        
    water_ranking = pd.concat([water_ranking_type, water_ranking_total], axis=0, ignore_index=True)\
        .round({'Value (ML)':2})\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))

    out_dict = {}
    for (region, w_type), df in water_ranking.groupby(['region_NRM', 'Type']):
        df = df.drop(columns='region_NRM')
        if region not in out_dict:
            out_dict[region] = {}
        if w_type not in out_dict[region]:
            out_dict[region][w_type] = {} 
        out_dict[region][w_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][w_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][w_type]['value'] = df.set_index('Year')['Value (ML)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Water_ranking_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- Overview  --------------------
    
    # sum
    water_sum = water_net_yield_NRM_region_non_all\
        .groupby(['region_NRM', 'Type', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value (ML)': 2})\
        .groupby(['region_NRM', 'Type'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
        
    water_sum.columns = ['region', 'name','data']
    water_sum['type'] = 'column'
    
    out_dict = {}
    for region, df in water_sum.groupby('region'):
        df = df.drop(columns=['region'])
        out_dict[region] = df.to_dict(orient='records')
        
    filename = f'Water_overview_NRM_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    
    # Ag
    water_ag = water_net_yield_NRM_region_non_all.query('Type == "Agricultural Landuse"')
    
    water_overview_ag = water_ag\
        .groupby(['region_NRM', 'Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value (ML)': 2})\
        .groupby(['region_NRM', 'Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
        
    water_overview_ag.columns = ['region', 'name','data']
    water_overview_ag['type'] = 'column'
    water_overview_ag['color'] = water_overview_ag.apply(lambda x: COLORS_LU[x['name']], axis=1)

        
    out_dict = {}
    for region, df in water_overview_ag.groupby('region'):
        df = df.drop(columns=['region'])
        out_dict[region] = df.to_dict(orient='records')

            
    filename = f'Water_overview_NRM_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
    # Am
    water_am = water_net_yield_NRM_region_non_all.query('Type == "Agricultural Management"')
    
    water_overview_am = water_am\
        .groupby(['region_NRM', 'Agri-Management', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value (ML)': 2})\
        .groupby(['region_NRM', 'Agri-Management'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()

    water_overview_am.columns = ['region', 'name', 'data']
    water_overview_am['type'] = 'column'
    water_overview_am['color'] = water_overview_am.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

    out_dict = {}
    for region, df in water_overview_am.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')

    filename = f'Water_overview_NRM_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Non-Ag
    water_nonag = water_net_yield_NRM_region_non_all.query('Type == "Non-Agricultural Land-use"')

    water_overview_nonag = water_nonag\
        .groupby(['region_NRM', 'Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value (ML)': 2})\
        .groupby(['region_NRM', 'Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()

    water_overview_nonag.columns = ['region', 'name', 'data']
    water_overview_nonag['type'] = 'column'
    water_overview_nonag['color'] = water_overview_nonag.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

    out_dict = {}
    for region, df in water_overview_nonag.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')

    filename = f'Water_overview_NRM_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- Water yield for Ag by NRM --------------------
    water_ag_AUS = water_net_yield_NRM_region\
        .query('Type == "Agricultural Landuse"')\
        .groupby(['Water Supply', 'Landuse', 'Year',])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(region='AUSTRALIA')
        
    water_ag = pd.concat([
        water_ag_AUS,
        water_net_yield_NRM_region.query('Type == "Agricultural Landuse"').rename(columns={'region_NRM': 'region'})
        ], ignore_index=True)
    
    df_region_wide = water_ag.groupby(['region', 'Water Supply', 'Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year', 'Value (ML)']].values.tolist())\
        .reset_index()
  
    df_region_wide.columns = ['region', 'water', 'name',  'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water), df in df_region_wide.groupby(['region', 'water']):
        df = df.drop(['region'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water] = df.to_dict(orient='records')
        
        
    filename = f'Water_Ag_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        

        
            
    # -------------------- Water yield for Am by NRM region --------------------
    water_am_AUS = water_net_yield_NRM_region\
        .query('Type == "Agricultural Management"')\
        .groupby(['Agri-Management', 'Water Supply', 'Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(region='AUSTRALIA')
        
    water_am = pd.concat(
        [water_am_AUS,
         water_net_yield_NRM_region.query('Type == "Agricultural Management"').rename(columns={'region_NRM': 'region'})],
        ignore_index=True
    )

    df_region_wide = water_am.groupby(['region', 'Agri-Management', 'Water Supply', 'Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year', 'Value (ML)']].values.tolist())\
        .reset_index()
    df_region_wide.columns = ['region', '_type', 'water', 'name',  'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region,_type,water), df in df_region_wide.groupby(['region', '_type', 'water']):
        df = df.drop(['region', '_type', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if _type not in out_dict[region]:
            out_dict[region][_type] = {}
        if water not in out_dict[region][_type]:
            out_dict[region][_type][water] = {}
        out_dict[region][_type][water] = df.to_dict(orient='records')
        
    filename = f'Water_Am_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
            
            
    # -------------------- Water yield for Non-Agricultural Land-use by NRM region --------------------
    water_nonag_AUS = water_net_yield_NRM_region\
        .query('Type == "Non-Agricultural Land-use"')\
        .groupby(['Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(
            region='AUSTRALIA',
            name_order=lambda x: x['Landuse'].apply(lambda y: LANDUSE_ALL_RENAMED.index(y)))\
        .sort_values('name_order')\
        .drop(columns=['name_order'])
        
    water_nonag = pd.concat([
        water_nonag_AUS,
        water_net_yield_NRM_region.query('Type == "Non-Agricultural Land-use"').rename(columns={'region_NRM': 'region'})
        ], ignore_index=True)

    df_region_wide = water_nonag.groupby(['region', 'Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year', 'Value (ML)']].values.tolist())\
        .reset_index()
    df_region_wide.columns = ['region', 'name', 'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
    df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_region_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
    filename = f'Water_NonAg_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        



    #########################################################
    #                   6) Biodiversity                     #
    #########################################################

    bio_rank_dict = {}
    
    # ---------------- Overall quality ----------------
    filter_str = '''
        category == "biodiversity"
        and base_name == "biodiversity_overall_priority_scores"
    '''.strip().replace('\n','')
    
    bio_paths = files.query(filter_str).reset_index(drop=True)
    bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])\
        .replace(RENAME_AM_NON_AG)\
        .rename(columns={'Contribution Relative to Base Year Level (%)': 'Value (%)'})\
        .query('abs(`Area Weighted Score (ha)`) > 1e-4')\
        .round({'Value (%)': 6})
    bio_df_non_all = bio_df.query('Water_supply != "ALL" and `Agri-Management` != "ALL"')
    bio_df_ag_non_all = bio_df_non_all.query('Type == "Agricultural Landuse"').copy()
    bio_df_am_non_all = bio_df_non_all.query('Type == "Agricultural Management"').copy()
    bio_df_nonag = bio_df_non_all.query('Type == "Non-Agricultural Land-use"').copy()
    
    # ---------------- Overall quality - Ranking -----------------
    bio_rank_type = bio_df_non_all\
        .groupby(['Year', 'region', 'Type'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Type', 'Value (%)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year','Type']).cumcount())\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))
    bio_rank_total = bio_df_non_all\
        .groupby(['Year', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))\
        .assign(Type='Total')
    bio_rank = pd.concat([bio_rank_type, bio_rank_total], axis=0, ignore_index=True)
    
    
    for region, df in bio_rank_total.groupby('region'):
        if region not in bio_rank_dict:
            bio_rank_dict[region] = {}
            bio_rank_dict[region]['Quality'] = {}

        df = df.drop(columns='region')
        bio_rank_dict[region]['Quality']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        bio_rank_dict[region]['Quality']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        bio_rank_dict[region]['Quality']['value'] = df.set_index('Year')['Value (%)'].apply( lambda x: format_with_suffix(x)).to_dict()


    out_dict = {}
    for (region, _type), df in bio_rank.groupby(['region', 'Type']):
        if region not in out_dict:
            out_dict[region] = {}
        if _type not in out_dict[region]:
            out_dict[region][_type] = {}
        out_dict[region][_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][_type]['value'] = df.set_index('Year')['Value (%)'].apply( lambda x: format_with_suffix(x)).to_dict()
        
    filename = 'BIO_quality_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')




    # ---------------- Overall quality - Overview ----------------
    # sum
    df_region = bio_df_non_all\
        .groupby(['region', 'Year', 'Type'])\
        .sum(numeric_only=True)\
        .reset_index()
    df_wide = df_region.groupby(['Type', 'region'])[['Year','Value (%)']]\
        .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['name', 'region', 'data']
    df_wide['type'] = 'column'

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

    filename = f'BIO_quality_overview_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    
    
    
    # ag
    df_region = bio_df_ag_non_all\
        .groupby(['Year', 'region', 'Landuse'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .query('abs(`Area Weighted Score (ha)`) > 1e-4')
    df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Value (%)']]\
        .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['name', 'region', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])


    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
    filename = f'BIO_quality_overview_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
            
    # am
    df_region = bio_df_am_non_all\
        .groupby(['Year', 'region', "Agri-Management"])\
        .sum(numeric_only=True)\
        .reset_index()\
        .query('abs(`Area Weighted Score (ha)`) > 1e-4')
    df_wide = df_region.groupby(["Agri-Management", 'region'])[['Year','Value (%)']]\
        .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['name', 'region', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)
    
    
    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
    filename = f'BIO_quality_overview_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # non-ag
    df_region = bio_df_nonag\
        .groupby(['Year', 'region', 'Landuse'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .query('abs(`Area Weighted Score (ha)`) > 1e-4')
    df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Value (%)']]\
        .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['name', 'region', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
    filename = f'BIO_quality_overview_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # ---------------- Overall quality - Ag ----------------
    bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()
    bio_df_wide = bio_df_ag.groupby(['region', 'Water_supply', 'Landuse'])[['Year','Value (%)']]\
        .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
        .reset_index()
    bio_df_wide.columns = ['region', 'water', 'name', 'data']
    bio_df_wide['type'] = 'column'
    bio_df_wide['color'] = bio_df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    bio_df_wide['name_order'] = bio_df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))   
    bio_df_wide = bio_df_wide.sort_values('name_order').drop(columns=['name_order'])
    
    out_dict = {}
    for (region,water), df in bio_df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water] = df.to_dict(orient='records')

    filename = f'BIO_quality_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
    # ---------------- Overall quality - Am ----------------
    bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

    df_wide = bio_df_am.groupby(['region', 'Water_supply', "Agri-Management" ])[['Year','Value (%)']]\
        .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['region', 'water', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

    out_dict = {}
    for (region,water), df in df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water] = df.to_dict(orient='records')

    filename = f'BIO_quality_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # ---------------- Overall quality - Non-Ag ----------------
    bio_df_nonag = bio_df.query('Type == "Non-Agricultural Land-use"').copy()
    df_wide = bio_df_nonag.groupby(['region', 'Landuse'])[['Year','Value (%)']]\
        .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
        .reset_index()
    df_wide.columns = ['region', 'name', 'data']
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

    filename = f'BIO_quality_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    
    
        
    if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':

        filter_str = '''
            category == "biodiversity" 
            and base_name == "biodiversity_GBF2_priority_scores"
        '''.strip('').replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')\
            .round({'Value (%)': 2})
            
        bio_df_non_all = bio_df.query('Water_supply != "ALL" and `Agri-Management` != "ALL"')
        bio_df_ag_non_all = bio_df_non_all.query('Type == "Agricultural Landuse"')
        bio_df_am_non_all = bio_df_non_all.query('Type == "Agricultural Management"')
        bio_df_nonag = bio_df_non_all.query('Type == "Non-Agricultural Land-use"')
        
        # ---------------- (GBF2) ranking  ----------------
        bio_rank_total = bio_df_non_all\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Area Weighted Score (ha)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if 'GBF2' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF2'] = {}

            bio_rank_dict[region]['GBF2']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF2']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF2']['value'] = df.set_index('Year')['Area Weighted Score (ha)'].apply( lambda x: format_with_suffix(x)).to_dict()
            

        # ---------------- (GBF2) overview  ----------------
        
        # sum
        bio_df_target = bio_df.groupby(['Year'])[['Priority Target (%)']].agg('first').reset_index()
        bio_df_target = bio_df_target[['Year','Priority Target (%)']].values.tolist()

        df_region = bio_df_non_all\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Type', 'region'])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'
        
        df_wide.loc[len(df_wide)] = ['Target (%)', 'AUSTRALIA', bio_df_target, 'line']

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')



        # ag
        df_region = bio_df_ag_non_all\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
                
        # am
        df_region = bio_df_am_non_all\
            .groupby(['Year', 'region', 'Agri-Management'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Agri-Management', 'region'])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'
        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
                
        # non-ag
        df_region = bio_df_nonag\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))


        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
        # ---------------- (GBF2) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()
        df_region = bio_df_ag\
            .groupby(['region', 'Water_supply', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_region.columns = ['region', 'water', 'name', 'data']
        df_region['type'] = 'column'
        df_region['color'] = df_region.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_region['name_order'] = df_region['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_region = df_region.sort_values('name_order').drop(columns=['name_order'])
        
        out_dict = {}
        for (region, water), df in df_region.groupby(['region', 'water']):
            df = df.drop(['region', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if water not in out_dict[region]:
                out_dict[region][water] = {}
            out_dict[region][water] = df.to_dict(orient='records')
            
        filename = f'BIO_GBF2_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
        
        # ---------------- (GBF2) Ag-Mgt  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()
        df_region = bio_df_am\
            .groupby(['region', 'Water_supply', 'Agri-Management'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_region.columns = ['region', 'water', 'name', 'data']
        df_region['type'] = 'column'
        df_region['color'] = df_region.apply(lambda x: COLORS_AM_NONAG[x['name']], axis=1)

        out_dict = {}
        for (region, water), df in df_region.groupby(['region', 'water']):
            df = df.drop(['region', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if water not in out_dict[region]:
                out_dict[region][water] = {}
            out_dict[region][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
        # ---------------- (GBF2) Non-Ag  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural Land-use"').copy()
        df_region = bio_df_nonag\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
            
            
            
    if settings.BIODIVERSITY_TARGET_GBF_3 != 'off':
        filter_str = '''
            category == "biodiversity" 
            and base_name.str.contains("biodiversity_GBF3")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path, low_memory=False) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Vegetation Group': 'species'})\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')\
            .round(6)
        bio_df_non_all = bio_df.query('Water_supply != "ALL" and `Agri-Management` != "ALL"')
        bio_df_ag_non_all = bio_df_non_all.query('Type == "Agricultural Landuse"')
        bio_df_am_non_all = bio_df_non_all.query('Type == "Agricultural Management"')
        bio_df_nonag = bio_df_non_all.query('Type == "Non-Agricultural Land-use"')
        
        
        # ---------------- (GBF3) Ranking  ----------------
        bio_rank_total = bio_df_non_all\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Area Weighted Score (ha)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))
            
        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if 'GBF3' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF3'] = {}
                
            bio_rank_dict[region]['GBF3']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF3']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF3']['value'] = df.set_index('Year')['Area Weighted Score (ha)'].apply(lambda x: format_with_suffix(x)).to_dict()


        # ---------------- (GBF3) Overview  ----------------
        
        # sum
        bio_df_target = bio_df_non_all.groupby(['Year', 'species'])[['Target_by_Percent']].agg('first').reset_index()

        df_region = bio_df_non_all\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Type', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
                
                
                
        # ag
        df_region = bio_df_ag_non_all\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # am
        df_region = bio_df_am_non_all\
            .groupby(['Year', 'region', 'Agri-Management'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Agri-Management', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'


        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # non-ag
        df_region = bio_df_nonag\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])


        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
        # ---------------- (GBF3) - Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()

        df_wide = bio_df_ag\
            .groupby(['region', 'species', 'Water_supply', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        
        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')
            
        filename = f'BIO_GBF3_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
        # ---------------- (GBF3) - Am  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()
        
        df_wide = bio_df_am\
            .groupby(['region', 'species', 'Water_supply', 'Agri-Management', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'Ag Mgt', 'Land-use', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['Land-use']], axis=1)

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
        # ---------------- (GBF3) - Non-Ag  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural Land-use"').copy()
        
        df_wide = bio_df_nonag\
            .groupby(['region', 'species', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            out_dict[region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            

    if settings.BIODIVERSITY_TARGET_GBF_4_SNES == 'on':
        
        filter_str = '''
            category == "biodiversity" 
            and base_name.str.contains("biodiversity_GBF4_SNES_scores")
        '''.strip().replace('\n', '')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')\
            .round(6)
        bio_df_non_all = bio_df.query('Water_supply != "ALL" and `Agri-Management` != "ALL"')
        bio_df_ag_non_all = bio_df_non_all.query('Type == "Agricultural Landuse"')
        bio_df_am_non_all = bio_df_non_all.query('Type == "Agricultural Management"')
        bio_df_nonag = bio_df_non_all.query('Type == "Non-Agricultural Land-use"')

        # ---------------- (GBF4 SNES) Ranking  ----------------
        bio_rank_total = bio_df_non_all\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))
            
        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if 'GBF4 (SNES)' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF4 (SNES)'] = {}

            bio_rank_dict[region]['GBF4 (SNES)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF4 (SNES)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF4 (SNES)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()



        # ---------------- (GBF4 SNES) Overview  ----------------

        # sum
        df_region = bio_df_non_all\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Type', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ag
        df_region = bio_df_ag_non_all\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # am
        df_region = bio_df_am_non_all\
            .groupby(['Year', 'region', 'Agri-Management'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Agri-Management', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # non-ag
        df_region = bio_df_nonag\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF4 SNES) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()

        df_wide = bio_df_ag\
            .groupby(['region', 'species', 'Water_supply', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # ---------------- (GBF4 SNES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        df_wide = bio_df_am\
            .groupby(['region', 'species', 'Water_supply', 'Agri-Management', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'Ag Mgt', 'Land-use', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['Land-use']], axis=1)

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF4 SNES) Non-ag  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural Land-use"').copy()

        df_wide = bio_df_nonag\
            .groupby(['region', 'species', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            out_dict[region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
            
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES == 'on':
        
        bio_paths = files.query('base_name.str.contains("biodiversity_GBF4_ECNES_scores")')
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')\
            .round(6)

        bio_df_non_all = bio_df.query('Water_supply != "ALL" and `Agri-Management` != "ALL"')
        bio_df_ag_non_all = bio_df_non_all.query('Type == "Agricultural Landuse"')
        bio_df_am_non_all = bio_df_non_all.query('Type == "Agricultural Management"')
        bio_df_nonag = bio_df_non_all.query('Type == "Non-Agricultural Land-use"')


        # ---------------- (GBF4 ECNES) Ranking  ----------------
        bio_rank_total = bio_df_non_all\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if 'GBF4 (ECNES)' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF4 (ECNES)'] = {}

            bio_rank_dict[region]['GBF4 (ECNES)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF4 (ECNES)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF4 (ECNES)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

        # ---------------- (GBF4 ECNES) Overview  ----------------

        # sum
        df_region = bio_df_non_all\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Type', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ag
        df_region = bio_df_ag_non_all\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # am
        df_region = bio_df_am_non_all\
            .groupby(['Year', 'region', 'Agri-Management'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Agri-Management', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # non-ag
        df_region = bio_df_nonag\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF4 ECNES) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()

        df_wide = bio_df_ag\
            .groupby(['region', 'species', 'Water_supply', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # ---------------- (GBF4 ECNES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        df_wide = bio_df_am\
            .groupby(['region', 'species', 'Water_supply', 'Agri-Management', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'Ag Mgt', 'Land-use', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['Land-use']], axis=1)

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF4 ECNES) Non-ag  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural Land-use"').copy()

        df_wide = bio_df_nonag\
            .groupby(['region', 'species', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            out_dict[region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
        
        
        

    if settings.BIODIVERSITY_TARGET_GBF_8 == 'on':
        
        filter_str = '''
            category == "biodiversity" 
            
            and base_name.str.contains("biodiversity_GBF8_species_scores")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Species':'species'})\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')\
            .round(6)

        bio_df_non_all = bio_df.query('Water_supply != "ALL" and `Agri-Management` != "ALL"')
        bio_df_ag_non_all = bio_df_non_all.query('Type == "Agricultural Landuse"')
        bio_df_am_non_all = bio_df_non_all.query('Type == "Agricultural Management"')
        bio_df_nonag = bio_df_non_all.query('Type == "Non-Agricultural Land-use"')

        # ---------------- (GBF8 SPECIES) Ranking  ----------------
        bio_rank_total = bio_df_non_all\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if 'GBF8 (SPECIES)' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF8 (SPECIES)'] = {}

            bio_rank_dict[region]['GBF8 (SPECIES)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF8 (SPECIES)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF8 (SPECIES)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

        # ---------------- (GBF8 SPECIES) Overview  ----------------

        # sum
        df_region = bio_df_non_all\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Type', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ag
        df_region = bio_df_ag_non_all\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # am
        df_region = bio_df_am_non_all\
            .groupby(['Year', 'region', 'Agri-Management'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Agri-Management', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # non-ag
        df_region = bio_df_nonag\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 SPECIES) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()

        df_wide = bio_df_ag\
            .groupby(['region', 'species', 'Water_supply', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # ---------------- (GBF8 SPECIES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        df_wide = bio_df_am\
            .groupby(['region', 'species', 'Water_supply', 'Agri-Management', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'Ag Mgt', 'Land-use', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['Land-use']], axis=1)

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 SPECIES) Non-ag  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural Land-use"').copy()

        df_wide = bio_df_nonag\
            .groupby(['region', 'species', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            out_dict[region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
        
        
    
        # ---------------- (GBF8 GROUP)  ----------------
        bio_paths = files.query('base_name.str.contains("biodiversity_GBF8_groups_scores")')
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Group':'species'})\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')\
            .round(6)

        bio_df_non_all = bio_df.query('Water_supply != "ALL" and `Agri-Management` != "ALL"')
        bio_df_ag_non_all = bio_df_non_all.query('Type == "Agricultural Landuse"')
        bio_df_am_non_all = bio_df_non_all.query('Type == "Agricultural Management"')
        bio_df_nonag = bio_df_non_all.query('Type == "Non-Agricultural Land-use"')

        # ---------------- (GBF8 GROUP) Ranking  ----------------
        bio_rank_total = bio_df_non_all\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if 'GBF8 (GROUP)' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF8 (GROUP)'] = {}

            bio_rank_dict[region]['GBF8 (GROUP)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF8 (GROUP)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF8 (GROUP)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

        # ---------------- (GBF8 GROUP) Overview  ----------------

        # sum
        df_region = bio_df_non_all\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Type', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ag
        df_region = bio_df_ag_non_all\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # am
        df_region = bio_df_am_non_all\
            .groupby(['Year', 'region', 'Agri-Management'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Agri-Management', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # non-ag
        df_region = bio_df_nonag\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .query('abs(`Area Weighted Score (ha)`) > 1e-4')
        df_wide = df_region.groupby(['Landuse', 'region'])[['Year','Area Weighted Score (ha)']]\
            .apply(lambda x: x[['Year', 'Area Weighted Score (ha)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['name', 'region', 'data']
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 GROUP) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Landuse"').copy()

        df_wide = bio_df_ag\
            .groupby(['region', 'species', 'Water_supply', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # ---------------- (GBF8 GROUP) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management"').copy()

        df_wide = bio_df_am\
            .groupby(['region', 'species', 'Water_supply', 'Agri-Management', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'water', 'Ag Mgt', 'Land-use', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['Land-use']], axis=1)

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 GROUP) Non-ag  ----------------
        bio_df_nonag = bio_df.query('Type == "Non-Agricultural Land-use"').copy()

        df_wide = bio_df_nonag\
            .groupby(['region', 'species', 'Landuse'])[['Year', 'Value (%)']]\
            .apply(lambda x: x[['Year', 'Value (%)']].values.tolist())\
            .reset_index()
        df_wide.columns = ['region', 'species', 'name', 'data']
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide.apply(lambda x: COLORS_LU[x['name']], axis=1)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            out_dict[region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


    # Save unified bio ranking data
    filename = 'BIO_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(bio_rank_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    #########################################################
    # Supporting information
    #########################################################       
    with open(f'{raw_data_dir}/model_run_settings.txt', 'r', encoding='utf-8') as src_file:
        settings_dict = {i.split(':')[0].strip(): ''.join(i.split(':')[1:]).strip() for i in src_file.readlines()}
        settings_dict = [{'parameter': k, 'val': v} for k, v in settings_dict.items()]
        
    with open(f'{raw_data_dir}/RES_{settings.RESFACTOR}_mem_log.txt', 'r', encoding='utf-8') as src_file:
        mem_logs = [i for i in src_file.readlines() if i != '\n']
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
        'mem_logs': mem_logs_obj
    }
    
    filename = 'Supporting_info'
    with open(f"{SAVE_DIR}/{filename}.js", 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(supporting, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    
