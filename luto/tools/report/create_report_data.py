
import os
import json
import re
import shutil
import pandas as pd
from glob import glob
from joblib import Parallel, delayed
import luto.settings as settings

from luto.economics.off_land_commodity import get_demand_df


# import functions
from luto.tools.report.data_tools import   (get_GHG_emissions_by_crop_lvstk_df,
                                            get_all_files, 
                                            get_quantity_df, 
                                            get_ag_rev_cost_df,
                                            get_water_df,  
                                            get_ag_dvar_area)

              
from luto.tools.report.data_tools.helper_func import (get_GHG_category, 
                                                      get_GHG_file_df,
                                                      get_rev_cost, 
                                                      target_GHG_2_Json, 
                                                      select_years)
                              

                                                     
from luto.tools.report.data_tools.parameters import (COMMODITIES_OFF_LAND, 
                                                     YR_BASE, 
                                                     COMMODITIES_ALL, 
                                                     LANDUSE_ALL,
                                                     LU_NATURAL,
                                                     NON_AG_LANDUSE)

# Get the output directory
def save_report_data(sim):
    
    # Get the output directory
    raw_data_dir = sim.path

    # Set the save directory    
    SAVE_DIR = f'{raw_data_dir}/DATA_REPORT/data'
    
    # Create the directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir)
    
    # Set the years to be int
    files['year'] = files['year'].astype(int)
    
    # Select the years to reduce the column number to 
    # avoid cluttering in the multi-level axis graphing
    years = sorted(files['year'].unique().tolist())
    years_select = select_years(years)


    ####################################################
    #                   1) Demand                      #
    ####################################################
    
    # Get the demand data
    DEMAND_DATA_long = get_demand_df()

    # Reorder the columns to match the order in COMMODITIES_ALL
    DEMAND_DATA_long = DEMAND_DATA_long.reindex(COMMODITIES_ALL, level=1).reset_index()

    # Add columns for on-land and off-land commodities
    DEMAND_DATA_long['on_off_land'] = DEMAND_DATA_long['COMMODITY'].apply(
        lambda x: 'On-land' if x not in COMMODITIES_OFF_LAND else 'Off-land')

    # Convert quanlity to million tonnes
    DEMAND_DATA_long['Quantity (tonnes, ML)'] = DEMAND_DATA_long['Quantity (tonnes, ML)'] / 1e6
    DEMAND_DATA_long = DEMAND_DATA_long.query('Year.isin(@years)')
    DEMAND_DATA_long.loc[:,'COMMODITY'] = DEMAND_DATA_long['COMMODITY'].str.replace('Beef lexp','Beef live export')
    DEMAND_DATA_long_filter_year = DEMAND_DATA_long.query('Year.isin(@years_select)')

    # Plot_1_1: {Total} for 'Domestic', 'Exports', 'Feed', 'Imports', 'Production'(Tonnes) 
    DEMAND_DATA_type = DEMAND_DATA_long_filter_year.groupby(['Year','Type']).sum(numeric_only=True).reset_index()
    DEMAND_DATA_type_wide = DEMAND_DATA_type.pivot(index='Year', columns='Type', values='Quantity (tonnes, ML)').reset_index()
    DEMAND_DATA_type_wide.to_csv(f'{SAVE_DIR}/production_1_demand_type_wide.csv', index=False)


    # Plot_1_2: {ON/OFF land} for 'Domestic', 'Exports', 'Feed', 'Imports', 'Production'(Tonnes) 
    DEMAND_DATA_on_off = DEMAND_DATA_long_filter_year.groupby(['Year','Type','on_off_land']).sum(numeric_only=True).reset_index()
    DEMAND_DATA_on_off_wide = DEMAND_DATA_on_off.pivot(index='on_off_land', 
                                                    columns=['Year','Type'], 
                                                    values='Quantity (tonnes, ML)')

    DEMAND_DATA_on_off_wide = DEMAND_DATA_on_off_wide.reindex(
        columns = pd.MultiIndex.from_product(DEMAND_DATA_on_off_wide.columns.levels)).reset_index()

    DEMAND_DATA_on_off_wide.to_csv(f'{SAVE_DIR}/production_2_demand_on_off_wide.csv', index=False)

    # Plot_1_3: {Commodity} 'Domestic', 'Exports', 'Feed', 'Imports', 'Production' (Tonnes)
    DEMAND_DATA_wide = DEMAND_DATA_long_filter_year.pivot(
        index=['COMMODITY'], 
        columns=['Year','Type'], 
        values='Quantity (tonnes, ML)')

    DEMAND_DATA_wide = DEMAND_DATA_wide.reindex(
        columns = pd.MultiIndex.from_product(DEMAND_DATA_wide.columns.levels)).reset_index()

    DEMAND_DATA_wide = DEMAND_DATA_wide.set_index('COMMODITY').reindex(COMMODITIES_ALL).reset_index()

    DEMAND_DATA_wide.to_csv(f'{SAVE_DIR}/production_3_demand_commodity.csv', index=False)


    # Plot_1-4_(1-2): Domestic On/Off land commodities (Million Tonnes)
    for idx,on_off_land in enumerate(DEMAND_DATA_long['on_off_land'].unique()):
        DEMAND_DATA_on_off_commodity = DEMAND_DATA_long.query('on_off_land == @on_off_land and Type == "Domestic" ')
        DEMAND_DATA_on_off_commodity_wide = DEMAND_DATA_on_off_commodity.pivot(
            index=['Year'], 
            columns=['COMMODITY'], 
            values='Quantity (tonnes, ML)').reset_index()
        
        # Remove the rows of all 0 values
        DEMAND_DATA_on_off_commodity_wide = DEMAND_DATA_on_off_commodity_wide.loc[:, (DEMAND_DATA_on_off_commodity_wide != 0).any(axis=0)]
        
        on_off_land = '_'.join(on_off_land.split(' '))
        DEMAND_DATA_on_off_commodity_wide.to_csv(f'{SAVE_DIR}/production_4_{idx+1}_demand_domestic_{on_off_land}_commodity.csv', index=False)

    # Plot_1-5_(1-4): Commodities for 'Exports','Feed','Imports','Production' (Million Tonnes)
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
        DEMAND_DATA_commodity_wide.to_json(f'{SAVE_DIR}/production_5_{idx+1}_demand_{Type}_commodity.json', orient='records')
        
    # Plot_1-5-6: Production (LUTO outputs, Million Tonnes)
    quantity_csv_paths = files.query('category == "quantity" and base_name == "quantity_comparison" and year_types == "single_year"').reset_index(drop=True)
    quantity_df = get_quantity_df(quantity_csv_paths)
    quantity_df['Commodity'] = quantity_df['Commodity'].str.replace('Beef lexp','Beef live export')

    quantity_df_wide = quantity_df.pivot_table(index=['year'], 
                                            columns='Commodity',
                                            values='Prod_targ_year (tonnes, ML)').reset_index()
    quantity_df_wide.to_csv(f'{SAVE_DIR}/production_5_6_demand_Production_commodity_from_LUTO.csv', index=False)




    ####################################################
    #                  2) Economics                    #
    ####################################################

    # Plot_2-2: Revenue and Cost data (Billion Dollars)
    revenue_df = get_ag_rev_cost_df(files, 'revenue')
    revenue_df['Source_type'] = revenue_df['Source_type'].str.replace(' Crop','')
    revenue_df['Irrigation'] = revenue_df['Irrigation'].replace({'dry': 'Dryland', 'irr': 'Irrigated'})

    keep_cols = ['year', 'value (billion)']
    loop_cols = revenue_df.columns.difference(keep_cols)

    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = revenue_df[take_cols].groupby(['year', col]).sum(numeric_only=True).reset_index()
        # convert to wide format
        df_wide = df.pivot_table(index=['year'], columns=col, values='value (billion)').reset_index()
        # save to csv
        df_wide.to_csv(f'{SAVE_DIR}/economics_1_revenue_{idx+1}_{col}_wide.csv', index=False)



    cost_df = get_ag_rev_cost_df(files, 'cost')
    cost_df['Source_type'] = cost_df['Source_type'].str.replace(' Crop','')
    cost_df['Irrigation'] = cost_df['Irrigation'].replace({'dry': 'Dryland', 'irr': 'Irrigated'})

    keep_cols = ['year', 'value (billion)']
    loop_cols = cost_df.columns.difference(keep_cols)

    # Plot_2-3: Cost data (Billion Dollars)
    for idx,col in enumerate(loop_cols):
        take_cols = keep_cols + [col]
        df = cost_df[take_cols].groupby(['year', col]).sum().reset_index()
        # convert to wide format
        df_wide = df.pivot_table(index=['year'], columns=col, values='value (billion)').reset_index()
        # save to csv
        df_wide.to_csv(f'{SAVE_DIR}/economics_2_cost_{idx+1}_{col}_wide.csv', index=False)


    # Plot_2-4: Revenue and Cost data (Billion Dollars)
    rev_cost_compare = get_rev_cost(revenue_df,cost_df)
    rev_cost_compare.to_csv(f'{SAVE_DIR}/economics_3_rev_cost_all.csv', index=False)



    ####################################################
    #                    3) Area Change                #
    ####################################################

    area_dvar_paths = files.query('category == "area" and year_types == "single_year"').reset_index(drop=True)
    area_dvar = get_ag_dvar_area(area_dvar_paths)
    area_dvar['Water'] = area_dvar['Water'].replace({'dry': 'Dryland', 'irr': 'Irrigated'})

    # Plot_3-1: Total Area (km2)
    lu_area_dvar = area_dvar.groupby(['Year','Land use']).sum(numeric_only=True).reset_index()
    lu_area_dvar = lu_area_dvar\
        .groupby('Land use')[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    lu_area_dvar.columns = ['name','data']
    lu_area_dvar['type'] = 'column'
    lu_area_dvar['sort_index'] = lu_area_dvar['name'].apply(lambda x: LANDUSE_ALL.index(x))
    lu_area_dvar = lu_area_dvar.sort_values('sort_index').drop('sort_index',axis=1)
    lu_area_dvar.to_json(f'{SAVE_DIR}/area_1_total_area_wide.json',orient='records')




    # Plot_3-2: Total Area (km2) by Irrigation
    lm_dvar_area = area_dvar.groupby(['Year','Water']).sum(numeric_only=True).reset_index()
    lm_dvar_area['Water'] = lm_dvar_area['Water'].replace({'dry': 'Dryland', 'irr': 'Irrigated'})

    lm_dvar_area = lm_dvar_area\
        .groupby(['Water'])[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    lm_dvar_area.columns = ['name','data']
    lm_dvar_area['type'] = 'column'
    lm_dvar_area.to_json(f'{SAVE_DIR}/area_2_irrigation_area_wide.json',orient='records')




    # Plot_3-3: Area (km2) by Non-Agricultural land use
    non_ag_dvar_area = area_dvar.query('Type == "Non-agricultural landuse"').reset_index(drop=True)
    non_ag_dvar_area = non_ag_dvar_area\
        .groupby(['Land use'])[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    non_ag_dvar_area.columns = ['name','data']
    non_ag_dvar_area['type'] = 'column'
    non_ag_dvar_area.to_json(f'{SAVE_DIR}/area_3_non_ag_lu_area_wide.json',orient='records')



    # Plot_3-4: Area (km2) by Agricultural management
    am_dvar_dfs = area_dvar_paths.query('base_name.str.contains("area_agricultural_management")').reset_index(drop=True)
    am_dvar_area = pd.concat([pd.read_csv(path) for path in am_dvar_dfs['path']], ignore_index=True)
    am_dvar_area['Area (million km2)'] = am_dvar_area['Area (ha)'] / 100 / 1e6

    am_dvar_area_type = am_dvar_area.groupby(['Year','Type']).sum(numeric_only=True).reset_index()

    am_dvar_area_type = am_dvar_area_type\
        .groupby(['Type'])[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    am_dvar_area_type.columns = ['name','data']
    am_dvar_area_type['type'] = 'column'
    am_dvar_area_type.to_json(f'{SAVE_DIR}/area_4_am_total_area_wide.json',orient='records')



    # Plot_3-5: Agricultural management Area (km2) by Land use
    am_dvar_area_lu = am_dvar_area.groupby(['Year','Land use']).sum(numeric_only=True).reset_index()

    am_dvar_area_lu = am_dvar_area_lu\
        .groupby(['Land use'])[['Year','Area (million km2)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million km2)']))))\
        .reset_index()
        
    am_dvar_area_lu.columns = ['name','data']
    am_dvar_area_lu['type'] = 'column'
    am_dvar_area_lu.to_json(f'{SAVE_DIR}/area_5_am_lu_area_wide.json',orient='records')


    # Plot_3-6/7: Area (km2) by Land use
    transition_path = files.query('category =="transition_matrix"')

    # Read the transition matrix (Area (ha))
    transition_df = pd.read_csv(transition_path['path'].values[0], index_col=0)

    # Convert the transition matrix to km2
    transition_df_area = transition_df / 100 

    # Get the total area of each land use
    total_area = transition_df_area.sum(axis=1).values.reshape(-1,1)

    # Calculate the percentage of each land use
    transition_df_pct = transition_df_area / total_area * 100
    transition_df_pct = transition_df_pct.fillna(0)

    # Add the total area to the transition matrix
    transition_df_area['SUM'] = transition_df_area.sum(axis=1)
    transition_df_area.loc['SUM'] = transition_df_area.sum(axis=0)

    heat_area = transition_df_area.style.background_gradient(cmap='Oranges', 
                                                            axis=1, 
                                                            subset=pd.IndexSlice[:transition_df_area.index[-2], :transition_df_area.columns[-2]]).format('{:,.0f}')
    
    heat_pct = transition_df_pct.style.background_gradient(cmap='Oranges', 
                                                        axis=1,
                                                        vmin=0, 
                                                        vmax=100).format('{:,.2f}')

    # Define the style
    # style = "<style>table, th, td {font-size: 7px;font-family: Helvetica, Arial, sans-serif;} </style>\n"
    # style = style + "<style>td {text-align: right; } </style>\n"
    # style = style + "<style>html { height: 100%; } </style>\n"

    # # Add the style to the HTML
    # heat_area_html = style + heat_area.to_html()
    # heat_pct_html = style + heat_pct.to_html()

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
    #                       4) GHGs                    #
    ####################################################
    GHG_files_onland = get_GHG_file_df(files)
    GHG_files_onland = GHG_files_onland.reset_index(drop=True).sort_values(['year','GHG_sum_t'])
    GHG_files_onland['GHG_sum_Mt'] = GHG_files_onland['GHG_sum_t'] / 1e6
    GHG_files_onland['base_name'] = GHG_files_onland['base_name'].replace({'Transition Penalty': 'Deforestation'})
    
    # Read the off-land GHG emissions
    last_year = GHG_files_onland['year'].max()
    GHG_file_off_land = glob(f'{raw_data_dir}/out_{last_year}/GHG_emissions_offland_commodity*.csv')[0]
    GHG_off_land = pd.read_csv(GHG_file_off_land).query('YEAR in @years')
    GHG_off_land['GHG_sum_Mt'] = GHG_off_land['Total GHG Emissions (tCO2e)'] / 1e6
    GHG_off_land = GHG_off_land.rename(columns={'YEAR':'year'})
    GHG_off_land['COMMODITY'] = GHG_off_land['COMMODITY'].apply(lambda x: x[0].capitalize() + x[1:])
    GHG_off_land['Emission Source'] = GHG_off_land['Emission Source'].replace({'CO2': 'Carbon Dioxide (CO2)',
                                                                               'CH4': 'Methane (CH4)',
                                                                               'N2O': 'Nitrous Oxide (N2O)'})
    
    # Plot_4-1: GHG of cumulative emissions (Mt)
    Emission_onland = GHG_files_onland.groupby('year')['GHG_sum_Mt'].sum(numeric_only = True).reset_index()
    Emission_offland = GHG_off_land.groupby('year').sum(numeric_only=True).reset_index()
    Emission_offland = Emission_offland[['year','GHG_sum_Mt']]
    Emission_offland['base_name'] = 'Off-land Commodity'
    
    Net_emission = pd.concat([Emission_onland,Emission_offland],axis=0)
    Net_emission = Net_emission.groupby(['year']).sum(numeric_only = True).reset_index()
    
    Cumsum_emissions = Net_emission.copy()
    Cumsum_emissions['Cumulative GHG emissions (Mt)'] = Cumsum_emissions.cumsum()['GHG_sum_Mt']
    Cumsum_emissions = Cumsum_emissions[['year','Cumulative GHG emissions (Mt)']]
    Cumsum_emissions.to_csv(f'{SAVE_DIR}/GHG_1_cunsum_emission_Mt.csv',index=False)

    # Plot_4-2: GHG from individual emission sectors (Mt)
    GHG_files_wide_onland = GHG_files_onland[['year','base_name','GHG_sum_Mt']]
    GHG_files_wide_offland = Emission_offland[['year','base_name','GHG_sum_Mt']]
    GHG_files_wide = pd.concat([GHG_files_wide_onland,GHG_files_wide_offland],axis=0)
    
    GHG_files_wide = GHG_files_wide\
        .groupby(['base_name'])[['year','GHG_sum_Mt']]\
        .apply(lambda x:list(map(list,zip(x['year'],x['GHG_sum_Mt']))))\
        .reset_index()
        
    GHG_files_wide.columns = ['name','data'] 
    GHG_files_wide['type'] = 'column'
    
    GHG_files_wide.loc[-1] = ['Net emissions', list(map(list,zip(Net_emission['year'],Net_emission['GHG_sum_Mt']))), 'line']
    GHG_files_wide.to_json(f'{SAVE_DIR}/GHG_2_individual_emission_Mt.json',orient='records')



    # Plot_4-3: GHG emission (Mt)
    GHG_emissions_long = get_GHG_category(GHG_files_onland,'Agricultural Landuse')
    GHG_emissions_long['Irrigation'] = GHG_emissions_long['Irrigation'].replace({'dry': 'Dryland', 'irr': 'Irrigated'})
    GHG_emissions_long['GHG Category'] = GHG_emissions_long['GHG Category'].replace({'CH4': 'Methane (CH4)', 
                                                                                     'N2O': 'Nitrous Oxide (N2O)', 
                                                                                     'CO2': 'Carbon Dioxide (CO2)'})

    # Plot_4-3-1: Agricultural Emission (on-land) by crop/lvstk sectors (Mt)
    GHG_Ag_emission_total_crop_lvstk = get_GHG_emissions_by_crop_lvstk_df(GHG_emissions_long)
    GHG_Ag_emission_total_crop_lvstk_wide = GHG_Ag_emission_total_crop_lvstk.pivot(index='Year', columns='Landuse_land_cat', values='Quantity (Mt CO2e)').reset_index()
    GHG_Ag_emission_total_crop_lvstk_wide.to_csv(f'{SAVE_DIR}/GHG_3_crop_lvstk_emission_Mt.csv',index=False)

    # Plot_4-3-2: Agricultural Emission (on-land) by dry/irrigation  (Mt)
    GHG_Ag_emission_total_dry_irr = GHG_emissions_long.groupby(['Year','Irrigation']).sum()['Quantity (Mt CO2e)'].reset_index()
    GHG_Ag_emission_total_dry_irr_wide = GHG_Ag_emission_total_dry_irr.pivot(index='Year', columns='Irrigation', values='Quantity (Mt CO2e)').reset_index()
    GHG_Ag_emission_total_dry_irr_wide.to_csv(f'{SAVE_DIR}/GHG_4_dry_irr_emission_Mt.csv',index=False)

    # Plot_4-3-3: Agricultural Emission (on-land) by GHG type sectors (Mt)
    GHG_Ag_emission_total_GHG_type = GHG_emissions_long.groupby(['Year','GHG Category']).sum()['Quantity (Mt CO2e)'].reset_index()
    GHG_Ag_emission_total_GHG_type_wide = GHG_Ag_emission_total_GHG_type.pivot(index='Year', columns='GHG Category', values='Quantity (Mt CO2e)').reset_index()
    GHG_Ag_emission_total_GHG_type_wide.to_csv(f'{SAVE_DIR}/GHG_5_category_emission_Mt.csv',index=False)

    # Plot_4-3-4: Agricultural Emission (on-land) by Sources (Mt)
    GHG_Ag_emission_total_Source = GHG_emissions_long.groupby(['Year','Sources']).sum()['Quantity (Mt CO2e)'].reset_index()
    GHG_Ag_emission_total_Source_wide = GHG_Ag_emission_total_Source.pivot(index='Year', columns='Sources', values='Quantity (Mt CO2e)').reset_index()
    GHG_Ag_emission_total_Source_wide.to_csv(f'{SAVE_DIR}/GHG_6_sources_emission_Mt.csv',index=False)


    # Plot_4-3-5: GHG emission (on-land) in start and end years (Mt)
    start_year,end_year = GHG_emissions_long['Year'].min(),GHG_emissions_long['Year'].max() 

    GHG_lu_lm = GHG_emissions_long\
            .groupby(['Year','Land use category','Land use','Irrigation'])\
            .sum()['Quantity (Mt CO2e)']\
            .reset_index()
            
    GHG_lu_lm_df_start = GHG_lu_lm.query('Year == @start_year').reset_index(drop=True)
    GHG_lu_lm_df_end = GHG_lu_lm.query('Year == @end_year').reset_index(drop=True)

    GHG_lu_lm_df_begin_end = pd.concat([GHG_lu_lm_df_start,GHG_lu_lm_df_end],axis=0)
    GHG_lu_lm_df_begin_end_wide = GHG_lu_lm_df_begin_end.pivot(index=['Year','Irrigation'], columns='Land use', values='Quantity (Mt CO2e)').reset_index()
    GHG_lu_lm_df_begin_end_wide['Irrigation'] = GHG_lu_lm_df_begin_end_wide.apply(lambda x: f"{x['Irrigation']} ({x['Year']})", axis=1)
    GHG_lu_lm_df_begin_end_wide.to_csv(f'{SAVE_DIR}/GHG_7_lu_lm_emission_Mt_wide.csv',index=False)


    # Plot_4-3-6: GHG emission (on-land) in the target year (Mt)
    GHG_lu_source = GHG_emissions_long\
                    .groupby(['Year','Land use','Irrigation','Sources'])\
                    .sum()['Quantity (Mt CO2e)']\
                    .reset_index()
            
    GHG_lu_source_target_yr = GHG_lu_source.query(f'Year == {end_year}')
    GHG_lu_source_nest_dict = target_GHG_2_Json(GHG_lu_source_target_yr)
                
    # save as json
    with open(f'{SAVE_DIR}/GHG_8_lu_source_emission_Mt.json', 'w') as outfile:
        json.dump(GHG_lu_source_nest_dict, outfile)
      
        
        
        
    # Plot_4-3-7: GHG emission (off-land) by commodity (Mt)
    GHG_off_land_commodity = GHG_off_land\
        .groupby(['year','COMMODITY'])\
        .sum(numeric_only=True)['GHG_sum_Mt'].reset_index()
        
    GHG_off_land_commodity_json = GHG_off_land_commodity\
        .groupby('COMMODITY')[['year','GHG_sum_Mt']]\
        .apply(lambda x:list(map(list,zip(x['year'],x['GHG_sum_Mt']))))\
        .reset_index()
        
    GHG_off_land_commodity_json.columns = ['name','data']
    GHG_off_land_commodity_json['type'] = 'column'
    GHG_off_land_commodity_json.to_json(f'{SAVE_DIR}/GHG_4_3_7_off_land_commodity_emission_Mt.json',orient='records')
    
    
    
    # Plot_4-3-8: GHG emission (off-land) by sources (Mt)
    GHG_off_land_sources = GHG_off_land\
        .groupby(['year','Emission Source'])\
        .sum(numeric_only=True)['GHG_sum_Mt'].reset_index()
        
    GHG_off_land_sources_json = GHG_off_land_sources\
        .groupby('Emission Source')[['year','GHG_sum_Mt']]\
        .apply(lambda x:list(map(list,zip(x['year'],x['GHG_sum_Mt']))))\
        .reset_index()
        
    GHG_off_land_sources_json.columns = ['name','data']
    GHG_off_land_sources_json['type'] = 'column'
    GHG_off_land_sources_json.to_json(f'{SAVE_DIR}/GHG_4_3_8_off_land_sources_emission_Mt.json',orient='records')
    
    
    
    
    # Plot_4-3-9: GHG emission (off-land) by Emission Type
    GHG_off_land_type = GHG_off_land\
        .groupby(['year','Emission Type'])\
        .sum(numeric_only=True)['GHG_sum_Mt'].reset_index()
        
    GHG_off_land_type_json = GHG_off_land_type\
        .groupby('Emission Type')[['year','GHG_sum_Mt']]\
        .apply(lambda x:list(map(list,zip(x['year'],x['GHG_sum_Mt']))))\
        .reset_index()
        
    GHG_off_land_type_json.columns = ['name','data']
    GHG_off_land_type_json['type'] = 'column'
    
    GHG_off_land_type_json.to_json(f'{SAVE_DIR}/GHG_4_3_9_off_land_type_emission_Mt.json',orient='records')



    # Plot_4-4: GHG sequestrations by Non-Agrilcultural sector (Mt)
    Non_ag_reduction_long = get_GHG_category(GHG_files_onland,'Non-Agricultural Landuse')

    # Create a fake data in year BASE_YEAR with all Quantity (Mt CO2e) as 0 values
    Non_ag_reduction_long_fake = Non_ag_reduction_long.copy()
    Non_ag_reduction_long_fake['Year'] = YR_BASE
    Non_ag_reduction_long_fake['Quantity (Mt CO2e)'] = 0

    # Concatenate the fake data with the real data
    Non_ag_reduction_long = pd.concat([Non_ag_reduction_long,Non_ag_reduction_long_fake],axis=0)

    Non_ag_reduction_total = Non_ag_reduction_long.groupby(['Year','Land use category'])\
        .sum()['Quantity (Mt CO2e)'].reset_index()

    Non_ag_reduction_source = Non_ag_reduction_long.groupby(['Year','Land use category','Sources'])\
        .sum()['Quantity (Mt CO2e)'].reset_index()    
        
    Non_ag_reduction_total_wide = Non_ag_reduction_total.pivot(index='Year', columns='Land use category', values='Quantity (Mt CO2e)').reset_index()
    Non_ag_reduction_total_wide.to_csv(f'{SAVE_DIR}/GHG_9_1_ag_reduction_total_wide_Mt.csv',index=False)

    Non_ag_reduction_source_wide = Non_ag_reduction_source.pivot(index='Year', columns='Sources', values='Quantity (Mt CO2e)').reset_index()
    Non_ag_reduction_source_wide.to_csv(f'{SAVE_DIR}/GHG_9_2_ag_reduction_source_wide_Mt.csv',index=False)



    # Plot_4-5: GHG reductions by Agricultural managements (Mt)
    Ag_man_sequestration_long = get_GHG_category(GHG_files_onland,'Agricultural Management')
    Ag_man_sequestration_long['Irrigation'] = Ag_man_sequestration_long['Irrigation'].replace({'dry': 'Dryland', 'irr': 'Irrigated'}) 

    # Plot_4-5-1: GHG reductions by Agricultural managements in total (Mt)
    Ag_man_sequestration_total = Ag_man_sequestration_long.groupby(['Year','GHG Category']).sum()['Quantity (Mt CO2e)'].reset_index()
    Ag_man_sequestration_total_wide = Ag_man_sequestration_total.pivot(index='Year',columns='GHG Category',values='Quantity (Mt CO2e)').reset_index()
    Ag_man_sequestration_total_wide.to_csv(f'{SAVE_DIR}/GHG_10_GHG_ag_man_df_wide_Mt.csv',index=False)


    # Plot_4-5-2: GHG reductions by Agricultural managements in subsector (Mt)
    Ag_man_sequestration_crop_lvstk_wide = Ag_man_sequestration_long.groupby(['Year','Land use category','Land category']).sum()['Quantity (Mt CO2e)'].reset_index()
    Ag_man_sequestration_crop_lvstk_wide['Landuse_land_cat'] = Ag_man_sequestration_crop_lvstk_wide.apply(lambda x: (x['Land use category'] + ' - ' + x['Land category']) 
                                    if (x['Land use category'] != x['Land category']) else x['Land use category'], axis=1)

    Ag_man_sequestration_crop_lvstk_wide = Ag_man_sequestration_crop_lvstk_wide.pivot(index='Year',columns='Landuse_land_cat',values='Quantity (Mt CO2e)').reset_index()
    Ag_man_sequestration_crop_lvstk_wide.to_csv(f'{SAVE_DIR}/GHG_11_GHG_ag_man_GHG_crop_lvstk_df_wide_Mt.csv',index=False)


    # Plot_4-5-3: GHG reductions by Agricultural managements in subsector (Mt)
    Ag_man_sequestration_dry_irr_total = Ag_man_sequestration_long.groupby(['Year','Irrigation']).sum()['Quantity (Mt CO2e)'].reset_index()
    Ag_man_sequestration_dry_irr_wide = Ag_man_sequestration_dry_irr_total.pivot(index='Year',columns='Irrigation',values='Quantity (Mt CO2e)').reset_index()
    Ag_man_sequestration_dry_irr_wide.to_csv(f'{SAVE_DIR}/GHG_12_GHG_ag_man_dry_irr_df_wide_Mt.csv',index=False)


    ####################################################
    #                     5) Water                     #
    ####################################################
    water_paths_total = files.query('category == "water" and year_types == "single_year" and ~base_name.str.contains("separate")').reset_index(drop=True)
    water_paths_separate = files.query('category == "water" and year_types == "single_year" and base_name.str.contains("separate")').reset_index(drop=True)

    water_df_total = get_water_df(water_paths_total)
    water_df_separate = pd.concat([pd.read_csv(path) for path in water_paths_separate['path']], ignore_index=True)
    water_df_separate['Water Use (ML)'] = water_df_separate['Water Use (ML)'].astype(float)
    water_df_separate['Irrigation'] = water_df_separate['Irrigation'].replace({'dry': 'Dryland', 'irr': 'Irrigated'}) 

    # Plot_5-1: Water use compared to limite (%)
    water_df_total_pct_wide = water_df_total.pivot(index='year', columns='REGION_NAME', values='PROPORTION_%')
    water_df_total_pct_wide.to_csv(f'{SAVE_DIR}/water_1_percent_to_limit.csv')

    # Plot_5-2: Water use compared to limite (ML)
    water_df_total_vol_wide = water_df_total.pivot(index='year', columns='REGION_NAME', values='TOT_WATER_REQ_ML')
    water_df_total_vol_wide.to_csv(f'{SAVE_DIR}/water_2_volume_to_limit.csv')

    # Plot_5-3: Water use by sector (ML)
    water_df_separate_lu_type = water_df_separate.groupby(['year','Landuse Type']).sum()[['Water Use (ML)']].reset_index()
    water_df_net = water_df_separate.groupby('year').sum(numeric_only=True).reset_index()

    water_df_separate_lu_type = water_df_separate_lu_type\
        .groupby(['Landuse Type'])[['Landuse Type','year','Water Use (ML)']]\
        .apply(lambda x:list(map(list,zip(x['year'],x['Water Use (ML)']))))\
        .reset_index()
        
    water_df_separate_lu_type.columns = ['name','data']
    water_df_separate_lu_type['type'] = 'column'

    water_df_separate_lu_type.loc[len(water_df_separate_lu_type)] = ['Net Volume', list(map(list,zip(water_df_net['year'],water_df_net['Water Use (ML)']))), 'line']

    water_df_separate_lu_type.to_json(f'{SAVE_DIR}/water_3_volume_by_sector.json',orient='records')



    # Plot_5-4: Water use by landuse (ML)
    water_df_seperate_lu = water_df_separate.groupby(['year','Landuse']).sum()[['Water Use (ML)']].reset_index()
    water_df_seperate_lu_wide = water_df_seperate_lu.pivot(index='year', columns='Landuse', values='Water Use (ML)').reset_index()
    # reorder the columns to match the order in LANDUSE_ALL
    water_df_seperate_lu_wide = water_df_seperate_lu_wide.reindex(
        columns = [water_df_seperate_lu_wide.columns[0]] + LANDUSE_ALL).reset_index(drop=True)
    water_df_seperate_lu_wide.to_csv(f'{SAVE_DIR}/water_4_volume_by_landuse.csv',index=False)

    # Plot_5-5: Water use by irrigation (ML)
    water_df_seperate_irr = water_df_separate.groupby(['year','Irrigation']).sum()[['Water Use (ML)']].reset_index()
    water_df_seperate_irr_wide = water_df_seperate_irr.pivot(index='year', columns='Irrigation', values='Water Use (ML)').reset_index()
    water_df_seperate_irr_wide.to_csv(f'{SAVE_DIR}/water_5_volume_by_irrigation.csv',index=False)



    #########################################################
    #                   6) Biodiversity                     #
    #########################################################

    # get biodiversity dataframe
    bio_paths = files.query('category == "biodiversity" and year_types == "single_year" and base_name == "biodiversity_separate"').reset_index(drop=True)
    bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
    bio_df['Biodiversity score (million)'] = bio_df['Biodiversity score'] / 1e6
    bio_df['Land management'] = bio_df['Land management'].replace({'dry': 'Dryland', 'irr': 'Irrigated'})

    # Filter out landuse that are reated to biodiversity
    bio_lucc = LU_NATURAL + NON_AG_LANDUSE


    # Plot_6-1: Biodiversity total by category
    bio_df_category = bio_df.groupby(['Year','Landuse type']).sum(numeric_only=True).reset_index()
    # bio_df_category_wide = bio_df_category.pivot(index='Year', columns='Landuse type', values='Biodiversity score (million)').reset_index()
    # bio_df_category_wide.to_csv(f'{SAVE_DIR}/biodiversity_1_total_score_by_category.csv',index=False)
    bio_df_category = bio_df_category\
        .groupby('Landuse type')[['Year','Biodiversity score (million)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Biodiversity score (million)']))))\
        .reset_index()
        
    bio_df_category.columns = ['name','data']
    bio_df_category['type'] = 'column'
    bio_df_category.to_json(f'{SAVE_DIR}/biodiversity_1_total_score_by_category.json',orient='records')


    # Plot_6-2: Biodiversity total by irrigation
    bio_df_irrigation = bio_df.groupby(['Year','Land management']).sum(numeric_only=True).reset_index()
    bio_df_irrigation = bio_df_irrigation\
        .groupby('Land management')[['Year','Biodiversity score (million)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Biodiversity score (million)']))))\
        .reset_index()
        
    bio_df_irrigation.columns = ['name','data']
    bio_df_irrigation['type'] = 'column'
    bio_df_irrigation.to_json(f'{SAVE_DIR}/biodiversity_2_total_score_by_irrigation.json',orient='records')


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
    natural_land_area = area_dvar.groupby(['Year','Land use']).sum(numeric_only=True).reset_index()
    natural_land_area = natural_land_area.query('`Land use` in @bio_lucc').copy()

    # million km2 to million ha
    natural_land_area.loc[:,'Area (million ha)'] = natural_land_area['Area (ha)'] / 1e6
    natural_land_area = natural_land_area\
        .groupby('Land use')[['Year','Area (million ha)']]\
        .apply(lambda x:list(map(list,zip(x['Year'],x['Area (million ha)']))))\
        .reset_index()
        
    natural_land_area['sort_index'] = natural_land_area['Land use'].apply(lambda x: bio_lucc.index(x))
    natural_land_area = natural_land_area.sort_values(['sort_index']).drop('sort_index',axis=1)

    natural_land_area.columns = ['name','data']
    natural_land_area['type'] = 'column'
    natural_land_area.to_json(f'{SAVE_DIR}/biodiversity_4_natural_land_area.json',orient='records')
    
    
    #########################################################
    #                         7) Maps                       #
    #########################################################
    map_files = files.query('base_ext == ".html" and year_types != "begin_end_year"')
    map_save_dir = f"{SAVE_DIR}/Map_data/"
    
    # Create the directory to save map_html if it does not exist
    if  os.path.exists(map_save_dir):
        # Delete the existing directory and the files in it
        if os.name == 'nt':  
            os.system(f'rd /s /q "{map_save_dir}"') # If the system is Windows
        else:  
            os.system(f'rm -rf {map_save_dir}') # If the system is Unix-based
        # Create new directory
        os.makedirs(map_save_dir)
    else:
        os.makedirs(map_save_dir)
    
    # Function to move a file from one location to another if the file exists
    def move_html(path_from, path_to):
        if os.path.exists(path_from):
            shutil.move(path_from, path_to)
    
    # Move the map files to the save directory
    tasks = [delayed(move_html)(row['path'], map_save_dir)
                for _,row in map_files.iterrows()]
    
    worker = min(settings.WRITE_THREADS, len(tasks))
    
    Parallel(n_jobs=worker)(tasks)
    
    


    #########################################################
    #              Report success info                      #
    #########################################################

    print('Report data created successfully!\n')