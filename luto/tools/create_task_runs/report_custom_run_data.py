import os
import json
import pandas as pd
import plotnine as p9

from luto.tools.create_task_runs.parameters import TASK_ROOT_DIR

# Get the grid search parameters
grid_search_params = pd.read_csv(f"{TASK_ROOT_DIR}/grid_search_parameters.csv")
grid_paras = set(grid_search_params.columns.tolist()) - set(['MEM', 'NCPUS', 'MODE'])

# Get the list of all run directories
run_dirs = [i for i in os.listdir(TASK_ROOT_DIR) if os.path.isdir(os.path.join(TASK_ROOT_DIR, i))]


# Get the last run directory
report_data = pd.DataFrame()
report_data_demand = pd.DataFrame()

for dir in run_dirs:
    
    # Get the last run directory
    run_idx = int(dir.split('_')[-1])
    run_paras = grid_search_params.query(f'run_idx == {run_idx}').to_dict(orient='records')[0]
    json_path = f"{TASK_ROOT_DIR}/{dir}/DATA_REPORT/data"
    
    # Continue if the json path does not exist
    if not os.path.exists(json_path):
        print(f"Path does not exist: {json_path}")
        continue
    
    # Get the profit data
    with open(f"{json_path}/economics_0_rev_cost_all_wide.json") as f:
        df = pd.json_normalize(json.load(f) , 'data', ['name']).rename(columns={0: 'year', 1: 'val'})
        df_profit = df.query('name == "Profit"')
    
    # Get the GHG deviation 
    with open(f"{json_path}/GHG_2_individual_emission_Mt.json") as f:
        df = pd.json_normalize(json.load(f), 'data', ['name']).rename(columns={0: 'year', 1: 'val'})   
        df_target = df.query('name == "GHG emissions limit"')
        df_actual = df.query('name == "Net emissions"')
        df_deviation = df_target.merge(df_actual, on='year', suffixes=('_target', '_actual'))
        df_deviation['name'] = 'GHG deviation'
        df_deviation['val'] = df_deviation['val_actual'] - df_deviation['val_target']
    
    # Get the demand data
    with open(f"{json_path}/production_5_5_demand_Production_commodity.json") as f_demand, \
         open(f"{json_path}/production_5_6_demand_Production_commodity_from_LUTO.json") as f_luto:
        df_demand = pd.json_normalize(json.load(f_luto), 'data', ['name']).rename(columns={0: 'year', 1: 'val'})
        df_luto = pd.json_normalize(json.load(f_demand), 'data', ['name']).rename(columns={0: 'year', 1: 'val'})
        df_delta = df_demand.merge(df_luto, on=['year', 'name'], suffixes=('_luto', '_demand'))
        df_delta['deviation_t'] = df_delta.eval('val_luto - val_demand')
        df_delta['deviation_%'] = df_delta.eval('(val_luto - val_demand ) / val_demand * 100')
        
    # Combine the data
    report_data = pd.concat([
        report_data,
        df_profit[['year','name', 'val']].assign(**run_paras),
        df_deviation[['year','name', 'val']].assign(**run_paras)
    ]).reset_index(drop=True)
    
    report_data_demand = pd.concat([
        report_data_demand,
        df_delta.assign(**run_paras)
    ]).reset_index(drop=True)




# Pivot the data
report_data_wide = report_data\
    .pivot(index=['year'] + list(grid_paras), columns='name', values='val')\
    .reset_index() \
    .query('year != 2010')



# Plot the data
p9.options.figure_size = (15, 8)
p9.options.dpi = 150

p = (p9.ggplot(report_data_wide, 
               p9.aes(x='GHG deviation', 
                      y='Profit', 
                      color='SOLVE_ECONOMY_WEIGHT')
               ) +
     p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
     p9.geom_point() +
     p9.theme_bw() 
    )

p = (p9.ggplot(report_data_wide, 
               p9.aes(x='SOLVE_ECONOMY_WEIGHT', 
                      y='Profit')
               ) +
     p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
     p9.geom_point() +
     p9.theme_bw() +
     p9.ylab('Profit (billion AUD)') 
    )


p = (p9.ggplot(report_data_wide, 
               p9.aes(x='SOLVE_ECONOMY_WEIGHT', 
                      y='GHG deviation')
               ) +
     p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
     p9.geom_point() +
     p9.theme_bw() +
     p9.ylab('GHG deviation (Mt)')
    )


p = (p9.ggplot(report_data_demand, 
            p9.aes(x='SOLVE_ECONOMY_WEIGHT', 
                    y='deviation_t',
                    fill='name')
            ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_col() +
    p9.theme_bw() +
    p9.guides(fill=p9.guide_legend(ncol=1))
    )


