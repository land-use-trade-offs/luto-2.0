import os
import json
import numpy as np
import pandas as pd
import plotnine as p9

from luto.tools.create_task_runs.parameters import TASK_ROOT_DIR
from luto.tools.report.data_tools import get_all_files


TASK_ROOT_DIR = "N:/LUF-Modelling/LUTO2_JZ/Custom_runs"

run_dirs = f"{TASK_ROOT_DIR}/20241206_Deviation_Profit_tradeoffs"

grid_search_params = pd.read_csv(f"{run_dirs}/grid_search_parameters.csv")



# Get the last run directory
report_data = pd.DataFrame()

for _, row in grid_search_params.iterrows():
    
    # Get the last run directory
    run_dir = f"{run_dirs}/Run_{row['run_idx']}/output"
    data_dir = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))][-1]
    json_path = f"{run_dir}/{data_dir}/DATA_REPORT/data"
    
    
    # Get the profit data
    with open(f"{json_path}/economics_0_rev_cost_all_wide.json") as f:
        data = json.load(f)
        
    df = pd.json_normalize(data, 'data', ['name', 'type'])\
           .rename(columns={0: 'year', 1: 'val'})
    df_profit = df.query('name == "Profit"')
    

    # Get the GHG deviation 
    with open(f"{json_path}/GHG_2_individual_emission_Mt.json") as f:
        data = json.load(f)
        
    df = pd.json_normalize(data, 'data', ['name', 'type'])\
           .rename(columns={0: 'year', 1: 'val'})
    df_target = df.query('name == "GHG emissions limit"')
    df_actual = df.query('name == "Net emissions"')
    df_deviation = df_target.merge(df_actual, on='year', suffixes=('_target', '_actual'))
    df_deviation['name'] = 'GHG deviation'
    df_deviation['val'] = df_deviation['val_actual'] - df_deviation['val_target']
    
    # Combine the data
    report_data = pd.concat([
        report_data,
        df_profit[['year','name', 'val']].assign(**row),
        df_deviation[['year','name', 'val']].assign(**row)
    ]).reset_index(drop=True)


# Pivot the data
report_data_wide = report_data.pivot(index=['year', 'run_idx', 'SOLVE_WEIGHT_DEVIATIONS'], columns='name', values='val').reset_index()      
report_data_wide = report_data_wide.query('year != "2010"')


# Plot the data
p = (p9.ggplot(report_data_wide, 
               p9.aes(x='GHG deviation', 
                      y='Profit', 
                      color='SOLVE_WEIGHT_DEVIATIONS')
               ) +
     p9.geom_point() +
     p9.theme_bw() +
     p9.theme(figure_size=(10, 5))
    )              
    




