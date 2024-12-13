import os
import json
import pandas as pd
import plotnine as p9
from luto.tools.create_task_runs.parameters import TASK_ROOT_DIR


def load_json_data(json_path, filename):
    with open(os.path.join(json_path, filename)) as f:
        return pd.json_normalize(json.load(f), 'data', ['name']).rename(columns={0: 'year', 1: 'val'})

def get_run_parameters(run_idx, grid_search_params):
    return grid_search_params.query(f'run_idx == {run_idx}').to_dict(orient='records')[0]

def process_profit_data(json_path):
    df = load_json_data(json_path, 'economics_0_rev_cost_all_wide.json')
    return df.query('name == "Profit"')

def process_ghg_deviation_data(json_path):
    df = load_json_data(json_path, 'GHG_2_individual_emission_Mt.json')
    df_target = df.query('name == "GHG emissions limit"')
    df_actual = df.query('name == "Net emissions"')
    df_deviation = df_target.merge(df_actual, on='year', suffixes=('_target', '_actual'))
    df_deviation['name'] = 'GHG deviation'
    df_deviation['val'] = df_deviation['val_actual'] - df_deviation['val_target']
    return df_deviation

def process_demand_data(json_path):
    df_demand = load_json_data(json_path, 'production_5_6_demand_Production_commodity_from_LUTO.json')
    df_luto = load_json_data(json_path, 'production_5_5_demand_Production_commodity.json')
    df_delta = df_demand.merge(df_luto, on=['year', 'name'], suffixes=('_luto', '_demand'))
    df_delta['deviation_t'] = df_delta.eval('val_luto - val_demand')
    df_delta['deviation_%'] = df_delta.eval('(val_luto - val_demand) / val_demand * 100')
    return df_delta


# Load the grid search parameters
grid_search_params = pd.read_csv(f"{TASK_ROOT_DIR}/grid_search_parameters.csv")
grid_paras = set(grid_search_params.columns.tolist()) - set(['MEM', 'NCPUS', 'MODE'])
run_dirs = [i for i in os.listdir(TASK_ROOT_DIR) if os.path.isdir(os.path.join(TASK_ROOT_DIR, i))]

# Loop through the run directories and extract the data
report_data = pd.DataFrame()
report_data_demand = pd.DataFrame()

for dir in run_dirs:
    run_idx = int(dir.split('_')[-1])
    run_paras = get_run_parameters(run_idx, grid_search_params)
    json_path = os.path.join(TASK_ROOT_DIR, dir, 'DATA_REPORT', 'data')

    if not os.path.exists(json_path):
        print(f"Path does not exist: {json_path}")
        continue

    df_profit = process_profit_data(json_path)
    df_deviation = process_ghg_deviation_data(json_path)
    df_delta = process_demand_data(json_path)

    report_data = pd.concat([
        report_data,
        df_profit[['year', 'name', 'val']].assign(**run_paras),
        df_deviation[['year', 'name', 'val']].assign(**run_paras)
    ]).reset_index(drop=True)

    report_data_demand = pd.concat([
        report_data_demand,
        df_delta.assign(**run_paras)
    ]).reset_index(drop=True)




# Plot the data
report_data_wide = report_data\
    .pivot(index=['year'] + list(grid_paras), columns='name', values='val')\
    .reset_index()\
    .query('year != 2010 and SOLVE_ECONOMY_WEIGHT <= 1 and SOLVE_ECONOMY_WEIGHT >= 0')

report_data_demand_filterd = report_data_demand \
    .query('year != 2010 and SOLVE_ECONOMY_WEIGHT <= 0.3 and SOLVE_ECONOMY_WEIGHT >= 0.2') \
    .sort_values('deviation_%')
    
    
p9.options.figure_size = (15, 8)
p9.options.dpi = 150


p_weight_vs_profit = (
    p9.ggplot(report_data_wide, p9.aes(x='SOLVE_ECONOMY_WEIGHT', y='Profit', color='DIET_DOM')) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_point() +
    p9.theme_bw() +
    # p9.scale_x_log10() +
    p9.ylab('Profit (billion AUD)')
    )

p_weight_vs_ghg = (
    p9.ggplot(report_data_wide, p9.aes(x='SOLVE_ECONOMY_WEIGHT', y='GHG deviation', color='DIET_DOM')) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_point() +
    p9.theme_bw() +
    # p9.scale_x_log10() +
    p9.ylab('GHG deviation (Mt)')
    )


p_ghg_vs_profit = (
    p9.ggplot(report_data_wide, p9.aes(x='GHG deviation', y='Profit', color='SOLVE_ECONOMY_WEIGHT')) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_point() +
    p9.scale_color_continuous(trans='log10') +
    p9.theme_bw()
    )


p_weigth_vs_demand = (
    p9.ggplot(report_data_demand_filterd, p9.aes(x='SOLVE_ECONOMY_WEIGHT', y='deviation_%', fill='name', group='DIET_DOM')) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_col(position='dodge') +
    p9.theme_bw() +
    p9.scale_x_log10() +
    p9.guides(fill=p9.guide_legend(ncol=1))
)

p_weigth_vs_demand = (
    p9.ggplot(report_data_demand_filterd, p9.aes(x='SOLVE_ECONOMY_WEIGHT', y='deviation_%', fill='DIET_DOM', pattern='name')) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_col(position='dodge') +
    p9.theme_bw() +
    p9.scale_x_log10() +
    p9.guides(fill=p9.guide_legend(ncol=1))
)