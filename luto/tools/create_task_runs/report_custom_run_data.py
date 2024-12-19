import os
import json
import pandas as pd
import plotnine as p9
from glob import glob
from luto.tools.create_task_runs.parameters import TASK_ROOT_DIR


def load_json_data(json_path, filename):
    with open(os.path.join(json_path, filename)) as f:
        return pd.json_normalize(json.load(f), 'data', ['name']).rename(columns={0: 'year', 1: 'val'})

def process_profit_data(json_path):
    df = load_json_data(json_path, 'economics_0_rev_cost_all_wide.json')
    return df.query('name == "Profit"')

def process_GHG_deviation_data(json_path):
    df = load_json_data(json_path, 'GHG_2_individual_emission_Mt.json')
    df_target = df.query('name == "GHG emissions limit"')
    df_actual = df.query('name == "Net emissions"')
    df_deviation = df_target.merge(df_actual, on='year', suffixes=('_target', '_actual'))
    df_deviation['name'] = 'GHG deviation'
    df_deviation['val'] = df_deviation['val_actual'] - df_deviation['val_target']
    return df_deviation

def process_GHG_deforestation_data(json_path):
    df = load_json_data(json_path, 'GHG_2_individual_emission_Mt.json') 
    return df.query('name == "Deforestation"')


def process_demand_data(json_path):
    df_demand = load_json_data(json_path, 'production_5_6_demand_Production_commodity_from_LUTO.json')
    df_luto = load_json_data(json_path, 'production_5_5_demand_Production_commodity.json')
    df_delta = df_demand.merge(df_luto, on=['year', 'name'], suffixes=('_luto', '_demand'))
    df_delta['deviation_t'] = df_delta.eval('val_luto - val_demand')
    df_delta['deviation_%'] = df_delta.eval('(val_luto - val_demand) / val_demand * 100')
    return df_delta

def process_task_root_dirs(task_root_dirs):
    report_data = pd.DataFrame()
    report_data_demand = pd.DataFrame()

    for task_root_dir in task_root_dirs:
        grid_search_params = pd.read_csv(f"{task_root_dir}/grid_search_parameters.csv")
        run_dirs = [i for i in os.listdir(task_root_dir) if os.path.isdir(os.path.join(task_root_dir, i))]

        for dir in run_dirs:
            run_idx = int(dir.split('_')[-1])
            run_paras = grid_search_params.query(f'run_idx == {run_idx}').to_dict(orient='records')[0]
            json_path = os.path.join(task_root_dir, dir, 'DATA_REPORT', 'data')

            if not os.path.exists(json_path):
                print(f"Path does not exist: {json_path}")
                continue

            df_profit = process_profit_data(json_path)
            df_ghg_deviation = process_GHG_deviation_data(json_path)
            df_ghg_deforestation = process_GHG_deforestation_data(json_path)
            df_demand_deviation = process_demand_data(json_path)

            report_data = pd.concat([
                report_data,
                df_profit[['year', 'name', 'val']].assign(**run_paras),
                df_ghg_deviation[['year', 'name', 'val']].assign(**run_paras),
                df_ghg_deforestation[['year', 'name', 'val']].assign(**run_paras)
            ]).reset_index(drop=True)

            report_data_demand = pd.concat([
                report_data_demand,
                df_demand_deviation.assign(**run_paras)
            ]).reset_index(drop=True)
            
    # Pivot the report data so that the `name` columns is split into separate columns
    report_data = report_data\
        .pivot(
            index=['year'] + grid_search_params.columns.tolist(), 
            columns='name', 
            values='val')\
        .reset_index()

    return report_data, report_data_demand




# Get the data
task_root_dirs = [i for i in glob('../*') if "Timeseries_RES20_REVERSEBILITY_TEST" in i]
report_data, report_data_demand = process_task_root_dirs(task_root_dirs)

# Filter the data
ghg_order = ['1.5C (67%) excl. avoided emis', '1.5C (50%) excl. avoided emis', '1.8C (67%) excl. avoided emis']
report_data['GHG_LIMITS_FIELD'] = pd.Categorical(report_data['GHG_LIMITS_FIELD'], categories=ghg_order, ordered=True)
report_data_demand['GHG_LIMITS_FIELD'] = pd.Categorical(report_data_demand['GHG_LIMITS_FIELD'], categories=ghg_order, ordered=True)


report_data_filter = report_data.loc[
    (report_data['year'] != 2010) &
    (report_data['GHG_CONSTRAINT_TYPE'] == "soft") &
    (report_data['BIODIVERSITY_LIMITS'] == "on") &
    (report_data['SOLVE_ECONOMY_WEIGHT'] >=0) &
    (report_data['SOLVE_ECONOMY_WEIGHT'] <=0.3)
].copy()

report_data_demand_filterd = report_data_demand.loc[
    (report_data_demand['year'] != 2010) &
    # (report_data_demand['deviation_%'].abs() > 1) &
    (report_data_demand['GHG_CONSTRAINT_TYPE'] == "soft") &
    (report_data_demand['BIODIVERSITY_LIMITS'] == "on") &
    (report_data['SOLVE_ECONOMY_WEIGHT'] >=0) &
    (report_data['SOLVE_ECONOMY_WEIGHT'] <=0.3)
].copy()




# Plotting
p9.options.figure_size = (15, 8)
p9.options.dpi = 300


# Time series
p_weight_vs_profit = (
    p9.ggplot(
        report_data_filter, 
        p9.aes(
            x='year', 
            y='Profit', 
            color='SOLVE_ECONOMY_WEIGHT', 
            # linetype='DIET_GLOB',
            group='SOLVE_ECONOMY_WEIGHT',
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_line(size=0.3) +
    p9.theme_bw() +
    # p9.scale_x_log10() +
    p9.ylab('Profit (billion AUD)')
    )


p_weight_vs_GHG_deviation = (
    p9.ggplot(
        report_data_filter, 
        p9.aes(
            x='year', 
            y='GHG deviation', 
            color='SOLVE_ECONOMY_WEIGHT', 
            # linetype='DIET_GLOB',
            group='SOLVE_ECONOMY_WEIGHT',
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_line() +
    p9.theme_bw() +
    # p9.scale_x_log10() +
    p9.ylab('GHG deviation (Mt)')
    )

p_weight_vs_GHG_deforestation = (
    p9.ggplot(
        report_data_filter, 
        p9.aes(
            x='year', 
            y='Deforestation', 
            color='SOLVE_ECONOMY_WEIGHT', 
            # linetype='DIET_GLOB',
            group='SOLVE_ECONOMY_WEIGHT',
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_line() +
    p9.theme_bw() +
    # p9.scale_x_log10() +
    p9.ylab('Deforestation (Mt)')
    )


p_weigth_vs_demand = (
    p9.ggplot(
        report_data_demand_filterd, 
        p9.aes(
            x='year', 
            y='deviation_%', 
            fill='name', 
            group='SOLVE_ECONOMY_WEIGHT'
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_col(position='dodge') +
    p9.theme_bw() +
    # p9.scale_x_log10() +
    p9.guides(fill=p9.guide_legend(ncol=1))
)








# Snapshoot plots
p_weight_vs_profit = (
    p9.ggplot(
        report_data_filter, 
        p9.aes(
            x='SOLVE_ECONOMY_WEIGHT', 
            y='Profit',
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_line(size=0.3) +
    p9.theme_bw() +
    # p9.scale_x_log10() +
    p9.ylab('Profit (billion AUD)')
    )


p_weight_vs_GHG_deviation = (
    p9.ggplot(
        report_data_filter, 
        p9.aes(
            x='SOLVE_ECONOMY_WEIGHT', 
            y='GHG deviation'
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_line() +
    p9.theme_bw() +
    # p9.scale_x_log10() +
    p9.ylab('GHG deviation (Mt)')
    )

p_weight_vs_GHG_deforestation = (
    p9.ggplot(
        report_data_filter, 
        p9.aes(
            x='SOLVE_ECONOMY_WEIGHT', 
            y='Deforestation', 
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_line() +
    p9.theme_bw() +
    # p9.scale_x_log10() +
    p9.ylab('Deforestation (Mt)')
    )


p_GHG_vs_profit = (
    p9.ggplot(report_data_filter, 
        p9.aes(
            x='GHG deviation', 
            y='Profit', 
            color='SOLVE_ECONOMY_WEIGHT', 
            shape='DIET_GLOB',
            group='interaction',
        )
        
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_point(size=0.1) +
    p9.theme_bw()
    )


p_weigth_vs_demand = (
    p9.ggplot(
        report_data_demand_filterd, 
        p9.aes(
            x='SOLVE_ECONOMY_WEIGHT', 
            y='deviation_%', 
            fill='name', 
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_col(position='dodge') +
    p9.theme_bw() +
    p9.scale_x_log10() +
    p9.guides(fill=p9.guide_legend(ncol=1))
)


