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


import plotnine as p9
from luto.tools.create_task_runs.create_grid_search_tasks import TASK_ROOT_DIR
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Plot settings
p9.options.figure_size = (12, 6)
p9.options.dpi = 300


# Get the data
task_root_dir = TASK_ROOT_DIR.rstrip('/')       # Or replace with the desired task run root dir

task_root_dir = '../Custom_runs/20250610_RES13_TEST_BETA_DEVIATION_AUD'
task_root_dir = '/g/data/jk53/jinzhu/LUTO/Custom_runs/20250610_RES13_TEST_BETA_DEVIATION_T'

report_data = process_task_root_dirs(task_root_dir)

print(report_data['Type'].unique())





# -------------- Plot demand deviation ----------------------
demand_df = report_data.query('Type == "Production_deviation_pct"').copy()

df_demand_avg = demand_df.eval('val = abs(val)'
    ).groupby(['SOLVE_WEIGHT_BETA']
    )[['val','run_idx']].agg(val=('val', 'mean'), run_idx=('run_idx', 'first')
    ).reset_index()



# Plot profit landscape without filtering
plot_landscape_profit = (
    p9.ggplot(
        df_demand_avg,
        p9.aes(
            x='SOLVE_WEIGHT_BETA', 
            y='val', 
        )
    ) +
    p9.geom_point(size=3,stroke=0) +
    p9.theme_bw() +
    # p9.facet_grid('GHG_EMISSIONS_LIMITS~BIODIVERSTIY_TARGET_GBF_2') +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)

# Plot individual demand landscape 
query_str = '''
    year == 2050
    '''.replace('\n', ' ').replace('  ', ' ')

demand_df_individual = demand_df.query(query_str).copy()

plot_landscape_profit = (
    p9.ggplot(
        demand_df_individual,
        p9.aes(
            x='SOLVE_WEIGHT_BETA', 
            y='val', 
        )
    ) +
    p9.geom_point(size=0.1) +
    p9.geom_vline(xintercept=0.9,color='red') +
    p9.theme_bw() +
    p9.facet_wrap('name', scales='free_y') +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) +
    p9.labs(
        x='Beta (B)', 
        y='Demand deviation (%)',
    )
)


# -------------- Plot profit ----------------------
query_str = '''
    name == "Profit" 
    '''.replace('\n', ' ').replace('  ', ' ')
    
df_profit = report_data.query(query_str).copy()



# Plot individual profit landscape 
query_str = '''
    abs(val) < 30
    '''.replace('\n', ' ').replace('  ', ' ')
    
df_profit_individual = df_profit.query(query_str).copy()

plot_landscape_profit = (
    p9.ggplot(
        df_profit_individual,
        p9.aes(
            x='SOLVE_WEIGHT_BETA', 
            y='val', 
        )
    ) +
    p9.geom_point(size=0.1) +
    p9.geom_vline(xintercept=0.9, color='red') +
    p9.theme_bw() +
    p9.facet_wrap('year', scales='free_y') +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) +
    p9.labs(
        x='Beta (B)', 
        y='Profit (million AUD)',
    )
)






# -------------- Plot total transition cost ----------------------
query_str = '''
    name == "Transition cost (Ag2Ag)" 
    and year != 2010
    '''.replace('\n', ' ').replace('  ', ' ')
    
df_profit = report_data.query(query_str).copy()

valid_runs_profit = set(report_data['run_idx']) - set(
    df_profit.query('abs(val) >= 30')['run_idx']
)

df_profit = df_profit.query('run_idx.isin(@valid_runs_profit)')

# Plot economic's deviation landscape without filtering
plot_landscape_demand = (
    p9.ggplot(
        df_profit,
        p9.aes(
            x='year',
            y='val', 
            color='SOLVE_WEIGHT_BETA',
            group='run_idx',
        )
    ) +
    p9.geom_line() +
    p9.facet_wrap('TRANSITION_HURDEL_FACTOR') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)





# -------------- Plot individual transition cost ----------------------

query_str = '''
    Type == "Transition_cost_million_AUD" 
    and year == 2020
    '''.replace('\n', ' ').replace('  ', ' ')


df_trans_costs = report_data.query(query_str).copy()
df_trans_costs['val'] = -df_trans_costs['val'] 
df_trans_costs['name_str'] = df_trans_costs['name'].map(lambda x: f'{x[0]} -> {x[1]}')


df_trans_costs_avg_beta = df_trans_costs.groupby(['name_str','SOLVE_WEIGHT_BETA']
    ).agg(val=('val','mean')).reset_index().query('val > 10')

df_trans_costs_avg_hurdel = df_trans_costs.groupby(['name_str','TRANSITION_HURDEL_FACTOR']
    ).agg(val=('val','mean')).reset_index().query('val > 100')


valid_trans = df_trans_costs_avg_hurdel.groupby('name_str')['val'].apply(
    lambda x: x.abs().max() > 500
).reset_index().query('val')['name_str'].tolist()


# Plot economic's deviation landscape without filtering
fig = (
    p9.ggplot(
        df_trans_costs_avg_beta.query('name_str.isin(@valid_trans)'),
        p9.aes(
            x='SOLVE_WEIGHT_BETA',
            y='val', 
        )
    ) +
    p9.geom_line() +
    p9.theme_bw() +
    p9.facet_wrap('name_str') +
    p9.theme(
        strip_text=p9.element_text(size=6), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)


fig = (
    p9.ggplot(
        df_trans_costs_avg_hurdel.query('name_str.isin(@valid_trans)'),
        p9.aes(
            x='TRANSITION_HURDEL_FACTOR',
            y='val', 
        )
    ) +
    p9.geom_line() +
    p9.theme_bw() +
    p9.facet_wrap('name_str') +
    p9.theme(
        strip_text=p9.element_text(size=6), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)




