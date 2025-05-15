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


import pandas as pd
import numpy as np
import plotnine as p9
import plotly.express as px
from luto.tools.create_task_runs.create_grid_search_tasks import TASK_ROOT_DIR
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Plot settings
p9.options.figure_size = (12, 6)
p9.options.dpi = 100


# Get the data
task_root_dir = TASK_ROOT_DIR.rstrip('/')       # Or replace with the desired task run root dir
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250512_DCCEEW_REPORT_01_CONNECTIVITY_ONF_OFF/"
report_data = process_task_root_dirs(task_root_dir)


# -------------- Plot the weight's landscape map (Demand's deviation) ----------------------
query_str = '''
    Type == "Production_deviation_pct" 
    and year != 2010
    '''.replace('\n', ' ').replace('  ', ' ')

# Get demand data
df_demand = report_data.query(query_str).copy()

df_demand_avg = df_demand.eval('val = abs(val)'
    ).groupby(['year', 'GHG_LIMITS_FIELD', 'BIODIVERSTIY_TARGET_GBF_2', 'SOLVE_WEIGHT_ALPHA', 'SOLVE_WEIGHT_BETA']
    )[['val','run_idx']].agg(val=('val', 'mean'), run_idx=('run_idx', 'first')
    ).reset_index()


# Plot demand's deviation landscape without filtering
plot_landscape_demand = (
    p9.ggplot(
        df_demand_avg,
        p9.aes(
            x='SOLVE_WEIGHT_ALPHA', 
            y='SOLVE_WEIGHT_BETA', 
            fill='val', 
        )
    ) +
    p9.geom_point(stroke=0) +
    # p9.facet_wrap('name') +
    p9.facet_grid('BIODIVERSTIY_TARGET_GBF_2~GHG_LIMITS_FIELD') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)



# Plot demand's deviation landscape with filtering
commodity_out_targets = {
    'Beef live export',
    'Beef meat',
    'Sheep live export',
    'Sheep meat',
    'Sheep wool',
}

valid_runs_demand = set(report_data['run_idx']) - set(
    df_demand.query(
        'name not in @commodity_out_targets and abs(val) >= 10'
    )['run_idx']
)

df_demand_avg = df_demand.query('run_idx.isin(@valid_runs_demand)'
    ).eval('val = abs(val)'
    ).groupby(['year', 'GHG_LIMITS_FIELD', 'BIODIVERSTIY_TARGET_GBF_2', 'SOLVE_WEIGHT_ALPHA', 'SOLVE_WEIGHT_BETA']
    )[['val','run_idx']].agg(val=('val', 'mean'), run_idx=('run_idx', 'first')
    ).reset_index()

plot_landscape_demand = (
    p9.ggplot(
        df_demand_avg,
        p9.aes(
            x='SOLVE_WEIGHT_ALPHA', 
            y='SOLVE_WEIGHT_BETA', 
            fill='val', 
        )
    ) +
    p9.geom_point(stroke=0) +
    # p9.facet_wrap('name') +
    p9.facet_grid('BIODIVERSTIY_TARGET_GBF_2~GHG_LIMITS_FIELD') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)


# -------------- Plot the weight's landscape map (Profit in Billion AUD) ----------------------
query_str = '''
    Type == "Profit_billion_AUD" 
    and run_idx.isin(@valid_runs_demand)
    '''.replace('\n', ' ').replace('  ', ' ')

# Get profit data
df_profit = report_data.query(query_str).query('year == 2050')


# Plot profit landscape without filtering
plot_landscape_profit = (
    p9.ggplot(
        df_profit,
        p9.aes(
            x='SOLVE_WEIGHT_ALPHA', 
            y='SOLVE_WEIGHT_BETA', 
            fill='val', 
        )
    ) +
    p9.geom_point(size=3,stroke=0) +
    # p9.facet_wrap('name') +
    p9.facet_grid('BIODIVERSTIY_TARGET_GBF_2~GHG_LIMITS_FIELD') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)


# Plot profit landscape with filtering
valid_runs_demand_profit = valid_runs_demand - set(
     report_data.query(query_str).query('val <= -100')['run_idx']
)

df_profit_filtered = report_data.query(query_str).query('run_idx.isin(@valid_runs_demand_profit)')


plot_landscape_profit = (
    p9.ggplot(
        df_profit_filtered,
        p9.aes(
            x='SOLVE_WEIGHT_ALPHA', 
            y='SOLVE_WEIGHT_BETA', 
            fill='val', 
        )
    ) +
    p9.geom_point(size=3,stroke=0) +
    # p9.facet_wrap('name') +
    p9.facet_grid('BIODIVERSTIY_TARGET_GBF_2~GHG_LIMITS_FIELD') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)



# -------------- Plot the weight's landscape map (biodiversity score) ----------------------
query_str = '''
    Type == "Biodiversity_obj_score"
    '''.replace('\n', ' ').replace('  ', ' ')
    
df_bio = report_data.query(query_str).query('run_idx.isin(@valid_runs_demand_profit)').copy()

df_bio_sum = df_bio.eval('val = val / 1e3'
    ).groupby(['year', 'GHG_LIMITS_FIELD', 'BIODIVERSTIY_TARGET_GBF_2', 'SOLVE_WEIGHT_ALPHA', 'SOLVE_WEIGHT_BETA']
    )['val'].sum().reset_index().query('year == 2050')


# Plot biodiversity landscape without filtering
plot_landscape_bio = (
    p9.ggplot(
        df_bio_sum,
        p9.aes(
            x='SOLVE_WEIGHT_ALPHA', 
            y='SOLVE_WEIGHT_BETA', 
            fill='val', 
        )
    ) +
    p9.geom_point(size=3,stroke=0) +
    # p9.facet_wrap('name') +
    p9.facet_grid('BIODIVERSTIY_TARGET_GBF_2~GHG_LIMITS_FIELD') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)



# ------------------------ Profit V.S. Biodiversity -------------------------
df_profit = report_data.query('Type == "Profit_billion_AUD"'
    ).query('run_idx.isin(@valid_runs_demand_profit)'
    ).copy()

profit_vs_bio_df = pd.merge(
    df_profit, 
    df_bio_sum, 
    on=['run_idx','year'],
    suffixes=['_profit','_bio']
    ).query('year == 2050')


# Interactive plot - Alpah as color
fig = px.scatter(
    profit_vs_bio_df,
    x='val_profit',
    y='val_bio',
    color='SOLVE_WEIGHT_ALPHA_profit', 
    facet_row='BIODIV_GBF_TARGET_2_DICT_profit',
    facet_col='GHG_LIMITS_FIELD_profit',
    hover_data=['run_idx', 'SOLVE_WEIGHT_ALPHA_profit', 'SOLVE_WEIGHT_BETA_profit'],
)


for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split('=')[-1]

fig.update_layout(
    template='plotly_white',
    title='Profit V.S. Biodiversity',
    yaxis_title='Profit V.S. Biodiversity',
)

fig.show()


# Interactive plot - Beta as color
fig = px.scatter(
    profit_vs_bio_df,
    x='val_profit',
    y='val_bio',
    color='SOLVE_WEIGHT_BETA_profit', 
    facet_row='BIODIV_GBF_TARGET_2_DICT_profit',
    facet_col='GHG_LIMITS_FIELD_profit',
    hover_data=['run_idx', 'SOLVE_WEIGHT_ALPHA_profit', 'SOLVE_WEIGHT_BETA_profit'],
)


for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split('=')[-1]

fig.update_layout(
    template='plotly_white',
    title='Profit V.S. Biodiversity',
    yaxis_title='Profit V.S. Biodiversity',
)

fig.show()





# ------------------------ Profit V.S. Demand's deviation -------------------------
df_profit = report_data.query('Type == "Profit_billion_AUD"'
    ).query('run_idx.isin(@valid_runs_demand_profit)'
    ).copy()

profit_vs_demand_df = pd.merge(
    df_profit, 
    df_demand_avg, 
    on=['run_idx','year'],
    suffixes=['_profit','_demand']
    ).query('year == 2050')


# Interactive plot - Alpah as color
fig = px.scatter(
    profit_vs_demand_df,
    x='val_profit',
    y='val_demand',
    color='SOLVE_WEIGHT_ALPHA_profit', 
    facet_row='BIODIV_GBF_TARGET_2_DICT_profit',
    facet_col='GHG_LIMITS_FIELD_profit',
    hover_data=['run_idx', 'SOLVE_WEIGHT_ALPHA_profit', 'SOLVE_WEIGHT_BETA_profit'],
)


for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split('=')[-1]

fig.update_layout(
    template='plotly_white',
    title='Profit V.S. Demand\'s deviation'
)

fig.show()


# Interactive plot - Beta as color
fig = px.scatter(
    profit_vs_demand_df,
    x='val_profit',
    y='val_demand',
    color='SOLVE_WEIGHT_BETA_profit', 
    facet_row='BIODIV_GBF_TARGET_2_DICT_profit',
    facet_col='GHG_LIMITS_FIELD_profit',
    hover_data=['run_idx', 'SOLVE_WEIGHT_ALPHA_profit', 'SOLVE_WEIGHT_BETA_profit'],
)


for annotation in fig.layout.annotations:
    annotation.text = annotation.text.split('=')[-1]

fig.update_layout(
    template='plotly_white',
    title='Profit V.S. Demand\'s deviation'
)

fig.show()





############### Get the optimal Alpha/Beta 
optim_alpha = 0.1
optim_beta = 0.9



# -------------------- Profit -------------------
query_str = '''
    Type == "Profit_billion_AUD"
    and run_idx.isin(@valid_runs_demand_profit)
    '''.replace('\n', ' ').replace('  ', ' ')
    
query_str_optim = ''' 
    SOLVE_WEIGHT_ALPHA >= @optim_alpha - 0.005 
    and SOLVE_WEIGHT_ALPHA <= @optim_alpha + 0.005
    and SOLVE_WEIGHT_BETA >= @optim_beta - 0.005
    and SOLVE_WEIGHT_BETA <= @optim_beta + 0.005
    '''.replace('\n', ' ').replace('  ', ' ')


# With optim filtering
df_profit = report_data.query(query_str).query('run_idx.isin(@valid_runs_demand_profit)').copy()

p_weight_vs_profit = (
    p9.ggplot(
        df_profit, 
        p9.aes(
            x='year', 
            y='val', 
            color='SOLVE_WEIGHT_ALPHA',
            group='run_idx'
        )
    ) +
    p9.facet_grid('GHG_LIMITS_FIELD~BIODIVERSTIY_TARGET_GBF_2') +
    p9.geom_line(size=0.3) +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8)
    ) +
    p9.ylab('Profit (billion AUD)')
)

# With optim filtering
df_profit_optim = df_profit.query(query_str_optim)

p_weight_vs_profit = (
    p9.ggplot(
        df_profit_optim, 
        p9.aes(
            x='year', 
            y='val', 
            color='BIODIVERSTIY_TARGET_GBF_2',
            group='run_idx'
        )
    ) +
    p9.facet_wrap('GHG_LIMITS_FIELD') +
    p9.geom_line(size=0.3) +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8)
    ) +
    p9.ylab('Profit (billion AUD)')
)



# ------------------ Demand ------------------
query_str = '''
    Type == "Production_deviation_pct"
    and run_idx.isin(@valid_runs_demand_profit)
    '''.replace('\n', ' ').replace('  ', ' ')
    
query_str_optim = ''' 
    SOLVE_WEIGHT_ALPHA >= @optim_alpha - 0.005 
    and SOLVE_WEIGHT_ALPHA <= @optim_alpha + 0.005
    and SOLVE_WEIGHT_BETA >= @optim_beta - 0.005
    and SOLVE_WEIGHT_BETA <= @optim_beta + 0.005
    '''.replace('\n', ' ').replace('  ', ' ')


# With optim filtering
df_demand = report_data.query(query_str).query('abs(val) >= 1')

p_weight_vs_demand = (
    p9.ggplot(
        df_demand, 
        p9.aes(
            x='year', 
            y='val', 
            fill='name',
            group='run_idx'   
        )
    ) +
    p9.facet_grid('BIODIVERSTIY_TARGET_GBF_2~GHG_LIMITS_FIELD') +
    p9.geom_col(position='jitter') +
    p9.theme_bw() +
    p9.theme(strip_text=p9.element_text(size=8)) +
    p9.guides(color=p9.guide_legend(ncol=1))
)

# With optim filtering
df_demand_optim = df_demand.query(query_str_optim)

p_weight_vs_demand = (
    p9.ggplot(
        df_demand_optim, 
        p9.aes(
            x='year', 
            y='val', 
            fill='name',
            group='run_idx'   
        )
    ) +
    p9.facet_grid('BIODIVERSTIY_TARGET_GBF_2~GHG_LIMITS_FIELD') +
    p9.geom_col(position='jitter') +
    p9.theme_bw() +
    p9.theme(strip_text=p9.element_text(size=8)) +
    p9.guides(color=p9.guide_legend(ncol=1))
)




# ------------------ Biodiversity ------------------
query_str = '''
    Type == "Biodiversity_obj_score"
    and run_idx.isin(@valid_runs_demand_profit)
    '''.replace('\n', ' ').replace('  ', ' ')

query_str_optim = ''' 
    SOLVE_WEIGHT_ALPHA >= @optim_alpha - 0.005 
    and SOLVE_WEIGHT_ALPHA <= @optim_alpha + 0.005
    and SOLVE_WEIGHT_BETA >= @optim_beta - 0.005
    and SOLVE_WEIGHT_BETA <= @optim_beta + 0.005
    '''.replace('\n', ' ').replace('  ', ' ')

    
df_bio = report_data.query(query_str).copy()

df_bio_sum = df_bio.groupby(['year', 'GHG_LIMITS_FIELD', 'BIODIVERSTIY_TARGET_GBF_2', 'SOLVE_WEIGHT_ALPHA', 'SOLVE_WEIGHT_BETA']
    )[['val','run_idx']].agg(val=('val', 'mean'), run_idx=('run_idx', 'first')
    ).reset_index()


# Without optim filtering
p_weight_vs_bio = (
    p9.ggplot(
        df_bio_sum, 
        p9.aes(
            x='year', 
            y='val', 
            color='SOLVE_WEIGHT_ALPHA',
            group='run_idx',
        )
    ) +
    p9.facet_grid('BIODIVERSTIY_TARGET_GBF_2~GHG_LIMITS_FIELD') +
    p9.geom_line(size=0.3) +
    p9.theme_bw() +
    p9.theme(strip_text=p9.element_text(size=8)) +
    p9.ylab('Mean Biodiversity Area Score')
)


# With optim filtering
df_bio_sum_optim = df_bio_sum.query(query_str_optim)

p_weight_vs_bio = (
    p9.ggplot(
        df_bio_sum_optim, 
        p9.aes(
            x='year', 
            y='val', 
            color='BIODIVERSTIY_TARGET_GBF_2',
            group='run_idx',
        )
    ) +
    p9.facet_wrap('GHG_LIMITS_FIELD') +
    p9.geom_line(size=0.3) +
    p9.theme_bw() +
    p9.theme(strip_text=p9.element_text(size=8)) +
    p9.ylab('Biodiversity Area Score')
)

