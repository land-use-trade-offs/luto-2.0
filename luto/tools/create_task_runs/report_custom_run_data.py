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
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Plot settings
p9.options.figure_size = (12, 6)
p9.options.dpi = 100


# Get the data
task_root_dir = "N:/LUF-Modelling/LUTO2_JZ/Custom_runs/20250414_GRID_SEARCH"
report_data = process_task_root_dirs(task_root_dir)

# Commodities allowed to not meet target
commodity_out_targets = {
    'Beef live export',
    'Beef meat',
    'Sheep live export',
    'Sheep meat',
    'Sheep wool',
}



# -------------- Weight landscape (Demand) ----------------------
query_str = '''
    Type == "Production_Mt" 
    and year != 2010
    '''.replace('\n', ' ').replace('  ', ' ')


# Get demand data
df_demand = report_data.query(query_str).copy()

df_demand['val'] = df_demand['val'].abs()
df_demand_avg = df_demand.groupby(
    ['year', 'GHG_LIMITS_FIELD', 'BIODIV_GBF_TARGET_2_DICT', 'SOLVE_WEIGHT_ALPHA', 'SOLVE_WEIGHT_BETA']
)['val'].mean().reset_index()



plot_landscape_demand = (
    p9.ggplot(
        df_demand_avg.query('year == 2050 and val < 10'),
        p9.aes(
            x='SOLVE_WEIGHT_ALPHA', 
            y='SOLVE_WEIGHT_BETA', 
            fill='val', 
        )
    ) +
    p9.geom_point(size=3,stroke=0) +
    # p9.facet_wrap('name') +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT~GHG_LIMITS_FIELD') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)





# ------------------ Demand ------------------

# Get valid runs
valid_runs = set(report_data['run_idx']) - set(
    df_demand.query(
        'name not in @commodity_out_targets and abs(val) >= 10'
    )['run_idx']
)
df_demand = df_demand.query('run_idx.isin(@valid_runs)')




p_weight_vs_demand = (
    p9.ggplot(
        df_demand, 
        p9.aes(
            x='year', 
            y='val', 
            fill='name',   
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT~GHG_LIMITS_FIELD') +
    p9.geom_col(position='jitter') +
    p9.theme_bw() +
    p9.theme(strip_text=p9.element_text(size=8)) +
    p9.guides(color=p9.guide_legend(ncol=1))
)

p_weight_vs_demand.save('F:/jinzhu/TMP/SOLVE_WEIGHT_plots/03_3_p_weight_vs_demand.svg')




# -------------------- Profit -------------------
query_str = '''
    Type == "Profit_billion_AUD" 
    '''.replace('\n', ' ').replace('  ', ' ')

df_profit = report_data.query(query_str).query('run_idx.isin(@valid_runs)').copy()

df_profit['group'] = (
    df_profit['GHG_LIMITS_FIELD'].astype(str) 
    + '_' 
    + df_profit['BIODIV_GBF_TARGET_2_DICT'].astype(str)
    + '_'
    + df_profit['SOLVE_WEIGHT_ALPHA'].astype(str)
    + '_'
    + df_profit['SOLVE_WEIGHT_BETA'].astype(str)
)

p_weight_vs_profit = (
    p9.ggplot(
        df_profit, 
        p9.aes(
            x='year', 
            y='val', 
            color='SOLVE_WEIGHT_ALPHA',
            linetype='BIODIV_GBF_TARGET_2_DICT',
            group='group'
        )
    ) +
    p9.facet_grid('GHG_LIMITS_FIELD~BIODIV_GBF_TARGET_2_DICT') +
    p9.geom_line(size=0.3) +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8)
    ) +
    # p9.scale_x_log10() +
    p9.ylab('Profit (billion AUD)')
)


plot_landscape_profit = (
    p9.ggplot(
        df_profit.query('SOLVE_WEIGHT_ALPHA<0.21 and SOLVE_WEIGHT_BETA > 0.79 and year == 2050'),
        p9.aes(
            x='SOLVE_WEIGHT_ALPHA', 
            y='SOLVE_WEIGHT_BETA', 
            fill='val', 
        )
    ) +
    p9.geom_point(size=3,stroke=0) +
    # p9.facet_wrap('name') +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT~GHG_LIMITS_FIELD') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)



# ------------------ Biodiversity ------------------
query_str = '''
    Type == "Biodiversity_area_score"
    '''.replace('\n', ' ').replace('  ', ' ')
    
df_bio = report_data.query(query_str).query('run_idx.isin(@valid_runs)').copy()

df_bio_sum = df_bio.groupby(
    ['year', 'GHG_LIMITS_FIELD', 'BIODIV_GBF_TARGET_2_DICT', 'SOLVE_WEIGHT_ALPHA']
)['val'].sum().reset_index()


p_weight_vs_bio = (
    p9.ggplot(
        df_bio_sum, 
        p9.aes(
            x='year', 
            y='val', 
            color='SOLVE_WEIGHT_ALPHA',
            group='SOLVE_WEIGHT_ALPHA',
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT~GHG_LIMITS_FIELD') +
    p9.geom_line(size=0.3) +
    p9.theme_bw() +
    p9.theme(strip_text=p9.element_text(size=8)) +
    p9.ylab('Mean Biodiversity Area Score')
)



df_bio_sum = df_bio.groupby(
    ['year', 'GHG_LIMITS_FIELD', 'BIODIV_GBF_TARGET_2_DICT', 'SOLVE_WEIGHT_ALPHA', 'SOLVE_WEIGHT_BETA']
)['val'].sum().reset_index()

df_bio_sum['val'] = df_bio_sum['val']/10000



plot_landscape_bio = (
    p9.ggplot(
        df_bio_sum.query('SOLVE_WEIGHT_ALPHA<0.21 and SOLVE_WEIGHT_BETA > 0.79 and year == 2050'),
        p9.aes(
            x='SOLVE_WEIGHT_ALPHA', 
            y='SOLVE_WEIGHT_BETA', 
            fill='val', 
        )
    ) +
    p9.geom_point(size=3,stroke=0) +
    # p9.facet_wrap('name') +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT~GHG_LIMITS_FIELD') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)







# ------------------------ Profit V.S. Biodiversity -------------------------

fix_year = 2050
fix_beta = 0.9
fix_alpha = 0.1


df_demand = report_data.query('Type == "Production_Mt" ').copy()
df_demand['val'] = df_demand['val'].abs()
df_demand_avg = df_demand.groupby(
    ['year', 'GHG_LIMITS_FIELD', 'BIODIV_GBF_TARGET_2_DICT', 'SOLVE_WEIGHT_ALPHA', 'SOLVE_WEIGHT_BETA']
)['val'].mean().reset_index()



fix_profit = report_data.query('Type == "Profit_billion_AUD" ').copy()

fix_bio = report_data.query('Type == "Biodiversity_area_score"').copy().groupby(
    ['year', 'GHG_LIMITS_FIELD', 'BIODIV_GBF_TARGET_2_DICT', 'SOLVE_WEIGHT_ALPHA','SOLVE_WEIGHT_BETA']
)['val'].sum().reset_index()


fix_df = pd.merge(
    fix_profit, 
    fix_bio, 
    on=['year',
        'GHG_LIMITS_FIELD', 
        'BIODIV_GBF_TARGET_2_DICT',
        'SOLVE_WEIGHT_ALPHA',
        'SOLVE_WEIGHT_BETA',
    ], 
).query('year == @fix_year and SOLVE_WEIGHT_BETA==@fix_beta and SOLVE_WEIGHT_ALPHA <=0.21')

profit_vs_bio = (
    p9.ggplot()
    + p9.geom_point(
        data=fix_df, 
        mapping=p9.aes(
            x='val_x', 
            y='val_y', 
            color='SOLVE_WEIGHT_ALPHA',
            group='SOLVE_WEIGHT_ALPHA')
    ) +
    p9.theme_bw() +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT~GHG_LIMITS_FIELD', scales='free') +
    p9.theme(strip_text=p9.element_text(size=8)) +
    p9.ylab('Profit V.S. Biodiversity')
)
 


df_demand = report_data.query('Type == "Production_Mt" and SOLVE_WEIGHT_ALPHA==@fix_alpha').copy()
df_demand['val'] = df_demand['val'].abs()
df_demand_avg = df_demand.groupby(
    ['year', 'GHG_LIMITS_FIELD', 'BIODIV_GBF_TARGET_2_DICT', 'SOLVE_WEIGHT_ALPHA', 'SOLVE_WEIGHT_BETA']
)['val'].mean().reset_index()

fix_df = pd.merge(
    fix_profit, 
    df_demand_avg, 
    on=['year',
        'GHG_LIMITS_FIELD', 
        'BIODIV_GBF_TARGET_2_DICT',
        'SOLVE_WEIGHT_ALPHA',
        'SOLVE_WEIGHT_BETA',
    ], 
).query('year == @fix_year and SOLVE_WEIGHT_BETA>0.89')

p_weight_vs_demand = (
    p9.ggplot()
    + p9.geom_point(
        data=fix_df, 
        mapping=p9.aes(
            x='val_x', 
            y='val_y', 
            color='SOLVE_WEIGHT_BETA',
            group='SOLVE_WEIGHT_BETA')
    ) +
    p9.theme_bw() +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT~GHG_LIMITS_FIELD', scales='free') +
    p9.theme(strip_text=p9.element_text(size=8)) +
    p9.ylab('Profit V.S. Demand')
)






