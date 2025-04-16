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
from glob import glob
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Plot settings
p9.options.figure_size = (12, 6)
p9.options.dpi = 100


# Get the data
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250414_GRID_SEARCH/"
report_data = process_task_root_dirs(task_root_dir)


# ------------------ Demand ------------------
query_str = '''
    Type == "Production_Mt" 
    and year != 2010
    '''.replace('\n', ' ').replace('  ', ' ')

df_demand = report_data.query(query_str).copy()

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
    and SOLVE_WEIGHT_BETA == 0.95
    '''.replace('\n', ' ').replace('  ', ' ')

df_profit = report_data.query(query_str).copy()

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


p_weight_vs_profit.save('F:/jinzhu/TMP/SOLVE_WEIGHT_plots/03_1_p_weight_vs_profit.svg')




# ------------------ Biodiversity ------------------
query_str = '''
    Type == "Biodiversity_area_score"
    and SOLVE_WEIGHT_BETA == 0.95
    '''.replace('\n', ' ').replace('  ', ' ')
    
df_bio = report_data.query(query_str).copy()

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





