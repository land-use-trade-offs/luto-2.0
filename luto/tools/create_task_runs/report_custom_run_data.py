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

import numpy as np
import pandas as pd
import plotnine as p9

from glob import glob
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Get the data
task_root_dirs = [i for i in glob('../Custom_runs/*') if "20250414_RES5_GRID_SEARCH_ALPHA_WEIGHTS" in i][:10]
report_data, report_data_demand = process_task_root_dirs(task_root_dirs)


# Weights
weight_alpha = 0.8
weight_beta = 0.98

# Filter the data
filter_rules = '''
    year != 2010 

'''.strip().replace('\n', '')

report_data_filter = report_data.query(filter_rules).copy()
report_data_filter['group'] = (
    report_data_filter['GHG_LIMITS_FIELD'].astype(str) 
    + '_' 
    + report_data_filter['BIODIV_GBF_TARGET_2_DICT'].astype(str)
    + '_'
    + report_data_filter['SOLVE_WEIGHT_ALPHA'].astype(str)
)


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

# ------------------
report_data_demand_filterd = (
    report_data_demand
    .query('SOLVE_WEIGHT_ALPHA==@weight_alpha and SOLVE_WEIGHT_BETA==@weight_beta')
    .query('abs(`deviation_%`) <=5 and abs(`deviation_%`)>=0.001')
)

p_weight_vs_demand = (
    p9.ggplot(
        report_data_demand_filterd, 
        p9.aes(
            x='year', 
            y='deviation_%', 
            fill='name', 
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT~GHG_LIMITS_FIELD') +
    p9.geom_col(position='stack') +
    p9.theme_bw() +
    p9.theme(strip_text=p9.element_text(size=8)) +
    p9.guides(color=p9.guide_legend(ncol=1))
)

p_weight_vs_demand.save('F:/jinzhu/TMP/SOLVE_WEIGHT_plots/03_3_p_weight_vs_demand.svg')




# p_weight_vs_GHG_deviation = (
#     p9.ggplot(
#         report_data_filter, 
#         p9.aes(
#             x='year', 
#             y='GHG deviation', 
#             color='GBF2_PENALTY', 
#             # linetype='DIET_GLOB',
#             group='GBF2_PENALTY',
#         )
#     ) +
#     # p9.facet_wrap('WATER_PENALTY', labeller='label_both') +
#     p9.geom_line() +
#     p9.theme_bw() +
#     # p9.scale_x_log10() +
#     p9.ylab('GHG deviation (Mt)')
#     )

# p_weight_vs_GHG_deviation.save('F:/jinzhu/TMP/SOLVE_WEIGHT_plots/03_2_p_weight_vs_GHG_deviation.svg')

# p_weight_vs_GHG_deforestation = (
#     p9.ggplot(
#         report_data_filter, 
#         p9.aes(
#             x='year', 
#             y='Total_Deforestation', 
#             # lintype='BIODIV_GBF_TARGET_2_DICT',
#             color='GBF2_PENALTY', 
#             # linetype='DIET_GLOB',
#             group='GBF2_PENALTY',
#         )
#     ) +
#     # p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
#     p9.geom_line() +
#     p9.theme_bw() +
#     # p9.scale_x_log10() +
#     p9.ylab('Deforestation (Mt)')
#     )


# # Snapshoot plots
# p_weight_vs_profit = (
#     p9.ggplot(
#         report_data_filter, 
#         p9.aes(
#             x='SOLVE_WEIGHT_ALPHA', 
#             y='Profit',
#         )
#     ) +
#     p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
#     p9.geom_line(size=0.3) +
#     p9.theme_bw() +
#     # p9.scale_x_log10() +
#     p9.ylab('Profit (billion AUD)')
#     )


# p_weight_vs_GHG_deviation = (
#     p9.ggplot(
#         report_data_filter, 
#         p9.aes(
#             x='SOLVE_WEIGHT_ALPHA', 
#             y='GHG deviation'
#         )
#     ) +
#     p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
#     p9.geom_line() +
#     p9.theme_bw() +
#     # p9.scale_x_log10() +
#     p9.ylab('GHG deviation (Mt)')
#     )

# p_weight_vs_GHG_deforestation = (
#     p9.ggplot(
#         report_data_filter, 
#         p9.aes(
#             x='SOLVE_WEIGHT_ALPHA', 
#             y='Deforestation', 
#         )
#     ) +
#     p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
#     p9.geom_line() +
#     p9.theme_bw() +
#     # p9.scale_x_log10() +
#     p9.ylab('Deforestation (Mt)')
#     )


# p_GHG_vs_profit = (
#     p9.ggplot(report_data_filter, 
#         p9.aes(
#             x='GHG deviation', 
#             y='Profit', 
#             color='SOLVE_WEIGHT_ALPHA', 
#             shape='DIET_GLOB',
#             group='interaction',
#         )
        
#     ) +
#     p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
#     p9.geom_point(size=0.1) +
#     p9.theme_bw()
#     )


# p_weigth_vs_demand = (
#     p9.ggplot(
#         report_data_demand_filterd, 
#         p9.aes(
#             x='SOLVE_ECONOMY_WEIGHT', 
#             y='deviation_%', 
#             fill='name', 
#         )
#     ) +
#     p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
#     p9.geom_col(position='dodge') +
#     p9.theme_bw() +
#     p9.scale_x_log10() +
#     p9.guides(fill=p9.guide_legend(ncol=1))
# )


