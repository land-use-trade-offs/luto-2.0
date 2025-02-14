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
import plotnine as p9

from glob import glob
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Get the data
task_root_dirs = [i for i in glob('../Custom_runs/*') if "20250207_RES10_Timeseries" in i]
report_data, report_data_demand = process_task_root_dirs(task_root_dirs)

# Reorder the data
ghg_order = [
    '1.5C (67%) excl. avoided emis', 
    # '1.5C (50%) excl. avoided emis', 
    '1.8C (67%) excl. avoided emis'
]
bio_target_order = [
    '{2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3}', 
    '{2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5}'
]
report_data['GHG_LIMITS_FIELD'] = pd.Categorical(report_data['GHG_LIMITS_FIELD'], categories=ghg_order, ordered=True)
report_data['BIODIV_GBF_TARGET_2_DICT'] = pd.Categorical(report_data['BIODIV_GBF_TARGET_2_DICT'], categories=bio_target_order, ordered=True)
report_data_demand['GHG_LIMITS_FIELD'] = pd.Categorical(report_data_demand['GHG_LIMITS_FIELD'], categories=ghg_order, ordered=True)

# Filter the data
#    
#    GHG_CONSTRAINT_TYPE == "hard" and
#    BIODIVERSTIY_TARGET_GBF_2 == "on" and
#    SOLVE_ECONOMY_WEIGHT >= 0 and
#    SOLVE_ECONOMY_WEIGHT <= 0.5
filter_rules = '''
    year != 2010 and
    DIET_DOM == "BAU" and
    GHG_CONSTRAINT_TYPE == "soft" and
    BIODIVERSTIY_TARGET_GBF_2 == "on" and
    MODE == "timeseries" and
    SOLVE_ECONOMY_WEIGHT <= 0.3
'''.strip().replace('\n', '')

report_data_filter = report_data.query(filter_rules).copy()
report_data_demand_filterd = report_data_demand.query(filter_rules).copy()




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


p_weight_vs_profit.save('F:/jinzhu/TMP/SOLVE_WEIGHT_plots/03_1_p_weight_vs_profit.svg')


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

p_weight_vs_GHG_deviation.save('F:/jinzhu/TMP/SOLVE_WEIGHT_plots/03_2_p_weight_vs_GHG_deviation.svg')

p_weight_vs_GHG_deforestation = (
    p9.ggplot(
        report_data_filter, 
        p9.aes(
            x='year', 
            y='Deforestation', 
            lintype='BIODIV_GBF_TARGET_2_DICT',
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


p_weight_vs_demand = (
    p9.ggplot(report_data_demand_filterd) +
    p9.geom_line(
        p9.aes(
            x='year', 
            y='deviation_%', 
            color='name',
            linetype='BIODIV_GBF_TARGET_2_DICT',
            size='name',
        ),
    ) +
    p9.scale_color_manual(
        values={
            'Sheep lexp': 'blue', 
            'Sheep meat': 'green', 
            'Sheep wool': 'red'
        },
        na_value='#bcbcbc'
    ) +
    p9.scale_size_manual(
        values={
            'Sheep lexp': 1, 
            'Sheep meat': 1, 
            'Sheep wool': 1
        },
        na_value=0.5
    ) +
    p9.facet_grid('SOLVE_ECONOMY_WEIGHT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.theme_bw() +
    p9.guides(color=p9.guide_legend(ncol=1))
)

p_weight_vs_demand.save('F:/jinzhu/TMP/SOLVE_WEIGHT_plots/03_3_p_weight_vs_demand.svg')






# # Snapshoot plots
# p_weight_vs_profit = (
#     p9.ggplot(
#         report_data_filter, 
#         p9.aes(
#             x='SOLVE_ECONOMY_WEIGHT', 
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
#             x='SOLVE_ECONOMY_WEIGHT', 
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
#             x='SOLVE_ECONOMY_WEIGHT', 
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
#             color='SOLVE_ECONOMY_WEIGHT', 
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


