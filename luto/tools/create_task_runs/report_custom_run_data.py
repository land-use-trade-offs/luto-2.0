import pandas as pd
import plotnine as p9

from glob import glob
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Get the data
task_root_dirs = [i for i in glob('../Custom_runs/*') if "20250206_1_RES10_Timeseries" in i]
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
    BIODIVERSTIY_TARGET_GBF_2 == "off" and
    MODE == "timeseries" and
    SOLVE_ECONOMY_WEIGHT == 0.3
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
            group='name'
        )
    ) +
    p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
    p9.geom_col(position='dodge') +
    p9.theme_bw() +
    # p9.coord_cartesian(ylim=(0, 50)) +
    p9.guides(fill=p9.guide_legend(ncol=1))
)






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


