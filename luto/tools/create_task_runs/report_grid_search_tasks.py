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
import plotnine as p9


from luto.tools.create_task_runs.helpers import get_hatch_patches, process_task_root_dirs
from luto.tools.create_task_runs.parameters import HATCH_PATTERNS, PLOT_COL_WIDTH
from itertools import cycle



# Get the data
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20251013_RES5_SACHIN_RUNS/"
report_data = process_task_root_dirs(task_root_dir).query('region == "AUSTRALIA"').copy()
print(report_data['Type'].unique())


# Plot settings
p9.options.figure_size = (12, 6)
p9.options.dpi = 100


# Define grouping hierarchy
warp_col = 'DYNAMIC_PRICE'
shift_col = 'CARBON_EFFECTS_WINDOW'

# Define jitter and hatch mappings
# Jitter is to offset bars so they are visible when overlapping
# Hatch is to add pattern to bars so they are visible when overlapping
n_shifts = report_data[shift_col].nunique()
shift_distances = (np.arange(n_shifts) - ((n_shifts) / 2)) * PLOT_COL_WIDTH

jitter_map = dict(zip(report_data[shift_col].sort_values().unique(), shift_distances))
hatch_map = dict(zip(report_data[shift_col].sort_values().unique(), cycle(HATCH_PATTERNS)))

report_data['jitter_val'] = report_data[shift_col].map(jitter_map)
report_data['hatch_val'] = report_data[shift_col].map(hatch_map)

# Convert to categorical for plotting order
report_data[shift_col] = report_data[shift_col].astype('category')
report_data[warp_col] = report_data[warp_col].astype('category')


####################### Area non-ag #######################
report_data_area = report_data\
    .query('Type == "Area_non_ag_lu_ha"')\
    .eval('value = value / 1e6')\
    .copy()\
    .reset_index(drop=True)

p = (
    p9.ggplot() +
    p9.geom_col(
        data=report_data_area,
        mapping = p9.aes(
            x='year + jitter_val',
            y='value',
            fill='name'
        ),
        position='stack',
        width=( PLOT_COL_WIDTH * 0.85),
        alpha=0.8
    ) +
    p9.facet_wrap(
        f'~{warp_col}',  
        labeller='label_both',
        ncol=3
    ) +
    p9.theme_bw() +
    p9.theme(
        figure_size=(16, 10),
        subplots_adjust={'wspace': 0.25, 'hspace': 0.25},
        axis_text_x=p9.element_text(rotation=45, hjust=1),
        legend_position='right',
        legend_title=p9.element_text(size=10),
        legend_text=p9.element_text(size=8),
        strip_text=p9.element_text(size=8),
    ) +
    p9.labs(
        x='Year',
        y='Area (million ha)',
        fill=''
    )
)

# Add patches to the base column plot
get_hatch_patches(p.draw(), report_data_area, warp_col, shift_col)


####################### Area ag-mgt #######################
report_data_area_agmgt = report_data\
    .query('Type == "Area_ag_man_ha"')\
    .groupby(['run_idx', warp_col, shift_col, 'ag_mgt', 'hatch_val', 'jitter_val', 'year'], observed=True)\
    .sum(numeric_only=True)\
    .reset_index()\
    .eval('value = value / 1e6')
    

p = (
    p9.ggplot() +
    p9.geom_col(
        data=report_data_area_agmgt,
        mapping = p9.aes(
            x='year + jitter_val',
            y='value',
            fill='ag_mgt'
        ),
        position='stack',
        width=( PLOT_COL_WIDTH * 0.85),
        alpha=0.8
    ) +
    p9.facet_wrap(
        f'~{warp_col}',  
        labeller='label_both',
        ncol=3
    ) +
    p9.theme_bw() +
    p9.theme(
        figure_size=(16, 10),
        subplots_adjust={'wspace': 0.25, 'hspace': 0.25},
        axis_text_x=p9.element_text(rotation=45, hjust=1),
        legend_position='right',
        legend_title=p9.element_text(size=10),
        legend_text=p9.element_text(size=8),
        strip_text=p9.element_text(size=8),
    ) +
    p9.labs(
        x='Year',
        y='Area (million ha)',
        fill=''
    )
)

# Add patches to the base column plot
get_hatch_patches(p.draw(), report_data_area_agmgt, warp_col, shift_col)





####################### GHG #######################
report_data_ghg = report_data\
    .query('Type == "GHG_tCO2e"')\
    .eval('value = value / 1e6')\
    .copy()\
    .reset_index(drop=True)

report_data_ghg_col = report_data_ghg.query('name not in ["Net emissions", "GHG emission limit"]').copy()
report_data_ghg_net = report_data_ghg.query('name == "Net emissions"').copy()
report_data_ghg_limit = report_data_ghg.query('name == "GHG emission limit"').copy()


p = (
    p9.ggplot() +
    p9.geom_col(
        data=report_data_ghg_col,
        mapping = p9.aes(
            x='year + jitter_val',
            y='value',
            fill='name'
        ),
        position='stack',
        width=( PLOT_COL_WIDTH * 0.85),
        alpha=0.8
    ) +
    p9.geom_line(
        data=report_data_ghg_net,
        mapping = p9.aes(
            x='year',
            y='value',
            color=shift_col,
        ),
    ) +
    p9.geom_line(
        data=report_data_ghg_limit,
        mapping = p9.aes(
            x='year',
            y='value',
            group=shift_col,
        ),
    ) +
    p9.facet_wrap(
        f'~{warp_col}',  
        labeller='label_both',
        ncol=3
    ) +
    p9.theme_bw() +
    p9.theme(
        figure_size=(16, 10),
        subplots_adjust={'wspace': 0.25, 'hspace': 0.25},
        axis_text_x=p9.element_text(rotation=45, hjust=1),
        legend_position='right',
        legend_title=p9.element_text(size=10),
        legend_text=p9.element_text(size=8),
        strip_text=p9.element_text(size=8),
    ) +
    p9.labs(
        x='Year',
        y='GHG (million tCO2e)',
        fill=''
    )
)

# Add patches to the base column plot
get_hatch_patches(p.draw(), report_data_ghg_col, warp_col, shift_col)





####################### Economics #######################
report_data_economics = report_data\
    .query('Type == "Economic_AUD"')\
    .eval('value = value / 1e9')\
    .copy()\
    .reset_index(drop=True)

report_data_economics_col = report_data_economics.query('name != "Profit"').copy()
report_data_economics_net = report_data_economics.query('name == "Profit"').copy()

p = (
    p9.ggplot() +
    p9.geom_col(
        data=report_data_economics_col,
        mapping = p9.aes(
            x='year + jitter_val',
            y='value',
            fill='name'
        ),
        position='stack',
        width=( PLOT_COL_WIDTH * 0.85),
        alpha=0.8
    ) +
    p9.geom_line(
        data=report_data_economics_net,
        mapping = p9.aes(
            x='year',
            y='value',
            color=shift_col,
        ),
    ) +
    p9.facet_wrap(
        f'~{warp_col}',  
        labeller='label_both',
        ncol=3
    ) +
    p9.theme_bw() +
    p9.theme(
        figure_size=(16, 10),
        subplots_adjust={'wspace': 0.25, 'hspace': 0.25},
        axis_text_x=p9.element_text(rotation=45, hjust=1),
        legend_position='right',
        legend_title=p9.element_text(size=10),
        legend_text=p9.element_text(size=8),
        strip_text=p9.element_text(size=8),
    ) +
    p9.labs(
        x='Year',
        y='AUD (billion tCO2e)',
        fill=''
    )
)



# Export to matplotlib to add hatches
get_hatch_patches(p.draw(), report_data_economics_col, warp_col, shift_col)



####################### Demand - Percent deviation from target #######################
demand_deviation = report_data\
    .query('Type == "Production_deviation_percent"')\
    .copy()\
    .reset_index(drop=True)
    
p = (
    p9.ggplot() +
    p9.geom_col(
        data=demand_deviation,
        mapping = p9.aes(
            x='year + jitter_val',
            y='value',
            fill='name'
        ),
        position='dodge',
    ) +
    p9.facet_wrap(
        f'~{warp_col}',  
        labeller='label_both',
        ncol=3
    ) +
    p9.theme_bw() +
    p9.theme(
        figure_size=(16, 10),
        subplots_adjust={'wspace': 0.25, 'hspace': 0.25},
        axis_text_x=p9.element_text(rotation=45, hjust=1),
        legend_position='right',
        legend_title=p9.element_text(size=10),
        legend_text=p9.element_text(size=8),
        strip_text=p9.element_text(size=8),
    ) +
    p9.labs(
        x='Year',
        y='Percent deviation from target (%)',
        color=''
    )
)



    