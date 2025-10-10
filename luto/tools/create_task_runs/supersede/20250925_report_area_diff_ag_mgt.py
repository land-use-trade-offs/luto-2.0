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
import matplotlib.patches as patches

from luto.tools.create_task_runs.create_grid_search_tasks import TASK_ROOT_DIR
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Plot settings
p9.options.figure_size = (12, 6)
p9.options.dpi = 100


# Get the data
report_data = process_task_root_dirs(TASK_ROOT_DIR).query('region == "AUSTRALIA"').copy()
print(report_data['Type'].unique())


####################### GHG #######################
report_data_ghg = report_data.query('Type == "GHG_tCO2e"').copy().reset_index()

hatch_map = {
    10: '....',
    20: '////',
    30: r'\\\\',
    40: '||||',
    50: '----'
}

jitter_map = {
    10: -2.5,
    20: -1.5,
    30: -0.5,
    40: 0.5,
    50: 1.5
}

# Add jitter to the data
report_data_ghg['jitter_val'] = report_data_ghg['GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT'].map(jitter_map)

# Group the data for rectangle plotting by CARBON_EFFECTS_WINDOW
rectangle_map = {}
for group_idx, (carbon_window, df) in enumerate(report_data_ghg.groupby(['CARBON_EFFECTS_WINDOW'], observed=True)):
    rectangles = pd.DataFrame()
    for (gbf2_cut, year), _df in df.query('name not in ["Net emissions", "GHG emission limit"]', engine='python').groupby(['GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT', 'year'], observed=True):
        rectangle = pd.DataFrame({
            'gbf2_cut': [gbf2_cut],
            'year': [year],
            'rect_x_start': [year + _df['jitter_val'].iloc[0] - 0.35],
            'rect_x_end': [year + _df['jitter_val'].iloc[0] + 0.35],
            'rect_y_start': [_df['value'][_df['value'] < 0].sum() / 1e6],
            'rect_y_end': [_df['value'][_df['value'] >= 0].sum() / 1e6]
        })
        rectangles = pd.concat([rectangle, rectangles], ignore_index=True)
    rectangle_map[group_idx] = rectangles

# Create the base plot with jittered bars
p = (
    p9.ggplot() +
    p9.geom_col(
        data=report_data_ghg.query('name not in ["Net emissions", "GHG emission limit"]', engine='python'),
        mapping = p9.aes(
            x='year + jitter_val',
            y='value / 1e6',
            fill='name'
        ),
        position='stack',
        width=0.7,
        alpha=0.8
    ) +
    p9.facet_wrap(
        '~CARBON_EFFECTS_WINDOW',
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

# Export to matplotlib to add hatches
fig = p.draw()

# Add hatch patterns for GBF2 percentage cuts
for ax, df in zip(fig.axes, rectangle_map.values()):
    for _, row in df.iterrows():
        rect = patches.Rectangle(
            (row['rect_x_start'], row['rect_y_start']),
            row['rect_x_end'] - row['rect_x_start'],
            row['rect_y_end'] - row['rect_y_start'],
            hatch=hatch_map[row['gbf2_cut']],
            fill=False,
            edgecolor='gray',
            linewidth=0.01,
            alpha=0.5,
            zorder=10
        )
        ax.add_patch(rect)

# Add legend for the hatches
legend_patches = [
    patches.Patch(facecolor='none', edgecolor='grey', hatch='....', label='10%'),
    patches.Patch(facecolor='none', edgecolor='grey', hatch='////', label='20%'),
    patches.Patch(facecolor='none', edgecolor='grey', hatch=r'\\\\', label='30%'),
    patches.Patch(facecolor='none', edgecolor='grey', hatch='||||', label='40%'),
    patches.Patch(facecolor='none', edgecolor='grey', hatch='----', label='50%')
]

# Place hatch legend at relative position
fig.legend(handles=legend_patches, title='GBF2 Priority Cut', loc='center', bbox_to_anchor=(0.9, 0.75), fontsize=8, frameon=True, facecolor='white', edgecolor='black')





####################### Economics #######################
report_data_economics = report_data.query('Type == "Economic_AUD" and BIODIVERSITY_TARGET_GBF_2 == "high"').copy().reset_index()
report_data_economics['name'].unique()

# Add jitter to the data
report_data_economics['jitter_val'] = report_data_economics['GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT'].map(jitter_map)


# Create the base plot with jittered bars
p = (
    p9.ggplot() +
    p9.geom_col(
        data=report_data_economics.query('name != "Profit"'),
        mapping = p9.aes(
            x='year + jitter_val',
            y='value / 1e9',
            fill='name'
        ),
        position='stack',
        width=0.7,
        alpha=0.8
    ) +
    p9.geom_line(
        data=report_data_economics.query('name == "Profit"'),
        mapping = p9.aes(
            x='year',
            y='value / 1e9',
            fill='name',
            group='run_idx'
        ),
        alpha=0.8
    ) +
    p9.facet_wrap(
        '~CARBON_EFFECTS_WINDOW',
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

# Export to matplotlib to add hatches
fig = p.draw()

# Add hatch patterns for GBF2 percentage cuts
for ax, df in zip(fig.axes, rectangle_map.values()):
    for _, row in df.iterrows():
        rect = patches.Rectangle(
            (row['rect_x_start'], row['rect_y_start']),
            row['rect_x_end'] - row['rect_x_start'],
            row['rect_y_end'] - row['rect_y_start'],
            hatch=hatch_map[row['gbf2_cut']],
            fill=False,
            edgecolor='gray',
            linewidth=0.01,
            alpha=0.5,
            zorder=10
        )
        ax.add_patch(rect)

# Add legend for the hatches
legend_patches = [
    patches.Patch(facecolor='none', edgecolor='grey', hatch='....', label='10%'),
    patches.Patch(facecolor='none', edgecolor='grey', hatch='////', label='20%'),
    patches.Patch(facecolor='none', edgecolor='grey', hatch=r'\\\\', label='30%'),
    patches.Patch(facecolor='none', edgecolor='grey', hatch='||||', label='40%'),
    patches.Patch(facecolor='none', edgecolor='grey', hatch='----', label='50%')
]

# Place hatch legend at relative position
fig.legend(handles=legend_patches, title='GBF2 Priority Cut', loc='center', bbox_to_anchor=(0.9, 0.75), fontsize=8, frameon=True, facecolor='white', edgecolor='black')


