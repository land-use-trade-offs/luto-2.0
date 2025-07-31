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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from luto.tools.create_task_runs.create_grid_search_tasks import TASK_ROOT_DIR
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Plot settings
p9.options.figure_size = (12, 6)
p9.options.dpi = 100


# Get the data
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250730_RES3_RCP_GHG_CUT_SENSITIVITI/"
report_data = process_task_root_dirs(task_root_dir)


report_data = report_data.rename(columns={
    'GHG_EMISSIONS_LIMITS': 'GHG',
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': 'Biodiversity_CUT',
    'BIODIVERSITY_TARGET_GBF_2': 'Biodiversity',
})


        


####################### Area #######################

print(report_data['Type'].unique())

area_landuse = report_data.query('Type == "Area_non_ag_lu_ha" and region == "Goulburn Broken"').copy()
area_landuse['val'] = area_landuse['val'] / 1e6  # Convert to million hectares

# Get jitter and hatch values
jitter_map = {
    20: -1.5,   
    30: -0.5,   
    40: 0.5,
    50: 1.5     
}

hatch_map = {
    20: '////', 
    30: r'\\\\',
    40: '||||', 
    50: '----'  
}

group_map = dict(enumerate(
    area_landuse.groupby(['GHG', 'Biodiversity'], observed=True).groups.keys()
))


# Transform the x-axis values to jittered values
area_landuse['jitter_val'] = area_landuse['Biodiversity_CUT'].map(jitter_map)


# Group the data for rectangle plotting
rectangle_map = {}
for group_idx, (_, df) in enumerate(area_landuse.groupby(['GHG', 'Biodiversity'], observed=True)):
    rectangles = pd.DataFrame()
    for (name, year), _df in df.groupby(['Biodiversity_CUT', 'year'], observed=True):
        rectangle = pd.DataFrame({
            'name': name,
            'year': year,
            'rect_x_start': year + _df['jitter_val'] - 0.4,
            'rect_x_end': year + _df['jitter_val'] + 0.4,
            'rect_y_start': 0,
            'rect_y_end': _df['val'].sum()
        })
        rectangles = pd.concat([rectangle, rectangles], ignore_index=True)
    rectangle_map[group_idx] = rectangles


p = (
    p9.ggplot(area_landuse) 
    + p9.geom_col(p9.aes(x='year + jitter_val', y='val', fill='name'), position='stack', alpha=0.8)
    + p9.theme_bw()
    + p9.labs(
        x='Year',
        y='Area (million ha)',
        fill='Agriculture Management Type'  
    )
    + p9.facet_grid('GHG ~ Biodiversity', scales='free_y', labeller='label_both')  
    + p9.guides(fill=p9.guide_legend(ncol=1))
    + p9.theme(
        legend_key=p9.element_rect(color=''),
        legend_position='right',  # Place legend on the right
        legend_box_margin=10,     # Add margin around the legend box
        figure_size=(16, 8)       # Increase figure width to accommodate legend
    )
)

# Export to matplotlib
fig = p.draw()

# Adjust the figure size
fig.set_size_inches(14, 8)  # Set the desired width and height in inches

# Plot rectangles from the prepared data
for ax, df in zip(fig.axes, rectangle_map.values()):
    for _, row in df.iterrows():
        rect = patches.Rectangle(
            (row['rect_x_start'], row['rect_y_start']),
            row['rect_x_end'] - row['rect_x_start'],
            row['rect_y_end'] - row['rect_y_start'],
            hatch=hatch_map[row['name']],
            fill=False, 
            edgecolor='gray', 
            linewidth=0.01,
            alpha=0.5,  # Adjust transparency
            zorder=0  # Set zorder to a lower value to place the patch below other elements
        )
        ax.add_patch(rect)

# Add legend for the hatches
legend_patches = [
    patches.Patch(facecolor='none', edgecolor='grey', hatch=hatch, label=label)
    for label, hatch in hatch_map.items()
]

ax.legend(handles=legend_patches, title='', loc=(1.14, 0.3), fontsize=10, frameon=False)
ax.text(1.15, 0.58, 'Priority percentage cut', ha='left', va='center', transform=ax.transAxes)

fig.show()







