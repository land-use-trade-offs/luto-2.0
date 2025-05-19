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
task_root_dir = TASK_ROOT_DIR.rstrip('/')       # Or replace with the desired task run root dir
report_data = process_task_root_dirs(task_root_dir)



####################### Area #######################
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250512_DCCEEW_REPORT_01_CONNECTIVITY_ONF_OFF/"
report_data = process_task_root_dirs(task_root_dir)
print(report_data['Type'].unique())


# Get jitter and hatch values
area_landuse = report_data.query('Type == "Area_ag_man_million_km2"').replace(np.nan, 'None')
jitter_map = {'DWI': -1, 'NCI': 1, 'None': 0}
hatch_map = {'DWI': '////', 'NCI': r'\\\\', 'None': ''}
area_landuse['jitter_val'] = area_landuse['CONNECTIVITY_SOURCE'].map(jitter_map)


# Group by CONNECTIVITY_SOURCE and prepare data for rectangle plotting
rectangles = pd.DataFrame()
for (name, year), group in area_landuse.groupby(['CONNECTIVITY_SOURCE', 'year']):
    rectangle = pd.DataFrame({
        'name': name,
        'year': year,
        'rect_x_start': year + group['jitter_val'] - 0.4,
        'rect_x_end': year + group['jitter_val'] + 0.4,
        'rect_y_start': 0,
        'rect_y_end': group['val'].sum()
    })
    rectangles = pd.concat([rectangle, rectangles], ignore_index=True)



p = (
    p9.ggplot(area_landuse) 
    + p9.geom_col(p9.aes(x='year + jitter_val', y='val', fill='name'), position='stack', alpha=0.8)  # stack columns by CONNECTIVITY_SOURCE
    + p9.theme_bw()
    + p9.labs(
        x='Year',
        y='Area (million km2)',
        fill='Agriculture Management Type'  
    )
    + p9.guides(fill=p9.guide_legend(ncol=1))  # Reduce legend icon size further
    + p9.theme(
        legend_key=p9.element_rect(color=''),  # Set legend key background to white
    )
)

# Export to matplotlib
fig = p.draw()

# Adjust the figure size
fig.set_size_inches(14, 8)  # Set the desired width and height in inches

# Now you can access the underlying matplotlib axes and modify them
ax = fig.axes[0]

# Plot rectangles from the prepared data
for _, row in rectangles.iterrows():
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
ax.legend(handles=legend_patches, title='', loc=(1.045, 0.15), fontsize=10, frameon=False)

# Show the final plot
fig.show()






######################## GBF-3 MVS #######################
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250512_DCCEEW_REPORT_02_GBF3_MVS_ON_OFF/"
report_data = process_task_root_dirs(task_root_dir)
print(report_data['Type'].unique())


# Get jitter and hatch values
area_landuse = report_data.query('Type == "Area_non_ag_lu_million_km2"').copy()
jitter_map = {'on': -0.5, 'off': 0.5}
hatch_map = {'on': '////', 'off': r'\\\\'}
area_landuse['jitter_val'] = area_landuse['BIODIVERSTIY_TARGET_GBF_3'].map(jitter_map)

# Group by BIODIVERSTIY_TARGET_GBF_3 and prepare data for rectangle plotting
rectangles = pd.DataFrame()
for (name, year), group in area_landuse.groupby(['BIODIVERSTIY_TARGET_GBF_3', 'year']):
    rectangle = pd.DataFrame({
        'name': name,
        'year': year,
        'rect_x_start': year + group['jitter_val'] - 0.4,
        'rect_x_end': year + group['jitter_val'] + 0.4,
        'rect_y_start': 0,
        'rect_y_end': group['val'].sum()
    })
    rectangles = pd.concat([rectangle, rectangles], ignore_index=True)

p = (
    p9.ggplot(area_landuse) 
    + p9.geom_col(p9.aes(x='year + jitter_val', y='val', fill='name'), position='stack', alpha=0.8)  # stack columns by GBF4_TARGET
    + p9.theme_bw()
    + p9.labs(
        x='Year',
        y='Area (million km2)',
        fill='Land Use Type'  
    )
    + p9.guides(fill=p9.guide_legend(ncol=1))  # Reduce legend icon size further
    + p9.theme(
        legend_key=p9.element_rect(color=''),  # Set legend key background to white
    )
)

# Export to matplotlib
fig = p.draw()

# Now you can access the underlying matplotlib axes and modify them
ax = fig.axes[0]

# Plot rectangles from the prepared data
for _, row in rectangles.iterrows():
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
ax.legend(handles=legend_patches, title='', loc=(1.045, 0.15), fontsize=10, frameon=False, labels=['GBF-3 ON', 'GBF-3 OFF'])

# Show the final plot
fig.show()













######################## Area - GBF-4 #######################
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250512_DCCEEW_REPORT_03_GBF4_NES_ON_OFF/"
report_data = process_task_root_dirs(task_root_dir)
print(report_data['Type'].unique())

# Get jitter and hatch values
area_landuse = report_data.query('Type == "Area_ag_man_million_km2"').copy()
jitter_map = {'on': -0.5, 'off': 0.5}
hatch_map = {'on': '////', 'off': r'\\\\'}
area_landuse['jitter_val'] = area_landuse['GBF4_TARGET'].map(jitter_map)

# Group by GBF4_TARGET and prepare data for rectangle plotting
rectangles = pd.DataFrame()
for (name, year), group in area_landuse.groupby(['GBF4_TARGET', 'year']):
    rectangle = pd.DataFrame({
        'name': name,
        'year': year,
        'rect_x_start': year + group['jitter_val'] - 0.4,
        'rect_x_end': year + group['jitter_val'] + 0.4,
        'rect_y_start': 0,
        'rect_y_end': group['val'].sum()
    })
    rectangles = pd.concat([rectangle, rectangles], ignore_index=True)

p = (
    p9.ggplot(area_landuse) 
    + p9.geom_col(p9.aes(x='year + jitter_val', y='val', fill='name'), position='stack', alpha=0.8)  # stack columns by GBF4_TARGET
    + p9.theme_bw()
    + p9.labs(
        x='Year',
        y='Area (million km2)',
        fill='Agriculture Management Type'  
    )
    + p9.guides(fill=p9.guide_legend(ncol=1))  # Reduce legend icon size further
    + p9.theme(
        legend_key=p9.element_rect(color=''),  # Set legend key background to white
    )
)

# Export to matplotlib
fig = p.draw()

# Now you can access the underlying matplotlib axes and modify them
ax = fig.axes[0]

# Plot rectangles from the prepared data
for _, row in rectangles.iterrows():
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
ax.legend(handles=legend_patches, title='', loc=(1.045, 0.15), fontsize=10, frameon=False, labels=['GBF-4 ON', 'GBF-4 OFF'])

# Show the final plot
fig.show()





####################### BIO GBF-8  #######################
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250512_DCCEEW_REPORT_04_GBF8_BIO_CLIMATE_OFF/"
report_data = process_task_root_dirs(task_root_dir)
print(report_data['Type'].unique())


# Get jitter and hatch values
area_landuse = report_data.query('Type == "Area_non_ag_lu_million_km2"').copy()
jitter_map = {'on': -0.5, 'off': 0.5}
hatch_map = {'on': '////', 'off': r'\\\\'}
area_landuse['jitter_val'] = area_landuse['BIODIVERSTIY_TARGET_GBF_8'].map(jitter_map)

# Group by BIODIVERSTIY_TARGET_GBF_8 and prepare data for rectangle plotting
rectangles = pd.DataFrame()
for (name, year), group in area_landuse.groupby(['BIODIVERSTIY_TARGET_GBF_8', 'year']):
    rectangle = pd.DataFrame({
        'name': name,
        'year': year,
        'rect_x_start': year + group['jitter_val'] - 0.4,
        'rect_x_end': year + group['jitter_val'] + 0.4,
        'rect_y_start': 0,
        'rect_y_end': group['val'].sum()
    })
    rectangles = pd.concat([rectangle, rectangles], ignore_index=True)
    
p = (
    p9.ggplot(area_landuse) 
    + p9.geom_col(p9.aes(x='year + jitter_val', y='val', fill='name'), position='stack', alpha=0.8)  # stack columns by CONNECTIVITY_SOURCE
    + p9.theme_bw()
    + p9.labs(
        x='Year',
        y='Area (million km2)',
        fill='Land Use Type'  
    )
    + p9.guides(fill=p9.guide_legend(ncol=1))  # Reduce legend icon size further
    + p9.theme(
        legend_key=p9.element_rect(color=''),  # Set legend key background to white
    )
)


# Export to matplotlib
fig = p.draw()

# Now you can access the underlying matplotlib axes and modify them
ax = fig.axes[0]
# Plot rectangles from the prepared data
for _, row in rectangles.iterrows():
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
ax.legend(handles=legend_patches, title='', loc=(1.045, 0.12), fontsize=10, frameon=False, labels=['GBF-8 ON', 'GBF-8 OFF'])
# Show the final plot
fig.show()






####################### No-Go  #######################
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250512_DCCEEW_REPORT_05_NOG_GO_ON_OFF/"
report_data = process_task_root_dirs(task_root_dir)
print(report_data['Type'].unique())


# Get jitter and hatch values
area_ep = report_data.query('Type == "Area_non_ag_lu_million_km2"').query('name == "Environmental plantings (mixed species)"').copy()
area_win_cereal = report_data.query('Type == "Area_all_lu_million_km2"').query('name == "Winter cereals"').copy()
area_landuse = pd.concat([area_ep, area_win_cereal], ignore_index=True)

jitter_map = {True: -0.5, False: 0.5}
hatch_map = {True: '////', False: r'\\\\'}
area_landuse['jitter_val'] = area_landuse['EXCLUDE_NO_GO_LU'].map(jitter_map)


# Group by EXCLUDE_NO_GO_LU and prepare data for rectangle plotting
rectangles = pd.DataFrame()
for (name, year), group in area_landuse.groupby(['EXCLUDE_NO_GO_LU', 'year']):
    rectangle = pd.DataFrame({
        'name': name,
        'year': year,
        'rect_x_start': year + group['jitter_val'] - 0.4,
        'rect_x_end': year + group['jitter_val'] + 0.4,
        'rect_y_start': 0,
        'rect_y_end': group['val'].sum()
    })
    rectangles = pd.concat([rectangle, rectangles], ignore_index=True)
    
p = (
    p9.ggplot(area_landuse) 
    + p9.geom_col(p9.aes(x='year + jitter_val', y='val', fill='name'), position='stack', alpha=0.8)  # stack columns by CONNECTIVITY_SOURCE
    + p9.theme_bw()
    + p9.labs(
        x='Year',
        y='Area (million km2)',
        fill='Land Use Type'  
    )
    + p9.guides(fill=p9.guide_legend(ncol=1))  # Reduce legend icon size further
    + p9.theme(
        legend_key=p9.element_rect(color=''),  # Set legend key background to white
    )
)


# Export to matplotlib
fig = p.draw()

# Now you can access the underlying matplotlib axes and modify them
ax = fig.axes[0]
# Plot rectangles from the prepared data
for _, row in rectangles.iterrows():
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
ax.legend(handles=legend_patches, title='', loc=(1.04, 0.3), fontsize=10, frameon=False, labels=['No-go ON', 'No-go OFF'])
# Show the final plot
fig.show()



####################### BIO DIFF HCAS  #######################
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250512_DCCEEW_REPORT_06_HCAS_50TH_CUSTOM/"
report_data = process_task_root_dirs(task_root_dir)
print(report_data['Type'].unique())


# Get jitter and hatch values
area_landuse = report_data.query('Type == "Area_ag_man_million_km2"').copy()
jitter_map = {'USER_DEFINED': -0.5, 'HCAS': 0.5}
hatch_map = {'USER_DEFINED': '////', 'HCAS': r'\\\\'}
area_landuse['jitter_val'] = area_landuse['HABITAT_CONDITION'].map(jitter_map)

# Group by HABITAT_CONDITION and prepare data for rectangle plotting
rectangles = pd.DataFrame()
for (name, year), group in area_landuse.groupby(['HABITAT_CONDITION', 'year']):
    rectangle = pd.DataFrame({
        'name': name,
        'year': year,
        'rect_x_start': year + group['jitter_val'] - 0.4,
        'rect_x_end': year + group['jitter_val'] + 0.4,
        'rect_y_start': 0,
        'rect_y_end': group['val'].sum()
    })
    rectangles = pd.concat([rectangle, rectangles], ignore_index=True)
    
p = (
    p9.ggplot(area_landuse) 
    + p9.geom_col(p9.aes(x='year + jitter_val', y='val', fill='name'), position='stack', alpha=0.8)  # stack columns by CONNECTIVITY_SOURCE
    + p9.theme_bw()
    + p9.labs(
        x='Year',
        y='Area (million km2)',
        fill='Agriculture Management Type'  
    )
    + p9.guides(fill=p9.guide_legend(ncol=1))  # Reduce legend icon size further
    + p9.theme(
        legend_key=p9.element_rect(color=''),  # Set legend key background to white
    )
)


# Export to matplotlib
fig = p.draw()

# Now you can access the underlying matplotlib axes and modify them
ax = fig.axes[0]
# Plot rectangles from the prepared data
for _, row in rectangles.iterrows():
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
ax.legend(handles=legend_patches, title='', loc=(1.045, 0.12), fontsize=10, frameon=False, labels=['USER_DEFINED', 'HCAS'])
# Show the final plot
fig.show()

