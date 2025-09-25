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

from itertools import product
from tqdm.auto import tqdm

from luto.tools.create_task_runs.create_grid_search_tasks import TASK_ROOT_DIR
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Plot settings
p9.options.figure_size = (12, 6)
p9.options.dpi = 100


# Get the data
task_root_dir = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20250924_RES5_HIGH_HIGH/"
report_data = process_task_root_dirs(task_root_dir).query('region == "AUSTRALIA"').copy()
print(report_data['Type'].unique())



####################### Area #######################


report_area_am = report_data.query('Type == "Area_ag_man_ha"'\
    ).copy(
    ).rename(columns={
        'BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR':'Beef',
        'SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR':'Sheep',
        }
    ).groupby(['run_idx','Beef', 'Sheep', 'year','ag_mgt']
    ).sum(numeric_only=True
    ).reset_index()
    
    
fig = (
    p9.ggplot(report_area_am) +
    p9.geom_col(
        p9.aes(
            x='year', 
            y='value', 
            fill='ag_mgt'
        ), 
        position='stack', 
        width=0.7
    ) +
    p9.facet_grid(
        'Sheep ~ Beef',
        labeller='label_both'
    ) +
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
        title='Area of Agricultural Management (ha)',
        x='Year',
        y='Area (ha)',
    )
)




