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
from luto.tools.create_task_runs.create_grid_search_tasks import TASK_ROOT_DIR
from luto.tools.create_task_runs.helpers import process_task_root_dirs


# Plot settings
p9.options.figure_size = (12, 6)
p9.options.dpi = 100


# Get the data
task_root_dir = TASK_ROOT_DIR.rstrip('/')       # Or replace with the desired task run root dir
report_data = process_task_root_dirs(task_root_dir)


# -------------- Plot the transition cost (Demand's deviation) ----------------------
query_str = '''
    name == "Transition cost (Ag2Ag)" 
    and year != 2010
    '''.replace('\n', ' ').replace('  ', ' ')
    
df_economics = report_data.query(query_str).copy()

valid_runs_demand = set(report_data['run_idx']) - set(
    df_economics.query('abs(val) >= 40')['run_idx']
)

df_economics = df_economics.query('run_idx.isin(@valid_runs_demand)')

# Plot economic's deviation landscape without filtering
plot_landscape_demand = (
    p9.ggplot(
        df_economics,
        p9.aes(
            x='year',
            y='val', 
            color='SOLVE_WEIGHT_BETA',
            group='run_idx',
        )
    ) +
    p9.geom_line() +
    p9.facet_grid('BIODIVERSTIY_TARGET_GBF_2~GHG_EMISSIONS_LIMITS') +
    p9.theme_bw() +
    p9.theme(
        strip_text=p9.element_text(size=8), 
        legend_position='bottom',
        legend_title=p9.element_blank(),
        legend_box='horizontal'
    ) 
)





