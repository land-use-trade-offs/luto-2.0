import os
import pandas as pd
import numpy as np
from luto.tools.create_task_runs.helpers import (
    create_grid_search_parameters, 
    create_grid_search_template, 
    create_task_runs
)
from luto.tools.create_task_runs.parameters import TASK_ROOT_DIR



# Create the task runs
grid_search_df = pd.read_csv(f"{TASK_ROOT_DIR}/grid_search_template.csv")


# 1) Submit task to a single linux machine, and run simulations parallely
create_task_runs(grid_search_df, mode='single', python_path='~/miniforge3/envs/luto/bin/python', n_workers=40, waite_mins=1.5)

# # 2) Submit task to multiple linux computation nodes
# create_task_runs(grid_search_df, mode='multiple')

# # 3) Submit task to a single windows machine, and run simulations parallely
# create_task_runs(grid_search_df, mode='single', python_path='F:/jinzhu/conda_env/luto/python.exe', n_workers=10, waite_mins=1.5)


