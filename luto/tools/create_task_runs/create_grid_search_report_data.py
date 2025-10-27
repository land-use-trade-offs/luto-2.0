import re, pathlib
import pandas as pd
import zipfile

from glob import glob



# Define the root dir for the task runs
TASK_ROOT_DIR = "/g/data/jk53/jinzhu/LUTO/Custom_runs/20251024_RES5_YERA5_RUNS/"


# Get all run directories, create report data directory
run_dirs = sorted(glob(f"{TASK_ROOT_DIR}/Run_*"))
report_data_dir = pathlib.Path(f"{TASK_ROOT_DIR}/Grid_Search_Report_Data")
report_data_dir.mkdir(exist_ok=True)

# Extract data from each run directory
for run_dir in run_dirs:
    # Skip if no archive found
    if not pathlib.Path(f"{run_dir}/Run_Archive.zip").exists():
        print(f"Skipping {run_dir} as no Run_Archive.zip found.")
        continue
    
    # Extract report data from the archive
    with zipfile.ZipFile(f"{run_dir}/Run_Archive.zip", 'r') as zip_ref:
        zip_ref.extractall(f"{report_data_dir}/{pathlib.Path(run_dir).name}", members=[
            f"{pathlib.Path(run_dir).name}/Report_Data/Report_Data.csv"
        ])
    









