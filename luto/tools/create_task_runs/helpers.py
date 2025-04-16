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

import os, re, time, json
import shutil, psutil, itertools, subprocess
import numpy as np
import pandas as pd

from glob import glob
from tqdm.auto import tqdm
from typing import Literal
from datetime import datetime
from joblib import delayed, Parallel

from luto import settings
from luto.tools.create_task_runs.parameters import (
    BIO_TARGET_ORDER, 
    EXCLUDE_DIRS, 
    GHG_ORDER, 
    TASK_ROOT_DIR
)






def copy_settings_as_df() -> pd.DataFrame:
    '''
    Save {k:v} in the settings.py file to a DataFrame.
    '''

    # Save the settings template to the root task folder
    if not os.path.exists(TASK_ROOT_DIR):
        os.makedirs(TASK_ROOT_DIR, exist_ok=True)
    
    # Get the settings from luto.settings
    with open('luto/settings.py', 'r') as file, \
         open(f'{TASK_ROOT_DIR}/non_str_val.txt', 'w') as non_str_val_file:
        
        # Regex patterns that matches variable assignments from settings
        lines = file.readlines()
        parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=") # Keys are uppercase and start with a letter
        settings_keys = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]

        # Reorder the settings dictionary to match the order in the settings.py file
        settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
        settings_dict = {key: settings_dict[key] for key in settings_keys if key in settings_dict}
        
        # Add parameters for the job submission
        settings_dict['JOB_NAME'] = 'auto'
        settings_dict['MEM'] = 'auto'
        settings_dict['QUEUE'] = 'normal'
        settings_dict['WRITE_THREADS'] = 10                     # 10 threads for writing is a safe number to avoid out-of-memory issues
        settings_dict['KEEP_OUTPUTS'] = True
        settings_dict['NCPUS'] = settings_dict['THREADS']//4*4  # Round down to the nearest multiple of 4
        settings_dict['TIME'] = '10:00:00'

        # Write the non-string values to a file
        for k, v in settings_dict.items():
            if not isinstance(v, str):
                non_str_val_file.write(f'{k}\n')

    # Create a template for custom settings
    settings_df = pd.DataFrame({k:[v] for k,v in settings_dict.items()}).T.reset_index()
    settings_df.columns = ['Name','Default_run']
    settings_df = settings_df.map(str)    
         
    return settings_df



def create_grid_search_permutations(grid_dict:dict) -> None:
    '''
    Permutate the grid search parameters and save them to a CSV file.\n
    '''
    # Create a list of dictionaries with all possible permutations
    grid_dict = {k: [str(i) for i in v] for k, v in grid_dict.items()}
    keys, values = zip(*grid_dict.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Save the grid search parameters to the root task folder
    permutations_df = pd.DataFrame(permutations)
    permutations_df.insert(0, 'run_idx', [i for i in range(1, len(permutations_df) + 1)])
    permutations_df.to_csv(f'{TASK_ROOT_DIR}/grid_search_parameters.csv', index=False)

    print(f'Grid search template has been created with {len(permutations_df)} permutations!')
    
    

def create_grid_search_settings_df(settings_df:pd.DataFrame = copy_settings_as_df()) -> pd.DataFrame:
    '''
    Create a dataframe that each row is an independent settings for a run.\n
    '''
    
    # Read the settings template file and the grid search parameters
    template_grid_search = settings_df.copy()
    permutations_df = pd.read_csv(f'{TASK_ROOT_DIR}/grid_search_parameters.csv')


    # Loop through the permutations DataFrame and create new columns with updated settings
    for _, row in permutations_df.iterrows():
        # Replace the settings using the key-value pairs in the permutation item
        new_column = template_grid_search['Default_run'].copy() 
        for k, v in row.items():
            if k != 'run_idx':
                new_column.loc[settings_df['Name'] == k] = v
                
        # Rename the column and add it to the DataFrame
        new_column.name = f'Run_{row["run_idx"]}'
        template_grid_search = pd.concat([template_grid_search, new_column.rename(f'Run_{row["run_idx"]:04}')], axis=1)
        
    # Check for NANs and save the template CSV
    num_nan = template_grid_search.isna().sum().sum()
    if num_nan > 0:
        print(f'{num_nan} NANs found in the template CSV! \n Check settings before proceeding!') 
        
    template_grid_search.to_csv(f'{TASK_ROOT_DIR}/grid_search_template.csv', index=False)

    return template_grid_search


def create_task_runs(custom_settings:pd.DataFrame, mode:Literal['single','cluster']='single', n_workers:int=4):
    '''
    Submit the tasks to the cluster using the custom settings.\n
    Parameters
     custom_settings (pd.DataFrame):The custom settings DataFrame.
     python_path (str, only works if mode == "single"): The path to the python executable.
     mode (str): The mode to submit the tasks. Options are "single" or "cluster".
     n_workers (int): The number of workers to use for parallel processing. 
    '''
    
    if mode not in ['single', 'cluster']:
        raise ValueError('Mode must be either "single" or "cluster"!')
   
    # Read the custom settings file
    custom_settings = custom_settings.dropna(how='all', axis=1)
    custom_settings = custom_settings.set_index('Name')
    
    # Replace special characters to underscore, making the column names valid python variables
    custom_settings = custom_settings.replace({'TRUE': 'True', 'FALSE': 'False'})
    custom_cols = [col for col in custom_settings.columns if col not in ['Default_run']]
    
    # Check if there are any custom settings
    if not custom_cols:
        raise ValueError('No custom settings found in the settings_template.csv file!')
    
    # Eval the non-string settings back to it type
    with open(f'{TASK_ROOT_DIR}/non_str_val.txt', 'r') as file:
            eval_vars = file.read().splitlines()
    
    # Skip run if report dir exist
    for col in custom_cols:
        custom_settings.loc[eval_vars, col] = custom_settings.loc[eval_vars, col].map(str).map(eval)
        if os.path.isdir(f'{TASK_ROOT_DIR}/{col}/DATA_REPORT/'):
            return f'Skip {col} because report folder found there ...'

    def process_col(col):
        custom_dict = update_settings(custom_settings[col].to_dict(), col)
        create_run_folders(col)
        write_custom_settings(f'{TASK_ROOT_DIR}/{col}', custom_dict)
        return submit_task(col, mode=mode)

    # Submit the tasks in parallel; Using 4 threads is a safe number to submit
    # tasks in login node. Or use the specified number of cpus if not in a linux system
    workers = min(n_workers, len(custom_cols))
    for msg in Parallel(n_jobs=workers, return_as='generator')(delayed(process_col)(col) for col in custom_cols):
        print(msg)


def copy_folder_custom(source, destination, ignore_dirs=None):
    
    ignore_dirs = set() if ignore_dirs is None else set(ignore_dirs)

    jobs = []
    os.makedirs(destination, exist_ok=True)
    for item in os.listdir(source):
        
        if item in ignore_dirs: continue   

        s = os.path.join(source, item)
        d = os.path.join(destination, item)
        jobs += copy_folder_custom(s, d) if os.path.isdir(s) else [(s, d)]
        
    return jobs      



def write_custom_settings(task_dir:str, settings_dict:dict):
    # Write the custom settings to the settings.py of each task
    with open(f'{task_dir}/luto/settings.py', 'w') as file, \
         open(f'{task_dir}/luto/settings_bash.py', 'w') as bash_file:
        
        for k, v in settings_dict.items():
            # List values need to be converted to bash arrays
            if isinstance(v, list):
                bash_file.write(f'{k}=({ " ".join([str(elem) for elem in v])})\n')
                file.write(f'{k}={v}\n')
            # Dict values need to be converted to bash variables
            elif isinstance(v, dict):
                file.write(f'{k}={v}\n')
                bash_file.write(f'# {k} is a dictionary, which is not natively supported in bash\n')
                for key, value in v.items():
                    key = str(key).replace(' ', '_').replace('(','').replace(')','')
                    if isinstance(value, list):
                        bash_file.write(f'{k}_{key}=({ " ".join([str(elem) for elem in value])})\n')
                    else:
                        bash_file.write(f'{k}_{key}={value}\n') 
            # If the value is a string, write it as a string
            elif isinstance(v, str):
                file.write(f'{k}="{v}"\n')
                bash_file.write(f'{k}="{v}"\n')
            # Write the rest as it is
            else:
                file.write(f'{k}={v}\n')
                bash_file.write(f'{k}={v}\n')
                


def update_settings(settings_dict:dict, col:str):

    # The input dir for each task will point to the absolute path of the input dir
    settings_dict['INPUT_DIR'] = os.path.abspath(settings_dict['INPUT_DIR']).replace('\\','/')
    settings_dict['DATA_DIR'] = settings_dict['INPUT_DIR']

    # Set the memory and time based on the resolution factor
    if int(settings_dict['RESFACTOR']) == 1:
        MEM = "250G"
    elif int(settings_dict['RESFACTOR']) == 2:
        MEM = "150G" 
    elif int(settings_dict['RESFACTOR']) <= 5:
        MEM = "100G"
    elif int(settings_dict['RESFACTOR']) <= 10:
        MEM = "80G"
    else:
        MEM = "40G"
        
    # Update the settings dictionary
    settings_dict['JOB_NAME'] = settings_dict['JOB_NAME'] if settings_dict['JOB_NAME'] != 'auto' else col
    settings_dict['MEM'] = settings_dict['MEM'] if settings_dict['MEM'] != 'auto' else MEM
    settings_dict['THREADS'] = int(settings_dict['NCPUS'])

    return settings_dict

    
    
    
    
def create_run_folders(col):
    # Change the directory to the root of the project
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('../../..')
    # Copy codes to the each custom run folder, excluding {EXCLUDE_DIRS} directories
    from_to_files = copy_folder_custom(os.getcwd(), f'{TASK_ROOT_DIR}/{col}', EXCLUDE_DIRS)
    worker = min(settings.WRITE_THREADS, len(from_to_files))
    Parallel(n_jobs=worker)(delayed(shutil.copy2)(s, d) for s, d in from_to_files)
    # Create an output folder for the task
    os.makedirs(f'{TASK_ROOT_DIR}/{col}/output', exist_ok=True)




def submit_task(col:str, mode:Literal['single','cluster']='single'):
    # Copy the slurm script to the task folder
    shutil.copyfile('luto/tools/create_task_runs/bash_scripts/task_cmd.sh', f'{TASK_ROOT_DIR}/{col}/task_cmd.sh')
    shutil.copyfile('luto/tools/create_task_runs/bash_scripts/python_script.py', f'{TASK_ROOT_DIR}/{col}/python_script.py')
    
    time.sleep(np.random.choice([1, 2]))
    
    if mode == 'single': 
        # Submit the task
        while True:
            try:
                subprocess.run(['python', 'python_script.py'], cwd=f'{TASK_ROOT_DIR}/{col}', check=True)
                break  # Exit the loop if the task succeeds
            except subprocess.CalledProcessError:
                print(f"Task {col} failed. Retrying in 10 seconds...")
                time.sleep(10)
    
    # Start the task if the os is linux
    elif mode == 'cluster' and os.name == 'posix':
        # Submit the task to the cluster
        subprocess.run(['bash', 'task_cmd.sh'], cwd=f'{TASK_ROOT_DIR}/{col}')
    
    else:
        raise ValueError('Mode must be either "single" or "cluster"!')
    
    return f'Task {col} has been submitted!'


def log_memory_usage(output_dir=settings.OUTPUT_DIR, mode='a', interval=1):
    '''
    Log the memory usage of the current process to a file.
    Parameters
        output_dir (str): The directory to save the memory log file.
        mode (str): The mode to open the file. Default is 'a' (append).
        interval (int): The interval in seconds to log the memory usage.
    '''
    
    with open(f'{output_dir}/RES_{settings.RESFACTOR}_{settings.MODE}_mem_log.txt', mode=mode) as file:
        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss
            children = process.children(recursive=True)
            # Include the memory usage of the child processes to get accurate memory usage under parallel processing
            if children:
                memory_usage += sum(child.memory_info().rss for child in children)
            memory_usage /= (1024 * 1024 * 1024)
            file.write(f'{timestamp}\t{memory_usage:.2f}\n')
            file.flush()  # Ensure data is written to the file immediately
            time.sleep(interval)
            
            

def load_json_data(json_path, filename):
    if not os.path.exists(os.path.join(json_path, filename)):
        run_idx = re.compile(r'(Run_\d+)').findall(json_path)[0]
        print(f'{run_idx}: File missing - {filename}!')
        return pd.DataFrame({'year': [], 'val': [], 'name': []})
    with open(os.path.join(json_path, filename)) as f:
        return pd.json_normalize(json.load(f), 'data', ['name']).rename(columns={0: 'year', 1: 'val'})

def process_profit_data(json_path):
    df = load_json_data(json_path, 'economics_0_rev_cost_all_wide.json')
    return df.query('name == "Profit"')

def process_GHG_deviation_data(json_path):
    df = load_json_data(json_path, 'GHG_2_individual_emission_Mt.json')
    df_target = df.query('name == "GHG emissions limit"')
    df_actual = df.query('name == "Net emissions"')
    df_deviation = df_target.merge(df_actual, on='year', suffixes=('_target', '_actual'))
    df_deviation['name'] = 'GHG deviation'
    df_deviation['val'] = df_deviation['val_actual'] - df_deviation['val_target']
    return df_deviation

def process_GHG_deforestation_data(json_path):
    
    df = load_json_data(json_path, 'GHG_2_individual_emission_Mt.json')
    if df.empty:
        return df
    
    df = df.query('name in ["Livestock natural to modified", "Unallocated natural to modified"]'
            ).pivot(index='year', columns='name', values='val'
            ).reset_index(
            ).eval("Total_Deforestation = `Livestock natural to modified` + `Unallocated natural to modified`"
            ).melt(
                id_vars='year', 
                value_vars=['Livestock natural to modified', 'Unallocated natural to modified', 'Total_Deforestation'], 
                var_name='name', 
                value_name='val')
    return df

def process_bio_GBF2_data(json_path):
    return load_json_data(json_path, 'biodiversity_priority_1_total_score_by_type.json')


def process_demand_data(json_path):
    df_demand = load_json_data(json_path, 'production_5_6_demand_Production_commodity_from_LUTO.json')
    df_luto = load_json_data(json_path, 'production_5_5_demand_Production_commodity.json')
    df_delta = df_demand.merge(df_luto, on=['year', 'name'], suffixes=('_luto', '_demand'))
    # df_delta['deviation_t'] = df_delta.eval('val_luto - val_demand')
    df_delta['val'] = df_delta.eval('(val_luto - val_demand) / val_demand * 100')
    return df_delta


def get_report_df(json_path, run_paras):
    
    df_profit = process_profit_data(json_path)
    df_ghg_deviation = process_GHG_deviation_data(json_path)
    df_ghg_deforestation = process_GHG_deforestation_data(json_path)
    df_demand_deviation = process_demand_data(json_path)
    df_bio_objective = process_bio_GBF2_data(json_path)

    report_df = pd.concat([
        df_profit[['year', 'name', 'val']].assign(Type='Profit_billion_AUD'),
        df_demand_deviation[['year', 'name', 'val']].assign(Type='Production_Mt'),
        df_ghg_deviation[['year', 'name', 'val']].assign(Type='GHG_Deviation_pct'),
        df_ghg_deforestation[['year', 'name', 'val']].assign(Type='GHG_Deforestration_Mt'),
        df_bio_objective[['year', 'name', 'val']].assign(Type='Biodiversity_area_score'),
    ]).assign(**run_paras).reset_index(drop=True)

    report_df['GHG_LIMITS_FIELD'] = report_df['GHG_LIMITS_FIELD'].replace(GHG_ORDER)
    report_df['BIODIV_GBF_TARGET_2_DICT'] = report_df['BIODIV_GBF_TARGET_2_DICT'].replace(BIO_TARGET_ORDER)

    return report_df



def process_task_root_dirs(task_root_dir, n_workers=4):
    

    grid_search_params = pd.read_csv(f"{task_root_dir}/grid_search_parameters.csv")
    run_dirs = [i for i in os.listdir(task_root_dir) if os.path.isdir(os.path.join(task_root_dir, i))]
    run_dirs = [i for i in run_dirs if 'Run_' in i]
    
    tasks = []
    for dir in run_dirs:
        run_idx = int(dir.split('_')[-1])
        run_paras = grid_search_params.query(f'run_idx == {int(run_idx)}').to_dict(orient='records')[0]
        # Locate the latest json data path
        json_path = next(
            # Find json in 'output' folder
            iter(sorted(glob(f"{task_root_dir}/{dir}/output/**/DATA_REPORT/data"), key=os.path.getmtime)),
            # Find json in 'DATA_REPORT' folder
            os.path.join(task_root_dir, dir, 'DATA_REPORT', 'data')
        )
        if not os.path.exists(json_path):
            print(f'{dir}: No json outputs find in Run_{run_idx}!')
            continue
        # Json to df
        tasks.append(delayed(get_report_df)(json_path, run_paras))
    
    return pd.concat(tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)), ignore_index=True)

        
