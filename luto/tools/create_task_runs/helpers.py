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

import os, re, json
import shutil, itertools, subprocess, zipfile
import pandas as pd

from tqdm.auto import tqdm
from typing import Literal
from joblib import delayed, Parallel

from luto import settings
from luto.tools.create_task_runs.parameters import EXCLUDE_DIRS, SERVER_PARAMS


def get_settings_df(task_root_dir:str) -> pd.DataFrame:
    '''
    Save the default settings file to a datafram.
    '''

    # Save the settings template to the root task folder
    if not os.path.exists(task_root_dir):
        os.makedirs(task_root_dir, exist_ok=True)
    
    # Get the settings from luto.settings
    with open('luto/settings.py', 'r') as file, \
         open(f'{task_root_dir}/non_str_val.txt', 'w') as non_str_val_file:
        
        # Regex patterns that matches variable assignments from settings
        lines = file.readlines()
        parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=") # Keys are uppercase and start with a letter
        settings_keys = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]

        # Reorder the settings dictionary to match the order in the settings.py file
        settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
        settings_dict = {key: settings_dict[key] for key in settings_keys if key in settings_dict}

        # Write the non-string values to a file; this helps to evaluate the settings later
        for k, v in settings_dict.items():
            if not isinstance(v, str):
                non_str_val_file.write(f'{k}\n')

    # Create a template for custom settings
    settings_df = pd.DataFrame({k:[v] for k,v in settings_dict.items()}).T.reset_index()
    settings_df.columns = ['Name','Default_run']
    settings_df = settings_df.map(str)    
         
    return settings_df



def get_grid_search_param_df(grid_dict:dict) -> None:
    '''
    Permutate the grid search parameters and save them to a datafram.
    '''
    # Create a list of dictionaries with all possible permutations
    grid_dict = {k: [str(i) for i in v] for k, v in grid_dict.items()}
    keys, values = zip(*grid_dict.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Save the grid search parameters to the root task folder
    permutations_df = pd.DataFrame(permutations)
    permutations_df.insert(0, 'run_idx', [i for i in range(1, len(permutations_df) + 1)])

    # Report the grid search parameters
    print(f'Grid search template has been created with {len(permutations_df)} permutations!')
    for k, v in grid_dict.items():
        if len(v) > 1:
            print(f'    {k:<50} : {len(v)} values')
    
    return permutations_df
    

def update_settings(settings_dict:dict, job_name:str):
    '''
    Update the task run settings with parameters for the server, and change the data path to absolute path.
    E.g. job name, input directory, raw data directory, and threads.
    '''
    settings_dict['JOB_NAME'] = job_name
    settings_dict['INPUT_DIR'] = os.path.abspath(settings_dict['INPUT_DIR']).replace('\\','/')
    settings_dict['RAW_DATA'] = os.path.abspath(settings_dict['RAW_DATA']).replace('\\','/')
    settings_dict['THREADS'] = settings_dict['NCPUS']

    return settings_dict


def get_grid_search_settings_df(task_root_dir:str, settings_df:pd.DataFrame, grid_search_param_df:pd.DataFrame) -> pd.DataFrame:
    '''
    Loop through the grid search parameters and create a settings template for each run.
    '''
    
    template_grid_search = settings_df.copy()
    task_dir = os.path.basename(os.path.normpath(task_root_dir))

    # Loop through the permutations DataFrame and create new columns with updated settings
    run_settings_dfs = []
    for _, row in grid_search_param_df.iterrows():
        settings_dict = template_grid_search.set_index('Name')['Default_run'].to_dict()
        settings_dict.update(row.to_dict())
        settings_dict = update_settings(settings_dict, f'{task_dir}_Run_{row['run_idx']:04}')
        run_settings_dfs.append(pd.Series(settings_dict, name=f'Run_{row['run_idx']:04}'))
    
    template_grid_search = pd.concat(run_settings_dfs, axis=1).reset_index(names='Name')
    template_grid_search.to_csv(f'{task_root_dir}/grid_search_template.csv', index=False)
    
    grid_search_param_df = grid_search_param_df.loc[:, grid_search_param_df.nunique() > 1]
    grid_search_param_df.to_csv(f'{task_root_dir}/grid_search_parameters_unique.csv', index=False)

    return template_grid_search



def copy_folder_custom(source, destination, ignore_dirs=None):
    ignore_dirs = set() if ignore_dirs is None else set(ignore_dirs)
    os.makedirs(destination, exist_ok=True)
    jobs = []
    for item in os.listdir(source):
        if item in ignore_dirs: continue   
        s = os.path.join(source, item)
        d = os.path.join(destination, item)
        jobs += copy_folder_custom(s, d) if os.path.isdir(s) else [(s, d)]
    return jobs   

def create_run_folders(task_root_dir:str, col:str, n_workers:int):
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    dst_dir = f'{task_root_dir}/{col}'
    # Copy the files from the source to the destination
    from_to_files = copy_folder_custom(src_dir, dst_dir, EXCLUDE_DIRS)
    Parallel(n_jobs=n_workers)(delayed(shutil.copy2)(s, d) for s, d in from_to_files)
    # Create an output folder for the task
    os.makedirs(f'{dst_dir}/output', exist_ok=True)
    
    
def submit_task(task_root_dir:str, col:str, mode:Literal['single','cluster'], max_concurrent_tasks): 
    shutil.copyfile('luto/tools/create_task_runs/bash_scripts/task_cmd.sh', f'{task_root_dir}/{col}/task_cmd.sh')
    shutil.copyfile('luto/tools/create_task_runs/bash_scripts/python_script.py', f'{task_root_dir}/{col}/python_script.py')
    
    # Wait until the number of running jobs is less than max_concurrent_tasks
    if os.name == 'posix':
        while True:
            try:
                running_jobs = int(subprocess.run('qselect | wc -l', shell=True, capture_output=True, text=True).stdout.strip())
            except Exception as e:
                print(f"Error checking running jobs: {e}")
            if running_jobs < max_concurrent_tasks:
                break
            else:
                print(f"Max concurrent tasks reached ({running_jobs}/{max_concurrent_tasks}), waiting to submit {col}...")
                import time; time.sleep(10)
        
    # Open log files for the task run
    with open(f'{task_root_dir}/{col}/run_std.log', 'w') as std_file, \
         open(f'{task_root_dir}/{col}/run_err.log', 'w') as err_file:
        if mode == 'single': 
            subprocess.run(['python', 'python_script.py'], cwd=f'{task_root_dir}/{col}', stdout=std_file, stderr=err_file, check=True)
        elif mode == 'cluster' and os.name == 'posix':
            subprocess.run(['bash', 'task_cmd.sh'], cwd=f'{task_root_dir}/{col}', stdout=std_file, stderr=err_file, check=True)
        else:
            raise ValueError('Mode must be either "single" or "cluster"!')

    
def write_settings(task_dir:str, settings_dict:dict):
    with open(f'{task_dir}/luto/settings.py', 'w') as file:
        for k, v in settings_dict.items():
            if isinstance(v, str):
                file.write(f'{k}="{v}"\n')
            else:
                file.write(f'{k}={v}\n')
        
                
def write_terminal_vars(task_dir:str, col:str, settings_dict:dict):
    with open(f'{task_dir}/luto/settings_bash.py', 'w') as bash_file:
        for key, value in settings_dict.items():
            if key not in SERVER_PARAMS:
                continue
            if isinstance(value, str):
                bash_file.write(f'export {key}="{value}"\n')
            else:
                bash_file.write(f'export {key}={value}\n')
        


def create_task_runs(
    task_root_dir:str, 
    custom_settings:pd.DataFrame, 
    mode:Literal['single','cluster']='single', 
    n_workers:int=4,
    max_concurrent_tasks:int=300,
) -> None:
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
    # Replace TRUE/FALSE (Excel) with True/False (Python)
    custom_settings = custom_settings.replace({'TRUE': 'True', 'FALSE': 'False'})
    # Check if there are any custom settings
    if custom_settings.columns.size == 0:
        raise ValueError('No custom settings found in the settings_template.csv file!')
    # Evaluate settings that are not originally strings
    with open(f'{task_root_dir}/non_str_val.txt', 'r') as file:
        eval_vars = file.read().splitlines()
        custom_settings.loc[eval_vars] = custom_settings.loc[eval_vars].map(str).map(eval)
        
    def task_wraper(col):
        settings_dict = custom_settings.loc[:, col].copy()
        create_run_folders(task_root_dir, col, n_workers)
        write_settings(f'{task_root_dir}/{col}', settings_dict)
        write_terminal_vars(f'{task_root_dir}/{col}', col, settings_dict)
        submit_task(task_root_dir, col, mode, max_concurrent_tasks)
    
    # Run the tasks in parallel
    tasks = [delayed(task_wraper)(col) for col in custom_settings.columns]
    for result in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
        pass



def get_json_data_from_zip(zip_path, json_filename):
    with zipfile.ZipFile(os.path.join(zip_path), 'r') as zip_ref:
        with zip_ref.open(f'DATA_REPORT/data/{json_filename}') as f:
            json_part = f.read().decode('utf-8').split('=')[1][:-2] # split at '=' and remove the trailing ';\n'
            json_data = json.loads(json_part)
    return json_data



def return_df_ag(json_data):
    '''
    Ag data is 3-level nested, with region, water supply, and records.
    '''
    records_with_metadata = []
    for region, region_data in json_data.items():
        for water_supply, records in region_data.items():
            for record in records:
                # Expand data points with metadata in one go
                for year, value in record['data']:
                    records_with_metadata.append({
                        'region': region,
                        'water_supply': water_supply, 
                        'name': record['name'],
                        'year': int(year),
                        'value': value
                    })
    return pd.DataFrame(records_with_metadata).query('water_supply != "ALL"')


def return_df_ag_mgt(json_data):
    '''
    Ag management data is 4-level nested, with region, ag-management, water supply, and records.
    '''
    records_with_metadata = []
    for region, region_data in json_data.items():
        for ag_mgt, ag_mgt_data in region_data.items():
            for water_supply, records in ag_mgt_data.items():
                for record in records:
                    # Expand data points with metadata in one go
                    for year, value in record['data']:
                        records_with_metadata.append({
                            'region': region,
                            'ag_mgt': ag_mgt,
                            'water_supply': water_supply, 
                            'name': record['name'],
                            'year': int(year),
                            'value': value
                        })
    return pd.DataFrame(records_with_metadata).query('ag_mgt != "ALL" and water_supply != "ALL"')

def return_df_plain(json_data):
    '''
    Plain data is 2-level nested, with region and records.
    '''
    records_with_metadata = []
    for region, records in json_data.items():
        for record in records:
            # Expand data points with metadata in one go
            for year, value in record['data']:
                records_with_metadata.append({
                    'region': region,
                    'name': record['name'],
                    'year': int(year),
                    'value': value
                })
    return pd.DataFrame(records_with_metadata)



def process_area_category(json_dir_path):
    json_data = get_json_data_from_zip(json_dir_path, 'Area_overview_1_Land-use.js')
    return return_df_plain(json_data)

def process_area_non_ag_lu(json_dir_path):
    json_data = get_json_data_from_zip(json_dir_path, 'Area_NonAg.js')
    return return_df_plain(json_data)

def process_area_ag_man(json_dir_path):
    json_data = get_json_data_from_zip(json_dir_path, 'Area_Am.js')
    return return_df_ag_mgt(json_data)

def process_economic_data(json_dir_path):
    json_data = get_json_data_from_zip(json_dir_path, 'Economics_overview_sum.js')
    return return_df_plain(json_data)

def process_production_deviation_data(json_dir_path):
    json_data = get_json_data_from_zip(json_dir_path, 'Production_overview_AUS_achive_percent.js')
    return return_df_plain(json_data)

def process_GHG_data(json_dir_path):
    json_data = get_json_data_from_zip(json_dir_path, 'GHG_overview_sum.js')
    return return_df_plain(json_data)

def process_bio_obj_data(json_dir_path):
    json_data = get_json_data_from_zip(json_dir_path, 'BIO_GBF2_overview_sum.js')
    return return_df_plain(json_data)



def get_report_df(json_dir_path, run_paras):
    
    df_area_all_lu = process_area_category(json_dir_path)
    df_area_non_ag_lu = process_area_non_ag_lu(json_dir_path)
    df_area_ag_man = process_area_ag_man(json_dir_path)
    df_economy = process_economic_data(json_dir_path)
    df_ghg = process_GHG_data(json_dir_path)
    df_demand_deviation = process_production_deviation_data(json_dir_path)
    df_bio_pct = process_bio_obj_data(json_dir_path)

    report_df = pd.concat([
        df_area_all_lu.assign(Type='Area_broad_category_ha'),
        df_area_non_ag_lu.assign(Type='Area_non_ag_lu_ha'),
        df_area_ag_man.assign(Type='Area_ag_man_ha'),
        df_economy.assign(Type='Economic_AUD'),
        df_demand_deviation.assign(Type='Production_deviation_percent'),
        df_ghg.assign(Type='GHG_tCO2e'),
        df_bio_pct.assign(Type='Bio_relative_to_PRE1750_percent'),
    ]).assign(**run_paras).reset_index(drop=True)

    return report_df



def process_task_root_dirs(task_root_dir, n_workers=10):
    
    grid_search_params = pd.read_csv(f"{task_root_dir}/grid_search_parameters_unique.csv")
    run_dirs = [i for i in os.listdir(task_root_dir) if os.path.isdir(os.path.join(task_root_dir, i))]
    run_dirs = sorted([i for i in run_dirs if 'Run_' in i])
    
    tasks = []
    for run_dir in run_dirs:
        run_idx = int(run_dir.split('_')[-1])
        run_paras = grid_search_params.query(f'run_idx == {int(run_idx)}').to_dict(orient='records')[0]

        # Depending on output structure, the report can be found in different places
        json_dir_path = os.path.join(task_root_dir, run_dir, 'Run_Archive.zip')
        if not os.path.exists(json_dir_path):
            print(f'Warning: No output found for {run_dir}, skipping...')
            continue
        tasks.append(delayed(get_report_df)(json_dir_path, run_paras))
        
        
    # Concatenate the results, only keep the columns with more than 1 unique value
    out_df = pd.concat(
        tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)), 
        ignore_index=True
    )
    
    return out_df

        
