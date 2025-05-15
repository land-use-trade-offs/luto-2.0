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
from luto.tools.create_task_runs.parameters import (
    BIO_TARGET_ORDER, 
    EXCLUDE_DIRS, 
    GHG_ORDER
)


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



def get_grid_search_param_df(task_root_dir:str, grid_dict:dict) -> None:
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
    permutations_df.to_csv(f'{task_root_dir}/grid_search_parameters.csv', index=False)

    # Report the grid search parameters
    print(f'Grid search template has been created with {len(permutations_df)} permutations!')
    for k, v in grid_dict.items():
        if len(v) > 1:
            print(f'    {k:<30} : {len(v)} values')
    
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
    
    
def submit_task(task_root_dir:str, col:str, mode:Literal['single','cluster']='single'): 
    shutil.copyfile('luto/tools/create_task_runs/bash_scripts/task_cmd.sh', f'{task_root_dir}/{col}/task_cmd.sh')
    shutil.copyfile('luto/tools/create_task_runs/bash_scripts/python_script.py', f'{task_root_dir}/{col}/python_script.py')
    
    with open(f'{task_root_dir}/{col}/run_std.log', 'w') as std_file, \
         open(f'{task_root_dir}/{col}/run_err.log', 'w') as err_file:
        if mode == 'single': 
            subprocess.run(['python', 'python_script.py'], cwd=f'{task_root_dir}/{col}', stdout=std_file, stderr=err_file)
        elif mode == 'cluster' and os.name == 'posix':
            subprocess.run(['bash', 'task_cmd.sh'], cwd=f'{task_root_dir}/{col}', stdout=std_file, stderr=err_file)
        else:
            raise ValueError('Mode must be either "single" or "cluster"!')
        
        return f'Task {col} has been submitted!'
    
    
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
            if key not in ['MEM', 'NCPUS', 'TIME', 'QUEUE', 'JOB_NAME']:
                continue
            if isinstance(value, str):
                bash_file.write(f'export {key}="{value}"\n')
            else:
                bash_file.write(f'export {key}={value}\n')
        


def create_task_runs(
    task_root_dir:str, 
    custom_settings:pd.DataFrame, 
    mode:Literal['single','cluster']='single', 
    n_workers:int=4
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
    
    # Create tasks
    tasks = []  
    for col in custom_settings.columns:
        settings_dict = custom_settings.loc[:, col].copy()
        create_run_folders(task_root_dir, col, n_workers)
        write_settings(f'{task_root_dir}/{col}', settings_dict)
        write_terminal_vars(f'{task_root_dir}/{col}', col, settings_dict)
        tasks.append(delayed(submit_task)(task_root_dir, col, mode=mode))
 
    for msg in Parallel(n_jobs=min(n_workers, len(tasks)), return_as='generator')(tasks):
        print(msg)


   
   
      
def return_empty_df(json_dir_path, filename):
    run_idx = re.compile(r'(Run_\d+)').findall(json_dir_path)[0]
    print(f'{run_idx}: File missing - {filename}!')
    return pd.DataFrame({'year': [], 'val': [], 'name': []})

def return_zipped_df(json_dir_path, filename):
    with zipfile.ZipFile(os.path.join(json_dir_path), 'r') as zip_ref:
        with zip_ref.open(f'data/{filename}') as f:
            return pd.json_normalize(json.load(f), 'data', ['name']).rename(columns={0: 'year', 1: 'val'})

def return_plain_df(json_dir_path, filename):
    with open(os.path.join(json_dir_path, filename), 'r') as f:
        return pd.json_normalize(json.load(f), 'data', ['name']).rename(columns={0: 'year', 1: 'val'})

def load_json_data(json_dir_path, filename):
    if not os.path.exists(os.path.join(json_dir_path)):
        return return_empty_df(json_dir_path, filename)
    elif filename.endswith('.zip'):
        return return_zipped_df(json_dir_path, filename)
    else:
        return return_plain_df(json_dir_path, filename)



def process_area_all_lu(json_dir_path):
    return load_json_data(json_dir_path, 'area_1_total_area_wide.json')

def process_area_non_ag_lu(json_dir_path):
    return load_json_data(json_dir_path, 'area_3_non_ag_lu_area_wide.json')

def process_area_ag_man(json_dir_path):
    return load_json_data(json_dir_path, 'area_4_am_total_area_wide.json')

def process_profit_data(json_dir_path):
    df = load_json_data(json_dir_path, 'economics_0_rev_cost_all_wide.json')
    return df.query('name == "Profit"')

def process_production_quantity_data(json_dir_path):
    return load_json_data(json_dir_path, 'production_5_6_demand_Production_commodity_from_LUTO.json')

def process_production_deviation_data(json_dir_path):
    df_demand = load_json_data(json_dir_path, 'production_5_6_demand_Production_commodity_from_LUTO.json')
    df_luto = load_json_data(json_dir_path, 'production_5_5_demand_Production_commodity.json')
    df_delta = df_demand.merge(df_luto, on=['year', 'name'], suffixes=('_luto', '_demand'))
    # df_delta['deviation_t'] = df_delta.eval('val_luto - val_demand')
    df_delta['val'] = df_delta.eval('(val_luto - val_demand) / val_demand * 100')
    return df_delta

def process_GHG_deviation_data(json_dir_path):
    df = load_json_data(json_dir_path, 'GHG_2_individual_emission_Mt.json')
    df_target = df.query('name == "GHG emissions limit"')
    df_actual = df.query('name == "Net emissions"')
    df_deviation = df_target.merge(df_actual, on='year', suffixes=('_target', '_actual'))
    df_deviation['name'] = 'GHG deviation'
    df_deviation['val'] = df_deviation['val_actual'] - df_deviation['val_target']
    return df_deviation

def process_bio_obj_data(json_dir_path):
    return load_json_data(json_dir_path, 'biodiversity_priority_1_total_score_by_type.json')



def get_report_df(json_dir_path, run_paras):
    
    df_area_all_lu = process_area_all_lu(json_dir_path)
    df_area_non_ag_lu = process_area_non_ag_lu(json_dir_path)
    df_area_ag_man = process_area_ag_man(json_dir_path)
    df_profit = process_profit_data(json_dir_path)
    df_ghg_deviation = process_GHG_deviation_data(json_dir_path)
    df_demand_deviation = process_production_deviation_data(json_dir_path)
    df_bio_objective = process_bio_obj_data(json_dir_path)

    report_df = pd.concat([
        df_area_all_lu[['year', 'name', 'val']].assign(Type='Area_all_lu_million_km2'),
        df_area_non_ag_lu[['year', 'name', 'val']].assign(Type='Area_non_ag_lu_million_km2'),
        df_area_ag_man[['year', 'name', 'val']].assign(Type='Area_ag_man_million_km2'),
        df_profit[['year', 'name', 'val']].assign(Type='Profit_billion_AUD'),
        df_demand_deviation[['year', 'name', 'val']].assign(Type='Production_deviation_pct'),
        df_ghg_deviation[['year', 'name', 'val']].assign(Type='GHG_Deviation_pct'),
        df_bio_objective[['year', 'name', 'val']].assign(Type='Biodiversity_obj_score'),
    ]).assign(**run_paras).reset_index(drop=True)

    report_df['GHG_LIMITS_FIELD'] = report_df['GHG_LIMITS_FIELD'].replace(GHG_ORDER)
    report_df['GBF2_TARGET_DICT'] = report_df['GBF2_TARGET_DICT'].replace(BIO_TARGET_ORDER)

    return report_df



def process_task_root_dirs(task_root_dir, n_workers=10):
    
    grid_search_params = pd.read_csv(f"{task_root_dir}/grid_search_parameters.csv")
    run_dirs = [i for i in os.listdir(task_root_dir) if os.path.isdir(os.path.join(task_root_dir, i))]
    run_dirs = [i for i in run_dirs if 'Run_' in i]
    
    tasks = []
    for dir in run_dirs:
        run_idx = int(dir.split('_')[-1])
        run_paras = grid_search_params.query(f'run_idx == {int(run_idx)}').to_dict(orient='records')[0]

        # First try to find the JSON data in tht zip file
        json_dir_path = os.path.join(task_root_dir, dir, 'DATA_REPORT.zip')
        
        # Then try to find the JSON data in the unzipped `output` folder
        output_dir = os.path.join(task_root_dir, dir, 'output')
        if (not os.path.exists(json_dir_path)) and os.path.exists(output_dir):
            last_dir = sorted([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])[-1]
            json_dir_path = os.path.join(output_dir, last_dir, 'DATA_REPORT', 'data')
        else:
            print(f'{dir}: No outputs found in Run_{run_idx}!')
            continue
        
        # If the JSON data path still does not exist, skip this run
        if not os.path.exists(json_dir_path):
            print(f'{dir}: DATA_REPORT not found in Run_{run_idx}!')
            continue
        
        # Json to df
        tasks.append(delayed(get_report_df)(json_dir_path, run_paras))
        
    # Concatenate the results, only keep the columns with more than 1 unique value
    out_df = pd.concat(tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)), ignore_index=True)
    out_df =  out_df.loc[:, out_df.nunique() > 1]
    
    return out_df

        
