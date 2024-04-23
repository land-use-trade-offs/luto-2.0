    

import os
import re
import keyword
from matplotlib.pylab import f
import pandas as pd
import shutil
from luto.settings import WRITE_THREADS
from joblib import delayed, Parallel

import luto.settings as settings
from luto.tools.create_task_runs.parameters import EXCLUDE_DIRS, PARAMS_TO_EVAL, TASK_ROOT_DIR

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
 
 
 
 
def is_valid_variable_name(name):
    if name in keyword.kwlist:
        print(f"{name} is a keyword")
        return False
    if not name:  # Check if the name is not empty
        print("Column name is empty")
        return False
    if not name[0].isalpha() and name[0] != '_':  # Must start with a letter or underscore
        print(f"{name} must start with a letter or underscore")
        return False
    if all((char.isalnum() or char == '_') for char in name):
        return True
    print(f"{name} must contain only letters, numbers, or underscores")
    return False





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






def create_settings_template(to_path:str=TASK_ROOT_DIR):
    
    if os.path.exists(f'{to_path}/settings_template.csv'):
        print('settings_template.csv already exists! Skip creating a new one!')
        return

    # Get the settings from luto.settings
    with open('luto/settings.py', 'r') as file:
        lines = file.readlines()

        # Regex patterns that matches variable assignments from settings
        parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=")
        settings_order = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]

        # Reorder the settings dictionary to match the order in the settings.py file
        settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
        settings_dict = {i: settings_dict[i] for i in settings_order if i in settings_dict}


    # Create a template for cutom settings
    settings_df = pd.DataFrame({k:[v] for k,v in settings_dict.items()}).T.reset_index()
    settings_df.columns = ['Name','Default_run']
    settings_df = settings_df.map(str)


    # Create a folder for the custom settings
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    settings_df.to_csv(f'{to_path}/settings_template.csv', index=False)
    
    
    
    
    
    
    
def create_task_folders(from_path:str=f'{TASK_ROOT_DIR}/settings_template.csv'):
    
    # Read the custom settings file
    custom_settings = pd.read_csv(from_path, index_col=0)
    custom_settings = custom_settings.replace({'TRUE': 'True', 'FALSE': 'False'})
    custom_settings.loc[PARAMS_TO_EVAL] = custom_settings.loc[PARAMS_TO_EVAL].map(eval)

    # Create a dictionary of custom settings
    reserve_cols = ['Default_run']
    custom_cols = [col for col in custom_settings.columns if col not in reserve_cols]
    
    
    for col in custom_cols:
        # Check if the column name is valid
        if not is_valid_variable_name(col):  
            raise ValueError(f'"{col}" is not a valid column name!')
    
        # Report the changed settings
        changed_params = 0
        for idx,_ in custom_settings.iterrows():
            if custom_settings.loc[idx,col] != custom_settings.loc[idx,'Default_run']:
                changed_params = changed_params + 1
                print(f'"{col}" has changed <{idx}>: "{custom_settings.loc[idx,"Default_run"]}" ==> "{custom_settings.loc[idx,col]}">')
        
        print(f'"{col}" has no changed parameters compared to "Default_run"') if changed_params == 0 else None
    
    
        # Copy the custom settings to the custom runs folder
        s_d = copy_folder_custom(os.getcwd(), f'{TASK_ROOT_DIR}/{col}', EXCLUDE_DIRS)
        worker = min(WRITE_THREADS, len(s_d))
        Parallel(n_jobs=worker)(delayed(shutil.copy2)(s, d) for s, d in s_d)
        
        # Create an output folder for the task
        os.makedirs(f'{TASK_ROOT_DIR}/{col}/output', exist_ok=True)
            
    
    
        # Write the custom settings to each task folder
        custom_dict = custom_settings[col].to_dict()
        
        # The input dir for each task will point to the absolute path of the input dir
        custom_dict['INPUT_DIR'] = os.path.abspath(custom_dict['INPUT_DIR']).replace('\\','/')
        custom_dict['DATA_DIR'] = custom_dict['INPUT_DIR']

        # Write the custom settings to the settings.py of each task
        with open(f'{TASK_ROOT_DIR}/{col}/luto/settings.py', 'w') as file:
            for k, v in custom_dict.items():
                if str(v).isdigit() or is_float(str(v)):
                    file.write(f'{k} = {v}\n')
                elif isinstance(v, str):
                    file.write(f'{k} = "{v}"\n')
                else:
                    file.write(f'{k} = {v}\n')












