
import os
import re
import shutil
import keyword
import pandas as pd

from joblib import delayed, Parallel

from luto.tools.create_task_runs.parameters import EXCLUDE_DIRS, PARAMS_TO_EVAL, TASK_ROOT_DIR
from luto import settings




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



def get_requirements():
    with open('requirements.txt', 'r') as file:
        requirements = file.read().splitlines()
        
    split_idx = requirements.index('# Below can only be installed with pip')
    
    conda_pkgs = " ".join(requirements[:split_idx])
    pip_pkgs = " ".join(requirements[split_idx+1:])
    
    return conda_pkgs, pip_pkgs
    
        
    


def create_settings_template(to_path:str=TASK_ROOT_DIR):


    # Save the settings template to the root task folder
    None if os.path.exists(to_path) else os.makedirs(to_path)
    
    conda_pkgs, pip_pkgs = get_requirements()
    with open(f'{to_path}/requirements_conda.txt', 'w') as conda_f, \
            open(f'{to_path}/requirements_pip.txt', 'w') as pip_f:
        conda_f.write(conda_pkgs)
        pip_f.write(pip_pkgs)
    
        
    if os.path.exists(f'{to_path}/settings_template.csv'):
        print('settings_template.csv already exists! Skip creating a new one!')
    else:
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
        settings_df.to_csv(f'{to_path}/settings_template.csv', index=False)
    
    
    
    
    
    
    
def create_task_runs(from_path:str=f'{TASK_ROOT_DIR}/settings_template.csv'):
    
    # Read the custom settings file
    custom_settings = pd.read_csv(from_path, index_col=0)
    custom_settings = custom_settings.replace({'TRUE': 'True', 'FALSE': 'False'})
    custom_settings.loc[PARAMS_TO_EVAL] = custom_settings.loc[PARAMS_TO_EVAL].map(eval)

    # Create a dictionary of custom settings
    reserve_cols = ['Default_run']
    custom_cols = [col for col in custom_settings.columns if col not in reserve_cols]

    if not custom_cols:
        raise ValueError('No custom settings found in the settings_template.csv file!')

    original_dir = os.getcwd()
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
        worker = min(settings.WRITE_THREADS, len(s_d))
        Parallel(n_jobs=worker)(delayed(shutil.copy2)(s, d) for s, d in s_d)

        # Create an output folder for the task
        os.makedirs(f'{TASK_ROOT_DIR}/{col}/output', exist_ok=True)



        # Write the custom settings to each task folder
        custom_dict = custom_settings[col].to_dict()

        # The input dir for each task will point to the absolute path of the input dir
        custom_dict['INPUT_DIR'] = os.path.abspath(custom_dict['INPUT_DIR']).replace('\\','/')
        custom_dict['DATA_DIR'] = custom_dict['INPUT_DIR']

        # Write the custom settings to the settings.py of each task
        with open(f'{TASK_ROOT_DIR}/{col}/luto/settings.py', 'w') as file, \
             open(f'{TASK_ROOT_DIR}/{col}/luto/settings_bash.py', 'w') as bash_file:
            for k, v in custom_dict.items():
                # List values need to be converted to bash arrays
                if isinstance(v, list):
                    bash_file.write(f'{k}=({ " ".join([str(elem) for elem in v])})\n')
                    file.write(f'{k}={v}\n')
                # Dict values need to be converted to bash variables
                elif isinstance(v, dict):
                    bash_file.write(f'# {k} is a dictionary, which is not natively supported in bash\n')
                    for key, value in v.items():
                        bash_file.write(f'{k}_{key}={value}\n')
                # Strings need to be enclosed in quotes
                elif isinstance(v, str):
                    file.write(f'{k}="{v}"\n')
                    bash_file.write(f'{k}="{v}"\n')         
                # The rest can be written as it is
                else:
                    file.write(f'{k}={v}\n')
                    bash_file.write(f'{k}={v}\n')
    

        # Copy the slurm script to the task folder
        shutil.copyfile('luto/tools/create_task_runs/bash_scripts/bash_cmd.sh', f'{TASK_ROOT_DIR}/{col}/slurm.sh')


        # Start the task if the os is linux
        if os.name == 'posix':
            os.chdir(f'{TASK_ROOT_DIR}/{col}')
            os.system('sbatch -p mem slurm.sh')
            os.chdir(original_dir)












