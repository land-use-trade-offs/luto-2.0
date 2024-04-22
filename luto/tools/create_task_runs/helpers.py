    

import os
import re
import keyword
import pandas as pd
import luto.settings as settings

from luto.tools.create_task_runs.parameters import PARAMS_TO_EVAL

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



def create_settings_template(to_path:str='../Custom_runs'):
    
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
    
    
    
    
def create_custom_settings(from_path:str='../Custom_runs/settings_template.csv'):
    # Read the custom settings file
    custom_settings = pd.read_csv(from_path, index_col=0)
    custom_settings = custom_settings.replace({'TRUE': 'True', 'FALSE': 'False'})
    custom_settings.loc[PARAMS_TO_EVAL] = custom_settings.loc[PARAMS_TO_EVAL].map(eval)

    # Create a dictionary of custom settings
    reserve_cols = ['Default_run']
    custom_cols = [col for col in custom_settings.columns if col not in reserve_cols]


    # Check if the column name is valid
    for col in custom_cols:
        if not is_valid_variable_name(col):  
            raise ValueError(f'"{col}" is not a valid column name!')

    # Report the changed settings
    for idx,row in custom_settings.iterrows():
        for col in custom_cols:
            if custom_settings.loc[idx,col] != custom_settings.loc[idx,'Default_run']:
                print(f'"{col}" has changed <{idx}>: "{custom_settings.loc[idx,"Default_run"]}" ==> "{custom_settings.loc[idx,col]}>')


    # Write the custom settings to different settings files
    custom_dict = custom_settings[custom_cols].to_dict()

    for para_d in custom_dict:
        with open(f'../Custom_runs/settings_{para_d}.py', 'w') as file:
            for k, v in custom_dict[para_d].items():
                if str(v).isdigit() or is_float(str(v)):
                    file.write(f'{k} = {v}\n')
                elif isinstance(v, str):
                    file.write(f'{k} = "{v}"\n')
                else:
                    file.write(f'{k} = {v}\n')
