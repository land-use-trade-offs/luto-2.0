import os
import shutil
import pandas as pd
from glob import glob
import argparse

from tools.helper_func import add_data_2_html


# # setting up working directory to root dir
# if  __name__ == '__main__':
#     os.chdir('../../..')

####################################################
#         setting up working variables             #
####################################################



# Get the output directory
parser = argparse.ArgumentParser()
parser.add_argument("-p", type=str, required=True, help="Output directory path")
args = parser.parse_args()

RAW_DATA_ROOT = args.p
RAW_DATA_ROOT = os.path.abspath(RAW_DATA_ROOT)
RAW_DATA_ROOT = os.path.normpath(RAW_DATA_ROOT).replace("\\", "/")

# Set the save directory    
REPORT_DIR = f'{RAW_DATA_ROOT}/DATA_REPORT'
if not os.path.exists(REPORT_DIR):
    raise Exception(f"Report directory not found: {REPORT_DIR}")
    


####################################################
#        Copy report html to the REPORT_DIR        #
####################################################  

shutil.copytree('luto/tools/report/tools/template_html', 
                f'{REPORT_DIR}/REPORT_HTML',
                dirs_exist_ok=True)


####################################################
#                Write data to HTML                #
#################################################### 

import os
import pandas as pd

from lxml import etree
from  lxml.etree import Element

# Get the tree of the index page
index_tree = etree.parse(f"{REPORT_DIR}/REPORT_HTML/index.html", 
                         etree.HTMLParser())

# Get the txt of the running settings
with open(f"{RAW_DATA_ROOT}/model_run_settings.txt",'r') as f:
    settings_txt = f.read()
    
# Replace the inner text of the <pre id="settingsTxt"></pre> with 
# the settings txt and make it to be invisible
settings_pre = index_tree.xpath('//pre[@id="settingsTxt"]')[0]
settings_pre.text = settings_txt
settings_pre.attrib['style'] = 'display:none'
  
    
# Write the index page
index_tree.write(f"{REPORT_DIR}/REPORT_HTML/index.html", 
                 pretty_print=True,
                 encoding='utf-8',
                 method='html')
    

    
    

 

# Get all html files needs data insertion
html_df = pd.DataFrame([['production',f"{REPORT_DIR}/REPORT_HTML/pages/production.html"],
                        ['economics',f"{REPORT_DIR}/REPORT_HTML/pages/economics.html"],
                        ["area",f"{REPORT_DIR}/REPORT_HTML/pages/land-use_area.html"],
                        ["GHG",f"{REPORT_DIR}/REPORT_HTML/pages/GHG_emissions.html"],
                        ["water",f"{REPORT_DIR}/REPORT_HTML/pages/water_usage.html"]])

html_df.columns = ['name','path']

# Get all data files
all_data_files = glob(f"{REPORT_DIR}/data/*")

# Add data path to html_df
html_df['data_path'] = html_df.apply(lambda x: [i for i in all_data_files if x['name'] in i ],axis=1)


# Parse html files
for idx,row in html_df.iterrows():
    
    html_path = row['path']
    data_pathes  = row['data_path']

    # Add data to html
    add_data_2_html(html_path,data_pathes)
    
    
#########################################################
#              Report success info                      #
#########################################################

print('\n Report html created successfully!! \n')