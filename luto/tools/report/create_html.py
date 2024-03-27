import os
import shutil
import pandas as pd
from glob import glob

from luto.tools.report.data_tools.helper_func import (add_data_2_html, 
                                                      add_settings_2_html)



####################################################
#         setting up working variables             #
####################################################

def data2html(sim):

    # Get the raw data directory
    raw_data_dir = sim.path

    # Set the save directory    
    report_dir = f'{raw_data_dir}/DATA_REPORT'
    
    # Check if the report directory exists
    if not os.path.exists(report_dir):
        raise Exception(f"Report directory not found: {report_dir}")
        
        
    ####################################################
    #        Copy report html to the report_dir        #
    ####################################################  

    # Copy the html template to the report directory
    shutil.copytree('luto/tools/report/data_tools/template_html', 
                    f'{report_dir}/REPORT_HTML',
                    dirs_exist_ok=True)


    ####################################################
    #                Write data to HTML                #
    #################################################### 

    # Add settings to the home page
    add_settings_2_html(report_dir, raw_data_dir)

    # Get all html files needs data insertion
    html_df = pd.DataFrame([['production',f"{report_dir}/REPORT_HTML/pages/production.html"],
                            ['economics',f"{report_dir}/REPORT_HTML/pages/economics.html"],
                            ["area",f"{report_dir}/REPORT_HTML/pages/land-use_area.html"],
                            ["GHG",f"{report_dir}/REPORT_HTML/pages/GHG_emissions.html"],
                            ["water",f"{report_dir}/REPORT_HTML/pages/water_usage.html"],
                            ['biodiversity',f"{report_dir}/REPORT_HTML/pages/biodiversity.html"],])

    html_df.columns = ['name','path']

    # Get all data files
    all_data_files = glob(f"{report_dir}/data/*")

    # Add data path to html_df
    html_df['data_path'] = html_df.apply(lambda x: [i for i in all_data_files if x['name'] in i ], axis=1)


    # Parse html files
    for idx,row in html_df.iterrows():
        
        html_path = row['path']
        data_pathes  = row['data_path']

        # Add data to html
        add_data_2_html(html_path,data_pathes)
        
        
    #########################################################
    #              Report success info                      #
    #########################################################

    print('Report html created successfully!\n')