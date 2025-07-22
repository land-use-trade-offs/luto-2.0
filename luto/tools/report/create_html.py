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

import os
import shutil
import pandas as pd
from glob import glob

from luto import settings
from luto.tools.report.data_tools.helper_func import add_data_2_html, add_txt_2_html
from luto.tools.report.data_tools.parameters import COLORS, SPATIAL_MAP_DICT, RENAME_AM_NON_AG



####################################################
#         setting up working variables             #
####################################################

def data2html(raw_data_dir):
    

    # Set the save directory    
    report_dir = f'{raw_data_dir}/DATA_REPORT'

    # Copy the html template to the report directory
    shutil.copytree(
        'luto/tools/report/data_tools/template_html', 
        f'{report_dir}/REPORT_HTML',
        dirs_exist_ok=True
    )
    
    # Get all html files needs data insertion
    html_df = pd.DataFrame([
        ['Production',f"{report_dir}/REPORT_HTML/pages/production.html"],
        ['Economics',f"{report_dir}/REPORT_HTML/pages/economics.html"],
        ["Area",f"{report_dir}/REPORT_HTML/pages/land-use_area.html"],
        ["GHG",f"{report_dir}/REPORT_HTML/pages/GHG_emissions.html"],
        ["Water",f"{report_dir}/REPORT_HTML/pages/water_usage.html"],
        ['BIO',f"{report_dir}/REPORT_HTML/pages/biodiversity.html"],
        ['Index',f"{report_dir}/REPORT_HTML/index.html"]
    ], columns=['name','HTML_path'])

    # Get all data files
    all_data_files = glob(f"{report_dir}/data/*")
    # Add data path to html_df
    html_df['data_path'] = html_df.apply(lambda x: [i for i in all_data_files if os.path.basename(i).startswith(x['name'])], axis=1)
    # Add the supporting info file to each html
    html_df['data_path'] = html_df['data_path'].apply(lambda x: x + [f"{report_dir}/data/Supporting_info.json"])

    # Parse each HTML file and add data
    for _,row in html_df.iterrows():
        add_data_2_html(row['HTML_path'], row['data_path'])
        

        
    #########################################################
    #              Report success info                      #
    #########################################################

    print('Report html created successfully!\n')
    
    