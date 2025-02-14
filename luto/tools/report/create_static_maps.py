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

# Append the LUTO directory to the system path, 
# so that the modules can be imported as individual process and parallel processed.
import os
import sys
LUTO_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(LUTO_dir)


import luto.settings as settings
from joblib import Parallel, delayed

from luto.tools.report.data_tools import  get_all_files
from luto.tools.report.data_tools.parameters import RENAME_AM_NON_AG
from luto.tools.report.map_tools import process_raster, save_map_to_html
from luto.tools.report.map_tools.map_making import create_png_map
from luto.tools.report.map_tools.helper import (get_map_meta, 
                                                get_map_fullname,
                                                get_scenario)




def TIF2MAP(raw_data_dir:str):

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir)
    # Get the initial tif files
    tif_files = files.query('base_ext == ".tiff" and year_types != "begin_end_year"')

    # Get the metadata for map making
    map_meta = get_map_meta()
    model_run_scenario = get_scenario(raw_data_dir)
    
    # Merge the metadata with the tif files
    tif_files_with_meta = tif_files.merge(map_meta, 
                                        left_on='category', 
                                        right_on='map_type')
    
    # Loop through the tif files and create the maps (PNG and HTML)
    tasks = (delayed(create_maps)(row, model_run_scenario) for _, row in tif_files_with_meta.iterrows())
    for out in Parallel(n_jobs=settings.THREADS, return_as='generator')(tasks):
        print(out)
    
        
     
def create_maps(row, model_run_scenario):
    """
    Creates a static map based on the given row data and model run scenario.

    Args:
        row (pandas.Series): A row of data containing information about the map.
        model_run_scenario (str): The model run scenario text (or path to).

    Returns:
        None
    """
    
    # Get the necessary variables
    tif_path = row['path']
    color_csv = row['color_csv']
    data_type = row['data_type']
    year = row['Year']
    legend_params = row['legend_params']
    
    # Process the raster, and get the necessary variables
    (center,                # center of the map (lat, lon)
    bounds_for_folium,      # bounds for folium (lat, lon)
    mercator_bbox,          # bounds for download base map (west, south, east, north <meters>)
    color_desc_dict         # color description dictionary ((R,G,B,A) <0-255>: description <str>)
    ) = process_raster(tif_path, color_csv, data_type)
    
    # Update the lucc description in the color_desc_dict with the RENAME_AM_NON_AG
    color_desc_dict = {k:RENAME_AM_NON_AG.get(v,v) for k, v in color_desc_dict.items()}

    # Create the annotation text for the map
    map_fullname = get_map_fullname(tif_path)
    inmap_text = f'''{map_fullname}\nScenario: {model_run_scenario}\nYear: {year}'''

    # Mosaic the projected_tif with base map, and overlay the shapefile
    create_png_map( tif_path = tif_path,
                    data_type = data_type,
                    color_desc_dict = color_desc_dict,
                    anno_text = inmap_text,
                    mercator_bbox = mercator_bbox,
                    legend_params = legend_params)
    
    
    # Save the map to HTML
    save_map_to_html(tif_path,
                     'luto/tools/report/Assets/AUS_adm/STE11aAust_mercator_simplified.shp', 
                     data_type,
                     center, 
                     bounds_for_folium,
                     color_desc_dict)
    
    
    return f'{map_fullname} of year {year} has been created.'
    
    