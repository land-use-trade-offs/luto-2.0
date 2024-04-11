from joblib import Parallel, delayed
import luto.settings as settings
from luto.data import Data

from luto.tools.report.data_tools import  get_all_files

from luto.tools.report.map_tools import process_raster, save_map_to_html
from luto.tools.report.map_tools.map_making import create_png_map
from luto.tools.report.map_tools.parameters import lucc_rename
from luto.tools.report.map_tools.helper import (get_map_meta, 
                                                get_map_fullname,
                                                get_scenario)




def TIF2MAP(data: Data):

    # Get the output directory
    raw_data_dir = data.path
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
    Parallel(n_jobs=settings.THREADS)(delayed(create_maps)(row, model_run_scenario) 
                                      for _, row in tif_files_with_meta.iterrows())
        
     
def create_maps(row, model_run_scenario):
    
    # Get the necessary variables
    tif_path = row['path']
    color_csv = row['color_csv']
    data_type = row['data_type']
    year = row['year']
    legend_params = row['legend_params']
    
    print(f'Making map for {row["base_name"]} in {year}...')
    
    # Process the raster, and get the necessary variables
    (center,                # center of the map (lat, lon)
    bounds_for_folium,      # bounds for folium (lat, lon)
    mercator_bbox,          # bounds for download base map (west, south, east, north <meters>)
    color_desc_dict         # color description dictionary ((R,G,B,A) <0-255>: description <str>)
    ) = process_raster(tif_path, color_csv, data_type)
    
    # Update the lucc description in the color_desc_dict with the lucc_rename
    color_desc_dict = {k:lucc_rename.get(v,v) for k, v in color_desc_dict.items()}

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
    
    









