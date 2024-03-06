import argparse
import os
import pandas as pd
import folium
import rasterio
from rasterio.merge import merge

from luto.tools.report.map_tools import process_raster
from luto.tools.report.map_tools.map_making import create_png_map
from luto.tools.report.map_tools.helper import (get_map_meta, 
                                                get_map_fullname,
                                                get_scenario)


from luto.tools.report.data_tools import  get_all_files




def TIF2PNG(sim):

    # Get the output directory
    raw_data_dir = sim.path

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir)

    # Get the initial tif files
    tif_files = files.query('base_ext == ".tiff" and year_types != "begin_end_year"')



    ###################################################################
    #           Get reclassification and color dictionary             #
    ###################################################################

    # Get the metadata for map making
    map_meta = get_map_meta()
    model_run_scenario = get_scenario(raw_data_dir)

    # Merge the metadata with the tif files
    tif_files_with_meta = tif_files.merge(map_meta, 
                                        left_on='category', 
                                        right_on='map_type')


    ###################################################################
    #           Get reclassification and color dictionary             #
    ###################################################################


    for _, row in tif_files_with_meta.iterrows():
        
        tif_path = row['path']
        color_csv = row['color_csv']
        data_type = row['data_type']
        map_num = row['map_num']
        
        year = row['year']
        map_fullname = get_map_fullname(tif_path)
        
        
        # Process the raster, and get the necessary variables
        (center, # center of the map
        bounds_for_folium, # bounds for folium
        mercator_bbox, # bounds for download base map
        color_desc_dict # color description dictionary
        ) = process_raster(tif_path, color_csv, data_type, map_num)



        # Create the annotation text for the map
        inmap_text = f'''{map_fullname}\nScenario: {model_run_scenario}\nYear: {year}'''


        create_png_map(tif_path = tif_path,
                        color_desc_dict = color_desc_dict,
                        anno_text = inmap_text,
                        mercator_bbox = mercator_bbox,
                        map_num = map_num)




    ###################################################################
    #       Merge processed map with basemap, create png map          #
    ###################################################################

        




    # m = folium.Map(center, zoom_start=3,zoom_control=False)

    # img = folium.raster_layers.ImageOverlay(
    #         name="Mercator projection SW",
    #         image=out_png,
    #         bounds=bounds,
    #         opacity=0.6,
    #         interactive=True,
    #         cross_origin=False,
    #         zindex=1,
    #     )

    # img.add_to(m)
    # m









