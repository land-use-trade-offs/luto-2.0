
import pandas as pd
import contextily as ctx
import matplotlib as mpl
from rasterio.coords import BoundingBox

from luto.tools.report.map_tools.parameters import (color_types,
                                                    map_note,
                                                    data_types,
                                                    legend_positions,
                                                    map_basename_rename)


# Function to download a basemap image
def download_basemap(bounds_mercator: list[str]):
    """
    Downloads a basemap image within the specified bounds in Mercator projection.

    Args:
        bounds_mercator (BoundingBox): The bounding box in Mercator projection. Defaults to None.

    Returns:
        tuple: A tuple containing the downloaded basemap image and its extent.
    """

    base_map, extent = ctx.bounds2raster(*bounds_mercator, 
                                        path='luto/ools/report/Assets/basemap.tif',
                                        source=ctx.providers.OpenStreetMap.Mapnik,
                                        zoom=7,
                                        n_connections=16,
                                        max_retries=4)
    return base_map, extent

   
# Function to create value-color dictionary for intergirized raster (1-100) 
def create_color_csv_0_100(color_scheme:str='YlOrRd',
                           save_path:str='luto/tools/report/Assets/float_img_colors.csv',
                           extra_color:dict={   0:(130, 130, 130, 255),
                                             -100:(225, 225, 225, 255)}):
    """
    Create a CSV file contains the value(1-100)-color(HEX) records.

    Parameters:
    - color_scheme (str): 
        The name of the color scheme to use. Default is 'YlOrRd'.
    - save_path (str): 
        The file path to save the color dictionary as a CSV file. Default is 'Assets/float_img_colors.csv'.
    - extra_color (dict): 
        Additional colors to include in the dictionary. Default is {-100:(225, 225, 225, 255)}.

    Returns:
        None
    """
    colors = mpl.colormaps[color_scheme]
    val_colors_dict = {i: colors(i/100) for i in range(1,101)}
    var_colors_dict = {k:tuple(int(num*255) for num in v) for k,v in val_colors_dict.items()}
    
    
    # If extra colors are specified, add them to the dictionary
    if extra_color:
        var_colors_dict.update(extra_color) 
    
    # Convert the RGBA values to HEX color codes
    var_colors_dict = {k: f"#{''.join(f'{c:02X}' for c in v)}" 
                       for k, v in var_colors_dict.items()}
    
    # Save the color dictionary to a CSV file
    color_df = pd.DataFrame(var_colors_dict.items(), columns=['lu_code', 'lu_color_HEX'])
    color_df.to_csv(save_path, index=False)
    
    
def get_map_meta():
    """
    Get the map metadata.

    Returns:
        map_meta (DataFrame): DataFrame containing map metadata with columns 'map_type', 'csv_path', 'legend_type', and 'legend_position'.
    """
    
    # Create a DataFrame from the color_types dictionary
    map_meta = pd.DataFrame(color_types.items(), 
                            columns=['map_type', 'color_csv'])
 
    # Add other metadata columns to the DataFrame
    map_meta['map_note'] = map_meta['map_type'].map(map_note)
    map_meta['data_type'] = map_meta['map_type'].map(data_types)
    map_meta['legend_position'] = map_meta['map_type'].map(legend_positions)
    
    # Explode the csv_path and map_note columns
    map_meta = map_meta.explode(['color_csv','map_note'])
    
    return map_meta.reset_index(drop=True)


def get_map_fullname(path:str):
    """
    Get the full name of a map based on its path.

    Args:
        path (str): The path of the map.

    Returns:
        str: The full name of the map.
    """
    for k,v in map_basename_rename.items():
        if k in path:
            return v


def get_scenario(data_root_dir:str):
    """
    Get the scenario name from the data root directory.

    Args:
        data_root_dir (str): The data root directory.

    Returns:
        str: The scenario name.
    """
    with open(f'{data_root_dir}/model_run_settings.txt', 'r') as f:
        for line in f:
            if 'GHG_LIMITS_FIELD' in line:
                return line.split(':')[-1].strip()

