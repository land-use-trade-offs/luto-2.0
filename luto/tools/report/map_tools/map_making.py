import os
import rasterio
from rasterio.plot import show
from rasterio.merge import merge

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
from luto.tools.report.map_tools.helper import download_basemap
from rasterio.coords import BoundingBox





def create_png_map(tif_path: str, 
                   map_note:str,
                   color_desc_dict: dict,
                   basemap_path: str = 'luto/tools/report/Assests/basemap.tif', 
                   shapefile_path: str ='luto/tools/report/Assests/AUS_adm/STE11aAust_mercator_simplified.shp',
                   anno_text: str = None,
                   mercator_bbox: BoundingBox = None):
    
    """
    Creates a PNG map by overlaying a raster image with a basemap, shapefile, annotation, scale bar, north arrow, and legend.

    Parameters:
    - tif_path (str): 
        The path to the input raster image.
    - map_note (str):
        The note for the map. Can be used to identify the color scheme used for the map.
    - color_desc_dict (dict): 
        A dictionary mapping color values to their descriptions for the legend.
    - basemap_path (str): 
        The path to the basemap image. Default is 'Assests/basemap.tif'.
    - shapefile_path (str): 
        The path to the shapefile for overlaying. Default is 'Assests\AUS_adm\STE11aAust_mercator_simplified.shp'.
    - anno_text (str): 
        The annotation text to be displayed on the map. Default is None.
    - mercator_bbox (BoundingBox): 
        The bounding box of the mercator projection. Default is None.

    Returns:
    - None
    """

    
    # Download basemap if it does not exist
    if not os.path.exists(basemap_path):
        if mercator_bbox is None:
            raise ValueError("The bounding box in Mercator projection is required to download the basemap.")
        print("Downloading basemap...")
        print('This Could take a while ...')
        print('Only download once ...')
        download_basemap(mercator_bbox)
    
    
    # Get the mercator input image
    out_base = os.path.splitext(tif_path)[0]
    if map_note is not None:
        in_mercator_path = f"{out_base}_mercator_{map_note}.tif"
        png_out_path = f"{out_base}_mosaic_{map_note}.png"
    else:
        in_mercator_path = f"{out_base}_mercator.tif"
        png_out_path = f"{out_base}_mosaic.png"
    
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(20, 20)) 

    # Mosaic the raster with the basemap
    with rasterio.open(in_mercator_path) as src, rasterio.open(basemap_path) as base:
        # Mosaic the raster with the basemap
        mosaic, out_transform = merge([src, base])
    

    # Display the mosaic raster
    show(mosaic, ax=ax, transform=out_transform)
    
    # Add annotation
    plt.annotate(anno_text, 
         xy=(0.07, 0.9), 
         xycoords='axes fraction',
         fontsize=25,
        #  fontweight = 'bold',
         ha='left', 
         va='center')
    

    # Overlay the shapefile
    # Load the shapefile with GeoPandas
    gdf = gpd.read_file(shapefile_path)
    gdf.boundary.plot(ax=ax, 
              color='grey', 
              linewidth=0.5, 
              edgecolor='grey', 
              facecolor='none')

    # # Create scale bar
    # ax.add_artist(ScaleBar(1, 
    #            "m", 
    #            location="lower right",
    #            border_pad=1,
    #            fixed_units="km",
    #            fixed_value=500,
    #            box_color="skyblue", 
    #            box_alpha=0))

    # # Create north arrow
    # x, y, arrow_length = 0.9, 0.9, 0.07
    # ax.annotate('N', 
    #     xy=(x, y), 
    #     xytext=(x, y-arrow_length),
    #     arrowprops=dict(facecolor='#5f5f5e', 
    #                     edgecolor='#5f5f5e', 
    #                     width=30, 
    #                     headwidth=45),
    #     ha='center', 
    #     va='center', 
    #     fontsize=25,
    #     color='#2f2f2f',
    #     xycoords=ax.transAxes)

    # Create legend
    patches = [mpatches.Patch(color=tuple(value / 255 for value in k), label=v) 
           for k, v in color_desc_dict.items()]

    plt.legend(handles=patches, 
           bbox_to_anchor=(0.09, 0.2), 
           loc=2, 
           borderaxespad=0.,
           ncol=2, 
           fontsize=20,
           framealpha=0)

    # Optionally remove axis
    ax.set_axis_off()
    plt.savefig(png_out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Delete the input raster
    os.remove(in_mercator_path)