
import re
import numpy as np
import nbformat as nbf
import xarray as xr
import rioxarray as rxr
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from itertools import product

from luto.data import Data
from luto.tools.Manual_jupyter_books.helpers.parameters import NOTEBOOK_META_DICT
from luto.tools.spatializers import get_coarse2D_map



def full_res_1d_raw_to_2d(data:Data, arr:np.ndarray) -> np.ndarray:
    '''
    This function converts a 1D numpy array to an 2D array with the same shape as `NLUM_MASK`.
    
    Inputs
    ------
    data: Data
        The Data object that contains the metadata of the 2D array.
    arr: np.ndarray
        The 1D array that will be converted to an 2D array.
    
    Returns
    -------
        The 2D array that has the same shape as `NLUM_MASK`.
    '''
    LUMAP_FullRes_2D = np.full(data.NLUM_MASK.shape, data.NODATA).astype(np.float32) 
    np.place(LUMAP_FullRes_2D, data.NLUM_MASK, arr)
    return LUMAP_FullRes_2D
    



def arr_to_xr(data:Data, arr:np.ndarray) -> xr.DataArray:
    '''
    This function converts a 1D numpy array to an 2D xarray DataArray with `transform` and `lon/lat`.
    
    Inputs
    ------
    data: Data
        The Data object that contains the metadata of the 2D array.  
    arr: np.ndarray
        The 1D array that will be converted to an xarray DataArray.
        
    Returns
    -------
        The xarray DataArray that contains the 2D array.
    '''
    
    # Check if the array is full resolution raw
    full_res_raw = arr.size == data.LUMAP_NO_RESFACTOR.size
    
    # Get the geo metadata of the array
    geo_meta = data.GEO_META_FULLRES if full_res_raw else data.GEO_META
    
    # Warp the 1D array to 2D
    arr_2d = full_res_1d_raw_to_2d(data, arr) if full_res_raw else get_coarse2D_map(data, arr)
    arr_2d = np.where(arr_2d == data.NODATA, np.nan, arr_2d)   # Mask the nodata values to nan

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**geo_meta) as dataset:
            # Write the array data to the virtual dataset
            dataset.write(arr_2d, 1)
            # Read the virtual dataset into an xarray DataArray
            da_raster = rxr.open_rasterio(memfile).squeeze(drop=True)
            # Make sure the DataArray has the correct values. The rxr.open_rasterio function will loss the values of the array
            da_raster.values = arr_2d
            # Drop all attributes
            da_raster.attrs = {}
            
    return da_raster

    


def mrj_to_xr(data:Data, in_mrj:np.ndarray) -> xr.DataArray:
    '''
    This function converts a `mrj` array to an xarray DataArray and give each dimension a valida name.
        - The `m` dimension will have names from ['dry', 'irr']; 
        - The `j` dimension will have names from ['Apples', 'Beef - modified land', 'Beef - natural land', 'Citrus', ...,  'Winter legumes', 'Winter oilseeds'].
        - The `r` dimension will be expanded into `x` and `y` (lon/lat) coordinates.
        
    Note
    ----
    When converting the 1D array to 2D, some pixels are lost. \n
    This is because the 1D array can not be converted to a perfect square. \n
    We fill these lost pixels with `np.nan`.
    
    Inputs
    ------
    data: Data
        The Data object that contains the metadata of the 2D array.
    in_mrj: np.ndarray
        The 3D array that will be converted to an xarray DataArray.

    Returns
    -------
        The xarray DataArray that is georeferenced and have valid dimension names.
    '''
    # The `j` dimension can be one of `AGRICULTURAL_LANDUSES` and `PRODUCTS`. Determine the correct dimension names according to the length of the `j` dimension.
    j_vals = data.AGRICULTURAL_LANDUSES if in_mrj.shape[2] == len(data.AGRICULTURAL_LANDUSES) else data.PRODUCTS
    
    mrj_xr = []
    for m,j in product(range(in_mrj.shape[0]), range(in_mrj.shape[2])):
        arr = in_mrj[m,:,j]
        lm = data.LANDMANS[m]
        lu = j_vals[j]
        map_xr = arr_to_xr(data, arr).expand_dims({'lm': [lm], 'lu': [lu]})
        mrj_xr.append(map_xr)
        
    return xr.combine_by_coords(mrj_xr)





def map_to_4band(_map:np.ndarray, color_dict:dict) -> np.ndarray:
    '''
    Convert a 2D map to a 4-band map (RGBA).
    
    Inputs
    ------
     - _map: 2D array, the map to be converted.
     - color_dict: dict, the color mapping dict.
    
    Returns
    -------
     - 4D array, the 4-band RGBA map.
    '''
    arr_4band = np.zeros((_map.shape[0], _map.shape[1], 4), dtype='uint8')

    for k, v in color_dict.items():
        arr_4band[_map == k] = v
        
    return arr_4band


def get_color_lookup(_map:np.ndarray, cell_names:list, colors:dict):
    '''
    Get the color lookup for the map.
    
    Input
    -----
    - _map: 1D np.ndarray, the map
    - cell_names: list, the cell names
    - colors: dict, the color dict
    
    Output
    ------
    - color_dict: dict, the color dict
    - color_desc_dict: dict, the color description dict
    '''
    cell_vals = [i for i in np.unique(_map) if not np.isnan(i)]

    color_dict = dict(zip(cell_vals, colors))
    color_desc_dict = dict(zip(colors, cell_names))
    return color_dict, color_desc_dict



def plot_4band_map(arr_4band:np.ndarray, color_desc_dict:dict, legend_opt:dict, ax=None):
    '''
    Plot the 4-band map.
    
    Input
    -----
    - arr_4band: np.ndarray, the 4-band map
    - color_desc_dict: dict, the color description dict
    - legend_opt: dict, the legend options
    - ax: matplotlib.axes.Axes, the axes to plot the map
        
    Output
    ------
    - None
    '''
    if ax is None:
        fig, ax = plt.subplots()
        ax.imshow(arr_4band)
    else:
        ax.imshow(arr_4band)

    patches = [mpatches.Patch(color=tuple(value / 255 for value in k), label=v) 
                for k, v in color_desc_dict.items()]

    plt.legend(handles=patches, **legend_opt)
    
    
    
    
def map_to_plot(_map, colors, cell_names, legend_params, ax=None):
    '''
    Function to plot the 4-band map.
    
    Inputs
    ------
    - _map: np.ndarray
        The map to be plotted.
    - colors: dict
        The color dictionary.
    - cell_names: list
        The cell names.
    - legend_params: dict
        The legend parameters.
        
    Returns
    -------
    - None
    '''
    # Get the color lookup dictionary
    color_dict, color_desc_dict = get_color_lookup(_map, cell_names, colors)

    # Get the 4-band map
    arr_4band = map_to_4band(_map, color_dict)

    # Plot the 4-band map
    plot_4band_map(arr_4band, color_desc_dict, legend_params, ax)
    
    
    
def add_meta_to_nb(ipath):

    # Search through each notebook and look for the text, add a tag if necessary
    ntbk = nbf.read(ipath, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        cell_tags = cell.get('metadata', {}).get('tags', [])
        for key, val in NOTEBOOK_META_DICT.items():
            if key in cell['source']:
                cell_tags = [val]
        if len(cell_tags) > 0:
            cell['metadata']['tags'] = cell_tags

    nbf.write(ntbk, ipath)