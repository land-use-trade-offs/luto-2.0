import os
import concurrent

import numpy as np

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.settings import NON_AG_LAND_USES, OUTPUT_DIR,WRITE_THREADS, MODE
from luto.data import Data
from luto.tools.spatializers import create_2d_map, write_gtiff
from luto.tools.report.write_input_data import ag_in_data, ag_mam_in_data, non_ag_in_data
from luto.solvers.input_data import get_input_data



def write_input2tiff(data: Data, year):
    """
    Write input data to TIFF files for a specific year of the simulation.

    Args:
        data (Data): The Data object.
        year (int): The year for which to write the input data.

    Raises:
        ValueError: If the specified year is not a valid year for the simulation.

    Returns:
        None
    """
    
    # Update the Geo metadata before writing the input data to TIFF files
    data.GEO_META['nodata'] = data.NODATA
    data.GEO_META['dtype'] = 'float32'
    data.GEO_META['compress'] = 'lzw'
    

    # Check if the year is valid
    if year not in data.lumaps.keys():
        raise ValueError(f"Year {year} is not a valid year for the simulation")

    # Create the output directory
    out_dir = f"{OUTPUT_DIR}/input_data/in_{year}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get the object containing the input data
    base_year = None
    if MODE == 'snapshot':
        base_year = data.YR_CAL_BASE
    elif MODE == 'timeseries':
        base_year = year - 1
    else:
        raise ValueError("MODE setting not valid. Must be either 'snapshot' or 'timeseries'. Please check settings.py.")

    input_data = get_input_data(data, base_year, year)
    # Global the {index: desc} dictionary. The <index> is the position of the array, and the <desc> is the description of the array
    get_idx2desc(data)
    # Write the input data to TIFF files
    input2tiff(data, input_data, out_dir)
    
    
    
    
def get_idx2desc(data: Data):
    
    global lm_code2desc, ag_idx2desc, am_idx2desc, non_ag_idx2desc, commodity_idx2desc, products_idx2desc

    lm_code2desc = dict(enumerate(data.LANDMANS))
    ag_idx2desc  = {v:k for k,v in data.DESC2AGLU.items()}
    am_idx2desc = {k:dict(enumerate(v)) for k,v in AG_MANAGEMENTS_TO_LAND_USES.items()}
    non_ag_idx2desc = dict(enumerate(NON_AG_LAND_USES.keys()))
    commodity_idx2desc = dict(enumerate(data.COMMODITIES))
    products_idx2desc = dict(enumerate(data.PRODUCTS))
    
    
def get_arry(in_data, in_attr, am=None):
    if am:
        return in_data.__getattribute__(in_attr)[am]
    else:
        return in_data.__getattribute__(in_attr)
    
def slice_lu_lm(in_arry, *args):
    slice_args = (*args, )
    return in_arry.__getitem__(slice_args)



# mrj -> tif
def write_ag_arr2tif(data: Data, in_data, in_attr, lm_idx, lu_idx, out_dir):
    
    out_file = f'{out_dir}/{in_attr}_{ag_idx2desc[lu_idx]}_{lm_code2desc[lm_idx]}.tif'
    in_arry = get_arry(in_data, in_attr)
    arr_lu_lm = slice_lu_lm(in_arry, lm_idx, slice(None), lu_idx)
    arr_lu_lm = create_2d_map(data, arr_lu_lm, -1)
    write_gtiff(arr_lu_lm, out_file, data=data)

    
# rk -> tif    
def write_non_ag_arr2tif(data: Data, in_data, in_attr, lu_idx, out_dir):
    
    out_file = f'{out_dir}/{in_attr}_{non_ag_idx2desc[lu_idx]}.tif'
    in_arry = get_arry(in_data, in_attr)
    arr_lu = slice_lu_lm(in_arry, slice(None), lu_idx)
    arr_lu = create_2d_map(data, arr_lu, -1)
    arr_lu = arr_lu.astype(np.float32, casting='unsafe')
    write_gtiff(arr_lu, out_file, data=data)

    
 # {am} mrj -> tif   
def write_am_arr2tif(data: Data, in_data, in_attr, am, lm_idx, lu_idx, out_dir):
    
    out_file = f'{out_dir}/{in_attr}_{am}_{am_idx2desc[am][lu_idx]}_{lm_code2desc[lm_idx]}.tif'
    in_arry = get_arry(in_data, in_attr, am)
    arr_lu_lm = slice_lu_lm(in_arry, lm_idx, slice(None), lu_idx)
    arr_lu_lm = create_2d_map(data, arr_lu_lm, -1)
    arr_lu_lm = arr_lu_lm.astype(np.float32, casting='unsafe')
    write_gtiff(arr_lu_lm,out_file, data=data)

    
# {ag} mrp -> tif    
def write_ag_product_arr2tif(data: Data, in_data, in_attr,lm_idx, product_code, out_dir):
    
    
    out_file = f'{out_dir}/{in_attr}_{products_idx2desc[product_code]}_{lm_code2desc[lm_idx]}.tif'
    in_arry = get_arry(in_data, in_attr)
    arr_p = slice_lu_lm(in_arry, lm_idx, slice(None), product_code)
    arr_p = create_2d_map(data, arr_p, -1) 
    arr_p = arr_p.astype(np.float32, casting='unsafe')
    write_gtiff(arr_p, out_file, data=data)

    
# {am} mrp -> tif   
def write_am_product_arr2tif(data: Data, in_data, in_attr, am, lm_idx, product_idx, out_dir):
    
    out_file = f'{out_dir}/{in_attr}_{am}_{products_idx2desc[product_idx]}_{lm_code2desc[lm_idx]}.tif'
    in_arry = get_arry(in_data, in_attr, am=am)
    arr_p = slice_lu_lm(in_arry, lm_idx, slice(None), product_idx)
    arr_p = create_2d_map(data, arr_p, -1) 
    arr_p = arr_p.astype(np.float32, casting='unsafe')
    write_gtiff(arr_p, out_file, data=data)

    
# {non_ag} crk -> tif
def write_non_ag_commodity_arr2tif(data: Data, in_data, in_attr, commodity_idx, non_ag_idx, out_dir):
    
    out_file = f'{out_dir}/{in_attr}_{non_ag_idx2desc[non_ag_idx]}_{commodity_idx2desc[commodity_idx]}.tif'
    in_arry = get_arry(in_data, in_attr)
    arr_c = slice_lu_lm(in_arry, commodity_idx, slice(None), non_ag_idx)
    arr_c = create_2d_map(data, arr_c, -1) 
    arr_c = arr_c.astype(np.float32, casting='unsafe')
    write_gtiff(arr_c, out_file, data=data)
    
    

def input2tiff(data: Data, input_data, out_dir):
    with concurrent.futures.ThreadPoolExecutor(WRITE_THREADS) as executor:
        futures = []

        # ag_mrj
        for att in ag_in_data:
            for lu_idx in ag_idx2desc:
                futures.extend(
                    executor.submit(
                        write_ag_arr2tif,
                        data,
                        input_data,
                        att,
                        lm_idx,
                        lu_idx,
                        out_dir,
                    )
                    for lm_idx in lm_code2desc
                )
        # am_mrj
        for att in ag_mam_in_data:
            for am in am_idx2desc:
                for lu_code in am_idx2desc[am]:
                    futures.extend(
                        executor.submit(
                            write_am_arr2tif,
                            data,
                            input_data,
                            att,
                            am,
                            lm_idx,
                            lu_code,
                            out_dir,
                        )
                        for lm_idx in lm_code2desc
                    )
        # non_ag_rk
        for att in non_ag_in_data:
            futures.extend(
                executor.submit(
                    write_non_ag_arr2tif,
                    data,
                    input_data,
                    att,
                    non_ag_idx,
                    out_dir,
                )
                for non_ag_idx in non_ag_idx2desc
            )
        # ag_q_mrp
        for lm_code in lm_code2desc:
            futures.extend(
                executor.submit(
                    write_ag_product_arr2tif,
                    data,
                    input_data,
                    'ag_q_mrp',
                    lm_code,
                    product_code,
                    out_dir,
                )
                for product_code in products_idx2desc
            )
        # ag_man_q_mrp
        for am in am_idx2desc:
            for lm_code in lm_code2desc:
                futures.extend(
                    executor.submit(
                        write_am_product_arr2tif,
                        data,
                        input_data,
                        'ag_man_q_mrp',
                        am,
                        lm_code,
                        product_code,
                        out_dir,
                    )
                    for product_code in products_idx2desc
                )
        # non_ag_q_crk
        for commodity_idx in commodity_idx2desc:
            futures.extend(
                executor.submit(
                    write_non_ag_commodity_arr2tif,
                    data,
                    input_data,
                    'non_ag_q_crk',
                    commodity_idx,
                    non_ag_idx,
                    out_dir,
                )
                for non_ag_idx in non_ag_idx2desc
            )
        # Execute the futures as they are completed, report the process
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')
