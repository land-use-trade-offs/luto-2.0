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
import sparse
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import geopandas as gpd

from luto import tools
import luto.settings as settings
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.non_agricultural.quantity as non_ag_quantity
import luto.economics.agricultural.water as ag_water

from collections import defaultdict
from typing import Any, Literal, Optional
from affine import Affine
from scipy.interpolate import interp1d
from math import ceil
from dataclasses import dataclass



def dict2matrix(d, fromlist, tolist):
    """Return 0-1 matrix mapping 'from-vectors' to 'to-vectors' using dict d."""
    A = np.zeros((len(tolist), len(fromlist)), dtype=np.int8)
    for j, jstr in enumerate(fromlist):
        for istr in d[jstr]:
            i = tolist.index(istr)
            A[i, j] = True
    return A


def get_base_am_vars(ncells, ncms, n_ag_lus):
    """
    Get the 2010 agricultural management option vars.
    It is assumed that no agricultural management options were used in 2010,
    so get zero arrays in the correct format.
    """
    am_vars = {}
    for am in settings.AG_MANAGEMENTS_TO_LAND_USES:
        am_vars[am] = np.zeros((ncms, ncells, n_ag_lus))

    return am_vars



def lumap2non_ag_l_mk(lumap, num_non_ag_land_uses: int):
    """
    Convert the land-use map to a decision variable X_rk, where 'r' indexes cell and
    'k' indexes non-agricultural land use.

    Cells used for agricultural purposes have value 0 for all k.
    """
    base_code = settings.NON_AGRICULTURAL_LU_BASE_CODE
    non_ag_lu_codes = list(range(base_code, base_code + num_non_ag_land_uses))

    # Set up a container array of shape r, k.
    x_rk = np.zeros((lumap.shape[0], num_non_ag_land_uses), dtype=bool)

    for i,k in enumerate(non_ag_lu_codes):
        kmap = np.where(lumap == k, True, False)
        x_rk[:, i] = kmap

    return x_rk.astype(bool)


@dataclass
class Data:
    """
    Contains all data required for the LUTO model to run. Loads all data upon initialisation.
    """

    # Restore identity-based hashing (dataclass __eq__ silently sets __hash__ = None)
    __hash__ = object.__hash__

    def __init__(self) -> None:
        """
        Sets up output containers (lumaps, lmmaps, etc) and loads all LUTO data, adjusted
        for resfactor.
        """
        # Path for write module - overwrite when provided with a base and target year
        self.path = None

        # The latest simulation year; 
        #   For simulation between 2010-2050, if the run stops at 2030, then it will be 2030
        #   The last_year is updated in the solve_timeseries.simulation module
        self.last_year = None

        # Setup output containers
        self.lumaps = {}
        self.lmmaps = {}
        self.ammaps = {}
        self.ag_dvars = {}
        self.non_ag_dvars = {}
        self.delta_dvars_ag2nonag = {}
        self.ag_man_dvars = {}
        self.delta_dvars_ag2ag = {}
        self.delta_dvars_nonag2ag = {}
        self.prod_data = {}
        self.obj_vals = {}

        print('')
        print(f'Data Initialization at RES{settings.RESFACTOR}')

        self.YR_CAL_BASE = 2010  # The base year, i.e. where year index yr_idx == 0.

        ###############################################################
        # Masking and spatial coarse graining.
        ###############################################################
        print("├── Setting up masking and spatial course graining data", flush=True)

        # Set resfactor multiplier
        self.RESMULT = settings.RESFACTOR ** 2

        # Set the nodata and non-ag code
        self.NODATA = -9999
        self.MASK_LU_CODE = -1

        # Load LUMAP without resfactor
        self.LUMAP_NO_RESFACTOR = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "lumap.nc")).astype(np.int8)                                         # 1D (ij flattend),  0-27 for land uses; -1 for non-agricultural land uses; All cells in Australia (land only)

        # NLUM mask.
        with rasterio.open(os.path.join(settings.INPUT_DIR, "NLUM_2010-11_mask.tif")) as rst:
            self.NLUM_MASK = rst.read(1).astype(np.int8)                                                                # 2D map,  0 for ocean, 1 for land
            self.LUMAP_2D_FULLRES = np.full_like(self.NLUM_MASK, self.NODATA, dtype=np.int16)                           # 2D map,  full of nodata (-9999)
            np.place(self.LUMAP_2D_FULLRES, self.NLUM_MASK == 1, self.LUMAP_NO_RESFACTOR.values)                               # 2D map,  -9999 for ocean; -1 for desert, urban, water, etc; 0-27 for land uses
            self.GEO_META_FULLRES = rst.meta                                                                            # dict,  key-value pairs of geospatial metadata for the full resolution land-use map
            self.GEO_META_FULLRES['dtype'] = 'float32'                                                                  # Set the data type to float32
            self.GEO_META_FULLRES['nodata'] = self.NODATA                                                               # Set the nodata value to -9999


        # Mask out non-agricultural, non-environmental plantings land (i.e., -1) from lumap 
        self.LUMASK = self.LUMAP_NO_RESFACTOR.values != self.MASK_LU_CODE                                                      # 1D (ij flattend);  `True` for land uses; `False` for desert, urban, water, etc
        self.FULLRES_1D_TO_2D = np.vstack(np.nonzero(self.NLUM_MASK))

        if settings.RESFACTOR > 1:
            fullres_rows, fullres_cols = self.FULLRES_1D_TO_2D

            # The block of every full-res land cell, numbered row-major on a grid of whole RF x RF blocks
            n_block_rows = ceil(self.NLUM_MASK.shape[0] / settings.RESFACTOR)
            n_block_cols = ceil(self.NLUM_MASK.shape[1] / settings.RESFACTOR)
            block_id_of_cell = (fullres_rows // settings.RESFACTOR) * n_block_cols + fullres_cols // settings.RESFACTOR

            # A block is a model cell if it holds >= 1 land-use cell
            has_land_use = np.bincount(block_id_of_cell, weights=self.LUMASK, minlength=n_block_rows * n_block_cols) > 0
            model_blocks = np.flatnonzero(has_land_use)

            # Each model cell's representative cell, where its inputs are read (MASK): the block's centre cell where the
            # centre is land; for a coastal block whose centre is sea, the block's land-use cell nearest the centre
            fullres_idx_2d = np.full(self.NLUM_MASK.shape, -1, dtype=np.int64)
            fullres_idx_2d[fullres_rows, fullres_cols] = np.arange(fullres_rows.size)
            block_rows, block_cols = np.divmod(model_blocks, n_block_cols)
            centre_rows = block_rows * settings.RESFACTOR + settings.RESFACTOR // 2
            centre_cols = block_cols * settings.RESFACTOR + settings.RESFACTOR // 2
            inside_grid = (centre_rows < self.NLUM_MASK.shape[0]) & (centre_cols < self.NLUM_MASK.shape[1])
            representative_cell = np.full(model_blocks.size, -1, dtype=np.int64)                  # -1: centre is sea (or off the grid)
            representative_cell[inside_grid] = fullres_idx_2d[centre_rows[inside_grid], centre_cols[inside_grid]]

            needs_substitute = np.zeros(n_block_rows * n_block_cols, dtype=bool)
            needs_substitute[model_blocks[representative_cell < 0]] = True
            candidate_cells = np.flatnonzero(self.LUMASK & needs_substitute[block_id_of_cell])       # land-use cells of those blocks
            candidate_blocks = block_id_of_cell[candidate_cells]
            distance_to_centre = ((fullres_rows[candidate_cells] - (candidate_blocks // n_block_cols * settings.RESFACTOR + settings.RESFACTOR // 2)) ** 2
                                  + (fullres_cols[candidate_cells] - (candidate_blocks % n_block_cols * settings.RESFACTOR + settings.RESFACTOR // 2)) ** 2)
            nearest_first = np.lexsort((candidate_cells, distance_to_centre, candidate_blocks))      # per block: nearest, ties by position
            substitute_blocks, first = np.unique(candidate_blocks[nearest_first], return_index=True)
            position_of_block = np.full(n_block_rows * n_block_cols, -1, dtype=np.int64)
            position_of_block[model_blocks] = np.arange(model_blocks.size)
            representative_cell[position_of_block[substitute_blocks]] = candidate_cells[nearest_first[first]]

            # The model cells in the full-res order of their representative cells: inputs are read with
            # `read_hdf(where=self.MASK)`, which returns rows in full-res order. Centre cells keep row-major block order;
            # a substitute cell can sit off its block row's centre row, so a coastal cell may come before a neighbour.
            in_read_order = np.argsort(representative_cell)
            model_blocks, representative_cell = model_blocks[in_read_order], representative_cell[in_read_order]
            self.RES_1D_TO_2D = np.vstack(np.divmod(model_blocks, n_block_cols))

            # Each full-res land cell reads its block's model-cell index (-1: block not modelled)
            model_cell_id_of_block = np.full(n_block_rows * n_block_cols, -1, dtype=np.int32)
            model_cell_id_of_block[model_blocks] = np.arange(model_blocks.size)
            self.FULL2RES_IDX_MAP = model_cell_id_of_block[block_id_of_cell]

            # MASK: the representative cell of each model cell, among the full-res land cells
            self.MASK = np.zeros(fullres_rows.size, dtype=bool)
            self.MASK[representative_cell] = True

            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES[settings.RESFACTOR//2::settings.RESFACTOR, settings.RESFACTOR//2::settings.RESFACTOR]
            self.GEO_META = self.update_geo_meta()

        elif settings.RESFACTOR == 1:
            # A block is one cell: the model cells are the land-use cells
            self.MASK = self.LUMASK
            self.RES_1D_TO_2D = self.FULLRES_1D_TO_2D[:, self.LUMASK]
            self.FULL2RES_IDX_MAP = np.where(self.LUMASK, np.cumsum(self.LUMASK) - 1, -1).astype(np.int32)
            self.GEO_META = self.GEO_META_FULLRES
            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES
        else:
            raise KeyError("Resfactor setting invalid")

        
        # Get the lon/lat coordinates.
        self.COORD_LON_LAT_2D_FULLRES = self.get_coord(np.nonzero(self.NLUM_MASK), self.GEO_META_FULLRES['transform'])     # 2D array([lon, ...], [lat, ...]);  lon/lat coordinates for each cell in Australia (land only)
        self.COORD_LON_LAT = [i[self.MASK] for i in self.COORD_LON_LAT_2D_FULLRES]  # Only keep the coordinates for the cells that are not masked out (i.e., land uses). 2D array([lon, ...], [lat, ...]);  lon/lat coordinates for each cell in Australia (land only) and not masked out
        
        

        ###############################################################
        # Load agricultural crop and livestock economic and yield data.
        ###############################################################
        print("├── Loading agricultural crop and livestock data", flush=True)
        #   (column, cell) at the model cells, the columns labelled (field, lm, lu) / (field, animal)
        #   Each table is read one column at a time: a chunk holds one column of every full-res cell, so a column is
        #   inflated whole and cut to the model cells (loading the whole table would hold every full-res column in
        #   memory). The cells are labelled 0 … NCELLS−1 (NCELLS is not set yet: the count of MASK).
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "agec_crops.nc")) as agec_crops_file:
            agec_crops = agec_crops_file['AGEC_CROPS']
            self.AGEC_CROPS = xr.DataArray(
                np.stack([agec_crops.isel(column=i).values[self.MASK] for i in range(agec_crops.sizes['column'])]),
                dims=['column', 'cell'],
                name='AGEC_CROPS',
                coords={
                    'field': ('column', agec_crops['field'].values),
                    'lm': ('column', agec_crops['lm'].values),
                    'lu': ('column', agec_crops['lu'].values),
                    'cell': np.arange(self.MASK.sum()),
                },
            ).set_xindex(['field', 'lm', 'lu'])
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "agec_lvstk.nc")) as agec_lvstk_file:
            agec_lvstk = agec_lvstk_file['AGEC_LVSTK']
            self.AGEC_LVSTK = xr.DataArray(
                np.stack([agec_lvstk.isel(column=i).values[self.MASK] for i in range(agec_lvstk.sizes['column'])]),
                dims=['column', 'cell'],
                name='AGEC_LVSTK',
                coords={
                    'field': ('column', agec_lvstk['field'].values),
                    'animal': ('column', agec_lvstk['animal'].values),
                    'cell': np.arange(self.MASK.sum()),
                },
            ).set_xindex(['field', 'animal'])
        
        # Price multipliers for livestock and crops over the years.
        self.CROP_PRICE_MULTIPLIERS = pd.read_excel(os.path.join(settings.INPUT_DIR, "ag_price_multipliers.xlsx"), sheet_name="AGEC_CROPS", index_col="Year")
        self.LVSTK_PRICE_MULTIPLIERS = pd.read_excel(os.path.join(settings.INPUT_DIR, "ag_price_multipliers.xlsx"), sheet_name="AGEC_LVSTK", index_col="Year")



        ###############################################################
        # Set up lists of land-uses, commodities etc.
        ###############################################################
        print("├── Setting up lists of land uses, commodities, etc.", flush=True)

        # Read in lexicographically ordered list of land-uses.
        self.AGRICULTURAL_LANDUSES = pd.read_csv((os.path.join(settings.INPUT_DIR, 'ag_landuses.csv')), header = None)[0].to_list()
        self.NON_AGRICULTURAL_LANDUSES = list(settings.NON_AG_LAND_USES.keys())

        self.NONAGLU2DESC = dict(zip(range(settings.NON_AGRICULTURAL_LU_BASE_CODE, settings.NON_AGRICULTURAL_LU_BASE_CODE + len(self.NON_AGRICULTURAL_LANDUSES)), self.NON_AGRICULTURAL_LANDUSES))
        self.DESC2NONAGLU = {value: key for key, value in self.NONAGLU2DESC.items()}
 
        # Get number of land-uses
        self.N_AG_LUS = len(self.AGRICULTURAL_LANDUSES)
        self.N_NON_AG_LUS = len(self.NON_AGRICULTURAL_LANDUSES)

        # Construct land-use index dictionary (distinct from LU_IDs!)
        self.AGLU2DESC = {i: lu for i, lu in enumerate(self.AGRICULTURAL_LANDUSES)}
        self.DESC2AGLU = {value: key for key, value in self.AGLU2DESC.items()}
        self.AGLU2DESC[-1] = 'Non-agricultural land'
        
        # Combine ag and non-ag landuses
        self.ALL_LANDUSES = self.AGRICULTURAL_LANDUSES + self.NON_AGRICULTURAL_LANDUSES
        self.ALLDESC2LU = {**self.DESC2AGLU, **self.DESC2NONAGLU}
        self.ALLLU2DESC = {**self.AGLU2DESC, **self.NONAGLU2DESC}

        # Some useful sub-sets of the land uses.
        self.LU_CROPS = [ lu for lu in self.AGRICULTURAL_LANDUSES if 'Beef' not in lu
                                                                  and 'Sheep' not in lu
                                                                  and 'Dairy' not in lu
                                                                  and 'Unallocated' not in lu
                                                                  and 'Non-agricultural' not in lu ]
        self.LU_LVSTK = [ lu for lu in self.AGRICULTURAL_LANDUSES if 'Beef' in lu
                                                        or 'Sheep' in lu
                                                        or 'Dairy' in lu ]
        self.LU_UNALL = [ lu for lu in self.AGRICULTURAL_LANDUSES if 'Unallocated' in lu ]
        self.LU_NATURAL = [
            self.DESC2AGLU["Beef - natural land"],
            self.DESC2AGLU["Dairy - natural land"],
            self.DESC2AGLU["Sheep - natural land"],
            self.DESC2AGLU["Unallocated - natural land"],
        ]
        self.LU_LVSTK_NATURAL = [lu for lu in self.LU_NATURAL if self.AGLU2DESC[lu] != 'Unallocated - natural land']
        self.LU_LVSTK_NATURAL_DESC = [self.AGLU2DESC[lu] for lu in self.LU_LVSTK_NATURAL]
        self.LU_MODIFIED_LAND = [self.DESC2AGLU[lu] for lu in self.AGRICULTURAL_LANDUSES if self.DESC2AGLU[lu] not in self.LU_NATURAL]
        
        self.LU_CROPS_INDICES = [self.AGRICULTURAL_LANDUSES.index(lu) for lu in self.AGRICULTURAL_LANDUSES if lu in self.LU_CROPS]
        self.LU_LVSTK_INDICES = [self.AGRICULTURAL_LANDUSES.index(lu) for lu in self.AGRICULTURAL_LANDUSES if lu in self.LU_LVSTK]
        self.LU_UNALL_INDICES = [self.AGRICULTURAL_LANDUSES.index(lu) for lu in self.AGRICULTURAL_LANDUSES if lu in self.LU_UNALL]

        self.NON_AG_LU_NATURAL = [
            self.DESC2NONAGLU["Environmental Plantings"],
            self.DESC2NONAGLU["Riparian Plantings"],
            self.DESC2NONAGLU["Sheep Agroforestry"],
            self.DESC2NONAGLU["Beef Agroforestry"],
            self.DESC2NONAGLU["Carbon Plantings (Block)"],
            self.DESC2NONAGLU["Sheep Carbon Plantings (Belt)"],
            self.DESC2NONAGLU["Beef Carbon Plantings (Belt)"],
            self.DESC2NONAGLU["BECCS"],
        ]

        # Define which land uses correspond to deep/shallow rooted water yield.
        self.LU_SHALLOW_ROOTED = [
            self.DESC2AGLU["Hay"], self.DESC2AGLU["Summer cereals"], self.DESC2AGLU["Summer legumes"],
            self.DESC2AGLU["Summer oilseeds"], self.DESC2AGLU["Winter cereals"], self.DESC2AGLU["Winter legumes"],
            self.DESC2AGLU["Winter oilseeds"], self.DESC2AGLU["Cotton"], self.DESC2AGLU["Other non-cereal crops"],
            self.DESC2AGLU["Rice"], self.DESC2AGLU["Vegetables"], self.DESC2AGLU["Dairy - modified land"],
            self.DESC2AGLU["Beef - modified land"], self.DESC2AGLU["Sheep - modified land"],
            self.DESC2AGLU["Unallocated - modified land"],
        ]
        self.LU_DEEP_ROOTED = [
            self.DESC2AGLU["Apples"], self.DESC2AGLU["Citrus"], self.DESC2AGLU["Grapes"], self.DESC2AGLU["Nuts"],
            self.DESC2AGLU["Pears"], self.DESC2AGLU["Plantation fruit"], self.DESC2AGLU["Stone fruit"],
            self.DESC2AGLU["Sugar"], self.DESC2AGLU["Tropical stone fruit"],
        ]

        # Derive land management types from AGEC.
        self.LANDMANS = set(self.AGEC_CROPS['lm'].values.tolist())  # Set comp., unique entries.
        self.LANDMANS = list(self.LANDMANS)  # Turn into list.
        self.LANDMANS.sort()  # Ensure lexicographic order.

        # Get number of land management types
        self.NLMS = len(self.LANDMANS)

        # List of products. Everything upper case to avoid mistakes.
        self.PR_CROPS = [s.upper() for s in self.LU_CROPS]
        self.PR_LVSTK = [
            'BEEF - MODIFIED LAND LEXP',
            'BEEF - MODIFIED LAND MEAT',
            'BEEF - NATURAL LAND LEXP',
            'BEEF - NATURAL LAND MEAT',
            
            'DAIRY - MODIFIED LAND',
            'DAIRY - NATURAL LAND',
            
            'SHEEP - MODIFIED LAND LEXP',
            'SHEEP - MODIFIED LAND MEAT',
            'SHEEP - MODIFIED LAND WOOL',
            'SHEEP - NATURAL LAND LEXP',
            'SHEEP - NATURAL LAND MEAT',
            'SHEEP - NATURAL LAND WOOL'
        ]
        
        # Sort each product category alphabetically, then concatenate
        self.PRODUCTS = self.PR_CROPS + self.PR_LVSTK
        self.PRODUCTS.sort()

        # Get number of products
        self.NPRS = len(self.PRODUCTS)

        # Some land-uses map to multiple products -- a dict and matrix to capture this.
        # Crops land-uses and crop products are one-one. Livestock is more complicated.
        self.LU2PR_DICT = {key: [key.upper()] if key in self.LU_CROPS else [] for key in self.AGRICULTURAL_LANDUSES}
        for lu in self.LU_LVSTK:
            for PR in self.PR_LVSTK:
                if lu.upper() in PR:
                    self.LU2PR_DICT[lu] = self.LU2PR_DICT[lu] + [PR]

        # A reverse dictionary for convenience.
        self.PR2LU_DICT = {pr: key for key, val in self.LU2PR_DICT.items() for pr in val}

        self.LU2PR = dict2matrix(self.LU2PR_DICT, self.AGRICULTURAL_LANDUSES, self.PRODUCTS)
        
        self.lu2pr_xr = xr.DataArray(
            self.LU2PR.astype(np.float32),
            dims=['product', 'lu'],
            coords={
                'product': self.PRODUCTS,
                'lu': self.AGRICULTURAL_LANDUSES
            },
        )


        # List of commodities. Everything lower case to avoid mistakes.
        # Basically collapse 'NATURAL LAND' and 'MODIFIED LAND' products and remove duplicates.
        self.COMMODITIES = { ( s.replace(' - NATURAL LAND', '')
                                .replace(' - MODIFIED LAND', '')
                                .lower() )
                                for s in self.PRODUCTS }
        self.COMMODITIES = list(self.COMMODITIES)
        self.COMMODITIES.sort()
        self.CM_CROPS = [s for s in self.COMMODITIES if s in [k.lower() for k in self.LU_CROPS]]

        # Get number of commodities
        self.NCMS = len(self.COMMODITIES)


        # Some commodities map to multiple products -- dict and matrix to capture this.
        # Crops commodities and products are one-one. Livestock is more complicated.
        self.CM2PR_DICT = { key.lower(): [key.upper()] if key in self.CM_CROPS else []
                    for key in self.COMMODITIES }
        for key, _ in self.CM2PR_DICT.items():
            if len(key.split())==1:
                head = key.split()[0]
                tail = 0
            else:
                head = key.split()[0]
                tail = key.split()[1]
            for PR in self.PR_LVSTK:
                if tail==0 and head.upper() in PR:
                    self.CM2PR_DICT[key] = self.CM2PR_DICT[key] + [PR]
                elif (head.upper()) in PR and (tail.upper() in PR):
                    self.CM2PR_DICT[key] = self.CM2PR_DICT[key] + [PR]
                else:
                    ... # Do nothing, this should be a crop.

        self.PR2CM = dict2matrix(self.CM2PR_DICT, self.COMMODITIES, self.PRODUCTS).T # Note the transpose.
        
        self.pr2cm_xr = xr.DataArray(
            self.PR2CM.astype(np.float32),
            dims=['Commodity', 'product'],
            coords={
                'Commodity': self.COMMODITIES,
                'product': self.PRODUCTS
            },
        )
        
        
        # Get the land-use indices for each commodity.
        self.CM2LU_IDX = defaultdict(list)
        for c in self.COMMODITIES:
            for lu in self.AGRICULTURAL_LANDUSES:
                if lu.split(' -')[0].lower() in c:
                    self.CM2LU_IDX[c].append(self.AGRICULTURAL_LANDUSES.index(lu))
                    
                    
        ###############################################################
        # Cost multiplier data.
        ###############################################################
        cost_mult_excel = pd.ExcelFile(os.path.join(settings.INPUT_DIR, 'cost_multipliers.xlsx'))
        self.AC_COST_MULTS = pd.read_excel(cost_mult_excel, "AC_multiplier", index_col="Year")
        self.QC_COST_MULTS = pd.read_excel(cost_mult_excel, "QC_multiplier", index_col="Year")
        self.FOC_COST_MULTS = pd.read_excel(cost_mult_excel, "FOC_multiplier", index_col="Year")
        self.FLC_COST_MULTS = pd.read_excel(cost_mult_excel, "FLC_multiplier", index_col="Year")
        self.FDC_COST_MULTS = pd.read_excel(cost_mult_excel, "FDC_multiplier", index_col="Year")
        self.WP_COST_MULTS = pd.read_excel(cost_mult_excel, "WP_multiplier", index_col="Year")["Water_delivery_price_multiplier"].to_dict()
        self.WATER_LICENSE_COST_MULTS = pd.read_excel(cost_mult_excel, "Water License Cost multiplier", index_col="Year")["Water_license_cost_multiplier"].to_dict()
        self.EST_COST_MULTS = pd.read_excel(cost_mult_excel, "Establishment cost multiplier", index_col="Year")["Establishment_cost_multiplier"].to_dict()
        self.MAINT_COST_MULTS = pd.read_excel(cost_mult_excel, "Maintennance cost multiplier", index_col="Year")["Maintennance_cost_multiplier"].to_dict()
        self.TRANS_COST_MULTS = {yr: v * settings.TRANSITION_COST_MULT for yr, v in pd.read_excel(cost_mult_excel, "Transitions cost multiplier", index_col="Year")["Transitions_cost_multiplier"].items()}
        self.SAVBURN_COST_MULTS = pd.read_excel(cost_mult_excel, "Savanna burning cost multiplier", index_col="Year")["Savanna_burning_cost_multiplier"].to_dict()
        self.IRRIG_COST_MULTS = pd.read_excel(cost_mult_excel, "Irrigation cost multiplier", index_col="Year")["Irrigation_cost_multiplier"].to_dict()
        self.BECCS_COST_MULTS = pd.read_excel(cost_mult_excel, "BECCS cost multiplier", index_col="Year")["BECCS_cost_multiplier"].to_dict()
        self.BECCS_REV_MULTS = pd.read_excel(cost_mult_excel, "BECCS revenue multiplier", index_col="Year")["BECCS_revenue_multiplier"].to_dict()
        self.FENCE_COST_MULTS = pd.read_excel(cost_mult_excel, "Fencing cost multiplier", index_col="Year")["Fencing_cost_multiplier"].to_dict()



        ###############################################################
        # Spatial layers.
        ###############################################################
        print("├── Setting up spatial layers data", flush=True)

        # Actual hectares per cell, including projection corrections.
        self.REAL_AREA_NO_RESFACTOR = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "real_area.nc"))
        real_area_r = self.REAL_AREA_NO_RESFACTOR.isel(cell=self.MASK)
        self.REAL_AREA = (real_area_r * self.RESMULT).assign_coords(cell=np.arange(real_area_r.sizes['cell']))  # over the model cells; TODO: adjusting using actual area

        # Derive NCELLS (number of spatial cells) from the area array.
        self.NCELLS = self.REAL_AREA.sizes['cell']
        
        # Initial (2010) ag decision variable (X_mrj).
        self.LMMAP_NO_RESFACTOR = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "lmmap.nc"))
        self.AG_L_MRJ = self.get_exact_resfactored_lumap_mrj()
        self.add_ag_dvars(self.YR_CAL_BASE, self.AG_L_MRJ)
        
        # Initial (2010) maximum ag dvar proportion
        #   For example, if a cell has 0.2 of ag land-use at beginning,
        #   then, the sum(ag + non_ag) in the following years should be <= 0.2.
        #   This is used as a constraint in the solver to prevent the model 
        #   from allocating more agricultural land than the initial proportion.
        self.AG_MASK_PROPORTION_R = self.AG_L_MRJ.sum(0).sum(1)

        # Initial (2010) land-use map, mapped as lexicographic land-use class indices.
        self.LU_RESFACTOR_CELLS = pd.DataFrame({
            'lu_code': list(self.DESC2AGLU.values()),
            'res_size': [ceil((self.LUMAP_NO_RESFACTOR == lu_code).sum().item() / self.RESMULT) for _,lu_code in self.DESC2AGLU.items()]
        }).sort_values('res_size').reset_index(drop=True)
        
        self.LUMAP = self.get_resfactored_lumap() if settings.RESFACTOR > 1 else self.LUMAP_NO_RESFACTOR.isel(cell=self.MASK).values
        self.add_lumap(self.YR_CAL_BASE, self.LUMAP)

        # Initial (2010) land management map.
        self.LMMAP = self.LMMAP_NO_RESFACTOR.isel(cell=self.MASK).values
        self.add_lmmap(self.YR_CAL_BASE, self.LMMAP)

        # Initial (2010) agricultural management maps - no cells are used for alternative agricultural management options.
        # Includes a separate AM map for each agricultural management option, because they can be stacked.
        self.AG_MAN_DESC = [am for am in settings.AG_MANAGEMENTS if settings.AG_MANAGEMENTS[am]]
        self.AG_MAN_LU_DESC = {am:settings.AG_MANAGEMENTS_TO_LAND_USES[am] for am in self.AG_MAN_DESC}
        self.AG_MAN_MAP = {am: np.zeros(self.NCELLS).astype("int8") for am in self.AG_MAN_DESC}
        self.N_AG_MANS = len(self.AG_MAN_DESC)
        self.add_ammaps(self.YR_CAL_BASE, self.AG_MAN_MAP)

        

        self.NON_AG_L_RK = lumap2non_ag_l_mk(
            self.LUMAP, len(self.NON_AGRICULTURAL_LANDUSES)     # Int8
        )
        self.add_non_ag_dvars(self.YR_CAL_BASE, self.NON_AG_L_RK)

        ###############################################################
        # Climate change impact data.
        ###############################################################
        print("├── Loading climate change data", flush=True)

        #   (column, cell) at the model cells, the columns labelled (lm, lu, year)
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "climate_change_impacts_" + settings.RCP + "_CO2_FERT_" + settings.CO2_FERT.upper() + ".nc")) as cci_file:
            cci = cci_file['CLIMATE_CHANGE_IMPACT']
            self.CLIMATE_CHANGE_IMPACT = xr.DataArray(
                np.stack([cci.isel(column=i).values[self.MASK] for i in range(cci.sizes['column'])]),
                dims=['column', 'cell'],
                name='CLIMATE_CHANGE_IMPACT',
                coords={
                    'lm': ('column', cci['lm'].values),
                    'lu': ('column', cci['lu'].values),
                    'year': ('column', cci['year'].values),
                    'cell': np.arange(self.MASK.sum()),
                },
            ).set_xindex(['lm', 'lu', 'year'])
        
        ###############################################################
        # Regional coverage layers, mainly for regional reporting.
        ###############################################################
        #   Each file holds the region code per cell and each code's name once (over `code`)
        region_nrm = xr.load_dataset(os.path.join(settings.INPUT_DIR, "region_nrm.nc"))
        
        self.REGION_NRM_CODE = region_nrm['NRM_CODE'].isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.REGION_NRM_NAME = region_nrm['NRM_NAME'].sel(code=self.REGION_NRM_CODE.values).values.astype(object)     # str objects, as before
        
        region_state = xr.load_dataset(os.path.join(settings.INPUT_DIR, "region_state.nc"))
        state_codes = region_state['STE_CODE11'].isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        state_names = region_state['STE_NAME11'].sel(code=np.unique(state_codes.values))     # the states with a model cell
        
        self.REGION_STATE_NAME2CODE = {str(name): int(code) for code, name in zip(state_names['code'].values, state_names.values)}
        self.REGION_STATE_NAME2CODE = dict(sorted(self.REGION_STATE_NAME2CODE.items()))     # Make sure the dict is sorted by state name, makes it consistent with renewable target.
        
        if 'Other Territories' in self.REGION_STATE_NAME2CODE:
            self.REGION_STATE_NAME2CODE.pop('Other Territories')                            # Remove 'Other Territories' from the dict.
        
        self.REGION_STATE_CODE = state_codes
        self.REGION_STATE_NAME = region_state['STE_NAME11'].sel(code=state_codes.values).values.astype(object)


        ###############################################################
        # No-Go areas; Regional adoption constraints.
        ###############################################################
        print("├── Loading no-go areas and regional adoption zones", flush=True)
   
        ##################### No-go areas
        self.NO_GO_LANDUSE_AG = []
        self.NO_GO_LANDUSE_NON_AG = []

        for lu in settings.NO_GO_VECTORS.keys():
            if lu in self.AGRICULTURAL_LANDUSES:
                self.NO_GO_LANDUSE_AG.append(lu)
            elif lu in self.NON_AGRICULTURAL_LANDUSES:
                self.NO_GO_LANDUSE_NON_AG.append(lu)
            else:
                raise KeyError(f"Land use '{lu}' in no-go area vector does not match any land use in the model.")

        no_go_arrs_ag = []
        no_go_arrs_non_ag = []

        for lu, no_go_path in settings.NO_GO_VECTORS.items():
            # Read the no-go area shapefile
            no_go_path = os.path.join(settings.INPUT_DIR, no_go_path)
            no_go_shp = gpd.read_file(no_go_path)
            # Check if the CRS is defined
            if no_go_shp.crs is None:
                raise ValueError(f"{no_go_path} does not have a CRS defined")
            # Rasterize the reforestation vector; Fill with 1.  0 is no-go, 1 is 'free' cells.
            with rasterio.open(settings.INPUT_DIR + '/NLUM_2010-11_mask.tif') as src:
                src_arr = src.read(1)
                src_meta = src.meta.copy()
                no_go_shp = no_go_shp.to_crs(src_meta['crs'])
                no_go_arr = rasterio.features.rasterize(
                    ((row['geometry'], 0) for _,row in no_go_shp.iterrows()),
                    out_shape=(src_meta['height'], src_meta['width']),
                    transform=src_meta['transform'],
                    fill=1,
                    dtype=np.int16
                )
                # Add the no-go area to the ag or non_ag list.
                if lu in self.NO_GO_LANDUSE_AG:
                    no_go_arrs_ag.append(no_go_arr[np.nonzero(src_arr)].astype(np.bool_))
                elif lu in self.NO_GO_LANDUSE_NON_AG:
                    no_go_arrs_non_ag.append(no_go_arr[np.nonzero(src_arr)].astype(np.bool_))

        self.NO_GO_REGION_AG = np.stack(no_go_arrs_ag, axis=0)[:, self.MASK]
        self.NO_GO_REGION_NON_AG = np.stack(no_go_arrs_non_ag, axis=0)[:, self.MASK]
        

        
        ################################
        # Regional adoption zones
        ################################
        if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
            self.REGIONAL_ADOPTION_ZONES = None
            self.REGIONAL_ADOPTION_TARGETS = None

        elif settings.REGIONAL_ADOPTION_CONSTRAINTS == "individual_cap":
            # Per-landuse caps (both ag and non-ag) read from the xlsx, scoped to REGIONAL_ADOPTION_ZONE.
            with xr.open_dataset(os.path.join(settings.INPUT_DIR, "regional_adoption_zones.nc")) as zones:
                self.REGIONAL_ADOPTION_ZONES = (
                    zones[settings.REGIONAL_ADOPTION_ZONE].load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
                )

            regional_adoption_targets = pd.read_excel(
                os.path.join(settings.INPUT_DIR, "regional_adoption_zones.xlsx"),
                sheet_name=settings.REGIONAL_ADOPTION_ZONE
            )

            self.REGIONAL_ADOPTION_TARGETS = regional_adoption_targets.iloc[
                [idx for idx, row in regional_adoption_targets.iterrows() if
                    all([row['ADOPTION_PERCENTAGE_2030']>=0,
                        row['ADOPTION_PERCENTAGE_2050']>=0,
                        row['ADOPTION_PERCENTAGE_2100']>=0])
                ]
            ]

            # Check missing zones due to high resfactor
            lost_zones = np.setdiff1d(
                self.REGIONAL_ADOPTION_TARGETS[settings.REGIONAL_ADOPTION_ZONE].unique(),
                np.unique(self.REGIONAL_ADOPTION_ZONES.values)
            ) if len(self.REGIONAL_ADOPTION_TARGETS) > 0 else np.array([])

            if len(lost_zones) > 0:
                print(f"│   ⚠ WARNING: {len(lost_zones)} regional adoption zones have no cells due to (RES{settings.RESFACTOR}). Please check if this is expected.", flush=True)
                self.REGIONAL_ADOPTION_TARGETS = self.REGIONAL_ADOPTION_TARGETS.query(
                    f"{settings.REGIONAL_ADOPTION_ZONE} not in {list(lost_zones)}"
                ).reset_index(drop=True)

        elif settings.REGIONAL_ADOPTION_CONSTRAINTS == "non_ag_sum_cap":
            # SUM-of-all-non-ag cap per region (NRM or State, controlled by REGIONAL_ADOPTION_NON_AG_REGION).
            # No xlsx involved; ag dvars are unconstrained by this mode.
            self.REGIONAL_ADOPTION_ZONES = None
            self.REGIONAL_ADOPTION_TARGETS = None

        else:
            raise ValueError(
                f"Unknown REGIONAL_ADOPTION_CONSTRAINTS={settings.REGIONAL_ADOPTION_CONSTRAINTS!r}. "
                "Expected one of: 'off', 'individual_cap', 'non_ag_sum_cap' "
                "(renamed 2026-10-15: 'on' -> 'individual_cap', 'NON_AG_CAP' -> 'non_ag_sum_cap')."
            )



        ###############################################################
        # Livestock related data.
        ###############################################################
        print("├── Loading livestock related data", flush=True)

        #   Each a layer over the model cells
        self.FEED_REQ = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "feed_req.nc")).isel(cell=self.MASK).fillna(0).assign_coords(cell=np.arange(self.NCELLS))
        self.PASTURE_KG_DM_HA = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "pasture_kg_dm_ha.nc")).isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.SAFE_PUR_NATL = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "safe_pur_natl.nc")).isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.SAFE_PUR_MODL = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "safe_pur_modl.nc")).isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))



        ###############################################################
        # Agricultural Management options data.
        ###############################################################
        print("├── Loading agricultural management options' data", flush=True)
        
        # Rename soil CO2E to match the AGGHG_CROPS/LVSTK data.
        bundle_rename = {"CO2E_KG_HA_SOIL_N_SURP": "CO2E_KG_HA_SOIL"}


        # Asparagopsis taxiformis data
        asparagopsis_file = os.path.join(settings.INPUT_DIR, "20260317_Bundle_MR.xlsx")
        self.ASPARAGOPSIS_DATA = {
            "Beef - modified land": pd.read_excel(asparagopsis_file, sheet_name="MR bundle (ext cattle)", index_col="Year"),
            "Sheep - modified land": pd.read_excel(asparagopsis_file, sheet_name="MR bundle (sheep)", index_col="Year"),
            "Dairy - natural land": pd.read_excel(asparagopsis_file, sheet_name="MR bundle (dairy)", index_col="Year"),
            "Dairy - modified land": pd.read_excel(asparagopsis_file, sheet_name="MR bundle (dairy)", index_col="Year")
        }
        
        # Ecological grazing data
        eco_grazing_file = os.path.join(settings.INPUT_DIR, "20231107_ECOGRAZE_Bundle.xlsx")
        self.ECOLOGICAL_GRAZING_DATA = {
            "Beef - modified land": pd.read_excel(eco_grazing_file, sheet_name="Ecograze bundle (ext cattle)", index_col="Year"),
            "Sheep - modified land": pd.read_excel(eco_grazing_file, sheet_name="Ecograze bundle (sheep)", index_col="Year"),
            "Dairy - modified land": pd.read_excel(eco_grazing_file, sheet_name="Ecograze bundle (dairy)", index_col="Year")
        }

        # Precision agriculture data
        prec_agr_file = os.path.join(settings.INPUT_DIR, "20260317_Bundle_AgTech_NE.xlsx")
        self.PRECISION_AGRICULTURE_DATA = {
            "cropping":     pd.read_excel(prec_agr_file, sheet_name="AgTech NE bundle (cropping)", index_col="Year"),
            "int_cropping": pd.read_excel(prec_agr_file, sheet_name="AgTech NE bundle (int cropping)", index_col="Year"),
            "horticulture": pd.read_excel(prec_agr_file, sheet_name="AgTech NE bundle (horticulture)", index_col="Year").rename(columns=bundle_rename),
        }

        # Load AgTech EI data
        agtech_ei_file = os.path.join(settings.INPUT_DIR, '20260317_Bundle_AgTech_EI.xlsx')
        self.AGTECH_EI_DATA = {
            "cropping":     pd.read_excel(agtech_ei_file, sheet_name='AgTech EI bundle (cropping)', index_col='Year'),
            "int_cropping": pd.read_excel(agtech_ei_file, sheet_name='AgTech EI bundle (int cropping)', index_col='Year', usecols=lambda c: not str(c).startswith('Unnamed')),
            "horticulture": pd.read_excel(agtech_ei_file, sheet_name='AgTech EI bundle (horticulture)', index_col='Year').rename(columns=bundle_rename),
        }

        # Load BioChar data (no int_cropping group)
        biochar_file = os.path.join(settings.INPUT_DIR, '20260401_Bundle_BC.xlsx')
        self.BIOCHAR_DATA = {
            "cropping":     pd.read_excel(biochar_file, sheet_name='Biochar (cropping)', index_col='Year').rename(columns=bundle_rename),
            "horticulture": pd.read_excel(biochar_file, sheet_name='Biochar (horticulture)', index_col='Year').rename(columns=bundle_rename),
        }
        
        # Load soil carbon data, convert C to CO2e (x 44/12), and average over years
        soil_carbon_t_ha = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "soil_carbon_t_ha.nc")).isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.SOIL_CARBON_AVG_T_CO2_HA_PER_YR = (
            soil_carbon_t_ha.astype(np.float32)
            * (44 / 12)  # Convert C to CO2e
            / settings.CARBON_EFFECTS_WINDOW
        )


        # #########################################################
        # RENEWABLE ENERGY DATA LOADING                           #
        # #########################################################
        if any(settings.RENEWABLES_OPTIONS.values()):

            print("├── Loading renewable energy data...", flush=True)
            
            # Renewable bundle data (productivity impacts, cost multipliers, etc)
            renewable_bundle = pd.read_csv(f'{settings.INPUT_DIR}/renewable_energy_bundle.csv')
            self.RENEWABLE_BUNDLE_WIND = renewable_bundle.query('Lever == "Onshore Wind"')
            self.RENEWABLE_BUNDLE_SOLAR = renewable_bundle.query('Lever == "Utility Solar PV"')

            # Renewable targets and prices
            self.RENEWABLE_TARGETS = (
                pd.read_csv(f'{settings.INPUT_DIR}/renewable_targets.csv')
                .sort_values('state')
                .query(f"scen == '{settings.RENEWABLE_TARGET_SCENARIO_TARGETS}'")
                .assign(Renewable_Target_MWh=lambda df: df['Renewable_Target_TWh'] * 1e6)
            )

            self.SOLAR_PRICES = pd.read_csv(f'{settings.INPUT_DIR}/renewable_price_AUD_MWh_solar.csv')
            self.WIND_PRICES = pd.read_csv(f'{settings.INPUT_DIR}/renewable_price_AUD_MWh_wind.csv')

            # Renewable energy related raster layers
            #   Compute before isel to avoid isel working on dask chunks
            #   which slows down the reading.
            with xr.open_dataset(f'{settings.INPUT_DIR}/renewable_energy_layers_1D.nc') as renewable_layers:
                self.RENEWABLE_LAYERS = (
                    renewable_layers
                    .sel(scenario=settings.RENEWABLE_TARGET_SCENARIO_INPUT_LAYERS)
                    .compute()
                    .isel(cell=self.MASK)
                )


            # Existing capacity 
            #    The existing capacity is MW/cell, so here we need to sum the resfactor cells 
            #    to get the total existing capacity in each resfactor cell. 
            renewable_existing_capacity_lyr = xr.load_dataarray(f'{settings.INPUT_DIR}/renewable_existing_capacity_MW_1D.nc')
            
            renewable_existing_capacity_solar = [
                xr.DataArray(
                    self.get_resfactored_sum(renewable_existing_capacity_lyr.sel(year=year, tech_name='Utility Solar PV')),
                    dims=['cell'],
                    coords={'cell': np.where(self.MASK)[0]}
                ).astype(np.float32)
                for year in renewable_existing_capacity_lyr['year'].values
            ]
            renewable_existing_capacity_wind = [
                xr.DataArray(
                    self.get_resfactored_sum(renewable_existing_capacity_lyr.sel(year=year, tech_name='Onshore Wind')),
                    dims=['cell'],
                    coords={'cell': np.where(self.MASK)[0]}
                ).astype(np.float32)
                for year in renewable_existing_capacity_lyr['year'].values
            ] 
            
            self.RENEWABLE_EXISTING_CAPACITY_LAYER_SOLAR_MWH_CELL = (
                xr.concat(renewable_existing_capacity_solar, dim='year')
                .assign_coords(year=renewable_existing_capacity_lyr['year'].values)
                * 24
                * 365
            )
            self.RENEWABLE_EXISTING_CAPACITY_LAYER_WIND_MWH_CELL = (
                xr.concat(renewable_existing_capacity_wind, dim='year')
                .assign_coords(year=renewable_existing_capacity_lyr['year'].values)
                * 24
                * 365
            )
            
            # Existing dvar fraction of renewable energy in each cell
            re_exist_frac_lyr = xr.load_dataarray(f'{settings.INPUT_DIR}/renewable_existing_capacity_area_fraction_1D.nc')

            re_exist_frac_solar = [
                xr.DataArray(
                    self.get_resfactored_average_fraction(re_exist_frac_lyr.sel(year=year, tech_name='Utility Solar PV')),
                    dims=['cell'],
                    coords={'cell': np.where(self.MASK)[0]}
                ).astype(np.float32)
                for year in re_exist_frac_lyr['year'].values
            ]
            re_exist_frac_wind = [
                xr.DataArray(
                    self.get_resfactored_average_fraction(re_exist_frac_lyr.sel(year=year, tech_name='Onshore Wind')),
                    dims=['cell'],
                    coords={'cell': np.where(self.MASK)[0]}
                ).astype(np.float32)
                for year in re_exist_frac_lyr['year'].values
            ]
            
            self.RENEWABME_EXISTING_DVAR_FRACTION_SOLAR = (
                xr.concat(re_exist_frac_solar, dim='year')
                .assign_coords(year=re_exist_frac_lyr['year'].values)
            )
            self.RENEWABME_EXISTING_DVAR_FRACTION_WIND = (
                xr.concat(re_exist_frac_wind, dim='year')
                .assign_coords(year=re_exist_frac_lyr['year'].values)
            )





        ###############################################################
        # Productivity data.
        ###############################################################
        print("├── Loading productivity data", flush=True)

        # Yield increases.
        if settings.PRODUCTIVITY_TREND == 'BAU':
            fpath = os.path.join(settings.INPUT_DIR, "yieldincreases_bau2022.csv")
            productivity_trend = pd.read_csv(fpath, header=[0, 1]).astype(np.float32)
            productivity_trend.index = productivity_trend.index + self.YR_CAL_BASE  # Adjust year to absolute year
            productivity_trend.index.name = 'Year'
            
            # Convert to xarray for easier accessing.
            self.PRODUCTIVITY_MUL_xr = (
                xr.DataArray(productivity_trend)
                .unstack('dim_1')
                .rename({'Year':'year', 'dim_1_level_0':'lm', 'dim_1_level_1':'product'})
            )
        else: 
            fpath = os.path.join(settings.INPUT_DIR, "yieldincreases_ag_2050.xlsx")
            productivity_trend = pd.read_excel(
                fpath, 
                sheet_name=f'{settings.PRODUCTIVITY_TREND.lower()}', 
                header=[0, 1],
                index_col=0
            ).astype(np.float32)
            
            # Convert to xarray for easier accessing.
            self.PRODUCTIVITY_MUL_xr = (
                xr.DataArray(productivity_trend)
                .unstack('dim_1')
                .rename({'Year':'lm', 'dim_1_level_1':'product', 'dim_0':'year'})
            )




        ###############################################################
        # Auxiliary Spatial Layers
        # (spatial layers not required for production calculation)
        ###############################################################
        print("├── Loading auxiliary spatial layers data", flush=True)

        # Load stream length data in metres of stream per cell
        self.STREAM_LENGTH = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "stream_length_m_cell.nc")).isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        cell_area_ha = self.REAL_AREA_NO_RESFACTOR.isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))    # the representative cell's own area

        # Calculate the proportion of the area of each cell within stream buffer (convert REAL_AREA from ha to m2 and divide m2 by m2)
        self.RP_PROPORTION =  (
            (2 * settings.RIPARIAN_PLANTING_BUFFER_WIDTH * self.STREAM_LENGTH) / (cell_area_ha * 10000)
        ).astype(np.float32)
        # Calculate the length of fencing required for each cell in per hectare terms for riparian plantings
        self.RP_FENCING_LENGTH = (
            (2 * settings.RIPARIAN_PLANTING_TORTUOSITY_FACTOR * self.STREAM_LENGTH) / cell_area_ha
        ).astype(np.float32)



        ###############################################################
        # Additional agricultural GHG data.
        ###############################################################
        print("├── Loading additional agricultural GHG data", flush=True)


        # Load greenhouse gas emissions from agriculture
        #   (column, cell) at the model cells, the columns labelled (field, lm, lu) / (animal, field); the irrigated
        #   pasture emissions one variable per column over cell
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "agGHG_crops.nc")) as agghg_crops_file:
            agghg_crops = agghg_crops_file['AGGHG_CROPS']
            self.AGGHG_CROPS = xr.DataArray(
                np.stack([agghg_crops.isel(column=i).values[self.MASK] for i in range(agghg_crops.sizes['column'])]),
                dims=['column', 'cell'],
                name='AGGHG_CROPS',
                coords={
                    'field': ('column', agghg_crops['field'].values),
                    'lm': ('column', agghg_crops['lm'].values),
                    'lu': ('column', agghg_crops['lu'].values),
                    'cell': np.arange(self.MASK.sum()),
                },
            ).set_xindex(['field', 'lm', 'lu'])
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "agGHG_lvstk.nc")) as agghg_lvstk_file:
            agghg_lvstk = agghg_lvstk_file['AGGHG_LVSTK']
            self.AGGHG_LVSTK = xr.DataArray(
                np.stack([agghg_lvstk.isel(column=i).values[self.MASK] for i in range(agghg_lvstk.sizes['column'])]),
                dims=['column', 'cell'],
                name='AGGHG_LVSTK',
                coords={
                    'animal': ('column', agghg_lvstk['animal'].values),
                    'field': ('column', agghg_lvstk['field'].values),
                    'cell': np.arange(self.MASK.sum()),
                },
            ).set_xindex(['animal', 'field'])
        self.AGGHG_IRRPAST = xr.load_dataset(os.path.join(settings.INPUT_DIR, "agGHG_irrpast.nc")).isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))


        # Raw transition cost matrix. In AUD/ha and ordered lexicographically.
        self.AG_TMATRIX = np.load(os.path.join(settings.INPUT_DIR, "ag_tmatrix.npy"))
        self.AG_TO_DESTOCKED_NATURAL_COSTS_HA = np.load(os.path.join(settings.INPUT_DIR, "ag_to_destock_tmatrix.npy"))

        
  
        # Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
        # When RESFACTOR > 1, average the block's land-use cells (LUMASK, the cells RESFACTOR 1 models); > 0 means eligible.
        x_mrj_full = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "x_mrj.nc"))                                       # (lm, cell, lu) over the FULL-RES cells, no cell coordinate
        if settings.RESFACTOR == 1:
            self.EXCLUDE = x_mrj_full.sel(lm=self.LANDMANS, lu=self.AGRICULTURAL_LANDUSES).isel(cell=self.MASK).values
        else:
            self.EXCLUDE = np.stack([
                np.stack([
                    self.get_resfactored_average_fraction(x_mrj_full.sel(lm=lm, lu=lu).values & self.LUMASK, use_valid_cell_count=False)
                    for lu in self.AGRICULTURAL_LANDUSES
                ], axis=-1)
                for lm in self.LANDMANS
            ]) > 0

            # The agricultural economics, GHG, pasture and climate arrays above were read at each block's
            # centre cell. Re-derive them from the block's eligible fine cells (see coarsen_guarded_arrays).
            self.LVSTK_COARSE = self.coarsen_guarded_arrays(x_mrj_full)



        ###############################################################
        # Non-agricultural data.
        ###############################################################
        print("├── Loading non-agricultural data", flush=True)

        # Load plantings economic data
        est_cost = xr.load_dataset(os.path.join(settings.INPUT_DIR, "plantings_est_cost_ha.nc")).isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS)).astype(np.float32)
        self.EP_EST_COST_HA = est_cost['EP_EST_COST_HA']
        self.RP_EST_COST_HA = self.EP_EST_COST_HA.copy()  # Riparian plantings have the same establishment cost as environmental plantings
        self.AF_EST_COST_HA = self.EP_EST_COST_HA.copy()  # Agroforestry plantings have the same establishment cost as environmental plantings
        self.CP_EST_COST_HA = est_cost['CP_EST_COST_HA']

        # Load fire risk data (reduced carbon sequestration by this amount); the levels are the settings.FIRE_RISK options
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "fire_risk.nc")) as fire_risk_ds:
            fire_risk = fire_risk_ds['FD_RISK'].sel(fire_risk=settings.FIRE_RISK, drop=True).load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))


        # Load environmental plantings (block) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        #   NOTE: Use .sel(age=...).load().isel(cell=self.MASK) instead of .sel(cell=self.MASK) 
        #   to avoid slow xarray label-based indexing on large cell dimensions
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_block.nc")) as ds:
            ds = ds.sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.EP_BLOCK_AVG_T_CO2_HA_PER_YR = (
            (ds['EP_BLOCK_TREES_T_CO2_HA'] + ds['EP_BLOCK_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['EP_BLOCK_SOIL_T_CO2_HA']
        ) / settings.CARBON_EFFECTS_WINDOW

        # Load environmental plantings (belt) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_belt.nc")) as ds:
            ds = ds.sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.EP_BELT_AVG_T_CO2_HA_PER_YR = (
            (ds['EP_BELT_TREES_T_CO2_HA'] + ds['EP_BELT_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['EP_BELT_SOIL_T_CO2_HA']
        ) / settings.CARBON_EFFECTS_WINDOW

        # Load environmental plantings (riparian) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_rip.nc")) as ds:
            ds = ds.sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.EP_RIP_AVG_T_CO2_HA_PER_YR = (
            (ds['EP_RIP_TREES_T_CO2_HA'] + ds['EP_RIP_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['EP_RIP_SOIL_T_CO2_HA']
        ) / settings.CARBON_EFFECTS_WINDOW

        # Load carbon plantings (block) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_cp_block.nc")) as ds:
            ds = ds.sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.CP_BLOCK_AVG_T_CO2_HA_PER_YR = (
            (ds['CP_BLOCK_TREES_T_CO2_HA'] + ds['CP_BLOCK_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['CP_BLOCK_SOIL_T_CO2_HA']
        ) / settings.CARBON_EFFECTS_WINDOW

        # Load farm forestry [i.e. carbon plantings (belt)] GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_cp_belt.nc")) as ds:
            ds = ds.sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.CP_BELT_AVG_T_CO2_HA_PER_YR = (
            (ds['CP_BELT_TREES_T_CO2_HA'] + ds['CP_BELT_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['CP_BELT_SOIL_T_CO2_HA']
        ) / settings.CARBON_EFFECTS_WINDOW

        # Agricultural land use to plantings raw transition costs:
        self.AG2EP_TRANSITION_COSTS_HA = np.load(
            os.path.join(settings.INPUT_DIR, "ag_to_ep_tmatrix.npy")
        )  # shape: (28,)

        # EP to agricultural land use transition costs:
        self.EP2AG_TRANSITION_COSTS_HA = np.load(
            os.path.join(settings.INPUT_DIR, "ep_to_ag_tmatrix.npy")
        )  # shape: (28,)
        
        
        ##############################################################
        # Transition cost for all land use
        #############################################################
        
        # Transition matrix from ag
        tmat_ag2ag_xr = xr.DataArray(
            self.AG_TMATRIX,
            dims=['from_lu','to_lu'],
            coords={'from_lu':self.AGRICULTURAL_LANDUSES, 'to_lu':self.AGRICULTURAL_LANDUSES }
        )
        tmat_ag2non_ag_xr = xr.DataArray(
            np.repeat(self.AG2EP_TRANSITION_COSTS_HA.reshape(-1,1), len(self.NON_AGRICULTURAL_LANDUSES), axis=1),
            dims=['from_lu','to_lu'],
            coords={'from_lu':self.AGRICULTURAL_LANDUSES, 'to_lu':self.NON_AGRICULTURAL_LANDUSES}
        )
        tmat_from_ag_xr = xr.concat([tmat_ag2ag_xr, tmat_ag2non_ag_xr], dim='to_lu')                        # Combine ag2ag and ag2non-ag
        tmat_from_ag_xr.loc[:,'Destocked - natural land'] = self.AG_TO_DESTOCKED_NATURAL_COSTS_HA           # Ag to Destock-natural has its own values
        
        
        # Transition matrix of non-ag to unallocated-modified land (land clearing)
        tmat_wood_clear = np.load(os.path.join(settings.INPUT_DIR, 'transition_cost_clearing_forest.npz'))
        
        tmat_clear_EP = tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']
        tmat_clear_RP = tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']
        tmat_clear_sheep_ag_forest = (tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']) * settings.AF_PROPORTION
        tmat_clear_beef_ag_forest = (tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']) * settings.AF_PROPORTION
        tmat_clear_CP = tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']
        tmat_clear_sheep_CP = (tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']) * settings.CP_BELT_PROPORTION
        tmat_clear_beef_CP = (tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']) * settings.CP_BELT_PROPORTION
        tmat_clear_BECCS = tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']
        tmat_clear_destocked_nat = tmat_wood_clear['tmat_clear_light_wood'] + tmat_wood_clear['tmat_clear_dense_wood']
        
        tmat_costs = np.array([
            tmat_clear_EP, tmat_clear_RP, tmat_clear_sheep_ag_forest, tmat_clear_beef_ag_forest,
            tmat_clear_CP, tmat_clear_sheep_CP, tmat_clear_beef_CP, tmat_clear_BECCS, tmat_clear_destocked_nat
        ]).T
        
        
        
        
        # Transition matrix from non-ag
        tmat_non_ag2ag_xr = xr.DataArray(
            np.repeat(self.EP2AG_TRANSITION_COSTS_HA.reshape(1,-1), len(self.NON_AGRICULTURAL_LANDUSES), axis=0),
            dims=['from_lu','to_lu'],
            coords={'from_lu':self.NON_AGRICULTURAL_LANDUSES, 'to_lu':self.AGRICULTURAL_LANDUSES }
        )
        tmat_non_ag2non_ag_xr = xr.DataArray(
            np.full((len(self.NON_AGRICULTURAL_LANDUSES), len(self.NON_AGRICULTURAL_LANDUSES)), np.nan),
            dims=['from_lu','to_lu'],
            coords={'from_lu':self.NON_AGRICULTURAL_LANDUSES, 'to_lu':self.NON_AGRICULTURAL_LANDUSES }
        )


        np.fill_diagonal(tmat_non_ag2non_ag_xr.values, 0)                                                   # Lu staty the same has 0 cost
        tmat_from_non_ag_xr = xr.concat([tmat_non_ag2ag_xr, tmat_non_ag2non_ag_xr], dim='to_lu')            # Combine non-ag2ag and non-ag2non-ag
        tmat_from_non_ag_xr.loc['Destocked - natural land', 'Unallocated - natural land'] = np.nan          # Destocked-natural can not transit to unallow-natural
        
   
        # Get the full transition cost matrix
        self.T_MAT = xr.concat([tmat_from_ag_xr, tmat_from_non_ag_xr], dim='from_lu')
        self.T_MAT.loc[self.NON_AGRICULTURAL_LANDUSES, [self.AGLU2DESC[i] for i in self.LU_NATURAL]] = np.nan       # non-ag2natural is not allowed
        self.T_MAT.loc[self.NON_AGRICULTURAL_LANDUSES, 'Unallocated - modified land'] = tmat_costs                  # Clearing non-ag land requires such cost
        self.T_MAT.loc['Destocked - natural land', self.LU_LVSTK_NATURAL_DESC] = self.T_MAT.loc['Unallocated - natural land', self.LU_LVSTK_NATURAL_DESC]   # Destocked-natural transits to LVSTK-natural has the same cost as unallocated-natural to LVSTK-natural
        self.T_MAT = self.T_MAT.astype(np.float32)

        # tools.plot_t_mat(self.T_MAT)
        
        

        ###############################################################
        # Water data.
        ###############################################################
        print("├── Loading water data", flush=True)
        
        # Initialize water constraints to avoid recalculating them every time.
        self.WATER_YIELD_LIMITS = None

        # Water requirements by land use -- LVSTK.
        wreq_lvstk_dry = pd.DataFrame()
        wreq_lvstk_irr = pd.DataFrame()

        # The rj-indexed arrays have zeroes where j is not livestock.
        for lu in self.AGRICULTURAL_LANDUSES:
            if lu in self.LU_LVSTK:
                # First find out which animal is involved.
                animal, _ = ag_quantity.lvs_veg_types(lu)
                # Water requirements per head are for drinking and irrigation.
                wreq_lvstk_dry[lu] = self.agec_lvstk(lu, 'dry').sel(field="WR_DRN", animal=animal).to_series() * settings.LIVESTOCK_DRINKING_WATER
                wreq_lvstk_irr[lu] = (
                    self.agec_lvstk(lu, 'irr').sel(field="WR_IRR", animal=animal).to_series()
                    + self.agec_lvstk(lu, 'irr').sel(field="WR_DRN", animal=animal).to_series() * settings.LIVESTOCK_DRINKING_WATER
                )
            else:
                wreq_lvstk_dry[lu] = 0.0
                wreq_lvstk_irr[lu] = 0.0

        # Water requirements by land use -- CROPS.
        wreq_crops_irr = pd.DataFrame()

        # The rj-indexed arrays have zeroes where j is not a crop.
        for lu in self.AGRICULTURAL_LANDUSES:
            if lu in self.LU_CROPS:
                wreq_crops_irr[lu] = self.AGEC_CROPS.sel(field="WR", lm="irr", lu=lu).to_series()
            else:
                wreq_crops_irr[lu] = 0.0

        # Add together as they have nans where not lvstk/crops
        self.WREQ_DRY_RJ = np.nan_to_num(wreq_lvstk_dry.to_numpy(dtype=np.float32))
        self.WREQ_IRR_RJ = np.nan_to_num(wreq_crops_irr.to_numpy(dtype=np.float32)) + np.nan_to_num(
            wreq_lvstk_irr.to_numpy(dtype=np.float32)
        )

        # Spatially explicit costs of a water licence per ML.
        self.WATER_LICENCE_PRICE = (
            xr.load_dataarray(os.path.join(settings.INPUT_DIR, "water_licence_price.nc")).isel(cell=self.MASK)
            .fillna(0)                                                          # NaN filled with 0 as before (see the A1 NaN audit)
            .assign_coords(cell=np.arange(self.NCELLS))
        )

        # Spatially explicit costs of water delivery per ML.
        self.WATER_DELIVERY_PRICE = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "water_delivery_price.nc")).isel(cell=self.MASK).fillna(0).assign_coords(cell=np.arange(self.NCELLS))
       

        # Water yields -- run off from a cell into catchment by deep-rooted, shallow-rooted, and natural land
        # The historical (1970-2000) yields and the deep-rooted proportion at the model cells, over cell
        water_yield_baselines = xr.load_dataset(os.path.join(settings.INPUT_DIR, "water_yield_baselines.nc")).isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        self.WATER_YIELD_HIST_DR = water_yield_baselines['WATER_YIELD_HIST_DR_ML_HA'].astype(np.float32)
        self.WATER_YIELD_HIST_SR = water_yield_baselines['WATER_YIELD_HIST_SR_ML_HA'].astype(np.float32)
        self.DEEP_ROOTED_PROPORTION = water_yield_baselines['DEEP_ROOTED_PROPORTION'].astype(np.float32)
        self.WATER_YIELD_HIST_NL = (
            water_yield_baselines['WATER_YIELD_HIST_DR_ML_HA'] * water_yield_baselines['DEEP_ROOTED_PROPORTION']
            + water_yield_baselines['WATER_YIELD_HIST_SR_ML_HA'] * (1 - water_yield_baselines['DEEP_ROOTED_PROPORTION'])
        ).astype(np.float32)

        # The yields of this SSP, DR / SR over (year, cell): not loaded here — the simulation reads its inputs year by year,
        # so get_water_yield reads a year when it is first asked for (a chunk holds one year of every cell)
        self.WATER_YIELD_SSP_FILE = os.path.join(settings.INPUT_DIR, f'water_yield_ssp{settings.SSP}_2010_2100_ml_ha.nc')
        self.WATER_YIELD_BY_YEAR = {}                                    # {year: Dataset of DR / SR}, filled by get_water_yield

        
        # Water yield from outside LUTO study area, historical (1970-2000), per region type: over region_id
        self.WATER_YIELD_OUTSIDE_LUTO_HIST = {}
        for region_type, region_file in [('River Region', 'water_region_rivreg.nc'), ('Drainage Division', 'water_region_draindiv.nc')]:
            with xr.open_dataset(os.path.join(settings.INPUT_DIR, region_file)) as water_region:
                self.WATER_YIELD_OUTSIDE_LUTO_HIST[region_type] = water_region['OUTSIDE_LUTO_HIST_ML'].load()
        
        # Water use for domestic and industrial sectors.
        water_use_domestic = pd.read_csv(os.path.join(settings.INPUT_DIR, "Water_Use_Domestic.csv")).query(f"REGION_TYPE == '{settings.WATER_REGION_DEF}'")
        self.WATER_USE_DOMESTIC = water_use_domestic.set_index('REGION_ID')['DOMESTIC_INDUSTRIAL_WATER_USE_ML'].to_dict()

        # Call the function to create watershed components
        self.VALID_WATERSHED_IDS = self.get_watershed_yield_components()

        # Get the water region index for each region
        self.WATER_REGION_INDEX_R = {k:(self.WATER_REGION_ID == k) for k in self.WATER_REGION_NAMES.keys()}

        # Place holder for Water Yield to avoid recalculating it every time.
        self.water_yield_regions_BASE_YR = None
        
        # Water yield targets for each region
        self.WATER_YIELD_TARGETS, self.WATER_RELAXED_REGION_RAW_TARGETS = ag_water.get_water_target_inside_LUTO_by_CCI(self)

        ###############################################################
        # Carbon sequestration by natural lands.
        ###############################################################
        print("├── Loading carbon sequestration by natural lands data", flush=True)

        '''
        ['NATURAL_LAND_AGB_TCO2_HA']
            CO2 in aboveground living biomass in natural land i.e., the part impacted by livestock 
        ['NATURAL_LAND_AGB_DEBRIS_TCO2_HA']
            CO2 in aboveground living biomass and debris in natural land i.e., the part impacted by fire
        ['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA']
            CO2 in aboveground living biomass and debris and soil in natural land i.e., the part impacted by land clearance
        '''
    
        # Load the natural land carbon data.
        nat_land_CO2 = xr.load_dataset(os.path.join(settings.INPUT_DIR, "natural_land_t_co2_ha.nc")).isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        
        # Get the carbon stock of unallowcated natural land
        self.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR = (
            nat_land_CO2['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA'] - (nat_land_CO2['NATURAL_LAND_AGB_DEBRIS_TCO2_HA'] * (100 - fire_risk) / 100)  # everyting minus the fire DAMAGE
        ) / settings.CARBON_EFFECTS_WINDOW
        
        
        ###############################################################
        # Calculate base year production 
        ###############################################################

        self.AG_MAN_L_MRJ_DICT = get_base_am_vars(self.NCELLS, self.NLMS, self.N_AG_LUS)
        self.add_ag_man_dvars(self.YR_CAL_BASE, self.AG_MAN_L_MRJ_DICT)
        
        print("├── Calculating base year productivity", flush=True)
        
        (
            self.prod_base_yr_potential_ag_mrp,
            self.prod_base_yr_potential_non_ag_rp,
            self.prod_base_yr_potential_am_amrp
        ) = self.get_potential_production_lyr(self.YR_CAL_BASE)

        (
            self.prod_base_yr_actual_ag_mrc,
            self.prod_base_yr_actual_non_ag_rc,
            self.prod_base_yr_actual_am_amrc
        ) = self.get_actual_production_lyr(self.YR_CAL_BASE, threshold=0.0)   # every share counted: an accounting total, the same computation as simulation.store_solution's for a target year (the 0.01 default is the maps' clean-up)

        yr_cal_base_prod_data = (
            self.prod_base_yr_actual_ag_mrc.sum(['cell','lm'])
            + self.prod_base_yr_actual_non_ag_rc.sum(['cell'])
            + self.prod_base_yr_actual_am_amrc.sum(['cell', 'am', 'lm'])
        ).compute().values
        
        self.add_production_data(self.YR_CAL_BASE, "Production", yr_cal_base_prod_data)

        
        # Place holders for base year values
        self.BASE_YR_economic_value = None                      # filled by `input_data.get_BASE_YR_economic_value`
        self.BASE_YR_production_t = yr_cal_base_prod_data       
        self.BASE_YR_GHG_t = None                               # filled by `input_data.get_BASE_YR_GHG_t`
        self.BASE_YR_water_ML = None                            # filled by `input_data.get_BASE_YR_water_ML`
        self.BASE_YR_overall_bio_value = None                   # filled by `input_data.get_BASE_YR_bio_quality_value`
        self.BASE_YR_GBF2_score = None                          # filled by `input_data.get_BASE_YR_GBF2_score`

        ###############################################################
        # Demand data.
        ###############################################################
        print("├── Loading demand data", flush=True)
        
        # Load demand multiplier data.
        #
        # These multipliers are the commodity demand trajectory that BELONGS to a carbon pathway, so
        # `GHG_EMISSIONS_LIMITS = 'off'` -- which maps to None in GHG_TARGETS_DICT and has no sheet of
        # its own -- means no pathway-driven demand shift at all: every multiplier is 1.0. The
        # workbook is still read in that case, but only to borrow its year index and commodity
        # columns, so the frame keeps the same shape however the carbon constraint is set.
        ghg_pathway = settings.GHG_TARGETS_DICT[settings.GHG_EMISSIONS_LIMITS]
        AusTIME_multipliers = pd.read_excel(
                f'{settings.INPUT_DIR}/AusTIMES_demand_multiplier.xlsx',
                sheet_name=(ghg_pathway or settings.GHG_TARGETS_DICT['low']).split(' ')[0] + ' Demand',
                index_col=0,
            ).T.reset_index(drop=True
            ).rename(columns={
                'Sorghum ':'winter cereals',    # the majority of winter cereals is Sorghum in Australia, so we map it this way
                'Canola':'winter oilseeds',     # the majority of winter oilseeds is Canola in Australia, so we map it this way
                'Sugar': 'sugar'
            }).drop(columns=['Cottonseed']      # LUTO do not have cottonseed in our model
            ).astype({'Year': int}
            ).set_index('Year')

        if ghg_pathway is None:                 # GHG off -- no carbon pathway, so no demand shift
            AusTIME_multipliers.loc[:, :] = 1.0

        # The AusTIMES multipliers on the demand of each (Commodity, year); 1 where AusTIMES gives none
        demand_multipliers = xr.DataArray(
            np.ones((len(self.COMMODITIES), AusTIME_multipliers.index.max() + 1 - 2010)),
            dims=['Commodity', 'year'],
            coords={'Commodity': self.COMMODITIES, 'year': np.arange(2010, AusTIME_multipliers.index.max() + 1)},
        )
        demand_multipliers.loc[
            {
                'Commodity': list(AusTIME_multipliers.columns), 
                'year': list(AusTIME_multipliers.index)
            }] = AusTIME_multipliers.to_numpy().T

        # Cache for elasticity multipliers (keyed by yr_cal)
        self.DEMAND_ELASTICITY_MUL = {}

        # Load demand data (actual production (tonnes, ML) by commodity) - from demand model:
        # the running scenario's chunks only; one variable per quantity over (Commodity, year)
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, 'demand_projections.nc')) as demand_ds:
            self.DEMAND_DATA = (
                demand_ds
                .sel(
                    SCENARIO=settings.SCENARIO, 
                    DIET_DOM=settings.DIET_DOM, 
                    DIET_GLOB=settings.DIET_GLOB,
                    CONVERGENCE=settings.CONVERGENCE, 
                    IMPORT_TREND=settings.IMPORT_TREND, 
                    WASTE=settings.WASTE,
                    FEED_EFFICIENCY=settings.FEED_EFFICIENCY)
                .reset_coords(drop=True)
                .load()
            )

        # Convert eggs from count to tonnes
        self.DEMAND_DATA = xr.where(
            self.DEMAND_DATA['Commodity'] == 'eggs', 
            self.DEMAND_DATA * settings.EGGS_AVG_WEIGHT / 1000 / 1000, 
            self.DEMAND_DATA
        )

        # The off-land commodities' production, and the on-land commodities' (the demand the solver meets)
        commodities = self.DEMAND_DATA['Commodity'].values
        self.DEMAND_OFFLAND = self.DEMAND_DATA['PRODUCTION'].sel(Commodity=[c for c in commodities if c in settings.OFF_LAND_COMMODITIES])
        self.DEMAND_C = self.DEMAND_DATA['PRODUCTION'].sel(Commodity=[c for c in commodities if c not in settings.OFF_LAND_COMMODITIES])

        if settings.APPLY_DEMAND_MULTIPLIERS:
            print(f"│   ├── Years before applying demand multipliers: {self.DEMAND_C['year'].min().item()} - {self.DEMAND_C['year'].max().item()}", flush=True)
            self.DEMAND_C = self.DEMAND_C * demand_multipliers       # aligned by label; the inner join keeps the years both cover
            print(f"│   └── Years after applying demand multipliers: {self.DEMAND_C['year'].min().item()} - {self.DEMAND_C['year'].max().item()}", flush=True)

        # Demand cannot be negative. Some projections are unclamped extrapolations that cross zero
        # (SSP5 'other non-cereal crops' does, from 2041), and a negative target becomes an equality
        # asking a sum of non-negative production terms to equal a negative number — infeasible by
        # construction, with no solver setting able to absorb it.
        negative = self.DEMAND_C < 0
        if negative.any():
            for commodity in negative['Commodity'].values[negative.any('year').values]:
                bad = self.DEMAND_C.sel(Commodity=commodity).where(negative.sel(Commodity=commodity), drop=True)
                print(f"│   ⚠ WARNING: negative demand clamped to 0 for '{commodity}' — "
                      f"{bad.sizes['year']} year(s) from {bad['year'].min().item()}, min {bad.min().item():,.0f}", flush=True)
            self.DEMAND_C = self.DEMAND_C.clip(min=0)

        # The demand the solver reads (row_inputs, the demand rows' rhs), float32 as it has always been
        self.DEMAND_C = self.DEMAND_C.astype(np.float32)

        # Price elasticity data, over Commodity
        demand_supply_elasticity = pd.read_csv(f'{settings.INPUT_DIR}/demand_elasticity.csv').set_index('Commodity')
        self.elasticity_demand = xr.DataArray(demand_supply_elasticity['Demand Elasticity(ED)'] * -1)  # ED is subtracted from supply becase demand curve are given as negative numbers
        self.elasticity_supply = xr.DataArray(demand_supply_elasticity['Supply Elasticity (Es)'])

   


        ###############################################################
        # Carbon emissions from off-land commodities.
        ###############################################################
        print("├── Loading off-land commodities' carbon emissions data", flush=True)

        # Read the greenhouse gas intensity data, one row per (commodity, emission source)
        off_land_ghg_intensity = pd.read_csv(f'{settings.INPUT_DIR}/agGHG_lvstk_off_land.csv')
        # Split the Emission Source column into two columns
        off_land_ghg_intensity[['Emission Type', 'Emission Source']] = off_land_ghg_intensity['Emission Source'].str.extract(r'^(.*?)\s*\((.*?)\)')
        intensity = xr.DataArray(
            off_land_ghg_intensity['Emission Intensity [ kg CO2eq / kg ]'].to_numpy(),
            dims=['emission'],
            coords={
                'Commodity': ('emission', off_land_ghg_intensity['COMMODITY'].to_numpy()),
                'Emission Type': ('emission', off_land_ghg_intensity['Emission Type'].to_numpy()),
                'Emission Source': ('emission', off_land_ghg_intensity['Emission Source'].to_numpy()),
            },
        )

        # The emissions of every (emission, year): the off-land demand of its commodity x its intensity (tCO2e)
        self.OFF_LAND_GHG_EMISSION = self.DEMAND_OFFLAND.sel(Commodity=intensity['Commodity']) * intensity

        # Their total per year: what the GHG cap must leave room for
        self.OFF_LAND_GHG_EMISSION_C = self.OFF_LAND_GHG_EMISSION.sum('emission')

        # Read the carbon price per tonne over the years (indexed by the relevant year)
        if settings.CARBON_PRICES_FIELD == 'CONSTANT':
            self.CARBON_PRICES = {yr: settings.CARBON_PRICE_COSTANT for yr in range(2010, 2101)}
        else:
            carbon_price_sheet = settings.CARBON_PRICES_FIELD or "Default"
            carbon_price_usecols = "A,B"
            carbon_price_col_names = ["Year", "Carbon_price_$_tCO2e"]
            carbon_price_sheet_index_col = "Year" # if carbon_price_sheet != "Default" else 0
            carbon_price_sheet_header = 0         # if carbon_price_sheet != "Default" else None

            self.CARBON_PRICES: dict[int, float] = pd.read_excel(
                os.path.join(settings.INPUT_DIR, 'carbon_prices.xlsx'),
                sheet_name=carbon_price_sheet,
                usecols=carbon_price_usecols,
                names=carbon_price_col_names,
                header=carbon_price_sheet_header,
                index_col=carbon_price_sheet_index_col,
            )["Carbon_price_$_tCO2e"].to_dict()
            


        ###############################################################
        # GHG targets data.
        ###############################################################
        print("├── Loading GHG targets data", flush=True)
        if settings.GHG_EMISSIONS_LIMITS != 'off':
            self.GHG_TARGETS = pd.read_excel(
                os.path.join(settings.INPUT_DIR, "GHG_targets.xlsx"), sheet_name="Data", index_col="YEAR"
            )
            self.GHG_TARGETS = self.GHG_TARGETS[settings.GHG_TARGETS_DICT[settings.GHG_EMISSIONS_LIMITS]].to_dict()



        ###############################################################
        # Savanna burning data.
        ###############################################################
        print("├── Loading savanna burning data", flush=True)

        # Read in the dataframe
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, 'cell_savanna_burning.nc')) as savburn:
            savburn = savburn[['ELIGIBLE_AREA', 'AEA_TOTAL_TCO2E_HA']].load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))

        # Load the layers as DataArrays over the model cells
        self.SAVBURN_ELIGIBLE =  savburn['ELIGIBLE_AREA']                    # 1 = areas eligible for early dry season savanna burning under the ERF, 0 = ineligible          
        self.SAVBURN_TOTAL_TCO2E_HA = savburn['AEA_TOTAL_TCO2E_HA']
        
        # # Avoided emissions from savanna burning
        # self.SAVBURN_AVEM_CH4_TCO2E_HA = savburn['SAV_AVEM_CH4_TCO2E_HA']  # Avoided emissions - methane
        # self.SAVBURN_AVEM_N2O_TCO2E_HA = savburn['SAV_AVEM_N2O_TCO2E_HA']  # Avoided emissions - nitrous oxide
        # self.SAVBURN_SEQ_CO2_TCO2E_HA = savburn['SAV_SEQ_CO2_TCO2E_HA']    # Additional carbon sequestration - carbon dioxide

        # Cost per hectare in dollars from settings
        self.SAVBURN_COST_HA = settings.SAVBURN_COST_HA_YR



        ###############################################################
        # Biodiversity priority conservation data. (GBF Target 2)
        ###############################################################
        
        print("├── Loading biodiversity data", flush=True)
        """
        Kunming-Montreal Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
        Ensure that by 2030 at least 30 per cent of areas of degraded terrestrial, inland water, and coastal and marine ecosystems are under effective restoration,
        in order to enhance biodiversity and ecosystem functions and services, ecological integrity and connectivity.
        """

        
        biodiv_contribution_lookup = pd.read_csv(os.path.join(settings.INPUT_DIR, 'bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv'))
        

        # ------------- Biodiversity priority scores for maximising overall biodiversity conservation in Australia ----------------------------
        
        # The priority of this SSP (ssp, cell) and the two connectivity layers (cell) at the model cells
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, 'bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.nc')) as bio_priority:
            biodiv_raw = bio_priority.sel(ssp=[settings.SSP]).load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        
        # Get the biodiversity quality score (unweighted; connectivity is applied further below)
        bio_quality_raw, performance_sheet = self.load_bio_quality_layer(settings.BIO_QUALITY_LAYER, biodiv_ds=biodiv_raw)

        
        # Get connectivity score
        match settings.CONNECTIVITY_SOURCE:
            case 'NCI':
                connectivity_score = biodiv_raw['DCCEEW_NCI'].values.astype(np.float32)
                connectivity_score = np.interp( biodiv_raw['DCCEEW_NCI'], (connectivity_score.min(), connectivity_score.max()), (settings.CONNECTIVITY_LB, 1)).astype('float32')
            case 'DWI':
                connectivity_score = biodiv_raw['NATURAL_AREA_CONNECTIVITY'].values.astype(np.float32)
                connectivity_score = np.interp(connectivity_score, (connectivity_score.min(), connectivity_score.max()), (1, settings.CONNECTIVITY_LB)).astype('float32')
            case 'NONE':
                connectivity_score = np.ones(self.NCELLS, dtype=np.float32)
            case _:
                raise ValueError(f"Invalid connectivity source: {settings.CONNECTIVITY_SOURCE}, must be 'NCI', 'DWI' or 'NONE'")
        
        self.CONNECTIVITY_SCORE = connectivity_score  

        # Get the HCAS contribution scale (0-1)
        match settings.HCAS_CONTRIBUTION_PERCENTILE:
            case '10' | '25' | '50' | '75' | '90':
                bio_HCAS_contribution_lookup = biodiv_contribution_lookup.set_index('lu')[f'PERCENTILE_{settings.HCAS_CONTRIBUTION_PERCENTILE}'].to_dict()         # Get the biodiversity degradation score at specified percentile (pd.DataFrame)
                unallow_nat_scale = bio_HCAS_contribution_lookup[self.DESC2AGLU['Unallocated - natural land']]                                          # Get the biodiversity degradation score for unallocated natural land (float)
                bio_HCAS_contribution_lookup = {int(k): v * (1 / unallow_nat_scale) for k, v in bio_HCAS_contribution_lookup.items()}                   # Normalise the biodiversity degradation score to the unallocated natural land score
            case 'CSV_DEFINED':
                bio_HCAS_contribution_lookup = biodiv_contribution_lookup.set_index('lu')['CSV_DEFINED'].to_dict()
            case 'AG_UNIFORM':
                bio_HCAS_contribution_lookup = biodiv_contribution_lookup.set_index('lu')['AG_UNIFORM'].to_dict()
                for _lu in self.LU_MODIFIED_LAND:
                    bio_HCAS_contribution_lookup[_lu] = float(settings.HCAS_AG_UNIFORM_CONTRIBUTION)
                print(f"│   ├── AG_UNIFORM habitat contribution: modified/cropped ag land = {settings.HCAS_AG_UNIFORM_CONTRIBUTION}", flush=True)
            case _:
                print(f"│   ⚠ WARNING: Invalid habitat condition source: {settings.HCAS_CONTRIBUTION_PERCENTILE}, must be one of [10, 25, 50, 75, 90], 'CSV_DEFINED', or 'AG_UNIFORM'", flush=True)
        
        self.BIO_HABITAT_CONTRIBUTION_LOOK_UP = {j: round(x, settings.ROUND_DECIMALS) for j, x in bio_HCAS_contribution_lookup.items()}                 # Round to the specified decimal places to avoid numerical issues in the GUROBI solver
        
        
        # Get the biodiversity quantity score for each land use in each cell
        self.BIO_QUALITY_RAW = bio_quality_raw * self.CONNECTIVITY_SCORE
        self.BIO_QUALITY_LDS = np.where(
            self.SAVBURN_ELIGIBLE.values,
            self.BIO_QUALITY_RAW  - (self.BIO_QUALITY_RAW * (1 - settings.BIO_CONTRIBUTION_LDS)),
            self.BIO_QUALITY_RAW
        )
        
  
        # Conservation performance curve 
        conservation_performance_curve = pd.read_excel(
            os.path.join(settings.INPUT_DIR, 'Biodiversity_conserve_performance.xlsx'),
            sheet_name=performance_sheet
        ).set_index('AREA_COVERAGE_PERCENT')['PRIORITY_RANK'].to_dict()


        # ------------------ Habitat condition impacts for habitat conservation (GBF2) in 'priority degraded areas' regions ---------------
        if settings.GBF2_TARGET != 'off':

            # Get the mask of 'priority degraded areas' for habitat conservation
            self.BIO_GBF2_MASK = bio_quality_raw >= conservation_performance_curve[settings.GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT]
            self.BIO_GBF2_MASK_LDS = np.where(
                self.SAVBURN_ELIGIBLE.values,
                self.BIO_GBF2_MASK  - (self.BIO_GBF2_MASK * (1 - settings.BIO_CONTRIBUTION_LDS)),
                self.BIO_GBF2_MASK
            )

            self.BIO_GBF2_BASE_YR = (
                np.einsum(
                    'j,mrj,r,r->r',
                    np.array(list(self.BIO_HABITAT_CONTRIBUTION_LOOK_UP.values())),
                    self.AG_L_MRJ,      # lumap in proportion representation if resfactored
                    self.BIO_GBF2_MASK,
                    self.REAL_AREA.values
                ) - (
                    self.SAVBURN_ELIGIBLE.values
                    * self.BIO_GBF2_MASK
                    * (1 - settings.BIO_CONTRIBUTION_LDS)
                    * self.REAL_AREA.values
                    * self.AG_MASK_PROPORTION_R
                )
            )


        # Renewable energy exclusion masks using biodiversity quality and independent cut values
        if any(settings.RENEWABLES_OPTIONS.values()) and settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
            self.RENEWABLE_GBF2_MASK_SOLAR = bio_quality_raw >= conservation_performance_curve[settings.RENEWABLE_GBF2_CUT_SOLAR]
            self.RENEWABLE_GBF2_MASK_WIND = bio_quality_raw >= conservation_performance_curve[settings.RENEWABLE_GBF2_CUT_WIND]

        # Renewable energy exclusion masks using EPBC MNES prioritization layer
        if any(settings.RENEWABLES_OPTIONS.values()) and settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
            mnes_rank_raw = xr.load_dataarray(os.path.join(settings.INPUT_DIR, 'renewable_QLD_EPBC_MNES_prioritization.nc')).values[self.MASK]
            mnes_performance = pd.read_csv(os.path.join(settings.INPUT_DIR, 'renewable_QLD_EPBC_MNES_prioritization_performance.csv')).set_index('AREA_COVERAGE_PERCENT')['PRIORITY_RANK'].to_dict()
            self.RENEWABLE_MNES_MASK_SOLAR = mnes_rank_raw >= mnes_performance[settings.RENEWABLE_EPBC_MNES_CUT_SOLAR]
            self.RENEWABLE_MNES_MASK_WIND = mnes_rank_raw >= mnes_performance[settings.RENEWABLE_EPBC_MNES_CUT_WIND]


        ###############################################################
        # GBF3 biodiversity data. (NVIS vegetation groups)
        ###############################################################
        
        # Always load the full NVIS vegetation data into memory for writing all vegetation group scores.
        self.GBF3_NVIS_LAYERS_ALL = self.get_NVIS_sparse_array()
        
        if settings.GBF3_NVIS_TARGET != 'off':
            print(f"│   ├── Loading GBF3 vegetation data (NVIS)", flush=True)
            print(f"│   │   ├── NRM region mode: {settings.GBF3_NVIS_REGION_MODE} | selected regions: {settings.GBF3_NVIS_SEL_REGION_TARGETS}", flush=True)

            nvis_targets_df = self.get_NVIS_targets_df(verbose=True)
            self.BIO_GBF3_NVIS_SEL = list(zip(nvis_targets_df['region'], nvis_targets_df['group']))

            # Resfactor each selected group inline (replaces get_NVIS_resfactord_array).
            sel_group = nvis_targets_df['group'].unique()
            nvis_layers_arr = xr.DataArray(
                np.array(
                    [self.get_resfactored_average_fraction(
                        self.GBF3_NVIS_LAYERS_ALL.sel(group=g).data.todense().astype(np.float32),
                        use_valid_cell_count=False,
                     ) for g in sel_group],
                    dtype=np.float32,
                ),
                dims=['group', 'cell'],
                coords={'group': sel_group, 'cell': np.arange(self.NCELLS)},
            )
            self.GBF3_NVIS_LAYERS_LDS = xr.where(
                self.SAVBURN_ELIGIBLE.values[None, :],
                nvis_layers_arr * settings.BIO_CONTRIBUTION_LDS,
                nvis_layers_arr,
            ).astype(np.float32)

            # IBRA_REG mode: the bioregion of every cell, so a (bioregion, group) target is masked to its own cells
            # (sampled at the cell like the NRM / state labels). Code = position in REGION_IBRA_NAMES, -1 = no bioregion.
            if settings.GBF3_NVIS_REGION_MODE == 'IBRA_REG':
                ibra = xr.load_dataarray(os.path.join(settings.INPUT_DIR, 'bio_GBF3_IBRA_Regions.nc'))                # bool (region, cell): read whole, then masked (a lazy boolean index on the compressed variable crawls)
                self.REGION_IBRA_NAMES = ibra['region'].values.tolist()
                in_region_vr = ibra.values[:, self.MASK]
                self.REGION_IBRA_CODE = np.where(in_region_vr.any(axis=0), in_region_vr.argmax(axis=0), -1).astype(np.int32)



        ##########################################################################
        #  Biodiversity environmental significance (GBF4)                        #
        ##########################################################################
        
        # Load full SNES/ECNES into memory for writing all species scores
        self.GBF4_SNES_LAYERS_ALL = self.get_SNES_sparse_array()
        self.GBF4_ECNES_LAYERS_ALL = self.get_ECNES_sparse_array()

        self.GBF4_SNES_META     = self.get_SNES_targets_df(include_all=False)
        self.BIO_GBF4_SNES_SEL  = [tuple(r) for r in self.GBF4_SNES_META[['region', 'SCIENTIFIC_NAME', 'presence']].values.tolist()]  # list of (region, species, presence) tuples for selection; presence is 'LIKELY' or 'LIKELY_AND_MAYBE'

        self.GBF4_ECNES_META    = self.get_ECNES_targets_df(include_all=False)
        self.BIO_GBF4_ECNES_SEL = [tuple(r) for r in self.GBF4_ECNES_META[['region', 'COMMUNITY', 'presence']].values.tolist()]

        if settings.GBF4_TARGET_SNES != 'off':

            print("│   ├── Loading environmental significance data (SNES)", flush=True)

            sel_sp_pres_pairs = [tuple(r) for r in self.GBF4_SNES_META[['SCIENTIFIC_NAME', 'presence']].drop_duplicates().values.tolist()]  # list of unique (species, presence) pairs for selection
            snes_layer_arr = xr.DataArray(
                np.array([
                    self.get_resfactored_average_fraction(
                        self.GBF4_SNES_LAYERS_ALL.sel(
                            species=sp, presence=('LIKELY' if pres == 'LIKELY' else 'LIKELY_AND_MAYBE')
                        ).data.todense().astype(np.float32),
                        use_valid_cell_count=False,
                    )
                    for sp, pres in sel_sp_pres_pairs
                ], dtype=np.float32),
                dims=['layer', 'cell'],
                coords={
                    'layer': pd.MultiIndex.from_tuples(sel_sp_pres_pairs, names=['species', 'presence']),
                    'cell': np.arange(self.NCELLS)
                }
            )
            
            self.GBF4_SNES_LAYERS_LDS = xr.where(
                self.SAVBURN_ELIGIBLE.values[None, :],
                snes_layer_arr * settings.BIO_CONTRIBUTION_LDS,
                snes_layer_arr,
            ).astype(np.float32)

        
        if settings.GBF4_TARGET_ECNES != 'off':
            print("│   ├── Loading environmental significance data (ECNES)", flush=True)

            sel_comm_pres_pairs              = [tuple(r) for r in self.GBF4_ECNES_META[['COMMUNITY', 'presence']].drop_duplicates().values.tolist()]
            ecnes_layer_arr = xr.DataArray(
                np.array([
                    self.get_resfactored_average_fraction(
                        self.GBF4_ECNES_LAYERS_ALL.sel(
                            species=comm, presence=('LIKELY' if pres == 'LIKELY' else 'LIKELY_AND_MAYBE')
                        ).data.todense().astype(np.float32),
                        use_valid_cell_count=False,
                    )
                    for comm, pres in sel_comm_pres_pairs
                ], dtype=np.float32),
                dims=['layer', 'cell'],
                coords={
                    'layer': pd.MultiIndex.from_tuples(sel_comm_pres_pairs, names=['species', 'presence']),
                    'cell': np.arange(self.NCELLS),
                },
            )
            self.GBF4_ECNES_LAYERS_LDS = xr.where(
                self.SAVBURN_ELIGIBLE.values[None, :],
                ecnes_layer_arr * settings.BIO_CONTRIBUTION_LDS,
                ecnes_layer_arr,
            ).astype(np.float32)

        
        
        ##########################################################################
        # Biodiversity species suitability under climate change (GBF8)            #
        ##########################################################################
        
        if settings.GBF8_TARGET != 'off':
            
            print("│   ├── Loading Species suitability data", flush=True)
            
            # Read in the species data from Carla Archibald (noted as GBF-8)
            BIO_GBF8_SPECIES_raw = xr.open_dataset(f'{settings.INPUT_DIR}/bio_GBF8_ssp{settings.SSP}_EnviroSuit.nc', chunks={'year':1,'species':1})['data']        
            bio_GBF8_baseline_score = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_SCORES.csv').sort_values(by='species', ascending=True)
            bio_GBF8_target_percent = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_TARGET.csv').sort_values(by='species', ascending=True)

            # The input CSV carries hand-filled 'USER_DEFINED_TARGET_PERCENT_{yr}' columns;
            # map them onto the 'TARGET_LEVEL_{yr}' columns the rest of the class consumes.
            for _yr in (2030, 2050, 2100):
                if f'TARGET_LEVEL_{_yr}' not in bio_GBF8_target_percent.columns:
                    bio_GBF8_target_percent[f'TARGET_LEVEL_{_yr}'] = bio_GBF8_target_percent.get(f'USER_DEFINED_TARGET_PERCENT_{_yr}')

            # 'low'/'medium'/'high' — uniform preset targets for ALL species BEFORE selection,
            # so the selection below keeps every species (preset > 0) regardless of the CSV.
            # 'CSV_DEFINED' keeps the CSV targets; only species with all three year targets
            # defined and > 0 are selected.
            if settings.GBF8_TARGET in ('medium', 'high'):
                for _yr, _pct in settings.GBF8_TARGETS_DICT[settings.GBF8_TARGET].items():
                    bio_GBF8_target_percent[f'TARGET_LEVEL_{_yr}'] = float(_pct)
                print(f"│   │   ├── '{settings.GBF8_TARGET}' level targets applied to all GBF8 species", flush=True)

            self.BIO_GBF8_SEL_SPECIES = [row['species'] for _,row in bio_GBF8_target_percent.iterrows()
                                        if all([row.get('TARGET_LEVEL_2030', 0)>0,
                                                row.get('TARGET_LEVEL_2050', 0)>0,
                                                row.get('TARGET_LEVEL_2100', 0)>0])]
            # 'AUSTRALIA' is the canonical nationwide label — the solver's region-mask bypass
            # and the rescale helper both match region == 'AUSTRALIA' (same as GBF3/GBF4).
            self.BIO_GBF8_SEL = [('AUSTRALIA', sp) for sp in self.BIO_GBF8_SEL_SPECIES]

            # .isin, not .query(f'species in {list}'): interpolating the ~10.6k-name list into a
            # query string is a ~700 KB expression the parser may choke on at all-species levels
            self.BIO_GBF8_OUTSDIE_LUTO_SCORE_SPECIES = bio_GBF8_baseline_score[bio_GBF8_baseline_score['species'].isin(set(self.BIO_GBF8_SEL_SPECIES))][['species', 'year', f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{settings.SSP}']]
            self.BIO_GBF8_OUTSDIE_LUTO_SCORE_GROUPS = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_SCORES_group.csv')[['group', 'year', f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{settings.SSP}']]
            
            self.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES = bio_GBF8_target_percent[bio_GBF8_target_percent['species'].isin(set(self.BIO_GBF8_SEL_SPECIES))]
            self.BIO_GBF8_BASELINE_SCORE_GROUPS = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_TARGET_group.csv')
            
            self.N_GBF8_SPECIES = len(self.BIO_GBF8_SEL_SPECIES)
            if self.BIO_GBF8_SEL_SPECIES:
                # Kept LAZY: get_GBF8_bio_layers_by_yr reads from the _cache spatial-match file
                # (see match_GBF8_bio_layers), so the raw raster stack is never needed in RAM —
                # a .compute() here would hold ~42 GB when all ~10.6k species are selected.
                self.BIO_GBF8_SPECIES_LAYER = BIO_GBF8_SPECIES_raw.sel(species=self.BIO_GBF8_SEL_SPECIES)
            else:
                print("│   │   ⚠ WARNING: No GBF8 species selected, proceeding with empty selection.", flush=True)
                self.BIO_GBF8_SPECIES_LAYER = BIO_GBF8_SPECIES_raw.isel(species=[])
            
            self.BIO_GBF8_GROUPS_LAYER = xr.load_dataset(f'{settings.INPUT_DIR}/bio_GBF8_ssp{settings.SSP}_EnviroSuit_group.nc')['data']
            self.BIO_GBF8_GROUPS_NAMES = [i.capitalize() for i in self.BIO_GBF8_GROUPS_LAYER['group'].values]
        

        ###############################################################
        # BECCS data.
        ###############################################################
        print("└── Loading BECCS data", flush=True)

        # Load dataframe
        with xr.open_dataset(os.path.join(settings.INPUT_DIR, 'cell_BECCS.nc')) as beccs:
            beccs = beccs[['BECCS_COSTS_AUD_HA_YR', 'BECCS_REV_AUD_HA_YR', 'BECCS_TCO2E_HA_YR', 'BECCS_MWH_HA_YR']].load().isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))

        # Capture as DataArrays over the model cells
        self.BECCS_COSTS_AUD_HA_YR = beccs['BECCS_COSTS_AUD_HA_YR']
        self.BECCS_REV_AUD_HA_YR = beccs['BECCS_REV_AUD_HA_YR']
        self.BECCS_TCO2E_HA_YR = beccs['BECCS_TCO2E_HA_YR']
        self.BECCS_MWH_HA_YR = beccs['BECCS_MWH_HA_YR']


    @property
    def AGMAN2LU(self) -> dict[str, list[int]]:
        """{ag-management option: [land-use codes]} for the ENABLED options, in settings order
        (the code view of AG_MAN_LU_DESC). A property, so a Data pickled before it existed has it too."""
        return {am: [self.DESC2AGLU[lu] for lu in lus] for am, lus in self.AG_MAN_LU_DESC.items()}

    @property
    def AGMAN2IDX(self) -> dict[str, int]:
        """{ag-management option: its fixed index} for EVERY option, enabled or not: its position in settings.AG_MANAGEMENTS.
        Switching an option off never moves another one, as a disabled non-ag land use keeps its k."""
        return {am: idx for idx, am in enumerate(settings.AG_MANAGEMENTS)}

    def read_sparse_layer(self, file_name: str) -> xr.DataArray:
        """A sparse biodiversity layer (GBF3 NVIS, GBF4 SNES / ECNES) as a sparse.COO-backed DataArray over the full-res
        cells: the file holds the COO's parts (COO_COORDS, COO_DATA), its labels as coordinates and its shape / dims /
        fill value as attributes; .sel / .isel select by label / position, .data.todense() materialises a slice."""
        layer = xr.load_dataset(os.path.join(settings.INPUT_DIR, file_name))
        coo = sparse.COO(
            layer['COO_COORDS'].values, layer['COO_DATA'].values, shape=tuple(layer.attrs['shape']),
            sorted=True, has_duplicates=False, fill_value=layer.attrs['fill_value'],
        )
        dims = list(layer.attrs['dims'])
        labels = {dim: layer[dim].values.astype(str) for dim in dims[:-1]}
        return xr.DataArray(coo, dims=dims, coords={**labels, 'cell': np.arange(coo.shape[-1])})

    def get_NVIS_sparse_array(self) -> xr.DataArray:
        """
        Load the full NVIS sparse array from disk and return as a sparse-backed xr.DataArray.

        Values are stored as fraction [0, 1] (divided by 100 from the uint8 [0-100] source).
        The underlying data is a sparse.COO; use .sel(group=g).data.todense() to materialise
        a single group slice without expanding the full array.

        Returns
        -------
        xr.DataArray
            shape (group, cell), sparse-backed, values in [0, 1].
        """
        layer = self.read_sparse_layer(f"bio_GBF3_{settings.GBF3_NVIS_TARGET_CLASS}_sparse.nc")
        return layer.copy(data=layer.data / np.float32(100))


    def get_NVIS_targets_df(self, verbose: bool = False) -> pd.DataFrame:
        """
        Load NVIS targets from CSV and return active constrained (region, group) pairs.

        Always reads from CSV. Stateless — no instance attributes read or written.
        Applies cascading filters: target check → dict overwrite → TARGET_CLASS
        → REGION_MODE → SEL_REGION_TARGETS → MIN_AREA_HA.

        Returns an empty DataFrame when GBF3_NVIS_TARGET is 'off'.

        Used by __init__ (BIO_GBF3_NVIS_SEL) and the solver via
        get_GBF3_NVIS_limit_score_inside_LUTO_by_yr.

        Returns:
            DataFrame with columns including 'region', 'group',
            'TARGET_LEVEL_2030', 'TARGET_LEVEL_2050', 'TARGET_LEVEL_2100'.
        """
        # Step 1: check GBF3_NVIS_TARGET
        if settings.GBF3_NVIS_TARGET == 'off':
            return pd.DataFrame(columns=['region', 'group', 'TARGET_LEVEL_2030', 'TARGET_LEVEL_2050', 'TARGET_LEVEL_2100'])

        if settings.GBF3_NVIS_REGION_MODE not in ('AUSTRALIA', 'NRM', 'IBRA_REG'):
            raise ValueError(
                f"Invalid GBF3_NVIS_REGION_MODE: '{settings.GBF3_NVIS_REGION_MODE}', "
                "must be 'AUSTRALIA', 'NRM', or 'IBRA_REG'"
            )

        # Steps 3+4: filter by GBF3_NVIS_TARGET_CLASS and GBF3_NVIS_REGION_MODE
        df = (
            pd.read_csv(settings.INPUT_DIR + "/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv")
            .rename(columns={'species': 'group'})
            .query(f"sheet_name == '{settings.GBF3_NVIS_TARGET_CLASS}' and region_level == '{settings.GBF3_NVIS_REGION_MODE}' and resfactor == {settings.RESFACTOR}")
            .sort_values(['group', 'region'])
            .reset_index(drop=True)
        )

        # AUSTRALIA mode: collapse all regional rows to a single 'AUSTRALIA' label
        # (solver checks region == 'AUSTRALIA' to bypass NRM masking).
        if settings.GBF3_NVIS_REGION_MODE == 'AUSTRALIA':
            df = df.assign(region='AUSTRALIA')

        # Step 2: apply GBF3_TARGETS_DICT if not None (i.e. not CSV_DEFINED / dict)
        target_dict = settings.GBF3_TARGETS_DICT[settings.GBF3_NVIS_TARGET]
        if settings.GBF3_NVIS_TARGET == 'SPECIFIED':
            # region-specific uniform levels from GBF3_NVIS_SEL_REGION_TARGETS = {region: {year: pct}};
            # every year must be given (a missing one raises)
            df = df[df['region'].isin(settings.GBF3_NVIS_SEL_REGION_TARGETS)].reset_index(drop=True)
            for reg, yr_dict in settings.GBF3_NVIS_SEL_REGION_TARGETS.items():
                for yr in (2030, 2050, 2100):
                    df.loc[df['region'] == reg, f'TARGET_LEVEL_{yr}'] = float(yr_dict[yr])
            print(f"│   │   ├── NVIS SPECIFIED targets applied: "
                  f"{ {r: v[2050] for r, v in settings.GBF3_NVIS_SEL_REGION_TARGETS.items()} } (2050 %)", flush=True)
        elif target_dict is not None:
            df['TARGET_LEVEL_2030'] = target_dict[2030]
            df['TARGET_LEVEL_2050'] = target_dict[2050]
            df['TARGET_LEVEL_2100'] = target_dict.get(2100, target_dict[2050])
        else:
            # CSV_DEFINED: keep CSV targets, filter out rows where any target is zero
            df = df[
                (df['TARGET_LEVEL_2030'] > 0) &
                (df['TARGET_LEVEL_2050'] > 0) &
                (df['TARGET_LEVEL_2100'] > 0)
            ].reset_index(drop=True)

        # Step 2b: per-(region, group) overrides, applied AFTER the uniform dict and independent of
        # GBF3_NVIS_TARGET mode — the NVIS counterpart of GBF4_SNES/ECNES_TARGETS_OVERRIDE. Note the
        # AUSTRALIA relabelling above has already happened, so keys use 'AUSTRALIA' in that mode.
        if settings.GBF3_NVIS_TARGETS_OVERRIDE:
            for (reg, grp), yr_dict in settings.GBF3_NVIS_TARGETS_OVERRIDE.items():
                mask = (df['region'] == reg) & (df['group'] == grp)
                for yr, pct in yr_dict.items():
                    col = f'TARGET_LEVEL_{yr}'
                    if col in df.columns:
                        df.loc[mask, col] = float(pct)
            print(f"│   │   ├── NVIS per-group target overrides applied: "
                  f"{len(settings.GBF3_NVIS_TARGETS_OVERRIDE)} pairs", flush=True)

        # Step 5: filter by GBF3_NVIS_SEL_REGION_TARGETS (NRM mode only; a dict's keys under 'SPECIFIED')
        if settings.GBF3_NVIS_REGION_MODE == 'NRM':
            df = df[df['region'].isin(list(settings.GBF3_NVIS_SEL_REGION_TARGETS))].reset_index(drop=True)

        # Step 6: drop (region, group) pairs with too little restorable habitat inside LUTO
        before = len(df)
        df = df[df['IN_LUTO_HA'] >= settings.GBF3_NVIS_MIN_AREA_HA].reset_index(drop=True)
        if before - len(df) and verbose:
            print(f"│   │   └── GBF3_NVIS_MIN_AREA_HA = {settings.GBF3_NVIS_MIN_AREA_HA}: dropped "
                  f"{before - len(df)} (region, group) pair(s) with IN_LUTO_HA below it", flush=True)

        return df


    def get_SNES_sparse_array(self) -> xr.DataArray:
        """
        Load the full SNES sparse array from disk and return as a sparse-backed xr.DataArray.

        Shape is (species, presence, cell) with presence=['LIKELY', 'LIKELY_AND_MAYBE'].
        Use .sel(species=sp, presence=pres).data.todense() to materialise a single slice.

        Returns
        -------
        xr.DataArray
            shape (species, presence, cell), sparse-backed.
        """
        return self.read_sparse_layer('bio_GBF4_SNES_sparse.nc')


    def get_ECNES_sparse_array(self) -> xr.DataArray:
        """
        Load the full ECNES sparse array from disk and return as a sparse-backed xr.DataArray.

        Shape is (species, presence, cell) with presence=['LIKELY', 'LIKELY_AND_MAYBE'].
        Note: the 'species' dimension name is inherited from the source NC and represents
        ecological communities, not individual species.
        Use .sel(species=comm, presence=pres).data.todense() to materialise a single slice.

        Returns
        -------
        xr.DataArray
            shape (species, presence, cell), sparse-backed.
        """
        return self.read_sparse_layer('bio_GBF4_ECNES_sparse.nc')


    def get_coord(self, index_ij: np.ndarray, trans):
        """
        Calculate the coordinates [[lon,...],[lat,...]] based on
        the given index [[row,...],[col,...]] and transformation matrix.

        Parameters
        index_ij (np.ndarray): A numpy array containing the row and column indices.
        trans (affin): An instance of the Transformation class.
        resfactor (int, optional): The resolution factor. Defaults to 1.

        Returns
        tuple: A tuple containing the x and y coordinates.
        """
        coord_x = trans.c + trans.a * (index_ij[1] + 0.5)    # Move to the center of the cell
        coord_y = trans.f + trans.e * (index_ij[0] + 0.5)    # Move to the center of the cell
        return coord_x, coord_y


    def update_geo_meta(self):
        """
        Update the geographic metadata based on the current settings.

        Note: When this function is called, the RESFACTOR is assumend to be > 1,
        because there is no need to update the metadata if the RESFACTOR is 1.

        Returns
            dict: The updated geographic metadata.
        """
        meta = self.GEO_META_FULLRES.copy()
        height, width =  self.LUMAP_2D_RESFACTORED.shape
        trans = list(self.GEO_META_FULLRES['transform'])
        trans[0] = trans[0] * settings.RESFACTOR    # Adjust the X resolution
        trans[4] = trans[4] * settings.RESFACTOR    # Adjust the Y resolution
        trans = Affine(*trans)
        meta.update(width=width, height=height, compress='lzw', driver='GTiff', transform=trans, nodata=self.NODATA, dtype='float32')
        return meta


    def add_lumap(self, yr: int, lumap: np.ndarray):
        """
        Safely adds a land-use map to the the Data object.
        """
        self.lumaps[yr] = lumap

    def add_lmmap(self, yr: int, lmmap: np.ndarray):
        """
        Safely adds a land-management map to the Data object.
        """
        self.lmmaps[yr] = lmmap

    def add_ammaps(self, yr: int, ammap: np.ndarray):
        """
        Safely adds an agricultural management map to the Data object.
        """
        self.ammaps[yr] = ammap

    def add_ag_dvars(self, yr: int, ag_dvars: np.ndarray):
        """
        Safely adds agricultural decision variables' values to the Data object.
        """
        self.ag_dvars[yr] = ag_dvars

    def add_delta_dvars_ag2ag(self, yr: int, dvar_D_ag2ag_mrj: dict | None):
        """
        Saves the solved ag→ag transition deltas, SOURCE-KEYED for true from→to flow reporting:
        {(from_m, from_j): ndarray(NLMS, ncells_src, N_AG_LUS) [to_m, local_r, to_j]} over each
        source's cells (get_base_dvar_mj_cell_map(self, base_year) recovers the global cell indices).
        """
        self.delta_dvars_ag2ag[yr] = dvar_D_ag2ag_mrj

    def add_non_ag_dvars(self, yr: int, non_ag_dvars: np.ndarray):
        """
        Safely adds non-agricultural decision variables' values to the Data object.
        """
        self.non_ag_dvars[yr] = non_ag_dvars

    def add_delta_dvars_ag2nonag(self, yr: int, dvar_D_ag2nonag_rk: dict | None):
        """
        Saves the solved ag→non-ag transition deltas, SOURCE-KEYED:
        {(from_m, from_j): ndarray(ncells_src, N_NON_AG_LUS) [local_r, k]} over each source's cells.
        """
        self.delta_dvars_ag2nonag[yr] = dvar_D_ag2nonag_rk

    def add_delta_dvars_nonag2ag(self, yr: int, dvar_D_nonag2ag_mrj: dict | None):
        """
        Saves the solved non-ag→ag transition deltas, SOURCE-KEYED:
        {from_k: ndarray(NLMS, ncells_k, N_AG_LUS) [to_m, local_r, to_j]} over each source's cells
        (get_base_nonag_dvar_k_cell_map) — e.g. reversible Destocked land converting back to ag.
        """
        self.delta_dvars_nonag2ag[yr] = dvar_D_nonag2ag_mrj

    def add_ag_man_dvars(self, yr: int, ag_man_dvars: dict[str, np.ndarray]):
        """
        Safely adds agricultural management decision variables' values to the Data object.
        """
        self.ag_man_dvars[yr] = ag_man_dvars
                


    def get_exact_resfactored_lumap_mrj(self):
        """
        Rather than picking the center cell when resfactoring the lumap, this function
        calculate the exact value of each land-use cell based from lumap to create dvars.

        E.g., given a resfactor of 5, then each resfactored dvar cell will cover a 5x5 area.
        If there are 9 Apple cells in the 5x5 area, then the dvar cell for it will be 9/25.

        """
        if settings.RESFACTOR == 1:
            return tools.lumap2ag_l_mrj(self.LUMAP_NO_RESFACTOR.values, self.LMMAP_NO_RESFACTOR.values)[:, self.MASK, :]

        lumap_mrj = np.zeros((self.NLMS, self.NCELLS, self.N_AG_LUS), dtype=np.float32)
        for idx_lu in self.DESC2AGLU.values():
            for idx_w, _ in enumerate(self.LANDMANS):
                # Get the cells with the same ID and water supply
                lu_arr = (self.LUMAP_NO_RESFACTOR.values == idx_lu) * (self.LMMAP_NO_RESFACTOR.values == idx_w)
                lumap_mrj[idx_w, :, idx_lu] = self.get_resfactored_average_fraction(lu_arr, use_valid_cell_count=False)
                    
        return lumap_mrj
    
    
    def get_resfactored_average_fraction(self, arr: np.ndarray, use_valid_cell_count: bool = True) -> np.ndarray:
        '''
        The block average of a full-res 1D array (len = the full-res land cells, 6956407), one value per model cell.

        Args:
            arr (np.ndarray): A 1D array over the full-res land cells.
            use_valid_cell_count (bool): If True (default), divide by the number of the block's cells where `arr` > 0
                (1 where there are none): the average over the cells that carry a value, e.g. a habitat fraction,
                so coastal blocks with fewer cells are not dwarfed against RES1.
                If False, divide by RESFACTOR² (all cells of a full block): the fraction of the whole coarse cell,
                required for lumap dvars, where REAL_AREA is the full coarse cell.

        Returns:
            np.ndarray: float32, one value per model cell (len = NCELLS).
        '''
        arr = np.asarray(arr)
        arr = np.where(np.isnan(arr), 0, arr) if arr.dtype.kind == 'f' else arr                 # NaN cells add nothing (as xarray's sum skipped them)
        block_total = tools.block_sum(self, arr)
        if use_valid_cell_count:
            denom = tools.block_sum(self, arr > 0)
            denom[denom == 0] = 1
        else:
            denom = settings.RESFACTOR ** 2
        return (block_total / denom).astype(np.float32)


    def get_resfactored_sum(self, arr: np.ndarray) -> np.ndarray:
        '''
        The block sum of a full-res 1D array over its land-use cells (LUMASK; the other cells do not count), one value
        per model cell.

        Args:
            arr (np.ndarray): A 1D array over the full-res land cells (len = 6956407).
        Returns:
            np.ndarray: float32, one value per model cell (len = NCELLS).
        '''
        arr = np.asarray(arr)
        arr = np.where(np.isnan(arr), 0, arr) if arr.dtype.kind == 'f' else arr                 # NaN cells add nothing (as xarray's sum skipped them)
        return tools.block_sum(self, arr * self.LUMASK).astype(np.float32)


    def get_resfactored_lumap(self) -> np.ndarray:
        """
        Coarsens the LUMAP to the specified resolution factor.
        """

        lumap_resfactored = np.zeros(self.NCELLS, dtype=np.int8) - 1
        fill_mask = np.ones(self.NCELLS, dtype=bool)

        # Fill resfactored land-use map with the land-use codes given their resfactored size
        for _,(lu_code, res_size) in self.LU_RESFACTOR_CELLS.iterrows():
            
            lu_avg = self.AG_L_MRJ[:,:,lu_code].sum(0) * fill_mask
            res_size = min(res_size, (lu_avg > 0).sum())
            
            # Assign the n-largets cells with the land-use code
            lu_idx = np.argsort(lu_avg)[-res_size:]  
            lumap_resfactored[lu_idx] = lu_code
            fill_mask[lu_idx] = False
            
        # Fill the remaining cells with the 'Unallocated - natural land' code
        lumap_resfactored[lumap_resfactored == -1] = self.DESC2AGLU['Unallocated - natural land']

        return lumap_resfactored


    def coarsen_guarded_arrays(self, x_mrj_full: xr.DataArray) -> dict:
        """
        RESFACTOR > 1: the agricultural economics, GHG, pasture and climate arrays are read at each block's
        centre cell, while EXCLUDE and AG_L_MRJ are block aggregates. An eligible coarse entry whose centre
        cell has no data is NaN, and the NaN becomes a zero-cost, zero-output land use.

        Each eligible coarse (lm, cell, lu) entry is replaced by the weighted mean over the block's fine cells
        that are eligible for (lm, lu), are land-use cells (LUMASK, the cells RESFACTOR 1 models) and carry
        data, with the weight its class requires. The weights nest, so each coarse product equals the block total
        of the fine products, e.g. sum(a) * mean_a(Yield) * mean_(a*Yield)(QC) = sum(a * Yield * QC). Entries
        that are not eligible keep their centre-cell value. An eligible entry is NaN only if no eligible fine
        cell in its block has data. EXCLUDE counts the same cells.

        Weight classes (a = fine cell area):
            area            a                           Yield, AC, FLC, FOC, FDC, WR, crop and irrigated-pasture GHG, PASTURE_KG_DM_HA
            production      a * Yield                   crop QC, P1, crop climate multiplier
            water           a * WR                      crop WP
                            head * WR                   WATER_DELIVERY_PRICE
            pasture         a * PASTURE_KG_DM_HA        SAFE_PUR
            capacity        pasture * SAFE_PUR          FEED_REQ
            head            capacity * FEED_REQ         livestock QC, WR, F, GHG, livestock climate multiplier
            head x F        head * F_k                  Q_k
            head x F x Q    head * F_k * Q_k            P_k
        A block whose weights sum to zero over cells with data (e.g. zero Yield) takes the area-weighted mean;
        its coarse product is zero either way.

        The crop arrays and the climate multiplier are indexed by (lm, lu) and are updated in place. The livestock
        inputs are shared by the four (lm, lu) entries of an animal, each with its own eligible cells and head, so
        their coarse values are returned per (lm, lu), read through `agec_lvstk`, `agghg_lvstk`, `agghg_irrpast`,
        `lvstk_pasture` and `water_delivery_price`. The shared attributes keep their centre-cell values.
        """
        crop_class = {
            'Yield': 'area', 'AC': 'area', 'FLC': 'area', 'FOC': 'area', 'FDC': 'area', 'WR': 'area',
            'QC': 'production', 'P1': 'production', 'WP': 'water'}
        lvstk_class = {
            'AC': 'area', 'FLC': 'area', 'FOC': 'area', 'FDC': 'area',
            'QC': 'head', 'WR_DRN': 'head', 'WR_IRR': 'head', 'F1': 'head', 'F2': 'head', 'F3': 'head',
            'Q1': 'head_F', 'Q2': 'head_F', 'Q3': 'head_F', 'P1': 'head_FQ', 'P2': 'head_FQ', 'P3': 'head_FQ'}

        def path(fname):
            return os.path.join(settings.INPUT_DIR, fname)
        cci_file = "climate_change_impacts_" + settings.RCP + "_CO2_FERT_" + settings.CO2_FERT.upper() + ".nc"

        # Coarse cell of each fine land cell (-1 if its block has none): the spatial lookup (S1).
        f2c = self.FULL2RES_IDX_MAP
        area = self.REAL_AREA_NO_RESFACTOR.values.astype(np.float64)

        # The fine inputs are read one full-res column at a time (a NetCDF chunk holds one column of every cell), so
        # peak memory stays near the few columns of one (lm, lu) (agec_crops alone is ~19 GB at full resolution).
        # Every block is reduced in one pass over all fine cells (bincount sums each block's cells in cell order).
        files = {name: xr.open_dataset(path(name)) for name in [
            "agec_crops.nc", "agGHG_crops.nc", cci_file, "agec_lvstk.nc", "agGHG_lvstk.nc", "agGHG_irrpast.nc",
            "feed_req.nc", "pasture_kg_dm_ha.nc", "safe_pur_natl.nc", "safe_pur_modl.nc", "water_delivery_price.nc"]}
        agec_crops, agghg_crops, cci = files["agec_crops.nc"]['AGEC_CROPS'], files["agGHG_crops.nc"]['AGGHG_CROPS'], files[cci_file]['CLIMATE_CHANGE_IMPACT']
        agec_lvstk, agghg_lvstk, agghg_irrpast = files["agec_lvstk.nc"]['AGEC_LVSTK'], files["agGHG_lvstk.nc"]['AGGHG_LVSTK'], files["agGHG_irrpast.nc"]

        def labels(table):
            # The column labels of a (column, cell) table, in column order: tuples of its label coordinates (the
            # levels of its index once loaded, the label coordinates as they sit in the file before)
            levels = table.indexes['column'].names if 'column' in table.indexes else \
                [name for name, coord in table.coords.items() if coord.dims == ('column',)]
            return list(zip(*[table[level].values.tolist() for level in levels]))
        position = {id(table): {label: i for i, label in enumerate(labels(table))} for table in [agec_crops, agghg_crops, cci, agec_lvstk, agghg_lvstk]}

        def fine(table, label):
            # One full-res column of a (column, cell) table, by its label
            return table[position[id(table)][label]].values.astype(np.float64)

        def weight_class(table, field):
            if field not in table:
                raise ValueError(f"coarsen_guarded_arrays: no weight class for '{field}'")
            return table[field]

        def as_float(centre):
            return np.array(centre, dtype=np.result_type(centre, np.float32), copy=True)     # PASTURE_KG_DM_HA and one AGGHG_IRRPAST column are integer

        # Outputs start as the centre-cell values (cell, column); the block means replace their eligible entries.
        agec_crops_labels, agghg_crops_labels, cci_labels = labels(self.AGEC_CROPS), labels(self.AGGHG_CROPS), labels(self.CLIMATE_CHANGE_IMPACT)
        crop_pairs = sorted({c[1:] for c in agec_crops_labels})
        agec_c = self.AGEC_CROPS.values.T.copy()
        agghg_c = self.AGGHG_CROPS.values.T.copy()
        cci_c = self.CLIMATE_CHANGE_IMPACT.values.T.copy()
        water_price_centre = np.nan_to_num(files["water_delivery_price.nc"]['WP'].values[self.MASK].astype(np.float64))   # WATER_DELIVERY_PRICE values (read later), float64
        irrpast_cols = [c for c in self.AGGHG_IRRPAST.data_vars if 'CO2E' in c]

        def animal_columns(table, level, animal):
            # The animal's columns of a livestock table (both label levels kept)
            return table.isel(column=table[level].values == animal)
        lvstk = {}
        for lm in self.LANDMANS:
            for lu in self.LU_LVSTK:
                animal, vegtype = ag_quantity.lvs_veg_types(lu)
                lvstk[lm, lu] = {
                    'AGEC_LVSTK': animal_columns(self.AGEC_LVSTK, 'animal', animal).values.T.copy(),
                    'AGGHG_LVSTK': animal_columns(self.AGGHG_LVSTK, 'animal', animal).values.T.copy(),
                    'AGGHG_IRRPAST': {c: as_float(self.AGGHG_IRRPAST[c].values) for c in irrpast_cols} if lm == 'irr' else None,
                    'FEED_REQ': as_float(self.FEED_REQ.values),
                    'PASTURE_KG_DM_HA': as_float(self.PASTURE_KG_DM_HA.values),
                    'SAFE_PUR': as_float((self.SAFE_PUR_NATL if vegtype == 'natural land' else self.SAFE_PUR_MODL).values),
                    'WATER_DELIVERY_PRICE': as_float(water_price_centre),
                }

        ok = f2c >= 0
        c_loc = f2c
        n_coarse = self.NCELLS
        a = area
        land_use = self.LUMASK

        def block_mean(v, w, elig):
            # Weighted mean of fine v over each block's cells in elig with finite v and w; NaN if there are none.
            k = elig & ok & np.isfinite(v) & np.isfinite(w)
            c = c_loc[k]
            num = np.bincount(c, weights=w[k] * v[k], minlength=n_coarse)
            den = np.bincount(c, weights=w[k], minlength=n_coarse)
            num_a = np.bincount(c, weights=a[k] * v[k], minlength=n_coarse)
            den_a = np.bincount(c, weights=a[k], minlength=n_coarse)
            out = np.full(n_coarse, np.nan)
            np.divide(num_a, den_a, out=out, where=den_a > 0)
            np.divide(num, den, out=out, where=den > 0)
            return out

        # Crops: AGEC_CROPS and AGGHG_CROPS by (field, lm, lu).
        for lm, lu in crop_pairs:
            m, j = self.LANDMANS.index(lm), self.DESC2AGLU[lu]
            elig, eligible = x_mrj_full.sel(lm=lm, lu=lu).values & land_use, self.EXCLUDE[m, :, j]
            if not eligible.any():
                continue
            weights = {'area': a, 'production': a * fine(agec_crops, ('Yield', lm, lu))}
            if ('WR', lm, lu) in position[id(agec_crops)]:
                weights['water'] = a * fine(agec_crops, ('WR', lm, lu))
            for i, col in enumerate(agec_crops_labels):
                if col[1:] == (lm, lu):
                    w = weights[weight_class(crop_class, col[0])]
                    agec_c[eligible, i] = block_mean(fine(agec_crops, col), w, elig)[eligible]
            for i, col in enumerate(agghg_crops_labels):
                if col[1:] == (lm, lu):
                    agghg_c[eligible, i] = block_mean(fine(agghg_crops, col), a, elig)[eligible]
            for i, col in enumerate(cci_labels):
                if col[:2] == (lm, lu):
                    cci_c[eligible, i] = block_mean(fine(cci, col), weights['production'], elig)[eligible]

        # Livestock: per (lm, lu), since an animal's inputs serve both land managements and both vegetation types.
        # The shared layers are read once; an animal's columns once for its four (lm, lu) entries.
        feed_req = files["feed_req.nc"]['FEED_REQ'].values.astype(np.float64)
        pasture = files["pasture_kg_dm_ha.nc"]['PASTURE_KG_DM_HA'].values.astype(np.float64)
        safe_pur = {'natural land': files["safe_pur_natl.nc"]['SAFE_PUR_NATL'].values.astype(np.float64),
                    'modified land': files["safe_pur_modl.nc"]['SAFE_PUR_MODL'].values.astype(np.float64)}
        water_price = files["water_delivery_price.nc"]['WP'].values.astype(np.float64)
        irrpast_fine = {col: agghg_irrpast[col].values.astype(np.float64) for col in irrpast_cols}
        animal_cache = {}
        for lu in self.LU_LVSTK:
            animal, vegtype = ag_quantity.lvs_veg_types(lu)
            if animal not in animal_cache:
                animal_cache = {animal: {
                    **{label: fine(agec_lvstk, label) for label in labels(agec_lvstk) if label[1] == animal},
                    **{label: fine(agghg_lvstk, label) for label in labels(agghg_lvstk) if label[0] == animal}}}
            column = animal_cache[animal]
            for m, lm in enumerate(self.LANDMANS):
                j = self.DESC2AGLU[lu]
                elig, eligible = x_mrj_full.sel(lm=lm, lu=lu).values & land_use, self.EXCLUDE[m, :, j]
                if not eligible.any():
                    continue
                out = lvstk[lm, lu]
                w = {'area': a, 'pasture': a * pasture}
                w['capacity'] = w['pasture'] * safe_pur[vegtype]
                w['head'] = w['capacity'] * feed_req
                wr = column['WR_DRN', animal] * settings.LIVESTOCK_DRINKING_WATER
                if lm == 'irr':
                    wr = wr + column['WR_IRR', animal]
                w['water'] = w['head'] * wr

                for i, (field, _) in enumerate(label for label in labels(self.AGEC_LVSTK) if label[1] == animal):
                    cls = weight_class(lvstk_class, field)
                    if cls == 'head_F':
                        wt = w['head'] * column[f'F{field[1]}', animal]
                    elif cls == 'head_FQ':
                        wt = w['head'] * column[f'F{field[1]}', animal] * column[f'Q{field[1]}', animal]
                    else:
                        wt = w[cls]
                    out['AGEC_LVSTK'][eligible, i] = block_mean(column[field, animal], wt, elig)[eligible]

                for i, col in enumerate(label for label in labels(self.AGGHG_LVSTK) if label[0] == animal):
                    out['AGGHG_LVSTK'][eligible, i] = block_mean(column[col], w['head'], elig)[eligible]

                if lm == 'irr':
                    for col in irrpast_cols:
                        out['AGGHG_IRRPAST'][col][eligible] = block_mean(irrpast_fine[col], a, elig)[eligible]

                for i, col in enumerate(cci_labels):
                    if col[:2] == (lm, lu):
                        cci_c[eligible, i] = block_mean(fine(cci, col), w['head'], elig)[eligible]

                out['FEED_REQ'][eligible] = block_mean(feed_req, w['capacity'], elig)[eligible]
                out['PASTURE_KG_DM_HA'][eligible] = block_mean(pasture, a, elig)[eligible]
                out['SAFE_PUR'][eligible] = block_mean(safe_pur[vegtype], w['pasture'], elig)[eligible]
                out['WATER_DELIVERY_PRICE'][eligible] = block_mean(water_price, w['water'], elig)[eligible]
        for opened in files.values():
            opened.close()

        # Back into the (column, cell) tables, labels unchanged
        self.AGEC_CROPS = self.AGEC_CROPS.copy(data=agec_c.T)
        self.AGGHG_CROPS = self.AGGHG_CROPS.copy(data=agghg_c.T)
        for (lm, lu), out in lvstk.items():
            animal, _ = ag_quantity.lvs_veg_types(lu)
            out['AGEC_LVSTK'] = animal_columns(self.AGEC_LVSTK, 'animal', animal).copy(data=out['AGEC_LVSTK'].T)
            out['AGGHG_LVSTK'] = animal_columns(self.AGGHG_LVSTK, 'animal', animal).copy(data=out['AGGHG_LVSTK'].T)
            if out['AGGHG_IRRPAST'] is not None:
                irrpast = self.AGGHG_IRRPAST.copy()
                for col, v in out['AGGHG_IRRPAST'].items():
                    irrpast[col] = ('cell', v)
                out['AGGHG_IRRPAST'] = irrpast

        self.CLIMATE_CHANGE_IMPACT = self.CLIMATE_CHANGE_IMPACT.copy(data=cci_c.T)
        return lvstk


    # The livestock inputs as read for one (lu, lm): the shared attributes at RESFACTOR 1, the per-(lm, lu)
    # block means at RESFACTOR > 1 (see coarsen_guarded_arrays). The tables stay xarray (selected by label by the
    # economics modules); the pasture layers and the delivery price enter the numpy yield arithmetic as arrays.
    def agec_lvstk(self, lu: str, lm: str) -> xr.DataArray:
        coarse = getattr(self, 'LVSTK_COARSE', None)
        return self.AGEC_LVSTK if coarse is None else coarse[lm, lu]['AGEC_LVSTK']

    def agghg_lvstk(self, lu: str, lm: str) -> xr.DataArray:
        coarse = getattr(self, 'LVSTK_COARSE', None)
        return self.AGGHG_LVSTK if coarse is None else coarse[lm, lu]['AGGHG_LVSTK']

    def agghg_irrpast(self, lu: str, lm: str) -> xr.Dataset:
        coarse = getattr(self, 'LVSTK_COARSE', None)
        return self.AGGHG_IRRPAST if coarse is None else coarse[lm, lu]['AGGHG_IRRPAST']

    def water_delivery_price(self, lu: str, lm: str) -> np.ndarray:
        coarse = getattr(self, 'LVSTK_COARSE', None)
        return self.WATER_DELIVERY_PRICE.values if coarse is None else coarse[lm, lu]['WATER_DELIVERY_PRICE']

    def lvstk_pasture(self, lu: str, lm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(FEED_REQ, PASTURE_KG_DM_HA, SAFE_PUR) for (lu, lm); SAFE_PUR is NATL or MODL by vegetation type."""
        _, vegtype = ag_quantity.lvs_veg_types(lu)
        coarse = getattr(self, 'LVSTK_COARSE', None)
        if coarse is not None:
            c = coarse[lm, lu]
            return c['FEED_REQ'], c['PASTURE_KG_DM_HA'], c['SAFE_PUR']
        safe_pur = self.SAFE_PUR_NATL if vegtype == 'natural land' else self.SAFE_PUR_MODL
        return self.FEED_REQ.values, self.PASTURE_KG_DM_HA.values, safe_pur.values



    def get_potential_production_lyr(self, yr_cal:int):
        '''
        Return the potential production data for a given year as xarray DataArrays.
        The returned DataArrays are spatial layers for `agricultural`, `non-agricultural`, and `agricultural management` commodities,
        where each cell is the production potential of a commodity.
        
        Note: the 'potential' means the production is calculated based on production potential matrices,
        meaning the production is calculated without considering the actual land-use, but only using the quality marices.
        '''
        
        # Calculate year index (i.e., number of years since 2010)
        yr_idx = yr_cal - self.YR_CAL_BASE
        
        # Get commodity matrices
        ag_q_mrp_xr = xr.DataArray(
            ag_quantity.get_quantity_matrices(self, yr_idx).astype(np.float32),
            dims=['lm','cell','product'],
            coords={
                'lm': self.LANDMANS,
                'cell': range(self.NCELLS),
                'product': self.PRODUCTS
            },
        ).assign_coords(
            region=('cell', self.REGION_NRM_NAME),
        )

        non_ag_crk_xr = xr.DataArray(
            non_ag_quantity.get_quantity_matrix(self, ag_q_mrp_xr).astype(np.float32),
            dims=['Commodity', 'cell', 'lu'],
            coords={
                'Commodity': self.COMMODITIES,
                'cell': range(self.NCELLS),
                'lu': self.NON_AGRICULTURAL_LANDUSES
            },
        ).assign_coords(
            region=('cell', self.REGION_NRM_NAME),
        )

        ag_man_q_amrp_xr = xr.DataArray(
            np.stack(list(ag_quantity.get_agricultural_management_quantity_matrices(self, ag_q_mrp_xr, yr_idx).values())).astype(np.float32),
            dims=['am', 'lm', 'cell', 'product'],
            coords={
                'am': self.AG_MAN_DESC,
                'lm': self.LANDMANS,
                'cell': np.arange(self.NCELLS),
                'product': self.PRODUCTS
            }
        ).assign_coords(
            region=('cell', self.REGION_NRM_NAME),
        )

        return (ag_q_mrp_xr.compute(), non_ag_crk_xr.compute(), ag_man_q_amrp_xr.compute())
    
    
    def get_actual_production_lyr(self, yr_cal:int, threshold: float = 0.01):
        '''
        Return the production data for a given year as xarray DataArrays.
        The returned DataArrays are spatial layers where each cell is the production of a commodity.
        Such as t/cell for apples, beef. Or ML/cell for milk.
        
        Note: the 'actual' means the production is calculated based on true decision variables, 
        meaning the production is calculated based on actual land-use areas.

        `threshold`: a cell whose total ag / non-ag / ag-mgt share is at or below it is zeroed before the product
        (the converters' map clean-up, 0.01 = 1 % of a cell). Pass 0 for an accounting total that counts every
        share (`simulation.store_solution`: the solver met demand with every share, so the total must too).
        '''
        # Get dvars and production potential matrices
        if yr_cal == self.YR_CAL_BASE:
            ag_X_mrj = self.AG_L_MRJ
            non_ag_X_rk = self.NON_AG_L_RK
            ag_man_X_mrj = self.AG_MAN_L_MRJ_DICT
            
            ag_q_mrp_xr = self.prod_base_yr_potential_ag_mrp
            non_ag_crk_xr = self.prod_base_yr_potential_non_ag_rp
            ag_man_q_amrp_xr = self.prod_base_yr_potential_am_amrp
            
        else: # In this case, the dvars are already appended from the solver
            ag_X_mrj = self.ag_dvars[yr_cal]
            non_ag_X_rk = self.non_ag_dvars[yr_cal]
            ag_man_X_mrj = self.ag_man_dvars[yr_cal]
            
            ag_q_mrp_xr, non_ag_crk_xr, ag_man_q_amrp_xr = self.get_potential_production_lyr(yr_cal)

        # Convert dvar array to xr.DataArray; Chunk the data to reduce memory usage
        ag_X_mrj_xr = tools.ag_mrj_to_xr(self, ag_X_mrj, threshold).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, self.NCELLS)})
        non_ag_X_rk_xr = tools.non_ag_rk_to_xr(self, non_ag_X_rk, threshold).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, self.NCELLS)})
        ag_man_X_amrj_xr = tools.am_mrj_to_xr(self, ag_man_X_mrj, threshold).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, self.NCELLS)})

        # Calculate the commodity production (BEFORE dimension expansion to avoid double counting)
        #   Using xr.dot() instead of broadcasting for better memory efficiency and performance
        ag_q_mrc = xr.dot((xr.dot(ag_X_mrj_xr, self.lu2pr_xr, dims=['lu']) * ag_q_mrp_xr), self.pr2cm_xr, dims=['product'])
        non_ag_p_rc = xr.dot(non_ag_X_rk_xr, non_ag_crk_xr, dims=['lu'])
        am_p_amrc = xr.dot((xr.dot(ag_man_X_amrj_xr, self.lu2pr_xr, dims=['lu']) * ag_man_q_amrp_xr), self.pr2cm_xr, dims=['product'])

        return ag_q_mrc, non_ag_p_rc, am_p_amrc
    
    
    
    def get_production_from_base_dvar_under_target_CCI_and_yield_change(self, yr_cal:int):
        '''
        Get the production aggregated stats (t/ML for each commodity) based on `YR_CAL_BASE` decision variables
        but under the target CCI and yield change of the given year.
        
        This function is used to calculate the 'what-if' production when there is not climate change impact and yield change
        since the `YR_CAL_BASE`. This is useful for evaluating the supply-delta caused by climate change and yield change.
        '''

        # Convert np.array to xr.DataArray; Chunk the data to reduce memory usage
        ag_X_mrj_xr = tools.ag_mrj_to_xr(self, self.AG_L_MRJ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, self.NCELLS)})
        non_ag_X_rk_xr = tools.non_ag_rk_to_xr(self, self.NON_AG_L_RK).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, self.NCELLS)})
        ag_man_X_amrj_xr = tools.am_mrj_to_xr(self, self.AG_MAN_L_MRJ_DICT).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, self.NCELLS)})

        # Get potential production layers
        ag_q_mrp_xr_target_yr, non_ag_crk_xr_target_yr, ag_man_q_amrp_xr_target_yr = self.get_potential_production_lyr(yr_cal)

        # Calculate total impact
        ag_production_c = xr.dot(
            xr.dot(ag_X_mrj_xr, self.lu2pr_xr, dims=['lu']) * ag_q_mrp_xr_target_yr, self.pr2cm_xr,
            dim=['cell', 'lm','product']
        ).compute()
        non_ag_production_c = xr.dot(non_ag_X_rk_xr, non_ag_crk_xr_target_yr, dims=['cell', 'lu'])
        am_p_amrc = xr.dot(
            (xr.dot(ag_man_X_amrj_xr, self.lu2pr_xr, dims=['lu']) * ag_man_q_amrp_xr_target_yr), self.pr2cm_xr,
            dims=['am', 'cell', 'lm','product']
        )

        return (ag_production_c.compute() + non_ag_production_c.compute() + am_p_amrc.compute())
    
    
    def get_elasticity_multiplier(self, yr_cal:int):
        '''
        Get the elasticity multiplier for a given year and land use.
        yr_cal: year (int).

        Returns:
            xr.DataArray: the price multiplier of every commodity, over Commodity (1 everywhere when DYNAMIC_PRICE is off).
        '''

        # Return cached result if available for this year
        if yr_cal in self.DEMAND_ELASTICITY_MUL:
            return self.DEMAND_ELASTICITY_MUL[yr_cal]

        # Get supply delta (0-based ratio)
        supply_base_dvar_base_production = self.BASE_YR_production_t
        supply_base_dvar_target_production = self.get_production_from_base_dvar_under_target_CCI_and_yield_change(yr_cal)
        delta_supply = (supply_base_dvar_target_production - supply_base_dvar_base_production) / supply_base_dvar_base_production

        # Get demand delta (0-based ratio)
        demand_base_year = self.DEMAND_C.sel(year=self.YR_CAL_BASE)
        demand_target_year = self.DEMAND_C.sel(year=yr_cal)
        delta_demand = (demand_target_year - demand_base_year) / demand_base_year

        # Calculate price_multiplier (1-based ratio); every term over Commodity, aligned by label
        price_delta = (delta_demand - delta_supply) / (self.elasticity_demand + self.elasticity_supply)
        elasticity_multiplier = price_delta + 1

        if settings.DYNAMIC_PRICE:
            self.DEMAND_ELASTICITY_MUL[yr_cal] = elasticity_multiplier
        else:
            self.DEMAND_ELASTICITY_MUL[yr_cal] = xr.ones_like(elasticity_multiplier)

        return self.DEMAND_ELASTICITY_MUL[yr_cal]
    
    
    
    def get_watershed_yield_components(self, valid_watershed_id:list[int] = None):
        """
        Get the water yield components for the specified watersheds.
        """
        if settings.WATER_REGION_DEF == 'River Region':
            region_file, id_name = 'water_region_rivreg.nc', 'HR_RIVREG_ID'
        elif settings.WATER_REGION_DEF == 'Drainage Division':
            region_file, id_name = 'water_region_draindiv.nc', 'HR_DRAINDIV_ID'
        else:
            raise ValueError(f"Unknown water region definition: {settings.WATER_REGION_DEF}. "
                        f"Must be either 'River Region' or 'Drainage Division'.")

        water_region = xr.load_dataset(os.path.join(settings.INPUT_DIR, region_file))

        self.WATER_REGION_ID = water_region[id_name].isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))   # the region of every model cell
        self.WATER_REGION_NAMES = dict(zip(water_region['region_id'].values, water_region['NAME'].values))
        self.WATER_REGION_HIST_LEVEL = water_region['HIST_BASELINE_ML']                                      # over region_id

        self.WATER_OUTSIDE_LUTO_BY_CCI = water_region['OUTSIDE_LUTO_ML'].sel(ssp=str(settings.SSP), drop=True)    # over (year, region_id)
        water_natural_land = water_region['NATURAL_LAND_ML'].sel(ssp=str(settings.SSP), drop=True)            # read as before; not used (reported as dead code)
        self.WATER_OUTSIDE_LUTO_HIST = self.WATER_YIELD_OUTSIDE_LUTO_HIST[settings.WATER_REGION_DEF]          # over region_id


        # Using high res factor will make some small river regions disappear; hence we update the data with validated river region ids
        if valid_watershed_id is None:
            valid_watershed_id = np.unique(self.WATER_REGION_ID)
        self.WATERSHED_DISAPPEARING = [self.WATER_REGION_NAMES[i] for i in set(self.WATER_REGION_NAMES.keys()) - set(valid_watershed_id)]
        
        if self.WATERSHED_DISAPPEARING:
            print(f"│   ⚠ WARNING: {len(self.WATERSHED_DISAPPEARING)} river regions are disappearing due to using high resolution factors.", flush=True)
            [print(f"       - {list(i)}") for i in np.array_split(self.WATERSHED_DISAPPEARING, len(self.WATERSHED_DISAPPEARING)//3)]
            self.WATER_REGION_NAMES = {k: v for k, v in self.WATER_REGION_NAMES.items() if k in valid_watershed_id}
            self.WATER_REGION_HIST_LEVEL = self.WATER_REGION_HIST_LEVEL.sel(region_id=[k for k in self.WATER_REGION_HIST_LEVEL['region_id'].values if k in valid_watershed_id])
            self.WATER_OUTSIDE_LUTO_BY_CCI = self.WATER_OUTSIDE_LUTO_BY_CCI.sel(region_id=[k for k in self.WATER_OUTSIDE_LUTO_BY_CCI['region_id'].values if k in valid_watershed_id])
            self.WATER_OUTSIDE_LUTO_HIST = self.WATER_OUTSIDE_LUTO_HIST.sel(region_id=[k for k in self.WATER_OUTSIDE_LUTO_HIST['region_id'].values if k in valid_watershed_id])
            self.WATER_USE_DOMESTIC = {k: v for k, v in self.WATER_USE_DOMESTIC.items() if k in valid_watershed_id}

        return valid_watershed_id
    
    
    def get_GBF2_target_for_yr_cal(self, yr_cal:int) -> float:
        """
        Get the target score for priority degrade areas conservation.
        
        Parameters
        ----------
        yr_cal : int
            The year for which to get the habitat condition score.
            
        Returns
        -------
        float
            The priority degrade areas conservation target for the given year.
        """
        
        bio_habitat_score_baseline_sum = (self.BIO_GBF2_MASK * self.REAL_AREA.values * self.AG_MASK_PROPORTION_R).sum()
        bio_habitat_score_base_yr_sum = self.BIO_GBF2_BASE_YR.sum()
        bio_habitat_score_base_yr_proportion = bio_habitat_score_base_yr_sum / bio_habitat_score_baseline_sum

        bio_habitat_target_proportion = [
            bio_habitat_score_base_yr_proportion + ((1 - bio_habitat_score_base_yr_proportion) * i)
            for i in settings.GBF2_TARGETS_DICT[settings.GBF2_TARGET].values()
        ]

        targets_key_years = {
            self.YR_CAL_BASE: bio_habitat_score_base_yr_sum, 
            **dict(zip(
                settings.GBF2_TARGETS_DICT[settings.GBF2_TARGET].keys(), 
                bio_habitat_score_baseline_sum * np.array(bio_habitat_target_proportion)
            ))
        }

        f = interp1d(
            list(targets_key_years.keys()),
            list(targets_key_years.values()),
            kind = "linear",
            fill_value = "extrapolate"
        )

        return f(yr_cal).item()  # Convert the interpolated value to a scalar
    

    def get_SNES_targets_df(self, include_all: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Load and filter SNES targets from CSV.

        Returns a DataFrame with selected (region, SCIENTIFIC_NAME, presence) rows.
        AUSTRALIA mode: presence in {'LIKELY', 'LIKELY_AND_MAYBE'}, region='AUSTRALIA'
        NRM mode:       presence='LIKELY', region=NRM region name
        Set include_all=True to return all NRM/STATE rows for writing/reporting.
        Dict targets are applied inside this function so all call sites stay consistent.
        """
        snes_df = (
            pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_SNES.csv', low_memory=False)
            .query(f"resfactor == {settings.RESFACTOR}")
            .query(f"presence == '{settings.GBF4_SNES_PRESENCE_CLASS}'")
            .reset_index(drop=True)
        )
        
        if include_all:
            return snes_df

        snes_df = (
            snes_df
            .query(f"region_level == '{settings.GBF4_SNES_REGION_MODE}'")
            .reset_index(drop=True)
        )

        if settings.GBF4_TARGET_SNES in ('medium', 'high'):
            # Uniform preset targets for ALL species at this presence/region
            # (the CSV's own TARGET_LEVEL_* values, mostly NaN, are ignored)
            for yr, pct in settings.GBF4_SNES_TARGETS_DICT[settings.GBF4_TARGET_SNES].items():
                col = f'TARGET_LEVEL_{yr}'
                if col in snes_df.columns:
                    snes_df[col] = float(pct)
            print(f"│   │   ├── '{settings.GBF4_TARGET_SNES}' level targets applied to all SNES species", flush=True)
        elif settings.GBF4_TARGET_SNES == 'SPECIFIED':
            # 'SPECIFIED' — select items exactly as 'CSV_DEFINED' (only rows with a CSV target), then give every
            # item in a region that region's uniform {year: pct} from GBF4_SNES_SEL_REGION_TARGETS
            # (every year must be given — a missing one raises)
            snes_df = snes_df.query("TARGET_LEVEL_2030 > 0").reset_index(drop=True)
            snes_df = snes_df[snes_df['region'].isin(settings.GBF4_SNES_SEL_REGION_TARGETS)].reset_index(drop=True)
            for reg, yr_dict in settings.GBF4_SNES_SEL_REGION_TARGETS.items():
                for yr in (2030, 2050, 2100):
                    snes_df.loc[snes_df['region'] == reg, f'TARGET_LEVEL_{yr}'] = float(yr_dict[yr])
            print(f"│   │   ├── SNES SPECIFIED targets applied: "
                  f"{ {r: v[2050] for r, v in settings.GBF4_SNES_SEL_REGION_TARGETS.items()} } (2050 %)", flush=True)
        else:
            # 'CSV_DEFINED' — keep CSV targets; only species with a defined 2030 target
            snes_df = snes_df.query("TARGET_LEVEL_2030 > 0").reset_index(drop=True)

        if settings.GBF4_SNES_REGION_MODE != 'AUSTRALIA':
            snes_df = snes_df[snes_df['region'].isin(list(settings.GBF4_SNES_SEL_REGION_TARGETS))].reset_index(drop=True)

        before = len(snes_df)
        snes_df = snes_df[snes_df['IN_LUTO_HA'] >= settings.GBF4_SNES_MIN_AREA_HA].reset_index(drop=True)
        if before - len(snes_df):
            print(f"│   │   ├── GBF4_SNES_MIN_AREA_HA = {settings.GBF4_SNES_MIN_AREA_HA}: dropped "
                  f"{before - len(snes_df)} SNES (region, species) pair(s) with IN_LUTO_HA below it", flush=True)

        if settings.GBF4_SNES_TARGETS_OVERRIDE:
            for (reg, sp), yr_dict in settings.GBF4_SNES_TARGETS_OVERRIDE.items():
                mask = (snes_df['region'] == reg) & (snes_df['SCIENTIFIC_NAME'] == sp)
                for yr, pct in yr_dict.items():
                    col = f'TARGET_LEVEL_{yr}'
                    if col in snes_df.columns:
                        snes_df.loc[mask, col] = float(pct)
            print(f"│   │   ├── SNES per-species target overrides applied: {len(settings.GBF4_SNES_TARGETS_OVERRIDE)} pairs", flush=True)

        print(f"│   │   └── {len(snes_df)} (species, region) constraints from {snes_df['SCIENTIFIC_NAME'].nunique()} species", flush=True)

        return snes_df


    def get_ECNES_targets_df(self, include_all: bool = False) -> pd.DataFrame:
        """Load and filter ECNES targets from CSV.

        Returns a DataFrame with selected (region, COMMUNITY, presence) rows.
        AUSTRALIA mode: presence in {'LIKELY', 'LIKELY_AND_MAYBE'}, region='AUSTRALIA'
        NRM mode:       presence='LIKELY', region=NRM region name
        Set include_all=True to return all rows for writing/reporting.
        Dict targets are applied inside this function so all call sites stay consistent.
        """
        ecnes_df = (
            pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_ECNES.csv')
            .query(f"resfactor == {settings.RESFACTOR}")
            .query(f"presence == '{settings.GBF4_ECNES_PRESENCE_CLASS}'")
            .reset_index(drop=True)
        )
        if include_all:
            return ecnes_df

        ecnes_df = (
            ecnes_df
            .query(f"region_level == '{settings.GBF4_ECNES_REGION_MODE}'")
            .reset_index(drop=True)
        )

        if settings.GBF4_TARGET_ECNES in ('medium', 'high'):
            # Uniform preset targets for ALL communities at this presence/region
            # (the CSV's own TARGET_LEVEL_* values, mostly NaN, are ignored)
            for yr, pct in settings.GBF4_ECNES_TARGETS_DICT[settings.GBF4_TARGET_ECNES].items():
                col = f'TARGET_LEVEL_{yr}'
                if col in ecnes_df.columns:
                    ecnes_df[col] = float(pct)
            print(f"│   │   ├── '{settings.GBF4_TARGET_ECNES}' level targets applied to all ECNES communities", flush=True)
        elif settings.GBF4_TARGET_ECNES == 'SPECIFIED':
            # 'SPECIFIED' — select items exactly as 'CSV_DEFINED' (only rows with a CSV target), then give every
            # item in a region that region's uniform {year: pct} from GBF4_ECNES_SEL_REGION_TARGETS
            # (every year must be given — a missing one raises)
            ecnes_df = ecnes_df.query("TARGET_LEVEL_2030 > 0").reset_index(drop=True)
            ecnes_df = ecnes_df[ecnes_df['region'].isin(settings.GBF4_ECNES_SEL_REGION_TARGETS)].reset_index(drop=True)
            for reg, yr_dict in settings.GBF4_ECNES_SEL_REGION_TARGETS.items():
                for yr in (2030, 2050, 2100):
                    ecnes_df.loc[ecnes_df['region'] == reg, f'TARGET_LEVEL_{yr}'] = float(yr_dict[yr])
            print(f"│   │   ├── ECNES SPECIFIED targets applied: "
                  f"{ {r: v[2050] for r, v in settings.GBF4_ECNES_SEL_REGION_TARGETS.items()} } (2050 %)", flush=True)
        else:
            # 'CSV_DEFINED' — keep CSV targets; only communities with a defined 2030 target
            ecnes_df = ecnes_df.query("TARGET_LEVEL_2030 > 0").reset_index(drop=True)

        if settings.GBF4_ECNES_REGION_MODE != 'AUSTRALIA':
            ecnes_df = ecnes_df[ecnes_df['region'].isin(list(settings.GBF4_ECNES_SEL_REGION_TARGETS))].reset_index(drop=True)

        before = len(ecnes_df)
        ecnes_df = ecnes_df[ecnes_df['IN_LUTO_HA'] >= settings.GBF4_ECNES_MIN_AREA_HA].reset_index(drop=True)
        if before - len(ecnes_df):
            print(f"│   │   ├── GBF4_ECNES_MIN_AREA_HA = {settings.GBF4_ECNES_MIN_AREA_HA}: dropped "
                  f"{before - len(ecnes_df)} ECNES (region, community) pair(s) with IN_LUTO_HA below it", flush=True)

        if settings.GBF4_ECNES_TARGETS_OVERRIDE:
            for (reg, com), yr_dict in settings.GBF4_ECNES_TARGETS_OVERRIDE.items():
                mask = (ecnes_df['region'] == reg) & (ecnes_df['COMMUNITY'] == com)
                for yr, pct in yr_dict.items():
                    col = f'TARGET_LEVEL_{yr}'
                    if col in ecnes_df.columns:
                        ecnes_df.loc[mask, col] = float(pct)
            print(f"│   │   ├── ECNES per-community target overrides applied: {len(settings.GBF4_ECNES_TARGETS_OVERRIDE)} pairs", flush=True)

        print(f"│   │   └── {len(ecnes_df)} (community, region) constraints from {ecnes_df['COMMUNITY'].nunique()} communities", flush=True)

        return ecnes_df


    def get_GBF3_NVIS_limit_score_inside_LUTO_by_yr(
            self,
            yr: int,
        ) -> xr.DataArray:
            """Interpolate GBF3 NVIS vegetation targets for the given year."""
            targets_df = self.get_NVIS_targets_df()
            sel_pairs = list(zip(targets_df['region'], targets_df['group']))

            GBF3_NVIS_target_percents = xr.DataArray(
                np.zeros(len(targets_df), dtype=np.float32),
                dims=['layer'],
                coords={'layer': pd.MultiIndex.from_tuples(sel_pairs, names=['region', 'group'])}
            )

            for _, row in targets_df.iterrows():
                f = interp1d(
                    [2010, 2030, 2050, 2100],
                    [min(row['BASEYEAR_LEVEL'], row['TARGET_LEVEL_2030']),
                    row['TARGET_LEVEL_2030'],
                    row['TARGET_LEVEL_2050'],
                    row['TARGET_LEVEL_2100']
                    ],
                    kind="linear",
                    fill_value="extrapolate",
                ) 
                GBF3_NVIS_target_percents.loc[dict(layer=(row['region'], row['group']))] = (
                    row['ALL_HA']
                    * (f(yr).item() / 100)  # Convert percentage to proportion
                    - row['NATURAL_OUT_LUTO_HA']
                )


            return GBF3_NVIS_target_percents
    
    
    def get_GBF4_SNES_target_inside_LUTO_by_year(self, yr: int) -> xr.DataArray:
        snes_df = self.GBF4_SNES_META

        result = xr.DataArray(
            np.zeros(len(self.BIO_GBF4_SNES_SEL), dtype=np.float32),
            dims=['layer'],
            coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF4_SNES_SEL, names=['region', 'species', 'presence'])},
        )
        for _, row in snes_df.iterrows():
            region   = row['region']
            species  = row['SCIENTIFIC_NAME']
            presence = row['presence']
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [min(row['BASEYEAR_LEVEL'], row['TARGET_LEVEL_2030']),
                 row['TARGET_LEVEL_2030'], row['TARGET_LEVEL_2050'], row['TARGET_LEVEL_2100']],
                kind='linear', fill_value='extrapolate',
            )
            interp_pct = float(f(yr))
            attainable_pct = row['ATTAINABLE_LEVEL'] - settings.GBF4_SNES_CAP_MARGIN
            target_pct = min(interp_pct, attainable_pct)
            if interp_pct > attainable_pct:
                print(f"│   ├── SNES target capped for '{species}' ({presence}) [{region}]: {interp_pct:.2f}% -> {attainable_pct:.2f}% (attainable - {settings.GBF4_SNES_CAP_MARGIN:g} margin)", flush=True)
            score_all_aus = row['ALL_HA'] * target_pct / 100
            score_out_LUTO = row['NATURAL_OUT_LUTO_HA']
            result.loc[dict(layer=(region, species, presence))] = score_all_aus - score_out_LUTO
        return result




    def get_GBF4_ECNES_target_inside_LUTO_by_year(self, yr: int) -> xr.DataArray:
        ecnes_df = self.GBF4_ECNES_META

        result = xr.DataArray(
            np.zeros(len(self.BIO_GBF4_ECNES_SEL), dtype=np.float32),
            dims=['layer'],
            coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF4_ECNES_SEL, names=['region', 'species', 'presence'])},
        )
        for _, row in ecnes_df.iterrows():
            region   = row['region']
            species  = row['COMMUNITY']
            presence = row['presence']
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [min(row['BASEYEAR_LEVEL'], row['TARGET_LEVEL_2030']),
                 row['TARGET_LEVEL_2030'], row['TARGET_LEVEL_2050'], row['TARGET_LEVEL_2100']],
                kind='linear', fill_value='extrapolate',
            )
            interp_pct = float(f(yr))
            attainable_pct = row['ATTAINABLE_LEVEL']
            target_pct = min(interp_pct, attainable_pct)
            if interp_pct > attainable_pct:
                print(f"│   ├── ECNES target capped for '{species}' ({presence}): {interp_pct:.2f}% -> {attainable_pct:.2f}% (attainable limit)", flush=True)
            score_all_aus = row['ALL_HA'] * target_pct / 100
            score_out_LUTO = row['NATURAL_OUT_LUTO_HA']
            result.loc[dict(layer=(region, species, presence))] = score_all_aus - score_out_LUTO
        return result


    def get_GBF8_bio_layers_by_yr(self, yr: int, level:Literal['species', 'group']='species'):
        '''
        Get the biodiversity suitability score [hectare weighted] for each species at the given year.
        
        The raw biodiversity suitability score [2D (shape, 808*978), (dtype, uint8, 0-100)] represents the
        suitability of each cell for each species/group. The spatial LINEAR interpolation to the cell
        centroids is precomputed once per (SSP, RESFACTOR) by `match_GBF8_bio_layers` and read from the
        `_cache` file; here only the LINEAR year interpolation runs.

        Because the coordinates are the controid of the `self.MASK` array, so the spatial interpolation is
        simultaneously a masking process.
        
        The suitability score is then weighted by the area (ha) of each cell. The area weighting is necessary 
        to ensure that the biodiversity suitability score will not be affected by different RESFACTOR (i.e., cell size) values.
        
        Parameters
        ----------
        yr : int
            The year for which to get the biodiversity suitability score.
        level : str, optional
            The level of the biodiversity suitability score, either 'species' or 'group'. The default is 'species'.
            
        Returns
        -------
        np.ndarray
            The biodiversity suitability score for each species at the given year.
        '''
        
        # The expensive spatial interpolation lives in a per-(SSP, RESFACTOR) disk cache (see
        # match_GBF8_bio_layers); here only the cheap year interpolation runs. Both steps are linear, so
        # cache-then-year-interp equals the original year-then-spatial order.
        with xr.open_dataarray(self.match_GBF8_bio_layers(level)) as layer:
            if level == 'species':
                layer = layer.sel(species=self.BIO_GBF8_SEL_SPECIES)    # the selection — any subset, any order

            # The two years a linear interpolation to `yr` reads, and the weight on the upper one; outside
            # the cached range the same two end points extrapolate (weight < 0 or > 1). Reading just the
            # pair and blending in float32 is what xr.interp would do, minus its 3-year window and float64.
            years = layer['year'].values
            i = int(np.clip(np.searchsorted(years, yr, side='right') - 1, 0, years.size - 2))
            w = np.float32((yr - years[i]) / (years[i + 1] - years[i]))
            lo, hi = layer.isel(year=[i, i + 1]).transpose('year', level, 'cell').values

        out = lo * (np.float32(1) - w) + hi * w

        # Apply Savanna Burning penalties (in place: one per-cell factor, broadcast over species)
        out *= np.where(self.SAVBURN_ELIGIBLE.values, np.float32(settings.BIO_CONTRIBUTION_LDS), np.float32(1))

        return out


    def match_GBF8_bio_layers(self, level: Literal['species', 'group'] = 'species') -> str:
        '''
        Interpolate the raw GBF8 rasters onto the cell centroids and cache that on disk, returning the
        cache path. It is the expensive part of `get_GBF8_bio_layers_by_yr` and turns only on SSP and
        RESFACTOR, so it is built once, over every species/group, and reused by any later run.
        '''
        import dask

        cache_path = os.path.join(
            settings.INPUT_DIR, '_cache', f'cache_bio_GBF8_ssp{settings.SSP}_RES{settings.RESFACTOR}_{level}.nc')
        if os.path.exists(cache_path):
            return cache_path

        print(f"│   ├── Building GBF8 {level} spatial-match cache (one-off per SSP/RESFACTOR)", flush=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw_path = (f'{settings.INPUT_DIR}/bio_GBF8_ssp{settings.SSP}_EnviroSuit.nc'
                    if level == 'species' else
                    f'{settings.INPUT_DIR}/bio_GBF8_ssp{settings.SSP}_EnviroSuit_group.nc')
        lon, lat = self.COORD_LON_LAT

        with dask.config.set(scheduler='synchronous'):                  # one chunk at a time: the raw stack is ~42 GB
            with xr.open_dataset(raw_path, chunks={'year': 1, level: 128}) as raw:
                matched = raw['data'].interp(
                    x=xr.DataArray(lon, dims='cell'),
                    y=xr.DataArray(lat, dims='cell'),
                    method='linear'                                     # Use LINEAR interpolation
                ).astype(np.float32)
                tmp_path = f'{cache_path}.tmp{os.getpid()}'
                matched.to_netcdf(tmp_path, encoding={'data': {
                    'zlib': True, 'complevel': 4, 'dtype': 'float32',
                    'chunksizes': (1, 1, self.NCELLS),                  # one species (group) per chunk: a subset reads only its own
                }})
        os.replace(tmp_path, cache_path)                                # atomic: a reader never sees a partial cache
        print(f"│   │   └── saved {os.path.basename(cache_path)}", flush=True)
        return cache_path


    def get_GBF8_target_inside_LUTO_by_yr(self, yr: int) -> xr.DataArray:
        '''
        Get the biodiversity suitability score (area weighted [ha]) for each species at the given year for the Inside LUTO natural land.
        Returns xr.DataArray with MultiIndex coord layer=(region, species).
        '''
        target_scores = self.get_GBF8_score_all_Australia_by_yr(yr) - self.get_GBF8_score_outside_natural_LUTO_by_yr(yr)
        return xr.DataArray(
            target_scores.astype(np.float32),
            dims=['layer'],
            coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF8_SEL, names=['region', 'species'])},
        )
    

    
    def get_GBF8_score_all_Australia_by_yr(self, yr: int):
        '''
        Get the biodiversity suitability score (area weighted [ha]) for each species at the given year for all Australia.
        '''
        # Get the target percentage for each species at the given year
        target_pct = []
        for _,row in self.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES.iterrows():
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [min(row['HABITAT_SUITABILITY_BASELINE_PERCENT'], row['TARGET_LEVEL_2030']), row[f'TARGET_LEVEL_2030'], row[f'TARGET_LEVEL_2050'], row[f'TARGET_LEVEL_2100']],
                kind="linear",
                fill_value="extrapolate",
            )
            target_pct.append(f(yr).item()) 
            
        # Calculate the target biodiversity suitability score for each species at the given year for all Australia
        target_scores_all_AUS = self.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES['HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA'] * (np.array(target_pct) / 100) # Convert the percentage to proportion
        return target_scores_all_AUS.values
    
    
    def get_GBF8_score_outside_natural_LUTO_by_yr(self, yr: int, level:Literal['species', 'group']='species'):
        '''
        Get the biodiversity suitability score (area weighted [ha]) for each species at the given year for the Outside LUTO natural land.
        '''
        
        if level == 'species':
            base_score = self.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES['HABITAT_SUITABILITY_BASELINE_SCORE_OUTSIDE_LUTO']
            proj_score = self.BIO_GBF8_OUTSDIE_LUTO_SCORE_SPECIES.pivot(index='species', columns='year').reset_index()
        elif level == 'group':
            base_score = self.BIO_GBF8_BASELINE_SCORE_GROUPS['HABITAT_SUITABILITY_BASELINE_SCORE_OUTSIDE_LUTO']
            proj_score = self.BIO_GBF8_OUTSDIE_LUTO_SCORE_GROUPS.pivot(index='group', columns='year').reset_index()
        else:
            raise ValueError("Invalid level. Must be 'species' or 'group'")
        
        # Put the base score to the proj_score
        proj_score.columns = proj_score.columns.droplevel() 
        proj_score[1990] = base_score.values
        
        # Interpolate the suitability score for each species/group at the given year
        outside_natural_scores = []
        for _,row in proj_score.iterrows():
            f = interp1d(
                [1990, 2030, 2050, 2070, 2090],
                [row[1990], row[2030], row[2050], row[2070], row[2090]],
                kind="linear",
                fill_value="extrapolate",
            )
            outside_natural_scores.append(f(yr).item())
        
        return  outside_natural_scores


    
    def get_regional_adoption_percent_by_year(self, yr: int):
        """
        Get the per-landuse regional adoption percentage for each region for the given year.
        Only active under REGIONAL_ADOPTION_CONSTRAINTS == 'individual_cap'.

        Return a list of tuples where each tuple contains 
        - the region ID, 
        - landuse name, 
        - the adoption percentage.
        """
        if settings.REGIONAL_ADOPTION_CONSTRAINTS != "individual_cap":
            return ()
        
        reg_adop_limits = []
        for _,row in self.REGIONAL_ADOPTION_TARGETS.iterrows():
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [row['BASE_LANDUSE_AREA_PERCENT'], row['ADOPTION_PERCENTAGE_2030'], row['ADOPTION_PERCENTAGE_2050'], row['ADOPTION_PERCENTAGE_2100']],
                kind="linear",
                fill_value="extrapolate",
            )
            reg_adop_limits.append((row[settings.REGIONAL_ADOPTION_ZONE], row['TARGET_LANDUSE'], f(yr).item()))
            
        return reg_adop_limits
    
    def get_regional_adoption_limit_ha_by_year(self, yr: int):
        """
        Get the per-landuse regional adoption area for each region for the given year.
        Only active under REGIONAL_ADOPTION_CONSTRAINTS == 'individual_cap'.

        Return a list of tuples where each tuple contains
        - the region ID,
        - landuse name,
        - the adoption area (ha).
        """
        if settings.REGIONAL_ADOPTION_CONSTRAINTS != "individual_cap":
            return ()
        
        reg_adop_limits = self.get_regional_adoption_percent_by_year(yr)
        reg_adop_limits_ha = []
        for reg, landuse, pct in reg_adop_limits:
            reg_total_area_ha = ((self.REGIONAL_ADOPTION_ZONES.values == reg) * self.REAL_AREA.values).sum()
            reg_adop_limits_ha.append((reg, landuse, reg_total_area_ha * pct / 100))
            
        return reg_adop_limits_ha

    def get_regional_adoption_non_ag_sum_limit_ha_by_year(self, yr: int):
        """
        Under REGIONAL_ADOPTION_CONSTRAINTS == 'non_ag_sum_cap', return per-region area caps
        (ha) on the SUM of all non-ag land uses.

        The region partition is selected by REGIONAL_ADOPTION_NON_AG_REGION:
        - 'NRM'   uses self.REGION_NRM_NAME
        - 'State' uses self.REGION_STATE_NAME

        Returns a list of (region_name, reg_ind, area_limit_ha) tuples. The percentage
        cap (REGIONAL_ADOPTION_NON_AG_CAP) is uniform across all regions and years.
        """
        if settings.REGIONAL_ADOPTION_CONSTRAINTS != 'non_ag_sum_cap':
            return ()
        if settings.REGIONAL_ADOPTION_NON_AG_CAP is None:
            return ()

        region_mode = settings.REGIONAL_ADOPTION_NON_AG_REGION
        if region_mode == 'NRM':
            region_arr = np.asarray(self.REGION_NRM_NAME)
        elif region_mode == 'State':
            region_arr = np.asarray(self.REGION_STATE_NAME)
        else:
            raise ValueError(
                f"Unknown REGIONAL_ADOPTION_NON_AG_REGION={region_mode!r}. Expected 'NRM' or 'State'."
            )

        pct_default = settings.REGIONAL_ADOPTION_NON_AG_CAP
        sel = settings.REGIONAL_ADOPTION_NON_AG_CAP_REGIONS          # [] -> all regions
        override = settings.REGIONAL_ADOPTION_NON_AG_CAP_OVERRIDE    # {} -> uniform cap for all
        limits = []
        for reg in np.unique(region_arr):
            if sel and reg not in sel:                 # scope: only cap the named regions
                continue
            pct = override.get(reg, pct_default)        # per-region override, else the uniform cap
            if pct is None:
                continue
            reg_ind = np.where(region_arr == reg)[0]
            reg_total_area_ha = self.REAL_AREA.values[reg_ind].sum()
            limits.append((reg, reg_ind, reg_total_area_ha * pct / 100))
        return limits


    def add_production_data(self, yr: int, data_type: str, prod_data: Any):
        """
        Safely save production data for a given year to the Data object.

        Parameters
        ----
        yr: int
            Year of production data being saved.
        data_type: str
            Type of production data being saved. Typically either 'Production', 'GHG Emissions' or 'Biodiversity'.
        prod_data: Any
            Actual production data to save.
        """
        if yr not in self.prod_data:
            self.prod_data[yr] = {}
        self.prod_data[yr][data_type] = prod_data

    def add_obj_vals(self, yr: int, obj_val: float):
        """
        Safely save objective value for a given year to the Data object
        """
        self.obj_vals[yr] = obj_val



    def get_carbon_price_by_yr_idx(self, yr_idx: int) -> float:
        """
        Return the price of carbon per tonne for a given year index (since 2010).
        The resulting year should be between 2010 - 2100
        """
        yr_cal = yr_idx + self.YR_CAL_BASE
        return self.get_carbon_price_by_year(yr_cal)

    def get_carbon_price_by_year(self, yr_cal: int) -> float:
        """
        Return the price of carbon per tonne for a given year.
        The resulting year should be between 2010 - 2100
        """
        if yr_cal not in self.CARBON_PRICES:
            raise ValueError(
                f"Carbon price data not given for the given year: {yr_cal}. "
                f"Year should be between {self.YR_CAL_BASE} and 2100."
            )
        return self.CARBON_PRICES[yr_cal]

    def get_water_yield(self, root: str, yr_cal: int) -> xr.DataArray:
        """The water yield of this SSP in one year at the model cells, ML/ha over cell (float32); root 'DR' = deep-rooted,
        'SR' = shallow-rooted. Read from the file the first time a year is asked for — one chunk per variable (the year's
        every cell), then the model cells — and kept: the water targets (in __init__) and every solve step ask for the
        simulated years again, so each is read once; memory is the simulated years only, not all 91."""
        if yr_cal not in self.WATER_YIELD_BY_YEAR:
            with xr.open_dataset(self.WATER_YIELD_SSP_FILE) as water_yield:
                year = water_yield[['DR_ML_HA', 'SR_ML_HA']].sel(year=yr_cal).load()
            self.WATER_YIELD_BY_YEAR[yr_cal] = year.isel(cell=self.MASK).assign_coords(cell=np.arange(self.NCELLS))
        return self.WATER_YIELD_BY_YEAR[yr_cal][f'{root}_ML_HA']

    def get_water_nl_yield_for_yr_idx(
        self,
        yr_idx: int,
        water_dr_yield: Optional[xr.DataArray] = None,
        water_sr_yield: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """
        Get the net land water yield array, ? inclusive of all cells that LUTO does not look at.?

        Returns
        -------
        xr.DataArray: over cell (NCELLS)
        """
        water_dr_yield = (
            water_dr_yield if water_dr_yield is not None
            else self.get_water_yield('DR', self.YR_CAL_BASE + yr_idx)
        )
        water_sr_yield = (
            water_sr_yield if water_sr_yield is not None
            else self.get_water_yield('SR', self.YR_CAL_BASE + yr_idx)
        )
        dr_prop = self.DEEP_ROOTED_PROPORTION

        return (dr_prop * water_dr_yield + (1 - dr_prop) * water_sr_yield)

    def load_bio_quality_layer(self, backend_layer: str, biodiv_ds=None) -> tuple[np.ndarray, str]:
        """Return (raw priority values on MASK, performance-curve sheet name) for a backend layer.

        The values are *unweighted* — connectivity is applied by the callers that want it.
        The GBF2 and renewable exclusion masks deliberately threshold these unweighted values,
        because the thresholds in Biodiversity_conserve_performance.xlsx are derived from the
        unweighted layers too.

        `biodiv_ds` lets __init__ pass the priority/connectivity Dataset it has already read (at the model cells),
        avoiding a second read of the file for the 'Suitability' backend.
        """
        if backend_layer == 'Suitability':
            if biodiv_ds is None:
                with xr.open_dataset(os.path.join(settings.INPUT_DIR, 'bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.nc')) as bio_priority:
                    biodiv_ds = bio_priority[['BIODIV_PRIORITY']].sel(ssp=[settings.SSP]).load().isel(cell=self.MASK)
            return biodiv_ds['BIODIV_PRIORITY'].sel(ssp=settings.SSP).values.astype(np.float32), f'ssp{settings.SSP}'

        if 'NES' in backend_layer:
            with xr.open_dataarray(os.path.join(settings.INPUT_DIR, 'bio_NES_Zonation.nc')) as nes_zonation:
                return nes_zonation.sel(layer=backend_layer).values[self.MASK].astype(np.float32), backend_layer

        if backend_layer == 'RHI':
            return (
                xr.load_dataarray(os.path.join(settings.INPUT_DIR, 'bio_RHI_Zonation.nc'))
                .values[self.MASK]
                .astype(np.float32)
            ), 'RHI'

        raise ValueError(
            f"Invalid biodiversity quality layer '{backend_layer}': must be 'Suitability', 'RHI', "
            f"or a '*NES_likely|may' layer."
        )

