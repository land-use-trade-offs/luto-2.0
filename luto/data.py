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
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import geopandas as gpd
import netCDF4 # necessary for running luto in Denethor

from luto import tools
import luto.settings as settings
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.non_agricultural.quantity as non_ag_quantity
import luto.economics.agricultural.water as ag_water
from luto.tools.Manual_jupyter_books.helpers import arr_to_xr

from collections import defaultdict
from typing import Any, Literal, Optional
from affine import Affine
from scipy.interpolate import interp1d
from math import ceil
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt



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
        self.ag_man_dvars = {}
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
        self.LUMAP_NO_RESFACTOR = pd.read_hdf(os.path.join(settings.INPUT_DIR, "lumap.h5")).to_numpy().astype(np.int8)   # 1D (ij flattend),  0-27 for land uses; -1 for non-agricultural land uses; All cells in Australia (land only)

        # NLUM mask.
        with rasterio.open(os.path.join(settings.INPUT_DIR, "NLUM_2010-11_mask.tif")) as rst:
            self.NLUM_MASK = rst.read(1).astype(np.int8)                                                                # 2D map,  0 for ocean, 1 for land
            self.LUMAP_2D_FULLRES = np.full_like(self.NLUM_MASK, self.NODATA, dtype=np.int16)                           # 2D map,  full of nodata (-9999)
            np.place(self.LUMAP_2D_FULLRES, self.NLUM_MASK == 1, self.LUMAP_NO_RESFACTOR)                               # 2D map,  -9999 for ocean; -1 for desert, urban, water, etc; 0-27 for land uses
            self.GEO_META_FULLRES = rst.meta                                                                            # dict,  key-value pairs of geospatial metadata for the full resolution land-use map
            self.GEO_META_FULLRES['dtype'] = 'float32'                                                                  # Set the data type to float32
            self.GEO_META_FULLRES['nodata'] = self.NODATA                                                               # Set the nodata value to -9999


        # Mask out non-agricultural, non-environmental plantings land (i.e., -1) from lumap 
        self.LUMASK = self.LUMAP_NO_RESFACTOR != self.MASK_LU_CODE                                                      # 1D (ij flattend);  `True` for land uses; `False` for desert, urban, water, etc
        self.LUMASK_2D_FULLRES = np.nan_to_num(arr_to_xr(self, self.LUMASK))
        

        # Return combined land-use and resfactor mask
        if settings.RESFACTOR > 1:
            
            # Get 2D coarsed array, where True means the res*res neighbourhood having >=1 land-use cells
            rf_mask = self.NLUM_MASK.copy()
            have_lu_cells = xr.DataArray(self.LUMASK_2D_FULLRES, dims=['y', 'x']).coarsen(y=settings.RESFACTOR, x=settings.RESFACTOR, boundary='pad').sum()
            have_lu_cells = np.repeat(np.repeat(have_lu_cells.data, settings.RESFACTOR, axis=0), settings.RESFACTOR, axis=1)
            have_lu_cells = have_lu_cells[:self.LUMASK_2D_FULLRES.shape[0], :self.LUMASK_2D_FULLRES.shape[1]]
            have_lu_cell_downsampled = have_lu_cells[settings.RESFACTOR//2::settings.RESFACTOR, settings.RESFACTOR//2::settings.RESFACTOR]

            # Get 2D fullres array, where True means 
            #   - the cell is the center of a res*res neighbourhood AND
            #   - having >=1 land-use cells
            lu_mask_fullres = np.zeros_like(rf_mask, dtype=bool)
            lu_mask_fullres[settings.RESFACTOR//2::settings.RESFACTOR, settings.RESFACTOR//2::settings.RESFACTOR] = have_lu_cell_downsampled

            # Get the coords (row, col) of the cells that are the center of a res*res neighbourhood having >=1 land-use cells
            self.COORD_ROW_COL_FULLRES = np.argwhere(rf_mask & lu_mask_fullres).T
            self.COORD_ROW_COL_RESFACTORED = (self.COORD_ROW_COL_FULLRES - (settings.RESFACTOR//2)) // settings.RESFACTOR
            
            # Get the 1D MASK for resfactoring all input datasets
            #   - the length is the number cells in Australia (land only).
            #   - the True values are the center of a res*res neighbourhood having >=1 land-use cells
            rf_mask[self.COORD_ROW_COL_FULLRES[0], self.COORD_ROW_COL_FULLRES[1]] = 2
            self.MASK = np.where(rf_mask[np.nonzero(self.NLUM_MASK)] == 2, True, False)
            
            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES[settings.RESFACTOR//2::settings.RESFACTOR, settings.RESFACTOR//2::settings.RESFACTOR]
            self.GEO_META = self.update_geo_meta()
            
        elif settings.RESFACTOR == 1:
            self.MASK = self.LUMASK
            self.GEO_META = self.GEO_META_FULLRES
            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES
            self.COORD_ROW_COL_FULLRES = np.argwhere(self.NLUM_MASK == 1).T
            self.COORD_ROW_COL_RESFACTORED = self.COORD_ROW_COL_FULLRES
        else:
            raise KeyError("Resfactor setting invalid")
        
        # Get the 2D MASK
        self.MASK_2D = np.nan_to_num(arr_to_xr(self, self.MASK))
        
        
        # Get the lon/lat coordinates.
        self.COORD_LON_LAT_2D_FULLRES = self.get_coord(np.nonzero(self.NLUM_MASK), self.GEO_META_FULLRES['transform'])     # 2D array([lon, ...], [lat, ...]);  lon/lat coordinates for each cell in Australia (land only)
        self.COORD_LON_LAT = [i[self.MASK] for i in self.COORD_LON_LAT_2D_FULLRES]  # Only keep the coordinates for the cells that are not masked out (i.e., land uses). 2D array([lon, ...], [lat, ...]);  lon/lat coordinates for each cell in Australia (land only) and not masked out
        
        

        ###############################################################
        # Load agricultural crop and livestock economic and yield data.
        ###############################################################
        print("├── Loading agricultural crop and livestock data", flush=True)
        self.AGEC_CROPS = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agec_crops.h5"), where=self.MASK)
        self.AGEC_LVSTK = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agec_lvstk.h5"), where=self.MASK)
        
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
        self.LANDMANS = {t[1] for t in self.AGEC_CROPS.columns}  # Set comp., unique entries.
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
        self.TRANS_COST_MULTS = pd.read_excel(cost_mult_excel, "Transitions cost multiplier", index_col="Year")["Transitions_cost_multiplier"].to_dict()
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
        self.REAL_AREA_NO_RESFACTOR = pd.read_hdf(os.path.join(settings.INPUT_DIR, "real_area.h5")).to_numpy()
        self.REAL_AREA = self.REAL_AREA_NO_RESFACTOR[self.MASK] * self.RESMULT  # TODO: adjusting using actual area

        # Derive NCELLS (number of spatial cells) from the area array.
        self.NCELLS = self.REAL_AREA.shape[0]
        
        # Initial (2010) ag decision variable (X_mrj).
        self.LMMAP_NO_RESFACTOR = pd.read_hdf(os.path.join(settings.INPUT_DIR, "lmmap.h5")).to_numpy()
        self.AG_L_MRJ = self.get_exact_resfactored_lumap_mrj()
        self.add_ag_dvars(self.YR_CAL_BASE, self.AG_L_MRJ)
        
        # Initial (2010) maximum ag dvar proportion
        #   For example, if a cell has 0.2 of ag land-use at beginning,
        #   then, the sum(ag + non_ag) in the following years should be <= 0.2.
        #   This is used as a constraint in the solver to prevent the model 
        #   from allocating more agricultural land than the initial proportion.
        self.AG_MASK_PROPORTION_R =  self.AG_L_MRJ.sum(0).sum(1)

        # Initial (2010) land-use map, mapped as lexicographic land-use class indices.
        self.LU_RESFACTOR_CELLS = pd.DataFrame({
            'lu_code': list(self.DESC2AGLU.values()),
            'res_size': [ceil((self.LUMAP_NO_RESFACTOR == lu_code).sum() / self.RESMULT) for _,lu_code in self.DESC2AGLU.items()]
        }).sort_values('res_size').reset_index(drop=True)
        
        self.LUMAP = self.get_resfactored_lumap() if settings.RESFACTOR > 1 else self.LUMAP_NO_RESFACTOR[self.MASK]
        self.add_lumap(self.YR_CAL_BASE, self.LUMAP)

        # Initial (2010) land management map.
        self.LMMAP = self.LMMAP_NO_RESFACTOR[self.MASK]
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

        self.CLIMATE_CHANGE_IMPACT = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "climate_change_impacts_" + settings.RCP + "_CO2_FERT_" + settings.CO2_FERT.upper() + ".h5"), where=self.MASK
        )
        
        # Convert to xarray DataArray for easier indexing.
        #   The YRS_CAL_BASE (2010) is not included in the climate change impact data, 
        #   so we add it here with a multiplier of 1.
        self.CLIMATE_CHANGE_IMPACT_xr = (
                xr.DataArray(self.CLIMATE_CHANGE_IMPACT)
                .unstack('dim_1')
                .rename({'dim_1_level_0':'lm', 'dim_1_level_1':'lu', 'dim_1_level_2':'year', 'CELL_ID':'cell'})
                .assign_coords(cell=range(self.NCELLS))     # Cell index from now will be from 0 to NCELLS-1
            )
        
        CLIMATE_CHANGE_IMPACT_xr_base_year = (
            self.CLIMATE_CHANGE_IMPACT_xr
            .isel(year=0, drop=True)
            .assign_coords(year=self.YR_CAL_BASE)
            .expand_dims('year')
            .notnull()
            .astype(np.float32)
        )
        
        self.CLIMATE_CHANGE_IMPACT_xr = xr.concat( 
            [CLIMATE_CHANGE_IMPACT_xr_base_year,self.CLIMATE_CHANGE_IMPACT_xr],
            dim='year'
        )
        
        ###############################################################
        # Regional coverage layers, mainly for regional reporting.
        ###############################################################
        REGION_NRM_r = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "REGION_NRM_r.h5"), where=self.MASK
        )        
        
        self.REGION_NRM_CODE = REGION_NRM_r['NRM_CODE']
        self.REGION_NRM_NAME = REGION_NRM_r['NRM_NAME'].values
        
        REGION_STATE_r = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "REGION_STATE_r.h5"), where=self.MASK
        )
        
        self.REGION_STATE_NAME2CODE = REGION_STATE_r.groupby('STE_NAME11', observed=True)['STE_CODE11'].first().to_dict()
        self.REGION_STATE_NAME2CODE = dict(sorted(self.REGION_STATE_NAME2CODE.items()))     # Make sure the dict is sorted by state name, makes it consistent with renewable target.
        
        if 'Other Territories' in self.REGION_STATE_NAME2CODE:
            self.REGION_STATE_NAME2CODE.pop('Other Territories')                            # Remove 'Other Territories' from the dict.
        
        self.REGION_STATE_CODE = REGION_STATE_r['STE_CODE11'].values
        self.REGION_STATE_NAME = REGION_STATE_r['STE_NAME11'].values.to_numpy()


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

        elif settings.REGIONAL_ADOPTION_CONSTRAINTS == "on":
            # Per-landuse caps (both ag and non-ag) read from the xlsx, scoped to REGIONAL_ADOPTION_ZONE.
            self.REGIONAL_ADOPTION_ZONES = pd.read_hdf(
                os.path.join(settings.INPUT_DIR, "regional_adoption_zones.h5"), where=self.MASK
            )[settings.REGIONAL_ADOPTION_ZONE].to_numpy()

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
                np.unique(self.REGIONAL_ADOPTION_ZONES)
            ) if len(self.REGIONAL_ADOPTION_TARGETS) > 0 else np.array([])

            if len(lost_zones) > 0:
                print(f"│   ⚠ WARNING: {len(lost_zones)} regional adoption zones have no cells due to (RES{settings.RESFACTOR}). Please check if this is expected.", flush=True)
                self.REGIONAL_ADOPTION_TARGETS = self.REGIONAL_ADOPTION_TARGETS.query(
                    f"{settings.REGIONAL_ADOPTION_ZONE} not in {list(lost_zones)}"
                ).reset_index(drop=True)

        elif settings.REGIONAL_ADOPTION_CONSTRAINTS == "NON_AG_CAP":
            # SUM-of-all-non-ag cap per region (NRM or State, controlled by REGIONAL_ADOPTION_NON_AG_REGION).
            # No xlsx involved; ag dvars are unconstrained by this mode.
            self.REGIONAL_ADOPTION_ZONES = None
            self.REGIONAL_ADOPTION_TARGETS = None

        else:
            raise ValueError(
                f"Unknown REGIONAL_ADOPTION_CONSTRAINTS={settings.REGIONAL_ADOPTION_CONSTRAINTS!r}. "
                "Expected one of: 'off', 'on', 'NON_AG_CAP'."
            )



        ###############################################################
        # Livestock related data.
        ###############################################################
        print("├── Loading livestock related data", flush=True)

        self.FEED_REQ = np.nan_to_num(
            pd.read_hdf(os.path.join(settings.INPUT_DIR, "feed_req.h5"), where=self.MASK).to_numpy()
        )
        self.PASTURE_KG_DM_HA = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "pasture_kg_dm_ha.h5"), where=self.MASK
        ).to_numpy()
        self.SAFE_PUR_NATL = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "safe_pur_natl.h5"), where=self.MASK
        ).to_numpy()
        self.SAFE_PUR_MODL = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "safe_pur_modl.h5"), where=self.MASK
        ).to_numpy()



        ###############################################################
        # Agricultural Management options data.
        ###############################################################
        print("├── Loading agricultural management options' data", flush=True)
        
        # Rename soil CO2E to match the AGGHG_CROPS/LVSTK data.
        _bundle_rename = {"CO2E_KG_HA_SOIL_N_SURP": "CO2E_KG_HA_SOIL"}


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
            "horticulture": pd.read_excel(prec_agr_file, sheet_name="AgTech NE bundle (horticulture)", index_col="Year").rename(columns=_bundle_rename),
        }

        # Load AgTech EI data
        agtech_ei_file = os.path.join(settings.INPUT_DIR, '20260317_Bundle_AgTech_EI.xlsx')
        self.AGTECH_EI_DATA = {
            "cropping":     pd.read_excel(agtech_ei_file, sheet_name='AgTech EI bundle (cropping)', index_col='Year'),
            "int_cropping": pd.read_excel(agtech_ei_file, sheet_name='AgTech EI bundle (int cropping)', index_col='Year', usecols=lambda c: not str(c).startswith('Unnamed')),
            "horticulture": pd.read_excel(agtech_ei_file, sheet_name='AgTech EI bundle (horticulture)', index_col='Year').rename(columns=_bundle_rename),
        }

        # Load BioChar data (no int_cropping group)
        biochar_file = os.path.join(settings.INPUT_DIR, '20260401_Bundle_BC.xlsx')
        self.BIOCHAR_DATA = {
            "cropping":     pd.read_excel(biochar_file, sheet_name='Biochar (cropping)', index_col='Year').rename(columns=_bundle_rename),
            "horticulture": pd.read_excel(biochar_file, sheet_name='Biochar (horticulture)', index_col='Year').rename(columns=_bundle_rename),
        }
        
        # Load soil carbon data, convert C to CO2e (x 44/12), and average over years
        self.SOIL_CARBON_AVG_T_CO2_HA_PER_YR = (
            pd.read_hdf(os.path.join(settings.INPUT_DIR, "soil_carbon_t_ha.h5"), where=self.MASK).to_numpy(dtype=np.float32) 
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
                .query('scen == @settings.RENEWABLE_TARGET_SCENARIO_TARGETS')
                .assign(Renewable_Target_MWh=lambda df: df['Renewable_Target_TWh'] * 1e6)
            )

            self.SOLAR_PRICES = pd.read_csv(f'{settings.INPUT_DIR}/renewable_price_AUD_MWh_solar.csv')
            self.WIND_PRICES = pd.read_csv(f'{settings.INPUT_DIR}/renewable_price_AUD_MWh_wind.csv')

            # Renewable energy related raster layers
            #   Compute before isel to avoid isel working on dask chunks
            #   which slows down the reading.
            self.RENEWABLE_LAYERS = (
                xr.open_dataset(f'{settings.INPUT_DIR}/renewable_energy_layers_1D.nc')
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
        self.STREAM_LENGTH = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "stream_length_m_cell.h5"), where=self.MASK
        ).to_numpy()

        # Calculate the proportion of the area of each cell within stream buffer (convert REAL_AREA from ha to m2 and divide m2 by m2)
        self.RP_PROPORTION =  (
            (2 * settings.RIPARIAN_PLANTING_BUFFER_WIDTH * self.STREAM_LENGTH) / (self.REAL_AREA_NO_RESFACTOR[self.MASK] * 10000)
        ).astype(np.float32)
        # Calculate the length of fencing required for each cell in per hectare terms for riparian plantings
        self.RP_FENCING_LENGTH = (
            (2 * settings.RIPARIAN_PLANTING_TORTUOSITY_FACTOR * self.STREAM_LENGTH) / self.REAL_AREA_NO_RESFACTOR[self.MASK]
        ).astype(np.float32)



        ###############################################################
        # Additional agricultural GHG data.
        ###############################################################
        print("├── Loading additional agricultural GHG data", flush=True)


        # Load greenhouse gas emissions from agriculture
        self.AGGHG_CROPS = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agGHG_crops.h5"), where=self.MASK)
        self.AGGHG_LVSTK = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agGHG_lvstk.h5"), where=self.MASK)
        self.AGGHG_IRRPAST = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agGHG_irrpast.h5"), where=self.MASK)


        # Raw transition cost matrix. In AUD/ha and ordered lexicographically.
        self.AG_TMATRIX = np.load(os.path.join(settings.INPUT_DIR, "ag_tmatrix.npy"))
        self.AG_TO_DESTOCKED_NATURAL_COSTS_HA = np.load(os.path.join(settings.INPUT_DIR, "ag_to_destock_tmatrix.npy"))

        
  
        # Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
        self.EXCLUDE = np.load(os.path.join(settings.INPUT_DIR, "x_mrj.npy"))
        self.EXCLUDE = self.EXCLUDE[:, self.MASK, :]  # Apply resfactor specially for the exclude matrix



        ###############################################################
        # Non-agricultural data.
        ###############################################################
        print("├── Loading non-agricultural data", flush=True)

        # Load plantings economic data
        self.EP_EST_COST_HA = pd.read_hdf(os.path.join(settings.INPUT_DIR, "ep_est_cost_ha.h5"), where=self.MASK).to_numpy(dtype=np.float32)
        self.RP_EST_COST_HA = self.EP_EST_COST_HA.copy()  # Riparian plantings have the same establishment cost as environmental plantings
        self.AF_EST_COST_HA = self.EP_EST_COST_HA.copy()  # Agroforestry plantings have the same establishment cost as environmental plantings
        self.CP_EST_COST_HA = pd.read_hdf(os.path.join(settings.INPUT_DIR, "cp_est_cost_ha.h5"), where=self.MASK).to_numpy(dtype=np.float32)

        # Load fire risk data (reduced carbon sequestration by this amount)
        fr_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, "fire_risk.h5"), where=self.MASK)
        fr_dict = {"low": "FD_RISK_PERC_5TH", "med": "FD_RISK_MEDIAN", "high": "FD_RISK_PERC_95TH"}
        fire_risk = fr_df[fr_dict[settings.FIRE_RISK]]


        # Load environmental plantings (block) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        #   NOTE: Use .sel(age=...).load().isel(cell=self.MASK) instead of .sel(cell=self.MASK) 
        #   to avoid slow xarray label-based indexing on large cell dimensions
        ds = xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_block.nc")).sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK)
        self.EP_BLOCK_AVG_T_CO2_HA_PER_YR = (
            (ds['EP_BLOCK_TREES_T_CO2_HA'] + ds['EP_BLOCK_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['EP_BLOCK_SOIL_T_CO2_HA']
        ).values / settings.CARBON_EFFECTS_WINDOW

        # Load environmental plantings (belt) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ds = xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_belt.nc")).sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK)
        self.EP_BELT_AVG_T_CO2_HA_PER_YR = (
            (ds['EP_BELT_TREES_T_CO2_HA'] + ds['EP_BELT_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['EP_BELT_SOIL_T_CO2_HA']
        ).values / settings.CARBON_EFFECTS_WINDOW

        # Load environmental plantings (riparian) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ds = xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_rip.nc")).sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK)
        self.EP_RIP_AVG_T_CO2_HA_PER_YR = (
            (ds['EP_RIP_TREES_T_CO2_HA'] + ds['EP_RIP_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['EP_RIP_SOIL_T_CO2_HA']
        ).values / settings.CARBON_EFFECTS_WINDOW

        # Load carbon plantings (block) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ds = xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_cp_block.nc")).sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK)
        self.CP_BLOCK_AVG_T_CO2_HA_PER_YR = (
            (ds['CP_BLOCK_TREES_T_CO2_HA'] + ds['CP_BLOCK_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['CP_BLOCK_SOIL_T_CO2_HA']
        ).values / settings.CARBON_EFFECTS_WINDOW

        # Load farm forestry [i.e. carbon plantings (belt)] GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ds = xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_cp_belt.nc")).sel(age=settings.CARBON_EFFECTS_WINDOW).load().isel(cell=self.MASK)
        self.CP_BELT_AVG_T_CO2_HA_PER_YR = (
            (ds['CP_BELT_TREES_T_CO2_HA'] + ds['CP_BELT_DEBRIS_T_CO2_HA'])
            * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ds['CP_BELT_SOIL_T_CO2_HA']
        ).values / settings.CARBON_EFFECTS_WINDOW

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
                wreq_lvstk_dry[lu] = self.AGEC_LVSTK["WR_DRN", animal] * settings.LIVESTOCK_DRINKING_WATER
                wreq_lvstk_irr[lu] = (
                    self.AGEC_LVSTK["WR_IRR", animal] + self.AGEC_LVSTK["WR_DRN", animal] * settings.LIVESTOCK_DRINKING_WATER
                )
            else:
                wreq_lvstk_dry[lu] = 0.0
                wreq_lvstk_irr[lu] = 0.0

        # Water requirements by land use -- CROPS.
        wreq_crops_irr = pd.DataFrame()

        # The rj-indexed arrays have zeroes where j is not a crop.
        for lu in self.AGRICULTURAL_LANDUSES:
            if lu in self.LU_CROPS:
                wreq_crops_irr[lu] = self.AGEC_CROPS["WR", "irr", lu]
            else:
                wreq_crops_irr[lu] = 0.0

        # Add together as they have nans where not lvstk/crops
        self.WREQ_DRY_RJ = np.nan_to_num(wreq_lvstk_dry.to_numpy(dtype=np.float32))
        self.WREQ_IRR_RJ = np.nan_to_num(wreq_crops_irr.to_numpy(dtype=np.float32)) + np.nan_to_num(
            wreq_lvstk_irr.to_numpy(dtype=np.float32)
        )

        # Spatially explicit costs of a water licence per ML.
        self.WATER_LICENCE_PRICE = np.nan_to_num(
                pd.read_hdf(os.path.join(settings.INPUT_DIR, "water_licence_price.h5"), where=self.MASK).to_numpy()
        )

        # Spatially explicit costs of water delivery per ML.
        self.WATER_DELIVERY_PRICE = np.nan_to_num(
                pd.read_hdf(os.path.join(settings.INPUT_DIR, "water_delivery_price.h5"), where=self.MASK).to_numpy()
        )
       

        # Water yields -- run off from a cell into catchment by deep-rooted, shallow-rooted, and natural land
        water_yield_baselines = pd.read_hdf(os.path.join(settings.INPUT_DIR, "water_yield_baselines.h5"), where=self.MASK)
        self.WATER_YIELD_HIST_DR = water_yield_baselines['WATER_YIELD_HIST_DR_ML_HA'].to_numpy(dtype = np.float32)
        self.WATER_YIELD_HIST_SR = water_yield_baselines["WATER_YIELD_HIST_SR_ML_HA"].to_numpy(dtype = np.float32)
        self.DEEP_ROOTED_PROPORTION = water_yield_baselines['DEEP_ROOTED_PROPORTION'].to_numpy(dtype = np.float32)
        self.WATER_YIELD_HIST_NL = water_yield_baselines.eval(
            'WATER_YIELD_HIST_DR_ML_HA * DEEP_ROOTED_PROPORTION + WATER_YIELD_HIST_SR_ML_HA * (1 - DEEP_ROOTED_PROPORTION)'
        ).to_numpy(dtype = np.float32)

        wyield_fname_dr = os.path.join(settings.INPUT_DIR, 'water_yield_ssp' + str(settings.SSP) + '_2010-2100_dr_ml_ha.h5')
        wyield_fname_sr = os.path.join(settings.INPUT_DIR, 'water_yield_ssp' + str(settings.SSP) + '_2010-2100_sr_ml_ha.h5')
        
        # Read water yield data
        self.WATER_YIELD_DR_FILE = pd.read_hdf(wyield_fname_dr, where=self.MASK).T.values
        self.WATER_YIELD_SR_FILE = pd.read_hdf(wyield_fname_sr, where=self.MASK).T.values
        
        
        # Water yield from outside LUTO study area.
        self.WATER_YIELD_OUTSIDE_LUTO_HIST = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_outside_LUTO_study_area_hist_1970_2000.h5'))
        
        # Water use for domestic and industrial sectors.
        water_use_domestic = pd.read_csv(os.path.join(settings.INPUT_DIR, "Water_Use_Domestic.csv")).query('REGION_TYPE == @settings.WATER_REGION_DEF')
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
        nat_land_CO2 = pd.read_hdf(os.path.join(settings.INPUT_DIR, "natural_land_t_co2_ha.h5"), where=self.MASK)
        
        # Get the carbon stock of unallowcated natural land
        self.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR = np.array(
            nat_land_CO2['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA'] - (nat_land_CO2['NATURAL_LAND_AGB_DEBRIS_TCO2_HA'] * (100 - fire_risk) / 100),  # everyting minus the fire DAMAGE
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
        ) = self.get_actual_production_lyr(self.YR_CAL_BASE)

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
        
        # Load demand multiplier data
        AusTIME_multipliers = pd.read_excel(
                f'{settings.INPUT_DIR}/AusTIMES_demand_multiplier.xlsx', 
                sheet_name=settings.GHG_TARGETS_DICT[settings.GHG_EMISSIONS_LIMITS].split(' ')[0] + ' Demand', 
                index_col=0,
            ).T.reset_index(drop=True
            ).rename(columns={
                'Sorghum ':'winter cereals',    # the majority of winter cereals is Sorghum in Australia, so we map it this way
                'Canola':'winter oilseeds',     # the majority of winter oilseeds is Canola in Australia, so we map it this way
                'Sugar': 'sugar'
            }).drop(columns=['Cottonseed']      # LUTO do not have cottonseed in our model
            ).astype({'Year': int}
            ).set_index('Year')
            
        demand_multipliers = pd.DataFrame(
            index=range(2010, AusTIME_multipliers.index.max() + 1),
            columns=self.COMMODITIES,
            data=1.0
        )
        
        demand_multipliers.loc[
            AusTIME_multipliers.index,
            AusTIME_multipliers.columns
        ] = AusTIME_multipliers
        
        demand_multipliers = demand_multipliers.T
        demand_multipliers.columns.name = 'YEAR'
        demand_multipliers.index.name = 'COMMODITY'

        # Cache for elasticity multipliers (keyed by yr_cal)
        self.DEMAND_ELASTICITY_MUL = {}

        # Load demand data (actual production (tonnes, ML) by commodity) - from demand model
        dd = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'demand_projections.h5'))
        
        # Select the demand data under the running scenariobbryan-January
        self.DEMAND_DATA = dd.loc[(
            settings.SCENARIO,
            settings.DIET_DOM,
            settings.DIET_GLOB,
            settings.CONVERGENCE,
            settings.IMPORT_TREND,
            settings.WASTE,
            settings.FEED_EFFICIENCY)
        ].copy()

        # Convert eggs from count to tonnes
        self.DEMAND_DATA.loc['eggs'] = self.DEMAND_DATA.loc['eggs'] * settings.EGGS_AVG_WEIGHT / 1000 / 1000

        # Get the off-land commodities
        self.DEMAND_OFFLAND = self.DEMAND_DATA.loc[self.DEMAND_DATA.query("COMMODITY in @settings.OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()

        # Remove off-land commodities
        self.DEMAND_C = self.DEMAND_DATA.loc[self.DEMAND_DATA.query("COMMODITY not in @settings.OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()
        
        
        if settings.APPLY_DEMAND_MULTIPLIERS:
            print(f"│   ├── Years before applying demand multipliers: {self.DEMAND_C.columns.min()} - {self.DEMAND_C.columns.max()}", flush=True)
            self.DEMAND_C = (self.DEMAND_C *  demand_multipliers).dropna(axis=1)
            print(f"│   └── Years after applying demand multipliers: {self.DEMAND_C.columns.min()} - {self.DEMAND_C.columns.max()}", flush=True)
        
        # Convert to numpy array of shape (91, 26)
        self.D_CY = self.DEMAND_C.to_numpy(dtype = np.float32).T
        self.D_CY_xr = xr.DataArray(
            self.D_CY, 
            dims=['year','Commodity'], 
            coords={
                'year':self.YR_CAL_BASE + np.arange(self.D_CY.shape[0]), 
                'Commodity':self.COMMODITIES
            }
        ) 

        # Price elasticity data
        demand_supply_elasticity = pd.read_csv(f'{settings.INPUT_DIR}/demand_elasticity.csv')
        demand_supply_elasticity = demand_supply_elasticity.sort_values('Commodity')    # Sort by commodity to ensure the order is correct
        demand_supply_elasticity['Demand Elasticity(ED)'] = demand_supply_elasticity['Demand Elasticity(ED)'] * -1 # ED is subtracted from supply becase demand curve are given as negative numbers
        
        self.elasticity_demand = demand_supply_elasticity['Demand Elasticity(ED)']
        self.elasticity_supply = demand_supply_elasticity['Supply Elasticity (Es)']

   


        ###############################################################
        # Carbon emissions from off-land commodities.
        ###############################################################
        print("├── Loading off-land commodities' carbon emissions data", flush=True)

        # Read the greenhouse gas intensity data
        off_land_ghg_intensity = pd.read_csv(f'{settings.INPUT_DIR}/agGHG_lvstk_off_land.csv')
        # Split the Emission Source column into two columns
        off_land_ghg_intensity[['Emission Type', 'Emission Source']] = off_land_ghg_intensity['Emission Source'].str.extract(r'^(.*?)\s*\((.*?)\)')

        # Get the emissions from the off-land commodities
        demand_offland_long = self.DEMAND_OFFLAND.stack().reset_index()
        demand_offland_long = demand_offland_long.rename(columns={ 0: 'DEMAND (tonnes)'})

        # Merge the demand and GHG intensity, and calculate the total GHG emissions
        off_land_ghg_emissions = demand_offland_long.merge(off_land_ghg_intensity, on='COMMODITY')
        off_land_ghg_emissions['Total GHG Emissions (tCO2e)'] = off_land_ghg_emissions.eval('`DEMAND (tonnes)` * `Emission Intensity [ kg CO2eq / kg ]`')

        # Keep only the relevant columns
        self.OFF_LAND_GHG_EMISSION = off_land_ghg_emissions[['YEAR',
                                                             'COMMODITY',
                                                             'Emission Type',
                                                             'Emission Source',
                                                             'Total GHG Emissions (tCO2e)']]

        # Get the GHG constraints for luto, shape is (91, 1)
        self.OFF_LAND_GHG_EMISSION_C = self.OFF_LAND_GHG_EMISSION.groupby(['YEAR']).sum(numeric_only=True).values

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
        savburn_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'cell_savanna_burning.h5'), where=self.MASK)

        # Load the columns as numpy arrays
        self.SAVBURN_ELIGIBLE =  savburn_df.ELIGIBLE_AREA.to_numpy()                    # 1 = areas eligible for early dry season savanna burning under the ERF, 0 = ineligible          
        self.SAVBURN_TOTAL_TCO2E_HA = savburn_df.AEA_TOTAL_TCO2E_HA.to_numpy()
        
        # # Avoided emissions from savanna burning
        # self.SAVBURN_AVEM_CH4_TCO2E_HA = savburn_df.SAV_AVEM_CH4_TCO2E_HA.to_numpy()  # Avoided emissions - methane
        # self.SAVBURN_AVEM_N2O_TCO2E_HA = savburn_df.SAV_AVEM_N2O_TCO2E_HA.to_numpy()  # Avoided emissions - nitrous oxide
        # self.SAVBURN_SEQ_CO2_TCO2E_HA = savburn_df.SAV_SEQ_CO2_TCO2E_HA.to_numpy()    # Additional carbon sequestration - carbon dioxide

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
        
        biodiv_raw = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.h5'), where=self.MASK)
        
        # Get the biodiversity quality score 
        if settings.BIO_QUALITY_LAYER == 'Suitability':
            bio_quality_raw = biodiv_raw[f'BIODIV_PRIORITY_SSP{settings.SSP}'].values
            performance_sheet = f'ssp{settings.SSP}'
        elif 'NES' in settings.BIO_QUALITY_LAYER:
            bio_quality_raw = xr.open_dataarray(f"{settings.INPUT_DIR}/bio_NES_Zonation.nc").sel(layer=settings.BIO_QUALITY_LAYER).compute().values[self.MASK]
            performance_sheet = settings.BIO_QUALITY_LAYER
        else:
            raise ValueError(f"Invalid biodiversity quality layer: {settings.BIO_QUALITY_LAYER}, must be 'Suitability' or contain '*NES_likely|may' layers")

        
        # Get connectivity score
        match settings.CONNECTIVITY_SOURCE:
            case 'NCI':
                connectivity_score = biodiv_raw['DCCEEW_NCI'].to_numpy(dtype=np.float32)
                connectivity_score = np.interp( biodiv_raw['DCCEEW_NCI'], (connectivity_score.min(), connectivity_score.max()), (settings.CONNECTIVITY_LB, 1)).astype('float32')
            case 'DWI':
                connectivity_score = biodiv_raw['NATURAL_AREA_CONNECTIVITY'].to_numpy(dtype=np.float32)
                connectivity_score = np.interp(connectivity_score, (connectivity_score.min(), connectivity_score.max()), (1, settings.CONNECTIVITY_LB)).astype('float32')
            case 'NONE':
                connectivity_score = np.ones(self.NCELLS, dtype=np.float32)
            case _:
                raise ValueError(f"Invalid connectivity source: {settings.CONNECTIVITY_SOURCE}, must be 'NCI', 'DWI' or 'NONE'")
            

        # Get the HCAS contribution scale (0-1)
        match settings.CONTRIBUTION_PERCENTILE:
            case 10 | 25 | 50 | 75 | 90:
                bio_HCAS_contribution_lookup = biodiv_contribution_lookup.set_index('lu')[f'PERCENTILE_{settings.CONTRIBUTION_PERCENTILE}'].to_dict()         # Get the biodiversity degradation score at specified percentile (pd.DataFrame)
                unallow_nat_scale = bio_HCAS_contribution_lookup[self.DESC2AGLU['Unallocated - natural land']]                                          # Get the biodiversity degradation score for unallocated natural land (float)
                bio_HCAS_contribution_lookup = {int(k): v * (1 / unallow_nat_scale) for k, v in bio_HCAS_contribution_lookup.items()}                   # Normalise the biodiversity degradation score to the unallocated natural land score
            case 'USER_DEFINED':
                bio_HCAS_contribution_lookup = biodiv_contribution_lookup.set_index('lu')['USER_DEFINED'].to_dict()
            case 'AG_UNIFORM':
                bio_HCAS_contribution_lookup = biodiv_contribution_lookup.set_index('lu')['AG_UNIFORM'].to_dict()                                     
            case _:
                print(f"│   ⚠ WARNING: Invalid habitat condition source: {settings.CONTRIBUTION_PERCENTILE}, must be one of [10, 25, 50, 75, 90], 'USER_DEFINED', or 'AG_UNIFORM'", flush=True)
        
        self.BIO_HABITAT_CONTRIBUTION_LOOK_UP = {j: round(x, settings.ROUND_DECIMALS) for j, x in bio_HCAS_contribution_lookup.items()}                 # Round to the specified decimal places to avoid numerical issues in the GUROBI solver
        
        
        # Get the biodiversity quantity score for each land use in each cell
        self.BIO_QUALITY_RAW = bio_quality_raw * connectivity_score                                           
        self.BIO_QUALITY_LDS = np.where(
            self.SAVBURN_ELIGIBLE, 
            self.BIO_QUALITY_RAW  - (self.BIO_QUALITY_RAW * (1 - settings.BIO_CONTRIBUTION_LDS)), 
            self.BIO_QUALITY_RAW
        )
        
  
        # Conservation performance curve 
        conservation_performance_curve = pd.read_excel(
            os.path.join(settings.INPUT_DIR, 'Biodiversity_conserve_performance.xlsx'),
            sheet_name=performance_sheet
        ).set_index('AREA_COVERAGE_PERCENT')['PRIORITY_RANK'].to_dict()


        # ------------------ Habitat condition impacts for habitat conservation (GBF2) in 'priority degraded areas' regions ---------------
        if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':

            # Get the mask of 'priority degraded areas' for habitat conservation
            self.BIO_GBF2_MASK = bio_quality_raw >= conservation_performance_curve[settings.GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT]
            self.BIO_GBF2_MASK_LDS = np.where(
                self.SAVBURN_ELIGIBLE,
                self.BIO_GBF2_MASK  - (self.BIO_GBF2_MASK * (1 - settings.BIO_CONTRIBUTION_LDS)),
                self.BIO_GBF2_MASK
            )

            self.BIO_GBF2_BASE_YR = (
                np.einsum(
                    'j,mrj,r,r->r',
                    np.array(list(self.BIO_HABITAT_CONTRIBUTION_LOOK_UP.values())),
                    self.AG_L_MRJ,      # lumap in proportion representation if resfactored
                    self.BIO_GBF2_MASK,
                    self.REAL_AREA
                ) - (
                    self.SAVBURN_ELIGIBLE
                    * self.BIO_GBF2_MASK
                    * (1 - settings.BIO_CONTRIBUTION_LDS)
                    * self.REAL_AREA
                    * self.AG_MASK_PROPORTION_R
                )
            )


        # Renewable energy exclusion masks using biodiversity quality and independent cut values
        if any(settings.RENEWABLES_OPTIONS.values()) and settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
            self.RENEWABLE_GBF2_MASK_SOLAR = bio_quality_raw >= conservation_performance_curve[settings.RENEWABLE_GBF2_CUT_SOLAR]
            self.RENEWABLE_GBF2_MASK_WIND = bio_quality_raw >= conservation_performance_curve[settings.RENEWABLE_GBF2_CUT_WIND]

        # Renewable energy exclusion masks using EPBC MNES prioritization layer
        if any(settings.RENEWABLES_OPTIONS.values()) and settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
            mnes_rank_raw = xr.open_dataset(os.path.join(settings.INPUT_DIR, 'renewable_QLD_EPBC_MNES_prioritization.nc'))['data'].values[self.MASK]
            mnes_performance = pd.read_csv(os.path.join(settings.INPUT_DIR, 'renewable_QLD_EPBC_MNES_prioritization_performance.csv')).set_index('AREA_COVERAGE_PERCENT')['PRIORITY_RANK'].to_dict()
            self.RENEWABLE_MNES_MASK_SOLAR = mnes_rank_raw >= mnes_performance[settings.RENEWABLE_EPBC_MNES_CUT_SOLAR]
            self.RENEWABLE_MNES_MASK_WIND = mnes_rank_raw >= mnes_performance[settings.RENEWABLE_EPBC_MNES_CUT_WIND]


        ###############################################################
        # GBF3 biodiversity data. (NVIS vegetation groups)
        ###############################################################
        if settings.BIODIVERSITY_TARGET_GBF_3_NVIS != 'off':
            print("│   ├── Loading GBF3 vegetation data (NVIS)", flush=True)

            
            # Read target DataFrame
            if settings.GBF3_NVIS_REGION_MODE == 'Australia':
                print(f"│   │   ├── NRM region mode: {settings.GBF3_NVIS_REGION_MODE}", flush=True)
                nvis_targets_df = pd.read_excel(
                    settings.INPUT_DIR + "/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.xlsx",
                    sheet_name=f'NVIS_{settings.GBF3_NVIS_TARGET_CLASS}'
                )
                nvis_targets_df.insert(0, 'region', 'Australia')

            elif settings.GBF3_NVIS_REGION_MODE == 'NRM':
                print(f"│   │   ├── NRM region mode: {settings.GBF3_NVIS_SELECTED_REGIONS}", flush=True)
                _nvis_raw = pd.read_excel(
                    settings.INPUT_DIR + "/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS_NRM.xlsx",
                    sheet_name=f'NVIS_{settings.GBF3_NVIS_TARGET_CLASS}'
                ).sort_values(by=['group', 'region'], ascending=True)
                nvis_targets_df = _nvis_raw.query(
                    "region in @settings.GBF3_NVIS_SELECTED_REGIONS and TARGET_LEVEL_2050 > 0"
                ).reset_index(drop=True)

                # Drop explicit trouble-maker (region, group) pairs from settings
                _nvis_excl_explicit = settings.GBF3_NVIS_EXCLUDE_REGION_GROUPS.get(
                    settings.GBF3_NVIS_TARGET_CLASS, []
                )
                if _nvis_excl_explicit:
                    _excl_set = set(_nvis_excl_explicit)
                    _before = len(nvis_targets_df)
                    nvis_targets_df = nvis_targets_df[
                        ~nvis_targets_df.apply(
                            lambda r: (r['region'], r['group']) in _excl_set, axis=1
                        )
                    ].reset_index(drop=True)
                    if _before != len(nvis_targets_df):
                        print(f"│   │   │   ├── Excluded {_before - len(nvis_targets_df)} NVIS groups via GBF3_NVIS_EXCLUDE_REGION_GROUPS", flush=True)

            elif settings.GBF3_NVIS_REGION_MODE == 'IBRA':
                print(f"│   │   ├── NRM region mode: {settings.GBF3_NVIS_REGION_MODE}", flush=True)
                nvis_targets_df = pd.read_excel(
                    settings.INPUT_DIR + "/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS_IBRA.xlsx",
                    sheet_name=f'NVIS_{settings.GBF3_NVIS_TARGET_CLASS}'
                ).sort_values(by='Region', ascending=True
                ).rename(columns={'Region': 'group'})

            else:
                raise ValueError(
                    f"Invalid GBF3_NVIS_REGION_MODE: '{settings.GBF3_NVIS_REGION_MODE}', "
                    "must be 'Australia', 'NRM', or 'IBRA'"
                )

            # Get selected vegetation groups as (region, group) pairs
            nvis_targets_df = nvis_targets_df.sort_values(by=['region', 'group']).reset_index(drop=True) 
            
            if settings.BIODIVERSITY_TARGET_GBF_3_NVIS == 'USER_DEFINED':
                nvis_targets_df = nvis_targets_df[
                    (nvis_targets_df['TARGET_LEVEL_2030'] > 0) &
                    (nvis_targets_df['TARGET_LEVEL_2050'] > 0) &
                    (nvis_targets_df['TARGET_LEVEL_2100'] > 0)
                ].reset_index(drop=True)
            else:
                nvis_targets_df[['TARGET_LEVEL_2030', 'TARGET_LEVEL_2050', 'TARGET_LEVEL_2100']] = \
                    settings.GBF3_TARGETS_DICT[settings.BIODIVERSITY_TARGET_GBF_3_NVIS]


            # Common: store target attributes
            self.BIO_GBF3_NVIS_SEL = list(zip(nvis_targets_df['region'], nvis_targets_df['group']))
            self.BIO_GBF3_NVIS_BASELINE_AND_TARGETS = nvis_targets_df

            # Common: process layers — always (group, n_cells)
            nvis_layers = xr.open_dataarray(settings.INPUT_DIR + f"/bio_GBF3_NVIS_{settings.GBF3_NVIS_TARGET_CLASS}.nc") 
            
            nvis_layers_sel = nvis_layers.sel(
                group=sorted(set(group for _, group in self.BIO_GBF3_NVIS_SEL))
            ) / 100.0  # Convert percentage (0-100) to fraction (0-1)
            
            nvis_layers_arr = xr.DataArray(
                np.array(
                    [self.get_resfactored_average_fraction(arr) for arr in nvis_layers_sel],
                    dtype=np.float32
                ),
                dims=['group', 'cell'],
                coords={
                    'group': nvis_layers_sel.group.values, 
                    'cell': np.arange(self.NCELLS)
                }
            )
            
            # Apply Savanna Burning — always to group-level layers (n_groups, n_cells)
            self.GBF3_NVIS_LAYERS_LDS = xr.where(
                self.SAVBURN_ELIGIBLE[None, :],
                nvis_layers_arr * settings.BIO_CONTRIBUTION_LDS,
                nvis_layers_arr
            ).astype(np.float32)

            # At RESFACTOR > 1 the CSV scores were computed at full-res; recompute them
            # from the resfactored layers so the constraint LHS and RHS are consistent.
            if settings.RESFACTOR > 1:
                nvis_targets_df = self._recompute_nvis_targets_at_rf(nvis_targets_df, nvis_layers_arr)
                self.BIO_GBF3_NVIS_BASELINE_AND_TARGETS = nvis_targets_df
                print(f"│   │   ├── Recomputed NVIS targets at RESFACTOR={settings.RESFACTOR}", flush=True)

            if settings.GBF3_NVIS_REGION_MODE == 'NRM':
                # NRM: one name per (group, region) constraint pair
                self.BIO_GBF3_NVIS_ID2DESC = {
                    i: f"{row['group']} [{row['region']}]"
                    for i, (_, row) in enumerate(nvis_targets_df.iterrows())
                }
            else:
                # Australia / IBRA: ensure target row order matches layer order
                self.BIO_GBF3_NVIS_BASELINE_AND_TARGETS['order'] = self.BIO_GBF3_NVIS_BASELINE_AND_TARGETS['group'].apply(
                    lambda x: list(nvis_layers_sel.group.values).index(x)
                )
                self.BIO_GBF3_NVIS_BASELINE_AND_TARGETS = self.BIO_GBF3_NVIS_BASELINE_AND_TARGETS.sort_values(by='order').drop(columns='order')
                self.BIO_GBF3_NVIS_ID2DESC = dict(enumerate(nvis_layers_sel.group.values))



            


        ##########################################################################
        #  Biodiersity environmental significance (GBF4)                         #
        ##########################################################################
        if settings.BIODIVERSITY_TARGET_GBF_4_SNES != 'off':

            print("│   ├── Loading environmental significance data (SNES)", flush=True)

            # Load SNES spatial layers
            BIO_GBF4_SPECIES_raw = xr.open_dataarray(f'{settings.INPUT_DIR}/bio_GBF4_SNES.nc', chunks={'species': 1})

            if settings.GBF4_SNES_REGION_MODE == 'Australia':
                # ---- Australia mode ----
                BIO_GBF4_SNES_score = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_SNES.csv').sort_values(by='SCIENTIFIC_NAME', ascending=True)

                # Drop trouble-maker species (rule_out_trouble_maker_speceis workflow)
                if settings.GBF4_SNES_EXCLUDE_REGION_SPECIES:
                    # Australia mode: no region, so match on species name only
                    _excl_sp = {sp for _, sp in settings.GBF4_SNES_EXCLUDE_REGION_SPECIES}
                    _before = len(BIO_GBF4_SNES_score)
                    BIO_GBF4_SNES_score = BIO_GBF4_SNES_score[~BIO_GBF4_SNES_score['SCIENTIFIC_NAME'].isin(_excl_sp)].reset_index(drop=True)
                    print(f"│   │   │   └── Excluded {_before - len(BIO_GBF4_SNES_score)} SNES species via GBF4_SNES_EXCLUDE_REGION_SPECIES", flush=True)

                # Whitelist: keep only listed species (applied after EXCLUDE)
                if settings.GBF4_SNES_INCLUDE_SPECIES:
                    _incl = list(settings.GBF4_SNES_INCLUDE_SPECIES)
                    _before = len(BIO_GBF4_SNES_score)
                    BIO_GBF4_SNES_score = BIO_GBF4_SNES_score[BIO_GBF4_SNES_score['SCIENTIFIC_NAME'].isin(_incl)].reset_index(drop=True)
                    print(f"│   │   │   └── Whitelist GBF4_SNES_INCLUDE_SPECIES kept {len(BIO_GBF4_SNES_score)}/{_before} SNES rows", flush=True)

                nc_species_index = set(BIO_GBF4_SPECIES_raw.coords['species'].values.tolist())

                likely_sel_raw = [row['SCIENTIFIC_NAME'] for _, row in BIO_GBF4_SNES_score.iterrows()
                                  if all([row.get('TARGET_LEVEL_2030_LIKELY', 0) > 0,
                                          row.get('TARGET_LEVEL_2050_LIKELY', 0) > 0,
                                          row.get('TARGET_LEVEL_2100_LIKELY', 0) > 0])]
                missing_likely = [sp for sp in likely_sel_raw if sp not in nc_species_index]
                if missing_likely:
                    print(f"│   │   ⚠ WARNING: {len(missing_likely)} SNES 'LIKELY' species have targets but no spatial data — excluded:", flush=True)
                    for sp in missing_likely:
                        print(f"│   │       ├── {sp}", flush=True)
                self.BIO_GBF4_SNES_LIKELY_SEL = [sp for sp in likely_sel_raw if sp in nc_species_index]

                lm_sel_raw = [row['SCIENTIFIC_NAME'] for _, row in BIO_GBF4_SNES_score.iterrows()
                              if all([row.get('TARGET_LEVEL_2030_LIKELY_MAYBE', 0) > 0,
                                      row.get('TARGET_LEVEL_2050_LIKELY_MAYBE', 0) > 0,
                                      row.get('TARGET_LEVEL_2100_LIKELY_MAYBE', 0) > 0])]
                missing_lm = [sp for sp in lm_sel_raw if sp not in nc_species_index]
                if missing_lm:
                    print(f"│   │   ⚠ WARNING: {len(missing_lm)} SNES 'LIKELY_MAYBE' species have targets but no spatial data — excluded:", flush=True)
                    for sp in missing_lm:
                        print(f"│   │       ├── {sp}", flush=True)
                self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL = [sp for sp in lm_sel_raw if sp in nc_species_index]

                if len(self.BIO_GBF4_SNES_LIKELY_SEL) == 0:
                    print("│   │   ⚠ WARNING: No 'LIKELY' SNES layers selected, proceeding with empty selection.", flush=True)

                likely_maybe_union = set(self.BIO_GBF4_SNES_LIKELY_SEL).intersection(self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL)
                if likely_maybe_union:
                    print(f"│   │   ⚠ WARNING: {len(likely_maybe_union)} duplicate SNES species targets found, using 'LIKELY' targets only:", flush=True)
                    for idx, name in enumerate(likely_maybe_union):
                        print(f"│   │       ├── {idx+1}) {name}", flush=True)
                    self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL = list(set(self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL) - likely_maybe_union)

                self.BIO_GBF4_SNES_SEL_ALL = self.BIO_GBF4_SNES_LIKELY_SEL + self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL
                self.BIO_GBF4_SNES_SPECIES_COORD = self.BIO_GBF4_SNES_SEL_ALL           # species names for xarray coord
                self.BIO_GBF4_SNES_SEL = [('Australia', sp) for sp in self.BIO_GBF4_SNES_SEL_ALL]
                self.BIO_GBF4_PRESENCE_SNES_SEL = (['LIKELY'] * len(self.BIO_GBF4_SNES_LIKELY_SEL)
                                                    + ['LIKELY_MAYBE'] * len(self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL))
                self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY = BIO_GBF4_SNES_score.query(f'SCIENTIFIC_NAME in {self.BIO_GBF4_SNES_LIKELY_SEL}')
                self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE = BIO_GBF4_SNES_score.query(f'SCIENTIFIC_NAME in {self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL}')

                snes_parts = []
                if self.BIO_GBF4_SNES_LIKELY_SEL:
                    snes_parts.append(BIO_GBF4_SPECIES_raw.sel(species=self.BIO_GBF4_SNES_LIKELY_SEL, presence='LIKELY'))
                if self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL:
                    snes_parts.append(BIO_GBF4_SPECIES_raw.sel(species=self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL, presence='LIKELY_AND_MAYBE'))
                snes_arr = xr.concat(snes_parts, dim='species') if snes_parts else BIO_GBF4_SPECIES_raw.isel(species=[], cell=slice(None))

                # Build per-(region=Australia, species) layers — no region mask applied
                snes_layers = xr.DataArray(
                    np.zeros((len(self.BIO_GBF4_SNES_SEL), self.NCELLS), dtype=np.float32),
                    dims=['layer', 'cell'],
                    coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF4_SNES_SEL, names=['region', 'species'])}
                )
                for i, (region, sp) in enumerate(self.BIO_GBF4_SNES_SEL):
                    snes_layers.values[i] = self.get_resfactored_average_fraction(snes_arr.sel(species=sp).values)
                self.BIO_GBF4_SPECIES_LAYERS = snes_layers

            else:
                # ---- NRM mode ----
                print(f"│   │   ├── NRM region mode: {settings.GBF4_SNES_SELECTED_REGIONS}", flush=True)

                snes_nrm_df = pd.read_csv(
                    settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_SNES_NRM.csv'
                ).sort_values(by=['SCIENTIFIC_NAME', 'region'], ascending=True)
                snes_nrm_df = snes_nrm_df[
                    snes_nrm_df['region'].isin(settings.GBF4_SNES_SELECTED_REGIONS)
                    & (snes_nrm_df['TARGET_LEVEL_2030_LIKELY'] > 0)
                ].reset_index(drop=True)

                # Drop trouble-maker (region, species) pairs — NRM mode matches on both
                if settings.GBF4_SNES_EXCLUDE_REGION_SPECIES:
                    _excl_set = set(settings.GBF4_SNES_EXCLUDE_REGION_SPECIES)
                    _before = len(snes_nrm_df)
                    snes_nrm_df = snes_nrm_df[
                        ~snes_nrm_df.apply(lambda r: (r['region'], r['SCIENTIFIC_NAME']) in _excl_set, axis=1)
                    ].reset_index(drop=True)
                    if _before != len(snes_nrm_df):
                        print(f"│   │   │   ├── Excluded {_before - len(snes_nrm_df)} SNES (region,species) pairs via GBF4_SNES_EXCLUDE_REGION_SPECIES", flush=True)

                # Whitelist: keep only listed species (applied after EXCLUDE)
                if settings.GBF4_SNES_INCLUDE_SPECIES:
                    _incl = list(settings.GBF4_SNES_INCLUDE_SPECIES)
                    _before = len(snes_nrm_df)
                    snes_nrm_df = snes_nrm_df[snes_nrm_df['SCIENTIFIC_NAME'].isin(_incl)].reset_index(drop=True)
                    print(f"│   │   │   └── Whitelist GBF4_SNES_INCLUDE_SPECIES kept {len(snes_nrm_df)}/{_before} SNES (species,region) rows", flush=True)

                if len(snes_nrm_df) == 0:
                    raise ValueError(
                        f"No valid GBF4 SNES NRM targets found for regions {settings.GBF4_SNES_SELECTED_REGIONS}. "
                        "Check BIODIVERSITY_GBF4_TARGET_SNES_NRM.csv."
                    )

                # Filter to species that actually exist in the spatial NetCDF layer
                available_in_nc = set(BIO_GBF4_SPECIES_raw.coords['species'].values.tolist())
                missing_in_nc = [sp for sp in snes_nrm_df['SCIENTIFIC_NAME'].unique() if sp not in available_in_nc]
                if missing_in_nc:
                    print(f"│   │   │   ⚠  {len(missing_in_nc)} NRM target species not found in spatial layers, skipping:", flush=True)
                    for sp in missing_in_nc[:5]:
                        print(f"│   │   │   ├── {sp}", flush=True)
                    if len(missing_in_nc) > 5:
                        print(f"│   │   │   └── ... and {len(missing_in_nc) - 5} more", flush=True)
                    snes_nrm_df = snes_nrm_df[snes_nrm_df['SCIENTIFIC_NAME'].isin(available_in_nc)].reset_index(drop=True)

                unique_species = snes_nrm_df['SCIENTIFIC_NAME'].unique().tolist()

                self.BIO_GBF4_SNES_LIKELY_SEL = unique_species
                self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL = []
                self.BIO_GBF4_SNES_SEL_ALL = [f"{sp} [{r}]" for sp, r in zip(snes_nrm_df['SCIENTIFIC_NAME'], snes_nrm_df['region'])]
                self.BIO_GBF4_SNES_SPECIES_COORD = self.BIO_GBF4_SNES_LIKELY_SEL        # unique species names for xarray coord
                self.BIO_GBF4_SNES_SEL = list(zip(snes_nrm_df['region'], snes_nrm_df['SCIENTIFIC_NAME']))
                self.BIO_GBF4_PRESENCE_SNES_SEL = ['LIKELY'] * len(snes_nrm_df)
                self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY = snes_nrm_df
                self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE = pd.DataFrame()

                # Load NRM full-res region array for per-(region, species) masking before resfactoring
                _nrm_full = np.asarray(
                    pd.read_hdf(os.path.join(settings.INPUT_DIR, 'REGION_NRM_r.h5'), columns=['NRM_NAME'])['NRM_NAME'].values,
                    dtype=object
                )

                snes_arr = BIO_GBF4_SPECIES_raw.sel(species=unique_species, presence='LIKELY')

                # Build per-(region, species) layers: region mask applied at full-res before resfactoring
                snes_layers = xr.DataArray(
                    np.zeros((len(self.BIO_GBF4_SNES_SEL), self.NCELLS), dtype=np.float32),
                    dims=['layer', 'cell'],
                    coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF4_SNES_SEL, names=['region', 'species'])}
                )
                for i, (region, sp) in enumerate(self.BIO_GBF4_SNES_SEL):
                    sp_arr = snes_arr.sel(species=sp).values
                    region_mask = (_nrm_full == region).astype(np.float32)
                    snes_layers.values[i] = self.get_resfactored_average_fraction(sp_arr * region_mask * self.LUMASK)
                self.BIO_GBF4_SPECIES_LAYERS = snes_layers

                # At RESFACTOR > 1, recompute SNES target scores from the resfactored
                # region-masked layers so constraint LHS and RHS are consistent.
                if settings.RESFACTOR > 1:
                    snes_nrm_df = self._recompute_snes_ecnes_targets_at_rf(
                        snes_nrm_df, 'SCIENTIFIC_NAME', snes_layers
                    )
                    self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY = snes_nrm_df
                    print(f"│   │   ├── Recomputed SNES targets at RESFACTOR={settings.RESFACTOR}", flush=True)

                print(f"│   │   └── {len(snes_nrm_df)} (species, region) constraints from {len(unique_species)} species", flush=True)
        
        
        if settings.BIODIVERSITY_TARGET_GBF_4_ECNES != 'off':
            print("│   ├── Loading environmental significance data (ECNES)", flush=True)

            # Load ECNES spatial layers
            BIO_GBF4_COMUNITY_raw = xr.open_dataarray(f'{settings.INPUT_DIR}/bio_GBF4_ECNES.nc', chunks={'species': 1})

            if settings.GBF4_ECNES_REGION_MODE == 'Australia':
                # ---- Australia mode ----
                BIO_GBF4_ECNES_score = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_ECNES.csv').sort_values(by='COMMUNITY', ascending=True)
                BIO_GBF4_ECNES_score.columns = BIO_GBF4_ECNES_score.columns.str.strip()

                # Drop trouble-maker communities (rule_out_trouble_maker_speceis workflow)
                if settings.GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES:
                    # Australia mode: no region, so match on community name only
                    _excl_comm = {c for _, c in settings.GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES}
                    _before = len(BIO_GBF4_ECNES_score)
                    BIO_GBF4_ECNES_score = BIO_GBF4_ECNES_score[~BIO_GBF4_ECNES_score['COMMUNITY'].isin(_excl_comm)].reset_index(drop=True)
                    print(f"│   │   │   └── Excluded {_before - len(BIO_GBF4_ECNES_score)} ECNES communities via GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES", flush=True)

                # Whitelist: keep only listed communities (applied after EXCLUDE)
                if settings.GBF4_ECNES_INCLUDE_COMMUNITIES:
                    _incl = list(settings.GBF4_ECNES_INCLUDE_COMMUNITIES)
                    _before = len(BIO_GBF4_ECNES_score)
                    BIO_GBF4_ECNES_score = BIO_GBF4_ECNES_score[BIO_GBF4_ECNES_score['COMMUNITY'].isin(_incl)].reset_index(drop=True)
                    print(f"│   │   │   └── Whitelist GBF4_ECNES_INCLUDE_COMMUNITIES kept {len(BIO_GBF4_ECNES_score)}/{_before} ECNES rows", flush=True)

                self.BIO_GBF4_ECNES_LIKELY_SEL = [row['COMMUNITY'] for _, row in BIO_GBF4_ECNES_score.iterrows()
                                                   if all([row.get('TARGET_LEVEL_2030_LIKELY', 0) > 0,
                                                           row.get('TARGET_LEVEL_2050_LIKELY', 0) > 0,
                                                           row.get('TARGET_LEVEL_2100_LIKELY', 0) > 0])]
                self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL = [row['COMMUNITY'] for _, row in BIO_GBF4_ECNES_score.iterrows()
                                                             if all([row.get('TARGET_LEVEL_2030_LIKELY_MAYBE', 0) > 0,
                                                                     row.get('TARGET_LEVEL_2050_LIKELY_MAYBE', 0) > 0,
                                                                     row.get('TARGET_LEVEL_2100_LIKELY_MAYBE', 0) > 0])]

                if len(self.BIO_GBF4_ECNES_LIKELY_SEL) + len(self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL) == 0:
                    print("│   │   ⚠ WARNING: No 'LIKELY' or 'LIKELY_MAYBE' ECNES layers selected, proceeding with empty selection.", flush=True)

                likely_maybe_union = set(self.BIO_GBF4_ECNES_LIKELY_SEL).intersection(self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL)
                if likely_maybe_union:
                    print(f"│   │   ⚠ WARNING: {len(likely_maybe_union)} duplicate ECNES community targets found, using 'LIKELY' targets only:", flush=True)
                    for idx, name in enumerate(likely_maybe_union):
                        print(f"│   │       ├── {idx+1}) {name}", flush=True)
                    self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL = list(set(self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL) - likely_maybe_union)

                self.BIO_GBF4_ECNES_SEL_ALL = self.BIO_GBF4_ECNES_LIKELY_SEL + self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL
                self.BIO_GBF4_ECNES_SPECIES_COORD = self.BIO_GBF4_ECNES_SEL_ALL         # community names for xarray coord
                self.BIO_GBF4_ECNES_SEL = [('Australia', c) for c in self.BIO_GBF4_ECNES_SEL_ALL]
                self.BIO_GBF4_PRESENCE_ECNES_SEL = (
                    ['LIKELY'] * len(self.BIO_GBF4_ECNES_LIKELY_SEL)
                    + ['LIKELY_MAYBE'] * len(self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL)
                )
                self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY = BIO_GBF4_ECNES_score.query(f'COMMUNITY in {self.BIO_GBF4_ECNES_LIKELY_SEL}')
                self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE = BIO_GBF4_ECNES_score.query(f'COMMUNITY in {self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL}')

                ecnes_parts = []
                if self.BIO_GBF4_ECNES_LIKELY_SEL:
                    ecnes_parts.append(BIO_GBF4_COMUNITY_raw.sel(species=self.BIO_GBF4_ECNES_LIKELY_SEL, presence='LIKELY').compute())
                if self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL:
                    ecnes_parts.append(BIO_GBF4_COMUNITY_raw.sel(species=self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL, presence='LIKELY_AND_MAYBE').compute())
                ecnes_arr = xr.concat(ecnes_parts, dim='species') if ecnes_parts else BIO_GBF4_COMUNITY_raw.isel(species=[], cell=slice(None))

                # Build per-(region=Australia, community) layers — no region mask applied
                ecnes_layers = xr.DataArray(
                    np.zeros((len(self.BIO_GBF4_ECNES_SEL), self.NCELLS), dtype=np.float32),
                    dims=['layer', 'cell'],
                    coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF4_ECNES_SEL, names=['region', 'species'])}
                )
                for i, (region, comm) in enumerate(self.BIO_GBF4_ECNES_SEL):
                    ecnes_layers.values[i] = self.get_resfactored_average_fraction(ecnes_arr.sel(species=comm).values)
                self.BIO_GBF4_COMUNITY_LAYERS = ecnes_layers

            else:
                # ---- NRM mode ----
                print(f"│   │   ├── NRM region mode: {settings.GBF4_ECNES_SELECTED_REGIONS}", flush=True)

                ecnes_nrm_df = pd.read_csv(
                    settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_ECNES_NRM.csv'
                ).sort_values(by=['COMMUNITY', 'region'], ascending=True)
                ecnes_nrm_df.columns = ecnes_nrm_df.columns.str.strip()
                ecnes_nrm_df = ecnes_nrm_df[
                    ecnes_nrm_df['region'].isin(settings.GBF4_ECNES_SELECTED_REGIONS)
                    & (ecnes_nrm_df['TARGET_LEVEL_2030_LIKELY'] > 0)
                ].reset_index(drop=True)

                # Drop trouble-maker (region, community) pairs — NRM mode matches on both
                if settings.GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES:
                    _excl_set = set(settings.GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES)
                    _before = len(ecnes_nrm_df)
                    ecnes_nrm_df = ecnes_nrm_df[
                        ~ecnes_nrm_df.apply(lambda r: (r['region'], r['COMMUNITY']) in _excl_set, axis=1)
                    ].reset_index(drop=True)
                    if _before != len(ecnes_nrm_df):
                        print(f"│   │   │   ├── Excluded {_before - len(ecnes_nrm_df)} ECNES (region,community) pairs via GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES", flush=True)

                # Whitelist: keep only listed communities (applied after EXCLUDE)
                if settings.GBF4_ECNES_INCLUDE_COMMUNITIES:
                    _incl = list(settings.GBF4_ECNES_INCLUDE_COMMUNITIES)
                    _before = len(ecnes_nrm_df)
                    ecnes_nrm_df = ecnes_nrm_df[ecnes_nrm_df['COMMUNITY'].isin(_incl)].reset_index(drop=True)
                    print(f"│   │   │   └── Whitelist GBF4_ECNES_INCLUDE_COMMUNITIES kept {len(ecnes_nrm_df)}/{_before} ECNES (community,region) rows", flush=True)

                if len(ecnes_nrm_df) == 0:
                    raise ValueError(
                        f"No valid GBF4 ECNES NRM targets found for regions {settings.GBF4_ECNES_SELECTED_REGIONS}. "
                        "Check BIODIVERSITY_GBF4_TARGET_ECNES_NRM.csv."
                    )

                # Filter to communities that actually exist in the spatial NetCDF layer
                available_in_nc = set(BIO_GBF4_COMUNITY_raw.coords['species'].values.tolist())
                missing_in_nc = [c for c in ecnes_nrm_df['COMMUNITY'].unique() if c not in available_in_nc]
                if missing_in_nc:
                    print(f"│   │   │   ⚠  {len(missing_in_nc)} NRM target communities not found in spatial layers, skipping:", flush=True)
                    for c in missing_in_nc[:5]:
                        print(f"│   │   │   ├── {c}", flush=True)
                    if len(missing_in_nc) > 5:
                        print(f"│   │   │   └── ... and {len(missing_in_nc) - 5} more", flush=True)
                    ecnes_nrm_df = ecnes_nrm_df[ecnes_nrm_df['COMMUNITY'].isin(available_in_nc)].reset_index(drop=True)

                unique_communities = ecnes_nrm_df['COMMUNITY'].unique().tolist()

                self.BIO_GBF4_ECNES_LIKELY_SEL = unique_communities
                self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL = []
                self.BIO_GBF4_ECNES_SEL_ALL = [f"{c} [{r}]" for c, r in zip(ecnes_nrm_df['COMMUNITY'], ecnes_nrm_df['region'])]
                self.BIO_GBF4_ECNES_SPECIES_COORD = self.BIO_GBF4_ECNES_LIKELY_SEL      # unique community names for xarray coord
                self.BIO_GBF4_ECNES_SEL = list(zip(ecnes_nrm_df['region'], ecnes_nrm_df['COMMUNITY']))
                self.BIO_GBF4_PRESENCE_ECNES_SEL = ['LIKELY'] * len(ecnes_nrm_df)
                self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY = ecnes_nrm_df
                self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE = pd.DataFrame()

                # Load NRM full-res region array for per-(region, community) masking before resfactoring
                _nrm_full = np.asarray(
                    pd.read_hdf(os.path.join(settings.INPUT_DIR, 'REGION_NRM_r.h5'), columns=['NRM_NAME'])['NRM_NAME'].values,
                    dtype=object
                )

                ecnes_arr = BIO_GBF4_COMUNITY_raw.sel(species=unique_communities, presence='LIKELY').compute()

                # Build per-(region, community) layers: region mask applied at full-res before resfactoring
                ecnes_layers = xr.DataArray(
                    np.zeros((len(self.BIO_GBF4_ECNES_SEL), self.NCELLS), dtype=np.float32),
                    dims=['layer', 'cell'],
                    coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF4_ECNES_SEL, names=['region', 'species'])}
                )
                for i, (region, comm) in enumerate(self.BIO_GBF4_ECNES_SEL):
                    comm_arr = ecnes_arr.sel(species=comm).values
                    region_mask = (_nrm_full == region).astype(np.float32)
                    ecnes_layers.values[i] = self.get_resfactored_average_fraction(comm_arr * region_mask * self.LUMASK)
                self.BIO_GBF4_COMUNITY_LAYERS = ecnes_layers

                # At RESFACTOR > 1, recompute ECNES target scores from the resfactored
                # region-masked layers so constraint LHS and RHS are consistent.
                if settings.RESFACTOR > 1:
                    ecnes_nrm_df = self._recompute_snes_ecnes_targets_at_rf(
                        ecnes_nrm_df, 'COMMUNITY', ecnes_layers
                    )
                    self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY = ecnes_nrm_df
                    print(f"│   │   ├── Recomputed ECNES targets at RESFACTOR={settings.RESFACTOR}", flush=True)

                print(f"│   │   └── {len(ecnes_nrm_df)} (community, region) constraints from {len(unique_communities)} communities", flush=True)
        
  
        
        ##########################################################################
        # Biodiersity species suitability under climate change (GBF8)            #
        ##########################################################################
        
        if settings.BIODIVERSITY_TARGET_GBF_8 != 'off':
            
            print("│   ├── Loading Species suitability data", flush=True)
            
            # Read in the species data from Carla Archibald (noted as GBF-8)
            BIO_GBF8_SPECIES_raw = xr.open_dataset(f'{settings.INPUT_DIR}/bio_GBF8_ssp{settings.SSP}_EnviroSuit.nc', chunks={'year':1,'species':1})['data']        
            bio_GBF8_baseline_score = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_SCORES.csv').sort_values(by='species', ascending=True)
            bio_GBF8_target_percent = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_TARGET.csv').sort_values(by='species', ascending=True)
            
            self.BIO_GBF8_SEL_SPECIES = [row['species'] for _,row in bio_GBF8_target_percent.iterrows()
                                        if all([row.get('TARGET_LEVEL_2030', 0)>0,
                                                row.get('TARGET_LEVEL_2050', 0)>0,
                                                row.get('TARGET_LEVEL_2100', 0)>0])]
            self.BIO_GBF8_SEL = [('Australia', sp) for sp in self.BIO_GBF8_SEL_SPECIES]

            self.BIO_GBF8_OUTSDIE_LUTO_SCORE_SPECIES = bio_GBF8_baseline_score.query(f'species in {self.BIO_GBF8_SEL_SPECIES}')[['species', 'year', f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{settings.SSP}']]
            self.BIO_GBF8_OUTSDIE_LUTO_SCORE_GROUPS = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_SCORES_group.csv')[['group', 'year', f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{settings.SSP}']]
            
            self.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES = bio_GBF8_target_percent.query(f'species in {self.BIO_GBF8_SEL_SPECIES}')
            self.BIO_GBF8_BASELINE_SCORE_GROUPS = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_TARGET_group.csv')
            
            self.N_GBF8_SPECIES = len(self.BIO_GBF8_SEL_SPECIES)
            if self.BIO_GBF8_SEL_SPECIES:
                self.BIO_GBF8_SPECIES_LAYER = BIO_GBF8_SPECIES_raw.sel(species=self.BIO_GBF8_SEL_SPECIES).compute()
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
        beccs_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'cell_BECCS_df.h5'), where=self.MASK)

        # Capture as numpy arrays
        self.BECCS_COSTS_AUD_HA_YR = beccs_df['BECCS_COSTS_AUD_HA_YR'].to_numpy()
        self.BECCS_REV_AUD_HA_YR = beccs_df['BECCS_REV_AUD_HA_YR'].to_numpy()
        self.BECCS_TCO2E_HA_YR = beccs_df['BECCS_TCO2E_HA_YR'].to_numpy()
        self.BECCS_MWH_HA_YR = beccs_df['BECCS_MWH_HA_YR'].to_numpy()

 

    def _recompute_nvis_targets_at_rf(
        self,
        nvis_targets_df: 'pd.DataFrame',
        nvis_layers_arr: 'xr.DataArray',
    ) -> 'pd.DataFrame':
        """
        Replace ALL_HA, IN_LUTO_HA, BASEYEAR_LEVEL, and ATTAINABLE_LEVEL in
        nvis_targets_df with values consistent with the current RESFACTOR.

        Follows the same logic as script_5_2_get_NVIS_SNES_ECNES_targets_by_regions.py:
          IN_LUTO_HA  = sum_{r in region}(layer[group,r] * REAL_AREA[r] * degrade[r])
          where degrade[r] = BIO_HABITAT_CONTRIBUTION_LOOK_UP[LUMAP[r]] (base year)

        NATURAL_OUT_LUTO_HA and TARGET_LEVEL_* are kept from the CSV.
        NON_NATURAL_OUT_LUTO_HA is fixed (outside LUTO, doesn't change with RF):
          derived as ALL_HA_csv * (1 - ATTAINABLE_LEVEL_csv / 100).
        """
        from luto import tools as _tools
        df = nvis_targets_df.copy()

        # Base-year degrade factor per LUTO cell — fractional weighted average
        # over land-use mix at the current RESFACTOR (matches script_5_2 logic).
        _degred_dict = xr.DataArray(
            list(self.BIO_HABITAT_CONTRIBUTION_LOOK_UP.values()),
            dims=['lu'], coords={'lu': self.AGRICULTURAL_LANDUSES}
        )
        degrade_r = (
            _tools.ag_mrj_to_xr(self, self.AG_L_MRJ).sum('lm') * _degred_dict
        ).fillna(0.0).sum('lu').values.astype(np.float32)  # (NCELLS,)

        has_region_col = 'region' in df.columns
        has_attainable_col = 'ATTAINABLE_LEVEL' in df.columns

        for df_i, row in df.iterrows():
            group  = row['group']
            region = row['region'] if has_region_col else 'Australia'

            g_arr = nvis_layers_arr.sel(group=group).values  # (NCELLS,)

            if region == 'Australia':
                in_luto_ha_rf = float((g_arr * self.REAL_AREA * degrade_r).sum())
            else:
                region_mask = (self.REGION_NRM_NAME == region)
                in_luto_ha_rf = float((g_arr[region_mask] * self.REAL_AREA[region_mask] * degrade_r[region_mask]).sum())

            nat_out = float(row['NATURAL_OUT_LUTO_HA'])
            if has_attainable_col:
                non_nat_out = float(row['ALL_HA']) * (1.0 - float(row['ATTAINABLE_LEVEL']) / 100.0)
            else:
                # Fallback: derive from balance
                non_nat_out = float(row['ALL_HA']) - float(row['IN_LUTO_HA']) - nat_out

            all_ha_rf = in_luto_ha_rf + nat_out + non_nat_out
            if all_ha_rf > 0:
                baseyear_rf   = (in_luto_ha_rf + nat_out) / all_ha_rf * 100.0
                attainable_rf = (1.0 - non_nat_out / all_ha_rf) * 100.0
            else:
                baseyear_rf = attainable_rf = 0.0

            df.at[df_i, 'ALL_HA']         = all_ha_rf
            df.at[df_i, 'IN_LUTO_HA']     = in_luto_ha_rf
            df.at[df_i, 'BASEYEAR_LEVEL'] = baseyear_rf
            if has_attainable_col:
                df.at[df_i, 'ATTAINABLE_LEVEL'] = attainable_rf

        return df

    def _recompute_snes_ecnes_targets_at_rf(
        self,
        df: 'pd.DataFrame',
        key_col: str,
        layers_xr: 'xr.DataArray',
    ) -> 'pd.DataFrame':
        """
        Replace BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY, BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY,
        BASEYEAR_LEVEL_LIKELY, and ATTAINABLE_LEVEL_LIKELY with values consistent with
        the current RESFACTOR.

        Follows the same logic as script_5_2_get_NVIS_SNES_ECNES_targets_by_regions.py.
        layers_xr must have dims ['layer', 'cell'] with layer as MultiIndex (region, species)
        and rows in the same order as df.

        BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY and TARGET_LEVEL_* are kept from the CSV.
        NON_NATURAL_OUT_LUTO_HA is fixed:
          derived as BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY * (1 - ATTAINABLE_LEVEL_LIKELY / 100).
        """
        from luto import tools as _tools
        df = df.copy()

        # Fractional weighted-average degrade — matches script_5_2 and step_1 validation.
        _degred_dict = xr.DataArray(
            list(self.BIO_HABITAT_CONTRIBUTION_LOOK_UP.values()),
            dims=['lu'], coords={'lu': self.AGRICULTURAL_LANDUSES}
        )
        degrade_r = (
            _tools.ag_mrj_to_xr(self, self.AG_L_MRJ).sum('lm') * _degred_dict
        ).fillna(0.0).sum('lu').values.astype(np.float32)  # (NCELLS,)

        for enum_i, (df_i, row) in enumerate(df.iterrows()):
            layer_r = layers_xr.values[enum_i]  # (NCELLS,) — region mask already baked in

            in_luto_ha_rf = float((layer_r * self.REAL_AREA * degrade_r).sum())

            nat_out     = float(row['BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY'])
            all_ha_csv  = float(row['BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY'])
            attain_csv  = float(row['ATTAINABLE_LEVEL_LIKELY'])
            non_nat_out = all_ha_csv * (1.0 - attain_csv / 100.0)

            all_ha_rf = in_luto_ha_rf + nat_out + non_nat_out
            if all_ha_rf > 0:
                baseyear_rf   = (in_luto_ha_rf + nat_out) / all_ha_rf * 100.0
                attainable_rf = (1.0 - non_nat_out / all_ha_rf) * 100.0
            else:
                baseyear_rf = attainable_rf = 0.0

            df.at[df_i, 'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY']      = all_ha_rf
            df.at[df_i, 'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY'] = in_luto_ha_rf
            df.at[df_i, 'BASEYEAR_LEVEL_LIKELY']                    = baseyear_rf
            df.at[df_i, 'ATTAINABLE_LEVEL_LIKELY']                  = attainable_rf

        return df

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

    def add_non_ag_dvars(self, yr: int, non_ag_dvars: np.ndarray):
        """
        Safely adds non-agricultural decision variables' values to the Data object.
        """
        self.non_ag_dvars[yr] = non_ag_dvars

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
            return tools.lumap2ag_l_mrj(self.LUMAP_NO_RESFACTOR, self.LMMAP_NO_RESFACTOR)[:, self.MASK, :]

        lumap_mrj = np.zeros((self.NLMS, self.NCELLS, self.N_AG_LUS), dtype=np.float32)
        for idx_lu in self.DESC2AGLU.values():
            for idx_w, _ in enumerate(self.LANDMANS):
                # Get the cells with the same ID and water supply
                lu_arr = (self.LUMAP_NO_RESFACTOR == idx_lu) * (self.LMMAP_NO_RESFACTOR == idx_w)
                lumap_mrj[idx_w, :, idx_lu] = self.get_resfactored_average_fraction(lu_arr, use_valid_cell_count=False)
                    
        return lumap_mrj
    
    
    def get_resfactored_average_fraction(self, arr: np.ndarray, use_valid_cell_count: bool = False) -> np.ndarray:
        '''
        Calculate the average value for each resfactored cell, given the input arr is masked by the land-use mask
        (has a length of the number of cells in the full-resolution 1D array, i.e., 6956407).

        Args:
            arr (np.ndarray): A 1D array containing the values for the full-resolution array, should be the length
                of the number of cells in the full-resolution array (i.e., 6956407).
            use_valid_cell_count (bool): If True (default), divide by the number of valid NLUM cells in each
                coarsened block. This preserves the per-valid-cell average (e.g. habitat fraction) and prevents
                edge/coastal blocks with fewer valid cells from being artificially dwarfed vs RES1.
                If False, divide by RESFACTOR² (all cells in block), which gives the fraction relative to the
                full coarse cell area — required for lumap dvars where REAL_AREA is the full coarse cell.

        Returns:
            np.ndarray: A 1D array containing the average values for each cell in the
            resfactored array, with the same length as the number of cells in the resfactored array.
        '''

        arr_2d = np.zeros_like(self.LUMAP_2D_FULLRES, dtype=np.float32)
        np.place(arr_2d, self.NLUM_MASK, arr)

        arr_2d_xr = xr.DataArray(arr_2d, dims=['y', 'x'])
        arr_sum = arr_2d_xr.coarsen(x=settings.RESFACTOR, y=settings.RESFACTOR, boundary='pad').sum()

        if use_valid_cell_count:
            # Divide by valid NLUM cells per block — preserves per-valid-cell average (e.g. habitat layers)
            nlum_2d_xr = xr.DataArray(np.nan_to_num(arr_to_xr(self, arr>0)), dims=['y', 'x'])
            denom = nlum_2d_xr.coarsen(x=settings.RESFACTOR, y=settings.RESFACTOR, boundary='pad').sum()
            denom = denom.where(denom > 0, other=1)  # avoid division by zero
        else:
            # Divide by RESFACTOR² — fraction relative to full coarse cell (required for lumap dvars)
            denom = settings.RESFACTOR ** 2

        arr_2d_xr_fullres = np.repeat(np.repeat((arr_sum / denom).values, settings.RESFACTOR, axis=1), settings.RESFACTOR, axis=0)
        arr_2d_xr_fullres = arr_2d_xr_fullres[:arr_2d.shape[0], :arr_2d.shape[1]]

        return arr_2d_xr_fullres[np.where(self.MASK_2D)]
    
    
    def get_resfactored_sum(self, arr: np.ndarray) -> np.ndarray:
        '''
        Calculate the sum for each resfactored cell, given the input arr is masked by the land-use mask
        (has a length of the number of cells in the full-resolution 1D array, i.e., 6956407).

        Args:
            arr (np.ndarray): A 1D array containing the values for the full-resolution array, should be the length
                of the number of cells in the full-resolution array (i.e., 6956407).
        Returns:
            np.ndarray: A 1D array containing the sum for each cell in the
            resfactored array, with the same length as the number of cells in the resfactored array.
        '''
        arr_2d = np.zeros_like(self.LUMAP_2D_FULLRES, dtype=np.float32)
        np.place(arr_2d, self.NLUM_MASK, arr)
        arr_2d_xr = xr.DataArray(arr_2d, dims=['y', 'x']) * self.LUMASK_2D_FULLRES  # Ensure outside LUTO cells are zeroed out and don't contribute to the sum
        arr_sum = arr_2d_xr.coarsen(x=settings.RESFACTOR, y=settings.RESFACTOR, boundary='pad').sum()
        result = arr_sum.values[self.COORD_ROW_COL_RESFACTORED[0], self.COORD_ROW_COL_RESFACTORED[1]]
        # At RF1, COORD_ROW_COL_RESFACTORED spans all NLUM cells (6956407); apply MASK to return only LUTO cells
        return result[self.MASK] if settings.RESFACTOR == 1 else result

    
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
            
        # Fill -1 with nearest neighbour values
        nearst_ind = distance_transform_edt(
            (lumap_resfactored == -1),
            return_distances=False,
            return_indices=True
        )
      
        return lumap_resfactored[*nearst_ind]



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
        
        # Get lumap of base year
        sim_year = sorted(set([self.YR_CAL_BASE]) | set(settings.SIM_YEARS)) 
        if yr_cal == self.YR_CAL_BASE:
            lumap = self.lumaps[self.YR_CAL_BASE]
        else:
            prev_year = sim_year[sim_year.index(yr_cal)-1]
            lumap = self.lumaps[prev_year]
                
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
            non_ag_quantity.get_quantity_matrix(self, ag_q_mrp_xr, lumap).astype(np.float32),
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
    
    
    def get_actual_production_lyr(self, yr_cal:int):
        '''
        Return the production data for a given year as xarray DataArrays.
        The returned DataArrays are spatial layers where each cell is the production of a commodity.
        Such as t/cell for apples, beef. Or ML/cell for milk.
        
        Note: the 'actual' means the production is calculated based on true decision variables, 
        meaning the production is calculated based on actual land-use areas.
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
        ag_X_mrj_xr = tools.ag_mrj_to_xr(self, ag_X_mrj).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, self.NCELLS)})
        non_ag_X_rk_xr = tools.non_ag_rk_to_xr(self, non_ag_X_rk).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, self.NCELLS)})
        ag_man_X_amrj_xr = tools.am_mrj_to_xr(self, ag_man_X_mrj).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, self.NCELLS)})

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
            dict: A dictionary with land use as keys and elasticity multipliers as values.
        '''

        # Return cached result if available for this year
        if yr_cal in self.DEMAND_ELASTICITY_MUL:
            return self.DEMAND_ELASTICITY_MUL[yr_cal]

        # Get supply delta (0-based ratio)
        supply_base_dvar_base_production = self.BASE_YR_production_t
        supply_base_dvar_target_production = self.get_production_from_base_dvar_under_target_CCI_and_yield_change(yr_cal)
        delta_supply = (supply_base_dvar_target_production - supply_base_dvar_base_production) / supply_base_dvar_base_production

        # Get demand delta (0-based ratio)
        demand_base_year = self.D_CY_xr.sel(year=self.YR_CAL_BASE)
        demand_target_year = self.D_CY_xr.sel(year=yr_cal)
        delta_demand = (demand_target_year - demand_base_year) / demand_base_year

        # Calculate price_multiplier (1-based ratio)
        price_delta = (delta_demand - delta_supply) / (self.elasticity_demand + self.elasticity_supply)
        elasticity_multiplier = (price_delta + 1).to_dataframe('multiplier')['multiplier'].to_dict()

        if settings.DYNAMIC_PRICE:
            self.DEMAND_ELASTICITY_MUL[yr_cal] = elasticity_multiplier
        else:
            self.DEMAND_ELASTICITY_MUL[yr_cal] = {k: 1 for k in elasticity_multiplier.keys()}

        return self.DEMAND_ELASTICITY_MUL[yr_cal]
    
    
    
    def get_watershed_yield_components(self, valid_watershed_id:list[int] = None):
        """
        Get the water yield components for the specified watersheds.
        """
        if settings.WATER_REGION_DEF == 'River Region':

            self.WATER_REGION_ID = pd.read_hdf(os.path.join(settings.INPUT_DIR, "rivreg_id.h5"), where=self.MASK).to_numpy()

            rr = pd.read_hdf(os.path.join(settings.INPUT_DIR, "rivreg_lut.h5"))
            self.WATER_REGION_NAMES = dict(zip(rr['HR_RIVREG_ID'], rr['HR_RIVREG_NAME']))  
            self.WATER_REGION_HIST_LEVEL = dict(zip(rr['HR_RIVREG_ID'], rr['WATER_YIELD_HIST_BASELINE_ML']))  
            
            rr_outside_luto = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_outside_LUTO_study_area_2010_2100_rr_ml.h5'))
            rr_outside_luto = rr_outside_luto.loc[:, pd.IndexSlice[:, settings.SSP]]
            rr_outside_luto.columns = rr_outside_luto.columns.droplevel('ssp')

            rr_natural_land = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_natural_land_2010_2100_rr_ml.h5'))
            rr_natural_land = rr_natural_land.loc[:, pd.IndexSlice[:, settings.SSP]]
            rr_natural_land.columns = rr_natural_land.columns.droplevel('ssp')
            rr_outside_luto = rr_outside_luto.reindex(columns=sorted(rr_outside_luto.columns))

            self.WATER_OUTSIDE_LUTO_BY_CCI = rr_outside_luto
            self.WATER_OUTSIDE_LUTO_HIST = self.WATER_YIELD_OUTSIDE_LUTO_HIST.query('Region_Type == "River Region"').set_index('Region_ID')['Water Yield (ML)'].to_dict()

        elif settings.WATER_REGION_DEF == 'Drainage Division':
            
            self.WATER_REGION_ID = pd.read_hdf(os.path.join(settings.INPUT_DIR, "draindiv_id.h5"), where=self.MASK).to_numpy()  # Drainage div ID mapped.

            dd = pd.read_hdf(os.path.join(settings.INPUT_DIR, "draindiv_lut.h5"))
            self.WATER_REGION_NAMES = dict(zip(dd['HR_DRAINDIV_ID'], dd['HR_DRAINDIV_NAME']))
            self.WATER_REGION_HIST_LEVEL = dict(zip(dd['HR_DRAINDIV_ID'], dd['WATER_YIELD_HIST_BASELINE_ML']))

            dd_outside_luto = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_outside_LUTO_study_area_2010_2100_dd_ml.h5'))
            dd_outside_luto = dd_outside_luto.loc[:, pd.IndexSlice[:, settings.SSP]]
            dd_outside_luto.columns = dd_outside_luto.columns.droplevel('ssp')

            dd_natural_land = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_natural_land_2010_2100_dd_ml.h5'))
            dd_natural_land = dd_natural_land.loc[:, pd.IndexSlice[:, settings.SSP]]
            dd_natural_land.columns = dd_natural_land.columns.droplevel('ssp')
            dd_natural_land = dd_natural_land.reindex(columns=sorted(dd_natural_land.columns))

            self.WATER_OUTSIDE_LUTO_BY_CCI = dd_outside_luto
            self.WATER_OUTSIDE_LUTO_HIST = self.WATER_YIELD_OUTSIDE_LUTO_HIST.query('Region_Type == "Drainage Division"').set_index('Region_ID')['Water Yield (ML)'].to_dict()
            
        else:
            raise ValueError(f"Unknown water region definition: {settings.WATER_REGION_DEF}. "
                        f"Must be either 'River Region' or 'Drainage Division'.")
            
            
        # Using high res factor will make some small river regions disappear; hence we update the data with validated river region ids
        if valid_watershed_id is None:
            valid_watershed_id = np.unique(self.WATER_REGION_ID)
        self.WATERSHED_DISAPPEARING = [self.WATER_REGION_NAMES[i] for i in set(self.WATER_REGION_NAMES.keys()) - set(valid_watershed_id)]
        
        if self.WATERSHED_DISAPPEARING:
            print(f"│   ⚠ WARNING: {len(self.WATERSHED_DISAPPEARING)} river regions are disappearing due to using high resolution factors.", flush=True)
            [print(f"       - {list(i)}") for i in np.array_split(self.WATERSHED_DISAPPEARING, len(self.WATERSHED_DISAPPEARING)//3)]
            self.WATER_REGION_NAMES = {k: v for k, v in self.WATER_REGION_NAMES.items() if k in valid_watershed_id}
            self.WATER_REGION_HIST_LEVEL = {k: v for k, v in self.WATER_REGION_HIST_LEVEL.items() if k in valid_watershed_id}
            self.WATER_OUTSIDE_LUTO_BY_CCI = self.WATER_OUTSIDE_LUTO_BY_CCI.loc[:, self.WATER_OUTSIDE_LUTO_BY_CCI.columns.isin(valid_watershed_id)]
            self.WATER_OUTSIDE_LUTO_HIST = {k: v for k, v in self.WATER_OUTSIDE_LUTO_HIST.items() if k in valid_watershed_id}
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
        
        bio_habitat_score_baseline_sum = (self.BIO_GBF2_MASK * self.REAL_AREA * self.AG_MASK_PROPORTION_R).sum()
        bio_habitat_score_base_yr_sum = self.BIO_GBF2_BASE_YR.sum()
        bio_habitat_score_base_yr_proportion = bio_habitat_score_base_yr_sum / bio_habitat_score_baseline_sum

        bio_habitat_target_proportion = [
            bio_habitat_score_base_yr_proportion + ((1 - bio_habitat_score_base_yr_proportion) * i)
            for i in settings.GBF2_TARGETS_DICT[settings.BIODIVERSITY_TARGET_GBF_2].values()
        ]

        targets_key_years = {
            self.YR_CAL_BASE: bio_habitat_score_base_yr_sum, 
            **dict(zip(
                settings.GBF2_TARGETS_DICT[settings.BIODIVERSITY_TARGET_GBF_2].keys(), 
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
    
    
    def get_GBF3_NVIS_limit_score_inside_LUTO_by_yr(self, yr: int) -> np.ndarray:
        '''
        Interpolate GBF3 NVIS vegetation targets for the given year.

        Args:
            yr: Target year

        Returns:
            Array of NVIS vegetation target scores inside LUTO area
        '''
        GBF3_NVIS_target_percents = xr.DataArray(
            np.zeros(len(self.BIO_GBF3_NVIS_BASELINE_AND_TARGETS), dtype=np.float32), 
            dims=['layer'],
            coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF3_NVIS_SEL, names=['region', 'group'])}
        )
        
        for _, row in self.BIO_GBF3_NVIS_BASELINE_AND_TARGETS.iterrows():
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
        snes_df = pd.concat([
            self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY,
            self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE,
        ], ignore_index=True)

        result = xr.DataArray(
            np.zeros(len(self.BIO_GBF4_SNES_SEL), dtype=np.float32),
            dims=['layer'],
            coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF4_SNES_SEL, names=['region', 'species'])},
        )
        for idx, row in snes_df.iterrows():
            layer = self.BIO_GBF4_PRESENCE_SNES_SEL[idx]
            region, species = self.BIO_GBF4_SNES_SEL[idx]
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [min(row[f'BASEYEAR_LEVEL_{layer}'], row[f'TARGET_LEVEL_2030_{layer}']),
                 row[f'TARGET_LEVEL_2030_{layer}'], row[f'TARGET_LEVEL_2050_{layer}'], row[f'TARGET_LEVEL_2100_{layer}']],
                kind='linear', fill_value='extrapolate',
            )
            interp_pct = float(f(yr))
            attainable_pct = row[f'ATTAINABLE_LEVEL_{layer}']
            target_pct = min(interp_pct, attainable_pct)
            if interp_pct > attainable_pct:
                print(f"│   ├── SNES target capped for '{species}' ({layer}): {interp_pct:.2f}% -> {attainable_pct:.2f}% (attainable limit)", flush=True)
            score_all_aus = row[f'BASELINE_LEVEL_ALL_AUSTRALIA_{layer}'] * target_pct / 100
            score_out_LUTO = row[f'BASEYEAR_SCORE_OUT_LUTO_NATURAL_{layer}']
            result.loc[dict(layer=(region, species))] = score_all_aus - score_out_LUTO
        return result


    def get_GBF4_ECNES_target_inside_LUTO_by_year(self, yr: int) -> xr.DataArray:
        ecnes_df = pd.concat([
            self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY,
            self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE,
        ], ignore_index=True)

        result = xr.DataArray(
            np.zeros(len(self.BIO_GBF4_ECNES_SEL), dtype=np.float32),
            dims=['layer'],
            coords={'layer': pd.MultiIndex.from_tuples(self.BIO_GBF4_ECNES_SEL, names=['region', 'species'])},
        )
        for idx, row in ecnes_df.iterrows():
            layer = self.BIO_GBF4_PRESENCE_ECNES_SEL[idx]
            region, species = self.BIO_GBF4_ECNES_SEL[idx]
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [min(row[f'BASEYEAR_LEVEL_{layer}'], row[f'TARGET_LEVEL_2030_{layer}']),
                 row[f'TARGET_LEVEL_2030_{layer}'], row[f'TARGET_LEVEL_2050_{layer}'], row[f'TARGET_LEVEL_2100_{layer}']],
                kind='linear', fill_value='extrapolate',
            )
            interp_pct = float(f(yr))
            attainable_pct = row[f'ATTAINABLE_LEVEL_{layer}']
            target_pct = min(interp_pct, attainable_pct)
            if interp_pct > attainable_pct:
                print(f"│   ├── ECNES target capped for '{species}' ({layer}): {interp_pct:.2f}% -> {attainable_pct:.2f}% (attainable limit)", flush=True)
            score_all_aus = row[f'BASELINE_LEVEL_ALL_AUSTRALIA_{layer}'] * target_pct / 100
            score_out_LUTO = row[f'BASEYEAR_SCORE_OUT_LUTO_NATURAL_{layer}']
            result.loc[dict(layer=(region, species))] = score_all_aus - score_out_LUTO
        return result


    def get_GBF8_bio_layers_by_yr(self, yr: int, level:Literal['species', 'group']='species'):
        '''
        Get the biodiversity suitability score [hectare weighted] for each species at the given year.
        
        The raw biodiversity suitability score [2D (shape, 808*978), (dtype, uint8, 0-100)] represents the 
        suitability of each cell for each species/group.  Here it is LINEARLY interpolated to the given year,
        then LINEARLY interpolated to the given spatial coordinates.
        
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
        
        input_lr = self.BIO_GBF8_SPECIES_LAYER if level == 'species' else self.BIO_GBF8_GROUPS_LAYER
        
        current_species_val = input_lr.interp(                          # Here the year interpolation is done first                      
            year=yr,
            method='linear', 
            kwargs={'fill_value': 'extrapolate'}
        ).interp(                                                       # Then the spatial interpolation and masking is done
            x=xr.DataArray(self.COORD_LON_LAT[0], dims='cell'),
            y=xr.DataArray(self.COORD_LON_LAT[1], dims='cell'),
            method='linear'                                             # Use LINEAR interpolation
        ).drop_vars(['year']).values
        
        # Apply Savanna Burning penalties
        current_species_val = np.where(
            self.SAVBURN_ELIGIBLE,
            current_species_val * settings.BIO_CONTRIBUTION_LDS,
            current_species_val
        )
        
        return current_species_val.astype(np.float32)
    

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
        Only active under REGIONAL_ADOPTION_CONSTRAINTS == 'on'.

        Return a list of tuples where each tuple contains 
        - the region ID, 
        - landuse name, 
        - the adoption percentage.
        """
        if settings.REGIONAL_ADOPTION_CONSTRAINTS != "on":
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
        Only active under REGIONAL_ADOPTION_CONSTRAINTS == 'on'.

        Return a list of tuples where each tuple contains
        - the region ID,
        - landuse name,
        - the adoption area (ha).
        """
        if settings.REGIONAL_ADOPTION_CONSTRAINTS != "on":
            return ()
        
        reg_adop_limits = self.get_regional_adoption_percent_by_year(yr)
        reg_adop_limits_ha = []
        for reg, landuse, pct in reg_adop_limits:
            reg_total_area_ha = ((self.REGIONAL_ADOPTION_ZONES == reg) * self.REAL_AREA).sum()
            reg_adop_limits_ha.append((reg, landuse, reg_total_area_ha * pct / 100))
            
        return reg_adop_limits_ha

    def get_regional_adoption_non_ag_sum_limit_ha_by_year(self, yr: int):
        """
        Under REGIONAL_ADOPTION_CONSTRAINTS == 'NON_AG_CAP', return per-region area caps
        (ha) on the SUM of all non-ag land uses.

        The region partition is selected by REGIONAL_ADOPTION_NON_AG_REGION:
        - 'NRM'   uses self.REGION_NRM_NAME
        - 'State' uses self.REGION_STATE_NAME

        Returns a list of (region_name, reg_ind, area_limit_ha) tuples. The percentage
        cap (REGIONAL_ADOPTION_NON_AG_CAP) is uniform across all regions and years.
        """
        if settings.REGIONAL_ADOPTION_CONSTRAINTS != 'NON_AG_CAP':
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

        pct = settings.REGIONAL_ADOPTION_NON_AG_CAP
        limits = []
        for reg in np.unique(region_arr):
            reg_ind = np.where(region_arr == reg)[0]
            reg_total_area_ha = self.REAL_AREA[reg_ind].sum()
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

    def get_water_nl_yield_for_yr_idx(
        self,
        yr_idx: int,
        water_dr_yield: Optional[np.ndarray] = None,
        water_sr_yield: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get the net land water yield array, ? inclusive of all cells that LUTO does not look at.?

        Returns
        -------
        np.ndarray: shape (NCELLS,)
        """
        water_dr_yield = (
            water_dr_yield if water_dr_yield is not None
            else self.WATER_YIELD_DR_FILE[yr_idx]
        )
        water_sr_yield = (
            water_sr_yield if water_sr_yield is not None
            else self.WATER_YIELD_SR_FILE[yr_idx]
        )
        dr_prop = self.DEEP_ROOTED_PROPORTION

        return (dr_prop * water_dr_yield + (1 - dr_prop) * water_sr_yield)
