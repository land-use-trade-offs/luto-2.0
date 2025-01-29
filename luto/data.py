# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
#
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>.


import os
import h5py

import xarray as xr
import numpy as np
import pandas as pd
import rasterio

import luto.settings as settings
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.non_agricultural.quantity as non_ag_quantity

from itertools import product
from collections import defaultdict
from joblib import Parallel, delayed

from dataclasses import dataclass
from typing import Any, Optional
from affine import Affine
from scipy.interpolate import interp1d
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.settings import INPUT_DIR, NON_AG_LAND_USES_REVERSIBLE, OUTPUT_DIR
from luto.tools.spatializers import upsample_array



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
    for am in AG_MANAGEMENTS_TO_LAND_USES:
        am_vars[am] = np.zeros((ncms, ncells, n_ag_lus))

    return am_vars


def lumap2ag_l_mrj(lumap, lmmap):
    """
    Return land-use maps in decision-variable (X_mrj) format.
    Where 'm' is land mgt, 'r' is cell, and 'j' is agricultural land-use.

    Cells used for non-agricultural land uses will have value 0 for all agricultural
    land uses, i.e. all r.
    """
    # Set up a container array of shape m, r, j.
    x_mrj = np.zeros((2, lumap.shape[0], 28), dtype=bool)   # TODO - remove 2

    # Populate the 3D land-use, land mgt mask.
    for j in range(28):
        # One boolean map for each land use.
        jmap = np.where(lumap == j, True, False).astype(bool)
        # Keep only dryland version.
        x_mrj[0, :, j] = np.where(lmmap == False, jmap, False)
        # Keep only irrigated version.
        x_mrj[1, :, j] = np.where(lmmap == True, jmap, False)

    return x_mrj.astype(bool)


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

    def __init__(self, timestamp: str) -> None:
        """
        Sets up output containers (lumaps, lmmaps, etc) and loads all LUTO data, adjusted
        for resfactor.
        """
        # Path for write module - overwrite when provided with a base and target year
        self.path = None
        # Timestamp of simulation to which this object belongs - will be updated each time a simulation
        # is run using this Data object.
        self.timestamp_sim = timestamp

        # Setup output containers
        self.lumaps = {}
        self.lmmaps = {}
        self.ammaps = {}
        self.ag_dvars = {}
        self.non_ag_dvars = {}
        self.ag_man_dvars = {}
        self.prod_data = {}
        self.obj_vals = {}

        # Containers for reprojected dvar data
        self.ag_dvars_2D_reproj_match = {}
        self.non_ag_dvars_2D_reproj_match = {}
        self.ag_man_dvars_2D_reproj_match = {}

        print('')
        print('Beginning data initialisation...')

        self.YR_CAL_BASE = 2010  # The base year, i.e. where year index yr_idx == 0.



        ###############################################################
        # Masking and spatial coarse graining.
        ###############################################################
        print("\tSetting up masking and spatial course graining data...", flush=True)

        # Set resfactor multiplier
        self.RESMULT = settings.RESFACTOR ** 2

        # Set the nodata and non-ag code
        self.NODATA = -9999
        self.MASK_LU_CODE = -1

        # Load LUMAP without resfactor
        self.LUMAP_NO_RESFACTOR = pd.read_hdf(os.path.join(INPUT_DIR, "lumap.h5")).to_numpy()                   # 1D (ij flattend),  0-27 for land uses; -1 for non-agricultural land uses; All cells in Australia (land only)

        # NLUM mask.
        with rasterio.open(os.path.join(INPUT_DIR, "NLUM_2010-11_mask.tif")) as rst:
            self.NLUM_MASK = rst.read(1).astype(np.int8)                                                        # 2D map,  0 for ocean, 1 for land
            self.LUMAP_2D = np.full_like(self.NLUM_MASK, self.NODATA, dtype=np.int16)                           # 2D map,  full of nodata (-9999)
            np.place(self.LUMAP_2D, self.NLUM_MASK == 1, self.LUMAP_NO_RESFACTOR)                               # 2D map,  -9999 for ocean; -1 for desert, urban, water, etc; 0-27 for land uses
            self.GEO_META_FULLRES = rst.meta                                                                    # dict,  key-value pairs of geospatial metadata for the full resolution land-use map
            self.GEO_META_FULLRES['dtype'] = 'float32'                                                          # Set the data type to float32
            self.GEO_META_FULLRES['nodata'] = self.NODATA                                                       # Set the nodata value to -9999

        # Mask out non-agricultural, non-environmental plantings land (i.e., -1) from lumap (True means included cells. Boolean dtype.)
        self.LUMASK = self.LUMAP_NO_RESFACTOR != self.MASK_LU_CODE                                              # 1D (ij flattend);  `True` for land uses; `False` for desert, urban, water, etc

        # Get the lon/lat coordinates.
        self.COORD_LON_LAT = self.get_coord(np.nonzero(self.NLUM_MASK), self.GEO_META_FULLRES['transform'])             # 2D array([lon, ...], [lat, ...]);  lon/lat coordinates for each cell in Australia (land only)

        # Return combined land-use and resfactor mask
        if settings.RESFACTOR > 1:

            # Create settings.RESFACTOR mask for spatial coarse-graining.
            rf_mask = self.NLUM_MASK.copy()
            nonzeroes = np.nonzero(rf_mask)
            rf_mask[int(settings.RESFACTOR/2)::settings.RESFACTOR, int(settings.RESFACTOR/2)::settings.RESFACTOR] = 0
            resmask = np.where(rf_mask[nonzeroes] == 0, True, False)

            # Superimpose resfactor mask upon land-use map mask (Boolean).
            self.MASK = self.LUMASK * resmask

            # Get the resfactored 2D lumap and x/y coordinates.
            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D[int(settings.RESFACTOR/2)::settings.RESFACTOR, int(settings.RESFACTOR/2)::settings.RESFACTOR]

            # Get the resfactored lon/lat coordinates.
            self.COORD_LON_LAT = self.COORD_LON_LAT[0][self.MASK], self.COORD_LON_LAT[1][self.MASK]

            # Update the geospatial metadata.
            self.GEO_META = self.update_geo_meta()


        elif settings.RESFACTOR == 1:
            self.MASK = self.LUMASK
            self.GEO_META = self.GEO_META_FULLRES

        else:
            raise KeyError("Resfactor setting invalid")



        ###############################################################
        # Load Xarray reference data.
        ###############################################################
        print("\tLoading reference Xarray for reproject decision variables...", flush=True)
        self.REPROJECT_TARGET_ID_MAP = xr.load_dataset(f'{settings.INPUT_DIR}/bio_id_map.nc')['data'].compute()
        self.REPROJECT_REFERENCE_MAP = xr.load_dataset(f'{settings.INPUT_DIR}/bio_mask.nc')['data'].compute()



        ###############################################################
        # Load agricultural crop and livestock data.
        ###############################################################
        print("\tLoading agricultural crop and livestock data...", flush=True)
        self.AGEC_CROPS = self.get_df_resfactor_applied(pd.read_hdf(os.path.join(INPUT_DIR, "agec_crops.h5")))
        self.AGEC_LVSTK = self.get_df_resfactor_applied(pd.read_hdf(os.path.join(INPUT_DIR, "agec_lvstk.h5")))
        
        # Price multipliers for livestock and crops over the years.
        self.CROP_PRICE_MULTIPLIERS = pd.read_excel(os.path.join(INPUT_DIR, "ag_price_multipliers.xlsx"), sheet_name="AGEC_CROPS", index_col="Year")
        self.LVSTK_PRICE_MULTIPLIERS = pd.read_excel(os.path.join(INPUT_DIR, "ag_price_multipliers.xlsx"), sheet_name="AGEC_LVSTK", index_col="Year")



        ###############################################################
        # Set up lists of land-uses, commodities etc.
        ###############################################################
        print("\tSetting up lists of land uses, commodities, etc...", flush=True)

        # Read in lexicographically ordered list of land-uses.
        self.AGRICULTURAL_LANDUSES = pd.read_csv((os.path.join(INPUT_DIR, 'ag_landuses.csv')), header = None)[0].to_list()
        self.NON_AGRICULTURAL_LANDUSES = list(settings.NON_AG_LAND_USES.keys())

        self.NONAGLU2DESC = dict(zip(range(settings.NON_AGRICULTURAL_LU_BASE_CODE,
                                    settings.NON_AGRICULTURAL_LU_BASE_CODE + len(self.NON_AGRICULTURAL_LANDUSES)),
                            self.NON_AGRICULTURAL_LANDUSES))

        self.DESC2NONAGLU = {value: key for key, value in self.NONAGLU2DESC.items()}

        # Get number of land-uses
        self.N_AG_LUS = len(self.AGRICULTURAL_LANDUSES)
        self.N_NON_AG_LUS = len(self.NON_AGRICULTURAL_LANDUSES)

        # Construct land-use index dictionary (distinct from LU_IDs!)
        self.AGLU2DESC = {i: lu for i, lu in enumerate(self.AGRICULTURAL_LANDUSES)}
        self.DESC2AGLU = {value: key for key, value in self.AGLU2DESC.items()}
        self.AGLU2DESC[-1] = 'Non-agricultural land'

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
        self.PR_LVSTK = [ s.upper() + ' ' + p
                          for s in self.LU_LVSTK if 'DAIRY' not in s.upper()
                          for p in ['LEXP', 'MEAT'] ]
        self.PR_LVSTK += [s.upper() for s in self.LU_LVSTK if 'DAIRY' in s.upper()]
        self.PR_LVSTK += [s.upper() + ' WOOL' for s in self.LU_LVSTK if 'SHEEP' in s.upper()]
        self.PRODUCTS = self.PR_CROPS + self.PR_LVSTK
        self.PRODUCTS.sort() # Ensure lexicographic order.

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
        
        
        # Get the land-use indices for each commodity.
        self.CM2LU_IDX = defaultdict(list)
        for c in self.COMMODITIES:
            for lu in self.AGRICULTURAL_LANDUSES:
                if lu.split(' -')[0].lower() in c:
                    self.CM2LU_IDX[c].append(self.AGRICULTURAL_LANDUSES.index(lu))



        ###############################################################
        # Spatial layers.
        ###############################################################
        print("\tSetting up spatial layers data...", flush=True)

        # Actual hectares per cell, including projection corrections.
        self.REAL_AREA_NO_RESFACTOR = pd.read_hdf(os.path.join(INPUT_DIR, "real_area.h5")).to_numpy()
        self.REAL_AREA = self.get_array_resfactor_applied(self.REAL_AREA_NO_RESFACTOR) * self.RESMULT

        # Derive NCELLS (number of spatial cells) from the area array.
        self.NCELLS = self.REAL_AREA.shape[0]
        
        # Initial (2010) ag decision variable (X_mrj).
        self.LMMAP_NO_RESFACTOR = pd.read_hdf(os.path.join(INPUT_DIR, "lmmap.h5")).to_numpy()
        self.AG_L_MRJ = self.get_exact_resfactored_lumap_mrj() 
        self.add_ag_dvars(self.YR_CAL_BASE, self.AG_L_MRJ)

        # Initial (2010) land-use map, mapped as lexicographic land-use class indices.
        self.LUMAP = self.AG_L_MRJ.sum(axis=0).argmax(axis=1).astype("int8")
        self.add_lumap(self.YR_CAL_BASE, self.LUMAP)

        # Initial (2010) land management map.
        self.LMMAP = self.get_array_resfactor_applied(self.LMMAP_NO_RESFACTOR)
        self.add_lmmap(self.YR_CAL_BASE, self.LMMAP)

        # Initial (2010) agricutural management maps - no cells are used for alternative agricultural management options.
        # Includes a separate AM map for each agricultural management option, because they can be stacked.
        self.AMMAP_DICT = {
            am: np.zeros(self.NCELLS).astype("int8") for am in AG_MANAGEMENTS_TO_LAND_USES
        }
        self.add_ammaps(self.YR_CAL_BASE, self.AMMAP_DICT)

        

        self.NON_AG_L_RK = lumap2non_ag_l_mk(
            self.LUMAP, len(self.NON_AGRICULTURAL_LANDUSES)     # Int8
        )
        self.add_non_ag_dvars(self.YR_CAL_BASE, self.NON_AG_L_RK)


        ###############################################################
        # Climate change impact data.
        ###############################################################
        print("\tLoading climate change data...", flush=True)

        self.CLIMATE_CHANGE_IMPACT = pd.read_hdf(
            os.path.join(INPUT_DIR, "climate_change_impacts_" + settings.RCP + "_CO2_FERT_" + settings.CO2_FERT.upper() + ".h5")
        )



        ###############################################################
        # Livestock related data.
        ###############################################################
        print("\tLoading livestock related data...", flush=True)

        self.FEED_REQ = np.nan_to_num(
            pd.read_hdf(os.path.join(INPUT_DIR, "feed_req.h5")).to_numpy()
        )
        self.PASTURE_KG_DM_HA = pd.read_hdf(
            os.path.join(INPUT_DIR, "pasture_kg_dm_ha.h5")
        ).to_numpy()
        self.SAFE_PUR_NATL = pd.read_hdf(os.path.join(INPUT_DIR, "safe_pur_natl.h5")).to_numpy()
        self.SAFE_PUR_MODL = pd.read_hdf(os.path.join(INPUT_DIR, "safe_pur_modl.h5")).to_numpy()



        ###############################################################
        # Agricultural management options data.
        ###############################################################
        print("\tLoading agricultural management options' data...", flush=True)

        # Asparagopsis taxiformis data
        asparagopsis_file = os.path.join(INPUT_DIR, "20231101_Bundle_MR.xlsx")
        self.ASPARAGOPSIS_DATA = {}
        self.ASPARAGOPSIS_DATA["Beef - natural land"] = pd.read_excel(
            asparagopsis_file, sheet_name="MR bundle (ext cattle)", index_col="Year"
        )
        self.ASPARAGOPSIS_DATA["Beef - modified land"] = pd.read_excel(
            asparagopsis_file, sheet_name="MR bundle (int cattle)", index_col="Year"
        )
        self.ASPARAGOPSIS_DATA["Sheep - natural land"] = pd.read_excel(
            asparagopsis_file, sheet_name="MR bundle (sheep)", index_col="Year"
        )
        self.ASPARAGOPSIS_DATA["Sheep - modified land"] = self.ASPARAGOPSIS_DATA[
            "Sheep - natural land"
        ]
        self.ASPARAGOPSIS_DATA["Dairy - natural land"] = pd.read_excel(
            asparagopsis_file, sheet_name="MR bundle (dairy)", index_col="Year"
        )
        self.ASPARAGOPSIS_DATA["Dairy - modified land"] = self.ASPARAGOPSIS_DATA[
            "Dairy - natural land"
        ]

        # Precision agriculture data
        prec_agr_file = os.path.join(INPUT_DIR, "20231101_Bundle_AgTech_NE.xlsx")
        self.PRECISION_AGRICULTURE_DATA = {}
        int_cropping_data = pd.read_excel(
            prec_agr_file, sheet_name="AgTech NE bundle (int cropping)", index_col="Year"
        )
        cropping_data = pd.read_excel(
            prec_agr_file, sheet_name="AgTech NE bundle (cropping)", index_col="Year"
        )
        horticulture_data = pd.read_excel(
            prec_agr_file, sheet_name="AgTech NE bundle (horticulture)", index_col="Year"
        )

        for lu in [
            "Hay",
            "Summer cereals",
            "Summer legumes",
            "Summer oilseeds",
            "Winter cereals",
            "Winter legumes",
            "Winter oilseeds",
        ]:
            # Cropping land uses
            self.PRECISION_AGRICULTURE_DATA[lu] = cropping_data

        for lu in ["Cotton", "Other non-cereal crops", "Rice", "Sugar", "Vegetables"]:
            # Intensive Cropping land uses
            self.PRECISION_AGRICULTURE_DATA[lu] = int_cropping_data

        for lu in [
            "Apples",
            "Citrus",
            "Grapes",
            "Nuts",
            "Pears",
            "Plantation fruit",
            "Stone fruit",
            "Tropical stone fruit",
        ]:
            # Horticulture land uses
            self.PRECISION_AGRICULTURE_DATA[lu] = horticulture_data

        # Ecological grazing data
        eco_grazing_file = os.path.join(INPUT_DIR, "20231107_ECOGRAZE_Bundle.xlsx")
        self.ECOLOGICAL_GRAZING_DATA = {}
        self.ECOLOGICAL_GRAZING_DATA["Beef - modified land"] = pd.read_excel(
            eco_grazing_file, sheet_name="Ecograze bundle (ext cattle)", index_col="Year"
        )
        self.ECOLOGICAL_GRAZING_DATA["Sheep - modified land"] = pd.read_excel(
            eco_grazing_file, sheet_name="Ecograze bundle (sheep)", index_col="Year"
        )
        self.ECOLOGICAL_GRAZING_DATA["Dairy - modified land"] = pd.read_excel(
            eco_grazing_file, sheet_name="Ecograze bundle (dairy)", index_col="Year"
        )

        # Load soil carbon data, convert C to CO2e (x 44/12), and average over years
        self.SOIL_CARBON_AVG_T_CO2_HA = self.get_array_resfactor_applied(
            pd.read_hdf(os.path.join(INPUT_DIR, "soil_carbon_t_ha.h5")).to_numpy(dtype=np.float32)
            * (44 / 12)
            / settings.SOC_AMORTISATION
        )

        # Load AgTech EI data
        prec_agr_file = os.path.join(INPUT_DIR, '20231107_Bundle_AgTech_EI.xlsx')
        self.AGTECH_EI_DATA = {}
        int_cropping_data = pd.read_excel( prec_agr_file, sheet_name='AgTech EI bundle (int cropping)', index_col='Year' )
        cropping_data = pd.read_excel( prec_agr_file, sheet_name='AgTech EI bundle (cropping)', index_col='Year' )
        horticulture_data = pd.read_excel( prec_agr_file, sheet_name='AgTech EI bundle (horticulture)', index_col='Year' )

        for lu in ['Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds',
                'Winter cereals', 'Winter legumes', 'Winter oilseeds']:
            # Cropping land uses
            self.AGTECH_EI_DATA[lu] = cropping_data

        for lu in ['Cotton', 'Other non-cereal crops', 'Rice', 'Sugar', 'Vegetables']:
            # Intensive Cropping land uses
            self.AGTECH_EI_DATA[lu] = int_cropping_data

        for lu in ['Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears',
                'Plantation fruit', 'Stone fruit', 'Tropical stone fruit']:
            # Horticulture land uses
            self.AGTECH_EI_DATA[lu] = horticulture_data

        # Load BioChar data
        biochar_file = os.path.join(INPUT_DIR, '20240918_Bundle_BC.xlsx')
        self.BIOCHAR_DATA = {}
        cropping_data = pd.read_excel( biochar_file, sheet_name='Biochar (cropping)', index_col='Year' )
        horticulture_data = pd.read_excel( biochar_file, sheet_name='Biochar (horticulture)', index_col='Year' )

        for lu in ['Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds',
                'Winter cereals', 'Winter legumes', 'Winter oilseeds']:
            # Cropping land uses
            self.BIOCHAR_DATA[lu] = cropping_data

        for lu in ['Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears',
                'Plantation fruit', 'Stone fruit', 'Tropical stone fruit']:
            # Horticulture land uses
            self.BIOCHAR_DATA[lu] = horticulture_data



        ###############################################################
        # Productivity data.
        ###############################################################
        print("\tLoading productivity data...", flush=True)

        # Yield increases.
        fpath = os.path.join(INPUT_DIR, "yieldincreases_bau2022.csv")
        self.BAU_PROD_INCR = pd.read_csv(fpath, header=[0, 1]).astype(np.float32)



        ###############################################################
        # Apply resfactor to various required data arrays 
        # and Calculate base year production 
        ###############################################################
        self.CLIMATE_CHANGE_IMPACT = self.get_df_resfactor_applied(self.CLIMATE_CHANGE_IMPACT)
        self.FEED_REQ = self.get_array_resfactor_applied(self.FEED_REQ)
        self.PASTURE_KG_DM_HA = self.get_array_resfactor_applied(self.PASTURE_KG_DM_HA)
        self.SAFE_PUR_MODL = self.get_array_resfactor_applied(self.SAFE_PUR_MODL)
        self.SAFE_PUR_NATL = self.get_array_resfactor_applied(self.SAFE_PUR_NATL)

        self.AG_MAN_L_MRJ_DICT = get_base_am_vars(self.NCELLS, self.NLMS, self.N_AG_LUS)
        self.add_ag_man_dvars(self.YR_CAL_BASE, self.AG_MAN_L_MRJ_DICT)
        
        print("\tCalculating base year productivity...", flush=True)
        yr_cal_base_prod_data = self.get_production(self.YR_CAL_BASE, self.LUMAP, self.LMMAP)
        self.add_production_data(self.YR_CAL_BASE, "Production", yr_cal_base_prod_data)



        ###############################################################
        # Auxiliary Spatial Layers
        # (spatial layers not required for production calculation)
        ###############################################################
        print("\tLoading auxiliary spatial layers data...", flush=True)

        # Load stream length data in metres of stream per cell
        self.STREAM_LENGTH = pd.read_hdf(
            os.path.join(INPUT_DIR, "stream_length_m_cell.h5")
        ).to_numpy()

        # Calculate the proportion of the area of each cell within stream buffer (convert REAL_AREA from ha to m2 and divide m2 by m2)
        self.RP_PROPORTION = self.get_array_resfactor_applied(
            ((2 * settings.RIPARIAN_PLANTING_BUFFER_WIDTH * self.STREAM_LENGTH) / (self.REAL_AREA_NO_RESFACTOR * 10000)).astype(np.float32)
        )
        # Calculate the length of fencing required for each cell in per hectare terms for riparian plantings
        self.RP_FENCING_LENGTH = self.get_array_resfactor_applied(
            ((2 * settings.RIPARIAN_PLANTING_TORTUOSITY_FACTOR * self.STREAM_LENGTH) / self.REAL_AREA_NO_RESFACTOR).astype(np.float32)
        )


        # Initial reprojected dvars data (2D xarray, ).
        self.add_ag_dvars_xr(self.YR_CAL_BASE, self.AG_L_MRJ)
        self.add_am_dvars_xr(self.YR_CAL_BASE, self.AG_MAN_L_MRJ_DICT)
        self.add_non_ag_dvars_xr(self.YR_CAL_BASE, self.NON_AG_L_RK)


        ###############################################################
        # Additional agricultural economic data.
        ###############################################################
        print("\tLoading additional agricultural economic data...", flush=True)


        # Load greenhouse gas emissions from agriculture
        self.AGGHG_CROPS = self.get_df_resfactor_applied(
            pd.read_hdf(os.path.join(INPUT_DIR, "agGHG_crops.h5"))
        )
        self.AGGHG_LVSTK = self.get_df_resfactor_applied(
            pd.read_hdf(os.path.join(INPUT_DIR, "agGHG_lvstk.h5"))
        )
        self.AGGHG_IRRPAST = self.get_array_resfactor_applied(
            pd.read_hdf(os.path.join(INPUT_DIR, "agGHG_irrpast.h5"))
        )

        # Raw transition cost matrix. In AUD/ha and ordered lexicographically.
        self.AG_TMATRIX = np.load(os.path.join(INPUT_DIR, "ag_tmatrix.npy"))
        
        # Apply penalty if a transition was occur from natural to modified land.
        for i,j in product(range(self.N_AG_LUS), range(self.N_AG_LUS)):
            if (not i in self.LU_MODIFIED_LAND) and (j in self.LU_MODIFIED_LAND):
                self.AG_TMATRIX[i,j] += settings.NATURAL_TO_MODIFIED_LAND_PENALTY
        
        # Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
        self.EXCLUDE = np.load(os.path.join(INPUT_DIR, "x_mrj.npy"))
        self.EXCLUDE = self.EXCLUDE[:, self.MASK, :]  # Apply resfactor specially for the exclude matrix



        ###############################################################
        # Non-agricultural data.
        ###############################################################
        print("\tLoading non-agricultural data...", flush=True)

        # Load plantings economic data
        self.EP_EST_COST_HA = self.get_array_resfactor_applied(
            pd.read_hdf(os.path.join(INPUT_DIR, "ep_est_cost_ha.h5")).to_numpy(dtype=np.float32)
        )
        self.CP_EST_COST_HA = self.get_array_resfactor_applied(
            pd.read_hdf(os.path.join(INPUT_DIR, "cp_est_cost_ha.h5")).to_numpy(dtype=np.float32)
        )

        # Load fire risk data (reduced carbon sequestration by this amount)
        fr_df = pd.read_hdf(os.path.join(INPUT_DIR, "fire_risk.h5"))
        fr_dict = {"low": "FD_RISK_PERC_5TH", "med": "FD_RISK_MEDIAN", "high": "FD_RISK_PERC_95TH"}
        fire_risk = fr_df[fr_dict[settings.FIRE_RISK]]

        # Load environmental plantings (block) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ep_df = pd.read_hdf(os.path.join(INPUT_DIR, "ep_block_avg_t_co2_ha_yr.h5"))
        self.EP_BLOCK_AVG_T_CO2_HA = self.get_array_resfactor_applied(
            (
                ep_df.EP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
                + ep_df.EP_BLOCK_BG_AVG_T_CO2_HA_YR
            ).to_numpy(dtype=np.float32)
        )

        # Load environmental plantings (belt) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ep_df = pd.read_hdf(os.path.join(INPUT_DIR, "ep_belt_avg_t_co2_ha_yr.h5"))
        self.EP_BELT_AVG_T_CO2_HA = self.get_array_resfactor_applied(
            (
                (ep_df.EP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL))
                + ep_df.EP_BELT_BG_AVG_T_CO2_HA_YR
            ).to_numpy(dtype=np.float32)
        )

        # Load environmental plantings (riparian) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ep_df = pd.read_hdf(os.path.join(INPUT_DIR, "ep_rip_avg_t_co2_ha_yr.h5"))
        self.EP_RIP_AVG_T_CO2_HA = self.get_array_resfactor_applied(
            (
                (ep_df.EP_RIP_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL))
                + ep_df.EP_RIP_BG_AVG_T_CO2_HA_YR
            ).to_numpy(dtype=np.float32)
        )

        # Load carbon plantings (block) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        cp_df = pd.read_hdf(os.path.join(INPUT_DIR, "cp_block_avg_t_co2_ha_yr.h5"))
        self.CP_BLOCK_AVG_T_CO2_HA = self.get_array_resfactor_applied(
            (
                (cp_df.CP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL))
                + cp_df.CP_BLOCK_BG_AVG_T_CO2_HA_YR
            ).to_numpy(dtype=np.float32)
        )

        # Load farm forestry [i.e. carbon plantings (belt)] GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        cp_df = pd.read_hdf(os.path.join(INPUT_DIR, "cp_belt_avg_t_co2_ha_yr.h5"))
        self.CP_BELT_AVG_T_CO2_HA = self.get_array_resfactor_applied(
            (
                (cp_df.CP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL))
                + cp_df.CP_BELT_BG_AVG_T_CO2_HA_YR
            ).to_numpy(dtype=np.float32)
        )

        # Agricultural land use to plantings raw transition costs:
        self.AG2EP_TRANSITION_COSTS_HA = np.load(
            os.path.join(INPUT_DIR, "ag_to_ep_tmatrix.npy")
        )  # shape: (28,)

        # EP to agricultural land use transition costs:
        self.EP2AG_TRANSITION_COSTS_HA = np.load(
            os.path.join(INPUT_DIR, "ep_to_ag_tmatrix.npy")
        )  # shape: (28,)


        ###############################################################
        # Water data.
        ###############################################################
        print("\tLoading water data...", flush=True)
        
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
        self.WATER_LICENCE_PRICE = self.get_array_resfactor_applied(
            np.nan_to_num(
                pd.read_hdf(os.path.join(INPUT_DIR, "water_licence_price.h5")).to_numpy()
            )
        )

        # Spatially explicit costs of water delivery per ML.
        self.WATER_DELIVERY_PRICE = self.get_array_resfactor_applied(
            np.nan_to_num(
                pd.read_hdf(os.path.join(INPUT_DIR, "water_delivery_price.h5")).to_numpy()
            )
        )

        # River regions.
        self.RIVREG_ID = self.get_array_resfactor_applied(
            pd.read_hdf(os.path.join(INPUT_DIR, "rivreg_id.h5")).to_numpy()  # River region ID mapped.
        )
        rr = pd.read_hdf(os.path.join(INPUT_DIR, "rivreg_lut.h5"))
        self.RIVREG_DICT = dict(
            zip(rr.HR_RIVREG_ID, rr.HR_RIVREG_NAME)
        )  # River region ID to Name lookup table
        self.RIVREG_LIMITS = dict(
            zip(rr.HR_RIVREG_ID, rr.WATER_YIELD_HIST_BASELINE_ML)
        )  # River region ID and water use limits

        # Drainage divisions
        self.DRAINDIV_ID = self.get_array_resfactor_applied(
            pd.read_hdf(os.path.join(INPUT_DIR, "draindiv_id.h5")).to_numpy()  # Drainage div ID mapped.
        )
        dd = pd.read_hdf(os.path.join(INPUT_DIR, "draindiv_lut.h5"))
        self.DRAINDIV_DICT = dict(
            zip(dd.HR_DRAINDIV_ID, dd.HR_DRAINDIV_NAME)
        )  # Drainage div ID to Name lookup table
        self.DRAINDIV_LIMITS = dict(
            zip(dd.HR_DRAINDIV_ID, dd.WATER_YIELD_HIST_BASELINE_ML)
        )  # Drainage div ID and water use limits


        # Water yields -- run off from a cell into catchment by deep-rooted, shallow-rooted, and natural land
        water_yield_baselines = pd.read_hdf(os.path.join(INPUT_DIR, "water_yield_baselines.h5"))
        self.WATER_YIELD_HIST_DR = self.get_array_resfactor_applied(
            water_yield_baselines['WATER_YIELD_HIST_DR_ML_HA'].to_numpy(dtype = np.float32)
        )
        self.WATER_YIELD_HIST_SR = self.get_array_resfactor_applied(
            water_yield_baselines["WATER_YIELD_HIST_SR_ML_HA"].to_numpy(dtype = np.float32)
        )
        self.DEEP_ROOTED_PROPORTION = self.get_array_resfactor_applied(
            water_yield_baselines['DEEP_ROOTED_PROPORTION'].to_numpy(dtype = np.float32)
        )
        self.WATER_YIELD_HIST_NL = self.get_array_resfactor_applied(
            water_yield_baselines.eval('WATER_YIELD_HIST_DR_ML_HA * DEEP_ROOTED_PROPORTION + \
                                        WATER_YIELD_HIST_SR_ML_HA * (1 - DEEP_ROOTED_PROPORTION)'
                                      ).to_numpy(dtype = np.float32)
        )
        wyield_fname_dr = os.path.join(INPUT_DIR, 'water_yield_ssp' + str(settings.SSP) + '_2010-2100_dr_ml_ha.h5')
        wyield_fname_sr = os.path.join(INPUT_DIR, 'water_yield_ssp' + str(settings.SSP) + '_2010-2100_sr_ml_ha.h5')
        
        # Read the data into memory with [...], so that it can be pickled.
        self.WATER_YIELD_DR_FILE = h5py.File(wyield_fname_dr, 'r')[f'Water_yield_GCM-Ensemble_ssp{settings.SSP}_2010-2100_DR_ML_HA_mean'][...][:, self.MASK]
        self.WATER_YIELD_SR_FILE = h5py.File(wyield_fname_sr, 'r')[f'Water_yield_GCM-Ensemble_ssp{settings.SSP}_2010-2100_SR_ML_HA_mean'][...][:, self.MASK] 
        

        # Water yield from outside LUTO study area.
        water_yield_oustide_luto_hist = pd.read_hdf(os.path.join(INPUT_DIR, 'water_yield_outside_LUTO_study_area_hist_1970_2000.h5'))
        
        if settings.WATER_REGION_DEF == 'River Region':
            rr_outside_luto = pd.read_hdf(os.path.join(INPUT_DIR, 'water_yield_outside_LUTO_study_area_2010_2100_rr_ml.h5'))
            rr_outside_luto = rr_outside_luto.loc[:, pd.IndexSlice[:, settings.SSP]]
            rr_outside_luto.columns = rr_outside_luto.columns.droplevel('ssp')

            rr_natural_land = pd.read_hdf(os.path.join(INPUT_DIR, 'water_yield_natural_land_2010_2100_rr_ml.h5'))
            rr_natural_land = rr_natural_land.loc[:, pd.IndexSlice[:, settings.SSP]]
            rr_natural_land.columns = rr_natural_land.columns.droplevel('ssp')

            self.WATER_OUTSIDE_LUTO_RR = rr_outside_luto
            self.WATER_OUTSIDE_LUTO_RR_HIST = water_yield_oustide_luto_hist.query('Region_Type == "River Region"').set_index('Region_ID')['Water Yield (ML)'].to_dict()
            self.WATER_UNDER_NATURAL_LAND_RR = rr_natural_land

        if settings.WATER_REGION_DEF == 'Drainage Division':
            dd_outside_luto = pd.read_hdf(os.path.join(INPUT_DIR, 'water_yield_outside_LUTO_study_area_2010_2100_dd_ml.h5'))
            dd_outside_luto = dd_outside_luto.loc[:, pd.IndexSlice[:, settings.SSP]]
            dd_outside_luto.columns = dd_outside_luto.columns.droplevel('ssp')

            dd_natural_land = pd.read_hdf(os.path.join(INPUT_DIR, 'water_yield_natural_land_2010_2100_dd_ml.h5'))
            dd_natural_land = dd_natural_land.loc[:, pd.IndexSlice[:, settings.SSP]]
            dd_natural_land.columns = dd_natural_land.columns.droplevel('ssp')

            self.WATER_OUTSIDE_LUTO_DD = dd_outside_luto
            self.WATER_OUTSIDE_LUTO_DD_HIST = water_yield_oustide_luto_hist.query('Region_Type == "Drainage Division"').set_index('Region_ID')['Water Yield (ML)'].to_dict()
            self.WATER_UNDER_NATURAL_LAND_DD = dd_natural_land
            
        # Place holder for Water Yield under River Region to avoid recalculating it every time.
        self.WATER_YIELD_RR_BASE_YR = None
        
        
        ###############################################################
        # Carbon sequestration by trees data.
        ###############################################################
        print("\tLoading carbon sequestration by trees data...", flush=True)

        # Load the remnant vegetation carbon data.
        rem_veg = pd.read_hdf(os.path.join(INPUT_DIR, "natural_land_t_co2_ha.h5")).to_numpy(
            dtype=np.float32
        )
        rem_veg = np.squeeze(rem_veg)  # Remove extraneous extra dimension

        # Discount by fire risk.
        self.NATURAL_LAND_T_CO2_HA = self.get_array_resfactor_applied(
            rem_veg * (fire_risk.to_numpy() / 100)
        )



        ###############################################################
        # Demand data.
        ###############################################################
        print("\tLoading demand data...", flush=True)

        # Load demand data (actual production (tonnes, ML) by commodity) - from demand model
        dd = pd.read_hdf(os.path.join(INPUT_DIR, 'demand_projections.h5') )

        # Select the demand data under the running scenario
        self.DEMAND_DATA = dd.loc[(settings.SCENARIO,
                                   settings.DIET_DOM,
                                   settings.DIET_GLOB,
                                   settings.CONVERGENCE,
                                   settings.IMPORT_TREND,
                                   settings.WASTE,
                                   settings.FEED_EFFICIENCY)].copy()

        # Convert eggs from count to tonnes
        self.DEMAND_DATA.loc['eggs'] = self.DEMAND_DATA.loc['eggs'] * settings.EGGS_AVG_WEIGHT / 1000 / 1000

        # Get the off-land commodities
        self.DEMAND_OFFLAND = self.DEMAND_DATA.loc[self.DEMAND_DATA.query("COMMODITY in @settings.OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()

        # Remove off-land commodities
        self.DEMAND_C = self.DEMAND_DATA.loc[self.DEMAND_DATA.query("COMMODITY not in @settings.OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()

        # Convert to numpy array of shape (91, 26)
        self.DEMAND_C = self.DEMAND_C.to_numpy(dtype = np.float32).T
        self.D_CY = self.DEMAND_C # new demand is in tonnes rather than deltas


        ###############################################################
        # Carbon emissions from off-land commodities.
        ###############################################################
        print("\tLoading off-land commodities' carbon emissions data...", flush=True)

        # Read the greenhouse gas intensity data
        off_land_ghg_intensity = pd.read_csv(f'{INPUT_DIR}/agGHG_lvstk_off_land.csv')
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
        carbon_price_sheet = settings.CARBON_PRICES_FIELD or "Default"
        carbon_price_usecols = "A,B"
        carbon_price_col_names = ["Year", "Carbon_price_$_tCO2e"]
        carbon_price_sheet_index_col = "Year" # if carbon_price_sheet != "Default" else 0
        carbon_price_sheet_header = 0         # if carbon_price_sheet != "Default" else None

        self.CARBON_PRICES: dict[int, float] = pd.read_excel(
            os.path.join(INPUT_DIR, 'carbon_prices.xlsx'),
            sheet_name=carbon_price_sheet,
            usecols=carbon_price_usecols,
            names=carbon_price_col_names,
            header=carbon_price_sheet_header,
            index_col=carbon_price_sheet_index_col,
        )["Carbon_price_$_tCO2e"].to_dict()


        ###############################################################
        # GHG targets data.
        ###############################################################
        print("\tLoading GHG targets data...", flush=True)

        # If GHG_LIMITS_TYPE == 'file' then import the Excel spreadsheet and import the results to a python dictionary {year: target (tCO2e), ...}
        if settings.GHG_LIMITS_TYPE == "file":
            self.GHG_TARGETS = pd.read_excel(
                os.path.join(INPUT_DIR, "GHG_targets.xlsx"), sheet_name="Data", index_col="YEAR"
            )
            self.GHG_TARGETS = self.GHG_TARGETS[settings.GHG_LIMITS_FIELD].to_dict()

        # If settings.GHG_LIMITS_TYPE == 'dict' then import the Excel spreadsheet and import the results to a python dictionary {year: target (tCO2e), ...}
        elif settings.GHG_LIMITS_TYPE == "dict":

            # Create a dictionary to hold the GHG target data
            self.GHG_TARGETS = {}  # pd.DataFrame(columns = ['TOTAL_GHG_TCO2E'])

            # # Create linear function f and interpolate
            f = interp1d(
                list(settings.GHG_LIMITS.keys()),
                list(settings.GHG_LIMITS.values()),
                kind="linear",
                fill_value="extrapolate",
            )
            # keys = range(2010, 2101)
            for yr in range(2010, 2101):
                self.GHG_TARGETS[yr] = f(yr)


        ###############################################################
        # Savanna burning data.
        ###############################################################
        print("\tLoading savanna burning data...", flush=True)

        # Read in the dataframe
        savburn_df = pd.read_hdf(os.path.join(INPUT_DIR, 'cell_savanna_burning.h5') )

        # Load the columns as numpy arrays
        self.SAVBURN_ELIGIBLE = savburn_df.ELIGIBLE_AREA.to_numpy()               # 1 = areas eligible for early dry season savanna burning under the ERF, 0 = ineligible
        self.SAVBURN_AVEM_CH4_TCO2E_HA = savburn_df.SAV_AVEM_CH4_TCO2E_HA.to_numpy()  # Avoided emissions - methane
        self.SAVBURN_AVEM_N2O_TCO2E_HA = savburn_df.SAV_AVEM_N2O_TCO2E_HA.to_numpy()  # Avoided emissions - nitrous oxide
        self.SAVBURN_SEQ_CO2_TCO2E_HA = savburn_df.SAV_SEQ_CO2_TCO2E_HA.to_numpy()    # Additional carbon sequestration - carbon dioxide
        self.SAVBURN_TOTAL_TCO2E_HA = self.get_array_resfactor_applied(
            savburn_df.AEA_TOTAL_TCO2E_HA.to_numpy()
        )

        # Cost per hectare in dollars from settings
        self.SAVBURN_COST_HA = settings.SAVBURN_COST_HA_YR


        ###############################################################
        # Biodiversity data.
        ###############################################################
        print("\tLoading biodiversity data...", flush=True)
        """
        Kunming-Montreal Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
        Ensure that by 2030 at least 30 per cent of areas of degraded terrestrial, inland water, and coastal and marine ecosystems are under effective restoration,
        in order to enhance biodiversity and ecosystem functions and services, ecological integrity and connectivity.
        """

        # Create a dictionary to hold the annual biodiversity target proportion data for GBF Target 2
        f = interp1d(
            list(settings.BIODIV_GBF_TARGET_2_DICT.keys()),
            list(settings.BIODIV_GBF_TARGET_2_DICT.values()),
            kind = "linear",
            fill_value = "extrapolate",
        )
        biodiv_GBF_target_2_proportions_2010_2100 = {yr: f(yr).item() for yr in range(2010, 2101)}



        # Get the connectivity score between 0 and 1, where 1 is the highest connectivity
        biodiv_priorities = pd.read_hdf(os.path.join(INPUT_DIR, 'biodiv_priorities.h5'))

        if settings.CONNECTIVITY_SOURCE == 'NCI':
            connectivity_score = biodiv_priorities['DCCEEW_NCI'].to_numpy(dtype = np.float32)
            connectivity_score = np.where(self.LUMASK, connectivity_score, 1)               # Set the connectivity score to 1 for cells outside the LUMASK
            connectivity_score = np.interp(connectivity_score, (connectivity_score.min(), connectivity_score.max()), (settings.CONNECTIVITY_LB, 1)).astype('float32')
        elif settings.CONNECTIVITY_SOURCE == 'DWI':
            connectivity_score = biodiv_priorities['NATURAL_AREA_CONNECTIVITY'].to_numpy(dtype = np.float32)
            connectivity_score = np.interp(connectivity_score, (connectivity_score.min(), connectivity_score.max()), (1, settings.CONNECTIVITY_LB)).astype('float32')
        elif settings.CONNECTIVITY_SOURCE == 'NONE':
            connectivity_score = 1
        else:
            raise ValueError(f"Invalid connectivity source: {settings.CONNECTIVITY_SOURCE}, must be 'NCI', 'DWI' or 'NONE'")



        # Get the Zonation output score between 0 and 1. biodiv_score_raw.sum() = 153 million
        biodiv_score_raw = biodiv_priorities['BIODIV_PRIORITY_SSP' + str(settings.SSP)].to_numpy(dtype = np.float32)
        # Weight the biodiversity score by the connectivity score
        self.BIODIV_SCORE_RAW_WEIGHTED = biodiv_score_raw * connectivity_score



        # Habitat degradation scale for agricultural land-use
        biodiv_degrade_df = pd.read_csv(os.path.join(INPUT_DIR, 'HABITAT_CONDITION.csv'))                                                               # Load the HCAS percentile data (pd.DataFrame)

        if settings.HABITAT_CONDITION == 'HCAS':
            '''
            The degradation weight score of "HCAS" are float values range between 0-1 indicating the suitability for wild animals survival.
            Here we average this dataset in year 2009, 2010, and 2011, then calculate the percentiles of the average score under each land-use type.
            '''
            self.BIODIV_HABITAT_DEGRADE_LOOK_UP = biodiv_degrade_df[['lu', f'PERCENTILE_{settings.HCAS_PERCENTILE}']]                                   # Get the biodiversity degradation score at specified percentile (pd.DataFrame)
            self.BIODIV_HABITAT_DEGRADE_LOOK_UP = {int(k):v for k,v in dict(self.BIODIV_HABITAT_DEGRADE_LOOK_UP.values).items()}                        # Convert the biodiversity degradation score to a dictionary {land-use-code: score}
            unalloc_nat_land_bio_score = self.BIODIV_HABITAT_DEGRADE_LOOK_UP[self.DESC2AGLU['Unallocated - natural land']]                              # Get the biodiversity degradation score for unallocated natural land (float)
            self.BIODIV_HABITAT_DEGRADE_LOOK_UP = {k:v*(1/unalloc_nat_land_bio_score) for k,v in self.BIODIV_HABITAT_DEGRADE_LOOK_UP.items()}           # Normalise the biodiversity degradation score to the unallocated natural land score

        elif settings.HABITAT_CONDITION == 'USER_DEFINED':
            self.BIODIV_HABITAT_DEGRADE_LOOK_UP = biodiv_degrade_df[['lu', 'USER_DEFINED']]
            self.BIODIV_HABITAT_DEGRADE_LOOK_UP = {int(k):v for k,v in dict(self.BIODIV_HABITAT_DEGRADE_LOOK_UP.values).items()}                        # Convert the biodiversity degradation score to a dictionary {land-use-code: score}

        else:
            raise ValueError(f"Invalid habitat condition source: {settings.HABITAT_CONDITION}, must be 'HCAS' or 'USER_DEFINED'")



        # Get the biodiversity degradation score (0-1) for each cell
        '''
        The degradation scores are float values range between 0-1 indicating the discount of biodiversity value for each cell.
        E.g., 0.8 means the biodiversity value of the cell is 80% of the original raw biodiversity value.
        '''
        biodiv_degrade_LDS = np.where(self.SAVBURN_ELIGIBLE, settings.LDS_BIODIVERSITY_VALUE, 1)                                            # Get the biodiversity degradation score for LDS burning (1D numpy array)
        biodiv_degrade_habitat = np.vectorize(self.BIODIV_HABITAT_DEGRADE_LOOK_UP.get)(self.LUMAP_NO_RESFACTOR).astype(np.float32)          # Get the biodiversity degradation score for each cell (1D numpy array)

        # Get the biodiversity damage under LDS burning (0-1) for each cell
        biodiv_degradation_raw_weighted_LDS = self.BIODIV_SCORE_RAW_WEIGHTED * (1 - biodiv_degrade_LDS)                     # Biodiversity damage under LDS burning (1D numpy array)
        biodiv_degradation_raw_weighted_habitat = self.BIODIV_SCORE_RAW_WEIGHTED * (1 - biodiv_degrade_habitat)             # Biodiversity damage under under HCAS (1D numpy array)

        # Get the biodiversity value at the beginning of the simulation
        self.BIODIV_RAW_WEIGHTED_LDS = self.BIODIV_SCORE_RAW_WEIGHTED - biodiv_degradation_raw_weighted_LDS                 # Biodiversity value under LDS burning (1D numpy array); will be used as base score for calculating ag/non-ag stratagies impacts on biodiversity
        biodiv_current_val = self.BIODIV_RAW_WEIGHTED_LDS - biodiv_degradation_raw_weighted_habitat                         # Biodiversity value at the beginning year (1D numpy array)
        biodiv_current_val = np.nansum(biodiv_current_val[self.LUMASK] * self.REAL_AREA_NO_RESFACTOR[self.LUMASK])          # Sum the biodiversity value within the LUMASK

        # Biodiversity values need to be restored under the GBF Target 2
        '''
        The biodiversity value to be restored is calculated as the difference between the 'Unallocated - natural land'
        and 'current land-use' regarding their biodiversity degradation scale.
        '''
        biodiv_degradation_val = (
            biodiv_degradation_raw_weighted_LDS +                                                                           # Biodiversity degradation from HCAS
            biodiv_degradation_raw_weighted_habitat                                                                         # Biodiversity degradation from LDS burning
        )
        biodiv_degradation_val = np.nansum(biodiv_degradation_val[self.LUMASK] * self.REAL_AREA_NO_RESFACTOR[self.LUMASK])  # Sum the biodiversity degradation value within the LUMASK

        # Multiply by biodiversity target to get the additional biodiversity score required to achieve the target
        self.BIODIV_GBF_TARGET_2 = {
            yr: biodiv_current_val + biodiv_degradation_val * biodiv_GBF_target_2_proportions_2010_2100[yr]
            for yr in range(2010, 2101)
        }



        ###############################################################
        # BECCS data.
        ###############################################################
        print("\tLoading BECCS data...", flush=True)

        # Load dataframe
        beccs_df = pd.read_hdf(os.path.join(INPUT_DIR, 'cell_BECCS_df.h5') )

        # Capture as numpy arrays
        self.BECCS_COSTS_AUD_HA_YR = self.get_array_resfactor_applied(beccs_df['BECCS_COSTS_AUD_HA_YR'].to_numpy())
        self.BECCS_REV_AUD_HA_YR = self.get_array_resfactor_applied(beccs_df['BECCS_REV_AUD_HA_YR'].to_numpy())
        self.BECCS_TCO2E_HA_YR = self.get_array_resfactor_applied(beccs_df['BECCS_TCO2E_HA_YR'].to_numpy())
        self.BECCS_MWH_HA_YR = self.get_array_resfactor_applied(beccs_df['BECCS_MWH_HA_YR'].to_numpy())


        ###############################################################
        # Cost multiplier data.
        ###############################################################
        cost_mult_excel = pd.ExcelFile(os.path.join(INPUT_DIR, 'cost_multipliers.xlsx'))
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
        # Apply resfactor to various arrays required for data loading.
        ###############################################################
        self.SAVBURN_ELIGIBLE = self.get_array_resfactor_applied(self.SAVBURN_ELIGIBLE)
        self.BIODIV_SCORE_RAW_WEIGHTED = self.get_array_resfactor_applied(self.BIODIV_SCORE_RAW_WEIGHTED)
        self.BIODIV_RAW_WEIGHTED_LDS = self.get_array_resfactor_applied(self.BIODIV_RAW_WEIGHTED_LDS)


        print("Data loading complete\n")

    def get_coord(self, index_ij: np.ndarray, trans):
        """
        Calculate the coordinates [[lon,...],[lat,...]] based on
        the given index [[row,...],[col,...]] and transformation matrix.

        Parameters:
        index_ij (np.ndarray): A numpy array containing the row and column indices.
        trans (affin): An instance of the Transformation class.
        resfactor (int, optional): The resolution factor. Defaults to 1.

        Returns:
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

        Returns:
            dict: The updated geographic metadata.
        """
        meta = self.GEO_META_FULLRES.copy()
        height, width = (self.GEO_META_FULLRES['height'], self.GEO_META_FULLRES['width'])  if settings.WRITE_FULL_RES_MAPS else self.LUMAP_2D_RESFACTORED.shape
        trans = list(self.GEO_META_FULLRES['transform'])
        trans[0] = trans[0] if settings.WRITE_FULL_RES_MAPS else trans[0] * settings.RESFACTOR    # Adjust the X resolution
        trans[4] = trans[4] if settings.WRITE_FULL_RES_MAPS else trans[4] * settings.RESFACTOR    # Adjust the Y resolution
        trans = Affine(*trans)
        meta.update(width=width, height=height, compress='lzw', driver='GTiff', transform=trans, nodata=self.NODATA, dtype='float32')
        return meta

    def get_array_resfactor_applied(self, array: np.ndarray):
        """
        Returns a version of the given array with the ResFactor applied.
        """
        return array[self.MASK]

    def get_df_resfactor_applied(self, df: pd.DataFrame):
        """
        Returns a version of the given DataFrame with the ResFactor applied.
        """
        return df.iloc[self.MASK]

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
            return lumap2ag_l_mrj(self.LUMAP_NO_RESFACTOR, self.LMMAP_NO_RESFACTOR)[:, self.MASK, :]

        # Create a 2D array of IDs for the LUMAP_2D_RESFACTORED
        lumap_2d_id = np.arange(self.LUMAP_2D_RESFACTORED.size).reshape(self.LUMAP_2D_RESFACTORED.shape)
        lumap_2d_id = upsample_array(self, lumap_2d_id, settings.RESFACTOR)    
        lumask_2d_no_resfactor = (self.LUMAP_2D != self.NODATA) & (self.LUMAP_2D != self.MASK_LU_CODE)    

        # Get the 2D water supply map at full resolution 
        lmmap_full_2d = np.full_like(self.NLUM_MASK, self.NODATA, dtype=np.int16)                           # 2D map,  full of nodata (-9999)
        np.place(lmmap_full_2d, self.NLUM_MASK == 1, self.LMMAP_NO_RESFACTOR)                               # 2D map,  -9999 for ocean; -1 for desert, urban, water, etc; 0-27 for land uses

        # Calculate the number of cells with each resfactored ID cell
        cell_count = np.bincount(lumap_2d_id.flatten(), lumask_2d_no_resfactor.flatten(), minlength=self.LUMAP_2D_RESFACTORED.size)
        lumap_resample_avg = np.zeros((len(self.LANDMANS), self.NCELLS, self.N_AG_LUS), dtype=np.float32)
                
        for idx_lu in self.DESC2AGLU.values():
            for idx_w, _ in enumerate(self.LANDMANS):
                # Get the cells with the same ID and water supply
                lumap_w = (self.LUMAP_2D == idx_lu) * (lmmap_full_2d == idx_w)
                cell_sum = np.bincount(lumap_2d_id.flatten(), lumap_w.flatten(), minlength=self.LUMAP_2D_RESFACTORED.size)

                # Calculate the average value of each ID cell
                with np.errstate(divide='ignore', invalid='ignore'):                    # Ignore the division by zero warning
                    cell_avg = cell_sum / cell_count
                    cell_avg[~np.isfinite(cell_avg)] = 0                                # Set the NaN and Inf to 0
                    
                # Reshape the 1D avg array to 2D array
                cell_avg_2d = cell_avg.reshape(self.LUMAP_2D_RESFACTORED.shape)
                # Upsample the 2D array from choarser resolution to finer resolution
                cell_avg_2d = upsample_array(self, cell_avg_2d, settings.RESFACTOR).astype(np.float32)
                # Only keep the cells within the luto study area
                cell_avg_1d = cell_avg_2d[np.nonzero(self.NLUM_MASK)]
                cell_avg_1d = cell_avg_1d[self.MASK]

                lumap_resample_avg[idx_w, :, idx_lu] = cell_avg_1d
                
        return lumap_resample_avg


    # Functions to add reprojected dvars to the output containers
    def add_ag_dvars_xr(self, yr: int, ag_dvar: np.ndarray):
        self.ag_dvars_2D_reproj_match[yr] = self.reproj_match_ag_dvar(ag_dvar, self.REPROJECT_TARGET_ID_MAP, self.REPROJECT_REFERENCE_MAP)

    def add_am_dvars_xr(self, yr: int, am_dvar: np.ndarray):
        self.ag_man_dvars_2D_reproj_match[yr] = self.reproj_match_am_dvar(am_dvar, self.REPROJECT_TARGET_ID_MAP, self.REPROJECT_REFERENCE_MAP)

    def add_non_ag_dvars_xr(self, yr: int, non_ag_dvar: np.ndarray):
        self.non_ag_dvars_2D_reproj_match[yr] = self.reproj_match_non_ag_dvar(non_ag_dvar, self.REPROJECT_TARGET_ID_MAP, self.REPROJECT_REFERENCE_MAP)


    # Functions to reproject and match the dvars to the target map
    def reproj_match_ag_dvar(self, ag_dvars:np.ndarray, target_id_map:xr.DataArray, target_ref_map:xr.DataArray):

        ag_dvars = self.ag_dvars_to_xr(ag_dvars)                # Convert the dvars to xarray

        # Parallelize the reprojection and matching
        def reproj_match(dvar, lm, lu):
            dvar = self.dvar_to_2D(dvar)                        # Convert the dvar to its 2D representation
            dvar = self.dvar_to_full_res(dvar)                  # Convert the 2D dvar to full resolution
            dvar = self.bincount_avg(target_id_map, dvar)       # Calculate the average of the dvar values in each target id cell.
            dvar = dvar.reshape(target_ref_map.shape)           # Reshape the dvar to match the target reference map
            dvar = xr.DataArray(dvar, dims=('y', 'x'), coords={'y': target_ref_map['y'], 'x': target_ref_map['x']})
            return dvar.expand_dims({'lm':[lm], 'lu':[lu]})

        tasks = [delayed(reproj_match)(ag_dvars.sel(lm=lm, lu=lu), lm, lu)
                 for lm,lu in product(self.LANDMANS, self.AGRICULTURAL_LANDUSES)]

        return  xr.combine_by_coords( [i for i in Parallel(n_jobs=10, backend='threading', return_as='generator')(tasks)])


    def reproj_match_am_dvar(self, am_dvars, target_id_map:xr.DataArray, target_ref_map:xr.DataArray):

        am_dvars = self.am_dvars_to_xr(am_dvars)

        # Parallelize the reprojection and matching
        def reproj_match(dvar, am, lm, lu):
            dvar = self.dvar_to_2D(dvar)
            dvar = self.dvar_to_full_res(dvar)
            dvar = self.bincount_avg(target_id_map, dvar)
            dvar = dvar.reshape(target_ref_map.shape)
            dvar = xr.DataArray(dvar, dims=('y', 'x'), coords={'y': target_ref_map['y'], 'x': target_ref_map['x']})
            return dvar.expand_dims({'am':[am], 'lm':[lm], 'lu':[lu]})

        # Parallelize the reprojection and matching
        tasks = [delayed(reproj_match)(am_dvars.sel(am=am, lm=lm, lu=lu), am, lm, lu)
                 for am,lm,lu in product(AG_MANAGEMENTS_TO_LAND_USES, self.LANDMANS, self.AGRICULTURAL_LANDUSES)]

        return xr.combine_by_coords([i for i in Parallel(n_jobs=10, backend='threading', return_as='generator')(tasks)])



    def reproj_match_non_ag_dvar(self, non_ag_dvars, target_id_map:xr.DataArray, target_ref_map:xr.DataArray):

        non_ag_dvars = self.non_ag_dvars_to_xr(non_ag_dvars)

        # Parallelize the reprojection and matching
        def reproj_match(dvar, lu):
            dvar = self.dvar_to_2D(dvar)
            dvar = self.dvar_to_full_res(dvar) if settings.RESFACTOR > 1 else dvar
            dvar = self.bincount_avg(target_id_map, dvar)
            dvar = dvar.reshape(target_ref_map.shape)
            dvar = xr.DataArray(dvar, dims=('y', 'x'), coords={'y': target_ref_map['y'], 'x': target_ref_map['x']})
            return dvar.expand_dims({'lu':[lu]})

        tasks = [delayed(reproj_match)(non_ag_dvars.sel(lu=lu),  lu) for lu in self.NON_AGRICULTURAL_LANDUSES]

        return xr.combine_by_coords([i for i in Parallel(n_jobs=10, backend='threading', return_as='generator')(tasks)])


    # Functions to convert dvars to xarray
    def ag_dvars_to_xr(self, ag_dvars: np.ndarray):
        ag_dvar_xr = xr.DataArray(
            ag_dvars,
            dims=('lm', 'cell', 'lu'),
            coords={
                'lm': self.LANDMANS,
                'cell': np.arange(ag_dvars.shape[1]),
                'lu': self.AGRICULTURAL_LANDUSES
            }
        ).reindex(lu=self.AGRICULTURAL_LANDUSES) # Reorder the dimensions to match the LUTO variable array indexing

        return ag_dvar_xr


    def am_dvars_to_xr(self, am_dvars: np.ndarray):
        am_dvar_l = []
        for am in am_dvars.keys():
            am_dvar_xr = xr.DataArray(
                am_dvars[am],
                dims=('lm', 'cell', 'lu'),
                coords={
                    'lm': self.LANDMANS,
                    'cell': np.arange(am_dvars[am].shape[1]),
                    'lu': self.AGRICULTURAL_LANDUSES})

            # Expand the am dimension, the dvar is a 4D array [am, lu, cell, lu]
            am_dvar_xr = am_dvar_xr.expand_dims({'am':[am]})
            am_dvar_l.append(am_dvar_xr)

        return xr.combine_by_coords(am_dvar_l).reindex(
            am=AG_MANAGEMENTS_TO_LAND_USES.keys(),
            lu=self.AGRICULTURAL_LANDUSES,
            lm=self.LANDMANS)   # Reorder the dimensions to match the LUTO variable array indexing


    def non_ag_dvars_to_xr(self, non_ag_dvars: np.ndarray):
        non_ag_dvar_xr = xr.DataArray(
            non_ag_dvars,
            dims=('cell', 'lu'),
            coords={
                'cell': np.arange(non_ag_dvars.shape[0]),
                'lu': self.NON_AGRICULTURAL_LANDUSES})

        return non_ag_dvar_xr.reindex(
            lu=self.NON_AGRICULTURAL_LANDUSES) # Reorder the dimensions to match the LUTO variable array indexing


    # Convert dvar to its 2D representation
    def dvar_to_2D(self, map_:np.ndarray)-> np.ndarray:
        '''
        Convert the dvar from 1D vector to 2D array.
        '''
        map_1D = self.LUMAP_2D_RESFACTORED.copy().astype(np.float32) if settings.RESFACTOR > 1 else self.LUMAP_2D.copy().astype(np.float32)
        np.place(map_1D, (map_1D != self.MASK_LU_CODE) & (map_1D != self.NODATA), map_)
        return map_1D


    # Upsample dvar to its full resolution representation
    def dvar_to_full_res(self, dvar_2D:np.ndarray) -> np.ndarray:
        '''
        Upsample the dvar to its full resolution (RESFACTOR=1) representation.
        '''
        dense_2D_shape = self.NLUM_MASK.shape
        dense_2D_map = np.repeat(np.repeat(dvar_2D, settings.RESFACTOR, axis=0), settings.RESFACTOR, axis=1)

        # Adjust the dense_2D_map size if it differs from the NLUM_MASK
        if dense_2D_map.shape[0] > dense_2D_shape[0] or dense_2D_map.shape[1] > dense_2D_shape[1]:
            dense_2D_map = dense_2D_map[:dense_2D_shape[0], :dense_2D_shape[1]]

        if dense_2D_map.shape[0] < dense_2D_shape[0] or dense_2D_map.shape[1] < dense_2D_shape[1]:
            pad_height = dense_2D_shape[0] - dense_2D_map.shape[0]
            pad_width = dense_2D_shape[1] - dense_2D_map.shape[1]
            dense_2D_map = np.pad(
                dense_2D_map,
                pad_width=((0, pad_height), (0, pad_width)),
                mode='edge')

        # Apply the masks
        filler_mask = self.LUMAP_2D != self.MASK_LU_CODE
        dense_2D_map = np.where(filler_mask, dense_2D_map, self.MASK_LU_CODE)
        dense_2D_map = np.where(self.NLUM_MASK, dense_2D_map, self.NODATA)
        return dense_2D_map


    # Calculate the average value of dvars within the target bin
    def bincount_avg(self, target_id_map:xr.DataArray, dvar:np.ndarray) -> np.ndarray:
        """
        Calculate the average value of each bin based on the target_id_map and dvar arrays.
        Here is where we reproject and match dvars to the target map. Essentially, we
        created an ID map for the target map, where each pixcel has the  index of the
        flattend target map but the same shape of the `NLUM_MASK` (see `N:/Data-Master/Biodiversity/biodiversity_contribution/Step_3_Match_lumap_to_biomap.py`).

        We use `bincount` to get the occurence of dvar cells with the same target ID,
        and get the sum of dvar values with the same target ID. At last, the average
        of dvar cells within each target cell is calculated as `occurance / sum`.

        Parameters:
        - target_id_map (xr.DataArray): Array containing the bin indices.
        - dvar (np.ndarray): Array containing the values corresponding to each bin.

        Returns:
        - np.ndarray: Array (1D) containing the average value of each bin.

        Note:
        - The average value is calculated by dividing the sum of values in each bin by the number of occurrences in that bin.
        - Division by zero will result in a nan value, which is handled by replacing it with zero.

        """
        # Only dvar > 0 are necessary for the calculation
        valid_mask = dvar > 0

        # Flatten arrays
        bin_flatten = target_id_map.values[valid_mask]
        weights_flatten = dvar[valid_mask]
        bin_occ = np.bincount(bin_flatten, minlength=target_id_map.max().values + 1)
        bin_sum = np.bincount(bin_flatten, weights=weights_flatten, minlength=target_id_map.max().values + 1)

        # Calculate the average value of each bin, ignoring division by zero (which will be nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_avg = (bin_sum / bin_occ).astype(np.float32)
            bin_avg = np.nan_to_num(bin_avg)

        return bin_avg



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

    def set_path(self, base_year, target_year) -> str:
        """Create a folder for storing outputs and return folder name."""

        # Get the years to write
        if settings.MODE == "snapshot":
            yr_all = [base_year, target_year]
        elif settings.MODE == "timeseries":
            yr_all = list(range(base_year, target_year + 1))

        # Create path name
        self.path = f"{OUTPUT_DIR}/{self.timestamp_sim}_RF{settings.RESFACTOR}_{yr_all[0]}-{yr_all[-1]}_{settings.MODE}"

        # Get all paths
        paths = (
            [self.path]
            + [f"{self.path}/out_{yr}" for yr in yr_all]
            + [f"{self.path}/out_{yr}/lucc_separate" for yr in yr_all[1:]]
        )  # Skip creating lucc_separate for base year

        # Add the path for the comparison between base-year and target-year if in the timeseries mode
        if settings.MODE == "timeseries":
            path_begin_end_compare = f"{self.path}/begin_end_compare_{yr_all[0]}_{yr_all[-1]}"
            paths = (
                paths
                + [path_begin_end_compare]
                + [
                    f"{path_begin_end_compare}/out_{yr_all[0]}",
                    f"{path_begin_end_compare}/out_{yr_all[-1]}",
                    f"{path_begin_end_compare}/out_{yr_all[-1]}/lucc_separate",
                ]
            )

        # Create all paths
        for p in paths:
            if not os.path.exists(p):
                os.mkdir(p)

        return self.path

    def get_production(
        self,
        yr_cal: int,
        lumap: np.ndarray,
        lmmap: np.ndarray,
    ) -> np.ndarray:
        """
        Return total production of commodities for a specific year...

        'yr_cal' is calendar year

        Can return base year production (e.g., year = 2010) or can return production for
        a simulated year if one exists (i.e., year = 2030).

        Includes the impacts of land-use change, productivity increases, and
        climate change on yield.
        """
        if yr_cal == self.YR_CAL_BASE:
            ag_X_mrj = self.AG_L_MRJ
            non_ag_X_rk = self.NON_AG_L_RK
            ag_man_X_mrj = self.AG_MAN_L_MRJ_DICT
            
        else:
            ag_X_mrj = lumap2ag_l_mrj(lumap, lmmap)
            non_ag_X_rk = lumap2non_ag_l_mk(lumap, len(settings.NON_AG_LAND_USES.keys()))
            ag_man_X_mrj = get_base_am_vars(self.NCELLS, self.NLMS, self.N_AG_LUS)

        # Calculate year index (i.e., number of years since 2010)
        yr_idx = yr_cal - self.YR_CAL_BASE

        # Get the quantity of each commodity produced by agricultural land uses
        ag_q_mrp = ag_quantity.get_quantity_matrices(self, yr_idx)

        # Convert map of land-use in mrj format to mrp format using vectorization
        ag_X_mrp = np.einsum('mrj,pj->mrp', ag_X_mrj, self.LU2PR.astype(bool))

        # Sum quantities in product (PR/p) representation.
        ag_q_p = np.einsum('mrp,mrp->p', ag_q_mrp, ag_X_mrp)

        # Transform quantities to commodity (CM/c) representation.
        ag_q_c = np.einsum('cp,p->c', self.PR2CM.astype(bool), ag_q_p)

        # Get the quantity of each commodity produced by non-agricultural land uses
        q_crk = non_ag_quantity.get_quantity_matrix(self, ag_q_mrp, lumap)
        non_ag_q_c = np.einsum('crk,rk->c', q_crk, non_ag_X_rk)

        # Get quantities produced by agricultural management options
        ag_man_q_mrp = ag_quantity.get_agricultural_management_quantity_matrices(self, ag_q_mrp, yr_idx)
        ag_man_q_c = np.zeros(self.NCMS)

        j2p = {j: [p for p in range(self.NPRS) if self.LU2PR[p, j]]
                        for j in range(self.N_AG_LUS)}
        for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():
            am_j_list = [self.DESC2AGLU[lu] for lu in am_lus]
            current_ag_man_X_mrp = np.zeros(ag_q_mrp.shape, dtype=np.float32)
            for j in am_j_list:
                for p in j2p[j]:
                    current_ag_man_X_mrp[:, :, p] = ag_man_X_mrj[am][:, :, j]

            ag_man_q_p = np.einsum('mrp,mrp->p', ag_man_q_mrp[am], current_ag_man_X_mrp)
            ag_man_q_c += np.einsum('cp,p->c', self.PR2CM.astype(bool), ag_man_q_p)

        # Return total commodity production as numpy array.
        total_q_c = ag_q_c + non_ag_q_c + ag_man_q_c
        return total_q_c

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
        Get the net land water yield array, inclusive of all cells that LUTO does not look at.

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
