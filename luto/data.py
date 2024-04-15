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


import os, time
import rasterio
from datetime import datetime
import math
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Any

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from luto.settings import (
    INPUT_DIR,
    RESFACTOR,
    MODE,
    DEMAND_CONSTRAINT_TYPE,
    OBJECTIVE,
    PENALTY,
    CO2_FERT,
    SOC_AMORTISATION,
    NON_AGRICULTURAL_LU_BASE_CODE,
    RISK_OF_REVERSAL,
    FIRE_RISK,
    CONNECTIVITY_WEIGHTING,
    BIODIV_TARGET,
    SSP,
    RCP,
    SCENARIO,
    DIET_DOM,
    DIET_GLOB,
    CONVERGENCE,
    IMPORT_TREND,
    WASTE,
    FEED_EFFICIENCY,
    GHG_LIMITS_TYPE,
    GHG_LIMITS,
    GHG_LIMITS_FIELD,
    RIPARIAN_PLANTINGS_BUFFER_WIDTH,
    RIPARIAN_PLANTINGS_TORTUOSITY_FACTOR,
    BIODIV_LIVESTOCK_IMPACT,
    LDS_BIODIVERSITY_VALUE,
    OFF_LAND_COMMODITIES,
    EGGS_AVG_WEIGHT,
    NON_AGRICULTURAL_LU_BASE_CODE,
)


# Try-Except to make sure {rasterio} can be loaded under different environment
try:
    import rasterio
except:
    from osgeo import gdal
    import rasterio


def dict2matrix(d, fromlist, tolist):
    """Return 0-1 matrix mapping 'from-vectors' to 'to-vectors' using dict d."""
    A = np.zeros((len(tolist), len(fromlist)), dtype=np.int8)
    for j, jstr in enumerate(fromlist):
        for istr in d[jstr]:
            i = tolist.index(istr)
            A[i, j] = True
    return A


def lvs_veg_types(lu) -> tuple[str, str]:
    """Return livestock and vegetation types of the livestock land-use `lu`.

    Args:
        lu (str): The livestock land-use.

    Returns:
        tuple: A tuple containing the livestock type and vegetation type.

    Raises:
        KeyError: If the livestock type or vegetation type cannot be identified.

    """

    # Determine type of livestock.
    if 'beef' in lu.lower():
        lvstype = 'BEEF'
    elif 'sheep' in lu.lower():
        lvstype = 'SHEEP'
    elif 'dairy' in lu.lower():
        lvstype = 'DAIRY'
    else:
        raise KeyError(f"Livestock type '{lu}' not identified.")

    # Determine type of vegetation.
    if 'natural' in lu.lower():
        vegtype = 'natural land'
    elif 'modified' in lu.lower():
        vegtype = 'modified land'
    else:
        raise KeyError(f"Vegetation type '{lu}' not identified.")

    return lvstype, vegtype


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
    base_code = NON_AGRICULTURAL_LU_BASE_CODE
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

        print('\n' + time.strftime("%Y-%m-%d %H:%M:%S"), 'Beginning data initialisation...')

        ###############################################################
        # Agricultural economic data.
        ###############################################################
        print("\tLoading agricultural economic data...", end=" ", flush=True)

        # Load the agro-economic data (constructed using dataprep.py).
        self.AGEC_CROPS = pd.read_hdf(os.path.join(INPUT_DIR, "agec_crops.h5"))
        self.AGEC_LVSTK = pd.read_hdf(os.path.join(INPUT_DIR, "agec_lvstk.h5"))

        # Load greenhouse gas emissions from agriculture
        self.AGGHG_CROPS = pd.read_hdf(os.path.join(INPUT_DIR, "agGHG_crops.h5"))
        self.AGGHG_LVSTK = pd.read_hdf(os.path.join(INPUT_DIR, "agGHG_lvstk.h5"))
        self.AGGHG_IRRPAST = pd.read_hdf(os.path.join(INPUT_DIR, "agGHG_irrpast.h5"))

        # Raw transition cost matrix. In AUD/ha and ordered lexicographically.
        self.AG_TMATRIX = np.load(os.path.join(INPUT_DIR, "ag_tmatrix.npy"))

        # Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
        self.EXCLUDE = np.load(os.path.join(INPUT_DIR, "x_mrj.npy"))
        print("Done.")

        ###############################################################
        # Miscellaneous parameters.
        ###############################################################
        print("\tLoading miscellaneous parameters...", end=" ", flush=True)

        # Derive NCELLS (number of spatial cells) from AGEC.
        (self.NCELLS,) = self.AGEC_CROPS.index.shape

        # The base year, i.e. where year index yr_idx == 0.
        self.YR_CAL_BASE = 2010
        print("Done.")

        ###############################################################
        # Set up lists of land-uses, commodities etc.
        ###############################################################
        print("\tSetting up lists of land uses, commodities, etc...", end=" ", flush=True)

        # Read in lexicographically ordered list of land-uses.
        self.AGRICULTURAL_LANDUSES = pd.read_csv((os.path.join(INPUT_DIR, 'ag_landuses.csv')), header = None)[0].to_list()
        self.NON_AGRICULTURAL_LANDUSES = pd.read_csv((os.path.join(INPUT_DIR, 'non_ag_landuses.csv')), header = None)[0].to_list()

        self.NONAGLU2DESC = dict(zip(range(NON_AGRICULTURAL_LU_BASE_CODE, 
                                    NON_AGRICULTURAL_LU_BASE_CODE + len(self.NON_AGRICULTURAL_LANDUSES)),
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
            self.DESC2NONAGLU["Agroforestry"],
            self.DESC2NONAGLU["Carbon Plantings (Block)"],
            self.DESC2NONAGLU["Carbon Plantings (Belt)"],
            self.DESC2NONAGLU["BECCS"],
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
        self.PR2LU_DICT = {}
        for key, val in self.LU2PR_DICT.items():
            for pr in val:
                self.PR2LU_DICT[pr] = key

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
        for key, value in self.CM2PR_DICT.items():
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
        print("Done.")

        ###############################################################
        # Agricultural management options data.
        ###############################################################
        print("\tLoading agricultural management options' data...", end=" ", flush=True)

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
        self.SOIL_CARBON_AVG_T_CO2_HA = (
            pd.read_hdf(os.path.join(INPUT_DIR, "soil_carbon_t_ha.h5")).to_numpy(dtype=np.float32)
            * (44 / 12)
            / SOC_AMORTISATION
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

        print("Done.")

        ###############################################################
        # Non-agricultural data.
        ###############################################################
        print("\tLoading non-agricultural data...", end=" ", flush=True)

        # Load plantings economic data
        self.EP_EST_COST_HA = pd.read_hdf(os.path.join(INPUT_DIR, "ep_est_cost_ha.h5")).to_numpy(
            dtype=np.float32
        )
        self.CP_EST_COST_HA = pd.read_hdf(os.path.join(INPUT_DIR, "cp_est_cost_ha.h5")).to_numpy(
            dtype=np.float32
        )

        # Load fire risk data (reduced carbon sequestration by this amount)
        fr_df = pd.read_hdf(os.path.join(INPUT_DIR, "fire_risk.h5"))
        fr_dict = {"low": "FD_RISK_PERC_5TH", "med": "FD_RISK_MEDIAN", "high": "FD_RISK_PERC_95TH"}
        fire_risk = fr_df[fr_dict[FIRE_RISK]]

        # Load environmental plantings (block) GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
        ep_df = pd.read_hdf(os.path.join(INPUT_DIR, "ep_block_avg_t_co2_ha_yr.h5"))
        self.EP_BLOCK_AVG_T_CO2_HA = (
            (ep_df.EP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL))
            + ep_df.EP_BLOCK_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)

        # Load environmental plantings (belt) GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
        ep_df = pd.read_hdf(os.path.join(INPUT_DIR, "ep_belt_avg_t_co2_ha_yr.h5"))
        self.EP_BELT_AVG_T_CO2_HA = (
            (ep_df.EP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL))
            + ep_df.EP_BELT_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)

        # Load environmental plantings (riparian) GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
        ep_df = pd.read_hdf(os.path.join(INPUT_DIR, "ep_rip_avg_t_co2_ha_yr.h5"))
        self.EP_RIP_AVG_T_CO2_HA = (
            (ep_df.EP_RIP_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL))
            + ep_df.EP_RIP_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)

        # Load carbon plantings (block) GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
        cp_df = pd.read_hdf(os.path.join(INPUT_DIR, "cp_block_avg_t_co2_ha_yr.h5"))
        self.CP_BLOCK_AVG_T_CO2_HA = (
            (cp_df.CP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL))
            + cp_df.CP_BLOCK_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)

        # Load farm forestry [i.e. carbon plantings (belt)] GHG sequestration (aboveground carbon discounted by RISK_OF_REVERSAL and FIRE_RISK)
        cp_df = pd.read_hdf(os.path.join(INPUT_DIR, "cp_belt_avg_t_co2_ha_yr.h5"))
        self.CP_BELT_AVG_T_CO2_HA = (
            (cp_df.CP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - RISK_OF_REVERSAL))
            + cp_df.CP_BELT_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)

        # Agricultural land use to plantings raw transition costs:
        self.AG2EP_TRANSITION_COSTS_HA = np.load(
            os.path.join(INPUT_DIR, "ag_to_ep_tmatrix.npy")
        )  # shape: (28,)

        # EP to agricultural land use transition costs:
        self.EP2AG_TRANSITION_COSTS_HA = np.load(
            os.path.join(INPUT_DIR, "ep_to_ag_tmatrix.npy")
        )  # shape: (28,)
        print("Done.")

        ###############################################################
        # Spatial layers.
        ###############################################################
        print("\tSetting up spatial layers data...", end=" ", flush=True)

        # NLUM mask.
        with rasterio.open(os.path.join(INPUT_DIR, "NLUM_2010-11_mask.tif")) as rst:
            self.NLUM_MASK = rst.read(1)

        # Actual hectares per cell, including projection corrections.
        self.REAL_AREA = pd.read_hdf(os.path.join(INPUT_DIR, "real_area.h5")).to_numpy()

        # Initial (2010) land-use map, mapped as lexicographic land-use class indices.
        self.LUMAP = pd.read_hdf(os.path.join(INPUT_DIR, "lumap.h5")).to_numpy()

        # Initial (2010) land management map.
        self.LMMAP = pd.read_hdf(os.path.join(INPUT_DIR, "lmmap.h5")).to_numpy()

        # Initial (2010) agricutural management maps - no cells are used for alternative agricultural management options.
        # Includes a separate AM map for each agricultural management option, because they can be stacked.
        self.AMMAP_DICT = {
            am: np.zeros(self.NCELLS).astype("int8") for am in AG_MANAGEMENTS_TO_LAND_USES
        }

        # Load stream length data in metres of stream per cell
        self.STREAM_LENGTH = pd.read_hdf(
            os.path.join(INPUT_DIR, "stream_length_m_cell.h5")
        ).to_numpy()

        # Calculate the proportion of the area of each cell within stream buffer (convert REAL_AREA from ha to m2 and divide m2 by m2)
        self.RP_PROPORTION = (
            (2 * RIPARIAN_PLANTINGS_BUFFER_WIDTH * self.STREAM_LENGTH) / (self.REAL_AREA * 10000)
        ).astype(np.float32)

        # Calculate the length of fencing required for each cell in per hectare terms for riparian plantings
        self.RP_FENCING_LENGTH = (
            (2 * RIPARIAN_PLANTINGS_TORTUOSITY_FACTOR * self.STREAM_LENGTH) / self.REAL_AREA
        ).astype(np.float32)
        print("Done.")

        ###############################################################
        # Masking and spatial coarse graining.
        ###############################################################
        print("\tSetting up masking and spatial course graining data...", end=" ", flush=True)

        # Set resfactor multiplier
        self.RESMULT = RESFACTOR**2

        # Mask out non-agricultural, non-environmental plantings land (i.e., -1) from lumap (True means included cells. Boolean dtype.)
        self.MASK_LU_CODE = -1
        self.LUMASK = self.LUMAP != self.MASK_LU_CODE

        # Return combined land-use and resfactor mask
        if RESFACTOR > 1:

            # Create resfactor mask for spatial coarse-graining.
            rf_mask = self.NLUM_MASK.copy()
            nonzeroes = np.nonzero(rf_mask)
            rf_mask[::RESFACTOR, ::RESFACTOR] = 0
            resmask = np.where(rf_mask[nonzeroes] == 0, True, False)

            # Superimpose resfactor mask upon land-use map mask (Boolean).
            self.MASK = self.LUMASK * resmask

        elif RESFACTOR == 1:
            self.MASK = self.LUMASK

        else:
            raise KeyError("RESFACTOR setting invalid")

        # Create a mask indices array for subsetting arrays
        self.MINDICES = np.where(self.MASK)[0].astype(np.int32)
        print("Done.")

        ###############################################################
        # Water data.
        ###############################################################
        print("\tLoading water data...", end=" ", flush=True)

        # Water requirements by land use -- LVSTK.
        wreq_lvstk_dry = pd.DataFrame()
        wreq_lvstk_irr = pd.DataFrame()

        # The rj-indexed arrays have zeroes where j is not livestock.
        for lu in self.AGRICULTURAL_LANDUSES:
            if lu in self.LU_LVSTK:
                # First find out which animal is involved.
                animal, _ = lvs_veg_types(lu)
                # Water requirements per head are for drinking and irrigation.
                wreq_lvstk_dry[lu] = self.AGEC_LVSTK["WR_DRN", animal]
                wreq_lvstk_irr[lu] = (
                    self.AGEC_LVSTK["WR_DRN", animal] + self.AGEC_LVSTK["WR_IRR", animal]
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
            pd.read_hdf(os.path.join(INPUT_DIR, "water_licence_price.h5")).to_numpy()
        )

        # Spatially explicit costs of water delivery per ML.
        self.WATER_DELIVERY_PRICE = np.nan_to_num(
            pd.read_hdf(os.path.join(INPUT_DIR, "water_delivery_price.h5")).to_numpy()
        )

        # River regions.
        self.RIVREG_ID = pd.read_hdf(
            os.path.join(INPUT_DIR, "rivreg_id.h5")
        ).to_numpy()  # River region ID mapped.
        rr = pd.read_hdf(os.path.join(INPUT_DIR, "rivreg_lut.h5"))
        self.RIVREG_DICT = dict(
            zip(rr.HR_RIVREG_ID, rr.HR_RIVREG_NAME)
        )  # River region ID to Name lookup table
        self.RIVREG_LIMITS = dict(
            zip(rr.HR_RIVREG_ID, rr.WATER_YIELD_HIST_BASELINE_ML)
        )  # River region ID and water use limits

        # Drainage divisions
        self.DRAINDIV_ID = pd.read_hdf(
            os.path.join(INPUT_DIR, "draindiv_id.h5")
        ).to_numpy()  # Drainage div ID mapped.
        dd = pd.read_hdf(os.path.join(INPUT_DIR, "draindiv_lut.h5"))
        self.DRAINDIV_DICT = dict(
            zip(dd.HR_DRAINDIV_ID, dd.HR_DRAINDIV_NAME)
        )  # Drainage div ID to Name lookup table
        self.DRAINDIV_LIMITS = dict(
            zip(dd.HR_DRAINDIV_ID, dd.WATER_YIELD_HIST_BASELINE_ML)
        )  # Drainage div ID and water use limits

        # Water yields -- run off from a cell into catchment by deep-rooted, shallow-rooted, and natural vegetation types
        water_yield_base = pd.read_hdf(os.path.join(INPUT_DIR, "water_yield_baselines.h5"))
        # WATER_YIELD_BASE_DR = water_yield_base['WATER_YIELD_HIST_DR_ML_HA'].to_numpy(dtype = np.float32)
        self.WATER_YIELD_BASE_SR = water_yield_base["WATER_YIELD_HIST_SR_ML_HA"].to_numpy(
            dtype=np.float32
        )
        self.WATER_YIELD_BASE = water_yield_base["WATER_YIELD_HIST_BASELINE_ML_HA"].to_numpy(
            dtype=np.float32
        )

        # fname_dr = os.path.join(INPUT_DIR, 'water_yield_ssp' + SSP + '_2010-2100_dr_ml_ha.h5')
        # fname_sr = os.path.join(INPUT_DIR, 'water_yield_ssp' + SSP + '_2010-2100_sr_ml_ha.h5')

        # wy_dr_file = h5py.File(fname_dr, 'r')
        # wy_sr_file = h5py.File(fname_sr, 'r')

        # Water yields for current year -- placeholder slice for year zero. ############### Better to slice off a year as the file is >2 GB  TODO
        # WATER_YIELD_NUNC_DR = wy_dr_file[list(wy_dr_file.keys())[0]][0]                   # This might go in the simulation module where year is specified to save loading into memory
        # WATER_YIELD_NUNC_SR = wy_sr_file[list(wy_sr_file.keys())[0]][0]

        print("Done.")

        ###############################################################
        # Carbon sequestration by trees data.
        ###############################################################
        print("\tLoading carbon sequestration by trees data...", end=" ", flush=True)

        # Load the remnant vegetation carbon data.
        rem_veg = pd.read_hdf(os.path.join(INPUT_DIR, "natural_land_t_co2_ha.h5")).to_numpy(
            dtype=np.float32
        )
        rem_veg = np.squeeze(rem_veg)  # Remove extraneous extra dimension

        # Discount by fire risk.
        self.NATURAL_LAND_T_CO2_HA = rem_veg * (fire_risk.to_numpy() / 100)
        print("Done.")

        ###############################################################
        # Climate change impact data.
        ###############################################################
        print("\tLoading climate change data...", end=" ", flush=True)

        self.CLIMATE_CHANGE_IMPACT = pd.read_hdf(
            os.path.join(
                INPUT_DIR, "climate_change_impacts_" + RCP + "_CO2_FERT_" + CO2_FERT.upper() + ".h5"
            )
        )
        print("Done.")

        ###############################################################
        # Livestock related data.
        ###############################################################
        print("\tLoading livestock related data...", end=" ", flush=True)

        self.FEED_REQ = np.nan_to_num(
            pd.read_hdf(os.path.join(INPUT_DIR, "feed_req.h5")).to_numpy()
        )
        self.PASTURE_KG_DM_HA = pd.read_hdf(
            os.path.join(INPUT_DIR, "pasture_kg_dm_ha.h5")
        ).to_numpy()
        self.SAFE_PUR_NATL = pd.read_hdf(os.path.join(INPUT_DIR, "safe_pur_natl.h5")).to_numpy()
        self.SAFE_PUR_MODL = pd.read_hdf(os.path.join(INPUT_DIR, "safe_pur_modl.h5")).to_numpy()
        print("Done.")

        ###############################################################
        # Productivity data.
        ###############################################################
        print("\tLoading productivity data...", end=" ", flush=True)

        # Yield increases.
        fpath = os.path.join(INPUT_DIR, "yieldincreases_bau2022.csv")
        self.BAU_PROD_INCR = pd.read_csv(fpath, header=[0, 1]).astype(np.float32)
        print("Done.")

        ###############################################################
        # Demand data.
        ###############################################################
        print("\tLoading demand data...", end=" ", flush=True)

        # Load demand data (actual production (tonnes, ML) by commodity) - from demand model     
        dd = pd.read_hdf(os.path.join(INPUT_DIR, 'demand_projections.h5') )

        # Select the demand data under the running scenario
        self.DEMAND_DATA = dd.loc[(SCENARIO, 
                                   DIET_DOM, 
                                   DIET_GLOB, 
                                   CONVERGENCE,
                                   IMPORT_TREND, 
                                   WASTE, 
                                   FEED_EFFICIENCY)].copy()

        # Convert eggs from count to tonnes
        self.DEMAND_DATA.loc['eggs'] = self.DEMAND_DATA.loc['eggs'] * EGGS_AVG_WEIGHT / 1000 / 1000

        # Get the off-land commodities
        self.DEMAND_OFFLAND = self.DEMAND_DATA.loc[self.DEMAND_DATA.query("COMMODITY in @OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()

        # Remove off-land commodities
        self.DEMAND_C = self.DEMAND_DATA.loc[self.DEMAND_DATA.query("COMMODITY not in @OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()

        # Convert to numpy array of shape (91, 26)
        self.DEMAND_C = self.DEMAND_C.to_numpy(dtype = np.float32).T
        self.D_CY = self.DEMAND_C # new demand is in tonnes rather than deltas
        print("Done.")
        

        ###############################################################
        # Carbon emissions from off-land commodities.
        ###############################################################
        print("\tLoading off-land commodities' carbon emissions data...", end=" ", flush=True)

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
        print("Done.")

        ###############################################################
        # GHG targets data.
        ###############################################################
        print("\tLoading GHG targets data...", end=" ", flush=True)

        # If GHG_LIMITS_TYPE == 'file' then import the Excel spreadsheet and import the results to a python dictionary {year: target (tCO2e), ...}
        if GHG_LIMITS_TYPE == "file":
            self.GHG_TARGETS = pd.read_excel(
                os.path.join(INPUT_DIR, "GHG_targets.xlsx"), sheet_name="Data", index_col="YEAR"
            )
            self.GHG_TARGETS = self.GHG_TARGETS[GHG_LIMITS_FIELD].to_dict()

        # If GHG_LIMITS_TYPE == 'dict' then import the Excel spreadsheet and import the results to a python dictionary {year: target (tCO2e), ...}
        elif GHG_LIMITS_TYPE == "dict":

            # Create a dictionary to hold the GHG target data
            self.GHG_TARGETS = {}  # pd.DataFrame(columns = ['TOTAL_GHG_TCO2E'])

            # # Create linear function f and interpolate
            f = interp1d(
                list(GHG_LIMITS.keys()),
                list(GHG_LIMITS.values()),
                kind="linear",
                fill_value="extrapolate",
            )
            keys = range(2010, 2101)
            for yr in range(2010, 2101):
                self.GHG_TARGETS[yr] = f(yr)
        print("Done.")

        ###############################################################
        # Savanna burning data.
        ###############################################################
        print("\tLoading savanna burning data...", end=" ", flush=True)

        # Read in the dataframe
        savburn_df = pd.read_hdf(os.path.join(INPUT_DIR, 'cell_savanna_burning.h5') )

        # Load the columns as numpy arrays
        self.SAVBURN_ELIGIBLE = savburn_df.ELIGIBLE_AREA.to_numpy()               # 1 = areas eligible for early dry season savanna burning under the ERF, 0 = ineligible
        self.SAVBURN_AVEM_CH4_TCO2E_HA = savburn_df.SAV_AVEM_CH4_TCO2E_HA.to_numpy()  # Avoided emissions - methane
        self.SAVBURN_AVEM_N2O_TCO2E_HA = savburn_df.SAV_AVEM_N2O_TCO2E_HA.to_numpy()  # Avoided emissions - nitrous oxide
        self.SAVBURN_SEQ_CO2_TCO2E_HA = savburn_df.SAV_SEQ_CO2_TCO2E_HA.to_numpy()    # Additional carbon sequestration - carbon dioxide
        self.SAVBURN_TOTAL_TCO2E_HA = savburn_df.AEA_TOTAL_TCO2E_HA.to_numpy()        # Total emissions abatement from EDS savanna burning

        # Cost per hectare in dollars
        self.SAVBURN_COST_HA = 2
        print("Done.")

        ###############################################################
        # Biodiversity data.
        ###############################################################
        print("\tLoading biodiversity data...", end=" ", flush=True)
        """
        Kunming-Montreal Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
        Ensure that by 2030 at least 30 per cent of areas of degraded terrestrial, inland water, and coastal and marine ecosystems are under effective restoration, in order to enhance biodiversity and ecosystem functions and services, ecological integrity and connectivity.
        """
        # Load biodiversity data
        biodiv_priorities = pd.read_hdf(os.path.join(INPUT_DIR, 'biodiv_priorities.h5') )

        # Get the Zonation output score between 0 and 1. BIODIV_SCORE_RAW.sum() = 153 million
        self.BIODIV_SCORE_RAW = biodiv_priorities['BIODIV_PRIORITY_SSP' + SSP].to_numpy(dtype = np.float32)

        # Get the natural area connectivity score between 0 and 1 (1 is highly connected, 0 is not connected)
        conn_score = biodiv_priorities['NATURAL_AREA_CONNECTIVITY'].to_numpy(dtype = np.float32)

        # Calculate weighted biodiversity score
        self.BIODIV_SCORE_WEIGHTED = self.BIODIV_SCORE_RAW - (self.BIODIV_SCORE_RAW * (1 - conn_score) * CONNECTIVITY_WEIGHTING)
        self.BIODIV_SCORE_WEIGHTED_LDS_BURNING = self.BIODIV_SCORE_WEIGHTED * LDS_BIODIVERSITY_VALUE

        # Calculate total biodiversity target score as the quality-weighted sum of biodiv raw score over the study area 
        biodiv_value_current = ( np.isin(self.LUMAP, 23) * self.BIODIV_SCORE_RAW +                                         # Biodiversity value of Unallocated - natural land 
                                 np.isin(self.LUMAP, [2, 6, 15]) * self.BIODIV_SCORE_RAW * (1 - BIODIV_LIVESTOCK_IMPACT)   # Biodiversity value of livestock on natural land 
                               ) * np.where(self.SAVBURN_ELIGIBLE, LDS_BIODIVERSITY_VALUE, 1) * self.REAL_AREA             # Reduce biodiversity value of area eligible for savanna burning 

        biodiv_value_target = ( ( np.isin(self.LUMAP, [2, 6, 15, 23]) * self.BIODIV_SCORE_RAW * self.REAL_AREA - biodiv_value_current ) +  # On natural land calculate the difference between the raw biodiversity score and the current score
                                np.isin(self.LUMAP, self.LU_MODIFIED_LAND) * self.BIODIV_SCORE_RAW * self.REAL_AREA                        # Calculate raw biodiversity score of modified land
                              ) * BIODIV_TARGET                                                                                            # Multiply by biodiversity target to get the additional biodiversity score required to achieve the target
                                
        # Sum the current biodiversity value and the additional biodiversity score required to meet the target
        self.TOTAL_BIODIV_TARGET_SCORE = biodiv_value_current.sum() + biodiv_value_target.sum()                         

        """
        TOTAL_BIODIV_TARGET_SCORE = ( 
                                    np.isin(LUMAP, 23) * BIODIV_SCORE_RAW * REAL_AREA +                                                   # Biodiversity value of Unallocated - natural land 
                                    np.isin(LUMAP, [2, 6, 15]) * BIODIV_SCORE_RAW * (1 - BIODIV_LIVESTOCK_IMPACT) * REAL_AREA +           # Biodiversity value of livestock on natural land 
                                    np.isin(LUMAP, [2, 6, 15]) * BIODIV_SCORE_RAW * BIODIV_LIVESTOCK_IMPACT * BIODIV_TARGET * REAL_AREA + # Add 30% improvement to the degraded part of livestock on natural land
                                    np.isin(LUMAP, LU_MODIFIED_LAND) * BIODIV_SCORE_RAW * BIODIV_TARGET * REAL_AREA                       # Add 30% improvement to modified land
                                    ).sum() 
        """
        print("Done.")

        ###############################################################
        # BECCS data.
        ###############################################################
        print("\tLoading BECCS data...", end=" ", flush=True)

        # Load dataframe
        beccs_df = pd.read_hdf(os.path.join(INPUT_DIR, 'cell_BECCS_df.h5') )

        # Capture as numpy arrays
        self.BECCS_COSTS_AUD_HA_YR = beccs_df['BECCS_COSTS_AUD_HA_YR'].to_numpy()
        self.BECCS_REV_AUD_HA_YR = beccs_df['BECCS_REV_AUD_HA_YR'].to_numpy()
        self.BECCS_TCO2E_HA_YR = beccs_df['BECCS_TCO2E_HA_YR'].to_numpy()
        self.BECCS_MWH_HA_YR = beccs_df['BECCS_MWH_HA_YR'].to_numpy()
        
        print("Done.")
        print("\nData loading complete.")

    def apply_resfactor(self):
        """
        Sub-set spatial data is based on the masks.
        """
        print("Adjusting data for resfactor...", end=" ", flush=True)
        self.NCELLS = self.MASK.sum()
        self.EXCLUDE = self.EXCLUDE[:, self.MASK, :]
        self.AGEC_CROPS = self.AGEC_CROPS.iloc[self.MASK]  # MultiIndex Dataframe [4218733 rows x 342 columns]
        self.AGEC_LVSTK = self.AGEC_LVSTK.iloc[self.MASK]  # MultiIndex Dataframe [4218733 rows x 39 columns]
        self.AGGHG_CROPS = self.AGGHG_CROPS.iloc[self.MASK]  # MultiIndex Dataframe [4218733 rows x ? columns]
        self.AGGHG_LVSTK = self.AGGHG_LVSTK.iloc[self.MASK]  # MultiIndex Dataframe [4218733 rows x ? columns]
        self.REAL_AREA = self.REAL_AREA[self.MASK] * self.RESMULT  # Actual Float32
        self.LUMAP = self.LUMAP[self.MASK]  # Int8
        self.LMMAP = self.LMMAP[self.MASK]  # Int8
        self.AMMAP_DICT = {
            am: array[self.MASK] for am, array in self.AMMAP_DICT.items()
        }  # Dictionary containing Int8 arrays
        self.AG_L_MRJ = lumap2ag_l_mrj(self.LUMAP, self.LMMAP)  # Boolean [2, 4218733, 28]
        self.NON_AG_L_RK = lumap2non_ag_l_mk(
            self.LUMAP, len(self.NON_AGRICULTURAL_LANDUSES)
        )  # Int8
        self.AG_MAN_L_MRJ_DICT = get_base_am_vars(
            self.NCELLS, self.NLMS, self.N_AG_LUS
        )  # Dictionary containing Int8 arrays
        self.WREQ_IRR_RJ = self.WREQ_IRR_RJ[self.MASK]                          # Water requirements for irrigated landuses
        self.WREQ_DRY_RJ = self.WREQ_DRY_RJ[self.MASK]                          # Water requirements for dryland landuses
        self.WATER_LICENCE_PRICE = self.WATER_LICENCE_PRICE[self.MASK]          # Int16
        self.WATER_DELIVERY_PRICE = self.WATER_DELIVERY_PRICE[self.MASK]        # Float32
        # self.WATER_YIELD_BASE_DR = bdata.WATER_YIELD_BASE_DR[self.MASK]       # Float32
        self.WATER_YIELD_BASE_SR = self.WATER_YIELD_BASE_SR[self.MASK]          # Float32
        self.WATER_YIELD_BASE = self.WATER_YIELD_BASE[self.MASK]                # Float32
        self.FEED_REQ = self.FEED_REQ[self.MASK]                                # Float32
        self.PASTURE_KG_DM_HA = self.PASTURE_KG_DM_HA[self.MASK]                # Int16
        self.SAFE_PUR_MODL = self.SAFE_PUR_MODL[self.MASK]                      # Float32
        self.SAFE_PUR_NATL = self.SAFE_PUR_NATL[self.MASK]                      # Float32
        self.RIVREG_ID = self.RIVREG_ID[self.MASK]                              # Int16
        self.DRAINDIV_ID = self.DRAINDIV_ID[self.MASK]                          # Int8
        self.CLIMATE_CHANGE_IMPACT = self.CLIMATE_CHANGE_IMPACT[self.MASK]
        self.EP_EST_COST_HA = self.EP_EST_COST_HA[self.MASK]                    # Float32
        self.AG2EP_TRANSITION_COSTS_HA = self.AG2EP_TRANSITION_COSTS_HA         # Float32
        self.EP2AG_TRANSITION_COSTS_HA = self.EP2AG_TRANSITION_COSTS_HA         # Float32
        self.EP_BLOCK_AVG_T_CO2_HA = self.EP_BLOCK_AVG_T_CO2_HA[self.MASK]      # Float32
        self.NATURAL_LAND_T_CO2_HA = self.NATURAL_LAND_T_CO2_HA[self.MASK]      # Float32
        self.SOIL_CARBON_AVG_T_CO2_HA = self.SOIL_CARBON_AVG_T_CO2_HA[self.MASK]
        self.AGGHG_IRRPAST = self.AGGHG_IRRPAST[self.MASK]                      # Float32
        self.BIODIV_SCORE_RAW = self.BIODIV_SCORE_RAW[self.MASK]                # Float32
        self.BIODIV_SCORE_WEIGHTED = self.BIODIV_SCORE_WEIGHTED[self.MASK]      # Float32
        self.BIODIV_SCORE_WEIGHTED_LDS_BURNING = (
            self.BIODIV_SCORE_WEIGHTED_LDS_BURNING[self.MASK]
        )
        self.RP_PROPORTION = self.RP_PROPORTION[self.MASK]                      # Float32
        self.RP_FENCING_LENGTH = self.RP_FENCING_LENGTH[self.MASK]              # Float32
        self.EP_RIP_AVG_T_CO2_HA = self.EP_RIP_AVG_T_CO2_HA[self.MASK]          # Float32
        self.EP_BELT_AVG_T_CO2_HA = self.EP_BELT_AVG_T_CO2_HA[self.MASK]        # Float32
        self.CP_BLOCK_AVG_T_CO2_HA = self.CP_BLOCK_AVG_T_CO2_HA[self.MASK]      # Float32
        self.CP_BELT_AVG_T_CO2_HA = self.CP_BELT_AVG_T_CO2_HA[self.MASK]        # Float32
        self.CP_EST_COST_HA = self.CP_EST_COST_HA[self.MASK]                    # Float32
        self.BECCS_COSTS_AUD_HA_YR = self.BECCS_COSTS_AUD_HA_YR[self.MASK]      # Float32
        self.BECCS_REV_AUD_HA_YR = self.BECCS_REV_AUD_HA_YR[self.MASK]          # Float32
        self.BECCS_TCO2E_HA_YR = self.BECCS_TCO2E_HA_YR[self.MASK]              # Float32
        self.SAVBURN_ELIGIBLE = self.SAVBURN_ELIGIBLE[self.MASK]                # Int8
        self.SAVBURN_TOTAL_TCO2E_HA = self.SAVBURN_TOTAL_TCO2E_HA[self.MASK]    # Float32

        # Slice this year off HDF5 bricks. TODO: This field is not in luto.data.
        # with h5py.File(bdata.fname_dr, 'r') as wy_dr_file:
        #     self.WATER_YIELD_DR = wy_dr_file[list(wy_dr_file.keys())[0]][yr_idx][self.MASK]
        # with h5py.File(bdata.fname_sr, 'r') as wy_sr_file:
        #     self.WATER_YIELD_SR = wy_sr_file[list(wy_sr_file.keys())[0]][yr_idx][self.MASK]

        print("Done.")

    def add_lumap(self, yr: int, lumap: np.ndarray):
        """
        Safely adds a land-use map to the the Data object.
        """
        if yr in self.lumaps:
            raise ValueError(f"LUMAP for year {yr} already stored in Data object.")
        self.lumaps[yr] = lumap

    def add_lmmap(self, yr: int, lmmap: np.ndarray):
        """
        Safely adds a land-management map to the Data object.
        """
        if yr in self.lmmaps:
            raise ValueError(f"LMMAP for year {yr} already stored in Data object.")
        self.lmmaps[yr] = lmmap

    def add_ammaps(self, yr: int, ammap: np.ndarray):
        """
        Safely adds an agricultural management map to the Data object.
        """
        if yr in self.ammaps:
            raise ValueError(f"LUMAP for year {yr} already stored in Data object.")
        self.ammaps[yr] = ammap

    def add_ag_dvars(self, yr: int, ag_dvars: np.ndarray):
        """
        Safely adds agricultural decision variables' values to the Data object.
        """
        if yr in self.ag_dvars:
            raise ValueError(f"Agr. DVars for year {yr} already stored in Data object.")
        self.ag_dvars[yr] = ag_dvars

    def add_non_ag_dvars(self, yr: int, non_ag_dvars: np.ndarray):
        """
        Safely adds non-agricultural decision variables' values to the Data object.
        """
        if yr in self.non_ag_dvars:
            raise ValueError(f"Non-agr. DVars for year {yr} already stored in Data object.")
        self.non_ag_dvars[yr] = non_ag_dvars

    def add_ag_man_dvars(self, yr: int, ag_man_dvars: dict[str, np.ndarray]):
        """
        Safely adds agricultural management decision variables' values to the Data object.
        """
        if yr in self.ag_man_dvars:
            raise ValueError(f"Ag management DVars for year {yr} already stored in Data object.")
        self.ag_man_dvars[yr] = ag_man_dvars

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
        if yr in self.obj_vals:
            raise ValueError(f"Objective value for year {yr} already stored in Data object.")
        self.obj_vals[yr] = obj_val

    def add_base_year_data_to_containers(self):
        """
        Adds all 'solution' data for the base year to the corresponding containers.

        To be called after applying resfactor to data using 'self.apply_resfactor()'
        """
        self.add_lumap(self.YR_CAL_BASE, self.LUMAP)
        self.add_lmmap(self.YR_CAL_BASE, self.LMMAP)
        self.add_ammaps(self.YR_CAL_BASE, self.AMMAP_DICT)
        self.add_ag_dvars(self.YR_CAL_BASE, self.AG_L_MRJ)
        self.add_non_ag_dvars(self.YR_CAL_BASE, self.NON_AG_L_RK)
        self.add_ag_man_dvars(self.YR_CAL_BASE, self.AG_MAN_L_MRJ_DICT)

    def set_path(self, base_year, target_year) -> str:
        """Create a folder for storing outputs and return folder name."""

        # Get the years to write
        if MODE == "snapshot":
            yr_all = [base_year, target_year]
        elif MODE == "timeseries":
            yr_all = list(range(base_year, target_year + 1))

        # Add some shorthand details about the model run
        post = (
            "_"
            + DEMAND_CONSTRAINT_TYPE
            + "_"
            + OBJECTIVE
            + "_RF"
            + str(RESFACTOR)
            + "_P1e"
            + str(int(math.log10(PENALTY)))
            + "_"
            + str(yr_all[0])
            + "-"
            + str(yr_all[-1])
            + "_"
            + MODE
            + "_"
            + str(int(self.GHG_TARGETS[yr_all[-1]] / 1e6))
            + "Mt"
        )

        # Create path name
        self.path = "output/" + self.timestamp_sim + post

        # Get all paths
        paths = (
            [self.path]
            + [f"{self.path}/out_{yr}" for yr in yr_all]
            + [f"{self.path}/out_{yr}/lucc_separate" for yr in yr_all[1:]]
        )  # Skip creating lucc_separate for base year

        # Add the path for the comparison between base-year and target-year if in the timeseries mode
        if MODE == "timeseries":
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
