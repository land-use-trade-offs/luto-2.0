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



import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional

from luto.data import Data
from luto import settings
from luto.economics import land_use_culling

import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.transitions as ag_transition
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.biodiversity as ag_biodiversity

import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.non_agricultural.biodiversity as non_ag_biodiversity
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.quantity as non_ag_quantity
import luto.economics.non_agricultural.transitions as non_ag_transition
import luto.economics.non_agricultural.revenue as non_ag_revenue


@dataclass
class SolverInputData:
    """
    An object that collects and stores all relevant data for solver.py.
    """   
    base_year: int                                                      # The base year of this solving process
    target_year: int                                                    # The target year of this solving process

    ag_g_mrj: np.ndarray                                                # Agricultural greenhouse gas emissions matrices.
    ag_w_mrj: np.ndarray                                                # Agricultural water yields matrices.
    ag_b_mrj: np.ndarray                                                # Agricultural biodiversity matrices.
    ag_x_mrj: np.ndarray                                                # Agricultural exclude matrices.
    ag_q_mrp: np.ndarray                                                # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
    ag_ghg_t_mrj: np.ndarray                                            # GHG emissions released during transitions between agricultural land uses.

    non_ag_g_rk: np.ndarray                                             # Non-agricultural greenhouse gas emissions matrix.
    non_ag_w_rk: np.ndarray                                             # Non-agricultural water yields matrix.
    non_ag_b_rk: np.ndarray                                             # Non-agricultural biodiversity matrix.
    non_ag_x_rk: np.ndarray                                             # Non-agricultural exclude matrices.
    non_ag_q_crk: np.ndarray                                            # Non-agricultural yield matrix.
    non_ag_lb_rk: np.ndarray                                            # Non-agricultural lower bound matrices.

    ag_man_g_mrj: dict                                                  # Agricultural Management options' GHG emission effects.
    ag_man_w_mrj: dict                                                  # Agricultural Management options' water yield effects.
    ag_man_b_mrj: dict                                                  # Agricultural Management options' biodiversity effects.
    ag_man_q_mrp: dict                                                  # Agricultural Management options' quantity effects.
    ag_man_limits: dict                                                 # Agricultural Management options' adoption limits.
    ag_man_lb_mrj: dict                                                 # Agricultural Management options' lower bounds.

    water_region_indices: dict[int, np.ndarray]                         # Water region indices -> dict. Key: region.
    water_region_names: dict[int, str]                                  # Water yield for the BASE_YR based on historical water yield layers.
      
    biodiv_contr_ag_j: np.ndarray                                       # Biodiversity contribution scale from agricultural land uses.
    biodiv_contr_non_ag_k: dict[int, float]                             # Biodiversity contribution scale from non-agricultural land uses.
    biodiv_contr_ag_man: dict[str, dict[int, np.ndarray]]               # Biodiversity contribution scale from agricultural management options.
    
    GBF2_raw_priority_degraded_area_r: np.ndarray                       # Raw areas (GBF2) from priority degrade areas - indexed by cell (r).
    GBF3_raw_MVG_area_vr: np.ndarray                                    # Raw areas (GBF3) from Major vegetation group - indexed by veg. group (v) and cell (r)
    GBF3_names: dict[int, str]                                          # Major vegetation groups names - indexed by major vegetation group (v).
    GBF3_ind: dict[str, int]                                            # Major vegetation groups indices - indexed by major vegetation group (v).
    GBF4_SNES_xr: np.ndarray                                            # Raw areas (GBF4) Species NES contribution data - indexed by species/ecological community (x) and cell (r).
    GBF4_SNES_names: dict[int, str]                                     # Species NES names - indexed by species/ecological community (x).
    GBF4_ECNES_xr: np.ndarray                                           # Raw areas (GBF4) Ecological community NES contribution data - indexed by species/ecological community (x) and cell (r).
    GBF4_ECNES_names: dict[int, str]                                    # Ecological community NES names - indexed by species/ecological community (x).
    GBF8_raw_species_area_sr: np.ndarray                                # Raw areas (GBF8) Species data - indexed by species (s) and cell (r).
    GBF8_species_names: dict[int, str]                                  # Species names - indexed by species (s).
    GBF8_species_indices: dict[int, float]                              # Species indices - indexed by species (s).

    savanna_eligible_r: np.ndarray                                      # Cells that are eligible for savanna burnining land use.
    priority_degraded_mask_idx: np.ndarray                              # Mask of priority degraded areas - indexed by cell (r).

    base_yr_prod: dict[str, tuple]                                      # Base year production of each commodity.
    scale_factors: dict[float]                                          # Scale factors for each input layer.

    economic_contr_mrj: float                                           # base year economic contribution matrix.
    economic_BASE_YR_prices: np.ndarray                                 # base year commodity prices.
    economic_target_yr_carbon_price: float                              # target year carbon price.
    
    offland_ghg: np.ndarray                                             # GHG emissions from off-land commodities.

    lu2pr_pj: np.ndarray                                                # Conversion matrix: land-use to product(s).
    pr2cm_cp: np.ndarray                                                # Conversion matrix: product(s) to commodity.
    limits: dict                                                        # Targets to use.
    desc2aglu: dict                                                     # Map of agricultural land use descriptions to codes.
    real_area: np.ndarray                                               # Area of each cell, indexed by cell (r)
                
    @property
    def ncms(self):
        # Number of commodities
        return self.pr2cm_cp.shape[0]           
    
    @property
    def n_ag_lms(self):
        # Number of agricultural landmans
        return self.ag_g_mrj.shape[0]

    @property
    def ncells(self):
        # Number of cells
        return self.ag_g_mrj.shape[1]

    @property
    def n_ag_lus(self):
        # Number of agricultural landuses
        return self.ag_g_mrj.shape[2]

    @property
    def n_non_ag_lus(self):
        # Number of non-agricultural landuses
        return self.non_ag_g_rk.shape[1]

    @property
    def nprs(self):
        # Number of products
        return self.ag_q_mrp.shape[2]

    @cached_property
    def am2j(self):
        # Map of agricultural management options to land use codes
        return {
            am: [self.desc2aglu[lu] for lu in am_lus]
            for am, am_lus in settings.AG_MANAGEMENTS_TO_LAND_USES.items()
            if settings.AG_MANAGEMENTS[am]
        }

    @cached_property
    def j2am(self):
        _j2am = defaultdict(list)
        for am, am_j_list in self.am2j.items():
            for j in am_j_list:
                _j2am[j].append(am)

        return _j2am

    @cached_property
    def j2p(self):
        return {
            j: [p for p in range(self.nprs) if self.lu2pr_pj[p, j]]
            for j in range(self.n_ag_lus)
        }

    @cached_property
    def ag_lu2cells(self):
        # Make an index of each cell permitted to transform to each land use / land management combination
        return {
            (m, j): np.where(self.ag_x_mrj[m, :, j])[0]
            for j in range(self.n_ag_lus)
            for m in range(self.n_ag_lms)
        }

    @cached_property
    def cells2ag_lu(self) -> dict[int, list[tuple[int, int]]]:
        ag_lu2cells = self.ag_lu2cells
        cells2ag_lu = defaultdict(list)
        for (m, j), j_cells in ag_lu2cells.items():
            for r in j_cells:
                cells2ag_lu[r].append((m,j))

        return dict(cells2ag_lu)

    @cached_property
    def non_ag_lu2cells(self) -> dict[int, np.ndarray]:
        return {k: np.where(self.non_ag_x_rk[:, k])[0] for k in range(self.n_non_ag_lus)}

    @cached_property
    def cells2non_ag_lu(self) -> dict[int, list[int]]:
        non_ag_lu2cells = self.non_ag_lu2cells
        cells2non_ag_lu = defaultdict(list)
        for k, k_cells in non_ag_lu2cells.items():
            for r in k_cells:
                cells2non_ag_lu[r].append(k)

        return dict(cells2non_ag_lu) 
    

def get_ag_c_mrj(data: Data, target_index):
    print('Getting agricultural cost matrices...', flush = True)
    output = ag_cost.get_cost_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_c_rk(data: Data, ag_c_mrj: np.ndarray, lumap: np.ndarray, target_year):
    print('Getting non-agricultural cost matrices...', flush = True)
    output = non_ag_cost.get_cost_matrix(data, ag_c_mrj, lumap, target_year)
    return output.astype(np.float32)


def get_ag_r_mrj(data: Data, target_index):
    print('Getting agricultural revenue matrices...', flush = True)
    output = ag_revenue.get_rev_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_r_rk(data: Data, ag_r_mrj: np.ndarray, base_year: int, target_year: int):
    print('Getting non-agricultural revenue matrices...', flush = True)
    output = non_ag_revenue.get_rev_matrix(data, target_year, ag_r_mrj, data.lumaps[base_year])
    return output.astype(np.float32)


def get_ag_g_mrj(data: Data, target_index):
    print('Getting agricultural GHG emissions matrices...', flush = True)
    output = ag_ghg.get_ghg_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_g_rk(data: Data, ag_g_mrj, base_year):
    print('Getting non-agricultural GHG emissions matrices...', flush = True)
    output = non_ag_ghg.get_ghg_matrix(data, ag_g_mrj, data.lumaps[base_year])
    return output.astype(np.float32)


def get_ag_w_mrj(data: Data, target_index, water_dr_yield: Optional[np.ndarray] = None, water_sr_yield: Optional[np.ndarray] = None):
    print('Getting agricultural water net yield matrices based on historical water yield layers ...', flush = True)
    output = ag_water.get_water_net_yield_matrices(data, target_index, water_dr_yield, water_sr_yield)
    return output.astype(np.float32)

def get_w_region_indices(data: Data):
    if settings.WATER_LIMITS == 'off':
        return {}
    print('Getting water region indices...', flush = True)
    return data.WATER_REGION_INDEX_R

def get_w_region_names(data: Data):
    if settings.WATER_LIMITS == 'off':
        return {}
    print('Getting water region names...', flush = True)
    return data.WATER_REGION_NAMES


def get_ag_b_mrj(data: Data):
    print('Getting agricultural biodiversity requirement matrices...', flush = True)
    output = ag_biodiversity.get_bio_overall_priority_score_matrices_mrj(data)
    return output.astype(np.float32)


def get_ag_biodiv_contr_j(data: Data) -> dict[int, float]:
    print('Getting biodiversity degredation data for agricultural land uses...', flush = True)
    return ag_biodiversity.get_ag_biodiversity_contribution(data)


def get_non_ag_biodiv_impact_k(data: Data) -> dict[int, float]:
    print('Getting biodiversity benefits data for non-agricultural land uses...', flush = True)
    return non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data)


def get_ag_man_biodiv_impacts(data: Data, target_year: int) -> dict[str, dict[str, float]]:
    print('Getting biodiversity benefits data for agricultural management options...', flush = True)
    return ag_biodiversity.get_ag_management_biodiversity_contribution(data, target_year)

def get_GBF2_priority_degrade_area_r(data: Data) -> np.ndarray:
    if settings.BIODIVERSITY_TARGET_GBF_2 == "off":
        return np.empty(0)
    print('Getting priority degrade area matrices...', flush = True)
    # Need to copy if the return value is a direct reference to the data
    output = data.BIO_PRIORITY_DEGRADED_AREAS_R.copy()
    return output

def get_GBF3_MVG_area_vr(data: Data):
    if settings.BIODIVERSITY_TARGET_GBF_3 == "off":
        return np.empty(0)
    print('Getting agricultural major vegetation groups matrices...', flush = True)
    output = ag_biodiversity.get_GBF3_major_vegetation_matrices_vr(data)
    return output

def get_GBF3_major_vegetation_names(data: Data) -> dict[int,str]:
    if settings.BIODIVERSITY_TARGET_GBF_3 == "off":
        return np.empty(0)
    print('Getting agricultural major vegetation groups names...', flush = True)
    return data.BIO_GBF3_ID2DESC

def get_GBF3_major_indices(data: Data) -> dict[str, int]:
    if settings.BIODIVERSITY_TARGET_GBF_3 == "off":
        return np.empty(0)
    print('Getting agricultural major vegetation groups indices...', flush = True)
    return data.MAJOR_VEG_INDECES

def get_GBF4_SNES_xr(data: Data) -> np.ndarray:
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES != "on":
        return np.empty(0)
    return ag_biodiversity.get_GBF4_SNES_matrix_sr(data)

def get_GBF4_SNES_names(data: Data) -> dict[int,str]:
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES != "on":
        return np.empty(0)
    print('Getting agricultural species NES names...', flush = True)
    return {x: name for x, name in enumerate(data.BIO_GBF4_SNES_SEL_ALL)}

def get_GBF4_ECNES_xr(data: Data) -> np.ndarray:
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES != "on":
        return np.empty(0)
    return ag_biodiversity.get_GBF4_ECNES_matrix_sr(data)

def get_GBF4_ECNES_names(data: Data) -> dict[int,str]:
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES != "on":
        return np.empty(0)
    print('Getting agricultural ecological community NES names...', flush = True)
    return {x: name for x, name in enumerate(data.BIO_GBF4_ECNES_SEL_ALL)}

def get_GBF8_species_area_sr(data: Data, target_year: int) -> np.ndarray:
    if settings.BIODIVERSITY_TARGET_GBF_8 != "on":
        return np.empty(0)
    print('Getting species conservation cell data...', flush = True)
    return ag_biodiversity.get_GBF8_matrix_sr(data, target_year)

def get_GBF8_species_names(data: Data) -> dict[int,str]:
    if settings.BIODIVERSITY_TARGET_GBF_8 != "on":
        return np.empty(0)
    print('Getting species conservation names...', flush = True)
    return {s: spec_name for s, spec_name in enumerate(data.BIO_GBF8_SEL_SPECIES)}

def get_GBF8_indices(data: Data, yr_cal) -> dict[int, float]:
    if settings.BIODIVERSITY_TARGET_GBF_8 != "on":
        return np.empty(0)
    print('Getting species conservation indices...', flush = True)
    species_matrix = data.get_GBF8_bio_layers_by_yr(yr_cal)
    return {s: np.where(species_matrix[s] > 0)[0] for s in range(data.N_GBF8_SPECIES)}


def get_non_ag_w_rk(
    data: Data, 
    ag_w_mrj: np.ndarray, 
    base_year, 
    target_year, 
    water_dr_yield: Optional[np.ndarray] = None, 
    water_sr_yield: Optional[np.ndarray] = None
    ):
    print('Getting non-agricultural water yield matrices...', flush = True)
    yr_idx = target_year - data.YR_CAL_BASE
    output = non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj, data.lumaps[base_year], yr_idx, water_dr_yield, water_sr_yield)
    return output.astype(np.float32)


def get_non_ag_b_rk(data: Data, ag_b_mrj: np.ndarray, base_year):
    print('Getting non-agricultural biodiversity requirement matrices...', flush = True)
    output = non_ag_biodiversity.get_breq_matrix(data, ag_b_mrj, data.lumaps[base_year])
    return output.astype(np.float32)


def get_ag_q_mrp(data: Data, target_index):
    print('Getting agricultural production quantity matrices...', flush = True)
    output = ag_quantity.get_quantity_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_q_crk(data: Data, ag_q_mrp: np.ndarray, base_year: int):
    print('Getting non-agricultural production quantity matrices...', flush = True)
    output = non_ag_quantity.get_quantity_matrix(data, ag_q_mrp, data.lumaps[base_year])
    return output.astype(np.float32)


def get_ag_ghg_t_mrj(data: Data, base_year):
    print('Getting agricultural transitions GHG emissions...', flush = True)
    output = ag_ghg.get_ghg_transition_emissions(data, data.lumaps[base_year])
    return output.astype(np.float32)


def get_ag_t_mrj(data: Data, target_index, base_year):
    print('Getting agricultural transition cost matrices...', flush = True)
    
    ag_t_mrj = ag_transition.get_transition_matrices_ag2ag_from_base_year(
        data, 
        target_index, 
        base_year
    ).astype(np.float32)
    
    # Transition costs occures if the base year is not the target year
    return ag_t_mrj if (base_year - data.YR_CAL_BASE != target_index) else np.zeros_like(ag_t_mrj).astype(np.float32)


def get_ag_to_non_ag_t_rk(data: Data, target_index, base_year, ag_t_mrj):
    print('Getting agricultural to non-agricultural transition cost matrices...', flush = True)
    non_ag_t_mrj = non_ag_transition.get_transition_matrix_ag2nonag( 
        data, 
        target_index, 
        data.lumaps[base_year], 
        data.lmmaps[base_year]
    ).astype(np.float32)
    # Transition costs occures if the base year is not the target year
    return non_ag_t_mrj if (base_year - data.YR_CAL_BASE != target_index) else np.zeros_like(non_ag_t_mrj).astype(np.float32)


def get_non_ag_to_ag_t_mrj(data: Data, base_year:int, target_index: int):
    print('Getting non-agricultural to agricultural transition cost matrices...', flush = True)
    
    non_ag_to_ag_mrj = non_ag_transition.get_transition_matrix_nonag2ag(
        data, 
        target_index, 
        data.lumaps[base_year], 
        data.lmmaps[base_year],
    ).astype(np.float32)
    # Transition costs occures if the base year is not the target year
    return non_ag_to_ag_mrj if (base_year - data.YR_CAL_BASE != target_index) else np.zeros_like(non_ag_to_ag_mrj).astype(np.float32)


def get_non_ag_t_rk(data: Data, base_year):
    print('Getting non-agricultural transition cost matrices...', flush = True)
    output = non_ag_transition.get_non_ag_transition_matrix(data)
    return output.astype(np.float32)


def get_ag_x_mrj(data: Data, base_year):
    print('Getting agricultural exclude matrices...', flush = True)
    output = ag_transition.get_to_ag_exclude_matrices(data, data.lumaps[base_year])
    return output


def get_non_ag_x_rk(data: Data, base_year):
    print('Getting non-agricultural exclude matrices...', flush = True)
    output = non_ag_transition.get_to_non_ag_exclude_matrices(data, data.lumaps[base_year])
    return output


def get_ag_man_lb_mrj(data: Data, base_year):
    print('Getting agricultural lower bound matrices...', flush = True)
    output = ag_transition.get_lower_bound_agricultural_management_matrices(data, base_year)
    return output


def get_non_ag_lb_rk(data: Data, base_year):
    print('Getting non-agricultural lower bound matrices...', flush = True)
    output = non_ag_transition.get_lower_bound_non_agricultural_matrices(data, base_year)
    return output


def get_ag_man_c_mrj(data: Data, target_index, ag_c_mrj: np.ndarray):
    print('Getting agricultural management options\' cost effects...', flush = True)
    output = ag_cost.get_agricultural_management_cost_matrices(data, ag_c_mrj, target_index)
    return output


def get_ag_man_g_mrj(data: Data, target_index):
    print('Getting agricultural management options\' GHG emission effects...', flush = True)
    output = ag_ghg.get_agricultural_management_ghg_matrices(data, target_index)
    return output


def get_ag_man_q_mrj(data: Data, target_index, ag_q_mrp: np.ndarray):
    print('Getting agricultural management options\' quantity effects...', flush = True)
    output = ag_quantity.get_agricultural_management_quantity_matrices(data, ag_q_mrp, target_index)
    return output


def get_ag_man_r_mrj(data: Data, target_index, ag_r_mrj: np.ndarray):
    print('Getting agricultural management options\' revenue effects...', flush = True)
    output = ag_revenue.get_agricultural_management_revenue_matrices(data, ag_r_mrj, target_index)
    return output


def get_ag_man_t_mrj(data: Data, target_index, ag_t_mrj: np.ndarray):
    print('Getting agricultural management options\' transition cost effects...', flush = True)
    output = ag_transition.get_agricultural_management_transition_matrices(data, ag_t_mrj, target_index)
    return output


def get_ag_man_w_mrj(data: Data, target_index):
    print('Getting agricultural management options\' water yield effects...', flush = True)
    output = ag_water.get_agricultural_management_water_matrices(data, target_index)
    return output


def get_ag_man_b_mrj(data: Data, target_index, ag_b_mrj: np.ndarray):
    print('Getting agricultural management options\' biodiversity effects...', flush = True)
    output = ag_biodiversity.get_agricultural_management_biodiversity_matrices(data, ag_b_mrj, target_index)
    return output


def get_ag_man_limits(data: Data, target_index):
    print('Getting agricultural management options\' adoption limits...', flush = True)
    output = ag_transition.get_agricultural_management_adoption_limits(data, target_index)
    return output


def get_economic_mrj(
    ag_c_mrj: np.ndarray,
    ag_r_mrj: np.ndarray,
    ag_t_mrj: np.ndarray,
    ag_to_non_ag_t_rk: np.ndarray,
    non_ag_c_rk: np.ndarray,
    non_ag_r_rk: np.ndarray,
    non_ag_t_rk: np.ndarray,
    non_ag_to_ag_t_mrj: np.ndarray,
    ag_man_c_mrj: dict[str, np.ndarray],
    ag_man_r_mrj: dict[str, np.ndarray],
    ag_man_t_mrj: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray|dict[str, np.ndarray]]:
    
    print('Getting base year economic matrix...', flush = True)
    
    if settings.OBJECTIVE == "maxprofit":
        # Pre-calculate profit (revenue minus cost) for each land use
        ag_obj_mrj = ag_r_mrj - (ag_c_mrj + ag_t_mrj + non_ag_to_ag_t_mrj)
        non_ag_obj_rk = non_ag_r_rk - (non_ag_c_rk + non_ag_t_rk + ag_to_non_ag_t_rk)

        # Get effects of alternative agr. management options (stored in a dict)
        ag_man_objs = {
            am: ag_man_r_mrj[am] - (ag_man_c_mrj[am] + ag_man_t_mrj[am]) 
            for am in settings.AG_MANAGEMENTS_TO_LAND_USES
        }

    elif settings.OBJECTIVE == "mincost":
        # Pre-calculate sum of production and transition costs
        ag_obj_mrj = ag_c_mrj + ag_t_mrj + non_ag_to_ag_t_mrj
        non_ag_obj_rk = non_ag_c_rk + non_ag_t_rk + ag_to_non_ag_t_rk

        # Store calculations for each agricultural management option in a dict
        ag_man_objs = {
            am: (ag_man_c_mrj[am] + ag_man_t_mrj[am])      
            for am in settings.AG_MANAGEMENTS_TO_LAND_USES
        }

    else:
        raise ValueError("Unknown objective!")

    ag_obj_mrj = np.nan_to_num(ag_obj_mrj)
    non_ag_obj_rk = np.nan_to_num(non_ag_obj_rk)
    ag_man_objs = {am: np.nan_to_num(arr) for am, arr in ag_man_objs.items()}

    return [ag_obj_mrj, non_ag_obj_rk, ag_man_objs]


def get_commodity_prices_BASE_YR(data: Data) -> np.ndarray:
    """
    Get the commodity prices for the target year.
    """
    print('Getting commodity prices...', flush = True)
    return ag_revenue.get_commodity_prices(data)


def get_target_yr_carbon_price(data: Data, target_year: int) -> float:
    return data.CARBON_PRICES[target_year]


def get_BASE_YR_economic_value(data: Data):
    """
    Calculate the economic value of the agricultural sector.
    """
    if data.BASE_YR_economic_value is not None:
        return data.BASE_YR_economic_value
    
    # Get the revenue and cost matrices
    r_mrj = ag_revenue.get_rev_matrices(data, 0)
    c_mrj = ag_cost.get_cost_matrices(data, 0)
    # Calculate the economic value
    if settings.OBJECTIVE == 'maxprofit':
        e_mrj = (r_mrj - c_mrj)
    elif settings.OBJECTIVE == 'mincost':
        e_mrj = c_mrj
    else:
        raise ValueError("Invalid `settings.OBJECTIVE`. Use 'maxprofit' or 'maxcost'.")
    
    data.BASE_YR_economic_value = np.einsum('mrj,mrj->', e_mrj, data.AG_L_MRJ)
    return data.BASE_YR_economic_value

def get_BASE_YR_production_t(data: Data):
    """
    Calculate the production of each commodity in the base year.
    """
    # Get the revenue and cost matrices
    return data.BASE_YR_production_t

def get_BASE_YR_GHG_t(data: Data):
    """
    Calculate the GHG emissions of the agricultural sector.
    """
    if data.BASE_YR_GHG_t is not None:
        return data.BASE_YR_GHG_t
    # Get the GHG matrices
    ag_g_mrj = get_ag_g_mrj(data, 0)
    data.BASE_YR_GHG_t = np.einsum('mrj,mrj->', ag_g_mrj, data.AG_L_MRJ)
    return data.BASE_YR_GHG_t
    
def get_BASE_YR_overall_bio_value(data: Data):
    """
    Calculate the economic value of the agricultural sector.
    """
    if data.BASE_YR_overall_bio_value is not None:
        return data.BASE_YR_overall_bio_value
    # Get the revenue and cost matrices
    ag_b_mrj = ag_biodiversity.get_bio_overall_priority_score_matrices_mrj(data)
    data.BASE_YR_overall_bio_value = np.einsum('mrj,mrj->', ag_b_mrj, data.AG_L_MRJ)
    return data.BASE_YR_overall_bio_value

def get_BASE_YR_GBF2_score(data: Data) -> np.ndarray:
    if settings.BIODIVERSITY_TARGET_GBF_2 == "off":
        return np.empty(0)
    if data.BASE_YR_GBF2_score is not None:
        return data.BASE_YR_GBF2_score
    print('Getting priority degrade area base year score...', flush = True)
    GBF2_ly_r = get_GBF2_priority_degrade_area_r(data)
    GBF2_contr_j = get_ag_biodiv_contr_j(data)
    base_yr_dvar_mrj = data.AG_L_MRJ
    data.BASE_YR_GBF2_score = np.einsum('r,j,mrj->', GBF2_ly_r, GBF2_contr_j, base_yr_dvar_mrj)
    return data.BASE_YR_GBF2_score

def get_BASE_YR_water_ML(data: Data) -> np.ndarray:
    """
    Calculate the water net yield of the agricultural sector.
    """
    if data.BASE_YR_water_ML is not None:
        return data.BASE_YR_water_ML
    # Get the water matrices
    ag_w_mrj = get_ag_w_mrj(data, 0)
    ag_w_index = get_w_region_indices(data)
    
    water_ML = []
    for _,idx in ag_w_index.items():
        water_ML.append(
            np.einsum('mrj, mrj->', ag_w_mrj[:, idx, :], data.AG_L_MRJ[:, idx, :])
        )
    data.BASE_YR_water_ML = np.array(water_ML)
    return data.BASE_YR_water_ML
    

def get_savanna_eligible_r(data: Data) -> np.ndarray:
    return np.where(data.SAVBURN_ELIGIBLE == 1)[0]


def get_priority_degraded_mask_idx(data: Data) -> np.ndarray:
    if settings.BIODIVERSITY_TARGET_GBF_2 == "off":
        return np.empty(0)
    return np.where(data.BIO_PRIORITY_DEGRADED_AREAS_R)[0]


def get_limits(data: Data, yr_cal: int, resale_factors) -> dict[str, Any]:
    """
    Gets the following limits for the solve:
    - Water net yield limits
    - GHG limits
    - Biodiversity limits
    - Regional adoption limits
    """
    print('Getting environmental limits...', flush = True)
    
    limits = {
        'demand': None,
        'water': None,
        'ghg': None,
        'GBF2': None,
        'GBF3': None,
        'GBF4_SNES': None,
        'GBF4_ECNES': None,
        'GBF8': None,
        'ag_regional_adoption': None,
        'non_ag_regional_adoption': None,
    }
    
    if True:    # Always set demand limits
        limits['demand'] = data.D_CY[yr_cal - data.YR_CAL_BASE]
        limits['demand_rescale'] = limits['demand'] / resale_factors['Demand']
    
    if settings.WATER_LIMITS == 'on':
        limits['water'] = data.WATER_YIELD_TARGETS
        limits['water_rescale'] = {k: v / resale_factors['Water'] for k, v in limits['water'].items()}
        
    if settings.GHG_EMISSIONS_LIMITS != 'off':
        limits['ghg'] = data.GHG_TARGETS[yr_cal]
        limits['ghg_rescale'] = limits['ghg'] / resale_factors['GHG']

    if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':
        limits["GBF2"] = data.get_GBF2_target_for_yr_cal(yr_cal)
        limits["GBF2_rescale"] = limits["GBF2"] / resale_factors['GBF2']

    if settings.BIODIVERSITY_TARGET_GBF_3 != 'off':
        limits["GBF3"] = data.get_GBF3_limit_score_inside_LUTO_by_yr(yr_cal)
        limits["GBF3_rescale"] = limits["GBF3"] / resale_factors['GBF3']
        
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES == "on":
        limits["GBF4_SNES"] = data.get_GBF4_SNES_target_inside_LUTO_by_year(yr_cal)
        limits["GBF4_SNES_rescale"] = limits["GBF4_SNES"] / resale_factors['GBF4_SNES']
        
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES == "on":
        limits["GBF4_ECNES"] = data.get_GBF4_ECNES_target_inside_LUTO_by_year(yr_cal)
        limits["GBF4_ECNES_rescale"] = limits["GBF4_ECNES"] / resale_factors['GBF4_ECNES']

    if settings.BIODIVERSITY_TARGET_GBF_8 == "on":
        limits["GBF8"] = data.get_GBF8_target_inside_LUTO_by_yr(yr_cal)
        limits["GBF8_rescale"] = limits["GBF8"] / resale_factors['GBF8']

    if settings.REGIONAL_ADOPTION_CONSTRAINTS == 'on':
        ag_reg_adoption, non_ag_reg_adoption = ag_transition.get_regional_adoption_limits(data, yr_cal)
        limits["ag_regional_adoption"] = ag_reg_adoption
        limits["non_ag_regional_adoption"] = non_ag_reg_adoption

    return limits


def rescale_solver_input_data(arries:list) -> None:
    """
    Rescale the solver input data based on `settings.RESCALE_FACTOR`.
    To resume the data, just multiply the arrays by the returned scale factor.
    
    After rescaling, the arrays will be rescaled to the magnitude (regardless of signs) between 0 and 1e3.
    """

    max_vals = []
    for arr in arries:
        if isinstance(arr, np.ndarray):
            arr = arr.astype(np.float32)
            max_vals.append(max(arr.max(), abs(arr.min())))
        elif isinstance(arr, dict):
            # Assume all dictionaries are {str: np.ndarray}
            max_vals.extend([max(v.max(), abs(v.min())) for v in arr.values()])
    
    scale = (np.max(max_vals) / settings.RESCALE_FACTOR).astype(np.float32)
    
    for arr in arries:
        if isinstance(arr, dict):
            # Update dictionary values in-place
            for k in arr:
                arr[k] /= scale
        elif isinstance(arr, np.ndarray):
            # Arrays are already updated in-place
            arr /= scale

    return scale


def get_input_data(data: Data, base_year: int, target_year: int) -> SolverInputData:
    """
    Using the given Data object, prepare a SolverInputData object for the solver.
    """

    target_index = target_year - data.YR_CAL_BASE
    
    ag_c_mrj = get_ag_c_mrj(data, target_index)
    ag_r_mrj = get_ag_r_mrj(data, target_index)
    ag_t_mrj = get_ag_t_mrj(data, target_index, base_year)
    ag_to_non_ag_t_rk = get_ag_to_non_ag_t_rk(data, target_index, base_year, ag_t_mrj)
    
    non_ag_c_rk = get_non_ag_c_rk(data, ag_c_mrj, data.lumaps[base_year], target_year)
    non_ag_r_rk = get_non_ag_r_rk(data, ag_r_mrj, base_year, target_year)
    non_ag_t_rk = get_non_ag_t_rk(data, base_year)
    non_ag_to_ag_t_mrj = get_non_ag_to_ag_t_mrj(data, base_year, target_index)
    
    ag_man_c_mrj = get_ag_man_c_mrj(data, target_index, ag_c_mrj)
    ag_man_r_mrj = get_ag_man_r_mrj(data, target_index, ag_r_mrj)
    ag_man_t_mrj = get_ag_man_t_mrj(data, target_index, ag_t_mrj)
    
    ag_obj_mrj, non_ag_obj_rk,  ag_man_objs=get_economic_mrj(
        ag_c_mrj,
        ag_r_mrj,
        ag_t_mrj,
        ag_to_non_ag_t_rk,
        non_ag_c_rk,
        non_ag_r_rk,
        non_ag_t_rk,
        non_ag_to_ag_t_mrj,
        ag_man_c_mrj,
        ag_man_r_mrj,
        ag_man_t_mrj
    )
    

    ag_g_mrj = get_ag_g_mrj(data, target_index)
    ag_w_mrj = get_ag_w_mrj(data, target_index)                             
    ag_b_mrj = get_ag_b_mrj(data)
    ag_x_mrj = get_ag_x_mrj(data, base_year)
    ag_q_mrp = get_ag_q_mrp(data, target_index)
    ag_ghg_t_mrj = get_ag_ghg_t_mrj(data, base_year)

    non_ag_g_rk = get_non_ag_g_rk(data, ag_g_mrj, base_year)
    non_ag_w_rk = get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year)
    non_ag_b_rk = get_non_ag_b_rk(data, ag_b_mrj, base_year)
    non_ag_x_rk = get_non_ag_x_rk(data, base_year)
    non_ag_q_crk = get_non_ag_q_crk(data, ag_q_mrp, base_year)
    non_ag_lb_rk = get_non_ag_lb_rk(data, base_year)
    
    ag_man_g_mrj=get_ag_man_g_mrj(data, target_index)
    ag_man_w_mrj=get_ag_man_w_mrj(data, target_index)
    ag_man_b_mrj=get_ag_man_b_mrj(data, target_index, ag_b_mrj)
    ag_man_q_mrp=get_ag_man_q_mrj(data, target_index, ag_q_mrp)
    ag_man_limits=get_ag_man_limits(data, target_index)                            
    ag_man_lb_mrj=get_ag_man_lb_mrj(data, base_year)
    
    water_region_indices=get_w_region_indices(data)
    water_region_names=get_w_region_names(data)
    
    biodiv_contr_ag_j=get_ag_biodiv_contr_j(data)
    biodiv_contr_non_ag_k=get_non_ag_biodiv_impact_k(data)
    biodiv_contr_ag_man=get_ag_man_biodiv_impacts(data, target_year)

    GBF2_raw_priority_degraded_area_r = get_GBF2_priority_degrade_area_r(data)
    GBF3_raw_MVG_area_vr=get_GBF3_MVG_area_vr(data)
    GBF3_names=get_GBF3_major_vegetation_names(data)
    GBF3_ind=get_GBF3_major_indices(data)
    GBF4_SNES_xr=get_GBF4_SNES_xr(data)
    GBF4_SNES_names=get_GBF4_SNES_names(data)
    GBF4_ECNES_xr=get_GBF4_ECNES_xr(data)
    GBF4_ECNES_names=get_GBF4_SNES_names(data)
    GBF8_raw_species_area_sr=get_GBF8_species_area_sr(data, target_year)
    GBF8_species_names=get_GBF8_species_names(data)
    GBF8_species_indices=get_GBF8_indices(data,target_year)

    savanna_eligible_r=get_savanna_eligible_r(data)
    priority_degraded_mask_idx=get_priority_degraded_mask_idx(data)

    
    scale_factors = {
        "Economy":       rescale_solver_input_data([ag_obj_mrj, non_ag_obj_rk, ag_man_objs]),
        "Demand":        rescale_solver_input_data([ag_q_mrp, non_ag_q_crk, ag_man_q_mrp]),
        "Biodiversity":  rescale_solver_input_data([ag_b_mrj, non_ag_b_rk, ag_man_b_mrj]),
        "GHG":(
          rescale_solver_input_data([ag_g_mrj, non_ag_g_rk, ag_man_g_mrj, ag_ghg_t_mrj])
          if settings.GHG_EMISSIONS_LIMITS != 'off' 
          else 1.0  
        ),        
        "Water":(
            rescale_solver_input_data([ag_w_mrj, non_ag_w_rk, ag_man_w_mrj])
            if settings.WATER_LIMITS == 'on'
            else 1.0
        ),         
        "GBF2":(
            rescale_solver_input_data([GBF2_raw_priority_degraded_area_r])
            if settings.BIODIVERSITY_TARGET_GBF_2 != "off"
            else 1.0),
        "GBF3":(
            rescale_solver_input_data([GBF3_raw_MVG_area_vr])
            if settings.BIODIVERSITY_TARGET_GBF_3 != "off"
            else 1.0
        ),
        "GBF4_SNES":(
            rescale_solver_input_data([GBF4_SNES_xr])
            if settings.BIODIVERSITY_TARGET_GBF_4_SNES == "on"
            else 1.0
        ),
        "GBF4_ECNES":(
            rescale_solver_input_data([GBF4_ECNES_xr])
            if settings.BIODIVERSITY_TARGET_GBF_4_ECNES == "on"
            else 1.0
        ),
        "GBF8":(
            rescale_solver_input_data([GBF8_raw_species_area_sr])
            if settings.BIODIVERSITY_TARGET_GBF_8 == "on"
            else 1.0
        ),
    }

    base_yr_prod = {
        "BASE_YR Economy(AUD)":        get_BASE_YR_economic_value(data),
        "BASE_YR Production (t)":      get_BASE_YR_production_t(data),
        "BASE_YR GHG (tCO2e)":         get_BASE_YR_GHG_t(data),
        "BASE_YR Water (ML)":          get_BASE_YR_water_ML(data),
        "BASE_YR Bio quality (score)": get_BASE_YR_overall_bio_value(data),
        "BASE_YR GBF_2 (score)":       get_BASE_YR_GBF2_score(data),
    }

    economic_contr_mrj=(ag_obj_mrj, non_ag_obj_rk,  ag_man_objs)
    economic_BASE_YR_prices=get_commodity_prices_BASE_YR(data)
    economic_target_yr_carbon_price=get_target_yr_carbon_price(data, target_year)
    
    offland_ghg=(
        data.OFF_LAND_GHG_EMISSION_C[target_index] / scale_factors["GHG"] 
        if settings.GHG_EMISSIONS_LIMITS != 'off' 
        else 0.0
    )

    lu2pr_pj=data.LU2PR
    pr2cm_cp=data.PR2CM
    limits=get_limits(data, target_year, scale_factors)
    desc2aglu=data.DESC2AGLU
    real_area=data.REAL_AREA

    land_use_culling.apply_agricultural_land_use_culling(
        ag_x_mrj, ag_c_mrj, ag_t_mrj, ag_r_mrj
    )
 
    return SolverInputData(
        base_year,
        target_year,
        
        ag_g_mrj,
        ag_w_mrj,
        ag_b_mrj,
        ag_x_mrj,
        ag_q_mrp,
        ag_ghg_t_mrj,
        
        non_ag_g_rk,
        non_ag_w_rk,
        non_ag_b_rk,
        non_ag_x_rk,
        non_ag_q_crk,
        non_ag_lb_rk,

        ag_man_g_mrj,
        ag_man_w_mrj,
        ag_man_b_mrj,
        ag_man_q_mrp,
        ag_man_limits,
        ag_man_lb_mrj,
        
        water_region_indices,
        water_region_names,
        
        biodiv_contr_ag_j,
        biodiv_contr_non_ag_k,
        biodiv_contr_ag_man,
        
        GBF2_raw_priority_degraded_area_r,
        GBF3_raw_MVG_area_vr,
        GBF3_names,
        GBF3_ind,
        GBF4_SNES_xr,
        GBF4_SNES_names,
        GBF4_ECNES_xr,
        GBF4_ECNES_names,
        GBF8_raw_species_area_sr,
        GBF8_species_names,
        GBF8_species_indices,
        
        savanna_eligible_r,
        priority_degraded_mask_idx,

        base_yr_prod,
        scale_factors,
        
        economic_contr_mrj,
        economic_BASE_YR_prices,
        economic_target_yr_carbon_price,
        
        offland_ghg,
        
        lu2pr_pj,
        pr2cm_cp,
        limits,
        desc2aglu,
        real_area
    )
