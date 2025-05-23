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

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional
import numpy as np

from luto import settings
from luto.economics import land_use_culling
from luto.settings import AG_MANAGEMENTS, AG_MANAGEMENTS_TO_LAND_USES
from luto.data import Data

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
    demand_c: np.ndarray                                                # The commodity demand of the target year

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

    ag_man_g_mrj: dict                                                  # Agricultural management options' GHG emission effects.
    ag_man_q_mrp: dict                                                  # Agricultural management options' quantity effects.
    ag_man_w_mrj: dict                                                  # Agricultural management options' water yield effects.
    ag_man_b_mrj: dict                                                  # Agricultural management options' biodiversity effects.
    ag_man_limits: dict                                                 # Agricultural management options' adoption limits.
    ag_man_lb_mrj: dict                                                 # Agricultural management options' lower bounds.

    water_yield_RR_BASE_YR: dict                                        # Water yield for the BASE_YR based on historical water yield layers .
    water_yield_outside_study_area: dict[int, float]                    # Water yield from outside LUTO study area -> dict. Key: region.
      
    biodiv_contr_ag_j: np.ndarray                                       # Biodiversity contribution scale from agricultural land uses.
    biodiv_contr_non_ag_k: dict[int, float]                             # Biodiversity contribution scale from non-agricultural land uses.
    biodiv_contr_ag_man: dict[str, dict[int, np.ndarray]]               # Biodiversity contribution scale from agricultural management options.
    
    GBF2_raw_priority_degraded_area_r: np.ndarray                       # Raw areas (GBF2) from priority degrade areas - indexed by cell (r).
    GBF3_raw_MVG_area_vr: np.ndarray                                    # Raw areas (GBF3) from Major vegetation group - indexed by veg. group (v) and cell (r)
    GBF4_snes_xr: np.ndarray                                            # Raw areas (GBF4) Species NES contribution data - indexed by species/ecological community (x) and cell (r).
    GBF4_ecnes_xr: np.ndarray                                           # Raw areas (GBF4) Ecological community NES contribution data - indexed by species/ecological community (x) and cell (r).
    GBF8_raw_species_area_sr: np.ndarray                                # Raw areas (GBF8) Species data - indexed by species (s) and cell (r).

    savanna_eligible_r: np.ndarray                                      # Cells that are eligible for savanna burnining land use.
    hir_eligible_r: np.ndarray                                          # Cells that are eligible for the HIR agricultural management option. 
    priority_degraded_mask_idx: np.ndarray                                # Mask of priority degraded areas - indexed by cell (r).

    economic_contr_mrj: float                                           # base year economic contribution matrix.
    economic_BASE_YR_prices: np.ndarray                                 # base year commodity prices.
    economic_target_yr_carbon_price: float                              # target year carbon price.
    
    base_yr_prod: dict[str, float]                                      # Base year production of each commodity.

    offland_ghg: np.ndarray                                             # GHG emissions from off-land commodities.

    lu2pr_pj: np.ndarray                                                # Conversion matrix: land-use to product(s).
    pr2cm_cp: np.ndarray                                                # Conversion matrix: product(s) to commodity.
    limits: dict                                                        # Targets to use.
    desc2aglu: dict                                                     # Map of agricultural land use descriptions to codes.
    resmult: float                                                      # Resolution factor multiplier from data.RESMULT
    real_area: np.ndarray                                               # Area of each cell, indexed by cell (r)
                
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
            for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items()
            if AG_MANAGEMENTS[am]
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
    
def get_demand_c(data, target_year):
    return data.D_CY[target_year - data.YR_CAL_BASE]
    
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


def get_w_outside_luto(data: Data, yr_cal: int):
    print('Getting water yield from outside LUTO study area...', flush = True)
    return ag_water.get_water_outside_luto_study_area_from_hist_level(data)

def get_w_BASE_YR(data: Data):
    print('Getting water yield for the BASE_YR based on historical water yield layers...', flush = True)
    return ag_water.calc_water_net_yield_BASE_YR(data)


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
    if settings.BIODIVERSTIY_TARGET_GBF_2 != "on":
        return np.empty(0)
    print('Getting priority degrade area matrices...', flush = True)
    output = ag_biodiversity.get_GBF2_bio_priority_degraded_areas_r(data)
    return output

def get_GBF3_MVG_area_vr(data: Data):
    if settings.BIODIVERSTIY_TARGET_GBF_3 != "on":
        return np.empty(0)
    print('Getting agricultural major vegetation groups matrices...', flush = True)
    output = ag_biodiversity.get_GBF3_major_vegetation_matrices_vr(data)
    return output


def get_GBF4_snes_xr(data: Data) -> np.ndarray:
    if settings.BIODIVERSTIY_TARGET_GBF_4_SNES != "on":
        return np.empty(0)
    return ag_biodiversity.get_GBF4_SNES_matrix_sr(data)


def get_GBF4_ecnes_xr(data: Data) -> np.ndarray:
    if settings.BIODIVERSTIY_TARGET_GBF_4_ECNES != "on":
        return np.empty(0)
    return ag_biodiversity.get_GBF4_ECNES_matrix_sr(data)

def get_GBF8_species_area_sr(data: Data, target_year: int) -> np.ndarray:
    if settings.BIODIVERSTIY_TARGET_GBF_8 != "on":
        return np.empty(0)
    print('Getting species conservation cell data...', flush = True)
    return ag_biodiversity.get_GBF8_species_conservation_matrix_sr(data, target_year)


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
    
    ag_t_mrj = ag_transition.get_transition_matrices_from_base_year(
        data, 
        target_index, 
        base_year
    ).astype(np.float32)
    # Transition costs occures if the base year is not the target year
    return ag_t_mrj if (base_year - data.YR_CAL_BASE != target_index) else np.zeros_like(ag_t_mrj).astype(np.float32)


def get_ag_to_non_ag_t_rk(data: Data, target_index, base_year, ag_t_mrj):
    print('Getting agricultural to non-agricultural transition cost matrices...', flush = True)
    non_ag_t_mrj = non_ag_transition.get_from_ag_transition_matrix( 
        data, 
        target_index, 
        base_year, 
        data.lumaps[base_year], 
        data.lmmaps[base_year],
        ag_t_mrj).astype(np.float32)
    # Transition costs occures if the base year is not the target year
    return non_ag_t_mrj if (base_year - data.YR_CAL_BASE != target_index) else np.zeros_like(non_ag_t_mrj).astype(np.float32)


def get_non_ag_to_ag_t_mrj(data: Data, base_year:int, target_index: int):
    print('Getting non-agricultural to agricultural transition cost matrices...', flush = True)
    
    non_ag_to_ag_mrj = non_ag_transition.get_to_ag_transition_matrix(
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


def get_ag_man_g_mrj(data: Data, target_index, ag_g_mrj: np.ndarray):
    print('Getting agricultural management options\' GHG emission effects...', flush = True)
    output = ag_ghg.get_agricultural_management_ghg_matrices(data, ag_g_mrj, target_index)
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
            for am in AG_MANAGEMENTS_TO_LAND_USES
        }

    elif settings.OBJECTIVE == "mincost":
        # Pre-calculate sum of production and transition costs
        ag_obj_mrj = ag_c_mrj + ag_t_mrj + non_ag_to_ag_t_mrj
        non_ag_obj_rk = non_ag_c_rk + non_ag_t_rk + ag_to_non_ag_t_rk

        # Store calculations for each agricultural management option in a dict
        ag_man_objs = {
            am: (ag_man_c_mrj[am] + ag_man_t_mrj[am])      
            for am in AG_MANAGEMENTS_TO_LAND_USES
        }

    else:
        raise ValueError("Unknown objective!")

    ag_obj_mrj = np.nan_to_num(ag_obj_mrj)
    non_ag_obj_rk = np.nan_to_num(non_ag_obj_rk)
    ag_man_objs = {am: np.nan_to_num(arr) for am, arr in ag_man_objs.items()}

    return [ag_obj_mrj, non_ag_obj_rk, ag_man_objs]



def get_commodity_prices(data: Data) -> np.ndarray:
    '''
    Get the prices of commodities in the base year. These prices will be used as multiplier
    to weight deviatios of commodity production from the target.
    '''
    
    commodity_lookup = {
        ('P1','BEEF'): 'beef meat',
        ('P3','BEEF'): 'beef lexp',
        ('P1','SHEEP'): 'sheep meat',
        ('P2','SHEEP'): 'sheep wool',
        ('P3','SHEEP'): 'sheep lexp',
        ('P1','DAIRY'): 'dairy',
    }

    commodity_prices = {}

    # Get the median price of each commodity
    for names, commodity in commodity_lookup.items():
        prices = np.nanpercentile(data.AGEC_LVSTK[names[0], names[1]], 50)
        prices = prices * 1000 if commodity == 'dairy' else prices # convert to per tonne for dairy
        commodity_prices[commodity] = prices

    # Get the median price of each crop; here need to use 'irr' because dry-Rice does exist in the data
    for name, col in data.AGEC_CROPS['P1','irr'].items():
        commodity_prices[name.lower()] = np.nanpercentile(col, 50)

    return np.array([commodity_prices[k] for k in data.COMMODITIES])
    
    
def get_target_yr_carbon_price(data: Data, target_year: int) -> float:
    return data.CARBON_PRICES[target_year]


def get_BASE_YR_economic_value(data: Data):
    """
    Calculate the economic value of the agricultural sector.
    """
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
    return np.einsum('mrj,mrj->', e_mrj, data.AG_L_MRJ)
    
    
def get_BASE_YR_biodiv_value(data: Data):
    """
    Calculate the economic value of the agricultural sector.
    """
    # Get the revenue and cost matrices
    ag_b_mrj = ag_biodiversity.get_bio_overall_priority_score_matrices_mrj(data)
    return np.einsum('mrj,mrj->', ag_b_mrj, data.AG_L_MRJ)
    
def get_BASE_YR_production_t(data: Data):
    """
    Calculate the production of each commodity in the base year.
    """
    # Get the revenue and cost matrices
    return data.get_production(data.YR_CAL_BASE, data.LUMAP, data.LMMAP).sum()


def get_savanna_eligible_r(data: Data) -> np.ndarray:
    return np.where(data.SAVBURN_ELIGIBLE == 1)[0]

def get_hir_eligible_r(data: Data) -> np.ndarray:
    return np.where(data.HIR_MASK == 1)[0]

def get_priority_degraded_mask_idx(data: Data) -> np.ndarray:
    return np.where(data.BIO_PRIORITY_DEGRADED_AREAS_MASK)[0]


def get_limits(
    data: Data, yr_cal: int,
) -> dict[str, Any]:
    """
    Gets the following limits for the solve:
    - Water net yield limits
    - GHG limits
    - Biodiversity limits
    - Regional adoption limits
    """
    print('Getting environmental limits...', flush = True)
    # Limits is a dictionary with heterogeneous value sets.
    limits = {}

    limits['water'] = ag_water.get_water_net_yield_limit_values(data)

    if settings.GHG_EMISSIONS_LIMITS == 'on':
        limits['ghg_ub'] = ag_ghg.get_ghg_limits(data, yr_cal)

    # If biodiversity limits are not turned on, set the limit to 0.
    limits["GBF2_priority_degrade_areas"] = (
        ag_biodiversity.get_GBF2_biodiversity_limits(data, yr_cal)
        if settings.BIODIVERSTIY_TARGET_GBF_2 == 'on'
        else 0
    )

    limits["GBF_3_major_vegetation_groups"] = (
        ag_biodiversity.get_GBF3_major_vegetation_group_limits(data, yr_cal)
        if settings.BIODIVERSTIY_TARGET_GBF_3 == 'on'
        else 0
    )

    limits["GBF4_SNES"] = ag_biodiversity.get_GBF4_SNES_limits(data, yr_cal)
    limits["GBF4_ECNES"] = ag_biodiversity.get_GBF4_ECNES_limits(data, yr_cal)
    limits["GBF8_species_conservation"] = ag_biodiversity.get_GBF8_species_conservation_limits(data, yr_cal)

    ag_reg_adoption, non_ag_reg_adoption = ag_transition.get_regional_adoption_limits(data, yr_cal)
    limits["ag_regional_adoption"] = ag_reg_adoption
    limits["non_ag_regional_adoption"] = non_ag_reg_adoption

    return limits


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
    ag_q_mrp = get_ag_q_mrp(data, target_index)
    ag_w_mrj = get_ag_w_mrj(data, target_index, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)     # Calculate water net yield matrices based on historical water yield layers
    ag_b_mrj = get_ag_b_mrj(data)
    ag_x_mrj = get_ag_x_mrj(data, base_year)

    land_use_culling.apply_agricultural_land_use_culling(
        ag_x_mrj, ag_c_mrj, ag_t_mrj, ag_r_mrj
    )

    return SolverInputData(
        base_year=base_year,
        target_year=target_year,
        demand_c=get_demand_c(data, target_year),
        
        ag_g_mrj=ag_g_mrj,
        ag_w_mrj=ag_w_mrj,
        ag_b_mrj=ag_b_mrj,
        ag_x_mrj=ag_x_mrj,
        ag_q_mrp=ag_q_mrp,
        ag_ghg_t_mrj=get_ag_ghg_t_mrj(data, base_year),

        non_ag_g_rk=get_non_ag_g_rk(data, ag_g_mrj, base_year),
        non_ag_w_rk=get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR),  # Calculate non-ag water yield matrices based on historical water yield layers
        non_ag_b_rk=get_non_ag_b_rk(data, ag_b_mrj, base_year),
        non_ag_x_rk=get_non_ag_x_rk(data, base_year),
        non_ag_q_crk=get_non_ag_q_crk(data, ag_q_mrp, base_year),
        non_ag_lb_rk=get_non_ag_lb_rk(data, base_year),
        
        ag_man_g_mrj=get_ag_man_g_mrj(data, target_index, ag_g_mrj),
        ag_man_q_mrp=get_ag_man_q_mrj(data, target_index, ag_q_mrp),
        ag_man_w_mrj=get_ag_man_w_mrj(data, target_index),
        ag_man_b_mrj=get_ag_man_b_mrj(data, target_index, ag_b_mrj),
        ag_man_limits=get_ag_man_limits(data, target_index),                            
        ag_man_lb_mrj=get_ag_man_lb_mrj(data, base_year),
        
        water_yield_outside_study_area=get_w_outside_luto(data, data.YR_CAL_BASE),      # Use the water net yield outside LUTO study area for the YR_CAL_BASE year
        water_yield_RR_BASE_YR=get_w_BASE_YR(data),                                     # Calculate water net yield for the BASE_YR (2010) based on historical water yield layers
        
        biodiv_contr_ag_j=get_ag_biodiv_contr_j(data),
        biodiv_contr_non_ag_k=get_non_ag_biodiv_impact_k(data),
        biodiv_contr_ag_man=get_ag_man_biodiv_impacts(data, target_year),

        GBF2_raw_priority_degraded_area_r = get_GBF2_priority_degrade_area_r(data),
        GBF3_raw_MVG_area_vr=get_GBF3_MVG_area_vr(data),
        GBF4_snes_xr=get_GBF4_snes_xr(data),
        GBF4_ecnes_xr=get_GBF4_ecnes_xr(data),
        GBF8_raw_species_area_sr=get_GBF8_species_area_sr(data, target_year),

        savanna_eligible_r=get_savanna_eligible_r(data),
        hir_eligible_r=get_hir_eligible_r(data),
        priority_degraded_mask_idx=get_priority_degraded_mask_idx(data),

        economic_contr_mrj=(ag_obj_mrj, non_ag_obj_rk,  ag_man_objs),
        economic_BASE_YR_prices=get_commodity_prices(data),
        economic_target_yr_carbon_price=get_target_yr_carbon_price(data, target_year), 
        
        base_yr_prod = {
            "BASE_YR Economy(AUD)": get_BASE_YR_economic_value(data),
            "BASE_YR Biodiversity (score)": get_BASE_YR_biodiv_value(data),
            "BASE_YR Production (t)": get_BASE_YR_production_t(data),
        },
        
        offland_ghg=data.OFF_LAND_GHG_EMISSION_C[target_index],

        lu2pr_pj=data.LU2PR,
        pr2cm_cp=data.PR2CM,
        limits=get_limits(data, target_year),
        desc2aglu=data.DESC2AGLU,
        resmult=data.RESMULT,
        real_area=data.REAL_AREA,
    )
