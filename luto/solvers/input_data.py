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
import xarray as xr

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
    ag_b_mrj: np.ndarray                                                # Agricultural biodiversity matrices based on bio-quality layer.
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
    
    renewable_solar_r: np.ndarray                                       # Renewable energy - solar yield matrix.
    renewable_wind_r: np.ndarray                                        # Renewable energy - wind yield matrix.
    exist_renewable_solar_r: np.ndarray                                 # Existing solar capacity converted to annual MWh per cell.
    exist_renewable_wind_r: np.ndarray                                  # Existing wind capacity converted to annual MWh per cell.
    
    region_state_r: np.ndarray                                          # Region state index for each cell.
    region_state_name2idx: dict[str, int]                               # Map of region state names to indices.
    region_NRM_names_r: np.ndarray                                      # Region NRM names for each cell.
    
    water_region_indices: dict[int, np.ndarray]                         # Water region indices -> dict. Key: region.
    water_region_names: dict[int, str]                                  # Water yield for the BASE_YR based on historical water yield layers.
      
    biodiv_contr_ag_j: np.ndarray                                       # Biodiversity contribution scale from agricultural land uses.
    biodiv_contr_non_ag_k: dict[int, float]                             # Biodiversity contribution scale from non-agricultural land uses.
    biodiv_contr_ag_man: dict[str, dict[int, np.ndarray]]               # Biodiversity contribution scale from agricultural management options.
    
    GBF2_mask_area_r: np.ndarray                                        # Raw areas (GBF2) from priority degrade areas - indexed by cell (r).
    GBF3_NVIS_pre_1750_area_vr: np.ndarray                              # Raw areas (GBF3) from NVIS vegetation - indexed by group (v) and cell (r)
    GBF3_NVIS_region_group: dict[int, str]                                     # GBF3 NVIS vegetation group names - indexed by group (v).
    GBF4_SNES_pre_1750_area_sr: xr.DataArray                            # Areas (GBF4) SNES - xr.DataArray[species, cell], species coord = unique species names.
    GBF4_SNES_region_species: list                                      # GBF4 SNES constraint pairs - list[(region, species)].
    GBF4_ECNES_pre_1750_area_sr: xr.DataArray                          # Areas (GBF4) ECNES - xr.DataArray[species, cell], species coord = unique community names.
    GBF4_ECNES_region_species: list                                     # GBF4 ECNES constraint pairs - list[(region, species)].
    GBF8_pre_1750_area_sr: xr.DataArray                                 # Areas (GBF8) - xr.DataArray[species, cell], species coord = species name strings.
    GBF8_region_species: list                                           # GBF8 constraint pairs - list[(region, species)].

    savanna_eligible_r: np.ndarray                                      # Cells that are eligible for savanna burnining land use.
    GBF2_mask_idx: np.ndarray                                           # Index of the mask of priority degraded areas.
    renewable_GBF2_mask_solar_idx: np.ndarray                            # Index of GBF2 mask for solar renewable exclusion.
    renewable_GBF2_mask_wind_idx: np.ndarray                             # Index of GBF2 mask for wind renewable exclusion.
    renewable_MNES_mask_solar_idx: np.ndarray                           # Index of EPBC MNES mask for solar renewable exclusion.
    renewable_MNES_mask_wind_idx: np.ndarray                            # Index of EPBC MNES mask for wind renewable exclusion.

    base_yr_prod: dict[str, tuple]                                      # Base year production of each commodity.
    scale_factors: dict[float]                                          # Scale factors for each input layer.

    economic_contr_mrj: float                                           # base year economic contribution matrix.
    economic_prices: np.ndarray                                         # base year commodity prices.
    economic_target_yr_carbon_price: float                              # target year carbon price.
    
    offland_ghg: np.ndarray                                             # GHG emissions from off-land commodities.

    lu2pr_pj: np.ndarray                                                # Conversion matrix: land-use to product(s).
    pr2cm_cp: np.ndarray                                                # Conversion matrix: product(s) to commodity.
    limits: dict                                                        # Targets to use.
    desc2aglu: dict                                                     # Map of agricultural land use descriptions to codes.
    real_area: np.ndarray                                               # Area of each cell, indexed by cell (r)
    ag_mask_proportion_r: np.ndarray                                    # Initial (2010) total agricultural land proportion per cell (r).
                
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
        # Number of Agricultural land-uses
        return self.ag_g_mrj.shape[2]

    @property
    def n_non_ag_lus(self):
        # Number of Non-Agricultural Land-uses
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
    output = ag_biodiversity.get_bio_quality_score_mrj(data)
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

def get_GBF2_mask_area_r(data: Data) -> np.ndarray:
    if settings.BIODIVERSITY_TARGET_GBF_2 == "off":
        return np.empty(0)
    print('Getting GBF2 mask area layer...', flush = True)
    output = ag_biodiversity.get_GBF2_MASK_area(data)
    return output

def get_GBF3_NVIS_pre_1750_area_vr(data: Data):
    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS == "off":
        return np.empty(0)
    print('Getting GBF3 NVIS vegetation matrices...', flush = True)
    output = ag_biodiversity.get_GBF3_NVIS_matrices_vr(data)
    return output

def get_GBF3_NVIS_region_group(data: Data) -> dict[int,str]:
    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS == "off":
        return {}
    print('Getting GBF3 NVIS vegetation group names...', flush = True)
    return data.BIO_GBF3_NVIS_SEL


def get_GBF4_SNES_pre_1750_area_sr(data: Data) -> xr.DataArray:
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES != "on":
        return np.empty(0)
    print('Getting GBF4 SNES species area matrices...', flush=True)
    return ag_biodiversity.get_GBF4_SNES_matrix_sr(data)

def get_GBF4_SNES_region_species(data: Data) -> list:
    if settings.BIODIVERSITY_TARGET_GBF_4_SNES != "on":
        return []
    print('Getting GBF4 SNES (region, species) constraint pairs...', flush=True)
    return data.BIO_GBF4_SNES_SEL

def get_GBF4_ECNES_pre_1750_area_sr(data: Data) -> xr.DataArray:
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES != "on":
        return np.empty(0)
    print('Getting GBF4 ECNES community area matrices...', flush=True)
    return ag_biodiversity.get_GBF4_ECNES_matrix_sr(data)

def get_GBF4_ECNES_region_species(data: Data) -> list:
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES != "on":
        return []
    print('Getting GBF4 ECNES (region, species) constraint pairs...', flush=True)
    return data.BIO_GBF4_ECNES_SEL

def get_GBF8_pre_1750_area_sr(data: Data, target_year: int) -> xr.DataArray:
    if settings.BIODIVERSITY_TARGET_GBF_8 != "on":
        return np.empty(0)
    print('Getting GBF8 species conservation area matrices...', flush=True)
    return ag_biodiversity.get_GBF8_matrix_sr(data, target_year)

def get_GBF8_region_species(data: Data) -> list:
    if settings.BIODIVERSITY_TARGET_GBF_8 != "on":
        return []
    print('Getting GBF8 (region, species) constraint pairs...', flush=True)
    return data.BIO_GBF8_SEL


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
    output = non_ag_transition.get_non_ag_to_non_ag_transition_matrix(data)
    return output


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

def get_potential_renewable_solar_r(data: Data, target_idx):
    print('Getting renewable energy - solar yield matrix...', flush = True)
    output = ag_quantity.get_quantity_renewable(data, 'Utility Solar PV', target_idx)
    return output

def get_potential_renewable_wind_r(data: Data, target_idx):
    print('Getting renewable energy - wind yield matrix...', flush = True)
    output = ag_quantity.get_quantity_renewable(data, 'Onshore Wind', target_idx)
    return output

def get_exist_renewable_fraction_solar_r(data: Data, yr_cal: int):
    print('Getting existing solar capacity (MWh/cell)...', flush=True)
    return ag_quantity.get_existing_renewable_dvar_fraction(data, 'Utility Solar PV', yr_cal)

def get_exist_renewable_fraction_wind_r(data: Data, yr_cal: int):
    print('Getting existing wind capacity (MWh/cell)...', flush=True)
    return ag_quantity.get_existing_renewable_dvar_fraction(data, 'Onshore Wind', yr_cal)

def get_exist_renewable_capacity_by_state_input(data: Data, yr_cal: int):
    print('Getting existing renewable capacity by state...', flush=True)
    return ag_quantity.get_exist_renewable_capacity_by_state(data, yr_cal)

def get_region_state_r(data: Data):
    print('Getting region state index for each cell...', flush = True)
    return data.REGION_STATE_CODE

def get_region_state_name2idx(data: Data):
    print('Getting map of region state names to indices...', flush = True)
    return data.REGION_STATE_NAME2CODE

def get_region_NRM_names_r(data: Data):
    print('Getting region NRM names for each cell...', flush = True)
    return data.REGION_NRM_NAME

def get_non_ag_lb_rk(data: Data, base_year):
    print('Getting non-agricultural lower bound matrices...', flush = True)
    output = non_ag_transition.get_lower_bound_non_agricultural_matrices(data, base_year)
    return output


def get_ag_man_c_mrj(data: Data, ag_c_mrj: np.ndarray, target_year):
    print('Getting agricultural management options\' cost effects...', flush = True)
    output = ag_cost.get_agricultural_management_cost_matrices(data, ag_c_mrj, target_year)
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


def get_ag_man_t_mrj(data: Data, target_index):
    print('Getting agricultural management options\' transition cost effects...', flush = True)
    output = ag_transition.get_agricultural_management_transition_matrices(data, target_index)
    return output


def get_ag_man_w_mrj(data: Data, target_index):
    print('Getting agricultural management options\' water yield effects...', flush = True)
    output = ag_water.get_agricultural_management_water_matrices(data, target_index)
    return output


def get_ag_man_b_mrj(data: Data, target_index, ag_b_mrj: np.ndarray):
    print('Getting agricultural management options\' biodiversity effects...', flush = True)
    output = ag_biodiversity.get_ag_mgt_biodiversity_matrices(data, ag_b_mrj, target_index)
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


def get_commodity_prices_target_yr(data: Data, yr_cal) -> np.ndarray:
    """
    Get the commodity prices for the target year.
    """
    print('Getting commodity prices...', flush = True)
    return ag_revenue.get_commodity_prices(data, yr_cal)


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
    
def get_BASE_YR_bio_quality_value(data: Data):
    """
    Calculate the economic value of the agricultural sector.
    """
    if data.BASE_YR_overall_bio_value is not None:
        return data.BASE_YR_overall_bio_value
    # Get the revenue and cost matrices
    ag_b_mrj = ag_biodiversity.get_bio_quality_score_mrj(data)
    data.BASE_YR_overall_bio_value = np.einsum('mrj,mrj->', ag_b_mrj, data.AG_L_MRJ)
    return data.BASE_YR_overall_bio_value

def get_BASE_YR_GBF2_score(data: Data) -> np.ndarray:
    if settings.BIODIVERSITY_TARGET_GBF_2 == "off":
        return np.empty(0)
    if data.BASE_YR_GBF2_score is not None:
        return data.BASE_YR_GBF2_score
    print('Getting priority degrade area base year score...', flush = True)
    data.BASE_YR_GBF2_score = data.BIO_GBF2_BASE_YR.sum()

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


def get_GBF2_mask_idx(data: Data) -> np.ndarray:
    if settings.BIODIVERSITY_TARGET_GBF_2 == "off":
        return np.empty(0)
    return np.where(data.BIO_GBF2_MASK_LDS)[0]


def get_renewable_GBF2_mask_solar_idx(data: Data) -> np.ndarray:
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_GBF2_MASK_SOLAR)[0]


def get_renewable_GBF2_mask_wind_idx(data: Data) -> np.ndarray:
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_GBF2_MASK_WIND)[0]


def get_renewable_MNES_mask_solar_idx(data: Data) -> np.ndarray:
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_MNES_MASK_SOLAR)[0]


def get_renewable_MNES_mask_wind_idx(data: Data) -> np.ndarray:
    if not any(settings.RENEWABLES_OPTIONS.values()) or not settings.EXCLUDE_RENEWABLES_IN_EPBC_MNES_MASK:
        return np.empty(0, dtype=int)
    return np.where(data.RENEWABLE_MNES_MASK_WIND)[0]


# Coefficients smaller than `settings.RESCALE_ZERO_THRESHOLD` (after rescaling to
# max == RESCALE_FACTOR == 1e3 by default) are zeroed out before being shipped to Gurobi.
# They contribute nothing to constraint sums dominated by O(RESCALE_FACTOR) entries, but
# they wreck barrier numerics by stretching the matrix coefficient range below Gurobi's
# recommended [1e-3, 1e6] band — the symptom is "Numerical trouble encountered" with a
# Matrix range like [5e-08, 1e+03].


def rescale_per_species_data(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rescale a 2D species×cell matrix per-species (per-row) independently.

    Each row is scaled so its `settings.RESCALE_PERCENTILE`-th percentile of |values| equals
    `settings.RESCALE_FACTOR` (default 1e3). Using a percentile (rather than the absolute max)
    is robust to outlier cells — a single cell with extreme presence no longer compresses the
    typical bulk of cells into a tiny range. Entries above the percentile end up above
    RESCALE_FACTOR after scaling (still within Gurobi's recommended [1e-3, 1e6] band).

    After rescaling, entries with |value| < `settings.RESCALE_ZERO_THRESHOLD` are zeroed out to
    keep the matrix coefficient range inside Gurobi's recommended band.

    Returns:
        scaled_arr: The rescaled 2D array (float32).
        scale_factors: 1D array of per-species scale factors. If `arr` is an xr.DataArray,
                       the returned `scale_factors` is an xr.DataArray that retains the row
                       coord (e.g. 'species' or 'group') so callers can do `.sel(...)`.
                       Multiply rescaled values by scale_factors[s] to recover original values.
    """
    arr_np = np.asarray(arr)
    abs_arr = np.abs(arr_np)
    row_refs = np.percentile(abs_arr, settings.RESCALE_PERCENTILE, axis=1)  # per-species reference
    # Avoid division by zero for rows whose percentile reference is 0 (e.g. very sparse rows).
    # Fall back to the row max; if that is also 0 use RESCALE_FACTOR (no-op scaling).
    zero_ref = row_refs == 0
    if zero_ref.any():
        row_max = abs_arr.max(axis=1)
        row_refs = np.where(zero_ref, np.where(row_max == 0, settings.RESCALE_FACTOR, row_max), row_refs)
    scale_factors_np = (row_refs / settings.RESCALE_FACTOR).astype(np.float32)
    scaled_np = (arr_np / scale_factors_np[:, np.newaxis]).astype(np.float32)
    scaled_np[np.abs(scaled_np) < settings.RESCALE_ZERO_THRESHOLD] = 0.0

    # Preserve the xarray wrapper (dims + coords) when the input is an xr.DataArray, so the
    # solver can still do scaled_arr.sel(group=...) and scale_factors.sel(species=...).
    if isinstance(arr, xr.DataArray):
        row_dim = arr.dims[0]
        scaled_arr = xr.DataArray(scaled_np, dims=arr.dims, coords=arr.coords, name=arr.name)
        scale_factors = xr.DataArray(
            scale_factors_np,
            dims=[row_dim],
            coords={row_dim: arr.coords[row_dim]} if row_dim in arr.coords else None,
        )
    else:
        scaled_arr = scaled_np
        scale_factors = scale_factors_np

    return scaled_arr, scale_factors


def rescale_solver_input_data(arries: list) -> tuple[list, float]:
    """
    Rescale the solver input data based on `settings.RESCALE_FACTOR`.
    Returns scaled copies (non-in-place) and the scale factor used.

    The scale factor is derived from the `settings.RESCALE_PERCENTILE`-th percentile of |values|
    pooled across all input arrays (rather than the absolute max), making it robust to outlier
    entries. Entries with |value| < `settings.RESCALE_ZERO_THRESHOLD` after rescaling are zeroed
    out to keep the matrix coefficient range inside Gurobi's recommended [1e-3, 1e6] band.
    To recover original values, multiply the scaled arrays by the returned scale factor.
    """

    # Pool all |values| across the inputs and take a single global percentile so every array
    # in this group shares one scale factor (preserves relative magnitudes between e.g. ag,
    # non-ag, and ag-management variants of the same quantity).
    flat_pieces = []
    for arr in arries:
        if isinstance(arr, np.ndarray):
            flat_pieces.append(np.abs(arr).ravel())
        elif isinstance(arr, dict):
            flat_pieces.extend(np.abs(v).ravel() for v in arr.values())

    if flat_pieces:
        all_abs = np.concatenate(flat_pieces)
        # Drop exact zeros so the percentile reflects the active coefficient distribution
        nonzero = all_abs[all_abs > 0]
        if nonzero.size > 0:
            ref = np.percentile(nonzero, settings.RESCALE_PERCENTILE)
        else:
            ref = settings.RESCALE_FACTOR  # all-zero input → no-op scaling
    else:
        ref = settings.RESCALE_FACTOR

    scale = (ref / settings.RESCALE_FACTOR).astype(np.float32)

    def _scale_and_threshold(v: np.ndarray) -> np.ndarray:
        out = (v / scale).astype(np.float32)
        out[np.abs(out) < settings.RESCALE_ZERO_THRESHOLD] = 0.0
        return out

    scaled = []
    for arr in arries:
        if isinstance(arr, np.ndarray):
            scaled.append(_scale_and_threshold(arr))
        elif isinstance(arr, dict):
            scaled.append({k: _scale_and_threshold(v) for k, v in arr.items()})
        else:
            scaled.append(arr)

    return scaled, scale

def get_limits(data: Data, yr_cal: int, resale_factors) -> dict[str, Any]:
    """
    Gets the following limits for the solve:
    - Water net yield limits
    - GHG limits
    - Biodiversity limits
    - Regional adoption limits
    """
    print('Getting environmental limits...', flush = True)
    
    limits = {}
    
    if True:    # Always set demand limits
        limits['demand'] = data.D_CY[yr_cal - data.YR_CAL_BASE]
        limits['demand_rescale'] = limits['demand'] / resale_factors['Demand']
    
    if settings.WATER_LIMITS == 'on':
        limits['water'] = data.WATER_YIELD_TARGETS
        limits['water_rescale'] = {k: v / resale_factors['Water'] for k, v in limits['water'].items()}
        
    if settings.GHG_EMISSIONS_LIMITS != 'off':
        limits['ghg'] = data.GHG_TARGETS[yr_cal]
        limits['ghg_rescale'] = limits['ghg'] / resale_factors['GHG']
        
    if any(settings.RENEWABLES_OPTIONS.values()):
        
        renewable_targets = data.RENEWABLE_TARGETS.query('Year == @yr_cal').set_index('state')
        limits['renewable_Utility Solar PV'] = renewable_targets.query('tech == "Utility Solar"')['Renewable_Target_MWh'].to_dict()
        limits['renewable_Onshore Wind'] = renewable_targets.query('tech == "Wind"')['Renewable_Target_MWh'].to_dict()
        limits['renewable_Utility Solar PV_rescale'] = {k: v / resale_factors['Renewable_Solar'] for k, v in limits['renewable_Utility Solar PV'].items()}
        limits['renewable_Onshore Wind_rescale'] = {k: v / resale_factors['Renewable_Wind'] for k, v in limits['renewable_Onshore Wind'].items()}
        
        renewable_existing_capacity = get_exist_renewable_capacity_by_state_input(data, yr_cal)
        limits['renewable_Utility Solar PV_exist'] = {state: vals['Utility Solar PV'] for state, vals in renewable_existing_capacity.items()}
        limits['renewable_Onshore Wind_exist']     = {state: vals['Onshore Wind']     for state, vals in renewable_existing_capacity.items()}
        limits['renewable_Utility Solar PV_exist_rescale'] = {k: v / resale_factors['Renewable_Solar'] for k, v in limits['renewable_Utility Solar PV_exist'].items()}
        limits['renewable_Onshore Wind_exist_rescale'] = {k: v / resale_factors['Renewable_Wind'] for k, v in limits['renewable_Onshore Wind_exist'].items()}

    if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':
        limits["GBF2"] = data.get_GBF2_target_for_yr_cal(yr_cal)
        limits["GBF2_rescale"] = limits["GBF2"] / resale_factors['GBF2']

    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS != 'off':
        limits["GBF3_NVIS"] = data.get_GBF3_NVIS_limit_score_inside_LUTO_by_yr(yr_cal)

    if settings.BIODIVERSITY_TARGET_GBF_4_SNES == "on":
        limits["GBF4_SNES"] = data.get_GBF4_SNES_target_inside_LUTO_by_year(yr_cal)

    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES == "on":
        limits["GBF4_ECNES"] = data.get_GBF4_ECNES_target_inside_LUTO_by_year(yr_cal)

    if settings.BIODIVERSITY_TARGET_GBF_8 == "on":
        limits["GBF8"] = data.get_GBF8_target_inside_LUTO_by_yr(yr_cal)

    if settings.REGIONAL_ADOPTION_CONSTRAINTS != 'off':
        ag_reg_adoption, non_ag_reg_adoption, non_ag_reg_adoption_sum = ag_transition.get_regional_adoption_limits(data, yr_cal)
        limits["ag_regional_adoption"] = ag_reg_adoption
        limits["non_ag_regional_adoption"] = non_ag_reg_adoption
        limits["non_ag_regional_adoption_sum"] = non_ag_reg_adoption_sum

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
    
    ag_man_c_mrj = get_ag_man_c_mrj(data, ag_c_mrj, target_year)
    ag_man_r_mrj = get_ag_man_r_mrj(data, target_index, ag_r_mrj)
    ag_man_t_mrj = get_ag_man_t_mrj(data, target_index)
    
    ag_obj_mrj, non_ag_obj_rk,  ag_man_objs = get_economic_mrj(
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
    ag_w_mrj = (
        get_ag_w_mrj(data, target_index) if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on' 
        else get_ag_w_mrj(data, target_index, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    )
    ag_b_mrj = get_ag_b_mrj(data)
    ag_x_mrj = get_ag_x_mrj(data, base_year)
    ag_q_mrp = get_ag_q_mrp(data, target_index)
    ag_ghg_t_mrj = get_ag_ghg_t_mrj(data, base_year)

    non_ag_g_rk = get_non_ag_g_rk(data, ag_g_mrj, base_year)
    non_ag_w_rk = (
        get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year)   
        if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on' 
        else get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    )
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
    
    renewable_solar_r=get_potential_renewable_solar_r(data, target_index)
    renewable_wind_r=get_potential_renewable_wind_r(data, target_index)
    exist_renewable_solar_r=get_exist_renewable_fraction_solar_r(data, target_year)
    exist_renewable_wind_r=get_exist_renewable_fraction_wind_r(data, target_year)

    region_state_r = get_region_state_r(data)
    region_state_name2idx = get_region_state_name2idx(data)
    region_NRM_names_r=get_region_NRM_names_r(data)
    
    water_region_indices=get_w_region_indices(data)
    water_region_names=get_w_region_names(data)
    
    biodiv_contr_ag_j=get_ag_biodiv_contr_j(data)
    biodiv_contr_non_ag_k=get_non_ag_biodiv_impact_k(data)
    biodiv_contr_ag_man=get_ag_man_biodiv_impacts(data, target_year)

    GBF2_mask_area_r = get_GBF2_mask_area_r(data)
    GBF3_NVIS_pre_1750_area_vr=get_GBF3_NVIS_pre_1750_area_vr(data)
    GBF3_NVIS_region_group=get_GBF3_NVIS_region_group(data)
    GBF4_SNES_pre_1750_area_sr=get_GBF4_SNES_pre_1750_area_sr(data)
    GBF4_SNES_region_species=get_GBF4_SNES_region_species(data)
    GBF4_ECNES_pre_1750_area_sr=get_GBF4_ECNES_pre_1750_area_sr(data)
    GBF4_ECNES_region_species=get_GBF4_ECNES_region_species(data)
    GBF8_pre_1750_area_sr=get_GBF8_pre_1750_area_sr(data, target_year)
    GBF8_region_species=get_GBF8_region_species(data)

    savanna_eligible_r=get_savanna_eligible_r(data)
    GBF2_mask_idx=get_GBF2_mask_idx(data)
    renewable_GBF2_mask_solar_idx=get_renewable_GBF2_mask_solar_idx(data)
    renewable_GBF2_mask_wind_idx=get_renewable_GBF2_mask_wind_idx(data)
    renewable_MNES_mask_solar_idx=get_renewable_MNES_mask_solar_idx(data)
    renewable_MNES_mask_wind_idx=get_renewable_MNES_mask_wind_idx(data)

    # Rescale solver input data
    [ag_obj_mrj, non_ag_obj_rk, ag_man_objs], economy_scale = rescale_solver_input_data([ag_obj_mrj, non_ag_obj_rk, ag_man_objs])
    [ag_q_mrp, non_ag_q_crk, ag_man_q_mrp],   demand_scale  = rescale_solver_input_data([ag_q_mrp, non_ag_q_crk, ag_man_q_mrp])
    [ag_b_mrj, non_ag_b_rk, ag_man_b_mrj],    biodiv_scale  = rescale_solver_input_data([ag_b_mrj, non_ag_b_rk, ag_man_b_mrj])

    # Get the scale factors
    if settings.GHG_EMISSIONS_LIMITS != 'off':
        [ag_g_mrj, non_ag_g_rk, ag_man_g_mrj, ag_ghg_t_mrj], ghg_scale = rescale_solver_input_data([ag_g_mrj, non_ag_g_rk, ag_man_g_mrj, ag_ghg_t_mrj])
    else:
        ghg_scale = 1.0
  
    if any(settings.RENEWABLES_OPTIONS.values()):
            [renewable_solar_r], renewable_solar_scale = rescale_solver_input_data([renewable_solar_r])
            [renewable_wind_r],  renewable_wind_scale  = rescale_solver_input_data([renewable_wind_r])
    else:
        renewable_solar_scale = 1.0
        renewable_wind_scale  = 1.0

    if settings.WATER_LIMITS == 'on':
        [ag_w_mrj, non_ag_w_rk, ag_man_w_mrj], water_scale = rescale_solver_input_data([ag_w_mrj, non_ag_w_rk, ag_man_w_mrj])
    else:
        water_scale = 1.0

    if settings.BIODIVERSITY_TARGET_GBF_2 != "off":
        [GBF2_mask_area_r], gbf2_scale = rescale_solver_input_data([GBF2_mask_area_r])
    else:
        gbf2_scale = 1.0

    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS != "off":
        GBF3_NVIS_pre_1750_area_vr, gbf3_nvis_scale = rescale_per_species_data(GBF3_NVIS_pre_1750_area_vr)
    else:
        gbf3_nvis_scale = 1.0


    if settings.BIODIVERSITY_TARGET_GBF_4_SNES == "on":
        GBF4_SNES_pre_1750_area_sr, gbf4_snes_scale = rescale_per_species_data(GBF4_SNES_pre_1750_area_sr)
    else:
        gbf4_snes_scale = 1.0

    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES == "on":
        GBF4_ECNES_pre_1750_area_sr, gbf4_ecnes_scale = rescale_per_species_data(GBF4_ECNES_pre_1750_area_sr)
    else:
        gbf4_ecnes_scale = 1.0

    if settings.BIODIVERSITY_TARGET_GBF_8 == "on":
        GBF8_pre_1750_area_sr, gbf8_scale = rescale_per_species_data(GBF8_pre_1750_area_sr)
    else:
        gbf8_scale = 1.0

    scale_factors = {
        "Economy":          economy_scale,
        "Demand":           demand_scale,
        "Biodiversity":     biodiv_scale,
        "GHG":              ghg_scale,
        "Water":            water_scale,
        "GBF2":             gbf2_scale,
        "GBF3_NVIS":        gbf3_nvis_scale,
        "GBF4_SNES":        gbf4_snes_scale,
        "GBF4_ECNES":       gbf4_ecnes_scale,
        "GBF8":             gbf8_scale,
        "Renewable_Solar":  renewable_solar_scale,
        "Renewable_Wind":   renewable_wind_scale,
    }

    base_yr_prod = {
        "BASE_YR Economy(AUD)":        get_BASE_YR_economic_value(data),
        "BASE_YR Production (t)":      get_BASE_YR_production_t(data),
        "BASE_YR GHG (tCO2e)":         get_BASE_YR_GHG_t(data),
        "BASE_YR Water (ML)":          get_BASE_YR_water_ML(data),
        "BASE_YR Bio quality (score)": get_BASE_YR_bio_quality_value(data),
        "BASE_YR GBF_2 (score)":       get_BASE_YR_GBF2_score(data),
    }

    economic_contr_mrj=(ag_obj_mrj, non_ag_obj_rk,  ag_man_objs)
    economic_prices=get_commodity_prices_target_yr(data, target_year)
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
    ag_mask_proportion_r=data.AG_MASK_PROPORTION_R

    ag_x_mrj = land_use_culling.apply_agricultural_land_use_culling(ag_x_mrj, ag_c_mrj, ag_t_mrj, ag_r_mrj)
 
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
        
        renewable_solar_r,
        renewable_wind_r,
        exist_renewable_solar_r,
        exist_renewable_wind_r,

        region_state_r,
        region_state_name2idx,
        region_NRM_names_r,
        
        water_region_indices,
        water_region_names,
        
        biodiv_contr_ag_j,
        biodiv_contr_non_ag_k,
        biodiv_contr_ag_man,
        
        GBF2_mask_area_r,
        GBF3_NVIS_pre_1750_area_vr,
        GBF3_NVIS_region_group,
        GBF4_SNES_pre_1750_area_sr,
        GBF4_SNES_region_species,
        GBF4_ECNES_pre_1750_area_sr,
        GBF4_ECNES_region_species,
        GBF8_pre_1750_area_sr,
        GBF8_region_species,
        
        savanna_eligible_r,
        GBF2_mask_idx,
        renewable_GBF2_mask_solar_idx,
        renewable_GBF2_mask_wind_idx,
        renewable_MNES_mask_solar_idx,
        renewable_MNES_mask_wind_idx,

        base_yr_prod,
        scale_factors,
        
        economic_contr_mrj,
        economic_prices,
        economic_target_yr_carbon_price,
        
        offland_ghg,
        
        lu2pr_pj,
        pr2cm_cp,
        limits,
        desc2aglu,
        real_area,
        ag_mask_proportion_r
    )
