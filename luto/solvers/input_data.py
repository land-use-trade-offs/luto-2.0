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
import pandas as pd
import xarray as xr

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional

from luto.data import Data
from luto import settings
import luto.tools as tools
# from luto.economics import land_use_culling   # DECOMMISSIONED: land-use culling is incompatible with
#   the per-source (from→to) flow transition model — it pruned the exclude matrix using a single
#   dominant-LU-per-cell transition cost, whereas costs/deltas are now keyed per (from_m, from_j) source,
#   and pruning after the flow-feasibility dicts are built would leave deltas targeting cells with no X var.

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

    non_ag_g_rk: np.ndarray                                             # Non-agricultural greenhouse gas emissions matrix.
    non_ag_w_rk: np.ndarray                                             # Non-agricultural water yields matrix.
    non_ag_b_rk: np.ndarray                                             # Non-agricultural biodiversity matrix.
    non_ag_q_crk: np.ndarray                                            # Non-agricultural yield matrix.

    ag_man_g_mrj: dict                                                  # Agricultural Management options' GHG emission effects.
    ag_man_w_mrj: dict                                                  # Agricultural Management options' water yield effects.
    ag_man_b_mrj: dict                                                  # Agricultural Management options' biodiversity effects.
    ag_man_q_mrp: dict                                                  # Agricultural Management options' quantity effects.
    ag_man_limits: dict                                                 # Agricultural Management options' adoption limits.
    ag_man_lb_mrj: dict                                                 # Agricultural Management options' lower bounds.

    dvar_base_ag_mrj: np.ndarray                                        # Agricultural base year decision variable values.
    dvar_base_non_ag_rk: np.ndarray                                     # Non-agricultural base year decision variable values.

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
    ag_fold_map: dict                                                   # θ-fold sliver bookkeeping; consumed in solver.py to build the accounting stream X_acct.
    acct_cells_mrj: dict                                                # {(m,j): cells} = feasible_ag_cells_mrj ∪ folded-sliver cells — the accounting needs to iterate over.
    biodiv_contr_non_ag_k: dict[int, float]                             # Biodiversity contribution scale from non-agricultural land uses.
    biodiv_contr_ag_man: dict[str, dict[int, np.ndarray]]               # Biodiversity contribution scale from agricultural management options.
    
    GBF2_mask_area_r: np.ndarray                                        # Raw areas (GBF2) from priority degrade areas - indexed by cell (r).
    GBF3_NVIS_pre_1750_area_vr: np.ndarray                              # Raw areas (GBF3) from NVIS vegetation - indexed by group (v) and cell (r)
    GBF3_NVIS_region_group: dict[int, str]                              # GBF3 NVIS vegetation group names - indexed by group (v).
    GBF4_SNES_pre_1750_area_sr: xr.DataArray                            # Areas (GBF4) SNES - xr.DataArray[layer, cell], layer coord is MultiIndex(species, presence). No region dim — region masking applied in solver.
    GBF4_SNES_region_species: list                                      # GBF4 SNES constraint triplets - list[(region, species, presence)].
    GBF4_ECNES_pre_1750_area_sr: xr.DataArray                           # Areas (GBF4) ECNES - xr.DataArray[layer, cell], layer coord is MultiIndex(species, presence). No region dim — region masking applied in solver.
    GBF4_ECNES_region_species: list                                     # GBF4 ECNES constraint triplets - list[(region, community, presence)].
    GBF8_pre_1750_area_sr: xr.DataArray                                 # Areas (GBF8) - xr.DataArray[species, cell], species coord = species name strings.
    GBF8_region_species: list                                           # GBF8 constraint pairs - list[(region, species)].

    savanna_eligible_r: np.ndarray                                      # Cells that are eligible for savanna burnining land use.
    GBF2_mask_idx: np.ndarray                                           # Index of the mask of priority degraded areas.
    renewable_GBF2_mask_solar_idx: np.ndarray                           # Index of GBF2 mask for solar renewable exclusion.
    renewable_GBF2_mask_wind_idx: np.ndarray                            # Index of GBF2 mask for wind renewable exclusion.
    renewable_MNES_mask_solar_idx: np.ndarray                           # Index of EPBC MNES mask for solar renewable exclusion.
    renewable_MNES_mask_wind_idx: np.ndarray                            # Index of EPBC MNES mask for wind renewable exclusion.

    base_yr_prod: dict[str, tuple]                                      # Base year production of each commodity.
    scale_factors: dict[float]                                          # Scale factors for each input layer.
    commodity_names: list[str]                                          # Commodity names (data.COMMODITIES order, alphabetical).

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

    # ── FROM-view: the cells of each (from_m, from_j) source; local_r in the dicts below indexes into these ──
    ag_source_cells: dict                                               # {(from_m,from_j): cell_idx}   — anchors ag2ag AND ag2nonag flow families (exact only).
    nonag_source_cells: dict                                            # {from_k: cell_idx}            — anchors nonag2ag AND nonag2nonag flow families (exact only).

    # ── FROM-view: cost of (from_m, from_j) -> (to_m, to_j); economy-rescaled ──
    flow_cost_ag2ag: dict                                               # dict[(from_m,from_j)] → ndarray(NLMS, ncells_src, N_AG_LUS).
    flow_cost_ag2nonag: dict                                            # dict[(from_m,from_j)] → dict[k → ndarray(ncells_src,)].
    flow_cost_nonag2ag: dict                                            # dict[k] → ndarray(NLMS, ncells_k, N_AG_LUS).
    flow_ghg_ag2ag: dict                                                # dict[(from_m,from_j)] → ndarray(NLMS, ncells_src, N_AG_LUS) raw t CO2, GHG-rescaled. Physical parallel of flow_cost_ag2ag; the GHG constraint sums Σ flow_ghg·D (source-correct transition emissions).

    # ── FROM-view: feasibility of (from_m, from_j) -> (to_m, to_j); True if a delta var may be created ──
    feasible_ag2ag_mrj: dict                                            # {(from_m,from_j): bool (NLMS, ncells_src, N_AG_LUS) [to_m, local_r, to_j]}.
    feasible_nonag2ag_mrj: dict                                         # {from_k: bool (NLMS, ncells_k, N_AG_LUS) [to_m, local_r, to_j]}.
    feasible_ag2nonag_rk: dict                                          # {(from_m,from_j): bool (ncells_src, N_NON_AG_LUS) [local_r, k]}.

    # ── TO-view: Target-keyed bounds (which target a cell may become, and its lb..ub) ──
    dvar_ub_ag: np.ndarray                                              # Ag target upper bound (NLMS, NCELLS, N_AG_LUS): ag2ag + nonag2ag reachable share (fractional).
    dvar_lb_ag: np.ndarray                                              # Ag target lower bound (NLMS, NCELLS, N_AG_LUS) — zeros for now.
    feasible_ag_cells_mrj: dict                                         # {(to_m,to_j): cell_idx} — cells with a routable >θ source (per-source, from ag_x_mrj>0) get an ag target var (was ag_lu2cells).

    # ── TO-view: Target-keyed bounds (which non-ag k a cell may become, and its lb..ub) ──
    dvar_ub_nonag: np.ndarray                                           # Non-ag TARGET upper bound (NCELLS, N_NON_AG_LUS); reachability + reversibility lock-in + RP/destock caps.
    dvar_lb_nonag: np.ndarray                                           # Non-ag TARGET lower bound (NCELLS, N_NON_AG_LUS); irreversible LU lock-in floor (0 for reversible).
    feasible_non_ag_cells: dict                                         # {k: cell_idx} — cells whose dvar_ub_nonag > 0 get a non-ag target var (was non_ag_lu2cells).

    @property
    def ncms(self):
        return len(self.commodity_names)
    
    @property
    def ncells(self):
        # Number of cells
        return self.ag_g_mrj.shape[1]
    
    @property
    def nlms(self):
        # Number of water managements
        return self.ag_g_mrj.shape[0]

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
    if settings.GBF2_TARGET == "off":
        return np.empty(0)
    print('Getting GBF2 mask area layer...', flush = True)
    output = ag_biodiversity.get_GBF2_MASK_area(data)
    return output

def get_GBF3_NVIS_pre_1750_area_vr(data: Data):
    if settings.GBF3_NVIS_TARGET == "off":
        return np.empty(0)
    print('Getting GBF3 NVIS vegetation matrices...', flush = True)
    output = ag_biodiversity.get_GBF3_NVIS_matrices_vr(data)
    return output

def get_GBF3_NVIS_region_group(data: Data) -> dict[int,str]:
    if settings.GBF3_NVIS_TARGET == "off":
        return {}
    print('Getting GBF3 NVIS vegetation group names...', flush = True)
    return data.BIO_GBF3_NVIS_SEL

def get_GBF4_SNES_pre_1750_area_sr(data: Data) -> xr.DataArray:
    if settings.GBF4_TARGET_SNES == 'off':
        return np.empty(0)
    print('Getting GBF4 SNES species area matrices...', flush=True)
    return ag_biodiversity.get_GBF4_SNES_matrix_sr(data)

def get_GBF4_SNES_region_species(data: Data) -> list:
    if settings.GBF4_TARGET_SNES == 'off':
        return []
    print('Getting GBF4 SNES (region, species, presence) constraint triplets...', flush=True)
    return data.BIO_GBF4_SNES_SEL

def get_GBF4_ECNES_pre_1750_area_sr(data: Data) -> xr.DataArray:
    if settings.GBF4_TARGET_ECNES == 'off':
        return np.empty(0)
    print('Getting GBF4 ECNES community area matrices...', flush=True)
    return ag_biodiversity.get_GBF4_ECNES_matrix_sr(data)

def get_GBF4_ECNES_region_species(data: Data) -> list:
    if settings.GBF4_TARGET_ECNES == 'off':
        return []
    print('Getting GBF4 ECNES (region, community, presence) constraint triplets...', flush=True)
    return data.BIO_GBF4_ECNES_SEL

def get_GBF8_pre_1750_area_sr(data: Data, target_year: int) -> xr.DataArray:
    if settings.GBF8_TARGET == "off":
        return np.empty(0)
    print('Getting GBF8 species conservation area matrices...', flush=True)
    return ag_biodiversity.get_GBF8_matrix_sr(data, target_year)

def get_GBF8_region_species(data: Data) -> list:
    if settings.GBF8_TARGET == "off":
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


def get_ag_t_mrj(data: Data, target_index, base_year):
    print('Getting agricultural transition cost matrices...', flush = True)
    # From-based flow-cost dict[(from_m, from_j)] -> ndarray(NLMS, ncells_src, N_AG_LUS), sliced per
    # source over each source's dvar>θ cells (the same cells `ag_source_cells` uses, so the solver
    # delta's local_r aligns with this dict's cell axis). Leaves are cast to float32 during the economy
    # rescale in get_input_data (no premature .astype on the dict).
    mj_cell_map = ag_transition.get_base_dvar_mj_cell_map(data, base_year)
    return {
        (from_m, from_j): ag_transition.get_transition_matrices_ag2ag(data, target_index, from_m, from_j, cell_idx)
        for (from_m, from_j), cell_idx in mj_cell_map.items()
    }


def get_non_ag_t_rk(data: Data, base_year):
    # nonag→nonag transition cost. Currently a ZERO matrix — non-ag LUs are not allowed to transition
    # to other non-ag LUs (get_nonag2nonag_transition_matrix returns zeros). Kept as an explicit hook
    # so the objective wiring is ready if non-ag↔non-ag transitions are ever priced.
    print('Getting non-agricultural transition cost matrices...', flush = True)
    output = non_ag_transition.get_nonag2nonag_transition_matrix(data)
    return output


def get_ag_x_mrj(data: Data, base_year):
    print('Getting agricultural exclude matrices...', flush = True)
    return ag_transition.get_to_ag_exclude_matrices(data, base_year)


def get_feasible_ag_cells_mrj(ag_x_mrj: np.ndarray, dvar_lb_ag: np.ndarray) -> dict:
    print('Getting feasible agricultural cells...', flush = True)
    n_lms, _ncells, n_lus = ag_x_mrj.shape
    eligible = (ag_x_mrj > 0) | (dvar_lb_ag > 0)
    return {
        (m, j): np.where(eligible[m, :, j])[0]
        for j in range(n_lus)
        for m in range(n_lms)
    }


def get_feasible_non_ag_cells(dvar_ub_nonag: np.ndarray, threshold: float = 0.0) -> dict:
    print('Getting feasible non-agricultural cells...', flush = True)
    n_k = dvar_ub_nonag.shape[1]
    return {k: np.where(dvar_ub_nonag[:, k] > threshold)[0] for k in range(n_k)}


def get_ag_source_cells(data: Data, base_year: int) -> dict:
    print('Getting agricultural source cells...', flush = True)
    return ag_transition.get_base_dvar_mj_cell_map(data, base_year)


def get_nonag_source_cells(data: Data, base_year: int) -> dict:
    print('Getting non-agricultural source cells...', flush = True)
    return non_ag_transition.get_base_nonag_dvar_k_cell_map(data, base_year)


def get_feasible_ag2ag_mrj(ag_x_mrj: np.ndarray, ag_source_cells: dict, T_ag2ag_reach_jj: np.ndarray) -> dict:
    """Ag2ag delta-var feasibility, SOURCE-KEYED like flow_cost_ag2ag:

        {(from_m, from_j): bool (NLMS, ncells_src, N_AG_LUS) [to_m, local_r, to_j]}

        feasible[to_m, local_r, to_j] = (ag_x_mrj[to_m, r, to_j] > 0)     target eligible (X var exists)
                                      ∧ T_MAT[from_j → to_j]              THIS source may make the move
                                      ∧ not the diagonal                  staying is not a transition

    One leaf per source over that source's dvar>θ cells (`ag_source_cells`, the same
    get_base_dvar_mj_cell_map slices that anchor the flow vars) — the solver adds one delta var per
    True entry, nothing more. `ag_x_mrj > 0` (exclude matrix: union reach × EXCLUDE × no-go) is the
    same quantity that creates the ag X vars, so every delta lands on an existing var.
    """
    print('Getting feasible ag2ag delta-var targets...', flush = True)
    eligible = ag_x_mrj > 0
    result = {}
    for (fm, fj), cells in ag_source_cells.items():
        valid = eligible[:, cells, :] & T_ag2ag_reach_jj[fj][None, None, :]     # (NLMS, ncells_src, N_AG)
        valid[fm, :, fj] = False                                                # drop the diagonal
        result[(fm, fj)] = valid
    return result


def get_feasible_nonag2ag_mrj(ag_x_mrj: np.ndarray, nonag_source_cells: dict, T_nonag2ag_reach_kj: np.ndarray) -> dict:
    """Nonag2ag delta-var feasibility, SOURCE-KEYED like flow_cost_nonag2ag:

        {from_k: bool (NLMS, ncells_k, N_AG_LUS) [to_m, local_r, to_j]}

    Same construction as get_feasible_ag2ag_mrj but from the non-ag sources (`nonag_source_cells`,
    e.g. reversible Destocked land converting back to ag). No diagonal to drop (cross-family).
    """
    print('Getting feasible nonag2ag delta-var targets...', flush = True)
    eligible = ag_x_mrj > 0
    return {
        fk: eligible[:, cells, :] & T_nonag2ag_reach_kj[fk][None, None, :]      # (NLMS, ncells_k, N_AG)
        for fk, cells in nonag_source_cells.items()
    }


def get_feasible_ag2nonag_rk(dvar_ub_nonag: np.ndarray, ag_source_cells: dict, T_ag2nonag_reach_jk: np.ndarray) -> dict:
    """Ag2nonag delta-var feasibility, SOURCE-KEYED like flow_cost_ag2nonag:

        {(from_m, from_j): bool (ncells_src, N_NON_AG_LUS) [local_r, k]}

    The target side gates on `dvar_ub_nonag > 0` — NOT raw T_MAT reach — because the non-ag ub
    carries extra zeroing caps (RP stream-buffer, Destocked eligibility, non-ag no-go): a raw-reach
    gate would point deltas at targets with no X var and land would vanish through the missing
    node-balance row. No diagonal to drop (cross-family).
    """
    print('Getting feasible ag2nonag delta-var targets...', flush = True)
    eligible = dvar_ub_nonag > 0
    return {
        (fm, fj): eligible[cells, :] & T_ag2nonag_reach_jk[fj][None, :]         # (ncells_src, N_NONAG)
        for (fm, fj), cells in ag_source_cells.items()
    }


def get_dvar_ub_ag(data: Data, base_year: int) -> np.ndarray:
    print('Getting agricultural target upper bounds...', flush = True)
    ub = (
        ag_transition.get_ag2ag_ub(data, base_year)
        + non_ag_transition.get_nonag2ag_ub(data, base_year)
    ).astype(np.float32)
    # A cell can always KEEP its base LU ⇒ ub must be ≥ base (exact Σfrac can land a hair below base
    # on float noise, e.g. 0.9999<1.0, which would break cell-usage saturation Σ X = ag_mask). Also ≥0.
    
    # NOTE: if a real gap is reported here (not float noise), some base land-use is banned by
    # EXCLUDE/no-go (e.g. a pre-reconciliation x_mrj.npy, or a no-go region overlapping the base map).
    # The raise does NOT let such cells keep their base LU — with ag_x_mrj=0 and lb=0 no X var exists
    # (get_feasible_ag_cells_mrj), so the solver force-converts that land; the raise only keeps the
    # lb <= base <= ub box coherent for bookkeeping (const clipping, has_any_ag_r).

    # FOLDED base: the solver's const/base is the folded dvar, so ub must cover THAT (dominant
    # entries carry their absorbed sub-θ mass; folded-away entries need no headroom).
    base = ag_transition.get_folded_base_ag_dvar(data, base_year)
    return tools.clamp_dvar_bound(ub, np.maximum(base, 0.0), np.inf, 'Ag ub raised to base')

def get_dvar_ub_nonag(data: Data, base_year):
    print('Getting non-agricultural target upper bounds...', flush = True)
    base_dvar_nonag = (
        data.non_ag_dvars[base_year] if base_year != data.YR_CAL_BASE
        else np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32)
    )
    ub = non_ag_transition.get_non_ag_ub_matrices(
        data,
        base_dvar_nonag_rk=base_dvar_nonag,
        base_dvar_ag_mrj=ag_transition.get_folded_base_ag_dvar(data, base_year),   # solver-world identity
    )
    return tools.clamp_dvar_bound(ub, np.maximum(base_dvar_nonag, 0.0), np.inf, 'NonAg ub raised to base')

def get_dvar_lb_ag(data: Data, base_year: int) -> np.ndarray:
    print('Getting agricultural target lower bounds...', flush = True)
    lb = ag_transition.get_ag2ag_lb(data, base_year)      # all zeros — sliver pin superseded by θ-folding
    base = ag_transition.get_folded_base_ag_dvar(data, base_year)
    # lb must sit in [0, base] (never above the base it locks in).
    return tools.clamp_dvar_bound(lb, 0.0, np.maximum(base, 0.0), 'Ag lb clamped to [0,base]')

def get_dvar_lb_nonag(data: Data, base_year):
    print('Getting non-agricultural lower bound matrices...', flush = True)
    lb = non_ag_transition.get_non_ag_lb_matrices(data, base_year)
    base = (
        data.non_ag_dvars[base_year].astype(np.float32) if base_year != data.YR_CAL_BASE
        else np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32)
    )
    return tools.clamp_dvar_bound(lb, 0.0, np.maximum(base, 0.0), 'NonAg lb clamped to [0,base]')

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

def get_exist_renewable_fraction_solar_r(data: Data, yr_cal: int = None):
    print('Getting existing solar capacity fraction (all years, solver ceiling)...', flush=True)
    # Existing real-world capacity and LUTO-simulated capacity compete for the same
    # cell space [0, 1]. We lock the maximum existing fraction (cumulative 2000-2035)
    # in advance so that simulated + existing never exceeds 1 in any period.
    # Using all years (yr_cal=99999) keeps the ceiling fixed across solver calls,
    # preventing lb > ub when new real-world capacity enters mid-simulation.
    return ag_quantity.get_existing_renewable_dvar_fraction(data, 'Utility Solar PV', 99999)

def get_exist_renewable_fraction_wind_r(data: Data, yr_cal: int = None):
    print('Getting existing wind capacity fraction (all years, solver ceiling)...', flush=True)
    # Same rationale as solar: lock maximum existing fraction to prevent simulated + existing > 1.
    return ag_quantity.get_existing_renewable_dvar_fraction(data, 'Onshore Wind', 99999)

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


def get_ag_man_c_mrj(data: Data, ag_c_mrj: np.ndarray, target_year):
    print('Getting agricultural management options\' cost effects...', flush = True)
    output = ag_cost.get_agricultural_management_cost_matrices(data, ag_c_mrj, target_year)
    return output


def get_ag_man_g_mrj(data: Data, target_index):
    print('Getting agricultural management options\' GHG emission effects...', flush = True)
    return ag_ghg.get_agricultural_management_ghg_matrices(data, target_index)


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
    non_ag_c_rk: np.ndarray,
    non_ag_r_rk: np.ndarray,
    non_ag_t_rk: np.ndarray,
    ag_man_c_mrj: dict[str, np.ndarray],
    ag_man_r_mrj: dict[str, np.ndarray],
    ag_man_t_mrj: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray|dict[str, np.ndarray]]:

    print('Getting base year economic matrix...', flush = True)

    # Land-use TRANSITION costs (ag2ag, ag2nonag, nonag2ag) are NOT baked here. They are charged in the
    # solver against the per-source delta vars via the source-keyed flow_cost dicts (Σ flow_cost·D).
    # get_economic_mrj is pure operating economics: revenue − (production cost). Two non-flow
    # terms remain: non_ag_t_rk (nonag→nonag — currently a ZERO matrix, since non-ag↛non-ag is disallowed;
    # kept as a hook if it's ever priced) and ag_man_t_mrj (ag-management adoption cost).
    if settings.OBJECTIVE == "maxprofit":
        # Pre-calculate profit (revenue minus cost) for each land use
        ag_obj_mrj = ag_r_mrj - ag_c_mrj
        non_ag_obj_rk = non_ag_r_rk - (non_ag_c_rk + non_ag_t_rk)

        # Get effects of alternative agr. management options (stored in a dict)
        ag_man_objs = {
            am: ag_man_r_mrj[am] - (ag_man_c_mrj[am] + ag_man_t_mrj[am])
            for am in settings.AG_MANAGEMENTS_TO_LAND_USES
        }

    elif settings.OBJECTIVE == "mincost":
        # Pre-calculate sum of production costs (land-use transition cost enters via flow_cost in the solver)
        ag_obj_mrj = ag_c_mrj
        non_ag_obj_rk = non_ag_c_rk + non_ag_t_rk

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
    if settings.GBF2_TARGET == "off":
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
    if settings.GBF2_TARGET == "off":
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


def get_limits(data: Data, yr_cal: int) -> dict[str, Any]:
    """
    Return raw (unscaled) constraint targets for the given calendar year.

    Keys returned depend on active settings:
      'demand', 'water', 'ghg',
      'renewable_Utility Solar PV', 'renewable_Onshore Wind',
      'renewable_Utility Solar PV_exist', 'renewable_Onshore Wind_exist',
      'GBF2', 'GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8',
      'ag_regional_adoption', 'non_ag_regional_adoption', 'non_ag_regional_adoption_sum'

    All values are raw (unscaled); the solver divides each by the appropriate
    ``scale_factors[key]`` entry inline when building constraints.
    """
    print('Getting environmental limits...', flush = True)
    
    limits = {}
    
    # Clamped again here, not only in Data.__init__: a resumed run loads a pickled Data and never
    # re-runs __init__, so a checkpoint written before the clamp existed would still carry negatives.
    limits['demand'] = np.maximum(data.D_CY[yr_cal - data.YR_CAL_BASE], 0.0)
    
    if settings.WATER_LIMITS == 'on':
        limits['water'] = data.WATER_YIELD_TARGETS
        
    if settings.GHG_EMISSIONS_LIMITS != 'off':
        limits['ghg'] = data.GHG_TARGETS[yr_cal]
        
    if any(settings.RENEWABLES_OPTIONS.values()):
        renewable_targets = data.RENEWABLE_TARGETS.query('Year == @yr_cal').set_index('state')
        limits['renewable_Utility Solar PV'] = renewable_targets.query('tech == "Utility Solar"')['Renewable_Target_MWh'].to_dict()
        limits['renewable_Onshore Wind'] = renewable_targets.query('tech == "Wind"')['Renewable_Target_MWh'].to_dict()
        
        renewable_existing_capacity = get_exist_renewable_capacity_by_state_input(data, yr_cal)
        limits['renewable_Utility Solar PV_exist'] = {state: vals['Utility Solar PV'] for state, vals in renewable_existing_capacity.items()}
        limits['renewable_Onshore Wind_exist']     = {state: vals['Onshore Wind']     for state, vals in renewable_existing_capacity.items()}

    if settings.GBF2_TARGET != 'off':
        limits["GBF2"] = data.get_GBF2_target_for_yr_cal(yr_cal)

    if settings.GBF3_NVIS_TARGET != 'off':
        limits["GBF3_NVIS"] = data.get_GBF3_NVIS_limit_score_inside_LUTO_by_yr(yr_cal)

    if settings.GBF4_TARGET_SNES != 'off':
        limits["GBF4_SNES"] = data.get_GBF4_SNES_target_inside_LUTO_by_year(yr_cal)

    if settings.GBF4_TARGET_ECNES != 'off':
        limits["GBF4_ECNES"] = data.get_GBF4_ECNES_target_inside_LUTO_by_year(yr_cal)

    if settings.GBF8_TARGET != "off":
        limits["GBF8"] = data.get_GBF8_target_inside_LUTO_by_yr(yr_cal)

    if settings.REGIONAL_ADOPTION_CONSTRAINTS != 'off':
        ag_reg_adoption, non_ag_reg_adoption, non_ag_reg_adoption_sum = ag_transition.get_regional_adoption_limits(data, yr_cal)
        limits["ag_regional_adoption"] = ag_reg_adoption
        limits["non_ag_regional_adoption"] = non_ag_reg_adoption
        limits["non_ag_regional_adoption_sum"] = non_ag_reg_adoption_sum

    return limits


def calc_geomean_scale(lhs_max: float, rhs_max: float) -> float:
    """
    Compute a single scale factor as the geometric mean of LHS and RHS magnitudes,
    normalised by ``settings.RESCALE_FACTOR`` (RF)::

        scale = sqrt(lhs_max * rhs_max) / RF

    After dividing both sides by this scale:
      - LHS_max_scaled = RF * sqrt(lhs_max / rhs_max)
      - RHS_scaled     = RF * sqrt(rhs_max / lhs_max)

    Both land symmetrically around RF in log space.
    Falls back to LHS-only (``lhs_max / RF``) when ``rhs_max`` is zero.
    """
    if lhs_max > 0.0 and rhs_max > 0.0:
        return float(np.sqrt(lhs_max * rhs_max) / settings.RESCALE_FACTOR)
    ref = lhs_max if lhs_max > 0.0 else settings.RESCALE_FACTOR
    return float(ref / settings.RESCALE_FACTOR)


def rescale_lhs(arrays: list) -> tuple[list, float]:
    """
    Rescale arrays using LHS-only scaling: ``scale = max(|LHS|) / RF``.

    All arrays in the group share one scale factor to preserve their relative
    magnitudes (e.g. ag, non-ag, and ag-management variants of the same quantity).
    Returns ``(scaled_arrays, scale_factor)``.
    """
    ref = 0.0
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            ref = max(ref, float(np.abs(arr).max()))
        elif isinstance(arr, dict):
            for v in arr.values():
                ref = max(ref, float(np.abs(v).max()))

    scale = np.float32(calc_geomean_scale(ref, 0.0))  # rhs_max=0 → LHS-only path

    scaled = []
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            scaled.append((arr / scale).astype(np.float32))
        elif isinstance(arr, dict):
            scaled.append({k: (v / scale).astype(np.float32) for k, v in arr.items()})
        else:
            scaled.append(arr)

    return scaled, float(scale)


def rescale_lhs_rhs(arrays: list, rhs_target) -> tuple[list, float]:
    """
    Rescale arrays using the geometric mean of max(|LHS|) and max(|RHS|)::

        scale = sqrt(max_lhs * max_rhs) / RF

    Both LHS coefficients and the RHS target land symmetrically around RF in log
    space, keeping both sides within Gurobi's recommended [1e-3, 1e6] band as long
    as ``max_rhs / max_lhs < 1e9`` (holds for all LUTO constraint types).

    Use this for aggregate constraints (GHG, Water, GBF2, Demand, Renewable) where
    per-cell LHS values and a national/regional RHS total would otherwise diverge.

    Args:
        arrays:     List of ``np.ndarray`` or ``dict[str, np.ndarray]``.
        rhs_target: Scalar float, ``dict[key, float]``, or ``np.ndarray``.
                    Representative magnitude is ``max(|rhs_target|)``.

    Returns:
        ``(scaled_arrays, scale_factor)``
    """
    lhs_max = 0.0
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            lhs_max = max(lhs_max, float(np.abs(arr).max()))
        elif isinstance(arr, dict):
            for v in arr.values():
                lhs_max = max(lhs_max, float(np.abs(v).max()))

    if isinstance(rhs_target, dict):
        rhs_max = max((abs(v) for v in rhs_target.values()), default=0.0)
    elif isinstance(rhs_target, np.ndarray):
        rhs_max = float(np.abs(rhs_target).max()) if rhs_target.size else 0.0
    else:
        rhs_max = abs(float(rhs_target))

    scale = np.float32(calc_geomean_scale(lhs_max, rhs_max))
    if scale == 0.0:
        scale = np.float32(1.0)

    scaled = []
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            scaled.append((arr / scale).astype(np.float32))
        elif isinstance(arr, dict):
            scaled.append({k: (v / scale).astype(np.float32) for k, v in arr.items()})
        else:
            scaled.append(arr)

    return scaled, float(scale)


def rescale_lhs_rhs_region_species(
    arr: xr.DataArray,
    constraint_list: list[tuple],
    region_NRM_names_r: np.ndarray,
    targets: xr.DataArray | None = None,
    layer_coord_names: tuple[str, ...] = ('region', 'species'),
    matrix_key_fn=None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Rescale a biodiversity matrix per constraint tuple using the geometric mean
    of max(|LHS|) and the tuple's RHS target.

    Calls :func:`calc_geomean_scale` for each constraint::

        scale = calc_geomean_scale(region_max, abs(target_val))

    When ``targets`` is ``None``, falls back to LHS-only scaling.

    Supports two matrix layouts:
    - ``row_coord = 'layer'`` (GBF4 ECNES): constraint tuples are (region, species);
      the matrix row key equals the full tuple.
    - ``row_coord = 'layer'`` with ``matrix_key_fn`` (GBF4 SNES): constraint tuples
      are (region, species, presence); ``matrix_key_fn`` extracts (species, presence)
      as the matrix row key, while region[0] drives the cell mask.
    - ``row_coord = 'group'`` or ``'species'`` (GBF3 NVIS, GBF8): constraint tuples
      are (region, species); the matrix row key is the last element (species/group).

    Region masking: first element of each constraint tuple is the region.
    ``'AUSTRALIA'`` → all cells; any other value → ``region_NRM_names_r == region``.
    In-place scaling is safe because NRM regions are non-overlapping — each cell
    belongs to at most one NRM region.

    ``layer_coord_names`` sets the MultiIndex level names on the returned
    ``scale_factors`` DataArray. The scale_factors MultiIndex mirrors
    ``constraint_list`` so callers can use ``scale_factors.sel(layer=constraint_tuple)``.

    Returns:
        scaled_arr: xr.DataArray, same shape as ``arr``.
        scale_factors: xr.DataArray with ``layer`` MultiIndex matching constraint_list.
    """
    arr_np = arr.values.copy().astype(np.float32)
    row_coord = arr.dims[0]
    row_names = arr.coords[row_coord].values
    row_name_to_idx = {name: i for i, name in enumerate(row_names)}
    use_tuple_key = (row_coord == 'layer')  # GBF4: row key is a tuple

    layers: list[tuple] = []
    sf_values: list[float] = []

    for constraint_tuple in constraint_list:
        region = constraint_tuple[0]

        if matrix_key_fn is not None:
            row_key = matrix_key_fn(constraint_tuple)
        elif use_tuple_key:
            row_key = constraint_tuple
        else:
            row_key = constraint_tuple[-1]

        row_idx = row_name_to_idx[row_key]

        cell_mask = (
            np.ones(arr_np.shape[1], dtype=bool)
            if region == "AUSTRALIA"
            else region_NRM_names_r == region
        )

        region_max = float(np.abs(arr_np[row_idx, cell_mask]).max()) if cell_mask.any() else 0.0

        if targets is not None:
            target_val = abs(float(targets.sel(layer=constraint_tuple).item()))
        else:
            target_val = 0.0

        scale_factor = np.float32(calc_geomean_scale(region_max, target_val))

        arr_np[row_idx, cell_mask] = arr_np[row_idx, cell_mask] / scale_factor

        layers.append(constraint_tuple)
        sf_values.append(float(scale_factor))

    scaled_arr = xr.DataArray(arr_np, dims=arr.dims, coords=arr.coords, name=arr.name)
    scale_factors = xr.DataArray(
        np.array(sf_values, dtype=np.float32),
        dims=["layer"],
        coords={"layer": pd.MultiIndex.from_tuples(layers, names=list(layer_coord_names))},
    )
    return scaled_arr, scale_factors



def get_input_data(data: Data, base_year: int, target_year: int) -> SolverInputData:
    """
    Using the given Data object, prepare a SolverInputData object for the solver.
    """

    target_index = target_year - data.YR_CAL_BASE
    ag_c_mrj     = get_ag_c_mrj(data, target_index)
    ag_r_mrj     = get_ag_r_mrj(data, target_index)

    # ── Transition costs — SOURCE-KEYED flow-cost dicts ──────────────
    # Sliced by base-year source ("(from_m, from_j)" for ag, "k" for non-ag) over each source's dvar>θ
    # cells; the solver creates a matching delta var per (source, cell, target) and charges
    # Σ flow_cost·D in the objective. get_economic_mrj bakes no land-use transition cost.

    # ag→ag: dict[(from_m, from_j)] → ndarray(NLMS, ncells_src, N_AG_LUS)
    flow_cost_ag2ag = get_ag_t_mrj(data, target_index, base_year)

    # ag→ag transition GHG EMISSIONS (raw t CO2), source-keyed — the physical parallel of
    # flow_cost_ag2ag. The GHG constraint sums Σ flow_ghg·D (source-correct transition emissions).
    # GHG-rescaled below.
    flow_ghg_ag2ag = ag_ghg.get_ghg_transition_emissions_from_base_year(data, base_year)

    # Per-source transition reachability (T_MAT finite ⇒ allowed) — decides which delta vars exist.
    T_ag2ag_reach_jj    = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.AGRICULTURAL_LANDUSES).values)
    T_ag2nonag_reach_jk = ~np.isnan(data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES,     to_lu=data.NON_AGRICULTURAL_LANDUSES).values)
    T_nonag2ag_reach_kj = ~np.isnan(data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values)

    # ag→nonag: dispatcher gives dict[lu_name → dict[(fm,fj)]]; transpose to dict[(fm,fj) → dict[k]]
    # so the solver loops ag sources first.
    flow_cost_ag2nonag = {}
    for _lu_name, _per_src in non_ag_transition.get_transition_matrix_ag2nonag(
        data, base_year, target_year
    ).items():
        _k = data.NON_AGRICULTURAL_LANDUSES.index(_lu_name)
        for _src, _arr in _per_src.items():
            flow_cost_ag2nonag.setdefault(_src, {})[_k] = _arr

    # nonag→ag: dispatcher gives dict[lu_name → dict[k]]; take the diagonal (cells in non-ag LU k
    # pay only LU k's own nonag→ag cost).
    flow_cost_nonag2ag = {}
    for _lu_name, _per_k in non_ag_transition.get_transition_matrix_nonag2ag(
        data, base_year, target_year
    ).items():
        _k = data.NON_AGRICULTURAL_LANDUSES.index(_lu_name)
        if _k in _per_k:
            flow_cost_nonag2ag[_k] = _per_k[_k]

    non_ag_c_rk                     = get_non_ag_c_rk(data, ag_c_mrj, data.lumaps[base_year], target_year)
    non_ag_r_rk                     = get_non_ag_r_rk(data, ag_r_mrj, base_year, target_year)
    non_ag_t_rk                     = get_non_ag_t_rk(data, base_year)

    ag_man_c_mrj                    = get_ag_man_c_mrj(data, ag_c_mrj, target_year)
    ag_man_r_mrj                    = get_ag_man_r_mrj(data, target_index, ag_r_mrj)
    ag_man_t_mrj                    = get_ag_man_t_mrj(data, target_index)
    
    ag_obj_mrj, non_ag_obj_rk,  ag_man_objs = get_economic_mrj(
        ag_c_mrj,
        ag_r_mrj,
        non_ag_c_rk,
        non_ag_r_rk,
        non_ag_t_rk,
        ag_man_c_mrj,
        ag_man_r_mrj,
        ag_man_t_mrj
    )
    

    ag_g_mrj                        = get_ag_g_mrj(data, target_index)
    ag_w_mrj                        = (
        get_ag_w_mrj(data, target_index) if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on' 
        else get_ag_w_mrj(data, target_index, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    )
    ag_b_mrj                        = get_ag_b_mrj(data)
    ag_x_mrj                        = get_ag_x_mrj(data, base_year)
    dvar_ub_ag                      = get_dvar_ub_ag(data, base_year)        # TO-view ag target upper bound (ag2ag + nonag2ag)
    dvar_lb_ag                      = get_dvar_lb_ag(data, base_year)        # TO-view ag target lower bound (zeros for now)
    ag_source_cells                 = get_ag_source_cells(data, base_year)   # FROM-view: cells holding each ag (from_m,from_j) source
    nonag_source_cells              = get_nonag_source_cells(data, base_year)# FROM-view: cells holding each non-ag source k
    ag_q_mrp                        = get_ag_q_mrp(data, target_index)

    non_ag_g_rk                     = get_non_ag_g_rk(data, ag_g_mrj, base_year)
    non_ag_w_rk                     = (
        get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year)   
        if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on' 
        else get_non_ag_w_rk(data, ag_w_mrj, base_year, target_year, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    )
    non_ag_b_rk                     = get_non_ag_b_rk(data, ag_b_mrj, base_year)
    dvar_ub_nonag                    = get_dvar_ub_nonag(data, base_year)
    feasible_non_ag_cells          = get_feasible_non_ag_cells(dvar_ub_nonag) # cells that get a target non-ag var (ub > 0)
    non_ag_q_crk                    = get_non_ag_q_crk(data, ag_q_mrp, base_year)
    dvar_lb_nonag                    = get_dvar_lb_nonag(data, base_year)
    
    ag_man_g_mrj                    = get_ag_man_g_mrj(data, target_index)
    ag_man_w_mrj                    = get_ag_man_w_mrj(data, target_index)
    ag_man_b_mrj                    = get_ag_man_b_mrj(data, target_index, ag_b_mrj)
    ag_man_q_mrp                    = get_ag_man_q_mrj(data, target_index, ag_q_mrp)
    ag_man_limits                   = get_ag_man_limits(data, target_index)                            
    ag_man_lb_mrj                   = get_ag_man_lb_mrj(data, base_year)
    
    renewable_solar_r               = get_potential_renewable_solar_r(data, target_index)
    renewable_wind_r                = get_potential_renewable_wind_r(data, target_index)
    exist_renewable_solar_r         = get_exist_renewable_fraction_solar_r(data, target_year)
    exist_renewable_wind_r          = get_exist_renewable_fraction_wind_r(data, target_year)

    region_state_r                  = get_region_state_r(data)
    region_state_name2idx           = get_region_state_name2idx(data)
    region_NRM_names_r              = get_region_NRM_names_r(data)
    
    water_region_indices            = get_w_region_indices(data)
    water_region_names              = get_w_region_names(data)
    
    biodiv_contr_ag_j               = get_ag_biodiv_contr_j(data)
    biodiv_contr_non_ag_k           = get_non_ag_biodiv_impact_k(data)
    biodiv_contr_ag_man             = get_ag_man_biodiv_impacts(data, target_year)

    GBF2_mask_area_r                = get_GBF2_mask_area_r(data)
    GBF3_NVIS_pre_1750_area_vr      = get_GBF3_NVIS_pre_1750_area_vr(data)
    GBF3_NVIS_region_group          = get_GBF3_NVIS_region_group(data)
    GBF4_SNES_pre_1750_area_sr      = get_GBF4_SNES_pre_1750_area_sr(data)
    GBF4_SNES_region_species        = get_GBF4_SNES_region_species(data)
    GBF4_ECNES_pre_1750_area_sr     = get_GBF4_ECNES_pre_1750_area_sr(data)
    GBF4_ECNES_region_species       = get_GBF4_ECNES_region_species(data)
    GBF8_pre_1750_area_sr           = get_GBF8_pre_1750_area_sr(data, target_year)
    GBF8_region_species             = get_GBF8_region_species(data)

    savanna_eligible_r              = get_savanna_eligible_r(data)
    GBF2_mask_idx                   = get_GBF2_mask_idx(data)
    renewable_GBF2_mask_solar_idx   = get_renewable_GBF2_mask_solar_idx(data)
    renewable_GBF2_mask_wind_idx    = get_renewable_GBF2_mask_wind_idx(data)
    renewable_MNES_mask_solar_idx   = get_renewable_MNES_mask_solar_idx(data)
    renewable_MNES_mask_wind_idx    = get_renewable_MNES_mask_wind_idx(data)

    # Fetch all raw limit targets once — reused in both rescaling and get_limits below.
    limits = get_limits(data, target_year)

    # Derive target eligibility from ag_x_mrj so it matches the mask the solver reads (per-source θ;
    # was ag_lu2cells). Stay-floor cells (dvar_lb_ag>0, sub-θ slivers locked in) are unioned in so
    # their var exists.
    feasible_ag_cells_mrj          = get_feasible_ag_cells_mrj(ag_x_mrj, dvar_lb_ag)  # cells that get a target ag var

    # Per-source delta-var feasibility — keyed/shaped like the flow_cost dicts; the solver adds one
    # delta var per True entry.
    feasible_ag2ag_mrj    = get_feasible_ag2ag_mrj(ag_x_mrj, ag_source_cells, T_ag2ag_reach_jj)
    feasible_nonag2ag_mrj = get_feasible_nonag2ag_mrj(ag_x_mrj, nonag_source_cells, T_nonag2ag_reach_kj)
    feasible_ag2nonag_rk  = get_feasible_ag2nonag_rk(dvar_ub_nonag, ag_source_cells, T_ag2nonag_reach_jk)

    # Rescale solver input data
    [ag_obj_mrj, non_ag_obj_rk, ag_man_objs], economy_scale = rescale_lhs([ag_obj_mrj, non_ag_obj_rk, ag_man_objs])

    # Put the source-keyed flow-cost dicts on the same economy band as the flat arrays above
    # (rescale_lhs only walks one dict level; these are 1–2 levels deep, so do it explicitly).
    # Leaves become float32, matching the flat-array convention.
    flow_cost_ag2ag    = {s: (v / economy_scale).astype(np.float32)     for s, v in flow_cost_ag2ag.items()}
    flow_cost_ag2nonag = {s: {k: (a / economy_scale).astype(np.float32) for k, a in p.items()} for s, p in flow_cost_ag2nonag.items()}
    flow_cost_nonag2ag = {k: (v / economy_scale).astype(np.float32)     for k, v in flow_cost_nonag2ag.items()}

    [ag_q_mrp, non_ag_q_crk, ag_man_q_mrp],   demand_scale = rescale_lhs_rhs([ag_q_mrp, non_ag_q_crk, ag_man_q_mrp], limits['demand'])
    [ag_b_mrj, non_ag_b_rk, ag_man_b_mrj],    biodiv_scale = rescale_lhs([ag_b_mrj, non_ag_b_rk, ag_man_b_mrj])

    [ag_g_mrj, non_ag_g_rk, ag_man_g_mrj], ghg_scale = (
        rescale_lhs_rhs([ag_g_mrj, non_ag_g_rk, ag_man_g_mrj], limits['ghg'])
        if settings.GHG_EMISSIONS_LIMITS != 'off' else
        ([ag_g_mrj, non_ag_g_rk, ag_man_g_mrj], 1.0)
    )
    # Put the source-keyed transition-GHG dict on the same GHG rescale band as ag_g_mrj
    # (the GHG constraint sums Σ flow_ghg·D — the source-correct transition emissions).
    flow_ghg_ag2ag = {s: (v / ghg_scale).astype(np.float32) for s, v in flow_ghg_ag2ag.items()}

    [renewable_solar_r], renewable_solar_scale = (
        rescale_lhs_rhs([renewable_solar_r], limits['renewable_Utility Solar PV'])
        if any(settings.RENEWABLES_OPTIONS.values()) else
        ([renewable_solar_r], 1.0)
    )
    
    [renewable_wind_r], renewable_wind_scale = (
        rescale_lhs_rhs([renewable_wind_r], limits['renewable_Onshore Wind'])
        if any(settings.RENEWABLES_OPTIONS.values()) else
        ([renewable_wind_r], 1.0)
    )
    
    [ag_w_mrj, non_ag_w_rk, ag_man_w_mrj], water_scale = (
        rescale_lhs_rhs([ag_w_mrj, non_ag_w_rk, ag_man_w_mrj], limits['water'])
        if settings.WATER_LIMITS == 'on' else
        ([ag_w_mrj, non_ag_w_rk, ag_man_w_mrj], 1.0)
    )
    
    [GBF2_mask_area_r], gbf2_scale = (
        rescale_lhs_rhs([GBF2_mask_area_r], limits['GBF2'])
        if settings.GBF2_TARGET != "off" else
        ([GBF2_mask_area_r], 1.0)
    )
    
    GBF3_NVIS_pre_1750_area_vr, gbf3_nvis_scale = (
        rescale_lhs_rhs_region_species(
            GBF3_NVIS_pre_1750_area_vr, GBF3_NVIS_region_group, region_NRM_names_r,
            targets=limits['GBF3_NVIS'],
        )
        if settings.GBF3_NVIS_TARGET != "off" else
        (GBF3_NVIS_pre_1750_area_vr, 1.0)
    )
    
    GBF4_SNES_pre_1750_area_sr, gbf4_snes_scale = (
        rescale_lhs_rhs_region_species(
            GBF4_SNES_pre_1750_area_sr, GBF4_SNES_region_species, region_NRM_names_r,
            targets=limits['GBF4_SNES'],
            layer_coord_names=('region', 'species', 'presence'),
            matrix_key_fn=lambda t: (t[1], t[2]),
        )
        if settings.GBF4_TARGET_SNES != 'off' else
        (GBF4_SNES_pre_1750_area_sr, 1.0)
    )
    
    GBF4_ECNES_pre_1750_area_sr, gbf4_ecnes_scale = (
        rescale_lhs_rhs_region_species(
            GBF4_ECNES_pre_1750_area_sr, GBF4_ECNES_region_species, region_NRM_names_r,
            targets=limits['GBF4_ECNES'],
            layer_coord_names=('region', 'species', 'presence'),
            matrix_key_fn=lambda t: (t[1], t[2]),
        )
        if settings.GBF4_TARGET_ECNES != 'off' else
        (GBF4_ECNES_pre_1750_area_sr, 1.0)
    )
    
    GBF8_pre_1750_area_sr, gbf8_scale = (
        rescale_lhs_rhs_region_species(
            GBF8_pre_1750_area_sr, GBF8_region_species, region_NRM_names_r,
            targets=limits['GBF8'],
        )
        if settings.GBF8_TARGET != "off" else
        (GBF8_pre_1750_area_sr, 1.0)
    )

    scale_factors = {
        "Economy":                      economy_scale,
        "Demand":                       demand_scale,
        "Biodiversity":                 biodiv_scale,
        "GHG":                          ghg_scale,
        "Water":                        water_scale,
        "GBF2":                         gbf2_scale,
        "GBF3_NVIS":                    gbf3_nvis_scale,
        "GBF4_SNES":                    gbf4_snes_scale,
        "GBF4_ECNES":                   gbf4_ecnes_scale,
        "GBF8":                         gbf8_scale,
        "Utility Solar PV":             renewable_solar_scale,
        "Onshore Wind":                 renewable_wind_scale,
    }

    base_yr_prod = {
        "BASE_YR Economy(AUD)":         get_BASE_YR_economic_value(data),
        "BASE_YR Production (t)":       get_BASE_YR_production_t(data),
        "BASE_YR GHG (tCO2e)":          get_BASE_YR_GHG_t(data),
        "BASE_YR Water (ML)":           get_BASE_YR_water_ML(data),
        "BASE_YR Bio quality (score)":  get_BASE_YR_bio_quality_value(data),
        "BASE_YR GBF_2 (score)":        get_BASE_YR_GBF2_score(data),
    }

    commodity_names = data.COMMODITIES

    # Bookkeeping for which under theta sliver are folded into which domain cells.  
    ag_fold_map = ag_transition.get_ag_dvar_fold_map(data, base_year)   

    # Accounting support: X_acct is nonzero at EXACTLY feasible_ag_cells ∪ {folded-sliver cells}.
    # A folded sliver has folded base 0, so it is absent from feasible_ag_cells; the per-j accounting builders
    # (economy/GHG/GBF2/demand) must iterate THIS union or they'd drop its (slivers/fd)·Xf[d] term and silently
    # collapse the two-stream back to the folded stream. The sliver cells come straight from the fold map.
    acct_cells_mrj = {}
    for m in range(data.NLMS):
        for j in range(data.N_AG_LUS):
            feas = feasible_ag_cells_mrj[m, j]
            if ag_fold_map['cells'].size:
                extra = ag_fold_map['cells'][(ag_fold_map['from_m'] == m) & (ag_fold_map['from_j'] == j)]
                acct_cells_mrj[m, j] = np.union1d(feas, extra) if extra.size else feas
            else:
                acct_cells_mrj[m, j] = feas

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
    desc2aglu=data.DESC2AGLU
    real_area=data.REAL_AREA
    ag_mask_proportion_r=data.AG_MASK_PROPORTION_R
    
    # Base year dvars
    # Base dvars are the node-balance "stay" constant; clip them into the cleaned [lb, ub] box so the
    # all-delta=0 stay point is feasible by construction (fixes base's own float noise, e.g. -1e-8<lb=0).
    # Bounds were already clamped so lb ≤ base ≤ ub for real values — this only bites on noise. Reported.
    dvar_base_ag_mrj    = tools.clamp_dvar_bound(ag_transition.get_folded_base_ag_dvar(data, base_year), dvar_lb_ag, dvar_ub_ag, 'Ag base clipped to [lb,ub]')
    dvar_base_non_ag_rk = tools.clamp_dvar_bound(data.non_ag_dvars[base_year], dvar_lb_nonag, dvar_ub_nonag, 'NonAg base clipped to [lb,ub]')


    return SolverInputData(
        base_year,
        target_year,

        ag_g_mrj,
        ag_w_mrj,
        ag_b_mrj,
        ag_x_mrj,
        ag_q_mrp,

        non_ag_g_rk,
        non_ag_w_rk,
        non_ag_b_rk,
        non_ag_q_crk,

        ag_man_g_mrj,
        ag_man_w_mrj,
        ag_man_b_mrj,
        ag_man_q_mrp,
        ag_man_limits,
        ag_man_lb_mrj,

        dvar_base_ag_mrj,
        dvar_base_non_ag_rk,
        
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
        ag_fold_map,
        acct_cells_mrj,
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
        commodity_names,

        economic_contr_mrj,
        economic_prices,
        economic_target_yr_carbon_price,
        
        offland_ghg,
        
        lu2pr_pj,
        pr2cm_cp,
        limits,
        desc2aglu,
        real_area,
        ag_mask_proportion_r,

        # FROM-view source-cell maps (anchor the per-source flow vars).
        ag_source_cells,
        nonag_source_cells,

        # Source-keyed flow transition costs.
        flow_cost_ag2ag,
        flow_cost_ag2nonag,
        flow_cost_nonag2ag,
        flow_ghg_ag2ag,

        # Per-source delta-var feasibility (keyed/shaped like the flow_cost dicts).
        feasible_ag2ag_mrj,
        feasible_nonag2ag_mrj,
        feasible_ag2nonag_rk,

        # TO-view ag target bounds + eligibility.
        dvar_ub_ag,
        dvar_lb_ag,
        feasible_ag_cells_mrj,

        # TO-view non-ag target bounds + eligibility.
        dvar_ub_nonag,
        dvar_lb_nonag,
        feasible_non_ag_cells,
    )

