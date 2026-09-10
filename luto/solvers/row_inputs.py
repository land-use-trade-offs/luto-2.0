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

from dataclasses import dataclass
from typing import Any

from luto.data import Data
from luto import settings

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


# ═══════════════════════════ RowInputs: what one step hands to the row families ═══════════════════════════

@dataclass
class RowInputs:
    base_year: int                                                      # Base year of this solve step.
    target_year: int                                                    # Target year of this solve step.

    ag_g_mrj: np.ndarray                                                # Agricultural GHG emissions [m, r, j].
    ag_w_mrj: np.ndarray                                                # Agricultural water net yield [m, r, j].
    ag_q_mrp: np.ndarray                                                # Agricultural production quantity [m, r, p] (p = product, not land use).

    non_ag_g_rk: np.ndarray                                             # Non-agricultural GHG emissions [r, k].
    non_ag_w_rk: np.ndarray                                             # Non-agricultural water net yield [r, k].
    non_ag_q_crk: np.ndarray                                            # Non-agricultural production quantity [c, r, k].

    ag_man_g_mrj: dict                                                  # {am: GHG emission effect [m, r, j_idx]}.
    ag_man_w_mrj: dict                                                  # {am: water net-yield effect [m, r, j_idx]}.
    ag_man_q_mrp: dict                                                  # {am: production quantity effect [m, r, p]}.
    ag_man_limits: dict                                                 # {am: {j: adoption limit}}.

    renewable_solar_r: np.ndarray                                       # Renewable energy - solar yield matrix.
    renewable_wind_r: np.ndarray                                        # Renewable energy - wind yield matrix.
    exist_renewable_solar_r: np.ndarray                                 # Existing solar capacity converted to annual MWh per cell.
    exist_renewable_wind_r: np.ndarray                                  # Existing wind capacity converted to annual MWh per cell.

    region_state_r: np.ndarray                                          # Region state index for each cell.
    region_state_name2idx: dict[str, int]                               # Map of region state names to indices.
    region_NRM_names_r: np.ndarray                                      # Region NRM names for each cell.

    water_region_indices: dict[int, np.ndarray]                         # {region id: cell indices} of the water regions.
    water_region_names: dict[int, str]                                  # {region id: region name}.

    biodiv_contr_ag_j: np.ndarray                                       # Biodiversity contribution scale per agricultural land use (j).
    biodiv_contr_non_ag_k: dict[int, float]                             # Biodiversity contribution scale per non-agricultural land use (k).
    biodiv_contr_ag_man: dict[str, dict[int, np.ndarray]]               # Biodiversity contribution scale per ag-management option and land use, per cell.

    GBF2_mask_area_r: np.ndarray                                        # GBF2 priority-degraded-area mask × real area, per cell (r).
    GBF3_NVIS_pre_1750_area_vr: np.ndarray                              # GBF3 pre-1750 NVIS vegetation area, per group (v) and cell (r).
    GBF3_NVIS_region_group: list                                        # GBF3 constraint pairs - list[(region, group)].
    GBF4_SNES_pre_1750_area_sr: xr.DataArray                            # GBF4 SNES pre-1750 area [layer=(species, presence), cell]; region masking happens in the solver.
    GBF4_SNES_region_species: list                                      # GBF4 SNES constraint triplets - list[(region, species, presence)].
    GBF4_ECNES_pre_1750_area_sr: xr.DataArray                           # GBF4 ECNES pre-1750 area [layer=(community, presence), cell]; region masking happens in the solver.
    GBF4_ECNES_region_species: list                                     # GBF4 ECNES constraint triplets - list[(region, community, presence)].
    GBF8_pre_1750_area_sr: xr.DataArray                                 # GBF8 pre-1750 species area [species, cell].
    GBF8_region_species: list                                           # GBF8 constraint pairs - list[(region, species)].

    commodity_names: list[str]                                          # Commodity names (data.COMMODITIES order).
    offland_ghg: np.ndarray                                             # Target-year GHG emissions from off-land commodities (tCO2e); 0.0 when GHG limits are off.
    lu2pr_pj: np.ndarray                                                # Conversion matrix: product (p) × land use (j).
    pr2cm_cp: np.ndarray                                                # Conversion matrix: commodity (c) × product (p).
    limits: dict                                                        # Raw constraint targets for the target year (see get_limits).
    real_area: np.ndarray                                               # Area of each cell (ha), per cell (r).
    ag_mask_proportion_r: np.ndarray                                    # Base-year (2010) agricultural proportion of each cell (r).

    trans_ghg_ag2ag: dict                                               # {(from_m, from_j): ndarray[to_m, local_r, to_j]} transition emissions, raw tCO2e, float32.

    @property
    def ncms(self):
        return len(self.commodity_names)


# ═══════════════════════════ the year's targets ═══════════════════════════

def get_limits(data: Data, yr_cal: int) -> dict[str, Any]:
    """The raw (unscaled) constraint targets of one calendar year — which keys are present depends on the
    active settings. The solver rescales each row together with its target (``row_builder.scale_rows``)."""
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

        print('Getting existing renewable capacity by state...', flush=True)
        renewable_existing_capacity = ag_quantity.get_exist_renewable_capacity_by_state(data, yr_cal)
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


# ═══════════════════════════ get_row_inputs: the row side's input data of one step ═══════════════════════════

def get_row_inputs(data: Data, base_year: int, target_year: int) -> RowInputs:
    """Every coefficient stream and target the row families read, loaded as one ``RowInputs`` — purely
    downstream of the economics modules and ``Data``; the column space is never seen here."""

    target_index = target_year - data.YR_CAL_BASE
    base_lumap = data.lumaps[base_year]

    # ── 1. transition GHG emissions, SOURCE-KEYED over each source's base-year cells: the physical
    #       parallel of the per-arc transition cost, so the emissions of a move are charged against
    #       its own source. The GHG row sums Σ flow_ghg·D; an arc's ``local_r`` indexes this cell axis.
    trans_ghg_ag2ag = {
        src: arr.astype(np.float32)
        for src, arr in ag_ghg.get_ghg_transition_emissions_from_base_year(data, base_year).items()
    }

    # ── 2. the physical streams the policy rows weight: GHG, water net yield, production quantity ──
    hist_water_yield = () if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on' else (data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)

    print('Getting agricultural GHG emissions matrices...', flush = True)
    ag_g_mrj = ag_ghg.get_ghg_matrices(data, target_index).astype(np.float32)

    print('Getting agricultural water net yield matrices based on historical water yield layers ...', flush = True)
    ag_w_mrj = ag_water.get_water_net_yield_matrices(data, target_index, *hist_water_yield).astype(np.float32)

    print('Getting agricultural production quantity matrices...', flush = True)
    ag_q_mrp = ag_quantity.get_quantity_matrices(data, target_index).astype(np.float32)

    print('Getting non-agricultural GHG emissions matrices...', flush = True)
    non_ag_g_rk = non_ag_ghg.get_ghg_matrix(data, ag_g_mrj, base_lumap).astype(np.float32)

    print('Getting non-agricultural water yield matrices...', flush = True)
    non_ag_w_rk = non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj, base_lumap, target_index, *hist_water_yield).astype(np.float32)

    print('Getting non-agricultural production quantity matrices...', flush = True)
    non_ag_q_crk = non_ag_quantity.get_quantity_matrix(data, ag_q_mrp, base_lumap).astype(np.float32)

    print("Getting agricultural management options' GHG emission effects...", flush = True)
    ag_man_g_mrj = {am: arr.astype(np.float32) for am, arr in ag_ghg.get_agricultural_management_ghg_matrices(data, target_index).items()}

    print("Getting agricultural management options' water yield effects...", flush = True)
    ag_man_w_mrj = {am: arr.astype(np.float32) for am, arr in ag_water.get_agricultural_management_water_matrices(data, target_index).items()}

    print("Getting agricultural management options' quantity effects...", flush = True)
    ag_man_q_mrp = {am: arr.astype(np.float32) for am, arr in ag_quantity.get_agricultural_management_quantity_matrices(data, ag_q_mrp, target_index).items()}

    print("Getting agricultural management options' adoption limits...", flush = True)
    ag_man_limits = ag_transition.get_agricultural_management_adoption_limits(data, target_index)

    # ── 3. renewable energy: the yields the state targets are met with, and the capacity already in the ground ──
    print('Getting renewable energy - solar yield matrix...', flush = True)
    renewable_solar_r = ag_quantity.get_quantity_renewable(data, 'Utility Solar PV', target_index).astype(np.float32)

    print('Getting renewable energy - wind yield matrix...', flush = True)
    renewable_wind_r = ag_quantity.get_quantity_renewable(data, 'Onshore Wind', target_index).astype(np.float32)

    # Existing real-world capacity and LUTO-simulated capacity compete for the same cell space [0, 1].
    # The maximum existing fraction (cumulative over ALL data years, hence yr_cal=99999) is locked in
    # advance, so the ceiling never decreases between periods and simulated + existing never exceeds 1.
    print('Getting existing solar capacity fraction (all years, solver ceiling)...', flush=True)
    exist_renewable_solar_r = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Utility Solar PV', 99999)

    print('Getting existing wind capacity fraction (all years, solver ceiling)...', flush=True)
    exist_renewable_wind_r = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Onshore Wind', 99999)

    # ── 4. regions: what the renewable, biodiversity and water rows group cells by ──
    region_state_r = data.REGION_STATE_CODE
    region_state_name2idx = data.REGION_STATE_NAME2CODE
    region_NRM_names_r = data.REGION_NRM_NAME
    water_region_indices = data.WATER_REGION_INDEX_R if settings.WATER_LIMITS != 'off' else {}
    water_region_names = data.WATER_REGION_NAMES if settings.WATER_LIMITS != 'off' else {}

    # ── 5. biodiversity: the contribution scales every GBF family shares, then each family's layer and selection ──
    print('Getting biodiversity degredation data for agricultural land uses...', flush = True)
    biodiv_contr_ag_j = ag_biodiversity.get_ag_biodiversity_contribution(data)

    print('Getting biodiversity benefits data for non-agricultural land uses...', flush = True)
    biodiv_contr_non_ag_k = non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data)

    print('Getting biodiversity benefits data for agricultural management options...', flush = True)
    biodiv_contr_ag_man = ag_biodiversity.get_ag_management_biodiversity_contribution(data, target_year)

    # each getter owns the setting that turns its family off, and returns an empty layer when it is
    print('Getting GBF2 mask area layer...', flush = True)
    GBF2_mask_area_r = ag_biodiversity.get_GBF2_MASK_area(data)

    print('Getting GBF3 NVIS vegetation matrices...', flush = True)
    GBF3_NVIS_pre_1750_area_vr = ag_biodiversity.get_GBF3_NVIS_matrices_vr(data)

    print('Getting GBF4 SNES species area matrices...', flush = True)
    GBF4_SNES_pre_1750_area_sr = ag_biodiversity.get_GBF4_SNES_matrix_sr(data)

    print('Getting GBF4 ECNES community area matrices...', flush = True)
    GBF4_ECNES_pre_1750_area_sr = ag_biodiversity.get_GBF4_ECNES_matrix_sr(data)

    print('Getting GBF8 species conservation area matrices...', flush = True)
    GBF8_pre_1750_area_sr = ag_biodiversity.get_GBF8_matrix_sr(data, target_year)

    # the (region, item) each family constrains — a Data attribute that only exists while the family is on
    GBF3_NVIS_region_group    = data.BIO_GBF3_NVIS_SEL  if settings.GBF3_NVIS_TARGET  != 'off' else {}
    GBF4_SNES_region_species  = data.BIO_GBF4_SNES_SEL  if settings.GBF4_TARGET_SNES  != 'off' else []
    GBF4_ECNES_region_species = data.BIO_GBF4_ECNES_SEL if settings.GBF4_TARGET_ECNES != 'off' else []
    GBF8_region_species       = data.BIO_GBF8_SEL       if settings.GBF8_TARGET       != 'off' else []

    # ── 6. the year's targets, and the off-land emissions the GHG cap must leave room for ──
    limits = get_limits(data, target_year)
    offland_ghg = (
        data.OFF_LAND_GHG_EMISSION_C[target_index]                         # raw tCO2e (row-rescaled in the solver)
        if settings.GHG_EMISSIONS_LIMITS != 'off'
        else 0.0
    )

    return RowInputs(
        base_year=base_year,
        target_year=target_year,

        ag_g_mrj=ag_g_mrj,
        ag_w_mrj=ag_w_mrj,
        ag_q_mrp=ag_q_mrp,
        non_ag_g_rk=non_ag_g_rk,
        non_ag_w_rk=non_ag_w_rk,
        non_ag_q_crk=non_ag_q_crk,
        ag_man_g_mrj=ag_man_g_mrj,
        ag_man_w_mrj=ag_man_w_mrj,
        ag_man_q_mrp=ag_man_q_mrp,
        ag_man_limits=ag_man_limits,

        renewable_solar_r=renewable_solar_r,
        renewable_wind_r=renewable_wind_r,
        exist_renewable_solar_r=exist_renewable_solar_r,
        exist_renewable_wind_r=exist_renewable_wind_r,

        region_state_r=region_state_r,
        region_state_name2idx=region_state_name2idx,
        region_NRM_names_r=region_NRM_names_r,
        water_region_indices=water_region_indices,
        water_region_names=water_region_names,

        biodiv_contr_ag_j=biodiv_contr_ag_j,
        biodiv_contr_non_ag_k=biodiv_contr_non_ag_k,
        biodiv_contr_ag_man=biodiv_contr_ag_man,

        GBF2_mask_area_r=GBF2_mask_area_r,
        GBF3_NVIS_pre_1750_area_vr=GBF3_NVIS_pre_1750_area_vr,
        GBF3_NVIS_region_group=GBF3_NVIS_region_group,
        GBF4_SNES_pre_1750_area_sr=GBF4_SNES_pre_1750_area_sr,
        GBF4_SNES_region_species=GBF4_SNES_region_species,
        GBF4_ECNES_pre_1750_area_sr=GBF4_ECNES_pre_1750_area_sr,
        GBF4_ECNES_region_species=GBF4_ECNES_region_species,
        GBF8_pre_1750_area_sr=GBF8_pre_1750_area_sr,
        GBF8_region_species=GBF8_region_species,

        commodity_names=data.COMMODITIES,
        offland_ghg=offland_ghg,
        lu2pr_pj=data.LU2PR,
        pr2cm_cp=data.PR2CM,
        limits=limits,
        real_area=data.REAL_AREA,
        ag_mask_proportion_r=data.AG_MASK_PROPORTION_R,
        trans_ghg_ag2ag=trans_ghg_ag2ag,
    )


# ═══════════════════════════ EconomicInputs: what the objective is built from ═══════════════════════════
#
# Raw AUD, float32: operating economics per column, and the transition cost of every arc keyed by its
# base-year source. ~300 MB at RES5, and wanted only while row_builder.get_obj turns it into the objective
# coefficient of every column — so it is loaded on its own and dropped, never carried on a RowInputs through the solve.

@dataclass
class EconomicInputs:
    ag_obj_mrj: np.ndarray                                            # Operating economics per agricultural (m, r, j).
    non_ag_obj_rk: np.ndarray                                           # Operating economics per non-agricultural (r, k).
    ag_man_objs: dict                                                   # {am: operating economics of the option [m, r, j_idx]}.
    flow_cost_ag2ag: dict                                               # {(from_m, from_j): ndarray[to_m, local_r, to_j]}.
    flow_cost_ag2nonag: dict                                            # {(from_m, from_j): {to_k: ndarray[local_r]}}.
    flow_cost_nonag2ag: dict                                            # {from_k: ndarray[to_m, local_r, to_j]}.


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


def get_economics(data: Data, base_year: int, target_year: int) -> EconomicInputs:
    """Cost, revenue and the per-arc transition cost of one step, as one ``EconomicInputs``."""

    target_index = target_year - data.YR_CAL_BASE
    base_lumap = data.lumaps[base_year]

    print('Getting agricultural cost matrices...', flush = True)
    ag_c_mrj = ag_cost.get_cost_matrices(data, target_index).astype(np.float32)

    print('Getting agricultural revenue matrices...', flush = True)
    ag_r_mrj = ag_revenue.get_rev_matrices(data, target_index).astype(np.float32)

    print('Getting non-agricultural cost matrices...', flush = True)
    non_ag_c_rk = non_ag_cost.get_cost_matrix(data, ag_c_mrj, base_lumap, target_year).astype(np.float32)

    print('Getting non-agricultural revenue matrices...', flush = True)
    non_ag_r_rk = non_ag_revenue.get_rev_matrix(data, target_year, ag_r_mrj, base_lumap).astype(np.float32)

    # nonag→nonag transition cost. Currently a ZERO matrix — non-ag LUs are not allowed to transition
    # to other non-ag LUs (get_nonag2nonag_transition_matrix returns zeros). Kept as an explicit hook
    # so the objective wiring is ready if non-ag↔non-ag transitions are ever priced.
    print('Getting non-agricultural transition cost matrices...', flush = True)
    non_ag_t_rk = non_ag_transition.get_nonag2nonag_transition_matrix(data)

    print("Getting agricultural management options' cost effects...", flush = True)
    ag_man_c_mrj = ag_cost.get_agricultural_management_cost_matrices(data, ag_c_mrj, target_year)

    print("Getting agricultural management options' revenue effects...", flush = True)
    ag_man_r_mrj = ag_revenue.get_agricultural_management_revenue_matrices(data, ag_r_mrj, target_index)

    print("Getting agricultural management options' transition cost effects...", flush = True)
    ag_man_t_mrj = ag_transition.get_agricultural_management_transition_matrices(data, target_index)

    # operating economics only — revenue − production cost; the land-use transition cost is charged per arc
    ag_obj_mrj, non_ag_obj_rk, ag_man_objs = get_economic_mrj(
        ag_c_mrj, ag_r_mrj, non_ag_c_rk, non_ag_r_rk, non_ag_t_rk, ag_man_c_mrj, ag_man_r_mrj, ag_man_t_mrj)
    ag_obj_mrj = ag_obj_mrj.astype(np.float32)
    non_ag_obj_rk = non_ag_obj_rk.astype(np.float32)
    ag_man_objs = {am: arr.astype(np.float32) for am, arr in ag_man_objs.items()}

    # ── the per-arc transition cost, SOURCE-KEYED over each source's base-year cells ──
    # Keyed by the base-year source ("(from_m, from_j)" for ag, "k" for non-ag); the column space carries
    # a matching arc per (source, cell, target) and the objective charges Σ flow_cost·D. An arc's
    # ``local_r`` indexes the cell axis of its own source's array.

    # ag→ag: dict[(from_m, from_j)] → ndarray(NLMS, ncells_src, N_AG_LUS)
    print('Getting agricultural transition cost matrices...', flush = True)
    flow_cost_ag2ag = {
        (from_m, from_j): ag_transition.get_transition_matrices_ag2ag(data, target_index, from_m, from_j, cell_idx).astype(np.float32)
        for (from_m, from_j), cell_idx in ag_transition.get_base_dvar_mj_cell_map(data, base_year).items()
    }

    # ag→nonag: the dispatcher gives dict[lu_name → dict[(from_m, from_j)]]; transposed to
    # dict[(from_m, from_j) → dict[k]] so the arcs loop ag sources first.
    flow_cost_ag2nonag = {}
    for lu_name, per_src in non_ag_transition.get_transition_matrix_ag2nonag(data, base_year, target_year).items():
        k = data.NON_AGRICULTURAL_LANDUSES.index(lu_name)
        for src, arr in per_src.items():
            flow_cost_ag2nonag.setdefault(src, {})[k] = arr.astype(np.float32)

    # nonag→ag: the dispatcher gives dict[lu_name → dict[k]]; take the diagonal (cells in non-ag
    # land use k pay only k's own nonag→ag cost).
    flow_cost_nonag2ag = {}
    for lu_name, per_k in non_ag_transition.get_transition_matrix_nonag2ag(data, base_year, target_year).items():
        k = data.NON_AGRICULTURAL_LANDUSES.index(lu_name)
        if k in per_k:
            flow_cost_nonag2ag[k] = per_k[k].astype(np.float32)

    return EconomicInputs(
        ag_obj_mrj=ag_obj_mrj,
        non_ag_obj_rk=non_ag_obj_rk,
        ag_man_objs=ag_man_objs,
        flow_cost_ag2ag=flow_cost_ag2ag,
        flow_cost_ag2nonag=flow_cost_ag2nonag,
        flow_cost_nonag2ag=flow_cost_nonag2ag,
    )
