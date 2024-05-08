from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import numpy as np

from luto import settings
from luto.economics import land_use_culling
from luto.settings import AG_MANAGEMENTS
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
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

    ag_t_mrj: np.ndarray  # Agricultural transition cost matrices.
    ag_c_mrj: np.ndarray  # Agricultural production cost matrices.
    ag_r_mrj: np.ndarray  # Agricultural production revenue matrices.
    ag_g_mrj: np.ndarray  # Agricultural greenhouse gas emissions matrices.
    ag_w_mrj: np.ndarray  # Agricultural water requirements matrices.
    ag_b_mrj: np.ndarray  # Agricultural biodiversity matrices.
    ag_x_mrj: np.ndarray  # Agricultural exclude matrices.
    ag_q_mrp: np.ndarray  # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
    ag_ghg_t_mrj: np.ndarray  # GHG emissions released during transitions between agricultural land uses.
    ag_to_non_ag_t_rk: np.ndarray  # Agricultural to non-agricultural transition cost matrix.
    
    non_ag_to_ag_t_mrj: np.ndarray  # Non-agricultural to agricultural transition cost matrices.
    non_ag_t_rk: np.ndarray  # Non-agricultural transition costs matrix
    non_ag_c_rk: np.ndarray  # Non-agricultural production cost matrix.
    non_ag_r_rk: np.ndarray  # Non-agricultural revenue matrix.
    non_ag_g_rk: np.ndarray  # Non-agricultural greenhouse gas emissions matrix.
    non_ag_w_rk: np.ndarray  # Non-agricultural water requirements matrix.
    non_ag_b_rk: np.ndarray  # Non-agricultural biodiversity matrix.
    non_ag_x_rk: np.ndarray  # Non-agricultural exclude matrices.
    non_ag_q_crk: np.ndarray  # Non-agricultural yield matrix.
    non_ag_lb_rk: np.ndarray # Non-agricultural lower bound matrices.

    ag_man_c_mrj: dict  # Agricultural management options' cost effects.
    ag_man_g_mrj: dict  # Agricultural management options' GHG emission effects.
    ag_man_q_mrp: dict  # Agricultural management options' quantity effects.
    ag_man_r_mrj: dict  # Agricultural management options' revenue effects.
    ag_man_t_mrj: dict  # Agricultural management options' transition cost effects.
    ag_man_w_mrj: dict  # Agricultural management options' water requirement effects.
    ag_man_b_mrj: dict  # Agricultural management options' biodiversity effects.
    ag_man_limits: dict  # Agricultural management options' adoption limits.
    ag_man_lb_mrj: dict  # Agricultural management options' lower bounds.

    offland_ghg: np.ndarray  # GHG emissions from off-land commodities.

    lu2pr_pj: np.ndarray  # Conversion matrix: land-use to product(s).
    pr2cm_cp: np.ndarray  # Conversion matrix: product(s) to commodity.
    limits: dict  # Targets to use.
    desc2aglu: dict  # Map of agricultural land use descriptions to codes.
    resmult: float  # Resolution factor multiplier from data.RESMULT

    @property
    def n_ag_lms(self):
        # Number of agricultural landmans
        return self.ag_t_mrj.shape[0]

    @property
    def ncells(self):
        # Number of cells
        return self.ag_t_mrj.shape[1]

    @property
    def n_ag_lus(self):
        # Number of agricultural landuses
        return self.ag_t_mrj.shape[2]

    @property
    def n_non_ag_lus(self):
        # Number of non-agricultural landuses
        return self.non_ag_c_rk.shape[1]

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
    

def get_ag_c_mrj(data: Data, target_index):
    print('Getting agricultural production cost matrices...', flush = True)
    output = ag_cost.get_cost_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_c_rk(data: Data):
    print('Getting non-agricultural production cost matrices...', flush = True)
    output = non_ag_cost.get_cost_matrix(data)
    return output.astype(np.float32)


def get_ag_r_mrj(data: Data, target_index):
    print('Getting agricultural production revenue matrices...', flush = True)
    output = ag_revenue.get_rev_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_r_rk(data: Data):
    print('Getting non-agricultural production revenue matrices...', flush = True)
    output = non_ag_revenue.get_rev_matrix(data)
    return output.astype(np.float32)


def get_ag_g_mrj(data: Data, target_index):
    print('Getting agricultural GHG emissions matrices...', flush = True)
    output = ag_ghg.get_ghg_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_g_rk(data: Data):
    print('Getting non-agricultural GHG emissions matrices...', flush = True)
    output = non_ag_ghg.get_ghg_matrix(data)
    return output.astype(np.float32)


def get_ag_w_mrj(data: Data, target_index):
    print('Getting agricultural water requirement matrices...', flush = True)
    output = ag_water.get_wreq_matrices(data, target_index)
    return output.astype(np.float32)


def get_ag_b_mrj(data: Data):
    print('Getting agricultural biodiversity requirement matrices...', flush = True)
    output = ag_biodiversity.get_breq_matrices(data)
    return output.astype(np.float32)


def get_non_ag_w_rk(data: Data):
    print('Getting non-agricultural water requirement matrices...', flush = True)
    output = non_ag_water.get_wreq_matrix(data)
    return output.astype(np.float32)


def get_non_ag_b_rk(data: Data):
    print('Getting non-agricultural biodiversity requirement matrices...', flush = True)
    output = non_ag_biodiversity.get_breq_matrix(data)
    return output.astype(np.float32)


def get_ag_q_mrp(data: Data, target_index):
    print('Getting agricultural production quantity matrices...', flush = True)
    output = ag_quantity.get_quantity_matrices(data, target_index)
    return output.astype(np.float32)


def get_non_ag_q_crk(data: Data):
    print('Getting non-agricultural production quantity matrices...', flush = True)
    output = non_ag_quantity.get_quantity_matrix(data)
    return output.astype(np.float32)


def get_ag_t_mrj(data: Data, target_index, base_year):
    print('Getting agricultural transition cost matrices...', flush = True)
    output = ag_transition.get_transition_matrices( data
                                                  , target_index
                                                  , base_year
                                                  , data.lumaps
                                                  , data.lmmaps)
    return output.astype(np.float32)


def get_ag_ghg_t_mrj(data: Data, base_year):
    print('Getting agricultural transitions GHG emissions...', flush = True)
    output = ag_ghg.get_ghg_transition_penalties(data, data.lumaps[base_year])
    return output.astype(np.float32)


def get_ag_to_non_ag_t_rk(data: Data, base_year):
    print('Getting agricultural to non-agricultural transition cost matrices...', flush = True)
    output = non_ag_transition.get_from_ag_transition_matrix(data
                                                           , base_year
                                                           , data.lumaps[base_year]
                                                           , data.lmmaps[base_year])
    return output.astype(np.float32)


def get_non_ag_to_ag_t_mrj(data: Data, base_year):
    print('Getting non-agricultural to agricultural transition cost matrices...', flush = True)
    output = non_ag_transition.get_to_ag_transition_matrix(data, base_year, data.lumaps[base_year], data.lmmaps[base_year])
    return output.astype(np.float32)


def get_non_ag_t_rk(data: Data, base_year):
    print('Getting non-agricultural transition cost matrices...', flush = True)
    output = non_ag_transition.get_non_ag_transition_matrix(data, base_year, data.lumaps[base_year], data.lmmaps[base_year])
    return output.astype(np.float32)


def get_ag_x_mrj(data: Data, base_year):
    print('Getting agricultural exclude matrices...', flush = True)
    output = ag_transition.get_exclude_matrices(data, base_year, data.lumaps)
    return output


def get_non_ag_x_rk(data: Data, base_year):
    print('Getting non-agricultural exclude matrices...', flush = True)
    output = non_ag_transition.get_exclude_matrices(data, data.lumaps[base_year])
    return output


def get_ag_man_lb_mrj(data: Data, base_year):
    print('Getting agricultural lower bound matrices...', flush = True)
    output = ag_transition.get_lower_bound_agricultural_management_matrices(data, base_year)
    return output


def get_non_ag_lb_rk(data: Data, base_year):
    print('Getting non-agricultural lower bound matrices...', flush = True)
    output = non_ag_transition.get_lower_bound_non_agricultural_matrices(data, base_year)
    return output


def get_ag_man_costs(data: Data, target_index, ag_c_mrj: np.ndarray):
    print('Getting agricultural management options\' cost effects...', flush = True)
    output = ag_cost.get_agricultural_management_cost_matrices(data, ag_c_mrj, target_index)
    return output


def get_ag_man_ghg(data: Data, target_index, ag_g_mrj):
    print('Getting agricultural management options\' GHG emission effects...', flush = True)
    output = ag_ghg.get_agricultural_management_ghg_matrices(data, ag_g_mrj, target_index)
    return output


def get_ag_man_quantity(data: Data, target_index, ag_q_mrp):
    print('Getting agricultural management options\' quantity effects...', flush = True)
    output = ag_quantity.get_agricultural_management_quantity_matrices(data, ag_q_mrp, target_index)
    return output


def get_ag_man_revenue(data: Data, target_index, ag_r_mrj):
    print('Getting agricultural management options\' revenue effects...', flush = True)
    output = ag_revenue.get_agricultural_management_revenue_matrices(data, ag_r_mrj, target_index)
    return output


def get_ag_man_transitions(data: Data, target_index, ag_t_mrj):
    print('Getting agricultural management options\' transition cost effects...', flush = True)
    output = ag_transition.get_agricultural_management_transition_matrices(data, ag_t_mrj, target_index)
    return output


def get_ag_man_water(data: Data, target_index, ag_w_mrj):
    print('Getting agricultural management options\' water requirement effects...', flush = True)
    output = ag_water.get_agricultural_management_water_matrices(data, ag_w_mrj, target_index)
    return output


def get_ag_man_biodiversity(data: Data):
    print('Getting agricultural management options\' biodiversity effects...', flush = True)
    output = ag_biodiversity.get_agricultural_management_biodiversity_matrices(data)
    return output


def get_ag_man_limits(data: Data, target_index):
    print('Getting agricultural management options\' adoption limits...', flush = True)
    output = ag_transition.get_agricultural_management_adoption_limits(data, target_index)
    return output


def get_limits(data: Data, target: int):
    print('Getting environmental limits...', flush = True)
    # Limits is a dictionary with heterogeneous value sets.
    limits = {}
    
    if settings.WATER_USE_LIMITS == 'on': limits['water'] = ag_water.get_wuse_limits(data)
    if settings.GHG_EMISSIONS_LIMITS == 'on':  limits['ghg'] = ag_ghg.get_ghg_limits(data, target)
    if settings.BIODIVERSITY_LIMITS == 'on':  limits['biodiversity'] = ag_biodiversity.get_biodiversity_limits(data, target)
    
    return limits


def get_input_data(data: Data, base_year: int, target_year: int) -> SolverInputData:
    """
    Using the given Data object, prepare a SolverInputData object for the solver.
    """
    target_index = target_year - data.YR_CAL_BASE

    ag_c_mrj = get_ag_c_mrj(data, target_index)
    ag_g_mrj = get_ag_g_mrj(data, target_index)
    ag_q_mrp = get_ag_q_mrp(data, target_index)
    ag_r_mrj = get_ag_r_mrj(data, target_index)
    ag_t_mrj = get_ag_t_mrj(data, target_index, base_year)
    ag_w_mrj = get_ag_w_mrj(data, target_index)
    ag_b_mrj = get_ag_b_mrj(data)
    ag_x_mrj = get_ag_x_mrj(data, base_year)

    land_use_culling.apply_agricultural_land_use_culling(
        ag_x_mrj, ag_c_mrj, ag_t_mrj, ag_r_mrj
    )

    return SolverInputData(
        ag_t_mrj=ag_t_mrj,
        ag_c_mrj=ag_c_mrj,
        ag_r_mrj=ag_r_mrj,
        ag_g_mrj=ag_g_mrj,
        ag_w_mrj=ag_w_mrj,
        ag_b_mrj=ag_b_mrj,
        ag_x_mrj=ag_x_mrj,
        ag_q_mrp=ag_q_mrp,
        ag_ghg_t_mrj=get_ag_ghg_t_mrj(data, base_year),
        ag_to_non_ag_t_rk=get_ag_to_non_ag_t_rk(data, base_year),
        non_ag_to_ag_t_mrj=get_non_ag_to_ag_t_mrj(data, base_year),
        non_ag_t_rk=get_non_ag_t_rk(data, base_year),
        non_ag_c_rk=get_non_ag_c_rk(data),
        non_ag_r_rk=get_non_ag_r_rk(data),
        non_ag_g_rk=get_non_ag_g_rk(data),
        non_ag_w_rk=get_non_ag_w_rk(data),
        non_ag_b_rk=get_non_ag_b_rk(data),
        non_ag_x_rk=get_non_ag_x_rk(data, base_year),
        non_ag_q_crk=get_non_ag_q_crk(data),
        non_ag_lb_rk=get_non_ag_lb_rk(data, base_year),
        ag_man_c_mrj=get_ag_man_costs(data, target_index, ag_c_mrj),
        ag_man_g_mrj=get_ag_man_ghg(data, target_index, ag_g_mrj),
        ag_man_q_mrp=get_ag_man_quantity(data, target_index, ag_q_mrp),
        ag_man_r_mrj=get_ag_man_revenue(data, target_index, ag_r_mrj),
        ag_man_t_mrj=get_ag_man_transitions(data, target_index, ag_t_mrj),
        ag_man_w_mrj=get_ag_man_water(data, target_index, ag_w_mrj),
        ag_man_b_mrj=get_ag_man_biodiversity(data),
        ag_man_limits=get_ag_man_limits(data, target_index),
        ag_man_lb_mrj=get_ag_man_lb_mrj(data, base_year),
        offland_ghg=data.OFF_LAND_GHG_EMISSION_C[target_index],
        lu2pr_pj=data.LU2PR,
        pr2cm_cp=data.PR2CM,
        limits=get_limits(data, target_year),
        desc2aglu=data.DESC2AGLU,
        resmult=data.RESMULT,
    )
