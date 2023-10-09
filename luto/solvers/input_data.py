from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
import numpy as np

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


@dataclass
class InputData:
    """
    An object that collects and stores all relevant data for solver.py.
    """

    ag_t_mrj: np.ndarray  # Agricultural transition cost matrices.
    ag_c_mrj: np.ndarray  # Agricultural production cost matrices.
    ag_r_mrj: np.ndarray  # Agricultural production revenue matrices.
    ag_g_mrj: np.ndarray  # Agricultural greenhouse gas emissions matrices.
    ag_w_mrj: np.ndarray  # Agricultural water requirements matrices.
    ag_x_mrj: np.ndarray  # Agricultural exclude matrices.
    ag_q_mrp: np.ndarray  # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
    ag_ghg_t_mrj: np.ndarray  # GHG emissions released during transitions between agricultural land uses.

    ag_to_non_ag_t_rk: np.ndarray  # Agricultural to non-agricultural transition cost matrix.
    non_ag_to_ag_t_mrj: np.ndarray  # Non-agricultural to agricultural transition cost matrices.
    non_ag_c_rk: np.ndarray  # Non-agricultural production cost matrix.
    non_ag_r_rk: np.ndarray  # Non-agricultural revenue matrix.
    non_ag_g_rk: np.ndarray  # Non-agricultural greenhouse gas emissions matrix.
    non_ag_w_rk: np.ndarray  # Non-agricultural water requirements matrix.
    non_ag_x_rk: np.ndarray  # Non-agricultural exclude matrices.
    non_ag_q_crk: np.ndarray  # Non-agricultural yield matrix.

    ag_man_c_mrj: np.ndarray  # Agricultural management options' cost effects.
    ag_man_g_mrj: np.ndarray  # Agricultural management options' GHG emission effects.
    ag_man_q_mrp: np.ndarray  # Agricultural management options' quantity effects.
    ag_man_r_mrj: np.ndarray  # Agricultural management options' revenue effects.
    ag_man_t_mrj: np.ndarray  # Agricultural management options' transition cost effects.
    ag_man_w_mrj: np.ndarray  # Agricultural management options' water requirement effects.
    ag_man_limits: np.ndarray  # Agricultural management options' adoption limits.

    lu2pr_pj: np.ndarray  # Conversion matrix: land-use to product(s).
    pr2cm_cp: np.ndarray  # Conversion matrix: product(s) to commodity.
    limits: dict  # Targets to use.
    desc2aglu: dict  # Map of agricultural land use descriptions to codes.

    cell_clusters: dict[
        int, list[int]
    ]  # Map of representative cell to list of cells in cluster

    def apply_clusters(self, new_cell_clusters: dict[int, list[int]]):
        self.cell_clusters = new_cell_clusters

        for repr_cell, cell_list in new_cell_clusters.items():
            if len(cell_list) == 1:
                continue

            cluster_size = len(cell_list)
            self.ag_t_mrj[:, repr_cell, :] = cluster_size * self.ag_t_mrj[:, repr_cell, :]
            self.ag_c_mrj[:, repr_cell, :] = cluster_size * self.ag_c_mrj[:, repr_cell, :]
            self.ag_r_mrj[:, repr_cell, :] = cluster_size * self.ag_r_mrj[:, repr_cell, :]
            self.ag_g_mrj[:, repr_cell, :] = cluster_size * self.ag_g_mrj[:, repr_cell, :]
            self.ag_w_mrj[:, repr_cell, :] = cluster_size * self.ag_w_mrj[:, repr_cell, :]
            self.ag_x_mrj[:, repr_cell, :] = cluster_size * self.ag_x_mrj[:, repr_cell, :]
            self.ag_q_mrp[:, repr_cell, :] = cluster_size * self.ag_q_mrp[:, repr_cell, :]
            self.ag_ghg_t_mrj[:, repr_cell, :] = cluster_size * self.ag_ghg_t_mrj[:, repr_cell, :]
            self.ag_to_non_ag_t_rk[repr_cell, :] = cluster_size * self.ag_to_non_ag_t_rk[repr_cell, :]
            self.non_ag_to_ag_t_mrj[:, repr_cell, :] = cluster_size * self.non_ag_to_ag_t_mrj[:, repr_cell, :]
            self.non_ag_c_rk[repr_cell, :] = cluster_size * self.non_ag_c_rk[repr_cell, :]
            self.non_ag_r_rk[repr_cell, :] = cluster_size * self.non_ag_r_rk[repr_cell, :]
            self.non_ag_g_rk[repr_cell, :] = cluster_size * self.non_ag_g_rk[repr_cell, :]
            self.non_ag_w_rk[repr_cell, :] = cluster_size * self.non_ag_w_rk[repr_cell, :]
            self.non_ag_x_rk[repr_cell, :] = cluster_size * self.non_ag_x_rk[repr_cell, :]
            self.non_ag_q_crk[:, repr_cell, :] = cluster_size * self.non_ag_q_crk[:, repr_cell, :]
            # self.ag_man_c_mrj[:, repr_cell, :] = cluster_size * self.ag_man_c_mrj[:, repr_cell, :]
            # self.ag_man_g_mrj[:, repr_cell, :] = cluster_size * self.ag_man_g_mrj[:, repr_cell, :]
            # self.ag_man_q_mrp[:, repr_cell, :] = cluster_size * self.ag_man_q_mrp[:, repr_cell, :]
            # self.ag_man_r_mrj[:, repr_cell, :] = cluster_size * self.ag_man_r_mrj[:, repr_cell, :]
            # self.ag_man_t_mrj[:, repr_cell, :] = cluster_size * self.ag_man_t_mrj[:, repr_cell, :]
            # self.ag_man_w_mrj[:, repr_cell, :] = cluster_size * self.ag_man_w_mrj[:, repr_cell, :]

            # exlude non-representative cells from exclusion matrices
            for cell in cell_list:
                if cell == repr_cell:
                    continue

                self.ag_x_mrj[:, cell, :] = 0
                self.non_ag_x_rk[cell, :] = 0


    @property
    def n_ag_lms(self):
        # Number of agricultural landmans
        return self.ag_t_mrj.shape[0]

    @property
    def ncells(self):
        # Number of clusters
        return self.ag_t_mrj.shape[1]

    @property
    def nclusters(self):
        # Number of clusters
        return len(self.cell_clusters)

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
    def non_ag_lu2cells(self):
        return {
            k: np.where(self.non_ag_x_rk[:, k])[0] for k in range(self.n_non_ag_lus)
        }
