from collections import defaultdict
from dataclasses import dataclass, fields
from luto.solvers.input_data import InputData
from luto import settings

import numpy as np


@dataclass
class ClusterKey:
    ag_t_mrj: np.ndarray  # Agricultural transition cost matrices.
    ag_c_mrj: np.ndarray  # Agricultural production cost matrices.

    # TODO: OTHER DIMENSIONS:
    # ag_r_mrj: np.ndarray  # Agricultural production revenue matrices.
    # ag_g_mrj: np.ndarray  # Agricultural greenhouse gas emissions matrices.
    # ag_w_mrj: np.ndarray  # Agricultural water requirements matrices.
    # ag_x_mrj: np.ndarray  # Agricultural exclude matrices.
    # ag_q_mrp: np.ndarray  # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
    # ag_ghg_t_mrj: np.ndarray  # GHG emissions released during transitions between agricultural land uses.

    # ag_to_non_ag_t_rk: np.ndarray  # Agricultural to non-agricultural transition cost matrix.
    # non_ag_to_ag_t_mrj: np.ndarray  # Non-agricultural to agricultural transition cost matrices.
    # non_ag_c_rk: np.ndarray  # Non-agricultural production cost matrix.
    # non_ag_r_rk: np.ndarray  # Non-agricultural revenue matrix.
    # non_ag_g_rk: np.ndarray  # Non-agricultural greenhouse gas emissions matrix.
    # non_ag_w_rk: np.ndarray  # Non-agricultural water requirements matrix.
    # non_ag_x_rk: np.ndarray  # Non-agricultural exclude matrices.
    # non_ag_q_crk: np.ndarray  # Non-agricultural yield matrix.

    # ag_man_c_mrj: np.ndarray  # Agricultural management options' cost effects.
    # ag_man_g_mrj: np.ndarray  # Agricultural management options' GHG emission effects.
    # ag_man_q_mrp: np.ndarray  # Agricultural management options' quantity effects.
    # ag_man_r_mrj: np.ndarray  # Agricultural management options' revenue effects.
    # ag_man_t_mrj: np.ndarray  # Agricultural management options' transition cost effects.
    # ag_man_w_mrj: np.ndarray  # Agricultural management options' water requirement effects.

    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)

    def __hash__(self) -> int:
        hashes = []
        for field in fields(self):
            if field.type == np.ndarray:
                field_hash = hash(getattr(self, field.name).data.tobytes())
            else:
                field_hash = hash(getattr(self, field.name))

            hashes.append(field_hash)

        return hash(tuple(hashes))


def _round_signif(x, p):
    """
    Given a value `x` (potentially an array or matrix), return a copy of that value
    where each element has been rounded to `p` significant figures.

    See https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy
    """
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def _get_cluster_key_from_cell(input_data: InputData, cell_index: int):
    return ClusterKey(
        ag_t_mrj=_round_signif(
            input_data.ag_t_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        ),
        ag_c_mrj=_round_signif(
            input_data.ag_c_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        ),
        # ag_r_mrj=_round_signif(
        #     input_data.ag_r_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_g_mrj=_round_signif(
        #     input_data.ag_g_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_w_mrj=_round_signif(
        #     input_data.ag_w_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_x_mrj=_round_signif(
        #     input_data.ag_x_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_q_mrp=_round_signif(
        #     input_data.ag_q_mrp[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_ghg_t_mrj=_round_signif(
        #     input_data.ag_ghg_t_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_to_non_ag_t_rk=_round_signif(
        #     input_data.ag_to_non_ag_t_rk[cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # non_ag_to_ag_t_mrj=_round_signif(
        #     input_data.non_ag_to_ag_t_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # non_ag_c_rk=_round_signif(
        #     input_data.non_ag_c_rk[cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # non_ag_r_rk=_round_signif(
        #     input_data.non_ag_r_rk[cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # non_ag_g_rk=_round_signif(
        #     input_data.non_ag_g_rk[cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # non_ag_w_rk=_round_signif(
        #     input_data.non_ag_w_rk[cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # non_ag_x_rk=_round_signif(
        #     input_data.non_ag_x_rk[cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # non_ag_q_crk=_round_signif(
        #     input_data.non_ag_q_crk[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_man_c_mrj=_round_signif(
        #     input_data.ag_man_c_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_man_g_mrj=_round_signif(
        #     input_data.ag_man_g_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_man_q_mrp=_round_signif(
        #     input_data.ag_man_q_mrp[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_man_r_mrj=_round_signif(
        #     input_data.ag_man_r_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_man_t_mrj=_round_signif(
        #     input_data.ag_man_t_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
        # ag_man_w_mrj=_round_signif(
        #     input_data.ag_man_w_mrj[:, cell_index, :], settings.CLUSTERING_SIGFIGS
        # ),
    )


def get_clusters(input_data: InputData) -> dict[int, list[int]]:
    """
    Clusters cells with similar data based on settings.CLUSTERING_SIGFIGS.

    Each cluster is represented by a single cell. The solver will use data
    related to the representative cell for each cluster, and distribute the solution for
    the representative cell among all other cells in the cluster once the solve is finished.

    Returns a dictionary mapping a representative cell index to a list of cells in that cluster.
    """
    print("Clustering cells... ", end=" ", flush=True)
    cells_by_cluster_key = defaultdict(list)
    for r in range(input_data.ncells):
        key = _get_cluster_key_from_cell(input_data, r)
        cells_by_cluster_key[key].append(r)

    print("Done.")
    return {cells[0]: cells for _, cells in cells_by_cluster_key.items()}
