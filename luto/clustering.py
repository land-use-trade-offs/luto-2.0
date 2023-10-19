from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Dict

from tqdm import tqdm
from luto.solvers.input_data import InputData
from luto import settings
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES

import math
import numpy as np
import hashlib


@dataclass
class ClusterKey:
    ag_t_mrj: np.ndarray                 # Agricultural transition cost matrices.
    ag_c_mrj: np.ndarray                 # Agricultural production cost matrices.
    ag_r_mrj: np.ndarray                 # Agricultural production revenue matrices.
    ag_g_mrj: np.ndarray                 # Agricultural greenhouse gas emissions matrices.
    ag_w_mrj: np.ndarray                 # Agricultural water requirements matrices.
    ag_x_mrj: np.ndarray                 # Agricultural exclude matrices.
    ag_q_mrp: np.ndarray                 # Agricultural yield matrices -- note the `p` (product) index instead of `j` (land-use).
    ag_ghg_t_mrj: np.ndarray             # GHG emissions released during transitions between agricultural land uses.

    non_ag_x_rk: np.ndarray                # Non-agricultural exclude matrices.
    # ag_to_non_ag_t_rk: np.ndarray        # Agricultural to non-agricultural transition cost matrix.
    # non_ag_to_ag_t_mrj: np.ndarray       # Non-agricultural to agricultural transition cost matrices.
    # non_ag_c_rk: np.ndarray              # Non-agricultural production cost matrix.
    # non_ag_r_rk: np.ndarray              # Non-agricultural revenue matrix.
    # non_ag_g_rk: np.ndarray              # Non-agricultural greenhouse gas emissions matrix.
    # non_ag_w_rk: np.ndarray              # Non-agricultural water requirements matrix.
    # non_ag_q_crk: np.ndarray             # Non-agricultural yield matrix.

    # ag_man_c_mrj: Dict[str, np.ndarray]  # Agricultural management options' cost effects.
    # ag_man_g_mrj: Dict[str, np.ndarray]  # Agricultural management options' GHG emission effects.
    # ag_man_q_mrp: Dict[str, np.ndarray]  # Agricultural management options' quantity effects.
    # ag_man_r_mrj: Dict[str, np.ndarray]  # Agricultural management options' revenue effects.
    # ag_man_t_mrj: Dict[str, np.ndarray]  # Agricultural management options' transition cost effects.
    # ag_man_w_mrj: Dict[str, np.ndarray]  # Agricultural management options' water requirement effects.

    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)

    def __hash__(self) -> int:
        hashes = []
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                field_hash = hashlib.sha256(value.data).hexdigest()
            elif isinstance(value, dict):
                dict_hashes = []
                for _, v in sorted(value.items(), key=lambda x: x[0]):
                    dict_hashes.append(hashlib.sha256(v.data).hexdigest())
                field_hash = tuple(dict_hashes)
            else:
                field_hash = hash(value)

            hashes.append(field_hash)

        return hash(tuple(hashes))
    

def round_decimals(x, p):
    """
    Given a value `x` (array or matrix), return a copy of that value where
    each element has been rounded to `p` decimal places.
    """
    x = np.asarray(x, dtype=np.float32)
    return np.round(x, decimals=p)


def round_significant_figures(x, p):
    """
    Given a value `x` (potentially an array or matrix), return a copy of that value
    where each element has been rounded to `p` significant figures.

    See https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy
    """
    x = np.asarray(x, dtype=np.float64)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def magnitude(x):
    if x == 0:
        return 0
    return int(math.floor(math.log10(x)))


def _get_decimals_to_round_to(input_data: InputData, n: int):
    """
    For each specified array, get the number of decimals to round the array to when 
    clustering cells with similar data.
    """
    decimals_to_round = {}
    
    arrays_to_round = [
        'ag_t_mrj',
        'ag_c_mrj',
        'ag_r_mrj',
        'ag_ghg_t_mrj',
        'ag_to_non_ag_t_rk',
        'non_ag_to_ag_t_mrj',
        'non_ag_c_rk',
        'non_ag_r_rk',
        'non_ag_g_rk',
        'non_ag_w_rk',
        'non_ag_q_crk',
    ]
    for array_name in arrays_to_round:
        array = getattr(input_data, array_name)
        max_val = abs(np.nanmax(array))
        mag = magnitude(max_val)
        decimals_to_round[array_name] = n - mag - 1

    # Handle special clustering cases:
    decimals_to_round['ag_w_mrj'] = -2
    decimals_to_round['ag_g_mrj'] = -2

    # Handle ag management dicts
    for am_dict in [
        'ag_man_c_mrj',
        'ag_man_g_mrj',
        'ag_man_q_mrp',
        'ag_man_r_mrj',
        'ag_man_t_mrj',
        'ag_man_w_mrj',
    ]:
        decimals_to_round[am_dict] = {}
        for am in AG_MANAGEMENTS_TO_LAND_USES:
            array = getattr(input_data, am_dict)[am]
            max_val = abs(np.nanmax(array))
            mag = magnitude(max_val)
            decimals_to_round[am_dict][am] = n - mag - 1

    return decimals_to_round


def _get_decimals_to_round_ag_q_mrp(input_data: InputData, n: int):
    decimals_to_round = {}
    for p in range(input_data.nprs):
        max_val = abs(np.nanmax(input_data.ag_q_mrp[:, :, p]))
        mag = magnitude(max_val)
        decimals = min(n - mag, 0)
        decimals_to_round[p] = decimals

    return decimals_to_round


def _get_rounded_ag_q_mrp_for_cell(r, input_data: InputData, decimals_to_round: dict):
    """
    Handle the rounding of the quantity matrix separately because magnitudes differ drastically 
    between products.
    """
    rounded_ag_q_mrp = np.zeros((input_data.n_ag_lms, input_data.nprs))

    for p in range(input_data.nprs):
        rounded_ag_q_mrp[:, p] = round_decimals(
            input_data.ag_q_mrp[:, r, p], 
            decimals_to_round[p]
        )

    return rounded_ag_q_mrp


def _round_ag_management_dict(
    r: int,
    ag_man_dict: dict, 
    decimals_to_round: dict,
) -> dict:
    """
    Applies the round_decimals function to an entire ag management dictionary.
    """
    return {
        am: round_decimals(array[:, r, :], decimals_to_round[am])
        for am, array in ag_man_dict.items()
    }


def _get_cluster_key_from_cell(
    input_data: InputData, 
    r: int, 
    decimals_to_round_arrays: dict, 
    decimals_to_round_ag_q_mrp: dict
):
    """
    For a cell, get the cluster key with which to cluster.
    Change the 'round_func' variable to alter the rounding metric used.
    """
    exclude_matrix_mj = input_data.ag_x_mrj[:, r, :]
    
    exclude_matrix_mp = np.zeros((input_data.n_ag_lms, input_data.nprs))
    for m in range(input_data.n_ag_lms):
        for j in range(input_data.n_ag_lus):
            if exclude_matrix_mj[m, j]:
                for p in input_data.j2p[j]:
                    exclude_matrix_mp[m, p] = 1

    cluster_key = ClusterKey(
        ag_t_mrj=round_decimals(
            exclude_matrix_mj * input_data.ag_t_mrj[:, r, :],
            decimals_to_round_arrays['ag_t_mrj'],
        ),
        ag_c_mrj=round_decimals(
            exclude_matrix_mj * input_data.ag_c_mrj[:, r, :],
            decimals_to_round_arrays['ag_c_mrj'],
        ),
        ag_r_mrj=round_decimals(
            exclude_matrix_mj * input_data.ag_r_mrj[:, r, :],
            decimals_to_round_arrays['ag_r_mrj'],
        ),
        ag_g_mrj=round_decimals(
            exclude_matrix_mj * input_data.ag_g_mrj[:, r, :],
            decimals_to_round_arrays['ag_g_mrj'],
        ),
        ag_w_mrj=round_decimals(
            exclude_matrix_mj * input_data.ag_w_mrj[:, r, :],
            decimals_to_round_arrays['ag_w_mrj'],
        ),
        ag_x_mrj=exclude_matrix_mj,
        ag_q_mrp=_get_rounded_ag_q_mrp_for_cell(r, input_data, decimals_to_round_ag_q_mrp),
        ag_ghg_t_mrj=round_decimals(
            exclude_matrix_mj * input_data.ag_ghg_t_mrj[:, r, :],
            decimals_to_round_arrays['ag_ghg_t_mrj'],
        ),
        non_ag_x_rk=input_data.non_ag_x_rk[r, :], 
        # ag_to_non_ag_t_rk=round_decimals(
        #     input_data.ag_to_non_ag_t_rk[r, :], 
        #     decimals_to_round_arrays['ag_to_non_ag_t_rk']
        # ),
        # non_ag_to_ag_t_mrj=round_decimals(
        #     input_data.non_ag_to_ag_t_mrj[:, r, :], 
        #     decimals_to_round_arrays['non_ag_to_ag_t_mrj']
        # ),
        # non_ag_c_rk=round_decimals(
        #     input_data.non_ag_c_rk[r, :], 
        #     decimals_to_round_arrays['non_ag_c_rk']
        # ),
        # non_ag_r_rk=round_decimals(
        #     input_data.non_ag_r_rk[r, :], 
        #     decimals_to_round_arrays['non_ag_r_rk']
        # ),
        # non_ag_g_rk=round_decimals(
        #     input_data.non_ag_g_rk[r, :], 
        #     decimals_to_round_arrays['non_ag_g_rk']
        # ),
        # non_ag_w_rk=round_decimals(
        #     input_data.non_ag_w_rk[r, :], 
        #     decimals_to_round_arrays['non_ag_w_rk']
        # ),
        # non_ag_q_crk=round_decimals(
        #     input_data.non_ag_q_crk[:, r, :], 
        #     decimals_to_round_arrays['non_ag_q_crk']
        # ),
        # ag_man_c_mrj=_round_ag_management_dict(
        #     r, input_data.ag_man_c_mrj, decimals_to_round_arrays['ag_man_c_mrj']
        # ),
        # ag_man_q_mrp=_round_ag_management_dict(
        #     r, input_data.ag_man_q_mrp, decimals_to_round_arrays['ag_man_q_mrp']
        # ),
        # ag_man_g_mrj=_round_ag_management_dict(
        #     r, input_data.ag_man_g_mrj, decimals_to_round_arrays['ag_man_g_mrj']
        # ),
        # ag_man_r_mrj=_round_ag_management_dict(
        #     r, input_data.ag_man_r_mrj, decimals_to_round_arrays['ag_man_r_mrj']
        # ),
        # ag_man_t_mrj=_round_ag_management_dict(
        #     r, input_data.ag_man_t_mrj, decimals_to_round_arrays['ag_man_t_mrj']
        # ),
        # ag_man_w_mrj=_round_ag_management_dict(
        #     r, input_data.ag_man_w_mrj, decimals_to_round_arrays['ag_man_w_mrj']
        # ),
    )

    return cluster_key


def get_clusters(input_data: InputData) -> dict[int, list[int]]:
    """
    Clusters cells with similar data based on settings.CLUSTERING_SIGFIGS.

    Each cluster is represented by a single cell. The solver will use data
    related to the representative cell for each cluster, and distribute the solution for
    the representative cell among all other cells in the cluster once the solve is finished.

    Returns a dictionary mapping a representative cell index to a list of cells in that cluster.
    """
    n = settings.CLUSTERING_SIGFIGS
    print(f"\nClustering cells (precision: {n} significant figures)...")
    cells_by_cluster_key = defaultdict(list)

    decimals_to_round_arrays = _get_decimals_to_round_to(input_data, n)
    decimals_to_round_ag_q_mrp = _get_decimals_to_round_ag_q_mrp(input_data, n)

    for r in tqdm(range(input_data.ncells)):
        key = _get_cluster_key_from_cell(input_data, r, decimals_to_round_arrays, decimals_to_round_ag_q_mrp)
        cells_by_cluster_key[key].append(r)

    cluster_sizes = np.array([len(cells) for cells in cells_by_cluster_key.values()])
    print(
        f"Clustering completed: {input_data.ncells} cells to {len(cluster_sizes)} clusters.\n"
        f"min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
        f"mean={cluster_sizes.mean()}, std={cluster_sizes.std()}"
    )
    print("Done.")
    return {cells[0]: cells for _, cells in cells_by_cluster_key.items()}
