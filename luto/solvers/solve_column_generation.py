from collections import defaultdict
from dataclasses import dataclass
import time
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum, tupledict, tuplelist

from itertools import chain
from functools import cached_property

from luto import settings


# Set Gurobi environment.
gurenv = gp.Env(empty=True)
gurenv.setParam("Method", settings.SOLVE_METHOD)
gurenv.setParam("OutputFlag", settings.VERBOSE)
gurenv.setParam("Threads", 1)
gurenv.setParam("WorkLimit", 3600)
gurenv.setParam("NumericFocus", 3)
gurenv.start()


# column generation params
MAX_NEGATIVE_REDUCED_COST = -0.000001
REDUCED_COST_CALC_GROUP_PRECISION = 2
DISSOLVE_CLUSTER_THRESHOLD = 0.1
# Set to negative number to disable cluster dissolving at the end
OBJECTIVE_STALL_THRESHOLD = 20
DEFAULT_CLUSTER_SPLIT_AMOUNT = 10


@dataclass
class Cluster:
    """
    A cluster of data
    """

    cluster_id: int
    cells: np.ndarray
    t_mrj: np.ndarray
    c_mrj: np.ndarray
    q_mrp: np.ndarray
    x_mrj: np.ndarray

    allowed_dry_land_uses: tuple[int]
    allowed_irr_land_uses: tuple[int]

    data_group_to_cells: dict
    cells_to_data_group: dict

    land_use_constraint = None

    def get_ct_mj(self, penalty):
        """
        Pre-calculate sum of production and transition costs and apply penalty
        """
        return (
            np.sum(self.c_mrj, axis=1) + np.sum(self.t_mrj, axis=1)
        ) / penalty

    @cached_property
    def q_mp(self):
        return self.q_mrp.sum(axis=1)


@dataclass
class VariablesStore:
    """
    Stores the column generation model's variables.
    """

    X_dry_kj: tupledict
    X_irr_kj: tupledict
    V_c: tupledict


@dataclass
class ConstraintsStore:
    """
    Stores the column generation model's constraints
    """

    land_use: dict
    demand_1: dict
    demand_2: dict
    water: dict


def _get_new_limits_from_clusters(
    list_of_clusters: List[Cluster],
    original_limits: dict,
    nlms: int,
    cells_to_regions: dict,
    new_clusters_to_add: List[Cluster],
) -> dict[str, Any]:
    """
    Given a list of new clusters, return the limits dictionary transformed to
    match the list of clusters.
    """
    clustered_limits = {}

    # make new water limits
    original_aqreq_mrj, original_aqreq_limits = original_limits["water"]

    clustered_aqreq_mkjR = defaultdict(int)  # key: m, k (cluster), j, region

    for cluster in list_of_clusters:
        for r in cluster.cells:
            if r not in cells_to_regions:
                continue

            reg = cells_to_regions[r]
            for m in range(nlms):
                for j in _get_allowed_land_usages(cluster, m):
                    clustered_aqreq_mkjR[
                        (m, cluster.cluster_id, j, reg)
                    ] += original_aqreq_mrj[m, r, j]

    # remake ind (the list of new clusters to add for the region)
    if not new_clusters_to_add:
        new_clusters_to_add = list_of_clusters

    cells_to_new_clusters: dict[int, int] = {
        r: cluster.cluster_id
        for cluster in new_clusters_to_add
        for r in cluster.cells
    }

    new_aqreq_limits = []
    for region, aqreq_reg_limit, ind in original_aqreq_limits:
        relevant_clusters = set()
        for cell in ind:
            cell_new_cluster = cells_to_new_clusters.get(cell)
            if cell_new_cluster is not None:
                relevant_clusters.add(cell_new_cluster)
        new_aqreq_limits.append(
            (
                region,
                aqreq_reg_limit,
                np.array(list(relevant_clusters), dtype=int),
            )
        )

    clustered_limits["water"] = [clustered_aqreq_mkjR, new_aqreq_limits]
    return clustered_limits


def _make_initial_data_clusters(
    max_cluster_size,
    t_mrj,
    c_mrj,
    q_mrp,
    x_mrj,
    nlms,
    limits: Dict[str, List[np.ndarray]],
    cells_to_regions: dict[int, int],
    cells_to_data_group: dict,
    cells_to_allowed_dry_lu: dict,
    cells_to_allowed_irr_lu: dict,
) -> Tuple[List[Cluster], Dict[str, Any]]:
    """
    Make the initial set of clusters.

    To keep the problem feasible, we need to cluster based on the allowed activities.
    Every cell in the cluster has to have the same set of allowed activities.
    """
    st = time.time()
    print("Making initial data clusters...", end=" ")
    # mapping from combinations of allowed land uses to the cells that have that combination
    # key: (allowed dry lu tuple, allowed irr lu tuple)
    allowed_land_uses_to_cells = defaultdict(list)

    for cell_num in range(x_mrj.shape[1]):
        dry_lu_allowed = cells_to_allowed_dry_lu[cell_num]
        irr_lu_allowed = cells_to_allowed_irr_lu[cell_num]
        allowed_lu = (dry_lu_allowed, irr_lu_allowed)

        allowed_land_uses_to_cells[allowed_lu].append(cell_num)

    # cluster based on allowed land uses
    all_cell_clusters = []
    for land_uses_tuple, lu_cells in allowed_land_uses_to_cells.items():
        lu_cells = np.array(lu_cells)
        cell_clusters = np.array_split(
            lu_cells, math.ceil(len(lu_cells) / max_cluster_size)
        )
        all_cell_clusters.append((land_uses_tuple, cell_clusters))

    list_of_clusters = []
    cluster_id = 0

    for land_uses_tuple, lu_cells_arrays in all_cell_clusters:
        allowed_dry_lu, allowed_irr_lu = land_uses_tuple
        for cells_array in lu_cells_arrays:
            cells_to_data_group_cluster = {
                r: cells_to_data_group[r] for r in cells_array
            }
            data_group_to_cells_cluster = defaultdict(list)
            for r, r_data_group in cells_to_data_group_cluster.items():
                data_group_to_cells_cluster[r_data_group].append(r)

            new_cluster = Cluster(
                cluster_id=cluster_id,
                cells=cells_array,
                t_mrj=t_mrj[:, cells_array, :],
                c_mrj=c_mrj[:, cells_array, :],
                q_mrp=q_mrp[:, cells_array, :],
                x_mrj=x_mrj[:, cells_array, :],
                allowed_dry_land_uses=allowed_dry_lu,
                allowed_irr_land_uses=allowed_irr_lu,
                data_group_to_cells=data_group_to_cells_cluster,
                cells_to_data_group=cells_to_data_group_cluster,
            )

            list_of_clusters.append(new_cluster)
            cluster_id += 1

    ft = time.time()
    print(f"Made {len(list_of_clusters)} clusters in {round(ft - st)}s")
    new_limits = _get_new_limits_from_clusters(
        list_of_clusters, limits, nlms, cells_to_regions, []
    )

    return list_of_clusters, new_limits


def _set_up_initial_model(
    list_of_clusters: List[Cluster],
    clustered_limits,
    demands,
    penalty,
    lu2pr_j,
    cm2pr_c,
) -> Tuple[gp.Model, VariablesStore, ConstraintsStore]:
    """
    Set up the initial clustered formulation of the LUTO 2.0 model.
    """
    print("Formulating the initial model...", end=" ")
    model = gp.Model("LUTO 2.0", env=gurenv)
    _, _, nprs = list_of_clusters[0].q_mrp.shape
    (ncms,) = demands.shape

    cluster_id_to_obj = {
        cluster.cluster_id: cluster for cluster in list_of_clusters
    }
    _, cluster_size, _ = list_of_clusters[
        0
    ].t_mrj.shape  # Number of landmans, cells in a cluster, landuses.

    # add variables
    X_dry_kj_tuples = tuplelist(
        [
            (cluster.cluster_id, j)
            for cluster in list_of_clusters
            for j in cluster.allowed_dry_land_uses
        ]
    )
    X_irr_kj_tuples = tuplelist(
        [
            (cluster.cluster_id, j)
            for cluster in list_of_clusters
            for j in cluster.allowed_irr_land_uses
        ]
    )

    X_dry_kj = model.addVars(X_dry_kj_tuples)
    X_irr_kj = model.addVars(X_irr_kj_tuples)
    V_c = model.addVars(range(ncms))

    variables_store = VariablesStore(
        X_dry_kj=X_dry_kj, X_irr_kj=X_irr_kj, V_c=V_c
    )

    # add objective
    objective = (
        quicksum(
            cluster.get_ct_mj(penalty)[0, j] * X_dry_kj[cluster.cluster_id, j]
            for cluster in list_of_clusters
            for j in cluster.allowed_dry_land_uses
        )
        + quicksum(
            cluster.get_ct_mj(penalty)[1, j] * X_irr_kj[cluster.cluster_id, j]
            for cluster in list_of_clusters
            for j in cluster.allowed_irr_land_uses
        )
        + quicksum(V_c[c] for c in range(ncms))
    )
    model.setObjective(objective)

    # add constraints
    # add land use constraints for new clusters
    land_use_constraints = {
        cluster.cluster_id: model.addConstr(
            quicksum(
                X_dry_kj[cluster.cluster_id, j]
                for j in cluster.allowed_dry_land_uses
            )
            + quicksum(
                X_irr_kj[cluster.cluster_id, j]
                for j in cluster.allowed_irr_land_uses
            )
            == 1,
        )
        for cluster in list_of_clusters
    }

    # Add demand constraints
    X_dry_pk = defaultdict(list)
    for cluster in list_of_clusters:
        for j in cluster.allowed_dry_land_uses:
            for p in lu2pr_j[j]:
                X_dry_pk[p, cluster.cluster_id].append(
                    X_dry_kj[cluster.cluster_id, j]
                )

    X_irr_pk = defaultdict(list)
    for cluster in list_of_clusters:
        for j in cluster.allowed_irr_land_uses:
            for p in lu2pr_j[j]:
                X_irr_pk[p, cluster.cluster_id].append(
                    X_irr_kj[cluster.cluster_id, j]
                )

    q_dry_p = {
        p: quicksum(
            cluster.q_mp[0, p] * var
            for cluster in list_of_clusters
            for var in X_dry_pk[p, cluster.cluster_id]
        )
        for p in range(nprs)
    }
    q_irr_p = {
        p: quicksum(
            cluster.q_mp[1, p] * var
            for cluster in list_of_clusters
            for var in X_irr_pk[p, cluster.cluster_id]
        )
        for p in range(nprs)
    }

    q_dry_c = {c: quicksum(q_dry_p[p] for p in cm2pr_c[c]) for c in range(ncms)}
    q_irr_c = {c: quicksum(q_irr_p[p] for p in cm2pr_c[c]) for c in range(ncms)}
    q_c = {c: q_dry_c[c] + q_irr_c[c] for c in range(ncms)}

    demand_1_constraints = {
        c: model.addConstr(
            (demands[c] - q_c[c]) <= V_c[c], name=f"demand_1-{c}"
        )
        for c in range(ncms)
    }
    demand_2_constraints = {
        c: model.addConstr(
            (q_c[c] - demands[c]) <= V_c[c], name=f"demand_2-{c}"
        )
        for c in range(ncms)
    }

    # Add water constraints if they don't already exist (first iteration)
    water_constraints = {}
    if "water" in clustered_limits:
        aqreq_mkjR, aqreq_limits = clustered_limits["water"]
        for reg, aqreq_reg_limit, ind in aqreq_limits:
            if ind.shape == (0,):
                continue

            dry_iter = (
                aqreq_mkjR[0, cluster_id, j, reg] * X_dry_kj[cluster_id, j]
                for cluster_id in ind
                for j in _get_allowed_land_usages(
                    cluster_id_to_obj[cluster_id], 0
                )
            )
            irr_iter = (
                aqreq_mkjR[1, cluster_id, j, reg] * X_irr_kj[cluster_id, j]
                for cluster_id in ind
                for j in _get_allowed_land_usages(
                    cluster_id_to_obj[cluster_id], 1
                )
            )

            aqreq_region = quicksum(chain(dry_iter, irr_iter))
            water_constraints[reg] = model.addConstr(
                aqreq_region <= aqreq_reg_limit
            )

    constraints_store = ConstraintsStore(
        land_use=land_use_constraints,
        demand_1=demand_1_constraints,
        demand_2=demand_2_constraints,
        water=water_constraints,
    )

    for k, cluster in enumerate(list_of_clusters):
        cluster.land_use_constraint = land_use_constraints[k]

    print("Done.")

    return model, variables_store, constraints_store


def _add_and_update_variables(
    model,
    variables_store: VariablesStore,
    clusters_to_remove,
    new_clusters_to_add,
    penalty,
) -> Tuple[gp.Model, VariablesStore]:
    """
    Deletes old variables associated with clusters that were split on the previous iteration.
    Adds new variables for the new clusters and updates the variables store.
    """
    # delete variables that correspond to clusters that have been split
    for cluster in clusters_to_remove:
        model.remove(variables_store.X_dry_kj.select(cluster.cluster_id, "*"))
        model.remove(variables_store.X_irr_kj.select(cluster.cluster_id, "*"))

    # add new variables for new clusters
    for cluster in new_clusters_to_add:
        cluster_ct_mrj = cluster.get_ct_mj(penalty)
        for j in cluster.allowed_dry_land_uses:
            variables_store.X_dry_kj[cluster.cluster_id, j] = model.addVar(
                obj=cluster_ct_mrj[0, j]
            )

        for j in cluster.allowed_irr_land_uses:
            variables_store.X_irr_kj[cluster.cluster_id, j] = model.addVar(
                obj=cluster_ct_mrj[1, j]
            )

    return model, variables_store


def _add_and_update_constraints(
    model,
    variables_store: VariablesStore,
    constraints_store: ConstraintsStore,
    clusters_to_remove: List[Cluster],
    new_clusters_to_add: List[Cluster],
    cluster_id_to_obj,
    lu2pr_j,
    cm2pr_c,
    clustered_limits,
):
    """
    Updates all constraints to handle the new set of clusters
    """
    aqreq_mkjR, aqreq_limits = clustered_limits["water"]

    # delete land constraints for clusters that were split in the previous iteration
    lu_to_remove = [
        constraints_store.land_use[cluster.cluster_id]
        for cluster in clusters_to_remove
    ]
    model.remove(lu_to_remove)
    for cluster in clusters_to_remove:
        del constraints_store.land_use[cluster.cluster_id]

    # update demand constraints
    X_dry_kj = variables_store.X_dry_kj
    X_irr_kj = variables_store.X_irr_kj
    X_dry_pk_new_clusters = defaultdict(list)
    for cluster in new_clusters_to_add:
        for j in cluster.allowed_dry_land_uses:
            for p in lu2pr_j[j]:
                X_dry_pk_new_clusters[p, cluster.cluster_id].append(
                    X_dry_kj[cluster.cluster_id, j]
                )

    X_irr_pk_new_clusters = defaultdict(list)
    for cluster in new_clusters_to_add:
        for j in cluster.allowed_irr_land_uses:
            for p in lu2pr_j[j]:
                X_irr_pk_new_clusters[p, cluster.cluster_id].append(
                    X_irr_kj[cluster.cluster_id, j]
                )

    for c, constraint in constraints_store.demand_1.items():
        for p in cm2pr_c[c]:
            for cluster in new_clusters_to_add:
                q_mp = cluster.q_mp
                for var in X_dry_pk_new_clusters[p, cluster.cluster_id]:
                    model.chgCoeff(constraint, var, -q_mp[0, p])

                for var in X_irr_pk_new_clusters[p, cluster.cluster_id]:
                    model.chgCoeff(constraint, var, -q_mp[1, p])

    for c, constraint in constraints_store.demand_2.items():
        for p in cm2pr_c[c]:
            for cluster in new_clusters_to_add:
                q_mp = cluster.q_mp
                for var in X_dry_pk_new_clusters[p, cluster.cluster_id]:
                    model.chgCoeff(constraint, var, q_mp[0, p])

                for var in X_irr_pk_new_clusters[p, cluster.cluster_id]:
                    model.chgCoeff(constraint, var, q_mp[1, p])

    region_to_new_clusters = {
        region_number: region_new_clusters
        for region_number, _, region_new_clusters in aqreq_limits
    }

    for reg, constraint in constraints_store.water.items():
        for k in region_to_new_clusters[reg]:
            cluster = cluster_id_to_obj[k]
            for j in cluster.allowed_dry_land_uses:
                model.chgCoeff(
                    constraint, X_dry_kj[k, j], aqreq_mkjR[0, k, j, reg]
                )

            for j in cluster.allowed_irr_land_uses:
                model.chgCoeff(
                    constraint, X_irr_kj[k, j], aqreq_mkjR[1, k, j, reg]
                )

    # add land use constraints for new clusters
    new_land_use_constraints = {
        cluster.cluster_id: model.addConstr(
            quicksum(
                X_dry_kj[cluster.cluster_id, j]
                for j in cluster.allowed_dry_land_uses
            )
            + quicksum(
                X_irr_kj[cluster.cluster_id, j]
                for j in cluster.allowed_irr_land_uses
            )
            == 1,
        )
        for cluster in new_clusters_to_add
    }
    constraints_store.land_use.update(new_land_use_constraints)

    return model, constraints_store


def _update_and_solve_model(
    model,
    list_of_clusters: List[Cluster],
    clustered_limits: Dict[str, Any],
    variables_store: VariablesStore,
    constraints_store: ConstraintsStore,
    cm2pr_c,
    lu2pr_j,
    clusters_to_remove: List[Cluster],
    new_clusters_to_add: List[Cluster],
    penalty,
) -> tuple:
    """
    Update the model by adding new variables that correspond to the new clusters and removing
    variables that correspond to clusters that were split. Also, update the constraints to
    account for the new list of clusters.
    """
    cluster_id_to_obj = {
        cluster.cluster_id: cluster for cluster in list_of_clusters
    }

    if new_clusters_to_add:
        # handle variables for the model
        model, variables_store = _add_and_update_variables(
            model,
            variables_store,
            clusters_to_remove,
            new_clusters_to_add,
            penalty,
        )

        # handle the constraints of the model
        model, constraints_store = _add_and_update_constraints(
            model,
            variables_store,
            constraints_store,
            clusters_to_remove,
            new_clusters_to_add,
            cluster_id_to_obj,
            lu2pr_j,
            cm2pr_c,
            clustered_limits,
        )

    # solve
    model.optimize()

    # collect dual values
    dual_values = {}

    dual_values["land_use_constraint_duals"] = {
        k: constraint.Pi for k, constraint in constraints_store.land_use.items()
    }

    dual_values["demand_1_constraint_duals"] = {
        commodity: constr.Pi
        for commodity, constr in constraints_store.demand_1.items()
    }

    dual_values["demand_2_constraint_duals"] = {
        commodity: constr.Pi
        for commodity, constr in constraints_store.demand_2.items()
    }

    dual_values["water_constraint_duals"] = {
        reg: mat_constr.Pi
        for reg, mat_constr in constraints_store.water.items()
    }

    return model, variables_store, constraints_store, dual_values, model.objVal


def _get_alpha_values(
    cluster: Cluster, semi_reduced_costs_mrj, nlms
) -> dict[int, float]:
    """
    Gets the alpha values for calculating the reduced costs.
    Alpha values are the coefficients of the dual values of the land use constraint.
    """
    # base the scores on the max potential of the cell in regard to land usage
    # score each cell, with a higher score being better
    cell_scores = {
        r: min(
            semi_reduced_costs_mrj[m, r, j]
            for m in range(nlms)
            for j in _get_allowed_land_usages(cluster, m)
        )
        for r in cluster.cells
    }

    alpha = {r: 0 for r, score in cell_scores.items() if score >= 0}
    if len(alpha) == len(cluster.cells):
        return {r: 1 / len(cluster.cells) for r in cluster.cells}

    q = sum(score for r, score in cell_scores.items() if r not in alpha)

    for r in cluster.cells:
        if r not in alpha:
            a = cell_scores[r] / q
            alpha[r] = a

    return alpha


def _get_allowed_land_usages(cluster: Cluster, m):
    if m == 0:
        return cluster.allowed_dry_land_uses

    elif m == 1:
        return cluster.allowed_irr_land_uses

    else:
        raise ValueError(
            f"Unknown land management option: {m} (should be either 0 or 1)"
        )


def _get_reduced_costs_of_cells(
    list_of_clusters: List[Cluster],
    ct_mrj,
    original_limits,
    q_mrp_coeff_sums_jmrc: dict,
    lu2cm_j,
    cells_to_regions: dict[int, int],
    land_use_constraint_duals,
    demand_1_constraint_duals,
    demand_2_constraint_duals,
    water_constraint_duals,
) -> Tuple[Dict[Tuple[int, int, int], float], List[Cluster]]:
    """
    Calculate the reduced costs at a cell level for the current iteration.
    """
    print("Calculating reduced costs...", end=" ")
    st = time.time()
    nlms, _, _, = ct_mrj.shape

    aqreq_mrj, _ = original_limits["water"]

    cell_reduced_costs_mrj = {}
    clusters_to_split = []

    for cluster in list_of_clusters:
        if len(cluster.cells) == 1:
            continue

        semi_reduced_cost_mrj = {}
        # calculate semi-reduced costs
        for cells_group in cluster.data_group_to_cells.values():
            r_0 = cells_group[0]
            r_0_water_constraint_dual = water_constraint_duals[
                cells_to_regions[r_0]
            ]

            for m in range(nlms):
                for j in _get_allowed_land_usages(cluster, m):
                    semi_reduced_cost = ct_mrj[m, r_0, j]

                    # subtract contribution of water constraint dual
                    semi_reduced_cost -= (
                        aqreq_mrj[m, r_0, j] * r_0_water_constraint_dual
                    )

                    # subtract contributions of demand constraints
                    for c in lu2cm_j[j]:
                        q_mrp_coeff_sum = q_mrp_coeff_sums_jmrc[j, m, r_0, c]
                        semi_reduced_cost += q_mrp_coeff_sum * (
                            demand_1_constraint_duals[c]
                            - demand_2_constraint_duals[c]
                        )

                    # set semi reduced costs for every cell in the group
                    for r in cells_group:
                        semi_reduced_cost_mrj[m, r, j] = semi_reduced_cost

        # get the alpha values
        alpha = _get_alpha_values(cluster, semi_reduced_cost_mrj, nlms)

        # calculate the reduced costs
        split_cluster = False
        for (m, r, j), src in semi_reduced_cost_mrj.items():
            reduced_cost = (
                src - alpha[r] * land_use_constraint_duals[cluster.cluster_id]
            )
            cell_reduced_costs_mrj[m, r, j] = reduced_cost
            split_cluster |= reduced_cost < MAX_NEGATIVE_REDUCED_COST

        if split_cluster:
            clusters_to_split.append(cluster)

    print(f"Done in {round(time.time() - st)} seconds.")

    return cell_reduced_costs_mrj, clusters_to_split


def _make_new_clusters_with_split(
    cluster: Cluster,
    cell_splits: list[list[int]],
    new_cluster_id_counter: int,
    t_mrj,
    c_mrj,
    x_mrj,
    q_mrp,
) -> tuple[list[Cluster], int]:
    """
    Make new clusters given the splits of the cells from the original cluster.
    """
    new_clusters = []
    for list_of_cells in cell_splits:
        cells_ary = np.array(list_of_cells)
        cells_to_data_group_cluster = {
            r: cluster.cells_to_data_group[r] for r in cells_ary
        }
        data_group_to_cells_cluster = defaultdict(list)

        for r, r_data_group in cells_to_data_group_cluster.items():
            data_group_to_cells_cluster[r_data_group].append(r)

        new_clusters.append(
            Cluster(
                cluster_id=new_cluster_id_counter,
                cells=np.array(list_of_cells),
                t_mrj=t_mrj[:, cells_ary, :],
                c_mrj=c_mrj[:, cells_ary, :],
                q_mrp=q_mrp[:, cells_ary, :],
                x_mrj=x_mrj[:, cells_ary, :],
                allowed_dry_land_uses=cluster.allowed_dry_land_uses,
                allowed_irr_land_uses=cluster.allowed_irr_land_uses,
                data_group_to_cells=data_group_to_cells_cluster,
                cells_to_data_group=cells_to_data_group_cluster,
            )
        )
        new_cluster_id_counter += 1

    return new_clusters, new_cluster_id_counter


def _split_cluster_based_on_land_use(
    cluster: Cluster,
    reduced_costs_mrj,
    nlms,
) -> list[list[int]]:
    """
    Split a cluster by grouping cells together by the land usage that corresponds to their
    minimum reduced cost.
    """
    # get the land usages of the minimum reduced costs
    land_use_to_cells_with_min_rc = defaultdict(list)

    for r in cluster.cells:
        land_use_rc = {
            (m, j): reduced_costs_mrj[m, r, j]
            for m in range(nlms)
            for j in _get_allowed_land_usages(cluster, m)
        }
        best_lu = min(
            land_use_rc, key=land_use_rc.get
        )  # get key of the lowest reduced cost use for cell
        land_use_to_cells_with_min_rc[best_lu].append(r)

    if len(land_use_to_cells_with_min_rc) > 1:
        return list(land_use_to_cells_with_min_rc.values())
    else:
        return np.array_split(cluster.cells, DEFAULT_CLUSTER_SPLIT_AMOUNT)


def _make_new_clusters_from_reduced_costs(
    current_clusters: List[Cluster],
    reduced_costs_mrj: dict,
    clusters_to_split: List[Cluster],
    original_limits: dict,
    nlms: int,
    t_mrj,
    c_mrj,
    x_mrj,
    q_mrp,
    cells_to_regions,
    dissolve_threshold: int,
) -> Tuple[List[Cluster], dict, List[Cluster], List[Cluster]]:
    """
    Make new clusters based on the current iteration's clusters and the calculated reduced costs.
    """
    print("Splitting up clusters...", end=" ")
    # take the min reduced cost to decide whether to split the cell or not
    clusters_next_iter = []
    new_cluster_id_counter = (
        max(cluster.cluster_id for cluster in current_clusters) + 1
    )

    clusters_to_remove = []
    new_clusters_added = []
    clusters_to_split_ids = {c.cluster_id for c in clusters_to_split}

    for current_cluster in current_clusters:
        if current_cluster.cluster_id not in clusters_to_split_ids:
            clusters_next_iter.append(current_cluster)
            continue

        if len(current_cluster.cells) <= dissolve_threshold:
            split = [[r] for r in current_cluster.cells]
        else:
            # split cluster based on what the reduced cost suggests the land usage should be
            split = _split_cluster_based_on_land_use(
                current_cluster, reduced_costs_mrj, nlms
            )

        new_clusters, new_cluster_id_counter = _make_new_clusters_with_split(
            current_cluster,
            split,
            new_cluster_id_counter,
            t_mrj,
            c_mrj,
            x_mrj,
            q_mrp,
        )
        assert len(new_clusters) > 1
        clusters_to_remove.append(current_cluster)
        clusters_next_iter.extend(new_clusters)
        new_clusters_added.extend(new_clusters)

    no_cluster_splits = len(clusters_to_split)
    if no_cluster_splits > 0:
        avg_clusters_per_split = len(new_clusters_added) / no_cluster_splits
        avg_split_cluster_size = (
            sum(len(c.cells) for c in clusters_to_split) / no_cluster_splits
        )
    else:
        avg_split_cluster_size = 0
        avg_clusters_per_split = 0

    print(
        f"Split {no_cluster_splits} clusters (average size {avg_split_cluster_size:.2f}), "
        f"average {avg_clusters_per_split:.2f} clusters per split."
    )

    new_limits = _get_new_limits_from_clusters(
        clusters_next_iter,
        original_limits,
        nlms,
        cells_to_regions,
        new_clusters_added,
    )

    return (
        clusters_next_iter,
        new_limits,
        clusters_to_remove,
        new_clusters_added,
    )


def make_lumap_from_cluster_solution(
    list_of_clusters: List[Cluster], variables_store: VariablesStore, nlus
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Makes the lumap and lmmap based on the solution found by the clustering version of the algorithm.
    """
    cell_dry_solutions = {}
    cell_irr_solutions = {}
    # get proportions of land use from the solutions
    for cluster in list_of_clusters:
        cluster_dry_sol_j = [0 for _ in range(nlus)]
        cluster_irr_sol_j = [0 for _ in range(nlus)]

        for j in cluster.allowed_dry_land_uses:
            cluster_dry_sol_j[j] = variables_store.X_dry_kj[
                cluster.cluster_id, j
            ].X

        for j in cluster.allowed_irr_land_uses:
            cluster_irr_sol_j[j] = variables_store.X_irr_kj[
                cluster.cluster_id, j
            ].X

        for r in cluster.cells:
            cell_dry_solutions[r] = np.array(cluster_dry_sol_j)
            cell_irr_solutions[r] = np.array(cluster_irr_sol_j)

    cell_dry_solutions = dict(sorted(cell_dry_solutions.items()))
    cell_irr_solutions = dict(sorted(cell_irr_solutions.items()))

    X_dry_jr = np.stack(list(cell_dry_solutions.values())).T
    X_irr_jr = np.stack(list(cell_irr_solutions.values())).T

    # Get array of X variables from the solution
    X_mrj = np.stack((X_dry_jr, X_irr_jr))

    # Collect optimised decision variables in tuple of 1D Numpy arrays.
    highpos_dry = X_dry_jr.argmax(axis=0)
    highpos_irr = X_irr_jr.argmax(axis=0)

    lumap = np.where(
        X_dry_jr.max(axis=0) >= X_irr_jr.max(axis=0), highpos_dry, highpos_irr
    )
    lmmap = np.where(X_dry_jr.max(axis=0) >= X_irr_jr.max(axis=0), 0, 1)
    return lumap, lmmap, X_mrj


def _get_cells_by_data_maps(
    ncells, ct_mrj, q_mrp, aqreq_mrj, cells_to_regions
) -> dict:
    """
    To speed up the calculations of the reduced costs, cells with identical data can be grouped together.
    Identical cells in the same cluster will have the same reduced costs, so the computation will only
    be done once.
    """
    data_group_to_cells = defaultdict(list)

    rounded_aqreq_mrj = aqreq_mrj.round(REDUCED_COST_CALC_GROUP_PRECISION)
    rounded_ct_mrj = ct_mrj.round(REDUCED_COST_CALC_GROUP_PRECISION)
    rounded_q_mrp = q_mrp.round(REDUCED_COST_CALC_GROUP_PRECISION)

    for r in range(ncells):
        cell_regions_key = cells_to_regions[r]
        cell_aqreqs_key = (
            tuple(rounded_aqreq_mrj[0, r, :]),
            tuple(rounded_aqreq_mrj[1, r, :]),
        )
        cell_cost_key = (
            tuple(rounded_ct_mrj[0, r, :]),
            tuple(rounded_ct_mrj[1, r, :]),
        )
        cell_yield_key = (
            tuple(rounded_q_mrp[0, r, :]),
            tuple(rounded_q_mrp[1, r, :]),
        )

        overall_key = (
            *cell_yield_key,
            *cell_cost_key,
            *cell_aqreqs_key,
            cell_regions_key,
        )
        data_group_to_cells[overall_key].append(r)

    # simplify the keys of the dict to be IDs
    data_group_to_cells = dict(enumerate(data_group_to_cells.values()))

    cells_to_data_group = {
        r: data_group_id
        for data_group_id, cells_group in data_group_to_cells.items()
        for r in cells_group
    }

    print(
        f"Found {ncells - len(data_group_to_cells)} duplicate cells to ignore during calculation of reduced costs."
    )

    return cells_to_data_group


def get_pre_calc_q_mrp_contributions(
    q_mrp,
    cells_to_allowed_dry_lu,
    cells_to_allowed_irr_lu,
    lu2cm_j,
    lu2pr_j,
    cm2pr_c,
) -> dict:
    """
    Pre-calculate the sums of the q_mrp matrix for later use in calculating the reduced costs.
    """
    q_mrp_coeff_sums_jmrc = {}
    ncells = q_mrp.shape[1]
    for r in range(ncells):
        q_mrp_r_dry = q_mrp[0, r, :]
        for j in cells_to_allowed_dry_lu[r]:
            for c in lu2cm_j[j]:
                q_mrp_coeff_sums_jmrc[j, 0, r, c] = sum(
                    q_mrp_r_dry[p] for p in lu2pr_j[j] if p in cm2pr_c[c]
                )

        q_mrp_r_irr = q_mrp[1, r, :]
        for j in cells_to_allowed_irr_lu[r]:
            for c in lu2cm_j[j]:
                q_mrp_coeff_sums_jmrc[j, 1, r, c] = sum(
                    q_mrp_r_irr[p] for p in lu2pr_j[j] if p in cm2pr_c[c]
                )

    return q_mrp_coeff_sums_jmrc


def solve_using_column_generation(
    t_mrj,
    c_mrj,
    q_mrp,
    d_c,
    penalty,
    x_mrj,
    lu2pr_pj,
    pr2cm_cp,
    original_limits,
    max_cluster_size,
):
    """
    Solves the LUTO 2.0 problem by clustering cells together and using column generation techniques
    to iterate the solution.
    """
    print("\nSetting up column generation solve...", time.ctime())
    start_time = time.time()
    nlms, ncells, nlus = t_mrj.shape
    (ncms,) = d_c.shape
    _, _, nprs = q_mrp.shape  # Number of products.

    original_aqreq_mrj, original_aqreq_limits = original_limits["water"]
    cells_to_regions = {}
    for reg, _, ind in original_aqreq_limits:
        for r in ind:
            assert r not in cells_to_regions
            cells_to_regions[r] = reg

    # build useful lookups mapping between commodities, products, and land usages
    cm2pr_c = {
        c: [p for p in range(nprs) if pr2cm_cp[c, p]] for c in range(ncms)
    }
    pr2cm_p = {
        p: [c for c in range(ncms) if p in cm2pr_c[c]] for p in range(nprs)
    }
    pr2lu_p = {
        p: [j for j in range(nlus) if lu2pr_pj[p, j]] for p in range(nprs)
    }
    lu2pr_j = {
        j: [p for p in range(nprs) if j in pr2lu_p[p]] for j in range(nlus)
    }

    lu2cm_j: dict[int, tuple] = {}
    for j in range(nlus):
        j_products = lu2pr_j[j]
        j_commodities = []
        for p in j_products:
            j_commodities.extend(pr2cm_p[p])
        lu2cm_j[j] = tuple(j_commodities)

    # pre-calculate coefficients for calculation of reduced costs later
    ct_mrj = (c_mrj + t_mrj) / penalty
    cells_to_data_group = _get_cells_by_data_maps(
        ncells, ct_mrj, q_mrp, original_aqreq_mrj, cells_to_regions
    )

    cells_to_allowed_dry_lu = {
        r: tuple(np.where(x_mrj[0, r, :] == 1)[0]) for r in range(ncells)
    }
    cells_to_allowed_irr_lu = {
        r: tuple(np.where(x_mrj[1, r, :] == 1)[0]) for r in range(ncells)
    }

    q_mrp_coeff_sums_jmrc = get_pre_calc_q_mrp_contributions(
        q_mrp,
        cells_to_allowed_dry_lu,
        cells_to_allowed_irr_lu,
        lu2cm_j,
        lu2pr_j,
        cm2pr_c,
    )

    # make the initial data clusters
    initial_clusters, initial_cluster_limits = _make_initial_data_clusters(
        max_cluster_size,
        t_mrj,
        c_mrj,
        q_mrp,
        x_mrj,
        nlms,
        original_limits,
        cells_to_regions,
        cells_to_data_group,
        cells_to_allowed_dry_lu,
        cells_to_allowed_irr_lu,
    )

    model, variables_store, constraints_store = _set_up_initial_model(
        initial_clusters,
        initial_cluster_limits,
        d_c,
        penalty,
        lu2pr_j,
        cm2pr_c,
    )

    # set up data for first iteration
    iter_num = 0
    iter_clusters = initial_clusters
    iter_cluster_limits = initial_cluster_limits
    clusters_to_remove = []
    new_clusters_to_add = []
    previous_objective = gp.GRB.INFINITY

    while True:
        iter_num += 1
        start_of_iter_clusters_count = len(iter_clusters)
        print(
            f"\nSolving iteration: {iter_num} | "
            f"number of clusters: {start_of_iter_clusters_count}..."
        )
        (
            model,
            variables_store,
            constraints_store,
            dual_values,
            current_objective,
        ) = _update_and_solve_model(
            model,
            iter_clusters,
            iter_cluster_limits,
            variables_store,
            constraints_store,
            cm2pr_c,
            lu2pr_j,
            clusters_to_remove,
            new_clusters_to_add,
            penalty,
        )
        print(f"Solved with objective value: {round(current_objective, 3)}.")

        # calculate the reduced costs
        reduced_costs_mrj, clusters_to_split = _get_reduced_costs_of_cells(
            iter_clusters,
            ct_mrj,
            original_limits,
            q_mrp_coeff_sums_jmrc,
            lu2cm_j,
            cells_to_regions,
            **dual_values,
        )

        if not clusters_to_split:
            break

        # if the objective isn't moving much, dissolve all the clusters that need to be split
        if previous_objective - current_objective <= OBJECTIVE_STALL_THRESHOLD:
            dissolve_threshold = max_cluster_size
        else:
            dissolve_threshold = max(
                round(max_cluster_size * DISSOLVE_CLUSTER_THRESHOLD), 4
            )

        # update the list of clusters for the next iteration
        (
            iter_clusters,
            iter_cluster_limits,
            clusters_to_remove,
            new_clusters_to_add,
        ) = _make_new_clusters_from_reduced_costs(
            iter_clusters,
            reduced_costs_mrj,
            clusters_to_split,
            original_limits,
            nlms,
            t_mrj,
            c_mrj,
            x_mrj,
            q_mrp,
            cells_to_regions,
            dissolve_threshold,
        )

        previous_objective = current_objective

    lumap, lmmap, X_mrj = make_lumap_from_cluster_solution(
        iter_clusters, variables_store, nlus
    )
    end_time = time.time()

    print(f"\nSolve completed in {round(end_time - start_time)} seconds.")

    return lumap, lmmap, X_mrj
