"""
Worker script: check one constraint against the base model.

Called by PBS jobs submitted from find_infeasible_ecnes.py.

Usage:
    python check_one_constraint.py <task_pkl> <base_model_mps> <result_json>
"""

import json
import pickle
import sys

import gurobipy as gp
from gurobipy import GRB


def main():
    task_path, base_model_path, result_path = sys.argv[1:4]

    with open(task_path, "rb") as f:
        info = pickle.load(f)

    rhs = info["rhs"]
    if rhs <= 0:
        result = dict(
            constr_name=info["constr_name"], target=rhs, max_lhs=0,
            gap=-rhs, ratio=float("inf"), feasible=True,
            n_vars=len(info["terms"]), marker="SKIP_RHS<=0",
        )
    else:
        model = gp.read(base_model_path)
        model.setParam("OutputFlag", 0)

        obj = gp.quicksum(
            coeff * model.getVarByName(var_name)
            for var_name, coeff in info["terms"]
            if model.getVarByName(var_name) is not None
        )
        model.setObjective(obj, GRB.MAXIMIZE)
        model.optimize()

        status = model.Status
        if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            max_lhs = model.ObjVal
            gap = rhs - max_lhs
            ratio = max_lhs / rhs if rhs != 0 else float("inf")
            is_feasible = max_lhs >= rhs - 1e-6
            marker = "OK" if is_feasible else "INFEASIBLE"
        elif status == GRB.INFEASIBLE:
            max_lhs, gap, ratio, is_feasible = float("-inf"), float("inf"), 0, False
            marker = "MODEL_INFEAS"
        else:
            max_lhs, gap, ratio, is_feasible = None, None, None, None
            marker = f"STATUS={status}"

        result = dict(
            constr_name=info["constr_name"], target=rhs, max_lhs=max_lhs,
            gap=gap, ratio=ratio, feasible=is_feasible,
            n_vars=len(info["terms"]), marker=marker,
        )

    with open(result_path, "w") as f:
        json.dump(result, f)
    print(f"Done: {result['marker']} | {result['constr_name']}")


if __name__ == "__main__":
    main()
