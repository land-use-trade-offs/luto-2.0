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

"""
Find all GBF4 ECNES constraints that are infeasible in the saved MPS model.

Removes ALL ECNES constraints to create a base model, then for each ECNES
constraint, maximizes its LHS to check if the target is achievable.

Usage (Jupyter):
    base_model, results = find_infeasible_ecnes()
    base_model, results = find_infeasible_ecnes("path/to/model.mps")
"""

import os
import tempfile

import gurobipy as gp
from gurobipy import GRB
from joblib import Parallel, delayed
from tqdm.auto import tqdm


DEFAULT_MPS = (
    "F:/Users/jinzhu/Documents/luto-2.0/output"
    "/2026_03_12__10_32_41_RF10_2010-2050/debug_model_2040_2050.mps"
)


def find_infeasible_ecnes(mps_path: str = DEFAULT_MPS, workers: int = 16):
    """
    Returns:
        base_model: Gurobi model with all ECNES constraints removed
        results: list of dicts with keys: name, target, max_lhs, gap, ratio, feasible, n_vars
    """
    # ── Load model ──
    print(f"Loading model from: {mps_path}")
    model = gp.read(mps_path)
    print(f"Model: {model.NumVars} variables, {model.NumConstrs} constraints\n")

    # ── Extract all ECNES constraint info before removing them ──
    ecnes_info = []
    for c in model.getConstrs():
        if "GBF4_ECNES" not in c.ConstrName:
            continue
        row = model.getRow(c)
        terms = [(row.getVar(k).VarName, row.getCoeff(k)) for k in range(row.size())]
        ecnes_info.append(dict(
            constr_name=c.ConstrName,
            rhs=c.RHS,
            terms=terms,
        ))

    print(f"Found {len(ecnes_info)} ECNES constraints\n")
    if not ecnes_info:
        print("No ECNES constraints found.")
        return model, []

    # ── Create base model: remove ALL ECNES constraints ──
    print("Removing all ECNES constraints to create base model...")
    base_model = model.copy()
    for info in ecnes_info:
        c = base_model.getConstrByName(info['constr_name'])
        base_model.remove(c)
    base_model.update()

    # Verify base model feasibility
    print("Checking base model feasibility...")
    base_model.setParam("OutputFlag", 0)
    base_model.optimize()

    if base_model.Status == GRB.INFEASIBLE:
        print("ERROR: Base model (without ANY ECNES) is INFEASIBLE!")
        print("The infeasibility comes from other constraints (demand, GHG, water, etc.)")
        return base_model, []

    print(f"Base model feasible (status={base_model.Status})\n")

    # ── Save base model to temp file for parallel workers ──
    fd, model_file = tempfile.mkstemp(suffix=".mps")
    os.close(fd)
    base_model.write(model_file)

    def _check_one(info):
        """Check one ECNES constraint in a separate process."""
        import gurobipy as _gp
        from gurobipy import GRB as _GRB

        rhs = info['rhs']
        if rhs <= 0:
            return dict(
                constr_name=info['constr_name'], target=rhs, max_lhs=0,
                gap=-rhs, ratio=float('inf'), feasible=True,
                n_vars=len(info['terms']), marker="SKIP_RHS<=0",
            )

        test_model = _gp.read(model_file)
        test_model.setParam("OutputFlag", 0)

        obj = _gp.quicksum(
            coeff * test_model.getVarByName(var_name)
            for var_name, coeff in info['terms']
            if test_model.getVarByName(var_name) is not None
        )
        test_model.setObjective(obj, _GRB.MAXIMIZE)
        test_model.optimize()

        status = test_model.Status
        if status in (_GRB.OPTIMAL, _GRB.SUBOPTIMAL):
            max_lhs = test_model.ObjVal
            gap = rhs - max_lhs
            ratio = max_lhs / rhs if rhs != 0 else float('inf')
            is_feasible = max_lhs >= rhs - 1e-6
            marker = "OK" if is_feasible else "INFEASIBLE"
        elif status == _GRB.INFEASIBLE:
            max_lhs, gap, ratio, is_feasible = float('-inf'), float('inf'), 0, False
            marker = "MODEL_INFEAS"
        else:
            max_lhs, gap, ratio, is_feasible = None, None, None, None
            marker = f"STATUS={status}"

        del test_model
        return dict(
            constr_name=info['constr_name'], target=rhs, max_lhs=max_lhs,
            gap=gap, ratio=ratio, feasible=is_feasible,
            n_vars=len(info['terms']), marker=marker,
        )

    # ── Test each ECNES constraint in parallel ──
    print(f"Testing {len(ecnes_info)} constraints in parallel (n_jobs={workers})...")
    gen = Parallel(n_jobs=workers, return_as='generator')(
        delayed(_check_one)(info) for info in ecnes_info
    )
    results = list(tqdm(gen, total=len(ecnes_info), desc="Checking ECNES"))

    os.remove(model_file)

    for i, r in enumerate(results):
        print(f"[{i+1}/{len(ecnes_info)}] {r['marker']:12s} | "
              f"max={r['max_lhs']:12.4f} target={r['target']:12.4f} gap={r['gap']:12.4f} "
              f"({r['ratio']:.1%}) | {r['constr_name']}")

    # ── Summary ──
    infeasible = [r for r in results if r['feasible'] is False]
    feasible = [r for r in results if r['feasible'] is True]

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(infeasible)} infeasible, {len(feasible)} feasible "
          f"out of {len(ecnes_info)} ECNES constraints")
    print(f"{'='*80}")

    if infeasible:
        infeasible.sort(key=lambda x: -(x['gap'] if x['gap'] != float('inf') else 1e30))
        print(f"\nINFEASIBLE ECNES constraints (sorted by gap, worst first):")
        print(f"{'Community':<65s} {'Target':>12s} {'Max LHS':>12s} {'Gap':>12s} {'Achieved':>10s}")
        print("-" * 115)
        for r in infeasible:
            if r['max_lhs'] == float('-inf'):
                print(f"{r['constr_name']:<65s} {r['target']:12.4f} {'model infeas':>12s} {'N/A':>12s} {'N/A':>10s}")
            else:
                print(f"{r['constr_name']:<65s} {r['target']:12.4f} {r['max_lhs']:12.4f} {r['gap']:12.4f} {r['ratio']:9.1%}")

    if feasible:
        feasible.sort(key=lambda x: x['ratio'])
        print(f"\nFEASIBLE ECNES constraints (tightest first):")
        for r in feasible[:10]:
            print(f"  {r['ratio']:7.1%} achievable | target={r['target']:12.4f} max={r['max_lhs']:12.4f} | {r['constr_name']}")
        if len(feasible) > 10:
            print(f"  ... and {len(feasible) - 10} more feasible constraints")

    return base_model, results
