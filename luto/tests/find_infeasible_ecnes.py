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

Removes ALL ECNES constraints to create a base model, then submits each ECNES
constraint check as a separate PBS job on NCI Gadi.

Usage:
    # Step 1: Prepare and submit PBS jobs
    from luto.tests.find_infeasible_ecnes import submit_ecnes_checks
    submit_ecnes_checks("path/to/model.mps")

    # Step 2: After all jobs finish, collect results
    from luto.tests.find_infeasible_ecnes import collect_results
    results = collect_results("path/to/ecnes_workdir")
"""

import json
import os
import pickle
import subprocess
import time

from tqdm.auto import tqdm

import gurobipy as gp
from gurobipy import GRB

# Worker script called by each PBS job (absolute path)
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), 'check_one_constraint.py')

DEFAULT_MPS = (
    "/g/data/jk53/jinzhu/LUTO/luto-2.0/output/"
    "2026_03_12__22_50_37_RF5_2010-2050/debug_model_2010_2015.mps"
)


def submit_ecnes_checks(
    mps_path: str = DEFAULT_MPS,
    work_dir: str | None = None,
    queue: str = "normalsr",
    ncpus: int = 16,
    mem: str = "64GB",
    walltime: str = "02:00:00",
    project: str = "jk53",
    max_concurrent: int = 200,
):
    """
    Extract ECNES constraints, create a base model, and submit one PBS job
    per constraint.

    Args:
        mps_path: Path to the saved MPS model file.
        work_dir: Directory for intermediate files and results. Defaults to
                  a subdirectory next to the MPS file.
        queue: PBS queue name.
        ncpus: CPUs per PBS job.
        mem: Memory per PBS job.
        walltime: Wall time per PBS job.
        project: NCI project code for storage and accounting.
        max_concurrent: Maximum number of concurrent PBS jobs.

    Returns:
        work_dir: Path to the working directory (needed for collect_results).
    """
    # ── Load model ──
    print(f"Loading model from: {mps_path}")
    model = gp.read(mps_path)
    print(f"Model: {model.NumVars} variables, {model.NumConstrs} constraints\n")

    # ── Extract all ECNES constraint info ──
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
        return None

    # ── Create base model: remove ALL ECNES constraints ──
    print("Removing all ECNES constraints to create base model...")
    base_model = model.copy()
    for info in ecnes_info:
        c = base_model.getConstrByName(info['constr_name'])
        base_model.remove(c)
    base_model.update()

    # # Verify base model feasibility
    # print("Checking base model feasibility...")
    # base_model.setParam("OutputFlag", 0)
    # base_model.optimize()

    # if base_model.Status == GRB.INFEASIBLE:
    #     print("ERROR: Base model (without ANY ECNES) is INFEASIBLE!")
    #     print("The infeasibility comes from other constraints (demand, GHG, water, etc.)")
    #     return None

    # print(f"Base model feasible (status={base_model.Status})\n")

    # ── Set up working directory ──
    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(mps_path), "ecnes_checks")
    os.makedirs(work_dir, exist_ok=True)
    results_dir = os.path.join(work_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    logs_dir = os.path.join(work_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Save base model
    base_model_path = os.path.join(work_dir, "base_model.mps")
    print(f"Saving base model to: {base_model_path}")
    base_model.write(base_model_path)

    # Save constraint info
    constraints_path = os.path.join(work_dir, "ecnes_constraints.pkl")
    with open(constraints_path, "wb") as f:
        pickle.dump(ecnes_info, f)

    # ── Submit one PBS job per constraint ──
    print(f"Submitting {len(ecnes_info)} PBS jobs (queue={queue}, ncpus={ncpus}, mem={mem})...")
    job_ids = []

    for idx, info in tqdm(enumerate(ecnes_info), total=len(ecnes_info), desc="Submitting PBS jobs"):
        # Save individual constraint info
        task_path = os.path.join(work_dir, f"task_{idx:04d}.pkl")
        with open(task_path, "wb") as f:
            pickle.dump(info, f)

        result_path = os.path.join(results_dir, f"result_{idx:04d}.json")

        # Generate the PBS script
        pbs_script = (
            f"#!/bin/bash\n"
            f"#PBS -N ecnes_{idx:04d}\n"
            f"#PBS -q {queue}\n"
            f"#PBS -l storage=scratch/{project}+gdata/{project}\n"
            f"#PBS -l ncpus={ncpus}\n"
            f"#PBS -l mem={mem}\n"
            f"#PBS -l jobfs=10GB\n"
            f"#PBS -l walltime={walltime}\n"
            f"#PBS -o {logs_dir}/ecnes_{idx:04d}.out\n"
            f"#PBS -e {logs_dir}/ecnes_{idx:04d}.err\n"
            f"\n"
            f"source ~/.bashrc\n"
            f"conda activate luto\n"
            f"\n"
            f"python {WORKER_SCRIPT} \\\n"
            f"    {task_path} \\\n"
            f"    {base_model_path} \\\n"
            f"    {result_path}\n"
        )

        pbs_path = os.path.join(work_dir, f"job_{idx:04d}.sh")
        with open(pbs_path, "w") as f:
            f.write(pbs_script)

        # Throttle concurrent jobs
        while True:
            try:
                running = subprocess.run(
                    "qselect | wc -l", shell=True,
                    capture_output=True, text=True, timeout=30,
                )
                n_running = int(running.stdout.strip())
                if n_running < max_concurrent:
                    break
                print(f"  Waiting: {n_running}/{max_concurrent} jobs running...")
                time.sleep(10)
            except Exception:
                break

        # Submit the job
        result = subprocess.run(
            ["qsub", pbs_path],
            capture_output=True, text=True,
        )
        job_id = result.stdout.strip()
        job_ids.append(job_id)

    # Save job IDs for tracking
    with open(os.path.join(work_dir, "job_ids.txt"), "w") as f:
        f.write("\n".join(job_ids))

    print(f"\nAll {len(ecnes_info)} jobs submitted.")
    print(f"Working directory: {work_dir}")
    print(f"\nMonitor with:  qstat -u $USER")
    print(f"Collect with:  from luto.tests.find_infeasible_ecnes import collect_results")
    print(f"               collect_results('{work_dir}')")

    return work_dir


def collect_results(work_dir: str):
    """
    Collect results from completed PBS jobs and print summary.

    Args:
        work_dir: Path returned by submit_ecnes_checks().

    Returns:
        results: list of dicts with keys: constr_name, target, max_lhs, gap, ratio, feasible, n_vars, marker
    """
    results_dir = os.path.join(work_dir, "results")

    # Load original constraint info for the total count
    with open(os.path.join(work_dir, "ecnes_constraints.pkl"), "rb") as f:
        ecnes_info = pickle.load(f)
    n_total = len(ecnes_info)

    # Collect available results
    results = []
    missing = []
    for idx in range(n_total):
        result_path = os.path.join(results_dir, f"result_{idx:04d}.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(json.load(f))
        else:
            missing.append(idx)

    print(f"Collected {len(results)}/{n_total} results")
    if missing:
        print(f"Missing results for {len(missing)} constraints (indices: {missing[:20]}{'...' if len(missing) > 20 else ''})")
        print("Check logs in:", os.path.join(work_dir, "logs"))
        print()

    if not results:
        return results

    # Print individual results
    for i, r in enumerate(results):
        max_lhs = r['max_lhs'] if r['max_lhs'] is not None else float('nan')
        target = r['target'] if r['target'] is not None else float('nan')
        gap = r['gap'] if r['gap'] is not None else float('nan')
        ratio = r['ratio'] if r['ratio'] is not None else float('nan')
        print(f"[{i+1}/{len(results)}] {r['marker']:12s} | "
              f"max={max_lhs:12.4f} target={target:12.4f} gap={gap:12.4f} "
              f"({ratio:.1%}) | {r['constr_name']}")

    # ── Summary ──
    infeasible = [r for r in results if r['feasible'] is False]
    feasible = [r for r in results if r['feasible'] is True]

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(infeasible)} infeasible, {len(feasible)} feasible "
          f"out of {len(results)} completed checks ({n_total} total)")
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

    # ── Save summary to work_dir ──
    summary_path = os.path.join(work_dir, "summary.json")
    summary = {
        "n_total": n_total,
        "n_completed": len(results),
        "n_missing": len(missing),
        "n_infeasible": len(infeasible),
        "n_feasible": len(feasible),
        "infeasible": sorted(infeasible, key=lambda x: -(x['gap'] if x['gap'] != float('inf') else 1e30)),
        "feasible": sorted(feasible, key=lambda x: x['ratio']),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")

