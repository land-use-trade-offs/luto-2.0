# Skill: Debug ECNES Infeasibility

This skill automates debugging GBF4 ECNES biodiversity constraint infeasibility in LUTO2. It runs a simulation with ECNES enabled, detects infeasibility from the log, and submits PBS jobs to identify which specific ecological communities are infeasible.

## Steps

### Step 1: Ensure ECNES is enabled in settings

Read `luto/settings.py` and verify that `BIODIVERSITY_TARGET_GBF_4_ECNES = 'on'`. If it is `'off'`, change it to `'on'`.

### Step 2: Run the LUTO simulation

Run the simulation from the project root (`/g/data/jk53/jinzhu/LUTO/luto-2.0`):

```bash
source ~/.bashrc && conda activate luto && python -c "import luto.simulation as sim; data = sim.load_data(); sim.run(data=data)"
```

This will produce an output directory under `output/` named with a timestamp, e.g., `output/2026_03_12__22_50_37_RF5_2010-2050/`.

**IMPORTANT**: This is a long-running command (can take 10+ minutes). Run it in the background and wait for completion.

### Step 3: Check the log for infeasibility

After the simulation completes, find the latest output directory and read the stdout log:

```bash
ls -td output/*/ | head -1
```

Then read the `LUTO_RUN__stdout.log` file in that directory. Look for the pattern:

```
Warning: Gurobi solver did not find an optimal/suboptimal solution for year YYYY. Status: 3
```

- **Status 3** = INFEASIBLE (Gurobi `GRB.INFEASIBLE`)
- If found, the log will also contain: `Saved Gurobi model to output/.../debug_model_XXXX_YYYY.mps`
- The `debug_model_XXXX_YYYY.mps` file is the saved model at the infeasible year, with `XXXX` being the base year and `YYYY` being the target year.

**If the log does NOT contain the infeasibility warning**: The simulation succeeded — no ECNES debugging is needed. Report this to the user and stop.

**If the log DOES contain the infeasibility warning**: Note the MPS file path and proceed to Step 4.

### Step 4: Submit PBS jobs to find infeasible ECNES constraints

Use the `find_infeasible_ecnes` module to submit one PBS job per ECNES constraint. Each job loads the base model (with all ECNES constraints removed), adds back one ECNES constraint's LHS as the objective, and maximizes it to check if the target RHS is achievable.

```python
# Run from the project root
import sys
sys.path.insert(0, '.')
from luto.tests.find_infeasible_ecnes import submit_ecnes_checks

work_dir = submit_ecnes_checks(
    mps_path="output/<timestamp>/debug_model_XXXX_YYYY.mps",
    queue="normalsr",
    ncpus=4,
    mem="32GB",
    walltime="02:00:00",
    project="jk53",
)
```

Replace `<timestamp>` and `XXXX_YYYY` with the actual values from Step 3.

This will:
1. Load the MPS model and extract all `GBF4_ECNES` constraints
2. Remove all ECNES constraints to create a base model
3. Verify the base model is feasible (if not, the infeasibility is from other constraints — stop and report)
4. Save the base model and individual constraint data to `work_dir`
5. Submit one PBS job per ECNES constraint via `qsub`

The function prints the `work_dir` path. Save it for Step 5.

### Step 5: Monitor and collect results

Tell the user to monitor jobs with:
```bash
qstat -u $USER
```

Once all jobs are complete, collect results:

```python
from luto.tests.find_infeasible_ecnes import collect_results
results = collect_results("<work_dir>")
```

This prints a summary showing:
- **INFEASIBLE** constraints: the ecological community target cannot be met even when it's the only ECNES constraint. These communities have targets that exceed the maximum achievable area.
- **FEASIBLE** constraints: the target is individually achievable. These are not the root cause of infeasibility (though combinations may still conflict).

### Step 6: Report findings to the user

Summarize:
1. Which year the simulation became infeasible
2. How many ECNES constraints were tested
3. Which ecological communities are individually infeasible (with their target vs. maximum achievable values)
4. Suggest the user consider relaxing targets for the infeasible communities, or switching `BIODIVERSITY_TARGET_GBF_4_ECNES` to `'off'`

## File Locations

| File | Purpose |
|------|---------|
| `luto/settings.py` | ECNES on/off setting (`BIODIVERSITY_TARGET_GBF_4_ECNES`) |
| `luto/simulation.py` | Main simulation, generates `debug_model_*.mps` on infeasibility |
| `luto/tests/find_infeasible_ecnes.py` | `submit_ecnes_checks()` and `collect_results()` |
| `luto/tests/check_one_ecnes.py` | Worker script called by each PBS job |
| `output/<timestamp>/LUTO_RUN__stdout.log` | Simulation log to check for infeasibility |
| `output/<timestamp>/debug_model_XXXX_YYYY.mps` | Saved Gurobi model at infeasible year |
