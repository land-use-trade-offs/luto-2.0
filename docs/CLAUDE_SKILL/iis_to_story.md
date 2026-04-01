# Skill: Translate IIS Analysis to Plain-Language Story

Given a task run directory (containing `Run_G*/` subdirs and `merged_grid_search_parameters_unique.csv`), read the IIS summaries, ILP files, and PBS stdout logs and produce a plain-language story table grouped by scenario.

---

## Inputs Required

| File | Location | Purpose |
|------|----------|---------|
| `iis_analysis_summary.txt` | `Run_G*/` | Constraint names, variable counts, bound values |
| `debug_model_*.ilp` | `Run_G*/` | Exact LP constraint equations and bounds |
| `run_*.o*` (PBS stdout) | `Run_G*/` | Gurobi barrier tables, NumericFocus retries |
| `merged_grid_search_parameters_unique.csv` | run root | scenario_group, local_run_idx, key settings |

---

## Step 1 — Find all infeasible runs

```python
from pathlib import Path
import pandas as pd

run_root = Path("/g/data/jk53/jinzhu/LUTO/Custom_runs/<TASK_DIR>")

# All runs that have an IIS summary
iis_runs = sorted(run_root.glob("Run_G*/iis_analysis_summary.txt"))
run_ids = [int(p.parent.name.replace("Run_G", "")) for p in iis_runs]

# Load parameters to get scenario group and local index
params = pd.read_csv(run_root / "merged_grid_search_parameters_unique.csv")
infeasible = params[params["global_run_idx"].isin(run_ids)].copy()
print(infeasible[["scenario_group","global_run_idx","local_run_idx"]].to_string())
```

---

## Step 2 — Read evidence for each run

For each infeasible run, collect three pieces of evidence:

### 2a. ILP file — exact constraint and bound

```python
ilp_path = run_root / f"Run_G{run_id:04d}" / "debug_model_*.ilp"
# Use glob to find it, then read
ilp_text = next(Path(run_root / f"Run_G{run_id:04d}").glob("debug_model_*.ilp")).read_text()
print(ilp_text)   # small file — always print in full
```

Key things to extract from the ILP:
- The constraint(s) in `Subject To` — cell usage equalities, ag-mam inequalities, `bio_GBF2_*`
- The bound(s) in `Bounds` — lower bounds on non-ag or ag-man variables
- Count of `>=` lines to judge scale (1–2 = single-cell; thousands = GBF2)

### 2b. IIS summary — named constraint and variable counts

```python
summary = (run_root / f"Run_G{run_id:04d}" / "iis_analysis_summary.txt").read_text()
print(summary)
```

Key things to extract:
- Named constraint (e.g. `Biodiversity GBF2: priority degraded area >= N`)
- `Variables: X total, Y locked, Z free`
- Standalone bounds list (HIR bounds = `ag_man dryland HIR-... [free]`)

### 2c. PBS stdout — solver behaviour

```python
import glob
stdout = sorted(glob.glob(str(run_root / f"Run_G{run_id:04d}/run_*.o*")))[-1]
lines = Path(stdout).read_text().splitlines()
# Find Barrier iteration table around infeasibility
for i, l in enumerate(lines):
    if "Barrier performed" in l or "Infeasible model" in l or "Numerical trouble" in l \
       or "NumericFocus" in l or "Solver status" in l:
        print(f"{i}: {l}")
```

Look for:
- How many barrier iterations ran before `Infeasible model`
- Whether `Numerical trouble encountered` appears
- Whether `Trying NumericFocus=2` appears — and whether the retry also fails
- Primal infeasibility column in last ~10 barrier rows (stuck at ~1e-2 = stagnation; large jump = numerical blow-up)

---

## Step 3 — Diagnose each run using these rules

### Rule A — Numerical stagnation (false infeasibility)

**ILP signals:** Scale is large (thousands of constraints) OR the single-cell ILP looks identical to a Rule B/D case but barrier stagnated first.

**PBS stdout signals:**
- Barrier ran >100 iterations before `Infeasible model`
- Primal infeasibility column stuck at ~1e-2 for the last 20+ rows
- Dual bound diverges at final rows (e.g. -1e6 → -1e9 → -2e13)
- `Numerical trouble encountered` present
- `Trying NumericFocus=2` retry also fails after many iterations (>100)

**Story:** "False infeasibility — Gurobi ran out of steam numerically. After N barrier iterations the primal residual was still stuck at ~1e-2 and never converged. Gurobi reported 'Infeasible' but the model likely has a feasible solution — the relaxed tolerances we use caused stagnation, not true structural conflict. Retry with `NumericFocus=3` (much slower) or tighter `BARRIER_CONVERGENCE_TOLERANCE`."

---

### Rule B — Non-ag lb > cell total (float32 rounding artefact)

**ILP signals:**
- Exactly 1 constraint + 1 bound in the ILP
- Constraint: `X_non_ag_K_CELL + X_ag_dry_... = CELL_TOTAL` (cell usage equality)
- Bound: `X_non_ag_K_CELL >= LB` where `LB > CELL_TOTAL`
- Gap `LB - CELL_TOTAL` is tiny: 3e-8 to 6e-8 (float32 rounding)
- All variables in the constraint have coefficient 0.0 in the objective (zero productive area)

**IIS summary signals:** `1 constraint(s), 1 bound(s)`, single cell, non_ag land use type shown

**Story:** "True infeasibility (rounding artefact) — [non_ag type] at cell [CELL] was assigned [VALUE] in [PREV_YEAR]. Float32 stores this as lower bound [LB] which is fractionally larger than the cell total [CELL_TOTAL] stored in the next model (gap = [GAP]). Since every other land use in this cell has zero productive area, there is no slack to absorb the discrepancy. Fix: clamp `lb = min(lb, cell_total)` when propagating non-ag lower bounds between years."

---

### Rule C — GBF2 target structurally unachievable

**ILP signals:**
- Named constraint `bio_GBF2_priority_degraded_area_limit` with thousands of `X_non_ag_0_*` variables
- Scale: thousands of `>=` lines in bounds

**IIS summary signals:**
- `Biodiversity GBF2: priority degraded area >= N` named constraint
- `Variables: X total, Y locked, Z free` — Y >> Z (most cells locked)
- Standalone bounds: 100+ `ag_man dryland HIR-... [free]` entries

**PBS stdout signals:**
- Solver terminates quickly under NumericFocus=2 (50–70 iterations, not hundreds)
- Dual bound crashes immediately to -1e10 or below — unbounded dual = provably infeasible primal

**Story:** "True infeasibility — GBF2 target unreachable. The GBF2 constraint requires [TARGET] ha of priority degraded area to be under Env Plantings by [YEAR]. Of [TOTAL] eligible cells, [LOCKED] are already locked by HIR assignments carried forward from [PREV_YEAR], leaving only [FREE] free cells. Their combined area falls short of the target. The dual bound crash to -1e11 in just [N] iterations confirms this is provably infeasible (not a numerics issue). Fix: reduce the GBF2 target, or prevent HIR adoption in GBF2-priority cells."

---

### Rule D — HIR lb > cell total (irreversibility conflict)

**ILP signals:**
- Exactly 2 constraints + 2 bounds in the ILP
- Constraint 1: `X_ag_dry_K_CELL + other_lu... = CELL_TOTAL` (cell usage equality)
- Constraint 2: `- X_ag_dry_K_CELL + X_ag_man_hir_K_CELL <= 0` (ag-man within land use)
- Bound: `X_ag_man_hir_K_CELL >= LB` where `LB > CELL_TOTAL`
- Gap `LB - CELL_TOTAL` is ~3e-6 (larger than float32 rounding — HIR was overfitted in prior year)

**Story:** "True infeasibility (HIR over-allocation) — HIR-[type] was assigned [LB] at cell [CELL] in [PREV_YEAR]. Because HIR is irreversible, that value becomes a lower bound carried into [YEAR]. But `X_ag_man_hir <= X_ag_dry_[LU]` and `X_ag_dry_[LU] <= cell_total = [CELL_TOTAL]`, so `X_ag_man_hir` can never reach [LB]. Gap = [GAP]. Fix: clamp HIR lower bounds to the cell total when propagating between years."

---

## Step 4 — Output format

Produce one markdown table per scenario group (only groups with infeasible runs), then a cross-group summary paragraph.

### Table columns

| Run | Local # | Year Failed | Truly Infeasible? | What happened |
|-----|---------|-------------|-------------------|---------------|

- **Run**: `G0007`
- **Local #**: local run index within the scenario group
- **Year Failed**: `2035–2040`
- **Truly Infeasible?**: `No — numerical stagnation` / `Yes — rounding (3e-8 gap)` / `Yes — GBF2 target` / `Yes — HIR conflict`
- **What happened**: 2–3 sentences from the story template, with actual numbers filled in

### Cross-group summary

After all tables, write one paragraph identifying: (1) which failures are likely solvable by fixing solver numerics, (2) which share the lb-clamping bug, and (3) which are genuinely structurally infeasible by design.

---

## LUTO Domain Reference

| Term | Meaning |
|------|---------|
| **HIR** | High Integrity Rangeland — irreversible ag-management; lower bound carried forward to all future years |
| **Non-ag lb** | Lower bound on non-ag allocation from prior year's solution, carried forward by the solver |
| **Cell total** | Fraction of a spatial cell occupied by LUTO modelled land; stored as float32; value can differ by ~3e-8 between years |
| **GBF2** | Global Biodiversity Framework Goal 2 — total priority degraded area under restoration; single inequality over thousands of cells |
| **Numerical stagnation** | Gurobi barrier reports "Infeasible" after many iterations when primal residual never converges; not a true infeasibility |
| **NumericFocus** | Gurobi parameter 0–3 controlling arithmetic precision; model retries at 2 on first stagnation; 3 not currently tried |
| **IIS** | Irreducible Infeasible Subsystem — the minimal set of constraints whose joint removal restores feasibility |
| **ILP file** | The IIS written out as a small LP file — shows exact constraints and bounds that are in conflict |
