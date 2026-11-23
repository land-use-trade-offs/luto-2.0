# Skill: Finding what makes a year infeasible or numerically unstable

Use this when a LUTO year does not solve: Gurobi reports `Numerical trouble encountered`, `Sub-optimal termination`,
`Infeasible`, `Infeasible or unbounded`, or barrier runs to its iteration limit without converging.

**The method is elimination.** Take the exact model Gurobi was given, change ONE thing at a time, and see which change
makes the failure go away. Do not start from a guess about the cause (coefficient range, a tight target, small values):
on the case this skill was written from, two days of tests on such guesses found nothing, and one round of elimination
found the cause (the worked example at the end).

The tool is `diagnose_mps.py`, kept in full in the last section of this guide (no separate file in the repo). It reads
the model file alone, imports no `luto`, and tells the row families apart by their constraint names. Write it out next
to the model before the first step (a working folder, a scratch folder on the cluster; not the repo):

```bash
sed -n '/^```python title="diagnose_mps.py"$/,/^```$/p' docs/CLAUDE_SKILL/diagnose_infeasible_or_numerical.md | sed '1d;$d' > diagnose_mps.py
python diagnose_mps.py --help
```

A change to the tool is made in this guide, so the next use starts from it.

---

## Step 0: Read what the run already says

Before any new solve, read the failing year's log and `out_<year>/` folder.

| Look at | What it tells |
|---|---|
| The solver's last lines | Which failure: `Infeasible`, `Numerical trouble`, `Sub-optimal termination`, iteration limit |
| The barrier table (`Objective`, `Residual` columns) | Where the primal residual stalls, and at which iteration |
| `Matrix range`, `Objective range`, `RHS range` | The model's ranges, to compare with a year that solved |
| `out_<year>/bound_report_<year>.csv` | Rows that are IMPOSSIBLE on their own (bound propagation). If one is listed, that is the answer: stop here |
| `out_<year>/shortfall_<year>.csv` (with `ELASTIC_FAMILIES`) | Rows that fell short, and by how much |

Bound propagation proves single-row facts only. A row made impossible, or redundant, by a SET of other rows is
invisible to it. That is what the steps below are for.

A solve that leaves barrier for simplex has already failed: stop it. Simplex on a full-resolution model can run for
days and its answer is not what the model would have produced.

## Step 1: Get the exact model

Set `WRITE_DEBUG_MPS = True` and rerun the failing year from its checkpoint, with every other setting unchanged. The
model is written to `<run>/debug_model_<base>_<year>.mps` before the solve, so the run may fail or be killed
afterwards. With `CELL_GROUPING` it is the grouped model, which is the one Gurobi saw.

- Never diagnose on a rebuilt approximation (another resolution, another year, a simplified build). The cause may not
  be in it.
- When the run is someone else's, copy their checkpoint, code and inputs and work on the copy.
- Building a RES1 year needs the memory of the run itself; analysing its MPS does not (a model of 8 M rows and 49 M entries used 16 to 18 GB
  with 24 CPUs; 28 GB with two such models loaded).

## Step 2: Reproduce, and split "infeasible" from "cannot reach the optimum"

Three solves, in parallel. Give the script the failing run's solver settings (`--scale-flag`, `--opt-tol`,
`--feas-tol`, `--bar-conv-tol`, `--threads`).

```bash
python diagnose_mps.py <model.mps> stats                     # no solve: ranges per family
python diagnose_mps.py <model.mps> reproduce --threads 24    # does the file fail as the run did?
python diagnose_mps.py <model.mps> feas      --threads 24    # zero objective
python diagnose_mps.py <model.mps> relax     --threads 24    # violation variable on every wide row
```

| `feas` | `relax` | Reading | Go to |
|---|---|---|---|
| solves | total violation ≈ 0 | The constraints are consistent. The trouble is in reaching the optimum | Step 3 |
| fails | names rows with a violation | **Infeasible.** The listed rows cannot hold together, by the listed amounts | Step 3, on those rows |
| fails | total violation ≈ 0 | Feasible but unstable even without an objective: suspect the structural rows or the bounds (`stats`: `lb > ub`, tiny right-hand sides) | Step 3 with `drop:structural/...` |

`relax` reports violations in the model's scaled units; multiply by the row's `scale` (`shadow_prices_<year>.csv`) for
tonnes, hectares or megalitres.

A failure can be borderline: the same file may give `Numerical trouble` in one run and crawl to the iteration limit in
another. Compare runs by the residual at the same iteration, not only by pass or fail.

## Step 3: Remove one row family at a time

One job per wide family, all at once:

```bash
for family in demand ghg water ag_mgt_adoption regional_adoption GBF2 GBF3_NVIS GBF4_SNES GBF4_ECNES GBF8 renewable_target; do
    python diagnose_mps.py <model.mps> drop:$family --threads 24 > drop_$family.log
done
```

(Submit each as its own job on a cluster. A family the model does not hold stops with a message naming the families it
does hold.)

How to read the results:

| Outcome of `drop:<family>` | Reading |
|---|---|
| Still fails | Not this family. Cross it off |
| Solves cleanly (residuals fall steadily, a few hundred iterations) and the objective stays in the same range | **The cause is in this family.** Go to Step 4 |
| Solves, but the objective jumps by an order of magnitude | Removing it made a different problem (demand usually does). Not evidence |
| Nothing helps | Try the objective (`feas` already did), then pairs of families, then the structural rows |

## Step 4: Look at the suspect family row by row

```bash
python diagnose_mps.py <model.mps> rows:<family>
```

One line per row, written to `diagnose_rows.csv`. What to look for:

| Column | Suspicious when |
|---|---|
| `negative_over_positive` | Exactly 1 on a "share ≤ limit × total" row: a 100 % limit, which other rows already impose |
| `on_the_box` | `redundant` (the bounds alone guarantee it) or `impossible` |
| `rhs` = 0 with a wide `a_min` to `a_max` | A homogeneous row whose rescale has nothing to anchor on |
| `positive` / `negative` counts | A row with one sign only that should have two, or the reverse |
| A few rows unlike the rest of their family | Those are the candidates |

`structure` (no solve) looks for the same thing over the whole model: rows the column bounds pin, make redundant or
make impossible, and rows that are a multiple of another row.

```bash
python diagnose_mps.py <model.mps> structure
```

Then ask what else in the model says the same thing as the candidate rows. A row that repeats other rows (their sum, or
implied by them) adds no restriction, but it has no interior, and barrier's linear system becomes close to singular.

## Step 5: Split the family, both ways

Removing the candidates must cure it, AND keeping only the candidates must still fail:

```bash
python diagnose_mps.py <model.mps> "drop:<family>/<regex of the candidates>" --threads 24
python diagnose_mps.py <model.mps> "drop:<family>/<regex of the others>"     --threads 24
```

Check the `removed N rows: {...}` line of each log before reading the result: the count and the family must be what
was meant. (The selector always names a family for this reason: option names such as `HIR` also appear in millions of
`ag_mgt_link` row names.)

## Step 6: Compare with a variant that solves

If a nearby configuration of the same year solves (no cell grouping, no conditioning of the grouped rows, another
tolerance), export its model too and compare. It separates "these rows" from "what was done to them": the cause may
need both, and then removing either one is a fix.

## Step 7: Confirm in LUTO, then fix at the source

1. Rerun the year in LUTO with the candidate rows switched off (`rows['active']` False before `build_solver`, as a
   run-time patch in a test script, not an edit of `luto/`).
2. Check the solve, and the targets of the full model at the solution (`grouping_<year>.csv` with cell grouping): a
   row that was truly redundant leaves every target where it was.
3. Fix where the row is built (`row_builder.get_<family>`), not in the solver's parameters.
4. Record the result in `docs/FINDINGS.md`.

## Step 8: When the cause is a real constraint that must stay

Elimination may end on a row that cannot be removed: a target that binds (the second worked example). Then, and only
then, try the solver's settings, ONE per solve, on the same exported model:

```bash
python diagnose_mps.py <model.mps> reproduce --threads 24 --scale-flag 3
python diagnose_mps.py <model.mps> reproduce --threads 24 --numeric-focus 2
python diagnose_mps.py <model.mps> reproduce --threads 24 --param Aggregate=0     # any Gurobi parameter
```

- Compare iterations and work units (machine-independent), and the objectives of the settings that solve: they must
  agree.
- A setting that passes on the exported model is a candidate, not a result: confirm it in the run's own
  configuration. On the second worked example the same setting took 278 iterations on the exported model and 550 in
  production.
- Also ask why the row is hard in THIS year: compare its rhs and shadow price with the years around it. A jump in a
  target series is an input question before it is a solver question.

---

## What does not find the cause

- **`NumericFocus`, another seed, tighter or looser tolerances, as the first move.** They may get one year through;
  the cause stays. They have their place at the end (Step 8), once elimination has shown the cause cannot be removed.
- **Dropping small coefficients or narrowing the matrix range.** It changes the answer (a net row such as GHG loses
  accuracy fast), and on the worked example it never touched the rows at fault.
- **`computeIIS`.** Useful on a coarse model (RES5 and coarser); too slow at full resolution, and it answers
  infeasibility only.
- **Judging a change by the solver's own objective and time.** A model that solves faster because rows were weakened
  may break the full model's targets. Check the targets.

## Cost

On Gadi, one solve of a full-resolution grouped model takes 0.1 to 3.4 h on 24 CPUs of the normal queue (about 48 SU
an hour): a full round of Step 3 is 10 to 12 jobs. Stop a job as soon as its outcome is clear: a run that has passed
the failing iteration with residuals still falling has answered the question.

---

## Worked example: RES1, 2042, cell grouping with drop 1e-6 and rescale (2026-11-22)

Barrier stalled with the primal residual near 700 at iteration 460 and reported `Numerical trouble`; simplex then ran
for more than 15 hours.

| Step | Result |
|---|---|
| 2 | `reproduce`: 1,000 iterations, not converged. `feas`: solved in 215. `relax`: total violation 1.7e-8. Feasible; the trouble is in the optimisation |
| 3 | Without GHG, water or GBF2: still failing. Objective entries under 0.01 zeroed: still failing. Without demand: solved, but objective 27 times larger (another problem). **Without the 65 `ag_mgt_adoption` rows: solved in 321 iterations** |
| 4 | `rows:ag_mgt_adoption`: 59 rows with `negative_over_positive` between 0.146 and 0.95; **6 rows at exactly 1.0** (HIR - Beef, HIR - Sheep, four Savanna Burning) |
| 4 | At a 100 % limit the row says "adopted area ≤ area of the land use". The `ag_mgt_link` rows (X_am − X_ag ≤ 0) say it column by column: the two HIR rows equal Σ (cells of the blob) × (its link row), entry for entry, and 18,697 of those link rows are pinned at equality by the bounds (HIR is not reversible) |
| 5 | Without the 6 rows: **solved in 408 iterations**. Without the other 59 (the 6 kept): `Numerical trouble` at 355 |
| 6 | The same year with no conditioning of the grouped rows holds the same 6 rows and solves in 261 iterations. Giving the adoption rows back their unconditioned scale, the 6 kept: still not converging after 762 |

Cause: six redundant rows together with the conditioning of the grouped wide rows. Removing either one cures it. The
fix at the source is not to build an adoption row whose limit is 100 % or more.

What it was not: small coefficients, the matrix or objective range, the GHG, water or GBF2 targets, the node-balance
rows.

## Second worked example: RES1, 2021, the GHG row in the first year it binds (2026-11-23)

Barrier ended in `Sub-optimal termination` after 699 iterations (809 on the exported model); the years around it
solve in about 300.

| Step | Result |
|---|---|
| 2 | `feas` solved in 30 iterations, `relax` total violation 4e-13: feasible |
| 3 | **Without the one GHG row: solved in 276 iterations.** Without water 475, without GBF2 584; without the adoption rows or the source-cap rows still sub-optimal |
| 4 | `structure`: no row is a multiple of another; the pinned node-balance rows are as many as in a year that solves. Nothing to remove |
| 0, again | The GHG limit falls from 73.9 to 47.5 Mt between 2020 and 2021 (the series changes from its historical reference to the target pathway); the row's shadow price is 0 up to 2020 and 89.5 AUD / tCO2e in 2021 |
| 8 | ScaleFlag 3 (the run used 2): solved in 278 iterations, 1,102 work units. NumericFocus 2 or 3: 249 iterations, 2,922 work units. Aggregate 0: 535 iterations. Presolve 2 and BarHomogeneous 0: still failing. The three objectives agree, and with the run's own simplex result |
| 7 | In the run's own configuration ScaleFlag 3 solved the year in 550 iterations (sub-optimal at 626 with ScaleFlag 2) |

Cause: a real constraint, binding for the first time after a one-year jump in its target. Nothing to remove; a solver
setting gets the year through, and the jump in the input series is the question to take back to the data.

Full record of both examples: `docs/FINDINGS.md`, 20261123.

---

## The script: `diagnose_mps.py`

```python title="diagnose_mps.py"
"""Find what makes a LUTO year infeasible or numerically unstable, on the model Gurobi was given.

The model is the debug MPS of the failing year (settings.WRITE_DEBUG_MPS = True writes
<run>/debug_model_<base>_<year>.mps before the solve). This script needs nothing else: it imports no luto, and tells the
row families apart by the constraint names row_builder gives them (GROUPS below). The method, the reading rules and a
worked examples are in docs/CLAUDE_SKILL/diagnose_infeasible_or_numerical.md, which holds this script (its last section).

Usage: python diagnose_mps.py <model.mps> <mode> [options]

mode
  stats             no solve: per row family the rows, entries, |a| range, zero / tiny right-hand sides; the columns
                    fixed by their bounds; the model's ranges
  structure         no solve: per row family, what the column bounds alone say of its rows (redundant / impossible /
                    pinned = the bounds leave the row no slack at all / open), and the rows that are multiples of another
                    row (the same constraint twice, a range, or two that contradict)
  rows:<selector>   no solve: every selected row on its own line - entries, positive / negative, |a| range, the ratio of
                    its negative to its positive entries (the limit of a "share <= limit x total" row), its activity
                    interval over the column bounds and what that says (redundant / impossible / open on the box)
  reproduce         the model as it is: does this file fail the way the run did?
  feas              the same constraints, zero objective: is the constraint set the problem, or the objective?
  relax[:<selector>]  a violation variable on every selected row (default: every wide row), objective = total
                    violation: which rows cannot hold together, and by how much (scaled units)
  drop:<selector>   the model without the selected rows, objective kept

selector            <family>            every row of that family (a name in GROUPS, or "wide", "structural", "all")
                    <family>/<regex>    its rows whose name matches the regex
                    A family is always named, so a pattern cannot reach into another family by accident.

Every solve is barrier as luto runs it (crossover off) and stops at "Numerical trouble" / "Sub-optimal termination" or
when Gurobi starts simplex: a model that leaves barrier has failed, and the simplex that follows can run for days.
Set the solver options to the failing run's (they are printed at the top of its log).
"""
import argparse
import re
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

# Row families by the constraint names luto/solvers/row_builder.py writes, first match wins.
GROUPS = [
    ('demand',            r'^demand_'),
    ('ghg',               r'^ghg_'),
    ('ag_mgt_adoption',   r'^const_ag_man_adoption_limit_'),
    ('regional_adoption', r'^reg_adopt_limit_'),
    ('water',             r'^water_yield_limit_'),
    ('renewable_target',  r'^renewable_.*_target_'),
    ('GBF2',              r'^bio_GBF2_'),
    ('GBF3_NVIS',         r'^bio_GBF3_'),
    ('GBF4_SNES',         r'^bio_GBF4_SNES_'),
    ('GBF4_ECNES',        r'^bio_GBF4_ECNES_'),
    ('GBF8',              r'^bio_GBF8_'),
    ('ag_mgt_link',       r'^const_ag_man_.*_usage_'),
    ('renewable_ceiling', r'^const_.*_solvable_ub_'),
    ('source_cap',        r'^srccap_'),
    ('node_balance',      r'^bal_'),
]
STRUCTURAL = ('ag_mgt_link', 'renewable_ceiling', 'source_cap', 'node_balance')      # millions of rows, a few entries each


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', type=Path)
    parser.add_argument('mode')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--scale-flag', type=int, default=2, help='settings.SCALE_FLAG')
    parser.add_argument('--feas-tol', type=float, default=1e-6, help='settings.FEASIBILITY_TOLERANCE')
    parser.add_argument('--opt-tol', type=float, default=1e-2, help='settings.OPTIMALITY_TOLERANCE')
    parser.add_argument('--bar-conv-tol', type=float, default=1e-5, help='settings.BARRIER_CONVERGENCE_TOLERANCE')
    parser.add_argument('--bar-homogeneous', type=int, default=1)
    parser.add_argument('--numeric-focus', type=int, default=0)
    parser.add_argument('--bar-iter-limit', type=int, default=1000)
    parser.add_argument('--param', action='append', default=[], metavar='NAME=VALUE',
                        help='any other Gurobi parameter, set last (repeatable): --param Aggregate=0 --param Presolve=2')
    parser.add_argument('--out', type=Path, default=None, help='folder of the CSVs (default: beside the model)')
    return parser.parse_args()


def family_of(names):
    """The family of every constraint, by its name; 'other' where no pattern of GROUPS matches."""
    family = np.full(len(names), 'other', dtype=object)
    open_rows = np.ones(len(names), dtype=bool)
    series = pd.Series(names)
    for group, pattern in GROUPS:
        hit = open_rows & series.str.contains(pattern, regex=True).values
        family[hit] = group
        open_rows &= ~hit
    return family


def select(selector, names, family):
    """The rows a selector names: <family>, or <family>/<regex> on the names of that family's rows."""
    group, _, pattern = selector.partition('/')
    if group == 'all':
        mask = np.ones(len(names), dtype=bool)
    elif group == 'wide':
        mask = ~np.isin(family, STRUCTURAL)
    elif group == 'structural':
        mask = np.isin(family, STRUCTURAL)
    else:
        assert group in set(family), f"no row of family {group!r}; the model holds {sorted(set(family))}"
        mask = family == group
    if pattern:
        mask = mask & pd.Series(names).str.contains(pattern, regex=True).values
    assert mask.any(), f"{selector!r} selects no row"
    return mask


def stop_outside_barrier(model, where):
    """Barrier that ends in trouble hands the model to simplex: stop there."""
    if where != GRB.Callback.MESSAGE:
        return
    message = model.cbGet(GRB.Callback.MSG_STRING)
    if "Sub-optimal termination" in message or "Numerical trouble" in message \
            or (message.lstrip().startswith("Iteration") and "Objective" in message):
        print(f"STOP: {message.strip()} -> terminate (no simplex)", flush=True)
        model.terminate()


def barrier(model, arguments):
    model.Params.Threads = arguments.threads
    model.Params.Method = 2
    model.Params.Crossover = 0
    model.Params.BarHomogeneous = arguments.bar_homogeneous
    model.Params.ScaleFlag = arguments.scale_flag
    model.Params.FeasibilityTol = arguments.feas_tol
    model.Params.OptimalityTol = arguments.opt_tol
    model.Params.BarConvTol = arguments.bar_conv_tol
    model.Params.NumericFocus = arguments.numeric_focus
    model.Params.BarIterLimit = arguments.bar_iter_limit
    for setting in arguments.param:
        name, value = setting.split('=', 1)
        model.setParam(name, float(value) if any(mark in value for mark in '.e') else int(value))
    model.optimize(stop_outside_barrier)
    verdict = {GRB.OPTIMAL: 'SOLVED', GRB.INFEASIBLE: 'INFEASIBLE', GRB.INF_OR_UNBD: 'INFEASIBLE OR UNBOUNDED',
               GRB.UNBOUNDED: 'UNBOUNDED', GRB.ITERATION_LIMIT: 'NOT CONVERGED (iteration limit)',
               GRB.NUMERIC: 'NUMERICAL TROUBLE', GRB.INTERRUPTED: 'STOPPED OUTSIDE BARRIER',
               GRB.SUBOPTIMAL: 'SUB-OPTIMAL'}.get(model.Status, f'status {model.Status}')
    objective = f", objective {model.ObjVal:.8g}" if model.SolCount else ""
    print(f"RESULT: {verdict}; {model.BarIterCount} barrier iterations, {model.Runtime:.0f} s{objective}", flush=True)


def per_row_max(values, indptr):
    """The largest of ``values`` in every CSR row (0 for an empty row)."""
    result = np.zeros(len(indptr) - 1)
    filled = np.diff(indptr) > 0
    result[filled] = np.maximum.reduceat(values, indptr[:-1][filled])
    return result


def report_stats(model, names, family, out_dir):
    matrix = model.getA().tocsr()
    rhs = np.array(model.getAttr('RHS', model.getConstrs()))
    variables = model.getVars()
    lb, ub, obj = (np.array(model.getAttr(attr, variables)) for attr in ('LB', 'UB', 'Obj'))
    magnitude = np.abs(matrix.data)
    entry_family = np.repeat(family, np.diff(matrix.indptr))
    lines = []
    for group in pd.unique(family):
        on_entries, on_rows = entry_family == group, family == group
        group_rhs = rhs[on_rows]
        nonzero_rhs = np.abs(group_rhs[group_rhs != 0])
        lines.append(dict(family=group, rows=int(on_rows.sum()), entries=int(on_entries.sum()),
                          a_min=magnitude[on_entries].min() if on_entries.any() else np.nan,
                          a_max=magnitude[on_entries].max() if on_entries.any() else np.nan,
                          rhs_zero=int((group_rhs == 0).sum()), rhs_under_1e_6=int((nonzero_rhs < 1e-6).sum()),
                          rhs_abs_max=np.abs(group_rhs).max()))
    table = pd.DataFrame(lines)
    table.to_csv(out_dir / 'diagnose_families.csv', index=False)
    print(table.to_string(index=False), flush=True)
    print(f"\nmatrix |a| [{magnitude.min():.1e}, {magnitude.max():.1e}]; objective |c| "
          f"[{np.abs(obj[obj != 0]).min():.1e}, {np.abs(obj).max():.1e}]; rhs |b| "
          f"[{np.abs(rhs[rhs != 0]).min():.1e}, {np.abs(rhs).max():.1e}]", flush=True)
    print(f"columns: {len(lb):,}; fixed by their bounds (lb == ub) {int((lb == ub).sum()):,}; "
          f"0 < ub < 1e-5 on {int(((ub > 0) & (ub < 1e-5)).sum()):,}; lb > ub on {int((lb > ub).sum()):,}", flush=True)


def report_structure(model, names, family, out_dir):
    """No solve: rows the bounds decide on their own, and rows that repeat another row."""
    matrix = model.getA().tocsr()
    constraints = model.getConstrs()
    rhs = np.array(model.getAttr('RHS', constraints))
    sense = np.array(model.getAttr('Sense', constraints))
    variables = model.getVars()
    lb, ub = (np.array(model.getAttr(attr, variables)) for attr in ('LB', 'UB'))

    # ── every row's activity interval over the column bounds ──
    positive = matrix.copy()
    positive.data = np.maximum(positive.data, 0.0)
    negative = matrix - positive
    lb_finite, ub_finite = np.where(np.isinf(lb), 0.0, lb), np.where(np.isinf(ub), 0.0, ub)
    low = positive @ lb_finite + negative @ ub_finite
    high = positive @ ub_finite + negative @ lb_finite
    low[(positive != 0) @ np.isinf(lb).astype(float) + (negative != 0) @ np.isinf(ub).astype(float) > 0] = -np.inf
    high[(positive != 0) @ np.isinf(ub).astype(float) + (negative != 0) @ np.isinf(lb).astype(float) > 0] = np.inf
    size = np.maximum(np.maximum(np.abs(rhs), per_row_max(np.abs(matrix.data), matrix.indptr)), 1.0)
    margin = 1e-9 * size
    above, below = low > rhs + margin, high < rhs - margin                # the row cannot come down to / up to its rhs
    at_most, at_least, equal = sense == '<', sense == '>', sense == '='
    status = np.full(len(rhs), 'open', dtype=object)
    status[(at_most & (high <= rhs + margin)) | (at_least & (low >= rhs - margin))] = 'redundant'
    status[(at_most & (np.abs(low - rhs) <= margin)) | (at_least & (np.abs(high - rhs) <= margin))
           | (equal & ((np.abs(low - rhs) <= margin) | (np.abs(high - rhs) <= margin)))] = 'pinned'
    status[(at_most & above) | (at_least & below) | (equal & (above | below))] = 'impossible'
    counts = pd.crosstab(pd.Series(family, name='family'), pd.Series(status, name='on the box'))
    counts.to_csv(out_dir / 'diagnose_structure_box.csv')
    print("rows by what the column bounds alone say (pinned: the bounds leave the row no slack, so it has no interior):", flush=True)
    print(counts.to_string(), flush=True)

    # ── rows that are a multiple of another row: equal ratio of two random projections, same entry count and column span ──
    random = np.random.default_rng(0)
    first, second = matrix @ random.uniform(1.0, 2.0, matrix.shape[1]), matrix @ random.uniform(1.0, 2.0, matrix.shape[1])
    entries = np.diff(matrix.indptr)
    filled = np.flatnonzero((entries > 0) & (second != 0))
    first_column = matrix.indices[matrix.indptr[:-1][filled]]
    last_column = matrix.indices[matrix.indptr[1:][filled] - 1]
    ratio = first[filled] / second[filled]
    order = np.lexsort((ratio, last_column, first_column, entries[filled]))
    ordered = filled[order]
    same = ((entries[ordered][1:] == entries[ordered][:-1]) & (first_column[order][1:] == first_column[order][:-1])
            & (last_column[order][1:] == last_column[order][:-1])
            & (np.abs(ratio[order][1:] - ratio[order][:-1]) <= 1e-10 * np.abs(ratio[order][:-1])))
    left, right = ordered[:-1][same], ordered[1:][same]                    # row pairs: right = factor x left
    factor = first[right] / first[left]
    same_rhs = np.abs(rhs[right] - factor * rhs[left]) <= 1e-9 * np.maximum(np.abs(rhs[right]), 1.0)
    flipped = factor < 0
    sense_left = np.where(flipped, np.where(sense[left] == '<', '>', np.where(sense[left] == '>', '<', '=')), sense[left])
    kind = np.where(same_rhs & (sense_left == sense[right]), 'the same constraint twice',
                    np.where(same_rhs, 'the same boundary, other sense', 'parallel, different rhs'))
    pairs = pd.DataFrame({'family': family[left], 'family_other': family[right], 'kind': kind, 'name': names[left],
                          'name_other': names[right], 'factor': factor, 'rhs': rhs[left], 'rhs_other': rhs[right]})
    pairs.to_csv(out_dir / 'diagnose_structure_parallel.csv', index=False)
    print(f"\nrows that are a multiple of another row: {len(pairs):,} pairs", flush=True)
    if len(pairs):
        print(pairs.groupby(['family', 'family_other', 'kind']).size().to_string(), flush=True)
        with pd.option_context('display.width', 250, 'display.max_colwidth', 60):
            print(pairs.groupby(['family', 'family_other', 'kind']).head(3).to_string(index=False), flush=True)
    print(f"\n-> {out_dir / 'diagnose_structure_box.csv'}, {out_dir / 'diagnose_structure_parallel.csv'}", flush=True)


def report_rows(model, names, family, mask, out_dir):
    """Every selected row on a line: what it holds, and what the column bounds alone say about it."""
    matrix = model.getA().tocsr()
    constraints = model.getConstrs()
    rhs = np.array(model.getAttr('RHS', constraints))
    sense = np.array(model.getAttr('Sense', constraints))
    variables = model.getVars()
    lb, ub = (np.array(model.getAttr(attr, variables)) for attr in ('LB', 'UB'))
    picked = np.flatnonzero(mask)
    assert len(picked) <= 20_000, f"{len(picked):,} rows selected: narrow the selector (rows:<family>/<regex>)"
    lines = []
    for row in picked:
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        columns, coefficients = matrix.indices[start:end], matrix.data[start:end]
        positive = coefficients > 0
        with np.errstate(invalid='ignore'):                                   # 0 x inf where a column has no upper bound
            low = np.nansum(np.where(positive, coefficients * lb[columns], coefficients * ub[columns]))
            high = np.nansum(np.where(positive, coefficients * ub[columns], coefficients * lb[columns]))
        margin = 1e-6 * max(abs(rhs[row]), np.abs(coefficients).max() if len(coefficients) else 0.0, 1.0)
        can_exceed, can_fall_short = high > rhs[row] + margin, low < rhs[row] - margin
        if sense[row] == '<':
            on_box = 'redundant' if not can_exceed else 'impossible' if low > rhs[row] + margin else 'open'
        elif sense[row] == '>':
            on_box = 'redundant' if not can_fall_short else 'impossible' if high < rhs[row] - margin else 'open'
        else:
            on_box = 'impossible' if (low > rhs[row] + margin or high < rhs[row] - margin) else 'open'
        negative_over_positive = (np.median(-coefficients[~positive]) / np.median(coefficients[positive])
                                  if positive.any() and (~positive).any() else np.nan)
        lines.append(dict(name=names[row], family=family[row], sense=sense[row], rhs=rhs[row], entries=end - start,
                          positive=int(positive.sum()), negative=int((~positive).sum()),
                          a_min=np.abs(coefficients).min() if len(coefficients) else np.nan,
                          a_max=np.abs(coefficients).max() if len(coefficients) else np.nan,
                          negative_over_positive=negative_over_positive, activity_low=low, activity_high=high,
                          on_the_box=on_box))
    table = pd.DataFrame(lines)
    table.to_csv(out_dir / 'diagnose_rows.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 500, 'display.max_colwidth', 70):
        print(table.to_string(index=False), flush=True)
    print(f"\n{len(table):,} rows -> {out_dir / 'diagnose_rows.csv'}", flush=True)


def solve_relaxed(model, names, family, mask, arguments, out_dir):
    """A violation variable on every selected row; minimise their sum."""
    constraints = model.getConstrs()
    picked = [constraints[row] for row in np.flatnonzero(mask)]
    model.setObjective(gp.LinExpr(), GRB.MINIMIZE)
    model.feasRelax(relaxobjtype=0, minrelax=False, vars=None, lbpen=None, ubpen=None,
                    constrs=picked, rhspen=[1.0] * len(picked))
    print(f"violation variables on {len(picked):,} rows: {pd.Series(family[mask]).value_counts().to_dict()}", flush=True)
    barrier(model, arguments)
    if not model.SolCount:
        return
    value = {variable.VarName: variable.X for variable in model.getVars()
             if variable.VarName.startswith(('ArtP_', 'ArtN_'))}
    lines = [dict(name=constraint.ConstrName, family=family[row], sense=constraint.Sense, rhs=constraint.RHS,
                  violation=value.get(f'ArtP_{constraint.ConstrName}', 0.0) - value.get(f'ArtN_{constraint.ConstrName}', 0.0))
             for row, constraint in zip(np.flatnonzero(mask), picked)]
    table = pd.DataFrame(lines)
    table['violation_over_rhs'] = table['violation'] / table['rhs'].abs().where(table['rhs'] != 0)
    table.to_csv(out_dir / 'diagnose_relax.csv', index=False)
    broken = table[table['violation'].abs() > 10 * arguments.feas_tol].sort_values('violation', key=np.abs, ascending=False)
    print(f"total violation {model.ObjVal:.6g} (scaled units); rows violated by more than {10 * arguments.feas_tol:g}: "
          f"{len(broken):,}", flush=True)
    if len(broken):
        with pd.option_context('display.width', 250, 'display.max_colwidth', 70):
            print(broken.head(50).to_string(index=False), flush=True)
        print("=> the selected rows cannot hold together: INFEASIBLE, by the amounts above", flush=True)
    else:
        print("=> FEASIBLE: every row holds; the trouble is in reaching the optimum, not in the constraints", flush=True)


def main():
    arguments = read_arguments()
    out_dir = arguments.out or arguments.model.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model = gp.read(str(arguments.model))
    names = np.array(model.getAttr('ConstrName', model.getConstrs()), dtype=object)
    family = family_of(names)
    print(f"read {arguments.model.name}: {model.NumConstrs:,} rows, {model.NumVars:,} columns, {model.NumNZs:,} entries "
          f"in {time.time() - start:.0f} s; families {pd.Series(family).value_counts().to_dict()}", flush=True)
    if (names[: min(len(names), 100)] == [f'R{row}' for row in range(min(len(names), 100))]).all():
        print("WARNING: the constraints carry Gurobi's default names: the families cannot be told apart "
              "(a name with a space makes the MPS writer drop every name)", flush=True)

    mode, _, selector = arguments.mode.partition(':')
    if mode == 'stats':
        report_stats(model, names, family, out_dir)
    elif mode == 'structure':
        report_structure(model, names, family, out_dir)
    elif mode == 'rows':
        report_rows(model, names, family, select(selector, names, family), out_dir)
    elif mode == 'reproduce':
        barrier(model, arguments)
    elif mode == 'feas':
        model.setObjective(gp.LinExpr(), GRB.MINIMIZE)
        barrier(model, arguments)
    elif mode == 'relax':
        solve_relaxed(model, names, family, select(selector or 'wide', names, family), arguments, out_dir)
    elif mode == 'drop':
        mask = select(selector, names, family)
        constraints = model.getConstrs()
        model.remove([constraints[row] for row in np.flatnonzero(mask)])
        model.update()
        print(f"removed {int(mask.sum()):,} rows: {pd.Series(family[mask]).value_counts().to_dict()}", flush=True)
        if mask.sum() <= 40:
            print("  " + "\n  ".join(names[mask]), flush=True)
        barrier(model, arguments)
    else:
        raise SystemExit(f"unknown mode {arguments.mode!r}; see --help")
    print(f"DONE {arguments.mode} in {time.time() - start:.0f} s", flush=True)


if __name__ == '__main__':
    main()
```
