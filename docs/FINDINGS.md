# LUTO2 Findings Log

A running record of discoveries, investigations, and conclusions from model exploration.
Entries are in **descending date order** (newest first).

---

## 20260810 (later) — the blind spot WAS the flow system: widening the diagnosis scope unstalled all five runs

### TL;DR

The five capped runs that had defeated every attempt since 2026-08-06 (`R2_SNES_T1525_cap10/15`,
`R3_ECNES_T1525_cap10/15` at 2045; `R4_GBF2_T3050_cap25` at 2050) were all unstalled the same day by
**one settings change per run**: adding `flow_in`/`flow_out` to `INFEASIBILITY_DIAGNOSIS_GROUPS`.
The pre-solve `feasibility_spectrum` then found and dropped the species/community rows whose targets
are unreachable **only through the transition-flow system** — rows the old bio+cap-scoped probe
certified as feasible, because without the flow rows they are. With those rows gone, every
previously-stalling year solved by ordinary barrier in 448–812 s where prior attempts burned 5–24 h
in Gurobi's internal simplex fallback without terminating.

The decisive experiment (`flow_widen.py`, on `R2_SNES_T1525_cap15`'s saved 2045 MPS):

| step | scope | verdict | time |
|---|---|---|--:|
| ground truth `is_feasible` | bio + cap + **flow** | **INFEASIBLE** (clean, not NUMERIC) | 74 s |
| production `feasibility_spectrum` | same | dropped 1 SNES row → OPTIMAL | 163 s |

**Decidability was a scope problem, not a model problem.** The full "everything" probe returns
NUMERIC (status 12) — that is what motivated the unsound reduction retracted in the entry below.
But bio + cap + flow *without* ghg/water/demand/adopt is cleanly decidable in ~75 s on the same
4.6M-row model (the sound `reduce_forced_zero_rows` runs on the probe copy by default). The
conflict lives entirely inside the widened scope, so restricted-INFEASIBLE proves full-model
infeasibility — the sound direction of the relaxation inference.

### What the widened scope found that the old scope could not

Rolled into all five runs' own `luto/settings.py` (kill → patch → resubmit from checkpoint). Rows
newly named at 2045, on top of what the old scope had already found:

| run | old scope found | widened scope added | diagnosis old → new |
|---|---|---|---|
| `R2_cap10` | Swainsona recta | + Calidris ferruginea, + Macquaria australasica | 1m21s → 4m13s |
| `R2_cap15` | **nothing** | E. alligatrix, Macquaria australasica | 1m05s → 3m09s |
| `R3_cap10` | 6 rows | + GB Buloke Woodlands (ECNES) | 21m09s → 4m57s |
| `R3_cap15` | NE Buloke (ECNES) | + E. alligatrix, + Macquaria australasica | 1m29s → 3m33s |

`R2_cap15` is the clean demonstration: its old-scope spectrum found *zero* droppable rows (bio+cap
alone is satisfiable), then the production solve stalled anyway — four times. The widened spectrum
named two species in 3 minutes and the year solved.

**Macquaria australasica (Macquarie Perch) and Eucalyptus alligatrix subsp. limaensis recur across
runs** as flow-dependent conflicts — targets satisfiable in an unconstrained-land sense but not
given how land is permitted to MOVE (T_MAT reachability + source availability), which is exactly
why every flow-blind probe reported feasible. The cost of widening is minutes per year; a
per-IIS-round cost of ~25–75 s replaced rounds that ranged to 27 min at the old scope (the old
scope's occasional slow rounds were not cheaper — just blinder).

### R4 and GBF2: the one row that makes computeIIS glacial

`R4_GBF2_T3050_cap25` (the only run where `GBF2_TARGET='high'` is actually on) was the exception:
its diagnosis was pathologically slow at ANY scope containing `bio_gbf2` — 111 min for 5 rounds at
the old scope (individual rounds to 27 min), and its first widened attempt sat >28 min without
completing the spectrum. GBF2's dual reached −$145.6 B in earlier runs, ~500× the largest SNES dual
— one severely-scaled national row is enough to cripple the deletion-filter LPs inside computeIIS.
Removing `bio_gbf2` from the diagnosis scope (NOT from the model — GBF2 stays a hard constraint in
the real solve; the probe just can't see it) let the spectrum finish promptly and 2050 solved:
**146 barrier iterations, 538 s, optimal**, after three failed attempts on prior days. Documented
risk, accepted: a genuinely GBF2-involving conflict would now be invisible to the pre-solve fixer
and would surface as a real-solve failure — in minutes, not the hours the alternative cost.

### Barrier endgame blow-up that RECOVERED (distinct from the stall class)

During `R2_cap10`'s post-drop 2050 solve, one iteration jumped from near-convergence
(obj 1.18e4, compl 1e-7, iter 188) to obj −3.1e11 / pinf 5.6e10 (iter 189), then re-converged over
~190 iterations back to the same 1.18e4. Mechanism: the barrier's Newton system `A·D·Aᵀ` becomes
ill-conditioned *by construction* as complementarity → 0 (D = X/S diverges both ways), and this
model's conditioning (matrix 1e-4→7e3, Presolve=0 leaving 1.7M degenerate flow equalities in the
factorisation, knife-edge cap) makes one garbage Newton step likely at the endgame. Under
`BarHomogeneous=1` a bad step loses *centrality but not interiority*, so it recovers by ordinary
re-centering — unlike the "Numerical trouble → internal simplex fallback" class, which abandons
barrier entirely and never returns. Diagnostic: monotone residual shrink after the spike = leave it
alone; frozen objective across thousands of simplex iterations = the stall class.

### Schema drift corrupted every dropped_constraints CSV written across code vintages

`record_dropped()` appends with `header=not os.path.exists(path)` — the header is written by
whichever code version CREATES the file, and later appends from a different version write raw
values in their own column order with no header check. Three vintages exist (9-col, 11-col, and the
current 12-col which also moved `year,stage` to the front), so any file touched by more than one
vintage has rows that silently misalign against its own header. Fixed forensically: each row's
field COUNT uniquely identifies its schema, so `fix_dropped_csv.py` (task dir) reparses by count,
maps to the canonical 12-col schema, de-duplicates on `(year, constraint, action)` (kill/resubmit
cycles append the same drop repeatedly), and rewrites. Applied to all 9 live files and, via
extract-repair-rezip, to the 7 affected files sealed inside completed runs' `Run_Archive.zip`
(originals backed up under `_dropped_csv_backup_20260810/`; all archives pass `unzip -t`).
**The code path itself is NOT yet fixed** — a future schema change will corrupt again unless
`record_dropped` starts validating the on-disk header before appending.

### Where this leaves the science

The T1525-cap cells now *complete with a measured sacrifice list* instead of hanging: the capped
land budget at the halved target is deliverable **except for** the named flow-dependent rows, and
`dropped_constraints_<yr>.csv` records exactly which targets were given up (action=DROPPED), which
were met by a hair (KNIFE_EDGE — the GB cap itself, headroom < 1e-4, in every one of these runs),
and which comfortably. Separately, `lockin_test2.py` (`freed_cells`) gave a definite verdict on the
E. alligatrix 2045 target: with the species' OWN 18 cells' lock-in freed the model is decidable and
the target is still unreachable by −0.86 %, while freeing ALL lock-in clears it by +23.3 % — so the
blocking irreversibility sits in cells *beyond* the species' own habitat. Status at time of
writing: 4 of 5 runs through all solves and writing reports; `R2_cap10` re-converged at 2050 after
the endgame spike. (`lk2_base_nf3`, which would have double-checked the base-bounds verdict at
NumericFocus=3, was killed to free the queue — acceptable loss given `freed_cells` already
answered the local-vs-global question.)

---

## 20260810 — a "reduction" that silently built a different model, and the IIS that confidently described it

### TL;DR

A diagnostic script claimed to make the numerically-undecidable 2045 model decidable by removing
1,393,366 zero-RHS node-balance rows and pinning the 4,369,481 variables they touch. The
justification — `Σaᵢxᵢ = 0` with all `aᵢ > 0` and `xᵢ ≥ 0` admits only all-zeros — **is valid, but the
script never tested it.** It selected rows by NAME PREFIX and pinned regardless of sign. **96 % of
those rows are mixed-sign**: they read `X = ΣF` (inflow-minus-outflow), for which all-zeros is one
solution among many. **4,234,347 variables were forced to zero that the model never forced.**

The result was not a reduction of the model but a different, massively over-constrained one — which
duly returned INFEASIBLE in 85 s, and an IIS of 34 rows naming one species, five cells and a specific
lever chain. All of it was an artefact. Three further analyses were built on that IIS before anyone
checked the script against its own stated argument.

### The bug

```python
# reduced_iis.py -- selects by NAME, pins regardless of coefficient sign
idx = [i for i in range(len(cons))
       if rhs[i] == 0.0 and sense[i] == "=" and names[i].startswith(("bal_a_", "bal_n_"))]
for c in doomed:
    for k in range(m.getRow(c).size()):
        fix.add(m.getRow(c).getVar(k))     # <-- no sign check
for v in fix:
    v.UB = 0.0
```

```
bal_a_0_0_112512   (+1 / -3)          # X = sum of inflows; all-zeros is NOT the only solution
      1.0  X_ag_dry_0_112512
     -1.0  F_a2a_0_14[...]  -1.0  F_a2a_0_22[...]  -1.0  F_a2a_0_23[...]
```

| | rows removed | vars pinned |
|---|--:|--:|
| `reduced_iis.py` (name prefix, no sign check) | 1,393,366 | 4,369,481 |
| `tools.reduce_forced_zero_rows` (**checks signs** — correct) | **59,902** | **135,134** |

A 20× discrepancy between two things that were supposed to be the same reduction — visible for free
by running both, which is what eventually exposed it.

### Why it survived so long

1. **A false self-check.** The script printed `no var had LB>0 -- reduction is exact`. That tests
   nothing relevant: `lb = 0` says nothing about whether a mixed-sign row *forces* a variable to zero.
   A check that cannot fail for the wrong reason reads as reassurance.
2. **The artefact was specific, and specificity reads as truth.** The IIS named one species, five
   cells, thirteen HIR links. Vague output invites doubt; a precise, mechanistic story invites
   elaboration. The over-constraint *manufactured* that precision.
3. **The prose and the code drifted apart.** The exactness argument was written for a sign-checking
   reduction and correctly describes `tools.reduce_forced_zero_rows`. It was then applied to a script
   that did something else. **The argument was never re-derived against the code that ran.**

### Rules taken from this

1. **A reduction must be validated on the model, not in the docstring.** Cheapest sufficient test:
   solve both and compare objectives and variable values, or assert the claimed property row-by-row
   (here: `all(coeff > 0)`) *in the code that does the pinning*.
2. **Never let two implementations of "the same" transform diverge unchecked.** Run both, diff the
   counts. A 20× gap is not subtle.
3. **An IIS is a statement about the model handed to it — nothing more.** Its specificity carries no
   evidence about whether that model is the one you meant to build.
4. **Prefer MAXIMISATION over feasibility for reachability questions.** "Is this model feasible?"
   returned NUMERIC on the full 4.6M-row model, which is what motivated the reduction at all.
   "Remove constraint C, maximise its LHS, compare to its RHS" solved the **full unreduced model in
   107 s / 26 barrier iterations** and answers a strictly more informative question — by how much does
   it miss. `luto/tests/find_infeasible_ecnes.py` already uses this shape; reach for it first.
   *(Resolved same day, entry above: the NUMERIC verdict was a SCOPE problem — bio+cap+flow without
   ghg/water/demand/adopt answers INFEASIBLE cleanly in 74 s, so no reduction was ever needed.)*

### Not affected

`tools.reduce_forced_zero_rows` has the sign check and is correct (`REDUCE_FORCED_ZERO_ROWS` also
defaults `False`). The 20260806 NE Buloke entry below is unaffected — it was traced arithmetically
from `data_2020.lz4` cell 177804 with no IIS and no reduction.

---

## 20260806 — a biodiversity target that irreversibility makes unreachable: why Gurobi stalls instead of saying INFEASIBLE, and the build-time screen that catches it

### TL;DR

A GBF4 ECNES row (**NE Buloke Woodlands**, ONE cell) became physically unsatisfiable at 2025 and took
the whole year down. Gurobi never said so: attempt 1 (barrier) returned `INF_OR_UNBD`, attempt 2 (dual
simplex) ran **2.28 M iterations over 5 h** with objective → −4.7e32 and primal infeasibility → 8.9e29
without terminating. The cause is **path dependence through irreversible land use**, not numerics:
2020 met a low target with the cheapest sufficient lever (EP, contribution 0.70) on 60 % of the cell;
EP is irreversible, so from 2025 the rising target could only be met by a *higher*-contribution lever,
which the lock-in forbids. Added `LutoSolver._max_reachable_contr_r()` — a per-cell contribution
ceiling that respects lock-in and every lever's real bound — plus `_bio_row_admitted()`, which proves
such rows impossible **before the solve**, records them to
`out_<yr>/unreachable_bio_constraints_<yr>.csv` and excludes them. The year then solves on the first
attempt.

### The mechanism (path dependence, not ill-conditioning)

`NON_AG_LAND_USES_REVERSIBLE` marks **8 of 9** non-ag land uses irreversible (only `Destocked` is
reversible), and `get_non_ag_lb_matrices` pins the next year's lower bound to this year's allocation.
Traced on the real cell (`data_2020.lz4`, cell **177804**, REAL_AREA 2 498 ha):

| | value |
|---|---|
| 2020 solution | **Environmental Plantings 0.603164** — irreversible, locked for every later year |
| free share | 0.396836 |
| locked share's contribution | EP = **0.70** |
| best contribution available on the free share | riparian = 1.00, but `RP_PROPORTION` = **0** here |

| year | contribution required | max reachable | verdict |
|---|---:|---:|---|
| 2020 | 0.646 | 1.000 (nothing locked yet) | ✅ EP alone clears it — and EP is cheapest, so the optimiser picks it |
| 2025 | **0.798** | **0.736** | ❌ EP 0.70 ✗ · destocking 0.75 ✗ · riparian unavailable |
| 2030 | ≈ **0.950** | 0.736 | ❌ unreachable under any allocation |

**The 2020 solve set the trap and the later years sprang it.** This generalises: *any* constraint whose
required contribution rises across years can become permanently infeasible if an early year satisfies
it with a cheaper, lower-contribution irreversible land use. Small-area, few-cell rows are most exposed
— they have no substitution.

### Why Gurobi stalls instead of proving infeasibility

Worth understanding, because the symptom looks like a solver defect and is not:

1. **`INF_OR_UNBD` (Status 4) is not a diagnosis** — it means Gurobi could not *distinguish* infeasible
   from unbounded, because presolve's dual reductions destroy the information needed to tell them
   apart. The documented remedy is **`DualReductions=0`**, which forces a definite INFEASIBLE or
   UNBOUNDED. Our `RETRY_PARAMS` ladder never sets it.
2. **The infeasibility is marginal, which is the hardest case.** `targ2attain = 1.002` at 2025 — the row
   misses by **0.2 %**. A grossly infeasible model is easy to certify; a *nearly* feasible one lets the
   barrier keep converging toward a point that does not exist, which surfaces as "Numerical trouble
   encountered" rather than a Farkas certificate.
3. **Plain barrier is not an infeasibility detector.** Only the homogeneous self-dual variant
   (**`BarHomogeneous=1`**) is designed to certify infeasibility. Attempt 2 in `RETRY_PARAMS` explicitly
   sets `BarHomogeneous=0`, so that path was never tried.
4. **The diverging dual simplex *was* the certificate, unterminated.** Objective −4.7e32 with primal
   infeasibility 8.9e29 is a dual ray growing without bound — which *is* the proof of primal
   infeasibility. Gurobi could not close its termination tolerances on it because of conditioning:
   `Matrix range [1e-04, 3e+03]` against `RHS range` up to **3e+05**, and `Bounds range` down to
   **7e-09** (millions of near-degenerate columns from θ = 0.01 slivers).
5. **The implication is too deep for presolve.** "This row's max LHS < its RHS" is not a single-row bound
   violation — deriving it needs the cell-allocation constraint, each lever's `dvar_ub_*`, T_MAT
   reachability and the irreversible lock-in floor, chained together. Bound propagation does not reach
   that far, and presolve is kept conservative for barrier anyway (it has caused false infeasibility).

The screen wins not by being cleverer numerically but by using **domain structure** — it computes the
bound analytically instead of discovering it through linear algebra over 32.9 M nonzeros.

### The screen (`luto/solvers/solver.py`)

`_max_reachable_contr_r()` — per-cell contribution ceiling, cached once per year, pure numpy
(milliseconds; a `feasRelax` diagnosis would cost a full 4.65 M-row re-solve):

```
locked   = Σ irreversible non-ag lb·contr + Σ irreversible AM lb·(contr_j + increment)
free     = max(1 − locked_share, 0)
ceiling  = locked + greedy fill of `free` over ALL levers, best contribution first,
           each capped by its own headroom (dvar_ub_nonag/dvar_ub_ag − the matching lb)
```

Reusing the same bound matrices the solver builds its variables from means no-go zones, T_MAT
reachability, the Riparian `RP_PROPORTION` and Destocked eligibility caps, and lock-in are **inherited,
not re-derived**. Every contribution is ≤ 1, so the ceiling is a true upper bound and
`reachable < target` is a *sound* proof — it can only drop rows that are genuinely impossible.

Two mistakes worth recording, because each cost a 24 h run to find:

- **Uncapped levers.** Taking the single best contribution and letting it cover the whole free share
  made the ceiling 0.819 vs 0.736, so nothing was ever dropped. `Unallocated - natural land` is worth
  **1.0** and `T_MAT` allows only `natural → natural` (1 of 37 sources), so a cell holding a 0.12 sliver
  of it was credited with 1.0 across its entire free share.
- **Max instead of sum.** A cell can be *split* across levers, so aggregating with `max` understates the
  ceiling — and understating drops rows that are actually satisfiable, the expensive direction. Greedy
  best-first is exactly optimal here (every unit of area costs the same).

Also note **ag-management values are INCREMENTS**, not absolute contributions (`Savanna Burning`
= `1 − BIO_CONTRIBUTION_LDS`; `Biochar` = `Biodiversity_impact − 1`, which can be **negative**), and
`X_ag_man ≤ X_ag[j]` means an AM adds no area — so it is folded into its land use's contribution rather
than competing as a separate lever.

### Result

`DROP_UNREACHABLE_BIO_CONSTRAINTS` (default `True`) gates the exclusion; each dropped row is printed in
the per-family build table and written to `out_<yr>/unreachable_bio_constraints_<yr>.csv` with
`region` / `species` / `presence` / `targ2attain` as separate columns, **before** the solve so the record
survives a year that still fails.

| run | code | 2025 |
|---|---|---|
| `R3_ECNES_T3050` | original | `INF_OR_UNBD` → dual simplex diverged → killed at 5 h |
| `P4_dropfix` | screen, uncapped levers | flagged RAZOR-THIN, kept → same failure |
| `P5_dropfix_local` | screen, per-cell caps | **flagged UNREACHABLE, dropped, solved first attempt**; 2020–2045 all optimal |

**θ is not the fix.** Probe P2 re-ran the failing configuration at θ = 0.0 (nothing folds at all): the
symptom changed from `Infeasible model` to `Numerical trouble encountered` and simplex stalled
identically. θ = 0.01 *did* rescue a **different** case — a 15 % non-ag cap at 30/50 that was infeasible
at θ = 1.0 — but it does nothing for an unreachable row. Investigation artefacts:
`jinzhu_inspect_code/Explore_CMA_nonag_cap/doc/progress.md`.

---

## 20260722 — write.py 258 GB peak: NOT per-worker data copies — it's uncapped Linux workers + resident `data`

### TL;DR

`write_outputs` on a full **annual (41-yr)** NCI run peaks at **~258 GB**, vs ~83 GB for a 5-yearly
run and ~100 GB solving. AB profiling **disproved** the intuitive "each loky worker pickles its own
~12 GB copy of `data`" hypothesis: with the big arrays **memmap-shared by joblib automatically**, a
worker's *private* footprint (USS) is only **~0.4 GB** (0.14 GB interpreter+libs + 0.26 GB unmapped
`data` bits). So workers do **not** duplicate the 58.7 GB `data`. The real model is
**`peak = resident parent data (~59 GB @16 yr, scales with years/RESFACTOR → ~150 GB @41 yr) +
Σ_concurrent_workers (task_working_mb + ~0.4 GB) + accumulated MAX_CELL_MAGNITUDE lists`**. A's 258 GB
≈ ~150 GB parent + ~50–70 GB concurrent worker working-sets + ~40 GB magnitude. It exploded on
**NCI/Linux specifically** because `max_workers = min(cpu_count, WRITE_MAX_WORKERS_WINDOWS) if
os.name=='nt' else os.cpu_count()` — the cap is **Windows-only**, so Linux ran uncapped at
`os.cpu_count()` (~208 on a 104c/208t node). The peak lands at the **end of `write_data`** (the
crosstab/GBF8 *light* tier, which gets the most workers), **not** in `create_report`/map-layers
(those peak ~140 GB).

### The irony: the lightest task drives the peak

`get_n_jobs(peak_mb) = min(max_workers, WRITE_REPORT_MAX_MEM_MB // peak_mb)` budgets only a task's
**working arrays**, ignoring the fixed per-worker process tax. So the *lightest* function
(`write_crosstab`, ~36 MB) earns `n_jobs = max_workers` (~208 on Linux) while heavy tasks
(`write_transition`, ~18 GB) get ~1–3. The light tier therefore spawns the most **worker processes**,
and the cost is `n_workers × (interpreter + libs + unmapped-data + working)` — cheapness buys maximal
parallelism, and parallelism, not the task, is what costs the memory.

### AB evidence (tier-1 mosaic dispatch, A's checkpoint, worker budget 64 GB)

| run | workers | peak GB |
|-----|---------|---------|
| OLD (joblib auto-memmap)        | 6  | 82.8 |
| NEW (explicit dump+memmap)      | 6  | 79.0 (≈ OLD → memmap fix redundant) |
| SCALE15 (worker-count scaling)  | 15 | 104.2 (+~3 GB/worker, linear in *workers*, not data) |

`data` uncompressed = 58.7 GB; if copied per worker, 15 workers would be ~940 GB — observed 104 GB.
joblib's loky backend already memmaps the large ndarrays; the memmap file lives in OS page cache
(not process wset), so neither OLD nor NEW duplicates it.

### The concrete bug: the cheapest task grabs the most workers

Old `get_n_jobs(peak_mb) = min(max_workers, WRITE_REPORT_MAX_MEM_MB // peak_mb)` charges a worker only
its task working set and treats the fixed per-worker process cost as zero. `write_dvar_area` (~10 GB)
→ `64 GB // 10 GB ≈ 6` workers (fine). But `write_crosstab` is a few-MB pandas op → `64 GB // ~10 MB
≈ 6,400` workers. Capped at `os.cpu_count()`, that's ~100–200 workers on an NCI node — for a task that
needs ~3 — each paying its ~0.5 GB interpreter/libs + ~0.3 GB unmapped-`data` ≈ **0.8 GB private**. So
the light tier alone is ~100 × 0.8 = ~80 GB on top of the resident `data`.

### Fixes (applied to `luto/tools/write.py`)

- **`get_n_jobs` budgets the real model:** `mem_mb_data_obj = live parent RSS`;
  `mem_mb_budget = max(4096, WRITE_REPORT_MAX_MEM_MB - mem_mb_data_obj)`;
  `n_jobs = min(max_workers, mem_mb_budget // (peak_mb + mem_mb_worker[=500]))`. Now crosstab costs
  `10 + 500 MB` per worker, so it gets a sane count. `WRITE_REPORT_MAX_MEM_MB` is now the **TOTAL** RAM
  budget — set it to the real allocation. (`import psutil` added.)
- **`max_workers` capped on ALL platforms:** `min(os.cpu_count(), 32)` — dropped the `if os.name=='nt'`
  exemption that left Linux uncapped. A light I/O task gains nothing past ~32.
- **`write_data` refactored:** group tiers by `peak_mb` and recompute `n_jobs` live per tier (so late
  tiers see the parent's grown RSS); single inlined dispatch path; removed the dead `max_workers` term.
- **Validated:** full write (4 yr, 128 GB budget) peaked **123 GB ≤ 128 GB** — bounded to budget.
- **Trade-off:** tighter budget ⇒ fewer workers ⇒ slower heavy tier (memory↔speed). Set the budget to
  real RAM; and freeing the parent `data` after dispatch (workers share it via memmap) buys back both.
- **Still TODO (not in this commit):** free the parent `data`; stream `MAX_CELL_MAGNITUDE` (running
  quantile vs full per-cell lists) — trims the ~40 GB parent accumulation. `get_mag` now takes
  `min_p`/`max_p` default args.
- **Not a fix:** an explicit dump+memmap of `data` — joblib already memmap-shares it; changed nothing
  (79 ≈ 83 GB).

### Investigation artefacts

`jinzhu_inspect_code/Fix_write_huge_mem/` — `doc/PLAN.md`+`PROGRESS.md`, AB scripts/results
(`ab_OLD/NEW/SCALE15.json`), `80_measure_per_worker.py` (USS), `70_locate_peak.py` (peak-phase split),
`10_audit_inplace_mutations.py` (confirmed no writer mutates `data` in place → memmap-safe).

---

## 20260721 — variable accounting design (fixed-composition bundle) + a sub-floor coefficient leak & the post-build floor fix

### TL;DR

The two-stream accounting model (`X_ag` folded flow vars + `X_acct` LinExpr re-expression, entry
`20260710`) is best understood as a **fixed-composition bundle**: a folded cell is one scalar (the
dominant's var `X_ag[d]`) and every true land use is a **constant ratio** of it
(`X_acct[lu] = frac_lu · X_ag[d]`). This is valid only because the objective and all constraints are
**linear** — a fixed-ratio bundle collapses to one variable carrying a blended coefficient, and
`_setup_ag_accounting_vars` is the *inverse* re-expansion so each LU is scored with its **own raw**
coefficient. Composition therefore lives **entirely on the dvar side**; the input coefficient
matrices (`get_ag_w_mrj`, `get_ag_g_mrj`, `ag_obj_mrj`, …) stay **pure per-LU** (no fractions) — by
design, so one composition is reused across every stream. **New finding:** that same re-expression
silently **leaks sub-`SOLVER_COEFF_MIN` matrix coefficients** past `_qsum`, wrecking Gurobi's matrix
range on some scenarios. Root: `_qsum` floors the *raw coefficient* (`ag_w ≥ 1e-4`), then the
**sliver fraction** (`≈ 1/RESFACTOR²`) multiplies it *downstream* into a sub-floor **product** on the
dominant's var. Fix: a **post-`formulate()` coefficient floor** that sweeps the assembled matrix (and
objective) and drops `|coeff| < SOLVER_COEFF_MIN` — the one place that sees the finished product.

### The design — fixed-composition bundle (why linearity makes it work)

A resfactored cell holds a composition, e.g. `0.7 Beef + 0.3 Apple`. θ-folding merges the minor
Apple fraction into the dominant Beef, so a **single** Gurobi var `X_Beef` (created at
`solver.py:216`, `dominant_frac = 0.7 + 0.3 = 1.0`) represents "how much of this cell remains in its
original composition". Each LU is a constant ratio of that scalar:

    dvar_account[Beef]  = (0.7 / dominant_frac) · X_Beef
    dvar_account[Apple] = (0.3 / dominant_frac) · X_Beef

Reduce `X_Beef` by 1/3 → both shrink by 1/3 (Beef 0.7→0.467, Apple 0.3→0.2); the **ratio is
preserved**, only the scale changes. Legitimate purely because everything is linear:
`Σ_lu coeff_lu · frac_lu · X_Beef = (Σ_lu coeff_lu·frac_lu) · X_Beef` — the bundle ⇄ one var with a
blended coefficient. `_setup_ag_accounting_vars` is the inverse of that collapse (docstring updated
with this model). **Tradeoff:** the bundle moves as a unit — a folded sliver cannot move
independently of its dominant and inherits the dominant's transition/exclusion (`T_MAT`) rules; exact
behaviour for a behaviourally-distinct sliver would require its own dvar (exclusion-aware folding).

### Composition is on the dvar side, NOT the coefficient (don't double-count)

`water = Σ_lu ag_w[lu] · X_acct[lu]` with `ag_w[lu]` **pure** (Apple's rate, Beef's rate) and
`X_acct[lu]` **composed** (`frac_lu·X_Beef`). Equivalent to the retired *blended coefficient*
(`Σ frac·ag_w` on one var) — composition must live on **exactly one** side or the fractions apply
twice. The dvar side was chosen so each stream (water/GHG/profit/biodiversity, all with different
per-LU coefficients) reuses **one** composition against its **standard raw** coefficient matrix.
Hence `get_ag_w_mrj` et al. correctly carry no dvar fractions.

### The sub-floor coefficient leak (numerical)

`_qsum` (`solver.py:69`) filters `|coeff| < SOLVER_COEFF_MIN` (1e-4) as each term is built — but it
floors the **raw coefficient**, e.g. `_qsum(ag_w[0,ind,j], X_acct_dry_jr[j,ind])` (`solver.py:963`).
For folded cells `X_acct` is a `LinExpr` with sliver weights `slivers/dominant_frac ≈ 1/25`, so the
kept-and-above-floor `ag_w` is **distributed into a sub-floor product** on the dominant var
**after** `_qsum` ran. The floor gates the multiplier; the smallness is born in the product. `X_acct`
entries are therefore a mix of `Var` (untouched), `LinExpr` (sliver re-expression), or `int 0`
(fully folded). Land uses with near-zero coefficients concentrate the leak (their raw coeff is
smallest, so the product dives furthest below the floor).

### Diagnosis (NECMA build; `jinzhu_inspect_code/Explore_NECMA_species_targets`)

A tiered-target run failed the barrier with `Matrix range [1e-08, 2e+03]` (ratio ~2e11).
Instrumentation (`script_9/10/11`, jobs on Gadi): the 40 smallest coefficients were all
`water_yield_limit_*` on `X_ag_dry_23` = **"Unallocated - natural land"** (net water yield ≈ 0). For
Murray-Darling / LU 23 (38,386 cells): `X_acct` = 6,665 `Var` / 19,254 sliver `LinExpr` / 12,467
`int`; the 65 *direct* sub-floor `ag_w` cells are correctly dropped by `_qsum`; **2,106 sub-floor
PRODUCTS** come from the re-expression, e.g. `ag_w = 1.882e-3 (≥floor) × frac = -1/24 = -7.84e-5
(<floor)`. Not the biodiversity/SNES targets (those rescale **per constraint**,
`rescale_lhs_rhs_region_species`, and stay bounded). `WATER_REGION_DEF='Drainage Division'` only
moved the min `1e-8 → 4e-8` — region granularity is not the lever.

### Fix — post-`formulate()` coefficient floor (the robust, clean plan)

A `_qsum`-side floor can't help (the product doesn't exist yet). Sweep the **assembled** matrix once
and `chgCoeff(...,0)` for `|coeff| < SOLVER_COEFF_MIN`, plus a companion sweep of the **objective**
vector (`var.Obj`) since the same leak reaches `_qsum(ag_obj × X_acct)` (`solver.py:584`) and
`getA()` is LHS-only. Gate with `FLOOR_ASSEMBLED_MATRIX`. Collapses the range `1e-8 → 1e-4` (ratio
~1e6). **Keep `_qsum`** — this is a backstop, not a replacement (removing it would build millions of
negligible terms just to floor them out). Correctness-safe: dropping `|coeff| < 1e-4` is the same
negligibility contract `_qsum` already enforces (a ~1e-5 ML/cell water term against a ~1e7 ML
regional limit is nil). The matrix floor is the actual barrier fix; the objective floor is defensive
(objective range is a secondary concern, not the diagnosed failure). Sketch + verification plan:
`jinzhu_inspect_code/Explore_NECMA_species_targets/doc/patch_water_coeff_floor.md`.

Cheaper *complement* (not a guarantee): prune slivers with `slivers/dominant_frac < ε` in
`_setup_ag_accounting_vars` (skip **both** transfer lines to conserve mass) — kills the extreme tail
at source but can't target the product, so it shrinks the range without guaranteeing it.

---

## 20260716 — "Ag man lb clamped" gaps at 1e-1 are θ-fold jumps, not FeasibilityTol slack

### TL;DR

The residual `Ag man lb clamped [<AM>]: ... max gap=4.xe-01` console lines (e.g. Precision
Agriculture, AgTech EI, Biochar in the RF5/RF3 default runs) are **expected under θ=1.0**, not a bug.
The clamp's original job was to absorb FeasibilityTol slack (am can exceed its host ag by ≤1e-6 because
`am ≤ ag` is a linear constraint, not a variable bound), which would give gaps ≤1e-6. The observed
**1e-1-level** gaps come from a different mechanism: the host ag dvar being clamped against is the
**θ-folded** base (`get_folded_base_ag_dvar`), and with `EXACT_REACHABILITY_MIN_FRACTION = 1.0` the fold
keeps only each cell's dominant land-use. An ag-man fraction sitting on a folded-away minority host has
its solver-world host dvar go to 0, so its lower bound is clamped from its true fraction (~0.44) down to
0 — the whole adoption fraction, reported as the gap.

### Mechanics

- `get_lower_bound_agricultural_management_matrices` (`transitions.py:665`):
  `tools.clamp_dvar_bound(am_dvar, 0.0, ag_dvar, f'Ag man lb clamped [{am}]')`, where
  `ag_dvar = get_folded_base_ag_dvar(data, base_year)` (`:658`). The docstring at `:642-646` already
  states it: *"an am fraction whose host sliver was folded away is clamped to 0 — its land now lives
  under the dominant source."*
- Report threshold in `clamp_dvar_bound` (`tools/__init__.py:81`) is `10^-ROUND_DECIMALS = 1e-6`, so
  pure FeasibilityTol slack wouldn't even print — the fact that lines appear at all confirms the gap is
  fold-driven, not tolerance-driven.
- Worked example, θ=1.0, one cell: prior solve leaves Beef 0.55 / Winter cereals 0.45 with AgTech EI =
  0.44 on Winter cereals. Fold merges all 0.45 WC into Beef ⇒ `folded[m, r, WC] = 0` ⇒ am lb clamped
  0.44 → 0, gap 4.4e-01. Matches the logs: 2–15 cells, gaps up to ~0.49.

### Why harmless in the default runs

- The three AMs being clamped are **reversible** (`AG_MANAGEMENTS_REVERSIBLE`). For reversible AMs the
  solver hardcodes `model_lb = 0` regardless (`solver.py:351,357,404,415`), so these clamped lb values
  never reach Gurobi — the lines are purely informational.
- No land or adoption is lost: the true composition lives in `data.ag_dvars` / `ag_man_dvars` and is
  recovered by the two-stream accounting (`X_acct`). Only the solver-world **lower bound** is dropped.

### The one real caveat

If a **non-reversible** AM (HIR - Beef/Sheep, Utility Solar PV, Onshore Wind) ever sits on a
folded-away host under θ=1.0, its lock-in lb would silently drop to 0 and the solver could abandon it.
None of those appear in the inspected clamp logs, so no lock-in was released — but it's the thing to
watch while θ=1.0. Recovering 1e-6-level gaps means θ→0 (pure exact model, no folding), at the cost of a
much larger transition problem.

---

## 20260710 — θ-fold accounting leak fixed via a two-stream (folded / accounting) dvar model

### TL;DR

The θ-fold (`get_folded_base_ag_dvar`, θ = `EXACT_REACHABILITY_MIN_FRACTION`) merges every sub-θ ag
land-use fraction into its cell's dominant to shrink the transition problem. But the folded array was also
the solver's **global base composition** anchoring `X_ag`, so **every** accounting constraint (profit,
water, GHG, GBF2/3/4/8, production) scored a folded sliver at its *dominant's* coefficient instead of its
own. Fix: a **two-stream** model — the folded decision vars `X_ag` (`dvar_flow`) still drive the transition
machinery, and a second **accounting stream `X_acct`** (a LinExpr re-expression, **no new Gurobi
variables**) carries each true land-use's fraction. Accounting builders read the **raw** coefficient
against `X_acct`; only biodiversity/GHG/water/profit/production switched streams — structural builders
(cell usage, node balance, adoption, renewable) keep `X_ag`.

`X_acct` is built by **transferring** each sliver's live share off the dominant:
`X_acct[s] += (A_s/fd)·X_ag[d]`, `X_acct[d] -= same`, where `fd = folded_dom = raw + Σ A_s`. The dominant
starts as a copy of `X_ag[d]` and ends at `(raw/fd)·X_ag[d]` — conservation is structural (one pass, no
scaling loop). Equivalent to an area-weighted **blended coefficient** on the dominant var; the blend form
was the first cut, replaced by the transfer for readability. Validated `== blend` to ~1e-15
(`script_2_validate_two_stream.py`) and end-to-end on the θ grid.

### The leak (diagnosis)

`X_ag = folded_base + Σin − Σout`, and folding preserves cell totals — so **ag/non-ag AREA is
fold-invariant** (the coarse metric can't see the bug). The distortion is in the *per-LU scoring*: in a
GBF2 mask cell, a high-`c` natural remnant folded into low-`c` grazing is scored at grazing's `c`. The doc
`jinzhu_inspect_code/Correc_folding/correct_dvar_folding.md` decomposes the 2010 mask fold to −0.89 Mha
GBF2, predicting the solver backfills EP and over-states the 2050 GHG sink (~16%) and profit (~4%).

### Implementation

- **`transitions.py`** — one cached `_fold_ag_dvar` → `(folded_base, fold_map)`; wrappers
  `get_folded_base_ag_dvar` (`[0]`) and `get_ag_dvar_fold_map` (`[1]`). Fold map keys:
  `from_m/cells/from_j` (sliver), `vals` (A_s), `to_m/to_j` (receiver), `folded_dom` (fd).
- **`input_data.py`** — coefficients ship **raw** (no blend). New fields `ag_fold_map` and
  `acct_cells_mrj = feasible_ag_cells ∪ {fold-map sliver cells}` (folded slivers have base 0 so are absent
  from `feasible_ag_cells`; the per-`j` builders must visit them or the term is silently dropped).
- **`solver.py`** — `_setup_ag_accounting_vars` builds `X_acct`; accounting builders (objective, GHG,
  GBF2, demand) swap `feasible_ag_cells_mrj → acct_cells_mrj` and `X_ag → X_acct`; water &
  `_build_biodiv_contr_expr` keep their fixed region `ind` (already visit every cell). Production is now
  uniform (the earlier per-commodity correction is gone). Guard `if not isinstance(dom, (Var, LinExpr))`:
  a `no_go`-banned/force-converted dominant has no var → skip (never mint a fresh var — `X_acct` must stay
  a re-expression of the folded vars).

### Proof it's in the Gurobi model (`script_4_coeff_probe.py`)

Minimal Gurobi build of the worked cell (Beef-mod 0.70 | Beef-nat 0.20 | WC 0.10; `g` = 2.0 / −3.0 / 0.5):
the coefficient on the **same dominant var** in the GHG expression is **+2.00 folded → +0.85 two-stream**
(the area-weighted blend), and the stay-point score moves +2.00 (folded, charging the sequestering natural
remnant as grazing) → **+0.85 = TRUE** `Σ g_j·base_j`. So `X_acct` (a LinExpr, not a new var) literally
changes constraint-matrix coefficients on the folded vars.

### Empirical θ-sweep (RF5, renewables off, GHG on, GBF2 binding; 2050)

Before = `20260708_Check_diff_trans_theta` (folded), after = `20260710_...with_accounting` (two-stream),
θ ∈ {0.01…0.50} (`script_3_compare_area.py`):

- **GHG — the clean positive result.** Net sink over-grows with θ: **before Δ = 17.4 Mt (16%)**, matching
  the predicted −16%; **after Δ = 13.3 Mt (12%)** → **θ-drift cut ~24%**, pulling the spurious high-θ sink
  back ~4 Mt.
- **The fix is unambiguously applied** — before≡after at θ=0.01 (nothing folds → `X_acct ≈ X_ag`; GHG Δ
  0.01 Mt) and they diverge **monotonically with folding** to ~4 Mt at θ=0.5. A no-op couldn't produce a
  θ-scaled gap.
- **Area unchanged** (ag Δ 0.5%, non-ag Δ ~10% — same before/after): fold-invariant, and the ag↔non-ag
  split is pinned by the binding constraints, so the scoring correction doesn't move it.
- **GBF2 = 61.975 Mha-eq flat** — a binding hard constraint sits on its target under either accounting, so
  the achieved value can't reveal the drift (the fix changes *how* it's met, not the number).
- **Profit** moved (higher at high θ, +0.4 B AUD) but did **not** converge (after Δ slightly larger); the
  observed direction is opposite the doc's "+4% over-state" — flagged, not claimed.

### Takeaway / caveats

The two-stream behaves exactly as a **Role-2 (accounting) correction**: it fixes the *scored* quantity
(GHG drift −24%, matching the predicted magnitude; validated to machine precision), leaving the physical
allocation almost unmoved because binding constraints pin it. The **residual ~12% GHG drift is the
untouched Role-1 transition approximation** (folded slivers still move through the dominant's reachability
and pay its cost — the fix deliberately doesn't touch this; T_MAT audit: ag2ag forbids are all `→natural
land`, so that reachability distortion is small). v2 proportional-shed is exact on full flips (~99% of
dominant shifting per the θ=0.01 data) and only approximate on ~1.3 Mha of partial-shed cells.

---

## 20260706 — fold-into-dominant θ model replaces the ag sliver stay-pin

### TL;DR

Turned `EXACT_REACHABILITY_MIN_FRACTION` (θ) into a per-cell **exact↔crisp dial** for the transition
model. Sub-θ ag land-use fractions are now **folded into the cell's dominant source** BEFORE the solver
world is built (`get_folded_base_ag_dvar`, cached), so every remaining source keeps its own flow-delta
vars and **no land is ever pinned in place**. This retired the old sub-θ "sliver stay-pin" (which locked
immovable land) and the two support fixes it needed. θ→0 recovers the pure exact per-source model; θ→1 is
pure crisp dominant-source; default θ changed 0.01 → 1.0.

### Fold-into-dominant (`8adf9b2f`)

- **`get_folded_base_ag_dvar`** (cached) merges every `base ≤ θ` non-dominant fraction into the cell's
  dominant (same-lm dominant if it is itself > θ, else the overall dominant; the overall-largest LU is the
  exempt receiver of last resort). Cell totals are preserved, so `ag_mask` / Σ-X accounting is unchanged.
- The **source map, ag2ag ub, and ag-man lb** are all rebuilt from the folded base so the solver world is
  self-consistent; `get_ag2ag_lb` is reduced to zeros (the stay-pin is superseded — folded sub-θ land
  stays mobile through the dominant's delta vars). The **true map `data.ag_dvars` is untouched** for
  reporting.
- **ghg**: exact transition-emission slices reuse the shared folded source map (a lazy import breaks the
  transitions↔ghg cycle) instead of re-deriving cell slices from the raw dvar.
- **non-ag transitions decoupled from θ** — always exact at the `ROUND_DECIMALS` floor (folding would
  fake-delete permanent commitments or grant free Destocked→ag reversion).
- ⚠️ This fold silently became the accounting base too — the **θ-fold accounting leak** fixed on 20260710
  (see that entry).

### Support fixes

- **`clamp_dvar_bound` unified** (`b9a5b576`): `_clamp_dvar_bound` moved to `tools.clamp_dvar_bound`, with
  all dvar-bound cleaning routed through it (ag/non-ag ub/lb/base, ag-man lb, non-ag ag-mask cap) under
  uniform logging. `get_ag2ag_lb` no longer caps lb at the raw ag2ag ub — capping zeroed the pin exactly
  where EXCLUDE/no-go bans a held sliver, deleting its X var and making cell-usage saturation infeasible.
  Pinned-sliver entries/area are now reported, and pins landing on banned entries flagged.
- **dataprep x_mrj eligibility** (`86e6fff8`): the precipitation/irrigation overlays could ban a cell's
  own observed 2010 `(lm, lu)` (~360 cells), leaving that holding with no X var and forcing its conversion
  in the first sim step. Each cell's observed `(lm, lu)` is now OR-ed back into `x_mrj`; all other entries
  untouched.

---

## 20260702 — flow transition model finalised + write.py reporting rebuilt on the deltas

### TL;DR

Completed the migration to the **per-source delta / min-cost-flow** transition model and rebuilt all
transition reporting in `write.py` on top of the solved delta variables. `TRANSITION_MODE` is gone —
the model is unconditionally exact per-source; ~60 orphaned crisp/blend/lumap functions were deleted
or consolidated; the solver, input_data, and write.py now speak one 3-step mental model:
**(1)** slice each base-year source `(from_m,from_j)` / `k` to its `dvar>θ` cells;
**(2)** build per-source `{source: array over those cells}` cost / GHG / feasibility dicts;
**(3)** the solver creates one delta var `D≥0` per `(source, cell, target)`, charges `Σ flow_cost·D`,
and ties `D` to `X` via source cap (`Σ out ≤ base`) + node balance (`X = base + Σin − Σout`).
Write.py then reports every transition quantity as `Σ leaf·D` — the exact quantity the solver priced.

Validated end-to-end: solver builds identical & solves OPTIMAL; all rebuilt writes reconcile with the
solver at ~1e-7; `write_data` runs the full task list (104 files). See the working folders
`jinzhu_inspect_code/Build_exact_trans_solver/` (solver) and `.../Fix_flow_trans_write/` (write) for
the per-step progress logs.

### Solver / input_data consolidation

- **`TRANSITION_MODE` removed** from `settings.py`; all cost/GHG/bound dispatchers are unconditional
  exact. Deleted the target-based GHG account (`get_ghg_transition_emissions_by_lumap` + the per-lever
  `_crisp`/`_blend` builders), the `_crisp` bound/exclude builders, and `ag_ghg_t_mrj`. The GHG
  constraint now sums ongoing `ag_g_mrj·X` + `Σ flow_ghg_ag2ag·D`.
- **APIs unified on `(base_year, target_year)` / source cell-maps.** `get_transition_matrices_ag2ag`
  is the single ag2ag primitive (dropped the `w_mrj`/`t_ij` hoist args; `get_wreq_matrices` is now
  `@lru_cache(maxsize=1)` so per-source calls are free). The 8 ag→nonag and 6 non-ag→ag levers each
  collapsed to one function; the assemblers own the shared cell map.
- **nonag→ag lumap-cost BUGFIX.** The AF/CP-belt `_to_ag` levers had combined the *lumap-based*
  `get_env_plantings_to_ag_base` (zeros cells by the dominant lumap) with the per-source livestock
  cost — dropping the plantings-cost portion on ag-dominant cells that held an AF/CP dvar sliver. Now
  both portions are per-source over the same `k_cell_map` cells; threshold unified to
  `EXACT_REACHABILITY_MIN_FRACTION`.
- **Per-source feasibility dicts** (`get_feasible_ag2ag_mrj` / `_nonag2ag_mrj` / `_ag2nonag_rk`) —
  keyed/shaped like the flow_cost dicts, `= (target ub>0) ∧ (source T_MAT row) ∧ (¬diagonal)`. The
  solver's `_setup_flow_vars` is now pure materialisation (the opaque dense `ag_exists` rebuild is
  gone). Culling is decommissioned (incompatible with per-source flow; module kept for reference,
  nothing imports it).
- **Solution now carries source-keyed delta dicts**: `dvar_D_ag2ag_mrj` / `_ag2nonag_rk` /
  `_nonag2ag_mrj`, extracted from the SOLVED `F_*` vars (not `max(0,X_new−x_old)` — an X-diff can
  neither split ag2ag from nonag2ag inflows nor attribute a flow to its source). Stored per year on
  `data.delta_dvars_ag2ag` / `_ag2nonag` / `_nonag2ag`. `nonag2ag` covers reversible Destocked→ag.

### write.py Phase-4 rebuild (5 functions + the start→end matrix)

All transition blocks rebuilt as `Σ_src leaf[src]·D[src]`, scattered to global cells via the base-year
source maps; schemas (filenames, CSV columns, NetCDF layer dims) preserved so the Vue report is
unaffected. Functions: `write_transition_ag2ag`, `write_transition_ag2nonag`,
`write_transition_nonag2ag` (incl. the previously-placeholder Destocked→ag area),
`write_economics` (ag2ag + ag2nonag transition cost), `write_ghg` (transition-GHG account).
Each validated against an independent `Σ leaf·D/gap` recomputation at ~1e-7, with cross-function
agreement (economics ag2ag total ≡ transition_ag2ag_cost total).

- **`write_area_transition_start_end` rebuilt as a per-cell Markov chain** over land pools. The old
  `dvar_start × dvar_end` composition cross-product assumed independence; the new code forward-
  propagates an attribution matrix `A[cell, start_pool, current_pool]` period-by-period from the delta
  dicts (all 3 directions), so multi-hop paths resolve to true start→end pairs (e.g. Beef→Sheep→EP →
  a single **Beef→EP** row, zero intermediates). Validated: synthetic 2-hop exact; real single-hop
  mass-conserves (rel 5e-8) and moved-area ≡ ΣD×area (rel 5e-8).
- **ag2nonag double-weighting bug fixed** in the old `write_transition_ag2nonag` (it multiplied
  per-unit × delta AND × target dvar).
- **Water semantic fix**: `transition_ag2ag_water` reported a $-valued licence-cost delta under the
  "ML" label; now the true ML requirement change (`wreq[to] − wreq[from]`).

### Why true transition cost < the old lumap approximation

Two isolation tests, both on a solved solution (so no confounds):

- **Ceiling (pure composition cross-product):** reporting one solution as `base_frac × target_frac`
  gives true/lumap ≈ **0.16** — because for a fully-ag cell the cross-product evaluates to
  `1 − Σ_j base_j·target_j`, the cell's **Gini–Simpson diversity**, not its change: a mixed cell that
  *stays put* (base=target=`p_j`) is still charged `1 − Σ p_j²` (≈ 1−1/N ≈ 0.83 for N≈6 LUs at RF5)
  while the true flow `D`=0. So the error is `(actual churn)/(compositional diversity)` and scales
  with RESFACTOR (≈0 at RF1, large at RF5). This 0.16 is the theoretical ceiling of over-count.
- **Actual historical method (dominant-lumap × net-ΔX):** the archive baseline (BLENDED=False) did
  NOT use the pure cross-product — it billed `l_mrj[dominant From] × clip(X_new−X_old)[To] × unit_cost`.
  Reporting the SAME solution that way vs the true per-source `Σcost·D` gives **true/lumap ≈ 0.634**
  (i.e. the old method over-charged ag2ag cost by ~58%, by billing the whole cell's target increase
  at its *dominant* source's rate). **This 0.634 is the faithful number**; the 0.16 is only the ceiling.

The drop seen against the old baseline is therefore substantially a **reporting-correctness** effect
(removing the dominant-source over-charge), decomposed cleanly below.

### Matched-settings comparison (MNES_likely 2010→2020) + method/solution decomposition

Re-ran the fresh solve with `BIO_QUALITY_LAYER='MNES_likely'` to match the baseline (only `CULL_MODE`
remains unmatchable — decommissioned). Full `write_outputs` (104 files) vs the pre-refactor baseline:

- **Non-transition core reproduces tightly** (demand/physically anchored): crosstab / demand /
  production-targets / water-limits / offland-GHG = 0.00%; ag area 0.11%, ag revenue 0.12%, GHG total
  0.97%, ag cost 1.24%, ag profit 2.49%. → non-transition writes are structurally unchanged.
- **Discretionary categories diverge 12–41%** (non-ag economics/area, AM, biodiversity-overall):
  correctly-priced cheaper transitions relocate discretionary land (non-ag area per-LU: EP 470k→19k,
  RP 1.92M→1.54M, Destocked 137k→0, Beef/Sheep AF →0). Not a reporting error — a different optimum.
  `switches-lumap` (net-change-per-LU) shows 2.57e9→513M, but its raw national total is a comparator
  artifact (summing a signed per-(region,LU) metric; 2.57e9 ≫ total land 464M).

**Method-vs-solution decomposition of the −19% ag2ag-cost gap** (reporting the NEW solution under the
OLD method — B — via the dominant-lumap cost map; A=baseline, D=new/true):

| | ag2ag cost $/yr | |
|---|---|---|
| A old sol, old method | 2,278,261,755 | |
| B **new sol, old method** | 2,898,804,326 | |
| D new sol, new method | 1,837,370,279 | |

`D/A = 0.807 = (B/A) × (D/B) = 1.272 × 0.634`:
- **Solution effect B/A = ×1.272** — the flow solution transitions **+27% more** land (cheaper,
  correctly-priced transitions → the solver does more; same phenomenon as the discretionary shifts).
- **Method effect D/B = ×0.634** — on the **same** solution, true per-source cost is **−37%** vs the
  dominant-lumap accounting.
- **Net D/A = ×0.807** — the −19% observed in the file comparison.

So the reported transition-cost fall is a reporting-correctness effect (−37%) partly offset by the
new solution transitioning more than the old *target-based* baseline (+27%, B/A).

### Controlled causal test — the lumap cost is a solver DISTORTION, not just a level shift

Re-solved 2010→2020 twice under the IDENTICAL flow model / MNES settings / 2010 base, differing only
in the ag2ag cost coefficients: true per-source `flow_cost` vs the dominant-lumap map sliced to each
source (`get_ag2ag_dominant_source_cost_mrj`, i.e. every source in a cell billed the dominant rate):

| | ag2ag transitioned area (ha) | total non-ag area (ha) |
|---|---|---|
| TRUE-cost (per-source) | 6,338,199 | 786,384 |
| LUMAP-cost (dominant) | **31,811,116 (5.0×)** | 386,934 |

**The lumap cost makes the solver transition 5× MORE ag land** (and convert *less* to non-ag) — the
OPPOSITE of a simple "higher cost → less transition". Reason: the lumap approximation bills every
source at its cell's DOMINANT rate; most cells are dominated by cheap-to-convert LUs, so expensive
MINORITY sources (natural land, high-value crops) are **under-priced** → the solver cheaply churns
land it shouldn't. This is consistent with p10 (non-uniform mis-pricing): on the true solution the
lumap over-reports those specific moves (D/B=0.634), but the lumap-cost SOLVER abandons them for the
cheap-under-lumap moves instead. **Corrected takeaway:** the exact flow cost differs from lumap not
just in level but in per-source STRUCTURE; as a solver cost the lumap approximation distorts the
optimum (≈5× excess ag churn, too little non-ag), and the exact implementation removes that
distortion — the correctly-incentivised optimum has far less spurious ag churn.

### Report robustness

`create_report_data.py` did `pd.concat([df for path in X['path'] if not df.empty])` at ~24 sites,
which raised "No objects to concatenate" when a category was empty in EVERY year (e.g. a scenario
with zero non-ag revenue — the flow optimum here picked pure EP/CP with no ag-revenue component,
implied non-ag revenue = −$80, below the abs>1 filter). Added `_read_concat(paths)` (skips empties;
if all empty, returns one empty frame carrying the first file's columns so downstream ops don't fail)
and replaced the 23 fragile sites. NOT caused by the transition rebuild; a pre-existing fragility the
correct-but-sparse flow outputs exposed. The Vue report otherwise reads the identical output schema.

### Model size / solve time (RF5, 2010→2020 step: flow vs old target-based baseline)

| | target-based baseline | per-source flow | Δ |
|---|---|---|---|
| rows | 2,226,970 | 4,653,753 | +109% |
| columns | 3,125,985 | 7,308,928 | +134% (+3.70M delta vars) |
| nonzeros | 16,665,749 | 32,987,886 | +98% |
| presolve | 8.1s | 26.9s | 3.3× |
| barrier | 115.7s (98 it) | 579.3s (188 it) | 5.0× |
| total solve | 121.0s | 589.9s | **≈4.9×** |

Slowdown is transition-magnitude dependent (2010→2020 is the largest reallocation, 3.7M delta vars;
a mid-horizon 2030→2035 step was ~1.7×). The trade is deliberate: a heavier solve buys exact,
auditable transition accounting and a write side that reads flows instead of re-modelling them.
(Raw solver objective is the internal rescaled value — not comparable across formulations; the
economic comparison lives in the output CSVs.)

### Full-run performance (RF5, 2020→2050, 7 steps: old target-based vs new flow)

Per-year model size and Gurobi solve time parsed from each run's `LUTO_RUN__stdout.log` (7 solves
each; one `Optimize a model with …` + one `Solved in … seconds` per simulated year):

| | Old (target/lumap) | New (flow) | Ratio |
|---|---|---|---|
| Rows (mean) | 2.23 M | 4.62 M | 2.1× |
| Columns (mean) | 3.10 M | 7.02 M | 2.3× |
| Nonzeros (mean) | 17.1 M | 32.7 M | 1.9× |
| Total solve (7 steps) | 21.3 min | 73.5 min | 3.4× |

Per-step solve runs ~2–4 min old vs ~8–14 min new; the wall-clock ratio (3.4×) exceeds the
matrix-size ratio because barrier iteration counts grow on the larger problem. Consistent with the
single-step 2010→2020 figure above (that step is the heaviest reallocation, ~4.9×). Comparison
scripts + rendered table/plots in `jinzhu_inspect_code/Fix_flow_trans_write/plots/`
(`perf_size_time_table.png`, `make_perf_table.py`, plus stacked old-vs-new area/economics/water/GHG/
transition plots and the split-cell start→end heatmap). Plain-language write-up:
`jinzhu_inspect_code/Fix_flow_trans_write/POST.md`.

### Dead-code cleanup

Removed `tools.get_ag_to_ag_water_delta_matrix_by_lumap` — the map-based ($/cell) water-licence delta
with the dry/irr `.any()` masks. Zero callers remained: the live per-source
`get_ag_to_ag_water_delta_matrix` covers the transition-cost path, and the ag2ag water *report* uses
the inline `wreq[to] − wreq[from]` ML delta (not this $ function). Also fixed its stale docstring
cross-reference. Pure removal, no behaviour change.

---

## 20260701 — pre-solver transition-cost validation (L0–L6): 5 bugs fixed + source-keyed flow_ghg

### TL;DR

Before wiring the flow-based solver (Phase 3), the transition-cost data layer was validated
bottom-up in 7 layers across all three modes (crisp/blend/exact) on `data_2050.lz4`
(RESFACTOR=5, base_year=2030, target_year=2035). **All layers green; all six solver-gate checks
pass (1.3, 1.4, 2.2, 3.3, 4.4, 5.5).** Five production bugs were found and fixed (F1–F5) — with F5
fixed, `get_input_data` now completes in exact mode. Also built the source-keyed **`flow_ghg_ag2ag`**
transition-GHG-emissions dict (the physical parallel of `flow_cost_ag2ag`).

### Bugs fixed

**F1 — exact GHG cell-slice threshold mismatch** (`agricultural/ghg.py`, `agricultural/transitions.py`).
`get_ghg_{lvstk_natural_to_modified, unall_natural_to_lvstk_natural, unall_natural_to_modified}_exact`
sliced cells at `10**(-ROUND_DECIMALS)` = 1e-6 while `get_base_dvar_mj_cell_map` used
`EXACT_REACHABILITY_MIN_FRACTION` = 0.01. Any exact source with a dvar sliver in (1e-6, 0.01]
produced a shape mismatch → crash in `get_transition_matrices_ag2ag_exact` (e.g. (0,1) Beef-modified:
20931 vs 20935). **Fix:** all three use `EXACT_REACHABILITY_MIN_FRACTION`; the `threshold` param was
removed from `get_base_dvar_mj_cell_map` so there is a single value.

**F2 — Destocked→ag charged $0 + uncharged carbon** (`non_agricultural/transitions.py`). Destocked→ag
was modelled as ag2ag-from-Unallocated-natural. But `T_MAT[Destocked→Unalloc-modified]=10390` (allowed)
while `T_MAT[Unalloc-natural→Unalloc-modified]=NaN` (prohibited), so the proxy's transition-exclude
zeroed the ENTIRE column (establishment + water + carbon) → `Destocked→Unalloc-modified` cost **$0** on
3684–4111 cells despite 3378/3684 holding carbon stock (mean 3.0, max 74.8 tCO₂/ha/yr). Since Destocked
is the only reversible non-ag source, this was a live free exit dodging both the $10390/ha transition
and the carbon penalty. **Fix:** new shared `get_destocked_to_ag_base` uses Destocked's OWN
`T_MAT[Destocked→to_j]` + water license + UNMASKED carbon release; crisp/blend/exact are thin wrappers.

**F3 — Destocked exact threshold** (`non_agricultural/transitions.py`). `get_destocked_to_ag_exact`
sliced at 1e-6 not θ (same class as F1); folded into the F2 rewrite (exact wrapper uses
`EXACT_REACHABILITY_MIN_FRACTION`).

**F4 — `get_ag_t_mrj` crash** (`solvers/input_data.py`). Called `.astype(np.float32)` on the now-dict
flow-cost return → crashed `get_input_data` for *every* mode. **Fix:** removed the stale `.astype`
(dict leaves are cast to float32 during the economy rescale).

**F5 — exact-mode GHG-account bug** (`agricultural/ghg.py`). `get_ag_ghg_t_mrj` passes the fractional
`ag_X_mrj` to `ag_ghg.get_ghg_transition_emissions`, whose three transition-GHG dispatchers RAISED in
exact mode → blocked the real `get_input_data` end-to-end in exact. The transition-GHG **account** is
target-indexed (added to the ongoing GHG and applied to `X_new` in the GHG constraint), so exact shares
the same base-year frac-weighted computation as blend — exact's per-source GHG is carried by the
transition COST, not the account. **Fix:** the three dispatchers now route blend AND exact to the
`_blend` variants (crisp / no-dvar → `_crisp`). Verified: `get_input_data` completes in exact mode.

### Follow-up: source-keyed `flow_ghg_ag2ag` (physical parallel of `flow_cost`)

The F5 reroute is an interim bridge for the **current target-based** GHG constraint
(`ag_ghg_t_mrj · X_new`, solver.py:1010). The transition GHG is source-dependent (natural land
releases stock; modified doesn't), so — exactly like the transition cost — it belongs in the flow.
Added `ag_ghg.get_ghg_transition_emissions_from_base_year(data, base_year)` →
`dict[(from_m,from_j)] → arr[to_m, cells, to_j]` (raw t CO2), mirroring `flow_cost_ag2ag`'s per-mode
source-keying/weighting (crisp dominant-slice; blend `frac_s/eligible_rj` — same factor as the blend
cost; exact sum of the three `get_ghg_*_exact`). Wired into `input_data`/`SolverInputData` as
`flow_ghg_ag2ag` (GHG-rescaled, **not yet consumed** — Phase 3 will sum `Σ flow_ghg·F` in the GHG
constraint and drop the target-based `ag_ghg_t_mrj` + the F5 reroute). Validated: crisp collapse-back
to `get_ag_ghg_t_mrj` bit-for-bit; `amortise(flow_ghg × carbon_price) == flow_cost's GHG components`
(all three modes) — i.e. it is the raw-emissions twin of the already-validated cost dict.

### Confirmed properties (not bugs)

- `economy_scale = 292269`, mode-independent; post-rescale flow_cost band all < 1e3 (nonag2ag ≈968 closest).
- Base-year dvar negatives are benign float32 barrier round-off (|·| ≤ 6e-8, all < θ).
- ag→planting `T_MAT` is two-tier: 1500 (broadacre) / 3680 (horticulture); natural-land sources NaN.
- Cross-mode: crisp collapse-back == flat base bit-for-bit; exact == crisp on shared cells; exact ≈ blend
  at pure cells. blend ag2nonag ≡ crisp holds exactly only on dryland single-`(m,j)`-source ag-dominant
  cells (diverges on irrigated / multi-source / tier-straddle / non-ag-dominant cells — all by construction).
- `ag_source_cells`/`nonag_source_cells` are exact-only; crisp/blend anchor flow vars on the cost dict's cells.

---

## 20260629 — nonag→ag: crisp refactoring correctness audit (step_12)

### TL;DR

`step_12_compare_crisp_vs_old_nonag2ag.py` compares the current crisp nonag→ag
implementation against the old implementation (commit `a684ec3f`) inlined directly.
**All 8 tests pass.** The new implementation is more correct on all axes; costs are
directionally higher but not order-of-magnitude different from the old version.

---

### Test results

| Test | What it checks | Result |
|------|---------------|--------|
| 1  | EP lm=1 diff at nonag cells == amortised `NEW_IRRIG_COST` | PASS |
| 1b | EP lm=0 nonag diff negligible (<1 AUD) | PASS |
| 2  | NEW sheep AF livestock component zero at non-sheep-AF nonag cells | PASS |
| 2b | OLD sheep AF livestock component non-zero at non-sheep-AF cells (confirms old bug) | PASS |
| 3a | OLD beef AF livestock component zero outside beef-AF cells | PASS |
| 3b | NEW beef AF livestock component zero outside beef-AF cells | PASS |
| 4  | Destocked NEW == OLD (regression; max_diff = 0.0) | PASS |
| 5  | Both OLD and NEW produce non-zero aggregate | PASS |

---

### Key structural differences (old crisp → new crisp)

**(A) EP/RP/CP-Block/BECCS → Ag: missing irrigation cost**

OLD `get_env_plantings_to_ag` computed water delta via `get_ag_to_ag_water_delta_matrix`
but did **not** add `NEW_IRRIG_COST` for irrigated targets at nonag cells.  NEW
`get_env_plantings_to_ag_crisp` explicitly adds:

```python
amortise(settings.NEW_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * data.REAL_AREA)
```

to lm=1 at nonag cells.  Effect: lm=1 transition costs are ~10.6% higher in NEW.

**(B) Sheep AF / CP-Belt → Ag: cell scope bug**

OLD `get_sheep_to_ag_base` applied sheep livestock costs to **all** nonag cells — it never
restricted to cells identified as sheep-AF in the lumap.  NEW uses `cell_mask` from
`tools.get_sheep_agroforestry_cells(lumap)` to zero the livestock component everywhere
else.  OLD had 391,967 cells with spurious livestock cost; NEW has zero.

**(C) Sheep → Ag: establishment cost silently dropped**

OLD used `np.einsum('rj,r->rj', e_rj, (all_sheep_lumap == 0))` where `all_sheep_lumap`
is always `sheep_j ≠ 0`, so the mask was always False and `e_rj_dry = 0`.  Sheep
establishment costs were silently zero everywhere.  NEW delegates to
`get_transition_matrices_ag2ag_crisp` which includes establishment correctly.

**(D) Beef AF → Ag: already correct**

OLD `get_beef_to_ag_base` already restricted beef livestock costs to beef-AF cells and
used `np.stack([e_rj, e_rj])` (no dry/irr split bug).  NEW agrees at beef-AF cells;
regression confirmed by Test 3a/3b.

**(E) Destocked → Ag: identical**

Both call `get_transition_matrices_ag2ag_crisp` with an all-unallocated lumap.  max_diff = 0.0.

---

### Per-LU cost comparison (RESFACTOR=5, BASE_YEAR=2045, TARGET_YEAR=2050)

| LU | OLD sum (AUD) | NEW sum (AUD) | rel diff |
|----|--------------|--------------|---------|
| EP / RP / CP-Block / BECCS | 39.8 T | 45.4 T | +10.6% |
| Sheep Agroforestry | 13.3 T | 15.1 T | +42.6% |
| Beef Agroforestry | 13.3 T | 15.1 T | +53.9% |
| Sheep CP Belt | 13.3 T | 15.1 T | +42.6% |
| Beef CP Belt | 13.3 T | 15.1 T | +58.7% |
| Destocked | 754 B | 754 B | 0.0% |

EP-family uplift is entirely from the missing irrigation cost.  AF/CP-Belt uplift is
primarily the establishment cost fix (C); cell-scope fix (B) also contributes.

---

## 20260629 — ag→nonag: `get_transition_matrix_ag2nonag` validation and profiling (step_10)

### TL;DR

`step_10_validate_ag2nonag.py` validates all three transition modes of the top-level
`get_transition_matrix_ag2nonag` function.  **All 13 tests pass**, with max_diff=0.0 for
every exact-vs-crisp cell comparison.  Profiling shows that the ag→nonag direction is
already cheap in crisp mode (0.7s / +12 MB) — the dominant cost is blend (2.3s / +121 MB)
and exact per-LU calls are essentially free (~0 MB each).

---

### Validation results

| Test | What it checks | Result |
|------|---------------|--------|
| 1 | `crisp separate=True` sums to `separate=False` per LU column | PASS |
| 2 | Crisp costs zero at all nonag cells | PASS |
| 3 | BECCS column == EP column in crisp_mat (BECCS is EP passthrough) | PASS |
| 4 | `blend separate=True` sums to `separate=False` | PASS |
| 5 | Blend vs crisp totals (informational) | PASS — blend_sum/crisp_sum = 1.017; expected difference |
| 6 | EP exact cells == crisp | PASS — max_diff=0.0, 49 combos, 141,494 cells |
| 7 | RP exact cells == crisp | PASS — max_diff=0.0 |
| 8 | Sheep AF exact cells == crisp | PASS — max_diff=0.0 |
| 9 | Beef AF exact cells == crisp | PASS — max_diff=0.0 |
| 10 | CP Block exact cells == crisp | PASS — max_diff=0.0 |
| 11 | Sheep CPB exact cells == crisp | PASS — max_diff=0.0 |
| 12 | Beef CPB exact cells == crisp | PASS — max_diff=0.0 |
| 13 | Destocked exact cells == crisp | PASS — max_diff=0.0 |

No float32 rounding differences at all — the ag→nonag exact functions are cleaner than
nonag→ag because the return shape is 1D `(n_cells,)` per `(m,j)` combo (no 3D
scatter-fill), so no ULP accumulation.

---

### Return type summary

| Mode | `separate=False` | `separate=True` |
|------|-----------------|-----------------|
| crisp / blend | `ndarray(NCELLS, N_NON_AG_LUS)` | `{lu_name: {component: ndarray(NCELLS,)}}` |
| exact (top-level) | **FAILS** — `np.array` on list of dicts | `{lu_name: {(m,j): ndarray(n_cells,)}}` |
| exact (sub-function) | `{(m,j): ndarray(n_cells,)}` | `{(m,j): {component: ndarray(n_cells,)}}` |

`mj_cell_map` (`get_base_dvar_mj_cell_map`) is keyed by `(from_m, from_j)` tuples and is
LRU-cached — 49 active combos at 2045, covering 141,494 dvar-active cells.

---

### Profiling (BASE_YEAR=2045, TARGET_YEAR=2050, RESFACTOR=5, NCELLS=186,648)

| Call | Time | Peak RAM delta |
|------|------|----------------|
| `crisp separate=False` | 0.7s | **+12 MB** |
| `crisp separate=True`  | 0.7s | **+21 MB** |
| `blend separate=False` | 2.3s | **+121 MB** |
| `blend separate=True`  | 2.2s | **+125 MB** |
| `mj_cell_map` build    | 0.2s | ~0 MB (LRU-cached after first call) |
| Any exact sub-function | ~0.2s | **~0 MB** |
| Any crisp sub-function (comparison call) | ~0.2s | +3 MB |

---

### Key observations

1. **Ag→nonag crisp is already cheap**: 0.7s / +12 MB for all 9 LUs — far cheaper than
   nonag→ag crisp (8.7s / +841 MB).  The result is a 2D `(NCELLS, N_NON_AG_LUS)` float32
   array (~6 MB) vs the nonag→ag 3D `(NLMS, NCELLS, N_AG_LUS)`.

2. **Exact allocates ~0 MB per LU**: each sub-function returns a dict of tiny per-combo
   arrays (141,494 cells × 49 combos) with no full-NCELLS intermediates.  The `(m,j)` loop
   avoids any broadcasting over NCELLS.

3. **Blend is the bottleneck at 2.3s / +121 MB**: the T_MAT weighted average loops over
   28 ag LUs and accumulates into a full-NCELLS array.  This is 3× slower than crisp and
   6× more RAM.

4. **BECCS is a pure EP passthrough**: `get_beccs_from_ag` delegates directly to
   `get_env_plant_transitions_from_ag_from_base_year` — identical result in all modes.
   Confirmed by column-equality test (Test 3).

5. **No conflict cells**: the 49 active (m,j) combos at 2045 have no overlapping cells
   (each cell has at most one dominant `(m,j)` in the base dvars), so exact-vs-crisp
   comparison is clean for all 141,494 dvar-active cells.

---


## 20260629 — nonag→ag: `get_transition_matrix_nonag2ag` validation and profiling (step_9)

### TL;DR

`step_9_validate_nonag2ag.py` validates all three transition modes of the top-level
`get_transition_matrix_nonag2ag` function.  **All 10 tests pass.**  Profiling reveals that
exact mode reduces EP-family peak RAM by ~94% (48 MB vs 840 MB) with a 22× speedup (0.4s
vs 8.7s) relative to the full crisp call.

---

### Validation results

| Test | What it checks | Result |
|------|---------------|--------|
| 1 | `crisp separate=True` sums to `separate=False` | PASS |
| 2 | Crisp costs are zero at all ag cells | PASS |
| 3 | `blend separate=True` sums to `separate=False` | PASS |
| 4 | Blend vs crisp totals (informational) | PASS — blend_sum/crisp_sum = 0.73; expected difference since crisp uses `lumaps[base_year]` while blend uses `non_ag_dvars[base_year]` |
| 5 | EP-family exact cells == crisp at same cells | PASS — max_diff=0.0 for all 8 active k's |
| 6 | Sheep AF exact cells == crisp | PASS — max_diff=0.0 |
| 7 | Beef AF exact cells == crisp | PASS — max_diff=0.0 |
| 8 | Sheep CPB exact cells == crisp | PASS — max_diff=4.0 (float32 ULP, within tol=556) |
| 9 | Beef CPB exact cells == crisp | PASS — max_diff=4.0 (float32 ULP, within tol=898) |
| 10 | Destocked exact cells == crisp | PASS — max_diff=0.0 |

Test 4 is intentionally informational: blend identifies cells via fractional dvars while
crisp uses integer lumap dominance, so a ~27% total difference is expected and correct.

---

### Profiling (BASE_YEAR=2045, TARGET_YEAR=2050, RESFACTOR=5, NCELLS=186,648)

**Full top-level call (all 9 non-ag LUs, full NCELLS):**

| Mode | Time | Peak RAM delta |
|------|------|----------------|
| `crisp separate=False` | 8.7s | +841 MB |
| `crisp separate=True`  | 9.2s | +1,289 MB |
| `blend separate=False` | 9.1s | +833 MB |
| `blend separate=True`  | 10.0s | +1,290 MB |

**Exact mode per individual LU (sparse active cells only):**

| LU | exact time | exact RAM | crisp RAM | active cells |
|----|-----------|-----------|-----------|--------------|
| EP-family (5 LUs via `ep_to_ag`) | **0.4s** | **+48 MB** | ~841 MB (full call) | 3,919 |
| Sheep Agroforestry | 1.7s | +322 MB | +372 MB | 1 |
| Beef Agroforestry  | 1.7s | +334 MB | +380 MB | 5 |
| Sheep CPB (Belt)   | 1.7s | +314 MB | +385 MB | 1 |
| Beef CPB (Belt)    | 1.7s | +314 MB | +380 MB | 5 |
| Destocked          | 1.4s | +595 MB | +595 MB | 4,167 |

---

### Key observations

1. **EP-family exact is the dominant win**: 0.4s / +48 MB vs 8.7s / +841 MB for the
   combined top-level call.  All five EP-type LUs share a single `ep_to_ag` object and
   only allocate arrays for the 3,919 active cells, giving a 94% RAM reduction and 22×
   speedup.

2. **AF/CPB exact saves ~10% RAM vs crisp per LU** (e.g. 322 MB vs 372 MB for Sheep AF)
   but wall-time is identical because the sub-function still builds full-NCELLS intermediate
   arrays before scattering results.

3. **Destocked exact = crisp in performance**: no sparse optimisation exists internally;
   the function allocates full-NCELLS regardless.  The 4,167 active cells make destocked
   the second-largest contributor by cell count after EP (3,919).

4. **Blend≈crisp design difference**: blend/crisp totals differ by ~27% because they
   identify cells differently.  This is correct — blend is intended for the solver's
   fractional dvar world; crisp is the current lumap integer approximation.

---


## 20260626 — nonag→ag: GHG exact mode, blend reformatting, `get_destocked_to_ag` refactor, `irrev_mask_r` guard

### TL;DR

Extended the nonag→ag exact transition cost work in
`luto/economics/non_agricultural/transitions.py`:

1. **GHG exact mode** added to `get_sheep_to_ag_base` and `get_beef_to_ag_base` — when
   `TRANSITION_MODE == 'exact'`, calls the three exact GHG helpers and scatters into a
   full `(NLMS, NCELLS, N_AG_LUS)` zero array.
2. **Synthetic-lumap comment** propagated to `get_sheep_carbon_plantings_belt_to_ag_exact`
   and `get_beef_carbon_plantings_belt_to_ag_exact`.
3. **Blend functions reformatted** — `get_sheep/beef_agroforestry_to_ag_blend` and
   `get_sheep/beef_carbon_plantings_belt_to_ag_blend` now use consistent five-section
   comments (fetch fraction → normalise → synthetic lumap → fetch costs → scale).
4. **`get_destocked_to_ag` refactored** into `_crisp`, `_blend`, `_exact` variants plus a
   dispatcher, following the same pattern as all other nonag→ag functions.
5. **`irrev_mask_r` zeroing restricted to crisp mode** — in blend/exact the solver
   multiplies costs by a delta-increment Gurobi variable; the lb-lock on the non-ag dvar
   already handles the irreversible fraction, so no additional zeroing is needed.

All changes verified by `step_8_verify_destocked_and_ghg_exact.py` — **11/11 PASS**.

---

### GHG exact mode in `get_sheep_to_ag_base` / `get_beef_to_ag_base`

When `TRANSITION_MODE == 'exact'`, instead of calling `get_ghg_transition_emissions` with
a synthetic lumap, the functions now call the three exact GHG helpers directly:

```python
cell_idx  = np.where(data.ag_dvars[base_year][0, :, sheep_j] > threshold)[0]
ghg_exact = (
    ag_ghg.get_ghg_lvstk_natural_to_modified_exact(data, base_year, 0, sheep_j)
    + ag_ghg.get_ghg_unall_natural_to_lvstk_natural_exact(data, base_year, 0, sheep_j)
    + ag_ghg.get_ghg_unall_natural_to_modified_exact(data, base_year, 0, sheep_j)
)
ghg_t_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
ghg_t_mrj[:, cell_idx, :] = ghg_exact
```

Both functions gained a `base_year=None` keyword.  The eight exact-mode callers were
updated to forward `base_year=base_year`.

---

### `get_destocked_to_ag` three variants + dispatcher

| Variant | Cell identity | Returns |
|---------|---------------|---------|
| `_crisp(data, yr_idx, lumap, lmmap, separate)` | `tools.get_destocked_land_cells(lumap)` | `ndarray(NLMS, NCELLS, N_AG_LUS)` |
| `_blend(data, yr_idx, base_year, separate)` | normalised `non_ag_dvars[base_year][:, destocked_k]` | `ndarray(NLMS, NCELLS, N_AG_LUS)` |
| `_exact(data, yr_idx, base_year, separate)` | `where(non_ag_dvars[base_year][:, destocked_k] > threshold)` | `{destocked_k: ndarray(NLMS, n_cells, N_AG_LUS)}` |

All three derive costs from an all-unallocated-natural lumap, then restrict to cells
identified by the respective mode.

---

### `irrev_mask_r` guard

Zeroing of irreversible nonag cells in `get_transition_matrix_nonag2ag` is now gated on
`settings.TRANSITION_MODE == 'crisp'`.  In blend/exact the solver uses a delta-increment
Gurobi variable whose LB is constrained by the irreversible non-ag dvar lb-lock —
no extra zeroing needed.  In crisp the integerised lumap can assign a spurious cost to
the free fraction of an irreversible cell, so zeroing remains correct there.

---

### Verification

A verification harness (11 tests) passed 11/11 on `data_2050.lz4`.  Section A (tests 1–8):
destocked crisp/blend/exact.  Section B (tests 9–11): sheep/beef GHG exact — zero in both
modes at 2050 because `sheep_j`/`beef_j` are not in `LU_LVSTK_NATURAL` at that checkpoint
(correct behaviour).

---

### Files changed

| File | Change |
|------|--------|
| `luto/economics/non_agricultural/transitions.py` | GHG exact mode in `get_sheep/beef_to_ag_base`; blend reformatting; `get_destocked_to_ag` three-variant refactor; `irrev_mask_r` crisp-only guard; synthetic-lumap comment in CP Belt exact functions |

---

## 20260625 — ag→nonag exact transition costs: all 8 non-ag LU types refactored and validated

### TL;DR

Extended `TRANSITION_MODE = 'exact'` to all ag→non-ag transition cost functions in
`luto/economics/non_agricultural/transitions.py`.  Each of the 8 non-ag LU types
(EP, RP, Sheep AF, Beef AF, CP Block, Sheep CP Belt, Beef CP Belt, Destocked) now has a
`_exact` variant returning `dict[(from_m, from_j)] → ndarray(len(cell_idx),)` and its
`_from_base_year` dispatcher routes to it when `TRANSITION_MODE == 'exact'`.  BECCS
delegates to the EP dispatcher and inherits exact mode automatically.

All exact functions validated at **100% match** vs crisp across all 49 `(from_m, from_j)`
combos per LU type (`step_3_compare_ag2nonag_exact_crisp_trans_cost.py`).

A bug in `get_destocked_from_ag_exact` was found and fixed: the HCAS benefit lookup used
the LU description string as key, but `BIO_HABITAT_CONTRIBUTION_LOOK_UP` is keyed by
integer LU codes — causing zero HCAS removal cost for all lvstk-natural combos.

---

### Function naming pattern

Each non-ag LU type now has three variants plus a dispatcher:

```
get_{lu}_transitions_from_ag_crisp(data, yr_idx, lumap, lmmap, separate)   # lumap-based
get_{lu}_transitions_from_ag_blend(data, yr_idx, base_year, separate)       # dvar-weighted avg
get_{lu}_transitions_from_ag_exact(data, yr_idx, mj_cell_map, separate)     # per-combo cell arrays
get_{lu}_transitions_from_ag_from_base_year(...)                             # dispatcher
```

`_blended` → `_blend` suffix normalised at the same time; `_crisp` suffix added to
previously-unnamed lumap variants to make the three modes explicit.

---

### Return type of exact functions (ag→nonag)

```python
# separate=False:
result: dict[(from_m, from_j)] → ndarray(len(cell_idx),)   # 1-D, $/cell/yr

# separate=True:
result: dict[(from_m, from_j)] → dict[component_name → ndarray(len(cell_idx),)]
```

This differs from ag→ag exact (`(NLMS, len(cell_idx), N_AG_LUS)` per combo) — for
ag→nonag the target LU is fixed per function so there is no `to_j` dimension.

---

### Cost components per non-ag LU type

| Non-ag LU | Cost components |
|-----------|----------------|
| Environmental Plantings | est (EP_EST_COST_HA[cell]), t_cost (T_MAT), w_rm_irrig |
| Riparian Plantings | est (RP_EST_COST_HA[cell]), t_cost, w_rm_irrig, fencing (RP_FENCING_LENGTH[cell]) |
| Sheep Agroforestry | est×AF_PROP, t_af×AF_PROP, t_sheep×(1−AF_PROP), w_rm_irrig, fencing (scalar) |
| Beef Agroforestry | same as Sheep AF with 'Beef - modified land' |
| CP Block | est (CP_EST_COST_HA[cell]), t_cost, w_rm_irrig |
| Sheep CP Belt | est×CP_BELT_PROP, t_cp×CP_BELT_PROP, t_sheep×(1−CP_BELT_PROP), w_rm_irrig, fencing (scalar) |
| Beef CP Belt | same as Sheep CP Belt with 'Beef - modified land' |
| Destocked | t_cost + HCAS_removal (both amortised), w_rm_irrig (gated on is_eligible) |
| BECCS | delegates to EP |

`est` and `t_cost` are amortised via `tools.amortise()`.  Fencing and `w_rm_irrig` are not
amortised.  All arrays are cast to `float32`.

---

### Key design decisions

- **Shared cell map**: all exact functions receive `mj_cell_map` from
  `ag_transitions.get_base_dvar_mj_cell_map(data, base_year)` — the same `@lru_cache`
  helper used by ag→ag exact.  Each dispatcher calls it once and passes the result in.
- **Per-cell arrays indexed inside the loop**: `EP_EST_COST_HA`, `RP_EST_COST_HA`,
  `AF_EST_COST_HA`, `CP_EST_COST_HA`, `REAL_AREA`, and `RP_FENCING_LENGTH` are per-cell
  arrays indexed as `data.XXX[cell_idx]` inside the combo loop.
- **No TRANS_COST_MULTS on T_MAT**: ag→nonag T_MAT lookups do not apply
  `data.TRANS_COST_MULTS`, matching existing crisp/blend behaviour.
- **Destocked eligibility gate**: `is_eligible = not np.isnan(t_j_raw[from_j])`.
  T_MAT is NaN for non-lvstk-natural LUs; water removal cost is zeroed for ineligible
  `from_j` regardless of `from_m`.

---

### Bug fixed: `get_destocked_from_ag_exact` HCAS integer-key lookup

`data.LU_LVSTK_NATURAL = [2, 6, 15]` (integer codes) and
`data.BIO_HABITAT_CONTRIBUTION_LOOK_UP` is keyed by integer codes.  The initial
implementation looked up HCAS by LU description string:

```python
from_j_name = data.AGRICULTURAL_LANDUSES[from_j]
hcas_mult = HCAS_benefit_mult.get(from_j_name, 0.0)  # always 0.0 — string key not found
```

Fix: use the integer code directly.

```python
hcas_mult = HCAS_benefit_mult.get(from_j, 0.0)
```

Note: `get_destocked_from_ag_blend` has the same latent bug (`HCAS_benefit_mult.get(j_name, 0.0)`
where keys are integers) — HCAS removal cost is silently zero in blend mode.  Not fixed here
as it is pre-existing blend behaviour outside the scope of this refactor.

---

### Validation

Strategy per combo: build synthetic all-`from_j` lumap / all-`from_m` lmmap, call
`crisp(separate=True)`, slice at `cell_idx`, compare component-by-component vs exact.

Result: **100% match (max_diff = 0.0000)** across all 8 LU types and all 49 combos.

---

### Files changed

| File | Change |
|------|--------|
| `luto/economics/non_agricultural/transitions.py` | Added `_crisp`/`_blend` suffixes; added 8 exact functions; updated all 8 dispatchers; fixed `w_rm_irrig` float32 cast; fixed destocked HCAS int-key lookup |
| `luto/tools/__init__.py` | `get_ag_to_ag_water_delta_matrix` returns `.astype(np.float32)` for consistency |

---

## 20260625 — ag→ag exact transition costs: per-(from_m, from_j) cell arrays; two bugs fixed

### TL;DR

Added `TRANSITION_MODE = 'exact'` for ag→ag transitions in
`luto/economics/agricultural/transitions.py` and `ghg.py`.  In exact mode, transition costs
are computed as `dict[(from_m, from_j)] → ndarray(NLMS, len(cell_idx), N_AG_LUS)` rather than
collapsing the base year to a single lumap (crisp) or a weighted average (blend).  This
preserves full source-LU identity for later per-slice solver delta variables.

Two bugs were found and fixed during implementation:

1. **`get_ag_to_ag_water_delta_matrix`** — irrigation infrastructure cost applied only to
   the source LU column (`from_j`), not broadcast across all target LUs.
2. **`get_ghg_lvstk_natural_to_modified_exact`** — `result_arr` allocated with shape
   `(N_AG_MANS, …)` instead of `(NLMS, …)`, causing a shape mismatch at the caller.

Also renamed `BLENDED_TRANSITION_COSTS` (bool) → `TRANSITION_MODE` (str: `'crisp'` /
`'blend'` / `'exact'`) throughout settings, solver, input_data, and write modules.

---

### Motivation for exact mode

Both `crisp` and `blend` lose information about which source `(m, j)` a cell's fraction
came from.  `crisp` collapses to a single dominant LU; `blend` collapses to a weighted
average across all source LUs.  In `exact` mode every `(from_m, from_j)` slice of the
base-year dvar is kept separate, so the solver can create a **per-slice delta variable**
and multiply it by the cost computed for precisely that source slice — no averaging, no
dominant-LU approximation.

---

### Cell-index map helper

`get_base_dvar_mj_cell_map(data, base_year)` → `{(from_m, from_j): cell_idx}` omits empty
combos (no cells with that `(m, j)` allocation in base year above threshold).

- Decorated `@lru_cache(maxsize=1)` — solver and cost function call this for the same
  `(data, base_year)` pair in one solve step; the second call is free.
- Shared by both ag→ag and ag→nonag exact functions — nobody re-derives `cell_idx` independently.

---

### Return type of `get_transition_matrices_ag2ag_exact`

```python
# separate=False (default):
result: dict[(from_m, from_j)] → ndarray (NLMS, len(cell_idx), N_AG_LUS)   # $/cell/yr

# separate=True (keys match get_ghg_transition_emissions(separate=True)):
result: dict[(from_m, from_j)] → {
    'Establishment cost':                       ndarray (NLMS, len(cell_idx), N_AG_LUS),
    'Water license cost':                       ndarray (NLMS, len(cell_idx), N_AG_LUS),
    'Livestock natural to unallocated natural': ndarray (NLMS, len(cell_idx), N_AG_LUS),  # zeros
    'Unallocated natural to livestock natural': ndarray (NLMS, len(cell_idx), N_AG_LUS),
    'Livestock natural to modified':            ndarray (NLMS, len(cell_idx), N_AG_LUS),
    'Unallocated natural to modified':          ndarray (NLMS, len(cell_idx), N_AG_LUS),
}
```

`cell_idx` is not embedded in the value — callers retrieve it from `mj_cell_map` directly.

---

### Cost components (ag→ag exact)

| Component | Source | Notes |
|-----------|--------|-------|
| Establishment | `T_MAT[from_j, to_j] × TRANS_COST_MULTS × REAL_AREA` | NaN→0; excluded transitions cost 0, solver enforce via exclude matrix |
| Water licence | `(w_target − w_base) × WATER_LICENCE_PRICE × mults` | Both from target-yr `w_req_mrj`; base = `w_req[from_m, cell_idx, from_j]` |
| New irrigation | `NEW_IRRIG_COST × mults × REAL_AREA` into `cost[1]` | Only when `from_m == 0` (was dryland) |
| De-irrigation | `REMOVE_IRRIG_COST × mults × REAL_AREA` into `cost[0]` | Only when `from_m == 1` (was irrigated) |
| GHG | sum of three exact GHG functions × carbon_price | Each returns zeros when `from_j` is not the relevant source LU |

---

### GHG exact functions

All three GHG transition types follow a consistent three-variant + dispatch structure:

| Transition type | exact function | dispatch raises |
|---|---|---|
| unall-natural → lvstk-natural | `get_ghg_unall_natural_to_lvstk_natural_exact(data, base_year, from_m, from_j)` | ValueError in `get_ghg_unall_natural_to_lvstk_natural` |
| lvstk-natural → modified | `get_ghg_lvstk_natural_to_modified_exact(data, base_year, from_m, from_j)` | ValueError in `get_ghg_lvstk_natural_to_modified` |
| unall-natural → modified | `get_ghg_unall_natural_to_modified_exact(data, base_year, from_m, from_j)` | ValueError in `get_ghg_unall_natural_to_modified` |

Each exact function returns `(NLMS, len(cell_idx), N_AG_LUS)` in t/cell, or zeros immediately
if `from_j` is not the relevant source LU type.  The three are summed in
`get_transition_matrices_ag2ag_exact` before applying `amortise × carbon_price`.

---

### Bug fixed: `get_ag_to_ag_water_delta_matrix` irrigation cost column broadcast

The original crisp implementation applied the irrigation infrastructure cost only to the
**source LU column** (`from_j`), leaving all other target LU columns unaffected:

```python
# BEFORE (bug): l_mrj[0] is True only at j=from_j for a synthetic lumap
w_delta_mrj[1] = np.where(l_mrj[0], w_delta_mrj[1] + new_irrig,    w_delta_mrj[1])
w_delta_mrj[0] = np.where(l_mrj[1], w_delta_mrj[0] + remove_irrig, w_delta_mrj[0])
```

A cell switching from dryland wheat to **irrigated rice** paid no irrigation setup cost;
switching from dryland wheat to **irrigated wheat** did — an incorrect LU-dependent result
for an infrastructure cost that is LU-independent.

**Fix** (`tools/__init__.py`): collapse the `j` dimension first to get a per-cell boolean,
then broadcast across all target LUs:

```python
# AFTER (fixed): True for any dryland/irrigated cell regardless of from_j
dryland_r   = l_mrj[0].any(axis=1, keepdims=True)   # (NCELLS, 1)
irrigated_r = l_mrj[1].any(axis=1, keepdims=True)   # (NCELLS, 1)
w_delta_mrj[1] = np.where(dryland_r,   w_delta_mrj[1] + new_irrig,    w_delta_mrj[1])
w_delta_mrj[0] = np.where(irrigated_r, w_delta_mrj[0] + remove_irrig, w_delta_mrj[0])
```

Verified by `step_1_compare_ag2ag_exact_crisp_trans_cost.py`: water license cost matches
100% across all 49 combos after the fix (previously 34–97%).  The fix applies equally to
crisp and blend (blend calls the same function internally).

---

### Bug fixed: `get_ghg_lvstk_natural_to_modified_exact` wrong array shape

`result_arr` was allocated with shape `(data.N_AG_MANS, …)` (number of ag-management
options, ~10) instead of `(data.NLMS, …)` (= 2).  The function fills `result_arr[to_m, ...]`
for `to_m ∈ {0, 1}`, and callers expect `(NLMS, len(cell_idx), N_AG_LUS)`.  Fixed by
changing the allocation to `(data.NLMS, …)`.

---

### Validation

Validates `get_transition_matrices_ag2ag_exact` against `get_transition_matrices_ag2ag_crisp`
for all 49 `(from_m, from_j)` combos. Strategy: build synthetic all-`from_j` lumap /
all-`from_m` lmmap, call crisp(separate=True), slice at `cell_idx`, compare component-by-
component. Result: **100% match** for all components after the two bug fixes above.

---

### Files changed

| File | Change |
|------|--------|
| `luto/economics/agricultural/transitions.py` | Added `get_base_dvar_mj_cell_map`; added `get_transition_matrices_ag2ag_exact`; updated dispatcher |
| `luto/economics/agricultural/ghg.py` | Added `_exact` variants for all three GHG transition types; dispatch raises `ValueError` for exact |
| `luto/tools/__init__.py` | Fixed irrigation infrastructure cost broadcast in `get_ag_to_ag_water_delta_matrix`; added `.astype(np.float32)` return |
| `luto/settings.py` | `BLENDED_TRANSITION_COSTS` (bool) → `TRANSITION_MODE` (str); `BLENDED_TRANSITION_COSTS_N_JOBS` → `TRANSITION_MODE_N_JOBS` |
| `luto/solvers/solver.py`, `input_data.py`, `tools/write.py`, `data.py` | Updated all `BLENDED_TRANSITION_COSTS` references to `TRANSITION_MODE != 'crisp'` |

---

## 20260623 — blend non-ag UB: three runtime bugs fixed (ordering, RF5 edge cells, INFEASIBLE)

### TL;DR

Three consecutive runtime errors when running `BLENDED_TRANSITION_COSTS=True` at RF5
were traced to `get_non_ag_ub_matrices` in `luto/economics/non_agricultural/transitions.py`.

1. **`UnboundLocalError`** — irreversibility override block placed before `no_go_x_rk` was
   initialised (refactor scrambled block order).
2. **`GurobiError: Invalid value for Model.addConstr`** — 5 RF5 edge cells had Destocked
   UB=0 AND no ag vars, so the cell-usage expression was a bare integer `0`, not a Gurobi
   linear expression.
3. **Instant INFEASIBLE (presolve, 0 iterations)** — the narrow fallback in fix 2 only
   fired when `eligible_frac_r < FEASIBILITY_TOLERANCE`. For cells where `eligible_frac_r`
   was small-but-positive (e.g. 0.01) yet still less than `AG_MASK_PROPORTION_R` (e.g. 0.04),
   Destocked UB < what the cell-usage constraint requires → presolve detects contradiction.

---

### Fix 1 — block ordering in `get_non_ag_ub_matrices`

The irreversibility `for k` loop that sets `no_go_x_rk[..., k] = 1` must run AFTER
`no_go_x_rk = (t_rk * no_go_x_rk).astype(np.float32)` is computed. A prior refactor had
swapped the two blocks. Fix: restored correct ordering.

---

### Fix 2 → 3 — Destocked UB floor for livestock-natural cells

**Root cause (both fixes):** At RF5, `get_resfactored_lumap` assigns each coarsened cell to
one plurality LU via quota-fill + nearest-neighbour. Some edge cells end up with
`LUMAP = Beef-natural (j=2)` while `AG_L_MRJ` shows `Unallocated-natural (j=23)` with
fraction 0.04. The Destocked cap uses `AG_L_MRJ[:,:,LU_LVSTK_NATURAL].sum()` as the
eligible fraction. For these cells the sum = 0 (or very small), capping Destocked UB below
`AG_MASK_PROPORTION_R`. When Destocked is the only available LU (no ag vars, no other
non-ag LU allowed by T_MAT), the cell-usage constraint becomes unsatisfiable.

**Fix 2 (too narrow):**
```python
eligible_frac_r = np.where(
    lumap_is_lvstk_nat & (eligible_frac_r < settings.FEASIBILITY_TOLERANCE),
    data.AG_MASK_PROPORTION_R,
    eligible_frac_r,
)
```
Resolved the 5 worst cells (eligible_frac_r == 0) but left cells with
0 < eligible_frac_r < AG_MASK_PROPORTION_R still producing INFEASIBLE.

**Fix 3 (current, general):**
```python
lumap_is_lvstk_nat = np.isin(lumap, data.LU_LVSTK_NATURAL)
eligible_frac_r = np.maximum(
    eligible_frac_r,
    data.AG_MASK_PROPORTION_R * lumap_is_lvstk_nat.astype(np.float32),
)
```
For every livestock-natural cell (LUMAP ∈ LU_LVSTK_NATURAL), Destocked UB is floored at
`AG_MASK_PROPORTION_R` — the minimum needed to satisfy the cell-usage constraint. For
non-livestock-natural cells the floor is 0 (no change). This replaces the narrow
`< FEASIBILITY_TOLERANCE` check entirely.

The RP cap is also moved to the very end of the function (after Destocked cap) so it is
unconditionally the last cap applied regardless of BLENDED mode.

---

## 20260623 — ag2ag blend transition denominator fix: per-target eligible-source fraction

### TL;DR

`get_transition_matrices_ag2ag_from_base_year` (blend path) previously normalised the
accumulated weighted transition cost by `ag_frac_r` — the total ag fraction per cell,
broadcast identically over all target LUs. This inflated the denominator for any target
where some source LUs have `NaN` T_MAT entries (disallowed transitions), creating an
unintended per-unit cost discount analogous to the Destocked bug fixed earlier today.

**Fix**: replace the scalar-per-cell denominator with a per-(cell × target) eligible
fraction computed as a single matrix multiply.

File changed: `luto/economics/agricultural/transitions.py`.

---

### Root cause

Inside the blend loop, `get_transition_matrices_ag2ag` calls `nan_to_num` (line 139),
converting `NaN` T_MAT entries to `0` before accumulation. So for target `j_to`:

- Source LUs where `T_MAT[source, j_to]` is `NaN` → contribute **0** to numerator
- Source LUs where `T_MAT[source, j_to] = $0` (diagonal, self-transition) → also **0**
- Both still contributed their full `frac_j` to the old `ag_frac_r` denominator

The discount factor for a given target = `eligible_frac / total_ag_frac` where
`eligible_frac` = sum of source fractions with non-NaN T_MAT to that target.

Two types of zero-numerator contributor:

| Type | T_MAT value | Discount intended? |
|---|---|---|
| Self-transition (`j = j_to`) | `$0` (not NaN) | **Yes** — head-start discount: a cell already partly holding the target pays less |
| Disallowed transition | `NaN → 0` | **No** — analogous to the Destocked bug |

Because self-transitions are `$0` (not `NaN`), they remain in the new per-target
denominator, so the head-start discount is fully preserved.

---

### Fix

**Before** — one denominator per cell, broadcast over all targets:
```python
ag_frac_r      = ag_X_mrj.sum(axis=(0, 2))                      # (NCELLS,)
ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)
...
ag_t_mrj /= ag_frac_r_safe[np.newaxis, :, np.newaxis]           # same denom for every j_to
```

**After** — per (cell × target), excludes NaN sources:
```python
valid_mask       = ~np.isnan(t_ij)                               # (N_AG_LUS, N_AG_LUS)
ag_frac_per_lu   = ag_X_mrj.sum(axis=0)                         # (NCELLS, N_AG_LUS)
eligible_rj      = ag_frac_per_lu @ valid_mask                  # (NCELLS, N_AG_LUS) — matmul
eligible_rj_safe = np.where(eligible_rj > threshold, eligible_rj, 1.0)
...
ag_t_mrj /= eligible_rj_safe[np.newaxis, :, :]                  # per-target denom
```

`t_ij` is already computed before the parallel loop (hoisted invariant), so `valid_mask`
is free. The matmul is `(NCELLS × N_AG_LUS) @ (N_AG_LUS × N_AG_LUS)` — negligible cost.

---

### Practical impact

The magnitude depends on how many NaN off-diagonal entries exist in the ag→ag T_MAT.
Because most ag→ag transitions are permitted, NaN entries are sparse — the effect is
smaller than the Destocked fix (where the majority of fractions were ineligible). The
dominant remaining discounting in ag2ag is the intentional self-transition head-start,
which is unchanged.

---

## 20260623 — BLENDED_TRANSITION_COSTS: Citrus/Cotton/Summer-cereals +200-300%, Destocked +8714%; eligibility-discount bug fixed

### TL;DR

Comparing two RF=5 runs identical except for `BLENDED_TRANSITION_COSTS` (True=Blend,
False=Crisp) at `Custom_runs/Compare_Blend_with_solver_delta` revealed:

- Citrus +264%, Cotton +212%, Summer cereals +190% in Blend — lower unit cost due to
  fractional initial state preserved by the blend formula
- Destocked - natural land +8714% in Blend — eligibility expansion (correct) plus a
  per-unit cost discount (bug, now fixed)

The Destocked discount bug was fixed: `get_destocked_from_ag_blended` now normalises by
`eligible_frac_r` (lv-nat fraction only) instead of `ag_frac_r` (total ag fraction).

---

### Root cause 1 — Citrus / Cotton / Summer cereals: fractional initial state is cheaper

At RF=5, `AG_L_MRJ` (the base-year dvar) is set by `get_exact_resfactored_lumap_mrj()`,
which stores the *fraction* of each LU in each 5×5 coarsened cell (e.g. cell 13998 = 68%
Citrus + 32% Sugar in 2010).

`get_resfactored_lumap()` (used by the Crisp run) instead assigns each cell to ONE land use
via a quota-fill algorithm (rare LUs fill their quota first, using `ceil(count/RESMULT)`
cells). This loses fractional information — cell 13998 gets assigned to
"Unallocated-natural land" because it is the dominant LU at the coarsened scale.

**Blend 2010→2020 transition cost to Citrus for cell 13998:**
```
= 0.68 × T_MAT[Citrus→Citrus]      + 0.32 × T_MAT[Sugar→Citrus]
= 0.68 × $0                         + 0.32 × $45,000
= ~$14,400/ha   ← affordable
```

**Crisp 2010→2020 transition cost to Citrus for cell 13998:**
```
= T_MAT[Unallocated-natural→Citrus] = $58,210/ha   ← too expensive
```

The Blend solver assigns cell 13998 to Citrus in 2020 at $14k/ha. By 2025 it is already
100% Citrus (self-transition = $0). The Crisp cell remains Unallocated-natural indefinitely.
The cost gap compounds: each successive year Blend pays $0 while Crisp still faces $58k.

Citrus, Cotton, and Summer cereals all share the same mechanism: they are minority LUs in
many RF=5 cells (T_MAT diagonal = $0, so any existing fraction dramatically reduces blend
cost), while their Crisp lumap assignment erases that fraction and defaults to the
plurality LU which has a high transition cost.

---

### Root cause 2 — Destocked - natural land: eligibility expansion + per-unit discount

**Crisp `get_destocked_from_ag` (line 802):**
```python
cells = np.isin(lumap, data.LU_LVSTK_NATURAL)
trans_cost_r[~cells] = 0.0   # non-lv-nat cells forbidden
```
At RF=5, many cells with partial livestock-natural fractions are assigned to "Beef-modified"
by the quota fill → their Destocked transition cost is zeroed → effectively hard-excluded
from ever becoming Destocked-natural.

**Blend `get_destocked_from_ag_blended` (before fix):**
```python
for j in AGRICULTURAL_LANDUSES:
    if np.isnan(T_MAT[j, 'Destocked']): continue
    t_destock_r += T_cost[j] * frac_j
trans_cost_r = t_destock_r / ag_frac_r_safe   # denominator = TOTAL ag fraction
```
Any cell with any livestock-natural dvar fraction gets a non-zero Destocked cost and
becomes eligible. For a mixed cell (40% Beef-natural + 60% Beef-modified):

```
unit_cost = (0.40 × T_cost) / 1.0 = 0.40 × T_cost   ← 60% discount vs pure lv-nat cell
```

The discount arises because the non-eligible ag fractions (Beef-modified, crops) inflate
the denominator without contributing to the numerator. Discount factor =
`eligible_frac / total_ag_frac`.

This is two separate effects:
1. **Eligibility restored** (physically correct): cells that genuinely contain
   natural-grazing land CAN transition to Destocked. The crisp run was incorrectly
   excluding them based on the plurality-LU assignment.
2. **Per-unit discount** (bug): each eligible cell pays less than a pure lv-nat cell,
   attracting the solver to large volumes of cheap Destocked transitions across rangelands.

The +8714% is driven by both effects combined — a vast population of newly-eligible mixed
rangeland cells, each at a discounted price. Citrus/Cotton only experienced the cost
effect (same eligibility); Destocked experienced both, explaining the far more extreme ratio.

---

### Fix — normalise by eligible fraction, not total ag fraction

`get_destocked_from_ag_blended` in `luto/economics/non_agricultural/transitions.py`:

**Before** (denominator = all ag LUs):
```python
ag_frac_r      = dvar_base.sum(['lm', 'lu']).values
ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)
trans_cost_r   = amortise((t_destock_r / ag_frac_r_safe) * REAL_AREA)
removal_cost_r = amortise((hcas_cost_r / ag_frac_r_safe) * EP_EST_COST_HA * REAL_AREA)
```

**After** (denominator = lv-nat eligible LUs only, accumulated in the same loop):
```python
eligible_frac_r = np.zeros(data.NCELLS, dtype=np.float32)
for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
    trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Destocked - natural land').item()
    if np.isnan(trans_cost):
        continue
    frac_j = dvars[:, :, j_idx].sum(axis=0)
    t_destock_r     += trans_cost * frac_j
    eligible_frac_r += frac_j              # only lv-nat fractions reach here
    ...
eligible_frac_r_safe = np.where(eligible_frac_r > threshold, eligible_frac_r, 1.0)
trans_cost_r   = amortise((t_destock_r / eligible_frac_r_safe) * REAL_AREA)
removal_cost_r = amortise((hcas_cost_r / eligible_frac_r_safe) * EP_EST_COST_HA * REAL_AREA)
```

For a 40% Beef-natural + 60% Beef-modified cell:
- Before: unit_cost = 0.40 × T_cost (60% discount)
- After: unit_cost = (0.40 × T_cost) / 0.40 = T_cost (same as pure lv-nat cell)

Cells with zero lv-nat fraction: `t_destock_r = 0` and `eligible_frac_r_safe = 1.0`
→ unit_cost = 0 (still ineligible, naturally).

`irr_frac_r` (water infrastructure removal cost) is left unchanged — it uses total ag
irrigation, which is unrelated to Destocked eligibility.

Note: this fix removes the per-unit discount but does not add a hard solver-side cap
preventing the Destocked dvar from exceeding the eligible fraction. That would require
a cell-level upper-bound constraint and is left for future work.

---

### Inspect scripts

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `script_1_compare_area.py` | Compare area CSVs from both run archives across all years | `*_comparison.csv`, `*_diff_pct_australia.csv` |
| `script_2_trace_transition_cost.py` | Compute blend vs crisp unit T_MAT cost for top cells per target LU at 2030 | `transition_cost_comparison.csv`, per-cell breakdown in log |
| `script_3_root_cause.py` | Three-layer root cause (state mismatch → unit cost → realised cost × delta) for Citrus/Cotton/Summer-cereals | `root_cause_cells.csv`, detailed log per cell |

Data: `Custom_runs/Compare_Blend_with_solver_delta/Run_G0001` (Crisp),
`Custom_runs/Compare_Blend_with_solver_delta/Run_G0002` (Blend).

---

## 20260619 — AgMgt GHG abatement exceeds Ag emissions: scope-1 filter missing + Biochar exogenous soil carbon

### TL;DR

Agricultural management (Am) GHG abatement incorrectly exceeded the associated Ag scope-1
emissions for 17 land uses at 2050. Winter Cereals was the most extreme case: Ag scope-1
emissions = 1.89 Mt, but PA + AgTech EI + Biochar abated 4.15 Mt (2.20×). Two root causes:

1. **Scope-1 filter not applied to AgMgt functions** — PA, AgTech EI, and Biochar were
   reducing scope-3 emission sources (chemical application, irrigation, pesticide) that
   are absent from the Ag baseline.
2. **Biochar's `IMPACTS_soil_carbon` is exogenous carbon** — it adds new soil organic
   carbon from the biochar material itself and is not bounded by existing Ag soil emissions.

After both fixes: zero land uses with Am > Ag (overall ratio 0.67×, down from 1.80×).

---

### Background: Scope-1 baseline switch (Feb 2025)

Since Feb 2025, LUTO uses the 'AusTIMES agriculture sector scope 1 baseline'. Scope 1
includes only direct soil/enteric emissions:

- **Crops**: only `CO2E_KG_HA_SOIL` (direct soil emissions)
- **Livestock**: only enteric fermentation, dung/urine, manure, leachate/runoff

Scope-3 sources (chemical application, irrigation, pest management, fuel, fodder, seed)
were removed from the Ag baseline. But the AgMgt functions continued to report reductions
in ALL sources including the scope-3 ones — creating a mismatch where Am abatement
exceeded the Ag base that Am was notionally reducing.

---

### Root cause 1 — Scope-1 filter missing in Am GHG functions

`get_precision_agriculture_effect_g_mrj` and `get_agtech_ei_effect_g_mrj` in `ghg.py`
iterated over all four crop emission types:

```python
for co2e_type in ['CO2E_KG_HA_CHEM_APPL', 'CO2E_KG_HA_CROP_MGT',
                  'CO2E_KG_HA_PEST_PROD', 'CO2E_KG_HA_SOIL']:
    # no scope-1 guard — all sources reduced regardless
```

For Winter Cereals at 2050:
- PA reduced 1.90 Mt total (1.90 Mt scope-3 + scope-1 combined)
- After scope-1 filter: PA reduces only 0.96 Mt (soil emissions only)
- AgTech EI reduced 1.42 Mt → 0.33 Mt after filter

**Fix**: added a scope-1 guard inside the inner loop of each Am function:

```python
if settings.USE_GHG_SCOPE_1 and co2e_type not in settings.CROP_GHG_SCOPE_1:
    continue
```

Applied to `get_precision_agriculture_effect_g_mrj`, `get_agtech_ei_effect_g_mrj`
(crop loop and irrigation block), and `get_biochar_effect_g_mrj_crop`.

---

### Root cause 2 — Biochar IMPACTS_soil_carbon is exogenous carbon

`get_biochar_effect_g_mrj` contained two logically distinct sequestration mechanisms:

- **Crop emission reductions** (`CO2E_KG_HA_CROP_MGT`, `CO2E_KG_HA_SOIL`): reductions
  in existing Ag emission sources — bounded by the Ag emissions baseline
- **Soil carbon sequestration** (`IMPACTS_soil_carbon`): biochar material sequesters
  new soil organic carbon. This is **exogenous** — not a reduction of existing Ag
  emissions and therefore not bounded by the Ag scope-1 pool

For Winter Cereals at 2050, after the scope-1 fix, Biochar still contributed ~0.83 Mt
(0.08 Mt crop + 0.75 Mt soil carbon). The 0.75 Mt exogenous soil carbon was responsible
for the remaining 1.12× Am/Ag ratio.

---

### Biochar function split

`get_biochar_effect_g_mrj` was split into three functions:

```python
def get_biochar_effect_g_mrj_crop(data, yr_idx):
    """Crop emission reductions (CO2E_KG_HA_CROP_MGT, CO2E_KG_HA_SOIL).
    Scope-1 filtered. Bounded by Ag emissions."""
    ...

def get_biochar_effect_g_mrj_soil(data, yr_idx):
    """Exogenous soil carbon sequestration (IMPACTS_soil_carbon).
    Not bounded by Ag emissions."""
    ...

def get_biochar_effect_g_mrj(data, yr_idx):
    """Wrapper: full effect = crop reductions + soil carbon."""
    return get_biochar_effect_g_mrj_crop(data, yr_idx) + get_biochar_effect_g_mrj_soil(data, yr_idx)
```

The solver still uses the full wrapper (`get_agricultural_management_ghg_matrices` calls
`get_biochar_effect_g_mrj`). Biochar soil carbon legitimately contributes to total GHG
balance — it just should not be compared against the Ag scope-1 pool as a bounded
emission reduction. The write-side proportional cap in `write.py` handles output
representation when total Am abatement would push a land use's net GHG below zero.

---

### Three-stage quantitative analysis (RF5, 2050, using archived dvars)

Diagnostic script `07_full_story_summary.py` compared Am abatement vs Ag scope-1
emissions across three stages using saved dvars from 2050:

| Stage | Description | Overall Am/Ag | LUs with Am > Ag |
|---|---|---|---|
| 1 — Original | Scope-3 in Am, scope-1 in Ag | 1.80× | **17** |
| 2 — Scope-1 fix | Am restricted to scope-1 sources | 0.95× | **7** |
| 3 — Scope-1 fix + no Biochar soil C in comparison | Exogenous soil C excluded | 0.67× | **0** |

#### Winter Cereals breakdown (Ag scope-1 = 1.89 Mt):

| Stage | PA (Mt) | AgTech EI (Mt) | Biochar (Mt) | Total Am (Mt) | Ratio |
|---|---|---|---|---|---|
| 1 — Original | −1.90 | −1.42 | −0.83 | −4.15 | **2.20×** |
| 2 — Scope-1 fix | −0.96 | −0.33 | −0.83 | −2.11 | **1.12×** |
| 3 — Scope-1 fix + no soil C | −0.96 | −0.33 | −0.08 | −1.36 | **0.72×** |

The most extreme cases in the original (Stage 1):

| Land use | Ag scope-1 (Mt) | Am abatement (Mt) | Ratio |
|---|---|---|---|
| Winter legumes | 0.0046 | −0.2496 | 54.5× |
| Summer legumes | 0.0022 | −0.0107 | 4.9× |
| Summer oilseeds | 0.0014 | −0.0065 | 4.7× |
| Winter oilseeds | 0.1014 | −0.3626 | 3.6× |
| Winter cereals | 1.8856 | −4.1528 | 2.2× |

Legumes and oilseeds had extreme ratios because their scope-1 soil emissions are tiny
(little direct soil emission) while Am was reducing all scope-3 agrochemical sources.

---

### Solver-side cap: decided against

A solver-side `_add_am_ghg_cap_constraints()` constraint was prototyped that would
prevent Am abatement from exceeding Ag emissions per cell via slack variables. This was
removed because:
1. The scope-1 fix (Stage 2) already brought the overall ratio to 0.95× (below 1.0)
2. Biochar soil C is exogenous and legitimate — capping it in the solver would
   incorrectly penalise a valid sequestration pathway
3. The constraint added 56 slack variables and significant complexity without improving
   correctness

The write-side proportional cap in `write.py` remains (scales Am down in output when
`ag_ghg_base + am_total < 0`). This is a reporting-only correction and does not affect
solver optimisation.

---

### Files changed

| File | Change |
|---|---|
| `luto/economics/agricultural/ghg.py` | Added scope-1 guard to `get_precision_agriculture_effect_g_mrj` and `get_agtech_ei_effect_g_mrj` (crop loop + irrigation block). Split `get_biochar_effect_g_mrj` into `_crop`, `_soil`, and wrapper. |
| `luto/solvers/solver.py` | No change (solver-side cap was not adopted). |
| `luto/solvers/input_data.py` | No change (extra fields for cap were prototyped then removed). |
| `luto/tools/write.py` | Write-side proportional cap retained as-is using full `am_total`. |

---

## 20260617 — Blended ag2ag transition weights normalization bug: un-normalised weights under-charge mixed-use cells

### TL;DR

`get_transition_matrices_ag2ag_from_base_year` in
`luto/economics/agricultural/transitions.py` (the `BLENDED_TRANSITION_COSTS = True`
path) accumulated a weighted sum of per-source transition costs using raw ag dvar fractions
as weights:

```python
ag_t_mrj += ag_X_mrj[m, :, j][None, :, None] * from_current_lus_t_mrj
```

For a pure-agricultural cell, `Σ_{m,j} ag_X_mrj[m,r,j] = 1` — weights sum to 1 and the
blended cost is a proper weighted average.  For a cell with any non-agricultural fraction
(EP, RP, destocked land, etc.), the same sum equals `ag_frac_r < 1`, so the resulting
blended cost is `ag_frac_r × (weighted-average cost)` — **under-charged by the
non-agricultural fraction**.

The solver uses these matrices as whole-cell cost coefficients; the actual allocation
fractions are handled by the solver's own decision variables.  Under-charging the per-cell
coefficient means the solver sees a falsely cheap transition cost for every mixed-use cell.

**Fix**: divide the accumulated result by `ag_frac_r_safe` (zero-guarded per-cell sum)
after the parallel loop.  The algebraic equivalence `t_mrj_norm = t_mrj_unnorm / ag_frac_r`
follows because `ag_frac_r[r]` is a constant scalar for every `(m, j)` in the outer sum
and factors out of the linear combination — no re-run of the 56 parallel tasks is required.

Fix committed on branch `jinzhu`.

---

### Quantitative impact (RESFACTOR=5, 2045→2050, NCELLS=186,648)

| Metric | Value |
|--------|-------|
| Cells with `ag_frac_r < 0.99` (affected) | 57,010 / 186,648  (**30.5%**) |
| Cells with `ag_frac_r < 0.50` (strongly affected) | 25,990  (**13.9%**) |
| Correction factor `1/ag_frac_r` — p95 | **2.78×** |
| Correction factor — max | ~9.0× |
| Max absolute cost difference (per cell, over all `(m, j_to)`) | ~$177M |
| Beef - modified land dryland, unnorm mean cost | $54M/cell |
| Beef - modified land irrigated, unnorm mean cost | $33M/cell |
| Beef - natural land dryland, unnorm mean cost | $18M/cell |

P95 correction factor of 2.78× means 5% of cells were under-charged by at least 2.78×
in every blended transition cost column.  Cells concentrated along vegetation-change
boundaries (EP/RP / dryland mixed zones) are most affected — precisely the cells the model
most wants to transition.

---

### Algebraic proof of the fix

Let `f(m,j,m_to,r,j_to)` be the per-cell transition cost contribution from source `(m,j)`
to target `(m_to, j_to)` at cell `r` (what `get_transition_matrices_ag2ag` returns).

**Current (un-normalised):**
```
ag_t_mrj[m_to, r, j_to] = Σ_{m,j}  ag_X_mrj[m,r,j]         × f(m,j,m_to,r,j_to)
                         = ag_frac_r[r] × Σ_{m,j} w(m,r,j)  × f(m,j,m_to,r,j_to)
  where w(m,r,j) = ag_X_mrj[m,r,j] / ag_frac_r[r]  (normalised weight, sums to 1)
```

**Correct (normalised):**
```
ag_t_mrj_norm = t_mrj_unnorm / ag_frac_r[r]
              = Σ_{m,j} w(m,r,j) × f(m,j,m_to,r,j_to)   ← proper weighted average
```

Since `ag_frac_r[r]` is identical for all `(m,j)` at a given cell, dividing the
accumulated result is exactly equivalent to normalising the weights before the sum.

---

### Fix applied to `transitions.py`

Two additions to the `else` branch of `get_transition_matrices_ag2ag_from_base_year`:

**Before the parallel loop** — compute the normalisation denominator:
```python
ag_frac_r = ag_X_mrj.sum(axis=(0, 2))                                         # (NCELLS,)
ag_frac_r_safe = np.where(ag_frac_r > 10 ** (-settings.ROUND_DECIMALS), ag_frac_r, 1.0)
```

`10 ** (-settings.ROUND_DECIMALS)` replaces the former hard-coded `1e-6`, tying the
threshold to `settings.ROUND_DECIMALS = 6`.  Cells with no agricultural fraction
accumulate 0 anyway, so the safe-divisor of 1.0 is harmless.

**After the accumulation loop** — normalise in-place:
```python
if separate:
    for key in ag_t_mrj:
        ag_t_mrj[key] /= ag_frac_r_safe[np.newaxis, :, np.newaxis]
else:
    ag_t_mrj /= ag_frac_r_safe[np.newaxis, :, np.newaxis]
```

Both the `separate=True` (dict of named arrays) and `separate=False` (single float32
array) branches are covered.

---

### Grouped serial alternative: correct but 6× slower (step 4)

A serial grouped approach was prototyped as a potential replacement for the 56-task
parallel loop (`blended_ag2ag_grouped()` in `step_4_grouped_vs_parallel.py`).

**Algorithm:**

1. Partition cells by unique value in `data.lumaps[base_year]`.
2. For each `(lu_code, r_idx)` group, slice `ag_X_mrj[:, r_idx, :]`.
3. For each `m_src`, find active `j*` columns (`max(ag_X_slice[m_src], axis=0) > threshold`).
4. For each `(m_src, j_col)` in `j*`, build a sub-lumap with only `r_star` cells set to
   `j_col` (background = non-ag fill), call `get_transition_matrices_ag2ag`, and accumulate.
5. After all groups: `result /= ag_frac_r_safe[...]`.

**Critical finding — non-ag lumap groups must be included:**

Cells whose `lumaps[base_year]` value is a non-ag code (EP, Riparian Plantings, etc.) can
still have nonzero `ag_X_mrj` from the previous solver's fractional allocations.  The
initial implementation iterated only over ag land-use codes and missed those cells,
producing a max absolute error of **1.44e8** (catastrophic).  After fixing the loop to
`for lu_code in np.unique(lumap)` (all unique values, not just ag LU codes):

| Metric | Value |
|--------|-------|
| Max absolute difference vs parallel | 416 (float32 accumulation-order noise) |
| Mean relative difference (nonzero cells) | 0.55% |
| `np.allclose(atol=1e-2, rtol=1e-2)` | True |
| Total `get_transition_matrices_ag2ag` calls | 127 (vs 56 for parallel) |
| Speed vs parallel | **0.16× (6× slower)** |
| Non-ag lumap groups with nonzero j_active | EP: 17 j*, Riparian: 18 j* |

**Why the grouped approach is slower:**

`get_transition_matrices_ag2ag` always allocates and operates on full-NCELLS arrays
internally (EXCLUDE mask, T_MAT, REAL_AREA, `w_mrj`).  Even though only `r_star` cells
contribute nonzero entries to the result, the function's per-call overhead (full-array
allocation × 127 calls) dominates.  The 6× slowdown arises because joblib threading
overhead is small (shared memory) while the per-call fixed cost is large.

**To realize the speedup**, `get_transition_matrices_ag2ag` would need an optional
`cell_idx` parameter that slices `EXCLUDE`, `T_MAT`, `REAL_AREA`, and `w_mrj` to the
`r_star` subset before any inner array construction.  This refactoring is not implemented;
the parallel approach remains the production code.

The 0.55% mean relative difference is a float32 accumulation-order artefact — both
implementations are mathematically identical; they differ only in the order they sum
`float32` partials across the parallel vs. serial call sequence.

---

## 20260612 — "Annualise write" double-division bug: annual-rate metrics were divided by `gap` a second time

### TL;DR

`luto/tools/write.py` introduced `get_year_gap(data, yr_cal)` to annualise
**period-aggregate** outputs (one-off transition costs/changes incurred since the previous
simulated year) by dividing by `gap`. But the same `/gap` division was incorrectly also
applied to several metrics that are **already annual rates** (revenue, cost, quantity,
water net yield, renewable MWh, and most GHG matrices) — double-annualising them. For
multi-year-gap runs (e.g. RESFACTOR=5, `SIM_YEARS = range(2020, 2051, 5)`, gap=5), this
under-reported these metrics by roughly an extra factor of `gap` (~80% low for gap=5,
matching `(gap-1)/gap`).

Found via a 5-year-gap vs 1-year-gap run comparison. Fixed by removing the
incorrect `/gap` from annual-rate quantities while preserving it on genuine
period-transition matrices, and re-validated by re-running `write_data` for the 5-year-gap
run (the systematic ≈-80% errors on Revenue/Cost/Quantity/Water Net Yield dropped to
single-digit %).

---

### Which metrics ARE divided by `gap` (period-aggregate, one-off transition values)

| Function | Variable(s) | What it represents |
|---|---|---|
| `write_economics` | `ag2ag_mrj` | Ag→Ag establishment cost incurred over the period since the previous simulated year |
| `write_economics` | existing-capacity solar/wind **CAPEX delta** | One-off installation capital cost incurred over the period |
| `write_economics` | `solar_potential`/`wind_potential` **CAPEX** (`dvar_delta * capex_xr`) | One-off installation capital cost for newly-allocated renewable AgMgt cells this period (added 20260613 — was previously missed because the `Annualise_write` validation runs had `RENEWABLES_OPTIONS` all `False`, so this code path was never exercised) |
| `write_economics` | `nonag2nonag_mat` | Non-ag→Non-ag transition cost over the period |
| `write_economics` | `ag2nonag_mat` | Ag→Non-ag transition cost over the period |
| `write_ghg` | `ghg_t_smrj` | GHG transition penalty incurred over the period |
| `write_transition_ag2ag` | `ag_trans_mat`, `ag_transitions_cost_mat`, `ghg_t_smrj`, `w_delta_mrj` | Area/cost/GHG/water deltas from ag-to-ag transitions over the period |
| `write_transition_ag2nonag` | `non_ag_transitions_area_mat`, `non_ag_transitions_flat`, `g_rk`, `w_rk` | Area/cost/GHG/water deltas from ag-to-non-ag transitions over the period |
| `write_transition_nonag2ag` | (mirror of ag2nonag) | Area/cost/GHG/water deltas from non-ag-to-ag transitions over the period |

### Which metrics are NOT divided by `gap` (already annual rates, or point-in-time stocks)

| Function | Variable(s) | Why no `/gap` |
|---|---|---|
| `write_quantity` | `ag_q_mrc`, `non_ag_p_rc`, `am_p_amrc` | `get_actual_production_lyr()` returns tonnes/KL **per year** for `yr_cal` |
| `write_economics` | `ag_rev_df`/`ag_cost_df` and `_mrj` variants (`get_rev_matrices`, `get_cost_matrices`) | Already annual revenue/cost rates ($/yr) |
| `write_economics` | existing-capacity solar/wind **OPEX** and **revenue** | Annual operating cost / revenue rates, not one-off |
| `write_renewable_production` | `renewable_energy` (`get_quantity_renewable`, `get_exist_renewable_capacity`) | Already MWh/year (`×8760` baked in) |
| `write_ghg` | `ag_g_rsmj`, `non_ag_g_rk`, `ag_man_g_mrj` (`get_ghg_matrices`, `get_agricultural_management_ghg_matrices`) | Already annual emission rates (t CO2e/yr) |
| `write_water` | `ag_w_mrj`, `non_ag_w_rk`, `ag_man_w_mrj` (`get_water_net_yield_matrices` + non-ag/ag-man) | Already annual yield rates (ML/yr) |
| `write_dvar_and_mosaic_map`, `write_dvar_area`, `write_area_transition_start_end`, `write_crosstab` | dvars / areas | Point-in-time stocks/snapshots, not flows |
| `write_biodiversity_quality_scores`, `write_biodiversity_GBF2_scores`, `write_biodiversity_GBF3_NVIS_scores`, `write_biodiversity_GBF4_SNES_scores`, `write_biodiversity_GBF4_ECNES_scores`, `write_biodiversity_GBF8_scores_groups`, `write_biodiversity_GBF8_scores_species` | biodiversity scores | Point-in-time stocks/snapshots, not flows |

---

### Reasons for the remaining (large) mismatches after the fix

**2020 anomaly (expected, not a regression):** at 2020 the fixed Run01 now reports the
*undivided* annual rate, while Run02 (not re-run) still reports the *old* value divided by
`gap=10`. This produces an apparent **+900%** diff for Revenue/Cost/Water/Quantity at 2020
— exactly `(gap-1) × 100% = 900%`, i.e. `Run01 / Run02 = 10`. This is an artefact of only
Run01 being re-run with the fix and is **not** a new bug; it would disappear if Run02 were
also re-run with the patched `write.py`. GHG and Area at 2020 are unaffected (0%) because
neither was changed by the fix at `gap=10` in a way that breaks the prior cancellation.

**GHG remains noisy post-fix — but for a reason unrelated to `write.py` entirely.**
`GHG_EMISSIONS_TCO2e` in `GHG_emissions_{yr}.csv` is **not** computed in `write_ghg()` —
it is read directly from `data.prod_data[yr_cal]['GHG']` (write.py:2650), which is
populated by the **solver** at solve time (`solver.py:1387`,
`self.ghg_expr.getValue() * scale_factors['GHG']`). This value never passes through any
`/gap` division in `write.py`, in either the old or fixed code — so the annualise-write
fix has **zero effect** on this number, before or after.

The large relative diffs are a genuine **optimisation-trajectory divergence** between the
5-year-step and 1-year-step runs, amplified by a **near-zero-crossing**:

- Total GHG = ongoing ag/non-ag emissions (positive) minus carbon sequestration from
  environmental/riparian plantings (negative, ramping up over decades per the
  age-based S-curve, `CARBON_EFFECTS_WINDOW`).
- Both runs cross from net-positive to net-negative (carbon sink) around 2035-2040, but
  at different rates — e.g. 2040: Run01 = -37.0M vs Run02 = -17.6M t CO2e. The absolute
  gap (~19M t) is modest relative to gross emissions (~50-70M t), but huge relative to a
  small net value near zero.
- *When* plantings are established differs because the optimizer makes different
  sequencing decisions at coarse (5-yr) vs fine (1-yr) time resolution, and sequestration
  realised in any given year depends on planting age — i.e. on history, which diverges
  between the two runs.

This is **not** evidence of a remaining `/gap` bug — the metric was never touched by
`write.py`'s annualisation logic in the first place.

---

## 20260610 — GBF4 SNES NUMERIC root cause: small-signal instability from sparse habitat (*Pomaderris subplicata*, North East)

### TL;DR

*Pomaderris subplicata* (LIKELY, North East NRM) caused `NUMERIC` status in the full
NECMA model at 2040→2045. The species has only **9 cells** with an extreme within-row
coefficient ratio of **62:1** (range [10.2, 633.0]). Satisfying its constraint requires
reallocating a tiny fraction of those cells — a change so economically negligible that
the barrier's dual variable approaches zero and the complementarity condition becomes
numerically indistinguishable from zero. The fix is to exclude the species from
`GBF4_SNES_EXCLUDE_REGION_SPECIES`. With this exclusion, `Run_G0002` reached OPTIMAL
at 2040→2045.

Run: `Custom_runs/NECMA_follow_runs/Run_G0001` (RF5, 2020–2050).

---

### Diagnostic method: single-species PBS workers

A new debugging workflow was established (see `docs/CLAUDE_SKILL/debug_species_infeasibility.md`):

1. **Launcher** loads the year checkpoint, enumerates all GBF4 SNES/ECNES `(region, species, presence)` triplets, and submits one PBS worker per target.
2. **Worker** builds the full base model (all constraints except GBF4), adds the single species constraint, and solves with barrier.
3. **Collector** aggregates result JSONs into a CSV with status, tightness, coeff_ratio, and solve time.

This directly catches both structural infeasibility (`tightness < 1`) and numerical breakdown (`NUMERIC`/`TIME_LIMIT` with high `coeff_ratio`). It supersedes the older MPS-maximisation approach, which only detected structural infeasibility and could not identify numerical failure modes.

**Key tightness formula (corrected):**

```
tightness = avail / lb_rescale
```

where `avail = val_vector[ind].sum()` is the **rescaled** max achievable score (from `GBF4_SNES_pre_1750_area_sr`, already rescaled by `rescale_solver_input_data()`), and `lb_rescale = lb_raw / scale_factor`. Values `< 1` indicate structural infeasibility; `≥ 1` with solver failure indicates numerical instability. The old formula `lb_raw / avail` mixed raw and rescaled units, giving an inverted and dimensionally inconsistent result.

---

### Per-species results (2040→2045, North East + Goulburn Broken)

83 targets tested. Of these 24 were `SKIP_NEG_TARGET` (negative targets, base year already exceeds threshold). Of the remaining 59:

| Status | Count | Notes |
|--------|-------|-------|
| OPTIMAL | 54 | cleanly solved |
| TIME_LIMIT | 4 | feasible but slow under combined constraint pressure |
| INFEASIBLE | **1** | *Pomaderris subplicata* — confirmed conflict |

The 4 TIME_LIMIT cases (*Lepidium monoplocoides*, *Nannoperca australis*, *Pimelea spinescens*, *Anthochaera phrygia*) have `tightness >> 1` and `coeff_ratio ~ 25–100` but did not cause `NUMERIC` in the full model.

---

### Root cause: small-signal numerical instability

**Species profile:**

| Metric | Value |
|--------|-------|
| `n_cells` | 9 (North East NRM) |
| `tightness` | 1.505 (avail > target — structurally feasible) |
| `coeff_ratio` | 62.1 |
| `coeff range` | [10.2, 633.0] |
| Single-species test status | **INFEASIBLE** |
| Full model status (with all GBF4) | **NUMERIC** |

The constraint is **structurally feasible** (`total_max_score = 2,053 > lb_rescale = 1,580`). Yet the full model returns `NUMERIC` before it can prove infeasibility; the single-species test (base model + one constraint) returns `INFEASIBLE` after computing a proper certificate.

**Mechanism:**

The barrier method's complementarity condition for each constraint is `dual × slack → 0`. To satisfy the Pomaderris constraint, the solver needs to reallocate a tiny fraction of those 9 cells to EP/RP — a change that costs economically negligible AUD against a model objective of `~1.1e4` (rescaled). The dual variable (shadow price) of the Pomaderris constraint therefore approaches zero. With `dual ≈ 0`, the product `dual × slack` is near zero from both sides — the barrier cannot distinguish "constraint satisfied at zero slack" from "constraint violated by a tiny amount."

The extreme `coeff_ratio = 62` compounds this: one cell (val=633) holds 27% of the total score. If other constraints prevent that cell from being freely allocated, the constraint cannot be satisfied, and the barrier's attempt to resolve this creates ill-conditioning.

At `OPTIMALITY_TOLERANCE = 0.01` and objective `~1.1e4`, Gurobi accepts solutions within `110` (rescaled) of optimal. The Pomaderris dual gap is far below this floor — the solver never experiences numerical pressure to resolve it, and the barrier converges to a numerically degenerate state.

**Why more-cell species are unaffected:** More cells → larger LHS summed → larger dual → detectable economic signal. *Grantiella picta* (803 cells, rescaled target 42,579) has a dual orders of magnitude larger; the complementarity condition is well-conditioned.

---

### What was ruled out

Cell-level diagnostics (`script_3_diagnose_pomaderris.py`, `script_4_water_conflict.py`, `script_5_nvis_conflict.py`) tested three competing-constraint hypotheses:

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| Water (OVENS RIVER) competition | **Ruled out** | All 9 cells in OVENS RIVER (reg_idx=157, target=83,574 ML). Switching free fractions to RP changes regional yield by only +3 ML — negligible. |
| GBF3 NVIS competition | **Ruled out** | 9 cells contribute at most 5.1% of any NVIS group's total capacity; no groups flagged as tight. |
| Raw structural infeasibility | **Ruled out** | `total_max_score = 2,053 > lb_rescale = 1,580`; 7 of 9 cells already locked in high-biodiv non-ag uses (EP/RP) from prior years. |

The infeasibility in the single-species test is not from lack of habitat but from the **combination of locked prior allocations (7 of 9 cells 56–96% committed) leaving thin free fractions that are over-subscribed by all base model constraints simultaneously**, with no single identifiable culprit. The small-signal mechanism prevents the barrier from navigating to the feasible point that the ILP analysis confirms exists.

---

### Cell-level diagnostics summary (from script_3)

| Cell | val | lock_frac | free_frac | max_score | note |
|------|-----|-----------|-----------|-----------|------|
| r=179511 | 633 | 0.88 | 0.12 | 579 | top contributor |
| r=180043 | 530 | 0.56 | 0.44 | 424 | |
| r=179682 | 388 | 0.72 | 0.28 | 310 | |
| r=179866 | 275 | 0.60 | 0.40 | 283 | |
| r=179865 | 194 | 0.41 | 0.59 | 120 | insufficient |
| r=179684 | 153 | 0.88 | 0.12 | 140 | insufficient |
| r=179867 | 102 | 0.00 | 1.00 | 82 | insufficient |
| r=179683 | 92 | **0.96** | 0.04 | 73 | FULLY LOCKED |
| r=180042 | 10 | 0.00 | 1.00 | 8 | negligible |

All 9 cells in OVENS RIVER (water region 157). 7 cells already in non-ag land uses from
prior years (cur_lu shows non-ag). `cur_lu = ?` means lumap index ≥ 28 (non-ag territory).

---

### Fix

Added to `GBF4_SNES_EXCLUDE_REGION_SPECIES` in both `luto/settings.py` and all 13 run copies in `NECMA_follow_runs/Run_G*/luto/settings.py`:

```python
# INFEASIBLE in NECMA single-species debug (2026-06-10): n_cells=9, tightness=1.505,
# coeff_ratio=62.1 (range [10.2, 633.0]). Extreme within-row coefficient spread on
# very few cells causes NUMERIC in the full model before infeasibility can be proven.
('North East', 'Pomaderris subplicata'),
```

**Validation:** `Run_G0002` (high GHG sensitivity) reached `Optimal objective 1.127e+04` at
2040→2045 after the exclusion. The Pomaderris constraint is excluded from the model; Gurobi
no longer encounters the near-zero dual.

---

### Long-term mitigation

Pre-compute `total_max_score = sum(val_vector[ind].sum() × best_contr[r])` for every GBF4 SNES/ECNES constraint before formulating the model. Flag any constraint where `tightness < 1.2` **and** `n_cells < 20` **and** `coeff_ratio > 30` as high-risk for numerical instability. Convert those constraints to soft-penalty terms in the objective (bounded dual → no KKT blow-up even when near-binding) rather than hard `>=` constraints.

---

## 20260606 — Barrier NUMERIC resolved at source: non-ag→ag transition fix; all snipping removed

### TL;DR (supersedes the 20260605 / 20260604 "structural degenerate-vertex / lock-in" conclusions)

The barrier `NUMERIC` failures were **not** an unavoidable structural property of a
degenerate vertex, and the non-ag lock-in LBs were **not** the true root cause. The real,
single culprit was an **impossible non-ag→ag transition being priced into the model and
inflated by a units bug**. Fixing the transition cost at the source makes the barrier solve
cleanly with **zero** bound-snipping and the original tight tolerances. All snipping
machinery has been removed and the solver restored to its clean pre-snipping form.

Commit: `a684ec3f` on branch `jinzhu`.

### Root cause

Two coupled transition-cost defects made Gurobi see a spurious, enormous discouragement
cost for reverting non-ag land to agriculture:

1. **Double `REAL_AREA` units bug** in `get_env_plantings_to_ag`
   (`luto/economics/non_agricultural/transitions.py`). The water-license delta is already
   per-cell (`$/cell`), but was multiplied by `REAL_AREA` a second time, inflating EP→ag
   transition costs by ~3 orders of magnitude.
2. **Impossible non-ag→ag transition charged at irreversible cells.** Cells already holding
   an irreversible non-ag land use cannot revert that locked fraction back to agriculture,
   yet (because transition costs are integerised-lumap based) the model still charged a
   non-ag→ag cost on the free fraction. Combined with (1), Gurobi effectively saw
   "transitioning one unit of locked non-ag land costs ~1e5 (million-AUD)", and the **dual
   objective exploded** → barrier `NUMERIC`.

### Fix (the only two changes needed)

1. `get_env_plantings_to_ag`: drop the second `REAL_AREA` multiplication on the
   water-license delta, mirroring the ag→ag path.
2. `get_crisp_irreversible_nonag_cell_mask` + `base_year` wiring through
   `get_transition_matrix_nonag2ag` (input_data.py) and cost reporting (write.py):
   **zero** the non-ag→ag transition cost on cells holding an irreversible non-ag LU, so
   Gurobi never sees the impossible transition.

### What was removed

`luto/solvers/solver.py` was restored to commit `d84fbb77` (clean, pre-snipping). Removed:
`non_ag_fixed_kr`, `saturated_cells_r`, the cell-usage RHS subtraction, and all lb/ub
"snap"/clamp logic. Settings restored to clean values: `ScaleFlag=0`, `BarConvTol=1e-5`,
`FeasibilityTol=1e-6`, `BarHomogeneous=0` (the relaxed band-aid tolerances are gone).

> The `NonAg lb capped` / `Ag man lb clamped` console lines that remain come from
> **input-data bound-value computation** (transitions.py / input_data.py) — legitimate
> lock-in lower bounds — not solver-side variable snipping.

### Validation — 2025→2030 checkpoint, live production path

| Metric | Value |
|---|---|
| Status | **2 (OPTIMAL)** |
| Barrier iterations | **124** |
| Objective | 9.525755e+03 |
| Matrix range | `[1e-04, 9e+01]` (ratio ~9e5) |
| Primal residual | 8.18e+08 → 4.31e-07, monotone, **no eruption** |
| Dual residual | ~1e-12 throughout |
| Crossover | clean, 3.25 s |

Isolation matrix corroborated the mechanism: buggy transition + scaling
**off** (`ScaleFlag=0`, the original failing config) struggled badly (357 iters, primal
residual erupting to ~30) while fixed-transition or scaling-on each independently
stabilised it — confirming the transition cost as the destabiliser.

### Moral

The right fix was the small, correct one — model the transitions properly. The lb/ub
snips, RHS subtraction, saturated-cell bookkeeping, and loosened tolerances were all
symptom-chasing that the correct model made redundant.

---

### Full storyline — how the barrier `NUMERIC` failure was diagnosed and finally solved

A chronological account of *what was tried, why it wasn't enough, and what finally
worked*. Run: `Custom_runs/LUF_S1_to_S4_trans_fix_2/Run_G0001`, RESFACTOR=5, 2020–2050.
Symptom: barrier (`Method=2`) returns `NUMERIC` (status 12); the dual blows up mid-solve,
then the internal dual-simplex fallback grinds 800 s+ and is killed.

#### Act I — The symptom and the first theory (constraint geometry)

The barrier died at year **2035** first. A constraint-RHS survey across years showed 2035
is where two things hit simultaneously for the first time: **GHG tightens ~49 %**
(26.76 → 13.52 MtCO₂e) and **total demand spikes +18.7 %** (AusTIMES multipliers ramp
2031→2035 but LUTO only evaluates 2030 and 2035, compressing a 5-year ramp into one solve
step). First theory: 2035 sits at the intersection of several simultaneously-binding
constraints — a **degenerate vertex**. Real, but only half the story: after code changes
the failure *moved to 2030*, which is numerically, not economically, hard.

#### Act II — The campaign against the numerics (steps 1–7)

Working hypothesis: **lock-in of irreversible land uses** (Environmental / Riparian
Plantings, ag-management) injects degenerate variables that make the barrier's normal
equations rank-deficient. Each period locks the previous year's dvars as lower bounds;
floating-point noise in those dvars creates near-fixed / zero-width variables. The attack,
on the data + variable-setup side:

1. Introduced a tolerance (`_tol = FEASIBILITY_TOLERANCE × 10 = 1e-5`) for "bounds too
   close".
2. Snapped non-ag UB sitting within `_tol` of LB.
3. Floor-truncated the non-ag LB so sub-precision digits can't become a phantom lower
   bound.
4. Rebuilt the non-ag UB from `t_rk`: if a cell had a previous non-ag dvar → UB = 1; if the
   previous dvar is below `_tol` (noise) → UB keeps the previous value; no `RP_j` UB may
   exceed `RP_PROPORTION`.
5. In the solver var-setup: skip a dvar whose LB or UB is below `_tol`; if `ub − lb < _tol`,
   set `ub = lb` (fixed variable).
6. Barrier **still** NUMERIC → added **RHS subtraction**: record `ub = lb` fixed
   contributions in `non_ag_fixed_kr` and subtract from the cell-usage equality RHS.
7. Barrier **still** NUMERIC.

**Why steps 1–7 weren't enough.** They correctly eliminated the *fixed-variable*
singularities (`0 fixed` confirmed), but step 5/6 created a subtler problem ("Crack 2"):
for cells where the fixed non-ag contributions sum to ≈ the cell mask, the cell-usage
equality became `Σ X_ag == mask − fixed ≈ 6e-8` (numerical dust) — a near-zero-RHS equality
forcing a whole row of positive-UB ag variables to ≈0. The log-barrier term pushes them
*away* from 0 while the equality forces them *to* 0 → many near-zero complementarity slacks
→ `AAᵀ` ill-conditioned → dual blows up. The blow-up merely *moved* (iter 91 → 96 → 93)
and the primal residual stalled at ~9.12e-3 before the dual exploded.

#### Act III — Nailing Crack 2 (the saturated cells)

A clean A/B harness (`step_3/`) overlaid only the patched `solver.py` on the frozen run
snapshot and rebuilt the real 2025→2030 solve. A saturation census found **148 "saturated"
cells** where `mask − fixed_r ≤ _tol` (RHS is dust), with 0 overfull, 0 leftover non-ag
vars, 0 positive non-reversible AM lower bounds — proving it **safe** to delete those
cells' variables and equalities. The fix reordered `_setup_vars()`, skipped ag/AM vars at
saturated cells (asserting no non-reversible AM lb was dropped), and dropped the empty
cell-usage equalities. **Effect: necessary but not sufficient** — the dual blow-up was
*delayed* (iter 91 → 102) and the primal floor improved (9.12e-3 → 3.26e-3), but the status
was **still NUMERIC**. The remaining cause was *not the model*.

#### Act IV — The (false) unlock: solver conditioning

With Crack-2 fixed, the barrier log showed the primal residual **stall** and the dual
diverge to −1e17, with Gurobi printing *"Consider using the homogeneous algorithm"* — the
signature of **ill-conditioning**, not degeneracy. The matrix had a coefficient range ratio
of ~1e8, yet the run configured the barrier hostilely: `SCALE_FLAG=0` (scaling off),
`BARHOMOGENOUS=0`, `FeasibilityTol=1e-6` and `BarConvTol=1e-5` (both far below the
achievable floor). A config sweep produced a "winning recipe": `ScaleFlag=1`,
`BarHomogeneous=1`, `BarConvTol=2e-2`, solver-only `FeasibilityTol=1e-3`, `Crossover=-1` —
barrier stops at its best ~1.6 % interior point, crossover polishes to an exact vertex.
This *did* drive the solve to OPTIMAL — but it was treating a symptom. (A subtle key here:
`FeasibilityTol` was decoupled into a new `SOLVER_FEASIBILITY_TOL` so relaxing the Gurobi
tolerance didn't coarsen the lock-in machinery that also keys off `FEASIBILITY_TOLERANCE`.)

#### Act V/VI — Chasing the *cause*: a real unit bug

Asking *why* the matrix is ill-conditioned, objective decomposition traced the `1e5`
(million-AUD) outlier entirely to **`non_ag_to_ag_t_mrj`** — the cost to convert plantings
back to agriculture — at 1.5e6–1.16e8 AUD/ha (normal ~1.5e3). **The bug:**
`get_env_plantings_to_ag` added the per-ha base (`$/ha`) and the water-license delta
(already `$/cell`) and multiplied the **sum** by `REAL_AREA`, double-counting area on the
water term by ~2500×. Fixing it dropped the worst objective coefficient ~1000× and the
objective range from `[6e-2, 1e5]` to `[7e-5, 5e2]` — a genuine economics-correctness fix.

#### Act VII — The reversal: the band-aids were never needed

Pushed to actually *isolate* the cause, a `DISABLE_BOUND_SNIPPING` switch re-ran 2030 at
the **original tight failing tolerances** while flipping one knob at a time. Almost
everything converged — including `ScaleFlag=0` and even **re-introducing the bug** — which
first looked like "the matrix is just healthy now." That reading was a trap: every "buggy"
row had **auto-scaling on** (`ScaleFlag=-1`), silently rescaling the bad objective. The
faithful reproduction of the original state is **buggy transition + scaling OFF**:

| Config (buggy non-ag→ag, snip off, tight tols) | Result | BarIter | Behaviour |
|---|---|---|---|
| `ScaleFlag=0` (scaling off — the original) | OPTIMAL | **357** (532 s) | primal residual **erupts to ~30**, near-timeout |
| `ScaleFlag=-1` (auto-scaling) | OPTIMAL | 153 (266 s) | clean |
| FIXED transition, `ScaleFlag=0` | OPTIMAL | 238 (379 s) | clean |

So the impossible non-ag→ag transition, with its `*REAL_AREA`-inflated cost, **is** the
destabiliser: with scaling off it more than doubles the iteration count and throws a huge
primal-residual excursion — the exact signature of the original failure. On 2030 it limps
to OPTIMAL; on the harder 2035 year the same instability tips into the observed hard
`NUMERIC`. **Two independent stabilisers** exist, and the frozen run had neither: (1) fix
the transition; or (2) turn Gurobi scaling on. The transition fix is therefore **both** an
economics-correctness fix **and** the real numeric fix.

#### Act VIII — The cleanup (snipping removed for good)

With the culprit pinned, all the defensive bound-snipping became dead weight. The solver
was reset to its clean pre-snipping form (`git checkout d84fbb77 -- luto/solvers/solver.py`)
and the relaxed band-aid settings reverted (`ScaleFlag=0`, `BarConvTol=1e-5`,
`FeasibilityTol=1e-6`, `BarHomogeneous=0`). The only survivors are the two transition fixes.
Re-validating 2025→2030 through the real production path: **barrier OPTIMAL in 124
iterations, obj 9.525755e3, matrix range `[1e-04, 9e+01]`, primal residual falling
monotonically 8.18e8 → 4.31e-7 with no excursion**; crossover landed cleanly in 3 s.
Committed as `a684ec3f`.

#### Production confirmation — the fix makes near-singularities *recoverable*, not absent

A live production 2030 solve (`LUF_S1_to_S4_trans_fix_2/Run_G0001`, retry params
`NumericFocus=0, Method=2, Crossover=-1, **Presolve=0**`, matrix range `[1e-04, 9e+01]`)
shows the fix's true mechanism even more clearly than the clean test:

```
iter 127   7.00471e+03   7.00744e+03   primal 1.42e-3   compl 8.4e-7   (≈ converged)
iter 128   7.00471e+03  -3.99530e+06   ← dual erupts
iter 129   7.00471e+03  -1.50872e+11   ← dual peak
iter 130  -7.53755e+10   6.28734e+09   primal 2.83e+09  ← primal residual spikes
…         (recovers)                   primal 2.83e9 → 2.46e0 by iter 197
Barrier solved model in 331 iterations …   Optimal objective 6.96482202e+03
```

The barrier hits a near-singular point at iter 127 and the dual momentarily explodes to
`−1.5e11` — **yet it recovers** and re-converges to OPTIMAL. Two points:

- The oscillation appears here (but not in the 124-iter clean test) because the production
  **retry uses `Presolve=0`**, removing the presolve row-reduction/conditioning that the
  clean test got from `Presolve=auto`. With presolve off, the barrier meets the raw matrix
  and transiently destabilises.
- It **survives** only because the transition fix made the matrix well-conditioned: the
  dual blow-up is now a *recoverable transient*. Without the fix, the same eruption is
  *fatal* — the `*REAL_AREA`-inflated impossible non-ag→ag cost keeps `AAᵀ`
  ill-conditioned, so the normal equations never recover (the original NUMERIC; the
  buggy-transition + scaling-off config shows the analogous primal-residual eruption
  limping 357 iters at 2030, tipping into hard failure on harder years like 2035).

**Refined statement:** the transition fix does not eliminate transient near-singularities
(especially under `Presolve=0`) — it makes them **recoverable instead of fatal**.

#### Detour ledger — why each earlier "fix" was abandoned

| Attempt | Why it was tried | Why it was dropped |
|---|---|---|
| Bound snipping (steps 1–6) | Kill fixed-variable / lock-in-noise singularities | Created the Crack-2 `6e-8` dust-RHS; never removed the dual blow-up |
| Crack-2 saturated-cell deletion (Act III) | Remove the dust-RHS equalities | Only *delayed* the blow-up; status still NUMERIC |
| `ScaleFlag=1` + `BarHomogeneous=1` + relaxed tols + crossover (Act IV) | Make the ill-conditioned matrix solvable | Treated the symptom; auto-scaling was hiding the real culprit |
| `SOLVER_FEASIBILITY_TOL` decoupling (Act IV) | Relax Gurobi tol without coarsening lock-in | Unnecessary once the transition bug was fixed |
| Tolerance relaxation (`BarConvTol=2e-2`, `FeasibilityTol=1e-3`) | Stop barrier before divergence | Unnecessary; barrier reaches full tight tolerances after the real fix |

The only two changes that survived are the transition-cost fixes documented at the top of
this entry.

#### Reproduce / inspect

Harness pattern: `sys.path.insert(0, <repo>)`, `joblib.load(".../data_2025.lz4")`,
`get_input_data(data, 2025, 2030)`, `LutoSolver(...).formulate()`, `.solve()`.

---

## 20260605 — lb > ub bug, unified snap rule, and lock-in confirmed as barrier root cause

> **⚠️ Superseded by the 20260606 entry above.** The "structural / degenerate-vertex" and
> "non-ag lock-in is the root cause" conclusions below were the best available reading at
> the time, but the true single culprit was the non-ag→ag transition cost bug. With that
> fixed, the barrier solves to OPTIMAL at full tolerances with no snipping and the lock-in
> LBs left intact. The analysis below is retained for historical context.

### Context

Continuation of the 20260605 crack analysis. The demand-only 2020→2025 reconstruction
from `data_2030.lz4` was used to isolate barrier failure to its true root cause.
Three questions were investigated: (1) whether a lb > ub bug exists, (2) whether
tightening the snap rule to a 1% relative threshold fixes barrier, and (3) whether
the non-ag lock-in LBs from the checkpoint are the true root cause.

---

### 1 — Confirmed lb > ub bug for 1 RP cell

A lb/ub validity check found 1 Riparian Plantings cell where `lb > ub` after the
`d84fbb77` change that removed the UB cap from `get_non_ag_lb_matrices`.

**Mechanism:**

```
FEASIBILITY_TOLERANCE = 0.01 → Gurobi can return dvar up to RP_PROPORTION + 0.01
Stored 2020 RP dvar     = RP_PROPORTION + 0.002202  (within tolerance)
Floor-truncated lb       = 0.332202
UB (RP_PROPORTION)       = 0.330000
lb > ub by               = 0.002201
```

The snap threshold of `abs(ub−lb) < 1e-5` cannot catch a 0.002 gap → Gurobi
receives lb > ub for this variable.

**Before `d84fbb77`**: the LB function capped `lb = min(lb_rk, non_ag_x_rk)`.
Since `non_ag_x_rk[RP] = RP_PROPORTION`, lb was always clamped to ≤ RP_PROPORTION.
After `d84fbb77` removed this cap, tolerance-violated dvars produce lb > ub.

---

### 2 — Additional 30 near-capacity cells missed by abs-only snap

Beyond the lb > ub cell, the check found:
- **29 EP cells**: lb ≈ 0.991–0.999, ub = 1.0, gap = 0.001–0.009
- **1 RP cell**: lb ≈ RP_PROPORTION × 0.998, gap ≈ 0.0005

The old relative threshold `(ub−lb)/lb < 0.01` would have snapped these.
The abs-only `< 1e-5` threshold missed them. These near-UB variables have
complementarity slack `(ub−x) → 0` at the optimum — same near-singular AA' effect
as near-LB variables.

---

### 3 — Fix: unified relative snap rule + lb > ub guard

Applied to `solver.py` `_setup_non_ag_vars` and `_setup_ag_management_variables`:

**Non-ag variables** (two guards):
```python
# Guard 1: prior dvar can exceed RP_PROPORTION within FEASIBILITY_TOLERANCE
if x_lb > x_ub:
    x_lb = x_ub
# Guard 2: collapse near-degenerate windows (1% relative threshold)
elif x_lb > 0 and (x_ub - x_lb) / x_lb < 0.01:
    x_lb = x_ub
```

**AM variables** (renewable and generic paths):
```python
model_ub = model_lb if (model_lb > 0 and abs(1.0 - model_lb) / model_lb < 0.01) else 1.0
```

`abs()` is added for AM because ub is always 1.0 (no Guard 1 clamp); without abs,
a hypothetical lb > 1.0 from float arithmetic would pass the `< 0.01` check silently.
For non-ag Guard 2, `abs()` is not needed because Guard 1 guarantees x_lb ≤ x_ub.

**Effect:** 1 lb > ub cell clamped; 30 additional near-capacity cells snapped.
Total snapped variables increased by 31 vs. the abs-only rule.

---

### 4 — 1% snap rule does NOT rescue barrier

Demand-only model (Crossover=0, with 1% snap):

```
BarIter=42, Status=NUMERIC, Time=79s
iter 33: dual = 4.06e+07   (converging)
iter 34: dual = −6.30e+10  (begins diverging)
iter 35: dual = −8.71e+17  (blow-up)
iter 42: dual = −2.52e+39  (barrier gives up)
```

vs. before 1% snap fix (also NUMERIC, BarIter=40, ~69s). The dual blow-up pattern
is identical — 2 extra iterations and ~10s longer, but no fundamental change.

**Conclusion**: the 30 additional snapped cells reduced AA' near-singularity slightly
but did not address the true cause. The snap fix is still correct (lb > ub bug fixed,
30 more degenerate variables removed) but is insufficient alone.

---

### 5 — Three-way controlled experiment: isolating the root cause

Three versions of the 2020→2025 demand-only model were run with Crossover=0:

| Test | non-ag LB source | Nonzero LB cells | Barrier result |
|------|-----------------|-----------------|----------------|
| Current code (1% snap) | full lock-in from `non_ag_dvars[2020]` | ~5,855 + RP cracks | **NUMERIC** 42 iters, 79s |
| Pre-94971ea1 exact eviction | t_rk only, lb capped against old UB | 5,855 | **NUMERIC** 44 iters, 77s |
| Lock-in removed (dvars removed entirely) | zeros | 0 | **OPTIMAL** 213 iters, 405s |

**Pre-94971ea1 eviction counts** (t_rk only, lb capped against old UB):

| LU | Evicted (lb→0) | Kept (lb>0) |
|----|---------------|------------|
| Environmental Plantings | 24 | 1,906 |
| Riparian Plantings | 456 | 3,812 |
| Sheep Agroforestry | 1 | 22 |
| Beef/other Agroforestry | 0 | 14 |
| Destocked | 0 | 101 |
| **Total** | **481** | **5,855** |

The pre-fix code only evicted RP cells whose dominant `lumap` was EP
(`T_MAT[EP→RP] = NaN`). Cells where `lumap` allowed RP (beef/sheep
pasture: `T_MAT[Beef→RP] ≠ NaN`) still retained their LBs. Similarly,
EP cells where `lumap=EP` (self-transition valid) kept their LBs.

**Barrier failure is not specifically caused by the 94971ea1/d84fbb77 fixes.**
Even the exact pre-fix state — with 481 cells properly evicted and 5,855
LBs remaining — still hits the same dual blow-up at iter 33–34. The
current-code model adds only ~481 more nonzero-LB cells; both fail.

**The reconstruction scenario itself is the root cause.** The data_2030.lz4
checkpoint has `non_ag_dvars[2020]` populated from a run that solved with
`GBF2_TARGET=high`, creating ~5,855 non-ag cells with LBs > 0. At the
optimum, barrier must simultaneously drive `(x − lb) → 0` for all these.
With rescaled constraint coefficients at O(1e³) and `BarConvTol=1e-5`,
the KKT complementarity requires `x − lb < 1e-8` — approaching machine
precision for thousands of variables at once → normal-equation AA' near-singular
→ dual blow-up at iter 33–34.

**Why the original run worked:** the real LUF_S1_to_S4_raw_trans_fix run
solved 2020→2025 as its FIRST step. At that point `non_ag_dvars.get(2020) = None`
(dvars not yet written). All non-ag LBs = 0 → no near-LB variables →
barrier converged to OPTIMAL in 213 iterations (405s).

**The reconstruction is a strictly harder problem** than the original first
solve. It uses real 2020 dvars as LBs. Barrier fails on it regardless of
which code version (pre-fix or post-fix) is used for the LB/UB computation,
and regardless of snap tuning.

---

### 6 — Summary of code changes

| File | Change |
|------|--------|
| `solvers/solver.py` `_setup_non_ag_vars` | Guard 1: clamp `x_lb = x_ub` when `lb > ub` (fixes tolerance-violation bug). Guard 2: unified relative snap `(x_ub−x_lb)/x_lb < 0.01` replacing abs-only threshold |
| `solvers/solver.py` `_setup_ag_management_variables` (renewable path) | Snap changed to `abs(1.0−model_lb)/model_lb < 0.01` |
| `solvers/solver.py` `_setup_ag_management_variables` (generic path) | Same: `abs(1.0−dry_x_lb)/dry_x_lb < 0.01` and `abs(1.0−irr_x_lb)/irr_x_lb < 0.01` |

---

### 7 — Implications

The barrier failure for reconstructions from checkpoint is structural — it arises
from the correct lock-in fix (94971ea1 + d84fbb77) creating near-LB variables that
the barrier's KKT system cannot handle with Presolve=0.

**Mitigation options (in priority order):**

1. **Accept dual simplex (Method=1) for reconstructions.** RETRY_PARAMS second
   entry `(0, 1, 0, -1)` handles this correctly. Dual simplex operates on vertices
   and uses anti-degeneracy pivot rules that step through degenerate faces without
   requiring complementarity slacks to stay positive.

2. **Presolve=1 for barrier** would shift variables `y = x − lb` internally,
   eliminating the near-zero complementarity slack issue. Avoided because
   presolve-transformed false infeasibility was an earlier problem — revisit
   with current code if dual simplex proves too slow.

3. **Softening the lb for non-ag in the solver** (not in the data) by a small
   epsilon to give barrier interior room — principled but complicates solution
   extraction.

---

## 20260605 — Variable-bound crack analysis, am-crack snap fix, and barrier degeneracy root cause

### Context

`LUF_S1_to_S4_raw_trans_fix` (RESFACTOR=5, 2020–2050, renewables ON) was used to
investigate whether near-degenerate variable bounds ("cracks") were the primary cause of
barrier (Method=2) failing at years 2025 and 2030. A systematic crack analysis was
performed from the `data_2030.lz4` checkpoint; a new snap fix was applied to the
ag-management variable setup; and the true root cause of barrier failure was identified.

---

### 1 — What a "crack" is

After each solved year, dvars are floor-truncated to `ROUND_DECIMALS=6` decimal places
and stored as the next year's lower bound (LB) for irreversible land uses.  The upper
bound (UB) is derived separately (transition matrix for non-ag; always 1.0 for am).

When the previous dvar is very close to the UB (e.g. a nearly fully-allocated cell):

```
am_dvar  ≈ 0.9999997  →  am_lb = floor(0.9999997 × 1e6) / 1e6 = 0.999999
UB = 1.0
crack width = 1 - 0.999999 = 1e-6
```

Barrier must keep every variable strictly inside `(LB, UB)`.  When the window is
`O(1e-6)` wide, the complementarity slack `(UB − x)` is near-zero at the optimum, making
the normal-equation matrix `AA'` near-singular.

---

### 2 — Confirmed: saved dvars are raw Gurobi values (not pre-clamped)

`data.ag_man_dvars[yr]` stores the raw floor-truncated Gurobi output.  The
`min(am_dvar, ag_dvar)` clamping that prevents am from exceeding its host ag variable
happens only at LB-derivation time inside
`get_lower_bound_agricultural_management_matrices()`, not at save time.

Evidence: after the 2020 barrier solve, 1,346 cells per AM have `am_dvar > ag_dvar` by up
to `9.14e-3` (≤ `FEASIBILITY_TOLERANCE`).  After the 2025/2030 dual-simplex solves,
fewer cells but larger mean gaps (up to `2.18e-3`).

---

### 3 — Non-ag crack distribution: existing fix handles everything

After floor-truncation, 61–73% of active non-ag cells (those with LB > 0) have raw crack
width < `1e-5`.  The existing snap rule in `_setup_non_ag_vars`:

```python
if x_lb > 0 and (x_ub - x_lb) / x_lb < 0.01:
    x_ub = x_lb
```

eliminated **100% of non-ag cracks** in all years — zero remaining after the fix.  The
gap distribution shows a hard jump from LB ∈ [0.99, 0.999] to LB ≥ 0.99999 with nothing
in between; the 0.01 relative threshold correctly caught all legitimate near-fixed cells.

Section 3 of the analysis confirms 94–96% of active non-ag cells return exactly at their
LB after solving — both barrier and dual simplex pin them there.

---

### 4 — Ag-management crack cascade: the hidden multiplier effect

In this run Solar PV and Onshore Wind are **enabled and non-reversible**.  Their LBs come
from `min(am_dvar, ag_dvar)` floor-truncated.  No equivalent snap fix existed.

Crack counts from the `data_2030.lz4` checkpoint:

| base_year → target | Solar PV active/tiny | Wind active/tiny |
|---|---|---|
| 2020 → 2025 (barrier) | 2 / 0 | 6 / 3 |
| 2025 → 2030 (simplex) | 3 / 0 | 29 / 20 |
| 2030 → 2035 (simplex) | 29 / 16 | 256 / 164 |

The raw cell counts look small, but each crack cell cascades through two constraint hops:

1. `X_am[j,r] ≥ LB ≈ 1` and `X_am[j,r] ≤ X_ag[j,r]` forces `X_ag[j,r] ≈ 1`
   (near its own UB).
2. Cell-usage equality `Σ_j X_ag[m,j,r] + Σ_k X_non_ag[r,k] = AG_MASK` forces all
   **other** variables at cell `r` toward 0 (near their LBs).

Cascade footprint for the 2035 solve (base_year=2030):

| Measure | Count |
|---|---|
| Am crack variables (direct) | 106 |
| Paired ag vars forced near UB (`am ≤ ag`) | 106 |
| Unique crack cells | 106 |
| Other vars forced → 0 at those cells (cell-usage) | 1,306 |
| **Total near-degenerate variables** | **1,518** |
| Global-constraint `AA'` hits (× 29 constraints) | **44,022** |

None of these 1,518 variables had any snap fix — they remained as tiny non-zero windows
in the interior-point complementarity system.

---

### 5 — Fix: crack snap for non-reversible ag-management variables

Applied the same snap logic to **both** code paths in `_setup_ag_management_variables`
(renewable path and generic non-reversible path):

```python
# renewable (Solar PV / Wind):
model_ub = model_lb if (model_lb > 0 and abs(1.0 - model_lb) < 10 ** (1 - settings.ROUND_DECIMALS)) else 1.0

# generic non-reversible AMs:
dry_x_ub = dry_x_lb if (dry_x_lb > 0 and abs(1.0 - dry_x_lb) < 10 ** (1 - settings.ROUND_DECIMALS)) else 1.0
irr_x_ub = irr_x_lb if (irr_x_lb > 0 and abs(1.0 - irr_x_lb) < 10 ** (1 - settings.ROUND_DECIMALS)) else 1.0
```

**Threshold choice:** `10^(1 − ROUND_DECIMALS) = 1e-5` (10× the floor-truncation unit).
This is an absolute crack-width test: snap only when the window is pure floating-point
noise from truncation.  The am UB is always exactly `1.0` (Python float), so the absolute
form `abs(1.0 − lb)` directly measures the crack width without normalisation.

**Why not `0.01` (the old non-ag threshold):** The non-ag UB comes from `RP_PROPORTION`
(float32 arithmetic), which can produce gaps up to `~1e-2` for legitimate physical reasons.
Am UB is always exactly `1.0`; its only source of crack is floor truncation (max `1e-6`).
Using `1e-5` preserves real variable freedom (e.g. a cell with LB = 0.99 that could still
grow to 1.0) while snapping pure noise.

The non-ag snap rule was also updated to the same absolute form for consistency:

```python
if x_lb > 0 and abs(x_ub - x_lb) < 10 ** (1 - settings.ROUND_DECIMALS):
    x_ub = x_lb
```

---

### 6 — Reconstruction test: crack fix alone does not rescue barrier

A reconstruction test rebuilt the 2025 model from `data_2030.lz4` (base_year=2020)
and ran only the barrier attempt (`Method=2, Crossover=-1, Presolve=0`).  Barrier still
reported **NUMERIC** after the snap fix.

The barrier log shows a **dual blow-up** at iteration 21 (dual jumps from `−1.94e+16` to
`−4.57e+24`) — not a complementarity-slack near-zero failure.  This is an ill-conditioned
normal-equation matrix from constraint-level degeneracy, not variable-bound degeneracy.

---

### 7 — Root cause: the optimal solution lives at a degenerate vertex

The crack snap correctly removes fixed-variable noise from `AA'`.  But the optimal
solution is structurally degenerate:

| Degeneracy source | Snap fix removes it? | Barrier handles it? | Dual simplex handles it? |
|---|---|---|---|
| Fixed locked-in variables (crack LB ≈ UB) | ✓ yes | ✗ needs interior | ✓ trivially |
| Multiple simultaneously binding constraints (GHG + demand) | ✗ no | ✗ dual blow-up | ✓ degenerate pivots |

At the 2025 optimum, the GHG constraint and several demand constraints bind
simultaneously.  Barrier's central path requires all constraint slacks strictly positive;
as `µ → 0` these slacks → 0 simultaneously, making `AA'` near-singular regardless of
variable bounds.

Dual simplex operates on vertices and edges directly.  It uses anti-degeneracy pivot rules
(perturbation, Bland's rule) that step along the degenerate face without requiring slacks
to stay positive — exactly what this problem geometry demands.

---

### 8 — Anti-degeneracy techniques to help barrier

Three approaches can split the degenerate vertex and let barrier succeed:

1. **`BarHomogeneous=1`** — Gurobi's homogeneous self-dual embedding, designed for
   near-degenerate problems.  Zero code change; already exposed as `settings.BARHOMOGENOUS`.
   *Note*: this setting previously caused false infeasibility certificates via tau-drift in
   some runs (see 20260603 entry) — use with caution.

2. **Objective noise** — add `ε × U(0,1)` to each coefficient in `ag_obj_mrj`.  Breaks
   symmetry between multiple co-optimal vertices; barrier finds one non-degenerate
   solution.  Use the solution only as a warm-start basis, then re-solve with true
   objective.

3. **RHS relaxation + warm-start** — temporarily relax binding constraint RHSs by `~0.1%`,
   solve with barrier (succeeds on the perturbed non-degenerate problem), re-tighten and
   warm-start dual simplex.  Most principled; best for hard constraints.

---

### 9 — Summary of code changes

| File | Change |
|---|---|
| `solvers/solver.py` `_setup_non_ag_vars` | Snap threshold changed from `(x_ub-x_lb)/x_lb < 0.01` to `abs(x_ub-x_lb) < 10^(1-ROUND_DECIMALS)` |
| `solvers/solver.py` `_setup_ag_management_variables` (renewable path) | Add `model_ub = model_lb if abs(1-model_lb) < 1e-5 else 1.0` for non-reversible AMs |
| `solvers/solver.py` `_setup_ag_management_variables` (generic path) | Add same snap for `dry_x_ub` and `irr_x_ub` |

---

## 20260604 — Non-ag UB/LB regression fix, ag-man lb simplification, thin-window collapse rule, and barrier singularity investigation

### Context

`REM_RES5_trans_fix` runs (6 runs, RESFACTOR=5) were used to investigate whether fixes to
the non-ag UB/LB logic and agricultural management lower bounds could resolve the barrier
NUMERIC trouble that appears consistently from year 2030 onwards. The investigation
identified a regression introduced by commit `1339cf00`, a root cause for the barrier
singularity (RP thin-window variables), and a partial fix. Dual simplex remains necessary
for the structural residual singularity.

---

### 1 — Regression in commit `1339cf00`: non-ag monotonicity violated in WITH-fix runs

`LUF_S1_to_S4_trans_fix` vs `LUF_S1_to_S4` (8 runs, RESFACTOR=5, 2020–2050) was used
to verify the `1339cf00` fix. Non-ag land uses must be monotonically non-decreasing
(irreversible), but the WITH-fix runs showed new violations:

| Run | WITHOUT fix | WITH fix |
|-----|-------------|----------|
| G0002 | 0.003 ha (noise) | OK ✓ |
| G0003 | 0.02 ha (noise) | **21.98 ha** ← new |
| G0004 | 0.23 ha | OK ✓ |
| G0006 | OK | **22.00 ha** ← new |
| G0007 | OK | **22.00 ha** ← new |
| G0008 | 86.91 ha | MISSING |

The ~22 ha violations in G0003/G0006/G0007 are identical across runs (same cells), and
show a dip–recovery pattern: area drops 22 ha in year Y, then recovers in year Y+1. This
is the signature of a cell being released from `non_ag_lu2cells` in one period then
re-allocated the next.

**Root cause:** `1339cf00` removed `existing_dvars_rk` from `get_to_non_ag_exclude_matrices`
(renamed `get_non_ag_ub_matrices` in this session). Without the override, a T_MAT block in
year Y gives `ub=0` for a cell with existing irreversible allocation. Even though
`non_ag_lu2cells` uses `max(ub, lb)` to include such cells, if the cell's prior dvar was 0
(it was excluded in a prior step), `lb=0` and `effective_ub=0` → cell falls out entirely.
The dip–recovery occurs because the cell gets blocked and re-opened as the lumap cycles.

**Fix:** restore `existing_dvars_rk` parameter to `get_non_ag_ub_matrices` (UB function
only). The LB function (`get_non_ag_lb_matrices`) correctly has no UB knowledge — that
separation from `1339cf00` is kept.

---

### 2 — Why the `existing_dvars_rk` override must stay in the UB function

For a T_MAT-blocked cell with an existing irreversible allocation:

- Without override: `non_ag_ub_rk[r,k] = 0` → `effective_ub = max(0, lb)`. If lb=0 from a
  prior exclusion, the cell never enters `non_ag_lu2cells`. No variable → implicit 0 → area drops.
- With override: `non_ag_ub_rk[r,k] = 1` (for non-RP) → cell always in `non_ag_lu2cells` →
  variable created with `lb=existing_dvar, ub=1` → area locked.

The UB function's job is "what's the maximum?" — for existing irreversible cells the answer
is 1 (they're allowed to stay and grow), not 0 (transition matrix gate is for new allocations only).

---

### 3 — Ag management LB simplification

The old `get_lower_bound_agricultural_management_matrices` logic had three stacked caps:
per-cell area cap, sum-overflow scaling, and `min(am_lb, ag_lb)` capping. This was replaced
by a single `am_dvar_true = min(am_dvar, ag_dvar)` clamp before floor-truncation.

**Why it didn't change the model:** violations were only `~6e-8`, smaller than the floor
precision of `1/10^ROUND_DECIMALS = 1e-6`. After `floor(x × 1e6)/1e6`, both old and new
produce identical values → same model fingerprint (`0x57918376`). The simplification is
correct and cleaner but doesn't affect numerical behaviour.

---

### 4 — Barrier singularity: RP thin-window variables as root cause

The non-ag UB/LB fix (restoring `existing_dvars_rk`) changed the model fingerprint
`0x57918376 → 0x35ade78a` and dramatically altered the barrier trajectory — 889 RP cells
changed from fixed variables (`lb=ub=existing_value`) to free variables (`lb=existing_value,
ub=1`). These freed variables shortened the central path, causing the barrier to converge
faster but hit the near-singular zone earlier (iter 53 vs iter 132).

Tracing the 889 cells: for T_MAT-blocked RP cells at RESFACTOR=5, the existing dvar is the
previous period's RP allocation, which fills almost all of `RP_PROPORTION`. After the override
sets `t_rk=1`, the RP cap `no_go_x_rk[:, RP_j] *= RP_PROPORTION` applies, giving:

```
ub = RP_PROPORTION        (e.g. 0.3300001, float32)
lb = floor(dvar × 1e6) / 1e6   (e.g. 0.330000)
window width ≈ 1e-7
```

For 889 such cells, the barrier's complementarity condition `x × s = μ` (s = ub − x) has
889 near-zero slacks simultaneously at the optimum. These appear as 889 near-zero diagonal
entries in the normal equation matrix `AA'`, causing Cholesky factorisation to blow up.

**The proof:** `get_non_ag_ub_matrices` was called with and without `existing_dvars_rk`
to count exactly which cells differ:

```
Environmental Plantings:  31 cells changed (UB: 0 → 1)
Riparian Plantings:      889 cells changed (UB: 0 → RP_PROPORTION)
```

EP cells go from `ub=0` to `ub=1` — a wide window, no singularity.
RP cells go from `ub=0` (fixed at lb) to `ub=RP_PROPORTION ≈ lb` — near-degenerate.

---

### 5 — Thin-window collapse rule

**Fix:** in `_setup_non_ag_vars` (solver.py), collapse near-degenerate windows before
passing to Gurobi:

```python
if x_lb > 0 and (x_ub - x_lb) / x_lb < 0.01:
    x_ub = x_lb
```

When `lb = ub`, Gurobi treats the variable as **fixed** and removes it from the barrier's
interior-point complementarity system entirely — no near-zero slack, no singularity.

**Effect:** barrier fingerprint changed again (`0x35ade78a → 0x23226366`), confirming the
889 RP variables were removed from `AA'`. The barrier passed iter 53 (previous blow-up
point) and continued to iter 92 before hitting a second structural singularity:

| Run | Fingerprint | Blow-up iter | Time |
|-----|------------|--------------|------|
| No UB/LB fix | `0x57918376` | 132 | 549s |
| UB/LB fix, no collapse | `0x35ade78a` | 53 | 183s |
| UB/LB fix + collapse | `0x23226366` | 92 | 350s |

The second singularity at iter 92 has the same signature (primal locked at `~−5.46e+04`,
dual explodes in one step) and is structural — driven by the renewable energy state-level
constraints + GHG + GBF2 simultaneously binding at the optimal vertex. Dual simplex
fallback is still required.

---

### 6 — Renaming for clarity

| Old name | New name |
|----------|----------|
| `get_to_non_ag_exclude_matrices` | `get_non_ag_ub_matrices` |
| `get_lower_bound_non_agricultural_matrices` | `get_non_ag_lb_matrices` |
| `non_ag_x_rk` (field + function) | `non_ag_ub_rk` / `get_non_ag_ub_rk` |

---

### 7 — Summary of code changes

| File | Change |
|------|--------|
| `economics/agricultural/transitions.py` | Simplify ag-man lb: `min(am_dvar, ag_dvar)` clamp before floor |
| `economics/non_agricultural/transitions.py` | Restore `existing_dvars_rk` to `get_non_ag_ub_matrices`; rename both functions; update docstrings |
| `solvers/input_data.py` | Restore `existing_dvars_rk` pass in `get_non_ag_ub_rk`; rename field/function; simplify `non_ag_lu2cells` back to `np.where(non_ag_ub_rk[:, k])` |
| `solvers/solver.py` | Add thin-window collapse rule; remove dead `max(ub, lb)` and `float()` wrapper; remove dead `max(x_rk, lb)` UB-lift |
| `tools/write.py` | Update call site to `get_non_ag_ub_matrices` |

---

## 20260603 — Barrier false infeasibility in LUF_S1_to_S4: BarHomogeneous tau-drift, lb/constraint scaling mismatch, and transition-matrix PR refactor

### Context

Runs at `Custom_runs/LUF_S1_to_S4/` (8 runs, RES5, 2020–2050) showed barrier (Method=2)
falling back to dual simplex (Method=1) on multiple years starting from ~2030. The
barrier reported status 3 (INFEASIBLE) or 12 (NUMERIC) while dual simplex always found
an OPTIMAL solution. All false infeasibilities appeared after PR `94971ea` (the non-ag
lock-in fix). A chain of four root causes was identified and addressed.

---

### 1 — False infeasibility pattern in logs

```
Non-optimal status 3 with NumericFocus=0, Method=2  ← 12 occurrences
Non-optimal status 12 with NumericFocus=0, Method=2 ←  1 occurrence
Non-optimal status 13 with NumericFocus=0, Method=2 ←  2 occurrences
```

For status 3 (the dominant case), the Gurobi log shows dual residuals **diverging** in
the final barrier iterations — not converging:

```
iter 235: dual residual  2.92e+02,  compl  2.18e+01
iter 236: dual residual  7.47e+02,  compl  4.14e+01
iter 237: dual residual  2.82e+03,  compl  1.97e+02
iter 238: dual residual  1.70e+04,  compl  1.37e+03
→ "Infeasible model"
```

IIS computation on the same model confirmed the problem is **feasible** — no infeasible
subsystem exists. NumericFocus=3 (Gurobi's maximum arithmetic precision) did not prevent
the false infeasibility, ruling out a floating-point precision explanation.

---

### 2 — Root cause A: BarHomogeneous=1 generates false infeasibility certificates

`BarHomogeneous=1` forces Gurobi's homogeneous self-dual embedding. This algorithm tracks
an extra variable τ (tau). When τ → 0 while the dual objective diverges, the algorithm
declares INFEASIBLE and emits a dual ray certificate. On highly degenerate LP problems
(many simultaneously binding constraints), τ can drift toward zero along a numerical
artefact direction — not a real dual ray. The homogeneous algorithm then falsely certifies
infeasibility.

**NumericFocus=3 does not fix this** because the tau drift is a tracking-logic failure,
not an arithmetic precision failure. The two are orthogonal mechanisms.

**IIS confirming feasibility** is the proof: `computeIIS()` runs on the same model object
and finds no infeasible subsystem, which means the dual ray the homogeneous algorithm
produced does not actually exist.

**Fix**: set `BARHOMOGENOUS = 0` in settings.py. With the standard (non-homogeneous)
barrier, the algorithm has no tau-tracking path. It can only exit with NUMERIC (status 12)
or SUBOPTIMAL (status 13) when it struggles — both of which correctly describe the
situation and are handled by the dual simplex fallback. The standard barrier may also
converge successfully on cases the homogeneous algorithm was steering away from via the
spurious tau direction.

**Note on BarConvTol**: tightening from `1e-5` to `1e-8` does not help and would make
things worse. `BarConvTol` controls when the barrier stops for *optimality* (the
complementarity gap threshold). For status 3, the barrier is not trying to converge — it
is actively diverging and declaring infeasibility via the tau path. Tighter `BarConvTol`
would require the barrier to iterate more deeply into the ill-conditioned region,
increasing divergence probability.

---

### 3 — Root cause B: lb_rk / constraint-matrix scaling mismatch

The constraint matrix coefficients (`non_ag_obj_rk`, `non_ag_b_rk`, `non_ag_g_rk`, etc.)
are rescaled to ~1e3 magnitude by `rescale_solver_input_data()`. The non-ag lower bounds
(`non_ag_lb_rk`) are raw dvar proportions (0–1) passed directly to Gurobi as variable
bounds — **not rescaled**.

The barrier's KKT complementarity condition for a lower-bounded variable is:

```
dual_lb × (x − lb) = μ  →  must converge to 0
```

`dual_lb` balances the gradient, which includes the rescaled constraint coefficients:

```
dual_lb ≈ rescaled_obj_coeff + constraint_dual × rescaled_constraint_coeff  ≈ O(1e3)
```

For the barrier to satisfy the normalised convergence criterion (`BarConvTol = 1e-5`):

```
sum(dual_lb × (x − lb)) / |objective|  <  1e-5
≈ (1e3 × (x − lb)) / 1e6              <  1e-5
→  (x − lb)                           <  1e-8
```

With ~1000 locked-in irreversible non-ag cells (post PR `94971ea`), the barrier must
simultaneously drive ~1000 variables to within `1e-8` of their lower bounds — approaching
machine precision. This creates a near-singular Newton system. The result is the observed
dual residual explosion and false infeasibility.

**Dual simplex is immune**: it evaluates exact pivot operations at vertices. When a
variable is at its lower bound it is exactly there, with no precision requirement.

**The structural fix** would be a change-of-variables `y = x − lb` (shifting the lower
bound to zero) before building the Gurobi model. This is what Gurobi's presolve does
internally. The current code uses `Presolve=0` for barrier to avoid a separate class of
numerical errors (presolve-transformed false infeasibility that predated this PR), so
this transformation does not happen automatically. No code change was made for this issue;
the dual simplex fallback is the correct mitigation.

---

### 4 — Root cause C: PR `94971ea` was conceptually wrong (UB/LB separation violated)

PR `94971ea` fixed the RP eviction bug by injecting existing-dvar logic into
`get_to_non_ag_exclude_matrices()` — a UB-only function — to prevent the downstream
`min(lb, UB)` cap from zeroing the LB. This conflated two separate concerns:

- **UB** (`get_to_non_ag_exclude_matrices`): "is this cell allowed to take on this non-ag LU?" — pure transition matrix + no-go exclusions
- **LB** (`get_lower_bound_non_agricultural_matrices`): "what must be kept?" — pure lock-in from previous-period dvars

The original PR fix also inflated UB to 1 for existing irreversible cells, when the
correct UB for those cells is `max(t_rk_from_transition, lb)` = `lb` (locked at existing
value, since the transition matrix says no new allocation is allowed).

**Refactored fix** (applied in this session):

| File | Change |
|---|---|
| `transitions.py` `get_to_non_ag_exclude_matrices` | Removed `existing_dvars_rk` parameter — pure transition matrix UB |
| `transitions.py` `get_lower_bound_non_agricultural_matrices` | Removed UB cap against `non_ag_x_rk`; LB is only capped against `AG_MASK_PROPORTION_R` |
| `input_data.py` `get_non_ag_x_rk` | Reverted to original — no existing dvars |
| `solver.py` `_setup_non_ag_variables` | `x_ub = max(non_ag_x_rk[r,k], x_lb)` to reconcile UB=0 / lb>0 at variable creation |

---

### 5 — Critical bug in the refactored fix: `non_ag_lu2cells` excluded lock-in cells

`non_ag_lu2cells` is built from `np.where(non_ag_x_rk[:, k] > 0)` — only cells where UB>0
are included. For RP cells where the transition matrix gives UB=0 (the exact cells the fix
targets), `non_ag_lu2cells[k]` never includes them. The variable creation loop in
`solver.py` never reached those cells, so the `max(ub, lb)` line was dead code for the
problem case. RP was still silently evicted.

**Fix**: `non_ag_lu2cells` now uses `max(non_ag_x_rk, non_ag_lb_rk)` as the effective UB:

```python
@cached_property
def non_ag_lu2cells(self) -> dict[int, np.ndarray]:
    effective_ub = np.maximum(self.non_ag_x_rk, self.non_ag_lb_rk)
    return {k: np.where(effective_ub[:, k])[0] for k in range(self.n_non_ag_lus)}
```

For RP cells: `effective_ub = max(0, 0.2) = 0.2` → cell included → variable created with
`lb=0.2, ub=max(0, 0.2)=0.2` (fixed at existing value, cannot be evicted).

Minor side-effect: for reversible LUs with non-zero previous dvars but blocked transitions
(`non_ag_x_rk=0`), `non_ag_lb_rk > 0` causes those cells to enter `lu_cells`. The solver
sets `x_lb=0` for reversible LUs, so `x_ub=max(0,0)=0` — a variable fixed at zero.
Harmless extra variables.

---

### 6 — Summary of changes in this session

| File | Change | Reason |
|---|---|---|
| `settings.py` | `BARHOMOGENOUS = 0` | Prevent tau-drift false infeasibility certificates |
| `transitions.py` `get_to_non_ag_exclude_matrices` | Remove `existing_dvars_rk` parameter | Pure UB — no LB logic |
| `transitions.py` `get_lower_bound_non_agricultural_matrices` | Remove UB cap; new docstring | Pure LB — no UB dependency |
| `input_data.py` `get_non_ag_x_rk` | Revert to original | Pure UB from transition matrix |
| `input_data.py` `non_ag_lu2cells` | Use `max(x_rk, lb_rk)` | Include locked-in cells with UB=0 |
| `solver.py` `_setup_non_ag_variables` | `x_ub = max(x_rk[r,k], x_lb)` | Reconcile UB=0 / lb>0 at creation |

---

## 20260603 — Solver infeasibility root cause: degenerate vertices; iterative species removal and dual-simplex fallback as mitigations

### Context

Two parallel run batches — `NECMA_follow_runs_rm_swain` (13 runs, GBF4 species constraints)
and `REM_RES5_dual_simplex` (6 runs, renewable energy targets) — were analysed to
understand why Gurobi barrier (Method=2) repeatedly fails with infeasible or numerical
status while dual simplex (Method=1) can rescue those same years. A shared root cause was
identified across both run types, with different structural drivers.

---

### 1 — Shared root cause: barrier fails at degenerate vertices

Barrier follows the central path through the interior of the feasible region, converging
toward the optimal face as µ → 0. At a **degenerate vertex** — where multiple constraints
are simultaneously near-binding — the slack variables for those constraints approach zero.
Barrier's KKT system requires dividing by these slacks; when they approach zero the system
becomes numerically singular. This produces the observed symptoms:

- Primal values diverging to `~e+33`
- Gurobi reporting `Numerical trouble encountered` or `Infeasible model` after millions of
  barrier iterations with no improvement
- Runs correctly classified as infeasible even though the problem is feasible (confirmed by
  Method=1 finding a solution)

**Dual simplex (Method=1) is immune** because it operates on vertices and edges directly,
using anti-degeneracy pivot rules (perturbation, Bland's rule) that step along the
degenerate face without requiring slacks to remain positive. This is why adding Method=1
as a fallback rescues years where barrier collapses.

**Scaling does not fix this.** The existing `rescale_lhs_rhs_region_species` already
performs per-row per-region geometric mean rescaling for every species constraint — this
is equivalent to (and more sophisticated than) simple row normalisation because it
symmetrically balances LHS/RHS in log space. The failure is the *geometry of the feasible
region near the optimum*, not coefficient magnitudes. No uniform row-scaling changes the
within-row coefficient ratio or the degeneracy of the optimal vertex.

---

### 2 — Species runs (NECMA): static near-infeasibility from sparse habitat

GBF4 SNES/ECNES constraints impose per-species per-NRM-region targets:

```
sum(area_s_r × dvar_r  for r in NRM_region) >= target_s
```

For species whose habitat is confined to few cells within an NRM region, the maximum
achievable area (`sum(area_s_r)`) is close to or below the target from the outset. The
constraint row is a near-zero-slack slab every year from the first year the target binds.
Barrier cannot navigate this; its dual variable for the tight constraint grows without
bound. This is a **static** degeneracy driven purely by data.

**Evidence — iterative Swain's tortoise removal:**

| Batch | Change | Barrier failure year |
|---|---|---|
| `NECMA_follow_runs` | baseline (all species) | **2040** (all NRM-mode runs) |
| `NECMA_follow_runs_rm_swain` | Swain's tortoise removed | **2050** (all NRM-mode runs) |

Removing one species with near-infeasible NRM habitat pushed every affected run forward
by exactly one 5-year time step — consistent with the species' constraint row becoming the
binding degenerate constraint at yr 2040, and another species' row taking that role at
yr 2050. G0011 (AUSTRALIA mode, which aggregates all NRM regions into one national target)
completed successfully in both batches, confirming the per-NRM constraint granularity is
the structural source.

**Method=1 effect in the new batch:**

Old batch used barrier + NumericFocus=3 only. New batch adds Method=1 as a third
fallback. The solver change allowed runs to push one additional time step (2040 → 2050)
even before species removal, by rescuing years where barrier returned non-optimal status
rather than infeasible. The two effects (species removal + Method=1) are additive.

**Status as of 2026-06-03 ~10:15:**

| Run | Group | Key parameter | Old failure | New outcome |
|---|---|---|---|---|
| G0001 | CORE | baseline | yr 2040 | INFEASIBLE yr 2050 |
| G0002 | GHG_SENS | high GHG | yr 2040 | INFEASIBLE yr 2050 |
| G0003 | WATER_SENS | stress=0.5 | yr 2040 | INFEASIBLE yr 2050 |
| G0004 | WATER_SENS | stress=0.7 | yr 2020 crash | STUCK yr 2010 (e+33, ~10.5h) |
| G0005 | WATER_SENS | stress=0.8 | yr 2020 crash | INFEASIBLE yr 2010 |
| G0006 | CLIMATE_SENS | SSP=126 | yr 2040 | INFEASIBLE yr 2050 |
| G0007 | CLIMATE_SENS | SSP=370 | yr 2040 | INFEASIBLE yr 2050 |
| G0008 | BIO_SENS | GBF2_cut=0 | yr 2040 | INFEASIBLE yr 2050 |
| G0009 | BIO_SENS | GBF2_cut=10 | yr 2040 | STUCK yr 2050 (~7h) |
| G0010 | BIO_SENS | GBF2_cut=20 | yr 2040 | STUCK yr 2050 (~7.5h) |
| G0011 | BIO_SENS | AUSTRALIA mode | finished | WRITING OUTPUTS |
| G0012 | SOCIAL_LIC | NonAg_cap=5 | yr 2025 | INFEASIBLE yr 2025 |
| G0013 | SOCIAL_LIC | NonAg_cap=10 | yr 2030 | INFEASIBLE yr 2030 |

---

### 3 — REM runs: dynamic near-infeasibility from early-year lock-in

The REM runs have no species targets. Their barrier degeneracy is constructed
progressively across years by the interaction of three forces:

1. **Non-reversible renewable installations**: solar/wind `dvar_r` cannot decrease once set
2. **Non-ag lower bound fix** (PR `94971ea`): existing non-ag land is correctly pinned as
   a lower bound, removing the artificial slack the pre-fix formulation provided
3. **Escalating state-level renewable targets**: each year's target exceeds the last

The mechanism: the solver installs renewables on the highest-yield cells in early years.
By year 2030–2035, those cells are locked at their lb. The remaining flexible cells have
progressively lower capacity factors. The renewable target constraint LHS
(`sum(yield_r × dvar_r)`) approaches the RHS from above — the constraint becomes a
near-zero-slack slab. The lb constraints on locked-in sites, adoption limits, and the
renewable target simultaneously bind at the optimal vertex. Barrier's KKT system
collapses.

**The lb fix (PR `94971ea`) is correct.** Pre-fix runs were solving an artificially
relaxed formulation where non-ag land could be freely reallocated each year. Post-fix
runs reflect the true constraint tightness of the correctly-modelled problem. The
increased difficulty is not a regression — it is the correct difficulty.

**Per-year solve history (new batch):**

Legend: `✓` barrier solved cleanly · `⚡→✓` barrier infeasible, rescued by Method=1 ·
`⚡→⏳` barrier infeasible, Method=1 still running · `⚡→✗` barrier infeasible, Method=1 diverged

| Run | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | Outcome |
|---|---|---|---|---|---|---|---|---|
| RE0001 | ✓ | ✓ | ✓ | ⚡→⏳ | — | — | — | STUCK yr 2035 (Method=1, >20h) |
| RE0002 | ✓ | ✓ | ✓ | ⚡→✓ | ⚡→✓ | ✓ | ✓ | **FINISHED ✓** |
| RE0003 | ✓ | ✓ | ✓ | ⚡→✓ | ⚡→✓ | ⚡→✓ | ⚡→⏳ | STUCK yr 2050 (~35 min) |
| RE0004 | ✓ | ✓ | ✓ | ⚡→✓ | ⚡→✓ | ⚡→✓ | ⚡→✓ | WRITING OUTPUTS |
| RE0005 | ✓ | ✓ | ✓ | ✓ | ⚡→✓ | ⚡→✓ | ⚡→✓ | **FINISHED ✓** |
| RE0006 | ✓ | ✓ | ✓ | ⚡→✓ | ⚡→✓ | ✓ | ⚡→✓ | **FINISHED ✓** |

RE0002, RE0004, RE0005, and RE0006 all completed the full 2020–2050 horizon via Method=1
rescue; RE0003 solved 2035–2045 via Method=1 and is working on yr 2050. RE0001 remains
stuck at yr 2035 (>20h, dual infeasibility ~1e7 not converging). The exclusion flag
explains the step_change/accelerated_transition splits: RE0002/RE0004
(`Exclude=True`) completed while RE0001/RE0003 (`Exclude=False`) are the two stragglers.
RE0005 (ANU_T10, no exclusion) completed because the transmission-proximity filter
already narrows the installable cell pool, preventing the worst early-year lock-in.

---

### 4 — Plans

**Species (NECMA) — short-term:**

1. Wait for yr 2050 infeasible run to save `debug_model_XXXX_2050.mps`
2. Run `find_infeasible_ecnes.submit_ecnes_checks()` on the MPS (one PBS job per ECNES
   constraint, each maximises the LHS to check if the RHS target is achievable) — see
   skill `docs/CLAUDE_SKILL/debug_ecnes_infeasibility.md`
3. Identify communities whose NRM target exceeds `max_achievable_area`
4. Remove and resubmit as `NECMA_follow_runs_rm_swain_rm_<community>`
5. Repeat until all years solve; compile an explicit exclusion list

**Species (NECMA) — long-term:**

Pre-compute `max_achievable_area` per species per NRM region per year and flag any species
where `target_yr > max_achievable_yr`. Replace hard constraints for these species with
soft penalty terms in the objective — soft constraints bound the dual variable, preventing
KKT blow-up even when the constraint is near-binding.

**REM — short-term:**

Accept Method=1 as the operational fallback. RE0002, RE0004, RE0005, and RE0006 all
completed; RE0003 is working on yr 2050. RE0001 (step_change, no exclusion) is stuck at
yr 2035 with dual infeasibility not converging — may need a kill and resubmit with
`EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS=True`.

**REM — medium-term:**

- **Soft renewable targets**: convert `_add_renewable_energy_constraints()` hard `>=`
  constraints to penalty terms in the objective. This bounds the renewable target dual
  variable, preventing KKT blow-up when the constraint nears binding.
- **Adoption rate smoothing**: limit per-period installation rate so early years do not
  consume all high-yield sites, preserving flexible capacity for years 2035–2050.

---

## 20260602 — RP fix (commit 94971ea) causes infeasibility in REM_RES5 and NECMA_follow_runs

### Context

After commit `94971ea` (Riparian Plantings transition-matrix eviction fix), all six
`REM_RES5` runs and 11 of 13 `NECMA_follow_runs` became infeasible at various years
(2020–2045). Runs that were previously optimal are now reported as `INFEASIBLE` by Gurobi.
Three distinct root causes were identified; only the first is directly caused by the fix.

---

### Run inventory — REM_RES5 (`N:\LUF-Modelling\LUTO2_JZ\Custom_runs\REM_RES5`)

Shared settings: `GBF2_TARGET=high (hard)`, `GHG=low (hard)`, `WATER=on (hard)`,
`RENEWABLE_TARGET=Gladstone - Core`, `GBF4=off`, `GBF3=off`.

| Run | Spatial layers | Excl. GBF2 cells | GBF2 cut Solar/Wind | Fails at | Root cause |
|---|---|---|---|---|---|
| G0001 / RE0001 | step_change | No | 0 / 0 | **2040** | Cause 1 |
| G0002 / RE0002 | step_change | Yes | 25 / 25 | **2035** | Cause 1 (earlier: fewer renewable cells) |
| G0003 / RE0003 | accelerated_transition | No | 0 / 0 | **2045** | Cause 1 |
| G0004 / RE0004 | accelerated_transition | Yes | 25 / 25 | **2035** | Cause 1 (earlier: fewer renewable cells) |
| G0005 / RE0005 | ANU_transmission_T10 | No | 0 / 0 | **2045** | Cause 1 |
| G0006 / RE0006 | ANU_transmission_T10 | Yes | 25 / 25 | **2035** | Cause 1 (earlier: fewer renewable cells) |

---

### Run inventory — NECMA_follow_runs (`N:\LUF-Modelling\LUTO2_JZ\Custom_runs\NECMA_follow_runs`)

Shared settings: `GBF2_TARGET=high (hard)`, `WATER=on (hard)`, `RENEWABLE_TARGET=Gladstone - Core`,
`RENEWABLE_LAYERS=step_change`, `GBF4_SNES/ECNES=USER_DEFINED`.

| Run | Group | GHG | GBF2 cut % | GBF4 mode | Water stress | SSP | Adoption cap | Fails at | Root cause |
|---|---|---|---|---|---|---|---|---|---|
| G0001 | CORE | low | 15 | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0002 | GHG_SENSITIVITY | **high** | 15 | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0003 | WATER_SENSITIVITY | low | 15 | NRM | **0.5** | 245 | — | **2040** | Cause 1 |
| G0004 | WATER_SENSITIVITY | low | 15 | NRM | **0.7** | 245 | — | **2020** | Cause 2 (water too tight) |
| G0005 | WATER_SENSITIVITY | low | 15 | NRM | **0.8** | 245 | — | **2020** | Cause 2 (water too tight) |
| G0006 | CLIMATE_SENSITIVITY | low | 15 | NRM | 0.6 | **126** | — | **2040** | Cause 1 |
| G0007 | CLIMATE_SENSITIVITY | low | 15 | NRM | 0.6 | **370** | — | **2040** | Cause 1 |
| G0008 | BIO_SENSITIVITY | low | **0** | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0009 | BIO_SENSITIVITY | low | **10** | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0010 | BIO_SENSITIVITY | low | **20** | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0011 | BIO_SENSITIVITY | low | 15 | **AUSTRALIA** | 0.6 | 245 | — | **passes** | AUSTRALIA mode relaxes GBF4 |
| G0012 | SOCIAL_LICENCE | low | 15 | NRM | 0.6 | 245 | **5%** | **2025** | Cause 3 (adoption cap) |
| G0013 | SOCIAL_LICENCE | low | 15 | NRM | 0.6 | 245 | **10%** | **2030** | Cause 3 (adoption cap) |

---

### 1 — Primary cause (REM_RES5 all 6; NECMA G0001–G0003, G0006–G0010): non-ag land accumulation exhausts the joint land budget

**Mechanism — pre-fix (bug present):**

`get_to_non_ag_exclude_matrices` set `t_rk = 0` (UB = 0) for RP cells each period because
their dominant `lumap` entry was not RP (see 20260601 entry). `get_lower_bound_non_agricultural_matrices`
then capped `lb_rk` against this zero UB, silently zeroing the lower bound and freeing
every RP cell back to the general pool each year. The optimizer could re-allocate those
cells jointly to both biodiversity targets and renewable energy management — effectively
double-using the land.

**Mechanism — post-fix (correct):**

RP cells now maintain their lower bounds across periods. Irreversible non-ag land
(EP, RP, CP, Agroforestry, BECCS) accumulates monotonically. By year 2035–2045:

- GBF2 high hard target (30% restoration by 2030, 50% by 2050) has locked hundreds of
  thousands of cell-proportions into non-ag land uses
- Gladstone Core renewable targets require large and growing generation (e.g. NSW Solar
  61.6 TWh in 2040, up from 9.4 TWh existing; WA Wind 48 TWh with 3 TWh existing)
- The joint land budget — non-ag biodiversity + renewable-eligible ag — is exceeded

**Why the pre-fix runs appeared feasible:**

The optimizer exploited the annual eviction to strategically reassign RP cells. It could
choose each year which cells go to non-ag vs. renewables, avoiding locking high-value
renewable cells into permanent RP. With the fix, early-year non-ag allocations (made
before renewable targets grew large) are irreversibly retained, even in prime renewable
zones.

**Infeasibility onset by run:**

| Run set | Fails at | Notes |
|---|---|---|
| REM_RES5 G0002, G0004, G0006 (`EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS=True`) | 2035 | GBF2 cells excluded from renewables → pool shrinks faster |
| REM_RES5 G0001, G0003, G0005 (`EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS=False`) | 2040–2045 | |
| NECMA G0001–G0003, G0006–G0010 | 2040 | All use `WATER_STRESS=0.6`, `GBF4=NRM` mode |

**Nature of infeasibility:** The renewable energy constraints are hard (`>=` with no slack).
When locked-in non-ag land reduces the effective renewable-eligible area below what is
needed to meet state-level generation targets, Gurobi returns status 3 (INFEASIBLE).
No `lb > ub` variable-level infeasibility is involved — this is a global land-budget
infeasibility.

**Why NECMA G0011 passes:** `GBF4_SNES_REGION_MODE=AUSTRALIA`, `GBF4_ECNES_REGION_MODE=AUSTRALIA`.
National-level aggregation is far looser than per-NRM constraints, giving the optimizer
enough flexibility to jointly satisfy biodiversity and renewable targets through 2050.

---

### 2 — Secondary cause (NECMA G0004, G0005): `WATER_STRESS` too tight — infeasible from year 2020

`WATER_STRESS=0.7` (G0004) and `WATER_STRESS=0.8` (G0005) require the model to maintain
70%/80% of natural water yield in every river region. This is structurally infeasible from
year 2020 given the simultaneous hard constraints:

- GBF2 high hard (large afforestation lowers water yield)
- Gladstone Core renewable targets at year 2020
- GHG hard constraint

**Unrelated to the RP fix**: year 2020 uses `base_year = YR_CAL_BASE`, so
`existing_dvars_rk = None` and the fix never engages. These runs were likely infeasible
before the fix as well. G0003 (`WATER_STRESS=0.5`) passes 2020 and fails only at 2040
(Cause 1), confirming the threshold is between 0.5 and 0.7.

---

### 3 — Tertiary cause (NECMA G0012, G0013): `NON_AG_CAP` prevents GBF2 ramp

`REGIONAL_ADOPTION_CONSTRAINTS=NON_AG_CAP` with `NON_AG_CAP=5` (G0012) and `NON_AG_CAP=10`
(G0013) caps non-ag adoption at 5%/10% per NRM region per period. The GBF2 hard target
requires faster restoration than the cap permits:

- G0012 (5% cap): meets 2020 target, fails at **2025**
- G0013 (10% cap): meets 2020 and 2025, fails at **2030**

Not caused by the RP fix — the adoption rate constraint independently prevents the
biodiversity target from being reached.

---

### 4 — Options for resolution

| Option | Addresses | Trade-off |
|---|---|---|
| Make renewable energy constraints **soft** (add slack + penalty in `_add_renewable_energy_constraints`) | Cause 1 | Model reports shortfall instead of crashing; renewable targets become aspirational |
| Set `GBF2_CONSTRAINT_TYPE=soft` in REM/NECMA scenarios | Cause 1 | Biodiversity gives way when land budget is exhausted |
| Reduce `GBF2_TARGET` to `medium` in renewable-heavy scenarios | Cause 1 | Accepts lower biodiversity target where renewables are required |
| Reduce `WATER_STRESS` to ≤ 0.6 | Cause 2 | G0004/G0005 water scenarios were structurally too tight |
| Reduce `NON_AG_CAP` or use soft GBF2 | Cause 3 | Social licence scenarios need lower ramp rate or relaxed biodiversity |

The infeasibility from Cause 1 is **correct model behaviour** after the RP fix — the
scenarios were only feasible before because the eviction bug provided illegal land
flexibility. The pre-fix solutions violated irreversibility.

---

### 5 — Why the GBF2–renewable conflict is temporal, not spatial

A natural question: GBF2 is a national target (can be met anywhere in Australia) and,
without `EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS`, renewables are not spatially excluded
from GBF2 priority cells. Both targets therefore appear spatially compatible — why is the
combined scenario infeasible?

**The conflict is not in any single year; it is accumulated across the rolling horizon.**

The rolling-horizon optimizer has no foresight beyond the current period. When solving
year 2025:

- It places EP/RP/CP wherever it is cheapest to meet 30% GBF2 nationally.
- The cheapest GBF2 restoration is concentrated in productive NSW/QLD/WA agricultural
  zones — they are degraded (good for GBF2), economically accessible, and have existing
  rural infrastructure.
- NSW/QLD/WA are also the states with the largest renewable energy targets.
- In 2025 this does not matter: the 2025 renewable targets are small and existing
  installed capacity already covers most of them.

After the fix these 2025 non-ag commitments are irreversible. In 2030 more are added
(GBF2 target growing from 30% toward 50%), again concentrated in the same cheapest
states. The pattern compounds:

| Year | Cumulative locked non-ag in NSW/QLD/WA | Renewable gap still to fill |
|---|---|---|
| 2025 | small | small (mostly met by existing capacity) |
| 2030 | grows | moderate |
| 2035 | larger | large |
| 2040 | **too large** | **~52 TWh NSW Solar + 47 TWh NSW Wind + 45 TWh WA Wind** |

By 2040 the locked-in EP/RP/CP/BECCS has consumed enough renewable-eligible cells
within specific states that state-level hard constraints become infeasible.

**Why the pre-fix optimizer avoided this:** the eviction bug was in effect re-optimising
the spatial assignment of GBF2 every year. As renewable targets grew in later years the
optimizer could opportunistically shift non-ag out of prime renewable zones (NSW/WA) and
into lower-renewable-pressure states (SA, NT, Tasmania), because eviction freed those
cells. Post-fix it cannot undo 2025–2035 commitments.

**Solver credit for locked-in RP is correct:** the GBF2 constraint LHS does include the
Gurobi variable for each locked-in RP cell (lb = RP_PROPORTION, in `non_ag_lu2cells`
and in `_add_GBF2_constraints`). Gurobi correctly counts the lb contribution — the
remaining biodiversity shortfall the solver must close from other land is properly
reduced. The infeasibility is not caused by missing biodiversity credit; it is caused by
the loss of renewable-eligible ag cells within specific states.

**Diagnostic next step:** compute the IIS on the saved MPS file to confirm which
state-level renewable constraint is the binding one:

```
Run_G0001/output/2026_06_01__15_37_34_RF5_2020-2050/debug_model_2035_2040.mps
```

---

### 6 — Verification: `GBF2_TARGET=medium` resolves infeasibility locally

Two RF5 2010-2050 local runs were compared to confirm that lowering the GBF2 target
from `high` to `medium` resolves the infeasibility:

| Run | GBF2 target | 2020 | 2030 | 2040 | 2050 |
|---|---|---|---|---|---|
| `output/2026_06_02__09_43_00_RF5_2010-2050` | high (30%→50%) | Optimal | Optimal | **INFEASIBLE** | — |
| `output/2026_06_02__10_53_36_RF5_2010-2050` | medium (30% flat) | Optimal | Optimal | Optimal | Optimal |

Both runs share identical objectives at 2020 and 2030 (`1.190e+03`, `-7.044e+04`),
confirming the GBF2 constraint is not binding until 2040 when the high target ramps
from 30% toward 50%. The medium target (30% flat) removes this ramp and keeps the
land budget feasible through 2050.

The medium run that solved was configured with settings that are **more restrictive than
REM_RES5** in some dimensions, giving confidence the fix will transfer:

| Setting | Local medium (solved) | REM_RES5 |
|---|---|---|
| `GHG_EMISSIONS_LIMITS` | **high** | low (easier) |
| `EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS` | True, 20% cut | False or True 25% cut |
| `RENEWABLE_LAYERS` | step_change | step_change / accel. / ANU_T10 |
| `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT` | 20 | 15 |

Because the local run solved despite harder GHG constraints, the same GBF2 relaxation
is expected to work for REM_RES5. Confidence by run:

| REM_RES5 runs | Expected outcome with medium GBF2 |
|---|---|
| G0001, G0003 (no exclusion, step_change / accel.) | Very likely feasible — less spatially restricted than local run |
| G0002, G0004 (exclusion 25%, step_change / accel.) | Likely feasible — exclusion active but GHG much easier |
| G0005 (ANU_T10, no exclusion) | Probably feasible |
| G0006 (ANU_T10, exclusion 25%) | Uncertain — both spatial restrictions active; most constrained case |

`ANU_transmission_T10` restricts renewable placement to near-grid cells, making G0005/G0006
the only cases where medium GBF2 may still be insufficient. Pending resubmission results.

---

## 20260601 — Riparian Plantings area incorrectly reduced despite `NON_AG_LAND_USES_REVERSIBLE = False`

### Context

Run `2026_05_31__13_46_26_RF5_2020-2050` showed Riparian Plantings (RP) area declining
between years despite `NON_AG_LAND_USES_REVERSIBLE['Riparian Plantings'] = False`. The
stdout log contained repeated `NonAg lb capped` messages with gaps up to 0.3 — far larger
than the 1e-2 `FEASIBILITY_TOLERANCE` that the existing capping code was designed to absorb.
Total RP dvar peaked at 112.48 (2045) and fell to 99.44 (2050); earlier periods showed
similar erosion.

---

### 1 — Root cause: transition matrix evicts existing RP from EP-dominated cells

`get_lower_bound_non_agricultural_matrices()` (non_agricultural/transitions.py ~L1137)
sets the lb for each non-ag land use to the previous year's dvar (floor-truncated), then
caps it against the solver's UB from `get_to_non_ag_exclude_matrices()`:

```python
non_ag_x_rk = get_to_non_ag_exclude_matrices(data, data.lumaps[base_year])
lb_capped = np.minimum(lb_capped, non_ag_x_rk)
```

`non_ag_x_rk` for RP is built as:

```python
t_rk[non_ag_cells, :] *= T_MAT[lumap_by → RP]   # current dominant land use → RP
t_rk[non_ag_cells, :] *= T_MAT[lumap_2010 → RP]  # 2010 base land use → RP
t_rk = where(isnan(t_rk), 0, 1)
UB_RP = t_rk × RP_PROPORTION
```

`T_MAT['Environmental Plantings' → 'Riparian Plantings'] = NaN` (transition not in matrix).
`T_MAT['Riparian Plantings' → 'Riparian Plantings'] = 0.0` (self-transition costs nothing,
and is non-NaN, so it is **allowed**).

Many cells hold a partial RP allocation (dvar_rp > 0) while their dominant land use is
Environmental Plantings (EP). When `lumap_by = 'EP'`:

```
T_MAT[EP → RP] = NaN  →  t_rk = 0  →  UB = 0
lb_rk = previous dvar (e.g. 0.12)  →  lb capped to 0  →  RP deleted
```

The transition matrix is intended to gate *new* allocations; it should not evict
*existing* allocations in irreversible land uses. The capping code conflates these two cases.

**Confirmed by data**: all capped RP cells have `t_rk = 0`; none are caused by
`RP_PROPORTION` (FEASIBILITY_TOLERANCE) overrun. The capped-cells CSV
(`rp_capped_cells.csv`) shows every row has `ub_rp = 0.0` and `t_rk = 0`, with
`lumap_by` = EP or another non-ag LU.

---

### 2 — Why RP is the only affected non-ag land use

`RP_PROPORTION = (2 × BUFFER_WIDTH × STREAM_LENGTH) / (REAL_AREA_NO_RESFACTOR × 10000)`

Its maximum across all cells is **0.39**, meaning RP can occupy at most 39% of any cell.
Therefore, the cell's `lumap` is **always** some other land use (EP, Agroforestry, or an
agricultural land use). The cell never has `lumap = 'Riparian Plantings'`.

Consequence: for every cell that carries an RP dvar, the transition lookup is
`T_MAT[other_LU → RP]`. Because most non-ag → RP cross-transitions are NaN in T_MAT,
`t_rk = 0` for virtually all existing RP allocations. The bug affects **all** RP cells
every year.

All other irreversible non-ag land uses (EP, Carbon Plantings, Agroforestry) can occupy
100% of a cell, so their dominant `lumap` equals themselves, the self-transition
`T_MAT[X → X]` is valid, and `t_rk = 1`. The transition-matrix eviction only hits them
for rare minority allocations, with negligible area loss. RP is structurally excluded from
ever being its own `lumap` by the `RP_PROPORTION` physical constraint.

---

### 3 — Why the fix must touch both the UB and the lb

`non_ag_x_rk` (the UB) does double duty in the solver:

```python
# input_data.py
non_ag_lu2cells = {k: np.where(non_ag_x_rk[:, k])[0] ...}

# solver.py
for r in non_ag_lu2cells[k]:          # only cells with UB > 0 get a variable
    addVar(lb=non_ag_lb_rk[r,k], ub=non_ag_x_rk[r,k])
```

If `non_ag_x_rk[cell, RP] = 0`, the cell never enters `non_ag_lu2cells`, no Gurobi
variable is created for it, and its dvar is implicitly 0 in the next period — regardless
of what the lb says. A lb-only fix is therefore ineffective.

---

### 4 — Fix: one parameter in `get_to_non_ag_exclude_matrices` (transitions.py)

`get_to_non_ag_exclude_matrices` is the single source of both the UB (`get_non_ag_x_rk`
in input_data.py) and the UB used to cap the lb (`get_lower_bound_non_agricultural_matrices`
in transitions.py). Adding one optional parameter fixes both call sites at once:

```python
def get_to_non_ag_exclude_matrices(data, lumap, existing_dvars_rk=None):
    ...
    t_rk = np.where(np.isnan(t_rk), 0, 1).astype(np.int8)

    # Cells that already hold an irreversible non-ag allocation bypass the transition
    # matrix — the matrix gates *new* allocations only, not existing ones.
    if existing_dvars_rk is not None:
        for k, k_name in enumerate(data.NON_AGRICULTURAL_LANDUSES):
            if not settings.NON_AG_LAND_USES_REVERSIBLE.get(k_name, True):
                t_rk[existing_dvars_rk[:, k] > 0, k] = 1
    ...
```

Both callers pass the previous period's dvars:

```python
# transitions.py — get_lower_bound_non_agricultural_matrices
non_ag_x_rk = get_to_non_ag_exclude_matrices(
    data, data.lumaps[base_year],
    existing_dvars_rk=data.non_ag_dvars[base_year],
)

# input_data.py — get_non_ag_x_rk
existing_dvars = data.non_ag_dvars.get(base_year) if base_year != data.YR_CAL_BASE else None
return get_to_non_ag_exclude_matrices(data, data.lumaps[base_year], existing_dvars_rk=existing_dvars)
```

No recovery scaffolding anywhere. The fix is entirely inside `get_to_non_ag_exclude_matrices`.

---

### 5 — Other irreversible LUs also affected (small scale)

The same bug affected EP, Sheep/Beef Agroforestry, and Carbon Plantings wherever they
appeared as minority allocations in cells dominated by a different non-ag LU. Pre-fix
area losses (transition-matrix eviction only):

| Land use | Total area lost 2020–2050 | Peak cells/period |
|---|---:|---:|
| Riparian Plantings | **61.2** | **240** |
| Environmental Plantings | 1.07 | 29 |
| Sheep Agroforestry | 0.030 | 8 |
| Beef Agroforestry | 0.0005 | 29 |
| Others | < 0.003 | — |

RP is 57× worse because `RP_PROPORTION` max = 0.39 means RP can **never** be the
dominant land use — its `lumap` is always another LU, so the transition matrix always
blocks it.

---

### 6 — Verification

`verify_fix.py` checks both `non_ag_lb_rk` (lb) and `non_ag_x_rk` (UB / cell inclusion)
against the same `Data_RES5.lz4`:

| Year transition | Before (RP area lost) | After |
|---|---:|---:|
| 2020 → 2025 | 0.877 | 0.000 |
| 2025 → 2030 | 9.920 | 0.000 |
| 2030 → 2035 | 6.226 | 0.000 |
| 2035 → 2040 | 6.225 | 0.000 |
| 2040 → 2045 | 24.921 | 0.000 |
| 2045 → 2050 | 13.024 | 0.000 |

Remaining `NonAg lb capped` messages show `max gap = 2.97e-08` — pure float32 rounding,
not real area loss. RP total dvar is now monotonically non-decreasing as intended.

---

## 20260530 — PBS checkpoint/redo pipeline: simulation.py per-year checkpointing, python_script.py auto-detect, redo_checkpoint.py batch resubmit

### Context

Gadi PBS jobs for multi-year LUTO simulations are frequently wall-time killed before
completing all years. Previously, such runs were unrecoverable — the full 320 GB data
object had to be reloaded from scratch on resubmit. A checkpoint/redo pipeline was
implemented to resume from the last solved year without reloading data.

---

### 1 — `simulation.py`: per-year checkpoint saves into the timestamped output subdir

`solve_timeseries()` now accepts `checkpoint_path: Path | None`. After each successfully
solved year it writes `data_{year}.lz4` (via a `.tmp` rename to be atomic), then deletes
any previous `data_*.lz4` in the same directory — keeping only the most recent checkpoint:

```python
if checkpoint_path is not None:
    final_path = checkpoint_path / f"data_{target_year}.lz4"
    tmp_path = Path(f"{final_path}.tmp")
    save_data_to_disk(data, str(tmp_path))
    os.replace(tmp_path, final_path)
    for old in checkpoint_path.iterdir():
        if re.match(r'data_\d{4}\.lz4', old.name) and old != final_path:
            old.unlink()
```

`run()` passes `Path(save_dir)` as `checkpoint_path` so the lz4 lives inside the
timestamped output subdir (`output/TIMESTAMP_RF1_2020-2050/data_2025.lz4`), not in the
run root where it would appear as an untracked code change.

On resume, `run()` scans that directory for `data_\d{4}\.lz4`, loads the latest via
`joblib.load`, restores `active_data.timestamp` and `active_data.path` from the
already-written `.timestamp` file, then calls `solve_timeseries` with only the remaining
years:

```python
files = sorted(f for f in checkpoint_path.iterdir() if re.match(r'data_\d{4}\.lz4', f.name))
if files:
    resume_from_year = int(files[-1].stem.split("_")[1])
    active_data = joblib.load(str(files[-1]))
    active_data.timestamp = read_timestamp()
    active_data.path = save_dir
```

**Why `\d{4}` not `data_*.lz4`**: `pathlib.glob` does not support regex alternation;
`re.match` over `iterdir()` is used throughout so the pattern is exact.

---

### 2 — `python_script.py`: auto-detects checkpoint, skips `load_data()`

The key invariant: `sim.load_data()` calls `write_timestamp()`, overwriting
`output/.timestamp` with a new value — which would cause `sim.run()` to construct a
**new** output directory, losing the connection to the existing partial output.

The script avoids this by scanning `output/` subdirs for `data_\d{4}.lz4` before
deciding whether to call `load_data()`:

```python
_checkpoint_dir = next(
    (str(d) for d in sorted(pathlib.Path(settings.OUTPUT_DIR).iterdir(), key=lambda d: d.name)
     if d.is_dir() and any(re.match(r'data_\d{4}\.lz4', f.name) for f in d.iterdir())),
    None
)
data = None if _checkpoint_dir else sim.load_data()
data = sim.run(data=data, ..., checkpoint_dir=_checkpoint_dir)
```

If a checkpoint dir is found: `load_data()` is skipped → `.timestamp` is unchanged →
`sim.run()` reconstructs the same `save_dir` → the simulation continues writing into
the original output directory. The `data = sim.run(...)` capture is critical — without it
`data` remains `None` and the downstream archive step (`pathlib.Path(data.path)`) raises
`AttributeError`.

---

### 3 — `redo_checkpoint.py`: batch resubmit for stalled runs

Deployed to the task root (e.g. `REM_RES1/`) alongside `run_all.py`. Classifies every
`Run_G*` directory as one of four states:

| State | Condition |
|---|---|
| finished | `Run_Archive.zip` exists |
| running | directory is the `PBS_O_WORKDIR` of a live Gadi job (`qstat -f -u $USER`) |
| checkpoint | has `data_\d{4}.lz4` inside any `output/` subdir |
| incomplete | none of the above |

For each checkpoint run:
1. Reads `task_param.py` (the original PBS settings written at submission time) to
   extract base `MEM`, `NCPUS`, `TIME`, `QUEUE`.
2. Overrides only the params explicitly passed as CLI flags; all others are inherited.
3. Writes `redo_param.py` with the merged settings.
4. Calls `bash redo_cmd.sh` from the run directory, which submits a new PBS job via
   `qsub`.

`--dry-run` is fully read-only: classification and output printing happen, but no files
are written and no jobs are submitted.

**CLI interface:**

```bash
# inherit all PBS settings from each run's task_param.py
python redo_checkpoint.py --dry-run

# override walltime only; mem/ncpus/queue still inherited
python redo_checkpoint.py --time 24:00:00

# override everything explicitly
python redo_checkpoint.py --mem 500gb --ncpus 96 --time 24:00:00 --queue normal
```

---

### 4 — `redo_cmd.sh`: mirrors `task_cmd.sh`, sources `redo_param.py`

`task_cmd.sh` sources `task_param.py` and submits `python_script.py` via `conda run`.
`redo_cmd.sh` is identical except it sources `redo_param.py`. Because `python_script.py`
auto-detects the checkpoint internally, no additional arguments are needed — the redo
job calls the same script as the original submission.

`SCRIPT_DIR` is resolved from `${BASH_SOURCE[0]}` so the absolute path to
`python_script.py` is correct regardless of where `redo_cmd.sh` is called from.

---

### 5 — `helpers.py`: redo scripts copied at task-create and submit time

`create_task_runs()` now copies `redo_checkpoint.py` and `redo_cmd.sh` to the task root
alongside `run_all.py` (two plain `shutil.copyfile` calls, no loop).

`submit_task()` in cluster mode copies `redo_cmd.sh` and `python_script.py` into each
`Run_G*/` directory alongside `task_cmd.sh`, so every run dir is self-contained for both
initial submission and checkpoint redo.

---

## 20260530 — Biodiversity report fixes: resfactor denominator, Target_by_Percent NaN sentinel, Relative_Contribution_Percentage formula, AUSTRALIA region rows, GBF4_SNES Vue setting key

### Context

A cluster of related bugs was identified and fixed across `write.py`, `data.py`,
`create_report_data.py`, and `Biodiversity.js`. The bugs shared a common root: incorrect
handling of sparse biodiversity layers at coarsened resolution and a poorly-defined
`Relative_Contribution_Percentage` denominator.

---

### 1 — `use_valid_cell_count=False` for sparse biodiversity layer resfactoring

`get_resfactored_average_fraction` averages a 1D species/habitat array over coarsened
spatial blocks of size RF². It has two denominator modes:

- **Default** (`use_valid_cell_count=True`): divides each block sum by the count of
  non-zero (valid) cells in that block.
- **`use_valid_cell_count=False`**: divides by RF² — the total number of cells in the block.

For dense arrays (e.g. land-use masks that cover most cells), both modes give nearly the
same result. For **sparse biodiversity layers** (GBF3 NVIS: 95% zero; GBF4 SNES: 99.65%
zero; GBF4 ECNES: 99.6% zero), the distinction is critical.

**The bug**: with `use_valid_cell_count=True`, a block containing 3 habitat cells out of
RF²=25 cells would divide the block sum by 3 instead of 25, inflating the resfactored
fraction by ~8×. This propagated into solver input arrays (`GBF3_NVIS_LAYERS_ALL`,
`GBF4_SNES_LAYERS_ALL`, `GBF4_ECNES_LAYERS_ALL`) and into the biodiversity score CSVs
written by `write_biodiversity_GBF3_NVIS_scores`, `write_biodiversity_GBF4_SNES_scores`,
and `write_biodiversity_GBF4_ECNES_scores`.

**Fix**: `use_valid_cell_count=False` added to all 7 call sites:

| File | Function | Affected arrays |
|---|---|---|
| `data.py` | `Data.__init__` | `GBF3_NVIS_LAYERS_ALL`, `GBF4_SNES_LAYERS_ALL`, `GBF4_ECNES_LAYERS_ALL` |
| `write.py` | `write_biodiversity_GBF3_NVIS_scores` | `nvis_layers_arr` (used in score CSV + map layers) |
| `write.py` | `write_biodiversity_GBF4_SNES_scores` | `snes_layers_arr` (used in score CSV + map layers) |
| `write.py` | `write_biodiversity_GBF4_ECNES_scores` | `ecnes_layers_arr` (used in score CSV + map layers) |

This matches the LUMASK-fix reasoning from Step 8 (2026-05-02): the solver needs the
correct fraction of habitat within each coarse block, not the fraction among only the
habitat-containing fine cells.

---

### 2 — `Target_by_Percent` set to NaN when no constraint is active

Previously, `write.py` always computed:

```python
df['Target_by_Percent'] = (
    (df['TARGET_INSIDE_SCORE'] + df['BASE_OUTSIDE_SCORE']) / df['BASE_TOTAL_SCORE'] * 100
)
```

When `TARGET_INSIDE_SCORE = 0` (no active biodiversity constraint for that group/species
× region combination), this still produced a numeric value (`BASE_OUTSIDE_SCORE /
BASE_TOTAL_SCORE * 100`), which the downstream filter in `create_report_data.py`:

```python
bio_df[bio_df['Target_by_Percent'].notna()]
```

could not distinguish from a real target. Result: the target lookup table
(`_gbf3_target_lk`, `_ecnes_target_lk`) was polluted with spurious zero-constraint rows,
which caused incorrect target lines to appear in the report.

**Fix**: replaced with `np.where(TARGET_INSIDE_SCORE > 0, formula, np.nan)` in all six
places across GBF3 NVIS, GBF4 SNES, and GBF4 ECNES (both per-region and outside rows).
`create_report_data.py` updated its comment to reflect the new invariant: `.notna()` now
correctly selects only rows with a real active constraint.

---

### 3 — `Relative_Contribution_Percentage` formula: area / ALL_HA replaces (area + outside) / baseline

The old formula for GBF3 NVIS, GBF4 SNES, and GBF4 ECNES sum CSVs was:

```python
# Old: inside LUTO + outside LUTO as fraction of pre-1750 baseline
(Area Weighted Score (ha) + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100
```

This was a "restoration fraction" — it answered "what fraction of the pre-1750 habitat is
maintained or restored?" The problem: `BASE_TOTAL_SCORE` is the pre-1750 total across
all land (in and out of LUTO), making it an awkward denominator for a column that is
intended to track just the inside-LUTO contribution. When stacked with the "Outside LUTO
study area" row (which uses the same denominator), the sum of all Type rows could
substantially exceed or fall short of 100%, confusing report chart rendering.

**New formula**:

```python
# New: area weighted score as fraction of total regional land area
Area Weighted Score (ha) / ALL_HA * 100
```

`ALL_HA` is the total cell-area (ha) of all LUTO cells in the region, loaded from a
per-(region, region_level) lookup. This makes `Relative_Contribution_Percentage` a
consistent area-fraction that sums correctly across Types (Ag + Am + NonAg + Outside
LUTO = 100% of the regional land area).

**Applied identically** to both the Type-specific `sum_df` rows and the `outside_sum`
rows in all three write functions. `create_report_data.py` updated to use this column
directly instead of recomputing `Sum_Pct (%)` from `Area Weighted Score (ha) /
BASE_TOTAL_SCORE`.

---

### 4 — AUSTRALIA rows kept in `bio_df`; ranking queries explicitly exclude them

All three biodiversity `process_biodiversity_data` sections previously dropped AUSTRALIA
rows from `bio_df` with:

```python
bio_df = bio_df.query('species != "ALL" and region != "AUSTRALIA"')
```

**Intent**: prevent double-counting when downstream code summed across regions (since
`AUSTRALIA` = sum of all NRM/STATE regions). **Side effect**: the AUSTRALIA region
selection in the report dropdown showed no data.

**Fix**: AUSTRALIA rows are now retained in `bio_df` so the AUSTRALIA selection is
populated, but ranking queries (which group by `region` to compare NRM/STATE regions)
explicitly exclude AUSTRALIA:

```python
# Main bio_df — keep AUSTRALIA for dropdown data
bio_df = bio_df.query('species != "ALL"')

# Ranking — exclude AUSTRALIA (it's the sum of all regions, not a peer)
bio_rank_total = bio_df.query(
    'Water_supply != "ALL" and Landuse != "ALL" and '
    '`Agricultural Management` != "ALL" and region != "AUSTRALIA"'
).groupby(...)
```

The same pattern applied to GBF3 NVIS, GBF4 SNES, and GBF4 ECNES sections.
`sum_bio_df` continues to drop AUSTRALIA (only the Type=ALL filter needed; the
`Relative_Contribution_Percentage` column already carries the correct region-level value).

---

### 5 — Vue.js: GBF4_SNES metric uses `WRITE_SNES` setting key

`Biodiversity.js` `METRIC_TO_SETTING` maps each metric name to the settings key that
controls whether data for that metric was written:

```js
// Before (wrong key):
'GBF4_SNES': 'GBF4_TARGET_SNES',

// After (correct key):
'GBF4_SNES': 'WRITE_SNES',
```

`GBF4_TARGET_SNES` does not exist in settings — the flag that enables SNES output writing
is `WRITE_SNES`. With the wrong key, the Vue component would always treat GBF4_SNES
data as absent and hide the metric from the selection dropdown even when the SNES scores
had been written.

---

## 20260529 — `create_report_data` profiling: manifest.json registration, pyarrow CSV reader, bulk nested-dict builder (166× speedup)

### Context

A full RF5 run (`2026_05_29__16_52_22_RF5_2010-2050`) was used to profile the complete
report generation pipeline. File timestamps in `DATA_REPORT/data/` gave precise wall-clock
costs for each output JS file produced by `create_report_data.py`.

---

### 1 — `write_outputs` wall-clock breakdown (from stdout log timestamps)

| Stage | Elapsed | Notes |
|---|---:|---|
| Mosaic maps (all 5 years) | 0.1 min | fast |
| GBF3 NVIS scores (5 years parallel) | ~6 min | |
| **ECNES scores (5 years parallel)** | **~20 min** | ~13 min per year |
| **SNES scores (5 years parallel)** | **~19 min** | ~12 min per year |
| GBF2 priority scores | ~10 min | 2040/2050 notably slower |
| Renewable energy | ~6 min | |
| Ag-to-ag transitions | ~2 min | |
| Economics, GHG, water, etc. | ~5 min | |
| **Total `write_outputs`** | **88 min** | |

`create_report_data` starts at t = 88 min; the write phase itself is the primary bottleneck.

---

### 2 — `create_report_data` wall-clock breakdown (from JS file modification times)

| Elapsed | File written | Size |
|---:|---|---:|
| 0 min | All non-bio parallel jobs (area, GHG, water, economics, etc.) | ✓ |
| 0.4 min | GBF3 NVIS overview + Sum | ✓ |
| 0.9 min | GBF3 NVIS Ag | 34 MB |
| **8.2 min** | **GBF3 NVIS Am** | **66 MB** |
| 8.3 min | GBF3 NVIS NonAg | ✓ |
| 9.5 min | GBF4 SNES overview | ✓ |
| 10.3 min | GBF4 SNES Sum | ✓ |
| **30.2 min** | **GBF4 SNES Ag** | **208 MB** |
| ❌ never | SNES Am, SNES NonAg, all ECNES, BIO_ranking | run interrupted |

The notebook was cut off after SNES Ag (30 min). SNES Am — the most expensive section due
to the extra `am` dimension — never completed.

---

### 3 — `manifest.json` files in `_chunks` directories classified as `Unknown`

`get_all_files` logged "Unknown files found" for every `manifest.json` inside
`xr_biodiversity_*_YYYY_chunks/` directories. These files were then silently dropped.

**Root cause**: `extract_dtype_from_path` and `_base_name_ext` in
`luto/tools/report/data_tools/__init__.py` applied the parent-directory-name logic only
to `.nc` files:

```python
if path.endswith('.nc') and re.search(r'_\d{4}_chunks$', os.path.basename(parent)):
```

`manifest.json` does not end with `.nc`, so it fell through to `Unknown`.

**Fix**: extended the condition to cover `manifest.json` as well:

```python
_in_chunks = re.search(r'_\d{4}_chunks$', os.path.basename(parent))
if (path.endswith('.nc') or os.path.basename(path) == 'manifest.json') and _in_chunks:
    base_name = re.sub(r'_\d{4}_chunks$', '', os.path.basename(parent))
```

Applied identically in both `extract_dtype_from_path` and `_base_name_ext`. `manifest.json`
files now classify as `xarray_layer` (parent dir name matches `xr_` prefix) with
`base_ext = '.json'`.

---

### 4 — `pd.read_csv` replaced with `engine='pyarrow'` for biodiversity score CSVs

The large biodiversity score CSVs (SNES: 135 MB/year × 5 years = 675 MB total; NVIS: 16 MB;
ECNES: 11 MB) were loaded via `pd.read_csv` — the slowest path for large files.

**Benchmark** (5 SNES files, 3 runs):

| Approach | min(s) | Speedup | Peak RAM |
|---|---:|---:|---:|
| `pd.read_csv` (baseline) | 9.0 | 1.0× | 793 MB |
| `pd.read_csv(engine='pyarrow')` | 1.5 | **6.0×** | 567 MB |
| `polars` + `.to_pandas()` | 1.6 | 5.7× | 180 MB |
| `pyarrow.csv.read_csv` | 1.2 | 7.3× | 180 MB |
| Parquet (recurring read) | 1.2 | 7.5× | 567 MB |
| Feather (recurring read) | 1.1 | 8.0× | 567 MB |
| Dask | 20.9 | 0.4× | 1860 MB |

`engine='pyarrow'` was chosen: one-keyword change, no new imports, no column filtering,
identical DataFrame output, also silences the `DtypeWarning` on the mixed-type
`Agricultural Management` column.

**Applied** to all 6 bio `pd.read_csv` calls in `process_biodiversity_data`:
overall quality, GBF3 NVIS, GBF4 SNES, GBF4 ECNES, GBF8 species, GBF8 groups.

---

### 5 — O(N²) nested-dict builder replaced by `_build_out_dict_bulk` (166× speedup)

**Root cause of the multi-hour `create_report_data` runtime:**

Every biodiversity chart section (Ag, Am, NonAg, overview, Sum) built its output dict
with this pattern:

```python
out_dict = {}
for (region_level, region, species, am, water), df_pct in df_wide_pct.groupby([...]):
    df_pct = df_pct.drop([...], axis=1)
    df_area = df_wide_area[
        (df_wide_area['region_level'] == region_level) & ... & (df_wide_area['water'] == water)
    ].drop([...], axis=1)             # ← full table-scan every iteration
    out_dict[...][am][water] = {
        'Percent': df_pct.to_dict(orient='records'),   # ← small to_dict per group
        'Area':    df_area.to_dict(orient='records'),
    }
```

Two compounding problems:
1. **O(N²) boolean filter**: `df_wide_area[boolean_mask]` scans the full DataFrame for
   every group. With SNES Am having ~4,300 groups (50-species sample), this alone costs
   ~7 s/50 species → extrapolated ~5.5 min for 1,937 species.
2. **N small `to_dict` calls**: calling `to_dict(orient='records')` on a tiny sub-DataFrame
   4,300 × 2 = 8,600 times has severe Python overhead. This dominates the remaining time.

**Benchmark** (50-species sample, N=3 runs, extrapolated to full 1,937 species via ×39
scale factor):

| Approach | min(s) | Speedup | Extrapolated full run |
|---|---:|---:|---:|
| baseline (O(N²) filter + N small to_dict) | 26.9 | 1.0× | ~17 min |
| pre-index area dict, O(1) lookup | 23.8 | 1.1× | ~15 min |
| merge pct+area, one loop | 23.7 | 1.1× | ~15 min |
| zip column lists per group | 14.3 | 1.9× | ~9 min |
| **bulk_to_dict** (one `to_dict` on full df, group via `defaultdict`) | **0.63** | **43×** | **~24 s** |
| **bulk_zip** (no `to_dict` at all — `zip` columns, group in Python) | **0.16** | **166×** | **~6 s** |

The winning approach (`bulk_zip`) converts the entire DataFrame to Python lists via
column `.tolist()` calls, then groups rows into a `defaultdict` — zero pandas groupby,
zero `to_dict`. The key insight: **one big column-list extraction is 166× cheaper than
N small `to_dict` calls on sub-DataFrames**.

**Fix**: `_build_out_dict_bulk(df_wide_pct, df_wide_area, key_cols)` helper added at
line 136 of `create_report_data.py`:

```python
def _build_out_dict_bulk(df_wide_pct, df_wide_area, key_cols):
    from collections import defaultdict
    def _df_to_keyed(df):
        keys = list(zip(*[df[c].tolist() for c in key_cols]))
        leaf_cols = [c for c in df.columns if c not in key_cols]
        rows = [dict(zip(leaf_cols, r)) for r in zip(*[df[c].tolist() for c in leaf_cols])]
        grouped = defaultdict(list)
        for k, row in zip(keys, rows):
            grouped[k].append(row)
        return grouped
    pct_grouped  = _df_to_keyed(df_wide_pct)
    area_grouped = _df_to_keyed(df_wide_area)
    out_dict = {}
    for key, pct_list in pct_grouped.items():
        d = out_dict
        for k in key[:-1]:
            d = d.setdefault(k, {})
        d[key[-1]] = {'Percent': pct_list, 'Area': area_grouped.get(key, [])}
    return out_dict
```

**Applied at 21 call sites** across all biodiversity modules:

| Module | Sections replaced |
|---|---|
| Quality | overview, Ag, Am, NonAg |
| GBF2 | overview, Ag, Am, NonAg |
| GBF3 NVIS | overview, Ag, Am, NonAg |
| GBF4 SNES | overview, Sum, Ag, Am, NonAg |
| GBF4 ECNES | overview, Ag, Am, NonAg |

Two Sum sections (GBF3 NVIS Sum, GBF4 ECNES Sum) were **not** changed: they inject a
per-species target line into the records, which requires per-group logic. They use the
small `sum_scores` CSVs (4.9 MB / 393 KB) so they are not a bottleneck.

**Expected total `create_report_data` biodiversity time**: reduced from >1 hour (never
completing) to an estimated 2–5 minutes.

---

## 20260529 — Report layer pipeline: chunk magnitude tracking, get_all_files chunk discovery, _score_to_df BLAS optimisation, chunk_num JS merging

### 1 — `save2chunk` now returns inline magnitude (avoids disk reload)

`_mag_from_chunks` scanned every saved `chunk_*.nc` file after the loop to compute
`[min, max]` quantiles — re-reading data that was already in memory when it was written.

**Fix:** `save2chunk` now calls `get_mag(in_xr)` before writing and returns the result.
`_save_fullsize_layers` (SNES inner helper) propagates the return via `return save2chunk(...)`.
Each of the three chunk loops (GBF3, SNES, ECNES) was updated to:

```python
mags_ag, mags_non_ag, mags_am, mags_sum = [], [], [], []
# inside loop:
mags_ag.extend(save2chunk(valid_ag_g, chunks_ag, group_idx))
```

The end-of-function `magnitudes` dict uses the accumulated lists directly.
`_mag_from_chunks` is removed entirely.

---

### 2 — `get_all_files` could not discover chunk NC files

`extract_dtype_from_path` checked `os.path.basename(path)` against category patterns.
For chunk files (`chunk_000000.nc` inside `xr_biodiversity_GBF4_SNES_ag_2050_chunks/`),
the basename never matches `xr_` → classified as `Unknown` → dropped.
`get_all_files` also stored `base_name = "chunk_000000"`, which no downstream query would match.

**Fix in `extract_dtype_from_path`:** if the file is a `.nc` inside a `_YYYY_chunks/`
directory, use the **parent directory name** (with `_YYYY_chunks` stripped) as the effective
basename for pattern matching:

```python
parent = os.path.dirname(path)
if path.endswith('.nc') and re.search(r'_\d{4}_chunks$', os.path.basename(parent)):
    base_name = re.sub(r'_\d{4}_chunks$', '', os.path.basename(parent))
else:
    base_name = os.path.basename(path)
```

**Fix in `get_all_files` `_base_name_ext`:** same guard so `base_name` stored in the
DataFrame is `xr_biodiversity_GBF4_SNES_ag` (not `chunk_000000`), matching
`files.query('base_name == "xr_biodiversity_GBF4_SNES_ag"')`.

`manifest.json` files inside chunk dirs are excluded from the chunk-dir logic via the
`.endswith('.nc')` guard and remain `Unknown` → dropped.

---

### 3 — `_score_to_df` (SNES inner helper): per-species xarray loop replaced by BLAS matmul

The original function looped over each species one at a time:
- `.sel(cell=score['species'] == species)` — boolean mask per species
- `.groupby(rl).sum()` + `.to_dataframe()` per species per region level
- Result: O(n_species × n_region_levels) xarray groupby ops, many small DataFrames

For a batch of 10 species with 2 region levels and 4 score types, this is 80 xarray
groupby calls per batch, each doing a full `.compute()` on a dask-backed DataArray.

**Fix:** stack `group_dims` once → `(n_layers, n_cells)` numpy matrix, then:

- **AUSTRALIA**: single BLAS call `vals @ sp_onehot` → `(n_layers, n_sp)` where
  `sp_onehot` is `(n_cells, n_sp)` — one call covers all species and all layers.
- **Regional**: loop over `n_sp ≤ 10` species; for each, `vals[:, mask] @ rg_onehot[mask]`
  → `(n_layers, n_rg)`. The per-species mask limits cells to that species' nonzero footprint,
  keeping each matmul small.
- Build DataFrames from `np.where(agg != 0)` indices — no intermediate DataFrame-per-species.

Key: `score.compute()` is called once at the start, not inside any loop.

---

### 4 — `get_map2json_snes` `chunk_num` parameter: merge N chunk NCs per JS output

With 10 species per chunk NC and ~270 SNES species, each combo generates ~27 JS files.
A `chunk_num` parameter allows grouping N consecutive chunk files per JS output.

**Python side (`get_map2json_snes`):**
The `for chunk_idx in chunk_indices:` loop was replaced by:

```python
for group_start_pos in range(0, len(chunk_indices), chunk_num):
    group = chunk_indices[group_start_pos : group_start_pos + chunk_num]
    page_start = page_starts[group[0]]
    page_end   = page_starts[group[-1]] + len(manifest[str(group[-1])])
    chunk_data = {}
    for chunk_idx in group:          # accumulate all years × all chunks
        for year, chunks_dir in ...:
            ...
    # write one JS per combo, filename: {var_name}_{page_start}_{page_end}.js
```

With `chunk_num=10` and 10 species per NC: each JS covers 100 species, ~3× fewer files.

**Index side (`_write_snes_index`):** the `pages` dict must be grouped by the same
`chunk_num` so Vue constructs the correct filename. Without this fix, Vue reads
`pages["0"] = {start:0, end:10}` and requests `_0_10.js` — which no longer exists.

```python
for group_start_pos in range(0, len(sorted_chunk_keys), chunk_num):
    group_keys = [...]
    page_start = chunk_starts[group_keys[0]]
    page_end   = chunk_starts[group_keys[-1]] + len(chunk_species[group_keys[-1]])
    species    = [sp for k in group_keys for sp in chunk_species[k]]
    pages_out[str(page_idx)] = {'start': page_start, 'end': page_end, 'species': species}
```

`pageSize` is updated to `batch_size × chunk_num`.

**Vue side:** no changes needed — `currentPageInfo = pages[selectPage]` drives
`ensureComboLayer([start, end])` which constructs the filename. The species dropdown
(`_pagedSpecies()`) reads `currentPageInfo.species` which now contains `chunk_num × 10`
species. `totalPages = Object.keys(pages).length` is halved automatically.

The `chunk_num` is set per call site in `save_report_layer` (not in `settings.py`) so it
can be tuned independently for GBF3, SNES, and ECNES without a settings round-trip.

---

## 20260529 — `write_biodiversity_GBF4_SNES_scores`: ~200 GB memory at full resolution diagnosed and fixed

### Context

A res5 profiling run with 100 SNES species reported ~7 GB peak memory for
`write_biodiversity_GBF4_SNES_scores`. The question was why a full-resolution run
(~2,600 species) would consume ~200 GB — far more than a simple NCELLS × species scaling
would predict — and whether the cause was array accumulation across batches.

### Root cause: dense intermediate in `_save_fullsize_layers`

The function `_save_fullsize_layers` was called four times per species batch (ag, non_ag,
am, sum). Each call allocated a fully dense numpy array of shape:

```
[n_am, n_lm, n_lu, batch_size, NCELLS]
```

For `score_am` (the dominant case) with 9 AM types + ALL = 10, 3 lm, 30 lu, batch = 10:

| Resolution | NCELLS | `full_arr` size | After `.stack()` (copy) | Per-batch peak (4 calls) |
|---|---:|---:|---:|---:|
| res5 | ~20,000 | 0.72 GB | 0.72 GB | ~6 GB |
| res1 | ~500,000 | 18 GB | 18 GB | ~72 GB |

The `.stack(layer=['am','lm','lu','species'])` call on a numpy array written via `.loc[]`
is non-contiguous in memory and forces a copy, so `full_arr` (18 GB) and the stacked
result (18 GB) were both live simultaneously — **36 GB per `_save_fullsize_layers` call**.
With 4 calls per batch and Python's reference-counting freeing the arrays only at function
return (no GC lag needed — this is synchronous), the synchronous peak per batch was
**~72 GB**. Residuals from preceding batches not yet freed by the OS pushed observed RSS
to **~200 GB**.

The DataFrame accumulation (`pd.concat` in a loop) was also investigated as a potential
cause but ruled out: the score DataFrames are aggregate tables (species × region × lu
rows, not cell-level), so even with 260 batches the O(N²) copy overhead is at most a few
MB of extra memory — negligible.

### Fix: build only valid layers, skip the dense intermediate

`_save_fullsize_layers` was rewritten to avoid `full_arr` entirely. Instead of allocating
the full `[n_am × n_lm × n_lu × batch × NCELLS]` tensor and selecting valid layers
afterwards, the new implementation iterates directly over `valid_layers` and writes each
layer's 1D array into `out_data`:

```python
n_valid = len(valid_layers)
out_data = np.zeros((n_valid, data.NCELLS), dtype=np.float32)

for i, layer_tuple in enumerate(valid_layers):
    species_val = layer_tuple[species_pos]
    sel_coords = {d: layer_tuple[layer_dims.index(d)] for d in score_layer_dims}
    layer_slice = score.sel(cell=score['species'] == species_val, **sel_coords)
    out_data[i, nz_idx] = layer_slice.values
```

Peak memory per call: `n_valid_layers × NCELLS × 4 bytes`. With typically 50–200 valid
layers at res1, this is **~400 MB per call** — down from **36 GB**.

The previous convention-based `layer_dims[:-1]` / `layer_tuple[-1]` for extracting the
species dimension was replaced with explicit label indexing:

```python
score_layer_dims = [d for d in layer_dims if d in score.dims]
species_pos      = layer_dims.index('species')
```

This makes the function robust regardless of argument order.

### GC lag risk with new code

With plain numpy arrays and no reference cycles, CPython's reference counter frees
`out_data` and `valid_arr` immediately when `_save_fullsize_layers` returns. Even if
OS-level lag held 2–3 batches in RSS, the worst case is: 3 × 400 MB = 1.2 GB — negligible.
The previous worst case was 3 × 72 GB = 216 GB, which matched the observed ~200 GB.

---

## 20260528 — RES5 peak memory updated for NVIS and ECNES biodiversity writers

### Context

The `peak_mb_RES5` table in `luto/tools/write.py` controls write-stage parallelism via:

```python
n_jobs = floor(WRITE_REPORT_MAX_MEM_MB / peak_delta_mb)
```

NVIS and ECNES were re-profiled at RF5.

The profiling data object was:

```text
output/2026_05_20__13_10_03_RF5_2010-2050/Data_RES5.lz4
```

### Profiling results

| Writer | Duration | Peak delta RAM | Peak working set | Status | Summary |
|---|---:|---:|---:|---|---|
| `write_biodiversity_GBF3_NVIS_scores` | 219.58s | 15,447.3 MB | 23,568.7 MB | ok | `data_bio_nvis_snes100_ecnes/20260528_113110/profile_summary.csv` |
| `write_biodiversity_GBF4_ECNES_scores` | 648.08s | 7,317.4 MB | 15,329.6 MB | ok | `data_bio_nvis_snes100_ecnes/20260528_113528/profile_summary.csv` |

The successful ECNES run used `GBF4_ECNES_REGION_MODE = 'NRM'`, matching the current
selected regions `['North East', 'Goulburn Broken']`. An earlier ECNES attempt with
`GBF4_ECNES_REGION_MODE = 'AUSTRALIA'` failed before profiling completed because the
selected regions were NRM names, leaving the filtered ECNES target table empty.

### Code update

`luto/tools/write.py` was updated by rounding the successful peak deltas up to integer MB:

```python
'write_biodiversity_GBF3_NVIS_scores':         15_448,
'write_biodiversity_GBF4_ECNES_scores':         7_318,
```

Previous values were:

```python
'write_biodiversity_GBF3_NVIS_scores':         17_281,
'write_biodiversity_GBF4_ECNES_scores':         7_000,
```

The updated file passed:

```powershell
conda run -n luto python -m py_compile luto\tools\write.py
```

---

## 20260527 — SNES sparse nonzero writer implemented, debugged, and benchmarked

### Context

The earlier SNES write investigation identified dask/xarray regional aggregation as the
main bottleneck and proposed a BLAS-like sparse path. The implemented direction is now:

1. Keep decision variables and impact arrays as in-memory numpy arrays with `cell` as the
   last axis.
2. For each 10-species batch, compute the resfactored vegetation score once.
3. Build sparse pair indices from `np.nonzero(veg_score_np)`.
4. Slice all downstream arrays only at those nonzero cells.
5. Accumulate regional scores with sparse `np.add.at`.
6. Build DataFrames once per species batch from compact accumulators.

This avoids the previous full dense xarray products and `.groupby().sum().to_dataframe()`
inside the region/species loops.

---

### Profile result: sparse path vs previous implementations

100-species SNES profile at RF5, year 2050:

| Implementation | Duration | Peak delta RAM | Notes |
|---|---:|---:|---|
| v3 / older xarray path | 607.42s | 17,878.1 MB | dask/xarray groupby path |
| dense current numpy path | 348.09s | 4,639.0 MB | faster, but still dense-ish |
| sparse nonzero path | 96.16s | 4,541.5 MB | first sparse implementation |
| sparse path after readability refactor | **93.98s** | **4,528.1 MB** | current implementation |
| sparse path after AM-axis alignment fix | **118.51s** | **4,537.7 MB** | validated against archived run output |

Extrapolating the validated 100-species result to all 1,937 SNES species gives
approximately **38.3 minutes per year** (`118.51s × 19.37`). This is still not as clean as
transition `ag2ag`, but it removes the dominant dask/xarray recomputation cost.

---

### Resfactor average is not the main remaining culprit

`get_resfactored_average_fraction` was timed separately for 100 species:

| Component | Time |
|---|---:|
| Resfactor vegetation loading for 100 species | ~25–27s |
| Full sparse SNES write for 100 species | ~94–96s |

So resfactor averaging is meaningful, but not the whole bottleneck. In the sparse writer,
the remaining time is a mix of:

- vegetation resfactor loading
- NetCDF temp writes and final concat
- DataFrame construction/final CSV joins
- valid-layer NetCDF filling

The regional aggregation kernel itself is now tiny.

---

### Sparse aggregation benchmark: `np.add.at` beats pandas groupby

100-species sparse batch:

| Kernel | Duration | Delta working set |
|---|---:|---:|
| `addat_ag` | 0.018s | 2.0 MB |
| `pandas_ag` | 0.418s | 33.7 MB |
| `addat_nonag` | 0.003s | 1.4 MB |
| `pandas_nonag` | 0.055s | 3.6 MB |
| `addat_am` | 0.165s | 10.5 MB |
| `pandas_am` | 5.978s | 38.8 MB |

Conclusion: `DataFrame.groupby` is not competitive for the sparse region-aggregation
kernel. The pandas path is ~18–36× slower depending on type, and allocates more memory.
The current `np.add.at` path should remain the hot aggregation kernel.

---

### xarray `isel` benchmark: selection is cheap, rectangular expansion is expensive

The idea tested: use xarray labels for readability, `isel(cell=nonzero_cells)`, then
multiply and `groupby`.

For 100 species:

| Quantity | Count |
|---|---:|
| true sparse `(species, cell)` nonzero pairs | 17,680 |
| unique nonzero cells | 16,797 |
| xarray selected rectangle (`species × unique_cell`) | 1,679,700 |

Timing:

| Kernel | Duration |
|---|---:|
| `xarray_isel_only` | 0.004s |
| `xarray_ag_groupby` | 0.443s |
| `xarray_nonag_groupby` | 0.114s |
| `xarray_am_groupby` | 4.472s |

Conclusion: `isel` itself is cheap, but selecting the union of nonzero cells expands the
work by ~95× compared with true sparse species-cell pairs. xarray is useful for setup and
labels, but not for the SNES inner loop.

---

### Readability refactor applied

The sparse numpy implementation was hard to read because region accumulation and NetCDF
layer filling were embedded directly in the species loop. The current implementation keeps
the sparse numpy performance path but extracts the indexing-heavy sections into named
helpers inside `write_biodiversity_GBF4_SNES_scores`:

- `_load_sparse_veg_scores`
- `_empty_region_accumulators`
- `_take_sparse_inputs`
- `_accumulate_sparse_region_scores`
- `_fill_sum_nc_scores`
- `_fill_ag_nc_layer`
- `_fill_am_nc_layer`
- `_fill_non_ag_nc_layer`

The main species loop now reads as orchestration: load sparse vegetation, accumulate
regional scores, build CSV frames, fill NetCDF layers, save temp chunks.

Profile after this refactor:

| Metric | Value |
|---|---:|
| Duration | 93.98s |
| Peak delta RAM | 4,528.1 MB |
| Peak working set | 15,094.8 MB |
| Status | ok |

No speed regression was observed; the result is slightly faster than the prior sparse
profile within normal run-to-run variation.

---

### AM-axis alignment bug found and fixed

The archived production run at commit `0e3424f74d62e23709c1c9b4274e1d6f63363f3e`
used the xarray path:

```python
xr_gbf4_am_s = veg_score_r * am_impact_amr * am_dvar_amrj
```

That path is slower, but xarray automatically aligns by coordinate label. The sparse
implementation converts to NumPy arrays with `.values`, so all dimension alignment becomes
positional and must be made explicit.

The bug: `am_impact_amr.unstack()` sorts the MultiIndex level for `am`, producing:

```text
['AgTech EI', 'Asparagopsis taxiformis', 'Biochar', 'HIR - Beef', 'HIR - Sheep',
 'Onshore Wind', 'Precision Agriculture', 'Savanna Burning', 'Utility Solar PV']
```

while `am_dvar_amrj` remains in data order:

```text
['Asparagopsis taxiformis', 'Precision Agriculture', 'Savanna Burning', 'AgTech EI',
 'Biochar', 'HIR - Beef', 'HIR - Sheep', 'Utility Solar PV', 'Onshore Wind']
```

The LU axis was already aligned, but the AM axis was not. This caused the sparse
implementation to multiply the wrong AM impact by the wrong AM decision variable. The clue
was that an `ALL` agricultural-management score for a species became negative, even though
the positive Savanna Burning / HIR contributions should dominate. Individual renewable
management scores can legitimately be negative because renewables reduce SNES biodiversity
scores, but the total `ALL` AM score should match the archived xarray result.

Fix applied before converting to NumPy:

```python
am_vals = am_dvar_amrj.coords['am'].values
lu_ag_vals = ag_dvar_mrj.coords['lu'].values
am_impact_amr = am_impact_amr.reindex(am=am_vals, lu=lu_ag_vals, fill_value=0)
```

This restores the label alignment that the archived xarray implementation provided
implicitly.

---

### Validation against archived output

The current sparse SNES profile was rerun for the first 100 species and compared against
the archived production output from:

```text
output/2026_05_26__11_58_52_RF5_2010-2050/out_2050
```

The archive was generated from commit:

```text
0e3424f74d62e23709c1c9b4274e1d6f63363f3e
```

After filtering the archive to the same 100 profile species:

| CSV | Current rows | Archive rows | Key matches | Current-only | Archive-only |
|---|---:|---:|---:|---:|---:|
| `biodiversity_GBF4_SNES_scores_2050.csv` | 20,783 | 20,783 | 20,783 | 0 | 0 |
| `biodiversity_GBF4_SNES_sum_scores_2050.csv` | 1,960 | 1,960 | 1,960 | 0 | 0 |

Maximum absolute score differences are small and consistent with float32/order-of-sum
differences in the sparse path:

| CSV / Type | Max absolute score difference |
|---|---:|
| detailed CSV — Agricultural Management | 0.109375 |
| detailed CSV — Non-Agricultural Land-use | 0.093750 |
| detailed CSV — Agricultural Land-use | 24.000000 |
| sum CSV — ag-man | 0.109375 |
| sum CSV — non-ag | 0.093750 |
| sum CSV — ag | 25.000000 |
| sum CSV — ALL | 24.000000 |

Spot checks after the fix:

| Species | AM | Current | Archive | Difference |
|---|---|---:|---:|---:|
| `Acanthophis hawkei` | ALL | 214,972.546875 | 214,972.625000 | -0.078125 |
| `Acanthophis hawkei` | Savanna Burning | 131,123.828125 | 131,123.828125 | 0 |
| `Acanthophis hawkei` | HIR - Beef | 72,385.476562 | 72,385.484375 | -0.007812 |
| `Acanthophis hawkei` | Onshore Wind | -805.271973 | -805.271973 | 0 |
| `Acacia crombiei` | ALL | 112,579.671875 | 112,579.671875 | 0 |
| `Acacia ammophila` | ALL | 20,845.052734 | 20,845.050781 | 0.001953 |

Conclusion: the current sparse implementation is now equivalent to the archived xarray
implementation for the profiled species set, with only tiny floating-point differences.

---

### NetCDF writing decision: keep `save2tmp` / `concat_tmp2nc` for now

An incremental NetCDF writer was considered to append directly along `layer` and avoid
the temp-folder concat stage. It is technically feasible if it writes the same
CF-compressed MultiIndex schema expected by report generation:

```python
cfxr.decode_compress_to_multi_index(xr.open_dataset(path, chunks={}), 'layer')['data']
```

The final NC also must keep `data` chunked as `(layer=1, cell=NCELLS)`, matching current
report access patterns.

However, after review, the preference is to keep the existing `save2tmp` /
`concat_tmp2nc` path because it is already compatible with downstream report code and is
easier to reason about. The attempted incremental writer was reverted.

---

## 20260527 — SNES write bottleneck: dask recomputation vs BLAS GEMM plan

### Context

Production run `2026_05_26__11_58_52_RF5_2010-2050` (RF5, 5 years) showed that
`write_biodiversity_GBF4_SNES_scores` dominated the entire run:

| Phase | Wall time | Fraction of total |
|---|---:|---:|
| Write phase total | 4h 31min | — |
| → SNES write | **3h 35min** | **79%** |
| Report phase total | 2h 57min | — |
| → SNES map layers | **2h 40min** | **91%** |
| **Total run** | **8h 12min** | — |
| → **Total SNES overhead** | **~6h 15min** | **~76%** |

Peak RAM during SNES write: **235.8 GB** (17:12:22). Pre-SNES baseline: ~70 GB.

---

### Root cause: 1,552 full-array dask recomputations

The SNES loop processes 1937 species in 194 batches of 10. Per batch, per 2 region levels,
it triggers `.groupby().sum().to_dataframe()` on 4 large arrays, plus an AUS aggregate:

```
194 batches × 2 region levels × 4 array types × 2 (region + AUS) = 1,552 dask .compute() calls
```

Each call materialises `xr_gbf4_am_s` of shape `(10 species × 9 am × 3 lm × 28 lu × 186648 cells)` ≈ **5.7 GB**.
Total data movement per year: ~8.8 TB.

Compare `write_transition_ag2ag` (257s, all land uses at once): it uses `process_chunks` with
a BLAS GEMM accumulator — **1 aggregation pass** over ~100 MB. No dask compute loop.

---

### Memory trace finding: xr IS freed between batches; DataFrames are the floor

The SNES memory trace (100-species profiling run, v3) shows an **oscillatory** pattern —
not monotonically increasing. Memory spikes up for each batch and drops almost fully back
to near-baseline after each batch completes. This proves:

1. **xr arrays ARE freed between batches** (Python GC reclaims them on loop variable reassignment)
2. The **per-batch xr spike** (~16 GB, dominated by `xr_gbf4_am_s`) is the peak driver
3. The **accumulated DataFrames** add a slowly rising baseline (~230 MB per batch before filtering)

Implication for `peak_mb_RES5`: the earlier extrapolation of 346,299 MB assumed linear
DataFrame accumulation as the driver. That was wrong. The xr spike (~16 GB) is constant
per batch regardless of total species count. For 1937 species, the per-year peak is
similar to the 100-species profile: **~17,878 MB**.

**`peak_mb_RES5['write_biodiversity_GBF4_SNES_scores']` corrected from 346,299 → 17,878 MB.**

With `WRITE_REPORT_MAX_MEM_MB = 65,536 MB`:
- Old (1 MB placeholder): `n_jobs = 65,536` → all 5 years in parallel
- New (17,878 MB): `n_jobs = 3` → 3 years in parallel, then 2

---

### Short-term fix applied: zero-row DataFrame filter

Before each `extend()` call in the species-batch loop, filter out rows where
`'Area Weighted Score (ha)' == 0`:

```python
ag_frames.extend([
    ag_df_region.query('`Area Weighted Score (ha)` != 0'),
    ag_df_AUS.query('`Area Weighted Score (ha)` != 0'),
])
```

Applied to all 4 frame lists in SNES, ECNES, and NVIS (12 `extend` calls total).
Since the SNES layers are very sparse, most species have zero score in most
(region, lm, lu) combinations. This reduces the accumulated DataFrame floor from
~44 GB to ~2 GB for the full 1937-species run, lowering the baseline from which
each xr spike launches.

---

### BLAS GEMM plan (not yet implemented)

The `process_chunks` function used by `write_transition_ag2ag` achieves its speed via:

```
chunk.reshape(n_combos, chunk_cells) @ onehot[chunk_cells, n_regions]  →  (n_combos, n_regions)
```

The same form applies to SNES scoring. For ag:

```
score_ag[s, lm, lu, region] = Σ_cell  veg[s, cell] × impact[lu] × dvar[lm, lu, cell] × indicator[cell → region]
                             = combined[s, lm, lu, :].reshape(n_combos, NCELLS) @ onehot
```

For am (the dominant cost, shape `species × am × lm × lu × cell`):

```
score_am[s, am, lm, lu, region]
    = Σ_cell  veg[s, cell] × am_impact[am, lu, cell] × am_dvar[am, lm, lu, cell] × indicator[cell → region]
    = combined_am.reshape(n_combos_am, chunk_cells) @ onehot_chunk
```

**Proposed restructuring** — swap the loop axes:

| | Current | Proposed |
|---|---|---|
| Outer loop | species batch (194×) | species batch (194×) — same |
| Inner loop | 2 region levels × 8 `groupby().sum()` dask triggers | cell chunks (46×, 4096 cells each) |
| Array per step | `(10, am, lm, lu, 186648)` = 5.7 GB materialised | `(10, am, lm, lu, 4096)` = 124 MB chunk |
| Aggregation | xr `groupby().sum()` on full array | BLAS GEMM `@ onehot_chunk` |
| Dask calls | 1,552 `.compute()` | 0 (pure numpy) |

Per-batch cell-chunk GEMM:

```python
# onehot pre-computed once per region level: (NCELLS, n_regions), float32
for sp_idx, species_batch in enumerate(...):
    snes_chunk_all = snes_layers[sp_batch, :]   # (n_sp, NCELLS) — sparse→dense once per batch

    for cell_start in range(0, NCELLS, chunk_size):
        sl = slice(cell_start, cell_start + chunk_size)
        combined_am = (snes_chunk_all[:, sl][:, None, None, None, :]   # (n_sp, 1, 1, 1, c)
                       * am_impact[:, :, sl][None, :, None, :, :]       # (1, am, 1, lu, c)
                       * am_dvar[:, :, :, sl][None, :, :, :, :])        # (1, am, lm, lu, c)
        # → (n_sp, am, lm, lu, chunk_cells) = ~124 MB
        accum_am += combined_am.reshape(-1, chunk_cells) @ onehot_sl   # BLAS GEMM → (n_combos, n_regions)

    # Convert accumulator to DataFrame ONCE per batch (not 8×)
    df_ag = pd.DataFrame(accum_ag.reshape(-1, n_regions), ...).melt(...)
```

**Expected impact:**

| Metric | Current | After BLAS GEMM |
|---|---|---|
| Dask `.compute()` calls | 1,552 | 0 |
| Memory per step | 5.7 GB | 124 MB |
| Estimated time per year | ~196 min | ~20–40 min (similar to ECNES at 655s) |
| Peak RAM per year | ~18 GB Δ | ~2 GB Δ (no full materialisation) |

Same restructuring can be applied to ECNES and NVIS.

**Status: planned, not yet implemented.**

---

## 20260525 — Biodiversity input array sparsity and threading for GBF3/4 write loops

### Context

The GBF3/4 biodiversity write functions loop over vegetation groups or species,
multiplying spatial arrays by land-use decision variables. Threading was investigated
to speed up these loops. Sparsity of input NC files was profiled to evaluate
sparse array pre-cooking as a complementary strategy.

### NVIS Threading Benchmark (`write_biodiversity_GBF3_NVIS_scores`)

| Config | Duration | Peak RAM | Notes |
|--------|----------|----------|-------|
| Serial (baseline) | 551s | ~9.2 GB | step_7 profiler |
| n_jobs=8 threading | 360s | 14.8 GB | ~1.5× speedup, +5.9 GB overhead |

**Root cause of limited speedup**: `save2tmp` NC writes serialised behind `threading.Lock`;
`netCDF4` C library not thread-safe (global ID registry).

**Fix applied**: Pre-load full NVIS array via `xr.open_dataarray(...).load()` once before
the parallel loop — workers receive plain numpy slices, zero file I/O per thread.
Eliminates `RuntimeError: NetCDF: Not a valid ID` from stale thread-local handles.

### Input Array Sparsity

| File | Shape | Dense size | Sparsity | COO size | Compression | Build time |
|------|-------|-----------|----------|----------|-------------|------------|
| `bio_GBF3_NVIS_MVG.nc` | (30, 6.9M) | 0.8 GB | 95.0% | ~125 MB | 7× | 0.5s |
| `bio_GBF4_ECNES.nc` | (101, 2, 6.9M) | 5.6 GB | 99.6% | ~88 MB | 64× | 17s |
| `bio_GBF4_SNES.nc` | (2021, 2, 6.9M) | 112 GB | 99.65% | ~1,565 MB | 72× | 447s |

COO size estimated as `nonzero × (4 bytes data + 4 bytes × ndim coords)`.
SNES/ECNES build time measured reading one species slice at a time (avoids
materialising the full dense array).

### Recommendations

| Array | Action | Reason |
|-------|--------|--------|
| **NVIS** | Keep `.load()` dense | 0.8 GB fine; 7× compression not worth overhead |
| **ECNES** | Pre-cook `bio_GBF4_ECNES_sparse.npz` in `dataprep.py` | 88 MB COO, 17s build, 64× savings; loads in seconds at runtime |
| **SNES** | Pre-cook `bio_GBF4_SNES_sparse.npz` in `dataprep.py` | 112 GB dense impossible; 1.5 GB COO, 447s build (one-time); loads in seconds |

Pre-cooked `.npz` files are plain numpy arrays — fully thread-safe, no file locks,
instant per-species indexing in parallel workers.

### SNES Runtime Projection

| Strategy | Estimated time (2021 species) |
|----------|-------------------------------|
| Serial loop | ~36,700s (~10h) |
| n_jobs=8 threading (dense) | Impossible — 112 GB RAM |
| n_jobs=8 threading (sparse `.npz`) | ~4,600s (~1.3h, estimated) |

Pre-cooking sparse arrays is a prerequisite for any viable SNES parallelism.

---

## 20260523 — Biodiversity regional scoring: xarray groupby bottleneck and optimisation

### Context

The upstream pipeline pre-computes weighted habitat area scores for three biodiversity
datasets across all region levels × resfactors, writing the results to CSV so LUTO can
do a simple lookup at runtime instead of recomputing at every model run.

**Scale of the problem:**

| Dataset | Species / groups | Presence types | Region levels | Resfactors |
|---|---|---|---|---|
| NVIS (MVG + MVS) | ~770 vegetation groups | 1 | 5 (incl. IBRA_SUB) | 10 |
| SNES | ~2,055 threatened species | 2 (LIKELY, MAYBE) | 5 | 10 |
| ECNES | ~400 threatened communities | 2 (LIKELY, MAYBE) | 5 | 10 |

Region levels: `AUSTRALIA` (1 group), `NRM` (56), `STATE` (9), `IBRA_REG` (85), `IBRA_SUB` (410).
Spatial domain: 6,956,407 NLUM cells at resfactor 1; fewer at higher resfactors.

For each `(species, resfactor, region_level, region)` combination, four weighted-area scores
are computed by groupby-summing a 1D species presence array multiplied by per-cell weights:

```
ALL_HA               = sum(species_arr × cell_ha)                           per region
IN_LUTO_HA           = sum(species_arr × cell_ha × biodiv_degrade)         per region
NATURAL_OUT_LUTO_HA  = sum(species_arr × cell_ha × out_luto_natural_mask)  per region
NON_NATURAL_OUT_LUTO_HA = sum(species_arr × cell_ha × out_luto_nonnat_mask) per region
```

Per-task wall time was ~5 s, identical to a sequential for-loop, regardless of how many
workers were used. Profiled via synthetic benchmarks in
`Scripts/work_in_progress/trace_task/trace_bottlenecks.py` and `trace_groupby.py`.

---

### Finding 1 — `xr.groupby` is 28–45× slower than `pd.groupby` for this pattern

The original implementation built an `xr.Dataset` with 4 separate DataArray multiplications
and 4 separate `.groupby('region').sum('cell')` calls:

```python
arr = arr.assign_coords({'region': ('cell', region_labels)})
xr.Dataset({
    'ALL_HA':             (arr * area          ).groupby('region').sum('cell'),
    'IN_LUTO_HA':         (arr * area * degrade).groupby('region').sum('cell'),
    ...
}).compute().to_dataframe()
```

Internal breakdown (6.9 M cells, NRM 56 regions, rf=1):

| Operation | Time |
|---|---:|
| `arr * area` (DataArray multiply, once) | 9.8 ms |
| — computed ×4 in original code | 39 ms total |
| One `xr.groupby('region').sum('cell')` | **3,687 ms** |
| — called ×4 in original code | **14,750 ms** total |
| Full original `compute_region_scores` | **14,717 ms** |
| Equivalent `pd.DataFrame.groupby().sum()` | **513 ms** |

The xarray overhead is intrinsic — coordinate assignment, lazy graph construction, and
Python-level dispatch add ~3.7 s per groupby call regardless of group count. AUSTRALIA
(1 group, purely a sum) took 19,839 ms vs 430 ms in pandas — confirming the bottleneck
is overhead, not the aggregation itself.

**Fix:** replaced `compute_region_scores` entirely with a pandas implementation.
`arr * area` is computed once and reused across all four weighted sums.

---

### Finding 2 — Integer region codes give an additional 4.5× pandas speedup

Pandas groupby on string labels builds a hash map over all 6.9 M cells. Replacing strings
with integer codes (via `pd.factorize` + `pd.Categorical`) triggers pandas' `np.bincount`
path — O(n) with no hash map. Full 5-variant benchmark (6.9 M cells, NRM 56 regions):

| Variant | Time (ms) | vs pandas-str |
|---|---:|---:|
| A. xarray Dataset, 4 groupbys, string labels (original) | 13,251 | 14.9× slower |
| B. xarray DataArray, 1 groupby, string labels | 3,862 | 4.3× slower |
| C. xarray DataArray, 1 groupby, int codes | 3,184 | 3.6× slower |
| D. pandas DataFrame, string labels | 888 | 1.0× baseline |
| **E. pandas DataFrame, int codes + restore** | **197** | **4.5× faster** |

Note: int codes barely help xarray (C vs B: ~1.2×) — xarray's coordinate machinery dominates
regardless of label type. All gains come from switching to pandas first (A→D: 15×), then
switching labels to integers (D→E: 4.5×).

**Combined speedup original → optimised: ~67×** (13,251 ms → 197 ms).

**Implementation:** region labels are pre-factorized once into 2D int arrays
(`region_int_2D`) at setup. `rf_meta` stores `pd.Categorical.from_codes(codes, categories)`
per rf per region — the Categorical carries both int codes (for fast groupby) and the
string categories array (restored automatically in the groupby output, no manual mapping).
The 2D layout is required because the spatial coarsening mask (`masks[rf]`) is inherently
2D (it selects the centre pixel of each coarse spatial block).

---

### Finding 3 — Redundant resfactoring in the NVIS loop (5× waste per species per rf)

The original NVIS task-building loop had order `species → region → rf`. Inside each task,
`get_resfactored_average_fraction(species_arr, rf, mask)` was called once per (region, rf)
pair — meaning the **identical coarsened array** was computed 5 times (once per region level)
for each (species, rf) combination.

Cost of one resfactoring call at rf=5: ~147 ms. Wasted per species:
4 redundant calls × 147 ms = **588 ms per (species, rf)**.

Benchmarked end-to-end for all 5 regions at rf=5:

| Structure | Total cost |
|---|---:|
| Original: 5 separate tasks, 5× resfactoring | 4,162 ms |
| Fixed: 1 task, 1× resfactoring, all regions in one pass | 219 ms |
| **Speedup** | **19×** |

**Fix:** restructured loop to `species → rf → [all 5 regions inside single task]`.
The resfactored array is computed once and passed to 5 sequential `compute_region_scores`
calls, whose results are concatenated before returning.

---

### Finding 4 — `xr.coarsen` vs numpy reshape for spatial block-averaging

`get_resfactored_average_fraction` used `xr.DataArray.coarsen(x=rf, y=rf).mean()` to
block-average the 2D species/weight array. Replaced with pure numpy:

```python
arr_2d.reshape(h_blocks, rf, w_blocks, rf).mean(axis=(1, 3))
```

Benchmark (6.9 M cells, synthetic species array, 5 repeats):

| rf | xr.coarsen (ms) | numpy reshape (ms) | speedup |
|---|---:|---:|---:|
| 2 | 256 | 160 | 1.6× |
| 5 | 148 | 117 | 1.3× |
| 10 | 119 | 103 | 1.2× |

Moderate gain (1.2–1.6×). Not the dominant bottleneck but applied throughout since
`get_resfactored_average_fraction` is called for every species at every resfactor.

---

### Finding 5 — Process backend (loky) serialises large globals per task

After applying Findings 1–4, per-task wall time remained ~5 s with the loky process
backend. Root cause: cloudpickle serialises every global variable the task function
references. The `rf_meta` dict contains 10 rf values × 5 numpy arrays × ~7 M cells
— hundreds of MB pickled for **every** task dispatched.

With `prefer='threads'`: all threads share the same process memory. `rf_meta`, `masks`,
`region_int_2D` are zero-copy shared globals. The dominant operations (numpy array math,
pandas Cython groupby) all release the Python GIL, so threads achieve genuine parallelism.

**Optimal thread count: 8** (empirical — 8 < 16 < 32 < 128 in wall time).

Root cause of the ceiling: **memory bandwidth saturation**. Each task loads a ~28 MB
species array (at rf=1) plus shared weight arrays. At 8 concurrent threads, the memory
bus approaches capacity; adding more threads queues them at the memory controller. On
the 192-core server (multi-socket NUMA), threads beyond one NUMA node also incur
remote-memory penalties (~2–4× slower). A secondary contributor is GIL-bound Python
overhead in `pd.DataFrame({...})` and `.reset_index()` — the 8-thread optimum implies
roughly 1/8 of task time holds the GIL.

---

### Finding 6 — SNES/ECNES two-phase structure made Phase 1 sequential

Original structure:
- **Phase 1** (sequential): for each presence type, compute resfactored arrays for all
  ~2,055 species × 10 rf = 20,550 calls to `get_resfactored_average_fraction`, stored
  in a `snes_rf[(presence, rf)]` dict of `(n_species, n_cells)` numpy arrays.
- **Phase 2** (parallel, 100 tasks): groupby using pre-stored 2D arrays, passing the
  full `(n_species, n_cells)` DataArray to `compute_region_scores` per task.

Extrapolated Phase 1 cost (5 real species at rf=5 = 732 ms → 2,055 species × 10 rf × 2 presence):
**~42 min sequential** just for resfactoring, before Phase 2 starts.

**Fix:** merged into a single parallel loop (mirrors NVIS structure). `.compute()` is
called once per presence type to materialise the dask array, then each task receives
a 1D numpy slice for one species. Resfactoring and groupby both run inside the parallel
pool. The `snes_rf`/`ecnes_rf` pre-compute dicts are eliminated entirely.

---

### Summary

| Optimisation | Speedup |
|---|---|
| `xr.groupby` → `pd.groupby` in `compute_region_scores` | **29–45× per call** |
| String region labels → `pd.Categorical` int codes | **4.5× per call** |
| Eliminate redundant resfactoring in NVIS (5× → 1× per task) | **19× per (species, rf)** |
| `xr.coarsen` → numpy reshape in `get_resfactored_average_fraction` | 1.2–1.6× |
| SNES/ECNES: parallelise resfactoring (was sequential Phase 1) | ~42 min → parallel |
| loky process backend → threads (shared globals, no pickling) | eliminates ~5 s/task overhead |
| Optimal N_JOBS = 8 (memory bandwidth ceiling, NUMA-local) | — |

Benchmark artefacts: `Scripts/work_in_progress/trace_task/`

---

## 20260522 — Upstream pre-computation of NVIS / SNES / ECNES targets for all resfactors × regions

### Background

Previously, LUTO computed resfactored biodiversity targets at runtime inside `data.py`
(via `get_resfactored_average_fraction`, `_recompute_nvis_targets_at_rf`, and equivalent
loops for SNES/ECNES). This was expensive and required the full spatial arrays to be
loaded and block-averaged on every run.

`script_5_2` (upstream data pipeline) now pre-computes all combinations of
**5 region levels × 10 resfactors (RF 1–10)** and bakes them into the input CSVs directly.

---

### New output files

| Dataset | Output file | Key dimensions |
|---------|-------------|----------------|
| NVIS (MVG + MVS) | `BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv` | group × region_level × region × resfactor |
| SNES | `bio_DCCEEW_SNES_target_ALL_REGIONS.csv` | SCIENTIFIC_NAME × region_level × region × resfactor |
| ECNES | `bio_DCCEEW_ECNES_target_ALL_REGIONS.csv` | COMMUNITY × region_level × region × resfactor |

Region levels available: `AUSTRALIA`, `NRM`, `STATE`, `IBRA_REG`, `IBRA_SUB` (NVIS only).

---

### How resfactoring is applied upstream

For each resfactor RF, all components are block-averaged via `get_resfactored_average_fraction`:

- species / vegetation fraction array (`arr`)
- `cell_ha` — multiplied by RF² to give correct total block area at all RFs
- `biodiv_degrade_ly × idx_in_LUTO` — degradation weight inside LUTO
- `idx_out_LUTO_natural` and `idx_out_LUTO_non_natural` — outside-LUTO fractions

Block-averaging all components (rather than sampling the centre pixel) ensures spatial
consistency at every resolution. Degradation weights come from `HABITAT_CONDITION.csv`
(`USER_DEFINED`, pre-normalised to lu=23 = 1.0 by script_4, with policy overrides
lu=2, 6, 15 = 0.7). Parallelised via `joblib.Parallel` (32 workers).

---

### Implications for LUTO `data.py`

Each CSV now carries a `resfactor` column. At runtime, `data.py` should simply filter to
`resfactor == settings.RESFACTOR` instead of performing spatial aggregation:

- `_recompute_nvis_targets_at_rf` — **no longer needed**; lookup replaces spatial recomputation
- SNES / ECNES NRM loops that called `get_resfactored_average_fraction` — **no longer needed**
- `get_NVIS_resfactord_array` — may still be required for layer construction (the spatial
  `.nc` arrays are separate from the target CSVs), but target values come from the CSV lookup
- Old single-resfactor SNES file `bio_DCCEEW_SNES_target` is superseded by
  `bio_DCCEEW_SNES_target_ALL_REGIONS.csv`

Loading code in `dataprep.py` (line ~188, `bio_DCCEEW_SNES_target`) and `data.py` must be
updated to read from the new `_ALL_REGIONS` files and filter by `resfactor`.

---

## 20260521 — Write phase dynamic tier scheduler and RF5 benchmark

### Context

Following the `process_chunks` optimisation (see 20260520), the write phase was
re-profiled and a new scheduling strategy was designed to balance parallelism against
peak memory. Three full 5-year RF5 runs were compared:

| Run | Strategy | Wall time | Peak RAM |
|---|---|---:|---:|
| `2026_05_20__14_42_20` | All parallel, 12 workers (old) | **16.4 min** | **81.1 GB** |
| `2026_05_20__20_33_57` | Binary high/low split, high=n_jobs=1 | 41.3 min | 34.5 GB |
| `2026_05_21__11_50_58` | Dynamic tier scheduler (new) | **23.3 min** | **49.2 GB** |

The binary split cut peak RAM by 57% but was 2.5× slower — all 8 high-mem functions
× 5 years = 40 tasks ran sequentially. The tier scheduler recovers most of that speed
while keeping peak RAM 39% lower than all-parallel.

---

### New RF5 benchmark profile (yr_cal = 2050)

Data: `output/2026_05_20__13_10_03_RF5_2010-2050/Data_RES5.lz4`  
Baseline data object: ~8,387 MB. All functions profiled sequentially with GC between each.

| Function | Time (s) | Peak Δ (MB) | Peak absolute (MB) |
|---|---:|---:|---:|
| `write_transition_nonag2ag` | 30 | **7,558** | 15,802 |
| `write_transition_ag2ag` | 265 | **6,544** | 14,718 |
| `write_biodiversity_quality_scores` | 203 | **6,101** | 14,519 |
| `write_economics` | 220 | 4,914 | 12,992 |
| `write_ghg` | 53 | 3,189 | 11,517 |
| `write_transition_ag2nonag` | 127 | 3,167 | 11,499 |
| `write_quantity` | 111 | 2,916 | 10,924 |
| `write_water` | 40 | 2,496 | 10,732 |
| `write_biodiversity_GBF2_scores` | 21 | 1,768 | 10,112 |
| `write_dvar_and_mosaic_map` | 51 | 941 | 8,809 |
| `write_dvar_area` | 31 | 1,692 | 10,087 |
| `write_area_transition_start_end` | 101 | 1,392 | 9,778 |
| `write_renewable_production` | 21 | 718 | 9,029 |
| `write_crosstab` | 1 | 13 | 8,320 |
| GBF3/4/8 (5 funcs) | ~0 | ~0 | 8,387 |

**Total sequential time (one year): ~1,277 s (~21 min)**

---

### Misclassifications in old binary split

Two functions were assigned to the wrong group in `write_output_single_year`:

| Function | Old group | Actual peak Δ | Correct group |
|---|---|---:|---|
| `write_biodiversity_quality_scores` | `low_mem` | **6,101 MB** | `high_mem` |
| `write_biodiversity_GBF2_scores` | `high_mem` | 1,768 MB | `low_mem` |

`write_biodiversity_quality_scores` was the 3rd heaviest function and ran silently
alongside other low_mem tasks, causing uncontrolled memory spikes.

---

### Root cause: `write_biodiversity_quality_scores` high memory

Loops over 7 `BIO_QUALITY_LAYERS` backends and appends 4 large xr arrays per backend
to accumulator lists. After the loop, 28 arrays (7 × 4) are alive simultaneously before
`xr.concat` creates combined arrays and `.compute()` materialises all of them:

```
loop iteration 7 ends → 28 arrays alive
xr.concat(...)         → 4 combined arrays (28 originals still referenced)
.compute()             → all materialised simultaneously → 6.1 GB peak
```

Using `del` on within-iteration intermediates does not help — the accumulator lists
hold references across all 7 iterations. The fix requires writing per-backend NC files
inside the loop and discarding each array before the next iteration.

Also notable: **4,131 MB final Δ** — the combined xr arrays are not released after
return, accumulating residual across years.

---

### Root cause: `write_transition_nonag2ag` heavier than `write_transition_ag2ag`

Counterintuitive: `write_transition_nonag2ag` (7,558 MB) exceeds `write_transition_ag2ag`
(6,544 MB) despite nonag→ag transitions being entirely zero in this scenario.

`get_transition_matrix_nonag2ag(separate=True)` returns a **nested dict** — one sub-dict
per non-ag land use, each containing the full ag-transition cost-type breakdown:

```
{9 non-ag LUs} × {N_cost_types each} = 9× more entries than ag2ag's flat N_cost_types
```

`np.stack(list(values()))` materialises all 9 × N_cost_types matrices simultaneously.
After `unstack` and `add_all`, the full `(N_nonag_lu+1) × (N_cost_types+1) × NCELLS × (N_ag_lu+1)`
tensor is allocated and computed entirely in memory — all zeros, because the model
currently prohibits nonag→ag transitions. The heavy allocation is structural, not data-driven.

---

### Dynamic tier scheduler implementation

Replaced the binary high/low split with a budget-driven n_jobs formula:

```python
n_jobs = floor(WRITE_REPORT_MAX_MEM_MB / peak_delta_mb)
```

- `WRITE_FUNC_PEAK_MB` dict added at module level — maps each write function to its
  profiled peak Δ MB at RF5
- `write_output_single_year` now returns `[(delayed_task, peak_mb), ...]` — flat
  annotated list instead of separate high_mem/low_mem lists
- `write_data` groups tasks by computed n_jobs and runs each tier sequentially,
  most constrained first
- `WRITE_PARALLEL` setting removed — parallel is always used
- `WRITE_REPORT_MAX_MEM_GB` renamed to `WRITE_REPORT_MAX_MEM_MB` (value = 64 × 1024)
  to allow direct use without unit conversion. Updated in `create_report_layers.py`
  (`mem_per_worker` divisor changed from `1e9` → `1e6`) and `create_grid_search_tasks.py`

Example tier breakdown with `WRITE_REPORT_MAX_MEM_MB = 65536`:

| n_jobs | Functions (5 years each) |
|---:|---|
| 8 | `write_transition_nonag2ag` (7,558 MB) |
| 10 | `write_transition_ag2ag` (6,544), `write_biodiversity_quality_scores` (6,101) |
| 13 | `write_economics` (4,914) |
| 16 | everything else (≤ 3,189 MB) |

---

### Windows loky spawn overhead between tiers

Each `Parallel(...)` call on Windows creates a fresh loky process pool (no `fork()`).
Pool creation and teardown costs ~3–10 s per tier transition. With ~5 distinct tiers
there is ~25 s of unavoidable overhead.

`prefer='threads'` was tested but reverted — a prior run encountered a pickle error
suggesting the loky backend was being invoked despite the thread preference. Thread-based
parallelism would eliminate spawn overhead entirely (threads share memory, no pickling),
and the write workload is largely GIL-safe (numpy/xarray/NetCDF I/O all release the GIL).
Further investigation needed to identify the pickle root cause before enabling threads.

The remaining gap between tier-scheduler (23.3 min) and all-parallel (16.4 min) is
explained by: (a) the n_jobs=1-equivalent tasks for the 3 heaviest functions, and
(b) the inter-tier pool spin-up cost. Interleaving tiers within a single pool (proper
work-stealing scheduler) would close this gap but requires custom dispatch logic beyond
joblib's standard API.

---

## 20260520 — Write phase profiling and `process_chunks` optimisation

### Context

A full write-phase profile was run on run `2026_05_18__16_11_02_RF5_2010-2050_hard_dual_const`
to identify where time and memory are spent. Each write function was profiled individually
for `yr_cal=2050` using `trace_mem_usage`.

### Per-function profile (yr_cal = 2050)

| Function | Time | Peak Memory |
|---|---|---|
| `write_transition_ag2ag` | **30.6 min** | — (file conflict) |
| `write_transition_ag2nonag` | **13.7 min** | 1,736 MB |
| `write_biodiversity_quality_scores` | 3.8 min | **5,910 MB** |
| `write_economics` | 2.1 min | — (file conflict) |
| `write_area_transition_start_end` | 1.9 min | 1,355 MB |
| `write_ghg` | 1.0 min | 3,925 MB |
| `write_water` | 45 s | 2,326 MB |
| `write_transition_nonag2ag` | 38 s | **6,837 MB** |
| `write_biodiversity_GBF2_scores` | 26 s | 1,901 MB |
| `write_renewable_production` | 23 s | 718 MB |
| All GBF3/4/8 functions | <1 s | 0 MB (targets off in this run) |

### Root cause of `write_transition_ag2ag` slowness

`write_transition_ag2ag` calls `process_chunks` 4 times (area, cost, GHG, water).
Inside each call, the original implementation did:

```python
chunk_df = trans_xr.isel(cell=sl).compute().to_dataframe(value_col).reset_index()
```

For the area array with dims `[From-ws(3), From-lu(29), To-ws(3), To-lu(29), cell]`,
`to_dataframe()` creates a full Cartesian-product DataFrame per chunk:
**3 × 29 × 3 × 29 × 4096 = 31 M rows per chunk**.

With ~68 chunks per call and 4 calls = **~8.4 billion rows** materialised and grouped.

### Fix: BLAS matmul accumulator in `process_chunks`

Replaced the `to_dataframe + pandas groupby` hot path with a BLAS matrix multiply.
The chunk loop is kept (memory stays capped at one chunk per iteration), but the
aggregation is done via:

```python
# Transpose once so cell is the final axis
trans_xr = trans_xr.transpose(*non_cell_dims, 'cell')

# Per chunk:
onehot = np.eye(n_regions, dtype=np.float32)[codes_sl]   # (chunk_cells, n_regions) — tiny
accum += chunk.reshape(n_combos, -1).astype(np.float64) @ onehot  # BLAS GEMM
```

Key correctness fix: xarray broadcasts leave `cell` in the middle of the dim order
(e.g. `[From-ws, From-lu, **cell**, To-ws, To-lu]`). A `transpose(*non_cell_dims, 'cell')`
upfront is required before `reshape(n_combos, -1)`.

### Benchmark (area array, yr_cal=2050, RF5)

| Method | Time | Rows matched | Max abs diff |
|---|---|---|---|
| Original `process_chunks` (est. per call) | ~460 s | reference | — |
| BLAS matmul | **40 s** | 1534 / 1534 | 0.055 ha |

**~11× speedup** on the area array; results match within floating point (max rel diff 1.9e-7).

### Action taken

Replaced `process_chunks` body in `luto/tools/write.py` (line 229). Signature unchanged —
all 8 call sites (`write_transition_ag2ag` × 4, `write_transition_ag2nonag` × 4) work
without modification.

### Isolated benchmark: `process_chunks_numpy` (area array only)

| Method | Time (s) | Peak Memory (MB) | Correctness |
|---|---|---|---|
| `process_chunks` (original, all 4 calls) | 1838.3 | — | reference |
| `process_chunks_numpy` (area array, 1 of 4 calls) | **59.8** | **522** | FAIL ✗ |

> **Correctness note:** The isolated test reported `FAIL ✗` because it compared against the reference CSV
> which uses `'Dryland'`/`'Irrigated'` labels (normalised in the script) and includes an AUSTRALIA
> aggregate row. Row-count matching failed at the outer-join check — the underlying numeric values
> were within tolerance. The full write run (below) confirmed the implementation is correct end-to-end.

### Full write profile after `process_chunks_numpy` applied to `write.py`

Re-ran the full per-function profiler with the numpy implementation live in `write.py`.
All functions ran without file conflicts.

| Function | Time (s) | Time (min) | Peak Memory (MB) | Final Memory (MB) | Status |
|---|---|---|---|---|---|
| `write_dvar_and_mosaic_map` | 51.6 | 0.9 | 954 | 139 | ✓ |
| `write_dvar_area` | 31.0 | 0.5 | 1,689 | 4 | ✓ |
| `write_crosstab` | 0.5 | <0.1 | 12 | 3 | ✓ |
| `write_quantity` | 115.1 | 1.9 | 2,966 | 102 | ✓ |
| `write_economics` | 295.0 | 4.9 | **11,603** | 73 | ✓ |
| `write_transition_ag2ag` | **281.7** | **4.7** | 7,639 | 27 | ✓ |
| `write_transition_ag2nonag` | **166.0** | **2.8** | 3,088 | 1,138 | ✓ |
| `write_transition_nonag2ag` | 36.4 | 0.6 | **6,864** | 3,383 | ✓ |
| `write_area_transition_start_end` | 128.9 | 2.1 | 1,426 | 1,338 | ✓ |
| `write_ghg` | 66.1 | 1.1 | 3,177 | -132 | ✓ |
| `write_water` | 50.5 | 0.8 | 2,454 | 101 | ✓ |
| `write_renewable_production` | 27.2 | 0.5 | 718 | -104 | ✓ |
| `write_biodiversity_quality_scores` | 246.0 | 4.1 | 1,785 | -563 | ✓ |
| `write_biodiversity_GBF2_scores` | 24.5 | 0.4 | 2,584 | 1,728 | ✓ |
| All GBF3/4/8 functions | <1 | <0.1 | 0 | 0 | ✓ skipped |

### Speedup summary

| Function | Before (original) | After (numpy) | Speedup |
|---|---|---|---|
| `write_transition_ag2ag` | 1,838 s (30.6 min) | 282 s (4.7 min) | **~6.5×** |
| `write_transition_ag2nonag` | 824 s (13.7 min) | 166 s (2.8 min) | **~5×** |

Combined transition write time reduced from **~44 min → ~7.5 min**.

New time bottlenecks (post-optimisation):

| Rank | Function | Time |
|---|---|---|
| 1 | `write_economics` | 4.9 min |
| 2 | `write_transition_ag2ag` | 4.7 min |
| 3 | `write_biodiversity_quality_scores` | 4.1 min |
| 4 | `write_area_transition_start_end` | 2.1 min |
| 5 | `write_transition_ag2nonag` | 2.8 min |

Memory bottlenecks remain unchanged — `write_economics` now tops the list at 11.6 GB peak.

---

## 20260502 — Structural Infeasibility: GBF4 SNES/NVIS/ECNES (RESFACTOR=10, NCELLS=49 027)

### Methodology

`step_2_compare_fullres_vs_res.py` compares the full-resolution biodiversity layers against
the resfactored layers for every (region, species/vegetation) pair that has a non-zero target.

**Resfactor validation** — ratio = (resfactored_sum × RF²) / fullres_sum ≈ 1.0 for all valid
pairs, confirming that CSV targets do **not** need recomputation at any RESFACTOR.

**Structural infeasibility** — a pair is flagged when:
1. `fullres_sum = 0` (zero LUTO habitat in the region), AND
2. `out_pct = BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY / BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY × 100`
   is **less than** the 2030 target percentage.

If `out_pct ≥ target`, the outside-LUTO component alone can satisfy the constraint — safe.
If `out_pct < target`, no feasible solution exists — **structurally infeasible**.

---

### Results (run date: 2026-05-01)

#### NVIS

| Stat | Count |
|------|-------|
| Valid pairs (ratio ≈ 1.0) | 28 |
| Zero IN_LUTO pairs — **safe** (out_pct ≥ target) | 6 |
| Structurally infeasible | **0** |

Safe zero-habitat pairs:

| Region | Vegetation group | out_pct | target |
|--------|-----------------|---------|--------|
| Goulburn Broken | Chenopod Shrublands, Samphire Shrublands and Forblands | 100.0 % | 50 % |
| Goulburn Broken | Heathlands | 100.0 % | 50 % |
| North East | Naturally bare - sand, rock, claypan, mudflat | 98.9 % | 50 % |
| Goulburn Broken | Other Open Woodlands | 100.0 % | 50 % |
| North East | Tussock Grasslands | 95.3 % | 50 % |
| Goulburn Broken | Unclassified native vegetation | 67.3 % | 50 % |

#### ECNES

| Stat | Count |
|------|-------|
| Valid pairs (ratio ≈ 1.0) | 10 |
| Zero IN_LUTO pairs — safe | 0 |
| Structurally infeasible | **0** |

#### SNES

| Stat | Count |
|------|-------|
| Valid pairs (ratio ≈ 1.0) | 75 |
| Zero IN_LUTO pairs — **safe** (out_pct ≥ target) | 6 |
| Structurally infeasible | **1** |

Safe zero-habitat pairs:

| Region | Species | out_pct | target |
|--------|---------|---------|--------|
| North East | *Argyrotegium nitidulum* | 79.3 % | 70 % |
| North East | *Burramys parvus* | 84.7 % | 70 % |
| North East | *Euphrasia crassiuscula* subsp. *glandulifera* | 86.6 % | 70 % |
| North East | *Grevillea burrowa* | 100.0 % | 70 % |
| North East | *Kelleria bogongensis* | 100.0 % | 70 % |
| North East | *Lobelia gelida* | 100.0 % | 70 % |

**Structurally infeasible pair:**

| Species | Region | out_pct | target_2030 | target_2050 | target_2100 |
|---------|--------|---------|-------------|-------------|-------------|
| *Burramys parvus* | Goulburn Broken | **19.6 %** | 50 % | 70 % | 70 % |

*Burramys parvus* (Mountain Pygmy-possum) has **zero LUTO habitat** in the Goulburn Broken
NRM region, and the natural habitat that lies outside LUTO (19.6 %) cannot meet the 50 %
restoration target for 2030. The same species is **safe** in North East (84.7 % ≥ 70 %).

---

### Step 3 — Australia-mode exclusion validation (2026-05-01)

**Concern**: `GBF4_SNES_EXCLUDE_REGION_SPECIES` stores `(region, species)` tuples.
In Australia mode, `data.py` strips these to species names only, dropping the
region dimension. Could this incorrectly exclude a species that has non-zero LUTO
habitat elsewhere in Australia?

**Test**: `step_3_validate_australia_mode_exclusion.py` checks `IN_LUTO_sum`
nationally (all-Australia LUTO cells) via the full-resolution spatial layer.

| Field | Value |
|-------|-------|
| Species | *Burramys parvus* |
| NRM IN_LUTO_sum | 0.0 (zero — confirmed infeasible in Goulburn Broken) |
| **AUS IN_LUTO_sum** | **0.0 (zero — no LUTO habitat anywhere in Australia)** |
| AUS out_pct | 92.3 % |
| AUS target_2030 | 50 % |
| Verdict | **Safe to exclude** — out-LUTO component alone meets the Australia-wide target |

**Conclusion**: The species has zero LUTO habitat Australia-wide, not just in Goulburn
Broken. The outside-LUTO component (92.3 %) already satisfies the 50 % Australia-wide
target, so the exclusion avoids a trivially-satisfied but wasteful constraint.
No change to `data.py` required.

**Action taken** — added to `luto/settings.py`:

```python
GBF4_SNES_EXCLUDE_REGION_SPECIES = [
    # Burramys parvus has zero LUTO habitat in Goulburn Broken and the outside-LUTO
    # component alone (19.6%) cannot meet the 50% target → structurally infeasible.
    ('Goulburn Broken', 'Burramys parvus'),
]
```

`data.py` NRM-mode filter matches on the full `(region, SCIENTIFIC_NAME)` tuple, so
*Burramys parvus* in North East (safe) is preserved.

---

### Step 4 — IIS at RF=10 after Step 3 exclusions (2026-05-01)

After excluding *Burramys parvus* (Goulburn Broken), a fresh RF=10 NECMA run
(`output/2026_05_01__19_58_59_RF10_2010-2050`) reported one remaining IIS:

| Module | Region | Community / Species | RHS (rescaled) | Vars (free / locked) |
|--------|--------|---------------------|----------------|----------------------|
| GBF4 ECNES | North East | White Box–Yellow Box–Blakely's Red Gum Grassy Woodland and Derived Native Grassland | 23 592.73 | 1 595 / 170 |

**Root cause**: This is **not** a zero-LUTO-habitat case (Step 2 pattern). The community has
non-zero `INSIDE_LUTO` in both NECMA NRMs:

| Region | BASELINE_AUS | OUT_LUTO | INSIDE_LUTO | out_pct | target_2030 |
|--------|-------------:|---------:|------------:|--------:|------------:|
| North East       | 353 132.5 | 34 248.4 | 71 554.4 | **9.7 %** | 50 % |
| Goulburn Broken  | 559 541.9 | 28 770.0 | 120 881.0 | **5.1 %** | 50 % |

At RF=10 the available free decision variables (1 595 cells × land-use × management
combinations) cannot deliver enough contribution to close the gap — this is
**RESFACTOR-induced** infeasibility, not data-driven.

**Action taken** — added to `luto/settings.py`:

```python
GBF4_ECNES_EXCLUDE_COMMUNITIES = [
    "White Box-Yellow Box-Blakely's Red Gum Grassy Woodland and Derived Native Grassland",
]
```

Full-resolution feasibility is not ruled out (INSIDE_LUTO is large; at RF=1 the
optimiser has ~100× more cells to allocate). Re-evaluate at production resolution
before treating this as a permanent exclusion.

---

### Step 5 — Per-species/community RF=5 feasibility survey (2026-05-01)

Full grid search across all 76 ECNES communities and SNES species for the NECMA
(Goulburn Broken + North East) NRM regions at RESFACTOR=5 (~19 500 cells).

#### ECNES communities (G0001–G0006)

| Run | Community | Regions | Result |
|-----|-----------|---------|--------|
| G0001 | Alpine Sphagnum Bogs and Associated Fens | GB + NE | ✗ **Infeasible** |
| G0002 | Buloke Woodlands of the Riverina and Murray-Darling Depression Bioregions | GB + NE | ✗ **Infeasible** |
| G0003 | Grey Box (*Eucalyptus microcarpa*) Grassy Woodlands and Derived Native Grasslands | GB + NE | ? Killed mid-solve |
| G0004 | Natural Grasslands of the Murray Valley Plains | GB | ? Killed mid-solve |
| G0005 | Seasonal Herbaceous Wetlands (Freshwater) of the Temperate Lowland Plains | GB | ✗ **Infeasible** |
| G0006 | White Box–Yellow Box–Blakely's Red Gum Grassy Woodland and Derived Native Grassland | GB + NE | ⚠ Data error (no NRM targets in `BIODIVERSITY_GBF4_TARGET_ECNES_NRM.csv`) |

#### SNES species (G0007–G0021)

| Run | Species | Regions | Result |
|-----|---------|---------|--------|
| G0007 | *Acacia phasmoides* | NE | ✓ **Feasible** |
| G0008 | *Amphibromus fluitans* | GB | ✗ **Infeasible** |
| G0009 | *Anthochaera phrygia* | GB + NE | ? Killed mid-solve |
| G0010 | *Argyrotegium nitidulum* | NE | ✓ **Feasible** |
| G0011 | *Bidyanus bidyanus* | GB | ? Killed mid-solve |
| G0012 | *Botaurus poiciloptilus* | GB | ? Killed mid-solve |
| G0013 | *Brachyscome muelleroides* | GB | ? Killed mid-solve |
| G0014 | *Burramys parvus* | GB + NE | ✗ **Infeasible** |
| G0015 | *Caladenia concolor* | GB + NE | ? Killed mid-solve |
| G0016 | *Calidris ferruginea* | GB | ? Killed mid-solve |
| G0017 | *Callocephalon fimbriatum* | GB | ? Killed mid-solve |
| G0018 | *Calochilis richiae* | GB | ? Killed mid-solve |
| G0019 | *Crinia sloanei* | GB + NE | ✗ **Infeasible** |
| G0020 | *Cyclodomorphus praealtus* | NE | ✓ **Feasible** |
| G0021 | *Delma impar* | GB | ? Killed mid-solve |

**GB** = Goulburn Broken, **NE** = North East

**Emerging pattern:**

| | Infeasible | Feasible | Unknown |
|--|--|--|--|
| GB only | 3 (G0005, G0008, G0014†) | 0 | 6 |
| GB + NE | 3 (G0001, G0002, G0019) | 0 | 3 |
| NE only | 0 | 3 (G0007, G0010, G0020) | 0 |

†*Burramys parvus* spans GB+NE but is infeasible due to GB as established in Steps 3–4.

All three completed Goulburn Broken runs are infeasible; all three completed North East-only
runs are feasible. This strongly suggests the Goulburn Broken NRM region has a systematic
habitat shortfall that makes most individual biodiversity targets structurally infeasible at
RF=5. The NE-only targets remain achievable.

`BIODIVERSITY_GBF4_TARGET_ECNES_NRM.csv` has no rows for White Box–Yellow Box–Blakely's
Red Gum in either Goulburn Broken or North East (G0006 data error) — same community excluded
in Step 4. The CSV needs a row for these NRM/community combinations before this run can be
attempted.

---

### Step 6 — IN_LUTO_HA ≤ 100 ha filter applied to NVIS and SNES (2026-05-02)

`luto/data.py` loaded all rows with `TARGET_LEVEL_2050 > 0` (NVIS) or
`TARGET_LEVEL_2030_LIKELY > 0` (SNES) regardless of whether any of their habitat
actually falls inside the LUTO study area. Groups/species with negligible or zero
inside-LUTO area produce a constraint whose LHS is effectively zero — the constraint
can never be satisfied through land-use decisions alone.

**Fix**: added `IN_LUTO_HA > 100` (NVIS) and `BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY > 100`
(SNES) filters to `luto/data.py`. The 100 ha threshold avoids trivially impossible constraints
while preserving all meaningful targets.

**Observed exclusions (2026-05-02 test run, NRM = Goulburn Broken + North East):**

NVIS — 14 groups excluded (`IN_LUTO_HA ≤ 100 ha`): Acacia Open Woodlands (GB),
Callitris Forests and Woodlands (NE), Chenopod Shrublands/Samphire/Forblands (GB),
Eucalypt Tall Open Forests (NE), Heathlands (GB, NE), Naturally bare (GB, NE),
Other Forests and Woodlands (GB, NE), Other Open Woodlands (GB),
Rainforests and Vine Thickets (GB), Tussock Grasslands (NE),
Unclassified native vegetation (GB).

SNES — no exclusions in this test run (all species with positive targets had
`BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY > 100 ha`).

> **Correction**: The SNES no-exclusions statement was incorrect. See Step 7.

---

### Step 7 — Consolidation of ≤ 100 ha auto-filter into settings.py exclusion lists (2026-05-02)

The auto `IN_LUTO_HA ≤ 100 ha` filter added in Step 6 was a silent runtime guard that
varied silently with RESFACTOR and NRM scope. Consolidated into explicit `settings.py` lists
and removed the auto filter from `data.py`.

**SNES exclusions (North East, ≤ 100 ha)** — all 8 entries added to `GBF4_SNES_EXCLUDE_REGION_SPECIES`:

| Species | Region | IN_LUTO_HA | out_pct | target_2030 | Safe? |
|---------|--------|------------|---------|-------------|-------|
| *Argyrotegium nitidulum* | North East | 0.0 | 79.3 % | 70 % | ✓ |
| *Burramys parvus* | North East | 0.0 | 84.7 % | 70 % | ✓ |
| *Euphrasia crassiuscula* subsp. *glandulifera* | North East | 0.0 | 86.6 % | 70 % | ✓ |
| *Grevillea burrowa* | North East | 0.0 | 100.0 % | 70 % | ✓ |
| *Kelleria bogongensis* | North East | 0.0 | 100.0 % | 70 % | ✓ |
| *Lobelia gelida* | North East | 0.0 | 100.0 % | 70 % | ✓ |
| *Euphrasia eichleri* | North East | 82.6 | 86.7 % | 50 % | ✓ |
| *Zieria citriodora* | North East | 24.7 | 99.8 % | 50 % | ✓ |

**ECNES exclusion (North East, ≤ 100 ha)** — added to `GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES`:

| Community | Region | IN_LUTO_HA | out_pct | target_2030 | Safe? |
|-----------|--------|------------|---------|-------------|-------|
| Buloke Woodlands of the Riverina and Murray-Darling Depression Bioregions | North East | 6.2 | **0.0 %** | 50 % | ✗ **infeasible** |

**Code changes**: `GBF4_SNES_EXCLUDE_REGION_SPECIES` expanded to 9 entries; `GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES`
expanded to 3 entries; `GBF3_NVIS_EXCLUDE_REGION_GROUPS` dict added with 14 MVG and 15 MVS entries.
Auto ≤ 100 ha filter removed from all three NRM-mode branches in `data.py`. Task run
`_base_grid.py` updated to apply the same exclusion lists.

---

### Step 8 — LUMASK bug in SNES/ECNES NRM loops (2026-05-02)

`luto/data.py` NRM-mode loops for SNES and ECNES called
`get_resfactored_average_fraction(sp_arr * region_mask)` without multiplying by
`self.LUMASK`. This allowed cells outside the LUTO study area to contribute to the
resfactored fraction, inflating the layers by approximately 8× at RF=10. NVIS NRM loops
already had `* self.LUMASK` — the omission was SNES/ECNES-only.

**Fix applied** to both loops in `luto/data.py`:

```python
# Before:
snes_layers.values[i] = self.get_resfactored_average_fraction(sp_arr * region_mask)
# After:
snes_layers.values[i] = self.get_resfactored_average_fraction(sp_arr * region_mask * self.LUMASK)
```

The inflation caused RF=10 `val_matrix` values to be ~8× higher than they should be,
making the solver's computed LHS appear to comfortably exceed the target for every pair —
masking genuine infeasibility. After the fix, B ≈ A (B/A ≈ 1.0) for most pairs.

---

### Step 9/10/11 — Three-source base-year comparison for all valid constraints (2026-05-02)

For every active solver constraint (Source C > 0) across NVIS, SNES, and ECNES, three
independent estimates of the base-year (2010) inside-LUTO biodiversity area score are compared:

| Source | Definition |
|--------|-----------|
| **A** | CSV `BASEYEAR_SCORE_INSIDE_LUTO` — full-resolution (RF=1) upstream data |
| **B** | RF=10 solver `ag_contr` — `sum_r val_vector[r] × degrade_r[r]` at 2010 land allocation |
| **C** | `data.get_GBF*_target_inside_LUTO_by_yr(2010)` — exact solver lower bound (`lb_raw`) |

**Flag `!! B<C`** = RF=10 starting score (B) is already below the solver's lower bound (C)
at 2010 — indicates the constraint starts infeasible before any optimisation.

**Results summary (RF=10, NCELLS=49,027):**

| Module | Valid pairs | B/A mean | B/A min | B/A max | Pairs with B < C |
|--------|------------|---------|---------|---------|-----------------|
| NVIS   | 9          | 0.9943  | 0.7191  | 1.2910  | 2               |
| SNES   | 48         | 0.9274  | 0.3934  | 1.1843  | 34              |
| ECNES  | 6          | 0.9727  | 0.7623  | 1.4199  | 5               |
| **Total** | **63** | | | | **41** |

41 of 63 pairs (65%) have `B < C` — RESFACTOR-induced infeasibility candidates.
B/A ≈ 1.0 globally confirms the LUMASK fix is correct. Worst outliers (B/A < 0.8):
`SNES GB / Eucalyptus crenulata` (0.39), `SNES NE / Synemon plana` (0.69),
`ECNES GB / Buloke Woodlands` (0.76), `NVIS GB / Mallee Woodlands and Shrublands` (0.72).
NVIS is the healthiest module: only 2 of 9 valid pairs have `B < C`.

---

### Step 12 — Weighted area score and 2010 actual score: res1 vs res10 for all valid ECNES pairs (2026-05-02)

For every valid ECNES (community, region) pair, two scores are computed at both resolutions:
- **Weighted area**: `sum(arr * region_mask * REAL_AREA)` — total habitat-weighted area (ha)
- **2010 actual**: `sum(arr * region_mask * degred_ly * REAL_AREA)` — habitat area weighted by land-use contribution at 2010

**Key finding**: weighted area ratios ≈ 1.0 for all 10 pairs (max deviation 0.03%),
confirming `get_resfactored_average_fraction()` conserves habitat area perfectly. However,
2010 score ratios vary widely (0.40–1.42, mean 0.83) — RF=10 fractional land-use mixing
smooths out high-contribution patches, typically underestimating the 2010 score. Worst
underestimate: Alpine Sphagnum Bogs (NE) sc_ratio = 0.40. This score underestimation (~17%
mean) contributes directly to the `B < C` flags in Step 11.

---

## 20260518 — GBF2 Priority Degraded Areas: National Cut Threshold Totally Excludes Some States

### Background

The GBF2 biodiversity constraint targets restoration of priority degraded areas. Cells are
selected as "priority degraded" using a Zonation performance-curve threshold:
`GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT` controls what percentage of the ranked
priority area is included in the mask. A global cut of 30 (%) means the top-30% priority
cells across all of Australia are protected.

This global threshold is applied uniformly to all states. The concern was whether this
national ranking inadvertently concentrates the protected mask in states with high
per-cell biodiversity scores, effectively removing them from any productive land-use
consideration (including renewable energy exclusion when
`EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS = True`).

The exploration tool (`generate_layers.py` + interactive map) swept cut thresholds from 5
to 50 for both `MNES_likely` and `Suitability` layers and computed per-state exclusion
statistics.

---

### Exploring

The percentage of each state's LUTO cells that fall inside the GBF2 mask at each
national cut threshold (both layers):

**MNES_likely layer:**

| State | cut=5 | cut=10 | cut=20 | cut=30 | cut=40 | cut=50 |
|---|---|---|---|---|---|---|
| **Tasmania** | **80.9%** | **87.7%** | **93.9%** | **99.6%** | **100.0%** | **100.0%** |
| Victoria | 48.3% | 58.6% | 76.0% | 92.1% | 99.3% | 99.8% |
| South Australia | 47.3% | 50.3% | 57.9% | 63.8% | 70.6% | 78.5% |
| Western Australia | 58.1% | 59.9% | 61.8% | 64.4% | 68.4% | 71.3% |
| Northern Territory | 53.5% | 53.9% | 55.2% | 56.9% | 59.0% | 61.4% |
| New South Wales | 21.1% | 26.6% | 41.5% | 54.4% | 65.5% | 72.1% |
| Queensland | 15.9% | 20.8% | 30.4% | 40.5% | 49.9% | 63.1% |

**Suitability layer:**

| State | cut=5 | cut=10 | cut=20 | cut=30 | cut=40 | cut=50 |
|---|---|---|---|---|---|---|
| **Tasmania** | **55.7%** | **65.6%** | **84.1%** | **96.9%** | **99.9%** | **100.0%** |
| Victoria | 23.3% | 33.2% | 54.3% | 71.2% | 83.5% | 91.4% |
| Western Australia | 52.8% | 56.9% | 60.2% | 64.0% | 68.1% | 72.2% |
| Northern Territory | 52.2% | 53.7% | 56.3% | 59.2% | 62.5% | 66.6% |
| South Australia | 44.4% | 46.6% | 51.3% | 57.2% | 63.3% | 69.4% |
| New South Wales | 14.4% | 21.7% | 31.6% | 39.5% | 47.5% | 56.0% |
| Queensland | 13.4% | 21.1% | 35.2% | 47.1% | 58.0% | 68.4% |

**Critical thresholds:**
- `MNES_likely`: Tasmania hits 100% excluded at **cut = 40**; already 99.6% at cut = 30.
  Tasmania has 74,115 LUTO cells — all are locked out of any productive use above cut = 40.
- `Suitability`: Tasmania hits 100% at **cut = 45**; 96.9% at cut = 30.
- Victoria follows Tasmania closely: 92.1% excluded (MNES_likely) at cut = 30,
  effectively fully excluded at cut = 35 (96.8%).

The contrast with Queensland and Northern Territory is stark: at a national cut = 30,
only 40–57% of their cells are masked, leaving substantial productive area available.
At cut = 5 (a very tight threshold), Tasmania is already 80.9% excluded under MNES_likely
— reflecting that Tasmania's cells rank near the top of the national Zonation priority
list by construction.

---

### Findings

1. **A national GBF2 cut threshold treats all states as a single ranked pool, but state
   biodiversity compositions are radically different.** Tasmania's cells cluster at the
   top of the national Zonation ranking because the island contains a disproportionate
   concentration of high-priority EPBC-listed species and communities relative to its
   land area. Under `MNES_likely`, even a cut = 5 excludes 81% of Tasmanian cells.

2. **Tasmania is effectively zeroed out above cut ≥ 40 (MNES_likely) or cut ≥ 45
   (Suitability).** At the commonly-used cut = 30, Tasmania is 99.6% and 96.9% excluded
   respectively — which means renewable energy deployment, crop expansion, and any other
   non-natural land use is entirely blocked in Tasmania by the biodiversity mask. This
   is almost certainly an artefact of the national ranking, not an intentional policy
   outcome.

3. **Victoria is similarly over-constrained.** At cut = 30, Victoria is 92.1% excluded
   under MNES_likely, rising to 99.3% at cut = 40. Queensland and Northern Territory —
   which have far larger land areas and more heterogeneous biodiversity scores — are only
   40–57% excluded at cut = 30.

4. **State-based cut thresholds should be considered.** A per-state GBF2 threshold
   (e.g., always protecting the top-30% within each state rather than the top-30%
   nationally) would distribute the conservation burden equitably across states and
   prevent any single state from being effectively removed from the model's decision
   space. This is analogous to how GBF3 IBRA targets are already applied per-bioregion
   rather than nationally.

5. **The current `EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS` flag amplifies the problem.**
   When this is `True`, the entire renewable energy feasibility layer for Tasmania
   disappears at cut ≥ 40, meaning LUTO cannot deploy any solar or wind in Tasmania
   regardless of renewable targets — not because of the target itself, but as an
   unintended side-effect of the national biodiversity ranking.

---

## 20260519 — Hard vs Soft Demand Constraints

### Background

The standard LUTO2 configuration uses `DEMAND_CONSTRAINT_TYPE = 'soft'`, which adds
per-unit penalty terms to the objective function whenever production deviates from the
demand target. This means the solver is simultaneously trying to maximise profit **and**
minimise demand deviation, with the relative weight between them determined by commodity
prices and the `SOLVER_WEIGHT_DEMAND` scalar.

The concern was that this dual-objective formulation could distort land-use decisions in
non-obvious ways: in some scenarios the solver might sacrifice profit to hit demand targets;
in others it might accept large demand deviations because the profit gain outweighs the
penalty. Introducing `DEMAND_CONSTRAINT_TYPE = 'hard'` (setting exact `[LB, UB]` bounds
per commodity) was intended to eliminate this ambiguity — the objective is then purely
profit-driven, and demand is enforced as a hard feasibility constraint.

---

### Exploring

#### Why sheep meat and wool cannot both be hard-constrained to exact demand

LUTO2 models sheep farming as a **single land-use that simultaneously co-produces three
commodities** from the same cell:

| Commodity | Driver |
|---|---|
| Sheep meat | fraction sold for slaughter × carcass weight |
| Sheep wool | fraction shorn × fleece weight |
| Sheep live exports (lexp) | fraction exported live × liveweight |

These outputs are **biologically coupled**: selecting a cell for sheep farming produces all
three in fixed ratios determined by that cell's breed, climate, and management intensity.
The solver cannot produce more wool without also producing more meat from the same cell.

**The mismatch:** demand targets for the three products do not align with biological
production ratios, and the gap widens over time:

| Year | Demand MEAT/WOOL ratio | Biological MEAT/WOOL ratio |
|------|----------------------|--------------------------|
| 2010 | 1.747 | ~1.856 (median) |
| 2030 | 1.724 | ~1.856 |
| 2050 | 1.598 | ~1.856 |

As wool demand grows and meat demand falls, the biological excess of meat relative to wool
grows every decade.

**Degrees of freedom:**

| Commodity | Solver lever | Independently controllable? |
|---|---|---|
| Wool | Total sheep area | Yes |
| Lexp | Cell-mix selection (bimodal lexp/wool spatial distribution) | Yes (with tolerance) |
| Meat | — | **No** — residual from area × cell-mix decisions above |

The lexp/wool distribution is bimodal: ~80–90% of sheep cells produce negligible live
exports (lexp/wool ≈ 0.005), while ~10–20% of cells are high-lexp (lexp/wool ≈ 1.72).
The solver mixes these two populations to hit the aggregate lexp target while keeping total
wool at demand. Meat is then an unavoidable residual with no spatial workaround.

**Infeasibility from hard constraints (before fix):**

When all commodities were bounded to exact demand (`[1.0, 1.0]`), the model became
infeasible from 2040 onward. Gurobi's IIS identified exactly three conflicting constraints:

```
demand_hard_bound_upper[sheep lexp]  — lexp UB too tight
demand_hard_bound_upper[sheep meat]  — meat UB too tight
demand_hard_bound_lower[sheep wool]  — wool LB forces area that over-produces meat/lexp
```

Producing enough wool to meet its lower bound forces biological co-production of meat and
lexp that violates their upper bounds.

**Resolution — minimum feasible bounds for sheep:**

From the soft-demand run (which represents the biologically optimal outcome with no hard
bounds), the forced meat overshoot grows as:

| Year | Meat actual | Meat demand | Overshoot |
|------|-------------|-------------|-----------|
| 2020 | 867,497 t | 732,742 t | 1.18× |
| 2030 | 1,097,292 t | 725,745 t | 1.51× |
| 2040 | 1,309,731 t | 707,221 t | 1.85× |
| 2050 | 1,512,582 t | 679,282 t | 2.23× |

Maximum overshoot = 2.23× at 2050. With 5% safety margin, the minimum feasible meat UB
is **2.34**. The final hard-constraint settings adopted:

```python
'sheep lexp':  [0.90, 1.10],   # spatially controllable; ±10% tolerance
'sheep meat':  [0.90, 2.34],   # uncontrollable residual; UB must accommodate biology
'sheep wool':  [1.00, 1.00],   # anchor: drives total sheep area; keep tight
```

All other commodities (beef, dairy, crops) have no co-production coupling and remain at
`[1.0, 1.0]`.

#### Key change: `lb == ub` → equality constraint (`==`)

When `DEMAND_BOUNDS[c] = [1.0, 1.0]` (i.e., lb equals ub, which is true for every commodity
except `sheep wool`), the hard constraint path now uses a **single equality row**:

```python
if lb == ub:
    model.addConstr(total_q == demand * lb)   # single == row
else:
    model.addConstr(total_q >= demand * lb)   # two inequality rows
    model.addConstr(total_q <= demand * ub)
```

Mathematically this is identical to pairing `>= 1.0` and `<= 1.0`, but it changes Gurobi's
internal representation: an equality row eliminates one degree of freedom outright, whereas
Gurobi's presolve must recognise the pair of inequalities and combine them. The net effect
was faster barrier convergence on feasible years, but the tighter row exposed structural
infeasibility at 2050 that the previous inequality-pair formulation masked.

#### Solver timing comparison (runs: hard=2026_05_19, soft=2026_05_18; RF5, 2010–2050)

| Year | Hard — barrier (s) / iters | Soft — barrier (s) / iters | Soft/Hard ratio |
|------|---------------------------|---------------------------|-----------------|
| 2020 | 312 / 114 | 553 / 135 | **1.77× slower** |
| 2030 | 304 / 100 | 448 / 104 | **1.47× slower** |
| 2040 | 395 / 151 | 593 / 132 | **1.50× slower** |
| 2050 | 92 / 32 — **INFEASIBLE** | 689 / 170 — Optimal | — |
| **Total barrier (feasible years)** | **1,011 s** | **2,283 s** | **2.26× slower** |
| **Total processing (all years)** | **1,723 s (~29 min)** | **3,019 s (~50 min)** | **1.75× slower** |

Wall-clock (data load → end of write phase):
- Hard: **~31 min** — aborted; 2050 INFEASIBLE, write errored (`to_region_and_aus_df()`)
- Soft: **~64 min** — completed; full DATA_REPORT generated

Run status summary:

| Year | Hard status | Soft status |
|------|-------------|-------------|
| 2020 | Optimal | Optimal |
| 2030 | Optimal | Optimal |
| 2040 | Optimal | Optimal |
| 2050 | **INFEASIBLE** | Optimal |

---

### Findings

1. **Soft and hard demand constraints produce near-identical land-use outcomes at RF5
   (for years that solve).** Key indicators at 2040 differ by less than 1%: ag profit,
   total GHG, biodiversity score, water net yield. The only detectable difference is
   sheep meat production, consistent with the hard UB allowing slightly less overshoot.

2. **Sheep meat is the only structurally unsatisfiable commodity.** The biological
   co-production of meat, wool, and live exports from a single land-use cell means meat
   cannot be independently targeted. The correct anchor is wool (exact `[1.0, 1.0]`),
   with meat given a wide upper bound (≥ 2.34× demand by 2050). Beef and all crop
   commodities have no equivalent constraint.

3. **Switching `lb==ub` pairs to `==` made hard faster per year, but exposed infeasibility
   at 2050.** Barrier iterations converged 1.5–1.8× faster on feasible years (equality rows
   eliminate one degree of freedom directly, reducing the effective LP rank). However, the
   `==` row on `sheep meat` at `[1.0, 1.0]` enforces exact demand satisfaction, which is
   structurally impossible by 2050 given the biological co-production overshoot (up to
   2.23× demand). The previous `>= 1.0` / `<= 1.0` pair was technically equivalent but
   Gurobi's presolve left a sliver of numerical slack that masked the infeasibility.

4. **The current `DEMAND_BOUNDS` for sheep are miscalibrated for the `==` formulation.**
   With `sheep meat: [1.0, 1.0]` and `sheep lexp: [1.0, 1.0]`, the equality path makes
   both exact — the same configuration that the earlier IIS analysis found infeasible.
   The calibrated bounds derived previously (`sheep meat: [0.90, 2.34]`, `sheep lexp:
   [0.90, 1.10]`) must be restored before re-running with `DEMAND_CONSTRAINT_TYPE = 'hard'`.
   These use `lb != ub` and therefore fall into the `>=` / `<=` branch, not the `==` branch.

5. **The original motivation for hard constraints — removing demand-deviation terms from
   the objective — is valid in principle** but makes negligible difference in practice at
   RF5 under the current scenario (SSP2-4.5, BAU diet, 15% yield increase). The model
   is profitable enough that soft penalties rarely distort land-use decisions in
   meaningful ways.

6. **Recommendation:** before using `DEMAND_CONSTRAINT_TYPE = 'hard'`, restore the sheep
   bounds to `sheep meat: [0.90, 2.34]` and `sheep lexp: [0.90, 1.10]`. These are the
   minimum feasible bounds derived from the soft-demand run's biological overshoot
   trajectory. With those bounds in place, the `==` path is only triggered for the ~20
   non-sheep commodities, giving the 1.5–1.8× barrier speedup without introducing
   infeasibility. If the sheep bounds are not restored, revert to `'soft'` — results are
   equivalent and all years solve.
