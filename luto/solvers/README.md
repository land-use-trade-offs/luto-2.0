# The solver: the design in seven points

The model is constructed from TWO ARRAYS: the column space and the row space. Everything in this directory builds one of
them, reads one of them, or hands them to an engine. Read this first; the module map at the end says which file does
what. (`CLAUDE.md` at the repo root holds the field-level detail.)

## 1. The model is constructed from a column space and a row space

The model is a linear program `A · x {<, =, >} rhs`, `min obj · x`. The COLUMNS are the unknowns, the ROWS the
constraints, `A` the one sparse matrix between them (`row_matrix`, a scipy CSR, rows × cols). The column space is
built first, from the base-year land use alone; the row space is built over it. The solver takes the two tables, the
matrix and the objective vector as they are and returns `x`.

## 2. The columns are the unknowns, stacked in blocks

Every Gurobi variable is one row of the column table, `cols` (an `xr.Dataset` on `col`), in block order:

    ag | nonag | am | ag2ag | ag2nonag | nonag2ag        (+ slack, see point 6)

`ag` / `nonag` / `am` are the land-use shares of every cell under every (land management, land use) / non-ag land use
/ ag-management option; the three arc blocks are the transitions between them. Node balance is NOT a column block:
it is the row family that ties the arcs to the shares (every arc leaves one node of its cell and lands on another, so
a cell's node-balance rows sum to Σ shares = Σ base).

## 3. The rows are the constraints, stacked in families

The row table, `rows` (an `xr.Dataset` on `row`), has one row per constraint; `row_matrix` row i is table row i, and
its columns are the column space. The rows come in FAMILIES, one block each, in the order `row_builder.get_rows` lists
them: the wide policy rows (demand, GHG, ag-mgt adoption, regional adoption, water, renewable targets), the structural
rows (ag-mgt link, renewable ceiling, source cap, node balance), and last the biodiversity rows (GBF2, GBF3 NVIS, GBF4
SNES / ECNES, GBF8). Every row carries `rhs`, `sense`, `name`, `scale` (its rescale factor) and `active`. A family's
rows are the mask `rows['family'].values == name`; there is no index.

## 4. Both tables are 1-D, with attributes per position

A column has `lb`, `ub`, `base` and the fields that say where it reads its input — `m, j` (ag land management, land
use), `k` (non-ag land use), `am_idx, j_idx` (the ag-management slot), `from_m, from_j, from_k, local_r` (where an arc
comes from), `cell` (every column) and `block`. A row has the same int fields where its family has them, plus the
labels `region`, `GBF_target`, `GBF4_presence`, `demand_commodity`, `demand_bound`. A field a block or family does not
carry is filled: −1 on an int, `None` on a label. Nothing on either table is a code.

## 5. The range of a row is known before any solve

Because every column has `lb` and `ub`, every row has an activity interval over the column box:

    lo = A⁺ · lb + A⁻ · ub        hi = A⁺ · ub + A⁻ · lb

Compared with `rhs`, a `>` row is REDUNDANT when `lo ≥ rhs` (met at every feasible point: it need not be built),
IMPOSSIBLE when `hi < rhs` (met at none: the year stops before the model is built), and OPEN otherwise — only an open
row needs the solve to say. `row_bounds.py` does this for every row, tightened by the bounds the rows imply (a
cell's shares sum to its base, an all-ones row bounds its columns together); `rowgen.py` does the sharper version for
the biodiversity targets (per cell the least and the most habitat any feasible point can give, `cell_floor_ceiling`,
weighted per target: `safe` / `unattainable` / `open`). Both are single-row proofs: they cannot see rows competing
for the same cells, so "not proven" is not "feasible".

## 6. Too many rows to build, and rows that cannot all be met — two mechanisms

**Row generation (memory).** GBF4 and GBF8 have thousands of species (GBF8 ~10.6 k; every row built up front is
~5 × 10⁹ entries, ~350 GB at RES5). Under `BIO_ROW_GENERATION` the biodiversity families build NO rows up front:
the screen of point 5 classes every target, a safe or unattainable target gets no row, and the open ones are added in
rounds — a small family in whole (eager), a large one by batches of the targets short at the current solution (lazy) —
each round a fresh solver over the tables with the rows appended, until no open target is short.

**The shortfall (infeasibility).** A family in `ELASTIC_FAMILIES` may fall short: each of its rows gets a shortfall
column (block `slack`), `a·x + rhs·s ≥ rhs`, `s` the fraction of the target missed, at `ELASTIC_PENALTY` = 1e10 AUD
per target missed in full — 100 million AUD per 1 % — linear. A target no feasible point can meet then costs its
penalty instead of making the year infeasible. Every family goes through the same `relax`, built up front or
generated, and the structural rows are never relaxed.

## 7. A penalty paid does not prove a target impossible — the two passes

At full cost, a row falls short when its target cannot be met OR when meeting it costs more than its penalty. To tell
the two apart, PASS 1 scales the economic part of the objective by `ELASTIC_PASS1_COST_SCALE` (0 in the study): only
the penalties are minimised, every hard row (demand, water, node balance, …) holds, so a row still short at the end
of pass 1 cannot be met TOGETHER with the rest of the year — "combinedly impossible" (not proven impossible alone: only
the screen of point 5 proves that). THE DROP, once, for every relaxed row: the rows short by more than
`ELASTIC_DROP_TOL` of their target are switched off (`active`, still scored and reported), every other relaxed row is
made hard (its shortfall column fixed at 0). PASS 2 solves at full cost, never infeasible: pass 1's point meets every
row kept. The twin run with pass 1 at full cost gives up the expensive targets too; the difference between the two
drop sets is what "expensive" means in the study.

## The tables are the only state

The four tables — `row_matrix`, `rows`, `cols`, `obj` — travel as ONE `Tables` (an immutable NamedTuple). Every
operation on them (`relax`, `append_rows`, `fix_slack`) takes a `Tables` and returns a new one; a dropped row is
flagged off, the table never shrinks. The solver is stateless: it is built from the tables for every solve and never
changes them.

## Module map

| point | module | what it holds |
|---|---|---|
| inputs | `row_inputs.py` | every coefficient stream, target and layer from the economics modules, as one `RowInputs`; the objective's streams as `EconomicInputs` |
| 2, 4 | `col_builder.py` | `get_cols`: the column table and its support (`ColSupport`: the input masks, the regions of every cell, `cell2col`, the position → column grids) |
| 3, 4 | `row_table.py` | the row schema (`ROW_SCHEMA`), `make_part` (a family's block and table), `rows_where` |
| 3 | `row_builder.py` | `get_rows`: every family as a `get_<family>` returning its part, stacked in the model's row order; `get_obj`: the objective vector; the biodiversity targets (`get_<family>_targets`, `build_GBF_rows`) |
| 5 | `row_bounds.py` | bound propagation over every row: the interval, the verdict, the implied bounds, the report |
| 6, 7 | `tables.py` | `Tables` and the operations on it: `relax`, `append_rows`, `fix_slack`, `scale_cost`, `elastic_absorbs`; the readers of a relaxed row, `shortfall`, `slack_unit` |
| 5, 6 | `rowgen.py` | `RowGen`: the targets, the screen, eager / lazy, the bookkeeping; `solve_rounds`: one pass by row generation |
| 7 | `step.py` | `solve_step`: one pass, or pass 1 → the drop → pass 2; `solve_pass`: one solve or the rounds |
| engine | `solver.py` | `LutoSolver`: the Gurobi model of the tables, `solve`; `build_solver` |
| after | `post_solve.py` | `x` scattered back into the LUTO land-use arrays |
| after | `reports.py` | the shadow prices, the shortfall report, the debug MPS |

The flow (`simulation.solve_timeseries`): `get_cols` → `get_row_inputs` → `get_rows` → `RowGen` (if row
generation) → `get_row_bounds` → `get_obj` → `solve_step(Tables(...))` → `post_solve` → `record_shadow_prices`,
`report_shortfall`, `RowGen.report`.
