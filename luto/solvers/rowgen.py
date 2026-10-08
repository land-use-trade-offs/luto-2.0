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

import time

import numpy as np
import pandas as pd
import xarray as xr

from scipy import sparse

from luto import settings
from luto.solvers.col_builder import ColSupport
from luto.solvers.row_builder import append_rows, bio_contribution, contract, fix_slack, get_bio_targets, relax, scale_cost
from luto.solvers.row_inputs import RowInputs
from luto.solvers.row_table import ROW_FILL, make_part, shortfall
from luto.solvers.solver import LutoSolver
from luto.solvers.tools import report_shortfall

# ═══════════════════════════ tolerances ═══════════════════════════

SCREEN_TOL = 1e-6      # relative: the screen must clear the target by more than this to call a target safe / unattainable


# ═══════════════════════════ RowGen: the row generation of one step ═══════════════════════════

class RowGen:
    """The targets of one step and their bookkeeping: built after ``get_rows`` (while the column support still holds
    ``cell2col`` / ``ag_mrj2col``); ``solve_rounds`` (below) asks it for the rows due and tells it where they landed; then
    ``report``. It never touches a solver."""

    # ─────────────── 1. set up: the targets, the screen, the starting point ───────────────

    def __init__(self, data, inputs: RowInputs, cols: xr.Dataset, support: ColSupport, base_year: int, target_year: int,
                 carry: set | None = None):
        t0 = time.time()

        # ── 1a. the targets of every family, as one table and one (target × cell) weight matrix ──
        self.year = target_year
        self.targets, self.W = get_bio_targets(inputs, support)               # row_builder: the families' get_<family>_targets, concatenated
        self.family = self.targets['family'].values.astype(str)
        self.name = self.targets['name'].values.astype(object)
        self.rhs = self.targets['rhs'].values.astype(np.float64)              # the target inside LUTO, raw units (ha; GBF8 suitability-weighted)
        self.mode = np.where(np.isin(self.family, settings.ELASTIC_FAMILIES), 'may_drop', 'hard').astype(object)
        n = self.rhs.size
        self.cell = cols['cell'].values
        self.ncells = inputs.ncells
        self.c = bio_contribution(inputs, cols)                                # the habitat contribution of every column
        self.bio_S = (support.cell2col @ sparse.diags(self.c)).tocsr()         # (cell × col), as get_rows lays it

        # ── 1b. the screen: safe / unattainable / open ──
        floor_r, ceil_r = cell_floor_ceiling(cols, support, self.c.astype(np.float64), self.ncells)
        self.floor = self.W @ floor_r
        self.ceil = self.W @ ceil_r
        margin = SCREEN_TOL * np.abs(self.rhs)
        cls = np.full(n, 'open', dtype=object)
        cls[self.ceil < self.rhs - margin] = 'unattainable'
        cls[(self.floor >= self.rhs + margin) | (self.rhs <= 0)] = 'safe'     # rhs <= 0: the land outside LUTO already meets it
        self.cls = cls
        self.open = cls == 'open'
        self.blocked = (cls == 'unattainable') & (self.mode == 'hard')         # a hard target no feasible point can meet: the year cannot be solved

        # ── 1c. the starting point: the previous step's land use, scored at this year's layers ──
        self.start = self.W @ self.habitat(start_point(data, cols, inputs, base_year))
        self.carry = np.isin(self.name, list(carry or ())) & self.open

        # ── 1d. eager or lazy, per family: the entries its open rows would take ──
        entries = np.diff(self.W.indptr) * (self.bio_S.nnz / self.ncells)      # per target: cells × columns per cell
        self.eager = np.zeros(n, dtype=bool)
        for fam in pd.unique(self.family):
            on = self.family == fam
            self.eager[on] = entries[on & self.open].sum() <= settings.ROWGEN_EAGER_MAX_ENTRIES

        # ── 1e. the bookkeeping, per target ──
        self.round_added = np.full(n, -1, dtype=np.int32)                      # the round a target's row entered (-1: never)
        self.pass_added = np.zeros(n, dtype=np.int8)                           # 1 / 2: the pass it entered in
        self.dropped = np.zeros(n, dtype=bool)                                 # pass 1 left it short: its row removed before pass 2
        self.s = np.full(n, np.nan)                                            # pass 1's shortfall, where the target had a relaxed row
        self.row_of = np.full(n, -1, dtype=np.int64)                           # its row in the solver's table
        self.final = None
        self.rounds = []

        # ── 1f. the log ──
        short0 = self.is_short(self.start)
        for fam in pd.unique(self.family):
            on = self.family == fam
            k = {c: int((cls[on] == c).sum()) for c in ('safe', 'unattainable', 'open')}
            print(f"│   row generation, {target_year}, {fam}: screen {k['safe']:,} safe / {k['unattainable']:,} unattainable / {k['open']:,} open "
                  f"of {int(on.sum()):,} targets; {int((short0 & self.open & on).sum()):,} open short at the {base_year} land use; "
                  f"{'eager' if self.eager[on].any() else 'lazy'}, {self.mode[on][0]}", flush=True)
        print(f"│   row generation, {target_year}: {int(self.carry.sum()):,} carried from the previous step ({time.time() - t0:.0f} s)", flush=True)
        if self.blocked.any():
            print(f"│   row generation, {target_year}: {int(self.blocked.sum()):,} HARD target(s) no feasible point can meet: "
                  + ", ".join(self.name[self.blocked][:10].tolist()) + (" …" if self.blocked.sum() > 10 else ""), flush=True)

    # ─────────────── 2. the pieces: habitat, shortness, the batch ───────────────

    def habitat(self, x: np.ndarray) -> np.ndarray:
        """Per cell: the habitat of the land use x (Σ c · x over the cell's columns), float32 for the product with W."""
        return np.bincount(self.cell, weights=self.c.astype(np.float64) * x[:self.cell.size], minlength=self.ncells).astype(np.float32)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Every target's score at x: W @ H(x), one product, no rows built."""
        return self.W @ self.habitat(x)

    def is_short(self, score: np.ndarray) -> np.ndarray:
        return score < self.rhs - settings.ROWGEN_SHORT_TOL * np.abs(self.rhs)

    def hardest(self, idx: np.ndarray, score: np.ndarray) -> np.ndarray:
        """``idx`` ordered by relative shortfall at ``score``, largest first; a LAZY target's batch capped — a share of
        the targets short (settings.ROWGEN_BATCH_SHARE) within [ROWGEN_BATCH_MIN, ROWGEN_BATCH_MAX] — an eager one's not."""
        lazy = idx[~self.eager[idx]]
        batch = int(np.clip(np.ceil(settings.ROWGEN_BATCH_SHARE * lazy.size), settings.ROWGEN_BATCH_MIN, settings.ROWGEN_BATCH_MAX))
        rel = (self.rhs[lazy] - score[lazy]) / np.abs(self.rhs[lazy])
        return np.union1d(idx[self.eager[idx]], lazy[np.argsort(-rel, kind='stable')][:batch])

    def round0(self) -> np.ndarray:
        """The targets round 0 starts with: the carried ones, every open target of an eager family, and the open
        targets of a lazy family short at the starting point, hardest first."""
        open_idx = np.flatnonzero(self.open)
        short = open_idx[self.is_short(self.start)[open_idx] & ~self.carry[open_idx] & ~self.eager[open_idx]]
        return np.union1d(np.union1d(np.flatnonzero(self.carry), open_idx[self.eager[open_idx]]), self.hardest(short, self.start))

    def short_without_row(self, score: np.ndarray) -> np.ndarray:
        """The open targets without a row that are short at ``score``: the next round's candidates."""
        return np.flatnonzero(self.open & (self.row_of < 0) & ~self.dropped & self.is_short(score))

    # ─────────────── 3. the rows of a round, and where they landed ───────────────

    def rows_for(self, idx: np.ndarray):
        """The rows of the targets ``idx`` as one part — W[idx] @ bio_S, contracted and rescaled row by row, family by
        family so each row carries its family's labels: (block, table, order), row i of the block being target order[i]."""
        blocks, tables, order = [], [], []
        for fam in pd.unique(self.family[idx]):
            on = idx[self.family[idx] == fam]
            block, rhs, scale = contract(self.W[on] @ self.bio_S, self.rhs[on], rescale=True)
            labels = {field: self.targets[field].values[on] for field in ('region', 'GBF_target', 'GBF4_presence') if field in self.targets}
            _, table = make_part(fam, block, rhs, '>', self.name[on].tolist(), scale, **labels)
            blocks.append(block)
            tables.append(table)
            order.append(on)
        return sparse.vstack(blocks, format='csr'), xr.concat(tables, dim='row', fill_value=ROW_FILL), np.concatenate(order)

    def placed(self, idx: np.ndarray, first: int, rnd: int, npass: int) -> None:
        """The targets ``idx`` (in the part's row order) now hold rows ``first`` … in the row table."""
        self.row_of[idx] = first + np.arange(idx.size)
        self.round_added[idx] = rnd
        self.pass_added[idx] = npass

    # ─────────────── 4. pass 1's shortfall, and the drop (solve_step decides; this records) ───────────────

    def shortfall(self, rows: xr.Dataset, x: np.ndarray) -> np.ndarray:
        """The fraction of the target missed, per generated row that was relaxed (0 where hard or absent)."""
        on = np.flatnonzero(self.row_of >= 0)
        return np.nan_to_num(shortfall(rows, x, self.row_of[on]))

    def record_pass1(self, rows: xr.Dataset, x: np.ndarray, dropped_rows: np.ndarray) -> None:
        """Pass 1's shortfall kept per target, and the targets whose rows the step dropped marked."""
        on = np.flatnonzero(self.row_of >= 0)
        self.s[on] = self.shortfall(rows, x)
        self.dropped[np.isin(self.row_of, dropped_rows) & (self.row_of >= 0)] = True

    # ─────────────── 5. after the step: the next step's start, the report ───────────────

    def carry_next(self, rows: xr.Dataset) -> set:
        """The targets whose rows the next step starts with: those whose row binds at this step's end (π ≠ 0 on the solved
        row table), or every row still in the model when no duals were read."""
        T = rows
        kept = np.flatnonzero((self.row_of >= 0) & ~self.dropped)
        if 'pi' in T:
            pi = T['pi'].values[self.row_of[kept]]
            kept = kept[np.isfinite(pi) & (np.abs(pi) > 0)]
        return set(self.name[kept].tolist())

    def report(self, rows: xr.Dataset, out_dir: str) -> None:
        """rowgen_<year>.csv (every target) and rowgen_rounds_<year>.csv; the counts in the log. ``rows``: the solved row
        table (the solver's, carrying ``pi``)."""
        # ── 5a. every target ──
        pi = np.full(self.rhs.size, np.nan)
        on = (self.row_of >= 0) & ~self.dropped
        if 'pi' in rows:
            pi[on] = rows['pi'].values[self.row_of[on]] * 1e6 / rows['scale'].values[self.row_of[on]]   # AUD per unit, as shadow_prices_<year>.csv
        final = self.final if self.final is not None else np.full(self.rhs.size, np.nan)
        below = self.is_short(final)
        status = np.where(self.dropped, 'dropped', self.cls)                                  # safe / unattainable / open / dropped
        frac = lambda score: np.divide(score, self.rhs, out=np.full(self.rhs.size, np.nan), where=self.rhs != 0)   # noqa: E731 — the score as a fraction of the target
        df = pd.DataFrame({'family': self.family, 'name': self.name,
                           **{field: self.targets[field].values for field in ('region', 'GBF_target', 'GBF4_presence') if field in self.targets},
                           'class': self.cls, 'mode': self.mode, 'status': status,
                           'target': self.rhs, 'floor': self.floor, 'ceiling': self.ceil, 'start': self.start, 'final': final,
                           'floor_frac': frac(self.floor), 'ceiling_frac': frac(self.ceil), 'start_frac': frac(self.start), 'final_frac': frac(final),
                           'carried': self.carry, 'eager': self.eager, 'pass_added': self.pass_added, 'round_added': self.round_added,
                           'shortfall_pass1': self.s, 'dropped': self.dropped, 'shadow_price': pi, 'below_target': below})
        df.to_csv(f"{out_dir}/rowgen_{self.year}.csv", index=False)
        # ── 5b. every round ──
        pd.DataFrame(self.rounds).rename(columns={'pass_': 'pass'}).to_csv(f"{out_dir}/rowgen_rounds_{self.year}.csv", index=False)
        # ── 5c. the checks: a safe target below target disproves the screen; an open one not dropped, the loop ──
        for fam in pd.unique(self.family):
            f = self.family == fam
            bad = int((below & f & (self.cls == 'safe')).sum())
            missed = int((below & f & self.open & ~self.dropped).sum())
            print(f"│   row generation, {self.year}, {fam}: {int((on & f).sum()):,} rows in the final model "
                  f"({int(((np.abs(np.nan_to_num(pi)) > 0) & f).sum()):,} binding), {int((self.dropped & f).sum()):,} dropped, "
                  f"{int(((self.cls == 'unattainable') & f).sum()):,} unattainable; below target: "
                  f"{int((below & f).sum()):,} (safe {bad} — must be 0; open not dropped {missed} — must be 0)", flush=True)
            if bad or missed:
                print(f"WARNING: row generation, {self.year}, {fam}: {bad} safe and {missed} open targets below target at the solution", flush=True)


# ═══════════════════════════ the step: one pass, or pass 1 → the drop → pass 2 ═══════════════════════════

def solve_step(A, rows, cols, obj, inputs: RowInputs, rowgen: "RowGen | None", target_year: int, out_dir: str, on_first_model=None):
    """The solve of one step over its tables — ONE policy for every row that may fall short, whether built up front
    (settings.ELASTIC_FAMILIES: relaxed here, first) or generated during the step (``rowgen``, relaxed as it is added):

        one pass    no row may fall short, or settings.ELASTIC_DROP_SHORT is off: the rounds (``solve_rounds``) at full
                    cost, the relaxed rows short at the penalty's price — the solution as it is
        two passes  settings.ELASTIC_DROP_SHORT: PASS 1 with the cost × settings.ELASTIC_PASS1_COST_SCALE and
                    settings.ELASTIC_PASS1_PARAMS (barrier, no crossover) finds the rows that cannot be met together with
                    the step; THE DROP, once, for every relaxed row: the rows short by more than settings.ROWGEN_SHORT_TOL of
                    their target are switched off (``active``; shortfall_<year>.csv, with ``dropped``), every other relaxed
                    row made hard (its shortfall column's ub = 0, ``fix_slack``); PASS 2 at full cost (settings.RETRY_PARAMS),
                    generated rows added hard. Pass 1's point meets every row kept, so pass 2 is never infeasible.

    The tables are the only state: every operation returns new ones and every solve builds a fresh ``LutoSolver`` from
    them. ``on_first_model`` is called with the first Gurobi model built (the debug MPS). Returns (accepted, x, status,
    solver) — the last solver, built from the final tables (``solver.cols`` / ``rows`` / ``A`` / ``obj``), x over its columns."""
    A, rows, cols, obj = relax(A, rows, cols, obj, np.flatnonzero(np.isin(rows['family'].values, settings.ELASTIC_FAMILIES) & rows['active'].values))
    relaxable = (rows['slack_col'].values >= 0).any() or (rowgen is not None and (rowgen.open & (rowgen.mode == 'may_drop')).any())
    two_pass = settings.ELASTIC_DROP_SHORT and relaxable
    if not two_pass:
        accepted, x, status, solver, *_ = solve_rounds(A, rows, cols, obj, inputs, rowgen, target_year, 1, settings.RETRY_PARAMS, 1.0, on_first_model)
        return accepted, x, status, solver

    # ── pass 1 ──
    scale = settings.ELASTIC_PASS1_COST_SCALE
    if scale != 1:
        print(f"Year {target_year}: pass 1 feasibility-first — the cost × {scale:g}", flush=True)
    accepted, x, status, solver, A, rows, cols, obj = solve_rounds(A, rows, cols, obj, inputs, rowgen, target_year, 1, settings.ELASTIC_PASS1_PARAMS, scale, on_first_model)
    if not accepted:
        return accepted, x, status, solver

    # ── the drop: every relaxed row at once ──
    on = np.flatnonzero((rows['slack_col'].values >= 0) & rows['active'].values)
    s = shortfall(rows, x, on)                                                             # the fraction of the target missed
    short = on[s > settings.ROWGEN_SHORT_TOL]
    report_shortfall(x, rows, target_year, out_dir, dropped=short)
    if rowgen is not None:
        rowgen.record_pass1(rows, x, short)
    if short.size == 0 and scale == 1 and rowgen is None:                                 # pass 1 missed nothing and ran at full cost: its solution is the hard one
        print(f"Year {target_year}: no relaxed row falls short — pass 1's solution is kept", flush=True)
        return True, x, status, solver
    active = rows['active'].values.copy()
    active[short] = False
    rows = rows.assign(active=(('row',), active))
    cols = fix_slack(cols)
    family = rows['family'].values[short].astype(str)
    print(f"Year {target_year}: {short.size:,} relaxed row(s) cannot be met together with the year — switched off"
          + (" (" + ", ".join(f"{f} {int((family == f).sum()):,}" for f in pd.unique(family)) + ")" if short.size else "")
          + f"; the other {on.size - short.size:,} made hard, the full cost restored — pass 2", flush=True)

    # ── pass 2 ──
    accepted, x, status, solver, *_ = solve_rounds(A, rows, cols, obj, inputs, rowgen, target_year, 2, settings.RETRY_PARAMS, 1.0, None)
    return accepted, x, status, solver


def solve_rounds(A, rows, cols, obj, inputs: RowInputs, rowgen: "RowGen | None", target_year: int, npass: int, params, cost_scale: float, on_first_model=None):
    """One pass over the tables: without row generation one solve; with it, round by round — the rows of the targets due
    appended (``append_rows``; in pass 1 a may-drop target's row relaxed), a fresh solver built and solved, every target
    scored, the open ones short and without a row joining the next round — until none is short.
    Returns (accepted, x, status, solver, A, rows, cols, obj): the last solver and the tables it was built from."""
    idx = rowgen.round0() if (rowgen is not None and npass == 1) else np.zeros(0, dtype=np.int64)
    score, solver = None, None
    for rnd in range(settings.ROWGEN_MAX_ROUNDS if rowgen is not None else 1):
        t = time.time()
        nnz = 0
        if rowgen is not None and idx.size:
            block, table, idx = rowgen.rows_for(idx)
            first = rows.sizes['row']
            A, rows = append_rows(A, rows, block, table)
            rowgen.placed(idx, first, rnd, npass)
            if npass == 1:
                A, rows, cols, obj = relax(A, rows, cols, obj, first + np.flatnonzero(rowgen.mode[idx] == 'may_drop'))
            nnz = int(block.nnz)
        t_add = time.time() - t
        if solver is not None:
            solver.gurobi_model.dispose()                                                    # the previous round's model: freed before the next is built
        solver = LutoSolver(cols, rows, A, scale_cost(obj, cols, cost_scale), inputs)
        solver.formulate()
        if on_first_model is not None:
            on_first_model(solver.gurobi_model)
            on_first_model = None
        t = time.time()
        accepted, x, status = solver.solve_with_retries(target_year, params)
        t_solve = time.time() - t
        if rowgen is None:
            return accepted, x, status, solver, A, rows, cols, obj
        rec = dict(pass_=npass, round=rnd, added=int(idx.size), added_entries=nnz, rows=int(((rowgen.row_of >= 0) & ~rowgen.dropped).sum()),
                   status=int(status) if status is not None else None, add_s=round(t_add, 1), solve_s=round(t_solve))
        if not accepted:
            rowgen.rounds.append(rec)
            return accepted, x, status, solver, A, rows, cols, obj
        score = rowgen.score(x)
        short_out = rowgen.short_without_row(score)
        s = rowgen.shortfall(rows, x) if npass == 1 else np.zeros(0)
        rec.update(short_in_model=int((s > settings.ROWGEN_SHORT_TOL).sum()), sum_s=float(s.sum()), short_outside=int(short_out.size),
                   bar_iters=int(solver.gurobi_model.BarIterCount), obj=float(solver.gurobi_model.ObjVal))
        rowgen.rounds.append(rec)
        print(f"│   row generation, {target_year}, pass {npass} round {rnd}: +{idx.size:,} rows ({nnz:,} entries) -> "
              f"{rec['rows']:,} rows; solve {t_solve:.0f} s; short with a row {rec['short_in_model']:,} (Σ s {rec['sum_s']:.3f}), "
              f"short without a row {short_out.size:,}", flush=True)
        if short_out.size == 0:
            break
        idx = rowgen.hardest(short_out, score)
    else:
        print(f"│   row generation, {target_year}: pass {npass} stopped at the round cap ({settings.ROWGEN_MAX_ROUNDS}) with "
              f"{short_out.size:,} open targets short and without a row — NOT converged", flush=True)
    rowgen.final = score
    return accepted, x, status, solver, A, rows, cols, obj


# ═══════════════════════════ the screen and the starting point ═══════════════════════════

def cell_floor_ceiling(cols: xr.Dataset, support: ColSupport, c: np.ndarray, ncells: int) -> tuple[np.ndarray, np.ndarray]:
    """Per cell: the least and the most habitat (contribution × share, dimensionless) any feasible point can give it —
    every cell's ag + non-ag shares sum to its base (node balance), locked shares (lb) stay, and no column holds more
    than its upper bound. The bounds are fractional and carry the transition rules (a target's ub is the base-year
    share of the cell's land uses that can reach it; Destocked is capped at the livestock-on-natural share, Riparian
    Plantings at the stream-buffer share), so the ceiling fills the cell's columns from the best contribution down,
    each up to its ub, until the cell's share is used — and the floor the same from the worst up."""
    block = cols['block'].values
    cell = cols['cell'].values
    lb = cols['lb'].values.astype(np.float64)
    ub = cols['ub'].values.astype(np.float64)
    base = cols['base'].values.astype(np.float64)

    # ── an ag column's contribution, lowered (raised) by every negative (positive) ag-mgt effect it may host ──
    eff_lo = c.copy()
    eff_hi = c.copy()
    am = np.flatnonzero(block == 'am')
    host = support.ag_mrj2col[cols['m'].values[am], cell[am], cols['j'].values[am]]
    np.add.at(eff_lo, host, np.minimum(c[am], 0.0))
    np.add.at(eff_hi, host, np.maximum(c[am], 0.0))

    # ── per cell: the base, the locked share, the least and the most contribution over its open columns ──
    land = np.flatnonzero(np.isin(block, ('ag', 'nonag')))                  # the columns whose shares sum to the cell's base
    base_r = np.bincount(cell[land], weights=base[land], minlength=ncells)
    lb_r = np.bincount(cell[land], weights=lb[land], minlength=ncells)
    free_r = np.maximum(base_r - lb_r, 0.0)                                 # the share of the cell that is not locked
    open_ = land[ub[land] > 0]
    min_c = np.full(ncells, np.inf)
    max_c = np.full(ncells, -np.inf)
    np.minimum.at(min_c, cell[open_], eff_lo[open_])
    np.maximum.at(max_c, cell[open_], eff_hi[open_])
    min_c[~np.isfinite(min_c)] = 0.0
    max_c[~np.isfinite(max_c)] = 0.0

    # ── the free share laid on the open columns in order of contribution, each up to its room (ub − lb) ──
    room = np.maximum(ub - lb, 0.0)

    def fill(value: np.ndarray, best_first: bool, rest: np.ndarray) -> np.ndarray:
        """Per cell: the free share on the open columns, best (or worst) ``value`` first, each up to its room; a share
        no room covers is scored at ``rest`` (the cell's own best / worst contribution), so the bound stays a bound."""
        order = np.lexsort((-value[open_] if best_first else value[open_], cell[open_]))
        col = open_[order]
        cell_of = cell[col]
        used = np.cumsum(room[col])
        first = np.r_[True, cell_of[1:] != cell_of[:-1]]
        before = used - room[col] - np.maximum.accumulate(np.where(first, used - room[col], 0.0))   # room already used in this cell
        take = np.clip(free_r[cell_of] - before, 0.0, room[col])
        left = np.maximum(free_r - np.bincount(cell_of, weights=take, minlength=ncells), 0.0)
        return np.bincount(cell_of, weights=take * value[col], minlength=ncells) + left * rest

    # ── the floor: the locked share at its least, the free share worst first; the ceiling: at its most, best first ──
    floor_r = np.bincount(cell[land], weights=lb[land] * eff_lo[land], minlength=ncells) + fill(eff_lo, False, min_c)
    ceil_r = np.bincount(cell[land], weights=lb[land] * eff_hi[land], minlength=ncells) + fill(eff_hi, True, max_c)
    return floor_r, ceil_r


def start_point(data, cols: xr.Dataset, inputs: RowInputs, base_year: int) -> np.ndarray:
    """The base year's land use as a point over the columns: an ag / non-ag column at its base share (the node-balance
    constant), an ag-mgt column at the base year's share of its option, every arc at 0."""
    x0 = np.where(np.isin(cols['block'].values, ('ag', 'nonag')), cols['base'].values, 0.0).astype(np.float64)
    am = np.flatnonzero(cols['block'].values == 'am')
    am_idx, m, cell, j, j_idx = (cols[f].values[am] for f in ('am_idx', 'm', 'cell', 'j', 'j_idx'))
    for option_idx, (option, lus) in enumerate(inputs.agman2lu.items()):
        X = data.ag_man_dvars[base_year].get(option)
        if X is None:
            continue
        on = am_idx == option_idx
        lu = j[on] if X.shape[-1] != len(lus) else j_idx[on]              # the stored arrays carry the full lu axis, or the option's own
        x0[am[on]] = np.asarray(X)[m[on], cell[on], lu]
    return x0
