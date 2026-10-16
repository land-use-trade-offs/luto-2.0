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
from functools import cached_property

import numpy as np
import pandas as pd
import xarray as xr

from scipy import sparse

from luto import settings
from luto.solvers.col_builder import ColSupport
from luto.solvers.row_builder import get_bio_contribution, build_GBF_rows, get_bio_targets
from luto.solvers.row_inputs import RowInputs
from luto.solvers.row_table import ROW_FILL
from luto.solvers.solver import build_solver
from luto.solvers.tables import Tables, append_rows, relax, shortfall

# ═══════════════════════════ RowGen: the row generation of one step ═══════════════════════════

class RowGen:
    """The targets of one step and their bookkeeping: built after ``get_rows`` (while the column support still holds
    ``cell2col`` / ``ag_mrj2col``); ``solve_rounds`` (below) asks it for the rows due and tells it where they landed; then
    ``report``. It never touches a solver.

    Every per-target fact lives in ONE table, ``targets`` (a DataFrame, one row per target, positional index — what
    ``rowgen_<year>.csv`` is written from): the target (``family``, ``name``, ``rhs``, the labels, ``mode``), the screen
    (``floor``, ``ceiling``, ``class``), the start (``start``, ``carried``), ``eager``, and the bookkeeping of the step
    (``row_of``, ``round_added``, ``pass_added``, ``dropped``, ``shortfall_pass1``, ``final``). Beside it, per target ×
    cell, ``W``; per column, ``bio_coeff`` and ``bio_S``."""

    LABELS = ('region', 'GBF_target', 'GBF4_presence')                       # the label columns a target's row carries

    # ─────────────── 1. set up: the targets, the screen, the starting point, eager / lazy ───────────────

    def __init__(self, data, inputs: RowInputs, cols: xr.Dataset, support: ColSupport, base_year: int, target_year: int,
                 carry: set | None = None):
        t0 = time.time()
        self.year = target_year
        self.cell = cols['cell'].values
        self.ncells = inputs.ncells
        self.rounds = []                                                       # one record per solve of the rounds
        self._set_up_targets(inputs, support)
        self._screen(inputs, cols, support)
        self._start(data, cols, inputs, base_year, carry)
        self._eager_or_lazy()
        self._bookkeeping()
        self._log(base_year, t0)

    def _set_up_targets(self, inputs: RowInputs, support: ColSupport) -> None:
        """1a. Every family's targets as one table and one (target × cell) weight matrix; a target's mode."""
        targets, self.W = get_bio_targets(inputs, support)                    # row_builder: the families' get_<family>_targets, concatenated
        targets['mode'] = np.where(np.isin(targets['family'].values.astype(str), settings.ELASTIC_FAMILIES), 'may_drop', 'hard').astype(object)
        self.targets = targets

    def _screen(self, inputs: RowInputs, cols: xr.Dataset, support: ColSupport) -> None:
        """1b. The screen: every target safe / unattainable / open, from the least and the most habitat of every cell."""
        self.bio_coeff = get_bio_contribution(inputs, cols)                    # the habitat contribution of every column
        self.bio_S = (support.cell2col @ sparse.diags(self.bio_coeff)).tocsr()  # (cell × col), as get_rows lays it
        floor_r, ceil_r = cell_floor_ceiling(cols, support, self.bio_coeff.astype(np.float64), self.ncells)
        floor, ceiling = self.W @ floor_r, self.W @ ceil_r
        rhs = self.rhs
        margin = settings.ROWGEN_SCREEN_TOL * np.abs(rhs)
        screen_class = np.full(rhs.size, 'open', dtype=object)
        screen_class[ceiling < rhs - margin] = 'unattainable'
        screen_class[(floor >= rhs + margin) | (rhs <= 0)] = 'safe'           # rhs <= 0: the land outside LUTO already meets it
        self.targets['floor'] = floor
        self.targets['ceiling'] = ceiling
        self.targets['class'] = screen_class

    def _start(self, data, cols: xr.Dataset, inputs: RowInputs, base_year: int, carry: set | None) -> None:
        """1c. The starting point: the previous step's land use, scored at this year's layers; the targets carried."""
        self.targets['start'] = self.W @ self.habitat(start_point(data, cols, inputs, base_year))
        self.targets['carried'] = np.isin(self.name, list(carry or ())) & self.open

    def _eager_or_lazy(self) -> None:
        """1d. Eager or lazy, per family: the entries its open rows would take."""
        entries = np.diff(self.W.indptr) * (self.bio_S.nnz / self.ncells)      # per target: cells × columns per cell
        family, open_ = self.family, self.open
        eager = np.zeros(family.size, dtype=bool)
        for fam in pd.unique(family):
            on = family == fam
            eager[on] = entries[on & open_].sum() <= settings.ROWGEN_EAGER_MAX_ENTRIES
        self.targets['eager'] = eager

    def _bookkeeping(self) -> None:
        """1e. What the step records per target."""
        n_targets = len(self.targets)
        self.targets['row_of'] = np.full(n_targets, -1, dtype=np.int64)               # its row in the solver's table (-1: none)
        self.targets['round_added'] = np.full(n_targets, -1, dtype=np.int32)          # the round its row entered (-1: never)
        self.targets['pass_added'] = np.zeros(n_targets, dtype=np.int8)               # 1 / 2: the pass it entered in
        self.targets['dropped'] = np.zeros(n_targets, dtype=bool)                     # pass 1 left it short: its row removed before pass 2
        self.targets['shortfall_pass1'] = np.full(n_targets, np.nan)                  # pass 1's shortfall, where the target had a relaxed row
        self.targets['final'] = np.full(n_targets, np.nan, dtype=np.float32)          # the score at the step's solution (float32, as W gives it)

    def _log(self, base_year: int, t0: float) -> None:
        """1f. The log."""
        short0 = self.is_short(self.start)
        family, screen_class, mode = self.family, self.screen_class, self.mode
        for fam in pd.unique(family):
            on = family == fam
            count = {screen: int((screen_class[on] == screen).sum()) for screen in ('safe', 'unattainable', 'open')}
            print(f"│   row generation, {self.year}, {fam}: screen {count['safe']:,} safe / {count['unattainable']:,} unattainable / {count['open']:,} open "
                  f"of {int(on.sum()):,} targets; {int((short0 & self.open & on).sum()):,} open short at the {base_year} land use; "
                  f"{'eager' if self.eager[on].any() else 'lazy'}, {mode[on][0]}", flush=True)
        print(f"│   row generation, {self.year}: {int(self.carried.sum()):,} carried from the previous step ({time.time() - t0:.0f} s)", flush=True)
        if self.blocked.any():
            print(f"│   row generation, {self.year}: {int(self.blocked.sum()):,} HARD target(s) no feasible point can meet: "
                  + ", ".join(self.name[self.blocked][:10].tolist()) + (" …" if self.blocked.sum() > 10 else ""), flush=True)

    # ─────────────── 2. the table's columns as arrays (read-only views), and the one setter ───────────────
    #     The columns set-up fixes (1a–1d) are read once and cached; the step's bookkeeping (1e, written by ``_set``) is read
    #     off the table at every access.

    @cached_property
    def family(self) -> np.ndarray:
        return self.targets['family'].values.astype(str)

    @cached_property
    def name(self) -> np.ndarray:
        return self.targets['name'].values.astype(object)

    @cached_property
    def rhs(self) -> np.ndarray:
        """The target inside LUTO, raw units (ha; GBF8 suitability-weighted)."""
        return self.targets['rhs'].values.astype(np.float64)

    @cached_property
    def mode(self) -> np.ndarray:
        return self.targets['mode'].to_numpy()

    @cached_property
    def screen_class(self) -> np.ndarray:
        return self.targets['class'].to_numpy()

    @cached_property
    def open(self) -> np.ndarray:
        return self.screen_class == 'open'

    @cached_property
    def blocked(self) -> np.ndarray:
        """A hard target no feasible point can meet: the year cannot be solved."""
        return (self.screen_class == 'unattainable') & (self.mode == 'hard')

    @cached_property
    def start(self) -> np.ndarray:
        return self.targets['start'].to_numpy()

    @cached_property
    def carried(self) -> np.ndarray:
        return self.targets['carried'].to_numpy()

    @cached_property
    def eager(self) -> np.ndarray:
        return self.targets['eager'].to_numpy()

    row_of = property(lambda self: self.targets['row_of'].to_numpy())
    dropped = property(lambda self: self.targets['dropped'].to_numpy())

    def _set(self, column: str, idx, value) -> None:
        """``targets[column][idx] = value``, the column's dtype kept (the table is copy-on-write: a column is replaced whole).
        Only the bookkeeping columns (1e) are set this way; a cached column above is never written after set-up."""
        values = self.targets[column].to_numpy(copy=True)
        values[idx] = value
        self.targets[column] = values

    # ─────────────── 3. the pieces: habitat, shortness, the batch ───────────────

    def habitat(self, x: np.ndarray) -> np.ndarray:
        """Per cell: the habitat of the land use x (Σ bio_coeff · x over the cell's columns), float32 for the product with W."""
        return np.bincount(self.cell, weights=self.bio_coeff.astype(np.float64) * x[:self.cell.size], minlength=self.ncells).astype(np.float32)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Every target's score at x: W @ H(x), one product, no rows built."""
        return self.W @ self.habitat(x)

    def is_short(self, score: np.ndarray) -> np.ndarray:
        rhs = self.rhs
        return score < rhs - settings.ROWGEN_SHORT_TOL * np.abs(rhs)

    def hardest(self, idx: np.ndarray, score: np.ndarray) -> np.ndarray:
        """``idx`` ordered by relative shortfall at ``score``, largest first; a LAZY target's batch capped — a share of
        the targets short (settings.ROWGEN_BATCH_SHARE) within [ROWGEN_BATCH_MIN, ROWGEN_BATCH_MAX] — an eager one's not."""
        eager, rhs = self.eager, self.rhs
        lazy = idx[~eager[idx]]
        batch = int(np.clip(np.ceil(settings.ROWGEN_BATCH_SHARE * lazy.size), settings.ROWGEN_BATCH_MIN, settings.ROWGEN_BATCH_MAX))
        relative_short = (rhs[lazy] - score[lazy]) / np.abs(rhs[lazy])
        return np.union1d(idx[eager[idx]], lazy[np.argsort(-relative_short, kind='stable')][:batch])

    def round0(self) -> np.ndarray:
        """The targets round 0 starts with: the carried ones, every open target of an eager family, and the open
        targets of a lazy family short at the starting point, hardest first."""
        open_idx = np.flatnonzero(self.open)
        carried, eager = self.carried, self.eager
        short = open_idx[self.is_short(self.start)[open_idx] & ~carried[open_idx] & ~eager[open_idx]]
        return np.union1d(np.union1d(np.flatnonzero(carried), open_idx[eager[open_idx]]), self.hardest(short, self.start))

    def short_without_row(self, score: np.ndarray) -> np.ndarray:
        """The open targets without a row that are short at ``score``: the next round's candidates."""
        return np.flatnonzero(self.open & (self.row_of < 0) & ~self.dropped & self.is_short(score))

    # ─────────────── 4. the rows of a round, and where they landed ───────────────

    def rows_for(self, idx: np.ndarray):
        """The rows of the targets ``idx`` as one part, family by family (``row_builder.build_GBF_rows``, as the rows built
        up front): (block, table, order), row i of the block being target order[i]."""
        family = self.family
        blocks, tables, order = [], [], []
        for fam in pd.unique(family[idx]):
            on = idx[family[idx] == fam]
            targets = self.targets.iloc[on][['rhs', 'name', *self.LABELS]].assign(row=True)
            block, table = build_GBF_rows(fam, (targets, self.W[on]), self.bio_S)
            blocks.append(block)
            tables.append(table)
            order.append(on)
        return sparse.vstack(blocks, format='csr'), xr.concat(tables, dim='row', fill_value=ROW_FILL), np.concatenate(order)

    def placed(self, idx: np.ndarray, first: int, rnd: int, npass: int) -> None:
        """The targets ``idx`` (in the part's row order) now hold rows ``first`` … in the row table."""
        self._set('row_of', idx, first + np.arange(idx.size))
        self._set('round_added', idx, rnd)
        self._set('pass_added', idx, npass)

    def final(self, score: np.ndarray) -> None:
        """The score at the pass's solution: every target's ``final``."""
        self.targets['final'] = score

    # ─────────────── 5. pass 1's shortfall, and the drop (step.solve_step decides; this records) ───────────────

    def shortfall(self, rows: xr.Dataset, x: np.ndarray) -> np.ndarray:
        """The fraction of the target missed, per generated row that was relaxed (0 where hard or absent)."""
        row_of = self.row_of
        return np.nan_to_num(shortfall(rows, x, row_of[row_of >= 0]))

    def record_pass1(self, rows: xr.Dataset, x: np.ndarray, dropped_rows: np.ndarray) -> None:
        """Pass 1's shortfall kept per target, and the targets whose rows the step dropped marked."""
        row_of = self.row_of
        self._set('shortfall_pass1', row_of >= 0, self.shortfall(rows, x))
        self._set('dropped', np.isin(row_of, dropped_rows) & (row_of >= 0), True)

    # ─────────────── 6. after the step: the next step's start, the report ───────────────

    def carry_next(self, rows: xr.Dataset) -> set:
        """The targets whose rows the next step starts with: those whose row binds at this step's end (π ≠ 0 on the solved
        row table), or every row still in the model when no duals were read."""
        row_of = self.row_of
        kept = np.flatnonzero((row_of >= 0) & ~self.dropped)
        if 'pi' in rows:
            pi = rows['pi'].values[row_of[kept]]
            kept = kept[np.isfinite(pi) & (np.abs(pi) > 0)]
        return set(self.name[kept].tolist())

    def report(self, rows: xr.Dataset, out_dir: str) -> None:
        """rowgen_<year>.csv (every target) and rowgen_rounds_<year>.csv; the counts in the log. ``rows``: the solved row
        table (the solver's, carrying ``pi``)."""
        targets = self.targets
        rhs, row_of, dropped, screen_class = self.rhs, self.row_of, self.dropped, self.screen_class
        family, open_ = self.family, self.open
        # ── 6a. every target ──
        pi = np.full(rhs.size, np.nan)
        in_model = (row_of >= 0) & ~dropped
        if 'pi' in rows:
            pi[in_model] = rows['pi'].values[row_of[in_model]] * 1e6 / rows['scale'].values[row_of[in_model]]   # AUD per unit, as shadow_prices_<year>.csv
        final = targets['final'].to_numpy()
        below = self.is_short(final)
        status = np.where(dropped, 'dropped', screen_class)                    # safe / unattainable / open / dropped

        def frac(score):
            """The score as a fraction of the target."""
            return np.divide(score, rhs, out=np.full(rhs.size, np.nan), where=rhs != 0)

        floor, ceiling, start = targets['floor'].to_numpy(), targets['ceiling'].to_numpy(), self.start
        df = pd.DataFrame({'family': family, 'name': self.name,
                           **{field: targets[field].values for field in self.LABELS if field in targets},
                           'class': screen_class, 'mode': self.mode, 'status': status,
                           'target': rhs, 'floor': floor, 'ceiling': ceiling, 'start': start, 'final': final,
                           'floor_frac': frac(floor), 'ceiling_frac': frac(ceiling), 'start_frac': frac(start), 'final_frac': frac(final),
                           'carried': self.carried, 'eager': self.eager, 'pass_added': targets['pass_added'].to_numpy(),
                           'round_added': targets['round_added'].to_numpy(), 'shortfall_pass1': targets['shortfall_pass1'].to_numpy(),
                           'dropped': dropped, 'shadow_price': pi, 'below_target': below})
        df.to_csv(f"{out_dir}/rowgen_{self.year}.csv", index=False)
        # ── 6b. every round ──
        pd.DataFrame(self.rounds).to_csv(f"{out_dir}/rowgen_rounds_{self.year}.csv", index=False)
        # ── 6c. the checks: a safe target below target disproves the screen; an open one not dropped, the loop ──
        for fam in pd.unique(family):
            in_family = family == fam
            bad = int((below & in_family & (screen_class == 'safe')).sum())
            missed = int((below & in_family & open_ & ~dropped).sum())
            print(f"│   row generation, {self.year}, {fam}: {int((in_model & in_family).sum()):,} rows in the final model "
                  f"({int(((np.abs(np.nan_to_num(pi)) > 0) & in_family).sum()):,} binding), {int((dropped & in_family).sum()):,} dropped, "
                  f"{int(((screen_class == 'unattainable') & in_family).sum()):,} unattainable; below target: "
                  f"{int((below & in_family).sum()):,} (safe {bad} — must be 0; open not dropped {missed} — must be 0)", flush=True)
            if bad or missed:
                print(f"WARNING: row generation, {self.year}, {fam}: {bad} safe and {missed} open targets below target at the solution", flush=True)


# ═══════════════════════════ the rounds of one pass ═══════════════════════════

def solve_rounds(tables: Tables, inputs: RowInputs, rowgen: RowGen, target_year: int, npass: int, params, cost_scale: float, on_first_model=None):
    """One pass by row generation (``step.solve_pass``), round by round: the rows of the targets due appended
    (``append_rows``; in pass 1 a may-drop target's row relaxed), a fresh solver built and solved, every target scored,
    the open ones short and without a row joining the next round — until none is short.
    Returns (accepted, x, status, solver, tables): the last solver and the tables it was built from."""
    idx = rowgen.round0() if npass == 1 else np.zeros(0, dtype=np.int64)
    score, solver = None, None
    for rnd in range(settings.ROWGEN_MAX_ROUNDS):
        t_start = time.time()
        nnz = 0
        if idx.size:
            block, table, idx = rowgen.rows_for(idx)
            first = tables.rows.sizes['row']
            tables = append_rows(tables, block, table)
            rowgen.placed(idx, first, rnd, npass)
            if npass == 1:
                tables = relax(tables, first + np.flatnonzero(rowgen.mode[idx] == 'may_drop'))
            nnz = int(block.nnz)
        t_add = time.time() - t_start
        if solver is not None:
            solver.gurobi_model.dispose()                                                    # the previous round's model: freed before the next is built
        solver = build_solver(tables, inputs, cost_scale, on_first_model)
        on_first_model = None
        t_start = time.time()
        accepted, x, status = solver.solve_with_retries(target_year, params)
        t_solve = time.time() - t_start
        round_record = {'pass': npass, 'round': rnd, 'added': int(idx.size), 'added_entries': nnz,
                        'rows': int(((rowgen.row_of >= 0) & ~rowgen.dropped).sum()),
                        'status': int(status) if status is not None else None, 'add_s': round(t_add, 1), 'solve_s': round(t_solve)}
        if not accepted:
            rowgen.rounds.append(round_record)
            return accepted, x, status, solver, tables
        score = rowgen.score(x)
        short_out = rowgen.short_without_row(score)
        shortfall_now = rowgen.shortfall(tables.rows, x) if npass == 1 else np.zeros(0)  # the relaxed rows' shortfall (pass 1)
        round_record.update(short_in_model=int((shortfall_now > settings.ELASTIC_DROP_TOL).sum()), sum_s=float(shortfall_now.sum()), short_outside=int(short_out.size),
                   bar_iters=int(solver.gurobi_model.BarIterCount), obj=float(solver.gurobi_model.ObjVal))
        rowgen.rounds.append(round_record)
        print(f"│   row generation, {target_year}, pass {npass} round {rnd}: +{idx.size:,} rows ({nnz:,} entries) -> "
              f"{round_record['rows']:,} rows; solve {t_solve:.0f} s; short with a row {round_record['short_in_model']:,} (Σ s {round_record['sum_s']:.3f}), "
              f"short without a row {short_out.size:,}", flush=True)
        if short_out.size == 0:
            break
        idx = rowgen.hardest(short_out, score)
    else:
        print(f"│   row generation, {target_year}: pass {npass} stopped at the round cap ({settings.ROWGEN_MAX_ROUNDS}) with "
              f"{short_out.size:,} open targets short and without a row — NOT converged", flush=True)
    rowgen.final(score)
    return accepted, x, status, solver, tables


# ═══════════════════════════ the screen and the starting point ═══════════════════════════

def cell_floor_ceiling(cols: xr.Dataset, support: ColSupport, bio_coeff: np.ndarray, ncells: int) -> tuple[np.ndarray, np.ndarray]:
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
    eff_lo = bio_coeff.copy()
    eff_hi = bio_coeff.copy()
    am = np.flatnonzero(block == 'am')
    host = support.ag_mrj2col[cols['m'].values[am], cell[am], cols['j'].values[am]]
    np.add.at(eff_lo, host, np.minimum(bio_coeff[am], 0.0))
    np.add.at(eff_hi, host, np.maximum(bio_coeff[am], 0.0))

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
    am_idx, m, cell, j, j_idx = (cols[field].values[am] for field in ('am_idx', 'm', 'cell', 'j', 'j_idx'))
    for option, lus in inputs.agman2lu.items():
        shares = data.ag_man_dvars[base_year].get(option)              # the option's base-year shares, (lm, cell, lu)
        if shares is None:
            continue
        on = am_idx == inputs.am_options.index(option)
        lu = j[on] if shares.shape[-1] != len(lus) else j_idx[on]              # the stored arrays carry the full lu axis, or the option's own
        x0[am[on]] = np.asarray(shares)[m[on], cell[on], lu]
    return x0
