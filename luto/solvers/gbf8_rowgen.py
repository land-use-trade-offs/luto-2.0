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
GBF8 by ROW GENERATION (settings.GBF8_ROW_GENERATION): every species' target, exactly, without building every row.

A GBF8 row is one species over every column with a habitat contribution in the cells it occupies (~0.5 M entries at
RES5); all ~10.6 k of them are ~5 × 10⁹ entries, ~350 GB. Most never bind, so the step solves with only the rows that
matter, found round by round. Per step:

  0. SCREEN (before the solve). Every cell's ag + non-ag shares sum to its base, so per cell
         floor_r = Σ lb·c + (base_r − Σ lb) · min c        the least habitat any feasible point leaves it
         ceil_r  = base_r · max c                           the most
     (min / max c over the cell's open ag / non-ag columns, an ag column's contribution lowered / raised by every
     negative / positive ag-mgt effect it may host). Per species floor_s = W_s · floor, ceil_s = W_s · ceil:
         SAFE          floor_s ≥ rhs      met at every feasible point: no row, ever
         UNATTAINABLE  ceil_s  < rhs      met at none: no row (it would make the step infeasible), still scored (D11)
         OPEN          otherwise          a candidate row
  1. PASS 1 — find the rows. Round 0 adds the species with a row at the previous step's end (``carry``) and the open
     species short at the previous step's land use (scored at this year's layers), hardest first; each row with a
     shortfall variable s in [0, 1] at settings.ELASTIC_PENALTY per target missed (a·x + rhs·s ≥ rhs), so no round is
     infeasible. Solve; score EVERY species at x (W @ H(x), one product, no rows built); the open species WITHOUT a row
     that are short join the next round, hardest first. Stop when none is short: x is then optimal for the model with
     every open species' row. The species left with s > 0 cannot be met together with the rest of the step.
  2. PASS 2 — the step as a policy. Those species' rows are dropped (still scored, listed as ``dropped``) and every
     shortfall variable removed, so every remaining row is hard; re-solve, and add as HARD rows any species pushed
     under by the land the dropped ones no longer hold. Pass 1's x meets every species not dropped, so pass 2 is never
     infeasible. Skipped when pass 1 left no species short (its x is then optimal for the hard model too).

The rows are the ``get_GBF8`` rows (``W[species] @ bio_S``, contracted and rescaled row by row, so building them in
batches changes nothing), appended to the live model (``addMConstr``) and to the solver's row table and A, so the
shadow prices and every reader of the table see them. Written per year: ``GBF8_rowgen_<year>.csv`` (every species:
class, round and pass its row entered, s, dropped, attainment at the start and at the solution) and
``GBF8_rowgen_rounds_<year>.csv``.
"""

import time

import numpy as np
import pandas as pd
import xarray as xr

from scipy import sparse

from luto import settings
from luto.solvers.col_builder import ColSupport
from luto.solvers.row_builder import bio_contribution, contract
from luto.solvers.row_inputs import RowInputs
from luto.solvers.row_table import ROW_FILL, make_part

# ═══════════════════════════ tolerances ═══════════════════════════

SCREEN_TOL = 1e-6      # relative: the screen must clear the target by more than this to call a species safe / unattainable
SHORT_TOL = 1e-4       # relative: a species is short when its score is below its target by more than this
S_TOL = 1e-6           # a shortfall variable above this is a target missed


# ═══════════════════════════ GBF8RowGen: the row generation of one step ═══════════════════════════

class GBF8RowGen:
    """The GBF8 row generation of one step: built after ``get_rows`` (while the column support still holds
    ``cell2col`` / ``ag_mrj2col``), then ``solve`` in place of ``solve_with_retries``, then ``report``."""

    # ─────────────── 1. set up: the inputs, the screen, the starting point ───────────────

    def __init__(self, data, inputs: RowInputs, cols: xr.Dataset, support: ColSupport, base_year: int, target_year: int,
                 carry: set | None = None):
        t0 = time.time()

        # ── 1a. the inputs: layers, targets, the contribution of every column ──
        self.year = target_year
        self.species = np.asarray(data.BIO_GBF8_SEL_SPECIES, dtype=object)
        self.W = inputs.GBF8_pre_1750_area_sr.data                            # (species, cell) float32: suitability · LDS · REAL_AREA
        self.rhs = inputs.limits['GBF8'].values.astype(np.float64)             # the target inside LUTO, per species (ha)
        self.outside = np.asarray(data.get_GBF8_score_outside_natural_LUTO_by_yr(target_year), dtype=np.float64)
        self.base_total = data.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES['HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA'].values.astype(np.float64)
        self.cell = cols['cell'].values
        self.ncells = inputs.ncells
        self.c = bio_contribution(inputs, cols)                                # the habitat contribution of every column
        self.bio_S = (support.cell2col @ sparse.diags(self.c)).tocsr()         # (cell x col), as get_rows lays it

        # ── 1b. the screen: safe / unattainable / open ──
        floor_r, ceil_r = cell_floor_ceiling(cols, support, self.c.astype(np.float64), self.ncells)
        self.floor = self.W @ floor_r.astype(np.float32)
        self.ceil = self.W @ ceil_r.astype(np.float32)
        margin = SCREEN_TOL * np.abs(self.rhs)
        cls = np.full(self.species.size, 'open', dtype=object)
        cls[self.ceil < self.rhs - margin] = 'unattainable'
        cls[(self.floor >= self.rhs + margin) | (self.rhs <= 0)] = 'safe'     # rhs <= 0: the land outside LUTO already meets it
        self.cls = cls
        self.open = cls == 'open'

        # ── 1c. the starting point: the previous step's land use, scored at this year's layers ──
        self.start = self.W @ self.habitat(start_point(data, cols, inputs, base_year))
        self.carry = np.isin(self.species, list(carry or ())) & self.open

        # ── 1d. the bookkeeping, per species ──
        self.round_added = np.full(self.species.size, -1, dtype=np.int32)    # the round a species' row entered (-1: never)
        self.pass_added = np.zeros(self.species.size, dtype=np.int8)         # 1 / 2: the pass it entered in
        self.dropped = np.zeros(self.species.size, dtype=bool)               # pass 1 left it short: its row removed in pass 2
        self.s = np.full(self.species.size, np.nan)                          # pass 1's shortfall, where the species had a row
        self.row_of = np.full(self.species.size, -1, dtype=np.int64)         # its row in the solver's table
        self.s_var = {}                                                      # species index -> its shortfall variable (pass 1)
        self.final = None
        self.rounds = []
        # ── 1e. the log ──
        n = {k: int((cls == k).sum()) for k in ('safe', 'unattainable', 'open')}
        print(f"│   GBF8 row generation, {target_year}: screen {n['safe']:,} safe / {n['unattainable']:,} unattainable / "
              f"{n['open']:,} open of {self.species.size:,} species; {int(self.is_short(self.start)[self.open].sum()):,} open "
              f"short at the {base_year} land use, {int(self.carry.sum()):,} carried from the previous step "
              f"({time.time() - t0:.0f} s)", flush=True)

    # ─────────────── 2. the pieces: habitat, shortness, the batch ───────────────

    def habitat(self, x: np.ndarray) -> np.ndarray:
        """Per cell: the habitat of the land use x (Σ c · x over the cell's columns), float32 for the product with W."""
        return np.bincount(self.cell, weights=self.c.astype(np.float64) * x[:self.cell.size], minlength=self.ncells).astype(np.float32)

    def is_short(self, score: np.ndarray) -> np.ndarray:
        return score < self.rhs - SHORT_TOL * np.abs(self.rhs)

    def hardest(self, idx: np.ndarray, score: np.ndarray) -> np.ndarray:
        """``idx`` ordered by relative shortfall at ``score``, largest first, capped at the round's batch: a share of
        the species short (settings.GBF8_ROWGEN_BATCH_SHARE), within [GBF8_ROWGEN_BATCH_MIN, GBF8_ROWGEN_BATCH_MAX]."""
        batch = int(np.clip(np.ceil(settings.GBF8_ROWGEN_BATCH_SHARE * idx.size), settings.GBF8_ROWGEN_BATCH_MIN, settings.GBF8_ROWGEN_BATCH_MAX))
        rel = (self.rhs[idx] - score[idx]) / np.abs(self.rhs[idx])
        return idx[np.argsort(-rel, kind='stable')][:batch]

    # ─────────────── 3. one round: add the rows, solve, score ───────────────

    def add(self, solver, idx: np.ndarray, rnd: int, npass: int) -> int:
        """The rows of the species ``idx`` on the live model — with a shortfall variable each in pass 1, hard in pass 2 —
        appended to the solver's row table and A. Returns the entries added."""
        if idx.size == 0:
            return 0
        # ── 3a. the rows: W[species] @ bio_S, contracted and rescaled row by row (as get_GBF8) ──
        model = solver.gurobi_model
        block, rhs, scale = contract(sparse.csr_matrix(self.W[idx]) @ self.bio_S, self.rhs[idx], rescale=True)
        names = [f"bio_GBF8_limit_AUSTRALIA_{s}".replace(" ", "_") for s in self.species[idx]]
        # ── 3b. onto the live model: with a shortfall variable each (pass 1), or hard (pass 2) ──
        if npass == 1:
            sign = -1.0 if model.ModelSense == -1 else 1.0                                     # a shortfall always COSTS objective
            s_vars = model.addVars(idx.size, lb=0.0, ub=1.0, obj=sign * settings.ELASTIC_PENALTY / 1e6, name="GBF8_short")   # million AUD
            model.update()
            s_list = [s_vars[i] for i in range(idx.size)]
            constrs = model.addMConstr(sparse.hstack([block, sparse.diags(rhs)], format='csr'), solver._vars + s_list, '>', rhs).tolist()   # a·x + rhs·s ≥ rhs, scaled
            self.s_var.update(zip(idx.tolist(), s_list))
        else:
            constrs = model.addMConstr(block, solver._vars, '>', rhs).tolist()
        model.setAttr('ConstrName', constrs, names)
        model.update()

        # ── 3c. into the solver's row table and A (row i of A is row i of the table) ──
        _, table = make_part('GBF8', block, rhs, '>', names, scale, region=['AUSTRALIA'] * idx.size, GBF_target=self.species[idx])
        table['constr'] = (('row',), np.asarray(constrs, dtype=object))
        first = solver.rows.sizes['row']
        solver.rows = xr.concat([solver.rows, table], dim='row', fill_value=ROW_FILL | {'constr': None})
        solver.A = sparse.vstack([solver.A, block], format='csr')
        self.row_of[idx] = first + np.arange(idx.size)
        self.round_added[idx] = rnd
        self.pass_added[idx] = npass
        return int(block.nnz)

    def solve_round(self, solver, target_year: int, idx: np.ndarray, rnd: int, npass: int, solve_with_retries):
        """Add ``idx``, solve, score every species: (accepted, x, status, score, the open species without a row short)."""
        # ── 3d. add, then solve ──
        t = time.time()
        nnz = self.add(solver, idx, rnd, npass)
        t_add = time.time() - t
        t = time.time()
        accepted, x, status = solve_with_retries(solver, target_year)
        t_solve = time.time() - t
        rec = dict(pass_=npass, round=rnd, added=int(idx.size), added_entries=nnz, rows=int((self.row_of >= 0).sum() - self.dropped.sum()),
                   status=int(status) if status is not None else None, add_s=round(t_add, 1), solve_s=round(t_solve))
        if not accepted:
            self.rounds.append(rec)
            return accepted, x, status, None, None
        # ── 3e. score every species at x; the open species without a row that are short ──
        score = self.W @ self.habitat(x)
        candidates = self.open & (self.row_of < 0) & ~self.dropped
        short_out = np.flatnonzero(candidates & self.is_short(score))
        s = np.array([self.s_var[i].X for i in np.flatnonzero(self.row_of >= 0) if i in self.s_var]) if npass == 1 else np.zeros(0)
        # ── 3f. the round's record ──
        model = solver.gurobi_model
        rec.update(short_in_model=int((s > S_TOL).sum()), sum_s=float(s.sum()), short_outside=int(short_out.size),
                   bar_iters=int(model.BarIterCount), obj=float(model.ObjVal))
        self.rounds.append(rec)
        print(f"│   GBF8 row generation, {target_year}, pass {npass} round {rnd}: +{idx.size:,} rows ({nnz:,} entries) -> "
              f"{rec['rows']:,} rows; solve {t_solve:.0f} s; short with a row {rec['short_in_model']:,} (Σ s {rec['sum_s']:.3f}), "
              f"short without a row {short_out.size:,}", flush=True)
        return accepted, x, status, score, short_out

    # ─────────────── 4. the step: pass 1, then pass 2 ───────────────

    def solve(self, solver, target_year: int, solve_with_retries):
        """Pass 1 and pass 2 on the formulated model (``solver.formulate()`` done). Returns (accepted, x, status) as
        ``solve_with_retries`` does, x the final solution over the column table."""
        max_rounds = settings.GBF8_ROWGEN_MAX_ROUNDS
        open_idx = np.flatnonzero(self.open)

        # ── 4a. pass 1: every row with its shortfall variable, round by round ──
        short0 = open_idx[self.is_short(self.start)[open_idx] & ~self.carry[open_idx]]
        idx = np.union1d(np.flatnonzero(self.carry), self.hardest(short0, self.start))
        for rnd in range(max_rounds):
            accepted, x, status, score, short_out = self.solve_round(solver, target_year, idx, rnd, 1, solve_with_retries)
            if not accepted:
                return accepted, x, status
            if short_out.size == 0:
                break
            idx = self.hardest(short_out, score)
        else:
            print(f"│   GBF8 row generation, {target_year}: pass 1 stopped at the round cap ({max_rounds}) with "
                  f"{short_out.size:,} open species short and without a row — NOT converged", flush=True)
        # ── 4b. what pass 1 leaves: the species short with a row ──
        rowed = np.flatnonzero(self.row_of >= 0)
        self.s[rowed] = [self.s_var[i].X for i in rowed]
        short = rowed[self.s[rowed] > S_TOL]

        # ── 4c. pass 2: the species pass 1 left short dropped, every other row hard ──
        if short.size:
            print(f"│   GBF8 row generation, {target_year}: {short.size:,} species cannot be met together with the step "
                  f"(pass 1 Σ s {self.s[rowed].sum():.3f}) — their rows dropped, every other GBF8 row made hard", flush=True)
            self.dropped[short] = True
            solver.remove_constraints_by_name(solver.rows['name'].values[self.row_of[short]].tolist())
            solver.gurobi_model.remove(list(self.s_var.values()))
            solver.gurobi_model.update()
            self.s_var = {}
            idx = np.zeros(0, dtype=np.int64)
            for rnd in range(max_rounds):
                accepted, x, status, score, short_out = self.solve_round(solver, target_year, idx, rnd, 2, solve_with_retries)
                if not accepted:
                    return accepted, x, status
                if short_out.size == 0:
                    break
                idx = self.hardest(short_out, score)
            else:
                print(f"│   GBF8 row generation, {target_year}: pass 2 stopped at the round cap ({max_rounds}) with "
                      f"{short_out.size:,} species short and without a row — NOT converged", flush=True)
        self.final = score
        return accepted, x, status

    # ─────────────── 5. after the step: the next step's start, the report ───────────────

    def carry_next(self, solver) -> set:
        """The species whose rows the next step starts with: those whose row binds at this step's end (π ≠ 0), or every
        row still in the model when no duals were read."""
        T = solver.rows
        kept = np.flatnonzero((self.row_of >= 0) & ~self.dropped)
        if 'pi' in T:
            pi = T['pi'].values[self.row_of[kept]]
            kept = kept[np.isfinite(pi) & (np.abs(pi) > 0)]
        return set(self.species[kept].tolist())

    def report(self, solver, out_dir: str) -> None:
        """GBF8_rowgen_<year>.csv (every species) and GBF8_rowgen_rounds_<year>.csv; the counts in the log."""
        # ── 5a. every species ──
        pct = lambda score: 100 * (score + self.outside) / self.base_total                    # noqa: E731 — attainment, as the target CSV's %
        pi = np.full(self.species.size, np.nan)
        on = (self.row_of >= 0) & ~self.dropped
        if 'pi' in solver.rows:
            pi[on] = solver.rows['pi'].values[self.row_of[on]] * 1e6 / solver.rows['scale'].values[self.row_of[on]]   # AUD per ha, as shadow_prices_<year>.csv
        final = self.final if self.final is not None else np.full(self.species.size, np.nan)
        below = self.is_short(final)
        status = np.where(self.dropped, 'dropped', self.cls)                                  # safe / unattainable / open / dropped
        df = pd.DataFrame({'species': self.species, 'class': self.cls, 'status': status,
                           'target_ha': self.rhs, 'floor_ha': self.floor, 'ceiling_ha': self.ceil,
                           'start_ha': self.start, 'final_ha': final,
                           'target_pct': pct(self.rhs), 'start_pct': pct(self.start), 'final_pct': pct(final),
                           'carried': self.carry, 'pass_added': self.pass_added, 'round_added': self.round_added,
                           'shortfall_pass1': self.s, 'dropped': self.dropped, 'shadow_price': pi, 'below_target': below})
        df.to_csv(f"{out_dir}/GBF8_rowgen_{self.year}.csv", index=False)
        # ── 5b. every round ──
        pd.DataFrame(self.rounds).rename(columns={'pass_': 'pass'}).to_csv(f"{out_dir}/GBF8_rowgen_rounds_{self.year}.csv", index=False)
        # ── 5c. the checks: a safe species below target disproves the screen; an open one not dropped, the loop ──
        bad = int((below & (self.cls == 'safe')).sum())
        missed = int((below & self.open & ~self.dropped).sum())
        print(f"│   GBF8 row generation, {self.year}: {int(on.sum()):,} rows in the final model "
              f"({int((np.abs(np.nan_to_num(pi)) > 0).sum()):,} binding), {int(self.dropped.sum()):,} dropped, "
              f"{int((self.cls == 'unattainable').sum()):,} unattainable; below target: "
              f"{int(below.sum()):,} (safe {bad} — must be 0; open not dropped {missed} — must be 0)", flush=True)
        if bad or missed:
            print(f"WARNING: GBF8 row generation, {self.year}: {bad} safe and {missed} open species below target at the solution", flush=True)


# ═══════════════════════════ the screen and the starting point ═══════════════════════════

def cell_floor_ceiling(cols: xr.Dataset, support: ColSupport, c: np.ndarray, ncells: int) -> tuple[np.ndarray, np.ndarray]:
    """Per cell: the least and the most habitat (contribution × share, dimensionless) any feasible point can give it —
    every cell's ag + non-ag shares sum to its base (node balance), locked shares (lb) stay."""
    block = cols['block'].values
    cell = cols['cell'].values
    lb = cols['lb'].values.astype(np.float64)
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
    locked_r = np.bincount(cell[land], weights=lb[land] * eff_lo[land], minlength=ncells)
    open_ = land[cols['ub'].values[land] > 0]
    min_c = np.full(ncells, np.inf)
    max_c = np.full(ncells, -np.inf)
    np.minimum.at(min_c, cell[open_], eff_lo[open_])
    np.maximum.at(max_c, cell[open_], eff_hi[open_])
    min_c[~np.isfinite(min_c)] = 0.0
    max_c[~np.isfinite(max_c)] = 0.0
    # ── the floor: the locked share at its own contribution, the rest at the least; the ceiling: all of it at the most ──
    floor_r = locked_r + np.maximum(base_r - lb_r, 0.0) * min_c
    ceil_r = base_r * max_c
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
