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
Bound propagation over the row space: every row's activity interval over the column box, and its verdict.

Every row is a linear form over columns whose bounds the column table already holds, so two sums per row give the
interval the form can occupy — lo = A⁺·lb + A⁻·ub, hi = A⁺·ub + A⁻·lb — and the interval against the rhs classifies
the row before any solver sees it: REDUNDANT (every point of the box satisfies it: dropping it is exact, whatever else
is in the model), IMPOSSIBLE (no point does: a diagnosis, never dropped), TIGHT (met only at the box's extreme),
NEAR_REDUNDANT (satisfied everywhere but within the margin: kept), STRADDLE (nothing learned).

The box alone overcounts: a policy row sums every column of a cell at its own ub while the cell-usage row caps their
sum, and the arcs have no ub at all. So the rows are also judged under the bounds the rows themselves imply —
(2a) a row whose entries are all positive, sense < or =, bounds each of its columns; (2b) a row whose entries are all
exactly 1 bounds its columns TOGETHER, so a row reaches at most its best coefficient times that group's room. Those
bounds hold at every feasible point, so IMPOSSIBLE under them is still a proof; a row that implies a bound keeps its box
verdict, and only the BOX verdict licenses a drop. Everything reads A, rhs, sense and scale off the row table and
lb / ub off the column table — no family, no engine — and the verdicts sit beside the row table on the same ``row``
dim. What no bound here sees is rows competing for the same cells: "nothing proven" is not "feasible".
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from luto import settings
from luto.solvers import row_table


STATUS = ('straddle', 'redundant', 'impossible', 'tight', 'near_redundant')      # the verdict codes, in code order
UNIT = {                                                                         # the report's unit per family (raw row = scaled row × scale); every other family is a structural row over shares
    'GBF2': 'ha', 'GBF3_NVIS': 'ha', 'GBF4_SNES': 'ha', 'GBF4_ECNES': 'ha', 'GBF8': 'ha',
    'water': 'ML', 'ghg': 'tCO2e', 'demand': 't', 'renewable': 'MWh',
    'regional_adoption_ag': 'ha', 'regional_adoption_nonag': 'ha', 'regional_adoption_nonag_sum': 'ha',
}


# ═══════════════════════════ get_row_bounds: the verdict of every row ═══════════════════════════

def get_row_bounds(rows: xr.Dataset, cols: xr.Dataset) -> xr.Dataset:
    """Every row's interval and verdict, as a table on the row table's own ``row`` dim: over the column box (``lo`` /
    ``hi`` / ``status``) and under the bounds the rows imply (``lo_implied`` / ``hi_implied`` / ``status_implied``),
    with the ``margin`` both are judged within (all in the table's scaled row space) and ``empty``; attrs
    ``preflight`` = {block: counts of the columns no row can explain}."""
    A = rows.attrs['A']
    lb = cols['lb'].values
    ub = cols['ub'].values
    rhs = rows['rhs'].values
    sense = rows['sense'].values

    # ── 1. the interval over the box: lo = A⁺·lb + A⁻·ub, hi = A⁺·ub + A⁻·lb, and the magnitude of the terms S = Σ|a|·|bound| ──
    lo, hi, S = row_intervals(A, lb, ub)

    # ── 2. the margin: relative to S, because the rounding of a sum grows with its terms (a water row nets yields
    #       against uses), and never under the solver's own tolerance — so IMPOSSIBLE is beyond anything the engine accepts ──
    margin = np.maximum(10 * settings.FEASIBILITY_TOLERANCE, settings.BOUND_PROP_REL_TOL * np.maximum(np.abs(rhs), S))

    # ── 3. the verdict over the box: the one a drop reads ──
    status = row_verdict(sense, rhs, lo, hi, margin)

    # ── 4. the bounds the rows imply: all-positive < / = rows bound each of their columns (2a), all-ones rows bound
    #       their columns together (2b); the margin rides on every implying rhs, so the bounds hold at anything the solver accepts ──
    source, unit = implying_rows(A, sense)
    ub_implied = implied_ub(A, rhs, lb, ub, margin, source)
    group_of_col, room_of_group = unit_groups(A, rhs, lb, margin, unit)
    lo_implied, hi_implied = grouped_intervals(A, lb, ub_implied, group_of_col, room_of_group)

    # ── 5. the verdict under them: a row that implied a bound keeps its box verdict — no row is judged by what it implied ──
    status_implied = np.where(source | unit, status, row_verdict(sense, rhs, lo_implied, hi_implied, margin)).astype(np.int8)

    # ── 6. the column preflight: columns no row touches, bounds no point can satisfy ──
    in_rows = np.bincount(A.indices, minlength=A.shape[1])
    preflight = {}
    for block, (start, stop) in cols.attrs['block_range'].items():
        span = slice(start, stop)
        preflight[block] = dict(columns=stop - start,
                                in_no_row=int((in_rows[span] == 0).sum()),
                                nan_bound=int(np.isnan(lb[span]).sum() + np.isnan(ub[span]).sum()),
                                lb_neg_inf=int(np.isneginf(lb[span]).sum()),
                                lb_above_ub=int((lb[span] > ub[span]).sum()))

    return xr.Dataset(
        dict(lo=(('row',), lo),
             hi=(('row',), hi),
             status=(('row',), status),
             lo_implied=(('row',), lo_implied),
             hi_implied=(('row',), hi_implied),
             status_implied=(('row',), status_implied),
             margin=(('row',), margin),
             empty=(('row',), np.diff(A.indptr) == 0)),
        attrs=dict(preflight=preflight))


def row_intervals(A, lb: np.ndarray, ub: np.ndarray, chunk_nnz: int = 50_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lo, hi and S of every row over the box, float64 — Gurobi reads the float32 coefficients as doubles, so this is
    the model the engine sees — and 0 on an empty row. Every lb is finite, so lo meets −inf only through a negative
    entry on an infinite ub and hi +inf only through a positive one: the two never meet in one sum. A is walked in slabs
    of whole rows of about ``chunk_nnz`` entries, so the per-entry arrays stay bounded."""
    n_rows = A.shape[0]
    indptr = A.indptr.astype(np.int64)
    lo = np.zeros(n_rows)
    hi = np.zeros(n_rows)
    S = np.zeros(n_rows)
    bound_mag = np.maximum(np.abs(lb), np.where(np.isfinite(ub), np.abs(ub), 0.0))
    r0 = 0
    while r0 < n_rows:
        r1 = int(np.searchsorted(indptr, indptr[r0] + chunk_nnz, side='right')) - 1
        r1 = min(max(r1, r0 + 1), n_rows)                                  # a row wider than a slab still advances, alone
        e0, e1 = indptr[r0], indptr[r1]
        a = A.data[e0:e1].astype(np.float64)
        col = A.indices[e0:e1]
        lb_e, ub_e = lb[col], ub[col]
        with np.errstate(invalid='ignore'):
            lo_e = np.where(a > 0, a * lb_e, a * ub_e)
            hi_e = np.where(a > 0, a * ub_e, a * lb_e)
        lo_e[a == 0] = 0.0                                                 # an explicit zero contributes nothing (0 · inf is NaN)
        hi_e[a == 0] = 0.0
        mag_e = np.abs(a) * bound_mag[col]
        nonempty = np.flatnonzero(np.diff(indptr[r0:r1 + 1]) > 0)
        if nonempty.size:
            starts = indptr[r0:r1][nonempty] - e0
            lo[r0 + nonempty] = np.add.reduceat(lo_e, starts)
            hi[r0 + nonempty] = np.add.reduceat(hi_e, starts)
            S[r0 + nonempty] = np.add.reduceat(mag_e, starts)
        r0 = r1
    return lo, hi, S


def row_verdict(sense: np.ndarray, rhs: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """The verdict of every row as a code into ``STATUS`` (m = the row's margin; NaN compares False, so a NaN row straddles):

        sense   redundant                    impossible                     tight                          near_redundant
          <     hi ≤ rhs − m                 lo > rhs + m                   lo ≥ rhs − m                   hi ≤ rhs + m
          >     lo ≥ rhs + m                 hi < rhs − m                   hi ≤ rhs + m                   lo ≥ rhs − m
          =     rhs − m ≤ lo and hi ≤ rhs + m   lo > rhs + m or hi < rhs − m   lo ≥ rhs − m or hi ≤ rhs + m   —

    Where two hold, the earlier column wins (an impossible row is also tight; a redundant one also near-redundant)."""
    le, ge, eq = sense == '<', sense == '>', sense == '='
    redundant = (le & (hi <= rhs - margin)) | (ge & (lo >= rhs + margin)) | (eq & (lo >= rhs - margin) & (hi <= rhs + margin))
    impossible = (le & (lo > rhs + margin)) | (ge & (hi < rhs - margin)) | (eq & ((lo > rhs + margin) | (hi < rhs - margin)))
    tight = (le & (lo >= rhs - margin)) | (ge & (hi <= rhs + margin)) | (eq & ((lo >= rhs - margin) | (hi <= rhs + margin)))
    near_redundant = (le & (hi <= rhs + margin)) | (ge & (lo >= rhs - margin))
    status = np.zeros(rhs.size, dtype=np.int8)
    status[near_redundant] = STATUS.index('near_redundant')
    status[tight] = STATUS.index('tight')
    status[impossible] = STATUS.index('impossible')
    status[redundant] = STATUS.index('redundant')
    return status


def implying_rows(A, sense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows bounds are read off: ``source`` — every entry > 0, sense < or = (a cap: cell usage, source cap, a
    regional cap, a demand ceiling) — and ``unit``, the source rows whose every entry is exactly 1."""
    n_rows = A.shape[0]
    min_a = np.full(n_rows, np.inf)
    max_a = np.full(n_rows, -np.inf)
    nonempty = np.flatnonzero(np.diff(A.indptr) > 0)
    if nonempty.size:
        starts = A.indptr[:-1][nonempty]
        min_a[nonempty] = np.minimum.reduceat(A.data, starts)
        max_a[nonempty] = np.maximum.reduceat(A.data, starts)
    le_or_eq = (sense == '<') | (sense == '=')
    source = le_or_eq & np.isfinite(min_a) & (min_a > 0)
    unit = source & (min_a == 1.0) & (max_a == 1.0)
    return source, unit


def implied_ub(A, rhs: np.ndarray, lb: np.ndarray, ub: np.ndarray, margin: np.ndarray, source: np.ndarray) -> np.ndarray:
    """(2a) Every column's ub tightened by the source rows that hold it: Σ a·x ≤ rhs + m with every other column at
    least at its lb leaves x_j ≤ lb_j + (rhs + m − Σ a·lb) / a_j — the smallest over the rows, one round."""
    rows_of_source = np.flatnonzero(source)
    S = A[rows_of_source].tocsr()
    local = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
    a = S.data.astype(np.float64)
    col = S.indices
    room = rhs[rows_of_source] + margin[rows_of_source] - np.bincount(local, weights=a * lb[col], minlength=S.shape[0])
    tightened = ub.copy()
    np.minimum.at(tightened, col, lb[col] + room[local] / a)
    return tightened


def unit_groups(A, rhs: np.ndarray, lb: np.ndarray, margin: np.ndarray, unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(2b) The groups: a group row's room is rhs + m − Σ lb over the whole row — what its columns can rise above their
    lb together — and every column belongs to the unit row holding it with the least room (−1 where none): any
    assignment bounds validly, the least room bounds tightest."""
    unit_rows = np.flatnonzero(unit)
    U = A[unit_rows].tocsr()
    local = np.repeat(np.arange(U.shape[0]), np.diff(U.indptr))
    room = rhs[unit_rows] + margin[unit_rows] - np.bincount(local, weights=lb[U.indices], minlength=U.shape[0])
    room_of_group = np.full(A.shape[0], np.inf)
    room_of_group[unit_rows] = room
    group_of_col = np.full(A.shape[1], -1, dtype=np.int64)
    if U.nnz:
        order = np.lexsort((unit_rows[local], room[local], U.indices))    # by column, then the least room, then the row
        col_sorted = U.indices[order]
        first = np.ones(order.size, dtype=bool)
        first[1:] = col_sorted[1:] != col_sorted[:-1]
        group_of_col[col_sorted[first]] = unit_rows[local[order[first]]]
    return group_of_col, room_of_group


def grouped_intervals(A, lb: np.ndarray, ub: np.ndarray, group_of_col: np.ndarray, room_of_group: np.ndarray,
                      chunk_nnz: int = 50_000_000) -> tuple[np.ndarray, np.ndarray]:
    """(2b) lo and hi of every row with each group's columns bounded together: over one group a row's entries reach at
    most Σ a·lb + min(Σ a⁺·(ub − lb), room · max a⁺) and at least Σ a·lb + max(Σ a⁻·(ub − lb), room · min a⁻) — a
    fractional-knapsack bound; an ungrouped column counts at its own bound. The columns are permuted so each group is
    contiguous, which makes one group's entries one run inside every sorted CSR row."""
    none = np.iinfo(np.int64).max
    order = np.argsort(np.where(group_of_col >= 0, group_of_col, none), kind='stable')
    group_sorted = group_of_col[order]
    Ap = A[:, order].tocsr()
    Ap.sort_indices()
    lb_p, ub_p = lb[order], ub[order]
    n_rows = Ap.shape[0]
    indptr = Ap.indptr.astype(np.int64)
    lo = np.zeros(n_rows)
    hi = np.zeros(n_rows)
    r0 = 0
    while r0 < n_rows:
        r1 = int(np.searchsorted(indptr, indptr[r0] + chunk_nnz, side='right')) - 1
        r1 = min(max(r1, r0 + 1), n_rows)
        e0, e1 = indptr[r0], indptr[r1]
        if e1 > e0:
            a = Ap.data[e0:e1].astype(np.float64)
            p = Ap.indices[e0:e1]
            row_e = np.repeat(np.arange(r0, r1), np.diff(indptr[r0:r1 + 1]))
            g = group_sorted[p]
            new_run = np.ones(a.size, dtype=bool)
            new_run[1:] = (row_e[1:] != row_e[:-1]) | (g[1:] != g[:-1])
            starts = np.flatnonzero(new_run)
            run_row, run_g = row_e[starts], g[starts]
            width = ub_p[p] - lb_p[p]
            with np.errstate(invalid='ignore'):
                pos_width = np.where(a > 0, a * width, 0.0)                         # a⁺·(ub − lb): +inf on an infinite ub
                neg_width = np.where(a < 0, a * width, 0.0)                         # a⁻·(ub − lb): −inf on an infinite ub
            sum_lb = np.add.reduceat(a * lb_p[p], starts)
            sum_pos = np.add.reduceat(pos_width, starts)
            sum_neg = np.add.reduceat(neg_width, starts)
            max_pos = np.maximum.reduceat(np.maximum(a, 0.0), starts)
            min_neg = np.minimum.reduceat(np.minimum(a, 0.0), starts)
            grouped = run_g >= 0
            room = np.where(grouped, room_of_group[np.where(grouped, run_g, 0)], np.inf)
            with np.errstate(invalid='ignore'):
                hi_run = sum_lb + np.where(grouped, np.minimum(sum_pos, room * max_pos), sum_pos)
                lo_run = sum_lb + np.where(grouped, np.maximum(sum_neg, room * min_neg), sum_neg)
            hi += np.bincount(run_row, weights=hi_run, minlength=n_rows)
            lo += np.bincount(run_row, weights=lo_run, minlength=n_rows)
        r0 = r1
    return lo, hi


# ═══════════════════════════ the response: drop the redundant rows of the listed families ═══════════════════════════

def drop_redundant_rows(rows: xr.Dataset, bounds: xr.Dataset, families) -> np.ndarray:
    """The rows REDUNDANT OVER THE BOX in the listed families, flagged off the row table before the model is built
    (``active`` off, ``redundant`` on), so they never reach the solver — exact: a row every point of the box satisfies
    stays satisfied whatever else is in the model. A row redundant only under implied bounds rests on other rows and
    is reported, not dropped; impossible, tight and near-redundant rows are never dropped. Returns the dropped rows."""
    redundant = bounds['status'].values == STATUS.index('redundant')
    hit = np.zeros(rows.sizes['row'], dtype=bool)
    for family in families:
        span = row_table.family_rows(rows, family)
        if span is not None:
            hit[span] = redundant[span]
    rows['active'] = (('row',), rows['active'].values & ~hit)
    rows['redundant'] = (('row',), rows['redundant'].values | hit)
    return np.flatnonzero(hit)


# ═══════════════════════════ the report: the log, and one CSV per year ═══════════════════════════

def report_row_bounds(rows: xr.Dataset, bounds: xr.Dataset, target_year: int, out_dir: str) -> None:
    """The log — rows per family × verdict over the box, the impossible and tight rows under implied bounds, the column
    preflight hits, every impossible row with its shortfall — and in ``out_dir``: ``bound_report_<year>.csv`` (every row
    redundant over the box, impossible over the box or under implied bounds, and tight in a family with a unit; raw units,
    the row's keys) and ``bound_preflight_<year>.csv`` (the column counts per block). Arithmetic on the tables only: any
    engine, any checkout that builds them gets the same files."""
    REDUNDANT, IMPOSSIBLE, TIGHT = STATUS.index('redundant'), STATUS.index('impossible'), STATUS.index('tight')
    status = bounds['status'].values
    status_implied = bounds['status_implied'].values
    rhs = rows['rhs'].values
    sense = rows['sense'].values
    lo, hi = bounds['lo'].values, bounds['hi'].values
    lo_implied, hi_implied = bounds['lo_implied'].values, bounds['hi_implied'].values

    # ── how far each row is met at its most favourable point under the implied bounds (best: < 0 = short even there,
    #    impossible) and at the least favourable point of the box (worst: ≥ 0 = met everywhere, redundant) ──
    best = np.where(sense == '>', hi_implied - rhs, np.where(sense == '<', rhs - lo_implied,
                    -np.maximum(np.maximum(lo_implied - rhs, rhs - hi_implied), 0.0)))
    worst = np.where(sense == '>', lo - rhs, np.where(sense == '<', rhs - hi, -np.maximum(np.abs(lo - rhs), np.abs(hi - rhs))))

    # ── the log ──
    print(f"│   ├── Bound propagation (rel tol {settings.BOUND_PROP_REL_TOL:g}): {status.size:,} rows; the verdict over the box, "
          f"then impossible / tight under the bounds the rows imply")
    print(f"│   │   {'family':<30s} {'rows':>10s} " + ' '.join(f'{name:>14s}' for name in STATUS) + f" {'dropped':>9s} {'imp·implied':>12s} {'tight·implied':>14s}")
    for family, (start, stop) in rows.attrs['family_range'].items():
        counts = np.bincount(status[start:stop], minlength=len(STATUS))
        n_dropped = int(rows['redundant'].values[start:stop].sum())
        n_imp = int((status_implied[start:stop] == IMPOSSIBLE).sum())
        n_tight = int((status_implied[start:stop] == TIGHT).sum())
        print(f"│   │   {family:<30s} {stop - start:>10,} " + ' '.join(f'{count:>14,}' for count in counts) + f" {n_dropped:>9,} {n_imp:>12,} {n_tight:>14,}")
    for block, counts in bounds.attrs['preflight'].items():
        hits = {what: count for what, count in counts.items() if what != 'columns' and count}
        if hits:
            print(f"│   │   columns of {block}: " + ', '.join(f'{count:,} {what}' for what, count in hits.items()))
    impossible = np.flatnonzero((status == IMPOSSIBLE) | (status_implied == IMPOSSIBLE))
    for r in impossible:
        family = row_table.decode(rows, 'family', [r])[0]
        scale = rows['scale'].values[r]
        print(f"│   │   IMPOSSIBLE {rows['name'].values[r]}: short by {-best[r] * scale:,.6g} {UNIT.get(family, 'share')} "
              f"(rhs {rhs[r] * scale:,.6g}; reachable [{lo_implied[r] * scale:,.6g}, {hi_implied[r] * scale:,.6g}] under the implied bounds, "
              f"[{lo[r] * scale:,.6g}, {hi[r] * scale:,.6g}] over the box)")

    # ── bound_report_<year>.csv: every redundant and impossible row, and the tight rows of the families with a physical
    #    unit — a structural row that pins its columns at a bound (a disabled land use's inflow guard, an adoption
    #    limit of 0) is tight by construction: counted above, not listed ──
    policy = np.zeros(status.size, dtype=bool)
    for family, (start, stop) in rows.attrs['family_range'].items():
        policy[start:stop] = family in UNIT
    listed = np.flatnonzero((status == REDUNDANT) | (status == IMPOSSIBLE) | (status_implied == IMPOSSIBLE)
                            | (((status == TIGHT) | (status_implied == TIGHT)) & policy))
    scale = rows['scale'].values[listed]
    family = row_table.decode(rows, 'family', listed)
    report = pd.DataFrame(dict(
        year=target_year,
        family=family,
        name=rows['name'].values[listed],
        status=np.asarray(STATUS, dtype=object)[status[listed]],
        status_implied=np.asarray(STATUS, dtype=object)[status_implied[listed]],
        sense=sense[listed],
        rhs_raw=rhs[listed] * scale,
        lo_raw=lo[listed] * scale,
        hi_raw=hi[listed] * scale,
        lo_implied_raw=lo_implied[listed] * scale,
        hi_implied_raw=hi_implied[listed] * scale,
        best_raw=best[listed] * scale,
        worst_raw=worst[listed] * scale,
        margin_raw=bounds['margin'].values[listed] * scale,
        unit=[UNIT.get(f, 'share') for f in family],
        empty=bounds['empty'].values[listed],
        dropped=rows['redundant'].values[listed]))
    for field in ('region', 'item', 'presence', 'bound', 'state', 'commodity', 'am_idx'):
        report[field] = row_table.decode(rows, field, listed)
    os.makedirs(out_dir, exist_ok=True)
    report.to_csv(f"{out_dir}/bound_report_{target_year}.csv", index=False)

    # ── bound_preflight_<year>.csv: the column counts per block ──
    pd.DataFrame([dict(year=target_year, block=block, **counts) for block, counts in bounds.attrs['preflight'].items()]) \
        .to_csv(f"{out_dir}/bound_preflight_{target_year}.csv", index=False)
    print(f"│   │   └── {len(report):,} row(s) reported -> {out_dir}/bound_report_{target_year}.csv")
