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

"""The STATE of a step: its four tables as ONE ``Tables`` and the operations on it — relax, append, fix, scale — and the
readers of a relaxed row (``shortfall``, ``slack_unit``). Pure functions: every operation takes a ``Tables`` and returns a
new one, the tables it holds never changed in place; the solver is built from it afresh (``solver.build_solver``).
``step.solve_step`` and ``rowgen.solve_rounds`` drive them; ``row_builder`` (what a row is) knows nothing of them."""

from typing import NamedTuple

import numpy as np
import xarray as xr

from scipy import sparse

from luto import settings
from luto.solvers.row_table import ROW_FILL


# ═══════════════════════════ the table operations of a step: relax, append, fix, scale ═══════════════════════════
#
# The tables (row_matrix, rows, cols, obj) are the only state of a step, carried as ONE ``Tables``: every operation here takes
# it and returns a new one (the tables it holds are never changed in place), and the solver is built from it afresh
# (``solver.build_solver``; LutoSolver is stateless). Used by step.solve_step and rowgen.solve_rounds.

class Tables(NamedTuple):
    """The four tables of a step — what ``relax``, ``append_rows`` and ``fix_slack`` take and return, and ``solver.build_solver``
    builds the model from. Immutable: an operation returns a new one (``tables._replace(...)``)."""
    row_matrix: sparse.csr_matrix
    rows: xr.Dataset
    cols: xr.Dataset
    obj: np.ndarray


def relax(tables: Tables, idx) -> Tables:
    """The rows ``idx`` may fall short: each gets its own shortfall column (block ``slack``) — a·x + rhs·s >= rhs on
    the scaled row, s the fraction of the target missed (sign-flipped on a < row), at settings.ELASTIC_PENALTY AUD per
    target missed in full (million AUD in the objective, as every objective coefficient). The column carries s in the
    row's own units, s' = unit · s (``slack_unit``: |rhs|) — coefficient ±1, ub and penalty rescaled to match
    — so the scaled rhs never enters the matrix; a reader divides by the unit (``shortfall``). ``rows['slack_col']``
    names the column. An EQUALITY row (a demand row: production = demand) gets two, a·x + s⁺ − s⁻ = rhs, each at the
    penalty — ``slack_col`` the one that lowers it, ``slack_col_up`` the one that raises it — so a demand that cannot be
    met in either direction shows up in the diagnosis. The structural rows (node balance, source cap, ag-mgt link) are
    never relaxed: a cell cannot hold more than its area. The ONE mechanism for every row that may fall short, built up
    front (settings.ELASTIC_FAMILIES) or generated."""
    row_matrix, rows, cols, obj = tables
    idx = np.asarray(idx, dtype=np.int64)
    idx = idx[(rows['slack_col'].values[idx] < 0) & rows['active'].values[idx]]   # not twice, not a dropped row
    if idx.size == 0:
        return tables
    sense = rows['sense'].values[idx]
    n_col, n_rows = cols.sizes['col'], idx.size
    # One unit of s moves the row's bound by the target's size, signed by the row's direction (a·x + m·s >= rhs lowers
    # a floor, a·x - m·s <= rhs raises a ceiling):
    #   GBF row:   m = rhs, s in [0, 1] — the fraction of the target area missed; rhs <= 0 means the land outside LUTO
    #              already meets it, and a slack that could only tighten the row stays 0
    #   other row: m = |rhs|, s in [0, inf) — a GHG limit (rhs = limit - off-land emissions) is negative under net
    #              sequestration, where rhs itself would make s TIGHTEN the row, and a ceiling can be overshot by more
    #              than its own size
    #   = row:     the first column as a >= row's (lowers it), a second with the opposite sign (raises it)
    rhs = rows['rhs'].values[idx]
    family = rows['family'].values[idx].astype(str)
    gbf = np.char.startswith(family, 'GBF')
    if ((rhs == 0) & ~gbf).any():
        print(f"│   elastic: {int(((rhs == 0) & ~gbf).sum()):,} non-GBF row(s) with a target of 0 cannot be relaxed", flush=True)
    # The column carries s' = unit · s (unit = |m|, slack_unit): m·s = sign(m)·s', so the coefficient is ±1 (0
    # where m = 0: the row cannot be relaxed), the ub × unit and the penalty / unit — the same LP, m kept out of the matrix
    unit = slack_unit(rhs)
    coef = (np.where(sense == '<', -1.0, 1.0) * np.where(gbf, np.sign(rhs), (rhs != 0).astype(np.float64))).astype(row_matrix.dtype)  # row_matrix's dtype (float32): hstack would upcast the whole matrix to float64
    eq = np.flatnonzero(sense == '=')                                        # the equality rows' second column, raising the row
    n_slack = n_rows + eq.size                                              # one column per row, two per equality row
    col_of = np.concatenate([idx, idx[eq]])
    entry = np.concatenate([coef, -coef[eq]])
    row_matrix = sparse.hstack([row_matrix, sparse.csr_matrix((entry, (col_of, np.arange(n_slack))), shape=(row_matrix.shape[0], n_slack))], format='csr', dtype=row_matrix.dtype)
    slack = xr.Dataset({field: (('col',), np.full(n_slack, 'slack' if field == 'block' else 0.0 if cols[field].dtype.kind == 'f' else -1,
                                                  dtype=cols[field].dtype)) for field in cols.data_vars})
    slack['ub'] = (('col',), np.concatenate([np.where(gbf, unit, np.inf), np.full(eq.size, np.inf)]).astype(cols['ub'].dtype))   # GBF: at most the whole target (s = 1)
    cols = xr.concat([cols, slack], 'col')
    penalty = settings.ELASTIC_PENALTY / 1e6 * (-1 if settings.OBJECTIVE == 'maxprofit' else 1)   # AUD -> million AUD, a cost per target missed
    obj = np.concatenate([obj, (penalty / unit).astype(obj.dtype), (penalty / unit[eq]).astype(obj.dtype)])
    slack_col = rows['slack_col'].values.copy()
    slack_col[idx] = n_col + np.arange(n_rows)
    slack_col_up = rows['slack_col_up'].values.copy()
    slack_col_up[idx[eq]] = n_col + n_rows + np.arange(eq.size)
    rows = rows.assign(slack_col=(('row',), slack_col), slack_col_up=(('row',), slack_col_up))
    print(f"│   elastic: {n_rows:,} row(s) of {sorted(set(family.tolist()))} get a shortfall column "
          f"at {settings.ELASTIC_PENALTY:,.0f} AUD per target missed" + (f" ({eq.size:,} equality rows: one each way)" if eq.size else ""), flush=True)
    return Tables(row_matrix, rows, cols, obj)


def append_rows(tables: Tables, block: sparse.csr_matrix, table: xr.Dataset) -> Tables:
    """Rows generated during the step (``rowgen``): the block under row_matrix — widened to its columns, a shortfall column added
    since the block was laid being 0 in it — and the table under the row table (row i of row_matrix is row i of the table)."""
    row_matrix, rows = tables.row_matrix, tables.rows
    if block.shape[1] < row_matrix.shape[1]:
        block = sparse.hstack([block, sparse.csr_matrix((block.shape[0], row_matrix.shape[1] - block.shape[1]), dtype=block.dtype)], format='csr')
    return tables._replace(row_matrix=sparse.vstack([row_matrix, block], format='csr'), rows=xr.concat([rows, table], dim='row', fill_value=ROW_FILL))


def fix_slack(tables: Tables) -> Tables:
    """Every shortfall column fixed at 0: every relaxed row still active is hard from here on (pass 2)."""
    cols = tables.cols
    ub = cols['ub'].values.copy()
    ub[cols['block'].values == 'slack'] = 0.0
    return tables._replace(cols=cols.assign(ub=(('col',), ub)))


def scale_cost(obj: np.ndarray, cols: xr.Dataset, scale: float) -> np.ndarray:
    """The objective with its economic part × ``scale``, the shortfall columns' penalties untouched
    (settings.ELASTIC_PASS1_COST_SCALE: pass 1 feasibility-first); ``scale`` = 1 is ``obj`` itself."""
    if scale == 1:
        return obj
    out = obj.astype(np.float64)
    out[cols['block'].values != 'slack'] *= scale
    return out


def elastic_absorbs(rows: xr.Dataset, bounds: xr.Dataset) -> np.ndarray:
    """Per row: True where the row's shortfall column (settings.ELASTIC_FAMILIES, ``relax``) can make it hold, so an IMPOSSIBLE verdict on it need not
    stop the year. A non-GBF row's slack is unbounded (m = |rhs|), so it absorbs anything unless rhs == 0; a GBF row's
    slack stops at s = 1, where the floor has fallen to 0 — it absorbs only if rhs > 0 and the row can still reach 0
    (hi_implied >= 0, on the scaled row as the bounds are)."""
    on = np.isin(rows['family'].values, settings.ELASTIC_FAMILIES) & rows['active'].values
    rhs = rows['rhs'].values
    gbf = np.char.startswith(rows['family'].values.astype(str), 'GBF')
    return on & np.where(gbf, (rhs > 0) & (bounds['hi_implied'].values >= 0), rhs != 0)


# ═══════════════════════════ the readers of a relaxed row ═══════════════════════════

def shortfall(rows: xr.Dataset, x: np.ndarray, idx=None) -> np.ndarray:
    """The fraction of its target each row ``idx`` (default: every row) misses at ``x``: its shortfall column(s) over
    the unit (``slack_unit``) — NaN where the row has none (a hard row), the sum of both where an equality row was
    relaxed both ways (``relax``). Every reader of a shortfall value comes through here."""
    idx = np.arange(rows.sizes['row']) if idx is None else np.asarray(idx)
    lo, up = rows['slack_col'].values[idx], rows['slack_col_up'].values[idx]
    out = np.full(idx.size, np.nan)
    on = lo >= 0
    out[on] = x[lo[on]] / slack_unit(rows['rhs'].values[idx][on])
    both = on & (up >= 0)
    out[both] += x[up[both]] / slack_unit(rows['rhs'].values[idx][both])
    return out


def slack_unit(rhs) -> np.ndarray:
    """The unit of a shortfall column (``relax``): |rhs| of its scaled row, 1 where
    rhs = 0. The column carries the size of the target missed, s' in [0, |rhs|] on a GBF row, at coefficient ±1 and the
    penalty / unit — the same LP as a·x + rhs·s >= rhs with s the fraction missed, s = s' / unit — so the scaled rhs (up to
    ~1e5) never sits in the matrix beside the row's own entries (down to SOLVER_COEFF_MIN): a ~1e9 range within one row,
    the suspect in pass 1's barrier "numerical trouble" (GBF8 35–55 %, 2026-10-02). Every reader of a shortfall value
    divides by it."""
    rhs = np.asarray(rhs, dtype=np.float64)
    return np.where(rhs != 0, np.abs(rhs), 1.0)
