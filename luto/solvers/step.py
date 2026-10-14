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

"""The solve of one step over its tables: one pass, or pass 1 → the drop → pass 2.

The policy for every row that may fall short — built up front (settings.ELASTIC_FAMILIES, relaxed here first) or
generated during the step (``rowgen``, relaxed as it is added) — lives here, once; a pass is one solve, or the rounds
of row generation (``rowgen.solve_rounds``). Every run goes through ``solve_step``, row generation on or off."""

import numpy as np
import pandas as pd

from luto import settings
from luto.solvers.reports import report_shortfall
from luto.solvers.row_inputs import RowInputs
from luto.solvers.rowgen import RowGen, solve_rounds
from luto.solvers.solver import build_solver
from luto.solvers.tables import Tables, fix_slack, relax, shortfall


def solve_step(tables: Tables, inputs: RowInputs, rowgen: RowGen | None, target_year: int, out_dir: str, on_first_model=None):
    """The solve of one step over its tables:

        one pass    no row may fall short, or settings.ELASTIC_DROP_SHORT is off: one pass at full cost, the relaxed
                    rows short at the penalty's price — the solution as it is
        two passes  settings.ELASTIC_DROP_SHORT: PASS 1 with the cost × settings.ELASTIC_PASS1_COST_SCALE and
                    settings.ELASTIC_PASS1_PARAMS (barrier, no crossover) finds the rows that cannot be met together with
                    the step; THE DROP, once, for every relaxed row: the rows short by more than settings.ELASTIC_DROP_TOL of
                    their target are switched off (``active``; shortfall_<year>.csv, with ``dropped``), every other relaxed
                    row made hard (its shortfall column's ub = 0, ``fix_slack``); PASS 2 at full cost (settings.RETRY_PARAMS),
                    generated rows added hard. Pass 1's point meets every row kept, so pass 2 is never infeasible.

    The tables (``tables.Tables``) are the only state: every operation returns a new one and every solve builds a
    fresh ``LutoSolver`` from it. ``on_first_model`` is called with the first Gurobi model built (the debug MPS). Returns
    (accepted, x, status, solver) — the last solver, built from the final tables (``solver.cols`` / ``rows`` / ``row_matrix`` /
    ``obj``), x over its columns."""
    rows = tables.rows
    tables = relax(tables, np.flatnonzero(np.isin(rows['family'].values, settings.ELASTIC_FAMILIES) & rows['active'].values))
    relaxable = (tables.rows['slack_col'].values >= 0).any() or (rowgen is not None and (rowgen.open & (rowgen.mode == 'may_drop')).any())
    if not (settings.ELASTIC_DROP_SHORT and relaxable):
        accepted, x, status, solver, _ = solve_pass(tables, inputs, rowgen, target_year, 1, settings.RETRY_PARAMS, 1.0, on_first_model)
        return accepted, x, status, solver

    # ── pass 1 ──
    scale = settings.ELASTIC_PASS1_COST_SCALE
    if scale != 1:
        print(f"Year {target_year}: pass 1 feasibility-first — the cost × {scale:g}", flush=True)
    accepted, x, status, solver, tables = solve_pass(tables, inputs, rowgen, target_year, 1, settings.ELASTIC_PASS1_PARAMS, scale, on_first_model)
    if not accepted:
        return accepted, x, status, solver

    # ── the drop: every relaxed row at once ──
    rows = tables.rows
    on = np.flatnonzero((rows['slack_col'].values >= 0) & rows['active'].values)
    short = on[shortfall(rows, x, on) > settings.ELASTIC_DROP_TOL]                         # the fraction of the target missed
    report_shortfall(x, rows, target_year, out_dir, dropped=short)
    if rowgen is not None:
        rowgen.record_pass1(rows, x, short)
    if short.size == 0 and scale == 1 and rowgen is None:                                 # pass 1 missed nothing and ran at full cost: its solution is the hard one
        print(f"Year {target_year}: no relaxed row falls short — pass 1's solution is kept", flush=True)
        return True, x, status, solver
    active = rows['active'].values.copy()
    active[short] = False
    tables = fix_slack(tables._replace(rows=rows.assign(active=(('row',), active))))
    family = rows['family'].values[short].astype(str)
    print(f"Year {target_year}: {short.size:,} relaxed row(s) cannot be met together with the year — switched off"
          + (" (" + ", ".join(f"{fam} {int((family == fam).sum()):,}" for fam in pd.unique(family)) + ")" if short.size else "")
          + f"; the other {on.size - short.size:,} made hard, the full cost restored — pass 2", flush=True)

    # ── pass 2 ──
    accepted, x, status, solver, _ = solve_pass(tables, inputs, rowgen, target_year, 2, settings.RETRY_PARAMS, 1.0)
    return accepted, x, status, solver


def solve_pass(tables: Tables, inputs: RowInputs, rowgen: RowGen | None, target_year: int, npass: int, params, cost_scale: float, on_first_model=None):
    """One pass over the tables: without row generation one solve; with it, the rounds (``rowgen.solve_rounds``).
    Returns (accepted, x, status, solver, tables): the last solver and the tables it was built from."""
    if rowgen is not None:
        return solve_rounds(tables, inputs, rowgen, target_year, npass, params, cost_scale, on_first_model)
    solver = build_solver(tables, inputs, cost_scale, on_first_model)
    accepted, x, status = solver.solve_with_retries(target_year, params)
    return accepted, x, status, solver, tables
