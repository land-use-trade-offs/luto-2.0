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
The Gurobi model of one step: A x T — the row table over the column table, and nothing else.
"""

import numpy as np
import gurobipy as gp
import xarray as xr
import luto.settings as settings

from gurobipy import GRB

from luto import tools


# Set Gurobi environment.
gurenv = gp.Env(logfilename="gurobi.log", empty=True)  # (empty = True)
gurenv.setParam("OutputFlag", settings.VERBOSE)
gurenv.setParam("OptimalityTol", settings.OPTIMALITY_TOLERANCE)
gurenv.setParam("FeasibilityTol", settings.FEASIBILITY_TOLERANCE)
gurenv.setParam("BarConvTol", settings.BARRIER_CONVERGENCE_TOLERANCE)
gurenv.setParam("ScaleFlag", settings.SCALE_FLAG)
gurenv.setParam("Threads", settings.THREADS)
gurenv.start()


class LutoSolver:
    """The Gurobi model of one step, returns the raw x."""

    def __init__(self, cols: xr.Dataset, rows: xr.Dataset):
        self.cols = cols
        self.rows = rows
        self.gurobi_model = gp.Model(f"LUTO {settings.VERSION}", env=gurenv)
        self.x = None                           # ONE MVar over the column table: every column, in Var.index order
        self._vars = None                       # model.getVars() in Var.index order (materialised once, for addMConstr)

    def formulate(self):
        """The model: every row of the column table as a variable, every row of the row table as a constraint, the objective."""
        print("Setting up the model...")
        self._setup_vars()
        self._setup_constraints()
        self._setup_objective()

    def _setup_vars(self):
        """Every column of the space as ONE addMVar over the table — lb / ub per row, Var.index order =
        table order (ag | nonag | am | ag2ag | ag2nonag | nonag2ag | cell_usage) — the names from the fields."""
        print("├── Setting up decision variables...")
        cols = self.cols
        lm_name = np.array(['dry', 'irr'])
        snake_of_slot = np.array([tools.am_name_snake_case(option) for option, lus in cols.attrs['agman2lu'].items() for _ in lus], dtype=object)   # the (option, lu) slots in slot order
        self.x = self.gurobi_model.addMVar(cols.attrs['n_all'], lb=cols['lb'].values, ub=cols['ub'].values, name="X")

        names_of = {                                                                # the name of every column of a block, from its fields
            'ag':         lambda t: [f"X_ag_{lm_name[m]}_{j}_{r}" 
                                     for m, j, r in zip(t['m'], t['j'], t['cell'])],
            'nonag':      lambda t: [f"X_non_ag_{k}_{r}" 
                                     for k, r in zip(t['k'], t['cell'])],
            'am':         lambda t: [f"X_ag_man_{lm_name[m]}_{snake_of_slot[slot]}_{j}_{r}".replace(" ", "_")
                                     for slot, m, j, r in zip(t['slot'], t['m'], t['j'], t['cell'])],
            'ag2ag':      lambda t: [f"F_a2a_{from_m}_{from_j}[{m},{local_r},{j}]"
                                     for from_m, from_j, m, local_r, j in zip(t['from_m'], t['from_j'], t['m'], t['local_r'], t['j'])],
            'ag2nonag':   lambda t: [f"F_a2n_{from_m}_{from_j}[{k},{local_r}]"
                                     for from_m, from_j, k, local_r in zip(t['from_m'], t['from_j'], t['k'], t['local_r'])],
            'nonag2ag':   lambda t: [f"F_n2a_{from_k}[{m},{local_r},{j}]"
                                     for from_k, m, local_r, j in zip(t['from_k'], t['m'], t['local_r'], t['j'])],
            'cell_usage': lambda t: [f"Rgconst_cell_usage_{cell}" 
                                     for cell in t['cell']],
        }
     
        for block, block_rows in cols.attrs['block_range'].items():
            span = slice(*block_rows)
            fields = {field: cols[field].values[span] for field in ('m', 'j', 'k', 'slot', 'from_m', 'from_j', 'from_k', 'local_r', 'cell')}
            self.gurobi_model.setAttr('VarName', self.x[span].tolist(), names_of[block](fields))
            print(f"│   {'└──' if block == 'cell_usage' else '├──'} {block:<10s} {span.stop - span.start:>12,} variables")

    def _setup_constraints(self):
        """Every row of the row table as ONE addMConstr (Constr.index = table order), the rows named, the
        handles kept on the table."""
        print("├── Adding the constraints...")
        model = self.gurobi_model
        model.update()                       # the ONE update before the first row: every variable exists
        self._vars = model.getVars()
        assert len(self._vars) == self.cols.attrs['n_all'], 'the model must hold exactly the columns of the space'
        T = self.rows
        constrs = model.addMConstr(T.attrs['A'], self._vars, np.asarray(T['sense'].values, dtype='<U1'), T['rhs'].values).tolist()
        model.setAttr('ConstrName', constrs, T['name'].values.tolist())
        T['constr'] = (('row',), np.array(constrs, dtype=object))
        for family, (start, stop) in T.attrs['family_range'].items():
            print(f"│   │   {family}: {stop - start:,} row(s)")

    def _setup_objective(self):
        """Objective obj · x: the coefficient of every column as the column table carries it
        (``row_builder.get_obj``: million AUD, dropped and floored — the coefficient contract is done there)."""
        print(f"├── Setting up the objective function to {settings.OBJECTIVE}...")
        obj = self.cols['obj'].values
        sense = {"mincost": GRB.MINIMIZE, "maxprofit": GRB.MAXIMIZE}.get(settings.OBJECTIVE)
        if sense is None:
            raise ValueError(f"Unknown objective: {settings.OBJECTIVE}")
        self.gurobi_model.setObjective(obj @ self.x, sense)
        print(f"│   └── objective: {int((obj != 0).sum()):,} nonzero coefficients over {obj.size:,} variables")

    def remove_constraints_by_name(self, names) -> None:
        """Drop rows: flagged inactive on the row table (it never shrinks, so a dropped row stays
        describable), then removed from the Gurobi model. The infeasibility flow in `simulation.py` drops rows this way."""
        if not names:
            return
        T = self.rows
        hit = np.isin(T['name'].values, np.asarray(sorted(set(names)), dtype=object)) & T['active'].values
        if not hit.any():
            return
        T['active'] = (('row',), T['active'].values & ~hit)
        self.gurobi_model.remove(list(T['constr'].values[hit]))
        self.gurobi_model.update()

    def restore_constraints_by_name(self, names) -> None:
        """Put dropped rows back: added to the Gurobi model again from the row table (same row, rhs,
        sense and name; new handles, appended after the existing rows), flagged active."""
        if not names:
            return
        T = self.rows
        hit = np.isin(T['name'].values, np.asarray(sorted(set(names)), dtype=object)) & ~T['active'].values
        if not hit.any():
            return
        constrs = self.gurobi_model.addMConstr(T.attrs['A'][hit], self._vars, np.asarray(T['sense'].values[hit], dtype='<U1'), T['rhs'].values[hit]).tolist()
        self.gurobi_model.setAttr('ConstrName', constrs, T['name'].values[hit].tolist())
        handles = T['constr'].values.copy()
        handles[hit] = constrs
        T['constr'] = (('row',), handles)
        T['active'] = (('row',), T['active'].values | hit)
        self.gurobi_model.update()

    def solve(self) -> np.ndarray | None:
        """Optimise; the value of every column (float64, in Var.index order = the column table's), or None
        when no solution is available (e.g. an infeasible model)."""
        print("Starting solve...\n")
        self.gurobi_model.optimize()
        if self.gurobi_model.SolCount == 0:
            print(f"No solution available (Status={self.gurobi_model.Status}, SolCount=0).\n", flush=True)
            return None
        print("Completed solve.\n", flush=True)
        return self.x.X
