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
import pandas as pd
import gurobipy as gp
import xarray as xr
import luto.settings as settings

from gurobipy import GRB

from luto import tools
from luto.solvers.row_inputs import RowInputs


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
    """The Gurobi model of one step — the column table, the row table, A and obj, as given — returns the raw x.
    Stateless with respect to the tables: it is built from them and never changes them (the rows' Gurobi handles and
    the duals go onto its own shallow copy of the row table, ``self.rows``); a step that changes its tables — a row
    added, relaxed or dropped — builds a new one (``rowgen.solve_rounds``)."""

    def __init__(self, cols: xr.Dataset, rows: xr.Dataset, A, obj: np.ndarray, inputs: RowInputs):
        self.cols = cols
        self.rows = rows.copy()         # shallow: the solve's outputs (``constr``, ``pi``) land here, not on the step's table
        self.A = A                      # (row x col) scipy CSR: row i of A is row i of the row table
        self.obj = obj                  # (col,) the objective coefficient of every column (million AUD, row_builder.get_obj)
        self.landmans = inputs.landmans         # the land-management names in m order: how a variable's name spells m
        self.options = list(inputs.agman2lu)    # the ag-management options in am_idx order: how a variable's name spells am_idx
        self.gurobi_model = gp.Model(f"LUTO {settings.VERSION}", env=gurenv)
        self.x = None                           # ONE MVar over the column table: every column, in Var.index order
        self._vars = None                       # model.getVars() in Var.index order (materialised once, for addMConstr)
        self.last_x = None                      # the last solve's value of every column (``solve``)

    def formulate(self):
        """The model: every row of the column table as a variable, every row of the row table as a constraint, the objective."""
        print("Setting up the model...")
        self._setup_vars()
        self._setup_constraints()
        self._setup_objective()

    def _setup_vars(self):
        """Every column of the space as ONE addMVar over the table — lb / ub per row, Var.index order =
        table order (ag | nonag | am | ag2ag | ag2nonag | nonag2ag) — the names from the fields."""
        print("├── Setting up decision variables...")
        cols = self.cols
        lm_name = np.array(self.landmans)                                           # the land managements in m order
        snake_of_option = np.array([tools.am_name_snake_case(option) for option in self.options], dtype=object)   # the options in am_idx order
        self.x = self.gurobi_model.addMVar(
            cols.sizes['col'], 
            lb=cols['lb'].values, 
            ub=cols['ub'].values, 
            name="X"
        )

        names_of = {                                                                # the name of every column of a block, from its fields
            'ag':         lambda t: [f"X_ag_{lm_name[m]}_{j}_{r}" 
                                     for m, j, r in zip(t['m'], t['j'], t['cell'])],
            'nonag':      lambda t: [f"X_non_ag_{k}_{r}" 
                                     for k, r in zip(t['k'], t['cell'])],
            'am':         lambda t: [f"X_ag_man_{lm_name[m]}_{snake_of_option[am_idx]}_{j}_{r}".replace(" ", "_")
                                     for am_idx, m, j, r in zip(t['am_idx'], t['m'], t['j'], t['cell'])],
            'ag2ag':      lambda t: [f"F_a2a_{from_m}_{from_j}[{m},{local_r},{j}]"
                                     for from_m, from_j, m, local_r, j in zip(t['from_m'], t['from_j'], t['m'], t['local_r'], t['j'])],
            'ag2nonag':   lambda t: [f"F_a2n_{from_m}_{from_j}[{k},{local_r}]"
                                     for from_m, from_j, k, local_r in zip(t['from_m'], t['from_j'], t['k'], t['local_r'])],
            'nonag2ag':   lambda t: [f"F_n2a_{from_k}[{m},{local_r},{j}]"
                                     for from_k, m, local_r, j in zip(t['from_k'], t['m'], t['local_r'], t['j'])],
            'slack':      lambda t: [f"S_{i}" for i in range(len(t['cell']))],       # a relaxed row's shortfall column (row_builder.relax)
        }
     
        block_of_col = cols['block'].values
        for block in pd.unique(block_of_col):                                       # the blocks, in table order
            on = np.flatnonzero(block_of_col == block)
            fields = {field: cols[field].values[on] for field in ('m', 'j', 'k', 'am_idx', 'from_m', 'from_j', 'from_k', 'local_r', 'cell')}
            self.gurobi_model.setAttr('VarName', self.x[on].tolist(), names_of[block](fields))
        blocks = pd.DataFrame({'block': pd.unique(block_of_col), 'variables': [int((block_of_col == block).sum()) for block in pd.unique(block_of_col)]})
        for line in blocks.to_markdown(index=False, tablefmt='psql', intfmt=',').split('\n'):
            print(f"│   {line}")

    def _setup_constraints(self):
        """Every ACTIVE row of the row table as ONE addMConstr (in table order), the rows named, the handles kept on
        the table — None on a row dropped before the build (a redundant row, ``row_bounds.drop_redundant_rows``)."""
        print("├── Adding the constraints...")
        model = self.gurobi_model
        model.update()                       # the ONE update before the first row: every variable exists
        self._vars = model.getVars()
        assert len(self._vars) == self.cols.sizes['col'], 'the model must hold exactly the columns of the space'
        T = self.rows
        active = T['active'].values
        built = np.flatnonzero(active)
        A = self.A if active.all() else self.A[built]
        constrs = model.addMConstr(A, self._vars, np.asarray(T['sense'].values[built], dtype='<U1'), T['rhs'].values[built]).tolist()
        model.setAttr('ConstrName', constrs, T['name'].values[built].tolist())
        handles = np.full(T.sizes['row'], None, dtype=object)
        handles[built] = constrs
        T['constr'] = (('row',), handles)
        family = T['family'].values
        families = pd.DataFrame([dict(family=name, constraints=int((family == name).sum()) - int(((family == name) & ~active).sum()),
                                      **{'dropped before the build': int(((family == name) & ~active).sum())})
                                 for name in pd.unique(family)])                                   # the families, in table order
        for line in families.to_markdown(index=False, tablefmt='psql', intfmt=',').split('\n'):
            print(f"│   {line}")

    def _setup_objective(self):
        """Objective obj · x: the coefficient of every column as given (``row_builder.get_obj``: million AUD,
        dropped and floored — the coefficient contract is done there)."""
        print(f"├── Setting up the objective function to {settings.OBJECTIVE}...")
        obj = self.obj
        sense = {"mincost": GRB.MINIMIZE, "maxprofit": GRB.MAXIMIZE}.get(settings.OBJECTIVE)
        if sense is None:
            raise ValueError(f"Unknown objective: {settings.OBJECTIVE}")
        self.gurobi_model.setObjective(obj @ self.x, sense)
        print(f"│   └── objective: {int((obj != 0).sum()):,} nonzero coefficients over {obj.size:,} variables")

    def solve_with_retries(self, target_year: int, params=None):
        """Run the ``params`` ladder (default settings.RETRY_PARAMS) against the current model. Returns (accepted, x, status)
        — x the raw solution vector over the column table (None when the solve left no solution).

        ``params`` is a list of (NumericFocus, Method, Crossover, Presolve, BarHomogeneous) tuples tried in order; only
        GRB.OPTIMAL is accepted.
        """
        x, status = None, None
        for nf, method, crossover, presolve, barhomogenous in (settings.RETRY_PARAMS if params is None else params):
            print(f"Trying NumericFocus={nf}, Method={method}, Crossover={crossover}, Presolve={presolve}, BarHomogeneous={barhomogenous} for year {target_year}...", flush=True)
            self.gurobi_model.Params.NumericFocus    = nf
            self.gurobi_model.Params.Method          = method
            self.gurobi_model.Params.Crossover       = crossover
            self.gurobi_model.Params.Presolve        = presolve
            self.gurobi_model.Params.BarHomogeneous  = barhomogenous

            x = self.solve()
            status = self.gurobi_model.Status
            if x is not None and status == GRB.OPTIMAL:
                print(f"Optimal solution found with NumericFocus={nf}, Method={method}", flush=True)
                return True, x, status
            print(f"Non-optimal status {status} with NumericFocus={nf}, Method={method}; retrying with next attempt if available.", flush=True)
        return False, x, status

    def solve(self) -> np.ndarray | None:
        """Optimise; the value of every column (float64, in Var.index order = the column table's), or None
        when no solution is available (e.g. an infeasible model)."""
        print("Starting solve...\n")
        self.gurobi_model.optimize()
        if self.gurobi_model.SolCount == 0:
            print(f"No solution available (Status={self.gurobi_model.Status}, SolCount=0).\n", flush=True)
            return None
        print("Completed solve.\n", flush=True)
        self.last_x = np.asarray(self.x.X, dtype=np.float64)
        return self.last_x
