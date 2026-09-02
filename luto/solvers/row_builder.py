"""Array-based constraint-row composition for the solver's policy families.

The solver's constraint methods (GBF2/3/4/8, water, GHG, demand) orchestrate; this
module owns the pure steps:

    extract_groups(luto_solver, input_data)    -> groups     bio families, once per build
    extract_structure(luto_solver, input_data) -> structure  policy families, once per build
    compose_row(groups, val_row, nvars, coeff_min) -> CSR    one constraint row

A constraint family is the factored product of a WEIGHTING layer over cells and a
per-cell CONTRIBUTION operator (never materialized as one matrix):

    A_family = V @ C
        V : one weighting row per constraint — a species/vegetation suitability
            layer, a region indicator, a mask area, or all-ones for a global row.
        C : the per-cell operator. NOT one merged matrix — a list of term GROUPS,
            one per (stream, land use/am/non-ag, lm) coefficient source, because
            the ``_qsum`` drop test runs on the PRE-FOLD product ``val * c`` in each
            stream's own dtype (numpy scalar / float32 array / python float).
            Merging streams would change which tiny terms are dropped.

compose_row applies the solver's coefficient pipeline (the four-stage floor contract,
matching what per-term LinExpr construction + ``_floor_assembled_matrix`` produce):

    1. q = val[cells] * c      in the stream's own dtype; drop |q| < SOLVER_COEFF_MIN
    2. contrib = float64(q) * w                     (fold weights — gurobipy doubles)
    3. sum_duplicates()                             (== LinExpr term-merging at add time)
    4. floor merged |coeff| < SOLVER_COEFF_MIN      (== _floor_assembled_matrix)

Column contract: a group's ``var`` values are Gurobi ``Var.index`` on the already-built
model (call ``model.update()`` before extracting); composed rows go to ``addMConstr``
against ``model.getVars()`` (whose order == Var.index).

DTYPE POLICY — float32 for data, float64 at the Gurobi boundary:

* **float32 (the data layers)**: the weighting layers (species/suitability matrices,
  region indicators, mask areas) and the per-cell coefficient streams are float32,
  and stage 1 computes ``val * c`` IN float32 deliberately — that is the dtype the
  per-term ``_qsum`` call sites used, so the SOLVER_COEFF_MIN drop test must see the
  same rounded values. This is also where the memory lives (a dense species matrix
  is GBs *because* it is float32).
* **float64 (everything Gurobi touches)**: Gurobi is a double-precision engine —
  every coefficient, RHS, bound and solution value it holds is a C double. Hence
  stage 2 casts to float64 before the fold weights (LinExpr coefficients are
  doubles), composed blocks and RHS vectors are float64 (they must equal what
  ``getA()``/``getAttr('RHS')`` return, bit for bit), and post-solve
  ``getAttr('X')`` values stay float64 (downcasting would truncate the solution
  to ~7 digits before the reporting mat-vecs, changing reported values for a
  trivial memory saving on thin boundary arrays).

Do not "optimize" a float64 at the Gurobi boundary to float32 — it changes the
numbers the model sees or reports; do not widen a float32 data stream to float64 —
it changes which tiny terms the stage-1 drop test keeps.

The bio families share ``extract_groups`` because they all score the SAME biodiversity
contribution streams — the groups are walked once per formulate (the solver's
``_bio_groups`` cache) and reused by GBF2/3/4/8. Water/GHG/demand have their own
coefficient streams, so they attach per-family ``c`` inline in their solver methods to
the shared ``extract_structure`` walk (the ``_policy_structure`` cache).
"""
import numpy as np
import gurobipy as gp
from scipy import sparse


def _extract_entries(obj_row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One object-array row (len ncells of Var | LinExpr | 0) -> term triples.

    Returns (cells, var_idx, w): a plain Var is one term with w = 1.0; a folded-cell
    LinExpr contributes one term per (var, weight) inside it. X_acct LinExprs carry no
    constant term (asserted — a constant would need an RHS shift we don't implement).
    """
    cells, var_idx, w = [], [], []
    for r, v in enumerate(obj_row):
        if isinstance(v, gp.Var):
            cells.append(r); var_idx.append(v.index); w.append(1.0)
        elif isinstance(v, gp.LinExpr):
            assert v.getConstant() == 0.0, f'nonzero constant in X_acct at cell {r}'
            for i in range(v.size()):
                cells.append(r); var_idx.append(v.getVar(i).index); w.append(v.getCoeff(i))
        # plain 0 (infeasible cell): no term
    return (np.asarray(cells, dtype=np.int32),
            np.asarray(var_idx, dtype=np.int32),
            np.asarray(w, dtype=np.float64))


def extract_groups(luto_solver, input_data) -> list[dict]:
    """Walk the solver's object arrays ONCE -> term groups (the factored C operator).

    Mirrors _build_biodiv_contr_expr's three loops, one group per _qsum call site,
    keeping each stream's ``c`` in its own dtype (bit-identity requirement).
    Requires ``luto_solver.gurobi_model.update()`` first — Var.index is -1 until then.
    """
    groups = []

    # -- agricultural accounting stream: scalar c per land use, dry + irr ----------
    for j in range(input_data.n_ag_lus):
        c = input_data.biodiv_contr_ag_j[j]                 # 0-d numpy scalar
        if c == 0:
            continue
        for lm, arr in (('dry', luto_solver.X_acct_dry_jr), ('irr', luto_solver.X_acct_irr_jr)):
            cells, var_idx, w = _extract_entries(arr[j])
            groups.append(dict(kind='ag', label=f'ag_j{j}_{lm}',
                               cells=cells, var=var_idx, w=w, c=c))

    # -- ag-management stream: per-cell c array, dry + irr -------------------------
    for am, am_j_list in input_data.am2j.items():
        for j_idx in range(len(am_j_list)):
            c_arr = input_data.biodiv_contr_ag_man[am][j_idx]
            if not np.any(c_arr):
                continue
            for lm, arr in (('dry', luto_solver.X_ag_man_dry_vars_jr[am]),
                            ('irr', luto_solver.X_ag_man_irr_vars_jr[am])):
                cells, var_idx, w = _extract_entries(arr[j_idx])
                groups.append(dict(kind='am', label=f'am_{am}_{j_idx}_{lm}'.replace(' ', '_'),
                                   cells=cells, var=var_idx, w=w,
                                   c=np.asarray(c_arr)[cells]))     # per-TERM c values

    # -- non-agricultural stream: python-float c per land use ----------------------
    for k in range(input_data.n_non_ag_lus):
        c = input_data.biodiv_contr_non_ag_k[k]             # plain python float
        if c == 0:
            continue
        cells, var_idx, w = _extract_entries(luto_solver.X_non_ag_vars_kr[k])
        groups.append(dict(kind='nonag', label=f'nonag_k{k}',
                           cells=cells, var=var_idx, w=w, c=float(c)))
    return groups


# ---------------------------------------------------------------------------------
# The family-independent STRUCTURE walk for the policy families (water/GHG/demand):
# they attach their own coefficient streams inline in their solver methods. Structure
# entries carry NO zero-c skipping — every (stream, j/am/k, lm) that owns variables
# yields one entry, because a land use with zero biodiversity contribution still uses
# water / emits / produces.
# ---------------------------------------------------------------------------------
def extract_structure(luto_solver, input_data) -> list[dict]:
    """Walk the object arrays ONCE -> family-independent term structure.

    One entry per (stream, j/am/k, lm): {kind, indices..., cells, var, w}. The per-cell
    identity of each entry's cells is the compacted solver cell axis; policy families
    gather their own coefficient arrays at those cells (values per cell are independent
    of which other cells share the slice, so per-term c equals the legacy slice bitwise).
    Requires ``luto_solver.gurobi_model.update()`` first."""
    out = []
    for j in range(input_data.n_ag_lus):
        for m, arr in enumerate((luto_solver.X_acct_dry_jr, luto_solver.X_acct_irr_jr)):
            cells, var_idx, w = _extract_entries(arr[j])
            out.append(dict(kind='ag', j=j, m=m, cells=cells, var=var_idx, w=w))
    for am, am_j_list in input_data.am2j.items():
        for j_idx, j in enumerate(am_j_list):
            for m, arr in enumerate((luto_solver.X_ag_man_dry_vars_jr[am],
                                     luto_solver.X_ag_man_irr_vars_jr[am])):
                cells, var_idx, w = _extract_entries(arr[j_idx])
                out.append(dict(kind='am', am=am, j_idx=j_idx, j=j, m=m,
                                cells=cells, var=var_idx, w=w))
    for k in range(input_data.n_non_ag_lus):
        cells, var_idx, w = _extract_entries(luto_solver.X_non_ag_vars_kr[k])
        out.append(dict(kind='nonag', k=k, cells=cells, var=var_idx, w=w))
    return out


def extract_raw_ag_structure(luto_solver, input_data) -> list[dict]:
    """Raw (non-accounting) ag decision-var structure: one entry per (j, lm) over
    ``X_ag_dry_vars_jr`` / ``X_ag_irr_vars_jr``.

    For families that bind the FLOW variables directly — regional adoption caps —
    rather than the fold-corrected accounting view (``X_acct_*``, which
    ``extract_structure`` walks). Entries here are always plain Vars (w == 1.0):
    the fold re-expression never applies to the raw registries.
    Requires ``luto_solver.gurobi_model.update()`` first."""
    out = []
    for j in range(input_data.n_ag_lus):
        for m, arr in enumerate((luto_solver.X_ag_dry_vars_jr, luto_solver.X_ag_irr_vars_jr)):
            cells, var_idx, w = _extract_entries(arr[j])
            out.append(dict(kind='ag_raw', j=j, m=m, cells=cells, var=var_idx, w=w))
    return out


def compose_row(groups, val_row: np.ndarray, nvars: int, coeff_min: float) -> sparse.csr_matrix:
    """One species layer row -> one 1 x nvars constraint row (four-stage pipeline)."""
    idx_parts, val_parts = [], []
    for g in groups:
        q = val_row[g['cells']] * g['c']                    # stage 1a: stream-dtype product
        keep = np.abs(q) >= coeff_min                       # stage 1b: _qsum drop (pre-fold)
        if not keep.any():
            continue
        contrib = q[keep].astype(np.float64) * g['w'][keep] # stage 2: fold weights, double
        idx_parts.append(g['var'][keep]); val_parts.append(contrib)
    if not idx_parts:
        return sparse.csr_matrix((1, nvars))
    var_idx = np.concatenate(idx_parts)
    vals = np.concatenate(val_parts)
    row = sparse.csr_matrix((vals, (np.zeros(len(vals), dtype=np.int32), var_idx)),
                            shape=(1, nvars))
    row.sum_duplicates()                                    # stage 3: EXPLICIT merge —
                                                            # CSR construction alone does NOT
                                                            # merge duplicates (scipy >= 1.13)
    row.data[np.abs(row.data) < coeff_min] = 0.0            # stage 4: assembled floor
    row.eliminate_zeros()
    row.sort_indices()
    return row
