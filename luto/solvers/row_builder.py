"""Constraint-row composition for the solver's policy families.

The solver's constraint methods (GBF2/3/4/8, water, GHG, demand, renewables) orchestrate;
this module owns the pure steps:

    extract_groups(input_data)                 -> groups     bio term groups (c attached)
    extract_structure(input_data)              -> structure  policy term structure (no c)
    attach_coeffs(structure, ag, am, nonag)    -> groups     structure + one family's c streams
    keep_terms(var, q, w)                      -> (var, value)  drop sub-floor coefficients, apply fold weight
    compose_row(groups, val_row, nvars)        -> CSR          one constraint row
    scale_rows(block, rhs)                     -> (block, rhs, scale)  row rescale of a composed block

A constraint family is the product of a WEIGHTING layer over cells and a per-cell
CONTRIBUTION operator, never materialized as one matrix:

    A_family = V @ C
        V : one weighting row per constraint — a species/vegetation suitability layer,
            a region indicator, a mask area, or all-ones for a global row.
        C : a list of term GROUPS, one per (stream, land use/am/non-ag, lm) coefficient
            source, each {cells, var, w, c}. Streams are kept apart because the drop
            test runs on the per-stream product ``val * c`` before the fold weight.

Coefficient contract — four steps, applied by compose_row to every row and by
input_data.get_obj_block + the solver's _setup_objective to the objective:

    1. DROP    q = val[cells] * c (float32); a term with |q| < SOLVER_COEFF_MIN is dropped,
               tested on the per-cell coefficient BEFORE the fold weight
    2. WEIGHT  contrib = q * w   (fold weight, float32; w = 1 for unfolded variables)
    3. MERGE   sum_duplicates()  — a variable repeated by fold terms gets one coefficient
    4. FLOOR   the merged coefficient is dropped again if |coeff| < SOLVER_COEFF_MIN

``keep_terms`` is steps 1–2; the callers do 3–4.

Columns: a group's ``var`` values are Gurobi ``Var.index``, read from the variable tables
(input_data.ag_var_table / ag_acct_table / nonag_var_table / am_var_table / flow_tables)
plus each block's column offset, so no ``model.update()`` is needed to build a row.
Composed rows go to ``addMConstr`` against ``model.getVars()`` (order == Var.index).

Dtypes: weighting layers, coefficient streams, fold weights and composed blocks are
float32 (gurobipy widens exactly at ``addMConstr``/``setObjective``). RHS vectors,
variable bounds and post-solve ``X`` values stay float64.
"""
import numpy as np
from scipy import sparse

import luto.settings as settings


def ag_acct_terms_by_mj(acct_table: dict, var_offset: int) -> dict:
    """(m, j) -> {cells, var, w} for the ACCOUNTING stream, read from input_data.ag_acct_table.

    Terms per (m, j) are ordered cell ascending, then the identity term before the fold
    terms (fold-map order); `compose_row` sums duplicates in this sequence.
    `var` = global Var.index = ag-block offset + table column.
    """
    rows, cols, w = acct_table['term_row'], acct_table['term_col'], acct_table['term_w']
    n_lus = acct_table['col'].shape[1]
    m_of, j_of, r_of = acct_table['m'][rows], acct_table['j'][rows], acct_table['r'][rows]
    key = m_of.astype(np.int64) * n_lus + j_of
    order = np.lexsort((np.arange(rows.size), r_of, key))          # key, then cell, then term order
    key, r_of, cols, w = key[order], r_of[order], cols[order], w[order]
    out = {}
    if key.size:
        bounds = np.flatnonzero(np.diff(key)) + 1
        starts = np.concatenate([[0], bounds]); stops = np.concatenate([bounds, [key.size]])
        for a, b in zip(starts, stops):
            m, j = divmod(int(key[a]), n_lus)
            out[(m, j)] = dict(cells=r_of[a:b].astype(np.int32),
                               var=(cols[a:b].astype(np.int64) + var_offset).astype(np.int32),
                               w=w[a:b].astype(np.float32))
    return out


def nonag_terms_by_k(var_table: dict, var_offset: int) -> dict:
    """k -> {cells, var, w=1} for the non-ag decision vars, read from input_data.nonag_var_table.
    Table rows for one k are cell-ascending."""
    k_col, r_col = var_table['k'], var_table['r']
    out = {}
    if k_col.size:
        bounds = np.flatnonzero(np.diff(k_col)) + 1
        starts = np.concatenate([[0], bounds]); stops = np.concatenate([bounds, [k_col.size]])
        for a, b in zip(starts, stops):
            out[int(k_col[a])] = dict(cells=r_col[a:b].astype(np.int32),
                                      var=(np.arange(a, b, dtype=np.int64) + var_offset).astype(np.int32),
                                      w=np.ones(b - a, dtype=np.float32))
    return out


def am_terms_by_key(var_table: dict, var_offset: int) -> dict:
    """(am_idx, j_idx, m) -> {cells, var, w=1} for the ag-management decision vars, read from
    input_data.am_var_table. Table rows for one key are cell-ascending."""
    am_c, ji_c, m_c, r_c = var_table['am'], var_table['j_idx'], var_table['m'], var_table['r']
    out = {}
    if am_c.size:
        key = (am_c.astype(np.int64) * (ji_c.max() + 1) + ji_c) * 2 + m_c
        bounds = np.flatnonzero(np.diff(key)) + 1                    # rows are grouped by key already
        starts = np.concatenate([[0], bounds]); stops = np.concatenate([bounds, [key.size]])
        for a, b in zip(starts, stops):
            out[(int(am_c[a]), int(ji_c[a]), int(m_c[a]))] = dict(
                cells=r_c[a:b].astype(np.int32),
                var=(np.arange(a, b, dtype=np.int64) + var_offset).astype(np.int32),
                w=np.ones(b - a, dtype=np.float32))
    return out


_EMPTY = dict(cells=np.array([], dtype=np.int32), var=np.array([], dtype=np.int32),
              w=np.array([], dtype=np.float32))


def extract_groups(input_data) -> list[dict]:
    """The bio term groups (the factored C operator), read from the term dicts on input_data
    once per formulate.

    One group per (stream, land use / am / non-ag, lm) coefficient source; each stream's
    ``c`` keeps its own dtype so the sub-floor drop test (|val * c| < SOLVER_COEFF_MIN, before
    the fold weight) sees the per-cell coefficient as computed. Land uses with zero
    biodiversity contribution yield no group.
    """
    groups = []

    # -- agricultural accounting stream: scalar c per land use, dry + irr ----------
    for j in range(input_data.n_ag_lus):
        c = input_data.biodiv_contr_ag_j[j]                 # 0-d numpy scalar
        if c == 0:
            continue
        for m, lm in ((0, 'dry'), (1, 'irr')):
            t = input_data.ag_acct_terms.get((m, j), _EMPTY)
            groups.append(dict(kind='ag', label=f'ag_j{j}_{lm}',
                               cells=t['cells'], var=t['var'], w=t['w'], c=c))

    # -- ag-management stream: per-cell c array, dry + irr -------------------------
    for am_idx, (am, am_j_list) in enumerate(input_data.am2j.items()):
        for j_idx in range(len(am_j_list)):
            c_arr = input_data.biodiv_contr_ag_man[am][j_idx]
            if not np.any(c_arr):
                continue
            for m, lm in ((0, 'dry'), (1, 'irr')):
                t = input_data.am_terms.get((am_idx, j_idx, m), _EMPTY)
                groups.append(dict(kind='am', label=f'am_{am}_{j_idx}_{lm}'.replace(' ', '_'),
                                   cells=t['cells'], var=t['var'], w=t['w'],
                                   c=np.asarray(c_arr)[t['cells']]))     # per-TERM c values

    # -- non-agricultural stream: python-float c per land use ----------------------
    for k in range(input_data.n_non_ag_lus):
        c = input_data.biodiv_contr_non_ag_k[k]             # plain python float
        if c == 0:
            continue
        t = input_data.nonag_terms.get(k, _EMPTY)
        groups.append(dict(kind='nonag', label=f'nonag_k{k}',
                           cells=t['cells'], var=t['var'], w=t['w'], c=float(c)))
    return groups


def extract_structure(input_data) -> list[dict]:
    """The family-independent term structure for the policy families (water / GHG / demand /
    renewables), read from the term dicts on input_data once per formulate.

    One entry per (stream, j/am/k, lm): {kind, indices..., cells, var, w} — every (stream,
    land use, lm) that owns variables, with no zero-c skipping (a land use with zero
    biodiversity contribution still uses water, emits and produces). ``cells`` are solver
    cell indices; each family gathers its own coefficient array at those cells
    (``attach_coeffs``, or its own loop) to make the groups ``compose_row`` takes.
    """
    out = []
    for j in range(input_data.n_ag_lus):
        for m in (0, 1):
            t = input_data.ag_acct_terms.get((m, j), _EMPTY)
            out.append(dict(kind='ag', j=j, m=m, cells=t['cells'], var=t['var'], w=t['w']))
    for am_idx, (am, am_j_list) in enumerate(input_data.am2j.items()):
        for j_idx, j in enumerate(am_j_list):
            for m in (0, 1):
                t = input_data.am_terms.get((am_idx, j_idx, m), _EMPTY)
                out.append(dict(kind='am', am=am, j_idx=j_idx, j=j, m=m,
                                cells=t['cells'], var=t['var'], w=t['w']))
    for k in range(input_data.n_non_ag_lus):
        t = input_data.nonag_terms.get(k, _EMPTY)
        out.append(dict(kind='nonag', k=k, cells=t['cells'], var=t['var'], w=t['w']))
    return out


def attach_coeffs(structure: list[dict], ag_c_mrj: np.ndarray, am_c_mrj: dict, nonag_c_rk: np.ndarray) -> list[dict]:
    """Attach one family's coefficient streams to the shared structure -> groups for ``compose_row``.

    Per entry, ``c`` is the family's per-cell coefficient at the entry's cells:
    ``ag_c_mrj[m, cells, j]``, ``am_c_mrj[am][m, cells, j_idx]`` or ``nonag_c_rk[cells, k]``.
    """
    groups = []
    for s in structure:
        if s['kind'] == 'ag':
            c = ag_c_mrj[s['m'], s['cells'], s['j']]
        elif s['kind'] == 'am':
            c = am_c_mrj[s['am']][s['m'], s['cells'], s['j_idx']]
        else:
            c = nonag_c_rk[s['cells'], s['k']]
        groups.append(dict(cells=s['cells'], var=s['var'], w=s['w'], c=c))
    return groups


def calc_geomean_scale(lhs_max: float, rhs_max: float) -> float:
    """One scale factor as the geometric mean of LHS and RHS magnitudes, normalised by
    ``settings.RESCALE_FACTOR`` (RF): ``sqrt(lhs_max * rhs_max) / RF``. After dividing both sides
    by it, LHS_max and RHS land symmetrically around RF in log space. Falls back to LHS-only
    (``lhs_max / RF``) when ``rhs_max`` is zero."""
    if lhs_max > 0.0 and rhs_max > 0.0:
        return float(np.sqrt(lhs_max * rhs_max) / settings.RESCALE_FACTOR)
    ref = lhs_max if lhs_max > 0.0 else settings.RESCALE_FACTOR
    return float(ref / settings.RESCALE_FACTOR)


def scale_rows(block: sparse.csr_matrix, rhs: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Row rescaling on the COMPOSED block.

    Per row i: ``scale_i = calc_geomean_scale(max|row_i|, |rhs_i|)``; row_i and rhs_i are divided
    by it, then the SCALED row is floored again (|coeff| < SOLVER_COEFF_MIN -> 0, the last step of
    the coefficient contract, repeated here because dividing can push a coefficient under the
    floor). Row scaling
    is an exact LP transformation (same feasible set, same optimum; only conditioning changes),
    and the scale is read from the coefficients Gurobi actually sees. Returns
    ``(block, rhs, scale)`` — multiplying each row and rhs back by ``scale`` restores the raw
    composed row, which is the gate's invariant (``va2_compare_models.py`` compares restored
    models). The block stays float32; ``rhs`` and ``scale`` are float64.
    """
    block = block.tocsr(copy=True)
    rhs = np.asarray(rhs, dtype=np.float64)
    nnz_row = np.diff(block.indptr)
    row_max = np.zeros(block.shape[0], dtype=np.float64)
    has = nnz_row > 0
    if has.any():
        row_max[has] = np.maximum.reduceat(np.abs(block.data).astype(np.float64), block.indptr[:-1][has])
    scale = np.fromiter((calc_geomean_scale(m, abs(r)) for m, r in zip(row_max, rhs)),
                        dtype=np.float64, count=block.shape[0])
    block.data = (block.data / np.repeat(scale, nnz_row)).astype(np.float32)
    block.data[np.abs(block.data) < settings.SOLVER_COEFF_MIN] = 0.0   # floor the scaled row
    block.eliminate_zeros(); block.sort_indices()
    return block, rhs / scale, scale


def keep_terms(var: np.ndarray, q: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Drop the sub-floor coefficients, then apply the fold weight.

    Every constraint row and the objective follow one four-step coefficient contract:
    (1) a term whose per-cell coefficient |q| < SOLVER_COEFF_MIN is dropped, tested BEFORE
    the fold weight; (2) the survivors are multiplied by their fold weight ``w`` (float32;
    1 for an unfolded variable); (3) terms that repeat a variable (fold terms) are merged
    into one coefficient; (4) the merged coefficient is dropped again if it is under
    SOLVER_COEFF_MIN. This function is steps 1 and 2, in one place, for ``compose_row``
    (every constraint family), ``input_data.get_obj_block`` (the objective) and the
    regional-adoption rows; the caller does steps 3 and 4. Returns (var index, value) of
    the kept terms."""
    keep = np.abs(q) >= settings.SOLVER_COEFF_MIN
    v = q[keep]
    if w is not None:
        v = v * w[keep]
    return var[keep], v


def compose_row(groups, val_row: np.ndarray, nvars: int) -> sparse.csr_matrix:
    """One weighting row over cells (``val_row``) -> one 1 x nvars constraint row.

    Per group the per-cell coefficient is ``val_row[cells] * c`` (the stream's own dtype);
    ``keep_terms`` drops the sub-floor terms and applies the fold weight; the kept terms of
    all groups are merged (a variable repeated by fold terms gets one coefficient) and the
    merged coefficient is floored at SOLVER_COEFF_MIN again — the four-step coefficient
    contract described on ``keep_terms``.
    """
    idx_parts, val_parts = [], []
    for g in groups:
        q = val_row[g['cells']] * g['c']                    # per-cell coefficient, stream dtype
        var, contrib = keep_terms(g['var'], q, g['w'])      # drop sub-floor terms, apply fold weight
        if contrib.size:
            idx_parts.append(var); val_parts.append(contrib)
    if not idx_parts:
        return sparse.csr_matrix((1, nvars), dtype=np.float32)
    var_idx = np.concatenate(idx_parts)
    vals = np.concatenate(val_parts)
    row = sparse.csr_matrix((vals, (np.zeros(len(vals), dtype=np.int32), var_idx)),
                            shape=(1, nvars))
    row.sum_duplicates()                                    # merge repeated variables — CSR construction
                                                            # alone does NOT merge duplicates (scipy >= 1.13)
    row.data[np.abs(row.data) < settings.SOLVER_COEFF_MIN] = 0.0   # floor the merged coefficient
    row.eliminate_zeros()
    row.sort_indices()
    return row
