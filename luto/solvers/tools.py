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


"""Why is this year infeasible? Ask Gurobi, by deleting constraint groups until it isn't.

Everything here works on a COPY of a `gurobipy.Model` and answers with the solver rather than with
a hand-derived bound. That matters for two reasons:

  * **Exactness.** A bound computed outside the model has to reproduce the model's own coefficients,
    and every such reproduction is a chance to drift — the accounting stream (`X_acct`), `_qsum`
    flooring, the ag-management fold. Deleting rows from the real model cannot drift.
  * **Extensibility.** A new constraint family needs one line in `CONSTRAINT_GROUPS`. No new maths,
    no new value function, nothing to keep in sync with `solver.py`.

The logic is deletion filtering, the same idea an IIS uses:

    remove group G -> model becomes feasible   =>  G is implicated
    remove group G -> still infeasible         =>  G alone is not the cause

Removing constraints is a RELAXATION, so the inference runs one way: a reduced model that is still
INFEASIBLE proves the full model infeasible. A reduced model that turns FEASIBLE proves only that
the removed group was necessary for the conflict.

Two entry points, one implementation:

    diagnose(luto_solver.gurobi_model)          in-run, no I/O — the model is already in memory
    diagnose(gp.read("debug_model_*.mps"))      post-mortem, days later

`simulation.py` writes that MPS before every solve, so the second form always has something to read.
NOTE the MPS is only useful if constraint names survive it: MPS is whitespace-delimited, and a
single name containing a space makes Gurobi discard EVERY name in the file and emit `c0, c1, ...`.
`check_constraint_names` exists to catch that before it costs you a diagnosis.

`resolve_infeasibility` is the production entry point: `simulation.py` calls it after `formulate()`
and before the retry ladder, so a year that cannot solve is identified in seconds rather than after
hours of barrier iterations ending in "INFEASIBLE OR UNBOUNDED".
"""

import time

import gurobipy as gp
import numpy as np
import pandas as pd

from gurobipy import GRB


# ---------------------------------------------------------------------------- #
# Constraint groups and naming                                                 #
# ---------------------------------------------------------------------------- #

# Constraint-name prefixes per group, taken from the `name=` arguments in solver.py. Grouping by
# name (rather than by a registry of builder methods) is what lets this work on a model read back
# from MPS, where the builders are long gone.
CONSTRAINT_GROUPS = {
    'cell_usage':   ('const_cell_usage_',),
    'ag_mgt_link':  ('const_ag_mam_dry_usage_', 'const_ag_mam_irr_usage_'),
    'ag_mgt_ub':    ('const_', ),                       # const_<am>_solvable_ub_<r>; see _GROUP_EXCLUDE
    'ag_mgt_adopt': ('const_ag_mam_adoption_limit_',),
    'demand':       ('demand_hard_bound_', 'demand_soft_bound_'),
    'ghg':          ('ghg_emissions_limit_',),
    'water':        ('water_yield_limit_',),
    'nonag_cap':    ('reg_adopt_limit_non_ag_sum_',),
    'adopt_ag':     ('reg_adopt_limit_ag_',),
    'adopt_nonag':  ('reg_adopt_limit_non_ag_',),       # excludes the _sum_ rows; see _GROUP_EXCLUDE
    'bio_gbf2':     ('bio_GBF2_',),
    'bio_nvis':     ('bio_GBF3_NVIS_limit_',),
    'bio_snes':     ('bio_GBF4_SNES_limit_',),
    'bio_ecnes':    ('bio_GBF4_ECNES_limit_',),
    'bio_gbf8':     ('bio_GBF8_limit_',),
    'renewable':    ('renewable_',),
    'flow_out':     ('srccap_a_', 'srccap_n_'),
    'flow_in':      ('bal_a_', 'bal_n_'),
}

# Prefixes overlap: 'const_' would swallow every const_* group, and 'reg_adopt_limit_non_ag_'
# would swallow the _sum_ rows. Longest-prefix-wins resolves it, but state the exclusions so the
# intent is readable rather than implied by string lengths.
_GROUP_EXCLUDE = {
    'ag_mgt_ub':   ('const_cell_usage_', 'const_ag_mam_'),
    'adopt_nonag': ('reg_adopt_limit_non_ag_sum_',),
}

# Never delete these when hunting for a cause: without them the model is not a land-use model at
# all. `cell_usage` is the equality that makes per-cell share scarce — remove it and every cell can
# hold one unit of every land use at once, so almost anything becomes "feasible" and the answer is
# meaningless. The ag-management links (X_ag_man <= X_ag[j]) are structural in the same way.
STRUCTURAL = ('cell_usage', 'ag_mgt_link')

STATUS = {
    GRB.OPTIMAL: 'OPTIMAL', GRB.INFEASIBLE: 'INFEASIBLE', GRB.INF_OR_UNBD: 'INF_OR_UNBD',
    GRB.UNBOUNDED: 'UNBOUNDED', GRB.TIME_LIMIT: 'TIME_LIMIT', GRB.SUBOPTIMAL: 'SUBOPTIMAL',
}


def group_of(name: str) -> str | None:
    """Which group a constraint belongs to, longest matching prefix wins."""
    best, best_len = None, -1
    for group, prefixes in CONSTRAINT_GROUPS.items():
        if any(name.startswith(x) for x in _GROUP_EXCLUDE.get(group, ())):
            continue
        for p in prefixes:
            if name.startswith(p) and len(p) > best_len:
                best, best_len = group, len(p)
    return best


def group_counts(model) -> pd.Series:
    """How many rows each group holds — the first thing to check on a model read from disk."""
    counts = pd.Series([group_of(c.ConstrName) for c in model.getConstrs()]).value_counts(dropna=False)
    return counts.rename(index={np.nan: '<ungrouped>'})


def check_constraint_names(model, limit: int = 10) -> list[str]:
    """Names that would be destroyed by an MPS round-trip.

    MPS is whitespace-delimited. Gurobi's response to ONE name containing a space is to discard
    every name in the file and write `c0, c1, c2 ...`, which silently turns the saved model into an
    artefact you cannot attribute anything to. Worth asserting rather than discovering later.
    """
    bad = [c.ConstrName for c in model.getConstrs() if any(ch.isspace() for ch in c.ConstrName)]
    if bad:
        print(f"WARNING: {len(bad):,} constraint name(s) contain whitespace — an MPS round-trip "
              f"will discard ALL names. First {min(limit, len(bad))}:")
        for n in bad[:limit]:
            print(f"    {n!r}")
    return bad


# ---------------------------------------------------------------------------- #
# Feasibility probes (always on copies — the caller's model is never touched)  #
# ---------------------------------------------------------------------------- #

def _feasibility_copy(model, time_limit: float | None = None, keep_groups=None,
                      verbose: bool = False):
    """A copy set up to answer feasibility only: zero objective, no dual reductions, quiet.

    `keep_groups` restricts the copy to those groups plus STRUCTURAL, discarding the rest. That is
    a RELAXATION, so the inference is one-directional and worth stating plainly:

        restricted model INFEASIBLE  =>  the FULL model is infeasible too, and the conflict lies
                                         entirely within the groups kept.
        restricted model FEASIBLE    =>  says NOTHING about the full model. A conflict involving a
                                         discarded group is invisible.

    It exists because it is dramatically cheaper — dropping demand, GHG, water, renewables and the
    transition-flow system roughly halves the model and took one measured feasibility solve from
    417 s to 13 s. Worth it when you already know which constraints bind; misleading if you do not.
    """
    m = model.copy()
    if keep_groups:
        keep = set(keep_groups) | set(STRUCTURAL)
        discard = [c for c in m.getConstrs() if group_of(c.ConstrName) not in keep]
        if discard:
            m.remove(discard)
            m.update()
            if verbose:
                print(f"│   │   ├── restricted to {sorted(keep)}: dropped {len(discard):,} row(s), "
                      f"{m.NumConstrs:,} remain")
    m.setObjective(0, GRB.MINIMIZE)     # feasibility, not optimality — far cheaper
    m.Params.OutputFlag = 0
    m.Params.DualReductions = 0         # so INF_OR_UNBD cannot hide a definite INFEASIBLE
    if time_limit:
        m.Params.TimeLimit = time_limit
    return m


def _remove_named(model, names) -> None:
    """Remove constraints from `model` by name. No-op on an empty list."""
    if not names:
        return
    doomed = set(names)
    model.remove([c for c in model.getConstrs() if c.ConstrName in doomed])
    model.update()


def _probe(model, dropped, time_limit: float | None = None, keep_groups=None,
           verbose: bool = False):
    """Feasibility copy with the already-dropped rows removed — one diagnosis round's model."""
    m = _feasibility_copy(model, time_limit, keep_groups, verbose)
    _remove_named(m, dropped)
    return m


def _iis_rows(model) -> list[str]:
    """Constraint names in the IIS. Call after `model.computeIIS()`."""
    return [c.ConstrName for c in model.getConstrs() if c.IISConstr]


def _by_group(names) -> dict[str, list[str]]:
    """Bucket constraint names by family; unknown prefixes land in '<ungrouped>'."""
    out = {}
    for n in names:
        out.setdefault(group_of(n) or '<ungrouped>', []).append(n)
    return out


def _droppable_rows(iis_names, droppable) -> list[str]:
    """IIS rows that policy allows sacrificing, least-valued group first.

    `droppable` is the caller's ordered list of groups, least-valued first; rows outside those
    groups are not candidates at all. The sort is stable, so within a group the IIS order is kept —
    the choice there is arbitrary anyway (a principled tie-break would need a per-row cost the
    model does not carry).
    """
    order = {g: i for i, g in enumerate(droppable)}
    rows = [n for n in iis_names if group_of(n) in order]
    rows.sort(key=lambda n: order[group_of(n)])
    return rows


def is_feasible(model, drop_groups=(), time_limit: float | None = None, keep_groups=None):
    """Solve a copy with `drop_groups` removed. Returns (status_name, seconds).

    The caller's model is never touched: every call works on `model.copy()`.
    """
    m = _feasibility_copy(model, time_limit, keep_groups)
    if drop_groups:
        m.remove([c for c in m.getConstrs() if group_of(c.ConstrName) in set(drop_groups)])
        m.update()
    t = time.time()
    m.optimize()
    return STATUS.get(m.Status, str(m.Status)), time.time() - t


# ---------------------------------------------------------------------------- #
# Production entry points (called by simulation.py around every year's solve)  #
# ---------------------------------------------------------------------------- #

def feasible_solve(model, groups=None, droppable=None, max_rounds: int = 20,
                   time_limit: float | None = None, verbose: bool = True):
    """Test each constraint group ALONE, before the real solve, and drop what cannot hold.

    Catches a different failure class from `resolve_infeasibility`, and catches it cheaply:

        this test        a group that is infeasible BY ITSELF — e.g. one ECNES community whose
                         target exceeds anything its cells could reach, which then takes the whole
                         year down with it.
        post-failure IIS groups that are each fine but conflict TOGETHER — e.g. three SNES species
                         that only become impossible once a regional land budget is imposed.

    No single-group test can see the second kind, and the IIS on the union is far more expensive
    than these, so the two are complements rather than alternatives.

    Each group is solved with STRUCTURAL kept and everything else discarded — the smallest model in
    which that group's rows still mean something. `cell_usage` must stay or land is not scarce;
    `ag_mgt_link` must stay or an ag-management bonus can be claimed without the land use under it,
    which would make rows look satisfiable when they are not.

    Returns (dropped_names, DataFrame) — the frame is the record to write to disk.
    """
    groups = list(groups or CONSTRAINT_GROUPS)
    droppable = list(droppable or [])
    present = set(group_counts(model).index)
    test_groups = [g for g in groups if g not in STRUCTURAL and g in present]
    dropped, records = [], []

    # Fast path: one solve with ALL tested groups kept together. Each single-group probe is a
    # RELAXATION of this joint model (removing the other groups can only relax it), so joint
    # OPTIMAL proves every group is feasible alone — the per-group sweep would find nothing.
    # One ~13 s restricted solve instead of one per group, on every healthy year. Any other joint
    # status (INFEASIBLE, or an unclassifiable one) falls through to per-group attribution.
    if len(test_groups) > 1:
        joint = _probe(model, dropped, time_limit, keep_groups=test_groups)
        joint.optimize()
        if joint.Status == GRB.OPTIMAL:
            if verbose:
                print(f"│   │   ├── all groups feasible together — per-group test skipped")
            return dropped, pd.DataFrame(records)
        if verbose:
            print(f"│   │   ├── joint probe {STATUS.get(joint.Status, joint.Status)} — "
                  f"testing each group alone")

    for group in test_groups:
        for rnd in range(1, max_rounds + 1):
            probe = _probe(model, dropped, time_limit, keep_groups=[group])
            probe.optimize()

            if probe.Status == GRB.OPTIMAL:
                if verbose and rnd == 1:
                    print(f"│   │   ├── {group:12s} feasible alone")
                break
            if probe.Status != GRB.INFEASIBLE:
                if verbose:
                    print(f"│   │   ├── {group:12s} {STATUS.get(probe.Status, probe.Status)} — skipped")
                break

            probe.computeIIS()
            iis = _iis_rows(probe)
            candidates = _droppable_rows(iis, droppable)
            if not candidates:
                if verbose:
                    print(f"│   │   ├── {group:12s} INFEASIBLE alone, but nothing droppable in it")
                records.append({'group': group, 'round': rnd, 'constraint': None,
                                'iis_size': len(iis), 'action': 'UNRESOLVABLE'})
                break

            victim = candidates[0]
            dropped.append(victim)
            records.append({'group': group, 'round': rnd, 'constraint': victim,
                            'iis_size': len(iis), 'action': 'DROPPED'})
            if verbose:
                print(f"│   │   ├── {group:12s} INFEASIBLE alone -> dropping {victim}")

    return dropped, pd.DataFrame(records)


def resolve_infeasibility(model, droppable=None, max_rounds: int = 20,
                          time_limit: float | None = None, verbose: bool = True,
                          keep_groups=None) -> dict:
    """Find the rows that make this year infeasible, and which to drop so it solves.

    `droppable` is the ordered list of constraint groups that may be sacrificed, least-valued first.
    It is a POLICY supplied by the caller, not a fact this module can decide: an IIS names a SET of
    rows that cannot all hold, and something has to choose which member to give up. Defaults to
    None — detect and report, drop nothing — so a caller must opt in deliberately. `simulation.py`
    passes `settings.DROP_UNREACHABLE_CONSTRAINTS`.

    Strictly stronger than a closed-form "is this row reachable on its own?" screen. Such a screen
    can only catch rows that are impossible ALONE: the three Goulburn Broken SNES species that broke
    the capped runs each pass one comfortably (target/attainable 0.72-0.84) and are impossible only
    in combination with the non-ag cap. An IIS sees that; a per-row bound cannot.

    Iterates because an IIS is *an* irreducible set, not the only one — two independent conflicts
    need two rounds. Detection runs entirely on copies; the caller applies the returned names to the
    real model.

    Returns {'dropped': [names], 'status': ..., 'rounds': [...]}.
    """
    dropped, rounds = [], []
    droppable = list(droppable or [])

    for rnd in range(1, max_rounds + 1):
        probe = _probe(model, dropped, time_limit, keep_groups, verbose=(rnd == 1 and verbose))

        # Straight to the IIS — no separate feasibility solve. The caller only reaches here because
        # the real solve already failed, so paying for a second optimize() to re-learn that would be
        # wasted. computeIIS runs its own analysis.
        #
        # It DOES raise if the model turns out to be feasible, and that is a real possibility rather
        # than a defensive nicety: this copy is restricted to `keep_groups`, so a conflict involving
        # an excluded group is absent from it, and a failure that was numerical rather than
        # infeasible leaves nothing to find either. Both cases mean "nothing here to drop".
        try:
            probe.computeIIS()
        except gp.GurobiError as exc:
            if verbose:
                print(f"│   ├── no conflict among the included groups ({exc}). Either the cause "
                      f"involves a group excluded by keep_groups, or the failure was numerical "
                      f"rather than infeasible.")
            return {'dropped': dropped, 'status': 'NO_CONFLICT_FOUND', 'rounds': rounds}

        iis = _iis_rows(probe)
        by_group = _by_group(iis)
        candidates = _droppable_rows(iis, droppable)
        rounds.append({'round': rnd, 'iis_size': len(iis),
                       'by_group': {g: len(v) for g, v in by_group.items()},
                       'candidates': len(candidates)})
        if verbose:
            print(f"│   ├── round {rnd}: IIS spans { {g: len(v) for g, v in by_group.items()} }")

        if not candidates:
            # The conflict lies entirely among rows we refuse to give up. Report it rather than
            # quietly relaxing something that defines the scenario.
            if verbose:
                print("│   │   └── no droppable row in the IIS: the conflict is between constraints "
                      "that define the scenario, so nothing can be relaxed without changing it.")
            return {'dropped': dropped, 'status': 'INFEASIBLE_UNRESOLVABLE',
                    'rounds': rounds, 'iis': by_group}

        # Give up the least-preferred group present, ONE row per round, so we never drop more than
        # the conflict actually requires.
        victim = candidates[0]
        dropped.append(victim)
        if verbose:
            print(f"│   │   └── dropping [{group_of(victim)}] {victim}")

    return {'dropped': dropped, 'status': 'MAX_ROUNDS', 'rounds': rounds}


# ---------------------------------------------------------------------------- #
# Post-mortem diagnosis (interactive; typically on a model read back from MPS) #
# ---------------------------------------------------------------------------- #

def find_blocking_groups(model, candidates=None, time_limit: float | None = None,
                         verbose: bool = True) -> pd.DataFrame:
    """Remove one group at a time; whichever removals restore feasibility are implicated.

    If NO single removal helps, the conflict needs two or more groups together — the pair sweep in
    `diagnose` picks that up. Structural groups are never removed: dropping `cell_usage` makes
    almost everything feasible and tells you nothing.
    """
    groups = [g for g in (candidates or CONSTRAINT_GROUPS)
              if g not in STRUCTURAL and g in set(group_counts(model).index)]
    rows = []
    for g in groups:
        status, secs = is_feasible(model, drop_groups=(g,), time_limit=time_limit)
        rows.append({'dropped': g, 'status': status, 'restores_feasibility': status == 'OPTIMAL',
                     'seconds': round(secs, 1)})
        if verbose:
            flag = '  <<< implicated' if status == 'OPTIMAL' else ''
            print(f"    drop {g:14s} -> {status:12s} ({secs:5.1f}s){flag}")
    return pd.DataFrame(rows)


def blocking_rows(model, time_limit: float | None = None) -> dict:
    """The individual rows Gurobi certifies as jointly unsatisfiable, grouped by family.

    An IIS is IRREDUCIBLE: removing any member makes the subsystem feasible. So an IIS containing
    one biodiversity row and one cap row is a certificate that those two conflict — no separate
    pairwise experiment needed. It returns *an* irreducible set, not the only one, so a second
    conflict elsewhere may be reported on a later pass.
    """
    m = _feasibility_copy(model, time_limit)
    m.optimize()
    if m.Status != GRB.INFEASIBLE:
        return {'status': STATUS.get(m.Status, str(m.Status)), 'rows': {}, 'bounds': {}}
    m.computeIIS()

    # Variable BOUNDS matter as much as rows here. An irreversibility lock-in is exactly a variable
    # whose LOWER bound cannot be given back — `get_non_ag_lb_matrices` pins it to last year's
    # allocation — so a lower bound appearing in the IIS names the locked lever, and the variable
    # name carries the cell. That is what turns "this row is unreachable" into "because cell R is
    # already committed to lever K".
    bounds = {'lower': [], 'upper': []}
    for v in m.getVars():
        if v.IISLB:
            bounds['lower'].append((v.VarName, v.LB))
        if v.IISUB:
            bounds['upper'].append((v.VarName, v.UB))
    return {'status': 'INFEASIBLE', 'rows': _by_group(_iis_rows(m)), 'bounds': bounds}


def diagnose(model, time_limit: float | None = None, do_pairs: bool = True,
             verbose: bool = True) -> dict:
    """Full report: is it infeasible, which groups are implicated, which rows.

    Order matters for cost. The IIS is cheap (~5 s on a 2.5 M-row model) and often answers outright,
    so it runs first; the group sweep is one solve per group and only earns its keep when the IIS
    result needs corroborating or when several conflicts coexist.
    """
    report = {}
    if verbose:
        print(f"\n=== diagnosing model: {model.NumVars:,} vars, {model.NumConstrs:,} constrs ===")
        print(group_counts(model).to_string())

    status, secs = is_feasible(model, time_limit=time_limit)
    report['status'] = status
    if verbose:
        print(f"\nfeasibility (zero objective, DualReductions=0): {status}  ({secs:.1f}s)")
    if status == 'OPTIMAL':
        if verbose:
            print("  Model is feasible — the failure was not infeasibility. Look at the objective, "
                  "the retry ladder, or numerics.")
        return report

    iis = blocking_rows(model, time_limit=time_limit)
    report['iis'] = iis
    if verbose and iis['rows']:
        print("\nIIS — irreducible set of jointly unsatisfiable rows:")
        for g, names in sorted(iis['rows'].items(), key=lambda kv: len(kv[1])):
            print(f"  {g:14s} {len(names):>6,} row(s)")
            for n in names[:8]:
                print(f"      {n}")
            if len(names) > 8:
                print(f"      ... and {len(names) - 8:,} more")
        implicated = [g for g in iis['rows'] if g not in STRUCTURAL]
        if len(implicated) > 1:
            print(f"\n  => the conflict spans {implicated}: each is satisfiable alone, and removing "
                  f"any one of these rows restores feasibility.")

    lower = iis.get('bounds', {}).get('lower', [])
    if verbose and lower:
        # A non-zero lower bound in the IIS is a LOCK-IN: share the model is forced to keep. The
        # variable name carries the cell, so this is what pinpoints "which cell, which lever".
        pinned = [(n, lb) for n, lb in lower if lb > 0]
        print(f"\nvariable bounds in the IIS: {len(lower):,} lower, "
              f"{len(iis['bounds']['upper']):,} upper; {len(pinned):,} lower bounds are NON-ZERO "
              f"(forced share — irreversibility lock-in):")
        for n, lb in sorted(pinned, key=lambda x: -x[1])[:10]:
            print(f"      {n:44s} lb={lb:.6f}")

    if do_pairs:
        if verbose:
            print("\ngroup sweep — remove one group at a time:")
        report['groups'] = find_blocking_groups(model, time_limit=time_limit, verbose=verbose)
        if verbose and not report['groups']['restores_feasibility'].any():
            print("  No single group restores feasibility: the conflict needs two or more "
                  "together. The IIS above names the specific rows.")
    return report
