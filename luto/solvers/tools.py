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

import gurobipy as gp
import numpy as np
import pandas as pd

from luto.solvers import row_table


# ---------------------------------------------------------------------------- #
# Shadow prices (called by simulation.py after an ACCEPTED solve)              #
# ---------------------------------------------------------------------------- #

PRICED = {                                      # the priced families, in the CSV's order: family -> (its 'constraint' label, the unit of its rhs)
    'GBF2': ('GBF2', 'ha'),
    'GBF3_NVIS': ('GBF3_NVIS', 'ha'),
    'GBF4_SNES': ('GBF4_SNES', 'ha'),
    'GBF4_ECNES': ('GBF4_ECNES', 'ha'),
    'GBF8': ('GBF8', 'ha'),
    'water': ('Water', 'ML'),
    'ghg': ('GHG', 'tCO2e'),
    'demand': ('Demand', 't'),
    'renewable': ('renewable', 'MWh'),          # the label is the renewable type, read off am_idx below
    'regional_adoption_ag': ('Regional_Adoption', 'ha'),
    'regional_adoption_nonag': ('Regional_Adoption', 'ha'),
    'regional_adoption_nonag_sum': ('Regional_Adoption', 'ha'),
}


def _priced_rows(T, family: str) -> np.ndarray:
    """The rows of one family the shadow prices cover, as row indices in table order: the ACTIVE rows, and the rows
    dropped before the build as redundant (priced 0) — empty where the family was not built. A row removed by name
    (``LutoSolver.remove_constraints_by_name``) is neither, and gets no price."""
    return np.flatnonzero((T['family'].values == family) & (T['active'].values | T['redundant'].values))


def _label(T, field: str, rows: np.ndarray) -> np.ndarray:
    """One label field of the row table at ``rows``, '' where the row's family has none."""
    labels = T[field].values[rows]
    return np.where(labels == None, '', labels).astype(object)   # noqa: E711 — an object array against None


def record_shadow_prices(luto_solver, target_year, out_dir) -> None:
    """Every priced row's shadow price into ``shadow_prices_{target_year}.csv``, as ONE query on the row table.

    The duals are read ONCE — one batched ``getAttr('Pi')`` over the priced rows in the model — and written onto the
    table as ``rows['pi']`` (NaN where not read; 0 on a row dropped before the build as redundant: every feasible point
    leaves it slack). Then every column of the CSV is a field of the table at the priced rows: ``shadow_price`` =
    pi · 1e6 / scale (the objective is million AUD, the row is scaled), ``shadow_price_AUD`` = pi · 1e6 · rhs, the
    labels read off its columns. Probes the simplex basis once first (barrier-only solves have unreliable duals → skip the year). The file is written fresh
    per year, so a resume / re-run simply overwrites the year's file.
    """
    T = luto_solver.rows
    rows = np.concatenate([_priced_rows(T, family) for family in PRICED])          # the CSV's rows, family by family in PRICED's order
    if not rows.size:
        print(f"No active constraints to record shadow prices for {target_year}.")
        return

    # ── the duals: Constr.Pi is a clean basic dual only when the accepted solve left a simplex basis; CBasis raises
    #    GurobiError on a barrier-only solve (no basis) → duals unreliable, skip the year. One handle off the table
    #    probes it — never `model.getConstrs()`, a Python list of every row of the model ──
    built = rows[T['active'].values[rows]]                                          # the priced rows in the model (the rest were dropped before the build)
    try:
        if built.size:
            _ = T['constr'].values[built[0]].CBasis
    except gp.GurobiError:
        print(f"Skipping shadow prices for {target_year}: accepted solve has no simplex basis "
              f"(barrier-only) — duals would be unreliable.")
        return
    pi = np.full(T.sizes['row'], np.nan)
    pi[rows] = 0.0
    pi[built] = luto_solver.gurobi_model.getAttr('Pi', T['constr'].values[built].tolist())
    T['pi'] = (('row',), pi)

    # ── the record: every column read off the table at the priced rows ──
    family  = T['family'].values[rows]
    name    = T['name'].values[rows]
    scale   = T['scale'].values[rows]
    rhs     = T['rhs'].values[rows]
    dropped = T['redundant'].values[rows]
    pi      = pi[rows]
    constraint = np.array([PRICED[f][0] for f in family], dtype=object)
    unit       = np.array([PRICED[f][1] for f in family], dtype=object)
    region, item, presence = _label(T, 'region', rows), _label(T, 'GBF_target', rows), _label(T, 'GBF4_presence', rows)
    # what each family reports where the table's own labels are not the CSV's
    on = family == 'GBF2'
    region[on] = 'Australia'
    on = np.isin(family, ('water', 'ghg', 'regional_adoption_ag', 'regional_adoption_nonag', 'regional_adoption_nonag_sum'))
    item[on] = name[on]                                                              # the row name says which region / cap ...
    region[on] = ''                                                                  # ... so the region column stays empty (the table's own region label is the id)
    on = family == 'demand'
    item[on] = _label(T, 'demand_commodity', rows)[on]
    presence[on] = _label(T, 'demand_bound', rows)[on]                               # eq / lower / upper: a commodity's paired bounds stay distinguishable
    on = family == 'renewable'
    constraint[on] = np.asarray(luto_solver.options, dtype=object)[T['am_idx'].values[rows][on]]   # the renewable type (its region is the state)

    shadow_price = pi * 1e6 / scale
    shadow_price_AUD = pi * 1e6 * rhs
    shadow_price_AUD[dropped] = 0.0                                                  # not -0.0 where the rhs is negative
    df = pd.DataFrame(dict(year=target_year, constraint=constraint, region=region, item=item, presence=presence,
                           pi_rescaled=pi, scale=scale, shadow_price=shadow_price, shadow_price_AUD=shadow_price_AUD,
                           unit=unit, dropped=dropped))
    df.to_csv(f"{out_dir}/shadow_prices_{target_year}.csv", index=False)
