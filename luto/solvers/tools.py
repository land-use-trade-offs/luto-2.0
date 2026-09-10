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

def _active_rows(T, family: str) -> np.ndarray:
    """The ACTIVE rows one family holds on the row table, as row indices in table order — empty where the
    family was not built or every row was dropped (``LutoSolver.remove_constraints_by_name``)."""
    span = row_table.family_rows(T, family)
    if span is None:
        return np.empty(0, dtype=np.int64)
    return np.arange(span.start, span.stop)[T['active'].values[span]]


def _labels(T, field: str, rows: np.ndarray) -> np.ndarray:
    """One coded key field of the row table at ``rows``, as labels."""
    return row_table.decode(T, field, rows)


def _price(constr, scale, unit: str) -> dict:
    """The numeric columns of one shadow-price record, from a row's Gurobi handle and its row scale."""
    So = 1e6                                    # the objective is in million AUD
    Ss = float(scale)
    pi = float(constr.Pi)
    return {"pi_rescaled": pi, "scale": Ss, "shadow_price": pi * So / Ss,
            "shadow_price_AUD": pi * So * float(constr.RHS), "unit": unit}


def calc_shadow_price_GBF2(luto_solver, inputs, target_year) -> pd.DataFrame:
    """GBF2 priority-degraded-area constraint shadow price (AUD per real ha of target)."""
    T = luto_solver.rows
    r = _active_rows(T, 'GBF2')
    return pd.DataFrame([
        {"year": target_year, "constraint": "GBF2", "region": "Australia", "item": "", "presence": "", **_price(constr, scale, "ha")}
        for constr, scale in zip(T['constr'].values[r], T['scale'].values[r])
    ])


def calc_shadow_price_GBF3_NVIS(luto_solver, inputs, target_year) -> pd.DataFrame:
    """GBF3 NVIS vegetation-group constraint shadow prices (AUD per real ha of target)."""
    T = luto_solver.rows
    r = _active_rows(T, 'GBF3_NVIS')
    return pd.DataFrame([
        {"year": target_year, "constraint": "GBF3_NVIS", "region": region, "item": group, "presence": "", **_price(constr, scale, "ha")}
        for constr, scale, region, group in zip(T['constr'].values[r], T['scale'].values[r], _labels(T, 'region', r), _labels(T, 'item', r))
    ])


def calc_shadow_price_GBF4_SNES(luto_solver, inputs, target_year) -> pd.DataFrame:
    """GBF4 SNES species constraint shadow prices (AUD per real ha of target)."""
    T = luto_solver.rows
    r = _active_rows(T, 'GBF4_SNES')
    return pd.DataFrame([
        {"year": target_year, "constraint": "GBF4_SNES", "region": region, "item": species, "presence": presence, **_price(constr, scale, "ha")}
        for constr, scale, region, species, presence in zip(T['constr'].values[r], T['scale'].values[r],
                                                            _labels(T, 'region', r), _labels(T, 'item', r), _labels(T, 'presence', r))
    ])


def calc_shadow_price_GBF4_ECNES(luto_solver, inputs, target_year) -> pd.DataFrame:
    """GBF4 ECNES ecological-community constraint shadow prices (AUD per real ha of target)."""
    T = luto_solver.rows
    r = _active_rows(T, 'GBF4_ECNES')
    return pd.DataFrame([
        {"year": target_year, "constraint": "GBF4_ECNES", "region": region, "item": community, "presence": presence, **_price(constr, scale, "ha")}
        for constr, scale, region, community, presence in zip(T['constr'].values[r], T['scale'].values[r],
                                                              _labels(T, 'region', r), _labels(T, 'item', r), _labels(T, 'presence', r))
    ])


def calc_shadow_price_GBF8(luto_solver, inputs, target_year) -> pd.DataFrame:
    """GBF8 species-conservation constraint shadow prices (AUD per real ha of target)."""
    T = luto_solver.rows
    r = _active_rows(T, 'GBF8')
    return pd.DataFrame([
        {"year": target_year, "constraint": "GBF8", "region": region, "item": species, "presence": "", **_price(constr, scale, "ha")}
        for constr, scale, region, species in zip(T['constr'].values[r], T['scale'].values[r], _labels(T, 'region', r), _labels(T, 'item', r))
    ])


def calc_shadow_price_Water(luto_solver, inputs, target_year) -> pd.DataFrame:
    """Per-region water-yield constraint shadow prices (AUD per real ML of target); ``item`` is the row name."""
    T = luto_solver.rows
    r = _active_rows(T, 'water')
    return pd.DataFrame([
        {"year": target_year, "constraint": "Water", "region": "", "item": name, "presence": "", **_price(constr, scale, "ML")}
        for constr, scale, name in zip(T['constr'].values[r], T['scale'].values[r], T['name'].values[r])
    ])


def calc_shadow_price_GHG(luto_solver, inputs, target_year) -> pd.DataFrame:
    """GHG-emissions constraint shadow price (AUD per real tCO2e of target); ``item`` is the row name."""
    T = luto_solver.rows
    r = _active_rows(T, 'ghg')
    return pd.DataFrame([
        {"year": target_year, "constraint": "GHG", "region": "", "item": name, "presence": "", **_price(constr, scale, "tCO2e")}
        for constr, scale, name in zip(T['constr'].values[r], T['scale'].values[r], T['name'].values[r])
    ])


def calc_shadow_price_Demand(luto_solver, inputs, target_year) -> pd.DataFrame:
    """Per-commodity production/demand constraint shadow prices (AUD per real tonne of demand).

    ``presence`` holds the bound kind (eq/lower/upper) so a commodity's paired bounds stay
    distinguishable; ``inputs`` is the step's ``RowInputs`` (the commodity names).
    """
    T = luto_solver.rows
    r = _active_rows(T, 'demand')
    return pd.DataFrame([
        {"year": target_year, "constraint": "Demand", "region": "", "item": inputs.commodity_names[commodity], "presence": bound, **_price(constr, scale, "t")}
        for constr, scale, commodity, bound in zip(T['constr'].values[r], T['scale'].values[r], T['commodity'].values[r], _labels(T, 'bound', r))
    ])


def calc_shadow_price_Renewable(luto_solver, inputs, target_year) -> pd.DataFrame:
    """State-level renewable-generation-target shadow prices (AUD per real MWh of target); ``constraint``
    is the renewable type, ``region`` the state. Every (type, state) row carries its own row scale."""
    T = luto_solver.rows
    r = _active_rows(T, 'renewable')
    options = luto_solver.cols.attrs['options']
    return pd.DataFrame([
        {"year": target_year, "constraint": options[am_idx], "region": state, "item": "", "presence": "", **_price(constr, scale, "MWh")}
        for constr, scale, am_idx, state in zip(T['constr'].values[r], T['scale'].values[r], T['am_idx'].values[r], _labels(T, 'state', r))
    ])


def calc_shadow_price_Regional_Adoption(luto_solver, inputs, target_year) -> pd.DataFrame:
    """Regional adoption area-cap shadow prices (AUD per real ha of cap), the three families in row order
    (ag, non-ag, non-ag sum); ``item`` is the row name. These rows are not rescaled, so scale = 1."""
    T = luto_solver.rows
    r = np.concatenate([_active_rows(T, family) for family in ('regional_adoption_ag', 'regional_adoption_nonag', 'regional_adoption_nonag_sum')])
    return pd.DataFrame([
        {"year": target_year, "constraint": "Regional_Adoption", "region": "", "item": name, "presence": "", **_price(constr, scale, "ha")}
        for constr, scale, name in zip(T['constr'].values[r], T['scale'].values[r], T['name'].values[r])
    ])


PRICED_FAMILIES = ('GBF2', 'GBF3_NVIS', 'GBF4_SNES', 'GBF4_ECNES', 'GBF8', 'water', 'ghg', 'demand', 'renewable',
                   'regional_adoption_ag', 'regional_adoption_nonag', 'regional_adoption_nonag_sum')   # the row-table families the readers above price


def record_shadow_prices(luto_solver, inputs, target_year, out_dir) -> None:
    """Compute every active constraint's shadow prices and write one CSV for the year.

    ``inputs`` is the step's ``RowInputs`` (the commodity names). Probes the simplex basis once
    (barrier-only solves have unreliable duals → skip the year), then concatenates the per-constraint
    calculators into ``shadow_prices_{target_year}.csv``. The file is written fresh per year, so a
    resume/re-run simply overwrites the year's file.
    """
    T = luto_solver.rows
    priced = np.concatenate([_active_rows(T, family) for family in PRICED_FAMILIES])
    if not priced.size:
        print(f"No active constraints to record shadow prices for {target_year}.")
        return

    # Constr.Pi is a clean basic dual only when the accepted solve left a simplex basis; CBasis
    # raises GurobiError on a barrier-only solve (no basis) → duals unreliable, skip the year.
    # One handle off the table probes it — never `model.getConstrs()`, a Python list of every row of the model.
    try:
        _ = T['constr'].values[priced[0]].CBasis
    except gp.GurobiError:
        print(f"Skipping shadow prices for {target_year}: accepted solve has no simplex basis "
              f"(barrier-only) — duals would be unreliable.")
        return

    # Each calculator returns rows for its active constraints, or a column-less empty frame.
    df = pd.concat(
        [calc(luto_solver, inputs, target_year) for calc in (
            calc_shadow_price_GBF2,
            calc_shadow_price_GBF3_NVIS,
            calc_shadow_price_GBF4_SNES,
            calc_shadow_price_GBF4_ECNES,
            calc_shadow_price_GBF8,
            calc_shadow_price_Water,
            calc_shadow_price_GHG,
            calc_shadow_price_Demand,
            calc_shadow_price_Renewable,
            calc_shadow_price_Regional_Adoption,
        )],
        ignore_index=True,
    )

    df.to_csv(f"{out_dir}/shadow_prices_{target_year}.csv", index=False)
    print(f"Recorded {len(df)} shadow prices for {target_year} -> {out_dir}/shadow_prices_{target_year}.csv")
