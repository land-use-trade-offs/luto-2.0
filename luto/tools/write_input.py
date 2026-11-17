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


import os
import sys
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

from luto import settings
from luto.data import Data

import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.transitions as ag_transitions
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.water as ag_water

import luto.economics.non_agricultural.revenue as non_ag_revenue
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.transitions as non_ag_transitions
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.water as non_ag_water


# ── Shared helpers ──────────────────────────────────────────────────────────────────────────────────────────

def ag_layers(data: Data, name, per_cell_mrj) -> list:
    """The layers of an agricultural input given per CELL as (lm, cell, lu) (AUD / t / ML per cell), per hectare:
    one for every (lm, lu), where the land use is allowed."""
    real_area = data.REAL_AREA.values.astype(np.float64)
    per_ha_mrj = np.nan_to_num(np.asarray(per_cell_mrj, dtype=np.float64) / real_area[None, :, None])

    layers = []
    allowed_mrj = ag_transitions.get_ag_allowed_mrj(data)
    for m, lm in enumerate(data.LANDMANS):
        for j, lu in enumerate(data.AGRICULTURAL_LANDUSES):
            allowed_r = allowed_mrj[m, :, j] > 0
            layers.append((name, lm, lu, np.where(allowed_r, per_ha_mrj[m, :, j], 0)))
    return layers


def non_ag_layers(data: Data, name, per_cell_rk, lm='ALL') -> list:
    """The layers of a non-agricultural input given per CELL as (cell, non-ag lu), per hectare: one per land use.
    `lm` is the name they are filed under: 'ALL' for an input of their own, 'NonAg' where they sit beside the
    dryland and irrigated layers of an agricultural input."""
    real_area = data.REAL_AREA.values.astype(np.float64)
    per_ha_rk = np.nan_to_num(np.asarray(per_cell_rk, dtype=np.float64) / real_area[:, None])
    return [
        (name, lm, lu, per_ha_rk[:, k])
        for k, lu in enumerate(data.NON_AGRICULTURAL_LANDUSES)
    ]


def input_theme(tag, catalogue, layers):
    """A theme as its writer hands it over: (tag, catalogue, layers), written by save_input_layers.

    catalogue   [{theme, input, unit, kind: 'layer' | 'mask' | 'chart'}]
    layers      [(input, lm, lu, values_r)], or with the option the layer belongs to as a fifth item
                [(input, lm, lu, values_r, option)] (option_label)."""
    return tag, catalogue, layers


def save_input_layers(data: Data, yr_cal, path, tag, catalogue, layers):
    """Write a theme's two files (see the module docstring).

    layers      [(input, lm, option, lu, values_r)]: `option` says which option of the model the layer is computed
                under ('S=245|C=on': option_label), 'ALL' for a layer no option changes. A layer with no value anywhere
                is left out, and so is an input of the catalogue that ends with no layer.
    An input of kind 'chart' has no layer: it is listed in the catalogue so the report offers it.
    """
    from luto.tools.write_output import save2nc          # write_output imports this module

    catalogue = pd.DataFrame(catalogue, columns=['theme', 'input', 'unit', 'kind']).drop_duplicates('input')
    kept = []
    for name, lm, option, lu, values_r in layers:
        values_r = np.nan_to_num(np.asarray(values_r, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(values_r):
            kept.append((name, lm, option, lu, values_r.astype(np.float32)))

    catalogue[catalogue['input'].isin({name for name, *_ in kept}) | (catalogue['kind'] == 'chart')]\
        .to_csv(os.path.join(path, f'inputs_catalogue_{tag}_{yr_cal}.csv'), index=False)
    if not kept:
        return f"Inputs ({tag}) for year {yr_cal}: no layer"

    layer_index = pd.MultiIndex.from_tuples([key for *key, _ in kept], names=['input', 'lm', 'option', 'lu'])
    layers_xr = xr.DataArray(
        np.stack([values for *_, values in kept]),
        dims=['layer', 'cell'],
        coords={**xr.Coordinates.from_pandas_multiindex(layer_index, 'layer'), 'cell': np.arange(data.NCELLS)},
    )
    save2nc(layers_xr, os.path.join(path, f'xr_input_layers_{tag}_{yr_cal}.nc'))

    return f"Inputs ({tag}) written for year {yr_cal}: {len(kept)} layers"


# ── Yield ───────────────────────────────────────────────────────────────────────────────────────────────────

def yield_trend_group(product) -> str | None:
    """The group a product's yield trend is given for (the trend files hold one series per group, repeated for
    each of its products); None for the unallocated land uses, which yield nothing."""
    product = str(product).upper()
    if product.startswith('UNALLOCATED'):
        return None
    if product.startswith('BEEF'):
        return 'Beef'
    if product.startswith('DAIRY'):
        return 'Dairy'
    if product.startswith('SHEEP'):
        return 'Sheep wool' if product.endswith('WOOL') else 'Sheep'
    return 'Crops'


def get_yield_trend_sources() -> pd.DataFrame:
    """Both yield-trend inputs the model can run on (settings.PRODUCTIVITY_TREND picks one), as one long table of
    multipliers on 2010: [source, scenario, kind, lm, group, Year, value].

    source 'BAU'      yieldincreases_bau2022.csv: one projection, from 2010 (the file's first row).
    source 'ABARES'   yieldincreases_ag_2050.xlsx, the `<scenario>_actual` sheets the model reads: the observed
                      productivity up to the last year every scenario agrees on (kind 'history'), then the
                      scenario's projection (kind 'projection').
    lm is 'ALL' where dryland and irrigated share the series; 'Sheep wool' is kept only where it differs from 'Sheep'."""
    def to_groups(table: pd.DataFrame) -> pd.DataFrame:
        """(year) x (lm, product) -> long [lm, group, Year, value], a product standing for its group"""
        long = table.stack([0, 1], future_stack=True).rename('value').reset_index()
        long.columns = ['Year', 'lm', 'product', 'value']
        long['group'] = long['product'].map(yield_trend_group)
        long = long.dropna(subset=['group', 'value']).groupby(['lm', 'group', 'Year'], as_index=False)['value'].mean()
        wide = long.pivot(index='Year', columns=['lm', 'group'], values='value')
        if 'Sheep wool' in wide.columns.get_level_values('group') and all(
                np.allclose(wide[(lm, 'Sheep wool')], wide[(lm, 'Sheep')]) for lm in wide.columns.get_level_values('lm').unique()):
            long = long[long['group'] != 'Sheep wool']
        if set(wide.columns.get_level_values('lm')) == {'dry', 'irr'} and np.allclose(wide['dry'], wide['irr'][wide['dry'].columns]):
            long = long[long['lm'] == 'dry'].assign(lm='ALL')
        return long.replace({'lm': {'dry': 'Dryland', 'irr': 'Irrigated'}})

    bau = pd.read_csv(os.path.join(settings.INPUT_DIR, 'yieldincreases_bau2022.csv'), header=[0, 1]).astype(np.float64)
    bau.index = bau.index + 2010                   # the csv has no year column: its first row is 2010
    parts = [to_groups(bau).assign(source='BAU', scenario='BAU', kind='projection')]

    scenarios = {
        scenario: pd.read_excel(os.path.join(settings.INPUT_DIR, 'yieldincreases_ag_2050.xlsx'), sheet_name=f'{scenario}_actual', header=[0, 1], index_col=0).astype(np.float64)
        for scenario in ['low', 'medium', 'high', 'very_high']
    }
    first, *others = scenarios.values()
    same_everywhere = np.all([np.isclose(first, other).all(axis=1) for other in others], axis=0)      # per year
    last_history_year = int(first.index[same_everywhere].max())
    parts.append(to_groups(first.loc[:last_history_year]).assign(source='ABARES', scenario='history', kind='history'))
    for scenario, table in scenarios.items():
        parts.append(to_groups(table.loc[last_history_year:]).assign(source='ABARES', scenario=scenario, kind='projection'))

    return pd.concat(parts, ignore_index=True)[['source', 'scenario', 'kind', 'lm', 'group', 'Year', 'value']]


def input_yield(data: Data, yr_cal, path):
    """The two yield assumptions.

    Climate change   a layer per (irrigation, land use), % change on the base year: the impact differs from cell
                     to cell.
    Yield trend      a chart only (kind 'chart'): one multiplier per product for the whole country, so there is
                     nothing to map. Both trend inputs the model can run on go to inputs_yield_trend_<year>.csv
                     (get_yield_trend_sources), for the report to draw the one in use beside the other."""
    yr_idx = yr_cal - data.YR_CAL_BASE
    allowed_mrj = ag_transitions.get_ag_allowed_mrj(data)

    layers = []
    for m, lm in enumerate(data.LANDMANS):
        for j, lu in enumerate(data.AGRICULTURAL_LANDUSES):
            products = [pr for pr in data.LU2PR_DICT[lu] if pr not in data.PR_STUBBLE]
            if not products:                           # an unallocated land use yields nothing
                continue
            allowed_r = allowed_mrj[m, :, j] > 0
            climate_r = ag_quantity.get_ccimpact(data, lu, lm, yr_idx)
            layers.append(('Climate change', lm, lu, np.where(allowed_r, (climate_r - 1) * 100, 0)))

    get_yield_trend_sources().to_csv(os.path.join(path, f'inputs_yield_trend_{yr_cal}.csv'), index=False)

    catalogue = [
        {'theme': 'Yield', 'input': 'Climate change', 'unit': '%', 'kind': 'layer'},
        {'theme': 'Yield', 'input': 'Yield trend', 'unit': '', 'kind': 'chart'},
    ]
    return input_theme('yield', catalogue, layers)


# ── Water ───────────────────────────────────────────────────────────────────────────────────────────────────

# The non-agricultural land uses the water model treats alike (economics/non_agricultural/water.py): riparian
# plantings call the environmental plantings function, BECCS the carbon plantings one, and the sheep and beef versions
# of a mixed land use differ by their drinking water alone. One layer per group; a land use not named here keeps a
# layer of its own. Destocked land is left out (NON_AG_WATER_NO_USE): it was never cleared, so against cleared land it
# would show a water use it does not have (it keeps the net yield of unallocated natural land).
NON_AG_WATER_GROUPS = {
    'Carbon plantings and BECCS': ['Carbon Plantings (Block)', 'BECCS'],
    'Environmental and riparian plantings': ['Environmental Plantings', 'Riparian Plantings'],
    'Farm forestry (with sheep or beef)': ['Sheep Carbon Plantings (Belt)', 'Beef Carbon Plantings (Belt)'],
    'Agroforestry (with sheep or beef)': ['Sheep Agroforestry', 'Beef Agroforestry'],
}
NON_AG_WATER_NO_USE = ['Destocked - natural land']


def input_water(data: Data, yr_cal, path):
    """Runoff and water requirement per land use, and the climate change impact on runoff.

    Runoff                            the BASE YEAR's, the same in every year's file (the historical runoff with
                                      WATER_CLIMATE_CHANGE_IMPACT off): a year's own runoff differs from it by a
                                      few percent, which a map of the level can not show.
    Climate change impact on runoff   what the years add to it: this year's runoff of a land use less its
                                      base-year runoff (nothing written with the setting off).
    Water requirement                 irrigation and livestock drinking water of the agricultural land uses. The
                                      non-agricultural land uses sit beside them under lm 'NonAg': the model
                                      gives them a net yield and no requirement, so theirs is the water they
                                      use against cleared land, the base-year runoff of shallow-rooted cover
                                      less their base-year net yield (the extra water their trees take, plus
                                      the drinking water of the livestock on the rest of a mixed land use).
                                      One layer per group of NON_AG_WATER_GROUPS, the mean of its land uses.
    Every input is a layer per (irrigation, land use)."""
    yr_idx = yr_cal - data.YR_CAL_BASE
    real_area = data.REAL_AREA.values.astype(np.float64)

    # The runoff the base year is read with: its own, or the historical one where the climate is held (None: the year's)
    base_dr, base_sr = (None, None) if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on' else (data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    runoff_base_mrj = ag_water.get_wyield_matrices(data, 0, base_dr, base_sr)                            # ML/cell
    if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on':
        runoff_change_mrj = ag_water.get_wyield_matrices(data, yr_idx) - runoff_base_mrj
    else:
        runoff_change_mrj = np.zeros_like(runoff_base_mrj)
    net_yield_base_mrj = ag_water.get_water_net_yield_matrices(data, 0, base_dr, base_sr)
    non_ag_net_yield_rk = non_ag_water.get_w_net_yield_matrix(data, net_yield_base_mrj, 0, base_dr, base_sr)   # ML/cell
    runoff_shallow_r = runoff_base_mrj[0, :, data.LU_SHALLOW_ROOTED[0]]                                        # ML/cell, the same for every shallow-rooted land use
    non_ag_water_use_rk = np.nan_to_num((runoff_shallow_r[:, None] - non_ag_net_yield_rk) / real_area[:, None])   # ML/ha

    non_ag_names = list(data.NON_AGRICULTURAL_LANDUSES)
    grouped_names = {name for members in NON_AG_WATER_GROUPS.values() for name in members}
    non_ag_groups = {**NON_AG_WATER_GROUPS, **{name: [name] for name in non_ag_names if name not in grouped_names and name not in NON_AG_WATER_NO_USE}}
    non_ag_group_layers = []
    for group, members in non_ag_groups.items():
        columns = [non_ag_names.index(name) for name in members if name in non_ag_names]
        if columns:
            water_use_r = non_ag_water_use_rk[:, columns].mean(axis=1)
            non_ag_group_layers.append(('Water requirement', 'NonAg', group, water_use_r))

    layers = (
        ag_layers(data, 'Runoff', runoff_base_mrj)
        + ag_layers(data, 'Water requirement', ag_water.get_wreq_matrices(data, yr_idx))
        + non_ag_group_layers
        + ag_layers(data, 'Climate change impact on runoff', runoff_change_mrj)
    )
    catalogue = [
        {'theme': 'Water', 'input': 'Runoff', 'unit': 'ML/ha', 'kind': 'layer'},
        {'theme': 'Water', 'input': 'Water requirement', 'unit': 'ML/ha', 'kind': 'layer'},
        {'theme': 'Water', 'input': 'Climate change impact on runoff', 'unit': 'ML/ha', 'kind': 'layer'},
    ]

    return input_theme('water', catalogue, layers)


# ── GHG ─────────────────────────────────────────────────────────────────────────────────────────────────────

def input_ghg(data: Data, yr_cal, path):
    """Emissions per hectare of every land use, one input: the agricultural land uses under dryland / irrigated,
    the non-agricultural ones beside them under lm 'NonAg' (negative: sequestration). And what savanna burning
    (early dry season) abates where it is eligible; the eligibility mask itself is not written, the abatement
    layer shows it."""
    yr_idx = yr_cal - data.YR_CAL_BASE

    ag_g_mrj = ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True)                 # t CO2e/cell
    non_ag_g_rk = non_ag_ghg.get_ghg_matrix(data, ag_g_mrj)                          # t CO2e/cell
    eligible_r = np.asarray(data.SAVBURN_ELIGIBLE.values, dtype=np.float64)
    abatement_r = np.asarray(data.SAVBURN_TOTAL_TCO2E_HA.values, dtype=np.float64) * eligible_r

    layers = (
        ag_layers(data, 'Emissions and sequestration', ag_g_mrj)
        + non_ag_layers(data, 'Emissions and sequestration', non_ag_g_rk, lm='NonAg')
        + [('Savanna burning: abatement', 'ALL', 'ALL', abatement_r)]
    )
    catalogue = [
        {'theme': 'GHG', 'input': 'Emissions and sequestration', 'unit': 't CO2e/ha', 'kind': 'layer'},
        {'theme': 'GHG', 'input': 'Savanna burning: abatement', 'unit': 't CO2e/ha', 'kind': 'layer'},
    ]
    return input_theme('ghg', catalogue, layers)


# ── Economics ───────────────────────────────────────────────────────────────────────────────────────────────

def ag_cost_part_layers(data: Data, yr_idx) -> tuple[list, list]:
    """The agricultural cost split into its parts (quantity, area, fixed labour, fixed operating, fixed depreciation,
    water; for the cereals grazed after harvest, the stubble sheep's two), per hectare: one input per part, a layer
    for every (lm, lu) that carries the part. Returns (layers, the names of the inputs in the order of the model)."""
    cost_rjms = ag_cost.get_cost_matrices(data, yr_idx, aggregate=False)             # AUD/cell; columns (lu, lm, part)
    real_area = data.REAL_AREA.values.astype(np.float64)
    allowed_mrj = ag_transitions.get_ag_allowed_mrj(data)

    layers, names = [], []
    for (lu, lm, part) in cost_rjms.columns:
        if lu not in data.LU_CROPS and lu not in data.LU_LVSTK:                       # unallocated land: a column of zeros
            continue
        name = 'Agricultural cost: ' + part.removesuffix(' cost').lower()
        allowed_r = allowed_mrj[data.LANDMANS.index(lm), :, data.AGRICULTURAL_LANDUSES.index(lu)] > 0
        per_ha_r = np.nan_to_num(cost_rjms[(lu, lm, part)].values.astype(np.float64) / real_area)
        layers.append((name, lm, lu, np.where(allowed_r, per_ha_r, 0)))
        if name not in names:
            names.append(name)
    return layers, names


def input_economics(data: Data, yr_cal, path):
    """Cost and revenue per hectare of every agricultural and non-agricultural land use; the agricultural cost
    also part by part."""
    yr_idx = yr_cal - data.YR_CAL_BASE

    ag_c_mrj = ag_cost.get_cost_matrices(data, yr_idx)                               # AUD/cell
    ag_r_mrj = ag_revenue.get_rev_matrices(data, yr_idx)
    non_ag_c_rk = non_ag_cost.get_cost_matrix(data, ag_c_mrj, yr_cal)
    non_ag_r_rk = non_ag_revenue.get_rev_matrix(data, yr_cal, ag_r_mrj)
    part_layers, part_names = ag_cost_part_layers(data, yr_idx)

    layers = (
        ag_layers(data, 'Agricultural cost', ag_c_mrj)
        + part_layers
        + ag_layers(data, 'Agricultural revenue', ag_r_mrj)
        + non_ag_layers(data, 'Non-agricultural cost', non_ag_c_rk)
        + non_ag_layers(data, 'Non-agricultural revenue', non_ag_r_rk)
    )
    catalogue = [
        {'theme': 'Economics', 'input': name, 'unit': 'AUD/ha', 'kind': 'layer'}
        for name in ['Agricultural cost', *part_names, 'Agricultural revenue', 'Non-agricultural cost', 'Non-agricultural revenue']
    ]
    return input_theme('economics', catalogue, layers)


def get_transition_matrix(data: Data) -> pd.DataFrame:
    """The from-to cost table of the model (data.T_MAT, AUD/ha), every land use to every other, with what each cost
    is the sum of: [from_lu, to_lu, part, cost, allowed]. Part 'Total' is the entry of T_MAT, listed for every pair;
    `allowed` is False where T_MAT has no value (the model can not make that change).

    The parts follow how data.py builds T_MAT:
      ag -> ag                       the items of the transition cost workbook (input/transition_cost_parts.csv)
      non-ag -> ag                   as leaving unallocated natural land for that land use
      non-ag -> unallocated modified clearing the planting (the tree share only, for the mixed land uses)
      ag -> non-ag                   one item: the workbook gives it as a single cost per land use
    A report whose inputs have no parts file gets the totals only."""
    lus = list(data.T_MAT['from_lu'].values)
    total = data.T_MAT.to_pandas().reindex(index=lus, columns=lus)
    rows = [(from_lu, to_lu, 'Total', 0.0 if pd.isna(total.loc[from_lu, to_lu]) else float(total.loc[from_lu, to_lu]), bool(pd.notna(total.loc[from_lu, to_lu])))
            for from_lu in lus for to_lu in lus]

    parts_file = os.path.join(settings.INPUT_DIR, 'transition_cost_parts.csv')
    if not os.path.exists(parts_file):
        return pd.DataFrame(rows, columns=['from_lu', 'to_lu', 'part', 'cost', 'allowed'])

    ag_parts = pd.read_csv(parts_file)
    non_ag = data.NON_AGRICULTURAL_LANDUSES
    parts = [tuple(row) for row in ag_parts.itertuples(index=False)]

    # non-ag -> ag: the row of unallocated natural land (data.py: EP2AG_TRANSITION_COSTS_HA; destocked land the same)
    from_natural = ag_parts[ag_parts['from_lu'] == 'Unallocated - natural land']
    parts += [(from_lu, to_lu, part, cost) for from_lu in non_ag for _, to_lu, part, cost in from_natural.itertuples(index=False)
              if to_lu != 'Unallocated - modified land']

    # non-ag -> unallocated modified land: clearing the planting
    clearing = np.load(os.path.join(settings.INPUT_DIR, 'transition_cost_clearing_forest.npz'))
    light, dense, barrier = (float(clearing[name]) for name in ('tmat_clear_light_wood', 'tmat_clear_dense_wood', 'tmat_clear_wood_barrier'))
    tree_share = {'Sheep Agroforestry': settings.AF_PROPORTION, 'Beef Agroforestry': settings.AF_PROPORTION,
                  'Sheep Carbon Plantings (Belt)': settings.CP_BELT_PROPORTION, 'Beef Carbon Plantings (Belt)': settings.CP_BELT_PROPORTION}
    for from_lu in non_ag:
        share = tree_share.get(from_lu, 1.0)
        if from_lu == 'Destocked - natural land':
            parts += [(from_lu, 'Unallocated - modified land', 'Clearing lightly wooded land', light)]
        else:
            parts += [(from_lu, 'Unallocated - modified land', 'Clearing: barrier effect', barrier * share)]
        parts += [(from_lu, 'Unallocated - modified land', 'Clearing heavily wooded land', dense * share)]

    # ag -> non-ag: a single cost per land use in the workbook
    parts += [(from_lu, to_lu, 'Leaving agriculture for non-ag land', float(total.loc[from_lu, to_lu]))
              for from_lu in data.AGRICULTURAL_LANDUSES for to_lu in non_ag
              if pd.notna(total.loc[from_lu, to_lu]) and total.loc[from_lu, to_lu] != 0]

    # Keep the parts of the pairs the model allows; say so where they do not add up to the entry of T_MAT
    parts = pd.DataFrame(parts, columns=['from_lu', 'to_lu', 'part', 'cost'])
    parts = parts[[pd.notna(total.loc[from_lu, to_lu]) for from_lu, to_lu in zip(parts['from_lu'], parts['to_lu'])]]
    summed = parts.groupby(['from_lu', 'to_lu'])['cost'].sum().unstack().reindex(index=lus, columns=lus).fillna(0)
    off = int(((summed - total.fillna(0)).abs() > 1).values.sum())
    if off:
        print(f'Transition cost matrix: the parts of {off} from-to pairs do not add up to data.T_MAT '
              f'(input/transition_cost_parts.csv from another version of the workbook than ag_tmatrix.npy?)', flush=True)
    return pd.concat([pd.DataFrame(rows, columns=['from_lu', 'to_lu', 'part', 'cost', 'allowed']), parts.assign(allowed=True)], ignore_index=True)


def input_transition_cost(data: Data, yr_cal, path):
    """The cost of leaving the BASE-YEAR land use of a cell for every agricultural (lm, lu) and every
    non-agricultural land use. A cell that held several land uses takes each one's cost weighted by the share of
    the cell it held. Upfront, or per year when the run amortises its upfront costs."""
    yr_idx = yr_cal - data.YR_CAL_BASE
    base_year = data.YR_CAL_BASE
    real_area = data.REAL_AREA.values.astype(np.float64)
    unit = 'AUD/ha/yr' if settings.AMORTISE_UPFRONT_COSTS else 'AUD/ha'

    allowed_mrj = ag_transitions.get_ag_allowed_mrj(data)
    share_mrj = data.ag_dvars[base_year]                                             # the share of a cell every (lm, lu) held
    source_cells = ag_transitions.get_base_dvar_mj_cell_map(data, base_year)         # {(from_m, from_j): cells}
    to_non_ag = non_ag_transitions.get_transition_matrix_ag2nonag(data, base_year, yr_cal)   # {lu: {(from_m, from_j): AUD/cell}}

    # Sum over the sources of a cell, each weighted by its share
    share_r = np.zeros(data.NCELLS)
    to_ag_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))                    # AUD/cell, [to_m, r, to_j]
    to_non_ag_rk = np.zeros((data.NCELLS, len(data.NON_AGRICULTURAL_LANDUSES)))      # AUD/cell
    for (from_m, from_j), cells in source_cells.items():
        share = share_mrj[from_m, cells, from_j]
        share_r[cells] += share
        to_ag_mrj[:, cells, :] += np.nan_to_num(ag_transitions.get_transition_matrices_ag2ag(data, yr_idx, from_m, from_j, cells)) * share[None, :, None]
        for k, lu in enumerate(data.NON_AGRICULTURAL_LANDUSES):
            if (from_m, from_j) in to_non_ag.get(lu, {}):
                to_non_ag_rk[cells, k] += np.nan_to_num(to_non_ag[lu][(from_m, from_j)]) * share

    has_source_r = share_r > 0
    safe_share_r = np.where(has_source_r, share_r, 1)

    layers = []
    for m, lm in enumerate(data.LANDMANS):
        for j, lu in enumerate(data.AGRICULTURAL_LANDUSES):
            allowed_r = (allowed_mrj[m, :, j] > 0) & has_source_r
            layers.append(('Transition cost to agriculture', lm, lu, np.where(allowed_r, to_ag_mrj[m, :, j] / safe_share_r / real_area, 0)))
    for k, lu in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        layers.append(('Transition cost to non-agriculture', 'ALL', lu, np.where(has_source_r, to_non_ag_rk[:, k] / safe_share_r / real_area, 0)))

    # The from-to table behind them, the same in every year: no map, a matrix (kind 'chart')
    get_transition_matrix(data).to_csv(os.path.join(path, f'inputs_transition_matrix_{yr_cal}.csv'), index=False)

    catalogue = [
        {'theme': 'Economics', 'input': 'Transition cost to agriculture', 'unit': unit, 'kind': 'layer'},
        {'theme': 'Economics', 'input': 'Transition cost to non-agriculture', 'unit': unit, 'kind': 'layer'},
        {'theme': 'Economics', 'input': 'Transition cost matrix', 'unit': 'AUD/ha', 'kind': 'chart'},
    ]
    return input_theme('transition_cost', catalogue, layers)


# ── Exclusion ───────────────────────────────────────────────────────────────────────────────────────────────

def input_exclusion(data: Data, yr_cal, path):
    """Where a land use may go at all (masks, not aggregated).

    Agricultural: 1 where the (lm, lu) is permitted — present in its SA2 in 2010, and outside the no-go regions.
    Non-agricultural: the share of the cell the land use may take (transition rules, no-go regions, physical caps)."""
    base_year = data.YR_CAL_BASE
    allowed_mrj = ag_transitions.get_ag_allowed_mrj(data)
    upper_rk = non_ag_transitions.get_non_ag_ub_matrices(data, data.non_ag_dvars[base_year], data.ag_dvars[base_year])

    layers = [
        ('Agricultural land use allowed', lm, lu, allowed_mrj[m, :, j])
        for m, lm in enumerate(data.LANDMANS) for j, lu in enumerate(data.AGRICULTURAL_LANDUSES)
    ] + [
        ('Non-agricultural share allowed', 'ALL', lu, upper_rk[:, k])
        for k, lu in enumerate(data.NON_AGRICULTURAL_LANDUSES)
    ]
    catalogue = [
        {'theme': 'Exclusion', 'input': 'Agricultural land use allowed', 'unit': '0 / 1', 'kind': 'mask'},
        {'theme': 'Exclusion', 'input': 'Non-agricultural share allowed', 'unit': '0 - 1', 'kind': 'mask'},
    ]
    # Which land use may follow which, whatever the cell: the pairs data.T_MAT has no value for. No map: the report
    # draws it from the from-to table that write_input_transition_cost writes (inputs_transition_matrix_<year>.csv).
    catalogue = catalogue + [{'theme': 'Exclusion', 'input': 'Transition exclusion matrix', 'unit': '', 'kind': 'chart'}]
    return input_theme('exclusion', catalogue, layers)


# ── Biodiversity ────────────────────────────────────────────────────────────────────────────────────────────

def get_connectivity_score_fullres(data: Data) -> np.ndarray:
    """The connectivity score of every full-resolution land cell, scaled as Data.__init__ scales the score of the
    model cells (CONNECTIVITY_SCORE): between settings.CONNECTIVITY_LB and 1, over the range the model cells span."""
    if settings.CONNECTIVITY_SOURCE == 'NONE':
        return np.ones(data.LUMASK.size, dtype=np.float64)
    variable = {'NCI': 'DCCEEW_NCI', 'DWI': 'NATURAL_AREA_CONNECTIVITY'}[settings.CONNECTIVITY_SOURCE]
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, 'bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.nc')) as bio_priority:
        raw = bio_priority[variable].values.astype(np.float32)
    span = (raw[data.MASK].min(), raw[data.MASK].max())                       # the range of the model cells, as data.py
    scaled_to = (settings.CONNECTIVITY_LB, 1) if settings.CONNECTIVITY_SOURCE == 'NCI' else (1, settings.CONNECTIVITY_LB)
    return np.interp(raw, span, scaled_to).astype(np.float64)


INPUT_HCAS_SOURCES = ['10', '25', '50', '75', '90', 'CSV_DEFINED', 'AG_UNIFORM']    # the options of settings.HCAS_CONTRIBUTION_PERCENTILE


def get_hcas_contribution(data: Data, source: str) -> dict:
    """{agricultural land-use code: habitat contribution (0-1)} under a source of settings.HCAS_CONTRIBUTION_PERCENTILE,
    worked out as Data.__init__ works out BIO_HABITAT_CONTRIBUTION_LOOK_UP for the source of the run."""
    lookup = pd.read_csv(os.path.join(settings.INPUT_DIR, 'bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv')).set_index('lu')
    if source in ('10', '25', '50', '75', '90'):
        # the percentile of the condition observed under each land use, relative to unallocated natural land
        contribution = lookup[f'PERCENTILE_{source}'].to_dict()
        natural_scale = contribution[data.DESC2AGLU['Unallocated - natural land']]
        contribution = {int(j): value / natural_scale for j, value in contribution.items()}
    elif source == 'CSV_DEFINED':
        contribution = lookup['CSV_DEFINED'].to_dict()
    elif source == 'AG_UNIFORM':
        # one value for every modified land use (the value of the settings when the cache is built); natural land keeps the table's
        contribution = lookup['AG_UNIFORM'].to_dict()
        for j in data.LU_MODIFIED_LAND:
            contribution[j] = float(settings.HCAS_AG_UNIFORM_CONTRIBUTION)
    else:
        raise ValueError(f'Unknown habitat contribution source: {source}')
    return {int(j): round(float(value), settings.ROUND_DECIMALS) for j, value in contribution.items()}


def get_hcas_table(data: Data) -> pd.DataFrame:
    """The habitat contribution of every land use (0-1), under every source of the agricultural values
    (settings.HCAS_CONTRIBUTION_PERCENTILE), as the rows of the report's table: theme, table, unit, option, group,
    row, value, reference. It reads the land-use names and codes of `data` and no spatial layer."""
    # The agricultural land uses in the groups of settings.LU2TYPE (cropping / intensive cropping / horticulture),
    # the others by what they are; then the non-agricultural land uses and savanna burning
    crop_groups = {'cropping': 'Cropping', 'int_cropping': 'Intensive cropping', 'horticulture': 'Horticulture'}
    def group_of(land_use):
        if land_use in settings.LU2TYPE:
            return crop_groups[settings.LU2TYPE[land_use]]
        return 'Unallocated land' if land_use.startswith('Unallocated') else 'Livestock'

    # The settings of the run that are not agricultural land uses: the same under every source
    other_contributions = [
        ('Non-agricultural land use', 'Environmental Plantings', settings.BIO_CONTRIBUTION_ENV_PLANTING),
        ('Non-agricultural land use', 'Riparian Plantings', settings.BIO_CONTRIBUTION_RIPARIAN_PLANTING),
        ('Non-agricultural land use', 'Agroforestry', settings.BIO_CONTRIBUTION_AGROFORESTRY),
        ('Non-agricultural land use', 'Carbon Plantings (Block)', settings.BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK),
        ('Non-agricultural land use', 'Carbon Plantings (Belt)', settings.BIO_CONTRIBUTION_CARBON_PLANTING_BELT),
        ('Non-agricultural land use', 'Destocked - natural land', settings.BIO_CONTRIBUTION_DESTOCKING),
        ('Agricultural management', 'Savanna burning (early dry season)', settings.BIO_CONTRIBUTION_LDS),
    ]
    group_order = ['Cropping', 'Intensive cropping', 'Horticulture', 'Livestock', 'Unallocated land', 'Non-agricultural land use', 'Agricultural management']
    # One table per source of the agricultural values (settings.HCAS_CONTRIBUTION_PERCENTILE): an option of the
    # table, as the quality layers are options of the map. `reference` is the land use's value at the 50th
    # percentile, which the report marks on the bar of every other source.
    reference = get_hcas_contribution(data, '50')
    records = []
    for source in INPUT_HCAS_SOURCES:
        contribution_of = get_hcas_contribution(data, source)
        contributions = [(group_of(data.AGLU2DESC[j]), data.AGLU2DESC[j], contribution, reference[j]) for j, contribution in contribution_of.items()] \
            + [(group, name, contribution, np.nan) for group, name, contribution in other_contributions]
        records += [
            {'theme': 'Biodiversity', 'table': 'Habitat contribution of a land use (HCAS)', 'unit': '0 - 1',
             'option': source, 'group': group, 'row': name, 'value': float(contribution), 'reference': reference_value}
            for group, name, contribution, reference_value in sorted(contributions, key=lambda item: group_order.index(item[0]))]
    return pd.DataFrame(records)


def input_biodiversity(data: Data, yr_cal, path):
    """The biodiversity quality layer of the run, the GBF2 priority degraded areas (mask), and — a table, not a
    map — the habitat contribution (HCAS) of every land use: inputs_tables_<year>.csv. The species, community and
    vegetation layers of GBF3 / GBF4 / GBF8 are not written here: there are thousands of them."""
    # Every quality layer the model can run on (settings.BIO_QUALITY_LAYERS), each an option: the layer weighted by
    # connectivity as data.BIO_QUALITY_RAW is, and the GBF2 mask it gives (data.py: the unweighted layer at or above
    # the level of its conservation performance curve at GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT).
    #
    # At RESFACTOR > 1 a model cell reads a layer at its block's centre cell, and where that cell is not a land-use
    # cell (public land, water: about one model cell in ten at RES5, the blocks along the edge of the model's land)
    # the value is that of land the model does not hold, often much higher. The maps show the mean over the block's
    # land-use cells instead (Data.get_resfactored_average_fraction of the layer with the other cells zeroed, divided
    # by the count of the cells that carry a value): the quality of the land the cell stands for.
    every_cell = np.ones(data.LUMASK.size, dtype=bool)
    connectivity = get_connectivity_score_fullres(data)
    layers = []
    for quality_layer in settings.BIO_QUALITY_LAYERS:
        option = option_label(BIO_QUALITY_LAYER=quality_layer)
        try:
            quality, performance_sheet = data.load_bio_quality_layer(quality_layer, mask=every_cell)
            curve = data.load_bio_performance_curve(performance_sheet)
        except Exception as error:                     # a layer whose file this input folder does not hold
            print(f'Inputs (biodiversity): quality layer {quality_layer} left out ({type(error).__name__}: {error})', flush=True)
            continue
        quality = np.asarray(quality, dtype=np.float64)
        in_priority_area = (quality >= curve[settings.GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT]).astype(np.float64)
        quality_r = data.get_resfactored_average_fraction(quality * connectivity * data.LUMASK, use_valid_cell_count=True)
        layers.append(('Biodiversity quality', 'ALL', 'ALL', quality_r.astype(np.float64), option))
        # the mask of a coarse cell: priority degraded area where at least half of its land-use cells are
        share_in_priority_area = data.get_resfactored_sum(in_priority_area) / data.get_resfactored_sum(np.ones(data.LUMASK.size))
        layers.append(('GBF2 priority degraded areas', 'ALL', 'ALL', (share_in_priority_area >= 0.5).astype(np.float64), option))
    catalogue = [
        {'theme': 'Biodiversity', 'input': 'Biodiversity quality', 'unit': 'score', 'kind': 'layer'},
        {'theme': 'Biodiversity', 'input': 'GBF2 priority degraded areas', 'unit': '0 / 1', 'kind': 'mask'},
    ]

    # The habitat contribution table is an input of its own in the report (no map: kind 'chart'), under the name it is written with
    catalogue.append({'theme': 'Biodiversity', 'input': 'Habitat contribution of a land use (HCAS)', 'unit': '0 - 1', 'kind': 'chart'})
    get_hcas_table(data).assign(Year=yr_cal).to_csv(os.path.join(path, f'inputs_tables_{yr_cal}.csv'), index=False)

    return input_theme('biodiversity', catalogue, layers)


# ── Social licence ──────────────────────────────────────────────────────────────────────────────────────────

def input_social_licence(data: Data, yr_cal, path):
    """The regional adoption caps of the year (settings.REGIONAL_ADOPTION_CONSTRAINTS), one row per (region, target):
    inputs_adoption_caps_<year>.csv [region, target, cap_percent, cap_ha, Year]. No map.

    individual_cap    a cap per (zone, land use), from regional_adoption_zones.xlsx, interpolated to the year
                      (data.get_regional_adoption_percent_by_year / get_regional_adoption_limit_ha_by_year).
    non_ag_sum_cap    one cap per region on the sum of all non-agricultural land uses
                      (data.get_regional_adoption_non_ag_sum_limit_ha_by_year).
    The percentages are of the region's area in the model; the hectares are what the solver's rows are built from."""
    mode = settings.REGIONAL_ADOPTION_CONSTRAINTS
    if mode == 'off':
        return f"Inputs (social licence) for year {yr_cal}: no regional adoption cap in this run"

    rows = []
    if mode == 'individual_cap':
        targets = data.REGIONAL_ADOPTION_TARGETS
        zone = settings.REGIONAL_ADOPTION_ZONE
        # a zone is listed by its name where the workbook has one (NRM_NAME, LGA_NAME10, ..., the last such column)
        name_columns = [column for column in targets.columns if 'NAME' in column.upper()]
        names = dict(zip(targets[zone], targets[name_columns[-1]])) if name_columns else {}
        percents = data.get_regional_adoption_percent_by_year(yr_cal)
        hectares = data.get_regional_adoption_limit_ha_by_year(yr_cal)
        rows = [
            {'region': str(names.get(region, region)), 'target': landuse, 'cap_percent': float(percent), 'cap_ha': float(limit_ha)}
            for (region, landuse, percent), (_, _, limit_ha) in zip(percents, hectares)
        ]
    elif mode == 'non_ag_sum_cap':
        real_area = data.REAL_AREA.values
        for region, region_cells, limit_ha in data.get_regional_adoption_non_ag_sum_limit_ha_by_year(yr_cal):
            region_ha = float(real_area[region_cells].sum())
            rows.append({'region': str(region), 'target': 'All non-agricultural land uses',
                         'cap_percent': float(limit_ha) / region_ha * 100 if region_ha > 0 else 0.0, 'cap_ha': float(limit_ha)})

    pd.DataFrame(rows, columns=['region', 'target', 'cap_percent', 'cap_ha']).assign(Year=yr_cal)\
        .to_csv(os.path.join(path, f'inputs_adoption_caps_{yr_cal}.csv'), index=False)
    return f"Inputs (social licence) written for year {yr_cal}: {len(rows)} caps"


# ── Renewable energy ────────────────────────────────────────────────────────────────────────────────────────

def input_renewable(data: Data, yr_cal, path):
    """The spatial layers of the renewable energy input (`data.RENEWABLE_LAYERS`), per technology: the capacity
    factor, the cost of install and of operation, and the distribution loss (one layer, the same for both
    technologies). The cost layers start in 2030: an earlier model year reads 2030, as the cost functions do.
    Only with a renewable type switched on (the layers are not loaded otherwise)."""
    if not any(settings.RENEWABLES_OPTIONS.values()):
        return f"Inputs (renewable) for year {yr_cal}: skipped, no renewable type is on"

    layer_year = max(yr_cal, int(data.RENEWABLE_LAYERS['year'].values[0]))
    renewable_layers = data.RENEWABLE_LAYERS.sel(year=layer_year)

    layers = []
    for technology in renewable_layers['tech_name'].values:
        technology_layers = renewable_layers.sel(tech_name=technology)
        capacity_r = np.nan_to_num(technology_layers['capacity_factor_multiplier'].values.astype(np.float64))
        install_r = np.nan_to_num(technology_layers['Cost_of_install_AUD_kw'].values.astype(np.float64))
        operation_r = np.nan_to_num(technology_layers['Cost_of_operation_AUD_kw'].values.astype(np.float64))
        layers += [
            ('Capacity factor', 'ALL', str(technology), capacity_r),
            ('Cost of install', 'ALL', str(technology), install_r),
            ('Cost of operation', 'ALL', str(technology), operation_r),
        ]
    loss_r = np.nan_to_num(renewable_layers['distribution_loss_factor_multiplier'].values.astype(np.float64))
    layers.append(('Distribution loss factor', 'ALL', 'ALL', loss_r))

    catalogue = [
        {'theme': 'Renewable energy', 'input': 'Capacity factor', 'unit': '0 - 1', 'kind': 'layer'},
        {'theme': 'Renewable energy', 'input': 'Cost of install', 'unit': 'AUD/kW', 'kind': 'layer'},
        {'theme': 'Renewable energy', 'input': 'Cost of operation', 'unit': 'AUD/kW', 'kind': 'layer'},
        {'theme': 'Renewable energy', 'input': 'Distribution loss factor', 'unit': '0 - 1', 'kind': 'layer'},
    ]
    return input_theme('renewable', catalogue, layers)


# ── Ag-management bundles ───────────────────────────────────────────────────────────────────────────────────

def input_ag_management(data: Data, yr_cal, path):
    """What every ag-management option changes on its land uses over the years (the bundle workbooks), as lines:
    inputs_bundles_<year>.csv, one row per (option, land-use group, variable, year). No map. The table holds every
    year of the workbooks, so it is the same file in every year's folder."""
    bundles = [
        ('Asparagopsis taxiformis', data.ASPARAGOPSIS_DATA),
        ('Ecological Grazing', data.ECOLOGICAL_GRAZING_DATA),
        ('Precision Agriculture', data.PRECISION_AGRICULTURE_DATA),
        ('AgTech EI', data.AGTECH_EI_DATA),
        ('Biochar', data.BIOCHAR_DATA),
    ]
    records = [
        table.select_dtypes('number')
        .rename_axis('Year').reset_index()
        .melt(id_vars='Year', var_name='variable', value_name='value')
        .dropna(subset=['value'])
        .assign(am=option, group=group)
        for option, groups in bundles for group, table in groups.items()
    ]
    pd.concat(records, ignore_index=True).to_csv(os.path.join(path, f'inputs_bundles_{yr_cal}.csv'), index=False)

    return f"Inputs (ag-management bundles) written for year {yr_cal}"



# ── The input cache: the inputs of every option, at one resolution, built once ────────────────────────────────
#
# The inputs do not come out of a run: they are what the model reads, whatever it then does with them. They are
# therefore built on their own -- at INPUT_CACHE_RESFACTOR whatever the resolution of the run, for the years
# INPUT_CACHE_YEARS, and for EVERY value of the options of INPUT_OPTIONS, not only the one a run picked -- into
# input/_cache/report_inputs_RES<n>_v<version>/, and every report copies them from there (ensure_input_cache). There
# is one cache, whatever the settings of a run: see get_input_cache_dir.

INPUT_CACHE_RESFACTOR = 5
INPUT_CACHE_YEARS = [2010, 2020, 2030, 2040, 2050]
INPUT_CACHE_VERSION = 1                     # raise it when the writers change, so every cache is built again

# The options every value of which is built: {setting: its values}. A layer is filed under the options it depends
# on (option_label); the others are built at the values the settings hold when the cache is built.
INPUT_OPTIONS = {
    'SSP': ['126', '245', '370', '585'],
    'CO2_FERT': ['on', 'off'],
    'USE_GHG_SCOPE_1': [True, False],
    'CARBON_EFFECTS_WINDOW': [50, 60, 70, 80, 90],
    'FIRE_RISK': ['low', 'med', 'high'],
    'BIO_QUALITY_LAYER': list(settings.BIO_QUALITY_LAYERS),
}

# The short name of an option in the label of a layer. The label goes into the name of the layer's file, and a
# report often sits deep in a folder tree: with the settings' own names a file path passed the 260 characters Windows
# allows, and the browser could not open the layer.
OPTION_SHORT_NAMES = {
    'SSP': 'S', 'CO2_FERT': 'C', 'USE_GHG_SCOPE_1': 'G', 'CARBON_EFFECTS_WINDOW': 'W',
    'FIRE_RISK': 'F', 'BIO_QUALITY_LAYER': 'Q',
}


def option_label(**values) -> str:
    """The name a layer is filed under for the option values it is computed with: 'S=245|C=on' (the settings
    by their short names, OPTION_SHORT_NAMES)."""
    return '|'.join(f'{OPTION_SHORT_NAMES[name]}={value}' for name, value in values.items())


def get_input_cache_dir() -> str:
    """The ONE folder of the input cache: input/_cache/report_inputs_RES<n>_v<version>. It is not built again for
    another set of settings: the inputs every option changes are all in it, and the others are those of the settings
    in force when it was built (listed in its DONE.json). Delete the folder, or raise INPUT_CACHE_VERSION, to build
    it again."""
    return os.path.abspath(os.path.join(settings.INPUT_DIR, '_cache', f'report_inputs_RES{INPUT_CACHE_RESFACTOR}_v{INPUT_CACHE_VERSION}'))


def clear_economics_caches():
    """Forget what the economics functions remember of a Data (functools.lru_cache), after an option has changed it."""
    for module in (ag_quantity, ag_revenue, ag_cost, ag_transitions, ag_ghg, ag_water,
                   non_ag_revenue, non_ag_cost, non_ag_transitions, non_ag_ghg, non_ag_water):
        for function in vars(module).values():
            if callable(getattr(function, 'cache_clear', None)):
                function.cache_clear()


def block_mean(data: Data, values_fullres) -> np.ndarray:
    """A full-resolution layer (one value per land cell) as one value per model cell: the mean over every land-use
    cell of the cell's block, Data.get_resfactored_sum of the layer over the count of those cells. For a layer that
    has a value on every land-use cell, of either sign (soil carbon and the carbon stock of natural land can be
    negative, a water yield can be zero): Data.get_resfactored_average_fraction counts only the cells above zero, so
    it would leave those out of the count while keeping them in the sum."""
    values = np.nan_to_num(np.asarray(values_fullres, dtype=np.float64))
    return (data.get_resfactored_sum(values) / data.get_resfactored_sum(np.ones(data.LUMASK.size))).astype(np.float32)


def block_mean_where_valued(data: Data, values_fullres) -> np.ndarray:
    """A full-resolution layer as one value per model cell: the mean over the land-use cells of the block that carry
    a value (> 0), 0 where none does. For a layer that only some cells have, and that is never negative (BECCS, the
    price of a water licence): Data.get_resfactored_average_fraction of the layer with the cells the model does not
    hold zeroed, so they count neither in the sum nor in the count."""
    values = np.nan_to_num(np.asarray(values_fullres, dtype=np.float64)) * data.LUMASK
    if (values < 0).any():
        raise ValueError('block_mean_where_valued: the layer has negative values, which the count of its cells (> 0) would leave out')
    return data.get_resfactored_average_fraction(values, use_valid_cell_count=True)


def use_block_mean_layers(data: Data):
    """Show the inputs as the value of the land a coarse cell stands for. At RESFACTOR > 1 Data.__init__ reads most
    layers at the centre cell of a model cell's block; where that cell is not a land-use cell (public land, water:
    one model cell in ten at RES5, along the edge of the model's land) the value is that of land the model does not
    hold. This reads those layers again at full resolution and gives each model cell the mean over its block's
    land-use cells (block_mean; block_mean_where_valued for a layer only some cells have). The agricultural tables are block means already (Data.coarsen_guarded_arrays), the
    climate and carbon options are set by set_climate_option / set_carbon_option. Nothing here is for a Data that is
    solved afterwards."""
    if settings.RESFACTOR == 1:
        return

    # The establishment cost of the plantings (as Data.__init__: riparian and agroforestry take the environmental one)
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, "plantings_est_cost_ha.nc")) as est_cost:
        data.EP_EST_COST_HA = data.EP_EST_COST_HA.copy(data=block_mean(data, est_cost['EP_EST_COST_HA'].values))
        data.CP_EST_COST_HA = data.CP_EST_COST_HA.copy(data=block_mean(data, est_cost['CP_EST_COST_HA'].values))
    data.RP_EST_COST_HA = data.EP_EST_COST_HA.copy()
    data.AF_EST_COST_HA = data.EP_EST_COST_HA.copy()

    # The price of a water licence (no price: 0, as Data.__init__ fills it)
    water_licence_price = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "water_licence_price.nc"))
    data.WATER_LICENCE_PRICE = data.WATER_LICENCE_PRICE.copy(data=block_mean_where_valued(data, water_licence_price.values).astype(data.WATER_LICENCE_PRICE.dtype))

    # The historical water yields; the yield of natural land is worked out cell by cell, then averaged
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, "water_yield_baselines.nc")) as baselines:
        hist_dr = baselines['WATER_YIELD_HIST_DR_ML_HA'].values.astype(np.float64)
        hist_sr = baselines['WATER_YIELD_HIST_SR_ML_HA'].values.astype(np.float64)
        deep_rooted = np.nan_to_num(baselines['DEEP_ROOTED_PROPORTION'].values.astype(np.float64))
    data.WATER_YIELD_HIST_DR = data.WATER_YIELD_HIST_DR.copy(data=block_mean(data, hist_dr))
    data.WATER_YIELD_HIST_SR = data.WATER_YIELD_HIST_SR.copy(data=block_mean(data, hist_sr))
    data.DEEP_ROOTED_PROPORTION = data.DEEP_ROOTED_PROPORTION.copy(data=block_mean(data, deep_rooted))
    data.WATER_YIELD_HIST_NL = data.WATER_YIELD_HIST_NL.copy(data=block_mean(data, hist_dr * deep_rooted + hist_sr * (1 - deep_rooted)))

    # Savanna burning: a cell is eligible where any of its land-use cells is, with the mean abatement of those cells
    # (a few are slightly negative, so the mean is their sum over their count)
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, 'cell_savanna_burning.nc')) as savburn:
        eligible = savburn['ELIGIBLE_AREA'].values.astype(np.float64)
        abatement = np.nan_to_num(savburn['AEA_TOTAL_TCO2E_HA'].values.astype(np.float64))
    eligible_cells = data.get_resfactored_sum(eligible)
    data.SAVBURN_ELIGIBLE = data.SAVBURN_ELIGIBLE.copy(data=(eligible_cells > 0).astype(data.SAVBURN_ELIGIBLE.dtype))
    data.SAVBURN_TOTAL_TCO2E_HA = data.SAVBURN_TOTAL_TCO2E_HA.copy(data=(data.get_resfactored_sum(abatement * eligible) / np.maximum(eligible_cells, 1)).astype(np.float32))

    # BECCS
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, 'cell_BECCS.nc')) as beccs:
        data.BECCS_COSTS_AUD_HA_YR = data.BECCS_COSTS_AUD_HA_YR.copy(data=block_mean_where_valued(data, beccs['BECCS_COSTS_AUD_HA_YR'].values))
        data.BECCS_REV_AUD_HA_YR = data.BECCS_REV_AUD_HA_YR.copy(data=block_mean_where_valued(data, beccs['BECCS_REV_AUD_HA_YR'].values))
        data.BECCS_TCO2E_HA_YR = data.BECCS_TCO2E_HA_YR.copy(data=block_mean_where_valued(data, beccs['BECCS_TCO2E_HA_YR'].values))
        data.BECCS_MWH_HA_YR = data.BECCS_MWH_HA_YR.copy(data=block_mean_where_valued(data, beccs['BECCS_MWH_HA_YR'].values))

    # The carbon of the plantings and of natural land, and the water yield of the climate: at the settings held
    set_carbon_option(data, settings.CARBON_EFFECTS_WINDOW, settings.FIRE_RISK)
    set_climate_option(data, settings.SSP, settings.CO2_FERT)


def set_climate_option(data: Data, ssp: str, co2_fert: str):
    """Put `data` under another climate: the SSP (so the RCP of its yield impacts) and CO2 fertilisation on or off. Loads again what
    Data.__init__ loaded for them, the same way: the climate change impact on yield, and the runoff file of the SSP."""
    rcp = 'rcp' + ssp[1] + 'p' + ssp[2]
    if rcp == 'rcp7p0':                                                        # as settings.py: SSP 370 reads the RCP 6.0 yield impacts
        rcp = 'rcp6p0'
    settings.SSP, settings.YIELD_RCP, settings.SCENARIO, settings.CO2_FERT = ssp, rcp, 'SSP' + ssp[0], co2_fert

    # as Data.__init__, "Climate change impact data"
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, "climate_change_impacts_" + settings.YIELD_RCP + "_CO2_FERT_" + settings.CO2_FERT.upper() + ".nc")) as cci_file:
        cci = cci_file['CLIMATE_CHANGE_IMPACT']
        data.CLIMATE_CHANGE_IMPACT = xr.DataArray(
            np.stack([cci.isel(column=i).values[data.MASK] for i in range(cci.sizes['column'])]),
            dims=['column', 'cell'],
            name='CLIMATE_CHANGE_IMPACT',
            coords={
                'lm': ('column', cci['lm'].values),
                'lu': ('column', cci['lu'].values),
                'year': ('column', cci['year'].values),
                'cell': np.arange(data.MASK.sum()),
            },
        ).set_xindex(['lm', 'lu', 'year'])

    # At RESFACTOR > 1 that is the value of each block's centre cell: the block means of the agricultural tables are
    # worked out again under this climate, as Data.__init__ does after reading them (Data.coarsen_guarded_arrays), so
    # that a map shows the value of the land a coarse cell stands for.
    if settings.RESFACTOR > 1:
        x_mrj_full = xr.load_dataarray(os.path.join(settings.INPUT_DIR, "x_mrj.nc"))
        data.LVSTK_COARSE = data.coarsen_guarded_arrays(x_mrj_full)

    # as Data.__init__, the water yield of the SSP (read year by year on demand: Data.get_water_yield)
    data.WATER_YIELD_SSP_FILE = os.path.join(settings.INPUT_DIR, f'water_yield_ssp{settings.SSP}_2010_2100_ml_ha.nc')
    data.WATER_YIELD_BY_YEAR = {}
    # At RESFACTOR > 1 the years of the cache are put in as block means (Data.get_water_yield would read each at the
    # block's centre cell); any other year is still read by Data.get_water_yield
    if settings.RESFACTOR > 1:
        with xr.open_dataset(data.WATER_YIELD_SSP_FILE) as water_yield:
            for yr_cal in INPUT_CACHE_YEARS:
                year = water_yield[['DR_ML_HA', 'SR_ML_HA']].sel(year=yr_cal).load()
                data.WATER_YIELD_BY_YEAR[yr_cal] = xr.Dataset(
                    {'DR_ML_HA': ('cell', block_mean(data, year['DR_ML_HA'].values)),
                     'SR_ML_HA': ('cell', block_mean(data, year['SR_ML_HA'].values))},
                    coords={'cell': np.arange(data.NCELLS)},
                )
    clear_economics_caches()


def set_carbon_option(data: Data, window: int, fire_risk: str):
    """Put `data` under another carbon accounting: the years the stock of a planting is averaged over, and the fire
    risk level. Works out again what Data.__init__ loaded for them, by the same formulas: the sequestration of the
    five plantings and the regrowth of natural land. Each is worked out for every full-resolution cell, then given to
    a model cell as the mean over its block's land-use cells (block_mean; at RESFACTOR 1 the cell's own value)."""
    settings.CARBON_EFFECTS_WINDOW, settings.FIRE_RISK = window, fire_risk
    cells = np.arange(data.NCELLS)

    with xr.open_dataset(os.path.join(settings.INPUT_DIR, "fire_risk.nc")) as fire_risk_ds:
        fire_risk_full = fire_risk_ds['FD_RISK'].sel(fire_risk=fire_risk, drop=True).load()

    # as Data.__init__: the five plantings, one after the other
    # Environmental plantings (block)
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_block.nc")) as ds:
        ds = ds.sel(age=window).load()
    ep_block = (
        (ds['EP_BLOCK_TREES_T_CO2_HA'] + ds['EP_BLOCK_DEBRIS_T_CO2_HA'])
        * (fire_risk_full / 100) * (1 - settings.RISK_OF_REVERSAL)
        + ds['EP_BLOCK_SOIL_T_CO2_HA']
    ) / window
    data.EP_BLOCK_AVG_T_CO2_HA_PER_YR = xr.DataArray(block_mean(data, ep_block.values), dims=['cell'], coords={'cell': cells})

    # Environmental plantings (belt)
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_belt.nc")) as ds:
        ds = ds.sel(age=window).load()
    ep_belt = (
        (ds['EP_BELT_TREES_T_CO2_HA'] + ds['EP_BELT_DEBRIS_T_CO2_HA'])
        * (fire_risk_full / 100) * (1 - settings.RISK_OF_REVERSAL)
        + ds['EP_BELT_SOIL_T_CO2_HA']
    ) / window
    data.EP_BELT_AVG_T_CO2_HA_PER_YR = xr.DataArray(block_mean(data, ep_belt.values), dims=['cell'], coords={'cell': cells})

    # Environmental plantings (riparian)
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_ep_rip.nc")) as ds:
        ds = ds.sel(age=window).load()
    ep_rip = (
        (ds['EP_RIP_TREES_T_CO2_HA'] + ds['EP_RIP_DEBRIS_T_CO2_HA'])
        * (fire_risk_full / 100) * (1 - settings.RISK_OF_REVERSAL)
        + ds['EP_RIP_SOIL_T_CO2_HA']
    ) / window
    data.EP_RIP_AVG_T_CO2_HA_PER_YR = xr.DataArray(block_mean(data, ep_rip.values), dims=['cell'], coords={'cell': cells})

    # Carbon plantings (block)
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_cp_block.nc")) as ds:
        ds = ds.sel(age=window).load()
    cp_block = (
        (ds['CP_BLOCK_TREES_T_CO2_HA'] + ds['CP_BLOCK_DEBRIS_T_CO2_HA'])
        * (fire_risk_full / 100) * (1 - settings.RISK_OF_REVERSAL)
        + ds['CP_BLOCK_SOIL_T_CO2_HA']
    ) / window
    data.CP_BLOCK_AVG_T_CO2_HA_PER_YR = xr.DataArray(block_mean(data, cp_block.values), dims=['cell'], coords={'cell': cells})

    # Farm forestry, i.e. carbon plantings (belt)
    with xr.open_dataset(os.path.join(settings.INPUT_DIR, "tCO2_ha_cp_belt.nc")) as ds:
        ds = ds.sel(age=window).load()
    cp_belt = (
        (ds['CP_BELT_TREES_T_CO2_HA'] + ds['CP_BELT_DEBRIS_T_CO2_HA'])
        * (fire_risk_full / 100) * (1 - settings.RISK_OF_REVERSAL)
        + ds['CP_BELT_SOIL_T_CO2_HA']
    ) / window
    data.CP_BELT_AVG_T_CO2_HA_PER_YR = xr.DataArray(block_mean(data, cp_belt.values), dims=['cell'], coords={'cell': cells})

    # The carbon stock of natural land (the one-off loss on clearing), and its regrowth per year
    nat_land_CO2 = xr.load_dataset(os.path.join(settings.INPUT_DIR, "natural_land_t_co2_ha.nc"))
    natural_stock = (
        nat_land_CO2['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA'] - (nat_land_CO2['NATURAL_LAND_AGB_DEBRIS_TCO2_HA'] * (100 - fire_risk_full) / 100)
    )
    data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA = xr.DataArray(block_mean(data, natural_stock.values), dims=['cell'], coords={'cell': cells})
    data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR = data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA / window
    clear_economics_caches()


def get_climate_options() -> list:
    """The (SSP, CO2 fertilisation) pairs the inputs hold a climate change impact file for. An SSP without one can not
    be loaded by the model either (Data.__init__ reads the file), so none of its inputs are built."""
    pairs = []
    for ssp in INPUT_OPTIONS['SSP']:
        for co2_fert in INPUT_OPTIONS['CO2_FERT']:
            rcp = f'rcp{ssp[1]}p{ssp[2]}'
            if rcp == 'rcp7p0':                                                # as settings.py: SSP 370 reads the RCP 6.0 yield impacts
                rcp = 'rcp6p0'
            file = f"climate_change_impacts_{rcp}_CO2_FERT_{co2_fert.upper()}.nc"
            if os.path.exists(os.path.join(settings.INPUT_DIR, file)):
                pairs.append((ssp, co2_fert))
            else:
                print(f'Inputs: SSP {ssp}, CO2 fertilisation {co2_fert} left out: no {file} in the inputs', flush=True)
    return pairs


def write_input_cache_layers(data: Data, raw_dir: str, years: list, writers=None):
    """Every input of every year of `years`, under every option it depends on, into `raw_dir`/out_<year> (one file
    pair per theme and year). `writers`: the writers to run, by name ('yield', 'water', 'ghg', ...; INPUT_THEMES),
    None for all of them: an option no wanted writer depends on is not set, so its data is not loaded. An option is set once and every year written under it: setting one loads data.
    At RESFACTOR > 1 the values are those of the land a coarse cell stands for: block means over its land-use cells.

    Climate (SSP, CO2 fertilisation)   the climate change impact on yield
    SSP                                the water inputs (runoff, its change, and the water requirement: livestock
                                       drinking water follows the stocking rate, so the climate), at the CO2
                                       fertilisation the settings hold
    Scope 1 only, or every source      the emissions of the agricultural land uses
    Carbon window, fire risk           the sequestration of the non-agricultural land uses
    Quality layer                      the biodiversity quality and the GBF2 mask it gives (input_biodiversity)
    Everything else is built at the values the settings hold when the cache is built."""
    held = {name: getattr(settings, name) for name in INPUT_OPTIONS}
    path_of = {yr_cal: os.path.join(raw_dir, f'out_{yr_cal}') for yr_cal in years}
    themes = {yr_cal: {} for yr_cal in years}                                  # year -> tag -> (catalogue, layers)
    wanted = lambda name: writers is None or name in writers

    def add(yr_cal, theme, option='ALL', keep=lambda lm: True):
        if not isinstance(theme, tuple):                                       # a writer with no theme (a table, or switched off)
            return
        tag, catalogue, layers = theme
        collected = themes[yr_cal].setdefault(tag, ([], []))
        collected[0].extend(catalogue)
        for name, lm, lu, values_r, *own_option in layers:
            if keep(lm):
                collected[1].append((name, lm, own_option[0] if own_option else option, lu, values_r))

    for ssp, co2_fert in get_climate_options() if wanted('yield') or wanted('water') else []:
        if not (wanted('yield') or co2_fert == held['CO2_FERT']):                # water alone: one CO2 setting
            continue
        set_climate_option(data, ssp, co2_fert)
        for yr_cal in years:
            if wanted('yield'):
                add(yr_cal, input_yield(data, yr_cal, path_of[yr_cal]), option_label(SSP=ssp, CO2_FERT=co2_fert))
            if wanted('water') and co2_fert == held['CO2_FERT']:
                add(yr_cal, input_water(data, yr_cal, path_of[yr_cal]), option_label(SSP=ssp))
        print(f'Inputs: climate SSP {ssp}, CO2 fertilisation {co2_fert} done', flush=True)
    set_climate_option(data, held['SSP'], held['CO2_FERT'])

    if wanted('ghg'):
        for scope_1 in INPUT_OPTIONS['USE_GHG_SCOPE_1']:
            settings.USE_GHG_SCOPE_1 = scope_1
            clear_economics_caches()
            for yr_cal in years:
                add(yr_cal, input_ghg(data, yr_cal, path_of[yr_cal]), option_label(USE_GHG_SCOPE_1=scope_1), keep=lambda lm: lm in data.LANDMANS)
        settings.USE_GHG_SCOPE_1 = held['USE_GHG_SCOPE_1']
        for window in INPUT_OPTIONS['CARBON_EFFECTS_WINDOW']:
            for fire_risk in INPUT_OPTIONS['FIRE_RISK']:
                set_carbon_option(data, window, fire_risk)
                for yr_cal in years:
                    add(yr_cal, input_ghg(data, yr_cal, path_of[yr_cal]), option_label(CARBON_EFFECTS_WINDOW=window, FIRE_RISK=fire_risk), keep=lambda lm: lm == 'NonAg')
        set_carbon_option(data, held['CARBON_EFFECTS_WINDOW'], held['FIRE_RISK'])
        print('Inputs: GHG options done', flush=True)

    messages = []
    for yr_cal in years:
        if wanted('ghg'):
            add(yr_cal, input_ghg(data, yr_cal, path_of[yr_cal]), keep=lambda lm: lm == 'ALL')    # savanna burning: no option
        for name, writer in (('economics', input_economics), ('transition_cost', input_transition_cost), ('exclusion', input_exclusion),
                             ('biodiversity', input_biodiversity), ('renewable', input_renewable),
                             ('ag_management', input_ag_management), ('social_licence', input_social_licence)):
            if wanted(name):
                add(yr_cal, writer(data, yr_cal, path_of[yr_cal]))
        messages += [save_input_layers(data, yr_cal, path_of[yr_cal], tag, catalogue, layers) for tag, (catalogue, layers) in themes[yr_cal].items()]
        themes[yr_cal] = None                                                  # written: let the arrays go
    return messages


# ── The species layers: one file per vegetation group / species / community ────────────────────────────────────
#
# The layers the biodiversity targets are counted over (GBF3 vegetation groups, GBF4 species and communities, GBF8
# species under climate change). There are thousands, so they are not part of the layer tree of the other inputs:
# they are kept SPECIES_PER_FILE to a file, data/map_layers/input_species/<set>/<variant>/<k>.js, in the order of the
# list of their set (data/Inputs_species.js), and the report loads the file of the one that is asked for.

INPUT_GBF8_YEARS = [2030, 2050]             # the years of the GBF8 rasters that are written (they hold 1990, 2030, 2050, 2070, 2090)


SPECIES_PER_FILE = 100                      # the items of a set that share a file (see write_species_files)


def species_leaf(values, high) -> dict:
    """One species layer as the report reads a map layer (create_report_layers.map2cells: one byte per cell, the
    value as a share of `high`)."""
    from luto.tools.report.create_report_layers import map2cells
    import base64

    _, cell_bytes, attrs = map2cells(xr.DataArray(np.asarray(values, dtype=np.float32), dims=('cell',)), False, (), (0.0, float(high)), None, None, per_ha=False)
    return {'cells_b64': base64.b64encode(cell_bytes).decode(), **attrs}


def write_species_files(save_dir, set_dir, variant, leaves):
    """The layers of one variant of a set, SPECIES_PER_FILE to a file, in the order of the set's list:
    data/map_layers/input_species/<set>/<variant>/<k>.js holds window["sp_<set>_<variant>_<k>"] = {position: leaf}
    for the items k * SPECIES_PER_FILE onwards. One file per item was tried first: the 21 000 files of GBF8 took
    longer to copy into a report than everything else took to build."""
    folder = os.path.join(save_dir, 'map_layers', 'input_species', set_dir, str(variant))
    os.makedirs(folder, exist_ok=True)
    for first in range(0, len(leaves), SPECIES_PER_FILE):
        number = first // SPECIES_PER_FILE
        with open(os.path.join(folder, f'{number}.js'), 'w') as file:
            file.write(f'window["sp_{set_dir}_{variant}_{number}"] = ')
            json.dump({position: leaves[position] for position in range(first, min(first + SPECIES_PER_FILE, len(leaves)))}, file, separators=(',', ':'))
            file.write(';\n')


def write_species_block_layers(data: Data, save_dir: str) -> dict:
    """The vegetation groups (GBF3) and the species and communities of national significance (GBF4), on the cells
    of `data`: the share of each cell the item covers, in percent. A cell of the cache is a block of full-resolution
    cells, and a small range would vanish if only the block's centre cell were read: the value is the mean over the
    block (Data.get_resfactored_average_fraction over all its cells), so a range that touches a block shows in it."""
    sets = {}
    sources = [
        ('GBF3 vegetation groups', 'GBF3', f'bio_GBF3_{settings.GBF3_NVIS_TARGET_CLASS}_sparse.nc', 'group', 1.0),       # stored as percent
        ('GBF4 species (SNES)', 'SNES', 'bio_GBF4_SNES_sparse.nc', 'species', 100.0),                                    # stored as a share
        ('GBF4 communities (ECNES)', 'ECNES', 'bio_GBF4_ECNES_sparse.nc', 'species', 100.0),
    ]
    for name, set_dir, file, item_dim, to_percent in sources:
        layer = data.read_sparse_layer(file)                                    # sparse over the full-resolution cells
        items = [str(item) for item in layer[item_dim].values]
        variants = [str(presence) for presence in layer['presence'].values] if 'presence' in layer.dims else ['ALL']
        for variant in variants:
            leaves = []
            for item in items:
                selected = layer.sel({item_dim: item, **({'presence': variant} if 'presence' in layer.dims else {})})
                full_res = np.asarray(selected.data.todense(), dtype=np.float32)
                leaves.append(species_leaf(data.get_resfactored_average_fraction(full_res, use_valid_cell_count=False) * to_percent, 100.0))
            write_species_files(save_dir, set_dir, variant, leaves)
        sets[name] = {'dir': set_dir, 'grid': 'map_input_cell_index', 'unit': '% of the cell', 'variants': variants,
                      'variant_name': 'Presence' if len(variants) > 1 else None, 'items': items, 'per_file': SPECIES_PER_FILE}
        print(f'Inputs (species): {name}: {len(items)} items x {len(variants)}', flush=True)
    return sets


def write_species_gbf8_layers(save_dir: str, limit: int | None = None) -> dict:
    """The climate suitability of every GBF8 species, as it is in the input: the rasters are read as they are, on
    their own grid (map_gbf8_cell_index.js: one cell per land pixel of the raster), for the SSP the settings hold and
    the years INPUT_GBF8_YEARS. `limit` keeps the first species only (to try it)."""
    from luto.tools.report.create_report_layers import write_cell_index
    import rioxarray  # noqa: F401  (the .rio accessor)

    raster_file = os.path.join(settings.INPUT_DIR, f'bio_GBF8_ssp{settings.SSP}_EnviroSuit.nc')
    group_file = os.path.join(settings.INPUT_DIR, f'bio_GBF8_ssp{settings.SSP}_EnviroSuit_group.nc')
    with xr.open_dataset(group_file) as groups:
        land = (groups['data'].max([dim for dim in groups['data'].dims if dim not in ('y', 'x')]) > 0).load()    # the pixels any group lives on

    # The grid: every land pixel numbered in raster order, drawn as the other layers are (EPSG:3857)
    numbers = np.full(land.shape, -1.0, dtype=np.float32)
    numbers[land.values] = np.arange(int(land.values.sum()), dtype=np.float32)
    numbered = xr.DataArray(numbers, dims=('y', 'x'), coords={'y': land['y'].values, 'x': land['x'].values})\
        .rio.write_nodata(-1.0).rio.write_crs('EPSG:4326').rio.reproject('EPSG:3857', nodata=-1.0)
    os.makedirs(os.path.join(save_dir, 'map_layers'), exist_ok=True)
    write_cell_index(np.round(numbered.values).astype(np.int32), {'transform': numbered.rio.transform()}, os.path.join(save_dir, 'map_layers'), name='map_gbf8_cell_index')

    with xr.open_dataset(raster_file) as raster:
        species = [str(name) for name in raster['species'].values][:limit]
        for year in INPUT_GBF8_YEARS:
            of_year = raster['data'].sel(year=year)
            write_species_files(save_dir, 'GBF8', year, [species_leaf(of_year.isel(species=position).values[land.values], 100.0) for position in range(len(species))])
            print(f'Inputs (species): GBF8 {year}: {len(species)} species', flush=True)
    return {'GBF8 species': {'dir': 'GBF8', 'grid': 'map_gbf8_cell_index', 'unit': 'climate suitability (0 - 100)',
                             'variants': [str(year) for year in INPUT_GBF8_YEARS], 'variant_name': 'Year', 'items': species,
                             'per_file': SPECIES_PER_FILE, 'note': f'SSP {settings.SSP}'}}


def write_species_inputs(data: Data, save_dir: str):
    """Every species layer (write_species_block_layers, write_species_gbf8_layers) and the list of them,
    data/Inputs_species.js: {set: {dir, grid, unit, variants, variant_name, items, per_file}}."""
    sets = {**write_species_block_layers(data, save_dir), **write_species_gbf8_layers(save_dir)}
    with open(os.path.join(save_dir, 'Inputs_species.js'), 'w') as file:
        file.write('window["Inputs_species"] = ')
        json.dump(sets, file, separators=(',', ':'))
        file.write(';\n')


# The themes of the Inputs view that can be rebuilt one by one (rebuild_input_cache_themes): the writers behind each,
# and the file of the report only that theme writes. Inputs_catalogue.js and Inputs_tables.js are keyed by theme
# and merged; the layers are merged input by input.
INPUT_THEMES = {
    'Yield':            {'writers': ['yield'],                        'files': ['Inputs_yield_trend.js']},
    'Water':            {'writers': ['water'],                        'files': []},
    'GHG':              {'writers': ['ghg'],                          'files': []},
    'Economics':        {'writers': ['economics', 'transition_cost'], 'files': ['Inputs_transition_matrix.js']},
    'Exclusion':        {'writers': ['exclusion'],                    'files': []},
    'Biodiversity':     {'writers': ['biodiversity'],                 'files': []},
    'Renewable energy': {'writers': ['renewable'],                    'files': []},
    'Ag management':    {'writers': ['ag_management'],                'files': ['Inputs_bundles.js']},
    'Social licence':   {'writers': ['social_licence'],               'files': ['Inputs_adoption_caps.js']},
}


def merge_input_themes(built_dir: str, cache_data_dir: str, themes: list):
    """The files of a rebuild of `themes` (in `built_dir`, laid out as the cache's `data`) into the cache: what the
    cache held for those themes is taken out first, the rest of the cache is left as it is."""
    from luto.tools.report.create_report_layers import safe_key

    def read_js(path):
        """(the text before the content, the content) of a window[...] = <json>; file, (None, {}) if it is not there"""
        if not os.path.exists(path):
            return None, {}
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        head = text[:text.index('=') + 1]
        return head, json.loads(text[len(head):].rstrip().rstrip(';'))

    def write_js(path, head, content, **layout):
        with open(path, 'w', encoding='utf-8') as file:
            file.write(head + ' ')
            json.dump(content, file, separators=(',', ':'), **layout)
            file.write(';\n')

    def combos_of(tree, path=()):
        """Every combo of an index tree: its branches are dicts, its last level a list"""
        if isinstance(tree, dict):
            for key, branch in tree.items():
                yield from combos_of(branch, path + (key,))
        else:
            for leaf in tree:
                yield path + (leaf,)

    # The catalogue and the tables: keyed by theme
    old_catalogue = read_js(os.path.join(cache_data_dir, 'Inputs_catalogue.js'))[1]
    for filename in ('Inputs_catalogue.js', 'Inputs_tables.js'):
        head, old = read_js(os.path.join(cache_data_dir, filename))
        new_head, new = read_js(os.path.join(built_dir, filename))
        for theme in themes:
            old.pop(theme, None)
            if theme in new:
                old[theme] = new[theme]
        write_js(os.path.join(cache_data_dir, filename), head or new_head, old)

    # The layers: every layer of the inputs of these themes out (files and index entries), the rebuilt ones in
    layer_dir = os.path.join(cache_data_dir, 'map_layers')
    head, index = read_js(os.path.join(layer_dir, 'map_in__index.js'))
    new_head, new_index = read_js(os.path.join(built_dir, 'map_layers', 'map_in__index.js'))
    stale = {entry['input'] for theme in themes for entry in old_catalogue.get(theme, [])} | set(new_index.get('tree', {}))
    for name in stale & set(index.get('tree', {})):
        for combo in combos_of(index['tree'].pop(name), (name,)):
            key = '__'.join(safe_key(part) for part in combo)
            index['combos'].pop(key, None)
            layer_file = os.path.join(layer_dir, f'map_in__{key}.js')
            if os.path.exists(layer_file):
                os.remove(layer_file)
    if new_index:
        index.setdefault('dims', new_index['dims'])
        index.setdefault('tree', {}).update(new_index['tree'])
        index.setdefault('combos', {}).update(new_index['combos'])
        for filename in os.listdir(os.path.join(built_dir, 'map_layers')):
            if filename != 'map_in__index.js':
                shutil.copy2(os.path.join(built_dir, 'map_layers', filename), os.path.join(layer_dir, filename))
    write_js(os.path.join(layer_dir, 'map_in__index.js'), head or new_head, index, indent=2)

    # The files only one theme writes
    for theme in themes:
        for filename in INPUT_THEMES[theme]['files']:
            if os.path.exists(os.path.join(built_dir, filename)):
                shutil.copy2(os.path.join(built_dir, filename), os.path.join(cache_data_dir, filename))


def build_input_cache(themes=None):
    """Build what the input cache lacks, and print where it is. Two parts, each marked done by a file of its own, so
    one can be built without the other: the inputs of every option (DONE.json) and the species layers
    (DONE_species.json).

    Meant for a process of its own, started by ensure_input_cache with settings.RESFACTOR set to INPUT_CACHE_RESFACTOR
    before anything else is imported: it loads the model data at that resolution and changes settings as it goes
    through the options. A part is built in a folder beside the cache and moved in when complete, so a part that
    is marked done is whole.

    `themes` (names of INPUT_THEMES): rebuild these themes only, in a cache that is already built, and merge them
    in (merge_input_themes); the rest of the cache, the species layers included, is left as it is."""
    from luto import tools
    from luto.tools.report import create_report_data, create_report_layers
    from luto.tools.report.data_tools import get_all_files

    cache_dir = get_input_cache_dir()
    need_inputs = not os.path.exists(os.path.join(cache_dir, 'DONE.json'))
    need_species = not os.path.exists(os.path.join(cache_dir, 'DONE_species.json'))
    if themes and need_inputs:
        themes = None                               # no cache to merge into: build it whole
    if themes:
        need_inputs, need_species = True, False
    if not (need_inputs or need_species):
        print(f'INPUT_CACHE_DIR={cache_dir}', flush=True)
        return cache_dir
    assert settings.RESFACTOR == INPUT_CACHE_RESFACTOR, 'build_input_cache: start it through ensure_input_cache'
    built_with = {name: repr(value) for name, value in sorted(vars(settings).items()) if name.isupper() and not callable(value)}

    # The inputs need none of the biodiversity targets' layers: left off, the data loads faster
    settings.SIM_YEARS = list(INPUT_CACHE_YEARS)
    settings.GBF3_NVIS_TARGET = settings.GBF4_TARGET_SNES = settings.GBF4_TARGET_ECNES = settings.GBF8_TARGET = 'off'
    data = Data()                               # at RESFACTOR > 1 its agricultural tables are block means (Data.coarsen_guarded_arrays): what the maps show
    use_block_mean_layers(data)                 # ... and the layers it read at the block's centre cell become block means too

    def move_in(building, marker, content):
        """The files of a finished part into the cache, then its marker"""
        shutil.copytree(os.path.join(building, 'data'), os.path.join(cache_dir, 'data'), dirs_exist_ok=True)
        shutil.rmtree(building, ignore_errors=True)
        with open(os.path.join(cache_dir, marker), 'w') as done:
            json.dump(content, done, indent=1, default=str)

    if need_inputs:
        building = f'{cache_dir}.building_{os.getpid()}'
        raw_dir, save_dir = os.path.join(building, 'raw'), os.path.join(building, 'data')
        template = xr.Dataset({'layer': tools.arr_to_xr(data, data.lumaps[data.YR_CAL_BASE].astype(np.float32))})
        for yr_cal in INPUT_CACHE_YEARS:
            path = os.path.join(raw_dir, f'out_{yr_cal}')
            os.makedirs(path, exist_ok=True)
            template.to_netcdf(os.path.join(path, f'xr_map_template_{yr_cal}.nc'))   # the grid the layers are drawn on
        writers = [writer for theme in themes for writer in INPUT_THEMES[theme]['writers']] if themes else None
        for message in write_input_cache_layers(data, raw_dir, INPUT_CACHE_YEARS, writers):
            print(message, flush=True)

        # The report's own files, made once: the layers (create_report_layers) and the tables (create_report_data)
        os.makedirs(os.path.join(save_dir, 'map_layers'), exist_ok=True)
        create_report_layers.save_input_layers(raw_dir, save_dir)
        print(create_report_data.process_inputs_data(get_all_files(raw_dir), save_dir), flush=True)
        shutil.rmtree(raw_dir)
        if themes:
            # These themes only: merged into the cache, whose marker notes when each theme was last rebuilt
            merge_input_themes(save_dir, os.path.join(cache_dir, 'data'), themes)
            shutil.rmtree(building, ignore_errors=True)
            with open(os.path.join(cache_dir, 'DONE.json'), 'r') as done:
                content = json.load(done)
            content.setdefault('themes_rebuilt', {}).update({theme: pd.Timestamp.now().isoformat(timespec='seconds') for theme in themes})
            with open(os.path.join(cache_dir, 'DONE.json'), 'w') as done:
                json.dump(content, done, indent=1, default=str)
            print(f'Themes rebuilt and merged into the cache: {", ".join(themes)}', flush=True)
        else:
            move_in(building, 'DONE.json', {'version': INPUT_CACHE_VERSION, 'resfactor': INPUT_CACHE_RESFACTOR, 'years': INPUT_CACHE_YEARS,
                                            'ncells': int(data.NCELLS), 'options': INPUT_OPTIONS, 'built_with': built_with})

    if need_species:
        building = f'{cache_dir}.building_species_{os.getpid()}'
        write_species_inputs(data, os.path.join(building, 'data'))
        move_in(building, 'DONE_species.json', {'version': INPUT_CACHE_VERSION, 'gbf8_ssp': built_with['SSP'], 'gbf8_years': INPUT_GBF8_YEARS})

    print(f'INPUT_CACHE_DIR={cache_dir}', flush=True)
    return cache_dir


def rebuild_input_cache_themes(themes: list) -> str:
    """Rebuild some themes of the input cache (names of INPUT_THEMES, e.g. ['Biodiversity']) and leave the rest as it
    is: for a change to the input files or the writer of one theme. About 7 minutes for a theme without options
    (the model data is loaded either way), against 20 for the whole cache. Reports already built keep the inputs
    they copied.

        python -c "from luto.tools.write_input import rebuild_input_cache_themes; rebuild_input_cache_themes(['Biodiversity'])"
    """
    unknown = [theme for theme in themes if theme not in INPUT_THEMES]
    if unknown or not themes:
        raise ValueError(f'Unknown input themes {unknown}: choose from {list(INPUT_THEMES)}')
    return ensure_input_cache(themes)


def ensure_input_cache(themes=None) -> str:
    """The folder of the input cache (its `data` is what a report copies), built first if it is not there: in a
    process of its own, because the build loads the model data at another resolution. What the build prints goes
    to <the folder>.log, to follow it while it runs (a quarter of an hour or more). `themes`: rebuild these themes
    even though the cache is there (rebuild_input_cache_themes)."""
    cache_dir = get_input_cache_dir()
    markers = [os.path.join(cache_dir, marker) for marker in ('DONE.json', 'DONE_species.json')]
    if all(os.path.exists(marker) for marker in markers) and not themes:
        return cache_dir

    code = (f'import luto.settings as settings; settings.RESFACTOR = {INPUT_CACHE_RESFACTOR}; '
            f'import luto.tools.write_input as write_input; write_input.build_input_cache({themes!r})')
    package_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    with open(cache_dir + '.log', 'a') as log:
        built = subprocess.run([sys.executable, '-u', '-c', code], cwd=package_parent, stdout=log, stderr=subprocess.STDOUT)
    if built.returncode != 0 or not all(os.path.exists(marker) for marker in markers):
        raise RuntimeError(f'The input cache could not be built: see {cache_dir}.log')
    return cache_dir
