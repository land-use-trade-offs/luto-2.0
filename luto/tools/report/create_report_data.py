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
import shutil
import json
import pandas as pd
import numpy as np
from luto import settings
from joblib import Parallel, delayed

from luto.economics.off_land_commodity import get_demand_df
from luto.tools.report.data_tools import get_all_files
from luto.tools.report.data_tools.parameters import (
    AG_LANDUSE,
    COLORS,
    COMMODITIES_ALL,
    COMMIDOTY_GROUP,
    GHG_CATEGORY,
    GHG_NAMES,
    GROUP_LU,
    LANDUSE_ALL_RENAMED,
    RENAME_AM_NON_AG,
    RENAME_NON_AG,
)


# Helper functions
def _read_concat(paths, ignore_index=True):
    """Read every CSV in `paths` and concat, skipping empty files. If ALL are empty, return a
    single empty frame carrying the first path's columns (schema) so downstream .assign/.replace/
    column ops do not crash. Replaces the fragile `pd.concat([... if not df.empty])` idiom, which
    raised "No objects to concatenate" when a category was empty in every year (e.g. a scenario
    with zero non-ag revenue)."""
    frames = [df for _p in paths if not (df := pd.read_csv(_p, engine='pyarrow')).empty]
    if frames:
        return pd.concat(frames, ignore_index=ignore_index)
    paths = list(paths)
    return pd.read_csv(paths[0], engine='pyarrow') if paths else pd.DataFrame()


def get_rank_color(x):
    """Get rank color based on value."""
    if x in [None, np.nan, 'N.A.']:
        return COLORS['N.A.']
    elif x <= 10:
        return COLORS['1-10']
    elif x <= 20:
        return COLORS['11-20']
    else:
        return COLORS['>=21']


def format_with_suffix(x):
    """Format number with suffix (K, M, B, T)."""
    if pd.isna(x) or x == 0:
        return "0"
    suffixes = ['', 'K', 'M', 'B', 'T']
    # Determine the appropriate suffix
    magnitude = 0
    while abs(x) >= 1000 and magnitude < len(suffixes)-1:
        magnitude += 1
        x /= 1000.0
    # Format with 2 significant digits
    if abs(x) < 100:
        formatted = f"{x:.2f}"
    else:
        formatted = f"{int(round(x))}"
    return f"{formatted} {suffixes[magnitude]}"


def annualise_points(years, values, sim_years=None) -> list:
    """Linearly interpolate `values` to every integer year between min and max of `sim_years`.

    `sim_years` is the full set of years actually simulated (e.g. [2020, 2025, ..., 2050]).
    `years`/`values` may be missing some `sim_years` (e.g. a land-use with zero area in 2020
    is dropped from that year's CSV) -- those years are treated as a real value of 0 before
    interpolating, so the series ramps from 0 rather than appearing abruptly. If `sim_years`
    is not given, it defaults to the unique values of `years`.

    Returns a list of {x, y[, opacity]} points (Highcharts series.data format). Years not
    present in `sim_years` are flagged with `opacity: 0.5` so the report can render
    interpolated (5-year-gap-filled) points at reduced opacity without affecting the
    underlying 'x' (year) values.
    """
    years = np.asarray(years)
    values = np.asarray(values, dtype=float)

    if sim_years is None:
        sim_years = np.unique(years)
    else:
        sim_years = np.asarray(sim_years)

    sim_values = pd.Series(values, index=years).groupby(level=0).sum().reindex(sim_years, fill_value=0.0).to_numpy()

    full_years = np.arange(sim_years.min(), sim_years.max() + 1)
    full_values = np.interp(full_years, sim_years, sim_values)
    is_interp = ~np.isin(full_years, sim_years)

    return [
        {'x': int(yr), 'y': val} if not interp else {'x': int(yr), 'y': val, 'opacity': 0.5}
        for yr, val, interp in zip(full_years.tolist(), full_values.tolist(), is_interp.tolist())
    ]


def groupby_to_records(df: pd.DataFrame, group_cols, out_cols, value_cols=('Year', 'Value (%)')):
    """Group `df` by `group_cols`, collect `value_cols` rows into a `data` column of
    annualised {x, y[, opacity]} points (see `annualise_points`).

    Returns a DataFrame with columns = `out_cols` (the last entry conventionally 'data').
    Robust to empty `df`: pandas 2.x returns a 2D DataFrame from `.apply` on empty
    groupby with a column subset, which breaks the usual `df_wide.columns = [...]` rename.
    """
    if df.empty:
        return pd.DataFrame(columns=list(out_cols))

    year_col, val_col = value_cols
    sim_years = np.sort(df[year_col].unique())
    s = df.groupby(list(group_cols))[list(value_cols)].apply(
        lambda x: annualise_points(x[year_col], x[val_col], sim_years)
    )
    wide = s.reset_index()
    wide.columns = list(out_cols)
    return wide



def bio_outside_series(bio_df: pd.DataFrame, cat: str, value_col: str = 'Value (%)') -> pd.DataFrame:
    """Build an "Outside LUTO study area" series df_wide for biodiversity per-Category charts.

    Outside rows in the underlying CSV are replicated across every (am, lm) combination
    with identical values, so we de-duplicate by selecting one canonical slice per chart:

      - cat='Ag':    one row per (region_level, region, species, Water_supply); pick `Agricultural Management == 'ALL'`.
      - cat='Am':    one row per (region_level, region, species, Water_supply, Agricultural Management); use all am values.
      - cat='NonAg': one row per (region_level, region, species); pick the (am=='ALL', lm=='ALL') aggregate.

    Returns a df_wide with the same columns the caller expects (including 'type', 'color', 'name').
    Returns an empty DataFrame when no outside rows exist (e.g. AUSTRALIA-mode CSVs).
    Pass value_col='Area Weighted Score (ha)' to get area-mode outside series.
    """
    outside = bio_df.query('Type == "Outside LUTO study area"')
    if outside.empty:
        return pd.DataFrame()

    if cat == 'Ag':
        sub = outside.query('`Agricultural Management` == "ALL"')
        df_wide = groupby_to_records(
            sub, ['region_level', 'region', 'species', 'Water_supply'], ['region_level', 'region', 'species', 'water', 'data'],
            value_cols=('Year', value_col),
        )
    elif cat == 'Am':
        df_wide = groupby_to_records(
            outside, ['region_level', 'region', 'species', 'Water_supply', 'Agricultural Management'],
            ['region_level', 'region', 'species', 'water', 'am', 'data'],
            value_cols=('Year', value_col),
        )
    elif cat == 'NonAg':
        sub = outside.query('`Agricultural Management` == "ALL" and Water_supply == "ALL"')
        df_wide = groupby_to_records(
            sub, ['region_level', 'region', 'species'], ['region_level', 'region', 'species', 'data'],
            value_cols=('Year', value_col),
        )
    else:
        return pd.DataFrame()

    if df_wide.empty:
        return df_wide
    df_wide['name'] = 'Outside LUTO study area'
    df_wide['type'] = 'column'
    df_wide['color'] = COLORS.get('Outside LUTO study area', '#E8E8E8')
    return df_wide


def build_out_dict_bulk(df_wide_pct, df_wide_area, key_cols):
    """Build a nested out_dict from df_wide_pct and df_wide_area using bulk column zipping.

    Replaces the O(N²) pattern of groupby-then-boolean-filter-per-group with a single
    pass over each full DataFrame, grouping rows in Python via defaultdict. ~100-160x
    faster than the original loop for large species × region × am combinations.

    The returned dict is nested in the order of key_cols, with leaf dicts:
        {'Percent': [...records...], 'Area': [...records...]}
    """
    from collections import defaultdict

    def _df_to_keyed(df):
        keys = list(zip(*[df[c].tolist() for c in key_cols]))
        leaf_cols = [c for c in df.columns if c not in key_cols]
        rows = [dict(zip(leaf_cols, r)) for r in zip(*[df[c].tolist() for c in leaf_cols])]
        grouped = defaultdict(list)
        for k, row in zip(keys, rows):
            grouped[k].append(row)
        return grouped

    pct_grouped  = _df_to_keyed(df_wide_pct)
    area_grouped = _df_to_keyed(df_wide_area)

    out_dict = {}
    for key, pct_list in pct_grouped.items():
        d = out_dict
        for k in key[:-1]:
            d = d.setdefault(k, {})
        d[key[-1]] = {'Percent': pct_list, 'Area': area_grouped.get(key, [])}
    return out_dict


def _paged_species_order(bio_paths: pd.DataFrame, chunks_base: str) -> list | None:
    """Species order for a paged metric, read from the manifest that cuts the map pages.

    `create_report_layers.get_map2json_paged` derives each map page's [start, end] from
    `manifest.json` in the *earliest* year's chunks dir, and the Vue builds its chart
    filenames from those same bounds. So the chart pages have to be cut on this list.

    Deriving the order from the score CSVs instead does not work: the CSVs drop rows whose
    area-weighted score is zero, while the manifest keeps every species that was batched.
    Any species in one but not the other shifts every later page boundary, and the Vue then
    requests a chart page file that was never written.

    Returns None when no manifest is found, so callers can fall back to the CSV-derived
    order (which is still correct whenever the two sets happen to coincide).
    """
    if bio_paths.empty:
        return None
    for _, row in bio_paths.sort_values('Year').iterrows():
        out_dir = os.path.dirname(row['path'])
        manifest_path = os.path.join(out_dir, f"{chunks_base}_{row['Year']}_chunks", 'manifest.json')
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        return [sp for key in sorted(manifest, key=int) for sp in manifest[key]]
    return None


def _write_paged_chart_js(
    out_dict: dict,
    filename_prefix: str,
    save_dir: str,
    page_size: int = 100,
    species_order: list | None = None,
) -> None:
    """Write out_dict split by species into page-sized JS files.

    out_dict must have structure: {region_level: {region: {species: <any>}}}
    Writes {filename_prefix}_{start}_{end}.js for each page of species.
    The Vue loads only the current page and aliases window[filename_prefix] to it.

    species_order: if provided, use this sorted list to define page boundaries so
    all chart files for the same metric share identical {start}_{end} ranges with
    the map-layer index (which uses the full species universe).  If None, the order
    is derived from out_dict itself.
    """
    if species_order is not None:
        all_species = list(species_order)
    else:
        all_species = sorted({
            sp
            for rl_data in out_dict.values()
            for r_data in rl_data.values()
            for sp in r_data
        })
    for page_start in range(0, max(len(all_species), 1), page_size):
        page_sp = all_species[page_start:page_start + page_size]
        if not page_sp:
            break
        page_end = page_start + len(page_sp)
        page_dict = {}
        for rl, rl_data in out_dict.items():
            page_dict[rl] = {}
            for region, r_data in rl_data.items():
                sliced = {sp: r_data[sp] for sp in page_sp if sp in r_data}
                if sliced:
                    page_dict[rl][region] = sliced
        fname = f'{filename_prefix}_{page_start}_{page_end}'
        with open(f'{save_dir}/{fname}.js', 'w') as f:
            f.write(f'window["{fname}"] = ')
            json.dump(page_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
    # Also write a flat combined file so Home.js can load all species in one request.
    with open(f'{save_dir}/{filename_prefix}.js', 'w') as f:
        f.write(f'window["{filename_prefix}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


def save_report_data(raw_data_dir:str):
    """
    Saves the report data in the specified directory.
    """
    # Set the save directory
    SAVE_DIR = f'{raw_data_dir}/DATA_REPORT/data'
    years = sorted(settings.SIM_YEARS)

    # Create the directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Move the GBF2 mask GeoJSON written by write_data into the report geo dir
    gbf2_src = os.path.join(raw_data_dir, 'biodiversity_GBF2_mask.js')
    gbf2_dst = os.path.join(SAVE_DIR, 'geo', 'biodiversity_GBF2_mask.js')
    if os.path.exists(gbf2_src) and not os.path.exists(gbf2_dst):
        os.makedirs(os.path.dirname(gbf2_dst), exist_ok=True)
        shutil.move(gbf2_src, gbf2_dst)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(raw_data_dir).reset_index(drop=True)
    files['Year'] = files['Year'].astype(int)
    files = files.query('Year.isin(@years)')

    # Land-use group mapping: {land_use: category}
    lu_group_map = {lu: cat for cat, lus in GROUP_LU.items() for lu in lus}

    # Create jobs for parallel execution
    jobs = [
        delayed(process_area_data)(files, SAVE_DIR, lu_group_map),
        delayed(process_production_data)(files, SAVE_DIR, years),
        delayed(process_economics_data)(files, SAVE_DIR),
        delayed(process_renewable_data)(files, SAVE_DIR, years),
        delayed(process_ghg_data)(files, SAVE_DIR, lu_group_map, years),
        delayed(process_water_data)(files, SAVE_DIR),
        delayed(process_transition_data)(files, SAVE_DIR),
        delayed(process_biodiversity_data)(files, SAVE_DIR),
        delayed(process_supporting_info_data)(SAVE_DIR, years, raw_data_dir),
    ]

    # Execute jobs in parallel
    num_jobs = len(jobs)
    for i, out in enumerate(Parallel(n_jobs=num_jobs, return_as='generator_unordered')(jobs)):
        print(f"│   ├── {out}") if i < num_jobs - 1 else print(f"│   └── {out}")




def process_area_data(files, SAVE_DIR, lu_group_map):
    """Process and save area change data (Section 1)."""
    area_dvar_paths = files.query('category == "area"').reset_index(drop=True)

    ag_dvar_dfs = area_dvar_paths.query('base_name == "area_agricultural_landuse"').reset_index(drop=True)
    ag_dvar_area = pd.concat([pd.read_csv(path) for path in ag_dvar_dfs['path']], ignore_index=True)
    ag_dvar_area['Source'] = 'Agricultural Land-use'
    ag_dvar_area['Category'] = ag_dvar_area['Land-use'].map(lu_group_map)
    ag_dvar_area['Area (ha)'] = ag_dvar_area['Area (ha)'].round(2)

    non_ag_dvar_dfs = area_dvar_paths.query('base_name == "area_non_agricultural_landuse"').reset_index(drop=True)
    non_ag_dvar_area = _read_concat(non_ag_dvar_dfs['path'])
    non_ag_dvar_area['Land-use'] = non_ag_dvar_area['Land-use'].replace(RENAME_NON_AG).infer_objects(copy=False)
    non_ag_dvar_area['Category'] = non_ag_dvar_area['Land-use'].map(lu_group_map)
    non_ag_dvar_area['Source'] = 'Non-Agricultural Land-use'
    non_ag_dvar_area['Area (ha)'] = non_ag_dvar_area['Area (ha)'].round(2)

    am_dvar_dfs = area_dvar_paths.query('base_name == "area_agricultural_management"').reset_index(drop=True)
    am_dvar_area = _read_concat(am_dvar_dfs['path'])
    am_dvar_area = am_dvar_area.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    am_dvar_area['Source'] = 'Agricultural Management'
    am_dvar_area['Area (ha)'] = am_dvar_area['Area (ha)'].round(2)


    # -------------------- Area ranking --------------------
    area_ranking_raw = pd.concat([
        ag_dvar_area.query('Water_supply != "ALL"'),
        non_ag_dvar_area,
        am_dvar_area.query('Water_supply != "ALL" and `Land-use` != "ALL"'),
    ])

    area_ranking_type = area_ranking_raw\
        .groupby(['Year', 'region_level', 'region', 'Source'])[['Area (ha)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'region_level', 'Source', 'Area (ha)'], ascending=[True, True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'region_level', 'Source']).cumcount())\
        .round({'Area (ha)': 2})

    area_ranking_total = area_ranking_raw\
        .groupby(['Year', 'region_level', 'region'])[["Area (ha)"]]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'region_level', 'Area (ha)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount(), Source='Total')\
        .round({'Area (ha)': 2})

    area_ranking = pd.concat([area_ranking_type, area_ranking_total], ignore_index=True)\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))


    out_dict = {}
    for (region_level, region, source), df in area_ranking.groupby(['region_level', 'region', 'Source']):
        df = df.drop(['region_level', 'region'], axis=1)

        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if source not in out_dict[region_level][region]:
            out_dict[region_level][region][source] = {}

        out_dict[region_level][region][source]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][source]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][source]['value'] = df.set_index('Year')['Area (ha)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Area_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Area overview --------------------

    area_df = pd.concat([
        ag_dvar_area.query('Water_supply != "ALL"'),
        non_ag_dvar_area,
        am_dvar_area.query('Water_supply != "ALL" and `Land-use` != "ALL"').assign(**{'Land-use':'Agricultural Management', 'Category':'Agricultural Management'}),
        ], ignore_index=True)

    group_cols = ['Land-use', 'Category', 'Source']
    for idx, col in enumerate(group_cols):

        df_region = area_df\
            .groupby(['Year', 'region_level', 'region', col])[['Area (ha)']]\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_wide = groupby_to_records(df_region, ['region_level', col, 'region'], ['region_level', 'name', 'region','data'], value_cols=('Year', 'Area (ha)'))
        df_wide['type'] = 'column'

        if col == "Land-use":
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        elif col == 'Category':
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
        elif col == 'Source':
            df_wide['name_order'] = df_wide['name'].apply(lambda x: ['Agricultural Management', 'Agricultural Land-use', 'Non-Agricultural Land-use'].index(x))
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

        out_dict = {}
        for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
            df = df.drop(['region_level', 'region'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            out_dict[region_level][region] = df.to_dict(orient='records')

        filename = f'Area_overview_{idx+1}_{col.replace(" ", "_")}'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')



    # -------------------- Area by Agricultural land --------------------
    df_wide = groupby_to_records(ag_dvar_area, ['region_level', 'region', 'Water_supply', 'Land-use'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Area (ha)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region_level, region, water), df in df_wide.groupby(['region_level', 'region', 'water']):
        df = df.drop(['region_level', 'region', 'water'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if water not in out_dict[region_level][region]:
            out_dict[region_level][region][water] = []
        out_dict[region_level][region][water] = df.to_dict(orient='records')

    filename = 'Area_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Area by Agricultural Management Area (ha) Land use --------------------
    df_wide = groupby_to_records(am_dvar_area .query('Type != "ALL"'), ['region_level', 'region', 'Water_supply', 'Land-use', 'Type'], ['region_level', 'region', 'water', 'landuse', 'name', 'data'], value_cols=('Year', 'Area (ha)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])


    out_dict = {}
    for (region_level, region, water, landuse), df in df_wide.groupby(['region_level', 'region', 'water', 'landuse']):
        df = df.drop(['region_level', 'region', 'water', 'landuse'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if water not in out_dict[region_level][region]:
            out_dict[region_level][region][water] = {}

        out_dict[region_level][region][water][landuse] = df.to_dict(orient='records')

    filename = f'Area_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Area by Non-Agricultural Land-use --------------------
    df_wide = groupby_to_records(non_ag_dvar_area, ['region_level', 'region', 'Land-use'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Area (ha)'))
    df_wide['type'] = 'column'

    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    filename = 'Area_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    return "Area data processing completed"



def process_economics_data(files, SAVE_DIR):
    
    # -------------------- Get the revenue and cost data --------------------
    revenue_ag_df = files.query('base_name == "economics_ag_revenue"').reset_index(drop=True)
    revenue_ag_df = pd.concat([pd.read_csv(path) for path in revenue_ag_df['path']], ignore_index=True)
    revenue_ag_df = revenue_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Agricultural Land-use (revenue)')

    cost_ag_df = files.query('base_name == "economics_ag_cost"').reset_index(drop=True)
    cost_ag_df = pd.concat([pd.read_csv(path) for path in cost_ag_df['path']], ignore_index=True)
    cost_ag_df = cost_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Agricultural Land-use (cost)')
    cost_ag_df['Value ($)'] = cost_ag_df['Value ($)'] * -1          # Convert cost to negative value

    revenue_am_df = files.query('base_name == "economics_am_revenue"').reset_index(drop=True)
    revenue_am_df = _read_concat(revenue_am_df['path'])
    revenue_am_df = revenue_am_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Agricultural Management (revenue)')

    cost_am_df = files.query('base_name == "economics_am_cost"').reset_index(drop=True)
    cost_am_df = _read_concat(cost_am_df['path'])
    cost_am_df = cost_am_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Agricultural Management (cost)')
    cost_am_df['Value ($)'] = cost_am_df['Value ($)'] * -1          # Convert cost to negative value

    revenue_non_ag_df = files.query('base_name == "economics_non_ag_revenue"').reset_index(drop=True)
    revenue_non_ag_df = _read_concat(revenue_non_ag_df['path'])
    revenue_non_ag_df = revenue_non_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Non-Agricultural Land-use (revenue)')

    cost_non_ag_df = files.query('base_name == "economics_non_ag_cost"').reset_index(drop=True)
    cost_non_ag_df = _read_concat(cost_non_ag_df['path'])
    cost_non_ag_df = cost_non_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Non-Agricultural Land-use (cost)')
    cost_non_ag_df['Value ($)'] = cost_non_ag_df['Value ($)'] * -1  # Convert cost to negative value

    cost_transition_ag2ag_df = files.query('base_name == "transition_ag2ag_cost"').reset_index(drop=True)
    cost_transition_ag2ag_df = _read_concat(cost_transition_ag2ag_df['path'])
    cost_transition_ag2ag_df = cost_transition_ag2ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Transition cost (Ag2Ag)')
    cost_transition_ag2ag_df['Value ($)'] = cost_transition_ag2ag_df['Cost ($)']  * -1          # Convert cost to negative value

    cost_transition_ag2non_ag_df = files.query('base_name == "transition_ag2nonag_cost"').reset_index(drop=True)
    cost_transition_ag2non_ag_df = _read_concat(cost_transition_ag2non_ag_df['path'])
    cost_transition_ag2non_ag_df = cost_transition_ag2non_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Transition cost (Ag2Non-Ag)')
    cost_transition_ag2non_ag_df['Value ($)'] = cost_transition_ag2non_ag_df['Cost ($)'] * -1   # Convert cost to negative value

    cost_transition_non_ag2ag_df = files.query('base_name == "transition_nonag2ag_cost"').reset_index(drop=True)
    cost_transition_non_ag2ag_df = _read_concat(cost_transition_non_ag2ag_df['path'])
    cost_transition_non_ag2ag_df = cost_transition_non_ag2ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Transition cost (Non-Ag2Ag)').dropna(subset=['Cost ($)'])
    cost_transition_non_ag2ag_df['Value ($)'] = cost_transition_non_ag2ag_df['Cost ($)'] * -1   # Convert cost to negative value

    order = [
        'Agricultural Land-use (revenue)',
        'Agricultural Management (revenue)',
        'Non-Agricultural Land-use (revenue)',
        'Agricultural Land-use (cost)',
        'Agricultural Management (cost)',
        'Non-Agricultural Land-use (cost)',
        'Transition cost (Ag2Ag)',
        'Transition cost (Ag2Non-Ag)',
        'Transition cost (Non-Ag2Ag)',
        'Profit'
    ]


    # -------------------- Economic ranking --------------------
    revenue_df = pd.concat([revenue_ag_df.query('Water_supply != "ALL" and Type != "ALL" and `Land-use` != "ALL"'), revenue_am_df.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Management Type` != "ALL"'), revenue_non_ag_df]
        ).groupby(['Year', 'region_level', 'region']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).sort_values(['Year', 'region_level', 'Value ($)'], ascending=[True, True, False]
        ).assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount()
        ).assign(Source='Revenue')

    cost_df = pd.concat(
        [
            cost_ag_df.query('Water_supply != "ALL" and Type != "ALL" and `Land-use` != "ALL"'),
            cost_am_df.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Management Type` != "ALL"'),
            cost_non_ag_df,
            cost_transition_ag2ag_df.query('`From-land-use` != "ALL" and `To-land-use` != "ALL" and `Cost-type` != "ALL"'),
            cost_transition_ag2non_ag_df.query('`From-land-use` != "ALL" and `To-land-use` != "ALL" and `Cost-type` != "ALL"'),
            cost_transition_non_ag2ag_df.query('`From-land-use` != "ALL" and `To-land-use` != "ALL" and `Cost-type` != "ALL"')
        ]
        ).groupby(['Year', 'region_level', 'region']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).assign(**{'Value ($)': lambda x: abs(x['Value ($)'])}
        ).sort_values(['Year', 'region_level', 'Value ($)'], ascending=[True, True, False]
        ).assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount()
        ).assign(Source='Cost')
        
    profit_df = revenue_df.merge(
        cost_df, on=['Year', 'region_level', 'region'], suffixes=('_revenue', '_cost')
        ).assign(**{'Value ($)': lambda x: x['Value ($)_revenue'] - x['Value ($)_cost']}
        ).drop(columns=['Value ($)_revenue', 'Value ($)_cost']
        ).sort_values(['Year', 'region_level', 'Value ($)'], ascending=[True, True, False]
        ).assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount()
        ).assign(Source='Profit')

    ranking_df = pd.concat([revenue_df, cost_df, profit_df]).assign(color= lambda x: x['Rank'].map(get_rank_color))


    out_dict = {}
    for (region_level, region, source), df in ranking_df.groupby(['region_level', 'region', 'Source']):
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if not source in out_dict[region_level][region]:
            out_dict[region_level][region][source] = {}

        df = df.drop(columns=['region_level', 'region'])
        out_dict[region_level][region][source]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][source]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][source]['value'] = df.set_index('Year')['Value ($)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Economics_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Economy overview --------------------

    # Overview: sum of revenue, cost, and profit by region
    rev_cost_net_region = pd.concat(
            [
                revenue_ag_df.query('Water_supply != "ALL" and Type != "ALL" and `Land-use` != "ALL"'),
                revenue_am_df.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Management Type` != "ALL"'),
                revenue_non_ag_df,
                cost_ag_df.query('Water_supply != "ALL" and Type != "ALL" and `Land-use` != "ALL"'),
                cost_am_df.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Management Type` != "ALL"'),
                cost_non_ag_df,
                cost_transition_ag2ag_df.query('`From-land-use` != "ALL" and `To-land-use` != "ALL" and `Cost-type` != "ALL"'),
                cost_transition_ag2non_ag_df.query('`From-land-use` != "ALL" and `To-land-use` != "ALL" and `Cost-type` != "ALL"'),
                cost_transition_non_ag2ag_df.query('`From-land-use` != "ALL" and `To-land-use` != "ALL" and `Cost-type` != "ALL"')
            ]
        ).round({'Value ($)': 2}
        ).query('abs(`Value ($)`) > 1e-4'
        ).reset_index(drop=True
        ).groupby(['region_level', 'region', 'Source', 'Year']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index()

    dfs = []
    for (region_level, region), df in rev_cost_net_region.groupby(['region_level', 'region']):
        df_col = groupby_to_records(df, ['Source'], ['name','data'], value_cols=('Year', 'Value ($)'))
        df_col['type'] = 'column'

        df_col.loc[len(df_col)] = [
            'Profit',
            df.groupby(['Year'])[['Value ($)']].sum(numeric_only=True).reset_index().values.tolist(),
            'line',
        ]
        df_col['region_level'] = region_level
        df_col['region'] = region
        dfs.append(df_col)

    rev_cost_wide_json = pd.concat(dfs, ignore_index=True)
    rev_cost_wide_json['name_order'] = rev_cost_wide_json['name'].map({name: i for i, name in enumerate(order)})
    rev_cost_wide_json = rev_cost_wide_json.sort_values(['region_level', 'region', 'name_order']).drop(columns=['name_order']).reset_index(drop=True)
    rev_cost_wide_json['color'] = rev_cost_wide_json['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region_level, region),df in rev_cost_wide_json.groupby(['region_level', 'region']):
        df = df.drop(columns=['region_level', 'region'])
        df.columns = ['name','data','type','color']
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    filename = 'Economics_overview_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # Overview: ag cost/revenue by type
    economics_ag = pd.concat([revenue_ag_df.query('Water_supply != "ALL" and Type != "ALL" and `Land-use` != "ALL"'), cost_ag_df.query('Water_supply != "ALL" and Type != "ALL" and `Land-use` != "ALL"')])\
        .query('abs(`Value ($)`) > 1')\
        .groupby(['region_level', 'region', 'Type', 'Year'])['Value ($)']\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value ($)': 2})


    df_wide = groupby_to_records(economics_ag, ['region_level', 'region', 'Type'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: list(COLORS.keys()).index(x) if x in COLORS else -1)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])


    out_dict = {}
    for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    filename = f'Economics_overview_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # Overview: ag-man cost/revenue by type
    economics_am = pd.concat(
        [
            revenue_am_df.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Management Type` != "ALL"').assign(Rev_Cost='Revenue'),
            cost_am_df.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Management Type` != "ALL"').assign(Rev_Cost='Cost')
        ]
        ).query('abs(`Value ($)`) > 1'
        ).round({'Value ($)': 2}
        ).groupby(['region_level', 'region', 'Management Type', 'Rev_Cost', 'Year'])[['Value ($)']
        ].sum(
        ).reset_index()

    df_wide = groupby_to_records(economics_am, ['region_level', 'region', 'Management Type', 'Rev_Cost'], ['region_level', 'region', 'name', 'Rev_Cost', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)

    out_dict = {}
    for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region', 'Rev_Cost'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    filename = f'Economics_overview_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # Overview: non-ag cost/revenue by type
    economics_non_ag = pd.concat([
            revenue_non_ag_df.assign(Rev_Cost='Revenue'),
            cost_non_ag_df.assign(Rev_Cost='Cost')]
        ).query('abs(`Value ($)`) > 1'
        ).round({'Value ($)': 2}
        ).groupby(['region_level', 'region', 'Land-use', 'Rev_Cost', 'Year'])[['Value ($)']
        ].sum(
        ).reset_index()

    df_wide = groupby_to_records(economics_non_ag, ['region_level', 'region', 'Land-use', 'Rev_Cost'], ['region_level', 'region', 'name', 'Rev_Cost', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'

    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, rev_cost), df in df_wide.groupby(['region_level', 'region', 'Rev_Cost']):
        df = df.drop(['region_level', 'region', 'Rev_Cost'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        out_dict[region_level][region][rev_cost] = df.to_dict(orient='records')
    profit_na_files = files.query('base_name == "economics_non_ag_profit"').reset_index(drop=True)
    profit_na_df = pd.concat([df for p in profit_na_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    profit_na_df = profit_na_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_na_df = profit_na_df.query('`Land-use` != "ALL"').round({'Value ($)': 2})

    df_profit = groupby_to_records(profit_na_df, ['region_level', 'region', 'Land-use'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_profit['type'] = 'column'
    df_profit['color'] = df_profit['name'].apply(lambda x: COLORS[x])
    df_profit['name_order'] = df_profit['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_profit = df_profit.sort_values('name_order').drop(columns=['name_order'])

    for (region_level, region), df in df_profit.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        out_dict[region_level][region]['Profit'] = df.to_dict(orient='records')

    filename = f'Economics_overview_Non_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')




    # -------------------- Economics for ag (separate files per MapType) --------------------

    def write_chart_js(out_dict, filename):
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

    # Ag Revenue: region → Type(source) → Water → [series by LU]
    ag_rev = revenue_ag_df.query('`Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
    df_wide = groupby_to_records(ag_rev, ['region_level', 'region', 'Type', 'Water_supply', 'Land-use'], ['region_level', 'region', '_type', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, _type, water), df in df_wide.groupby(['region_level', 'region', '_type', 'water']):
        df = df.drop(['region_level', 'region', '_type', 'water'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if _type not in out_dict[region_level][region]:
            out_dict[region_level][region][_type] = {}
        out_dict[region_level][region][_type][water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Ag_revenue')

    # Ag Cost: region → Type(source) → Water → [series by LU]
    ag_cost = cost_ag_df.query('`Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
    df_wide = groupby_to_records(ag_cost, ['region_level', 'region', 'Type', 'Water_supply', 'Land-use'], ['region_level', 'region', '_type', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, _type, water), df in df_wide.groupby(['region_level', 'region', '_type', 'water']):
        df = df.drop(['region_level', 'region', '_type', 'water'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if _type not in out_dict[region_level][region]:
            out_dict[region_level][region][_type] = {}
        out_dict[region_level][region][_type][water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Ag_cost')

    # Ag Profit: region → Water → [series by LU]
    profit_ag_files = files.query('base_name == "economics_ag_profit"').reset_index(drop=True)
    profit_ag_df = pd.concat([pd.read_csv(p) for p in profit_ag_files['path']], ignore_index=True)
    profit_ag_df = profit_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_ag_df = profit_ag_df.query('`Land-use` != "ALL"').round({'Value ($)': 2})

    df_profit = groupby_to_records(profit_ag_df, ['region_level', 'region', 'Water_supply', 'Land-use'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_profit['type'] = 'column'
    df_profit['color'] = df_profit['name'].apply(lambda x: COLORS[x])
    df_profit['name_order'] = df_profit['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_profit = df_profit.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, water), df in df_profit.groupby(['region_level', 'region', 'water']):
        df = df.drop(['region_level', 'region', 'water'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        out_dict[region_level][region][water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Ag_profit')


    # Ag Transition (Ag2Ag): region → Type(source) → Water → [series by To-LU]
    ag2ag_files = files.query('base_name == "economics_ag_transition_Ag2Ag"').reset_index(drop=True)
    ag2ag_df = pd.concat([df for p in ag2ag_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    ag2ag_df = ag2ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    ag2ag_df['Value ($)'] = ag2ag_df['Value ($)'] * -1   # Convert cost to negative

    out_dict = {}
    if not ag2ag_df.empty:
        ag2ag_filt = ag2ag_df.query('`To_Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
        if not ag2ag_filt.empty:
            df_wide = groupby_to_records(ag2ag_filt, ['region_level', 'region', 'Type', 'Water_supply', 'To_Land-use'], ['region_level', 'region', '_type', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
            df_wide['type'] = 'column'
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#999999'))
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            for (region_level, region, _type, water), df in df_wide.groupby(['region_level', 'region', '_type', 'water']):
                df = df.drop(['region_level', 'region', '_type', 'water'], axis=1)
                if region_level not in out_dict:
                    out_dict[region_level] = {}
                if region not in out_dict[region_level]:
                    out_dict[region_level][region] = {}
                if _type not in out_dict[region_level][region]:
                    out_dict[region_level][region][_type] = {}
                out_dict[region_level][region][_type][water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Ag_transition_ag2ag')

    # Ag Transition (NonAg2Ag): region → Type(source) → Water → [series by From-LU]
    nonag2ag_files = files.query('base_name == "economics_ag_transition_NonAg2Ag"').reset_index(drop=True)
    out_dict = {}
    if not nonag2ag_files.empty:
        _dfs = [df for p in nonag2ag_files['path'] if not (df := pd.read_csv(p)).empty]
        if _dfs:
            nonag2ag_df = pd.concat(_dfs, ignore_index=True)
            nonag2ag_df = nonag2ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
            nonag2ag_df['Value ($)'] = nonag2ag_df['Value ($)'] * -1   # Convert cost to negative
            nonag2ag_filt = nonag2ag_df.query('`From_Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
            if not nonag2ag_filt.empty:
                df_wide = groupby_to_records(nonag2ag_filt, ['region_level', 'region', 'Type', 'Water_supply', 'From_Land-use'], ['region_level', 'region', '_type', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
                df_wide['type'] = 'column'
                df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#999999'))
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                for (region_level, region, _type, water), df in df_wide.groupby(['region_level', 'region', '_type', 'water']):
                    df = df.drop(['region_level', 'region', '_type', 'water'], axis=1)
                    if region_level not in out_dict:
                        out_dict[region_level] = {}
                    if region not in out_dict[region_level]:
                        out_dict[region_level][region] = {}
                    if _type not in out_dict[region_level][region]:
                        out_dict[region_level][region][_type] = {}
                    out_dict[region_level][region][_type][water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Ag_transition_nonag2ag')


    # -------------------- Economics for ag-management (separate files per MapType) --------------------

    # Am Revenue: region → AgMgt → Water → [series by LU]  (matches map: AgMgt → Water → LU)
    am_rev = revenue_am_df.query('`Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
    df_wide = groupby_to_records(am_rev, ['region_level', 'region', 'Management Type', 'Water_supply', 'Land-use'], ['region_level', 'region', 'am', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#999999'))
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, am, water), df in df_wide.groupby(['region_level', 'region', 'am', 'water']):
        df = df.drop(['region_level', 'region', 'am', 'water'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if am not in out_dict[region_level][region]:
            out_dict[region_level][region][am] = {}
        out_dict[region_level][region][am][water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Am_revenue')

    # Am Cost: region → AgMgt → Water → Cost_type → [series by LU]
    am_cost = cost_am_df.query('`Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
    df_wide = groupby_to_records(am_cost, ['region_level', 'region', 'Management Type', 'Water_supply', 'Cost_type', 'Land-use'], ['region_level', 'region', 'am', 'water', 'cost_type', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#999999'))
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, am, water, cost_type), df in df_wide.groupby(['region_level', 'region', 'am', 'water', 'cost_type']):
        df = df.drop(['region_level', 'region', 'am', 'water', 'cost_type'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if am not in out_dict[region_level][region]:
            out_dict[region_level][region][am] = {}
        if water not in out_dict[region_level][region][am]:
            out_dict[region_level][region][am][water] = {}
        out_dict[region_level][region][am][water][cost_type] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Am_cost')

    # Am Profit: region → AgMgt → Water → [series by LU]
    profit_am_files = files.query('base_name == "economics_am_profit"').reset_index(drop=True)
    profit_am_df = pd.concat([df for p in profit_am_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    profit_am_df = profit_am_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_am_df = profit_am_df.query('`Land-use` != "ALL"').round({'Value ($)': 2})

    df_profit = groupby_to_records(profit_am_df, ['region_level', 'region', 'Management Type', 'Water_supply', 'Land-use'], ['region_level', 'region', 'am', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_profit['type'] = 'column'
    df_profit['color'] = df_profit['name'].apply(lambda x: COLORS.get(x, '#999999'))
    df_profit['name_order'] = df_profit['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
    df_profit = df_profit.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, am, water), df in df_profit.groupby(['region_level', 'region', 'am', 'water']):
        df = df.drop(['region_level', 'region', 'am', 'water'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if am not in out_dict[region_level][region]:
            out_dict[region_level][region][am] = {}
        out_dict[region_level][region][am][water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Am_profit')


    # -------------------- Economics for non-ag (separate files per MapType) --------------------

    # NonAg Revenue: region → [series by LU]
    na_rev = revenue_non_ag_df.query('`Land-use` != "ALL" and abs(`Value ($)`) > 1').round({'Value ($)': 2})
    df_wide = groupby_to_records(na_rev, ['region_level', 'region', 'Land-use'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_NonAg_revenue')

    # NonAg Cost: region → [series by LU]
    na_cost = cost_non_ag_df.query('`Land-use` != "ALL" and abs(`Value ($)`) > 1').round({'Value ($)': 2})
    df_wide = groupby_to_records(na_cost, ['region_level', 'region', 'Land-use'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_NonAg_cost')

    # NonAg Profit: region → [series by LU]
    profit_na_files = files.query('base_name == "economics_non_ag_profit"').reset_index(drop=True)
    profit_na_df = pd.concat([df for p in profit_na_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    profit_na_df = profit_na_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_na_df = profit_na_df.query('`Land-use` != "ALL"').round({'Value ($)': 2})

    df_profit = groupby_to_records(profit_na_df, ['region_level', 'region', 'Land-use'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_profit['type'] = 'column'
    df_profit['color'] = df_profit['name'].apply(lambda x: COLORS[x])
    df_profit['name_order'] = df_profit['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_profit = df_profit.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region), df in df_profit.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_NonAg_profit')

    # NonAg Transition (Ag2NonAg): region → [series by LU]
    t_ag2nonag_files = files.query('base_name == "economics_non_ag_transition_Ag2NonAg"').reset_index(drop=True)
    t_ag2nonag_df = pd.concat([df for p in t_ag2nonag_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    t_ag2nonag_df = t_ag2nonag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    t_ag2nonag_df['Value ($)'] = t_ag2nonag_df['Value ($)'] * -1   # Convert cost to negative
    t_ag2nonag_filt = t_ag2nonag_df.query('`Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')

    out_dict = {}
    if not t_ag2nonag_filt.empty:
        df_wide = groupby_to_records(t_ag2nonag_filt, ['region_level', 'region', 'Land-use'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
            df = df.drop(['region_level', 'region'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            out_dict[region_level][region] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_NonAg_transition_ag2nonag')

    # NonAg Transition (NonAg2NonAg): region → [series by LU]
    t_nonag2nonag_files = files.query('base_name == "economics_non_ag_transition_NonAg2NonAg"').reset_index(drop=True)
    out_dict = {}
    if not t_nonag2nonag_files.empty:
        _dfs = [df for p in t_nonag2nonag_files['path'] if not (df := pd.read_csv(p)).empty]
        if _dfs:
            t_nonag2nonag_df = pd.concat(_dfs, ignore_index=True)
            t_nonag2nonag_df = t_nonag2nonag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
            t_nonag2nonag_df['Value ($)'] = t_nonag2nonag_df['Value ($)'] * -1   # Convert cost to negative
            t_nonag2nonag_filt = t_nonag2nonag_df.query('`Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
            if not t_nonag2nonag_filt.empty:
                df_wide = groupby_to_records(t_nonag2nonag_filt, ['region_level', 'region', 'Land-use'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
                df_wide['type'] = 'column'
                df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#999999'))
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
                    df = df.drop(['region_level', 'region'], axis=1)
                    if region_level not in out_dict:
                        out_dict[region_level] = {}
                    out_dict[region_level][region] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_NonAg_transition_nonag2nonag')


    # -------------------- Economics Sum (Ag + Am + NonAg profit) --------------------
    # Load profit CSVs
    profit_ag_files = files.query('base_name == "economics_ag_profit"').reset_index(drop=True)
    profit_ag_sum_df = pd.concat([pd.read_csv(p) for p in profit_ag_files['path']], ignore_index=True)
    profit_ag_sum_df = profit_ag_sum_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_ag_sum_df = profit_ag_sum_df.query('`Land-use` != "ALL" and Water_supply != "ALL"')

    profit_am_files = files.query('base_name == "economics_am_profit"').reset_index(drop=True)
    profit_am_sum_df = pd.concat([df for p in profit_am_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    profit_am_sum_df = profit_am_sum_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_am_sum_df = profit_am_sum_df.query('`Land-use` != "ALL" and Water_supply != "ALL" and `Management Type` != "ALL"')
    # Sum over management types to get (region, Water_supply, Land-use, Year) level
    profit_am_sum_df = profit_am_sum_df.groupby(['region_level', 'region', 'Water_supply', 'Land-use', 'Year'])[['Value ($)']].sum().reset_index()

    profit_na_files = files.query('base_name == "economics_non_ag_profit"').reset_index(drop=True)
    profit_na_sum_df = pd.concat([df for p in profit_na_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    profit_na_sum_df = profit_na_sum_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_na_sum_df = profit_na_sum_df.query('`Land-use` != "ALL"')
    # Assign nonag to Dryland to avoid double counting
    profit_na_sum_df['Water_supply'] = 'Dryland'

    # Aggregate each to Type level (sum over all land uses, water supplies, management types)
    econ_sum_ag = profit_ag_sum_df.groupby(['region_level', 'region', 'Year'])[['Value ($)']].sum().reset_index().assign(Type='Agricultural Land-use')
    econ_sum_am = profit_am_sum_df.groupby(['region_level', 'region', 'Year'])[['Value ($)']].sum().reset_index().assign(Type='Agricultural Management')
    econ_sum_nonag = profit_na_sum_df.groupby(['region_level', 'region', 'Year'])[['Value ($)']].sum().reset_index().assign(Type='Non-Agricultural Land-use')

    econ_sum_type = pd.concat([econ_sum_ag, econ_sum_am, econ_sum_nonag], ignore_index=True)

    df_wide = groupby_to_records(econ_sum_type, ['region_level', 'region', 'Type'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    filename = 'Economics_Sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')




    # # -------------------- Transition cost for Ag2Ag --------------------
    # cost_transition_ag2ag_df['Value ($)'] = cost_transition_ag2ag_df['Value ($)'] * -1  # Convert from negative to positive
    # group_cols = ['Type', 'From-land-use', 'To-land-use']

    # for idx, col in enumerate(group_cols):
    #     df_AUS = cost_transition_ag2ag_df\
    #         .groupby(['Year', col])[['Value ($)']]\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()\
    #         .assign(region='AUSTRALIA')
    #     df_AUS_wide.columns = ['name', 'data','region']
    #     df_AUS_wide['type'] = 'column'

    #     df_region = cost_transition_ag2ag_df\
    #         .groupby(['Year', 'region', col])\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()
    #     df_region_wide.columns = ['name', 'region', 'data']
    #     df_region_wide['type'] = 'column'


    #     df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)

    #     out_dict = {}
    #     for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
    #         df = df.drop(['region_level', 'region'], axis=1)
    #         out_dict[region] = df.to_dict(orient='records')

    #     filename = f'Economics_transition_split_ag2ag_{idx+1}_{col.replace(" ", "_")}'
    #     with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
    #         f.write(f'window["{filename}"] = ')
    #         json.dump(out_dict, f, separators=(',', ':'), indent=2)
    #         f.write(';\n')


    # # -------------------- Transition cost matrix for Ag2Ag --------------------
    # cost_transition_ag2ag_trans_mat_AUS = cost_transition_ag2ag_df\
    #     .groupby(['Year','From-land-use', 'To-land-use'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})\
    #     .query('abs(`Value ($)`) > 1e-4')\
    #     .assign(region='AUSTRALIA')

    # cost_transition_ag2ag_trans_mat_region_df = cost_transition_ag2ag_df\
    #     .groupby(['Year','From-land-use', 'To-land-use', 'region'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})


    # cost_transition_ag2ag_trans_mat = pd.concat([
    #     cost_transition_ag2ag_trans_mat_AUS,
    #     cost_transition_ag2ag_trans_mat_region_df
    # ])


    # out_dict_area = {}
    # for (region_level, region, year), df in cost_transition_ag2ag_trans_mat.groupby(['region_level', 'region', 'Year']):

    #     out_dict_area.setdefault(region, {})

    #     transition_mat = df.pivot(index='From-land-use', columns='To-land-use', values='Value ($)')
    #     transition_mat = transition_mat.reindex(index=AG_LANDUSE, columns=AG_LANDUSE)
    #     transition_mat = transition_mat.fillna(0)
    #     total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)

    #     transition_mat['SUM'] = transition_mat.sum(axis=1)
    #     transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

    #     heat_area = transition_mat.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
    #     ).format('{:,.0f}')

    #     heat_area_html = heat_area.to_html()
    #     heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)

    #     out_dict_area[region][str(year)] = rf'{heat_area_html}'

    # filename = 'Economics_transition_mat_ag2ag'
    # with open(f'{SAVE_DIR}/{filename}.js', 'w', encoding='utf-8') as f:
    #     f.write(f'window["{filename}"] = ')
    #     json.dump(out_dict_area, f, separators=(',', ':'), indent=2)
    #     f.write(';\n')






    # # -------------------- Transition cost for Ag2Non-Ag --------------------
    # cost_transition_ag2non_ag_df['Value ($)'] = cost_transition_ag2non_ag_df['Value ($)'] * -1  # Convert from negative to positive
    # group_cols = ['Cost-type', 'From-land-use', 'To-land-use']

    # for idx, col in enumerate(group_cols):
    #     df_AUS = cost_transition_ag2non_ag_df\
    #         .groupby(['Year', col])[['Value ($)']]\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()\
    #         .assign(region='AUSTRALIA')
    #     df_AUS_wide.columns = ['name', 'data','region']
    #     df_AUS_wide['type'] = 'column'

    #     df_region = cost_transition_ag2non_ag_df\
    #         .groupby(['Year', 'region', col])\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()
    #     df_region_wide.columns = ['name', 'region', 'data']
    #     df_region_wide['type'] = 'column'


    #     df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)

    #     out_dict = {}
    #     for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
    #         df = df.drop(['region_level', 'region'], axis=1)
    #         out_dict[region] = df.to_dict(orient='records')

    #     filename = f'Economics_transition_split_Ag2NonAg_{idx+1}_{col.replace(" ", "_")}'
    #     with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
    #         f.write(f'window["{filename}"] = ')
    #         json.dump(out_dict, f, separators=(',', ':'), indent=2)
    #         f.write(';\n')


    # # -------------------- Transition cost matrix for Ag2Non-Ag --------------------
    # cost_transition_ag2nonag_trans_mat_AUS = cost_transition_ag2non_ag_df\
    #     .groupby(['Year','From-land-use', 'To-land-use'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})\
    #     .assign(region='AUSTRALIA')

    # cost_transition_ag2nonag_trans_mat_region_df = cost_transition_ag2non_ag_df\
    #     .groupby(['Year','From-land-use', 'To-land-use', 'region'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})


    # cost_transition_ag2nonag_trans_mat = pd.concat([
    #     cost_transition_ag2nonag_trans_mat_AUS,
    #     cost_transition_ag2nonag_trans_mat_region_df
    # ])


    # out_dict_area = {}
    # for (region_level, region, year), df in cost_transition_ag2nonag_trans_mat.groupby(['region_level', 'region', 'Year']):

    #     out_dict_area.setdefault(region, {})

    #     transition_mat = df.pivot(index='From-land-use', columns='To-land-use', values='Value ($)')
    #     transition_mat = transition_mat.reindex(index=AG_LANDUSE, columns=RENAME_NON_AG.values())
    #     transition_mat = transition_mat.fillna(0)
    #     total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)

    #     transition_mat['SUM'] = transition_mat.sum(axis=1)
    #     transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

    #     heat_area = transition_mat.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
    #     ).format('{:,.0f}')

    #     heat_area_html = heat_area.to_html()
    #     heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)

    #     out_dict_area[region][str(year)] = rf'{heat_area_html}'

    # filename = 'Economics_transition_mat_ag2nonag'
    # with open(f'{SAVE_DIR}/{filename}.js', 'w', encoding='utf-8') as f:
    #     f.write(f'window["{filename}"] = ')
    #     json.dump(out_dict_area, f, separators=(',', ':'), indent=2)
    #     f.write(';\n')





    # # -------------------- Transition cost for Non-Ag to Ag --------------------
    # cost_transition_non_ag2ag_df['Value ($)'] = cost_transition_non_ag2ag_df['Value ($)'] * -1  # Convert from negative to positive
    # group_cols = ['Cost-type', 'From-land-use', 'To-land-use']

    # for idx, col in enumerate(group_cols):
    #     df_AUS = cost_transition_non_ag2ag_df\
    #         .groupby(['Year', col])[['Value ($)']]\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_AUS_wide = df_AUS.groupby([col])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()\
    #         .assign(region='AUSTRALIA')
    #     df_AUS_wide.columns = ['name', 'data','region']
    #     df_AUS_wide['type'] = 'column'

    #     df_region = cost_transition_non_ag2ag_df\
    #         .groupby(['Year', 'region', col])\
    #         .sum(numeric_only=True)\
    #         .reset_index()\
    #         .round({'Value ($)': 2})
    #     df_region_wide = df_region.groupby([col, 'region'])[['Year','Value ($)']]\
    #         .apply(lambda x: x[['Year', 'Value ($)']].values.tolist())\
    #         .reset_index()
    #     df_region_wide.columns = ['name', 'region', 'data']
    #     df_region_wide['type'] = 'column'


    #     df_wide = pd.concat([df_AUS_wide, df_region_wide], axis=0, ignore_index=True)

    #     out_dict = {}
    #     for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
    #         df = df.drop(['region_level', 'region'], axis=1)
    #         out_dict[region] = df.to_dict(orient='records')

    #     filename = f'Economics_transition_split_NonAg2Ag_{idx+1}_{col.replace(" ", "_")}'
    #     with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
    #         f.write(f'window["{filename}"] = ')
    #         json.dump(out_dict, f, separators=(',', ':'), indent=2)
    #         f.write(';\n')



    # # -------------------- Transition cost matrix for Non-Ag to Ag --------------------
    # cost_transition_nonag2ag_trans_mat_AUS = cost_transition_non_ag2ag_df\
    #     .groupby(['Year','From-land-use', 'To-land-use'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})\
    #     .assign(region='AUSTRALIA')

    # cost_transition_nonag2ag_trans_mat_region_df = cost_transition_non_ag2ag_df\
    #     .groupby(['Year','From-land-use', 'To-land-use', 'region'])\
    #     .sum(numeric_only=True)\
    #     .reset_index()\
    #     .round({'Value ($)': 2})


    # cost_transition_nonag2ag_trans_mat = pd.concat([
    #     cost_transition_nonag2ag_trans_mat_AUS,
    #     cost_transition_nonag2ag_trans_mat_region_df
    # ])


    # out_dict_area = {}
    # for (region_level, region, year), df in cost_transition_nonag2ag_trans_mat.groupby(['region_level', 'region', 'Year']):

    #     out_dict_area.setdefault(region, {})

    #     transition_mat = df.pivot(index='From-land-use', columns='To-land-use', values='Value ($)')
    #     transition_mat = transition_mat.reindex(index=RENAME_NON_AG.values(), columns=AG_LANDUSE)
    #     transition_mat = transition_mat.fillna(0)
    #     total_area_from = transition_mat.sum(axis=1).values.reshape(-1, 1)

    #     transition_mat['SUM'] = transition_mat.sum(axis=1)
    #     transition_mat.loc['SUM'] = transition_mat.sum(axis=0)

    #     heat_area = transition_mat.style.background_gradient(
    #         cmap='Oranges',
    #         axis=1,
    #         subset=pd.IndexSlice[:transition_mat.index[-2], :transition_mat.columns[-2]]
    #     ).format('{:,.0f}')

    #     heat_area_html = heat_area.to_html()
    #     heat_area_html = re.sub(r'(?<!\d)0(?!\d)', '-', heat_area_html)

    #     out_dict_area[region][str(year)] = rf'{heat_area_html}'

    # filename = 'Economics_transition_mat_nonag2ag'
    # with open(f'{SAVE_DIR}/{filename}.js', 'w', encoding='utf-8') as f:
    #     f.write(f'window["{filename}"] = ')
    #     json.dump(out_dict_area, f, separators=(',', ':'), indent=2)
    #     f.write(';\n')

    return "Economics data processing completed"


def process_production_data(files, SAVE_DIR, years):
    """Process and save production data (Section 2)."""
    quantity_df = files.query('base_name == "quantity_production_t_separate"')
    quantity_df = pd.concat([pd.read_csv(path) for path in quantity_df['path']])\
        .assign(Commodity = lambda x: x['Commodity'].str.capitalize())\
        .replace({'Sheep lexp': 'Sheep live export', 'Beef lexp': 'Beef live export'})\
        .infer_objects(copy=False)\
        .assign(group = lambda x: x['Commodity'].map(COMMIDOTY_GROUP.get))\
        .replace(RENAME_AM_NON_AG)\
        .infer_objects(copy=False)\
        .query(f'Year.isin({years}) and abs(`Production (t/KL)`) > 1')\
        .query('Commodity != "All"')\
        .round({'`Production (t/KL)`': 2})

    quantity_ag = quantity_df.query('Type == "Agricultural"').copy()
    quantity_am = quantity_df.query('Type == "Agricultural_Management"').copy()
    quantity_non_ag = quantity_df.query('Type == "Non_Agricultural"').copy()

    # Fill 0 for empty non-agr dataframe
    if quantity_non_ag.empty:
        quantity_non_ag = pd.DataFrame(
        [{'Commodity': 'beef meat', 'Type': 'Non_Agricultural', 'Year': 2050, 'region': 'ACT', 'region_level': 'region_state', 'Production (t/KL)': 0},
         {'Commodity': 'beef meat', 'Type': 'Non_Agricultural', 'Year': 2020, 'region': 'AUSTRALIA', 'region_level': 'region_state', 'Production (t/KL)': 0}]
        ).assign(Commodity = lambda x: x['Commodity'].str.capitalize())


    # -------------------- Demand --------------------

    DEMAND_DATA = get_demand_df()\
        .query(f'Year.isin({years}) and abs(`Quantity (tonnes, KL)`) > 1')\
        .replace({'Beef lexp': 'Beef live export', 'Sheep lexp': 'Sheep live export'})\
        .infer_objects(copy=False)\
        .set_index(['Commodity', 'Type', 'Year'])\
        .reindex(COMMODITIES_ALL, level=0)\
        .reset_index()\
        .replace(RENAME_AM_NON_AG)\
        .infer_objects(copy=False)\
        .assign(group = lambda x: x['Commodity'].map(COMMIDOTY_GROUP.get))

    # Convert imports to negative values, making it below zero in the stacked column chart
    DEMAND_DATA_long = DEMAND_DATA.query('Type != "Production" ')
    DEMAND_DATA_long.loc[DEMAND_DATA_long['Type'] == 'Imports', 'Quantity (tonnes, KL)'] *= -1

    DEMAND_target = DEMAND_DATA.query('Type == "Production"')




    # -------------------- Ranking --------------------

    quantity_rank = pd.concat([quantity_ag.query('Water_supply != "ALL"'), quantity_non_ag, quantity_am.query('Water_supply != "ALL" and Commodity != "ALL"')])\
        .groupby(['Year', 'region_level', 'region', 'group'])[['Production (t/KL)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'region_level', 'group', 'Production (t/KL)'], ascending=[True, True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'region_level', 'group']).cumcount())\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))\
        .assign(Year=lambda x: x['Year'].astype(int))\
        .round({'Production (t/KL)': 2})

    out_dict = {}
    for (region_level, region, group), df in quantity_rank.groupby(['region_level', 'region', 'group']):
        df = df.drop(['region_level', 'region'], axis=1)

        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if group not in out_dict[region_level][region]:
            out_dict[region_level][region][group] = {}

        out_dict[region_level][region][group]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][group]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][group]['value'] = df.set_index('Year')['Production (t/KL)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Production_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Overview --------------------

    # sum
    demand_type_wide = groupby_to_records(DEMAND_DATA_long .groupby(['Year', 'Type'])[['Quantity (tonnes, KL)']] .sum(numeric_only=True) .reset_index() .round({'Quantity (tonnes, KL)': 2}), ['Type'], ['name', 'data'], value_cols=('Year', 'Quantity (tonnes, KL)'))
    demand_type_wide['type'] = 'column'
    demand_type_wide['color'] = demand_type_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {'AUSTRALIA': demand_type_wide.to_dict(orient='records')}

    filename = 'Production_overview_demand_type'
    with open(fr'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    # seperate plot data
    for _type in ['Domestic', 'Exports', 'Imports', 'Feed']:
        
        demand_group = DEMAND_DATA_long\
            .query('Type == @_type')\
            .groupby(['Year', 'Type', 'group'])[['Quantity (tonnes, KL)']]\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Quantity (tonnes, KL)': 2})\
            .groupby(['Type', 'group'])[['Year', 'Quantity (tonnes, KL)']]\
            .apply(lambda x: x[['Year', 'Quantity (tonnes, KL)']].values.tolist())\
            .reset_index()
            
        demand_group = demand_group.drop(columns=['Type'])
        demand_group.columns = ['name', 'data']
        demand_group['type'] = 'column'
        demand_group['color'] = demand_group['name'].apply(lambda x: COLORS[x])

        out_dict = {'AUSTRALIA': demand_group.to_dict(orient='records')}

        filename = f'Production_overview_{_type}'
        with open(fr'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


    # -------------------- Overview: Australia production achievement (%) --------------------
    quantity_diff = files.query('base_name == "quantity_comparison"').reset_index(drop=True)
    quantity_diff = pd.concat([pd.read_csv(path) for path in quantity_diff['path']], ignore_index=True)
    quantity_diff = quantity_diff.replace({'Sheep lexp': 'Sheep live export', 'Beef lexp': 'Beef live export'}).infer_objects(copy=False)
    quantity_diff = quantity_diff[['Year','Commodity','Prop_diff (%)']].rename(columns={'Prop_diff (%)': 'Demand Achievement (%)'})

    mask_AUS = quantity_diff.groupby('Commodity'
        )['Demand Achievement (%)'
        ].transform(lambda x: abs(round(x) - 100) > 0.01)
    quantity_diff_AUS = quantity_diff[mask_AUS].copy()
    quantity_diff_wide_AUS = quantity_diff_AUS\
        .groupby(['Commodity'])[['Year','Demand Achievement (%)']]\
        .apply(lambda x: x[['Year','Demand Achievement (%)']].values.tolist())\
        .reset_index()

    quantity_diff_wide_AUS['type'] = 'line'
    quantity_diff_wide_AUS.columns = ['name','data', 'type']
    quantity_diff_wide_AUS['color'] = quantity_diff_wide_AUS['name'].apply(lambda x: COLORS[x])

    quantity_diff_wide_AUS_data = {
        'AUSTRALIA': quantity_diff_wide_AUS.to_dict(orient='records')
    }
    filename = 'Production_overview_AUS_achive_percent'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(quantity_diff_wide_AUS_data, f, separators=(',', ':'), indent=2)
        f.write(';\n')




    # -------------------- Commodity production for ag --------------------
    df_wide = groupby_to_records(quantity_ag, ['region_level', 'region', 'Water_supply', 'Commodity'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Production (t/KL)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, water), df in df_wide.groupby(['region_level', 'region', 'water']):
        df = df.drop(['region_level', 'region', 'water'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if water not in out_dict[region_level][region]:
            out_dict[region_level][region][water] = {}
        out_dict[region_level][region][water] = df.to_dict(orient='records')

    filename = f'Production_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Commodity production for ag-man --------------------
    # Hierarchy: region_level → region → am → water → [series(name=Commodity)]
    # Matches Economics Am pattern so the chart is split by land use.
    am_prod = quantity_am.query('Commodity != "ALL"').copy()

    df_wide = groupby_to_records(am_prod, ['region_level', 'region', 'am', 'Water_supply', 'Commodity'], ['region_level', 'region', 'am', 'water', 'name', 'data'], value_cols=('Year', 'Production (t/KL)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#999999'))
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x) if x in COMMODITIES_ALL else 999)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    # Add am="ALL" rows: sum across all AgMgts, keep per-commodity series
    am_all_agg = am_prod\
        .groupby(['region_level', 'region', 'Water_supply', 'Commodity', 'Year'])\
        .sum(numeric_only=True)\
        .reset_index()
    df_wide_am_ALL = groupby_to_records(am_all_agg, ['region_level', 'region', 'Water_supply', 'Commodity'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Production (t/KL)'))
    df_wide_am_ALL.insert(2, 'am', 'ALL')
    df_wide_am_ALL['type'] = 'column'
    df_wide_am_ALL['color'] = df_wide_am_ALL['name'].apply(lambda x: COLORS.get(x, '#999999'))
    df_wide_am_ALL['name_order'] = df_wide_am_ALL['name'].apply(lambda x: COMMODITIES_ALL.index(x) if x in COMMODITIES_ALL else 999)
    df_wide_am_ALL = df_wide_am_ALL.sort_values('name_order').drop(columns=['name_order'])

    df_wide = pd.concat([df_wide, df_wide_am_ALL], axis=0, ignore_index=True)

    out_dict = {}
    for (region_level, region, am, water), df in df_wide.groupby(['region_level', 'region', 'am', 'water']):
        df = df.drop(['region_level', 'region', 'am', 'water'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if am not in out_dict[region_level][region]:
            out_dict[region_level][region][am] = {}
        out_dict[region_level][region][am][water] = df.to_dict(orient='records')

    filename = f'Production_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Commodity production for non-ag --------------------
    df_wide = groupby_to_records(quantity_non_ag, ['region_level', 'region', 'Commodity'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Production (t/KL)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        out_dict.setdefault(region_level, {})[region] = df.to_dict(orient='records')

    filename = f'Production_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Sum production (Ag + Am + NonAg) --------------------
    # Assign non_ag Water_supply='Dryland' to avoid double counting, then sum all types
    quantity_non_ag_with_water = quantity_non_ag.copy()
    quantity_non_ag_with_water['Water_supply'] = 'Dryland'

    quantity_sum = pd.concat([
        quantity_ag.query('Water_supply != "ALL"')[['region_level', 'region', 'Water_supply', 'Commodity', 'Year', 'Production (t/KL)']],
        quantity_am.query('Water_supply != "ALL" and Commodity != "ALL"')[['region_level', 'region', 'Water_supply', 'Commodity', 'Year', 'Production (t/KL)']],
        quantity_non_ag_with_water[['region_level', 'region', 'Water_supply', 'Commodity', 'Year', 'Production (t/KL)']],
    ], ignore_index=True)

    # Add ALL water level
    quantity_sum_all_water = quantity_sum\
        .groupby(['region_level', 'region', 'Commodity', 'Year'])[['Production (t/KL)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(Water_supply='ALL')
    quantity_sum = pd.concat([quantity_sum_all_water, quantity_sum], ignore_index=True)

    # Group by region, water, commodity → time series
    df_wide = groupby_to_records(quantity_sum .groupby(['region_level', 'region', 'Water_supply', 'Commodity', 'Year'])[['Production (t/KL)']] .sum(numeric_only=True) .reset_index(), ['region_level', 'region', 'Water_supply', 'Commodity'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Production (t/KL)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x) if x in COMMODITIES_ALL else 999)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, water), df in df_wide.groupby(['region_level', 'region', 'water']):
        df = df.drop(['region_level', 'region', 'water'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        out_dict[region_level][region][water] = df.to_dict(orient='records')

    filename = f'Production_Sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    return "Production data processing completed"


def process_renewable_data(files, SAVE_DIR, years):
    """Process and save renewable energy data (ag-mgt level only)."""

    re_files = files.query('base_name == "renewable_energy_with_existing"').reset_index(drop=True)

    if re_files.empty:
        return "Renewable energy data processing skipped (no files found)"

    _re_dfs = [df for path in re_files['path'] if not (df := pd.read_csv(path, engine='pyarrow')).empty]
    if not _re_dfs:
        return "Renewable energy data processing skipped (all CSV files are empty)"
    re_df = pd.concat(_re_dfs, ignore_index=True)

    # Rename am labels to match COLORS keys
    re_df['am'] = re_df['am'].replace(RENAME_AM_NON_AG)

    # Exclude lu=ALL only; am=ALL and lm=ALL are valid hierarchy buttons in the report
    re_am_df = re_df.query('lu != "ALL"')

    # ---- Renewable energy by region_level → region → AgMgt (am → lm → lu) ----
    df_wide = groupby_to_records(
        re_am_df,
        ['region_level', 'region', 'am', 'lm', 'lu'],
        ['region_level', 'region', 'am', 'lm', 'name', 'data'],
        value_cols=('Year', 'Value (MWh)'),
    )
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#AAAAAA'))
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    # Targets (state-level only → tagged as region_state)
    re_targets_files = files.query('base_name.str.contains("renewable_energy_targets")').reset_index(drop=True)
    re_targets_df = _read_concat(re_targets_files['path'])
    re_targets_df['region_level'] = 'region_state'
    re_targets_df_wide = groupby_to_records(re_targets_df, ['region_level', 'region', 'am', 'lm'], ['region_level', 'region', 'am', 'lm', 'data'], value_cols=('Year', 'Value (MWh)'))
    re_targets_df_wide['type'] = 'line'
    re_targets_df_wide['name'] = 'Target'
    re_targets_df_wide['color'] = "#424040"

    # Merge production + targets (targets only join region_state rows)
    re_df_wide = pd.concat([df_wide, re_targets_df_wide], ignore_index=True)

    # Build nested dict: region_level → region → am → lm → [series]
    out_dict = {}
    for (region_level, region, am, lm), df in re_df_wide.groupby(['region_level', 'region', 'am', 'lm']):
        df = df.drop(columns=['region_level', 'region', 'am', 'lm'])
        out_dict.setdefault(region_level, {}).setdefault(region, {}).setdefault(am, {})[lm] = df.to_dict(orient='records')

    filename = 'Renewable_energy_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    return "Renewable energy data processing completed"


def process_ghg_data(files, SAVE_DIR, lu_group_map, years):
    """Process and save GHG emissions data (Section 4)."""
    '''GHG is written to disk no matter if GHG_EMISSIONS_LIMITS is 'off' or 'on' '''

    filter_str = '''
    category == "GHG" 
    and base_name.str.contains("GHG_emissions") 
    '''.replace('\n', ' ').replace('  ', ' ')

    GHG_files = files.query(filter_str).reset_index(drop=True)

    GHG_ag = GHG_files.query('base_name.str.contains("agricultural_landuse")').reset_index(drop=True)
    GHG_ag = pd.concat([pd.read_csv(path) for path in GHG_ag['path']], ignore_index=True)
    GHG_ag = GHG_ag.replace(GHG_NAMES).infer_objects(copy=False).round({'Value (t CO2e)': 2})

    GHG_non_ag = GHG_files.query('base_name.str.contains("no_ag_reduction")').reset_index(drop=True)
    GHG_non_ag = _read_concat(GHG_non_ag['path'])
    GHG_non_ag = GHG_non_ag.replace(RENAME_AM_NON_AG).infer_objects(copy=False).round({'Value (t CO2e)': 2})
    
    GHG_ag_man = GHG_files.query('base_name.str.contains("agricultural_management")').reset_index(drop=True)
    GHG_ag_man = _read_concat(GHG_ag_man['path'])
    GHG_ag_man = GHG_ag_man.replace(RENAME_AM_NON_AG).infer_objects(copy=False).round({'Value (t CO2e)': 2})

    GHG_transition = GHG_files.query('base_name.str.contains("transition_penalty")').reset_index(drop=True)
    GHG_transition = _read_concat(GHG_transition['path'])
    GHG_transition = GHG_transition.replace(RENAME_AM_NON_AG).infer_objects(copy=False).round({'Value (t CO2e)': 2})
    GHG_transition = GHG_transition.query('Type != "ALL" and Water_supply != "ALL"').reset_index(drop=True)

    GHG_off_land = GHG_files.query('base_name.str.contains("offland_commodity")')
    GHG_off_land = pd.concat([pd.read_csv(path) for path in GHG_off_land['path']], ignore_index=True).round({'Value (t CO2e)': 2})
    GHG_off_land['Value (t CO2e)'] = GHG_off_land['Total GHG Emissions (tCO2e)']
    GHG_off_land['Commodity'] = GHG_off_land['COMMODITY'].apply(lambda x: x[0].capitalize() + x[1:])
    GHG_off_land = GHG_off_land.drop(columns=['COMMODITY', 'Total GHG Emissions (tCO2e)'])
    GHG_off_land['Emission Source'] = GHG_off_land['Emission Source']\
        .replace({
            'CO2': 'Carbon Dioxide (CO2)',
            'CH4': 'Methane (CH4)',
            'N2O': 'Nitrous Oxide (N2O)'
        }).infer_objects(copy=False)

    GHG_land = pd.concat([GHG_ag, GHG_non_ag, GHG_ag_man, GHG_transition], axis=0)\
        .query('abs(`Value (t CO2e)`) > 1')\
        .reset_index(drop=True)
    GHG_land_non_all = pd.concat([GHG_ag.query('Water_supply != "ALL" and Source != "ALL" and `Land-use` != "ALL"'), GHG_non_ag.query('`Land-use` != "ALL"'), GHG_ag_man.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Agricultural Management Type` != "ALL"'), GHG_transition], axis=0)\
        .query('abs(`Value (t CO2e)`) > 1')\
        .reset_index(drop=True)

    GHG_land['Land-use type'] = GHG_land['Land-use'].map(lu_group_map)
    GHG_land_non_all['Land-use type'] = GHG_land_non_all['Land-use'].map(lu_group_map)

    net_offland_AUS = GHG_off_land.groupby('Year')[['Value (t CO2e)']].sum(numeric_only=True).reset_index()
    net_offland_AUS_wide = net_offland_AUS[['Year','Value (t CO2e)']].values.tolist()


    GHG_limit = GHG_files.query('base_name == "GHG_emissions"')
    GHG_limit = pd.concat([pd.read_csv(path) for path in GHG_limit['path']], ignore_index=True)
    GHG_limit = GHG_limit.query('Variable == "GHG_EMISSIONS_LIMIT_TCO2e"').copy()
    GHG_limit['Value (t CO2e)'] = GHG_limit['Emissions (t CO2e)']
    GHG_limit_wide = list(map(list,zip(GHG_limit['Year'],GHG_limit['Value (t CO2e)'])))
    
    order_GHG = [
        'Agricultural Land-use',
        'Agricultural Management',
        'Non-Agricultural Land-use',
        'Off-land emissions',
        'Unallocated natural to modified',
        'Unallocated natural to livestock natural',
        'Livestock natural to modified',
        'Net emissions',
        'GHG emission limit'
    ]


    # -------------------- GHG overview --------------------

    # sum
    GHG_region = {}
    for (region_level, region),df in GHG_land_non_all.groupby(['region_level', 'region']):
        df_reg = groupby_to_records(df .groupby(['Year','Type'])[['Value (t CO2e)']] .sum(numeric_only=True) .reset_index(), ['Type'], ['name','data'], value_cols=('Year', 'Value (t CO2e)'))
        df_reg['type'] = 'column'

        # Add on Year, never positionally. A region need not carry every simulation year: the
        # `abs(Value) > 1` filter above drops a whole (region, year) group when nothing in it
        # clears 1 tCO2e — e.g. a no-target baseline with no non-ag land in the base year. Adding
        # `.values` to a Series then raised "operands could not be broadcast together", and
        # zip(years, ...) silently mislabelled the remaining points. Reindex on the full `years`
        # so every series still spans the x-axis, with 0 where a year genuinely has nothing.
        net_land = df.groupby('Year')['Value (t CO2e)'].sum().reindex(years, fill_value=0)

        if region == "AUSTRALIA":
            df_reg.loc[len(df_reg)] = ['Off-land emissions', net_offland_AUS_wide,  'column']
            df_reg.loc[len(df_reg)] = ['GHG emission limit', GHG_limit_wide, 'line']
            net_offland = GHG_off_land.groupby('Year')['Value (t CO2e)'].sum().reindex(years, fill_value=0)
            net_total = net_land.add(net_offland, fill_value=0)
            df_reg.loc[len(df_reg)] = ['Net emissions', list(zip(net_total.index, net_total.values)), 'line']
        else:
            df_reg.loc[len(df_reg)] = ['Net emissions', list(zip(net_land.index, net_land.values)), 'line']
                

        df_reg['name_order'] = df_reg['name'].apply(lambda x: order_GHG.index(x))
        df_reg = df_reg.sort_values('name_order').drop(columns=['name_order'])
        df_reg['color'] = df_reg['name'].apply(lambda x: COLORS[x])
        if region_level not in GHG_region:
            GHG_region[region_level] = {}
        GHG_region[region_level][region] = json.loads(df_reg.to_json(orient='records'))


    filename = 'GHG_overview_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(GHG_region, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
        
    # Ag
    GHG_ag_non_all_wide = groupby_to_records(GHG_ag.query('Water_supply != "ALL" and Source != "ALL" and `Land-use` != "ALL"') .groupby(['region_level','region','Land-use','Year'])[['Value (t CO2e)']] .sum(numeric_only=True) .reset_index(), ['region_level','region','Land-use'], ['region_level','region', 'name','data'], value_cols=('Year', 'Value (t CO2e)'))
    GHG_ag_non_all_wide['type'] = 'column'
    GHG_ag_non_all_wide['color'] = GHG_ag_non_all_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region_level, region),df in GHG_ag_non_all_wide.groupby(['region_level', 'region']):
        df = df.drop(columns=['region_level', 'region'])
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')
    

    filename = 'GHG_overview_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Am
    GHG_ag_man_non_all_wide = groupby_to_records(GHG_ag_man.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Agricultural Management Type` != "ALL"') .groupby(['region_level', 'region', 'Agricultural Management Type', 'Year'])[['Value (t CO2e)']] .sum(numeric_only=True) .reset_index(), ['region_level', 'region', 'Agricultural Management Type'], ['region_level', 'region', 'name','data'], value_cols=('Year', 'Value (t CO2e)'))
    GHG_ag_man_non_all_wide['type'] = 'column'
    GHG_ag_man_non_all_wide['color'] = GHG_ag_man_non_all_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region_level, region), df in GHG_ag_man_non_all_wide.groupby(['region_level', 'region']):
        df = df.drop(columns=['region_level', 'region'])
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')
    

    filename = 'GHG_overview_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Non-Ag
    GHG_non_ag_wide = groupby_to_records(GHG_non_ag .query('`Land-use` != "ALL"') .groupby(['region_level','region','Land-use','Year'])[['Value (t CO2e)']] .sum(numeric_only=True) .reset_index(), ['region_level','region','Land-use'], ['region_level','region','name','data'], value_cols=('Year', 'Value (t CO2e)'))
    GHG_non_ag_wide['type'] = 'column'
    GHG_non_ag_wide['color'] = GHG_non_ag_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region_level, region),df in GHG_non_ag_wide.groupby(['region_level', 'region']):
        df = df.drop(columns=['region_level', 'region'])
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')
    

    filename = 'GHG_overview_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    


    # -------------------- GHG Sum (Ag + Am + NonAg + Transition) --------------------
    # Aggregate each component to Type level (sum over all land uses, water, sources)
    ghg_sum_ag_base = GHG_ag.query('Water_supply != "ALL" and Source != "ALL" and `Land-use` != "ALL"')\
        .groupby(['region_level', 'region', 'Year'])[['Value (t CO2e)']].sum().reset_index()
    ghg_sum_trans = GHG_transition.groupby(['region_level', 'region', 'Year'])[['Value (t CO2e)']].sum().reset_index()
    ghg_sum_ag = pd.concat([ghg_sum_ag_base, ghg_sum_trans], ignore_index=True)\
        .groupby(['region_level', 'region', 'Year'])[['Value (t CO2e)']].sum().reset_index()\
        .assign(Type='Agricultural Land-use')

    ghg_sum_am = GHG_ag_man.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Agricultural Management Type` != "ALL"')\
        .groupby(['region_level', 'region', 'Year'])[['Value (t CO2e)']].sum().reset_index()\
        .assign(Type='Agricultural Management')

    ghg_sum_nonag = GHG_non_ag.query('`Land-use` != "ALL"')\
        .groupby(['region_level', 'region', 'Year'])[['Value (t CO2e)']].sum().reset_index()\
        .assign(Type='Non-Agricultural Land-use')

    ghg_sum_type = pd.concat([ghg_sum_ag, ghg_sum_am, ghg_sum_nonag], ignore_index=True)
    ghg_sum_type = ghg_sum_type.query('abs(`Value (t CO2e)`) > 1').reset_index(drop=True)

    ghg_off_land_by_year = GHG_off_land.groupby('Year')[['Value (t CO2e)']].sum(numeric_only=True).reset_index()

    df_wide = groupby_to_records(ghg_sum_type, ['region_level', 'region', 'Type'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value (t CO2e)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    # Add Net emissions and GHG emission limit lines for AUSTRALIA
    if 'AUSTRALIA' in out_dict.get('AUSTRALIA', {}):
        net_values = (
            ghg_sum_type.query('region == "AUSTRALIA"')
            .groupby('Year')['Value (t CO2e)'].sum()
            + ghg_off_land_by_year.set_index('Year')['Value (t CO2e)']
        )
        net_australia_wide = [[y, v] for y, v in zip(net_values.index.tolist(), net_values.values.tolist())]
        out_dict['AUSTRALIA']['AUSTRALIA'].append({'name': 'Net emissions',    'data': net_australia_wide, 'type': 'line', 'color': COLORS['Net emissions']})
        out_dict['AUSTRALIA']['AUSTRALIA'].append({'name': 'GHG emission limit', 'data': GHG_limit_wide,      'type': 'line', 'color': COLORS['GHG emission limit']})

    filename = 'GHG_Sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- GHG ranking --------------------
    GHG_rank_emission_region = GHG_land_non_all\
        .query('`Value (t CO2e)` > 0')\
        .groupby(['Year', 'region_level', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'region_level', 'Value (t CO2e)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount())\
        .assign(Type='GHG emissions')
        
    off_land_by_year = GHG_off_land.groupby('Year')['Value (t CO2e)'].sum()
    mask = GHG_rank_emission_region['region'] == 'AUSTRALIA'
    GHG_rank_emission_region.loc[mask, 'Value (t CO2e)'] += (
        GHG_rank_emission_region.loc[mask, 'Year'].map(off_land_by_year).values
    )
    
    GHG_rank_sequestration_region = GHG_land_non_all\
        .query('`Value (t CO2e)` < 0')\
        .assign(**{'Value (t CO2e)': lambda x: abs(x['Value (t CO2e)'])})\
        .groupby(['Year', 'region_level', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'region_level', 'Value (t CO2e)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount())\
        .assign(Type='GHG sequestrations')
    GHG_rank_region_net = GHG_rank_emission_region\
        .merge(GHG_rank_sequestration_region, on=['Year', 'region_level', 'region'], how='outer', suffixes=('_emission', '_sequestration'))\
        .assign(**{'Value (t CO2e)': lambda x: x['Value (t CO2e)_emission'] - x['Value (t CO2e)_sequestration']})\
        .assign(Type='Total')


    GHG_rank = pd.concat([
        GHG_rank_emission_region, 
        GHG_rank_sequestration_region, 
        GHG_rank_region_net,
        ], axis=0, ignore_index=True).reset_index(drop=True)\
        .round({'Value (t CO2e)':2})\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))
    

    out_dict = {}
    for (region_level, region, e_type), df in GHG_rank.groupby(['region_level', 'region', 'Type']):
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if e_type not in out_dict[region_level][region]:
            out_dict[region_level][region][e_type] = {}

        df = df.drop(columns=['region_level', 'region'])
        out_dict[region_level][region][e_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][e_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][e_type]['value'] = df.set_index('Year')['Value (t CO2e)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'GHG_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        



    # -------------------- GHG by agricultural land-use --------------------
    GHG_ag = GHG_land.query('Type == "Agricultural Land-use" and `Land-use` != "ALL"')
    GHG_CO2 = GHG_ag.query('~Source.isin(@GHG_CATEGORY.keys())').copy()
    GHG_CO2['GHG Category'] = 'CO2'

    GHG_nonCO2 = GHG_ag.query('Source.isin(@GHG_CATEGORY.keys())').copy()
    GHG_nonCO2['GHG Category'] = GHG_nonCO2['Source'].apply(lambda x: GHG_CATEGORY[x].keys())
    GHG_nonCO2['Multiplier'] = GHG_nonCO2['Source'].apply(lambda x: GHG_CATEGORY[x].values())
    GHG_nonCO2 = GHG_nonCO2.explode(['GHG Category','Multiplier']).reset_index(drop=True)
    GHG_nonCO2['Value (t CO2e)'] = GHG_nonCO2['Value (t CO2e)'] * GHG_nonCO2['Multiplier']
    GHG_nonCO2 = GHG_nonCO2.drop(columns=['Multiplier'])

    GHG_ag_emissions_long = pd.concat([GHG_CO2, GHG_nonCO2], axis=0).reset_index(drop=True)
    GHG_ag_emissions_long['GHG Category'] = GHG_ag_emissions_long['GHG Category']\
        .replace({
            'CH4': 'Methane (CH4)',
            'N2O': 'Nitrous Oxide (N2O)',
            'CO2': 'Carbon Dioxide (CO2)'
        }).infer_objects(copy=False)

    df_wide = GHG_ag_emissions_long\
        .groupby(['region_level', 'region', 'Source', 'Water_supply', 'Land-use', 'Year'])[['Value (t CO2e)']]\
        .sum()\
        .reset_index()\
        .round({'Value (t CO2e)': 2})
    df_wide = groupby_to_records(df_wide, ['region_level', 'region', 'Source', 'Water_supply', 'Land-use'], ['region_level', 'region', 'source', 'water', 'name', 'data'], value_cols=('Year', 'Value (t CO2e)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, water, source), df in df_wide.groupby(['region_level', 'region', 'water', 'source']):
        df = df.drop(['region_level', 'region', 'water', 'source'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if water not in out_dict[region_level][region]:
            out_dict[region_level][region][water] = {}
        if source not in out_dict[region_level][region][water]:
            out_dict[region_level][region][water][source] = {}
        out_dict[region_level][region][water][source] = df.to_dict(orient='records')
        
    filename = 'GHG_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- GHG by Non-Agricultural --------------------
    Non_ag_reduction_long = GHG_land.query('Type == "Non-Agricultural Land-use" and `Land-use` != "ALL"').reset_index(drop=True)
    Non_ag_reduction_long['Value (t CO2e)'] *= -1  # Convert from negative to positive
    
    df_region = Non_ag_reduction_long\
        .groupby(['Year', 'region_level', 'region', 'Land-use'])[['Value (t CO2e)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value (t CO2e)': 2})
    df_wide = groupby_to_records(df_region, ['region_level', 'region', 'Land-use'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value (t CO2e)'))
    df_wide['type'] = 'column'
    
    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
    
    out_dict = {}
    for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    filename = f'GHG_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- GHG by Agricultural Managements --------------------
    Ag_man_sequestration_long = GHG_land\
        .query('Type == "Agricultural Management" and `Agricultural Management Type` != "ALL"')\
        .reset_index(drop=True)
    Ag_man_sequestration_long['Value (t CO2e)'] = Ag_man_sequestration_long['Value (t CO2e)'] * -1  # Convert from negative to positive

    df_wide = groupby_to_records(Ag_man_sequestration_long, ['region_level', 'region', 'Water_supply', 'Land-use', 'Agricultural Management Type'], ['region_level', 'region', 'water', 'landuse', 'name', 'data'], value_cols=('Year', 'Value (t CO2e)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region_level, region, water, landuse), df in df_wide.groupby(['region_level', 'region', 'water', 'landuse']):
        df = df.drop(['region_level', 'region', 'water', 'landuse'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if water not in out_dict[region_level][region]:
            out_dict[region_level][region][water] = {}
        out_dict[region_level][region][water][landuse] = df.to_dict(orient='records')

    filename = 'GHG_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

        



    return "GHG data processing completed"


def process_water_data(files, SAVE_DIR):
    """Process and save water data (Section 5)."""
    
    water_files = files.query('category == "water"').reset_index(drop=True)

    if water_files.empty:
        return "Water data processing skipped (no files found)"
    
    ############ Watershed level  ##############

    water_net_yield_watershed_region = water_files.query('base_name == "water_yield_separate_watershed"')
    water_net_yield_watershed_region = pd.concat([pd.read_csv(path) for path in water_net_yield_watershed_region['path']], ignore_index=True)
    water_net_yield_watershed_AUS = water_net_yield_watershed_region\
        .groupby(['Water Supply',  'Landuse', 'Type', 'Agricultural Management', 'Year'], dropna=False)[['Water Net Yield (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(Region='AUSTRALIA')
    water_net_yield_watershed = pd.concat([water_net_yield_watershed_region, water_net_yield_watershed_AUS], ignore_index=True)
    water_net_yield_watershed = water_net_yield_watershed\
        .replace(RENAME_AM_NON_AG)\
        .infer_objects(copy=False)\
        .query('abs(`Water Net Yield (ML)`) > 1e-4')\
        .rename(columns={'Water Net Yield (ML)': 'Value (ML)'})


    hist_and_public_wny_water_region = water_files.query('base_name == "water_yield_limits_and_public_land"')
    hist_and_public_wny_water_region = pd.concat([pd.read_csv(path) for path in hist_and_public_wny_water_region['path']], ignore_index=True)
    
    water_outside_LUTO = hist_and_public_wny_water_region[['Year','Region', 'Water yield outside LUTO (ML)']].rename(
        columns={'Water yield outside LUTO (ML)': 'Value (ML)'} )
    water_outside_LUTO = pd.concat([
        water_outside_LUTO,
        water_outside_LUTO.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_climate_change_impact = hist_and_public_wny_water_region[['Year','Region', 'Climate Change Impact (ML)']].rename(
        columns={'Climate Change Impact (ML)': 'Value (ML)'})
    water_climate_change_impact = pd.concat([
        water_climate_change_impact,
        water_climate_change_impact.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_domestic_use = hist_and_public_wny_water_region[['Year','Region', 'Domestic Water Use (ML)']]\
        .rename(columns={'Domestic Water Use (ML)': 'Value (ML)'})
    water_domestic_use['Value (ML)'] *= -1  # Domestic water use is negative, indicating a water loss (consumption)
    water_domestic_use = pd.concat([
        water_domestic_use,
        water_domestic_use.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_yield_limit = hist_and_public_wny_water_region[['Year','Region', 'Water Yield Limit (ML)']].rename(
        columns={'Water Yield Limit (ML)': 'Value (ML)'})
    water_yield_limit = pd.concat([
        water_yield_limit,
        water_yield_limit.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_net_yield = hist_and_public_wny_water_region[['Year','Region', 'Water Net Yield (ML)']].rename(
        columns={'Water Net Yield (ML)': 'Value (ML)'})
    water_net_yield = pd.concat([
        water_net_yield,
        water_net_yield.groupby(['Year'])[['Value (ML)']].sum(numeric_only=True).assign(Region='AUSTRALIA').reset_index()
    ], ignore_index=True)
    
    water_targets_before_relaxation = water_files.query('base_name == "water_yield_relaxed_region_raw"')
    water_targets_before_relaxation = pd.concat([pd.read_csv(path) for path in water_targets_before_relaxation['path']])\
        .drop(columns=['Region Id'])\
        .rename(columns={'Region Name': 'Region', 'Target': 'Value (ML)'})


    # -------------------- Water yield overview --------------------
    water_inside_LUTO_wide = water_net_yield_watershed\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby(['Region', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .round({'Value (ML)': 2})\
        .reset_index()\
        .groupby(['Region'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_outside_LUTO_wide = water_outside_LUTO\
        .groupby(['Region'])[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_CCI_wide = water_climate_change_impact\
        .groupby('Region')[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_domestic_wide = water_domestic_use\
        .groupby('Region')[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_net_yield_wide = water_net_yield\
        .groupby('Region')[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    water_limit_wide = water_yield_limit\
        .groupby('Region')[['Year','Value (ML)']]\
        .apply(lambda x: x[['Year','Value (ML)']].values.tolist())\
        .reset_index()
    
    # -------------------- Water yield overview --------------------
    water_yield_region = {}
    for reg_name in water_net_yield['Region'].unique():
        
        water_inside = water_inside_LUTO_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_outside = water_outside_LUTO_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_CCI = water_CCI_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_domestic = water_domestic_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_net_yield_sum = water_net_yield_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        water_limit = water_limit_wide.query('Region == @reg_name').values.flatten().tolist()[1]
        
        water_df = pd.DataFrame([
            ['Water Yield Inside LUTO Study Area', water_inside, 'column', None, None],
            ['Water Yield Outside LUTO Study Area', water_outside, 'column', None, None],
            ['Climate Change Impact', water_CCI, 'column', None, None],
            ['Domestic Water Use', water_domestic, 'column', None, None],
            ['Water Net Yield', water_net_yield_sum, 'line', None, None],
            ['Water Limit (model)', water_limit, 'line', 'black', None],
        ],
            columns=['name', 'data','type','color','dashStyle']
        )
        
        # Add historical water limit if it exists for this region
        if reg_name in water_targets_before_relaxation['Region'].values:
            raw_targets = water_targets_before_relaxation.query('`Region` == @reg_name')[['Year','Value (ML)']].values.tolist()
            water_df.loc[len(water_df)] = ['Water Limit (historical level)', raw_targets, 'line', '#2176cc', 'Dash']

        water_yield_region[reg_name] = water_df.to_dict(orient='records')
        
    filename = 'Water_overview_watershed'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as outfile:
        outfile.write(f'window["{filename}"] = ')
        json.dump(water_yield_region, outfile, separators=(',', ':'), indent=2)
        outfile.write(';\n')
        
        
    ############ NRM region level  ##############  
       
    water_net_yield_NRM_region_region = water_files.query('base_name == "water_yield_separate_NRM"')
    water_net_yield_NRM_region_region = pd.concat([pd.read_csv(path) for path in water_net_yield_NRM_region_region['path']], ignore_index=True)
    water_net_yield_NRM_region_AUS = water_net_yield_NRM_region_region\
        .groupby(['Water Supply', 'Landuse',  'Type', 'Agricultural Management', 'Year'], dropna=False)[['Water Net Yield (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(region_NRM='AUSTRALIA')
    water_net_yield_NRM_region = pd.concat([water_net_yield_NRM_region_region, water_net_yield_NRM_region_AUS])\
        .replace(RENAME_AM_NON_AG)\
        .infer_objects(copy=False)\
        .query('abs(`Water Net Yield (ML)`) > 1e-4')\
        .rename(columns={'Water Net Yield (ML)': 'Value (ML)'})


    # -------------------- Water yield ranking by NRM --------------------
    water_ranking_type = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby(['region_NRM', 'Type', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Type', 'Value (ML)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year','Type']).cumcount())
        
    water_ranking_total = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby(['region_NRM', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (ML)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby('Year').cumcount())\
        .assign(Type='Total')
        
    water_ranking = pd.concat([water_ranking_type, water_ranking_total], axis=0, ignore_index=True)\
        .round({'Value (ML)':2})\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))

    out_dict = {}
    for (region, w_type), df in water_ranking.groupby(['region_NRM', 'Type']):
        df = df.drop(columns='region_NRM')
        region_level = 'NRM'  # Water data uses NRM level
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if w_type not in out_dict[region_level][region]:
            out_dict[region_level][region][w_type] = {} 
        out_dict[region_level][region][w_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][w_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region_level][region][w_type]['value'] = df.set_index('Year')['Value (ML)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Water_ranking_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- Overview  --------------------
    
    # sum
    water_sum = groupby_to_records(water_net_yield_NRM_region .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"') .groupby(['region_NRM', 'Type', 'Year'])[['Value (ML)']] .sum(numeric_only=True) .reset_index() .round({'Value (ML)': 2}), ['region_NRM', 'Type'], ['region', 'name','data'], value_cols=('Year', 'Value (ML)'))
    water_sum['type'] = 'column'
    water_sum['color'] = water_sum['name'].apply(lambda x: COLORS[x])
    water_sum['region_level'] = 'NRM'

    out_dict = {}
    for (region_level, region), df in water_sum.groupby(['region_level', 'region']):
        df = df.drop(columns=['region_level', 'region'])
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    filename = f'Water_overview_NRM_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    
    # Ag
    water_ag = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .query('Type == "Agricultural Land-use"')
    
    water_overview_ag = groupby_to_records(water_ag .groupby(['region_NRM', 'Landuse', 'Year'])[['Value (ML)']] .sum(numeric_only=True) .reset_index() .round({'Value (ML)': 2}), ['region_NRM', 'Landuse'], ['region', 'name','data'], value_cols=('Year', 'Value (ML)'))
    water_overview_ag['type'] = 'column'
    water_overview_ag['color'] = water_overview_ag['name'].map(COLORS)
    water_overview_ag['region_level'] = 'NRM'
    out_dict = {}
    for (region_level, region), df in water_overview_ag.groupby(['region_level', 'region']):
        df = df.drop(columns=['region_level', 'region'])
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

            
    filename = f'Water_overview_NRM_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
    # Am
    water_am = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .query('Type == "Agricultural Management"')
    
    water_overview_am = groupby_to_records(water_am .groupby(['region_NRM', 'Agricultural Management', 'Year'])[['Value (ML)']] .sum(numeric_only=True) .reset_index() .round({'Value (ML)': 2}), ['region_NRM', 'Agricultural Management'], ['region', 'name', 'data'], value_cols=('Year', 'Value (ML)'))
    water_overview_am['type'] = 'column'
    water_overview_am['color'] = water_overview_am['name'].map(COLORS)
    
    water_overview_am['region_level'] = 'NRM'  # Add region_level column

    out_dict = {}
    for (region_level, region), df in water_overview_am.groupby(['region_level', 'region']):
        df = df.drop(columns=['region_level', 'region'])
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    filename = f'Water_overview_NRM_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Non-Ag
    water_nonag = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .query('Type == "Non-Agricultural Land-use"')

    water_overview_nonag = groupby_to_records(water_nonag .groupby(['region_NRM', 'Landuse', 'Year'])[['Value (ML)']] .sum(numeric_only=True) .reset_index() .round({'Value (ML)': 2}), ['region_NRM', 'Landuse'], ['region', 'name', 'data'], value_cols=('Year', 'Value (ML)'))
    water_overview_nonag['type'] = 'column'
    water_overview_nonag['color'] = water_overview_nonag['name'].map(COLORS)
    
    water_overview_nonag['region_level'] = 'NRM'  # Add region_level column

    out_dict = {}
    for (region_level, region), df in water_overview_nonag.groupby(['region_level', 'region']):
        df = df.drop(columns=['region_level', 'region'])
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')

    filename = f'Water_overview_NRM_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- Water yield for Ag by NRM --------------------
    water_ag_AUS = water_net_yield_NRM_region\
        .query('Type == "Agricultural Land-use"')\
        .groupby(['Water Supply', 'Landuse', 'Year',])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(region='AUSTRALIA', region_level='AUSTRALIA')
        
    water_ag = pd.concat([
        water_ag_AUS,
        water_net_yield_NRM_region.query('Type == "Agricultural Land-use" and region_NRM != "AUSTRALIA"').rename(columns={'region_NRM': 'region'}).assign(region_level='NRM')
        ], ignore_index=True)\
        .query('Landuse != "ALL" and `Water Supply` != "ALL"')

    # Add ALL water aggregate
    water_ag_all_water = water_ag\
        .groupby(['region_level', 'region', 'Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True).reset_index()\
        .assign(**{'Water Supply': 'ALL'})
    water_ag = pd.concat([water_ag, water_ag_all_water], ignore_index=True)

    water_ag_sim_years = np.sort(water_ag['Year'].unique())
    df_region_wide = water_ag.groupby(['region_level', 'region', 'Water Supply', 'Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: annualise_points(x['Year'], x['Value (ML)'], water_ag_sim_years))\
        .reset_index()
  
    df_region_wide.columns = ['region_level', 'region', 'water', 'name',  'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide['name'].map(COLORS)
    df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region, water), df in df_region_wide.groupby(['region_level', 'region', 'water']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if water not in out_dict[region_level][region]:
            out_dict[region_level][region][water] = {}
        out_dict[region_level][region][water] = df.to_dict(orient='records')
        
        
    filename = f'Water_Ag_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        

        
            
    # -------------------- Water yield for Am by NRM region --------------------
    water_am_AUS = water_net_yield_NRM_region\
        .query('Type == "Agricultural Management"')\
        .groupby(['Agricultural Management', 'Water Supply', 'Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(region='AUSTRALIA')\
        .assign(region_level='AUSTRALIA')

    water_am_NRM = water_net_yield_NRM_region\
        .query('Type == "Agricultural Management" and region_NRM != "AUSTRALIA"')\
        .rename(columns={'region_NRM': 'region'})\
        .assign(region_level='NRM')

    water_am = pd.concat([water_am_AUS, water_am_NRM], ignore_index=True).query('`Agricultural Management` != "ALL"')

    # Add ALL water aggregate
    water_am_all_water = water_am\
        .query('`Water Supply` != "ALL"')\
        .groupby(['region_level', 'region', 'Landuse', 'Agricultural Management', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True).reset_index()\
        .assign(**{'Water Supply': 'ALL'})
    water_am = pd.concat([water_am, water_am_all_water], ignore_index=True)

    water_am_sim_years = np.sort(water_am['Year'].unique())
    df_region_wide = water_am.groupby(['region_level', 'region', 'Water Supply', 'Landuse', 'Agricultural Management'])[['Year','Value (ML)']]\
        .apply(lambda x: annualise_points(x['Year'], x['Value (ML)'], water_am_sim_years))\
        .reset_index()
    df_region_wide.columns = ['region_level', 'region', 'water', 'landuse', 'name',  'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region_level, region, water, landuse), df in df_region_wide.groupby(['region_level', 'region', 'water', 'landuse']):
        df = df.drop(['region_level', 'region', 'water', 'landuse'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        if region not in out_dict[region_level]:
            out_dict[region_level][region] = {}
        if water not in out_dict[region_level][region]:
            out_dict[region_level][region][water] = {}
        out_dict[region_level][region][water][landuse] = df.to_dict(orient='records')

    filename = f'Water_Am_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
            
            
    # -------------------- Water yield for Non-Agricultural Land-use by NRM region --------------------
    water_nonag_AUS = water_net_yield_NRM_region\
        .query('Type == "Non-Agricultural Land-use" and Landuse != "ALL"')\
        .groupby(['Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(
            region='AUSTRALIA',
            region_level='AUSTRALIA',
            name_order=lambda x: x['Landuse'].apply(lambda y: LANDUSE_ALL_RENAMED.index(y)))\
        .sort_values('name_order')\
        .drop(columns=['name_order'])

    water_nonag_NRM = water_net_yield_NRM_region\
        .query('Type == "Non-Agricultural Land-use" and Landuse != "ALL" and region_NRM != "AUSTRALIA"')\
        .rename(columns={'region_NRM': 'region'})\
        .assign(region_level='NRM')

    water_nonag = pd.concat([water_nonag_AUS, water_nonag_NRM], ignore_index=True)

    water_nonag_sim_years = np.sort(water_nonag['Year'].unique())
    df_region_wide = water_nonag.groupby(['region_level', 'region', 'Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: annualise_points(x['Year'], x['Value (ML)'], water_nonag_sim_years))\
        .reset_index()
    df_region_wide.columns = ['region_level', 'region', 'name', 'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide['name'].map(COLORS)
    df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region_level, region), df in df_region_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        if region_level not in out_dict:
            out_dict[region_level] = {}
        out_dict[region_level][region] = df.to_dict(orient='records')
    
    filename = f'Water_NonAg_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Water yield Sum (Ag + Am + NonAg) by NRM region --------------------
    # Ag part (per NRM, exclude AUSTRALIA to avoid double-counting)
    water_ag_nrm = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .query('Type == "Agricultural Land-use" and region_NRM != "AUSTRALIA"')\
        [['region_NRM', 'Water Supply', 'Landuse', 'Year', 'Value (ML)']]

    # Am part: sum over Agricultural Management to collapse that dimension
    water_am_nrm = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .query('Type == "Agricultural Management" and region_NRM != "AUSTRALIA"')\
        .groupby(['region_NRM', 'Water Supply', 'Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True).reset_index()

    # NonAg part: assign to Dryland (NonAg has no irrigation dimension)
    water_nonag_nrm = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .query('Type == "Non-Agricultural Land-use" and region_NRM != "AUSTRALIA"')\
        .assign(**{'Water Supply': 'Dryland'})\
        [['region_NRM', 'Water Supply', 'Landuse', 'Year', 'Value (ML)']]

    water_sum_nrm = pd.concat([water_ag_nrm, water_am_nrm, water_nonag_nrm], ignore_index=True)

    # Aggregate each NRM component to Type level, then build AUS total
    water_ag_type_nrm = water_ag_nrm.groupby(['region_NRM', 'Year'])[['Value (ML)']].sum().reset_index().rename(columns={'region_NRM': 'region'}).assign(Type='Agricultural Land-use')
    water_am_type_nrm = water_am_nrm.groupby(['region_NRM', 'Year'])[['Value (ML)']].sum().reset_index().rename(columns={'region_NRM': 'region'}).assign(Type='Agricultural Management')
    water_nonag_type_nrm = water_nonag_nrm.groupby(['region_NRM', 'Year'])[['Value (ML)']].sum().reset_index().rename(columns={'region_NRM': 'region'}).assign(Type='Non-Agricultural Land-use')

    water_ag_type_aus = water_ag_nrm.groupby('Year')[['Value (ML)']].sum().reset_index().assign(region='AUSTRALIA', Type='Agricultural Land-use')
    water_am_type_aus = water_am_nrm.groupby('Year')[['Value (ML)']].sum().reset_index().assign(region='AUSTRALIA', Type='Agricultural Management')
    water_nonag_type_aus = water_nonag_nrm.groupby('Year')[['Value (ML)']].sum().reset_index().assign(region='AUSTRALIA', Type='Non-Agricultural Land-use')

    water_sum_type = pd.concat([
        water_ag_type_nrm, water_am_type_nrm, water_nonag_type_nrm,
        water_ag_type_aus, water_am_type_aus, water_nonag_type_aus,
    ], ignore_index=True)
    water_sum_type['region_level'] = 'NRM'

    df_region_wide = groupby_to_records(water_sum_type, ['region_level', 'region', 'Type'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value (ML)'))
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide['name'].map(COLORS)

    out_dict = {}
    for (region_level, region), df in df_region_wide.groupby(['region_level', 'region']):
        df = df.drop(['region_level', 'region'], axis=1)
        out_dict.setdefault(region_level, {})[region] = df.to_dict(orient='records')

    filename = 'Water_Sum_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    return "Water data processing completed"


def process_transition_data(files, SAVE_DIR):

    
    # --------------------- Transition Area start-end --------------------
    # JSON structure: region → from_water → to_water → {x_categories, y_categories, data, max_val}
    # x_categories (To-LU):   ag LUs + non-ag LUs + ALL  (all possible destinations)
    # y_categories (From-LU): ag LUs + ALL only           (land can only transition FROM ag)
    # Single start→end snapshot covering ag2ag + ag2non_ag; lives in the last year output dir.

    start_end_file = files.query('base_name == "transition_matrix_start_end"').iloc[0]

    trans_start_end_df = (
        pd.read_csv(start_end_file['path'])
        .replace(RENAME_AM_NON_AG)
        .infer_objects(copy=False)
        .round({'Area (ha)': 2})
    )

    # x-axis (To-LU): ag first, non-ag last, ALL at end — all possible destinations
    non_ag_names = set(RENAME_NON_AG.values())
    se_lus_set = (
        set(trans_start_end_df.loc[trans_start_end_df['From-land-use'] != 'ALL', 'From-land-use'].unique())
        | set(trans_start_end_df.loc[trans_start_end_df['To-land-use'] != 'ALL', 'To-land-use'].unique())
    )
    se_x_lus_orig = sorted(se_lus_set - non_ag_names) + sorted(se_lus_set & non_ag_names) + ['ALL']
    se_x_lus = se_x_lus_orig[:-1] + ['ALL']  # wrapped labels for JSON; ALL kept as-is
    se_y_lus = sorted(se_lus_set - non_ag_names) + ['ALL']

    se_out_dict = {}
    for (region, from_water, to_water), grp in trans_start_end_df.groupby(
        ['region', 'From-water-supply', 'To-water-supply']
    ):
        pivot = (
            grp
            .pivot_table(
                index='From-land-use',
                columns='To-land-use',
                values='Area (ha)',
                aggfunc='sum',
                fill_value=0,
            )
        )
        x_lu_to_idx = {lu: i for i, lu in enumerate(se_x_lus_orig)}
        y_lu_to_idx = {lu: i for i, lu in enumerate(se_y_lus)}
        x_all_idx, y_all_idx = len(se_x_lus) - 1, len(se_y_lus) - 1
        points, max_val = [], 0.0
        for (from_lu, to_lu), val in pivot.stack().items():
            xi, yi = x_lu_to_idx.get(to_lu), y_lu_to_idx.get(from_lu)
            if xi is None or yi is None or val <= 0:
                continue
            val = round(float(val), 2)
            if xi == x_all_idx or yi == y_all_idx:
                points.append({'x': xi, 'y': yi, 'value': val, 'color': '#f8f8f8'})
            elif from_lu == to_lu:
                points.append({'x': xi, 'y': yi, 'value': val, 'color': '#cccccc'})
            else:
                points.append([xi, yi, val])
                if val > max_val:
                    max_val = val

        se_out_dict.setdefault(region, {}).setdefault(from_water, {})[to_water] = {
            'x_categories': se_x_lus,
            'y_categories': se_y_lus,
            'data': points,
            'max_val': round(max_val, 2),
        }

    filename = 'Transition_start_end_area'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(se_out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    # --------------------- Transition Area year-by-years (ag2ag) --------------------
    # JSON structure: region → from_water → to_water → year → {x_categories, y_categories, data, max_val}
    # Mirrors the map layer hierarchy (From-water-supply → To-water-supply → … → year).
    # Vue selection chain: region → from_water button → to_water button → year slider → ready-to-plot leaf.

    trans_area_files = files.query('base_name == "transition_ag2ag_area"').reset_index(drop=True)

    # Collect all years from file paths so the base year (e.g. 2010) is represented
    # even when its CSV is empty and has no transitions to plot.
    all_years_from_files = sorted(
        os.path.basename(p).split('_')[-1].replace('.csv', '')
        for p in trans_area_files['path']
    )

    trans_area_df = (
        _read_concat(trans_area_files['path'])
        .replace(RENAME_AM_NON_AG)
        .infer_objects(copy=False)
        .round({'Transition Area (ha)': 2})
    )
    trans_area_df['Year'] = trans_area_df['Year'].astype(str)  # string key for JSON


    # Land-use categories: sorted alphabetically, ALL appended at end as summary row/column.
    individual_lus = (
        set(trans_area_df.loc[trans_area_df['From-land-use'] != 'ALL', 'From-land-use'].unique())
        | set(trans_area_df.loc[trans_area_df['To-land-use']   != 'ALL', 'To-land-use'].unique())
    )
    all_lus = sorted(individual_lus) + ['ALL']  # ALL appended at end as summary row/column
    all_x_lus = all_lus

    out_dict = {}
    global_max_val = 0.0
    for (yr, region_level, region, from_water, to_water), grp in trans_area_df.groupby(
        ['Year', 'region_level', 'region', 'From-water-supply', 'To-water-supply']
    ):
        # Build pivot: From-LU (rows/y) × To-LU (cols/x)
        pivot = (
            grp
            .pivot_table(
                index='From-land-use',
                columns='To-land-use',
                values='Transition Area (ha)',
                aggfunc='sum',
                fill_value=0,
            )
        )
        lu_to_idx = {lu: i for i, lu in enumerate(all_lus)}
        all_idx = len(all_lus) - 1
        points = []
        for (from_lu, to_lu), val in pivot.stack().items():
            xi, yi = lu_to_idx.get(to_lu), lu_to_idx.get(from_lu)
            if xi is None or yi is None or val <= 0:
                continue
            val = round(float(val), 2)
            if xi == all_idx or yi == all_idx:
                points.append({'x': xi, 'y': yi, 'value': val, 'color': '#f8f8f8'})
            elif from_lu == to_lu:
                points.append({'x': xi, 'y': yi, 'value': val, 'color': '#cccccc'})
            else:
                points.append([xi, yi, val])
                if val > global_max_val:
                    global_max_val = val

        out_dict.setdefault(region_level, {}).setdefault(region, {}).setdefault(from_water, {}).setdefault(to_water, {})[yr] = {
            'x_categories': all_x_lus,   # To-LU: wrapped labels for Highcharts x-axis
            'y_categories': all_lus,      # From-LU: plain labels for Highcharts y-axis
            'data': points,
        }

    # Set a consistent max_val across all years for a uniform legend scale
    global_max_val = round(global_max_val, 2)
    for rl_dict in out_dict.values():
        for region_dict in rl_dict.values():
            for fw_dict in region_dict.values():
                for tw_dict in fw_dict.values():
                    for leaf in tw_dict.values():
                        leaf['max_val'] = global_max_val


    filename = 'Transition_ag2ag_area'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    # --------------------- Transition Area year-by-years (ag2nonag) --------------------
    # JSON structure: region → from_water → to_water → year → {x_categories, y_categories, data, max_val}
    # To-water-supply value: 'Dryland' (non-ag land uses are always dryland; raw coord 'dry' → renamed to 'Dryland') + 'ALL'.
    # x-axis (To-LU): non-ag land uses + ALL;  y-axis (From-LU): ag land uses + ALL.

    trans_area_ag2nonag_files = files.query('base_name == "transition_ag2nonag_area"').reset_index(drop=True)

    all_years_ag2nonag = sorted(
        os.path.basename(p).split('_')[-1].replace('.csv', '')
        for p in trans_area_ag2nonag_files['path']
    )

    trans_area_ag2nonag_df = (
        _read_concat(trans_area_ag2nonag_files['path'])
        .replace(RENAME_AM_NON_AG)
        .infer_objects(copy=False)
        .round({'Transition Area (ha)': 2})
    )
    trans_area_ag2nonag_df['Year'] = trans_area_ag2nonag_df['Year'].astype(str)

    # Separate From-LU (ag) and To-LU (non-ag) land uses for ag2nonag transition
    non_ag_names = set(RENAME_NON_AG.values())
    from_lus_ag2nonag = set(trans_area_ag2nonag_df.loc[trans_area_ag2nonag_df['From-land-use'] != 'ALL', 'From-land-use'].unique())
    to_lus_ag2nonag   = set(trans_area_ag2nonag_df.loc[trans_area_ag2nonag_df['To-land-use']   != 'ALL', 'To-land-use'].unique())
    # y-axis (From-LU): ag land uses + ALL  (land can only transition FROM ag)
    all_y_lus_ag2nonag = sorted(from_lus_ag2nonag - non_ag_names) + ['ALL']
    # x-axis (To-LU): non-ag land uses + ALL  (land transitions TO non-ag)
    all_x_lus_ag2nonag = sorted(to_lus_ag2nonag & non_ag_names) + ['ALL']

    out_ag2nonag_dict = {}
    global_max_val_ag2nonag = 0.0
    for (yr, region_level, region, from_water, to_water), grp in trans_area_ag2nonag_df.groupby(
        ['Year', 'region_level', 'region', 'From-water-supply', 'To-water-supply']
    ):
        pivot = grp.pivot_table(
            index='From-land-use',
            columns='To-land-use',
            values='Transition Area (ha)',
            aggfunc='sum',
            fill_value=0,
        )
        lu_x_to_idx = {lu: i for i, lu in enumerate(all_x_lus_ag2nonag)}
        lu_y_to_idx = {lu: i for i, lu in enumerate(all_y_lus_ag2nonag)}
        x_all_idx = len(all_x_lus_ag2nonag) - 1
        y_all_idx = len(all_y_lus_ag2nonag) - 1
        points = []
        for (from_lu, to_lu), val in pivot.stack().items():
            xi, yi = lu_x_to_idx.get(to_lu), lu_y_to_idx.get(from_lu)
            if xi is None or yi is None or val <= 0:
                continue
            val = round(float(val), 2)
            if xi == x_all_idx or yi == y_all_idx:
                points.append({'x': xi, 'y': yi, 'value': val, 'color': '#f8f8f8'})
            elif from_lu == to_lu:
                points.append({'x': xi, 'y': yi, 'value': val, 'color': '#cccccc'})
            else:
                points.append([xi, yi, val])
                if val > global_max_val_ag2nonag:
                    global_max_val_ag2nonag = val
        out_ag2nonag_dict.setdefault(region_level, {}).setdefault(region, {}).setdefault(from_water, {}).setdefault(to_water, {})[yr] = {
            'x_categories': all_x_lus_ag2nonag,
            'y_categories': all_y_lus_ag2nonag,
            'data': points,
        }

    # Set a consistent max_val across all years for a uniform legend scale
    global_max_val_ag2nonag = round(global_max_val_ag2nonag, 2)
    for rl_dict in out_ag2nonag_dict.values():
        for region_dict in rl_dict.values():
            for fw_dict in region_dict.values():
                for tw_dict in fw_dict.values():
                    for leaf in tw_dict.values():
                        leaf['max_val'] = global_max_val_ag2nonag


    filename = 'Transition_ag2nonag_area'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_ag2nonag_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # --------------------- Transition Cost year-by-years (ag2ag) --------------------
    # JSON structure: region → cost_type → year → {x_categories, y_categories, data, max_val}
    # Cost CSV hierarchy: From-land-use → To-land-use → Cost-type (no water dims).
    # cost_type values: individual cost-type strings + 'ALL'.

    trans_cost_ag2ag_files = files.query('base_name == "transition_ag2ag_cost"').reset_index(drop=True)

    if not trans_cost_ag2ag_files.empty:
        trans_cost_ag2ag_df = (
            _read_concat(trans_cost_ag2ag_files['path'])
            .replace(RENAME_AM_NON_AG)
            .infer_objects(copy=False)
            .round({'Cost ($)': 2})
        )
        trans_cost_ag2ag_df['Year'] = trans_cost_ag2ag_df['Year'].astype(str)

        cost_lus_ag2ag = (
            set(trans_cost_ag2ag_df.loc[trans_cost_ag2ag_df['From-land-use'] != 'ALL', 'From-land-use'].unique())
            | set(trans_cost_ag2ag_df.loc[trans_cost_ag2ag_df['To-land-use'] != 'ALL',   'To-land-use'].unique())
        )
        all_lus_cost_ag2ag = sorted(cost_lus_ag2ag) + ['ALL']

        out_cost_ag2ag_dict = {}
        global_max_val_cost_ag2ag = 0.0
        for (yr, region_level, region, cost_type), grp in trans_cost_ag2ag_df.groupby(['Year', 'region_level', 'region', 'Cost-type']):
            pivot = grp.pivot_table(
                index='From-land-use', columns='To-land-use',
                values='Cost ($)', aggfunc='sum', fill_value=0,
            )
            lu_to_idx = {lu: i for i, lu in enumerate(all_lus_cost_ag2ag)}
            all_idx = len(all_lus_cost_ag2ag) - 1
            points = []
            for (from_lu, to_lu), val in pivot.stack().items():
                xi, yi = lu_to_idx.get(to_lu), lu_to_idx.get(from_lu)
                if xi is None or yi is None or val <= 0:
                    continue
                val = round(float(val), 2)
                if xi == all_idx or yi == all_idx:
                    points.append({'x': xi, 'y': yi, 'value': val, 'color': '#f8f8f8'})
                elif from_lu == to_lu:
                    points.append({'x': xi, 'y': yi, 'value': val, 'color': '#cccccc'})
                else:
                    points.append([xi, yi, val])
                    if val > global_max_val_cost_ag2ag:
                        global_max_val_cost_ag2ag = val
            out_cost_ag2ag_dict.setdefault(region_level, {}).setdefault(region, {}).setdefault(cost_type, {})[yr] = {
                'x_categories': all_lus_cost_ag2ag,
                'y_categories': all_lus_cost_ag2ag,
                'data': points,
            }

        global_max_val_cost_ag2ag = round(global_max_val_cost_ag2ag, 2)
        for rl_dict in out_cost_ag2ag_dict.values():
            for region_dict in rl_dict.values():
                for ct_dict in region_dict.values():
                    for leaf in ct_dict.values():
                        leaf['max_val'] = global_max_val_cost_ag2ag

        filename = 'Transition_ag2ag_cost'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_cost_ag2ag_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


    # --------------------- Transition Cost year-by-years (ag2nonag) --------------------
    # JSON structure: region → cost_type → year → {x_categories, y_categories, data, max_val}
    # Cost CSV hierarchy: From-land-use (ag) → To-land-use (non-ag) → Cost-type (no water dims).

    trans_cost_ag2nonag_files = files.query('base_name == "transition_ag2nonag_cost"').reset_index(drop=True)

    if not trans_cost_ag2nonag_files.empty:
        trans_cost_ag2nonag_df = (
            _read_concat(trans_cost_ag2nonag_files['path'])
            .replace(RENAME_AM_NON_AG)
            .infer_objects(copy=False)
            .round({'Cost ($)': 2})
        )
        trans_cost_ag2nonag_df['Year'] = trans_cost_ag2nonag_df['Year'].astype(str)

        non_ag_names_cost = set(RENAME_NON_AG.values())
        from_lus_cost_ag2nonag = set(trans_cost_ag2nonag_df.loc[trans_cost_ag2nonag_df['From-land-use'] != 'ALL', 'From-land-use'].unique())
        to_lus_cost_ag2nonag   = set(trans_cost_ag2nonag_df.loc[trans_cost_ag2nonag_df['To-land-use']   != 'ALL', 'To-land-use'].unique())
        all_y_lus_cost_ag2nonag = sorted(from_lus_cost_ag2nonag - non_ag_names_cost) + ['ALL']   # ag only
        all_x_lus_cost_ag2nonag = sorted(to_lus_cost_ag2nonag & non_ag_names_cost) + ['ALL']     # non-ag only

        out_cost_ag2nonag_dict = {}
        global_max_val_cost_ag2nonag = 0.0
        for (yr, region_level, region, cost_type), grp in trans_cost_ag2nonag_df.groupby(['Year', 'region_level', 'region', 'Cost-type']):
            pivot = grp.pivot_table(
                index='From-land-use', columns='To-land-use',
                values='Cost ($)', aggfunc='sum', fill_value=0,
            )
            lu_x_to_idx = {lu: i for i, lu in enumerate(all_x_lus_cost_ag2nonag)}
            lu_y_to_idx = {lu: i for i, lu in enumerate(all_y_lus_cost_ag2nonag)}
            x_all_idx = len(all_x_lus_cost_ag2nonag) - 1
            y_all_idx = len(all_y_lus_cost_ag2nonag) - 1
            points = []
            for (from_lu, to_lu), val in pivot.stack().items():
                xi, yi = lu_x_to_idx.get(to_lu), lu_y_to_idx.get(from_lu)
                if xi is None or yi is None or val <= 0:
                    continue
                val = round(float(val), 2)
                if xi == x_all_idx or yi == y_all_idx:
                    points.append({'x': xi, 'y': yi, 'value': val, 'color': '#f8f8f8'})
                elif from_lu == to_lu:
                    points.append({'x': xi, 'y': yi, 'value': val, 'color': '#cccccc'})
                else:
                    points.append([xi, yi, val])
                    if val > global_max_val_cost_ag2nonag:
                        global_max_val_cost_ag2nonag = val
            if region_level not in out_cost_ag2nonag_dict:
                out_cost_ag2nonag_dict[region_level] = {}
            if region not in out_cost_ag2nonag_dict[region_level]:
                out_cost_ag2nonag_dict[region_level][region] = {}
            if cost_type not in out_cost_ag2nonag_dict[region_level][region]:
                out_cost_ag2nonag_dict[region_level][region][cost_type] = {}
            out_cost_ag2nonag_dict[region_level][region][cost_type][yr] = {
                'x_categories': all_x_lus_cost_ag2nonag,
                'y_categories': all_y_lus_cost_ag2nonag,
                'data': points,
            }

        global_max_val_cost_ag2nonag = round(global_max_val_cost_ag2nonag, 2)
        for region_level_dict in out_cost_ag2nonag_dict.values():
            for region_dict in region_level_dict.values():
                for ct_dict in region_dict.values():
                    for leaf in ct_dict.values():
                        leaf['max_val'] = global_max_val_cost_ag2nonag

        filename = 'Transition_ag2nonag_cost'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_cost_ag2nonag_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


    return "Transition data processing completed"


def process_biodiversity_data(files, SAVE_DIR):
    """Process and save biodiversity data (Section 6)."""

    bio_rank_dict = {}
    
    # ---------------- Overall quality ----------------
    filter_str = '''
        category == "biodiversity"
        and base_name == "biodiversity_overall_priority_scores"
    '''.strip().replace('\n','')
    
    bio_paths = files.query(filter_str).reset_index(drop=True)
    bio_df = _read_concat(bio_paths['path'])\
        .replace(RENAME_AM_NON_AG)\
        .infer_objects(copy=False)\
        .rename(columns={'Contribution Relative to Base Year Level (%)': 'Value (%)'})\
        .round({'Value (%)': 6})

    all_backends = list(bio_df['backend'].unique())
    default_backend = settings.BIO_QUALITY_LAYER if settings.BIO_QUALITY_LAYER in all_backends else all_backends[0]
    bio_df_default = bio_df[bio_df['backend'] == default_backend]

    # ---------------- Overall quality - Ranking (default backend only → feeds BIO_ranking) -----------------
    bio_rank_type = bio_df_default.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby(['Year', 'region_level', 'region', 'Type'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'region_level', 'Type', 'Value (%)'], ascending=[True, True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'region_level', 'Type']).cumcount())\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))
    bio_rank_total = bio_df_default.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby(['Year', 'region_level', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'region_level', 'Value (%)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount())\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))\
        .assign(Type='Total')

    for (region_level, region), df in bio_rank_total.groupby(['region_level', 'region']):
        df = df.drop(columns=['region_level', 'region'])
        if region_level not in bio_rank_dict:
            bio_rank_dict[region_level] = {}
        if region not in bio_rank_dict[region_level]:
            bio_rank_dict[region_level][region] = {}
        if 'Quality' not in bio_rank_dict[region_level][region]:
            bio_rank_dict[region_level][region]['Quality'] = {}
        bio_rank_dict[region_level][region]['Quality']['Rank']  = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        bio_rank_dict[region_level][region]['Quality']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        bio_rank_dict[region_level][region]['Quality']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

    # ---------------- BIO_quality_ranking.js (default backend only — keeps region→type flat format for Home.js) -----------------
    bio_rank = pd.concat([bio_rank_type, bio_rank_total], axis=0, ignore_index=True)
    ranking_out = {}
    for (region_level, region, _type), df in bio_rank.groupby(['region_level', 'region', 'Type']):
        df = df.drop(columns=['region_level', 'region'])
        if region_level not in ranking_out:
            ranking_out[region_level] = {}
        if region not in ranking_out[region_level]:
            ranking_out[region_level][region] = {}
        if _type not in ranking_out[region_level][region]:
            ranking_out[region_level][region][_type] = {}
        ranking_out[region_level][region][_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        ranking_out[region_level][region][_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        ranking_out[region_level][region][_type]['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

    filename = 'BIO_quality_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(ranking_out, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # ---------------- Overall quality - Overview (backend → region_level → region → {Percent, Area}) ----------------
    overview_out = {}
    for backend in all_backends:
        bio_df_b = bio_df[bio_df["backend"] == backend]
        df_region = bio_df_b.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['region_level', 'region', 'Year', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide_pct = groupby_to_records(df_region, ['region_level', 'Type', 'region'], ['region_level', 'name', 'region', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'
        df_wide_pct['color'] = df_wide_pct['name'].apply(lambda x: COLORS[x])
        df_wide_area = groupby_to_records(df_region, ['region_level', 'Type', 'region'], ['region_level', 'name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'
        df_wide_area['color'] = df_wide_area['name'].apply(lambda x: COLORS[x])
        overview_out[backend] = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region'])

    filename = f'BIO_quality_overview_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(overview_out, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # ---------------- Overall quality - Ag (backend → region_level → region → water → {Percent, Area}) ----------------
    ag_out = {}
    for backend in all_backends:
        bio_df_b = (bio_df[bio_df["backend"] == backend])\
            .query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()
        df_wide_pct = groupby_to_records(bio_df_b, ['region_level', 'region', 'Water_supply', 'Landuse'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        df_wide_area = groupby_to_records(bio_df_b, ['region_level', 'region', 'Water_supply', 'Landuse'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        ag_out[backend] = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'water'])

    filename = f'BIO_quality_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(ag_out, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    # ---------------- Overall quality - Am (backend → region_level → region → am → water → {Percent, Area}) ----------------
    am_out = {}
    for backend in all_backends:
        bio_df_b = bio_df[bio_df["backend"] == backend]
        bio_df_am = bio_df_b.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()
        _am_all_src = bio_df_b.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"')
        df_wide_pct = groupby_to_records(bio_df_am, ['region_level', 'region', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        wall_pct = groupby_to_records(_am_all_src, ['region_level', 'region', 'Water_supply', 'Landuse'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        wall_pct['am'] = 'ALL'; wall_pct['type'] = 'column'; wall_pct['color'] = wall_pct['name'].map(COLORS)
        df_wide_pct = pd.concat([df_wide_pct, wall_pct], ignore_index=True)
        df_wide_area = groupby_to_records(bio_df_am, ['region_level', 'region', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        wall_area = groupby_to_records(_am_all_src, ['region_level', 'region', 'Water_supply', 'Landuse'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        wall_area['am'] = 'ALL'; wall_area['type'] = 'column'; wall_area['color'] = wall_area['name'].map(COLORS)
        df_wide_area = pd.concat([df_wide_area, wall_area], ignore_index=True)
        am_out[backend] = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'am', 'water'])

    filename = f'BIO_quality_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(am_out, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    # ---------------- Overall quality - Non-Ag (backend → region_level → region → {Percent, Area}) ----------------
    nonag_out = {}
    for backend in all_backends:
        bio_df_b = (bio_df[bio_df["backend"] == backend])\
            .query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .query('Type == "Non-Agricultural Land-use"')
        df_wide_pct = groupby_to_records(bio_df_b, ['region_level', 'region', 'Landuse'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        df_wide_area = groupby_to_records(bio_df_b, ['region_level', 'region', 'Landuse'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        nonag_out[backend] = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region'])

    filename = f'BIO_quality_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(nonag_out, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    
    
        
    if settings.GBF2_TARGET != 'off':

        filter_str = '''
            category == "biodiversity" 
            and base_name == "biodiversity_GBF2_priority_scores"
        '''.strip('').replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([pd.read_csv(path) for path in bio_paths['path']])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .round({'Value (%)': 2})

        # ---------------- (GBF2) ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region_level', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'region_level', 'Area Weighted Score (ha)'], ascending=[True, True, False])\
            .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount())\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for (region_level, region), df in bio_rank_total.groupby(['region_level', 'region']):
            df = df.drop(columns=['region_level', 'region'])
            if region_level not in bio_rank_dict:
                bio_rank_dict[region_level] = {}
            if region not in bio_rank_dict[region_level]:
                bio_rank_dict[region_level][region] = {}
            if 'GBF2' not in bio_rank_dict[region_level][region]:
                bio_rank_dict[region_level][region]['GBF2'] = {}

            bio_rank_dict[region_level][region]['GBF2']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF2']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF2']['value'] = df.set_index('Year')['Area Weighted Score (ha)'].apply( lambda x: format_with_suffix(x)).to_dict()
            

        # ---------------- (GBF2) overview  ----------------

        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region_level', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()

        # --- Percent series ---
        df_wide_pct = groupby_to_records(df_region, ['region_level', 'Type', 'region'], ['region_level', 'name', 'region', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'
        _net_pct = df_region.groupby(['Year', 'region_level', 'region'])['Value (%)'].sum().reset_index()
        _net_pct_w = groupby_to_records(_net_pct, ['region_level', 'region'], ['region_level', 'region', 'data'], value_cols=('Year', 'Value (%)'))
        _net_pct_w['name'] = 'Net Value (%)'; _net_pct_w['type'] = 'line'
        df_wide_pct = pd.concat([df_wide_pct, _net_pct_w], ignore_index=True)
        df_wide_pct['color'] = df_wide_pct['name'].apply(lambda x: COLORS[x])

        # --- Area series ---
        df_wide_area = groupby_to_records(df_region, ['region_level', 'Type', 'region'], ['region_level', 'name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'
        _net_area = df_region.groupby(['Year', 'region_level', 'region'])['Area Weighted Score (ha)'].sum().reset_index()
        _net_area_w = groupby_to_records(_net_area, ['region_level', 'region'], ['region_level', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        _net_area_w['name'] = 'Net Value (ha)'; _net_area_w['type'] = 'line'
        df_wide_area = pd.concat([df_wide_area, _net_area_w], ignore_index=True)
        df_wide_area['color'] = df_wide_area['name'].apply(lambda x: COLORS[x])

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region'])

        filename = f'BIO_GBF2_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')



        # ---------------- (GBF2) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        # --- Percent series ---
        df_wide_pct = groupby_to_records(bio_df_ag, ['region_level', 'region', 'Water_supply', 'Landuse'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        net_pct = bio_df_ag.groupby(['Year', 'region_level', 'region', 'Water_supply'])['Value (%)'].sum().reset_index()
        net_pct_w = groupby_to_records(net_pct, ['region_level', 'region', 'Water_supply'], ['region_level', 'region', 'water', 'data'], value_cols=('Year', 'Value (%)'))
        net_pct_w['name'] = 'Net Value (%)'; net_pct_w['type'] = 'line'; net_pct_w['color'] = COLORS['Net Value (%)']
        df_wide_pct = pd.concat([df_wide_pct, net_pct_w], ignore_index=True)

        # --- Area series ---
        df_wide_area = groupby_to_records(bio_df_ag, ['region_level', 'region', 'Water_supply', 'Landuse'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        net_area = bio_df_ag.groupby(['Year', 'region_level', 'region', 'Water_supply'])['Area Weighted Score (ha)'].sum().reset_index()
        net_area_w = groupby_to_records(net_area, ['region_level', 'region', 'Water_supply'], ['region_level', 'region', 'water', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        net_area_w['name'] = 'Net Value (ha)'; net_area_w['type'] = 'line'; net_area_w['color'] = COLORS['Net Value (ha)']
        df_wide_area = pd.concat([df_wide_area, net_area_w], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'water'])

        filename = f'BIO_GBF2_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF2) Ag-Mgt  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()
        _am_all_src = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"')

        # --- Percent series ---
        df_wide_pct = groupby_to_records(bio_df_am, ['region_level', 'region', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        wall_pct = groupby_to_records(_am_all_src, ['region_level', 'region', 'Water_supply', 'Landuse'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        wall_pct['am'] = 'ALL'; wall_pct['type'] = 'column'; wall_pct['color'] = wall_pct['name'].map(COLORS)
        net_am_pct = bio_df_am.groupby(['Year', 'region_level', 'region', 'Water_supply', 'Agricultural Management'])['Value (%)'].sum().reset_index()
        net_am_pct_w = groupby_to_records(net_am_pct, ['region_level', 'region', 'Water_supply', 'Agricultural Management'], ['region_level', 'region', 'water', 'am', 'data'], value_cols=('Year', 'Value (%)'))
        net_am_pct_w['name'] = 'Net Value (%)'; net_am_pct_w['type'] = 'line'; net_am_pct_w['color'] = COLORS['Net Value (%)']
        net_all_pct = _am_all_src.groupby(['Year', 'region_level', 'region', 'Water_supply'])['Value (%)'].sum().reset_index()
        net_all_pct_w = groupby_to_records(net_all_pct, ['region_level', 'region', 'Water_supply'], ['region_level', 'region', 'water', 'data'], value_cols=('Year', 'Value (%)'))
        net_all_pct_w['am'] = 'ALL'; net_all_pct_w['name'] = 'Net Value (%)'; net_all_pct_w['type'] = 'line'; net_all_pct_w['color'] = COLORS['Net Value (%)']
        df_wide_pct = pd.concat([df_wide_pct, wall_pct, net_am_pct_w, net_all_pct_w], ignore_index=True)

        # --- Area series ---
        df_wide_area = groupby_to_records(bio_df_am, ['region_level', 'region', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        wall_area = groupby_to_records(_am_all_src, ['region_level', 'region', 'Water_supply', 'Landuse'], ['region_level', 'region', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        wall_area['am'] = 'ALL'; wall_area['type'] = 'column'; wall_area['color'] = wall_area['name'].map(COLORS)
        net_am_area = bio_df_am.groupby(['Year', 'region_level', 'region', 'Water_supply', 'Agricultural Management'])['Area Weighted Score (ha)'].sum().reset_index()
        net_am_area_w = groupby_to_records(net_am_area, ['region_level', 'region', 'Water_supply', 'Agricultural Management'], ['region_level', 'region', 'water', 'am', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        net_am_area_w['name'] = 'Net Value (ha)'; net_am_area_w['type'] = 'line'; net_am_area_w['color'] = COLORS['Net Value (ha)']
        net_all_area = _am_all_src.groupby(['Year', 'region_level', 'region', 'Water_supply'])['Area Weighted Score (ha)'].sum().reset_index()
        net_all_area_w = groupby_to_records(net_all_area, ['region_level', 'region', 'Water_supply'], ['region_level', 'region', 'water', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        net_all_area_w['am'] = 'ALL'; net_all_area_w['name'] = 'Net Value (ha)'; net_all_area_w['type'] = 'line'; net_all_area_w['color'] = COLORS['Net Value (ha)']
        df_wide_area = pd.concat([df_wide_area, wall_area, net_am_area_w, net_all_area_w], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'am', 'water'])

        filename = f'BIO_GBF2_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF2) Non-Ag  ----------------
        _g2_nonag_src = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')

        # --- Percent series ---
        df_wide_pct = groupby_to_records(_g2_nonag_src, ['region_level', 'region', 'Landuse'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        net_pct = _g2_nonag_src.groupby(['Year', 'region_level', 'region'])['Value (%)'].sum().reset_index()
        net_pct_w = groupby_to_records(net_pct, ['region_level', 'region'], ['region_level', 'region', 'data'], value_cols=('Year', 'Value (%)'))
        net_pct_w['name'] = 'Net Value (%)'; net_pct_w['type'] = 'line'; net_pct_w['color'] = COLORS['Net Value (%)']
        df_wide_pct = pd.concat([df_wide_pct, net_pct_w], ignore_index=True)

        # --- Area series ---
        df_wide_area = groupby_to_records(_g2_nonag_src, ['region_level', 'region', 'Landuse'], ['region_level', 'region', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        net_area = _g2_nonag_src.groupby(['Year', 'region_level', 'region'])['Area Weighted Score (ha)'].sum().reset_index()
        net_area_w = groupby_to_records(net_area, ['region_level', 'region'], ['region_level', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        net_area_w['name'] = 'Net Value (ha)'; net_area_w['type'] = 'line'; net_area_w['color'] = COLORS['Net Value (ha)']
        df_wide_area = pd.concat([df_wide_area, net_area_w], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region'])

        filename = f'BIO_GBF2_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
            
            
    # Shared Type → display-name mapping for biodiversity Sum charts (used by GBF3/4 blocks)
    _SUM_TYPE_DISPLAY = {
        'ag': 'Agricultural Land-use',
        'non-ag': 'Non-Agricultural Land-use',
        'ag-man': 'Agricultural Management',
        'Outside LUTO study area': 'Outside LUTO study area',
    }

    # A WRITE_* setting alone does not guarantee files exist: in 'selected' mode the writer emits
    # nothing when the matching target is off (nothing is constrained, so there is no selection to
    # write). Reading that empty set gives a frame with no 'species' column, and the queries below then
    # fail with an opaque NameError -- AFTER the whole simulation has solved. Guard on the files being
    # present, as the region-species blocks further down already do via os.path.exists.
    _nvis_written = not files.query('base_name.str.contains("biodiversity_GBF3_NVIS_scores")').empty
    if settings.WRITE_GBF3_NVIS != 'off' and _nvis_written:
        filter_str = '''
            category == "biodiversity"
            and base_name.str.contains("biodiversity_GBF3_NVIS_scores")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = _read_concat(bio_paths['path'], ignore_index=False)
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Vegetation Group': 'species'})\
            .round(6)
        # Drop the per-species 'ALL' aggregate (it's re-aggregated explicitly in sum charts).
        # Keep AUSTRALIA rows so the AUSTRALIA region selection shows data in the report.
        bio_df = bio_df.query('species != "ALL"')
        _nvis_species_order = (_paged_species_order(bio_paths, 'xr_biodiversity_GBF3_NVIS_ag')
                               or sorted(bio_df['species'].unique().tolist()))

        # ---------------- (GBF3-NVIS) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL" and region != "AUSTRALIA"')\
            .groupby(['Year', 'region_level', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'region_level', 'Area Weighted Score (ha)'], ascending=[True, True, False])\
            .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))
            
        for (region_level, region), df in bio_rank_total.groupby(['region_level', 'region']):
            df = df.drop(columns=['region_level', 'region'])
            if region_level not in bio_rank_dict:
                bio_rank_dict[region_level] = {}
            if region not in bio_rank_dict[region_level]:
                bio_rank_dict[region_level][region] = {}
            if 'GBF3-NVIS' not in bio_rank_dict[region_level][region]:
                bio_rank_dict[region_level][region]['GBF3-NVIS'] = {}
                
            bio_rank_dict[region_level][region]['GBF3-NVIS']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF3-NVIS']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF3-NVIS']['value'] = df.set_index('Year')['Area Weighted Score (ha)'].apply(lambda x: format_with_suffix(x)).to_dict()


        # ---------------- (GBF3-NVIS) Overview  ----------------

        # Build target lookup once (species × region → Target_by_Percent, BASE_TOTAL_SCORE).
        # Target_by_Percent is NaN when no constraint is active (write.py sets it to NaN
        # when TARGET_INSIDE_SCORE = 0), so .notna() correctly selects only real targets.
        _gbf3_target_lk = (
            bio_df[bio_df['Target_by_Percent'].notna()]
            [['species', 'region', 'region_level', 'Target_by_Percent', 'BASE_TOTAL_SCORE']]
            .drop_duplicates(['species', 'region', 'region_level'])
        )


        # Sum-tab: normalise by ALL_HA (total pre-1750 baseline across all veg groups) so
        # the stacked bar shows sum(area)/ALL_HA*100, not a meaningless sum of per-group %.
        # Inside rows: exclude 'ALL' aggregates (keep leaf lm × lu combinations only).
        # Outside rows: use only the am='ALL', lm='ALL' aggregate row to avoid cross-join
        # inflation (each group's outside area is replicated across every am×lm combo).
        _inside = bio_df.query(
            'Type != "Outside LUTO study area" and species != "ALL" '
            'and Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"'
        )
        _outside = bio_df.query(
            'Type == "Outside LUTO study area" and Water_supply == "ALL" and `Agricultural Management` == "ALL"'
        )
        df_region = (
            pd.concat([_inside, _outside])
            .groupby(['Year', 'region_level', 'region', 'species', 'Type'])
            .agg({'Area Weighted Score (ha)': 'sum', 'ALL_HA': 'first'})
            .reset_index()
            .assign(**{'Sum_Pct (%)': lambda d: d['Area Weighted Score (ha)'] / d['ALL_HA'] * 100})
        )
        df_wide_pct = groupby_to_records(df_region, ['region_level', 'Type', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'], value_cols=('Year', 'Sum_Pct (%)'))
        df_wide_pct['type'] = 'column'
        df_wide_pct['color'] = df_wide_pct['name'].apply(lambda x: COLORS[x])

        df_wide_area = groupby_to_records(df_region, ['region_level', 'Type', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'
        df_wide_area['color'] = df_wide_area['name'].apply(lambda x: COLORS[x])

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species'])

        _write_paged_chart_js(out_dict, 'BIO_GBF3_NVIS_overview_sum', SAVE_DIR, species_order=_nvis_species_order)

        # --- BIO_GBF3_NVIS_Sum: per-species Type breakdown from pre-computed sum CSV ---
        sum_bio_paths = files.query(
            'category == "biodiversity" and base_name.str.contains("biodiversity_GBF3_NVIS_sum_scores")'
        )
        if not sum_bio_paths.empty:
            sum_bio_df = pd.concat(
                [df for p in sum_bio_paths['path'] if not (df := pd.read_csv(p, low_memory=False)).empty],
                ignore_index=True,
            ).rename(columns={'Vegetation Group': 'species'})
            sum_bio_df = sum_bio_df[sum_bio_df['Type'] != 'ALL'].copy()
            sum_bio_df['name'] = sum_bio_df['Type'].map(_SUM_TYPE_DISPLAY).fillna(sum_bio_df['Type'])
            df_wide_sum_pct = groupby_to_records(sum_bio_df, ['region_level', 'name', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'],
                value_cols=('Year', 'Relative_Contribution_Percentage'),
            )
            df_wide_sum_pct['type'] = 'column'
            df_wide_sum_pct['color'] = df_wide_sum_pct['name'].apply(lambda x: COLORS[x])

            df_wide_sum_area = groupby_to_records(sum_bio_df, ['region_level', 'name', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'],
                value_cols=('Year', 'Area Weighted Score (ha)'),
            )
            df_wide_sum_area['type'] = 'column'
            df_wide_sum_area['color'] = df_wide_sum_area['name'].apply(lambda x: COLORS[x])

            _sum_years = sorted(sum_bio_df['Year'].unique().tolist())
            _target_color = COLORS.get('Target (%)', '#040404')

            out_dict_sum = {}
            for (region_level, region, species), df_pct in df_wide_sum_pct.groupby(['region_level', 'region', 'species']):
                df_pct = df_pct.drop(['region_level', 'region', 'species'], axis=1)
                df_area = df_wide_sum_area[(df_wide_sum_area['region_level'] == region_level) & (df_wide_sum_area['region'] == region) & (df_wide_sum_area['species'] == species)].drop(['region_level', 'region', 'species'], axis=1)
                pct_records = df_pct.to_dict(orient='records')
                area_records = df_area.to_dict(orient='records')
                # Add target line if this species×region has a constraint target
                _trow = _gbf3_target_lk[(_gbf3_target_lk['species'] == species) & (_gbf3_target_lk['region'] == region) & (_gbf3_target_lk['region_level'] == region_level)]
                if not _trow.empty:
                    t_pct = float(_trow['Target_by_Percent'].iloc[0])
                    t_area = t_pct / 100 * float(_trow['BASE_TOTAL_SCORE'].iloc[0])
                    pct_records = pct_records + [{'name': 'Target (%)', 'type': 'line', 'color': _target_color, 'data': [[yr, t_pct] for yr in _sum_years]}]
                    area_records = area_records + [{'name': 'Target (ha)', 'type': 'line', 'color': _target_color, 'data': [[yr, t_area] for yr in _sum_years]}]
                if region_level not in out_dict_sum:
                    out_dict_sum[region_level] = {}
                if region not in out_dict_sum[region_level]:
                    out_dict_sum[region_level][region] = {}
                out_dict_sum[region_level][region][species] = {
                    'Percent': pct_records,
                    'Area':    area_records,
                }
            _write_paged_chart_js(out_dict_sum, 'BIO_GBF3_NVIS_Sum', SAVE_DIR, species_order=_nvis_species_order)


        # ---------------- (GBF3-NVIS) - Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide_pct = groupby_to_records(bio_df_ag, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        df_wide_pct = pd.concat([df_wide_pct, bio_outside_series(bio_df, 'Ag')], ignore_index=True)

        df_wide_area = groupby_to_records(bio_df_ag, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        df_wide_area = pd.concat([df_wide_area, bio_outside_series(bio_df, 'Ag', value_col='Area Weighted Score (ha)')], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species', 'water'])

        _write_paged_chart_js(out_dict, 'BIO_GBF3_NVIS_Ag', SAVE_DIR, species_order=_nvis_species_order)

        # ---------------- (GBF3-NVIS) - Am  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()
        _bio_df_am_all = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"')

        df_wide_pct = groupby_to_records(bio_df_am, ['region_level', 'region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'species', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        wall_pct = groupby_to_records(_bio_df_am_all, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        wall_pct['am'] = 'ALL'; wall_pct['type'] = 'column'; wall_pct['color'] = wall_pct['name'].map(COLORS)
        df_wide_pct = pd.concat([df_wide_pct, wall_pct, bio_outside_series(bio_df, 'Am')], ignore_index=True)

        df_wide_area = groupby_to_records(bio_df_am, ['region_level', 'region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'species', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        wall_area = groupby_to_records(_bio_df_am_all, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        wall_area['am'] = 'ALL'; wall_area['type'] = 'column'; wall_area['color'] = wall_area['name'].map(COLORS)
        df_wide_area = pd.concat([df_wide_area, wall_area, bio_outside_series(bio_df, 'Am', value_col='Area Weighted Score (ha)')], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species', 'am', 'water'])

        _write_paged_chart_js(out_dict, 'BIO_GBF3_NVIS_Am', SAVE_DIR, species_order=_nvis_species_order)

        # ---------------- (GBF3-NVIS) - Non-Ag  ----------------
        _g3_nonag_src = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')

        df_wide_pct = groupby_to_records(_g3_nonag_src, ['region_level', 'region', 'species', 'Landuse'], ['region_level', 'region', 'species', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        df_wide_pct = pd.concat([df_wide_pct, bio_outside_series(bio_df, 'NonAg')], ignore_index=True)

        df_wide_area = groupby_to_records(_g3_nonag_src, ['region_level', 'region', 'species', 'Landuse'], ['region_level', 'region', 'species', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        df_wide_area = pd.concat([df_wide_area, bio_outside_series(bio_df, 'NonAg', value_col='Area Weighted Score (ha)')], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species'])

        _write_paged_chart_js(out_dict, 'BIO_GBF3_NVIS_NonAg', SAVE_DIR, species_order=_nvis_species_order)
            
            
    # IBRA reporting branch disabled (GBF3 IBRA pipeline incomplete).


    # A WRITE_* setting alone does not guarantee files exist: in 'selected' mode the writer emits
    # nothing when the matching target is off (nothing is constrained, so there is no selection to
    # write). Reading that empty set gives a frame with no 'species' column, and the queries below then
    # fail with an opaque NameError -- AFTER the whole simulation has solved. Guard on the files being
    # present, as the region-species blocks further down already do via os.path.exists.
    _snes_written = not files.query('base_name.str.contains("biodiversity_GBF4_SNES_scores")').empty
    if settings.WRITE_GBF4_SNES != 'off' and _snes_written:

        filter_str = '''
            category == "biodiversity"
            and base_name.str.contains("biodiversity_GBF4_SNES_scores")
        '''.strip().replace('\n', '')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = _read_concat(bio_paths['path'], ignore_index=False)
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .round(6)
        # Drop the per-species 'ALL' aggregate so it is not surfaced as a selectable
        # species in the report dropdowns; sum charts re-aggregate explicitly.
        # Keep AUSTRALIA rows so the AUSTRALIA region selection shows data in the report.
        bio_df = bio_df.query('species != "ALL"')
        _snes_species_order = (_paged_species_order(bio_paths, 'xr_biodiversity_GBF4_SNES_ag')
                               or sorted(bio_df['species'].unique().tolist()))
        # ---------------- (GBF4 SNES) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL" and region != "AUSTRALIA"')\
            .groupby(['Year', 'region_level', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'region_level', 'Value (%)'], ascending=[True, True, False])\
            .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))
            
        for (region_level, region), df in bio_rank_total.groupby(['region_level', 'region']):
            df = df.drop(columns=['region_level', 'region'])
            if region_level not in bio_rank_dict:
                bio_rank_dict[region_level] = {}
            if region not in bio_rank_dict[region_level]:
                bio_rank_dict[region_level][region] = {}
            if 'GBF4 (SNES)' not in bio_rank_dict[region_level][region]:
                bio_rank_dict[region_level][region]['GBF4 (SNES)'] = {}

            bio_rank_dict[region_level][region]['GBF4 (SNES)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF4 (SNES)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF4 (SNES)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()



        # ---------------- (GBF4 SNES) Overview  ----------------

        # sum: normalise by ALL_HA so the chart shows sum(area)/ALL_HA*100 not sum of per-species %.
        _inside = bio_df.query(
            'Type != "Outside LUTO study area" and species != "ALL" '
            'and Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"'
        )
        _outside = bio_df.query(
            'Type == "Outside LUTO study area" and Water_supply == "ALL" and `Agricultural Management` == "ALL"'
        )
        df_region = (
            pd.concat([_inside, _outside])
            .groupby(['Year', 'region_level', 'region', 'species', 'Type'])
            .agg({'Area Weighted Score (ha)': 'sum', 'ALL_HA': 'first'})
            .reset_index()
            .assign(**{'Sum_Pct (%)': lambda d: d['Area Weighted Score (ha)'] / d['ALL_HA'] * 100})
        )
        df_wide_pct = groupby_to_records(df_region, ['region_level', 'Type', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'], value_cols=('Year', 'Sum_Pct (%)'))
        df_wide_pct['type'] = 'column'
        df_wide_pct['color'] = df_wide_pct['name'].apply(lambda x: COLORS[x])

        df_wide_area = groupby_to_records(df_region, ['region_level', 'Type', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'
        df_wide_area['color'] = df_wide_area['name'].apply(lambda x: COLORS[x])

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species'])

        _write_paged_chart_js(out_dict, 'BIO_GBF4_SNES_overview_sum', SAVE_DIR, species_order=_snes_species_order)

        # --- BIO_GBF4_SNES_Sum: per-species Type breakdown from pre-computed sum CSV ---
        sum_bio_paths = files.query(
            'category == "biodiversity" and base_name.str.contains("biodiversity_GBF4_SNES_sum_scores")'
        )
        if not sum_bio_paths.empty:
            sum_bio_df = pd.concat(
                [df for p in sum_bio_paths['path'] if not (df := pd.read_csv(p, low_memory=False)).empty],
                ignore_index=True,
            )
            sum_bio_df = sum_bio_df[sum_bio_df['Type'] != 'ALL'].copy()
            sum_bio_df['name'] = sum_bio_df['Type'].map(_SUM_TYPE_DISPLAY).fillna(sum_bio_df['Type'])
            df_wide_sum_pct = groupby_to_records(sum_bio_df, ['region_level', 'name', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'],
                value_cols=('Year', 'Relative_Contribution_Percentage'),
            )
            df_wide_sum_pct['type'] = 'column'
            df_wide_sum_pct['color'] = df_wide_sum_pct['name'].apply(lambda x: COLORS[x])
            df_wide_sum_area = groupby_to_records(sum_bio_df, ['region_level', 'name', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'],
                value_cols=('Year', 'Area Weighted Score (ha)'),
            )
            df_wide_sum_area['type'] = 'column'
            df_wide_sum_area['color'] = df_wide_sum_area['name'].apply(lambda x: COLORS[x])
            out_dict_sum = build_out_dict_bulk(df_wide_sum_pct, df_wide_sum_area, ['region_level', 'region', 'species'])
            _write_paged_chart_js(out_dict_sum, 'BIO_GBF4_SNES_Sum', SAVE_DIR, species_order=_snes_species_order)

        # ---------------- (GBF4 SNES) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide_pct = groupby_to_records(bio_df_ag, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        df_wide_pct = pd.concat([df_wide_pct, bio_outside_series(bio_df, 'Ag')], ignore_index=True)

        df_wide_area = groupby_to_records(bio_df_ag, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        df_wide_area = pd.concat([df_wide_area, bio_outside_series(bio_df, 'Ag', value_col='Area Weighted Score (ha)')], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species', 'water'])

        _write_paged_chart_js(out_dict, 'BIO_GBF4_SNES_Ag', SAVE_DIR, species_order=_snes_species_order)

        # ---------------- (GBF4 SNES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()
        _bio_df_snes_am_all = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"')

        df_wide_pct = groupby_to_records(bio_df_am, ['region_level', 'region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'species', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        wall_pct = groupby_to_records(_bio_df_snes_am_all, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        wall_pct['am'] = 'ALL'; wall_pct['type'] = 'column'; wall_pct['color'] = wall_pct['name'].map(COLORS)
        df_wide_pct = pd.concat([df_wide_pct, wall_pct, bio_outside_series(bio_df, 'Am')], ignore_index=True)

        df_wide_area = groupby_to_records(bio_df_am, ['region_level', 'region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'species', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        wall_area = groupby_to_records(_bio_df_snes_am_all, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        wall_area['am'] = 'ALL'; wall_area['type'] = 'column'; wall_area['color'] = wall_area['name'].map(COLORS)
        df_wide_area = pd.concat([df_wide_area, wall_area, bio_outside_series(bio_df, 'Am', value_col='Area Weighted Score (ha)')], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species', 'am', 'water'])

        _write_paged_chart_js(out_dict, 'BIO_GBF4_SNES_Am', SAVE_DIR, species_order=_snes_species_order)

        # ---------------- (GBF4 SNES) Non-ag  ----------------
        _g4s_nonag_src = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')

        df_wide_pct = groupby_to_records(_g4s_nonag_src, ['region_level', 'region', 'species', 'Landuse'], ['region_level', 'region', 'species', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        df_wide_pct = pd.concat([df_wide_pct, bio_outside_series(bio_df, 'NonAg')], ignore_index=True)

        df_wide_area = groupby_to_records(_g4s_nonag_src, ['region_level', 'region', 'species', 'Landuse'], ['region_level', 'region', 'species', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        df_wide_area = pd.concat([df_wide_area, bio_outside_series(bio_df, 'NonAg', value_col='Area Weighted Score (ha)')], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species'])

        _write_paged_chart_js(out_dict, 'BIO_GBF4_SNES_NonAg', SAVE_DIR, species_order=_snes_species_order)
            
            
            
    # A WRITE_* setting alone does not guarantee files exist: in 'selected' mode the writer emits
    # nothing when the matching target is off (nothing is constrained, so there is no selection to
    # write). Reading that empty set gives a frame with no 'species' column, and the queries below then
    # fail with an opaque NameError -- AFTER the whole simulation has solved. Guard on the files being
    # present, as the region-species blocks further down already do via os.path.exists.
    _ecnes_written = not files.query('base_name.str.contains("biodiversity_GBF4_ECNES_scores")').empty
    if settings.WRITE_GBF4_ECNES != 'off' and _ecnes_written:
        #---------------- (GBF4 ECNES) ----------------
        bio_paths = files.query('base_name.str.contains("biodiversity_GBF4_ECNES_scores")')
        bio_df = _read_concat(bio_paths['path'], ignore_index=False)
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={
                'Contribution Relative to Pre-1750 Level (%)': 'Value (%)',
                'Target by Percent (%)': 'Target_by_Percent',
            })\
            .round(6)
        # Drop the per-species 'ALL' aggregate (re-aggregated explicitly in sum charts).
        # Keep AUSTRALIA rows so the AUSTRALIA region selection shows data in the report.
        bio_df = bio_df.query('species != "ALL"')
        _ecnes_species_order = (_paged_species_order(bio_paths, 'xr_biodiversity_GBF4_ECNES_ag')
                                or sorted(bio_df['species'].unique().tolist()))

        # Build target lookup once (species × region → Target_by_Percent, BASE_TOTAL_SCORE).
        # Target_by_Percent is NaN when no constraint is active (write.py sets it to NaN
        # when TARGET_INSIDE_SCORE = 0), so .notna() correctly selects only real targets.
        _ecnes_target_lk = (
            bio_df[bio_df['Target_by_Percent'].notna()]
            [['species', 'region', 'region_level', 'Target_by_Percent', 'BASE_TOTAL_SCORE']]
            .drop_duplicates(['species', 'region', 'region_level'])
        )

        # ---------------- (GBF4 ECNES) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL" and region != "AUSTRALIA"')\
            .groupby(['Year', 'region_level', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'region_level', 'Value (%)'], ascending=[True, True, False])\
            .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for (region_level, region), df in bio_rank_total.groupby(['region_level', 'region']):
            df = df.drop(columns=['region_level', 'region'])
            if region_level not in bio_rank_dict:
                bio_rank_dict[region_level] = {}
            if region not in bio_rank_dict[region_level]:
                bio_rank_dict[region_level][region] = {}
            if 'GBF4 (ECNES)' not in bio_rank_dict[region_level][region]:
                bio_rank_dict[region_level][region]['GBF4 (ECNES)'] = {}

            bio_rank_dict[region_level][region]['GBF4 (ECNES)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF4 (ECNES)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF4 (ECNES)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

        # ---------------- (GBF4 ECNES) Overview  ----------------

        # sum: normalise by ALL_HA so the chart shows sum(area)/ALL_HA*100 not sum of per-community %.
        _inside = bio_df.query(
            'Type != "Outside LUTO study area" and species != "ALL" '
            'and Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"'
        )
        _outside = bio_df.query(
            'Type == "Outside LUTO study area" and Water_supply == "ALL" and `Agricultural Management` == "ALL"'
        )
        df_region = (
            pd.concat([_inside, _outside])
            .groupby(['Year', 'region_level', 'region', 'species', 'Type'])
            .agg({'Area Weighted Score (ha)': 'sum', 'ALL_HA': 'first'})
            .reset_index()
            .assign(**{'Sum_Pct (%)': lambda d: d['Area Weighted Score (ha)'] / d['ALL_HA'] * 100})
        )
        df_wide_pct = groupby_to_records(df_region, ['region_level', 'Type', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'], value_cols=('Year', 'Sum_Pct (%)'))
        df_wide_pct['type'] = 'column'
        df_wide_pct['color'] = df_wide_pct['name'].apply(lambda x: COLORS[x])

        df_wide_area = groupby_to_records(df_region, ['region_level', 'Type', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'
        df_wide_area['color'] = df_wide_area['name'].apply(lambda x: COLORS[x])

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species'])

        _write_paged_chart_js(out_dict, 'BIO_GBF4_ECNES_overview_sum', SAVE_DIR, species_order=_ecnes_species_order)

        # --- BIO_GBF4_ECNES_Sum: per-species Type breakdown from pre-computed sum CSV ---
        sum_bio_paths = files.query(
            'category == "biodiversity" and base_name.str.contains("biodiversity_GBF4_ECNES_sum_scores")'
        )
        if not sum_bio_paths.empty:
            sum_bio_df = pd.concat(
                [df for p in sum_bio_paths['path'] if not (df := pd.read_csv(p, low_memory=False)).empty],
                ignore_index=True,
            )
            sum_bio_df = sum_bio_df[sum_bio_df['Type'] != 'ALL'].copy()
            sum_bio_df['name'] = sum_bio_df['Type'].map(_SUM_TYPE_DISPLAY).fillna(sum_bio_df['Type'])
            sum_bio_df['_Pct_AllHA'] = sum_bio_df['Area Weighted Score (ha)'] / sum_bio_df['ALL_HA'] * 100
            df_wide_sum_pct = groupby_to_records(sum_bio_df, ['region_level', 'name', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'],
                value_cols=('Year', '_Pct_AllHA'),
            )
            df_wide_sum_pct['type'] = 'column'
            df_wide_sum_pct['color'] = df_wide_sum_pct['name'].apply(lambda x: COLORS[x])

            df_wide_sum_area = groupby_to_records(sum_bio_df, ['region_level', 'name', 'region', 'species'], ['region_level', 'name', 'region', 'species', 'data'],
                value_cols=('Year', 'Area Weighted Score (ha)'),
            )
            df_wide_sum_area['type'] = 'column'
            df_wide_sum_area['color'] = df_wide_sum_area['name'].apply(lambda x: COLORS[x])

            _ecnes_sum_years = sorted(sum_bio_df['Year'].unique().tolist())
            _ecnes_target_color = COLORS.get('Target (%)', '#040404')

            out_dict_sum = {}
            for (region_level, region, species), df_pct in df_wide_sum_pct.groupby(['region_level', 'region', 'species']):
                df_pct = df_pct.drop(['region_level', 'region', 'species'], axis=1)
                df_area = df_wide_sum_area[(df_wide_sum_area['region_level'] == region_level) & (df_wide_sum_area['region'] == region) & (df_wide_sum_area['species'] == species)].drop(['region_level', 'region', 'species'], axis=1)
                pct_records = df_pct.to_dict(orient='records')
                area_records = df_area.to_dict(orient='records')
                _trow = _ecnes_target_lk[(_ecnes_target_lk['species'] == species) & (_ecnes_target_lk['region'] == region) & (_ecnes_target_lk['region_level'] == region_level)]
                if not _trow.empty:
                    t_pct = float(_trow['Target_by_Percent'].iloc[0])
                    t_area = t_pct / 100 * float(_trow['BASE_TOTAL_SCORE'].iloc[0])
                    pct_records = pct_records + [{'name': 'Target (%)', 'type': 'line', 'color': _ecnes_target_color, 'data': [[yr, t_pct] for yr in _ecnes_sum_years]}]
                    area_records = area_records + [{'name': 'Target (ha)', 'type': 'line', 'color': _ecnes_target_color, 'data': [[yr, t_area] for yr in _ecnes_sum_years]}]
                if region_level not in out_dict_sum:
                    out_dict_sum[region_level] = {}
                if region not in out_dict_sum[region_level]:
                    out_dict_sum[region_level][region] = {}
                out_dict_sum[region_level][region][species] = {
                    'Percent': pct_records,
                    'Area':    area_records,
                }
            _write_paged_chart_js(out_dict_sum, 'BIO_GBF4_ECNES_Sum', SAVE_DIR, species_order=_ecnes_species_order)

        # ---------------- (GBF4 ECNES) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide_pct = groupby_to_records(bio_df_ag, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        df_wide_pct = pd.concat([df_wide_pct, bio_outside_series(bio_df, 'Ag')], ignore_index=True)

        df_wide_area = groupby_to_records(bio_df_ag, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        df_wide_area = pd.concat([df_wide_area, bio_outside_series(bio_df, 'Ag', value_col='Area Weighted Score (ha)')], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species', 'water'])

        _write_paged_chart_js(out_dict, 'BIO_GBF4_ECNES_Ag', SAVE_DIR, species_order=_ecnes_species_order)

        # ---------------- (GBF4 ECNES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()
        _bio_df_ecnes_am_all = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"')

        df_wide_pct = groupby_to_records(bio_df_am, ['region_level', 'region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'species', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        wall_pct = groupby_to_records(_bio_df_ecnes_am_all, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        wall_pct['am'] = 'ALL'; wall_pct['type'] = 'column'; wall_pct['color'] = wall_pct['name'].map(COLORS)
        df_wide_pct = pd.concat([df_wide_pct, wall_pct, bio_outside_series(bio_df, 'Am')], ignore_index=True)

        df_wide_area = groupby_to_records(bio_df_am, ['region_level', 'region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'species', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        wall_area = groupby_to_records(_bio_df_ecnes_am_all, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        wall_area['am'] = 'ALL'; wall_area['type'] = 'column'; wall_area['color'] = wall_area['name'].map(COLORS)
        df_wide_area = pd.concat([df_wide_area, wall_area, bio_outside_series(bio_df, 'Am', value_col='Area Weighted Score (ha)')], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species', 'am', 'water'])

        _write_paged_chart_js(out_dict, 'BIO_GBF4_ECNES_Am', SAVE_DIR, species_order=_ecnes_species_order)

        # ---------------- (GBF4 ECNES) Non-ag  ----------------
        _g4e_nonag_src = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')

        df_wide_pct = groupby_to_records(_g4e_nonag_src, ['region_level', 'region', 'species', 'Landuse'], ['region_level', 'region', 'species', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide_pct['type'] = 'column'; df_wide_pct['color'] = df_wide_pct['name'].map(COLORS)
        df_wide_pct['_ord'] = df_wide_pct['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_pct = df_wide_pct.sort_values('_ord').drop(columns=['_ord'])
        df_wide_pct = pd.concat([df_wide_pct, bio_outside_series(bio_df, 'NonAg')], ignore_index=True)

        df_wide_area = groupby_to_records(_g4e_nonag_src, ['region_level', 'region', 'species', 'Landuse'], ['region_level', 'region', 'species', 'name', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide_area['type'] = 'column'; df_wide_area['color'] = df_wide_area['name'].map(COLORS)
        df_wide_area['_ord'] = df_wide_area['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide_area = df_wide_area.sort_values('_ord').drop(columns=['_ord'])
        df_wide_area = pd.concat([df_wide_area, bio_outside_series(bio_df, 'NonAg', value_col='Area Weighted Score (ha)')], ignore_index=True)

        out_dict = build_out_dict_bulk(df_wide_pct, df_wide_area, ['region_level', 'region', 'species'])

        _write_paged_chart_js(out_dict, 'BIO_GBF4_ECNES_NonAg', SAVE_DIR, species_order=_ecnes_species_order)
    
    
    

    if settings.GBF8_TARGET == 'on':
        
        filter_str = '''
            category == "biodiversity" 
            
            and base_name.str.contains("biodiversity_GBF8_species_scores")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = _read_concat(bio_paths['path'], ignore_index=False)
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Species':'species'})\
            .round(6)

        # ---------------- (GBF8 SPECIES) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region_level', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'region_level', 'Value (%)'], ascending=[True, True, False])\
            .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for (region_level, region), df in bio_rank_total.groupby(['region_level', 'region']):
            df = df.drop(columns=['region_level', 'region'])
            if region_level not in bio_rank_dict:
                bio_rank_dict[region_level] = {}
            if region not in bio_rank_dict[region_level]:
                bio_rank_dict[region_level][region] = {}
            if 'GBF8 (SPECIES)' not in bio_rank_dict[region_level][region]:
                bio_rank_dict[region_level][region]['GBF8 (SPECIES)'] = {}

            bio_rank_dict[region_level][region]['GBF8 (SPECIES)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF8 (SPECIES)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF8 (SPECIES)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

        # ---------------- (GBF8 SPECIES) Overview  ----------------

        # sum
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region_level', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = groupby_to_records(df_region, ['region_level', 'Type', 'region'], ['region_level', 'name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        out_dict = {}
        for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
            df = df.drop(['region_level', 'region'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            out_dict[region_level][region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 SPECIES) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide = groupby_to_records(bio_df_ag, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide['type'] = 'column'; df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['_ord'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('_ord').drop(columns=['_ord'])
        out_dict = {}
        for (region_level, region, species, water), df in df_wide.groupby(['region_level', 'region', 'species', 'water']):
            df = df.drop(['region_level', 'region', 'species', 'water'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            if region not in out_dict[region_level]:
                out_dict[region_level][region] = {}
            if species not in out_dict[region_level][region]:
                out_dict[region_level][region][species] = {}
            out_dict[region_level][region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 SPECIES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()

        df_wide = groupby_to_records(bio_df_am, ['region_level', 'region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'species', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide['type'] = 'column'; df_wide['color'] = df_wide['name'].map(COLORS)
        wall = groupby_to_records(bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"'), ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        wall['am'] = 'ALL'; wall['type'] = 'column'; wall['color'] = wall['name'].map(COLORS)
        df_wide = pd.concat([df_wide, wall], ignore_index=True)
        out_dict = {}
        for (region_level, region, species, am, water), df in df_wide.groupby(['region_level', 'region', 'species', 'am', 'water']):
            df = df.drop(['region_level', 'region', 'species', 'am', 'water'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            if region not in out_dict[region_level]:
                out_dict[region_level][region] = {}
            if species not in out_dict[region_level][region]:
                out_dict[region_level][region][species] = {}
            if am not in out_dict[region_level][region][species]:
                out_dict[region_level][region][species][am] = {}
            out_dict[region_level][region][species][am][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 SPECIES) Non-ag  ----------------
        _g8sp_nonag_src = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')

        df_wide = groupby_to_records(_g8sp_nonag_src, ['region_level', 'region', 'species', 'Landuse'], ['region_level', 'region', 'species', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide['type'] = 'column'; df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['_ord'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('_ord').drop(columns=['_ord'])
        out_dict = {}
        for (region_level, region, species), df in df_wide.groupby(['region_level', 'region', 'species']):
            df = df.drop(['region_level', 'region', 'species'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            if region not in out_dict[region_level]:
                out_dict[region_level][region] = {}
            out_dict[region_level][region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
        
        
    
        # ---------------- (GBF8 GROUP)  ----------------
        bio_paths = files.query('base_name.str.contains("biodiversity_GBF8_groups_scores")')
        bio_df = _read_concat(bio_paths['path'], ignore_index=False)
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Group':'species'})\
            .round(6)

        # ---------------- (GBF8 GROUP) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region_level', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'region_level', 'Value (%)'], ascending=[True, True, False])\
            .assign(Rank=lambda x: x.groupby(['Year', 'region_level']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for (region_level, region), df in bio_rank_total.groupby(['region_level', 'region']):
            df = df.drop(columns=['region_level', 'region'])
            if region_level not in bio_rank_dict:
                bio_rank_dict[region_level] = {}
            if region not in bio_rank_dict[region_level]:
                bio_rank_dict[region_level][region] = {}
            if 'GBF8 (GROUP)' not in bio_rank_dict[region_level][region]:
                bio_rank_dict[region_level][region]['GBF8 (GROUP)'] = {}

            bio_rank_dict[region_level][region]['GBF8 (GROUP)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF8 (GROUP)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region_level][region]['GBF8 (GROUP)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

        # ---------------- (GBF8 GROUP) Overview  ----------------

        # sum
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region_level', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = groupby_to_records(df_region, ['region_level', 'Type', 'region'], ['region_level', 'name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        out_dict = {}
        for (region_level, region), df in df_wide.groupby(['region_level', 'region']):
            df = df.drop(['region_level', 'region'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            out_dict[region_level][region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 GROUP) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide = groupby_to_records(bio_df_ag, ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide['type'] = 'column'; df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['_ord'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('_ord').drop(columns=['_ord'])
        out_dict = {}
        for (region_level, region, species, water), df in df_wide.groupby(['region_level', 'region', 'species', 'water']):
            df = df.drop(['region_level', 'region', 'species', 'water'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            if region not in out_dict[region_level]:
                out_dict[region_level][region] = {}
            if species not in out_dict[region_level][region]:
                out_dict[region_level][region][species] = {}
            out_dict[region_level][region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 GROUP) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()

        df_wide = groupby_to_records(bio_df_am, ['region_level', 'region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region_level', 'region', 'species', 'water', 'am', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide['type'] = 'column'; df_wide['color'] = df_wide['name'].map(COLORS)
        wall = groupby_to_records(bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"'), ['region_level', 'region', 'species', 'Water_supply', 'Landuse'], ['region_level', 'region', 'species', 'water', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        wall['am'] = 'ALL'; wall['type'] = 'column'; wall['color'] = wall['name'].map(COLORS)
        df_wide = pd.concat([df_wide, wall], ignore_index=True)
        out_dict = {}
        for (region_level, region, species, am, water), df in df_wide.groupby(['region_level', 'region', 'species', 'am', 'water']):
            df = df.drop(['region_level', 'region', 'species', 'am', 'water'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            if region not in out_dict[region_level]:
                out_dict[region_level][region] = {}
            if species not in out_dict[region_level][region]:
                out_dict[region_level][region][species] = {}
            if am not in out_dict[region_level][region][species]:
                out_dict[region_level][region][species][am] = {}
            out_dict[region_level][region][species][am][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 GROUP) Non-ag  ----------------
        _g8gr_nonag_src = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')

        df_wide = groupby_to_records(_g8gr_nonag_src, ['region_level', 'region', 'species', 'Landuse'], ['region_level', 'region', 'species', 'name', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide['type'] = 'column'; df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['_ord'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('_ord').drop(columns=['_ord'])
        out_dict = {}
        for (region_level, region, species), df in df_wide.groupby(['region_level', 'region', 'species']):
            df = df.drop(['region_level', 'region', 'species'], axis=1)
            if region_level not in out_dict:
                out_dict[region_level] = {}
            if region not in out_dict[region_level]:
                out_dict[region_level][region] = {}
            out_dict[region_level][region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


    # Save unified bio ranking data
    filename = 'BIO_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(bio_rank_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    return "Biodiversity data processing completed"


def process_supporting_info_data(SAVE_DIR, years, raw_data_dir):
    """Process and save supporting information data (Section 7)."""
    with open(f'{raw_data_dir}/model_run_settings.txt', 'r', encoding='utf-8') as f:
        settings_dict = [
            {'parameter': k.strip(), 'val': v.strip()}
            for line in f if ':' in line
            for k, v in [line.split(':', 1)]
        ]

    mem_log_path = f'{raw_data_dir}/RES_{settings.RESFACTOR}_mem_log.txt'
    mem_logs_obj = []
    if os.path.exists(mem_log_path):
        with open(mem_log_path, 'r', encoding='utf-8') as f:
            rows = [line.split('\t') for line in f if line.strip()]
        if rows:
            mem_logs_df = pd.DataFrame(rows, columns=['time', 'mem (GB)'])
            mem_logs_df['time'] = pd.to_datetime(mem_logs_df['time'], format='%Y-%m-%d %H:%M:%S').astype('int64') // 10**6
            mem_logs_df['mem (GB)'] = mem_logs_df['mem (GB)'].str.strip().astype(float)
            mem_logs_obj = [{'name': f'Memory Usage (RES {settings.RESFACTOR})', 'data': mem_logs_df.values.tolist()}]

    _last_yr = max(years)

    # Build GBF3 NVIS selected region-species map (species × region pairs with a constraint target).
    # Uses one year's CSV (targets are constant across years); stored in supporting_info for downstream visuals.
    gbf3_nvis_selected_region_species = {}
    if settings.WRITE_GBF3_NVIS != 'off':
        _scores_path = os.path.join(raw_data_dir, f'out_{_last_yr}', f'biodiversity_GBF3_NVIS_scores_{_last_yr}.csv')
        if os.path.exists(_scores_path):
            _sc = pd.read_csv(_scores_path, low_memory=False, usecols=['Vegetation Group', 'region', 'region_level', 'Target_by_Percent'])
            _sc = _sc[_sc['Target_by_Percent'].notna() & (_sc['region'] != 'AUSTRALIA')][['Vegetation Group', 'region', 'region_level']].drop_duplicates()
            for _, row in _sc.iterrows():
                rl, r, s = row['region_level'], row['region'], row['Vegetation Group']
                gbf3_nvis_selected_region_species.setdefault(rl, {}).setdefault(r, [])
                if s not in gbf3_nvis_selected_region_species[rl][r]:
                    gbf3_nvis_selected_region_species[rl][r].append(s)

    # Build GBF4 SNES selected region-species map (species × region pairs with a constraint target).
    snes_selected_region_species = {}
    if settings.WRITE_GBF4_SNES != 'off':
        _scores_path = os.path.join(raw_data_dir, f'out_{_last_yr}', f'biodiversity_GBF4_SNES_scores_{_last_yr}.csv')
        if os.path.exists(_scores_path):
            _sc = pd.read_csv(_scores_path, low_memory=False, usecols=['species', 'region', 'region_level', 'Target by Percent (%)'])
            _sc = _sc[_sc['Target by Percent (%)'].notna() & (_sc['region'] != 'AUSTRALIA')][['species', 'region', 'region_level']].drop_duplicates()
            for _, row in _sc.iterrows():
                rl, r, s = row['region_level'], row['region'], row['species']
                snes_selected_region_species.setdefault(rl, {}).setdefault(r, [])
                if s not in snes_selected_region_species[rl][r]:
                    snes_selected_region_species[rl][r].append(s)

    # Build GBF4 ECNES selected region-species map (ecological communities × region pairs with a constraint target).
    ecnes_selected_region_species = {}
    if settings.WRITE_GBF4_ECNES != 'off':
        _scores_path = os.path.join(raw_data_dir, f'out_{_last_yr}', f'biodiversity_GBF4_ECNES_scores_{_last_yr}.csv')
        if os.path.exists(_scores_path):
            _sc = pd.read_csv(_scores_path, low_memory=False, usecols=['species', 'region', 'region_level', 'Target by Percent (%)'])
            _sc = _sc[_sc['Target by Percent (%)'].notna() & (_sc['region'] != 'AUSTRALIA')][['species', 'region', 'region_level']].drop_duplicates()
            for _, row in _sc.iterrows():
                rl, r, s = row['region_level'], row['region'], row['species']
                ecnes_selected_region_species.setdefault(rl, {}).setdefault(r, [])
                if s not in ecnes_selected_region_species[rl][r]:
                    ecnes_selected_region_species[rl][r].append(s)

    supporting = {
        'model_run_settings': settings_dict,
        'years': years,
        'colors': COLORS,
        'COLORSing': COLORS,
        'mem_logs': mem_logs_obj,
        'renewables_enabled': any(settings.RENEWABLES_OPTIONS.values()),
        'GBF3_NVIS_REGION_MODE': settings.GBF3_NVIS_REGION_MODE,
        'gbf3_nvis_selected_region_species': gbf3_nvis_selected_region_species,
        'snes_selected_region_species': snes_selected_region_species,
        'ecnes_selected_region_species': ecnes_selected_region_species,
    }
    
    filename = 'Supporting_info'
    with open(f"{SAVE_DIR}/{filename}.js", 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(supporting, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    


    return "Supporting information data processing completed"
