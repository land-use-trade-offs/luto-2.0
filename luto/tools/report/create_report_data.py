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


def _groupby_to_records(df: pd.DataFrame, group_cols, out_cols, value_cols=('Year', 'Value (%)')):
    """Group `df` by `group_cols`, collect `value_cols` rows as lists into a `data` column.
    Returns a DataFrame with columns = `out_cols` (the last entry conventionally 'data').
    Robust to empty `df`: pandas 2.x returns a 2D DataFrame from `.apply` on empty
    groupby with a column subset, which breaks the usual `df_wide.columns = [...]` rename.
    """
    if df.empty:
        return pd.DataFrame(columns=list(out_cols))
    s = df.groupby(list(group_cols))[list(value_cols)].apply(lambda x: x.values.tolist())
    wide = s.reset_index()
    wide.columns = list(out_cols)
    return wide


def _bio_outside_series(bio_df: pd.DataFrame, cat: str) -> pd.DataFrame:
    """Build an "Outside LUTO study area" series df_wide for biodiversity per-Category charts.

    Outside rows in the underlying CSV are replicated across every (am, lm) combination
    with identical values, so we de-duplicate by selecting one canonical slice per chart:

      - cat='Ag':    one row per (region, species, Water_supply); pick `Agricultural Management == 'ALL'`.
      - cat='Am':    one row per (region, species, Water_supply, Agricultural Management); use all am values.
      - cat='NonAg': one row per (region, species); pick the (am=='ALL', lm=='ALL') aggregate.

    Returns a df_wide with the same columns the caller expects (including 'type', 'color', 'name').
    Returns an empty DataFrame when no outside rows exist (e.g. AUSTRALIA-mode CSVs).
    """
    outside = bio_df.query('Type == "Outside LUTO study area"')
    if outside.empty:
        return pd.DataFrame()

    if cat == 'Ag':
        sub = outside.query('`Agricultural Management` == "ALL"')
        df_wide = _groupby_to_records(
            sub, ['region', 'species', 'Water_supply'], ['region', 'species', 'water', 'data']
        )
    elif cat == 'Am':
        df_wide = _groupby_to_records(
            outside, ['region', 'species', 'Water_supply', 'Agricultural Management'],
            ['region', 'species', 'water', 'am', 'data']
        )
    elif cat == 'NonAg':
        sub = outside.query('`Agricultural Management` == "ALL" and Water_supply == "ALL"')
        df_wide = _groupby_to_records(
            sub, ['region', 'species'], ['region', 'species', 'data']
        )
    else:
        return pd.DataFrame()

    if df_wide.empty:
        return df_wide
    df_wide['name'] = 'Outside LUTO study area'
    df_wide['type'] = 'column'
    df_wide['color'] = COLORS.get('Outside LUTO study area', '#E8E8E8')
    return df_wide


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
    non_ag_dvar_area = pd.concat([df for path in non_ag_dvar_dfs['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    non_ag_dvar_area['Land-use'] = non_ag_dvar_area['Land-use'].replace(RENAME_NON_AG).infer_objects(copy=False)
    non_ag_dvar_area['Category'] = non_ag_dvar_area['Land-use'].map(lu_group_map)
    non_ag_dvar_area['Source'] = 'Non-Agricultural Land-use'
    non_ag_dvar_area['Area (ha)'] = non_ag_dvar_area['Area (ha)'].round(2)

    am_dvar_dfs = area_dvar_paths.query('base_name == "area_agricultural_management"').reset_index(drop=True)
    am_dvar_area = pd.concat([df for path in am_dvar_dfs['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
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
        .groupby(['Year', 'region', 'Source'])[['Area (ha)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Source', 'Area (ha)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'Source']).cumcount())\
        .round({'Area (ha)': 2})

    area_ranking_total = area_ranking_raw\
        .groupby(['Year', 'region'])[["Area (ha)"]]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Area (ha)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount(), Source='Total')\
        .round({'Area (ha)': 2})

    area_ranking = pd.concat([area_ranking_type, area_ranking_total], ignore_index=True)\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))


    out_dict = {}
    for (region, source), df in area_ranking.groupby(['region', 'Source']):
        df = df.drop(['region'], axis=1)

        if region not in out_dict:
            out_dict[region] = {}
        if source not in out_dict[region]:
            out_dict[region][source] = {}

        out_dict[region][source]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['value'] = df.set_index('Year')['Area (ha)'].apply( lambda x: format_with_suffix(x)).to_dict()

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
            .groupby(['Year', 'region', col])[['Area (ha)']]\
            .sum(numeric_only=True)\
            .reset_index()\
            .round({'Area (ha)': 2})
        df_wide = _groupby_to_records(df_region, [col, 'region'], ['name', 'region','data'], value_cols=('Year', 'Area (ha)'))
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
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'Area_overview_{idx+1}_{col.replace(" ", "_")}'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')



    # -------------------- Area by Agricultural land --------------------
    df_wide = _groupby_to_records(ag_dvar_area, ['region', 'Water_supply', 'Land-use'], ['region', 'water', 'name', 'data'], value_cols=('Year', 'Area (ha)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region, water), df in df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = []
        out_dict[region][water] = df.to_dict(orient='records')

    filename = 'Area_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Area by Agricultural Management Area (ha) Land use --------------------
    df_wide = _groupby_to_records(am_dvar_area .query('Type != "ALL"'), ['region', 'Water_supply', 'Land-use', 'Type'], ['region', 'water', 'landuse', 'name', 'data'], value_cols=('Year', 'Area (ha)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])


    out_dict = {}
    for (region, water, landuse), df in df_wide.groupby(['region', 'water', 'landuse']):
        df = df.drop(['region', 'water', 'landuse'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}

        out_dict[region][water][landuse] = df.to_dict(orient='records')

    filename = f'Area_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Area by Non-Agricultural Land-use --------------------
    df_wide = _groupby_to_records(non_ag_dvar_area, ['region', 'Land-use'], ['region', 'name', 'data'], value_cols=('Year', 'Area (ha)'))
    df_wide['type'] = 'column'

    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop('region', axis=1)
        out_dict[region] = df.to_dict(orient='records')

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
    revenue_am_df = pd.concat([df for path in revenue_am_df['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    revenue_am_df = revenue_am_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Agricultural Management (revenue)')

    cost_am_df = files.query('base_name == "economics_am_cost"').reset_index(drop=True)
    cost_am_df = pd.concat([df for path in cost_am_df['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    cost_am_df = cost_am_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Agricultural Management (cost)')
    cost_am_df['Value ($)'] = cost_am_df['Value ($)'] * -1          # Convert cost to negative value

    revenue_non_ag_df = files.query('base_name == "economics_non_ag_revenue"').reset_index(drop=True)
    revenue_non_ag_df = pd.concat([df for path in revenue_non_ag_df['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    revenue_non_ag_df = revenue_non_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Non-Agricultural Land-use (revenue)')

    cost_non_ag_df = files.query('base_name == "economics_non_ag_cost"').reset_index(drop=True)
    cost_non_ag_df = pd.concat([df for path in cost_non_ag_df['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    cost_non_ag_df = cost_non_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Non-Agricultural Land-use (cost)')
    cost_non_ag_df['Value ($)'] = cost_non_ag_df['Value ($)'] * -1  # Convert cost to negative value

    cost_transition_ag2ag_df = files.query('base_name == "transition_ag2ag_cost"').reset_index(drop=True)
    cost_transition_ag2ag_df = pd.concat([df for path in cost_transition_ag2ag_df['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    cost_transition_ag2ag_df = cost_transition_ag2ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Transition cost (Ag2Ag)')
    cost_transition_ag2ag_df['Value ($)'] = cost_transition_ag2ag_df['Cost ($)']  * -1          # Convert cost to negative value

    cost_transition_ag2non_ag_df = files.query('base_name == "transition_ag2nonag_cost"').reset_index(drop=True)
    cost_transition_ag2non_ag_df = pd.concat([df for path in cost_transition_ag2non_ag_df['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    cost_transition_ag2non_ag_df = cost_transition_ag2non_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False).assign(Source='Transition cost (Ag2Non-Ag)')
    cost_transition_ag2non_ag_df['Value ($)'] = cost_transition_ag2non_ag_df['Cost ($)'] * -1   # Convert cost to negative value

    cost_transition_non_ag2ag_df = files.query('base_name == "transition_nonag2ag_cost"').reset_index(drop=True)
    cost_transition_non_ag2ag_df = pd.concat([df for path in cost_transition_non_ag2ag_df['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
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
        ).groupby(['Year', 'region']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank=lambda x: x.groupby(['Year']).cumcount()
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
        ).groupby(['Year', 'region']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index(
        ).assign(**{'Value ($)': lambda x: abs(x['Value ($)'])}
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank=lambda x: x.groupby(['Year']).cumcount()
        ).assign(Source='Cost')
        
    profit_df = revenue_df.merge(
        cost_df, on=['Year', 'region'], suffixes=('_revenue', '_cost')
        ).assign(**{'Value ($)': lambda x: x['Value ($)_revenue'] - x['Value ($)_cost']}
        ).drop(columns=['Value ($)_revenue', 'Value ($)_cost']
        ).sort_values(['Year', 'Value ($)'], ascending=[True, False]
        ).assign(Rank=lambda x: x.groupby(['Year']).cumcount()
        ).assign(Source='Profit')

    ranking_df = pd.concat([revenue_df, cost_df, profit_df]).assign(color= lambda x: x['Rank'].map(get_rank_color))


    out_dict = {}
    for (region, source), df in ranking_df.groupby(['region', 'Source']):
        if region not in out_dict:
            out_dict[region] = {}
        if not source in out_dict[region]:
            out_dict[region][source] = {}

        df = df.drop(columns='region')
        out_dict[region][source]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][source]['value'] = df.set_index('Year')['Value ($)'].apply( lambda x: format_with_suffix(x)).to_dict()

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
        ).groupby(['region', 'Source', 'Year']
        )[['Value ($)']].sum(numeric_only=True
        ).reset_index()

    dfs = []
    for region, df in rev_cost_net_region.groupby('region'):
        df_col = _groupby_to_records(df, ['Source'], ['name','data'], value_cols=('Year', 'Value ($)'))
        df_col['type'] = 'column'

        df_col.loc[len(df_col)] = [
            'Profit',
            df.groupby(['Year'])[['Value ($)']].sum(numeric_only=True).reset_index().values.tolist(),
            'line',
        ]
        df_col['region'] = region
        dfs.append(df_col)

    rev_cost_wide_json = pd.concat(dfs, ignore_index=True)
    rev_cost_wide_json['name_order'] = rev_cost_wide_json['name'].map({name: i for i, name in enumerate(order)})
    rev_cost_wide_json = rev_cost_wide_json.sort_values(['region', 'name_order']).drop(columns=['name_order']).reset_index(drop=True)
    rev_cost_wide_json['color'] = rev_cost_wide_json['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for region,df in rev_cost_wide_json.groupby('region'):
        df = df.drop(columns='region')
        df.columns = ['name','data','type','color']
        out_dict[region] = df.to_dict(orient='records')

    filename = 'Economics_overview_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # Overview: ag cost/revenue by type
    economics_ag = pd.concat([revenue_ag_df.query('Water_supply != "ALL" and Type != "ALL" and `Land-use` != "ALL"'), cost_ag_df.query('Water_supply != "ALL" and Type != "ALL" and `Land-use` != "ALL"')])\
        .query('abs(`Value ($)`) > 1')\
        .groupby(['region', 'Type','Year'])['Value ($)']\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value ($)': 2})


    df_wide = _groupby_to_records(economics_ag, ['region', 'Type'], ['region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: list(COLORS.keys()).index(x) if x in COLORS else -1)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])


    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop('region', axis=1)
        out_dict[region] = df.to_dict(orient='records')

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
        ).groupby(['region', 'Management Type', 'Rev_Cost', 'Year'])[['Value ($)']
        ].sum(
        ).reset_index()

    df_wide = _groupby_to_records(economics_am, ['region', 'Management Type', 'Rev_Cost'], ['region', 'name', 'Rev_Cost', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region', 'Rev_Cost'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

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
        ).groupby(['region', 'Land-use', 'Rev_Cost', 'Year'])[['Value ($)']
        ].sum(
        ).reset_index()

    df_wide = _groupby_to_records(economics_non_ag, ['region', 'Land-use', 'Rev_Cost'], ['region', 'name', 'Rev_Cost', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'

    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, rev_cost), df in df_wide.groupby(['region', 'Rev_Cost']):
        df = df.drop(['region', 'Rev_Cost'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        out_dict[region][rev_cost] = df.to_dict(orient='records')

    # Add Profit from pre-computed profit CSVs
    profit_na_files = files.query('base_name == "economics_non_ag_profit"').reset_index(drop=True)
    profit_na_df = pd.concat([df for p in profit_na_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    profit_na_df = profit_na_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_na_df = profit_na_df.query('`Land-use` != "ALL"').round({'Value ($)': 2})

    df_profit = _groupby_to_records(profit_na_df, ['region', 'Land-use'], ['region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_profit['type'] = 'column'
    df_profit['color'] = df_profit['name'].apply(lambda x: COLORS[x])
    df_profit['name_order'] = df_profit['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_profit = df_profit.sort_values('name_order').drop(columns=['name_order'])

    for region, df in df_profit.groupby('region'):
        df = df.drop(['region'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        out_dict[region]['Profit'] = df.to_dict(orient='records')

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
    df_wide = _groupby_to_records(ag_rev, ['region', 'Type', 'Water_supply', 'Land-use'], ['region', '_type', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, _type, water), df in df_wide.groupby(['region', '_type', 'water']):
        df = df.drop(['region', '_type', 'water'], axis=1)
        out_dict.setdefault(region, {}).setdefault(_type, {})[water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Ag_revenue')

    # Ag Cost: region → Type(source) → Water → [series by LU]
    ag_cost = cost_ag_df.query('`Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
    df_wide = _groupby_to_records(ag_cost, ['region', 'Type', 'Water_supply', 'Land-use'], ['region', '_type', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, _type, water), df in df_wide.groupby(['region', '_type', 'water']):
        df = df.drop(['region', '_type', 'water'], axis=1)
        out_dict.setdefault(region, {}).setdefault(_type, {})[water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Ag_cost')

    # Ag Profit: region → Water → [series by LU]
    profit_ag_files = files.query('base_name == "economics_ag_profit"').reset_index(drop=True)
    profit_ag_df = pd.concat([pd.read_csv(p) for p in profit_ag_files['path']], ignore_index=True)
    profit_ag_df = profit_ag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_ag_df = profit_ag_df.query('`Land-use` != "ALL"').round({'Value ($)': 2})

    df_profit = _groupby_to_records(profit_ag_df, ['region', 'Water_supply', 'Land-use'], ['region', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_profit['type'] = 'column'
    df_profit['color'] = df_profit['name'].apply(lambda x: COLORS[x])
    df_profit['name_order'] = df_profit['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_profit = df_profit.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water), df in df_profit.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        out_dict.setdefault(region, {})[water] = df.to_dict(orient='records')
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
            df_wide = _groupby_to_records(ag2ag_filt, ['region', 'Type', 'Water_supply', 'To_Land-use'], ['region', '_type', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
            df_wide['type'] = 'column'
            df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#999999'))
            df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
            df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
            for (region, _type, water), df in df_wide.groupby(['region', '_type', 'water']):
                df = df.drop(['region', '_type', 'water'], axis=1)
                out_dict.setdefault(region, {}).setdefault(_type, {})[water] = df.to_dict(orient='records')
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
                df_wide = _groupby_to_records(nonag2ag_filt, ['region', 'Type', 'Water_supply', 'From_Land-use'], ['region', '_type', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
                df_wide['type'] = 'column'
                df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#999999'))
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                for (region, _type, water), df in df_wide.groupby(['region', '_type', 'water']):
                    df = df.drop(['region', '_type', 'water'], axis=1)
                    out_dict.setdefault(region, {}).setdefault(_type, {})[water] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Ag_transition_nonag2ag')


    # -------------------- Economics for ag-management (separate files per MapType) --------------------

    # Am Revenue: region → Water → LU → [series by mgmt]
    am_rev = revenue_am_df.query('`Management Type` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
    df_wide = _groupby_to_records(am_rev, ['region', 'Water_supply', 'Land-use', 'Management Type'], ['region', 'water', 'landuse', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region, water, landuse), df in df_wide.groupby(['region', 'water', 'landuse']):
        df = df.drop(['region', 'water', 'landuse'], axis=1)
        out_dict.setdefault(region, {}).setdefault(water, {})[landuse] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Am_revenue')

    # Am Cost: region → Water → LU → [series by mgmt]
    am_cost = cost_am_df.query('`Management Type` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')
    df_wide = _groupby_to_records(am_cost, ['region', 'Water_supply', 'Land-use', 'Management Type'], ['region', 'water', 'landuse', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region, water, landuse), df in df_wide.groupby(['region', 'water', 'landuse']):
        df = df.drop(['region', 'water', 'landuse'], axis=1)
        out_dict.setdefault(region, {}).setdefault(water, {})[landuse] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Am_cost')

    # Am Profit: region → Water → LU → [series by mgmt]
    profit_am_files = files.query('base_name == "economics_am_profit"').reset_index(drop=True)
    profit_am_df = pd.concat([df for p in profit_am_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    profit_am_df = profit_am_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_am_df = profit_am_df.query('`Management Type` != "ALL"').round({'Value ($)': 2})

    df_profit = _groupby_to_records(profit_am_df, ['region', 'Water_supply', 'Land-use', 'Management Type'], ['region', 'water', 'landuse', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_profit['type'] = 'column'
    df_profit['color'] = df_profit['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region, water, landuse), df in df_profit.groupby(['region', 'water', 'landuse']):
        df = df.drop(['region', 'water', 'landuse'], axis=1)
        out_dict.setdefault(region, {}).setdefault(water, {})[landuse] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_Am_profit')


    # -------------------- Economics for non-ag (separate files per MapType) --------------------

    # NonAg Revenue: region → [series by LU]
    na_rev = revenue_non_ag_df.query('`Land-use` != "ALL" and abs(`Value ($)`) > 1').round({'Value ($)': 2})
    df_wide = _groupby_to_records(na_rev, ['region', 'Land-use'], ['region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop('region', axis=1)
        out_dict[region] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_NonAg_revenue')

    # NonAg Cost: region → [series by LU]
    na_cost = cost_non_ag_df.query('`Land-use` != "ALL" and abs(`Value ($)`) > 1').round({'Value ($)': 2})
    df_wide = _groupby_to_records(na_cost, ['region', 'Land-use'], ['region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop('region', axis=1)
        out_dict[region] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_NonAg_cost')

    # NonAg Profit: region → [series by LU]
    profit_na_files = files.query('base_name == "economics_non_ag_profit"').reset_index(drop=True)
    profit_na_df = pd.concat([df for p in profit_na_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    profit_na_df = profit_na_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_na_df = profit_na_df.query('`Land-use` != "ALL"').round({'Value ($)': 2})

    df_profit = _groupby_to_records(profit_na_df, ['region', 'Land-use'], ['region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_profit['type'] = 'column'
    df_profit['color'] = df_profit['name'].apply(lambda x: COLORS[x])
    df_profit['name_order'] = df_profit['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_profit = df_profit.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_profit.groupby('region'):
        df = df.drop('region', axis=1)
        out_dict[region] = df.to_dict(orient='records')
    write_chart_js(out_dict, 'Economics_NonAg_profit')

    # NonAg Transition (Ag2NonAg): region → [series by LU]
    t_ag2nonag_files = files.query('base_name == "economics_non_ag_transition_Ag2NonAg"').reset_index(drop=True)
    t_ag2nonag_df = pd.concat([df for p in t_ag2nonag_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    t_ag2nonag_df = t_ag2nonag_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    t_ag2nonag_df['Value ($)'] = t_ag2nonag_df['Value ($)'] * -1   # Convert cost to negative
    t_ag2nonag_filt = t_ag2nonag_df.query('`Land-use` != "ALL"').round({'Value ($)': 2}).query('abs(`Value ($)`) > 1')

    out_dict = {}
    if not t_ag2nonag_filt.empty:
        df_wide = _groupby_to_records(t_ag2nonag_filt, ['region', 'Land-use'], ['region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        for region, df in df_wide.groupby('region'):
            df = df.drop('region', axis=1)
            out_dict[region] = df.to_dict(orient='records')
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
                df_wide = _groupby_to_records(t_nonag2nonag_filt, ['region', 'Land-use'], ['region', 'name', 'data'], value_cols=('Year', 'Value ($)'))
                df_wide['type'] = 'column'
                df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#999999'))
                df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
                df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
                for region, df in df_wide.groupby('region'):
                    df = df.drop('region', axis=1)
                    out_dict[region] = df.to_dict(orient='records')
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
    profit_am_sum_df = profit_am_sum_df.groupby(['region', 'Water_supply', 'Land-use', 'Year'])[['Value ($)']].sum().reset_index()

    profit_na_files = files.query('base_name == "economics_non_ag_profit"').reset_index(drop=True)
    profit_na_sum_df = pd.concat([df for p in profit_na_files['path'] if not (df := pd.read_csv(p)).empty], ignore_index=True)
    profit_na_sum_df = profit_na_sum_df.replace(RENAME_AM_NON_AG).infer_objects(copy=False)
    profit_na_sum_df = profit_na_sum_df.query('`Land-use` != "ALL"')
    # Assign nonag to Dryland to avoid double counting
    profit_na_sum_df['Water_supply'] = 'Dryland'

    # Combine all three
    econ_sum = pd.concat([
        profit_ag_sum_df[['region', 'Water_supply', 'Land-use', 'Year', 'Value ($)']],
        profit_am_sum_df[['region', 'Water_supply', 'Land-use', 'Year', 'Value ($)']],
        profit_na_sum_df[['region', 'Water_supply', 'Land-use', 'Year', 'Value ($)']],
    ], ignore_index=True)

    # Group to sum ag+am for same land uses, then add ALL water aggregate
    econ_sum = econ_sum.groupby(['region', 'Water_supply', 'Land-use', 'Year'])[['Value ($)']].sum().reset_index()

    econ_sum_all_water = econ_sum.groupby(['region', 'Land-use', 'Year'])[['Value ($)']].sum().reset_index().assign(Water_supply='ALL')
    econ_sum = pd.concat([econ_sum_all_water, econ_sum], ignore_index=True)

    df_wide = _groupby_to_records(econ_sum, ['region', 'Water_supply', 'Land-use'], ['region', 'water', 'name', 'data'], value_cols=('Year', 'Value ($)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x) if x in LANDUSE_ALL_RENAMED else 999)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water), df in df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        out_dict[region][water] = df.to_dict(orient='records')

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
    #     for region, df in df_wide.groupby('region'):
    #         df = df.drop(['region'], axis=1)
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
    # for (region,year),df in cost_transition_ag2ag_trans_mat.groupby(['region', 'Year']):

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
    #     for region, df in df_wide.groupby('region'):
    #         df = df.drop(['region'], axis=1)
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
    # for (region,year),df in cost_transition_ag2nonag_trans_mat.groupby(['region', 'Year']):

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
    #     for region, df in df_wide.groupby('region'):
    #         df = df.drop(['region'], axis=1)
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
    # for (region,year),df in cost_transition_nonag2ag_trans_mat.groupby(['region', 'Year']):

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
        [{'Commodity': 'beef meat', 'Type': 'Non_Agricultural', 'Year': 2050, 'region':'ACT', 'Production (t/KL)': 0},
         {'Commodity': 'beef meat', 'Type': 'Non_Agricultural', 'Year': 2020, 'region':'AUSTRALIA', 'Production (t/KL)': 0}]
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
        .groupby(['Year', 'region', 'group'])[['Production (t/KL)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'group', 'Production (t/KL)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year', 'group']).cumcount())\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))\
        .assign(Year=lambda x: x['Year'].astype(int))\
        .round({'Production (t/KL)': 2})

    out_dict = {}
    for (region, group), df in quantity_rank.groupby(['region', 'group']):
        df = df.drop(['region'], axis=1)

        if region not in out_dict:
            out_dict[region] = {}
        if group not in out_dict[region]:
            out_dict[region][group] = {}

        out_dict[region][group]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][group]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][group]['value'] = df.set_index('Year')['Production (t/KL)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Production_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Overview --------------------

    # sum
    demand_type_wide = _groupby_to_records(DEMAND_DATA_long .groupby(['Year', 'Type'])[['Quantity (tonnes, KL)']] .sum(numeric_only=True) .reset_index() .round({'Quantity (tonnes, KL)': 2}), ['Type'], ['name', 'data'], value_cols=('Year', 'Quantity (tonnes, KL)'))
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
    df_wide = _groupby_to_records(quantity_ag, ['region', 'Water_supply', 'Commodity'], ['region', 'water', 'name', 'data'], value_cols=('Year', 'Production (t/KL)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water), df in df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water] = df.to_dict(orient='records')

    filename = f'Production_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Commodity production for ag-man --------------------
    df_wide = _groupby_to_records(quantity_am, ['region', 'Water_supply', 'Commodity', 'am'], ['region', 'water', 'commodity', 'name', 'data'], value_cols=('Year', 'Production (t/KL)'))
    
    df_wide_c_ALL = quantity_am\
        .groupby(['region', 'Water_supply', 'am','Year',])\
        .sum(numeric_only=True)\
        .reset_index()\
        .groupby(['region', 'Water_supply', 'am'])[['Year','Production (t/KL)']]\
        .apply(lambda x: x[['Year','Production (t/KL)']].values.tolist())\
        .reset_index()
    df_wide_c_ALL['Commodity'] = 'ALL'
    df_wide_c_ALL.columns = ['region', 'water', 'name', 'data', 'commodity']
    
    
    df_wide = pd.concat([df_wide, df_wide_c_ALL], axis=0, ignore_index=True)
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    

    out_dict = {}
    for (region, water, commodity), df in df_wide.groupby(['region', 'water', 'commodity']):
        df = df.drop(['region', 'water', 'commodity'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water][commodity] = df.to_dict(orient='records')

    filename = f'Production_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- Commodity production for non-ag --------------------
    df_wide = _groupby_to_records(quantity_non_ag, ['region', 'Commodity'], ['region', 'name', 'data'], value_cols=('Year', 'Production (t/KL)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

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
        quantity_ag.query('Water_supply != "ALL"')[['region', 'Water_supply', 'Commodity', 'Year', 'Production (t/KL)']],
        quantity_am.query('Water_supply != "ALL" and Commodity != "ALL"')[['region', 'Water_supply', 'Commodity', 'Year', 'Production (t/KL)']],
        quantity_non_ag_with_water[['region', 'Water_supply', 'Commodity', 'Year', 'Production (t/KL)']],
    ], ignore_index=True)

    # Add ALL water level
    quantity_sum_all_water = quantity_sum\
        .groupby(['region', 'Commodity', 'Year'])[['Production (t/KL)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .assign(Water_supply='ALL')
    quantity_sum = pd.concat([quantity_sum_all_water, quantity_sum], ignore_index=True)

    # Group by region, water, commodity → time series
    df_wide = _groupby_to_records(quantity_sum .groupby(['region', 'Water_supply', 'Commodity', 'Year'])[['Production (t/KL)']] .sum(numeric_only=True) .reset_index(), ['region', 'Water_supply', 'Commodity'], ['region', 'water', 'name', 'data'], value_cols=('Year', 'Production (t/KL)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    df_wide['name_order'] = df_wide['name'].apply(lambda x: COMMODITIES_ALL.index(x) if x in COMMODITIES_ALL else 999)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water), df in df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        out_dict[region][water] = df.to_dict(orient='records')

    filename = f'Production_Sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    return "Production data processing completed"


def process_renewable_data(files, SAVE_DIR, years):
    """Process and save renewable energy data (ag-mgt level only)."""

    re_state_files = files.query('base_name == "renewable_energy_with_existing_state"').reset_index(drop=True)
    
    if re_state_files.empty:
        return "Renewable energy data processing skipped (no files found)"

    _re_dfs = [df for path in re_state_files['path'] if not (df := pd.read_csv(path)).empty]
    if not _re_dfs:
        return "Renewable energy data processing skipped (all CSV files are empty)"
    re_df = pd.concat(_re_dfs, ignore_index=True)

    # Rename am labels to match COLORS keys
    re_df['am'] = re_df['am'].replace(RENAME_AM_NON_AG)

    # Exclude lu=ALL only; am=ALL and lm=ALL are valid hierarchy buttons in the report
    re_am_df = re_df.query('lu != "ALL"')

    # -------------------- Renewable energy by AgMgt (am → lm → lu hierarchy) --------------------
    df_wide = _groupby_to_records(re_am_df, ['region', 'am', 'lm', 'lu'], ['region', 'am', 'lm', 'name', 'data'], value_cols=('Year', 'Value (MWh)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS.get(x, '#AAAAAA'))
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
    
    # Get targets
    re_targets_files = files.query('base_name.str.contains("renewable_energy_targets")').reset_index(drop=True)
    re_targets_df = pd.concat([df for path in re_targets_files['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    re_targets_df_wide = _groupby_to_records(re_targets_df, ['region', 'am', 'lm'], ['region', 'am', 'lm', 'data'], value_cols=('Year', 'Value (MWh)'))
    re_targets_df_wide['type'] = 'line'
    re_targets_df_wide['name'] = 'Target'
    re_targets_df_wide['color'] = "#424040"

    # Save to json
    re_df_wide = pd.concat([df_wide, re_targets_df_wide], ignore_index=True)
    out_dict = {}
    for (region, am, lm), df in re_df_wide.groupby(['region', 'am', 'lm']):
        df = df.drop(columns=['region', 'am', 'lm'])
        out_dict.setdefault(region, {}).setdefault(am, {})[lm] = df.to_dict(orient='records')

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
    GHG_non_ag = pd.concat([df for path in GHG_non_ag['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    GHG_non_ag = GHG_non_ag.replace(RENAME_AM_NON_AG).infer_objects(copy=False).round({'Value (t CO2e)': 2})
    
    GHG_ag_man = GHG_files.query('base_name.str.contains("agricultural_management")').reset_index(drop=True)
    GHG_ag_man = pd.concat([df for path in GHG_ag_man['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
    GHG_ag_man = GHG_ag_man.replace(RENAME_AM_NON_AG).infer_objects(copy=False).round({'Value (t CO2e)': 2})

    GHG_transition = GHG_files.query('base_name.str.contains("transition_penalty")').reset_index(drop=True)
    GHG_transition = pd.concat([df for path in GHG_transition['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
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
    GHG_land_non_all = pd.concat([GHG_ag.query('Water_supply != "ALL" and Source != "ALL" and `Land-use` != "ALL"'), GHG_non_ag, GHG_ag_man.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Agricultural Management Type` != "ALL"'), GHG_transition], axis=0)\
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
    for region,df in GHG_land_non_all.groupby('region'):
        df_reg = _groupby_to_records(df .groupby(['Year','Type'])[['Value (t CO2e)']] .sum(numeric_only=True) .reset_index(), ['Type'], ['name','data'], value_cols=('Year', 'Value (t CO2e)'))
        df_reg['type'] = 'column'

        if region == "AUSTRALIA":
            df_reg.loc[len(df_reg)] = ['Off-land emissions', net_offland_AUS_wide,  'column']
            df_reg.loc[len(df_reg)] = ['GHG emission limit', GHG_limit_wide, 'line']
            df_reg.loc[len(df_reg)] = [
                'Net emissions', 
                list(zip(years, (df.groupby('Year')['Value (t CO2e)'].sum().values + GHG_off_land.groupby('Year')['Value (t CO2e)'].sum()))),
                'line'
            ]
        else:
            df_reg.loc[len(df_reg)] = [
                'Net emissions', 
                list(zip(years, (df.groupby('Year')['Value (t CO2e)'].sum().values))),
                'line'
            ]
                

        df_reg['name_order'] = df_reg['name'].apply(lambda x: order_GHG.index(x))
        df_reg = df_reg.sort_values('name_order').drop(columns=['name_order'])
        df_reg['color'] = df_reg['name'].apply(lambda x: COLORS[x])
        GHG_region[region] = json.loads(df_reg.to_json(orient='records'))


    filename = 'GHG_overview_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(GHG_region, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
        
    # Ag
    GHG_ag_non_all_wide = _groupby_to_records(GHG_ag.query('Water_supply != "ALL" and Source != "ALL" and `Land-use` != "ALL"') .groupby(['region','Land-use','Year'])[['Value (t CO2e)']] .sum(numeric_only=True) .reset_index(), ['region','Land-use'], ['region', 'name','data'], value_cols=('Year', 'Value (t CO2e)'))
    GHG_ag_non_all_wide['type'] = 'column'
    GHG_ag_non_all_wide['color'] = GHG_ag_non_all_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for region,df in GHG_ag_non_all_wide.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')
    

    filename = 'GHG_overview_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Am
    GHG_ag_man_non_all_wide = _groupby_to_records(GHG_ag_man.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Agricultural Management Type` != "ALL"') .groupby(['region', 'Agricultural Management Type', 'Year'])[['Value (t CO2e)']] .sum(numeric_only=True) .reset_index(), ['region', 'Agricultural Management Type'], ['region', 'name','data'], value_cols=('Year', 'Value (t CO2e)'))
    GHG_ag_man_non_all_wide['type'] = 'column'
    GHG_ag_man_non_all_wide['color'] = GHG_ag_man_non_all_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for region, df in GHG_ag_man_non_all_wide.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')
    

    filename = 'GHG_overview_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Non-Ag
    GHG_non_ag_wide = _groupby_to_records(GHG_non_ag .query('`Land-use` != "ALL"') .groupby(['region','Land-use','Year'])[['Value (t CO2e)']] .sum(numeric_only=True) .reset_index(), ['region','Land-use'], ['region','name','data'], value_cols=('Year', 'Value (t CO2e)'))
    GHG_non_ag_wide['type'] = 'column'
    GHG_non_ag_wide['color'] = GHG_non_ag_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for region,df in GHG_non_ag_wide.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')
    

    filename = 'GHG_overview_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    


    # -------------------- GHG Sum (Ag + Am + NonAg + Transition + Off-land) --------------------

    # Ag: sum over source, keep water/lu/year
    ghg_ag_sum = GHG_ag.query('Water_supply != "ALL" and Source != "ALL" and `Land-use` != "ALL"')\
        .groupby(['region', 'Water_supply', 'Land-use', 'Year'])[['Value (t CO2e)']]\
        .sum(numeric_only=True).reset_index()

    # Am: sum over management type, keep water/lu/year
    ghg_am_sum = GHG_ag_man.query('Water_supply != "ALL" and `Land-use` != "ALL" and `Agricultural Management Type` != "ALL"')\
        .groupby(['region', 'Water_supply', 'Land-use', 'Year'])[['Value (t CO2e)']]\
        .sum(numeric_only=True).reset_index()

    # NonAg: assign to Dryland water supply
    ghg_non_ag_sum = GHG_non_ag.query('`Land-use` != "ALL"').reset_index(drop=True)[['region', 'Land-use', 'Year', 'Value (t CO2e)']].copy()
    ghg_non_ag_sum['Water_supply'] = 'Dryland'

    # Transition: sum over Land-use per transition type; use Type as the series name
    ghg_transition_sum = GHG_transition\
        .groupby(['region', 'Water_supply', 'Type', 'Year'])[['Value (t CO2e)']]\
        .sum(numeric_only=True).reset_index()\
        .rename(columns={'Type': 'Land-use'})

    # Combine Ag + Am + NonAg + Transition
    ghg_sum = pd.concat([ghg_ag_sum, ghg_am_sum, ghg_non_ag_sum, ghg_transition_sum], ignore_index=True)
    ghg_sum = ghg_sum.query('abs(`Value (t CO2e)`) > 1').reset_index(drop=True)

    # Add ALL water aggregate (computed before off_land to avoid double-counting)
    ghg_sum_all_water = ghg_sum\
        .groupby(['region', 'Land-use', 'Year'])[['Value (t CO2e)']]\
        .sum(numeric_only=True).reset_index().assign(Water_supply='ALL')
    ghg_sum = pd.concat([ghg_sum_all_water, ghg_sum], ignore_index=True)

    # Off-land: replicate across ALL/Dryland/Irrigated so it appears on every water tab
    ghg_off_land_by_year = GHG_off_land.groupby('Year')[['Value (t CO2e)']].sum(numeric_only=True).reset_index()
    ghg_off_land_rows = pd.concat([
        ghg_off_land_by_year.assign(Water_supply=w, **{'Land-use': 'Off-land emissions', 'region': 'AUSTRALIA'})
        for w in ['ALL', 'Dryland', 'Irrigated']
    ], ignore_index=True).query('abs(`Value (t CO2e)`) > 1')
    ghg_sum = pd.concat([ghg_sum, ghg_off_land_rows], ignore_index=True)

    df_wide = _groupby_to_records(ghg_sum, ['region', 'Water_supply', 'Land-use'], ['region', 'water', 'name', 'data'], value_cols=('Year', 'Value (t CO2e)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])
    _ghg_sum_order = LANDUSE_ALL_RENAMED + [
        'Unallocated natural to modified',
        'Unallocated natural to livestock natural',
        'Livestock natural to modified',
        'Off-land emissions',
    ]
    df_wide['name_order'] = df_wide['name'].apply(lambda x: _ghg_sum_order.index(x) if x in _ghg_sum_order else 999)
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water), df in df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        out_dict[region][water] = df.to_dict(orient='records')

    # Add Net emissions and GHG emission limit lines for AUSTRALIA across all water tabs
    if 'AUSTRALIA' in out_dict:
        net_values = (
            ghg_sum_all_water.query('region == "AUSTRALIA"')
            .groupby('Year')['Value (t CO2e)'].sum()
            + ghg_off_land_by_year.set_index('Year')['Value (t CO2e)']
        )
        net_australia_wide = [[y, v] for y, v in zip(net_values.index.tolist(), net_values.values)]
        for water in out_dict['AUSTRALIA']:
            out_dict['AUSTRALIA'][water].append({'name': 'Net emissions',    'data': net_australia_wide, 'type': 'line', 'color': COLORS['Net emissions']})
            out_dict['AUSTRALIA'][water].append({'name': 'GHG emission limit', 'data': GHG_limit_wide,      'type': 'line', 'color': COLORS['GHG emission limit']})

    filename = 'GHG_Sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # -------------------- GHG ranking --------------------
    GHG_rank_emission_region = GHG_land_non_all\
        .query('`Value (t CO2e)` > 0')\
        .groupby(['Year', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (t CO2e)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
        .assign(Type='GHG emissions')
        
    GHG_rank_emission_region.loc[
        GHG_rank_emission_region['region'] == 'AUSTRALIA', 
        'Value (t CO2e)'] += GHG_off_land.groupby('Year')['Value (t CO2e)'].sum().values
    
    GHG_rank_sequestration_region = GHG_land_non_all\
        .query('`Value (t CO2e)` < 0')\
        .assign(**{'Value (t CO2e)': lambda x: abs(x['Value (t CO2e)'])})\
        .groupby(['Year', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (t CO2e)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
        .assign(Type='GHG sequestrations')
    GHG_rank_region_net = GHG_rank_emission_region\
        .merge(GHG_rank_sequestration_region, on=['Year', 'region'], how='outer', suffixes=('_emission', '_sequestration'))\
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
    for (region, e_type), df in GHG_rank.groupby(['region', 'Type']):
        if region not in out_dict:
            out_dict[region] = {}
        if e_type not in out_dict[region]:
            out_dict[region][e_type] = {}

        df = df.drop(columns='region')
        out_dict[region][e_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][e_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][e_type]['value'] = df.set_index('Year')['Value (t CO2e)'].apply( lambda x: format_with_suffix(x)).to_dict()

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
        .groupby(['region', 'Source', 'Water_supply', 'Land-use', 'Year'])[['Value (t CO2e)']]\
        .sum()\
        .reset_index()\
        .round({'Value (t CO2e)': 2})
    df_wide = _groupby_to_records(df_wide, ['region', 'Source', 'Water_supply', 'Land-use'], ['region', 'source', 'water', 'name', 'data'], value_cols=('Year', 'Value (t CO2e)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water, source), df in df_wide.groupby(['region', 'water', 'source']):
        df = df.drop(['region', 'water', 'source'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        if source not in out_dict[region][water]:
            out_dict[region][water][source] = {}
        out_dict[region][water][source] = df.to_dict(orient='records')
        
    filename = 'GHG_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- GHG by Non-Agricultural --------------------
    Non_ag_reduction_long = GHG_land.query('Type == "Non-Agricultural Land-use" and `Land-use` != "ALL"').reset_index(drop=True)
    Non_ag_reduction_long['Value (t CO2e)'] *= -1  # Convert from negative to positive
    
    df_region = Non_ag_reduction_long\
        .groupby(['Year', 'region', 'Land-use'])[['Value (t CO2e)']]\
        .sum(numeric_only=True)\
        .reset_index()\
        .round({'Value (t CO2e)': 2})
    df_wide = _groupby_to_records(df_region, ['Land-use', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Value (t CO2e)'))
    df_wide['type'] = 'column'
    
    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
    
    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

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

    df_wide = _groupby_to_records(Ag_man_sequestration_long, ['region', 'Water_supply', 'Land-use', 'Agricultural Management Type'], ['region', 'water', 'landuse', 'name', 'data'], value_cols=('Year', 'Value (t CO2e)'))
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region, water, landuse), df in df_wide.groupby(['region', 'water', 'landuse']):
        df = df.drop(['region', 'water', 'landuse'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water][landuse] = df.to_dict(orient='records')

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
        if region not in out_dict:
            out_dict[region] = {}
        if w_type not in out_dict[region]:
            out_dict[region][w_type] = {} 
        out_dict[region][w_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][w_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][w_type]['value'] = df.set_index('Year')['Value (ML)'].apply( lambda x: format_with_suffix(x)).to_dict()

    filename = 'Water_ranking_NRM'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')



    # -------------------- Overview  --------------------
    
    # sum
    water_sum = _groupby_to_records(water_net_yield_NRM_region .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"') .groupby(['region_NRM', 'Type', 'Year'])[['Value (ML)']] .sum(numeric_only=True) .reset_index() .round({'Value (ML)': 2}), ['region_NRM', 'Type'], ['region', 'name','data'], value_cols=('Year', 'Value (ML)'))
    water_sum['type'] = 'column'
    water_sum['color'] = water_sum['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for region, df in water_sum.groupby('region'):
        df = df.drop(columns=['region'])
        out_dict[region] = df.to_dict(orient='records')

    filename = f'Water_overview_NRM_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    
    # Ag
    water_ag = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .query('Type == "Agricultural Land-use"')
    
    water_overview_ag = _groupby_to_records(water_ag .groupby(['region_NRM', 'Landuse', 'Year'])[['Value (ML)']] .sum(numeric_only=True) .reset_index() .round({'Value (ML)': 2}), ['region_NRM', 'Landuse'], ['region', 'name','data'], value_cols=('Year', 'Value (ML)'))
    water_overview_ag['type'] = 'column'
    water_overview_ag['color'] = water_overview_ag['name'].map(COLORS)

        
    out_dict = {}
    for region, df in water_overview_ag.groupby('region'):
        df = df.drop(columns=['region'])
        out_dict[region] = df.to_dict(orient='records')

            
    filename = f'Water_overview_NRM_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
    # Am
    water_am = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .query('Type == "Agricultural Management"')
    
    water_overview_am = _groupby_to_records(water_am .groupby(['region_NRM', 'Agricultural Management', 'Year'])[['Value (ML)']] .sum(numeric_only=True) .reset_index() .round({'Value (ML)': 2}), ['region_NRM', 'Agricultural Management'], ['region', 'name', 'data'], value_cols=('Year', 'Value (ML)'))
    water_overview_am['type'] = 'column'
    water_overview_am['color'] = water_overview_am['name'].map(COLORS)

    out_dict = {}
    for region, df in water_overview_am.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')

    filename = f'Water_overview_NRM_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # Non-Ag
    water_nonag = water_net_yield_NRM_region\
        .query('`Water Supply` != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .query('Type == "Non-Agricultural Land-use"')

    water_overview_nonag = _groupby_to_records(water_nonag .groupby(['region_NRM', 'Landuse', 'Year'])[['Value (ML)']] .sum(numeric_only=True) .reset_index() .round({'Value (ML)': 2}), ['region_NRM', 'Landuse'], ['region', 'name', 'data'], value_cols=('Year', 'Value (ML)'))
    water_overview_nonag['type'] = 'column'
    water_overview_nonag['color'] = water_overview_nonag['name'].map(COLORS)

    out_dict = {}
    for region, df in water_overview_nonag.groupby('region'):
        df = df.drop(columns='region')
        out_dict[region] = df.to_dict(orient='records')

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
        .assign(region='AUSTRALIA')
        
    water_ag = pd.concat([
        water_ag_AUS,
        water_net_yield_NRM_region.query('Type == "Agricultural Land-use" and region_NRM != "AUSTRALIA"').rename(columns={'region_NRM': 'region'})
        ], ignore_index=True)\
        .query('Landuse != "ALL" and `Water Supply` != "ALL"')

    df_region_wide = water_ag.groupby(['region', 'Water Supply', 'Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: [[int(r[0]), r[1]] for r in x[['Year', 'Value (ML)']].values.tolist()])\
        .reset_index()
  
    df_region_wide.columns = ['region', 'water', 'name',  'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide['name'].map(COLORS)
    df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water), df in df_region_wide.groupby(['region', 'water']):
        df = df.drop(['region'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water] = df.to_dict(orient='records')
        
        
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
        .assign(region='AUSTRALIA')

    water_am = pd.concat(
        [water_am_AUS,
         water_net_yield_NRM_region\
             .query('Type == "Agricultural Management" and region_NRM != "AUSTRALIA"')\
             .rename(columns={'region_NRM': 'region'})],
        ignore_index=True
    ).query('`Agricultural Management` != "ALL"')

    df_region_wide = water_am.groupby(['region', 'Water Supply', 'Landuse', 'Agricultural Management'])[['Year','Value (ML)']]\
        .apply(lambda x: [[int(r[0]), r[1]] for r in x[['Year', 'Value (ML)']].values.tolist()])\
        .reset_index()
    df_region_wide.columns = ['region', 'water', 'landuse', 'name',  'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for (region, water, landuse), df in df_region_wide.groupby(['region', 'water', 'landuse']):
        df = df.drop(['region', 'water', 'landuse'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water][landuse] = df.to_dict(orient='records')

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
            name_order=lambda x: x['Landuse'].apply(lambda y: LANDUSE_ALL_RENAMED.index(y)))\
        .sort_values('name_order')\
        .drop(columns=['name_order'])

    water_nonag = pd.concat([
        water_nonag_AUS,
        water_net_yield_NRM_region.query('Type == "Non-Agricultural Land-use" and Landuse != "ALL" and region_NRM != "AUSTRALIA"').rename(columns={'region_NRM': 'region'})
        ], ignore_index=True)

    df_region_wide = water_nonag.groupby(['region', 'Landuse'])[['Year','Value (ML)']]\
        .apply(lambda x: [[int(r[0]), r[1]] for r in x[['Year', 'Value (ML)']].values.tolist()])\
        .reset_index()
    df_region_wide.columns = ['region', 'name', 'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide['name'].map(COLORS)
    df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_region_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
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

    # AUS aggregate: sum over all NRM regions
    water_sum_AUS = water_sum_nrm\
        .groupby(['Water Supply', 'Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True).reset_index().assign(region='AUSTRALIA')

    water_sum = pd.concat([
        water_sum_AUS,
        water_sum_nrm.rename(columns={'region_NRM': 'region'})
    ], ignore_index=True)

    # Add ALL water aggregate
    water_sum_all_water = water_sum\
        .groupby(['region', 'Landuse', 'Year'])[['Value (ML)']]\
        .sum(numeric_only=True).reset_index().assign(**{'Water Supply': 'ALL'})
    water_sum = pd.concat([water_sum_all_water, water_sum], ignore_index=True)

    df_region_wide = water_sum.groupby(['region', 'Water Supply', 'Landuse'])[['Year', 'Value (ML)']]\
        .apply(lambda x: [[int(r[0]), r[1]] for r in x[['Year', 'Value (ML)']].values.tolist()])\
        .reset_index()
    df_region_wide.columns = ['region', 'water', 'name', 'data']
    df_region_wide['type'] = 'column'
    df_region_wide['color'] = df_region_wide['name'].map(COLORS)
    df_region_wide['name_order'] = df_region_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_region_wide = df_region_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for (region, water), df in df_region_wide.groupby(['region', 'water']):
        df = df.drop(['region'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        out_dict[region][water] = df.to_dict(orient='records')

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
        pd.concat([df for path in trans_area_files['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
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
    for (yr, region, from_water, to_water), grp in trans_area_df.groupby(
        ['Year', 'region', 'From-water-supply', 'To-water-supply']
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

        out_dict.setdefault(region, {}).setdefault(from_water, {}).setdefault(to_water, {})[yr] = {
            'x_categories': all_x_lus,   # To-LU: wrapped labels for Highcharts x-axis
            'y_categories': all_lus,      # From-LU: plain labels for Highcharts y-axis
            'data': points,
        }

    # Set a consistent max_val across all years for a uniform legend scale
    global_max_val = round(global_max_val, 2)
    for region_dict in out_dict.values():
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
        pd.concat([df for path in trans_area_ag2nonag_files['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
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
    for (yr, region, from_water, to_water), grp in trans_area_ag2nonag_df.groupby(
        ['Year', 'region', 'From-water-supply', 'To-water-supply']
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
        out_ag2nonag_dict.setdefault(region, {}).setdefault(from_water, {}).setdefault(to_water, {})[yr] = {
            'x_categories': all_x_lus_ag2nonag,
            'y_categories': all_y_lus_ag2nonag,
            'data': points,
        }

    # Set a consistent max_val across all years for a uniform legend scale
    global_max_val_ag2nonag = round(global_max_val_ag2nonag, 2)
    for region_dict in out_ag2nonag_dict.values():
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
            pd.concat([df for path in trans_cost_ag2ag_files['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
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
        for (yr, region, cost_type), grp in trans_cost_ag2ag_df.groupby(['Year', 'region', 'Cost-type']):
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
            out_cost_ag2ag_dict.setdefault(region, {}).setdefault(cost_type, {})[yr] = {
                'x_categories': all_lus_cost_ag2ag,
                'y_categories': all_lus_cost_ag2ag,
                'data': points,
            }

        global_max_val_cost_ag2ag = round(global_max_val_cost_ag2ag, 2)
        for region_dict in out_cost_ag2ag_dict.values():
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
            pd.concat([df for path in trans_cost_ag2nonag_files['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)
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
        for (yr, region, cost_type), grp in trans_cost_ag2nonag_df.groupby(['Year', 'region', 'Cost-type']):
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
            out_cost_ag2nonag_dict.setdefault(region, {}).setdefault(cost_type, {})[yr] = {
                'x_categories': all_x_lus_cost_ag2nonag,
                'y_categories': all_y_lus_cost_ag2nonag,
                'data': points,
            }

        global_max_val_cost_ag2nonag = round(global_max_val_cost_ag2nonag, 2)
        for region_dict in out_cost_ag2nonag_dict.values():
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
    bio_df = pd.concat([df for path in bio_paths['path'] if not (df := pd.read_csv(path)).empty], ignore_index=True)\
        .replace(RENAME_AM_NON_AG)\
        .infer_objects(copy=False)\
        .rename(columns={'Contribution Relative to Base Year Level (%)': 'Value (%)'})\
        .round({'Value (%)': 6})
        
    # ---------------- Overall quality - Ranking -----------------
    bio_rank_type = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby(['Year', 'region', 'Type'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Type', 'Value (%)'], ascending=[True, True, False])\
        .assign(Rank=lambda x: x.groupby(['Year','Type']).cumcount())\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))
    bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby(['Year', 'region'])\
        .sum(numeric_only=True)\
        .reset_index()\
        .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
        .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
        .assign(color=lambda x: x['Rank'].map(get_rank_color))\
        .assign(Type='Total')
    bio_rank = pd.concat([bio_rank_type, bio_rank_total], axis=0, ignore_index=True)
    
    
    for region, df in bio_rank_total.groupby('region'):
        if region not in bio_rank_dict:
            bio_rank_dict[region] = {}
            bio_rank_dict[region]['Quality'] = {}

        df = df.drop(columns='region')
        bio_rank_dict[region]['Quality']['Rank']  = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        bio_rank_dict[region]['Quality']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        bio_rank_dict[region]['Quality']['value'] = df.set_index('Year')['Value (%)'].apply( lambda x: format_with_suffix(x)).to_dict()


    out_dict = {}
    for (region, _type), df in bio_rank.groupby(['region', 'Type']):
        if region not in out_dict:
            out_dict[region] = {}
        if _type not in out_dict[region]:
            out_dict[region][_type] = {}
        out_dict[region][_type]['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
        out_dict[region][_type]['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
        out_dict[region][_type]['value'] = df.set_index('Year')['Value (%)'].apply( lambda x: format_with_suffix(x)).to_dict()
        
    filename = 'BIO_quality_ranking'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')




    # ---------------- Overall quality - Overview ----------------
    # sum
    df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby(['region', 'Year', 'Type'])\
        .sum(numeric_only=True)\
        .reset_index()
    df_wide = _groupby_to_records(df_region, ['Type', 'region'], ['name', 'region', 'data'])
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

    filename = f'BIO_quality_overview_sum'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    
    
    
    # ag
    df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Land-use"')\
        .groupby(['Year', 'region', 'Landuse'])\
        .sum(numeric_only=True)\
        .reset_index()
    df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'])
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])


    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

    filename = f'BIO_quality_overview_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')

    # am
    df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Management"')\
        .groupby(['Year', 'region', "Agricultural Management"])\
        .sum(numeric_only=True)\
        .reset_index()
    df_wide = _groupby_to_records(df_region, ["Agricultural Management", 'region'], ['name', 'region', 'data'])
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)


    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

    filename = f'BIO_quality_overview_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')


    # non-ag
    df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')\
        .groupby(['Year', 'region', 'Landuse'])\
        .sum(numeric_only=True)\
        .reset_index()
    df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'])
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')
        
    filename = f'BIO_quality_overview_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # ---------------- Overall quality - Ag ----------------
    bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()
    bio_df_wide = _groupby_to_records(bio_df_ag, ['region', 'Water_supply', 'Landuse'], ['region', 'water', 'name', 'data'])
    bio_df_wide['type'] = 'column'
    bio_df_wide['color'] = bio_df_wide['name'].map(COLORS)
    bio_df_wide['name_order'] = bio_df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))   
    bio_df_wide = bio_df_wide.sort_values('name_order').drop(columns=['name_order'])
    
    out_dict = {}
    for (region,water), df in bio_df_wide.groupby(['region', 'water']):
        df = df.drop(['region', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if water not in out_dict[region]:
            out_dict[region][water] = {}
        out_dict[region][water] = df.to_dict(orient='records')

    filename = f'BIO_quality_Ag'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
    # ---------------- Overall quality - Am ----------------
    bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()

    df_wide = _groupby_to_records(bio_df_am, ['region', 'Water_supply', "Agricultural Management", 'Landuse'], ['region', 'water', 'am', 'name', 'data'])
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)

    df_wide_all_am = _groupby_to_records(bio_df .query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"'), ['region', 'Water_supply', 'Landuse'], ['region', 'water', 'name', 'data'])
    df_wide_all_am['am'] = 'ALL'
    df_wide_all_am['type'] = 'column'
    df_wide_all_am['color'] = df_wide_all_am['name'].map(COLORS)

    df_wide = pd.concat([df_wide, df_wide_all_am], ignore_index=True)

    out_dict = {}
    for (region, am, water), df in df_wide.groupby(['region', 'am', 'water']):
        df = df.drop(['region', 'am', 'water'], axis=1)
        if region not in out_dict:
            out_dict[region] = {}
        if am not in out_dict[region]:
            out_dict[region][am] = {}
        if water not in out_dict[region][am]:
            out_dict[region][am][water] = {}
        out_dict[region][am][water] = df.to_dict(orient='records')

    filename = f'BIO_quality_Am'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
        
        
    # ---------------- Overall quality - Non-Ag ----------------
    df_wide = _groupby_to_records(bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"'), ['region', 'Landuse'], ['region', 'name', 'data'])
    df_wide['type'] = 'column'
    df_wide['color'] = df_wide['name'].map(COLORS)
    df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
    df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

    out_dict = {}
    for region, df in df_wide.groupby('region'):
        df = df.drop(['region'], axis=1)
        out_dict[region] = df.to_dict(orient='records')

    filename = f'BIO_quality_NonAg'
    with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(out_dict, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    
    
        
    if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':

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
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Area Weighted Score (ha)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if region not in bio_rank_dict:
                bio_rank_dict[region] = {}
            if 'GBF2' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF2'] = {}

            bio_rank_dict[region]['GBF2']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF2']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF2']['value'] = df.set_index('Year')['Area Weighted Score (ha)'].apply( lambda x: format_with_suffix(x)).to_dict()
            

        # ---------------- (GBF2) overview  ----------------
        
        # sum
        bio_df_target = bio_df.groupby(['Year'])[['Priority Target (%)']].agg('first').reset_index()
        bio_df_target = bio_df_target[['Year','Priority Target (%)']].values.tolist()

        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Type', 'region'], ['name', 'region', 'data'])
        df_wide['type'] = 'column'

        df_wide.loc[len(df_wide)] = ['Target (%)', 'AUSTRALIA', bio_df_target, 'line']
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')



        # ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'])
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        
        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
                
        # am
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Management"')\
            .groupby(['Year', 'region', 'Agricultural Management'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Agricultural Management', 'region'], ['name', 'region', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # non-ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'])
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))


        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
        # ---------------- (GBF2) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()
        df_region = _groupby_to_records(bio_df_ag, ['region', 'Water_supply', 'Landuse'], ['region', 'water', 'name', 'data'])
        df_region['type'] = 'column'
        df_region['color'] = df_region['name'].map(COLORS)
        df_region['name_order'] = df_region['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_region = df_region.sort_values('name_order').drop(columns=['name_order'])
        
        out_dict = {}
        for (region, water), df in df_region.groupby(['region', 'water']):
            df = df.drop(['region', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if water not in out_dict[region]:
                out_dict[region][water] = {}
            out_dict[region][water] = df.to_dict(orient='records')
            
        filename = f'BIO_GBF2_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
        
        # ---------------- (GBF2) Ag-Mgt  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_am, ['region', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region', 'water', 'am', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)

        df_wide_all_am = _groupby_to_records(bio_df .query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"'), ['region', 'Water_supply', 'Landuse'], ['region', 'water', 'name', 'data'])
        df_wide_all_am['am'] = 'ALL'
        df_wide_all_am['type'] = 'column'
        df_wide_all_am['color'] = df_wide_all_am['name'].map(COLORS)

        df_wide = pd.concat([df_wide, df_wide_all_am], ignore_index=True)

        out_dict = {}
        for (region, am, water), df in df_wide.groupby(['region', 'am', 'water']):
            df = df.drop(['region', 'am', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if am not in out_dict[region]:
                out_dict[region][am] = {}
            if water not in out_dict[region][am]:
                out_dict[region][am][water] = {}
            out_dict[region][am][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
        # ---------------- (GBF2) Non-Ag  ----------------
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF2_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
            
            
            
    if settings.BIODIVERSITY_TARGET_GBF_3_NVIS != 'off' and settings.GBF3_NVIS_REGION_MODE != 'IBRA':
        filter_str = '''
            category == "biodiversity"
            and base_name.str.contains("biodiversity_GBF3_NVIS")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([df for path in bio_paths['path'] if not (df := pd.read_csv(path, low_memory=False)).empty])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Vegetation Group': 'species'})\
            .round(6)
        # Drop the per-species 'ALL' aggregate so it is not surfaced as a selectable
        # vegetation group in the report dropdowns; sum charts re-aggregate explicitly.
        bio_df = bio_df.query('species != "ALL"')

        # ---------------- (GBF3-NVIS) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Area Weighted Score (ha)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))
            
        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if region not in bio_rank_dict:
                bio_rank_dict[region] = {}
            if 'GBF3-NVIS' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF3-NVIS'] = {}
                
            bio_rank_dict[region]['GBF3-NVIS']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF3-NVIS']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF3-NVIS']['value'] = df.set_index('Year')['Area Weighted Score (ha)'].apply(lambda x: format_with_suffix(x)).to_dict()


        # ---------------- (GBF3-NVIS) Overview  ----------------
        
        # sum (contribution percentage; species == 'ALL' excluded)
        bio_df_target = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').groupby(['Year', 'species'])[['Target_by_Percent']].agg('first').reset_index()

        df_region = bio_df.query('species != "ALL" and Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Type', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_NVIS_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')



        # ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_NVIS_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # am
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Management"')\
            .groupby(['Year', 'region', 'Agricultural Management'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Agricultural Management', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_NVIS_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # non-ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])


        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_NVIS_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # ---------------- (GBF3-NVIS) - Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_ag, ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        df_wide = pd.concat([df_wide, _bio_outside_series(bio_df, 'Ag')], ignore_index=True)

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')
            
        filename = f'BIO_GBF3_NVIS_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
        # ---------------- (GBF3-NVIS) - Am  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_am, ['region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region', 'species', 'water', 'am', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)

        df_wide_all_am = _groupby_to_records(bio_df .query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"'), ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide_all_am['am'] = 'ALL'
        df_wide_all_am['type'] = 'column'
        df_wide_all_am['color'] = df_wide_all_am['name'].map(COLORS)

        df_wide = pd.concat([df_wide, df_wide_all_am, _bio_outside_series(bio_df, 'Am')], ignore_index=True)

        out_dict = {}
        for (region, species, am, water), df in df_wide.groupby(['region', 'species', 'am', 'water']):
            df = df.drop(['region', 'species', 'am', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if am not in out_dict[region][species]:
                out_dict[region][species][am] = {}
            if water not in out_dict[region][species][am]:
                out_dict[region][species][am][water] = {}
            out_dict[region][species][am][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_NVIS_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
        # ---------------- (GBF3-NVIS) - Non-Ag  ----------------
        df_wide = _groupby_to_records(bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"'), ['region', 'species', 'Landuse'], ['region', 'species', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        df_wide = pd.concat([df_wide, _bio_outside_series(bio_df, 'NonAg')], ignore_index=True)

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            out_dict[region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF3_NVIS_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
    # IBRA reporting branch disabled (GBF3 IBRA pipeline incomplete).


    if settings.BIODIVERSITY_TARGET_GBF_4_SNES == 'on':
        
        filter_str = '''
            category == "biodiversity" 
            and base_name.str.contains("biodiversity_GBF4_SNES_scores")
        '''.strip().replace('\n', '')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([df for path in bio_paths['path'] if not (df := pd.read_csv(path)).empty])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .round(6)
        # Drop the per-species 'ALL' aggregate so it is not surfaced as a selectable
        # species in the report dropdowns; sum charts re-aggregate explicitly.
        bio_df = bio_df.query('species != "ALL"')
        # ---------------- (GBF4 SNES) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))
            
        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if region not in bio_rank_dict:
                bio_rank_dict[region] = {}
            if 'GBF4 (SNES)' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF4 (SNES)'] = {}

            bio_rank_dict[region]['GBF4 (SNES)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF4 (SNES)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF4 (SNES)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()



        # ---------------- (GBF4 SNES) Overview  ----------------

        # sum (contribution percentage; species == 'ALL' excluded)
        df_region = bio_df.query('species != "ALL" and Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Type', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # am
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Management"')\
            .groupby(['Year', 'region', 'Agricultural Management'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Agricultural Management', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # non-ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF4 SNES) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_ag, ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        df_wide = pd.concat([df_wide, _bio_outside_series(bio_df, 'Ag')], ignore_index=True)

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # ---------------- (GBF4 SNES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_am, ['region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region', 'species', 'water', 'am', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)

        df_wide_all_am = _groupby_to_records(bio_df .query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"'), ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide_all_am['am'] = 'ALL'
        df_wide_all_am['type'] = 'column'
        df_wide_all_am['color'] = df_wide_all_am['name'].map(COLORS)

        df_wide = pd.concat([df_wide, df_wide_all_am, _bio_outside_series(bio_df, 'Am')], ignore_index=True)

        out_dict = {}
        for (region, species, am, water), df in df_wide.groupby(['region', 'species', 'am', 'water']):
            df = df.drop(['region', 'species', 'am', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if am not in out_dict[region][species]:
                out_dict[region][species][am] = {}
            if water not in out_dict[region][species][am]:
                out_dict[region][species][am][water] = {}
            out_dict[region][species][am][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF4 SNES) Non-ag  ----------------
        df_wide = _groupby_to_records(bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"'), ['region', 'species', 'Landuse'], ['region', 'species', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        df_wide = pd.concat([df_wide, _bio_outside_series(bio_df, 'NonAg')], ignore_index=True)

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            out_dict[region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_SNES_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
            
            
            
    if settings.BIODIVERSITY_TARGET_GBF_4_ECNES == 'on':
        
        bio_paths = files.query('base_name.str.contains("biodiversity_GBF4_ECNES_scores")')
        bio_df = pd.concat([df for path in bio_paths['path'] if not (df := pd.read_csv(path)).empty])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)'})\
            .round(6)
        # Drop the per-species 'ALL' aggregate so it is not surfaced as a selectable
        # species in the report dropdowns; sum charts re-aggregate explicitly.
        bio_df = bio_df.query('species != "ALL"')

        # ---------------- (GBF4 ECNES) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if region not in bio_rank_dict:
                bio_rank_dict[region] = {}
            if 'GBF4 (ECNES)' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF4 (ECNES)'] = {}

            bio_rank_dict[region]['GBF4 (ECNES)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF4 (ECNES)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF4 (ECNES)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

        # ---------------- (GBF4 ECNES) Overview  ----------------

        # sum (contribution percentage; species == 'ALL' excluded)
        df_region = bio_df.query('species != "ALL" and Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Type', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Value (%)'))
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # am
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Management"')\
            .groupby(['Year', 'region', 'Agricultural Management'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Agricultural Management', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].apply(lambda x: COLORS[x])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # non-ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF4 ECNES) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_ag, ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        df_wide = pd.concat([df_wide, _bio_outside_series(bio_df, 'Ag')], ignore_index=True)

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # ---------------- (GBF4 ECNES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_am, ['region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region', 'species', 'water', 'am', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)

        df_wide_all_am = _groupby_to_records(bio_df .query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"'), ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide_all_am['am'] = 'ALL'
        df_wide_all_am['type'] = 'column'
        df_wide_all_am['color'] = df_wide_all_am['name'].map(COLORS)

        df_wide = pd.concat([df_wide, df_wide_all_am, _bio_outside_series(bio_df, 'Am')], ignore_index=True)

        out_dict = {}
        for (region, species, am, water), df in df_wide.groupby(['region', 'species', 'am', 'water']):
            df = df.drop(['region', 'species', 'am', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if am not in out_dict[region][species]:
                out_dict[region][species][am] = {}
            if water not in out_dict[region][species][am]:
                out_dict[region][species][am][water] = {}
            out_dict[region][species][am][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF4 ECNES) Non-ag  ----------------
        df_wide = _groupby_to_records(bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"'), ['region', 'species', 'Landuse'], ['region', 'species', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])
        df_wide = pd.concat([df_wide, _bio_outside_series(bio_df, 'NonAg')], ignore_index=True)

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            out_dict[region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF4_ECNES_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
        
        
        

    if settings.BIODIVERSITY_TARGET_GBF_8 == 'on':
        
        filter_str = '''
            category == "biodiversity" 
            
            and base_name.str.contains("biodiversity_GBF8_species_scores")
        '''.strip().replace('\n','')
        
        bio_paths = files.query(filter_str).reset_index(drop=True)
        bio_df = pd.concat([df for path in bio_paths['path'] if not (df := pd.read_csv(path)).empty])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Species':'species'})\
            .round(6)

        # ---------------- (GBF8 SPECIES) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if region not in bio_rank_dict:
                bio_rank_dict[region] = {}
            if 'GBF8 (SPECIES)' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF8 (SPECIES)'] = {}

            bio_rank_dict[region]['GBF8 (SPECIES)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF8 (SPECIES)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF8 (SPECIES)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

        # ---------------- (GBF8 SPECIES) Overview  ----------------

        # sum
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Type', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # am
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Management"')\
            .groupby(['Year', 'region', 'Agricultural Management'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Agricultural Management', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # non-ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 SPECIES) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_ag, ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # ---------------- (GBF8 SPECIES) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_am, ['region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region', 'species', 'water', 'am', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)

        df_wide_all_am = _groupby_to_records(bio_df .query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"'), ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide_all_am['am'] = 'ALL'
        df_wide_all_am['type'] = 'column'
        df_wide_all_am['color'] = df_wide_all_am['name'].map(COLORS)

        df_wide = pd.concat([df_wide, df_wide_all_am], ignore_index=True)

        out_dict = {}
        for (region, species, am, water), df in df_wide.groupby(['region', 'species', 'am', 'water']):
            df = df.drop(['region', 'species', 'am', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if am not in out_dict[region][species]:
                out_dict[region][species][am] = {}
            if water not in out_dict[region][species][am]:
                out_dict[region][species][am][water] = {}
            out_dict[region][species][am][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 SPECIES) Non-ag  ----------------
        df_wide = _groupby_to_records(bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"'), ['region', 'species', 'Landuse'], ['region', 'species', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            out_dict[region][species] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_SPECIES_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')
        
        
    
        # ---------------- (GBF8 GROUP)  ----------------
        bio_paths = files.query('base_name.str.contains("biodiversity_GBF8_groups_scores")')
        bio_df = pd.concat([df for path in bio_paths['path'] if not (df := pd.read_csv(path)).empty])
        bio_df = bio_df.replace(RENAME_AM_NON_AG)\
            .infer_objects(copy=False)\
            .rename(columns={'Contribution Relative to Pre-1750 Level (%)': 'Value (%)', 'Group':'species'})\
            .round(6)

        # ---------------- (GBF8 GROUP) Ranking  ----------------
        bio_rank_total = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region'])\
            .sum(numeric_only=True)\
            .reset_index()\
            .sort_values(['Year', 'Value (%)'], ascending=[True, False])\
            .assign(Rank=lambda x: x.groupby(['Year']).cumcount())\
            .assign(Type='Total')\
            .assign(color=lambda x: x['Rank'].map(get_rank_color))

        for region, df in bio_rank_total.groupby('region'):
            df = df.drop(columns='region')
            if region not in bio_rank_dict:
                bio_rank_dict[region] = {}
            if 'GBF8 (GROUP)' not in bio_rank_dict[region]:
                bio_rank_dict[region]['GBF8 (GROUP)'] = {}

            bio_rank_dict[region]['GBF8 (GROUP)']['Rank'] = df.set_index('Year')['Rank'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF8 (GROUP)']['color'] = df.set_index('Year')['color'].replace({np.nan: None}).to_dict()
            bio_rank_dict[region]['GBF8 (GROUP)']['value'] = df.set_index('Year')['Value (%)'].apply(lambda x: format_with_suffix(x)).to_dict()

        # ---------------- (GBF8 GROUP) Overview  ----------------

        # sum
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"')\
            .groupby(['Year', 'region', 'Type'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Type', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_overview_sum'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_overview_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # am
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Agricultural Management"')\
            .groupby(['Year', 'region', 'Agricultural Management'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Agricultural Management', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_overview_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # non-ag
        df_region = bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"')\
            .groupby(['Year', 'region', 'Landuse'])\
            .sum(numeric_only=True)\
            .reset_index()
        df_wide = _groupby_to_records(df_region, ['Landuse', 'region'], ['name', 'region', 'data'], value_cols=('Year', 'Area Weighted Score (ha)'))
        df_wide['type'] = 'column'

        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for region, df in df_wide.groupby('region'):
            df = df.drop(['region'], axis=1)
            out_dict[region] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_overview_NonAg'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 GROUP) Ag  ----------------
        bio_df_ag = bio_df.query('Type == "Agricultural Land-use" and Landuse != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_ag, ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species, water), df in df_wide.groupby(['region', 'species', 'water']):
            df = df.drop(['region', 'species', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if water not in out_dict[region][species]:
                out_dict[region][species][water] = {}
            out_dict[region][species][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_Ag'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')


        # ---------------- (GBF8 GROUP) Agricultural Management  ----------------
        bio_df_am = bio_df.query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` != "ALL"').copy()

        df_wide = _groupby_to_records(bio_df_am, ['region', 'species', 'Water_supply', 'Agricultural Management', 'Landuse'], ['region', 'species', 'water', 'am', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)

        df_wide_all_am = _groupby_to_records(bio_df .query('Type == "Agricultural Management" and Landuse != "ALL" and `Agricultural Management` == "ALL"'), ['region', 'species', 'Water_supply', 'Landuse'], ['region', 'species', 'water', 'name', 'data'])
        df_wide_all_am['am'] = 'ALL'
        df_wide_all_am['type'] = 'column'
        df_wide_all_am['color'] = df_wide_all_am['name'].map(COLORS)

        df_wide = pd.concat([df_wide, df_wide_all_am], ignore_index=True)

        out_dict = {}
        for (region, species, am, water), df in df_wide.groupby(['region', 'species', 'am', 'water']):
            df = df.drop(['region', 'species', 'am', 'water'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            if am not in out_dict[region][species]:
                out_dict[region][species][am] = {}
            if water not in out_dict[region][species][am]:
                out_dict[region][species][am][water] = {}
            out_dict[region][species][am][water] = df.to_dict(orient='records')

        filename = f'BIO_GBF8_GROUP_Am'
        with open(f'{SAVE_DIR}/{filename}.js', 'w') as f:
            f.write(f'window["{filename}"] = ')
            json.dump(out_dict, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        # ---------------- (GBF8 GROUP) Non-ag  ----------------
        df_wide = _groupby_to_records(bio_df.query('Water_supply != "ALL" and Landuse != "ALL" and `Agricultural Management` != "ALL"').query('Type == "Non-Agricultural Land-use"'), ['region', 'species', 'Landuse'], ['region', 'species', 'name', 'data'])
        df_wide['type'] = 'column'
        df_wide['color'] = df_wide['name'].map(COLORS)
        df_wide['name_order'] = df_wide['name'].apply(lambda x: LANDUSE_ALL_RENAMED.index(x))
        df_wide = df_wide.sort_values('name_order').drop(columns=['name_order'])

        out_dict = {}
        for (region, species), df in df_wide.groupby(['region', 'species']):
            df = df.drop(['region', 'species'], axis=1)
            if region not in out_dict:
                out_dict[region] = {}
            if species not in out_dict[region]:
                out_dict[region][species] = {}
            out_dict[region][species] = df.to_dict(orient='records')

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

    supporting = {
        'model_run_settings': settings_dict,
        'years': years,
        'colors': COLORS,
        'COLORSing': COLORS,
        'mem_logs': mem_logs_obj,
        'renewables_enabled': any(settings.RENEWABLES_OPTIONS.values()),
        'GBF3_NVIS_REGION_MODE': settings.GBF3_NVIS_REGION_MODE,
    }
    
    filename = 'Supporting_info'
    with open(f"{SAVE_DIR}/{filename}.js", 'w') as f:
        f.write(f'window["{filename}"] = ')
        json.dump(supporting, f, separators=(',', ':'), indent=2)
        f.write(';\n')
    


    return "Supporting information data processing completed"
