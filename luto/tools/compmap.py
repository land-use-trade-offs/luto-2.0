# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
#
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>.

"""
To compare LUTO 2.0 spatial arrays.
"""


import numpy as np
import pandas as pd
from collections import defaultdict

import luto.settings as settings
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


def lumap_crossmap(oldmap, newmap, ag_landuses, non_ag_landuses):
    # Produce the cross-tabulation matrix with optional labels.
    crosstab = pd.crosstab(oldmap, newmap, margins = True)

    # Need to make sure each land use (both agricultural and non-agricultural) appears in the index/columns
    reindex = (
            list(range(len(ag_landuses)))
        + [ settings.NON_AGRICULTURAL_LU_BASE_CODE + lu for lu in range(len(non_ag_landuses)) ]
        + [ "All" ]
    )
    crosstab = crosstab.reindex(reindex, axis = 0, fill_value = 0)
    crosstab = crosstab.reindex(reindex, axis = 1, fill_value = 0)

    lus = ag_landuses + non_ag_landuses + ['Total']
    crosstab.columns = lus
    crosstab.index = lus

    # Calculate net switches to land use (negative means switch away).
    switches = crosstab.iloc[-1, 0:-1] - crosstab.iloc[0:-1, -1]
    nswitches = np.abs(switches).sum()
    switches['Total'] = nswitches
    switches['Total [%]'] = int(np.around(100 * nswitches / oldmap.shape[0]))

    return crosstab, switches


def lmmap_crossmap(oldmap, newmap):
    # Produce the cross-tabulation matrix with optional labels.
    crosstab = pd.crosstab(oldmap, newmap, margins = True)
    crosstab = crosstab.reindex(crosstab.index, axis=1, fill_value=0)

    # Calculate net switches to land use (negative means switch away).
    switches = crosstab.iloc[-1, 0:-1] - crosstab.iloc[0:-1, -1]
    nswitches = np.abs(switches).sum()
    switches['Total'] = nswitches
    switches['Total [%]'] = int(np.around(100 * nswitches / oldmap.shape[0]))

    return crosstab, switches


def ammap_crossmap(oldmap, newmap, am):
    # Produce the cross-tabulation matrix with optional labels for a single ammap.
    crosstab = pd.crosstab(oldmap, newmap, margins = True)

    reindex =  [0, 1, 'All']
    crosstab = crosstab.reindex(reindex, axis = 0, fill_value = 0)
    crosstab = crosstab.reindex(reindex, axis = 1, fill_value = 0)

    ind_names = ['(None)', am, 'Total']
    crosstab.columns = ind_names
    crosstab.index = ind_names

    # Calculate net switches to land use (negative means switch away).
    switches = crosstab.iloc[-1, 0:-1] - crosstab.iloc[0:-1, -1]
    nswitches = np.abs(switches).sum()
    switches['Total'] = nswitches
    switches['Total [%]'] = int(np.around(100 * nswitches / oldmap.shape[0]))

    return crosstab, switches






def crossmap_irrstat( lumap_old
                    , lmmap_old
                    , lumap_new
                    , lmmap_new
                    , ag_landuses
                    , non_ag_landuses ):

    ludict = {}
    ludict[-2] = 'Non-agricultural land (dry)'
    ludict[-1] = 'Non-agricultural land (irr)'

    for j, lu in enumerate(ag_landuses):
        ludict[j*2] = lu + ' (dry)'
        ludict[j*2 + 1] = lu + ' (irr)'

    base = settings.NON_AGRICULTURAL_LU_BASE_CODE
    for k, lu in enumerate(non_ag_landuses):
        ludict[k + base] = lu

    highpos_old = np.where(lmmap_old == 0, lumap_old * 2, lumap_old * 2 + 1)
    highpos_new = np.where(lmmap_new == 0, lumap_new * 2, lumap_new * 2 + 1)

    # Produce the cross-tabulation matrix with labels.
    crosstab = pd.crosstab(highpos_old, highpos_new, margins=True)
    # Make sure the cross-tabulation matrix is square.
    crosstab = crosstab.reindex(crosstab.index, axis=1, fill_value=0)
    names = [ludict[i] for i in crosstab.columns if i != 'All']
    names.append('All')
    crosstab.columns = names
    crosstab.index = names

    # Calculate net switches to land use (negative means switch away).
    df = pd.DataFrame()
    cells_prior = crosstab.iloc[:, -1]
    cells_after = crosstab.iloc[-1, :]
    df['Cells prior [ # ]'] = cells_prior
    df['Cells after [ # ]'] = cells_after
    df.fillna(0, inplace=True)
    df = df.astype(np.int64)
    switches = df['Cells after [ # ]']-  df['Cells prior [ # ]']
    nswitches = np.abs(switches).sum()
    pswitches = int(np.around(100 * nswitches / lumap_old.shape[0]))

    df['Switches [ # ]'] = switches
    df['Switches [ % ]'] = 100 * switches / cells_prior
    df.loc['Total'] = cells_prior.sum(), cells_after.sum(), nswitches, pswitches

    return crosstab, df


def crossmap_amstat( am
                   , lumap_old
                   , ammap_old
                   , lumap_new
                   , ammap_new
                   , ag_landuses ):
    """
    For a given agricultural management option, make a crossmap showing cell changes at a land use level.
    """
    ludict = {}
    ludict[-2] = 'Non-agricultural land (None)'

    for j, lu in enumerate(ag_landuses):
        ludict[j*2] = lu + ' (None)'
        ludict[j*2 + 1] = lu + f' ({am})'

    # Collect all j values that apply to the given AM option
    am_j = [j for j, lu_name in enumerate(ag_landuses) if lu_name in AG_MANAGEMENTS_TO_LAND_USES[am]]
    am_j = np.array(am_j)

    # Encode information about the given AM usage by doubling up land uses
    highpos_old = np.where(ammap_old == 0, lumap_old * 2, lumap_old * 2 + 1)
    highpos_new = np.where(ammap_new == 0, lumap_new * 2, lumap_new * 2 + 1)

    # Produce the cross-tabulation matrix with labels.
    crosstab = pd.crosstab(highpos_old, highpos_new, margins=True)
    # Include all land uses and AM versions of land uses in the crosstab
    non_am_idx = list(range(0, 2 * len(ag_landuses), 2))
    am_idx = [2*i + 1 for i in am_j]
    reindex = sorted(non_am_idx + am_idx) + ['All']
    crosstab = crosstab.reindex(reindex, axis=0, fill_value=0)
    crosstab = crosstab.reindex(reindex, axis=1, fill_value=0)

    names = [ludict[i] for i in crosstab.columns if i != 'All']
    names.append('All')
    crosstab.columns = names
    crosstab.index = names

    # Calculate net switches to land use (negative means switch away).
    df = pd.DataFrame()
    am_lus = [ludict[i] for i in am_idx]
    cells_prior = crosstab.iloc[:, -1].loc[am_lus]
    cells_after = crosstab.iloc[-1, :].loc[am_lus]
    df['Cells prior [ # ]'] = cells_prior
    df['Cells after [ # ]'] = cells_after
    df.fillna(0, inplace=True)
    df = df.astype(np.int64)
    switches = df['Cells after [ # ]']-  df['Cells prior [ # ]']
    nswitches = np.abs(switches).sum()
    pswitches = int(np.around(100 * nswitches / lumap_old.shape[0]))

    df['Switches [ # ]'] = switches
    df['Switches [ % ]'] = 100 * switches / cells_prior
    df.loc['Total'] = cells_prior.sum(), cells_after.sum(), nswitches, pswitches

    return crosstab, df




def df_spare2densi(df):
    '''Function to fill a multilevel df to sensified format
    Input:
        pd.DataFrame

    Output:
        pd.DataFrame
    
    What does that mean?
    For example, a df has multilevel columns of ([A,(1,3)],[B,(2,4)]),
    after passing this function, it will become ([A,(1,2,3,4)], [B,(1,2,3,4)])
    
    Why this is necessary?
    Because we want to perform matrics multiplication using the input df,
    and this function fill nan values to "missing" colums, e.g., [A,(2,4)]
    so that the output df has a nice rectangular shape to be multiplied with

    
    '''

    # convert the mulilevel columns to a df, get the {unique value} and {count} of each level
    df_col = pd.DataFrame(df.columns.tolist())
    df_col_unique = {idx:df_col[idx].unique().tolist() for idx in df_col.columns}
    df_col_unique = {k:sorted(v) for k,v in df_col_unique.items()} # IMPORTANT, to keep the columns in lexicall order
    df_col_count = dict(df_col.nunique())
    
    # get the product from column of all levels
    df_col_product = list(product(*df_col_unique.values()))
    df_col_product = sorted(df_col_product,key=lambda x:[x[i] for i in range(df_col.shape[1])]) # IMPORTANT, to order the columns
    
    # expande the original df with df_col_product 
    # so that we can finally convert it to a n-d rectangular np.array 
    expand_df = df.reindex(columns=df_col_product,fill_value=np.nan)

    return expand_df