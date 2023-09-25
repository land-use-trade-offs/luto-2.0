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
from luto.ag_managements import AG_MANAGEMENTS_INDEXING, SORTED_AG_MANAGEMENTS, AG_MANAGEMENTS_TO_LAND_USES


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


def ammap_crossmap(oldmap, newmap):
    # Produce the cross-tabulation matrix with optional labels.
    crosstab = pd.crosstab(oldmap, newmap, margins = True)

    reindex =  list(range(len(SORTED_AG_MANAGEMENTS) + 1)) + ['All']
    crosstab = crosstab.reindex(reindex, axis = 0, fill_value = 0)
    crosstab = crosstab.reindex(reindex, axis = 1, fill_value = 0)

    crosstab.columns = ['None'] + SORTED_AG_MANAGEMENTS + ['Total']
    crosstab.index = ['None'] + SORTED_AG_MANAGEMENTS + ['Total']

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


def crossmap_amstat( lumap_old
                   , ammap_old
                   , lumap_new
                   , ammap_new
                   , ag_landuses
                   , non_ag_landuses ):

    # Make new land maps where each land use / ag man combination is encoded as a separate number
    # First, make a mapping from new codes to names, then a mapping from the old lummap and ammap codes to the new codes
    n_am = len(SORTED_AG_MANAGEMENTS)
    lu_names_map = {}
    old_codes_to_new_codes = {}
    lu_names_map[-1] = 'Non-agricultural land (None)'
    
    lu2am = defaultdict(list)
    for am, lu_list in AG_MANAGEMENTS_TO_LAND_USES.items():
        for lu in lu_list:
            lu2am[lu].append(am)
    am2mapcode = {am: code for code, am in AG_MANAGEMENTS_INDEXING.items()}

    counter = 0
    for j, lu in enumerate(ag_landuses):
        lu_names_map[counter] = lu + ' (None)'
        old_codes_to_new_codes[j, 0] = counter
        counter += 1

        for am in lu2am[lu]:
            lu_names_map[counter] = f'{lu} ({am})'
            old_codes_to_new_codes[j, am2mapcode[am]] = counter
            counter += 1

    base = settings.NON_AGRICULTURAL_LU_BASE_CODE
    for k, lu in enumerate(non_ag_landuses):
        lu_names_map[counter] = lu
        old_codes_to_new_codes[k + base, 0] = counter
        counter += 1

    n_new_codes = counter
    
    # Make the new land maps with the new codes
    old_map_highpos = np.zeros(len(lumap_old))
    for idx, j in enumerate(lumap_old):
        n = ammap_old[idx]
        old_map_highpos[idx] = old_codes_to_new_codes[j, n]

    new_map_highpos = np.zeros(len(lumap_new))
    for idx, j in enumerate(lumap_new):
        n = ammap_new[idx]
        new_map_highpos[idx] = old_codes_to_new_codes[j, n]

    # Make crosstab
    crosstab = pd.crosstab(old_map_highpos, new_map_highpos, margins=True)

    # Rename index and columns of crosstab to match the land use / ag management names for readability
    reindex = list(range(n_new_codes)) + ['All']
    crosstab = crosstab.reindex(reindex, axis=0, fill_value=0)
    crosstab = crosstab.reindex(reindex, axis=1, fill_value=0)

    names = [lu_names_map[i] for i in crosstab.columns if i != 'All']
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
