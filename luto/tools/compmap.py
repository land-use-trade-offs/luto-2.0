#!/bin/env python3
#
# compmap.py - to compare neoLUTO spatial arrays.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
#
# Created: 2021-09-17
# Last modified: 2022-01-21
#

import numpy as np
import pandas as pd

def crossmap(oldmap, newmap, landuses=None):
    """Return cross-tabulation matrix and net switching counts."""

    # Produce the cross-tabulation matrix with optional labels.
    crosstab = pd.crosstab(oldmap, newmap, margins=True)
    # Make sure the cross-tabulation matrix is square.
    crosstab = crosstab.reindex(crosstab.index, axis=1, fill_value=0)
    if landuses is not None:
        lus = landuses + ['Total']
        crosstab.columns = lus
        crosstab.index = lus

    # Calculate net switches to land use (negative means switch away).
    switches = crosstab.iloc[-1, 0:-1] - crosstab.iloc[0:-1, -1]
    nswitches = np.abs(switches).sum()
    switches['Total'] = nswitches
    switches['Total [%]'] = int(np.around(100 * nswitches / oldmap.shape[0]))

    return crosstab, switches

def crossmap_irrstat(lumap_old, lmmap_old, lumap_new, lmmap_new, landuses=None):
    if landuses is not None:
        ludict = {}
        ludict[-2] = 'Non-agricultural land (dry)'
        ludict[-1] = 'Non-agricultural land (irr)'
        for j, lu in enumerate(landuses):
            ludict[j*2] = lu + ' (dry)'
            ludict[j*2 + 1] = lu + ' (irr)'

    highpos_old = np.where(lmmap_old == 0, lumap_old * 2, lumap_old * 2 + 1)
    highpos_new = np.where(lmmap_new == 0, lumap_new * 2, lumap_new * 2 + 1)

    # Produce the cross-tabulation matrix with labels.
    crosstab = pd.crosstab(highpos_old, highpos_new, margins=True)
    # Make sure the cross-tabulation matrix is square.
    crosstab = crosstab.reindex(crosstab.index, axis=1, fill_value=0)
    names = [ludict[i] for i in crosstab.columns if i != 'All']
    names.append('All')
    # columns = [ludict[i] for i in crosstab.columns if i != 'All']
    # columns.append('All')
    # index = [ludict[i] for i in crosstab.index if i != 'All']
    # index.append('All')
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





