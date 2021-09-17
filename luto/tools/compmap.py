#!/bin/env python3
#
# compmap.py - to compare neoLUTO spatial arrays.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
#
# Created: 2021-09-17
# Last modified: 2021-09-17
#

import numpy as np
import pandas as pd

def crossmap(oldmap, newmap, landuses=None):
    """Return cross-tabulation matrix and net switching counts."""

    # Produce the cross-tabulation matrix with optional labels.
    crosstab = pd.crosstab(oldmap, newmap, margins=True)
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
