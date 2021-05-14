#!/bin/env python3
#
# spatial.py - sub module to load and prepare spatially explicit data.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-04-13
# Last modified: 2021-04-30
#

import importlib
import os.path

import pandas as pd
import numpy as np

from luto.data import INPUT_DIR

# Two key input .csv files from which all data are constructed.
#
# All the relevant data by SLA06 or whatever other unit of aggregation.
costs = pd.read_csv(os.path.join(INPUT_DIR, 'data-sla06.csv'))
# Concordance between LUTO cells and SLA06 IDs or other units of aggregation.
cells = pd.read_csv(os.path.join(INPUT_DIR, 'cell-sla06.csv'))

# Extract the sorted list of land uses from the data and produce derivative lists.
lus = sorted(costs['SPREAD_CLASS'].unique().tolist())
nlus = len(lus)
luirrs = [(lu, irr) for lu in lus for irr in (0, 1)]
nluirrs = len(luirrs)

landuses = []
for li in luirrs:
    if li[1] == 0: irrstatus = 'dry'
    else: irrstatus = 'irr'
    landuses.append(li[0] + '_' + irrstatus)

# Select the columns of interest.
cols = [ 'Q1'    # Quantities of primary product.
       , 'Q2'    # Quantities of secondary product.
       , 'QC'    # Quantity Costs.
       , 'AC'    # Area Costs.
       , 'FOC'   # Fixed Operation Costs.
       , 'FLC'   # Fixed Labour Costs.
       , 'FDC'   # Fixed Depreciation Costs.
       , 'WR'    # Water Required.
       , 'WP'    # Water Price.
       , 'WC'    # Water Costs.
       ]

def colluirr(input_dir=INPUT_DIR):
    """Return a LUTO-cell indexed, column-landuse-irrigation DataFrame."""
    # Prepare columnwise multi-indexed pandas.DataFrame.
    columns = pd.MultiIndex.from_product( [cols, landuses]
                                        , names=['DATUM', 'LANDUSE_IRR'] )
    index = range(cells.shape[0])
    df = pd.DataFrame(columns=columns, index=index)

    # Perform merges and put resulting arrays in DataFrame.
    colluirr = [ (col, lu, irr) for col in cols
                                for lu in lus
                                for irr in (0, 1) ]
    for col, lu, irr in colluirr:
        if irr == 0: irrstatus = 'dry'
        else: irrstatus = 'irr'
        luirr = lu + '_' + irrstatus
        mask = ( (costs['SPREAD_CLASS'] == lu)
               & (costs['IRRIGATED'] == irr) )
        merged = cells.merge(costs[mask], how='left')
        df[col, luirr] = merged[col].values

    return df

def write_data_feather(fname=os.path.join(INPUT_DIR, 'col-lu-irr.feather')):
    """Write a feather file with the col-lu-irr data. Flattens multi-index."""
    # Produce the DataFrame.
    df = colluir()

    # Flatten the column-wise multi-index.
    df.columns = ['_'.join(col) for col in df.columns.values]

    # Write the feather.
    df.to_feather(fname)

def read_data_feather(fname=os.path.join(INPUT_DIR, 'col-lu-irr.feather')):
    """Read a feather file with the col-lu-irr data. Flattens multi-index."""
    # Read in the feather.
    df = pd.read_feather(fname)

    # Un-flatten the column-wise multi-index.
    cols = df.columns.to_list()
    coltups = []
    for col in cols:    # Split column headers into (DATUM, LANDUSE_IRR) tuples.
        split = col.split(sep='_', maxsplit=1)
        coltups.append((split[0], split[1]))

    # Create the multi index for the columns using the tuples.
    mindex = pd.MultiIndex.from_tuples(coltups, names=['DATUM', 'LANDUSE_IRR'])

    # Re-index the columns with the multi index.
    df.columns = mindex

    # Return the DataFrame.
    return df

def wherelu(df=None):
    """Return boolean matrix indicating allowed land-uses j for each cell r.

    Requires a Pandas DataFrame like from read_data_feather(). If none provided
    it will call `colluir()` which will result in waiting.
    """
    if df is None: df = colluirr()
    return df['Q1'].where(pd.isna, True).where(pd.notna, False)




