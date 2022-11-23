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
Sub module to load and prepare agro-economic (spatial) data.
"""


import os.path

import pandas as pd
import numpy as np

def build_agec_crops(concordance, data, columns=None, lus=None, lms=None):
    """Return LUTO-cell wise, multi-indexed column-landuse-landman DataFrame."""

    # If no list of columns is provided, infer it from provided data.
    if columns is None:
        columns = data.columns.to_list()

    # If no list of land uses provided, infer it from the SPREAD_CLASS column.
    if lus is None:
        lus = sorted(data['LU_DESC'].unique().tolist())

    # If no list of land managements provided, use irrigation status only.
    if lms is None:
        lms = ['dry', 'irr']

    # Produce a multi-indexed version data and merge to yield cell-based table.
    crops_ptable = data.pivot_table( index='SA2_ID'
                                   , columns=['Irrigation', 'LU_DESC'] )
    agec_crops = concordance.merge( crops_ptable
                                  , left_on='SA2_ID'
                                  , right_on=crops_ptable.index
                                  , how='left' )

    # Some columns have to go.
    agec_crops = agec_crops.drop(['CELL_ID', 'SA2_ID'], axis=1)

    # The merge flattens the multi-index to tuples, so unflatten here.
    ts = [(t[0], lms[t[1]], t[2]) for t in agec_crops.columns]
    agec_crops.columns = pd.MultiIndex.from_tuples(ts)

    return agec_crops

def build_agec_lvstk(data):
    """Return LUTO-cell wise, multi-indexed column-landuse DataFrame."""

    # Get only the columns that vary with the land-use.
    animals = [ c for c in data.columns if 'BEEF' in c
                                        or 'SHEEP' in c
                                        or 'DAIRY' in c ]
    agec_lvstk = data[animals]

    # Prepare columns for multi-indexing, i.e. turn into a list of tuples.
    cols = agec_lvstk.columns.to_list()
    cols = [tuple(c.split(sep='_')) for c in cols]
    cols = [(c[0]+'_'+c[1], c[2]) if c[0]=='WR' else c for c in cols]
    cols = [ (c[0] + '_' + c[1] + '_' + c[3] + '_' +c[4], c[2])
             if c[0]=='FEED' else c for c in cols ]

    # Make and set the multi-index.
    mcolindex = pd.MultiIndex.from_tuples(cols)
    agec_lvstk.columns = mcolindex

    return agec_lvstk


def concord(concordance, data, columns=None, lus=None, lms=None):
    """Return LUTO-cell wise, multi-indexed column-landuse-landman DataFrame."""

    # If no list of columns is provided, infer it from provided data.
    if columns is None:
        columns = data.columns.to_list()

    # If no list of land uses provided, infer it from the SPREAD_CLASS column.
    if lus is None:
        lus = sorted(data['LU_DESC'].unique().tolist())

    # If no list of land managements provided, use irrigation status only.
    if lms is None:
        lms = ['dry', 'irr']

    # Prepare column-wise multi-indexed pandas.DataFrame.
    mcolindex = pd.MultiIndex.from_product( [columns, lms, lus]
                                          , names = [ 'DATUM'
                                                    , 'LANDMANS'
                                                    , 'LANDUSE' ] )
    index = range(concordance.shape[0])
    df = pd.DataFrame(columns=mcolindex, index=index)

    # Perform merges and put resulting arrays in DataFrame.
    colluirr = [ (col, lu, irr) for col in columns
                                for lu in lus
                                for irr in (0, 1) ]
    for col, lu, irr in colluirr:
        if irr == 0: irrstatus = 'dry'
        else: irrstatus = 'irr'
        mask = ( (data['SPREAD_CLASS'] == lu)
               & (data['IRRIGATED'] == irr) )
        merged = concordance.merge(data[mask], how='left')
        df[col, irrstatus, lu] = merged[col].values

    return df

def exclude(df):
    """Return exclude matrix inferred from passed AGEC mjr-dataframe."""
    # The set of all land-management types.
    lms = {t[1] for t in df.columns}

    # Any economic variable will do. Here, choose `AC`.
    dfq = df['AC']

    # Build a tuple of slices, each slice a boolean exclude matrix for a lm.
    slices = tuple( dfq[lm].where(pd.isna, True).where(pd.notna, False)
                    for lm in lms )

    return np.stack(slices)
