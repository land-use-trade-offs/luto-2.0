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
Pure helper functions and other tools.
"""


import time
import os.path

import pandas as pd
import numpy as np


def lumap2x_mrj_old(lumap, lmmap):
    """Return land-use maps in decision-variable (X_mrj) format."""
    
    drystack = []
    irrstack = []
    for j in range(28):
        jmap = np.where( lumap == j, 1, 0 )                  # One boolean map for each land use.
        drystack.append( np.where( lmmap == 0, jmap, 0 ) )   # Keep only dryland version.
        irrstack.append( np.where( lmmap == 1, jmap, 0 ) )   # Keep only irrigated version.
    
    x_dry_rj = np.stack( drystack, axis = 1 )
    x_irr_rj = np.stack( irrstack, axis = 1 )
    
    return np.stack((x_dry_rj, x_irr_rj))         # In x_mrj format.


def lumap2x_mrj(lumap, lmmap):
    """Return land-use maps in decision-variable (X_mrj) format.
       Where 'm' is land mgt, 'r' is cell, and 'j' is land-use."""
    
    # Set up a container array of shape m, r, j. 
    x_mrj = np.zeros((2, lumap.shape[0], 28), dtype = bool)
    
    # Populate the 3D land-use, land mgt mask. 
    for j in range(28):
        jmap = np.where( lumap == j, True, False ).astype(bool)    # One boolean map for each land use.
        x_mrj[0, :, j] = np.where( lmmap == False, jmap, False )   # Keep only dryland version.
        x_mrj[1, :, j] = np.where( lmmap == True, jmap, False )    # Keep only irrigated version.
    
    return x_mrj



def timethis(function, *args, **kwargs):
    """Generic wrapper to time functions."""

    # Start the wall clock.
    start = time.time()
    start_time = time.localtime()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", start_time)

    # Call the function.
    return_value = function(*args, **kwargs)

    # Stop the wall clock.
    stop = time.time()
    stop_time = time.localtime()
    stop_time_str = time.strftime("%Y-%m-%d %H:%M:%S", stop_time)

    print()
    print("Start time: %s" % start_time_str)
    print("Stop time: %s" % stop_time_str)
    print("Elapsed time: %d seconds." % (stop - start))

    return return_value

def mergeorderly(dict1, dict2):
    """Return merged dictionary with keys in alphabetic order."""
    list1 = list(dict1.keys())
    list2 = list(dict2.keys())
    lst = sorted(list1 + list2)
    merged = {}
    for key in lst:
        if key in list1:
            merged[key] = dict1[key]
        else:
            merged[key] = dict2[key]
    return merged

def crosstabulate(before, after, labels):
    """Return a cross-tabulation matrix as a Pandas DataFrame."""

    # The base DataFrame
    ct = pd.crosstab(before, after, margins=True)

    # Replace the numbered row labels with corresponding entries in `labels`.
    rows = [labels[int(i)] for i in ct.index if i != 'All']
    # The sum row was excluded, now add it back.
    rows.append('All')

    # Replace the numbered column labels with corresponding entries in `labels`.
    cols = [labels[int(i)] for i in ct.columns if i != 'All']
    # The sum column was excluded, now add it back.
    cols.append('All')

    # Replace the row and column labels.
    ct.index = rows
    ct.columns = cols

    return ct

def ctabnpystocsv(before, after, labels):
    """Write cross-tabulation of `before` and `after` from .npys to .csv."""

    # Load the .npy arrays.
    pre = np.load(before)
    post = np.load(after)

    # Get the file names as strings without extension.
    prename = os.path.splitext(os.path.basename(before))[0]
    postname = os.path.splitext(os.path.basename(after))[0]
    dirname = os.path.dirname(before)

    # Produce the cross tabulation and write the file to dir of first file.
    ct = crosstabulate(pre, post, labels)
    ct.to_csv(os.path.join(prename + str(2) + postname + ".csv"))

def inspect(lumap, highpos, d_j, q_rj, c_rj, landuses):

    # Prepare the pandas.DataFrame.
    columns = [ "Pre Cells [#]"
              , "Pre Costs [AUD]"
              , "Pre Deviation [units]"
              , "Pre Deviation [%]"
              , "Post Cells [#]"
              , "Post Costs [AUD]"
              , "Post Deviation [units]"
              , "Post Deviation [%]"
              , "Delta Total Cells [#]"
              , "Delta Total Cells [%]"
              , "Delta Costs [AUD]"
              , "Delta Costs [%]"
              , "Delta Moved Cells [#]"
              , "Delta Deviation [units]"
              , "Delta Deviation [%]" ]
    index = landuses
    df = pd.DataFrame(index=index, columns=columns)

    precost = 0
    postcost = 0

    # Reduced land-use list with half of the land-uses.
    lus = [lu for lu in landuses if '_dry' in lu]

    for j, lu in enumerate(lus):
        k = 2*j
        prewhere_dry = np.where(lumap == k, 1, 0)
        prewhere_irr = np.where(lumap == k+1, 1, 0)
        precount_dry = prewhere_dry.sum()
        precount_irr = prewhere_irr.sum()
        prequantity_dry = prewhere_dry @ q_rj.T[k]
        prequantity_irr = prewhere_irr @ q_rj.T[k+1]

        precost_dry = prewhere_dry @ c_rj.T[k]
        precost_irr = prewhere_irr @ c_rj.T[k+1]

        postwhere_dry = np.where(highpos == k, 1, 0)
        postwhere_irr = np.where(highpos == k+1, 1, 0)
        postcount_dry = postwhere_dry.sum()
        postcount_irr = postwhere_irr.sum()
        postquantity_dry = postwhere_dry @ q_rj.T[k]
        postquantity_irr = postwhere_irr @ q_rj.T[k+1]

        postcost_dry = postwhere_dry @ c_rj.T[k]
        postcost_irr = postwhere_irr @ c_rj.T[k+1]

        predeviation = prequantity_dry + prequantity_irr - d_j[j]
        predevfrac = predeviation / d_j[j]

        postdeviation = postquantity_dry + postquantity_irr - d_j[j]
        postdevfrac = postdeviation / d_j[j]

        df.loc[lu] = [ precount_dry
                     , precost_dry
                     , predeviation
                     , 100 * predevfrac
                     , postcount_dry
                     , postcost_dry
                     , postdeviation
                     , 100 * postdevfrac
                     , postcount_dry - precount_dry
                     , 100 * (postcount_dry - precount_dry) / precount_dry
                     , postcost_dry - precost_dry
                     , 100 * (postcost_dry - precost_dry) / precost_dry
                     , np.sum(postwhere_dry - (prewhere_dry*postwhere_dry))
                     , np.abs(postdeviation) - np.abs(predeviation)
                     , 100 * ( np.abs(postdeviation) - np.abs(predeviation)
                             / predeviation ) ]

        lu_irr = lu.replace('_dry', '_irr')
        df.loc[lu_irr] = [ precount_irr
                         , precost_irr
                         , predeviation
                         , 100 * predevfrac
                         , postcount_irr
                         , postcost_irr
                         , postdeviation
                         , 100 * postdevfrac
                         , postcount_irr - precount_irr
                         , 100 * (postcount_irr - precount_irr) / precount_irr
                         , postcost_irr - precost_irr
                         , 100 * (postcost_irr - precost_irr) / precost_irr
                         , np.sum(postwhere_irr - (prewhere_irr*postwhere_irr))
                         , np.abs(postdeviation) - np.abs(predeviation)
                         , 100 * ( np.abs(postdeviation) - np.abs(predeviation)
                                 / predeviation ) ]

        precost += prewhere_dry @ c_rj.T[k] + prewhere_irr @ c_rj.T[k+1]
        postcost += postwhere_dry @ c_rj.T[k] + postwhere_irr @ c_rj.T[k+1]

    df.loc['total'] = [df[col].sum() for col in df.columns]

    return df
