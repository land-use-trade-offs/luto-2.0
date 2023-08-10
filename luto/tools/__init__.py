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

from luto.economics.quantity import get_quantity_matrices


def show_map(yr_cal):
    """Show a plot of the lumap of `yr_cal`."""
    plotmap(lumaps[yr_cal], labels = bdata.LU2DESC)
    
    
def get_production(data, yr_cal):
    """Return total production of commodities for a specific year...
       
    Can return base year production (e.g., year = 2010) or can return production for 
    a simulated year if one exists (i.e., year = 2030) check sim.info()).
    
    Includes the impacts of land-use change, productivity increases, and 
    climate change on yield."""
    
    # Calculate year index (i.e., number of years since 2010)
    yr_idx = yr_cal - data.YR_CAL_BASE
    
    # Get the maps in decision-variable format. 0/1 array of shape j, r
    X_mrj = sim.dvars[yr_cal]
    
    # Get the quantity matrices. Quantity array of shape m, r, p
    q_mrp = get_quantity_matrices(data, yr_idx)
    
    # Convert map of land-use in mrj format to mrp format
    X_mrp = np.stack( [ X_mrj[:, :, j] for p in range(data.NPRS) 
                                       for j in range(data.NLUS)
                                       if data.LU2PR[p, j] 
                      ], axis = 2 )

    # Sum quantities in product (PR/p) representation.
    q_p = np.sum( q_mrp * X_mrp, axis = (0, 1), keepdims = False)
    
    # Transform quantities to commodity (CM/c) representation.
    q_c = [ sum( q_p[p] for p in range(data.NPRS) if data.PR2CM[c, p] )
                        for c in range(data.NCMS) ]

    # Return total commodity production as numpy array.
    return np.array(q_c)


def get_production_copy(sim, yr_cal):
    """Return total production of commodities for a specific year...
       
    Can return base year production (e.g., year = 2010) or can return production for 
    a simulated year if one exists (i.e., year = 2030) check sim.info()).
    
    Includes the impacts of land-use change, productivity increases, and 
    climate change on yield."""
    
    # Calculate year index (i.e., number of years since 2010)
    yr_idx = yr_cal - sim.data.YR_CAL_BASE

    # Acquire local names for matrices and shapes.
    lu2pr_pj = sim.data.LU2PR
    pr2cm_cp = sim.data.PR2CM
    nlus = sim.data.NLUS
    nprs = sim.data.NPRS
    ncms = sim.data.NCMS

    # Get the maps in decision-variable format. 0/1 array of shape j, r
    X_dry = sim.dvars[yr_cal][0].T
    X_irr = sim.dvars[yr_cal][1].T

    # Get the quantity matrices. Quantity array of shape m, r, p
    q_mrp = get_quantity_matrices(sim.data, yr_idx)

    X_dry_pr = [ X_dry[j] for p in range(nprs) for j in range(nlus)
                    if lu2pr_pj[p, j] ]
    X_irr_pr = [ X_irr[j] for p in range(nprs) for j in range(nlus)
                    if lu2pr_pj[p, j] ]

    # Quantities in product (PR/p) representation by land management (dry/irr).
    q_dry_p = [ q_mrp[0, :, p] @ X_dry_pr[p] for p in range(nprs) ]
    q_irr_p = [ q_mrp[1, :, p] @ X_irr_pr[p] for p in range(nprs) ]

    # Transform quantities to commodity (CM/c) representation by land management (dry/irr).
    q_dry_c = [ sum(q_dry_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                for c in range(ncms) ]
    q_irr_c = [ sum(q_irr_p[p] for p in range(nprs) if pr2cm_cp[c, p])
                for c in range(ncms) ]

    # Total quantities in commodity (CM/c) representation.
    q_c = [q_dry_c[c] + q_irr_c[c] for c in range(ncms)]

    # Return total commodity production.
    return np.array(q_c)



def lumap2l_mrj(lumap, lmmap):
    """Return land-use maps in decision-variable (X_mrj) format.
       Where 'm' is land mgt, 'r' is cell, and 'j' is land-use."""
    
    # Set up a container array of shape m, r, j. 
    x_mrj = np.zeros((2, lumap.shape[0], 28), dtype = bool)
    
    # Populate the 3D land-use, land mgt mask. 
    for j in range(28):
        jmap = np.where( lumap == j, True, False ).astype(bool)    # One boolean map for each land use.
        x_mrj[0, :, j] = np.where( lmmap == False, jmap, False )   # Keep only dryland version.
        x_mrj[1, :, j] = np.where( lmmap == True, jmap, False )    # Keep only irrigated version.
    
    return x_mrj.astype(bool)



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
