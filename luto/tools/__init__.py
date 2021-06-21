#!/bin/env python3
#
# __init__.py - pure helper functions and other tools.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-05-21
# Last modified: 2021-06-21
#

import time
import os.path

import pandas as pd
import numpy as np

from luto.tools.gtiffutils import highpos2gtiff

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
    ct.to_csv(os.path.join(dirname, prename + str(2) + postname + ".csv"))

