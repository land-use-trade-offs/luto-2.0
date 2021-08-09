#!/bin/env python3
#
# resfactor.py - tools to run solver on course-grained spatial domains.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-07-30
# Last modified: 2021-08-09
#

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def coursify(array, resfactor, mrj=True):
    """Return course-grained version of array, coursened by `resfactor`."""
    # Every index * `resfactor` is sampled. New array length varies.
    if mrj: return array[:, ::2]
    else: return array[::resfactor]

def uncoursify(array, resfactor, presize=None, mrj=True):
    """Return array inflated by `resfactor`. Inverse of `coursify()`."""
    if presize is None: presize = resfactor * array.shape[0]

    def uncourse(array, resfactor, presize):
        """Return uncoursed array, inflated along axis==0."""

        # Determine the shape of the output array as well as the dtype.
        bshape = list(array.shape)
        bshape[0] = presize
        bshape = tuple(bshape)
        brray = np.ones(bshape, dtype=array.dtype)

        # Output will be length presize. Second for-loop to fill out tail end.
        for i in range(0, len(brray) - resfactor, resfactor):
            for k in range(resfactor):
                brray[i+k] = array[i // resfactor]
        i += resfactor
        for k in range(presize - (array.shape[0]-1)*resfactor):
            brray[i+k] = array[i // resfactor]

        return brray

    if mrj:
        nlms, ncellscourse, nlus = array.shape
        mrjarray = np.zeros((nlms, presize, nlus), dtype=array.dtype)
        for m in range(nlms):
            mrjarray[m] = uncourse(array[m], resfactor, presize)
        return mrjarray
    else:
        return uncourse(array, resfactor, presize)
