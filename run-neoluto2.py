#!/bin/env python3
#
# run-neoluto.py - to run neoLUTO.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-07-28
# Last modified: 2021-07-29
#

import os.path

import numpy as np

# Load data module and initialise with ANO Scenario 236.
import luto.data as data

from luto.economics.cost import get_cost_matrices
from luto.economics.quantity import get_quantity_matrices
from luto.economics.transitions import get_transition_matrices

from luto.solvers.solver import coursify, uncoursify
from luto.solvers.solver import solve

from luto.tools import timethis, inspect, ctabnpystocsv
from luto.tools.highposgtiff import write_highpos_gtiff

# Required input data.
year = 0
lumap = np.load(os.path.join(data.INPUT_DIR, 'lumap.npy'))
lmmap = np.load(os.path.join(data.INPUT_DIR, 'lmmap.npy'))
t_mrj = get_transition_matrices(year, lumap, lmmap)
c_mrj = get_cost_matrices(year)
q_mrj = get_quantity_matrices(year)

nlms, ncells, nlus = t_mrj.shape

d_j = np.zeros(nlus) # Prepare the demands array - one cell per LU.
for j in range(nlus):
    # The demand for LU j is the dot product of the yield vector with a
    # summation vector indicating where LU actually occurs as per SID array.
    d_j[j] = ( q_mrj[0].T[j] @ np.where((lumap==j) & (lmmap==0), 1, 0)
             + q_mrj[1].T[j] @ np.where((lumap==j) & (lmmap==1), 1, 0) )

# Default penalty level.
p = 1

# Possible land uses j at cell r.
x_mrj = data.x_mrj

params_data = { "t_mrj" : t_mrj
              , "c_mrj" : c_mrj
              , "q_mrj" : q_mrj
              , "d_j"   : d_j
              , "p"     : p
              , "x_mrj" : x_mrj
              }

def randomparams(nlms, ncells, nlus, p=1):
    # Bogus lumap.
    lumap = np.random.randint(nlus, size=ncells)

    # Transition cost matrix.
    t_ij = 10 * np.random.random((nlus, nlus))
    for i in range(nlus): t_ij[i, i] = 0
    t_rj = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))
    t_mrj = np.stack((t_rj, 1.5*t_rj))

    # Production cost matrix.
    c_mrj = 10 * np.random.random((nlms, ncells, nlus))

    # Yield matrix.
    q_mrj = 10 * np.random.random((nlms, ncells, nlus))

    # Demands.
    d_j = 10 * np.random.random(nlus)

    # Exclude matrix.
    x_mrj = np.ones_like(t_mrj)

    return { "t_mrj" : t_mrj
           , "c_mrj" : c_mrj
           , "q_mrj" : q_mrj
           , "d_j"  : d_j
           , "p"     : p
           , "x_mrj" : x_mrj
           }

def run_params(params, resfactor=1):

    if resfactor == 1:
        lumap, lmmap = timethis( solve
                               , params["t_mrj"]
                               , params["c_mrj"]
                               , params["q_mrj"]
                               , params["d_j"]
                               , params["p"]
                               , params["x_mrj"] )
    else:
        nlms, ncells, nlus = params["t_mrj"].shape

        print("Applying resfactor =", str(resfactor))

        print("\t" + "Course-graining input arrays...")
        t_mrj = coursify(params["t_mrj"], resfactor)
        c_mrj = coursify(params["c_mrj"], resfactor)
        q_mrj = coursify(params["q_mrj"], resfactor)
        x_mrj = coursify(params["x_mrj"], resfactor)

        print("\t" + "Adjusting demands...")
        d_j = params["d_j"] / resfactor

        print("\t" + "Adjusting penalty level...")
        p = params["p"] * resfactor

        highpos = timethis( solve
                          , t_mrj
                          , c_mrj
                          , q_mrj
                          , d_j
                          , p
                          , x_mrj )

        print("Inflating highpos output array to original original extent...")
        highpos = uncoursify(highpos, resfactor, presize=ncells)

    return lumap, lmmap
