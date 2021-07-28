#!/bin/env python3
#
# run-neoluto.py - to run neoLUTO.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-07-28
# Last modified: 2021-07-28
#

import os.path

import numpy as np

# Load data module and initialise with ANO Scenario 236.
import luto.data as data

from luto.economics.cost import get_cost_matrix
from luto.economics.quantity import get_quantity_matrix
from luto.economics.transitions import get_transition_matrix

from luto.solvers.solver import coursify, uncoursify
from luto.solvers.solver import solve

from luto.tools import timethis, inspect, ctabnpystocsv
from luto.tools.highposgtiff import write_highpos_gtiff

# Required input data.
year = 0
lumap = np.load(os.path.join(data.INPUT_DIR, 'lumap.npy'))
lmmap = np.load(os.path.join(data.INPUT_DIR, 'lmmap.npy'))
t_mrj = get_transition_matrix_old(year, lumap)
c_mrj = get_cost_matrix_old(year)
q_mrj = get_quantity_matrix_old(year)

nlms = t_mrj.shape[0]
ncells = t_mrj.shape[1]
nlus = t_rmj.shape[2]

d_j = np.zeros(data.NLUS) # Prepare the demands array - one cell per LU.

for j in range(nlus):
    # The demand for LU j is the dot product of the yield vector with a
    # summation vector indicating where LU actually occurs as per SID array.
    d_j[j] = ( q_mrj[0].T[j] @ np.where( (lumap==j) & (lmmap==0), 1, 0)
             + q_mrj[1].T[j] @ np.where( (lumap==j) & (lmmap==1), 1, 0) )

# Default penalty level.
p = 1

# Possible land uses j at cell r.
x_rj = data.x_rj

params_data = { "t_mrj" : t_rj
              , "c_mrj" : c_rj
              , "q_mrj" : q_rj
              , "d_j"  : d_j
              , "p"    : p
              , "x_mrj" : x_rj
              }

def randomparams(ncells, nlus, p=1):
    # `nlus` needs to be an even number because land-uses come as dry/irr pairs.
    if nlus % 2 == 1:
        raise ValueError("nlus needs to be an even integer")
    else:
        # Bogus lumap.
        lumap = np.random.randint(nlus, size=ncells)

        # Transition cost matrix.
        t_ij = 10 * np.random.random((nlus, nlus))
        for i in range(nlus): t_ij[i, i] = 0

        t_rj = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))

        # Production cost matrix.
        c_rj = 10 * np.random.random((ncells, nlus))

        # Yield matrix.
        q_rj = 10 * np.random.random((ncells, nlus))

        # Demands.
        d_j = 10 * np.random.random(nlus)

        # Exclude matrix.
        x_rj = np.ones_like(t_rj)

        return { "t_rj" : t_rj
               , "c_rj" : c_rj
               , "q_rj" : q_rj
               , "d_j"  : d_j
               , "p"    : p
               , "x_rj" : x_rj
               }

def run_params(params, resfactor=1):

    if resfactor == 1:
        highpos = timethis( solve
                        , params["t_rj"]
                        , params["c_rj"]
                        , params["q_rj"]
                        , params["d_j"]
                        , params["p"]
                        , params["x_rj"] )
    else:
        ncells, nlus = params["t_rj"].shape

        print("Applying resfactor =", str(resfactor))

        print("\t" + "Course-graining input arrays...")
        t_rj = coursify(params["t_rj"], resfactor)
        c_rj = coursify(params["c_rj"], resfactor)
        q_rj = coursify(params["q_rj"], resfactor)
        x_rj = coursify(params["x_rj"], resfactor)

        print("\t" + "Adjusting demands...")
        d_j = params["d_j"] / resfactor

        print("\t" + "Adjusting penalty level...")
        p = params["p"] * resfactor

        highpos = timethis( solve
                          , t_rj
                          , c_rj
                          , q_rj
                          , d_j
                          , p
                          , x_rj )

        print("Inflating highpos output array to original original extent...")
        highpos = uncoursify(highpos, resfactor, presize=ncells)

    return highpos

def run(p=1, write=True):
    highpos = timethis( solve
                      , t_rj
                      , c_rj
                      , q_rj
                      , d_j
                      , p
                      , x_rj )

    df = inspect(lumap, highpos, d_j, q_rj, c_rj, data.LANDUSES)

    print(df)

    if write:
        np.save('highpos.npy', highpos)
        df.to_csv('diff.csv')
        write_highpos_gtiff('highpos.npy', 'highpos')
        ctabnpystocsv( os.path.join(data.INPUT_DIR, 'lumap.npy')
                     , 'highpos.npy'
                     , data.LANDUSES )


    return df, highpos

def run_decigoogol_penalty():
    highpos = timethis( solve
                      , t_rj
                      , c_rj
                      , q_rj
                      , d_j
                      , 10**10
                      , x_rj )

    df = inspect(lumap, highpos, d_j, q_rj, c_rj, data.LANDUSES)

    print(df)

    return df, highpos

def run_random(ncells, nlus, p=1, resfactor=1):
    """Run model with random data, penalty-level `p` and shape ncells, nlus."""

    # `nlus` needs to be an even number because land-uses come as dry/irr pairs.
    if nlus % 2 == 1:
        raise ValueError("nlus needs to be an even integer")
    else:
        # Bogus lumap.
        lumap = np.random.randint(nlus, size=ncells)

        # Transition cost matrix.
        t_ij = 10 * np.random.random((nlus, nlus))
        for i in range(nlus): t_ij[i, i] = 0

        t_rj = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))

        # Production cost matrix.
        c_rj = 10 * np.random.random((ncells, nlus))

        # Yield matrix.
        q_rj = 10 * np.random.random((ncells, nlus))

        # Demands.
        d_j = 10 * np.random.random(nlus)

        # Exclude matrix.
        x_rj = np.ones_like(t_rj) # np.random.randint(2, size=(ncells, nlus), dtype=np.int8)

        if resfactor == 1:
            highpos = timethis( solve
                            , t_rj
                            , c_rj
                            , q_rj
                            , d_j
                            , p
                            , x_rj )
        else:
            print("Applying resfactor =", str(resfactor))

            print("\t" + "Course-graining input arrays...")
            t_rj = coursify(t_rj, resfactor)
            c_rj = coursify(c_rj, resfactor)
            q_rj = coursify(q_rj, resfactor)
            x_rj = coursify(x_rj, resfactor)

            print("\t" + "Adjusting demands...")
            d_j /= resfactor

            print("\t" + "Adjusting penalty level...")
            p *= resfactor

            highpos = timethis( solve
                            , t_rj
                            , c_rj
                            , q_rj
                            , d_j
                            , p
                            , x_rj )
            print("Inflating highpos output array to original original extent...")
            highpos = uncoursify(highpos, resfactor, presize=ncells)

    return highpos

def runstack_random(ncells, nlus, p=1):
    """Run stacked model w/ random data, penalty-level `p` and shape ncells, nlus."""

    # Bogus lumap.
    lumap = np.random.randint(nlus, size=ncells)

    # Transition cost matrix.
    t_ij = 10 * np.random.random((nlus, nlus))
    for i in range(nlus): t_ij[i, i] = 0

    t_rj = np.stack(tuple(t_ij[lumap[r]] for r in range(ncells)))
    tm_rj = 0.1 * t_rj

    # Production cost matrix.
    c_rj = 10 * np.random.random((ncells, nlus))
    cm_rj = 0.1 * c_rj

    # Yield matrix.
    q_rj = 10 * np.random.random((ncells, nlus))
    qm_rj = 5 * q_rj

    # Demands.
    d_j = 20 * np.random.random(nlus)

    # Exclude matrix.
    x_rj = np.ones((ncells, nlus)) # np.random.randint(2, size=(ncells, nlus), dtype=np.int8)

    highpos, mngmnt = timethis( stacksolve
                              , t_rj
                              , tm_rj
                              , c_rj
                              , cm_rj
                              , q_rj
                              , qm_rj
                              , d_j
                              , p
                              , x_rj )

    return highpos, mngmnt
