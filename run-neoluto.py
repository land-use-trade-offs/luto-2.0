#!/bin/env python3
#
# run-neoluto.py - to run neoLUTO.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-04-28
# Last modified: 2021-06-01
#

import os.path

import numpy as np

# Load data module and initialise with ANO Scenario 236.
import luto.data as data

from luto.economics.cost import get_cost_matrix
from luto.economics.quantity import get_quantity_matrix
from luto.economics.transitions import get_transition_matrix

from luto.solvers.solver import solve, inspect

from luto.tools import timethis

# Required input data - for now, just year zero.
#
year = 0

# Present land-use map in highpos format. Based on oldLUTO's SID array.
lumap = np.load(os.path.join(data.INPUT_DIR, 'lumap.npy'))

# Transition cost to commodity j at cell r.
t_rj = get_transition_matrix(year, lumap)

# Cost of producing commodity j at cell r.
c_rj = get_cost_matrix(year)

# Yield of commodity j at cell r.
q_rj = get_quantity_matrix(year)

# Demand and penalty for surplus/deficit for commodity j.
#
# As a first-order, semi-realistic, base-line demand, use current production.
# For penalties the spatially averaged cost of a LU is used.
# d_j = np.zeros(int(data.NLUS / 2)) # Prepare the demands array - one cell per LU.
# p_j = np.zeros(int(data.NLUS / 2)) # Prepare the penalty array - one cell per LU.
d_j = np.zeros(data.NLUS // 2) # Prepare the demands array - one cell per LU.
p_j = np.zeros(data.NLUS // 2) # Prepare the penalty array - one cell per LU.

for j in range(data.NLUS // 2):
    k = 2 * j
    # The demand for LU j is the dot product of the yield vector with a
    # summation vector indicating where LU actually occurs as per SID array.
    #        [ yield of j for all r ] . [ all r where j occurs ]
    d_j[j] = ( q_rj.T[k]   @ np.where(lumap == k, 1, 0)
             + q_rj.T[k+1] @ np.where(lumap == k+1, 1, 0) )
    p_j[j] = max(c_rj.T[k].max(), c_rj.T[k+1].max())
# for j in range(data.NLUS):
    # d_j[j] = q_rj.T[j]   @ np.where(lumap == j, 1, 0)
    # p_j[j] = c_rj.T[j].mean()

# Possible land uses j at cell r.
x_rj = data.x_rj
# x_rj = np.ones((data.NCELLS, data.NLUS), dtype=np.int8) # data.x_rj
# x_rj = np.random.randint(2, size=(data.NCELLS, data.NLUS), dtype=np.int8)

def run():
    highpos = timethis( solve
                      , t_rj
                      , c_rj
                      , q_rj
                      , d_j
                      , p_j
                      , x_rj )
    df, precost, postcost = inspect(lumap, highpos, d_j, q_rj, c_rj)

    print(df)
    print("LU Cost Prior:", precost)
    print("LU Cost Posterior:", postcost)

    return df, precost, postcost, highpos

def run_normalised():
    highpos = timethis( solve
                      , t_rj
                      , c_rj
                      , q_rj
                      , d_j
                      , 10**16 # p_j
                      , x_rj
                      , pen_norm = True )
    df, precost, postcost = inspect(lumap, highpos, d_j, q_rj, c_rj)

    print(df)
    print("LU Cost Prior:", precost)
    print("LU Cost Posterior:", postcost)

    return df, precost, postcost, highpos

