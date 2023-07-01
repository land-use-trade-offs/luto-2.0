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
Data about transitions costs.
"""


import os.path

import numpy as np
import pandas as pd
import numpy_financial as npf

from luto.economics.cost import get_cost_matrices
from luto.economics.water import get_wreq_matrices

import luto.settings as settings
from luto.tools import lumap2l_mrj


def amortise(cost, rate = settings.DISCOUNT_RATE, horizon = settings.AMORTISATION_PERIOD):
    """Return NPV of future `cost` amortised to annual value at discount `rate` over `horizon` years."""
    return -1 * npf.pmt(rate, horizon, pv = cost, fv = 0, when = 'begin')


def get_exclude_matrices(data, lumap):
    """Return x_mrj exclude matrices.

    An exclude matrix indicates whether switching land-use for a certain cell r 
    with land-use i to all other land-uses j under all land management types 
    (i.e., dryland, irrigated) m is possible. 

    Parameters
    ----------

    data: object/module
        Data object or module with fields like in `luto.data`.
    lumap : numpy.ndarray
        Present land-use map, i.e. highpos (shape=ncells, dtype=int).

    Returns
    -------

    numpy.ndarray
        x_mrj exclude matrix. The m-slices correspond to the
        different land-management versions of the land-use `j` to switch _to_.
        With m==0 conventional dry-land, m==1 conventional irrigated.
    """    
    # Boolean exclusion matrix based on SA2/NLUM agricultural land-use data (in mrj structure).
    # Effectively, this ensures that in any SA2 region the only combinations of land-use and land management
    # that can occur in the future are those that occur in 2010 (i.e., base_year)
    x_mrj = data.EXCLUDE

    # Raw transition-cost matrix is in $/ha and lexicographically ordered by land-use.
    t_ij = data.TMATRIX
    
    # Transition costs from current land-use to all other land-uses j using current land-use map (in $/ha).
    t_rj = t_ij[lumap]

    # To be excluded based on disallowed switches as specified in transition cost matrix i.e., where t_rj is NaN.
    t_rj = np.where(np.isnan(t_rj), 0, 1)

    # Overall exclusion as elementwise, logical `and` of the 0/1 exclude matrices.
    x_mrj = (x_mrj * t_rj).astype(np.int8)
    
    return x_mrj


def get_transition_matrices(data, year, lumap, lmmap):
    """Return t_mrj transition-cost matrices.

    A transition-cost matrix gives the cost of switching a cell r from its 
    current land-use and land management type to every other land-use and land 
    management type. The base costs are taken from the raw transition costs in 
    the `data` module and additional costs are added depending on the land 
    management type (e.g. costs of irrigation infrastructure). 

    Parameters
    ----------

    data: object/module
        Data object or module with fields like in `luto.data`.
    year : int
        Number of years from base year, counting from zero.
    lumap : numpy.ndarray
        Present land-use map, i.e. highpos (shape = ncells, dtype=int).
    lmmap : numpy.ndarray
        Present land-management map (shape = ncells, dtype=int).

    Returns
    -------

    numpy.ndarray
        t_mrj transition-cost matrices. The m-slices correspond to the
        different land management types, r is grid cell, and j is land-use.
    """
    
    # Return l_mrj (Boolean) for current land-use and land management
    l_mrj = lumap2l_mrj(lumap, lmmap)
    
    
    # -------------------------------------------------------------- #
    # Establishment costs (upfront, amortised to annual).            #
    # -------------------------------------------------------------- #
            
    # Raw transition-cost matrix is in $/ha and lexigraphically ordered (shape: land-use x land-use).
    t_ij = data.TMATRIX

    # Non-irrigation related transition costs for cell r to change to land-use j calculated based on lumap (in $/ha).
    t_rj = t_ij[lumap]
        
    # Amortise upfront costs to annualised costs
    t_rj = amortise(t_rj)
    
    
    # -------------------------------------------------------------- #
    # Opportunity costs (annual).                                    #    
    # -------------------------------------------------------------- #
            
    # Get cost matrices and convert from $/cell to $/ha.
    c_mrj = get_cost_matrices(data, year) / data.REAL_AREA[:, np.newaxis]
    
    # Calculate production cost of current land-use and land management for each grid cell r.
    c_r = (c_mrj * l_mrj).sum(axis = 0).sum(axis = 1)
    
    # Opportunity cost calculated as the diff in production cost between current land-use and all other land-uses j
    o_delta_mrj = c_mrj - c_r[:, np.newaxis]
    
    # Cost for switching to Unallocated - modified land is zero
    o_delta_mrj[0, :, data.LU_UNALL_INDICES] = 0
    
    # Cost for switching to the same land-use and irr status is zero.
    o_delta_mrj = np.where(l_mrj, 0, o_delta_mrj)
    
    
    # -------------------------------------------------------------- #
    # Water license costs (upfront, amortised to annual).            #
    # -------------------------------------------------------------- #
    
    # Get water requirements from current agriculture, converting water requirements for LVSTK from ML per head to ML per hectare.
    w_mrj = get_wreq_matrices(data, year)
    
    # Calculate water requirements of current land-use and land management 
    w_r = (w_mrj * l_mrj).sum(axis = 0).sum(axis = 1)
    
    # Net water requirements calculated as the diff in water requirements between current land-use and all other land-uses j.
    w_net_mrj = w_mrj - w_r[:, np.newaxis]
    
    # Water license cost calculated as net water requirements (ML/ha) x licence price ($/ML).
    w_delta_mrj = w_net_mrj * data.WATER_LICENCE_PRICE[:, np.newaxis]
    
    # When land-use changes from dryland to irrigated add $10k per hectare
    w_delta_mrj[1] = np.where(l_mrj[0], w_delta_mrj[1] + 10000, w_delta_mrj[1])

    # When land-use changes from irrigated to dryland add $3k per hectare
    w_delta_mrj[0] = np.where(l_mrj[1], w_delta_mrj[0] + 3000, w_delta_mrj[0])
    
    # Amortise upfront costs to annualised costs
    w_delta_mrj = amortise(w_delta_mrj)
    
    
    # -------------------------------------------------------------- #
    # Total costs.                                                   #
    # -------------------------------------------------------------- #
    
    # Sum annualised costs of land-use and land management transition in $ per ha
    t_mrj = w_delta_mrj + o_delta_mrj + t_rj

    # Ensure cost for switching to the same land-use and land management is zero.
    t_mrj = np.where(l_mrj, 0, t_mrj)
    
    # Convert to $ per cell including resfactor
    t_mrj *= data.REAL_AREA[:, np.newaxis]
    
    return t_mrj
