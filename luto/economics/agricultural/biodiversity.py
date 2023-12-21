"""
Pure functions to calculate biodiversity by lm, lu.
"""


from typing import Dict
import numpy as np


def get_breq_matrices(data, yr_idx):
    """Return b_mrj water requirement matrices by land management, cell, and land-use type."""
    
    return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))