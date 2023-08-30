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
Writes model output and statistics to files.
"""


import os
from datetime import datetime

import numpy as np
import pandas as pd
from luto.tools import get_production
from luto.tools.spatializers import *
from luto.tools.compmap import *
import luto.economics.agricultural.water as ag_water
import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.non_agricultural.ghg as non_ag_ghg


def get_path():
    """Create a folder for storing outputs and return folder name."""
    
    path = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')
    path = 'output/' + path 
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def write_outputs(sim, yr_cal, path):
    """Write outputs for simulation 'sim', calendar year, demands d_c, and path"""
    
    # write_files(sim, path)
    write_production(sim, yr_cal, path)
    write_water(sim, yr_cal, path)
    write_ghg(sim, yr_cal, path)


def write_files(sim, path):
    """Writes numpy arrays and geotiffs to file"""
    
    print('\nWriting numpy arrays and geotiff outputs to', path)
    
    for yr_cal in sim.lumaps:
        
        # Save raw agricultural decision variables (boolean array).
        ag_X_mrj_fname = 'ag_X_mrj_' + str(yr_cal) + '.npy'
        np.save(os.path.join(path, ag_X_mrj_fname), sim.ag_dvars[yr_cal])

        # Save raw non-agricultural decision variables (boolean array).
        non_ag_X_rk_fname = 'non_ag_X_rk_' + str(yr_cal) + '.npy'
        np.save(os.path.join(path, non_ag_X_rk_fname), sim.non_ag_dvars[yr_cal])
        
        # Write out raw numpy arrays for land-use and land management
        lumap_fname = 'lumap_' + str(yr_cal) + '.npy'
        lmmap_fname = 'lmmap_' + str(yr_cal) + '.npy'
        np.save(os.path.join(path, lumap_fname), sim.lumaps[yr_cal])
        np.save(os.path.join(path, lmmap_fname), sim.lmmaps[yr_cal])

        # Recreate full resolution 2D arrays and write out GeoTiffs for land-use and land management
        lumap, lmmap = recreate_2D_maps(sim, yr_cal)
        
        lumap_fname = 'lumap_' + str(yr_cal) + '.tiff'
        lmmap_fname = 'lmmap_' + str(yr_cal) + '.tiff'
        write_gtiff(lumap, os.path.join(path, lumap_fname))
        write_gtiff(lmmap, os.path.join(path, lmmap_fname))


def write_production(sim, yr_cal, path): 
    """Write out land-use and production data"""

    print('\nWriting production outputs to', path)
    
    # Calculate year index
    yr_idx = yr_cal - sim.data.YR_CAL_BASE
    
    # Calculate data for quantity comparison between base year and target year
    prod_base = sim.data.PROD_2010_C                # get_production(sim.data, sim.data.YR_CAL_BASE, sim.data.L_MRJ)  # Get commodity quantities produced in 2010
    prod_targ = get_production(sim.data, yr_cal, sim.ag_dvars[yr_cal], sim.non_ag_dvars[yr_cal])  # Get commodity quantities produced in target year
    demands = sim.data.D_CY[yr_idx]                 # Get commodity demands for target year
    abs_diff = prod_targ - demands                  # Diff between target year production and demands in absolute terms (i.e. tonnes etc)
    prop_diff = ( prod_targ / demands ) * 100       # Target year production as a proportion of demands (%)
    
    # Write to pandas dataframe
    df = pd.DataFrame()
    df['Commodity'] = sim.data.COMMODITIES
    df['Prod_base_year (tonnes, KL)'] = prod_base
    df['Prod_targ_year (tonnes, KL)'] = prod_targ
    df['Demand (tonnes, KL)'] = demands
    df['Abs_diff (tonnes, KL)'] = abs_diff
    df['Prop_diff (%)'] = prop_diff
    df.to_csv(os.path.join(path, 'quantity_comparison.csv'), index = False)

    # LUS = ['Non-agricultural land'] + sim.data.AGRICULTURAL_LANDUSES + sim.data.NON_AGRICULTURAL_LANDUSES
    ctlu, swlu = crossmap( sim.lumaps[sim.data.YR_CAL_BASE]
                         , sim.lumaps[yr_cal]
                         , sim.data.AGRICULTURAL_LANDUSES
                         , sim.data.NON_AGRICULTURAL_LANDUSES )
    ctlm, swlm = crossmap(sim.lmmaps[sim.data.YR_CAL_BASE], sim.lmmaps[yr_cal])

    cthp, swhp = crossmap_irrstat( sim.lumaps[sim.data.YR_CAL_BASE], sim.lmmaps[sim.data.YR_CAL_BASE]
                                 , sim.lumaps[yr_cal], sim.lmmaps[yr_cal]
                                 , sim.data.AGRICULTURAL_LANDUSES
                                 , sim.data.NON_AGRICULTURAL_LANDUSES )

    ctlu.to_csv(os.path.join(path, 'crosstab-lumap.csv'))
    ctlm.to_csv(os.path.join(path, 'crosstab-lmmap.csv'))

    swlu.to_csv(os.path.join(path, 'switches-lumap.csv'))
    swlm.to_csv(os.path.join(path, 'switches-lmmap.csv'))

    cthp.to_csv(os.path.join(path, 'crosstab-irrstat.csv'))
    swhp.to_csv(os.path.join(path, 'switches-irrstat.csv'))


def write_water(sim, yr_cal, path):
    """Calculate water use totals. Takes a simulation object, a numeric
       target calendar year (e.g., 2030), and an output path as input."""

    print('\nWriting water outputs to', path)
    
    # Convert calendar year to year index.
    yr_idx = yr_cal - sim.data.YR_CAL_BASE
    
    # Get water use for year in mrj format
    ag_w_mrj = ag_water.get_wreq_matrices(sim.data, yr_idx)
    non_ag_w_rk = non_ag_water.get_wreq_matrix(sim.data)

    # Prepare a data frame.
    df = pd.DataFrame( columns=[ 'REGION_ID'
                               , 'REGION_NAME'
                               , 'WATER_USE_LIMIT_ML'
                               , 'TOT_WATER_REQ_ML'
                               , 'ABS_DIFF_ML'
                               , 'PROPORTION_%'  ] )

    # Get 2010 water use limits used as constraints in model
    wuse_limits = ag_water.get_wuse_limits(sim.data)

    # Set up data for river regions or drainage divisions
    if settings.WATER_REGION_DEF == 'RR':
        regions = settings.WATER_RIVREGS
        region_id = sim.data.RIVREG_ID
        region_dict = sim.data.RIVREG_DICT
        
    elif settings.WATER_REGION_DEF == 'DD':
        regions = settings.WATER_DRAINDIVS
        region_id = sim.data.DRAINDIV_ID
        region_dict = sim.data.DRAINDIV_DICT
        
    else: print('Incorrect option for WATER_REGION_DEF in settings')
    
    # Loop through specified water regions
    for i, region in enumerate(regions):
        
        # Get indices of cells in region
        ind = np.flatnonzero(region_id == region).astype(np.int32)
        
        # Calculate water requirements by agriculture for year and region.
        wreq_reg = (
              ( ag_w_mrj[:, ind, :] * sim.ag_dvars[yr_cal][:, ind, :] ).sum()  # Agricultural contribution
            + ( non_ag_w_rk[ind, :] * sim.non_ag_dvars[yr_cal][ind, :] ).sum()         # Non-agricultural contribution
        )
        
        # Calculate water use limits
        wul = wuse_limits[i][1]
        
        # Calculate absolute and proportional difference between water use target and actual water use
        abs_diff = wreq_reg - wul
        if wul > 0:
            prop_diff = (wreq_reg / wul) * 100
        else:
            prop_diff = np.nan
        
        # Add to dataframe
        df.loc[i] = ( region
                    , region_dict[region]
                    , wul
                    , wreq_reg 
                    , abs_diff
                    , prop_diff )
    
    # Write to CSV with 2 DP
    df.to_csv( os.path.join(path, 'water_demand_vs_use.csv')
             , index = False
             , float_format = '{:0,.2f}'.format)
    

def write_ghg(sim, yr_cal, path):
    """Calculate total GHG emissions. Takes a simulation object, a target calendar 
       year (e.g., 2030), and an output path as input."""

    print('\nWriting GHG outputs to', path)
        
    # Convert calendar year to year index.
    yr_idx = yr_cal - sim.data.YR_CAL_BASE

    # Get greenhouse gas emissions in mrj format
    ag_g_mrj = ag_ghg.get_ghg_matrices(sim.data, yr_idx, sim.lumaps[yr_cal])
    non_ag_g_rk = non_ag_ghg.get_ghg_matrix(sim.data)

    # Prepare a data frame.
    df = pd.DataFrame( columns=[ 'GHG_EMISSIONS_LIMIT_TCO2e'
                               , 'GHG_EMISSIONS_TCO2e' ] )

    # Get GHG emissions limits used as constraints in model
    ghg_limits = ag_ghg.get_ghg_limits(sim.data)

    # Calculate the GHG emissions from agriculture for year.
    ghg_emissions = (
          ( ag_g_mrj * sim.ag_dvars[yr_cal] ).sum()         # Agricultural contribution
        + ( non_ag_g_rk * sim.non_ag_dvars[yr_cal] ).sum()  # Non-agricultural contribution
    )
    
    # Add to dataframe
    df.loc[0] = ("{:,.0f}".format(ghg_limits), "{:,.0f}".format(ghg_emissions))
    
    # Save to file
    df.to_csv(os.path.join(path, 'GHG_emissions.csv'), index = False)