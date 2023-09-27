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


import os, math
from datetime import datetime
import numpy as np
import pandas as pd

import luto.settings as settings
from luto import tools
from luto.tools.spatializers import *
from luto.tools.compmap import *
import luto.economics.agricultural.water as ag_water
import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.non_agricultural.ghg as non_ag_ghg
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


def get_path():
    """Create a folder for storing outputs and return folder name."""
    
    # Get date and time
    path = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')
    
    # Add some shorthand details about the model run
    post = '_' + settings.DEMAND_CONSTRAINT_TYPE + '_' + settings.OBJECTIVE + '_RF' + str(settings.RESFACTOR) + '_P1e' + str(int(math.log10(settings.PENALTY))) + '_W' + settings.WATER_USE_LIMITS + '_G' + settings.GHG_EMISSIONS_LIMITS
    
    # Create path name
    path = 'output/' + path + post
    
    # Create folder 
    if not os.path.exists(path):
        os.mkdir(path)
    
    # Return path name
    return path



def write_outputs(sim, yr_cal, path):
    """Write outputs for simulation 'sim', calendar year, demands d_c, and path"""
    
    write_files(sim, path)
    write_settings(path)
    write_production(sim, yr_cal, path)
    write_water(sim, yr_cal, path)
    write_ghg(sim, yr_cal, path)
    write_ghg_seperate(sim, yr_cal, path)


def write_settings(path):
    """Write model run settings"""
    
    with open(os.path.join(path, 'model_run_settings.txt'), 'w') as f:
        f.write('LUTO version %s\n' % settings.VERSION)
        f.write('SSP %s, RCP %s\n' %(settings.SSP, settings.RCP))
        f.write('DISCOUNT_RATE: %s\n' % settings.DISCOUNT_RATE)
        f.write('AMORTISATION_PERIOD: %s\n' % settings.AMORTISATION_PERIOD)
        f.write('RESFACTOR: %s\n' % settings.RESFACTOR)
        f.write('MODE: %s\n' % settings.MODE)
        f.write('OBJECTIVE: %s\n' % settings.OBJECTIVE)
        f.write('PENALTY: %s\n' % settings.PENALTY)
        f.write('OPTIMALITY_TOLERANCE: %s\n' % settings.OPTIMALITY_TOLERANCE)
        f.write('THREADS: %s\n' % settings.THREADS)
        f.write('ENV_PLANTING_COST_PER_HA_PER_YEAR: %s\n' % settings.ENV_PLANTING_COST_PER_HA_PER_YEAR)
        f.write('CARBON_PRICE_PER_TONNE: %s\n' % settings.CARBON_PRICE_PER_TONNE)
        f.write('WATER_USE_LIMITS: %s\n' % settings.WATER_USE_LIMITS)
        f.write('GHG_EMISSIONS_LIMITS: %s\n' % settings.GHG_EMISSIONS_LIMITS)
        f.write('GHG_REDUCTION_PERCENTAGE: %s\n' % settings.GHG_REDUCTION_PERCENTAGE)
        f.write('WATER_REGION_DEF: %s\n' % settings.WATER_REGION_DEF)


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

        # Save raw agricultural management decision variables
        for am in AG_MANAGEMENTS_TO_LAND_USES:
            snake_case_am = tools.am_name_snake_case(am)
            am_X_mrj_fname = 'ag_man_X_mrj_' + snake_case_am + "_" + str(yr_cal) + ".npy"
            np.save(os.path.join(path, am_X_mrj_fname), sim.ag_man_dvars[yr_cal])
        
        # Write out raw numpy arrays for land-use and land management
        lumap_fname = 'lumap_' + str(yr_cal) + '.npy'
        lmmap_fname = 'lmmap_' + str(yr_cal) + '.npy'
        np.save(os.path.join(path, lumap_fname), sim.lumaps[yr_cal])
        np.save(os.path.join(path, lmmap_fname), sim.lmmaps[yr_cal])

        # Recreate full resolution 2D arrays and write out GeoTiffs for land-use and land management
        lumap, lmmap, ammap = recreate_2D_maps(sim, yr_cal)
        
        lumap_fname = 'lumap_' + str(yr_cal) + '.tiff'
        lmmap_fname = 'lmmap_' + str(yr_cal) + '.tiff'
        ammap_fname = 'ammap_' + str(yr_cal) + '.tiff'
        write_gtiff(lumap, os.path.join(path, lumap_fname))
        write_gtiff(lmmap, os.path.join(path, lmmap_fname))
        write_gtiff(ammap, os.path.join(path, ammap_fname))


def write_production(sim, yr_cal, path): 
    """Write out land-use and production data"""

    print('\nWriting production outputs to', path)
    
    # Calculate year index
    yr_idx = yr_cal - sim.data.YR_CAL_BASE
    
    # Calculate data for quantity comparison between base year and target year
    prod_base = sim.data.PROD_2010_C                # tools.get_production(sim.data, sim.data.YR_CAL_BASE, sim.data.L_MRJ)  # Get commodity quantities produced in 2010
    prod_targ = tools.get_production( sim.data
                                    , yr_cal
                                    , sim.ag_dvars[yr_cal]
                                    , sim.non_ag_dvars[yr_cal] 
                                    , sim.ag_man_dvars[yr_cal] )  # Get commodity quantities produced in target year
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
    ctlu, swlu = lumap_crossmap( sim.lumaps[sim.data.YR_CAL_BASE]
                               , sim.lumaps[yr_cal]
                               , sim.data.AGRICULTURAL_LANDUSES
                               , sim.data.NON_AGRICULTURAL_LANDUSES )
    ctlm, swlm = lmmap_crossmap(sim.lmmaps[sim.data.YR_CAL_BASE], sim.lmmaps[yr_cal])

    ctam, swam = ammap_crossmap(sim.ammaps[sim.data.YR_CAL_BASE], sim.ammaps[yr_cal])

    cthp, swhp = crossmap_irrstat( sim.lumaps[sim.data.YR_CAL_BASE], sim.lmmaps[sim.data.YR_CAL_BASE]
                                 , sim.lumaps[yr_cal], sim.lmmaps[yr_cal]
                                 , sim.data.AGRICULTURAL_LANDUSES
                                 , sim.data.NON_AGRICULTURAL_LANDUSES )
    
    ctas, swas = crossmap_amstat( sim.lumaps[sim.data.YR_CAL_BASE], sim.ammaps[sim.data.YR_CAL_BASE]
                                , sim.lumaps[yr_cal], sim.ammaps[yr_cal]
                                , sim.data.AGRICULTURAL_LANDUSES
                                , sim.data.NON_AGRICULTURAL_LANDUSES )
        
    ctlu.to_csv(os.path.join(path, 'crosstab-lumap.csv'))
    ctlm.to_csv(os.path.join(path, 'crosstab-lmmap.csv'))
    ctam.to_csv(os.path.join(path, 'crosstab-ammap.csv'))

    swlu.to_csv(os.path.join(path, 'switches-lumap.csv'))
    swlm.to_csv(os.path.join(path, 'switches-lmmap.csv'))
    swam.to_csv(os.path.join(path, 'switches-ammap.csv'))

    cthp.to_csv(os.path.join(path, 'crosstab-irrstat.csv'))
    swhp.to_csv(os.path.join(path, 'switches-irrstat.csv'))

    ctas.to_csv(os.path.join(path, 'crosstab-amstat.csv'))
    swas.to_csv(os.path.join(path, 'switches-amstat.csv'))


def write_water(sim, yr_cal, path):
    """Calculate water use totals. Takes a simulation object, a numeric
       target calendar year (e.g., 2030), and an output path as input."""

    print('\nWriting water outputs to', path)
    
    # Convert calendar year to year index.
    yr_idx = yr_cal - sim.data.YR_CAL_BASE
    
    # Get water use for year in mrj format
    ag_w_mrj = ag_water.get_wreq_matrices(sim.data, yr_idx)
    non_ag_w_rk = non_ag_water.get_wreq_matrix(sim.data)
    ag_man_w_mrj = ag_water.get_agricultural_management_water_matrices(sim.data, ag_w_mrj, yr_idx)

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
              ( ag_w_mrj[:, ind, :] * sim.ag_dvars[yr_cal][:, ind, :] ).sum()   # Agricultural contribution
            + ( non_ag_w_rk[ind, :] * sim.non_ag_dvars[yr_cal][ind, :] ).sum()  # Non-agricultural contribution
        )

        for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():                  # Agricultural managements contribution
            am_j = np.array([sim.data.DESC2AGLU[lu] for lu in am_lus])
            wreq_reg += ( ag_man_w_mrj[am][:, ind, :]
                        * sim.ag_man_dvars[yr_cal][am][:, ind[:,np.newaxis], am_j] ).sum()

        
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
    ag_g_mrj = ag_ghg.get_ghg_matrices(sim.data, yr_idx)
    non_ag_g_rk = non_ag_ghg.get_ghg_matrix(sim.data)
    ag_man_g_mrj = ag_ghg.get_agricultural_management_ghg_matrices(sim.data, ag_g_mrj, yr_idx)

    # Prepare a data frame.
    df = pd.DataFrame( columns=[ 'GHG_EMISSIONS_LIMIT_TCO2e'
                               , 'GHG_EMISSIONS_TCO2e' ] )

    # Get GHG emissions limits used as constraints in model
    ghg_limits = ag_ghg.get_ghg_limits(sim.data)

    # Calculate the GHG emissions from agriculture for year.
    ghg_t_2010 = ag_ghg.get_ghg_transition_penalties(sim.data, sim.lumaps[2010])
    ghg_emissions = (
          ( (ag_g_mrj + ghg_t_2010) * sim.ag_dvars[yr_cal] ).sum()  # Agricultural contribution
        + ( non_ag_g_rk * sim.non_ag_dvars[yr_cal] ).sum()          # Non-agricultural contribution
    )

    for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():          # Agricultural managements contribution
        am_j = np.array([sim.data.DESC2AGLU[lu] for lu in am_lus])
        ghg_emissions += ( ag_man_g_mrj[am] 
                         * sim.ag_man_dvars[yr_cal][am][:, :, am_j] ).sum() 

    # Add to dataframe
    df.loc[0] = ("{:,.0f}".format(ghg_limits), "{:,.0f}".format(ghg_emissions))
    
    # Save to file
    df.to_csv(os.path.join(path, 'GHG_emissions.csv'), index = False)
    
def write_ghg_seperate(sim, yr_cal, path):
    # set the aggregate to False, so to calculate the GHG seperately 
    # (i.e, get the GHG emissions according to sources [electricty, chemical fertilizer ...])
    aggregate = False
        
    # Convert calendar year to year index.
    yr_idx = yr_cal - sim.data.YR_CAL_BASE
    
    # -------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural landuse #
    # -------------------------------------------------------#
    
    # get the ghg_df
    ag_g_df = ag_ghg.get_ghg_matrices(sim.data, yr_idx, aggregate)
    
    # fill the ag_g_df so that it can be reshaped to a n-d array
    ag_g_df, ag_g_col_unique, ag_g_col_count = tools.df_sparse2dense(ag_g_df)
    
    
    # reshape the df to the shape of [romjs]
    ag_g_df_arr = ag_g_df.to_numpy(na_value=0).reshape(ag_g_df.shape[0],   # r: row, i.e, each valid pixel
                                                       ag_g_col_count[0],  # o: origin, i.e, each origin [crop, lvstk, unallocated]
                                                       ag_g_col_count[1],  # m: land management: [dry,irr]
                                                       ag_g_col_count[2],  # j: land use [Apples, Beef, ...]
                                                       ag_g_col_count[3])  # s: GHG source [chemical fertilizer, electricity, ...]
    
    # use einsum to do the multiplication, 
    # easy to understand, don't need to worry too much about the dimensionality
    ag_dvar_mrj = sim.ag_dvars[yr_cal]                                               # mrj
    GHG_emission_seperate = np.einsum('romjs,mrj -> rms', ag_g_df_arr, ag_dvar_mrj)  # rms
    
    # warp the array back to a df
    GHG_emission_seperate = pd.DataFrame(GHG_emission_seperate.reshape((GHG_emission_seperate.shape[0],-1)),
                                         columns=pd.MultiIndex.from_product((ag_g_col_unique[1],    # m: land management
                                                                             ag_g_col_unique[3])))  # s: GHG source 
    
    # add landuse describtion
    GHG_emission_seperate['lu'] = [sim.data.AGLU2DESC[x] for x in sim.data.LUMAP]
    
    # sumarize the GHG as (lucc * [lu|source])
    GHG_emission_seperate_summary = GHG_emission_seperate.groupby('lu').sum(0).reset_index()
    GHG_emission_seperate_summary = GHG_emission_seperate_summary.set_index('lu')
    
    # Save to pickle file, so to keep the multilvel columns
    GHG_emission_seperate.to_csv(os.path.join(path, 'GHG_emissions_seperate.csv'))
    
