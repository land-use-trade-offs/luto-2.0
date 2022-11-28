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

from luto.tools.highposgtiff import *
from luto.tools.compmap import *
from luto.economics.water import *
import luto.settings as settings


# Summarise and write outputs to file. 'year' is calendar year (e.g., 2030)
def write(sim, year, d_c):
    
    # Create a directory for the files.
    path = datetime.today().strftime('%Y_%m_%d__%H_%M_%S')
    path = 'output/' + path 
    if not os.path.exists(path):
        os.mkdir(path)
    
    print('\nWriting outputs to', path, '\n')
    

    # Write out the target year numpy arrays
    lumap_fname = 'lumap' + str(year) + '.npy'
    lmmap_fname = 'lmmap' + str(year) + '.npy'
    np.save(os.path.join(path, lumap_fname), sim.lumaps[year])
    np.save(os.path.join(path, lmmap_fname), sim.lmmaps[year])
    
    # Write out the 2010 GeoTiffs for land-use and land management
    lumap_fname = 'lumap' + str(sim.data.ANNUM) + '.tiff'
    lmmap_fname = 'lmmap' + str(sim.data.ANNUM) + '.tiff'
    write_lumap_gtiff(sim.lumaps[sim.data.ANNUM], os.path.join(path, lumap_fname))
    write_lumap_gtiff(sim.lmmaps[sim.data.ANNUM], os.path.join(path, lmmap_fname))
    
    # Write out the target year GeoTiffs for land-use and land management
    lumap_fname = 'lumap' + str(year) + '.tiff'
    lmmap_fname = 'lmmap' + str(year) + '.tiff'
    write_lumap_gtiff(sim.lumaps[year], os.path.join(path, lumap_fname))
    write_lumap_gtiff(sim.lmmaps[year], os.path.join(path, lmmap_fname))
    
    # Calculate data for quantity comparison
    prod_base = sim.get_production(sim.data.ANNUM)     # Get commodity quantities produced in 2010 
    prod_targ = sim.get_production(year)           # Get commodity quantities produced in target year
    demands = d_c[year - sim.data.ANNUM]           # Get commodity demands for target year
    abs_diff = prod_targ - demands                 # Diff between taget year production and demands in absolute terms (i.e. tonnes etc)
    prop_diff = (abs_diff / prod_targ) * 100       # Diff between taget year production and demands in relative terms (i.e. %)

    df = pd.DataFrame()
    df['Commodity'] = sim.data.COMMODITIES
    df['Prod_base_year (tonnes, KL)'] = prod_base
    df['Prod_targ_year (tonnes, KL)'] = prod_targ
    df['Demand (tonnes, KL)'] = demands
    df['Abs_diff (tonnes, KL)'] = abs_diff
    df['Prop_diff (%)'] = prop_diff
    df.to_csv(os.path.join(path, 'quantity_comparison.csv'), index = False)
    
    # Save decision variables. Commented out as the output is 1.8GB
    # np.save(os.path.join(path, 'decvars-mrj.npy'), sim.dvars[year])

    LUS = ['Non-agricultural land'] + sim.data.LANDUSES

    ctlu, swlu = crossmap(sim.lumaps[sim.data.ANNUM], sim.lumaps[year], LUS)
    ctlm, swlm = crossmap(sim.lmmaps[sim.data.ANNUM], sim.lmmaps[year])

    cthp, swhp = crossmap_irrstat( sim.lumaps[sim.data.ANNUM], sim.lmmaps[sim.data.ANNUM]
                                 , sim.lumaps[year], sim.lmmaps[year]
                                 , sim.data.LANDUSES )

    ctlu.to_csv(os.path.join(path, 'crosstab-lumap.csv'))
    ctlm.to_csv(os.path.join(path, 'crosstab-lmmap.csv'))

    swlu.to_csv(os.path.join(path, 'switches-lumap.csv'))
    swlm.to_csv(os.path.join(path, 'switches-lmmap.csv'))

    cthp.to_csv(os.path.join(path, 'crosstab-irrstat.csv'))
    swhp.to_csv(os.path.join(path, 'switches-irrstat.csv'))



def get_water_totals(data, sim, year):
    """Calculate water use totals. 
       Takes a data object, a simulation object, and a numeric year (e.g., 2030) as input."""
    
    # Get land-use and land management maps from sim object and slice to ag cells    
    lumap = sim.lumaps[year][data.mindices]
    lmmap = sim.lmmaps[year][data.mindices]

    # Get the lumap + lmmap in decision variable format.
    X_mrj = lumap2x_mrj(lumap, lmmap)
    
    # Get 2010 water requirement in mrj format
    aqreq_mrj = get_aqreq_matrices(data)

    # Prepare a data frame.
    df = pd.DataFrame( columns=[ 'REGION_ID'
                               , 'REGION_NAME'
                               , 'WATER_USE_LIMIT_ML'
                               , 'TOT_WATER_REQ_ML' ] )

    # Get water use limits used as constraints in model
    _, aqreq_limits = get_aqreq_limits(data)


    # Set up data for river regions or drainage divisions
    if settings.WATER_REGION_DEF == 'RR':
        regions = settings.WATER_RIVREGS
        region_id = data.RIVREG_ID
        region_dict = data.RIVREG_DICT
        
    elif settings.WATER_REGION_DEF == 'DD':
        regions = settings.WATER_DRAINDIVS
        region_id = data.DRAINDIV_ID
        region_dict = data.DRAINDIV_DICT
        
    else: print('Incorrect option for WATER_REGION_DEF in settings')
    
    # Loop through specified water regions
    for i, region in enumerate(regions):
        
        # Get indices of cells in region
        ind = np.flatnonzero(region_id == region).astype(np.int32)
        
        # Calculate the 2010 water requiremnents by agriculture for region.
        aqreq_reg = (aqreq_mrj[:, ind, :] * 
                               X_mrj[:, ind, :]).sum()
        
        # Add to dataframe
        df.loc[i] = (region, region_dict[region], aqreq_limits[i][1], aqreq_reg)
    
    df.to_csv(os.path.join(path, 'water_demand_vs_use.csv'), index = False)