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

# Summarise and write outputs to file. 'year' is calendar year (e.g., 2030)
def write(data, sim, year):
    
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
    lumap_fname = 'lumap' + str(data.ANNUM) + '.tiff'
    lmmap_fname = 'lmmap' + str(data.ANNUM) + '.tiff'
    write_lumap_gtiff(sim.lumaps[data.ANNUM], os.path.join(path, lumap_fname))
    write_lumap_gtiff(sim.lmmaps[data.ANNUM], os.path.join(path, lmmap_fname))
    
    # Write out the target year GeoTiffs for land-use and land management
    lumap_fname = 'lumap' + str(year) + '.tiff'
    lmmap_fname = 'lmmap' + str(year) + '.tiff'
    write_lumap_gtiff(sim.lumaps[year], os.path.join(path, lumap_fname))
    write_lumap_gtiff(sim.lmmaps[year], os.path.join(path, lmmap_fname))
    
    # Calculate data for quantity comparison
    prod_base = sim.get_production(data.ANNUM)     # Get commodity quantities produced in 2010 
    prod_targ = sim.get_production(year)           # Get commodity quantities produced in target year
    demands = sim.d_c[year - data.ANNUM]           # Get commodity demands for target year
    abs_diff = prod_targ - demands                 # Diff between taget year production and demands in absolute terms (i.e. tonnes etc)
    prop_diff = (abs_diff / prod_targ) * 100       # Diff between taget year production and demands in relative terms (i.e. %)

    df = pd.DataFrame()
    df['Commodity'] = data.COMMODITIES
    df['Prod_base_year (tonnes, KL)'] = prod_base
    df['Prod_targ_year (tonnes, KL)'] = prod_targ
    df['Demand (tonnes, KL)'] = demands
    df['Abs_diff (tonnes, KL)'] = abs_diff
    df['Prop_diff (%)'] = prop_diff
    df.to_csv(os.path.join(path, 'quantity_comparison.csv'), index = False)
    
    # Save decision variables. Commented out as the output is 1.8GB
    # np.save(os.path.join(path, 'decvars-mrj.npy'), sim.dvars[year])

    LUS = ['Non-agricultural land'] + data.LANDUSES

    ctlu, swlu = crossmap(sim.lumaps[data.ANNUM], sim.lumaps[year], LUS)
    ctlm, swlm = crossmap(sim.lmmaps[data.ANNUM], sim.lmmaps[year])

    cthp, swhp = crossmap_irrstat( sim.lumaps[data.ANNUM], sim.lmmaps[data.ANNUM]
                                 , sim.lumaps[year], sim.lmmaps[year]
                                 , data.LANDUSES )

    ctlu.to_csv(os.path.join(path, 'crosstab-lumap.csv'))
    ctlm.to_csv(os.path.join(path, 'crosstab-lmmap.csv'))

    swlu.to_csv(os.path.join(path, 'switches-lumap.csv'))
    swlm.to_csv(os.path.join(path, 'switches-lmmap.csv'))

    cthp.to_csv(os.path.join(path, 'crosstab-irrstat.csv'))
    swhp.to_csv(os.path.join(path, 'switches-irrstat.csv'))

