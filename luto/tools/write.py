#!/bin/env python3
#
# write.py - writes model output and statistics to files.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-02-22
# Last modified: 2022-01-21
#

import os
from datetime import datetime

import numpy as np
import pandas as pd

from luto.tools.highposgtiff import *
from luto.tools.compmap import *

def write(data, sim, year):
    # Create a directory for the files.
    path = datetime.today().strftime('%Y%m%d')
    if not os.path.exists(path):
        os.mkdir(path)

    lumap_fname = 'lumap' + str(year) + '.npy'
    lmmap_fname = 'lmmap' + str(year) + '.npy'
    np.save(os.path.join(path, lumap_fname), sim.lumaps[year])
    np.save(os.path.join(path, lmmap_fname), sim.lmmaps[year])

    lumap_fname = 'lumap' + str(year) + '.tiff'
    lmmap_fname = 'lmmap' + str(year) + '.tiff'
    write_lumap_gtiff(sim.lumaps[year], os.path.join(path, lumap_fname))
    write_lumap_gtiff(sim.lmmaps[year], os.path.join(path, lmmap_fname))

    d_c = sim.get_production()
    pr2011 = sim.get_production(sim.lumaps[year], sim.lmmaps[year])

    diff = pr2011 - d_c

    df = pd.DataFrame()
    df['Surplus'] = diff
    df['Commodity'] = data.COMMODITIES
    df.to_csv(os.path.join(path, 'demand-surplus-deficit.csv'), index=False)

    np.save('decvars-mrj.npy', sim.dvars[year])

    LUS = ['Non-agricultural land'] + data.LANDUSES

    ctlu, swlu = crossmap(sim.lumaps[2010], sim.lumaps[year], LUS)
    ctlm, swlm = crossmap(sim.lmmaps[2010], sim.lmmaps[year])

    cthp, swhp = crossmap_irrstat( sim.lumaps[2010], sim.lmmaps[2010]
                                 , sim.lumaps[year], sim.lmmaps[year]
                                 , data.LANDUSES )

    ctlu.to_csv(os.path.join(path, 'crosstab-lumap.csv'))
    ctlm.to_csv(os.path.join(path, 'crosstab-lmmap.csv'))

    swlu.to_csv(os.path.join(path, 'switches-lumap.csv'))
    swlm.to_csv(os.path.join(path, 'switches-lmmap.csv'))

    cthp.to_csv(os.path.join(path, 'crosstab-irrstat.csv'))
    swhp.to_csv(os.path.join(path, 'switches-irrstat.csv'))

