#!/bin/env python3
#
# write.py - writes model output and statistics to files.
#
# Author: Fjalar de Haan (f.dehaan@deakin.edu.au)
# Created: 2021-02-22
# Last modified: 2021-10-06
#

import os
from datetime import datetime

import numpy as np
import pandas as pd

from luto.tools.highposgtiff import *
from luto.tools.compmap import *

def write(data, sim):
    # Create a directory for the files.
    path = datetime.today().strftime('%Y%m%d')
    if not os.path.exists(path):
        os.mkdir(path)

    np.save(os.path.join(path, 'lumap2011.npy'), sim.lumaps[2011])
    np.save(os.path.join(path, 'lmmap2011.npy'), sim.lmmaps[2011])

    LUS = ['Non-agricultural land'] + data.LANDUSES

    ctlu, swlu = crossmap(sim.lumaps[2010], sim.lumaps[2011], LUS)
    ctlm, swlm = crossmap(sim.lmmaps[2010], sim.lmmaps[2011])

    cthp, swhp = crossmap_irrstat( sim.lumaps[2010], sim.lmmaps[2010]
                                 , sim.lumaps[2011], sim.lmmaps[2011]
                                 , data.LANDUSES )

    ctlu.to_csv(os.path.join(path, 'crosstab-lumap.csv'))
    ctlm.to_csv(os.path.join(path, 'crosstab-lmmap.csv'))

    swlu.to_csv(os.path.join(path, 'switches-lumap.csv'))
    swlm.to_csv(os.path.join(path, 'switches-lmmap.csv'))

    cthp.to_csv(os.path.join(path, 'crosstab-irrstat.csv'))
    swhp.to_csv(os.path.join(path, 'switches-irrstat.csv'))

    write_lumap_gtiff(sim.lumaps[2011], os.path.join(path, 'lumap2011.tiff'))
    write_lumap_gtiff(sim.lmmaps[2011], os.path.join(path, 'lmmap2011.tiff'))

    d_c = sim.get_production()
    pr2011 = sim.get_production(sim.lumaps[2011], sim.lmmaps[2011])

    diff = pr2011 - d_c

    df = pd.DataFrame()
    df['Surplus'] = diff
    df['Commodity'] = data.COMMODITIES
    df.to_csv(os.path.join(path, 'demand-surplus-deficit.csv'), index=False)
