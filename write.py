import numpy as np
import pandas as pd

from luto.tools.highposgtiff import *
from luto.tools.compmap import *

def write(data, sim):
    np.save('lumap2011-norf-p100.npy', sim.lumaps[2011])
    np.save('lmmap2011-norf-p100.npy', sim.lmmaps[2011])

    LUS = ['Non-agricultural land'] + data.LANDUSES

    ctlu, swlu = crossmap(sim.lumaps[2010], sim.lumaps[2011], LUS)
    ctlm, swlm = crossmap(sim.lmmaps[2010], sim.lmmaps[2011])

    ctlu.to_csv('crosstab-lumap.csv')
    ctlm.to_csv('crosstab-lmmap.csv')

    swlu.to_csv('switches-lumap.csv')
    swlm.to_csv('switches-lmmap.csv')

    write_lumap_gtiff(sim.lumaps[2011], 'lumap2011-norf-p100.tiff')
    write_lumap_gtiff(sim.lmmaps[2011], 'lmmap2011-norf-p100.tiff')

    d_c = sim.get_production()
    pr2011 = sim.get_production(sim.lumaps[2011], sim.lmmaps[2011])

    diff = pr2011 - d_c

    df = pd.DataFrame()
    df['Surplus'] = diff
    df['Commodity'] = data.COMMODITIES
    df.to_csv('demand-surplus-deficit.csv', index=False)
