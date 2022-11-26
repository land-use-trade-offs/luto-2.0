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
LUTO 2.0 temporary helper code.
"""


from luto.dataprep import create_new_dataset
create_new_dataset()


%load_ext memory_profiler
import luto.simulation as sim
%memit sim.run( 2010, 2030, sim.d_c ) #, 1000000, style = 'snapshot', resfactor = False )

import luto.simulation as sim
sim.run( 2010, 2030, sim.d_c )

from luto.tools.write import write
write(sim.data, sim, 2030)

from luto.economics.water import *
data = sim.data
df_2010 = get_water_totals(data, sim.lumaps[2010], sim.lmmaps[2010])
df_2030 = get_water_totals(data, sim.lumaps[2030], sim.lmmaps[2030])



# Example of code to inspect things 
sim.bdata.AQ_REQ_CROPS_DRY_RJ

import sys
sys.getsizeof(sim.data.AQ_REQ_LVSTK_DRY_RJ)

import luto.simulation as sim
t_mrj, c_mrj, q_mrp, x_mrj = temp(2010, 2030, sim.d_c, 1000)

#Check compare output maps
import numpy as np
new = np.load('N:/Planet-A/LUF-Modelling/LUTO2_BB/LUTO2/output/2022_11_20__09_23_06/lumap2030.npy')
old = np.load('N:/Planet-A/LUF-Modelling/LUTO2_BB/LUTO2/output/2022_11_20__09_24_47/lumap2030.npy')
ind = np.flatnonzero(old != new)
ind.shape
for row in ind:
    print(old[row], new[row])

# Get commodity production totals when no resfactor used and check against raw data
x = sim.get_production()
for i, j in enumerate(sim.data.COMMODITIES):
    print(j, x[i])

import pandas as pd
raw_econ = pd.read_hdf('N:/Planet-A/Data-Master/Profit_map/cell_ag_data.h5')
raw_yield = raw_econ.groupby(['SPREAD_DESC'], observed = True, as_index = False).agg(CROPS_TONNES = ('CROPS_TONNES', 'sum'),
                                                                                     MEAT_EXCL_LEXP_TONNES = ('MEAT_EXCL_LEXP_TONNES', 'sum'),
                                                                                     MEAT_INCL_LEXP_TONNES = ('MEAT_INCL_LEXP_TONNES', 'sum'),
                                                                                     LIVE_EXP_HEAD = ('LIVE_EXP_HEAD', 'sum'),
                                                                                     LIVE_EXP_MEAT_TONNES = ('LIVE_EXP_MEAT_TONNES', 'sum'),
                                                                                     MILK_KL = ('MILK_KL', 'sum'),
                                                                                     WOOL_TONNES = ('WOOL_TONNES', 'sum'))
         



dem_SSP2 = np.load('N:/Planet-A/LUF-Modelling/fdh-archive/data/neoluto-data/new-data-and-domain/demands-ssp2-2010-2100.npy')


agec_crops_fdh_neo = pd.read_hdf('N:/Planet-A/LUF-Modelling/fdh-archive/dev/neoluto/input/agec-crops-c9.hdf5')

agec_crops_fdh = pd.read_hdf('N:/Planet-A/LUF-Modelling/fdh-archive/data/neoluto-data/new-data-and-domain/agec-crops-c9.hdf5')

agec_crops_fdh_neo.equals(agec_crops_fdh)




for lu in sim.bdata.LANDUSES:
    if lu in x_dry.columns:
        print(lu, 'True')
    else:
        print(lu, 'False')
        
        
        

def temp( base    # Base year from which the data is taken.
        , target  # Year to be solved for.
        , demands # Demands in the form of a d_c array.
        , penalty # Penalty level.
        # , limits  # (Environmental) limits as additional soft/hard constraints.
        ):
    """Solve the linear programme using the `base` lumap for `target` year."""

    # Synchronise base and target years across module so matrix-getters know.
    sync_years(base, target)

    return get_t_mrj(), get_c_mrj(), get_q_mrp(), get_x_mrj()
