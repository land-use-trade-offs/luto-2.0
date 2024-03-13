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

# # To run LUTO, execute steps 1-4 below...

# 1. Refresh input data (if required)
from luto.dataprep import create_new_dataset
create_new_dataset()

# 2. Run the simulation and profile memory use
%load_ext memory_profiler
import luto.simulation as sim
%memit sim.run( 2010, 2050 )

# 3. Write the ouputs to file
from luto.tools.write import *
write_outputs(sim)



# Minimalist run code
from luto.dataprep import create_new_dataset
create_new_dataset()

import luto.simulation as sim
sim.run( 2010, 2050 )
from luto.tools.write import write_outputs
write_outputs(sim)





#################################################### Pixel-level data testing code

import pandas as pd
import numpy as np
from luto.tools import amortise
import luto.simulation as sim
import luto.economics.agricultural.transitions as ag_transition
import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.revenue as ag_revenue
from luto.economics.agricultural.ghg import get_ghg_transition_penalties
from luto import settings
import random as rand
  
# Load data object into memory
data = sim.Data(sim.bdata, 2030)

# Read in the lmap dataframe which contains land-use and land management mapping from 2010, mask and reset index
lmap = pd.read_hdf('../raw_data/cell_LU_mapping.h5')[data.MASK].reset_index()

# Load cropping and irrigation exclusion factors and mask them
prec_over_175mm = pd.read_hdf('../raw_data/cell_biophysical_df.h5')['AVG_GROW_SEAS_PREC_GE_175_MM_YR'].to_numpy()[data.MASK]
pot_irr_areas = pd.read_hdf('../raw_data/cell_zones_df.h5')['POTENTIAL_IRRIGATION_AREAS'].to_numpy()[data.MASK]

# Get array containing cell index values where potential irrigation == 1
irr_idx = np.nonzero(pot_irr_areas)[0]

# Load land use and land management maps
lumap = data.LUMAP
lmmap = data.LMMAP
lumaps = {2010:lumap}
lmmaps = {2010:lmmap}
d = {0:'Dryland', 1:'Irrigated'}

# Calculate the exclude, transistions costs, and cost of production matrices
x_mrj = ag_transition.get_exclude_matrices(data, 2010, lumaps)
t_mrj = ag_transition.get_transition_matrices(data, 20, 2010, lumaps, lmmaps)
gct_mrj = get_ghg_transition_penalties(data, lumap) * settings.CARBON_PRICE_PER_TONNE
c_mrj = ag_cost.get_cost_matrices(data, 0, lumap) - gct_mrj
r_mrj = ag_revenue.get_rev_matrices(data, 0)

# Load ag_X_mrj file from a run with the same resfactor
ag_X_mrj = np.load('output/2023_09_02__14_02_41/ag_X_mrj_2030.npy')

# Get cells which are Sheep - modified land in 2030
sheep_mod_2030 = ag_X_mrj[..., data.AGRICULTURAL_LANDUSES.index('Sheep - modified land')]

# Get cells which are Unallocated - natural land in 2010
uall_nat_2010 = lumap == data.AGRICULTURAL_LANDUSES.index('Unallocated - natural land')

# Get cells which change from Unall - nat to Sheep - mod
xind = np.nonzero( (sheep_mod_2030 * uall_nat_2010)[0] )[0]

def spot_check(r):
    
    # Get SA2_ID
    sa2 = lmap.loc[r, 'SA2_ID'].item()
    
    print('\nCELL_ID', r, data.AGRICULTURAL_LANDUSES[lumap[r]], ',', d[lmmap[r]], ', SA2', sa2, ', RF excl', str(prec_over_175mm[r]) + ', IRR excl', str(pot_irr_areas[r]), '\n')
    
    # Print unique land-use and land management types in the SA2
    df = lmap.query('SA2_ID == @sa2')[['IRRIGATION', 'LU_DESC']].drop_duplicates()
    df = df.sort_values(['IRRIGATION', 'LU_DESC'])
    print(df)
    
    print('\n' + f'{"Dryland land-uses" : <48}{"Irrigated land-uses" : <0}')
    
    print(f'{"X" : <4}{"TransCosts" : <12}{"GHG_Costs" : <12}{"Costs" : <10}{"Rev" : <10}{"X" : <4}{"TransCosts" : <12}{"GHG_Costs" : <12}{"Costs" : <10}{"Rev" : <10}')
            
    for i in range(len(data.AGRICULTURAL_LANDUSES)):
        print(f'{x_mrj[0, r, i] : <4}{t_mrj[0, r, i] : <12,.0f}{gct_mrj[0, r, i] : <12,.0f}{c_mrj[0, r, i] : <10,.0f}{r_mrj[0, r, i] : <10,.0f}{x_mrj[1, r, i] : <4}{t_mrj[1, r, i] : <12,.0f}{gct_mrj[1, r, i] : <12,.0f}{c_mrj[1, r, i] : <10,.0f}{r_mrj[1, r, i] : <10,.0f}{data.AGRICULTURAL_LANDUSES[i] : <10}')

r = rand.randint(0, data.NCELLS)
r = rand.choice(irr_idx)
r = rand.choice(xind)
spot_check(r)

####################################################





write_files(sim, path)
write_production(sim, 2030, d_c, path)
write_water(sim, 2030, path)
write_ghg(sim, 2030, path)

data = sim.data
df_2010 = get_water_totals(data, sim.lumaps[2010], sim.lmmaps[2010])
df_2030 = get_water_totals(data, sim.lumaps[2030], sim.lmmaps[2030])



# Example of code to inspect things 
sim.bdata.WREQ_CROPS_DRY_RJ

import sys
sys.getsizeof(sim.data.WREQ_LVSTK_DRY_RJ)

import luto.simulation as sim
t_mrj, c_mrj, q_mrp, l_mrj = temp(2010, 2030, sim.d_c, 1000)

#Check compare output maps
import numpy as np
new = np.load('N:/LUF-Modelling/LUTO2_BB/LUTO2/output/2022_11_20__09_23_06/lumap2030.npy')
old = np.load('N:/LUF-Modelling/LUTO2_BB/LUTO2/output/2022_11_20__09_24_47/lumap2030.npy')
ind = np.flatnonzero(old != new)
ind.shape
for row in ind:
    print(old[row], new[row])

# Get commodity production totals when no resfactor used and check against raw data
x = sim.get_production()
for i, j in enumerate(sim.data.COMMODITIES):
    print(j, x[i])

import pandas as pd
raw_econ = pd.read_hdf('N:/Data-Master/Profit_map/cell_ag_data.h5')
raw_yield = raw_econ.groupby(['SPREAD_DESC'], observed = True, as_index = False).agg(CROPS_TONNES = ('CROPS_TONNES', 'sum'),
                                                                                     MEAT_EXCL_LEXP_TONNES = ('MEAT_EXCL_LEXP_TONNES', 'sum'),
                                                                                     MEAT_INCL_LEXP_TONNES = ('MEAT_INCL_LEXP_TONNES', 'sum'),
                                                                                     LIVE_EXP_HEAD = ('LIVE_EXP_HEAD', 'sum'),
                                                                                     LIVE_EXP_MEAT_TONNES = ('LIVE_EXP_MEAT_TONNES', 'sum'),
                                                                                     MILK_KL = ('MILK_KL', 'sum'),
                                                                                     WOOL_TONNES = ('WOOL_TONNES', 'sum'))
         



dem_SSP2 = np.load('N:/LUF-Modelling/fdh-archive/data/neoluto-data/new-data-and-domain/demands-ssp2-2010-2100.npy')


agec_crops_fdh_neo = pd.read_hdf('N:/LUF-Modelling/fdh-archive/dev/neoluto/input/agec-crops-c9.hdf5')

agec_crops_fdh = pd.read_hdf('N:/LUF-Modelling/fdh-archive/data/neoluto-data/new-data-and-domain/agec-crops-c9.hdf5')

agec_crops_fdh_neo.equals(agec_crops_fdh)


AGEC_CROPS.columns.get_level_values(2).unique()

AGGHG_CROPS.columns.levels

AGGHG_CROPS.loc[0:10, (slice(None), 'dry', 'Winter cereals')]


for lu in sim.bdata.AGRICULTURAL_LANDUSES:
    if lu in x_dry.columns:
        print(lu, 'True')
    else:
        print(lu, 'False')
        
def i(array):
    print('Size in memory {:.2f} MB'.format(array.itemsize * array.size / 2**20))
    print('Shape %s' % str(array.shape))
    print('Datatype %s' % str(array.dtype))

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
