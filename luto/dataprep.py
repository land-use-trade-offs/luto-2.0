# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.



"""
Script to load and prepare input data based on build.ipynb by F. de Haan
and Brett Bryan, Deakin University

"""


# Load libraries
import numpy as np
import pandas as pd
import shutil, os, time, h5py

from joblib import Parallel, delayed
from luto.settings import INPUT_DIR, RAW_DATA



def create_new_dataset():
    """Creates a new LUTO input dataset from source data"""

    # Set up a timer and print the time
    start_time = time.time()
    print('\nBeginning input data refresh at', time.strftime("%H:%M:%S", time.localtime()) + '...')


    ############### Copy key input data layers from their source folders to the raw_data folder for processing

    # Set data input paths
    luto_1D_inpath = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/1D_Parameter_Timeseries/'
    luto_2D_inpath = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/'
    # luto_3D_inpath = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/3D_Spatial_Timeseries/'
    luto_4D_inpath = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/'
    fdh_inpath = 'N:/LUF-Modelling/fdh-archive/data/neoluto-data/new-data-and-domain/'
    profit_map_inpath = 'N:/Data-Master/Profit_map/'
    nlum_inpath = 'N:/Data-Master/National_Landuse_Map/'
    BECCS_inpath = 'N:/Data-Master/BECCS/From_CSIRO/20211124_as_submitted/'
    GHG_off_land_inpath = 'N:/LUF-Modelling/Food_demand_AU/au.food.demand/Inputs/Off_land_GHG_emissions'
    bio_GBF_4a_inpath = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km_to_NetCDF/'
    bio_GBF_4b_inpath = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/To_NetCDF/'
    bio_NVIS_inpath = 'N:/Data-Master/NVIS/'
    bio_HACS_inpath = 'N:/Data-Master/Habitat_condition_assessment_system/Data/Processed/'

    # Set data output paths
    raw_data = RAW_DATA + '/' # '../raw_data/'
    outpath = INPUT_DIR + '/'

    # Delete the data folders' contents
    for file in os.scandir(outpath):
        if file.name != '.gitignore':
            os.remove(file.path)
    for file in os.scandir(raw_data): os.remove(file.path)

    # Copy raw data files from their source into raw_data folder for further processing

    shutil.copyfile(fdh_inpath + 'tmatrix-cat2lus.csv', raw_data + 'tmatrix_cat2lus.csv')
    shutil.copyfile(fdh_inpath + 'transitions_costs_20230901.xlsx', raw_data + 'transitions_costs_20230901.xlsx')

    shutil.copyfile(profit_map_inpath + 'NLUM_SPREAD_LU_ID_Mapped_Concordance.h5', raw_data + 'NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')

    shutil.copyfile(luto_2D_inpath + 'cell_LU_mapping.h5', raw_data + 'cell_LU_mapping.h5')
    shutil.copyfile(luto_2D_inpath + 'cell_zones_df.h5', raw_data + 'cell_zones_df.h5')
    shutil.copyfile(luto_2D_inpath + 'cell_livestock_data.h5', raw_data + 'cell_livestock_data.h5')
    shutil.copyfile(luto_2D_inpath + 'SA2_livestock_GHG_data.h5', raw_data + 'SA2_livestock_GHG_data.h5')
    shutil.copyfile(luto_2D_inpath + 'SA2_irrigated_pasture_GHG_data.h5', raw_data + 'SA2_irrigated_pasture_GHG_data.h5')
    shutil.copyfile(luto_2D_inpath + 'SA2_crop_data.h5', raw_data + 'SA2_crop_data.h5')
    shutil.copyfile(luto_2D_inpath + 'SA2_crop_GHG_data.h5', raw_data + 'SA2_crop_GHG_data.h5')
    shutil.copyfile(luto_2D_inpath + 'cell_biophysical_df.h5', raw_data + 'cell_biophysical_df.h5')
    shutil.copyfile(luto_2D_inpath + 'SA2_climate_damage_mult.h5', raw_data + 'SA2_climate_damage_mult.h5')

    shutil.copyfile('N:/LUF-Modelling/Food_demand_AU/au.food.demand/Outputs/All_LUTO_demand_scenarios_with_convergences.csv',  raw_data + 'All_LUTO_demand_scenarios_with_convergences.csv')

    # Read raw BECCS data from CSIRO and save as HDF5
    BECCS_raw = pd.read_pickle(BECCS_inpath + 'df_info_best_grid_20211116.pkl')


    # Copy data straight to LUTO input folder, no processing required

    shutil.copyfile(fdh_inpath + 'yieldincreases-bau2022.csv', outpath + 'yieldincreases_bau2022.csv')
    shutil.copyfile(nlum_inpath + 'NLUM_2010-11_mask.tif', outpath + 'NLUM_2010-11_mask.tif')
    shutil.copyfile(nlum_inpath + 'ag_landuses.csv', outpath + 'ag_landuses.csv')

    shutil.copyfile(luto_1D_inpath + 'GHG_targets_20240421.xlsx', outpath + 'GHG_targets.xlsx')
    shutil.copyfile(luto_1D_inpath + 'carbon_prices_20240612.xlsx', outpath + 'carbon_prices.xlsx')
    shutil.copyfile(luto_1D_inpath + 'ag_price_multipliers_20240612.xlsx', outpath + 'ag_price_multipliers.xlsx')
    shutil.copyfile(luto_1D_inpath + 'cost_multipliers_20240612.xlsx', outpath + 'cost_multipliers.xlsx')

    shutil.copyfile(luto_2D_inpath + 'cell_savanna_burning.h5', outpath + 'cell_savanna_burning.h5')

    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp126_2010-2100_DR_ML_HA_mean.h5', outpath + 'water_yield_ssp126_2010-2100_dr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp126_2010-2100_SR_ML_HA_mean.h5', outpath + 'water_yield_ssp126_2010-2100_sr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp245_2010-2100_DR_ML_HA_mean.h5', outpath + 'water_yield_ssp245_2010-2100_dr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp245_2010-2100_SR_ML_HA_mean.h5', outpath + 'water_yield_ssp245_2010-2100_sr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp370_2010-2100_DR_ML_HA_mean.h5', outpath + 'water_yield_ssp370_2010-2100_dr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp370_2010-2100_SR_ML_HA_mean.h5', outpath + 'water_yield_ssp370_2010-2100_sr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp585_2010-2100_DR_ML_HA_mean.h5', outpath + 'water_yield_ssp585_2010-2100_dr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp585_2010-2100_SR_ML_HA_mean.h5', outpath + 'water_yield_ssp585_2010-2100_sr_ml_ha.h5')

    # Copy agricultural management datafiles
    shutil.copyfile(luto_1D_inpath + '20231101_Bundle_MR.xlsx', outpath + '20231101_Bundle_MR.xlsx')
    shutil.copyfile(luto_1D_inpath + '20231101_Bundle_AgTech_NE.xlsx', outpath + '20231101_Bundle_AgTech_NE.xlsx')
    shutil.copyfile(luto_1D_inpath + '20231107_ECOGRAZE_Bundle.xlsx', outpath + '20231107_ECOGRAZE_Bundle.xlsx')
    shutil.copyfile(luto_1D_inpath + '20231107_Bundle_AgTech_EI.xlsx', outpath + '20231107_Bundle_AgTech_EI.xlsx')
    shutil.copyfile(luto_1D_inpath + '20240918_Bundle_BC.xlsx', outpath + '20240918_Bundle_BC.xlsx')

    # Copy biodiversity GBF-4A files
    shutil.copyfile(bio_GBF_4a_inpath + 'BIODIVERSITY_GBF4A_SCORES.csv', outpath + 'BIODIVERSITY_GBF4A_SCORES.csv')
    shutil.copyfile(bio_GBF_4a_inpath + 'BIODIVERSITY_GBF4A_TARGET.csv', outpath + 'BIODIVERSITY_GBF4A_TARGET.csv')
    
    shutil.copyfile(bio_GBF_4a_inpath + 'bio_ssp126_Condition_group.nc', outpath + 'bio_ssp126_Condition_group.nc')
    shutil.copyfile(bio_GBF_4a_inpath + 'bio_ssp126_EnviroSuit.nc', outpath + 'bio_ssp126_EnviroSuit.nc')
    shutil.copyfile(bio_GBF_4a_inpath + 'bio_ssp245_Condition_group.nc', outpath + 'bio_ssp245_Condition_group.nc')
    shutil.copyfile(bio_GBF_4a_inpath + 'bio_ssp245_EnviroSuit.nc', outpath + 'bio_ssp245_EnviroSuit.nc')
    shutil.copyfile(bio_GBF_4a_inpath + 'bio_ssp370_Condition_group.nc', outpath + 'bio_ssp370_Condition_group.nc')
    shutil.copyfile(bio_GBF_4a_inpath + 'bio_ssp370_EnviroSuit.nc', outpath + 'bio_ssp370_EnviroSuit.nc')
    shutil.copyfile(bio_GBF_4a_inpath + 'bio_ssp585_Condition_group.nc', outpath + 'bio_ssp585_Condition_group.nc')
    shutil.copyfile(bio_GBF_4a_inpath + 'bio_ssp585_EnviroSuit.nc', outpath + 'bio_ssp585_EnviroSuit.nc')
    
    # Copy biodiversity GBF-4B files
    shutil.copyfile(bio_GBF_4b_inpath + 'bio_DCCEEW_ECNES_target.csv', outpath + 'bio_DCCEEW_ECNES_target.csv')
    shutil.copyfile(bio_GBF_4b_inpath + 'bio_DCCEEW_SNES_target.csv', outpath + 'bio_DCCEEW_SNES_target.csv')
    
    shutil.copyfile(bio_GBF_4b_inpath + 'bio_DCCEEW_ECNES.nc', outpath + 'bio_DCCEEW_ECNES.nc')
    shutil.copyfile(bio_GBF_4b_inpath + 'bio_DCCEEW_SNES.nc', outpath + 'bio_DCCEEW_SNES.nc')
    
    
    # Copy biodiversity HACS data from DCCEEW
    shutil.copyfile(bio_HACS_inpath + 'HABITAT_CONDITION.csv', outpath + 'HABITAT_CONDITION.csv')
    
    
    # Copy biodiversity NVIS data
    shutil.copyfile(bio_NVIS_inpath + 'NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS7_0_AUST_PRE_MVS_HIGH_SPATIAL_DETAIL.nc', outpath + 'NVIS_MVS_HIGH_SPATIAL_DETAIL.nc')
    shutil.copyfile(bio_NVIS_inpath + 'NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS7_0_AUST_PRE_MVS_LOW_SPATIAL_DETAIL.nc', outpath + 'NVIS_MVS_LOW_SPATIAL_DETAIL.nc')
    shutil.copyfile(bio_NVIS_inpath + 'NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS7_0_AUST_PRE_MVG_HIGH_SPATIAL_DETAIL.nc', outpath + 'NVIS_MVG_HIGH_SPATIAL_DETAIL.nc')
    shutil.copyfile(bio_NVIS_inpath + 'NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS7_0_AUST_PRE_MVG_LOW_SPATIAL_DETAIL.nc', outpath + 'NVIS_MVG_LOW_SPATIAL_DETAIL.nc')
    
    shutil.copyfile(bio_NVIS_inpath + 'NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_MVG_HIGH_SPATIAL_DETAIL.csv', outpath + 'NVIS_MVG_HIGH_SPATIAL_DETAIL.csv')
    shutil.copyfile(bio_NVIS_inpath + 'NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_MVG_LOW_SPATIAL_DETAIL.csv', outpath + 'NVIS_MVG_LOW_SPATIAL_DETAIL.csv')
    shutil.copyfile(bio_NVIS_inpath + 'NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_MVS_HIGH_SPATIAL_DETAIL.csv', outpath + 'NVIS_MVS_HIGH_SPATIAL_DETAIL.csv')
    shutil.copyfile(bio_NVIS_inpath + 'NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_MVS_LOW_SPATIAL_DETAIL.csv', outpath + 'NVIS_MVS_LOW_SPATIAL_DETAIL.csv')
    

    ############### Read data

    # Read cell_LU_mapping.h5
    lmap = pd.read_hdf(raw_data + 'cell_LU_mapping.h5')

    # Read in the cell_zones_df dataframe
    zones = pd.read_hdf(raw_data + 'cell_zones_df.h5')

    # Read in from-to costs in category-to-category format
    # tmcat = pd.read_csv(raw_data + 'tmatrix_categories.csv', index_col = 0)
    tmcat = pd.read_excel( raw_data + 'transitions_costs_20230901.xlsx'
                          , sheet_name = 'Current'
                          , usecols = 'B:M'
                          , skiprows = 5
                          , nrows = 11
                          , index_col = 0)

    # Read transition costs from agricultural land to environmental plantings
    ag_to_new_land_uses = pd.read_excel( raw_data + 'transitions_costs_20230901.xlsx'
                                       , sheet_name = 'Ag_to_new_land-uses'
                                       , usecols = 'B,C'
                                       , index_col = 0 )

    # Read the categories to land-uses concordance
    cat2lus = pd.read_csv(raw_data + 'tmatrix_cat2lus.csv').to_numpy()

    # Read livestock data
    lvstk = pd.read_hdf(raw_data + 'cell_livestock_data.h5')

    # Read livestock GHG emissions data
    lvstkGHG = pd.read_hdf(raw_data + 'SA2_livestock_GHG_data.h5')

    # Read irrigated pasture GHG emissions data
    irrpastGHG = pd.read_hdf(raw_data + 'SA2_irrigated_pasture_GHG_data.h5')

    # Read crops data
    crops = pd.read_hdf(raw_data + 'SA2_crop_data.h5')

    # Read crops GHG emissions data
    cropsGHG = pd.read_hdf(raw_data + 'SA2_crop_GHG_data.h5')

    # Read biophysical data
    bioph = pd.read_hdf(raw_data + 'cell_biophysical_df.h5')

    # Read SPREAD to LU_ID concordance table.
    ut = pd.read_hdf(raw_data + 'NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')

    # Read raw climate impact data.
    cci_raw = pd.read_hdf(raw_data + 'SA2_climate_damage_mult.h5')

    # Read in demand data
    demand = pd.read_csv(raw_data + 'All_LUTO_demand_scenarios_with_convergences.csv')


    ############### Save the layer of states
    state_id = lmap['SA2_ID'].astype(str).str[0].astype(np.int8).values
    np.save(outpath + 'state_id.npy', state_id) # shape: (j,)
    

    ############### Create the CELL_ID to SA2_ID concordance table

    # Make sure the CELL_ID starts at zero, rather than one
    lmap['CELL_ID'] = lmap.eval('CELL_ID - 1')

    # Select appropriate columns
    concordance = lmap[['CELL_ID', 'SA2_ID']]




    ############### Create landuses -- lexicographically ordered list of land-uses (strings)

    # Read in ag-landuse, which is a lexicographically ordered list
    ag_landuses = pd.read_csv(outpath + 'ag_landuses.csv', header = None)[0].to_list()

    # Create a non-agricultural landuses file
    # Do not sort the whole list alphabetically when adding new landuses to the model.
    non_ag_landuses = ['Environmental Plantings', 'Riparian Plantings', 'Agroforestry', 'Carbon Plantings (Block)', 'Carbon Plantings (Belt)', 'BECCS']

    # Save to file
    pd.DataFrame(non_ag_landuses).to_csv(outpath + 'non_ag_landuses.csv', index = False, header = False)



    ############### Create lumap -- 2010 land-use mapping.

    # Read in the 2010 land-use mapping, -1 indicates cells outside the LUTO study area
    lumap = lmap['LU_ID_LUTO']

    # Save to file (int8)
    lumap.to_hdf(outpath + 'lumap.h5', key = 'lumap', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    # Get the index indicating the cells outside the LUTO study area
    idx_out_LUTO = (lumap == -1).values                                             # shape=6956407, sum=2737674
    idx_inside_LUTO = (lumap != -1).values                                          # shape=6956407, sum=4218733



    ############### Create lmmap -- present (2010) land management mapping.

    # For now only 'rain-fed' ('dry' == 0) and 'irrigated' ('irr' == 1) available.
    # lmmap = lmap['IRRIGATION'].to_numpy()

    # Save to file (int8)
    lmap['IRRIGATION'].to_hdf(outpath + 'lmmap.h5', key = 'lmmap', mode = 'w', format = 'fixed', index = False, complevel = 9)




    ############### Create real_area -- hectares per cell, corrected for geographic map projection.

    # Select the appropriate column
    # real_area = zones['CELL_HA'].to_numpy()

    # Save to file
    zones['CELL_HA'].to_hdf(outpath + 'real_area.h5', key = 'real_area', mode = 'w', format = 'fixed', index = False, complevel = 9)




    ############### Create ag_tmatrix -- agricultural transition cost matrix

    # Produce a dictionary for ease of look up.
    l2c = dict([(row[1], row[0]) for row in cat2lus])

    # Prepare indices.
    indices = [(lu1, lu2) for lu1 in ag_landuses for lu2 in ag_landuses]

    # Prepare the DataFrame.
    t = pd.DataFrame(index = ag_landuses, columns = ag_landuses, dtype = np.float32)

    # Fill the DataFrame.
    for i in indices: t.loc[i] = tmcat.loc[l2c[i[0]], l2c[i[1]]]

    # Switching to existing land use (i.e. not switching) does not cost anything.
    for lu in ag_landuses: t.loc[lu, lu] = 0

    # Extract the actual ag_tmatrix Numpy array.
    ag_tmatrix = t.to_numpy()

    # Save numpy array (float32)
    np.save(outpath + 'ag_tmatrix.npy', ag_tmatrix)




    ############### Create cost vector for transitioning from environmental plantings to agricultural land
    ep_to_ag_t = t.loc['Unallocated - natural land'].to_numpy()
    # add cost of establishing irrigation for irrigated cells
    np.save(outpath + 'ep_to_ag_tmatrix.npy', ep_to_ag_t)  # shape: (j,)




    ############### Create cost vector for transitioning from agricultural land to environmental plantings
    ag_to_ep_t = ag_to_new_land_uses.to_numpy()[:, 0]
    # add cost for removal of irrigation for irrigated cells
    np.save(outpath + 'ag_to_ep_tmatrix.npy', ag_to_ep_t)  # shape: (j,)




    ############### Livestock spatial data

    # Save out feed requirement
    lvstk['FEED_REQ'].to_hdf(outpath + 'feed_req.h5', key = 'feed_req', mode = 'w', format = 'fixed', index = False, complevel = 9) ############### Check - just for mapped livestock

    # Round off and save out pasture productivity
    pasture_kg = lvstk['PASTURE_KG_DM_HA'].round(0).astype(np.int16)
    pasture_kg.to_hdf(outpath + 'pasture_kg_dm_ha.h5', key = 'pasture_kg_dm_ha', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Save out safe pasture utilisation rate - natural land
    lvstk['SAFE_PUR_NATL'].to_hdf(outpath + 'safe_pur_natl.h5', key = 'safe_pur_natl', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Save out safe pasture utilisation rate - modified land
    lvstk['SAFE_PUR_MODL'].to_hdf(outpath + 'safe_pur_modl.h5', key = 'safe_pur_modl', mode = 'w', format = 'fixed', index = False, complevel = 9)




    ############### Water delivery and license price data

    # Get water delivery price from the livestock data (crops is same) and save to file
    lvstk['WP'].to_hdf(outpath + 'water_delivery_price.h5', key = 'water_delivery_price', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Get water license price and save to file
    # bioph['WATER_PRICE_ML_BOM'].to_hdf(outpath + 'water_licence_price.h5', key = 'water_licence_price', mode = 'w', format = 'fixed', index = False, complevel = 9)
    bioph['WATER_PRICE_ML_ABARES'].to_hdf(outpath + 'water_licence_price.h5', key = 'water_licence_price', mode = 'w', format = 'fixed', index = False, complevel = 9)




    ############### Soil organic carbon data

    # Get soil organic carbon data and save to file
    bioph['SOC_T_HA_TOP_30CM'].to_hdf(outpath + 'soil_carbon_t_ha.h5', key = 'soil_carbon_t_ha', mode = 'w', format = 'fixed', index = False, complevel = 9)




    ############### Calculate exclusion matrix -- x_mrj

    # Turn it into a pivot table. Non-NaN entries are permitted land-uses in the SA2.
    ut_ptable = ut.pivot_table(index = 'SA2_ID', columns = ['IRRIGATION', 'LU_DESC'], observed = False)['LU_ID']
    x_dry = concordance.merge(ut_ptable[0].reset_index(), on = 'SA2_ID', how = 'left')
    x_irr = concordance.merge(ut_ptable[1].reset_index(), on = 'SA2_ID', how = 'left')
    x_dry = x_dry.drop(columns = ['CELL_ID', 'SA2_ID'])
    x_irr = x_irr.drop(columns = ['CELL_ID', 'SA2_ID'])

    # Some land uses never occur at all (like dryland rice or grazing on irrigated natural land).
    for lu in ag_landuses:
        if lu not in x_dry.columns:
            x_dry[lu] = np.nan
        if lu not in x_irr.columns:
            x_irr[lu] = np.nan

    # 'Unallocated - modified land' can occur anywhere but is never irrigated.
    x_dry['Unallocated - modified land'] = 22
    x_irr['Unallocated - modified land'] = np.nan

    # 'Unallocated - natural land' is never irrigated. Occurs where transitions costs matrix (i.e., t_mrj) allows it.
    x_dry['Unallocated - natural land'] = 23
    x_irr['Unallocated - natural land'] = np.nan

    # Ensure lexicographical order.
    x_dry.sort_index(axis = 'columns', inplace = True)
    x_irr.sort_index(axis = 'columns', inplace = True)

    # Turn into 'boolean' arrays.
    x_dry = np.where( np.isnan(x_dry.to_numpy()), 0, 1 )
    x_irr = np.where( np.isnan(x_irr.to_numpy()), 0, 1 )

    # Get a list of cropping land-uses and return their indices in lexicographic land-use list.
    lu_crops = [ lu for lu in ag_landuses if 'Beef' not in lu
                                       and 'Sheep' not in lu
                                       and 'Dairy' not in lu
                                       and 'Unallocated' not in lu
                                       and 'Non-agricultural' not in lu ]
    clus = [ ag_landuses.index(lu) for lu in lu_crops ]

    # Allow dryland cropping only if precipitation is over 175mm in growing season.
    prec_over_175mm = bioph['AVG_GROW_SEAS_PREC_GE_175_MM_YR'].to_numpy()
    x_dry[:, clus] *= prec_over_175mm[:, np.newaxis]

    # Irrigated land-use is only allowed in potential irrigation areas.
    potential_irrigation_areas = zones['POTENTIAL_IRRIGATION_AREAS'].to_numpy()
    x_irr *= potential_irrigation_areas[:, np.newaxis]

    # Stack arrays.
    x_mrj = np.stack((x_dry, x_irr)).astype(bool)

    # Save to file
    np.save(outpath + 'x_mrj.npy', x_mrj)


    ############### Get water yield historical baseline data

    # Select historical (1970 - 2000) water yield under deep rooted and shallow rooted vegetation
    water_yield_baselines = bioph[['WATER_YIELD_HIST_SR_ML_HA', 'WATER_YIELD_HIST_DR_ML_HA', 'DEEP_ROOTED_PROPORTION']]

    # Save to file
    water_yield_baselines.to_hdf(outpath + 'water_yield_baselines.h5', key = 'water_yield_baselines', mode = 'w', format = 'fixed', index = False, complevel = 9)



    ############### Calculate baseline Pre-European water yield by river region and drainage division data

    # Calculate baseline Pre-European water yield
    zones['WATER_YIELD_HIST_BASELINE_ML'] = bioph['WATER_YIELD_HIST_BASELINE_ML_HA'] * zones['CELL_HA']

    # Create a LUT of river regions ID, name, and historical baseline water yield.
    rivreg_lut = zones.groupby(['HR_RIVREG_ID'], as_index = False, observed = True)\
                      .agg(HR_RIVREG_NAME = ('HR_RIVREG_NAME', 'first'),
                           WATER_YIELD_HIST_BASELINE_ML = ('WATER_YIELD_HIST_BASELINE_ML', 'sum'))

    # Save to HDF5 file.
    rivreg_lut.to_hdf(outpath + 'rivreg_lut.h5', key = 'rivreg_lut', mode = 'w', format = 'table', index = False, complevel = 9)

    # Save river region ID map to file
    zones['HR_RIVREG_ID'].to_hdf(outpath + 'rivreg_id.h5', key = 'rivreg_id', mode = 'w', format = 'fixed', index = False, complevel = 9)


    # Create a LUT of drainage division ID, name, and historical baseline water yield.
    draindiv_lut = zones.groupby(['HR_DRAINDIV_ID'], as_index = False, observed = True)\
        .agg(HR_DRAINDIV_NAME = ('HR_DRAINDIV_NAME', 'first'),
             WATER_YIELD_HIST_BASELINE_ML = ('WATER_YIELD_HIST_BASELINE_ML', 'sum'))


    draindiv_lut.to_hdf(outpath + 'draindiv_lut.h5', key = 'draindiv_lut', mode = 'w', format = 'table', index = False, complevel = 9)

    # Save drainage division ID map to file
    zones['HR_DRAINDIV_ID'].to_hdf(outpath + 'draindiv_id.h5', key = 'draindiv_id', mode = 'w', format = 'fixed', index = False, complevel = 9)



    # Calculate water yield for each drainage division and river region based on the given SSP/index
    def calculate_water_yield(ssp:str, idx:np.ndarray, water_zone:str):

        # Get the water and cell df based on the idx
        water_df = water_yield_baselines.iloc[idx]
        cell_attr = zones.iloc[idx]

        # Get the water yield data based on the SSP
        fn = 'Water_yield_GCM-Ensemble_ssp' + ssp + '_2010-2100_DR_ML_HA_mean'
        with h5py.File(luto_4D_inpath + fn + '.h5', 'r') as h5:
            dr_arr = h5[fn][:][:, idx]
        fn = 'Water_yield_GCM-Ensemble_ssp' + ssp + '_2010-2100_SR_ML_HA_mean'
        with h5py.File(luto_4D_inpath + fn + '.h5', 'r') as h5:
            sr_arr = h5[fn][:][:, idx]

        # Calculate the water yield for the cells outside the LUTO study area
        base_arr = dr_arr * np.array(water_df['DEEP_ROOTED_PROPORTION']) + sr_arr * np.array(1 - water_df['DEEP_ROOTED_PROPORTION'])
        base_arr = base_arr * cell_attr['CELL_HA'].values[np.newaxis,:]     # Convert from ML/HA to ML
        base_arr_df = pd.DataFrame(base_arr.T, columns = range(2010, 2101))

        # Warp the df to long format to link each cells with its river regions and drainage divisions
        base_arr_df[['HR_DRAINDIV_ID', 'HR_RIVREG_ID']] = cell_attr[['HR_DRAINDIV_ID', 'HR_RIVREG_ID']].values.tolist()
        base_arr_df = base_arr_df.melt(id_vars = ['HR_DRAINDIV_ID', 'HR_RIVREG_ID'], var_name = 'Year', value_name = 'Water_yield_ML')


        # Get the total water yield for each drainage division and river region
        water_yield_dd = base_arr_df.groupby(['HR_DRAINDIV_ID', 'Year'], as_index = False, observed = True).agg(Water_yield_ML = ('Water_yield_ML', 'sum'))
        water_yield_rr = base_arr_df.groupby(['HR_RIVREG_ID', 'Year'], as_index = False, observed = True).agg(Water_yield_ML = ('Water_yield_ML', 'sum'))
        water_yield_dd = water_yield_dd.rename(columns = {'HR_DRAINDIV_ID': 'Region_ID'})
        water_yield_rr = water_yield_rr.rename(columns = {'HR_RIVREG_ID': 'Region_ID'})

        # Append the region and ssp info
        water_yield_dd['ssp'] = ssp
        water_yield_rr['ssp'] = ssp
        water_yield_dd['water_zone'] = water_zone
        water_yield_rr['water_zone'] = water_zone
        water_yield_dd['Region_type'] = 'Drainage division'
        water_yield_rr['Region_type'] = 'River region'

        return pd.concat([water_yield_dd, water_yield_rr], ignore_index=True)

    # Run the function in parallel
    tasks = [
        delayed(calculate_water_yield)(ssp, idx, zone)
        for ssp in ['126', '245', '370', '585']
        for (idx, zone) in [
            (idx_out_LUTO, 'outside_LUTO_study_area'), 
            (lmap['CELL_ID'], 'all_AUS_lands')
        ]
    ]
    results = Parallel(n_jobs = len(tasks))(tasks)

    # Save to disk
    water_yield_df = pd.concat(results, ignore_index=True)

    water_yield_outside_LUTO = water_yield_df.query('water_zone == "outside_LUTO_study_area"')
    water_yield_outside_LUTO = water_yield_outside_LUTO.pivot( index=['Year'], columns=['Region_type', 'Region_ID','ssp'], values='Water_yield_ML')
    water_yield_outside_LUTO = water_yield_outside_LUTO.sort_index(axis=1, level=1)
    water_yield_outside_LUTO_dd = water_yield_outside_LUTO['Drainage division']
    water_yield_outside_LUTO_rr = water_yield_outside_LUTO['River region']
    water_yield_outside_LUTO_dd.to_hdf(os.path.join(outpath, 'water_yield_outside_LUTO_study_area_2010_2100_dd_ml.h5'), key='water_yield_outside_LUTO_study_area_2010_2100_dd_ml', mode='w', format='table', complevel=9)
    water_yield_outside_LUTO_rr.to_hdf(os.path.join(outpath, 'water_yield_outside_LUTO_study_area_2010_2100_rr_ml.h5'), key='water_yield_outside_LUTO_study_area_2010_2100_rr_ml', mode='w', format='table', complevel=9)

    water_yield_all_AUS = water_yield_df.query('water_zone == "all_AUS_lands"')
    water_yield_all_AUS = water_yield_all_AUS.pivot( index=['Year'], columns=['Region_type', 'Region_ID','ssp'], values='Water_yield_ML')
    water_yield_all_AUS = water_yield_all_AUS.sort_index(axis=1, level=1)
    water_yield_all_AUS_dd = water_yield_all_AUS['Drainage division']
    water_yield_all_AUS_rr = water_yield_all_AUS['River region']
    water_yield_all_AUS_dd.to_hdf(os.path.join(outpath, 'water_yield_natural_land_2010_2100_dd_ml.h5'), key='water_yield_natural_land_2010_2100_dd_ml', mode='w', format='table', complevel=9)
    water_yield_all_AUS_rr.to_hdf(os.path.join(outpath, 'water_yield_natural_land_2010_2100_rr_ml.h5'), key='water_yield_natural_land_2010_2100_rr_ml', mode='w', format='table', complevel=9)

    # Calculate water yield for each drainage division and river region based on historical baseline
    water_yield_hist_baseline_ml_ha = water_yield_baselines.eval(
    'WATER_YIELD_HIST_DR_ML_HA * DEEP_ROOTED_PROPORTION + WATER_YIELD_HIST_SR_ML_HA * (1 - DEEP_ROOTED_PROPORTION)'
    ).to_numpy(dtype = np.float32)
    
    water_yield_hist_baseline_ml = water_yield_hist_baseline_ml_ha * zones['CELL_HA'].to_numpy(dtype = np.float32)
    water_yield_hist_baseline_ml_outside_LUTO = water_yield_hist_baseline_ml[idx_out_LUTO]

    dd_id_outside_LUTO = zones[idx_out_LUTO]['HR_DRAINDIV_ID'].values
    rr_id_outside_LUTO = zones[idx_out_LUTO]['HR_RIVREG_ID'].values
    water_yield_outside_LUTO_hist_dd = dict(enumerate(np.bincount(dd_id_outside_LUTO, water_yield_hist_baseline_ml_outside_LUTO)))
    water_yield_outside_LUTO_hist_rr = dict(enumerate(np.bincount(rr_id_outside_LUTO, water_yield_hist_baseline_ml_outside_LUTO)))
    
    water_yield_outside_LUTO_hist_dd_df = pd.DataFrame(water_yield_outside_LUTO_hist_dd, index=['Water Yield (ML)']).T.reset_index(names='Region_ID')
    water_yield_outside_LUTO_hist_rr_df = pd.DataFrame(water_yield_outside_LUTO_hist_rr, index=['Water Yield (ML)']).T.reset_index(names='Region_ID')
    water_yield_outside_LUTO_hist_rr_df.insert(0, 'Region_Type', 'River Region')
    water_yield_outside_LUTO_hist_dd_df.insert(0, 'Region_Type', 'Drainage Division')
    
    water_yield_outside_LUTO_hist_df = pd.concat([water_yield_outside_LUTO_hist_dd_df, water_yield_outside_LUTO_hist_rr_df], ignore_index=True)
    water_yield_outside_LUTO_hist_df = water_yield_outside_LUTO_hist_df.query('Region_ID != 0')
    water_yield_outside_LUTO_hist_df.to_hdf(
        os.path.join(outpath, 'water_yield_outside_LUTO_study_area_hist_1970_2000.h5'), 
        key='water_yield_outside_LUTO_study_area_hist_1970_2000', 
        mode='w', 
        format='table', 
        complevel=9
    )



    ############### Get biodiversity priority layers

    # Biodiversity priorities under the four SSPs
    biodiv_priorities = bioph[[
        'BIODIV_PRIORITY_SSP126',
        'BIODIV_PRIORITY_SSP245',
        'BIODIV_PRIORITY_SSP370',
        'BIODIV_PRIORITY_SSP585',
        'NATURAL_AREA_CONNECTIVITY',
        'DCCEEW_NCI']].copy()

    # Save to file
    biodiv_priorities.to_hdf(outpath + 'biodiv_priorities.h5', key = 'biodiv_priorities', mode = 'w', format = 'fixed', index = False, complevel = 9)




    ############### Get stream length

    # Get stream lenght per cell
    stream_length_m_cell = bioph['RIP_LENGTH_M_CELL'].copy()

    # Save to file
    stream_length_m_cell.to_hdf(outpath + 'stream_length_m_cell.h5', key = 'stream_length_m_cell', mode = 'w', format = 'fixed', index = False, complevel = 9)





    ############### Forest and reforestation data

    # # Carbon stock in mature forest on natural land and save to file
    # s = pd.DataFrame(columns=['REMNANT_VEG_T_CO2_HA'])
    # s['REMNANT_VEG_T_CO2_HA'] = bioph['REMNANT_VEG_T_CO2_HA']
    # s.to_hdf(outpath + 'natural_land_t_co2_ha.h5', key = 'natural_land_t_co2_ha', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    s = pd.DataFrame(columns=['NATURAL_LAND_AGB_TCO2_HA', 'NATURAL_LAND_AGB_DEBRIS_TCO2_HA', 'NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA'])
    s['NATURAL_LAND_AGB_TCO2_HA'] = bioph['NATURAL_LAND_AGB_TCO2_HA']
    s['NATURAL_LAND_AGB_DEBRIS_TCO2_HA'] = bioph['NATURAL_LAND_AGB_DEBRIS_TCO2_HA']
    s['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA'] = bioph['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA']
    s.to_hdf(outpath + 'natural_land_t_co2_ha.h5', key = 'natural_land_t_co2_ha', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Average annual carbon sequestration by Environmental Plantings (block plantings) and save to file
    s = pd.DataFrame(columns=['EP_BLOCK_AG_AVG_T_CO2_HA_YR', 'EP_BLOCK_BG_AVG_T_CO2_HA_YR'])
    s['EP_BLOCK_AG_AVG_T_CO2_HA_YR'] = bioph.eval('EP_BLOCK_TREES_AVG_T_CO2_HA_YR + EP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR')
    s['EP_BLOCK_BG_AVG_T_CO2_HA_YR'] = bioph['EP_BLOCK_SOIL_AVG_T_CO2_HA_YR']
    s.to_hdf(outpath + 'ep_block_avg_t_co2_ha_yr.h5', key = 'ep_block_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Average annual carbon sequestration by Environmental Plantings (belt plantings) and save to file
    s = pd.DataFrame(columns=['EP_BELT_AG_AVG_T_CO2_HA_YR', 'EP_BELT_BG_AVG_T_CO2_HA_YR'])
    s['EP_BELT_AG_AVG_T_CO2_HA_YR'] = bioph.eval('EP_BELT_TREES_AVG_T_CO2_HA_YR + EP_BELT_DEBRIS_AVG_T_CO2_HA_YR')
    s['EP_BELT_BG_AVG_T_CO2_HA_YR'] = bioph['EP_BELT_SOIL_AVG_T_CO2_HA_YR']
    s.to_hdf(outpath + 'ep_belt_avg_t_co2_ha_yr.h5', key = 'ep_belt_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Average annual carbon sequestration by Environmental Plantings (riparian plantings) and save to file
    s = pd.DataFrame(columns=['EP_RIP_AG_AVG_T_CO2_HA_YR', 'EP_RIP_BG_AVG_T_CO2_HA_YR'])
    s['EP_RIP_AG_AVG_T_CO2_HA_YR'] = bioph.eval('EP_RIP_TREES_AVG_T_CO2_HA_YR + EP_RIP_DEBRIS_AVG_T_CO2_HA_YR')
    s['EP_RIP_BG_AVG_T_CO2_HA_YR'] = bioph['EP_RIP_SOIL_AVG_T_CO2_HA_YR']
    s.to_hdf(outpath + 'ep_rip_avg_t_co2_ha_yr.h5', key = 'ep_rip_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)


    # Average annual carbon sequestration by Hardwood Plantings (block plantings) and save to file
    s = pd.DataFrame(columns=['CP_BLOCK_AG_AVG_T_CO2_HA_YR', 'CP_BLOCK_BG_AVG_T_CO2_HA_YR'])
    s['CP_BLOCK_AG_AVG_T_CO2_HA_YR'] = bioph.eval('CP_BLOCK_TREES_AVG_T_CO2_HA_YR + CP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR')
    s['CP_BLOCK_BG_AVG_T_CO2_HA_YR'] = bioph['CP_BLOCK_SOIL_AVG_T_CO2_HA_YR']
    s.to_hdf(outpath + 'cp_block_avg_t_co2_ha_yr.h5', key = 'cp_block_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Average annual carbon sequestration by Hardwood Plantings (belt plantings) and save to file
    s = pd.DataFrame(columns=['CP_BELT_AG_AVG_T_CO2_HA_YR', 'CP_BELT_BG_AVG_T_CO2_HA_YR'])
    s['CP_BELT_AG_AVG_T_CO2_HA_YR'] = bioph.eval('CP_BELT_TREES_AVG_T_CO2_HA_YR + CP_BELT_DEBRIS_AVG_T_CO2_HA_YR')
    s['CP_BELT_BG_AVG_T_CO2_HA_YR'] = bioph['CP_BELT_SOIL_AVG_T_CO2_HA_YR']
    s.to_hdf(outpath + 'cp_belt_avg_t_co2_ha_yr.h5', key = 'cp_belt_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    # Average annual carbon sequestration by Human Induced Regrowth (block plantings) and save to file
    s = pd.DataFrame(columns=['HIR_BLOCK_AG_AVG_T_CO2_HA_YR', 'HIR_BLOCK_BG_AVG_T_CO2_HA_YR'])
    s['HIR_BLOCK_AG_AVG_T_CO2_HA_YR'] = bioph.eval('HIR_BLOCK_TREES_AVG_T_CO2_HA_YR + HIR_BLOCK_DEBRIS_AVG_T_CO2_HA_YR')
    s['HIR_BLOCK_BG_AVG_T_CO2_HA_YR'] = bioph['HIR_BLOCK_SOIL_AVG_T_CO2_HA_YR']
    s.to_hdf(outpath + 'hir_block_avg_t_co2_ha_yr.h5', key = 'hir_block_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)
    

    # MASK for Human Induced Regrowth (riparian plantings) and save to file
    hir_mask = bioph['AVG_AN_PREC_MM_YR'] <= 300
    np.save(outpath + 'hir_mask.npy', hir_mask.values)  # shape: (6956407,)

    # Fire risk low, medium, and high and save to file
    s = bioph[['FD_RISK_PERC_5TH', 'FD_RISK_MEDIAN', 'FD_RISK_PERC_95TH']].copy()
    s.to_hdf(outpath + 'fire_risk.h5', key = 'fire_risk', mode = 'w', format = 'fixed', index = False, complevel = 9)


    # Adjust the establishment costs using CPI; Source https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/latest-release
    # The original data is in 2021 AUD (CPI 118.8 in Jun-2021), need to convert to 2010 AUD (CPI 99.2 in Jun-2011)
    bioph['EP_EST_COST_HA_CPI_ADJ'] = bioph['EP_EST_COST_HA'] * 99.2 / 118.8
    bioph['CP_EST_COST_HA_CPI_ADJ'] = bioph['CP_EST_COST_HA'] * 99.2 / 118.8

    # Average establishment costs for Environmental Plantings ($/ha) and save to file
    bioph['EP_EST_COST_HA_CPI_ADJ'].to_hdf(outpath + 'ep_est_cost_ha.h5', key = 'ep_est_cost_ha', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Average establishment costs for Carbon Plantings ($/ha) and save to file
    bioph['CP_EST_COST_HA_CPI_ADJ'].to_hdf(outpath + 'cp_est_cost_ha.h5', key = 'cp_est_cost_ha', mode = 'w', format = 'fixed', index = False, complevel = 9)





    ############### Calculate climate impacts - includes data with and without CO2 fertilisation

    CO2F_dict = {'CO2_FERT_OFF': 0 , 'CO2_FERT_ON':1 }

    # Distill list of RCP labels from dataset
    rcps = sorted(list({col[1] for col in cci_raw.columns})) # curly brackets remove duplicates

    # Create luid_desc -- old LU_ID to LU_DESC concordance table
    luid_desc = lmap.groupby('LU_ID').first()['LU_DESC'].astype(str)


    def process_climate_impacts(co2_on_off, rcp, cci_raw, concordance, luid_desc, outpath):

        co2_on_off_idx = CO2F_dict[co2_on_off]

        # Slice off RCP data and get rid of MultiIndex.
        cci_ptable = cci_raw[co2_on_off_idx][rcp].reset_index()

        # Merge with SA2-Cell concordance to obtain cell-based table.
        cci = concordance.merge(cci_ptable, on = 'SA2_ID', how = 'left')

        # Not all columns are needed, drop unwanted cols.
        cci = cci.drop(['SA2_ID'], axis = 1)

        # Convert columns to integer (need 64-bit integer which can accomodate NaNs)
        cci['IRRIGATION'] = cci['IRRIGATION'].astype('Int64')
        cci['LU_ID'] = cci['LU_ID'].astype('Int64')

        # Create the MultiIndex structure
        cci = cci.pivot( index = 'CELL_ID',
                        columns = ['IRRIGATION', 'LU_ID'],
                        values = ['YR_2020', 'YR_2050', 'YR_2080']
                    ).dropna( axis = 'columns', how = 'all')

        # Name the YEAR level and reorder levels
        cci.columns.set_names('YEAR', level = 0, inplace = True)
        cci = cci.reorder_levels( ['IRRIGATION', 'LU_ID', 'YEAR'], axis = 1 )

        # Convert land management/use column names to strings and years to integers.
        lmid_desc = {0: 'dry', 1: 'irr'}
        coltups = [ (lmid_desc[col[0]], luid_desc[col[1]], int(col[2][3:]))
                    for col in cci.columns ]
        cci.columns = pd.MultiIndex.from_tuples(coltups)

        # Sort land use in lexicographical order
        cci.sort_index(axis = 1, inplace = True)

        # Write to HDF5 file.
        fname = outpath + 'climate_change_impacts_' + rcp + '_' + co2_on_off + '.h5'
        kname = 'climate_change_impacts_' + rcp
        cci.to_hdf(fname, key = kname, mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Parallelise the process
    jobs = ([delayed(process_climate_impacts)(on_off, rcp, cci_raw, concordance, luid_desc, outpath ) for on_off in CO2F_dict for rcp in rcps])
    Parallel(n_jobs=min(50, len(CO2F_dict) * len(rcps)), prefer='processes')(jobs) # Reduce processing time from ~25 min to ~3 min





    ############### Agricultural economics - crops

    # Produce a multi-indexed version of the crops data.
    # crops_ptable = crops.pivot_table(index = 'SA2_ID', columns = ['Irrigation', 'LU_DESC'])
    crops_ptable = crops.drop(['LU_DESC', 'Irrigation', 'Area', 'Prod'], axis = 1)

    # Merge to create a cell-based table.
    agec_crops = concordance.merge( crops_ptable, on = 'SA2_ID', how = 'left' )

    # Drop unnecessary columns.
    agec_crops = agec_crops.drop(['SA2_ID'], axis = 1)

    # Convert columns to integer (need 64-bit integer which can accomodate NaNs)
    agec_crops['IRRIGATION'] = agec_crops['IRRIGATION'].astype('Int64')
    agec_crops['LU_ID'] = agec_crops['LU_ID'].astype('Int64')

    # Create the MultiIndex structure
    agec_crops = agec_crops.pivot( index = 'CELL_ID',
                                   columns = ['IRRIGATION', 'LU_ID'],
                                   values = ['Yield', 'P1', 'AC', 'QC', 'FDC', 'FLC', 'FOC', 'WR', 'WP']
                                 ).dropna( axis = 'columns', how = 'all')

    # The merge flattens the multi-index to tuples, so unflatten back to multi-index
    lms = ['dry', 'irr']
    ts = [(col[0], lms[col[1]], luid_desc[col[2]]) for col in agec_crops.columns]
    agec_crops.columns = pd.MultiIndex.from_tuples(ts)

    # Sort land use in lexicographical order
    agec_crops.sort_index(axis = 1, inplace = True)

    # Save to HDF5
    agec_crops.to_hdf(outpath + 'agec_crops.h5', key = 'agec_crops', mode = 'w', format = 'fixed', index = False, complevel = 9)




    ############### Agricultural economics - livestock

    # Get only the livestock economics columns (i.e., those that vary with land-use).
    lvstk_cols = [c for c in lvstk.columns if any(x in c for x in ['BEEF', 'SHEEP', 'DAIRY'])]
    agec_lvstk = lvstk[lvstk_cols]

    # Prepare columns for multi-indexing (i.e. turn into a list of tuples).
    # cols = agec_lvstk.columns.to_list()
    cols = [tuple(c.split(sep = '_')) for c in lvstk_cols]
    cols = [(c[0] + '_' + c[1], c[2]) if c[0] == 'WR' else c for c in cols]

    # Make and set the multi-index.
    agec_lvstk.columns = pd.MultiIndex.from_tuples(cols)

    # Save to HDF5
    agec_lvstk.to_hdf(outpath + 'agec_lvstk.h5', key = 'agec_lvstk', mode = 'w', format = 'fixed', index = False, complevel = 9)




    ############### Agricultural Greenhouse Gas Emissions - crops

    # Produce a multi-indexed version of the crops data.
    cropsGHG_ptable = cropsGHG.drop('LU_DESC', axis = 1)

    # Merge to create a cell-based table.
    agGHG_crops = concordance.merge( cropsGHG_ptable, on = 'SA2_ID', how = 'left' )

    # Drop unnecessary columns and fill NaNs with zeros.
    agGHG_crops = agGHG_crops.drop(['SA2_ID'], axis = 1)

    # Convert columns to integer (need 64-bit integer which can accomodate NaNs)
    agGHG_crops['IRRIGATION'] = agGHG_crops['IRRIGATION'].astype('Int64')
    agGHG_crops['LU_ID'] = agGHG_crops['LU_ID'].astype('Int64')

    # Create the MultiIndex structure
    GHG_sources_crops = ['CO2E_KG_HA_FERT_PROD', 'CO2E_KG_HA_PEST_PROD', 'CO2E_KG_HA_IRRIG', 'CO2E_KG_HA_CHEM_APPL', 'CO2E_KG_HA_CROP_MGT', 'CO2E_KG_HA_CULTIV', 'CO2E_KG_HA_HARVEST', 'CO2E_KG_HA_SOWING', 'CO2E_KG_HA_SOIL']
    agGHG_crops = agGHG_crops.pivot( index = 'CELL_ID',
                                     columns = ['IRRIGATION', 'LU_ID'],
                                     values = GHG_sources_crops
                                   ).dropna( axis = 'columns', how = 'all' )

    # The merge flattens the multi-index to tuples, so unflatten back to multi-index
    ts = [(col[0], lms[col[1]], luid_desc[col[2]]) for col in agGHG_crops.columns]
    agGHG_crops.columns = pd.MultiIndex.from_tuples(ts)

    # Sort land use in lexicographical order
    agGHG_crops.sort_index(axis = 1, inplace = True)

    # Save to HDF5
    agGHG_crops.to_hdf(outpath + 'agGHG_crops.h5', key = 'agGHG_crops', mode = 'w', format = 'fixed', index = False, complevel = 9)



    # TEMPORARY CODE to Check total crop GHG emissions

    agGHGmap = lmap[['CELL_ID', 'SA2_ID', 'LU_ID', 'IRRIGATION']].merge( cropsGHG_ptable,
                                                                         on = ['SA2_ID', 'LU_ID', 'IRRIGATION'],
                                                                         how = 'left' )

    agGHGmap['REAL_AREA'] = pd.read_hdf('input/real_area.h5')

    agGHGmap = agGHGmap.query('5 <= LU_ID <= 25')
    print('Number of NaNs =', agGHGmap[agGHGmap.isna().any(axis=1)].shape[0])


    agGHGmap.eval('TCO2E_CELL_FERT_PROD = CO2E_KG_HA_FERT_PROD * REAL_AREA / 1000', inplace = True)
    agGHGmap.eval('TCO2E_CELL_PEST_PROD = CO2E_KG_HA_PEST_PROD * REAL_AREA / 1000', inplace = True)
    agGHGmap.eval('TCO2E_CELL_IRRIG = CO2E_KG_HA_IRRIG * REAL_AREA / 1000', inplace = True)
    agGHGmap.eval('TCO2E_CELL_CHEM_APPL = CO2E_KG_HA_CHEM_APPL * REAL_AREA / 1000', inplace = True)
    agGHGmap.eval('TCO2E_CELL_CROP_MGT = CO2E_KG_HA_CROP_MGT * REAL_AREA / 1000', inplace = True)
    agGHGmap.eval('TCO2E_CELL_CULTIV = CO2E_KG_HA_CULTIV * REAL_AREA / 1000', inplace = True)
    agGHGmap.eval('TCO2E_CELL_HARVEST = CO2E_KG_HA_HARVEST * REAL_AREA / 1000', inplace = True)
    agGHGmap.eval('TCO2E_CELL_SOWING = CO2E_KG_HA_SOWING * REAL_AREA / 1000', inplace = True)
    agGHGmap.eval('TCO2E_CELL_SOIL = CO2E_KG_HA_SOIL * REAL_AREA / 1000', inplace = True)

    agGHGmap.eval('Total_GHG = (TCO2E_CELL_FERT_PROD + \
                                TCO2E_CELL_PEST_PROD + \
                                TCO2E_CELL_IRRIG + \
                                TCO2E_CELL_CHEM_APPL + \
                                TCO2E_CELL_CROP_MGT + \
                                TCO2E_CELL_CULTIV + \
                                TCO2E_CELL_HARVEST + \
                                TCO2E_CELL_SOWING + \
                                TCO2E_CELL_SOIL)', inplace = True)

    print(agGHGmap[['TCO2E_CELL_FERT_PROD',
                   'TCO2E_CELL_PEST_PROD',
                   'TCO2E_CELL_IRRIG',
                   'TCO2E_CELL_CHEM_APPL',
                   'TCO2E_CELL_CROP_MGT',
                   'TCO2E_CELL_CULTIV',
                   'TCO2E_CELL_HARVEST',
                   'TCO2E_CELL_SOWING',
                   'TCO2E_CELL_SOIL']].sum())

    agGHGmap['Total_GHG'].sum()



    ############### Agricultural Greenhouse Gas Emissions - livestock

    # Merge to create a cell-based table. Added future_stack = True to silence warning: The previous implementation of stack is deprecated and will be removed in a future version of pandas. See the What's New notes for pandas 2.1.0 for details. Specify future_stack=True to adopt the new implementation and silence this warning.
    agGHG_lvstk = concordance.merge( lvstkGHG.stack(level = 0, future_stack = True).reset_index()
                                   , on = 'SA2_ID'
                                   , how = 'left' )
    # Drop unnecessary columns.
    agGHG_lvstk = agGHG_lvstk.drop(['SA2_ID'], axis = 1)

    # Create the MultiIndex structure
    agGHG_lvstk = agGHG_lvstk.pivot( index = 'CELL_ID',
                                     columns = ['Livestock type'],
                                     values = ['CO2E_KG_HEAD_DUNG_URINE', 'CO2E_KG_HEAD_ELEC', 'CO2E_KG_HEAD_ENTERIC', 'CO2E_KG_HEAD_FODDER', 'CO2E_KG_HEAD_FUEL', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'CO2E_KG_HEAD_MANURE_MGT', 'CO2E_KG_HEAD_SEED']
                                   ).dropna( axis = 'columns', how = 'all')

    # Change the level names
    agGHG_lvstk.columns.set_names(['INDICATOR', 'LIVESTOCK'], level = [0, 1], inplace = True)

    # Swap column levels
    agGHG_lvstk = agGHG_lvstk.reorder_levels([1, 0], axis = 1)

    # Sort land use in lexicographical order
    agGHG_lvstk.sort_index(axis = 1, inplace = True)

    # Save to HDF5
    agGHG_lvstk.to_hdf(outpath + 'agGHG_lvstk.h5', key = 'agGHG_lvstk', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Copy over irrigated pasture emissions
    agGHG_irrpast = concordance.merge( irrpastGHG, on = 'SA2_ID', how = 'left' )

    # Save to HDF5
    agGHG_irrpast.to_hdf(outpath + 'agGHG_irrpast.h5', key = 'agGHG_irrpast', mode = 'w', format = 'fixed', index = False, complevel = 9)


    # CODE to Check total livestock GHG emissions - see N:\Data-Master\LUTO_2.0_input_data\Scripts\2_assemble_agricultural_data.py


    ############### Agricultural Greenhouse Gas Emissions - livestock - off-land

    # Read in raw data
    agGHG_lvstk_off_land_eggs = pd.read_csv(f"{GHG_off_land_inpath}/GLEAM3Results2024-02-16_Eggs_ANZ_AR6.csv")
    agGHG_lvstk_off_land_meat = pd.read_csv(f"{GHG_off_land_inpath}/GLEAM3Results2024-02-16_Meat_ANZ_AR6.csv")

    # Rename the HerdType column to math the LUTO convention
    agGHG_lvstk_off_land_eggs['COMMODITY'] = agGHG_lvstk_off_land_eggs['HerdType'].replace({'Chickens': 'eggs'})
    agGHG_lvstk_off_land_meat['COMMODITY'] = agGHG_lvstk_off_land_meat['HerdType'].replace({'Pigs': 'pork',
                                                                                           'Chickens': 'chicken'})

    # Create df for aquaculture, fill 0 for the GHG values
    agGHG_lvstk_off_land_aquaculture = agGHG_lvstk_off_land_eggs.copy()
    agGHG_lvstk_off_land_aquaculture['Emission Source'] = agGHG_lvstk_off_land_eggs['Emission Source']
    agGHG_lvstk_off_land_aquaculture['Animal'] = 'Aquaculture'
    agGHG_lvstk_off_land_aquaculture['HerdType'] = 'Aquaculture'
    agGHG_lvstk_off_land_aquaculture['COMMODITY'] = 'aquaculture'
    agGHG_lvstk_off_land_aquaculture['Emissions [ t CO2eq ]'] = 0
    agGHG_lvstk_off_land_aquaculture['Emission Intensity [ kg CO2eq / kg ]'] = 0
    agGHG_lvstk_off_land_aquaculture['Production [ t ]'] = None



    agGHG_lvstk_off_land = pd.concat([agGHG_lvstk_off_land_eggs,
                                      agGHG_lvstk_off_land_meat,
                                      agGHG_lvstk_off_land_aquaculture], axis = 0)

    # Define the GHG emissions that need to be considered by LUTO
    #       {Feed}, {Land Use Change}, and {Post Farm} have already
    #       being included in LUTO. So there is no need to include them here.
    exclude_GHG = ['Feed (CO2)',
                   'Feed (N2O)',
                   'Feed (CH4)',
                   'LUC: soy and palm (CO2)',
                   'LUC: pasture expansion (CO2)',
                   'Post-farm (CO2)',]

    # Query the GHG emissions that need to be included in LUTO
    agGHG_lvstk_off_land = agGHG_lvstk_off_land.query("`Emission Source` not in @exclude_GHG")

    # Filter only the off-land commodities
    agGHG_lvstk_off_land = agGHG_lvstk_off_land.query("COMMODITY in ['pork', 'chicken', 'eggs', 'aquaculture']")

    # Save to the input data folder
    agGHG_lvstk_off_land.to_csv(outpath + 'agGHG_lvstk_off_land.csv', index = False)



    ############### Agricultural demand

    # Convert NaNs to zeros
    demand.fillna(0, inplace = True)

    # Rename columns
    demand = demand.rename(columns = {'Scenario': 'SCENARIO',
                                      'Domestic_diet': 'DIET_DOM',
                                      'Global_diet': 'DIET_GLOB',
                                      'Convergence': 'CONVERGENCE',
                                      'Imports': 'IMPORT_TREND',
                                      'Waste': 'WASTE',
                                      'Feed': 'FEED_EFFICIENCY',
                                      'SPREAD_Commodity': 'COMMODITY',
                                      'Year': 'YEAR',
                                      'domestic': 'DOMESTIC',
                                      'exports': 'EXPORTS',
                                      'imports': 'IMPORTS',
                                      'feed': 'FEED',
                                      'All_demand': 'PRODUCTION'})

    # Create the MultiIndex structure
    demand = demand.pivot( index = ['SCENARIO', 'DIET_DOM', 'DIET_GLOB', 'CONVERGENCE', 'IMPORT_TREND', 'WASTE', 'FEED_EFFICIENCY', 'COMMODITY'],
                         columns = ['YEAR'],
                          values = ['DOMESTIC', 'EXPORTS', 'IMPORTS', 'FEED', 'PRODUCTION'])

    # Save to HDF5
    demand.to_hdf(outpath + 'demand_projections.h5', key = 'demand_projections', mode = 'w', format = 'fixed', index = False, complevel = 9)



    ############## BECCS data
    # from Lei Gao at CSIRO (N:\Data-Master\BECCS\From_CSIRO\20211124_as_submitted)

    # Get latitude, longitude coordinates for study area
    cell_xy = zones[['CELL_ID', 'X', 'Y', 'CELL_HA']].copy()

    # Round off XY coordinates to cell_df to join BECCS data
    cell_xy[['X', 'Y']] = round(cell_xy[['X', 'Y']] * 100).astype('int64')

    # Round off XY coordinates to cell_BECCS_raw to join BECCS data
    BECCS_raw[['X', 'Y']] = round(BECCS_raw[['Longitude', 'Latitude']] * 100).astype('int64')

    # Grab REAL_AREA
    BECCS_raw = BECCS_raw.merge(cell_xy.iloc[:, -3:], how = 'left', on = ['X', 'Y'])

    # Create conversion factor to convert costs and revenue to equal annual equivalent terms (dx.doi.org/10.1016/j.landusepol.2009.09.012)
    r, t = 0.05, 30
    annualiser = (r * (1 + r) ** t) / ((1 + r) ** t - 1)

    # Calculate cost of biomass production for BECCS (not including opportunity costs, "N:\Data-Master\BECCS\From_CSIRO\20211124_as_submitted\LUF_BECCS_Methods_20211116.docx")
    BECCS_raw['BECCS_COSTS_AUD_HA_YR'] = ( ( BECCS_raw['Biomass_production_cost_no_oppotunity (2010AUD)'] +
                                      BECCS_raw['Biomass_transportation_cost (2010AUD)'] +
                                      BECCS_raw['Biomass_processing_cost (2010AUD)'] +
                                      BECCS_raw['CO2_transportation_cost (2010AUD)'] +
                                      BECCS_raw['CO2_storage_cost (2010AUD)'] )     # sum costs in NPV $2010 terms over 30 years
                                      / BECCS_raw['CELL_HA']                        # convert to per hectare values
                                  ) * annualiser                             # convert to equal annual equivalent values

    # Calculate revenue from biomass production for BECCS assuming a 50% profit share with the processing plant
    BECCS_raw['BECCS_REV_AUD_HA_YR'] = (0.5 * BECCS_raw['Revenue_electricity (2010AUD)'] / BECCS_raw['CELL_HA']) * annualiser

    # Calculate average annual net CO2 removal capacity per hectare from biomass production for BECCS over the 30 year lifespan
    BECCS_raw['BECCS_TCO2E_HA_YR'] = (BECCS_raw['Net_carbon_removal (tCO2e)'] / BECCS_raw['CELL_HA']) / 30

    # Calculate average annual amount of renewable energy produced per hectare over the 30 year lifespan
    BECCS_raw['BECCS_MWH_HA_YR'] = (BECCS_raw['Electricity_produced (Mwh)'] / BECCS_raw['CELL_HA']) / 30

    # Downsample to float32
    cols = ['BECCS_COSTS_AUD_HA_YR', 'BECCS_REV_AUD_HA_YR', 'BECCS_TCO2E_HA_YR', 'BECCS_MWH_HA_YR']
    BECCS_raw[cols] = BECCS_raw[cols].astype('float32')

    # Merge table to tmp dataframe to get values for all cells
    BECCS_raw = BECCS_raw.drop(columns = 'CELL_HA')
    cell_xy = cell_xy.merge(BECCS_raw.iloc[:, -6:], how = 'left', on = ['X', 'Y'])
    cell_xy = cell_xy.drop(columns = ['X', 'Y', 'CELL_HA'])

    # Save to HDF5 file
    cell_xy.to_hdf(outpath + 'cell_BECCS_df.h5', key = 'cell_BECCS_df', mode = 'w', format = 'fixed', index = False, complevel = 9)


    # Complete processing and report back
    laps_time = round(time.time() - start_time)
    print('Completed input data refresh at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ', taking', laps_time, 'seconds')



