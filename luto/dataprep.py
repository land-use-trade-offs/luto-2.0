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
Script to load and prepare input data based on build.ipynb by F. de Haan
and Brett Bryan, Deakin University

"""


# Load libraries
import csv
import numpy as np
import pandas as pd
import shutil, os, time
from luto.settings import INPUT_DIR, RAW_DATA

    
def create_new_dataset():
    """Creates a new LUTO input dataset from source data"""
    
    # Set up a timer and print the time
    start_time = time.time()
    print('Beginning input data refresh at', time.strftime("%H:%M:%S", time.localtime()) + '...')
    
    
    ############### Copy key input data layers from their source folders to the raw_data folder for processing
    
    # Set data input paths
    luto_1D_inpath = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/1D_Parameter_Timeseries/'
    luto_2D_inpath = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/'
    luto_3D_inpath = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/3D_Spatial_Timeseries/'
    luto_4D_inpath = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/'
    fdh_inpath = 'N:/LUF-Modelling/fdh-archive/data/neoluto-data/new-data-and-domain/'
    profit_map_inpath = 'N:/Data-Master/Profit_map/'
    nlum_inpath = 'N:/Data-Master/National_Landuse_Map/'
    
    # Set data output paths
    raw_data = RAW_DATA + '/' # '../raw_data/'
    outpath = INPUT_DIR + '/'
    
    # Delete the data folders' contents
    for file in os.scandir(outpath): 
        if file.name != '.gitignore':
            os.remove(file.path)
    for file in os.scandir(raw_data): os.remove(file.path)
    
    # Copy in the raw data files from their source
    shutil.copyfile(fdh_inpath + 'tmatrix-cat2lus.csv', raw_data + 'tmatrix_cat2lus.csv')
    shutil.copyfile(fdh_inpath + 'transitions_costs_20230816.xlsx', raw_data + 'transitions_costs_20230816.xlsx')    

    shutil.copyfile(profit_map_inpath + 'NLUM_SPREAD_LU_ID_Mapped_Concordance.h5', raw_data + 'NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')
    # shutil.copyfile(profit_map_inpath + 'cell_ag_data.h5', raw_data + 'cell_ag_data.h5')

    shutil.copyfile(luto_2D_inpath + 'cell_LU_mapping.h5', raw_data + 'cell_LU_mapping.h5')
    shutil.copyfile(luto_2D_inpath + 'cell_zones_df.h5', raw_data + 'cell_zones_df.h5')
    shutil.copyfile(luto_2D_inpath + 'cell_livestock_data.h5', raw_data + 'cell_livestock_data.h5')
    shutil.copyfile(luto_2D_inpath + 'SA2_livestock_GHG_data.h5', raw_data + 'SA2_livestock_GHG_data.h5')
    shutil.copyfile(luto_2D_inpath + 'SA2_crop_data.h5', raw_data + 'SA2_crop_data.h5')
    shutil.copyfile(luto_2D_inpath + 'SA2_crop_GHG_data.h5', raw_data + 'SA2_crop_GHG_data.h5')
    shutil.copyfile(luto_2D_inpath + 'cell_biophysical_df.h5', raw_data + 'cell_biophysical_df.h5')
    shutil.copyfile(luto_2D_inpath + 'SA2_climate_damage_mult.h5', raw_data + 'SA2_climate_damage_mult.h5')
    
    # Copy data straight to LUTO input folder, no processing required
    shutil.copyfile(fdh_inpath + 'yieldincreases-bau2022.csv', outpath + 'yieldincreases_bau2022.csv')
    shutil.copyfile(nlum_inpath + 'NLUM_2010-11_mask.tif', outpath + 'NLUM_2010-11_mask.tif')
    
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp126_2010-2100_DR_ML_HA_mean.h5', outpath + 'water_yield_ssp126_2010-2100_dr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp126_2010-2100_SR_ML_HA_mean.h5', outpath + 'water_yield_ssp126_2010-2100_sr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp245_2010-2100_DR_ML_HA_mean.h5', outpath + 'water_yield_ssp245_2010-2100_dr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp245_2010-2100_SR_ML_HA_mean.h5', outpath + 'water_yield_ssp245_2010-2100_sr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp370_2010-2100_DR_ML_HA_mean.h5', outpath + 'water_yield_ssp370_2010-2100_dr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp370_2010-2100_SR_ML_HA_mean.h5', outpath + 'water_yield_ssp370_2010-2100_sr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp585_2010-2100_DR_ML_HA_mean.h5', outpath + 'water_yield_ssp585_2010-2100_dr_ml_ha.h5')
    shutil.copyfile(luto_4D_inpath + 'Water_yield_GCM-Ensemble_ssp585_2010-2100_SR_ML_HA_mean.h5', outpath + 'water_yield_ssp585_2010-2100_sr_ml_ha.h5')

    # Load delta demands file
    shutil.copyfile(luto_1D_inpath + 'demand_deltas_c.npy', outpath + 'demand_deltas_c.npy')
    
    
    
    ############### Read data
    
    # Read cell_LU_mapping.h5 
    lmap = pd.read_hdf(raw_data + 'cell_LU_mapping.h5')
    
    # Read in the cell_zones_df dataframe
    zones = pd.read_hdf(raw_data + 'cell_zones_df.h5')
    
    # Read in from-to costs in category-to-category format
    # tmcat = pd.read_csv(raw_data + 'tmatrix_categories.csv', index_col = 0)
    tmcat = pd.read_excel( raw_data + 'transitions_costs_20230816.xlsx'
                          , sheet_name = 'Current'
                          , usecols = 'B:M'
                          , skiprows = 5
                          , nrows = 11
                          , index_col = 0)

    # Read transition costs from agricultural land to environmental plantings
    ag_to_new_land_uses = pd.read_excel( raw_data + 'transitions_costs_20230816.xlsx'
                                       , sheet_name = 'Ag_to_new_land-uses'
                                       , usecols = 'B,C'
                                       , index_col = 0 )

    # Read the categories to land-uses concordance
    cat2lus = pd.read_csv(raw_data + 'tmatrix_cat2lus.csv').to_numpy()
    
    # Read livestock data
    lvstk = pd.read_hdf(raw_data + 'cell_livestock_data.h5')
    
    # Read livestock GHG emissions data
    lvstkGHG = pd.read_hdf(raw_data + 'SA2_livestock_GHG_data.h5')
    
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
    
    
    
    
    ############### Create the CELL_ID to SA2_ID concordance table
    
    # Make sure the CELL_ID starts at zero, rather than one
    lmap['CELL_ID'] = lmap.eval('CELL_ID - 1')
    
    # Select appropriate columns
    concordance = lmap[['CELL_ID', 'SA2_ID']]
    
    
    
    
    ############### Create landuses -- lexicographically ordered list of land-uses (strings)
    
    # Lexicographically ordered list of land-uses
    ag_landuses = sorted(lmap['LU_DESC'].unique().to_list())
    ag_landuses.remove('Non-agricultural land')
    
    # Save to file
    pd.DataFrame(ag_landuses).to_csv(outpath + 'ag_landuses.csv', index = False, header = False)

    # Create a non-agricultural landuses file
    # Do not sort the whole list alphabetically when adding new landuses to the model.
    non_ag_landuses = ["Environmental Plantings"]
    with open(outpath + 'non_ag_landuses.csv', 'w') as non_ag_lu_csv:
        writer = csv.writer(non_ag_lu_csv)
        for lu in non_ag_landuses:
            writer.writerow([lu])
    
    
    
    ############### Create lumap -- 2010 land-use mapping.
    
    # Map land-uses by lexicographical index on the map. Use -1 for anything _not_ in the land-use list.
    lucode = [-1 if (r not in ag_landuses) else ag_landuses.index(r) for r in lmap['LU_DESC']]
    
    # Convert to series and downcast to int8
    lumap = pd.to_numeric( pd.Series(lucode), downcast = 'integer' )
    
    # Save to file. HDF5 takes up way less disk space 
    lumap.to_hdf(outpath + 'lumap.h5', key = 'lumap', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    
    
    
    ############### Create lmmap -- present (2010) land management mapping.
    
    # For now only 'rain-fed' ('dry' == 0) and 'irrigated' ('irr' == 1) available.
    lmmap = lmap['IRRIGATION'].to_numpy()
    
    # Save to file (int8)
    lmap['IRRIGATION'].to_hdf(outpath + 'lmmap.h5', key = 'lmmap', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    
    
    
    ############### Create real_area -- hectares per cell, corrected for geographic map projection.
    
    # Select the appropriate column
    real_area = zones['CELL_HA'].to_numpy()
    
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




    ############### Create cost matrices for transitioning from environmental plantings to agricultural land
    ep_to_ag_t = t.loc['Unallocated - natural land'].to_numpy()
    # add cost of establishing irrigation for irrigated cells
    ep_to_ag_t = np.stack([ep_to_ag_t, ep_to_ag_t + 7500])

    np.save(outpath + 'ep_to_ag_tmatrix.npy', ep_to_ag_t)  # shape: (m, j)




    ############### Create cost matrices for transitioning from agricultural land to environmental plantings
    ag_to_ep_t = ag_to_new_land_uses.to_numpy()[:, 0]
    # add cost for removal of irrigation for irrigated cells
    ag_to_ep_t = np.stack([ag_to_ep_t, ag_to_ep_t + 3000])

    np.save(outpath + 'ag_to_ep_tmatrix.npy', ag_to_ep_t)  # shape: (m, j)



    
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
    bioph['WATER_PRICE_ML_BOM'].to_hdf(outpath + 'water_licence_price.h5', key = 'water_licence_price', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    
    
    
    ############### Calculate exclusion matrix -- x_mrj
    
    # Turn it into a pivot table. Non-NaN entries are permitted land-uses in the SA2.
    ut_ptable = ut.pivot_table(index = 'SA2_ID', columns = ['IRRIGATION', 'LU_DESC'])['LU_ID']
    x_dry = concordance.merge(ut_ptable[0], on = 'SA2_ID', how = 'left')
    x_irr = concordance.merge(ut_ptable[1], on = 'SA2_ID', how = 'left')
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
    
    
    
    
    
    ############### Get river region and drainage division data 
    
    # Create a LUT of river regions ID and name and save to HDF5.
    rivreg_lut = zones.groupby(['HR_RIVREG_ID'], observed = True)['HR_RIVREG_NAME'].first()
    rivreg_lut.to_hdf(outpath + 'rivreg_lut.h5', key = 'rivreg_lut', mode = 'w', format = 'table', index = False, complevel = 9)
    
    # Save river region ID map to file
    zones['HR_RIVREG_ID'].to_hdf(outpath + 'rivreg_id.h5', key = 'rivreg_id', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    
    # Create a LUT of drainage division ID and name and save to HDF5.
    draindiv_lut = zones.groupby(['HR_DRAINDIV_ID'], observed = True)['HR_DRAINDIV_NAME'].first()
    draindiv_lut.to_hdf(outpath + 'draindiv_lut.h5', key = 'draindiv_lut', mode = 'w', format = 'table', index = False, complevel = 9)
    
    # Save drainage division ID map to file
    zones['HR_DRAINDIV_ID'].to_hdf(outpath + 'draindiv_id.h5', key = 'draindiv_id', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    
    
    
    
    ############### Get water yield historical baseline data 
    
    # Select historical (1970 - 2000) water yield under deep rooted and shallow rooted vegetation
    water_yield_baselines = bioph[['WATER_YIELD_HIST_DR_ML_HA', 'WATER_YIELD_HIST_SR_ML_HA']]
    
    # Save to file
    water_yield_baselines.to_hdf(outpath + 'water_yield_baselines.h5', key = 'water_yield_baselines', mode = 'w', format = 'fixed', index = False, complevel = 9)
        
    
    
    
    ############### Forest and reforestation data
    
    # Carbon stock in mature forest on natural land and save to file
    bioph['REMNANT_VEG_T_CO2_HA'].to_hdf(outpath + 'natural_land_t_co2_ha.h5', key = 'natural_land_t_co2_ha', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    
    # Average annual carbon sequestration by Environmental Plantings (block plantings) and save to file
    s = bioph.eval('EP_BLOCK_TREES_AVG_T_CO2_HA_YR + EP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR + EP_BLOCK_SOIL_AVG_T_CO2_HA_YR')
    s.to_hdf(outpath + 'ep_block_avg_t_co2_ha_yr.h5', key = 'ep_block_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Average annual carbon sequestration by Environmental Plantings (belt plantings) and save to file
    s = bioph.eval('EP_BELT_TREES_AVG_T_CO2_HA_YR + EP_BELT_DEBRIS_AVG_T_CO2_HA_YR + EP_BELT_SOIL_AVG_T_CO2_HA_YR')
    s.to_hdf(outpath + 'ep_belt_avg_t_co2_ha_yr.h5', key = 'ep_belt_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Average annual carbon sequestration by Environmental Plantings (riparian plantings) and save to file
    s = bioph.eval('EP_RIP_TREES_AVG_T_CO2_HA_YR + EP_RIP_DEBRIS_AVG_T_CO2_HA_YR + EP_RIP_SOIL_AVG_T_CO2_HA_YR')
    s.to_hdf(outpath + 'ep_rip_avg_t_co2_ha_yr.h5', key = 'ep_rip_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    
    # Average annual carbon sequestration by Hardwood Plantings (block plantings) and save to file
    s = bioph.eval('CP_BLOCK_TREES_AVG_T_CO2_HA_YR + CP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR + CP_BLOCK_SOIL_AVG_T_CO2_HA_YR')
    s.to_hdf(outpath + 'hp_block_avg_t_co2_ha_yr.h5', key = 'hp_block_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Average annual carbon sequestration by Hardwood Plantings (belt plantings) and save to file
    s = bioph.eval('CP_BELT_TREES_AVG_T_CO2_HA_YR + CP_BELT_DEBRIS_AVG_T_CO2_HA_YR + CP_BELT_SOIL_AVG_T_CO2_HA_YR')
    s.to_hdf(outpath + 'hp_belt_avg_t_co2_ha_yr.h5', key = 'hp_belt_avg_t_co2_ha_yr', mode = 'w', format = 'fixed', index = False, complevel = 9)

    
    # Average establishment costs for Environmental Plantings ($/ha) and save to file
    bioph['EP_EST_COST_HA'].to_hdf(outpath + 'ep_est_cost_ha.h5', key = 'ep_est_cost_ha', mode = 'w', format = 'fixed', index = False, complevel = 9)

    # Average establishment costs for Hardwood Plantings ($/ha) and save to file
    bioph['CP_EST_COST_HA'].to_hdf(outpath + 'hp_est_cost_ha.h5', key = 'hp_est_cost_ha', mode = 'w', format = 'fixed', index = False, complevel = 9)
    
    
    
    
    ############### Calculate climate impacts   ***** ADD FACILITY TO TURN OFF CO2 FERTILIZATION???? *****
    
    # Distill list of RCP labels from dataset
    rcps = sorted(list({col[0] for col in cci_raw.columns})) # curly brackets remove duplicates
    
    # Create luid_desc -- old LU_ID to LU_DESC concordance table
    luid_desc = lmap.groupby('LU_ID').first()['LU_DESC'].astype(str)
    
    # Loop through RCPs and format climate change impacts table
    for rcp in rcps:    # rcp = 'rcp2p6'
        
        # Slice off RCP data and get rid of MultiIndex.
        cci_ptable = cci_raw[rcp].reset_index()
        
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
        
        # Check against previous data
        cci_orig = pd.read_hdf('N:/LUF-Modelling/LUTO2_BB/LUTO2/input/climate_change_impacts_' + rcp + '.h5')
        print('Climate change impact for', rcp, 'matches previous data =', cci.equals(cci_orig))
        
        # Write to HDF5 file.
        fname = outpath + 'climate_change_impacts_' + rcp + '.h5'
        kname = 'climate_change_impacts_' + rcp 
        cci.to_hdf(fname, key = kname, mode = 'w', format = 'fixed', index = False, complevel = 9)
        
        
        
    
    ############### Agricultural economics - crops
        
    # Produce a multi-indexed version of the crops data.
    # crops_ptable = crops.pivot_table(index = 'SA2_ID', columns = ['Irrigation', 'LU_DESC'])
    crops_ptable = crops.drop(['LU_DESC', 'Irrigation', 'Area_ABS', 'Prod_ABS'], axis = 1)
    
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
    
    # Check against previous data
    aec_orig = pd.read_hdf('N:/LUF-Modelling/LUTO2_BB/LUTO2/input/agec_crops.h5')
    print('agec_crops matches previous data =', agec_crops.equals(aec_orig))
    
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
    
    # Check against previous data
    lvstk_orig = pd.read_hdf('N:/LUF-Modelling/LUTO2_BB/LUTO2/input/agec_lvstk.h5')
    print('agec_lvstk matches previous data =', agec_lvstk.equals(lvstk_orig))

    # Save to HDF5
    agec_lvstk.to_hdf(outpath + 'agec_lvstk.h5', key = 'agec_lvstk', mode = 'w', format = 'fixed', index = False, complevel = 9)
        
        
        
    
    ############### Agricultural Greenhouse Gas Emissions - crops
        
    # Produce a multi-indexed version of the crops data.
    # cropsGHG_ptable = cropsGHG.drop('LU_ID', axis = 1).pivot_table(index = 'SA2_ID', columns = ['IRRIGATION', 'LU_DESC'])
    cropsGHG_ptable = cropsGHG.drop('LU_DESC', axis = 1)
    
    # Merge to create a cell-based table.
    agGHG_crops = concordance.merge( cropsGHG_ptable, on = 'SA2_ID', how = 'left' )
    
    # Drop unnecessary columns and fill NaNs with zeros.
    agGHG_crops = agGHG_crops.drop(['SA2_ID'], axis = 1)
        
    # Convert columns to integer (need 64-bit integer which can accomodate NaNs)
    agGHG_crops['IRRIGATION'] = agGHG_crops['IRRIGATION'].astype('Int64')
    agGHG_crops['LU_ID'] = agGHG_crops['LU_ID'].astype('Int64')
            
    # Create the MultiIndex structure
    agGHG_crops = agGHG_crops.pivot( index = 'CELL_ID', 
                                   columns = ['IRRIGATION', 'LU_ID'], 
                                   values = ['CO2E_KG_HA_FERT_PROD', 'CO2E_KG_HA_PEST_PROD', 'CO2E_KG_HA_IRRIG', 'CO2E_KG_HA_CHEM_APPL', 'CO2E_KG_HA_CROP_MGT', 'CO2E_KG_HA_CULTIV', 'CO2E_KG_HA_HARVEST', 'CO2E_KG_HA_SOWING', 'CO2E_KG_HA_SOIL_N_SURP']
                                 ).dropna( axis = 'columns', how = 'all')
    
    # The merge flattens the multi-index to tuples, so unflatten back to multi-index
    ts = [(col[0], lms[col[1]], luid_desc[col[2]]) for col in agGHG_crops.columns]
    agGHG_crops.columns = pd.MultiIndex.from_tuples(ts)
                
    # Sort land use in lexicographical order
    agGHG_crops.sort_index(axis = 1, inplace = True)
    
    # Check against previous data (note previous data had zeros instead of NaNs)
    agGHG_orig = pd.read_hdf('N:/LUF-Modelling/LUTO2_BB/LUTO2/input/agGHG_crops.h5')
    print('agGHG_crops matches previous data =', agGHG_crops.fillna(0).equals(agGHG_orig))

    # Save to HDF5
    agGHG_crops.to_hdf(outpath + 'agGHG_crops.h5', key = 'agGHG_crops', mode = 'w', format = 'fixed', index = False, complevel = 9)
        
        
        
    
    ############### Agricultural Greenhouse Gas Emissions - livestock
    
    # Merge to create a cell-based table.
    agGHG_lvstk = concordance.merge( lvstkGHG.stack(level = 0).reset_index()
                                   , on = 'SA2_ID'
                                   , how = 'left' )
    
    # Drop unnecessary columns.
    agGHG_lvstk = agGHG_lvstk.drop(['SA2_ID'], axis = 1)
    
    # Create the MultiIndex structure
    agGHG_lvstk = agGHG_lvstk.pivot( index = 'CELL_ID', 
                                     columns = ['Livestock type'], 
                                     values = ['CO2E_KG_HEAD_DRN_WATER', 'CO2E_KG_HEAD_DUNG_URINE', 'CO2E_KG_HEAD_ELEC', 'CO2E_KG_HEAD_ENTERIC', 'CO2E_KG_HEAD_FODDER', 'CO2E_KG_HEAD_FUEL', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'CO2E_KG_HEAD_IRR_WATER', 'CO2E_KG_HEAD_MANURE_MGT', 'CO2E_KG_HEAD_SEED']
                                   ).dropna( axis = 'columns', how = 'all')
    
    # Change the level names
    agGHG_lvstk.columns.set_names(['INDICATOR', 'LIVESTOCK'], level = [0, 1], inplace = True)
        
    # Swap column levels
    agGHG_lvstk = agGHG_lvstk.reorder_levels([1, 0], axis = 1)
    
    # Sort land use in lexicographical order
    agGHG_lvstk.sort_index(axis = 1, inplace = True)    
    
    # Save to HDF5
    agGHG_lvstk.to_hdf(outpath + 'agGHG_lvstk.h5', key = 'agGHG_lvstk', mode = 'w', format = 'fixed', index = False, complevel = 9)



    
    
    # Complete processing and report back
    t = round(time.time() - start_time)
    print('Completed input data refresh at', time.strftime("%H:%M:%S", time.localtime()), ', taking', t, 'seconds')





