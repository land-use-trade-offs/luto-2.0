#!/usr/bin/env python
"""Compare LUTO's 2010-base inputs with the 2021-base products and with public reference data (evidence for the methods note).

Reads two input packages and writes comparison tables; it builds no model input.
  - 2010-base: the inherited 2010-build files (--input, input/Nick_build/inherited_2010).
  - 2021-base: the outputs of the 2021-base scripts (--base2021, input/Nick_build, one folder per script; table BASE2021):
    nlum_map/; pasture_productivity/pasture_kg_dm_ha.h5; livestock_map/ lumap.h5, lmmap.h5, lvstk_k.h5, stubble_dse_ha.h5
    and the livestock reconciliation CSVs; eligibility/x_mrj.npy; ghg_factors/ agGHG_lvstk.h5, agGHG_irrpast.h5,
    agGHG_lvstk_off_land.csv and ghg_crop_rates.csv (its nir_area_ha column carries the NIR Vol 2 areas); crop_economics/
    agGHG_crops.h5, agec_crops.h5 and the crop calibration CSVs; livestock_economics/ agec_lvstk.h5,
    water_delivery_price.h5 and livestock_water_per_head.csv.
Reference sources (public, downloaded at run time, sha256-checked against pinned values):
  - Australia's National Greenhouse Accounts API (NIR 2024), StatesOutputTimeSeries: the livestock, fertiliser and off-land
    nodes of REF, Australia and the eight states. A changing source: the 454 raw responses are pinned one by one.
  - DCCEEW NIR 2024 activity table, agriculture livestock (state populations, FY2010-11 and FY2020-21).
  - ABS 7121.0 Agricultural Commodities 2015-16 to 2020-21 and ABS 7503.0 Value of Agricultural Commodities Produced
    2015-16 to 2020-21 (state livestock head 2020-21; Tasmanian "Other oilseeds").
  - ABS 4618.0 Water Use on Australian Farms 2020-21 (ASGS Edition 3 and Murray-Darling Basin cubes).
Inherited 2010-build inputs, each sha256 in the provenance file: see INHERITED. Every table is computed from the named
files only; the provenance file lists each table's inputs with their sha256.

Tables in --out (CSV, one '#' header comment row naming what is compared, the sources, units and labels; then the data):
  t1_livestock_head_state_type.csv     1. head by state and type, 2021-base map, against NIR FY2020-21 and ABS 7121.0
                                          2020-21, with the named rows of livestock_recon_state_type.csv
  t2_landuse_area_state_class.csv      2. area by LUTO class and state: 2010 map, NLUM v7.1 2010-11 control, NLUM v7.1
                                          2020-21 before typing, 2021-base map; method change and land-use change
  t3a_ghg_fy2021_rows.csv              3. emissions FY2020-21 by NIR category, state and row, 2021-base map and factors,
  t3b_ghg_fy2021_category_state.csv       against NIR (AR5), split into activity and factor parts
  t3c_ghg_fy2011.csv                      and FY2010-11 on the 2010 map with the 2010 factors
  t4a_crops_by_crop_state.csv          4. crop production and value on the 2021-base map against the ABS window means;
  t4b_crops_named_rows.csv                the named rows; Tasmanian Other oilseeds;
  t4c_tas_other_oilseeds.csv              the sown share by crop group and state against ABS 7121.0 2020-21
  t4d_sown_share_group_state.csv          and NIR Vol 2 A5.5.8.1 areas, with the 14 rows of unverified cause marked
  t5_water_drainage_division.csv       5. irrigation water by drainage division against ABS 4618.0 2020-21
  t6a_pasture_productivity_state.csv   6. pasture productivity 2021 against 2010 by state; x_mrj eligible area by (lm, lu)
  t6b_eligibility_state_lm_lu.csv         and state, 2021 against 2010, with the cells gained and lost
  compare_inputs_summary.txt           key national numbers, each with the files it is computed from
  provenance.csv        (last) sources, inherited and 2021-base inputs (sha256), per-table inputs, outputs

Method notes:
  - Emissions use an offline mirror of LUTO's agricultural GHG arithmetic (ghg.py, quantity.get_yield_pot,
    data.get_stubble_effective_rate, the off-land block of data.py) at RESFACTOR 1, scope 1 (USE_GHG_SCOPE_1);
    operations and dtypes follow LUTO's, evaluated on the allocated
    cells of each (lm, lu). The climate multiplier is 1 (base-year inputs). 2021-base DSE per head = the
    NIR-intake ratings of livestock_map.py; 2010-base DSE per head = 8 / 1.5 / 17 and k = FEED_REQ
    (the 2010 model). Category mapping and the activity / factor split as the FY2021 base-year emissions check
    (FY2020-21) and the FY2010-11 reconciliation (three GWP readings of the 2010 factors).
  - Crops: the ABS window means are those crop_economics.py
    computed (crop_production_window.csv), so the table checks the products against the windows they were built on.
  - Water: ABS 4618.0 2020-21 SA2 values apportioned to drainage divisions; pasture water = head on irrigated
    livestock cells x WR_IRR.
  - Rules stated in REF: grazing pseudo classes, the named-crop rule, tolerances.

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/compare_inputs.py --input input/Nick_build/inherited_2010 --base2021 input/Nick_build --out input/Nick_build/compare_inputs
        --cache input/Nick_build/_cache [--offline] [--live-check] [--source NAME=PATH] [--source-dir DIR]

Dependencies: numpy, pandas, numexpr, PyTables, openpyxl, xlrd, rasterio, scipy. No luto imports.
Memory: peak resident memory about 15 GB on the full grid (measured October 2026); run on a machine with at least 32 GB
of RAM.
Deterministic: rows sorted, floats written with a fixed format, LF line endings; no run timestamps in the tables.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
from importlib import metadata

import numpy as np
import pandas as pd
import rasterio

# ----------------------------------------------------------------------------- sources and pins
ANGA = ('https://greenhouseaccounts.climatechange.gov.au/api/StatesOutputTimeSeries?sectorid={s}&urlencodedlocationlist={l}'
        '&fuelid=1&gasid={g}')
ABS_OLD = 'https://www.abs.gov.au/ausstats/subscriber.nsf/log?openagent&'
ABS_OLD2 = 'https://www.abs.gov.au/AUSSTATS/subscriber.nsf/log?openagent&'
ABS_AC = 'https://www.abs.gov.au/statistics/industry/agriculture/agricultural-commodities-australia/'
ABS_VA = 'https://www.abs.gov.au/statistics/industry/agriculture/value-agricultural-commodities-produced-australia/'
ABS_WU = 'https://www.abs.gov.au/statistics/industry/agriculture/water-use-australian-farms/'
SOURCES = {   # name: (url, sha256 of the raw file or response, note); URLs and pins as the PR C tools that use the same files
    'nir_livestock': ('https://www.dcceew.gov.au/sites/default/files/documents/'
                      'activity-table-1990-2024-agriculture-livestock-national.xlsx',
                      '669ac799188e97f993dc85662886856295fe2dc0e68e26590058d765e97efa59',
                      'DCCEEW NIR 2024 activity table, agriculture livestock, CC BY 4.0'),
    'abs_7121_2020_21': (ABS_AC + '2020-21/AGCDCASGS202021.xlsx',
                         '0fdbd3e398b598a24c8bc315e5bd7cae36e8ea4fb6f6cea5a7b745e0950bedf5',
                         'ABS 7121.0 Agricultural Commodities, Australia, 2020-21, ASGS Ed 3 cube, CC BY 4.0'),
    'VACPDCASGS202021.xlsx': (ABS_VA + '2020-21/VACPDCASGS202021.xlsx',
                              'f737632f72d77f36cdff601487f2618cde703adb818727c6e5f7378209ea7eab', 'ABS 7503.0 2020-21, CC BY 4.0'),
    '7121do001_201516_state.xls': (
        ABS_OLD + '7121do001_201516.xls&7121.0&Data%20Cubes&DC87B0B6B88A1FCFCA25816A00177E45&0&2015-16&28.07.2017&Previous',
        'faa084ae4671c9a1360fb8621278f823088d85dbf6cff859d3286632c81a7660', 'ABS 7121.0 2015-16, state, CC BY'),
    '75030do001_201516_state.xls': (
        ABS_OLD + '75030do001_201516.xls&7503.0&Data%20Cubes&548E8E628E7D11D8CA25816A00177ED8&0&2015-16&28.07.2017&Previous',
        'afeee00a3a9522a3cbaf33b3557b5dd9a729f7dcae5d8284c109c11fa407ef8d', 'ABS 7503.0 2015-16, state, CC BY'),
    '71210do001_201617_state.xls': (
        ABS_OLD2 + '71210do001_201617.xls&7121.0&Data%20Cubes&EE7BA2BEC70DFF04CA2582910013A626&0&2016-17&21.05.2018&Previous',
        'accfa40f456ec2aeb06ee114a88896d702d04ab0471a5fdc7880a7d2fa1d3262', 'ABS 7121.0 2016-17, state, CC BY'),
    '75030do001_201617_state.xls': (
        ABS_OLD2 + '75030do001_201617.xls&7503.0&Data%20Cubes&1D4B9105890F1A41CA2582910013AA02&0&2016-17&21.05.2018&Previous',
        '90fab9cc19e86bc52f285da320c239a3e6482c2b66df0bba6f17b84eb7062727', 'ABS 7503.0 2016-17, state, CC BY'),
    '71210do005_201718.csv': (ABS_AC + '2017-18/71210do005_201718.csv',
                              'ab35f5a53d6d6675ea341be2711845b20727c1a34447da275903a242881c2645', 'ABS 7121.0 2017-18, CC BY 4.0'),
    '75030do003_201718.csv': (ABS_VA + '2017-18/75030do003_201718.csv',
                              '1f5c2cd7f55b25d85f5427c01bffb2dafdaa1579931ccdc1612482e614efe0d1', 'ABS 7503.0 2017-18, CC BY 4.0'),
    '71210do005_201819.csv': (ABS_AC + '2018-19/71210do005_201819.csv',
                              '012d8979a54f8413d88fa63df3e9b7ee70f9549318bfaeaa68e1ab1c18bcf0ca', 'ABS 7121.0 2018-19, CC BY 4.0'),
    '75030do003_201819.csv': (ABS_VA + '2018-19/75030do003_201819.csv',
                              'e9b99a53abdb5947cef7e541f176044dfa9699f1d0dd8d202ac1d6e48f0c8f42', 'ABS 7503.0 2018-19, CC BY 4.0'),
    '71210DO001_201920_SA4.xlsx': (ABS_AC + '2019-20/71210DO001_201920.xlsx',
                                   'db0a67410010bff0374a37fbda2f70a5b18c231983b09fda91bc32378a59f15e', 'ABS 7121.0 2019-20, CC BY 4.0'),
    '75030DO001_201920_SA4.xlsx': (ABS_VA + '2019-20/75030DO001_201920.xlsx',
                                   '6adeb1b0931a8da0240e1eaed38626e097ad78f1007b1cf4def35e44d397aaec', 'ABS 7503.0 2019-20, CC BY 4.0'),
    'WUAFDCASGS202021.xlsx': (ABS_WU + '2020-21/WUAFDCASGS202021.xlsx',
                              '1453aacfcb132edb1609b36aad157530d02e2da3387702892b34f2368d6cd2eb', 'ABS 4618.0 2020-21, ASGS Ed 3, CC BY 4.0'),
    'WUAFDCMDB202021.xlsx': (ABS_WU + '2020-21/WUAFDCMDB202021.xlsx',
                             'b3e9c973a07d73ca89fe685a5d3d09f42471b4c8b67ab7b58190c05bba8c47dd', 'ABS 4618.0 2020-21, MDB, CC BY 4.0'),
}
INHERITED = ['lumap.h5', 'lmmap.h5', 'real_area.h5', 'state_id.npy', 'NLUM_2010-11_mask.tif', 'ag_landuses.csv',
             'safe_pur_natl.h5', 'safe_pur_modl.h5', 'pasture_kg_dm_ha.h5', 'feed_req.h5', 'agGHG_crops.h5', 'agGHG_lvstk.h5',
             'agGHG_irrpast.h5', 'agGHG_lvstk_off_land.csv', 'demand_projections.h5', 'x_mrj.npy', 'draindiv_id.h5',
             'draindiv_lut.h5']
BASE2021 = {   # key: path under --base2021 (input/Nick_build: <script>/<file>), producing script
    'class_alloc_2010_11_lumap.npy': ('nlum_map/class_alloc_2010_11_lumap.npy', 'nlum_map.py'),
    'class_alloc_2010_11_lmmap.npy': ('nlum_map/class_alloc_2010_11_lmmap.npy', 'nlum_map.py'),
    'class_alloc_2010_11_category.npy': ('nlum_map/class_alloc_2010_11_category.npy', 'nlum_map.py'),
    'class_alloc_2020_21_lumap.npy': ('nlum_map/class_alloc_2020_21_lumap.npy', 'nlum_map.py'),
    'class_alloc_2020_21_lmmap.npy': ('nlum_map/class_alloc_2020_21_lmmap.npy', 'nlum_map.py'),
    'class_alloc_2020_21_category.npy': ('nlum_map/class_alloc_2020_21_category.npy', 'nlum_map.py'),
    'sa2_2021_cells.npy': ('nlum_map/sa2_2021_cells.npy', 'nlum_map.py'),
    'pasture_kg_dm_ha.h5': ('pasture_productivity/pasture_kg_dm_ha.h5', 'pasture_productivity.py'),
    'lumap.h5': ('livestock_map/lumap.h5', 'livestock_map.py'),
    'lmmap.h5': ('livestock_map/lmmap.h5', 'livestock_map.py'),
    'lvstk_k.h5': ('livestock_map/lvstk_k.h5', 'livestock_map.py'),
    'stubble_dse_ha.h5': ('livestock_map/stubble_dse_ha.h5', 'livestock_map.py'),
    'livestock_dse_ratings.csv': ('livestock_map/livestock_dse_ratings.csv', 'livestock_map.py'),
    'livestock_recon_state_type.csv': ('livestock_map/livestock_recon_state_type.csv', 'livestock_map.py'),
    'x_mrj.npy': ('eligibility/x_mrj.npy', 'eligibility.py'),
    'agGHG_lvstk.h5': ('ghg_factors/agGHG_lvstk.h5', 'ghg_factors.py'),
    'agGHG_irrpast.h5': ('ghg_factors/agGHG_irrpast.h5', 'ghg_factors.py'),
    'agGHG_lvstk_off_land.csv': ('ghg_factors/agGHG_lvstk_off_land.csv', 'ghg_factors.py'),
    'ghg_crop_rates.csv': ('ghg_factors/ghg_crop_rates.csv', 'ghg_factors.py'),
    'agGHG_crops.h5': ('crop_economics/agGHG_crops.h5', 'crop_economics.py'),
    'agec_crops.h5': ('crop_economics/agec_crops.h5', 'crop_economics.py'),
    'crop_production_window.csv': ('crop_economics/crop_production_window.csv', 'crop_economics.py'),
    'crop_yield_calibration.csv': ('crop_economics/crop_yield_calibration.csv', 'crop_economics.py'),
    'crop_water_calibration.csv': ('crop_economics/crop_water_calibration.csv', 'crop_economics.py'),
    'agec_lvstk.h5': ('livestock_economics/agec_lvstk.h5', 'livestock_economics.py'),
    'water_delivery_price.h5': ('livestock_economics/water_delivery_price.h5', 'livestock_economics.py'),
    'livestock_water_per_head.csv': ('livestock_economics/livestock_water_per_head.csv', 'livestock_economics.py'),
}
NCELL = 6956407
LOC = {3: 'AUSTRALIA', 6: 'NSW', 11: 'VIC', 8: 'QLD', 9: 'SA', 12: 'WA', 10: 'TAS', 7: 'NT', 5: 'ACT'}
STS = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
ST_NAME = {1: 'NSW', 2: 'VIC', 3: 'QLD', 4: 'SA', 5: 'WA', 6: 'TAS', 7: 'NT', 8: 'ACT', 9: 'OT'}
LMS = ('dry', 'irr')
TYPES = ('BEEF', 'SHEEP', 'DAIRY')
LV = ('Beef', 'Sheep', 'Dairy')
FY21, FY11 = '2020-21', '2010-11'

# ============================================================================= reference table (hand-entered)
# Each entry with its source. Changing any is a ruling.
REF = {
    # LUTO field <- NIR 2024 nodes (ANGA sector ids), summed (as ghg_factors.py REF)
    'lvstk_nodes': {
        'DAIRY': {'ENTERIC': [684], 'MANURE_MGT': [737, 5557, 5562, 5555], 'DUNG_URINE': [5835, 5837],
                  'IND_LEACH_RUNOFF': [5851, 5855]},
        'BEEF': {'ENTERIC': [5811], 'MANURE_MGT': [5823], 'DUNG_URINE': [5836, 5838], 'IND_LEACH_RUNOFF': [5852, 5856]},
        'SHEEP': {'ENTERIC': [705], 'MANURE_MGT': [945], 'DUNG_URINE': [5321, 5340], 'IND_LEACH_RUNOFF': [5377, 5409]}},
    'lvstk_group': {'DAIRY': 'Number of dairy cattle', 'BEEF': 'Number of beef cattle - pasture', 'SHEEP': 'Number of sheep'},
    # NIR category label of each LUTO livestock field (as the FY2021 base-year emissions check)
    'category': {'ENTERIC': '3.A', 'MANURE_MGT': '3.B (+3.D.a.2.a dairy)', 'DUNG_URINE': '3.D.a.3',
                 'IND_LEACH_RUNOFF': '3.D.b (manure)'},
    # fertiliser N2O nodes per NIR system: direct 3.D.a.1, deposition 3.D.b.1.i, leaching 3.D.b.2.i (ghg_factors.py REF)
    'fert_systems': {'irr_pasture': (5285, 5364, 5396), 'irr_crop': (5286, 5365, 5397), 'dry_pasture': (5287, 5366, 5398),
                     'dry_crop': (5288, 5367, 5399), 'sugar': (5289, 5368, 5400), 'cotton': (5290, 5369, 5401),
                     'hort': (5291, 5370, 5402)},
    'hort_lus': ['Vegetables', 'Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears', 'Plantation fruit', 'Stone fruit',
                 'Tropical stone fruit'],
    # off-land, national, AR5 (gas 103): swine and poultry nodes
    'swine_nodes': [720, 1092, 5263, 5279, 5300, 5322, 5341, 5378, 5410],
    'poultry_nodes': [5249, 5559, 5563, 5307, 5348, 5385, 5417],
    # LUTO default scenario of demand_projections.h5 (luto settings) and egg mass 60 g (settings.EGGS_AVG_WEIGHT)
    'demand_keys': ('SSP2', 'BAU', 'BAU', 2050, 'Trend', 1, 'BAU'), 'egg_g': 60,
    'offland_commodities': ['pork', 'chicken', 'eggs', 'aquaculture'],
    # LUTO model constants (quantity.py): DSE per head of the 2010 model and grassfed factors
    'dse_2010': {'BEEF': 8, 'SHEEP': 1.5, 'DAIRY': 17}, 'grassfed': {'BEEF': 0.85, 'SHEEP': 0.85, 'DAIRY': 0.65},
    'scope1_lvstk': ['CO2E_KG_HEAD_DUNG_URINE', 'CO2E_KG_HEAD_ENTERIC', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF',
                     'CO2E_KG_HEAD_MANURE_MGT'],                       # settings.py LVSTK_GHG_SCOPE_1
    'scope1_crop': ['CO2E_KG_HA_SOIL'],                                # settings.py CROP_GHG_SCOPE_1
    'stubble_lus': ['Winter cereals', 'Summer cereals'],                # settings.STUBBLE_LAND_USES
    'stubble_agec_fields': ['QC', 'WR_DRN', 'F1', 'Q1', 'P1', 'F2', 'Q2', 'P2', 'F3', 'Q3', 'P3'],   # data.py
    # GWP readings of the 2010 factors (CH4, N2O)
    'gwp_cases': {'SAR': (21, 310), 'AR4': (25, 298), 'AR5': (28, 265)},
    'tol_rel': 0.034,          # within-tolerance flag of the FY2021 base-year emissions check
    # Land use (item 2): class allocation cells not yet typed to a livestock type carry lu 23 or 22 with these
    # categories (nlum_map.py CATEGORY); they are reported as pseudo classes
    'pseudo': {12: 'Grazing untyped - natural land', 13: 'Grazing untyped - natural land',
               22: 'Grazing untyped - modified land', 23: 'Grazing untyped - modified land'},
    # Crops (item 4): the named crops (as ruled: Tasmanian vegetables, summer legumes, tropical stone fruit,
    # apples, pears, other non-cereal crops; summer oilseeds over-placed likewise), as (crop, state or None for all
    # states); a state row is named when its tonnes not placed differ from 0 by more than 0.5 t
    # (the over-placement threshold)
    'named_crops': [('Vegetables', 'TAS'), ('Summer legumes', None), ('Tropical stone fruit', None), ('Apples', None),
                   ('Pears', None), ('Other non-cereal crops', None), ('Summer oilseeds', None)], 'row_t': 0.5,
    'named_crop_label': 'NLUM maps less area than ABS records (NLUM against ABS, cause unverified); tonnes not placed',
    'nomap_label': 'production without map area',
    # Water (item 5): ABS 4618 2020-21 items
    'water_items': {'total': 'AGWATRPRODAML_F', 'pasture_grazed': 'AGWATRPACRAMLGRZ_F', 'hay_silage': 'AGWATRPHAYSILAML_F'},
    # Tasmanian Other oilseeds: ABS 7121 / 7503 state files FY2016-21
    'oilseed_files': [(2016, '7121do001_201516_state.xls', '7121'), (2016, '75030do001_201516_state.xls', '7503'),
                      (2017, '71210do001_201617_state.xls', '7121'), (2017, '75030do001_201617_state.xls', '7503'),
                      (2018, '71210do005_201718.csv', '7121'), (2018, '75030do003_201718.csv', '7503'),
                      (2019, '71210do005_201819.csv', '7121'), (2019, '75030do003_201819.csv', '7503'),
                      (2020, '71210DO001_201920_SA4.xlsx', '7121'), (2020, '75030DO001_201920_SA4.xlsx', '7503'),
                      (2021, 'abs_7121_2020_21', '7121'), (2021, 'VACPDCASGS202021.xlsx', '7503')],
    'oilseed_pattern': r'Oilseeds - Other oilseeds',
    # Sown share (item 4): crop groups: name -> (LUTO lu codes, lm codes,
    # ABS 7121.0 2020-21 Table 1 commodity codes (area, ha), NIR Vol 2 A5.5.8.1 system or None). Flags below 0.4 or above 1.05.
    'sown_groups': {
        'Winter cereals': ([25], [0, 1], ['AGCEREAL_AHAWHT_F', 'AGCEREAL_AHAOAT_F', 'AGCEREAL_AHABAR_F', 'AGCEREAL_AHAOTH_F'], None),
        'Summer cereals (excl. rice)': ([18], [0, 1], ['AGCEREAL_AHASOR_F', 'AGCEREAL_AHAMAI_F'], None),
        'Rice': ([13], [1], ['AGCEREAL_AHARIC_F'], None),
        'Winter oilseeds': ([27], [0, 1], ['AGOTHCROP_AHACAN_F'], None),
        'Summer oilseeds': ([20], [0, 1], ['AGOTHCROP_AHAOIL_F'], None),
        'Oilseeds (winter + summer)': ([27, 20], [0, 1], ['AGOTHCROP_AHACAN_F', 'AGOTHCROP_AHAOIL_F'], None),
        'Legumes (winter + summer)': ([26, 19], [0, 1], ['AGOTHCROP_AHACHI_F', 'AGOTHCROP_AHALEN_F', 'AGOTHCROP_AHALUP_F',
                                                         'AGOTHCROP_AHAOPU_F'], None),
        'Cotton': ([4], [0, 1], ['AGCOTTPLTDAHA_F'], 'cotton'),
        'Cotton, irrigated': ([4], [1], ['AGCOTTPLTDAHAIRR_F'], None),
        'Cotton, dryland': ([4], [0], ['AGCOTTPLTDAHANON_F'], None),
        'Sugar': ([17], [0, 1], ['AGSUGARCANAHA_F'], 'sugar'),
        'Hay': ([8], [0, 1], ['AGPASCRHAYSILAHA_F'], None),
        'Other non-cereal crops': ([10], [0, 1], ['AGOTHCROP_AHA_F'], None),
        'Vegetables': ([24], [0, 1], ['AGVEGE_AHA_F'], None),
        'Grapes': ([7], [0, 1], ['AGGRAPE_AHA_F'], None),
        'Fruit and nuts (excl. grapes)': ([0, 3, 9, 11, 12, 16, 21], [0, 1], ['AGFRUT_AHA_F'], None),
        'Broadacre (cereals and other crops incl. sugar, cotton)': ([25, 18, 13, 27, 20, 26, 19, 4, 17, 10], [0, 1],
                                                                    ['AGBROADCRPALLAHA_F'], None),
        'All crops (incl. hay and horticulture)': ([25, 18, 13, 27, 20, 26, 19, 4, 17, 10, 8, 24, 7, 0, 3, 9, 11, 12, 16, 21],
                                                   [0, 1], ['AGCROPTOTAHA_F'], None),
        'NIR non-irrigated crops': ([25, 18, 26, 19, 27, 20, 8, 10], [0], [], 'dry_crop'),
        'NIR irrigated crops': ([25, 18, 26, 19, 27, 20, 8, 10, 13], [1], [], 'irr_crop'),
        'NIR horticulture': ([24, 0, 11, 16, 21, 3, 9, 12, 7], [0, 1], [], 'hort'),
        'NIR irrigated pasture': ([1, 5, 14], [1], [], 'irr_pasture'),
        'NIR non-irrigated crops (excl. hay)': ([25, 18, 26, 19, 27, 20, 10], [0], [], 'dry_crop'),
        'NIR irrigated crops (excl. hay)': ([25, 18, 26, 19, 27, 20, 10, 13], [1], [], 'irr_crop'),
        'Sugar (ABS cut for crushing)': ([17], [0, 1], ['AGSUGARCRSAHA_F'], None)},
    'sown_band': (0.4, 1.05), 'small_ha': 5000,
    # Cause codes of flagged rows, rules in order (C7 carries the figures it cites as text)
    'sown_causes': {
        'C1a': 'C1 small area: sown area under 5 kha with no NLUM area in the state',
        'C8a': 'C8 NLUM v7.1 has no area of this class in the state',
        'C1': 'C1 small area: LUTO and sown area both under 5 kha (about 50 cells or fewer)',
        'C9': 'C9 NIR has no area for this state and system (NT and WA cotton grown in FY2021 per ABS)',
        'C2': 'C2 source scope: NIR crop systems exclude hay; the excl.-hay comparison is in range',
        'C3': 'C3 source scope: ABS hay and silage includes pasture cut for hay or silage (grazing land in NLUM)',
        'C10': 'C10 not a crop (area ratio, not a sown share): NLUM 4.2.0 irrigated pasture by state differs from NIR irrigated '
               'pasture; LUTO area tracks the native 4.2.0 area plus held irrigated livestock cells; mechanism not verified',
        'C4': 'C4 source scope: ABS all other crops n.e.c. is wider than NLUM 3.3.0 cropping (commodity unspecified)',
        'C5': 'C5 cropping land not sown in the year: dryland cotton is opportunistic, planted on part of the land NLUM maps in '
              'the cotton system',
        'C6': 'C6 NLUM v7.1 oilseed attribution: the probability split puts summer oilseed area where ABS records little (LUTO '
              'area equals the native NLUM area)',
        'C7': 'C7 NSW sugar: NIR area (12.9 kha) is below ABS cut for crushing (17.4) and total cane (23.7; NSW cane is mostly a '
              'two-year crop); NIR notes limited ABS data for 2020-21; LUTO adds 7.0 kha of 2010 sugar held by the footprint to '
              'the native 28.5',
        'C8': 'C8 NLUM v7.1 class area differs from the statistic in this state (LUTO area equals the native NLUM area; source '
              'difference, cause not verified)'},
    # The 14 rows "NLUM against ABS/NIR, cause unverified": (group, state, statistic)
    'unverified_rows': [('Winter oilseeds', 'TAS', 'abs'), ('Legumes (winter + summer)', 'TAS', 'abs'), ('Hay', 'NT', 'abs'),
                  ('Vegetables', 'TAS', 'abs'), ('Grapes', 'VIC', 'abs'),
                  ('Broadacre (cereals and other crops incl. sugar, cotton)', 'NT', 'abs'),
                  ('All crops (incl. hay and horticulture)', 'TAS', 'abs'), ('All crops (incl. hay and horticulture)', 'NT', 'abs'),
                  ('NIR irrigated crops', 'VIC', 'nir'), ('NIR irrigated crops', 'TAS', 'nir'), ('NIR horticulture', 'WA', 'nir'),
                  ('NIR horticulture', 'TAS', 'nir'), ('NIR irrigated crops (excl. hay)', 'VIC', 'nir'),
                  ('NIR irrigated crops (excl. hay)', 'TAS', 'nir')],
}
# ============================================================================= end of reference table
SYSTEMS = list(REF['fert_systems'])


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- cache (shared, content-addressed)
def cache_path(cache, digest, url):
    ext = os.path.splitext(url.split('?')[0].rsplit('/', 1)[-1])[1] or ('.json' if '/api/' in url else '')
    return os.path.join(cache, digest[:2], digest + ext)


def cache_index_add(cache, name, url, digest, nbytes):
    p = os.path.join(cache, 'index.csv')
    rows = {}
    if os.path.exists(p):
        with open(p, encoding='utf-8', newline='') as f:
            rows = {(r['sha256'], r['url']): r for r in csv.DictReader(f)}
    rows[(digest, url)] = dict(sha256=digest, url=url, bytes=str(nbytes), name=name)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        w = csv.DictWriter(f, fieldnames=['sha256', 'url', 'bytes', 'name'], lineterminator='\n')
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


def download(url):
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-compare-inputs/1.0)',
                                                       'Accept': 'application/json, */*'})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:          # network retry
            last = e
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f'download failed from {url}: {last}')


def fetch(name, cache, offline, overrides):
    """Path of the pinned raw file `name` in the cache; download it, or take an override copy, on a cache miss.
    Any byte difference from the pin stops the run; a changed download is set aside in _quarantine, never re-pinned."""
    url, pin, _ = SOURCES[name]
    p = cache_path(cache, pin, url)
    if os.path.exists(p):
        if sha_file(p) != pin:
            raise SystemExit(f'{name}: cached file {p} does not match its pin {pin}; remove it and rerun')
        return p
    if name in overrides:
        with open(overrides[name], 'rb') as f:
            data = f.read()
    elif offline:
        raise SystemExit(f'--offline: {name} ({url}) is not in the cache {cache}')
    else:
        data = download(url)
        if name.startswith('anga_'):
            time.sleep(0.2)
    got = sha(data)
    if got != pin:
        q = os.path.join(cache, '_quarantine')
        os.makedirs(q, exist_ok=True)
        qp = os.path.join(q, f'{name}_{got}')
        with open(qp, 'wb') as f:
            f.write(data)
        raise SystemExit(f'{name}: {url} gives sha256 {got} ({len(data)} bytes), pinned {pin}. '
                         f'Copy set aside at {qp}; the pin is not changed.')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + '.part', 'wb') as f:
        f.write(data)
    os.replace(p + '.part', p)
    cache_index_add(cache, name, url, pin, len(data))
    return p


def live_check(names):
    """A live run samples the changing source against the manifest and stops if any response differs."""
    for n in names:
        url, pin, _ = SOURCES[n]
        got = sha(download(url))
        if got != pin:
            raise SystemExit(f'{n}: live response {url} has sha256 {got}, pinned {pin}; the source has changed')
        time.sleep(0.2)


# ----------------------------------------------------------------------------- reference readers
def anga_frame(paths):
    """Long table of the pinned API responses (one row per location, sector, gas and year; as ghg_factors.py)."""
    recs = []
    for n in sorted(paths, key=lambda n: tuple(int(x[1:]) for x in n.split('_')[1:])):
        loc, sec, gas = (int(x[1:]) for x in n.split('_')[1:])
        d = json.load(open(paths[n], encoding='utf-8'))
        assert d['error'] == '', (n, d['error'])
        for x in d.get('data') or []:
            assert (x['l'], x['s'], x['g']) == (loc, sec, gas), n
            recs.append((loc, sec, gas, x['date'], x['v']))
    A = pd.DataFrame(recs, columns=['loc_id', 'sector_id', 'gas_id', 'year_end', 'value'])
    A['loc_name'] = A.loc_id.map(LOC)
    A['fy_label'] = (A.year_end - 1).astype(str) + '-' + (A.year_end % 100).astype(str).str.zfill(2)
    return A.sort_values(['loc_id', 'sector_id', 'gas_id', 'year_end'])


def nir_populations(xlsx, fys):
    """NIR activity table: population by sheet (state), group, class and FY (as ghg_factors.py)."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = []
    for st in ['Australia', 'NSW', 'Qld', 'Vic', 'WA', 'SA', 'NT', 'Tas', 'ACT']:
        R = list(wb[st].iter_rows(values_only=True))
        hdr = R[5]
        ycols = {i: h for i, h in enumerate(hdr) if isinstance(h, str) and '-' in h and h[:2] in ('19', '20')}
        grp = sub = None
        for r in R[6:55]:
            if r[0]:
                grp = r[0].strip()
                sub = None
                continue
            lab = next((x for x in r[1:4] if isinstance(x, str)), None)
            if lab is None:
                continue
            if r[1] and r[2] is None and r[3] is None and all(r[i] is None for i in ycols):
                sub = r[1]
                continue
            cls = lab if not sub or lab == sub else f'{sub} | {lab}'
            for i, fy in ycols.items():
                if fy in fys:
                    rows.append(dict(state=st.upper(), group=grp, cls=cls.strip(), fy=fy, value=r[i]))
    return pd.DataFrame(rows)


def nir_state_heads(xlsx):
    """NIR state populations FY2020-21 by type, summed over classes (as livestock_map.py)."""
    out = []
    names = {'NSW': 1, 'Vic': 2, 'Qld': 3, 'SA': 4, 'WA': 5, 'Tas': 6, 'NT': 7, 'ACT': 8}
    for st in ['NSW', 'Qld', 'Vic', 'WA', 'SA', 'NT', 'Tas', 'ACT']:
        d = pd.read_excel(xlsx, sheet_name=st, header=None)
        h = d.iloc[5]
        ycols = [c for c in d.columns if isinstance(h[c], str) and '-' in h[c] and h[c][:2] in ('19', '20') and h[c] == FY21]
        for grp, a, b in [('beef', 'Number of beef cattle - pasture', 'Number of dairy cattle'),
                          ('dairy', 'Number of dairy cattle', 'Number of feedlot'), ('sheep', 'Number of sheep', 'Number of pigs')]:
            i0 = d.index[d[0].astype(str).str.startswith(a)][0]
            i1 = d.index[d[0].astype(str).str.startswith(b)][0]
            for i in range(i0 + 1, i1):
                for c in ycols:
                    out.append(dict(state=names[st], type=grp, value=pd.to_numeric(d.iloc[i, c], errors='coerce')))
    return pd.DataFrame(out).groupby(['state', 'type']).value.sum()


def abs_state_livestock(xlsx):
    """ABS 7121.0 2020-21 'Table 1' state head by type (state rows of livestock_map.py abs_livestock)."""
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)['Table 1']
    rows = list(ws.iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r[0] == 'Region code')
    df = pd.DataFrame(rows[hi + 1:], columns=['code', 'name', 'commodity_code', 'commodity', 'value', 'flag', 'nb', 'nb_rse'])
    lab = {'Livestock - Meat cattle - Total (no.)': 'beef', 'Livestock - Dairy cattle - Total (no.)': 'dairy',
           'Livestock - Sheep and lambs - Total (no.)': 'sheep'}
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    s = df[df.code.astype(str).str.fullmatch(r'[1-9]') & df.commodity.isin(list(lab))].copy()
    s['state'] = s.code.astype(int)
    s['type'] = s.commodity.map(lab)
    return s.groupby(['state', 'type']).value.sum()


# ABS 7121 / 7503 state files (the readers of crop_economics.py)
STATE_BY_LABEL = {'Australia': 'AUS', 'New South Wales': 'NSW', 'Victoria': 'VIC', 'Queensland': 'QLD', 'South Australia': 'SA',
                  'Western Australia': 'WA', 'Tasmania': 'TAS', 'Northern Territory': 'NT', 'Australian Capital Territory': 'ACT'}
KNOWN_UNITS = ['t/ha', 'kg/ha', 'kg/tree', 'kg/m2', 'ha', 't', 'kg', 'no.', '%', 'm2', 'dozens', '$']
SHEETS_7503_201516 = ['Aust.', 'NSW', 'Vic.', 'Qld', 'SA', 'WA', 'Tas.', 'NT', 'ACT']


def parse_unit(desc):
    hits = [t for t in (t.strip() for t in re.findall(r'\(([^()]*)\)', desc)) if t in KNOWN_UNITS]
    return hits[-1] if hits else ''


def read_csv_flex(raw, hdr_skip=4):
    try:
        txt = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        txt = raw.decode('cp1252')
    return pd.read_csv(io.StringIO(txt), skiprows=hdr_skip, dtype=str, keep_default_na=False)


def read_excel_cube(path, sheets=None):
    x = pd.ExcelFile(path)
    out = []
    for sh in x.sheet_names:
        if sheets is not None and sh not in sheets:
            continue
        raw = x.parse(sh, header=None, dtype=str)
        hdr = next((i for i in range(min(12, len(raw))) if str(raw.iat[i, 0]).strip() == 'Region code'), None)
        if hdr is None:
            continue
        body = raw.iloc[hdr + 1:].copy().fillna('')
        body = body.iloc[:, :raw.shape[1]].copy()
        if body.shape[1] == 8:
            body.columns = ['rc', 'label', 'code', 'desc', 'v1', 'r1', 'v2', 'r2']
        else:
            body.columns = ['rc', 'label', 'desc', 'v1', 'r1', 'v2', 'r2'][:body.shape[1]]
            body.insert(2, 'code', '')
        out.append(body)
    return pd.concat(out, ignore_index=True)


def abs_state_rows(path, fname):
    """State rows (rc 0-8 with a state label) of one ABS 7121 / 7503 state file as rc, label, desc, v1, v2 strings."""
    if fname.endswith('.csv'):
        d = read_csv_flex(open(path, 'rb').read())
        d.columns = ['rc', 'label', 'code', 'desc', 'v1', 'r1', 'v2', 'r2']
    else:
        d = read_excel_cube(path, sheets=SHEETS_7503_201516 if fname == '75030do001_201516_state.xls' else None)
    d['rc'] = d['rc'].map(lambda s: str(s).strip()[:-2] if re.fullmatch(r'\d+\.0', str(s).strip()) else str(s).strip())
    d['desc'] = d['desc'].astype(str).str.strip()
    d = d[d['rc'].isin([str(i) for i in range(9)]) & d['label'].isin(STATE_BY_LABEL.keys()) & (d['desc'] != '')].copy()
    d['state'] = d['label'].map(STATE_BY_LABEL)
    return d


def tas_other_oilseeds(P):
    """ABS Tasmanian 'Other oilseeds' production (t), area (ha) and local value ($, nominal), FY2016-21."""
    rows = []
    for fy, f, ser in REF['oilseed_files']:
        d = abs_state_rows(P[f], f)
        d = d[(d.state == 'TAS') & d.desc.str.contains(REF['oilseed_pattern'], regex=True)]
        num = lambda s: pd.to_numeric(str(s).replace(',', '').strip(), errors='coerce')
        if ser == '7121':
            for _, r in d.iterrows():
                u = parse_unit(r.desc)
                if u in ('t', 'ha'):
                    rows.append(dict(fy_end=fy, measure='production_t' if u == 't' else 'area_ha', value=num(r.v1),
                                     source_file=f, item=r.desc))
        else:
            for _, r in d[d.desc.str.endswith('Other oilseeds')].iterrows():
                rows.append(dict(fy_end=fy, measure='local_value_aud', value=num(r.v2), source_file=f, item=r.desc))
    return pd.DataFrame(rows)


def sown_share(lu, lm, ST, area, held, abs_xlsx, NA):
    """Sown share = sown area / LUTO cropping-land area by crop group and state, against ABS 7121.0 2020-21 state areas and NIR
    Vol 2 A5.5.8.1 FY2020-21 areas, with flags and cause codes. No native-target
    column: no PR C tool writes the table it comes from."""
    names = {1: 'NSW', 2: 'VIC', 3: 'QLD', 4: 'SA', 5: 'WA', 6: 'TAS', 7: 'NT', 8: 'ACT'}
    ab = pd.read_excel(abs_xlsx, 'Table 1', header=6)
    ab.columns = ['code', 'label', 'ccode', 'desc', 'est', 'rse', 'nbus', 'nbus_rse']
    ab = ab[ab.code.astype(str).isin([str(i) for i in range(0, 9)])].copy()
    ab['code'] = ab.code.astype(int)
    ab['estn'] = pd.to_numeric(ab.est, errors='coerce')      # 'np' (not published): missing, counted
    a64 = area.astype(np.float64)
    rows = []
    for gname, (lus, lms, codes, nsys) in REF['sown_groups'].items():
        sel = np.isin(lu, lus) & np.isin(lm, lms)
        sets = [([s_], names[s_]) for s_ in range(1, 9)] + [(list(range(1, 9)), 'AUS')] + ([([1, 8], 'NSW+ACT')] if nsys else [])
        for keys, label in sets:
            m = sel & np.isin(ST, keys)
            r = dict(group=gname, state=label, luto_area_ha=float(a64[m].sum()), luto_allocated_ha=float(a64[m & ~held].sum()),
                     luto_held_ha=float(a64[m & held].sum()), abs_area_ha=np.nan, abs_np_entries=np.nan, abs_max_rse_pct=np.nan,
                     nir_area_ha=np.nan)
            if codes:
                x = ab[ab.code.isin([0] if label == 'AUS' else keys) & ab.ccode.isin(codes)]
                r['abs_area_ha'] = float(x.estn.sum()) if len(x) and x.estn.notna().any() else np.nan
                r['abs_np_entries'] = int(x.estn.isna().sum())
                r['abs_max_rse_pct'] = float(pd.to_numeric(x.rse, errors='coerce').max()) if len(x) else np.nan
            if nsys:     # NIR's NSW block is NSW + ACT: no NIR value for NSW or ACT alone
                nk = ([] if label in ('NSW', 'ACT') else ['NSW'] if label == 'NSW+ACT'
                      else sorted(set(STS) - {'ACT'}) if label == 'AUS' else [label])
                r['nir_area_ha'] = float(sum(NA[(nsys, k)] for k in nk)) if nk else np.nan
            rows.append(r)
    D = pd.DataFrame(rows)
    lo, hi = REF['sown_band']
    for s_ in ('abs', 'nir'):
        D[f'sown_share_{s_}'] = D[f'{s_}_area_ha'] / D.luto_area_ha.where(D.luto_area_ha > 0)
        D[f'flag_{s_}'] = np.where(D[f'sown_share_{s_}'] < lo, f'below {lo}', np.where(D[f'sown_share_{s_}'] > hi, f'above {hi}', ''))
        D.loc[D[f'{s_}_area_ha'].isna(), f'flag_{s_}'] = ''
        D.loc[(D[f'{s_}_area_ha'] > 0) & (D.luto_area_ha == 0), f'flag_{s_}'] = 'LUTO area 0'
    C = REF['sown_causes']

    def ok(g, st, src):
        r = D[(D.group == g) & (D.state == st)]
        v = r[f'sown_share_{src}']
        return len(r) == 1 and v.notna().all() and (lo <= float(v.iloc[0]) <= hi)

    def cause(r, src):
        fl = r[f'flag_{src}']
        if not fl:
            return ''
        sown = r[f'{src}_area_ha']
        g, st = r.group, r.state
        if fl == 'LUTO area 0':
            return C['C1a'] if sown < REF['small_ha'] else C['C8a']
        if max(r.luto_area_ha, sown if pd.notna(sown) else 0) < REF['small_ha']:
            return C['C1']
        if src == 'nir' and sown == 0:
            return C['C9']
        if src == 'nir' and g in ('NIR non-irrigated crops', 'NIR irrigated crops') and ok(g + ' (excl. hay)', st, 'nir'):
            return C['C2']
        if g == 'Hay' and fl == f'above {hi}':
            return C['C3']
        if g == 'NIR irrigated pasture':
            return C['C10']
        if g == 'Other non-cereal crops':
            return C['C4']
        if g == 'Cotton, dryland' or (g == 'Cotton' and st == 'NT'):
            return C['C5']
        if g == 'Summer oilseeds' or (g == 'Oilseeds (winter + summer)' and st in ('QLD', 'TAS')):
            return C['C6']
        if g == 'Sugar' and st == 'NSW+ACT' and src == 'nir':
            return C['C7']
        return C['C8']
    for s_ in ('abs', 'nir'):
        D[f'cause_{s_}'] = [cause(r, s_) for _, r in D.iterrows()]
    stat = {(g, st): src for g, st, src in REF['unverified_rows']}
    sr = [stat.get((g, st), '') for g, st in zip(D.group, D.state)]
    D['unverified_row_status'] = [('' if not s_ else (D.at[i, f'cause_{s_}'].split()[0] if D.at[i, f'cause_{s_}'] else 'in range'))
                              for i, s_ in zip(D.index, sr)]
    D['unverified_row_statistic'] = pd.Series(sr, index=D.index).map({'abs': 'ABS 2020-21', 'nir': 'NIR FY2021', '': ''})
    assert (D.unverified_row_statistic != '').sum() == len(REF['unverified_rows'])
    return D


def fix_state_ties(sa2, ST, mask_path):
    """As livestock_map.py: a cell whose SA2 state digit disagrees with state_id takes the SA2 of the nearest cell
    (grid distance) whose SA2 digit equals that state_id and agrees with its own state_id."""
    from scipy.spatial import cKDTree
    with rasterio.open(mask_path) as r:
        mk = r.read(1) == 1
    rows, cols = np.nonzero(mk)
    dig = sa2 // 10**8
    bad = np.where(dig != ST)[0]
    out = sa2.copy()
    for s in np.unique(ST[bad]):
        good = np.where((dig == s) & (ST == s))[0]
        b = bad[ST[bad] == s]
        if len(good) == 0:
            continue
        tree = cKDTree(np.c_[rows[good], cols[good]])
        _, j = tree.query(np.c_[rows[b], cols[b]])
        out[b] = sa2[good[j]]
    return out


# ----------------------------------------------------------------------------- package readers
def read_h5_columns(path, want):
    """Selected columns of a pandas 'table' HDF5 frame with one values block, read in row chunks (the crop tables hold
    300+ float32 columns on 6.96 M rows). Column positions are checked against pandas on the first rows."""
    import tables
    with pd.HDFStore(path, 'r') as s:
        key = s.keys()[0]
    head = pd.read_hdf(path, key=key, stop=1000)
    cols = [tuple(c) for c in head.columns]
    pos = [cols.index(tuple(c)) for c in want]
    with tables.open_file(path) as h:
        t = h.get_node(key + '/table')
        assert t.colnames == ['index', 'values_block_0'], t.colnames
        n = t.nrows
        out = np.empty((n, len(pos)), dtype=head.dtypes.iloc[0])
        step = 400000
        for a in range(0, n, step):
            blk = t.read(a, min(n, a + step), field='values_block_0')
            out[a:a + len(blk)] = blk[:, pos]
        idx = t.read(0, 1000, field='index')
    assert (idx == head.index.to_numpy()).all() and n == NCELL
    assert np.array_equal(out[:1000], head[[tuple(c) for c in want]].to_numpy(), equal_nan=True)
    return {tuple(c): out[:, i] for i, c in enumerate(want)}, cols


def h5_series(path):
    return pd.read_hdf(path).to_numpy()


# ----------------------------------------------------------------------------- GHG arithmetic (offline mirror)
def lvs_type(lu):
    l = lu.lower()
    animal = 'BEEF' if 'beef' in l else 'SHEEP' if 'sheep' in l else 'DAIRY' if 'dairy' in l else None
    veg = 'natural land' if 'natural' in l else 'modified land' if 'modified' in l else None
    if animal is None or veg is None:
        raise KeyError(lu)
    return animal, veg


def split_landuses(landuses):
    crops = [lu for lu in landuses if not any(s in lu for s in ('Beef', 'Sheep', 'Dairy', 'Unallocated', 'Non-agricultural'))]
    lvstk = [lu for lu in landuses if any(s in lu for s in LV)]
    return crops, lvstk


def effective_stubble_rate(rate, agec_lvstk, agghg_lvstk, water_price):
    """data.get_stubble_effective_rate: rate 0 where the sheep per-head data is missing; NaN stays NaN."""
    agghg = agghg_lvstk.loc[:, ('SHEEP', slice(None))]
    sources = [c for c in agghg.columns if c[1] in REF['scope1_lvstk']]
    sheep = (np.isfinite(agec_lvstk.loc[:, [(f, 'SHEEP') for f in REF['stubble_agec_fields']]].to_numpy(dtype=np.float64)).all(axis=1)
             & np.isfinite(agghg[sources].to_numpy(dtype=np.float64)).all(axis=1)
             & np.isfinite(np.asarray(water_price, dtype=np.float64)))
    return np.where(sheep | ~np.isfinite(rate), rate, 0.0)


def yield_pot(g, lu, lm, sel):
    """quantity.get_yield_pot on the cells `sel` (head/ha), climate multiplier 1."""
    animal, veg = lvs_type(lu)
    y = g['k'][animal][sel] * g['pasture'][sel] / (365 * g['dse'][animal] * REF['grassfed'][animal])
    y *= (g['sn'] if veg == 'natural land' else g['sm'])[sel]
    if lm == 'irr':
        y *= 2
    return y


def stubble_head_cell(g, sel):
    return g['k']['SHEEP'][sel] * g['rate'][sel] / (g['dse']['SHEEP'] * REF['grassfed']['SHEEP']) * g['area'][sel]


def div1000(x):
    """x / 1000 as ghg.py computes it on its DataFrames: pandas evaluates frames of 1e6 elements or more with numexpr,
    whose float32 division differs from numpy's in the last bit on about 1 per cent of values. LUTO's frames (every
    agricultural cell) always take that path, so it is used here on the subset of cells too."""
    import numexpr
    x = np.ascontiguousarray(x)
    return numexpr.evaluate('x / 1000')


def ghg_by_state(g):
    """LUTO base-year emissions (t CO2e) by (source, lm, lu, state) on the map, head by (animal, kind, state), and the area
    and NaN-SOIL area by NIR system and state. g: dict of cell arrays and tables (see ghg_inputs).
    Order of rows and operations as the offline mirror of LUTO's GHG arithmetic, evaluated on the allocated cells."""
    LUS, lumap, lmmap, area, state = g['LUS'], g['lumap'], g['lmmap'], g['area'], g['state']
    crops, lvstk = split_landuses(LUS)
    lvcols = {an: [c for c in g['lvstk'].columns if c[0] == an and c[1] in REF['scope1_lvstk']] for an in TYPES}
    ipcols = [c for c in g['irrpast'].columns if 'CO2E' in c]
    E, H, AR = [], [], []

    def add(src, lm, lu, sel, v):
        v = np.nan_to_num(np.asarray(v).astype(np.float64))
        for s, x in pd.Series(v).groupby(state[sel]).sum().items():
            E.append((src, lm, lu, s, x))
    for m, lm in enumerate(LMS):
        for lu in LUS:
            sel = (lumap == LUS.index(lu)) & (lmmap == m)
            if not sel.any():
                continue
            a = area[sel]
            if lu in crops:
                if lu not in g['crop_lus'][lm]:
                    continue
                if (lm, lu) in g['soil']:
                    add('TCO2E_SOIL', lm, lu, sel, div1000(g['soil'][(lm, lu)][sel]) * a)
                if lu in g['lu_stubble']:
                    hc = stubble_head_cell(g, sel)
                    for c in lvcols['SHEEP']:
                        add('STUBBLE_SHEEP_' + c[1].replace('CO2E_KG_HEAD', 'TCO2E'), lm, lu, sel,
                            np.nan_to_num(g['lvstk'][c].to_numpy()[sel] / 1000 * hc))
            elif lu in lvstk:
                an, _ = lvs_type(lu)
                y = yield_pot(g, lu, lm, sel)
                for c in lvcols[an]:
                    add(c[1].replace('CO2E_KG_HEAD', 'TCO2E'), lm, lu, sel, div1000(g['lvstk'][c].to_numpy()[sel] * y) * a)
                if lm == 'irr':
                    for c in ipcols:
                        add(c.replace('CO2E_KG_HA', 'TCO2E'), lm, lu, sel, div1000(g['irrpast'][c].to_numpy()[sel]) * a)
    E = pd.DataFrame(E, columns=['source', 'lm', 'lu', 'state', 't'])
    # head: pasture head per (lu, lm), stubble head per cereal lu (both lm)
    for lu in lvstk:
        an, _ = lvs_type(lu)
        for m, lm in enumerate(LMS):
            sel = (lumap == LUS.index(lu)) & (lmmap == m)
            h = (yield_pot(g, lu, lm, sel) * area[sel]).astype(np.float64)
            H += [(an, 'pasture', k, v) for k, v in pd.Series(h).groupby(state[sel]).sum().items()]
    for lu in g['lu_stubble']:
        sel = lumap == LUS.index(lu)
        hc = stubble_head_cell(g, sel).astype(np.float64)
        H += [('SHEEP', 'stubble', k, v) for k, v in pd.Series(hc).groupby(state[sel]).sum().items()]
    H = pd.DataFrame(H, columns=['animal', 'kind', 'state', 'hd']).groupby(['animal', 'kind', 'state']).hd.sum().reset_index()
    # cropping and irrigated-pasture area by NIR system, and the area of cells with a NaN SOIL factor
    a64 = area.astype(np.float64)
    for lu in LUS:
        for m, lm in enumerate(LMS):
            if 'Unallocated' in lu or (any(x in lu for x in LV) and lm == 'dry'):
                continue
            sel = (lumap == LUS.index(lu)) & (lmmap == m)
            if any(x in lu for x in LV):
                soil = g['irrpast']['CO2E_KG_HA_SOIL'].to_numpy(dtype=float)
            else:
                soil = g['soil'][(lm, lu)].astype(float) if (lm, lu) in g['soil'] else np.full(len(area), np.nan)
            nanc = sel & ~np.isfinite(soil)
            nan_by = pd.Series(a64[nanc]).groupby(state[nanc]).sum()
            AR += [(fert_system(lu, lm), k, v, nan_by.get(k, 0.)) for k, v in pd.Series(a64[sel]).groupby(state[sel]).sum().items()]
    AR = pd.DataFrame(AR, columns=['system', 'state', 'area', 'nan_area']).groupby(['system', 'state'])[['area', 'nan_area']].sum()
    return E, H, AR


def head_cells(g):
    """LUTO head per cell on the map (pasture head of the cell's type, stubble head on cereal cells), float64."""
    LUS, lumap, lmmap, area = g['LUS'], g['lumap'], g['lmmap'], g['area']
    head = np.zeros(len(area))
    stub = np.zeros(len(area))
    typ = np.full(len(area), -1, np.int8)
    for lu in split_landuses(LUS)[1]:
        an, _ = lvs_type(lu)
        for m, lm in enumerate(LMS):
            sel = (lumap == LUS.index(lu)) & (lmmap == m)
            head[sel] = (yield_pot(g, lu, lm, sel) * area[sel]).astype(np.float64)
            typ[sel] = TYPES.index(an)
    for lu in g['lu_stubble']:
        sel = lumap == LUS.index(lu)
        stub[sel] = stubble_head_cell(g, sel).astype(np.float64)
    return head, stub, typ


def fert_system(lu, lm):
    if any(a in lu for a in LV):
        return 'irr_pasture'
    if lu == 'Sugar':
        return 'sugar'
    if lu == 'Cotton':
        return 'cotton'
    if lu in REF['hort_lus']:
        return 'hort'
    return 'irr_crop' if lm == 'irr' else 'dry_crop'


def offland(demand, intensity_csv, year):
    """data.py off-land block for one year: rows with 'Total GHG Emissions (tCO2e)'."""
    D = demand.loc[REF['demand_keys']].copy()
    D.loc['eggs'] = D.loc['eggs'] * REF['egg_g'] / 1000 / 1000
    prod = D.loc[REF['offland_commodities'], 'PRODUCTION'][year]
    dl = pd.DataFrame(dict(COMMODITY=prod.index, YEAR=year, **{'DEMAND (tonnes)': prod.values}))
    it = pd.read_csv(intensity_csv)
    it[['Emission Type', 'Emission Source']] = it['Emission Source'].str.extract(r'^(.*?)\s*\((.*?)\)')
    m = dl.merge(it, on='COMMODITY')
    m['Total GHG Emissions (tCO2e)'] = m.eval('`DEMAND (tonnes)` * `Emission Intensity [ kg CO2eq / kg ]`')
    return m[['YEAR', 'COMMODITY', 'Emission Type', 'Emission Source', 'Total GHG Emissions (tCO2e)']]


# ----------------------------------------------------------------------------- NIR side
def nir_lvstk_state(A, P, fy):
    """NIR emissions (Gg CO2e, AR5) and populations by (type, field, state incl. AUSTRALIA) for one FY (ghg_factors.py
    lvstk_factors)."""
    a = A[(A.gas_id == 103) & (A.fy_label == fy) & A.loc_name.isin(STS + ['AUSTRALIA'])]
    Pf = P[P.fy == fy]
    rows = []
    for an, flds in REF['lvstk_nodes'].items():
        p = Pf[Pf.group == REF['lvstk_group'][an]].groupby('state').value.sum().reindex(STS + ['AUSTRALIA']).fillna(0.)
        for f, nodes in flds.items():
            e = a[a.sector_id.isin(nodes)].groupby('loc_name').value.sum().reindex(STS + ['AUSTRALIA']).fillna(0.)
            for s in STS + ['AUSTRALIA']:
                rows.append(dict(animal=an, field=f, state=s, nir_gg_co2e=e[s], nir_pop=p[s],
                                 kg_co2e_head=(e[s] * 1e6 / p[s]) if p[s] > 0 else np.nan))
    return pd.DataFrame(rows)


def nir_areas(rates_csv):
    """NIR 2024 Vol 2 Table A5.5.8.1 FY2020-21 crop production areas (ha) by (system, state), as ghg_factors.py writes them
    in ghg_crop_rates.csv (column nir_area_ha; its REF table nir_area_2021, read from the Vol 2 PDF, sha256 9d2eab75...,
    pages 200-203). NIR's NSW block is NSW + ACT. Not entered a second time here."""
    d = pd.read_csv(rates_csv)
    d = d[d.nir_area_ha.notna()]
    out = {(r.system, r.state): float(r.nir_area_ha) for r in d.itertuples()}
    assert {k[0] for k in out} == set(SYSTEMS) and {k[1] for k in out} == set(STS) - {'ACT'}, 'NIR area table incomplete'
    return out


def nir_fert_state(A, fy, NA):
    """NIR fertiliser N2O (t CO2e, direct + deposition + leaching) by system and state, the NIR area NA (ACT on the NSW/ACT
    area) and the rate per NIR hectare (ghg_factors.py fert_systems, nir_ha_rates)."""
    a = A[(A.gas_id == 103) & (A.fy_label == fy) & A.loc_name.isin(STS)]

    def g_(nodes):
        return a[a.sector_id.isin(nodes)].groupby('loc_name').value.sum().reindex(STS).fillna(0.) * 1000
    rows = []
    for name, (d, dep, le) in REF['fert_systems'].items():
        ed, edep, ele = g_([d]), g_([dep]), g_([le])
        for s in STS:
            rows.append(dict(system=name, state=s, fert_n2o_t=ed[s] + edep[s] + ele[s],
                             nir_area_ha=NA.get((name, 'NSW' if s == 'ACT' else s), np.nan)))
    M = pd.DataFrame(rows)
    M['fert_n2o_kg_ha_nir'] = np.where(M.nir_area_ha > 0, M.fert_n2o_t * 1000 / M.nir_area_ha, np.nan)
    return M


def nir_national(A, fy, nodes, gas):
    """t CO2e, Australia."""
    return A[A.sector_id.isin(nodes) & (A.gas_id == gas) & (A.fy_label == fy) & (A.loc_name == 'AUSTRALIA')].value.sum() * 1000


# ----------------------------------------------------------------------------- item 3: FY2020-21 rows
def recon_fy2021(E, H, AR, A, P, off, NA):
    HL = H.groupby(['animal', 'state']).hd.sum()
    F = nir_lvstk_state(A, P, FY21)
    F = F[F.state != 'AUSTRALIA']
    R = []
    for _, f in F.iterrows():
        L = E[(E.source == 'TCO2E_' + f.field) & E.lu.str.startswith(f.animal.capitalize()) & (E.state == f.state)].t.sum()
        if f.animal == 'SHEEP':
            L += E[(E.source == 'STUBBLE_SHEEP_TCO2E_' + f.field) & (E.state == f.state)].t.sum()
        N = f.nir_gg_co2e * 1000
        hl = HL.get((f.animal, f.state), 0.)
        fac = f.kg_co2e_head / 1000 if np.isfinite(f.kg_co2e_head) else 0.
        act = (hl - f.nir_pop) * fac
        R.append(dict(row=f'{f.animal} {f.field}', category=REF['category'][f.field], state=f.state, luto_t=L, nir_t=N,
                      diff_t=L - N, activity_t=act, nan_factor_cells_t=0., factor_resid_t=L - N - act, luto_activity=hl,
                      nir_activity=f.nir_pop, unit='head'))
    So = E[E.source == 'TCO2E_SOIL'].copy()
    So['system'] = [fert_system(a, b) for a, b in zip(So.lu, So.lm)]
    M = nir_fert_state(A, FY21, NA)
    for _, r in M[(M.system != 'dry_pasture') & (M.state != 'ACT')].iterrows():
        L = So[(So.system == r.system) & (So.state == r.state)].t.sum()
        N = r.fert_n2o_t
        al = AR.area.get((r.system, r.state), 0.)
        na = r.nir_area_ha if np.isfinite(r.nir_area_ha) else 0.
        rate = r.fert_n2o_kg_ha_nir / 1000 if np.isfinite(r.fert_n2o_kg_ha_nir) else 0.
        act = (al - na) * rate
        nanf = -AR.nan_area.get((r.system, r.state), 0.) * rate
        R.append(dict(row=f'fertiliser N2O {r.system}', category='3.D.a.1 + 3.D.b.1.i + 3.D.b.2.i', state=r.state, luto_t=L,
                      nir_t=N, diff_t=L - N, activity_t=act, nan_factor_cells_t=nanf, factor_resid_t=L - N - act - nanf,
                      luto_activity=al, nir_activity=na, unit='ha'))
    L_act = So[So.state.isin(['ACT', 'OT', 'NA'])].t.sum()
    if L_act:
        R.append(dict(row='fertiliser N2O, ACT/other (NIR in NSW)', category='3.D fertiliser', state='ACT', luto_t=L_act, nir_t=0.,
                      diff_t=L_act, activity_t=L_act, nan_factor_cells_t=0., factor_resid_t=0., luto_activity=np.nan,
                      nir_activity=0., unit='ha'))
    Noff = nir_national(A, FY21, REF['swine_nodes'], 103) + nir_national(A, FY21, REF['poultry_nodes'], 103)
    is_energy = off['Emission Type'].str.contains('energy', na=False)
    Loff = off.loc[~is_energy, 'Total GHG Emissions (tCO2e)'].sum()
    R.append(dict(row='off-land swine + poultry', category='3.A.3, 3.B.3, 3.B.4.g, 3.B.5, 3.D (swine, poultry)', state='AUS',
                  luto_t=Loff, nir_t=Noff, diff_t=Loff - Noff, activity_t=0., nan_factor_cells_t=0., factor_resid_t=Loff - Noff,
                  luto_activity=np.nan, nir_activity=np.nan, unit='t product'))
    X = pd.DataFrame(R)
    X['rel'] = X.diff_t / X.nir_t.replace(0, np.nan)
    X['within_tol'] = X.rel.abs() <= REF['tol_rel']
    X['group'] = np.where(X.row.str.startswith('fertiliser'), 'crop+irr pasture fertiliser N2O',
                          np.where(X.row.str.startswith('off-land'), 'off-land', X.row.str.split().str[0] + ' ' + X.category))
    lvf = ['TCO2E_SOIL', 'TCO2E_ENTERIC', 'TCO2E_MANURE_MGT', 'TCO2E_DUNG_URINE', 'TCO2E_IND_LEACH_RUNOFF']
    ie = E[E.source.str.startswith('TCO2E_') & (E.lm == 'irr') & E.lu.str.contains('Beef|Sheep|Dairy') & ~E.source.isin(lvf)]
    scope = dict(irrpast_energy_lifecycle_t=float(ie.t.sum()), offland_energy_t=float(off.loc[is_energy, 'Total GHG Emissions (tCO2e)'].sum()))
    return X, scope, ie.groupby('state').t.sum()


def category_state(X):
    """Group x state sums, plus Australia per group and the total."""
    cols = ['luto_t', 'nir_t', 'diff_t', 'activity_t', 'nan_factor_cells_t', 'factor_resid_t']
    G = X.groupby(['group', 'state'])[cols].sum()
    G['rel'] = G.diff_t / G.nir_t.replace(0, np.nan)
    G = G.reset_index()
    C = X.groupby('group')[cols].sum().reset_index()
    C['state'] = 'AUS (sum of rows)'
    T = X[cols].sum().to_frame().T
    T['group'] = 'all groups'
    T['state'] = 'AUS (sum of rows)'
    C = pd.concat([C, T], ignore_index=True)
    C['rel'] = C.diff_t / C.nir_t.replace(0, np.nan)
    return pd.concat([G, C], ignore_index=True)


# ----------------------------------------------------------------------------- item 3: FY2010-11
def recon_fy2011(E, H, A, P, off):
    """FY2010-11 reconciliation on the 2010 map with the 2010 factors, national and by state. Per livestock component:
    L5 = L converted to AR5 under the GWP case (CH4 x 28/g_ch4, N2O x 265/g_n2o; the CH4 share of the mapped nodes from the
    national NIR CH4 and N2O, inferred for MANURE_MGT; state rows take the national share); diff = gwp + activity + factor
    + joint exactly, gwp = L - L5, activity = (H_L - H_N) f_N, factor = H_N (f_L5 - f_N), joint = (H_L - H_N)(f_L5 - f_N).
    H_N is the FY2010-11 population of the state (Australia sheet for the national row) and H_L the 2010-map head."""
    Pf = P[P.fy == FY11]
    HL = H[H.kind == 'pasture'].groupby(['animal', 'state']).hd.sum()
    rows = []
    a103 = A[(A.gas_id == 103) & (A.fy_label == FY11)]
    names = {'BEEF': 'Beef', 'DAIRY': 'Dairy', 'SHEEP': 'Sheep'}
    for an, flds in REF['lvstk_nodes'].items():
        pop = Pf[Pf.group == REF['lvstk_group'][an]].groupby('state').value.sum()
        for f, nodes in flds.items():
            ch4, n2o = nir_national(A, FY11, nodes, 108), nir_national(A, FY11, nodes, 109)
            sh = ch4 / (ch4 + n2o) if ch4 + n2o else 0.
            Le = E[E.lu.str.startswith(names[an]) & (E.source == 'TCO2E_' + f)].groupby('state').t.sum()
            Ne = a103[a103.sector_id.isin(nodes)].groupby('loc_name').value.sum() * 1000
            for s in STS + ['AUS']:
                if s == 'AUS':
                    L, N, hl, hn = Le.sum(), ch4 + n2o, HL[an].sum() if an in HL.index.get_level_values(0) else 0., pop.get('AUSTRALIA', 0.)
                else:
                    L, N, hl, hn = Le.get(s, 0.), Ne.get(s, 0.), HL.get((an, s), 0.), pop.get(s, 0.)
                for case, (gc, gn) in REF['gwp_cases'].items():
                    L5 = L * (sh * 28 / gc + (1 - sh) * 265 / gn)
                    fL = L5 / hl if hl else np.nan
                    fN = N / hn if hn else np.nan
                    if hl and hn:
                        act, fac, jnt = (hl - hn) * fN, hn * (fL - fN), (hl - hn) * (fL - fN)
                    else:   # no head on one side: the whole remainder is activity
                        act, fac, jnt = L5 - N, 0., 0.
                    rows.append(dict(case=case, component=f'{f} {an}', state=s, luto_raw_t=L, nir_ar5_t=N, diff_t=L - N,
                                     gwp_t=L - L5, activity_t=act, factor_t=fac, joint_t=jnt, scope_t=0., luto_head=hl,
                                     nir_head=hn, ch4_share=sh))
    # crop and irrigated-pasture SOIL against NIR fertiliser N2O of the crop and irrigated-pasture systems (joint row)
    sysn = [n for k, v in REF['fert_systems'].items() if k != 'dry_pasture' for n in v]
    Ls = E[E.source == 'TCO2E_SOIL'].groupby('state').t.sum()
    Ns = a103[a103.sector_id.isin(sysn)].groupby('loc_name').value.sum() * 1000
    Ns_nat = nir_national(A, FY11, sysn, 109)
    for s in STS + ['AUS']:
        L = Ls.sum() if s == 'AUS' else Ls.get(s, 0.)
        N = Ns_nat if s == 'AUS' else Ns.get(s, 0.)
        for case, (gc, gn) in REF['gwp_cases'].items():
            L5 = L * 265 / gn
            rows.append(dict(case=case, component='Crop + irrigated-pasture SOIL (joint)', state=s, luto_raw_t=L, nir_ar5_t=N,
                             diff_t=L - N, gwp_t=L - L5, activity_t=np.nan, factor_t=np.nan, joint_t=L5 - N, scope_t=0.))
    lvf = ['TCO2E_SOIL', 'TCO2E_ENTERIC', 'TCO2E_MANURE_MGT', 'TCO2E_DUNG_URINE', 'TCO2E_IND_LEACH_RUNOFF']
    Lie = E[E.source.str.startswith('TCO2E_') & ~E.source.isin(lvf)].groupby('state').t.sum()
    Loff = off.groupby('Emission Source')['Total GHG Emissions (tCO2e)'].sum()
    sw, po = nir_national(A, FY11, REF['swine_nodes'], 103), nir_national(A, FY11, REF['poultry_nodes'], 103)
    Lo = Loff.get('CH4', 0.) + Loff.get('N2O', 0.)
    for case in REF['gwp_cases']:
        for s in STS + ['AUS']:
            v = Lie.sum() if s == 'AUS' else Lie.get(s, 0.)
            rows.append(dict(case=case, component='Irrigated-pasture energy fields (scope)', state=s, luto_raw_t=v, nir_ar5_t=0.,
                             diff_t=v, scope_t=v))
        rows.append(dict(case=case, component='Off-land energy CO2 (scope)', state='AUS', luto_raw_t=Loff.get('CO2', 0.),
                         nir_ar5_t=0., diff_t=Loff.get('CO2', 0.), scope_t=Loff.get('CO2', 0.)))
        rows.append(dict(case=case, component='Off-land CH4+N2O (joint, AR6 as filed)', state='AUS', luto_raw_t=Lo,
                         nir_ar5_t=sw + po, diff_t=Lo - sw - po, joint_t=Lo - sw - po))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- output helpers
def write_table(out, name, comment, df, sort, P, inputs):
    """One CSV: a '#' comment row, then the data with sorted rows, floats as %.6f, LF line ends."""
    assert '\n' not in comment and all(ord(c) < 128 for c in comment), name
    df = df.sort_values(sort, kind='mergesort').reset_index(drop=True) if sort else df.reset_index(drop=True)
    df = df.drop(columns=[c for c in ('o', 'ro', 'go') if c in df.columns])     # sort helpers
    p = os.path.join(out, name)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# ' + comment + '\n')
        df.to_csv(f, index=False, lineterminator='\n', float_format='%.6f')
    P['tables'][name] = (inputs, len(df))
    return df


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'numexpr', 'tables', 'openpyxl', 'xlrd', 'rasterio', 'scipy'):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    out['gdal'] = rasterio.__gdal_version__
    return out


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', required=True, help='folder of the inherited 2010-base files listed in INHERITED (input/Nick_build/inherited_2010)')
    ap.add_argument('--base2021', required=True, help='the Nick_build folder holding one output folder per script (input/Nick_build)')
    ap.add_argument('--out', required=True, help='output folder for the tables')
    ap.add_argument('--cache', required=True, help='shared raw-download cache (content-addressed by sha256)')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss')
    ap.add_argument('--live-check', action='store_true', help='also fetch every API response and stop if any differs from its pin')
    ap.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                    help='a local copy of a pinned source (must match its pin); NAME in SOURCES or anga_L<loc>_S<sector>_G<gas>')
    ap.add_argument('--source-dir', default=None, metavar='DIR',
                    help='a folder of local copies of the API responses named L<loc>_S<sector>_G<gas>.json (each must match its pin)')
    a = ap.parse_args()
    t0 = time.time()
    overrides = dict(s.split('=', 1) for s in a.source)
    assert set(overrides) <= set(SOURCES), set(overrides) - set(SOURCES)
    if a.source_dir:
        for n in SOURCES:
            p = os.path.join(a.source_dir, n[len('anga_'):] + '.json')
            if n.startswith('anga_') and n not in overrides and os.path.exists(p):
                overrides[n] = p
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    src = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    if a.live_check:
        live_check([k for k in SOURCES if k.startswith('anga_')])
    print('sources ready', f'{time.time() - t0:.0f}s', flush=True)
    I = {f: os.path.join(a.input, f) for f in INHERITED}
    B = {k: os.path.join(a.base2021, *v[0].split('/')) for k, v in BASE2021.items()}
    in_sha = {('input', f): sha_file(p) for f, p in I.items()}
    in_sha.update({('base2021', k): sha_file(p) for k, p in B.items()})
    PV = dict(tables={})
    fi = lambda *fs: [('input', f) for f in fs]
    fb = lambda *fs: [('base2021', f) for f in fs]
    fs_ = lambda *fs: [('source', f) for f in fs]
    anga = [k for k in SOURCES if k.startswith('anga_')]

    # ---- common arrays
    LUS = pd.read_csv(I['ag_landuses.csv'], header=None)[0].tolist()
    area = pd.read_hdf(I['real_area.h5']).to_numpy() * 1                  # float32, RESMULT 1
    ST = np.load(I['state_id.npy']).astype(np.int64)
    state = pd.Series(ST).map(ST_NAME).to_numpy().astype('U3')
    lu10, lm10 = h5_series(I['lumap.h5']), h5_series(I['lmmap.h5'])
    lu21, lm21 = h5_series(B['lumap.h5']), h5_series(B['lmmap.h5'])
    assert len(area) == len(lu10) == len(lu21) == NCELL
    A = anga_frame({k: src[k] for k in anga})
    P = nir_populations(src['nir_livestock'], (FY11, FY21))
    sn, sm = h5_series(I['safe_pur_natl.h5']), h5_series(I['safe_pur_modl.h5'])
    demand = pd.read_hdf(I['demand_projections.h5'])

    # ---- 2021-base GHG inputs
    dse_t = pd.read_csv(B['livestock_dse_ratings.csv'], float_precision='round_trip').set_index('animal').intake_kg_dm_day
    lv21 = pd.read_hdf(B['agGHG_lvstk.h5'])
    ip21 = pd.read_hdf(B['agGHG_irrpast.h5'])
    agl = pd.read_hdf(B['agec_lvstk.h5'])
    wp = h5_series(B['water_delivery_price.h5'])
    stub = h5_series(B['stubble_dse_ha.h5'])
    k21 = pd.read_hdf(B['lvstk_k.h5'])
    cr21_cols = [tuple(c) for c in pd.read_hdf(B['agGHG_crops.h5'], stop=1).columns]
    soil21, _ = read_h5_columns(B['agGHG_crops.h5'], [c for c in cr21_cols if c[0] == 'CO2E_KG_HA_SOIL'])
    g21 = dict(LUS=LUS, lumap=lu21, lmmap=lm21, area=area, state=state, lvstk=lv21, irrpast=ip21,
               soil={(c[1], c[2]): v for c, v in soil21.items()},
               crop_lus={lm: {c[2] for c in cr21_cols if c[0] == 'CO2E_KG_HA_CHEM_APPL' and c[1] == lm} for lm in LMS},
               k={t: np.nan_to_num(k21[t].to_numpy()) for t in TYPES}, pasture=h5_series(B['pasture_kg_dm_ha.h5']), sn=sn, sm=sm,
               rate=np.nan_to_num(effective_stubble_rate(stub, agl, lv21, wp)), dse={t: float(dse_t[t]) for t in TYPES},
               lu_stubble=list(REF['stubble_lus']))
    E21, H21, AR21 = ghg_by_state(g21)
    off21 = offland(demand, B['agGHG_lvstk_off_land.csv'], 2021)
    NA = nir_areas(B['ghg_crop_rates.csv'])
    X21, scope21, ie21 = recon_fy2021(E21, H21, AR21, A, P, off21, NA)
    print('FY2021 emissions done', f'{time.time() - t0:.0f}s', flush=True)

    # ---- 2010-base GHG inputs
    lv10 = pd.read_hdf(I['agGHG_lvstk.h5'])
    ip10 = pd.read_hdf(I['agGHG_irrpast.h5'])
    cr10_cols = [tuple(c) for c in pd.read_hdf(I['agGHG_crops.h5'], stop=1).columns]
    soil10, _ = read_h5_columns(I['agGHG_crops.h5'], [c for c in cr10_cols if c[0] == 'CO2E_KG_HA_SOIL'])
    feed = np.nan_to_num(h5_series(I['feed_req.h5']))
    g10 = dict(LUS=LUS, lumap=lu10, lmmap=lm10, area=area, state=state, lvstk=lv10, irrpast=ip10,
               soil={(c[1], c[2]): v for c, v in soil10.items()},
               crop_lus={lm: {c[2] for c in cr10_cols if c[0] == 'CO2E_KG_HA_CHEM_APPL' and c[1] == lm} for lm in LMS},
               k={t: feed for t in TYPES}, pasture=h5_series(I['pasture_kg_dm_ha.h5']), sn=sn, sm=sm, rate=None,
               dse=dict(REF['dse_2010']), lu_stubble=[])
    E10, H10, _ = ghg_by_state(g10)
    off10 = offland(demand, I['agGHG_lvstk_off_land.csv'], 2010)
    X11 = recon_fy2011(E10, H10, A, P, off10)
    print('FY2011 emissions done', f'{time.time() - t0:.0f}s', flush=True)

    # ================================================================ item 1: livestock head
    nir_h = nir_state_heads(src['nir_livestock'])
    abs_h = abs_state_livestock(src['abs_7121_2020_21'])
    head21, stubh21, typ21 = head_cells(g21)
    rec = pd.read_csv(B['livestock_recon_state_type.csv'])
    named = [c for c in rec.columns if c.startswith('row: ')]
    T1 = []
    for _, r in rec.iterrows():
        ti = ['beef', 'sheep', 'dairy'].index(r.type)
        m = ST == r.state
        hp = head21[m & (typ21 == ti)].sum()
        hs = stubh21[m].sum() if r.type == 'sheep' else 0.
        d = dict(state=ST_NAME[r.state], type=r.type, nir_head_fy2021=nir_h.get((r.state, r.type), 0.),
                 abs_head_2020_21=abs_h.get((r.state, r.type), 0.), map_pasture_head=hp, map_stubble_head=hs)
        for c in named:
            d[c] = r[c]
        d['closure_nir_minus_map_minus_named_rows'] = d['nir_head_fy2021'] - hp - hs - sum(r[c] for c in named)
        d['recon_allocated_pasture_head'] = r.allocated_pasture_head
        d['recon_allocated_stubble_head'] = r.allocated_stubble_head
        d['map_minus_recon_pasture_head'] = hp - r.allocated_pasture_head
        d['map_minus_recon_stubble_head'] = hs - r.allocated_stubble_head
        d['recon_nir_state_total'] = r.nir_state_total
        T1.append(d)
    T1 = pd.DataFrame(T1)
    tot = T1.drop(columns=['state']).groupby('type').sum().reset_index()
    tot['state'] = 'AUS'
    T1 = pd.concat([T1, tot], ignore_index=True)
    T1['nir_over_abs'] = T1.nir_head_fy2021 / T1.abs_head_2020_21.where(T1.abs_head_2020_21 > 0)
    T1['map_head_over_nir'] = (T1.map_pasture_head + T1.map_stubble_head) / T1.nir_head_fy2021.where(T1.nir_head_fy2021 > 0)
    T1['state_order'] = T1.state.map({v: k for k, v in ST_NAME.items()}).fillna(99)
    T1 = write_table(a.out, 't1_livestock_head_state_type.csv',
                     'Item 1. Livestock head (head) by state and type on the 2021-base map (LUTO head = k x pasture x safe '
                     'utilisation x area x (2 if irrigated) / (365 x DSE x grassfed), NIR-intake DSE, climate multiplier 1; '
                     'stubble head on cereal cells at the effective stubble rate) against NIR 2024 activity table FY2020-21 '
                     'state populations and ABS 7121.0 2020-21 state head; named rows from livestock_recon_state_type.csv '
                     '(livestock_map.py) so that NIR = map pasture head + map stubble head + named rows + closure; recon_* are '
                     'the reconciliation values for the map_minus_recon check.',
                     T1.drop(columns='state_order').assign(o=T1.state_order), ['o', 'type'], PV,
                     fb('lumap.h5', 'lmmap.h5', 'lvstk_k.h5', 'pasture_kg_dm_ha.h5', 'stubble_dse_ha.h5', 'agec_lvstk.h5',
                        'agGHG_lvstk.h5', 'water_delivery_price.h5', 'livestock_dse_ratings.csv', 'livestock_recon_state_type.csv')
                     + fi('real_area.h5', 'state_id.npy', 'safe_pur_natl.h5', 'safe_pur_modl.h5', 'ag_landuses.csv')
                     + fs_('nir_livestock', 'abs_7121_2020_21'))
    print('item 1 done', f'{time.time() - t0:.0f}s', flush=True)

    # ================================================================ item 2: land use
    cats = {}
    maps = {'2010_map': (lu10, lm10, None),
            '2010_11_control': (np.load(B['class_alloc_2010_11_lumap.npy']), np.load(B['class_alloc_2010_11_lmmap.npy']),
                                np.load(B['class_alloc_2010_11_category.npy'])),
            '2020_21_before_typing': (np.load(B['class_alloc_2020_21_lumap.npy']), np.load(B['class_alloc_2020_21_lmmap.npy']),
                                      np.load(B['class_alloc_2020_21_category.npy'])),
            '2021_map': (lu21, lm21, None)}
    a64 = area.astype(np.float64)
    parts = []
    for name, (lu_, lm_, cat_) in maps.items():
        lu_ = lu_.astype(np.int64)
        lab = np.where(lu_ < 0, 'Non-agricultural land', np.array(LUS, dtype=object)[np.maximum(lu_, 0)])
        code = lu_.astype(str).astype(object)
        if cat_ is not None:
            for c, nm in REF['pseudo'].items():
                pm = cat_ == c
                lab[pm] = nm
                code[pm] = 'G-' + nm.split(' - ')[1].split()[0][:3]
        df = pd.DataFrame(dict(state=state, lu_code=code, land_use=lab, lm=np.where(lm_ == 1, 'irr', 'dry'), area=a64))
        g = df.groupby(['state', 'lu_code', 'land_use', 'lm']).area.agg(['sum', 'size'])
        g.columns = [f'area_{name}_ha', f'cells_{name}']
        parts.append(g)
    T2 = pd.concat(parts, axis=1).fillna(0.).reset_index()
    graze = {'natural': ['Beef - natural land', 'Sheep - natural land', 'Dairy - natural land', 'Grazing untyped - natural land'],
             'modified': ['Beef - modified land', 'Sheep - modified land', 'Dairy - modified land', 'Grazing untyped - modified land']}
    agg = []
    for k_, lst in graze.items():
        s_ = T2[T2.land_use.isin(lst)].groupby(['state', 'lm']).sum(numeric_only=True).reset_index()
        s_['lu_code'] = 'agg'
        s_['land_use'] = f'All grazing - {k_} land (aggregate row)'
        agg.append(s_)
    T2 = pd.concat([T2] + agg, ignore_index=True)
    nat = T2.groupby(['lu_code', 'land_use', 'lm']).sum(numeric_only=True).reset_index()
    nat['state'] = 'AUS'
    T2 = pd.concat([T2, nat], ignore_index=True)
    T2['method_change_ha'] = T2['area_2010_11_control_ha'] - T2['area_2010_map_ha']
    T2['landuse_change_ha'] = T2['area_2020_21_before_typing_ha'] - T2['area_2010_11_control_ha']
    T2['typing_change_ha'] = T2['area_2021_map_ha'] - T2['area_2020_21_before_typing_ha']
    T2['total_change_ha'] = T2['area_2021_map_ha'] - T2['area_2010_map_ha']
    for c in [c for c in T2.columns if c.startswith('cells_')]:
        T2[c] = T2[c].astype(np.int64)
    T2['o'] = T2.state.map({v: k for k, v in ST_NAME.items()}).fillna(99)
    T2 = write_table(a.out, 't2_landuse_area_state_class.csv',
                     'Item 2. Land-use area (ha, real_area) and cells by LUTO class (lu, lm) and state: the 2010 map (input '
                     'lumap/lmmap), the NLUM v7.1 2010-11 control and the 2020-21 class allocation as nlum_map.py leaves them '
                     '(grazing cells not yet typed: categories 12 and 13 reported as Grazing untyped - natural land, 22 and 23 as '
                     'Grazing untyped - modified land; every other cell under its lu), and the 2021-base map (livestock_map.py). '
                     'method_change = 2010-11 control minus 2010 map (same year, different method); landuse_change = 2020-21 '
                     'minus 2010-11 (same method, different year); typing_change = 2021 map minus 2020-21 before typing; the '
                     'aggregate grazing rows add typed and untyped grazing so the method and land-use changes compare like with '
                     'like. State = state_id.',
                     T2, ['o', 'state', 'lu_code', 'land_use', 'lm'], PV,
                     fi('lumap.h5', 'lmmap.h5', 'real_area.h5', 'state_id.npy', 'ag_landuses.csv')
                     + fb('lumap.h5', 'lmmap.h5', 'class_alloc_2010_11_lumap.npy', 'class_alloc_2010_11_lmmap.npy',
                          'class_alloc_2010_11_category.npy', 'class_alloc_2020_21_lumap.npy', 'class_alloc_2020_21_lmmap.npy',
                          'class_alloc_2020_21_category.npy'))
    print('item 2 done', f'{time.time() - t0:.0f}s', flush=True)

    # ================================================================ item 3: emissions tables
    in3_21 = (fb('lumap.h5', 'lmmap.h5', 'agGHG_crops.h5', 'agGHG_lvstk.h5', 'agGHG_irrpast.h5', 'agGHG_lvstk_off_land.csv',
                 'lvstk_k.h5', 'pasture_kg_dm_ha.h5', 'stubble_dse_ha.h5', 'agec_lvstk.h5', 'water_delivery_price.h5',
                 'livestock_dse_ratings.csv', 'ghg_crop_rates.csv')
              + fi('real_area.h5', 'state_id.npy', 'safe_pur_natl.h5', 'safe_pur_modl.h5', 'ag_landuses.csv',
                   'demand_projections.h5') + fs_('nir_livestock') + [('source', k) for k in anga])
    X21o = X21.copy()
    for k_, v in scope21.items():
        X21o = pd.concat([X21o, pd.DataFrame([dict(row=f'scope: {k_}', category='not in the NIR comparison (LUTO scope rows)',
                                                    state='AUS', luto_t=v, group='scope')])], ignore_index=True)
    X21o['o'] = X21o.state.map({**{s: i for i, s in enumerate(STS)}, 'AUS': 98}).fillna(99)
    X21o['ro'] = np.arange(len(X21o))
    write_table(a.out, 't3a_ghg_fy2021_rows.csv',
                'Item 3, FY2020-21 rows. LUTO base-year emissions (t CO2e) on the 2021-base map with the 2021-base factors '
                '(offline mirror of ghg.py, scope 1, RESFACTOR 1, NIR-intake DSE, climate multiplier 1) against NIR 2024 '
                'FY2020-21 (greenhouse accounts API, AR5, gas 103), by row and state; attribution: '
                'activity = (LUTO activity - NIR activity) x NIR factor (head x kg CO2e/head; ha x NIR '
                'fertiliser N2O per NIR ha, Vol 2 Table A5.5.8.1 areas), nan_factor_cells = area with a NaN SOIL factor x rate, '
                'factor_resid = the rest; off-land national against NIR swine + poultry nodes; scope rows are LUTO-only. '
                'within_tol: |rel| <= 0.034.',
                X21o, ['o', 'ro'], PV, in3_21)
    T3b = category_state(X21)
    T3b['o'] = T3b.state.map({**{s: i for i, s in enumerate(STS)}}).fillna(99)
    write_table(a.out, 't3b_ghg_fy2021_category_state.csv',
                'Item 3, FY2020-21 by NIR category group and state (sums of t3a rows; '
                't CO2e, AR5). Groups: <TYPE> <NIR category> for the livestock fields, crop+irr pasture fertiliser N2O, '
                'off-land (national); AUS rows are the sums of the rows.',
                T3b, ['group', 'o', 'state'], PV, in3_21)
    in3_11 = (fi('lumap.h5', 'lmmap.h5', 'agGHG_crops.h5', 'agGHG_lvstk.h5', 'agGHG_irrpast.h5', 'agGHG_lvstk_off_land.csv',
                 'feed_req.h5', 'pasture_kg_dm_ha.h5', 'real_area.h5', 'state_id.npy', 'safe_pur_natl.h5', 'safe_pur_modl.h5',
                 'ag_landuses.csv', 'demand_projections.h5') + fs_('nir_livestock') + [('source', k) for k in anga])
    X11['o'] = X11.state.map({**{s: i for i, s in enumerate(STS)}, 'AUS': 98})
    write_table(a.out, 't3c_ghg_fy2011.csv',
                'Item 3, FY2010-11. LUTO base-year emissions (t CO2e) on the 2010 map with the 2010 factors (offline mirror of '
                'ghg.py, scope 1, RESFACTOR 1, DSE per head 8 / 1.5 / 17, k = FEED_REQ, no stubble, climate multiplier 1; '
                'off-land at LUTO year 2010) against NIR 2024 FY2010-11 (API, AR5), by component and state, '
                'decomposition under three GWP readings of the 2010 factors (case): diff = gwp + activity + '
                'factor + joint (+ scope); nir_head = NIR FY2010-11 population of the state (Australia sheet for AUS), '
                'luto_head = 2010-map head; CH4 share from national NIR CH4 and N2O, used for the state rows too (inferred).',
                X11, ['case', 'component', 'o'], PV, in3_11)
    print('item 3 done', f'{time.time() - t0:.0f}s', flush=True)

    # ================================================================ item 4: crops
    ag_cols = [tuple(c) for c in pd.read_hdf(B['agec_crops.h5'], stop=1).columns]
    want = [c for c in ag_cols if c[0] in ('Yield', 'P1') or (c[0] == 'WR' and c[1] == 'irr')]
    AGC, _ = read_h5_columns(B['agec_crops.h5'], want)
    crops = [c for c in LUS if not any(k_ in c for k_ in ('Beef', 'Sheep', 'Dairy', 'Unallocated'))]
    a64 = area.astype(float)
    D = []
    for c in crops:
        j = LUS.index(c)
        for m, lm in enumerate(LMS):
            if ('Yield', lm, c) not in AGC:
                continue
            msk = (lu21 == j) & (lm21 == m)
            y, p1 = AGC[('Yield', lm, c)], AGC[('P1', lm, c)]
            for s in np.unique(state[msk]):
                k_ = msk & (state == s)
                D.append(dict(lu=c, lm=lm, state=s, ha=a64[k_].sum(), prod_t=np.nansum(y[k_] * a64[k_]),
                              value_aud=np.nansum(y[k_] * p1[k_] * a64[k_]), p1_area_sum=np.nansum(p1[k_].astype(float) * a64[k_]),
                              nan_alloc_cells=int(np.isnan(y[k_]).sum())))
    D = pd.DataFrame(D)
    S4 = D.groupby(['lu', 'state'])[['ha', 'prod_t', 'value_aud', 'p1_area_sum', 'nan_alloc_cells']].sum().reset_index()
    win = pd.read_csv(B['crop_production_window.csv'])
    S4 = S4.merge(win[['lu', 'state', 'q_mean', 'v_mean']], on=['lu', 'state'], how='outer')
    S4['prod_ratio'] = S4.prod_t / S4.q_mean
    S4['value_ratio'] = S4.value_aud / S4.v_mean
    S4['p1_area_weighted'] = S4.p1_area_sum / S4.ha.where(S4.ha > 0)
    in4 = (fb('lumap.h5', 'lmmap.h5', 'agec_crops.h5', 'crop_production_window.csv', 'crop_yield_calibration.csv')
           + fi('real_area.h5', 'state_id.npy', 'ag_landuses.csv'))
    S4o = S4.drop(columns='p1_area_sum')
    write_table(a.out, 't4a_crops_by_crop_state.csv',
                'Item 4. Crop production (t) and farm-gate value (FY2021 $) by crop and state on the 2021-base map (sum over '
                'allocated cells of Yield x area and Yield x P1 x area, agec_crops.h5, both lm; ha = map area) against the ABS '
                'window means crop_economics.py uses (q_mean t, v_mean $, crop_production_window.csv: FY2017-25 production and '
                'value windows); p1_area_weighted = P1 '
                'area-weighted over the allocated cells ($/t); a ratio below 1 is production not placed, above 1 over-placed.',
                S4o, ['lu', 'state'], PV, in4)
    cal = pd.read_csv(B['crop_yield_calibration.csv'])
    N4 = cal.merge(S4[['lu', 'state', 'prod_t', 'ha', 'p1_area_weighted', 'q_mean', 'v_mean']], on=['lu', 'state'], how='left')
    N4 = N4.merge(win[['lu', 'state', 'p_farmgate_fy2021']], on=['lu', 'state'], how='left')
    N4['t_not_placed'] = N4.Q_abs_t - N4.prod_t.fillna(0.)
    N4['unit_value_window'] = np.where(N4.p_farmgate_fy2021.notna(), N4.p_farmgate_fy2021, N4.p1_area_weighted)
    N4['value_at_window_price_aud'] = N4.t_not_placed * N4.unit_value_window
    N4['value_at_p1_aud'] = N4.t_not_placed * N4.p1_area_weighted
    N4['pct_of_abs_t'] = np.where(N4.Q_abs_t > 0, N4.t_not_placed / N4.Q_abs_t * 100, np.nan)
    natn = N4.groupby('lu')[['t_not_placed', 'Q_abs_t']].sum()
    share = (natn.t_not_placed / natn.Q_abs_t).rename('crop_national_not_placed_share')
    N4 = N4.merge(share.reset_index(), on='lu', how='left')
    N4['named_crop'] = [any(c == lu_ and (s_ is None or s_ == st_) for c, s_ in REF['named_crops']) for lu_, st_ in zip(N4.lu, N4.state)]
    N4['named_row'] = N4.named_crop & (N4.t_not_placed.abs() > REF['row_t'])
    N4['over_placed'] = (N4.prod_t.fillna(0.) - N4.Q_abs_t) > REF['row_t']
    N4['mechanism'] = np.where(N4.over_placed & N4.banded & (N4.K < N4.K_national), 'lower yield cap',
                               np.where(N4.over_placed & (N4.Q_abs_t <= 0), 'no ABS production: national factor on mapped area',
                                        np.where(N4.over_placed, 'other', '')))
    N4['label_named'] = np.where(N4.label.notna(), N4.label,
                                 np.where(N4.named_row & (N4.ha.fillna(0) == 0), REF['nomap_label'],
                                          np.where(N4.named_row, REF['named_crop_label'], '')))
    N4 = N4[N4.named_row | N4.over_placed | N4.label.notna()]
    cols4 = ['lu', 'state', 'Q_abs_t', 'map_alloc_ha', 'ha', 'K', 'K_national', 'K_used', 'banded', 'placed_t', 'prod_t',
             't_not_placed', 'pct_of_abs_t', 'unit_value_window', 'value_at_window_price_aud', 'p1_area_weighted',
             'value_at_p1_aud', 'crop_national_not_placed_share', 'named_crop',
             'named_row', 'over_placed', 'mechanism', 'label', 'label_named']
    N4 = write_table(a.out, 't4b_crops_named_rows.csv',
                     'Item 4, named rows. Every named (crop, state) row: under-mapped crops (the crops named, '
                     'Tasmanian vegetables, summer legumes, tropical stone fruit, apples, pears, other non-cereal crops and '
                     'summer oilseeds; state rows with |t_not_placed| > 0.5 t), over-placed rows (over_placed: placed on the map '
                     'above the ABS mean by > 0.5 t, with the cause in mechanism), their labels (from '
                     'crop_yield_calibration.csv), and every row that file labels. t_not_placed = ABS window mean - tonnes on '
                     'the map (negative = over-placed); value_at_window_price = t x the window farm-gate unit value (FY2021 $/t, '
                     'P1 area-weighted where the window has none); value_at_p1 = t x P1 '
                     'area-weighted on the allocated cells; placed_t, K, K_national, K_used, banded, '
                     'label from crop_yield_calibration.csv (crop_economics.py).',
                     N4[cols4], ['lu', 'state'], PV, in4)
    OS = tas_other_oilseeds(src)
    OSw = OS.pivot_table(index='fy_end', columns='measure', values='value', aggfunc='sum').reset_index()
    OSw['source_files'] = OS.groupby('fy_end').source_file.apply(lambda s: '; '.join(sorted(set(s)))).to_numpy()
    OSw = pd.concat([OSw, pd.DataFrame([dict(fy_end='mean FY2017-21', **OSw[OSw.fy_end.between(2017, 2021)][
        ['area_ha', 'local_value_aud', 'production_t']].mean().to_dict(), source_files='mean of the rows above')])], ignore_index=True)
    OSw['fy_end'] = OSw.fy_end.astype(str)
    write_table(a.out, 't4c_tas_other_oilseeds.csv',
                'Item 4. ABS Tasmanian "Other oilseeds" (crops not itemised): production (t), area (ha) and local '
                'value ($, nominal) by financial year (fy_end = year the FY ends), from the ABS 7121.0 and 7503.0 state files; '
                'no LUTO crop group takes it (summer oilseeds = ABARES sunflower + soybeans, 0 t in TAS; winter oilseeds = '
                'canola); not modelled.',
                OSw, ['fy_end'], PV, fs_(*sorted({f for _, f, _ in REF['oilseed_files']})))
    held21 = np.load(B['class_alloc_2020_21_category.npy']) == 51
    T4d = sown_share(lu21, lm21, ST, area, held21, src['abs_7121_2020_21'], NA)
    T4d['go'] = T4d.group.map({g: i for i, g in enumerate(REF['sown_groups'])})
    T4d['o'] = T4d.state.map({**{v: k for k, v in ST_NAME.items()}, 'AUS': 9, 'NSW+ACT': 10})
    T4d = write_table(a.out, 't4d_sown_share_group_state.csv',
                      'Item 4, sown share. Sown share = sown area / LUTO cropping-land area on the 2021-base map '
                      '(lumap, lmmap; real_area ha; luto_held_ha = cells of class-allocation category 51, held at the 2010 map) by '
                      'crop group and state, against ABS 7121.0 2020-21 Table 1 state areas (abs_np_entries = np cells) and NIR 2024 '
                      'Vol 2 Table A5.5.8.1 FY2020-21 areas (as ghg_factors.py carries them in ghg_crop_rates.csv; the NSW block is '
                      'NSW + ACT, so NSW and ACT alone have none). Flags below 0.4 or above 1.05; cause codes C1 to C10 by the rules '
                      'in the reference table (sown_causes); unverified_row_* mark the 14 rows of unverified cause '
                      'and their status on this map (cause code of the named statistic, or in range).',
                      T4d, ['go', 'o'], PV,
                      fb('lumap.h5', 'lmmap.h5', 'class_alloc_2020_21_category.npy', 'ghg_crop_rates.csv')
                      + fi('real_area.h5', 'state_id.npy') + fs_('abs_7121_2020_21'))
    print('item 4 done', f'{time.time() - t0:.0f}s', flush=True)

    # ================================================================ item 5: water
    dd = h5_series(I['draindiv_id.h5']).ravel()
    lut = pd.read_hdf(I['draindiv_lut.h5'])
    DN = dict(zip(lut.HR_DRAINDIV_ID.astype(int), lut.HR_DRAINDIV_NAME.astype(str)))
    cw = np.zeros(NCELL)
    nanha = np.zeros(NCELL)
    for j, c in enumerate(LUS):
        if ('WR', 'irr', c) not in AGC:
            continue
        m = (lu21 == j) & (lm21 == 1)
        wr = AGC[('WR', 'irr', c)]
        cw[m] = np.nan_to_num(wr[m]) * a64[m]
        nanha[m] = np.where(np.isnan(wr[m]), a64[m], 0)
    pw = np.zeros(NCELL)
    irrl = (lm21 == 1) & (typ21 >= 0)
    for i, t in enumerate(TYPES):
        m = irrl & (typ21 == i)
        pw[m] = np.nan_to_num(head21[m] * agl[('WR_IRR', t)].to_numpy()[m].astype(float))
    W = pd.DataFrame(dict(d=dd, crop=cw, past=pw, nanha=nanha)).groupby('d').sum()
    # ABS 4618.0 2020-21 SA2 values apportioned to divisions
    wq = pd.read_excel(src['WUAFDCASGS202021.xlsx'], 'Table 1', header=6)
    wq.columns = ['code', 'label', 'ccode', 'desc', 'est', 'rse', 'nbus', 'nbus_rse']
    wq['code'] = wq.code.astype(str)
    wq['v'] = pd.to_numeric(wq.est, errors='coerce')
    IT = REF['water_items']
    sa2r, natr = wq[wq.code.str.len() == 9], wq[wq.code == '0']
    Wsa2 = sa2r[sa2r.ccode.isin(IT.values())].pivot_table(index='code', columns='ccode', values='v', aggfunc='sum').rename(
        columns={v: k for k, v in IT.items()})
    Wsa2.index = Wsa2.index.astype(np.int64)
    NATW = {k: float(natr.loc[natr.ccode == v, 'v'].iloc[0]) for k, v in IT.items()}
    s21 = fix_state_ties(np.load(B['sa2_2021_cells.npy']).astype(np.int64), ST, I['NLUM_2010-11_mask.tif'])
    irr_c, fp_c = (lu21 >= 0) & (lm21 == 1), lu21 >= 0

    def shares(mask):
        t = pd.DataFrame({'s': s21[mask], 'd': dd[mask], 'a': a64[mask]}).groupby(['s', 'd']).a.sum()
        return t / t.groupby(level=0).transform('sum')
    Sh = [shares(irr_c), shares(fp_c), shares(np.ones(NCELL, bool))]
    rows, split_vol = [], 0.
    basis = {'irrigated cells': 0., 'footprint cells': 0., 'all cells': 0., 'no cell (SA2 not on grid)': 0.}
    for code_, r in Wsa2.iterrows():
        for nm, sh in zip(('irrigated cells', 'footprint cells', 'all cells'), Sh):
            if code_ in sh.index.get_level_values(0):
                w = sh.loc[code_]
                basis[nm] += np.nan_to_num(r.total)
                break
        else:
            basis['no cell (SA2 not on grid)'] += np.nan_to_num(r.total)
            continue
        if len(w) > 1:
            split_vol += np.nan_to_num(r.total)
        for dv, f in w.items():
            rows.append(dict(draindiv_id=int(dv), **{k: np.nan_to_num(r[k]) * f for k in IT}))
    RW = pd.DataFrame(rows).groupby('draindiv_id')[list(IT)].sum()
    RW['crops_hort'] = RW.total - RW.pasture_grazed - RW.hay_silage
    ids = sorted(set(RW.index) | set(int(i) for i in W.index if i in DN))
    T5 = pd.DataFrame(dict(row_kind='drainage division', draindiv_id=ids, draindiv=[DN.get(i, str(i)) for i in ids]))
    T5['luto_crops_ML'] = W.crop.reindex(ids).fillna(0).to_numpy()
    T5['luto_pasture_ML'] = W.past.reindex(ids).fillna(0).to_numpy()
    T5['luto_total_ML'] = T5.luto_crops_ML + T5.luto_pasture_ML
    for k_ in ('total', 'pasture_grazed', 'hay_silage', 'crops_hort'):
        T5[f'abs_2020_21_{k_}_ML'] = RW[k_].reindex(ids).fillna(0).to_numpy()
    T5['luto_over_abs_total'] = T5.luto_total_ML / T5.abs_2020_21_total_ML.where(T5.abs_2020_21_total_ML > 0)
    T5['irr_area_without_WR_ha'] = W.nanha.reindex(ids).fillna(0).to_numpy()
    sumc = T5.drop(columns=['row_kind', 'draindiv_id', 'draindiv', 'luto_over_abs_total']).sum()
    nat5 = [dict(row_kind='Australia', draindiv_id=0, draindiv='Australia (sum of divisions; ABS = SA2 sum)', **sumc.to_dict(),
                 luto_over_abs_total=sumc.luto_total_ML / sumc.abs_2020_21_total_ML)]
    nat5.append(dict(row_kind='ABS national', draindiv_id=0, draindiv='ABS 4618.0 2020-21 national total (Australia row)',
                     abs_2020_21_total_ML=NATW['total'], abs_2020_21_pasture_grazed_ML=NATW['pasture_grazed'],
                     abs_2020_21_hay_silage_ML=NATW['hay_silage'],
                     abs_2020_21_crops_hort_ML=NATW['total'] - NATW['pasture_grazed'] - NATW['hay_silage'],
                     luto_total_ML=sumc.luto_total_ML, luto_over_abs_total=sumc.luto_total_ML / NATW['total']))
    nat5.append(dict(row_kind='named row', draindiv_id=0, draindiv='ABS national total minus SA2 sum (suppressed np cells, '
                     'rounding)', abs_2020_21_total_ML=NATW['total'] - Wsa2.total.sum()))
    cwc = pd.read_csv(B['crop_water_calibration.csv'])
    lwh = pd.read_csv(B['livestock_water_per_head.csv'])
    v_crop = cwc.V_abs_ml.sum()
    v_past = lwh.drop_duplicates('state').V_ml.sum()
    nat5.append(dict(row_kind='ABS window FY2017-21', draindiv_id=0,
                     draindiv='ABS 4618.0 FY2017-21 mean, crop groups (crop_water_calibration.csv) + grazed pasture '
                     '(livestock_water_per_head.csv), the windows the 2021-base water was built on',
                     abs_2020_21_total_ML=np.nan, luto_crops_ML=cwc.placed_ml.sum(), luto_total_ML=sumc.luto_total_ML,
                     window_fy2017_21_crops_ML=v_crop, window_fy2017_21_pasture_ML=v_past,
                     window_fy2017_21_total_ML=v_crop + v_past, luto_over_window_total=sumc.luto_total_ML / (v_crop + v_past)))
    mq = pd.read_excel(src['WUAFDCMDB202021.xlsx'], 'Table 1', header=6)
    mq.columns = ['code', 'label', 'ccode', 'desc', 'est', 'nbus']
    mdb = float(pd.to_numeric(mq[(mq.label == 'Murray Darling Basin') & (mq.ccode == IT['total'])].est).iloc[0])
    nat5.append(dict(row_kind='check', draindiv_id=7, draindiv='MDB cube total against the SA2-apportioned division 7',
                     abs_2020_21_total_ML=mdb, abs_mdb_division7_sa2_apportioned_ML=float(RW.total.get(7, 0.))))
    nat5.append(dict(row_kind='check', draindiv_id=0, draindiv='ABS SA2 volume split across divisions (apportioned by '
                     'irrigated-cell, footprint or all-cell area of the SA2 in each division)', abs_2020_21_total_ML=split_vol))
    for k_, v in basis.items():
        nat5.append(dict(row_kind='check', draindiv_id=0, draindiv=f'ABS SA2 volume apportioned by {k_}', abs_2020_21_total_ML=v))
    T5 = pd.concat([T5, pd.DataFrame(nat5)], ignore_index=True)
    T5['o'] = T5.row_kind.map({'drainage division': 0, 'Australia': 1, 'ABS national': 2, 'named row': 3,
                               'ABS window FY2017-21': 4, 'check': 5})
    T5['ro'] = np.arange(len(T5))
    write_table(a.out, 't5_water_drainage_division.csv',
                'Item 5. Irrigation water (ML) by drainage division, 2021-base: crops = sum over irrigated crop cells of WR(irr) '
                'x area (agec_crops.h5); pasture = LUTO head on irrigated livestock cells x WR_IRR '
                '(ML/head, agec_lvstk.h5); against ABS 4618.0 2020-21 (WUAFDCASGS202021.xlsx items AGWATRPRODAML_F '
                'total, AGWATRPACRAMLGRZ_F grazed pasture, AGWATRPHAYSILAML_F hay and silage; crops_hort = total minus the '
                'two), SA2 values (SA2_2021 key with state ties fixed) split across divisions by irrigated-cell area, '
                'else footprint, else all cells (inferred apportionment); national rows, the FY2017-21 window '
                'and the MDB cube check below the divisions.',
                T5, ['o', 'ro'], PV,
                fb('lumap.h5', 'lmmap.h5', 'agec_crops.h5', 'agec_lvstk.h5', 'lvstk_k.h5', 'pasture_kg_dm_ha.h5',
                   'livestock_dse_ratings.csv', 'sa2_2021_cells.npy', 'crop_water_calibration.csv', 'livestock_water_per_head.csv')
                + fi('real_area.h5', 'state_id.npy', 'safe_pur_natl.h5', 'safe_pur_modl.h5', 'draindiv_id.h5', 'draindiv_lut.h5',
                     'NLUM_2010-11_mask.tif', 'ag_landuses.csv') + fs_('WUAFDCASGS202021.xlsx', 'WUAFDCMDB202021.xlsx'))
    print('item 5 done', f'{time.time() - t0:.0f}s', flush=True)

    # ================================================================ item 6: pasture and eligibility
    pk10, pk21 = g10['pasture'].astype(np.float64), g21['pasture'].astype(np.float64)
    lv10c = np.isin(lu10, [LUS.index(l) for l in LUS if any(x in l for x in LV)])
    lv21c = typ21 >= 0
    R6 = []
    for s in list(ST_NAME.values()) + ['AUS']:
        m = (state == s) if s != 'AUS' else np.ones(NCELL, bool)
        d = dict(state=s, cells=int(m.sum()), area_ha=a64[m].sum(),
                 mean_2010_kg_dm_ha=np.average(pk10[m], weights=a64[m]), mean_2021_kg_dm_ha=np.average(pk21[m], weights=a64[m]),
                 cells_changed=int((pk10[m] != pk21[m]).sum()))
        for nm, cm in (('2010_map_grazing', lv10c), ('2021_map_grazing', lv21c)):
            mm = m & cm
            d[f'area_{nm}_ha'] = a64[mm].sum()
            d[f'mean_2010_on_{nm}'] = np.average(pk10[mm], weights=a64[mm]) if mm.any() else np.nan
            d[f'mean_2021_on_{nm}'] = np.average(pk21[mm], weights=a64[mm]) if mm.any() else np.nan
        d['ratio_2021_over_2010'] = d['mean_2021_kg_dm_ha'] / d['mean_2010_kg_dm_ha']
        R6.append(d)
    R6 = pd.DataFrame(R6)
    R6['o'] = R6.state.map({v: k for k, v in ST_NAME.items()}).fillna(99)
    write_table(a.out, 't6a_pasture_productivity_state.csv',
                'Item 6. Pasture productivity (kg DM/ha/yr), 2021-base pasture_kg_dm_ha.h5 (pasture_productivity.py, AussieGRASS '
                'FY2016-25) against the 2010-base input pasture_kg_dm_ha.h5, area-weighted (real_area) means by state over '
                'every cell, and over the grazing cells of the 2010 map and of the 2021-base map; cells_changed = cells whose '
                'value differs.',
                R6, ['o'], PV, fi('pasture_kg_dm_ha.h5', 'lumap.h5', 'real_area.h5', 'state_id.npy', 'ag_landuses.csv')
                + fb('pasture_kg_dm_ha.h5', 'lumap.h5', 'lmmap.h5'))
    x10 = np.load(I['x_mrj.npy'], mmap_mode='r')
    x21 = np.load(B['x_mrj.npy'], mmap_mode='r')
    assert x10.shape == x21.shape == (2, NCELL, len(LUS))
    nst = len(ST_NAME) + 1
    R6b = []
    for m, lm in enumerate(LMS):
        xa = np.ascontiguousarray(x10[m])
        xb = np.ascontiguousarray(x21[m])
        for j, lu in enumerate(LUS):
            e10, e21 = xa[:, j], xb[:, j]
            q = {'eligible_2010': e10, 'eligible_2021': e21, 'gained': e21 & ~e10, 'lost': e10 & ~e21}
            cnt = {k_: np.bincount(ST[v], minlength=nst) for k_, v in q.items()}
            ara = {k_: np.bincount(ST[v], weights=a64[v], minlength=nst) for k_, v in q.items()}
            for si, s in ST_NAME.items():
                R6b.append(dict(state=s, lm=lm, lu=lu, **{f'cells_{k_}': int(cnt[k_][si]) for k_ in q},
                                **{f'area_{k_}_ha': ara[k_][si] for k_ in q}))
        del xa, xb
    R6b = pd.DataFrame(R6b)
    nat6 = R6b.groupby(['lm', 'lu']).sum(numeric_only=True).reset_index()
    nat6['state'] = 'AUS'
    R6b = pd.concat([R6b, nat6], ignore_index=True)
    R6b['o'] = R6b.state.map({v: k for k, v in ST_NAME.items()}).fillna(99)
    write_table(a.out, 't6b_eligibility_state_lm_lu.csv',
                'Item 6. Land-use eligibility x_mrj (bool, lm x cell x lu): cells and area (ha) eligible in the 2010-base input '
                'x_mrj.npy and the 2021-base x_mrj.npy (eligibility.py), by state, lm and lu, with the cells and area gained '
                '(eligible 2021, not 2010) and lost (eligible 2010, not 2021).',
                R6b, ['o', 'lm', 'lu'], PV, fi('x_mrj.npy', 'real_area.h5', 'state_id.npy', 'ag_landuses.csv') + fb('x_mrj.npy'))
    print('item 6 done', f'{time.time() - t0:.0f}s', flush=True)

    # ================================================================ summary
    fmt = lambda v: f'{v:,.0f}'
    Lw = []
    w = Lw.append
    w('compare_inputs summary. Every number is computed in this run from the files named in brackets (sha256 in')
    w('provenance.csv). Units: head, ha, t, t CO2e (AR5 unless stated), ML, FY2021 $.')
    w('')
    w('1. Livestock head, 2021-base map, Australia [t1_livestock_head_state_type.csv]')
    for _, r in T1[T1.state == 'AUS'].iterrows():
        w(f'   {r.type}: NIR FY2020-21 {fmt(r.nir_head_fy2021)}; ABS 2020-21 {fmt(r.abs_head_2020_21)}; map pasture head '
          f'{fmt(r.map_pasture_head)}; map stubble head {fmt(r.map_stubble_head)}; named rows '
          f'{fmt(sum(r[c] for c in named))}; closure {r.closure_nir_minus_map_minus_named_rows:,.3f}')
    w('')
    w('2. Land-use area, Australia, aggregate rows (ha) [t2_landuse_area_state_class.csv]')
    for _, r in T2[(T2.state == 'AUS') & (T2.lu_code == 'agg')].iterrows():
        w(f'   {r.land_use} {r.lm}: 2010 map {fmt(r.area_2010_map_ha)}; 2010-11 control {fmt(r.area_2010_11_control_ha)}; '
          f'2020-21 {fmt(r.area_2020_21_before_typing_ha)}; 2021 map {fmt(r.area_2021_map_ha)}; method change '
          f'{fmt(r.method_change_ha)}; land-use change {fmt(r.landuse_change_ha)}')
    crop_m = (T2.state == 'AUS') & ~T2.lu_code.isin(['agg', '-1']) & ~T2.land_use.str.contains('Beef|Sheep|Dairy|Unalloc|Grazing')
    cs = T2[crop_m][['area_2010_map_ha', 'area_2010_11_control_ha', 'area_2020_21_before_typing_ha', 'area_2021_map_ha']].sum()
    w(f'   all crops: 2010 map {fmt(cs.iloc[0])}; 2010-11 control {fmt(cs.iloc[1])}; 2020-21 {fmt(cs.iloc[2])}; '
      f'2021 map {fmt(cs.iloc[3])}')
    w('')
    w('3. Emissions [t3a, t3b, t3c]')
    t = X21[['luto_t', 'nir_t', 'diff_t', 'activity_t', 'nan_factor_cells_t', 'factor_resid_t']].sum()
    w(f'   FY2020-21, 2021-base map and factors, all rows: LUTO {fmt(t.luto_t)}; NIR {fmt(t.nir_t)}; diff {fmt(t.diff_t)} '
      f'(activity {fmt(t.activity_t)}, NaN-factor cells {fmt(t.nan_factor_cells_t)}, factor residual {fmt(t.factor_resid_t)}); '
      f'scope rows (not in NIR): irrigated-pasture energy and lifecycle {fmt(scope21["irrpast_energy_lifecycle_t"])}, '
      f'off-land energy {fmt(scope21["offland_energy_t"])}')
    for grp, d in X21.groupby('group'):
        w(f'     {grp}: LUTO {fmt(d.luto_t.sum())}; NIR {fmt(d.nir_t.sum())}; diff {fmt(d.diff_t.sum())}')
    for case in REF['gwp_cases']:
        d = X11[(X11.case == case) & (X11.state == 'AUS')]
        w(f'   FY2010-11, 2010 map and factors, {case} reading: LUTO {fmt(d.luto_raw_t.sum())}; NIR {fmt(d.nir_ar5_t.sum())}; '
          f'diff {fmt(d.diff_t.sum())} = gwp {fmt(d.gwp_t.sum())} + activity {fmt(d.activity_t.sum())} + factor '
          f'{fmt(d.factor_t.sum())} + joint {fmt(d.joint_t.sum())} + scope {fmt(d.scope_t.sum())}')
    w('')
    w('4. Crops, 2021-base map against the ABS window means, Australia [t4a, t4b, t4c]')
    s4 = S4[['prod_t', 'q_mean', 'value_aud', 'v_mean']].sum()
    w(f'   all crops: tonnes on map {fmt(s4.prod_t)} against {fmt(s4.q_mean)} (ratio {s4.prod_t / s4.q_mean:.4f}); value '
      f'{fmt(s4.value_aud)} against {fmt(s4.v_mean)} (ratio {s4.value_aud / s4.v_mean:.4f})')
    nr = N4[N4.named_row]
    w(f'   Named rows (under-mapped crops): {len(nr)} rows, crops {", ".join(sorted(nr.lu.unique()))}; net t not placed '
      f'{fmt(nr.t_not_placed.sum())}, value at window prices {fmt(nr.value_at_window_price_aud.sum())}')
    ov = N4[N4.over_placed]
    w(f'   Over-placed rows: {len(ov)}; t above ABS {fmt(-ov.t_not_placed.sum())}; value at P1 {fmt(-ov.value_at_p1_aud.sum())}')
    mo = OSw[OSw.fy_end == 'mean FY2017-21'].iloc[0]
    w(f'   Tasmanian Other oilseeds, mean FY2017-21: {fmt(mo.production_t)} t, {fmt(mo.area_ha)} ha')
    w('')
    w('5. Irrigation water [t5_water_drainage_division.csv]')
    w(f'   LUTO crops {fmt(sumc.luto_crops_ML)} ML, pasture {fmt(sumc.luto_pasture_ML)} ML, total {fmt(sumc.luto_total_ML)} ML; '
      f'ABS 2020-21 SA2 sum {fmt(sumc.abs_2020_21_total_ML)} ML (ratio {sumc.luto_total_ML / sumc.abs_2020_21_total_ML:.4f}); '
      f'ABS national {fmt(NATW["total"])} ML; FY2017-21 window {fmt(v_crop + v_past)} ML (ratio '
      f'{sumc.luto_total_ML / (v_crop + v_past):.4f}); MDB cube {fmt(mdb)} ML against SA2-apportioned division 7 '
      f'{fmt(RW.total.get(7, 0.))} ML')
    w('')
    w('6. Pasture productivity and eligibility [t6a, t6b]')
    r6 = R6[R6.state == 'AUS'].iloc[0]
    w(f'   pasture, Australia area-weighted: 2010 {r6.mean_2010_kg_dm_ha:,.1f}, 2021 {r6.mean_2021_kg_dm_ha:,.1f} kg DM/ha '
      f'(ratio {r6.ratio_2021_over_2010:.4f}); on 2021-map grazing cells 2010 {r6.mean_2010_on_2021_map_grazing:,.1f}, 2021 '
      f'{r6.mean_2021_on_2021_map_grazing:,.1f}')
    e6 = R6b[R6b.state == 'AUS'][['area_eligible_2010_ha', 'area_eligible_2021_ha', 'area_gained_ha', 'area_lost_ha',
                                  'cells_gained', 'cells_lost']].sum()
    w(f'   x_mrj, Australia, all (lm, lu): eligible area 2010 {fmt(e6.area_eligible_2010_ha)} ha, 2021 '
      f'{fmt(e6.area_eligible_2021_ha)} ha; gained {fmt(e6.cells_gained)} cell-uses ({fmt(e6.area_gained_ha)} ha), lost '
      f'{fmt(e6.cells_lost)} cell-uses ({fmt(e6.area_lost_ha)} ha)')
    with open(os.path.join(a.out, 'compare_inputs_summary.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(Lw) + '\n')

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    PR = [['source', k, url, pin, 'sha256 of raw file or API response', note] for k, (url, pin, note) in SOURCES.items()]
    PR += [['inherited', f, '', in_sha[('input', f)], 'inherited 2010-base input (inherited_2010)', ''] for f in INHERITED]
    PR += [['upstream', k, '', in_sha[('base2021', k)], f'2021-base product of input/Nick_build/scripts/{v[1]}', v[0]] for k, v in BASE2021.items()]
    api = sorted(k for k in SOURCES if k.startswith('anga_'))
    api_sha = sha(''.join(f'{k} {SOURCES[k][1]}' + chr(10) for k in api).encode())
    for tname, (ins, nrow) in PV['tables'].items():
        if any(f.startswith('anga_') for _, f in ins):
            PR.append(['table_input', tname, '', api_sha, 'source', f'{len(api)} API responses anga_* (sha256 of the sorted '
                       '"name pin" lines; each pin in the source rows)'])
        for kind, f in ins:
            if f.startswith('anga_'):
                continue
            h = SOURCES[f][1] if kind == 'source' else in_sha[(kind, f)]
            PR.append(['table_input', tname, '', h, kind, f])
    PR += [['script', 'input/Nick_build/scripts/compare_inputs.py', '', sha(me), 'sha256 of script', '']]
    PR += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    PR += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    for tname, (_, nrow) in sorted(PV['tables'].items()):
        PR.append(['output', tname, '', sha_file(os.path.join(a.out, tname)), f'sha256; {nrow} rows', ''])
    PR.append(['output', 'compare_inputs_summary.txt', '', sha_file(os.path.join(a.out, 'compare_inputs_summary.txt')), 'sha256', ''])
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], PR)
    print('done', f'{time.time() - t0:.0f}s')


# ============================================================================= API response pins
# "loc sector gas sha256" of each raw StatesOutputTimeSeries response, retrieved 2026-10-04/05.
# Locations: 3 AUSTRALIA, 6 NSW, 11 VIC, 8 QLD, 9 SA,
# 12 WA, 10 TAS, 7 NT, 5 ACT. Gas: 103 CO2e AR5 (Gg), 108 CH4 and 109 N2O as CO2e AR5 (Gg).
ANGA_PINS = """
3 684 103 49b551a7061e6b4c34df485e81db72f6d5ff742d8702dbc5157b088d9a775858
3 684 108 ebfefe626705defbd44558a544b728325ed4fa9312c080cc9bf5b6e4c0cd7738
3 684 109 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 705 103 1211e278bdf1ef51e2a70b3a31b755c2e20d23c9623fb332f9baf7de2dde93a3
3 705 108 4886f1ad088823c91feaecf32064c2e33edbce9bbabd871775437646f364709e
3 705 109 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 720 103 b86f5dece8e43e1724db4a95f83601da7322967a54db5a2601860f0d103a1c3d
3 737 103 ea990f2119393c3d53182878a7689b510abecb9a1c782cb9cc02933a0ac5ec7f
3 737 108 7b6e3fc1ba17e79b0eccaf353af736bfaf844de1f7d45c445be11a4c6f92cb3a
3 737 109 33a0d32a0c7a79bfc44f221d193935bd998df689ecc1a4f5eaf7431b683f7c6f
3 945 103 6d9cf71f438c189284cfbacbceec4ffae09adb053a78431bf0b05f557decfafb
3 945 108 4ff0b83b738419031dceacbf43fa3d68f28c51c3b99954d2699efc6641ac2d09
3 945 109 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 1092 103 ad8d8eca5f49ca85555073854495da39232a433dfaa347cf9191959a3da0431e
3 5249 103 8f33b7cfa330db7f928acddf8865adfbbddc4dab1f2971a974dd576ed471d857
3 5263 103 55efbb9399e8f6e454e5c13911bfe96477a5fdf7330416b672a5c222cc488150
3 5279 103 a209e79ebacca2ad115df16f88a0d4b1e43022c86aa7e35b4632cb45e2757e7b
3 5285 103 7d8c885deaadf28c8260459d2458ee63066ef0dbb949b03a5154954a2d7b6b9a
3 5285 109 1b1163c799c71140c210d27607a4008bd40709bfa32020f30cd02992bea91290
3 5286 103 e746e52d549cb49f9eb9583f7a7197f1961d6953ed5b8badbd3531dcd4487b23
3 5286 109 dca6b151020dd8deea5d5e8a1d461947b8b869a994f10935617e5bab8bcbcc94
3 5287 103 f8f3b33531b44487d1263ac4c596530582027b12d616a8e6c746419fb174b66f
3 5288 103 f0d2b7a87f8f1d5e989bf85ebd2fe9080face4cd90be308f5631bd51a5227a3f
3 5288 109 e184537471736795f177db7dde847dc2263d848f44de658605b4eb27e1a91b77
3 5289 103 9c5d7323c4dbbec25f0828a1f4d1f3f3538611fea64f8eadd539c107a3bf5769
3 5289 109 c1859c84f1b22ff4cbae07ae35f05a81b5f4bf47494074a90383270f4659a893
3 5290 103 5d5653123c93f62a85e6f3cbe61577385243fc33eb29cd7c63a804b92502b83b
3 5290 109 dd2aaa6c4992fd65751641d5dae4a9b7791e7c25b0616782e692fa1e32ae30e7
3 5291 103 7077b00b1f3674e43a06a65036b5f4f5c8314bce720a131c6709c57ee6eb70bf
3 5291 109 a6b1a06908ae8c20538a0f3fda52ff3bf995a1b326f8e71005f0b77f53f62bf7
3 5300 103 5210380003864744844e8cf1250c072865451831c261a72843b3102ac7929645
3 5307 103 613db6d1d4db6a3d3fb385d5ebc19263fcd73617acc953eabfb2b547d97a685c
3 5321 103 1ff9443503ebeaaa667e8b9e2313b33147ce9bd51d0f6eb4ce07d9932726e6fe
3 5321 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5321 109 bfc96b31f40340c928c81eedba8c84cdc52eacedd332793341f1f2085dc74c16
3 5322 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5340 103 4307e6cca95d8cefcf591c09fc2f6b972a53203f6f56b7dcebc06464bdce9cdf
3 5340 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5340 109 233c2efe45d66121523a81e3bc9f0332fb9f4b20bd2f794d21a483331610e918
3 5341 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5348 103 5121353e0f218a2ca62c539bf3316acb205c87362af451579592ef62476200fa
3 5364 103 a1e1e3e22fdb149a6eb021238776a8c5a8ce7ec4b32dd32310dd1f445b596920
3 5364 109 2b72bcc508ba19af551d9c990219777c6b4a863afde376d921af3e2cd0e02d5b
3 5365 103 adc643e79f52d17aa64398af97ea317e94a7fb43357560844a2cbdf799379688
3 5365 109 e1de2e480e9dab71cd3b0347e2a605911354bcf73d700d9890c0550db28a6980
3 5366 103 ba1399a53f2c2e94d379cc4035d35142fb2d5080e9977c39fcc20ad6faf0db6b
3 5367 103 1f38f4b56938d45cb63a9be5188f0b56afb16040e50c7213ae9283ffa1ae10dc
3 5367 109 fa54737d9dc0b42275a1cf7f69717a9c718d443e0edf0b59a7e6e39ff181d84e
3 5368 103 ba7e93c58d3c413b1a280a7ad1f26afef453162bab42fc71e2ecec6d288c1c50
3 5368 109 93268389b8075479ef77f8864a1785d36717b65574c6cf2680f071e316dda89d
3 5369 103 0e4ee3cdc840e86087ed14e0c7cef72e17129de671c5192fbc6cec0d80192269
3 5369 109 2c6738379112c4ea09f42fec1b0bf6c8514eb57c247d8fced1101564390db559
3 5370 103 c6fad1a15ecb4b62a5f091d856698cb03f5f40c5241dbeb6729fc719d652f174
3 5370 109 c70c03a1614f1b84a9b46cd4ec906375529e7e5746bde14c392a2d6fb1a7107e
3 5377 103 9632bb8b728cb09fa9e96d8e1ac824e214db74b2db9b90b5a387df469c1bbc66
3 5377 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5377 109 cccfa5f9e857f4a3212328d392b40b8d8a289649f42a1d97099ce303535815e8
3 5378 103 ec31e6396a0874352735c2fd506e765ed5e363e4b3c21500d8f60fec2b910cd5
3 5385 103 8295a8301ae52665422c85329c712716f0f8f541c399fc714c4ae76ef4b3b2cd
3 5396 103 8d8e85fee61fda11d44b62eb43500182dcd1a1179a04f9e1cc525784691d4d37
3 5396 109 085e09e88c29a81141cbf648a9abe5d7a264b559311632a925ad2cd8927936a8
3 5397 103 345cc8bbd37eaf605b06ae92c18778826e2e9c8623b9a56736ff2c95d0813c14
3 5397 109 6e457c966181ddeb43a3a8dda4d1cb7b39520900d8b39b1b25344caa2f7fe540
3 5398 103 2a0caa233d3faad14cf23d16f10a92113144b47fb59da5a87c36e5ba20e88576
3 5399 103 c73342f35b80cc9eff5f0ca31bd8198dc89b6d2ea2d24b1fe385717d9e7bddc3
3 5399 109 50338a0eee73eba6f058495640f3a5648546b206d8a93071c4cfbffb2c34e0d6
3 5400 103 84e3720186aae4d0690a6911e8f478d1b474dc5eb1d279f75d11d2a9aa77dcec
3 5400 109 afb398bbf0b03d2637f4b3f7e9f198b6caa14528d722e42d5865e271ca705eee
3 5401 103 0c54f7875851915706d18de394e64400e1ab3b1778a4574e6d7228003472a93c
3 5401 109 83fd70ff314ebe360e333d61abaeca5de878022752fefec2a435d60e5721d162
3 5402 103 cb49721c61677b6127533f850fb1e5c66d51969f3a9e8ddf6fab228516c94f75
3 5402 109 a4ec07468d53224e2998b14a54ccb094b42ab4c0a610880d623d93c52ea2101f
3 5409 103 e5a98f26913899c13bb8526f9fea1cb3e05d0977289b86fbfc06100d1f1b8b82
3 5409 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5409 109 65d3ac31958bb1a2f8ea5af7d4c16de3a514a4a5be3453c46ce75f381475ef95
3 5410 103 7c006f57e7770c7fb3cfd431cc07c60ddede0c4388a521b95ba6d4a49d51facb
3 5417 103 e113fb333baeec05e70c4a08816a2eb61d57f07e903add1e679be2a1e5777af7
3 5555 103 12e2d10ae10b4c1883d8047328995f7941d0c8640806691ed53aeef0c2ee000c
3 5555 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5555 109 36ca09c6639e5f20149f8e70531cb5791ea90f3d17636f811b72a1db1930ba91
3 5557 103 d77c9651e0da4f9816c5122357a91c227ec8453cdd3906a53c973b8eb995056a
3 5557 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5557 109 b7a5015e7466bc6196f3f2ea741d1136238686cf350b7d2e4c35634314b944a4
3 5559 103 2d8d37986eac821f076ab7c6649408db2d1971471cbb2163addbe0a28c40aa71
3 5562 103 753ba6d91b3a9ce943a76c34dc980e175854ae747ce55e206319883836f08963
3 5562 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5562 109 c453a9360c119a90155191e35efee2716955b51a86c0e62ede1fbddd91df588d
3 5563 103 01e806b4200019dfe443f386ba369f1aa4f3fba9a3ec8e13ba6f6ec5bea1077b
3 5811 103 179951afe8a4a109b17b86155baf2ec65240cdf9a559e88b244ea89472aaec2e
3 5811 108 7ee4392795ac91dbd1fb7c3e6387348d73b0908e65330c08b5a699df1baf4cd2
3 5811 109 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5823 103 08b86c6fe96f8072be94154fa4f06b1846134a5d9836be4e3534b7352c291032
3 5823 108 a8ad4a0cda2ccbbe71c4e8e08d949c81c2dc4d164df2407761c0155181cbbf7e
3 5823 109 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5835 103 8cc24cf55f5d80da55c59d2f721bf80695684cd71509bef84cce7e563ac88972
3 5835 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5835 109 e4538cd70aad5715471546315af63e9c64f5fdd10ebb136768b8170b3706e524
3 5836 103 8e8cbc1051fad2fb8b3a452debe614ba5c051e96797737fe7008fd769752da8f
3 5836 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5836 109 620de5835e1d016573ae9f6ae20af11009b0ffd53ff4203e81380d594f4489e3
3 5837 103 3662d86b00b1ad6fb47154c4319d06d3427f8ba2c27f5bf06d75df7f7f14a99b
3 5837 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5837 109 486da84eb54fcd49e9e2507b481012a5181ebfc2834c0834d1e80d38d3e84ca4
3 5838 103 57487b1dbfd0938e70e3a2c940e91e9f379e8508385a3aa555b770f497c080eb
3 5838 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5838 109 38d09351c84d09463f8cabfaeaf28964dd7375da69020f64495740fb93b3240e
3 5851 103 f4d7052d36bc33fcf90de8b124fc5e7d84716d6918be0b79987cfe1af39beec7
3 5851 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5851 109 44c7fa30ebf2354ec6e47837dc1a3deee73961c0b685e2d63820f5cfa437effb
3 5852 103 95703d4a7ac707f7cec0561b99ab8e6438ed7d8d3b14708f2202da2224f66ad4
3 5852 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5852 109 3b3f141e714619ce60d627958429abb70f0a2150f7a2f9116771d1f7334da026
3 5855 103 c80e54b90c4136ff3440cc8cf1e3695e4a3f6c7eb2449bee26ff056d4d6a6a88
3 5855 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5855 109 2cec8e11447361e8da234bc5c232b709e97342c3e07943d27f1e29469555996e
3 5856 103 40c55236275ce6984742e1658c98648f733e5cdf5e51df158cfc61b24bedb139
3 5856 108 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
3 5856 109 e1c0ff4ee6aeb81d31d94a9cd66daad91744d0ff895bf2e8d1c7a05606776fa5
5 684 103 27731b15f9910c16c35fd90f974c186bd72ebf88497327285fdde34163155c7a
5 705 103 d6193d1f89da62be47a0f21ea9b76e8ea230d1477e7d64308af6c10b6672e014
5 737 103 9ddaa87d2462bd3c628eba8985d09e8f3f39acd482a773921a10e7ec965cc201
5 945 103 48abb5655718f585b13bf603672c66666fbff5b90271beaf4dc1e0cb53a70b00
5 5285 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5286 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5287 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5288 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5289 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5290 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5291 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5321 103 c6fb1400ab5a8f6a824d1b54bb277b4db9183c8901b93c5be41d92da6b9861b5
5 5340 103 e6bb5afc6d8317367f172949527bc8fa8f5a94bd6720f4478affb77abf7071eb
5 5364 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5365 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5366 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5367 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5368 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5369 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5370 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5377 103 6055393c19a9ee6f59e8107a61898f1035fdf6c4345e89d8298e9fe6aa830e45
5 5396 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5397 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5398 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5399 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5400 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5401 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5402 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
5 5409 103 5db9f03ab15abf7c3b096c3b0d4a631dfa93d285e905cf9942784f52e0d51931
5 5555 103 c52067c79d9eb505c4373fe7429bc2feb062838ce8b770310126f535c12c190b
5 5557 103 e7e83e427f909724190b39b4d5a84645aeff0e4392774c60a37249bb70a638f7
5 5562 103 87b03adf1d2c6db5f6812e33ef61f1d8db613f61213abc98522045e15599547b
5 5811 103 6616b61e3bb26930c6df214b6567ce7ae6e3e7801d0e1ca772b67a1adc1ba8d6
5 5823 103 f2ac9a950a39276670f243bcd1f39c9604b9ee61346866d059245e663dde0962
5 5835 103 bdbdedbe02a58530353acc56060651856fa37cb694adbb0fdac5c5860ed0d548
5 5836 103 91dc3ad90b5501f4c214a0a9636b1453626089e56b9907bfb971fa6474e3f12d
5 5837 103 19faebd77961882ea85741cff1b517f9edafbc1354d2c82d0c74673b554a93a6
5 5838 103 4e2f12810d7b7f8c5559f6ba214a729245ad4133710729c180349595d4d57869
5 5851 103 1a519d09411cdd40302ed3640e448fd9e5df96e5ab9be0bfbd1cf8fb4fa2e5ac
5 5852 103 0a389dff238316e8876fcbbc26e2dc0be72f45f924e7cf21f617b47917325855
5 5855 103 d31949b8f7ce13e0a5d8e03a2ca9ac411ded237ea979636a5a1ea58ba12e2f61
5 5856 103 aa7a9dcbff08c7459ca7084fdc56566abc7c9bbb95744776bf55fc92711932cd
6 684 103 af463bc47188012b71dd0903ea3910d83fec73db0492c32783229dafe2c0e859
6 705 103 418c05039e13b7651dffbe13926d835156fd9cfa2c8a67a7a096d4a4da29a5e9
6 737 103 8e73e600b814438e08246f3e9f6075fb0d1ed9054bf84deceff858df0585bd44
6 945 103 69cedbd30fea02e52c9f9de54b522f4e519e0f54db3ab44cd3674bb5fa7ccdec
6 5285 103 c9f2abb58bc2d381197b6499125ce8a85e23aef3e505759d7ec6aeaa3b44d3ec
6 5286 103 f9262c870a3437e52dde037a87550904e90e6087c72356147e461fee7570f5e0
6 5287 103 f14b116c07313eb0b96e3b1c400f3ffd67bd35e9ef5f094fbd0df70d560c5681
6 5288 103 7f9964592fa4c614200f4141b721a7058e8dbedf9fa3badea7df148473d6f3a3
6 5289 103 ba54d372eb6e6dc8787e0df792681529c265c7a776340d5c662b75b409ebb4be
6 5290 103 5891b1ff62700fd3ef56715e66eb8bdd99da37914692219140dfe4466045209c
6 5291 103 c8fad9ea90c21c09470c6b924bfbb044be06107dc9141526f682be637a6aefe5
6 5321 103 fc6d6ea6df706728c08351b43521bc3e5bab5871e2c904105cb67fa67c5d1643
6 5340 103 6eff4ba1c71c66002b7f02bbe233f7c915276f7a20a15bd720314bc85cd161da
6 5364 103 06f62f6c5406d36b8d6f9ec7e7c72ec4dac83beabfe56a821e770e5926e0f82e
6 5365 103 541dbbabdd2e28a2419a5e32c4dea411ca1a6cdfa08c3a01e1720ea3929d3aa3
6 5366 103 3aefc71c6e72299141016c77ed0dbcf036c2791cdca15e632a12154da579cd8e
6 5367 103 8d2d0d2ed6ec4f1de738cb0b6d8b1893881083ffcc7c1e4f2d13182f4495c4c4
6 5368 103 08e192a1826b996ccf3bdd545c721deb893521523881fabe9f79f1c557233a35
6 5369 103 bc3d3ade8a1119bcfd127cb5bcd490bd8a4edfa92bd2a9f02e74ae9fc3cbf5be
6 5370 103 cbc8c582d615233ec67ad3b671fa9c70f59dcb313f1368461ccbfab67f5b5371
6 5377 103 7fca9d33317c96fa4a9840196754c520f0dbb5732172755f169c42e10df44d0d
6 5396 103 a969682af8ba08b9e220f72c5df786fb3a7d11c894913de1cc4ca18fd8a0c062
6 5397 103 25438125381309d9b3bdba7642eeeb134f9962c293a845f60b1859864f351800
6 5398 103 378c37ec6eabe937597f8f26f59e24820988b016a22bf91a6f32991613452375
6 5399 103 f19f89616aa4d699442acf3d1bbd525b64187cbcbb1f2b565c1bea26ab8a516e
6 5400 103 bf8eeabc2cd54f7533c6bb21c02400750e7866d67883634156f12121ff864f80
6 5401 103 47d7a88a2ffb395d134d1f12fef77792ae629b31f455fac1a4c4db629aad6fab
6 5402 103 102154bad7ab56ba3ce78fbc647d3abe312dc1b601a1ea3a48e3ed9ca2759afe
6 5409 103 d1c830dbc752d4ad513fdad093068f85fd6cd3bfb38f28ea4853577703e84faf
6 5555 103 0d06e664d681dae1e48fc9467ae80cf668e7cfd9c8da401100b3f1106b596c4e
6 5557 103 b49031851590ebd71fa1e1cdfc9f6828b6e4105e0495f3bf6885778342db52b2
6 5562 103 9ae6eff6bb219717de86ca31f53d45c350e9976f524b36b5e17c6bd434ff2f77
6 5811 103 e6bee68136982c423234206ae1267db3daacdcf234f39d2e8665d76f38012111
6 5823 103 04584d5d80c250fdccdc8f6dbbbd148e85631b45f768080eb2805c1f052d1c29
6 5835 103 8fee96b1cf3115128cd50d6880be4c9fec9a43ea632d2de730d7e2718540d3b5
6 5836 103 6360f3615c45ecd0613fc2b58530acb44a534fdbabcf7a454b52f30f8bf7bcba
6 5837 103 22b55504b896fe7c09a03a9e0bba953c3f6771fdc2bc098c7c2b22bd2887367e
6 5838 103 0deb57ee0c556299f4b3a87b477cae2a67c4b9dba291d7bedc058f57c91415cb
6 5851 103 4073627b2f1b8f9acdfa464b50e5f210651b0ed929cdb591ed5d550f5a103459
6 5852 103 13b73b39cedab6b9df5769a3cc0ea5b980dc8a55c5ae92e0b5dda85cb4e202eb
6 5855 103 f9797cb19c3efdbe7e54bbe16c2cb05787bd7dc7ede56d55645f10b02021da7d
6 5856 103 0bfbac9ffe287d2abc5ee772e5d2816dd45ecaf6b81fef790e5fcc41ffc827b3
7 684 103 858cc51f5daa6cb4ad4a5a925962362d2c9480bf8ecd56593a4ba529dd79898f
7 705 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 737 103 cd863f47a3474b6bd5fa32ad0025e847f8bbbffe217f7e0a0a2266b70ee113c1
7 945 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5285 103 96d06e891693b72218aa0b3b3bf0d55666691ed982a77f86d89c50fefd6dd35f
7 5286 103 87609d45c83cf595788fd729eec99e08af5761361021613f7923aebc50c43e24
7 5287 103 7b4ab5ef8a4707e3c0ce4a8fc25bfe19793e4b7d851a80e82aab10d37168a30c
7 5288 103 f472ac4c5b4b8011c3cb753ba72aacf70c4cdabf0237caaefec7199fdbd879aa
7 5289 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5290 103 80d62896ea76da42c725ed4104dea7a619c28101ff73865afdcb074f131706f1
7 5291 103 c87ddc9853b4351f36dfce06553cc1c2b24916e600a5900063039a9d0ee751f2
7 5321 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5340 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5364 103 d4bd989a9daef3f3f8e308429aa786b16dfdbc6c32ea69cc581524b836d0076f
7 5365 103 be7fb75e3c51437fd7a78a1458fcbe3695508d639805bb46dedde52e7ae82542
7 5366 103 df9ddacce7d42e52389160f3990863d3ad6e2bc64083ebd3d42e0d2bfd9b782c
7 5367 103 bf99584df9cf96e07ee0dd8344a678a0cb4bb572f1503b09aa090b3f1ea7b679
7 5368 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5369 103 4c0d8778164c5a22efb3eb353310ada40caf1452cedbe26b4386054eb446558a
7 5370 103 7e3151a0ffb0684e68e2a3ac5477dc34c6bcc8b9776aba3aa14696a385ac3f92
7 5377 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5396 103 197a9e6294b49c76fdb1e29c5643992f903e5a36d628da39f7e9ad333d257bca
7 5397 103 60c94ca97f5c29a50ed66023fb477010fcb88051cd798aa654170abb934c04e8
7 5398 103 690f37a691ab4c8051304471054037ff8971d78248119c94d9f2138b072fbcfe
7 5399 103 3aa2d2cad72c5d3b822d1cd7c8e2245ca965324a1986e43db3e5fe19a480e660
7 5400 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5401 103 59e661925ade70dbc538e4cf2abf0191cf32bc79192acf0679450c4406ac9557
7 5402 103 07dd5086ac0abc4292c3c51965461d5919cdb24427a4e733f3e65c687299f9e4
7 5409 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
7 5555 103 63113d9d42d35dcaadec922823b281b41dacbe21306af8c860ac6b038fcd2085
7 5557 103 3abf3cc31c9e8b21da913d71939f41985041f40cc8da304e5ee33a73f187b41f
7 5562 103 3f198d2e624fba6311ec5b52efd86c5f43a4e3731e4c468de617d134232c5672
7 5811 103 e87926f3da146fdc427dee02ef51ff1baa47f4c99935970b17fc915ee3cada28
7 5823 103 cfcde21a48555a85dfa35e7f704a979e900d8b2b4c4bf04ddb9837f0ecf4c3f3
7 5835 103 903d25656ba914cd82fb2bbf5f990e4078d3fcbf82de0005afc8d9ba1d09c544
7 5836 103 754752e60b9008535572ba248670e0453ad51bc993d587e9e5e0fad264944afb
7 5837 103 3689b00544885f7545d87540fb01ee6f40a09d56ad7312a3c4b685b64453a3b3
7 5838 103 bd72aadc8f950bab0228f055697837f3fd69ca8fe08ac6fcb0f5bc01fbcbd345
7 5851 103 3ec4654eab4b3332986da3390e2ae414338a4461685496c0d14faff955e21dad
7 5852 103 af3a7176874314f4f39358e8a81d2e0d017351aabff7d2bf934773b520e81c89
7 5855 103 15429b43b49f80eacf494ee6827ad32548df15f85a78abbb0c32419e082d9b24
7 5856 103 abc752c1e50d5ba2f4e477e0ed1562aabde514e9115e70bd24d767914b162d54
8 684 103 7867dff25d3bb290bfb2245818246e5ee0bb1ac19bb5bcbd657170af4a0c25c8
8 705 103 9d9f946673a505aefacb64284f68507bedb640270ee9fafba775c6a3a5d9f9e1
8 737 103 731ae719b1f137f6dbb4f79cb84f7c1331f49b28e8d9665bc3349ba79e8a1c52
8 945 103 435c35b99b2ed2faaac5fe8702340bb9ae6945ff708efaab7bbe02ff687828bd
8 5285 103 871b4f611d507626a33dd8871b65bef3e57a1535989f95a68d3a8bc65c567ab2
8 5286 103 42803ef542f885a22bdbcfccf6cfec08d7a9f0cb81b81871a8e9452d9786e420
8 5287 103 3a133b6546e795cc10889c9c9895f84646a9d27f4afbb508afc1850d98ee3cae
8 5288 103 8aacf8e5d91fe9683305c0e45490a2e8fe8e4adda009a6efbd73f783b2e87383
8 5289 103 0c4b80d0d89fc3999604ba00fcd435f64678a821d0251e05b1b44068c3010acd
8 5290 103 f4a65b22701fbde53184acfbb5a4f55cb0d737b55fb3938a7628449856298726
8 5291 103 5071475477c1e329370f6d46f078f93fe52eb6a6166e6c9187626fa6f54e5661
8 5321 103 b2772db527b559c7cd1b6a50f06d950b89647150be2e6a9c35be386600cffcbf
8 5340 103 f664aa467867135e7565f9125f7e85edbad27eb8f556cb48f31a6f3b14cfcde4
8 5364 103 3a1900c1b21d014b415804ed90a1a5e2ed506de5b54a97e3d61a855ce15ea5b5
8 5365 103 24d41b5b30b558d7901eb8aced3e4249777a61fca20448ac016d5a2afc972661
8 5366 103 867819930fcceb9b10aebb7a944466dad97320528a3b238d9668b3cbd58d9072
8 5367 103 02798075f251e02b14364f86509941230c3d3a535850edfad94630e1e232ad82
8 5368 103 806a2fcdf3ab3c871559a2058994bf23593f6e747ead7895ced4724619db4533
8 5369 103 3e143a6f9424e8f4450197820936a7f56db726d1674164f2306183b6225a948d
8 5370 103 48ff2c700b5360d1a39902d0d614289188096a59986718dd05f7cdd8693bcc35
8 5377 103 f15458f72603f557e030956a06d82907ed4975e6e5318239b0e273680f9197b3
8 5396 103 6b33ffa266b3a4161ea0284421938940b6c3d455dd85535f0c27ae2da033343c
8 5397 103 6b4715fbe7f515a0c24772118a7514e1e7444dc0e80d80928de26c5d75b0d680
8 5398 103 8bc17c81d39ac7093876a69e39f54410c4ca715b6ac41ada85e699276611a2ff
8 5399 103 3fd7598df039e0a658884eb439dadd545e057d6294f4841f25dcf1ac3d68cc76
8 5400 103 9599a6a91e4c3802b8e9c8172f4051bf24fa899896e14ca6a7109bd902d68610
8 5401 103 a8c82f5be984b36a26f4e227cf2a69b789d7168fb3589dac7024edde271f3461
8 5402 103 8ec69d3773477f0b2a95e72424271c869832ca79b64ceb20d98be033e0e66414
8 5409 103 02d2a59cef10928554e1967013e18b2ae26aa4ccdf0bf5687e0242db1000ae19
8 5555 103 6c283262144ef90bd366b6843e9ee90308f8a59ad56cb0831c640a7b7941e51a
8 5557 103 54f28b37303086e74ea06ab76d5438695124a04b1f7c419bed3e0b0cc8dee09f
8 5562 103 5c23c60b1f5df867d0f16b51fa62f7b9b0798083e2d28b2f9744750cfa48c494
8 5811 103 1bc7e6864e36fff59cea5f1be6010a7c858ab1b5817e627fc104815b48904727
8 5823 103 62d2eed7fb4c03dcbba7988bde1718258ac41dcbc5b947c419c3a8bdb2e0ff1c
8 5835 103 91aa321ce4dbc4c78cc2c4acd78c136808f4cafaff9f6f207d6fb4b3b0537395
8 5836 103 d9b3ecfed46ee502fcfcb6ced5e574d48e007e3c88753ea0128661404d6e5503
8 5837 103 ab02baf0865c4305e5951488c6bc16aef08e9448eca941377c838f4f179c3053
8 5838 103 467e839da1ac438b1c84e03ac7d07aac0c04da92c97509753882147d77c5ce5b
8 5851 103 b769cc762900912804fe533da71ada9b479fbfad0ab5eec0a5554c947dad9469
8 5852 103 6782df0b7f7514f6d087a2ef157f9c7dba2613024b16d703df9b39a1baddb431
8 5855 103 50dc818613984b2e39311ba2bef8fea02d6a1810baa4f8617813aaca6d8d3d81
8 5856 103 4ba66bf950bf3e925d61c5ffe40abfed5af297c2baf64d24d35c5d0986f122de
9 684 103 038db83fc486d8e2a2f0bf1307cf330411bcb8d3280de0f04363e03a891788ee
9 705 103 47d0203fc7efa87c5e0899cded9810e8008bae7ffa5fb219fa180fe4fb3fc99d
9 737 103 3def416f7cf9473d288ada7f349a6de3cc85f6a55652b5f6e78928183e8d0a8d
9 945 103 dc1a300c9799cc1862e7ab8d47daf124daf818cdb636940444a3329dc39b7cd8
9 5285 103 6e41b50faec39fff2c8a3b1e3fbe1a9be03be92a01795aeb52b44ca01bbdbceb
9 5286 103 08829ea59ea2053b3cc8466bcc6262456fc4fa7c36b541dca589bcb8f11afb64
9 5287 103 3002b739c29342797be21077841a2bf058f3357d8779feb100167e7151275b29
9 5288 103 18dba5acfcfa42fb0e81a2577cd68605729b1c7286f1694de70601c6e9a8c8c7
9 5289 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5290 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5291 103 ae649277e05871c3c4ed35a98858855f5217e08cbb4707eb2b2facf34091354b
9 5321 103 0fceba68bfc4c613ec225d54e7e48f7d844f45bc81ae7438571e06073828e9ab
9 5340 103 fea480b09a0ceb4501ace4954c3709ec780fb2ed9a51165491cc78fddba63a6d
9 5364 103 bf59b1c21b97827ded383c4ab9f19188087fc648e03bd3744d05e409b1271331
9 5365 103 b413fac686c6a4eafb5af7c6d529d7649ec25afa72930a727774d91ea7592b17
9 5366 103 55c8c67b34db2897c5dba1fc9040247806efd5d83cdbdc1ecd9c7117f53de7ce
9 5367 103 6c76bcaf72090bb3f9ef6cf965d2d4b701e29581b73426b6cedc75ac1fbfbfe6
9 5368 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5369 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5370 103 55bf77559d812340d4ec28caeca908c54ff15abbb6f6cb99b381c486648bfb32
9 5377 103 0112d6bacdaec0540cb4b7e0bb05182eafa29c9451fc04d75116dc35bdd2ce9a
9 5396 103 9e35b5b2aaa031ba13d038e6086acaffbcc27dd4562b0a3a95c75c1776c87927
9 5397 103 df3d0e5550b7fe7fdda93b9b6bf722f9da9976f537b9a80cb05a91d48b8f38a9
9 5398 103 16d856af65c4d8b7eac800d7a9dafbcaebb8fdbfa8bf081189cff82dda54ddfc
9 5399 103 138160966e9524395c2dbd6abdd645afeb7a59108c637e13fd36ac3a2d86947f
9 5400 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5401 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
9 5402 103 336ad02b9c4781531a47e707990f080c6eae481ffef1366566b8cbbd8c82ea90
9 5409 103 6331e706b842484855f44f78bf7f56ab2cac5ac067ca80d896e6b168cae3410f
9 5555 103 f5f8a4d5e13da1336b459bbce1a8702878c43c24b2fd6bc57348bb57252276be
9 5557 103 6983a2d3d6f476cfe049d2a448738b15c599e71da4ce3f23debe987480dca78a
9 5562 103 0c2f60ebb52f14d68101265e60a98f205ad1d39a0816d59006ade9fd86e17a62
9 5811 103 3297ca22631ddbd9de1c98bb9d869bace3b2ffe4c3c998bf2655ffd9cfdd1850
9 5823 103 396a52d06f6f44b252e432e0df44c96ca908fbc31a962f5715918e9400e6b704
9 5835 103 9d4c51eec8713e3aa9fb5cb5748b1d9c08acd6fef863f3f9e316938dce1f19d0
9 5836 103 a7365c587a168172b3808b61bace9c6f31cc7dfdc8c610651c1f6a785f48e8ae
9 5837 103 437ab03b94f11e871b56ea5126732b3272563049e51d4b6ffdc4f344037a92dc
9 5838 103 99d448ff2ffe8e8f67fcda370604024d085e3dc6712b129bb1b9ef031b98009d
9 5851 103 e62cffc968514d1ec421d1cbfd6bb655651a30e3bc7b56ed7a60a8996d85ed0c
9 5852 103 625b2fb6f647eefa6964c437e63e09ebd473095d9cd37e40acc3c6eb5ae8ac00
9 5855 103 c8759355dde0738754e698d10b0f9969b73a0d4a0ce2ba0361eb8581319594a3
9 5856 103 c886f4cf0567c60b416f4e61a1e6d546bf2ebb0747458a54aeced087cdd5747c
10 684 103 578d7b2c7f97e0ad14b3be2418b6541b2083d9244dba298aefb5addf451d0aad
10 705 103 7a3161123037a1b8c4399f26aeba63626c107ef503b2e669363e7dd5009bd095
10 737 103 8902eec2cdcc50f1c66a7ebc37aa05ea784f77efcb0c303be6237f66954796ce
10 945 103 b02a0a654776290f15ba40a214d7dc09afdea0e58dbdc461b2e6d47965fad118
10 5285 103 0ad6bf7330e23773c111aab47546b32aa82d29930303c12726dd1ffbe9a5fde9
10 5286 103 666287f415d38171c6c9b087e823773c9f183c8e6d41db27202421385b5c9bfb
10 5287 103 b28e6df338e414025b204f1039a5f5de612eb6ad43152e4b7eb6a7bc7c2303e6
10 5288 103 e39f5f9a634eef26b403e3914facbac73c5d0bda10f9b5c449e1f227797729ad
10 5289 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5290 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5291 103 af631ad1caf6652fd63a122c2831548e2b109f6a64bcf326747e09708c72b405
10 5321 103 d8004d17513df3f76d94b21373f8cd77b45603411a261b652dcd603527dfcd9b
10 5340 103 fece20d20b2bf3ac4686b4436ffdd6786d3141d775c4bb3aa91f8d53b9c9e26e
10 5364 103 3cc05d22402b204f5596276a0ede4b059e13699a65836fe0192b50c5f0122587
10 5365 103 8733c3558faffe3812a373cd8e3400fc95b79f17e8445bf60e8049c14796c0db
10 5366 103 1fd40687345a0003c41968c1bc3e5327387dbae733d6a0cf4b5469ae44ada896
10 5367 103 4cf45f82b45f4d88472fb9b76fc74e5bf9285948f6fe9193e46e23cad11266fa
10 5368 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5369 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5370 103 ac595f1c2a0503172062d843a972506e8a8c52a94a5892fa2850b7415ad222a8
10 5377 103 0ad39592a0e343e7067cb4a4530a1562e9bef51a89a169c6408d6f1c0e81cef0
10 5396 103 b172fac272463d1b0c1cdf012a7da0e3cf151dc1fbbc02f33d6f115547ee01e7
10 5397 103 3dbbbe182b1a2b041e472c42ad4b6a4e7d7c9a5caf2abde66726174022e860e4
10 5398 103 e9051beb23bf6209772d4c4a983ed45b30fe04ae974e367fd80d8441148eed8f
10 5399 103 78707b473305060780adc68319e04a22eedb6476d37999eceb534422c72f3bfd
10 5400 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5401 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
10 5402 103 dfee5af63f7177ac1fad75c9b287a45cbea27f08af2e4924188015caf297ff38
10 5409 103 250a5b92ee07ec5bec0388d785396dcd6a03b53d1b4ef4dc91d7798366c70a17
10 5555 103 46826755c64adb919464d77a9f5bb9293866fd68f38d375b7c36125924e39fd0
10 5557 103 61b08295ac3e408958b32780566a929d3514f2ac8bb91c611f6df1403d02939d
10 5562 103 a27167a20e6cf97c054cf85d2e7adae7276639fec5dce6c7c7b89a1c77d4f44a
10 5811 103 e73c87b79154be13cb0af18bb68d461f7f4c2534df5c52a1df56388f0245ae69
10 5823 103 9a6be86d906631230c3c4cc8342d6b1771b37cc965d44971840875beafe6b59c
10 5835 103 d2302bca3cb23ad333d1bf4983d61ee73a7852b3eacae68ea7be7ba9682608f5
10 5836 103 9eff44697921500809767444156da240d56bbd829fe8445816a0dbed71afa8ad
10 5837 103 93a94c5c2f4adf8f2a320d7f29ae4346f41ca9b13369229e6c4e474d27848a52
10 5838 103 f6bf071d352543e217f855007647dcf9c3cf60b891f507de7f0fc6860a9a8caa
10 5851 103 99c9fd8ebdb1c43ca711929538f75c126f847210aca95bb19b98285b476bde33
10 5852 103 b16e00b8c4cfc101b674039af9e7dc96f00bdebb4edb13a3ab99343d321b6f79
10 5855 103 2800e13d970c38b24b074dffe11448ea8932f02f81b8451f0351f9c7f1f2f281
10 5856 103 87d14bd057df888337e58f7f1842c1fb1ea16d2999334eaecd07ac7d7e1c6fd8
11 684 103 d50b8911757e265b472182132c9c8bb3cdb8a4b3d6df82ba3b29bf5eff373296
11 705 103 c1a67fbc772192142c60f472a2f39eaf5974c4dea1fb393a099577078ecc7c2d
11 737 103 15066f68f62143f80a4a15df5434548ae072fdce160a218eeb10a7776122acc5
11 945 103 333d305619bc87605c00bc03444bdb6de187ce8aaf0d4320f365f65b22237d18
11 5285 103 478a19a7c9041e3211e8d575f414c186ffc50cf98dacbbf8daea6fc1030b2515
11 5286 103 83bf68275264db68b64d10e5610447aab572afca09af531e4573818d720b4c5b
11 5287 103 c09c81d9e22728e170e3623c2c8ebfc91dc0d905ec14ead54ccddc6f57428be9
11 5288 103 480e06d1fc8c2c4f7d0c83b9f632fc27c4d17777cb58a1eeb121206b0a035aa4
11 5289 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5290 103 63d419fd55ccdd5736799270cb725e276e3a2f887d8f77748748816a66791bdc
11 5291 103 85f999cfb6f7321c62e65b27148e283ad133abc78f93918f5177d1b3a510b180
11 5321 103 1859e8003e739dd15e1f1689412639e6df2c5f3770c166af6da639bac87f1ed9
11 5340 103 55794574ee94ec6be17c531ff90f764b59ef11646df3b9d9b0a798e46c9a7506
11 5364 103 19f81fbd343155cb239b71a60a2ba9d03aecef8dfa3fbfe4972cb921f692715c
11 5365 103 e0bbd92bb08b3f28ea3e0583422b9d71d77f0939d0c4035c59b8c316e9b3ed00
11 5366 103 45e83c27d7d4a8832e9eb098316506441f34d7d23068485ff41692861054da63
11 5367 103 12203328fa467fef02fb8befa952198ced3340e1d690924a3c9111dc61b236dd
11 5368 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5369 103 0cfca3515137105b1d9046add0fb733c00f32f21ede5eee6e881ddcf701e591a
11 5370 103 6e3b27809e3b47b0ab323ae692d5c90a8f5b7485ef9796c2a33a271164155c6b
11 5377 103 5d7fa17da6b474bc86daf5b73315d107472a08237faa5a09765e0d68b8cfac29
11 5396 103 3f0e437c0129e07e1dd4569062f8d49a29ab58e9a71f0641e32df83836e9e783
11 5397 103 74dbe1bb74b5321c1ec22eb5793011fc11a7f945d0b1920d538bb4de9c465edd
11 5398 103 b5875a841a6b7d54f3ba12e6f072a6908f7ad31b151740c3580e0418dfb300ef
11 5399 103 94fc90f372a38a02153c3ffd37cf632022818ec258353b0f7c0a6bec510174ab
11 5400 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5401 103 67284aa87ccde878eb8c82a52d32c0f74c445470259502a1f0aff0024295f27a
11 5402 103 4047ef229c8ce26880ed290295745ae4d73332169c6a408d13de6ba1450000d2
11 5409 103 f022cfe1c4bf9dab97313d212e2bd8417d22ee025c4c7a99ac25efbe3d2b1a27
11 5555 103 8e17efab940502e1b2866f02b818db31db61626d1e52090ad350ac01e9335532
11 5557 103 6a667b01e248e19e38bc33935ae42cdafc8c97e064f6f1a101ef340a45ed5a9c
11 5562 103 b9dc5f0b62b789f7ad6231330de3123f80ee921381b2c02c62b6b37d34a40099
11 5811 103 3156a32185ca7cedd9547154dcfd864caf86c1533be963ff987cd17cd70f0c73
11 5823 103 c278fe006f765ba84e51587d8134fb4a67df3d85ce0179f3eda891d9dc683904
11 5835 103 2eb1cba8dcc6ededa1709c26fb096687734b6d6da4fc1c7324a4a2990c6048d3
11 5836 103 d9949e3b335bea3dfd97e9bc5180333b5d0b2743d4f9190f3686dbb884ea9b10
11 5837 103 896a5162ddd9857801947e898ac235acd60692bd8cf4b5773dff4f6d2d50fc05
11 5838 103 a83f8709c24a05f0c4fd8dbc7254a69bc91024f874f5c559d1317b627c3e1141
11 5851 103 eb47b7663b8bd9644ea28aee49b73b46f60f0c6001795b7bfcbb2ea69b3f78dd
11 5852 103 c5a1b1830bfe53146b596795143f8e1ddbc6a6ba0c866c7929837abd0517acf5
11 5855 103 18dcb745b4c0b09b52face8cccd8603fc89357e9988bd37658fe9b5ed56ab971
11 5856 103 f598880e9c47a2eb63f653fbc3f83e1d74dd3bff407cadc1497a3a310a3f87ad
12 684 103 913dd2fb5fb392bfa19dd7eb8cc7a7be2520903b9294f2d608285bd110bc6f15
12 705 103 130db68b5cac3ebddabf44d50865032d817a4bef27db8f2d9cd80cc8aac434ed
12 737 103 4f5870736981b4de761a58e7e6244fc5c611309e7c0bde6b22070cab8a4b386e
12 945 103 94b47ad0afec3d069bbb0821403b2308dd857bd5fc8c695119cdbac2cf868adc
12 5285 103 aa77f331adf6e96e6cace7b952af9d473c3e85d11d5e72e2de3c91be3b24f514
12 5286 103 260ce54d11ee631e8e4bd72d7df00a8129bb4d504c2b1523701bf916b6c8645f
12 5287 103 83b756445e07e7379624278cd9dc17312b0449cbef8af705e521386d5c8aecb7
12 5288 103 4086459389c9c6a35a76d3ab4457d476e917a57b94d17928731dcf05dab5db42
12 5289 103 536c6341069998e65902c0b5f70840be7dbf51712f8993083758df0ecbf3b339
12 5290 103 f16b157bf743e27d4c81343fc5adc8fd6c6390aa7ea515c15374bdd5db178f98
12 5291 103 5193a5f6432c166a13b0db8f17f24cb5626f5704dee1610c9c24baf7df2b0d8e
12 5321 103 f49b5adcf65d3637f4255eedf94c8eebc056eae89cff66f9a86d673bf7cddd2e
12 5340 103 f8e77158b895aaf95cef0627b866cfd45102388a0eee1ace144001b593e8f8a5
12 5364 103 74398cdab1b74a1f9bed1a1efc7c0be7ace34ba9bfed87ede0100e0d1851e588
12 5365 103 5ee590ea6509bdd8d0213f505cc28a77bd14e8fdc0e3932c0816b054115bd722
12 5366 103 48ea25e85348c03200607faa79abb89a97d5ecd5fbbc64bd200ec9349b771efd
12 5367 103 06fb756dbda4f3b7af22381d8a9705291d7c62efcc2f566c39f32835d7284511
12 5368 103 ba13416ab285bdc19bce97089c092fac804d0f93f88db977a736dd3202e9977b
12 5369 103 c8c0dabb341fb3671b1a1f4f36147fc179951ac88a86941d8007c6d75d229136
12 5370 103 f55f5eb608b3e14d629ab95e2efec4645de3c26aa19b72e7477ba7011a74d558
12 5377 103 a9578c40c7e180336bd4b4f64c53ed415eb149fd74b0fbf0f7087ab43368ef77
12 5396 103 8f87e38166373386c6eff199d93f53e4da204c87d88bc8f8b9deb1f6135bf35c
12 5397 103 1c0fb78c746e4db6a9a162dc769b237a0efa2048cd1ce011e03305ae6d0a6638
12 5398 103 b5a3a59bc2abb2372b66038f0fd82f417f6e329f2589a0aa39b9ac9cc897dc50
12 5399 103 d37923d676171734b664fcfb73bb125d6bd0587800d4ba3ceb0ac32cd325f305
12 5400 103 78824f2a9a5252e6c7b80680509a37f9fce7507cc6dbed7e6dfd5e395d157cd0
12 5401 103 489a432afac25c5e213ae53287abb84552bdeee97204075117fc500fe9dda502
12 5402 103 824ac615fb65509a5d7a8c43976b2d69796fee18e645c753becfc408075e6ff5
12 5409 103 a3e6ec867806a60a368c8ee66c59183fa3c8a77c45bfb91d294c124c0f82e898
12 5555 103 326b6e31069b90c2999ffd2f2c8d91a307623d89dfb87500ece3b12d9d1be580
12 5557 103 0e6745488da4669346dd5c6031a5fb674928cd366e290a61e679b713318a9d3e
12 5562 103 9bc8682d7d6b462f28b8a86fa4d3470dc456f5f03645dc672772ba79322b6de8
12 5811 103 bfdc0416cc58c3c0f6c248f95d1f3d3bead83e9d9d95b46fa9bed81f79ce2e70
12 5823 103 7e05afe57adcca1846d92dbad1c0d2dd8b00bde97acce03e2051d84c9c5fd909
12 5835 103 3f7ab93d666237d2be2fbd498fc2f2cc30461881ea4dae80a31ee65ecf4d5cbc
12 5836 103 90e7346d8685aebd7755699dafa8c5e035cb798db42cc1c0ec3acf4832656290
12 5837 103 870e0e3f7c811461e81040647118d611b44c77184e7b3d00731e22fea7606f9d
12 5838 103 467d1367bbc19f66c768665ac111508fea64f98207a970e0220fa8af5af1c8e1
12 5851 103 1b006a7dcd01365676f1a5687f9912cafcaa12af417bd5acdffcc24651abdad7
12 5852 103 c37da3398b26e11a5f69f80f356f30ff8705375c1fb3f7cb4fee64c60ba08113
12 5855 103 671dc7a24e415e5cb869bc82e16fb705caab90509d40608fd94685301c000b83
12 5856 103 2d4e8487b480a3e713448c0fd145a85de8df436e508f45ad5b51e4bfd28bdb15
"""
for _ln in ANGA_PINS.split('\n'):
    if _ln.strip():
        _l, _s, _g, _p = _ln.split()
        SOURCES[f'anga_L{_l}_S{_s}_G{_g}'] = (ANGA.format(s=_s, l=_l, g=_g), _p, f'National Greenhouse Accounts API (NIR 2024), '
                                              f'{LOC[int(_l)]}, sector {_s}, gas {_g}; raw response pinned (changing source)')

if __name__ == '__main__':
    main()
