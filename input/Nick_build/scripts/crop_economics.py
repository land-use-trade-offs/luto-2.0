#!/usr/bin/env python
"""Build LUTO's 2021-base crop economics (agec_crops.h5) and the crop energy and lifecycle GHG fields (agGHG_crops.h5).

Sources (public, downloaded at run time, sha256-checked against pinned values):
  - ABS 6401.0 Consumer Price Index, Australia, August 2026 (series A2325846C, all groups, quarterly): FY2021 dollars.
  - ABS 7121.0 Agricultural Commodities and 7503.0 Value of Agricultural Commodities Produced, FY2015-16 to FY2021-22
    (state files; SA2 files for FY2015-16 and FY2020-21).
  - ABS Australian Agriculture: Broadacre Crops, Horticulture and Livestock data cubes, 2022-23 to 2024-25.
  - ABS 4618.0 Water Use on Australian Farms, FY2016-17 to FY2020-21.
  - ABARES Australian Crop Report, September 2026 (state tables) and the ABARES Farm Data Portal (state and national files).
  - ABS ASGS July 2011 SA2 boundaries (GDA94), rasterised on the LUTO grid for the FY2015-16 SA2 yield index.
Inputs from the other 2021-base tools: sa2_2021_cells.npy (nlum_map.py); the final lumap.h5 and lmmap.h5
(livestock_map.py); x_mrj.npy (eligibility.py); agGHG_crops.h5 (ghg_factors.py).
Inherited 2010-build inputs from --input (inherited_2010/, copies of 2010-build files), each sha256 recorded in the provenance file: agec_crops.h5 (the file
layout; the cost-to-revenue ratios and cost shares of the ratio-group crops, the cost relativities of the FDP-group crops, the
irrigated to dry yield ratio, the water relativities within a group), agGHG_crops.h5 (the levels of the energy and lifecycle
fields), water_delivery_price.h5 (WP), lumap.h5 and lmmap.h5 (2010 map, cells preferred for those levels), real_area.h5,
state_id.npy, ag_landuses.csv and NLUM_2010-11_mask.tif. The outputs are therefore not built from public sources alone.

Method. Per hectare of cropping land, FY2021 dollars, windows FY2017-25 (production, prices, costs)
and FY2017-21 (water):
  1. Production and farm-gate value by LUTO crop and state: ABARES crop report and ABS 7121 / AA levied production; unit value
     = ABS 7503 local value / production or AA local value of levied crop / levied production (broadacre); ABS AA
     horticulture FY2021-25 with FY2017-20 ABS core items spliced to it; Other non-cereal crops: ABS 7503 local value
     at the state grain-group farm-gate price (inferred).
  2. Yield = K(crop, state) x SA2 yield index (x the 2010 file's irrigated / dry ratio on irrigated land), K so that placed
     tonnes on allocated cells equal the window mean, K within one-third to three times the national K; on cells without
     the crop on the base map (either lm) the index is capped at 1, so their yield is at most K (x the ratio). P1 = state
     farm-gate unit value.
  3. Costs: FDP group (cereals, oilseeds, legumes, hay) from ABARES FDP All Broadacre crop costs per sown hectare x sown
     share, split across (crop, lm) by the 2010 file's relativities; QC = 0 (prices net of marketing). Ratio group
     (13 crops): the 2010 file's cost-to-revenue ratio x new revenue - INTERIM pending the cost budgets.
  4. WR: ABS 4618 volume by group and state over LUTO irrigated area, x the SA2_2021 application-rate index and the 2010
     relativity within the group, band as yields. Dry WR = 0.
  5. WP = the 2010 water_delivery_price.h5 with the state median on blank cells - INTERIM pending the ACCC reports.
  6. agGHG_crops.h5 = the ghg_factors.py file with the 8 energy and lifecycle fields rebuilt on allocated or eligible cells:
     2010 intensities (state mean) x sown share; IRRIG = new WR x the 2010 IRRIG / WR ratio (2010-basis levels,
     INTERIM). Dry Pears take dry Apples; dry Pears SOIL NaN cells take the SA2, state, then national mean.
  Values on every allocated or eligible cell of the (lm, lu); NaN elsewhere. Intermediate tables pass through the same CSV
  write and read (and float32 storage) as the earlier multi-script build, so the outputs reproduce it.
Outputs in --out: agec_crops.h5, agGHG_crops.h5 (layouts of the 2010-base files plus the (*, dry, Pears) columns);
crop_production_window.csv; crop_yield_calibration.csv and crop_water_calibration.csv (with the calibration labels);
interim_items_crop_economics.csv (INTERIM and inferred items); agec_crops.nc and agGHG_crops.nc (NetCDF copies in the
jinzhu layout, AGEC_CROPS and AGGHG_CROPS over (column, cell), levels field, lm, lu; nc_layout.py);
provenance.csv (last).

Run as a plain script from the repository root (no luto imports):

    python input/Nick_build/scripts/crop_economics.py --input input/Nick_build/inherited_2010 --nlum-map input/Nick_build/nlum_map --map-dir input/Nick_build/livestock_map
        --x-mrj input/Nick_build/eligibility/x_mrj.npy --ghg-crops input/Nick_build/ghg_factors/agGHG_crops.h5 --out input/Nick_build/crop_economics
        --cache input/Nick_build/_cache [--offline] [--source NAME=PATH]

Dependencies: numpy, pandas, PyTables, openpyxl, xlrd, rasterio, geopandas, scipy, xarray, netCDF4. No luto imports
(nc_layout.py is a sibling module).
Memory: peak resident memory 130.3 GiB on the full grid (measured 19 October 2026 in a full input build: resident set
size of the process and its children, sampled every 0.5 s with psutil on Windows; 130.5 GiB in the 22 October 2026 build);
run time 54 minutes with the NetCDF copies (measured the same way, 22 October 2026); run on a machine with at least 256 GB
of RAM.
Output values are deterministic; HDF5 file bytes carry a group timestamp, so provenance records dataset content hashes.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
import zipfile
from importlib import metadata

import numpy as np
import pandas as pd

import nc_layout as NC
import rasterio

# ----------------------------------------------------------------------------- sources and pins
ABS_OLD = 'https://www.abs.gov.au/ausstats/subscriber.nsf/log?openagent&'
ABS_OLD2 = 'https://www.abs.gov.au/AUSSTATS/subscriber.nsf/log?openagent&'
ABS_AC = 'https://www.abs.gov.au/statistics/industry/agriculture/agricultural-commodities-australia/'
ABS_VA = 'https://www.abs.gov.au/statistics/industry/agriculture/value-agricultural-commodities-produced-australia/'
ABS_AA = 'https://www.abs.gov.au/statistics/industry/agriculture/australian-agriculture-'
ABS_WU = 'https://www.abs.gov.au/statistics/industry/agriculture/water-use-australian-farms/'
SOURCES = {   # name (the raw file name; used as the row label of the ABS tables): (url, sha256 of the raw file, note)
    'CPI-Quarterly_aug2026.zip': (
        'https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/aug-2026/'
        'CPI-Quarterly.zip', '0b97b11375f6cdbf87b59a85d9c4cd990950e43d09aeea4504221f72d9af3cbd',
        'ABS 6401.0 CPI, Australia, Aug 2026, CC BY 4.0'),
    'AustCropRrt20260901_v1.0.0_asset3.xlsx': (
        'https://daff.ent.sirsidynix.net.au/client/en_AU/search/asset/1038523/3/__AustCropRrt20260901_v1.0.0.xlsx',
        '807ae8af9214ffc4c9e636d79ea8c7091eeac0950ca865917677fc1934e3e38f',
        'ABARES Australian Crop Report, September 2026, state tables, CC BY 4.0'),
    'fdp-state-historical.csv': (
        'https://www.agriculture.gov.au/sites/default/files/documents/fdp-state-historical.csv',
        '65d31d827532ad8246277596202a2eb326ab23d0b53f7c1815905c8f828f828b', 'ABARES Farm Data Portal, state, CC BY 4.0'),
    'fdp-national-historical.csv': (
        'https://www.agriculture.gov.au/sites/default/files/documents/fdp-national-historical.csv',
        'ffb28831b9b8e9ff658381cb0761f9b8ea3817204c02cf075b356955271edad0', 'ABARES Farm Data Portal, national, CC BY 4.0'),
    '1270055001_sa2_2011_aust_shape.zip': (
        ABS_OLD2 + '1270055001_sa2_2011_aust_shape.zip&1270.0.55.001&Data%20Cubes&7130A5514535C5FCCA257801000D3FBD&0&'
        'July%202011&23.12.2010&Latest', 'f18ca51d3fe00ee3c7d678eef5dcfdf745ecfbf45ea3e499aee287620eb0c6d1',
        'ABS 1270.0.55.001 ASGS Vol 1, July 2011, SA2 boundaries, GDA94, CC BY'),
    # ABS 7121.0 and 7503.0
    '7121do001_201516_state.xls': (
        ABS_OLD + '7121do001_201516.xls&7121.0&Data%20Cubes&DC87B0B6B88A1FCFCA25816A00177E45&0&2015-16&28.07.2017&Previous',
        'faa084ae4671c9a1360fb8621278f823088d85dbf6cff859d3286632c81a7660', 'ABS 7121.0 2015-16, state, CC BY'),
    '7121do004_201516_SA2.zip': (
        ABS_OLD + '7121do004_201516.zip&7121.0&Data%20Cubes&3B9C809B49A11744CA2581C900105EE7&0&2015-16&31.10.2017&Previous',
        'fc078b828d7eb57309ccea7ddd7cada8cf49df4ee1b3d84a2d8e3a73b4a184bb', 'ABS 7121.0 2015-16, SA2 (ASGS 2011), CC BY'),
    '75030do001_201516_state.xls': (
        ABS_OLD + '75030do001_201516.xls&7503.0&Data%20Cubes&548E8E628E7D11D8CA25816A00177ED8&0&2015-16&28.07.2017&Previous',
        'afeee00a3a9522a3cbaf33b3557b5dd9a729f7dcae5d8284c109c11fa407ef8d', 'ABS 7503.0 2015-16, state, CC BY'),
    '75030do005_201516.zip': (
        ABS_OLD + '75030do005_201516.zip&7503.0&Data%20Cubes&5A85B6D5B150DE28CA2581C900105FFD&0&2015-16&31.10.2017&Previous',
        '7a1fce19ad13871d700a92602d46a49efeb41c76201befda6406bc8bd1384e3d', 'ABS 7503.0 2015-16, SA2 (ASGS 2011), CC BY'),
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
    'abs_7121_2020_21': (ABS_AC + '2020-21/AGCDCASGS202021.xlsx',
                         '0fdbd3e398b598a24c8bc315e5bd7cae36e8ea4fb6f6cea5a7b745e0950bedf5',
                         'ABS 7121.0 2020-21, ASGS Ed 3 cube (file AGCDCASGS202021.xlsx), CC BY 4.0'),
    'VACPDCASGS202021.xlsx': (ABS_VA + '2020-21/VACPDCASGS202021.xlsx',
                              'f737632f72d77f36cdff601487f2618cde703adb818727c6e5f7378209ea7eab', 'ABS 7503.0 2020-21, CC BY 4.0'),
    'AGCDCNAT_STATE202122.xlsx': (ABS_AC + '2021-22/AGCDCNAT_STATE202122.xlsx',
                                  'f4da88e46584788dcc3af46f62b37d922e25307ea01068c983824e217286afb6', 'ABS 7121.0 2021-22, CC BY 4.0'),
    'VACPDCNAT_STATE202122.xlsx': (ABS_VA + '2021-22/VACPDCNAT_STATE202122.xlsx',
                                   'cade129e4061b00361c1f612ad8a2ebcc43891ee1e7cfeb153987fdf413287d3', 'ABS 7503.0 2021-22, CC BY 4.0'),
    # ABS Australian Agriculture
    'AABDC_Winter_202223.xlsx': (ABS_AA + 'broadacre-crops/2022-23/AABDC_Winter_Broadacre_202223.xlsx',
                                 '096b29c87e9cbe909e04c806df28424c3d221be525c822bb18d896cb4369859a', 'ABS AA broadacre, CC BY 4.0'),
    'AABDC_Summer_202223.xlsx': (ABS_AA + 'broadacre-crops/2022-23/AABDC_Summer_Broadacre_202223.xlsx',
                                 '3779e0540df49d2cba8c2c6732c096976309b624c533962ad6a3850a035a2a81', 'ABS AA broadacre, CC BY 4.0'),
    'AABDC_Sugarcane_202223.xlsx': (ABS_AA + 'broadacre-crops/2022-23/AABDC_Sugarcane_202223.xlsx',
                                    '6a1b0f03fb2b64098c0dff2d8339db95bfaa9a75e475b12008d5f560f3cf3397', 'ABS AA broadacre, CC BY 4.0'),
    'AABDC_Winter_202324.xlsx': (ABS_AA + 'broadacre-crops/2023-24/AABDC_Winter_Broadacre_202324.xlsx',
                                 'a38d5a4ab90486aea72cdae35c9985b63d199bc70bfa082712f557a26315fd0b', 'ABS AA broadacre, CC BY 4.0'),
    'AABDC_Summer_202324.xlsx': (ABS_AA + 'broadacre-crops/2023-24/AABDC_Summer_202324.xlsx',
                                 '1f5310dcdd39245b441ffb1f0608487b8edb60bf47102484d3ce772c47c0dfa5', 'ABS AA broadacre, CC BY 4.0'),
    'AABDC_Sugarcane_202324.xlsx': (ABS_AA + 'broadacre-crops/2023-24/AABDC_Sugarcane_202324.xlsx',
                                    '391662423b4959fc05f13436dd3290584a4dfeb9f8347912bde8e9c33adafe43', 'ABS AA broadacre, CC BY 4.0'),
    'AABDC_Winter_202425.xlsx': (ABS_AA + 'broadacre-crops/2024-25/AABDC_Winter_202425.xlsx',
                                 'f93d9377c55c71cb25b62df82621d70c797af835f424a501d028aa42f1aabdaa', 'ABS AA broadacre, CC BY 4.0'),
    'AABDC_Summer_202425.xlsx': (ABS_AA + 'broadacre-crops/2024-25/AABDC_Summer_202425.xlsx',
                                 '8be53878688fd755e94eb41ea4ae73ae64b2dd3b19ed896775165335eb0a43d6', 'ABS AA broadacre, CC BY 4.0'),
    'AABDC_Sugarcane_202425.xlsx': (ABS_AA + 'broadacre-crops/2024-25/AABDC_Sugarcane_202425.xlsx',
                                    'a6d137d3bc29bf5f068bb31c66b8ce040c12ba2b9ca7e5ef1ea1a8ee430e410f', 'ABS AA broadacre, CC BY 4.0'),
    'AAHDC_Hort_202223.xlsx': (ABS_AA + 'horticulture/2022-23/AAHDC_Aust_Horticulture_202223.xlsx',
                               'f791b3d44035fd048e27f72e8df4670cbae95895f2fd336cbf96092f321b77e8', 'ABS AA horticulture, CC BY 4.0'),
    'AAHDC_Hort_202324.xlsx': (ABS_AA + 'horticulture/2023-24/AAHDC_Aust_Horticulture_202324.xlsx',
                               '83cff2463adf089f50501618d026bc4ccb07cd3df1a7d20f1b46fc8e084b051b', 'ABS AA horticulture, CC BY 4.0'),
    'AAHDC_Hort_202425.xlsx': (ABS_AA + 'horticulture/2024-25/AAHDC_Aust_Horticulture_202425.xlsx',
                               '4677d16c1b16f1ebe42221c9aa4f6ed7b59ac9bd044389f7f40642a2ed1c9fa2', 'ABS AA horticulture, CC BY 4.0'),
    'AAHDC_Hort_Avocado_and_macadamia_202223.xlsx': (
        ABS_AA + 'horticulture/2022-23/AAHDC_Hort_Avocado%20and%20macadamia_202223.xlsx',
        'f395ce515c7f36755d350a8278ec9672783613e7293ff6730de391afaea8394b', 'ABS AA horticulture (experimental), CC BY 4.0'),
    'AAHDC_Hort_Selected_Commodities_202324.xlsx': (
        ABS_AA + 'horticulture/2023-24/AAHDC_Hort_Selected_Commodities_202324.xlsx',
        'cac1821ef8cb0c3f705eab48896c0e4714c76a0ff16e37263ddd393adec60b19', 'ABS AA horticulture (experimental), CC BY 4.0'),
    'AAHDC_Hort_Selected_Commodities_202425.xlsx': (
        ABS_AA + 'horticulture/2024-25/AAHDC_Hort_Selected_Commodities_202425.xlsx',
        'e566027e60066d98f2fb682abb61db14c95752340fd51a67f3947c17d022c6b6', 'ABS AA horticulture (experimental), CC BY 4.0'),
    'AALDC_Value_livestock_202223.xlsx': (
        ABS_AA + 'livestock/2022-23/AALDC_Value%20of%20livestock%20and%20products.xlsx',
        'f37a0b3c9fee9472d5118681954802e041b020bbda8a365520896b834e62aa8b', 'ABS AA livestock, CC BY 4.0'),
    'AALDC_Value_of_livestock_and_products_2023-24.xlsx': (
        ABS_AA + 'livestock/2023-24/AALDC_Value%20of%20livestock%20and%20products%202023-24.xlsx',
        '9380a74189d5fa08017ffed5021ad97f511f5fb80cabfffa243da5d628f89241', 'ABS AA livestock, CC BY 4.0'),
    'AALDC_Value_livestock_202425.xlsx': (
        ABS_AA + 'livestock/2024-25/AALDC_Value%20of%20livestock%20and%20products%202024-25.xlsx',
        '6bd3cf8d7c3dc16527cf87c729948f3591b05ec66780b42d05a1bcd8ec1b11cd', 'ABS AA livestock, CC BY 4.0'),
    # ABS 4618.0
    '46180do002_201617.xls': (
        ABS_OLD2 + '46180do002_201617.xls&4618.0&Data%20Cubes&3AB49B0730AFF6D2CA2582910013A4F1&0&2016-17&21.05.2018&Latest',
        '3f5ac8615b081aad60cd44e772825c02d564471d97caf5e45ee22cddc931513a', 'ABS 4618.0 2016-17, SA4, CC BY'),
    '46180do003_201617.csv': (
        ABS_OLD2 + '46180do003_201617.csv&4618.0&Data%20Cubes&1C5CB2D1CB896BF7CA2582910013A521&0&2016-17&21.05.2018&Latest',
        '33e527cba3f15542aa7c5c7b19c342efabafbfa76fdcbec89efc4625e2f6f486', 'ABS 4618.0 2016-17, NRM, CC BY'),
    '46180do004_201718.csv': (ABS_WU + '2017-18/46180do004_201718.csv',
                              '0a4dd0aef3367587dd94f7c433ac7c9c564f7a8b218b73aaee4788930c3acbdb', 'ABS 4618.0 2017-18, SA4, CC BY 4.0'),
    '46180do003_201718.csv': (ABS_WU + '2017-18/46180do003_201718.csv',
                              '1971fcb7bea3d8db4da236adb56be1766b73fd422d3944ef2f02f8f3dae6bc74', 'ABS 4618.0 2017-18, NRM, CC BY 4.0'),
    '46180do004_201819.csv': (ABS_WU + '2018-19/46180do004_201819.csv',
                              '1fd573f409cf1fd418c7385eb788c48397e5fa26699accd821cc131f889ac4db', 'ABS 4618.0 2018-19, SA4, CC BY 4.0'),
    '46180do003_201819.csv': (ABS_WU + '2018-19/46180do003_201819.csv',
                              'b1f5ddee74908e635365e5f1eebb043c71c32a8f99dfa46a6115c03cfddfbe5c', 'ABS 4618.0 2018-19, NRM, CC BY 4.0'),
    '46180DO001_201920_SA4.xlsx': (ABS_WU + '2019-20/46180DO001_201920.xlsx',
                                   '07dc36e1d3dbb254e89c8c7e3e5afed233e9007c26205567dadcb1248fa42e51', 'ABS 4618.0 2019-20, SA4, CC BY 4.0'),
    '46180DO002_201920_NRM_MDB.xlsx': (ABS_WU + '2019-20/46180DO002_201920.xlsx',
                                       'e61c2740abd95cd554a1320fbfa9facf8366b7cb0bea2c1deb75bda9383b33b7',
                                       'ABS 4618.0 2019-20, NRM and MDB, CC BY 4.0'),
    'WUAFDCASGS202021.xlsx': (ABS_WU + '2020-21/WUAFDCASGS202021.xlsx',
                              '1453aacfcb132edb1609b36aad157530d02e2da3387702892b34f2368d6cd2eb', 'ABS 4618.0 2020-21, ASGS Ed 3, CC BY 4.0'),
    'WUAFDCNRM202021.xlsx': (ABS_WU + '2020-21/WUAFDCNRM202021.xlsx',
                             '335c0d7bef83281a6a248e9ef12054b4faca2801ca40c1a959d7bfd6ae91f448', 'ABS 4618.0 2020-21, NRM, CC BY 4.0'),
    'WUAFDCMDB202021.xlsx': (ABS_WU + '2020-21/WUAFDCMDB202021.xlsx',
                             'b3e9c973a07d73ca89fe685a5d3d09f42471b4c8b67ab7b58190c05bba8c47dd', 'ABS 4618.0 2020-21, MDB, CC BY 4.0'),
}
ABS_LABEL = {'abs_7121_2020_21': 'AGCDCASGS202021.xlsx'}     # row label (raw file name) of a source with another name
INHERITED = ['agec_crops.h5', 'agGHG_crops.h5', 'water_delivery_price.h5', 'lumap.h5', 'lmmap.h5', 'real_area.h5',
             'state_id.npy', 'ag_landuses.csv', 'NLUM_2010-11_mask.tif']
NCELL = 6956407
ST = {1: 'NSW', 2: 'VIC', 3: 'QLD', 4: 'SA', 5: 'WA', 6: 'TAS', 7: 'NT', 8: 'ACT', 9: 'OT'}
STATES = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
LMN = {0: 'dry', 1: 'irr'}

# ============================================================================= reference table (hand-entered)
# Analyst choices of the method, carried unchanged; each with its basis. Changing any is a ruling.
REF = {
    'cpi_series': 'A2325846C',      # ABS 6401.0 Table 1/2 member 6401018.xlsx, All groups CPI, Australia, sheet Data1 col B
    'base_fy': 2021,                # FY2020-21 dollars
    'window': range(2017, 2026),    # production, price and FDP window FY2017-25
    'band': (1 / 3, 3),             # SA2 index clip and the state factor within [1/3, 3] x national factor
    'uncropped_index_cap': 1.0,     # SA2 index cap on cells without the crop on the base map (either lm): at most the state factor
    'sown_clip': (0.1, 1.5),        # sown share clip for the energy and lifecycle fields
    'water_window': (2017, 2021),   # ABS 4618 FY2017-21
}
# ABS file roles: (FY_end, series, kind)
ABS_STATE_FILES = [('7121do001_201516_state.xls', 2016, '7121', 'xls'), ('75030do001_201516_state.xls', 2016, '7503', 'xls'),
                   ('71210do001_201617_state.xls', 2017, '7121', 'xls'), ('75030do001_201617_state.xls', 2017, '7503', 'xls'),
                   ('71210do005_201718.csv', 2018, '7121', 'csv'), ('75030do003_201718.csv', 2018, '7503', 'csv'),
                   ('71210do005_201819.csv', 2019, '7121', 'csv'), ('75030do003_201819.csv', 2019, '7503', 'csv'),
                   ('71210DO001_201920_SA4.xlsx', 2020, '7121', 'xls'), ('75030DO001_201920_SA4.xlsx', 2020, '7503', 'xls'),
                   ('AGCDCASGS202021.xlsx', 2021, '7121', 'xls'), ('VACPDCASGS202021.xlsx', 2021, '7503', 'xls'),
                   ('AGCDCNAT_STATE202122.xlsx', 2022, '7121', 'xls'), ('VACPDCNAT_STATE202122.xlsx', 2022, '7503', 'xls')]
SHEETS_7503_201516 = ['Aust.', 'NSW', 'Vic.', 'Qld', 'SA', 'WA', 'Tas.', 'NT', 'ACT']
AA_FILES = [('AABDC_Winter_202223.xlsx', 'AA_broadacre'), ('AABDC_Summer_202223.xlsx', 'AA_broadacre'),
            ('AABDC_Sugarcane_202223.xlsx', 'AA_broadacre'), ('AABDC_Winter_202324.xlsx', 'AA_broadacre'),
            ('AABDC_Summer_202324.xlsx', 'AA_broadacre'), ('AABDC_Sugarcane_202324.xlsx', 'AA_broadacre'),
            ('AABDC_Winter_202425.xlsx', 'AA_broadacre'), ('AABDC_Summer_202425.xlsx', 'AA_broadacre'),
            ('AAHDC_Hort_202223.xlsx', 'AA_hort'), ('AAHDC_Hort_202324.xlsx', 'AA_hort'), ('AAHDC_Hort_202425.xlsx', 'AA_hort'),
            ('AAHDC_Hort_Avocado_and_macadamia_202223.xlsx', 'AA_hort_experimental'),
            ('AAHDC_Hort_Selected_Commodities_202324.xlsx', 'AA_hort_experimental'),
            ('AAHDC_Hort_Selected_Commodities_202425.xlsx', 'AA_hort_experimental'),
            ('AALDC_Value_livestock_202223.xlsx', 'AA_livestock'),
            ('AALDC_Value_of_livestock_and_products_2023-24.xlsx', 'AA_livestock'),
            ('AALDC_Value_livestock_202425.xlsx', 'AA_livestock'),
            ('AABDC_Sugarcane_202425.xlsx', 'AA_broadacre')]
# ABS 4618 files by FY; first source wins on duplicate keys
WATER_FILES = {2017: [('46180do002_201617.xls', 'SA4'), ('46180do003_201617.csv', 'NRM')],
               2018: [('46180do004_201718.csv', 'SA4'), ('46180do003_201718.csv', 'NRM')],
               2019: [('46180do004_201819.csv', 'SA4'), ('46180do003_201819.csv', 'NRM')],
               2020: [('46180DO001_201920_SA4.xlsx', 'SA4'), ('46180DO002_201920_NRM_MDB.xlsx', 'NRM')],
               2021: [('WUAFDCASGS202021.xlsx', 'ASGS3'), ('WUAFDCNRM202021.xlsx', 'NRM'), ('WUAFDCMDB202021.xlsx', 'NRM')]}
# Broadacre commodity -> (ABARES crop, ABS 7121 production pattern, ABS 7503 local value pattern, AA levied prefix)
COM = {
    'wheat': ('Wheat', r'Wheat for grain - Production', r'Cereal crops - Wheat for grain$', 'Wheat for grain or seed'),
    'barley': ('Barley', r'Barley for grain - Production', r'Cereal crops - Barley for grain$', 'Barley for grain or seed'),
    'oats': ('Oats', r'Oats for grain - Production', r'Cereal crops - Oats for grain$', 'Oats for grain or seed'),
    'triticale': ('Triticale', r'(?:Triticale for grain|All other cereals for grain or seed) - Production',
                  r'Cereal crops - (?:Triticale for grain|All other cereals for grain or seed)(?: \(a\))?$', None),
    'sorghum': ('Grain sorghum', r'Sorghum for grain - Production', r'Cereal crops - Sorghum for grain$', 'Sorghum for grain or seed'),
    'maize': ('Corn (maize)', r'Maize for grain - Production', r'Cereal crops - Maize for grain$', 'Maize for grain or seed'),
    'rice': ('Rice', r'Rice for grain - Production', r'Cereal crops - Rice for grain$', 'Rice for grain or seed'),
    'canola': ('Canola', r'Canola - Production', r'Oilseeds - Canola$', 'Canola for grain or seed'),
    'sunflower': ('Sunflower', r'Other oilseeds - Production', r'Oilseeds - Other oilseeds$', None),
    'soybeans': ('Soybeans', r'Other oilseeds - Production', r'Oilseeds - Other oilseeds$', None),
    'chickpeas': ('Chickpeas', r'Chickpeas - Production', r'Chickpeas$', 'Chickpeas for grain or seed'),
    'lentils': ('Lentils', r'Lentils - Production', r'Lentils$', 'Lentils for grain or seed'),
    'lupins': ('Lupins', r'Lupins - Production', r'Lupins$', 'Lupins for grain or seed'),
    'fieldpeas': ('Field peas', r'Other pulses - Production', r'Other pulses(?: \(b\))?$', 'Field peas for grain or seed'),
    'faba': ('Faba beans', r'Other pulses - Production', r'Other pulses(?: \(b\))?$', 'Faba beans for grain or seed'),
    'cotton': ('Cotton lint a', r'Cotton \(irrigated and non-irrigated\) - Lint production',
               r'Cotton lint \(irrigated and non-irrigated\)$', None),
    'sugar': (None, r'Sugar cane - Cut for crushing - Production', r'Sugar cane - Cut for crushing$', 'Sugarcane for processing'),
    'hay': (None, r'(?:cereal and other crops cut for hay - (?:Total - )?Production|cereal and other crops cut for hay - Total '
                  r'production|Total crops cut for hay - Production)', r'^Hay - Total$', None),
}
LU_COM = {'Winter cereals': ['wheat', 'barley', 'oats', 'triticale'], 'Summer cereals': ['sorghum', 'maize'], 'Rice': ['rice'],
          'Winter oilseeds': ['canola'], 'Summer oilseeds': ['sunflower', 'soybeans'],
          'Winter legumes': ['chickpeas', 'lentils', 'lupins', 'fieldpeas', 'faba'], 'Cotton': ['cotton'], 'Sugar': ['sugar'],
          'Hay': ['hay']}
# Horticulture: AA item names (first token of item_desc) and the ABS 7121/7503 core items of the FY2017-20 splice
AA_HORT = {
    'Apples': ['Apples'], 'Pears': ['Pears', 'Nashi'], 'Citrus': ['Oranges', 'Mandarins', 'Lemons and limes', 'Grapefruit'],
    'Nuts': ['Almonds', 'Macadamias', 'Pecans', 'Pistachios', 'Hazelnuts', 'Chestnuts', 'Walnuts'],
    'Stone fruit': ['Nectarines and peaches', 'Cherries', 'Apricots', 'Plums', 'Olives', 'Persimmons'],
    'Tropical stone fruit': ['Mangoes', 'Avocados', 'Lychees', 'Custard Apples'],
    'Plantation fruit': ['Bananas', 'Pineapples', 'Papaya/Pawpaw', 'Passionfruit', 'Strawberries', 'Blueberries', 'Rubus berries',
                         'Kiwifruit'],
    'Grapes': ['Wine grapes', 'Table grapes', 'Dried grapes'],
    'Vegetables': ['Artichokes', 'Asparagus', 'Beans, fresh', 'Beetroot', 'Broccoli/Baby broccoli', 'Brussels sprouts', 'Cabbage',
                   'Capsicums', 'Carrots', 'Cauliflower', 'Celery', 'Chillies', 'Cucumbers', 'Eggplant',
                   'English spinach/Silverbeet/Kale', 'Fennel', 'Garlic', 'Ginger', 'Leafy Asian vegetables',
                   'Leafy salad vegetables', 'Leeks', 'Lettuce (head)', 'Muskmelons', 'Onions',
                   'Parsley and other fresh herbs excluding fennel', 'Parsnips', 'Peas', 'Potatoes', 'Pumpkins', 'Sweet corn',
                   'Sweet potatoes', 'Tomatoes', 'Watermelons', 'Zucchini'],
}
CORE = {
    'Apples': ['Apples'], 'Pears': ['Pears \\(including Nashi\\)'], 'Citrus': ['Oranges', 'Mandarins'],
    'Nuts': ['Almonds', 'Macadamias'], 'Stone fruit': ['Cherries', 'Nectarines', 'Peaches', 'Olives'],
    'Tropical stone fruit': ['Mangoes', 'Avocados'], 'Plantation fruit': ['Bananas', 'Pineapples', 'Strawberries'],
    'Grapes': ['Grapes for wine production', 'Grapes for all other uses'],
    'Vegetables': ['Potatoes', 'Carrots', 'Onions', 'Pumpkins \\(including butternut\\)|Pumpkins', 'Sweet corn', 'Melons.*',
                   'Cabbages', 'Cauliflowers', 'Broccoli', 'Beans \\(including french and runner(?: beans)?\\)', 'Tomatoes',
                   'Lettuces'],
}
CORE_VAL_NAME = {'Grapes for wine production': 'Grapes - Wine production', 'Grapes for all other uses': 'Grapes - All other uses'}
GRAIN_GROUPS = ['Winter cereals', 'Summer cereals', 'Winter legumes', 'Summer legumes', 'Winter oilseeds', 'Summer oilseeds']
# SA2 yield index items (production regex, area regex or None = LUTO map area of the crop in the SA2)
SA2_PAT = {
    'Winter cereals': (r'(Wheat|Barley|Oats|Triticale) for grain - Production|All other cereals for grain or seed - Production',
                       r'(Wheat|Barley|Oats|Triticale) for grain - Area|All other cereals for grain or seed - Area'),
    'Summer cereals': (r'(Sorghum|Maize) for grain - Production', r'(Sorghum|Maize) for grain - Area'),
    'Rice': (r'Rice for grain - Production', r'Rice for grain - Area'),
    'Winter oilseeds': (r'Canola - Production', r'Canola - Area'),
    'Summer oilseeds': (r'Other oilseeds - Production', r'Other oilseeds - Area'),
    'Winter legumes': (r'(Chickpeas|Lentils|Lupins|Faba beans) - Production', r'(Chickpeas|Lentils|Lupins|Faba beans) - Area'),
    'Summer legumes': (r'Mung beans - Production', r'Mung beans - Area'),
    'Cotton': (r'Cotton \(irrigated and non-irrigated\) - Lint production', r'Cotton \(irrigated and non-irrigated\) - Area'),
    'Sugar': (r'Sugar cane - Cut for crushing - Production', r'Sugar cane - Cut for crushing - Area'),
    'Hay': (r'cereal and other crops cut for hay - (Total - )?Production|cut for hay - Total production', None),
    'Apples': (r'Apples - Production', None), 'Pears': (r'Pears \(including Nashi\) - Production', None),
    'Citrus': (r'Citrus fruit - .* - Production', None), 'Nuts': (r'Nuts - .* - Production', None),
    'Stone fruit': (r'Stone fruit - .* - Production|Olives - Production', None),
    'Tropical stone fruit': (r'(Mangoes|Avocados) - Production', None),
    'Plantation fruit': (r'(Bananas|Pineapples|Kiwifruit|Strawberries|Blueberries) - Production', None),
    'Grapes': (r'Grapes - Total - Production', r'Grapes - Total - Total area'),
    'Vegetables': (r'^Vegetables.*(Potatoes - Total production|Potatoes - Production|Carrots - Production|Onions - Production|'
                   r'Pumpkins - Production|Sweet corn - Production|Melons - Production|Cabbages - Production|Cauliflowers - '
                   r'Production|Tomatoes - Total production|Tomatoes - Production|Broccoli - Production|Lettuces - Total '
                   r'production|Lettuces - Production|Beans .* - Production)', None),
}
# Costs: FDP group and ratio group, FDP cost items by LUTO cost field, FDP state names
FDP_GROUP = ['Winter cereals', 'Summer cereals', 'Winter oilseeds', 'Summer oilseeds', 'Winter legumes', 'Summer legumes', 'Hay']
RATIO_GROUP = ['Apples', 'Pears', 'Citrus', 'Nuts', 'Stone fruit', 'Tropical stone fruit', 'Plantation fruit', 'Grapes',
               'Vegetables', 'Cotton', 'Rice', 'Sugar', 'Other non-cereal crops']
FDP_CAT = {'AC': ['Seed ($)', 'Fertiliser ($)', 'Crop and pasture chemicals ($)', 'Fuel oil and grease ($)',
                  'Repairs and maintenance ($)', 'Contracts - cropping ($)', 'Electricity ($)', 'Other materials ($)',
                  'Other services ($)'],
           'FLC': ['Wages for hired labor ($)', 'Imputed labor cost ($)', 'Payments to sharefarmers ($)'],
           'FOC': ['Insurance ($)', 'Shire and PPB rates ($)', 'Accounting services ($)', 'Advisory services ($)', 'Telephone ($)',
                   'Other administration expenses ($)', 'Leasing charges ($)'],
           'FDC': ['Depreciation ($)']}
FDP_ST = {'New South Wales': 'NSW', 'Victoria': 'VIC', 'Queensland': 'QLD', 'South Australia': 'SA', 'Western Australia': 'WA',
          'Tasmania': 'TAS', 'Northern Territory': 'NT', 'Australia': 'AUS'}
HAY_AREA = r'cereal and other crops cut for hay - (?:Total - )?Area|cut for hay - Total area|Total crops cut for hay - Area'
# Water groups: ABS 4618 crop group pattern and LUTO land uses
WATER_G = {
    'broadacre': (r'Cereals for grain or seed, excluding rice|Other cereals for grain or seed|Other broadacre crops|Other crops n\.e\.c\.',
                  ['Winter cereals', 'Summer cereals', 'Winter oilseeds', 'Summer oilseeds', 'Winter legumes', 'Summer legumes',
                   'Other non-cereal crops']),
    'hay': (r'cut for hay|cut for silage', ['Hay']),
    'fruit': (r'^Fruit trees', ['Apples', 'Pears', 'Citrus', 'Nuts', 'Stone fruit', 'Tropical stone fruit', 'Plantation fruit']),
    'grapes': (r'^Grapevines', ['Grapes']), 'vegetables': (r'^Vegetables', ['Vegetables']),
    'rice': (r'^Rice$', ['Rice']), 'cotton': (r'^Cotton$', ['Cotton']), 'sugar': (r'^Sugar cane$', ['Sugar'])}
WATER_STN = {'New South Wales': 'NSW', 'Victoria': 'VIC', 'Queensland': 'QLD', 'South Australia': 'SA', 'Western Australia': 'WA',
             'Tasmania': 'TAS', 'Northern Territory': 'NT', 'Australian Capital Territory': 'ACT'}
# Energy and lifecycle fields: sown area items by LUTO crop
F8 = ['CHEM_APPL', 'CROP_MGT', 'CULTIV', 'FERT_PROD', 'HARVEST', 'IRRIG', 'PEST_PROD', 'SOWING']
SOWN_ABARES = {'Winter cereals': ['Wheat', 'Barley', 'Oats', 'Triticale'], 'Summer cereals': ['Grain sorghum', 'Corn (maize)'],
               'Rice': ['Rice'], 'Winter oilseeds': ['Canola'], 'Summer oilseeds': ['Sunflower', 'Soybeans'],
               'Winter legumes': ['Chickpeas', 'Lentils', 'Lupins', 'Field peas', 'Faba beans'], 'Cotton': ['Cotton lint a']}
SOWN_ABS = {'Hay': HAY_AREA, 'Sugar': r'Sugar cane - Cut for crushing - Area', 'Vegetables': r'^Vegetables - Total area',
            'Grapes': r'Grapes - Total - Total area', 'Other non-cereal crops': r'All other crops n\.e\.c\. - Area'}
# Calibration labels
LABEL = {
    'banded_under': 'NLUM maps less area than ABS records (NLUM against ABS, cause unverified); tonnes not placed',
    'over_cap': 'NLUM maps more area than ABS records at the lower yield cap',
    'over_noabs': 'NLUM maps more area than ABS records; no ABS production published ({}); national yield factor',
    'water_banded': 'NLUM against ABS, cause unverified; water not placed or added',
}
NOABS_STATUS = {('Summer oilseeds', 'TAS'): 'zero', ('Winter oilseeds', 'NT'): 'zero',
                ('Summer oilseeds', 'NT'): 'zero or suppressed, unverified'}
# Inferred values: rules in code, listed in the interim table with their computed values
INFERRED = {
    'fieldpeas_faba_uv': 'Field peas and faba beans: one national real unit value ($/t) = median of the ABS national other-pulses '
                         'unit value FY2020 and FY2021 and the AA levied unit values FY2023-25 (ABS 7503 other pulses include '
                         'summer pulses in QLD and NSW)',
    'summer_legumes': 'Summer legumes: ABS state other pulses minus ABARES field peas + faba, FY2020-21, tonnes and value; negative '
                      'residuals 0 (low confidence)',
    'oncc_p1': 'Other non-cereal crops P1 = state grain-group farm-gate price (pooled window value / tonnes of the six grain '
               'groups), national pooled price where the state has none; tonnes = ABS 7503 value / P1',
    'dry_pears': 'Dry Pears (new column): agGHG energy, lifecycle and SOIL from dry Apples; agec costs from the irrigated '
                 'Pears cost-to-revenue ratio (the 2010 file has no dry Pears)',
}
# ============================================================================= end of reference table


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
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (luto-crop-economics/1.0)', 'Accept': '*/*'})
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


def trip(df, index=False, **read):
    """The earlier multi-script build wrote each intermediate table to CSV and read it back with pandas' default parser,
    which is not round-trip exact for every float. Repeat that write and read in memory so the outputs reproduce it."""
    return pd.read_csv(io.StringIO(df.to_csv(index=index)), **read)


# ----------------------------------------------------------------------------- step 1: CPI
def cpi_table(zp):
    """FY-average CPI and the deflator to FY2021 dollars, FY2006 on."""
    b = zipfile.ZipFile(zp).read('6401018.xlsx')
    d = pd.read_excel(io.BytesIO(b), sheet_name='Data1', header=None, skiprows=10)
    assert pd.read_excel(io.BytesIO(b), sheet_name='Data1', header=None).iloc[9, 1] == REF['cpi_series']
    s = pd.Series(d[1].astype(float).values, index=pd.to_datetime(d[0]))
    fy = s.groupby(s.index.year + (s.index.month >= 7)).agg(['mean', 'size'])
    fy = fy[fy['size'] == 4]['mean'].rename('cpi_fy_mean').to_frame()
    fy.index.name = 'FY_end'
    fy['to_FY2021'] = fy.loc[REF['base_fy'], 'cpi_fy_mean'] / fy['cpi_fy_mean']
    return trip(fy.loc[2006:], index=True, index_col=0)['to_FY2021']


# ----------------------------------------------------------------------------- step 2: ABS 7121 / 7503 / AA tables
STATE_BY_LABEL = {'Australia': 'AUS', 'New South Wales': 'NSW', 'Victoria': 'VIC', 'Queensland': 'QLD', 'South Australia': 'SA',
                  'Western Australia': 'WA', 'Tasmania': 'TAS', 'Northern Territory': 'NT', 'Australian Capital Territory': 'ACT'}
STATE_BY_DIGIT = {'1': 'NSW', '2': 'VIC', '3': 'QLD', '4': 'SA', '5': 'WA', '6': 'TAS', '7': 'NT', '8': 'ACT', '9': 'OT'}
KNOWN_UNITS = ['t/ha', 'kg/ha', 'kg/tree', 'kg/m2', 'ha', 't', 'kg', 'no.', '%', 'm2', 'dozens', '$']
UNIT_RE = re.compile(r'\(([^()]*)\)')
AA_UNIT_RE = re.compile(r'\(([^()]*)\)\s*$')
OUT_COLS = ['FY_end', 'state', 'item_code', 'item_desc', 'measure', 'unit', 'value', 'value_orig', 'unit_std', 'rse', 'flag',
            'series', 'release_date', 'source_file', 'source_sheet', 'note']


def fy_end(label):
    return int(re.match(r'\s*(\d{4})-(\d{2})', str(label)).group(1)) + 1


def parse_unit(desc):
    hits = [t for t in (t.strip() for t in UNIT_RE.findall(desc)) if t in KNOWN_UNITS]
    return hits[-1] if hits else ''


def classify_7121(desc, unit):
    if unit == 'ha':
        return 'area_ha', 1.0, 'ha'
    if unit == 'm2':
        return 'area_ha', 1e-4, 'ha'
    if unit == 't':
        return 'production_t', 1.0, 't'
    if unit == 'kg':
        return 'production_t', 1e-3, 't'
    if unit == 'no.':
        if re.search(r'\b[Tt]rees\b|\b[Vv]ines\b', desc):
            return 'trees_no', 1.0, 'no.'
        return 'production_other', 1.0, 'no.'
    if unit == '$':
        return 'production_other', 1.0, '$'
    return 'production_other', 1.0, unit


def read_csv_flex(raw, hdr_skip=4):
    try:
        txt = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        txt = raw.decode('cp1252')
    return pd.read_csv(io.StringIO(txt), skiprows=hdr_skip, dtype=str, keep_default_na=False)


def num(s):
    if s is None:
        return np.nan, 'blank'
    if isinstance(s, (int, float, np.integer, np.floating)):
        if isinstance(s, float) and np.isnan(s):
            return np.nan, 'blank'
        return float(s), ''
    t = str(s).strip().replace(',', '')
    if t == '':
        return np.nan, 'blank'
    try:
        return float(t), ''
    except ValueError:
        return np.nan, str(s).strip()


def release_of(src, kind):
    """'Released at ...' date of an ABS file (bytes for csv, path for Excel)."""
    if kind == 'csv':
        txt = src[:600].decode('cp1252', 'replace')
    else:
        x = pd.ExcelFile(src)
        parts = []
        for sh in x.sheet_names[:2]:
            d = x.parse(sh, header=None, nrows=8)
            parts += [str(v) for v in d.values.ravel() if isinstance(v, str)]
        txt = '\n'.join(parts)
    m = re.search(r'Released at[^\n]*?\)\s*(?:\w{3}\s+)?(\d{1,2} \w+ \d{4})', txt)
    return pd.to_datetime(m.group(1), format='%d %B %Y').strftime('%Y-%m-%d') if m else ''


def frame_from_rows(df):
    if df.shape[1] == 8:
        df.columns = ['rc', 'label', 'code', 'desc', 'v1', 'r1', 'v2', 'r2']
    else:
        df.columns = ['rc', 'label', 'desc', 'v1', 'r1', 'v2', 'r2'][:df.shape[1]]
        df.insert(2, 'code', '')
    return df


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
        body = frame_from_rows(body.iloc[:, :raw.shape[1]].copy())
        body['sheet'] = sh
        out.append(body)
    return pd.concat(out, ignore_index=True)


def clean_rc(s):
    s = str(s).strip()
    return s[:-2] if re.fullmatch(r'\d+\.0', s) else s


def _select(df, keep):
    df = df.copy()
    df['rc'] = df['rc'].map(clean_rc)
    df['desc'] = df['desc'].astype(str).str.strip()
    if keep == 'state':
        m = df['rc'].isin([str(i) for i in range(9)]) & df['label'].isin(STATE_BY_LABEL.keys())
    else:
        m = df['rc'].str.fullmatch(r'\d{9}')
    df = df[m & (df['desc'] != '')].copy()
    if keep == 'state':
        df['state'] = df['label'].map(STATE_BY_LABEL)
    else:
        df['sa2_code'] = df['rc']
        df['sa2_name'] = df['label']
        df['state'] = df['rc'].str[0].map(STATE_BY_DIGIT)
    return df


def normalise_7121(df, fy, series, source_file, rel, keep='state', sa2_asgs=None):
    df = _select(df, keep)
    uniq = df['desc'].drop_duplicates()
    unit_map = {d: parse_unit(d) for d in uniq}
    cls_map = {d: classify_7121(d, unit_map[d]) for d in uniq}
    df['unit'] = df['desc'].map(unit_map)
    df['measure'] = df['desc'].map(lambda d: cls_map[d][0])
    df['factor'] = df['desc'].map(lambda d: cls_map[d][1])
    df['unit_std'] = df['desc'].map(lambda d: cls_map[d][2])
    est_v = df['v1'].astype(str).str.strip().str.replace(',', '', regex=False)
    est_num = pd.to_numeric(est_v, errors='coerce')
    e = pd.DataFrame({'FY_end': fy, 'state': df['state'], 'item_code': df['code'].astype(str).str.strip(),
                      'item_desc': df['desc'], 'measure': df['measure'], 'unit': df['unit'], 'value_orig': est_num,
                      'value': est_num * df['factor'], 'unit_std': df['unit_std'], 'rse': df['r1'].astype(str).str.strip(),
                      'flag': np.where(est_num.isna(), est_v.where(est_v != '', 'blank'), ''), 'series': series,
                      'release_date': rel, 'source_file': source_file, 'source_sheet': df['sheet'], 'note': ''})
    if keep != 'state':
        e['sa2_code'] = df['sa2_code']
        e['sa2_name'] = df['sa2_name']
        e['asgs_edition'] = sa2_asgs
    nb_v = df['v2'].astype(str).str.strip().str.replace(',', '', regex=False)
    nb_num = pd.to_numeric(nb_v, errors='coerce')
    keepb = ~nb_v.isin(['', '..'])
    b = pd.DataFrame({'FY_end': fy, 'state': df['state'], 'item_code': df['code'].astype(str).str.strip(),
                      'item_desc': df['desc'], 'measure': 'number_businesses', 'unit': 'no.', 'value_orig': nb_num,
                      'value': nb_num, 'unit_std': 'no.', 'rse': df['r2'].astype(str).str.strip(),
                      'flag': np.where(nb_num.isna(), nb_v, ''), 'series': series, 'release_date': rel,
                      'source_file': source_file, 'source_sheet': df['sheet'], 'note': ''})[keepb]
    if keep != 'state':
        b['sa2_code'] = df['sa2_code'][keepb]
        b['sa2_name'] = df['sa2_name'][keepb]
        b['asgs_edition'] = sa2_asgs
    return pd.concat([e, b], ignore_index=True)


def normalise_7503(df, fy, series, source_file, rel, keep='state', sa2_asgs=None):
    df = _select(df, keep)
    parts = []
    for vcol, rcol, meas in [('v1', 'r1', 'gross_value_aud'), ('v2', 'r2', 'local_value_aud')]:
        v = df[vcol].astype(str).str.strip().str.replace(',', '', regex=False)
        n = pd.to_numeric(v, errors='coerce')
        p = pd.DataFrame({'FY_end': fy, 'state': df['state'], 'item_code': df['code'].astype(str).str.strip(),
                          'item_desc': df['desc'], 'measure': meas, 'unit': '$', 'value_orig': n, 'value': n, 'unit_std': 'AUD',
                          'rse': df[rcol].astype(str).str.strip(), 'flag': np.where(n.isna(), v.where(v != '', 'blank'), ''),
                          'series': series, 'release_date': rel, 'source_file': source_file, 'source_sheet': df['sheet'],
                          'note': ''})
        if keep != 'state':
            p['sa2_code'] = df['sa2_code']
            p['sa2_name'] = df['sa2_name']
            p['asgs_edition'] = sa2_asgs
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def aa_measure(desc):
    d = desc.strip()
    m = AA_UNIT_RE.search(d)
    unit = m.group(1).strip() if m else ''
    low = d.lower()
    if unit == 'ha':
        return 'area_ha', 1.0, 'ha', unit
    if unit == 't':
        return 'production_t', 1.0, 't', unit
    if unit == '$m':
        return ('farm_gate_value_aud', 1e6, 'AUD', unit) if 'farm gate value' in low else ('production_other', 1.0, unit, unit)
    if unit == '$':
        if 'local value' in low:
            return 'local_value_aud', 1.0, 'AUD', unit
        if 'gross value' in low:
            return 'gross_value_aud', 1.0, 'AUD', unit
        return 'production_other', 1.0, unit, unit
    if unit == 'no.':
        if 'levy payers' in low or 'businesses' in low:
            return 'number_businesses', 1.0, 'no.', unit
        return 'production_other', 1.0, 'no.', unit
    return 'production_other', 1.0, unit, unit


def read_aa_workbook(src, fname, series):
    x = pd.ExcelFile(src)
    rel = release_of(src, 'xlsx')
    out = []
    for sh in x.sheet_names:
        if not sh.startswith('Table'):
            continue
        raw = x.parse(sh, header=None)
        hdr = next(i for i in range(min(12, len(raw))) if 'Data item' in [str(v).strip() for v in raw.iloc[i].tolist()])
        hv = [str(v).strip() for v in raw.iloc[hdr].tolist()]
        ci_item, ci_reg = hv.index('Data item'), hv.index('Region')
        ycols = [(j, h) for j, h in enumerate(hv) if re.match(r'^\d{4}-\d{2}', h)]
        body = raw.iloc[hdr + 1:]
        body = body[body.iloc[:, ci_item].notna()]
        for _, r in body.iterrows():
            reg = str(r.iloc[ci_reg]).strip()
            if reg not in STATE_BY_LABEL:
                continue
            desc = str(r.iloc[ci_item]).strip()
            meas, fac, ustd, uorig = aa_measure(desc)
            for j, h in ycols:
                n, fl = num(r.iloc[j])
                mm = re.match(r'^\d{4}-\d{2}(.*)$', h)
                note = 'column header ' + h if mm and mm.group(1).strip() else ''
                out.append((fy_end(h), STATE_BY_LABEL[reg], '', desc, meas, uorig, n * fac if n == n else np.nan, n, ustd, '',
                            fl if n != n else '', series, rel, fname, sh, note))
    return pd.DataFrame(out, columns=OUT_COLS)


def abs_tables(P):
    """ABS state table (7121, 7503 FY2016-22; AA FY2023-25 with the latest-release flag) and SA2 table (FY2016, FY2021)
    both after a CSV round trip."""
    parts, cubes = [], {}
    for f, fy, ser, kind in ABS_STATE_FILES:
        if kind == 'csv':
            raw = open(P[f], 'rb').read()
            r = release_of(raw, 'csv')
            d = read_csv_flex(raw)
            d.columns = ['rc', 'label', 'code', 'desc', 'v1', 'r1', 'v2', 'r2']
            d['sheet'] = ''
        else:
            r = release_of(P[f], 'xlsx')
            d = read_excel_cube(P[f], sheets=SHEETS_7503_201516 if f == '75030do001_201516_state.xls' else None)
        cubes[f] = (d, r)
        parts.append((normalise_7121 if ser == '7121' else normalise_7503)(d, fy, ser, f, r))
    for f, ser in AA_FILES:
        parts.append(read_aa_workbook(P[f], f, ser))
    st = pd.concat(parts, ignore_index=True)
    st['latest_release'] = 1
    aa = st['series'].str.startswith('AA_')
    key = ['FY_end', 'state', 'item_desc', 'measure', 'series']
    mx = st[aa].groupby(key)['release_date'].transform('max')
    st.loc[aa, 'latest_release'] = (st.loc[aa, 'release_date'] == mx).astype(int)
    st = st[(st['FY_end'] >= 2016) & (st['FY_end'] <= 2025)]
    st = st[OUT_COLS + ['latest_release']].sort_values(['FY_end', 'series', 'source_file', 'state', 'item_desc', 'measure'])
    st = st.reset_index(drop=True)
    sa = []
    for f, fy, fn in [('7121do004_201516_SA2.zip', 2016, normalise_7121), ('75030do005_201516.zip', 2016, normalise_7503)]:
        z = zipfile.ZipFile(P[f])
        b = z.read(z.namelist()[0])
        d = read_csv_flex(b)
        d.columns = ['rc', 'label', 'code', 'desc', 'v1', 'r1', 'v2', 'r2']
        d['sheet'] = ''
        sa.append(fn(d, fy, fn.__name__[-4:], f, release_of(b, 'csv'), keep='sa2', sa2_asgs=2011))
    for f, fn in [('AGCDCASGS202021.xlsx', normalise_7121), ('VACPDCASGS202021.xlsx', normalise_7503)]:
        sa.append(fn(cubes[f][0], 2021, fn.__name__[-4:], f, cubes[f][1], keep='sa2', sa2_asgs=2021))
    s2 = pd.concat(sa, ignore_index=True)
    s2 = s2[['FY_end', 'state', 'sa2_code', 'sa2_name', 'asgs_edition'] + [c for c in OUT_COLS if c not in ('FY_end', 'state')]]
    return trip(st, low_memory=False), trip(s2, low_memory=False)


# ----------------------------------------------------------------------------- step 3: ABARES crop report, FDP, ABS 4618
def acr_table(path):
    """ABARES Australian Crop Report state area and production FY2017-25."""
    import openpyxl
    sm = {'Australia': 'AUS', 'New South Wales': 'NSW', 'Queensland': 'QLD', 'South Australia': 'SA', 'Tasmania': 'TAS',
          'Victoria': 'VIC', 'Western Australia': 'WA'}
    stat = {'': 'actual', 's': 'estimate', 'f': 'forecast'}
    wb = openpyxl.load_workbook(path, data_only=True)
    rows = []
    for ws in wb.worksheets[1:]:
        R = list(ws.iter_rows(values_only=True))
        h = [r for r in R if r[1] == 'Crops'][0]
        cols = {}
        for j, c in enumerate(h):
            if isinstance(c, str):
                m = re.match(r'^(\d{4}).(\d{2})\s*([sf])?$', c.strip())
                if m and 2017 <= int(m.group(1)) + 1 <= 2025:
                    cols[j] = (int(m.group(1)) + 1, stat[m.group(3) or ''], c)
        crop = None
        for r in R:
            if (r[1] and r[2] is None and r[1] not in ('Crops', 'Winter crops', 'Summer crops') and not r[1].startswith('a Cotton')
                    and r[1] != ws.title and not r[1].startswith('Australia grains') and 'grains, oilseeds' not in r[1]):
                crop = r[1]
            if r[1] in ('Area', 'Production') and crop:
                for j, (fy, s, lab) in cols.items():
                    v = r[j]
                    if v is None:
                        continue
                    meas = 'area_ha' if r[1] == 'Area' else 'production_t'
                    rows.append((fy, sm[ws.title], crop, meas, v * 1000, s, lab, ws.title, r[2]))
    d = pd.DataFrame(rows, columns=['FY_end', 'state', 'crop', 'measure', 'value', 'status', 'orig_label', 'sheet', 'orig_unit'])
    a = d[d.measure == 'area_ha'].set_index(['FY_end', 'state', 'crop', 'status', 'sheet'])
    p = d[d.measure == 'production_t'].set_index(['FY_end', 'state', 'crop', 'status', 'sheet'])
    y = (p.value / a.value.where(a.value > 0)).dropna().reset_index()
    y['measure'] = 'yield_t_ha'
    y['orig_label'] = 'derived: production_t/area_ha'
    y['orig_unit'] = 'derived'
    d = pd.concat([d, y], ignore_index=True)
    d['source_file'] = 'AustCropRrt20260901_v1.0.0_asset3.xlsx'
    d = d[['FY_end', 'state', 'crop', 'measure', 'value', 'status', 'source_file', 'sheet', 'orig_label', 'orig_unit']]
    return trip(d.sort_values(['state', 'crop', 'measure', 'FY_end']))


def fdp_wide(state_csv, nat_csv):
    """ABARES FDP state and national files, one row per (Year, State, Industry)."""
    s = pd.read_csv(state_csv)
    n = pd.read_csv(nat_csv)
    n['State'] = 'Australia'
    long = pd.concat([s, n[s.columns]], ignore_index=True)
    key = ['Year', 'State', 'Industry', 'Variable']
    long['_occ'] = long.groupby(key).cumcount()
    isdup = long.duplicated(key, keep=False)
    long['col'] = np.where(isdup, long['Variable'] + '__dup' + (long['_occ'] + 1).astype(str), long['Variable'])
    vals = long.pivot(index=['Year', 'State', 'Industry'], columns='col', values='Value')
    rses = long.pivot(index=['Year', 'State', 'Industry'], columns='col', values='RSE')
    rses.columns = [c + '_rse' for c in rses.columns]
    cols_v = sorted(vals.columns)
    wide = pd.concat([vals[cols_v], rses[[c + '_rse' for c in cols_v]]], axis=1).reset_index()
    wide = wide.sort_values(['State', 'Industry', 'Year']).reset_index(drop=True)
    return trip(wide, low_memory=False)


W_MEAS = {'Total area grown (ha)': 'area_grown_ha', 'Total area (ha)': 'area_grown_ha', 'Area watered (ha)': 'area_watered_ha',
          'Volume applied (ML)': 'volume_applied_ml', 'Application rate (ML/ha)': 'application_rate_ml_ha'}
W_RE_MAIN = re.compile(r'^(?:Water use - )?(?P<grp>.+?) - (?P<m>Total area grown \(ha\)|Total area \(ha\)|Area watered \(ha\)|'
                       r'Volume applied \(ML\)|Application rate \(ML/ha\))(?: \([a-z]\))?$')
W_RE_TOT = re.compile(r'^(?:Water use - )?Total (?P<m>area watered \(ha\)|volume applied \(ML\)|application rate \(ML/ha\))'
                      r'(?: \([a-z]\))?$')
W_TOTM = {'area watered (ha)': 'area_watered_ha', 'volume applied (ML)': 'volume_applied_ml',
          'application rate (ML/ha)': 'application_rate_ml_ha'}


def water_read(path, fname):
    """One ABS 4618 file as (code, label, ccode, item, est, est_rse, bus, bus_rse) strings."""
    def norm(df):
        cols = {}
        for c in df.columns:
            k = str(c).strip()
            if k == 'Region code':
                cols[c] = 'code'
            elif k == 'Region label':
                cols[c] = 'label'
            elif k == 'Commodity code':
                cols[c] = 'ccode'
            elif k.startswith('Commodity description'):
                cols[c] = 'item'
            elif k == 'Estimate':
                cols[c] = 'est'
            elif k.startswith('Estimate - Rel'):
                cols[c] = 'est_rse'
            elif k == 'Number of agricultural businesses':
                cols[c] = 'bus'
            elif k.startswith('Number of agricultural businesses - Rel'):
                cols[c] = 'bus_rse'
        df = df.rename(columns=cols)
        for c in ['ccode', 'est_rse', 'bus', 'bus_rse']:
            if c not in df.columns:
                df[c] = None
        return df[['code', 'label', 'ccode', 'item', 'est', 'est_rse', 'bus', 'bus_rse']]

    def clean(df):
        df = df[df['item'].notna() & df['label'].notna()].copy()

        def cc(v):
            s = str(v).strip()
            try:
                return str(int(float(s)))
            except Exception:
                return s
        df['code'] = df['code'].map(cc)
        return df

    def hdr(raw):
        for i in range(min(30, len(raw))):
            if str(raw.iloc[i, 0]).strip() == 'Region code':
                return i
        raise ValueError('no header')
    if fname.lower().endswith('.csv'):
        raw = pd.read_csv(path, header=None, encoding='latin1', dtype=str)
        hi = hdr(raw)
        d = raw.iloc[hi + 1:].copy()
        d.columns = [str(x) for x in raw.iloc[hi]]
        return clean(norm(d))
    x = pd.ExcelFile(path)
    outs = []
    for s in x.sheet_names:
        if s in ('Contents', 'Guide to estimates'):
            continue
        raw = pd.read_excel(x, sheet_name=s, header=None, dtype=object)
        hi = hdr(raw)
        d = raw.iloc[hi + 1:].copy()
        d.columns = [str(x_) for x_ in raw.iloc[hi]]
        d = clean(norm(d))
        d['sheet'] = s
        outs.append(d)
    return pd.concat(outs, ignore_index=True)


def water_long(P):
    """ABS 4618 FY2017-21 long table, first source wins on duplicate keys."""
    def parse_item(item):
        s = item.strip()
        m = W_RE_TOT.match(s)
        if m:
            return 'Total', W_TOTM[m.group('m')]
        m = W_RE_MAIN.match(s)
        if m and not s.startswith(('Water source', 'Irrigation expenditure', 'Area of holding')):
            return m.group('grp'), W_MEAS[m.group('m')]
        return '', 'other'

    def val(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan, ''
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v), ''
        s = str(v).strip()
        if s == '' or s.lower() == 'nan':
            return np.nan, ''
        try:
            return float(s.replace(',', '')), ''
        except ValueError:
            return np.nan, s

    def rse_val(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ''
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v).strip()
        return '' if s.lower() == 'nan' else s

    def rtype(code, kind):
        if code == '0':
            return 'AUS'
        if len(code) == 1:
            return 'state'
        if code in ('950', '955'):
            return 'MDB'
        if kind in ('SA4', 'NRM'):
            return kind
        if kind == 'ASGS3':
            return 'SA2' if len(code) == 9 else ('SA4' if len(code) == 3 else 'UNKNOWN')
        return 'UNKNOWN'
    rows = []
    for fy, lst in WATER_FILES.items():
        for fname, kind in lst:
            d = water_read(P[fname], fname).reset_index(drop=True)
            ests = d['est'].map(val)
            buss = d['bus'].map(val)
            for i in range(len(d)):
                r = d.iloc[i]
                code, label, item = r['code'], str(r['label']).strip(), str(r['item']).strip()
                rt = rtype(code, kind)
                ed = ({2021: 'ASGS Edition 3 (2021)'}.get(fy, 'ASGS 2016 [unverified: edition not stated in file]')
                      if rt in ('SA4', 'SA2') else '')
                grp, meas = parse_item(item)
                e_v, e_f = ests.iloc[i]
                rows.append((fy, rt, code, label, ed, item, grp, meas, e_v, rse_val(r['est_rse']), e_f, fname))
                b_v, b_f = buss.iloc[i]
                if not (np.isnan(b_v) and b_f == '') and b_f != '..':
                    rows.append((fy, rt, code, label, ed, item, grp, 'businesses', b_v, rse_val(r['bus_rse']), b_f, fname))
    df = pd.DataFrame(rows, columns=['FY_end', 'region_type', 'region_code', 'region_name', 'edition', 'item_desc', 'crop_group',
                                     'measure', 'value', 'rse', 'flag', 'source_file'])
    df = df.drop_duplicates(['FY_end', 'region_type', 'region_code', 'item_desc', 'measure'], keep='first')
    df = df.sort_values(['FY_end', 'region_type', 'region_code', 'item_desc', 'measure'], kind='stable')
    return trip(df, low_memory=False)


# ----------------------------------------------------------------------------- step 4: SA2 2011 key
def sa2_2011_cells(shp_zip, mask_path):
    """SA2 2011 (SA2_MAIN11) per LUTO cell: polygon at the cell centre, else the nearest labelled pixel (as nlum_map.py does
    for 2021)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt
    with rasterio.open(mask_path) as r:
        mk = r.read(1) == 1
        T = r.transform
    with tempfile.TemporaryDirectory() as td:
        zipfile.ZipFile(shp_zip).extractall(td)      # the cached zip has no .zip extension (old ABS URL)
        g = gpd.read_file(os.path.join(td, 'SA2_2011_AUST.shp'))
    assert str(g.crs) == 'EPSG:4283', g.crs
    g = g[g['SA2_MAIN11'].str.fullmatch(r'[0-9]{9}')].copy()
    g['code'] = g['SA2_MAIN11'].astype(np.int64)
    gg = g[g.geometry.notna() & ~g.geometry.is_empty]
    ras = rasterize(((geom, int(c)) for geom, c in zip(gg.geometry, gg.code)), out_shape=mk.shape,
                    transform=T, fill=-1, all_touched=False, dtype='int32')
    cells = ras[mk].astype(np.int32)
    idx = distance_transform_edt(ras == -1, return_distances=False, return_indices=True)
    near = ras[idx[0], idx[1]]
    fill = cells.copy()
    miss = cells == -1
    fill[miss] = near[mk][miss]
    return fill


# ----------------------------------------------------------------------------- step 5: 2010 economics on the final map
def current_econ(M, cur):
    """The 2010 agec_crops values on allocated cells by (state, lm, lu), area-weighted, with the cost-to-revenue ratio."""
    lu, lm, area, sid, lus, crops = M['lu'], M['lm'], M['area'], M['state_id'], M['lus'], M['crops']
    out = []
    for m, mn in LMN.items():
        for c in crops:
            if ('Yield', mn, c) not in cur.columns:
                continue
            j = lus.index(c)
            alloc = (lu == j) & (lm == m)
            f = {k: cur[(k, mn, c)].values[alloc].astype(float) for k in ['Yield', 'P1', 'AC', 'QC', 'FLC', 'FOC', 'FDC', 'WR', 'WP']}
            a = area[alloc]
            st = sid[alloc]
            rev = f['Yield'] * f['P1']
            cost = f['QC'] * f['Yield'] + f['AC'] + f['FLC'] + f['FOC'] + f['FDC'] + (f['WR'] * f['WP'] if mn == 'irr' else 0)
            for s in np.unique(st):
                k = (st == s) & ~np.isnan(rev) & ~np.isnan(cost)
                if k.sum() == 0:
                    out.append(dict(state=ST[s], lm=mn, lu=c, ha=a[st == s].sum(), ha_with_values=0.0))
                    continue
                w = a[k]
                r = dict(state=ST[s], lm=mn, lu=c, ha=a[st == s].sum(), ha_with_values=w.sum())
                for kk, v in f.items():
                    r[kk] = np.average(v[k], weights=w)
                r['rev_ha'] = np.average(rev[k], weights=w)
                r['cost_ha'] = np.average(cost[k], weights=w)
                r['cost_rev_ratio'] = r['cost_ha'] / r['rev_ha'] if r['rev_ha'] else np.nan
                out.append(r)
    return trip(pd.DataFrame(out))


# ----------------------------------------------------------------------------- step 6: production and price windows
def broadacre_window(abs_all, abr_all, cpi):
    """Broadacre production (t) and farm-gate value (FY2021 $) by LUTO crop and state, window means."""
    abs_ = abs_all[(abs_all.latest_release == 1) & (abs_all.state != 'AUS')]
    abr = abr_all[(abr_all.state != 'AUS') & (abr_all.measure == 'production_t')]
    FY = list(REF['window'])

    def abs_series(pat, measure, years=FY, series_prefix=None):
        x = abs_[abs_.item_desc.str.contains(pat, regex=True, case=False) & (abs_.measure == measure) & abs_.FY_end.isin(years)]
        if series_prefix is not None:
            x = x[x.series.astype(str).str.startswith(series_prefix)]
        # several matching descriptions per (state, FY) are the same quantity renamed: take the max, not the sum
        g = x.groupby(['state', 'FY_end', 'item_desc']).value.sum().reset_index()
        dup = g.groupby(['state', 'FY_end']).size()
        assert (dup <= 2).all(), (pat, dup[dup > 2])
        return g.groupby(['state', 'FY_end']).value.max()

    rows = []
    for c, (abr_name, ppat, vpat, aa) in COM.items():
        for st in STATES:
            for fy in FY:
                q, qsrc = np.nan, None
                if abr_name is not None and st in ('NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS'):
                    v = abr[(abr.crop == abr_name) & (abr.state == st) & (abr.FY_end == fy)].value
                    if len(v):
                        q, qsrc = float(v.sum()), 'ABARES ACR Sep2026'
                rows.append(dict(com=c, state=st, FY=fy, q=q, qsrc=qsrc))
    P = pd.DataFrame(rows)
    for c, (abr_name, ppat, vpat, aa) in COM.items():
        if abr_name is not None and c not in ('sunflower', 'soybeans', 'fieldpeas', 'faba'):
            for (st, fy), v in abs_series(ppat, 'production_t', range(2017, 2023)).items():
                m = (P.com == c) & (P.state == st) & (P.FY == fy) & P.q.isna()
                P.loc[m, ['q', 'qsrc']] = [v, 'ABS 7121']
        if abr_name is None:
            for (st, fy), v in abs_series(ppat, 'production_t', range(2017, 2023)).items():
                P.loc[(P.com == c) & (P.state == st) & (P.FY == fy), ['q', 'qsrc']] = [v, 'ABS 7121']
            if aa:
                for (st, fy), v in abs_series(re.escape(aa) + ' - Levied production', 'production_t', range(2023, 2026), 'AA').items():
                    P.loc[(P.com == c) & (P.state == st) & (P.FY == fy), ['q', 'qsrc']] = [v, 'ABS AA levied']
    U = []
    for c, (abr_name, ppat, vpat, aa) in COM.items():
        val = abs_series(vpat, 'local_value_aud', range(2017, 2023), '7503')
        qq = None if c in ('fieldpeas', 'faba') else abs_series(ppat, 'production_t', range(2017, 2023))
        for (st, fy), v in val.items():
            q = qq.get((st, fy), np.nan) if qq is not None else np.nan
            if c in ('fieldpeas', 'faba'):
                q = P[(P.com.isin(['fieldpeas', 'faba'])) & (P.state == st) & (P.FY == fy)].q.sum()
            if q and q > 0 and v > 0:
                U.append(dict(com=c, state=st, FY=fy, uv=v / q, usrc='ABS 7503/7121'))
        if aa:
            lv = abs_series(re.escape(aa) + r' - Local value of levied crop', 'local_value_aud', range(2023, 2026), 'AA')
            lq = abs_series(re.escape(aa) + r' - Levied production', 'production_t', range(2023, 2026), 'AA')
            for (st, fy), v in lv.items():
                q = lq.get((st, fy), np.nan)
                if q and q > 0 and v > 0:
                    U.append(dict(com=c, state=st, FY=fy, uv=v / q, usrc='ABS AA levied'))
    U = pd.DataFrame(U)
    U['uv_real'] = U.uv * U.FY.map(cpi)
    P = P.merge(U[['com', 'state', 'FY', 'uv_real', 'usrc']], on=['com', 'state', 'FY'], how='left')
    st_mean = U.groupby(['com', 'state']).uv_real.mean()
    nat_mean = U.groupby('com').uv_real.median()
    fill1 = P.uv_real.isna() & P.set_index(['com', 'state']).index.isin(st_mean.index)
    P.loc[fill1, 'uv_real'] = [st_mean[(c, s)] for c, s in P.loc[fill1, ['com', 'state']].itertuples(index=False)]
    P.loc[fill1, 'usrc'] = 'state window mean (filled year)'
    fill2 = P.uv_real.isna() & P.com.isin(nat_mean.index)
    P.loc[fill2, 'uv_real'] = P.loc[fill2, 'com'].map(nat_mean)
    P.loc[fill2, 'usrc'] = 'national median (filled)'
    # inferred (INFERRED['fieldpeas_faba_uv']): one national real unit value for field peas and faba beans
    absA = abs_all[(abs_all.latest_release == 1) & (abs_all.state == 'AUS')]

    def aus(pat, meas, fy):
        x = absA[absA.item_desc.str.contains(pat, regex=True) & (absA.measure == meas) & (absA.FY_end == fy)].value
        return float(x.max()) if len(x) else np.nan
    cands = [aus(r'Other pulses(?: \(b\))?$', 'local_value_aud', fy) / aus(r'Other pulses - Production', 'production_t', fy) * cpi[fy]
             for fy in (2020, 2021)]
    cands += list(U[(U.com.isin(['fieldpeas', 'faba'])) & (U.usrc == 'ABS AA levied')].uv_real)
    uv_fp = float(np.nanmedian(cands))
    m = P.com.isin(['fieldpeas', 'faba'])
    P.loc[m, 'uv_real'] = uv_fp
    P.loc[m, 'usrc'] = 'national other-pulses unit value (inferred)'
    P['value_real'] = P.q * P.uv_real
    # inferred (INFERRED['summer_legumes']): summer legumes residual, FY2020 and FY2021
    SL = []
    for fy in (2020, 2021):
        q_op = abs_series(r'Other pulses - Production', 'production_t', [fy])
        v_op = abs_series(r'Other pulses(?: \(b\))?$', 'local_value_aud', [fy], '7503')
        for st in STATES:
            qo, vo = q_op.get((st, fy), np.nan), v_op.get((st, fy), np.nan)
            qfp = P[(P.com.isin(['fieldpeas', 'faba'])) & (P.state == st) & (P.FY == fy)].q.sum()
            rq = max(qo - qfp, 0.0) if not np.isnan(qo) else np.nan
            rv = max(vo * cpi[fy] - qfp * uv_fp, 0.0) if not np.isnan(vo) else np.nan
            SL.append(dict(lu='Summer legumes', state=st, FY=fy, q=rq, v=rv))
    SL = pd.DataFrame(SL)
    P['lu'] = P.com.map({c: lu for lu, cs in LU_COM.items() for c in cs})
    G = P.groupby(['lu', 'state', 'FY']).agg(q=('q', lambda x: x.sum(min_count=1)),
                                            v=('value_real', lambda x: x.sum(min_count=1))).reset_index()
    G = pd.concat([G, SL[['lu', 'state', 'FY', 'q', 'v']]])
    Wm = G.groupby(['lu', 'state']).agg(q_mean=('q', 'mean'), v_mean=('v', 'mean'),
                                        years=('q', lambda x: int(x.notna().sum()))).reset_index()
    Wm['p_farmgate_fy2021'] = Wm.v_mean / Wm.q_mean
    return trip(Wm), dict(uv_fp=uv_fp, uv_fp_candidates=cands, summer_legumes=SL)


def hort_window(abs_all, cpi):
    """Horticulture production and farm-gate value, AA FY2021-25 with ABS core items FY2017-20 spliced."""
    d = abs_all[(abs_all.latest_release == 1) & (abs_all.state != 'AUS')]
    aa = d[d.series == 'AA_hort'].copy()
    aa['n'] = aa.item_desc.str.split(' - ').str[0]
    # Each farm-gate value counted once: the 2022-23 release spells 'Nectaries and peaches' (the same item, superseded for
    # FY2023 by the latest release), and wine grapes value is printed in Table 1 (group summary) and Table 6 (wine grapes),
    # equal up to rounding. One row per (state, FY, item, measure): latest release first, then the commodity table.
    aa.loc[aa.n == 'Nectaries and peaches', 'n'] = 'Nectarines and peaches'
    aa['_t1'] = (aa.source_sheet == 'Table 1').astype(int)
    aa = aa.sort_values(['release_date', '_t1'], ascending=[False, True], kind='stable').drop_duplicates(
        ['state', 'FY_end', 'n', 'measure'])
    rows = []
    for lu, names in AA_HORT.items():
        for meas in ['production_t', 'farm_gate_value_aud']:
            g = aa[aa.n.isin(names) & (aa.measure == meas)].groupby(['state', 'FY_end']).value.sum(min_count=1)
            for (st, fy), v in g.items():
                rows.append(dict(lu=lu, state=st, FY=fy, basis='AA', measure=meas, value=v))

    def core_series(lu, meas):
        out = {}
        for nm in CORE[lu]:
            if meas == 'production_t':
                pat = rf' - (?:{nm}) - (?:Total )?(?:[Pp]roduction|Total production)'
                x = d[d.series.astype(str).str.startswith('7121') & (d.measure == 'production_t') & d.item_desc.str.contains(pat, regex=True)
                      & ~d.item_desc.str.contains('Fresh market|fresh market|Processing|processing|Undercover|Outdoor|Crush', regex=True)]
            else:
                pat = rf' - (?:{CORE_VAL_NAME.get(nm, nm)})(?: - Outdoor and undercover)?$'
                x = d[d.series.astype(str).str.startswith('7503') & (d.measure == 'local_value_aud') & d.item_desc.str.contains(pat, regex=True)]
            g = x.groupby(['state', 'FY_end', 'item_desc']).value.sum().reset_index().groupby(['state', 'FY_end']).value.max()
            for k, v in g.items():
                out[k] = out.get(k, 0.0) + v
        return out
    for lu in CORE:
        for meas, m2 in [('production_t', 'production_t'), ('local_value_aud', 'farm_gate_value_aud')]:
            for (st, fy), v in core_series(lu, meas).items():
                if 2017 <= fy <= 2022:
                    rows.append(dict(lu=lu, state=st, FY=fy, basis='7121core', measure=m2, value=v))
    T = pd.DataFrame(rows)
    piv = T.pivot_table(index=['lu', 'state', 'measure', 'FY'], columns='basis', values='value', aggfunc='sum')
    ov = piv[piv.index.get_level_values('FY').isin([2021, 2022])].dropna()
    ov = ov[(ov['7121core'] > 0) & (ov['AA'] > 0)]
    ratio = ov.groupby(['lu', 'state', 'measure']).apply(lambda g: g['AA'].sum() / g['7121core'].sum())
    natr = ov.groupby(['lu', 'measure']).apply(lambda g: g['AA'].sum() / g['7121core'].sum())
    out = []
    for (lu, st, meas, fy), r in piv.iterrows():
        if fy >= 2021 and not np.isnan(r.get('AA', np.nan)):
            v, src = r['AA'], 'AA'
        elif fy <= 2020 and not np.isnan(r.get('7121core', np.nan)):
            k, ksrc = ratio.get((lu, st, meas), np.nan), 'state'
            if np.isnan(k):
                k, ksrc = natr.get((lu, meas), np.nan), 'national'
            v, src = r['7121core'] * k, f'7121core x splice ({ksrc})'
        else:
            continue
        out.append(dict(lu=lu, state=st, measure=meas, FY=fy, value=v, src=src))
    O = pd.DataFrame(out)
    O.loc[O.measure == 'farm_gate_value_aud', 'value'] *= O.loc[O.measure == 'farm_gate_value_aud', 'FY'].map(cpi)
    q =O[O.measure == 'production_t'].groupby(['lu', 'state']).value.agg(['mean', 'count'])
    v = O[O.measure == 'farm_gate_value_aud'].groupby(['lu', 'state']).value.mean()
    Wm = pd.DataFrame({'q_mean': q['mean'], 'v_mean': v, 'years': q['count']}).reset_index()
    Wm['p_farmgate_fy2021'] = Wm.v_mean / Wm.q_mean
    return trip(Wm)


def oncc_window(abs_all, cpi, bwin):
    """Other non-cereal crops: ABS 7503 'All other crops n.e.c.' value at the state grain-group farm-gate price."""
    d = abs_all[(abs_all.latest_release == 1) & (abs_all.state != 'AUS')]
    x = d[d.item_desc.str.contains(r'All other crops n\.e\.c\.$', regex=True) & (d.measure == 'local_value_aud')
          & d.series.astype(str).str.startswith('7503')]
    v = x.groupby(['state', 'FY_end']).value.max().reset_index()
    v['v_real'] = v.value * v.FY_end.map(cpi)
    g = bwin[bwin.lu.isin(GRAIN_GROUPS)].dropna(subset=['q_mean'])
    g = g.groupby('state')[['q_mean', 'v_mean']].sum()
    g = g[g.q_mean > 0]
    p_st = g.v_mean / g.q_mean
    p_nat = g.v_mean.sum() / g.q_mean.sum()
    Wm = v.groupby('state').agg(v_mean=('v_real', 'mean'), years=('v_real', 'size')).reset_index()
    # inferred (INFERRED['oncc_p1'])
    Wm['p_farmgate_fy2021'] = Wm.state.map(p_st).fillna(p_nat)
    Wm['q_mean'] = Wm.v_mean / Wm.p_farmgate_fy2021
    Wm['lu'] = 'Other non-cereal crops'
    Wm['p_src'] = np.where(Wm.state.isin(p_st.index), 'state grain-group farm-gate price', 'national grain-group farm-gate price')
    return trip(Wm), dict(p_state=p_st, p_nat=p_nat)


# ----------------------------------------------------------------------------- step 7: yields and P1
def crop_yield(M, sa2_long, s11, s21, cur, win):
    """Yield = K x SA2 index (x rho on irrigated land), K banded; P1 by state. float32."""
    lu, lm, area, state, lus, crops, x = M['lu'], M['lm'], M['area'], M['state'], M['lus'], M['crops'], M['x']
    lo, hi = REF['band']
    sa2 = sa2_long[['FY_end', 'sa2_code', 'item_desc', 'measure', 'value']]

    def luto_area_sa2(c, key):
        m = lu == lus.index(c)
        return pd.Series(area[m]).groupby(key[m]).sum()

    def sa2_index(c):
        out = {}
        for fy, key in [(2016, s11), (2021, s21)]:
            d = sa2[sa2.FY_end == fy]
            pp, ap = SA2_PAT[c]
            q = d[d.item_desc.str.contains(pp, regex=True) & (d.measure == 'production_t')].groupby('sa2_code').value.sum()
            a = (d[d.item_desc.str.contains(ap, regex=True) & (d.measure == 'area_ha')].groupby('sa2_code').value.sum() if ap
                 else luto_area_sa2(c, key))
            df = pd.DataFrame({'q': q, 'a': a}).dropna()
            df = df[(df.q > 0) & (df.a > 0)]
            if df.empty:
                continue
            df['st'] = pd.Series(df.index.astype(str).str[0].astype(int)).map(ST).values
            sty = df.groupby('st').apply(lambda g: g.q.sum() / g.a.sum())
            df['r'] = (df.q / df.a) / df.st.map(sty)
            out[fy] = df.r.clip(lo, hi)
        n = np.zeros(len(lu))
        acc = np.zeros(len(lu))
        for fy, key in [(2016, s11), (2021, s21)]:
            if fy not in out:
                continue
            v = pd.Series(key).map(out[fy]).values
            ok = ~np.isnan(v)
            acc[ok] += v[ok]
            n[ok] += 1
        return np.where(n > 0, acc / np.maximum(n, 1), 1.0), {fy: len(s) for fy, s in out.items()}

    def rho(c, s):    # irrigated / dry yield ratio of the 2010 file on the final map (inherited)
        d = cur[(cur.lu == c)].pivot_table(index='state', columns='lm', values='Yield')
        if {'dry', 'irr'} <= set(d.columns):
            if s in d.index and not np.isnan(d.loc[s].get('dry', np.nan)) and not np.isnan(d.loc[s].get('irr', np.nan)):
                return d.loc[s, 'irr'] / d.loc[s, 'dry'], 'state'
            nat = cur[cur.lu == c].groupby('lm').apply(
                lambda g: np.average(g.Yield.dropna(), weights=g.loc[g.Yield.notna(), 'ha_with_values']) if g.Yield.notna().any() else np.nan)
            if not np.isnan(nat.get('dry', np.nan)) and not np.isnan(nat.get('irr', np.nan)):
                return nat['irr'] / nat['dry'], 'national'
        return 1.0, 'none'

    Y, P, rep = {}, {}, []
    for c in crops:
        if c not in set(win.lu):
            rep.append(dict(lu=c, note='no production table (not built here)'))
            continue
        j = lus.index(c)
        r, npat = sa2_index(c) if c in SA2_PAT else (np.ones(len(lu)), {})
        elig = {m: x[m, j].astype(bool) for m in (0, 1)}
        alloc = {m: (lu == j) & (lm == m) for m in (0, 1)}
        # Cells without the crop on the base map take an index of at most the cap (the SA2 pattern is measured on the cropped
        # area); allocated cells keep theirs, so K and the placed tonnes are unchanged
        r_y = np.where(lu == j, r, np.minimum(r, REF['uncropped_index_cap']))
        Yc = {0: np.full(len(lu), np.nan), 1: np.full(len(lu), np.nan)}
        Ks = {}
        for s in STATES:
            ins = state == s
            w = win[(win.lu == c) & (win.state == s)]
            Q = float(w.q_mean.iloc[0]) if len(w) and not np.isnan(w.q_mean.iloc[0]) else 0.0
            rh, rsrc = rho(c, s)
            D = (r * area)[alloc[0] & ins].sum() + rh * (r * area)[alloc[1] & ins].sum()
            Ks[s] = (Q / D if D > 0 and Q > 0 else np.nan, rh, rsrc, Q, D)
        good = [(k[0], k[4]) for k in Ks.values() if not np.isnan(k[0])]
        Knat = sum(k * d for k, d in good) / sum(d for _, d in good) if good else np.nan
        for s, (K, rh, rsrc, Q, D) in Ks.items():
            ins = state == s
            Kuse = K if not np.isnan(K) else Knat
            Kband = float(np.clip(Kuse, Knat * lo, Knat * hi)) if not np.isnan(Kuse) and not np.isnan(Knat) else Kuse
            banded = (not np.isnan(Kuse)) and abs(Kband - Kuse) > 1e-12
            Kuse = Kband
            for m in (0, 1):
                cells = (elig[m] | alloc[m]) & ins
                Yc[m][cells] = Kuse * (rh if m == 1 else 1.0) * r_y[cells]
            placed = Kuse * D if not np.isnan(Kuse) else 0.0
            rep.append(dict(lu=c, state=s, Q_abs_t=Q, map_alloc_ha=area[(alloc[0] | alloc[1]) & ins].sum(), rho=rh, rho_src=rsrc,
                            K=K, K_national=Knat, K_used=Kuse, banded=banded, placed_t=placed, unplaced_t=Q - placed,
                            note=('national K: map area without production' if np.isnan(K) and D > 0 else
                                  'production without map area' if Q > 0 and D == 0 else ''), sa2_pattern_rows=str(npat)))
        Y[c] = {m: Yc[m].astype(np.float32) for m in (0, 1)}
        pw = win[(win.lu == c)].dropna(subset=['p_farmgate_fy2021'])
        pnat = (pw.v_mean.sum() / pw.q_mean.sum()) if len(pw) else np.nan
        pc = np.full(len(lu), np.nan)
        for s in STATES:
            v = pw[pw.state == s].p_farmgate_fy2021
            pc[state == s] = float(v.iloc[0]) if len(v) else pnat
        P[c] = pc.astype(np.float32)
    R = pd.DataFrame(rep)
    over = (R.placed_t - R.Q_abs_t > 0.5)
    lab = np.where(over & (R.Q_abs_t <= 0),
                   [LABEL['over_noabs'].format(NOABS_STATUS.get((c, s), 'zero or suppressed, unverified')) for c, s in zip(R.lu, R.state)],
                   np.where(over, LABEL['over_cap'], np.where(R.banded.fillna(False).astype(bool) & (R.unplaced_t > 0.5),
                                                               LABEL['banded_under'], '')))
    R['label'] = lab
    return Y, P, R


# ----------------------------------------------------------------------------- step 8: costs
def crop_costs(M, cur10, YP, fdp, abr_all, abs_all, cpi):
    """AC, QC, FLC, FOC, FDC per ha of cropping land: FDP group from ABARES FDP; ratio group on the 2010 cost-to-revenue
    ratio (INTERIM). float32."""
    lu, lm, area, state, lus = M['lu'], M['lm'], M['area'], M['state'], M['lus']
    f = fdp[(fdp.Industry == 'All Broadacre') & fdp.Year.between(min(REF['window']), max(REF['window']))].copy()
    f['st'] = f.State.map(FDP_ST)
    rows = []
    for _, r in f.iterrows():
        k = r['Total crop gross receipts ($)'] / r['Total cash receipts ($)']
        for c, items in FDP_CAT.items():
            v = sum(r.get(i, 0.0) if not pd.isna(r.get(i, np.nan)) else 0.0 for i in items) * cpi[r.Year]
            rows.append(dict(st=r.st, Year=r.Year, cat=c, crop_cost=v * k, area_cropped=r['Total area cropped (ha)'], crop_share=k))
    F = pd.DataFrame(rows)
    Fs = F.groupby(['st', 'cat']).apply(lambda g: g.crop_cost.sum() / g.area_cropped.sum()).unstack('cat')
    abr = abr_all[(abr_all.measure == 'area_ha') & (abr_all.state != 'AUS')]
    abr = abr[~abr.crop.isin(['Cottonseed a', 'Cotton lint a', 'Rice'])]
    sown = abr.groupby(['state', 'FY_end']).value.sum().groupby('state').mean()
    absl = abs_all[(abs_all.latest_release == 1)]
    hay_a = absl[absl.item_desc.str.contains(HAY_AREA, regex=True) & (absl.measure == 'area_ha') & absl.FY_end.between(2017, 2022)
                 ].groupby(['state', 'FY_end']).value.max().groupby('state').mean()
    sown = sown.add(hay_a, fill_value=0)
    inF = np.isin(lu, [lus.index(c) for c in FDP_GROUP])
    luto_a = pd.Series(area[inF]).groupby(state[inF]).sum()
    ss = (sown / luto_a).rename('sown_share')

    def cur_fields(m, c, mask):
        return {k: cur10[(k, LMN[m], c)].values[mask].astype(float) for k in ['Yield', 'P1', 'AC', 'QC', 'FLC', 'FOC', 'FDC']}
    rel = []
    for c in FDP_GROUP:    # relativities of the 2010 file (total non-water cost per ha), inherited
        j = lus.index(c)
        for m in (0, 1):
            if ('AC', LMN[m], c) not in cur10.columns:
                continue
            msk = (lu == j) & (lm == m)
            g = cur_fields(m, c, msk)
            tot = g['AC'] + g['FLC'] + g['FOC'] + g['FDC'] + g['QC'] * g['Yield']
            for s in np.unique(state[msk]):
                k = (state[msk] == s) & ~np.isnan(tot)
                rel.append(dict(lu=c, m=m, st=s, cost=np.average(tot[k], weights=area[msk][k]) if k.any() else np.nan,
                                ha=area[msk][state[msk] == s].sum()))
    R = pd.DataFrame(rel)
    natc = R.dropna().groupby(['lu', 'm']).apply(lambda g: np.average(g.cost, weights=g.ha))
    R['cost'] = R.apply(lambda r: r.cost if not np.isnan(r.cost) else natc.get((r.lu, r.m), np.nan), axis=1)
    R['cost'] = R.cost.fillna(R.cost.median())
    R['rel'] = R.groupby('st').apply(lambda g: g.cost / np.average(g.cost, weights=g.ha)).reset_index(level=0, drop=True)
    OUT = {}
    for c in FDP_GROUP + RATIO_GROUP:
        j = lus.index(c)
        for m in (0, 1):
            key = f'Y|{m}|{c}'
            if key not in YP:
                continue
            Y = YP[key].astype(float)
            P = YP[f'P|{c}'].astype(float)
            cells = ~np.isnan(Y)
            res = {k: np.full(len(lu), np.nan) for k in ['AC', 'QC', 'FLC', 'FOC', 'FDC']}
            if c in FDP_GROUP:
                for s in np.unique(state[cells]):
                    k = cells & (state == s)
                    fs = Fs.loc[s] if s in Fs.index else Fs.loc['AUS']
                    sh = ss.get(s, np.nan)
                    sh = sh if not np.isnan(sh) else float((sown.sum() / luto_a.sum()))
                    rr = R[(R.lu == c) & (R.m == m) & (R.st == s)].rel
                    rv = (float(rr.iloc[0]) if len(rr) else float(np.average(R[(R.lu == c) & (R.m == m)].rel, weights=R[(R.lu == c) & (R.m == m)].ha))
                          if len(R[(R.lu == c) & (R.m == m)]) else 1.0)
                    for cat in ['AC', 'FLC', 'FOC', 'FDC']:
                        res[cat][k] = fs[cat] * sh * rv
                    res['QC'][k] = 0.0
            else:   # INTERIM: 2010 non-water cost-to-revenue ratio and category shares x new revenue
                rr = []
                for mm in (m, 1 - m):
                    if ('AC', LMN[mm], c) not in cur10.columns:
                        continue
                    msk = (lu == j) & (lm == mm)
                    g = cur_fields(mm, c, msk)
                    rev = g['Yield'] * g['P1']
                    parts = {'AC': g['AC'], 'QC': g['QC'] * g['Yield'], 'FLC': g['FLC'], 'FOC': g['FOC'], 'FDC': g['FDC']}
                    ok = ~np.isnan(rev) & (rev > 0)
                    for s in list(np.unique(state[msk])) + ['AUS']:
                        k = ok & ((state[msk] == s) if s != 'AUS' else True)
                        if k.sum() == 0:
                            continue
                        w = area[msk][k]
                        R0 = np.average(rev[k], weights=w)
                        sh = {p: np.average(v[k], weights=w) / R0 for p, v in parts.items()}
                        rr.append(dict(mm=mm, st=s, ratio=sum(sh.values()), **{f's_{p}': v for p, v in sh.items()}))
                RR = pd.DataFrame(rr)
                for s in np.unique(state[cells]):
                    k = cells & (state == s)
                    pick = None
                    for cond in [(RR.mm == m) & (RR.st == s), (RR.mm == m) & (RR.st == 'AUS'), (RR.mm == 1 - m) & (RR.st == s),
                                 (RR.mm == 1 - m) & (RR.st == 'AUS')]:
                        if len(RR) and cond.any():
                            pick = RR[cond].iloc[0]
                            break
                    if pick is None:
                        continue
                    rev = Y[k] * P[k]
                    for p in ['AC', 'FLC', 'FOC', 'FDC']:
                        res[p][k] = pick[f's_{p}'] * rev
                    res['QC'][k] = pick['s_QC'] * P[k]
            for k2, v in res.items():
                OUT[f'{k2}|{m}|{c}'] = v.astype(np.float32)
    return OUT


# ----------------------------------------------------------------------------- step 9: water requirement
def crop_water(M, wl_all, s21, cur):
    """WR (ML per ha of irrigated cropping land) on allocated or eligible irrigated cells; band as yields."""
    lu, lm, area, state, lus, x = M['lu'], M['lm'], M['area'], M['state'], M['lus'], M['x']
    lo, hi = REF['band']
    a0, a1 = REF['water_window']
    wl = wl_all[wl_all.FY_end.between(a0, a1)].copy()

    def grp(cg):
        for g, (pat, _) in WATER_G.items():
            if pd.notna(cg) and pd.Series([cg]).str.contains(pat, regex=True).iloc[0]:
                return g
        return None
    wl['g'] = wl.crop_group.map(grp)
    stw = wl[(wl.region_type == 'state') & wl.g.notna() & (wl.measure == 'volume_applied_ml')].copy()
    stw['st'] = stw.region_name.map(WATER_STN).fillna(stw.region_name)
    V = stw.groupby(['g', 'st', 'FY_end']).value.sum().groupby(['g', 'st']).mean()
    sa = wl[(wl.region_type == 'SA2') & (wl.FY_end == 2021) & wl.g.notna() & wl.measure.isin(['volume_applied_ml', 'area_watered_ha'])]
    sa = sa.pivot_table(index=['g', 'region_code'], columns='measure', values='value', aggfunc='sum').dropna()
    sa = sa[(sa.volume_applied_ml > 0) & (sa.area_watered_ha > 0)].reset_index()
    sa['st'] = sa.region_code.astype(int).astype(str).str[0].astype(int).map(ST)
    stt = sa.groupby(['g', 'st']).apply(lambda q: q.volume_applied_ml.sum() / q.area_watered_ha.sum())
    sa['r'] = (sa.volume_applied_ml / sa.area_watered_ha) / [stt[(g, s)] for g, s in zip(sa.g, sa.st)]
    sa['r'] = sa.r.clip(lo, hi)
    cur = cur[cur.lm == 'irr'].dropna(subset=['WR'])
    WR, rep = {}, []
    for g, (pat, members) in WATER_G.items():
        rmap = sa[sa.g == g].set_index(sa[sa.g == g].region_code.astype(int)).r
        r = pd.Series(s21).map(rmap).fillna(1.0).values
        natV = V[g].sum() if g in V.index.get_level_values(0) else 0.0
        natA = sum(area[(lu == lus.index(c)) & (lm == 1)].sum() for c in members)
        natWR = natV / natA if natA > 0 else np.nan
        for c in members:
            WR[c] = np.full(len(lu), np.nan)
        for s in STATES:
            ins = state == s
            relv = {}
            for c in members:    # WR relativity of the 2010 file within the group (inherited)
                q = cur[(cur.lu == c) & (cur.state == s)].WR
                relv[c] = (float(q.iloc[0]) if len(q) else
                           (np.average(cur[cur.lu == c].WR, weights=cur[cur.lu == c].ha_with_values) if (cur.lu == c).any() else np.nan))
            base = np.nanmean(list(relv.values())) if any(not np.isnan(v) for v in relv.values()) else 1.0
            relv = {c: (v / base if not np.isnan(v) else 1.0) for c, v in relv.items()}
            D = sum((relv[c] * r * area)[(lu == lus.index(c)) & (lm == 1) & ins].sum() for c in members)
            Vgs = float(V.get((g, s), 0.0))
            Ags = sum(area[(lu == lus.index(c)) & (lm == 1) & ins].sum() for c in members)
            if D > 0 and Vgs > 0:
                k, note = Vgs / D, ''
            elif D > 0:
                k, note = natWR, 'no ABS volume: national WR (volume added)'
            else:
                k, note = natWR, 'no LUTO irrigated area' + (': ABS volume unplaced' if Vgs > 0 else '')
            kraw, banded = k, False
            if not np.isnan(k) and not np.isnan(natWR):
                k = float(np.clip(k, natWR * lo, natWR * hi))
                if abs(k - kraw) > 1e-9:
                    note = (note + '; ' if note else '') + f'banded (exact {kraw:.2f})'
                    banded = True
            for c in members:
                j = lus.index(c)
                cells = ins & (x[1, j].astype(bool) | ((lu == j) & (lm == 1)))
                WR[c][cells] = k * relv[c] * r[cells]
            placed = sum((WR[c] * area)[(lu == lus.index(c)) & (lm == 1) & ins].sum() for c in members)
            rep.append(dict(group=g, state=s, V_abs_ml=Vgs, luto_irr_ha=Ags, k=k, k_national=natWR, placed_ml=placed,
                            unplaced_ml=Vgs - placed, note=note, label=LABEL['water_banded'] if banded else ''))
    return {c: v.astype(np.float32) for c, v in WR.items()}, pd.DataFrame(rep)


# ----------------------------------------------------------------------------- step 10: energy and lifecycle fields
def energy_fields(M, orig, cur10, WRn, abr_all, abs_all):
    """The 8 energy and lifecycle fields (kg CO2e per ha of cropping land) on allocated or eligible cells (2010-basis
    levels x sown share; IRRIG = new WR x the 2010 IRRIG / WR ratio). float32."""
    lu, lm, area, lus, crops, x = M['lu'], M['lm'], M['area'], M['lus'], M['crops'], M['x']
    lu10, lm10 = M['lu10'], M['lm10']
    lo, hi = REF['sown_clip']
    # plain numpy state codes: the same values as the pandas string array, with fast masks and lookups
    state = np.asarray(M['state'], dtype='U3')
    ust, si = np.unique(state, return_inverse=True)
    ins = {s: state == s for s in ust}

    def lookup(d, fill):    # = pd.Series(state).map(d).fillna(fill).values
        v = np.array([d.get(s, np.nan) for s in ust], dtype=float)
        return np.where(np.isnan(v), fill, v)[si]
    abr = abr_all[(abr_all.measure == 'area_ha') & (abr_all.state != 'AUS')]
    absl = abs_all[(abs_all.latest_release == 1)]
    sown = {}
    for c, cs in SOWN_ABARES.items():
        sown[c] = abr[abr.crop.isin(cs)].groupby(['state', 'FY_end']).value.sum().groupby('state').mean()
    for c, pat in SOWN_ABS.items():
        sown[c] = absl[absl.item_desc.str.contains(pat, regex=True) & (absl.measure == 'area_ha') & absl.FY_end.between(2017, 2022)
                       ].groupby(['state', 'FY_end']).value.max().groupby('state').mean()
    SH = {}
    for c in crops:
        m = lu == lus.index(c)
        la = pd.Series(area[m]).groupby(state[m]).sum()
        SH[c] = ((sown[c] / la).clip(lo, hi) if c in sown else pd.Series(dtype=float))

    def state_mean(vals, mask_pref, mask_all):
        out = {}
        for s in ust:
            for mk in (mask_pref, mask_all):
                k = mk & ins[s] & ~np.isnan(vals)
                if k.any():
                    out[s] = np.average(vals[k], weights=area[k])
                    break
        k = mask_all & ~np.isnan(vals)
        return out, (np.average(vals[k], weights=area[k]) if k.any() else np.nan)
    OUT = {}
    for c in crops:
        j = lus.index(c)
        for m in (0, 1):
            src_c, src_m = (('Apples', 0) if (c == 'Pears' and m == 0) else (c, m))    # INFERRED['dry_pears']
            if ('CO2E_KG_HA_SOIL', LMN[src_m], src_c) not in orig.columns:
                continue
            cells = x[m, j].astype(bool) | ((lu == j) & (lm == m))
            pref = (lu10 == lus.index(src_c)) & (lm10 == src_m)
            allc = np.ones(len(lu), bool)
            wr_o = cur10[('WR', LMN[src_m], src_c)].values.astype(float)
            sh = lookup(SH[c].to_dict(), 1.0) if (c in SH and len(SH[c])) else np.ones(len(state))
            for f in F8:
                v = orig[('CO2E_KG_HA_' + f, LMN[src_m], src_c)].values.astype(float)
                res = np.full(len(lu), np.nan, np.float32)
                if f == 'IRRIG' and m == 1:
                    ratio = np.where((wr_o > 0) & ~np.isnan(v), v / np.where(wr_o > 0, wr_o, np.nan), np.nan)
                    rm, rn = state_mean(ratio, pref, allc)
                    rr = lookup(rm, rn)
                    res[cells] = WRn[c][cells] * rr[cells] if c in WRn else np.nan
                elif f == 'IRRIG':
                    res[cells] = 0.0
                else:
                    im, inat = state_mean(v, pref, allc)
                    iv = lookup(im, inat) * sh
                    res[cells] = iv[cells]
                res[cells] = np.nan_to_num(res[cells], nan=0.0)
                OUT[f'{f}|{m}|{c}'] = res
    return OUT, SH


# ----------------------------------------------------------------------------- step 11: assemble
def assemble_agec(M, cols10, YP, CO, WR, wdp):
    """agec_crops: Yield, P1, AC, QC, FLC, FOC, FDC, WR, WP on allocated or eligible cells; WP INTERIM."""
    lu, lm, state, lus, crops, x = M['lu'], M['lm'], M['state'], M['lus'], M['crops'], M['x']
    N = len(lu)
    wmed = pd.Series(wdp).groupby(state).median()
    wp_fill = np.where(np.isnan(wdp), pd.Series(state).map(wmed).values, wdp)
    cols = {}
    for c in crops:
        j = lus.index(c)
        for m in (0, 1):
            if f'Y|{m}|{c}' not in YP:
                continue
            cells = x[m, j].astype(bool) | ((lu == j) & (lm == m))
            if not cells.any() and (('Yield', LMN[m], c) not in cols10):
                continue

            def put(field, arr):
                a = np.full(N, np.nan)
                a[cells] = np.asarray(arr, float)[cells]
                cols[(field, LMN[m], c)] = a
            put('Yield', YP[f'Y|{m}|{c}'])
            put('P1', YP[f'P|{c}'])
            for k in ['AC', 'QC', 'FLC', 'FOC', 'FDC']:
                put(k, CO[f'{k}|{m}|{c}'])
            put('WR', WR[c] if (m == 1 and c in WR) else np.zeros(N))
            put('WP', wp_fill)
    A = pd.DataFrame(cols)
    A.columns = pd.MultiIndex.from_tuples(A.columns)
    A = A.sort_index(axis=1)
    A.index = pd.RangeIndex(N).astype(np.int32)
    A.index.name = 'CELL_ID'
    return A


def assemble_ghg(M, G, EN, area, s21):
    """agGHG_crops: the ghg_factors.py file with the 8 fields replaced on allocated or eligible cells; dry Pears SOIL from dry
    Apples, NaN cells filled with the SA2_2021, state, then national real-area-weighted mean."""
    lu, lm, state, lus, crops, x = M['lu'], M['lm'], M['state'], M['lus'], M['crops'], M['x']
    N = len(lu)
    filled = {}
    for c in crops:
        j = lus.index(c)
        for m in (0, 1):
            if f'SOWING|{m}|{c}' not in EN:
                continue
            cells = x[m, j].astype(bool) | ((lu == j) & (lm == m))
            for f in F8:
                key = ('CO2E_KG_HA_' + f, LMN[m], c)
                base = G[key].values.astype(np.float32).copy() if key in G.columns else np.full(N, np.nan, np.float32)
                base[cells] = EN[f'{f}|{m}|{c}'][cells]
                G[key] = base
            k2 = ('CO2E_KG_HA_SOIL', LMN[m], c)
            if k2 not in G.columns:
                v = G[('CO2E_KG_HA_SOIL', 'dry', 'Apples')].values.astype(np.float32).copy()
                ok = ~np.isnan(v)
                need = cells & ~ok
                filled[k2] = int(need.sum())
                for key in (s21, state):
                    mn = pd.Series(v[ok] * area[ok]).groupby(key[ok]).sum() / pd.Series(area[ok]).groupby(key[ok]).sum()
                    fv = pd.Series(key[need]).map(mn).values
                    v[np.where(need)[0][~np.isnan(fv)]] = fv[~np.isnan(fv)]
                    need = cells & np.isnan(v)
                v[need] = np.average(v[ok], weights=area[ok])
                G[k2] = v
    G = G.sort_index(axis=1).astype(np.float32)
    G.index = pd.RangeIndex(N).astype(np.int32)
    G.index.name = 'CELL_ID'
    return G, filled


# ----------------------------------------------------------------------------- provenance and outputs
def content_sha(a):
    a = np.ascontiguousarray(a)
    return sha(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes())


def pkg_versions():
    out = {}
    for p in ('numpy', 'pandas', 'tables', 'openpyxl', 'xlrd', 'rasterio', 'geopandas', 'pyogrio', 'shapely', 'scipy', 'xarray',
              'netCDF4'):
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


def interim_table(info):
    I = [
        ('WP (water delivery price)', 'WP, every (lm, crop) column', 'INTERIM',
         '2010-build water_delivery_price.h5 (price year and source unverified), state median on blank cells',
         'rebuild from the ACCC water monitoring reports', 'before the gate run'),
        ('ratio-group crop costs', 'AC, QC, FLC, FOC, FDC of ' + '; '.join(RATIO_GROUP), 'INTERIM',
         '2010-build agec_crops.h5 non-water cost-to-revenue ratio and cost shares by state and lm x new revenue',
         'hand-fetched state cost budgets where the ratio differs by more than 0.10', 'when the budgets arrive'),
        ('crop energy and lifecycle fields', 'agGHG_crops CO2E_KG_HA_' + ', '.join(F8) + ' (every crop)', 'INTERIM',
         '2010-build agGHG_crops.h5 intensities (state mean) x sown share; IRRIG = WR x 2010 IRRIG/WR ratio; report-only',
         'no public crop-level fuel or chemical series identified', 'not scheduled'),
        ('FDP-group cost split', 'AC, FLC, FOC, FDC of ' + '; '.join(FDP_GROUP), 'INHERITED',
         'split across (crop, lm) by the 2010-build agec_crops.h5 relative total cost per ha by state', 'gross-margin relativities '
         '(planned, not implemented)', 'not scheduled'),
        ('irrigated / dry yield ratio', 'Yield irr of every crop', 'INHERITED',
         'rho = 2010-build agec_crops.h5 irrigated / dry yield on the final map by state, else national', 'none identified',
         'not scheduled'),
        ('field peas and faba unit value', 'P1 and value of Winter legumes', 'inferred, not sourced',
         INFERRED['fieldpeas_faba_uv'] + f"; value {info['uv_fp']!r} FY2021 $/t from "
         + ', '.join(repr(float(v)) for v in info['uv_fp_candidates']), 'a published field pea and faba farm-gate value by state',
         'not scheduled'),
        ('summer legumes residual', 'Yield and P1 of Summer legumes', 'inferred, not sourced', INFERRED['summer_legumes'],
         'a published mungbean production and value by state', 'not scheduled'),
        ('other non-cereal crops price', 'P1 and Yield of Other non-cereal crops', 'inferred, not sourced',
         INFERRED['oncc_p1'] + '; ' + ', '.join(f'{s} {float(v)!r}' for s, v in info['oncc_p'].items()) + f"; national {info['oncc_nat']!r}",
         'none identified', 'not scheduled'),
        ('dry Pears', 'agec_crops (*, dry, Pears); agGHG_crops (*, dry, Pears)', 'inferred, not sourced', INFERRED['dry_pears'],
         'none identified', 'not scheduled'),
    ]
    return pd.DataFrame(I, columns=['item', 'columns_or_crops', 'status', 'basis', 'replaced_by', 'when'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', required=True, help='folder of the inherited 2010-build files listed in INHERITED (input/Nick_build/inherited_2010)')
    ap.add_argument('--nlum-map', required=True, help='output folder of nlum_map.py (sa2_2021_cells.npy)')
    ap.add_argument('--map-dir', required=True, help='folder holding the final 2021-base lumap.h5 and lmmap.h5 (livestock_map.py)')
    ap.add_argument('--x-mrj', required=True, help='x_mrj.npy from eligibility.py')
    ap.add_argument('--ghg-crops', required=True, help='agGHG_crops.h5 from ghg_factors.py')
    ap.add_argument('--out', required=True, help='output folder (e.g. input/Nick_build/crop_economics)')
    ap.add_argument('--cache', required=True, help='shared raw-download cache (content-addressed by sha256)')
    ap.add_argument('--offline', action='store_true', help='fail instead of downloading on a cache miss')
    ap.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                    help='a local copy of a pinned source (must match its pin); NAME in ' + ', '.join(SOURCES))
    a = ap.parse_args()
    t0 = time.time()
    overrides = dict(s.split('=', 1) for s in a.source)
    assert set(overrides) <= set(SOURCES), set(overrides) - set(SOURCES)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    zips = {k: fetch(k, a.cache, a.offline, overrides) for k in SOURCES}
    P = {ABS_LABEL.get(k, k): v for k, v in zips.items()}
    print('sources ready', f'{time.time() - t0:.0f}s', flush=True)
    inp = {f: os.path.join(a.input, f) for f in INHERITED}
    upstream = {'sa2_2021_cells.npy': os.path.join(a.nlum_map, 'sa2_2021_cells.npy'), 'lumap.h5': os.path.join(a.map_dir, 'lumap.h5'),
                'lmmap.h5': os.path.join(a.map_dir, 'lmmap.h5'), 'x_mrj.npy': a.x_mrj, 'agGHG_crops.h5': a.ghg_crops}
    in_sha = {('inherited', f): sha_file(p) for f, p in inp.items()}
    in_sha.update({('upstream', f): sha_file(p) for f, p in upstream.items()})

    # ---- public tables
    cpi = cpi_table(P['CPI-Quarterly_aug2026.zip'])
    abs_st, abs_sa2 = abs_tables(P)
    abr = acr_table(P['AustCropRrt20260901_v1.0.0_asset3.xlsx'])
    fdp = fdp_wide(P['fdp-state-historical.csv'], P['fdp-national-historical.csv'])
    wl = water_long(P)
    s11 = sa2_2011_cells(P['1270055001_sa2_2011_aust_shape.zip'], inp['NLUM_2010-11_mask.tif'])
    print('tables done', f'{time.time() - t0:.0f}s', flush=True)

    # ---- map, eligibility and the 2010 files
    lus = pd.read_csv(inp['ag_landuses.csv'], header=None)[0].tolist()
    sid = np.load(inp['state_id.npy'])
    M = dict(lu=pd.read_hdf(upstream['lumap.h5']).values, lm=pd.read_hdf(upstream['lmmap.h5']).values,
             area=pd.read_hdf(inp['real_area.h5']).values.astype(float), state_id=sid, state=pd.Series(sid).map(ST).values,
             lus=lus, crops=[c for c in lus if not any(k in c for k in ('Beef', 'Sheep', 'Dairy', 'Unallocated'))],
             x=np.ascontiguousarray(np.load(upstream['x_mrj.npy']).transpose(0, 2, 1)),     # (lm, lu, cell)
             lu10=pd.read_hdf(inp['lumap.h5']).values, lm10=pd.read_hdf(inp['lmmap.h5']).values)
    s21 = np.load(upstream['sa2_2021_cells.npy'])
    assert len(M['lu']) == len(M['area']) == len(s21) == len(s11) == M['x'].shape[2] == NCELL
    cur10 = pd.read_hdf(inp['agec_crops.h5'])

    # ---- economics
    cur = current_econ(M, cur10)
    bwin, binfo = broadacre_window(abs_st, abr, cpi)
    hwin = hort_window(abs_st, cpi)
    owin, oinfo = oncc_window(abs_st, cpi, bwin)
    win = pd.concat([bwin, hwin, owin])
    Y, P1, ycal = crop_yield(M, abs_sa2, s11, s21, cur, win)
    YP = {f'Y|{m}|{c}': Y[c][m] for c in Y for m in (0, 1)}
    YP.update({f'P|{c}': P1[c] for c in P1})
    CO = crop_costs(M, cur10, YP, fdp, abr, abs_st, cpi)
    WR, wcal = crop_water(M, wl, s21, cur)
    print('economics done', f'{time.time() - t0:.0f}s', flush=True)
    orig = pd.read_hdf(inp['agGHG_crops.h5'])
    EN, SH = energy_fields(M, orig, cur10, WR, abr, abs_st)
    del orig
    A = assemble_agec(M, set(cur10.columns), YP, CO, WR, pd.read_hdf(inp['water_delivery_price.h5']).values.astype(float))
    del cur10
    G, filled = assemble_ghg(M, pd.read_hdf(upstream['agGHG_crops.h5']), EN, M['area'], s21)
    print('assembled', f'{time.time() - t0:.0f}s', flush=True)

    # ---- write (layouts of the 2010-base files)
    outs = {'agec_crops.h5': ('agec_crops', A), 'agGHG_crops.h5': ('agGHG_crops', G)}
    ref = {'agec_crops.h5': pd.read_hdf(inp['agec_crops.h5'], stop=1), 'agGHG_crops.h5': pd.read_hdf(upstream['agGHG_crops.h5'], stop=1)}
    for f, (key, df) in outs.items():
        r = ref[f]
        assert set(r.columns) <= set(df.columns) and len(df) == NCELL, f
        assert set(df.dtypes) == set(r.dtypes) and df.index.name == r.index.name and df.index.dtype == r.index.dtype, f
        df.to_hdf(os.path.join(a.out, f), key=key, mode='w', format='table', complevel=9, complib='zlib', index=False)
    ncs = [NC.write_like(a.out, 'agec_crops', A, ['field', 'lm', 'lu']),                 # NetCDF copies, jinzhu layout
           NC.write_like(a.out, 'agGHG_crops', G, ['field', 'lm', 'lu'])]
    del A, G
    pd.concat([bwin.assign(table='broadacre'), hwin.assign(table='horticulture'), owin.assign(table='other non-cereal crops')]
              ).to_csv(os.path.join(a.out, 'crop_production_window.csv'), index=False, lineterminator='\n')
    ycal.to_csv(os.path.join(a.out, 'crop_yield_calibration.csv'), index=False, lineterminator='\n')
    wcal.to_csv(os.path.join(a.out, 'crop_water_calibration.csv'), index=False, lineterminator='\n')
    IT = interim_table(dict(uv_fp=binfo['uv_fp'], uv_fp_candidates=binfo['uv_fp_candidates'], oncc_p=oinfo['p_state'].to_dict(),
                            oncc_nat=float(oinfo['p_nat'])))
    IT.to_csv(os.path.join(a.out, 'interim_items_crop_economics.csv'), index=False, lineterminator='\n')
    print('written', f'{time.time() - t0:.0f}s', flush=True)

    # ---- provenance (last)
    me = open(os.path.abspath(__file__), 'rb').read()
    PV = [['source', k, url, pin, 'sha256 of raw file', note] for k, (url, pin, note) in SOURCES.items()]
    PV += [['inherited', f, '', in_sha[('inherited', f)], 'inherited 2010-build input (inherited_2010)', ''] for f in INHERITED]
    PV += [['upstream', f, '', in_sha[('upstream', f)], 'input from another 2021-base tool', upstream[f].replace('\\', '/')]
           for f in upstream]
    PV += [['interim', r.item, '', '', r.status, f'{r.columns_or_crops} | basis: {r.basis} | replaced by: {r.replaced_by} | when: {r.when}']
           for r in IT.itertuples()]
    PV += [['fill', f'{k[0]} {k[1]} {k[2]}', '', '', f'{n} cells filled (SA2_2021, state, national mean of dry Apples)', 'dry Pears fill']
           for k, n in filled.items()]
    PV += [['script', 'input/Nick_build/scripts/crop_economics.py', '', sha(me), 'sha256 of script', '']]
    PV += [['package', k, '', v, 'version', ''] for k, v in sorted(pkg_versions().items())]
    PV += [['python', sys.version.split()[0], '', '', 'interpreter', '']]
    for f, (key, _) in outs.items():
        p = os.path.join(a.out, f)
        back = pd.read_hdf(p)
        PV.append(['output', f, '', sha_file(p), 'sha256 of file (HDF5 bytes vary between runs)', ''])
        PV.append(['dataset', f'{f}:/{key} values', '', content_sha(back.to_numpy()), 'sha256 of dtype, shape and array bytes',
                   f'{back.shape[1]} columns {back.to_numpy().dtype} {back.shape}'])
        PV.append(['dataset', f'{f}:/{key} index', '', content_sha(back.index.to_numpy()), 'sha256 of dtype, shape and array bytes',
                   'cell order'])
        del back
    for f in ('crop_production_window.csv', 'crop_yield_calibration.csv', 'crop_water_calibration.csv',
              'interim_items_crop_economics.csv'):
        PV.append(['output', f, '', sha_file(os.path.join(a.out, f)), 'sha256', ''])
    PV += [['netcdf', os.path.basename(p), '', h, 'sha256 of NetCDF content (nc_layout.content_sha)', 'checked equal to the .h5 arrays'] for p, h in ncs]
    write_csv(os.path.join(a.out, 'provenance.csv'), ['kind', 'item', 'url', 'sha256', 'note', 'source'], PV)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
